# agent_evolve

Plug-and-play, LLM-driven **multi-objective optimization** via Pareto-guided
evolution. A tiny problem-agnostic core and a harness-agnostic evolutionary loop,
with a typed **pydantic-ai** integration behind a single `optimize()` API.

The loop generates candidates, evaluates them against your black-box objective,
learns textual *constraint* and *performance* memory from failures and the Pareto
front, and breeds offspring from the front. Switching the harness changes **zero**
lines of the loop.

## Install

```bash
# core + the in-tree pydantic-ai harness
pip install -e ".[pydantic_ai]"
```

Provider keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, ...) and the
optional `AGENTEVOLVE_MODEL` / `AGENTEVOLVE_HARNESS` are read from the process
environment. To keep them in a file instead, name it explicitly:

```bash
export AGENTEVOLVE_DOTENV=.env          # or: AgentEvolveSettings.from_env(dotenv_path=".env")
```

The library never *searches* for a `.env`. It will not walk up the directory
tree, so it cannot pick up credentials belonging to an enclosing repository.
Values already in the environment always outrank the file.

### Proving a run made no provider call

`env -u OPENROUTER_API_KEY ...` alone does **not** hold: any later
`load_dotenv(path, override=False)` puts the key straight back, because
`override=False` defers only to variables that are *present*, and a removed one
is absent. A process cannot tell "never set" from "deliberately unset", so say
which it is:

```bash
AGENTEVOLVE_SCRUBBED=OPENROUTER_API_KEY python run_campaign.py
```

Named variables are removed from the environment and can never be reintroduced
from a file, whatever a caller asks for. Every `.env` read goes through
`agent_evolve.settings.load_credentials`, which returns a `CredentialLoad`
record naming exactly what it introduced, refused, and removed — seal that
record in the run's receipt rather than asserting a constant. A run that needs
no credentials at all can say so with `allow_credentials=()`.

## Quickstart

```python
from agent_evolve import optimize, ObjectiveSpec, ValidationOutcome

class Knapsack:
    candidate_model = None  # optional pydantic schema for the LLM

    @property
    def objectives(self):
        return [ObjectiveSpec("total_value", "max"), ObjectiveSpec("total_weight", "min")]

    def validate_detailed(self, config) -> ValidationOutcome:
        if not config.get("selection"):
            return ValidationOutcome(False, "structural", "selection must be non-empty")
        return ValidationOutcome(True)

    def evaluate(self, config) -> dict:
        ...  # return {"total_value": ..., "total_weight": ...}

    def search_space_description(self) -> str:
        return "Pick a subset of item indices ..."

result = optimize(Knapsack(), harness="pydantic_ai", model="openai:gpt-4o",
                  pop_size=8, generations=5, seed=0)
print(result.best.configuration, result.best.objectives)
print("Pareto:", len(result.pareto_front), "evaluations:", result.evaluations)
```

The loop depends on the `Harness` port rather than pydantic-ai directly. Custom
integrations can therefore be registered without changing optimization logic.

## Implementing your own `Problem`

Implement the `Problem` protocol (`agent_evolve.core.problem`):

- **Required:** `objectives` (a `Sequence[ObjectiveSpec]`) and
  `evaluate(config) -> dict[str, float]`. Raise `ValueError(msg)` from `evaluate`
  for infeasible configs — the message becomes LLM feedback.
- **Optional:** `validate_detailed(config) -> ValidationOutcome` (a fast pre-check
  with a `failure_phase` label), `search_space_description() -> str`,
  `render_candidate(config) -> str` (compact view used in failure/Pareto lists),
  `candidate_key(config) -> str` (de-dup key), and the attributes
  `candidate_model` (pydantic schema), `constraints_description`, `example_config`,
  and `directives` (see below).

That is the entire integration surface; everything else is provided. See
[`docs/integration_guide.md`](docs/integration_guide.md) for a worked walkthrough.

### Owning your prompts: the `Directives` port

Prompt wording is supplied by a `Directives` provider rather than hard-coded in
the adapter. The backbone ships `DefaultDirectives`, used automatically when a
problem does not set one. A problem owns its prompts by exposing a `directives`
attribute:

```python
from agent_evolve import DefaultDirectives

class MyDirectives(DefaultDirectives):
    def compose_initial(self, context: str, n_candidates: int) -> str:
        return f"{context}\nPropose exactly {n_candidates} valid candidates."

class MyProblem:
    directives = MyDirectives()
    ...
```

The pydantic-ai harness reads `ctx.directives`. Problem-specific *content*
(architecture, constraints, examples) should flow through
`search_space_description()` / `constraints_description`, keeping directives
reusable.

## Architecture (ports & adapters)

```
core/      problem + ObjectiveSpec + ValidationOutcome; Pareto + minimax; formatting; stats  (no LLM, no I/O)
session/   run_evolution_loop + evaluate_batch          (depends only on core + the Harness port)
harness/   Harness protocol + registry + Directives port + DefaultDirectives
integrations/pydantic_ai                                 (adapter; never imported by core)
api.py     optimize()  — the composition root
```

Design rules: dependencies point inward only; orchestration/state/seeding/logging
live in Python (`session/`), adapters do exactly one thing (turn a composed
instruction into a parsed result), and prompt text is supplied through the
`Directives` port (`harness/directives.py`) — generic by default, owned by the
problem when it wants to. The pydantic-ai adapter requests typed structured
outputs: candidate operations return `list[CandidateConfig]`, while insight
operations return `list[str]` or `str`, without a JSON-string round trip.

## Tests

```bash
python -m pytest tests/
```

The suite covers Pareto/minimax, evaluation routing, event-store contracts, the
full loop with a deterministic fake harness, and offline equivalence between the
pydantic-ai typed-output path and the fake harness.

# Integration guide: adding a new optimization problem

`agent_evolve` is a problem-agnostic backbone. Integrating a new problem means
implementing one protocol (`Problem`), optionally supplying prompt `Directives`,
and calling `optimize()`. Nothing in `core/`, `session/`, or the harness adapters
needs to change. The knapsack example under
[`examples/`](../examples) is the worked reference.

## 1. Implement the `Problem` protocol

`agent_evolve.core.problem.Problem` is a structural (duck-typed) protocol. You do
not subclass anything — just provide the members below.

### Required

| Member | Signature | Purpose |
| --- | --- | --- |
| `objectives` | `Sequence[ObjectiveSpec]` (property/attr) | Names + `"min"`/`"max"` direction for each objective. |
| `evaluate` | `evaluate(config) -> dict[str, float]` | Black-box scoring. Return one float per objective name. Raise `ValueError(msg)` for an infeasible config — `msg` becomes LLM feedback. |

### Optional (sensible defaults provided)

| Member | Signature | Default behavior |
| --- | --- | --- |
| `validate_detailed` | `(config) -> ValidationOutcome` | Fast pre-check before `evaluate`, with a `failure_phase` label ("structural"/"constraint"/...). Defaults to "always ok". |
| `search_space_description` | `() -> str` | Free-text description injected into the prompt context. |
| `render_candidate` | `(config) -> str` | Compact one-line view used inside failure / Pareto lists. Defaults to JSON. |
| `candidate_key` | `(config) -> str` | De-duplication key so identical proposals are not re-evaluated. Defaults to sorted JSON. |
| `candidate_model` | attribute: a `pydantic.BaseModel` subclass | Typed schema the LLM fills (`list[CandidateConfig]`). Without it, a permissive dict schema is used. |
| `constraints_description` | attribute: `str` | Extra constraint text appended to the context. |
| `directives` | attribute: a `Directives` | Prompt wording (see below). Defaults to `DefaultDirectives`. |

`config` is whatever the LLM produces — a `dict` (from `candidate_model.model_dump()`
or the permissive schema). Treat it as plain data.

## 2. (Optional) Own your prompts via `Directives`

Prompt wording is not baked into the adapter; it comes from a `Directives`
provider (`agent_evolve.harness.directives`). The backbone ships
`DefaultDirectives`, used automatically when a problem sets no `directives`.

The cleanest way to own prompts is to subclass `DefaultDirectives` and override
only what differs:

```python
from agent_evolve import DefaultDirectives

class MyDirectives(DefaultDirectives):
    def compose_initial(self, context: str, n_candidates: int) -> str:
        return f"{context}\nPropose exactly {n_candidates} valid candidates."
```

Then point your problem at it:

```python
class MyProblem:
    directives = MyDirectives()
    ...
```

Keep *content* (architecture facts, constraints, examples) in
`search_space_description()` / `constraints_description` so the directives stay
reusable; the `Directives` are about *how to ask*, not *what the problem is*.

### The `Directives` contract

```python
compose_initial(context, n_candidates) -> str
compose_regenerate(context, failed_str, n_candidates, constraint_instruction, performance_insights) -> str
compose_offspring(context, pareto_str, n_candidates, constraint_instruction, performance_insights) -> str
compose_regenerate_offspring(context, failed_str, pareto_str, n_candidates, constraint_instruction, performance_insights) -> str
compose_failure_insights(context, failed_str, n_failed) -> str
compose_constraint_instruction(context, failed_str, previous="") -> str
compose_performance_insights(context, stats_str, pareto_str, previous="") -> str
```

The pydantic-ai integration uses the problem's `candidate_model` as its
structured-output type. Candidate operations return `list[CandidateConfig]`;
insight operations return `list[str]` or `str`.

## 3. Run it

```python
from agent_evolve import optimize

result = optimize(
    MyProblem(),
    harness="pydantic_ai",
    model="openai:gpt-4o",
    pop_size=8, generations=5, seed=0,
)
print(result.best.configuration, result.best.objectives)
print("Pareto:", len(result.pareto_front), "evaluations:", result.evaluations)
```

`optimize()` resolves `directives = problem.directives or DefaultDirectives()`,
binds the chosen harness, and runs the harness-agnostic loop (generation,
evaluation, de-dup, constraint/performance learning, Pareto-front breeding). The
harness is the only swappable part; switching it changes zero loop lines.

## Existing integration: MEDEA

`medea_agentevolve` is a full integration: it implements `MedeaMappingProblem`
(objectives = energy/cycles/area, a 6-level Simba `MappingRecommendation` schema,
Timeloop/Accelergy evaluation) and its own prompt directives. It depends on
`agent_evolve` purely as a library—a template for any new problem.

# agent_evolve

Multi-objective optimization driven by a language model, with the chance
baseline built in.

You describe your problem in five methods. It proposes candidates, evaluates
them, keeps a Pareto front, and breeds from it. The model reads your
constraints and your failures, which is the part a blind sampler cannot do.

It also ships the tool that tells you whether any of that is helping on *your*
problem, because sometimes it is not, and you should find that out in thirty
seconds rather than after a month.

## Install

```bash
pip install agent_evolve            # core; the random proposer works immediately
pip install 'agent_evolve[llm]'     # adds the model-driven proposer
```

## Try it without spending anything

No credentials, no network, no provider account:

```bash
python examples/build_flags/run.py
```

```
evaluations  40 (+2 served from cache)
pareto front 7

 runtime_ms   binary_kb  configuration
     658.82      563.70  Ofast avx2 lto=full unroll=0 fast=1 inline=581
     673.99      461.39  O3 avx512 lto=full unroll=0 fast=1 inline=39
     760.97      387.86  O2 avx2 lto=thin unroll=0 fast=1 inline=117
    1059.19      233.60  Os none lto=thin unroll=0 fast=0 inline=75
```

That run used the uninformed sampler. It is a real run of the real loop, and it
is also the baseline any model has to beat.

## Is the model worth paying for on your problem?

```bash
agent_evolve check yourpkg.problem:problem --budget 40
```

Runs an uninformed sampler and a model against the same problem, the same
budget and the same evaluator, then says which won and how confidently.

Run it before you spend anything. On one benchmark we studied, a single median
random draw already accounted for roughly 80% of what a full optimization run
produced -- no optimizer could be told from any other on it. Your problem might
be one of those. This command finds out.

Pass `--baseline-only` to run just the free arm.

## Describing your problem: five obligations

That is the whole integration surface.

```python
from pydantic import BaseModel, Field
from agent_evolve import ObjectiveSpec, ValidationOutcome, optimize

class Config(BaseModel):                       # 1. what a candidate looks like
    workers: int = Field(..., ge=1, le=64)
    strategy: str

class MyProblem:
    candidate_model = Config

    objectives = [                             #    what you are optimizing
        ObjectiveSpec("throughput", "max"),
        ObjectiveSpec("cost", "min"),
    ]

    def seeds(self):                           # 2. where to start
        return [{"workers": 8, "strategy": "balanced"}]

    def validate(self, config):                # 3. cheap rejection, explained
        if config["workers"] > 32 and config["strategy"] == "balanced":
            return ValidationOutcome(
                False, "constraint",
                "balanced strategy supports at most 32 workers; "
                "reduce workers or use 'sharded'",
            )
        return ValidationOutcome(True)

    def materialize(self, config):             # 4. candidate -> what you measure
        return (config["workers"], config["strategy"])

    def evaluate(self, artifact):              # 5. measure it
        workers, strategy = artifact
        return {"throughput": ..., "cost": ...}

result = optimize(MyProblem(), budget=40)
print(result.best.configuration, result.best.objectives)
```

`budget` counts artifacts measured. It is a hard cap: say 40 and you are billed
for 40.

**Why `materialize` is separate from `evaluate`.** Two configurations often
produce the same artifact -- the same build, the same mapping, the same
deployment. Materializing first means the second one is free instead of being
paid for twice. Put anything cheap and deterministic there, and keep `evaluate`
for the expensive part. Declare your bounds in the schema too: anything reading
it, including the baseline, then draws only legal candidates.

Problems written against the older two-method contract (`objectives` and
`evaluate`) keep working unchanged.

## Choosing a proposer

```python
optimize(problem, budget=40, proposer="random")   # free, no credentials, the baseline
optimize(problem, budget=40, proposer="llm")      # a model
optimize(problem, budget=40)                      # auto: llm if a key exists, else random
```

`auto` says out loud which one it picked. Set a provider key
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) in your environment.

## Configuration

Three environment variables affect behaviour, and none is required:

| Variable | Meaning | Default |
| --- | --- | --- |
| `AGENTEVOLVE_MODEL` | model id | `openrouter:openai/gpt-5.6-luna` |
| `AGENTEVOLVE_HARNESS` | adapter for the `llm` proposer | `pydantic_ai` |
| `AGENTEVOLVE_TEMPERATURE` | sampling temperature | provider default |

Two more control credentials, and exist because of a real defect:

| Variable | Meaning |
| --- | --- |
| `AGENTEVOLVE_DOTENV` | path to a `.env` to load. Nothing is loaded unless you name it. |
| `AGENTEVOLVE_SCRUBBED` | names to remove and never reintroduce from a file. |

This library never searches upward for a `.env`. An earlier version called
`dotenv.load_dotenv()` with no argument, which walks up the directory tree until
it finds any `.env` -- so a run inside a monorepo silently adopted an unrelated
project's credentials, and a run launched specifically to prove it made no
provider call was handed the key back by the very next line.

## Where this helps, and where it does not

**Read [`docs/scope.md`](docs/scope.md) before adopting it.** The summary:

- Strongest on **categorical, constrained** problems with **expensive
  evaluators** and **small budgets** -- up to **5.16x** fewer evaluations than
  the best classical arm, where uniform random search never finished at all.
- Marginal on **smooth continuous** problems: **1.28x**, and random matches it
  31% of the time.
- **Slower in wall-clock** unless each evaluation costs roughly **290 seconds**
  or more. In direct comparison NSGA-II ran **9.1x faster**.
- On some problems it **loses** to random search, and on some benchmarks no
  optimizer can be distinguished from any other at all.

## Examples

| Example | Credentials | What it shows |
| --- | --- | --- |
| `examples/build_flags/` | none | The regime this tool suits: categorical, constrained, expensive evaluator |
| `examples/knapsack/` | none | The five obligations, shortest possible; `--compare` runs both arms |

The default model is cheap on purpose -- $0.10 per million input tokens and
$0.60 per million output -- and `agent_evolve check` prints the model and its
price before it spends anything, so a default nobody chose cannot bill anyone.

## Tests

```bash
pip install 'agent_evolve[dev]'
python -m pytest tests/              # the shipped package, offline, ~90 seconds
python -m pytest tests/ -m research  # the research suite; needs the corpus
```

The default run is what someone who cloned this repository can actually use.
Tests driving the research campaign scripts are marked `research` and
deselected; they are also skipped rather than failed when the corpus is
absent.

## License

MIT. See [LICENSE](LICENSE).

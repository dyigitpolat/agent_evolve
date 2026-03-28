# agent_evolve

LLM-driven multi-objective optimisation via Pareto-guided evolution.

An LLM acts as an intelligent search operator: it generates candidate
configurations, learns from failures via consolidated constraint instructions,
analyses performance patterns, and produces offspring inspired by the Pareto
front — all through structured prompts with typed outputs.

The core algorithm is a [Kedi](./kedi/) program (`evolve.kedi`).  Python
modules supply the problem protocol, result types, and Pareto utilities.  The
same `.kedi` program runs from the Python API **or** the kedi CLI.

## Installation

`kedi` is included as a Git submodule at `./kedi`.  Clone with submodules, or
after cloning run `git submodule update --init --recursive`.

```bash
# From the agent_evolve directory
pip install -e .

# With DSPy adapter support
pip install -e ".[dspy]"
```

The editable install pulls `kedi` from the submodule (`file:./kedi` in
`pyproject.toml`).

## Quick Start — Python API

```python
from agent_evolve import AgentEvolver, ObjectiveSpec

class MyProblem:
    @property
    def objectives(self):
        return [
            ObjectiveSpec("cost", "min"),
            ObjectiveSpec("quality", "max"),
        ]

    def evaluate(self, config):
        x, y = config["x"], config["y"]
        return {"cost": x + y, "quality": (x * y) ** 0.5}

    def validate(self, config):
        if config.get("x", -1) < 0 or config.get("y", -1) < 0:
            raise ValueError("x and y must be non-negative")
        return True

    def search_space_description(self):
        return "Config: {'x': float, 'y': float}. Both non-negative, range 0–100."

evolver = AgentEvolver(model="openai:gpt-4o", pop_size=8, generations=3)
result = evolver.optimize(MyProblem())

print(result.best.configuration)
print(result.best.objectives)
```

## Quick Start — CLI

Create a `problem_def.py` in your working directory that exposes two names:

```python
# problem_def.py
from agent_evolve import ObjectiveSpec

class MyProblem:
    # ... same as above ...
    pass

problem = MyProblem()

config = {
    "pop_size": 8,
    "generations": 3,
    "candidates_per_batch": 5,
    "max_regen_rounds": 10,
    "verbose": True,
}
```

Then run:

```bash
kedi path/to/agent_evolve/src/agent_evolve/evolve.kedi --adapter-model openai:gpt-4o
```

Both paths execute the identical `evolve.kedi` program.

## Defining a Problem

A problem is any Python object with two required attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `objectives` | `Sequence[ObjectiveSpec]` | At least one objective to optimise. |
| `evaluate(config)` | `-> Dict[str, float]` | Return objective values for a configuration. |

Optional (detected via `hasattr`):

| Method | Type | Description |
|--------|------|-------------|
| `validate(config)` | `-> bool` | Fast feasibility pre-check.  Raise `ValueError` for feedback. |
| `search_space_description()` | `-> str` | Free-text description of config format, ranges, and constraints. |

**Feedback:** Raise `ValueError("descriptive message")` from `validate` or
`evaluate` to provide textual feedback to the LLM.  These messages are
included in failure-analysis prompts so the agent can learn from mistakes.

## Configuration

`AgentEvolver` accepts these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"openai:gpt-4o"` | LLM model string (pydantic-ai format). |
| `adapter_type` | `"pydantic"` | `"pydantic"` or `"dspy"`. |
| `pop_size` | 8 | Valid candidates per generation. |
| `generations` | 5 | Number of evolutionary generations. |
| `candidates_per_batch` | 5 | Candidates requested per LLM call. |
| `max_regen_rounds` | 10 | Max regeneration attempts when too few valid. |
| `config_schema` | `None` | Optional dict describing config structure. |
| `example_config` | `None` | Optional example configuration. |
| `constraints_description` | `None` | Optional constraint text for LLM. |

For CLI usage these go in the `config` dict inside `problem_def.py`.

## Algorithm Overview

```
Generation 1 — Initial Sampling
  LLM generates diverse candidates
  Evaluate → valid / failed
  Failed → LLM failure insights → constraint instructions → regenerate
  Repeat until pop_size valid
  Compute Pareto front
  LLM generates performance insights

Generation 2..N — Evolution
  LLM generates offspring from Pareto front
  Evaluate → valid / failed
  Failed → regenerate with Pareto reference + failure insights
  Update Pareto front
  LLM updates performance insights

Return SearchResult (best, pareto_front, all_candidates, history)
```

The full loop lives in `evolve.kedi`.  LLM interactions are Kedi DSL
procedures; evaluation, Pareto computation, and statistics are Python
utilities imported from `agent_evolve._support` and `agent_evolve.results`.

## API Reference

### `agent_evolve.ObjectiveSpec`

```python
@dataclass(frozen=True)
class ObjectiveSpec:
    name: str                    # e.g. "cost", "accuracy"
    goal: Literal["min", "max"]  # optimisation direction
```

### `agent_evolve.Candidate`

```python
@dataclass(frozen=True)
class Candidate(Generic[ConfigT]):
    configuration: ConfigT
    objectives: Dict[str, float]
    metadata: Dict[str, Any]
```

### `agent_evolve.SearchResult`

```python
@dataclass(frozen=True)
class SearchResult(Generic[ConfigT]):
    objectives: Sequence[ObjectiveSpec]
    best: Candidate[ConfigT]
    pareto_front: List[Candidate[ConfigT]]
    all_candidates: List[Candidate[ConfigT]]
    history: List[Dict[str, Any]]
```

### Selection Utilities

- `compute_pareto_front(candidates, objectives)` — non-dominated subset
- `select_best_candidate(pareto, objectives)` — lexicographic pick
- `select_minimax_rank(candidates, objectives)` — balanced pick (smallest worst-rank)

## Project Structure

```
agent_evolve/
├── kedi/                 # Git submodule (kedi-lang/kedi)
├── src/agent_evolve/
│   ├── __init__.py       # Public API
│   ├── problem.py        # Problem protocol + ObjectiveSpec
│   ├── results.py        # Candidate, SearchResult, Pareto utilities
│   ├── optimizer.py      # AgentEvolver (thin wrapper)
│   ├── _support.py       # Internal: evaluation, formatting, stats
│   └── evolve.kedi       # The algorithm
├── examples/knapsack/    # Worked example
├── tests/                # Unit tests
├── pyproject.toml
└── README.md
```

## License

MIT

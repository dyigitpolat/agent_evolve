# agent_evolve

> **DRAFT for the OSS release — nothing here is published.** This file is a
> packaging draft; the shipping README is `README.md`. Open questions and
> release blockers live in `PACKAGING_TODO.md`.

Multi-objective optimization driven by a language model, **with the chance
baseline built in**. You describe a problem as five typed obligations, one
function runs it, and one command tells you whether the model is beating
uninformed sampling on *your* problem before you spend anything.

The package installs and runs with **no provider, no credential, and no
network**: `proposer="random"` is a first-class path, because uninformed
sampling over the same schema, budget, and evaluator is the control every
claim below is measured against.

## Install

```bash
pip install agent_evolve            # core: pydantic + python-dotenv, nothing else
pip install 'agent_evolve[llm]'     # adds the model-driven proposer (pydantic-ai)
```

## Quickstart: credential-free, a dozen lines

```python
from typing import Literal
from pydantic import BaseModel
from agent_evolve import ObjectiveSpec, optimize

class Config(BaseModel):
    cores: Literal[1, 2, 4, 8]
    cache: Literal["none", "l1", "l2"]

class Tune:
    candidate_model = Config
    objectives = [ObjectiveSpec("speed", "max"), ObjectiveSpec("watts", "min")]
    def seeds(self): return [{"cores": 1, "cache": "none"}]
    def evaluate(self, c): return {"speed": c["cores"] * {"none": 1.0, "l1": 1.4, "l2": 1.6}[c["cache"]], "watts": float(c["cores"])}

result = optimize(Tune(), budget=32, proposer="random")   # $0, no network
print(result.best.configuration, result.best.objectives)
```

That runs the genetic loop over the schema's discrete choices, costs nothing,
and prints a real recommendation (`best` is the minimax-rank pick over the
Pareto front; the front itself is `result.pareto_front`). Leaving
`proposer="auto"` does the same thing when no credential is found, and says
so. With a credential set, `proposer="llm"` puts a model in the loop — and
`agent_evolve check mypkg.problem:problem` runs both arms and reports whether
the model earned its cost.

## Describing your problem: the five obligations

The whole integration surface is `agent_evolve.contract.Problem` — five
obligations, all required, nothing else:

| # | obligation | what it is |
|---|---|---|
| 1 | `candidate_model` / `objectives` | the pydantic schema a proposal must satisfy, and what is being optimized — at least one `ObjectiveSpec(name, "min"/"max")` |
| 2 | `seeds()` | the starting points. Evaluated before any proposal, so every run answers "did this beat what I already had" |
| 3 | `validate(config)` | cheap, side-effect-free feasibility check returning a `ValidationOutcome` whose message says what is wrong *and* what would be acceptable — it is fed back to the proposer verbatim |
| 4 | `materialize(config)` | candidate → the artifact that gets measured. Expensive deterministic work goes here: two configs that materialize to equal artifacts are paid for once, not twice |
| 5 | `evaluate(artifact)` | measure it; return exactly the declared objectives, or raise `ValueError` with a usable message |

A pre-contract problem exposing only `objectives` + `evaluate` still works:
`as_problem` adapts it (identity materialization, empty seeds) — the
quickstart above leans on exactly that for `validate` and `materialize`.

## Scope: what is measured, and what is not

Every number in this table is quoted verbatim from a measured, journalled
result in the research archive (`papers/agent_evolve_aaai_2027/research_artifacts/`,
not shipped with the package). Scope is exactly the named domain, budget, and
seed count — nothing wider is claimed. The negative results are part of the
scope, not footnotes.

Sources: `head_to_head/aug03_ws_a_complete.md` (WS-A: the BOiLS `log2`
domain, B = 150, best-scalar, N = 3 per live arm, lower is better),
`timeloop_census/tl_ladder3_verdict.md` (Timeloop mapper co-design, exact
34-config frontier recall at B = 16, N = 15 per cell), and
`timeloop_census/tl_initsel_stage1.md` (same venue, initialization channel).

| scope statement | measured result, quoted exactly | source |
|---|---|---|
| The provider-free genetic loop beat a conventional MOEA at matched budget — on the one domain where that was measured. | `log2`, B=150: unguided median **1.407250** (min 1.403583, max 1.414250, N=3) vs NSGA-II pop20 median **1.410042** (N=6); beats the uniform median 1.417083 on 3/3 seeds. | aug03_ws_a_complete.md |
| **Negative (log2 p05 miss): "beats chance, not decisively."** The loop does not clear the uniform-p05 "decisive" bar — and neither does NSGA-II. Do not expect decisive dominance over well-seeded random search there. | unguided 1.40725 vs p05 bar 1.40567 — a miss of **0.0016**. | aug03_ws_a_complete.md |
| Structured recombination over a genome beat whole-artifact LLM rewrite at matched budget and anchor — with clean separation; the rewrite loop saturated below random search. | every unguided seed beat every `optimize_anything` seed (our worst 1.41425 < their best 1.42292); `optimize_anything` median **1.424083** at B=150 is worse than the uniform median **1.417083**. | aug03_ws_a_complete.md |
| **Negative: LLM operator guidance *hurt* on log2.** Headroom for guidance exists (oracle), but the shipped chooser did not capture it — do not buy `proposer="llm"` guidance expecting a lift there. | guided 1.421583 = **−0.0143** the wrong way vs unguided; oracle (perfect crossover choice) median **1.401667** clears the decisive bar; luna→oracle gap **0.0199**. | aug03_ws_a_complete.md |
| This is not a domain-specialist replacement. It is a cheap generic loop, priced accordingly. | unguided sits **0.0138** above the BOiLS specialist bar 1.3935; the specialist spent **41 h of acquisition** vs our **zero** guidance seconds and ~62 min of evaluation; whole design ≈ **$0.35** provider spend. | aug03_ws_a_complete.md |
| LLM guidance content does real work on Timeloop — but the measured effect is exactly one frontier config out of 34, at every model tier. | guided − shuffled premium **+0.029412** (= 1/34) at every rung; within-rung paired p_mean **0.0137 / 0.0024 / 0.0645** (luna/terra/sol, N=15). | tl_ladder3_verdict.md |
| **Negative (C4 content-flat): paying for a stronger model does not buy better guidance content.** Capability lifts compliance/request-shape, not content, and saturates at the middle rung. | guided medians 0.117647 → 0.147059 → 0.147059; shuffled 0.088235 → 0.117647 → 0.117647; premium flat at +0.029412 on all three rungs — "Under no reading does the premium grow with capability." Capability×content interaction **UNSUPPORTED**. | tl_ladder3_verdict.md |
| **Negative (INIT CHANNEL NULL): LLM-selected initialization picks genuinely better starting material, and it still does not raise the ceiling.** A better start substituted for what the search would have found anyway. | gen-0 frontier members 43 vs 21 for the matched random control (median 3 vs 1 per cell; paired one-sided p = **0.003540**), yet final recall median **0.147059** equals the sealed comparator "to the last digit". | tl_initsel_stage1.md |
| Whole-study costs stayed in single-digit dollars — the regime this package is for. | log2 head-to-head design ≈ **$0.35**; terra ladder **$0.8934** (22.3% of the $4 cap); init study **$1.0875** (under the $3 cap). | all three |

If your problem is a cheap-to-evaluate toy, none of the above transfers: the
LLM's per-call overhead is never repaid there (see `docs/scope.md` and the
knapsack example's own warning). Run `agent_evolve check` on *your* problem;
half a minute of it beats any table.

## MIGRATION: coming from pymoo

pymoo users can swap `minimize(...)` for `optimize(...)` without rewriting
their problem: wrap it with
`agent_evolve.integrations.pymoo_adapter.from_pymoo` and see the **swap
demo** at `examples/pymoo_swap/swap_demo.py` — the same pymoo problem object
run both ways (NSGA-II vs `optimize(..., proposer="random")`), 60 evaluations
each, no credential. The dev venv carries `pymoo 0.6.2` for it; a working
two-arm comparison on a native problem is `python examples/knapsack/run.py
--compare`.

| pymoo | agent_evolve |
|---|---|
| `Problem(n_var, n_obj, xl, xu)` box bounds | a pydantic `candidate_model` — fields are the variables; bounds/choices live in the schema |
| all-minimize `F` columns (negate to maximize) | named `ObjectiveSpec("speed", "max")` — direction is declared, never encoded by sign |
| `_evaluate(x, out)` filling `out["F"]` | `evaluate(artifact)` returning `{"name": value}`; `materialize()` builds what gets measured, and equal artifacts are paid for once |
| constraints `out["G"] <= 0` | `validate(config)` returning an explained `ValidationOutcome` (cheap, pre-evaluation) or `ValueError` from `evaluate` (measurement-revealed) |
| `minimize(problem, NSGA2(pop_size=...), ("n_evals", N))` | `optimize(problem, budget=N)` — budget counts evaluator calls, the expensive thing, and is the only sizing number |
| `res.X`, `res.F` | `result.pareto_front` (each `Candidate` has `.configuration` and `.objectives`), `result.best` |
| sampling/initial population | `seeds()` — evaluated first, so the run reports whether it beat what you already had |

Two honest scope notes for migrants. First, the genetic loop's operators work
over **discrete loci** — `enum`/`Literal`/`bool` fields
(`agent_evolve.policies.genetic`) — so `from_pymoo` *quantizes* each
continuous box-bounded axis onto an evenly spaced `Literal` grid (default 16
points, both endpoints included). If grid resolution is what your ZDT/DTLZ-style
problem is about, NSGA-II remains the right tool — the measured comparison in
the scope table is on a discrete flow-synthesis domain, not a continuous one.
Second, keep running your NSGA-II arm: `agent_evolve check` exists precisely
because on some problems uninformed sampling already does the job (see the
table).

## License

MIT — `LICENSE.draft` in this packaging draft; holder line pending owner
review before release.

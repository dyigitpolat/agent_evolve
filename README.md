# agent_evolve

**A drop-in replacement for NSGA-II that cannot do worse than the classical
optimizer it replaces.** A language model authors the search's generative
machinery — the sampler that proposes candidates, the surrogate that screens
them, the prior that seeds them — and every one of those authored objects has to
beat a measured comparator on *today's* evidence before it is allowed to act. If
it does not, it does not run, and the classical path runs instead.

So the floor is the classical baseline and the ceiling is measured, not
promised. That is a guarantee about the machinery, not a hope about the model,
and the reason we can put it in the first paragraph is that we watched it fire:

| the authored object | what arbitration did to it | evidence |
| --- | --- | --- |
| model-written **surrogate** for pre-screening | Admitted, then beaten. Across the two columns with plentiful cross-validation evidence it **cleared the gate on 46 refreshes and won 0 of them**; the cheap 100-mapping fidelity took **all 313** installs. In the richest column alone it cleared **41 of 188** refreshes and won none. The guided and rule arms came out **byte-identical on 36/36 paired seeds at both budgets** — a contribution of literally zero. | wave L, `aug15_llm_value_tlv2.md` |
| measurement-conditioned **re-authoring channel** | Fired **447 times** and came out **below its own shuffled-evidence control**. Does not ship. | wave L, `aug15_measurement_conditioned.md` |
| graded **locus prior** | Neither beat the stack without it nor separated from its own control. Ships as documented venue knowledge, not as run-time reasoning. | wave N, `aug16_p1b_margin.md` |
| model **choosing among evolutionary operators** | Eight mechanisms, eight sealed verdicts: guided runs sit at the *unguided* median. The published result is the negative one. | the paper, §ledger |

An artifact that clears a gate 46 times and never wins is not being blocked, it
is being beaten — and *the fallback is the arm that actually ran*. When the
authored machinery does earn its place, the same instruments say by how much: on
a real RTL-to-GDS physical-design flow the authored stack reached a sealed target
in **1.886× fewer charged evaluations than NSGA-II** (median charges, `N = 48` in
four blocks, `p = 0.0013`) and — the decomposition that makes it attributable
rather than merely positive — **1.959× fewer than our own credential-free loop**,
which does not separate from NSGA-II at all.

Read [what is measured and what is not](#what-is-measured-and-what-is-not)
before adopting it. Four of ten measured venues host a claim, and the reasons
the other six host nothing are published rather than omitted.

---

## Install

```bash
pip install agent_evolve                # core: pydantic + python-dotenv, nothing else
pip install 'agent_evolve[llm]'         # adds the model-driven proposer
pip install 'agent_evolve[pymoo]'       # adds the classical comparator + the swap example
```

Python 3.11 or newer. The core install has **two** dependencies, needs no
credential, no provider account and no network, and is a first-class path rather
than a degraded one: uninformed sampling over the same schema, the same budget
and the same evaluator is the control every number above is measured against.

## Run it with no credential and no cost

```console
$ python examples/build_flags/run.py
generation 0: 7 evaluated, population 7
generation 1: 6 evaluated, 13 of 40 budget used
generation 2: 6 evaluated, 17 of 40 budget used
generation 3: 6 evaluated, 22 of 40 budget used
generation 4: 7 evaluated, 28 of 40 budget used
generation 5: 8 evaluated, 34 of 40 budget used
generation 6: 6 evaluated, 38 of 40 budget used
generation 7: 2 evaluated, 39 of 40 budget used
generation 8: 0 evaluated, 39 of 40 budget used
generation 9: 1 evaluated, 39 of 40 budget used
generation 10: 0 evaluated, 39 of 40 budget used
generation 11: 0 evaluated, 39 of 40 budget used
generation 12: 1 evaluated, 40 of 40 budget used
budget exhausted after 40 evaluations; stopping at gen 13

evaluations  40
pareto front 11

 runtime_ms   binary_kb  configuration
     604.49      648.82  Ofast avx512 lto=full unroll=0 fast=1 inline=873
     655.01      511.13  Ofast avx2 lto=full unroll=1 fast=1 inline=32
     702.86      510.29  Ofast avx2 lto=none unroll=0 fast=1 inline=32
     727.10      480.83  Ofast sse4 lto=thin unroll=0 fast=1 inline=190
     757.49      430.52  Ofast none lto=full unroll=0 fast=1 inline=190
     795.89      285.59  Os avx2 lto=full unroll=1 fast=1 inline=190
     802.25      270.26  Os avx2 lto=full unroll=1 fast=1 inline=75
     841.13      252.94  Os avx2 lto=full unroll=0 fast=1 inline=190
     850.47      239.84  Os avx2 lto=full unroll=0 fast=1 inline=75
    1016.59      222.10  Os none lto=full unroll=0 fast=1 inline=75
    1027.03      216.10  Os none lto=full unroll=0 fast=0 inline=75
```

That is the real loop, on the uninformed sampler, for nothing. `budget` counts
artifacts measured and it is a hard cap: say 40 and you are billed for 40. The
generations that evaluate 0 are `validate` doing its work — every child was
rejected before anything was built, and a rejection is not an artifact. Where a
generation evaluates more than the budget moves (generation 7 evaluates 2 and
charges 1), the difference is `materialize` recognising an artifact already
measured.

`inline_threshold` is a bounded integer rather than an enumeration, and the
front above moves along it — 32, 75, 190, 873. Until 0.5.0 it did not: bounded
numeric loci declared no finite domain, so the operators left them frozen. At
the same 40-evaluation budget this front held 6 rows, every one of them at
`inline=75`.

## Before you spend anything: ask whether *any* optimizer can win here

`diagnose` needs no model and no credential. It probes your problem through its
own `validate`/`materialize`/`evaluate` pipeline and answers the prior question —
is there headroom at this budget at all?

```console
$ agent_evolve diagnose examples.build_flags.problem_def:problem --budget 40
agent_evolve diagnose: examples.build_flags.problem_def:problem

problem check: BuildFlags
  budget assessed  40 evaluations
  probe spent      120 draws, 73 evaluated  (failure rate 39%)

  search space  (6 loci)
    opt_level:6 vectorize:4 link_time:3 unroll_loops:2 fast_math:2
    inline_threshold:64(projected)
    undeclared domains: none
  ...
  headroom at budget 40
    runtime_ms (min)  best of probe 618.31  expected best of 40 random 625.113 +/- 13.2479  headroom 6.80339  [below noise]
    binary_kb (min)  best of probe 232  expected best of 40 random 240.515 +/- 14.3874  headroom 8.51526  [below noise]

  verdict
    The best value the probe found on every objective is within noise of
    what 40 random draws are expected to reach: no optimizer can demonstrate
    an advantage here at this budget. Raise the budget, or reshape the
    search space, before crediting any optimizer with a win.
```

Note what it just did: it returned a **no-headroom verdict on our own shipped
example** rather than a win. That is the same arbitration as the table at the
top, one level up — a tool that will tell you not to use it is the only kind
whose recommendation means anything.

The verdict got *harder* to dismiss in 0.5.0, not easier. It used to arrive with
a footnote — one of the six loci declared no finite domain, so the probe could
not vary `inline_threshold` and any headroom along that axis was invisible to
the check. That axis is now projected onto 64 points and the probe does vary it;
`undeclared domains: none`. The verdict is the same, and it is now a statement
about the whole space rather than about five sixths of it. `64(projected)` marks
an axis read off bounds rather than declared as a set — searchable, but on the
grid rather than the continuum.

When `diagnose` says there *is* headroom, `agent_evolve check` is the next
question: does a model beat uninformed sampling on your problem, at the same
budget, against the same evaluator? `--baseline-only` runs just the free arm.

## Replacing NSGA-II: the whole diff

`examples/pymoo_swap/` holds two files that optimize the **same pymoo problem
object**. `nsga2_baseline.py` is what a pymoo user already writes.
`agentevolve_swap.py` is the same file after the swap. This is the complete
difference between them:

```diff
--- examples/pymoo_swap/nsga2_baseline.py
+++ examples/pymoo_swap/agentevolve_swap.py
@@
-"""pymoo NSGA-II on a pymoo problem: the classical arm. See README.md in this directory."""
+"""agent_evolve on the same pymoo problem: the swapped arm. See README.md in this directory."""

 from pymoo.problems import get_problem

 # ---- the problem: untouched by the swap, which is the whole claim -----------
 problem = get_problem("zdt1", n_var=4)

 # ---- the optimizer: this block, and only this block, is the diff -----------
-from pymoo.algorithms.moo.nsga2 import NSGA2
-from pymoo.optimize import minimize
+from agent_evolve import optimize
+from agent_evolve.integrations.pymoo_adapter import from_pymoo

-res = minimize(problem, NSGA2(pop_size=12), ("n_gen", 5), seed=1, verbose=False)
-evaluations = res.algorithm.evaluator.n_eval
-front = sorted((float(row[0]), float(row[1])) for row in res.F)
+res = optimize(from_pymoo(problem), budget=60, proposer="random", seed=1)
+evaluations = res.evaluations
+front = sorted((c.objectives["f0"], c.objectives["f1"]) for c in res.pareto_front)
```

Six changed lines, of which one is a docstring and two are the import of a
different optimizer. **Three lines are API contact**: call it, read the
evaluation count, read the front. Your problem definition does not change at
all — the pymoo `Problem` object is passed through untouched.

Both arms run credential-free, at 60 evaluations each:

```console
$ python examples/pymoo_swap/nsga2_baseline.py
arm          : nsga2_baseline
evaluations  : 60
front size   : 4
best f0      : 0.002854
best f1      : 1.896090

$ python examples/pymoo_swap/agentevolve_swap.py
arm          : agentevolve_swap
evaluations  : 60
front size   : 3
best f0      : 0.000000
best f1      : 0.869703
```

`tests/test_pymoo_swap_acceptance.py` **recomputes that diff on every suite run**
and fails if it grows past six lines, or if the API-contact count moves off
three, or if the two files stop optimizing the same object. The drop-in claim is
exactly as true as the diff is short, so the diff is the test and not the prose.
It also runs both scripts and checks they spent the same budget and reported the
same quantities. It deliberately does **not** assert which arm wins: that is a
measurement, not an acceptance criterion.

ZDT1 is a smooth continuous problem, which is *not* the regime this tool is for
(see below). `from_pymoo` quantizes each continuous box-bounded axis onto an
evenly spaced `Literal` grid — 16 points by default, both endpoints included —
because the genetic operators read their allowed values from the schema and
refuse to invent any. If grid resolution is what your problem is about, NSGA-II
remains the right tool.

### Coming from pymoo: the concept map

| pymoo | agent_evolve |
| --- | --- |
| `Problem(n_var, n_obj, xl, xu)` box bounds | a pydantic `candidate_model` — fields are the variables, bounds and choices live in the schema |
| all-minimize `F` columns (negate to maximize) | named `ObjectiveSpec("speed", "max")`; direction is declared, never encoded by sign |
| `_evaluate(x, out)` filling `out["F"]` | `evaluate(artifact)` returning `{"name": value}`; `materialize()` builds what gets measured, and equal artifacts are paid for once |
| constraints `out["G"] <= 0` | `validate(config)` returning an explained `ValidationOutcome` (cheap, pre-evaluation), or `ValueError` from `evaluate` (measurement-revealed) |
| `minimize(problem, NSGA2(pop_size=...), ("n_evals", N))` | `optimize(problem, budget=N)` — budget counts evaluator calls, the expensive thing, and is the only sizing number |
| `res.X`, `res.F` | `result.pareto_front` (each `Candidate` has `.configuration` and `.objectives`), `result.best` |
| sampling / initial population | `seeds()` — evaluated first, so the run reports whether it beat what you already had |

## Describing your problem: five obligations

That is the whole integration surface (`agent_evolve.contract.Problem`).

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

The `validate` message is fed back to the proposer verbatim, so say what is
wrong **and** what would be acceptable.

**Why `materialize` is separate from `evaluate`.** Two configurations often
produce the same artifact — the same build, the same mapping, the same
deployment. Materializing first means the second one is free instead of being
paid for twice. Put anything cheap and deterministic there and keep `evaluate`
for the expensive part.

**Declare your domains in the schema.** Anything that reads it — including the
baseline — then draws only legal candidates.

**A bounded number is a domain.** Since 0.5.0, `workers: int = Field(..., ge=1,
le=64)` above is a searchable axis: bounded integers are enumerated outright
when the span is at most **64** and projected onto 64 evenly spread points
otherwise, and a doubly bounded `float` becomes a **16**-point grid with a point
on each inclusive endpoint. `multipleOf` is honoured, so every value the
projection emits is one the schema would accept, and an excluded endpoint is
never emitted. If an axis deserves a different resolution, say so on the field:

```python
inline_threshold: int = Field(..., ge=0, le=1000,
                              json_schema_extra={"agent_evolve": {"grid": 16}})
# 16 points: 0, 67, 133, 200, ..., 933, 1000
```

The override is clamped to `[2, 256]` — one point is a constant, and a domain
nobody can enumerate cheaply is not one an operator can draw from. `diagnose` marks a
projected axis as `inline_threshold:64(projected)`, because *64 values* means
something different when the schema declared 64 values than when it declared an
interval: the optimizer moves on the grid, not on the continuum.

A locus with **no finite reading** — an open `str`, a bound on only one side —
is still one the operators leave alone, and `diagnose` still names those
explicitly under `undeclared domains`, because a search space that cannot be
varied is the commonest reason a run goes nowhere.

Problems written against the older two-method contract (`objectives` and
`evaluate`) keep working unchanged: `as_problem` adapts them with identity
materialization and empty seeds.

## Choosing a proposer

```python
optimize(problem, budget=40, proposer="random")   # free, no credentials, the baseline
optimize(problem, budget=40, proposer="llm")      # a model
optimize(problem, budget=40)                      # auto: llm if a key exists, else random
```

`auto` says out loud which one it picked.

**What `proposer="llm"` buys is the configuration the ablations picked.**
Model-proposed initialization — the six-arm ablation's strongest arm, at 11×
fewer evaluations to target, better on **40 of 40** paired seeds, for **one**
call. Model-authored surrogate screening — the shipped default, unchanged, and
still admitted only when it out-validates the rules, which is the arbitration
the first table describes rather than a win claim. And at `budget >= 48`, a
crossed screen sized from the budget, read by a model-weighted graded prior:
the ablation's guidance arm, 4.60×. Below 48 the screen is skipped and the
prior stays the rule form, because the prior only ever acts on a screen's
evidence. Every one of those resolutions is announced through `on_progress`
rather than taken silently, and each is overridable by name (`prior=`,
`structure_budget=`, `authorship=`).

**What it does not buy is the per-offspring chooser**, which is `chooser="off"`
by default: a model call per offspring returned **ten sealed null verdicts** and
consumed 61% of that ablation's whole ledger for **0.94×** the speed of doing
nothing. `chooser="llm"` still buys it — the negative result is published, not
hidden, and the mechanism is reachable — and asking for it on a run that makes
no model call is refused by name rather than quietly ignored.

## What is measured, and what is not

Every claim here is **venue-scoped**: it is a statement about the named domain,
budget, estimand and seed count, and nothing wider. There is no all-domains
sentence in this README because there is none in the paper either.

**Of ten measured venues, four host a general claim.** The reasons the other six
host nothing — a venue that cannot separate any two optimizers, a sweep struck
for venue validity, an axis that read null — are published rather than omitted.

**What fires:**

- **The dominance row.** On a real RTL-to-GDS physical-design flow: **1.886×**
  fewer charged evaluations than NSGA-II to a sealed target (median charges,
  `N = 48` in four blocks, `p = 0.0013`), and **1.959×** fewer than our own
  credential-free loop, which does not separate from NSGA-II at all. A second
  dominance row stands on accelerator architecture search.
  *And it still misses the bar that venue's own specialists set: 54.0 against
  43.5.* This is a cheap generic loop priced accordingly, not a domain
  specialist.
- **Authoring quality scales on two axes.** Model class and test-time reasoning
  effort, by the same amount on one venue, stack, estimand and scorer; composed
  on one grid the two are super-additive. At **zero** effort the capability jump
  is not measurable at all.
- **The buy on the effort axis is `none` → `high`**: median exact-front recall
  `0.426471 → 0.588235`, step **+0.161765** at 2.75× its own bar, `N = 48` per
  arm, `p = 5×10⁻⁶`, achieved power 1.000.

**What is not established, stated as plainly as what is:**

- **Saturation above `high` is UNRESOLVED, not established.** `high` → `xhigh`
  did not separate: +0.0230, `p = 0.245`, achieved power **0.073**. `max` is
  descriptive only (`n = 20`, never laddered, in no test).
- **`xhigh` is a request, not a dose.** At the same label one model reasoned
  32,156 tokens where another reasoned 53,698. The label names what you asked
  for, not what you got, and the two are not the same variable.
- **The step from mechanism to endpoint is attenuated 35.2×.** The dose that
  moves an authored artifact by **+0.161765** median recall moves the search
  endpoint by **+0.004596** — real, overwhelmingly significant (`p = 5×10⁻⁶`,
  `N = 48`), non-inferior and superior, and *small*. We price that step rather
  than assuming it.
- **Absolute endpoint levels are not comparable across configurations**, and we
  do not quote them across them.
- **The ladders rest on one venue family.**
- **Model guidance of *operator choice* did not work.** Eight mechanisms, eight
  sealed verdicts, guided at the unguided median. Since 0.5.0 it is **off by
  default** and reachable only by asking for it by name (`chooser="llm"`): a
  negative result belongs in the product as a switch nobody is billed for, not
  as a default.
- **Two of our own published sentences are withdrawn**, with the defect in each
  stated in the paper's discussion. We do not re-assert them from the archive.

**Regime guidance** — where this helps and where it does not, with the
wall-clock and break-even numbers — is in [`docs/scope.md`](docs/scope.md). The
short version: strongest on categorical, constrained problems with expensive
evaluators and small budgets; marginal on smooth continuous problems, which are
Bayesian-optimization territory; and **slower in wall-clock** than a classical
optimizer unless each evaluation costs minutes. Read it before adopting.

The evidence archive itself (sealed preregistrations, spend ledgers, numbers
bundles) is **not distributed with the package** — it is the research record
behind the paper, and rows in it assert and re-verify their own absolute paths.
The wave names in the table at the top are its addresses.

## The credentialed path, and what it costs

Everything above runs without a credential. The model-driven proposer needs one,
and it is documented separately because it spends money.

```bash
pip install 'agent_evolve[llm]'
export AGENTEVOLVE_DOTENV=/path/to/your/.env      # nothing is loaded unless you name it
agent_evolve check yourpkg.problem:problem --budget 40
```

`check` runs an uninformed sampler and a model against the same problem, the
same budget and the same evaluator, then says which won. It **prints the model
and its price before it spends anything**, so a default nobody chose cannot bill
anyone. The default model is cheap on purpose: `$0.10` per million input tokens
and `$0.60` per million output.

Every run reports what it actually spent, and `--json` gives you the figure:

```console
$ agent_evolve run mypkg.problem:problem --budget 12 --proposer llm --seed 5 --json
{
  "evaluations": 10,
  "provider_usage": {
    "calls": 26,
    "cost_usd": "0.013226",
    "input_tokens": 18045,
    "output_tokens": 19035,
    "model": "openrouter:openai/gpt-5.6-luna",
    "reported_by": "openrouter response usage; cost derived from the package's published price table"
  }
}
```

Two things about that block. The tokens are the provider's own count and the
price is the table above, and `reported_by` says so — a dollar figure that does
not name what measured it is a number nobody checked. And `calls: 0` on a
credential-free run means *counted and none occurred*, not *not recorded*: the
block is always present.

A 12-evaluation run cost **1.3 cents**. In the research record, whole multi-cell
studies stayed in single-digit dollars.

This library **never searches upward for a `.env`**, and that is a fix rather
than a preference. An earlier version called `dotenv.load_dotenv()` with no
argument, which walks up the directory tree until it finds any `.env` — so a run
inside a monorepo silently adopted an unrelated project's credentials, and a run
launched specifically to prove it made no provider call was handed the key back
by the very next line. Name the file or nothing is loaded.

## Configuration

None of these is required.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AGENTEVOLVE_MODEL` | model id | `openrouter:openai/gpt-5.6-luna` |
| `AGENTEVOLVE_HARNESS` | adapter for the `llm` proposer | `pydantic_ai` |
| `AGENTEVOLVE_TEMPERATURE` | sampling temperature | provider default |
| `AGENTEVOLVE_DOTENV` | path to a `.env` to load; nothing is loaded unless you name it | unset |
| `AGENTEVOLVE_SCRUBBED` | names to remove and never reintroduce from a file | unset |

## CLI

```
agent_evolve version
agent_evolve diagnose PROBLEM [--budget N] [--probe N]      # no model, no credential
agent_evolve check    PROBLEM [--budget N] [--baseline-only]
agent_evolve run      PROBLEM [--budget N] [--proposer auto|llm|random]
                              [--strategy auto|genetic|authoring]
                              [--authorship ...] [--prior ...] [--effort ...]
                              [--chooser off|llm] [--structure-budget auto|N]
                              [--seal PATH] [--journal PATH] [--json]
```

`PROBLEM` is `module:attribute`. `run --json` prints one machine-readable
document — `best`, `pareto_front`, `evaluations`, `history`, `telemetry`,
`provider_usage` — instead of prose, for scripted use.

## Examples

| Example | Credentials | What it shows |
| --- | --- | --- |
| `examples/build_flags/` | none | The regime this tool suits: categorical, constrained, expensive evaluator |
| `examples/knapsack/` | none | The five obligations, shortest possible. Still an honest negative, on better grounds since 0.5.0: its selection locus now projects (`selection[0]:10(projected)`, `undeclared domains: none`) and the probe does vary it, and `diagnose` still refuses to credit any optimizer — 40 random draws are expected to reach the same best value (150) and the same best weight (5) the probe found |
| `examples/pymoo_swap/` | none (needs `[pymoo]`) | The NSGA-II swap, as two files and a six-line diff |

## Tests

```bash
pip install 'agent_evolve[dev,pymoo]'
python -m pytest tests/test_public_contract.py           # the shipped surface, ~1 second
python -m pytest tests/                                  # the package, offline
python -m pytest tests/ -m research                      # needs the research corpus
```

The default run is what someone who cloned this repository can actually use.
Tests driving the research campaign are marked `research` and deselected; they
are skipped rather than failed when the corpus is absent.

The project's `addopts` include `-q`, and on pytest 9 that suppresses the final
pass count. Add `-p no:cacheprovider` and drop `-q` (`python -m pytest tests/
-p no:cacheprovider --tb=line`) if you want the count printed.

## License

MIT — see [LICENSE](LICENSE). **The copyright holder line is pending the
project owner's sign-off and must be settled before any public release**; see
`PACKAGING_TODO.md`.

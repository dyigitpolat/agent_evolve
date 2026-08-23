# The pymoo NSGA-II swap, as an executable acceptance test

Two files here optimize the **same pymoo problem object**:

| file | arm |
| --- | --- |
| `nsga2_baseline.py` | pymoo's NSGA-II, as a pymoo user already writes it |
| `agentevolve_swap.py` | the same file after swapping in `agent_evolve` |

Both need no credential, no provider account and no network. Both spend exactly
60 evaluations.

```bash
pip install 'agentevolve-optimizer[pymoo]'
python examples/pymoo_swap/nsga2_baseline.py
python examples/pymoo_swap/agentevolve_swap.py
```

## The whole difference

```console
$ diff -u examples/pymoo_swap/nsga2_baseline.py examples/pymoo_swap/agentevolve_swap.py
```

```diff
@@ -1,4 +1,4 @@
-"""pymoo NSGA-II on a pymoo problem: the classical arm. See README.md in this directory."""
+"""agent_evolve on the same pymoo problem: the swapped arm. See README.md in this directory."""

 from pathlib import Path

@@ -8,12 +8,12 @@
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
 # ---- end of the diff -------------------------------------------------------

 print(f"arm          : {Path(__file__).stem}")
```

Six changed lines. One is a docstring, two are the import of a different
optimizer, and **three are API contact**: call it, read the evaluation count,
read the front. The problem definition, the reporting and everything else are
byte-identical — the pymoo `Problem` object is passed through untouched, which is
the actual content of the drop-in claim.

## Why this is a test and not a walkthrough

`tests/test_pymoo_swap_acceptance.py` runs in the default suite and:

* recomputes the diff above and **fails if it grows** past six changed lines, or
  if the API-contact count moves off three;
* asserts both files still build the same pymoo problem, so the comparison cannot
  quietly become two different problems;
* asserts the swap imports only the top-level package and the named integration
  module, and reads no private attribute;
* runs both scripts and checks they spent the same budget and reported the same
  quantities.

The first four need no pymoo, so they cannot be skipped into irrelevance in a
minimal environment. The last one skips without pymoo installed.

If a change to the public API makes the six-line budget unreachable, raising the
number in that test is the wrong repair. A swap that costs more lines is a
finding about the API's genericity, and it belongs in the API.

## What it does *not* assert

Which arm wins. That is a measurement, not an acceptance criterion. ZDT1 is a
smooth continuous problem — precisely the regime `docs/scope.md` says this tool
is marginal on — and `from_pymoo` quantizes each continuous axis onto a 16-point
`Literal` grid because the genetic operators refuse to invent values the schema
does not declare. Keep running your NSGA-II arm; `agent_evolve diagnose` and
`agent_evolve check` exist because on many problems the classical arm, or plain
uninformed sampling, already does the job.

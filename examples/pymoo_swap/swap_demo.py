"""The pymoo swap: same problem object, three code lines each way, 60 evaluations each.

A pymoo user runs NSGA-II with ``minimize(...)``. The agent_evolve user wraps
the *same* problem object with ``from_pymoo`` and calls ``optimize(...)``.
Nothing here needs a credential or the network: ``proposer="random"`` is the
zero-cost control arm, and the point of the demo is the swap, not the winner.

Run it::

    .venv/bin/python examples/pymoo_swap/swap_demo.py
"""

from pymoo.problems import get_problem

problem = get_problem("zdt1", n_var=4)   # small built-in pymoo problem

# -- Way 1: pymoo's NSGA-II (12 x 5 = 60 evaluations) ------------------------
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

res = minimize(problem, NSGA2(pop_size=12), ("n_gen", 5), seed=1, verbose=False)

# -- Way 2: agent_evolve on the same object (budget = 60 evaluations) --------
from agent_evolve.api import optimize
from agent_evolve.integrations.pymoo_adapter import from_pymoo

result = optimize(from_pymoo(problem), budget=60, proposer="random", seed=1)

# ----------------------------------------------------------------------------
print(f"pymoo NSGA-II : {len(res.F):2d} nondominated points after 60 evals; "
      f"best f0={res.F[:, 0].min():.4f}, best f1={res.F[:, 1].min():.4f}")
print(f"agent_evolve  : {len(result.pareto_front):2d} nondominated points after "
      f"{result.evaluations} evals; "
      f"best f0={min(c.objectives['f0'] for c in result.pareto_front):.4f}, "
      f"best f1={min(c.objectives['f1'] for c in result.pareto_front):.4f}")
print(f"recommended   : {result.best.configuration} -> {result.best.objectives}")

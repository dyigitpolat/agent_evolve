"""pymoo NSGA-II on a pymoo problem: the classical arm. See README.md in this directory."""

from pathlib import Path

from pymoo.problems import get_problem

# ---- the problem: untouched by the swap, which is the whole claim -----------
problem = get_problem("zdt1", n_var=4)

# ---- the optimizer: this block, and only this block, is the diff -----------
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

res = minimize(problem, NSGA2(pop_size=12), ("n_gen", 5), seed=1, verbose=False)
evaluations = res.algorithm.evaluator.n_eval
front = sorted((float(row[0]), float(row[1])) for row in res.F)
# ---- end of the diff -------------------------------------------------------

print(f"arm          : {Path(__file__).stem}")
print(f"evaluations  : {evaluations}")
print(f"front size   : {len(front)}")
print(f"best f0      : {min(f0 for f0, _ in front):.6f}")
print(f"best f1      : {min(f1 for _, f1 in front):.6f}")

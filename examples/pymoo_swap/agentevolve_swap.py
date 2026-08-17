"""agent_evolve on the same pymoo problem: the swapped arm. See README.md in this directory."""

from pathlib import Path

from pymoo.problems import get_problem

# ---- the problem: untouched by the swap, which is the whole claim -----------
problem = get_problem("zdt1", n_var=4)

# ---- the optimizer: this block, and only this block, is the diff -----------
from agent_evolve import optimize
from agent_evolve.integrations.pymoo_adapter import from_pymoo

res = optimize(from_pymoo(problem), budget=60, proposer="random", seed=1)
evaluations = res.evaluations
front = sorted((c.objectives["f0"], c.objectives["f1"]) for c in res.pareto_front)
# ---- end of the diff -------------------------------------------------------

print(f"arm          : {Path(__file__).stem}")
print(f"evaluations  : {evaluations}")
print(f"front size   : {len(front)}")
print(f"best f0      : {min(f0 for f0, _ in front):.6f}")
print(f"best f1      : {min(f1 for _, f1 in front):.6f}")

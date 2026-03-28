#!/usr/bin/env python3
"""Run the knapsack example via the Python API.

Usage
-----
    python run.py                              # uses default model
    python run.py --model anthropic:claude-haiku-4-5

For CLI usage with the same problem, place ``problem_def.py`` in the working
directory and run::

    kedi path/to/agent_evolve/src/agent_evolve/evolve.kedi --adapter-model openai:gpt-4o
"""

import argparse
import sys
from pathlib import Path

import dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Repo .env first, then knapsack/.env (later file wins). override=True beats shell/IDE env.
dotenv.load_dotenv(_REPO_ROOT / ".env", override=True)
dotenv.load_dotenv(_REPO_ROOT / "examples" / "knapsack" / ".env", override=True)

sys.path.insert(0, str(_REPO_ROOT / "src"))

from agent_evolve import AgentEvolver
from problem_def import KnapsackProblem


def main() -> None:
    parser = argparse.ArgumentParser(description="agent_evolve knapsack example")
    parser.add_argument("--model", default="openai:gpt-4o", help="LLM model string")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--pop-size", type=int, default=6)
    args = parser.parse_args()

    evolver = AgentEvolver(
        model=args.model,
        pop_size=args.pop_size,
        generations=args.generations,
        candidates_per_batch=4,
        max_regen_rounds=5,
    )

    result = evolver.optimize(KnapsackProblem())

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best configuration: {result.best.configuration}")
    print(f"Best objectives:    {result.best.objectives}")
    print(f"Pareto front size:  {len(result.pareto_front)}")
    print(f"Total evaluated:    {len(result.all_candidates)}")

    if result.pareto_front:
        print("\nPareto front:")
        for i, c in enumerate(result.pareto_front, 1):
            print(f"  {i}. {c.configuration}  ->  {c.objectives}")


if __name__ == "__main__":
    main()

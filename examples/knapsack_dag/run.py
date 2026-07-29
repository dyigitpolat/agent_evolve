#!/usr/bin/env python3
"""Run the DAG + synergy knapsack example via the Python API.

Usage
-----
    python run.py
    python run.py --model anthropic:claude-haiku-4-5
"""

import argparse
import sys
from pathlib import Path

from agent_evolve.settings import load_credentials
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_credentials(_REPO_ROOT / ".env", override=True, optional=True)
load_credentials(_REPO_ROOT / "examples" / "knapsack_dag" / ".env", override=True, optional=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_evolve import optimize
from problem_def import DagSynergyKnapsackProblem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="agent_evolve DAG synergy knapsack example"
    )
    parser.add_argument("--model", default=None, help="LLM model string")
    parser.add_argument(
        "--harness",
        default="pydantic_ai",
        help="registered harness id (default: pydantic_ai)",
    )
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=8)
    args = parser.parse_args()

    result = optimize(
        DagSynergyKnapsackProblem(),
        harness=args.harness,
        model=args.model,
        pop_size=args.pop_size,
        generations=args.generations,
        candidates_per_batch=4,
        max_regen_rounds=5,
        log=lambda m: print(m, flush=True),
    )

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

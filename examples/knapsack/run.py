#!/usr/bin/env python3
"""Run the knapsack example.

No credentials needed::

    python examples/knapsack/run.py

That uses the uninformed sampler, costs nothing, touches no network, and
prints a real Pareto front. It is also the baseline: to find out whether a
model beats it on this problem, run both arms::

    python examples/knapsack/run.py --compare

With a provider credential set, ``--proposer llm`` uses a model instead.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_evolve import optimize  # noqa: E402
from problem_def import Knapsack  # noqa: E402


def _show(label: str, result) -> None:
    print(f"\n{label}")
    print(f"  evaluations  {result.evaluations}")
    print(f"  pareto front {len(result.pareto_front)}")
    print(f"  best         {result.best.configuration} -> {result.best.objectives}")
    for i, c in enumerate(
        sorted(result.pareto_front, key=lambda c: c.objectives.get("total_weight", 0.0)), 1
    ):
        print(f"    {i}. {c.configuration} -> {c.objectives}")


def _best_value(result) -> float:
    values = [c.objectives["total_value"] for c in result.pareto_front]
    return max(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="agent_evolve knapsack example")
    parser.add_argument("--budget", type=int, default=40, help="evaluations")
    parser.add_argument(
        "--proposer",
        default="random",
        choices=("auto", "llm", "random"),
        help="'random' is the default here: no credentials, no cost",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run the model against the uninformed baseline and say which won",
    )
    args = parser.parse_args()

    problem = Knapsack()

    if not args.compare:
        result = optimize(
            problem,
            budget=args.budget,
            model=args.model,
            proposer=args.proposer,
            seed=args.seed,
            on_progress=lambda m: print(m, flush=True),
        )
        _show(f"result ({args.proposer} proposer)", result)
        return

    print("Running the uninformed baseline five times, then the model once.\n")
    baselines = [
        optimize(problem, budget=args.budget, proposer="random", seed=s) for s in range(5)
    ]
    baseline_bests = [_best_value(r) for r in baselines]
    _show("random baseline (first of five)", baselines[0])
    print(f"\n  best total_value across five random runs: {baseline_bests}")
    print(f"  median: {statistics.median(baseline_bests):g}")

    try:
        model_result = optimize(
            problem, budget=args.budget, model=args.model, proposer="llm", seed=args.seed
        )
    except Exception as error:  # noqa: BLE001
        print(f"\n  model run unavailable ({type(error).__name__}: {error})")
        print("  The baseline above still stands, and cost nothing.")
        return

    _show("model", model_result)
    model_best = _best_value(model_result)
    median = statistics.median(baseline_bests)
    print(f"\n  model best total_value {model_best:g} vs random median {median:g}")
    if model_best > median:
        print("  The model beat the uninformed baseline on this run.")
    else:
        print(
            "  The model did not beat uninformed sampling here. On a problem this "
            "cheap to\n  evaluate, that is the expected outcome -- see docs/scope.md."
        )


if __name__ == "__main__":
    main()

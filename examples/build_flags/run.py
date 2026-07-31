#!/usr/bin/env python3
"""Run the compiler-flag example. No credentials, no toolchain, no cost.

    python examples/build_flags/run.py
    python examples/build_flags/run.py --proposer llm     # needs a credential
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_evolve import optimize  # noqa: E402
from problem_def import BuildFlags  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="agent_evolve build-flag example")
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--proposer", default="random", choices=("auto", "llm", "random"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = optimize(
        BuildFlags(),
        budget=args.budget,
        model=args.model,
        proposer=args.proposer,
        seed=args.seed,
        on_progress=lambda m: print(m, flush=True),
    )

    print(f"\nevaluations  {result.evaluations}")
    print(f"pareto front {len(result.pareto_front)}\n")
    print(f"{'runtime_ms':>11}  {'binary_kb':>10}  configuration")
    for c in sorted(result.pareto_front, key=lambda c: c.objectives["runtime_ms"]):
        cfg = c.configuration
        flags = (
            f"{cfg['opt_level']} {cfg['vectorize']} lto={cfg['link_time']} "
            f"unroll={int(cfg['unroll_loops'])} fast={int(cfg['fast_math'])} "
            f"inline={cfg['inline_threshold']}"
        )
        print(f"{c.objectives['runtime_ms']:>11.2f}  {c.objectives['binary_kb']:>10.2f}  {flags}")


if __name__ == "__main__":
    main()

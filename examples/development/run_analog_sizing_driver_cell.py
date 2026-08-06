"""Drive analog_sizing through the GENERIC driver -- provider-free, then luna.

This is the acceptance run for claim 1 at the driver level: a workload that
implements the five obligations goes through `agent_evolve.driver` with no
bespoke runner.  Everything workload-specific is carried by the kit; this file
composes the kit and reports, and contains no campaign logic of its own.

`--api-key-env` unset runs the identical campaign path with `api_key=None`,
which is free.  Supplying it spends the model allowance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from agent_evolve.agentic import freeze_json  # noqa: E402
from agent_evolve.driver import run_workload_campaign  # noqa: E402
from examples.benchmarks.analog_sizing import create_default_benchmark  # noqa: E402
from examples.benchmarks.analog_sizing.campaign_workload import (  # noqa: E402
    compose_analog_sizing_campaign_workload,
)
from examples.benchmarks.analog_sizing.evaluator import (  # noqa: E402
    AnalogEvaluatorSettings,
    NgspiceSubprocessEvaluator,
)


def _preflight() -> tuple[object, str]:
    """Qualify the simulator the way the evaluator itself does.

    `which ngspice` is the wrong check -- the build is pinned at a fixed root
    and deliberately kept off PATH so a distro package cannot shadow it.  The
    receipt records the version the evaluator actually resolved.
    """

    settings = AnalogEvaluatorSettings()
    evaluator = NgspiceSubprocessEvaluator(settings)
    version = getattr(evaluator, "ngspice_version", "unknown")
    return (
        freeze_json(
            {
                "schema_version": 1,
                "qualification": "ngspice_subprocess_preflight",
                "simulator_path": settings.ngspice,
                "simulator_version": str(version),
            }
        ),
        str(version),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--evaluator-concurrency", type=int, default=2)
    parser.add_argument("--outer-seed", type=int, default=20260802)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    arm = "luna" if api_key else "provider_free"

    receipt, version = _preflight()
    kit = compose_analog_sizing_campaign_workload(
        benchmark=create_default_benchmark(),
        evaluator_preflight_receipt=receipt,
        resource_lease_receipt=freeze_json(
            {
                "schema_version": 1,
                "lease": "local_process",
                "concurrency": args.evaluator_concurrency,
            }
        ),
        evaluator_concurrency_cap=args.evaluator_concurrency,
    )

    started = time.perf_counter()
    run = run_workload_campaign(
        kit,
        generations=args.generations,
        evaluator_concurrency=args.evaluator_concurrency,
        outer_seed=args.outer_seed,
        api_key=api_key,
    )
    wall = time.perf_counter() - started

    reachable = getattr(run, "model_reachable_seats", None)
    evaluated = getattr(run, "evaluated_seats", None)
    share = (
        (reachable / evaluated)
        if isinstance(reachable, int) and isinstance(evaluated, int) and evaluated
        else None
    )
    record = {
        "arm": arm,
        "simulator_version": version,
        "generations": args.generations,
        "outer_seed": args.outer_seed,
        "wall_time_s_outer": wall,
        "wall_time_s_reported": getattr(run, "wall_time_s", None),
        "provider_calls": getattr(run, "provider_calls", None),
        "evaluated_seats": evaluated,
        "model_reachable_seats": reachable,
        "model_reachable_share_of_evaluated_seats": share,
        "final_front_size": len(getattr(run, "final_front", ()) or ()),
    }
    print(json.dumps(record, indent=1, default=str))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

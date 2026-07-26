#!/usr/bin/env python3
"""Airfoil-v7 validation, split qualification, and provider launch CLI.

The default and ``--offline`` modes perform no provider or CFD I/O.  Real
execution is deliberately split into two prospectively frozen phases:

* ``--qualify-seeds`` authorizes exactly two serial CFD candidate evaluations
  and cannot construct a provider stack or read credentials.
* ``--live`` requires a separate provider manifest bound to the completed raw
  seed receipts; it replays those receipts and sends only child requests to
  the provider/CFD routes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.benchmarks.engibench_airfoil.v7_experiment_support import (  # noqa: E402
    run_offline_verification_sync,
    validation_record,
)
from examples.benchmarks.engibench_airfoil.v7_launch import (  # noqa: E402
    SeedQualificationDependencies,
    create_seed_qualification_benchmark,
    execute_live_with_dependencies,
    execute_seed_qualification_with_dependencies,
    production_live_dependencies,
    write_launch_manifest,
    write_seed_qualification_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--build-seed-manifest", type=Path, metavar="PATH")
    modes.add_argument("--qualify-seeds", action="store_true")
    modes.add_argument("--build-launch-manifest", type=Path, metavar="PATH")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument("--launch-manifest", type=Path)
    parser.add_argument("--qualification-result", type=Path)
    parser.add_argument("--verification-report", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    return parser


def _require(parser: argparse.ArgumentParser, value: object, flag: str) -> object:
    if value is None:
        parser.error(f"{flag} is required for the selected mode")
    return value


def _print(record: object) -> None:
    print(json.dumps(record, allow_nan=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.build_seed_manifest is not None:
        record = write_seed_qualification_manifest(
            args.build_seed_manifest,
            run_id=str(_require(parser, args.run_id, "--run-id")),
            output_dir=Path(_require(parser, args.output_dir, "--output-dir")),
            verification_report_path=Path(
                _require(parser, args.verification_report, "--verification-report")
            ),
        )
        _print(record)
        return 0

    if args.qualify_seeds:
        manifest = Path(_require(parser, args.seed_manifest, "--seed-manifest"))
        record = execute_seed_qualification_with_dependencies(
            manifest,
            SeedQualificationDependencies(
                benchmark_factory=create_seed_qualification_benchmark,
            ),
        )
        _print(record)
        return 0

    if args.build_launch_manifest is not None:
        record = write_launch_manifest(
            args.build_launch_manifest,
            run_id=str(_require(parser, args.run_id, "--run-id")),
            output_dir=Path(_require(parser, args.output_dir, "--output-dir")),
            qualification_result_path=Path(
                _require(
                    parser,
                    args.qualification_result,
                    "--qualification-result",
                )
            ),
        )
        _print(record)
        return 0

    if args.live:
        manifest = Path(_require(parser, args.launch_manifest, "--launch-manifest"))
        record = execute_live_with_dependencies(
            manifest,
            production_live_dependencies(),
        )
        _print(record)
        return 0

    unused = {
        "--seed-manifest": args.seed_manifest,
        "--launch-manifest": args.launch_manifest,
        "--qualification-result": args.qualification_result,
        "--verification-report": args.verification_report,
        "--run-id": args.run_id,
        "--output-dir": args.output_dir,
    }
    supplied = [name for name, value in unused.items() if value is not None]
    if supplied:
        parser.error(
            "mode-specific arguments require an explicit execution/build mode: "
            + ", ".join(supplied)
        )
    record = (
        run_offline_verification_sync() if args.offline else validation_record()
    )
    _print(record)
    return 0 if record.get("overall_pass", True) is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

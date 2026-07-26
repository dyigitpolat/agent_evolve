#!/usr/bin/env python3
"""Build, execute, or explicitly resume the provider-free Airfoil-v7 oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.benchmarks.engibench_airfoil.v7_finite_oracle import (  # noqa: E402
    execute_oracle,
    write_oracle_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build-manifest", type=Path, metavar="PATH")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verification-report", type=Path)
    return parser


def _required(parser: argparse.ArgumentParser, value: object, flag: str) -> object:
    if value is None:
        parser.error(f"{flag} is required for the selected mode")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.build_manifest is not None:
        if args.manifest is not None:
            parser.error("--manifest is not accepted with --build-manifest")
        record = write_oracle_manifest(
            args.build_manifest,
            run_id=str(_required(parser, args.run_id, "--run-id")),
            output_dir=Path(_required(parser, args.output_dir, "--output-dir")),
            verification_report_path=Path(
                _required(
                    parser,
                    args.verification_report,
                    "--verification-report",
                )
            ),
        )
    else:
        extras = {
            "--run-id": args.run_id,
            "--output-dir": args.output_dir,
            "--verification-report": args.verification_report,
        }
        supplied = [name for name, value in extras.items() if value is not None]
        if supplied:
            parser.error("execution modes reject build-only arguments: " + ", ".join(supplied))
        record = execute_oracle(
            Path(_required(parser, args.manifest, "--manifest")),
            resume=bool(args.resume),
        )
    print(json.dumps(record, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen provider-free suite and seal its conformance release gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    write_bytes_atomic,
    write_json_atomic,
)
from examples.development import (  # noqa: E402
    run_openrouter_streaming_conformance as conformance,
)


def execute(*, output_dir: Path) -> dict[str, object]:
    """Execute the exact focused suite without credentials or provider access."""

    root = output_dir.expanduser().resolve(strict=False)
    if root.exists():
        raise FileExistsError(root)
    source_before = conformance._source_identity()
    tests_before = conformance._focused_test_source_identity()
    environment = dict(os.environ)
    environment.pop("OPENROUTER_API_KEY", None)
    with tempfile.TemporaryDirectory(
        prefix="agent_evolve_conformance_pytest_"
    ) as temporary:
        junit_temporary = Path(temporary) / "focused_tests.junit.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            *conformance.FOCUSED_PYTEST_ARGUMENTS,
            "--junitxml",
            str(junit_temporary),
        ]
        completed = subprocess.run(
            command,
            cwd=AGENT_EVOLVE_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            timeout=900,
        )
        if completed.returncode != 0:
            sys.stderr.buffer.write(completed.stdout)
            sys.stderr.buffer.write(completed.stderr)
            raise conformance.ConformanceRunError(
                "focused provider-free conformance tests failed"
            )
        junit_payload = junit_temporary.read_bytes()

    root.mkdir(parents=True, exist_ok=False)
    junit_path = root / "focused_tests.junit.xml"
    write_bytes_atomic(junit_path, junit_payload)
    config = conformance.build_config(
        first_event_timeout_seconds=(
            conformance.DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS
        ),
        idle_timeout_seconds=conformance.DEFAULT_IDLE_TIMEOUT_SECONDS,
        absolute_timeout_seconds=conformance.DEFAULT_ABSOLUTE_TIMEOUT_SECONDS,
    )
    gate = conformance.build_provider_free_release_gate(
        config=config,
        junit_report_path=junit_path,
        pytest_exit_code=completed.returncode,
        source_identity_before=source_before,
        focused_test_source_identity_before=tests_before,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    write_json_atomic(root / "release_gate.json", gate)
    finalization = finalize_run_directory(
        root,
        status="provider_free_release_gate_passed",
    )
    return {
        "release_gate_dir": str(root),
        "test_count": gate["test_execution"]["counts"]["tests"],
        "release_gate_commitment_sha256": gate[
            "release_gate_commitment_sha256"
        ],
        "finalization_sha256": finalization["finalization_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = execute(output_dir=args.output_dir)
    except (
        conformance.ConformanceRunError,
        FileExistsError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only CLI for deterministic BOiLS budgeted-v5 post-run analysis.

The live run and sealed oracle are explicit inputs.  The command verifies their
durable finalization bindings before importing any outcomes into the pure
scorer.  It never writes inside either evidence directory and never overwrites
an existing output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from examples.benchmarks.boils_abc.budgeted_v5_analysis import (
    BoilsV5RunAnalysisInput,
    OracleSealExpectation,
    analyze_budgeted_v5_run,
    enumerate_matched_random_portfolios,
    known_local_oracle_v1_expectation,
    parse_sealed_single_edit_oracle,
)


CLI_SCHEMA_ID = "boils_abc_budgeted_v5_offline_analysis_cli_v1"


class BoilsV5AnalysisCliError(RuntimeError):
    """Durable inputs or the requested output location are unsafe."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file():
        raise BoilsV5AnalysisCliError(f"{name} is not a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoilsV5AnalysisCliError(f"{name} is not valid JSON") from exc
    if type(value) is not dict:
        raise BoilsV5AnalysisCliError(f"{name} must contain a JSON object")
    return value, payload


def _finalized_file_record(
    finalized: Mapping[str, object], *, filename: str, name: str
) -> Mapping[str, object]:
    if finalized.get("schema_version") != 1 or finalized.get("status") != "succeeded":
        raise BoilsV5AnalysisCliError(f"{name} is not a successful v1 finalization")
    files = finalized.get("files")
    if not isinstance(files, Mapping):
        raise BoilsV5AnalysisCliError(f"{name} has no file manifest")
    record = files.get(filename)
    if not isinstance(record, Mapping):
        raise BoilsV5AnalysisCliError(f"{name} does not bind {filename}")
    return record


def _verify_bound_payload(
    payload: bytes, record: Mapping[str, object], *, name: str
) -> str:
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if type(expected_bytes) is not int or expected_bytes != len(payload):
        raise BoilsV5AnalysisCliError(f"{name} byte count differs from finalization")
    actual = _sha256(payload)
    if type(expected_sha256) is not str or expected_sha256 != actual:
        raise BoilsV5AnalysisCliError(f"{name} digest differs from finalization")
    return actual


def _load_finalized_run(
    run_dir: Path,
) -> tuple[dict[str, object], str, str]:
    resolved = run_dir.resolve(strict=True)
    if not resolved.is_dir():
        raise BoilsV5AnalysisCliError("run_dir is not a directory")
    finalized, finalized_payload = _read_json_object(
        resolved / "finalized.json", name="run finalized.json"
    )
    if finalized.get("schema_version") != 1 or finalized.get("status") != "succeeded":
        raise BoilsV5AnalysisCliError("run is not successfully finalized")
    files = finalized.get("files")
    if not isinstance(files, Mapping) or "summary.json" not in files:
        raise BoilsV5AnalysisCliError("run finalization does not bind summary.json")
    for filename, raw_record in sorted(files.items()):
        if (
            type(filename) is not str
            or Path(filename).name != filename
            or filename == "finalized.json"
            or not isinstance(raw_record, Mapping)
        ):
            raise BoilsV5AnalysisCliError(
                "run finalization contains an unsafe file entry"
            )
        path = resolved / filename
        if not path.is_file():
            raise BoilsV5AnalysisCliError(f"finalized run file is absent: {filename}")
        payload = path.read_bytes()
        _verify_bound_payload(payload, raw_record, name=f"run file {filename}")
        if filename.endswith(".jsonl") and "lines" in raw_record:
            if raw_record.get("lines") != len(payload.splitlines()):
                raise BoilsV5AnalysisCliError(
                    f"run file {filename} line count differs from finalization"
                )
    summary, summary_payload = _read_json_object(
        resolved / "summary.json", name="run summary.json"
    )
    summary_record = files["summary.json"]
    assert isinstance(summary_record, Mapping)
    summary_sha256 = _verify_bound_payload(
        summary_payload, summary_record, name="run summary.json"
    )
    return summary, summary_sha256, _sha256(finalized_payload)


def _load_finalized_oracle(
    summary_path: Path,
    finalized_path: Path,
) -> tuple[dict[str, object], str, str]:
    summary, summary_payload = _read_json_object(
        summary_path.resolve(strict=True), name="sealed oracle summary"
    )
    finalized, finalized_payload = _read_json_object(
        finalized_path.resolve(strict=True), name="sealed oracle finalization"
    )
    record = _finalized_file_record(
        finalized, filename="summary.json", name="sealed oracle finalization"
    )
    summary_sha256 = _verify_bound_payload(
        summary_payload, record, name="sealed oracle summary"
    )
    return summary, summary_sha256, _sha256(finalized_payload)


def analyze_finalized_v5_run(
    run_dir: Path,
    oracle_summary_path: Path,
    oracle_finalized_path: Path,
    *,
    oracle_expectation: OracleSealExpectation | None = None,
) -> dict[str, object]:
    """Verify durable inputs and return deterministic analysis without writes."""

    summary, run_summary_sha256, run_finalized_sha256 = _load_finalized_run(run_dir)
    if summary.get("development_only") is not True:
        raise BoilsV5AnalysisCliError("run summary is not development-only")
    if summary.get("protocol_acceptance_passed") is not True:
        raise BoilsV5AnalysisCliError(
            "run summary did not pass protocol acceptance; refusing analysis"
        )
    projection = summary.get("offline_analysis_input")
    if not isinstance(projection, Mapping):
        raise BoilsV5AnalysisCliError(
            "run summary lacks the fail-closed offline_analysis_input projection"
        )
    run_input = BoilsV5RunAnalysisInput.from_record(projection)
    if (
        summary.get("protocol_acceptance_passed")
        is not run_input.protocol_acceptance_passed
    ):
        raise BoilsV5AnalysisCliError(
            "run summary and offline projection disagree on protocol acceptance"
        )
    if (
        summary.get("post_hoc_development_protocol_correction") is not True
        or summary.get("protocol_correction")
        != run_input.protocol_correction.to_record()
    ):
        raise BoilsV5AnalysisCliError(
            "run summary and offline projection disagree on correction disclosure"
        )
    oracle_summary, oracle_summary_sha256, oracle_finalized_sha256 = (
        _load_finalized_oracle(oracle_summary_path, oracle_finalized_path)
    )
    expectation = (
        known_local_oracle_v1_expectation()
        if oracle_expectation is None
        else oracle_expectation
    )
    oracle = parse_sealed_single_edit_oracle(
        oracle_summary,
        source_summary_sha256=oracle_summary_sha256,
        expectation=expectation,
    )
    distribution = enumerate_matched_random_portfolios(oracle, run_input.palette_spec)
    analysis = analyze_budgeted_v5_run(run_input, oracle, distribution)
    return {
        "schema_id": CLI_SCHEMA_ID,
        "input_bindings": {
            "run_summary_sha256": run_summary_sha256,
            "run_finalized_sha256": run_finalized_sha256,
            "oracle_summary_sha256": oracle_summary_sha256,
            "oracle_finalized_sha256": oracle_finalized_sha256,
        },
        "analysis": analysis,
    }


def deterministic_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _write_new_output(
    path: Path,
    payload: bytes,
    *,
    protected_directories: Sequence[Path],
) -> None:
    resolved_parent = path.parent.resolve(strict=True)
    resolved = resolved_parent / path.name
    if any(
        _is_within(resolved, directory.resolve(strict=True))
        for directory in protected_directories
    ):
        raise BoilsV5AnalysisCliError(
            "analysis output cannot be created inside an evidence directory"
        )
    if resolved.exists():
        raise BoilsV5AnalysisCliError(
            "analysis output already exists; refusing overwrite"
        )
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(resolved_parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # The file was newly created by this command, so removing a partial
        # output does not mutate any pre-existing evidence.
        resolved.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and score a finalized BOiLS budgeted-v5 development run "
            "against its explicit sealed single-edit oracle."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("oracle_summary", type=Path)
    parser.add_argument("oracle_finalized", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="create a new JSON file outside both evidence directories",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    oracle_expectation: OracleSealExpectation | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    result = analyze_finalized_v5_run(
        arguments.run_dir,
        arguments.oracle_summary,
        arguments.oracle_finalized,
        oracle_expectation=oracle_expectation,
    )
    payload = deterministic_json_bytes(result)
    if arguments.output is None:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    else:
        _write_new_output(
            arguments.output,
            payload,
            protected_directories=(
                arguments.run_dir,
                arguments.oracle_summary.parent,
                arguments.oracle_finalized.parent,
            ),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``.
    raise SystemExit(main())

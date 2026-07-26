#!/usr/bin/env python3
"""Run the frozen no-LLM Airfoil sensitivity/repeatability gate once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from examples.benchmarks.engibench_airfoil.problem_def import (
    AirfoilPanelProblem,
    AirfoilPanelSettings,
    candidate_sha256,
)


GATE_ID = "airfoil_external_panel_v1_sensitivity_repeatability_gate_20260714"
EXPECTED_ORDER = (
    "neutral_repeat",
    "shape_thickness_plus_005",
    "alpha_plus_025",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"candidate manifest SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("gate_id") != GATE_ID:
        raise RuntimeError("candidate manifest gate_id mismatch")
    rows = manifest.get("candidates")
    if not isinstance(rows, list) or tuple(row.get("slot") for row in rows) != EXPECTED_ORDER:
        raise RuntimeError("candidate manifest order mismatch")
    if manifest.get("evaluation_order") != list(EXPECTED_ORDER):
        raise RuntimeError("candidate manifest evaluation_order mismatch")
    for row in rows:
        candidate = json.loads(row["canonical_candidate_json"])
        if candidate_sha256(candidate) != row["candidate_sha256"]:
            raise RuntimeError(f"candidate identity mismatch for {row['slot']}")
        if hashlib.sha256(row["canonical_candidate_json"].encode("utf-8")).hexdigest() != row[
            "candidate_sha256"
        ]:
            raise RuntimeError(f"candidate canonical bytes mismatch for {row['slot']}")
    return manifest, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    manifest, rows = _load_manifest(args.candidate_manifest, args.expected_manifest_sha256)
    args.output_dir.mkdir(parents=True)
    base_settings = AirfoilPanelSettings.local_default()
    settings = AirfoilPanelSettings(
        python_executable=base_settings.python_executable,
        evaluator_script=base_settings.evaluator_script,
        dataset_arrow=base_settings.dataset_arrow,
        output_root=args.output_dir / "evaluator_records",
        work_root=base_settings.work_root / GATE_ID,
        cpu_set=base_settings.cpu_set,
        mpi_cores=base_settings.mpi_cores,
        timeout_seconds=base_settings.timeout_seconds,
        expected_dataset_sha256=base_settings.expected_dataset_sha256,
    )
    problem = AirfoilPanelProblem(settings)
    result_path = args.output_dir / "gate_results.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "status": "running",
        "started_utc": _utc_now(),
        "provider_calls": 0,
        "candidate_manifest": {
            "path": str(args.candidate_manifest.resolve()),
            "sha256": args.expected_manifest_sha256,
        },
        "candidate_order": list(EXPECTED_ORDER),
        "resource_allocation": {
            "cpu_set": settings.cpu_set,
            "mpi_cores": settings.mpi_cores,
            "omp_threads_per_rank": 1,
        },
        "timing_semantics": {
            "outer_problem_wall_seconds": (
                "time.perf_counter around native AirfoilPanelProblem.evaluate_detailed; includes "
                "lazy evaluator setup, subprocess startup, dataset load, decode/validation, "
                "provenance, three RANS calls, durable record publication, and subprocess cleanup"
            ),
            "inner_three_rans_wall_seconds": (
                "evaluator record monotonic interval around the three sequential RANS calls and "
                "per-point cleanup only"
            ),
        },
        "results": [],
    }
    _write_json_atomic(result_path, record)
    try:
        for row in rows:
            candidate = json.loads(row["canonical_candidate_json"])
            outer_started = time.perf_counter()
            evaluation = problem.evaluate_detailed(candidate)
            outer_seconds = time.perf_counter() - outer_started
            evaluator_record_sha256 = _sha256_file(evaluation.record_path)
            result = {
                "slot": row["slot"],
                "candidate_sha256": row["candidate_sha256"],
                "status": "evaluated",
                "objectives": evaluation.objective_values,
                "outer_problem_wall_seconds": outer_seconds,
                "inner_three_rans_wall_seconds": evaluation.wall_seconds,
                "outer_minus_inner_seconds": outer_seconds - evaluation.wall_seconds,
                "evaluator_calls": evaluation.record["evaluator_calls"],
                "points": evaluation.record["points"],
                "decoder_audit": evaluation.record["decoder_audit"],
                "task_sha256": evaluation.record["task_sha256"],
                "evaluator_record": {
                    "path": str(evaluation.record_path.resolve()),
                    "sha256": evaluator_record_sha256,
                },
            }
            record["results"].append(result)
            _write_json_atomic(result_path, record)
    except Exception as exc:
        record.update(
            {
                "status": "failed_stop_no_retry_no_replacement",
                "finished_utc": _utc_now(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json_atomic(result_path, record)
        raise
    record.update(
        {
            "status": "completed",
            "finished_utc": _utc_now(),
            "candidate_evaluations": len(record["results"]),
            "physical_rans_calls": sum(item["evaluator_calls"] for item in record["results"]),
        }
    )
    _write_json_atomic(result_path, record)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "status": record["status"],
                "candidate_evaluations": record["candidate_evaluations"],
                "physical_rans_calls": record["physical_rans_calls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

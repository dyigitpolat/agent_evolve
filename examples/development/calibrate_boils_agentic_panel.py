#!/usr/bin/env python3
"""Calibrate the real length-20 BOiLS development panel without an LLM.

This runner exists only to select a 10--30 second development fidelity. It
records the complete pinned evaluator result and does not establish benchmark
quality, a baseline, or a method comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.benchmarks.boils_abc.actions import (  # noqa: E402
    CandidateConfig,
    config_sha256,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    CURRENT_CIRCUIT_NAMES,
)


DEFAULT_PANEL = ("multiplier", "sin", "sqrt")
DEFAULT_LOG_ROOT = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "boils_agentic_development"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _semantic_projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration_sha256": result["configuration_sha256"],
        "sequence": result["sequence"],
        "abc_binary_sha256": result["abc_binary_sha256"],
        "lut_inputs": result["lut_inputs"],
        "total_lut_count": result["total_lut_count"],
        "total_levels": result["total_levels"],
        "max_levels": result["max_levels"],
        "circuits": [
            {
                "name": item["circuit_name"],
                "sha256": item["circuit_sha256"],
                "lut_count": item["lut_count"],
                "levels": item["levels"],
                "equivalent": item["diagnostics"]["equivalent"],
                "status": item["diagnostics"]["status"],
            }
            for item in result["circuit_results"]
        ],
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "actions": AGENT_EVOLVE_ROOT
        / "examples"
        / "benchmarks"
        / "boils_abc"
        / "actions.py",
        "evaluator": AGENT_EVOLVE_ROOT
        / "examples"
        / "benchmarks"
        / "boils_abc"
        / "evaluator.py",
        "problem": AGENT_EVOLVE_ROOT
        / "examples"
        / "benchmarks"
        / "boils_abc"
        / "problem_def.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _finalize(run_dir: Path, status: str) -> None:
    files: dict[str, dict[str, object]] = {}
    for name in (
        "manifest.json",
        "runner_source.py",
        "evaluations.jsonl",
        "summary.json",
        "failure.json",
    ):
        path = run_dir / name
        if not path.exists():
            continue
        payload = path.read_bytes()
        record: dict[str, object] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if name.endswith(".jsonl"):
            record["lines"] = len(payload.splitlines())
        files[name] = record
    _write_json(
        run_dir / "finalized.json",
        {
            "schema_version": 1,
            "status": status,
            "completed_at_utc": _utc_now(),
            "files": files,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="length20_panel_calibration_20260713")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--circuits", default=",".join(DEFAULT_PANEL))
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--per-circuit-timeout-s", type=float, default=60.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.repetitions <= 0:
        raise SystemExit("repetitions must be positive")
    if args.cpu < 0:
        raise SystemExit("cpu must be non-negative")
    panel = tuple(part.strip() for part in args.circuits.split(",") if part.strip())
    if not panel or len(set(panel)) != len(panel):
        raise SystemExit("circuits must be a non-empty duplicate-free CSV")
    unknown = tuple(sorted(set(panel).difference(CURRENT_CIRCUIT_NAMES)))
    if unknown:
        raise SystemExit(
            f"unknown circuits {unknown!r}; allowed={CURRENT_CIRCUIT_NAMES!r}"
        )
    run_dir = args.log_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    runner_source = Path(__file__).resolve().read_bytes()
    with (run_dir / "runner_source.py").open("xb") as stream:
        stream.write(runner_source)
        stream.flush()
        os.fsync(stream.fileno())

    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=panel,
        affinity_sets=((args.cpu,),),
        per_circuit_timeout_s=args.per_circuit_timeout_s,
    )
    evaluator = BoilsAbcEvaluator(settings)
    candidate = CandidateConfig()
    expected_config_hash = config_sha256(candidate)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "started_at_utc": _utc_now(),
        "development_only": True,
        "claim_boundary": (
            "Length-20 evaluator fidelity calibration only; not a benchmark, "
            "baseline, optimizer, SOTA, or wall-clock comparison."
        ),
        "panel": list(panel),
        "repetitions": args.repetitions,
        "candidate": candidate.model_dump(mode="json"),
        "configuration_sha256": expected_config_hash,
        "evaluator_provenance": evaluator.provenance(),
        "source_sha256": _source_hashes(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "process_affinity_at_start": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)

    semantic_hashes: list[str] = []
    elapsed: list[float] = []
    circuit_elapsed: dict[str, list[float]] = {name: [] for name in panel}
    evaluations_path = run_dir / "evaluations.jsonl"
    try:
        with evaluations_path.open("xb") as stream:
            for repetition in range(args.repetitions):
                started_at = _utc_now()
                detailed = evaluator.evaluate(candidate)
                result = detailed.as_dict()
                if result["configuration_sha256"] != expected_config_hash:
                    raise RuntimeError("evaluator returned a different configuration hash")
                semantic = _semantic_projection(result)
                semantic_hash = hashlib.sha256(
                    _canonical_json(semantic).encode("ascii")
                ).hexdigest()
                record = {
                    "schema_version": 1,
                    "repetition": repetition,
                    "started_at_utc": started_at,
                    "completed_at_utc": _utc_now(),
                    "semantic_result_sha256": semantic_hash,
                    "semantic_result": semantic,
                    "detailed_result": result,
                }
                stream.write((_canonical_json(record) + "\n").encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
                semantic_hashes.append(semantic_hash)
                elapsed.append(float(detailed.elapsed_s))
                for item in detailed.circuit_results:
                    circuit_elapsed[item.circuit_name].append(
                        float(item.diagnostics.elapsed_s)
                    )

        summary = {
            "schema_version": 1,
            "status": "succeeded",
            "completed_at_utc": _utc_now(),
            "configuration_sha256": expected_config_hash,
            "panel": list(panel),
            "repetitions": args.repetitions,
            "all_semantic_results_identical": len(set(semantic_hashes)) == 1,
            "semantic_result_sha256": semantic_hashes[0],
            "elapsed_s": {
                "values": elapsed,
                "min": min(elapsed),
                "median": statistics.median(elapsed),
                "max": max(elapsed),
                "mean": statistics.fmean(elapsed),
            },
            "per_circuit_elapsed_s": {
                name: {
                    "values": values,
                    "median": statistics.median(values),
                }
                for name, values in circuit_elapsed.items()
            },
            "target_latency_band_s": [10.0, 30.0],
            "median_within_target_latency_band": (
                10.0 <= statistics.median(elapsed) <= 30.0
            ),
        }
        _write_json(run_dir / "summary.json", summary)
    except BaseException as exc:
        _write_json(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "failure_type": type(exc).__name__,
                "safe_message": "BOiLS development calibration failed",
            },
        )
        _finalize(run_dir, "failed")
        raise
    _finalize(run_dir, "succeeded")
    print(_canonical_json({"run_dir": str(run_dir), "status": "succeeded"}))


if __name__ == "__main__":
    main()

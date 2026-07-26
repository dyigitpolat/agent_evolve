#!/usr/bin/env python3
"""Run the frozen engine-only continuation of the BOiLS recombination cube.

This continuation consumes the permanently failed v3 record, reuses its exact
fresh-C observation, and evaluates only the three previously unseen engine
children AD, BD, and ABD.  It makes no LLM call and supports no model result.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
)
from examples.development import run_agentic_probe as support  # noqa: E402
from examples.development import run_boils_agentic_pilot as v1  # noqa: E402
from examples.development import run_boils_recombination_v3 as v3  # noqa: E402


RUN_ID = "boils_recombination_engine_v4_20260714"
ENGINE_CPUS = (8, 9, 11)
CHILD_WORKERS = 3
PER_CANDIDATE_TIMEOUT_SECONDS = 60
QUALITY_HORIZON_SECONDS = 60
HARD_CLEANUP_DEADLINE_SECONDS = 180
CPU_ADMISSION_WINDOWS = 3
CPU_ADMISSION_INTERVAL_SECONDS = 1.0
MAX_CPU_BUSY_FRACTION = 0.10
REFERENCE_POINT = (8_028, 71)
NEW_ARM_ORDER = ("AD", "BD", "ABD")
ALL_ARM_ORDER = v3.ALL_ARM_ORDER
EXPECTED_PHYSICAL_EVALUATIONS = 3

EXPECTED_ABC_SHA256 = v3.EXPECTED_ABC_SHA256
EXPECTED_CIRCUIT_SHA256 = v3.EXPECTED_CIRCUIT_SHA256
EXPECTED_SEED_OBJECTIVES = v3.EXPECTED_SEED_OBJECTIVES

ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
PREREGISTRATION_PATH = (
    ARTIFACT_ROOT
    / "70_boils_recombination_failure_and_engine_continuation_preregistration.md"
)
EXPECTED_PREREGISTRATION_BYTES = 7_638
EXPECTED_PREREGISTRATION_SHA256 = (
    "be9d29a594f96295078678d9c347fe1ae68195f7bdec10ff1dfd70b9af4e1ace"
)
DEVELOPMENT_LOG_ROOT = ARTIFACT_ROOT / "experiment_logs" / "boils_agentic_development"
FAILED_V3_RUN_DIR = DEVELOPMENT_LOG_ROOT / "boils_recombination_v3_20260714"
DEFAULT_LOG_ROOT = DEVELOPMENT_LOG_ROOT

# Historical absence claims must be replayed against the evidence census that
# existed when the v4 block was frozen.  Scanning ARTIFACT_ROOT dynamically made
# the verifier self-invalidating: later, correctly published v5 evaluations of
# AD/BD/ABD caused an earlier preregistration fact to fail.  A caller supplying a
# different root still receives the adversarial all-files scan used by tests.
PRE_V4_EVALUATION_LOG_CENSUS = (
    "experiment_logs/boils_agentic_development/boils_agentic_pilot_v1_20260713/evaluations.jsonl",
    "experiment_logs/boils_agentic_development/boils_local_oracle_v1_20260714/evaluations.jsonl",
    "experiment_logs/boils_agentic_development/boils_patch_native_pilot_v2_20260713/evaluations.jsonl",
    "experiment_logs/boils_agentic_development/boils_recombination_v3_20260714/evaluations.jsonl",
    "experiment_logs/boils_agentic_development/boils_seed_preflight_v1_20260713/evaluations.jsonl",
    "experiment_logs/boils_agentic_development/length20_panel_calibration_v1_20260713/evaluations.jsonl",
    "experiment_logs/boils_agentic_development/length20_panel_calibration_v2_20260713/evaluations.jsonl",
    "experiment_logs/boils_agentic_development/length20_panel_calibration_v3_log2_20260713/evaluations.jsonl",
)

FAILED_V3_FILES: dict[str, str] = {
    "finalized.json": "40466ef7958014085c25b1a771ee5d592873ee4d9ac8d2d5bee4ee849935b453",
    "manifest.json": "601ed1185cabd02400b17d844fbd6579843a2a27b0b010b138c2bb6f2f844866",
    "failure.json": "63bddb92d6a704b8068acfb4da6eab9c2d5aa1df5f1de9b64975aad3167346e6",
    "evaluations.jsonl": "4fb6269b27c86e6b57d81539314c6d410e6c5ec5235de3554b81467b622d6d38",
    "events.jsonl": "1172d709ef2723c859d2b148b694030b2ca36ac12d17e567767c88df23da6b84",
    "queue_outcomes.jsonl": "94f292ada522da1b95b04dd559ec614e30a1550781ed54458d11ce15df413002",
}

CUBE = v3.CUBE
CUBE_BY_LABEL = v3.CUBE_BY_LABEL
CHILD_SCHEDULE = tuple(
    v3.oracle.CandidateSpec(
        label=label,
        frozen_order=order,
        sequence=CUBE_BY_LABEL[label].sequence,
        boils_configuration_sha256=CUBE_BY_LABEL[label].boils_configuration_sha256,
        typed_json_configuration_sha256=CUBE_BY_LABEL[
            label
        ].typed_json_configuration_sha256,
    )
    for order, label in enumerate(NEW_ARM_ORDER)
)

EXPECTED_SEALED_C_PROJECTION: dict[str, object] = {
    "arm": "C",
    "source": "sealed_failed_v3_fresh_parent",
    "boils_configuration_sha256": CUBE_BY_LABEL["C"].boils_configuration_sha256,
    "typed_json_configuration_sha256": CUBE_BY_LABEL[
        "C"
    ].typed_json_configuration_sha256,
    "objectives": {"total_lut_count": 7_944, "total_levels": 69},
    "cec_passed": True,
    "cpu_affinity": [8],
    "publication_sequence": 1,
    "published_elapsed_ns": 19_154_075_548,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return support._canonical_json(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"sealed JSONL is malformed at {path.name}:{line_number}"
            ) from exc
        if type(value) is not dict:
            raise RuntimeError(f"sealed JSONL row is not an object: {path.name}")
        rows.append(value)
    return rows


def verify_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError("engine-v4 preregistration is missing")
    payload = path.read_bytes()
    if (
        len(payload) != EXPECTED_PREREGISTRATION_BYTES
        or _sha256_bytes(payload) != EXPECTED_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("engine-v4 preregistration identity changed")
    return {
        "source": str(path),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _parse_exact_v3_c(row: Mapping[str, object]) -> dict[str, object]:
    candidate = row.get("candidate")
    observation = row.get("observation")
    if row.get("status") != "succeeded" or type(candidate) is not dict:
        raise RuntimeError("failed-v3 evaluation is not one successful C record")
    if type(observation) is not dict:
        raise RuntimeError("failed-v3 C observation is missing")
    expected = CUBE_BY_LABEL["C"]
    if (
        candidate.get("label") != "C"
        or candidate.get("boils_configuration_sha256")
        != expected.boils_configuration_sha256
        or candidate.get("typed_json_configuration_sha256")
        != expected.typed_json_configuration_sha256
        or observation.get("configuration_sha256")
        != expected.boils_configuration_sha256
        or observation.get("sequence") != list(expected.sequence)
        or observation.get("abc_binary_sha256") != EXPECTED_ABC_SHA256
        or observation.get("lut_inputs") != 6
        or observation.get("cpu_affinity") != [8]
    ):
        raise RuntimeError("failed-v3 C identity/provenance/affinity changed")
    objectives = (
        v3.oracle._as_exact_int(observation.get("total_lut_count"), "v3 C LUT"),
        v3.oracle._as_exact_int(observation.get("total_levels"), "v3 C levels"),
    )
    circuits = observation.get("circuit_results")
    if objectives != EXPECTED_SEED_OBJECTIVES or not (
        type(circuits) is list
        and len(circuits) == 1
        and circuits[0].get("circuit_name") == "log2"
        and circuits[0].get("circuit_sha256") == EXPECTED_CIRCUIT_SHA256
        and type(circuits[0].get("diagnostics")) is dict
        and circuits[0]["diagnostics"].get("status") == "passed"
        and circuits[0]["diagnostics"].get("equivalent") is True
        and circuits[0]["diagnostics"].get("cpu_affinity") == [8]
    ):
        raise RuntimeError("failed-v3 C objective or mandatory CEC changed")
    projection = {
        "arm": "C",
        "source": "sealed_failed_v3_fresh_parent",
        "boils_configuration_sha256": expected.boils_configuration_sha256,
        "typed_json_configuration_sha256": expected.typed_json_configuration_sha256,
        "objectives": {
            "total_lut_count": objectives[0],
            "total_levels": objectives[1],
        },
        "cec_passed": True,
        "cpu_affinity": [8],
        "publication_sequence": row.get("publication_sequence"),
        "published_elapsed_ns": row.get("published_elapsed_ns"),
    }
    if projection != EXPECTED_SEALED_C_PROJECTION:
        raise RuntimeError("failed-v3 sealed C projection changed")
    return projection


def scan_unseen_children(
    root: Path = ARTIFACT_ROOT,
) -> dict[str, object]:
    """Replay the pre-v4 absence claim over its frozen evidence census.

    A non-default ``root`` intentionally scans every nested evaluation log so
    adversarial fixtures and prospective callers retain the generic guard.
    """

    expected_hashes = {
        CUBE_BY_LABEL[label].boils_configuration_sha256 for label in NEW_ARM_ORDER
    }
    scanned: list[dict[str, object]] = []
    hits: list[dict[str, object]] = []
    if root.resolve() == ARTIFACT_ROOT.resolve():
        paths = tuple(root / relative for relative in PRE_V4_EVALUATION_LOG_CENSUS)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError("the frozen pre-v4 evaluation census is incomplete")
        scan_scope = "frozen_pre_v4_evaluation_census"
    else:
        paths = tuple(sorted(root.rglob("evaluations.jsonl")))
        scan_scope = "caller_supplied_recursive_root"
    for path in paths:
        payload = path.read_bytes()
        relative = str(path.relative_to(root))
        scanned.append(
            {
                "path": relative,
                "bytes": len(payload),
                "lines": len(payload.splitlines()),
                "sha256": _sha256_bytes(payload),
            }
        )
        for digest in sorted(expected_hashes):
            if digest.encode("ascii") in payload:
                hits.append({"path": relative, "boils_configuration_sha256": digest})
    if hits:
        raise RuntimeError("AD/BD/ABD is no longer an unseen physical arm set")
    return {
        "root": str(root),
        "scan_scope": scan_scope,
        "frozen_relative_paths": (
            list(PRE_V4_EVALUATION_LOG_CENSUS)
            if scan_scope == "frozen_pre_v4_evaluation_census"
            else None
        ),
        "evaluations_logs_scanned": len(scanned),
        "files": scanned,
        "target_hashes": sorted(expected_hashes),
        "hits": [],
        "all_three_unseen": True,
    }


def verify_failed_v3_bundle(
    run_dir: Path = FAILED_V3_RUN_DIR,
    *,
    evaluations_root: Path = ARTIFACT_ROOT,
) -> dict[str, object]:
    """Hash- and grammar-bind the terminal v3 failure and its sole C result."""

    files: dict[str, dict[str, object]] = {}
    for name, expected_hash in FAILED_V3_FILES.items():
        path = run_dir / name
        if not path.is_file():
            raise RuntimeError(f"failed-v3 source is missing: {name}")
        payload = path.read_bytes()
        observed_hash = _sha256_bytes(payload)
        if observed_hash != expected_hash:
            raise RuntimeError(f"failed-v3 source hash changed: {name}")
        files[name] = {
            "source": str(path),
            "bytes": len(payload),
            "sha256": observed_hash,
            **({"lines": len(payload.splitlines())} if name.endswith(".jsonl") else {}),
        }

    finalized = json.loads((run_dir / "finalized.json").read_text(encoding="utf-8"))
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    if finalized.get("status") != "failed" or failure.get("status") != "failed":
        raise RuntimeError("v3 is not durably closed as failed")
    finalized_files = finalized.get("files")
    if type(finalized_files) is not dict:
        raise RuntimeError("failed-v3 terminal index lacks its file table")
    for name in FAILED_V3_FILES:
        if name == "finalized.json":
            continue
        indexed = finalized_files.get(name)
        if type(indexed) is not dict or indexed.get("sha256") != FAILED_V3_FILES[name]:
            raise RuntimeError(f"failed-v3 finalized index changed: {name}")

    evaluations = _read_json_lines(run_dir / "evaluations.jsonl")
    events = _read_json_lines(run_dir / "events.jsonl")
    queue = _read_json_lines(run_dir / "queue_outcomes.jsonl")
    if len(evaluations) != 1:
        raise RuntimeError("failed v3 must contain exactly one physical evaluation")
    sealed_c = _parse_exact_v3_c(evaluations[0])
    event_types = [str(row.get("event_type")) for row in events]
    required_grammar = [
        "recombination_block_started",
        "candidate_submitted",
        "evaluation_published",
        "fresh_seed_gate_passed",
        "prediction_requested",
    ]
    submitted = [
        row.get("arm")
        for row in events
        if row.get("event_type") == "candidate_submitted"
    ]
    if (
        event_types != required_grammar
        or submitted != ["C"]
        or "prediction_completed" in event_types
        or any(label in submitted for label in NEW_ARM_ORDER)
    ):
        raise RuntimeError("failed-v3 event grammar no longer closes before children")
    if not (
        len(queue) == 1
        and queue[0].get("status") == "succeeded"
        and type(queue[0].get("attempts")) is list
        and len(queue[0]["attempts"]) == 1
    ):
        raise RuntimeError("failed-v3 provider queue fact changed")

    return {
        "run_id": run_dir.name,
        "status": "failed",
        "files": files,
        "semantic_closure": {
            "physical_evaluations": 1,
            "only_arm": "C",
            "event_grammar": event_types,
            "prediction_requested": True,
            "prediction_completed": False,
            "new_child_submissions": 0,
            "provider_result_or_calibration_available": False,
        },
        "sealed_c": sealed_c,
        "unseen_child_scan": scan_unseen_children(evaluations_root),
    }


@dataclass(frozen=True, slots=True)
class CpuTickRow:
    cpu: int
    counters: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.counters)

    @property
    def idle(self) -> int:
        return self.counters[3] + self.counters[4]

    def as_dict(self) -> dict[str, object]:
        return {
            "cpu": self.cpu,
            "counters": list(self.counters),
            "total": self.total,
            "idle_plus_iowait": self.idle,
        }


def parse_proc_stat(
    text: str, cpus: Sequence[int] = ENGINE_CPUS
) -> dict[int, CpuTickRow]:
    if type(text) is not str:
        raise TypeError("/proc/stat reader must return text")
    wanted = tuple(cpus)
    if len(wanted) != len(set(wanted)) or any(type(cpu) is not int for cpu in wanted):
        raise ValueError("selected CPU set must contain distinct exact integers")
    parsed: dict[int, CpuTickRow] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        suffix = fields[0][3:]
        if not suffix.isdigit() or int(suffix) not in wanted:
            continue
        cpu = int(suffix)
        if cpu in parsed or len(fields) < 6:
            raise RuntimeError(f"/proc/stat has a malformed or duplicate cpu{cpu} row")
        try:
            counters = tuple(int(value) for value in fields[1:])
        except ValueError as exc:
            raise RuntimeError(
                f"/proc/stat cpu{cpu} has a non-integer counter"
            ) from exc
        if any(value < 0 for value in counters):
            raise RuntimeError(f"/proc/stat cpu{cpu} has a negative counter")
        parsed[cpu] = CpuTickRow(cpu=cpu, counters=counters)
    missing = [cpu for cpu in wanted if cpu not in parsed]
    if missing:
        raise RuntimeError(f"/proc/stat is missing selected CPUs: {missing}")
    return {cpu: parsed[cpu] for cpu in wanted}


def _read_proc_stat() -> str:
    return Path("/proc/stat").read_text(encoding="utf-8")


def sample_cpu_admission(
    *,
    reader: Callable[[], str] = _read_proc_stat,
    sleeper: Callable[[float], object] = time.sleep,
    cpus: Sequence[int] = ENGINE_CPUS,
    windows: int = CPU_ADMISSION_WINDOWS,
    interval_seconds: float = CPU_ADMISSION_INTERVAL_SECONDS,
    max_busy_fraction: float = MAX_CPU_BUSY_FRACTION,
) -> dict[str, object]:
    """Measure three consecutive CPU-idle windows without mutating run state."""

    if windows != CPU_ADMISSION_WINDOWS or interval_seconds != 1.0:
        raise ValueError("engine-v4 admission freezes three one-second windows")
    if not math.isfinite(max_busy_fraction) or max_busy_fraction != 0.10:
        raise ValueError("engine-v4 admission freezes a 0.10 busy ceiling")
    samples = [parse_proc_stat(reader(), cpus)]
    window_records: list[dict[str, object]] = []
    for window_index in range(windows):
        sleeper(interval_seconds)
        current = parse_proc_stat(reader(), cpus)
        previous = samples[-1]
        per_cpu: list[dict[str, object]] = []
        for cpu in cpus:
            before = previous[cpu]
            after = current[cpu]
            if len(before.counters) != len(after.counters):
                raise RuntimeError(f"/proc/stat cpu{cpu} counter width changed")
            deltas = tuple(
                new - old
                for old, new in zip(before.counters, after.counters, strict=True)
            )
            if any(delta < 0 for delta in deltas):
                raise RuntimeError(f"/proc/stat cpu{cpu} counters regressed")
            total_delta = sum(deltas)
            idle_delta = deltas[3] + deltas[4]
            if total_delta <= 0:
                raise RuntimeError(f"/proc/stat cpu{cpu} total delta is not positive")
            busy_delta = total_delta - idle_delta
            busy_fraction = busy_delta / total_delta
            # Compare the frozen 10% boundary in integer arithmetic so an
            # exactly-on-threshold sample cannot fail through float rounding.
            if (
                busy_delta < 0
                or not math.isfinite(busy_fraction)
                or busy_delta * 10 > total_delta
            ):
                raise RuntimeError(
                    f"cpu{cpu} failed idle admission in window {window_index + 1}: "
                    f"busy_fraction={busy_fraction:.6f}"
                )
            per_cpu.append(
                {
                    "cpu": cpu,
                    "counter_deltas": list(deltas),
                    "total_delta": total_delta,
                    "idle_delta": idle_delta,
                    "busy_delta": busy_delta,
                    "busy_fraction": busy_fraction,
                    "passed": True,
                }
            )
        window_records.append(
            {
                "window": window_index + 1,
                "interval_seconds": interval_seconds,
                "cpus": per_cpu,
                "passed": True,
            }
        )
        samples.append(current)
    return {
        "schema_version": 1,
        "source": "/proc/stat",
        "selected_cpus": list(cpus),
        "window_count": windows,
        "interval_seconds": interval_seconds,
        "max_busy_fraction": max_busy_fraction,
        "parser_rule": "sum all reported counters; idle is idle+iowait",
        "samples": [
            {str(cpu): rows[cpu].as_dict() for cpu in cpus} for rows in samples
        ],
        "windows": window_records,
        "passed": True,
    }


def prepare_launch_admission(
    *,
    run_dir: Path,
    preregistration_loader: Callable[[], Mapping[str, object]] = verify_preregistration,
    failed_v3_loader: Callable[[], Mapping[str, object]] = verify_failed_v3_bundle,
    cpu_reader: Callable[[], str] = _read_proc_stat,
    sleeper: Callable[[float], object] = time.sleep,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Perform every read-only launch gate before a run directory may exist."""

    if run_dir.exists():
        raise RuntimeError("engine-v4 run directory already exists")
    preregistration = dict(preregistration_loader())
    failed_v3 = dict(failed_v3_loader())
    _validate_bound_inputs(failed_v3=failed_v3, preregistration=preregistration)
    admission = sample_cpu_admission(reader=cpu_reader, sleeper=sleeper)
    _validate_admission_record(admission)
    if run_dir.exists():
        raise RuntimeError("a launch gate unexpectedly mutated the run directory")
    return preregistration, failed_v3, admission


def _validate_admission_record(admission: Mapping[str, object]) -> None:
    windows = admission.get("windows")
    samples = admission.get("samples")
    if not (
        admission.get("passed") is True
        and admission.get("selected_cpus") == list(ENGINE_CPUS)
        and admission.get("window_count") == CPU_ADMISSION_WINDOWS
        and admission.get("interval_seconds") == CPU_ADMISSION_INTERVAL_SECONDS
        and admission.get("max_busy_fraction") == MAX_CPU_BUSY_FRACTION
        and type(windows) is list
        and len(windows) == CPU_ADMISSION_WINDOWS
        and type(samples) is list
        and len(samples) == CPU_ADMISSION_WINDOWS + 1
        and all(type(sample) is dict for sample in samples)
    ):
        raise RuntimeError("engine-v4 CPU admission record is invalid")
    for window_index, window in enumerate(windows):
        rows = window.get("cpus") if type(window) is dict else None
        if not (
            window.get("window") == window_index + 1
            and window.get("interval_seconds") == CPU_ADMISSION_INTERVAL_SECONDS
            and window.get("passed") is True
            and type(rows) is list
            and [row.get("cpu") for row in rows] == list(ENGINE_CPUS)
        ):
            raise RuntimeError("engine-v4 CPU admission window is invalid")
        for row in rows:
            cpu = row["cpu"]
            before = samples[window_index].get(str(cpu))
            after = samples[window_index + 1].get(str(cpu))
            if type(before) is not dict or type(after) is not dict:
                raise RuntimeError("engine-v4 CPU admission sample is invalid")
            before_counters = before.get("counters")
            after_counters = after.get("counters")
            if not (
                type(before_counters) is list
                and type(after_counters) is list
                and len(before_counters) == len(after_counters)
                and len(before_counters) >= 5
                and all(type(value) is int for value in before_counters)
                and all(type(value) is int for value in after_counters)
                and before.get("cpu") == cpu
                and after.get("cpu") == cpu
                and before.get("total") == sum(before_counters)
                and after.get("total") == sum(after_counters)
                and before.get("idle_plus_iowait")
                == before_counters[3] + before_counters[4]
                and after.get("idle_plus_iowait")
                == after_counters[3] + after_counters[4]
            ):
                raise RuntimeError("engine-v4 CPU admission counters are invalid")
            deltas = [
                new - old
                for old, new in zip(before_counters, after_counters, strict=True)
            ]
            total_delta = sum(deltas)
            idle_delta = deltas[3] + deltas[4]
            busy_delta = total_delta - idle_delta
            busy_fraction = busy_delta / total_delta if total_delta > 0 else math.inf
            if not (
                row.get("passed") is True
                and row.get("counter_deltas") == deltas
                and row.get("total_delta") == total_delta
                and row.get("idle_delta") == idle_delta
                and row.get("busy_delta") == busy_delta
                and row.get("busy_fraction") == busy_fraction
                and total_delta > 0
                and busy_delta >= 0
                and math.isfinite(busy_fraction)
                and busy_delta * 10 <= total_delta
            ):
                raise RuntimeError("engine-v4 CPU admission row is invalid")


def _validate_bound_inputs(
    *,
    failed_v3: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> None:
    if preregistration != {
        "source": str(PREREGISTRATION_PATH),
        "bytes": EXPECTED_PREREGISTRATION_BYTES,
        "sha256": EXPECTED_PREREGISTRATION_SHA256,
    }:
        raise RuntimeError("engine-v4 preregistration was not exactly bound")
    files = failed_v3.get("files")
    closure = failed_v3.get("semantic_closure")
    scan = failed_v3.get("unseen_child_scan")
    if not (
        failed_v3.get("run_id") == FAILED_V3_RUN_DIR.name
        and failed_v3.get("status") == "failed"
        and type(files) is dict
        and all(
            type(files.get(name)) is dict and files[name].get("sha256") == expected_hash
            for name, expected_hash in FAILED_V3_FILES.items()
        )
        and type(closure) is dict
        and closure.get("physical_evaluations") == 1
        and closure.get("only_arm") == "C"
        and closure.get("prediction_requested") is True
        and closure.get("prediction_completed") is False
        and closure.get("new_child_submissions") == 0
        and closure.get("provider_result_or_calibration_available") is False
        and closure.get("event_grammar")
        == [
            "recombination_block_started",
            "candidate_submitted",
            "evaluation_published",
            "fresh_seed_gate_passed",
            "prediction_requested",
        ]
        and type(scan) is dict
        and scan.get("all_three_unseen") is True
        and scan.get("hits") == []
        and scan.get("target_hashes")
        == sorted(
            CUBE_BY_LABEL[label].boils_configuration_sha256 for label in NEW_ARM_ORDER
        )
        and type(scan.get("files")) is list
        and scan.get("evaluations_logs_scanned") == len(scan["files"])
        and scan.get("evaluations_logs_scanned", 0) > 0
    ):
        raise RuntimeError("failed-v3 continuation bundle was not exactly bound")
    if failed_v3.get("sealed_c") != EXPECTED_SEALED_C_PROJECTION:
        raise RuntimeError("failed-v3 sealed C projection was not exactly bound")


def _provenance_is_exact(provenance: object) -> bool:
    circuits = provenance.get("circuits") if type(provenance) is dict else None
    return bool(
        type(provenance) is dict
        and provenance.get("abc_binary_sha256") == EXPECTED_ABC_SHA256
        and type(circuits) is list
        and len(circuits) == 1
        and circuits[0].get("name") == "log2"
        and circuits[0].get("sha256") == EXPECTED_CIRCUIT_SHA256
        and provenance.get("lut_inputs") == 6
        and provenance.get("per_circuit_timeout_s")
        == float(PER_CANDIDATE_TIMEOUT_SECONDS)
        and provenance.get("affinity_sets") == [[cpu] for cpu in ENGINE_CPUS]
    )


def _assert_evaluator_provenance(evaluator: object) -> dict[str, object]:
    provenance_method = getattr(evaluator, "provenance", None)
    if not callable(provenance_method):
        raise RuntimeError("evaluator does not expose provenance")
    provenance = provenance_method()
    if not _provenance_is_exact(provenance):
        raise RuntimeError("engine-v4 evaluator provenance gate failed")
    return copy.deepcopy(provenance)


def _validate_deferred_oracle(deferred_oracle: Mapping[str, object]) -> None:
    expected_hashes = {
        name: v3.EVIDENCE_SOURCES[name][1]
        for name in ("oracle_finalized", "oracle_summary")
    }
    if not (
        deferred_oracle.get("verified") is True
        and deferred_oracle.get("source_sha256") == expected_hashes
        and deferred_oracle.get("confirmed_arms") == ["C", "A", "B", "D"]
        and type(deferred_oracle.get("full_oracle_sensitivity")) is dict
        and deferred_oracle["full_oracle_sensitivity"].get("hypervolume") == 700
    ):
        raise RuntimeError("deferred local-oracle bundle was not exactly bound")


def _objective_tuple(row: Mapping[str, object]) -> tuple[int, int]:
    return v3._objective_tuple(row)


def _physical_identity_gate(outcomes: Sequence[Mapping[str, object]]) -> None:
    if len(outcomes) != EXPECTED_PHYSICAL_EVALUATIONS:
        raise RuntimeError("engine-v4 must consume exactly three physical outcomes")
    for expected_label, outcome in zip(NEW_ARM_ORDER, outcomes, strict=True):
        arm = CUBE_BY_LABEL[expected_label]
        if (
            outcome.get("label") != expected_label
            or outcome.get("boils_configuration_sha256")
            != arm.boils_configuration_sha256
            or outcome.get("typed_json_configuration_sha256")
            != arm.typed_json_configuration_sha256
        ):
            raise RuntimeError("engine-v4 physical identity gate failed")


def _interaction(
    values: Mapping[str, tuple[int, int]], terms: Sequence[tuple[int, str]]
) -> tuple[int, int]:
    return tuple(
        sum(coefficient * values[label][objective] for coefficient, label in terms)
        for objective in range(2)
    )


def analyze_engine_cube(
    *,
    child_outcomes: Sequence[Mapping[str, object]],
    sealed_c: Mapping[str, object],
    deferred_oracle: Mapping[str, object],
    evaluator_provenance: Mapping[str, object],
    failed_v3: Mapping[str, object],
    preregistration: Mapping[str, object],
    admission: Mapping[str, object],
    started_ns: int,
    completed_ns: int,
) -> dict[str, object]:
    """Analyze the fixed cube without any model prediction or calibration."""

    _physical_identity_gate(child_outcomes)
    if (
        sealed_c.get("objectives")
        != {
            "total_lut_count": EXPECTED_SEED_OBJECTIVES[0],
            "total_levels": EXPECTED_SEED_OBJECTIVES[1],
        }
        or sealed_c.get("cec_passed") is not True
    ):
        raise RuntimeError("sealed failed-v3 C gate failed during analysis")

    rows: list[dict[str, object]] = []
    child_by_label = {
        str(outcome["label"]): copy.deepcopy(dict(outcome))
        for outcome in child_outcomes
    }
    for arm in CUBE:
        if arm.known_objectives is not None:
            objective = arm.known_objectives
            rows.append(
                {
                    **arm.identity_record(),
                    "valid": True,
                    "cec_passed": True,
                    "candidate_local_failure_status": None,
                    "objective_source": (
                        "sealed_failed_v3_fresh_parent"
                        if arm.label == "C"
                        else "sealed_pre_block"
                    ),
                    "objectives": {
                        "total_lut_count": objective[0],
                        "total_levels": objective[1],
                    },
                    "publication_sequence": (
                        sealed_c.get("publication_sequence")
                        if arm.label == "C"
                        else None
                    ),
                    "published_elapsed_ns": (
                        sealed_c.get("published_elapsed_ns")
                        if arm.label == "C"
                        else None
                    ),
                    "submission_elapsed_ns": None,
                    "cpu_affinity": sealed_c.get("cpu_affinity")
                    if arm.label == "C"
                    else None,
                }
            )
            continue
        physical = child_by_label[arm.label]
        rows.append(
            {
                **arm.identity_record(),
                "valid": physical.get("valid") is True,
                "cec_passed": physical.get("cec_passed") is True,
                "candidate_local_failure_status": physical.get(
                    "candidate_local_failure_status"
                ),
                "objective_source": "fresh_engine_v4_block",
                "objectives": copy.deepcopy(physical.get("objectives")),
                "publication_sequence": physical.get("publication_sequence"),
                "published_elapsed_ns": physical.get("published_elapsed_ns"),
                "submission_elapsed_ns": physical.get("submission_elapsed_ns"),
                "cpu_affinity": copy.deepcopy(physical.get("cpu_affinity")),
            }
        )

    by_arm = {str(row["arm"]): row for row in rows}
    values = {
        label: _objective_tuple(by_arm[label])
        for label in ALL_ARM_ORDER
        if by_arm[label].get("valid") is True
    }
    preblock = [by_arm[label] for label in ("C", "A", "B", "D", "AB")]
    preblock_front = v3._front(preblock)
    preblock_hv = v3.oracle.hypervolume(
        [_objective_tuple(row) for row in preblock], REFERENCE_POINT
    )
    if {row["arm"] for row in preblock_front} != {"D", "AB"} or preblock_hv != 213:
        raise RuntimeError("pre-block archive failed front {D,AB}/HV 213 gate")

    combined_front = v3._front(rows)
    combined_front_arms = {str(row["arm"]) for row in combined_front}
    valid_vectors = [_objective_tuple(row) for row in rows if row.get("valid") is True]
    new_arm_decisions: list[dict[str, object]] = []
    for label in NEW_ARM_ORDER:
        row = by_arm[label]
        if row.get("valid") is not True:
            new_arm_decisions.append(
                {
                    "arm": label,
                    "valid": False,
                    "candidate_local_failure_status": row[
                        "candidate_local_failure_status"
                    ],
                    "objectives": None,
                    "preblock_dominators": None,
                    "preblock_arms_dominated": None,
                    "enters_combined_front": False,
                    "unique_objective_vector_on_combined_cube_front": False,
                    "marginal_fixed_reference_hv_gain": None,
                    "contributes_search_value": False,
                }
            )
            continue
        point = _objective_tuple(row)
        dominators = [
            str(other["arm"])
            for other in preblock
            if v3._dominates(_objective_tuple(other), point)
        ]
        dominated = [
            str(other["arm"])
            for other in preblock
            if v3._dominates(point, _objective_tuple(other))
        ]
        marginal = (
            v3.oracle.hypervolume(
                [*(_objective_tuple(other) for other in preblock), point],
                REFERENCE_POINT,
            )
            - preblock_hv
        )
        enters_front = label in combined_front_arms
        unique_front_vector = enters_front and valid_vectors.count(point) == 1
        new_arm_decisions.append(
            {
                "arm": label,
                "valid": True,
                "candidate_local_failure_status": None,
                "objectives": copy.deepcopy(row["objectives"]),
                "preblock_dominators": dominators,
                "preblock_arms_dominated": dominated,
                "enters_combined_front": enters_front,
                "unique_objective_vector_on_combined_cube_front": unique_front_vector,
                "marginal_fixed_reference_hv_gain": marginal,
                "contributes_search_value": unique_front_vector or marginal > 0,
            }
        )

    interaction_terms: dict[str, tuple[tuple[int, str], ...]] = {
        "I_AB": ((1, "AB"), (-1, "A"), (-1, "B"), (1, "C")),
        "I_AD": ((1, "AD"), (-1, "A"), (-1, "D"), (1, "C")),
        "I_BD": ((1, "BD"), (-1, "B"), (-1, "D"), (1, "C")),
        "I_ABD": (
            (1, "ABD"),
            (-1, "AB"),
            (-1, "AD"),
            (-1, "BD"),
            (1, "A"),
            (1, "B"),
            (1, "D"),
            (-1, "C"),
        ),
    }
    interactions: dict[str, dict[str, object]] = {}
    interaction_values: dict[str, tuple[int, int]] = {}
    for name, terms in interaction_terms.items():
        missing = [label for _, label in terms if label not in values]
        missing = list(dict.fromkeys(missing))
        if missing:
            interactions[name] = {
                "available": False,
                "missing_arms": missing,
                "total_lut_count": None,
                "total_levels": None,
            }
        else:
            residual = _interaction(values, terms)
            interaction_values[name] = residual
            interactions[name] = {
                "available": True,
                "missing_arms": [],
                "total_lut_count": residual[0],
                "total_levels": residual[1],
                "sign_interpretation": (
                    "negative=favorable synergy; positive=antagonism for minimized objective"
                ),
            }

    complete_cube = all(label in values for label in ALL_ARM_ORDER)
    if complete_cube:
        additive = _interaction(values, ((1, "A"), (1, "B"), (1, "D"), (-2, "C")))
        main_plus_pair = _interaction(
            values,
            (
                (1, "AB"),
                (1, "AD"),
                (1, "BD"),
                (-1, "A"),
                (-1, "B"),
                (-1, "D"),
                (1, "C"),
            ),
        )
        triple = values["ABD"]
        arithmetic: dict[str, object] = {
            "available": True,
            "missing_arms": [],
            "observed_ABD": list(triple),
            "additive_main_effect_prediction": list(additive),
            "additive_main_effect_residual": [
                triple[index] - additive[index] for index in range(2)
            ],
            "main_plus_pair_effect_prediction": list(main_plus_pair),
            "main_plus_pair_effect_residual": [
                triple[index] - main_plus_pair[index] for index in range(2)
            ],
            "third_order_residual_matches_I_ABD": tuple(
                triple[index] - main_plus_pair[index] for index in range(2)
            )
            == interaction_values["I_ABD"],
            "objective_order": ["total_lut_count", "total_levels"],
        }
    else:
        arithmetic = {
            "available": False,
            "missing_arms": [label for label in NEW_ARM_ORDER if label not in values],
            "objective_order": ["total_lut_count", "total_levels"],
        }

    sensitivity = deferred_oracle.get("full_oracle_sensitivity")
    if type(sensitivity) is not dict or sensitivity.get("hypervolume") != 700:
        raise RuntimeError("sealed local-oracle sensitivity failed HV 700 gate")
    sensitivity_front = sensitivity.get("front")
    if type(sensitivity_front) is not list:
        raise RuntimeError("sealed local-oracle sensitivity front is invalid")
    sensitivity_points = [
        (
            v3.oracle._as_exact_int(
                row["objectives"]["total_lut_count"], "sensitivity LUT"
            ),
            v3.oracle._as_exact_int(
                row["objectives"]["total_levels"], "sensitivity levels"
            ),
        )
        for row in sensitivity_front
    ]
    if v3.oracle.hypervolume(sensitivity_points, REFERENCE_POINT) != 700:
        raise RuntimeError("sealed local-oracle sensitivity front did not reproduce")
    sensitivity_marginals: list[dict[str, object]] = []
    for label in NEW_ARM_ORDER:
        row = by_arm[label]
        marginal = None
        if row.get("valid") is True:
            marginal = (
                v3.oracle.hypervolume(
                    [*sensitivity_points, _objective_tuple(row)], REFERENCE_POINT
                )
                - 700
            )
        sensitivity_marginals.append(
            {
                "arm": label,
                "valid": row.get("valid") is True,
                "marginal_hv_gain": marginal,
            }
        )
    sensitivity_terminal_hv = v3.oracle.hypervolume(
        [
            *sensitivity_points,
            *(
                _objective_tuple(by_arm[label])
                for label in NEW_ARM_ORDER
                if by_arm[label].get("valid") is True
            ),
        ],
        REFERENCE_POINT,
    )

    elapsed_ns = completed_ns - started_ns
    affinity_values = [
        tuple(by_arm[label].get("cpu_affinity") or ()) for label in NEW_ARM_ORDER
    ]
    affinity_gate = (
        all(len(value) == 1 and value[0] in ENGINE_CPUS for value in affinity_values)
        and len(set(affinity_values)) == len(affinity_values)
        and {value[0] for value in affinity_values} == set(ENGINE_CPUS)
    )
    publication_sequences = [
        by_arm[label].get("publication_sequence") for label in NEW_ARM_ORDER
    ]
    persistence_gate = all(
        type(value) is int for value in publication_sequences
    ) and set(publication_sequences) == {1, 2, 3}
    mandatory_cec_gate = all(
        row.get("cec_passed") is True
        for row in (by_arm[label] for label in NEW_ARM_ORDER)
        if row.get("valid") is True
    ) and all(
        by_arm[label].get("candidate_local_failure_status") != "cec_failed_or_missing"
        for label in NEW_ARM_ORDER
    )
    quality_by_arm = []
    for label in NEW_ARM_ORDER:
        row = by_arm[label]
        published = row.get("published_elapsed_ns")
        submitted = row.get("submission_elapsed_ns")
        duration = (
            published - submitted
            if type(published) is int and type(submitted) is int
            else None
        )
        quality_by_arm.append(
            {
                "arm": label,
                "submission_to_publication_ns": duration,
                "quality_horizon_met": (
                    duration is not None
                    and duration <= QUALITY_HORIZON_SECONDS * 1_000_000_000
                ),
            }
        )
    quality_horizon_met = all(row["quality_horizon_met"] for row in quality_by_arm)
    protocol_gates = {
        "preregistration_exact": preregistration.get("sha256")
        == EXPECTED_PREREGISTRATION_SHA256,
        "failed_v3_exact_and_only_C": failed_v3.get("semantic_closure", {}).get(
            "new_child_submissions"
        )
        == 0,
        "cpu_idle_admission_passed": admission.get("passed") is True,
        "engine_patch_replay_and_identity": all(
            arm.patch_record is not None
            and arm.patch_record.get("replay_verified") is True
            and arm.patch_record.get("target_hash")
            == arm.typed_json_configuration_sha256
            for arm in CUBE
        ),
        "exact_three_unique_physical_evaluations": len(
            {row["boils_configuration_sha256"] for row in child_outcomes}
        )
        == 3,
        "evaluator_provenance_exact": _provenance_is_exact(evaluator_provenance),
        "persistence_exact": persistence_gate,
        "affinity_exact_distinct": affinity_gate,
        "mandatory_cec": mandatory_cec_gate,
        "hard_cleanup_deadline_met": elapsed_ns
        <= HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000,
        "deferred_oracle_after_durable_children": deferred_oracle.get("source_sha256")
        == {
            name: v3.EVIDENCE_SOURCES[name][1]
            for name in ("oracle_finalized", "oracle_summary")
        },
        "zero_llm_calls": True,
    }
    invalid_arms = [
        {
            "arm": label,
            "candidate_local_failure_status": by_arm[label][
                "candidate_local_failure_status"
            ],
        }
        for label in NEW_ARM_ORDER
        if by_arm[label].get("valid") is not True
    ]
    contributes = any(
        row["contributes_search_value"] is True for row in new_arm_decisions
    )
    terminal_hv = v3.oracle.hypervolume(
        [_objective_tuple(row) for row in rows if row.get("valid") is True],
        REFERENCE_POINT,
    )
    return {
        "schema_version": 1,
        "status": "succeeded" if complete_cube else "partial_candidate_local_invalid",
        "completed_at_utc": _utc_now(),
        "development_only": True,
        "protocol_acceptance_passed": all(protocol_gates.values()),
        "claim_boundary": (
            "One post-hoc BOiLS/log2 deterministic recombination cube; not an "
            "optimizer, memory, genericity, SOTA, or wall-clock claim."
        ),
        "cube_outcomes": rows,
        "engine_materialized_patches": {
            arm.label: copy.deepcopy(arm.patch_record) for arm in CUBE
        },
        "partial_negative_record": {
            "present": bool(invalid_arms),
            "invalid_arms": invalid_arms,
            "fixed_arms_consumed_without_retry_or_replacement": bool(invalid_arms),
        },
        "interactions": interactions,
        "triple_prediction_arithmetic": arithmetic,
        "pareto": {
            "primary_comparison_archive": ["C", "A", "B", "D", "AB"],
            "preblock_front": ["D", "AB"],
            "preblock_front_objective_sort": [row["arm"] for row in preblock_front],
            "combined_front": [row["arm"] for row in combined_front],
            "new_arm_decisions": new_arm_decisions,
        },
        "hypervolume": {
            "reference_point": list(REFERENCE_POINT),
            "preblock": preblock_hv,
            "terminal": terminal_hv,
            "delta": terminal_hv - preblock_hv,
        },
        "sealed_local_oracle_sensitivity": {
            "primary_decision_uses_this": False,
            "preblock_hypervolume": 700,
            "terminal_hypervolume": sensitivity_terminal_hv,
            "delta": sensitivity_terminal_hv - 700,
            "front": copy.deepcopy(sensitivity_front),
            "new_arm_marginals": sensitivity_marginals,
        },
        "model": {
            "logical_llm_calls": 0,
            "result_reported": False,
            "calibration_reported": False,
            "reason": "engine-only continuation",
        },
        "decision": {
            "deterministic_disjoint_recombination_advances": contributes,
            "interaction_recording_advances": complete_cube,
            "rule": (
                "at least one valid new arm has a unique combined-front vector "
                "or positive marginal fixed-reference hypervolume"
            ),
        },
        "resources": {
            "physical_evaluations": EXPECTED_PHYSICAL_EVALUATIONS,
            "logical_llm_calls": 0,
            "retries": 0,
            "replacements": 0,
            "elapsed_ns_from_child_wave_start": elapsed_ns,
            "quality_horizon_ns_from_each_child_submission": QUALITY_HORIZON_SECONDS
            * 1_000_000_000,
            "quality_by_arm": quality_by_arm,
            "quality_horizon_met": quality_horizon_met,
            "quality_horizon_failure": not quality_horizon_met,
            "hard_cleanup_deadline_ns": HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000,
            "child_cpu_affinities": [list(value) for value in affinity_values],
        },
        "protocol_gates": protocol_gates,
        "scientific_completeness": {
            "all_three_new_arms_valid": complete_cube,
            "complete_cube_interactions_available": complete_cube,
            "quality_horizon_met": quality_horizon_met,
        },
        "limitations": [
            "The cube is one post-hoc log2 development mechanism test.",
            "Timing is descriptive on a shared host despite CPU-idle admission.",
            "Cross-run interaction arithmetic is descriptive, not randomized factorial evidence.",
            "The sealed local oracle is sensitivity-only and cannot affect the operator rule.",
        ],
    }


async def run_engine_block(
    *,
    evaluator: object,
    recorder: v3.oracle.EvaluationPublicationRecorder,
    trace: v3.oracle.TraceRecorder,
    failed_v3: Mapping[str, object],
    preregistration: Mapping[str, object],
    admission: Mapping[str, object],
    deferred_oracle_loader: Callable[[], Mapping[str, object]] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Submit the exact three-child engine wave once, with no provider path."""

    _validate_bound_inputs(failed_v3=failed_v3, preregistration=preregistration)
    _validate_admission_record(admission)
    provenance = _assert_evaluator_provenance(evaluator)

    started_ns = clock_ns()
    recorder.begin(started_ns)
    trace.begin(started_ns)
    trace.emit(
        "engine_continuation_started",
        preregistration_sha256=EXPECTED_PREREGISTRATION_SHA256,
        failed_v3_finalized_sha256=FAILED_V3_FILES["finalized.json"],
        sealed_parent_source="failed-v3 C; not reevaluated",
        submission_order=list(NEW_ARM_ORDER),
        logical_llm_calls=0,
        quality_horizon_ns=QUALITY_HORIZON_SECONDS * 1_000_000_000,
        hard_cleanup_deadline_ns=HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000,
        evaluator_provenance=provenance,
    )
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=CHILD_WORKERS,
        thread_name_prefix="boils-recombination-engine-v4",
    )
    futures = []
    submission_elapsed: dict[str, int] = {}
    try:
        for spec in CHILD_SCHEDULE:
            submitted_ns = clock_ns()
            submission_elapsed[spec.label] = max(0, submitted_ns - started_ns)
            trace.emit(
                "candidate_submitted",
                arm=spec.label,
                **spec.identity_record(),
                submission_selected_by="engine_fixed_schedule",
                retry_allowed=False,
                replacement_allowed=False,
            )
            futures.append(
                loop.run_in_executor(
                    executor,
                    lambda current=spec: v3.oracle._evaluate_one(
                        evaluator=evaluator,
                        recorder=recorder,
                        spec=current,
                    ),
                )
            )
        try:
            remaining_ns = HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000 - (
                clock_ns() - started_ns
            )
            if remaining_ns <= 0:
                raise RuntimeError("engine-v4 hard cleanup deadline expired")
            child_outcomes = await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=remaining_ns / 1_000_000_000,
            )
        except TimeoutError as exc:
            for future in futures:
                future.cancel()
            raise RuntimeError("engine-v4 hard cleanup deadline expired") from exc
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    completed_ns = clock_ns()
    if completed_ns - started_ns > HARD_CLEANUP_DEADLINE_SECONDS * 1_000_000_000:
        raise RuntimeError("engine-v4 hard cleanup deadline was exceeded")
    if len(recorder.records()) != EXPECTED_PHYSICAL_EVALUATIONS:
        raise RuntimeError("engine-v4 did not durably publish exactly three outcomes")
    _physical_identity_gate(child_outcomes)
    for outcome in child_outcomes:
        if outcome.get("valid") is True and outcome.get("cec_passed") is not True:
            raise RuntimeError("successful engine-v4 arm lacks mandatory CEC")
        if outcome.get("candidate_local_failure_status") == "cec_failed_or_missing":
            raise RuntimeError("engine-v4 mandatory CEC failed")
        outcome["submission_elapsed_ns"] = submission_elapsed[str(outcome["label"])]
    affinities = [
        tuple(outcome.get("cpu_affinity") or ()) for outcome in child_outcomes
    ]
    if not (
        all(len(value) == 1 and value[0] in ENGINE_CPUS for value in affinities)
        and len(set(affinities)) == 3
        and {value[0] for value in affinities} == set(ENGINE_CPUS)
    ):
        raise RuntimeError("engine-v4 exact distinct affinity gate failed")

    trace.emit(
        "fixed_child_wave_durable",
        report_order=list(NEW_ARM_ORDER),
        physical_publications=len(recorder.records()),
        candidate_local_invalids=sum(
            outcome.get("valid") is not True for outcome in child_outcomes
        ),
    )
    deferred_oracle = dict(
        v3.verify_deferred_oracle_evidence()
        if deferred_oracle_loader is None
        else deferred_oracle_loader()
    )
    _validate_deferred_oracle(deferred_oracle)
    trace.emit(
        "sealed_local_oracle_verified",
        chronology="after_all_three_physical_outcomes_were_durable",
        sensitivity_hypervolume=700,
    )
    summary = analyze_engine_cube(
        child_outcomes=child_outcomes,
        sealed_c=failed_v3["sealed_c"],
        deferred_oracle=deferred_oracle,
        evaluator_provenance=provenance,
        failed_v3=failed_v3,
        preregistration=preregistration,
        admission=admission,
        started_ns=started_ns,
        completed_ns=completed_ns,
    )
    summary["failed_v3_source"] = copy.deepcopy(dict(failed_v3))
    summary["preregistration"] = copy.deepcopy(dict(preregistration))
    summary["resource_admission"] = copy.deepcopy(dict(admission))
    summary["deferred_oracle_verification"] = copy.deepcopy(deferred_oracle)
    summary["evaluator_provenance"] = provenance
    if summary["protocol_acceptance_passed"] is not True:
        raise RuntimeError("engine-v4 strict protocol acceptance gate failed")
    trace.emit(
        "engine_continuation_analysis_completed",
        protocol_acceptance_passed=True,
        terminal_hypervolume=summary["hypervolume"]["terminal"],
        recombination_advances=summary["decision"][
            "deterministic_disjoint_recombination_advances"
        ],
    )
    return summary


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "offline_tests": AGENT_EVOLVE_ROOT
        / "tests/test_run_boils_recombination_engine_v4_offline.py",
        "v3_cube_engine": Path(v3.__file__).resolve(),
        "oracle_evaluation_audit": Path(v3.oracle.__file__).resolve(),
        "actions": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/actions.py",
        "evaluator": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
        "typed_patch": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/variation/typed_patch.py",
        "patch_domain": AGENT_EVOLVE_ROOT / "src/agent_evolve/domain/patch.py",
    }
    return {name: support._sha256(path) for name, path in paths.items()}


def _manifest(
    *,
    evaluator: BoilsAbcEvaluator,
    admission: Mapping[str, object],
    failed_v3: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "started_at_utc": _utc_now(),
        "development_only": True,
        "claim_boundary": (
            "One post-hoc BOiLS/log2 deterministic recombination cube; not an "
            "optimizer, memory, genericity, SOTA, or wall-clock claim."
        ),
        "preregistration": copy.deepcopy(dict(preregistration)),
        "failed_v3_source": copy.deepcopy(dict(failed_v3)),
        "sealed_local_oracle_sources": {
            name: {
                "source": str(v3.EVIDENCE_SOURCES[name][0]),
                "sha256": v3.EVIDENCE_SOURCES[name][1],
            }
            for name in ("oracle_finalized", "oracle_summary")
        },
        "resource_admission": copy.deepcopy(dict(admission)),
        "cube": {
            "all_arm_order": list(ALL_ARM_ORDER),
            "physical_order": list(NEW_ARM_ORDER),
            "sealed_parent_reevaluated": False,
            "arms": [
                {
                    **arm.identity_record(),
                    "configuration": arm.configuration,
                    "known_objectives": (
                        None
                        if arm.known_objectives is None
                        else {
                            "total_lut_count": arm.known_objectives[0],
                            "total_levels": arm.known_objectives[1],
                        }
                    ),
                    "engine_patch": copy.deepcopy(arm.patch_record),
                }
                for arm in CUBE
            ],
        },
        "task": {
            "circuit": "log2",
            "circuit_sha256": EXPECTED_CIRCUIT_SHA256,
            "abc_sha256": EXPECTED_ABC_SHA256,
            "mapping": "LUT-6 followed by mandatory CEC",
            "logical_cpus": list(ENGINE_CPUS),
            "per_candidate_timeout_seconds": PER_CANDIDATE_TIMEOUT_SECONDS,
            "quality_horizon_seconds_from_each_submission": QUALITY_HORIZON_SECONDS,
            "hard_cleanup_deadline_seconds": HARD_CLEANUP_DEADLINE_SECONDS,
            "physical_evaluations": EXPECTED_PHYSICAL_EVALUATIONS,
            "retries": 0,
            "replacement_candidates": 0,
            "private_cache_free": True,
        },
        "model": {
            "logical_llm_calls": 0,
            "result_reported": False,
            "calibration_reported": False,
        },
        "analysis": {
            "reference_point": list(REFERENCE_POINT),
            "primary_comparison_archive": ["C", "A", "B", "D", "AB"],
            "required_preblock_front": ["D", "AB"],
            "required_preblock_hypervolume": 213,
            "interactions": ["I_AB", "I_AD", "I_BD", "I_ABD"],
            "search_value_rule": (
                "unique objective vector on combined cube front OR positive "
                "marginal HV versus the primary archive"
            ),
            "sealed_local_oracle_role": "post-outcome labeled sensitivity only",
        },
        "evaluator_provenance": evaluator.provenance(),
        "source_sha256": _source_hashes(),
        "python_source_snapshot": support._source_snapshot(
            (
                AGENT_EVOLVE_ROOT / "src",
                AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc",
                AGENT_EVOLVE_ROOT / "examples/development",
            )
        ),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "process_affinity_at_start": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        },
    }


_DURABLE_V3_COPY_NAMES = {name: f"sealed_failed_v3_{name}" for name in FAILED_V3_FILES}


def _finalize(run_dir: Path, status: str) -> None:
    names = (
        "manifest.json",
        "runner_source.py",
        "preregistration.md",
        "resource_admission.json",
        *_DURABLE_V3_COPY_NAMES.values(),
        "sealed_oracle_finalized.json",
        "sealed_oracle_summary.json",
        "events.jsonl",
        "evaluations.jsonl",
        "summary.json",
        "failure.json",
    )
    files: dict[str, dict[str, object]] = {}
    for name in names:
        path = run_dir / name
        if not path.exists():
            continue
        payload = path.read_bytes()
        files[name] = {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            **({"lines": len(payload.splitlines())} if name.endswith(".jsonl") else {}),
        }
    support._write_json(
        run_dir / "finalized.json",
        {
            "schema_version": 1,
            "status": status,
            "completed_at_utc": _utc_now(),
            "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
            "failed_v3_finalized_sha256": FAILED_V3_FILES["finalized.json"],
            "files": files,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--cpus", default=",".join(str(cpu) for cpu in ENGINE_CPUS))
    parser.add_argument(
        "--per-candidate-timeout-seconds",
        type=int,
        default=PER_CANDIDATE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--quality-horizon-seconds", type=int, default=QUALITY_HORIZON_SECONDS
    )
    parser.add_argument(
        "--hard-cleanup-deadline-seconds",
        type=int,
        default=HARD_CLEANUP_DEADLINE_SECONDS,
    )
    return parser


def _assert_frozen_cli(args: argparse.Namespace) -> None:
    expected = {
        "run_id": RUN_ID,
        "cpus": ",".join(str(cpu) for cpu in ENGINE_CPUS),
        "per_candidate_timeout_seconds": PER_CANDIDATE_TIMEOUT_SECONDS,
        "quality_horizon_seconds": QUALITY_HORIZON_SECONDS,
        "hard_cleanup_deadline_seconds": HARD_CLEANUP_DEADLINE_SECONDS,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise SystemExit(f"engine-v4 freezes --{name.replace('_', '-')}={value}")


def main() -> None:
    args = _parser().parse_args()
    _assert_frozen_cli(args)
    run_dir = args.log_root.resolve() / RUN_ID
    # Admission is intentionally outside the durable-run try/finally.  Failure
    # leaves no directory and consumes no physical arm.
    preregistration, failed_v3, admission = prepare_launch_admission(run_dir=run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)
    status = "failed"
    event_writer: v1.DurableJsonlWriter | None = None
    evaluation_writer: v1.DurableJsonlWriter | None = None
    try:
        support._write_json(run_dir / "resource_admission.json", admission)
        if (
            json.loads(
                (run_dir / "resource_admission.json").read_text(encoding="utf-8")
            )
            != admission
        ):
            raise RuntimeError("durable CPU admission record failed exact replay")
        shutil.copyfile(Path(__file__).resolve(), run_dir / "runner_source.py")
        shutil.copyfile(PREREGISTRATION_PATH, run_dir / "preregistration.md")
        if support._sha256(run_dir / "preregistration.md") != (
            EXPECTED_PREREGISTRATION_SHA256
        ):
            raise RuntimeError("durable preregistration copy failed its hash gate")
        for name, destination in _DURABLE_V3_COPY_NAMES.items():
            shutil.copyfile(FAILED_V3_RUN_DIR / name, run_dir / destination)
            if support._sha256(run_dir / destination) != FAILED_V3_FILES[name]:
                raise RuntimeError(f"durable failed-v3 copy failed: {name}")
        for source_name, destination in (
            ("oracle_finalized", "sealed_oracle_finalized.json"),
            ("oracle_summary", "sealed_oracle_summary.json"),
        ):
            source, expected_hash = v3.EVIDENCE_SOURCES[source_name]
            shutil.copyfile(source, run_dir / destination)
            if support._sha256(run_dir / destination) != expected_hash:
                raise RuntimeError(f"durable oracle copy failed: {source_name}")

        event_writer = v1.DurableJsonlWriter(run_dir / "events.jsonl")
        evaluation_writer = v1.DurableJsonlWriter(run_dir / "evaluations.jsonl")
        trace = v3.oracle.TraceRecorder(event_writer)
        recorder = v3.oracle.EvaluationPublicationRecorder(
            evaluation_writer,
            trace,
            schedule=CHILD_SCHEDULE,
        )
        settings = AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=("log2",),
            affinity_sets=tuple((cpu,) for cpu in ENGINE_CPUS),
            per_circuit_timeout_s=float(PER_CANDIDATE_TIMEOUT_SECONDS),
        )
        evaluator = BoilsAbcEvaluator(settings, observer=recorder)
        _assert_evaluator_provenance(evaluator)
        if support._sha256(run_dir / "runner_source.py") != support._sha256(
            Path(__file__).resolve()
        ):
            raise RuntimeError("durable runner copy failed its hash gate")
        support._write_json(
            run_dir / "manifest.json",
            _manifest(
                evaluator=evaluator,
                admission=admission,
                failed_v3=failed_v3,
                preregistration=preregistration,
            ),
        )
        summary = asyncio.run(
            run_engine_block(
                evaluator=evaluator,
                recorder=recorder,
                trace=trace,
                failed_v3=failed_v3,
                preregistration=preregistration,
                admission=admission,
            )
        )
        summary["evaluator_observations"] = v1._evaluation_log_summary(
            run_dir / "evaluations.jsonl"
        )
        support._write_json(run_dir / "summary.json", summary)
        status = "succeeded"
    except BaseException as exc:
        support._write_json(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "failure_type": type(exc).__name__,
                "safe_message": "BOiLS engine-v4 continuation failed; inspect durable traces",
            },
        )
        raise
    finally:
        if event_writer is not None:
            event_writer.close()
        if evaluation_writer is not None:
            evaluation_writer.close()
        _finalize(run_dir, status)
    print(_canonical_json({"run_dir": str(run_dir), "status": status}))


if __name__ == "__main__":
    main()

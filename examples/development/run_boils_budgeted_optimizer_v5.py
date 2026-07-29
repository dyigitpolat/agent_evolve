#!/usr/bin/env python3
"""Run the frozen two-generation BOiLS budgeted-agentic optimizer v5.

This is a post-hoc workflow-development kill test on BOiLS/log2.  It is not
held-out benchmark evidence.  A separate no-I/O preview freezes the five exact
model prompts and randomized card assignments before CPU admission.  The live
generator remains disabled until its preparation trace exactly replays that
durable readiness record.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials  # noqa: E402

from agent_evolve.application.agentic_evolution import (  # noqa: E402
    AgenticEvolutionEngine,
    EvolutionCandidate,
    PreparedInvocation,
    ProposalAuthority,
)
from agent_evolve.application.budgeted_optimizer import (  # noqa: E402
    BudgetedAgenticOptimizer,
    OptimizerBudget,
    OptimizerResult,
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.gated_agentic_generator import (  # noqa: E402
    AgenticTelemetryPolicy,
    TelemetryGatedAgenticGenerator,
)
from agent_evolve.application.insight_memory import InsightMemoryBank  # noqa: E402
from agent_evolve.application.pareto_archive import (  # noqa: E402
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.domain.lineage import CandidateOccurrence  # noqa: E402
from agent_evolve.domain.typed_json import (  # noqa: E402
    canonical_typed_json_bytes,
    typed_json_sha256,
)
from agent_evolve.domain.llm_task_queue import AttemptRequestVariant  # noqa: E402
from agent_evolve.infrastructure.ids import DeterministicIdFactory  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES,
    MAX_SCHEMA_REPAIR_REQUIRED_PATHS,
    MAX_SCHEMA_REPAIR_SCHEMA_NODES,
    OutcomePublicationPolicy,
    SCHEMA_REPAIR_POLICY_ID,
    SCHEMA_REPAIR_POLICY_MANIFEST,
    SCHEMA_REPAIR_POLICY_VERSION,
    SchemaRepairAttemptPolicy,
    create_production_queued_runner,
    structured_generation_outcome_record,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter  # noqa: E402
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    AgenticGenerator,
    ReflectionGenerationRequest,
    VariationGenerationRequest,
)
from examples.benchmarks.boils_abc import budgeted_v5_support as v5  # noqa: E402
from examples.benchmarks.boils_abc.budgeted_v5_planner import (  # noqa: E402
    BoilsBudgetedV5Planner,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    ABC_SOURCE_COMMIT,
    AbcEvaluatorSettings,
    BOILS_SOURCE_COMMIT,
    BoilsAbcEvaluator,
    BoilsEvaluation,
    BoilsEvaluationFailure,
    BoilsEvaluationObservation,
    CURRENT_ABC_SHA256,
    EPFL_SOURCE_COMMIT,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem  # noqa: E402


RUN_ID = "boils_budgeted_optimizer_v5_attempt4_20260714"
READINESS_SCHEMA_VERSION = 2
MODEL = "deepseek/deepseek-v4-pro"
CANONICAL_RESOLVED_MODEL = "deepseek/deepseek-v4-pro-20260423"
ALLOWED_RESOLVED_MODELS = (MODEL, CANONICAL_RESOLVED_MODEL)
PROVIDER_ONLY = ("streamlake",)
ALLOWED_RESOLVED_PROVIDERS = ("StreamLake",)
SELECTED_ENDPOINT_NAME = f"StreamLake | {CANONICAL_RESOLVED_MODEL}"
REQUIRED_ENDPOINT_CAPABILITIES = (
    "max_tokens",
    "temperature",
    "tools",
    "tool_choice",
    "response_format",
)
ENGINE_CPUS = (5, 6, 8, 23)
CPU_SIBLING_PAIRS = ((5, 69), (6, 70), (8, 72), (23, 87))
CPU_ADMISSION_CPUS = (*ENGINE_CPUS, 69, 70, 72, 87)
PILOT_CIRCUITS = ("log2",)
PER_CANDIDATE_TIMEOUT_SECONDS = 120

QUEUE_MAX_IN_FLIGHT = 5
QUEUE_MAX_PENDING = 10
QUEUE_MAX_ATTEMPTS = 2
QUEUE_ATTEMPT_TIMEOUT_SECONDS = 60
QUEUE_BASE_BACKOFF_SECONDS = 1
QUEUE_MAX_BACKOFF_SECONDS = 8
JITTER_SEED = 20_260_714
JITTER_DOMAIN = "boils-budgeted-v5-jitter-v1"

MAX_INPUT_TOKENS = 10_000
MAX_OUTPUT_TOKENS = 960
MAX_REASONING_TOKENS = 960
PROMPT_PRICE_USD_PER_TOKEN = Decimal("0.0000007134")
COMPLETION_PRICE_USD_PER_TOKEN = Decimal("0.0000014268")
CACHE_READ_PRICE_USD_PER_TOKEN = Decimal("0.00000005945")
DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD = Decimal("0.009873456")
DERIVED_MAX_ACCEPTED_RUN_COST_USD = Decimal("0.049367280")
MAX_SUCCESSFUL_RESPONSE_COST_USD = Decimal("0.010")
MAX_ACCEPTED_TERMINAL_RESPONSE_COST_USD = Decimal("0.050")
MAX_POTENTIALLY_BILLABLE_ATTEMPT_COST_USD = Decimal("0.100")
TEMPERATURE = 0.2

BUDGET = OptimizerBudget(
    max_unique_evaluations=9,
    max_logical_llm_calls=5,
    max_generations=2,
)
CPU_ADMISSION_WINDOWS = 3
CPU_ADMISSION_INTERVAL_SECONDS = 1.0
MAX_CPU_BUSY_FRACTION = 1.0
EXECUTION_CLASS = "shared_host_quality_only"
WALL_CLOCK_CLAIMS_ALLOWED = False
TIMING_EXCLUSION_REASON = (
    "Shared-host scheduler contention is recorded but not controlled; this run "
    "may inform optimizer quality and mechanism diagnosis only."
)
EXPECTED_CIRCUIT_SHA256 = (
    "c0d052af4e95de4c1327a2ceddd855518a052a8f3a3960e6d58c5b5ca65c0dde"
)
EXPECTED_TASKSET_SHA256 = (
    "a9c851792e54e91fba7b827019380abee54e715b6817899c835e4f221354b260"
)
ID_NAMESPACE = "boils_v5_attempt4_20260714"

ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_LOG_ROOT = ARTIFACT_ROOT / "experiment_logs" / "boils_agentic_development"
PRICING_SNAPSHOT_PATH = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_pricing_snapshot_20260714.json"
)
EXPECTED_PRICING_SNAPSHOT_SHA256 = (
    "5adea5e08d7aea5eb89de010e1750890fe6b7f70a3f7fe733a08996d0b8b7204"
)
CAPABILITY_SNAPSHOT_PATH = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_capability_snapshot_20260714.json"
)
EXPECTED_CAPABILITY_SNAPSHOT_SHA256 = (
    "131d0fef27cb24350f9c067ea7407cd9279ddbe242eef77e29451390a750a671"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _record_sha256(domain: str, value: object) -> str:
    if type(domain) is not str or not domain:
        raise ValueError("hash domain must be non-empty")
    return hashlib.sha256(
        b"boils-budgeted-v5-runner:v2\x00"
        + domain.encode("ascii")
        + b"\x00"
        + _canonical_json_bytes(value)
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attempt4_pricing_envelope_record() -> dict[str, object]:
    """Return the prospective cap derived from the immutable unit-price evidence."""

    derived_per_call = (
        Decimal(MAX_INPUT_TOKENS) * PROMPT_PRICE_USD_PER_TOKEN
        + Decimal(MAX_OUTPUT_TOKENS + MAX_REASONING_TOKENS)
        * COMPLETION_PRICE_USD_PER_TOKEN
    )
    derived_run = Decimal(BUDGET.max_logical_llm_calls) * derived_per_call
    if not (
        derived_per_call == DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD
        and derived_run == DERIVED_MAX_ACCEPTED_RUN_COST_USD
        and derived_per_call <= MAX_SUCCESSFUL_RESPONSE_COST_USD
        and derived_run <= MAX_ACCEPTED_TERMINAL_RESPONSE_COST_USD
    ):
        raise RuntimeError("attempt4 pricing envelope constants changed")
    body = {
        "schema_version": 1,
        "envelope_id": "boils_v5_attempt4_streamlake_token_caps_v1",
        "source_pricing_snapshot_sha256": EXPECTED_PRICING_SNAPSHOT_SHA256,
        "max_logical_calls": BUDGET.max_logical_llm_calls,
        "max_input_tokens_per_call": MAX_INPUT_TOKENS,
        "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
        "max_reasoning_tokens_per_call": MAX_REASONING_TOKENS,
        "prompt_usd_per_token": format(PROMPT_PRICE_USD_PER_TOKEN, "f"),
        "completion_usd_per_token": format(COMPLETION_PRICE_USD_PER_TOKEN, "f"),
        "cache_read_usd_per_token": format(CACHE_READ_PRICE_USD_PER_TOKEN, "f"),
        "reasoning_accounting": (
            "reasoning cap charged once more at the completion-token rate"
        ),
        "derived_max_cost_usd_per_call": str(DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD),
        "frozen_cost_ceiling_usd_per_call": str(MAX_SUCCESSFUL_RESPONSE_COST_USD),
        "derived_max_accepted_run_cost_usd": str(DERIVED_MAX_ACCEPTED_RUN_COST_USD),
        "frozen_accepted_run_ceiling_usd": str(MAX_ACCEPTED_TERMINAL_RESPONSE_COST_USD),
    }
    return {
        **body,
        "envelope_sha256": _record_sha256("attempt4-pricing-envelope", body),
    }


def attempt4_mechanism_contract_record() -> dict[str, object]:
    return {
        "failed_slot_continuation_policy_id": (v5.FAILED_SLOT_CONTINUATION_POLICY_ID),
        "failed_slot_continuation_policy_version": (
            v5.FAILED_SLOT_CONTINUATION_POLICY_VERSION
        ),
        "failed_slot_substitution_allowed": False,
        "batch_incremental_coverage_policy_id": (
            v5.BATCH_INCREMENTAL_COVERAGE_POLICY_ID
        ),
        "batch_incremental_coverage_policy_version": (
            v5.BATCH_INCREMENTAL_COVERAGE_POLICY_VERSION
        ),
        "front_aligned_reward_policy_id": v5.FRONT_ALIGNED_REWARD_POLICY_ID,
        "front_aligned_reward_policy_version": (v5.FRONT_ALIGNED_REWARD_POLICY_VERSION),
        "front_extension_raw_credit_hex": v5.FRONT_EXTENSION_RAW_CREDIT.hex(),
    }


def schema_repair_policy_record() -> dict[str, object]:
    manifest = SCHEMA_REPAIR_POLICY_MANIFEST
    if not (
        manifest.policy_id == SCHEMA_REPAIR_POLICY_ID
        and manifest.policy_version == SCHEMA_REPAIR_POLICY_VERSION
        and manifest.max_suffix_utf8_bytes == MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES
        and manifest.max_required_paths == MAX_SCHEMA_REPAIR_REQUIRED_PATHS
        and manifest.max_schema_nodes == MAX_SCHEMA_REPAIR_SCHEMA_NODES
    ):
        raise RuntimeError("schema-repair policy manifest changed")
    return {
        "durable_queue_outcome_schema_version": 4,
        "attempt_request_variants": [
            AttemptRequestVariant.ORIGINAL.value,
            AttemptRequestVariant.SCHEMA_REPAIR_V2.value,
        ],
        "policy_id": manifest.policy_id,
        "policy_version": manifest.policy_version,
        "max_required_paths": manifest.max_required_paths,
        "max_schema_nodes": manifest.max_schema_nodes,
        "max_suffix_utf8_bytes": manifest.max_suffix_utf8_bytes,
        "template_sha256": manifest.template_sha256,
        "policy_sha256": manifest.policy_sha256,
        "explicit_factory_injection": True,
    }


def durable_write_json(path: Path, value: object) -> str:
    """Atomically publish and directory-fsync one canonical JSON document."""

    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(payload).hexdigest()


class DurableDirectoryPublishError(OSError):
    """The caller owns a newly created directory whose dirent fsync failed."""


def durable_mkdir(path: Path) -> None:
    """Create an exclusive run directory and durably publish its dirent."""

    created = False
    try:
        path.mkdir(parents=True, exist_ok=False)
        created = True
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException as exc:
        if created:
            raise DurableDirectoryPublishError(
                f"created run directory but failed to durably publish it: {exc}"
            ) from exc
        raise


def durable_copy_file(source: Path, destination: Path) -> str:
    """Exclusively copy, file-fsync, and directory-fsync one source snapshot."""

    payload = source.read_bytes()
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(payload).hexdigest()


def _source_bundle_entries() -> tuple[tuple[str, Path], ...]:
    sources = tuple(
        (
            path.relative_to(AGENT_EVOLVE_ROOT).as_posix(),
            path,
        )
        for path in sorted((AGENT_EVOLVE_ROOT / "src" / "agent_evolve").rglob("*.py"))
    )
    benchmark_root = AGENT_EVOLVE_ROOT / "examples" / "benchmarks" / "boils_abc"
    live_boils_names = (
        "__init__.py",
        "actions.py",
        "evaluator.py",
        "problem_def.py",
        "variation_catalog.py",
        "budgeted_v5_support.py",
        "budgeted_v5_planner.py",
    )
    extras = (
        *(
            (
                (benchmark_root / name).relative_to(AGENT_EVOLVE_ROOT).as_posix(),
                benchmark_root / name,
            )
            for name in live_boils_names
        ),
        (
            Path(__file__).resolve().relative_to(AGENT_EVOLVE_ROOT).as_posix(),
            Path(__file__).resolve(),
        ),
        ("pyproject.toml", AGENT_EVOLVE_ROOT / "pyproject.toml"),
        ("uv.lock", AGENT_EVOLVE_ROOT / "uv.lock"),
    )
    entries = tuple(sorted((*sources, *extras), key=lambda item: item[0]))
    names = tuple(name for name, _ in entries)
    if len(set(names)) != len(names) or any(not path.is_file() for _, path in entries):
        raise RuntimeError("source bundle entries are missing or duplicated")
    return entries


def source_closure_record() -> dict[str, object]:
    """Hash the complete executable source closure without writing anything."""

    entry_records: list[dict[str, object]] = []
    for name, source in _source_bundle_entries():
        payload = source.read_bytes()
        entry_records.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    body = {
        "schema_version": 1,
        "entry_count": len(entry_records),
        "entries": entry_records,
    }
    return {**body, "closure_sha256": _record_sha256("source-closure", body)}


def validate_source_closure_record(value: Mapping[str, object]) -> dict[str, object]:
    """Fail closed on a malformed or internally inconsistent closure record."""

    record = copy.deepcopy(dict(value))
    rows = record.get("entries")
    body = {key: item for key, item in record.items() if key != "closure_sha256"}
    if not (
        record.get("schema_version") == 1
        and type(rows) is list
        and record.get("entry_count") == len(rows)
        and record.get("closure_sha256") == _record_sha256("source-closure", body)
    ):
        raise RuntimeError("source closure record changed")
    for row in rows:
        if not (
            type(row) is dict
            and set(row) == {"path", "bytes", "sha256"}
            and type(row["path"]) is str
            and row["path"]
            and type(row["bytes"]) is int
            and row["bytes"] >= 0
            and type(row["sha256"]) is str
            and len(row["sha256"]) == 64
        ):
            raise RuntimeError("source closure entry changed")
    paths = [row["path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError("source closure paths changed")
    return record


def durable_source_bundle(destination: Path) -> dict[str, object]:
    """Write a deterministic uncompressed tar of the complete live source closure."""

    entry_records: list[dict[str, object]] = []
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("xb") as stream:
        with tarfile.open(
            fileobj=stream,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive:
            for name, source in _source_bundle_entries():
                payload = source.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
                entry_records.append(
                    {
                        "path": name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    directory_fd = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    body = {
        "schema_version": 1,
        "format": "ustar_uncompressed",
        "metadata_normalization": {
            "mtime": 0,
            "mode": "0644",
            "uid": 0,
            "gid": 0,
            "uname": "",
            "gname": "",
            "entry_order": "ascending_posix_relative_path",
        },
        "entry_count": len(entry_records),
        "entries": entry_records,
        "bundle_bytes": destination.stat().st_size,
        "bundle_sha256": _file_sha256(destination),
    }
    return {**body, "record_sha256": _record_sha256("source-bundle", body)}


def verify_source_bundle(path: Path, record: Mapping[str, object]) -> None:
    """Re-hash the tar, every tar entry, and every corresponding live source."""

    rows = record.get("entries")
    expected_normalization = {
        "mtime": 0,
        "mode": "0644",
        "uid": 0,
        "gid": 0,
        "uname": "",
        "gname": "",
        "entry_order": "ascending_posix_relative_path",
    }
    body = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "record_sha256"
    }
    if (
        record.get("schema_version") != 1
        or type(rows) is not list
        or record.get("format") != "ustar_uncompressed"
        or record.get("metadata_normalization") != expected_normalization
        or record.get("entry_count") != len(rows)
        or record.get("bundle_sha256") != _file_sha256(path)
        or record.get("bundle_bytes") != path.stat().st_size
        or record.get("record_sha256") != _record_sha256("source-bundle", body)
    ):
        raise RuntimeError("durable source bundle record changed")
    for row in rows:
        if not (
            type(row) is dict
            and set(row) == {"path", "bytes", "sha256"}
            and type(row["path"]) is str
            and type(row["bytes"]) is int
            and row["bytes"] >= 0
            and type(row["sha256"]) is str
            and len(row["sha256"]) == 64
        ):
            raise RuntimeError("durable source bundle entry record changed")
    expected_by_name = {name: source for name, source in _source_bundle_entries()}
    if [row.get("path") for row in rows] != list(expected_by_name):
        raise RuntimeError("durable source bundle entry set changed")
    with tarfile.open(path, mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != list(expected_by_name):
            raise RuntimeError("durable source bundle member order changed")
        for member, row in zip(members, rows, strict=True):
            if not (
                member.isfile()
                and member.mtime == 0
                and member.mode == 0o644
                and member.uid == 0
                and member.gid == 0
                and member.uname == ""
                and member.gname == ""
            ):
                raise RuntimeError("durable source bundle metadata changed")
            extracted = archive.extractfile(member)
            if extracted is None:  # pragma: no cover - member.isfile gate.
                raise RuntimeError("durable source bundle member is unreadable")
            payload = extracted.read()
            live_payload = expected_by_name[member.name].read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if not (
                member.size == row.get("bytes") == len(payload)
                and digest == row.get("sha256")
                and payload == live_payload
            ):
                raise RuntimeError(f"source bundle/live source mismatch: {member.name}")


class DurableJsonlWriter:
    """Thread-safe fsync-before-return JSONL publication boundary."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("x", encoding="utf-8")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._lock = threading.Lock()
        self._closed = False

    def write(self, value: Mapping[str, object]) -> None:
        payload = _canonical_json_bytes(dict(value)).decode("ascii") + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("durable JSONL writer is closed")
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.close()
                self._closed = True


class EvaluationObservationRecorder:
    """Durably publish real BOiLS observations from evaluator worker threads."""

    def __init__(self, writer: DurableJsonlWriter) -> None:
        self._writer = writer
        self._lock = threading.Lock()
        self._sequence = 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._sequence

    def __call__(self, observation: BoilsEvaluationObservation) -> None:
        if type(observation) is BoilsEvaluation:
            status = "succeeded"
        elif type(observation) is BoilsEvaluationFailure:
            status = "candidate_local_failure"
        else:  # pragma: no cover - closed evaluator observation union.
            raise TypeError("unknown BOiLS evaluation observation")
        with self._lock:
            self._sequence += 1
            self._writer.write(
                {
                    "schema_version": 1,
                    "observation_sequence": self._sequence,
                    "recorded_at_utc": _utc_now(),
                    "status": status,
                    "observation": observation.as_dict(),
                }
            )


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

    def to_record(self) -> dict[str, object]:
        return {
            "cpu": self.cpu,
            "counters": list(self.counters),
            "total": self.total,
            "idle_plus_iowait": self.idle,
        }


def parse_proc_stat(
    text: str, cpus: Sequence[int] = CPU_ADMISSION_CPUS
) -> dict[int, CpuTickRow]:
    if type(text) is not str:
        raise TypeError("/proc/stat payload must be text")
    selected = tuple(cpus)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(cpu) is not int or cpu < 0 for cpu in selected)
    ):
        raise ValueError("CPU selection must be distinct non-negative integers")
    rows: dict[int, CpuTickRow] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == "cpu" or not fields[0].startswith("cpu"):
            continue
        suffix = fields[0][3:]
        if not suffix.isdigit() or int(suffix) not in selected:
            continue
        cpu = int(suffix)
        if cpu in rows or len(fields) < 6:
            raise RuntimeError(f"malformed or duplicate cpu{cpu} row")
        try:
            counters = tuple(int(value) for value in fields[1:])
        except ValueError as exc:
            raise RuntimeError(f"cpu{cpu} contains a non-integer counter") from exc
        if any(value < 0 for value in counters):
            raise RuntimeError(f"cpu{cpu} contains a negative counter")
        rows[cpu] = CpuTickRow(cpu, counters)
    missing = [cpu for cpu in selected if cpu not in rows]
    if missing:
        raise RuntimeError(f"/proc/stat is missing selected CPUs: {missing}")
    return {cpu: rows[cpu] for cpu in selected}


def _parse_cpu_list(text: str) -> tuple[int, ...]:
    """Parse the Linux canonical comma/range CPU-list representation."""

    if type(text) is not str or not text.strip():
        raise RuntimeError("CPU topology list is empty")
    values: list[int] = []
    for cell in text.strip().split(","):
        bounds = cell.split("-")
        if len(bounds) == 1 and bounds[0].isdigit():
            values.append(int(bounds[0]))
        elif (
            len(bounds) == 2
            and bounds[0].isdigit()
            and bounds[1].isdigit()
            and int(bounds[0]) <= int(bounds[1])
        ):
            values.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise RuntimeError("CPU topology list is malformed")
    result = tuple(sorted(set(values)))
    if len(result) != len(values):
        raise RuntimeError("CPU topology list contains duplicate CPUs")
    return result


def cpu_topology_record(
    *,
    reader: Callable[[Path], str] = lambda path: path.read_text(encoding="ascii"),
) -> dict[str, object]:
    """Bind every evaluator CPU and its SMT sibling from Linux sysfs."""

    expected_by_cpu = {cpu: pair for pair in CPU_SIBLING_PAIRS for cpu in pair}
    rows: list[dict[str, object]] = []
    for cpu in CPU_ADMISSION_CPUS:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        raw = reader(path)
        siblings = _parse_cpu_list(raw)
        if siblings != expected_by_cpu[cpu]:
            raise RuntimeError(f"cpu{cpu} SMT sibling topology changed: {siblings!r}")
        rows.append(
            {
                "cpu": cpu,
                "source": str(path),
                "source_text": raw.strip(),
                "thread_siblings": list(siblings),
            }
        )
    body = {
        "schema_version": 1,
        "engine_evaluator_cpus": list(ENGINE_CPUS),
        "admission_cpus": list(CPU_ADMISSION_CPUS),
        "smt_sibling_pairs": [list(pair) for pair in CPU_SIBLING_PAIRS],
        "rows": rows,
    }
    return {**body, "topology_sha256": _record_sha256("cpu-topology", body)}


def validate_cpu_topology_record(value: Mapping[str, object]) -> dict[str, object]:
    record = copy.deepcopy(dict(value))
    body = {key: item for key, item in record.items() if key != "topology_sha256"}
    rows = record.get("rows")
    if not (
        record.get("schema_version") == 1
        and record.get("engine_evaluator_cpus") == list(ENGINE_CPUS)
        and record.get("admission_cpus") == list(CPU_ADMISSION_CPUS)
        and record.get("smt_sibling_pairs")
        == [list(pair) for pair in CPU_SIBLING_PAIRS]
        and type(rows) is list
        and len(rows) == len(CPU_ADMISSION_CPUS)
        and record.get("topology_sha256") == _record_sha256("cpu-topology", body)
    ):
        raise RuntimeError("CPU topology record changed")
    expected_by_cpu = {cpu: pair for pair in CPU_SIBLING_PAIRS for cpu in pair}
    for cpu, row in zip(CPU_ADMISSION_CPUS, rows, strict=True):
        if not (
            type(row) is dict
            and row.get("cpu") == cpu
            and row.get("source")
            == f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            and row.get("thread_siblings") == list(expected_by_cpu[cpu])
            and _parse_cpu_list(str(row.get("source_text"))) == expected_by_cpu[cpu]
        ):
            raise RuntimeError("CPU topology row changed")
    return record


def sample_cpu_admission(
    *,
    reader: Callable[[], str] = lambda: Path("/proc/stat").read_text(encoding="utf-8"),
    sleeper: Callable[[float], object] = time.sleep,
    cpus: Sequence[int] = CPU_ADMISSION_CPUS,
    topology: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Record three load windows on evaluator CPUs and all SMT siblings."""

    selected = tuple(cpus)
    if selected != CPU_ADMISSION_CPUS:
        raise ValueError("CPU admission must cover the frozen CPUs and SMT siblings")
    frozen_topology = validate_cpu_topology_record(
        cpu_topology_record() if topology is None else topology
    )
    samples = [parse_proc_stat(reader(), selected)]
    windows: list[dict[str, object]] = []
    for ordinal in range(1, CPU_ADMISSION_WINDOWS + 1):
        sleeper(CPU_ADMISSION_INTERVAL_SECONDS)
        current = parse_proc_stat(reader(), selected)
        prior = samples[-1]
        cpu_rows: list[dict[str, object]] = []
        for cpu in selected:
            before, after = prior[cpu], current[cpu]
            if len(before.counters) != len(after.counters):
                raise RuntimeError(f"cpu{cpu} counter width changed")
            deltas = tuple(
                new - old
                for old, new in zip(before.counters, after.counters, strict=True)
            )
            if any(delta < 0 for delta in deltas):
                raise RuntimeError(f"cpu{cpu} counters regressed")
            total = sum(deltas)
            idle = deltas[3] + deltas[4]
            busy = total - idle
            if total <= 0 or busy < 0:
                fraction = math.inf if total <= 0 else busy / total
                raise RuntimeError(
                    f"cpu{cpu} failed counter-integrity admission in window "
                    f"{ordinal}: "
                    f"busy_fraction={fraction:.6f}"
                )
            fraction = busy / total
            if fraction > MAX_CPU_BUSY_FRACTION:
                raise RuntimeError(
                    f"cpu{cpu} failed counter-integrity admission in window "
                    f"{ordinal}: "
                    f"busy_fraction={fraction:.6f}"
                )
            cpu_rows.append(
                {
                    "cpu": cpu,
                    "counter_deltas": list(deltas),
                    "total_delta": total,
                    "idle_delta": idle,
                    "busy_delta": busy,
                    "busy_fraction": fraction,
                    "passed": True,
                }
            )
        windows.append(
            {
                "window": ordinal,
                "interval_seconds": CPU_ADMISSION_INTERVAL_SECONDS,
                "cpus": cpu_rows,
                "passed": True,
            }
        )
        samples.append(current)
    body = {
        "schema_version": 1,
        "source": "/proc/stat",
        "selected_cpus": list(selected),
        "engine_evaluator_cpus": list(ENGINE_CPUS),
        "smt_sibling_pairs": [list(pair) for pair in CPU_SIBLING_PAIRS],
        "cpu_topology": frozen_topology,
        "window_count": CPU_ADMISSION_WINDOWS,
        "interval_seconds": CPU_ADMISSION_INTERVAL_SECONDS,
        "max_busy_fraction": MAX_CPU_BUSY_FRACTION,
        "execution_contract": execution_contract_record(),
        "cpu_sampling_policy": cpu_sampling_policy_record(),
        "pass_semantics": "counter_integrity_only",
        "samples": [
            {str(cpu): sample[cpu].to_record() for cpu in selected}
            for sample in samples
        ],
        "windows": windows,
        "passed": True,
    }
    record = {**body, "admission_sha256": _record_sha256("cpu-admission", body)}
    return validate_cpu_admission_record(record, topology=frozen_topology)


def execution_contract_record() -> dict[str, object]:
    """Return the fail-closed claim boundary for this shared-host run."""

    return {
        "evidence_class": EXECUTION_CLASS,
        "authorized_scope": "objective_quality_only",
        "shared_host": True,
        "timing_data_role": "operational_observability_only",
        "timing_comparison_claim_authorized": False,
        "wall_clock_claim_authorized": False,
        "wall_clock_claims_allowed": WALL_CLOCK_CLAIMS_ALLOWED,
        "wall_clock_dominance_claim_authorized": False,
        "timing_exclusion_reason": TIMING_EXCLUSION_REASON,
    }


def validate_execution_contract(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError("shared-host execution claim boundary changed")
    record = copy.deepcopy(value)
    if record != execution_contract_record():
        raise RuntimeError("shared-host execution claim boundary changed")
    return record


def cpu_sampling_policy_record() -> dict[str, object]:
    return {
        "mode": "record_only_shared_host_load_observation",
        "window_count": CPU_ADMISSION_WINDOWS,
        "interval_seconds": CPU_ADMISSION_INTERVAL_SECONDS,
        "max_busy_fraction": MAX_CPU_BUSY_FRACTION,
        "shared_load_rejection_authorized": False,
        "pass_semantics": "counter_integrity_only",
    }


def validate_cpu_sampling_policy(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError("CPU sampling claim boundary changed")
    record = copy.deepcopy(value)
    if record != cpu_sampling_policy_record():
        raise RuntimeError("CPU sampling claim boundary changed")
    return record


def validate_cpu_admission_record(
    value: Mapping[str, object],
    *,
    topology: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate record-only sampling and recompute every utilization row."""

    record = copy.deepcopy(dict(value))
    body = {key: item for key, item in record.items() if key != "admission_sha256"}
    samples = record.get("samples")
    windows = record.get("windows")
    frozen_topology = validate_cpu_topology_record(record.get("cpu_topology", {}))
    if topology is not None and frozen_topology != validate_cpu_topology_record(
        topology
    ):
        raise RuntimeError("CPU admission topology differs from readiness")
    if not (
        record.get("schema_version") == 1
        and record.get("source") == "/proc/stat"
        and record.get("selected_cpus") == list(CPU_ADMISSION_CPUS)
        and record.get("engine_evaluator_cpus") == list(ENGINE_CPUS)
        and record.get("smt_sibling_pairs")
        == [list(pair) for pair in CPU_SIBLING_PAIRS]
        and record.get("window_count") == CPU_ADMISSION_WINDOWS
        and record.get("interval_seconds") == CPU_ADMISSION_INTERVAL_SECONDS
        and record.get("max_busy_fraction") == MAX_CPU_BUSY_FRACTION
        and record.get("pass_semantics") == "counter_integrity_only"
        and record.get("passed") is True
        and type(samples) is list
        and len(samples) == CPU_ADMISSION_WINDOWS + 1
        and type(windows) is list
        and len(windows) == CPU_ADMISSION_WINDOWS
        and record.get("admission_sha256") == _record_sha256("cpu-admission", body)
    ):
        raise RuntimeError("CPU admission record changed")
    validate_execution_contract(record.get("execution_contract"))
    validate_cpu_sampling_policy(record.get("cpu_sampling_policy"))

    parsed_samples: list[dict[int, tuple[int, ...]]] = []
    expected_keys = {str(cpu) for cpu in CPU_ADMISSION_CPUS}
    for sample in samples:
        if type(sample) is not dict or set(sample) != expected_keys:
            raise RuntimeError("CPU admission sample coverage changed")
        parsed: dict[int, tuple[int, ...]] = {}
        for cpu in CPU_ADMISSION_CPUS:
            row = sample[str(cpu)]
            counters = row.get("counters") if type(row) is dict else None
            if not (
                type(counters) is list
                and len(counters) >= 5
                and all(type(item) is int and item >= 0 for item in counters)
            ):
                raise RuntimeError("CPU admission sample counters changed")
            values = tuple(counters)
            if not (
                row.get("cpu") == cpu
                and row.get("total") == sum(values)
                and row.get("idle_plus_iowait") == values[3] + values[4]
            ):
                raise RuntimeError("CPU admission sample arithmetic changed")
            parsed[cpu] = values
        parsed_samples.append(parsed)

    for ordinal, window in enumerate(windows, start=1):
        if type(window) is not dict:
            raise RuntimeError("CPU admission window changed")
        rows = window.get("cpus")
        if not (
            window.get("window") == ordinal
            and window.get("interval_seconds") == CPU_ADMISSION_INTERVAL_SECONDS
            and window.get("passed") is True
            and type(rows) is list
            and len(rows) == len(CPU_ADMISSION_CPUS)
        ):
            raise RuntimeError("CPU admission window changed")
        prior = parsed_samples[ordinal - 1]
        current = parsed_samples[ordinal]
        for cpu, row in zip(CPU_ADMISSION_CPUS, rows, strict=True):
            before, after = prior[cpu], current[cpu]
            if len(before) != len(after):
                raise RuntimeError("CPU admission counter width changed")
            deltas = tuple(new - old for old, new in zip(before, after, strict=True))
            if any(delta < 0 for delta in deltas):
                raise RuntimeError("CPU admission counters regressed")
            total = sum(deltas)
            idle = deltas[3] + deltas[4]
            busy = total - idle
            if total <= 0 or busy < 0:
                raise RuntimeError("CPU admission counters are invalid")
            expected = {
                "cpu": cpu,
                "counter_deltas": list(deltas),
                "total_delta": total,
                "idle_delta": idle,
                "busy_delta": busy,
                "busy_fraction": busy / total,
                "passed": True,
            }
            if row != expected or expected["busy_fraction"] > MAX_CPU_BUSY_FRACTION:
                raise RuntimeError("CPU admission utilization row changed")
    return record


def _settings() -> AbcEvaluatorSettings:
    return AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=PILOT_CIRCUITS,
        affinity_sets=tuple((cpu,) for cpu in ENGINE_CPUS),
        per_circuit_timeout_s=float(PER_CANDIDATE_TIMEOUT_SECONDS),
    )


def validate_evaluator_provenance(value: Mapping[str, object]) -> dict[str, object]:
    provenance = copy.deepcopy(dict(value))
    circuits = provenance.get("circuits")
    if not (
        provenance.get("abc_binary_sha256") == CURRENT_ABC_SHA256
        and provenance.get("boils_source_commit") == BOILS_SOURCE_COMMIT
        and provenance.get("abc_source_identity") == f"git:{ABC_SOURCE_COMMIT}"
        and provenance.get("circuit_suite_identity") == f"git:{EPFL_SOURCE_COMMIT}"
        and type(circuits) is list
        and len(circuits) == 1
        and circuits[0].get("name") == "log2"
        and circuits[0].get("sha256") == EXPECTED_CIRCUIT_SHA256
        and provenance.get("lut_inputs") == 6
        and provenance.get("per_circuit_timeout_s")
        == float(PER_CANDIDATE_TIMEOUT_SECONDS)
        and provenance.get("affinity_sets") == [[cpu] for cpu in ENGINE_CPUS]
        and provenance.get("taskset_binary_sha256") == EXPECTED_TASKSET_SHA256
    ):
        raise RuntimeError("BOiLS evaluator provenance differs from frozen v5")
    return provenance


def evaluator_provenance_sha256(value: Mapping[str, object]) -> str:
    return _record_sha256("evaluator-provenance", validate_evaluator_provenance(value))


def _preview_seed(ids: DeterministicIdFactory) -> EvolutionCandidate:
    configuration = v5.PARENT_C_CONFIGURATION
    artifact_hash = hashlib.sha256(
        canonical_typed_json_bytes(configuration)
    ).hexdigest()
    occurrence = CandidateOccurrence(
        candidate_id=ids.new_candidate_id(),
        configuration_hash=typed_json_sha256(configuration),
        configuration_artifact_hash=artifact_hash,
        proposal_sequence=1,
    )
    return EvolutionCandidate(
        occurrence=occurrence,
        configuration=configuration,
        objectives=v5.PARENT_C_OBJECTIVES,
        valid=True,
        generation=0,
        label="seed_0",
    )


class _NeverCalledGenerator:
    async def propose(self, request: VariationGenerationRequest):  # pragma: no cover
        del request
        raise AssertionError("readiness preparation started a provider call")

    async def reflect(self, request: ReflectionGenerationRequest):  # pragma: no cover
        del request
        raise AssertionError("v5 readiness cannot reflect")


class _NoEvaluationBoundary:
    def evaluate(self, configuration: object):  # pragma: no cover
        del configuration
        raise AssertionError("readiness preparation started an evaluation")


def _prepared_record(
    prepared: PreparedInvocation,
    card_ids_by_reference: Mapping[object, str],
) -> dict[str, object]:
    if (
        prepared.call_id is None
        or prepared.proposal_authority is not ProposalAuthority.MODEL
    ):
        raise RuntimeError("readiness accepts model-authored preparations only")
    selected = prepared.variation_case.selected_insights
    body = {
        "label": prepared.plan.label,
        "phase": prepared.plan.phase,
        "operator_invocation_id": prepared.operator_invocation_id.value,
        "call_id": prepared.call_id.value,
        "candidate_id": prepared.candidate_id.value,
        "proposal_sequence": prepared.proposal_sequence,
        "prompt_sha256": hashlib.sha256(prepared.prompt.encode("utf-8")).hexdigest(),
        "selected_insights": [
            {
                "insight_id": reference.insight_id.value,
                "version": reference.version,
                "card_id": card_ids_by_reference.get(reference),
            }
            for reference in selected
        ],
        "atomic_replacement_option_hashes": [
            typed_json_sha256(option)
            for option in prepared.plan.atomic_replacement_options
        ],
    }
    return {**body, "preparation_sha256": _record_sha256("prepared-call", body)}


def prepare_readiness_manifest() -> dict[str, object]:
    """Freeze G1 decisions/prompts with isolated IDs, RNG, memory, and no I/O."""

    ids = DeterministicIdFactory(ID_NAMESPACE)
    memory, references = v5.build_v5_insight_memory(ids)
    seed = _preview_seed(ids)
    problem = BoilsAbcProblem(_settings(), evaluator=_NoEvaluationBoundary())
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=_NeverCalledGenerator(),
        id_factory=ids,
        memory=memory,
        seed=v5.ENGINE_RNG_SEED,
        initial_proposal_sequence=seed.occurrence.proposal_sequence,
        evaluator_concurrency=len(ENGINE_CPUS),
        prompt_builder=v5.BoilsV5RolePromptRouter(),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    archive = ParetoArchive(
        problem.objectives,
        evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
    )
    archive.consider(seed)
    snapshot = archive.snapshot()
    state = OptimizerState(
        generation=0,
        candidates=(seed,),
        archive=snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
        unique_evaluations=1,
        logical_llm_calls=0,
    )
    planner = BoilsBudgetedV5Planner(ids)
    plan = planner.plan(state, BUDGET)
    model_plans = tuple(
        slot.plan
        for slot in plan.slots
        if slot.proposal_authority is ProposalAuthority.MODEL
    )
    if len(model_plans) != 5:
        raise RuntimeError("readiness requires exactly five model plans")
    prepared, _ = engine.prepare_invocations(
        model_plans,
        reward_binding=plan.reward.binding,
    )
    card_ids = {reference: card_id for card_id, reference in references.entries}
    calls = tuple(_prepared_record(item, card_ids) for item in prepared)
    expected_assignments = {
        slot_id: card_id for slot_id, card_id in v5.EXPECTED_MEMORY_ASSIGNMENTS
    }
    actual_assignments = {
        str(row["label"]): row["selected_insights"][0]["card_id"]
        for row in calls
        if row["selected_insights"]
    }
    if actual_assignments != expected_assignments:
        raise RuntimeError(
            "readiness card assignments differ from frozen counterbalance"
        )
    g1_decision = planner.generation1_decision
    if g1_decision is None:
        raise RuntimeError("readiness planner did not freeze generation one")
    topology = cpu_topology_record()
    source_closure = source_closure_record()
    body = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "development_only": True,
        "post_hoc_development_protocol_correction": True,
        "execution_contract": execution_contract_record(),
        "cpu_sampling_policy": cpu_sampling_policy_record(),
        "protocol_correction": v5.protocol_correction_record(),
        "chronology": (
            "prepared_before_cpu_admission_then_written_and_fsynced_after_"
            "admission_before_physical_seed_or_provider"
        ),
        "model": MODEL,
        "provider_options": {"only": list(PROVIDER_ONLY)},
        "allowed_resolved_providers": list(ALLOWED_RESOLVED_PROVIDERS),
        "budget": BUDGET.to_trace_record(),
        "attempt4_pricing_envelope": attempt4_pricing_envelope_record(),
        "attempt4_mechanism_contract": attempt4_mechanism_contract_record(),
        "schema_repair_policy": schema_repair_policy_record(),
        "engine_cpus": list(ENGINE_CPUS),
        "cpu_admission_cpus": list(CPU_ADMISSION_CPUS),
        "cpu_topology": topology,
        "source_closure": source_closure,
        "memory": references.to_manifest_record(),
        "support": v5.support_manifest_record("0" * 64),
        "generation_one_decision": g1_decision.to_trace_record(),
        "prepared_model_calls": list(calls),
        "no_external_calls_or_evaluations": True,
    }
    return {**body, "readiness_sha256": _record_sha256("readiness", body)}


def validate_readiness_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on stale or internally inconsistent attempt-4 readiness."""

    if not isinstance(value, Mapping):
        raise TypeError("readiness manifest must be a mapping")
    record = copy.deepcopy(dict(value))
    body = {key: item for key, item in record.items() if key != "readiness_sha256"}
    if not (
        record.get("schema_version") == READINESS_SCHEMA_VERSION
        and record.get("run_id") == RUN_ID
        and record.get("readiness_sha256") == _record_sha256("readiness", body)
    ):
        raise RuntimeError("attempt4 readiness manifest identity changed")
    return record


def _actual_preparation_projection(event: Mapping[str, object]) -> dict[str, object]:
    selected = event.get("selected_insights")
    if type(selected) is not list:
        raise RuntimeError("invocation preparation omitted selected insight refs")
    body = {
        "label": event.get("label"),
        "phase": event.get("phase"),
        "operator_invocation_id": event.get("operator_invocation_id"),
        "call_id": event.get("call_id"),
        "candidate_id": event.get("candidate_id"),
        "proposal_sequence": event.get("proposal_sequence"),
        "prompt_sha256": event.get("prompt_sha256"),
        "selected_insights": [
            {
                "insight_id": row.get("insight_id"),
                "version": row.get("version"),
                "card_id": None,
            }
            for row in selected
            if type(row) is dict
        ],
        "atomic_replacement_option_hashes": event.get(
            "atomic_replacement_option_hashes"
        ),
    }
    return body


class ReadinessTraceVerifier:
    """Enable provider only after exact G1 decision and preparation replay."""

    def __init__(self, readiness: Mapping[str, object]) -> None:
        validate_readiness_manifest(readiness)
        validate_execution_contract(readiness.get("execution_contract"))
        validate_cpu_sampling_policy(readiness.get("cpu_sampling_policy"))
        rows = readiness.get("prepared_model_calls")
        memory = readiness.get("memory")
        generation_one = readiness.get("generation_one_decision")
        if (
            type(rows) is not list
            or len(rows) != 5
            or type(memory) is not dict
            or type(generation_one) is not dict
        ):
            raise ValueError("readiness manifest lacks five model calls or memory")
        card_rows = memory.get("card_references")
        if type(card_rows) is not list:
            raise ValueError("readiness memory lacks card references")
        self._card_by_ref = {
            (row["insight_id"], row["version"]): row["card_id"] for row in card_rows
        }
        self._expected = {str(row["label"]): copy.deepcopy(row) for row in rows}
        self._expected_generation_one = copy.deepcopy(generation_one)
        self._generation_one_verified = False
        self._seen: dict[str, dict[str, object]] = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def observe(self, event: Mapping[str, object]) -> None:
        if event.get("event_type") == "boils_v5_generation1_decided":
            if self._generation_one_verified:
                raise RuntimeError("generation-one decision was duplicated")
            actual_decision = {
                key: copy.deepcopy(value)
                for key, value in event.items()
                if key not in {"stream_sequence", "source_sequence", "domain"}
            }
            if actual_decision != self._expected_generation_one:
                raise RuntimeError(
                    "live generation-one decision differs from readiness"
                )
            self._generation_one_verified = True
            self._ready = len(self._seen) == len(self._expected)
            return
        if event.get("event_type") != "invocation_prepared":
            return
        if event.get("proposal_authority") != ProposalAuthority.MODEL.value:
            return
        actual = _actual_preparation_projection(event)
        label = actual["label"]
        if type(label) is not str or label not in self._expected:
            raise RuntimeError("unexpected model invocation before readiness closure")
        for selected in actual["selected_insights"]:
            key = (selected["insight_id"], selected["version"])
            selected["card_id"] = self._card_by_ref.get(key)
        actual["preparation_sha256"] = _record_sha256("prepared-call", actual)
        if actual != self._expected[label]:
            raise RuntimeError(f"live preparation differs from readiness: {label}")
        if label in self._seen:
            raise RuntimeError("model invocation preparation was duplicated")
        self._seen[label] = actual
        if self._generation_one_verified and len(self._seen) == len(self._expected):
            self._ready = True

    def assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("provider call attempted before readiness replay closed")


class TraceRecorder:
    def __init__(
        self,
        writer: DurableJsonlWriter,
        verifier: ReadinessTraceVerifier,
    ) -> None:
        self._writer = writer
        self._verifier = verifier
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(self, event: Mapping[str, object]) -> None:
        with self._lock:
            self._sequence += 1
            record = dict(event)
            source_sequence = record.pop("sequence", None)
            durable = {
                "stream_sequence": self._sequence,
                "source_sequence": source_sequence,
                "domain": "boils_abc_log2_length20_budgeted_v5",
                **record,
            }
            self._writer.write(durable)
            self._verifier.observe(durable)


class PreflightGuardedGenerator:
    """Refuse provider and reflection use outside the frozen v5 boundary."""

    def __init__(self, generator: AgenticGenerator, verifier: ReadinessTraceVerifier):
        if not isinstance(generator, AgenticGenerator):
            raise TypeError("generator must implement AgenticGenerator")
        self.generator = generator
        self.verifier = verifier

    async def propose(self, request: VariationGenerationRequest):
        self.verifier.assert_ready()
        return await self.generator.propose(request)

    async def reflect(self, request: ReflectionGenerationRequest):
        del request
        raise RuntimeError("budgeted v5 has no reflection call")


def telemetry_policy() -> AgenticTelemetryPolicy:
    return AgenticTelemetryPolicy(
        requested_model=MODEL,
        allowed_resolved_models=ALLOWED_RESOLVED_MODELS,
        allowed_resolved_providers=ALLOWED_RESOLVED_PROVIDERS,
        max_cost_usd=MAX_SUCCESSFUL_RESPONSE_COST_USD,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_reasoning_tokens=MAX_REASONING_TOKENS,
        max_attempt_count=QUEUE_MAX_ATTEMPTS,
    )


@dataclass(frozen=True, slots=True)
class DurableManifestBinding:
    readiness_path: Path
    readiness_file_sha256: str
    launch_path: Path
    launch_file_sha256: str
    source_snapshots: tuple[tuple[Path, str], ...] = ()
    source_bundle_path: Path | None = None
    source_bundle_record: Mapping[str, object] | None = None

    def verify(self) -> None:
        for path, expected in (
            (self.readiness_path, self.readiness_file_sha256),
            (self.launch_path, self.launch_file_sha256),
            *self.source_snapshots,
        ):
            if not path.is_file() or _file_sha256(path) != expected:
                raise RuntimeError("durable launch manifest identity changed")
        if (self.source_bundle_path is None) != (self.source_bundle_record is None):
            raise RuntimeError("source bundle binding is incomplete")
        if self.source_bundle_path is not None:
            assert self.source_bundle_record is not None
            verify_source_bundle(self.source_bundle_path, self.source_bundle_record)


def _candidate_summary(candidate: EvolutionCandidate | None) -> object:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id.value,
        "proposal_sequence": candidate.occurrence.proposal_sequence,
        "label": candidate.label,
        "generation": candidate.generation,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "objectives": candidate.objective_map,
        "valid": candidate.valid,
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "selected_insights": [
            {"insight_id": ref.insight_id.value, "version": ref.version}
            for ref in candidate.selected_insight_refs
        ],
    }


def _exact_objectives(candidate: EvolutionCandidate) -> dict[str, int]:
    values = candidate.objective_map
    if set(values) != {"total_lut_count", "total_levels"}:
        raise RuntimeError("analysis projection requires the exact BOiLS objectives")
    result: dict[str, int] = {}
    for name in ("total_lut_count", "total_levels"):
        value = values[name]
        if not math.isfinite(value) or value != int(value):
            raise RuntimeError("analysis projection requires exact integer objectives")
        result[name] = int(value)
    return result


def _single_parent_edit(
    candidate: EvolutionCandidate,
    *,
    expected_index: int,
    allowed_replacements: Sequence[str],
) -> dict[str, object]:
    configuration = candidate.configuration_dict
    sequence = configuration.get("sequence")
    if type(sequence) is not list or len(sequence) != len(v5.PARENT_C_SEQUENCE):
        raise RuntimeError("analysis projection received a malformed BOiLS sequence")
    differences = tuple(
        index
        for index, (parent, child) in enumerate(
            zip(v5.PARENT_C_SEQUENCE, sequence, strict=True)
        )
        if parent != child
    )
    if differences != (expected_index,):
        raise RuntimeError("G1 analysis row is not its exact singleton edit")
    replacement = sequence[expected_index]
    if type(replacement) is not str or replacement not in allowed_replacements:
        raise RuntimeError("G1 analysis edit is outside its frozen palette")
    return {"index": expected_index, "replacement": replacement}


def _offline_analysis_input(
    result: OptimizerResult,
    *,
    planner: BoilsBudgetedV5Planner,
    readiness: Mapping[str, object],
    protocol_acceptance_passed: bool,
) -> dict[str, object]:
    """Build the scorer-only projection without importing any analysis code."""

    if len(result.generation_receipts) != 2:
        raise RuntimeError("analysis projection requires two closed generations")
    g1_receipt, g2_receipt = result.generation_receipts
    g1_decision = planner.generation1_decision
    g2_decision = planner.generation2_decision
    if g1_decision is None or g2_decision is None:
        raise RuntimeError("analysis projection requires both planner decisions")

    g1_order = ("G1-A1", "G1-A2", "G1-D1", "G1-D2", "G1-U", "G1-X")
    g2_order = ("G2-E", "G2-X")
    if tuple(item.slot.slot_id for item in g1_receipt.slot_results) != g1_order:
        raise RuntimeError("analysis projection G1 slot order changed")
    if tuple(item.slot.slot_id for item in g2_receipt.slot_results) not in {
        (),
        ("G2-E",),
        g2_order,
    }:
        raise RuntimeError("analysis projection G2 slot order changed")

    slot_decisions = {item.slot_id: item for item in g1_decision.slots}

    def exact_replacements(slot_id: str) -> list[str]:
        replacements = [
            option.replacement for option in slot_decisions[slot_id].palette.palette
        ]
        if not replacements or any(type(value) is not str for value in replacements):
            raise RuntimeError("analysis palettes require exact string replacements")
        return replacements

    for index in (
        v5.AREA_PATH_INDEX,
        v5.DEPTH_PATH_INDEX,
        v5.UNCERTAINTY_PATH_INDEX,
        v5.COVERAGE_PATH_INDEX,
    ):
        if type(index) is not int:
            raise RuntimeError("analysis palette indices must be exact integers")
    coverage_replacements = exact_replacements("G1-X")
    if len(coverage_replacements) != 1:
        raise RuntimeError("analysis coverage palette must be an exact singleton")
    palette_spec = {
        "area": {
            "index": v5.AREA_PATH_INDEX,
            "replacements": exact_replacements("G1-A1"),
        },
        "depth": {
            "index": v5.DEPTH_PATH_INDEX,
            "replacements": exact_replacements("G1-D1"),
        },
        "uncertainty": {
            "index": v5.UNCERTAINTY_PATH_INDEX,
            "replacements": exact_replacements("G1-U"),
        },
        "coverage": {
            "index": v5.COVERAGE_PATH_INDEX,
            "replacement": coverage_replacements[0],
        },
    }
    path_by_slot = {
        "G1-A1": (v5.AREA_PATH_INDEX, palette_spec["area"]["replacements"]),
        "G1-A2": (v5.AREA_PATH_INDEX, palette_spec["area"]["replacements"]),
        "G1-D1": (v5.DEPTH_PATH_INDEX, palette_spec["depth"]["replacements"]),
        "G1-D2": (v5.DEPTH_PATH_INDEX, palette_spec["depth"]["replacements"]),
        "G1-U": (
            v5.UNCERTAINTY_PATH_INDEX,
            palette_spec["uncertainty"]["replacements"],
        ),
        "G1-X": (
            v5.COVERAGE_PATH_INDEX,
            [palette_spec["coverage"]["replacement"]],
        ),
    }
    g1_slots: list[dict[str, object]] = []
    g1_slot_by_candidate_id: dict[str, str] = {}
    g1_result_by_slot = {item.slot.slot_id: item for item in g1_receipt.slot_results}
    for slot_id in g1_order:
        slot_result = g1_result_by_slot[slot_id]
        candidate = slot_result.outcome.candidate
        if candidate is None:
            raise RuntimeError("analysis projection cannot encode a missing G1 arm")
        index, allowed = path_by_slot[slot_id]
        g1_slots.append(
            {
                "slot_id": slot_id,
                "proposal_authority": slot_result.slot.proposal_authority.value,
                "edit": _single_parent_edit(
                    candidate,
                    expected_index=index,
                    allowed_replacements=allowed,
                ),
                "objectives": _exact_objectives(candidate),
                "typed_json_configuration_sha256": (
                    candidate.occurrence.configuration_hash
                ),
            }
        )
        g1_slot_by_candidate_id[candidate.candidate_id.value] = slot_id

    readiness_calls = readiness.get("prepared_model_calls")
    readiness_memory = readiness.get("memory")
    if type(readiness_calls) is not list or type(readiness_memory) is not dict:
        raise RuntimeError("analysis projection lacks readiness assignments")
    reference_rows = readiness_memory.get("card_references")
    if type(reference_rows) is not list:
        raise RuntimeError("analysis projection lacks card reference identities")
    reference_by_card = {
        row["card_id"]: (row["insight_id"], row["version"]) for row in reference_rows
    }
    card_definition_by_id = {
        definition.card_id: definition for definition in v5.INSIGHT_CARD_DEFINITIONS
    }
    treatment_assignments: list[dict[str, object]] = []
    for slot_id in ("G1-A1", "G1-A2", "G1-D1", "G1-D2"):
        matches = tuple(row for row in readiness_calls if row.get("label") == slot_id)
        if len(matches) != 1 or len(matches[0].get("selected_insights", ())) != 1:
            raise RuntimeError("analysis projection card assignment is ambiguous")
        card_id = matches[0]["selected_insights"][0].get("card_id")
        definition = card_definition_by_id.get(card_id)
        candidate = g1_result_by_slot[slot_id].outcome.candidate
        if definition is None or candidate is None:
            raise RuntimeError("analysis projection card assignment is unknown")
        selected_refs = tuple(
            (reference.insight_id.value, reference.version)
            for reference in candidate.selected_insight_refs
        )
        if selected_refs != (reference_by_card[card_id],):
            raise RuntimeError("live candidate/card assignment differs from readiness")
        treatment_assignments.append(
            {
                "stratum_id": ("area" if definition.role is v5.AREA_ROLE else "depth"),
                "slot_id": slot_id,
                "treatment": (
                    "real"
                    if definition.treatment is v5.CardTreatment.REAL
                    else "placebo"
                ),
            }
        )
    for stratum in ("area", "depth"):
        if sorted(
            row["treatment"]
            for row in treatment_assignments
            if row["stratum_id"] == stratum
        ) != ["placebo", "real"]:
            raise RuntimeError("analysis projection treatment counterbalance changed")

    selected_pair_rows = tuple(
        row
        for row in (g2_decision.selection.exploit, g2_decision.selection.coverage)
        if row is not None
    )
    actual_g2 = {item.slot.slot_id: item for item in g2_receipt.slot_results}
    if len(selected_pair_rows) != len(actual_g2):
        raise RuntimeError("analysis projection G2 selection cardinality changed")
    g2_slots: list[dict[str, object]] = []
    for ordinal, slot_id in enumerate(g2_order):
        slot_result = actual_g2.get(slot_id)
        if slot_result is None:
            g2_slots.append(
                {
                    "slot_id": slot_id,
                    "branch_slot_ids": [],
                    "objectives": None,
                    "typed_json_configuration_sha256": None,
                    "branch_preservation_verified": False,
                    "provider_telemetry_present": False,
                    "skipped": True,
                }
            )
            continue
        candidate = slot_result.outcome.candidate
        materialized = slot_result.slot.materialized
        pair = selected_pair_rows[ordinal].pair
        if candidate is None or materialized is None:
            raise RuntimeError("analysis projection G2 materialization is missing")
        parent_ids = tuple(
            parent.candidate_id.value for parent in slot_result.slot.plan.parents
        )
        pair_ids = tuple(value.value for value in pair.pair_ids)
        branch_slots = [g1_slot_by_candidate_id.get(value) for value in parent_ids]
        if not (
            parent_ids == pair_ids
            and all(type(value) is str for value in branch_slots)
            and candidate.preservation_verified is True
            and candidate.call_telemetry is None
            and pair.target_configuration_sha256
            == candidate.occurrence.configuration_hash
            and pair.materialization_receipt_sha256
            == materialized.materialization_receipt_hash
            and materialized.candidate_id == candidate.candidate_id
        ):
            raise RuntimeError("analysis projection G2 replay receipt mismatch")
        g2_slots.append(
            {
                "slot_id": slot_id,
                "branch_slot_ids": branch_slots,
                "objectives": _exact_objectives(candidate),
                "typed_json_configuration_sha256": (
                    candidate.occurrence.configuration_hash
                ),
                "branch_preservation_verified": True,
                "provider_telemetry_present": False,
                "skipped": False,
            }
        )

    return {
        "schema_id": "boils_abc_budgeted_v5_analysis_input_v2",
        "agentic_model_id": MODEL,
        "development_only": True,
        "post_hoc_development_protocol_correction": True,
        "execution_contract": validate_execution_contract(
            readiness.get("execution_contract")
        ),
        "protocol_correction": v5.protocol_correction_record(),
        "protocol_acceptance_passed": protocol_acceptance_passed,
        "palette_spec": palette_spec,
        "g1_slots": g1_slots,
        "treatment_assignments": treatment_assignments,
        "g2_slots": g2_slots,
    }


def _result_summary(
    result: OptimizerResult,
    *,
    memory: InsightMemoryBank,
    planner: BoilsBudgetedV5Planner,
    verifier: ReadinessTraceVerifier,
    provenance_sha256: str,
    readiness: Mapping[str, object],
    evaluation_cache: Mapping[str, int | None],
) -> dict[str, object]:
    execution_contract = validate_execution_contract(
        readiness.get("execution_contract")
    )
    cpu_sampling_policy = validate_cpu_sampling_policy(
        readiness.get("cpu_sampling_policy")
    )
    receipts = result.generation_receipts
    g1 = receipts[0] if len(receipts) > 0 else None
    g2 = receipts[1] if len(receipts) > 1 else None
    all_slots = tuple(slot for receipt in receipts for slot in receipt.slot_results)
    missing_g1_slot_ids = tuple(
        item.slot.slot_id
        for item in (() if g1 is None else g1.slot_results)
        if item.outcome.candidate is None
    )

    def accepted_slot(item: object) -> bool:
        outcome = getattr(item, "outcome", None)
        candidate = None if outcome is None else outcome.candidate
        return bool(
            candidate is not None
            and outcome.call_failure_type is None
            and candidate.valid
            and candidate.operator_compliant
            and candidate.evidence_compliant
        )

    costs = tuple(
        slot.outcome.candidate.call_telemetry.cost_usd
        for slot in all_slots
        if slot.outcome.candidate is not None
        and slot.outcome.candidate.call_telemetry is not None
    )
    if any(cost is None for cost in costs):
        raise RuntimeError("accepted model candidate has missing cost telemetry")
    accepted_cost = sum((cost for cost in costs if cost is not None), Decimal("0"))
    gates = {
        "readiness_replayed_before_provider": verifier.ready,
        "two_generations_completed": len(receipts) == 2,
        "five_logical_model_calls": result.final_state.logical_llm_calls == 5,
        "unique_evaluations_within_cap": result.final_state.unique_evaluations <= 9,
        "generation_one_has_six_slots": g1 is not None and len(g1.slot_results) == 6,
        "generation_one_six_valid_compliant": (
            g1 is not None
            and len(g1.slot_results) == 6
            and all(accepted_slot(item) for item in g1.slot_results)
        ),
        "generation_two_is_typed_engine_only_or_skip": (
            g2 is not None
            and 0 <= len(g2.slot_results) <= 2
            and g2.reserved_logical_llm_calls == 0
            and all(
                item.slot.proposal_authority is ProposalAuthority.ENGINE
                for item in g2.slot_results
            )
        ),
        "generation_two_non_skipped_valid_compliant": (
            g2 is not None and all(accepted_slot(item) for item in g2.slot_results)
        ),
        "no_missing_or_rejected_slot": all(accepted_slot(item) for item in all_slots),
        "four_randomized_memory_trials": len(memory.trials) == 4,
        "accepted_terminal_cost_within_envelope": (
            accepted_cost <= MAX_ACCEPTED_TERMINAL_RESPONSE_COST_USD
        ),
        "evaluation_cache_closed_and_exact": (
            evaluation_cache.get("misses") == result.final_state.unique_evaluations
            and evaluation_cache.get("in_flight") == 0
        ),
    }
    accepted = all(gates.values())
    mechanism_gates = {
        "two_distinct_g2_unions_materialized": (
            g2 is not None and len(g2.slot_results) == 2
        ),
        "at_least_one_g2_union_positive_reward": bool(
            g2 is not None and any(item.outcome.reward > 0 for item in g2.slot_results)
        ),
    }
    offline_analysis_input = (
        _offline_analysis_input(
            result,
            planner=planner,
            readiness=readiness,
            protocol_acceptance_passed=True,
        )
        if accepted
        else None
    )
    summary = {
        "schema_version": 1,
        "status": "succeeded" if accepted else "protocol_rejected",
        "completed_at_utc": _utc_now(),
        "development_only": True,
        "post_hoc_development_protocol_correction": True,
        "execution_contract": execution_contract,
        "cpu_sampling_policy": cpu_sampling_policy,
        "protocol_correction": v5.protocol_correction_record(),
        "attempt4_mechanism_contract": attempt4_mechanism_contract_record(),
        "failed_slot_continuation": {
            "missing_g1_slot_ids": list(missing_g1_slot_ids),
            "substitution_allowed": False,
            "g2_checkpoint_closed": g2 is not None,
            "protocol_acceptance_requires_no_missing_slot": True,
        },
        "claim_boundary": (
            "Shared-host quality-only post-hoc BOiLS/log2 workflow kill test; "
            "timing and latency are operational observability only; no timing "
            "comparison, wall-clock, wall-clock-dominance, held-out, SOTA, or "
            "genericity claim."
        ),
        "protocol_acceptance_passed": accepted,
        "result_sha256": result.result_hash,
        "final_archive_snapshot_sha256": result.final_state.archive_snapshot_hash,
        "evaluator_provenance_sha256": provenance_sha256,
        "resources": {
            "logical_llm_calls": result.final_state.logical_llm_calls,
            "unique_physical_evaluations": result.final_state.unique_evaluations,
            "generation_count": result.final_state.generation,
            "accepted_terminal_response_cost_usd": str(accepted_cost),
            "accepted_terminal_response_cost_ceiling_usd": str(
                MAX_ACCEPTED_TERMINAL_RESPONSE_COST_USD
            ),
            "potentially_billable_attempt_count_ceiling": (
                BUDGET.max_logical_llm_calls * QUEUE_MAX_ATTEMPTS
            ),
            "potentially_billable_attempt_cost_envelope_usd": str(
                MAX_POTENTIALLY_BILLABLE_ATTEMPT_COST_USD
            ),
            "billing_caveat": (
                "Failed provider attempts may be billed without terminal cost "
                "telemetry; the larger attempt-level value is an envelope, not "
                "an observed or guaranteed billed total."
            ),
            "evaluation_cache": copy.deepcopy(dict(evaluation_cache)),
        },
        "generations": [
            {
                "generation": receipt.generation,
                "plan_sha256": receipt.plan_hash,
                "receipt_sha256": receipt.receipt_hash,
                "reserved_logical_llm_calls": receipt.reserved_logical_llm_calls,
                "reserved_unique_evaluations": receipt.reserved_unique_evaluations,
                "slots": [
                    {
                        "slot_id": item.slot.slot_id,
                        "role": item.slot.role,
                        "proposal_authority": item.slot.proposal_authority.value,
                        "reward": item.outcome.reward,
                        "call_failure_type": item.outcome.call_failure_type,
                        "candidate": _candidate_summary(item.outcome.candidate),
                    }
                    for item in receipt.slot_results
                ],
            }
            for receipt in receipts
        ],
        "front": [
            _candidate_summary(candidate)
            for candidate in result.final_state.archive.front_candidates
        ],
        "memory": {
            "entry_count": len(memory.entries),
            "trial_count": len(memory.trials),
        },
        "planner": planner.to_summary_record(),
        "gates": gates,
        "post_run_mechanism_gates": mechanism_gates,
        "offline_analysis_input": offline_analysis_input,
    }
    return summary


async def run_workflow(
    *,
    problem: BoilsAbcProblem,
    generator: AgenticGenerator,
    evaluator_provenance_sha256_value: str,
    readiness: Mapping[str, object],
    manifests: DurableManifestBinding,
    event_writer: DurableJsonlWriter,
) -> dict[str, object]:
    """Execute the exact seed, five-call G1, and engine-only G2."""

    validate_readiness_manifest(readiness)
    validate_execution_contract(readiness.get("execution_contract"))
    validate_cpu_sampling_policy(readiness.get("cpu_sampling_policy"))
    manifests.verify()  # Must close before the first physical seed evaluation.
    ids = DeterministicIdFactory(ID_NAMESPACE)
    memory, references = v5.build_v5_insight_memory(ids)
    if references.to_manifest_record() != readiness.get("memory"):
        raise RuntimeError("live card identities differ from readiness")
    verifier = ReadinessTraceVerifier(readiness)
    trace = TraceRecorder(event_writer, verifier)
    gated = TelemetryGatedAgenticGenerator(generator, telemetry_policy())
    guarded = PreflightGuardedGenerator(gated, verifier)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=guarded,
        id_factory=ids,
        memory=memory,
        seed=v5.ENGINE_RNG_SEED,
        evaluator_concurrency=len(ENGINE_CPUS),
        trace_sink=trace.emit,
        prompt_builder=v5.BoilsV5RolePromptRouter(),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    planner = BoilsBudgetedV5Planner(ids, decision_sink=trace.emit)
    archive = ParetoArchive(
        engine.objectives,
        evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
    )
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=archive,
        planner=planner,
        budget=BUDGET,
        seed_admission_policy=v5.ExactCSeedAdmissionPolicy(
            evaluator_provenance_sha256_value
        ),
        trace_sink=trace.emit,
    )
    result = await optimizer.run((v5.parent_c_config(),))
    evaluation_cache = await engine.evaluation_cache_snapshot()
    return _result_summary(
        result,
        memory=memory,
        planner=planner,
        verifier=verifier,
        provenance_sha256=evaluator_provenance_sha256_value,
        readiness=readiness,
        evaluation_cache=evaluation_cache,
    )


def _behavior_source_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "support": Path(v5.__file__).resolve(),
        "planner": Path(
            sys.modules[BoilsBudgetedV5Planner.__module__].__file__
        ).resolve(),
        "engine": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/agentic_evolution.py",
        "optimizer": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/budgeted_optimizer.py",
        "telemetry_gate": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/gated_agentic_generator.py",
        "queue": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
        "evaluator": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
        "problem": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/problem_def.py",
        "variation_catalog": AGENT_EVOLVE_ROOT
        / "examples/benchmarks/boils_abc/variation_catalog.py",
        "insight_memory": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/insight_memory.py",
        "pareto_archive": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/pareto_archive.py",
        "materialized_variation": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/materialized_variation.py",
        "llm_task_queue": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/llm_task_queue.py",
        "task_keyed_palette": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/task_keyed_palette.py",
        "disjoint_pairs": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/disjoint_pairs.py",
        "disjoint_recombination": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/variation/disjoint_recombination.py",
        "frozen_reward": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/reward/frozen_archive.py",
        "llm_backoff": AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/llm_backoff.py",
        "agentic_adapter": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
        "provider_adapter": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
    }


def _source_hashes() -> dict[str, str]:
    sources = {
        **_behavior_source_paths(),
        "pricing_snapshot": PRICING_SNAPSHOT_PATH,
        "endpoint_capability_snapshot": CAPABILITY_SNAPSHOT_PATH,
    }
    return {name: _file_sha256(path) for name, path in sources.items()}


def _runtime_dependency_versions() -> dict[str, str | None]:
    """Record executed distributions in addition to lockfile intent."""

    versions: dict[str, str | None] = {}
    for distribution in ("pydantic-ai", "pydantic", "openai", "httpx", "uv"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _validate_attempt3_pricing_snapshot(pricing: object) -> None:
    """Validate the immutable attempt3 source snapshot without mutating it."""

    if type(pricing) is not dict:
        raise RuntimeError("StreamLake pricing snapshot semantics changed")
    model = pricing.get("model")
    selected_endpoint = pricing.get("selected_endpoint")
    derivation = pricing.get("attempt3_gate_derivation")
    if not (
        type(model) is dict
        and model.get("requested_slug") == MODEL
        and model.get("canonical_slug") == CANONICAL_RESOLVED_MODEL
        and type(selected_endpoint) is dict
        and selected_endpoint.get("name") == SELECTED_ENDPOINT_NAME
        and selected_endpoint.get("provider_name") == "StreamLake"
        and selected_endpoint.get("provider_request_slug") == PROVIDER_ONLY[0]
        and selected_endpoint.get("pricing_usd_per_token")
        == {
            "prompt": format(PROMPT_PRICE_USD_PER_TOKEN, "f"),
            "completion": format(COMPLETION_PRICE_USD_PER_TOKEN, "f"),
            "input_cache_read": format(CACHE_READ_PRICE_USD_PER_TOKEN, "f"),
        }
        and type(derivation) is dict
        and derivation.get("provider_routing")
        == {"eligible_provider_count": 1, "only": list(PROVIDER_ONLY)}
        and derivation.get("max_input_tokens_per_call") == MAX_INPUT_TOKENS
        and derivation.get("max_output_tokens_per_call") == 640
        and derivation.get("max_reasoning_tokens_per_call") == 640
        and derivation.get("max_logical_calls") == BUDGET.max_logical_llm_calls
        and derivation.get("derived_max_cost_usd_per_call") == "0.008960304"
        and derivation.get("frozen_cost_ceiling_usd_per_call")
        == str(MAX_SUCCESSFUL_RESPONSE_COST_USD)
        and derivation.get("derived_accepted_run_ceiling_usd")
        == str(MAX_ACCEPTED_TERMINAL_RESPONSE_COST_USD)
        and derivation.get("ten_potentially_billable_attempt_envelope_usd")
        == str(MAX_POTENTIALLY_BILLABLE_ATTEMPT_COST_USD)
        and attempt4_pricing_envelope_record()["derived_max_cost_usd_per_call"]
        == str(DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD)
    ):
        raise RuntimeError("StreamLake pricing snapshot semantics changed")


def _validate_attempt3_capability_snapshot(capability: object) -> None:
    if type(capability) is not dict:
        raise RuntimeError("StreamLake capability snapshot semantics changed")
    selected_endpoint = capability.get("selected_endpoint")
    provider_registry = capability.get("provider_registry")
    relevance = capability.get("attempt3_relevance")
    if not (
        capability.get("requested_model_alias") == MODEL
        and capability.get("canonical_model_slug") == CANONICAL_RESOLVED_MODEL
        and type(selected_endpoint) is dict
        and selected_endpoint.get("name") == SELECTED_ENDPOINT_NAME
        and selected_endpoint.get("provider_name") == "StreamLake"
        and selected_endpoint.get("provider_request_slug") == PROVIDER_ONLY[0]
        and type(selected_endpoint.get("supported_parameters")) is list
        and set(REQUIRED_ENDPOINT_CAPABILITIES).issubset(
            selected_endpoint["supported_parameters"]
        )
        and type(provider_registry) is dict
        and provider_registry.get("name") == "StreamLake"
        and provider_registry.get("slug") == PROVIDER_ONLY[0]
        and type(relevance) is dict
        and relevance.get("provider_only") == list(PROVIDER_ONLY)
        and relevance.get("allowed_resolved_models") == list(ALLOWED_RESOLVED_MODELS)
        and relevance.get("allowed_resolved_providers")
        == list(ALLOWED_RESOLVED_PROVIDERS)
        and relevance.get("required_capabilities_present")
        == list(REQUIRED_ENDPOINT_CAPABILITIES)
        and "provider_order" not in relevance
        and "allow_fallbacks" not in relevance
    ):
        raise RuntimeError("StreamLake capability snapshot semantics changed")


def _launch_manifest(
    *,
    readiness: Mapping[str, object],
    readiness_file_sha256: str,
    admission: Mapping[str, object],
    provenance: Mapping[str, object],
    provenance_sha256: str,
    source_snapshots: Mapping[str, Mapping[str, object]],
    source_bundle_path: Path,
    source_bundle_record: Mapping[str, object],
) -> dict[str, object]:
    validate_readiness_manifest(readiness)
    execution_contract = validate_execution_contract(
        readiness.get("execution_contract")
    )
    cpu_sampling_policy = validate_cpu_sampling_policy(
        readiness.get("cpu_sampling_policy")
    )
    pricing_envelope = attempt4_pricing_envelope_record()
    if readiness.get("attempt4_pricing_envelope") != pricing_envelope:
        raise RuntimeError("attempt4 pricing envelope changed after readiness")
    mechanism_contract = attempt4_mechanism_contract_record()
    if readiness.get("attempt4_mechanism_contract") != mechanism_contract:
        raise RuntimeError("attempt4 mechanism contract changed after readiness")
    repair_policy = schema_repair_policy_record()
    if readiness.get("schema_repair_policy") != repair_policy:
        raise RuntimeError("schema-repair policy changed after readiness")
    validated_admission = validate_cpu_admission_record(
        admission, topology=readiness.get("cpu_topology", {})
    )
    if not (
        validated_admission.get("execution_contract") == execution_contract
        and validated_admission.get("cpu_sampling_policy") == cpu_sampling_policy
    ):
        raise RuntimeError("shared-host execution claim boundary changed")
    if _file_sha256(PRICING_SNAPSHOT_PATH) != EXPECTED_PRICING_SNAPSHOT_SHA256:
        raise RuntimeError("StreamLake pricing snapshot identity changed")
    if _file_sha256(CAPABILITY_SNAPSHOT_PATH) != (EXPECTED_CAPABILITY_SNAPSHOT_SHA256):
        raise RuntimeError("StreamLake capability snapshot identity changed")
    pricing = json.loads(PRICING_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    _validate_attempt3_pricing_snapshot(pricing)
    capability = json.loads(CAPABILITY_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    _validate_attempt3_capability_snapshot(capability)
    live_source_hashes = _source_hashes()
    frozen_source_closure = validate_source_closure_record(
        readiness.get("source_closure", {})
    )
    if not (
        source_bundle_record.get("entry_count") == frozen_source_closure["entry_count"]
        and source_bundle_record.get("entries") == frozen_source_closure["entries"]
    ):
        raise RuntimeError("durable source bundle differs from frozen readiness")
    for name, snapshot in source_snapshots.items():
        if live_source_hashes.get(name) != snapshot.get("sha256"):
            raise RuntimeError(
                f"behavior source changed after durable snapshot: {name}"
            )
    verify_source_bundle(source_bundle_path, source_bundle_record)
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "started_at_utc": _utc_now(),
        "development_only": True,
        "post_hoc_development_protocol_correction": True,
        "execution_contract": execution_contract,
        "cpu_sampling_policy": cpu_sampling_policy,
        "protocol_correction": v5.protocol_correction_record(),
        "attempt4_mechanism_contract": mechanism_contract,
        "readiness_sha256": readiness["readiness_sha256"],
        "readiness_file_sha256": readiness_file_sha256,
        "resource_admission": validated_admission,
        "evaluator_provenance": copy.deepcopy(dict(provenance)),
        "evaluator_provenance_sha256": provenance_sha256,
        "support": v5.support_manifest_record(provenance_sha256),
        "model": MODEL,
        "allowed_resolved_models": list(ALLOWED_RESOLVED_MODELS),
        "provider": "openrouter",
        "provider_options": {"only": list(PROVIDER_ONLY)},
        "allowed_resolved_providers": list(ALLOWED_RESOLVED_PROVIDERS),
        "queue": {
            "max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "max_pending": QUEUE_MAX_PENDING,
            "max_attempts": QUEUE_MAX_ATTEMPTS,
            "attempt_timeout_seconds": QUEUE_ATTEMPT_TIMEOUT_SECONDS,
            "base_backoff_seconds": QUEUE_BASE_BACKOFF_SECONDS,
            "max_backoff_seconds": QUEUE_MAX_BACKOFF_SECONDS,
            "jitter": {
                "kind": "task_keyed_sha256",
                "seed": JITTER_SEED,
                "domain": JITTER_DOMAIN,
            },
            "terminal_outcome_publication": "required_fsync_before_downstream",
            "missingness": "continue_eligible_g2_without_substitution",
            "schema_repair_policy": repair_policy,
        },
        "telemetry_policy": telemetry_policy().to_trace_record(),
        "cost_envelopes": {
            "derived_conservative_per_response_usd": str(
                DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD
            ),
            "per_accepted_terminal_response_usd": str(MAX_SUCCESSFUL_RESPONSE_COST_USD),
            "five_accepted_terminal_responses_usd": str(
                MAX_ACCEPTED_TERMINAL_RESPONSE_COST_USD
            ),
            "ten_potentially_billable_attempts_usd": str(
                MAX_POTENTIALLY_BILLABLE_ATTEMPT_COST_USD
            ),
            "pricing_snapshot_sha256": EXPECTED_PRICING_SNAPSHOT_SHA256,
            "attempt4_derived_envelope": pricing_envelope,
        },
        "endpoint_capability_snapshot": {
            "sha256": EXPECTED_CAPABILITY_SNAPSHOT_SHA256,
            "requested_model_alias": MODEL,
            "canonical_model_slug": CANONICAL_RESOLVED_MODEL,
            "provider_request_slug": PROVIDER_ONLY[0],
            "provider_name": "StreamLake",
            "provider_registry_slug": PROVIDER_ONLY[0],
            "provider_options": {"only": list(PROVIDER_ONLY)},
            "required_capabilities": list(REQUIRED_ENDPOINT_CAPABILITIES),
        },
        "budget": BUDGET.to_trace_record(),
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "source_sha256": live_source_hashes,
        "durable_source_snapshots": copy.deepcopy(dict(source_snapshots)),
        "durable_source_bundle": copy.deepcopy(dict(source_bundle_record)),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "credential_variable": "OPENROUTER_API_KEY",
            "runtime_dependency_versions": _runtime_dependency_versions(),
        },
    }


async def _run_live(
    *,
    problem: BoilsAbcProblem,
    provenance_sha256: str,
    readiness: Mapping[str, object],
    manifests: DurableManifestBinding,
    event_writer: DurableJsonlWriter,
    queue_writer: DurableJsonlWriter,
) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=MODEL,
        max_connections=QUEUE_MAX_IN_FLIGHT,
        timeout_seconds=float(QUEUE_ATTEMPT_TIMEOUT_SECONDS),
        provider_options={"only": list(PROVIDER_ONLY)},
        app_title="AgentEvolve AAAI 2027 BOiLS budgeted optimizer v5",
    )
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=QUEUE_MAX_IN_FLIGHT,
        max_pending=QUEUE_MAX_PENDING,
        max_attempts=QUEUE_MAX_ATTEMPTS,
        attempt_timeout_ns=QUEUE_ATTEMPT_TIMEOUT_SECONDS * 1_000_000_000,
        base_backoff_ns=QUEUE_BASE_BACKOFF_SECONDS * 1_000_000_000,
        max_backoff_ns=QUEUE_MAX_BACKOFF_SECONDS * 1_000_000_000,
        attempt_request_policy=SchemaRepairAttemptPolicy(),
        jitter_policy=DeterministicHashJitter(
            seed=JITTER_SEED,
            domain=JITTER_DOMAIN,
        ),
        close_generator=True,
        outcome_sink=lambda outcome: queue_writer.write(
            structured_generation_outcome_record(outcome)
        ),
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
    )
    async with runner:
        return await run_workflow(
            problem=problem,
            generator=PydanticAIAgenticGenerator(runner),
            evaluator_provenance_sha256_value=provenance_sha256,
            readiness=readiness,
            manifests=manifests,
            event_writer=event_writer,
        )


def _finalize(run_dir: Path, status: str) -> None:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(run_dir.iterdir()):
        if path.name == "finalized.json" or not path.is_file():
            continue
        payload = path.read_bytes()
        record: dict[str, object] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if path.suffix == ".jsonl":
            record["lines"] = len(payload.splitlines())
        files[path.name] = record
    durable_write_json(
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
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return parser


def prepare_pre_directory_admission(
    run_dir: Path,
    *,
    readiness_builder: Callable[[], dict[str, object]] = prepare_readiness_manifest,
    topology_builder: Callable[[], dict[str, object]] = cpu_topology_record,
    admission_sampler: Callable[..., dict[str, object]] = sample_cpu_admission,
    source_closure_builder: Callable[[], dict[str, object]] = source_closure_record,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Prepare the frozen plan and admit resources without writing a run dir."""

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be a Path")
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    readiness = readiness_builder()
    validate_readiness_manifest(readiness)
    validate_execution_contract(readiness.get("execution_contract"))
    validate_cpu_sampling_policy(readiness.get("cpu_sampling_policy"))
    frozen_source_closure = validate_source_closure_record(
        readiness.get("source_closure", {})
    )
    topology = topology_builder()
    if topology != readiness.get("cpu_topology"):
        raise RuntimeError("CPU topology changed after readiness preparation")
    admission = validate_cpu_admission_record(
        admission_sampler(topology=topology), topology=topology
    )
    if run_dir.exists():
        raise FileExistsError(f"run directory appeared during admission: {run_dir}")
    if source_closure_builder() != frozen_source_closure:
        raise RuntimeError("source closure changed during CPU admission")
    return readiness, topology, admission


def main() -> None:
    args = _parser().parse_args()
    run_dir = args.log_root.resolve() / RUN_ID

    # Resource admission is deliberately pre-directory.  On a shared host a
    # transiently busy CPU is an operational retry, not a consumed scientific
    # run ID.  Prompt/card/plan preparation remains pure and precedes admission.
    readiness, live_topology, admission = prepare_pre_directory_admission(run_dir)

    status = "failed"
    directory_owned = False
    event_writer: DurableJsonlWriter | None = None
    queue_writer: DurableJsonlWriter | None = None
    evaluation_writer: DurableJsonlWriter | None = None
    try:
        try:
            durable_mkdir(run_dir)
        except DurableDirectoryPublishError:
            directory_owned = True
            raise
        directory_owned = True

        # No provider, filesystem evaluator, or physical seed has been touched.
        readiness_path = run_dir / "readiness_manifest.json"
        readiness_file_sha256 = durable_write_json(readiness_path, readiness)

        snapshot_sources = _behavior_source_paths()
        snapshot_destinations = {
            name: run_dir / f"source_snapshot_{name}.py" for name in snapshot_sources
        }
        source_snapshots: dict[str, dict[str, object]] = {}
        for name, source in snapshot_sources.items():
            destination = snapshot_destinations[name]
            digest = durable_copy_file(source, destination)
            if digest != _file_sha256(source):
                raise RuntimeError("durable source snapshot differs from live source")
            source_snapshots[name] = {
                "source": str(source),
                "snapshot": destination.name,
                "sha256": digest,
            }
        source_bundle_path = run_dir / "source_bundle.tar"
        source_bundle_record = durable_source_bundle(source_bundle_path)

        # Admission already passed before run-directory creation.  Revalidate
        # immutable topology after source persistence, then publish the receipt.
        if cpu_topology_record() != live_topology:
            raise RuntimeError("CPU topology changed after resource admission")
        durable_write_json(run_dir / "resource_admission.json", admission)

        evaluation_writer = DurableJsonlWriter(run_dir / "evaluations.jsonl")
        recorder = EvaluationObservationRecorder(evaluation_writer)
        evaluator = BoilsAbcEvaluator(_settings(), observer=recorder)
        provenance = validate_evaluator_provenance(evaluator.provenance())
        provenance_sha256 = evaluator_provenance_sha256(provenance)
        problem = BoilsAbcProblem(_settings(), evaluator=evaluator)

        launch = _launch_manifest(
            readiness=readiness,
            readiness_file_sha256=readiness_file_sha256,
            admission=admission,
            provenance=provenance,
            provenance_sha256=provenance_sha256,
            source_snapshots=source_snapshots,
            source_bundle_path=source_bundle_path,
            source_bundle_record=source_bundle_record,
        )
        launch_path = run_dir / "launch_manifest.json"
        launch_file_sha256 = durable_write_json(launch_path, launch)
        manifests = DurableManifestBinding(
            readiness_path,
            readiness_file_sha256,
            launch_path,
            launch_file_sha256,
            tuple(
                (snapshot_destinations[name], str(record["sha256"]))
                for name, record in sorted(source_snapshots.items())
            ),
            source_bundle_path,
            source_bundle_record,
        )

        event_writer = DurableJsonlWriter(run_dir / "events.jsonl")
        queue_writer = DurableJsonlWriter(run_dir / "queue_outcomes.jsonl")
        load_credentials(WORKSPACE_ROOT / ".env", optional=True)
        summary = asyncio.run(
            _run_live(
                problem=problem,
                provenance_sha256=provenance_sha256,
                readiness=readiness,
                manifests=manifests,
                event_writer=event_writer,
                queue_writer=queue_writer,
            )
        )
        summary["evaluator_observation_count"] = recorder.count
        if recorder.count != summary["resources"]["unique_physical_evaluations"]:
            raise RuntimeError(
                "durable evaluator observation count differs from physical misses"
            )
        durable_write_json(run_dir / "summary.json", summary)
        status = (
            "succeeded"
            if summary["protocol_acceptance_passed"] is True
            else "protocol_rejected"
        )
    except BaseException as exc:
        if directory_owned:
            durable_write_json(
                run_dir / "failure.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "failure_type": type(exc).__name__,
                    "safe_message": (
                        "BOiLS budgeted-v5 failed closed; inspect durable manifests "
                        "and sanitized traces. No missing arm is replaced."
                    ),
                },
            )
        raise
    finally:
        for writer in (event_writer, queue_writer, evaluation_writer):
            if writer is not None:
                writer.close()
        if directory_owned:
            _finalize(run_dir, status)


if __name__ == "__main__":
    main()

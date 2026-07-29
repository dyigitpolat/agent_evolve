#!/usr/bin/env python3
"""Prepare or run the provider-only Airfoil-v7 Stage-A portfolio experiment.

Manifest construction and verification are provider-free.  ``--live`` is the
only path that loads ``OPENROUTER_API_KEY``; it composes the benchmark-neutral
Pydantic-AI ports around the shared queued runner and never evaluates a
candidate.  The benchmark-local harness remains provider- and credential-free.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
from typing import Any, Protocol


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from pydantic import BaseModel  # noqa: E402

from agent_evolve.infrastructure.ids import DeterministicIdFactory  # noqa: E402
from agent_evolve.integrations.pydantic_ai import (  # noqa: E402
    REFLECTION_OUTPUT_CONTRACT_NOTE,
    REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256,
    REFLECTION_WIRE_CONTRACT_REVISION,
    render_reflection_prompt,
)
from agent_evolve.integrations.pydantic_ai import (  # noqa: E402
    agentic_generator as reflection_adapter,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    AttemptedStructuredGenerationResponse,
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    OpenRouterReasoningConfig,
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (  # noqa: E402
    PydanticAIPortfolioSelectionPolicy,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    OutcomePublicationPolicy,
    SCHEMA_REPAIR_POLICY_MANIFEST,
    SchemaRepairAttemptPolicy,
    create_production_queued_runner,
    structured_generation_outcome_record,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter  # noqa: E402
from agent_evolve.ports.agentic_generator import AgenticGenerator  # noqa: E402
from agent_evolve.ports.portfolio_selection import (  # noqa: E402
    PortfolioSelectionPolicy,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_development import (  # noqa: E402
    BASE_DEVELOPMENT_DESIGN_ID,
    DEFAULT_SEALED_ORACLE_DIR,
    DEVELOPMENT_DESIGN_ID,
    DirectoryDevelopmentRecordSink,
    EXECUTION_REVISION_CLASS,
    MECHANISM_REVISION_ORDINAL,
    STAGE_A_MAX_OUTPUT_TOKENS,
    development_plan_record,
    execute_provider_ready_stage_a,
    prepare_provider_ready_stage_a,
    verify_sealed_finite_oracle,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_RUN_ROOT = (
    ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7" / "portfolio_stage_a"
)
METHOD_ARTIFACT = (
    ARTIFACT_ROOT
    / "115_generic_portfolio_selector_development_and_transfer_protocol.md"
)
V1R1_REVISION_ADDENDUM = (
    ARTIFACT_ROOT / "117_airfoil_v7_stage_a_v1r1_prelaunch_addendum.md"
)
V1R2_REVISION_ARTIFACT = (
    ARTIFACT_ROOT
    / "118_airfoil_v7_stage_a_v1r1_provider_grammar_failure.md"
)
V1R2_REVISION_ARTIFACT_SHA256 = (
    "a9555218e45209a7cfa020a057da5762ca3f5bafb5340066b3d94e879bf40ba3"
)
ROUTE_SNAPSHOTS = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_capability_snapshot_20260714.json",
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_pricing_snapshot_20260714.json",
)

MODEL = "deepseek/deepseek-v4-pro"
ALLOWED_RESOLVED_MODELS = (
    MODEL,
    "deepseek/deepseek-v4-pro-20260423",
)
PROVIDER_ONLY = ("streamlake",)
ALLOWED_RESOLVED_PROVIDERS = ("StreamLake",)
MAX_INPUT_TOKENS = 640_000
MAX_OUTPUT_TOKENS = 384_000
REASONING_MAX_TOKENS = 4_096
QUEUE_MAX_IN_FLIGHT = 8
QUEUE_MAX_PENDING = 16
QUEUE_MAX_ATTEMPTS = 2
ATTEMPT_TIMEOUT_NS = 180_000_000_000
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 20_260_715
JITTER_DOMAIN = "airfoil-v7-oracle-portfolio-stage-a-v1"
EXPECTED_LOGICAL_CALLS = 11
MAXIMUM_CURRENT_PHYSICAL_PROVIDER_ATTEMPTS = 22
V1_LOGICAL_REFLECTION_CALLS = 8
V1_LOGICAL_SELECTOR_CALLS = 0
V1_PHYSICAL_PROVIDER_ATTEMPTS = 16
V1R1_LOGICAL_REFLECTION_CALLS = 8
V1R1_LOGICAL_SELECTOR_CALLS = 0
V1R1_PHYSICAL_PROVIDER_ATTEMPTS = 8
ARTIFACT_115_LOGICAL_CALL_CEILING = 23
CUMULATIVE_PREDECESSOR_LOGICAL_CALLS = (
    V1_LOGICAL_REFLECTION_CALLS + V1R1_LOGICAL_REFLECTION_CALLS
)
CUMULATIVE_PREDECESSOR_PHYSICAL_PROVIDER_ATTEMPTS = (
    V1_PHYSICAL_PROVIDER_ATTEMPTS + V1R1_PHYSICAL_PROVIDER_ATTEMPTS
)
CUMULATIVE_LOGICAL_CALLS_AFTER_R2 = (
    CUMULATIVE_PREDECESSOR_LOGICAL_CALLS + EXPECTED_LOGICAL_CALLS
)
LOGICAL_CALLS_ABOVE_OLD_DERIVED_CEILING = (
    CUMULATIVE_LOGICAL_CALLS_AFTER_R2 - ARTIFACT_115_LOGICAL_CALL_CEILING
)

MANIFEST_KIND = "airfoil_v7_oracle_portfolio_stage_a_provider_launch"
MANIFEST_FRAMING = b"agent-evolve:airfoil-v7-portfolio-stage-a-manifest:v1\x00"
SOURCE_FRAMING = b"agent-evolve:airfoil-v7-portfolio-stage-a-source:v1\x00"
FINAL_FRAMING = b"agent-evolve:airfoil-v7-portfolio-stage-a-final:v1\x00"
_SAFE_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")

V1_RUN_ID = "ae7_portfolio_stage_a_0715_0256"
V1_RUN_DIR = DEFAULT_RUN_ROOT / V1_RUN_ID
V1_EXTERNAL_MANIFEST = (
    DEFAULT_RUN_ROOT / "manifests" / f"{V1_RUN_ID}.manifest.json"
)
V1_LAUNCH_MANIFEST_FILE_SHA256 = (
    "b4d5dd7d4d55c88c6e842883ac13bdc7a69ba0da73f5ec232174d062d6064448"
)
V1_MANIFEST_COMMITMENT_SHA256 = (
    "d77efca534ef757de28d4a8651f6ea82f741178b67db063963cdacbc988c96e2"
)
V1_FINALIZED_FILE_SHA256 = (
    "34adc9b4d7568b16332624e94a7d49ab974045e26f2e5755a1161306f3428cf0"
)
V1_FINALIZATION_SHA256 = (
    "8af4cc5dd13334078fa5d42acebc62c8c724a14f09c2d82e855349de638b843a"
)
V1_RECURSIVE_CONTENT_SHA256 = (
    "4b155380961f420b876da1bb3d48891829aa69b9ed7701b2e002774628a52973"
)
V1_QUEUE_FILE_SHA256 = (
    "8a72100c7ee0945ec5873801ef31c5bdc439d568af18338ac12265387c971f2d"
)
V1_JOURNAL_FILE_SHA256 = (
    "e28316fe79e4473f3a2f785071d856038b73c69387bf74531c0afa6952af2e33"
)
V1_DEVELOPMENT_PLAN_FILE_SHA256 = (
    "2d52ee4189443415c2431f720abdd3efdfb4286c7cf071d880f231ebd1229f10"
)

V1R1_RUN_ID = "ae7_portfolio_stage_a_v1r1_jsonpath_0715_0315"
V1R1_RUN_DIR = DEFAULT_RUN_ROOT / V1R1_RUN_ID
V1R1_EXTERNAL_MANIFEST = (
    DEFAULT_RUN_ROOT / "manifests" / f"{V1R1_RUN_ID}.manifest.json"
)
V1R1_LAUNCH_MANIFEST_FILE_SHA256 = (
    "3d91aa164faae373cfc89ee06fdcd5c01a91492b06ce6570db1b11d8cfcd9e8c"
)
V1R1_MANIFEST_COMMITMENT_SHA256 = (
    "03349627644cfd119d48b143076dd3f3f5bb0240d2d33150942798b8cf928231"
)
V1R1_FINALIZED_FILE_SHA256 = (
    "c5328af6fb7b257d2479d628ec7e2c9096962a33afbe7edecf7174499139d3d8"
)
V1R1_FINALIZATION_SHA256 = (
    "71dd532c0604c0bbc3a4dce0085de1cf6a5d69d618bca90247ac22ef94df8936"
)
V1R1_RECURSIVE_CONTENT_SHA256 = (
    "80fcb03f406f162daebcf52270f4ad9fb8a8c2c9882525e98b36a8b47e96ff25"
)
V1R1_QUEUE_FILE_SHA256 = (
    "f377497d7745dc30a7083c999add171ca81039fb3ba0258a25caec05e8749f28"
)
V1R1_JOURNAL_FILE_SHA256 = (
    "8c20727bf8315b621d19c5f66fa9c47aee1454690a3073a4cb6fc6d90d8612d1"
)
V1R1_DEVELOPMENT_PLAN_FILE_SHA256 = (
    "dba61a7accb29088fc5ea61e78f9c5e9fcb55860de1417c4755bdd17e4716b12"
)

EXPECTED_REFLECTION_PROVIDER_PATH_PATTERN = r"^\$([.\[].*)?$"
EXPECTED_REFLECTION_PROVIDER_PATH_PATTERN_SHA256 = (
    "9774d7fbc3f23aced5abaa6b033060b2bfc9702a235924d8a9459d5cf76d8ba2"
)
EXPECTED_REFLECTION_WIRE_CONTRACT_REVISION = (
    "reflection_wire_jsonpath_contract_v3_provider_grammar"
)
EXPECTED_REFLECTION_WIRE_CONTRACT_REVISION_SHA256 = (
    "28563b2b0f118e49d9d245a1fe7bfef52d8ad412cb7ab778e249e5f4e9176760"
)
EXPECTED_REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256 = (
    "03b92db7cdb9b7c1f92a2508616047c450355f31eb2ff5c660063e1945b48138"
)
EXPECTED_V1R2_RENDERED_REFLECTION_PROMPT_SHA256 = (
    "5ee16f0f07e74539b2e1d33e645e72a89f79a55543718941a8790f17eabfb081",
    "f2e44721ca6e73b75afc7e842e5c35039c359dfed051458023627d63dc5ae817",
    "910847456e133a59a7e66a1fbabb06adbed8322c808699b678b16df0029a9a0c",
    "7b03eaad790e319b2cd52e3e01fbcf81655c0739a0ebe19b2cb811db67902d73",
    "941df103cba59dc67508a355ada38ae081cf5c05870bd8e28a2b3cc4567e186f",
    "95e41428a70b5bc5c8c8268d304920c7580ec9fbc13864355134e8c97b228248",
    "011bb1e75215d5c654f4af6acf17f95c51ffb60ac40b129c565b4d5bf5578aab",
    "a09fe3f951183b352e7f677acd5cbe95df5f83fd250ac49dff0e19e5ee98665d",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _domain_sha256(value: object, framing: bytes) -> str:
    return hashlib.sha256(framing + _canonical_bytes(value)).hexdigest()


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, value: object) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        _directory_fsync(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


class DurableJsonlWriter:
    """Thread-safe append, flush, and fsync before publication returns."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, value: Mapping[str, object]) -> None:
        payload = _canonical_bytes(dict(value)).decode("ascii") + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("JSONL writer is closed")
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.close()
            self._closed = True


def _file_binding(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    content = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in path.expanduser().resolve(strict=True).read_bytes().splitlines():
        value = json.loads(line)
        if type(value) is not dict:
            raise RuntimeError(f"JSONL row in {path.name} is not an object")
        rows.append(value)
    return tuple(rows)


def _recursive_content_identity(run_dir: Path) -> tuple[int, str]:
    """Recompute the predecessor's published recursive finalization identity."""

    root = run_dir.expanduser().resolve(strict=True)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name != "finalized.json"
            and not path.name.endswith(".tmp")
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    aggregate = hashlib.sha256(FINAL_FRAMING)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        encoded = relative.encode("utf-8", errors="strict")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    return len(paths), aggregate.hexdigest()


def v1_failure_binding() -> dict[str, object]:
    """Authenticate the untouched v1 schema-admission failure and call ledger."""

    run_dir = V1_RUN_DIR.expanduser().resolve(strict=True)
    launch_path = run_dir / "launch_manifest.json"
    finalized_path = run_dir / "finalized.json"
    queue_path = run_dir / "provider_queue_outcomes.jsonl"
    journal_path = run_dir / "prompt_response_journal.jsonl"
    development_plan_path = run_dir / "development_plan.json"
    launch_binding = _file_binding(launch_path)
    external_binding = _file_binding(V1_EXTERNAL_MANIFEST)
    finalized_binding = _file_binding(finalized_path)
    queue_binding = _file_binding(queue_path)
    journal_binding = _file_binding(journal_path)
    development_plan_binding = _file_binding(development_plan_path)
    if (
        launch_binding["sha256"] != V1_LAUNCH_MANIFEST_FILE_SHA256
        or external_binding["sha256"]
        != V1_LAUNCH_MANIFEST_FILE_SHA256
        or launch_path.read_bytes()
        != V1_EXTERNAL_MANIFEST.resolve(strict=True).read_bytes()
        or finalized_binding["sha256"] != V1_FINALIZED_FILE_SHA256
        or queue_binding["sha256"] != V1_QUEUE_FILE_SHA256
        or journal_binding["sha256"] != V1_JOURNAL_FILE_SHA256
        or development_plan_binding["sha256"]
        != V1_DEVELOPMENT_PLAN_FILE_SHA256
    ):
        raise RuntimeError("the sealed predecessor failure artifact drifted")

    launch = json.loads(launch_path.read_bytes())
    finalized = json.loads(finalized_path.read_bytes())
    if (
        type(launch) is not dict
        or launch.get("run_id") != V1_RUN_ID
        or launch.get("design_id") != BASE_DEVELOPMENT_DESIGN_ID
        or launch.get("manifest_sha256")
        != V1_MANIFEST_COMMITMENT_SHA256
        or type(finalized) is not dict
        or finalized.get("status") != "failed"
        or finalized.get("finalization_sha256")
        != V1_FINALIZATION_SHA256
        or finalized.get("recursive_content_sha256")
        != V1_RECURSIVE_CONTENT_SHA256
        or finalized.get("recursive_file_count") != 6
    ):
        raise RuntimeError("the predecessor manifest or finalization is unsupported")
    observed_count, observed_recursive_sha = _recursive_content_identity(run_dir)
    if (
        observed_count != finalized["recursive_file_count"]
        or observed_recursive_sha != finalized["recursive_content_sha256"]
    ):
        raise RuntimeError("the predecessor recursive content seal failed")

    queue_rows = _jsonl_objects(queue_path)
    task_ids = tuple(row.get("task_id") for row in queue_rows)
    attempts = tuple(
        attempt
        for row in queue_rows
        for attempt in (
            row.get("attempts") if type(row.get("attempts")) is list else []
        )
    )
    issue_identities = {
        (
            issue.get("category"),
            tuple(issue.get("location", ())),
        )
        for attempt in attempts
        if type(attempt) is dict
        for issue in (
            attempt.get("failure", {}).get("validation_issues", [])
            if type(attempt.get("failure")) is dict
            else []
        )
        if type(issue) is dict
    }
    expected_issue = {
        (
            "bounds_or_length",
            ("insights", "item", "affected_paths", "item"),
        )
    }
    if (
        len(queue_rows) != V1_LOGICAL_REFLECTION_CALLS
        or len(set(task_ids)) != V1_LOGICAL_REFLECTION_CALLS
        or any(row.get("status") != "attempts_exhausted" for row in queue_rows)
        or len(attempts) != V1_PHYSICAL_PROVIDER_ATTEMPTS
        or issue_identities != expected_issue
    ):
        raise RuntimeError("the predecessor provider-attempt ledger drifted")
    journal_rows = _jsonl_objects(journal_path)
    request_rows = [
        row for row in journal_rows if row.get("record_type") == "request"
    ]
    if (
        len(request_rows) != V1_LOGICAL_REFLECTION_CALLS
        or any(
            row.get("operation") != "oracle_portfolio_reflect"
            for row in request_rows
        )
        or any(row.get("operation") == "oracle_portfolio_select" for row in journal_rows)
    ):
        raise RuntimeError("the predecessor logical-call journal drifted")

    return {
        "schema_version": 1,
        "run_id": V1_RUN_ID,
        "design_id": BASE_DEVELOPMENT_DESIGN_ID,
        "status": "failed_before_any_selector_call",
        "failure_class": "pre_treatment_reflection_wire_contract_mismatch",
        "launch_manifest": launch_binding,
        "external_manifest": external_binding,
        "finalized": finalized_binding,
        "finalization_sha256": V1_FINALIZATION_SHA256,
        "recursive_content_sha256": V1_RECURSIVE_CONTENT_SHA256,
        "recursive_file_count": observed_count,
        "provider_queue_outcomes": queue_binding,
        "prompt_response_journal": journal_binding,
        "development_plan": development_plan_binding,
        "logical_reflection_calls": V1_LOGICAL_REFLECTION_CALLS,
        "logical_selector_calls": V1_LOGICAL_SELECTOR_CALLS,
        "physical_provider_attempts": V1_PHYSICAL_PROVIDER_ATTEMPTS,
        "provider_billing_usd": None,
        "provider_billing_status": "unknown_unreconciled",
        "unique_sanitized_validation_issue": {
            "category": "bounds_or_length",
            "location": ["insights", "item", "affected_paths", "item"],
        },
    }


def v1r1_failure_binding() -> dict[str, object]:
    """Authenticate the untouched v1r1 provider-grammar rejection ledger."""

    run_dir = V1R1_RUN_DIR.expanduser().resolve(strict=True)
    launch_path = run_dir / "launch_manifest.json"
    finalized_path = run_dir / "finalized.json"
    queue_path = run_dir / "provider_queue_outcomes.jsonl"
    journal_path = run_dir / "prompt_response_journal.jsonl"
    development_plan_path = run_dir / "development_plan.json"
    launch_binding = _file_binding(launch_path)
    external_binding = _file_binding(V1R1_EXTERNAL_MANIFEST)
    finalized_binding = _file_binding(finalized_path)
    queue_binding = _file_binding(queue_path)
    journal_binding = _file_binding(journal_path)
    development_plan_binding = _file_binding(development_plan_path)
    if (
        launch_binding["sha256"] != V1R1_LAUNCH_MANIFEST_FILE_SHA256
        or external_binding["sha256"] != V1R1_LAUNCH_MANIFEST_FILE_SHA256
        or launch_path.read_bytes()
        != V1R1_EXTERNAL_MANIFEST.resolve(strict=True).read_bytes()
        or finalized_binding["sha256"] != V1R1_FINALIZED_FILE_SHA256
        or queue_binding["sha256"] != V1R1_QUEUE_FILE_SHA256
        or journal_binding["sha256"] != V1R1_JOURNAL_FILE_SHA256
        or development_plan_binding["sha256"]
        != V1R1_DEVELOPMENT_PLAN_FILE_SHA256
    ):
        raise RuntimeError("the sealed v1r1 failure artifact drifted")

    launch = json.loads(launch_path.read_bytes())
    finalized = json.loads(finalized_path.read_bytes())
    if (
        type(launch) is not dict
        or launch.get("run_id") != V1R1_RUN_ID
        or launch.get("design_id")
        != "airfoil_v7_oracle_portfolio_stage_a_v1r1_jsonpath_wire"
        or launch.get("base_method_design_id") != BASE_DEVELOPMENT_DESIGN_ID
        or launch.get("execution_revision_class")
        != "pre_treatment_wire_contract_repair"
        or launch.get("mechanism_revision_ordinal") != 0
        or launch.get("manifest_sha256") != V1R1_MANIFEST_COMMITMENT_SHA256
        or type(finalized) is not dict
        or finalized.get("status") != "failed"
        or finalized.get("finalization_sha256") != V1R1_FINALIZATION_SHA256
        or finalized.get("recursive_content_sha256")
        != V1R1_RECURSIVE_CONTENT_SHA256
        or finalized.get("recursive_file_count") != 6
    ):
        raise RuntimeError("the v1r1 manifest or finalization is unsupported")
    observed_count, observed_recursive_sha = _recursive_content_identity(run_dir)
    if (
        observed_count != finalized["recursive_file_count"]
        or observed_recursive_sha != finalized["recursive_content_sha256"]
    ):
        raise RuntimeError("the v1r1 recursive content seal failed")

    queue_rows = _jsonl_objects(queue_path)
    task_ids = tuple(row.get("task_id") for row in queue_rows)
    attempts = tuple(
        attempt
        for row in queue_rows
        for attempt in (
            row.get("attempts") if type(row.get("attempts")) is list else []
        )
    )
    if (
        len(queue_rows) != V1R1_LOGICAL_REFLECTION_CALLS
        or len(set(task_ids)) != V1R1_LOGICAL_REFLECTION_CALLS
        or any(row.get("status") != "terminal_failure" for row in queue_rows)
        or any(row.get("response") is not None for row in queue_rows)
        or len(attempts) != V1R1_PHYSICAL_PROVIDER_ATTEMPTS
        or any(
            type(attempt) is not dict
            or attempt.get("attempt_number") != 1
            or attempt.get("status") != "terminal_failure"
            or attempt.get("will_retry") is not False
            or type(attempt.get("failure")) is not dict
            or attempt["failure"].get("kind") != "invalid_request"
            or attempt["failure"].get("status_code") != 400
            or attempt["failure"].get("retryable") is not False
            for attempt in attempts
        )
    ):
        raise RuntimeError("the v1r1 provider-attempt ledger drifted")

    journal_rows = _jsonl_objects(journal_path)
    request_rows = [
        row for row in journal_rows if row.get("record_type") == "request"
    ]
    failure_rows = [
        row
        for row in journal_rows
        if row.get("record_type") == "response_failure"
    ]
    if (
        len(request_rows) != V1R1_LOGICAL_REFLECTION_CALLS
        or len(failure_rows) != V1R1_LOGICAL_REFLECTION_CALLS
        or {row.get("call_id") for row in request_rows} != set(task_ids)
        or {row.get("call_id") for row in failure_rows} != set(task_ids)
        or any(
            row.get("operation") != "oracle_portfolio_reflect"
            for row in request_rows
        )
        or any(
            "Revision: reflection_wire_jsonpath_contract_v2."
            not in str(row.get("prompt"))
            for row in request_rows
        )
        or any(
            row.get("record_type") not in {"request", "response_failure"}
            for row in journal_rows
        )
        or any(
            row.get("operation") == "oracle_portfolio_select"
            for row in journal_rows
        )
    ):
        raise RuntimeError("the v1r1 logical-call journal drifted")

    return {
        "schema_version": 1,
        "run_id": V1R1_RUN_ID,
        "design_id": (
            "airfoil_v7_oracle_portfolio_stage_a_v1r1_jsonpath_wire"
        ),
        "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
        "status": "failed_before_any_inference_or_selector_call",
        "failure_class": "pre_treatment_provider_grammar_rejection",
        "launch_manifest": launch_binding,
        "external_manifest": external_binding,
        "finalized": finalized_binding,
        "finalization_sha256": V1R1_FINALIZATION_SHA256,
        "recursive_content_sha256": V1R1_RECURSIVE_CONTENT_SHA256,
        "recursive_file_count": observed_count,
        "provider_queue_outcomes": queue_binding,
        "prompt_response_journal": journal_binding,
        "development_plan": development_plan_binding,
        "logical_reflection_calls": V1R1_LOGICAL_REFLECTION_CALLS,
        "logical_selector_calls": V1R1_LOGICAL_SELECTOR_CALLS,
        "physical_provider_attempts": V1R1_PHYSICAL_PROVIDER_ATTEMPTS,
        "http_status_codes": [400],
        "accepted_provider_responses": 0,
        "provider_inferences": 0,
        "provider_billing_usd": None,
        "provider_billing_status": "not_reported_for_http_400",
    }


def predecessor_failure_binding() -> dict[str, object]:
    """Return both authenticated invalid pre-treatment executions."""

    return {
        "schema_version": 1,
        "ordered_run_ids": [V1_RUN_ID, V1R1_RUN_ID],
        "v1": v1_failure_binding(),
        "v1r1": v1r1_failure_binding(),
    }


def _source_paths() -> tuple[Path, ...]:
    paths = {
        path.resolve(strict=True)
        for path in (AGENT_EVOLVE_ROOT / "src" / "agent_evolve").rglob("*.py")
        if path.is_file()
    }
    paths.update(
        path.resolve(strict=True)
        for path in (
            AGENT_EVOLVE_ROOT / "examples" / "benchmarks" / "engibench_airfoil"
        ).glob("*.py")
        if path.is_file()
    )
    paths.update(
        {
            Path(__file__).resolve(strict=True),
            (AGENT_EVOLVE_ROOT / "pyproject.toml").resolve(strict=True),
            (AGENT_EVOLVE_ROOT / "uv.lock").resolve(strict=True),
        }
    )
    return tuple(sorted(paths, key=lambda item: str(item)))


def source_snapshot() -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(SOURCE_FRAMING)
    for path in _source_paths():
        content = path.read_bytes()
        label = path.relative_to(AGENT_EVOLVE_ROOT).as_posix()
        files[label] = {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        encoded = label.encode("utf-8", errors="strict")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    return {
        "schema_version": 1,
        "framing": SOURCE_FRAMING[:-1].decode("ascii"),
        "file_count": len(files),
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


def _route_snapshot_binding() -> dict[str, object]:
    capability = json.loads(ROUTE_SNAPSHOTS[0].read_bytes())
    pricing = json.loads(ROUTE_SNAPSHOTS[1].read_bytes())
    endpoint = capability.get("selected_endpoint")
    model = pricing.get("model")
    price_endpoint = pricing.get("selected_endpoint")
    if (
        capability.get("requested_model_alias") != MODEL
        or capability.get("canonical_model_slug") != ALLOWED_RESOLVED_MODELS[1]
        or type(endpoint) is not dict
        or endpoint.get("provider_name") != "StreamLake"
        or endpoint.get("provider_request_slug") != "streamlake"
        or endpoint.get("max_completion_tokens") != MAX_OUTPUT_TOKENS
        or type(model) is not dict
        or model.get("max_completion_tokens") != MAX_OUTPUT_TOKENS
        or type(price_endpoint) is not dict
        or price_endpoint.get("provider_request_slug") != "streamlake"
    ):
        raise RuntimeError("dated route snapshots do not bind the selected route")
    return {
        "capability": _file_binding(ROUTE_SNAPSHOTS[0]),
        "pricing": _file_binding(ROUTE_SNAPSHOTS[1]),
        "canonical_model": capability["canonical_model_slug"],
        "provider_name": endpoint["provider_name"],
        "provider_request_slug": endpoint["provider_request_slug"],
        "context_length": endpoint["context_length"],
        "max_completion_tokens": endpoint["max_completion_tokens"],
    }


def provider_policy_record() -> dict[str, object]:
    if MAX_OUTPUT_TOKENS != STAGE_A_MAX_OUTPUT_TOKENS:
        raise RuntimeError("harness and provider completion ceilings differ")
    return {
        "requested_model": MODEL,
        "allowed_resolved_models": list(ALLOWED_RESOLVED_MODELS),
        "provider": "openrouter",
        "provider_options": {
            "only": list(PROVIDER_ONLY),
            "allow_fallbacks": False,
        },
        "allowed_resolved_providers": list(ALLOWED_RESOLVED_PROVIDERS),
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_max_tokens": REASONING_MAX_TOKENS,
        "queue": {
            "max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "max_pending": QUEUE_MAX_PENDING,
            "max_attempts": QUEUE_MAX_ATTEMPTS,
            "attempt_timeout_seconds": ATTEMPT_TIMEOUT_NS // 1_000_000_000,
            "backoff": {
                "kind": "exponential_with_deterministic_task_keyed_full_jitter",
                "base_seconds": BASE_BACKOFF_NS // 1_000_000_000,
                "max_seconds": MAX_BACKOFF_NS // 1_000_000_000,
                "seed": JITTER_SEED,
                "domain": JITTER_DOMAIN,
            },
            "schema_repair": SCHEMA_REPAIR_POLICY_MANIFEST.to_trace_record(),
            "terminal_outcome_publication": "required_fsync_before_return",
        },
        "route_snapshot": _route_snapshot_binding(),
    }


def _oracle_binding(run_dir: Path) -> dict[str, object]:
    oracle = verify_sealed_finite_oracle(run_dir)
    return {
        "path": str(oracle.run_dir),
        "seal": oracle.seal_record(),
        "oracle_manifest_file": _file_binding(oracle.run_dir / "oracle_manifest.json"),
        "oracle_result_file": _file_binding(oracle.run_dir / "oracle_result.json"),
        "finalization_file": _file_binding(oracle.run_dir / "finalized.json"),
    }


def _plan_record(run_dir: Path) -> dict[str, object]:
    oracle, design, calls = prepare_provider_ready_stage_a(
        run_dir=run_dir,
        id_factory=DeterministicIdFactory("airfoil_oracle_stage_a"),
    )
    plan = development_plan_record(oracle, design, calls)
    if (
        plan.get("logical_reflection_calls") != 8
        or plan.get("planned_selector_calls") != 3
        or plan.get("new_candidate_evaluations") != 0
    ):
        raise RuntimeError("Stage-A plan accounting drifted")
    return plan


_PLAN_EXECUTION_METADATA_FIELDS = (
    "design_id",
    "base_method_design_id",
    "execution_revision_class",
    "mechanism_revision_ordinal",
)


def _scientific_plan_projection(
    plan: Mapping[str, object],
) -> dict[str, object]:
    projected = dict(plan)
    for field in _PLAN_EXECUTION_METADATA_FIELDS:
        projected.pop(field, None)
    return projected


def _scientific_surface_binding(
    current_plan: Mapping[str, object],
) -> dict[str, object]:
    """Prove v1r2 changes execution grammar, not Stage-A science."""

    v1_path = V1_RUN_DIR.resolve(strict=True) / "development_plan.json"
    v1r1_path = V1R1_RUN_DIR.resolve(strict=True) / "development_plan.json"
    v1_plan = json.loads(v1_path.read_bytes())
    v1r1_plan = json.loads(v1r1_path.read_bytes())
    if type(v1_plan) is not dict or type(v1r1_plan) is not dict:
        raise RuntimeError("predecessor development plan is malformed")
    projections = (
        _scientific_plan_projection(v1_plan),
        _scientific_plan_projection(v1r1_plan),
        _scientific_plan_projection(current_plan),
    )
    if projections[0] != projections[1] or projections[1] != projections[2]:
        raise RuntimeError("Stage-A scientific surfaces changed across revisions")
    projection_bytes = _canonical_bytes(projections[2])
    return {
        "schema_version": 1,
        "unchanged": True,
        "projection_sha256": hashlib.sha256(projection_bytes).hexdigest(),
        "excluded_execution_metadata_fields": list(
            _PLAN_EXECUTION_METADATA_FIELDS
        ),
        "v1_development_plan": _file_binding(v1_path),
        "v1r1_development_plan": _file_binding(v1r1_path),
    }


def _reflection_v3_wire_binding() -> dict[str, object]:
    pattern = getattr(reflection_adapter, "_JSON_PATH_PATTERN", None)
    pattern_sha = (
        hashlib.sha256(pattern.encode("utf-8", errors="strict")).hexdigest()
        if type(pattern) is str
        else None
    )
    revision_sha = hashlib.sha256(
        REFLECTION_WIRE_CONTRACT_REVISION.encode("utf-8", errors="strict")
    ).hexdigest()
    if (
        pattern != EXPECTED_REFLECTION_PROVIDER_PATH_PATTERN
        or pattern_sha != EXPECTED_REFLECTION_PROVIDER_PATH_PATTERN_SHA256
        or REFLECTION_WIRE_CONTRACT_REVISION
        != EXPECTED_REFLECTION_WIRE_CONTRACT_REVISION
        or revision_sha != EXPECTED_REFLECTION_WIRE_CONTRACT_REVISION_SHA256
        or REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256
        != EXPECTED_REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256
        or hashlib.sha256(
            REFLECTION_OUTPUT_CONTRACT_NOTE.encode(
                "utf-8", errors="strict"
            )
        ).hexdigest()
        != EXPECTED_REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256
    ):
        raise RuntimeError("generic reflection v3 wire contract drifted")
    return {
        "wire_contract_revision": REFLECTION_WIRE_CONTRACT_REVISION,
        "wire_contract_revision_sha256": revision_sha,
        "provider_path_pattern": pattern,
        "provider_path_pattern_sha256": pattern_sha,
        "output_contract_note_sha256": REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256,
    }


def _provider_schema_compatibility_evidence() -> dict[str, object]:
    """Bind the honest evidence boundary for the v3 provider-grammar repair."""

    artifact = _file_binding(V1R2_REVISION_ARTIFACT)
    if artifact["sha256"] != V1R2_REVISION_ARTIFACT_SHA256:
        raise RuntimeError("v1r2 revision artifact drifted")
    return {
        "schema_version": 1,
        "revision_artifact": artifact,
        "sealed_negative_evidence": {
            "run_id": V1R1_RUN_ID,
            "recursive_content_sha256": V1R1_RECURSIVE_CONTENT_SHA256,
            "physical_provider_attempts": V1R1_PHYSICAL_PROVIDER_ATTEMPTS,
            "http_status_codes": [400],
            "provider_inferences": 0,
        },
        "positive_evidence_status": (
            "operator_observed_probe_described_in_revision_artifact"
        ),
        "positive_probe_raw_bytes_cryptographically_bound": False,
        "prelaunch_provider_acceptance_proven": False,
        "scientific_response_reused": False,
    }


def _rendered_reflection_prompt_bindings(
    plan: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    calls = plan.get("reflection_calls")
    if type(calls) is not list or len(calls) != V1R1_LOGICAL_REFLECTION_CALLS:
        raise RuntimeError("Stage-A reflection plan cardinality drifted")
    bindings: list[dict[str, object]] = []
    for ordinal, call in enumerate(calls, start=1):
        if type(call) is not dict:
            raise RuntimeError("Stage-A reflection call record is malformed")
        call_id = call.get("call_id")
        prompt = call.get("prompt")
        planned_sha = call.get("prompt_sha256")
        if type(call_id) is not str or type(prompt) is not str:
            raise RuntimeError("Stage-A reflection prompt binding is malformed")
        original_sha = hashlib.sha256(
            prompt.encode("utf-8", errors="strict")
        ).hexdigest()
        if planned_sha != original_sha:
            raise RuntimeError("Stage-A high-level prompt hash drifted")
        rendered = render_reflection_prompt(prompt)
        expected_suffix = f"\n\n{REFLECTION_OUTPUT_CONTRACT_NOTE}"
        if rendered == prompt or not rendered.endswith(expected_suffix):
            raise RuntimeError(
                "Stage-A provider prompt did not admit the frozen reflection note"
            )
        rendered_bytes = rendered.encode("utf-8", errors="strict")
        bindings.append(
            {
                "ordinal": ordinal,
                "call_id": call_id,
                "high_level_prompt_sha256": original_sha,
                "provider_prompt_sha256": hashlib.sha256(rendered_bytes).hexdigest(),
                "provider_prompt_utf8_bytes": len(rendered_bytes),
            }
        )
    observed_hashes = tuple(
        str(binding["provider_prompt_sha256"]) for binding in bindings
    )
    if observed_hashes != EXPECTED_V1R2_RENDERED_REFLECTION_PROMPT_SHA256:
        raise RuntimeError("v1r2 rendered reflection prompt hashes drifted")
    return tuple(bindings)


def revision_lineage_record(plan: Mapping[str, object]) -> dict[str, object]:
    """Freeze the final execution repair and cumulative resource ledger."""

    if plan.get("design_id") != DEVELOPMENT_DESIGN_ID:
        raise RuntimeError("Stage-A development revision identity drifted")
    if (
        plan.get("base_method_design_id") != BASE_DEVELOPMENT_DESIGN_ID
        or plan.get("execution_revision_class") != EXECUTION_REVISION_CLASS
        or plan.get("mechanism_revision_ordinal") != MECHANISM_REVISION_ORDINAL
    ):
        raise RuntimeError("Stage-A development revision lineage drifted")
    rendered = _rendered_reflection_prompt_bindings(plan)
    return {
        "schema_version": 1,
        "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
        "execution_revision_id": DEVELOPMENT_DESIGN_ID,
        "execution_revision_class": EXECUTION_REVISION_CLASS,
        "mechanism_revision_ordinal": MECHANISM_REVISION_ORDINAL,
        "predecessor": v1r1_failure_binding(),
        "predecessor_runs": predecessor_failure_binding(),
        "change_commitment": {
            "scope": "generic_reflection_provider_boundary_only",
            "provider_wire": _reflection_v3_wire_binding(),
            "provider_prompt_bindings": list(rendered),
            "provider_schema_compatibility_evidence": (
                _provider_schema_compatibility_evidence()
            ),
            "scientific_surface_binding": _scientific_surface_binding(plan),
            "allowed_changes": [
                (
                    "replace only the rejected generic JSON-path regex grammar "
                    "with the language-equivalent provider-compatible form in "
                    "local validation and the provider-visible schema"
                ),
                (
                    "update only the wire revision, derived output-contract note, "
                    "and eight derived rendered-prompt hashes"
                ),
            ],
            "frozen_unchanged_surfaces": [
                "sealed adaptation oracle and all 80 outcomes",
                "eight outcome-independent family-contiguous shards",
                "reflection evidence, action, metric, and family vocabularies",
                "M/P/N selector projections and survival gates",
                "requested model, provider route, temperature, and token ceilings",
                "queue concurrency, attempt limit, timeout, and backoff policy",
                "zero candidate evaluations and zero CFD calls",
            ],
            "no_further_execution_revision_permitted": True,
        },
        "cumulative_accounting": {
            "accounting_basis": "logical_calls_separate_from_physical_attempts",
            "v1_realized": {
                "logical_calls": V1_LOGICAL_REFLECTION_CALLS,
                "physical_provider_attempts": V1_PHYSICAL_PROVIDER_ATTEMPTS,
                "logical_selector_calls": V1_LOGICAL_SELECTOR_CALLS,
            },
            "v1r1_realized": {
                "logical_calls": V1R1_LOGICAL_REFLECTION_CALLS,
                "physical_provider_attempts": V1R1_PHYSICAL_PROVIDER_ATTEMPTS,
                "http_status_codes": [400],
                "provider_inferences": 0,
                "logical_selector_calls": V1R1_LOGICAL_SELECTOR_CALLS,
            },
            "cumulative_before_current": {
                "logical_calls": CUMULATIVE_PREDECESSOR_LOGICAL_CALLS,
                "physical_provider_attempts": (
                    CUMULATIVE_PREDECESSOR_PHYSICAL_PROVIDER_ATTEMPTS
                ),
            },
            "current_plan": {
                "logical_reflection_calls": 8,
                "logical_selector_calls": 3,
                "logical_calls": EXPECTED_LOGICAL_CALLS,
                "maximum_physical_provider_attempts": (
                    MAXIMUM_CURRENT_PHYSICAL_PROVIDER_ATTEMPTS
                ),
            },
            "cumulative_logical_calls_after_current": CUMULATIVE_LOGICAL_CALLS_AFTER_R2,
            "maximum_cumulative_physical_provider_attempts_after_current": (
                CUMULATIVE_PREDECESSOR_PHYSICAL_PROVIDER_ATTEMPTS
                + MAXIMUM_CURRENT_PHYSICAL_PROVIDER_ATTEMPTS
            ),
            "artifact_115_old_derived_logical_call_ceiling": ARTIFACT_115_LOGICAL_CALL_CEILING,
            "logical_calls_above_old_derived_ceiling": LOGICAL_CALLS_ABOVE_OLD_DERIVED_CEILING,
            "old_derived_ceiling_compliance_claimed": False,
            "literal_paid_call_cap_compliance_claimed": False,
            "separate_diagnostic_schema_probe_requests": 2,
            "diagnostic_probe_requests_included_in_stage_a_totals": False,
            "additional_revision_calls_permitted": 0,
            "no_further_execution_revision_permitted": True,
        },
    }


def _validate_target(run_id: str, output_dir: Path) -> Path:
    if type(run_id) is not str or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be one safe lowercase path component")
    target = output_dir.expanduser().resolve()
    if target.name != run_id:
        raise ValueError("output_dir basename must equal run_id")
    return target


def build_manifest_record(
    *,
    run_id: str,
    output_dir: Path,
    oracle_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
) -> dict[str, object]:
    """Build a complete provider-free commitment without reading credentials."""

    target = _validate_target(run_id, output_dir)
    oracle_path = oracle_dir.expanduser().resolve(strict=True)
    plan = _plan_record(oracle_path)
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "output_dir": str(target),
        "claim_boundary": (
            "Post-hoc oracle-backed method development; not held-out efficacy, "
            "genericity, SOTA, or wall-clock evidence."
        ),
        "design_id": DEVELOPMENT_DESIGN_ID,
        "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
        "execution_revision_class": EXECUTION_REVISION_CLASS,
        "mechanism_revision_ordinal": MECHANISM_REVISION_ORDINAL,
        "revision_lineage": revision_lineage_record(plan),
        "provider_policy": provider_policy_record(),
        "experiment": {
            "logical_reflection_calls": 8,
            "logical_selector_calls": 3,
            "logical_calls": EXPECTED_LOGICAL_CALLS,
            "candidate_evaluations": 0,
            "cfd_calls": 0,
            "execution": "8 concurrent reflections then 3 concurrent selectors",
        },
        "method_artifact": _file_binding(METHOD_ARTIFACT),
        "v1r1_revision_addendum": _file_binding(V1R1_REVISION_ADDENDUM),
        "v1r2_revision_artifact": _provider_schema_compatibility_evidence()[
            "revision_artifact"
        ],
        "oracle": _oracle_binding(oracle_path),
        "development_plan": plan,
        "source_snapshot": source_snapshot(),
        "durability": {
            "manifest": "atomic_write_fsync_directory",
            "queue_jsonl": "append_flush_fsync_required_before_response_use",
            "journal_jsonl": "append_flush_fsync",
            "result": "atomic_write_fsync_directory",
            "finalization": "recursive_content_seal_excluding_finalized_json",
        },
        "credentials_read": False,
        "provider_dispatch_performed": False,
    }
    record["manifest_sha256"] = _domain_sha256(record, MANIFEST_FRAMING)
    return record


def write_manifest(
    path: Path,
    *,
    run_id: str,
    output_dir: Path,
    oracle_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
) -> dict[str, object]:
    record = build_manifest_record(
        run_id=run_id,
        output_dir=output_dir,
        oracle_dir=oracle_dir,
    )
    write_json_atomic(path, record)
    return record


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    path: Path
    record: dict[str, object]
    run_id: str
    output_dir: Path
    oracle_dir: Path
    manifest_sha256: str
    source_sha256: str


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.expanduser().resolve(strict=True).read_bytes())
    if type(value) is not dict:
        raise TypeError("manifest root must be an object")
    return value


def verify_manifest(
    path: Path,
    *,
    require_output_absent: bool = True,
) -> VerifiedManifest:
    resolved = path.expanduser().resolve(strict=True)
    record = _load_object(resolved)
    claimed = record.get("manifest_sha256")
    unsigned = dict(record)
    unsigned.pop("manifest_sha256", None)
    if (
        type(claimed) is not str
        or _LOWER_SHA256.fullmatch(claimed) is None
        or claimed != _domain_sha256(unsigned, MANIFEST_FRAMING)
    ):
        raise RuntimeError("manifest self-hash failed")
    if record.get("kind") != MANIFEST_KIND or record.get("schema_version") != 1:
        raise RuntimeError("manifest kind or schema is unsupported")
    if (
        record.get("design_id") != DEVELOPMENT_DESIGN_ID
        or record.get("base_method_design_id") != BASE_DEVELOPMENT_DESIGN_ID
        or record.get("execution_revision_class") != EXECUTION_REVISION_CLASS
        or record.get("mechanism_revision_ordinal") != MECHANISM_REVISION_ORDINAL
    ):
        raise RuntimeError("manifest execution revision identity drifted")
    run_id = record.get("run_id")
    output = record.get("output_dir")
    oracle_record = record.get("oracle")
    if type(run_id) is not str or type(output) is not str or type(oracle_record) is not dict:
        raise RuntimeError("manifest target binding is malformed")
    output_dir = _validate_target(run_id, Path(output))
    oracle_path_value = oracle_record.get("path")
    if type(oracle_path_value) is not str:
        raise RuntimeError("manifest oracle path is malformed")
    oracle_dir = Path(oracle_path_value).resolve(strict=True)
    if require_output_absent and output_dir.exists():
        raise FileExistsError(output_dir)
    observed_source = source_snapshot()
    if record.get("source_snapshot") != observed_source:
        raise RuntimeError("source snapshot drifted after manifest preparation")
    if record.get("provider_policy") != provider_policy_record():
        raise RuntimeError("provider route or queue policy drifted")
    if record.get("method_artifact") != _file_binding(METHOD_ARTIFACT):
        raise RuntimeError("method artifact drifted")
    if record.get("v1r1_revision_addendum") != _file_binding(
        V1R1_REVISION_ADDENDUM
    ):
        raise RuntimeError("Stage-A v1r1 execution-revision addendum drifted")
    if record.get("v1r2_revision_artifact") != (
        _provider_schema_compatibility_evidence()["revision_artifact"]
    ):
        raise RuntimeError("Stage-A v1r2 execution-revision artifact drifted")
    if oracle_record != _oracle_binding(oracle_dir):
        raise RuntimeError("sealed oracle binding drifted")
    observed_plan = _plan_record(oracle_dir)
    if record.get("development_plan") != observed_plan:
        raise RuntimeError("provider-ready Stage-A plan drifted")
    if record.get("revision_lineage") != revision_lineage_record(observed_plan):
        raise RuntimeError("Stage-A revision lineage or predecessor drifted")
    source_sha = observed_source.get("sha256")
    if type(source_sha) is not str:
        raise RuntimeError("source snapshot SHA is malformed")
    return VerifiedManifest(
        path=resolved,
        record=record,
        run_id=run_id,
        output_dir=output_dir,
        oracle_dir=oracle_dir,
        manifest_sha256=claimed,
        source_sha256=source_sha,
    )


def reverify_source(verified: VerifiedManifest) -> dict[str, object]:
    observed = source_snapshot()
    if observed != verified.record["source_snapshot"]:
        raise RuntimeError("source drifted before provider dispatch")
    return {
        "source_sha256": observed["sha256"],
        "source_file_count": observed["file_count"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _model_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return {"runtime_type": f"{type(value).__module__}.{type(value).__qualname__}"}


class AuditedStructuredRunner:
    """Verify source and route around every queued structured logical call."""

    def __init__(
        self,
        runner: Callable[[StructuredGenerationRequest[Any]], Awaitable[object]],
        *,
        pre_dispatch: Callable[[StructuredGenerationRequest[Any]], Mapping[str, object]],
        journal_sink: Callable[[Mapping[str, object]], None],
    ) -> None:
        self._runner = runner
        self._pre_dispatch = pre_dispatch
        self._journal_sink = journal_sink
        self._lock = asyncio.Lock()
        self._ordinal = 0

    async def __call__(self, request: StructuredGenerationRequest[Any]) -> object:
        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        if request.max_output_tokens > MAX_OUTPUT_TOKENS:
            raise RuntimeError("request exceeds the frozen provider completion ceiling")
        async with self._lock:
            self._ordinal += 1
            ordinal = self._ordinal
            verification = dict(self._pre_dispatch(request))
            self._journal_sink(
                {
                    "schema_version": 1,
                    "record_type": "request",
                    "logical_call_ordinal": ordinal,
                    "call_id": request.call_id.value,
                    "operation": request.operation,
                    "prompt": request.prompt,
                    "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
                    "output_tool_name": request.output_tool_name,
                    "max_output_tokens": request.max_output_tokens,
                    "temperature": request.temperature,
                    "source_verification": verification,
                }
            )
        try:
            raw = await self._runner(request)
        except BaseException as exc:
            self._journal_sink(
                {
                    "schema_version": 1,
                    "record_type": "response_failure",
                    "logical_call_ordinal": ordinal,
                    "call_id": request.call_id.value,
                    "failure_type": type(exc).__name__,
                }
            )
            raise
        if type(raw) is AttemptedStructuredGenerationResponse:
            response = raw.response
            attempts = raw.attempt_count
        elif type(raw) is StructuredGenerationResponse:
            response = raw
            attempts = 1
        else:
            raise TypeError("queued runner returned an unsupported response")
        if (
            response.requested_model != MODEL
            or response.resolved_model not in ALLOWED_RESOLVED_MODELS
            or response.resolved_provider not in ALLOWED_RESOLVED_PROVIDERS
            or response.input_tokens > MAX_INPUT_TOKENS
            or response.output_tokens > MAX_OUTPUT_TOKENS
            or response.reasoning_tokens > REASONING_MAX_TOKENS
            or attempts > QUEUE_MAX_ATTEMPTS
        ):
            self._journal_sink(
                {
                    "schema_version": 1,
                    "record_type": "response_route_rejected",
                    "logical_call_ordinal": ordinal,
                    "call_id": request.call_id.value,
                }
            )
            raise RuntimeError("provider response violated the frozen route policy")
        content = _model_value(response.value)
        self._journal_sink(
            {
                "schema_version": 1,
                "record_type": "response",
                "logical_call_ordinal": ordinal,
                "call_id": request.call_id.value,
                "content": content,
                "content_sha256": hashlib.sha256(_canonical_bytes(content)).hexdigest(),
                "telemetry": {
                    "requested_model": response.requested_model,
                    "resolved_model": response.resolved_model,
                    "resolved_provider": response.resolved_provider,
                    "provider_response_id": response.provider_response_id,
                    "finish_reason": response.finish_reason,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                    "cache_read_tokens": response.cache_read_tokens,
                    "cache_write_tokens": response.cache_write_tokens,
                    "cost_usd": None if response.cost_usd is None else str(response.cost_usd),
                    "latency_ns": response.latency_ns,
                    "attempt_count": attempts,
                },
            }
        )
        return raw


class LiveStackLike(Protocol):
    generator: AgenticGenerator
    selector: PortfolioSelectionPolicy

    async def __aenter__(self) -> "LiveStackLike": ...

    async def __aexit__(self, *_: object) -> None: ...


@dataclass(slots=True)
class LiveStack:
    runner: Any
    generator: AgenticGenerator
    selector: PortfolioSelectionPolicy

    async def __aenter__(self) -> "LiveStack":
        await self.runner.__aenter__()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.runner.__aexit__(None, None, None)


def create_live_stack(
    *,
    api_key: str,
    queue_sink: Callable[[Mapping[str, object]], None],
    journal_sink: Callable[[Mapping[str, object]], None],
    pre_dispatch: Callable[[StructuredGenerationRequest[Any]], Mapping[str, object]],
) -> LiveStack:
    """Compose the reusable provider ports; do not enter or dispatch them."""

    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=MODEL,
        max_connections=QUEUE_MAX_IN_FLIGHT,
        timeout_seconds=ATTEMPT_TIMEOUT_NS / 1_000_000_000,
        provider_options={"only": list(PROVIDER_ONLY), "allow_fallbacks": False},
        reasoning_config=OpenRouterReasoningConfig(max_tokens=REASONING_MAX_TOKENS),
        app_title="AgentEvolve AAAI 2027 oracle portfolio Stage A",
    )
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=QUEUE_MAX_IN_FLIGHT,
        max_pending=QUEUE_MAX_PENDING,
        max_attempts=QUEUE_MAX_ATTEMPTS,
        attempt_timeout_ns=ATTEMPT_TIMEOUT_NS,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        jitter_policy=DeterministicHashJitter(
            seed=JITTER_SEED,
            domain=JITTER_DOMAIN,
        ),
        close_generator=True,
        outcome_sink=lambda outcome: queue_sink(
            structured_generation_outcome_record(outcome)
        ),
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        attempt_request_policy=SchemaRepairAttemptPolicy(),
    )
    audited = AuditedStructuredRunner(
        runner,
        pre_dispatch=pre_dispatch,
        journal_sink=journal_sink,
    )
    return LiveStack(
        runner=runner,
        generator=PydanticAIAgenticGenerator(audited),
        selector=PydanticAIPortfolioSelectionPolicy(audited),
    )


StageExecutor = Callable[..., Awaitable[dict[str, object]]]
StackFactory = Callable[..., LiveStackLike]


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    credential_loader: Callable[[], str]
    stack_factory: StackFactory
    stage_executor: StageExecutor = execute_provider_ready_stage_a
    enforce_provider_accounting: bool = True


def production_dependencies() -> LiveDependencies:
    """Return lazy factories without consulting the environment or ``.env``."""

    def load_key() -> str:
        from agent_evolve.settings import load_credentials

        load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
        return os.environ.get("OPENROUTER_API_KEY", "")

    return LiveDependencies(
        credential_loader=load_key,
        stack_factory=lambda **kwargs: create_live_stack(**kwargs),
    )


def _provider_accounting(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    task_ids = tuple(row.get("task_id") for row in rows)
    passed = (
        len(rows) == EXPECTED_LOGICAL_CALLS
        and len(set(task_ids)) == EXPECTED_LOGICAL_CALLS
        and all(row.get("status") == "succeeded" for row in rows)
        and all(
            isinstance(row.get("attempts"), list)
            and 1 <= len(row["attempts"]) <= QUEUE_MAX_ATTEMPTS
            for row in rows
        )
    )
    return {
        "expected_logical_calls": EXPECTED_LOGICAL_CALLS,
        "terminal_queue_outcomes": len(rows),
        "unique_task_ids": len(set(task_ids)),
        "all_succeeded": all(row.get("status") == "succeeded" for row in rows),
        "passed": passed,
    }


def _finalize(run_dir: Path, *, status: str) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(FINAL_FRAMING)
    paths = sorted(
        (
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and path.name != "finalized.json"
            and not path.name.endswith(".tmp")
        ),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(run_dir).as_posix()
        content = path.read_bytes()
        files[relative] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            **({"jsonl_lines": len(content.splitlines())} if path.suffix == ".jsonl" else {}),
        }
        encoded = relative.encode("utf-8", errors="strict")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    record: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "framing": FINAL_FRAMING[:-1].decode("ascii"),
        "recursive_file_count": len(files),
        "recursive_content_sha256": aggregate.hexdigest(),
        "files": files,
    }
    record["finalization_sha256"] = _domain_sha256(record, FINAL_FRAMING)
    write_json_atomic(run_dir / "finalized.json", record)
    return record


def execute_with_dependencies(
    manifest_path: Path,
    dependencies: LiveDependencies,
) -> dict[str, object]:
    """Execute only the provider Stage A through injected composition roots."""

    verified = verify_manifest(manifest_path, require_output_absent=True)
    run_dir = verified.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    _directory_fsync(run_dir.parent)
    shutil.copyfile(verified.path, run_dir / "launch_manifest.json")
    with (run_dir / "launch_manifest.json").open("rb") as stream:
        os.fsync(stream.fileno())
    _directory_fsync(run_dir)

    queue_writer = DurableJsonlWriter(run_dir / "provider_queue_outcomes.jsonl")
    journal_writer = DurableJsonlWriter(run_dir / "prompt_response_journal.jsonl")
    source_writer = DurableJsonlWriter(run_dir / "source_verifications.jsonl")
    queue_rows: list[dict[str, object]] = []
    queue_lock = threading.Lock()
    status = "failed"
    pending: BaseException | None = None
    summary: dict[str, object] | None = None
    credentials_read = False
    try:
        source_writer.write({"stage": "post_run_directory_creation", **reverify_source(verified)})

        def pre_dispatch(request: StructuredGenerationRequest[Any]) -> Mapping[str, object]:
            row = {
                "stage": "pre_provider_dispatch",
                "call_id": request.call_id.value,
                **reverify_source(verified),
            }
            source_writer.write(row)
            return row

        def queue_sink(record: Mapping[str, object]) -> None:
            row = dict(record)
            queue_writer.write(row)
            with queue_lock:
                queue_rows.append(row)

        source_writer.write({"stage": "pre_credential_load", **reverify_source(verified)})
        api_key = dependencies.credential_loader()
        credentials_read = True
        if type(api_key) is not str or not api_key.strip():
            raise RuntimeError("OPENROUTER_API_KEY is unavailable")
        stack = dependencies.stack_factory(
            api_key=api_key,
            queue_sink=queue_sink,
            journal_sink=journal_writer.write,
            pre_dispatch=pre_dispatch,
        )

        async def run() -> dict[str, object]:
            async with stack:
                return await dependencies.stage_executor(
                    generator=stack.generator,
                    selector=stack.selector,
                    run_dir=verified.oracle_dir,
                    id_factory=DeterministicIdFactory("airfoil_oracle_stage_a"),
                    sink=DirectoryDevelopmentRecordSink(run_dir),
                )

        result = asyncio.run(run())
        write_json_atomic(run_dir / "result.json", result)
        accounting = _provider_accounting(queue_rows)
        if dependencies.enforce_provider_accounting and not accounting["passed"]:
            raise RuntimeError("provider accounting drifted from the 8+3 design")
        status = "completed_provider_only_stage_a"
        summary = {
            "schema_version": 1,
            "status": status,
            "run_id": verified.run_id,
            "design_id": DEVELOPMENT_DESIGN_ID,
            "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
            "execution_revision_class": EXECUTION_REVISION_CLASS,
            "mechanism_revision_ordinal": MECHANISM_REVISION_ORDINAL,
            "manifest_sha256": verified.manifest_sha256,
            "source_sha256": verified.source_sha256,
            "credentials_read": credentials_read,
            "provider_accounting": accounting,
            "candidate_evaluations": 0,
            "cfd_calls": 0,
            "claim_boundary": verified.record["claim_boundary"],
            "survives_stage_a_v1": result.get("survives_stage_a_v1"),
        }
        write_json_atomic(run_dir / "summary.json", summary)
    except BaseException as exc:
        pending = exc
        write_json_atomic(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "design_id": DEVELOPMENT_DESIGN_ID,
                "base_method_design_id": BASE_DEVELOPMENT_DESIGN_ID,
                "execution_revision_class": EXECUTION_REVISION_CLASS,
                "mechanism_revision_ordinal": MECHANISM_REVISION_ORDINAL,
                "failure_type": type(exc).__name__,
                "safe_message": str(exc)[:1_024],
                "credentials_read": credentials_read,
            },
        )
    finally:
        queue_writer.close()
        journal_writer.close()
        source_writer.close()
        finalization = _finalize(run_dir, status=status)
    if pending is not None:
        raise pending
    assert summary is not None
    return {**summary, "finalization": finalization}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build-manifest", type=Path, metavar="PATH")
    modes.add_argument("--verify-manifest", type=Path, metavar="PATH")
    modes.add_argument("--live", type=Path, metavar="MANIFEST")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_SEALED_ORACLE_DIR)
    return parser


def _required(parser: argparse.ArgumentParser, value: object, flag: str) -> object:
    if value is None:
        parser.error(f"{flag} is required for the selected mode")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.build_manifest is not None:
        record: object = write_manifest(
            args.build_manifest,
            run_id=str(_required(parser, args.run_id, "--run-id")),
            output_dir=Path(_required(parser, args.output_dir, "--output-dir")),
            oracle_dir=args.oracle_dir,
        )
    elif args.verify_manifest is not None:
        if args.run_id is not None or args.output_dir is not None:
            parser.error("verification rejects build-only target arguments")
        verified = verify_manifest(args.verify_manifest)
        record = {
            "status": "verified_provider_ready",
            "run_id": verified.run_id,
            "manifest_sha256": verified.manifest_sha256,
            "source_sha256": verified.source_sha256,
            "output_dir": str(verified.output_dir),
            "credentials_read": False,
            "provider_dispatch_performed": False,
        }
    else:
        if args.run_id is not None or args.output_dir is not None:
            parser.error("--live rejects build-only target arguments")
        record = execute_with_dependencies(args.live, production_dependencies())
    print(json.dumps(record, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

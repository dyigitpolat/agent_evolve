"""Single-shard recovery for the failed Airfoil-v7 portfolio Stage A run.

The composition authenticates and strictly replays seven immutable v1r2
responses, delegates only the predetermined missing reflection and selector
calls 9--11, and preserves the original run unchanged.  Manifest construction
and verification never read credentials or contact a provider.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any, Protocol

from pydantic import BaseModel

AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    PORTFOLIO_SELECTION_TOOL_NAME,
    PydanticAIPortfolioSelectionPolicy,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    OutcomePublicationPolicy,
    SchemaRepairAttemptPolicy,
    create_production_queued_runner,
    structured_generation_outcome_record,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter
from agent_evolve.ports.agentic_generator import AgenticGenerator
from agent_evolve.ports.portfolio_selection import PortfolioSelectionPolicy
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_development import (
    DEFAULT_SEALED_ORACLE_DIR,
    DirectoryDevelopmentRecordSink,
    PreparedReflectionCall,
    _validate_card_for_call,
    execute_provider_ready_stage_a,
    prepare_provider_ready_stage_a,
)
from examples.development import (
    run_airfoil_v7_oracle_portfolio_stage_a as frozen,
)


WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
RUN_ROOT = ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7" / "portfolio_stage_a"
SOURCE_RUN_ID = "ae7_portfolio_stage_a_v1r2_grammar_0715_0333"
SOURCE_RUN_DIR = RUN_ROOT / SOURCE_RUN_ID
PROTOCOL_ARTIFACT = (
    ARTIFACT_ROOT
    / "119_airfoil_v7_stage_a_v1r2_single_shard_recovery_protocol.md"
)
PROTOCOL_ARTIFACT_SHA256 = (
    "8586bb580510b2429f72eaf05ba6b6aa0f389d5d3c8a3110ffb5330a79354647"
)

DESIGN_ID = "airfoil_v7_portfolio_stage_a_v1r2_single_shard_recovery"
EXECUTION_CLASS = "post_hoc_single_shard_protocol_deviation"
MANIFEST_KIND = "agent_evolve_airfoil_v7_portfolio_stage_a_recovery_manifest_v1"
MANIFEST_FRAMING = b"agent-evolve:airfoil-v7-stage-a-recovery-manifest:v1\x00"

MODEL = frozen.MODEL
ALLOWED_RESOLVED_MODELS = frozen.ALLOWED_RESOLVED_MODELS
ALLOWED_RESOLVED_PROVIDERS = frozen.ALLOWED_RESOLVED_PROVIDERS
PROVIDER_ONLY = frozen.PROVIDER_ONLY
MAX_INPUT_TOKENS = frozen.MAX_INPUT_TOKENS
MAX_OUTPUT_TOKENS = frozen.MAX_OUTPUT_TOKENS
QUEUE_MAX_IN_FLIGHT = frozen.QUEUE_MAX_IN_FLIGHT
QUEUE_MAX_PENDING = frozen.QUEUE_MAX_PENDING
QUEUE_MAX_ATTEMPTS = frozen.QUEUE_MAX_ATTEMPTS
ATTEMPT_TIMEOUT_NS = frozen.ATTEMPT_TIMEOUT_NS
BASE_BACKOFF_NS = frozen.BASE_BACKOFF_NS
MAX_BACKOFF_NS = frozen.MAX_BACKOFF_NS
JITTER_SEED = frozen.JITTER_SEED
JITTER_DOMAIN = frozen.JITTER_DOMAIN

MISSING_CALL_ID = "call_airfoil_oracle_stage_a_000006"
MISSING_SHARD_ID = "trim_1"
SELECTOR_CALL_IDS = (
    "call_airfoil_oracle_stage_a_000009",
    "call_airfoil_oracle_stage_a_000010",
    "call_airfoil_oracle_stage_a_000011",
)
LIVE_CALL_IDS = (MISSING_CALL_ID, *SELECTOR_CALL_IDS)
ARCHIVED_CONTENT_SHA256 = {
    "call_airfoil_oracle_stage_a_000001": (
        "f372a638455982355bbc9dfc1e5936380ad0b2096d70173e069858d9f1c5a958"
    ),
    "call_airfoil_oracle_stage_a_000002": (
        "f73323dc27f8f980f461121d5893acd57cd2d5654621ef8fde9de53970b17fc2"
    ),
    "call_airfoil_oracle_stage_a_000003": (
        "2a404dbc212b0423375b06fdae69f1d6ea3c58c3cf18e513a983daef1e768daa"
    ),
    "call_airfoil_oracle_stage_a_000004": (
        "ee0e8af903521a3ddc4fd5ed012c8f4d74ffb62517e2da2bf5c9f7f8e524e06a"
    ),
    "call_airfoil_oracle_stage_a_000005": (
        "3ad8ff8dc7103fe57db30ff7fe8d4ba5ee19c5f08994edf522c49873e7486ee0"
    ),
    "call_airfoil_oracle_stage_a_000007": (
        "c400664515431be72eedb5bba57149814a9425a451ec0ba93476560f2fb23a86"
    ),
    "call_airfoil_oracle_stage_a_000008": (
        "6e29f772ede889a4325fa9bf051ecae06713043b2a056993bbbe37463e988bdc"
    ),
}
EXPECTED_CARD_CONTENT_SHA256 = {
    "call_airfoil_oracle_stage_a_000001": (
        "fa69a955b239ceaf20ea0d623640c4c9c7615c101ab6cd6b6f700b5a4dbe1904"
    ),
    "call_airfoil_oracle_stage_a_000002": (
        "f8c6425653486b27e73738fe3247f65343829d45f7731467d058d6c5056e4378"
    ),
    "call_airfoil_oracle_stage_a_000003": (
        "59bdac1a4b7eab3ccf6f3f822210fb8e2bbb9e94dfd6b6241e5734ab0f49636a"
    ),
    "call_airfoil_oracle_stage_a_000004": (
        "5fd7d1f33d58c9467d17c67af2204bf7a7d49d2a53970c4623b4949c24debc88"
    ),
    "call_airfoil_oracle_stage_a_000005": (
        "7c070cc30d59ae1cd072ff9a94c05e68f86d602c3c562f07b3a5aaa00e8baf91"
    ),
    "call_airfoil_oracle_stage_a_000007": (
        "396765b5d9efd3af5761bec2048d46e70feb1cb18b433f33e256cde1fbd2ee90"
    ),
    "call_airfoil_oracle_stage_a_000008": (
        "4f3885ddb48e6a6a1b32cf74fdacc227302ae8ac8f62917730191e88412774f6"
    ),
}

SOURCE_FINALIZATION_SHA256 = (
    "a9e23cfcf787daf41a7b2d757890c4d3d86a284b6ab2a7be4f91464c21d891d7"
)
SOURCE_RECURSIVE_SHA256 = (
    "a2700081285c53075c39b2769a7a188c9fa8aa51822783ddd6f14e0d2506d0cb"
)
SOURCE_QUEUE_SHA256 = (
    "95fc290e8037ad25713a777a76cfada93dc1d8d2be78a2f720b69be0044a1786"
)
SOURCE_JOURNAL_SHA256 = (
    "320b344677c89d98cf57903303e2323aa78eefb64e940f2e906fff4be8cf7c4c"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _schema_sha256(output_type: type[object]) -> str:
    if not issubclass(output_type, BaseModel):
        raise TypeError("recovery output type must be a Pydantic BaseModel")
    return hashlib.sha256(
        _canonical_bytes(output_type.model_json_schema(mode="validation"))
    ).hexdigest()


def _request_binding(request: StructuredGenerationRequest[Any]) -> dict[str, object]:
    StructuredGenerationRequest.__post_init__(request)
    return {
        "call_id": request.call_id.value,
        "operation": request.operation,
        "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
        "output_schema_sha256": _schema_sha256(request.output_type),
        "output_tool_name": request.output_tool_name,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
    }


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    request_row: dict[str, object]
    response_row: dict[str, object]
    record_ordinal: int

    @property
    def call_id(self) -> str:
        value = self.request_row.get("call_id")
        if type(value) is not str:
            raise RuntimeError("archived request has no call ID")
        return value


class ArchivedReplayRunner:
    """Strictly replay seven rows and optionally delegate four closed live IDs."""

    def __init__(
        self,
        entries: tuple[ReplayEntry, ...],
        *,
        expected_bindings: Mapping[str, Mapping[str, object]] | None = None,
        live_runner: Callable[
            [StructuredGenerationRequest[Any]], Awaitable[object]
        ] | None = None,
        replay_sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        self._entries = {entry.call_id: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise ValueError("archived call IDs are not unique")
        self._expected = {
            key: dict(value) for key, value in (expected_bindings or {}).items()
        }
        self._live_runner = live_runner
        self._replay_sink = replay_sink
        self.observed_bindings: dict[str, dict[str, object]] = {}
        self.consumed: set[str] = set()

    def _validate_original_request(
        self,
        request: StructuredGenerationRequest[Any],
        entry: ReplayEntry,
    ) -> dict[str, object]:
        row = entry.request_row
        binding = _request_binding(request)
        if (
            row.get("call_id") != request.call_id.value
            or row.get("operation") != request.operation
            or row.get("prompt") != request.prompt
            or row.get("prompt_sha256") != binding["prompt_sha256"]
            or row.get("output_tool_name") != request.output_tool_name
            or row.get("max_output_tokens") != request.max_output_tokens
            or row.get("temperature") != request.temperature
        ):
            raise RuntimeError("archived request binding drifted")
        expected = self._expected.get(request.call_id.value)
        if expected is not None and binding != expected:
            raise RuntimeError("archived output schema or request drifted")
        return binding

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> object:
        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        call_id = request.call_id.value
        if call_id in self.consumed:
            raise RuntimeError("logical recovery call was already consumed")
        entry = self._entries.get(call_id)
        if entry is not None:
            binding = self._validate_original_request(request, entry)
            content = entry.response_row.get("content")
            content_sha = hashlib.sha256(_canonical_bytes(content)).hexdigest()
            if content_sha != entry.response_row.get("content_sha256"):
                raise RuntimeError("archived response content digest drifted")
            value = request.output_type.model_validate(content, strict=True)
            telemetry = entry.response_row.get("telemetry")
            if type(telemetry) is not dict:
                raise RuntimeError("archived response telemetry is absent")
            response = StructuredGenerationResponse(
                value=value,
                requested_model=str(telemetry["requested_model"]),
                resolved_model=str(telemetry["resolved_model"]),
                resolved_provider=str(telemetry["resolved_provider"]),
                provider_response_id=telemetry.get("provider_response_id"),
                finish_reason=telemetry.get("finish_reason"),
                input_tokens=int(telemetry["input_tokens"]),
                output_tokens=int(telemetry["output_tokens"]),
                reasoning_tokens=int(telemetry["reasoning_tokens"]),
                cache_read_tokens=int(telemetry["cache_read_tokens"]),
                cache_write_tokens=int(telemetry["cache_write_tokens"]),
                cost_usd=(
                    None
                    if telemetry.get("cost_usd") is None
                    else Decimal(str(telemetry["cost_usd"]))
                ),
                latency_ns=int(telemetry["latency_ns"]),
            )
            self.observed_bindings[call_id] = binding
            if self._replay_sink is not None:
                self._replay_sink(
                    {
                        "schema_version": 1,
                        "record_type": "archived_response_replayed",
                        "source_run_id": SOURCE_RUN_ID,
                        "source_finalization_sha256": SOURCE_FINALIZATION_SHA256,
                        "source_record_ordinal": entry.record_ordinal,
                        "request": binding,
                        "content_sha256": content_sha,
                    }
                )
            self.consumed.add(call_id)
            return AttemptedStructuredGenerationResponse(
                response=response,
                attempt_count=int(telemetry["attempt_count"]),
            )
        if call_id not in LIVE_CALL_IDS or self._live_runner is None:
            raise RuntimeError("request is not authorized by the recovery plan")
        if call_id == MISSING_CALL_ID:
            expected = self._expected.get(call_id)
            if expected is None or _request_binding(request) != expected:
                raise RuntimeError("missing reflection request drifted")
        elif (
            request.operation != "select_portfolio"
            or request.output_tool_name != PORTFOLIO_SELECTION_TOOL_NAME
            or request.max_output_tokens != MAX_OUTPUT_TOKENS
            or request.temperature != 0.0
        ):
            raise RuntimeError("selector request escaped the frozen live surface")
        self.consumed.add(call_id)
        return await self._live_runner(request)


@dataclass(frozen=True, slots=True)
class ArchiveEvidence:
    entries: tuple[ReplayEntry, ...]
    record: dict[str, object]
    bindings: dict[str, dict[str, object]]
    missing_binding: dict[str, object]


class _RequestCaptured(RuntimeError):
    pass


def _capture_missing_binding(call: PreparedReflectionCall) -> dict[str, object]:
    observed: dict[str, object] | None = None

    async def capture(request: StructuredGenerationRequest[Any]) -> object:
        nonlocal observed
        observed = _request_binding(request)
        raise _RequestCaptured

    async def run() -> None:
        try:
            await PydanticAIAgenticGenerator(capture).reflect(call.request)
        except _RequestCaptured:
            return
        raise RuntimeError("missing reflection request was not captured")

    asyncio.run(run())
    if observed is None:
        raise RuntimeError("missing reflection request binding is absent")
    return observed


def authenticate_source_archive() -> ArchiveEvidence:
    """Authenticate v1r2 and strictly revalidate all seven durable cards."""

    finalized_path = SOURCE_RUN_DIR / "finalized.json"
    queue_path = SOURCE_RUN_DIR / "provider_queue_outcomes.jsonl"
    journal_path = SOURCE_RUN_DIR / "prompt_response_journal.jsonl"
    finalized = frozen._load_object(finalized_path)
    count, recursive_sha = frozen._recursive_content_identity(SOURCE_RUN_DIR)
    if (
        finalized.get("status") != "failed"
        or finalized.get("finalization_sha256") != SOURCE_FINALIZATION_SHA256
        or count != 6
        or recursive_sha != SOURCE_RECURSIVE_SHA256
        or frozen._file_binding(queue_path)["sha256"] != SOURCE_QUEUE_SHA256
        or frozen._file_binding(journal_path)["sha256"] != SOURCE_JOURNAL_SHA256
    ):
        raise RuntimeError("sealed v1r2 recovery source drifted")

    queue_rows = frozen._jsonl_objects(queue_path)
    journal_rows = frozen._jsonl_objects(journal_path)
    requests = {
        str(row["call_id"]): row
        for row in journal_rows
        if row.get("record_type") == "request"
    }
    responses_with_ordinals = {
        str(row["call_id"]): (ordinal, row)
        for ordinal, row in enumerate(journal_rows)
        if row.get("record_type") == "response"
    }
    rejected = [
        row
        for row in journal_rows
        if row.get("record_type") == "response_route_rejected"
    ]
    queue_by_id = {str(row["task_id"]): row for row in queue_rows}
    expected_ids = {
        f"call_airfoil_oracle_stage_a_{ordinal:06d}" for ordinal in range(1, 9)
    }
    if (
        len(queue_rows) != 8
        or set(queue_by_id) != expected_ids
        or any(row.get("status") != "succeeded" for row in queue_rows)
        or any(len(row.get("attempts", [])) != 1 for row in queue_rows)
        or set(requests) != expected_ids
        or set(responses_with_ordinals) != set(ARCHIVED_CONTENT_SHA256)
        or len(rejected) != 1
        or rejected[0].get("call_id") != MISSING_CALL_ID
        or any(row.get("operation") == "select_portfolio" for row in journal_rows)
    ):
        raise RuntimeError("v1r2 ledger shape is not the frozen 8/7/1 failure")

    entries: list[ReplayEntry] = []
    for call_id, expected_sha in ARCHIVED_CONTENT_SHA256.items():
        ordinal, response_row = responses_with_ordinals[call_id]
        if (
            response_row.get("content_sha256") != expected_sha
            or hashlib.sha256(
                _canonical_bytes(response_row.get("content"))
            ).hexdigest()
            != expected_sha
        ):
            raise RuntimeError("archived content identity drifted")
        telemetry = response_row.get("telemetry")
        queue_response = queue_by_id[call_id].get("response")
        if type(telemetry) is not dict or type(queue_response) is not dict:
            raise RuntimeError("archived provider telemetry is absent")
        comparable = dict(telemetry)
        comparable.pop("attempt_count", None)
        if comparable != queue_response or telemetry.get("attempt_count") != 1:
            raise RuntimeError("journal and queue provider telemetry disagree")
        entries.append(
            ReplayEntry(
                request_row=requests[call_id],
                response_row=response_row,
                record_ordinal=ordinal,
            )
        )

    _, _, prepared = prepare_provider_ready_stage_a(
        run_dir=DEFAULT_SEALED_ORACLE_DIR,
        id_factory=DeterministicIdFactory("airfoil_oracle_stage_a"),
    )
    by_id = {call.request.call_id.value: call for call in prepared}
    if by_id[MISSING_CALL_ID].shard.shard_id != MISSING_SHARD_ID:
        raise RuntimeError("the frozen missing call no longer maps to trim_1")
    runner = ArchivedReplayRunner(tuple(entries))

    async def revalidate() -> list[dict[str, object]]:
        generator = PydanticAIAgenticGenerator(runner)
        cards: list[dict[str, object]] = []
        for call_id in sorted(ARCHIVED_CONTENT_SHA256):
            prepared_call = by_id[call_id]
            result = await generator.reflect(prepared_call.request)
            card = _validate_card_for_call(prepared_call, result)
            expected = EXPECTED_CARD_CONTENT_SHA256[call_id]
            if card.draft.content_sha256 != expected:
                raise RuntimeError("validated archived card identity drifted")
            cards.append(
                {
                    "call_id": call_id,
                    "shard_id": prepared_call.shard.shard_id,
                    "card_key": card.card_key,
                    "journal_content_sha256": ARCHIVED_CONTENT_SHA256[call_id],
                    "card_content_sha256": card.draft.content_sha256,
                }
            )
        return cards

    cards = asyncio.run(revalidate())
    missing_binding = _capture_missing_binding(by_id[MISSING_CALL_ID])
    missing_request = requests[MISSING_CALL_ID]
    if (
        missing_binding["prompt_sha256"] != missing_request.get("prompt_sha256")
        or missing_binding["call_id"] != MISSING_CALL_ID
        or missing_binding["max_output_tokens"] != MAX_OUTPUT_TOKENS
        or missing_binding["temperature"] != 0.0
    ):
        raise RuntimeError("the exact missing request drifted from v1r2")
    source_record = {
        "schema_version": 1,
        "run_id": SOURCE_RUN_ID,
        "status": "failed_after_eight_provider_responses_before_any_selector",
        "finalization_sha256": SOURCE_FINALIZATION_SHA256,
        "recursive_content_sha256": SOURCE_RECURSIVE_SHA256,
        "recursive_file_count": count,
        "queue_file": frozen._file_binding(queue_path),
        "journal_file": frozen._file_binding(journal_path),
        "finalized_file": frozen._file_binding(finalized_path),
        "archived_cards": cards,
        "missing_call_id": MISSING_CALL_ID,
        "missing_shard_id": MISSING_SHARD_ID,
        "provider_calls": 8,
        "durable_cards": 7,
        "selector_calls": 0,
        "candidate_evaluations": 0,
        "cfd_calls": 0,
    }
    return ArchiveEvidence(
        entries=tuple(entries),
        record=source_record,
        bindings=dict(runner.observed_bindings),
        missing_binding=missing_binding,
    )


def source_snapshot() -> dict[str, object]:
    """Bind the frozen implementation surface plus this thin composition."""

    return {
        "schema_version": 1,
        "frozen_implementation": frozen.source_snapshot(),
        "recovery_composition": frozen._file_binding(Path(__file__)),
    }


def provider_policy_record() -> dict[str, object]:
    policy = dict(frozen.provider_policy_record())
    policy.pop("reasoning_max_tokens", None)
    policy["reasoning"] = {
        "request_control": {"effort": "xhigh"},
        "hard_reasoning_token_cap": None,
        "accounting": "reasoning_tokens_included_in_output_tokens",
        "admission": "0 <= reasoning_tokens <= output_tokens",
    }
    return policy


def _protocol_binding() -> dict[str, object]:
    binding = frozen._file_binding(PROTOCOL_ARTIFACT)
    if binding["sha256"] != PROTOCOL_ARTIFACT_SHA256:
        raise RuntimeError("artifact 119 drifted")
    return binding


def build_manifest_record(
    *,
    run_id: str,
    output_dir: Path,
    oracle_dir: Path = DEFAULT_SEALED_ORACLE_DIR,
) -> dict[str, object]:
    """Build the provider-free four-call recovery commitment."""

    target = frozen._validate_target(run_id, output_dir)
    archive = authenticate_source_archive()
    oracle_path = oracle_dir.expanduser().resolve(strict=True)
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "output_dir": str(target),
        "design_id": DESIGN_ID,
        "method_design_id": frozen.DEVELOPMENT_DESIGN_ID,
        "execution_class": EXECUTION_CLASS,
        "claim_boundary": (
            "Post-hoc Stage-A development recovery and explicit protocol "
            "deviation; not held-out efficacy, genericity, SOTA, or wall-clock "
            "evidence."
        ),
        "protocol_artifact": _protocol_binding(),
        "source_archive": archive.record,
        "archived_request_bindings": archive.bindings,
        "missing_request_binding": archive.missing_binding,
        "oracle": frozen._oracle_binding(oracle_path),
        "development_plan": frozen._plan_record(oracle_path),
        "provider_policy": provider_policy_record(),
        "experiment": {
            "archived_reflection_replays": 7,
            "live_reflection_calls": 1,
            "live_selector_calls": 3,
            "live_logical_calls": 4,
            "maximum_live_physical_attempts": 8,
            "authorized_live_call_ids": list(LIVE_CALL_IDS),
            "candidate_evaluations": 0,
            "cfd_calls": 0,
            "execution": (
                "seven strict offline replays plus fixed call6; then concurrent "
                "selectors 9-11"
            ),
        },
        "source_snapshot": source_snapshot(),
        "credentials_read": False,
        "provider_dispatch_performed": False,
    }
    record["manifest_sha256"] = frozen._domain_sha256(record, MANIFEST_FRAMING)
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
    frozen.write_json_atomic(path, record)
    return record


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    path: Path
    record: dict[str, object]
    run_id: str
    output_dir: Path
    oracle_dir: Path
    manifest_sha256: str
    archive: ArchiveEvidence


def verify_manifest(
    path: Path,
    *,
    require_output_absent: bool = True,
) -> VerifiedManifest:
    resolved = path.expanduser().resolve(strict=True)
    record = frozen._load_object(resolved)
    claimed = record.get("manifest_sha256")
    unsigned = dict(record)
    unsigned.pop("manifest_sha256", None)
    if (
        type(claimed) is not str
        or claimed != frozen._domain_sha256(unsigned, MANIFEST_FRAMING)
        or record.get("schema_version") != 1
        or record.get("kind") != MANIFEST_KIND
        or record.get("design_id") != DESIGN_ID
        or record.get("method_design_id") != frozen.DEVELOPMENT_DESIGN_ID
        or record.get("execution_class") != EXECUTION_CLASS
    ):
        raise RuntimeError("recovery manifest identity or self-hash failed")
    run_id = record.get("run_id")
    output_value = record.get("output_dir")
    if type(run_id) is not str or type(output_value) is not str:
        raise RuntimeError("recovery manifest target is malformed")
    output_dir = frozen._validate_target(run_id, Path(output_value))
    if require_output_absent and output_dir.exists():
        raise FileExistsError(output_dir)
    oracle_record = record.get("oracle")
    if type(oracle_record) is not dict or type(oracle_record.get("path")) is not str:
        raise RuntimeError("recovery oracle binding is malformed")
    oracle_dir = Path(str(oracle_record["path"])).resolve(strict=True)
    archive = authenticate_source_archive()
    if (
        record.get("protocol_artifact") != _protocol_binding()
        or record.get("source_archive") != archive.record
        or record.get("archived_request_bindings") != archive.bindings
        or record.get("missing_request_binding") != archive.missing_binding
        or record.get("source_snapshot") != source_snapshot()
        or record.get("provider_policy") != provider_policy_record()
        or oracle_record != frozen._oracle_binding(oracle_dir)
        or record.get("development_plan") != frozen._plan_record(oracle_dir)
        or record.get("experiment", {}).get("authorized_live_call_ids")
        != list(LIVE_CALL_IDS)
    ):
        raise RuntimeError("recovery manifest dependency or archive binding drifted")
    return VerifiedManifest(
        path=resolved,
        record=record,
        run_id=run_id,
        output_dir=output_dir,
        oracle_dir=oracle_dir,
        manifest_sha256=claimed,
        archive=archive,
    )


def reverify_source(verified: VerifiedManifest) -> dict[str, object]:
    observed = source_snapshot()
    if observed != verified.record["source_snapshot"]:
        raise RuntimeError("recovery source snapshot drifted")
    frozen_sha = observed["frozen_implementation"]["sha256"]
    composition_sha = observed["recovery_composition"]["sha256"]
    return {
        "frozen_source_sha256": frozen_sha,
        "recovery_composition_sha256": composition_sha,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


class RecoveryAuditedStructuredRunner:
    """Journal live calls and enforce aggregate, rather than soft, reasoning."""

    def __init__(
        self,
        runner: Callable[
            [StructuredGenerationRequest[Any]], Awaitable[object]
        ],
        *,
        pre_dispatch: Callable[
            [StructuredGenerationRequest[Any]], Mapping[str, object]
        ],
        journal_sink: Callable[[Mapping[str, object]], None],
    ) -> None:
        self._runner = runner
        self._pre_dispatch = pre_dispatch
        self._journal_sink = journal_sink
        self._ordinal = 0
        self._lock = threading.Lock()

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> object:
        with self._lock:
            self._ordinal += 1
            ordinal = self._ordinal
        verification = self._pre_dispatch(request)
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
            raise TypeError("queued recovery runner returned an unsupported response")
        route_checks = {
            "requested_model": response.requested_model == MODEL,
            "resolved_model": response.resolved_model in ALLOWED_RESOLVED_MODELS,
            "resolved_provider": (
                response.resolved_provider in ALLOWED_RESOLVED_PROVIDERS
            ),
            "input_tokens": response.input_tokens <= MAX_INPUT_TOKENS,
            "output_tokens": response.output_tokens <= MAX_OUTPUT_TOKENS,
            "reasoning_in_output": (
                0 <= response.reasoning_tokens <= response.output_tokens
            ),
            "attempt_count": 1 <= attempts <= QUEUE_MAX_ATTEMPTS,
        }
        if not all(route_checks.values()):
            self._journal_sink(
                {
                    "schema_version": 1,
                    "record_type": "response_route_rejected",
                    "logical_call_ordinal": ordinal,
                    "call_id": request.call_id.value,
                    "route_checks": route_checks,
                }
            )
            raise RuntimeError("provider response violated recovery route policy")
        content = frozen._model_value(response.value)
        self._journal_sink(
            {
                "schema_version": 1,
                "record_type": "response",
                "logical_call_ordinal": ordinal,
                "call_id": request.call_id.value,
                "content": content,
                "content_sha256": hashlib.sha256(
                    _canonical_bytes(content)
                ).hexdigest(),
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
                    "cost_usd": (
                        None if response.cost_usd is None else str(response.cost_usd)
                    ),
                    "latency_ns": response.latency_ns,
                    "attempt_count": attempts,
                    "reasoning_token_accounting": "included_in_output_tokens",
                },
            }
        )
        return raw


@dataclass(slots=True)
class LiveStack:
    runner: Any
    hybrid: ArchivedReplayRunner
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
    archive: ArchiveEvidence,
    queue_sink: Callable[[Mapping[str, object]], None],
    journal_sink: Callable[[Mapping[str, object]], None],
    replay_sink: Callable[[Mapping[str, object]], None],
    pre_dispatch: Callable[
        [StructuredGenerationRequest[Any]], Mapping[str, object]
    ],
) -> LiveStack:
    """Compose the existing queue with xhigh reasoning and selective replay."""

    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=MODEL,
        max_connections=QUEUE_MAX_IN_FLIGHT,
        timeout_seconds=ATTEMPT_TIMEOUT_NS / 1_000_000_000,
        provider_options={"only": list(PROVIDER_ONLY), "allow_fallbacks": False},
        reasoning_config=OpenRouterReasoningConfig(effort="xhigh"),
        app_title="AgentEvolve AAAI 2027 Stage A single-shard recovery",
    )
    queued = create_production_queued_runner(
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
    audited = RecoveryAuditedStructuredRunner(
        queued,
        pre_dispatch=pre_dispatch,
        journal_sink=journal_sink,
    )
    expected = dict(archive.bindings)
    expected[MISSING_CALL_ID] = archive.missing_binding
    hybrid = ArchivedReplayRunner(
        archive.entries,
        expected_bindings=expected,
        live_runner=audited,
        replay_sink=replay_sink,
    )
    return LiveStack(
        runner=queued,
        hybrid=hybrid,
        generator=PydanticAIAgenticGenerator(hybrid),
        selector=PydanticAIPortfolioSelectionPolicy(hybrid),
    )


class LiveStackLike(Protocol):
    hybrid: ArchivedReplayRunner
    generator: AgenticGenerator
    selector: PortfolioSelectionPolicy

    async def __aenter__(self) -> "LiveStackLike": ...

    async def __aexit__(self, *_: object) -> None: ...


StageExecutor = Callable[..., Awaitable[dict[str, object]]]
StackFactory = Callable[..., LiveStackLike]


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    credential_loader: Callable[[], str]
    stack_factory: StackFactory
    stage_executor: StageExecutor = execute_provider_ready_stage_a
    enforce_accounting: bool = True


def production_dependencies() -> LiveDependencies:
    def load_key() -> str:
        from dotenv import load_dotenv

        load_dotenv(WORKSPACE_ROOT / ".env", override=False)
        return os.environ.get("OPENROUTER_API_KEY", "")

    return LiveDependencies(
        credential_loader=load_key,
        stack_factory=lambda **kwargs: create_live_stack(**kwargs),
    )


def _provider_accounting(
    queue_rows: Sequence[Mapping[str, object]],
    journal_rows: Sequence[Mapping[str, object]],
    replay_rows: Sequence[Mapping[str, object]],
    consumed: set[str],
) -> dict[str, object]:
    task_ids = tuple(str(row.get("task_id")) for row in queue_rows)
    request_ids = tuple(
        str(row.get("call_id"))
        for row in journal_rows
        if row.get("record_type") == "request"
    )
    response_ids = tuple(
        str(row.get("call_id"))
        for row in journal_rows
        if row.get("record_type") == "response"
    )
    replay_ids = tuple(
        str(row.get("request", {}).get("call_id"))
        for row in replay_rows
        if type(row.get("request")) is dict
    )
    passed = (
        len(queue_rows) == 4
        and set(task_ids) == set(LIVE_CALL_IDS)
        and all(row.get("status") == "succeeded" for row in queue_rows)
        and all(
            type(row.get("attempts")) is list
            and 1 <= len(row["attempts"]) <= QUEUE_MAX_ATTEMPTS
            for row in queue_rows
        )
        and len(request_ids) == 4
        and set(request_ids) == set(LIVE_CALL_IDS)
        and len(response_ids) == 4
        and set(response_ids) == set(LIVE_CALL_IDS)
        and len(replay_ids) == 7
        and set(replay_ids) == set(ARCHIVED_CONTENT_SHA256)
        and consumed == set(ARCHIVED_CONTENT_SHA256).union(LIVE_CALL_IDS)
    )
    return {
        "expected_live_logical_calls": 4,
        "terminal_queue_outcomes": len(queue_rows),
        "live_request_records": len(request_ids),
        "live_response_records": len(response_ids),
        "archived_replay_records": len(replay_ids),
        "authorized_live_call_ids": list(LIVE_CALL_IDS),
        "passed": passed,
    }


def _precredential_replay_records(
    archive: ArchiveEvidence,
) -> tuple[dict[str, object], ...]:
    cards = {
        str(row["call_id"]): row for row in archive.record["archived_cards"]
    }
    records: list[dict[str, object]] = []
    for entry in sorted(archive.entries, key=lambda item: item.call_id):
        call_id = entry.call_id
        records.append(
            {
                "schema_version": 1,
                "record_type": "archived_response_authenticated_precredential",
                "phase": "before_credential_load",
                "source_run_id": SOURCE_RUN_ID,
                "source_finalization_sha256": SOURCE_FINALIZATION_SHA256,
                "source_record_ordinal": entry.record_ordinal,
                "request": archive.bindings[call_id],
                "journal_content_sha256": ARCHIVED_CONTENT_SHA256[call_id],
                "validated_card": cards[call_id],
            }
        )
    return tuple(records)


def execute_with_dependencies(
    manifest_path: Path,
    dependencies: LiveDependencies,
) -> dict[str, object]:
    """Execute seven authenticated replays and exactly four authorized calls."""

    verified = verify_manifest(manifest_path, require_output_absent=True)
    run_dir = verified.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen._directory_fsync(run_dir.parent)
    shutil.copyfile(verified.path, run_dir / "launch_manifest.json")
    with (run_dir / "launch_manifest.json").open("rb") as stream:
        os.fsync(stream.fileno())
    frozen._directory_fsync(run_dir)

    queue_writer = frozen.DurableJsonlWriter(
        run_dir / "provider_queue_outcomes.jsonl"
    )
    journal_writer = frozen.DurableJsonlWriter(
        run_dir / "prompt_response_journal.jsonl"
    )
    replay_writer = frozen.DurableJsonlWriter(
        run_dir / "archived_replay_journal.jsonl"
    )
    source_writer = frozen.DurableJsonlWriter(run_dir / "source_verifications.jsonl")
    queue_rows: list[dict[str, object]] = []
    journal_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    replayed_call_ids: set[str] = set()
    rows_lock = threading.Lock()
    status = "failed"
    pending: BaseException | None = None
    summary: dict[str, object] | None = None
    credentials_read = False
    try:
        source_writer.write(
            {"stage": "post_run_directory_creation", **reverify_source(verified)}
        )
        for attestation in _precredential_replay_records(verified.archive):
            replay_writer.write(attestation)
            replay_rows.append(attestation)
        source_writer.write(
            {
                "stage": "archived_replays_authenticated_precredential",
                "archived_replay_attestations": len(replay_rows),
                **reverify_source(verified),
            }
        )

        def pre_dispatch(
            request: StructuredGenerationRequest[Any],
        ) -> Mapping[str, object]:
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
            with rows_lock:
                queue_rows.append(row)

        def journal_sink(record: Mapping[str, object]) -> None:
            row = dict(record)
            journal_writer.write(row)
            with rows_lock:
                journal_rows.append(row)

        def replay_sink(record: Mapping[str, object]) -> None:
            row = dict(record)
            request = row.get("request")
            call_id = (
                None if type(request) is not dict else request.get("call_id")
            )
            expected = {
                item["request"]["call_id"]: item for item in replay_rows
            }
            if (
                type(call_id) is not str
                or call_id not in expected
                or row.get("content_sha256")
                != expected[call_id]["journal_content_sha256"]
                or request != expected[call_id]["request"]
            ):
                raise RuntimeError(
                    "runtime replay escaped its precredential attestation"
                )
            with rows_lock:
                replayed_call_ids.add(call_id)

        source_writer.write(
            {"stage": "pre_credential_load", **reverify_source(verified)}
        )
        api_key = dependencies.credential_loader()
        credentials_read = True
        if type(api_key) is not str or not api_key.strip():
            raise RuntimeError("OPENROUTER_API_KEY is unavailable")
        stack = dependencies.stack_factory(
            api_key=api_key,
            archive=verified.archive,
            queue_sink=queue_sink,
            journal_sink=journal_sink,
            replay_sink=replay_sink,
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
        accounting = _provider_accounting(
            queue_rows,
            journal_rows,
            replay_rows,
            stack.hybrid.consumed,
        )
        accounting["runtime_archived_replays"] = len(replayed_call_ids)
        accounting["runtime_archived_replay_ids_match"] = (
            replayed_call_ids == set(ARCHIVED_CONTENT_SHA256)
        )
        accounting["passed"] = bool(
            accounting["passed"]
            and accounting["runtime_archived_replay_ids_match"]
        )
        if dependencies.enforce_accounting and not accounting["passed"]:
            raise RuntimeError("recovery accounting drifted from the 7+1+3 design")
        frozen.write_json_atomic(run_dir / "result.json", result)
        status = "completed_provider_only_stage_a_recovery"
        summary = {
            "schema_version": 1,
            "status": status,
            "run_id": verified.run_id,
            "design_id": DESIGN_ID,
            "method_design_id": frozen.DEVELOPMENT_DESIGN_ID,
            "execution_class": EXECUTION_CLASS,
            "manifest_sha256": verified.manifest_sha256,
            "credentials_read": credentials_read,
            "provider_accounting": accounting,
            "candidate_evaluations": 0,
            "cfd_calls": 0,
            "claim_boundary": verified.record["claim_boundary"],
            "survives_stage_a_v1": result.get("survives_stage_a_v1"),
        }
        frozen.write_json_atomic(run_dir / "summary.json", summary)
    except BaseException as exc:
        pending = exc
        frozen.write_json_atomic(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "design_id": DESIGN_ID,
                "execution_class": EXECUTION_CLASS,
                "failure_type": type(exc).__name__,
                "safe_message": str(exc)[:1_024],
                "credentials_read": credentials_read,
            },
        )
    finally:
        queue_writer.close()
        journal_writer.close()
        replay_writer.close()
        source_writer.close()
        finalization = frozen._finalize(run_dir, status=status)
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
            "archived_cards_revalidated": len(verified.archive.entries),
            "authorized_live_call_ids": list(LIVE_CALL_IDS),
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

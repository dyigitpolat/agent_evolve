#!/usr/bin/env python3
"""Three-call Airfoil-v7 qualifier for provider-wire policy v4.

``prepare`` is credential-free.  It verifies the finalized v2 run, rehydrates
the eight already-paid G1 records, reconstructs the accepted reflection and
the exact M/P/N views, chooses one aligned 20-option block by a frozen hash
rule, and seals the three provider requests in a finalized preparation
directory.

``live`` verifies that preparation and the complete closed source set before
reading ``OPENROUTER_API_KEY``.  It then dispatches exactly the three selected
M/P/N block calls concurrently.  The run stops after typed resolution and
generic semantic-health assessment: it never allocates actions, opens G2,
calls a selected-action evaluator, or performs CFD.

This is a representation/transport qualifier, not an efficacy experiment.
The historical reflection is replayed provenance and is not counted as a new
provider call.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Protocol

from pydantic import BaseModel

from agent_evolve.settings import load_credentials  # noqa: E402
from agent_evolve.application.action_forecast_partitioning import (
    ActionForecastHealthSubsetPolicyBinding,
    ActionForecastHealthPolicyBinding,
    action_forecast_block_call_id,
    build_action_forecast_partition_layout,
    lenient_action_forecast_health_v2_policy,
)
from agent_evolve.application.reflection_workflow import (
    ReflectionShardResult,
    ReflectionWorkflowResult,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.integrations.pydantic_ai.action_forecast import (
    ACTION_FORECAST_BLOCK_TOOL_NAME,
    ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256,
    ACTION_FORECAST_V4_POLICY_VERSION,
    PydanticAIActionForecastV4BlockPolicy,
    plan_action_forecast_v4_block_request,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (
    ProgressAwareOpenRouterConfig,
    ProgressAwareRetryMode,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai import queued_runner
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockRequest,
    ActionForecastBlockResult,
    ActionForecastPartitionLayout,
    ActionForecastPartitionPolicyBinding,
    ActionForecastRequest,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationResult,
)
from agent_evolve.ports.artifact_store import canonical_json_bytes, decode_json_bytes
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
)
from examples.development import airfoil_v7_two_stage_agent_evolution as airfoil
from examples.development import durable_run_artifacts
from examples.development import run_airfoil_v7_two_stage_generation as v2_launcher
from examples.development.airfoil_v7_two_stage_agent_evolution import (
    AirfoilTwoStageForecastArms,
    PreparedAirfoilTwoStageGeneration,
)
from examples.development.durable_run_artifacts import (
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    file_identity,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)

try:
    # This generic overload is intentionally the only health seam used by the
    # pilot.  Until it lands, preparation remains usable and live execution
    # fails closed instead of embedding Airfoil-specific health logic here.
    from agent_evolve.application.action_forecast_partitioning import (
        assess_resolved_action_forecast_block_health as _GENERIC_BLOCK_HEALTH,
        assess_resolved_action_forecast_block_subset_health as _GENERIC_SUBSET_HEALTH,
    )
except ImportError:  # pragma: no cover - removed once the generic API lands.
    _GENERIC_BLOCK_HEALTH = None
    _GENERIC_SUBSET_HEALTH = None


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_RUN_ROOT = ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7" / "wire_v3"
DEFAULT_FROZEN_V2_RUN = (
    ARTIFACT_ROOT
    / "experiment_logs"
    / "airfoil_v7"
    / "two_stage"
    / "ae7_generic_two_stage_generation_v2_20260715"
)
DEFAULT_HISTORICAL_V4_PREPARED_RUN = (
    DEFAULT_RUN_ROOT / "ae7_forecast_wire_v4_qualifier_prepared_20260715"
)

MODEL = "deepseek/deepseek-v4-pro"
CANONICAL_MODEL = "deepseek/deepseek-v4-pro-20260423"
ALLOWED_RESOLVED_MODELS = (MODEL, CANONICAL_MODEL)
PROVIDER_SLUG = "streamlake"
RESOLVED_PROVIDER = "StreamLake"
MAX_OUTPUT_TOKENS = 384_000
MAX_ATTEMPTS = 2
CONCURRENCY = 3
MAX_PENDING = 3
CONNECT_TIMEOUT_SECONDS = 90.0
FIRST_EVENT_TIMEOUT_SECONDS = 180
IDLE_TIMEOUT_SECONDS = 120
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 2_026_071_503
JITTER_DOMAIN = "airfoil-v7-forecast-wire-v3-pilot"
PROGRESS_MAX_UNFSYNCED_ROWS = 64

PARTITION_POLICY_ID = "fixed_contiguous_action_forecast_partition"
PARTITION_POLICY_VERSION = 1
PARTITION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:fixed-contiguous-action-forecast-partition:v1:"
    b"maximal-complete-contiguous-rows-under-row-and-metric-cell-bounds"
).hexdigest()
BLOCK_ROWS = 20
BLOCK_METRIC_CELLS = 40
EXPECTED_PROVIDER_WIRE_POLICY_VERSION = 4
EXPECTED_PROVIDER_WIRE_POLICY_DEFINITION_SHA256 = (
    "79cf864675cb9500062ecd86ce591c637adb3f0ec1e980576f212a47d3ad070a"
)
EXPECTED_HEALTH_POLICY_VERSION = 2
EXPECTED_HEALTH_POLICY_DEFINITION_SHA256 = (
    "14fc199feff062d231e9b7721080816ebdb4b55ba7688195a7172c6b36dc57ae"
)
EXPECTED_HISTORICAL_PREPARATION_COMMITMENT_SHA256 = (
    "37e027c70217d526fb6640e0e3e547ec72eb313fdca23a3290a9334b582c3088"
)
EXPECTED_HISTORICAL_PLANNED_WAVE_FILE_SHA256 = (
    "943a76f08d088c357773a937eac2e95442da1a05c004d56e2260d5ae6473882d"
)
ELIGIBLE_SUBSET_POLICY_ID = "eligible_target_action_forecast_health_subset"
ELIGIBLE_SUBSET_POLICY_VERSION = 1
ELIGIBLE_SUBSET_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:eligible-target-action-forecast-health-subset:v1:"
    b"within-selected-block-rows-whose-exact-option-id-is-in-the-"
    b"precommitted-allocation-eligible-option-set"
).hexdigest()

EXPECTED_V2_FINALIZATION_SHA256 = (
    "03418756ebd5031222261c9e06d776a54a0c6102d3da155e1bb8efff464ca752"
)
EXPECTED_V2_RECURSIVE_SHA256 = (
    "c6daea0851af5dcb0eb9c2983ca6d41cc3d1f5a5f3a4fef32242bff8fee26124"
)
EXPECTED_V2_REFLECTION_SHA256 = (
    "b1b114ce7104fb669db27b47afbe7aa95ce69000e8791ce07713051000d3db8f"
)
EXPECTED_V2_CARDS_VIEWS_SHA256 = (
    "86c1d37f1fca79db6a7254eafee11f4f7e2f82c1d2cf0f607231d3ea61c60b1b"
)
EXPECTED_V2_READINESS_SHA256 = (
    "d5047b200b97a6843d821007655c7d6a42c1644f8d67b9cfd3485fc468bb5612"
)
EXPECTED_V2_MANIFEST_COMMITMENT_SHA256 = (
    "b43d9ebd3eec47ad1db76e2b79a2634fb6dd9a14b241f1883e5b9d9ab0e0f483"
)
EXPECTED_CONTRACT_SHA256 = (
    "69a4111b1b5654b13bace5ae9400f27fba45be97803334d8f0f9be9802610f92"
)
EXPECTED_MEMORY_CARD_SNAPSHOT_SHA256 = (
    "73e235b961a0582199b2ad1e409628d371315bd1992444903fd6d15503b8d4a1"
)
EXPECTED_PLACEBO_CARD_SNAPSHOT_SHA256 = (
    "e19973455d52fa38cf562586d9ec73b99bed5380504c185fb087eed824226d63"
)
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "098ad451957b858244aada260fb376e4ab949c3d4da5d16faf9318a5b3a20418"
)
EXPECTED_MEMORY_VIEW_RECEIPT_SHA256 = (
    "9bad5ec86c731788f019ecda9b8629426265be414ba33b59e119272b7fcf1200"
)
EXPECTED_PLACEBO_VIEW_RECEIPT_SHA256 = (
    "b5dee7187be9bfbb8464b99ee7af9e0fa32ef79d82191b75589eb1ce39400506"
)
EXPECTED_V2_ARM_REQUEST_SHA256S = (
    "b5acd4ceb3de7e23ea437deae82f7d0a9f2338b8ce83992ab67be2a6d3bb54e7",
    "ab45ed45f594ded48b4dba96c35eef971d04f7457ce6570877ee7aadc052ba13",
    "25eaf1f32fb34a4b6a2834782526619e563f7585c508f15e82b613786716488e",
)
EXPECTED_CURRENT_ARM_REQUEST_SHA256S = (
    "1bb49e68332cc0e60fd5a5e0c5aa49d02bb39cc71760c527564a2230f47087cf",
    "1a69dbd9fd5685f8d005d627726fe207cc1d89d2cfbca209fc19a692ba447430",
    "4a4d571ea6a8f852c7dcee4b78a2051131c91c1cbe7a1f2ac0268996162c7eac",
)
EXPECTED_V2_INSTRUCTION_SHA256 = (
    "166fc92d7cfc1f0f5bad7a1dc66b9218765cc4e6dfd8d0eeffba23e417b02237"
)
EXPECTED_CURRENT_INSTRUCTION_SHA256 = (
    "823f0d685b0fa69da8fac3432d121e86b3b1f4e2d052ce1293fa6f456b8b374a"
)

_SELECTION_FRAMING = b"agent-evolve:forecast-wire-pilot-block-selection:v1\x00"
_PREPARED_FRAMING = b"agent-evolve:forecast-wire-v3-prepared:v1\x00"
_MANIFEST_FRAMING = b"agent-evolve:forecast-wire-v3-live-manifest:v1\x00"


class ForecastWirePilotError(RuntimeError):
    """Fail-closed error for this one representation qualifier."""


class GenericBlockHealthUnavailable(ForecastWirePilotError):
    """The provider-neutral block-health overload has not landed yet."""


def _load_object(path: Path) -> dict[str, object]:
    value = decode_json_bytes(path.expanduser().resolve(strict=True).read_bytes())
    if type(value) is not dict:
        raise ForecastWirePilotError(f"{path.name} must contain one JSON object")
    return value


def _sha256_record(framing: bytes, value: object) -> str:
    return hashlib.sha256(framing + canonical_json_bytes(value)).hexdigest()


def _source_paths() -> tuple[Path, ...]:
    paths = set(v2_launcher._source_paths())
    paths.add(Path(__file__))
    test_path = AGENT_EVOLVE_ROOT / "tests" / "test_run_airfoil_v7_forecast_wire_v3_pilot.py"
    if test_path.is_file():
        paths.add(test_path)
    return tuple(sorted(paths, key=lambda value: value.resolve().as_posix()))


def current_source_identity() -> dict[str, object]:
    return source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)


def build_config() -> ProgressAwareOpenRouterConfig:
    return ProgressAwareOpenRouterConfig(
        model_name=MODEL,
        provider_only=(PROVIDER_SLUG,),
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=FIRST_EVENT_TIMEOUT_SECONDS * 1_000_000_000,
            idle_timeout_ns=IDLE_TIMEOUT_SECONDS * 1_000_000_000,
            absolute_timeout_ns=None,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=5_000_000_000,
                transport_retire_timeout_ns=5_000_000_000,
            ),
        ),
        max_connections=CONCURRENCY,
        max_pending=MAX_PENDING,
        max_attempts=MAX_ATTEMPTS,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        jitter_seed=JITTER_SEED,
        jitter_domain=JITTER_DOMAIN,
        app_title="AgentEvolve Airfoil-v7 forecast-wire v3 pilot",
        reasoning_config=OpenRouterReasoningConfig(effort="high"),
        retry_mode=ProgressAwareRetryMode.TRANSPORT_ONLY,
    )


def partition_policy() -> ActionForecastPartitionPolicyBinding:
    return ActionForecastPartitionPolicyBinding(
        policy_id=PARTITION_POLICY_ID,
        policy_version=PARTITION_POLICY_VERSION,
        policy_definition_sha256=PARTITION_POLICY_DEFINITION_SHA256,
        max_rows_per_block=BLOCK_ROWS,
        max_metric_cells_per_block=BLOCK_METRIC_CELLS,
    )


def eligible_subset_policy() -> ActionForecastHealthSubsetPolicyBinding:
    return ActionForecastHealthSubsetPolicyBinding(
        policy_id=ELIGIBLE_SUBSET_POLICY_ID,
        policy_version=ELIGIBLE_SUBSET_POLICY_VERSION,
        policy_definition_sha256=ELIGIBLE_SUBSET_POLICY_DEFINITION_SHA256,
    )


def _telemetry_from_record(value: object) -> AgenticCallTelemetry:
    if type(value) is not dict:
        raise ForecastWirePilotError("historical reflection telemetry is malformed")
    cost = value.get("cost_usd")
    try:
        return AgenticCallTelemetry(
            requested_model=str(value["requested_model"]),
            resolved_model=str(value["resolved_model"]),
            resolved_provider=str(value["resolved_provider"]),
            provider_response_id=(
                None
                if value.get("provider_response_id") is None
                else str(value["provider_response_id"])
            ),
            finish_reason=(
                None
                if value.get("finish_reason") is None
                else str(value["finish_reason"])
            ),
            input_tokens=int(value["input_tokens"]),
            output_tokens=int(value["output_tokens"]),
            reasoning_tokens=int(value["reasoning_tokens"]),
            cache_read_tokens=int(value["cache_read_tokens"]),
            cache_write_tokens=int(value["cache_write_tokens"]),
            cost_usd=None if cost is None else Decimal(str(cost)),
            latency_ns=int(value["latency_ns"]),
            attempt_count=int(value["attempt_count"]),
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        raise ForecastWirePilotError(
            "historical reflection telemetry failed typed reconstruction"
        ) from error


def _draft_from_record(value: object) -> InsightDraft:
    if type(value) is not dict:
        raise ForecastWirePilotError("historical reflection draft is malformed")
    try:
        predictions = tuple(
            MetricEffectPrediction(
                metric_id=str(item["metric_id"]),
                direction=MetricEffectDirection(str(item["direction"])),
            )
            for item in value["effect_predictions"]
            if type(item) is dict
        )
        if len(predictions) != len(value["effect_predictions"]):
            raise TypeError("one effect prediction is not an object")
        return InsightDraft(
            claim=str(value["claim"]),
            trigger=str(value["trigger"]),
            mechanism=str(value["mechanism"]),
            affected_paths=tuple(str(item) for item in value["affected_paths"]),
            evidence_summary=str(value["evidence_summary"]),
            confidence=float(value["confidence"]),
            evidence_contrast_ids=tuple(
                str(item) for item in value["evidence_contrast_ids"]
            ),
            effect_predictions=predictions,
            recommended_option_families=tuple(
                str(item) for item in value["recommended_option_families"]
            ),
            recommended_option_ids=tuple(
                str(item) for item in value["recommended_option_ids"]
            ),
            action_template=(
                None
                if value.get("action_template") is None
                else str(value["action_template"])
            ),
            falsification_condition=(
                None
                if value.get("falsification_condition") is None
                else str(value["falsification_condition"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ForecastWirePilotError(
            "historical reflection draft failed typed reconstruction"
        ) from error


def reflection_from_record(
    record: Mapping[str, object],
    *,
    expected_contrast_ids: tuple[str, ...],
) -> ReflectionWorkflowResult:
    """Reconstruct and validate the accepted shared v2 generation result."""

    if record.get("schema_version") != 1:
        raise ForecastWirePilotError("historical reflection schema changed")
    if record.get("logical_calls_used") != 1 or record.get("logical_call_ids") != [
        "call_ae7x4v2_000001"
    ]:
        raise ForecastWirePilotError("historical reflection call binding changed")
    rows = record.get("cards")
    if type(rows) is not list or len(rows) != 8:
        raise ForecastWirePilotError("historical reflection must contain eight cards")
    telemetry = _telemetry_from_record(record.get("telemetry"))
    parsed: list[tuple[str, LLMCallId, InsightDraft]] = []
    for row in rows:
        if type(row) is not dict:
            raise ForecastWirePilotError("historical reflection card is malformed")
        draft = _draft_from_record(row.get("draft"))
        if row.get("draft_content_sha256") != draft.content_sha256:
            raise ForecastWirePilotError("historical reflection draft hash changed")
        try:
            parsed.append(
                (
                    str(row["contrast_id"]),
                    LLMCallId(str(row["call_id"])),
                    draft,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ForecastWirePilotError(
                "historical reflection card binding is malformed"
            ) from error
    if tuple(sorted(value[0] for value in parsed)) != expected_contrast_ids:
        raise ForecastWirePilotError("historical reflection contrast coverage changed")
    generation = ReflectionGenerationResult(
        insights=tuple(value[2] for value in parsed),
        telemetry=telemetry,
    )
    result = ReflectionWorkflowResult(
        tuple(
            ReflectionShardResult(
                contrast_id=contrast_id,
                call_id=call_id,
                draft=draft,
                generation_result=generation,
            )
            for contrast_id, call_id, draft in sorted(parsed)
        )
    )
    if result.logical_llm_calls_used != 1:
        raise ForecastWirePilotError("historical reflection no longer has one call")
    return result


def _verify_v2_run(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    root = run_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    if (
        finalization.get("finalization_sha256")
        != EXPECTED_V2_FINALIZATION_SHA256
        or finalization.get("recursive_content_sha256")
        != EXPECTED_V2_RECURSIVE_SHA256
        or finalization.get("status") != "transport_incomplete"
    ):
        raise ForecastWirePilotError("frozen v2 finalization identity changed")
    files = finalization.get("files")
    if type(files) is not dict:
        raise ForecastWirePilotError("frozen v2 finalization lacks file bindings")
    expected_files = {
        "reflection_result.json": EXPECTED_V2_REFLECTION_SHA256,
        "cards_views_requests.json": EXPECTED_V2_CARDS_VIEWS_SHA256,
        "provider_free_readiness.json": EXPECTED_V2_READINESS_SHA256,
    }
    for name, expected_sha256 in expected_files.items():
        binding = files.get(name)
        if type(binding) is not dict or binding.get("sha256") != expected_sha256:
            raise ForecastWirePilotError(f"frozen v2 {name} identity changed")
    manifest = _load_object(root / "manifest.json")
    if (
        manifest.get("manifest_commitment_sha256")
        != EXPECTED_V2_MANIFEST_COMMITMENT_SHA256
    ):
        raise ForecastWirePilotError("frozen v2 manifest commitment changed")
    return finalization, manifest


def _request_plan_payload(
    request: StructuredGenerationRequest[Any],
) -> dict[str, object]:
    request.__post_init__()
    schema = request.output_type.model_json_schema()
    return {
        "schema_version": 1,
        "call_contract": {
            "call_id": request.call_id.value,
            "operation": request.operation,
            "prompt_utf8_bytes": len(request.prompt.encode("utf-8")),
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "output_type": request.output_type.__name__,
            "output_tool_name": request.output_tool_name,
            "schema_sha256": hashlib.sha256(canonical_json_bytes(schema)).hexdigest(),
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "provider_attempt_id": None,
        },
        "prompt": request.prompt,
        "output_json_schema": schema,
    }


def _schema_contains_provider_number(value: object) -> bool:
    if type(value) is dict:
        if value.get("type") in {"integer", "number"}:
            return True
        return any(_schema_contains_provider_number(item) for item in value.values())
    if type(value) is list:
        return any(_schema_contains_provider_number(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class PilotBundle:
    frozen_v2_run: Path
    oracle_dir: Path
    v2_finalization: Mapping[str, object]
    preparation: PreparedAirfoilTwoStageGeneration
    reflection: ReflectionWorkflowResult
    arms: AirfoilTwoStageForecastArms
    arm_requests: tuple[ActionForecastRequest, ...]
    layout: ActionForecastPartitionLayout
    selected_block_index: int
    eligible_g2_global_row_indices: tuple[int, ...]
    selected_block_requests: tuple[ActionForecastBlockRequest, ...]
    planned_calls: tuple[StructuredGenerationRequest[Any], ...]
    selection_record: Mapping[str, object]
    arm_request_sha256s: tuple[str, str, str]
    selected_block_records: tuple[
        Mapping[str, object], Mapping[str, object], Mapping[str, object]
    ]
    planned_call_payloads: tuple[
        Mapping[str, object], Mapping[str, object], Mapping[str, object]
    ]
    layout_record: Mapping[str, object]


def _validate_rebuilt_arms(
    arms: AirfoilTwoStageForecastArms,
    recorded: Mapping[str, object],
) -> None:
    arms.__post_init__()
    if (
        arms.preparation.contract.identity_sha256 != EXPECTED_CONTRACT_SHA256
        or arms.memory_request.card_snapshot_sha256
        != EXPECTED_MEMORY_CARD_SNAPSHOT_SHA256
        or arms.placebo_request.card_snapshot_sha256
        != EXPECTED_PLACEBO_CARD_SNAPSHOT_SHA256
        or arms.source_registry.registry_sha256 != EXPECTED_SOURCE_REGISTRY_SHA256
        or arms.memory_receipt.receipt_sha256
        != EXPECTED_MEMORY_VIEW_RECEIPT_SHA256
        or arms.placebo_receipt.receipt_sha256
        != EXPECTED_PLACEBO_VIEW_RECEIPT_SHA256
    ):
        raise ForecastWirePilotError("reconstructed M/P/N identity changed")
    recorded_arms = recorded.get("arms")
    if type(recorded_arms) is not dict:
        raise ForecastWirePilotError("frozen cards/views record is malformed")
    rebuilt = arms.to_record()
    for field in ("source_registry", "entries", "eligible_g2_option_ids"):
        if rebuilt.get(field) != recorded_arms.get(field):
            raise ForecastWirePilotError(
                f"reconstructed cards/views {field} changed"
            )
    recorded_treatments = recorded_arms.get("arms")
    if type(recorded_treatments) is not dict:
        raise ForecastWirePilotError("frozen arm treatment record is malformed")
    for arm, snapshot, receipt in (
        (
            "m",
            arms.memory_request.card_snapshot_sha256,
            arms.memory_receipt.receipt_sha256,
        ),
        (
            "p",
            arms.placebo_request.card_snapshot_sha256,
            arms.placebo_receipt.receipt_sha256,
        ),
    ):
        value = recorded_treatments.get(arm)
        if (
            type(value) is not dict
            or value.get("card_snapshot_sha256") != snapshot
            or type(value.get("view_receipt")) is not dict
            or value["view_receipt"].get("receipt_sha256") != receipt
        ):
            raise ForecastWirePilotError(f"reconstructed {arm} treatment changed")

    recorded_requests = recorded.get("arm_requests")
    if type(recorded_requests) is not list or len(recorded_requests) != 3:
        raise ForecastWirePilotError("frozen arm request ledger is malformed")
    rebuilt_requests = (
        arms.memory_request,
        arms.placebo_request,
        arms.catalog_only_request,
    )
    for index, (arm, request, old_sha256, current_sha256) in enumerate(
        zip(
            ("m", "p", "n"),
            rebuilt_requests,
            EXPECTED_V2_ARM_REQUEST_SHA256S,
            EXPECTED_CURRENT_ARM_REQUEST_SHA256S,
            strict=True,
        )
    ):
        row = recorded_requests[index]
        if (
            type(row) is not dict
            or row.get("arm") != arm
            or row.get("request_sha256") != old_sha256
            or request.request_sha256 != current_sha256
        ):
            raise ForecastWirePilotError(
                f"frozen/current {arm} request order or identity changed"
            )
        frozen_request = row.get("request")
        if type(frozen_request) is not dict:
            raise ForecastWirePilotError(f"frozen {arm} request is malformed")
        current_request = request.to_record()
        frozen_comparable = dict(frozen_request)
        current_comparable = dict(current_request)
        frozen_instruction = frozen_comparable.pop("instruction_sha256", None)
        current_instruction = current_comparable.pop("instruction_sha256", None)
        if (
            frozen_instruction != EXPECTED_V2_INSTRUCTION_SHA256
            or current_instruction != EXPECTED_CURRENT_INSTRUCTION_SHA256
            or frozen_comparable != current_comparable
        ):
            raise ForecastWirePilotError(
                f"current {arm} request differs beyond the intentional instruction"
            )


_BASE_WIRE_FIELDS = frozenset(
    {
        "probability_valid_codes",
        "median_effect_codes",
        "lower_uncertainty_codes",
        "upper_uncertainty_codes",
    }
)


def _exact_enum(value: object) -> bool:
    return (
        type(value) is dict
        and value.get("type") == "string"
        and type(value.get("enum")) is list
        and bool(value["enum"])
        and all(type(item) is str and item for item in value["enum"])
        and len(set(value["enum"])) == len(value["enum"])
    )


def _validate_wire_schema(
    schema: Mapping[str, object],
    *,
    evidence_expected: bool,
) -> None:
    expected_fields = set(_BASE_WIRE_FIELDS)
    if evidence_expected:
        expected_fields.add("evidence_slot_codes")
    properties = schema.get("properties")
    required = schema.get("required")
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or type(properties) is not dict
        or set(properties) != expected_fields
        or type(required) is not list
        or len(required) != len(expected_fields)
        or set(required) != expected_fields
        or _schema_contains_provider_number(schema)
    ):
        raise ForecastWirePilotError("pilot schema is not an exact closed enum object")
    probability = properties["probability_valid_codes"]
    if (
        type(probability) is not dict
        or probability.get("type") != "array"
        or probability.get("minItems") != BLOCK_ROWS
        or probability.get("maxItems") != BLOCK_ROWS
        or not _exact_enum(probability.get("items"))
    ):
        raise ForecastWirePilotError("pilot validity wire is not an exact 20-vector")
    for field_name in expected_fields - {"probability_valid_codes"}:
        matrix = properties[field_name]
        if (
            type(matrix) is not dict
            or matrix.get("type") != "array"
            or matrix.get("minItems") != BLOCK_ROWS
            or matrix.get("maxItems") != BLOCK_ROWS
        ):
            raise ForecastWirePilotError(f"pilot {field_name} row shape changed")
        row = matrix.get("items")
        if (
            type(row) is not dict
            or row.get("type") != "array"
            or row.get("minItems") != 2
            or row.get("maxItems") != 2
            or not _exact_enum(row.get("items"))
        ):
            raise ForecastWirePilotError(f"pilot {field_name} is not exact 20x2")


def _validate_planned_wire_contracts(
    plans: tuple[StructuredGenerationRequest[Any], ...],
) -> None:
    if len(plans) != 3:
        raise ForecastWirePilotError("pilot wire preflight requires M/P/N")
    schemas = tuple(value.output_type.model_json_schema() for value in plans)
    _validate_wire_schema(schemas[0], evidence_expected=True)
    _validate_wire_schema(schemas[1], evidence_expected=True)
    _validate_wire_schema(schemas[2], evidence_expected=False)
    if schemas[0] != schemas[1]:
        raise ForecastWirePilotError("M/P grounded block schemas differ")
    for index, plan in enumerate(plans):
        lines = plan.prompt.splitlines()
        try:
            frame = json.loads(
                lines[lines.index("ALL-OPTION ACTION FORECAST CONTRACT") + 1]
            )
        except (ValueError, IndexError, json.JSONDecodeError) as error:
            raise ForecastWirePilotError("pilot prompt frame is malformed") from error
        expected_evidence = index < 2
        if (
            type(frame) is not dict
            or type(frame.get("ordered_options")) is not list
            or len(frame["ordered_options"]) != BLOCK_ROWS
            or [value.get("global_row_index") for value in frame["ordered_options"]]
            != list(
                range(
                    int(frame["forecast_frame"]["global_row_start"]),
                    int(frame["forecast_frame"]["global_row_stop"]),
                )
            )
            or type(frame.get("forecast_metrics")) is not list
            or len(frame["forecast_metrics"]) != 2
            or frame.get("output_contract", {}).get(
                "one_evidence_slot_required_per_metric"
            )
            is not expected_evidence
        ):
            raise ForecastWirePilotError("pilot prompt 20x2/global/evidence frame changed")


def build_pilot_bundle(
    *,
    frozen_v2_run: Path = DEFAULT_FROZEN_V2_RUN,
    oracle_dir: Path = airfoil.DEFAULT_SEALED_ORACLE_DIR,
) -> PilotBundle:
    """Rebuild all provider requests without reading a credential."""

    frozen_root = frozen_v2_run.expanduser().resolve(strict=True)
    finalization, _manifest = _verify_v2_run(frozen_root)
    preparation = airfoil.prepare_airfoil_v7_two_stage_generation(oracle_dir)
    expected_contrasts = tuple(
        sorted(value.contrast_id for value in preparation.observations)
    )
    reflection_record = _load_object(frozen_root / "reflection_result.json")
    reflection = reflection_from_record(
        reflection_record,
        expected_contrast_ids=expected_contrasts,
    )
    arms = airfoil.build_airfoil_v7_forecast_arms(preparation, reflection)
    recorded_cards = _load_object(frozen_root / "cards_views_requests.json")
    _validate_rebuilt_arms(arms, recorded_cards)

    requests = (
        arms.memory_request,
        arms.placebo_request,
        arms.catalog_only_request,
    )
    policy = partition_policy()
    layouts = tuple(build_action_forecast_partition_layout(value, policy) for value in requests)
    if len({value.layout_sha256 for value in layouts}) != 1:
        raise ForecastWirePilotError("M/P/N partition layouts differ")
    layout = layouts[0]
    if layout.block_count != 4 or any(value.row_count != BLOCK_ROWS for value in layout.blocks):
        raise ForecastWirePilotError("Airfoil pilot layout is not four 20-row blocks")

    health_policy = lenient_action_forecast_health_v2_policy()
    subset_policy = eligible_subset_policy()
    if (
        ACTION_FORECAST_V4_POLICY_VERSION
        != EXPECTED_PROVIDER_WIRE_POLICY_VERSION
        or ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
        != EXPECTED_PROVIDER_WIRE_POLICY_DEFINITION_SHA256
        or health_policy.policy_version != EXPECTED_HEALTH_POLICY_VERSION
        or health_policy.policy_definition_sha256
        != EXPECTED_HEALTH_POLICY_DEFINITION_SHA256
        or health_policy.extreme_abs_normalized_median != 32.0
    ):
        raise ForecastWirePilotError("provider-wire v4 health binding changed")
    selection_payload = {
        "schema_version": 1,
        "source_v2_finalization_sha256": EXPECTED_V2_FINALIZATION_SHA256,
        "source_reflection_result_sha256": EXPECTED_V2_REFLECTION_SHA256,
        "finite_contract_identity_sha256": preparation.contract.identity_sha256,
        "partition_layout_sha256": layout.layout_sha256,
        "action_forecast_policy_definition_sha256": (
            ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
        ),
        "health_policy_binding_sha256": health_policy.binding_sha256,
        "eligible_subset_policy_binding_sha256": subset_policy.binding_sha256,
    }
    digest = _sha256_record(_SELECTION_FRAMING, selection_payload)
    selected_index = int(digest, 16) % layout.block_count
    selected = tuple(
        ActionForecastBlockRequest(
            request=request,
            layout=arm_layout,
            block=arm_layout.blocks[selected_index],
            block_call_id=action_forecast_block_call_id(
                request,
                arm_layout,
                arm_layout.blocks[selected_index],
            ),
        )
        for request, arm_layout in zip(requests, layouts, strict=True)
    )
    plans = tuple(
        plan_action_forecast_v4_block_request(value) for value in selected
    )
    if len({value.call_id for value in plans}) != 3:
        raise ForecastWirePilotError("pilot physical call identities repeat")
    if any(
        value.output_tool_name != ACTION_FORECAST_BLOCK_TOOL_NAME
        or value.max_output_tokens != MAX_OUTPUT_TOKENS
        for value in plans
    ):
        raise ForecastWirePilotError("pilot block output authorization changed")
    # Apply the same provider-boundary firewall as the v2 full-frame runner:
    # neither prompts/schemas nor physical call IDs may expose M/P/N identity.
    # The helper also checks that the two grounded arms have identical
    # operation/tool/schema contracts.
    try:
        v2_launcher._validate_provider_boundary_blinding(plans)
    except BaseException as error:
        raise ForecastWirePilotError(
            "pilot provider boundary leaks control-plane identity"
        ) from error
    _validate_planned_wire_contracts(plans)
    block = layout.blocks[selected_index]
    eligible_option_ids = frozenset(preparation.eligible_g2_option_ids)
    eligible_global_rows = tuple(
        global_index
        for global_index in range(block.global_row_start, block.global_row_stop)
        if preparation.contract.options[global_index].option_id
        in eligible_option_ids
    )
    if len(eligible_global_rows) < health_policy.minimum_rows:
        raise ForecastWirePilotError(
            "selected block has too few eligible G2 rows for subset health"
        )
    selection_record = {
        "schema_version": 1,
        "selection_domain": _SELECTION_FRAMING.decode("ascii").rstrip("\x00"),
        "selection_payload": selection_payload,
        "selection_digest": digest,
        "selected_block_index": selected_index,
        "selected_block": block.to_record(),
        "selected_option_ids": [
            value.option_id
            for value in preparation.contract.options[
                block.global_row_start : block.global_row_stop
            ]
        ],
        "eligible_g2_global_row_indices": list(eligible_global_rows),
        "eligible_g2_option_ids": [
            preparation.contract.options[value].option_id
            for value in eligible_global_rows
        ],
    }
    return PilotBundle(
        frozen_v2_run=frozen_root,
        oracle_dir=oracle_dir.expanduser().resolve(strict=True),
        v2_finalization=finalization,
        preparation=preparation,
        reflection=reflection,
        arms=arms,
        arm_requests=requests,
        layout=layout,
        selected_block_index=selected_index,
        eligible_g2_global_row_indices=eligible_global_rows,
        selected_block_requests=selected,
        planned_calls=plans,
        selection_record=selection_record,
        arm_request_sha256s=tuple(
            value.request_sha256 for value in requests
        ),  # type: ignore[arg-type]
        selected_block_records=tuple(
            value.to_record() for value in selected
        ),  # type: ignore[arg-type]
        planned_call_payloads=tuple(
            _request_plan_payload(value) for value in plans
        ),  # type: ignore[arg-type]
        layout_record=layout.to_record(),
    )


def _arm_identity_record(bundle: PilotBundle) -> dict[str, object]:
    return {
        "schema_version": 1,
        "historical_reflection": {
            "new_provider_call": False,
            "logical_call_ids": [value.value for value in bundle.reflection.call_ids],
            "draft_content_sha256s": [
                value.draft.content_sha256 for value in bundle.reflection.shards
            ],
        },
        "finite_contract_identity_sha256": EXPECTED_CONTRACT_SHA256,
        "source_registry_sha256": EXPECTED_SOURCE_REGISTRY_SHA256,
        "arms": {
            "m": {
                "request_sha256": bundle.arm_request_sha256s[0],
                "card_snapshot_sha256": EXPECTED_MEMORY_CARD_SNAPSHOT_SHA256,
                "view_receipt_sha256": EXPECTED_MEMORY_VIEW_RECEIPT_SHA256,
            },
            "p": {
                "request_sha256": bundle.arm_request_sha256s[1],
                "card_snapshot_sha256": EXPECTED_PLACEBO_CARD_SNAPSHOT_SHA256,
                "view_receipt_sha256": EXPECTED_PLACEBO_VIEW_RECEIPT_SHA256,
            },
            "n": {
                "request_sha256": bundle.arm_request_sha256s[2],
                "card_count": 0,
                "evidence_mode": "catalog_only",
            },
        },
    }


def _planned_wave_record(bundle: PilotBundle) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "durably_precommitted_before_live_credential_read",
        "calls": [
            {
                "arm": arm,
                "forecast_request_sha256": request_sha256,
                "block_request": dict(block_record),
                "exact_provider_request": dict(plan_payload),
            }
            for arm, request_sha256, block_record, plan_payload in zip(
                ("m", "p", "n"),
                bundle.arm_request_sha256s,
                bundle.selected_block_records,
                bundle.planned_call_payloads,
                strict=True,
            )
        ],
    }


def _protocol_record(
    bundle: PilotBundle,
    *,
    target_live_run_dir: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "airfoil_v7_forecast_wire_v3_pilot",
        "scientific_scope": "representation_transport_qualifier_not_efficacy",
        "route": v2_launcher.route_binding(),
        "queue": build_config().to_manifest_record(),
        "forecast_policy": {
            "policy_id": PydanticAIActionForecastV4BlockPolicy.policy_id,
            "policy_version": ACTION_FORECAST_V4_POLICY_VERSION,
            "policy_definition_sha256": (
                ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
            ),
            "provider_wire_version": 4,
            "wire": "v4_ordinal_enum_matrices_with_logarithmic_tails",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "partition_policy": dict(bundle.layout_record["partition_policy"]),
        "layout": dict(bundle.layout_record),
        "selection": dict(bundle.selection_record),
        "health_policy": (
            lenient_action_forecast_health_v2_policy().to_record()
        ),
        "eligible_g2_subset_health": {
            "subset_policy": eligible_subset_policy().to_record(),
            "included_global_row_indices": list(
                bundle.eligible_g2_global_row_indices
            ),
            "required_receipts_per_wave": 3,
        },
        "execution": {
            "authorized_target_live_run_dir": str(target_live_run_dir),
            "logical_provider_calls": 3,
            "concurrent_calls": 3,
            "physical_attempts_per_logical_call": [1, MAX_ATTEMPTS],
            "retry_mode": "transport_only",
            "schema_repair": False,
            "logical_rerun": False,
            "settle_all_before_adjudication": True,
        },
        "resource_accounting": {
            "historical_reflection_reused": 1,
            "historical_reflection_counted_as_new_call": False,
            "frozen_g1_terminal_records_rehydrated_this_reconstruction": 8,
            "sealed_oracle_structural_verification_performed": True,
            "new_candidate_evaluations": 0,
            "allocation_calls": 0,
            "g2_openings": 0,
            "selected_action_evaluator_calls": 0,
            "new_cfd_calls": 0,
        },
        "terminal_statuses": [
            "incomplete",
            "typed_but_semantically_degenerate",
            "wire_qualified",
        ],
    }


def _exclusive_directory(path: Path) -> Path:
    target = path.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    return target


def execute_prepare(
    *,
    run_dir: Path,
    target_live_run_dir: Path,
    frozen_v2_run: Path = DEFAULT_FROZEN_V2_RUN,
    oracle_dir: Path = airfoil.DEFAULT_SEALED_ORACLE_DIR,
) -> dict[str, object]:
    """Create and seal one credential-free provider-request preparation."""

    target = target_live_run_dir.expanduser().resolve(strict=False)
    preparation_target = run_dir.expanduser().resolve(strict=False)
    if target == preparation_target:
        raise ForecastWirePilotError("prepared and live directories must differ")
    if target.exists():
        raise FileExistsError(f"authorized live directory already exists: {target}")
    root = _exclusive_directory(preparation_target)
    try:
        write_json_atomic(
            root / "preparation_started.json",
            {
                "schema_version": 1,
                "credential_read_attempted": False,
                "provider_client_constructed": False,
                "provider_call_attempted": False,
            },
        )
        bundle = build_pilot_bundle(
            frozen_v2_run=frozen_v2_run,
            oracle_dir=oracle_dir,
        )
        source = current_source_identity()
        protocol = _protocol_record(bundle, target_live_run_dir=target)
        arms = _arm_identity_record(bundle)
        wave = _planned_wave_record(bundle)
        write_json_atomic(root / "protocol.json", protocol)
        write_json_atomic(root / "reconstructed_arm_identities.json", arms)
        write_json_atomic(root / "planned_block_wave.json", wave)
        record: dict[str, object] = {
            "schema_version": 1,
            "status": "prepared",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(root),
            "authorized_target_live_run_dir": str(target),
            "frozen_v2_run": str(bundle.frozen_v2_run),
            "oracle_dir": str(bundle.oracle_dir),
            "closed_source_identity": source,
            "runtime_identity": v2_launcher.runtime_identity(),
            "protocol_file": file_identity(root / "protocol.json", relative_to=root),
            "arm_identities_file": file_identity(
                root / "reconstructed_arm_identities.json", relative_to=root
            ),
            "planned_wave_file": file_identity(
                root / "planned_block_wave.json", relative_to=root
            ),
            "credential_read_attempted": False,
            "provider_client_constructed": False,
            "provider_call_attempted": False,
            "frozen_g1_terminal_records_rehydrated": 8,
            "new_candidate_evaluations": 0,
        }
        record["preparation_commitment_sha256"] = _sha256_record(
            _PREPARED_FRAMING, record
        )
        write_json_atomic(root / "prepared.json", record)
        finalization = finalize_run_directory(root, status="prepared")
        return {"run_dir": str(root), "prepared": record, "finalization": finalization}
    except BaseException as error:
        if not (root / "result.json").exists():
            write_json_atomic(
                root / "result.json",
                {
                    "schema_version": 1,
                    "status": "incomplete",
                    "failure_type": type(error).__name__,
                    "credential_read_attempted": False,
                    "provider_call_attempted": False,
                    "new_candidate_evaluations": 0,
                    "new_cfd_calls": 0,
                },
            )
        if not (root / "finalized.json").exists():
            finalize_run_directory(root, status="incomplete")
        raise


@dataclass(frozen=True, slots=True)
class VerifiedPreparation:
    run_dir: Path
    record: Mapping[str, object]
    finalization: Mapping[str, object]
    bundle: PilotBundle
    wave: Mapping[str, object]


def verify_prepared(run_dir: Path) -> VerifiedPreparation:
    root = run_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    if finalization.get("status") != "prepared":
        raise ForecastWirePilotError("pilot preparation is not finalized as prepared")
    record = _load_object(root / "prepared.json")
    unsigned = dict(record)
    commitment = unsigned.pop("preparation_commitment_sha256", None)
    if commitment != _sha256_record(_PREPARED_FRAMING, unsigned):
        raise ForecastWirePilotError("pilot preparation commitment changed")
    source = current_source_identity()
    if record.get("closed_source_identity") != source:
        raise ForecastWirePilotError("closed source changed after pilot preparation")
    runtime = v2_launcher.runtime_identity()
    if record.get("runtime_identity") != runtime:
        raise ForecastWirePilotError("runtime changed after pilot preparation")
    target_value = record.get("authorized_target_live_run_dir")
    if type(target_value) is not str:
        raise ForecastWirePilotError("authorized live directory is malformed")
    target = Path(target_value)
    if (
        not target.is_absolute()
        or str(target.expanduser().resolve(strict=False)) != target_value
        or target == root
    ):
        raise ForecastWirePilotError("authorized live directory is not canonical")
    frozen_path = record.get("frozen_v2_run")
    oracle_path = record.get("oracle_dir")
    if type(frozen_path) is not str or type(oracle_path) is not str:
        raise ForecastWirePilotError("pilot preparation input paths are malformed")
    bundle = build_pilot_bundle(
        frozen_v2_run=Path(frozen_path),
        oracle_dir=Path(oracle_path),
    )
    wave = _load_object(root / "planned_block_wave.json")
    if wave != _planned_wave_record(bundle):
        raise ForecastWirePilotError("reconstructed live block wave changed")
    if _load_object(root / "protocol.json") != _protocol_record(
        bundle,
        target_live_run_dir=target,
    ):
        raise ForecastWirePilotError("reconstructed live protocol changed")
    if _load_object(root / "reconstructed_arm_identities.json") != _arm_identity_record(bundle):
        raise ForecastWirePilotError("reconstructed live arm identities changed")
    return VerifiedPreparation(root, record, finalization, bundle, wave)


@dataclass(slots=True)
class LiveClaim:
    run_dir: Path
    prepared: VerifiedPreparation
    claim_record: Mapping[str, object]
    active: bool = True

    def close(self) -> None:
        self.active = False


def claim_live(*, prepared_dir: Path, run_dir: Path) -> LiveClaim:
    """Verify every scientific input and claim output before credential read."""

    prepared = verify_prepared(prepared_dir)
    requested_target = run_dir.expanduser().resolve(strict=False)
    if prepared.record.get("authorized_target_live_run_dir") != str(
        requested_target
    ):
        raise ForecastWirePilotError(
            "live directory differs from the preparation's one-shot target"
        )
    root = _exclusive_directory(requested_target)
    record = {
        "schema_version": 1,
        "status": "claimed_before_credential_read",
        "prepared_dir": str(prepared.run_dir),
        "preparation_commitment_sha256": prepared.record[
            "preparation_commitment_sha256"
        ],
        "preparation_finalization_sha256": prepared.finalization[
            "finalization_sha256"
        ],
        "closed_source_identity": prepared.record["closed_source_identity"],
        "runtime_identity": prepared.record["runtime_identity"],
        "credential_read_attempted": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "frozen_g1_terminal_records_rehydrated": 8,
        "new_candidate_evaluations": 0,
    }
    write_json_atomic(root / "precredential_claim.json", record)
    return LiveClaim(root, prepared, record)


class Runner(Protocol):
    async def __aenter__(self) -> "Runner": ...
    async def __aexit__(self, *args: object) -> None: ...
    async def __call__(self, request: StructuredGenerationRequest[Any]) -> object: ...
    async def snapshot(self) -> object: ...


BlockHealthAssessor = Callable[..., object]
BlockSubsetHealthAssessor = Callable[..., object]


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    runner_factory: Callable[..., Runner] = create_progress_aware_openrouter_runner
    block_health_assessor: BlockHealthAssessor | None = field(
        default=_GENERIC_BLOCK_HEALTH
    )
    block_subset_health_assessor: BlockSubsetHealthAssessor | None = field(
        default=_GENERIC_SUBSET_HEALTH
    )

    def __post_init__(self) -> None:
        if not callable(self.runner_factory):
            raise TypeError("runner_factory must be callable")
        if self.block_health_assessor is not None and not callable(
            self.block_health_assessor
        ):
            raise TypeError("block_health_assessor must be callable or None")
        if self.block_subset_health_assessor is not None and not callable(
            self.block_subset_health_assessor
        ):
            raise TypeError(
                "block_subset_health_assessor must be callable or None"
            )


class _PrecommittedRecordingRunner:
    """Enforce exact prepared calls and persist admitted typed code matrices."""

    def __init__(
        self,
        delegate: Runner,
        *,
        run_dir: Path,
        prepared_plans: tuple[StructuredGenerationRequest[Any], ...],
        submission_journal: DurableJsonlJournal,
    ) -> None:
        self._delegate = delegate
        self._run_dir = run_dir
        self._prepared = {
            value.call_id.value: _request_plan_payload(value)
            for value in prepared_plans
        }
        self._submitted: set[str] = set()
        self._lock = threading.Lock()
        self._submission_journal = submission_journal
        self.typed_wire_count = 0

    @property
    def submitted_call_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._submitted))

    async def __call__(self, request: StructuredGenerationRequest[Any]) -> object:
        payload = _request_plan_payload(request)
        call_id = request.call_id.value
        with self._lock:
            if call_id in self._submitted:
                raise ForecastWirePilotError("a pilot logical call was submitted twice")
            if self._prepared.get(call_id) != payload:
                raise ForecastWirePilotError(
                    "dispatched pilot call differs from durable precommit"
                )
            self._submitted.add(call_id)
            # This is actual delegate-submission evidence, distinct from the
            # three provider requests frozen during preparation/live setup.
            self._submission_journal.append(
                {
                    "schema_version": 1,
                    "call_id": call_id,
                    "prompt_sha256": payload["call_contract"]["prompt_sha256"],
                    "schema_sha256": payload["call_contract"]["schema_sha256"],
                    "submitted_to_queue_delegate": True,
                }
            )
        raw = await self._delegate(request)
        if type(raw) is AttemptedStructuredGenerationResponse:
            response = raw.response
        elif type(raw) is StructuredGenerationResponse:
            response = raw
        else:
            return raw
        if not isinstance(response.value, BaseModel):
            return raw
        wire_record = {
            "schema_version": 1,
            "call_id": call_id,
            "prompt_sha256": payload["call_contract"]["prompt_sha256"],
            "schema_sha256": payload["call_contract"]["schema_sha256"],
            "typed_output_type": type(response.value).__name__,
            "typed_code_matrices": response.value.model_dump(mode="json"),
        }
        write_json_atomic(self._run_dir / f"typed_wire_{call_id}.json", wire_record)
        with self._lock:
            self.typed_wire_count += 1
        return raw


def _telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
    value.__post_init__()
    return {
        "requested_model": value.requested_model,
        "resolved_model": value.resolved_model,
        "resolved_provider": value.resolved_provider,
        "provider_response_id": value.provider_response_id,
        "finish_reason": value.finish_reason,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "reasoning_tokens": value.reasoning_tokens,
        "cache_read_tokens": value.cache_read_tokens,
        "cache_write_tokens": value.cache_write_tokens,
        "cost_usd": None if value.cost_usd is None else str(value.cost_usd),
        "latency_ns": value.latency_ns,
        "attempt_count": value.attempt_count,
    }


def _validate_route(value: AgenticCallTelemetry | None) -> None:
    if type(value) is not AgenticCallTelemetry:
        raise ForecastWirePilotError("pilot block telemetry is missing")
    value.__post_init__()
    if (
        value.requested_model != MODEL
        or value.resolved_model not in ALLOWED_RESOLVED_MODELS
        or value.resolved_provider != RESOLVED_PROVIDER
        or value.provider_response_id is None
        or value.finish_reason is None
        or value.input_tokens <= 0
        or value.output_tokens <= 0
        or value.latency_ns <= 0
        or value.cost_usd is None
        or value.cost_usd <= 0
        or not 1 <= value.attempt_count <= MAX_ATTEMPTS
    ):
        raise ForecastWirePilotError("pilot block escaped the frozen StreamLake route")


def _health_record(value: object) -> dict[str, object]:
    to_record = getattr(value, "to_record", None)
    if not callable(to_record):
        raise ForecastWirePilotError("block-health assessor returned no receipt")
    record = to_record()
    if type(record) is not dict or type(record.get("passes")) is not bool:
        raise ForecastWirePilotError("block-health receipt is malformed")
    return record


def _physical_attempt_accounting(
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    physical_attempt_count = 0
    successful_physical_attempt_count = 0
    scheduled_retry_count = 0
    for outcome in outcomes:
        attempts = outcome.get("attempts")
        if type(attempts) is not list:
            raise ForecastWirePilotError("queue outcome attempt ledger is malformed")
        if not 1 <= len(attempts) <= MAX_ATTEMPTS:
            raise ForecastWirePilotError("logical call escaped the attempt bound")
        for attempt in attempts:
            if type(attempt) is not dict or type(attempt.get("will_retry")) is not bool:
                raise ForecastWirePilotError("queue physical attempt is malformed")
            physical_attempt_count += 1
            successful_physical_attempt_count += int(
                attempt.get("status") == "succeeded"
            )
            scheduled_retry_count += int(attempt["will_retry"] is True)
    return {
        "physical_attempt_count": physical_attempt_count,
        "successful_physical_attempt_count": successful_physical_attempt_count,
        "scheduled_retry_count": scheduled_retry_count,
        "max_physical_attempts_per_logical_call": MAX_ATTEMPTS,
        "retry_mode": "transport_only",
        "schema_repair_count": 0,
        "logical_rerun_count": 0,
    }


def _accepted_usage_accounting(
    accepted: Sequence[
        tuple[str, ActionForecastBlockRequest, ActionForecastBlockResult]
    ],
) -> dict[str, object]:
    telemetry = [value.telemetry for _arm, _request, value in accepted]
    if any(type(value) is not AgenticCallTelemetry for value in telemetry):
        raise ForecastWirePilotError("accepted response lacks exact telemetry")
    typed = [value for value in telemetry if type(value) is AgenticCallTelemetry]
    return {
        "accepted_response_count": len(typed),
        "input_tokens": sum(value.input_tokens for value in typed),
        "output_tokens": sum(value.output_tokens for value in typed),
        "reasoning_tokens": sum(value.reasoning_tokens for value in typed),
        "cache_read_tokens": sum(value.cache_read_tokens for value in typed),
        "cache_write_tokens": sum(value.cache_write_tokens for value in typed),
        "latency_ns_sum": sum(value.latency_ns for value in typed),
        "cost_usd": str(
            sum(
                (value.cost_usd for value in typed if value.cost_usd is not None),
                start=Decimal("0"),
            )
        ),
    }


async def _run_live_async(
    *,
    claim: LiveClaim,
    api_key: str,
    dependencies: LiveDependencies,
    progress: Any,
    submission_journal: DurableJsonlJournal,
    outcome_journal: DurableJsonlJournal,
    snapshot_journal: DurableJsonlJournal,
) -> dict[str, object]:
    bundle = claim.prepared.bundle
    outcomes: list[dict[str, object]] = []

    def outcome_sink(outcome: object) -> None:
        progress.flush()
        record = queued_runner.structured_generation_outcome_record(outcome)  # type: ignore[arg-type]
        outcome_journal.append(record)
        outcomes.append(record)

    runner = dependencies.runner_factory(
        api_key=api_key,
        config=build_config(),
        progress_sink=progress,
        outcome_sink=outcome_sink,
    )
    async with runner:
        write_json_atomic(
            claim.run_dir / "runner_constructed.json",
            {
                "schema_version": 1,
                "runner_constructed": True,
                "provider_call_attempted": False,
                "configuration": build_config().to_manifest_record(),
            },
        )
        snapshot_journal.append(
            v2_launcher._queue_snapshot_record(
                await runner.snapshot(), stage="before_three_block_wave"
            )
        )
        recording = _PrecommittedRecordingRunner(
            runner,
            run_dir=claim.run_dir,
            prepared_plans=bundle.planned_calls,
            submission_journal=submission_journal,
        )
        policy = PydanticAIActionForecastV4BlockPolicy(recording)
        raw_results = await asyncio.gather(
            *(
                policy.forecast_block(block_request)
                for block_request in bundle.selected_block_requests
            ),
            return_exceptions=True,
        )
        snapshot_journal.append(
            v2_launcher._queue_snapshot_record(
                await runner.snapshot(), stage="after_three_block_wave"
            )
        )

    failures: list[dict[str, object]] = []
    accepted: list[tuple[str, ActionForecastBlockRequest, ActionForecastBlockResult]] = []
    for arm, block_request, raw in zip(
        ("m", "p", "n"),
        bundle.selected_block_requests,
        raw_results,
        strict=True,
    ):
        if isinstance(raw, BaseException):
            failures.append(
                {
                    "arm": arm,
                    "call_id": block_request.block_call_id.value,
                    "failure_type": type(raw).__name__,
                }
            )
            continue
        if type(raw) is not ActionForecastBlockResult:
            failures.append(
                {
                    "arm": arm,
                    "call_id": block_request.block_call_id.value,
                    "failure_type": "NonActionForecastBlockResult",
                }
            )
            continue
        try:
            raw.__post_init__()
            _validate_route(raw.telemetry)
        except BaseException as error:
            failures.append(
                {
                    "arm": arm,
                    "call_id": block_request.block_call_id.value,
                    "failure_type": type(error).__name__,
                }
            )
            continue
        accepted.append((arm, block_request, raw))
        assert raw.telemetry is not None
        write_json_atomic(
            claim.run_dir / f"resolved_block_{arm}.json",
            {
                "schema_version": 1,
                "arm": arm,
                "block_request": block_request.to_record(),
                "forecasts": raw.forecasts.to_record(),
                "telemetry": _telemetry_record(raw.telemetry),
            },
        )

    health_records: list[dict[str, object]] = []
    subset_health_records: list[dict[str, object]] = []
    if not failures and len(accepted) == 3:
        assessor = dependencies.block_health_assessor
        subset_assessor = dependencies.block_subset_health_assessor
        if assessor is None or subset_assessor is None:
            failures.append(
                {
                    "arm": None,
                    "call_id": None,
                    "failure_type": GenericBlockHealthUnavailable.__name__,
                }
            )
        else:
            for arm, block_request, result in accepted:
                try:
                    assessment = assessor(
                        block_request,
                        result.forecasts,
                        member_id=arm,
                        health_policy=(
                            lenient_action_forecast_health_v2_policy()
                        ),
                    )
                    health = _health_record(assessment)
                    health_records.append({"arm": arm, "assessment": health})
                    write_json_atomic(
                        claim.run_dir / f"block_health_{arm}.json", health
                    )
                    subset_assessment = subset_assessor(
                        block_request,
                        result.forecasts,
                        member_id=arm,
                        health_policy=(
                            lenient_action_forecast_health_v2_policy()
                        ),
                        subset_policy=eligible_subset_policy(),
                        included_global_row_indices=(
                            bundle.eligible_g2_global_row_indices
                        ),
                    )
                    subset_health = _health_record(subset_assessment)
                    subset_health_records.append(
                        {"arm": arm, "assessment": subset_health}
                    )
                    write_json_atomic(
                        claim.run_dir / f"eligible_g2_subset_health_{arm}.json",
                        subset_health,
                    )
                except BaseException as error:
                    failures.append(
                        {
                            "arm": arm,
                            "call_id": block_request.block_call_id.value,
                            "failure_type": type(error).__name__,
                        }
                    )

    expected_call_ids = tuple(
        value.block_call_id.value for value in bundle.selected_block_requests
    )
    successful_attempt_validation: dict[str, object] | None = None
    if len(outcomes) != 3:
        failures.append(
            {
                "arm": None,
                "call_id": None,
                "failure_type": "IncompleteQueueTerminalLedger",
            }
        )
    elif not failures:
        try:
            successful_attempt_validation = progress.validate_successful_attempts(
                outcomes,
                expected_call_ids=expected_call_ids,
                expected_prompt_sha256_by_call={
                    value.call_id.value: str(
                        _request_plan_payload(value)["call_contract"]["prompt_sha256"]
                    )
                    for value in bundle.planned_calls
                },
            )
        except BaseException as error:
            failures.append(
                {
                    "arm": None,
                    "call_id": None,
                    "failure_type": type(error).__name__,
                }
            )
    if recording.submitted_call_ids != tuple(sorted(expected_call_ids)):
        failures.append(
            {
                "arm": None,
                "call_id": None,
                "failure_type": "SubmittedCallSetChanged",
            }
        )

    submitted_count = len(recording.submitted_call_ids)
    qualification_counts = {
        "planned": len(bundle.planned_calls),
        "submitted": submitted_count,
        "terminal_outcomes": len(outcomes),
        "typed_wires": recording.typed_wire_count,
        "accepted_blocks": len(accepted),
        "health_assessments": len(health_records),
        "eligible_subset_health_assessments": len(subset_health_records),
    }
    if not failures and (
        any(value != 3 for value in qualification_counts.values())
        or successful_attempt_validation is None
    ):
        failures.append(
            {
                "arm": None,
                "call_id": None,
                "failure_type": "WireQualificationCardinalityMismatch",
            }
        )

    try:
        physical_attempts = _physical_attempt_accounting(outcomes)
        accepted_usage = _accepted_usage_accounting(accepted)
        if successful_attempt_validation is not None and (
            successful_attempt_validation.get("scheduled_retry_count")
            != physical_attempts["scheduled_retry_count"]
        ):
            raise ForecastWirePilotError("retry accounting receipts disagree")
    except BaseException as error:
        physical_attempts = {
            "physical_attempt_count": 0,
            "successful_physical_attempt_count": 0,
            "scheduled_retry_count": 0,
            "max_physical_attempts_per_logical_call": MAX_ATTEMPTS,
            "retry_mode": "transport_only",
            "schema_repair_count": 0,
            "logical_rerun_count": 0,
        }
        accepted_usage = {"accepted_response_count": 0, "cost_usd": "0"}
        failures.append(
            {
                "arm": None,
                "call_id": None,
                "failure_type": type(error).__name__,
            }
        )

    accounting = {
        "schema_version": 1,
        "authorized_logical_call_count": 3,
        "new_logical_provider_calls": submitted_count,
        "qualification_counts": qualification_counts,
        "physical_attempts": physical_attempts,
        "successful_attempt_validation": successful_attempt_validation,
        "accepted_usage": accepted_usage,
    }
    write_json_atomic(claim.run_dir / "call_accounting.json", accounting)

    if failures:
        status = "incomplete"
    elif any(
        not row["assessment"]["passes"]
        for row in (*health_records, *subset_health_records)
    ):
        status = "typed_but_semantically_degenerate"
    else:
        status = "wire_qualified"
    if failures:
        write_json_atomic(
            claim.run_dir / "call_failures.json",
            {"schema_version": 1, "failures": failures},
        )
    return {
        "schema_version": 1,
        "status": status,
        "scientific_scope": "representation_transport_qualifier_not_efficacy",
        "authorized_logical_provider_calls": 3,
        "new_logical_provider_calls": submitted_count,
        "historical_reflection_reused": 1,
        "historical_reflection_counted_as_new_call": False,
        "planned_logical_call_count": 3,
        "submitted_logical_call_count": len(recording.submitted_call_ids),
        "accepted_typed_block_count": len(accepted),
        "typed_wire_artifact_count": recording.typed_wire_count,
        "terminal_queue_outcome_count": len(outcomes),
        "health_assessment_count": len(health_records),
        "eligible_subset_health_assessment_count": len(subset_health_records),
        "health_pass_count": sum(
            1 for row in health_records if row["assessment"]["passes"] is True
        ),
        "eligible_subset_health_pass_count": sum(
            1
            for row in subset_health_records
            if row["assessment"]["passes"] is True
        ),
        "qualification_counts": qualification_counts,
        "physical_attempts": physical_attempts,
        "successful_attempt_validation": successful_attempt_validation,
        "accepted_usage": accepted_usage,
        "failures": failures,
        "frozen_g1_terminal_records_rehydrated": 8,
        "new_candidate_evaluations": 0,
        "allocation_calls": 0,
        "g2_openings": 0,
        "selected_action_evaluator_calls": 0,
        "new_cfd_calls": 0,
    }


def _manifest(claim: LiveClaim) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": "airfoil_v7_forecast_wire_v3_pilot_live",
        "run_dir": str(claim.run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(claim.prepared.run_dir),
        "preparation_commitment_sha256": claim.prepared.record[
            "preparation_commitment_sha256"
        ],
        "preparation_finalization_sha256": claim.prepared.finalization[
            "finalization_sha256"
        ],
        "closed_source_identity": claim.claim_record["closed_source_identity"],
        "runtime_identity": claim.claim_record["runtime_identity"],
        "route": v2_launcher.route_binding(),
        "queue": build_config().to_manifest_record(),
        "provider_dispatch_authorized": True,
        "logical_call_count": 3,
        "downstream_actions_authorized": [],
    }
    record["manifest_commitment_sha256"] = _sha256_record(
        _MANIFEST_FRAMING, record
    )
    return record


def execute_live(
    *,
    claim: LiveClaim,
    api_key: str,
    dependencies: LiveDependencies = LiveDependencies(),
) -> dict[str, object]:
    """Execute exactly the prepared three-block wave and finalize every status."""

    if type(claim) is not LiveClaim or not claim.active:
        raise ForecastWirePilotError("live execution requires an active claim")
    if type(api_key) is not str or not api_key:
        raise ForecastWirePilotError("live API key is unavailable")
    dependencies.__post_init__()
    root = claim.run_dir
    source = current_source_identity()
    runtime = v2_launcher.runtime_identity()
    if (
        source != claim.claim_record.get("closed_source_identity")
        or source != claim.prepared.record.get("closed_source_identity")
    ):
        raise ForecastWirePilotError("closed source changed before dispatch")
    if (
        runtime != claim.claim_record.get("runtime_identity")
        or runtime != claim.prepared.record.get("runtime_identity")
    ):
        raise ForecastWirePilotError("runtime changed before dispatch")
    if claim.prepared.record.get("authorized_target_live_run_dir") != str(root):
        raise ForecastWirePilotError("claimed directory escaped one-shot authorization")
    _validate_planned_wire_contracts(claim.prepared.bundle.planned_calls)
    write_json_atomic(root / "manifest.json", _manifest(claim))
    write_json_atomic(root / "planned_block_wave.json", dict(claim.prepared.wave))
    planned_journal = DurableJsonlJournal(root / "planned_calls.jsonl")
    submission_journal = DurableJsonlJournal(root / "submitted_calls.jsonl")
    progress_journal = BatchedDurableJsonlJournal(
        root / "stream_progress.jsonl",
        max_unfsynced_rows=PROGRESS_MAX_UNFSYNCED_ROWS,
    )
    progress = v2_launcher._ProgressRecorder(progress_journal)
    outcome_journal = DurableJsonlJournal(root / "queue_outcomes.jsonl")
    snapshot_journal = DurableJsonlJournal(root / "queue_snapshots.jsonl")
    for ordinal, (arm, plan) in enumerate(
        zip(("m", "p", "n"), claim.prepared.bundle.planned_calls, strict=True),
        start=1,
    ):
        payload = _request_plan_payload(plan)
        planned_journal.append(
            {
                "schema_version": 1,
                "ordinal": ordinal,
                "arm": arm,
                **payload["call_contract"],
            }
        )
    result: dict[str, object]
    pending: BaseException | None = None
    try:
        result = asyncio.run(
            _run_live_async(
                claim=claim,
                api_key=api_key,
                dependencies=dependencies,
                progress=progress,
                submission_journal=submission_journal,
                outcome_journal=outcome_journal,
                snapshot_journal=snapshot_journal,
            )
        )
    except BaseException as error:
        pending = error
        result = {
            "schema_version": 1,
            "status": "incomplete",
            "failure_type": type(error).__name__,
            "scientific_scope": "representation_transport_qualifier_not_efficacy",
            "authorized_logical_provider_calls": 3,
            "new_logical_provider_calls": len(
                read_jsonl(root / "submitted_calls.jsonl")
            ),
            "planned_logical_call_count": 3,
            "submitted_logical_call_count": len(
                read_jsonl(root / "submitted_calls.jsonl")
            ),
            "frozen_g1_terminal_records_rehydrated": 8,
            "new_candidate_evaluations": 0,
            "allocation_calls": 0,
            "g2_openings": 0,
            "selected_action_evaluator_calls": 0,
            "new_cfd_calls": 0,
        }
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            progress.flush()
        except BaseException as error:
            cleanup_errors.append(error)
        for resource in (
            planned_journal,
            submission_journal,
            progress_journal,
            outcome_journal,
            snapshot_journal,
        ):
            try:
                resource.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if current_source_identity() != claim.claim_record.get(
            "closed_source_identity"
        ):
            cleanup_errors.append(
                ForecastWirePilotError("closed source changed during pilot")
            )
        if cleanup_errors and pending is None:
            pending = cleanup_errors[0]
            result = {
                **result,
                "status": "incomplete",
                "failure_type": type(pending).__name__,
            }
    try:
        result["credential_read_attempted"] = True
        result["credentials_read"] = True
        result["provider_client_constructed"] = (
            root / "runner_constructed.json"
        ).is_file()
        submitted_count = len(read_jsonl(root / "submitted_calls.jsonl"))
        result["planned_logical_call_count"] = len(
            read_jsonl(root / "planned_calls.jsonl")
        )
        result["submitted_logical_call_count"] = submitted_count
        result["authorized_logical_provider_calls"] = 3
        result["new_logical_provider_calls"] = submitted_count
        result["provider_call_attempted"] = submitted_count > 0
        if "physical_attempts" not in result:
            try:
                result["physical_attempts"] = _physical_attempt_accounting(
                    read_jsonl(root / "queue_outcomes.jsonl")
                )
            except BaseException as error:
                result["physical_attempt_accounting_failure_type"] = type(
                    error
                ).__name__
        write_json_atomic(root / "result.json", result)
        finalization = finalize_run_directory(root, status=str(result["status"]))
    finally:
        claim.close()
    return {
        "run_dir": str(root),
        "result": result,
        "finalization": finalization,
        "pending_error_type": None if pending is None else type(pending).__name__,
    }


def _finalize_credential_abort(
    claim: LiveClaim,
    error: BaseException,
    *,
    credentials_read: bool,
) -> None:
    if not claim.active:
        return
    root = claim.run_dir
    if (root / "finalized.json").exists():
        claim.close()
        return
    planned_rows = (
        read_jsonl(root / "planned_calls.jsonl")
        if (root / "planned_calls.jsonl").is_file()
        else []
    )
    submitted_rows = (
        read_jsonl(root / "submitted_calls.jsonl")
        if (root / "submitted_calls.jsonl").is_file()
        else []
    )
    outcome_rows = (
        read_jsonl(root / "queue_outcomes.jsonl")
        if (root / "queue_outcomes.jsonl").is_file()
        else []
    )
    physical_attempts: dict[str, object] | None = None
    if outcome_rows:
        try:
            physical_attempts = _physical_attempt_accounting(outcome_rows)
        except BaseException:
            physical_attempts = None
    write_json_atomic(
        root / "result.json",
        {
            "schema_version": 1,
            "status": "incomplete",
            "failure_type": type(error).__name__,
            "credential_read_attempted": True,
            "credentials_read": credentials_read,
            "provider_client_constructed": (
                root / "runner_constructed.json"
            ).is_file(),
            "planned_logical_call_count": len(planned_rows),
            "submitted_logical_call_count": len(submitted_rows),
            "authorized_logical_provider_calls": 3,
            "new_logical_provider_calls": len(submitted_rows),
            "provider_call_attempted": bool(submitted_rows),
            "physical_attempts": physical_attempts,
            "frozen_g1_terminal_records_rehydrated": 8,
            "new_candidate_evaluations": 0,
            "allocation_calls": 0,
            "g2_openings": 0,
            "selected_action_evaluator_calls": 0,
            "new_cfd_calls": 0,
        },
    )
    finalize_run_directory(root, status="incomplete")
    claim.close()


def finalize_precredential_abort(claim: LiveClaim, error: BaseException) -> None:
    _finalize_credential_abort(claim, error, credentials_read=False)


def finalize_postcredential_abort(claim: LiveClaim, error: BaseException) -> None:
    _finalize_credential_abort(claim, error, credentials_read=True)


def _load_dotenv_api_key() -> str:
    """Load one credential at the live CLI boundary only.

    Routed through ``load_credentials`` so a name declared in
    ``AGENTEVOLVE_SCRUBBED`` stays unset. Reading the file directly, as this
    once did, defeated the scrub outright -- it preferred the file's value over
    the process environment, so removing the key changed nothing.
    """

    env_path = WORKSPACE_ROOT / ".env"
    if env_path.is_file():
        load_credentials(env_path, allow_credentials=("OPENROUTER_API_KEY",))
    value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise ForecastWirePilotError("OPENROUTER_API_KEY is unavailable")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "live"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--target-live-run-dir", type=Path)
    parser.add_argument("--frozen-v2-run", type=Path, default=DEFAULT_FROZEN_V2_RUN)
    parser.add_argument(
        "--oracle-dir", type=Path, default=airfoil.DEFAULT_SEALED_ORACLE_DIR
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "prepare":
        if arguments.target_live_run_dir is None:
            raise SystemExit("prepare requires --target-live-run-dir")
        execute_prepare(
            run_dir=arguments.run_dir,
            target_live_run_dir=arguments.target_live_run_dir,
            frozen_v2_run=arguments.frozen_v2_run,
            oracle_dir=arguments.oracle_dir,
        )
        return 0
    if arguments.prepared_dir is None:
        raise SystemExit("live requires --prepared-dir")
    claim = claim_live(
        prepared_dir=arguments.prepared_dir,
        run_dir=arguments.run_dir,
    )
    try:
        api_key = _load_dotenv_api_key()
    except BaseException as error:
        finalize_precredential_abort(claim, error)
        raise
    try:
        result = execute_live(claim=claim, api_key=api_key)
    except BaseException as error:
        finalize_postcredential_abort(claim, error)
        raise
    return 0 if result["result"]["status"] == "wire_qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())

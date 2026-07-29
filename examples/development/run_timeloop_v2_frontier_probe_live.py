#!/usr/bin/env python3
"""Freeze and execute the prospective Timeloop full-support G6 campaign.

``prepare`` executes the complete six-generation campaign with local typed
doubles, freezes the exact source closure and 3-D efficacy endpoint, and emits
the only preregistration accepted by ``live``.  It never reads a credential or
runs a candidate in Docker.

``live`` first replays the provider-free construction gate, validates the
preregistration and pinned Docker image, and only then reads the OpenRouter
credential.  The paid path uses one shared concurrency-limited queue for six
typed selector calls and one delayed exact-action reflection call.  Every
provider attempt, typed response, campaign event, engine event, reflection,
selection decision, and physical Timeloop evaluation is durably journaled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development.launch_record import (  # noqa: E402
    install_launch_recorder,
    instrument_startup_window,
    uninstall_launch_recorder,
    record_campaign_launch,
)

# Observe the launch environment before any module body below reads it.  This
# runner resolves most of its configuration at import time, so the observer
# has to be installed here to see those reads.  Instrumentation only, and
# only for the provider-free ``prepare`` mode; ``live`` is untouched.
install_launch_recorder()
# `live` gets a reversible environment-only window instead, closed again
# before any timed work; `prepare` keeps its lifetime instrumentation.
instrument_startup_window()

from agent_evolve.settings import load_credentials  # noqa: E402

from agent_evolve.agentic import (  # noqa: E402
    DeterministicIdFactory,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignExecutionEvent,
    CampaignJournalAck,
)
from agent_evolve.application.evaluation_accounting import (  # noqa: E402
    CampaignEvaluationAccounting,
)
from agent_evolve.application.action_forecast_partitioning import (  # noqa: E402
    ConcurrentActionForecastWave,
)
from agent_evolve.application.outcome_conditioned_portfolio_selection import (  # noqa: E402
    OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256,
    OutcomeConditionedPortfolioSelectionPolicy,
    outcome_conditioned_selected_predictions,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    CAMPAIGN_ARCHIVE_CONTEXT_KEY,
    CampaignIdentifiableReflectionInput,
)
from agent_evolve.application.portfolio_recombination import (  # noqa: E402
    bind_portfolio_recombination_source_utilities,
)
from agent_evolve.domain.ids import validate_id_namespace  # noqa: E402
from agent_evolve.domain.llm_task_queue import PartitionedRetryBudget  # noqa: E402
from agent_evolve.domain.typed_json import (  # noqa: E402
    FrozenJsonObject,
    thaw_json,
)
from agent_evolve.infrastructure.artifacts.filesystem import (  # noqa: E402
    FileSystemArtifactStore,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    AttemptedStructuredGenerationResponse,
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.action_forecast import (  # noqa: E402
    ACTION_FORECAST_POLICY_DEFINITION_SHA256,
    PydanticAIActionForecastBlockPolicy,
)
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (  # noqa: E402
    OpenRouterModelExecutionProfile,
    openrouter_model_execution_profile,
)
from agent_evolve.integrations.pydantic_ai.campaign_acquisition import (  # noqa: E402
    CampaignAcquisitionMode,
    build_campaign_acquisition_allocator,
    build_campaign_proposal_support_policy,
    campaign_operator_assay_minimum_from_environment,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    ProgressAwareOpenRouterConfig,
    ProgressAwareRetryMode,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    StructuredEvidencePublicationPolicy,
    structured_generation_outcome_record,
)
from agent_evolve.application.identifiable_reflection_evidence import (  # noqa: E402
    cluster_identifiable_mutation_reflection_hypotheses,
)
from agent_evolve.policies.reward.affine_hypervolume_3d import (  # noqa: E402
    AffineHypervolume3DSpec,
    AffineHypervolumeArchiveUtility3D,
    AffineHypervolumeSnapshot3D,
    audit_affine_reference_envelope_3d,
)
from agent_evolve.policies.selection.structural_posterior_slate import (  # noqa: E402
    build_terminal_tapered_family_exposure_phases,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationResponse,
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
)
from agent_evolve.ports.action_forecast import (  # noqa: E402
    ActionForecastPartitionPolicyBinding,
)
from examples.benchmarks.timeloop_codesign.v2.campaign_reflection import (  # noqa: E402
    build_timeloop_v2_identifiable_learning_envelope,
    build_timeloop_v2_identifiable_reflection_request,
)
from examples.benchmarks.timeloop_codesign.v2.affine_utility import (  # noqa: E402
    timeloop_v2_affine_hypervolume_spec,
)
from examples.benchmarks.timeloop_codesign.v2.candidate import (  # noqa: E402
    candidate_sha256,
)
from examples.benchmarks.timeloop_codesign.v2.detailed_evaluation import (  # noqa: E402
    compose_timeloop_v2_detailed_benchmark,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (  # noqa: E402
    PINNED_IMAGE_ID,
    PINNED_IMAGE_REF,
    TimeloopV2CandidateInfeasibleError,
    TimeloopV2DockerEvaluator,
    TimeloopV2Evaluation,
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (  # noqa: E402
    frozen_network_panel,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    write_json_atomic,
)
from examples.development.run_timeloop_v2_provider_free_campaign import (  # noqa: E402
    ARCHIVE_CONTEXT_MODE,
    COMMON_CANDIDATE_POOL_SIZE,
    FEASIBILITY_WITNESS_MODE,
    GENERATION_COUNT,
    PLANNED_CANDIDATE_OCCURRENCES,
    PLANNED_LOGICAL_CALLS,
    PORTFOLIO_WIDTH,
    VARIATION_TOPOLOGY,
    run_provider_free_timeloop_campaign,
    run_timeloop_campaign,
)


def _common_pool_enabled() -> bool:
    return FEASIBILITY_WITNESS_MODE.value == "task_keyed_common_pool"


def _typed_candidate_infeasible_events(
    engine_traces: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Return terminal candidate failures from the engine-owned ledger.

    The Docker observer is deliberately downstream of the exact static
    mapspace precheck.  Consequently its call count cannot account for a
    candidate that is proven infeasible before native invocation.  The engine
    ledger is the authoritative workload-neutral occurrence ledger and keeps
    that outcome distinct from orchestration failure.
    """

    if type(engine_traces) is not list or any(
        type(value) is not dict for value in engine_traces
    ):
        raise TypeError("engine_traces must be a list of exact dictionaries")
    events: list[dict[str, object]] = []
    for value in engine_traces:
        if value.get("event_type") != "candidate_evaluated":
            continue
        detailed = value.get("detailed_evaluation")
        failure = detailed.get("failure") if type(detailed) is dict else None
        if (
            value.get("valid") is False
            and type(failure) is dict
            and failure.get("category") == "candidate"
            and failure.get("code") == "evaluator_declared_infeasible"
        ):
            events.append(value)
    return tuple(events)


def _pre_simulator_infeasible_count(
    events: tuple[dict[str, object], ...],
) -> int:
    """Count typed failures that intentionally bypassed native simulation."""

    if type(events) is not tuple or any(type(value) is not dict for value in events):
        raise TypeError("events must be an exact tuple of dictionaries")
    count = 0
    for value in events:
        detailed = value.get("detailed_evaluation")
        checks = detailed.get("checks") if type(detailed) is dict else None
        if type(checks) is not list:
            raise ValueError("candidate-infeasible evidence lacks typed checks")
        native = tuple(
            check
            for check in checks
            if type(check) is dict
            and check.get("name") == "native_simulator_invocation"
        )
        if len(native) != 1:
            raise ValueError(
                "candidate-infeasible evidence lacks one native-invocation check"
            )
        observed = native[0].get("observed_value")
        if (
            native[0].get("status") == "not_applicable"
            and type(observed) is dict
            and observed.get("native_simulator_invoked") is False
        ):
            count += 1
    return count


def _portfolio_candidate_infeasible_count(run: object) -> int:
    """Count unscorable selector actions from authenticated stage receipts."""

    execution = getattr(run, "execution", None)
    receipts = getattr(execution, "stage_receipts", None)
    if type(receipts) is not tuple:
        raise TypeError("campaign run lacks exact stage receipts")
    total = 0
    for receipt in receipts:
        if receipt.kind.value != "portfolio":
            continue
        result = thaw_json(receipt.result)
        if type(result) is not dict:
            raise TypeError("portfolio stage result must be an object")
        value = result.get("candidate_infeasible_count")
        if type(value) is not int or value < 0:
            raise ValueError("portfolio result lacks candidate-infeasible accounting")
        total += value
    return total


ACQUISITION_MODE = CampaignAcquisitionMode(
    os.environ.get(
        "AGENT_EVOLVE_ACQUISITION_MODE",
        "model_top_k" if _common_pool_enabled() else "full_support",
    )
)
OPERATOR_ASSAY_MINIMUM = campaign_operator_assay_minimum_from_environment(os.environ)
if (
    ACQUISITION_MODE is CampaignAcquisitionMode.OPERATOR_STRATIFIED
    and OPERATOR_ASSAY_MINIMUM > VARIATION_TOPOLOGY.required_composite_proposals
):
    raise ValueError("operator assay minimum exceeds hierarchical proposal minimum")


def _calibrated_allocator():
    return build_campaign_acquisition_allocator(
        ACQUISITION_MODE,
        common_pool_enabled=_common_pool_enabled(),
        operator_assay_minimum=OPERATOR_ASSAY_MINIMUM,
        family_exposure_phases=(
            build_terminal_tapered_family_exposure_phases(
                family="composite_r2",
                terminal_wave_index=GENERATION_COUNT - 1,
            )
            if ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
            else None
        ),
    )


def _proposal_support_policy():
    return build_campaign_proposal_support_policy(
        ACQUISITION_MODE,
        common_pool_enabled=_common_pool_enabled(),
    )


def _acquisition_execution_label() -> str:
    return {
        CampaignAcquisitionMode.FULL_SUPPORT: "full_support_k8_to_k8",
        CampaignAcquisitionMode.MODEL_TOP_K: "common_pool_model_top_k8_to_k4",
        CampaignAcquisitionMode.CALIBRATED_FRONTIER: (
            "common_pool_structural_posterior_k8_to_k4"
        ),
        CampaignAcquisitionMode.HIERARCHICAL_SUPPORT: (
            "hierarchical_support_structural_k8_to_k4"
        ),
        CampaignAcquisitionMode.OPERATOR_STRATIFIED: (
            "operator_stratified_hierarchical_k8_to_k4"
        ),
        CampaignAcquisitionMode.HORIZON_BOUNDED: (
            "horizon_bounded_hierarchical_k8_to_k4"
        ),
    }[ACQUISITION_MODE]


def _allocation_policy_identity() -> tuple[str, int, str]:
    allocator = _calibrated_allocator()
    return (
        allocator.policy_id,
        allocator.policy_version,
        allocator.definition_sha256,
    )


# Artifact location is an execution concern, not part of the acquisition arm.
# Keep it invariant across witness modes so the workload-owned systematic
# execution contract can locate and authenticate prepare/live summaries without
# duplicating optimizer configuration in filesystem-routing logic.
ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/timeloop_codesign/full_support_g6"
)
GATE_A_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/timeloop_codesign/v2_real_gate_a_20260717/run_001"
    / "evaluator_calls"
)
QUALIFICATION_ROOTS = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/timeloop_codesign/register_disabled_target_repair_v2_20260719",
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/timeloop_codesign/max_architecture_corner_20260719",
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/timeloop_codesign/pessimistic_reference_corner_20260719",
)
HISTORICAL_REFERENCE_ENVELOPE_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/timeloop_codesign/thin_adapter_tradeoff_panel_20260716_01"
    / "timeloop-5f4840dbe5464d08a53622bda2d70710"
)
MODEL_PROFILE_NAME = os.environ.get("AGENT_EVOLVE_MODEL_PROFILE", "deepseek")
MODEL_EXECUTION_PROFILE: OpenRouterModelExecutionProfile = (
    openrouter_model_execution_profile(MODEL_PROFILE_NAME)
)
MODEL = MODEL_EXECUTION_PROFILE.requested_model
PROVIDER_ONLY = MODEL_EXECUTION_PROFILE.provider_only
RESOLVED_PROVIDER = MODEL_EXECUTION_PROFILE.accepted_resolved_providers[0]
MAX_OUTPUT_TOKENS = MODEL_EXECUTION_PROFILE.max_output_tokens
TEMPERATURE = MODEL_EXECUTION_PROFILE.temperature
PORTFOLIO_SELECTOR_MODE = os.environ.get(
    "AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE",
    "calibrated",
)
if PORTFOLIO_SELECTOR_MODE not in {"calibrated", "outcome_conditioned"}:
    raise ValueError(
        "AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE must be calibrated or "
        "outcome_conditioned"
    )
_ACTION_FORECAST_BLOCK_ROWS_RAW = os.environ.get(
    "AGENT_EVOLVE_ACTION_FORECAST_BLOCK_ROWS",
    "32",
)
if (
    not _ACTION_FORECAST_BLOCK_ROWS_RAW.isascii()
    or not _ACTION_FORECAST_BLOCK_ROWS_RAW.isdigit()
    or not 8 <= int(_ACTION_FORECAST_BLOCK_ROWS_RAW) <= 64
):
    raise ValueError("AGENT_EVOLVE_ACTION_FORECAST_BLOCK_ROWS must lie in [8,64]")
ACTION_FORECAST_BLOCK_ROWS = int(_ACTION_FORECAST_BLOCK_ROWS_RAW)
ACTION_FORECAST_PARTITION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:generic-campaign-action-forecast-partition:v1;"
    + f"rows={ACTION_FORECAST_BLOCK_ROWS};cells-per-row=objective-count".encode(
        "ascii"
    )
).hexdigest()
CPU_SET = "8"
EVALUATOR_TIMEOUT_S = 180.0
AGENT_CONCURRENCY = MODEL_EXECUTION_PROFILE.effective_max_connections(default=3)
AGENT_QUEUE_CAPACITY = 8
PARTITIONED_RETRY_BUDGET = (
    PartitionedRetryBudget(output_invalid_retries=2, transport_retries=2)
    if ACQUISITION_MODE
    in {
        CampaignAcquisitionMode.CALIBRATED_FRONTIER,
        CampaignAcquisitionMode.HIERARCHICAL_SUPPORT,
        CampaignAcquisitionMode.OPERATOR_STRATIFIED,
        CampaignAcquisitionMode.HORIZON_BOUNDED,
    }
    else None
)
MAX_ATTEMPTS = 5 if PARTITIONED_RETRY_BUDGET is not None else 3
MAX_CACHE_REUSE_OCCURRENCES = 6
FIRST_EVENT_TIMEOUT_NS = 300_000_000_000
IDLE_TIMEOUT_NS = 300_000_000_000
CLEANUP_TIMEOUT_NS = 5_000_000_000
CONNECT_TIMEOUT_SECONDS = 90.0
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000


class _OutcomeConditionedSelectorAdapter:
    """One-port Timeloop adapter for the generic outcome-aware selector.

    The campaign composer consumes this single object for selection, prompt
    audit, feedback decoding, and policy identity.  Workload integration does
    not receive Timeloop-specific search hints or effect equations.
    """

    policy_definition_sha256 = (
        OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256
    )
    prompt_definition_sha256 = ACTION_FORECAST_POLICY_DEFINITION_SHA256

    def __init__(self, delegate: OutcomeConditionedPortfolioSelectionPolicy) -> None:
        if type(delegate) is not OutcomeConditionedPortfolioSelectionPolicy:
            raise TypeError("delegate must be the exact outcome-conditioned policy")
        self._delegate = delegate

    async def select(self, request):
        return await self._delegate.select(request)

    def render(self, request):
        return self._delegate.render(request)

    def decode_selected_predictions(self, scope, wave, result):
        return outcome_conditioned_selected_predictions(
            scope=scope,
            wave=wave,
            result=result,
        )


def _outcome_conditioned_selector(*, runner: Any, benchmark: Any):
    semantics = benchmark.optimization_semantics
    if semantics is None:
        raise TypeError("Timeloop outcome-conditioned selector requires semantics")
    delegate = OutcomeConditionedPortfolioSelectionPolicy(
        forecaster=ConcurrentActionForecastWave(
            block_policy=PydanticAIActionForecastBlockPolicy(runner),
            max_concurrency=AGENT_CONCURRENCY,
        ),
        optimization_semantics=semantics,
        partition_policy=ActionForecastPartitionPolicyBinding(
            policy_id="generic_campaign_action_forecast_blocks",
            policy_version=1,
            policy_definition_sha256=(
                ACTION_FORECAST_PARTITION_DEFINITION_SHA256
            ),
            max_rows_per_block=ACTION_FORECAST_BLOCK_ROWS,
            max_metric_cells_per_block=(
                ACTION_FORECAST_BLOCK_ROWS * len(benchmark.objectives)
            ),
        ),
        action_semantics_factory=None,
        metric_projector=None,
        risk_aversion=0.5,
        diversity_weight=0.05,
        beam_width=256,
    )
    return _OutcomeConditionedSelectorAdapter(delegate)


_PROTOCOL_IDENTITIES = {
    CampaignAcquisitionMode.FULL_SUPPORT: (
        "timeloop_v2_exact_action_memory_g6_v10",
        b"agent-evolve:timeloop-v2-exact-action-memory-g6-protocol:v10;",
        b"agent-evolve:timeloop-v2-exact-action-memory-g6:v10\x00",
        "v10",
    ),
    CampaignAcquisitionMode.MODEL_TOP_K: (
        "timeloop_v2_exact_action_memory_g6_v10",
        b"agent-evolve:timeloop-v2-exact-action-memory-g6-protocol:v10;",
        b"agent-evolve:timeloop-v2-exact-action-memory-g6:v10\x00",
        "v10",
    ),
    CampaignAcquisitionMode.CALIBRATED_FRONTIER: (
        "timeloop_v2_calibrated_frontier_g6_v11",
        b"agent-evolve:timeloop-v2-calibrated-frontier-g6-protocol:v11;",
        b"agent-evolve:timeloop-v2-calibrated-frontier-g6:v11\x00",
        "v11",
    ),
    CampaignAcquisitionMode.HIERARCHICAL_SUPPORT: (
        "timeloop_v2_hierarchical_successor_g6_v12",
        b"agent-evolve:timeloop-v2-hierarchical-successor-g6-protocol:v12;",
        b"agent-evolve:timeloop-v2-hierarchical-successor-g6:v12\x00",
        "v12",
    ),
    CampaignAcquisitionMode.OPERATOR_STRATIFIED: (
        "timeloop_v2_operator_stratified_successor_g6_v13",
        b"agent-evolve:timeloop-v2-operator-stratified-g6-protocol:v13;",
        b"agent-evolve:timeloop-v2-operator-stratified-g6:v13\x00",
        "v13",
    ),
    CampaignAcquisitionMode.HORIZON_BOUNDED: (
        "timeloop_v2_horizon_bounded_successor_g6_v16",
        b"agent-evolve:timeloop-v2-horizon-bounded-g6-protocol:v16;",
        b"agent-evolve:timeloop-v2-horizon-bounded-g6:v16\x00",
        "v16",
    ),
}
(
    PROTOCOL_ID,
    _PROTOCOL_DEFINITION_PREFIX,
    _CAMPAIGN_IDENTITY_PREFIX,
    _PROTOCOL_VERSION,
) = _PROTOCOL_IDENTITIES[ACQUISITION_MODE]
PROTOCOL_DEFINITION_SHA256 = hashlib.sha256(
    _PROTOCOL_DEFINITION_PREFIX
    + b"g6;typed-selector;delayed-identifiable-exact-action-reflection;"
    b"prospective-complete-support-cohort;randomized-hard-memory-dose;"
    b"closed-semantic-schema-repair-guidance;"
    b"3d-affine-hv;"
    b"selected-outcome-calibration=true;contextual-prior-outcome-feedback=true;"
    b"reference-envelope-gate=true;prepare-live-id-namespace-parity=true;"
    b"unique-normalized-reflection-claims=true;canonical-summary-keys=true;"
    b"joint-contextual-memory-dose-feasibility=true"
).hexdigest()
TASK_SHA256 = hashlib.sha256(
    b"agent-evolve:timeloop-v2-resnet50-full-support-task:v1"
).hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _campaign_sha256(replicate_seed: int) -> str:
    return hashlib.sha256(
        _CAMPAIGN_IDENTITY_PREFIX
        + bytes.fromhex(MODEL_EXECUTION_PROFILE.profile_sha256)
        + replicate_seed.to_bytes(16, "big", signed=True)
    ).hexdigest()


def _id_namespace(replicate_seed: int) -> str:
    value = f"tlv2_mem_{MODEL_PROFILE_NAME}_{replicate_seed}_{_PROTOCOL_VERSION}"
    validate_id_namespace(value)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("Timeloop live record is not a frozen object")
    return result


def _telemetry_record(value: object) -> dict[str, object]:
    return {
        "requested_model": getattr(value, "requested_model"),
        "resolved_model": getattr(value, "resolved_model"),
        "resolved_provider": getattr(value, "resolved_provider"),
        "provider_response_id": getattr(value, "provider_response_id"),
        "finish_reason": getattr(value, "finish_reason"),
        "input_tokens": getattr(value, "input_tokens"),
        "output_tokens": getattr(value, "output_tokens"),
        "reasoning_tokens": getattr(value, "reasoning_tokens"),
        "cache_read_tokens": getattr(value, "cache_read_tokens"),
        "cache_write_tokens": getattr(value, "cache_write_tokens"),
        "cost_usd": (
            None
            if getattr(value, "cost_usd") is None
            else str(getattr(value, "cost_usd"))
        ),
        "latency_ns": getattr(value, "latency_ns"),
        "attempt_count": getattr(value, "attempt_count", 1),
    }


def _progress_record(value: StructuredStreamProgress) -> dict[str, object]:
    value.__post_init__()
    return {
        "call_id": value.call_id,
        "provider_attempt_id": value.provider_attempt_id,
        "sequence": value.sequence,
        "kind": value.kind.value,
        "channel": value.channel.value,
        "elapsed_ns": value.elapsed_ns,
        "event_content_utf8_bytes": value.event_content_utf8_bytes,
        "cumulative_content_utf8_bytes": value.cumulative_content_utf8_bytes,
        "rolling_content_sha256": value.rolling_content_sha256,
    }


def _provider_config(replicate_seed: int) -> ProgressAwareOpenRouterConfig:
    return ProgressAwareOpenRouterConfig(
        model_name=MODEL,
        provider_only=PROVIDER_ONLY,
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=FIRST_EVENT_TIMEOUT_NS,
            idle_timeout_ns=IDLE_TIMEOUT_NS,
            absolute_timeout_ns=None,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=CLEANUP_TIMEOUT_NS,
                transport_retire_timeout_ns=CLEANUP_TIMEOUT_NS,
            ),
        ),
        max_connections=AGENT_CONCURRENCY,
        max_pending=AGENT_QUEUE_CAPACITY,
        max_attempts=MAX_ATTEMPTS,
        retry_budget=PARTITIONED_RETRY_BUDGET,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        rate_limit_backoff_floor_ns=(
            MODEL_EXECUTION_PROFILE.rate_limit_backoff_floor_ns
        ),
        jitter_seed=replicate_seed,
        jitter_domain=f"{PROTOCOL_ID}:{MODEL_PROFILE_NAME}",
        app_title="AgentEvolve AAAI 2027 Timeloop full-support G6",
        reasoning_config=MODEL_EXECUTION_PROFILE.reasoning_config,
        structured_output_mode=(MODEL_EXECUTION_PROFILE.structured_output_mode),
        structured_output_strict=(MODEL_EXECUTION_PROFILE.structured_output_strict),
        json_schema_dialect=MODEL_EXECUTION_PROFILE.json_schema_dialect,
        provider_require_parameters=(
            MODEL_EXECUTION_PROFILE.provider_require_parameters
        ),
        supports_forced_tool_choice=(
            MODEL_EXECUTION_PROFILE.supports_forced_tool_choice
        ),
        retry_mode=(ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_BOUNDED_SCHEMA_REPAIR),
    )


def _model_profile_sha256() -> str:
    return MODEL_EXECUTION_PROFILE.profile_sha256


def _utility_spec() -> AffineHypervolume3DSpec:
    return timeloop_v2_affine_hypervolume_spec()


@dataclass(frozen=True, slots=True)
class _RecombinationUtilityBinder:
    utility: AffineHypervolumeArchiveUtility3D

    def bind(self, *, source_archive_utility, source_wave, source_result):
        snapshot = self.utility.require_snapshot(source_archive_utility)
        candidates = {
            value.candidate_id: value for value in source_result.scored_candidates
        }
        marginal = {
            candidate_id: snapshot.marginal_gain(candidate.objective_map)
            for candidate_id, candidate in candidates.items()
        }
        pair = {
            pair_ids: snapshot.joint_gain(
                tuple(candidates[value].objective_map for value in pair_ids)
            )
            for pair_ids in combinations(sorted(candidates), 2)
        }
        return bind_portfolio_recombination_source_utilities(
            snapshot=source_archive_utility,
            source_wave=source_wave,
            source_result=source_result,
            marginal_utilities=marginal,
            exact_pair_utilities=pair,
        )


def _source_paths() -> tuple[Path, ...]:
    roots = (
        AGENT_EVOLVE_ROOT / "src/agent_evolve",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/timeloop_codesign/v2",
    )
    paths = {
        path.resolve(strict=True)
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    paths.update(
        {
            (AGENT_EVOLVE_ROOT / "pyproject.toml").resolve(strict=True),
            (AGENT_EVOLVE_ROOT / "uv.lock").resolve(strict=True),
            Path(__file__).resolve(strict=True),
            (
                AGENT_EVOLVE_ROOT
                / "examples/development/run_timeloop_v2_provider_free_campaign.py"
            ).resolve(strict=True),
            (
                AGENT_EVOLVE_ROOT
                / "examples/development/run_timeloop_v2_uniform_control.py"
            ).resolve(strict=True),
            (
                AGENT_EVOLVE_ROOT
                / "examples/development/uniform_feasible_portfolio_control.py"
            ).resolve(strict=True),
            (
                AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py"
            ).resolve(strict=True),
            (
                AGENT_EVOLVE_ROOT / "examples/development/launch_record.py"
            ).resolve(strict=True),
        }
    )
    return tuple(
        sorted(paths, key=lambda item: item.relative_to(WORKSPACE_ROOT).as_posix())
    )


def _snapshot_sources(run_dir: Path, paths: tuple[Path, ...]) -> dict[str, object]:
    destination_root = run_dir / "source_snapshot"
    rows: list[dict[str, object]] = []
    aggregate = hashlib.sha256(b"agent-evolve:source-set:v1\x00")
    for path in paths:
        label = path.relative_to(WORKSPACE_ROOT).as_posix()
        content = path.read_bytes()
        destination = destination_root / label
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        label_bytes = label.encode("utf-8", errors="strict")
        aggregate.update(len(label_bytes).to_bytes(8, "big"))
        aggregate.update(label_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
        rows.append(
            {
                "path": label,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "snapshot_directory": "source_snapshot",
        "file_count": len(rows),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def _require_source_closure(expected: str) -> dict[str, object]:
    current = source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)
    if current["aggregate_sha256"] != expected:
        raise RuntimeError("sealed Timeloop campaign source changed")
    return current


def _read_live_api_key() -> str:
    load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
    load_credentials(AGENT_EVOLVE_ROOT / ".env", override=False, optional=True)
    # The credential window is closed. Everything after this line -- all of
    # the timed work -- runs in an unmodified process, and the launch record
    # says so with instrumented_phases: ["startup"].
    uninstall_launch_recorder()
    value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    return value


def _load_gate_a_baseline() -> dict[str, object]:
    paths = tuple(sorted(GATE_A_ROOT.glob("timeloop-v2-*/result.json")))
    if len(paths) != 18:
        raise RuntimeError(
            "authenticated Timeloop Gate-A baseline must contain 18 results"
        )
    points: list[dict[str, float]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        objectives = value.get("objectives")
        if type(objectives) is not dict:
            raise RuntimeError("Gate-A result omitted objectives")
        points.append(
            {key: float(objectives[key]) for key in _utility_spec().metric_ids}
        )
    snapshot = AffineHypervolumeSnapshot3D.create(
        spec=_utility_spec(), archive_points=tuple(points)
    )
    return {
        "role": "historical_development_diagnostic_not_matched_control",
        "result_count": len(points),
        "identity": source_identity(paths, relative_to=WORKSPACE_ROOT),
        "affine_hypervolume_hex": snapshot.base_hypervolume.hex(),
        "reference_contains_all": all(
            all(value < 1.0 for value in _utility_spec().normalize(point))
            for point in points
        ),
    }


def _load_real_qualification_evidence() -> dict[str, object]:
    records: list[dict[str, object]] = []
    evidence_paths: list[Path] = []
    envelope_points: list[dict[str, float]] = []
    for root in QUALIFICATION_ROOTS:
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            raise RuntimeError("Timeloop real qualification did not pass")
        output_dir = Path(str(summary["output_dir"])).resolve(strict=True)
        paths = tuple(
            sorted(
                (
                    summary_path.resolve(strict=True),
                    (output_dir / "result.json").resolve(strict=True),
                    (output_dir / "host_receipt.json").resolve(strict=True),
                ),
                key=lambda item: item.relative_to(WORKSPACE_ROOT).as_posix(),
            )
        )
        evidence_paths.extend(paths)
        objectives = summary.get("objectives")
        if type(objectives) is not dict:
            raise RuntimeError("Timeloop qualification omitted objectives")
        point = {key: float(objectives[key]) for key in _utility_spec().metric_ids}
        envelope_points.append(point)
        records.append(
            {
                "qualification_id": root.name,
                "candidate_sha256": summary["candidate_sha256"],
                "compiled_plan_sha256": summary["compiled_plan_sha256"],
                "objectives": point,
                "evaluator_elapsed_s_hex": float(summary["evaluator_elapsed_s"]).hex(),
                "content_identity": source_identity(paths, relative_to=WORKSPACE_ROOT),
            }
        )

    historical_paths = tuple(
        HISTORICAL_REFERENCE_ENVELOPE_ROOT / name
        for name in ("candidate.json", "host_receipt.json", "result.json")
    )
    historical_result = json.loads(historical_paths[-1].read_text(encoding="utf-8"))
    historical_objectives = historical_result.get("objectives")
    if type(historical_objectives) is not dict:
        raise RuntimeError("historical Timeloop envelope omitted objectives")
    historical_point = {
        key: float(historical_objectives[key]) for key in _utility_spec().metric_ids
    }
    evidence_paths.extend(historical_paths)
    envelope_points.append(historical_point)
    historical_record = {
        "role": "pre_experiment_historical_reference_envelope_only",
        "evaluator_id": historical_result.get("evaluator_id"),
        "candidate_sha256": historical_result.get("candidate_sha256"),
        "objectives": historical_point,
        "content_identity": source_identity(
            historical_paths, relative_to=WORKSPACE_ROOT
        ),
    }
    envelope_identity = source_identity(
        tuple(evidence_paths), relative_to=WORKSPACE_ROOT
    )
    envelope_audit = audit_affine_reference_envelope_3d(
        spec=_utility_spec(),
        points=tuple(envelope_points),
        evidence_identity_sha256=str(envelope_identity["aggregate_sha256"]),
    )
    if envelope_audit["strictly_contains_all"] is not True:
        raise RuntimeError(
            "fixed Timeloop reference does not contain its qualification envelope"
        )
    return {
        "schema_version": 1,
        "role": "prospective_real_evaluator_and_reference_qualification",
        "records": records,
        "historical_reference_envelope": historical_record,
        "reference_envelope_evidence_identity": envelope_identity,
        "reference_envelope_audit": envelope_audit,
    }


def _contextual_history_counts_by_cutoff(
    contextual_histories: list[dict[str, object]],
) -> dict[str, list[int]]:
    """Project dynamic cutoffs into canonical JSON object keys."""

    result: dict[str, list[int]] = {}
    for history in contextual_histories:
        cutoff = str(int(history["cutoff_wave_index_exclusive"]))
        actions = history["actions"]
        if type(actions) is not list:
            raise TypeError("campaign contextual history actions must be a list")
        result.setdefault(cutoff, []).append(len(actions))
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _g5_memory_path_audit(summary: dict[str, object]) -> dict[str, object]:
    """Distinguish a realized assay from its valid no-support recourse.

    Reflection-derived cards are admitted only when one card is compatible
    with both parent-local finite contracts.  That eligibility is necessarily
    data dependent: a healthy campaign may either randomize the admitted card
    over active/neutral lanes or emit the typed no-yield recourse.  Only the
    first path realizes the memory estimand, but both paths are valid workflow
    outcomes when their downstream evidence is internally consistent.
    """

    resolutions = summary["matched_memory_support_resolutions"]
    plans = summary["matched_memory_control_plans"]
    learning = summary["portfolio_stage_learning_records"]
    if type(resolutions) is not list or type(plans) is not list:
        raise TypeError("Timeloop memory audit collections must be lists")
    if type(learning) is not list or len(learning) < 3:
        raise ValueError("Timeloop memory audit requires the G5 learning record")
    preparation = learning[2]["evidence"]["generation_audit_preparation"]
    outcomes = preparation["matched_memory_control_outcomes"]
    if type(outcomes) is not list:
        raise TypeError("matched memory control outcomes must be a list")

    active_neutral_realized = (
        summary["memory_trials"] == 0
        and summary["bounded_g5_dose_request_count"] == 1
        and summary["bounded_g5_dose_result_count"] == 1
        and summary["bounded_g5_dose_assessments_pass"] is True
        and len(resolutions) == 1
        and resolutions[0]["selected_card_key"] is not None
        and len(resolutions[0]["selected_lane_supports"]) == 2
        and len(plans) == 1
        and {value["arm"] for value in plans[0]["assignments"]} == {"m", "n"}
        and plans[0]["single_block_card_effect_identified"] is False
        and plans[0]["online_score_update_allowed"] is False
        and preparation["projection"] is None
        and len(outcomes) == 1
        and summary["typed_recourse_receipts"] == 0
    )
    typed_no_shared_support_recourse = (
        summary["memory_trials"] == 0
        and summary["bounded_g5_dose_request_count"] == 0
        and summary["bounded_g5_dose_result_count"] == 0
        and len(resolutions) == 1
        and resolutions[0]["selected_card_key"] is None
        and resolutions[0]["eligible_card_keys"] == []
        and resolutions[0]["selected_lane_supports"] == []
        and plans == []
        and summary["typed_recourse_receipts"] == 1
        and preparation["projection"] is None
        and outcomes == []
    )
    return {
        "schema_version": 1,
        "active_neutral_assay_realized": active_neutral_realized,
        "typed_no_shared_support_recourse_realized": (typed_no_shared_support_recourse),
        "workflow_path_valid": (
            active_neutral_realized or typed_no_shared_support_recourse
        ),
        "memory_effect_claim_available": active_neutral_realized,
    }


def _construction_probe(replicate_seed: int) -> dict[str, object]:
    utility = AffineHypervolumeArchiveUtility3D(_utility_spec())
    run = run_provider_free_timeloop_campaign(
        outer_seed=replicate_seed,
        id_namespace=_id_namespace(replicate_seed),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        calibrated_allocator=_calibrated_allocator(),
        archive_utility=utility,
        recombination_utility_binder=_RecombinationUtilityBinder(utility),
    )
    summary = run.summary()
    contextual_histories = [
        thaw_json(request.context)["campaign_contextual_history"]
        for request, _ in run.selector.results
    ]
    archive_contexts = [
        thaw_json(request.context).get(CAMPAIGN_ARCHIVE_CONTEXT_KEY)
        for request, _ in run.selector.results
    ]
    contextual_counts = [len(value["actions"]) for value in contextual_histories]
    contextual_by_cutoff = _contextual_history_counts_by_cutoff(contextual_histories)
    memory_path_audit = _g5_memory_path_audit(summary)
    expected_proposal_support = _proposal_support_policy()
    proposal_support_policy_bound = all(
        result.supplemental_audit is not None
        and (
            ("proposal_support" not in thaw_json(result.supplemental_audit.payload))
            if expected_proposal_support is None
            else (
                thaw_json(result.supplemental_audit.payload)
                .get("proposal_support", {})
                .get("policy")
                == {
                    "policy_id": expected_proposal_support.policy_id,
                    "policy_version": expected_proposal_support.policy_version,
                    "definition_sha256": (expected_proposal_support.definition_sha256),
                }
            )
        )
        for _, result in run.selector.results
    )
    gates = {
        "six_generations": summary["generations_completed"] == GENERATION_COUNT,
        "planned_candidate_occurrences_exact": (
            summary["candidate_occurrences"] == PLANNED_CANDIDATE_OCCURRENCES
        ),
        "seven_logical_calls": summary["logical_agent_calls"] == PLANNED_LOGICAL_CALLS,
        "six_k8_selector_calls": (
            summary["selector_calls"] == 6
            and summary["k8_typed_proposals"] == 6
            and summary["direct_portfolio_selections"] == 0
        ),
        "one_g2_reflection": summary["reflection_generations"] == [2],
        "canonical_reflection_replays": summary["canonical_reflection_records"] == 1,
        "g5_memory_path_typed_and_audited": memory_path_audit["workflow_path_valid"]
        is True,
        "authenticated_mutation_observation_count_exact": (
            summary["authenticated_action_observations"] == 6 * PORTFOLIO_WIDTH
        ),
        "feedback_and_forecast_observation_counts_exact": (
            summary["outcome_feedback_receipts"] == 6
            and summary["forecast_calibration_observations"] == 6 * PORTFOLIO_WIDTH * 3
        ),
        "prior_only_contextual_feedback_reaches_later_waves": (
            contextual_counts[:2] == [0, 0]
            and all(value > 0 for value in contextual_counts[2:])
            and list(contextual_by_cutoff) == ["1", "3", "5"]
            and all(
                action["wave_index"] < history["cutoff_wave_index_exclusive"]
                for history in contextual_histories
                for action in history["actions"]
            )
        ),
        "provider_and_docker_free": (
            summary["provider_calls"] == 0 and summary["docker_calls"] == 0
        ),
        "full_support_policy_bound": all(
            result.supplemental_audit is not None
            and thaw_json(result.supplemental_audit.payload)["allocator_policy"][
                "definition_sha256"
            ]
            == _allocation_policy_identity()[2]
            for _, result in run.selector.results
        ),
        "proposal_support_policy_bound_when_configured": (
            proposal_support_policy_bound
        ),
        "archive_context_arm_exact": (
            all(value is None for value in archive_contexts)
            if ARCHIVE_CONTEXT_MODE.value == "off"
            else all(
                type(value) is dict
                and value.get("projector", {}).get("projector_id")
                == "authenticated_affine_frontier_context"
                and value.get("payload", {})
                .get("optimization_frame", {})
                .get("dimension")
                == 3
                and value.get("payload", {})
                .get("epistemic_cutoff", {})
                .get("current_or_future_candidate_outcomes_consulted")
                is False
                for value in archive_contexts
            )
        ),
    }
    return {
        "schema_version": 1,
        "replicate_seed": replicate_seed,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "summary": summary,
        "contextual_history_action_counts": contextual_counts,
        "contextual_history_action_counts_by_cutoff": contextual_by_cutoff,
        "g5_memory_path_audit": memory_path_audit,
        "archive_context_projections": [
            (
                None
                if value is None
                else {
                    "projector": value["projector"],
                    "archive_utility_snapshot_sha256": value[
                        "archive_utility_snapshot_sha256"
                    ],
                    "parent_configuration_sha256": value["parent_configuration_sha256"],
                    "projection_sha256": value["projection_sha256"],
                    "dimension": value["payload"]["optimization_frame"]["dimension"],
                }
            )
            for value in archive_contexts
        ],
        "selection_decisions": [
            {
                "request_sha256": request.request_sha256,
                "decision": result.decision.to_audit_record(),
                "supplemental_audit": (
                    None
                    if result.supplemental_audit is None
                    else result.supplemental_audit.to_record()
                ),
            }
            for request, result in run.selector.results
        ],
        "method": {
            "acquisition_mode": ACQUISITION_MODE.value,
            "execution_label": _acquisition_execution_label(),
            "proposal_support_policy": (
                None
                if expected_proposal_support is None
                else expected_proposal_support.to_record()
            ),
            "method_definition_sha256": (
                None
                if summary.get("experiment_profile") is None
                else summary["experiment_profile"]["method_definition_sha256"]
            ),
        },
    }


def _preregistration(
    *,
    replicate_seed: int,
    source_sha256: str,
    probe: dict[str, object],
    baseline: dict[str, object],
    qualification: dict[str, object],
) -> dict[str, object]:
    allocation_policy_id, allocation_policy_version, allocation_definition = (
        _allocation_policy_identity()
    )
    return {
        "schema_version": 1,
        "experiment_id": (
            f"timeloop_v2_{'common_pool' if _common_pool_enabled() else 'full_support'}_"
            f"{MODEL_PROFILE_NAME}_{PORTFOLIO_SELECTOR_MODE}_g6_"
            f"seed_{replicate_seed}_{_PROTOCOL_VERSION}"
        ),
        "claim_boundary": (
            "prospective_developmental_agentic_trace_and_candidate_quality_run;"
            "not_a_sota_claim;historical_gate_a_is_not_a_matched_control"
        ),
        "source_aggregate_sha256": source_sha256,
        "protocol_id": PROTOCOL_ID,
        "protocol_definition_sha256": PROTOCOL_DEFINITION_SHA256,
        "campaign_sha256": _campaign_sha256(replicate_seed),
        "task_sha256": TASK_SHA256,
        "outer_seed": replicate_seed,
        "replicate_seed": replicate_seed,
        "model": MODEL,
        "model_execution_profile": MODEL_EXECUTION_PROFILE.to_record(),
        "provider_only": list(PROVIDER_ONLY),
        "resolved_provider": RESOLVED_PROVIDER,
        "reasoning_effort": MODEL_EXECUTION_PROFILE.reasoning_effort,
        "reasoning_mode": None,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature_hex": (None if TEMPERATURE is None else TEMPERATURE.hex()),
        "provider_config": _provider_config(replicate_seed).to_manifest_record(),
        "schedule": {
            "generations": GENERATION_COUNT,
            "candidate_occurrences": PLANNED_CANDIDATE_OCCURRENCES,
            "maximum_cache_reuse_occurrences": MAX_CACHE_REUSE_OCCURRENCES,
            "logical_calls": PLANNED_LOGICAL_CALLS,
            "selector_calls": 6,
            "reflection_calls": 1,
            "proposal_width": 8,
            "candidate_universe_width": (
                COMMON_CANDIDATE_POOL_SIZE if _common_pool_enabled() else 8
            ),
            "evaluation_width": PORTFOLIO_WIDTH,
            "portfolio_generations": [1, 3, 5],
            "recombination_generations": [2, 4, 6],
        },
        "selection_policy": {
            "implementation_mode": PORTFOLIO_SELECTOR_MODE,
            "outcome_conditioned_all_action": (
                None
                if PORTFOLIO_SELECTOR_MODE != "outcome_conditioned"
                else {
                    "base_policy_definition_sha256": (
                        OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256
                    ),
                    "prompt_definition_sha256": (
                        ACTION_FORECAST_POLICY_DEFINITION_SHA256
                    ),
                    "partition_policy_definition_sha256": (
                        ACTION_FORECAST_PARTITION_DEFINITION_SHA256
                    ),
                    "max_rows_per_block": ACTION_FORECAST_BLOCK_ROWS,
                    "all_finite_actions_forecast": True,
                    "terminal_probe_quota": 0,
                    "optional_workload_action_semantics": False,
                    "optional_exact_metric_projection": False,
                }
            ),
            "policy_id": allocation_policy_id,
            "policy_version": allocation_policy_version,
            "definition_sha256": allocation_definition,
            "acquisition_mode": ACQUISITION_MODE.value,
            "proposal_support_policy": (
                None
                if _proposal_support_policy() is None
                else _proposal_support_policy().to_record()
            ),
            "feasibility_witness_mode": FEASIBILITY_WITNESS_MODE.value,
            "archive_context_mode": ARCHIVE_CONTEXT_MODE.value,
        },
        "utility": _utility_spec().to_record(),
        "utility_definition_sha256": _utility_spec().definition_sha256,
        "evaluator": {
            "image_ref": PINNED_IMAGE_REF,
            "image_id": PINNED_IMAGE_ID,
            "cpu_set": CPU_SET,
            "timeout_s_hex": float(EVALUATOR_TIMEOUT_S).hex(),
            "network_panel": "resnet50",
            "external_concurrency": 1,
        },
        "health_gates": {
            "provider_free_construction_probe_passes": True,
            "reference_qualification_envelope_passes": True,
            "source_closure_unchanged": True,
            "pinned_docker_preflight_passes_before_credential_read": True,
            "campaign_completes_six_generations": True,
            "candidate_occurrences_match_preregistered_plan": True,
            "logical_agent_calls_equal_7": True,
            "six_selector_and_one_reflection_responses": True,
            "every_success_uses_exact_model_provider": True,
            "every_success_matches_profile_reasoning_contract": True,
            "six_outcome_feedback_receipts": True,
            "selected_forecast_observations_match_preregistered_plan": True,
            "prior_only_contextual_history_reaches_g3_and_g5": True,
            "all_candidate_outcomes_terminal_and_typed": True,
            "all_successful_objectives_inside_fixed_reference": True,
            "runner_cleanup_released": True,
            "proposal_support_policy_bound_when_configured": True,
        },
        "candidate_quality_endpoints": {
            "primary": "final_all_evaluated_affine_3d_hv_minus_two_seed_hv",
            "promotion_gate": "strictly_positive_primary_gain",
            "secondary": [
                "at_least_one_evaluated_candidate_strictly_dominates_a_seed",
                "final_nondominated_front_size",
                "historical_gate_a_hv_difference_diagnostic_only",
            ],
        },
        "historical_gate_a": baseline,
        "real_qualification_evidence": qualification,
        "provider_free_probe_identity_sha256": typed_json_sha256(freeze_json(probe)),
    }


def _validate_preregistration(
    *,
    path: Path,
    replicate_seed: int,
    source_sha256: str,
    probe: dict[str, object],
    baseline: dict[str, object],
    qualification: dict[str, object],
) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    value = json.loads(payload.decode("utf-8", errors="strict"))
    expected = _preregistration(
        replicate_seed=replicate_seed,
        source_sha256=source_sha256,
        probe=probe,
        baseline=baseline,
        qualification=qualification,
    )
    if value != expected:
        raise RuntimeError("preregistration differs from the exact prepared contract")
    return {
        "path": resolved.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


class _ObservedJournal:
    def __init__(self, journal: DurableJsonlJournal, started_ns: int) -> None:
        self.journal = journal
        self.started_ns = started_ns

    def append(self, value: dict[str, object]) -> None:
        self.journal.append(
            {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - self.started_ns
                    ),
                    "observed_at_utc": _utc_now(),
                },
                "authenticated_record": value,
            }
        )


class _ExecutionJournal:
    def __init__(self, journal: _ObservedJournal) -> None:
        self.journal = journal

    async def append(self, event: CampaignExecutionEvent):
        self.journal.append({"authenticated_campaign_event": event.to_record()})
        return CampaignJournalAck(event.event_sha256, True)


class _CountingRunner:
    def __init__(self, delegate: object, journal: _ObservedJournal) -> None:
        self.delegate = delegate
        self.journal = journal
        self.calls = 0
        self.responses: list[dict[str, object]] = []

    async def __call__(self, request):
        self.calls += 1
        ordinal = self.calls
        response = await self.delegate(request)
        if type(response) is AttemptedStructuredGenerationResponse:
            response.__post_init__()
            inner = response.response
            attempt_count = response.attempt_count
        elif type(response) is StructuredGenerationResponse:
            response.__post_init__()
            inner = response
            attempt_count = 1
        else:
            raise TypeError("queued runner returned a foreign response envelope")
        telemetry = _telemetry_record(inner)
        telemetry["attempt_count"] = attempt_count
        record = {
            "logical_call_ordinal": ordinal,
            "call_id": request.call_id.value,
            "operation": request.operation,
            "telemetry": telemetry,
        }
        self.responses.append(record)
        self.journal.append(record)
        return response

    async def aclose(self) -> None:
        await self.delegate.aclose()

    async def snapshot(self):
        return await self.delegate.snapshot()


@dataclass(slots=True)
class _OwnedRunner:
    runner: _CountingRunner

    async def close(self) -> FrozenJsonObject:
        await self.runner.aclose()
        snapshot = await self.runner.snapshot()
        return _object(
            {
                "ownership": "timeloop_campaign_runtime",
                "runner_closed": bool(snapshot.closed),
                "pending": snapshot.pending,
                "in_flight": snapshot.in_flight,
            }
        )


class _LiveReflectionExecutor:
    def __init__(
        self,
        *,
        generator: PydanticAIAgenticGenerator,
        ids: DeterministicIdFactory,
        optimization_semantics: object,
        journal: _ObservedJournal,
    ) -> None:
        self.generator = generator
        self.ids = ids
        self.optimization_semantics = optimization_semantics
        self.journal = journal
        self.generations: list[int] = []
        self.records: list[FrozenJsonObject] = []
        self.inputs: list[CampaignIdentifiableReflectionInput] = []
        self.prompts: list[str] = []

    async def reflect(self, reflection_input: CampaignIdentifiableReflectionInput):
        cluster_count = len(
            cluster_identifiable_mutation_reflection_hypotheses(
                reflection_input.evidence.contrasts
            )
        )
        insight_count = min(8, cluster_count)
        if insight_count < 2:
            raise RuntimeError(
                "live Timeloop reflection requires at least two exact-action clusters"
            )
        request = build_timeloop_v2_identifiable_reflection_request(
            call_id=self.ids.new_llm_call_id(),
            reflection_input=reflection_input,
            optimization_semantics=self.optimization_semantics,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            min_insights=insight_count,
            max_insights=insight_count,
        )
        result = await self.generator.reflect(request)
        telemetry = result.telemetry
        try:
            MODEL_EXECUTION_PROFILE.validate_telemetry(telemetry)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "reflection violated model/provider/reasoning gates"
            ) from error
        envelope = build_timeloop_v2_identifiable_learning_envelope(
            reflection_input=reflection_input,
            request=request,
            result=result,
            optimization_semantics=self.optimization_semantics,
        )
        self.generations.append(reflection_input.query.wave.source_generation)
        self.records.append(envelope)
        self.inputs.append(reflection_input)
        self.prompts.append(request.prompt)
        self.journal.append(
            {
                "call_id": request.call_id.value,
                "source_generation": reflection_input.query.wave.source_generation,
                "input_sha256": reflection_input.input_sha256,
                "prompt_sha256": hashlib.sha256(
                    request.prompt.encode("utf-8")
                ).hexdigest(),
                "evidence_snapshot_sha256": reflection_input.evidence.snapshot_sha256,
                "contrast_count": len(reflection_input.evidence.contrasts),
                "insight_count": len(result.insights),
                "insights": [value.content_record() for value in result.insights],
                "learning_envelope": thaw_json(envelope),
                "telemetry": _telemetry_record(telemetry),
            }
        )
        return envelope


class _CountingEvaluator:
    def __init__(self, delegate: TimeloopV2DockerEvaluator, journal: _ObservedJournal):
        self.delegate = delegate
        self.journal = journal
        self.calls = 0
        self.observations: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def evaluate(self, config: object) -> TimeloopV2Evaluation:
        started_ns = time.perf_counter_ns()
        with self._lock:
            self.calls += 1
            ordinal = self.calls
        config_sha256 = candidate_sha256(config)
        try:
            result = self.delegate.evaluate(config)
        except TimeloopV2CandidateInfeasibleError as error:
            observation = error.observation
            record = {
                "ordinal": ordinal,
                "status": "candidate_infeasible",
                "configuration_sha256": config_sha256,
                "candidate_sha256": observation.candidate_sha256,
                "elapsed_ns": time.perf_counter_ns() - started_ns,
                "incomplete_medoid_ordinals": list(
                    observation.incomplete_medoid_ordinals
                ),
                "output_dir": str(observation.output_dir.resolve()),
            }
            with self._lock:
                self.observations.append(record)
                self.journal.append(record)
            raise
        except BaseException as error:
            record = {
                "ordinal": ordinal,
                "status": "runtime_failure",
                "configuration_sha256": config_sha256,
                "elapsed_ns": time.perf_counter_ns() - started_ns,
                "failure_type": type(error).__qualname__,
                "failure_sha256": _sha(f"{type(error).__qualname__}\x00{error}"),
            }
            with self._lock:
                self.observations.append(record)
                self.journal.append(record)
            raise
        record = {
            "ordinal": ordinal,
            "status": "passed",
            "configuration_sha256": config_sha256,
            "candidate_sha256": result.candidate_sha256,
            "compiled_plan_sha256": result.compiled_plan_sha256,
            "objectives": dict(result.objective_values),
            "evaluator_elapsed_s": result.evaluator_elapsed_s,
            "queue_wait_s": result.queue_wait_s,
            "elapsed_ns": time.perf_counter_ns() - started_ns,
            "output_dir": str(result.output_dir.resolve()),
        }
        with self._lock:
            self.observations.append(record)
            self.journal.append(record)
        return result


def _open_journals(
    run_dir: Path, started_ns: int
) -> tuple[dict[str, Any], dict[str, _ObservedJournal]]:
    raw: dict[str, Any] = {
        "campaign": DurableJsonlJournal(run_dir / "campaign_events.jsonl"),
        "engine": DurableJsonlJournal(run_dir / "engine_events.jsonl"),
        "evaluations": DurableJsonlJournal(run_dir / "evaluator_observations.jsonl"),
        "requests": DurableJsonlJournal(run_dir / "request_evidence.jsonl"),
        "outputs": DurableJsonlJournal(run_dir / "output_evidence.jsonl"),
        "outcomes": DurableJsonlJournal(run_dir / "queue_outcomes.jsonl"),
        "outbound": DurableJsonlJournal(run_dir / "outbound_requests.jsonl"),
        "responses": DurableJsonlJournal(run_dir / "logical_responses.jsonl"),
        "reflections": DurableJsonlJournal(run_dir / "reflection_records.jsonl"),
        "progress": BatchedDurableJsonlJournal(
            run_dir / "stream_progress.jsonl", max_unfsynced_rows=32
        ),
    }
    observed = {
        key: _ObservedJournal(value, started_ns)
        for key, value in raw.items()
        if key != "progress"
    }
    return raw, observed


def _pareto_front(points: list[dict[str, float]]) -> list[dict[str, float]]:
    front: list[dict[str, float]] = []
    for point in points:
        dominated = any(
            all(other[key] <= point[key] for key in _utility_spec().metric_ids)
            and any(other[key] < point[key] for key in _utility_spec().metric_ids)
            for other in points
        )
        if not dominated and point not in front:
            front.append(point)
    return sorted(
        front, key=lambda value: tuple(value[key] for key in _utility_spec().metric_ids)
    )


def _strictly_dominates(first: dict[str, float], second: dict[str, float]) -> bool:
    keys = _utility_spec().metric_ids
    return all(first[key] <= second[key] for key in keys) and any(
        first[key] < second[key] for key in keys
    )


def _execute_live(
    *,
    run_dir: Path,
    replicate_seed: int,
    source_sha256: str,
    docker_preflight: dict[str, object],
    reference_qualification_passed: bool,
) -> dict[str, object]:
    started_ns = time.perf_counter_ns()
    id_namespace = _id_namespace(replicate_seed)
    raw_journals, journals = _open_journals(run_dir, started_ns)
    runner: _CountingRunner | None = None
    try:
        settings = TimeloopV2Settings(
            output_root=run_dir / "evaluator_calls",
            cpu_set=CPU_SET,
            timeout_s=EVALUATOR_TIMEOUT_S,
        )
        panel = frozen_network_panel("resnet50")
        raw_evaluator = TimeloopV2DockerEvaluator(settings, panel)
        evaluator = _CountingEvaluator(raw_evaluator, journals["evaluations"])
        benchmark = compose_timeloop_v2_detailed_benchmark(
            settings,
            panel,
            artifact_store=FileSystemArtifactStore(run_dir / "artifact_store"),
            evaluator=evaluator,
        )
        detailed = benchmark.detailed_evaluator
        if detailed is None:
            raise RuntimeError("Timeloop live benchmark omitted detailed evidence")

        api_key = _read_live_api_key()

        def progress_sink(value: StructuredStreamProgress) -> None:
            raw_journals["progress"].append(
                {
                    "observation": {
                        "monotonic_ns_since_execution_start": (
                            time.perf_counter_ns() - started_ns
                        ),
                        "observed_at_utc": _utc_now(),
                    },
                    "authenticated_record": _progress_record(value),
                }
            )

        def outcome_sink(value: object) -> None:
            raw_journals["progress"].flush()
            journals["outcomes"].append(structured_generation_outcome_record(value))

        delegate = create_progress_aware_openrouter_runner(
            api_key=api_key,
            config=_provider_config(replicate_seed),
            progress_sink=progress_sink,
            outcome_sink=outcome_sink,
            request_evidence_sink=lambda value: journals["requests"].append(value),
            output_evidence_sink=lambda value: journals["outputs"].append(value),
            outbound_request_manifest_sink=lambda value: journals["outbound"].append(
                value
            ),
            evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
        )
        runner = _CountingRunner(delegate, journals["responses"])
        generator = PydanticAIAgenticGenerator(runner)
        utility = AffineHypervolumeArchiveUtility3D(_utility_spec())
        portfolio_selector_override = (
            _outcome_conditioned_selector(runner=runner, benchmark=benchmark)
            if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
            else None
        )

        reflection_holder: dict[str, _LiveReflectionExecutor] = {}

        def reflection_factory(ids: DeterministicIdFactory, semantics: object):
            executor = _LiveReflectionExecutor(
                generator=generator,
                ids=ids,
                optimization_semantics=semantics,
                journal=journals["reflections"],
            )
            reflection_holder["executor"] = executor
            return executor

        run = run_timeloop_campaign(
            benchmark=benchmark,
            evaluator=evaluator,
            execution_mode=(
                f"real_docker_{MODEL_PROFILE_NAME}_{_acquisition_execution_label()}_g6"
            ),
            id_namespace=id_namespace,
            campaign_sha256=_campaign_sha256(replicate_seed),
            evaluator_contract_sha256=detailed.evaluator_identity.evaluator_context_sha256,
            protocol_id=PROTOCOL_ID,
            protocol_definition_sha256=PROTOCOL_DEFINITION_SHA256,
            task_sha256=TASK_SHA256,
            evaluator_preflight_receipt=_object(
                {
                    "qualified": True,
                    "mode": (
                        f"real_docker_{MODEL_PROFILE_NAME}_"
                        f"{_acquisition_execution_label()}_g6"
                    ),
                    "preflight": docker_preflight,
                }
            ),
            resource_lease_receipt=_object(
                {
                    "resource": "serial_timeloop_docker_cpu_8",
                    "active": True,
                    "evaluator_concurrency": 1,
                }
            ),
            docker_enabled=True,
            scientific_claim="prospective_developmental_trace_and_candidate_quality",
            outer_seed=replicate_seed,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            calibrated_runner=runner,
            calibrated_allocator=_calibrated_allocator(),
            portfolio_selector_override=portfolio_selector_override,
            reflection_executor_factory=reflection_factory,
            model_profile_sha256=_model_profile_sha256(),
            selector_policy_binding_id=(_acquisition_execution_label()),
            reflection_policy_binding_id="identifiable_mutation_reflection",
            provider_enabled=True,
            owned_resources=_OwnedRunner(runner),
            execution_journal=_ExecutionJournal(journals["campaign"]),
            engine_trace_sink=lambda value: journals["engine"].append(dict(value)),
            archive_utility=utility,
            recombination_utility_binder=_RecombinationUtilityBinder(utility),
            model_execution_profile=MODEL_EXECUTION_PROFILE,
            evaluator_concurrency=1,
            agent_concurrency=AGENT_CONCURRENCY,
            agent_queue_capacity=AGENT_QUEUE_CAPACITY,
        )

        base_summary = run.summary()
        expected_proposal_support = _proposal_support_policy()
        memory_path_audit = _g5_memory_path_audit(base_summary)
        stage_occurrences = tuple(
            value.candidate_occurrence_count for value in run.execution.stage_receipts
        )
        stage_unique_evaluations = tuple(
            value.unique_evaluation_count for value in run.execution.stage_receipts
        )
        evaluation_accounting = CampaignEvaluationAccounting(
            planned_candidate_occurrences=PLANNED_CANDIDATE_OCCURRENCES,
            seed_occurrences=(
                run.execution.counters.candidate_occurrences - sum(stage_occurrences)
            ),
            seed_unique_evaluations=(
                run.execution.counters.unique_evaluations
                - sum(stage_unique_evaluations)
            ),
            stage_occurrences=stage_occurrences,
            stage_unique_evaluations=stage_unique_evaluations,
            candidate_occurrences=run.execution.counters.candidate_occurrences,
            unique_evaluations=run.execution.counters.unique_evaluations,
        )
        typed_candidate_infeasible_events = _typed_candidate_infeasible_events(
            run.engine_traces
        )
        typed_candidate_infeasible_count = len(typed_candidate_infeasible_events)
        pre_simulator_infeasible_count = _pre_simulator_infeasible_count(
            typed_candidate_infeasible_events
        )
        portfolio_candidate_infeasible_count = _portfolio_candidate_infeasible_count(
            run
        )
        planned_forecast_actions = len(run.selector.results) * PORTFOLIO_WIDTH
        expected_forecast_observations = (
            planned_forecast_actions - portfolio_candidate_infeasible_count
        ) * 3
        contextual_histories = [
            thaw_json(request.context)["campaign_contextual_history"]
            for request, _ in run.selector.results
        ]
        contextual_by_cutoff = _contextual_history_counts_by_cutoff(
            contextual_histories
        )
        contextual_feedback_gate = (
            list(contextual_by_cutoff) == ["1", "3", "5"]
            and all(len(value) == 2 for value in contextual_by_cutoff.values())
            and contextual_by_cutoff["1"] == [0, 0]
            and all(value > 0 for value in contextual_by_cutoff["3"])
            and all(value > 0 for value in contextual_by_cutoff["5"])
            and all(
                action["wave_index"] < history["cutoff_wave_index_exclusive"]
                for history in contextual_histories
                for action in history["actions"]
            )
        )
        for request, result in run.selector.results:
            journals["responses"].append(
                {
                    "record_kind": "selection_decision",
                    "request_sha256": request.request_sha256,
                    "decision": result.decision.to_audit_record(),
                    "supplemental_audit": (
                        None
                        if result.supplemental_audit is None
                        else result.supplemental_audit.to_record()
                    ),
                    "telemetry": (
                        None
                        if result.telemetry is None
                        else _telemetry_record(result.telemetry)
                    ),
                }
            )

        successes = [
            dict(value["objectives"])
            for value in evaluator.observations
            if value["status"] == "passed"
        ]
        runtime_failures = [
            value
            for value in evaluator.observations
            if value["status"] == "runtime_failure"
        ]
        candidate_infeasible = [
            value
            for value in evaluator.observations
            if value["status"] == "candidate_infeasible"
        ]
        if len(successes) < 2:
            raise RuntimeError("live campaign did not preserve two successful seeds")
        seed_points = successes[:2]
        seed_hv = AffineHypervolumeSnapshot3D.create(
            spec=_utility_spec(), archive_points=tuple(seed_points)
        ).base_hypervolume
        final_snapshot = AffineHypervolumeSnapshot3D.create(
            spec=_utility_spec(), archive_points=tuple(successes)
        )
        final_hv = final_snapshot.base_hypervolume
        front = _pareto_front(successes)

        outbound_rows = read_jsonl(run_dir / "outbound_requests.jsonl")
        validated_outbound = [
            validate_openrouter_outbound_request_manifest_record(
                row["authenticated_record"]
            )
            for row in outbound_rows
        ]
        response_telemetry = [value["telemetry"] for value in runner.responses]
        exact_route_and_reasoning = all(
            value["requested_model"] == MODEL
            and value["resolved_model"]
            in MODEL_EXECUTION_PROFILE.accepted_resolved_models
            and value["resolved_provider"]
            in MODEL_EXECUTION_PROFILE.accepted_resolved_providers
            and value["finish_reason"]
            in MODEL_EXECUTION_PROFILE.accepted_finish_reasons
            and MODEL_EXECUTION_PROFILE.accepts_reasoning_tokens(
                value.get("reasoning_tokens")
            )
            for value in response_telemetry
        )
        outbound_gate = all(
            value["settings"]["model"] == MODEL
            and value["settings"]["provider"]
            == {"allow_fallbacks": False, "only": list(PROVIDER_ONLY)}
            and value["settings"]["reasoning"]
            == MODEL_EXECUTION_PROFILE.outbound_reasoning_setting
            and value["settings"]["max_completion_tokens"] == MAX_OUTPUT_TOKENS
            for value in validated_outbound
        )
        reference_contains_all = all(
            all(value < 1.0 for value in _utility_spec().normalize(point))
            for point in successes
        )
        dominates_seed = any(
            _strictly_dominates(point, seed)
            for point in successes[2:]
            for seed in seed_points
        )
        health = {
            "reference_qualification_envelope_passed": (
                reference_qualification_passed is True
            ),
            "six_generations": base_summary["generations_completed"]
            == GENERATION_COUNT,
            "planned_candidate_occurrences_exact": (
                base_summary["candidate_occurrences"] == PLANNED_CANDIDATE_OCCURRENCES
            ),
            "exact_evaluation_accounting": True,
            "bounded_cache_reuse": evaluation_accounting.within_cache_reuse_limit(
                MAX_CACHE_REUSE_OCCURRENCES
            ),
            "physical_calls_match_unique_evaluations": (
                evaluator.calls + pre_simulator_infeasible_count
                == evaluation_accounting.unique_evaluations
            ),
            "seven_logical_calls": base_summary["logical_agent_calls"]
            == 7
            and (
                runner.calls >= 7
                if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
                else runner.calls == 7
            ),
            "six_selector_responses": len(run.selector.results) == 6,
            "six_outcome_feedback_receipts": (
                base_summary["outcome_feedback_receipts"] == 6
            ),
            "selected_forecast_observations_match_preregistered_plan": (
                base_summary["forecast_calibration_observations"]
                == expected_forecast_observations
            ),
            "prior_only_contextual_feedback_reaches_g3_and_g5": (
                contextual_feedback_gate
            ),
            "one_g2_reflection": base_summary["reflection_generations"] == [2],
            "g5_memory_path_typed_and_audited": memory_path_audit["workflow_path_valid"]
            is True,
            "seven_successful_provider_responses": (
                len(runner.responses) >= 7
                if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
                else len(runner.responses) == 7
            ),
            "exact_model_provider_reasoning_contract": exact_route_and_reasoning,
            "outbound_transport_contract": outbound_gate,
            "no_runtime_evaluator_failures": not runtime_failures,
            "all_candidate_outcomes_terminal_and_typed": (
                len(successes)
                + len(candidate_infeasible)
                + len(runtime_failures)
                + pre_simulator_infeasible_count
                == evaluation_accounting.unique_evaluations
            ),
            "fixed_reference_contains_successes": reference_contains_all,
            "positive_hypervolume_gain_over_seeds": final_hv > seed_hv,
            "at_least_one_candidate_dominates_seed": dominates_seed,
            "cleanup_released": run.execution.cleanup_receipt.released,
            "source_closure_unchanged": (
                _require_source_closure(source_sha256)["aggregate_sha256"]
                == source_sha256
            ),
            "proposal_support_policy_bound_when_configured": all(
                result.supplemental_audit is not None
                and (
                    thaw_json(result.supplemental_audit.payload).get("schema_version")
                    == 1
                    if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
                    else
                    (
                        "proposal_support"
                        not in thaw_json(result.supplemental_audit.payload)
                    )
                    if expected_proposal_support is None
                    else (
                        thaw_json(result.supplemental_audit.payload)
                        .get("proposal_support", {})
                        .get("policy")
                        == {
                            "policy_id": expected_proposal_support.policy_id,
                            "policy_version": (
                                expected_proposal_support.policy_version
                            ),
                            "definition_sha256": (
                                expected_proposal_support.definition_sha256
                            ),
                        }
                    )
                )
                for _, result in run.selector.results
            ),
        }
        status = "completed_healthy" if all(health.values()) else "completed_unhealthy"
        gate_a = _load_gate_a_baseline()
        return {
            "schema_version": 1,
            "status": status,
            "replicate_seed": replicate_seed,
            "health": health,
            "campaign": base_summary,
            "method": {
                "acquisition_mode": ACQUISITION_MODE.value,
                "execution_label": _acquisition_execution_label(),
                "proposal_support_policy": (
                    None
                    if expected_proposal_support is None
                    else expected_proposal_support.to_record()
                ),
                "method_definition_sha256": (
                    None
                    if base_summary.get("experiment_profile") is None
                    else base_summary["experiment_profile"]["method_definition_sha256"]
                ),
                "experiment_definition_sha256": (
                    None
                    if base_summary.get("experiment_profile") is None
                    else base_summary["experiment_profile"][
                        "experiment_definition_sha256"
                    ]
                ),
            },
            "g5_memory_path_audit": memory_path_audit,
            "evaluation_accounting": evaluation_accounting.to_record(),
            "wall_s": (time.perf_counter_ns() - started_ns) / 1e9,
            "provider": {
                "logical_calls": runner.calls,
                "successful_responses": len(runner.responses),
                "physical_attempts": len(validated_outbound),
                "response_telemetry": response_telemetry,
                "total_cost_usd": str(
                    sum(float(value["cost_usd"] or 0.0) for value in response_telemetry)
                ),
            },
            "evaluator": {
                "physical_calls": evaluator.calls,
                "successful": len(successes),
                "candidate_infeasible": typed_candidate_infeasible_count,
                "physical_evaluator_candidate_infeasible": len(candidate_infeasible),
                "pre_simulator_candidate_infeasible": (pre_simulator_infeasible_count),
                "runtime_failures": len(runtime_failures),
                "median_evaluator_elapsed_s": (
                    sorted(
                        float(value["evaluator_elapsed_s"])
                        for value in evaluator.observations
                        if value["status"] == "passed"
                    )[len(successes) // 2]
                    if successes
                    else None
                ),
            },
            "terminal_accounting": {
                "unique_evaluations": evaluation_accounting.unique_evaluations,
                "physical_evaluator_calls": evaluator.calls,
                "pre_simulator_candidate_infeasible": (pre_simulator_infeasible_count),
                "planned_forecast_actions": planned_forecast_actions,
                "portfolio_candidate_infeasible": (
                    portfolio_candidate_infeasible_count
                ),
                "expected_forecast_observations": expected_forecast_observations,
                "observed_forecast_observations": base_summary[
                    "forecast_calibration_observations"
                ],
            },
            "candidate_quality": {
                "utility_definition_sha256": _utility_spec().definition_sha256,
                "seed_hypervolume_hex": seed_hv.hex(),
                "final_hypervolume_hex": final_hv.hex(),
                "absolute_gain_hex": (final_hv - seed_hv).hex(),
                "relative_gain": (
                    None if seed_hv == 0 else (final_hv - seed_hv) / seed_hv
                ),
                "nondominated_front_size": len(front),
                "nondominated_front": front,
                "dominates_seed": dominates_seed,
                "historical_gate_a_hypervolume_hex": gate_a["affine_hypervolume_hex"],
                "historical_gate_a_role": gate_a["role"],
                "beats_historical_gate_a_diagnostic": (
                    final_hv > float.fromhex(str(gate_a["affine_hypervolume_hex"]))
                ),
            },
            "reflection_records": [
                thaw_json(value) for value in reflection_holder["executor"].records
            ],
            "selection_decision_count": len(run.selector.results),
            "contextual_history_action_counts_by_cutoff": contextual_by_cutoff,
            "source_closure_sha256": source_sha256,
        }
    finally:
        for journal in raw_journals.values():
            journal.close()


def _manifest(
    *,
    run_id: str,
    mode: str,
    replicate_seed: int,
    source: dict[str, object],
    snapshot: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "replicate_seed": replicate_seed,
        "created_at_utc": _utc_now(),
        "source_identity": source,
        "source_snapshot": snapshot,
        "model_profile_sha256": _model_profile_sha256(),
        "portfolio_selector_mode": PORTFOLIO_SELECTOR_MODE,
        "outcome_conditioned_all_action": (
            None
            if PORTFOLIO_SELECTOR_MODE != "outcome_conditioned"
            else {
                "policy_definition_sha256": (
                    OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256
                ),
                "prompt_definition_sha256": ACTION_FORECAST_POLICY_DEFINITION_SHA256,
                "partition_definition_sha256": (
                    ACTION_FORECAST_PARTITION_DEFINITION_SHA256
                ),
                "max_rows_per_block": ACTION_FORECAST_BLOCK_ROWS,
            }
        ),
        "feasibility_witness_mode": FEASIBILITY_WITNESS_MODE.value,
        "common_candidate_pool_size": (
            COMMON_CANDIDATE_POOL_SIZE if _common_pool_enabled() else None
        ),
        "engine_evaluation_width": PORTFOLIO_WIDTH,
        "archive_context_mode": ARCHIVE_CONTEXT_MODE.value,
        "acquisition_mode": ACQUISITION_MODE.value,
        "acquisition_execution_label": _acquisition_execution_label(),
        "proposal_support_policy": (
            None
            if _proposal_support_policy() is None
            else _proposal_support_policy().to_record()
        ),
        "provider_config": _provider_config(replicate_seed).to_manifest_record(),
        "utility_definition_sha256": _utility_spec().definition_sha256,
    }


def _record_launch(
    args: argparse.Namespace,
    run_dir: Path,
    paths: tuple[Path, ...],
    source: dict[str, object],
) -> None:
    """Publish the complete launch environment beside the source closure.

    Called twice: once as soon as the run directory exists, so that even a
    failed preparation records how it was launched, and once again just
    before finalization, when the observed ambient-input set is complete.
    Never raises; a failure is journaled as ``launch_record_error.json``.
    """

    record_campaign_launch(
        mode=args.mode,
        run_id=args.run_id,
        run_dir=run_dir,
        workspace_root=WORKSPACE_ROOT,
        agent_evolve_root=AGENT_EVOLVE_ROOT,
        source_paths=paths,
        source_closure=source,
        dotenv_paths=(WORKSPACE_ROOT / ".env", AGENT_EVOLVE_ROOT / ".env"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "live"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicate-seed", required=True, type=int)
    parser.add_argument("--prereg")
    args = parser.parse_args()

    run_dir = (ARTIFACT_ROOT / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        paths = _source_paths()
        source = source_identity(paths, relative_to=WORKSPACE_ROOT)
        snapshot = _snapshot_sources(run_dir, paths)
        if (
            source["aggregate_sha256"] != snapshot["aggregate_sha256"]
            or source["file_count"] != snapshot["file_count"]
        ):
            raise RuntimeError("source changed while creating the launch snapshot")
        write_json_atomic(
            run_dir / "manifest.json",
            _manifest(
                run_id=args.run_id,
                mode=args.mode,
                replicate_seed=args.replicate_seed,
                source=source,
                snapshot=snapshot,
            ),
        )
        _record_launch(args, run_dir, paths, source)
        probe = _construction_probe(args.replicate_seed)
        write_json_atomic(run_dir / "provider_free_construction_probe.json", probe)
        if not probe["all_gates_pass"]:
            raise RuntimeError("provider-free Timeloop construction probe failed")
        baseline = _load_gate_a_baseline()
        write_json_atomic(run_dir / "historical_gate_a_baseline.json", baseline)
        qualification = _load_real_qualification_evidence()
        write_json_atomic(run_dir / "real_qualification_evidence.json", qualification)
        _require_source_closure(str(source["aggregate_sha256"]))

        if args.mode == "prepare":
            if args.prereg is not None:
                raise RuntimeError("prepare mode does not accept --prereg")
            prereg = _preregistration(
                replicate_seed=args.replicate_seed,
                source_sha256=str(source["aggregate_sha256"]),
                probe=probe,
                baseline=baseline,
                qualification=qualification,
            )
            write_json_atomic(run_dir / "preregistration_template.json", prereg)
            summary = {
                "schema_version": 1,
                "status": "prepared_without_credential_provider_or_candidate_docker_run",
                "replicate_seed": args.replicate_seed,
                "credential_read": False,
                "provider_calls": 0,
                "candidate_docker_evaluations": 0,
                "provider_free_construction_probe_passed": True,
                "source_aggregate_sha256": source["aggregate_sha256"],
                "utility_definition_sha256": _utility_spec().definition_sha256,
                "preregistration": prereg,
                "live_command_shape": (
                    "uv run python examples/development/"
                    "run_timeloop_v2_frontier_probe_live.py live "
                    "--run-id <id> --replicate-seed <seed> "
                    "--prereg <workspace-path>"
                ),
            }
            write_json_atomic(run_dir / "summary.json", summary)
            _record_launch(args, run_dir, paths, source)
            final = finalize_run_directory(run_dir, status=str(summary["status"]))
            print(json.dumps({**summary, "finalization": final}, sort_keys=True))
            return 0

        if args.prereg is None:
            raise RuntimeError("live mode requires --prereg")
        prereg_identity = _validate_preregistration(
            path=Path(args.prereg),
            replicate_seed=args.replicate_seed,
            source_sha256=str(source["aggregate_sha256"]),
            probe=probe,
            baseline=baseline,
            qualification=qualification,
        )
        write_json_atomic(run_dir / "preregistration_identity.json", prereg_identity)

        settings = TimeloopV2Settings(
            output_root=run_dir / "evaluator_calls",
            cpu_set=CPU_SET,
            timeout_s=EVALUATOR_TIMEOUT_S,
        )
        docker_preflight = TimeloopV2DockerEvaluator(
            settings, frozen_network_panel("resnet50")
        ).preflight()
        write_json_atomic(run_dir / "docker_preflight.json", docker_preflight)
        _require_source_closure(str(source["aggregate_sha256"]))

        summary = _execute_live(
            run_dir=run_dir,
            replicate_seed=args.replicate_seed,
            source_sha256=str(source["aggregate_sha256"]),
            docker_preflight=docker_preflight,
            reference_qualification_passed=(
                qualification["reference_envelope_audit"]["strictly_contains_all"]
                is True
            ),
        )
        write_json_atomic(run_dir / "summary.json", summary)
        final = finalize_run_directory(run_dir, status=str(summary["status"]))
        print(json.dumps({**summary, "finalization": final}, sort_keys=True))
        return 0 if summary["status"] == "completed_healthy" else 2
    except BaseException as error:
        if not (run_dir / "summary.json").exists():
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "schema_version": 1,
                    "status": "failed_before_completion",
                    "failure_type": type(error).__qualname__,
                    "failure_sha256": _sha(f"{type(error).__qualname__}\x00{error}"),
                },
            )
        if not (run_dir / "finalized.json").exists():
            finalize_run_directory(run_dir, status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

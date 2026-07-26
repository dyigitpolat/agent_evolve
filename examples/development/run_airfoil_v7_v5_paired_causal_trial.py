#!/usr/bin/env python3
"""Fresh prospective Airfoil-v7 v5 paired causal-trial harness.

``prepare`` reconstructs the already-frozen G1/reflection/card evidence from
the finalized v2 artifact, binds the method-independent paired block schedule,
assigns M/P/N calls to opaque provider slots, and fsyncs the exact v5 prompts,
schemas, call IDs, endpoint, allocation defaults, source identity, runtime, and
one-shot live target.  It reads no credential and opens no oracle outcome file.

``live`` re-verifies that sealed preparation before a credential can be read,
then submits exactly three opaque-assigned v5 calls through the bounded
progress-aware queue.  All calls settle and a terminal provider ledger is
atomically materialized before any allocation hook can run.  Two generic
allocator methods are committed and read back before one benchmark adapter
can open a union-only, one-shot outcome capability.  Exact 19-action ranks are
deliberately deferred to a separate post-commit rank-only authority.

This is one developmental block.  It supports neither significance claims nor
efficacy/generalization beyond the frozen block and protocol.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import threading
from typing import Any, Protocol

from pydantic import BaseModel

from agent_evolve.application.action_allocation_frame_commit import (
    build_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_allocation_frame_commit_v3 import (
    build_operational_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_allocation_frame_v3 import (
    OperationalGreedyForecastFrameAllocator,
)
from agent_evolve.application.action_forecast_partitioning import (
    action_forecast_block_call_id,
    assess_resolved_action_forecast_block_health,
    assess_resolved_action_forecast_block_subset_health,
    build_action_forecast_partition_layout,
    lenient_action_forecast_health_policy,
)
from agent_evolve.application.paired_allocation_comparison import (
    AllocationComparisonMethodWave,
    build_paired_allocation_comparison_commitment,
)
from agent_evolve.application.treatment_assignment import (
    assign_treatment_occurrences,
)
from agent_evolve.domain.finite_variation import bind_finite_action_evidence
from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.integrations.pydantic_ai import (
    action_forecast as action_forecast_adapter,
    queued_runner,
)
from agent_evolve.integrations.pydantic_ai.action_forecast import (
    ACTION_FORECAST_BLOCK_TOOL_NAME,
    ACTION_FORECAST_POLICY_DEFINITION_SHA256,
    ACTION_FORECAST_POLICY_VERSION,
    PydanticAIActionForecastBlockPolicy,
    plan_action_forecast_block_request,
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
from agent_evolve.ports.action_allocation_frame_v3 import (
    AllocationScoreResolutionBinding,
    AllocationV3SeedSamplingLaw,
    AllocationV3SelectionBinding,
    AllocationV3TieMode,
    OperationalFrameActionAllocationRequest,
)
from agent_evolve.ports.action_allocation_frame_commit_v3 import (
    OperationalFrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockRequest,
    ActionForecastBlockResult,
    ActionForecastPartitionLayout,
    ActionForecastRequest,
    resolve_action_forecast_block,
    validate_resolved_action_forecast_block,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.artifact_store import canonical_json_bytes, decode_json_bytes
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
)
from agent_evolve.ports.treatment_assignment import (
    OpaqueProviderSlotId,
    ProspectiveTreatmentAssignmentReceipt,
    TreatmentAssignment,
    TreatmentAssignmentInput,
    TreatmentId,
    TreatmentOccurrence,
    TreatmentOccurrenceId,
)
from examples.benchmarks.engibench_airfoil.v7_contract import TASK_SHA256
from examples.development import airfoil_v7_two_stage_agent_evolution as airfoil
from examples.development import run_airfoil_v7_forecast_wire_v3_pilot as historical
from examples.development import run_airfoil_v7_two_stage_generation as v2_launcher
from examples.development import run_airfoil_v7_v5_sealed_end_to_end_replay as sealed_replay
from examples.development.airfoil_v7_paired_block_schedule import (
    AirfoilV7PairedBlockAssignment,
    prepare_airfoil_v7_paired_block_assignment,
)
from examples.development.airfoil_v7_two_stage_agent_evolution import (
    AirfoilDevelopmentEvaluation,
    AirfoilG1ActionObservation,
    AirfoilObservedMetric,
    AirfoilTwoStageForecastArms,
    PreparedAirfoilTwoStageGeneration,
    VerifiedAirfoilPredecisionOracle,
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


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_RUN_ROOT = (
    ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7" / "paired_causal_v1"
)
DEFAULT_FROZEN_V2_RUN = historical.DEFAULT_FROZEN_V2_RUN
DEFAULT_ORACLE_DIR = airfoil.DEFAULT_SEALED_ORACLE_DIR

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
JITTER_SEED = 2_026_071_504
JITTER_DOMAIN = "airfoil-v7-v5-paired-causal-trial"
PROGRESS_MAX_UNFSYNCED_ROWS = 64

PUBLIC_SEED = 20_260_715
ASSIGNMENT_PUBLIC_SEED = "airfoil.v7.v5.paired.causal.20260715"
PORTFOLIO_SIZE = 3
V3_RISK_AVERSION = 0.5
V3_DIVERSITY_WEIGHT = 0.0
V3_MAXIMUM_INDISTINGUISHABLE_SCORE_GAP = 2.0**-20
V3_RESOLUTION_POLICY_ID = "airfoil_v7_normalized_utility_resolution"
V3_RESOLUTION_POLICY_VERSION = 1
V3_RESOLUTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:airfoil-v7-normalized-utility-resolution:v1;"
    b"utility-domain=[0,1];maximum-indistinguishable-gap=2^-20"
).hexdigest()
V3_SELECTION_POLICY_ID = "airfoil_v7_task_keyed_public_hash_rank"
V3_SELECTION_POLICY_VERSION = 1
V3_SELECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:airfoil-v7-task-keyed-public-hash-rank:v1;"
    b"fixed-public-seed=20260715;seed-provenance=frozen-g1-sample-receipt;"
    b"allocation-unit=task-contract-layout-block-mask-k;"
    b"treatment,provider,forecast,outcome-identities=excluded"
).hexdigest()

PRIMARY_ENDPOINT_ID = "airfoil_v7_selected_three_set_log_joint_failure"
PRIMARY_ENDPOINT_VERSION = 1
PRIMARY_ENDPOINT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:airfoil-v7-selected-three-set-log-joint-failure:v1;"
    b"quality=clip(-delta_v/0.005,-60,60)+0.05*tanh(-delta_f/0.001);"
    b"member-log-failure=log(sigmoid(-quality));"
    b"selected-set-endpoint=sum-member-log-failure;lower-is-better;k=3;"
    b"contrasts=p-minus-m,n-minus-m,v2-minus-v3;positive-favors-m-or-v3;"
    b"raw-selected-union-only"
).hexdigest()

_EXPERIMENT_FRAMING = b"agent-evolve:airfoil-v7-v5-paired-causal-trial:v1\x00"
_PREPARED_FRAMING = b"agent-evolve:airfoil-v7-v5-paired-prepared:v1\x00"
_MANIFEST_FRAMING = b"agent-evolve:airfoil-v7-v5-paired-live:v1\x00"
_TERMINAL_LEDGER_FRAMING = (
    b"agent-evolve:airfoil-v7-v5-paired-terminal-provider-ledger:v1\x00"
)


class FreshPairedTrialError(RuntimeError):
    """The fresh paired trial escaped its prospectively frozen contract."""


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    value = decode_json_bytes(path.expanduser().resolve(strict=True).read_bytes())
    if type(value) is not dict:
        raise FreshPairedTrialError(f"{path.name} must contain one JSON object")
    return value


def build_config() -> ProgressAwareOpenRouterConfig:
    """Return the exact three-call StreamLake queue configuration."""

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
        app_title="AgentEvolve Airfoil-v7 v5 paired causal trial",
        reasoning_config=OpenRouterReasoningConfig(effort="high"),
        retry_mode=ProgressAwareRetryMode.TRANSPORT_ONLY,
    )


def score_resolution_binding() -> AllocationScoreResolutionBinding:
    return AllocationScoreResolutionBinding(
        policy_id=V3_RESOLUTION_POLICY_ID,
        policy_version=V3_RESOLUTION_POLICY_VERSION,
        policy_definition_sha256=V3_RESOLUTION_POLICY_DEFINITION_SHA256,
        maximum_indistinguishable_score_gap=(
            V3_MAXIMUM_INDISTINGUISHABLE_SCORE_GAP
        ),
    )


def allocation_unit_key(
    assignment: AirfoilV7PairedBlockAssignment,
) -> str:
    """Return one task-keyed unit shared by every treatment and method."""

    selected = assignment.selected_block
    return (
        f"airfoil-v7:{TASK_SHA256}:contract:"
        f"{assignment.contract.identity_sha256}:layout:"
        f"{assignment.layout.layout_sha256}:block:{selected.block_spec_sha256}:"
        f"mask:{assignment.schedule.request.eligible_mask.mask_sha256}:k3"
    )


def tie_selection_binding(
    assignment: AirfoilV7PairedBlockAssignment,
) -> AllocationV3SelectionBinding:
    return AllocationV3SelectionBinding(
        policy_id=V3_SELECTION_POLICY_ID,
        policy_version=V3_SELECTION_POLICY_VERSION,
        policy_definition_sha256=V3_SELECTION_POLICY_DEFINITION_SHA256,
        mode=AllocationV3TieMode.PUBLIC_HASH_RANK,
        seed_sampling_law=AllocationV3SeedSamplingLaw.FIXED_PUBLIC,
        seed_provenance_sha256=(
            assignment.schedule.request.public_seed_source_sha256
        ),
        public_seed=PUBLIC_SEED,
        allocation_unit_key=allocation_unit_key(assignment),
    )


def member_quality(*, delta_f: float, delta_v: float) -> float:
    """Prospectively frozen actual-outcome member quality."""

    for name, value in (("delta_f", delta_f), ("delta_v", delta_v)):
        if type(value) is not float or not math.isfinite(value):
            raise TypeError(f"{name} must be a finite canonical float")
    return max(-60.0, min(60.0, -delta_v / 0.005)) + 0.05 * math.tanh(
        -delta_f / 0.001
    )


def member_log_failure(*, delta_f: float, delta_v: float) -> float:
    """Compute log(sigmoid(-quality)) without subtractive cancellation."""

    quality = member_quality(delta_f=delta_f, delta_v=delta_v)
    if quality >= 0.0:
        return -quality - math.log1p(math.exp(-quality))
    return -math.log1p(math.exp(quality))


def exact_three_set_rank(
    *,
    selected_option_ids: tuple[str, str, str],
    eligible_option_ids: tuple[str, ...],
    metric_deltas: Mapping[str, tuple[float, float]],
) -> tuple[int, float]:
    """Rank a selected three-set among all exact eligible three-sets."""

    if (
        type(selected_option_ids) is not tuple
        or len(selected_option_ids) != PORTFOLIO_SIZE
        or len(set(selected_option_ids)) != PORTFOLIO_SIZE
    ):
        raise ValueError("selected_option_ids must be one unique exact three-set")
    if (
        type(eligible_option_ids) is not tuple
        or len(eligible_option_ids) != 19
        or len(set(eligible_option_ids)) != 19
    ):
        raise ValueError("eligible_option_ids must be the exact 19-action block")
    if not set(selected_option_ids).issubset(eligible_option_ids):
        raise ValueError("selected three-set escapes the eligible block")
    if set(metric_deltas) != set(eligible_option_ids):
        raise ValueError("metric_deltas must cover all 19 eligible actions exactly")

    def endpoint(option_ids: tuple[str, ...]) -> float:
        return float(
            sum(
                member_log_failure(
                    delta_f=metric_deltas[value][0],
                    delta_v=metric_deltas[value][1],
                )
                for value in option_ids
            )
        )

    selected = endpoint(tuple(sorted(selected_option_ids)))
    values = tuple(
        endpoint(tuple(value))
        for value in itertools.combinations(sorted(eligible_option_ids), 3)
    )
    if len(values) != 969:
        raise AssertionError("19 choose 3 changed")
    return 1 + sum(value < selected for value in values), selected


def primary_endpoint_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "endpoint_id": PRIMARY_ENDPOINT_ID,
        "endpoint_version": PRIMARY_ENDPOINT_VERSION,
        "definition_sha256": PRIMARY_ENDPOINT_DEFINITION_SHA256,
        "member_quality": (
            "clip(-delta_v/0.005,-60,60)+0.05*tanh(-delta_f/0.001)"
        ),
        "member_log_failure": "log(sigmoid(-quality))_stable_softplus_form",
        "portfolio_endpoint": "selected_three_set_sum_member_log_failure",
        "direction": "lower_is_better",
        "observable_authority": {
            "raw_outcomes": "committed_selected_union_only",
            "portfolio_size": 3,
            "requires_unselected_raw_outcomes": False,
        },
        "primary_contrasts": {
            "within_method": [
                "endpoint_p_minus_endpoint_m",
                "endpoint_n_minus_endpoint_m",
            ],
            "paired_method": [
                "endpoint_m_v2_minus_v3",
                "endpoint_p_v2_minus_v3",
                "endpoint_n_v2_minus_v3",
            ],
            "positive_difference_favors_m_or_v3": True,
        },
        "secondary_endpoints": {
            "exact_three_set_competition_rank": {
                "eligible_action_count": 19,
                "portfolio_size": 3,
                "exact_three_set_count": 969,
                "rank": (
                    "1+count(other_exact_three_sets_with_strictly_lower_endpoint)"
                ),
                "tie_policy": "competition_rank",
                "status": (
                    "not_computed_without_separate_postcommit_reference_release"
                ),
            },
            "additional": ["best_member_rank", "forecast_calibration"],
        },
        "claim_boundary": {
            "developmental_blocks": 1,
            "significance_claim": False,
            "efficacy_generalization": False,
        },
        "conditional_postcommit_rank_authority": {
            "required_for_primary_endpoint": False,
            "eligible_only_after_both_allocation_commits_and_selected_union_completion": (
                True
            ),
            "frozen_eligible_mask_action_count": 19,
            "reference_cached_reads": 19,
            "raw_unselected_outcomes_returned": False,
            "returns": ["selected_set_ranks", "denominator_count"],
            "evaluator_wall_clock_claim": False,
            "status": (
                "not_computed_without_separate_postcommit_reference_release"
            ),
        },
    }


def _source_paths() -> tuple[Path, ...]:
    paths = set(historical._source_paths())
    paths.add(Path(__file__))
    paths.add(Path(sealed_replay.__file__))
    paths.add(
        Path(sys.modules[prepare_airfoil_v7_paired_block_assignment.__module__].__file__)
    )
    paths.add(Path(sys.modules[__name__].__file__))
    test_path = AGENT_EVOLVE_ROOT / "tests" / "test_run_airfoil_v7_v5_paired_causal_trial.py"
    if test_path.is_file():
        paths.add(test_path)
    return tuple(sorted(paths, key=lambda value: value.resolve().as_posix()))


def current_source_identity() -> dict[str, object]:
    return source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)


def _evaluation_from_record(value: object) -> AirfoilDevelopmentEvaluation:
    if type(value) is not dict or value.get("valid") is not True:
        raise FreshPairedTrialError("historical G1 evaluation is malformed")
    raw_metrics = value.get("metrics")
    if type(raw_metrics) is not list:
        raise FreshPairedTrialError("historical G1 metrics are malformed")
    metrics = tuple(
        AirfoilObservedMetric(
            metric_id=str(metric["metric_id"]),
            parent_value=float.fromhex(str(metric["parent_value_hex"])),
            child_value=float.fromhex(str(metric["child_value_hex"])),
        )
        for metric in raw_metrics
        if type(metric) is dict
    )
    if len(metrics) != len(raw_metrics):
        raise FreshPairedTrialError("one historical G1 metric is malformed")
    result = AirfoilDevelopmentEvaluation(
        option_id=str(value["option_id"]),
        option_identity_sha256=str(value["option_identity_sha256"]),
        child_configuration_sha256=str(value["child_configuration_sha256"]),
        family=str(value["family"]),
        metrics=metrics,
        terminal_record_sha256=str(value["terminal_record_sha256"]),
        raw_receipt_sha256=str(value["raw_receipt_sha256"]),
        active_wall_seconds=float.fromhex(str(value["active_wall_seconds_hex"])),
        outer_wall_seconds=float.fromhex(str(value["outer_wall_seconds_hex"])),
    )
    if result.to_record() != value:
        raise FreshPairedTrialError("historical G1 evaluation receipt changed")
    return result


def _historical_preparation_from_artifact(
    *,
    frozen_v2_run: Path,
    oracle_dir: Path,
    schedule_assignment: AirfoilV7PairedBlockAssignment,
) -> PreparedAirfoilTwoStageGeneration:
    """Rehydrate exact historical G1 evidence without opening oracle files."""

    readiness = _load_object(frozen_v2_run / "provider_free_readiness.json")
    raw = readiness.get("preparation")
    if type(raw) is not dict:
        raise FreshPairedTrialError("frozen readiness lacks its preparation")
    contract = schedule_assignment.contract
    raw_observations = raw.get("g1_observations")
    if type(raw_observations) is not list or len(raw_observations) != 8:
        raise FreshPairedTrialError("frozen readiness lacks exact G1 observations")
    observations: list[AirfoilG1ActionObservation] = []
    for row in raw_observations:
        if type(row) is not dict:
            raise FreshPairedTrialError("historical G1 observation is malformed")
        evaluation = _evaluation_from_record(row.get("evaluation"))
        binding = bind_finite_action_evidence(
            contrast_id=str(row["contrast_id"]),
            contract=contract,
            option_id=evaluation.option_id,
        )
        if binding.to_record() != row.get("finite_action_evidence"):
            raise FreshPairedTrialError("historical G1 action binding changed")
        observation = AirfoilG1ActionObservation(
            diagnostic_rank=int(row["diagnostic_rank"]),
            operator_invocation_id=OperatorInvocationId(
                str(row["operator_invocation_id"])
            ),
            parent_candidate_id=CandidateId(str(row["parent_candidate_id"])),
            child_candidate_id=CandidateId(str(row["child_candidate_id"])),
            option_id=evaluation.option_id,
            family=evaluation.family,
            option_identity_sha256=evaluation.option_identity_sha256,
            child_configuration_sha256=evaluation.child_configuration_sha256,
            evaluation=evaluation,
            contrast_id=str(row["contrast_id"]),
            action_binding=binding,
        )
        if observation.to_record() != row:
            raise FreshPairedTrialError("historical G1 observation changed")
        observations.append(observation)

    seal = raw.get("oracle_seal")
    if type(seal) is not dict:
        raise FreshPairedTrialError("historical oracle seal is malformed")
    synthetic_oracle = VerifiedAirfoilPredecisionOracle(
        run_dir=oracle_dir.expanduser().resolve(strict=False),
        contract=contract,
        run_id=str(seal["run_id"]),
        manifest_sha256=str(seal["manifest_sha256"]),
        source_sha256=str(seal["source_sha256"]),
        recursive_content_sha256=str(seal["recursive_content_sha256"]),
        recursive_file_count=int(seal["recursive_file_count"]),
        oracle_result_file_sha256=str(seal["oracle_result_file_sha256"]),
        file_bindings={},
    )
    evaluator = airfoil.AirfoilV7SealedOracleDevelopmentEvaluator(
        synthetic_oracle
    )
    evaluator.authorize_initial_g1(schedule_assignment.g1_sample)
    frozen_seal = freeze_json(seal)
    if type(frozen_seal) is not FrozenJsonObject:
        raise FreshPairedTrialError("historical oracle seal did not freeze")
    sampled = {member.option_id for member in schedule_assignment.g1_sample.members}
    preparation = PreparedAirfoilTwoStageGeneration(
        contract=contract,
        sample=schedule_assignment.g1_sample,
        observations=tuple(observations),
        reflection_request=airfoil._build_reflection_request(
            contract=contract,
            observations=tuple(observations),
        ),
        parent_metric_values=airfoil._parent_metric_values(),
        metric_scales=airfoil._metric_scales(),
        eligible_g2_option_ids=tuple(
            sorted(
                option.option_id
                for option in contract.options
                if option.option_id not in sampled
            )
        ),
        oracle_seal=frozen_seal,
        evaluator=evaluator,
    )
    if preparation.to_record() != raw:
        raise FreshPairedTrialError(
            "artifact-rehydrated preparation differs from frozen v2"
        )
    return preparation


@dataclass(frozen=True, slots=True)
class OpaquePlannedCall:
    provider_slot_assignment: TreatmentAssignment
    block_request: ActionForecastBlockRequest = field(repr=False, compare=False)
    plan: StructuredGenerationRequest[Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.provider_slot_assignment) is not TreatmentAssignment:
            raise TypeError("provider_slot_assignment must be exact")
        self.provider_slot_assignment.__post_init__()
        if type(self.block_request) is not ActionForecastBlockRequest:
            raise TypeError("block_request must be exact")
        self.block_request.__post_init__()
        self.plan.__post_init__()
        if (
            self.provider_slot_assignment.call_identity
            != self.block_request.block_call_id.value
            or self.provider_slot_assignment.request_identity_sha256
            != self.block_request.block_request_sha256
            or self.plan.call_id != self.block_request.block_call_id
        ):
            raise ValueError("opaque provider assignment differs from its exact call")


@dataclass(frozen=True, slots=True)
class TrialBundle:
    frozen_v2_run: Path
    oracle_dir: Path
    source_v2_finalization: Mapping[str, object]
    preparation: PreparedAirfoilTwoStageGeneration
    arms: AirfoilTwoStageForecastArms
    schedule_assignment: AirfoilV7PairedBlockAssignment
    layout: ActionForecastPartitionLayout
    selected_block_requests: tuple[ActionForecastBlockRequest, ...]
    planned_calls: tuple[StructuredGenerationRequest[Any], ...]
    experiment_record: Mapping[str, object]
    treatment_assignment: ProspectiveTreatmentAssignmentReceipt
    opaque_calls: tuple[OpaquePlannedCall, ...]

    def __post_init__(self) -> None:
        if self.schedule_assignment.selected_block.block_index != 2:
            raise ValueError("fresh trial must use prospective replicate-0 block 2")
        if len(self.schedule_assignment.eligible_global_row_indices) != 19:
            raise ValueError("fresh trial must have exactly 19 eligible G2 rows")
        if len(self.selected_block_requests) != 3 or len(self.planned_calls) != 3:
            raise ValueError("fresh trial requires exactly three scientific calls")
        self.treatment_assignment.__post_init__()
        if len(self.opaque_calls) != 3:
            raise ValueError("fresh trial requires exactly three opaque calls")
        for value in self.opaque_calls:
            value.__post_init__()

    def arm_for_occurrence(self, occurrence_id: str) -> str:
        for occurrence in self.treatment_assignment.occurrence_input_order:
            if occurrence.occurrence_id.value == occurrence_id:
                return occurrence.treatment_id.value
        raise KeyError(occurrence_id)


def _experiment_record(
    *,
    source_finalization: Mapping[str, object],
    preparation: PreparedAirfoilTwoStageGeneration,
    schedule_assignment: AirfoilV7PairedBlockAssignment,
    layout: ActionForecastPartitionLayout,
    block_requests: tuple[ActionForecastBlockRequest, ...],
    plans: tuple[StructuredGenerationRequest[Any], ...],
) -> dict[str, object]:
    health = lenient_action_forecast_health_policy()
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": "airfoil_v7_v5_paired_causal_trial",
        "developmental_scope": "one_prospectively_selected_block",
        "source_v2_finalization_sha256": source_finalization[
            "finalization_sha256"
        ],
        "source_v2_recursive_content_sha256": source_finalization[
            "recursive_content_sha256"
        ],
        "historical_evidence_reuse": {
            "g1_observation_count": 8,
            "reflection_logical_call_count": 1,
            "new_reflection_calls": 0,
            "cards_rebuilt_from_frozen_receipts": True,
            "oracle_outcome_files_read_during_prepare": 0,
        },
        "finite_contract_identity_sha256": preparation.contract.identity_sha256,
        "paired_block_assignment": schedule_assignment.to_record(),
        "partition_layout_sha256": layout.layout_sha256,
        "selected_block_request_sha256s": [
            value.block_request_sha256 for value in block_requests
        ],
        "planned_call_contracts": [
            historical._request_plan_payload(value)["call_contract"]
            for value in plans
        ],
        "forecast_policy": {
            "policy_id": PydanticAIActionForecastBlockPolicy.policy_id,
            "policy_version": ACTION_FORECAST_POLICY_VERSION,
            "policy_definition_sha256": (
                ACTION_FORECAST_POLICY_DEFINITION_SHA256
            ),
            "provider_wire_version": 5,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "health_policy": health.to_record(),
        "allocation_methods": {
            "common_frame": {
                "subset_policy": (
                    sealed_replay._allocation_subset_policy().to_record()
                ),
                "health_subset_policy": (
                    historical.eligible_subset_policy().to_record()
                ),
                "portfolio_size": PORTFOLIO_SIZE,
                "eligible_action_count": 19,
                "candidate_score_count_per_arm": 19 + 18 + 17,
            },
            "v2": {
                "risk_aversion_hex": airfoil.ALLOCATOR_RISK_AVERSION.hex(),
                "diversity_weight_hex": airfoil.ALLOCATOR_DIVERSITY_WEIGHT.hex(),
                "method_kind": "audited_frame_v2",
                "durable_phase_commit_required": True,
                "score_diagnostic": {
                    "policy_id": sealed_replay.ALLOCATION_DIAGNOSTIC_POLICY_ID,
                    "policy_version": (
                        sealed_replay.ALLOCATION_DIAGNOSTIC_POLICY_VERSION
                    ),
                    "policy_definition_sha256": (
                        sealed_replay.ALLOCATION_DIAGNOSTIC_POLICY_DEFINITION_SHA256
                    ),
                },
                "surface_gate": (
                    sealed_replay._allocation_gate_policy().to_record()
                ),
            },
            "v3": {
                "risk_aversion_hex": V3_RISK_AVERSION.hex(),
                "diversity_weight_hex": V3_DIVERSITY_WEIGHT.hex(),
                "score_resolution": score_resolution_binding().to_record(),
                "tie_selection": tie_selection_binding(
                    schedule_assignment
                ).to_record(),
                "portfolio_size": PORTFOLIO_SIZE,
            },
        },
        "primary_endpoint": primary_endpoint_record(),
        "execution": {
            "logical_provider_calls": 3,
            "concurrent_calls": 3,
            "max_pending": 3,
            "max_transport_only_attempts_per_call": 2,
            "settle_all_before_terminal_ledger": True,
            "terminal_ledger_fsync_before_allocation": True,
            "schema_repair": False,
            "logical_rerun": False,
        },
        "claim_boundary": {
            "significance_claim": False,
            "efficacy_generalization": False,
        },
    }
    record["experiment_commitment_sha256"] = _hash(
        _EXPERIMENT_FRAMING,
        record,
    )
    return record


def _treatment_assignment(
    *,
    experiment_commitment_sha256: str,
    block_requests: tuple[ActionForecastBlockRequest, ...],
) -> ProspectiveTreatmentAssignmentReceipt:
    occurrences = tuple(
        TreatmentOccurrence(
            occurrence_id=TreatmentOccurrenceId(f"paired.forecast.{index:02d}"),
            treatment_id=TreatmentId(arm),
            call_identity=request.block_call_id.value,
            request_identity_sha256=request.block_request_sha256,
        )
        for index, (arm, request) in enumerate(
            zip(("m", "p", "n"), block_requests, strict=True)
        )
    )
    return assign_treatment_occurrences(
        TreatmentAssignmentInput(
            experiment_commitment_sha256=experiment_commitment_sha256,
            public_seed_material=ASSIGNMENT_PUBLIC_SEED,
            occurrences=occurrences,
            provider_slot_ids=tuple(
                OpaqueProviderSlotId(f"opaque.paired.slot.{index:02d}")
                for index in range(3)
            ),
        )
    )


def build_trial_bundle(
    *,
    frozen_v2_run: Path = DEFAULT_FROZEN_V2_RUN,
    oracle_dir: Path = DEFAULT_ORACLE_DIR,
    schedule_assignment: AirfoilV7PairedBlockAssignment | None = None,
) -> TrialBundle:
    """Rebuild all exact scientific/provider inputs without reading a credential."""

    root = frozen_v2_run.expanduser().resolve(strict=True)
    source_finalization, _manifest = historical._verify_v2_run(root)
    if schedule_assignment is None:
        schedule_assignment = prepare_airfoil_v7_paired_block_assignment(
            replicate_index=0
        )
    elif type(schedule_assignment) is not AirfoilV7PairedBlockAssignment:
        raise TypeError("schedule_assignment must be exact or None")
    schedule_assignment.__post_init__()
    preparation = _historical_preparation_from_artifact(
        frozen_v2_run=root,
        oracle_dir=oracle_dir,
        schedule_assignment=schedule_assignment,
    )
    expected_contrasts = tuple(
        sorted(value.contrast_id for value in preparation.observations)
    )
    reflection = historical.reflection_from_record(
        _load_object(root / "reflection_result.json"),
        expected_contrast_ids=expected_contrasts,
    )
    arms = airfoil.build_airfoil_v7_forecast_arms(preparation, reflection)
    historical._validate_rebuilt_arms(
        arms,
        _load_object(root / "cards_views_requests.json"),
    )
    requests = (
        arms.memory_request,
        arms.placebo_request,
        arms.catalog_only_request,
    )
    layouts = tuple(
        build_action_forecast_partition_layout(
            request,
            historical.partition_policy(),
        )
        for request in requests
    )
    if any(value.to_record() != layouts[0].to_record() for value in layouts[1:]):
        raise FreshPairedTrialError("M/P/N partition layouts differ")
    layout = layouts[0]
    if layout.layout_sha256 != schedule_assignment.layout.layout_sha256:
        raise FreshPairedTrialError("generic schedule names another partition layout")
    selected_index = schedule_assignment.selected_block.block_index
    block_requests = tuple(
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
    plans = tuple(plan_action_forecast_block_request(value) for value in block_requests)
    if any(
        value.output_type.__name__ != "AllOptionActionForecastMatrixV5"
        or value.output_tool_name != ACTION_FORECAST_BLOCK_TOOL_NAME
        or value.max_output_tokens != MAX_OUTPUT_TOKENS
        for value in plans
    ):
        raise FreshPairedTrialError("fresh provider plan is not the exact v5 wire")
    historical._validate_planned_wire_contracts(plans)
    v2_launcher._validate_provider_boundary_blinding(plans)
    experiment = _experiment_record(
        source_finalization=source_finalization,
        preparation=preparation,
        schedule_assignment=schedule_assignment,
        layout=layout,
        block_requests=block_requests,
        plans=plans,
    )
    assignment = _treatment_assignment(
        experiment_commitment_sha256=str(
            experiment["experiment_commitment_sha256"]
        ),
        block_requests=block_requests,
    )
    opaque_calls = tuple(
        OpaquePlannedCall(
            provider_slot_assignment=value,
            block_request=block_requests[value.occurrence_input_index],
            plan=plans[value.occurrence_input_index],
        )
        for value in assignment.slot_to_occurrence
    )
    bundle = TrialBundle(
        frozen_v2_run=root,
        oracle_dir=oracle_dir.expanduser().resolve(strict=False),
        source_v2_finalization=source_finalization,
        preparation=preparation,
        arms=arms,
        schedule_assignment=schedule_assignment,
        layout=layout,
        selected_block_requests=block_requests,
        planned_calls=plans,
        experiment_record=experiment,
        treatment_assignment=assignment,
        opaque_calls=opaque_calls,
    )
    bundle.__post_init__()
    return bundle


def _exclusive_directory(path: Path) -> Path:
    target = path.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    return target


def _historical_evidence_binding_record(bundle: TrialBundle) -> dict[str, object]:
    files = bundle.source_v2_finalization.get("files")
    if type(files) is not dict:
        raise FreshPairedTrialError("frozen v2 finalization lacks file bindings")
    return {
        "schema_version": 1,
        "source_finalization_sha256": bundle.source_v2_finalization[
            "finalization_sha256"
        ],
        "source_recursive_content_sha256": bundle.source_v2_finalization[
            "recursive_content_sha256"
        ],
        "provider_free_readiness_file": files["provider_free_readiness.json"],
        "reflection_result_file": files["reflection_result.json"],
        "cards_views_requests_file": files["cards_views_requests.json"],
        "historical_g1_observation_count": 8,
        "historical_reflection_logical_call_count": 1,
        "oracle_outcome_file_reads_during_prepare": 0,
        "synthetic_evaluator_boundary": {
            "purpose": "outcome_free_request_reconstruction_only",
            "file_bindings_empty": True,
            "selected_union_authority": False,
            "rank_authority": False,
            "live_postcommit_must_reverify_real_sealed_oracle": True,
        },
    }


def _planned_opaque_wave_record(bundle: TrialBundle) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "durably_precommitted_before_live_credential_read",
        "assignment_receipt_sha256": bundle.treatment_assignment.receipt_sha256,
        "calls_in_opaque_provider_slot_order": [
            {
                "opaque_provider_slot_id": (
                    value.provider_slot_assignment.opaque_provider_slot_id.value
                ),
                "occurrence_id": value.provider_slot_assignment.occurrence_id.value,
                "call_id": value.block_request.block_call_id.value,
                "block_request_sha256": value.block_request.block_request_sha256,
                "exact_provider_request": historical._request_plan_payload(
                    value.plan
                ),
            }
            for value in bundle.opaque_calls
        ],
        "provider_visible_treatment_labels": False,
        "content_blinding_claimed": False,
    }


def _protocol_record(
    bundle: TrialBundle,
    *,
    target_live_run_dir: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "airfoil_v7_v5_paired_causal_trial_preparation",
        "authorized_target_live_run_dir": str(target_live_run_dir),
        "experiment": dict(bundle.experiment_record),
        "route": v2_launcher.route_binding(),
        "queue": build_config().to_manifest_record(),
        "paired_block_assignment_receipt_sha256": (
            bundle.schedule_assignment.assignment_receipt_sha256
        ),
        "generic_full_schedule_sha256": (
            bundle.schedule_assignment.schedule.full_schedule_sha256
        ),
        "paired_allocation_schedule_binding_sha256": (
            bundle.schedule_assignment.schedule.selected_block_receipt_sha256
        ),
        "treatment_assignment_receipt_sha256": (
            bundle.treatment_assignment.receipt_sha256
        ),
        "health_policy": lenient_action_forecast_health_policy().to_record(),
        "eligible_subset_policy": historical.eligible_subset_policy().to_record(),
        "primary_endpoint": primary_endpoint_record(),
        "chronology": [
            "paired_block_assignment_release_fsync",
            "historical_g1_reflection_card_reconstruction",
            "exact_v5_provider_wave_precommit_fsync",
            "preparation_recursive_finalization",
            "live_claim_before_credential_read",
            "three_opaque_calls_settle",
            "terminal_provider_ledger_fsync",
            "paired_v2_v3_allocation_commits",
            "selected_union_evaluation",
        ],
        "conditional_postcommit_extension": {
            "step": "rank_only_reference_authority",
            "required_for_primary_endpoint": False,
            "status": (
                "not_computed_without_separate_postcommit_reference_release"
            ),
        },
        "claim_boundary": {
            "developmental_blocks": 1,
            "significance_claim": False,
            "efficacy_generalization": False,
        },
    }


def execute_prepare(
    *,
    run_dir: Path,
    target_live_run_dir: Path,
    frozen_v2_run: Path = DEFAULT_FROZEN_V2_RUN,
    oracle_dir: Path = DEFAULT_ORACLE_DIR,
) -> dict[str, object]:
    """Create a finalized preparation with block release preceding G1 reuse."""

    target = target_live_run_dir.expanduser().resolve(strict=False)
    preparation_target = run_dir.expanduser().resolve(strict=False)
    if target == preparation_target:
        raise FreshPairedTrialError("prepared and live directories must differ")
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
                "oracle_outcome_file_read_attempted": False,
            },
        )

        # This durable release must happen before any artifact carrying G1
        # outcome evidence is decoded.  It is the chronology proof that block
        # assignment was method- and outcome-independent.
        schedule_assignment = prepare_airfoil_v7_paired_block_assignment(
            replicate_index=0
        )
        assignment_release = {
            "schema_version": 1,
            "status": "released_before_historical_g1_artifact_decode",
            "assignment": schedule_assignment.to_record(),
            "selected_block_index": schedule_assignment.selected_block.block_index,
            "eligible_global_row_indices": list(
                schedule_assignment.eligible_global_row_indices
            ),
            "oracle_outcome_file_reads_before_release": 0,
            "credentials_read_before_release": False,
            "provider_calls_before_release": 0,
        }
        write_json_atomic(
            root / "paired_block_assignment_release.json",
            assignment_release,
        )
        if _load_object(root / "paired_block_assignment_release.json") != (
            assignment_release
        ):
            raise FreshPairedTrialError("paired block assignment read-back changed")

        bundle = build_trial_bundle(
            frozen_v2_run=frozen_v2_run,
            oracle_dir=oracle_dir,
            schedule_assignment=schedule_assignment,
        )
        source = current_source_identity()
        protocol = _protocol_record(bundle, target_live_run_dir=target)
        evidence = _historical_evidence_binding_record(bundle)
        wave = _planned_opaque_wave_record(bundle)
        write_json_atomic(root / "protocol.json", protocol)
        write_json_atomic(root / "historical_evidence_binding.json", evidence)
        write_json_atomic(
            root / "treatment_assignment.json",
            bundle.treatment_assignment.to_record(),
        )
        write_json_atomic(root / "primary_endpoint.json", primary_endpoint_record())
        write_json_atomic(root / "planned_opaque_wave.json", wave)

        record: dict[str, object] = {
            "schema_version": 1,
            "status": "prepared",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(root),
            "authorized_target_live_run_dir": str(target),
            "frozen_v2_run": str(bundle.frozen_v2_run),
            "oracle_dir_for_postcommit_reverification": str(bundle.oracle_dir),
            "closed_source_identity": source,
            "runtime_identity": v2_launcher.runtime_identity(),
            "assignment_release_file": file_identity(
                root / "paired_block_assignment_release.json",
                relative_to=root,
            ),
            "protocol_file": file_identity(root / "protocol.json", relative_to=root),
            "historical_evidence_file": file_identity(
                root / "historical_evidence_binding.json",
                relative_to=root,
            ),
            "treatment_assignment_file": file_identity(
                root / "treatment_assignment.json",
                relative_to=root,
            ),
            "primary_endpoint_file": file_identity(
                root / "primary_endpoint.json",
                relative_to=root,
            ),
            "planned_wave_file": file_identity(
                root / "planned_opaque_wave.json",
                relative_to=root,
            ),
            "experiment_commitment_sha256": bundle.experiment_record[
                "experiment_commitment_sha256"
            ],
            "credential_read_attempted": False,
            "provider_client_constructed": False,
            "provider_call_attempted": False,
            "oracle_outcome_file_reads": 0,
            "historical_g1_artifact_observations_rehydrated": 8,
            "new_candidate_evaluations": 0,
        }
        record["preparation_commitment_sha256"] = _hash(
            _PREPARED_FRAMING,
            record,
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
                    "oracle_outcome_file_reads": 0,
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
    bundle: TrialBundle
    wave: Mapping[str, object]


def verify_prepared(run_dir: Path) -> VerifiedPreparation:
    root = run_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    if finalization.get("status") != "prepared":
        raise FreshPairedTrialError("paired preparation is not finalized as prepared")
    record = _load_object(root / "prepared.json")
    unsigned = dict(record)
    commitment = unsigned.pop("preparation_commitment_sha256", None)
    if commitment != _hash(_PREPARED_FRAMING, unsigned):
        raise FreshPairedTrialError("paired preparation commitment changed")
    if record.get("closed_source_identity") != current_source_identity():
        raise FreshPairedTrialError("closed source changed after preparation")
    if record.get("runtime_identity") != v2_launcher.runtime_identity():
        raise FreshPairedTrialError("runtime changed after preparation")
    target_value = record.get("authorized_target_live_run_dir")
    if type(target_value) is not str:
        raise FreshPairedTrialError("authorized live target is malformed")
    target = Path(target_value)
    if (
        not target.is_absolute()
        or str(target.expanduser().resolve(strict=False)) != target_value
        or target == root
    ):
        raise FreshPairedTrialError("authorized live target is not canonical")
    frozen_value = record.get("frozen_v2_run")
    oracle_value = record.get("oracle_dir_for_postcommit_reverification")
    if type(frozen_value) is not str or type(oracle_value) is not str:
        raise FreshPairedTrialError("prepared input paths are malformed")
    schedule_assignment = prepare_airfoil_v7_paired_block_assignment(
        replicate_index=0
    )
    expected_release = {
        "schema_version": 1,
        "status": "released_before_historical_g1_artifact_decode",
        "assignment": schedule_assignment.to_record(),
        "selected_block_index": schedule_assignment.selected_block.block_index,
        "eligible_global_row_indices": list(
            schedule_assignment.eligible_global_row_indices
        ),
        "oracle_outcome_file_reads_before_release": 0,
        "credentials_read_before_release": False,
        "provider_calls_before_release": 0,
    }
    if _load_object(root / "paired_block_assignment_release.json") != expected_release:
        raise FreshPairedTrialError("paired block release changed")
    bundle = build_trial_bundle(
        frozen_v2_run=Path(frozen_value),
        oracle_dir=Path(oracle_value),
        schedule_assignment=schedule_assignment,
    )
    wave = _load_object(root / "planned_opaque_wave.json")
    if wave != _planned_opaque_wave_record(bundle):
        raise FreshPairedTrialError("prepared opaque wave changed")
    if _load_object(root / "protocol.json") != _protocol_record(
        bundle,
        target_live_run_dir=target,
    ):
        raise FreshPairedTrialError("prepared protocol changed")
    if _load_object(root / "historical_evidence_binding.json") != (
        _historical_evidence_binding_record(bundle)
    ):
        raise FreshPairedTrialError("historical evidence binding changed")
    if _load_object(root / "treatment_assignment.json") != (
        bundle.treatment_assignment.to_record()
    ):
        raise FreshPairedTrialError("treatment assignment changed")
    if _load_object(root / "primary_endpoint.json") != primary_endpoint_record():
        raise FreshPairedTrialError("primary endpoint changed")
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
    """Verify and claim the exact one-shot live directory before credential read."""

    prepared = verify_prepared(prepared_dir)
    requested = run_dir.expanduser().resolve(strict=False)
    if prepared.record.get("authorized_target_live_run_dir") != str(requested):
        raise FreshPairedTrialError("live target differs from one-shot preparation")
    root = _exclusive_directory(requested)
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
        "experiment_commitment_sha256": prepared.bundle.experiment_record[
            "experiment_commitment_sha256"
        ],
        "closed_source_identity": prepared.record["closed_source_identity"],
        "runtime_identity": prepared.record["runtime_identity"],
        "credential_read_attempted": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "oracle_outcome_file_reads": 0,
    }
    write_json_atomic(root / "precredential_claim.json", record)
    if _load_object(root / "precredential_claim.json") != record:
        raise FreshPairedTrialError("precredential live claim read-back changed")
    return LiveClaim(root, prepared, record)


def _durable_phase_commit_record(phase_commit: object) -> dict[str, object]:
    receipt = getattr(phase_commit, "receipt", None)
    payload = getattr(phase_commit, "payload", None)
    if receipt is None or payload is None:
        raise FreshPairedTrialError("allocation phase commit is malformed")
    return {
        "schema_version": 1,
        "receipt": receipt.to_record(),
        "payload": thaw_json(payload),
    }


@dataclass(frozen=True, slots=True)
class PostLedgerContext:
    """Narrow inverted boundary for problem-specific paired adjudication."""

    claim: LiveClaim = field(repr=False, compare=False)
    accepted: tuple[
        tuple[str, ActionForecastBlockRequest, ActionForecastBlockResult], ...
    ] = field(repr=False, compare=False)
    terminal_ledger: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.claim) is not LiveClaim or not self.claim.active:
            raise TypeError("post-ledger context requires an active exact claim")
        if (
            type(self.accepted) is not tuple
            or tuple(value[0] for value in self.accepted) != ("m", "p", "n")
            or any(
                type(value[1]) is not ActionForecastBlockRequest
                or type(value[2]) is not ActionForecastBlockResult
                for value in self.accepted
            )
        ):
            raise ValueError("accepted blocks must be exact canonical M/P/N")
        if self.terminal_ledger.get("status") != "all_three_calls_terminal":
            raise ValueError("post-ledger context lacks a complete terminal ledger")
        commitment = self.terminal_ledger.get("commitment_sha256")
        unsigned = dict(self.terminal_ledger)
        unsigned.pop("commitment_sha256", None)
        if commitment != _hash(_TERMINAL_LEDGER_FRAMING, unsigned):
            raise ValueError("terminal provider ledger commitment changed")
        bundle = self.claim.prepared.bundle
        if _load_object(self.claim.run_dir / "terminal_provider_ledger.json") != (
            dict(self.terminal_ledger)
        ):
            raise ValueError("context ledger differs from its durable read-back")
        if (
            self.terminal_ledger.get("experiment_commitment_sha256")
            != bundle.experiment_record["experiment_commitment_sha256"]
            or self.terminal_ledger.get("treatment_assignment_receipt_sha256")
            != bundle.treatment_assignment.receipt_sha256
            or self.terminal_ledger.get("terminal_outcome_count") != 3
            or self.terminal_ledger.get("successful_outcome_count") != 3
            or self.terminal_ledger.get("all_calls_settled_before_allocation")
            is not True
            or self.terminal_ledger.get(
                "allocation_started_before_ledger_fsync"
            )
            is not False
            or self.terminal_ledger.get(
                "oracle_outcome_files_read_before_ledger_fsync"
            )
            != 0
        ):
            raise ValueError("terminal ledger escaped its prepared experiment")

        slots = self.terminal_ledger.get("opaque_slots")
        if type(slots) is not list or len(slots) != 3:
            raise ValueError("terminal ledger lacks exact opaque slots")
        accepted_by_arm = {value[0]: value for value in self.accepted}
        for opaque, slot in zip(bundle.opaque_calls, slots, strict=True):
            if type(slot) is not dict:
                raise ValueError("one terminal opaque slot is malformed")
            assignment = opaque.provider_slot_assignment
            expected_occurrence = next(
                value
                for value in bundle.treatment_assignment.occurrence_input_order
                if value.occurrence_id == assignment.occurrence_id
            )
            if (
                assignment.call_identity != opaque.block_request.block_call_id.value
                or assignment.request_identity_sha256
                != opaque.block_request.block_request_sha256
                or expected_occurrence.call_identity != assignment.call_identity
                or expected_occurrence.request_identity_sha256
                != assignment.request_identity_sha256
                or slot.get("opaque_provider_slot_id")
                != assignment.opaque_provider_slot_id.value
                or slot.get("occurrence_id") != assignment.occurrence_id.value
                or slot.get("call_id") != assignment.call_identity
            ):
                raise ValueError("terminal slot differs from its opaque assignment")
            outcome = slot.get("terminal_outcome")
            if (
                type(outcome) is not dict
                or outcome.get("task_id") != assignment.call_identity
                or outcome.get("status") != "succeeded"
                or outcome.get("cancellation_reason") is not None
            ):
                raise ValueError("terminal slot is not the exact successful call")

            arm = expected_occurrence.treatment_id.value
            accepted_arm, observed_request, result = accepted_by_arm[arm]
            if accepted_arm != arm:
                raise AssertionError("accepted arm lookup changed")
            canonical_index = ("m", "p", "n").index(arm)
            expected_request = bundle.selected_block_requests[canonical_index]
            plan = bundle.planned_calls[canonical_index]
            if (
                observed_request.block_request_sha256
                != expected_request.block_request_sha256
                or observed_request.to_record() != expected_request.to_record()
                or plan.call_id.value != assignment.call_identity
            ):
                raise ValueError("accepted request differs from prepared canonical arm")
            result.__post_init__()
            validate_resolved_action_forecast_block(
                expected_request,
                result.forecasts,
            )
            if (
                result.forecasts.policy_id
                != PydanticAIActionForecastBlockPolicy.policy_id
                or result.forecasts.policy_version
                != ACTION_FORECAST_POLICY_VERSION
                or result.forecasts.policy_definition_sha256
                != ACTION_FORECAST_POLICY_DEFINITION_SHA256
            ):
                raise ValueError("accepted forecasts differ from the frozen v5 policy")
            _validate_route(result.telemetry)
            assert result.telemetry is not None

            plan_payload = historical._request_plan_payload(plan)
            typed_wire = _load_object(
                self.claim.run_dir / f"typed_wire_{assignment.call_identity}.json"
            )
            typed_payload = typed_wire.get("typed_code_matrices")
            if (
                typed_wire.get("opaque_provider_slot_id")
                != assignment.opaque_provider_slot_id.value
                or typed_wire.get("occurrence_id") != assignment.occurrence_id.value
                or typed_wire.get("call_id") != assignment.call_identity
                or typed_wire.get("prompt_sha256")
                != plan_payload["call_contract"]["prompt_sha256"]
                or typed_wire.get("schema_sha256")
                != plan_payload["call_contract"]["schema_sha256"]
                or typed_wire.get("typed_output_type")
                != plan.output_type.__name__
                or typed_wire.get("treatment_label_sent_as_metadata") is not False
                or type(typed_payload) is not dict
            ):
                raise ValueError("typed wire differs from the prepared opaque call")
            wire_value = plan.output_type.model_validate(typed_payload, strict=True)
            spec = expected_request.block
            options = expected_request.request.finite_variation_contract.options[
                spec.global_row_start : spec.global_row_stop
            ]
            drafts = action_forecast_adapter._drafts_from_wire(
                expected_request.request,
                wire_value,
                options,
                provider_wire_version=5,
            )
            replayed = resolve_action_forecast_block(
                expected_request,
                drafts,
                policy_id=PydanticAIActionForecastBlockPolicy.policy_id,
                policy_version=ACTION_FORECAST_POLICY_VERSION,
                policy_definition_sha256=ACTION_FORECAST_POLICY_DEFINITION_SHA256,
            )
            if replayed.to_record() != result.forecasts.to_record():
                raise ValueError("accepted forecasts differ from the durable v5 wire")

            resolved_record = {
                "schema_version": 1,
                "arm": arm,
                "block_request": expected_request.to_record(),
                "forecasts": result.forecasts.to_record(),
                "telemetry": _telemetry_record(result.telemetry),
            }
            if _load_object(
                self.claim.run_dir / f"resolved_block_{arm}.json"
            ) != resolved_record:
                raise ValueError("accepted result differs from its durable receipt")

            attempts = outcome.get("attempts")
            telemetry = result.telemetry
            if (
                type(attempts) is not list
                or len(attempts) != telemetry.attempt_count
                or not attempts
                or attempts[-1].get("status") != "succeeded"
            ):
                raise ValueError("terminal attempts differ from accepted telemetry")
            for attempt in attempts:
                if type(attempt) is not dict:
                    raise ValueError("one terminal attempt is malformed")
                evidence = attempt.get("request_evidence")
                if (
                    type(evidence) is not dict
                    or evidence.get("variant") != "original"
                    or evidence.get("prompt_sha256")
                    != plan_payload["call_contract"]["prompt_sha256"]
                    or type(evidence.get("provider_attempt_id")) is not str
                ):
                    raise ValueError("terminal attempt names another request")
            expected_response = _telemetry_record(telemetry)
            expected_response.pop("attempt_count")
            if outcome.get("response") != expected_response:
                raise ValueError("terminal response differs from accepted telemetry")


PairedAdjudicator = Callable[[PostLedgerContext], Mapping[str, object]]


def _primary_endpoint_from_evaluations(
    option_ids: tuple[str, str, str],
    evaluations: Mapping[str, AirfoilDevelopmentEvaluation],
) -> float:
    if len(set(option_ids)) != PORTFOLIO_SIZE:
        raise FreshPairedTrialError("primary endpoint requires one unique three-set")
    total = 0.0
    for option_id in option_ids:
        evaluation = evaluations.get(option_id)
        if evaluation is None:
            raise FreshPairedTrialError("selected union omitted a committed option")
        metric_by_id = {value.metric_id: value for value in evaluation.metrics}
        try:
            delta_f = metric_by_id[airfoil.OBJECTIVE_METRIC_ID].delta
            delta_v = metric_by_id[airfoil.VIOLATION_METRIC_ID].delta
        except KeyError as error:
            raise FreshPairedTrialError(
                "selected Airfoil evaluation lacks canonical metrics"
            ) from error
        total += member_log_failure(delta_f=delta_f, delta_v=delta_v)
    return float(total)


def _selected_endpoint_analysis(
    *,
    methods: tuple[AllocationComparisonMethodWave, ...],
    evaluations: Mapping[str, AirfoilDevelopmentEvaluation],
) -> dict[str, object]:
    by_method: dict[str, dict[str, float]] = {}
    selected_sets: dict[str, dict[str, list[str]]] = {}
    for method in methods:
        endpoints: dict[str, float] = {}
        selections: dict[str, list[str]] = {}
        for execution in method.executions:
            arm = execution.treatment_occurrence.treatment_id.value
            option_ids = tuple(
                value.option_id for value in execution.result.decision.members
            )
            if len(option_ids) != PORTFOLIO_SIZE:
                raise FreshPairedTrialError("one allocation changed portfolio size")
            exact_ids = (option_ids[0], option_ids[1], option_ids[2])
            endpoints[arm] = _primary_endpoint_from_evaluations(
                exact_ids,
                evaluations,
            )
            selections[arm] = list(option_ids)
        if set(endpoints) != {"m", "p", "n"}:
            raise FreshPairedTrialError("one method lacks exact M/P/N endpoints")
        by_method[method.comparison_method_id] = endpoints
        selected_sets[method.comparison_method_id] = selections

    if set(by_method) != {"audited_frame_v2", "operational_frame_v3"}:
        raise FreshPairedTrialError("paired endpoint names unexpected methods")
    v2 = by_method["audited_frame_v2"]
    v3 = by_method["operational_frame_v3"]

    def encoded(values: Mapping[str, float]) -> dict[str, object]:
        return {
            key: {"value": value, "value_hex": value.hex()}
            for key, value in sorted(values.items())
        }

    within = {
        method: {
            "endpoint_p_minus_endpoint_m": values["p"] - values["m"],
            "endpoint_n_minus_endpoint_m": values["n"] - values["m"],
        }
        for method, values in by_method.items()
    }
    paired = {
        f"endpoint_{arm}_v2_minus_v3": v2[arm] - v3[arm]
        for arm in ("m", "p", "n")
    }
    return {
        "schema_version": 1,
        "primary_endpoint": primary_endpoint_record(),
        "selected_option_ids": selected_sets,
        "endpoint_by_method_and_arm": {
            method: encoded(values) for method, values in sorted(by_method.items())
        },
        "within_method_contrasts_positive_favors_m": {
            method: encoded(values) for method, values in sorted(within.items())
        },
        "paired_v2_minus_v3_contrasts_positive_favors_v3": encoded(paired),
        "exact_three_set_competition_rank": {
            "status": (
                "not_computed_without_separate_postcommit_reference_release"
            ),
            "denominator_count": 969,
            "selected_set_ranks": None,
            "raw_unselected_outcomes_returned": False,
        },
    }


def adjudicate_airfoil_paired_allocations(
    context: PostLedgerContext,
) -> Mapping[str, object]:
    """Commit generic v2/v3 methods, then open one Airfoil union capability."""

    context.__post_init__()
    claim = context.claim
    bundle = claim.prepared.bundle
    root = claim.run_dir
    assignment = bundle.treatment_assignment
    eligible_rows = bundle.schedule_assignment.eligible_global_row_indices
    if len(eligible_rows) != 19:
        raise FreshPairedTrialError("paired allocation lost the exact 19-row subset")

    v2_replays = tuple(
        sealed_replay._build_arm_replay_with_utility(
            arm=arm,
            block_request=block_request,
            forecasts=result.forecasts,
            eligible_rows=eligible_rows,
            assignment=assignment,
            occurrence=occurrence,
            utility=bundle.preparation.utility,
        )
        for (arm, block_request, result), occurrence in zip(
            context.accepted,
            assignment.occurrence_input_order,
            strict=True,
        )
    )
    expected_candidate_scores = 19 + 18 + 17
    if any(
        replay.health_record.get("passes") is not True
        or replay.subset_health_record.get("passes") is not True
        or not replay.execution.result.audit.passes
        or replay.execution.result.audit.candidate_score_count
        != expected_candidate_scores
        for replay in v2_replays
    ):
        raise FreshPairedTrialError("audited allocator-v2 pre-outcome gate failed")
    v2_executions = tuple(value.execution for value in v2_replays)

    v3_allocator = OperationalGreedyForecastFrameAllocator()
    v3_executions_list: list[OperationalFrameActionAllocationTreatmentExecution] = []
    for replay, occurrence in zip(
        v2_replays,
        assignment.occurrence_input_order,
        strict=True,
    ):
        request = OperationalFrameActionAllocationRequest(
            allocation=replay.frame_request,
            risk_aversion=V3_RISK_AVERSION,
            diversity_weight=V3_DIVERSITY_WEIGHT,
            score_resolution=score_resolution_binding(),
            tie_selection=tie_selection_binding(bundle.schedule_assignment),
        )
        result = v3_allocator.allocate(request)
        if (
            not result.audit.passes
            or result.audit.candidate_score_count != expected_candidate_scores
        ):
            raise FreshPairedTrialError(
                "operational allocator-v3 pre-outcome gate failed"
            )
        v3_executions_list.append(
            OperationalFrameActionAllocationTreatmentExecution(
                treatment_assignment=assignment,
                treatment_occurrence=occurrence,
                request=request,
                result=result,
            )
        )
    v3_executions = tuple(v3_executions_list)

    write_json_atomic(
        root / "paired_allocation_attempts.json",
        {
            "schema_version": 1,
            "outcome_opened": False,
            "eligible_action_count_per_arm": 19,
            "portfolio_size": PORTFOLIO_SIZE,
            "candidate_scores_per_arm_per_method": expected_candidate_scores,
            "v2_arms": [sealed_replay._arm_record(value) for value in v2_replays],
            "v3_executions": [value.to_record() for value in v3_executions],
        },
    )

    upstream = str(bundle.experiment_record["experiment_commitment_sha256"])
    terminal = str(context.terminal_ledger["commitment_sha256"])
    v2_commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=upstream,
        terminal_provider_ledger_commitment_sha256=terminal,
        executions=v2_executions,
    )
    v2_record = _durable_phase_commit_record(v2_commit)
    write_json_atomic(root / "durable_allocation_v2_commit.json", v2_record)
    if _load_object(root / "durable_allocation_v2_commit.json") != v2_record:
        raise FreshPairedTrialError("v2 allocation commit read-back changed")

    v3_commit = build_operational_frame_action_allocation_phase_commit(
        upstream_input_sha256=upstream,
        terminal_provider_ledger_commitment_sha256=terminal,
        executions=v3_executions,
    )
    v3_record = _durable_phase_commit_record(v3_commit)
    write_json_atomic(root / "durable_allocation_v3_commit.json", v3_record)
    if _load_object(root / "durable_allocation_v3_commit.json") != v3_record:
        raise FreshPairedTrialError("v3 allocation commit read-back changed")

    schedule_binding = (
        bundle.schedule_assignment.schedule.selected_block_receipt_sha256
    )
    methods = (
        AllocationComparisonMethodWave(
            comparison_method_id="audited_frame_v2",
            schedule_binding_sha256=schedule_binding,
            executions=v2_executions,
            phase_commit=v2_commit,
        ),
        AllocationComparisonMethodWave(
            comparison_method_id="operational_frame_v3",
            schedule_binding_sha256=schedule_binding,
            executions=v3_executions,
            phase_commit=v3_commit,
        ),
    )
    paired = build_paired_allocation_comparison_commitment(methods)
    paired_record = paired.to_record()
    write_json_atomic(
        root / "paired_allocation_comparison_commitment.json",
        paired_record,
    )
    if _load_object(root / "paired_allocation_comparison_commitment.json") != (
        paired_record
    ):
        raise FreshPairedTrialError("paired generic commitment read-back changed")
    benchmark_commitment = airfoil.bind_airfoil_mpn_paired_allocation_commitment(
        bundle.arms,
        methods,
        paired,
        expected_schedule_binding_sha256=schedule_binding,
    )
    benchmark_record = benchmark_commitment.to_record()
    write_json_atomic(
        root / "airfoil_paired_allocation_commitment.json",
        benchmark_record,
    )
    if _load_object(root / "airfoil_paired_allocation_commitment.json") != (
        benchmark_record
    ):
        raise FreshPairedTrialError("Airfoil paired commitment read-back changed")

    # This is the first permitted real-oracle access in the fresh run.  The
    # prepare-side evaluator had no file bindings and could not cross this seam.
    write_json_atomic(
        root / "postcommit_oracle_access_started.json",
        {
            "schema_version": 1,
            "both_method_commits_read_back": True,
            "paired_generic_commitment_read_back": True,
            "airfoil_paired_commitment_read_back": True,
            "raw_authority": "committed_selected_union_only",
        },
    )
    oracle = airfoil.verify_airfoil_v7_predecision_oracle(bundle.oracle_dir)
    frozen_seal = thaw_json(bundle.preparation.oracle_seal)
    if (
        oracle.contract.identity_sha256
        != bundle.preparation.contract.identity_sha256
        or oracle.seal_record() != frozen_seal
    ):
        raise FreshPairedTrialError("postcommit sealed oracle differs from prepare")
    evaluator = airfoil.AirfoilV7SealedOracleDevelopmentEvaluator(oracle)
    evaluator.authorize_initial_g1(bundle.preparation.sample)
    capability = evaluator.open_paired_postdecision_evaluation(
        benchmark_commitment
    )
    unique_evaluations = capability.evaluate_selected_union()
    by_id = {value.option_id: value for value in unique_evaluations}
    if (
        set(by_id) != set(benchmark_commitment.selected_option_ids)
        or not 3 <= len(by_id) <= 18
    ):
        raise FreshPairedTrialError("paired evaluator returned another selected union")

    endpoint_analysis = _selected_endpoint_analysis(
        methods=methods,
        evaluations=by_id,
    )
    logical_slots = [
        {
            "comparison_method_id": method.comparison_method_id,
            "arm": execution.treatment_occurrence.treatment_id.value,
            "rank": member.rank,
            "option_id": member.option_id,
            "evaluation": by_id[member.option_id].to_record(),
        }
        for method in methods
        for execution in method.executions
        for member in execution.result.decision.members
    ]
    if len(logical_slots) != 18:
        raise FreshPairedTrialError("paired logical evaluator slots changed")
    outcomes_record = {
        "schema_version": 1,
        "raw_outcome_authority": "committed_selected_union_only",
        "unselected_outcomes_exposed": False,
        "logical_slot_count": 18,
        "unique_cached_read_count": len(by_id),
        "unique_evaluations": [
            by_id[value].to_record() for value in sorted(by_id)
        ],
        "logical_slots": logical_slots,
        "primary_endpoint_analysis": endpoint_analysis,
        "new_cfd_calls": 0,
        "evaluator_wall_clock_claim": False,
    }
    write_json_atomic(root / "selected_union_outcomes.json", outcomes_record)
    return {
        "schema_version": 1,
        "status": "completed_paired_selected_union_primary_endpoint",
        "allocation_method_count": 2,
        "allocation_arm_count_per_method": 3,
        "candidate_score_count_per_arm_per_method": expected_candidate_scores,
        "total_candidate_score_count": 2 * 3 * expected_candidate_scores,
        "paired_comparison_commitment_sha256": paired.commitment_sha256,
        "airfoil_paired_commitment_sha256": (
            benchmark_commitment.commitment_sha256
        ),
        "logical_selected_evaluation_slots": 18,
        "unique_selected_cached_reads": len(by_id),
        "primary_endpoint_analysis": endpoint_analysis,
        "exact_three_set_rank_status": (
            "not_computed_without_separate_postcommit_reference_release"
        ),
        "oracle_outcome_file_opened_after_commits": True,
        "selected_action_evaluator_calls": len(by_id),
        "new_candidate_evaluations": 0,
        "new_cfd_calls": 0,
    }


class Runner(Protocol):
    async def __aenter__(self) -> "Runner": ...
    async def __aexit__(self, *args: object) -> None: ...
    async def __call__(self, request: StructuredGenerationRequest[Any]) -> object: ...
    async def snapshot(self) -> object: ...


BlockHealthAssessor = Callable[..., object]
BlockSubsetHealthAssessor = Callable[..., object]
_PRODUCTION_RUNNER_FACTORY = create_progress_aware_openrouter_runner
_PRODUCTION_BLOCK_HEALTH_ASSESSOR = assess_resolved_action_forecast_block_health
_PRODUCTION_SUBSET_HEALTH_ASSESSOR = (
    assess_resolved_action_forecast_block_subset_health
)
_PRODUCTION_PAIRED_ADJUDICATOR = adjudicate_airfoil_paired_allocations


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    """Replaceable system edges; the orchestration policy remains closed."""

    runner_factory: Callable[..., Runner] = _PRODUCTION_RUNNER_FACTORY
    block_health_assessor: BlockHealthAssessor = (
        _PRODUCTION_BLOCK_HEALTH_ASSESSOR
    )
    block_subset_health_assessor: BlockSubsetHealthAssessor = (
        _PRODUCTION_SUBSET_HEALTH_ASSESSOR
    )
    paired_adjudicator: PairedAdjudicator = _PRODUCTION_PAIRED_ADJUDICATOR

    def __post_init__(self) -> None:
        for name in (
            "runner_factory",
            "block_health_assessor",
            "block_subset_health_assessor",
            "paired_adjudicator",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


def _production_live_dependencies() -> LiveDependencies:
    result = LiveDependencies()
    if (
        result.runner_factory is not _PRODUCTION_RUNNER_FACTORY
        or result.block_health_assessor is not _PRODUCTION_BLOCK_HEALTH_ASSESSOR
        or result.block_subset_health_assessor
        is not _PRODUCTION_SUBSET_HEALTH_ASSESSOR
        or result.paired_adjudicator is not _PRODUCTION_PAIRED_ADJUDICATOR
    ):
        raise FreshPairedTrialError("production live dependency identity changed")
    return result


class _PrecommittedRecordingRunner:
    """Enforce exact opaque calls and persist admitted typed v5 matrices."""

    def __init__(
        self,
        delegate: Runner,
        *,
        run_dir: Path,
        opaque_calls: tuple[OpaquePlannedCall, ...],
        submission_journal: DurableJsonlJournal,
    ) -> None:
        self._delegate = delegate
        self._run_dir = run_dir
        self._prepared = {
            value.plan.call_id.value: (
                historical._request_plan_payload(value.plan),
                value.provider_slot_assignment,
            )
            for value in opaque_calls
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
        payload = historical._request_plan_payload(request)
        call_id = request.call_id.value
        with self._lock:
            prepared = self._prepared.get(call_id)
            if prepared is None or prepared[0] != payload:
                raise FreshPairedTrialError(
                    "dispatched v5 call differs from its opaque precommit"
                )
            if call_id in self._submitted:
                raise FreshPairedTrialError("one paired logical call was submitted twice")
            self._submitted.add(call_id)
            assignment = prepared[1]
            self._submission_journal.append(
                {
                    "schema_version": 1,
                    "opaque_provider_slot_id": assignment.opaque_provider_slot_id.value,
                    "occurrence_id": assignment.occurrence_id.value,
                    "call_id": call_id,
                    "prompt_sha256": payload["call_contract"]["prompt_sha256"],
                    "schema_sha256": payload["call_contract"]["schema_sha256"],
                    "submitted_to_queue_delegate": True,
                    "treatment_label_sent_as_metadata": False,
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
            "opaque_provider_slot_id": assignment.opaque_provider_slot_id.value,
            "occurrence_id": assignment.occurrence_id.value,
            "call_id": call_id,
            "prompt_sha256": payload["call_contract"]["prompt_sha256"],
            "schema_sha256": payload["call_contract"]["schema_sha256"],
            "typed_output_type": type(response.value).__name__,
            "typed_code_matrices": response.value.model_dump(mode="json"),
            "treatment_label_sent_as_metadata": False,
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
        raise FreshPairedTrialError("paired block telemetry is missing")
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
        raise FreshPairedTrialError("paired block escaped the frozen StreamLake route")


def _health_record(value: object) -> dict[str, object]:
    to_record = getattr(value, "to_record", None)
    if not callable(to_record):
        raise FreshPairedTrialError("health assessor returned no receipt")
    record = to_record()
    if type(record) is not dict or type(record.get("passes")) is not bool:
        raise FreshPairedTrialError("health assessor returned a malformed receipt")
    return record


def _physical_attempt_accounting(
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    physical = 0
    succeeded = 0
    retries = 0
    for outcome in outcomes:
        attempts = outcome.get("attempts")
        if type(attempts) is not list or not 1 <= len(attempts) <= MAX_ATTEMPTS:
            raise FreshPairedTrialError("terminal attempt ledger is malformed")
        for attempt in attempts:
            if type(attempt) is not dict or type(attempt.get("will_retry")) is not bool:
                raise FreshPairedTrialError("one physical attempt is malformed")
            physical += 1
            succeeded += int(attempt.get("status") == "succeeded")
            retries += int(attempt["will_retry"] is True)
    return {
        "physical_attempt_count": physical,
        "successful_physical_attempt_count": succeeded,
        "scheduled_retry_count": retries,
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
        raise FreshPairedTrialError("accepted block lacks exact telemetry")
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


def _materialize_terminal_provider_ledger(
    *,
    claim: LiveClaim,
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    bundle = claim.prepared.bundle
    if len(outcomes) != 3:
        raise FreshPairedTrialError("exactly three terminal outcomes are required")
    by_call: dict[str, Mapping[str, object]] = {}
    for value in outcomes:
        call_id = value.get("task_id")
        if type(call_id) is not str or call_id in by_call:
            raise FreshPairedTrialError("terminal outcomes have malformed task IDs")
        by_call[call_id] = value
    opaque_call_ids = tuple(value.plan.call_id.value for value in bundle.opaque_calls)
    if set(by_call) != set(opaque_call_ids):
        raise FreshPairedTrialError("terminal outcomes differ from the opaque wave")
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "all_three_calls_terminal",
        "experiment_commitment_sha256": bundle.experiment_record[
            "experiment_commitment_sha256"
        ],
        "treatment_assignment_receipt_sha256": (
            bundle.treatment_assignment.receipt_sha256
        ),
        "opaque_slots": [
            {
                "opaque_provider_slot_id": value.provider_slot_assignment.opaque_provider_slot_id.value,
                "occurrence_id": value.provider_slot_assignment.occurrence_id.value,
                "call_id": value.plan.call_id.value,
                "terminal_outcome": dict(by_call[value.plan.call_id.value]),
            }
            for value in bundle.opaque_calls
        ],
        "terminal_outcome_count": 3,
        "successful_outcome_count": sum(
            value.get("status") == "succeeded" for value in by_call.values()
        ),
        "all_calls_settled_before_allocation": True,
        "allocation_started_before_ledger_fsync": False,
        "oracle_outcome_files_read_before_ledger_fsync": 0,
    }
    record["commitment_sha256"] = _hash(_TERMINAL_LEDGER_FRAMING, record)
    write_json_atomic(claim.run_dir / "terminal_provider_ledger.json", record)
    if _load_object(claim.run_dir / "terminal_provider_ledger.json") != record:
        raise FreshPairedTrialError("terminal provider ledger read-back changed")
    return record


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
        # Queue completion cannot become visible downstream until all stream
        # progress preceding it is durable.
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
    recording = _PrecommittedRecordingRunner(
        runner,
        run_dir=claim.run_dir,
        opaque_calls=bundle.opaque_calls,
        submission_journal=submission_journal,
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
                await runner.snapshot(),
                stage="before_three_opaque_calls",
            )
        )
        policy = PydanticAIActionForecastBlockPolicy(recording)
        raw_in_opaque_order = await asyncio.gather(
            *(
                policy.forecast_block(value.block_request)
                for value in bundle.opaque_calls
            ),
            return_exceptions=True,
        )
        snapshot_journal.append(
            v2_launcher._queue_snapshot_record(
                await runner.snapshot(),
                stage="after_three_opaque_calls",
            )
        )

    # This is the hard chronology seam: every terminal result is normalized,
    # fsynced, and read back before route, health, allocation, or oracle logic.
    terminal_ledger = _materialize_terminal_provider_ledger(
        claim=claim,
        outcomes=outcomes,
    )

    raw_by_arm: dict[str, object] = {}
    for opaque_call, raw in zip(
        bundle.opaque_calls,
        raw_in_opaque_order,
        strict=True,
    ):
        arm = bundle.arm_for_occurrence(
            opaque_call.provider_slot_assignment.occurrence_id.value
        )
        if arm in raw_by_arm:
            raise FreshPairedTrialError("opaque assignment maps two calls to one arm")
        raw_by_arm[arm] = raw
    if set(raw_by_arm) != {"m", "p", "n"}:
        raise FreshPairedTrialError("opaque wave did not restore canonical M/P/N")

    failures: list[dict[str, object]] = []
    accepted: list[
        tuple[str, ActionForecastBlockRequest, ActionForecastBlockResult]
    ] = []
    for arm, block_request in zip(
        ("m", "p", "n"),
        bundle.selected_block_requests,
        strict=True,
    ):
        raw = raw_by_arm[arm]
        if isinstance(raw, BaseException):
            failures.append(
                {
                    "phase": "provider_or_decode",
                    "arm": arm,
                    "call_id": block_request.block_call_id.value,
                    "failure_type": type(raw).__name__,
                }
            )
            continue
        if type(raw) is not ActionForecastBlockResult:
            failures.append(
                {
                    "phase": "provider_or_decode",
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
                    "phase": "route_validation",
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

    expected_call_ids = tuple(
        value.block_call_id.value for value in bundle.selected_block_requests
    )
    successful_attempt_validation: dict[str, object] | None = None
    if terminal_ledger["successful_outcome_count"] == 3:
        try:
            successful_attempt_validation = progress.validate_successful_attempts(
                outcomes,
                expected_call_ids=expected_call_ids,
                expected_prompt_sha256_by_call={
                    value.call_id.value: str(
                        historical._request_plan_payload(value)["call_contract"][
                            "prompt_sha256"
                        ]
                    )
                    for value in bundle.planned_calls
                },
            )
        except BaseException as error:
            failures.append(
                {
                    "phase": "attempt_validation",
                    "arm": None,
                    "call_id": None,
                    "failure_type": type(error).__name__,
                }
            )
    else:
        failures.append(
            {
                "phase": "terminal_provider_ledger",
                "arm": None,
                "call_id": None,
                "failure_type": "OneOrMoreTerminalCallsFailed",
            }
        )
    if recording.submitted_call_ids != tuple(sorted(expected_call_ids)):
        failures.append(
            {
                "phase": "submission_cardinality",
                "arm": None,
                "call_id": None,
                "failure_type": "SubmittedCallSetChanged",
            }
        )

    health_records: list[dict[str, object]] = []
    subset_health_records: list[dict[str, object]] = []
    if not failures and len(accepted) == 3:
        for arm, block_request, result in accepted:
            try:
                health = _health_record(
                    dependencies.block_health_assessor(
                        block_request,
                        result.forecasts,
                        member_id=arm,
                        health_policy=lenient_action_forecast_health_policy(),
                    )
                )
                subset_health = _health_record(
                    dependencies.block_subset_health_assessor(
                        block_request,
                        result.forecasts,
                        member_id=arm,
                        health_policy=lenient_action_forecast_health_policy(),
                        subset_policy=historical.eligible_subset_policy(),
                        included_global_row_indices=(
                            bundle.schedule_assignment.eligible_global_row_indices
                        ),
                    )
                )
                health_records.append({"arm": arm, "assessment": health})
                subset_health_records.append(
                    {"arm": arm, "assessment": subset_health}
                )
                write_json_atomic(claim.run_dir / f"block_health_{arm}.json", health)
                write_json_atomic(
                    claim.run_dir / f"eligible_subset_health_{arm}.json",
                    subset_health,
                )
            except BaseException as error:
                failures.append(
                    {
                        "phase": "semantic_health",
                        "arm": arm,
                        "call_id": block_request.block_call_id.value,
                        "failure_type": type(error).__name__,
                    }
                )

    physical_attempts = _physical_attempt_accounting(outcomes)
    accepted_usage = _accepted_usage_accounting(accepted)
    qualification_counts = {
        "planned": 3,
        "submitted": len(recording.submitted_call_ids),
        "terminal_outcomes": len(outcomes),
        "typed_wires": recording.typed_wire_count,
        "accepted_blocks": len(accepted),
        "health_assessments": len(health_records),
        "eligible_subset_health_assessments": len(subset_health_records),
    }
    accounting = {
        "schema_version": 1,
        "authorized_logical_call_count": 3,
        "new_logical_provider_calls": len(recording.submitted_call_ids),
        "qualification_counts": qualification_counts,
        "physical_attempts": physical_attempts,
        "successful_attempt_validation": successful_attempt_validation,
        "accepted_usage": accepted_usage,
        "terminal_provider_ledger_commitment_sha256": terminal_ledger[
            "commitment_sha256"
        ],
        "allocation_may_start_only_after_terminal_ledger_readback": True,
    }
    write_json_atomic(claim.run_dir / "call_accounting.json", accounting)

    health_passes = (
        len(health_records) == 3
        and len(subset_health_records) == 3
        and all(
            row["assessment"]["passes"] is True
            for row in (*health_records, *subset_health_records)
        )
    )
    paired_result: Mapping[str, object] | None = None
    if not failures and health_passes:
        try:
            paired_result = dependencies.paired_adjudicator(
                PostLedgerContext(
                    claim=claim,
                    accepted=tuple(accepted),
                    terminal_ledger=terminal_ledger,
                )
            )
            if (
                not isinstance(paired_result, Mapping)
                or type(paired_result.get("status")) is not str
            ):
                raise FreshPairedTrialError(
                    "paired adjudicator returned a malformed result"
                )
        except BaseException as error:
            failures.append(
                {
                    "phase": "paired_allocation_or_selected_union",
                    "arm": None,
                    "call_id": None,
                    "failure_type": type(error).__name__,
                }
            )

    if failures:
        status = "incomplete"
    elif not health_passes:
        status = "typed_but_semantically_degenerate"
    elif paired_result is None:
        status = "incomplete"
    else:
        status = str(paired_result["status"])
    if failures:
        write_json_atomic(
            claim.run_dir / "failures.json",
            {"schema_version": 1, "failures": failures},
        )
    selected_reads = (
        0
        if paired_result is None
        else int(paired_result.get("unique_selected_cached_reads", 0))
    )
    return {
        "schema_version": 1,
        "status": status,
        "scientific_scope": "one_prospective_developmental_paired_block",
        "authorized_logical_provider_calls": 3,
        "new_logical_provider_calls": len(recording.submitted_call_ids),
        "planned_logical_call_count": 3,
        "submitted_logical_call_count": len(recording.submitted_call_ids),
        "terminal_queue_outcome_count": len(outcomes),
        "terminal_provider_ledger_commitment_sha256": terminal_ledger[
            "commitment_sha256"
        ],
        "accepted_typed_block_count": len(accepted),
        "typed_wire_artifact_count": recording.typed_wire_count,
        "health_assessment_count": len(health_records),
        "eligible_subset_health_assessment_count": len(subset_health_records),
        "health_pass_count": sum(
            row["assessment"]["passes"] is True for row in health_records
        ),
        "eligible_subset_health_pass_count": sum(
            row["assessment"]["passes"] is True
            for row in subset_health_records
        ),
        "qualification_counts": qualification_counts,
        "physical_attempts": physical_attempts,
        "successful_attempt_validation": successful_attempt_validation,
        "accepted_usage": accepted_usage,
        "paired_result": None if paired_result is None else dict(paired_result),
        "failures": failures,
        "historical_g1_terminal_records_rehydrated": 8,
        "historical_reflection_reused": 1,
        "historical_reflection_counted_as_new_call": False,
        "allocation_execution_count": (
            0 if paired_result is None else 6
        ),
        "selected_action_evaluator_calls": selected_reads,
        "new_candidate_evaluations": 0,
        "new_cfd_calls": 0,
        "oracle_outcome_access_attempted": (
            claim.run_dir / "postcommit_oracle_access_started.json"
        ).is_file(),
    }


def _manifest(claim: LiveClaim, *, release_mode: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": "airfoil_v7_v5_paired_causal_trial_live",
        "run_dir": str(claim.run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_dir": str(claim.prepared.run_dir),
        "preparation_commitment_sha256": claim.prepared.record[
            "preparation_commitment_sha256"
        ],
        "preparation_finalization_sha256": claim.prepared.finalization[
            "finalization_sha256"
        ],
        "experiment_commitment_sha256": claim.prepared.bundle.experiment_record[
            "experiment_commitment_sha256"
        ],
        "closed_source_identity": claim.claim_record["closed_source_identity"],
        "runtime_identity": claim.claim_record["runtime_identity"],
        "route": v2_launcher.route_binding(),
        "queue": build_config().to_manifest_record(),
        "execution_mode": (
            "paid_release" if release_mode else "provider_free_injected_test"
        ),
        "provider_dispatch_authorized": release_mode,
        "logical_call_count": 3,
        "opaque_provider_slot_count": 3,
        "downstream_actions_authorized": [
            "health_v3",
            "audited_frame_v2_allocation",
            "operational_frame_v3_allocation",
            "paired_selected_union_evaluation",
        ],
        "downstream_chronology": (
            "only_after_terminal_provider_ledger_exact_readback"
        ),
        "rank_only_reference_authority_authorized": False,
        "live_completion_status_authorized": release_mode,
    }
    record["manifest_commitment_sha256"] = _hash(_MANIFEST_FRAMING, record)
    return record


def _validate_release_completion(
    *,
    root: Path,
    result: Mapping[str, object],
) -> None:
    expected_status = "completed_paired_selected_union_primary_endpoint"
    if result.get("status") != expected_status:
        raise FreshPairedTrialError("release completion has an unexpected status")
    paired_result = result.get("paired_result")
    if (
        type(paired_result) is not dict
        or paired_result.get("status") != expected_status
    ):
        raise FreshPairedTrialError("release completion lacks exact paired result")

    ledger = _load_object(root / "terminal_provider_ledger.json")
    v2_commit = _load_object(root / "durable_allocation_v2_commit.json")
    v3_commit = _load_object(root / "durable_allocation_v3_commit.json")
    generic = _load_object(root / "paired_allocation_comparison_commitment.json")
    benchmark = _load_object(root / "airfoil_paired_allocation_commitment.json")
    access = _load_object(root / "postcommit_oracle_access_started.json")
    outcomes = _load_object(root / "selected_union_outcomes.json")
    if (
        ledger.get("status") != "all_three_calls_terminal"
        or ledger.get("successful_outcome_count") != 3
        or result.get("terminal_provider_ledger_commitment_sha256")
        != ledger.get("commitment_sha256")
    ):
        raise FreshPairedTrialError("completed release lacks its terminal ledger")
    method_receipts = generic.get("method_receipts")
    benchmark_methods = benchmark.get("method_commits")
    if (
        type(method_receipts) is not list
        or len(method_receipts) != 2
        or type(benchmark_methods) is not list
        or len(benchmark_methods) != 2
    ):
        raise FreshPairedTrialError("completed release lacks two method commits")
    generic_by_method = {
        str(value.get("comparison_method_id")): value
        for value in method_receipts
        if type(value) is dict
    }
    benchmark_by_method = {
        str(value.get("comparison_method_id")): value
        for value in benchmark_methods
        if type(value) is dict
    }
    if set(generic_by_method) != {
        "audited_frame_v2",
        "operational_frame_v3",
    } or set(benchmark_by_method) != set(generic_by_method):
        raise FreshPairedTrialError("completed release method IDs changed")
    phase_receipts = {
        "audited_frame_v2": v2_commit.get("receipt", {}).get("receipt_sha256")
        if type(v2_commit.get("receipt")) is dict
        else None,
        "operational_frame_v3": v3_commit.get("receipt", {}).get(
            "receipt_sha256"
        )
        if type(v3_commit.get("receipt")) is dict
        else None,
    }
    for method_id, phase_receipt in phase_receipts.items():
        if (
            type(phase_receipt) is not str
            or generic_by_method[method_id].get(
                "allocation_phase_commit_receipt_sha256"
            )
            != phase_receipt
            or benchmark_by_method[method_id].get(
                "allocation_phase_commit_receipt_sha256"
            )
            != phase_receipt
        ):
            raise FreshPairedTrialError("completed release phase receipts disagree")
    if (
        generic.get("commitment_sha256")
        != paired_result.get("paired_comparison_commitment_sha256")
        or benchmark.get("paired_comparison_commitment_sha256")
        != generic.get("commitment_sha256")
        or benchmark.get("commitment_sha256")
        != paired_result.get("airfoil_paired_commitment_sha256")
        or generic.get("logical_slot_count") != 18
        or benchmark.get("logical_slot_count") != 18
    ):
        raise FreshPairedTrialError("completed release paired commitments disagree")
    if access != {
        "schema_version": 1,
        "both_method_commits_read_back": True,
        "paired_generic_commitment_read_back": True,
        "airfoil_paired_commitment_read_back": True,
        "raw_authority": "committed_selected_union_only",
    }:
        raise FreshPairedTrialError("completed release oracle chronology changed")
    unique = outcomes.get("unique_evaluations")
    logical = outcomes.get("logical_slots")
    if type(unique) is not list or type(logical) is not list or len(logical) != 18:
        raise FreshPairedTrialError("completed release selected outcomes are malformed")
    committed_union = benchmark.get("selected_option_ids")
    returned_union = sorted(
        str(value.get("option_id")) for value in unique if type(value) is dict
    )
    logical_union = sorted(
        {str(value.get("option_id")) for value in logical if type(value) is dict}
    )
    if (
        type(committed_union) is not list
        or returned_union != committed_union
        or logical_union != committed_union
        or outcomes.get("unique_cached_read_count") != len(committed_union)
        or outcomes.get("raw_outcome_authority")
        != "committed_selected_union_only"
        or outcomes.get("unselected_outcomes_exposed") is not False
        or outcomes.get("primary_endpoint_analysis")
        != paired_result.get("primary_endpoint_analysis")
        or result.get("selected_action_evaluator_calls") != len(committed_union)
    ):
        raise FreshPairedTrialError("completed release selected union disagrees")


def _execute_live_common(
    *,
    claim: LiveClaim,
    api_key: str,
    dependencies: LiveDependencies,
    release_mode: bool,
) -> dict[str, object]:
    """Execute exactly one prepared opaque wave and finalize every status."""

    if type(claim) is not LiveClaim or not claim.active:
        raise FreshPairedTrialError("live execution requires an active exact claim")
    if type(api_key) is not str or not api_key:
        raise FreshPairedTrialError("live API key is unavailable")
    dependencies.__post_init__()
    if release_mode and (
        dependencies.runner_factory is not _PRODUCTION_RUNNER_FACTORY
        or dependencies.block_health_assessor
        is not _PRODUCTION_BLOCK_HEALTH_ASSESSOR
        or dependencies.block_subset_health_assessor
        is not _PRODUCTION_SUBSET_HEALTH_ASSESSOR
        or dependencies.paired_adjudicator is not _PRODUCTION_PAIRED_ADJUDICATOR
    ):
        raise FreshPairedTrialError(
            "paid release requires exact production dependency identities"
        )
    root = claim.run_dir
    source = current_source_identity()
    runtime = v2_launcher.runtime_identity()
    if (
        source != claim.claim_record.get("closed_source_identity")
        or source != claim.prepared.record.get("closed_source_identity")
    ):
        raise FreshPairedTrialError("closed source changed before dispatch")
    if (
        runtime != claim.claim_record.get("runtime_identity")
        or runtime != claim.prepared.record.get("runtime_identity")
    ):
        raise FreshPairedTrialError("runtime changed before dispatch")
    if claim.prepared.record.get("authorized_target_live_run_dir") != str(root):
        raise FreshPairedTrialError("claimed directory escaped one-shot authorization")
    historical._validate_planned_wire_contracts(claim.prepared.bundle.planned_calls)
    v2_launcher._validate_provider_boundary_blinding(
        claim.prepared.bundle.planned_calls
    )

    write_json_atomic(root / "manifest.json", _manifest(claim, release_mode=release_mode))
    write_json_atomic(root / "planned_opaque_wave.json", dict(claim.prepared.wave))
    planned_journal = DurableJsonlJournal(root / "planned_calls.jsonl")
    submission_journal = DurableJsonlJournal(root / "submitted_calls.jsonl")
    progress_journal = BatchedDurableJsonlJournal(
        root / "stream_progress.jsonl",
        max_unfsynced_rows=PROGRESS_MAX_UNFSYNCED_ROWS,
    )
    progress = v2_launcher._ProgressRecorder(progress_journal)
    outcome_journal = DurableJsonlJournal(root / "queue_outcomes.jsonl")
    snapshot_journal = DurableJsonlJournal(root / "queue_snapshots.jsonl")
    for ordinal, opaque_call in enumerate(
        claim.prepared.bundle.opaque_calls,
        start=1,
    ):
        payload = historical._request_plan_payload(opaque_call.plan)
        assignment = opaque_call.provider_slot_assignment
        planned_journal.append(
            {
                "schema_version": 1,
                "ordinal": ordinal,
                "opaque_provider_slot_id": assignment.opaque_provider_slot_id.value,
                "occurrence_id": assignment.occurrence_id.value,
                **payload["call_contract"],
                "treatment_label_sent_as_metadata": False,
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
        if release_mode and result.get("status") == (
            "completed_paired_selected_union_primary_endpoint"
        ):
            _validate_release_completion(root=root, result=result)
        elif release_mode and result.get("status") not in {
            "incomplete",
            "typed_but_semantically_degenerate",
        }:
            raise FreshPairedTrialError("release runner returned a foreign status")
    except BaseException as error:
        pending = error
        result = {
            "schema_version": 1,
            "status": "incomplete",
            "failure_type": type(error).__name__,
            "scientific_scope": "one_prospective_developmental_paired_block",
            "authorized_logical_provider_calls": 3,
            "historical_g1_terminal_records_rehydrated": 8,
            "historical_reflection_reused": 1,
            "new_candidate_evaluations": 0,
            "allocation_execution_count": 0,
            "selected_action_evaluator_calls": 0,
            "new_cfd_calls": 0,
            "oracle_outcome_access_attempted": (
                root / "postcommit_oracle_access_started.json"
            ).is_file(),
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
                FreshPairedTrialError("closed source changed during paired trial")
            )
        if cleanup_errors and pending is None:
            pending = cleanup_errors[0]
            result = {
                **result,
                "status": "incomplete",
                "failure_type": type(pending).__name__,
            }

    if not release_mode:
        underlying_status = str(result.get("status"))
        result = {
            **result,
            "underlying_provider_free_test_status": underlying_status,
            "status": "provider_free_injected_test_completed",
            "release_eligible": False,
            "paid_provider_evidence": False,
        }

    try:
        planned_count = len(read_jsonl(root / "planned_calls.jsonl"))
        submitted_count = len(read_jsonl(root / "submitted_calls.jsonl"))
        outcome_count = len(read_jsonl(root / "queue_outcomes.jsonl"))
        result["credential_read_attempted"] = True
        result["credentials_read"] = True
        result["provider_client_constructed"] = (
            root / "runner_constructed.json"
        ).is_file()
        result["planned_logical_call_count"] = planned_count
        result["submitted_logical_call_count"] = submitted_count
        result["terminal_queue_outcome_count"] = outcome_count
        result["authorized_logical_provider_calls"] = 3
        result["new_logical_provider_calls"] = submitted_count
        result["provider_call_attempted"] = submitted_count > 0
        result["terminal_provider_ledger_materialized"] = (
            root / "terminal_provider_ledger.json"
        ).is_file()
        if "physical_attempts" not in result and outcome_count:
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


def execute_live(*, claim: LiveClaim, api_key: str) -> dict[str, object]:
    """Paid release entry: dependency injection is intentionally unavailable."""

    return _execute_live_common(
        claim=claim,
        api_key=api_key,
        dependencies=_production_live_dependencies(),
        release_mode=True,
    )


def _execute_live_with_dependencies_for_test(
    *,
    claim: LiveClaim,
    dependencies: LiveDependencies,
) -> dict[str, object]:
    """Provider-free composition seam that can never emit a live-success status."""

    return _execute_live_common(
        claim=claim,
        api_key="provider-free-injected-test-key",
        dependencies=dependencies,
        release_mode=False,
    )


def _finalize_credential_abort(
    claim: LiveClaim,
    error: BaseException,
    *,
    credentials_read: bool,
) -> None:
    if not claim.active:
        return
    root = claim.run_dir
    if (root / "finalized.json").is_file():
        claim.close()
        return
    planned = (
        read_jsonl(root / "planned_calls.jsonl")
        if (root / "planned_calls.jsonl").is_file()
        else []
    )
    submitted = (
        read_jsonl(root / "submitted_calls.jsonl")
        if (root / "submitted_calls.jsonl").is_file()
        else []
    )
    outcomes = (
        read_jsonl(root / "queue_outcomes.jsonl")
        if (root / "queue_outcomes.jsonl").is_file()
        else []
    )
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
            "planned_logical_call_count": len(planned),
            "submitted_logical_call_count": len(submitted),
            "terminal_queue_outcome_count": len(outcomes),
            "authorized_logical_provider_calls": 3,
            "new_logical_provider_calls": len(submitted),
            "provider_call_attempted": bool(submitted),
            "terminal_provider_ledger_materialized": (
                root / "terminal_provider_ledger.json"
            ).is_file(),
            "historical_g1_terminal_records_rehydrated": 8,
            "new_candidate_evaluations": 0,
            "allocation_execution_count": 0,
            "selected_action_evaluator_calls": 0,
            "new_cfd_calls": 0,
            "oracle_outcome_access_attempted": (
                root / "postcommit_oracle_access_started.json"
            ).is_file(),
        },
    )
    finalize_run_directory(root, status="incomplete")
    claim.close()


def finalize_precredential_abort(claim: LiveClaim, error: BaseException) -> None:
    _finalize_credential_abort(claim, error, credentials_read=False)


def finalize_postcredential_abort(claim: LiveClaim, error: BaseException) -> None:
    _finalize_credential_abort(claim, error, credentials_read=True)


def _load_dotenv_api_key() -> str:
    env_path = WORKSPACE_ROOT / ".env"
    value: str | None = None
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                value = candidate.strip().strip('"').strip("'")
                break
    if not value:
        value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise FreshPairedTrialError("OPENROUTER_API_KEY is unavailable")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "live"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--target-live-run-dir", type=Path)
    parser.add_argument("--frozen-v2-run", type=Path, default=DEFAULT_FROZEN_V2_RUN)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
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
        execution = execute_live(claim=claim, api_key=api_key)
    except BaseException as error:
        finalize_postcredential_abort(claim, error)
        raise
    return (
        0
        if execution["result"]["status"]
        == "completed_paired_selected_union_primary_endpoint"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

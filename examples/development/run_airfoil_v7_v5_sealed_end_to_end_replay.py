#!/usr/bin/env python3
"""Outcome-gated v5 replay of the immutable Airfoil-v7 v4 wire qualifier.

This development harness deliberately does *not* make a provider call.  It
reconstructs the frozen M/P/N block requests, decodes the already-sealed enum
matrices with the current v5 decoder, runs the generic health/frame/allocation
boundaries, durably commits all three decisions, and only then opens the exact
selected union in the sealed development oracle.

The result is a post-hoc decoder/allocation replay.  It is neither a fresh
causal treatment comparison nor new model-efficacy evidence.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from agent_evolve.application.action_allocation_frame import (
    AuditedGreedyForecastFrameAllocator,
)
from agent_evolve.application.action_allocation_frame_commit import (
    build_frame_action_allocation_phase_commit,
    validate_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_evidence_consistency import (
    assess_presented_action_evidence_block_consistency,
)
from agent_evolve.application.action_forecast_partitioning import (
    assess_resolved_action_forecast_block_health,
    assess_resolved_action_forecast_block_subset_health,
    lenient_action_forecast_health_policy,
)
from agent_evolve.application.treatment_assignment import (
    assign_treatment_occurrences,
)
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.integrations.pydantic_ai.action_forecast import (
    ACTION_FORECAST_POLICY_DEFINITION_SHA256,
    ACTION_FORECAST_POLICY_VERSION,
    PydanticAIActionForecastBlockPolicy,
)
from agent_evolve.ports.action_allocation_frame import (
    ActionAllocationFrameSubsetPolicyBinding,
    AllocationCandidateScoreDiagnostic,
    AllocationCandidateScoreDiagnosticInput,
    AllocationScoreDiagnosticBinding,
    AllocationSurfaceGatePolicyBinding,
    FrameActionAllocationRequest,
    ResolvedActionForecastAllocationFrame,
    bind_action_forecast_block_subset_allocation_frame,
)
from agent_evolve.ports.action_allocation_frame_commit import (
    FrameActionAllocationTreatmentExecution,
    frame_source_call_and_request_identity,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockRequest,
    ActionForecastBlockResult,
    ResolvedActionForecastBlock,
)
from agent_evolve.ports.artifact_store import (
    canonical_json_bytes,
    decode_json_bytes,
)
from agent_evolve.ports.presented_action_evidence import (
    PresentedActionEvidenceCell,
    PresentedActionEvidenceProvenanceKind,
)
from agent_evolve.ports.structured_generator import StructuredGenerationResponse
from agent_evolve.ports.treatment_assignment import (
    OpaqueProviderSlotId,
    TreatmentAssignmentInput,
    TreatmentId,
    TreatmentOccurrence,
    TreatmentOccurrenceId,
)
from examples.development import airfoil_v7_two_stage_agent_evolution as airfoil
from examples.development import run_airfoil_v7_forecast_wire_v3_pilot as pilot
from examples.development.durable_run_artifacts import (
    file_identity,
    finalize_run_directory,
    read_jsonl,
    verify_finalized_run_directory,
    write_json_atomic,
)
from examples.benchmarks.engibench_airfoil.v7_finite_oracle import (
    OBJECTIVE_NAME,
    PARENT_METRICS,
    VIOLATION_NAME,
)


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_SOURCE_LIVE_DIR = (
    ARTIFACT_ROOT
    / "experiment_logs"
    / "airfoil_v7"
    / "wire_v3"
    / "ae7_forecast_wire_v4_qualifier_live_20260715"
)
DEFAULT_RUN_DIR = (
    ARTIFACT_ROOT
    / "experiment_logs"
    / "airfoil_v7"
    / "wire_v3"
    / "analysis"
    / "v5_sealed_end_to_end_allocation_replay_v2_20260715"
)

EXPECTED_SOURCE_FINALIZATION_SHA256 = (
    "2f6bb58339169dcb13711912302f566d484f22f4907d92054da6a6d84f68838f"
)
EXPECTED_SOURCE_RECURSIVE_SHA256 = (
    "baea835c90c755d2dab3d3d72586adc6f786e46e7c10611bf6520e0ad5bda16a"
)
EXPECTED_SOURCE_FILE_COUNT = 23
PREDECESSOR_REPLAY_FINALIZATION_SHA256 = (
    "c525b883b95672aad21d46e6558f1040e9de91c8bd97c21bced215dbd8a4bdfd"
)
PREDECESSOR_REPLAY_RECURSIVE_SHA256 = (
    "c2343d98a0389b1e4012dfe2c691c9e1e1c417b3b5001f5aad388a433020493b"
)
EXPECTED_V5_POLICY_VERSION = 5
EXPECTED_V5_POLICY_DEFINITION_SHA256 = (
    "422989ca43713dcba396e0cce4b6b1907c9d5c7a92b282a9ed15b171045910e7"
)
EXPECTED_ELIGIBLE_COUNT = 18
PORTFOLIO_SIZE = 3
EXPECTED_CANDIDATE_SCORES_PER_ARM = 18 + 17 + 16
EXPECTED_LOGICAL_EVALUATION_SLOTS = 9
EXPECTED_THREE_SET_COUNT = 816

REPLAY_SCOPE = (
    "posthoc_sealed_v4_wire_v5_decoder_allocation_replay_"
    "not_fresh_model_or_causal_efficacy"
)
ASSIGNMENT_PUBLIC_SEED = "airfoil.v5.sealed.replay.20260715"

ALLOCATION_SUBSET_POLICY_ID = "eligible_target_action_allocation_subset"
ALLOCATION_SUBSET_POLICY_VERSION = 1
ALLOCATION_SUBSET_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:eligible-target-action-allocation-subset:v1;"
    b"rows=selected-partition-block-intersection-precommitted-g2;"
    b"canonical-global-row-order;health-receipts-bound-as-parents"
).hexdigest()

ALLOCATION_DIAGNOSTIC_POLICY_ID = "normalized_forecast_tail_diagnostic"
ALLOCATION_DIAGNOSTIC_POLICY_VERSION = 1
ALLOCATION_DIAGNOSTIC_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:normalized-forecast-tail-diagnostic:v1;"
    b"candidate-only;boundary=any-abs-normalized-p50-ge-32;"
    b"tail=any-confidence-le-1-over-33;portfolio-history-excluded"
).hexdigest()

ALLOCATION_GATE_POLICY_ID = "development_noncollapsed_allocation_surface"
ALLOCATION_GATE_POLICY_VERSION = 1
ALLOCATION_GATE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:development-noncollapsed-allocation-surface:v1;"
    b"minimum-distinct-finite-scores=2;maximum-top-tie-share=0.5;"
    b"maximum-boundary-or-extreme-share=0.95;"
    b"minimum-winner-runner-gap=0"
).hexdigest()

ACTUAL_SET_UTILITY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:airfoil-v7-posthoc-actual-set-utility:v1;"
    b"valid-terminal-probability=1;per-member-quality="
    b"clip(-delta-v/0.005,-60,60)+0.05*tanh(-delta-f/0.001);"
    b"member-usefulness=sigmoid(quality);"
    b"set-utility=1-product(1-member-usefulness);higher-is-better"
).hexdigest()

_EXPERIMENT_DOMAIN = b"agent-evolve:sealed-wire-v5-replay-experiment:v1\x00"
_LEDGER_DOMAIN = b"agent-evolve:sealed-wire-terminal-provider-ledger:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:sealed-wire-v5-replay-result:v1\x00"


class SealedReplayError(RuntimeError):
    """A fail-closed replay gate rejected before outcome authority."""


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    value = decode_json_bytes(path.expanduser().resolve(strict=True).read_bytes())
    if type(value) is not dict:
        raise SealedReplayError(f"{path.name} must contain one JSON object")
    return value


def _exclusive_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _allocation_subset_policy() -> ActionAllocationFrameSubsetPolicyBinding:
    return ActionAllocationFrameSubsetPolicyBinding(
        policy_id=ALLOCATION_SUBSET_POLICY_ID,
        policy_version=ALLOCATION_SUBSET_POLICY_VERSION,
        policy_definition_sha256=ALLOCATION_SUBSET_POLICY_DEFINITION_SHA256,
    )


def _allocation_gate_policy() -> AllocationSurfaceGatePolicyBinding:
    return AllocationSurfaceGatePolicyBinding(
        policy_id=ALLOCATION_GATE_POLICY_ID,
        policy_version=ALLOCATION_GATE_POLICY_VERSION,
        policy_definition_sha256=ALLOCATION_GATE_POLICY_DEFINITION_SHA256,
        minimum_distinct_finite_scores=2,
        maximum_top_tie_share=0.5,
        maximum_boundary_or_extreme_share=0.95,
        minimum_winner_runner_gap=0.0,
    )


@dataclass(frozen=True, slots=True)
class _NormalizedTailDiagnostic:
    """Benchmark-bound interpretation injected into the generic audit seam."""

    frame: ResolvedActionForecastAllocationFrame

    def __post_init__(self) -> None:
        if type(self.frame) is not ResolvedActionForecastAllocationFrame:
            raise TypeError("frame must be an exact allocation frame")

    def __call__(
        self,
        request: AllocationCandidateScoreDiagnosticInput,
    ) -> AllocationCandidateScoreDiagnostic:
        request.__post_init__()
        by_label = {
            f"row_{global_index:08d}": forecast
            for global_index, forecast in zip(
                self.frame.global_row_indices,
                self.frame.forecasts,
                strict=True,
            )
        }
        candidate = by_label.get(request.candidate_label)
        if candidate is None:
            raise ValueError("allocation diagnostic received a foreign row label")
        scale_by_metric = {
            value.metric_id: value.delta_scale
            for value in self.frame.request.metric_scales
        }
        boundary_or_tail = False
        for metric in candidate.metric_forecasts:
            scale = scale_by_metric[metric.metric_id]
            if (
                abs(metric.p50_delta / scale) >= 32.0
                or metric.confidence <= (1.0 / 33.0)
            ):
                boundary_or_tail = True
        return AllocationCandidateScoreDiagnostic(
            boundary_or_extreme=boundary_or_tail
        )


def _allocation_diagnostic(
    frame: ResolvedActionForecastAllocationFrame,
) -> AllocationScoreDiagnosticBinding:
    return AllocationScoreDiagnosticBinding(
        diagnostic=_NormalizedTailDiagnostic(frame),
        policy_id=ALLOCATION_DIAGNOSTIC_POLICY_ID,
        policy_version=ALLOCATION_DIAGNOSTIC_POLICY_VERSION,
        policy_definition_sha256=(
            ALLOCATION_DIAGNOSTIC_POLICY_DEFINITION_SHA256
        ),
    )


def _verify_source_live(source_live_dir: Path) -> dict[str, object]:
    finalization = verify_finalized_run_directory(source_live_dir)
    if (
        finalization.get("status") != "wire_qualified"
        or finalization.get("finalization_sha256")
        != EXPECTED_SOURCE_FINALIZATION_SHA256
        or finalization.get("recursive_content_sha256")
        != EXPECTED_SOURCE_RECURSIVE_SHA256
        or type(finalization.get("files")) is not dict
        or len(finalization["files"]) != EXPECTED_SOURCE_FILE_COUNT
    ):
        raise SealedReplayError("immutable v4 source qualifier seal changed")
    return finalization


def _experiment_record(
    bundle: pilot.PilotBundle,
    source_finalization: Mapping[str, object],
) -> dict[str, object]:
    if (
        ACTION_FORECAST_POLICY_VERSION != EXPECTED_V5_POLICY_VERSION
        or ACTION_FORECAST_POLICY_DEFINITION_SHA256
        != EXPECTED_V5_POLICY_DEFINITION_SHA256
    ):
        raise SealedReplayError("default decoder is no longer the frozen v5 policy")
    health = lenient_action_forecast_health_policy()
    record: dict[str, object] = {
        "schema_version": 1,
        "scope": REPLAY_SCOPE,
        "postdecision_analysis_measure_repair": {
            "predecessor_finalization_sha256": (
                PREDECESSOR_REPLAY_FINALIZATION_SHA256
            ),
            "predecessor_recursive_content_sha256": (
                PREDECESSOR_REPLAY_RECURSIVE_SHA256
            ),
            "repair": (
                "rank-equivalent joint-failure complement arithmetic avoids "
                "rounding 1-small_probability to one"
            ),
            "forecast_allocation_or_evaluator_policy_changed": False,
        },
        "source_live_finalization_sha256": source_finalization[
            "finalization_sha256"
        ],
        "source_live_recursive_content_sha256": source_finalization[
            "recursive_content_sha256"
        ],
        "finite_contract_identity_sha256": (
            bundle.preparation.contract.identity_sha256
        ),
        "selected_block_index": bundle.selected_block_index,
        "selected_block_spec_sha256": (
            bundle.layout.blocks[bundle.selected_block_index].block_spec_sha256
        ),
        "selected_block_request_sha256s": [
            value.block_request_sha256 for value in bundle.selected_block_requests
        ],
        "eligible_g2_global_row_indices": list(
            bundle.eligible_g2_global_row_indices
        ),
        "eligible_g2_option_ids": [
            bundle.preparation.contract.options[index].option_id
            for index in bundle.eligible_g2_global_row_indices
        ],
        "decoder": {
            "policy_version": ACTION_FORECAST_POLICY_VERSION,
            "policy_definition_sha256": (
                ACTION_FORECAST_POLICY_DEFINITION_SHA256
            ),
            "provider_calls_authorized": 0,
        },
        "health_policy": health.to_record(),
        "health_subset_policy": pilot.eligible_subset_policy().to_record(),
        "allocation_subset_policy": _allocation_subset_policy().to_record(),
        "allocation_diagnostic_policy": {
            "policy_id": ALLOCATION_DIAGNOSTIC_POLICY_ID,
            "policy_version": ALLOCATION_DIAGNOSTIC_POLICY_VERSION,
            "policy_definition_sha256": (
                ALLOCATION_DIAGNOSTIC_POLICY_DEFINITION_SHA256
            ),
        },
        "allocation_gate_policy": _allocation_gate_policy().to_record(),
        "allocation_gate_status": (
            "posthoc_provisional_for_historical_replay_not_prospectively_"
            "preregistered"
        ),
        "allocator": {
            "risk_aversion_hex": airfoil.ALLOCATOR_RISK_AVERSION.hex(),
            "diversity_weight_hex": airfoil.ALLOCATOR_DIVERSITY_WEIGHT.hex(),
            "portfolio_size": PORTFOLIO_SIZE,
            "expected_candidate_scores_per_arm": (
                EXPECTED_CANDIDATE_SCORES_PER_ARM
            ),
        },
        "outcome_access": {
            "durable_allocation_commit_required": True,
            "logical_evaluation_slots": EXPECTED_LOGICAL_EVALUATION_SLOTS,
            "unique_cached_reads_min": 3,
            "unique_cached_reads_max": 9,
            "replacement_forbidden": True,
            "new_cfd_calls": 0,
        },
    }
    record["experiment_commitment_sha256"] = _hash(_EXPERIMENT_DOMAIN, record)
    return record


def _treatment_assignment(
    bundle: pilot.PilotBundle,
    experiment_commitment_sha256: str,
):
    occurrences = tuple(
        TreatmentOccurrence(
            occurrence_id=TreatmentOccurrenceId(f"replay.forecast.{index:02d}"),
            treatment_id=TreatmentId(arm),
            call_identity=block_request.block_call_id.value,
            request_identity_sha256=block_request.block_request_sha256,
        )
        for index, (arm, block_request) in enumerate(
            zip(("m", "p", "n"), bundle.selected_block_requests, strict=True)
        )
    )
    assignment_input = TreatmentAssignmentInput(
        experiment_commitment_sha256=experiment_commitment_sha256,
        public_seed_material=ASSIGNMENT_PUBLIC_SEED,
        occurrences=occurrences,
        provider_slot_ids=tuple(
            OpaqueProviderSlotId(f"opaque.replay.slot.{index:02d}")
            for index in range(3)
        ),
    )
    return assign_treatment_occurrences(assignment_input)


def _terminal_provider_ledger(
    *,
    source_live_dir: Path,
    bundle: pilot.PilotBundle,
    source_finalization: Mapping[str, object],
) -> dict[str, object]:
    outcomes = read_jsonl(source_live_dir / "queue_outcomes.jsonl")
    expected_calls = tuple(
        value.block_call_id.value for value in bundle.selected_block_requests
    )
    by_call = {str(value.get("task_id")): value for value in outcomes}
    if set(by_call) != set(expected_calls) or any(
        by_call[call_id].get("status") != "succeeded"
        for call_id in expected_calls
    ):
        raise SealedReplayError("sealed terminal provider ledger is incomplete")
    call_accounting = _load_object(source_live_dir / "call_accounting.json")
    if (
        call_accounting.get("authorized_logical_call_count") != 3
        or call_accounting.get("new_logical_provider_calls") != 3
    ):
        raise SealedReplayError("sealed provider call accounting changed")
    record: dict[str, object] = {
        "schema_version": 1,
        "source_finalization_sha256": source_finalization[
            "finalization_sha256"
        ],
        "source_recursive_content_sha256": source_finalization[
            "recursive_content_sha256"
        ],
        "expected_call_ids": list(expected_calls),
        "terminal_outcomes": [by_call[value] for value in expected_calls],
        "source_file_bindings": [
            file_identity(source_live_dir / name, relative_to=WORKSPACE_ROOT)
            for name in (
                "call_accounting.json",
                "queue_outcomes.jsonl",
                "stream_progress.jsonl",
                "submitted_calls.jsonl",
            )
        ],
        "ledger_materialized_before_allocation_commit": True,
        "provider_calls_made_by_replay": 0,
    }
    record["commitment_sha256"] = _hash(_LEDGER_DOMAIN, record)
    return record


async def _decode_v5_block(
    block_request: ActionForecastBlockRequest,
    typed_wire_record: Mapping[str, object],
) -> ActionForecastBlockResult:
    if typed_wire_record.get("call_id") != block_request.block_call_id.value:
        raise SealedReplayError("typed wire names another physical block call")
    payload = typed_wire_record.get("typed_code_matrices")
    if type(payload) is not dict:
        raise SealedReplayError("typed wire lacks its admitted enum matrices")

    async def replay_runner(request: Any) -> StructuredGenerationResponse[Any]:
        if request.call_id != block_request.block_call_id:
            raise SealedReplayError("v5 replay planner changed physical call identity")
        if request.output_type.__name__ != "AllOptionActionForecastMatrixV5":
            raise SealedReplayError("replay planner did not request the explicit v5 wire")
        value = request.output_type.model_validate(payload, strict=True)
        return StructuredGenerationResponse(
            value=value,
            requested_model="sealed/provider-free-replay",
            resolved_model="sealed/provider-free-replay",
            resolved_provider="sealed/provider-free-replay",
            provider_response_id=None,
            finish_reason="replay",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=0,
        )

    return await PydanticAIActionForecastBlockPolicy(replay_runner).forecast_block(
        block_request
    )


def _presented_evidence_cells(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
) -> tuple[PresentedActionEvidenceCell, ...]:
    """Project prompt-visible empirical deltas for direct G1 rows in a block."""

    option_identities = {value.option_identity_sha256 for value in block.forecasts}
    cells: list[PresentedActionEvidenceCell] = []
    for card in block_request.request.cards:
        card.__post_init__()
        if card.source_binding is None:
            raise SealedReplayError("grounded card lacks source provenance")
        payload = thaw_json(card.prompt_payload)
        if type(payload) is not dict:
            raise SealedReplayError("card prompt payload is not an object")
        empirical = payload.get("empirical_facts")
        if type(empirical) is not list or len(empirical) != 1:
            raise SealedReplayError("Airfoil card must present one empirical fact")
        fact = empirical[0]
        facts = fact.get("facts") if type(fact) is dict else None
        deltas = facts.get("observed_metric_deltas") if type(facts) is dict else None
        if type(deltas) is not list:
            raise SealedReplayError("Airfoil card empirical deltas are malformed")
        for binding in card.finite_action_evidence:
            if binding.option_identity_sha256 not in option_identities:
                continue
            if card.derived_view_receipt is None:
                provenance_kind = (
                    PresentedActionEvidenceProvenanceKind.CARD_SOURCE_RECEIPT
                )
                provenance_sha256 = card.source_binding.source_receipt_sha256
            else:
                provenance_kind = (
                    PresentedActionEvidenceProvenanceKind.CARD_VIEW_RECEIPT
                )
                provenance_sha256 = card.derived_view_receipt.receipt_sha256
            for delta in deltas:
                if type(delta) is not dict:
                    raise SealedReplayError("one presented metric delta is malformed")
                cells.append(
                    PresentedActionEvidenceCell(
                        option_identity_sha256=binding.option_identity_sha256,
                        metric_id=str(delta["metric_id"]),
                        presented_delta=float(delta["child_minus_parent_delta"]),
                        card_key=card.card_key,
                        action_evidence_binding_identity_sha256=(
                            binding.identity_sha256
                        ),
                        provenance_kind=provenance_kind,
                        provenance_sha256=provenance_sha256,
                    )
                )
    return tuple(sorted(cells, key=lambda value: value.sort_key))


@dataclass(frozen=True, slots=True)
class _ArmReplay:
    arm: str
    block_request: ActionForecastBlockRequest
    forecasts: ResolvedActionForecastBlock
    frame_request: FrameActionAllocationRequest
    execution: FrameActionAllocationTreatmentExecution
    health_record: Mapping[str, object]
    subset_health_record: Mapping[str, object]
    evidence_record: Mapping[str, object] | None


def _build_arm_replay_with_utility(
    *,
    arm: str,
    block_request: ActionForecastBlockRequest,
    forecasts: ResolvedActionForecastBlock,
    eligible_rows: tuple[int, ...],
    assignment: Any,
    occurrence: TreatmentOccurrence,
    utility: Any,
) -> _ArmReplay:
    """Build one arm while keeping the benchmark utility explicitly injected."""

    health_policy = lenient_action_forecast_health_policy()
    whole = assess_resolved_action_forecast_block_health(
        block_request,
        forecasts,
        member_id=arm,
        health_policy=health_policy,
    )
    subset = assess_resolved_action_forecast_block_subset_health(
        block_request,
        forecasts,
        member_id=arm,
        health_policy=health_policy,
        subset_policy=pilot.eligible_subset_policy(),
        included_global_row_indices=eligible_rows,
    )
    evidence_cells = _presented_evidence_cells(block_request, forecasts)
    evidence = (
        None
        if not evidence_cells
        else assess_presented_action_evidence_block_consistency(
            block_request,
            forecasts,
            evidence_cells,
        )
    )
    parent_receipts = [whole.receipt_sha256, subset.receipt_sha256]
    if evidence is not None:
        parent_receipts.append(evidence.receipt_sha256)
    frame = bind_action_forecast_block_subset_allocation_frame(
        block_request,
        forecasts,
        included_global_row_indices=eligible_rows,
        subset_policy=_allocation_subset_policy(),
        parent_receipt_sha256s=tuple(sorted(parent_receipts)),
    )
    request = FrameActionAllocationRequest(
        frame=frame,
        eligible_option_ids=tuple(
            sorted(value.option_id for value in frame.forecasts)
        ),
        portfolio_size=PORTFOLIO_SIZE,
        utility=utility,
    )
    allocator = AuditedGreedyForecastFrameAllocator(
        risk_aversion=airfoil.ALLOCATOR_RISK_AVERSION,
        diversity_weight=airfoil.ALLOCATOR_DIVERSITY_WEIGHT,
        score_diagnostic=_allocation_diagnostic(frame),
        gate_policy=_allocation_gate_policy(),
    )
    result = allocator.assess(request)
    execution = FrameActionAllocationTreatmentExecution(
        treatment_assignment=assignment,
        treatment_occurrence=occurrence,
        request=request,
        result=result,
    )
    return _ArmReplay(
        arm=arm,
        block_request=block_request,
        forecasts=forecasts,
        frame_request=request,
        execution=execution,
        health_record=whole.to_record(),
        subset_health_record=subset.to_record(),
        evidence_record=None if evidence is None else evidence.to_record(),
    )


async def _decode_all_v5_blocks(
    bundle: pilot.PilotBundle,
    typed_by_call: Mapping[str, Mapping[str, object]],
) -> tuple[ActionForecastBlockResult, ...]:
    results = await asyncio.gather(
        *(
            _decode_v5_block(
                block_request,
                typed_by_call[block_request.block_call_id.value],
            )
            for block_request in bundle.selected_block_requests
        )
    )
    return tuple(results)


def _actual_member_quality(row: Mapping[str, object]) -> float:
    objectives = row.get("objectives")
    violations = row.get("violations")
    if type(objectives) is not dict or type(violations) is not dict:
        raise SealedReplayError("oracle rank row lacks Airfoil metrics")
    delta_f = float(objectives[OBJECTIVE_NAME]) - float(PARENT_METRICS[OBJECTIVE_NAME])
    delta_v = float(violations[VIOLATION_NAME]) - float(PARENT_METRICS[VIOLATION_NAME])
    return max(-60.0, min(60.0, -delta_v / 0.005)) + 0.05 * math.tanh(
        -delta_f / 0.001
    )


def _actual_member_usefulness(row: Mapping[str, object]) -> float:
    quality = _actual_member_quality(row)
    return (
        1.0 / (1.0 + math.exp(-quality))
        if quality >= 0.0
        else math.exp(quality) / (1.0 + math.exp(quality))
    )


def _actual_member_log_failure(row: Mapping[str, object]) -> float:
    """Return log(sigmoid(-quality)) without forming ``1-usefulness``."""

    quality = _actual_member_quality(row)
    if quality >= 0.0:
        return -quality - math.log1p(math.exp(-quality))
    return -math.log1p(math.exp(quality))


def _actual_set_log_joint_failure(
    option_ids: tuple[str, ...],
    row_by_id: Mapping[str, Mapping[str, object]],
) -> float:
    return float(sum(_actual_member_log_failure(row_by_id[value]) for value in option_ids))


def _actual_set_joint_failure(
    option_ids: tuple[str, ...],
    row_by_id: Mapping[str, Mapping[str, object]],
) -> float:
    return float(math.exp(_actual_set_log_joint_failure(option_ids, row_by_id)))


def _actual_set_utility(
    option_ids: tuple[str, ...],
    row_by_id: Mapping[str, Mapping[str, object]],
) -> float:
    return float(-math.expm1(_actual_set_log_joint_failure(option_ids, row_by_id)))


def _lexicographic_key(row: Mapping[str, object]) -> tuple[float, float]:
    objectives = row.get("objectives")
    violations = row.get("violations")
    if type(objectives) is not dict or type(violations) is not dict:
        raise SealedReplayError("oracle rank row lacks lexicographic metrics")
    return float(violations[VIOLATION_NAME]), float(objectives[OBJECTIVE_NAME])


def _rank_map(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    keys = tuple(_lexicographic_key(value) for value in rows)
    result: dict[str, int] = {}
    for row, key in zip(rows, keys, strict=True):
        option_id = row.get("option_id")
        if type(option_id) is not str:
            raise SealedReplayError("oracle result row lacks option_id")
        result[option_id] = 1 + sum(other < key for other in keys)
    return result


def _average_ranks(values: Sequence[Any]) -> tuple[float, ...]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for index in ordered[start:stop]:
            result[index] = rank
        start = stop
    return tuple(result)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _calibration_record(
    *,
    forecasts: ResolvedActionForecastBlock,
    eligible_option_ids: tuple[str, ...],
    row_by_id: Mapping[str, Mapping[str, object]],
    metric_scales: Mapping[str, float],
) -> dict[str, object]:
    forecast_by_id = {value.option_id: value for value in forecasts.forecasts}
    normalized_errors: list[float] = []
    coverage: list[bool] = []
    directions: list[bool] = []
    predicted_keys: list[tuple[float, float]] = []
    actual_keys: list[tuple[float, float]] = []
    brier: list[float] = []
    for option_id in eligible_option_ids:
        forecast = forecast_by_id[option_id]
        actual = row_by_id[option_id]
        objectives = actual["objectives"]
        violations = actual["violations"]
        assert type(objectives) is dict and type(violations) is dict
        actual_delta_by_metric = {
            airfoil.OBJECTIVE_METRIC_ID: float(objectives[OBJECTIVE_NAME])
            - float(PARENT_METRICS[OBJECTIVE_NAME]),
            airfoil.VIOLATION_METRIC_ID: float(violations[VIOLATION_NAME])
            - float(PARENT_METRICS[VIOLATION_NAME]),
        }
        metric_by_id = {value.metric_id: value for value in forecast.metric_forecasts}
        for metric_id, actual_delta in actual_delta_by_metric.items():
            predicted = metric_by_id[metric_id]
            normalized_errors.append(
                abs(predicted.p50_delta - actual_delta) / metric_scales[metric_id]
            )
            coverage.append(predicted.p10_delta <= actual_delta <= predicted.p90_delta)
            directions.append(
                (predicted.p50_delta == 0.0 and actual_delta == 0.0)
                or (
                    predicted.p50_delta != 0.0
                    and actual_delta != 0.0
                    and (predicted.p50_delta > 0.0) == (actual_delta > 0.0)
                )
            )
        predicted_keys.append(
            (
                metric_by_id[airfoil.VIOLATION_METRIC_ID].p50_delta,
                metric_by_id[airfoil.OBJECTIVE_METRIC_ID].p50_delta,
            )
        )
        actual_keys.append(
            (
                float(violations[VIOLATION_NAME]),
                float(objectives[OBJECTIVE_NAME]),
            )
        )
        brier.append((forecast.probability_valid - 1.0) ** 2)
    return {
        "eligible_action_count": len(eligible_option_ids),
        "metric_cell_count": len(normalized_errors),
        "p50_normalized_mae": fmean(normalized_errors),
        "direction_accuracy": sum(directions) / len(directions),
        "p10_p90_coverage": sum(coverage) / len(coverage),
        "validity_brier_all_cached_actions_valid": fmean(brier),
        "predicted_actual_order_spearman": _pearson(
            _average_ranks(predicted_keys),
            _average_ranks(actual_keys),
        ),
    }


def _posthoc_analysis(
    *,
    oracle_result: Mapping[str, object],
    eligible_option_ids: tuple[str, ...],
    arm_replays: tuple[_ArmReplay, ...],
) -> dict[str, object]:
    raw_rows = oracle_result.get("results")
    if type(raw_rows) is not list or len(raw_rows) != 80 or any(
        type(value) is not dict for value in raw_rows
    ):
        raise SealedReplayError("sealed oracle result lacks the complete 80 rows")
    rows = tuple(value for value in raw_rows if type(value) is dict)
    row_by_id = {str(value["option_id"]): value for value in rows}
    if len(row_by_id) != 80 or not set(eligible_option_ids).issubset(row_by_id):
        raise SealedReplayError("oracle result option coverage changed")
    subset_rows = tuple(row_by_id[value] for value in eligible_option_ids)
    global_ranks = _rank_map(rows)
    subset_ranks = _rank_map(subset_rows)
    all_sets = tuple(combinations(eligible_option_ids, PORTFOLIO_SIZE))
    if len(all_sets) != EXPECTED_THREE_SET_COUNT:
        raise SealedReplayError("authenticated subset no longer has C(18,3)=816")
    failure_by_set = {
        value: _actual_set_joint_failure(value, row_by_id) for value in all_sets
    }
    optimum_set = min(
        all_sets,
        key=lambda value: (failure_by_set[value], value),
    )
    optimum_failure = failure_by_set[optimum_set]
    optimum_utility = _actual_set_utility(optimum_set, row_by_id)
    metric_scales = {
        value.metric_id: value.delta_scale
        for value in arm_replays[0].block_request.request.metric_scales
    }

    arm_records: dict[str, object] = {}
    selected_by_arm: dict[str, tuple[str, ...]] = {}
    for replay in arm_replays:
        selected = tuple(
            value.option_id for value in replay.execution.result.decision.members
        )
        selected_by_arm[replay.arm] = selected
        canonical_selected = tuple(sorted(selected))
        joint_failure = _actual_set_joint_failure(canonical_selected, row_by_id)
        log_joint_failure = _actual_set_log_joint_failure(
            canonical_selected,
            row_by_id,
        )
        utility = _actual_set_utility(canonical_selected, row_by_id)
        best_subset_rank = min(subset_ranks[value] for value in selected)
        best_global_rank = min(global_ranks[value] for value in selected)
        arm_records[replay.arm] = {
            "selected_option_ids_in_allocator_rank_order": list(selected),
            "members": [
                {
                    "allocator_rank": index,
                    "option_id": option_id,
                    "family": row_by_id[option_id]["family"],
                    "global_rank": global_ranks[option_id],
                    "within_subset_rank": subset_ranks[option_id],
                    "objective": row_by_id[option_id]["objectives"],
                    "violation": row_by_id[option_id]["violations"],
                }
                for index, option_id in enumerate(selected, start=1)
            ],
            "best_global_rank": best_global_rank,
            "best_within_subset_rank": best_subset_rank,
            "actual_set_utility": utility,
            "actual_set_joint_failure_probability": joint_failure,
            "actual_set_log_joint_failure": log_joint_failure,
            "actual_set_utility_regret_stable_complement_arithmetic": (
                joint_failure - optimum_failure
            ),
            "exact_uniform_three_set_utility_percentile_0_best": (
                sum(value < joint_failure for value in failure_by_set.values())
                / len(failure_by_set)
            ),
            "exact_uniform_three_set_best_rank_percentile_0_best": (
                sum(
                    min(subset_ranks[item] for item in candidate)
                    < best_subset_rank
                    for candidate in all_sets
                )
                / len(all_sets)
            ),
            "family_diversity": len(
                {str(row_by_id[value]["family"]) for value in selected}
            ),
            "calibration_over_all_18_block_eligible_actions": (
                _calibration_record(
                    forecasts=replay.forecasts,
                    eligible_option_ids=eligible_option_ids,
                    row_by_id=row_by_id,
                    metric_scales=metric_scales,
                )
            ),
        }

    overlaps: dict[str, object] = {}
    for left, right in (("m", "p"), ("m", "n"), ("p", "n")):
        left_set = set(selected_by_arm[left])
        right_set = set(selected_by_arm[right])
        intersection = left_set & right_set
        union = left_set | right_set
        overlaps[f"{left}_{right}"] = {
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union),
            "shared_option_ids": sorted(intersection),
        }

    typed_arm_records = {
        arm: value for arm, value in arm_records.items() if type(value) is dict
    }
    m = typed_arm_records["m"]
    p = typed_arm_records["p"]
    n = typed_arm_records["n"]
    return {
        "schema_version": 1,
        "scope": "postdecision_posthoc_analysis_not_online_information",
        "aggregate_oracle_result_decoded_postdecision": True,
        "aggregate_oracle_row_count": len(rows),
        "aggregate_decode_is_not_a_selected_evaluator_read": True,
        "authenticated_subset_size": len(eligible_option_ids),
        "three_set_count": len(all_sets),
        "actual_set_utility_definition_sha256": (
            ACTUAL_SET_UTILITY_DEFINITION_SHA256
        ),
        "block_optimal_three_set": {
            "option_ids": list(optimum_set),
            "actual_set_utility": optimum_utility,
            "actual_set_joint_failure_probability": optimum_failure,
            "actual_set_log_joint_failure": _actual_set_log_joint_failure(
                optimum_set,
                row_by_id,
            ),
        },
        "arms": arm_records,
        "primary_best_within_subset_rank_contrasts_positive_favors_m": {
            "p_minus_m": int(p["best_within_subset_rank"])
            - int(m["best_within_subset_rank"]),
            "n_minus_m": int(n["best_within_subset_rank"])
            - int(m["best_within_subset_rank"]),
        },
        "secondary_actual_set_utility_contrasts_positive_favors_m": {
            "m_minus_p_stable_complement_arithmetic": float(
                p["actual_set_joint_failure_probability"]
            )
            - float(m["actual_set_joint_failure_probability"]),
            "m_minus_n_stable_complement_arithmetic": float(
                n["actual_set_joint_failure_probability"]
            )
            - float(m["actual_set_joint_failure_probability"]),
        },
        "overlap": overlaps,
    }


def _arm_record(replay: _ArmReplay) -> dict[str, object]:
    result = replay.execution.result
    return {
        "arm": replay.arm,
        "block_request_sha256": replay.block_request.block_request_sha256,
        "v5_forecast_block_receipt_sha256": replay.forecasts.receipt_sha256,
        "whole_health": dict(replay.health_record),
        "eligible_subset_health": dict(replay.subset_health_record),
        "presented_evidence_consistency": (
            None if replay.evidence_record is None else dict(replay.evidence_record)
        ),
        "allocation_frame": replay.frame_request.frame.to_record(),
        "allocation_request": replay.frame_request.to_record(),
        "allocation_result": result.to_record(),
        "allocation_decision": result.decision.to_record(),
        "allocation_audit": result.audit.to_record(),
        "candidate_score_count": result.audit.candidate_score_count,
        "selected_option_ids": [
            value.option_id for value in result.decision.members
        ],
    }


def execute_replay(
    *,
    run_dir: Path = DEFAULT_RUN_DIR,
    source_live_dir: Path = DEFAULT_SOURCE_LIVE_DIR,
    frozen_v2_run: Path = pilot.DEFAULT_FROZEN_V2_RUN,
    oracle_dir: Path = airfoil.DEFAULT_SEALED_ORACLE_DIR,
) -> dict[str, object]:
    """Execute one credential/provider/CFD-free outcome-gated replay."""

    root = _exclusive_directory(run_dir)
    phase = "source_verification"
    outcome_opened = False
    try:
        source_root = source_live_dir.expanduser().resolve(strict=True)
        source_finalization = _verify_source_live(source_root)
        bundle = pilot.build_pilot_bundle(
            frozen_v2_run=frozen_v2_run,
            oracle_dir=oracle_dir,
        )
        if len(bundle.eligible_g2_global_row_indices) != EXPECTED_ELIGIBLE_COUNT:
            raise SealedReplayError("selected block no longer has 18 eligible G2 rows")

        phase = "prospective_replay_assignment"
        experiment = _experiment_record(bundle, source_finalization)
        experiment_sha256 = str(experiment["experiment_commitment_sha256"])
        assignment = _treatment_assignment(bundle, experiment_sha256)
        # This assignment is prospective only to this post-hoc allocation
        # replay.  It makes no retrospective claim about provider dispatch.
        write_json_atomic(
            root / "predecision_protocol.json",
            {
                "experiment": experiment,
                "treatment_assignment": assignment.to_record(),
                "assignment_scope": (
                    "prospective_to_replay_allocation_not_original_provider_wave"
                ),
                "typed_wire_json_decoded_before_this_receipt": False,
                "source_seal_verification_read_opaque_bytes": True,
                "oracle_outcomes_opened_before_this_receipt": False,
            },
        )

        phase = "terminal_provider_ledger"
        terminal_ledger = _terminal_provider_ledger(
            source_live_dir=source_root,
            bundle=bundle,
            source_finalization=source_finalization,
        )
        write_json_atomic(root / "terminal_provider_ledger.json", terminal_ledger)
        terminal_ledger_commitment = str(terminal_ledger["commitment_sha256"])

        phase = "v5_decode_health_and_allocation"
        typed_by_call: dict[str, dict[str, object]] = {}
        # Open immutable wire matrices in the prospective opaque-slot order,
        # then return them to canonical occurrence order through the receipt.
        for assigned in assignment.slot_to_occurrence:
            call_id = assigned.call_identity
            if call_id is None:
                raise SealedReplayError("assigned replay occurrence lacks call binding")
            path = source_root / f"typed_wire_{call_id}.json"
            typed_by_call[call_id] = _load_object(path)

        decoded = asyncio.run(_decode_all_v5_blocks(bundle, typed_by_call))
        # ``asyncio.gather`` above returns exact typed results or raises before
        # allocation.  No partial arm may proceed.
        arm_replays = tuple(
            _build_arm_replay_with_utility(
                arm=arm,
                block_request=block_request,
                forecasts=result.forecasts,
                eligible_rows=bundle.eligible_g2_global_row_indices,
                assignment=assignment,
                occurrence=occurrence,
                utility=bundle.preparation.utility,
            )
            for arm, block_request, result, occurrence in zip(
                ("m", "p", "n"),
                bundle.selected_block_requests,
                decoded,
                assignment.occurrence_input_order,
                strict=True,
            )
        )
        write_json_atomic(
            root / "allocation_attempts.json",
            {
                "schema_version": 1,
                "outcome_opened": False,
                "arms": [_arm_record(value) for value in arm_replays],
            },
        )
        gate_failures = [
            value.arm
            for value in arm_replays
            if (
                value.health_record.get("passes") is not True
                or value.subset_health_record.get("passes") is not True
                or not value.execution.result.audit.passes
                or value.execution.result.audit.candidate_score_count
                != EXPECTED_CANDIDATE_SCORES_PER_ARM
            )
        ]
        if gate_failures:
            raise SealedReplayError(
                "pre-outcome health/allocation gate failed for "
                + ",".join(gate_failures)
            )

        phase = "durable_allocation_commit"
        executions = tuple(value.execution for value in arm_replays)
        phase_commit = build_frame_action_allocation_phase_commit(
            upstream_input_sha256=experiment_sha256,
            terminal_provider_ledger_commitment_sha256=(
                terminal_ledger_commitment
            ),
            executions=executions,
        )
        durable_commit_record = {
            "schema_version": 1,
            "receipt": phase_commit.receipt.to_record(),
            "payload": thaw_json(phase_commit.payload),
        }
        write_json_atomic(
            root / "durable_allocation_phase_commit.json",
            durable_commit_record,
        )
        if _load_object(root / "durable_allocation_phase_commit.json") != (
            durable_commit_record
        ):
            raise SealedReplayError("fsynced allocation commit failed exact readback")
        validate_frame_action_allocation_phase_commit(executions, phase_commit)

        phase = "selected_union_capability"
        benchmark_commitment = airfoil.bind_airfoil_mpn_frame_allocation_commitment(
            bundle.arms,
            executions,
            phase_commit,
        )
        capability = (
            bundle.preparation.evaluator.open_postdecision_evaluation(
                benchmark_commitment
            )
        )
        outcome_opened = True
        unique_evaluations = capability.evaluate_selected()
        unique_by_id = {value.option_id: value for value in unique_evaluations}
        if not 3 <= len(unique_by_id) <= 9:
            raise SealedReplayError("selected cached union escaped the 3--9 bound")
        if set(unique_by_id) != set(benchmark_commitment.selected_option_ids):
            raise SealedReplayError("selected capability returned another cached union")
        logical_slots = [
            {
                "arm": replay.arm,
                "slot": index,
                "option_id": member.option_id,
                "evaluation": unique_by_id[member.option_id].to_record(),
            }
            for replay in arm_replays
            for index, member in enumerate(
                replay.execution.result.decision.members,
                start=1,
            )
        ]
        if len(logical_slots) != EXPECTED_LOGICAL_EVALUATION_SLOTS:
            raise SealedReplayError("logical evaluator-slot accounting changed")
        write_json_atomic(
            root / "selected_outcomes.json",
            {
                "schema_version": 1,
                "benchmark_commitment": benchmark_commitment.to_record(),
                "logical_evaluation_slot_count": len(logical_slots),
                "unique_cached_read_count": len(unique_by_id),
                "unique_cached_option_ids": sorted(unique_by_id),
                "logical_slots": logical_slots,
                "replacement_count": 0,
                "provider_calls": 0,
                "new_cfd_calls": 0,
            },
        )

        phase = "posthoc_analysis"
        oracle_result_path = (
            oracle_dir.expanduser().resolve(strict=True) / "oracle_result.json"
        )
        oracle_result_bytes = oracle_result_path.read_bytes()
        oracle_seal = thaw_json(bundle.preparation.oracle_seal)
        if (
            type(oracle_seal) is not dict
            or hashlib.sha256(oracle_result_bytes).hexdigest()
            != oracle_seal.get("oracle_result_file_sha256")
        ):
            raise SealedReplayError("postdecision oracle aggregate differs from its seal")
        oracle_result = _load_object(oracle_result_path)
        eligible_option_ids = tuple(
            sorted(
                bundle.preparation.contract.options[index].option_id
                for index in bundle.eligible_g2_global_row_indices
            )
        )
        analysis = _posthoc_analysis(
            oracle_result=oracle_result,
            eligible_option_ids=eligible_option_ids,
            arm_replays=arm_replays,
        )
        write_json_atomic(root / "posthoc_analysis.json", analysis)

        phase = "result"
        result: dict[str, object] = {
            "schema_version": 1,
            "status": "completed_posthoc_sealed_wire_v5_allocation_replay",
            "scope": REPLAY_SCOPE,
            "postdecision_analysis_measure_repair_predecessor": {
                "finalization_sha256": PREDECESSOR_REPLAY_FINALIZATION_SHA256,
                "recursive_content_sha256": PREDECESSOR_REPLAY_RECURSIVE_SHA256,
            },
            "experiment_commitment_sha256": experiment_sha256,
            "treatment_assignment_receipt_sha256": assignment.receipt_sha256,
            "terminal_provider_ledger_commitment_sha256": (
                terminal_ledger_commitment
            ),
            "allocation_phase_commit_receipt_sha256": (
                phase_commit.receipt.receipt_sha256
            ),
            "arms": [_arm_record(value) for value in arm_replays],
            "accounting": {
                "new_provider_calls": 0,
                "new_llm_calls": 0,
                "new_cfd_calls": 0,
                "candidate_scores_per_arm": (
                    EXPECTED_CANDIDATE_SCORES_PER_ARM
                ),
                "candidate_scores_total": (
                    3 * EXPECTED_CANDIDATE_SCORES_PER_ARM
                ),
                "logical_evaluation_slots": len(logical_slots),
                "unique_cached_reads": len(unique_by_id),
                "replacement_count": 0,
            },
            "posthoc_analysis": analysis,
            "claim_boundary": (
                "historical-wire developmental replay; no fresh randomization, "
                "causal effect, average treatment effect, significance, or SOTA claim"
            ),
            "allocation_gate_claim_boundary": (
                "the tie/gap thresholds are provisional and post-hoc; passing "
                "does not establish selection robustness"
            ),
        }
        result["result_receipt_sha256"] = _hash(_RESULT_DOMAIN, result)
        write_json_atomic(root / "result.json", result)
        finalization = finalize_run_directory(
            root,
            status="completed_posthoc_sealed_wire_v5_allocation_replay",
        )
        return {"result": result, "finalization": finalization}
    except BaseException as error:
        write_json_atomic(
            root / "failure.json",
            {
                "schema_version": 1,
                "status": "incomplete",
                "failed_phase": phase,
                "failure_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "outcome_capability_opened": outcome_opened,
                "provider_calls_by_replay": 0,
                "new_cfd_calls": 0,
                "replacement_count": 0,
            },
        )
        finalize_run_directory(root, status="incomplete")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--source-live-dir", type=Path, default=DEFAULT_SOURCE_LIVE_DIR
    )
    parser.add_argument("--frozen-v2-run", type=Path, default=pilot.DEFAULT_FROZEN_V2_RUN)
    parser.add_argument("--oracle-dir", type=Path, default=airfoil.DEFAULT_SEALED_ORACLE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    execute_replay(
        run_dir=arguments.run_dir,
        source_live_dir=arguments.source_live_dir,
        frozen_v2_run=arguments.frozen_v2_run,
        oracle_dir=arguments.oracle_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

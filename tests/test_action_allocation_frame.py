from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import permutations
import math
from dataclasses import replace

import pytest

from agent_evolve.application.action_allocation import (
    GreedyRiskAdjustedDiversityAllocator,
)
from agent_evolve.application.action_allocation_frame import (
    AllocationSurfaceGateRejected,
    AuditedGreedyForecastFrameAllocator,
)
from agent_evolve.application.action_allocation_frame_commit import (
    FrameActionAllocationCommitRejected,
    build_frame_action_allocation_phase_commit,
    validate_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_allocation_frame_commit_v3 import (
    OperationalFrameActionAllocationCommitRejected,
    build_operational_frame_action_allocation_phase_commit,
    validate_operational_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_allocation_frame_v3 import (
    OperationalGreedyForecastFrameAllocator,
    OperationalTieAllocationRejected,
)
from agent_evolve.application.paired_allocation_comparison import (
    AllocationComparisonMethodWave,
    build_paired_allocation_comparison_commitment,
    validate_paired_allocation_comparison_commitment,
)
from agent_evolve.application.action_forecast_partitioning import (
    build_action_forecast_block_requests,
    build_action_forecast_partition_layout,
)
from agent_evolve.application.treatment_assignment import (
    assign_treatment_occurrences,
)
from agent_evolve.core.action_semantics import (
    ActionAxisSemantics,
    ActionSpaceSemantics,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.action_allocation import (
    ActionAllocationRequest,
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
)
from agent_evolve.ports.action_allocation_frame import (
    ActionAllocationFrameSubsetPolicyBinding,
    AllocationCandidateScoreDiagnostic,
    AllocationCandidateScoreDiagnosticInput,
    AllocationScoreDiagnosticBinding,
    AllocationSurfaceGatePolicyBinding,
    FrameActionAllocationRequest,
    allocation_score_multiset_sha256,
    bind_action_forecast_block_allocation_frame,
    bind_action_forecast_block_subset_allocation_frame,
    bind_complete_action_forecast_allocation_frame,
    validate_frame_action_portfolio_decision,
)
from agent_evolve.ports.action_allocation_frame_commit import (
    FrameActionAllocationTreatmentExecution,
    frame_source_call_and_request_identity,
)
from agent_evolve.ports.action_allocation_frame_commit_v3 import (
    OperationalFrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.action_allocation_frame_v3 import (
    AllocationScoreResolutionBinding,
    AllocationV3SeedSamplingLaw,
    AllocationV3SelectionBinding,
    AllocationV3TieMode,
    OperationalFrameActionAllocationRequest,
    validate_operational_frame_action_allocation_result,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastDraft,
    ActionForecastEvidenceMode,
    ActionForecastPartitionPolicyBinding,
    ActionForecastRequest,
    ActionMetricForecast,
    MetricForecastScale,
    ParentMetricValue,
    resolve_action_forecast_block,
    resolve_action_forecasts,
)
from agent_evolve.ports.treatment_assignment import (
    OpaqueProviderSlotId,
    TreatmentAssignmentInput,
    TreatmentId,
    TreatmentOccurrence,
    TreatmentOccurrenceId,
)
from agent_evolve.application.two_stage_action_evolution import (
    TwoStageActionPhase,
    TwoStageActionPhaseCommit,
    TwoStageActionPhaseReceipt,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _request(*, namespace: str = "alpha", option_count: int = 20) -> ActionForecastRequest:
    parent = _frozen({"coordinate": 0})
    parent_sha256 = typed_json_sha256(parent)
    family = f"{namespace}_coordinate"
    contract = FiniteVariationContract(
        catalog_id=f"{namespace}_catalog",
        catalog_version=1,
        catalog_definition_sha256=_sha(f"{namespace}-catalog-v1"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=f"{namespace}.action.{index:02d}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen({"coordinate": index + 1}),
                family=family,
                description=f"Choose sealed coordinate {index + 1}.",
            )
            for index in range(option_count)
        ),
    )
    metric_id = f"objective:{namespace}_cost"
    semantics = OptimizationSemantics(
        semantics_id=f"{namespace}_semantics",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id=metric_id,
                name=f"{namespace}_cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="One deterministic fixture cost.",
                aggregation="One scalar.",
                witness_interpretation="Lower is better.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.LEXICOGRAPHIC,
            metric_priority=(metric_id,),
            description="Minimize the fixture cost.",
            equivalence="Equal costs are equivalent.",
            policy_id=f"{namespace}_ordering",
            policy_version=1,
            definition_sha256=_sha(f"{namespace}-ordering-v1"),
        ),
    )
    action_semantics = ActionSpaceSemantics(
        semantics_id=f"{namespace}_action_space",
        semantics_version=1,
        catalog_identities=(
            (
                contract.catalog_id,
                contract.catalog_version,
                contract.catalog_definition_sha256,
            ),
        ),
        axes=(
            ActionAxisSemantics(
                axis_id=f"{namespace}_coordinate",
                configuration_paths=("$.coordinate",),
                option_families=(family,),
                definition="One sealed scalar coordinate intervention.",
                independence="Options are mutually exclusive alternatives.",
                excluded_interpretations=(
                    "Option labels do not reveal metric outcomes.",
                ),
            ),
        ),
    )
    return ActionForecastRequest(
        call_id=LLMCallId(f"call_{namespace}_allocation_frame"),
        operation="forecast_all_actions",
        instruction="Forecast every supplied action and required metric.",
        context=_frozen({"fixture": namespace}),
        optimization_semantics=semantics,
        action_semantics=action_semantics,
        finite_variation_contract=contract,
        cards=(),
        source_registry=None,
        evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY,
        experimental_view_receipt=None,
        parent_metric_values=(ParentMetricValue(metric_id, 100.0),),
        metric_scales=(
            MetricForecastScale(metric_id, 1.0, _sha(f"{namespace}-scale")),
        ),
        temperature=0.0,
    )


def _drafts(request: ActionForecastRequest) -> tuple[ActionForecastDraft, ...]:
    metric_id = request.required_metric_ids[0]
    return tuple(
        ActionForecastDraft(
            option_id=option.option_id,
            probability_valid=0.95,
            metric_forecasts=(
                ActionMetricForecast(
                    metric_id=metric_id,
                    p10_delta=float(index),
                    p50_delta=float(index + 1),
                    p90_delta=float(index + 2),
                    confidence=0.75,
                    citations=(),
                ),
            ),
        )
        for index, option in enumerate(request.finite_variation_contract.options)
    )


def _batch(request: ActionForecastRequest):
    return resolve_action_forecasts(
        request,
        _drafts(request),
        policy_id="fixture_forecast",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-forecast-v1"),
    )


def _block_source(request: ActionForecastRequest):
    policy = ActionForecastPartitionPolicyBinding(
        policy_id="single_fixture_block",
        policy_version=1,
        policy_definition_sha256=_sha("single-fixture-block-v1"),
        max_rows_per_block=20,
        max_metric_cells_per_block=20,
    )
    layout = build_action_forecast_partition_layout(request, policy)
    (block_request,) = build_action_forecast_block_requests(request, layout)
    block = resolve_action_forecast_block(
        block_request,
        _drafts(request),
        policy_id="fixture_forecast",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-forecast-v1"),
    )
    return block_request, block


class _AdditiveUtility:
    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        request.__post_init__()
        values: list[float] = []
        for member in request.members:
            metric = member.metric_forecasts[0]
            values.append(
                {
                    ForecastQuantile.P10: metric.p10_delta,
                    ForecastQuantile.P50: metric.p50_delta,
                    ForecastQuantile.P90: metric.p90_delta,
                }[request.quantile]
            )
        # Lower predicted cost is better; the offset keeps all results finite
        # and positive without changing greedy ordering.
        return float(1000.0 - sum(values))


class _ConstantUtility:
    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        request.__post_init__()
        return 1.0


def _utility(*, saturated: bool = False) -> ForecastPortfolioUtilityBinding:
    return ForecastPortfolioUtilityBinding(
        utility=_ConstantUtility() if saturated else _AdditiveUtility(),
        policy_id="constant_utility" if saturated else "additive_utility",
        policy_version=1,
        definition_sha256=_sha(
            "constant-utility-v1" if saturated else "additive-utility-v1"
        ),
    )


class _BoundaryDiagnostic:
    def __init__(self, *, flag_unit_boundary: bool) -> None:
        self.flag_unit_boundary = flag_unit_boundary

    def __call__(
        self,
        request: AllocationCandidateScoreDiagnosticInput,
    ) -> AllocationCandidateScoreDiagnostic:
        request.__post_init__()
        return AllocationCandidateScoreDiagnostic(
            boundary_or_extreme=(
                self.flag_unit_boundary and request.score.total_utility == 1.0
            )
        )


def _diagnostic(*, flag_unit_boundary: bool = True) -> AllocationScoreDiagnosticBinding:
    return AllocationScoreDiagnosticBinding(
        diagnostic=_BoundaryDiagnostic(flag_unit_boundary=flag_unit_boundary),
        policy_id="fixture_boundary_diagnostic",
        policy_version=1,
        policy_definition_sha256=_sha(
            f"fixture-boundary-diagnostic-{flag_unit_boundary}-v1"
        ),
    )


def _gate() -> AllocationSurfaceGatePolicyBinding:
    return AllocationSurfaceGatePolicyBinding(
        policy_id="strict_fixture_surface",
        policy_version=1,
        policy_definition_sha256=_sha("strict-fixture-surface-v1"),
        minimum_distinct_finite_scores=2,
        maximum_top_tie_share=0.5,
        maximum_boundary_or_extreme_share=0.95,
        minimum_winner_runner_gap=0.25,
    )


def _allocator() -> AuditedGreedyForecastFrameAllocator:
    return AuditedGreedyForecastFrameAllocator(
        risk_aversion=0.0,
        diversity_weight=0.0,
        score_diagnostic=_diagnostic(),
        gate_policy=_gate(),
    )


def _subset_request(*, namespace: str = "alpha") -> FrameActionAllocationRequest:
    request = _request(namespace=namespace)
    block_request, block = _block_source(request)
    frame = bind_action_forecast_block_subset_allocation_frame(
        block_request,
        block,
        included_global_row_indices=tuple(range(1, 19)),
        subset_policy=ActionAllocationFrameSubsetPolicyBinding(
            policy_id="fixture_eligible_rows",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-eligible-rows-v1"),
        ),
        parent_receipt_sha256s=tuple(
            sorted((_sha("block-health"), _sha("eligible-subset-health")))
        ),
    )
    return FrameActionAllocationRequest(
        frame=frame,
        eligible_option_ids=tuple(
            sorted(value.option_id for value in frame.forecasts)
        ),
        portfolio_size=3,
        utility=_utility(),
    )


def _treatment_executions(
    *,
    namespaces: tuple[str, ...] = ("alpha", "beta", "gamma"),
    treatment_ids: tuple[str, ...] = ("m", "p", "n"),
    failing_index: int | None = None,
) -> tuple[FrameActionAllocationTreatmentExecution, ...]:
    if len(namespaces) != len(treatment_ids):
        raise AssertionError("fixture treatment counts differ")
    requests = tuple(_subset_request(namespace=value) for value in namespaces)
    occurrences = tuple(
        TreatmentOccurrence(
            occurrence_id=TreatmentOccurrenceId(f"trial.{index:02d}"),
            treatment_id=TreatmentId(treatment_id),
            call_identity=frame_source_call_and_request_identity(request)[0],
            request_identity_sha256=frame_source_call_and_request_identity(request)[1],
        )
        for index, (treatment_id, request) in enumerate(
            zip(treatment_ids, requests, strict=True)
        )
    )
    assignment = assign_treatment_occurrences(
        TreatmentAssignmentInput(
            experiment_commitment_sha256=_sha("frame-allocation-experiment"),
            public_seed_material="public.seed.frame.allocation",
            occurrences=occurrences,
            provider_slot_ids=tuple(
                OpaqueProviderSlotId(f"opaque.slot.{index:02d}")
                for index in range(len(occurrences))
            ),
        )
    )
    executions: list[FrameActionAllocationTreatmentExecution] = []
    for index, (occurrence, request) in enumerate(
        zip(occurrences, requests, strict=True)
    ):
        effective_request = (
            replace(request, utility=_utility(saturated=True))
            if failing_index == index
            else request
        )
        result = (
            _allocator().assess(effective_request)
            if failing_index == index
            else _allocator().allocate(effective_request)
        )
        executions.append(
            FrameActionAllocationTreatmentExecution(
                treatment_assignment=assignment,
                treatment_occurrence=occurrence,
                request=effective_request,
                result=result,
            )
        )
    return tuple(executions)


def test_complete_and_block_frames_preserve_exact_source_receipts() -> None:
    request = _request()
    batch = _batch(request)
    complete = bind_complete_action_forecast_allocation_frame(
        request,
        batch,
        parent_receipt_sha256s=(_sha("complete-health"),),
    )
    block_request, block = _block_source(request)
    block_frame = bind_action_forecast_block_allocation_frame(
        block_request,
        block,
        parent_receipt_sha256s=(_sha("block-health"),),
    )

    assert complete.source_forecast_receipt_sha256 == batch.receipt_sha256
    assert block_frame.source_forecast_receipt_sha256 == block.receipt_sha256
    assert complete.receipt_sha256 != block_frame.receipt_sha256
    assert complete.to_record()["options"] == block_frame.to_record()["options"]

    with pytest.raises(ValueError, match="unique and canonically sorted"):
        replace(
            block_frame,
            parent_receipt_sha256s=tuple(
                reversed(sorted((_sha("z-parent"), _sha("a-parent"))))
            ),
        )


def test_subset_frame_binds_order_identities_policy_and_parent_receipts() -> None:
    allocation_request = _subset_request()
    frame = allocation_request.frame
    record = frame.to_record()

    assert frame.global_row_indices == tuple(range(1, 19))
    assert len(frame.forecasts) == 18
    assert [value["global_row_index"] for value in record["options"]] == list(
        range(1, 19)
    )
    assert record["parent_receipt_sha256s"] == sorted(
        (_sha("block-health"), _sha("eligible-subset-health"))
    )
    assert record["subset_policy"]["policy_id"] == "fixture_eligible_rows"

    with pytest.raises(ValueError, match="unique and in global order"):
        replace(frame, global_row_indices=tuple(reversed(frame.global_row_indices)))
    with pytest.raises(ValueError, match="authenticated parent receipt"):
        replace(frame, parent_receipt_sha256s=())
    with pytest.raises(ValueError, match="eligible option IDs must be unique"):
        replace(
            allocation_request,
            eligible_option_ids=tuple(reversed(allocation_request.eligible_option_ids)),
        )

    changed = replace(
        frame,
        subset_policy=ActionAllocationFrameSubsetPolicyBinding(
            policy_id="alternate_eligible_rows",
            policy_version=1,
            policy_definition_sha256=_sha("alternate-eligible-rows-v1"),
        ),
    )
    assert changed.receipt_sha256 != frame.receipt_sha256


def test_complete_frame_service_matches_allocator_v2_decision() -> None:
    request = _request(option_count=6)
    batch = _batch(request)
    utility = _utility()
    eligible = tuple(sorted(value.option_id for value in batch.forecasts))
    legacy_request = ActionAllocationRequest(
        forecast_request=request,
        forecasts=batch,
        eligible_option_ids=eligible,
        portfolio_size=3,
        utility=utility,
    )
    legacy = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=0.0,
        diversity_weight=0.0,
    ).allocate(legacy_request)
    frame_request = FrameActionAllocationRequest(
        frame=bind_complete_action_forecast_allocation_frame(request, batch),
        eligible_option_ids=eligible,
        portfolio_size=3,
        utility=utility,
    )
    audited = _allocator().allocate(frame_request)

    assert [value.option_id for value in audited.decision.members] == [
        value.option_id for value in legacy.decision.members
    ]
    assert [value.greedy_step_score for value in audited.decision.members] == [
        value.greedy_step_score for value in legacy.decision.members
    ]
    assert audited.decision.allocator_policy_id == legacy.decision.allocator_policy_id
    assert audited.decision.allocator_policy_version == 2
    assert (
        audited.decision.allocator_configuration_sha256
        == legacy.decision.allocator_configuration_sha256
    )


def test_eighteen_by_three_surface_scores_exactly_fifty_one_extensions() -> None:
    request = _subset_request()
    result = _allocator().allocate(request)

    assert result.audit.passes is True
    assert [value.candidate_count for value in result.audit.steps] == [18, 17, 16]
    assert result.audit.candidate_score_count == 18 + 17 + 16 == 51
    assert result.decision.candidate_evaluations == 51
    assert len(result.decision.members) == 3
    assert all(value.top_tie_count == 1 for value in result.audit.steps)
    assert all(value.winner_runner_gap == 1.0 for value in result.audit.steps)
    assert all(not value.tie_break_used for value in result.audit.steps)
    validate_frame_action_portfolio_decision(request, result.decision)


def test_allocation_request_identity_is_not_rehashed_per_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _subset_request()
    original = FrameActionAllocationRequest.request_sha256.fget
    assert original is not None
    calls = 0

    def counted(value: FrameActionAllocationRequest) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(
        FrameActionAllocationRequest,
        "request_sha256",
        property(counted),
    )
    result = _allocator().allocate(request)

    assert result.decision.candidate_evaluations == 51
    # One local identity at entry and one independent terminal validator call;
    # the 51 candidate diagnostics consume the precomputed digest.
    assert calls == 2


def test_saturated_tied_surface_is_durably_assessed_then_rejected() -> None:
    base = _subset_request()
    request = replace(base, utility=_utility(saturated=True))
    allocator = _allocator()

    attempt = allocator.assess(request)
    assert attempt.audit.passes is False
    assert attempt.audit.candidate_score_count == 51
    assert all(value.distinct_finite_score_count == 1 for value in attempt.audit.steps)
    assert [value.top_tie_count for value in attempt.audit.steps] == [18, 17, 16]
    assert all(value.tie_break_used for value in attempt.audit.steps)
    assert all(value.boundary_or_extreme_share == 1.0 for value in attempt.audit.steps)
    assert all(
        set(value.failure_codes)
        == {
            "boundary_extreme_concentration",
            "insufficient_distinct_scores",
            "top_tie_concentration",
            "winner_runner_gap_too_small",
        }
        for value in attempt.audit.steps
    )

    with pytest.raises(AllocationSurfaceGateRejected) as captured:
        allocator.allocate(request)
    assert captured.value.result == attempt
    assert captured.value.result.to_record()["authorized"] is False


def test_audit_tamper_fails_closed() -> None:
    result = _allocator().allocate(_subset_request())
    first = result.audit.steps[0]

    with pytest.raises(ValueError, match="tie_break_used differs"):
        replace(first, top_tie_count=2)
    with pytest.raises(ValueError, match="candidate_score_count differs"):
        replace(result.audit, candidate_score_count=52)
    with pytest.raises(ValueError, match="audit is bound to another decision"):
        replace(
            result,
            audit=replace(
                result.audit,
                decision_receipt_sha256=_sha("foreign-decision"),
            ),
        )


def test_surface_diagnostics_and_opaque_labels_are_domain_invariant() -> None:
    alpha = _allocator().allocate(_subset_request(namespace="alpha"))
    beta = _allocator().allocate(_subset_request(namespace="beta"))

    # Request and decision receipts correctly retain domain identity.
    assert alpha.decision.receipt_sha256 != beta.decision.receipt_sha256
    assert alpha.audit.receipt_sha256 != beta.audit.receipt_sha256
    # Surface rows contain no option/metric labels and are identical because
    # the ordered numeric allocation problem is identical.
    assert alpha.audit.steps == beta.audit.steps
    assert [value.winner_candidate_label for value in alpha.audit.steps] == [
        "row_00000001",
        "row_00000002",
        "row_00000003",
    ]
    assert all(
        "alpha" not in str(value.to_record()) and "beta" not in str(value.to_record())
        for value in alpha.audit.steps
    )


def test_frame_allocation_commit_binds_complete_ordered_records_and_ledger() -> None:
    executions = _treatment_executions()
    commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("upstream-forecast-wave"),
        terminal_provider_ledger_commitment_sha256=_sha("durable-ledger-file"),
        executions=executions,
    )
    binding = validate_frame_action_allocation_phase_commit(executions, commit)
    payload = thaw_json(commit.payload)

    assert commit.receipt.phase is TwoStageActionPhase.ALLOCATE
    assert commit.receipt.input_sha256 == binding.input_sha256
    assert payload["treatment_occurrence_order"] == [
        value.treatment_occurrence.to_record() for value in executions
    ]
    assert payload["terminal_provider_ledger_commitment"] == {
        "commitment_sha256": _sha("durable-ledger-file"),
        "durability_scope": (
            "external_materialization_and_fsync_required_before_commit"
        ),
        "digest_alone_proves_durability": False,
    }
    assert payload["all_allocation_surface_audits_pass"] is True
    assert payload["evaluator_authority_eligible"] is True
    for raw, execution in zip(
        payload["treatment_executions"], executions, strict=True
    ):
        assert raw["allocation_request"] == execution.request.to_record()
        assert raw["decision"] == execution.result.decision.to_record()
        assert raw["audit"] == execution.result.audit.to_record()
        assert raw["audited_result_receipt_sha256"] == (
            execution.result.receipt_sha256
        )


def test_frame_allocation_commit_supports_arbitrary_repeated_treatments() -> None:
    executions = _treatment_executions(
        namespaces=("alpha", "beta", "gamma", "delta"),
        treatment_ids=("baseline", "memory", "memory", "recombine"),
    )
    commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("four-treatment-input"),
        terminal_provider_ledger_commitment_sha256=_sha("four-treatment-ledger"),
        executions=executions,
    )

    assert len(thaw_json(commit.payload)["treatment_executions"]) == 4
    assert validate_frame_action_allocation_phase_commit(
        executions,
        commit,
    ).treatment_assignment.receipt_sha256 == (
        executions[0].treatment_assignment.receipt_sha256
    )


def test_frame_allocation_commit_rejects_order_identity_and_payload_tampering() -> None:
    executions = _treatment_executions()
    with pytest.raises(ValueError, match="occurrence input order"):
        build_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha("ordered-input"),
            terminal_provider_ledger_commitment_sha256=_sha("ordered-ledger"),
            executions=tuple(reversed(executions)),
        )
    with pytest.raises(
        ValueError,
        match="prospective assignment|another frame source request",
    ):
        replace(
            executions[0],
            treatment_occurrence=replace(
                executions[0].treatment_occurrence,
                request_identity_sha256=_sha("foreign-source-request"),
            ),
        )

    commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("ordered-input"),
        terminal_provider_ledger_commitment_sha256=_sha("ordered-ledger"),
        executions=executions,
    )
    payload = thaw_json(commit.payload)
    payload["treatment_executions"][0]["audit"]["passes"] = False
    frozen = freeze_json(payload)
    tampered = TwoStageActionPhaseCommit(
        receipt=TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.ALLOCATE,
            input_sha256=commit.receipt.input_sha256,
            output_sha256=typed_json_sha256(frozen),
        ),
        payload=frozen,
    )
    with pytest.raises(ValueError, match="differs from exact execution payloads"):
        validate_frame_action_allocation_phase_commit(executions, tampered)


def test_failed_allocation_surface_cannot_authorize_commit() -> None:
    executions = _treatment_executions(failing_index=1)
    assert executions[1].result.audit.passes is False
    with pytest.raises(FrameActionAllocationCommitRejected, match="must pass"):
        build_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha("failed-input"),
            terminal_provider_ledger_commitment_sha256=_sha("failed-ledger"),
            executions=executions,
        )


def test_upstream_and_terminal_ledger_commitments_change_both_phase_hashes() -> None:
    executions = _treatment_executions()
    original = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("input-one"),
        terminal_provider_ledger_commitment_sha256=_sha("ledger-one"),
        executions=executions,
    )
    changed_input = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("input-two"),
        terminal_provider_ledger_commitment_sha256=_sha("ledger-one"),
        executions=executions,
    )
    changed_ledger = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("input-one"),
        terminal_provider_ledger_commitment_sha256=_sha("ledger-two"),
        executions=executions,
    )

    assert len(
        {
            original.receipt.input_sha256,
            changed_input.receipt.input_sha256,
            changed_ledger.receipt.input_sha256,
        }
    ) == 3
    assert len(
        {
            original.receipt.output_sha256,
            changed_input.receipt.output_sha256,
            changed_ledger.receipt.output_sha256,
        }
    ) == 3


class _ScoreSumUtility:
    """Fixture utility whose one-member marginals are exact supplied scores."""

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        request.__post_init__()
        attribute = {
            ForecastQuantile.P10: "p10_delta",
            ForecastQuantile.P50: "p50_delta",
            ForecastQuantile.P90: "p90_delta",
        }[request.quantile]
        return float(
            sum(
                getattr(member.metric_forecasts[0], attribute)
                for member in request.members
            )
        )


class _DoubleUtility:
    """Intentionally different executable hidden behind matching metadata."""

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        return 2.0 * _ScoreSumUtility()(request)


_SCORE_SUM_UTILITY = _ScoreSumUtility()


def _scored_frame_request(
    scores: tuple[float, ...],
    *,
    portfolio_size: int = 1,
    namespace: str = "vthree",
) -> FrameActionAllocationRequest:
    request = _request(namespace=namespace, option_count=len(scores))
    metric_id = request.required_metric_ids[0]
    drafts = tuple(
        ActionForecastDraft(
            option_id=option.option_id,
            probability_valid=0.95,
            metric_forecasts=(
                ActionMetricForecast(
                    metric_id=metric_id,
                    p10_delta=score,
                    p50_delta=score,
                    p90_delta=score,
                    confidence=0.75,
                    citations=(),
                ),
            ),
        )
        for option, score in zip(
            request.finite_variation_contract.options,
            scores,
            strict=True,
        )
    )
    forecasts = resolve_action_forecasts(
        request,
        drafts,
        policy_id="vthree_fixture_forecast",
        policy_version=1,
        policy_definition_sha256=_sha("vthree-fixture-forecast-v1"),
    )
    frame = bind_complete_action_forecast_allocation_frame(request, forecasts)
    return FrameActionAllocationRequest(
        frame=frame,
        eligible_option_ids=tuple(
            sorted(value.option_id for value in frame.forecasts)
        ),
        portfolio_size=portfolio_size,
        utility=ForecastPortfolioUtilityBinding(
            utility=_SCORE_SUM_UTILITY,
            policy_id="vthree_score_sum",
            policy_version=1,
            definition_sha256=_sha("vthree-score-sum-v1"),
        ),
    )


def _operational_request(
    scores: tuple[float, ...],
    *,
    maximum_gap: float,
    mode: AllocationV3TieMode = AllocationV3TieMode.PUBLIC_HASH_RANK,
    seed_sampling_law: AllocationV3SeedSamplingLaw = (
        AllocationV3SeedSamplingLaw.FIXED_PUBLIC
    ),
    public_seed: int = 0xA110CA7E,
    allocation_unit_key: str = "fixture.shared.allocation.unit",
    portfolio_size: int = 1,
    namespace: str = "vthree",
    diversity_weight: float = 0.0,
) -> OperationalFrameActionAllocationRequest:
    return OperationalFrameActionAllocationRequest(
        allocation=_scored_frame_request(
            scores,
            portfolio_size=portfolio_size,
            namespace=namespace,
        ),
        risk_aversion=0.0,
        diversity_weight=diversity_weight,
        score_resolution=AllocationScoreResolutionBinding(
            policy_id="fixture_scientific_resolution",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-scientific-resolution-v1"),
            maximum_indistinguishable_score_gap=maximum_gap,
        ),
        tie_selection=AllocationV3SelectionBinding(
            policy_id="fixture_public_task_rank",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-public-task-rank-v1"),
            mode=mode,
            seed_sampling_law=seed_sampling_law,
            seed_provenance_sha256=_sha("fixture-fixed-public-seed-provenance"),
            public_seed=public_seed,
            allocation_unit_key=allocation_unit_key,
        ),
    )


def _operational_treatment_executions(
    requests: tuple[OperationalFrameActionAllocationRequest, ...],
) -> tuple[OperationalFrameActionAllocationTreatmentExecution, ...]:
    occurrences = tuple(
        TreatmentOccurrence(
            occurrence_id=TreatmentOccurrenceId(f"vthree.trial.{index:02d}"),
            treatment_id=TreatmentId(("m", "p", "n", "x")[index]),
            call_identity=frame_source_call_and_request_identity(
                request.allocation
            )[0],
            request_identity_sha256=frame_source_call_and_request_identity(
                request.allocation
            )[1],
        )
        for index, request in enumerate(requests)
    )
    assignment = assign_treatment_occurrences(
        TreatmentAssignmentInput(
            experiment_commitment_sha256=_sha("allocator-v3-experiment"),
            public_seed_material="allocator.v3.assignment.public.seed",
            occurrences=occurrences,
            provider_slot_ids=tuple(
                OpaqueProviderSlotId(f"opaque.vthree.slot.{index:02d}")
                for index in range(len(occurrences))
            ),
        )
    )
    allocator = OperationalGreedyForecastFrameAllocator()
    return tuple(
        OperationalFrameActionAllocationTreatmentExecution(
            treatment_assignment=assignment,
            treatment_occurrence=occurrence,
            request=request,
            result=allocator.assess(request),
        )
        for occurrence, request in zip(occurrences, requests, strict=True)
    )


def test_allocator_v3_near_tie_uses_benchmark_bound_resolution() -> None:
    runner = math.nextafter(1.0, 0.0)
    exact_gap = 1.0 - runner
    request = _operational_request(
        (1.0, runner, 0.0),
        maximum_gap=exact_gap,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)
    (step,) = result.audit.steps

    assert step.raw_top_tie_count == 1
    assert step.raw_runner_gap == exact_gap
    assert step.operational_top_count == 2
    assert step.random_oracle_prior_seed_law_reference_weight is None
    assert result.audit.score_resolution == request.score_resolution
    assert result.to_record()["authorized"] is True


def test_allocator_v3_operational_threshold_equality_is_inclusive() -> None:
    request = _operational_request(
        (1.0, 0.875, 0.0),
        maximum_gap=0.125,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)
    (step,) = result.audit.steps

    assert step.raw_runner_gap == 0.125
    assert step.raw_top_tie_count == 1
    assert step.operational_top_count == 2
    assert step.operational_top_candidate_labels == (
        "row_00000000",
        "row_00000001",
    )
    assert step.to_record()[
        "random_oracle_prior_seed_law_reference_weight"
    ] is None
    assert request.tie_selection.to_record()["selection_probability_claim"] == (
        "point_mass_after_fixed_public_seed_no_reference_weight"
    )
    assert request.tie_selection.to_record()["seed_provenance_boundary"] == {
        "digest_alone_proves_pre_forecast_preregistration": False,
        "chronology_requires_external_durable_release": True,
    }


def test_allocator_v3_uniform_seed_records_only_prior_reference_weight() -> None:
    request = _operational_request(
        (1.0, 0.875, 0.0),
        maximum_gap=0.125,
        seed_sampling_law=AllocationV3SeedSamplingLaw.UNIFORM_UINT64,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)
    (step,) = result.audit.steps

    assert step.random_oracle_prior_seed_law_reference_weight == "1/2"
    assert step.to_record()[
        "random_oracle_prior_seed_law_reference_weight"
    ] == {
        "numerator": 1,
        "denominator": 2,
        "ratio": "1/2",
        "interpretation": (
            "prior_reference_under_uniform_uint64_random_oracle_model"
        ),
        "is_conditional_propensity": False,
    }
    assert request.tie_selection.to_record()["selection_probability_claim"] == (
        "random_oracle_prior_reference_not_conditional_propensity"
    )


def test_allocator_v3_exact_partial_tie_records_raw_and_operational_sets() -> None:
    request = _operational_request(
        (1.0, 1.0, 0.0, -1.0),
        maximum_gap=0.0,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)
    (step,) = result.audit.steps

    assert step.candidate_count == 4
    assert step.distinct_finite_score_count == 3
    assert step.raw_top_tie_count == 2
    assert step.operational_top_count == 2
    assert step.raw_runner_gap == 0.0
    assert step.raw_top_candidate_labels == step.operational_top_candidate_labels
    selected = next(
        value
        for value in step.candidates
        if value.candidate_label == step.selected_candidate_label
    )
    assert step.selected_public_rank_sha256 == selected.public_rank_sha256
    assert selected.public_rank_sha256 == min(
        value.public_rank_sha256
        for value in step.candidates
        if value.candidate_label in step.operational_top_candidate_labels
    )


def test_allocator_v3_fail_closed_mode_durably_rejects_operational_tie() -> None:
    request = _operational_request(
        (1.0, 1.0, 0.0),
        maximum_gap=0.0,
        mode=AllocationV3TieMode.FAIL_CLOSED,
    )
    allocator = OperationalGreedyForecastFrameAllocator()
    attempt = allocator.assess(request)

    assert attempt.audit.passes is False
    assert attempt.audit.steps[0].failure_codes == (
        "operational_tie_fail_closed",
    )
    assert attempt.to_record()["authorized"] is False
    with pytest.raises(OperationalTieAllocationRejected) as captured:
        allocator.allocate(request)
    assert captured.value.result == attempt


def test_allocator_v3_public_rank_is_permutation_and_concurrency_invariant() -> None:
    request = _operational_request(
        (1.0, 1.0, 1.0, 0.0),
        maximum_gap=0.0,
        portfolio_size=2,
    )
    binding = request.tie_selection
    identities = tuple(
        value.option_identity_sha256
        for value in request.allocation.frame.forecasts[:3]
    )
    expected = min(
        identities,
        key=lambda value: binding.public_rank_sha256(
            step=1,
            option_identity_sha256=value,
        ),
    )
    for order in permutations(identities):
        observed = min(
            order,
            key=lambda value: binding.public_rank_sha256(
                step=1,
                option_identity_sha256=value,
            ),
        )
        assert observed == expected

    allocator = OperationalGreedyForecastFrameAllocator()
    first = allocator.allocate(request)
    second = allocator.allocate(request)
    assert first.to_record() == second.to_record()
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = tuple(
            pool.map(
                lambda _index: allocator.allocate(request).receipt_sha256,
                range(32),
            )
        )
    assert set(receipts) == {first.receipt_sha256}


def test_allocator_v3_supports_arbitrary_n_k_and_zero_diversity() -> None:
    request = _operational_request(
        (7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0),
        maximum_gap=0.0,
        portfolio_size=4,
        diversity_weight=0.0,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)

    assert [value.candidate_count for value in result.audit.steps] == [7, 6, 5, 4]
    assert result.audit.candidate_score_count == 22
    assert len(result.decision.members) == 4
    for step in result.audit.steps:
        for candidate in step.candidates:
            assert candidate.score.diversity_reward == 0.0
            assert candidate.primary_risk_adjusted_utility == (
                candidate.score.p50_utility - candidate.score.risk_penalty
            )
            assert candidate.to_record()["primary_risk_adjusted_utility_hex"] == (
                candidate.primary_risk_adjusted_utility.hex()
            )


def test_allocator_v3_cross_arm_common_key_and_parallel_commit_seam() -> None:
    requests = tuple(
        _operational_request(
            (3.0, 2.0, 1.0),
            maximum_gap=0.0,
            namespace="vthree_common_wave",
        )
        for _index in range(3)
    )
    assert len(
        {value.allocator_configuration_sha256 for value in requests}
    ) == 1
    shared_identity = requests[0].allocation.frame.forecasts[0].option_identity_sha256
    assert len(
        {
            value.tie_selection.public_rank_sha256(
                step=1,
                option_identity_sha256=shared_identity,
            )
            for value in requests
        }
    ) == 1

    executions = _operational_treatment_executions(requests)
    commit = build_operational_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("allocator-v3-upstream"),
        terminal_provider_ledger_commitment_sha256=_sha("allocator-v3-ledger"),
        executions=executions,
    )
    binding = validate_operational_frame_action_allocation_phase_commit(
        executions,
        commit,
    )
    payload = thaw_json(commit.payload)
    assert payload["commit_kind"] == "operational_tie_frame_action_allocation_v3"
    assert payload["evaluator_authority_eligible"] is True
    assert binding.common_allocator_configuration_sha256 == (
        requests[0].allocator_configuration_sha256
    )
    assert binding.to_record()["common_allocation_unit_key"] == (
        "fixture.shared.allocation.unit"
    )
    assert binding.to_record()["common_tie_selection"] == (
        requests[0].tie_selection.to_record()
    )
    assert binding.to_record()["common_tie_selection"]["seed_sampling_law"] == (
        "fixed_public"
    )
    assert binding.to_record()["common_tie_selection"][
        "seed_provenance_sha256"
    ] == _sha("fixture-fixed-public-seed-provenance")
    assert binding.to_record()["common_score_resolution"] == (
        requests[0].score_resolution.to_record()
    )
    assert binding.to_record()["preregistration_chronology_boundary"] == {
        "seed_provenance_digest_alone_proves_pre_forecast_timing": False,
        "external_durable_release_required_for_chronology_claim": True,
        "external_release_must_be_bound_by_upstream_input": True,
        "upstream_input_sha256": _sha("allocator-v3-upstream"),
    }

    # The legacy builder's exact-type boundary remains unchanged.
    with pytest.raises(ValueError, match="exact execution tuple"):
        build_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha("legacy-upstream"),
            terminal_provider_ledger_commitment_sha256=_sha("legacy-ledger"),
            executions=executions,  # type: ignore[arg-type]
        )


def test_allocator_v3_commit_rejects_cross_arm_key_or_mode_drift() -> None:
    requests = (
        _operational_request(
            (3.0, 2.0),
            maximum_gap=0.0,
            namespace="vthree_left",
            allocation_unit_key="shared.left",
        ),
        _operational_request(
            (3.0, 2.0),
            maximum_gap=0.0,
            namespace="vthree_right",
            allocation_unit_key="shared.right",
        ),
    )
    executions = _operational_treatment_executions(requests)
    with pytest.raises(ValueError, match="share resolution|share one allocator-v3"):
        build_operational_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha("allocator-v3-key-drift"),
            terminal_provider_ledger_commitment_sha256=_sha("allocator-v3-ledger"),
            executions=executions,
        )


@pytest.mark.parametrize("drift", ("utility", "eligible_set", "portfolio_size"))
def test_allocator_v3_commit_rejects_mixed_comparison_frames(drift: str) -> None:
    left = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
        namespace="vthree_paired_structure",
    )
    right = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
        namespace="vthree_paired_structure",
        portfolio_size=2 if drift == "portfolio_size" else 1,
    )
    if drift == "utility":
        right = replace(
            right,
            allocation=replace(
                right.allocation,
                utility=ForecastPortfolioUtilityBinding(
                    utility=_SCORE_SUM_UTILITY,
                    policy_id="foreign_score_sum",
                    policy_version=1,
                    definition_sha256=_sha("foreign-score-sum-v1"),
                ),
            ),
        )
    elif drift == "eligible_set":
        right = replace(
            right,
            allocation=replace(
                right.allocation,
                eligible_option_ids=right.allocation.eligible_option_ids[:-1],
            ),
        )
    assert left.allocator_configuration_sha256 == (
        right.allocator_configuration_sha256
    )
    executions = _operational_treatment_executions((left, right))
    with pytest.raises(ValueError, match="share utility, portfolio budget"):
        build_operational_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha(f"allocator-v3-mixed-{drift}"),
            terminal_provider_ledger_commitment_sha256=_sha(
                "allocator-v3-mixed-ledger"
            ),
            executions=executions,
        )


def test_allocator_v3_commit_rejects_same_metadata_different_executable() -> None:
    left = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
        namespace="vthree_executable_identity",
    )
    right = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
        namespace="vthree_executable_identity",
    )
    right = replace(
        right,
        allocation=replace(
            right.allocation,
            utility=ForecastPortfolioUtilityBinding(
                utility=_DoubleUtility(),
                policy_id=left.allocation.utility.policy_id,
                policy_version=left.allocation.utility.policy_version,
                definition_sha256=left.allocation.utility.definition_sha256,
            ),
        ),
    )
    assert left.allocation.utility.to_record() == right.allocation.utility.to_record()
    executions = _operational_treatment_executions((left, right))
    with pytest.raises(ValueError, match="same in-process utility executable"):
        build_operational_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha("allocator-v3-executable-drift"),
            terminal_provider_ledger_commitment_sha256=_sha(
                "allocator-v3-executable-ledger"
            ),
            executions=executions,
        )


def test_allocator_v3_commit_rejects_forecast_policy_drift() -> None:
    left = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
        namespace="vthree_forecast_policy",
    )
    right = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
        namespace="vthree_forecast_policy",
    )
    batch = right.allocation.frame.complete_batch
    assert batch is not None
    foreign_batch = replace(
        batch,
        policy_id="foreign_forecast_policy",
        policy_version=2,
        policy_definition_sha256=_sha("foreign-forecast-policy-v2"),
    )
    foreign_frame = bind_complete_action_forecast_allocation_frame(
        right.allocation.frame.request,
        foreign_batch,
    )
    right = replace(
        right,
        allocation=replace(right.allocation, frame=foreign_frame),
    )
    assert left.allocation.frame.forecasts == right.allocation.frame.forecasts
    executions = _operational_treatment_executions((left, right))
    with pytest.raises(ValueError, match="share utility, portfolio budget"):
        build_operational_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha("allocator-v3-forecast-policy-drift"),
            terminal_provider_ledger_commitment_sha256=_sha(
                "allocator-v3-forecast-policy-ledger"
            ),
            executions=executions,
        )


def test_allocator_v3_tampered_rank_selection_and_request_fail_closed() -> None:
    request = _operational_request(
        (1.0, 1.0, 0.0),
        maximum_gap=0.0,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)
    step = result.audit.steps[0]
    first_candidate = step.candidates[0]
    tampered_candidates = (
        replace(first_candidate, public_rank_sha256=_sha("forged-rank")),
        *step.candidates[1:],
    )
    tampered_step = replace(step, candidates=tampered_candidates)
    with pytest.raises(ValueError, match="public rank differs"):
        replace(result.audit, steps=(tampered_step,))

    alternate_label = next(
        value
        for value in step.operational_top_candidate_labels
        if value != step.selected_candidate_label
    )
    alternate = next(
        value for value in step.candidates if value.candidate_label == alternate_label
    )
    wrong_selection = replace(
        step,
        selected_candidate_label=alternate_label,
        selected_public_rank_sha256=alternate.public_rank_sha256,
    )
    with pytest.raises(ValueError, match="smallest public SHA rank"):
        replace(result.audit, steps=(wrong_selection,))

    foreign_request = replace(
        request,
        tie_selection=replace(
            request.tie_selection,
            public_seed=request.tie_selection.public_seed + 1,
        ),
    )
    with pytest.raises(ValueError, match="another request|configuration"):
        validate_operational_frame_action_allocation_result(foreign_request, result)


def test_allocator_v3_coherent_score_forgery_is_recomputed_and_rejected() -> None:
    request = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)
    step = result.audit.steps[0]
    target = next(
        value for value in step.candidates if value.marginal_total_utility == 1.0
    )
    forged_score = replace(
        target.score,
        p10_utility=1.25,
        p50_utility=1.25,
        p90_utility=1.25,
        downside_utility=1.25,
        total_utility=1.25,
    )
    forged_candidate = replace(
        target,
        score=forged_score,
        marginal_total_utility=1.25,
    )
    forged_candidates = tuple(
        forged_candidate if value.candidate_label == target.candidate_label else value
        for value in step.candidates
    )
    forged_step = replace(
        step,
        candidates=forged_candidates,
        score_multiset_sha256=allocation_score_multiset_sha256(
            tuple(value.marginal_total_utility for value in forged_candidates)
        ),
    )
    # Every stored algebraic summary remains internally coherent.  Only an
    # independent utility rerun over the authenticated frame can detect this.
    forged_audit = replace(result.audit, steps=(forged_step,))
    forged_result = replace(result, audit=forged_audit)
    with pytest.raises(ValueError, match="recomputed benchmark utility"):
        validate_operational_frame_action_allocation_result(
            request,
            forged_result,
        )


def test_allocator_v3_coherent_member_and_final_score_forgery_is_rejected() -> None:
    request = _operational_request(
        (3.0, 2.0, 1.0),
        maximum_gap=0.0,
    )
    result = OperationalGreedyForecastFrameAllocator().allocate(request)
    (member,) = result.decision.members
    forged_score = replace(
        member.greedy_step_score,
        p10_utility=3.5,
        p50_utility=3.5,
        p90_utility=3.5,
        downside_utility=3.5,
        total_utility=3.5,
    )
    forged_member = replace(
        member,
        greedy_step_score=forged_score,
        marginal_total_utility=3.5,
    )
    forged_decision = replace(
        result.decision,
        members=(forged_member,),
        final_score=forged_score,
    )
    forged_audit = replace(
        result.audit,
        decision_receipt_sha256=forged_decision.receipt_sha256,
    )
    forged_result = replace(
        result,
        decision=forged_decision,
        audit=forged_audit,
    )
    with pytest.raises(ValueError, match="audited v3 winner"):
        validate_operational_frame_action_allocation_result(
            request,
            forged_result,
        )


def test_allocator_v3_fail_closed_wave_cannot_authorize_commit() -> None:
    requests = tuple(
        _operational_request(
            (1.0, 1.0, 0.0),
            maximum_gap=0.0,
            mode=AllocationV3TieMode.FAIL_CLOSED,
            namespace=namespace,
        )
        for namespace in ("vthree_fail_m", "vthree_fail_p")
    )
    executions = _operational_treatment_executions(requests)
    assert all(not value.result.audit.passes for value in executions)
    with pytest.raises(OperationalFrameActionAllocationCommitRejected):
        build_operational_frame_action_allocation_phase_commit(
            upstream_input_sha256=_sha("allocator-v3-failed-wave"),
            terminal_provider_ledger_commitment_sha256=_sha("allocator-v3-ledger"),
            executions=executions,
        )


def test_allocator_v3_durable_commit_payload_tamper_is_rejected() -> None:
    requests = tuple(
        _operational_request(
            (3.0, 2.0, 1.0),
            maximum_gap=0.0,
            namespace="vthree_tamper_wave",
        )
        for _index in range(2)
    )
    executions = _operational_treatment_executions(requests)
    commit = build_operational_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("allocator-v3-tamper-upstream"),
        terminal_provider_ledger_commitment_sha256=_sha("allocator-v3-tamper-ledger"),
        executions=executions,
    )
    payload = thaw_json(commit.payload)
    payload["treatment_executions"][0]["audit"]["steps"][0][
        "selected_public_rank_sha256"
    ] = _sha("forged-selected-rank")
    frozen = freeze_json(payload)
    tampered = TwoStageActionPhaseCommit(
        receipt=TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.ALLOCATE,
            input_sha256=commit.receipt.input_sha256,
            output_sha256=typed_json_sha256(frozen),
        ),
        payload=frozen,
    )
    with pytest.raises(ValueError, match="differs from exact executions"):
        validate_operational_frame_action_allocation_phase_commit(
            executions,
            tampered,
        )


def _paired_allocator_method_waves():
    requests = tuple(
        _operational_request(
            (3.0, 2.0, 1.0),
            maximum_gap=0.0,
            namespace="paired_allocator_comparison",
        )
        for _index in range(3)
    )
    v3_executions = _operational_treatment_executions(requests)
    assignment = v3_executions[0].treatment_assignment
    v2_allocator = _allocator()
    v2_executions = tuple(
        FrameActionAllocationTreatmentExecution(
            treatment_assignment=assignment,
            treatment_occurrence=v3.treatment_occurrence,
            request=v3.request.allocation,
            result=v2_allocator.allocate(v3.request.allocation),
        )
        for v3 in v3_executions
    )
    upstream = _sha("paired-allocation-common-upstream")
    ledger = _sha("paired-allocation-common-ledger")
    schedule = _sha("paired-allocation-common-schedule")
    v2_commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=upstream,
        terminal_provider_ledger_commitment_sha256=ledger,
        executions=v2_executions,
    )
    v3_commit = build_operational_frame_action_allocation_phase_commit(
        upstream_input_sha256=upstream,
        terminal_provider_ledger_commitment_sha256=ledger,
        executions=v3_executions,
    )
    return (
        AllocationComparisonMethodWave(
            comparison_method_id="allocator_v2",
            schedule_binding_sha256=schedule,
            executions=v2_executions,
            phase_commit=v2_commit,
        ),
        AllocationComparisonMethodWave(
            comparison_method_id="allocator_v3",
            schedule_binding_sha256=schedule,
            executions=v3_executions,
            phase_commit=v3_commit,
        ),
    )


def test_paired_allocator_comparison_binds_both_methods_before_outcomes() -> None:
    methods = _paired_allocator_method_waves()
    commitment = build_paired_allocation_comparison_commitment(methods)

    assert [value.comparison_method_id for value in commitment.methods] == [
        "allocator_v2",
        "allocator_v3",
    ]
    assert commitment.logical_slot_count == 6
    assert commitment.to_record()["outcomes_read_by_builder"] is False
    replayed = validate_paired_allocation_comparison_commitment(
        methods,
        commitment,
    )
    assert replayed.commitment_sha256 == commitment.commitment_sha256


@pytest.mark.parametrize(
    "drift",
    (
        "duplicate_method",
        "schedule",
        "upstream",
        "ledger",
        "treatment_wave",
        "frame",
        "eligibility",
        "budget",
    ),
)
def test_paired_allocator_comparison_rejects_common_wave_drift(drift: str) -> None:
    v2, v3 = _paired_allocator_method_waves()
    if drift == "duplicate_method":
        v3 = replace(v3, comparison_method_id=v2.comparison_method_id)
    elif drift == "schedule":
        v3 = replace(v3, schedule_binding_sha256=_sha("foreign-schedule"))
    elif drift in {"upstream", "ledger"}:
        v3 = replace(
            v3,
            phase_commit=build_operational_frame_action_allocation_phase_commit(
                upstream_input_sha256=(
                    _sha("foreign-upstream")
                    if drift == "upstream"
                    else _sha("paired-allocation-common-upstream")
                ),
                terminal_provider_ledger_commitment_sha256=(
                    _sha("foreign-ledger")
                    if drift == "ledger"
                    else _sha("paired-allocation-common-ledger")
                ),
                executions=v3.executions,  # type: ignore[arg-type]
            ),
        )
    elif drift == "treatment_wave":
        executions = v3.executions
        occurrences = tuple(value.treatment_occurrence for value in executions)
        assignment = assign_treatment_occurrences(
            TreatmentAssignmentInput(
                experiment_commitment_sha256=_sha("foreign-treatment-wave"),
                public_seed_material="foreign.treatment.wave.seed",
                occurrences=occurrences,
                provider_slot_ids=tuple(
                    OpaqueProviderSlotId(f"foreign.slot.{index:02d}")
                    for index in range(len(occurrences))
                ),
            )
        )
        changed = tuple(
            replace(value, treatment_assignment=assignment)
            for value in executions
        )
        v3 = replace(
            v3,
            executions=changed,
            phase_commit=build_operational_frame_action_allocation_phase_commit(
                upstream_input_sha256=_sha(
                    "paired-allocation-common-upstream"
                ),
                terminal_provider_ledger_commitment_sha256=_sha(
                    "paired-allocation-common-ledger"
                ),
                executions=changed,
            ),
        )
    else:
        allocator = OperationalGreedyForecastFrameAllocator()
        changed_executions = []
        for execution in v3.executions:
            base = execution.request.allocation
            if drift == "eligibility":
                base = replace(
                    base,
                    eligible_option_ids=base.eligible_option_ids[:-1],
                )
            elif drift == "budget":
                base = replace(base, portfolio_size=2)
            else:
                batch = base.frame.complete_batch
                assert batch is not None
                foreign_batch = replace(
                    batch,
                    policy_id="paired_foreign_forecast",
                    policy_version=2,
                    policy_definition_sha256=_sha(
                        "paired-foreign-forecast-v2"
                    ),
                )
                base = replace(
                    base,
                    frame=bind_complete_action_forecast_allocation_frame(
                        base.frame.request,
                        foreign_batch,
                    ),
                )
            request = replace(execution.request, allocation=base)
            changed_executions.append(
                replace(
                    execution,
                    request=request,
                    result=allocator.allocate(request),
                )
            )
        changed_tuple = tuple(changed_executions)
        v3 = replace(
            v3,
            executions=changed_tuple,
            phase_commit=build_operational_frame_action_allocation_phase_commit(
                upstream_input_sha256=_sha(
                    "paired-allocation-common-upstream"
                ),
                terminal_provider_ledger_commitment_sha256=_sha(
                    "paired-allocation-common-ledger"
                ),
                executions=changed_tuple,
            ),
        )
    with pytest.raises(ValueError, match="method IDs|common"):
        build_paired_allocation_comparison_commitment((v2, v3))


def test_paired_allocator_comparison_rejects_post_commit_drift() -> None:
    methods = _paired_allocator_method_waves()
    commitment = build_paired_allocation_comparison_commitment(methods)
    v2, v3 = methods
    drifted = replace(v3, schedule_binding_sha256=_sha("post-commit-drift"))

    with pytest.raises(ValueError, match="common"):
        validate_paired_allocation_comparison_commitment(
            (v2, drifted),
            commitment,
        )

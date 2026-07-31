from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import pytest

from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContext,
    AcquisitionCertifiedSlateContextRegistry,
    AcquisitionCertifiedSlatePolicy,
)
from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlate,
    CalibratedSlateMember,
    MetricOptimizationGoal,
    SlateAllocationRequest,
    SlateAllocationRole,
    SlateMetricObjective,
    SlateRoleProposal,
    SlateStructuralEvidence,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
)
from agent_evolve.policies.selection.regret_bounded_slate import (
    RegretBoundedSlatePolicy,
    ResidualInformationAssayValuePolicy,
    SlateFutureValueAuthority,
    ZeroSlateFutureValuePolicy,
)
from agent_evolve.ports.agentic_generator import (
    MetricEffectDirection,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
)
from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScoreDecision,
    FiniteAcquisitionBatchScoreRequest,
    FiniteAcquisitionSlateScore,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _AdditiveFakeBatchScorer:
    weights: tuple[tuple[str, float], ...]
    policy_id: str = "fixture_additive_batch_scorer"
    policy_version: int = 1
    definition_sha256: str = _sha("fixture additive batch scorer v1")

    def score(
        self,
        request: FiniteAcquisitionBatchScoreRequest,
    ) -> FiniteAcquisitionBatchScoreDecision:
        weight_by_id = dict(self.weights)
        return FiniteAcquisitionBatchScoreDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            scores=tuple(
                FiniteAcquisitionSlateScore(
                    slate=slate,
                    log_acquisition_value=float(
                        sum(weight_by_id[value] for value in slate.candidate_ids)
                    ),
                )
                for slate in request.slates
            ),
        )


def _request_and_context() -> tuple[
    SlateAllocationRequest,
    AcquisitionCertifiedSlateContext,
]:
    scope = ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("prompt"),
        selector_policy_definition_sha256=_sha("selector"),
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
    )
    decision_sha256 = _sha("model decision")
    parent_sha256 = _sha("parent")
    contract_sha256 = _sha("finite contract")
    archive_sha256 = _sha("archive")
    option_ids = tuple(
        [f"anchor.{index}" for index in range(1, 5)]
        + [f"residual.{index}" for index in range(1, 5)]
    )
    members: list[CalibratedSlateMember] = []
    candidates: list[FiniteAcquisitionCandidate] = []
    for rank, option_id in enumerate(option_ids, start=1):
        identity = _sha(f"identity:{option_id}")
        predictions = tuple(
            ForecastPredictionReceipt(
                scope=scope,
                wave_index=1,
                selector_decision_sha256=decision_sha256,
                parent_candidate_identity_sha256=parent_sha256,
                option_id=option_id,
                option_identity_sha256=identity,
                family=f"family.f{rank}",
                metric_id=metric_id,
                asserted_direction=MetricEffectDirection.UNKNOWN,
                confidence=ForecastConfidenceBin.UNKNOWN,
            )
            for metric_id in ("cost", "quality")
        )
        members.append(
            CalibratedSlateMember(
                model_rank=rank,
                option_id=option_id,
                option_identity_sha256=identity,
                family=f"family.f{rank}",
                locus_key=f"locus.l{rank}",
                phenotype_identity_sha256=_sha(f"phenotype:{option_id}"),
                supporting_card_keys=(),
                role_proposal=SlateRoleProposal.EXPLOIT,
                rationale_sha256=_sha(f"rationale:{option_id}"),
                predictions=predictions,
                structural_evidence=SlateStructuralEvidence(
                    frozen_archive_snapshot_sha256=archive_sha256,
                    evidence_receipt_sha256=_sha(f"evidence:{option_id}"),
                    archive_novelty_score=rank / 8.0,
                    structural_coverage_score=(9 - rank) / 8.0,
                ),
            )
        )
        candidates.append(
            FiniteAcquisitionCandidate(
                candidate_id=option_id,
                configuration_sha256=_sha(f"configuration:{option_id}"),
                features=(rank / 10.0, (rank + 1) / 10.0),
            )
        )
    slate = CalibratedSlate(
        scope=scope,
        wave_index=1,
        selector_decision_sha256=decision_sha256,
        parent_candidate_identity_sha256=parent_sha256,
        finite_contract_sha256=contract_sha256,
        members=tuple(members),
    )
    request = SlateAllocationRequest(
        slate=slate,
        portfolio_size=4,
        objectives=tuple(
            SlateMetricObjective(
                metric_id=metric_id,
                goal=goal,
                weight=1.0,
                definition_sha256=_sha(f"objective:{metric_id}"),
            )
            for metric_id, goal in (
                ("cost", MetricOptimizationGoal.MINIMIZE),
                ("quality", MetricOptimizationGoal.MAXIMIZE),
            )
        ),
        assigned_card_keys=(),
    )
    context = AcquisitionCertifiedSlateContext(
        campaign_scope_sha256=_sha("campaign"),
        finite_contract_sha256=contract_sha256,
        cutoff_index=1,
        seed=17,
        objectives=(
            FiniteAcquisitionObjective("cost", "min", 0.0, 10.0),
            FiniteAcquisitionObjective("quality", "max", 10.0, 0.0),
        ),
        observations=(
            FiniteAcquisitionObservation(
                candidate_id="observed.1",
                configuration_sha256=_sha("observed configuration"),
                features=(0.0, 0.0),
                objectives=(("cost", 5.0), ("quality", 5.0)),
            ),
        ),
        candidates=tuple(candidates),
        reference_option_ids=tuple(sorted(option_ids[:4])),
    )
    return request, context


def _policy(
    context: AcquisitionCertifiedSlateContext,
    weights: tuple[tuple[str, float], ...],
) -> AcquisitionCertifiedSlatePolicy:
    registry = AcquisitionCertifiedSlateContextRegistry()
    registry.register(context)
    return AcquisitionCertifiedSlatePolicy(
        context_provider=registry,
        scorer=_AdditiveFakeBatchScorer(weights),
    )


def test_certified_residual_replaces_only_when_it_beats_complete_anchor() -> None:
    request, context = _request_and_context()
    weights = tuple(
        (value.option_id, float(9 - value.model_rank))
        for value in request.slate.members
    )
    weights = tuple(
        (option_id, 20.0 if option_id == "residual.1" else value)
        for option_id, value in weights
    )

    decision = _policy(context, weights).select(request)

    assert decision.reference_option_ids == (
        "anchor.1",
        "anchor.2",
        "anchor.3",
        "anchor.4",
    )
    assert decision.selected_option_ids == (
        "anchor.1",
        "anchor.2",
        "anchor.3",
        "residual.1",
    )
    assert decision.certificate_margin == 15.0
    assert decision.reference_member_count == 3
    assert decision.feasible_slate_count == 70
    assert {value.role for value in decision.selected} == {
        SlateAllocationRole.ACQUISITION_CERTIFIED
    }
    decision.revalidate()


def test_complete_anchor_is_retained_on_an_acquisition_tie() -> None:
    request, context = _request_and_context()
    weights = tuple((value.option_id, 1.0) for value in request.slate.members)

    decision = _policy(context, weights).select(request)

    assert decision.selected_option_ids == decision.reference_option_ids
    assert decision.certificate_margin == 0.0
    assert decision.reference_member_count == request.portfolio_size
    decision.revalidate()


def test_context_registry_allows_exact_replay_and_rejects_conflicting_rebind() -> None:
    _, context = _request_and_context()
    registry = AcquisitionCertifiedSlateContextRegistry()

    registry.register_many((context, context))
    registry.register(context)

    assert registry.context_for(context.finite_contract_sha256) == context
    assert registry.to_record()["registered_context_count"] == 1

    conflicting = replace(context, seed=context.seed + 1)
    try:
        registry.register(conflicting)
    except ValueError as error:
        assert "append-only" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("conflicting context replacement was accepted")
    assert registry.context_for(context.finite_contract_sha256) == context


def _rbie_policy(
    context: AcquisitionCertifiedSlateContext,
    weights: tuple[tuple[str, float], ...],
    *,
    retention: float,
    assay_value: float | None = None,
    allow_assay: bool = False,
    calibration_error_bound: float | None = None,
    minimum_residual_audit_members: int = 0,
) -> RegretBoundedSlatePolicy:
    registry = AcquisitionCertifiedSlateContextRegistry()
    registry.register(context)
    future = (
        ZeroSlateFutureValuePolicy()
        if assay_value is None
        else ResidualInformationAssayValuePolicy(assay_value)
    )
    return RegretBoundedSlatePolicy(
        context_provider=registry,
        scorer=_AdditiveFakeBatchScorer(weights),
        future_value_policy=future,
        minimum_acquisition_retention_ratio=retention,
        minimum_residual_audit_members=minimum_residual_audit_members,
        calibration_error_bound=calibration_error_bound,
        allow_development_assay=allow_assay,
    )


def _small_gap_weights(
    request: SlateAllocationRequest,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (
            value.option_id,
            1.0
            if value.option_id.startswith("anchor.")
            else 0.95
            if value.option_id == "residual.1"
            else 0.0,
        )
        for value in request.slate.members
    )


def test_rbie_buys_one_residual_only_inside_explicit_assay_envelope() -> None:
    request, context = _request_and_context()
    decision = _rbie_policy(
        context,
        _small_gap_weights(request),
        retention=0.95,
        assay_value=0.06,
        allow_assay=True,
        calibration_error_bound=0.02,
    ).select(request)

    assert decision.selected_option_ids == (
        "anchor.1",
        "anchor.2",
        "anchor.3",
        "residual.1",
    )
    assert decision.reference_member_count == 3
    assert decision.acquisition_regret == pytest.approx(0.05)
    assert decision.acquisition_retention_ratio == pytest.approx(math.exp(-0.05))
    assert decision.acquisition_retention_ratio >= 0.95
    assert decision.selected_future_value.authority is (
        SlateFutureValueAuthority.DEVELOPMENT_ASSAY
    )
    assert decision.selected_broker_value == pytest.approx(0.01)
    assert decision.conditional_return_gap_lower_bound == pytest.approx(-0.09)
    assert {value.role for value in decision.selected} == {
        SlateAllocationRole.REGRET_BOUNDED_INFORMATION
    }
    decision.revalidate()


def test_rbie_forced_residual_audit_buys_identification_without_assay_value() -> None:
    request, context = _request_and_context()
    decision = _rbie_policy(
        context,
        _small_gap_weights(request),
        retention=0.95,
        minimum_residual_audit_members=1,
    ).select(request)

    assert decision.selected_option_ids == (
        "anchor.1",
        "anchor.2",
        "anchor.3",
        "residual.1",
    )
    assert decision.minimum_residual_audit_members == 1
    assert decision.reference_member_count == 3
    assert decision.selected_future_value.authority is SlateFutureValueAuthority.ZERO
    assert decision.to_record()["selected_residual_member_count"] == 1
    decision.revalidate()


def test_rbie_rejects_same_residual_outside_tighter_envelope() -> None:
    request, context = _request_and_context()
    decision = _rbie_policy(
        context,
        _small_gap_weights(request),
        retention=0.98,
        assay_value=1.0,
        allow_assay=True,
    ).select(request)

    assert decision.selected_option_ids == decision.reference_option_ids
    assert decision.acquisition_regret == 0.0
    assert decision.acquisition_retention_ratio == 1.0
    decision.revalidate()


def test_rbie_development_value_requires_explicit_authority() -> None:
    request, context = _request_and_context()
    with pytest.raises(ValueError, match="development-only"):
        _rbie_policy(
            context,
            _small_gap_weights(request),
            retention=0.95,
            assay_value=0.06,
            allow_assay=False,
        ).select(request)


def test_rbie_zero_future_value_recovers_acquisition_certified_choice() -> None:
    request, context = _request_and_context()
    weights = tuple(
        (
            value.option_id,
            20.0 if value.option_id == "residual.1" else float(9 - value.model_rank),
        )
        for value in request.slate.members
    )
    decision = _rbie_policy(
        context,
        weights,
        retention=1.0,
    ).select(request)

    assert decision.selected_option_ids == (
        "anchor.1",
        "anchor.2",
        "anchor.3",
        "residual.1",
    )
    assert decision.acquisition_regret == 0.0
    assert decision.selected_future_value.authority is SlateFutureValueAuthority.ZERO
    decision.revalidate()

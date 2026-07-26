from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import agent_evolve.policies.selection.calibrated_slate as calibrated_slate_module
from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlate,
    CalibratedSlateMember,
    MetricOptimizationGoal,
    SlateAllocationMode,
    SlateAllocationRequest,
    SlateAllocationRole,
    SlateMetricObjective,
    SlateRoleProposal,
    SlateStructuralEvidence,
    TraceCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.model_anchored_slate import (
    ModelAnchoredCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    StructuralPosteriorSlatePolicy,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationObservation,
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
    MeaningfulDirectionAdjudicationReceipt,
    MeaningfulDirectionRequest,
    build_calibration_snapshot,
    observe_forecast,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseCardSupport,
    PortfolioMemoryDoseMember,
    assess_proposed_portfolio_memory_dose,
)


FIXTURE = Path(__file__).with_name("fixtures") / (
    "boils_repaired_g6_direction_cells_v1.json"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scope(label: str = "fixture") -> ForecastCalibrationScope:
    return ForecastCalibrationScope(
        model_profile_sha256=_sha(f"{label}:model"),
        prompt_definition_sha256=_sha(f"{label}:prompt"),
        selector_policy_definition_sha256=_sha(f"{label}:selector"),
        benchmark_sha256=_sha(f"{label}:benchmark"),
        session_sha256=_sha(f"{label}:session"),
    )


def _categorical_observation(
    *,
    scope: ForecastCalibrationScope,
    wave: int,
    decision_sha256: str,
    parent_sha256: str,
    option_id: str,
    option_sha256: str,
    family: str,
    child_outcome_sha256: str,
    metric_id: str,
    predicted: str,
    actual: str,
    confidence: ForecastConfidenceBin = ForecastConfidenceBin.UNKNOWN,
) -> ForecastCalibrationObservation:
    prediction = ForecastPredictionReceipt(
        scope=scope,
        wave_index=wave,
        selector_decision_sha256=decision_sha256,
        parent_candidate_identity_sha256=parent_sha256,
        option_id=option_id,
        option_identity_sha256=option_sha256,
        family=family,
        metric_id=metric_id,
        asserted_direction=MetricEffectDirection(predicted),
        confidence=confidence,
    )
    adjudication = MeaningfulDirectionAdjudicationReceipt(
        request_sha256=_sha(
            f"request:{wave}:{decision_sha256}:{option_sha256}:{metric_id}"
        ),
        benchmark_sha256=scope.benchmark_sha256,
        session_sha256=scope.session_sha256,
        wave_index=wave,
        parent_candidate_identity_sha256=parent_sha256,
        option_id=option_id,
        option_identity_sha256=option_sha256,
        metric_id=metric_id,
        parent_outcome_sha256=_sha(f"parent-outcome:{parent_sha256}"),
        child_outcome_sha256=child_outcome_sha256,
        actual_direction=MetricEffectDirection(actual),
        adjudicator_policy_id="fixture_exact_sign",
        adjudicator_policy_version=1,
        adjudicator_definition_sha256=_sha("fixture exact sign adjudicator"),
    )
    return ForecastCalibrationObservation(prediction, adjudication)


def _fixture_observations() -> tuple[
    dict[str, object],
    ForecastCalibrationScope,
    tuple[ForecastCalibrationObservation, ...],
]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    scope = _scope()
    observations: list[ForecastCalibrationObservation] = []
    for candidate in data["candidates"]:
        for metric in candidate["metrics"]:
            observations.append(
                _categorical_observation(
                    scope=scope,
                    wave=candidate["wave_index"],
                    decision_sha256=candidate["selector_decision_sha256"],
                    parent_sha256=candidate["parent_candidate_identity_sha256"],
                    option_id=candidate["option_id"],
                    option_sha256=candidate["option_identity_sha256"],
                    family=candidate["family"],
                    child_outcome_sha256=candidate["child_outcome_sha256"],
                    metric_id=metric["metric_id"],
                    predicted=metric["predicted"],
                    actual=metric["actual"],
                )
            )
    return data, scope, tuple(observations)


@dataclass(frozen=True, slots=True)
class _ThresholdAdjudicator:
    policy_id: str = "portable_threshold_direction"
    policy_version: int = 1
    definition_sha256: str = _sha("portable threshold direction v1")
    threshold: float = 0.25

    def adjudicate(
        self, request: MeaningfulDirectionRequest
    ) -> MeaningfulDirectionAdjudicationReceipt:
        delta = request.child_metric_value - request.parent_metric_value
        if delta > self.threshold:
            direction = MetricEffectDirection.INCREASE
        elif delta < -self.threshold:
            direction = MetricEffectDirection.DECREASE
        else:
            direction = MetricEffectDirection.UNCHANGED
        return MeaningfulDirectionAdjudicationReceipt(
            request_sha256=request.request_sha256,
            benchmark_sha256=request.benchmark_sha256,
            session_sha256=request.session_sha256,
            wave_index=request.wave_index,
            parent_candidate_identity_sha256=(request.parent_candidate_identity_sha256),
            option_id=request.option_id,
            option_identity_sha256=request.option_identity_sha256,
            metric_id=request.metric_id,
            parent_outcome_sha256=request.parent_outcome_sha256,
            child_outcome_sha256=request.child_outcome_sha256,
            actual_direction=direction,
            adjudicator_policy_id=self.policy_id,
            adjudicator_policy_version=self.policy_version,
            adjudicator_definition_sha256=self.definition_sha256,
        )


def test_benchmark_injected_adjudicator_stores_only_categorical_observation() -> None:
    scope = _scope("adjudicator")
    prediction = ForecastPredictionReceipt(
        scope=scope,
        wave_index=1,
        selector_decision_sha256=_sha("decision"),
        parent_candidate_identity_sha256=_sha("parent"),
        option_id="option.alpha",
        option_identity_sha256=_sha("option alpha"),
        family="family.alpha",
        metric_id="objective:quality",
        asserted_direction=MetricEffectDirection.UNCHANGED,
        confidence=ForecastConfidenceBin.HIGH,
    )
    request = MeaningfulDirectionRequest(
        benchmark_sha256=scope.benchmark_sha256,
        session_sha256=scope.session_sha256,
        wave_index=1,
        parent_candidate_identity_sha256=_sha("parent"),
        option_id="option.alpha",
        option_identity_sha256=_sha("option alpha"),
        metric_id="objective:quality",
        parent_outcome_sha256=_sha("parent outcome"),
        child_outcome_sha256=_sha("child outcome"),
        parent_metric_value=4.0,
        child_metric_value=4.2,
    )

    observation = observe_forecast(prediction, request, _ThresholdAdjudicator())

    assert observation.correctness is True
    record = observation.to_record()
    assert "parent_metric_value" not in json.dumps(record)
    assert "child_metric_value" not in json.dumps(record)
    with pytest.raises(ValueError, match="do not join"):
        ForecastCalibrationObservation(
            replace(prediction, option_id="option.foreign"),
            observation.adjudication,
        )


def test_exact_repaired_trace_replays_twenty_of_forty_six_calibration() -> None:
    data, scope, observations = _fixture_observations()
    assert data["source_summary_sha256"] == (
        "313bdaa64c7f5ffb0beb971dcf6bb37e04ca5dbd7c1991e2ec2c5af2766a98d3"
    )
    snapshot = build_calibration_snapshot(
        observations,
        scope=scope,
        cutoff_wave_index_exclusive=6,
    )

    assert snapshot.observation_count == data["expected"]["observation_count"] == 48
    assert snapshot.abstention_count == data["expected"]["abstention_count"] == 2
    assert snapshot.scorable_count == data["expected"]["scorable_count"] == 46
    assert snapshot.correct_count == data["expected"]["correct_count"] == 20
    assert snapshot.empirical_accuracy == pytest.approx(20 / 46)
    assert snapshot.to_record()["leakage_guard"] == (
        "only_observation_wave_lt_exclusive_cutoff"
    )


def _slate_request(
    *,
    scope: ForecastCalibrationScope,
    snapshot,
    wave: int,
    metric_goals: tuple[tuple[str, MetricOptimizationGoal], ...],
    confidence: ForecastConfidenceBin = ForecastConfidenceBin.UNKNOWN,
) -> SlateAllocationRequest:
    decision_sha = _sha(f"slate decision {wave}")
    parent_sha = _sha(f"slate parent {wave}")
    archive_sha = _sha(f"archive before {wave}")
    directions = (
        (MetricEffectDirection.DECREASE, MetricEffectDirection.INCREASE),
        (MetricEffectDirection.UNCHANGED, MetricEffectDirection.INCREASE),
        (MetricEffectDirection.DECREASE, MetricEffectDirection.UNCHANGED),
        (MetricEffectDirection.INCREASE, MetricEffectDirection.DECREASE),
        (MetricEffectDirection.DECREASE, MetricEffectDirection.INCREASE),
        (MetricEffectDirection.UNCHANGED, MetricEffectDirection.DECREASE),
        (MetricEffectDirection.INCREASE, MetricEffectDirection.INCREASE),
        (MetricEffectDirection.DECREASE, MetricEffectDirection.DECREASE),
    )
    members: list[CalibratedSlateMember] = []
    for index, direction_pair in enumerate(directions, start=1):
        option_id = f"option.slate{index}"
        option_sha = _sha(option_id)
        family = f"family.f{index % 5}"
        predictions = tuple(
            ForecastPredictionReceipt(
                scope=scope,
                wave_index=wave,
                selector_decision_sha256=decision_sha,
                parent_candidate_identity_sha256=parent_sha,
                option_id=option_id,
                option_identity_sha256=option_sha,
                family=family,
                metric_id=metric_id,
                asserted_direction=direction,
                confidence=confidence,
            )
            for (metric_id, _), direction in zip(metric_goals, direction_pair)
        )
        cards = (
            ("card.a", "card.b")
            if index == 3
            else ("card.a",)
            if index % 2
            else ("card.b",)
        )
        members.append(
            CalibratedSlateMember(
                model_rank=index,
                option_id=option_id,
                option_identity_sha256=option_sha,
                family=family,
                locus_key=f"locus.l{index}",
                phenotype_identity_sha256=_sha(f"phenotype {index}"),
                supporting_card_keys=cards,
                role_proposal=(
                    SlateRoleProposal.EXPLOIT
                    if index <= 3
                    else SlateRoleProposal.FALSIFY
                    if index <= 6
                    else SlateRoleProposal.COVERAGE
                ),
                rationale_sha256=_sha(f"rationale {index}"),
                predictions=predictions,
                structural_evidence=SlateStructuralEvidence(
                    frozen_archive_snapshot_sha256=archive_sha,
                    evidence_receipt_sha256=_sha(f"structure {index}"),
                    archive_novelty_score=index / 8.0,
                    structural_coverage_score=(9 - index) / 8.0,
                ),
            )
        )
    slate = CalibratedSlate(
        scope=scope,
        wave_index=wave,
        selector_decision_sha256=decision_sha,
        parent_candidate_identity_sha256=parent_sha,
        finite_contract_sha256=_sha(f"contract {wave}"),
        members=tuple(members),
    )
    objectives = tuple(
        SlateMetricObjective(
            metric_id=metric_id,
            goal=goal,
            weight=1.0,
            definition_sha256=_sha(f"objective {metric_id} {goal.value}"),
        )
        for metric_id, goal in metric_goals
    )
    return SlateAllocationRequest(
        slate=slate,
        portfolio_size=4,
        objectives=objectives,
        assigned_card_keys=("card.a", "card.b"),
        calibration_snapshot=snapshot,
    )


def test_repaired_trace_prior_snapshot_drives_audited_k8_allocation() -> None:
    _, scope, observations = _fixture_observations()
    prior = build_calibration_snapshot(
        observations,
        scope=scope,
        cutoff_wave_index_exclusive=5,
    )
    assert prior.observation_count == 32
    assert {value.prediction.wave_index for value in prior.observations} == {1, 3}
    request = _slate_request(
        scope=scope,
        snapshot=prior,
        wave=5,
        metric_goals=(
            ("total_levels", MetricOptimizationGoal.MINIMIZE),
            ("total_lut_count", MetricOptimizationGoal.MINIMIZE),
        ),
    )

    default = TraceCalibratedSlatePolicy().select(request)
    calibrated = TraceCalibratedSlatePolicy(
        SlateAllocationMode.CALIBRATED_FOUR_ROLE
    ).select(request)

    assert [value.option_id for value in default.selected] == [
        "option.slate1",
        "option.slate2",
        "option.slate3",
        "option.slate4",
    ]
    assert {value.role for value in calibrated.selected} == {
        SlateAllocationRole.CALIBRATED_EXPLOIT,
        SlateAllocationRole.MEMORY_HYPOTHESIS,
        SlateAllocationRole.FALSIFICATION_DISAGREEMENT,
        SlateAllocationRole.STRUCTURAL_COVERAGE,
    }
    assert len({value.option_id for value in calibrated.selected}) == 4
    assert calibrated.administered_card_keys == ("card.a", "card.b")
    assert calibrated.prior_only is True
    assert any(
        metric.calibration_source != "declared_prior"
        for row in calibrated.score_rows
        for metric in row.metric_scores
    )
    record = calibrated.to_record()
    assert record["calibration_snapshot_sha256"] == prior.snapshot_sha256
    assert record["claim_scope"].endswith("not_efficacy_or_outcome_claim")
    assert all(
        observation.prediction.wave_index < request.slate.wave_index
        for observation in prior.observations
    )


def test_bounded_memory_dose_is_enforced_by_all_k8_to_k4_allocators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, scope, observations = _fixture_observations()
    prior = build_calibration_snapshot(
        observations,
        scope=scope,
        cutoff_wave_index_exclusive=5,
    )
    legacy = _slate_request(
        scope=scope,
        snapshot=prior,
        wave=5,
        metric_goals=(
            ("total_levels", MetricOptimizationGoal.MINIMIZE),
            ("total_lut_count", MetricOptimizationGoal.MINIMIZE),
        ),
    )
    members = tuple(
        replace(
            value,
            supporting_card_keys=(
                ("card.a",)
                if value.model_rank == 1
                else ("card.b",)
                if value.model_rank == 2
                else ()
            ),
        )
        for value in legacy.slate.members
    )
    slate = replace(legacy.slate, members=members)
    supports = tuple(
        PortfolioMemoryDoseCardSupport(
            card_key=card_key,
            card_content_sha256=_sha(f"{card_key}:content"),
            finite_contract_identity_sha256=slate.finite_contract_sha256,
            compatible_options=(
                (
                    members[rank - 1].option_id,
                    members[rank - 1].option_identity_sha256,
                ),
            ),
            support_policy_id="test_exact_support",
            support_policy_version=1,
            support_policy_definition_sha256=_sha("test exact support"),
        )
        for card_key, rank in (("card.a", 1), ("card.b", 2))
    )
    dose = BoundedPortfolioMemoryDoseContract(
        card_supports=supports,
        proposed_supported_member_bounds=(2, 2),
        evaluated_supported_member_bounds=(2, 2),
        minimum_unattributed_proposed_members=6,
        minimum_unattributed_evaluated_members=2,
    )
    proposal_assessment = assess_proposed_portfolio_memory_dose(
        dose,
        tuple(
            PortfolioMemoryDoseMember(
                rank=value.model_rank,
                option_id=value.option_id,
                option_identity_sha256=value.option_identity_sha256,
                supporting_card_keys=value.supporting_card_keys,
            )
            for value in members
        ),
    )
    request = replace(
        legacy,
        slate=slate,
        memory_dose_contract=dose,
        proposal_memory_dose_assessment=proposal_assessment,
    )

    original_assessor = calibrated_slate_module.assess_allocated_slate_memory_dose
    trace_assessment_calls = 0

    def counted_assessor(*args, **kwargs):
        nonlocal trace_assessment_calls
        trace_assessment_calls += 1
        return original_assessor(*args, **kwargs)

    monkeypatch.setattr(
        calibrated_slate_module,
        "assess_allocated_slate_memory_dose",
        counted_assessor,
    )
    trace_decision = TraceCalibratedSlatePolicy(
        SlateAllocationMode.CALIBRATED_FOUR_ROLE
    ).select(request)
    # K8-to-K4 has C(8, 4)=70 unique subsets. The trace allocator may assess
    # each subset once plus the winning role order, never all 8P4 permutations.
    assert 0 < trace_assessment_calls <= 71
    decisions = (
        trace_decision,
        ModelAnchoredCalibratedSlatePolicy(model_anchor_count=3).select(request),
        StructuralPosteriorSlatePolicy().select(request),
    )
    for decision in decisions:
        assert decision.memory_dose_assessment is not None
        assert decision.memory_dose_assessment.passed
        selected = {value.option_id for value in decision.selected}
        assert {members[0].option_id, members[1].option_id}.issubset(selected)
        assert len(decision.memory_dose_assessment.supported_member_ranks) == 2
        assert len(decision.memory_dose_assessment.unattributed_member_ranks) == 2

    leaked = build_calibration_snapshot(
        observations,
        scope=scope,
        cutoff_wave_index_exclusive=6,
    )
    with pytest.raises(ValueError, match="beyond current wave"):
        TraceCalibratedSlatePolicy(SlateAllocationMode.CALIBRATED_FOUR_ROLE).select(
            replace(request, calibration_snapshot=leaked)
        )


def test_portable_mixed_maximize_minimize_objectives_and_exact_receipt() -> None:
    scope = _scope("mixed-objectives")
    observations = (
        _categorical_observation(
            scope=scope,
            wave=1,
            decision_sha256=_sha("mixed decision 1"),
            parent_sha256=_sha("mixed parent 1"),
            option_id="history.cost",
            option_sha256=_sha("history cost"),
            family="family.f1",
            child_outcome_sha256=_sha("history cost outcome"),
            metric_id="objective:cost",
            predicted="decrease",
            actual="decrease",
            confidence=ForecastConfidenceBin.HIGH,
        ),
        _categorical_observation(
            scope=scope,
            wave=1,
            decision_sha256=_sha("mixed decision 1"),
            parent_sha256=_sha("mixed parent 1"),
            option_id="history.quality",
            option_sha256=_sha("history quality"),
            family="family.f2",
            child_outcome_sha256=_sha("history quality outcome"),
            metric_id="objective:quality",
            predicted="increase",
            actual="increase",
            confidence=ForecastConfidenceBin.HIGH,
        ),
    )
    snapshot = build_calibration_snapshot(
        observations,
        scope=scope,
        cutoff_wave_index_exclusive=2,
    )
    request = _slate_request(
        scope=scope,
        snapshot=snapshot,
        wave=2,
        metric_goals=(
            ("objective:cost", MetricOptimizationGoal.MINIMIZE),
            ("objective:quality", MetricOptimizationGoal.MAXIMIZE),
        ),
        confidence=ForecastConfidenceBin.HIGH,
    )

    decision = TraceCalibratedSlatePolicy(
        SlateAllocationMode.CALIBRATED_FOUR_ROLE
    ).select(request)
    replay = TraceCalibratedSlatePolicy(
        SlateAllocationMode.CALIBRATED_FOUR_ROLE
    ).select(request)

    assert decision.decision_sha256 == replay.decision_sha256
    exploit = next(
        value
        for value in decision.selected
        if value.role is SlateAllocationRole.CALIBRATED_EXPLOIT
    )
    assert exploit.role_score is not None and exploit.role_score > 0.0
    assert decision.prior_only is True
    with pytest.raises(ValueError, match="eight-member slate"):
        TraceCalibratedSlatePolicy(SlateAllocationMode.CALIBRATED_FOUR_ROLE).select(
            replace(
                request, slate=replace(request.slate, members=request.slate.members[:7])
            )
        )

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import combinations

import pytest

from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlate,
    CalibratedSlateMember,
    MetricOptimizationGoal,
    SlateAllocationRequest,
    SlateMetricObjective,
    SlateRoleProposal,
    SlateStructuralEvidence,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationObservation,
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
    MeaningfulDirectionAdjudicationReceipt,
    build_calibration_snapshot,
)
from agent_evolve.policies.selection.model_anchored_slate import (
    ModelAnchoredCalibratedSlatePolicy,
    ModelAnchoredSlateRole,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    StructuralPosteriorSlatePolicy,
    StructuralPosteriorSlateRole,
)
from agent_evolve.policies.selection.frontier_probe_slate import (
    FrontierProbeSlatePolicy,
    FrontierProbeSlateRole,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scope(label: str = "anchored") -> ForecastCalibrationScope:
    return ForecastCalibrationScope(
        model_profile_sha256=_sha(f"{label}:model"),
        prompt_definition_sha256=_sha(f"{label}:prompt"),
        selector_policy_definition_sha256=_sha(f"{label}:selector"),
        benchmark_sha256=_sha(f"{label}:benchmark"),
        session_sha256=_sha(f"{label}:session"),
    )


def _observation(
    *,
    scope: ForecastCalibrationScope,
    ordinal: int,
    asserted: MetricEffectDirection,
    actual: MetricEffectDirection,
    confidence: ForecastConfidenceBin,
) -> ForecastCalibrationObservation:
    option_id = f"history.h{ordinal:02d}"
    option_sha256 = _sha(option_id)
    decision_sha256 = _sha("history decision")
    parent_sha256 = _sha("history parent")
    prediction = ForecastPredictionReceipt(
        scope=scope,
        wave_index=1,
        selector_decision_sha256=decision_sha256,
        parent_candidate_identity_sha256=parent_sha256,
        option_id=option_id,
        option_identity_sha256=option_sha256,
        family="family.history",
        metric_id="objective:cost",
        asserted_direction=asserted,
        confidence=confidence,
    )
    adjudication = MeaningfulDirectionAdjudicationReceipt(
        request_sha256=_sha(f"history request {ordinal}"),
        benchmark_sha256=scope.benchmark_sha256,
        session_sha256=scope.session_sha256,
        wave_index=1,
        parent_candidate_identity_sha256=parent_sha256,
        option_id=option_id,
        option_identity_sha256=option_sha256,
        metric_id="objective:cost",
        parent_outcome_sha256=_sha("history parent outcome"),
        child_outcome_sha256=_sha(f"history child outcome {ordinal}"),
        actual_direction=actual,
        adjudicator_policy_id="fixture_direction",
        adjudicator_policy_version=1,
        adjudicator_definition_sha256=_sha("fixture direction v1"),
    )
    return ForecastCalibrationObservation(prediction, adjudication)


def _prior_observations(
    scope: ForecastCalibrationScope,
) -> tuple[ForecastCalibrationObservation, ...]:
    return tuple(
        [
            _observation(
                scope=scope,
                ordinal=index,
                asserted=MetricEffectDirection.DECREASE,
                actual=MetricEffectDirection.DECREASE,
                confidence=ForecastConfidenceBin.HIGH,
            )
            for index in range(4)
        ]
        + [
            _observation(
                scope=scope,
                ordinal=index,
                asserted=MetricEffectDirection.DECREASE,
                actual=MetricEffectDirection.INCREASE,
                confidence=ForecastConfidenceBin.LOW,
            )
            for index in range(4, 8)
        ]
    )


_DEFAULT_DIRECTIONS = (
    MetricEffectDirection.DECREASE,
    MetricEffectDirection.DECREASE,
    MetricEffectDirection.DECREASE,
    MetricEffectDirection.DECREASE,
    MetricEffectDirection.DECREASE,
    MetricEffectDirection.UNKNOWN,
    MetricEffectDirection.DECREASE,
    MetricEffectDirection.INCREASE,
)
_DEFAULT_CONFIDENCES = (
    ForecastConfidenceBin.MEDIUM,
    ForecastConfidenceBin.MEDIUM,
    ForecastConfidenceBin.MEDIUM,
    ForecastConfidenceBin.MEDIUM,
    ForecastConfidenceBin.LOW,
    ForecastConfidenceBin.UNKNOWN,
    ForecastConfidenceBin.HIGH,
    ForecastConfidenceBin.HIGH,
)


def _request(
    *,
    label: str = "anchored",
    wave: int = 2,
    snapshot_cutoff: int = 2,
    directions: tuple[MetricEffectDirection, ...] = _DEFAULT_DIRECTIONS,
    confidences: tuple[ForecastConfidenceBin, ...] = _DEFAULT_CONFIDENCES,
    supporting_cards: dict[int, tuple[str, ...]] | None = None,
    assigned_cards: tuple[str, ...] = (),
    structural_scores: tuple[float, ...] = (0.5,) * 8,
    allowed_pairs: tuple[tuple[str, str], ...] | None = None,
    min_distinct_families: int | None = None,
    include_history: bool = True,
) -> SlateAllocationRequest:
    scope = _scope(label)
    observations = _prior_observations(scope) if include_history else ()
    snapshot = build_calibration_snapshot(
        observations,
        scope=scope,
        cutoff_wave_index_exclusive=snapshot_cutoff,
        family_min_support=99,
    )
    decision_sha256 = _sha(f"{label}:decision:{wave}")
    parent_sha256 = _sha(f"{label}:parent:{wave}")
    archive_sha256 = _sha(f"{label}:archive:{wave}")
    cards = supporting_cards or {}
    members = tuple(
        CalibratedSlateMember(
            model_rank=index,
            option_id=f"option.o{index}",
            option_identity_sha256=_sha(f"{label}:option:{index}"),
            family=f"family.f{index}",
            locus_key=f"locus.l{index}",
            phenotype_identity_sha256=_sha(f"{label}:phenotype:{index}"),
            supporting_card_keys=cards.get(index, ()),
            role_proposal=SlateRoleProposal.EXPLOIT,
            rationale_sha256=_sha(f"{label}:rationale:{index}"),
            predictions=(
                ForecastPredictionReceipt(
                    scope=scope,
                    wave_index=wave,
                    selector_decision_sha256=decision_sha256,
                    parent_candidate_identity_sha256=parent_sha256,
                    option_id=f"option.o{index}",
                    option_identity_sha256=_sha(f"{label}:option:{index}"),
                    family=f"family.f{index}",
                    metric_id="objective:cost",
                    asserted_direction=directions[index - 1],
                    confidence=confidences[index - 1],
                ),
            ),
            structural_evidence=SlateStructuralEvidence(
                frozen_archive_snapshot_sha256=archive_sha256,
                evidence_receipt_sha256=_sha(f"{label}:evidence:{index}"),
                archive_novelty_score=structural_scores[index - 1],
                structural_coverage_score=structural_scores[index - 1],
            ),
        )
        for index in range(1, 9)
    )
    return SlateAllocationRequest(
        slate=CalibratedSlate(
            scope=scope,
            wave_index=wave,
            selector_decision_sha256=decision_sha256,
            parent_candidate_identity_sha256=parent_sha256,
            finite_contract_sha256=_sha(f"{label}:contract:{wave}"),
            members=members,
        ),
        portfolio_size=4,
        objectives=(
            SlateMetricObjective(
                metric_id="objective:cost",
                goal=MetricOptimizationGoal.MINIMIZE,
                weight=1.0,
                definition_sha256=_sha("minimize objective cost"),
            ),
        ),
        assigned_card_keys=assigned_cards,
        calibration_snapshot=snapshot,
        pairwise_disjoint_option_id_pairs=allowed_pairs,
        min_distinct_families=min_distinct_families,
    )


def test_exact_top_three_plus_best_prior_calibrated_choice() -> None:
    request = _request()

    decision = ModelAnchoredCalibratedSlatePolicy().select(request)

    assert [value.option_id for value in decision.selected] == [
        "option.o1",
        "option.o2",
        "option.o3",
        "option.o7",
    ]
    assert decision.retained_anchor_option_ids == (
        "option.o1",
        "option.o2",
        "option.o3",
    )
    assert decision.calibrated_fill_option_ids == ("option.o7",)
    assert decision.calibrated_fill_exploitation_score == pytest.approx(5 / 6)
    assert [value.role for value in decision.selected] == [
        ModelAnchoredSlateRole.MODEL_ANCHOR,
        ModelAnchoredSlateRole.MODEL_ANCHOR,
        ModelAnchoredSlateRole.MODEL_ANCHOR,
        ModelAnchoredSlateRole.PRIOR_CALIBRATED_FILL,
    ]
    assert decision.prior_only is True
    assert decision == ModelAnchoredCalibratedSlatePolicy().select(request)
    assert len(decision.decision_sha256) == 64
    assert (
        decision.policy_configuration_sha256
        == ModelAnchoredCalibratedSlatePolicy().configuration_sha256
    )


def test_incompatible_anchors_are_displaced_only_as_feasibility_requires() -> None:
    option_ids = tuple(f"option.o{index}" for index in range(1, 9))
    allowed_pairs = tuple(
        pair
        for pair in combinations(option_ids, 2)
        if pair != ("option.o1", "option.o2")
    )
    request = _request(label="incompatible", allowed_pairs=allowed_pairs)

    decision = ModelAnchoredCalibratedSlatePolicy().select(request)

    assert len(decision.retained_anchor_option_ids) == 2
    assert "option.o3" in decision.retained_anchor_option_ids
    assert not {"option.o1", "option.o2"}.issubset(
        decision.retained_anchor_option_ids
    )
    assert len(decision.calibrated_fill_option_ids) == 2


def test_assigned_card_coverage_overrides_the_best_calibrated_fill() -> None:
    request = _request(
        label="cards",
        supporting_cards={8: ("card.required",)},
        assigned_cards=("card.required",),
    )

    decision = ModelAnchoredCalibratedSlatePolicy().select(request)

    assert decision.retained_anchor_option_ids == (
        "option.o1",
        "option.o2",
        "option.o3",
    )
    assert decision.calibrated_fill_option_ids == ("option.o8",)
    assert decision.administered_card_keys == ("card.required",)


def test_structural_then_model_rank_ties_are_deterministic() -> None:
    same_direction = (MetricEffectDirection.DECREASE,) * 8
    same_confidence = (ForecastConfidenceBin.MEDIUM,) * 8
    request = _request(
        label="tie",
        directions=same_direction,
        confidences=same_confidence,
        include_history=False,
        structural_scores=(0.5, 0.5, 0.5, 0.75, 0.75, 0.5, 0.5, 0.5),
    )
    policy = ModelAnchoredCalibratedSlatePolicy()

    first = policy.select(request)
    second = policy.select(request)

    assert first.calibrated_fill_option_ids == ("option.o4",)
    assert first.decision_sha256 == second.decision_sha256
    assert first.to_record() == second.to_record()


def test_prior_only_guard_rejects_snapshot_cutoff_after_current_wave() -> None:
    request = _request(
        label="leaked-cutoff",
        wave=2,
        snapshot_cutoff=3,
    )

    with pytest.raises(ValueError, match="cutoff reaches beyond current wave"):
        ModelAnchoredCalibratedSlatePolicy().select(request)


def test_receipt_contains_no_numeric_outcome_values_and_revalidates_exactly() -> None:
    decision = ModelAnchoredCalibratedSlatePolicy().select(
        _request(label="receipt")
    )

    record_text = json.dumps(decision.to_record(), sort_keys=True)
    assert "parent_metric_value" not in record_text
    assert "child_metric_value" not in record_text
    assert decision.to_record()["claim_scope"].endswith(
        "not_efficacy_or_outcome_claim"
    )
    decision.revalidate()
    with pytest.raises(ValueError, match="exact model-anchored allocation"):
        replace(
            decision,
            calibrated_fill_option_ids=("option.o8",),
        )


def test_card_administration_and_family_constraints_can_make_slate_infeasible() -> None:
    request = _request(
        label="no-card",
        assigned_cards=("card.missing",),
        min_distinct_families=4,
    )

    with pytest.raises(ValueError, match="no feasible K4 subset"):
        ModelAnchoredCalibratedSlatePolicy().select(request)


def test_anchor_configuration_identifies_full_top_four_fallback_mode() -> None:
    assert (
        ModelAnchoredCalibratedSlatePolicy(2).configuration_sha256
        != ModelAnchoredCalibratedSlatePolicy(3).configuration_sha256
    )
    request = _request(label="full-top-four")
    decision = ModelAnchoredCalibratedSlatePolicy(4).select(request)
    assert [value.option_id for value in decision.selected] == [
        "option.o1",
        "option.o2",
        "option.o3",
        "option.o4",
    ]
    assert decision.retained_anchor_option_ids == (
        "option.o1",
        "option.o2",
        "option.o3",
        "option.o4",
    )
    assert decision.calibrated_fill_option_ids == ()
    assert all(
        value.role is ModelAnchoredSlateRole.MODEL_ANCHOR
        for value in decision.selected
    )


def test_full_top_four_anchor_uses_calibrated_fill_only_when_infeasible() -> None:
    option_ids = tuple(f"option.o{index}" for index in range(1, 9))
    allowed_pairs = tuple(
        pair
        for pair in combinations(option_ids, 2)
        if pair != ("option.o1", "option.o2")
    )
    decision = ModelAnchoredCalibratedSlatePolicy(4).select(
        _request(label="full-top-four-fallback", allowed_pairs=allowed_pairs)
    )
    assert len(decision.retained_anchor_option_ids) == 3
    assert len(decision.calibrated_fill_option_ids) == 1
    assert not {"option.o1", "option.o2"}.issubset(
        {value.option_id for value in decision.selected}
    )


def test_structural_posterior_spends_epistemic_role_on_novel_abstention() -> None:
    request = _request(
        label="structural-posterior-abstention",
        directions=(
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.UNKNOWN,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.DECREASE,
        ),
        confidences=(ForecastConfidenceBin.MEDIUM,) * 8,
        structural_scores=(0.1, 0.1, 0.1, 0.1, 0.8, 1.0, 0.7, 0.6),
    )

    decision = StructuralPosteriorSlatePolicy().select(request)

    epistemic = next(
        value
        for value in decision.selected
        if value.role is StructuralPosteriorSlateRole.EPISTEMIC_STRUCTURAL
    )
    assert epistemic.option_id == "option.o6"
    row = next(value for value in decision.score_rows if value.option_id == "option.o6")
    assert row.raw_epistemic_score == 1.0
    assert row.epistemic_structural_score == 1.0
    assert row.metric_scores[0].explicit_abstention is True
    assert decision.prior_only is True


def test_calibrated_frontier_inverts_below_chance_direction_forecasts() -> None:
    request = _request(
        label="structural-posterior-negative-skill",
        directions=(
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.INCREASE,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.INCREASE,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.INCREASE,
            MetricEffectDirection.DECREASE,
            MetricEffectDirection.INCREASE,
        ),
        confidences=(ForecastConfidenceBin.LOW,) * 8,
    )
    scope = request.slate.scope
    anti_increase = tuple(
        _observation(
            scope=scope,
            ordinal=index,
            asserted=MetricEffectDirection.INCREASE,
            actual=MetricEffectDirection.DECREASE,
            confidence=ForecastConfidenceBin.LOW,
        )
        for index in range(20, 24)
    )
    request = replace(
        request,
        calibration_snapshot=build_calibration_snapshot(
            (*_prior_observations(scope), *anti_increase),
            scope=scope,
            cutoff_wave_index_exclusive=2,
            family_min_support=99,
        ),
    )

    decision = StructuralPosteriorSlatePolicy().select(request)
    rows = {value.option_id: value for value in decision.score_rows}
    assert rows["option.o1"].calibrated_exploitation_score == pytest.approx(-2 / 3)
    assert rows["option.o2"].calibrated_exploitation_score == pytest.approx(2 / 3)
    exploit = next(
        value
        for value in decision.selected
        if value.role is StructuralPosteriorSlateRole.CALIBRATED_EXPLOIT
    )
    assert exploit.option_id in {"option.o2", "option.o4", "option.o6", "option.o8"}


def test_calibrated_frontier_treats_model_card_citations_as_diagnostics_only() -> None:
    base = _request(
        label="structural-posterior-citation-authority",
        assigned_cards=("card.advisory",),
    )
    cited = replace(
        base,
        slate=replace(
            base.slate,
            members=tuple(
                replace(
                    member,
                    supporting_card_keys=("card.advisory",)
                    if member.model_rank == 8
                    else (),
                )
                for member in base.slate.members
            ),
        ),
    )

    uncited_decision = StructuralPosteriorSlatePolicy().select(base)
    cited_decision = StructuralPosteriorSlatePolicy().select(cited)

    assert [value.option_id for value in uncited_decision.selected] == [
        value.option_id for value in cited_decision.selected
    ]
    assert [value.role_score for value in uncited_decision.selected] == [
        value.role_score for value in cited_decision.selected
    ]
    assert uncited_decision.administered_card_keys == ()
    assert cited_decision.administered_card_keys == ()
    cited_row = next(
        value for value in cited_decision.score_rows if value.option_id == "option.o8"
    )
    assert cited_row.model_declared_assigned_card_keys == ("card.advisory",)


def test_structural_posterior_is_deterministic_and_outcome_blind() -> None:
    request = _request(label="structural-posterior-replay")
    policy = StructuralPosteriorSlatePolicy()

    first = policy.select(request)
    second = policy.select(request)

    assert first.decision_sha256 == second.decision_sha256
    assert first.to_record() == second.to_record()
    record_text = json.dumps(first.to_record(), sort_keys=True)
    assert "parent_metric_value" not in record_text
    assert "child_metric_value" not in record_text
    with pytest.raises(ValueError, match="exact structural-posterior allocation"):
        replace(first, joint_score=first.joint_score + 1.0)
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        ModelAnchoredCalibratedSlatePolicy(5)


def _two_metric_request(
    *,
    label: str,
    directions_by_rank: dict[
        int,
        tuple[MetricEffectDirection, MetricEffectDirection],
    ],
    structural_scores: tuple[float, ...] = (0.5,) * 8,
) -> SlateAllocationRequest:
    source = _request(
        label=label,
        directions=(MetricEffectDirection.DECREASE,) * 8,
        confidences=(ForecastConfidenceBin.MEDIUM,) * 8,
        structural_scores=structural_scores,
    )
    members = []
    for member in source.slate.members:
        first_direction, second_direction = directions_by_rank.get(
            member.model_rank,
            (
                MetricEffectDirection.DECREASE,
                MetricEffectDirection.INCREASE,
            ),
        )
        first = replace(
            member.predictions[0],
            asserted_direction=first_direction,
        )
        second = replace(
            member.predictions[0],
            metric_id="objective:quality",
            asserted_direction=second_direction,
        )
        members.append(
            replace(
                member,
                predictions=tuple(
                    sorted((first, second), key=lambda value: value.metric_id)
                ),
            )
        )
    slate = replace(source.slate, members=tuple(members))
    return replace(
        source,
        slate=slate,
        objectives=(
            source.objectives[0],
            SlateMetricObjective(
                metric_id="objective:quality",
                goal=MetricOptimizationGoal.MAXIMIZE,
                weight=1.0,
                definition_sha256=_sha("maximize objective quality"),
            ),
        ),
    )


def test_frontier_probe_projects_mating_constraints_and_keeps_model_top_four() -> None:
    option_ids = tuple(f"option.o{index}" for index in range(1, 9))
    allowed_pairs = tuple(
        pair
        for pair in combinations(option_ids, 2)
        if pair != ("option.o1", "option.o2")
    )
    request = _request(
        label="frontier-projection",
        directions=(MetricEffectDirection.DECREASE,) * 8,
        allowed_pairs=allowed_pairs,
        min_distinct_families=4,
    )

    decision = FrontierProbeSlatePolicy().select(request)

    assert [value.option_id for value in decision.selected] == [
        "option.o1",
        "option.o2",
        "option.o3",
        "option.o4",
    ]
    assert decision.selected_probe_option_id is None
    assert decision.projection.removed_pairwise_constraint is True
    assert decision.projection.removed_min_distinct_families == 4
    assert decision.request.pairwise_disjoint_option_id_pairs is None
    assert decision.request.min_distinct_families is None
    assert decision.source_request.request_sha256 == request.request_sha256


def test_frontier_probe_distinguishes_partial_from_full_vector_abstention() -> None:
    request = _two_metric_request(
        label="frontier-full-abstention",
        directions_by_rank={
            5: (
                MetricEffectDirection.UNKNOWN,
                MetricEffectDirection.INCREASE,
            ),
            6: (
                MetricEffectDirection.UNKNOWN,
                MetricEffectDirection.UNKNOWN,
            ),
            7: (
                MetricEffectDirection.UNKNOWN,
                MetricEffectDirection.UNKNOWN,
            ),
        },
        structural_scores=(0.1, 0.1, 0.1, 0.1, 1.0, 0.7, 0.9, 0.6),
    )

    decision = FrontierProbeSlatePolicy().select(request)

    assert [value.option_id for value in decision.selected] == [
        "option.o1",
        "option.o2",
        "option.o3",
        "option.o7",
    ]
    assert decision.selected_probe_option_id == "option.o7"
    assert decision.available_full_abstention_option_ids == (
        "option.o6",
        "option.o7",
    )
    assert decision.member_evidence[4].unknown_metric_ids == (
        "objective:cost",
    )
    assert decision.member_evidence[4].full_vector_abstention is False
    assert decision.selected[-1].role is (
        FrontierProbeSlateRole.FULL_ABSTENTION_FRONTIER_PROBE
    )


def test_frontier_probe_treats_cards_as_advisory_without_a_dose_contract() -> None:
    request = _request(
        label="frontier-advisory-card",
        directions=(MetricEffectDirection.DECREASE,) * 8,
        supporting_cards={8: ("card.advisory",)},
        assigned_cards=("card.advisory",),
    )

    decision = FrontierProbeSlatePolicy().select(request)

    assert [value.option_id for value in decision.selected] == [
        "option.o1",
        "option.o2",
        "option.o3",
        "option.o4",
    ]
    assert decision.administered_card_keys == ()


def test_frontier_probe_uses_safety_recourse_for_duplicate_phenotype() -> None:
    source = _request(
        label="frontier-phenotype-recourse",
        directions=(MetricEffectDirection.DECREASE,) * 8,
    )
    members = list(source.slate.members)
    members[3] = replace(
        members[3],
        phenotype_identity_sha256=members[0].phenotype_identity_sha256,
    )
    request = replace(source, slate=replace(source.slate, members=tuple(members)))

    decision = FrontierProbeSlatePolicy().select(request)

    assert [value.option_id for value in decision.selected] == [
        "option.o1",
        "option.o2",
        "option.o3",
        "option.o5",
    ]
    assert decision.ideal_target_feasible is False
    assert decision.selected[-1].role is FrontierProbeSlateRole.SAFETY_RECOURSE
    assert decision.distinct_phenotype_count == 4


def test_frontier_probe_receipt_is_deterministic_and_exactly_replayable() -> None:
    request = _request(label="frontier-replay")
    policy = FrontierProbeSlatePolicy()

    first = policy.select(request)
    second = policy.select(request)

    assert first.decision_sha256 == second.decision_sha256
    assert first.to_record() == second.to_record()
    assert first.prior_only is True
    assert "parent_metric_value" not in json.dumps(first.to_record())
    with pytest.raises(ValueError, match="exact frontier-probe allocation"):
        replace(first, retained_target_count=3)

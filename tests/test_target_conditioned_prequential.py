from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
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
    ForecastCalibrationSnapshot,
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    score_structural_posterior_slate,
)
from agent_evolve.policies.selection.target_conditioned_features import (
    FEATURE_NAMES,
    TargetConditionedFeatureProjectionRequest,
    TargetConditionedPortableFeatureProjector,
    project_portable_transition,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (
    TargetConditionedAllocationContext,
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    BASE_REALIZABILITY_PROJECTOR_ID,
    PrequentialLinearGaussianHead,
    RealizablePortfolioSet,
    TargetConditionedAcquisitionProfile,
    TargetConditionedAcquisitionState,
    TargetConditionedMemberFeatures,
    TargetConditionedMetaPrior,
    TargetConditionedPrequentialSlatePolicy,
    TargetConditionedSelectedObservation,
    TargetConditionedSlateRequest,
    enumerate_base_realizable_portfolios,
    update_target_conditioned_state,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjection,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _allocation_request() -> SlateAllocationRequest:
    scope = ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("prompt"),
        selector_policy_definition_sha256=_sha("selector"),
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
    )
    parent_sha256 = typed_json_sha256(freeze_json({"x": 0}))
    selector_sha256 = _sha("selector decision")
    archive_sha256 = _sha("archive")
    members = []
    for rank in range(1, 9):
        option_id = f"option.o{rank}"
        option_sha256 = _sha(option_id)
        family = "family.alpha" if rank <= 3 else "family.beta"
        members.append(
            CalibratedSlateMember(
                model_rank=rank,
                option_id=option_id,
                option_identity_sha256=option_sha256,
                family=family,
                locus_key=f"locus.l{rank}",
                phenotype_identity_sha256=_sha(f"phenotype {rank}"),
                supporting_card_keys=("card.a",),
                role_proposal=SlateRoleProposal.EXPLOIT,
                rationale_sha256=_sha(f"rationale {rank}"),
                predictions=(
                    ForecastPredictionReceipt(
                        scope=scope,
                        wave_index=1,
                        selector_decision_sha256=selector_sha256,
                        parent_candidate_identity_sha256=parent_sha256,
                        option_id=option_id,
                        option_identity_sha256=option_sha256,
                        family=family,
                        metric_id="objective:cost",
                        asserted_direction=MetricEffectDirection.DECREASE,
                        confidence=ForecastConfidenceBin.UNKNOWN,
                    ),
                    ForecastPredictionReceipt(
                        scope=scope,
                        wave_index=1,
                        selector_decision_sha256=selector_sha256,
                        parent_candidate_identity_sha256=parent_sha256,
                        option_id=option_id,
                        option_identity_sha256=option_sha256,
                        family=family,
                        metric_id="objective:quality",
                        asserted_direction=MetricEffectDirection.INCREASE,
                        confidence=ForecastConfidenceBin.UNKNOWN,
                    ),
                ),
                structural_evidence=SlateStructuralEvidence(
                    frozen_archive_snapshot_sha256=archive_sha256,
                    evidence_receipt_sha256=_sha(f"evidence {rank}"),
                    archive_novelty_score=rank / 8.0,
                    structural_coverage_score=(9 - rank) / 8.0,
                ),
            )
        )
    return SlateAllocationRequest(
        slate=CalibratedSlate(
            scope=scope,
            wave_index=1,
            selector_decision_sha256=selector_sha256,
            parent_candidate_identity_sha256=parent_sha256,
            finite_contract_sha256=_sha("finite contract"),
            members=tuple(members),
        ),
        portfolio_size=2,
        objectives=(
            SlateMetricObjective(
                metric_id="objective:cost",
                goal=MetricOptimizationGoal.MINIMIZE,
                weight=1.0,
                definition_sha256=_sha("cost objective"),
            ),
            SlateMetricObjective(
                metric_id="objective:quality",
                goal=MetricOptimizationGoal.MAXIMIZE,
                weight=1.0,
                definition_sha256=_sha("quality objective"),
            ),
        ),
        assigned_card_keys=("card.a",),
        calibration_snapshot=ForecastCalibrationSnapshot(
            scope=scope,
            cutoff_wave_index_exclusive=1,
            observations=(),
        ),
        min_distinct_families=2,
        required_option_ids=("option.o1",),
    )


def _head(*, rhs: tuple[float, float]) -> PrequentialLinearGaussianHead:
    return PrequentialLinearGaussianHead(
        feature_names=("bias", "signal"),
        means=(0.0, 0.0),
        scales=(1.0, 1.0),
        precision=((1.0, 0.0), (0.0, 1.0)),
        rhs=rhs,
        residual_variance=1.0,
    )


@dataclass(frozen=True, slots=True)
class _ContextProvider:
    context: TargetConditionedAllocationContext
    provider_id: str = "fixture.target_context"
    provider_version: int = 1
    definition_sha256: str = _sha("fixture target context provider")

    def context_for(
        self, request: SlateAllocationRequest
    ) -> TargetConditionedAllocationContext:
        self.context.require_request(request)
        return self.context


def _targeted_request() -> TargetConditionedSlateRequest:
    allocation = _allocation_request()
    state = TargetConditionedAcquisitionState(
        campaign_scope_sha256=_sha("campaign scope"),
        training_data_sha256=_sha("portable training panel"),
        marginal_head=_head(rhs=(0.0, 1.0)),
        direction_head=_head(rhs=(0.0, 0.0)),
    )
    features = tuple(
        TargetConditionedMemberFeatures(
            option_id=member.option_id,
            option_identity_sha256=member.option_identity_sha256,
            feature_names=("bias", "signal"),
            values=(1.0, float(member.model_rank)),
            projector_id="portable.fixture",
            projector_version=1,
            projector_definition_sha256=_sha("portable fixture projector"),
        )
        for member in allocation.slate.members
    )
    target = CampaignPortfolioFrontierTarget(
        allocator_id="fixture",
        allocator_version=1,
        definition_sha256=_sha("fixture allocator"),
        archive_utility_snapshot_sha256=_sha("archive utility"),
        lane_id="lane.1",
        parent_configuration_sha256=(
            allocation.slate.parent_candidate_identity_sha256
        ),
        direction_id="quality",
        opportunity_rank=1,
        payload=freeze_json({"direction": "quality", "target": 0.5}),
    )
    return TargetConditionedSlateRequest(
        allocation_request=allocation,
        frontier_target=target,
        state=state,
        member_features=features,
        realizable_portfolios=enumerate_base_realizable_portfolios(allocation),
        campaign_generation=1,
        remaining_proposal_horizon=0,
    )


def test_linear_gaussian_head_updates_immutably_from_sufficient_statistics() -> None:
    head = _head(rhs=(0.0, 0.0))

    updated = head.update(((1.0, 2.0), (1.0, 3.0)), (0.5, 1.0))

    assert head.precision == ((1.0, 0.0), (0.0, 1.0))
    assert updated.precision == ((3.0, 5.0), (5.0, 14.0))
    assert updated.rhs == (1.5, 4.0)
    assert updated.head_sha256 != head.head_sha256
    assert updated.to_record()["head_sha256"] == updated.head_sha256


def test_exact_base_universe_and_terminal_selection_respect_generic_constraints() -> None:
    request = _targeted_request()
    assert request.realizable_portfolios.option_id_sets == (
        ("option.o1", "option.o4"),
        ("option.o1", "option.o5"),
        ("option.o1", "option.o6"),
        ("option.o1", "option.o7"),
        ("option.o1", "option.o8"),
    )
    policy = TargetConditionedPrequentialSlatePolicy(
        TargetConditionedAcquisitionProfile(
            direction_weight=0.5,
            uncertainty_weight=1.0,
            maximum_remaining_horizon=2,
        )
    )

    decision = policy.select(request)

    assert tuple(value.option_id for value in decision.selected) == (
        "option.o1",
        "option.o8",
    )
    assert decision.prior_only is True
    assert decision.feasible_subset_count == 5
    assert decision.administered_card_keys == ("card.a",)
    assert decision.to_record()["claim_scope"] == (
        "allocation_receipt_not_efficacy_or_outcome_claim"
    )


def test_reserved_base_projector_cannot_claim_an_incomplete_universe() -> None:
    request = _targeted_request()
    incomplete = replace(
        request.realizable_portfolios,
        option_id_sets=request.realizable_portfolios.option_id_sets[:-1],
    )

    with pytest.raises(ValueError, match="not complete"):
        replace(request, realizable_portfolios=incomplete)

    with pytest.raises(ValueError, match="foreign identity"):
        replace(
            request,
            realizable_portfolios=RealizablePortfolioSet(
                source_request_sha256=(
                    request.allocation_request.request_sha256
                ),
                projector_id=BASE_REALIZABILITY_PROJECTOR_ID,
                projector_version=999,
                projector_definition_sha256=_sha("foreign base projector"),
                option_id_sets=request.realizable_portfolios.option_id_sets,
            ),
        )


def test_generation_barrier_updates_exactly_the_selected_rows() -> None:
    request = _targeted_request()
    decision = TargetConditionedPrequentialSlatePolicy(
        TargetConditionedAcquisitionProfile(
            direction_weight=0.5,
            uncertainty_weight=0.25,
            maximum_remaining_horizon=2,
        )
    ).select(request)
    features = {value.option_id: value for value in request.member_features}
    observations = tuple(
        TargetConditionedSelectedObservation(
            decision_sha256=decision.decision_sha256,
            campaign_generation=1,
            option_id=selected.option_id,
            option_identity_sha256=selected.option_identity_sha256,
            feature_row_sha256=features[selected.option_id].feature_row_sha256,
            feature_values=features[selected.option_id].values,
            normalized_marginal_utility=float(index),
            normalized_target_improvement=0.5 if index == 1 else -0.25,
            evaluator_receipt_sha256=_sha(f"evaluation {selected.option_id}"),
        )
        for index, selected in enumerate(decision.selected, start=1)
    )

    receipt = update_target_conditioned_state(
        request.state,
        decisions=(decision,),
        observations=observations,
    )

    assert request.state.cutoff_generation == 0
    assert request.state.selected_observation_count == 0
    assert receipt.next_state.cutoff_generation == 1
    assert receipt.next_state.selected_observation_count == 2
    assert receipt.next_state.state_sha256 != request.state.state_sha256
    assert receipt.to_record()["rejected_outcomes_consulted"] is False

    unselected = features["option.o5"]
    invalid = replace(
        observations[-1],
        option_id=unselected.option_id,
        option_identity_sha256=unselected.option_identity_sha256,
        feature_row_sha256=unselected.feature_row_sha256,
        feature_values=unselected.values,
    )
    with pytest.raises(ValueError, match="exactly every selected"):
        update_target_conditioned_state(
            request.state,
            decisions=(decision,),
            observations=(*observations[:-1], invalid),
        )


def test_frozen_portable_profile_and_meta_prior_decode_with_hash_checks() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
        "trap_portable_profile_v1.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))

    profile = TargetConditionedAcquisitionProfile.from_record(
        artifact["profile"]
    )
    prior = TargetConditionedMetaPrior.from_record(artifact["meta_prior"])
    state = prior.initial_state(campaign_scope_sha256=_sha("new campaign"))

    assert profile.direction_weight == 0.5
    assert profile.uncertainty_weight == 0.25
    assert len(prior.marginal_head.feature_names) == 70
    assert prior.marginal_head.feature_names == FEATURE_NAMES
    assert state.cutoff_generation == 0
    assert state.selected_observation_count == 0
    assert state.training_data_sha256 == prior.training_data_sha256

    corrupted = dict(artifact["meta_prior"])
    corrupted["training_data_sha256"] = _sha("corrupted")
    with pytest.raises(ValueError, match="identity mismatch"):
        TargetConditionedMetaPrior.from_record(corrupted)


def test_portable_projector_builds_exact_prior_schema_from_typed_generic_facts() -> None:
    allocation = _allocation_request()
    parent_configuration = freeze_json({"x": 0})
    transitions = tuple(
        project_portable_transition(
            option_id=member.option_id,
            option_identity_sha256=member.option_identity_sha256,
            parent_configuration=parent_configuration,
            child_configuration=freeze_json({"x": member.model_rank}),
        )
        for member in sorted(allocation.slate.members, key=lambda value: value.option_id)
    )
    archive_sha256 = _sha("archive utility snapshot")
    archive_context = CampaignPortfolioArchiveContextProjection(
        projector_id="fixture.affine",
        projector_version=1,
        definition_sha256=_sha("fixture affine projector"),
        archive_utility_snapshot_sha256=archive_sha256,
        parent_configuration_sha256=(
            allocation.slate.parent_candidate_identity_sha256
        ),
        payload=freeze_json(
            {
                "optimization_frame": {
                    "axes": [
                        {"metric_id": "objective:cost"},
                        {"metric_id": "objective:quality"},
                    ],
                    "reference_directions": [
                        {
                            "direction_id": "axis_1_extreme",
                            "normalized_importance_decimal": ["1", "0"],
                        },
                        {
                            "direction_id": "balanced_tradeoff",
                            "normalized_importance_decimal": ["0.5", "0.5"],
                        },
                    ],
                    "base_hypervolume_decimal": "0.3",
                },
                "archive": {
                    "normalized_points_decimal": [
                        ["0.1", "0.7"],
                        ["0.5", "0.2"],
                    ]
                },
                "parent": {"normalized_point_decimal": ["0.2", "0.4"]},
            }
        ),
    )
    target = CampaignPortfolioFrontierTarget(
        allocator_id="fixture",
        allocator_version=1,
        definition_sha256=_sha("fixture target allocator"),
        archive_utility_snapshot_sha256=archive_sha256,
        lane_id="lane.0",
        parent_configuration_sha256=(
            allocation.slate.parent_candidate_identity_sha256
        ),
        direction_id="axis_1_extreme",
        opportunity_rank=1,
        payload=freeze_json(
            {
                "target_direction": {
                    "normalized_weights_decimal": ["1", "0"],
                    "opportunity_from_ideal_decimal": "0.5",
                },
                "assigned_parent": {
                    "normalized_point_decimal": ["0.2", "0.4"],
                    "achievement_decimal": "0.21",
                    "regret_above_archive_best_decimal": "0.11",
                },
            }
        ),
    )
    projection_request = TargetConditionedFeatureProjectionRequest(
        allocation_request=allocation,
        structural_score_rows=tuple(
            sorted(
                score_structural_posterior_slate(allocation),
                key=lambda value: value.option_id,
            )
        ),
        transition_receipts=transitions,
        archive_context=archive_context,
        frontier_target=target,
        campaign_generation=1,
        lane_slot=0,
        remaining_proposal_horizon=2,
    )

    projected = TargetConditionedPortableFeatureProjector().project(
        projection_request
    )

    assert len(projected) == 8
    assert all(value.feature_names == FEATURE_NAMES for value in projected)
    first = dict(zip(FEATURE_NAMES, projected[0].values, strict=True))
    assert first["rank_1"] == 1.0
    assert first["generation_1"] == 1.0
    assert first["parent_slot_0"] == 1.0
    assert first["transition_change_count"] == 1.0
    assert first["transition_numeric_fraction"] == 1.0
    assert first["transition_numeric_sign"] == 1.0
    assert first["parent_desirability_mean"] == pytest.approx(0.7)
    assert first["archive_point_count_log"] == pytest.approx(math.log1p(2))
    assert first["target_favorable_fraction"] == 1.0
    assert first["off_target_favorable_fraction"] == 1.0
    assert first["target_posterior_correctness"] == 0.5
    assert first["target_signed_evidence"] == 1.0
    assert first["target_reliability_adjusted_evidence"] == 0.0
    assert first["remaining_proposal_horizon_fraction"] == 1.0

    artifact_path = (
        Path(__file__).resolve().parents[2]
        / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
        "trap_portable_profile_v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    profile = TargetConditionedAcquisitionProfile.from_record(
        artifact["profile"]
    )
    prior = TargetConditionedMetaPrior.from_record(artifact["meta_prior"])
    k4_request = replace(
        allocation,
        portfolio_size=4,
        required_option_ids=(),
    )
    context = TargetConditionedAllocationContext(
        finite_contract_sha256=k4_request.slate.finite_contract_sha256,
        cutoff_receipt_sha256=_sha("pre-call cutoff"),
        archive_context=archive_context,
        frontier_target=target,
        state=prior.initial_state(campaign_scope_sha256=_sha("campaign scope")),
        transition_receipts=transitions,
        campaign_generation=1,
        lane_slot=0,
        remaining_proposal_horizon=2,
    )

    decision = TargetConditionedSlateAllocatorAdapter(
        context_provider=_ContextProvider(context),
        profile=profile,
    ).select(k4_request)

    assert len(decision.selected) == 4
    assert decision.prior_only is True
    assert all(
        value.role.value == "target_conditioned_acquisition"
        for value in decision.selected
    )
    assert decision.request.member_features[0].feature_names == FEATURE_NAMES

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryEntry,
    InsightOrigin,
)
from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.campaign_diagnostic_blocks import (
    CampaignDiagnosticSupportCardInput,
    CampaignDiagnosticSupportLaneInput,
)
from agent_evolve.application.portfolio_memory_dose import (
    PortfolioMemoryDoseCardSemantics,
    derive_portfolio_memory_dose_card_support,
)
from agent_evolve.application.portfolio_memory_matched_control import (
    PortfolioMemoryLaneSupportResolver,
    PortfolioMemoryMatchedControlOutcome,
    PortfolioMemoryMatchedControlPlanner,
    PortfolioMemoryMatchedSupportResolver,
    materialize_portfolio_memory_matched_arm,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryMatchedControlWavePlan,
    PortfolioRewardAggregationBinding,
    PortfolioVariationWaveRequest,
)
from agent_evolve.application.portfolio_projection import (
    admit_portfolio_card_sources,
    portfolio_card_from_insight_entry,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
    bind_finite_action_evidence,
)
from agent_evolve.domain.ids import (
    CandidateId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.policies.memory.balanced_subset_blocks import (
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.memory.compatibility_matching import (
    LaneCardMatchingCard,
    LaneCardMatchingLane,
)
from agent_evolve.policies.selection.calibrated_portfolio_binding import (
    common_pool_required_option_ids,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD,
    PortfolioExperimentalArm,
    PortfolioSelectionRequest,
)
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
)


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _contract() -> FiniteVariationContract:
    parent = _object({"x": 0, "y": 0})
    parent_sha256 = typed_json_sha256(parent)
    return FiniteVariationContract(
        catalog_id="matched_memory_test",
        catalog_version=1,
        catalog_definition_sha256="a" * 64,
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="alpha.x1",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object({"x": 1, "y": 0}),
                family="alpha",
                description="Opaque first action.",
            ),
            FiniteVariationOption(
                option_id="beta.y1",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object({"x": 0, "y": 1}),
                family="beta",
                description="Opaque second action.",
            ),
        ),
    )


def _entry_and_card():
    contract = _contract()
    contrast_id = "b" * 64
    binding = bind_finite_action_evidence(
        contrast_id=contrast_id,
        contract=contract,
        option_id="alpha.x1",
    )
    draft = InsightDraft(
        claim="A bounded action improved the prior observation.",
        trigger="A compatible finite action is available.",
        mechanism="Retest the exact action under the current parent.",
        affected_paths=("$.x",),
        evidence_summary="One authenticated direct contrast.",
        confidence=0.5,
        evidence_contrast_ids=(contrast_id,),
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="cost",
                direction=MetricEffectDirection.DECREASE,
            ),
        ),
        recommended_option_families=("alpha",),
        recommended_option_ids=("alpha.x1",),
        action_template="Retest the exact finite action.",
        falsification_condition="Reject if the held-out effect reverses.",
    )
    entry = InsightMemoryEntry(
        reference=InsightRef(InsightId("insight_matched_memory_test"), 1),
        draft=draft,
        initial_score=0.0,
        lifecycle_state=InsightLifecycleState.QUARANTINED,
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=InsightEvidenceLineage(
            reflection_call_id=LLMCallId("call_matched_memory_source"),
            source_operator_invocation_ids=(
                OperatorInvocationId("operator_matched_memory_source"),
            ),
            source_candidate_ids=(
                CandidateId("candidate_matched_memory_source"),
            ),
            available_contrast_ids=(contrast_id,),
            cited_contrast_ids=(contrast_id,),
            finite_action_bindings=(binding,),
        ),
    )
    card = portfolio_card_from_insight_entry(
        entry,
        card_key="card.matched",
        prompt_payload=_object(
            {
                "claim": draft.claim,
                "mechanism": draft.mechanism,
            }
        ),
        evidence_sha256="c" * 64,
        source_receipt_sha256="d" * 64,
        assigned_score=0.25,
    )
    return contract, entry, card, admit_portfolio_card_sources((entry,), (card,))


def _units() -> tuple[StableMemoryAssignmentUnit, ...]:
    return tuple(
        StableMemoryAssignmentUnit(
            unit_key=f"matched.g05.lane{index}",
            generation=5,
            lane_id=f"lane_{index}",
        )
        for index in range(2)
    )


def _parent(contract: FiniteVariationContract) -> EvolutionCandidate:
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId("candidate_matched_memory_parent"),
            configuration_hash=contract.parent_configuration_sha256,
            configuration_artifact_hash="9" * 64,
            proposal_sequence=0,
        ),
        configuration=contract.parent_configuration,
        objectives=(("cost", 0.0),),
        valid=True,
        generation=0,
        label="matched-memory-parent",
    )


def _wave_for_view(
    *,
    contract: FiniteVariationContract,
    plan,
    view,
    support,
) -> PortfolioVariationWaveRequest:
    context = _object({"same_context": True})
    dose = None
    if view.memory_dose_allowed:
        dose = BoundedPortfolioMemoryDoseContract(
            card_supports=(support,),
            proposed_supported_member_bounds=(1, 1),
            evaluated_supported_member_bounds=(1, 1),
            minimum_unattributed_proposed_members=1,
            minimum_unattributed_evaluated_members=1,
        )
    request = PortfolioSelectionRequest(
        call_id=LLMCallId(
            f"call_matched_wave_{view.assignment.arm.value.lower()}"
        ),
        operation="select_portfolio",
        instruction="Select from the same sealed finite action pool.",
        context=context,
        finite_variation_contract=contract,
        cards=view.cards,
        portfolio_size=2,
        required_metric_ids=("cost",),
        require_supporting_cards=False,
        source_registry=view.source_registry,
        experimental_view_receipt=view.experimental_view_receipt,
        candidate_pool_required_option_ids=(
            view.required_common_pool_option_ids
        ),
        memory_dose_contract=dose,
    )
    projection = PortfolioMemoryContextProjectionBinding.exact_identity(
        request.context_sha256
    )
    matched = PortfolioMemoryMatchedControlWavePlan(
        plan=plan,
        assignment=view.assignment,
        arm_view=view,
        aggregation=PortfolioRewardAggregationBinding(
            aggregate=lambda outcomes: float(
                max(outcome.reward for outcome in outcomes)
            ),
            aggregation_id="matched_test_max_reward",
            aggregation_version=1,
            definition_sha256="8" * 64,
        ),
        context_projection=projection,
    )
    return PortfolioVariationWaveRequest(
        selection_request=request,
        parent=_parent(contract),
        generation=5,
        label_prefix=f"matched_{view.assignment.arm.value.lower()}",
        matched_memory_control=matched,
    )


def test_matched_plan_materializes_source_identical_memory_and_neutral_views() -> None:
    contract, entry, card, registry = _entry_and_card()
    plan = PortfolioMemoryMatchedControlPlanner().plan(
        reference=entry.reference,
        exact_context_sha256="e" * 64,
        ordered_units=_units(),
        active_unit_rank=1,
    )

    first = materialize_portfolio_memory_matched_arm(
        plan=plan,
        assignment=plan.assignments[0],
        source_card=card,
        source_registry=registry,
        finite_variation_contract=contract,
    )
    second = materialize_portfolio_memory_matched_arm(
        plan=plan,
        assignment=plan.assignments[1],
        source_card=card,
        source_registry=registry,
        finite_variation_contract=contract,
    )

    assert first.assignment.arm is PortfolioExperimentalArm.NEUTRAL
    assert second.assignment.arm is PortfolioExperimentalArm.MEMORY
    assert (
        first.required_common_pool_option_ids
        == second.required_common_pool_option_ids
    )
    assert first.required_common_pool_option_ids == ("alpha.x1",)
    assert first.cards[0].reference == second.cards[0].reference == entry.reference
    assert first.cards[0].source_binding == second.cards[0].source_binding
    assert first.cards[0].prompt_payload == CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD
    assert first.cards[0].finite_action_evidence == ()
    assert second.cards == (card,)
    assert first.memory_dose_allowed is False
    assert second.memory_dose_allowed is True
    assert plan.to_record()["single_block_card_effect_identified"] is False
    assert plan.to_record()["online_score_update_allowed"] is False

    requests = tuple(
        PortfolioSelectionRequest(
            call_id=LLMCallId(f"call_matched_view_{index}"),
            operation="select_portfolio",
            instruction="Select from the same sealed finite action pool.",
            context=_object({"same_context": True}),
            finite_variation_contract=contract,
            cards=view.cards,
            portfolio_size=2,
            required_metric_ids=("cost",),
            require_supporting_cards=False,
            source_registry=view.source_registry,
            experimental_view_receipt=view.experimental_view_receipt,
            candidate_pool_required_option_ids=(
                view.required_common_pool_option_ids
            ),
        )
        for index, view in enumerate((first, second))
    )
    assert (
        requests[0].finite_variation_contract
        == requests[1].finite_variation_contract
    )
    assert requests[0].context_sha256 == requests[1].context_sha256
    assert common_pool_required_option_ids(requests[0]) == ("alpha.x1",)
    assert common_pool_required_option_ids(requests[1]) == ("alpha.x1",)


def test_matched_support_resolver_selects_one_card_supported_by_both_lanes() -> None:
    contract, _entry, card, _registry = _entry_and_card()
    lanes = tuple(
        CampaignDiagnosticSupportLaneInput(
            lane=LaneCardMatchingLane(
                lane_id=f"lane_{index}",
                lane_identity_sha256=str(index + 1) * 64,
            ),
            finite_variation_contract=contract,
        )
        for index in range(2)
    )
    supported = CampaignDiagnosticSupportCardInput(
        card=LaneCardMatchingCard(
            card_key=card.card_key,
            card_identity_sha256=card.source_binding.binding_sha256,
        ),
        semantics=PortfolioMemoryDoseCardSemantics(
            card_key=card.card_key,
            card_content_sha256=card.content_sha256,
            affected_paths=("$.x",),
            recommended_option_families=("alpha",),
            recommended_option_ids=("alpha.x1",),
        ),
    )
    resolution = PortfolioMemoryMatchedSupportResolver().resolve(
        lanes=lanes,
        cards=(supported,),
        selection_key_sha256="f" * 64,
    )

    assert resolution.eligible
    assert resolution.eligible_card_keys == (card.card_key,)
    assert resolution.selected_card_key == card.card_key
    assert tuple(
        resolution.support_for(f"lane_{index}").compatible_options[0][0]
        for index in range(2)
    ) == ("alpha.x1", "alpha.x1")
    assert resolution.to_record()["card_vs_neutral_effect_identified"] is False

    unsupported = replace(
        supported,
        semantics=replace(
            supported.semantics,
            recommended_option_ids=("alpha.absent",),
        ),
    )
    failed = PortfolioMemoryMatchedSupportResolver().resolve(
        lanes=lanes,
        cards=(unsupported,),
        selection_key_sha256="f" * 64,
    )
    assert not failed.eligible
    assert failed.selected_card_key is None
    assert failed.to_record()["eligible"] is False


def test_lane_support_resolver_preserves_a_supported_treatment_when_peer_fails() -> None:
    contract, _entry, card, _registry = _entry_and_card()
    supported_card = CampaignDiagnosticSupportCardInput(
        card=LaneCardMatchingCard(
            card_key=card.card_key,
            card_identity_sha256=card.source_binding.binding_sha256,
        ),
        semantics=PortfolioMemoryDoseCardSemantics(
            card_key=card.card_key,
            card_content_sha256=card.content_sha256,
            affected_paths=("$.x",),
            recommended_option_families=("alpha",),
            recommended_option_ids=("alpha.x1",),
        ),
    )
    supported_lane = CampaignDiagnosticSupportLaneInput(
        lane=LaneCardMatchingLane(
            lane_id="supported",
            lane_identity_sha256="1" * 64,
        ),
        finite_variation_contract=contract,
    )
    unsupported_lane = CampaignDiagnosticSupportLaneInput(
        lane=LaneCardMatchingLane(
            lane_id="unsupported",
            lane_identity_sha256="2" * 64,
        ),
        finite_variation_contract=replace(contract, options=(contract.options[1],)),
    )
    resolver = PortfolioMemoryLaneSupportResolver()

    admitted = resolver.resolve(
        lane=supported_lane,
        cards=(supported_card,),
        selection_key_sha256="a" * 64,
    )
    rejected = resolver.resolve(
        lane=unsupported_lane,
        cards=(supported_card,),
        selection_key_sha256="b" * 64,
    )

    assert admitted.eligible
    assert admitted.selected_card_key == card.card_key
    assert admitted.selected_support is not None
    assert admitted.selected_support.compatible_options[0][0] == "alpha.x1"
    assert admitted.to_record()["online_causal_credit_allowed"] is False
    assert not rejected.eligible
    assert rejected.selected_support is None


def test_matched_arm_fails_closed_without_current_exact_action_support() -> None:
    contract, entry, card, registry = _entry_and_card()
    plan = PortfolioMemoryMatchedControlPlanner().plan(
        reference=entry.reference,
        exact_context_sha256="e" * 64,
        ordered_units=_units(),
        active_unit_rank=0,
    )
    unsupported = replace(contract, options=(contract.options[1],))

    with pytest.raises(ValueError, match="complete support"):
        materialize_portfolio_memory_matched_arm(
            plan=plan,
            assignment=plan.assignments[0],
            source_card=card,
            source_registry=registry,
            finite_variation_contract=unsupported,
        )


def test_one_observed_pair_is_not_promoted_to_causal_or_online_credit() -> None:
    outcome = PortfolioMemoryMatchedControlOutcome(
        plan_sha256="1" * 64,
        generation=5,
        reference=InsightRef(InsightId("insight_matched_outcome"), 1),
        aggregation_binding_sha256="4" * 64,
        active_view_sha256="5" * 64,
        neutral_view_sha256="6" * 64,
        active_result_receipt_sha256="2" * 64,
        neutral_result_receipt_sha256="3" * 64,
        active_wave_reward=0.25,
        neutral_wave_reward=0.10,
    )

    assert outcome.observed_active_minus_neutral == pytest.approx(0.15)
    record = outcome.to_record()
    assert record["single_block_card_effect_identified"] is False
    assert record["online_score_update_allowed"] is False


def test_wave_request_binds_each_matched_arm_and_shared_action_pool() -> None:
    contract, entry, card, registry = _entry_and_card()
    context_sha256 = typed_json_sha256(_object({"same_context": True}))
    plan = PortfolioMemoryMatchedControlPlanner().plan(
        reference=entry.reference,
        exact_context_sha256=context_sha256,
        ordered_units=_units(),
        active_unit_rank=0,
    )
    views = tuple(
        materialize_portfolio_memory_matched_arm(
            plan=plan,
            assignment=assignment,
            source_card=card,
            source_registry=registry,
            finite_variation_contract=contract,
        )
        for assignment in plan.assignments
    )
    support = derive_portfolio_memory_dose_card_support(
        PortfolioMemoryDoseCardSemantics(
            card_key=card.card_key,
            card_content_sha256=card.content_sha256,
            affected_paths=("$.x",),
            recommended_option_families=("alpha",),
            recommended_option_ids=("alpha.x1",),
        ),
        contract,
    )
    waves = tuple(
        _wave_for_view(
            contract=contract,
            plan=plan,
            view=view,
            support=support,
        )
        for view in views
    )

    assert tuple(
        wave.matched_memory_control.assignment.arm for wave in waves
    ) == (
        PortfolioExperimentalArm.MEMORY,
        PortfolioExperimentalArm.NEUTRAL,
    )
    assert waves[0].selection_request.memory_dose_contract is not None
    assert waves[1].selection_request.memory_dose_contract is None
    assert {
        wave.matched_memory_control.plan.plan_sha256 for wave in waves
    } == {plan.plan_sha256}

    missing_shared_pool = replace(
        waves[1].selection_request,
        candidate_pool_required_option_ids=(),
    )
    with pytest.raises(ValueError, match="required actions differ"):
        replace(waves[1], selection_request=missing_shared_pool)

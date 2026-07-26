"""Provider-free tests for generic bounded, relevance-aware card dose."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.portfolio_memory_dose import (
    PortfolioMemoryDoseCardSemantics,
    PortfolioMemoryDoseSupportScope,
    PortfolioMemoryTransferTier,
    assess_portfolio_memory_context_transfer,
    assess_portfolio_memory_transfer_ladder,
    derive_portfolio_memory_advisory_card_support,
    derive_portfolio_memory_dose_card_support,
)
from agent_evolve.application.finite_action_transition import (
    EmpiricalFiniteActionTransition,
)
from agent_evolve.application.portfolio_memory_transfer import (
    PortfolioMemoryTransferCard,
    PortfolioMemoryTransferLane,
    PortfolioMemoryTransferLaneResolver,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.domain.typed_json import typed_json_sha256
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseMember,
    PortfolioMemoryDoseRejected,
    PortfolioMemoryDoseStage,
    PortfolioMemoryDoseViolation,
    assess_evaluated_portfolio_memory_dose,
    assess_proposed_portfolio_memory_dose,
    require_passing_portfolio_memory_dose,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _contract() -> FiniteVariationContract:
    parent = freeze_json({"shape": {"radius": 1.0}, "budget": 4, "solver": "a"})
    assert parent.__class__.__name__ == "FrozenJsonObject"
    rows = (
        (
            "option.shape.2",
            "shape",
            {"shape": {"radius": 2.0}, "budget": 4, "solver": "a"},
        ),
        (
            "option.shape.3",
            "shape",
            {"shape": {"radius": 3.0}, "budget": 4, "solver": "a"},
        ),
        (
            "option.budget.2",
            "budget",
            {"shape": {"radius": 1.0}, "budget": 2, "solver": "a"},
        ),
        (
            "option.budget.3",
            "budget",
            {"shape": {"radius": 1.0}, "budget": 3, "solver": "a"},
        ),
        (
            "option.budget.5",
            "budget",
            {"shape": {"radius": 1.0}, "budget": 5, "solver": "a"},
        ),
        (
            "option.solver.b",
            "solver",
            {"shape": {"radius": 1.0}, "budget": 4, "solver": "b"},
        ),
        (
            "option.solver.c",
            "solver",
            {"shape": {"radius": 1.0}, "budget": 4, "solver": "c"},
        ),
        # This option changes two coordinates and must not be attributed to the
        # single-path shape card despite one overlapping edit.
        (
            "option.shape_budget",
            "shape",
            {"shape": {"radius": 2.0}, "budget": 3, "solver": "a"},
        ),
        # Replacing the ancestor object is broader than the declared leaf and
        # must not be treated as leaf-scoped support.
        (
            "option.shape.replace",
            "shape",
            {"shape": "opaque", "budget": 4, "solver": "a"},
        ),
    )
    return FiniteVariationContract(
        catalog_id="dose_test_catalog",
        catalog_version=1,
        catalog_definition_sha256=_sha("dose-test-catalog"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=typed_json_sha256(parent),
                family=family,
                child_configuration=freeze_json(child),
                description=f"Opaque action {option_id}",
            )
            for option_id, family, child in rows
        ),
    )


def _dose() -> tuple[FiniteVariationContract, BoundedPortfolioMemoryDoseContract]:
    finite = _contract()
    shape = derive_portfolio_memory_dose_card_support(
        PortfolioMemoryDoseCardSemantics(
            card_key="card.shape",
            card_content_sha256=_sha("shape-card"),
            affected_paths=("$.shape.radius",),
            recommended_option_families=("shape",),
        ),
        finite,
    )
    budget = derive_portfolio_memory_dose_card_support(
        PortfolioMemoryDoseCardSemantics(
            card_key="card.budget",
            card_content_sha256=_sha("budget-card"),
            affected_paths=("$.budget",),
            recommended_option_families=("budget",),
            recommended_option_ids=("option.budget.3",),
        ),
        finite,
    )
    return finite, BoundedPortfolioMemoryDoseContract(
        card_supports=(budget, shape),
        proposed_supported_member_bounds=(2, 2),
        evaluated_supported_member_bounds=(2, 2),
        minimum_unattributed_proposed_members=6,
        minimum_unattributed_evaluated_members=2,
    )


def _member(
    finite: FiniteVariationContract,
    rank: int,
    option_id: str,
    cards: tuple[str, ...] = (),
) -> PortfolioMemoryDoseMember:
    option = finite.resolve(option_id)
    return PortfolioMemoryDoseMember(
        rank=rank,
        option_id=option.option_id,
        option_identity_sha256=option.identity_sha256,
        supporting_card_keys=cards,
    )


def test_exact_k8_to_k4_dose_preserves_two_prompt_exposed_exploration_slots() -> None:
    finite, dose = _dose()
    proposed = (
        _member(finite, 1, "option.shape.2", ("card.shape",)),
        _member(finite, 2, "option.budget.3", ("card.budget",)),
        _member(finite, 3, "option.solver.b"),
        _member(finite, 4, "option.shape.3"),
        _member(finite, 5, "option.budget.5"),
        _member(finite, 6, "option.solver.c"),
        _member(finite, 7, "option.budget.2"),
        _member(finite, 8, "option.shape_budget"),
    )
    proposal = assess_proposed_portfolio_memory_dose(dose, proposed)
    require_passing_portfolio_memory_dose(proposal)
    assert proposal.stage is PortfolioMemoryDoseStage.PROPOSED_SLATE
    assert proposal.supported_member_ranks == (1, 2)
    assert proposal.unattributed_member_ranks == (3, 4, 5, 6, 7, 8)
    assert proposal.to_record()["unattributed_members_are_blinded_controls"] is False

    evaluated = (
        _member(finite, 1, "option.shape.2", ("card.shape",)),
        _member(finite, 2, "option.budget.3", ("card.budget",)),
        _member(finite, 3, "option.solver.b"),
        _member(finite, 4, "option.shape.3"),
    )
    assessment = assess_evaluated_portfolio_memory_dose(
        dose,
        evaluated,
        proposal_assessment=proposal,
    )
    require_passing_portfolio_memory_dose(assessment)
    assert assessment.supported_member_ranks == (1, 2)
    assert assessment.unattributed_member_ranks == (3, 4)
    assert assessment.proposal_assessment_sha256 == proposal.assessment_sha256


def test_support_derivation_rejects_partial_overlap_of_joint_action() -> None:
    finite, dose = _dose()
    shape = dose.support_for("card.shape")
    assert tuple(value[0] for value in shape.compatible_options) == (
        "option.shape.2",
        "option.shape.3",
    )
    assert "option.shape_budget" not in {
        value[0] for value in shape.compatible_options
    }
    assert "option.shape.replace" not in {
        value[0] for value in shape.compatible_options
    }
    assert tuple(
        value[0] for value in dose.support_for("card.budget").compatible_options
    ) == ("option.budget.3",)


def _budget_transition_contract(
    parent_budget: int,
    *,
    parent_radius: float = 1.0,
) -> FiniteVariationContract:
    parent = freeze_json(
        {
            "shape": {"radius": parent_radius},
            "budget": parent_budget,
            "solver": "a",
        }
    )
    return FiniteVariationContract(
        catalog_id="dose_transition_catalog",
        catalog_version=1,
        catalog_definition_sha256=_sha("dose-transition-catalog"),
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="option.budget.3",
                parent_configuration_sha256=typed_json_sha256(parent),
                family="budget",
                child_configuration=freeze_json(
                    {
                        "shape": {"radius": parent_radius},
                        "budget": 3,
                        "solver": "a",
                    }
                ),
                description="Set the budget to three.",
            ),
        ),
    )


def test_empirical_support_requires_the_authenticated_parent_child_transition() -> None:
    transition = EmpiricalFiniteActionTransition(
        contrast_id=_sha("dose-transition-contrast"),
        source_observation_sha256=_sha("dose-source-observation"),
        source_evidence_id=_sha("dose-source-evidence"),
        event_index=4,
        workload_instance_sha256=_sha("dose-workload"),
        evaluator_contract_sha256=_sha("dose-evaluator"),
        campaign_sha256=_sha("dose-campaign"),
        option_id="option.budget.3",
        option_identity_sha256=_sha("dose-source-option"),
        option_family="budget",
        finite_contract_identity_sha256=_sha("dose-source-contract"),
        affected_path="$.budget",
        parent_value=freeze_json(4),
        child_value=freeze_json(3),
        parent_configuration_sha256=_sha("dose-source-parent"),
        child_configuration_sha256=_sha("dose-source-child"),
        action_semantics_compiler_id="finite_portfolio_action_semantics",
        action_semantics_compiler_version=2,
        action_semantics_definition_sha256=_sha("dose-action-semantics"),
    )
    semantics = PortfolioMemoryDoseCardSemantics(
        card_key="card.budget_transition",
        card_content_sha256=_sha("budget-transition-card"),
        affected_paths=("$.budget",),
        recommended_option_families=("budget",),
        recommended_option_ids=("option.budget.3",),
        empirical_transitions=(transition,),
    )

    on_trigger = derive_portfolio_memory_dose_card_support(
        semantics,
        _budget_transition_contract(4),
    )
    assert tuple(value[0] for value in on_trigger.compatible_options) == (
        "option.budget.3",
    )
    with pytest.raises(ValueError, match="no compatible action"):
        derive_portfolio_memory_dose_card_support(
            semantics,
            _budget_transition_contract(2),
        )


def test_exact_source_parent_scope_separates_advice_from_forced_replay() -> None:
    source = _budget_transition_contract(4)
    shifted = _budget_transition_contract(4, parent_radius=2.0)
    source_option = source.resolve("option.budget.3")
    transition = EmpiricalFiniteActionTransition(
        contrast_id=_sha("exact-context-contrast"),
        source_observation_sha256=_sha("exact-context-observation"),
        source_evidence_id=_sha("exact-context-evidence"),
        event_index=7,
        workload_instance_sha256=_sha("exact-context-workload"),
        evaluator_contract_sha256=_sha("exact-context-evaluator"),
        campaign_sha256=_sha("exact-context-campaign"),
        option_id=source_option.option_id,
        option_identity_sha256=source_option.identity_sha256,
        option_family=source_option.family,
        finite_contract_identity_sha256=source.identity_sha256,
        affected_path="$.budget",
        parent_value=freeze_json(4),
        child_value=freeze_json(3),
        parent_configuration_sha256=source.parent_configuration_sha256,
        child_configuration_sha256=source_option.child_configuration_sha256,
        action_semantics_compiler_id="finite_portfolio_action_semantics",
        action_semantics_compiler_version=2,
        action_semantics_definition_sha256=_sha("exact-context-semantics"),
    )
    local = PortfolioMemoryDoseCardSemantics(
        card_key="card.exact_context",
        card_content_sha256=_sha("exact-context-card"),
        affected_paths=("$.budget",),
        recommended_option_families=("budget",),
        recommended_option_ids=("option.budget.3",),
        empirical_transitions=(transition,),
    )
    exact = replace(
        local,
        support_scope=PortfolioMemoryDoseSupportScope.EXACT_SOURCE_PARENT,
    )

    # The shifted parent retains the same local budget precondition, so the
    # transition remains meaningful advice but is not authorized as a dose.
    assert derive_portfolio_memory_dose_card_support(local, shifted)
    with pytest.raises(ValueError, match="no compatible action"):
        derive_portfolio_memory_dose_card_support(exact, shifted)
    shifted_assessment = assess_portfolio_memory_context_transfer(exact, shifted)
    assert shifted_assessment.local_intervention_support_available is True
    assert shifted_assessment.exact_source_parent_match is False
    assert shifted_assessment.exact_action_replay_authorized is False
    assert shifted_assessment.to_record()["transfer_authority"] == "advisory_only"

    exact_support = derive_portfolio_memory_dose_card_support(exact, source)
    assert exact_support.support_policy_id.startswith("exact_parent_")
    source_assessment = assess_portfolio_memory_context_transfer(exact, source)
    assert source_assessment.exact_source_parent_match is True
    assert source_assessment.exact_action_replay_authorized is True


def test_typed_transfer_ladder_preserves_replay_and_advisory_authority() -> None:
    source = _budget_transition_contract(4)
    shifted = _budget_transition_contract(4, parent_radius=2.0)
    path_analogy = _budget_transition_contract(2, parent_radius=2.0)
    source_option = source.resolve("option.budget.3")
    transition = EmpiricalFiniteActionTransition(
        contrast_id=_sha("ladder-contrast"),
        source_observation_sha256=_sha("ladder-observation"),
        source_evidence_id=_sha("ladder-evidence"),
        event_index=8,
        workload_instance_sha256=_sha("ladder-workload"),
        evaluator_contract_sha256=_sha("ladder-evaluator"),
        campaign_sha256=_sha("ladder-campaign"),
        option_id=source_option.option_id,
        option_identity_sha256=source_option.identity_sha256,
        option_family=source_option.family,
        finite_contract_identity_sha256=source.identity_sha256,
        affected_path="$.budget",
        parent_value=freeze_json(4),
        child_value=freeze_json(3),
        parent_configuration_sha256=source.parent_configuration_sha256,
        child_configuration_sha256=source_option.child_configuration_sha256,
        action_semantics_compiler_id="finite_portfolio_action_semantics",
        action_semantics_compiler_version=2,
        action_semantics_definition_sha256=_sha("ladder-semantics"),
    )
    semantics = PortfolioMemoryDoseCardSemantics(
        card_key="card.transfer_ladder",
        card_content_sha256=_sha("transfer-ladder-card"),
        affected_paths=("$.budget",),
        recommended_option_families=("budget",),
        recommended_option_ids=("option.budget.3",),
        empirical_transitions=(transition,),
        support_scope=PortfolioMemoryDoseSupportScope.EXACT_SOURCE_PARENT,
    )

    exact = assess_portfolio_memory_transfer_ladder(semantics, source)
    assert exact.tier is PortfolioMemoryTransferTier.EXACT_ACTION_REPLAY
    assert exact.exact_replay_option_ids == ("option.budget.3",)
    assert exact.exact_action_replay_authorized is True
    assert exact.causal_memory_credit_authorized is False

    local = assess_portfolio_memory_transfer_ladder(semantics, shifted)
    assert local.tier is PortfolioMemoryTransferTier.LOCAL_ACTION_ADVISORY
    assert local.local_advisory_option_ids == ("option.budget.3",)
    assert local.exact_replay_option_ids == ()
    assert local.advisory_delivery_authorized is True
    assert local.to_record()["transfer_calibration"] == "uncalibrated_advisory"

    analogy = assess_portfolio_memory_transfer_ladder(semantics, path_analogy)
    assert analogy.tier is PortfolioMemoryTransferTier.PATH_FAMILY_ADVISORY
    assert analogy.path_family_advisory_option_ids == ("option.budget.3",)
    assert analogy.local_advisory_option_ids == ()
    assert analogy.to_record()["historical_option_identity_relaxed"] is True
    support = derive_portfolio_memory_advisory_card_support(
        semantics,
        path_analogy,
    )
    assert tuple(value[0] for value in support.compatible_options) == (
        "option.budget.3",
    )
    assert support.support_policy_id == "portfolio_memory_typed_transfer_ladder"


def test_transfer_lane_resolver_exposes_best_available_advisory_tier() -> None:
    contract = _budget_transition_contract(2, parent_radius=2.0)
    supported = PortfolioMemoryTransferCard(
        card_key="card.budget_advisory",
        card_identity_sha256=_sha("budget-advisory-identity"),
        semantics=PortfolioMemoryDoseCardSemantics(
            card_key="card.budget_advisory",
            card_content_sha256=_sha("budget-advisory-content"),
            affected_paths=("$.budget",),
            recommended_option_families=("budget",),
        ),
    )
    unsupported = PortfolioMemoryTransferCard(
        card_key="card.shape_advisory",
        card_identity_sha256=_sha("shape-advisory-identity"),
        semantics=PortfolioMemoryDoseCardSemantics(
            card_key="card.shape_advisory",
            card_content_sha256=_sha("shape-advisory-content"),
            affected_paths=("$.shape.radius",),
            recommended_option_families=("shape",),
        ),
    )
    resolution = PortfolioMemoryTransferLaneResolver().resolve(
        lane=PortfolioMemoryTransferLane(
            lane_id="elite",
            lane_identity_sha256=_sha("elite-lane"),
            finite_variation_contract=contract,
        ),
        cards=(supported, unsupported),
        selection_key_sha256=_sha("transfer-selection"),
    )
    assert resolution.eligible is True
    assert resolution.selected_card_key == "card.budget_advisory"
    assert resolution.selected_assessment is not None
    assert (
        resolution.selected_assessment.tier
        is PortfolioMemoryTransferTier.PATH_FAMILY_ADVISORY
    )
    assert resolution.selected_support is not None
    assert resolution.to_record()["online_causal_credit_allowed"] is False


@pytest.mark.parametrize(
    ("proposed", "violation"),
    (
        (
            (
                ("option.shape.2", ("card.shape",)),
                ("option.budget.5", ("card.budget",)),
                ("option.solver.b", ()),
                ("option.shape.3", ()),
                ("option.budget.3", ()),
            ),
            PortfolioMemoryDoseViolation.INCOMPATIBLE_CARD_ACTION,
        ),
        (
            (
                ("option.shape.2", ("card.budget", "card.shape")),
                ("option.budget.3", ()),
                ("option.solver.b", ()),
                ("option.shape.3", ()),
                ("option.budget.5", ()),
            ),
            PortfolioMemoryDoseViolation.TOO_MANY_CARDS_PER_MEMBER,
        ),
        (
            (
                ("option.shape.2", ("card.shape",)),
                ("option.budget.3", ()),
                ("option.solver.b", ()),
                ("option.shape.3", ()),
                ("option.budget.5", ()),
            ),
            PortfolioMemoryDoseViolation.ASSIGNED_CARD_OMITTED,
        ),
        (
            (
                ("option.shape.2", ("card.shape",)),
                ("option.budget.3", ("card.budget",)),
                ("option.solver.b", ("card.foreign",)),
                ("option.shape.3", ()),
                ("option.budget.5", ()),
            ),
            PortfolioMemoryDoseViolation.FOREIGN_CARD_ATTRIBUTION,
        ),
    ),
)
def test_proposal_failures_are_closed_receipts(
    proposed: tuple[tuple[str, tuple[str, ...]], ...],
    violation: PortfolioMemoryDoseViolation,
) -> None:
    finite, dose = _dose()
    members = tuple(
        _member(finite, rank, option_id, cards)
        for rank, (option_id, cards) in enumerate(proposed, start=1)
    )
    assessment = assess_proposed_portfolio_memory_dose(dose, members)
    assert not assessment.passed
    assert violation in assessment.violations
    with pytest.raises(PortfolioMemoryDoseRejected) as raised:
        require_passing_portfolio_memory_dose(assessment)
    assert raised.value.assessment is assessment


def test_evaluated_dose_cannot_rewrite_an_uncited_proposal_member() -> None:
    finite, dose = _dose()
    proposed = (
        _member(finite, 1, "option.shape.2", ("card.shape",)),
        _member(finite, 2, "option.budget.3", ("card.budget",)),
        _member(finite, 3, "option.solver.b"),
        _member(finite, 4, "option.shape.3"),
        _member(finite, 5, "option.budget.5"),
        _member(finite, 6, "option.solver.c"),
        _member(finite, 7, "option.budget.2"),
        _member(finite, 8, "option.shape_budget"),
    )
    proposal = assess_proposed_portfolio_memory_dose(dose, proposed)
    require_passing_portfolio_memory_dose(proposal)
    evaluated = (
        _member(finite, 1, "option.shape.3", ("card.shape",)),
        _member(finite, 2, "option.budget.3", ("card.budget",)),
        _member(finite, 3, "option.solver.b"),
        _member(finite, 4, "option.budget.5"),
    )
    assessment = assess_evaluated_portfolio_memory_dose(
        dose,
        evaluated,
        proposal_assessment=proposal,
    )
    assert PortfolioMemoryDoseViolation.EVALUATED_MEMBER_NOT_IN_PROPOSAL in (
        assessment.violations
    )

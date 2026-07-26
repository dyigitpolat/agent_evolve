from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.campaign_diagnostic_blocks import (
    CampaignDiagnosticCompatibilityAudit,
    CampaignDiagnosticCompatibilityStatus,
    CampaignDiagnosticCompleteSupportCohortSelection,
    CampaignDiagnosticCompleteSupportResolver,
    CampaignDiagnosticCohortSelectionStatus,
    CampaignDiagnosticSingletonBlockPlanner,
    CampaignDiagnosticSupportCardInput,
    CampaignDiagnosticSupportLaneInput,
)
from agent_evolve.application.portfolio_memory_dose import (
    PortfolioMemoryDoseCardSemantics,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import InsightId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.policies.memory.balanced_subset_blocks import (
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.memory.compatibility_matching import (
    LaneCardCompatibility,
    LaneCardMatchingCard,
    LaneCardMatchingInput,
    LaneCardMatchingLane,
)
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256


def _reference(index: int) -> InsightRef:
    return InsightRef(InsightId(f"insight_x{index:02d}"), 1)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii", errors="strict")).hexdigest()


def _units(count: int) -> tuple[StableMemoryAssignmentUnit, ...]:
    return tuple(
        StableMemoryAssignmentUnit(
            unit_key=f"diagnostic.g03.lane{index:02d}",
            generation=3,
            lane_id=f"lane{index:02d}",
        )
        for index in range(count)
    )


@pytest.mark.parametrize(
    ("active_count", "control_count", "lane_count", "rank"),
    ((1, 1, 2, 1), (2, 0, 2, 0), (1, 2, 3, 5)),
)
def test_complete_singleton_block_gives_every_card_treated_and_control_support(
    active_count: int,
    control_count: int,
    lane_count: int,
    rank: int,
) -> None:
    references = tuple(_reference(value) for value in range(lane_count))
    block = CampaignDiagnosticSingletonBlockPlanner().plan(
        active_references=references[:active_count],
        control_references=references[active_count : active_count + control_count],
        exact_context_sha256="a" * 64,
        estimand_stratum_sha256="b" * 64,
        ordered_units=_units(lane_count),
        full_block_permutation_rank=rank,
    )

    assert len(block.assignment_plan.assignments) == lane_count
    assert all(value.treated_count == 1 for value in block.assignment_plan.support)
    assert all(
        value.control_count == lane_count - 1 for value in block.assignment_plan.support
    )
    assert block.to_record()["plan_sha256"] == block.assignment_plan.receipt_sha256


def test_singleton_block_rejects_underfilled_or_overfilled_lanes() -> None:
    with pytest.raises(ValueError, match="exactly fill"):
        CampaignDiagnosticSingletonBlockPlanner().plan(
            active_references=(_reference(0),),
            control_references=(),
            exact_context_sha256="a" * 64,
            estimand_stratum_sha256="b" * 64,
            ordered_units=_units(2),
            full_block_permutation_rank=0,
        )


def _matching_input(*, missing_last_edge: bool = False) -> LaneCardMatchingInput:
    lanes = tuple(
        LaneCardMatchingLane(
            lane_id=f"lane_{index}",
            lane_identity_sha256=str(index + 1) * 64,
        )
        for index in range(2)
    )
    cards = tuple(
        LaneCardMatchingCard(
            card_key=f"card_{index}",
            card_identity_sha256=str(index + 3) * 64,
        )
        for index in range(2)
    )
    pairs = tuple(
        (lane, card)
        for lane in lanes
        for card in cards
    )
    if missing_last_edge:
        pairs = pairs[:-1]
    return LaneCardMatchingInput(
        lanes=lanes,
        cards=cards,
        compatibilities=tuple(
            LaneCardCompatibility(
                lane_id=lane.lane_id,
                card_key=card.card_key,
                compatibility_evidence_sha256=("a" if index % 2 == 0 else "b") * 64,
            )
            for index, (lane, card) in enumerate(pairs)
        ),
    )


def test_complete_compatibility_audit_authorizes_balanced_randomization() -> None:
    audit = CampaignDiagnosticCompatibilityAudit(_matching_input())

    assert audit.status is CampaignDiagnosticCompatibilityStatus.ELIGIBLE
    assert audit.eligible
    assert audit.missing_pairs == ()
    record = audit.to_record()
    assert record["complete_bipartite_support"] is True
    assert record["randomized_singleton_assignment_allowed"] is True
    assert record["card_vs_neutral_effect_identified"] is False
    assert record["causal_credit_allowed"] is False
    assert record["online_score_update_allowed"] is False
    assert record["observed_positive_edge_count"] == 4
    assert record["audit_sha256"] == audit.audit_sha256


def test_incomplete_compatibility_audit_fails_closed_despite_full_matching() -> None:
    value = _matching_input(missing_last_edge=True)
    audit = CampaignDiagnosticCompatibilityAudit(value)

    assert audit.status is CampaignDiagnosticCompatibilityStatus.INELIGIBLE
    assert not audit.eligible
    assert audit.missing_pairs == (("lane_1", "card_1"),)
    assert (
        audit.to_record()["randomized_singleton_assignment_allowed"] is False
    )
    assert audit.to_record()["causal_credit_allowed"] is False


def test_complete_support_cohort_is_selected_before_assignment() -> None:
    base = _matching_input()
    third = LaneCardMatchingCard(
        card_key="card_2",
        card_identity_sha256="5" * 64,
    )
    expanded = LaneCardMatchingInput(
        lanes=base.lanes,
        cards=(*base.cards, third),
        compatibilities=tuple(
            sorted(
                (
                    *base.compatibilities,
                    LaneCardCompatibility(
                        lane_id="lane_0",
                        card_key="card_2",
                        compatibility_evidence_sha256="c" * 64,
                    ),
                ),
                key=lambda value: (value.lane_id, value.card_key),
            )
        ),
    )
    selection = CampaignDiagnosticCompleteSupportCohortSelection(
        matching_input=expanded,
        cohort_size=2,
        selection_key_sha256="d" * 64,
    )

    assert selection.status is CampaignDiagnosticCohortSelectionStatus.ELIGIBLE
    assert selection.full_support_card_keys == ("card_0", "card_1")
    assert selection.selected_card_keys == ("card_0", "card_1")
    assert selection.selected_matching_input is not None
    audit = CampaignDiagnosticCompatibilityAudit(
        selection.selected_matching_input
    )
    assert audit.eligible
    assert selection.to_record()["provider_fields_consulted"] is False
    assert selection.to_record()["outcome_values_consulted"] is False


def test_complete_support_cohort_fails_closed_when_positivity_is_insufficient() -> None:
    selection = CampaignDiagnosticCompleteSupportCohortSelection(
        matching_input=_matching_input(missing_last_edge=True),
        cohort_size=2,
        selection_key_sha256="d" * 64,
    )

    assert not selection.eligible
    assert selection.selected_card_keys == ()
    assert selection.selected_matching_input is None


def _finite_contract(*, include_solver_c: bool) -> FiniteVariationContract:
    parent = freeze_json({"shape": 1, "budget": 4, "solver": "a"})
    rows = [
        ("option.shape.2", "shape", {"shape": 2, "budget": 4, "solver": "a"}),
        ("option.budget.3", "budget", {"shape": 1, "budget": 3, "solver": "a"}),
    ]
    if include_solver_c:
        rows.append(
            ("option.solver.c", "solver", {"shape": 1, "budget": 4, "solver": "c"})
        )
    return FiniteVariationContract(
        catalog_id="diagnostic_support_test",
        catalog_version=1,
        catalog_definition_sha256=_sha("diagnostic-support-test"),
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


def _support_card(
    card_key: str,
    *,
    family: str,
    path: str,
    option_id: str,
) -> CampaignDiagnosticSupportCardInput:
    return CampaignDiagnosticSupportCardInput(
        card=LaneCardMatchingCard(
            card_key=card_key,
            card_identity_sha256=_sha(f"binding:{card_key}"),
        ),
        semantics=PortfolioMemoryDoseCardSemantics(
            card_key=card_key,
            card_content_sha256=_sha(f"content:{card_key}"),
            affected_paths=(path,),
            recommended_option_families=(family,),
            recommended_option_ids=(option_id,),
        ),
    )


def test_generic_support_resolver_selects_only_lane_complete_exact_actions() -> None:
    lanes = tuple(
        CampaignDiagnosticSupportLaneInput(
            lane=LaneCardMatchingLane(
                lane_id=f"lane_{index}",
                lane_identity_sha256=_sha(f"lane:{index}"),
            ),
            finite_variation_contract=_finite_contract(
                include_solver_c=index == 0
            ),
        )
        for index in range(2)
    )
    cards = (
        _support_card(
            "card_budget",
            family="budget",
            path="$.budget",
            option_id="option.budget.3",
        ),
        _support_card(
            "card_shape",
            family="shape",
            path="$.shape",
            option_id="option.shape.2",
        ),
        _support_card(
            "card_solver",
            family="solver",
            path="$.solver",
            option_id="option.solver.c",
        ),
    )

    resolution = CampaignDiagnosticCompleteSupportResolver().resolve(
        lanes=lanes,
        cards=cards,
        cohort_size=2,
        selection_key_sha256=_sha("external-selection-key"),
    )

    assert resolution.eligible
    assert resolution.cohort_selection.full_support_card_keys == (
        "card_budget",
        "card_shape",
    )
    assert resolution.cohort_selection.selected_card_keys == (
        "card_budget",
        "card_shape",
    )
    assert resolution.matching.is_full
    assert resolution.compatibility_audit.eligible
    assert resolution.support_for("lane_1", "card_shape").compatible_options[0][0] == (
        "option.shape.2"
    )
    assert tuple(
        (value.lane_id, value.card_key) for value in resolution.rejected_edges
    ) == (("lane_1", "card_solver"),)
    record = resolution.to_record()
    assert record["provider_fields_consulted"] is False
    assert record["outcome_values_consulted"] is False
    assert record["receipt_sha256"] == resolution.receipt_sha256


def test_generic_support_resolver_fails_closed_without_two_complete_cards() -> None:
    lanes = tuple(
        CampaignDiagnosticSupportLaneInput(
            lane=LaneCardMatchingLane(
                lane_id=f"lane_{index}",
                lane_identity_sha256=_sha(f"short-lane:{index}"),
            ),
            finite_variation_contract=_finite_contract(
                include_solver_c=index == 0
            ),
        )
        for index in range(2)
    )
    cards = (
        _support_card(
            "card_shape",
            family="shape",
            path="$.shape",
            option_id="option.shape.2",
        ),
        _support_card(
            "card_solver",
            family="solver",
            path="$.solver",
            option_id="option.solver.c",
        ),
    )

    resolution = CampaignDiagnosticCompleteSupportResolver().resolve(
        lanes=lanes,
        cards=cards,
        cohort_size=2,
        selection_key_sha256=_sha("external-selection-key"),
    )

    assert not resolution.eligible
    assert not resolution.cohort_selection.eligible
    assert not resolution.compatibility_audit.eligible

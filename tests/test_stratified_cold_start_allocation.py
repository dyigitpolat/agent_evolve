from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.stratified_cold_start_allocation import (
    LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_DEFINITION_SHA256,
    LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_ID,
    STRATIFIED_COLD_START_ALLOCATOR_DEFINITION_SHA256,
    StratifiedColdStartAllocationRequest,
    StratifiedColdStartProposal,
    SupportProportionalLowDiscrepancyStratifiedAllocator,
    SupportProportionalStratifiedColdStartAllocator,
)
from agent_evolve.domain.ids import CandidateId


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _proposal(
    expert: str,
    rank: int,
    *,
    role: str,
    scale: int,
    phenotype: str | None = None,
) -> StratifiedColdStartProposal:
    return StratifiedColdStartProposal(
        proposal_id=f"proposal_{expert}_{rank}",
        phenotype_identity_sha256=_sha(
            phenotype or f"phenotype:{expert}:{rank}"
        ),
        expert_id=expert,
        native_rank=rank,
        parent_ids=(CandidateId(f"candidate_parent_{expert}_{rank}"),),
        role_id=role,
        variation_scale=scale,
        structural_cell=f"{role}.r{scale}",
    )


def _v29_shaped_universe() -> tuple[StratifiedColdStartProposal, ...]:
    interaction = tuple(
        _proposal(
            "residual_interaction",
            rank,
            role="interaction",
            scale=2,
        )
        for rank in range(1, 9)
    )
    local = tuple(
        _proposal(
            "residual_local_exploit",
            rank,
            role="local_exploit" if rank <= 3 else "structural_coverage",
            scale=1,
        )
        for rank in range(1, 7)
    )
    counterfactual = (
        _proposal(
            "residual_counterfactual_coverage",
            3,
            role="structural_coverage",
            scale=1,
        ),
        _proposal(
            "residual_counterfactual_coverage",
            5,
            role="structural_coverage",
            scale=1,
        ),
        _proposal(
            "residual_counterfactual_coverage",
            6,
            role="restart",
            scale=2,
        ),
    )
    return interaction + local + counterfactual


def _request(
    proposals: tuple[StratifiedColdStartProposal, ...],
    slots: int,
    *,
    decision_index: int = 1,
) -> StratifiedColdStartAllocationRequest:
    return StratifiedColdStartAllocationRequest(
        decision_scope_sha256=_sha("sealed-v29-shaped-universe"),
        decision_index=decision_index,
        proposals=proposals,
        evaluation_slots=slots,
    )


def test_k2_covers_two_largest_experts_without_comparing_native_scores() -> None:
    decision = SupportProportionalStratifiedColdStartAllocator().select(
        _request(_v29_shaped_universe(), 2)
    )

    assert tuple(
        (value.proposal.expert_id, value.proposal.native_rank)
        for value in decision.members
    ) == (
        ("residual_interaction", 1),
        ("residual_local_exploit", 1),
    )
    assert {
        value.expert_id: value.allocated_slots for value in decision.lanes
    } == {
        "residual_counterfactual_coverage": 0,
        "residual_interaction": 1,
        "residual_local_exploit": 1,
    }


def test_k4_preserves_lane_floor_and_samples_interaction_midrank() -> None:
    decision = SupportProportionalStratifiedColdStartAllocator().select(
        _request(_v29_shaped_universe(), 4)
    )

    assert tuple(
        (value.proposal.expert_id, value.proposal.native_rank)
        for value in decision.members
    ) == (
        ("residual_interaction", 1),
        ("residual_local_exploit", 1),
        ("residual_counterfactual_coverage", 3),
        ("residual_interaction", 5),
    )
    assert {
        value.expert_id: value.allocated_slots for value in decision.lanes
    } == {
        "residual_counterfactual_coverage": 1,
        "residual_interaction": 2,
        "residual_local_exploit": 1,
    }
    record = decision.to_record()
    assert record["outcomes_consulted"] is False
    assert record["allocator"]["definition_sha256"] == (
        STRATIFIED_COLD_START_ALLOCATOR_DEFINITION_SHA256
    )
    assert record["coverage"] == {
        "selected_expert_count": 3,
        "selected_role_count": 3,
        "selected_variation_scale_count": 2,
        "selected_parent_count": 4,
        "selected_structural_cell_count": 3,
    }


def test_request_and_decision_are_permutation_invariant() -> None:
    proposals = _v29_shaped_universe()
    forward = _request(proposals, 4)
    reverse = _request(tuple(reversed(proposals)), 4)

    allocator = SupportProportionalStratifiedColdStartAllocator()
    assert forward.request_sha256 == reverse.request_sha256
    assert allocator.select(forward) == allocator.select(reverse)


def test_duplicate_materialized_phenotypes_fail_before_allocation() -> None:
    left = _proposal(
        "expert_left",
        1,
        role="local_exploit",
        scale=1,
        phenotype="shared",
    )
    right = _proposal(
        "expert_right",
        1,
        role="interaction",
        scale=2,
        phenotype="shared",
    )

    with pytest.raises(
        ValueError,
        match="collapse duplicate phenotypes",
    ):
        _request((left, right), 1)


def test_low_discrepancy_first_block_preserves_head_and_interior_coverage() -> None:
    decision = SupportProportionalLowDiscrepancyStratifiedAllocator().select(
        _request(_v29_shaped_universe(), 4)
    )

    assert tuple(
        (value.proposal.expert_id, value.proposal.native_rank)
        for value in decision.members
    ) == (
        ("residual_interaction", 1),
        ("residual_local_exploit", 1),
        ("residual_counterfactual_coverage", 3),
        ("residual_interaction", 5),
    )
    record = decision.to_record()
    assert record["allocator"] == {
        "allocator_id": LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_ID,
        "allocator_version": 1,
        "definition_sha256": (
            LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_DEFINITION_SHA256
        ),
    }


def test_low_discrepancy_second_block_moves_every_lane_off_its_head() -> None:
    decision = SupportProportionalLowDiscrepancyStratifiedAllocator().select(
        _request(_v29_shaped_universe(), 4, decision_index=2)
    )

    assert tuple(
        (value.proposal.expert_id, value.proposal.native_rank)
        for value in decision.members
    ) == (
        ("residual_interaction", 3),
        ("residual_local_exploit", 4),
        ("residual_counterfactual_coverage", 5),
        ("residual_interaction", 7),
    )


def test_single_lane_schedule_closes_all_six_native_rank_strata() -> None:
    proposals = tuple(
        _proposal(
            "residual_local_exploit",
            rank,
            role="local_exploit",
            scale=1,
        )
        for rank in range(1, 7)
    )
    allocator = SupportProportionalLowDiscrepancyStratifiedAllocator()

    selected = tuple(
        allocator.select(
            _request(proposals, 1, decision_index=decision_index)
        ).members[0].proposal.native_rank
        for decision_index in range(1, 7)
    )

    assert selected == (1, 4, 2, 5, 3, 6)
    assert set(selected) == set(range(1, 7))

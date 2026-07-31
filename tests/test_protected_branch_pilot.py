from __future__ import annotations

import asyncio
import hashlib

import pytest

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.protected_branch_pilot import (
    ProtectedBranchBinding,
    ProtectedBranchPilotPolicy,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json, thaw_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _fixture(
    *,
    duplicate_across_branches: bool = False,
    evaluation_slots: int = 6,
):
    expert_ids = (
        "branch_a.coverage",
        "branch_a.local",
        "branch_b.coverage",
        "branch_b.local",
        "numerical",
    )
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("protected-branch-campaign"),
        prior_state_sha256=_sha("protected-branch-prior"),
        decision_index=1,
        phase=SearchPhase.BASIN_EXPANSION,
        remaining_decisions=3,
        remaining_evaluations=24,
        evaluation_slots=evaluation_slots,
        expert_proposal_slots=tuple((value, 3) for value in expert_ids),
        proposal_context=freeze_json({"test": "protected-branch"}),
        reference_escrow_slots=0,
    )
    context = MaterializedActionContext(
        campaign_scope_sha256=request.campaign_scope_sha256,
        decision_index=request.decision_index,
        phase=request.phase,
        remaining_decisions=request.remaining_decisions,
        remaining_evaluations=request.remaining_evaluations,
        residual_frontier_cell="test.frontier",
        parent_position_cell="test.parent",
        archive_relation_cell="unknown_pre_eval",
        structural_signature_sha256=_sha("protected-branch-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.unknown",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    proposals = []
    for expert_id in expert_ids:
        actions = []
        for rank in range(1, 4):
            phenotype_label = f"{expert_id}:{rank}"
            if (
                duplicate_across_branches
                and rank == 1
                and expert_id in ("branch_a.coverage", "branch_b.coverage")
            ):
                phenotype_label = "cross-branch-duplicate"
            actions.append(
                MaterializedActionDescriptor(
                    context=context,
                    configuration=freeze_json(
                        {"expert_id": expert_id, "rank": rank}
                    ),
                    phenotype_identity_sha256=_sha(phenotype_label),
                    expert_id=expert_id,
                    native_rank=rank,
                    parent_ids=(CandidateId("candidate_shared_parent"),),
                    operator_id="test_mutation",
                    target_candidate_id=CandidateId(
                        f"candidate_{expert_id.replace('.', '_')}_{rank}"
                    ),
                    role_id="test_challenger",
                    normalized_evaluation_cost=1.0,
                    reference_action=False,
                )
            )
        proposals.append(
            MaterializedActionProposalBatch(
                request_sha256=request.request_sha256,
                expert_id=expert_id,
                expert_version=1,
                expert_definition_sha256=_sha(f"expert:{expert_id}"),
                actions=tuple(actions),
                evidence=freeze_json(
                    {
                        "candidate_outcomes_observed": False,
                        "test_market": True,
                    }
                ),
            )
        )
    policy = ProtectedBranchPilotPolicy(
        branch_bindings=(
            ProtectedBranchBinding(
                branch_id="branch_a",
                expert_ids=("branch_a.coverage", "branch_a.local"),
                pilot_slots=2,
            ),
            ProtectedBranchBinding(
                branch_id="branch_b",
                expert_ids=("branch_b.coverage", "branch_b.local"),
                pilot_slots=2,
            ),
        )
    )
    return request, tuple(proposals), policy


def _action_market(proposals):
    return {
        action.action_sha256: action
        for proposal in proposals
        for action in proposal.actions
    }


def test_protected_branch_pilots_are_balanced_replayable_and_leave_residual():
    request, proposals, policy = _fixture()
    first = asyncio.run(policy.require(request, proposals))
    second = asyncio.run(policy.require(request, proposals))

    assert first.requirement_sha256 == second.requirement_sha256
    assert len(first.required_action_sha256s) == 4
    market = _action_market(proposals)
    selected = [market[value] for value in first.required_action_sha256s]
    assert len({value.phenotype_identity_sha256 for value in selected}) == 4
    assert sum(value.expert_id.startswith("branch_a.") for value in selected) == 2
    assert sum(value.expert_id.startswith("branch_b.") for value in selected) == 2

    evidence = thaw_json(first.evidence)
    certificate = evidence["market_capacity_certificate"]
    assert certificate["evaluation_slots"] == request.evaluation_slots
    assert certificate["protected_pilot_count"] == 4
    assert certificate["residual_seats"] == 2
    assert certificate["global_completion_witness"] is True
    assert evidence["unbound_expert_ids"] == ["numerical"]
    assert all(
        row["candidate_outcomes_observed"] is False
        for row in evidence["selection_trace"]
    )


def test_cross_branch_phenotype_collision_is_backfilled_within_branch():
    request, proposals, policy = _fixture(duplicate_across_branches=True)
    requirement = asyncio.run(policy.require(request, proposals))
    market = _action_market(proposals)
    selected = [market[value] for value in requirement.required_action_sha256s]

    assert len(selected) == 4
    assert len({value.phenotype_identity_sha256 for value in selected}) == 4
    assert sum(value.expert_id.startswith("branch_a.") for value in selected) == 2
    assert sum(value.expert_id.startswith("branch_b.") for value in selected) == 2


def test_policy_fails_closed_when_protected_floors_exceed_k():
    request, proposals, _ = _fixture(evaluation_slots=3)
    policy = ProtectedBranchPilotPolicy(
        branch_bindings=(
            ProtectedBranchBinding(
                branch_id="branch_a",
                expert_ids=("branch_a.coverage", "branch_a.local"),
                pilot_slots=2,
            ),
            ProtectedBranchBinding(
                branch_id="branch_b",
                expert_ids=("branch_b.coverage", "branch_b.local"),
                pilot_slots=2,
            ),
        )
    )

    with pytest.raises(ValueError, match="floors exceed"):
        asyncio.run(policy.require(request, proposals))


def test_branch_bindings_are_a_partition():
    with pytest.raises(ValueError, match="cannot belong to two branches"):
        ProtectedBranchPilotPolicy(
            branch_bindings=(
                ProtectedBranchBinding(
                    branch_id="branch_a",
                    expert_ids=("shared.expert",),
                    pilot_slots=1,
                ),
                ProtectedBranchBinding(
                    branch_id="branch_b",
                    expert_ids=("shared.expert",),
                    pilot_slots=1,
                ),
            )
        )

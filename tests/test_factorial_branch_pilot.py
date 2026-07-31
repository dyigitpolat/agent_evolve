from __future__ import annotations

import asyncio
import hashlib

import pytest

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.factorial_branch_pilot import (
    FactorialBranchPilotPolicy,
    factorial_candidates_from_materialized_actions,
    select_factorial_pilot_candidates,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.protected_branch_pilot import (
    ProtectedBranchBinding,
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
    evaluation_slots: int = 8,
    pilot_slots_per_branch: int = 3,
    duplicate_across_branches: bool = False,
):
    experts = (
        ("branch_a.coverage", "coverage"),
        ("branch_a.interaction", "interaction"),
        ("branch_a.local", "local"),
        ("branch_b.coverage", "coverage"),
        ("branch_b.interaction", "interaction"),
        ("branch_b.local", "local"),
        ("numerical", "numerical"),
    )
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("factorial-pilot-campaign"),
        prior_state_sha256=_sha("factorial-pilot-prior"),
        decision_index=1,
        phase=SearchPhase.BASIN_EXPANSION,
        remaining_decisions=3,
        remaining_evaluations=24,
        evaluation_slots=evaluation_slots,
        expert_proposal_slots=tuple(
            (expert_id, 6) for expert_id, _ in experts
        ),
        proposal_context=freeze_json({"test": "factorial-pilot"}),
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
        structural_signature_sha256=_sha("factorial-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.unknown",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    proposals = []
    for expert_id, role_id in experts:
        actions = []
        for rank in range(1, 7):
            phenotype = f"{expert_id}:{rank}"
            if (
                duplicate_across_branches
                and rank == 1
                and expert_id
                in ("branch_a.coverage", "branch_b.coverage")
            ):
                phenotype = "cross-branch-duplicate"
            actions.append(
                MaterializedActionDescriptor(
                    context=context,
                    configuration=freeze_json(
                        {"expert_id": expert_id, "rank": rank}
                    ),
                    phenotype_identity_sha256=_sha(phenotype),
                    expert_id=expert_id,
                    native_rank=rank,
                    parent_ids=(
                        CandidateId(
                            f"candidate_parent_{role_id}_{(rank - 1) % 3}"
                        ),
                    ),
                    operator_id=f"{role_id}.r{1 + rank % 2}",
                    target_candidate_id=CandidateId(
                        f"candidate_{expert_id.replace('.', '_')}_{rank}"
                    ),
                    role_id=role_id,
                    normalized_evaluation_cost=1.0,
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
    policy = FactorialBranchPilotPolicy(
        branch_bindings=(
            ProtectedBranchBinding(
                branch_id="branch_a",
                expert_ids=(
                    "branch_a.coverage",
                    "branch_a.interaction",
                    "branch_a.local",
                ),
                pilot_slots=pilot_slots_per_branch,
            ),
            ProtectedBranchBinding(
                branch_id="branch_b",
                expert_ids=(
                    "branch_b.coverage",
                    "branch_b.interaction",
                    "branch_b.local",
                ),
                pilot_slots=pilot_slots_per_branch,
            ),
        ),
        rank_layer_count=3,
        beam_width=512,
    )
    return request, tuple(proposals), policy


def _market(proposals):
    return {
        action.action_sha256: action
        for proposal in proposals
        for action in proposal.actions
    }


def test_factorial_pilot_covers_model_role_and_rank_layers() -> None:
    request, proposals, policy = _fixture()
    first = asyncio.run(policy.require(request, proposals))
    second = asyncio.run(policy.require(request, proposals))

    assert first.requirement_sha256 == second.requirement_sha256
    assert len(first.required_action_sha256s) == 6
    evidence = thaw_json(first.evidence)
    selection = evidence["factorial_selection"]
    coverage = selection["coverage"]
    assert coverage["branch_count"] == 2
    assert coverage["branch_role_cell_count"] == 6
    assert coverage["branch_rank_layer_cell_count"] == 6
    assert coverage["role_rank_layer_cell_count"] == 6
    assert coverage["rank_layer_count"] == 3
    assert evidence["market_capacity_certificate"]["residual_seats"] == 2
    assert evidence["candidate_outcomes_observed"] is False


def test_four_slot_factorial_pilot_spans_roles_and_rank_layers_per_branch() -> None:
    request, proposals, _ = _fixture(pilot_slots_per_branch=2)
    policy = FactorialBranchPilotPolicy(
        branch_bindings=(
            ProtectedBranchBinding(
                branch_id="branch_a",
                expert_ids=(
                    "branch_a.coverage",
                    "branch_a.interaction",
                    "branch_a.local",
                ),
                pilot_slots=2,
            ),
            ProtectedBranchBinding(
                branch_id="branch_b",
                expert_ids=(
                    "branch_b.coverage",
                    "branch_b.interaction",
                    "branch_b.local",
                ),
                pilot_slots=2,
            ),
        ),
        rank_layer_count=3,
        beam_width=256,
    )
    requirement = asyncio.run(policy.require(request, proposals))
    coverage = thaw_json(requirement.evidence)["factorial_selection"][
        "coverage"
    ]

    assert len(requirement.required_action_sha256s) == 4
    assert coverage["branch_role_cell_count"] == 4
    assert coverage["branch_rank_layer_cell_count"] == 4
    assert coverage["role_rank_layer_cell_count"] == 4


def test_cross_branch_collision_is_backfilled_without_losing_coverage() -> None:
    request, proposals, policy = _fixture(duplicate_across_branches=True)
    requirement = asyncio.run(policy.require(request, proposals))
    market = _market(proposals)
    selected = [
        market[action_sha] for action_sha in requirement.required_action_sha256s
    ]

    assert len(selected) == 6
    assert len(
        {value.phenotype_identity_sha256 for value in selected}
    ) == 6
    coverage = thaw_json(requirement.evidence)["factorial_selection"][
        "coverage"
    ]
    assert coverage["branch_role_cell_count"] == 6
    assert coverage["branch_rank_layer_cell_count"] == 6


def test_factorial_pilot_fails_closed_when_floor_exceeds_k() -> None:
    request, proposals, policy = _fixture(
        evaluation_slots=5,
        pilot_slots_per_branch=3,
    )

    with pytest.raises(ValueError, match="floors exceed"):
        asyncio.run(policy.require(request, proposals))


def test_factorial_selector_extends_canonical_anchors_without_replacing_them():
    request, proposals, policy = _fixture()
    actions = tuple(
        action for proposal in proposals for action in proposal.actions
    )
    candidates = factorial_candidates_from_materialized_actions(
        actions,
        branch_bindings=policy.branch_bindings,
        rank_layer_count=3,
    )
    by_branch = {
        branch_id: tuple(
            value
            for value in candidates
            if value.branch_id == branch_id
        )
        for branch_id in ("branch_a", "branch_b")
    }
    anchors = tuple(
        sorted(
            (
                by_branch["branch_a"][0].action_sha256,
                by_branch["branch_a"][1].action_sha256,
                by_branch["branch_b"][0].action_sha256,
                by_branch["branch_b"][1].action_sha256,
            )
        )
    )
    selection = select_factorial_pilot_candidates(
        candidates,
        branch_quotas=(("branch_a", 3), ("branch_b", 3)),
        rank_layer_count=3,
        beam_width=256,
        decision_scope_sha256=request.request_sha256,
        decision_index=request.decision_index,
        anchor_action_sha256s=anchors,
    )

    assert set(anchors) <= {
        value.action_sha256 for value in selection.selected
    }
    assert selection.anchor_action_sha256s == anchors
    assert len(selection.selected) == 6
    assert selection.to_record()["coverage"]["branch_role_cell_count"] >= 4

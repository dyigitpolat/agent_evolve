from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.action_score_authorities import (
    NativeRankMaterializedActionScorer,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.protected_action_committee import (
    ActionCommitteeArmBinding,
    ProtectedActionCommitteePolicy,
)
from agent_evolve.application.precommitted_portfolio_racing import (
    PortfolioRacePolicyBinding,
    PrecommittedPortfolioRacePlanner,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.single_score_action_allocation import (
    SingleScoreMaterializedActionPolicy,
)
from agent_evolve.application.source_exposure_allocation import (
    MinimumExpertSourceExposureSlateFeasibility,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json, thaw_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _FixedBallot:
    policy_id: str
    member_ranks: tuple[int, ...]
    foreign_requirement_identity: bool = False
    policy_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"fixed-ballot:{self.policy_id}:{self.member_ranks}"),
        )

    async def require(self, request, proposals):
        actions = {
            action.native_rank: action
            for proposal in proposals
            for action in proposal.actions
        }
        return MaterializedActionAllocationRequirement(
            policy_id=(
                "foreign_ballot"
                if self.foreign_requirement_identity
                else self.policy_id
            ),
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            required_action_sha256s=tuple(
                sorted(actions[rank].action_sha256 for rank in self.member_ranks)
            ),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "candidate_outcomes_observed": False,
                    "fixed_test_ballot": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _RequiredNativeRankFeasibility:
    required_rank: int
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"required-native-rank:{self.required_rank}"),
        )

    def permits(self, actions):
        return (
            type(actions) is tuple
            and len(
                {
                    value.phenotype_identity_sha256
                    for value in actions
                }
            )
            == len(actions)
            and any(
                value.native_rank == self.required_rank
                for value in actions
            )
        )


def _fixture():
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("committee-campaign"),
        prior_state_sha256=_sha("committee-prior"),
        decision_index=2,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=8,
        evaluation_slots=4,
        expert_proposal_slots=(("committee_expert", 7),),
        proposal_context=freeze_json({"test": "committee"}),
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
        structural_signature_sha256=_sha("committee-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.trace",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    actions = tuple(
        MaterializedActionDescriptor(
            context=context,
            configuration=freeze_json({"rank": rank}),
            phenotype_identity_sha256=(
                _sha("committee-phenotype:3")
                if rank == 7
                else _sha(f"committee-phenotype:{rank}")
            ),
            expert_id="committee_expert",
            native_rank=rank,
            parent_ids=(CandidateId("candidate_committee_parent"),),
            operator_id="committee_mutation",
            target_candidate_id=CandidateId(f"candidate_committee_{rank}"),
            role_id="committee_test",
            normalized_evaluation_cost=1.0,
        )
        for rank in range(1, 8)
    )
    proposal = MaterializedActionProposalBatch(
        request_sha256=request.request_sha256,
        expert_id="committee_expert",
        expert_version=1,
        expert_definition_sha256=_sha("committee-expert"),
        actions=actions,
        evidence=freeze_json(
            {
                "candidate_outcomes_observed": False,
                "sealed_test_market": True,
            }
        ),
    )
    ballots = (
        ActionCommitteeArmBinding(
            "favorable",
            _FixedBallot("favorable_ballot", (1, 2, 5, 6)),
            0.3,
        ),
        ActionCommitteeArmBinding(
            "neutral",
            _FixedBallot("neutral_ballot", (1, 2, 3, 4)),
            0.5,
        ),
        ActionCommitteeArmBinding(
            "risk",
            _FixedBallot("risk_ballot", (1, 3, 5, 7)),
            0.2,
        ),
    )
    policy = ProtectedActionCommitteePolicy(
        arm_bindings=ballots,
        protected_arm_id="neutral",
        protected_slots=1,
        audit_slots=1,
        audit_seed_sha256=_sha("committee-audit-seed"),
    )
    return request, (proposal,), actions, policy


def test_protected_committee_is_exact_k_replayable_and_propensity_logged():
    request, proposals, actions, policy = _fixture()
    first = asyncio.run(policy.require(request, proposals))
    second = asyncio.run(policy.require(request, proposals))

    assert first.requirement_sha256 == second.requirement_sha256
    assert len(first.required_action_sha256s) == request.evaluation_slots
    selected = {
        value.action_sha256: value
        for value in actions
        if value.action_sha256 in first.required_action_sha256s
    }
    assert len(
        {value.phenotype_identity_sha256 for value in selected.values()}
    ) == request.evaluation_slots
    assert actions[0].action_sha256 in selected

    evidence = thaw_json(first.evidence)
    selected_rows = [
        value for value in evidence["ballot_union"] if value["selected"]
    ]
    trace = evidence["selection_trace"]
    assert [value["ordinal"] for value in trace] == [1, 2, 3, 4]
    assert {value["action_sha256"] for value in trace} == set(
        first.required_action_sha256s
    )
    assert all(
        value["candidate_outcomes_observed"] is False for value in trace
    )
    kinds = Counter(value["selection_kind"] for value in selected_rows)
    assert Counter(value["allocation_kind"] for value in trace) == kinds
    assert kinds["protected_floor"] == 1
    assert kinds["randomized_disagreement_audit"] == 1
    committee = evidence["committee"]
    assert committee["effective_audit_slots"] == 1
    assert float.fromhex(
        committee["audit_marginal_inclusion_probability_hex"]
    ) == 1.0 / committee["audit_pool_size"]
    assert float.fromhex(committee["exact_action_set_propensity"]) == (
        1.0 / committee["audit_pool_size"]
    )
    assert len(committee["replaced_baseline_action_sha256s"]) == 1
    row_by_action = {
        value["action_sha256"]: value
        for value in evidence["ballot_union"]
    }
    replaced = row_by_action[
        committee["replaced_baseline_action_sha256s"][0]
    ]
    assert float.fromhex(
        replaced["final_marginal_inclusion_probability_hex"]
    ) == 0.0
    audit_rows = [
        value
        for value in evidence["ballot_union"]
        if value["audit_eligible"]
    ]
    assert all(
        float.fromhex(
            value["final_marginal_inclusion_probability_hex"]
        )
        == 1.0 / committee["audit_pool_size"]
        for value in audit_rows
    )
    assert evidence["candidate_outcomes_observed"] is False
    assert evidence["workload_model_provider_branches"] is False


def test_committee_allocations_compose_with_generic_sequential_planner():
    request, proposals, _, protected = _fixture()
    consensus = ProtectedActionCommitteePolicy(
        arm_bindings=protected.arm_bindings,
        protected_arm_id=protected.protected_arm_id,
        protected_slots=1,
        audit_slots=1,
        audit_seed_sha256=_sha("committee-consensus-audit-seed"),
    )
    conservative = ProtectedActionCommitteePolicy(
        arm_bindings=protected.arm_bindings,
        protected_arm_id=protected.protected_arm_id,
        protected_slots=3,
        audit_slots=0,
        audit_seed_sha256=_sha("committee-conservative-audit-seed"),
    )
    plan = asyncio.run(
        PrecommittedPortfolioRacePlanner(
            branch_bindings=(
                PortfolioRacePolicyBinding(
                    branch_id="consensus",
                    policy=consensus,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="conservative",
                    policy=conservative,
                ),
            ),
            pilot_policy=None,
            pilot_slots=2,
        ).plan(request, proposals)
    )

    assert plan.frozen_branch_ids == ("consensus", "conservative")
    assert len(plan.pilot_action_sha256s) == 2
    assert all(
        set(plan.pilot_action_sha256s).issubset(
            branch.requirement.required_action_sha256s
        )
        for branch in plan.branches
    )
    assert {
        value.lane_id.split(".", maxsplit=1)[0]
        for value in plan.pilot_lane_bindings
    }.issubset(
        {
            "allocation_kind:protected_floor",
            "allocation_kind:weighted_consensus",
            "allocation_kind:randomized_disagreement_audit",
        }
    )
    plan.__post_init__()


def test_protected_committee_collapses_duplicate_phenotypes_before_audit():
    request, proposals, actions, policy = _fixture()
    requirement = asyncio.run(policy.require(request, proposals))
    evidence = thaw_json(requirement.evidence)
    phenotype_rows = [
        value["phenotype_identity_sha256"]
        for value in evidence["ballot_union"]
    ]

    assert len(phenotype_rows) == len(set(phenotype_rows))
    duplicate_action_ids = {
        actions[2].action_sha256,
        actions[6].action_sha256,
    }
    retained_duplicates = {
        value["action_sha256"]
        for value in evidence["ballot_union"]
        if value["action_sha256"] in duplicate_action_ids
    }
    assert len(retained_duplicates) == 1


def test_duplicate_phenotype_retains_protected_arm_membership():
    request, proposals, actions, _ = _fixture()
    policy = ProtectedActionCommitteePolicy(
        arm_bindings=(
            ActionCommitteeArmBinding(
                "favorable",
                _FixedBallot(
                    "favorable_duplicate_ballot",
                    (1, 2, 3, 5),
                ),
                0.45,
            ),
            ActionCommitteeArmBinding(
                "neutral",
                _FixedBallot(
                    "neutral_duplicate_ballot",
                    (1, 4, 6, 7),
                ),
                0.1,
            ),
            ActionCommitteeArmBinding(
                "risk",
                _FixedBallot(
                    "risk_duplicate_ballot",
                    (2, 3, 4, 5),
                ),
                0.45,
            ),
        ),
        protected_arm_id="neutral",
        protected_slots=1,
        audit_slots=0,
        audit_seed_sha256=_sha("committee-duplicate-audit-seed"),
    )

    requirement = asyncio.run(policy.require(request, proposals))
    evidence = thaw_json(requirement.evidence)
    duplicate_phenotype = actions[2].phenotype_identity_sha256
    duplicate_row = next(
        value
        for value in evidence["ballot_union"]
        if value["phenotype_identity_sha256"] == duplicate_phenotype
    )

    assert duplicate_row["action_sha256"] == actions[2].action_sha256
    assert duplicate_row["arm_ids"] == ["favorable", "neutral", "risk"]
    assert duplicate_row["support_count"] == 3
    assert duplicate_row["protected_member"] is True


def test_protected_committee_downweights_behaviorally_redundant_ballots():
    request, proposals, _, policy = _fixture()
    requirement = asyncio.run(policy.require(request, proposals))
    evidence = thaw_json(requirement.evidence)
    arms = {
        value["arm_id"]: value
        for value in evidence["arms"]
    }

    for arm_id, arm in arms.items():
        similarities = {
            other: float.fromhex(value)
            for other, value in arm[
                "pairwise_jaccard_similarity_hex"
            ].items()
        }
        assert similarities[arm_id] == 1.0
        assert float.fromhex(arm["behavioral_redundancy_hex"]) > 1.0
        assert float.fromhex(arm["effective_weight_hex"]) < float.fromhex(
            arm["weight_hex"]
        )
        for other_arm_id, similarity in similarities.items():
            reverse = float.fromhex(
                arms[other_arm_id][
                    "pairwise_jaccard_similarity_hex"
                ][arm_id]
            )
            assert similarity == reverse


def test_protected_committee_rejects_changed_arm_policy_identity():
    request, proposals, _, policy = _fixture()
    bindings = list(policy.arm_bindings)
    bindings[0] = ActionCommitteeArmBinding(
        "favorable",
        _FixedBallot(
            "favorable_ballot",
            (1, 2, 5, 6),
            foreign_requirement_identity=True,
        ),
        0.3,
    )
    malformed = ProtectedActionCommitteePolicy(
        arm_bindings=tuple(bindings),
        protected_arm_id=policy.protected_arm_id,
        protected_slots=policy.protected_slots,
        audit_slots=policy.audit_slots,
        audit_seed_sha256=policy.audit_seed_sha256,
    )

    try:
        asyncio.run(malformed.require(request, proposals))
    except ValueError as error:
        assert "changed its policy identity" in str(error)
    else:
        raise AssertionError("foreign arm requirement identity was accepted")


def test_single_score_support_arm_derives_exact_k_from_request():
    request, proposals, actions, _ = _fixture()
    policy = SingleScoreMaterializedActionPolicy(
        scorer=NativeRankMaterializedActionScorer()
    )

    requirement = asyncio.run(policy.require(request, proposals))
    selected = {
        value.action_sha256: value
        for value in actions
        if value.action_sha256 in requirement.required_action_sha256s
    }
    evidence = thaw_json(requirement.evidence)

    assert len(selected) == request.evaluation_slots
    assert len(
        {
            value.phenotype_identity_sha256
            for value in selected.values()
        }
    ) == request.evaluation_slots
    assert len(evidence["selection_trace"]) == request.evaluation_slots
    assert evidence["candidate_outcomes_observed"] is False
    assert evidence["workload_model_provider_branches"] is False


def test_committee_repairs_and_filters_for_final_slate_feasibility():
    request, proposals, actions, source = _fixture()
    feasibility = _RequiredNativeRankFeasibility(required_rank=6)
    policy = ProtectedActionCommitteePolicy(
        arm_bindings=source.arm_bindings,
        protected_arm_id=source.protected_arm_id,
        protected_slots=source.protected_slots,
        audit_slots=source.audit_slots,
        audit_seed_sha256=source.audit_seed_sha256,
        slate_feasibility=feasibility,
    )

    requirement = asyncio.run(policy.require(request, proposals))
    selected = tuple(
        value
        for value in actions
        if value.action_sha256 in requirement.required_action_sha256s
    )
    evidence = thaw_json(requirement.evidence)
    committee = evidence["committee"]

    assert feasibility.permits(selected)
    assert any(value.native_rank == 6 for value in selected)
    assert committee["feasibility_replacements"]
    assert committee["effective_audit_slots"] == 0
    assert committee["valid_audit_subset_count"] == 1
    assert float.fromhex(committee["exact_action_set_propensity"]) == 1.0


def test_expert_source_slate_feasibility_is_workload_and_model_opaque():
    _, _, actions, _ = _fixture()
    feasibility = MinimumExpertSourceExposureSlateFeasibility(
        expert_group_bindings=(
            ("lane_a.expert", "source_a"),
            ("lane_b.expert", "source_b"),
        ),
        minimum_exposures=(("source_a", 2), ("source_b", 2)),
    )
    balanced = tuple(
        replace(
            value,
            expert_id=(
                "lane_a.expert" if index < 3 else "lane_b.expert"
            ),
        )
        for index, value in enumerate(actions[:6])
    )
    collapsed = tuple(
        replace(
            value,
            expert_id=(
                "lane_a.expert" if index < 5 else "lane_b.expert"
            ),
        )
        for index, value in enumerate(actions[:6])
    )

    assert feasibility.permits(balanced)
    assert not feasibility.permits(collapsed)

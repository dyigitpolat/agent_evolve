from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.source_exposure_allocation import (
    ExplicitExpertSourceGroupProjection,
    MinimumSourceExposureAllocationPolicy,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json, thaw_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _AlphaOnlyBasePolicy:
    policy_id: str = "test_alpha_only_base"
    policy_version: int = 1
    definition_sha256: str = _sha("test-alpha-only-base")

    async def require(self, request, proposals):
        selected = tuple(
            action
            for proposal in proposals
            if proposal.expert_id == "expert_alpha"
            for action in proposal.actions
        )
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            required_action_sha256s=tuple(
                sorted(value.action_sha256 for value in selected)
            ),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "selection_trace": [
                        {
                            "ordinal": ordinal,
                            "allocation_kind": "base_score_lane",
                            "score_lane": "base",
                            "action_sha256": action.action_sha256,
                            "candidate_outcomes_observed": False,
                        }
                        for ordinal, action in enumerate(selected, start=1)
                    ],
                    "candidate_outcomes_observed": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _PriorityScorer:
    scorer_id: str = "test_source_priority"
    scorer_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha("test-source-priority"),
        )

    async def score(self, request, proposals):
        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )
        scores = tuple(
            MaterializedActionScore(
                action_sha256=action.action_sha256,
                value=float(
                    (10 if action.expert_id == "expert_beta" else 0)
                    - action.native_rank
                ),
            )
            for action in sorted(
                actions,
                key=lambda value: value.action_sha256,
            )
        )
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            scores=scores,
            candidate_outcomes_observed=False,
            evidence_sha256=_sha("test-source-priority-evidence"),
        )


def _fixture():
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("source-exposure-campaign"),
        prior_state_sha256=_sha("source-exposure-prior"),
        decision_index=1,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=8,
        evaluation_slots=4,
        expert_proposal_slots=(
            ("expert_alpha", 4),
            ("expert_beta", 4),
        ),
        proposal_context=freeze_json({"test": "source-exposure"}),
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
        structural_signature_sha256=_sha("source-exposure-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.trace",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    proposals = []
    for expert_id in ("expert_alpha", "expert_beta"):
        actions = tuple(
            MaterializedActionDescriptor(
                context=context,
                configuration=freeze_json(
                    {"expert_id": expert_id, "rank": rank}
                ),
                phenotype_identity_sha256=_sha(
                    f"source-exposure-phenotype:{expert_id}:{rank}"
                ),
                expert_id=expert_id,
                native_rank=rank,
                parent_ids=(),
                operator_id="test_mutation",
                target_candidate_id=CandidateId(
                    f"candidate_{expert_id}_{rank}"
                ),
                role_id="local_exploit",
                normalized_evaluation_cost=1.0,
                reference_action=False,
            )
            for rank in range(1, 5)
        )
        proposals.append(
            MaterializedActionProposalBatch(
                request_sha256=request.request_sha256,
                expert_id=expert_id,
                expert_version=1,
                expert_definition_sha256=_sha(
                    f"source-exposure-expert:{expert_id}"
                ),
                actions=actions,
                evidence=freeze_json(
                    {
                        "candidate_outcomes_observed": False,
                        "sealed_once": True,
                    }
                ),
            )
        )
    return request, tuple(proposals)


async def _exercise_source_exposure_repair() -> None:
    request, proposals = _fixture()
    requirement = await MinimumSourceExposureAllocationPolicy(
        base_policy=_AlphaOnlyBasePolicy(),
        priority_scorer=_PriorityScorer(),
        source_projection=ExplicitExpertSourceGroupProjection(
            expert_group_bindings=(
                ("expert_alpha", "engine_a"),
                ("expert_beta", "engine_b"),
            )
        ),
        minimum_exposures=(
            ("engine_a", 2),
            ("engine_b", 2),
        ),
    ).require(request, proposals)

    action_by_sha256 = {
        action.action_sha256: action
        for proposal in proposals
        for action in proposal.actions
    }
    selected = tuple(
        action_by_sha256[value]
        for value in requirement.required_action_sha256s
    )
    assert len(selected) == request.evaluation_slots
    assert sum(
        value.expert_id == "expert_alpha" for value in selected
    ) == 2
    assert sum(
        value.expert_id == "expert_beta" for value in selected
    ) == 2
    assert len(
        {value.phenotype_identity_sha256 for value in selected}
    ) == request.evaluation_slots
    evidence = thaw_json(requirement.evidence)
    assert evidence["base_group_counts"] == {
        "engine_a": 4,
        "engine_b": 0,
    }
    assert evidence["final_group_counts"] == {
        "engine_a": 2,
        "engine_b": 2,
    }
    assert len(evidence["replacements"]) == 2
    assert sum(
        value["allocation_kind"] == "source_exposure_floor"
        for value in evidence["selection_trace"]
    ) == 2
    assert evidence["candidate_outcomes_observed"] is False
    requirement.__post_init__()


def test_source_exposure_floor_repairs_collapsed_base_slate() -> None:
    asyncio.run(_exercise_source_exposure_repair())

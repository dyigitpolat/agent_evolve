"""Dynamic exact-K allocation from one sealed outcome-blind score authority.

This is the small inverted-API adapter used when an independent score view
participates as one action-committee ballot. It derives K from the request,
validates complete score coverage, preserves unique phenotypes, and can reject
only authenticated hard infeasibility. It contains no workload, model,
provider, prompt, or simulator branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import ClassVar

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScoreBatch,
    MaterializedActionScorePort,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityPort,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
    assess_hard_feasibility,
    hard_feasibility_decision_batch_sha256,
    validate_hard_feasibility_port,
)


SINGLE_SCORE_ACTION_ALLOCATION_POLICY_ID = (
    "single_score_materialized_action_allocator"
)
SINGLE_SCORE_ACTION_ALLOCATION_POLICY_VERSION = 1
_DEFINITION_DOMAIN = (
    b"agent-evolve:single-score-materialized-action-allocation:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _scorer_identity(
    scorer: MaterializedActionScorePort,
) -> tuple[str, int, str]:
    if not isinstance(scorer, MaterializedActionScorePort):
        raise TypeError("scorer must implement its score port")
    scorer_id = getattr(scorer, "scorer_id", None)
    scorer_version = getattr(scorer, "scorer_version", None)
    definition_sha256 = getattr(scorer, "definition_sha256", None)
    if type(scorer_id) is not str or not scorer_id:
        raise ValueError("scorer_id must be non-empty")
    if type(scorer_version) is not int or scorer_version <= 0:
        raise ValueError("scorer_version must be positive")
    require_sha256(definition_sha256, "scorer definition_sha256")
    return scorer_id, scorer_version, definition_sha256


@dataclass(frozen=True, slots=True)
class SingleScoreMaterializedActionPolicy:
    """Nominate the request's exact-K phenotype-unique score head."""

    scorer: MaterializedActionScorePort = field(
        repr=False,
        compare=False,
    )
    hard_feasibility: HardFeasibilityPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    policy_id: ClassVar[str] = SINGLE_SCORE_ACTION_ALLOCATION_POLICY_ID
    policy_version: ClassVar[int] = (
        SINGLE_SCORE_ACTION_ALLOCATION_POLICY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        scorer_id, scorer_version, scorer_definition = _scorer_identity(
            self.scorer
        )
        feasibility_identity = (
            None
            if self.hard_feasibility is None
            else validate_hard_feasibility_port(self.hard_feasibility)
        )
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                _DEFINITION_DOMAIN
                + _canonical_json(
                    {
                        "schema_version": 1,
                        "policy_id": self.policy_id,
                        "policy_version": self.policy_version,
                        "scorer": {
                            "scorer_id": scorer_id,
                            "scorer_version": scorer_version,
                            "definition_sha256": scorer_definition,
                        },
                        "capacity": "request_evaluation_slots",
                        "ranking": (
                            "descending_finite_score_then_action_sha256"
                        ),
                        "collision": (
                            "skip_exact_materialized_phenotype_and_continue"
                        ),
                        "hard_feasibility": (
                            None
                            if feasibility_identity is None
                            else {
                                "policy_id": feasibility_identity[0],
                                "policy_version": feasibility_identity[1],
                                "definition_sha256": feasibility_identity[2],
                            }
                        ),
                        "hard_feasibility_rejection": (
                            "authenticated_infeasible_only"
                        ),
                        "candidate_outcomes_observed": False,
                        "workload_model_provider_branches": False,
                    }
                )
            ).hexdigest(),
        )

    async def require(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionAllocationRequirement:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )
        action_by_sha256 = {
            value.action_sha256: value for value in actions
        }
        if len(action_by_sha256) != len(actions):
            raise ValueError("proposal union repeats an action identity")
        action_sha256s = tuple(sorted(action_by_sha256))

        batch = await self.scorer.score(request, proposals)
        if type(batch) is not MaterializedActionScoreBatch:
            raise TypeError("scorer returned a foreign batch")
        batch.__post_init__()
        scorer_identity = _scorer_identity(self.scorer)
        if (
            batch.scorer_id,
            batch.scorer_version,
            batch.scorer_definition_sha256,
        ) != scorer_identity:
            raise ValueError("score batch differs from scorer identity")
        if (
            batch.residual_request_sha256 != request.request_sha256
            or batch.proposal_sha256s != proposal_sha256s
            or tuple(value.action_sha256 for value in batch.scores)
            != action_sha256s
        ):
            raise ValueError("score batch differs from sealed action market")

        infeasible: set[str] = set()
        feasibility_decisions = []
        if self.hard_feasibility is not None:
            for action_sha256 in action_sha256s:
                decision = assess_hard_feasibility(
                    self.hard_feasibility,
                    HardFeasibilityRequest(
                        campaign_scope_sha256=(
                            request.campaign_scope_sha256
                        ),
                        cutoff_index=request.decision_index,
                        configuration=action_by_sha256[
                            action_sha256
                        ].configuration,
                    ),
                )
                feasibility_decisions.append(
                    (action_sha256, decision)
                )
                if decision.verdict is HardFeasibilityVerdict.INFEASIBLE:
                    infeasible.add(action_sha256)

        score_by_action = {
            value.action_sha256: value.value for value in batch.scores
        }
        selected = []
        phenotypes: set[str] = set()
        selection_trace = []
        for action in sorted(
            actions,
            key=lambda value: (
                -score_by_action[value.action_sha256],
                value.action_sha256,
            ),
        ):
            if (
                action.action_sha256 in infeasible
                or action.phenotype_identity_sha256 in phenotypes
            ):
                continue
            selected.append(action)
            phenotypes.add(action.phenotype_identity_sha256)
            selection_trace.append(
                {
                    "ordinal": len(selected),
                    "allocation_kind": "single_score_head",
                    "score_lane": batch.scorer_id,
                    "action_sha256": action.action_sha256,
                    "score_hex": score_by_action[
                        action.action_sha256
                    ].hex(),
                    "candidate_outcomes_observed": False,
                }
            )
            if len(selected) == request.evaluation_slots:
                break
        if len(selected) != request.evaluation_slots:
            raise ValueError(
                "single score cannot fill a feasible phenotype-unique K-slate"
            )

        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=tuple(
                sorted(value.action_sha256 for value in selected)
            ),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "score_batch_sha256": batch.batch_sha256,
                    "scorer": {
                        "scorer_id": scorer_identity[0],
                        "scorer_version": scorer_identity[1],
                        "definition_sha256": scorer_identity[2],
                    },
                    "selection_trace": selection_trace,
                    "hard_feasibility": {
                        "enabled": self.hard_feasibility is not None,
                        "decision_batch_sha256": (
                            None
                            if not feasibility_decisions
                            else hard_feasibility_decision_batch_sha256(
                                tuple(
                                    decision.decision_sha256
                                    for _action_sha256, decision
                                    in feasibility_decisions
                                )
                            )
                        ),
                        "rejected_action_sha256s": sorted(infeasible),
                        "unknown_actions_remain_eligible": True,
                    },
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "SINGLE_SCORE_ACTION_ALLOCATION_POLICY_ID",
    "SINGLE_SCORE_ACTION_ALLOCATION_POLICY_VERSION",
    "SingleScoreMaterializedActionPolicy",
]

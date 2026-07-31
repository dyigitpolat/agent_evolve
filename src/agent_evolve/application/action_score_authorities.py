"""Workload-neutral score authorities for sealed materialized actions.

These scorers preserve two hypotheses that should not be collapsed into a
transferred learned model:

* the proposal engine's native within-expert semantic ordering; and
* strictly prior target-campaign empirical archive return.

Both implement the same opaque prequential score port used by learned,
numerical, proxy, and future optimizer-engine authorities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

from agent_evolve.application.materialized_action_broker import (
    EMPIRICAL_RETURN_ESTIMATOR_DEFINITION_SHA256,
    MATERIALIZED_ACTION_BROKER_DEFINITION_SHA256,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256


NATIVE_RANK_ACTION_SCORER_VERSION = 1
TARGET_EMPIRICAL_RETURN_ACTION_SCORER_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_NATIVE_DEFINITION_DOMAIN = (
    b"agent-evolve:native-rank-action-score-authority:v1\x00"
)
_NATIVE_EVIDENCE_DOMAIN = (
    b"agent-evolve:native-rank-action-score-evidence:v1\x00"
)
_EMPIRICAL_DEFINITION_DOMAIN = (
    b"agent-evolve:target-empirical-return-score-authority:v1\x00"
)
_EMPIRICAL_EVIDENCE_DOMAIN = (
    b"agent-evolve:target-empirical-return-score-evidence:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _sealed_universe(
    request: ResidualPortfolioDecisionRequest,
    proposals: tuple[MaterializedActionProposalBatch, ...],
) -> tuple[
    tuple[str, ...],
    tuple[MaterializedActionDescriptor, ...],
]:
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
    if len(actions) != len({value.action_sha256 for value in actions}):
        raise ValueError("proposal union repeats an action identity")
    if any(
        value.context.decision_index != request.decision_index
        for value in actions
    ):
        raise ValueError("action context crosses the decision cutoff")
    return proposal_sha256s, actions


@dataclass(frozen=True, slots=True)
class NativeRankMaterializedActionScorer:
    """Retain the engine's native rank as an independent cold-start lane."""

    scorer_id: str = "provider_native_rank"
    scorer_version: int = NATIVE_RANK_ACTION_SCORER_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.scorer_id, name="scorer_id")
        if (
            type(self.scorer_version) is not int
            or self.scorer_version <= 0
        ):
            raise ValueError("scorer_version must be positive")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _NATIVE_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "scorer_id": self.scorer_id,
                    "scorer_version": self.scorer_version,
                    "score": "reciprocal_positive_exact_native_rank",
                    "cross_expert_calibration_claimed": False,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    async def score(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionScoreBatch:
        self.__post_init__()
        proposal_sha256s, raw_actions = _sealed_universe(
            request,
            proposals,
        )
        actions = tuple(raw_actions)
        scores = tuple(
            sorted(
                (
                    MaterializedActionScore(
                        action_sha256=action.action_sha256,
                        value=float(1.0 / action.native_rank),
                    )
                    for action in actions
                ),
                key=lambda value: value.action_sha256,
            )
        )
        evidence_sha256 = _hash(
            _NATIVE_EVIDENCE_DOMAIN,
            {
                "scorer_definition_sha256": self.definition_sha256,
                "residual_request_sha256": request.request_sha256,
                "proposal_sha256s": list(proposal_sha256s),
                "native_ranks": [
                    {
                        "action_sha256": action.action_sha256,
                        "expert_id": action.expert_id,
                        "native_rank": action.native_rank,
                    }
                    for action in sorted(
                        actions,
                        key=lambda value: value.action_sha256,
                    )
                ],
                "candidate_outcomes_observed": False,
            },
        )
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            scores=scores,
            candidate_outcomes_observed=False,
            evidence_sha256=evidence_sha256,
        )


@dataclass(frozen=True, slots=True)
class TargetEmpiricalReturnMaterializedActionScorer:
    """Rank by target-campaign empirical return with native-rank cold start."""

    broker: RegretBrokeredMaterializedActionPolicy = field(
        repr=False,
        compare=False,
    )
    scorer_id: str = "target_empirical_return"
    native_rank_tie_scale: float = 1e-6
    scorer_version: int = TARGET_EMPIRICAL_RETURN_ACTION_SCORER_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.scorer_id, name="scorer_id")
        if (
            type(self.scorer_version) is not int
            or self.scorer_version <= 0
        ):
            raise ValueError("scorer_version must be positive")
        if type(self.broker) is not RegretBrokeredMaterializedActionPolicy:
            raise TypeError("broker must be exact")
        self.broker.__post_init__()
        if self.broker.return_value is not None:
            raise ValueError(
                "target empirical authority cannot contain a transfer prior"
            )
        if type(self.broker.ledger) is not MaterializedActionEvidenceLedger:
            raise TypeError("broker ledger must be exact")
        if (
            type(self.native_rank_tie_scale) is not float
            or not math.isfinite(self.native_rank_tie_scale)
            or not 0.0 < self.native_rank_tie_scale < 1e-3
        ):
            raise ValueError(
                "native_rank_tie_scale must be a finite float in (0, 1e-3)"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _EMPIRICAL_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "scorer_id": self.scorer_id,
                    "scorer_version": self.scorer_version,
                    "broker_definition_sha256": (
                        MATERIALIZED_ACTION_BROKER_DEFINITION_SHA256
                    ),
                    "return_estimator_definition_sha256": (
                        EMPIRICAL_RETURN_ESTIMATOR_DEFINITION_SHA256
                    ),
                    "hierarchical_kappa_hex": (
                        self.broker.hierarchical_kappa.hex()
                    ),
                    "score": (
                        "empirical_return_mean_plus_remaining_unit_interval_"
                        "times_native_rank_tie"
                    ),
                    "native_rank_tie_scale_hex": (
                        self.native_rank_tie_scale.hex()
                    ),
                    "candidate_outcomes_observed": False,
                    "strictly_prior_target_outcomes_only": True,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def _validate_cutoff(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> None:
        for outcome in self.broker.ledger.outcomes:
            if (
                outcome.action.context.decision_index
                >= request.decision_index
            ):
                raise ValueError(
                    "empirical return ledger crosses the decision cutoff"
                )
        for credit in self.broker.ledger.delayed_credits:
            if credit.available_at_decision_index > request.decision_index:
                raise ValueError(
                    "delayed credit is unavailable at the decision cutoff"
                )
        for resolved in self.broker.ledger.resolved_returns:
            if resolved.available_at_decision_index > request.decision_index:
                raise ValueError(
                    "resolved return is unavailable at the decision cutoff"
                )

    async def score(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionScoreBatch:
        self.__post_init__()
        proposal_sha256s, raw_actions = _sealed_universe(
            request,
            proposals,
        )
        self._validate_cutoff(request)
        actions = tuple(raw_actions)
        rows: list[dict[str, object]] = []
        scores: list[MaterializedActionScore] = []
        for action in sorted(
            actions,
            key=lambda value: value.action_sha256,
        ):
            broker_score = self.broker.score(action)
            empirical_mean = broker_score.return_estimate.mean
            value = empirical_mean + (
                (1.0 - empirical_mean)
                * self.native_rank_tie_scale
                / action.native_rank
            )
            scores.append(
                MaterializedActionScore(
                    action_sha256=action.action_sha256,
                    value=float(value),
                )
            )
            rows.append(
                {
                    "action_sha256": action.action_sha256,
                    "native_rank": action.native_rank,
                    "empirical_mean_hex": empirical_mean.hex(),
                    "score_hex": value.hex(),
                    "broker_score": broker_score.to_record(),
                }
            )
        evidence_sha256 = _hash(
            _EMPIRICAL_EVIDENCE_DOMAIN,
            {
                "scorer_definition_sha256": self.definition_sha256,
                "residual_request_sha256": request.request_sha256,
                "proposal_sha256s": list(proposal_sha256s),
                "ledger_counts": {
                    "outcomes": len(self.broker.ledger.outcomes),
                    "delayed_credits": len(
                        self.broker.ledger.delayed_credits
                    ),
                    "resolved_returns": len(
                        self.broker.ledger.resolved_returns
                    ),
                },
                "rows": rows,
                "candidate_outcomes_observed": False,
            },
        )
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            scores=tuple(scores),
            candidate_outcomes_observed=False,
            evidence_sha256=evidence_sha256,
        )


__all__ = [
    "NATIVE_RANK_ACTION_SCORER_VERSION",
    "TARGET_EMPIRICAL_RETURN_ACTION_SCORER_VERSION",
    "NativeRankMaterializedActionScorer",
    "TargetEmpiricalReturnMaterializedActionScorer",
]

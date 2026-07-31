"""Outcome-blind composition of heterogeneous action consequence scores.

The optimizer must not silently treat one fallible consequence estimator as
ground truth.  This module gives the application core a workload-neutral seam
for composing multiple pre-evaluation score authorities:

* every scorer authenticates the exact residual request and sealed proposal
  universe that it inspected;
* scorers may be learned models, calibrated proposer forecasts, numerical
  acquisitions, or deterministic audit policies;
* the core allocates an explicit quota to each scorer and de-duplicates exact
  materialized phenotypes while walking that scorer's ranking; and
* any evaluator capacity not nominated by the portfolio remains available to
  the downstream empirical broker.

Scores are intentionally opaque.  Workload adapters own feature projection and
model fitting; this core owns only information-boundary validation, identity,
quota composition, and durable evidence.  No current candidate outcome is
available at this boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityDecision,
    HardFeasibilityPort,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
    assess_hard_feasibility,
    hard_feasibility_decision_batch_sha256,
    validate_hard_feasibility_port,
)


PREQUENTIAL_SCORE_PORTFOLIO_POLICY_ID = (
    "prequential_quota_score_portfolio_allocator"
)
PREQUENTIAL_SCORE_PORTFOLIO_POLICY_VERSION = 1
PREQUENTIAL_SCORE_PORTFOLIO_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:prequential-quota-score-portfolio:v1;"
    b"input=prior-cutoff-request+complete-sealed-proposal-union;"
    b"score-semantics=opaque-injected-authorities;"
    b"composition=canonical-round-robin-explicit-quota;"
    b"within-authority=descending-finite-score+action-sha256-tie-break;"
    b"collision=skip-exact-materialized-phenotype-and-continue;"
    b"unreserved-capacity=downstream-empirical-broker;"
    b"candidate-outcomes=false;async=true;"
    b"workload-model-provider-branches=false"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_SCORE_BATCH_DOMAIN = b"agent-evolve:materialized-action-score-batch:v1\x00"
_RELIABILITY_EVIDENCE_DOMAIN = (
    b"agent-evolve:materialized-action-score-reliability:v1\x00"
)
_ADAPTIVE_POLICY_DOMAIN = (
    b"agent-evolve:reliability-adaptive-score-portfolio:v1\x00"
)
RELIABILITY_ADAPTIVE_SCORE_PORTFOLIO_POLICY_ID = (
    "reliability_adaptive_score_portfolio_allocator"
)
RELIABILITY_ADAPTIVE_SCORE_PORTFOLIO_POLICY_VERSION = 2


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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _scorer_identity(
    scorer: "MaterializedActionScorePort",
) -> tuple[str, int, str]:
    if not isinstance(scorer, MaterializedActionScorePort):
        raise TypeError("scorer must implement MaterializedActionScorePort")
    identity = (
        getattr(scorer, "scorer_id", None),
        getattr(scorer, "scorer_version", None),
        getattr(scorer, "definition_sha256", None),
    )
    _require_token(identity[0], name="scorer_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("scorer_version must be positive")
    require_sha256(identity[2], "scorer definition_sha256")
    return identity  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class MaterializedActionScore:
    """One opaque, finite, pre-evaluation score for one materialized action."""

    action_sha256: str
    value: float

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError("score value must be a finite exact float")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "value_hex": self.value.hex(),
        }


@dataclass(frozen=True, slots=True)
class MaterializedActionScoreBatch:
    """Authenticated complete scoring of one sealed proposal universe."""

    scorer_id: str
    scorer_version: int
    scorer_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    scores: tuple[MaterializedActionScore, ...]
    candidate_outcomes_observed: bool
    evidence_sha256: str
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.scorer_id, name="scorer_id")
        if type(self.scorer_version) is not int or self.scorer_version <= 0:
            raise ValueError("scorer_version must be positive")
        require_sha256(
            self.scorer_definition_sha256,
            "scorer_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if (
            type(self.proposal_sha256s) is not tuple
            or not self.proposal_sha256s
            or self.proposal_sha256s
            != tuple(sorted(set(self.proposal_sha256s)))
        ):
            raise ValueError("proposal_sha256s must be non-empty and canonical")
        for value in self.proposal_sha256s:
            require_sha256(value, "proposal_sha256")
        if type(self.scores) is not tuple or not self.scores:
            raise ValueError("scores must be a non-empty exact tuple")
        for value in self.scores:
            if type(value) is not MaterializedActionScore:
                raise TypeError("scores must contain exact score values")
            value.__post_init__()
        score_actions = tuple(value.action_sha256 for value in self.scores)
        if score_actions != tuple(sorted(set(score_actions))):
            raise ValueError("scores must cover unique actions canonically")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be an exact bool")
        if self.candidate_outcomes_observed:
            raise ValueError("prequential scores cannot observe current outcomes")
        require_sha256(self.evidence_sha256, "evidence_sha256")
        object.__setattr__(
            self,
            "batch_sha256",
            _hash(_SCORE_BATCH_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scorer": {
                "scorer_id": self.scorer_id,
                "scorer_version": self.scorer_version,
                "definition_sha256": self.scorer_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "scores": [value.to_record() for value in self.scores],
            "candidate_outcomes_observed": self.candidate_outcomes_observed,
            "evidence_sha256": self.evidence_sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "batch_sha256": self.batch_sha256}


@runtime_checkable
class MaterializedActionScorePort(Protocol):
    """Score a complete sealed population using strictly prior evidence."""

    scorer_id: str
    scorer_version: int
    definition_sha256: str

    async def score(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionScoreBatch: ...


@dataclass(frozen=True, slots=True)
class MaterializedActionScoreReliabilityEvidence:
    """Pre-evaluation authority calibration for one scorer and request."""

    scorer_id: str
    scorer_version: int
    scorer_definition_sha256: str
    residual_request_sha256: str
    component_authorities: tuple[tuple[str, float], ...]
    overall_reliability: float
    candidate_outcomes_observed: bool
    source_evidence_sha256: str
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.scorer_id, name="scorer_id")
        if type(self.scorer_version) is not int or self.scorer_version <= 0:
            raise ValueError("scorer_version must be positive")
        require_sha256(
            self.scorer_definition_sha256,
            "scorer_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if (
            type(self.component_authorities) is not tuple
            or not self.component_authorities
            or tuple(value[0] for value in self.component_authorities)
            != tuple(
                sorted(
                    {value[0] for value in self.component_authorities}
                )
            )
        ):
            raise ValueError(
                "component_authorities must be non-empty and canonical"
            )
        for component_id, authority in self.component_authorities:
            _require_token(component_id, name="reliability component")
            if (
                type(authority) is not float
                or not math.isfinite(authority)
                or not 0.0 <= authority <= 1.0
            ):
                raise ValueError("component authority must lie in [0, 1]")
        if (
            type(self.overall_reliability) is not float
            or not math.isfinite(self.overall_reliability)
            or not 0.0 <= self.overall_reliability <= 1.0
        ):
            raise ValueError("overall_reliability must lie in [0, 1]")
        if self.overall_reliability != min(
            value for _, value in self.component_authorities
        ):
            raise ValueError(
                "overall reliability must be the minimum component authority"
            )
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError(
                "candidate_outcomes_observed must be an exact bool"
            )
        if self.candidate_outcomes_observed:
            raise ValueError(
                "score reliability cannot observe current outcomes"
            )
        require_sha256(
            self.source_evidence_sha256,
            "source_evidence_sha256",
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            _hash(_RELIABILITY_EVIDENCE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scorer": {
                "scorer_id": self.scorer_id,
                "scorer_version": self.scorer_version,
                "definition_sha256": self.scorer_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "component_authorities": {
                component_id: authority.hex()
                for component_id, authority in self.component_authorities
            },
            "overall_reliability_hex": self.overall_reliability.hex(),
            "candidate_outcomes_observed": (
                self.candidate_outcomes_observed
            ),
            "source_evidence_sha256": self.source_evidence_sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "evidence_sha256": self.evidence_sha256,
        }


@runtime_checkable
class MaterializedActionScoreReliabilityPort(Protocol):
    """Expose authenticated prior-only reliability after scoring a batch."""

    scorer_id: str
    scorer_version: int
    definition_sha256: str

    def reliability(
        self,
        residual_request_sha256: str,
        component_ids: tuple[str, ...],
    ) -> MaterializedActionScoreReliabilityEvidence: ...


@dataclass(frozen=True, slots=True)
class PrequentialQuotaScorePortfolioPolicy:
    """Nominate a phenotype-unique quota from each independent score lane."""

    scorers: tuple[MaterializedActionScorePort, ...]
    scorer_quotas: tuple[tuple[str, int], ...]

    policy_id: ClassVar[str] = PREQUENTIAL_SCORE_PORTFOLIO_POLICY_ID
    policy_version: ClassVar[int] = PREQUENTIAL_SCORE_PORTFOLIO_POLICY_VERSION
    definition_sha256: ClassVar[str] = (
        PREQUENTIAL_SCORE_PORTFOLIO_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if type(self.scorers) is not tuple or not self.scorers:
            raise ValueError("scorers must be a non-empty exact tuple")
        identities = tuple(_scorer_identity(value) for value in self.scorers)
        scorer_ids = tuple(value[0] for value in identities)
        if scorer_ids != tuple(sorted(set(scorer_ids))):
            raise ValueError("scorers must use canonical unique IDs")
        if (
            type(self.scorer_quotas) is not tuple
            or not self.scorer_quotas
        ):
            raise ValueError("scorer_quotas must be a non-empty exact tuple")
        quota_ids: list[str] = []
        for scorer_id, quota in self.scorer_quotas:
            _require_token(scorer_id, name="quota scorer_id")
            if type(quota) is not int or quota <= 0:
                raise ValueError("every scorer quota must be positive")
            quota_ids.append(scorer_id)
        if tuple(quota_ids) != tuple(sorted(set(quota_ids))):
            raise ValueError("scorer_quotas must use canonical unique IDs")
        if tuple(quota_ids) != scorer_ids:
            raise ValueError("scorer_quotas must exactly cover scorers")
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(self.definition_sha256, "definition_sha256")

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

        quota_total = sum(value for _, value in self.scorer_quotas)
        if quota_total > request.evaluation_slots:
            raise ValueError("score portfolio quotas exceed evaluation capacity")

        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )
        action_by_sha256 = {value.action_sha256: value for value in actions}
        if len(action_by_sha256) != len(actions):
            raise ValueError("proposal union repeats an action identity")
        action_sha256s = tuple(sorted(action_by_sha256))

        batches = tuple(
            await asyncio.gather(
                *(scorer.score(request, proposals) for scorer in self.scorers)
            )
        )
        batch_by_scorer: dict[str, MaterializedActionScoreBatch] = {}
        scorer_by_id = {
            _scorer_identity(value)[0]: value for value in self.scorers
        }
        for batch in batches:
            if type(batch) is not MaterializedActionScoreBatch:
                raise TypeError("scorer returned a foreign score batch")
            batch.__post_init__()
            scorer = scorer_by_id.get(batch.scorer_id)
            if scorer is None:
                raise ValueError("score batch names a foreign scorer")
            if (
                batch.scorer_id,
                batch.scorer_version,
                batch.scorer_definition_sha256,
            ) != _scorer_identity(scorer):
                raise ValueError("score batch differs from its scorer identity")
            if batch.residual_request_sha256 != request.request_sha256:
                raise ValueError("score batch targets another residual request")
            if batch.proposal_sha256s != proposal_sha256s:
                raise ValueError("score batch targets another proposal universe")
            if tuple(value.action_sha256 for value in batch.scores) != (
                action_sha256s
            ):
                raise ValueError("score batch must exactly cover sealed actions")
            batch_by_scorer[batch.scorer_id] = batch

        ranked_by_scorer = {
            scorer_id: tuple(
                sorted(
                    batch_by_scorer[scorer_id].scores,
                    key=lambda value: (-value.value, value.action_sha256),
                )
            )
            for scorer_id, _ in self.scorer_quotas
        }
        selected: list[str] = []
        selected_phenotypes: set[str] = set()
        cursors = {scorer_id: 0 for scorer_id, _ in self.scorer_quotas}
        selected_by_scorer = {
            scorer_id: [] for scorer_id, _ in self.scorer_quotas
        }
        maximum_quota = max(value for _, value in self.scorer_quotas)
        for ordinal in range(maximum_quota):
            for scorer_id, quota in self.scorer_quotas:
                if ordinal >= quota:
                    continue
                ranking = ranked_by_scorer[scorer_id]
                while cursors[scorer_id] < len(ranking):
                    score = ranking[cursors[scorer_id]]
                    cursors[scorer_id] += 1
                    action = action_by_sha256[score.action_sha256]
                    if (
                        action.phenotype_identity_sha256
                        in selected_phenotypes
                    ):
                        continue
                    selected.append(action.action_sha256)
                    selected_phenotypes.add(
                        action.phenotype_identity_sha256
                    )
                    selected_by_scorer[scorer_id].append(
                        {
                            "action_sha256": action.action_sha256,
                            "score_hex": score.value.hex(),
                            "rank_position": cursors[scorer_id],
                        }
                    )
                    break
                else:
                    raise ValueError(
                        "score portfolio cannot satisfy phenotype-unique quotas"
                    )

        if len(selected) != quota_total:
            raise AssertionError("score portfolio did not close its quota")
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=tuple(sorted(selected)),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "score_batch_sha256s": {
                        scorer_id: batch_by_scorer[scorer_id].batch_sha256
                        for scorer_id, _ in self.scorer_quotas
                    },
                    "scorer_quotas": {
                        scorer_id: quota
                        for scorer_id, quota in self.scorer_quotas
                    },
                    "selected_by_scorer": selected_by_scorer,
                    "nomination_slots": quota_total,
                    "evaluation_slots": request.evaluation_slots,
                    "downstream_unreserved_slots": (
                        request.evaluation_slots - quota_total
                    ),
                    "candidate_outcomes_observed": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ReliabilityAdaptiveScorePortfolioPolicy:
    """Scale primary authority and refill exact-infeasibility rejections.

    The optional hard-feasibility port is deliberately proof-only.  An
    ``UNKNOWN`` verdict remains eligible; only an authenticated
    ``INFEASIBLE`` decision removes an action.  Each score authority then
    continues down its own frozen ranking, preserving quota provenance while
    avoiding a workload-specific branch in this policy.
    """

    scorers: tuple[MaterializedActionScorePort, ...]
    primary_scorer_id: str
    primary_reliability: MaterializedActionScoreReliabilityPort = field(
        repr=False,
        compare=False,
    )
    reliability_component_ids: tuple[str, ...]
    minimum_primary_fraction: float = 0.5
    hard_feasibility: HardFeasibilityPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    policy_id: ClassVar[str] = (
        RELIABILITY_ADAPTIVE_SCORE_PORTFOLIO_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        RELIABILITY_ADAPTIVE_SCORE_PORTFOLIO_POLICY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.scorers) is not tuple or len(self.scorers) < 2:
            raise ValueError(
                "adaptive portfolio requires a primary and fallback scorer"
            )
        identities = tuple(_scorer_identity(value) for value in self.scorers)
        scorer_ids = tuple(value[0] for value in identities)
        if scorer_ids != tuple(sorted(set(scorer_ids))):
            raise ValueError("scorers must use canonical unique IDs")
        _require_token(self.primary_scorer_id, name="primary_scorer_id")
        if self.primary_scorer_id not in scorer_ids:
            raise ValueError("primary scorer is absent")
        if not isinstance(
            self.primary_reliability,
            MaterializedActionScoreReliabilityPort,
        ):
            raise TypeError(
                "primary_reliability must implement its runtime port"
            )
        primary_identity = next(
            value
            for value in identities
            if value[0] == self.primary_scorer_id
        )
        if (
            self.primary_reliability.scorer_id,
            self.primary_reliability.scorer_version,
            self.primary_reliability.definition_sha256,
        ) != primary_identity:
            raise ValueError(
                "primary reliability identity differs from its scorer"
            )
        if (
            type(self.reliability_component_ids) is not tuple
            or not self.reliability_component_ids
            or self.reliability_component_ids
            != tuple(sorted(set(self.reliability_component_ids)))
        ):
            raise ValueError(
                "reliability components must be non-empty and canonical"
            )
        for value in self.reliability_component_ids:
            _require_token(value, name="reliability component")
        if (
            type(self.minimum_primary_fraction) is not float
            or not math.isfinite(self.minimum_primary_fraction)
            or not 0.0 < self.minimum_primary_fraction <= 1.0
        ):
            raise ValueError(
                "minimum_primary_fraction must lie in (0, 1]"
            )
        hard_feasibility_identity = None
        if self.hard_feasibility is not None:
            hard_feasibility_identity = validate_hard_feasibility_port(
                self.hard_feasibility
            )
        _require_token(self.policy_id, name="policy_id")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _ADAPTIVE_POLICY_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "scorers": [
                        {
                            "scorer_id": scorer_id,
                            "scorer_version": scorer_version,
                            "definition_sha256": definition_sha256,
                        }
                        for scorer_id, scorer_version, definition_sha256
                        in identities
                    ],
                    "primary_scorer_id": self.primary_scorer_id,
                    "reliability_component_ids": list(
                        self.reliability_component_ids
                    ),
                    "minimum_primary_fraction_hex": (
                        self.minimum_primary_fraction.hex()
                    ),
                    "primary_quota_rule": (
                        "ceil_fraction_floor_plus_nearest_integer_of_"
                        "reliability_times_remaining_capacity"
                    ),
                    "fallback_rule": (
                        "canonical_equal_integer_split_with_early_remainder"
                    ),
                    "hard_feasibility": (
                        None
                        if hard_feasibility_identity is None
                        else {
                            "policy_id": hard_feasibility_identity[0],
                            "policy_version": hard_feasibility_identity[1],
                            "definition_sha256": hard_feasibility_identity[2],
                        }
                    ),
                    "hard_feasibility_rejection_rule": (
                        "reject_only_authenticated_infeasible;"
                        "retain_feasible_and_unknown"
                    ),
                    "hard_feasibility_refill_rule": (
                        "continue_same_authority_frozen_ranking"
                    ),
                    "composition": (
                        "canonical_round_robin_phenotype_unique"
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
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
        scorer_by_id = {
            _scorer_identity(value)[0]: value for value in self.scorers
        }
        batches = tuple(
            await asyncio.gather(
                *(scorer.score(request, proposals) for scorer in self.scorers)
            )
        )
        batch_by_scorer: dict[str, MaterializedActionScoreBatch] = {}
        for batch in batches:
            if type(batch) is not MaterializedActionScoreBatch:
                raise TypeError("scorer returned a foreign score batch")
            batch.__post_init__()
            scorer = scorer_by_id.get(batch.scorer_id)
            if scorer is None:
                raise ValueError("score batch names a foreign scorer")
            if (
                batch.scorer_id,
                batch.scorer_version,
                batch.scorer_definition_sha256,
            ) != _scorer_identity(scorer):
                raise ValueError("score batch differs from scorer identity")
            if (
                batch.residual_request_sha256 != request.request_sha256
                or batch.proposal_sha256s != proposal_sha256s
                or tuple(value.action_sha256 for value in batch.scores)
                != action_sha256s
            ):
                raise ValueError("score batch differs from sealed universe")
            batch_by_scorer[batch.scorer_id] = batch

        reliability = self.primary_reliability.reliability(
            request.request_sha256,
            self.reliability_component_ids,
        )
        if type(reliability) is not MaterializedActionScoreReliabilityEvidence:
            raise TypeError("reliability port returned foreign evidence")
        reliability.__post_init__()
        primary_batch = batch_by_scorer[self.primary_scorer_id]
        if (
            reliability.scorer_id,
            reliability.scorer_version,
            reliability.scorer_definition_sha256,
        ) != (
            primary_batch.scorer_id,
            primary_batch.scorer_version,
            primary_batch.scorer_definition_sha256,
        ):
            raise ValueError("reliability evidence names another scorer")
        if reliability.residual_request_sha256 != request.request_sha256:
            raise ValueError("reliability evidence names another request")

        capacity = request.evaluation_slots
        minimum_primary = math.ceil(
            capacity * self.minimum_primary_fraction
        )
        adaptive_capacity = capacity - minimum_primary
        primary_quota = minimum_primary + math.floor(
            reliability.overall_reliability * adaptive_capacity + 0.5
        )
        primary_quota = min(
            capacity,
            max(minimum_primary, primary_quota),
        )
        fallback_ids = tuple(
            value
            for value in sorted(scorer_by_id)
            if value != self.primary_scorer_id
        )
        residual = capacity - primary_quota
        quotient, remainder = divmod(residual, len(fallback_ids))
        quotas = {
            self.primary_scorer_id: primary_quota,
            **{
                scorer_id: quotient + (1 if index < remainder else 0)
                for index, scorer_id in enumerate(fallback_ids)
            },
        }
        if sum(quotas.values()) != capacity:
            raise AssertionError("adaptive quotas do not close")

        ranked_by_scorer = {
            scorer_id: tuple(
                sorted(
                    batch_by_scorer[scorer_id].scores,
                    key=lambda value: (-value.value, value.action_sha256),
                )
            )
            for scorer_id in sorted(scorer_by_id)
        }
        feasibility_decisions: tuple[
            tuple[str, HardFeasibilityDecision], ...
        ] = ()
        infeasible_action_sha256s: set[str] = set()
        if self.hard_feasibility is not None:
            audited: list[tuple[str, HardFeasibilityDecision]] = []
            for action_sha256 in action_sha256s:
                action = action_by_sha256[action_sha256]
                decision = assess_hard_feasibility(
                    self.hard_feasibility,
                    HardFeasibilityRequest(
                        campaign_scope_sha256=request.campaign_scope_sha256,
                        cutoff_index=request.decision_index,
                        configuration=action.configuration,
                    ),
                )
                audited.append((action_sha256, decision))
                if decision.verdict is HardFeasibilityVerdict.INFEASIBLE:
                    infeasible_action_sha256s.add(action_sha256)
            feasibility_decisions = tuple(audited)
            if len(actions) - len(infeasible_action_sha256s) < capacity:
                raise ValueError(
                    "hard-feasibility-screened proposal union cannot fill "
                    "evaluation capacity"
                )
        selected: list[str] = []
        selected_phenotypes: set[str] = set()
        cursors = {scorer_id: 0 for scorer_id in scorer_by_id}
        selected_by_scorer = {
            scorer_id: [] for scorer_id in sorted(scorer_by_id)
        }
        for ordinal in range(max(quotas.values())):
            for scorer_id in sorted(scorer_by_id):
                if ordinal >= quotas[scorer_id]:
                    continue
                ranking = ranked_by_scorer[scorer_id]
                while cursors[scorer_id] < len(ranking):
                    score = ranking[cursors[scorer_id]]
                    cursors[scorer_id] += 1
                    action = action_by_sha256[score.action_sha256]
                    if action.action_sha256 in infeasible_action_sha256s:
                        continue
                    if (
                        action.phenotype_identity_sha256
                        in selected_phenotypes
                    ):
                        continue
                    selected.append(action.action_sha256)
                    selected_phenotypes.add(
                        action.phenotype_identity_sha256
                    )
                    selected_by_scorer[scorer_id].append(
                        {
                            "action_sha256": action.action_sha256,
                            "score_hex": score.value.hex(),
                            "rank_position": cursors[scorer_id],
                        }
                    )
                    break
                else:
                    raise ValueError(
                        "adaptive portfolio cannot close a unique quota"
                    )
        if len(selected) != capacity:
            raise AssertionError("adaptive portfolio did not fill capacity")
        feasibility_verdict_counts = {
            verdict.value: sum(
                decision.verdict is verdict
                for _action_sha256, decision in feasibility_decisions
            )
            for verdict in HardFeasibilityVerdict
        }
        feasibility_record = {
            "enabled": self.hard_feasibility is not None,
            "policy": (
                None
                if self.hard_feasibility is None
                else {
                    "policy_id": self.hard_feasibility.policy_id,
                    "policy_version": self.hard_feasibility.policy_version,
                    "definition_sha256": (
                        self.hard_feasibility.definition_sha256
                    ),
                }
            ),
            "verdict_counts": feasibility_verdict_counts,
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
            "rejected_actions": [
                {
                    "action_sha256": action_sha256,
                    "decision": decision.to_record(),
                }
                for action_sha256, decision in feasibility_decisions
                if decision.verdict
                is HardFeasibilityVerdict.INFEASIBLE
            ],
            "rejection_authority": (
                "authenticated_exact_infeasibility_only"
            ),
            "unknown_actions_remain_eligible": True,
            "refill": "same_authority_frozen_ranking",
            "candidate_outcomes_observed": False,
        }
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=tuple(sorted(selected)),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "score_batch_sha256s": {
                        scorer_id: batch_by_scorer[
                            scorer_id
                        ].batch_sha256
                        for scorer_id in sorted(batch_by_scorer)
                    },
                    "primary_reliability": reliability.to_record(),
                    "minimum_primary_quota": minimum_primary,
                    "effective_quotas": {
                        scorer_id: quotas[scorer_id]
                        for scorer_id in sorted(quotas)
                    },
                    "selected_by_scorer": selected_by_scorer,
                    "hard_feasibility": feasibility_record,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "MaterializedActionScore",
    "MaterializedActionScoreBatch",
    "MaterializedActionScorePort",
    "MaterializedActionScoreReliabilityEvidence",
    "MaterializedActionScoreReliabilityPort",
    "PREQUENTIAL_SCORE_PORTFOLIO_POLICY_DEFINITION_SHA256",
    "PREQUENTIAL_SCORE_PORTFOLIO_POLICY_ID",
    "PREQUENTIAL_SCORE_PORTFOLIO_POLICY_VERSION",
    "PrequentialQuotaScorePortfolioPolicy",
    "RELIABILITY_ADAPTIVE_SCORE_PORTFOLIO_POLICY_ID",
    "RELIABILITY_ADAPTIVE_SCORE_PORTFOLIO_POLICY_VERSION",
    "ReliabilityAdaptiveScorePortfolioPolicy",
]

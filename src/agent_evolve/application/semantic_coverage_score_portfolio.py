"""Outcome-blind, set-aware composition of sealed action score authorities.

The policy consumes two inverted APIs:

* opaque scalar score authorities rank every sealed materialized action; and
* a semantic-cell projection exposes only categorical predicted objective
  directions and whether the action descends from a previously generated
  candidate.

It never parses workload configurations, objective names, model identities, or
provider identities.  An exact recursive-lineage partition protects promising
descendants without allowing them to spill into ordinary score lanes, fixed
scorer shares conserve independent ranking authorities, and a rank-bounded
semantic coverage term prevents an evaluation slate from collapsing onto one
predicted trade-off direction.
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
    MaterializedActionDescriptor,
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
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityDecision,
    HardFeasibilityPort,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
    assess_hard_feasibility,
    hard_feasibility_decision_batch_sha256,
    validate_hard_feasibility_port,
)


SEMANTIC_COVERAGE_SCORE_PORTFOLIO_POLICY_ID = (
    "semantic_coverage_score_portfolio_allocator"
)
SEMANTIC_COVERAGE_SCORE_PORTFOLIO_POLICY_VERSION = 1

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_METRIC_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,255}$")
_DIRECTIONS = frozenset({"decrease", "increase", "unchanged"})
_CELL_BATCH_DOMAIN = b"agent-evolve:action-semantic-cell-batch:v1\x00"
_POLICY_DOMAIN = b"agent-evolve:semantic-coverage-score-portfolio:v1\x00"


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
    scorer: MaterializedActionScorePort,
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
class MaterializedActionSemanticCell:
    """One sealed action's portable categorical coverage coordinates."""

    action_sha256: str
    direction_signature: tuple[tuple[str, str], ...]
    recursive_lineage: bool

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        if type(self.direction_signature) is not tuple:
            raise TypeError("direction_signature must be an exact tuple")
        metric_ids: list[str] = []
        for metric_id, direction in self.direction_signature:
            if (
                type(metric_id) is not str
                or _METRIC_ID.fullmatch(metric_id) is None
            ):
                raise ValueError("semantic metric_id has invalid syntax")
            if direction not in _DIRECTIONS:
                raise ValueError("semantic direction is not categorical")
            metric_ids.append(metric_id)
        if tuple(metric_ids) != tuple(sorted(set(metric_ids))):
            raise ValueError("semantic directions must be metric-canonical")
        if type(self.recursive_lineage) is not bool:
            raise TypeError("recursive_lineage must be an exact bool")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "direction_signature": [
                {"metric_id": metric_id, "direction": direction}
                for metric_id, direction in self.direction_signature
            ],
            "recursive_lineage": self.recursive_lineage,
        }


@dataclass(frozen=True, slots=True)
class MaterializedActionSemanticCellBatch:
    """Authenticated semantic cells for a complete sealed proposal union."""

    projection_id: str
    projection_version: int
    projection_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    cells: tuple[MaterializedActionSemanticCell, ...]
    candidate_outcomes_observed: bool
    evidence: FrozenJsonObject
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.projection_id, name="projection_id")
        if type(self.projection_version) is not int or self.projection_version <= 0:
            raise ValueError("projection_version must be positive")
        require_sha256(
            self.projection_definition_sha256,
            "projection_definition_sha256",
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
        if type(self.cells) is not tuple or not self.cells:
            raise ValueError("semantic cells must be a non-empty exact tuple")
        for value in self.cells:
            if type(value) is not MaterializedActionSemanticCell:
                raise TypeError("cells must contain exact semantic cells")
            value.__post_init__()
        action_sha256s = tuple(value.action_sha256 for value in self.cells)
        if action_sha256s != tuple(sorted(set(action_sha256s))):
            raise ValueError("semantic cells must be action-canonical")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be exact")
        if self.candidate_outcomes_observed:
            raise ValueError("semantic cells cannot observe current outcomes")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("semantic evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "batch_sha256",
            _hash(_CELL_BATCH_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "projection": {
                "projection_id": self.projection_id,
                "projection_version": self.projection_version,
                "definition_sha256": self.projection_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "cells": [value.to_record() for value in self.cells],
            "candidate_outcomes_observed": self.candidate_outcomes_observed,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "batch_sha256": self.batch_sha256}


@runtime_checkable
class MaterializedActionSemanticCellProjectionPort(Protocol):
    """Project portable semantic cells from a sealed proposal population."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionSemanticCellBatch: ...


def _apportion(
    capacity: int,
    shares: tuple[tuple[str, float], ...],
) -> dict[str, int]:
    raw = {name: capacity * share for name, share in shares}
    quotas = {name: math.floor(value) for name, value in raw.items()}
    remaining = capacity - sum(quotas.values())
    order = sorted(
        raw,
        key=lambda name: (-(raw[name] - quotas[name]), name),
    )
    for name in order[:remaining]:
        quotas[name] += 1
    if sum(quotas.values()) != capacity:
        raise AssertionError("largest-remainder apportionment did not close")
    return quotas


@dataclass(frozen=True, slots=True)
class SemanticCoverageScorePortfolioPolicy:
    """Conserve score lanes under one explicit recursive-lineage contract."""

    scorers: tuple[MaterializedActionScorePort, ...]
    scorer_capacity_fractions: tuple[tuple[str, float], ...]
    lineage_scorer_id: str
    lineage_member_scorer_id: str
    lineage_deficit_refill_scorer_id: str
    lineage_capacity_fraction: float
    semantic_projection: MaterializedActionSemanticCellProjectionPort = field(
        repr=False,
        compare=False,
    )
    coverage_strength: float = 1.0 / 3.0
    allow_recursive_score_lane_spillover: bool = False
    hard_feasibility: HardFeasibilityPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    policy_id: ClassVar[str] = (
        SEMANTIC_COVERAGE_SCORE_PORTFOLIO_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        SEMANTIC_COVERAGE_SCORE_PORTFOLIO_POLICY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.scorers) is not tuple or len(self.scorers) < 2:
            raise ValueError("semantic portfolio requires multiple scorers")
        identities = tuple(_scorer_identity(value) for value in self.scorers)
        scorer_ids = tuple(value[0] for value in identities)
        if scorer_ids != tuple(sorted(set(scorer_ids))):
            raise ValueError("scorers must use canonical unique IDs")
        if (
            type(self.scorer_capacity_fractions) is not tuple
            or not self.scorer_capacity_fractions
        ):
            raise ValueError("scorer fractions must be a non-empty exact tuple")
        fraction_ids = tuple(
            value[0] for value in self.scorer_capacity_fractions
        )
        if fraction_ids != scorer_ids:
            raise ValueError("scorer fractions must canonically cover scorers")
        for scorer_id, fraction in self.scorer_capacity_fractions:
            _require_token(scorer_id, name="fraction scorer_id")
            if (
                type(fraction) is not float
                or not math.isfinite(fraction)
                or not 0.0 <= fraction < 1.0
            ):
                raise ValueError("scorer fractions must lie in [0, 1)")
        _require_token(self.lineage_scorer_id, name="lineage_scorer_id")
        if self.lineage_scorer_id not in scorer_ids:
            raise ValueError("lineage scorer is absent")
        _require_token(
            self.lineage_member_scorer_id,
            name="lineage_member_scorer_id",
        )
        if self.lineage_member_scorer_id not in scorer_ids:
            raise ValueError("lineage member scorer is absent")
        _require_token(
            self.lineage_deficit_refill_scorer_id,
            name="lineage_deficit_refill_scorer_id",
        )
        if self.lineage_deficit_refill_scorer_id not in scorer_ids:
            raise ValueError("lineage deficit refill scorer is absent")
        if (
            type(self.lineage_capacity_fraction) is not float
            or not math.isfinite(self.lineage_capacity_fraction)
            or not 0.0 <= self.lineage_capacity_fraction < 1.0
        ):
            raise ValueError("lineage fraction must lie in [0, 1)")
        total_fraction = math.fsum(
            value for _, value in self.scorer_capacity_fractions
        ) + self.lineage_capacity_fraction
        if not math.isclose(total_fraction, 1.0, abs_tol=1.0e-12):
            raise ValueError("scorer and lineage fractions must sum to one")
        if (
            not isinstance(
                self.semantic_projection,
                MaterializedActionSemanticCellProjectionPort,
            )
            or type(self.semantic_projection.projection_version) is not int
            or self.semantic_projection.projection_version <= 0
        ):
            raise TypeError("semantic projection must implement its port")
        _require_token(
            self.semantic_projection.projection_id,
            name="semantic projection_id",
        )
        require_sha256(
            self.semantic_projection.definition_sha256,
            "semantic projection definition_sha256",
        )
        if (
            type(self.coverage_strength) is not float
            or not math.isfinite(self.coverage_strength)
            or not 0.0 < self.coverage_strength < 1.0
        ):
            raise ValueError("coverage_strength must lie in (0, 1)")
        if type(self.allow_recursive_score_lane_spillover) is not bool:
            raise TypeError(
                "allow_recursive_score_lane_spillover must be an exact bool"
            )
        feasibility_identity = (
            None
            if self.hard_feasibility is None
            else validate_hard_feasibility_port(self.hard_feasibility)
        )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _POLICY_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "scorers": [
                        {
                            "scorer_id": scorer_id,
                            "scorer_version": scorer_version,
                            "definition_sha256": definition_sha256,
                            "capacity_fraction_hex": dict(
                                self.scorer_capacity_fractions
                            )[scorer_id].hex(),
                        }
                        for scorer_id, scorer_version, definition_sha256
                        in identities
                    ],
                    "lineage": {
                        "scorer_id": self.lineage_scorer_id,
                        "member_scorer_id": self.lineage_member_scorer_id,
                        "deficit_refill_scorer_id": (
                            self.lineage_deficit_refill_scorer_id
                        ),
                        "capacity_fraction_hex": (
                            self.lineage_capacity_fraction.hex()
                        ),
                        "cell": "recursive_lineage_by_expert_id",
                        "partition": (
                            "pilot_floor_then_recursive_score_lane_competition"
                            if self.allow_recursive_score_lane_spillover
                            else (
                                "exact_maximum_then_nonrecursive_score_lanes"
                            )
                        ),
                    },
                    "semantic_projection": {
                        "projection_id": (
                            self.semantic_projection.projection_id
                        ),
                        "projection_version": (
                            self.semantic_projection.projection_version
                        ),
                        "definition_sha256": (
                            self.semantic_projection.definition_sha256
                        ),
                    },
                    "coverage": {
                        "feature": "categorical_metric_direction_signature",
                        "strength_hex": self.coverage_strength.hex(),
                        "score": (
                            "(1-strength)*within-lane-rank-percentile+"
                            "strength*new-direction-indicator"
                        ),
                    },
                    "capacity_apportionment": (
                        "largest_remainder_fractional_quota"
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
        gathered = await asyncio.gather(
            *(scorer.score(request, proposals) for scorer in self.scorers),
            self.semantic_projection.project(request, proposals),
        )
        score_batches = gathered[:-1]
        semantic_batch = gathered[-1]
        if type(semantic_batch) is not MaterializedActionSemanticCellBatch:
            raise TypeError("semantic projection returned a foreign batch")
        semantic_batch.__post_init__()
        if (
            semantic_batch.residual_request_sha256 != request.request_sha256
            or semantic_batch.proposal_sha256s != proposal_sha256s
            or tuple(value.action_sha256 for value in semantic_batch.cells)
            != action_sha256s
        ):
            raise ValueError("semantic batch differs from sealed universe")
        cell_by_action = {
            value.action_sha256: value for value in semantic_batch.cells
        }

        batch_by_scorer: dict[str, MaterializedActionScoreBatch] = {}
        for batch in score_batches:
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

        infeasible: set[str] = set()
        feasibility_decisions: list[
            tuple[str, HardFeasibilityDecision]
        ] = []
        if self.hard_feasibility is not None:
            for action_sha256 in action_sha256s:
                decision = assess_hard_feasibility(
                    self.hard_feasibility,
                    HardFeasibilityRequest(
                        campaign_scope_sha256=request.campaign_scope_sha256,
                        cutoff_index=request.decision_index,
                        configuration=action_by_sha256[
                            action_sha256
                        ].configuration,
                    ),
                )
                feasibility_decisions.append((action_sha256, decision))
                if decision.verdict is HardFeasibilityVerdict.INFEASIBLE:
                    infeasible.add(action_sha256)
        admissible = tuple(
            value
            for value in actions
            if value.action_sha256 not in infeasible
        )
        if len(admissible) < request.evaluation_slots:
            raise ValueError("screened proposal union cannot fill capacity")

        score_by_scorer = {
            scorer_id: {
                value.action_sha256: value.value
                for value in batch_by_scorer[scorer_id].scores
            }
            for scorer_id in scorer_by_id
        }
        rankings = {
            scorer_id: tuple(
                sorted(
                    admissible,
                    key=lambda value, lane=scorer_id: (
                        -score_by_scorer[lane][value.action_sha256],
                        value.native_rank,
                        value.action_sha256,
                    ),
                )
            )
            for scorer_id in scorer_by_id
        }
        denominator = max(1, len(admissible) - 1)
        percentiles = {
            scorer_id: {
                value.action_sha256: 1.0 - index / denominator
                for index, value in enumerate(rankings[scorer_id])
            }
            for scorer_id in scorer_by_id
        }
        shares = (
            ("__lineage__", self.lineage_capacity_fraction),
            *self.scorer_capacity_fractions,
        )
        quotas = _apportion(request.evaluation_slots, shares)
        lineage_quota = quotas.pop("__lineage__")

        selected: list[MaterializedActionDescriptor] = []
        phenotypes: set[str] = set()
        selection_trace: list[dict[str, object]] = []

        def choose(
            candidates: tuple[MaterializedActionDescriptor, ...],
            *,
            lane: str,
            allocation_kind: str,
        ) -> bool:
            eligible = tuple(
                value
                for value in candidates
                if value.phenotype_identity_sha256 not in phenotypes
            )
            if not eligible:
                return False
            covered = {
                cell_by_action[value.action_sha256].direction_signature
                for value in selected
                if cell_by_action[value.action_sha256].direction_signature
            }

            def coordinates(
                value: MaterializedActionDescriptor,
            ) -> tuple[float, float, float]:
                rank = percentiles[lane][value.action_sha256]
                signature = cell_by_action[
                    value.action_sha256
                ].direction_signature
                novelty = (
                    1.0 if signature and signature not in covered else 0.0
                )
                combined = (
                    (1.0 - self.coverage_strength) * rank
                    + self.coverage_strength * novelty
                )
                return combined, rank, novelty

            winner = min(
                eligible,
                key=lambda value: (
                    -coordinates(value)[0],
                    value.native_rank,
                    value.action_sha256,
                ),
            )
            combined, rank, novelty = coordinates(winner)
            selected.append(winner)
            phenotypes.add(winner.phenotype_identity_sha256)
            selection_trace.append(
                {
                    "ordinal": len(selected),
                    "allocation_kind": allocation_kind,
                    "score_lane": lane,
                    "action_sha256": winner.action_sha256,
                    "score_hex": score_by_scorer[lane][
                        winner.action_sha256
                    ].hex(),
                    "rank_percentile_hex": rank.hex(),
                    "direction_novelty_hex": novelty.hex(),
                    "combined_selection_score_hex": combined.hex(),
                    "direction_signature": [
                        {
                            "metric_id": metric_id,
                            "direction": direction,
                        }
                        for metric_id, direction
                        in cell_by_action[
                            winner.action_sha256
                        ].direction_signature
                    ],
                    "recursive_lineage": cell_by_action[
                        winner.action_sha256
                    ].recursive_lineage,
                    "candidate_outcomes_observed": False,
                }
            )
            return True

        lineage_expert_ids = tuple(
            sorted(
                {
                    value.expert_id
                    for value in admissible
                    if cell_by_action[
                        value.action_sha256
                    ].recursive_lineage
                }
            )
        )
        lineage_cell_champions = tuple(
            min(
                (
                    value
                    for value in admissible
                    if value.expert_id == expert_id
                    and cell_by_action[
                        value.action_sha256
                    ].recursive_lineage
                ),
                key=lambda value: (
                    -score_by_scorer[self.lineage_member_scorer_id][
                        value.action_sha256
                    ],
                    value.native_rank,
                    value.action_sha256,
                ),
            )
            for expert_id in lineage_expert_ids
        )
        lineage_remaining = tuple(lineage_cell_champions)
        while lineage_remaining and len(selection_trace) < lineage_quota:
            if not choose(
                lineage_remaining,
                lane=self.lineage_scorer_id,
                allocation_kind="recursive_lineage_cell",
            ):
                break
            lineage_remaining = tuple(
                value
                for value in lineage_remaining
                if value.action_sha256 != selected[-1].action_sha256
            )
        lineage_selected = len(selected)

        score_lane_rankings = {
            scorer_id: tuple(
                value
                for value in ranking
                if (
                    self.allow_recursive_score_lane_spillover
                    or not cell_by_action[
                        value.action_sha256
                    ].recursive_lineage
                )
            )
            for scorer_id, ranking in rankings.items()
        }
        for ordinal in range(max(quotas.values(), default=0)):
            for scorer_id in sorted(quotas):
                if ordinal >= quotas[scorer_id]:
                    continue
                if not choose(
                    score_lane_rankings[scorer_id],
                    lane=scorer_id,
                    allocation_kind="score_lane",
                ):
                    raise ValueError("semantic score lane cannot close")
        while len(selected) < request.evaluation_slots:
            if not choose(
                score_lane_rankings[
                    self.lineage_deficit_refill_scorer_id
                ],
                lane=self.lineage_deficit_refill_scorer_id,
                allocation_kind="lineage_deficit_refill",
            ):
                raise ValueError("lineage deficit refill cannot close capacity")
        if len(selected) != request.evaluation_slots:
            raise AssertionError("semantic portfolio exceeded capacity")

        feasibility_record = {
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
            "rejected_actions": [
                {
                    "action_sha256": action_sha256,
                    "decision": decision.to_record(),
                }
                for action_sha256, decision in feasibility_decisions
                if decision.verdict
                is HardFeasibilityVerdict.INFEASIBLE
            ],
            "unknown_actions_remain_eligible": True,
            "candidate_outcomes_observed": False,
        }
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
                    "score_batch_sha256s": {
                        scorer_id: batch_by_scorer[
                            scorer_id
                        ].batch_sha256
                        for scorer_id in sorted(batch_by_scorer)
                    },
                    "semantic_cell_batch_sha256": (
                        semantic_batch.batch_sha256
                    ),
                    "nominal_capacity_quotas": {
                        "recursive_lineage": lineage_quota,
                        **{
                            scorer_id: quotas[scorer_id]
                            for scorer_id in sorted(quotas)
                        },
                    },
                    "realized_lineage_count": lineage_selected,
                    "final_recursive_lineage_count": sum(
                        cell_by_action[
                            value.action_sha256
                        ].recursive_lineage
                        for value in selected
                    ),
                    "lineage_deficit_refill_count": (
                        lineage_quota - lineage_selected
                    ),
                    "lineage_partition": (
                        "pilot_floor_then_recursive_score_lane_competition"
                        if self.allow_recursive_score_lane_spillover
                        else (
                            "exact_maximum_then_nonrecursive_score_lanes"
                        )
                    ),
                    "lineage_member_scorer_id": (
                        self.lineage_member_scorer_id
                    ),
                    "coverage_strength_hex": (
                        self.coverage_strength.hex()
                    ),
                    "selection_trace": selection_trace,
                    "hard_feasibility": feasibility_record,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "MaterializedActionSemanticCell",
    "MaterializedActionSemanticCellBatch",
    "MaterializedActionSemanticCellProjectionPort",
    "SEMANTIC_COVERAGE_SCORE_PORTFOLIO_POLICY_ID",
    "SEMANTIC_COVERAGE_SCORE_PORTFOLIO_POLICY_VERSION",
    "SemanticCoverageScorePortfolioPolicy",
]

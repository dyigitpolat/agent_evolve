"""Outcome-blind exposure floors over heterogeneous proposal sources.

The application core must be able to conserve genuinely different search
processes without learning what a "model", "provider", or workload is.  This
module therefore exposes two small inverted APIs:

* a projection assigns every sealed materialized action to an opaque source
  group; and
* an ordinary pre-evaluation score authority ranks candidates within groups.

The allocation decorator first obtains one complete K-slate from an injected
base policy.  It then performs the minimum deterministic replacements needed
to satisfy declared group floors, choosing replacements and victims by the
same injected score.  No current candidate outcome exists at this boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
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
    MaterializedActionAllocationPolicyPort,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

SOURCE_EXPOSURE_ALLOCATION_POLICY_ID = (
    "minimum_source_exposure_allocator"
)
SOURCE_EXPOSURE_ALLOCATION_POLICY_VERSION = 1
MINIMUM_EXPERT_SOURCE_EXPOSURE_SLATE_FEASIBILITY_VERSION = 1
EXPLICIT_EXPERT_SOURCE_GROUP_PROJECTION_ID = (
    "explicit_expert_source_group_projection"
)
EXPLICIT_EXPERT_SOURCE_GROUP_PROJECTION_VERSION = 1

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_ASSIGNMENT_BATCH_DOMAIN = (
    b"agent-evolve:materialized-source-group-assignment-batch:v1\x00"
)
_PROJECTION_DEFINITION_DOMAIN = (
    b"agent-evolve:explicit-expert-source-group-projection:v1\x00"
)
_POLICY_DEFINITION_DOMAIN = (
    b"agent-evolve:minimum-source-exposure-allocation:v1\x00"
)
_SLATE_FEASIBILITY_DEFINITION_DOMAIN = (
    b"agent-evolve:minimum-expert-source-exposure-slate-feasibility:v1\x00"
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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _policy_identity(
    value: MaterializedActionAllocationPolicyPort,
) -> tuple[str, int, str]:
    if not isinstance(value, MaterializedActionAllocationPolicyPort):
        raise TypeError("base policy must implement its application port")
    identity = (
        getattr(value, "policy_id", None),
        getattr(value, "policy_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="base policy_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("base policy_version must be positive")
    require_sha256(identity[2], "base policy definition_sha256")
    return identity  # type: ignore[return-value]


def _scorer_identity(
    value: MaterializedActionScorePort,
) -> tuple[str, int, str]:
    if not isinstance(value, MaterializedActionScorePort):
        raise TypeError("priority scorer must implement its application port")
    identity = (
        getattr(value, "scorer_id", None),
        getattr(value, "scorer_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="priority scorer_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("priority scorer_version must be positive")
    require_sha256(identity[2], "priority scorer definition_sha256")
    return identity  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class MaterializedActionSourceGroup:
    action_sha256: str
    group_id: str

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _require_token(self.group_id, name="group_id")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "group_id": self.group_id,
        }


@dataclass(frozen=True, slots=True)
class MaterializedActionSourceGroupBatch:
    projection_id: str
    projection_version: int
    projection_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    assignments: tuple[MaterializedActionSourceGroup, ...]
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
        if type(self.assignments) is not tuple or not self.assignments:
            raise ValueError("assignments must be a non-empty exact tuple")
        for value in self.assignments:
            if type(value) is not MaterializedActionSourceGroup:
                raise TypeError("assignments must contain exact values")
            value.__post_init__()
        action_ids = tuple(value.action_sha256 for value in self.assignments)
        if action_ids != tuple(sorted(set(action_ids))):
            raise ValueError("assignments must be action-canonical")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be exact")
        if self.candidate_outcomes_observed:
            raise ValueError("source groups cannot observe current outcomes")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("source-group evidence must be frozen")
        object.__setattr__(
            self,
            "batch_sha256",
            _hash(
                _ASSIGNMENT_BATCH_DOMAIN,
                {
                    "schema_version": 1,
                    "projection": {
                        "projection_id": self.projection_id,
                        "projection_version": self.projection_version,
                        "definition_sha256": (
                            self.projection_definition_sha256
                        ),
                    },
                    "residual_request_sha256": (
                        self.residual_request_sha256
                    ),
                    "proposal_sha256s": list(self.proposal_sha256s),
                    "assignments": [
                        value.to_record() for value in self.assignments
                    ],
                    "candidate_outcomes_observed": False,
                    "evidence_sha256": typed_json_sha256(self.evidence),
                },
            ),
        )

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            "schema_version": 1,
            "projection": {
                "projection_id": self.projection_id,
                "projection_version": self.projection_version,
                "definition_sha256": self.projection_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "assignments": [
                value.to_record() for value in self.assignments
            ],
            "candidate_outcomes_observed": False,
            "evidence_sha256": typed_json_sha256(self.evidence),
            "batch_sha256": self.batch_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class MaterializedActionSourceGroupProjectionPort(Protocol):
    projection_id: str
    projection_version: int
    definition_sha256: str

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionSourceGroupBatch: ...


def _projection_identity(
    value: MaterializedActionSourceGroupProjectionPort,
) -> tuple[str, int, str]:
    if not isinstance(value, MaterializedActionSourceGroupProjectionPort):
        raise TypeError("source projection must implement its application port")
    identity = (
        getattr(value, "projection_id", None),
        getattr(value, "projection_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="source projection_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("source projection_version must be positive")
    require_sha256(identity[2], "source projection definition_sha256")
    return identity  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ExplicitExpertSourceGroupProjection:
    """Map opaque expert identities to opaque source groups."""

    expert_group_bindings: tuple[tuple[str, str], ...]
    projection_id: str = EXPLICIT_EXPERT_SOURCE_GROUP_PROJECTION_ID
    projection_version: int = (
        EXPLICIT_EXPERT_SOURCE_GROUP_PROJECTION_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.expert_group_bindings) is not tuple
            or not self.expert_group_bindings
            or self.expert_group_bindings
            != tuple(sorted(self.expert_group_bindings))
        ):
            raise ValueError(
                "expert_group_bindings must be non-empty and canonical"
            )
        expert_ids: list[str] = []
        for expert_id, group_id in self.expert_group_bindings:
            _require_token(expert_id, name="expert_id")
            _require_token(group_id, name="group_id")
            expert_ids.append(expert_id)
        if len(expert_ids) != len(set(expert_ids)):
            raise ValueError("expert_group_bindings repeat an expert")
        _require_token(self.projection_id, name="projection_id")
        if (
            self.projection_version
            != EXPLICIT_EXPERT_SOURCE_GROUP_PROJECTION_VERSION
        ):
            raise ValueError("projection_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _PROJECTION_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projection_id": self.projection_id,
                    "projection_version": self.projection_version,
                    "expert_group_bindings": [
                        {
                            "expert_id": expert_id,
                            "group_id": group_id,
                        }
                        for expert_id, group_id
                        in self.expert_group_bindings
                    ],
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionSourceGroupBatch:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        group_by_expert = dict(self.expert_group_bindings)
        assignments: list[MaterializedActionSourceGroup] = []
        proposal_sha256s: list[str] = []
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
            group_id = group_by_expert.get(proposal.expert_id)
            if group_id is None:
                raise ValueError("proposal expert has no source-group binding")
            proposal_sha256s.append(proposal.proposal_sha256)
            assignments.extend(
                MaterializedActionSourceGroup(
                    action_sha256=action.action_sha256,
                    group_id=group_id,
                )
                for action in proposal.actions
            )
        if set(group_by_expert) != {
            value.expert_id for value in proposals
        }:
            raise ValueError("source-group bindings name absent experts")
        return MaterializedActionSourceGroupBatch(
            projection_id=self.projection_id,
            projection_version=self.projection_version,
            projection_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(sorted(proposal_sha256s)),
            assignments=tuple(
                sorted(assignments, key=lambda value: value.action_sha256)
            ),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "expert_count": len(group_by_expert),
                    "group_ids": sorted(set(group_by_expert.values())),
                    "mapping_is_explicit": True,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MinimumExpertSourceExposureSlateFeasibility:
    """Require phenotype uniqueness and opaque expert-group exposure floors."""

    expert_group_bindings: tuple[tuple[str, str], ...]
    minimum_exposures: tuple[tuple[str, int], ...]
    feasibility_version: int = (
        MINIMUM_EXPERT_SOURCE_EXPOSURE_SLATE_FEASIBILITY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        projection = ExplicitExpertSourceGroupProjection(
            expert_group_bindings=self.expert_group_bindings,
        )
        if (
            type(self.minimum_exposures) is not tuple
            or not self.minimum_exposures
            or self.minimum_exposures
            != tuple(sorted(self.minimum_exposures))
        ):
            raise ValueError(
                "minimum_exposures must be non-empty and canonical"
            )
        group_ids: list[str] = []
        for group_id, minimum in self.minimum_exposures:
            _require_token(group_id, name="minimum exposure group_id")
            if type(minimum) is not int or minimum <= 0:
                raise ValueError("minimum exposures must be positive")
            group_ids.append(group_id)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("minimum_exposures repeat a group")
        available_groups = {
            group_id for _expert_id, group_id
            in self.expert_group_bindings
        }
        if not set(group_ids) <= available_groups:
            raise ValueError("minimum exposure group has no expert binding")
        if (
            self.feasibility_version
            != MINIMUM_EXPERT_SOURCE_EXPOSURE_SLATE_FEASIBILITY_VERSION
        ):
            raise ValueError("feasibility_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _SLATE_FEASIBILITY_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "feasibility_version": self.feasibility_version,
                    "source_projection_definition_sha256": (
                        projection.definition_sha256
                    ),
                    "expert_group_bindings": [
                        {
                            "expert_id": expert_id,
                            "group_id": group_id,
                        }
                        for expert_id, group_id
                        in self.expert_group_bindings
                    ],
                    "minimum_exposures": [
                        {"group_id": group_id, "minimum": minimum}
                        for group_id, minimum in self.minimum_exposures
                    ],
                    "phenotype_unique": True,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def permits(
        self,
        actions: tuple[MaterializedActionDescriptor, ...],
    ) -> bool:
        self.__post_init__()
        if type(actions) is not tuple or not actions:
            return False
        if any(
            type(value) is not MaterializedActionDescriptor
            for value in actions
        ):
            raise TypeError("slate contains a foreign action")
        for value in actions:
            value.__post_init__()
        phenotypes = tuple(
            value.phenotype_identity_sha256 for value in actions
        )
        if len(phenotypes) != len(set(phenotypes)):
            return False
        group_by_expert = dict(self.expert_group_bindings)
        groups = []
        for value in actions:
            group_id = group_by_expert.get(value.expert_id)
            if group_id is None:
                return False
            groups.append(group_id)
        return all(
            groups.count(group_id) >= minimum
            for group_id, minimum in self.minimum_exposures
        )


def _selection_trace(
    requirement: MaterializedActionAllocationRequirement,
) -> tuple[dict[str, object], ...]:
    evidence = thaw_json(requirement.evidence)
    if type(evidence) is not dict:
        raise TypeError("base allocation evidence must be an object")
    raw = evidence.get("selection_trace")
    if type(raw) is not list:
        raise ValueError("base allocation evidence lacks selection_trace")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for fallback, value in enumerate(raw, start=1):
        if type(value) is not dict:
            raise TypeError("base selection rows must be objects")
        row = dict(value)
        action_sha256 = row.get("action_sha256")
        if type(action_sha256) is not str:
            raise TypeError("base selection row lacks action_sha256")
        require_sha256(action_sha256, "base action_sha256")
        if action_sha256 in seen:
            raise ValueError("base selection trace repeats an action")
        seen.add(action_sha256)
        ordinal = row.get("ordinal", fallback)
        if type(ordinal) is not int or ordinal <= 0:
            raise ValueError("base selection ordinal must be positive")
        row["ordinal"] = ordinal
        rows.append(row)
    if seen != set(requirement.required_action_sha256s):
        raise ValueError("base selection trace differs from required actions")
    return tuple(
        sorted(
            rows,
            key=lambda value: (
                int(value["ordinal"]),
                str(value["action_sha256"]),
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class MinimumSourceExposureAllocationPolicy:
    """Decorate a complete base slate with score-ranked source floors."""

    base_policy: MaterializedActionAllocationPolicyPort = field(
        repr=False,
        compare=False,
    )
    priority_scorer: MaterializedActionScorePort = field(
        repr=False,
        compare=False,
    )
    source_projection: MaterializedActionSourceGroupProjectionPort = field(
        repr=False,
        compare=False,
    )
    minimum_exposures: tuple[tuple[str, int], ...]
    policy_id: ClassVar[str] = SOURCE_EXPOSURE_ALLOCATION_POLICY_ID
    policy_version: ClassVar[int] = (
        SOURCE_EXPOSURE_ALLOCATION_POLICY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        base_identity = _policy_identity(self.base_policy)
        scorer_identity = _scorer_identity(self.priority_scorer)
        projection_identity = _projection_identity(
            self.source_projection
        )
        if (
            type(self.minimum_exposures) is not tuple
            or not self.minimum_exposures
            or self.minimum_exposures
            != tuple(sorted(self.minimum_exposures))
        ):
            raise ValueError(
                "minimum_exposures must be non-empty and canonical"
            )
        group_ids: list[str] = []
        for group_id, minimum in self.minimum_exposures:
            _require_token(group_id, name="minimum exposure group_id")
            if type(minimum) is not int or minimum <= 0:
                raise ValueError("minimum exposures must be positive")
            group_ids.append(group_id)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("minimum_exposures repeat a group")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _POLICY_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "base_policy": {
                        "policy_id": base_identity[0],
                        "policy_version": base_identity[1],
                        "definition_sha256": base_identity[2],
                    },
                    "priority_scorer": {
                        "scorer_id": scorer_identity[0],
                        "scorer_version": scorer_identity[1],
                        "definition_sha256": scorer_identity[2],
                    },
                    "source_projection": {
                        "projection_id": projection_identity[0],
                        "projection_version": projection_identity[1],
                        "definition_sha256": projection_identity[2],
                    },
                    "minimum_exposures": [
                        {"group_id": group_id, "minimum": minimum}
                        for group_id, minimum in self.minimum_exposures
                    ],
                    "repair": (
                        "minimum_score_ranked_replacements_of_lowest_score_"
                        "removable_members"
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
        actions: list[MaterializedActionDescriptor] = []
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
            actions.extend(proposal.actions)
        action_by_sha256 = {
            value.action_sha256: value for value in actions
        }
        if len(action_by_sha256) != len(actions):
            raise ValueError("proposal market repeats an action")
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        base, score_batch, group_batch = await asyncio.gather(
            self.base_policy.require(request, proposals),
            self.priority_scorer.score(request, proposals),
            self.source_projection.project(request, proposals),
        )
        if type(base) is not MaterializedActionAllocationRequirement:
            raise TypeError("base policy returned a foreign requirement")
        base.__post_init__()
        if type(score_batch) is not MaterializedActionScoreBatch:
            raise TypeError("priority scorer returned a foreign batch")
        score_batch.__post_init__()
        if type(group_batch) is not MaterializedActionSourceGroupBatch:
            raise TypeError("source projection returned a foreign batch")
        group_batch.__post_init__()
        if (
            (
                base.residual_request_sha256,
                base.proposal_sha256s,
            )
            != (request.request_sha256, proposal_sha256s)
            or (
                score_batch.residual_request_sha256,
                score_batch.proposal_sha256s,
            )
            != (request.request_sha256, proposal_sha256s)
            or (
                group_batch.residual_request_sha256,
                group_batch.proposal_sha256s,
            )
            != (request.request_sha256, proposal_sha256s)
        ):
            raise ValueError("allocation inputs saw different sealed markets")
        action_ids = tuple(sorted(action_by_sha256))
        if (
            tuple(value.action_sha256 for value in score_batch.scores)
            != action_ids
            or tuple(
                value.action_sha256 for value in group_batch.assignments
            )
            != action_ids
        ):
            raise ValueError("score/group batches do not cover the market")
        if len(base.required_action_sha256s) != request.evaluation_slots:
            raise ValueError("source exposure requires a complete base K-slate")
        if sum(value for _group, value in self.minimum_exposures) > (
            request.evaluation_slots
        ):
            raise ValueError("minimum source exposures exceed K")

        score_by_action = {
            value.action_sha256: value.value
            for value in score_batch.scores
        }
        group_by_action = {
            value.action_sha256: value.group_id
            for value in group_batch.assignments
        }
        minima = dict(self.minimum_exposures)
        if not set(minima).issubset(set(group_by_action.values())):
            raise ValueError("a required source group is absent")
        base_trace = _selection_trace(base)
        selected = [
            action_by_sha256[str(value["action_sha256"])]
            for value in base_trace
        ]
        if len(
            {value.phenotype_identity_sha256 for value in selected}
        ) != len(selected):
            raise ValueError("base K-slate repeats a phenotype")
        original_rows = {
            str(value["action_sha256"]): dict(value)
            for value in base_trace
        }
        inserted: set[str] = set()
        replacements: list[dict[str, object]] = []

        def group_counts() -> dict[str, int]:
            result: dict[str, int] = {}
            for action in selected:
                group_id = group_by_action[action.action_sha256]
                result[group_id] = result.get(group_id, 0) + 1
            return result

        rankings = tuple(
            sorted(
                actions,
                key=lambda value: (
                    -score_by_action[value.action_sha256],
                    value.native_rank,
                    value.expert_id,
                    value.action_sha256,
                ),
            )
        )
        for group_id, minimum in self.minimum_exposures:
            while group_counts().get(group_id, 0) < minimum:
                counts = group_counts()
                removable_indices = tuple(
                    index
                    for index, action in enumerate(selected)
                    if not action.reference_action
                    and counts[group_by_action[action.action_sha256]]
                    > minima.get(
                        group_by_action[action.action_sha256],
                        0,
                    )
                )
                if not removable_indices:
                    raise ValueError(
                        "source floors leave no removable base member"
                    )
                replacement_pair: tuple[
                    MaterializedActionDescriptor,
                    int,
                ] | None = None
                selected_ids = {
                    value.action_sha256 for value in selected
                }
                for candidate in rankings:
                    if (
                        candidate.action_sha256 in selected_ids
                        or group_by_action[candidate.action_sha256]
                        != group_id
                    ):
                        continue
                    eligible_victims = tuple(
                        index
                        for index in removable_indices
                        if all(
                            other_index == index
                            or other.phenotype_identity_sha256
                            != candidate.phenotype_identity_sha256
                            for other_index, other in enumerate(selected)
                        )
                    )
                    if not eligible_victims:
                        continue
                    victim_index = min(
                        eligible_victims,
                        key=lambda index: (
                            score_by_action[
                                selected[index].action_sha256
                            ],
                            -selected[index].native_rank,
                            selected[index].action_sha256,
                        ),
                    )
                    replacement_pair = (candidate, victim_index)
                    break
                if replacement_pair is None:
                    raise ValueError(
                        "source floor lacks a unique-phenotype replacement"
                    )
                candidate, victim_index = replacement_pair
                victim = selected[victim_index]
                selected[victim_index] = candidate
                inserted.add(candidate.action_sha256)
                replacements.append(
                    {
                        "ordinal": len(replacements) + 1,
                        "required_group_id": group_id,
                        "inserted_action_sha256": (
                            candidate.action_sha256
                        ),
                        "inserted_score_hex": score_by_action[
                            candidate.action_sha256
                        ].hex(),
                        "removed_action_sha256": victim.action_sha256,
                        "removed_group_id": group_by_action[
                            victim.action_sha256
                        ],
                        "removed_score_hex": score_by_action[
                            victim.action_sha256
                        ].hex(),
                        "candidate_outcomes_observed": False,
                    }
                )

        final_counts = group_counts()
        if any(
            final_counts.get(group_id, 0) < minimum
            for group_id, minimum in self.minimum_exposures
        ):
            raise AssertionError("source exposure repair did not close")
        if len(selected) != request.evaluation_slots or len(
            {value.phenotype_identity_sha256 for value in selected}
        ) != len(selected):
            raise AssertionError("source exposure repair corrupted K")
        selection_trace: list[dict[str, object]] = []
        for ordinal, action in enumerate(selected, start=1):
            if action.action_sha256 in inserted:
                row: dict[str, object] = {
                    "ordinal": ordinal,
                    "allocation_kind": "source_exposure_floor",
                    "score_lane": score_batch.scorer_id,
                    "action_sha256": action.action_sha256,
                    "score_hex": score_by_action[
                        action.action_sha256
                    ].hex(),
                    "source_group_id": group_by_action[
                        action.action_sha256
                    ],
                    "candidate_outcomes_observed": False,
                }
            else:
                row = dict(original_rows[action.action_sha256])
                row["ordinal"] = ordinal
                row["source_group_id"] = group_by_action[
                    action.action_sha256
                ]
                row["source_exposure_status"] = "base_retained"
            selection_trace.append(row)
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
                    "base_requirement_sha256": base.requirement_sha256,
                    "priority_score_batch_sha256": (
                        score_batch.batch_sha256
                    ),
                    "source_group_batch_sha256": group_batch.batch_sha256,
                    "minimum_exposures": [
                        {"group_id": group_id, "minimum": minimum}
                        for group_id, minimum in self.minimum_exposures
                    ],
                    "base_group_counts": {
                        group_id: sum(
                            group_by_action[action_sha256] == group_id
                            for action_sha256
                            in base.required_action_sha256s
                        )
                        for group_id in sorted(set(group_by_action.values()))
                    },
                    "final_group_counts": {
                        group_id: final_counts[group_id]
                        for group_id in sorted(final_counts)
                    },
                    "replacements": replacements,
                    "selection_trace": selection_trace,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "EXPLICIT_EXPERT_SOURCE_GROUP_PROJECTION_ID",
    "EXPLICIT_EXPERT_SOURCE_GROUP_PROJECTION_VERSION",
    "SOURCE_EXPOSURE_ALLOCATION_POLICY_ID",
    "SOURCE_EXPOSURE_ALLOCATION_POLICY_VERSION",
    "MINIMUM_EXPERT_SOURCE_EXPOSURE_SLATE_FEASIBILITY_VERSION",
    "ExplicitExpertSourceGroupProjection",
    "MaterializedActionSourceGroup",
    "MaterializedActionSourceGroupBatch",
    "MaterializedActionSourceGroupProjectionPort",
    "MinimumSourceExposureAllocationPolicy",
    "MinimumExpertSourceExposureSlateFeasibility",
]

"""Outcome-blind evaluation allocation across heterogeneous proposal experts.

An LLM's native ranks are meaningful only within the proposal set produced by
that expert.  Treating ranks from different experts as a single calibrated
score silently collapses source diversity; trusting only each expert's head
silently assumes rank monotonicity before any real evaluator evidence exists.

This module implements a small, workload-neutral cold-start design:

* proposals are partitioned by opaque ``expert_id``;
* one coverage slot is reserved per expert when the budget permits;
* residual slots are apportioned by support size using exact D'Hondt
  quotients; and
* each expert's quota samples the head and deterministic interior rank
  quantiles.

The policy sees no objective value, forecast score, workload identity, model
identity, provider identity, or prompt text.  It is intended only for an
unidentified cold-start block.  Once real outcomes exist, a prior-only
prequential acquisition policy should replace it rather than mix post-treatment
evidence into this design.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionDescriptor,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256

STRATIFIED_COLD_START_ALLOCATOR_ID = (
    "support_proportional_rank_stratified_cold_start"
)
STRATIFIED_COLD_START_ALLOCATOR_VERSION = 1
STRATIFIED_COLD_START_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:support-proportional-rank-stratified-cold-start:v1;"
    b"eligibility=authenticated-unique-materialized-phenotypes;"
    b"lane=opaque-expert-id;"
    b"coverage=floor-one-per-lane-when-capacity-permits;"
    b"partial-coverage=largest-support-first;"
    b"residual-apportionment=exact-dhondt-support-over-quota-plus-one;"
    b"within-lane=head-preserving-floor-j-times-n-over-quota;"
    b"native-rank=within-expert-order-only;"
    b"audited-axes=role,variation-scale,parent,structural-cell;"
    b"outcomes-consulted=false;"
    b"forbidden-inputs=workload-id,model-id,provider-id,objective-name,prompt"
).hexdigest()

LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_ID = (
    "support_proportional_low_discrepancy_rank_stratified_exploration"
)
LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_VERSION = 1
LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:support-proportional-low-discrepancy-rank-stratified-"
    b"exploration:v1;"
    b"eligibility=authenticated-unique-materialized-phenotypes;"
    b"lane=opaque-expert-id;"
    b"coverage=floor-one-per-lane-when-capacity-permits;"
    b"partial-coverage=largest-support-first;"
    b"residual-apportionment=exact-dhondt-support-over-quota-plus-one;"
    b"within-lane=deduplicated-base-two-radical-inverse-rank-permutation;"
    b"schedule=contiguous-quota-block-cycled-by-decision-index;"
    b"native-rank=within-expert-order-only;"
    b"audited-axes=role,variation-scale,parent,structural-cell;"
    b"outcomes-consulted=false;"
    b"forbidden-inputs=workload-id,model-id,provider-id,objective-name,prompt"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_PROPOSAL_DOMAIN = b"agent-evolve:stratified-cold-start-proposal:v1\x00"
_REQUEST_DOMAIN = b"agent-evolve:stratified-cold-start-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:stratified-cold-start-decision:v1\x00"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _require_token(value: str, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _candidate(value: CandidateId, name: str) -> None:
    if type(value) is not CandidateId:
        raise TypeError(f"{name} must contain exact CandidateId values")
    CandidateId.__post_init__(value)


@dataclass(frozen=True, slots=True, eq=False)
class StratifiedColdStartProposal:
    """Minimal authenticated proposal view required by the cold-start design."""

    proposal_id: str
    phenotype_identity_sha256: str
    expert_id: str
    native_rank: int
    parent_ids: tuple[CandidateId, ...]
    role_id: str
    variation_scale: int
    structural_cell: str
    proposal_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.proposal_id, "proposal_id")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        _require_token(self.expert_id, "expert_id")
        if type(self.native_rank) is not int or self.native_rank <= 0:
            raise ValueError("native_rank must be a positive exact integer")
        if type(self.parent_ids) is not tuple or len(self.parent_ids) > 8:
            raise ValueError(
                "parent_ids must be an exact tuple with arity at most eight"
            )
        for value in self.parent_ids:
            _candidate(value, "parent_ids")
        parent_values = tuple(value.value for value in self.parent_ids)
        if len(parent_values) != len(set(parent_values)):
            raise ValueError("parent_ids must be unique")
        _require_token(self.role_id, "role_id")
        if type(self.variation_scale) is not int or self.variation_scale <= 0:
            raise ValueError("variation_scale must be a positive exact integer")
        _require_token(self.structural_cell, "structural_cell")
        object.__setattr__(
            self,
            "proposal_sha256",
            _hash(_PROPOSAL_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "proposal_id": self.proposal_id,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "expert_id": self.expert_id,
            "native_rank": self.native_rank,
            "parent_ids": [value.value for value in self.parent_ids],
            "parent_arity": len(self.parent_ids),
            "role_id": self.role_id,
            "variation_scale": self.variation_scale,
            "structural_cell": self.structural_cell,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "proposal_sha256": self.proposal_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is StratifiedColdStartProposal
            and self.proposal_sha256 == other.proposal_sha256
        )

    __hash__ = None


def stratified_proposal_from_materialized_action(
    action: MaterializedActionDescriptor,
    *,
    proposal_id: str,
    variation_scale: int,
    structural_cell: str,
) -> StratifiedColdStartProposal:
    """Project the generic materialized-action API onto the cold-start view."""

    if type(action) is not MaterializedActionDescriptor:
        raise TypeError("action must be an exact MaterializedActionDescriptor")
    action.__post_init__()
    return StratifiedColdStartProposal(
        proposal_id=proposal_id,
        phenotype_identity_sha256=action.phenotype_identity_sha256,
        expert_id=action.expert_id,
        native_rank=action.native_rank,
        parent_ids=action.parent_ids,
        role_id=action.role_id,
        variation_scale=variation_scale,
        structural_cell=structural_cell,
    )


@dataclass(frozen=True, slots=True, eq=False)
class StratifiedColdStartAllocationRequest:
    """One outcome-blind finite proposal universe and its evaluator capacity."""

    decision_scope_sha256: str
    decision_index: int
    proposals: tuple[StratifiedColdStartProposal, ...]
    evaluation_slots: int
    request_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.decision_scope_sha256, "decision_scope_sha256")
        if type(self.decision_index) is not int or self.decision_index <= 0:
            raise ValueError("decision_index must be a positive exact integer")
        if type(self.proposals) is not tuple or not self.proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        for value in self.proposals:
            if type(value) is not StratifiedColdStartProposal:
                raise TypeError("proposals must contain exact proposal views")
            value.__post_init__()
        proposal_ids = tuple(value.proposal_id for value in self.proposals)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("proposal_id values must be unique")
        proposal_hashes = tuple(value.proposal_sha256 for value in self.proposals)
        if len(proposal_hashes) != len(set(proposal_hashes)):
            raise ValueError("proposal identities must be unique")
        phenotypes = tuple(
            value.phenotype_identity_sha256 for value in self.proposals
        )
        if len(phenotypes) != len(set(phenotypes)):
            raise ValueError(
                "upstream materialization must collapse duplicate phenotypes"
            )
        if (
            type(self.evaluation_slots) is not int
            or not 1 <= self.evaluation_slots <= len(self.proposals)
        ):
            raise ValueError(
                "evaluation_slots must fit the finite proposal universe"
            )
        object.__setattr__(
            self,
            "request_sha256",
            _hash(_REQUEST_DOMAIN, self._unsigned_record()),
        )

    @property
    def canonical_proposals(self) -> tuple[StratifiedColdStartProposal, ...]:
        return tuple(sorted(self.proposals, key=lambda value: value.proposal_id))

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "decision_scope_sha256": self.decision_scope_sha256,
            "decision_index": self.decision_index,
            "evaluation_slots": self.evaluation_slots,
            "proposal_sha256s": [
                value.proposal_sha256 for value in self.canonical_proposals
            ],
            "proposal_universe_count": len(self.proposals),
            "outcomes_consulted": False,
        }

    def to_record(self, *, include_proposals: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "request_sha256": self.request_sha256,
        }
        if include_proposals:
            record["proposals"] = [
                value.to_record() for value in self.canonical_proposals
            ]
        return record

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is StratifiedColdStartAllocationRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class StratifiedColdStartLaneAllocation:
    expert_id: str
    support_count: int
    allocated_slots: int

    def __post_init__(self) -> None:
        _require_token(self.expert_id, "expert_id")
        if type(self.support_count) is not int or self.support_count <= 0:
            raise ValueError("support_count must be a positive exact integer")
        if (
            type(self.allocated_slots) is not int
            or not 0 <= self.allocated_slots <= self.support_count
        ):
            raise ValueError("allocated_slots must fit the expert support")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "expert_id": self.expert_id,
            "support_count": self.support_count,
            "allocated_slots": self.allocated_slots,
        }


@dataclass(frozen=True, slots=True)
class StratifiedColdStartAllocationMember:
    proposal: StratifiedColdStartProposal
    selection_ticket_ordinal: int
    lane_support_count: int
    lane_allocated_slots: int
    lane_quantile_ordinal: int
    target_rank_index: int
    selected_rank_index: int

    def __post_init__(self) -> None:
        if type(self.proposal) is not StratifiedColdStartProposal:
            raise TypeError("proposal must be an exact proposal view")
        self.proposal.__post_init__()
        for name in (
            "selection_ticket_ordinal",
            "lane_support_count",
            "lane_allocated_slots",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.lane_allocated_slots > self.lane_support_count:
            raise ValueError("lane allocation exceeds its support")
        if (
            type(self.lane_quantile_ordinal) is not int
            or not 0 <= self.lane_quantile_ordinal < self.lane_allocated_slots
        ):
            raise ValueError("lane_quantile_ordinal lies outside its quota")
        for name in ("target_rank_index", "selected_rank_index"):
            value = getattr(self, name)
            if (
                type(value) is not int
                or not 0 <= value < self.lane_support_count
            ):
                raise ValueError(f"{name} lies outside the expert support")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "proposal": self.proposal.to_record(),
            "selection_ticket_ordinal": self.selection_ticket_ordinal,
            "lane_support_count": self.lane_support_count,
            "lane_allocated_slots": self.lane_allocated_slots,
            "lane_quantile_ordinal": self.lane_quantile_ordinal,
            "target_rank_index": self.target_rank_index,
            "selected_rank_index": self.selected_rank_index,
        }


@dataclass(frozen=True, slots=True, eq=False)
class StratifiedColdStartAllocationDecision:
    request_sha256: str
    lanes: tuple[StratifiedColdStartLaneAllocation, ...]
    members: tuple[StratifiedColdStartAllocationMember, ...]
    allocator_id: str = STRATIFIED_COLD_START_ALLOCATOR_ID
    allocator_version: int = STRATIFIED_COLD_START_ALLOCATOR_VERSION
    allocator_definition_sha256: str = (
        STRATIFIED_COLD_START_ALLOCATOR_DEFINITION_SHA256
    )
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        _require_token(self.allocator_id, "allocator_id")
        if type(self.allocator_version) is not int or self.allocator_version <= 0:
            raise ValueError("allocator_version must be a positive exact integer")
        require_sha256(
            self.allocator_definition_sha256,
            "allocator_definition_sha256",
        )
        if type(self.lanes) is not tuple or not self.lanes:
            raise ValueError("lanes must be a non-empty exact tuple")
        for value in self.lanes:
            if type(value) is not StratifiedColdStartLaneAllocation:
                raise TypeError("lanes must contain exact lane allocations")
            value.__post_init__()
        if tuple(value.expert_id for value in self.lanes) != tuple(
            sorted(value.expert_id for value in self.lanes)
        ):
            raise ValueError("lanes must use canonical expert order")
        if type(self.members) is not tuple or not self.members:
            raise ValueError("members must be a non-empty exact tuple")
        for value in self.members:
            if type(value) is not StratifiedColdStartAllocationMember:
                raise TypeError("members must contain exact allocation members")
            value.__post_init__()
        if tuple(value.selection_ticket_ordinal for value in self.members) != tuple(
            range(1, len(self.members) + 1)
        ):
            raise ValueError("member ticket ordinals must be contiguous")
        if len(
            {value.proposal.phenotype_identity_sha256 for value in self.members}
        ) != len(self.members):
            raise ValueError("selected members must have unique phenotypes")
        lane_by_id = {value.expert_id: value for value in self.lanes}
        selected_by_lane: dict[str, int] = {}
        for member in self.members:
            lane = lane_by_id.get(member.proposal.expert_id)
            if lane is None:
                raise ValueError("selected member names an absent expert lane")
            if (
                member.lane_support_count != lane.support_count
                or member.lane_allocated_slots != lane.allocated_slots
            ):
                raise ValueError("member and lane allocation evidence disagree")
            selected_by_lane[member.proposal.expert_id] = (
                selected_by_lane.get(member.proposal.expert_id, 0) + 1
            )
        if any(
            selected_by_lane.get(value.expert_id, 0) != value.allocated_slots
            for value in self.lanes
        ):
            raise ValueError("selected members do not close the lane quotas")
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_DECISION_DOMAIN, self._unsigned_record()),
        )

    @property
    def selected_proposal_ids(self) -> tuple[str, ...]:
        return tuple(value.proposal.proposal_id for value in self.members)

    def _coverage_record(self) -> dict[str, object]:
        parents = {
            parent.value
            for value in self.members
            for parent in value.proposal.parent_ids
        }
        return {
            "selected_expert_count": len(
                {value.proposal.expert_id for value in self.members}
            ),
            "selected_role_count": len(
                {value.proposal.role_id for value in self.members}
            ),
            "selected_variation_scale_count": len(
                {value.proposal.variation_scale for value in self.members}
            ),
            "selected_parent_count": len(parents),
            "selected_structural_cell_count": len(
                {value.proposal.structural_cell for value in self.members}
            ),
        }

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "allocator": {
                "allocator_id": self.allocator_id,
                "allocator_version": self.allocator_version,
                "definition_sha256": self.allocator_definition_sha256,
            },
            "request_sha256": self.request_sha256,
            "lanes": [value.to_record() for value in self.lanes],
            "members": [value.to_record() for value in self.members],
            "coverage": self._coverage_record(),
            "outcomes_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "decision_sha256": self.decision_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is StratifiedColdStartAllocationDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class SupportProportionalStratifiedColdStartAllocator:
    """Allocate a cold-start evaluator block without comparing expert scores."""

    def _lanes(
        self,
        request: StratifiedColdStartAllocationRequest,
    ) -> dict[str, tuple[StratifiedColdStartProposal, ...]]:
        grouped: dict[str, list[StratifiedColdStartProposal]] = {}
        for proposal in request.canonical_proposals:
            grouped.setdefault(proposal.expert_id, []).append(proposal)
        return {
            expert_id: tuple(
                sorted(values, key=lambda value: (value.native_rank, value.proposal_id))
            )
            for expert_id, values in grouped.items()
        }

    @staticmethod
    def _quotas(
        lanes: dict[str, tuple[StratifiedColdStartProposal, ...]],
        slots: int,
    ) -> dict[str, int]:
        support = {key: len(value) for key, value in lanes.items()}
        quotas = {key: 0 for key in lanes}
        ordered_by_support = tuple(
            sorted(lanes, key=lambda key: (-support[key], key))
        )

        # A complete coverage floor is possible only when every lane can
        # receive one evaluation. Under a smaller budget, retain the largest
        # authenticated supports rather than invent incomparable expert scores.
        initial = (
            tuple(sorted(lanes))
            if slots >= len(lanes)
            else ordered_by_support[:slots]
        )
        for key in initial:
            quotas[key] = 1

        remaining = slots - len(initial)
        while remaining:
            eligible = tuple(
                key for key in lanes if quotas[key] < support[key]
            )
            if not eligible:  # pragma: no cover - request capacity prevents it.
                raise AssertionError("stratified quota capacity unexpectedly exhausted")
            selected = min(
                eligible,
                key=lambda key: (
                    -Fraction(support[key], quotas[key] + 1),
                    -support[key],
                    key,
                ),
            )
            quotas[selected] += 1
            remaining -= 1
        return quotas

    def select(
        self,
        request: StratifiedColdStartAllocationRequest,
    ) -> StratifiedColdStartAllocationDecision:
        if type(request) is not StratifiedColdStartAllocationRequest:
            raise TypeError(
                "request must be an exact StratifiedColdStartAllocationRequest"
            )
        request.__post_init__()
        lanes = self._lanes(request)
        quotas = self._quotas(lanes, request.evaluation_slots)
        lane_priority = tuple(
            sorted(lanes, key=lambda key: (-len(lanes[key]), key))
        )

        members: list[StratifiedColdStartAllocationMember] = []
        maximum_quota = max(quotas.values())
        for quantile_ordinal in range(maximum_quota):
            for expert_id in lane_priority:
                quota = quotas[expert_id]
                if quantile_ordinal >= quota:
                    continue
                lane = lanes[expert_id]
                target_index = (quantile_ordinal * len(lane)) // quota
                proposal = lane[target_index]
                members.append(
                    StratifiedColdStartAllocationMember(
                        proposal=proposal,
                        selection_ticket_ordinal=len(members) + 1,
                        lane_support_count=len(lane),
                        lane_allocated_slots=quota,
                        lane_quantile_ordinal=quantile_ordinal,
                        target_rank_index=target_index,
                        selected_rank_index=target_index,
                    )
                )

        if len(members) != request.evaluation_slots:
            raise AssertionError("selected member count does not close capacity")
        lane_records = tuple(
            StratifiedColdStartLaneAllocation(
                expert_id=expert_id,
                support_count=len(lanes[expert_id]),
                allocated_slots=quotas[expert_id],
            )
            for expert_id in sorted(lanes)
        )
        return StratifiedColdStartAllocationDecision(
            request_sha256=request.request_sha256,
            lanes=lane_records,
            members=tuple(members),
        )


def _base_two_low_discrepancy_rank_permutation(size: int) -> tuple[int, ...]:
    """Return a head-first, progressively space-filling rank permutation."""

    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive exact integer")
    selected: list[int] = []
    selected_set: set[int] = set()
    sequence_index = 0
    while len(selected) < size:
        value = sequence_index
        numerator = 0
        denominator = 1
        while value:
            denominator *= 2
            numerator = 2 * numerator + (value & 1)
            value >>= 1
        rank_index = (numerator * size) // denominator
        if rank_index not in selected_set:
            selected.append(rank_index)
            selected_set.add(rank_index)
        sequence_index += 1
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class SupportProportionalLowDiscrepancyStratifiedAllocator(
    SupportProportionalStratifiedColdStartAllocator
):
    """Cycle rank strata across decisions without consulting outcomes.

    Decision one reproduces the head-plus-interior behavior of the original
    cold-start policy. Later decisions consume the next contiguous block of a
    base-two low-discrepancy permutation. For a six-member lane with one slot,
    the rank schedule is 1, 4, 2, 5, 3, 6. With two slots it is
    (1, 4), (2, 5), (3, 6). The schedule is exact, deterministic, and closes
    every finite rank stratum before repeating.
    """

    def select(
        self,
        request: StratifiedColdStartAllocationRequest,
    ) -> StratifiedColdStartAllocationDecision:
        if type(request) is not StratifiedColdStartAllocationRequest:
            raise TypeError(
                "request must be an exact StratifiedColdStartAllocationRequest"
            )
        request.__post_init__()
        lanes = self._lanes(request)
        quotas = self._quotas(lanes, request.evaluation_slots)
        lane_priority = tuple(
            sorted(lanes, key=lambda key: (-len(lanes[key]), key))
        )

        members: list[StratifiedColdStartAllocationMember] = []
        maximum_quota = max(quotas.values())
        selected_by_lane: dict[
            str, tuple[tuple[int, StratifiedColdStartProposal], ...]
        ] = {}
        for expert_id, lane in lanes.items():
            quota = quotas[expert_id]
            if quota == 0:
                selected_by_lane[expert_id] = ()
                continue
            permutation = _base_two_low_discrepancy_rank_permutation(len(lane))
            offset = ((request.decision_index - 1) * quota) % len(lane)
            selected_by_lane[expert_id] = tuple(
                (
                    permutation[(offset + ordinal) % len(lane)],
                    lane[permutation[(offset + ordinal) % len(lane)]],
                )
                for ordinal in range(quota)
            )

        for rank_schedule_ordinal in range(maximum_quota):
            for expert_id in lane_priority:
                quota = quotas[expert_id]
                if rank_schedule_ordinal >= quota:
                    continue
                selected_index, proposal = selected_by_lane[expert_id][
                    rank_schedule_ordinal
                ]
                members.append(
                    StratifiedColdStartAllocationMember(
                        proposal=proposal,
                        selection_ticket_ordinal=len(members) + 1,
                        lane_support_count=len(lanes[expert_id]),
                        lane_allocated_slots=quota,
                        lane_quantile_ordinal=rank_schedule_ordinal,
                        target_rank_index=selected_index,
                        selected_rank_index=selected_index,
                    )
                )

        if len(members) != request.evaluation_slots:
            raise AssertionError("selected member count does not close capacity")
        lane_records = tuple(
            StratifiedColdStartLaneAllocation(
                expert_id=expert_id,
                support_count=len(lanes[expert_id]),
                allocated_slots=quotas[expert_id],
            )
            for expert_id in sorted(lanes)
        )
        return StratifiedColdStartAllocationDecision(
            request_sha256=request.request_sha256,
            lanes=lane_records,
            members=tuple(members),
            allocator_id=LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_ID,
            allocator_version=LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_VERSION,
            allocator_definition_sha256=(
                LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_DEFINITION_SHA256
            ),
        )


__all__ = [
    "LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_DEFINITION_SHA256",
    "LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_ID",
    "LOW_DISCREPANCY_STRATIFIED_ALLOCATOR_VERSION",
    "STRATIFIED_COLD_START_ALLOCATOR_DEFINITION_SHA256",
    "STRATIFIED_COLD_START_ALLOCATOR_ID",
    "STRATIFIED_COLD_START_ALLOCATOR_VERSION",
    "StratifiedColdStartAllocationDecision",
    "StratifiedColdStartAllocationMember",
    "StratifiedColdStartAllocationRequest",
    "StratifiedColdStartLaneAllocation",
    "StratifiedColdStartProposal",
    "SupportProportionalLowDiscrepancyStratifiedAllocator",
    "SupportProportionalStratifiedColdStartAllocator",
    "stratified_proposal_from_materialized_action",
]

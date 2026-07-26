"""Authenticated elite/explorer parent selection over generic Pareto archives.

The policy exposes two stable lanes to generation and memory planners:

``elite``
    Rotates over exact members of the archive's current Pareto front.

``explorer``
    Rotates over the best available dominated nondomination rank.  When no
    dominated history exists it uses a distinct current-front member.  An
    exact singleton front is the only case in which it reuses the elite, and
    that exception is explicit in the receipt.

Selection depends only on the task identity and the authenticated archive
state.  It contains no workload semantics and makes no causal-propensity
claim.  Complete front and ranking evidence is carried in every receipt so a
consumer can reject stale, foreign, or tampered parent choices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import OptimizerState
from agent_evolve.application.pareto_archive import ParetoCandidateRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.archive_elite import (
    ArchiveReservoirRankedCandidate,
    _canonical_json,
    _candidate_reference,
    _eligible_front_sha256,
    _eligible_ranking_sha256,
    _ranked_eligible_history,
    _reference_key,
    _reference_record,
    _reservoir_member_key,
    _validated_front,
)


POLICY_ID = "task_keyed_archive_elite_explorer_parent"
POLICY_VERSION = 1
SCHEMA_VERSION = 1
ROTATION_LAW_ID = "task_archive_lane_pool_sha256_cyclic_v1"
_MAX_ROTATION_INDEX = (1 << 63) - 1
_DEFINITION_DOMAIN = b"agent-evolve:archive-elite-explorer-parent:def:v1\x00"
_POOL_DOMAIN = b"agent-evolve:archive-elite-explorer-parent:pool:v1\x00"
_ROTATION_DOMAIN = b"agent-evolve:archive-elite-explorer-parent:rotation:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:archive-elite-explorer-parent:receipt:v1\x00"
_DEFINITION = {
    "archive_authority": "exact ParetoArchiveSnapshot and snapshot hash",
    "eligible_history": (
        "complete archive-authenticated, gate-passing unique-configuration history"
    ),
    "elite_lane": "exact current Pareto-front member",
    "explorer_lane": (
        "best available nondomination rank greater than one; otherwise distinct "
        "current-front fallback; singleton reuse only for a singleton front"
    ),
    "lane_ids": ["elite", "explorer"],
    "rotation_law_id": ROTATION_LAW_ID,
    "benchmark_branching": False,
    "causal_propensity_claim": False,
    "stale_or_foreign_candidate_admission": False,
}
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_DOMAIN + _canonical_json(_DEFINITION)
).hexdigest()


class EliteExplorerLaneId(str, Enum):
    """Stable lane identifiers for variation and balanced-memory plans."""

    ELITE = "elite"
    EXPLORER = "explorer"


class EliteExplorerLaneSource(str, Enum):
    """Authenticated source from which a lane parent was selected."""

    CURRENT_PARETO_FRONT = "current_pareto_front"
    BEST_DOMINATED_RANK = "best_dominated_rank"
    DISTINCT_FRONT_FALLBACK = "distinct_front_fallback"
    SINGLETON_REUSE_FALLBACK = "singleton_reuse_fallback"


class EliteExplorerFallbackReason(str, Enum):
    """Why the explorer lane departed from its preferred dominated source."""

    NONE = "none"
    NO_DOMINATED_HISTORY = "no_dominated_history"
    SINGLETON_FRONT = "singleton_front"


def _selection_pool_sha256(
    lane_id: EliteExplorerLaneId,
    pool: tuple[ParetoCandidateRef, ...],
) -> str:
    if type(lane_id) is not EliteExplorerLaneId:
        raise TypeError("lane_id must be an EliteExplorerLaneId")
    if type(pool) is not tuple or not pool:
        raise ValueError("selection_pool must be a non-empty exact tuple")
    for reference in pool:
        _reference_record(reference)
    return hashlib.sha256(
        _POOL_DOMAIN
        + lane_id.value.encode("ascii")
        + b"\x00"
        + _canonical_json([_reference_record(reference) for reference in pool])
    ).hexdigest()


def _rotation_anchor(
    *,
    task_sha256: str,
    archive_snapshot_hash: str,
    eligible_front_sha256: str,
    eligible_ranking_sha256: str,
    lane_id: EliteExplorerLaneId,
    source: EliteExplorerLaneSource,
    nondomination_rank: int,
    fallback_reason: EliteExplorerFallbackReason,
    selection_pool_sha256: str,
    cardinality: int,
) -> int:
    require_sha256(task_sha256, "task_sha256")
    require_sha256(archive_snapshot_hash, "archive_snapshot_hash")
    require_sha256(eligible_front_sha256, "eligible_front_sha256")
    require_sha256(eligible_ranking_sha256, "eligible_ranking_sha256")
    require_sha256(selection_pool_sha256, "selection_pool_sha256")
    if type(lane_id) is not EliteExplorerLaneId:
        raise TypeError("lane_id must be an EliteExplorerLaneId")
    if type(source) is not EliteExplorerLaneSource:
        raise TypeError("source must be an EliteExplorerLaneSource")
    if type(fallback_reason) is not EliteExplorerFallbackReason:
        raise TypeError("fallback_reason must be an EliteExplorerFallbackReason")
    if type(nondomination_rank) is not int or nondomination_rank <= 0:
        raise ValueError("nondomination_rank must be a positive exact integer")
    if type(cardinality) is not int or cardinality <= 0:
        raise ValueError("selection-pool cardinality must be positive")
    digest = hashlib.sha256(
        _ROTATION_DOMAIN
        + bytes.fromhex(task_sha256)
        + bytes.fromhex(archive_snapshot_hash)
        + bytes.fromhex(eligible_front_sha256)
        + bytes.fromhex(eligible_ranking_sha256)
        + lane_id.value.encode("ascii")
        + b"\x00"
        + source.value.encode("ascii")
        + b"\x00"
        + nondomination_rank.to_bytes(8, "big", signed=False)
        + fallback_reason.value.encode("ascii")
        + b"\x00"
        + bytes.fromhex(selection_pool_sha256)
    ).digest()
    return int.from_bytes(digest, "big", signed=False) % cardinality


@dataclass(frozen=True, slots=True)
class EliteExplorerLaneReceipt:
    """One stable lane's source, rotation, and selected parent identity."""

    lane_id: EliteExplorerLaneId
    source: EliteExplorerLaneSource
    nondomination_rank: int
    fallback_reason: EliteExplorerFallbackReason
    selection_pool: tuple[ParetoCandidateRef, ...]
    rotation_anchor: int
    selected_ordinal: int
    selected_parent: ParetoCandidateRef
    selection_pool_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.lane_id) is not EliteExplorerLaneId:
            raise TypeError("lane_id must be an EliteExplorerLaneId")
        if type(self.source) is not EliteExplorerLaneSource:
            raise TypeError("source must be an EliteExplorerLaneSource")
        if type(self.fallback_reason) is not EliteExplorerFallbackReason:
            raise TypeError("fallback_reason must be an EliteExplorerFallbackReason")
        if type(self.nondomination_rank) is not int or self.nondomination_rank <= 0:
            raise ValueError("nondomination_rank must be a positive exact integer")
        if type(self.selection_pool) is not tuple or not self.selection_pool:
            raise ValueError("selection_pool must be a non-empty exact tuple")
        for reference in self.selection_pool:
            _reference_record(reference)
        if len({value.candidate_id for value in self.selection_pool}) != len(
            self.selection_pool
        ):
            raise ValueError("selection_pool contains duplicate candidate IDs")
        if len({value.configuration_hash for value in self.selection_pool}) != len(
            self.selection_pool
        ):
            raise ValueError("selection_pool contains duplicate configurations")
        computed_pool_sha256 = _selection_pool_sha256(
            self.lane_id,
            self.selection_pool,
        )
        if self.selection_pool_sha256 not in ("", computed_pool_sha256):
            raise ValueError("selection_pool_sha256 does not identify selection_pool")
        object.__setattr__(self, "selection_pool_sha256", computed_pool_sha256)
        if type(self.rotation_anchor) is not int or not (
            0 <= self.rotation_anchor < len(self.selection_pool)
        ):
            raise ValueError("rotation_anchor must lie within selection_pool")
        if type(self.selected_ordinal) is not int or not (
            0 <= self.selected_ordinal < len(self.selection_pool)
        ):
            raise ValueError("selected_ordinal must lie within selection_pool")
        _reference_record(self.selected_parent)
        if self.selected_parent != self.selection_pool[self.selected_ordinal]:
            raise ValueError("selected_parent does not match selected_ordinal")

    def revalidate(self) -> None:
        if type(self) is not EliteExplorerLaneReceipt:
            raise TypeError("lane must be an exact EliteExplorerLaneReceipt")
        EliteExplorerLaneReceipt.__post_init__(self)

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "lane_id": self.lane_id.value,
            "source": self.source.value,
            "nondomination_rank": self.nondomination_rank,
            "fallback_reason": self.fallback_reason.value,
            "selection_pool": [
                _reference_record(reference) for reference in self.selection_pool
            ],
            "selection_pool_sha256": self.selection_pool_sha256,
            "rotation_anchor": self.rotation_anchor,
            "selected_ordinal": self.selected_ordinal,
            "selected_parent": _reference_record(self.selected_parent),
        }


def _validate_complete_authority(
    eligible_front: tuple[ParetoCandidateRef, ...],
    eligible_ranking: tuple[ArchiveReservoirRankedCandidate, ...],
) -> None:
    if type(eligible_front) is not tuple or not eligible_front:
        raise ValueError("eligible_front must be a non-empty exact tuple")
    for reference in eligible_front:
        _reference_record(reference)
    if eligible_front != tuple(sorted(eligible_front, key=_reference_key)):
        raise ValueError("eligible_front must preserve canonical archive order")
    if len({value.candidate_id for value in eligible_front}) != len(eligible_front):
        raise ValueError("eligible_front contains duplicate candidate IDs")
    if len({value.configuration_hash for value in eligible_front}) != len(
        eligible_front
    ):
        raise ValueError("eligible_front contains duplicate configurations")

    if type(eligible_ranking) is not tuple or not eligible_ranking:
        raise ValueError("eligible_ranking must be a non-empty exact tuple")
    if any(
        type(member) is not ArchiveReservoirRankedCandidate
        for member in eligible_ranking
    ):
        raise TypeError("eligible_ranking must contain exact ranked candidates")
    for member in eligible_ranking:
        ArchiveReservoirRankedCandidate.__post_init__(member)
    if eligible_ranking != tuple(sorted(eligible_ranking, key=_reservoir_member_key)):
        raise ValueError("eligible_ranking does not preserve policy order")
    ranking_references = tuple(member.reference for member in eligible_ranking)
    if len({value.candidate_id for value in ranking_references}) != len(
        ranking_references
    ):
        raise ValueError("eligible_ranking contains duplicate candidate IDs")
    if len({value.configuration_hash for value in ranking_references}) != len(
        ranking_references
    ):
        raise ValueError("eligible_ranking contains duplicate configurations")
    observed_ranks = {member.nondomination_rank for member in eligible_ranking}
    if observed_ranks != set(range(1, max(observed_ranks) + 1)):
        raise ValueError("eligible_ranking has non-contiguous nondomination ranks")

    rank_one_references = {
        member.reference
        for member in eligible_ranking
        if member.nondomination_rank == 1
    }
    if not set(eligible_front).issubset(rank_one_references):
        raise ValueError("eligible_front is not contained in ranking rank one")


def _expected_lane_specs(
    *,
    task_sha256: str,
    archive_snapshot_hash: str,
    eligible_front: tuple[ParetoCandidateRef, ...],
    eligible_ranking: tuple[ArchiveReservoirRankedCandidate, ...],
    rotation_index: int,
) -> tuple[EliteExplorerLaneReceipt, EliteExplorerLaneReceipt]:
    front_sha256 = _eligible_front_sha256(eligible_front)
    ranking_sha256 = _eligible_ranking_sha256(eligible_ranking)

    elite_pool = eligible_front
    elite_pool_sha256 = _selection_pool_sha256(EliteExplorerLaneId.ELITE, elite_pool)
    elite_anchor = _rotation_anchor(
        task_sha256=task_sha256,
        archive_snapshot_hash=archive_snapshot_hash,
        eligible_front_sha256=front_sha256,
        eligible_ranking_sha256=ranking_sha256,
        lane_id=EliteExplorerLaneId.ELITE,
        source=EliteExplorerLaneSource.CURRENT_PARETO_FRONT,
        nondomination_rank=1,
        fallback_reason=EliteExplorerFallbackReason.NONE,
        selection_pool_sha256=elite_pool_sha256,
        cardinality=len(elite_pool),
    )
    elite_ordinal = (elite_anchor + rotation_index) % len(elite_pool)
    elite = EliteExplorerLaneReceipt(
        lane_id=EliteExplorerLaneId.ELITE,
        source=EliteExplorerLaneSource.CURRENT_PARETO_FRONT,
        nondomination_rank=1,
        fallback_reason=EliteExplorerFallbackReason.NONE,
        selection_pool=elite_pool,
        rotation_anchor=elite_anchor,
        selected_ordinal=elite_ordinal,
        selected_parent=elite_pool[elite_ordinal],
    )

    dominated_ranks = tuple(
        sorted(
            {
                member.nondomination_rank
                for member in eligible_ranking
                if member.nondomination_rank > 1
            }
        )
    )
    if dominated_ranks:
        explorer_rank = dominated_ranks[0]
        explorer_source = EliteExplorerLaneSource.BEST_DOMINATED_RANK
        fallback_reason = EliteExplorerFallbackReason.NONE
        explorer_pool = tuple(
            member.reference
            for member in eligible_ranking
            if member.nondomination_rank == explorer_rank
        )
    elif len(eligible_front) > 1:
        explorer_rank = 1
        explorer_source = EliteExplorerLaneSource.DISTINCT_FRONT_FALLBACK
        fallback_reason = EliteExplorerFallbackReason.NO_DOMINATED_HISTORY
        explorer_pool = tuple(
            reference
            for reference in eligible_front
            if reference != elite.selected_parent
        )
    else:
        explorer_rank = 1
        explorer_source = EliteExplorerLaneSource.SINGLETON_REUSE_FALLBACK
        fallback_reason = EliteExplorerFallbackReason.SINGLETON_FRONT
        explorer_pool = eligible_front

    explorer_pool_sha256 = _selection_pool_sha256(
        EliteExplorerLaneId.EXPLORER,
        explorer_pool,
    )
    explorer_anchor = _rotation_anchor(
        task_sha256=task_sha256,
        archive_snapshot_hash=archive_snapshot_hash,
        eligible_front_sha256=front_sha256,
        eligible_ranking_sha256=ranking_sha256,
        lane_id=EliteExplorerLaneId.EXPLORER,
        source=explorer_source,
        nondomination_rank=explorer_rank,
        fallback_reason=fallback_reason,
        selection_pool_sha256=explorer_pool_sha256,
        cardinality=len(explorer_pool),
    )
    explorer_ordinal = (explorer_anchor + rotation_index) % len(explorer_pool)
    explorer = EliteExplorerLaneReceipt(
        lane_id=EliteExplorerLaneId.EXPLORER,
        source=explorer_source,
        nondomination_rank=explorer_rank,
        fallback_reason=fallback_reason,
        selection_pool=explorer_pool,
        rotation_anchor=explorer_anchor,
        selected_ordinal=explorer_ordinal,
        selected_parent=explorer_pool[explorer_ordinal],
    )
    return elite, explorer


@dataclass(frozen=True, slots=True)
class ArchiveEliteExplorerParentSelectionReceipt:
    """Replayable binding of two named parents to complete archive authority."""

    task_sha256: str
    optimizer_generation: int
    archive_snapshot_hash: str
    eligible_front: tuple[ParetoCandidateRef, ...]
    eligible_ranking: tuple[ArchiveReservoirRankedCandidate, ...]
    rotation_index: int
    lanes: tuple[EliteExplorerLaneReceipt, EliteExplorerLaneReceipt]
    schema_version: int = field(init=False, default=SCHEMA_VERSION)
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    policy_definition_sha256: str = field(
        init=False,
        default=POLICY_DEFINITION_SHA256,
    )
    rotation_law_id: str = field(init=False, default=ROTATION_LAW_ID)
    eligible_front_sha256: str = field(init=False, default="")
    eligible_ranking_sha256: str = field(init=False, default="")
    receipt_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.policy_definition_sha256 != POLICY_DEFINITION_SHA256
            or self.rotation_law_id != ROTATION_LAW_ID
        ):
            raise ValueError("elite/explorer receipt uses a foreign policy identity")
        require_sha256(self.task_sha256, "task_sha256")
        require_sha256(self.archive_snapshot_hash, "archive_snapshot_hash")
        if type(self.optimizer_generation) is not int or self.optimizer_generation < 0:
            raise ValueError("optimizer_generation must be non-negative")
        _validate_complete_authority(self.eligible_front, self.eligible_ranking)
        computed_front_sha256 = _eligible_front_sha256(self.eligible_front)
        if self.eligible_front_sha256 not in ("", computed_front_sha256):
            raise ValueError("eligible_front_sha256 does not identify eligible_front")
        object.__setattr__(self, "eligible_front_sha256", computed_front_sha256)
        computed_ranking_sha256 = _eligible_ranking_sha256(self.eligible_ranking)
        if self.eligible_ranking_sha256 not in ("", computed_ranking_sha256):
            raise ValueError(
                "eligible_ranking_sha256 does not identify eligible_ranking"
            )
        object.__setattr__(self, "eligible_ranking_sha256", computed_ranking_sha256)
        if type(self.rotation_index) is not int or not (
            0 <= self.rotation_index <= _MAX_ROTATION_INDEX
        ):
            raise ValueError("rotation_index must be an exact non-negative int63")
        if type(self.lanes) is not tuple or len(self.lanes) != 2:
            raise ValueError("lanes must contain the exact elite/explorer pair")
        if any(type(lane) is not EliteExplorerLaneReceipt for lane in self.lanes):
            raise TypeError("lanes must contain exact EliteExplorerLaneReceipt values")
        for lane in self.lanes:
            lane.revalidate()
        if tuple(lane.lane_id for lane in self.lanes) != (
            EliteExplorerLaneId.ELITE,
            EliteExplorerLaneId.EXPLORER,
        ):
            raise ValueError("lanes must preserve stable elite/explorer order")

        expected_lanes = _expected_lane_specs(
            task_sha256=self.task_sha256,
            archive_snapshot_hash=self.archive_snapshot_hash,
            eligible_front=self.eligible_front,
            eligible_ranking=self.eligible_ranking,
            rotation_index=self.rotation_index,
        )
        if self.lanes != expected_lanes:
            raise ValueError("lanes do not replay the elite/explorer rotation law")
        if (
            self.lanes[0].selected_parent == self.lanes[1].selected_parent
            and self.lanes[1].source
            is not EliteExplorerLaneSource.SINGLETON_REUSE_FALLBACK
        ):
            raise ValueError("elite and explorer parents must be distinct")

        computed_receipt_sha256 = hashlib.sha256(
            _RECEIPT_DOMAIN + _canonical_json(self._record_without_hash())
        ).hexdigest()
        if self.receipt_sha256 not in ("", computed_receipt_sha256):
            raise ValueError("receipt_sha256 does not verify")
        object.__setattr__(self, "receipt_sha256", computed_receipt_sha256)

    def _record_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "rotation_law_id": self.rotation_law_id,
            "task_sha256": self.task_sha256,
            "optimizer_generation": self.optimizer_generation,
            "archive_snapshot_hash": self.archive_snapshot_hash,
            "eligible_front": [
                _reference_record(reference) for reference in self.eligible_front
            ],
            "eligible_front_sha256": self.eligible_front_sha256,
            "eligible_ranking": [
                member.to_trace_record() for member in self.eligible_ranking
            ],
            "eligible_ranking_sha256": self.eligible_ranking_sha256,
            "rotation_index": self.rotation_index,
            "lanes": [lane.to_trace_record() for lane in self.lanes],
        }

    def revalidate(self) -> None:
        if type(self) is not ArchiveEliteExplorerParentSelectionReceipt:
            raise TypeError(
                "receipt must be an exact ArchiveEliteExplorerParentSelectionReceipt"
            )
        ArchiveEliteExplorerParentSelectionReceipt.__post_init__(self)

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {**self._record_without_hash(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ArchiveEliteExplorerParentSelection:
    """Exactly two stable-lane parents and their authenticated receipt."""

    parents: tuple[EvolutionCandidate, EvolutionCandidate]
    receipt: ArchiveEliteExplorerParentSelectionReceipt

    def __post_init__(self) -> None:
        if type(self.parents) is not tuple or len(self.parents) != 2:
            raise ValueError("parents must contain the exact elite/explorer pair")
        if any(type(parent) is not EvolutionCandidate for parent in self.parents):
            raise TypeError("parents must contain exact EvolutionCandidate values")
        for parent in self.parents:
            EvolutionCandidate.__post_init__(parent)
        if type(self.receipt) is not ArchiveEliteExplorerParentSelectionReceipt:
            raise TypeError("receipt must be an exact elite/explorer receipt")
        self.receipt.revalidate()
        if tuple(_candidate_reference(parent) for parent in self.parents) != tuple(
            lane.selected_parent for lane in self.receipt.lanes
        ):
            raise ValueError("selected parent values differ from their lane receipts")
        if (
            self.parents[0].candidate_id == self.parents[1].candidate_id
            and self.receipt.lanes[1].source
            is not EliteExplorerLaneSource.SINGLETON_REUSE_FALLBACK
        ):
            raise ValueError(
                "parent occurrences must be distinct outside singleton reuse"
            )

    @property
    def elite(self) -> EvolutionCandidate:
        """Return the parent assigned to the stable elite lane."""

        return self.parents[0]

    @property
    def explorer(self) -> EvolutionCandidate:
        """Return the parent assigned to the stable explorer lane."""

        return self.parents[1]

    def parent_for(self, lane_id: EliteExplorerLaneId) -> EvolutionCandidate:
        """Resolve a parent by stable lane identity."""

        if type(lane_id) is not EliteExplorerLaneId:
            raise TypeError("lane_id must be an EliteExplorerLaneId")
        return self.parents[0 if lane_id is EliteExplorerLaneId.ELITE else 1]

    def revalidate(self) -> None:
        if type(self) is not ArchiveEliteExplorerParentSelection:
            raise TypeError(
                "selection must be an exact ArchiveEliteExplorerParentSelection"
            )
        ArchiveEliteExplorerParentSelection.__post_init__(self)


def validate_archive_elite_explorer_parent_selection(
    state: OptimizerState,
    selection: ArchiveEliteExplorerParentSelection,
) -> None:
    """Reject stale, foreign, incompletely ranked, or tampered lane choices."""

    front_candidates, front_references = _validated_front(state)
    ranked_candidates, ranked_members = _ranked_eligible_history(state)
    if type(selection) is not ArchiveEliteExplorerParentSelection:
        raise TypeError(
            "selection must be an exact ArchiveEliteExplorerParentSelection"
        )
    selection.revalidate()
    receipt = selection.receipt
    if receipt.optimizer_generation != state.generation:
        raise ValueError("elite/explorer selection is stale for this generation")
    if receipt.archive_snapshot_hash != state.archive_snapshot_hash:
        raise ValueError("elite/explorer selection is stale for this snapshot")
    if receipt.eligible_front != front_references:
        raise ValueError("receipt does not bind the complete current archive front")
    if receipt.eligible_ranking != ranked_members:
        raise ValueError("receipt does not bind the complete current eligible ranking")

    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in (*front_candidates, *ranked_candidates)
    }
    expected_parents = tuple(
        candidates_by_id[lane.selected_parent.candidate_id] for lane in receipt.lanes
    )
    if selection.parents != expected_parents:
        raise ValueError("elite/explorer parents are foreign to current history")


class ArchiveEliteExplorerParentSelector(Protocol):
    """Inverted planner seam for two stable archive parent lanes."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        expected_archive_snapshot_hash: str,
        rotation_index: int = 0,
    ) -> ArchiveEliteExplorerParentSelection: ...

    def to_record(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class TaskKeyedArchiveEliteExplorerParentPolicy:
    """Select explicit elite and explorer lanes without workload semantics."""

    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(
        init=False,
        default=POLICY_DEFINITION_SHA256,
    )

    def _validate_identity(self) -> None:
        if type(self) is not TaskKeyedArchiveEliteExplorerParentPolicy:
            raise TypeError(
                "policy must be an exact TaskKeyedArchiveEliteExplorerParentPolicy"
            )
        if (
            self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.definition_sha256 != POLICY_DEFINITION_SHA256
        ):
            raise ValueError("elite/explorer selector uses a foreign identity")

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        expected_archive_snapshot_hash: str,
        rotation_index: int = 0,
    ) -> ArchiveEliteExplorerParentSelection:
        self._validate_identity()
        require_sha256(task_sha256, "task_sha256")
        require_sha256(
            expected_archive_snapshot_hash,
            "expected_archive_snapshot_hash",
        )
        if type(rotation_index) is not int or not (
            0 <= rotation_index <= _MAX_ROTATION_INDEX
        ):
            raise ValueError("rotation_index must be an exact non-negative int63")
        front_candidates, front_references = _validated_front(state)
        ranked_candidates, ranked_members = _ranked_eligible_history(state)
        if expected_archive_snapshot_hash != state.archive_snapshot_hash:
            raise ValueError("expected archive snapshot is stale")

        lanes = _expected_lane_specs(
            task_sha256=task_sha256,
            archive_snapshot_hash=state.archive_snapshot_hash,
            eligible_front=front_references,
            eligible_ranking=ranked_members,
            rotation_index=rotation_index,
        )
        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in (*front_candidates, *ranked_candidates)
        }
        receipt = ArchiveEliteExplorerParentSelectionReceipt(
            task_sha256=task_sha256,
            optimizer_generation=state.generation,
            archive_snapshot_hash=state.archive_snapshot_hash,
            eligible_front=front_references,
            eligible_ranking=ranked_members,
            rotation_index=rotation_index,
            lanes=lanes,
        )
        selection = ArchiveEliteExplorerParentSelection(
            parents=tuple(
                candidates_by_id[lane.selected_parent.candidate_id] for lane in lanes
            ),
            receipt=receipt,
        )
        validate_archive_elite_explorer_parent_selection(state, selection)
        return selection

    def to_record(self) -> dict[str, object]:
        self._validate_identity()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "lane_ids": [lane.value for lane in EliteExplorerLaneId],
            "rotation_law_id": ROTATION_LAW_ID,
        }


__all__ = [
    "ArchiveEliteExplorerParentSelection",
    "ArchiveEliteExplorerParentSelectionReceipt",
    "ArchiveEliteExplorerParentSelector",
    "EliteExplorerFallbackReason",
    "EliteExplorerLaneId",
    "EliteExplorerLaneReceipt",
    "EliteExplorerLaneSource",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "ROTATION_LAW_ID",
    "TaskKeyedArchiveEliteExplorerParentPolicy",
    "validate_archive_elite_explorer_parent_selection",
]

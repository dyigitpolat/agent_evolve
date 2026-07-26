"""Precommitted balanced complete-subset blocks for causal memory trials.

The planner in this module is deliberately provider- and workload-neutral.  It
binds an immutable :class:`MemoryScoreSnapshot` to an ordered collection of
stable experimental units *before* any provider request is made.  Every full
block visits each member of the canonical ``C(n, k)`` insight-subset catalog
exactly once.  A trailing partial block samples without replacement from that
same catalog.

Randomness is kept outside the policy.  Callers supply exact integer ranks:

* one permutation rank in ``[0, C(n,k)!)`` for every complete block;
* for a remainder of size ``r``, one combination rank in
  ``[0, C(C(n,k),r))`` and one permutation rank in ``[0, r!)``.

The planner only maps those ranks to a frozen schedule.  Consequently the
randomization law is exact, easy to replay, and independent of provider
latency, completion order, evaluator output, or benchmark semantics.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import overload

from agent_evolve.domain.insight import InsightRef
from agent_evolve.policies.memory.randomized_subset import (
    InsightSelectionDecision,
    InsightSelectionMode,
)
from agent_evolve.policies.memory.staged_causal import (
    DeterministicMemoryControlPolicy,
    MemoryScoreSnapshot,
    insight_selection_decision_sha256,
)


BALANCED_SUBSET_BLOCK_POLICY_ID = "balanced_complete_k_subset_blocks"
BALANCED_SUBSET_BLOCK_POLICY_VERSION = 1
_POLICY_DEFINITION = (
    b"agent-evolve:balanced-complete-k-subset-blocks:v1:"
    b"canonical-lexicographic-catalog;externally-ranked-full-permutations;"
    b"externally-ranked-no-replacement-remainder;immutable-snapshot-and-units;"
    b"uniform-control-materialization;treated-and-control-support-preflight"
)
BALANCED_SUBSET_BLOCK_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    _POLICY_DEFINITION
).hexdigest()
_PLAN_RECEIPT_DOMAIN = b"agent-evolve:balanced-subset-block-plan:v1\x00"
_SAFE_UNIT_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash_record(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _reference_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


def _require_unit_token(value: str, name: str) -> None:
    if type(value) is not str or _SAFE_UNIT_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded durable identifier")


def _require_rank(rank: int, *, stop: int, name: str) -> None:
    if type(rank) is not int or rank < 0 or rank >= stop:
        raise ValueError(f"{name} must lie in [0, {stop})")


def _unrank_combination(
    values: tuple[int, ...],
    subset_size: int,
    rank: int,
    *,
    name: str,
) -> tuple[int, ...]:
    """Map one exact lexicographic combination rank to its members."""

    combination_count = math.comb(len(values), subset_size)
    _require_rank(rank, stop=combination_count, name=name)
    selected_indices: list[int] = []
    remaining_rank = rank
    start = 0
    for position in range(subset_size):
        remaining_positions = subset_size - position - 1
        for index in range(start, len(values)):
            suffix_count = math.comb(
                len(values) - index - 1,
                remaining_positions,
            )
            if remaining_rank < suffix_count:
                selected_indices.append(index)
                start = index + 1
                break
            remaining_rank -= suffix_count
    return tuple(values[index] for index in selected_indices)


def _unrank_permutation(
    values: tuple[int, ...],
    rank: int,
    *,
    name: str,
) -> tuple[int, ...]:
    """Map one exact Lehmer rank to a permutation of ``values``."""

    permutation_count = math.factorial(len(values))
    _require_rank(rank, stop=permutation_count, name=name)
    remaining = list(values)
    result: list[int] = []
    remaining_rank = rank
    for slots in range(len(values), 0, -1):
        block_size = math.factorial(slots - 1)
        index, remaining_rank = divmod(remaining_rank, block_size)
        result.append(remaining.pop(index))
    return tuple(result)


def canonical_memory_subset_catalog(
    snapshot: MemoryScoreSnapshot,
    subset_size: int,
) -> tuple[tuple[InsightRef, ...], ...]:
    """Return the canonical lexicographic ``C(n,k)`` catalog for a snapshot."""

    if not isinstance(snapshot, MemoryScoreSnapshot):
        raise TypeError("snapshot must be a MemoryScoreSnapshot")
    references = tuple(entry.reference for entry in snapshot.entries)
    if type(subset_size) is not int:
        raise TypeError("subset_size must be an exact integer")
    if subset_size <= 0 or subset_size >= len(references):
        raise ValueError(
            "subset_size must lie strictly between zero and the snapshot size "
            "so treated and control support are possible"
        )
    return tuple(itertools.combinations(references, subset_size))


@dataclass(frozen=True, slots=True)
class StableMemoryAssignmentUnit:
    """One prospectively named provider-call slot in the frozen schedule."""

    unit_key: str
    generation: int
    lane_id: str

    def __post_init__(self) -> None:
        _require_unit_token(self.unit_key, "unit_key")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative exact integer")
        _require_unit_token(self.lane_id, "lane_id")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "unit_key": self.unit_key,
            "generation": self.generation,
            "lane_id": self.lane_id,
        }


@dataclass(frozen=True, slots=True)
class MemorySubsetSupport:
    """Realized selected/unselected support for one exact insight version."""

    reference: InsightRef
    treated_count: int
    control_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference, InsightRef):
            raise TypeError("reference must be an InsightRef")
        for name in ("treated_count", "control_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")

    @property
    def has_overlap(self) -> bool:
        """Whether both treated and control observations are scheduled."""

        return self.treated_count > 0 and self.control_count > 0

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "reference": _reference_record(self.reference),
            "treated_count": self.treated_count,
            "control_count": self.control_count,
            "has_overlap": self.has_overlap,
        }


@dataclass(frozen=True, slots=True)
class BalancedSubsetBlockAssignment:
    """One stable unit's exact catalog assignment and uniform decision."""

    schedule_position: int
    unit: StableMemoryAssignmentUnit
    block_index: int
    block_position: int
    is_remainder: bool
    subset_rank: int
    decision: InsightSelectionDecision

    def __post_init__(self) -> None:
        for name in (
            "schedule_position",
            "block_index",
            "block_position",
            "subset_rank",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if type(self.unit) is not StableMemoryAssignmentUnit:
            raise TypeError("unit must be an exact StableMemoryAssignmentUnit")
        self.unit.__post_init__()
        if type(self.is_remainder) is not bool:
            raise TypeError("is_remainder must be a bool")
        if not isinstance(self.decision, InsightSelectionDecision):
            raise TypeError("decision must be an InsightSelectionDecision")
        if self.decision.mode is not InsightSelectionMode.EXPLORE_UNIFORM:
            raise ValueError("balanced block assignments must use uniform controls")

    @property
    def selection_decision_sha256(self) -> str:
        return insight_selection_decision_sha256(self.decision)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        probability = self.decision.selected_subset_probability
        return {
            "schedule_position": self.schedule_position,
            "unit": self.unit.to_record(),
            "block_index": self.block_index,
            "block_position": self.block_position,
            "is_remainder": self.is_remainder,
            "subset_rank": self.subset_rank,
            "selected": [
                _reference_record(reference) for reference in self.decision.selected
            ],
            "selected_subset_probability": [
                probability.numerator,
                probability.denominator,
            ],
            "selection_decision_sha256": self.selection_decision_sha256,
        }


def _validate_ordered_units(
    ordered_units: tuple[StableMemoryAssignmentUnit, ...],
) -> None:
    if type(ordered_units) is not tuple or not ordered_units:
        raise ValueError("ordered_units must be a non-empty exact tuple")
    if any(type(unit) is not StableMemoryAssignmentUnit for unit in ordered_units):
        raise TypeError(
            "ordered_units must contain exact StableMemoryAssignmentUnit values"
        )
    for unit in ordered_units:
        unit.__post_init__()
    unit_keys = tuple(unit.unit_key for unit in ordered_units)
    if len(set(unit_keys)) != len(unit_keys):
        raise ValueError("ordered_units cannot repeat a stable unit_key")
    generation_lanes = tuple((unit.generation, unit.lane_id) for unit in ordered_units)
    if len(set(generation_lanes)) != len(generation_lanes):
        raise ValueError("ordered_units cannot repeat a generation/lane pair")


def _scheduled_subset_ranks(
    *,
    unit_count: int,
    catalog_size: int,
    full_block_permutation_ranks: tuple[int, ...],
    remainder_selection_rank: int | None,
    remainder_permutation_rank: int | None,
) -> tuple[int, ...]:
    full_block_count, remainder_size = divmod(unit_count, catalog_size)
    if type(full_block_permutation_ranks) is not tuple:
        raise TypeError("full_block_permutation_ranks must be an exact tuple")
    if len(full_block_permutation_ranks) != full_block_count:
        raise ValueError(
            "full_block_permutation_ranks must provide exactly one rank per "
            "complete catalog block"
        )

    catalog_ranks = tuple(range(catalog_size))
    schedule: list[int] = []
    for block_index, permutation_rank in enumerate(full_block_permutation_ranks):
        schedule.extend(
            _unrank_permutation(
                catalog_ranks,
                permutation_rank,
                name=f"full_block_permutation_ranks[{block_index}]",
            )
        )

    if remainder_size == 0:
        if remainder_selection_rank is not None:
            raise ValueError(
                "remainder_selection_rank must be None when there is no remainder"
            )
        if remainder_permutation_rank is not None:
            raise ValueError(
                "remainder_permutation_rank must be None when there is no remainder"
            )
        return tuple(schedule)

    if remainder_selection_rank is None or remainder_permutation_rank is None:
        raise ValueError(
            "a partial block requires exact remainder selection and permutation ranks"
        )
    selected = _unrank_combination(
        catalog_ranks,
        remainder_size,
        remainder_selection_rank,
        name="remainder_selection_rank",
    )
    schedule.extend(
        _unrank_permutation(
            selected,
            remainder_permutation_rank,
            name="remainder_permutation_rank",
        )
    )
    return tuple(schedule)


def _support_for(
    references: tuple[InsightRef, ...],
    assignments: tuple[BalancedSubsetBlockAssignment, ...],
) -> tuple[MemorySubsetSupport, ...]:
    return tuple(
        MemorySubsetSupport(
            reference=reference,
            treated_count=sum(
                reference in assignment.decision.selected for assignment in assignments
            ),
            control_count=sum(
                reference not in assignment.decision.selected
                for assignment in assignments
            ),
        )
        for reference in references
    )


@dataclass(frozen=True, slots=True, eq=False)
class BalancedSubsetBlockPlan:
    """Authenticated pre-provider schedule and its replay receipt."""

    snapshot: MemoryScoreSnapshot
    ordered_units: tuple[StableMemoryAssignmentUnit, ...]
    subset_size: int
    catalog: tuple[tuple[InsightRef, ...], ...]
    full_block_permutation_ranks: tuple[int, ...]
    remainder_selection_rank: int | None
    remainder_permutation_rank: int | None
    assignments: tuple[BalancedSubsetBlockAssignment, ...]
    support: tuple[MemorySubsetSupport, ...]
    policy_id: str = BALANCED_SUBSET_BLOCK_POLICY_ID
    policy_version: int = BALANCED_SUBSET_BLOCK_POLICY_VERSION
    policy_definition_sha256: str = BALANCED_SUBSET_BLOCK_POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MemoryScoreSnapshot):
            raise TypeError("snapshot must be a MemoryScoreSnapshot")
        _validate_ordered_units(self.ordered_units)
        expected_catalog = canonical_memory_subset_catalog(
            self.snapshot,
            self.subset_size,
        )
        if self.catalog != expected_catalog:
            raise ValueError("catalog is not the canonical C(n,k) subset catalog")
        if self.policy_id != BALANCED_SUBSET_BLOCK_POLICY_ID:
            raise ValueError("plan names an unsupported policy_id")
        if self.policy_version != BALANCED_SUBSET_BLOCK_POLICY_VERSION:
            raise ValueError("plan names an unsupported policy_version")
        if (
            self.policy_definition_sha256
            != BALANCED_SUBSET_BLOCK_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("balanced subset-block policy definition changed")

        scheduled_ranks = _scheduled_subset_ranks(
            unit_count=len(self.ordered_units),
            catalog_size=len(self.catalog),
            full_block_permutation_ranks=self.full_block_permutation_ranks,
            remainder_selection_rank=self.remainder_selection_rank,
            remainder_permutation_rank=self.remainder_permutation_rank,
        )
        if type(self.assignments) is not tuple or len(self.assignments) != len(
            self.ordered_units
        ):
            raise ValueError("assignments must cover every ordered unit exactly once")
        if any(
            type(value) is not BalancedSubsetBlockAssignment
            for value in self.assignments
        ):
            raise TypeError(
                "assignments must contain exact BalancedSubsetBlockAssignment values"
            )

        controls = DeterministicMemoryControlPolicy()
        catalog_size = len(self.catalog)
        full_block_count = len(self.ordered_units) // catalog_size
        for position, (unit, subset_rank, assignment) in enumerate(
            zip(
                self.ordered_units,
                scheduled_ranks,
                self.assignments,
                strict=True,
            )
        ):
            assignment.__post_init__()
            expected_decision = controls.uniform(
                snapshot=self.snapshot,
                subset_size=self.subset_size,
                subset_rank=subset_rank,
            )
            expected_block_index, expected_block_position = divmod(
                position,
                catalog_size,
            )
            expected_remainder = expected_block_index >= full_block_count
            if (
                assignment.schedule_position != position
                or assignment.unit != unit
                or assignment.block_index != expected_block_index
                or assignment.block_position != expected_block_position
                or assignment.is_remainder is not expected_remainder
                or assignment.subset_rank != subset_rank
                or assignment.decision != expected_decision
            ):
                raise ValueError(
                    "assignment does not replay from the frozen snapshot, units, "
                    "and exact ranks"
                )
            if assignment.decision.selected != self.catalog[subset_rank]:
                raise ValueError("assignment decision differs from its catalog subset")

        references = tuple(entry.reference for entry in self.snapshot.entries)
        expected_support = _support_for(references, self.assignments)
        if self.support != expected_support:
            raise ValueError("support report does not match the frozen assignments")
        unsupported = tuple(value for value in self.support if not value.has_overlap)
        if unsupported:
            details = ", ".join(
                f"{value.reference.insight_id.value}@{value.reference.version}"
                f"(treated={value.treated_count},control={value.control_count})"
                for value in unsupported
            )
            raise ValueError(
                "balanced subset-block preflight requires treated and control "
                f"support for every insight; missing overlap: {details}"
            )

    @property
    def full_block_count(self) -> int:
        return len(self.ordered_units) // len(self.catalog)

    @property
    def remainder_size(self) -> int:
        return len(self.ordered_units) % len(self.catalog)

    @overload
    def assignment_for(
        self,
        generation: int,
        lane_id: str,
        *,
        unit_key: None = None,
    ) -> BalancedSubsetBlockAssignment: ...

    @overload
    def assignment_for(
        self,
        generation: str,
        lane_id: None = None,
        *,
        unit_key: None = None,
    ) -> BalancedSubsetBlockAssignment: ...

    @overload
    def assignment_for(
        self,
        generation: None = None,
        lane_id: None = None,
        *,
        unit_key: str,
    ) -> BalancedSubsetBlockAssignment: ...

    def assignment_for(
        self,
        generation: int | str | None = None,
        lane_id: str | None = None,
        *,
        unit_key: str | None = None,
    ) -> BalancedSubsetBlockAssignment:
        """Resolve by ``(generation, lane_id)`` or by one stable unit key."""

        self.__post_init__()
        if unit_key is not None:
            if generation is not None or lane_id is not None:
                raise ValueError(
                    "unit_key cannot be combined with generation or lane_id"
                )
            _require_unit_token(unit_key, "unit_key")
            matches = tuple(
                value for value in self.assignments if value.unit.unit_key == unit_key
            )
        elif type(generation) is str and lane_id is None:
            _require_unit_token(generation, "unit_key")
            matches = tuple(
                value for value in self.assignments if value.unit.unit_key == generation
            )
        else:
            if type(generation) is not int or generation < 0:
                raise ValueError(
                    "generation must be a non-negative exact integer when "
                    "resolving by lane"
                )
            if lane_id is None:
                raise ValueError("lane_id is required when resolving by generation")
            _require_unit_token(lane_id, "lane_id")
            matches = tuple(
                value
                for value in self.assignments
                if value.unit.generation == generation and value.unit.lane_id == lane_id
            )
        if not matches:
            raise KeyError("no balanced memory assignment matches the stable unit")
        if len(matches) != 1:  # pragma: no cover - plan validation forbids this.
            raise RuntimeError("stable memory assignment identity is ambiguous")
        return matches[0]

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "policy_definition_sha256": self.policy_definition_sha256,
            },
            "score_snapshot_sha256": self.snapshot.snapshot_sha256,
            "subset_size": self.subset_size,
            "catalog": [
                [_reference_record(reference) for reference in subset]
                for subset in self.catalog
            ],
            "ordered_units": [unit.to_record() for unit in self.ordered_units],
            "exact_ranks": {
                "full_block_permutation_ranks": list(self.full_block_permutation_ranks),
                "remainder_selection_rank": self.remainder_selection_rank,
                "remainder_permutation_rank": self.remainder_permutation_rank,
            },
            "full_block_count": self.full_block_count,
            "remainder_size": self.remainder_size,
            "assignments": [value.to_record() for value in self.assignments],
            "support": [value.to_record() for value in self.support],
        }

    @property
    def receipt_sha256(self) -> str:
        """Digest authenticating the complete prospective plan and receipt."""

        self.__post_init__()
        return _hash_record(_PLAN_RECEIPT_DOMAIN, self._unsigned_record())

    @property
    def plan_sha256(self) -> str:
        """Alias emphasizing that the receipt authenticates the whole plan."""

        return self.receipt_sha256

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is BalancedSubsetBlockPlan
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class BalancedSubsetBlockPlanner:
    """Map externally generated exact ranks to a frozen balanced schedule."""

    def plan(
        self,
        *,
        snapshot: MemoryScoreSnapshot,
        ordered_units: tuple[StableMemoryAssignmentUnit, ...],
        subset_size: int,
        full_block_permutation_ranks: tuple[int, ...],
        remainder_selection_rank: int | None = None,
        remainder_permutation_rank: int | None = None,
    ) -> BalancedSubsetBlockPlan:
        """Preflight, materialize, and authenticate one prospective schedule.

        Complete blocks are contiguous slices of ``ordered_units``.  A caller
        that wants one complete block per stable lane should therefore supply
        lane-major units; no lane semantics are baked into this policy.
        """

        catalog = canonical_memory_subset_catalog(snapshot, subset_size)
        _validate_ordered_units(ordered_units)
        scheduled_ranks = _scheduled_subset_ranks(
            unit_count=len(ordered_units),
            catalog_size=len(catalog),
            full_block_permutation_ranks=full_block_permutation_ranks,
            remainder_selection_rank=remainder_selection_rank,
            remainder_permutation_rank=remainder_permutation_rank,
        )
        controls = DeterministicMemoryControlPolicy()
        full_block_count = len(ordered_units) // len(catalog)
        assignments = tuple(
            BalancedSubsetBlockAssignment(
                schedule_position=position,
                unit=unit,
                block_index=position // len(catalog),
                block_position=position % len(catalog),
                is_remainder=(position // len(catalog)) >= full_block_count,
                subset_rank=subset_rank,
                decision=controls.uniform(
                    snapshot=snapshot,
                    subset_size=subset_size,
                    subset_rank=subset_rank,
                ),
            )
            for position, (unit, subset_rank) in enumerate(
                zip(ordered_units, scheduled_ranks, strict=True)
            )
        )
        references = tuple(entry.reference for entry in snapshot.entries)
        return BalancedSubsetBlockPlan(
            snapshot=snapshot,
            ordered_units=ordered_units,
            subset_size=subset_size,
            catalog=catalog,
            full_block_permutation_ranks=full_block_permutation_ranks,
            remainder_selection_rank=remainder_selection_rank,
            remainder_permutation_rank=remainder_permutation_rank,
            assignments=assignments,
            support=_support_for(references, assignments),
        )

    def replay(
        self,
        receipt: BalancedSubsetBlockPlan,
        *,
        snapshot: MemoryScoreSnapshot,
        ordered_units: tuple[StableMemoryAssignmentUnit, ...],
    ) -> BalancedSubsetBlockPlan:
        """Rebuild a receipt from its ranks and reject any basis mismatch."""

        if type(receipt) is not BalancedSubsetBlockPlan:
            raise TypeError("receipt must be an exact BalancedSubsetBlockPlan")
        receipt.__post_init__()
        replayed = self.plan(
            snapshot=snapshot,
            ordered_units=ordered_units,
            subset_size=receipt.subset_size,
            full_block_permutation_ranks=(receipt.full_block_permutation_ranks),
            remainder_selection_rank=receipt.remainder_selection_rank,
            remainder_permutation_rank=receipt.remainder_permutation_rank,
        )
        if replayed.receipt_sha256 != receipt.receipt_sha256:
            raise ValueError(
                "replayed balanced subset-block plan does not match the receipt"
            )
        return replayed


__all__ = [
    "BALANCED_SUBSET_BLOCK_POLICY_DEFINITION_SHA256",
    "BALANCED_SUBSET_BLOCK_POLICY_ID",
    "BALANCED_SUBSET_BLOCK_POLICY_VERSION",
    "BalancedSubsetBlockAssignment",
    "BalancedSubsetBlockPlan",
    "BalancedSubsetBlockPlanner",
    "MemorySubsetSupport",
    "StableMemoryAssignmentUnit",
    "canonical_memory_subset_catalog",
]

"""Canonical workload-neutral matching of stable lanes to compatible cards.

This policy consumes only immutable identities and a prospectively derived
bipartite compatibility graph.  It knows nothing about prompts, providers,
finite-action compilers, memory banks, or benchmark configurations.  A caller
must derive every compatibility edge before any current-wave outcome exists.

The objective is closed and deterministic:

1. maximize the number of assigned lanes;
2. among maximum matchings, minimize the lane-ordered assignment vector, where
   card keys use lexical order and an unmatched marker sorts after every card.

Consequently a feasible graph yields one canonical full (all-lanes) matching.
An infeasible graph yields the canonical maximum partial matching and a typed,
hash-authenticated receipt rather than an exception or an implicit fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.patch import require_sha256


CANONICAL_LANE_CARD_MATCHING_POLICY_ID = (
    "canonical_maximum_lane_card_compatibility_matching"
)
CANONICAL_LANE_CARD_MATCHING_POLICY_VERSION = 1
CANONICAL_LANE_CARD_MATCHING_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:canonical-maximum-lane-card-compatibility-matching:v1;"
    b"canonical-identity-bound-bipartite-graph;maximize-assigned-lanes;"
    b"lexicographically-minimize-lane-ordered-card-vector;unmatched-sorts-last;"
    b"full-means-every-lane-matched;typed-maximum-partial-infeasibility;"
    b"provider-evaluator-and-workload-neutral;exact-replay-receipt"
).hexdigest()

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_GRAPH_DOMAIN = b"agent-evolve:lane-card-compatibility-graph:v1\x00"
_INPUT_DOMAIN = b"agent-evolve:lane-card-matching-input:v1\x00"
_EDGE_DOMAIN = b"agent-evolve:lane-card-compatibility-edge:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:lane-card-matching-receipt:v1\x00"


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


def _require_identifier(value: str, *, name: str) -> None:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must use the canonical identifier grammar")


@dataclass(frozen=True, slots=True)
class LaneCardMatchingLane:
    """One stable consumer lane and its exact caller-owned identity."""

    lane_id: str
    lane_identity_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.lane_id, name="lane_id")
        require_sha256(self.lane_identity_sha256, "lane_identity_sha256")

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "lane_id": self.lane_id,
            "lane_identity_sha256": self.lane_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class LaneCardMatchingCard:
    """One assignable memory card and its exact caller-owned identity."""

    card_key: str
    card_identity_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.card_key, name="card_key")
        require_sha256(self.card_identity_sha256, "card_identity_sha256")

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "card_key": self.card_key,
            "card_identity_sha256": self.card_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class LaneCardCompatibility:
    """One prospectively established lane/card compatibility edge.

    ``compatibility_evidence_sha256`` is supplied by trusted application code.
    For finite-action workloads it should identify the exact support-derivation
    receipt for this lane's current finite contract and this card's semantics.
    """

    lane_id: str
    card_key: str
    compatibility_evidence_sha256: str
    edge_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_identifier(self.lane_id, name="lane_id")
        _require_identifier(self.card_key, name="card_key")
        require_sha256(
            self.compatibility_evidence_sha256,
            "compatibility_evidence_sha256",
        )
        object.__setattr__(
            self,
            "edge_sha256",
            _hash(_EDGE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, str]:
        return {
            "lane_id": self.lane_id,
            "card_key": self.card_key,
            "compatibility_evidence_sha256": self.compatibility_evidence_sha256,
        }

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {**self._unsigned_record(), "edge_sha256": self.edge_sha256}


@dataclass(frozen=True, slots=True, eq=False)
class LaneCardMatchingInput:
    """Canonical exact populations and complete positive compatibility graph."""

    lanes: tuple[LaneCardMatchingLane, ...]
    cards: tuple[LaneCardMatchingCard, ...]
    compatibilities: tuple[LaneCardCompatibility, ...]

    def __post_init__(self) -> None:
        if type(self.lanes) is not tuple or not self.lanes or any(
            type(value) is not LaneCardMatchingLane for value in self.lanes
        ):
            raise ValueError("lanes must be a non-empty exact lane tuple")
        if type(self.cards) is not tuple or not self.cards or any(
            type(value) is not LaneCardMatchingCard for value in self.cards
        ):
            raise ValueError("cards must be a non-empty exact card tuple")
        if type(self.compatibilities) is not tuple or any(
            type(value) is not LaneCardCompatibility
            for value in self.compatibilities
        ):
            raise TypeError("compatibilities must be an exact edge tuple")
        for lane in self.lanes:
            lane.__post_init__()
        for card in self.cards:
            card.__post_init__()
        for compatibility in self.compatibilities:
            compatibility.__post_init__()

        lane_ids = tuple(value.lane_id for value in self.lanes)
        card_keys = tuple(value.card_key for value in self.cards)
        if lane_ids != tuple(sorted(set(lane_ids))):
            raise ValueError("lanes must use unique canonical lane_id order")
        if card_keys != tuple(sorted(set(card_keys))):
            raise ValueError("cards must use unique canonical card_key order")
        edge_keys = tuple(
            (value.lane_id, value.card_key) for value in self.compatibilities
        )
        if edge_keys != tuple(sorted(set(edge_keys))):
            raise ValueError(
                "compatibilities must use unique canonical lane/card order"
            )
        unknown_lanes = tuple(
            lane_id for lane_id, _ in edge_keys if lane_id not in set(lane_ids)
        )
        unknown_cards = tuple(
            card_key for _, card_key in edge_keys if card_key not in set(card_keys)
        )
        if unknown_lanes or unknown_cards:
            raise ValueError("compatibility edge names a foreign lane or card")

    def _graph_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "lanes": [value.to_record() for value in self.lanes],
            "cards": [value.to_record() for value in self.cards],
            "compatibilities": [
                value.to_record() for value in self.compatibilities
            ],
        }

    @property
    def compatibility_graph_sha256(self) -> str:
        return _hash(_GRAPH_DOMAIN, self._graph_record())

    def _unsigned_record(self) -> dict[str, object]:
        return {
            **self._graph_record(),
            "compatibility_graph_sha256": self.compatibility_graph_sha256,
        }

    @property
    def input_sha256(self) -> str:
        return _hash(_INPUT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "input_sha256": self.input_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is LaneCardMatchingInput
            and self.input_sha256 == other.input_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class LaneCardAssignment:
    """One exact selected edge in the canonical maximum matching."""

    lane: LaneCardMatchingLane
    card: LaneCardMatchingCard
    compatibility_edge_sha256: str

    def __post_init__(self) -> None:
        if type(self.lane) is not LaneCardMatchingLane:
            raise TypeError("lane must be an exact LaneCardMatchingLane")
        if type(self.card) is not LaneCardMatchingCard:
            raise TypeError("card must be an exact LaneCardMatchingCard")
        self.lane.__post_init__()
        self.card.__post_init__()
        require_sha256(
            self.compatibility_edge_sha256,
            "compatibility_edge_sha256",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "lane": self.lane.to_record(),
            "card": self.card.to_record(),
            "compatibility_edge_sha256": self.compatibility_edge_sha256,
        }


class LaneCardMatchingStatus(str, Enum):
    """Closed result of the all-lanes matching requirement."""

    FULL = "full"
    INFEASIBLE = "infeasible"


def _adjacency(
    matching_input: LaneCardMatchingInput,
) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = {
        value.lane_id: [] for value in matching_input.lanes
    }
    for edge in matching_input.compatibilities:
        values[edge.lane_id].append(edge.card_key)
    return {key: tuple(value) for key, value in values.items()}


def _maximum_cardinality(
    lane_ids: tuple[str, ...],
    available_card_keys: frozenset[str],
    adjacency: dict[str, tuple[str, ...]],
) -> int:
    """Return bipartite maximum cardinality for one residual graph."""

    card_to_lane: dict[str, str] = {}

    def augment(lane_id: str, seen_cards: set[str]) -> bool:
        for card_key in adjacency[lane_id]:
            if card_key not in available_card_keys or card_key in seen_cards:
                continue
            seen_cards.add(card_key)
            incumbent = card_to_lane.get(card_key)
            if incumbent is None or augment(incumbent, seen_cards):
                card_to_lane[card_key] = lane_id
                return True
        return False

    for lane_id in lane_ids:
        augment(lane_id, set())
    return len(card_to_lane)


def _canonical_maximum_pairs(
    matching_input: LaneCardMatchingInput,
) -> tuple[tuple[str, str], ...]:
    """Derive the lexicographically least lane vector at maximum cardinality."""

    matching_input.__post_init__()
    lane_ids = tuple(value.lane_id for value in matching_input.lanes)
    card_keys = frozenset(value.card_key for value in matching_input.cards)
    adjacency = _adjacency(matching_input)
    target = _maximum_cardinality(lane_ids, card_keys, adjacency)
    used: set[str] = set()
    fixed_cardinality = 0
    selected: list[tuple[str, str]] = []

    for index, lane_id in enumerate(lane_ids):
        residual_lanes = lane_ids[index + 1 :]
        available = card_keys.difference(used)
        # ``None`` is deliberately last: an earlier lane is matched whenever
        # doing so can retain the globally maximum cardinality.
        candidates: tuple[str | None, ...] = (
            *(value for value in adjacency[lane_id] if value in available),
            None,
        )
        chosen: str | None | object = _NO_CHOICE
        for candidate in candidates:
            next_used = used if candidate is None else used.union((candidate,))
            attainable = fixed_cardinality + (candidate is not None)
            attainable += _maximum_cardinality(
                residual_lanes,
                card_keys.difference(next_used),
                adjacency,
            )
            if attainable == target:
                chosen = candidate
                break
        if chosen is _NO_CHOICE:  # pragma: no cover - None always preserves a maximum.
            raise AssertionError("canonical matching construction lost its optimum")
        if type(chosen) is str:
            selected.append((lane_id, chosen))
            used.add(chosen)
            fixed_cardinality += 1

    if fixed_cardinality != target:
        raise AssertionError("canonical matching differs from maximum cardinality")
    return tuple(selected)


_NO_CHOICE = object()


def _assignments_for(
    matching_input: LaneCardMatchingInput,
    pairs: tuple[tuple[str, str], ...],
) -> tuple[LaneCardAssignment, ...]:
    lanes = {value.lane_id: value for value in matching_input.lanes}
    cards = {value.card_key: value for value in matching_input.cards}
    edges = {
        (value.lane_id, value.card_key): value
        for value in matching_input.compatibilities
    }
    return tuple(
        LaneCardAssignment(
            lane=lanes[lane_id],
            card=cards[card_key],
            compatibility_edge_sha256=edges[(lane_id, card_key)].edge_sha256,
        )
        for lane_id, card_key in pairs
    )


@dataclass(frozen=True, slots=True, eq=False)
class LaneCardMatchingReceipt:
    """Self-validating full or typed-infeasible canonical matching receipt."""

    matching_input: LaneCardMatchingInput
    status: LaneCardMatchingStatus
    assignments: tuple[LaneCardAssignment, ...]
    unmatched_lanes: tuple[LaneCardMatchingLane, ...]
    unused_cards: tuple[LaneCardMatchingCard, ...]
    maximum_cardinality: int
    required_cardinality: int
    policy_id: str = CANONICAL_LANE_CARD_MATCHING_POLICY_ID
    policy_version: int = CANONICAL_LANE_CARD_MATCHING_POLICY_VERSION
    policy_definition_sha256: str = (
        CANONICAL_LANE_CARD_MATCHING_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if type(self.matching_input) is not LaneCardMatchingInput:
            raise TypeError("matching_input must be an exact LaneCardMatchingInput")
        self.matching_input.__post_init__()
        if type(self.status) is not LaneCardMatchingStatus:
            raise TypeError("status must be an exact LaneCardMatchingStatus")
        if (
            self.policy_id != CANONICAL_LANE_CARD_MATCHING_POLICY_ID
            or self.policy_version != CANONICAL_LANE_CARD_MATCHING_POLICY_VERSION
            or self.policy_definition_sha256
            != CANONICAL_LANE_CARD_MATCHING_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("unsupported lane/card matching policy")

        expected_pairs = _canonical_maximum_pairs(self.matching_input)
        expected_assignments = _assignments_for(self.matching_input, expected_pairs)
        if self.assignments != expected_assignments:
            raise ValueError("assignments differ from canonical maximum matching")
        assigned_lane_ids = {value.lane.lane_id for value in self.assignments}
        assigned_card_keys = {value.card.card_key for value in self.assignments}
        expected_unmatched = tuple(
            value
            for value in self.matching_input.lanes
            if value.lane_id not in assigned_lane_ids
        )
        expected_unused = tuple(
            value
            for value in self.matching_input.cards
            if value.card_key not in assigned_card_keys
        )
        if self.unmatched_lanes != expected_unmatched:
            raise ValueError("unmatched_lanes differ from canonical maximum matching")
        if self.unused_cards != expected_unused:
            raise ValueError("unused_cards differ from canonical maximum matching")
        expected_maximum = len(expected_assignments)
        expected_required = len(self.matching_input.lanes)
        if (
            type(self.maximum_cardinality) is not int
            or self.maximum_cardinality != expected_maximum
        ):
            raise ValueError("maximum_cardinality differs from exact replay")
        if (
            type(self.required_cardinality) is not int
            or self.required_cardinality != expected_required
        ):
            raise ValueError("required_cardinality must equal the lane count")
        expected_status = (
            LaneCardMatchingStatus.FULL
            if expected_maximum == expected_required
            else LaneCardMatchingStatus.INFEASIBLE
        )
        if self.status is not expected_status:
            raise ValueError("status differs from exact matching feasibility")

    @property
    def is_full(self) -> bool:
        return self.status is LaneCardMatchingStatus.FULL

    @property
    def deficiency(self) -> int:
        return self.required_cardinality - self.maximum_cardinality

    def assignment_for(self, lane_id: str) -> LaneCardAssignment:
        """Resolve one match; an infeasible unmatched lane raises ``KeyError``."""

        self.__post_init__()
        _require_identifier(lane_id, name="lane_id")
        matches = tuple(
            value for value in self.assignments if value.lane.lane_id == lane_id
        )
        if not matches:
            raise KeyError("lane is unmatched or foreign to this receipt")
        return matches[0]

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "policy_definition_sha256": self.policy_definition_sha256,
            },
            "objective": {
                "primary": "maximum_cardinality",
                "tie_break": "lexicographically_least_lane_ordered_card_vector",
                "unmatched_sorts_after_cards": True,
                "full_requires_every_lane": True,
            },
            "matching_input_sha256": self.matching_input.input_sha256,
            "compatibility_graph_sha256": (
                self.matching_input.compatibility_graph_sha256
            ),
            "matching_input": self.matching_input.to_record(),
            "status": self.status.value,
            "maximum_cardinality": self.maximum_cardinality,
            "required_cardinality": self.required_cardinality,
            "deficiency": self.deficiency,
            "assignments": [value.to_record() for value in self.assignments],
            "unmatched_lanes": [
                value.to_record() for value in self.unmatched_lanes
            ],
            "unused_cards": [value.to_record() for value in self.unused_cards],
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is LaneCardMatchingReceipt
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


class LaneCardMatchingVerificationError(RuntimeError):
    """A supplied receipt differs from deterministic canonical replay."""


@dataclass(frozen=True, slots=True)
class CanonicalLaneCardMatchingPlanner:
    """Build and replay one canonical maximum lane/card matching receipt."""

    def plan(
        self,
        *,
        lanes: tuple[LaneCardMatchingLane, ...],
        cards: tuple[LaneCardMatchingCard, ...],
        compatibilities: tuple[LaneCardCompatibility, ...],
    ) -> LaneCardMatchingReceipt:
        matching_input = LaneCardMatchingInput(
            lanes=lanes,
            cards=cards,
            compatibilities=compatibilities,
        )
        pairs = _canonical_maximum_pairs(matching_input)
        assignments = _assignments_for(matching_input, pairs)
        assigned_lane_ids = {value.lane.lane_id for value in assignments}
        assigned_card_keys = {value.card.card_key for value in assignments}
        maximum = len(assignments)
        required = len(lanes)
        return LaneCardMatchingReceipt(
            matching_input=matching_input,
            status=(
                LaneCardMatchingStatus.FULL
                if maximum == required
                else LaneCardMatchingStatus.INFEASIBLE
            ),
            assignments=assignments,
            unmatched_lanes=tuple(
                value for value in lanes if value.lane_id not in assigned_lane_ids
            ),
            unused_cards=tuple(
                value for value in cards if value.card_key not in assigned_card_keys
            ),
            maximum_cardinality=maximum,
            required_cardinality=required,
        )

    def replay(
        self,
        receipt: LaneCardMatchingReceipt,
        *,
        lanes: tuple[LaneCardMatchingLane, ...],
        cards: tuple[LaneCardMatchingCard, ...],
        compatibilities: tuple[LaneCardCompatibility, ...],
    ) -> LaneCardMatchingReceipt:
        if type(receipt) is not LaneCardMatchingReceipt:
            raise TypeError("receipt must be an exact LaneCardMatchingReceipt")
        receipt.__post_init__()
        expected = self.plan(
            lanes=lanes,
            cards=cards,
            compatibilities=compatibilities,
        )
        if expected.receipt_sha256 != receipt.receipt_sha256:
            raise LaneCardMatchingVerificationError(
                "lane/card matching receipt differs from deterministic replay"
            )
        return expected


__all__ = [
    "CANONICAL_LANE_CARD_MATCHING_POLICY_DEFINITION_SHA256",
    "CANONICAL_LANE_CARD_MATCHING_POLICY_ID",
    "CANONICAL_LANE_CARD_MATCHING_POLICY_VERSION",
    "CanonicalLaneCardMatchingPlanner",
    "LaneCardAssignment",
    "LaneCardCompatibility",
    "LaneCardMatchingCard",
    "LaneCardMatchingInput",
    "LaneCardMatchingLane",
    "LaneCardMatchingReceipt",
    "LaneCardMatchingStatus",
    "LaneCardMatchingVerificationError",
]

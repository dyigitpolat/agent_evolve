"""Provider-free checks for canonical compatibility-aware lane/card matching."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from agent_evolve.policies.memory.compatibility_matching import (
    CANONICAL_LANE_CARD_MATCHING_POLICY_DEFINITION_SHA256,
    CANONICAL_LANE_CARD_MATCHING_POLICY_ID,
    CANONICAL_LANE_CARD_MATCHING_POLICY_VERSION,
    CanonicalLaneCardMatchingPlanner,
    LaneCardCompatibility,
    LaneCardMatchingCard,
    LaneCardMatchingInput,
    LaneCardMatchingLane,
    LaneCardMatchingStatus,
    LaneCardMatchingVerificationError,
)


def _lane(name: str, identity_digit: str) -> LaneCardMatchingLane:
    return LaneCardMatchingLane(
        lane_id=f"lane_{name}",
        lane_identity_sha256=identity_digit * 64,
    )


def _card(name: str, identity_digit: str) -> LaneCardMatchingCard:
    return LaneCardMatchingCard(
        card_key=f"card_{name}",
        card_identity_sha256=identity_digit * 64,
    )


def _edge(
    lane: LaneCardMatchingLane,
    card: LaneCardMatchingCard,
    evidence_digit: str,
) -> LaneCardCompatibility:
    return LaneCardCompatibility(
        lane_id=lane.lane_id,
        card_key=card.card_key,
        compatibility_evidence_sha256=evidence_digit * 64,
    )


def test_canonical_full_matching_preserves_global_feasibility() -> None:
    """The first lexical edge is skipped when it would destroy a full match."""

    lane_a, lane_b, lane_c = (
        _lane("a", "1"),
        _lane("b", "2"),
        _lane("c", "3"),
    )
    card_a, card_b, card_c = (
        _card("a", "4"),
        _card("b", "5"),
        _card("c", "6"),
    )
    compatibilities = (
        _edge(lane_a, card_a, "a"),
        _edge(lane_a, card_b, "b"),
        _edge(lane_b, card_a, "c"),
        _edge(lane_c, card_b, "d"),
        _edge(lane_c, card_c, "e"),
    )

    receipt = CanonicalLaneCardMatchingPlanner().plan(
        lanes=(lane_a, lane_b, lane_c),
        cards=(card_a, card_b, card_c),
        compatibilities=compatibilities,
    )

    assert receipt.status is LaneCardMatchingStatus.FULL
    assert receipt.is_full
    assert receipt.maximum_cardinality == receipt.required_cardinality == 3
    assert receipt.deficiency == 0
    assert tuple(
        (value.lane.lane_id, value.card.card_key) for value in receipt.assignments
    ) == (
        ("lane_a", "card_b"),
        ("lane_b", "card_a"),
        ("lane_c", "card_c"),
    )
    assert receipt.unmatched_lanes == ()
    assert receipt.unused_cards == ()
    assert receipt.assignment_for("lane_a").card.card_key == "card_b"


def test_tie_break_is_lexically_canonical_and_extra_cards_remain_unused() -> None:
    lane_a, lane_b = _lane("a", "1"), _lane("b", "2")
    card_a, card_b, card_c = (
        _card("a", "3"),
        _card("b", "4"),
        _card("c", "5"),
    )
    compatibilities = tuple(
        _edge(lane, card, evidence)
        for lane, card, evidence in (
            (lane_a, card_a, "a"),
            (lane_a, card_b, "b"),
            (lane_b, card_a, "c"),
            (lane_b, card_b, "d"),
        )
    )

    receipt = CanonicalLaneCardMatchingPlanner().plan(
        lanes=(lane_a, lane_b),
        cards=(card_a, card_b, card_c),
        compatibilities=compatibilities,
    )

    assert tuple(
        (value.lane.lane_id, value.card.card_key) for value in receipt.assignments
    ) == (("lane_a", "card_a"), ("lane_b", "card_b"))
    assert receipt.unused_cards == (card_c,)


def test_infeasible_graph_returns_typed_authenticated_maximum_receipt() -> None:
    lane_a, lane_b, lane_c = (
        _lane("a", "1"),
        _lane("b", "2"),
        _lane("c", "3"),
    )
    card_a, card_b = _card("a", "4"), _card("b", "5")
    compatibilities = (
        _edge(lane_a, card_a, "a"),
        _edge(lane_b, card_a, "b"),
        _edge(lane_c, card_b, "c"),
    )

    receipt = CanonicalLaneCardMatchingPlanner().plan(
        lanes=(lane_a, lane_b, lane_c),
        cards=(card_a, card_b),
        compatibilities=compatibilities,
    )

    assert receipt.status is LaneCardMatchingStatus.INFEASIBLE
    assert not receipt.is_full
    assert receipt.maximum_cardinality == 2
    assert receipt.required_cardinality == 3
    assert receipt.deficiency == 1
    # The unmatched marker sorts after cards, so the earliest compatible lane
    # receives card_a in the canonical maximum partial matching.
    assert tuple(
        (value.lane.lane_id, value.card.card_key) for value in receipt.assignments
    ) == (("lane_a", "card_a"), ("lane_c", "card_b"))
    assert receipt.unmatched_lanes == (lane_b,)
    assert receipt.unused_cards == ()
    with pytest.raises(KeyError, match="unmatched or foreign"):
        receipt.assignment_for("lane_b")

    record = receipt.to_record()
    assert record["status"] == "infeasible"
    assert record["deficiency"] == 1
    assert record["receipt_sha256"] == receipt.receipt_sha256
    assert record["policy"] == {
        "policy_id": CANONICAL_LANE_CARD_MATCHING_POLICY_ID,
        "policy_version": CANONICAL_LANE_CARD_MATCHING_POLICY_VERSION,
        "policy_definition_sha256": (
            CANONICAL_LANE_CARD_MATCHING_POLICY_DEFINITION_SHA256
        ),
    }


def test_empty_positive_graph_is_a_zero_cardinality_infeasible_receipt() -> None:
    lanes = (_lane("a", "1"), _lane("b", "2"))
    cards = (_card("a", "3"), _card("b", "4"))

    receipt = CanonicalLaneCardMatchingPlanner().plan(
        lanes=lanes,
        cards=cards,
        compatibilities=(),
    )

    assert receipt.status is LaneCardMatchingStatus.INFEASIBLE
    assert receipt.maximum_cardinality == 0
    assert receipt.deficiency == 2
    assert receipt.assignments == ()
    assert receipt.unmatched_lanes == lanes
    assert receipt.unused_cards == cards


def test_receipt_binds_populations_edges_evidence_policy_and_exact_replay() -> None:
    lanes = (_lane("a", "1"), _lane("b", "2"))
    cards = (_card("a", "3"), _card("b", "4"))
    edges = (
        _edge(lanes[0], cards[0], "a"),
        _edge(lanes[1], cards[1], "b"),
    )
    planner = CanonicalLaneCardMatchingPlanner()
    receipt = planner.plan(lanes=lanes, cards=cards, compatibilities=edges)
    replayed = planner.replay(
        receipt,
        lanes=lanes,
        cards=cards,
        compatibilities=edges,
    )

    assert replayed.to_record() == receipt.to_record()
    record = receipt.to_record()
    matching_input = record["matching_input"]
    assert isinstance(matching_input, dict)
    assert record["matching_input_sha256"] == matching_input["input_sha256"]
    assert (
        record["compatibility_graph_sha256"]
        == matching_input["compatibility_graph_sha256"]
    )
    assert receipt.assignments[0].compatibility_edge_sha256 == edges[0].edge_sha256

    changed_edges = (
        replace(edges[0], compatibility_evidence_sha256="c" * 64),
        edges[1],
    )
    changed = planner.plan(
        lanes=lanes,
        cards=cards,
        compatibilities=changed_edges,
    )
    assert (
        changed.matching_input.compatibility_graph_sha256
        != receipt.matching_input.compatibility_graph_sha256
    )
    assert changed.matching_input.input_sha256 != receipt.matching_input.input_sha256
    assert changed.receipt_sha256 != receipt.receipt_sha256
    with pytest.raises(LaneCardMatchingVerificationError, match="deterministic replay"):
        planner.replay(
            receipt,
            lanes=lanes,
            cards=cards,
            compatibilities=changed_edges,
        )


def test_input_fails_closed_on_noncanonical_duplicate_or_foreign_edges() -> None:
    lanes = (_lane("a", "1"), _lane("b", "2"))
    cards = (_card("a", "3"), _card("b", "4"))
    edge_a = _edge(lanes[0], cards[0], "a")
    edge_b = _edge(lanes[1], cards[1], "b")

    with pytest.raises(ValueError, match="canonical lane_id order"):
        LaneCardMatchingInput(
            lanes=tuple(reversed(lanes)),
            cards=cards,
            compatibilities=(edge_a, edge_b),
        )
    with pytest.raises(ValueError, match="canonical lane/card order"):
        LaneCardMatchingInput(
            lanes=lanes,
            cards=cards,
            compatibilities=(edge_b, edge_a),
        )
    with pytest.raises(ValueError, match="canonical lane/card order"):
        LaneCardMatchingInput(
            lanes=lanes,
            cards=cards,
            compatibilities=(edge_a, edge_a),
        )
    with pytest.raises(ValueError, match="foreign lane or card"):
        LaneCardMatchingInput(
            lanes=lanes,
            cards=cards,
            compatibilities=(
                LaneCardCompatibility(
                    lane_id="lane_foreign",
                    card_key=cards[0].card_key,
                    compatibility_evidence_sha256="f" * 64,
                ),
            ),
        )

    receipt = CanonicalLaneCardMatchingPlanner().plan(
        lanes=lanes,
        cards=cards,
        compatibilities=(edge_a, edge_b),
    )
    with pytest.raises(FrozenInstanceError):
        receipt.status = LaneCardMatchingStatus.INFEASIBLE

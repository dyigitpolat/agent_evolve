"""Front-proximity admission: support restriction, blindness, identity."""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from agent_evolve.application.front_proximity_admission import (
    FrontProximityAdmission,
    FrontProximityAdmissionConfig,
    anchors_from_corpus,
    chebyshev_excess,
)
from agent_evolve.application.sequential_market_replay import (
    FrozenScoreTopKReplayPolicy,
    LaneHeadsReplayPolicy,
    MarketCandidateRecord,
    MarketRecord,
    ReplaySelection,
    SequentialMarketReplay,
    market_record_from_corpus,
)

CORPUS = pathlib.Path(
    "/home/yigit/repos/research_stuff/papers/agent_evolve_aaai_2027/"
    "research_artifacts/data/jul27_allocator_replay_corpus_v1"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _point(levels: float, luts: float):
    return (("a", levels), ("b", luts))


def _market(candidate_count: int = 8) -> MarketRecord:
    candidates = []
    for index in range(candidate_count):
        candidates.append(
            MarketCandidateRecord(
                action_sha256=_sha(f"candidate-{index}"),
                engine_id="engine.one" if index % 2 else "engine.two",
                native_rank=index // 2 + 1,
                frozen_score=1.0 - index / 100.0,
                forecast=None,
                evaluated=True,
                feasible=True,
                objectives=_point(0.5 - index / 1000.0, 0.5),
            )
        )
    return MarketRecord(
        market_id="test_market",
        archive_points=(_point(0.6, 0.6), _point(0.4, 0.9)),
        hv_reference_point=_point(1.0, 1.0),
        candidates=tuple(candidates),
    )


def _anchors(record: MarketRecord) -> dict:
    # Half the market anchors on the front, half deep in the interior.
    anchors = {}
    for index, candidate in enumerate(record.candidates):
        anchors[candidate.action_sha256] = (
            _point(0.55, 0.55) if index % 2 == 0 else _point(0.95, 0.95)
        )
    return anchors


def test_chebyshev_excess_sign_matches_front_dominance() -> None:
    archive = (_point(0.6, 0.6),)
    assert chebyshev_excess(_point(0.5, 0.5), archive, ("a", "b")) < 0.0
    assert chebyshev_excess(_point(0.6, 0.6), archive, ("a", "b")) == 0.0
    assert chebyshev_excess(_point(0.7, 0.7), archive, ("a", "b")) > 0.0
    # incomparable-but-worse-on-one-axis scores by that worst axis
    assert chebyshev_excess(
        _point(0.5, 0.9), archive, ("a", "b")
    ) == pytest.approx(0.3)
    # a dominated archive point must not lower the excess
    assert chebyshev_excess(
        _point(0.7, 0.7), (_point(0.6, 0.6), _point(0.9, 0.9)), ("a", "b")
    ) == pytest.approx(0.1)


def test_chebyshev_excess_rejects_a_foreign_metric_frame() -> None:
    with pytest.raises(ValueError):
        chebyshev_excess(
            (("a", 0.5), ("c", 0.5)), (_point(0.6, 0.6),), ("a", "b")
        )


def test_admissible_drops_the_dominated_half() -> None:
    record = _market()
    arm = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(),
        anchors=_anchors(record),
        config=FrontProximityAdmissionConfig(keep_fraction=0.5),
    )
    kept = arm.admissible(
        record=record,
        revealed=(),
        selectable_action_sha256s=tuple(
            value.action_sha256 for value in record.candidates
        ),
    )
    front_anchored = {
        value.action_sha256
        for index, value in enumerate(record.candidates)
        if index % 2 == 0
    }
    assert set(kept) == front_anchored


def test_keep_fraction_one_is_the_identity_support() -> None:
    record = _market()
    support = tuple(
        sorted(value.action_sha256 for value in record.candidates)
    )
    arm = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(),
        anchors=_anchors(record),
        config=FrontProximityAdmissionConfig(keep_fraction=1.0),
    )
    assert arm.admissible(
        record=record, revealed=(), selectable_action_sha256s=support
    ) == support


def test_unknown_anchors_are_scored_at_the_median_not_evicted() -> None:
    record = _market()
    anchors = _anchors(record)
    orphan = record.candidates[1].action_sha256  # interior anchor
    anchors.pop(orphan)
    arm = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(),
        anchors=anchors,
        config=FrontProximityAdmissionConfig(keep_fraction=0.75),
    )
    kept = arm.admissible(
        record=record,
        revealed=(),
        selectable_action_sha256s=tuple(
            value.action_sha256 for value in record.candidates
        ),
    )
    assert orphan in kept


def test_minimum_support_floor_is_respected() -> None:
    record = _market()
    arm = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(),
        anchors=_anchors(record),
        config=FrontProximityAdmissionConfig(
            keep_fraction=0.01, minimum_support=3
        ),
    )
    kept = arm.admissible(
        record=record,
        revealed=(),
        selectable_action_sha256s=tuple(
            value.action_sha256 for value in record.candidates
        ),
    )
    assert len(kept) >= 3


def test_selection_stays_inside_the_admissible_support() -> None:
    """Every seat must fall inside the support admissible AT THAT SEAT."""

    record = _market()
    arm = FrontProximityAdmission(
        inner=FrozenScoreTopKReplayPolicy(),
        anchors=_anchors(record),
        config=FrontProximityAdmissionConfig(keep_fraction=0.5),
    )
    result = SequentialMarketReplay().run(
        record=record, policy=arm, budget=3
    )
    selectable = {value.action_sha256 for value in record.candidates}
    revealed: list = []
    for receipt in result.receipts:
        kept = arm.admissible(
            record=record,
            revealed=tuple(revealed),
            selectable_action_sha256s=tuple(sorted(selectable)),
        )
        assert receipt.action_sha256 in kept
        selectable.discard(receipt.action_sha256)
        revealed.append(receipt)
    assert len(result.receipts) == 3


def test_an_escaping_inner_policy_is_rejected() -> None:
    record = _market()
    escaped = record.candidates[1].action_sha256

    class Escaper:
        policy_id = "test.escaper"

        def select(self, **_kwargs) -> ReplaySelection:
            return ReplaySelection(
                action_sha256=escaped, selection_propensity=1.0
            )

    arm = FrontProximityAdmission(
        inner=Escaper(),
        anchors=_anchors(record),
        config=FrontProximityAdmissionConfig(keep_fraction=0.5),
    )
    with pytest.raises(ValueError):
        arm.select(
            record=record,
            revealed=(),
            selectable_action_sha256s=tuple(
                value.action_sha256 for value in record.candidates
            ),
            step_index=0,
            budget=3,
        )


def test_admission_never_reads_an_unrevealed_outcome() -> None:
    """Flipping every unselected outcome must not move the support."""

    record = _market()
    anchors = _anchors(record)
    arm = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(), anchors=anchors
    )
    support = tuple(
        sorted(value.action_sha256 for value in record.candidates)
    )
    before = arm.admissible(
        record=record, revealed=(), selectable_action_sha256s=support
    )
    mutated = MarketRecord(
        market_id=record.market_id,
        archive_points=record.archive_points,
        hv_reference_point=record.hv_reference_point,
        candidates=tuple(
            MarketCandidateRecord(
                action_sha256=value.action_sha256,
                engine_id=value.engine_id,
                native_rank=value.native_rank,
                frozen_score=value.frozen_score,
                forecast=value.forecast,
                evaluated=value.evaluated,
                feasible=value.feasible,
                objectives=_point(0.01, 0.01),
            )
            for value in record.candidates
        ),
    )
    after = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(), anchors=anchors
    ).admissible(
        record=mutated, revealed=(), selectable_action_sha256s=support
    )
    assert before == after


def test_definition_sha_moves_with_config_and_inner_policy() -> None:
    record = _market()
    anchors = _anchors(record)
    base = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(), anchors=anchors
    )
    same = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(), anchors=anchors
    )
    other_config = FrontProximityAdmission(
        inner=LaneHeadsReplayPolicy(),
        anchors=anchors,
        config=FrontProximityAdmissionConfig(keep_fraction=0.25),
    )
    other_inner = FrontProximityAdmission(
        inner=FrozenScoreTopKReplayPolicy(), anchors=anchors
    )
    assert base.definition_sha256 == same.definition_sha256
    assert base.definition_sha256 != other_config.definition_sha256
    assert base.definition_sha256 != other_inner.definition_sha256


@pytest.mark.parametrize(
    "value", [0.0, -0.5, 1.5, float("nan"), 1, "0.5"]
)
def test_config_rejects_an_invalid_keep_fraction(value) -> None:
    with pytest.raises((ValueError, TypeError)):
        FrontProximityAdmissionConfig(keep_fraction=value)


@pytest.mark.skipif(
    not (CORPUS / "manifest.json").exists(), reason="corpus not present"
)
def test_corpus_anchor_extraction_matches_the_replay_identity() -> None:
    payload = json.loads(
        (CORPUS / "boils_v70_qwen_support_horizon_b32.json").read_text()
    )
    record = market_record_from_corpus(payload)
    anchors = anchors_from_corpus(payload)
    known = {value.action_sha256 for value in record.candidates}
    assert anchors
    assert set(anchors) <= known
    metric_ids = record.metric_ids
    for point in anchors.values():
        assert tuple(sorted(dict(point))) == metric_ids


@pytest.mark.skipif(
    not (CORPUS / "manifest.json").exists(), reason="corpus not present"
)
def test_corpus_anchor_excess_spans_both_sides_of_the_front() -> None:
    """The arm is only meaningful where anchors actually differ in depth.

    Guards the measurement in the anatomy memo: the jul27 corpus BOiLS
    markets do contain dominated-anchor mass, so the corpus is a valid
    replication set for this signal rather than a degenerate one.
    """

    payload = json.loads(
        (CORPUS / "boils_v70_qwen_support_horizon_b32.json").read_text()
    )
    record = market_record_from_corpus(payload)
    anchors = anchors_from_corpus(payload)
    excess = [
        chebyshev_excess(point, record.archive_points, record.metric_ids)
        for point in anchors.values()
    ]
    assert excess
    assert min(excess) <= 0.0 < max(excess)

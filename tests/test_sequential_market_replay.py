from __future__ import annotations

import hashlib
import itertools
import json
import pathlib

import pytest

from agent_evolve.application.sequential_market_replay import (
    ExactHypervolumeGainPort,
    FrozenScoreTopKReplayPolicy,
    LaneHeadsReplayPolicy,
    MarketCandidateRecord,
    MarketRecord,
    NativeRankRoundRobinReplayPolicy,
    SequentialMarketReplay,
    UniformRandomReplayPolicy,
    V8LiteReplayPolicy,
    exact_hypervolume,
    market_record_from_corpus,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LiteAllocationConfig,
    V8LiteAllocationPolicy,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _point(x: float, y: float) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((("m.x", float(x)), ("m.y", float(y)))))


_REFERENCE = _point(10.0, 10.0)


def _candidate(
    label: str,
    *,
    engine_id: str = "engine.a",
    native_rank: int,
    frozen_score: float | None = None,
    objectives: tuple[tuple[str, float], ...] | None,
    evaluated: bool = True,
) -> MarketCandidateRecord:
    return MarketCandidateRecord(
        action_sha256=_sha(f"replay:{label}"),
        engine_id=engine_id,
        native_rank=native_rank,
        frozen_score=frozen_score,
        forecast=None,
        evaluated=evaluated,
        feasible=evaluated and objectives is not None,
        objectives=objectives,
    )


def _tiny_market() -> MarketRecord:
    # Archive {(4, 4)} with reference (10, 10): HV(archive) = 36.
    # c1 (2, 6) adds 8; c2 (6, 2) adds 8; jointly they add 16.
    # c3 (5, 5) is dominated and adds 0.
    return MarketRecord(
        market_id="tiny.market",
        archive_points=(_point(4.0, 4.0),),
        hv_reference_point=_REFERENCE,
        candidates=(
            _candidate(
                "c1",
                native_rank=1,
                frozen_score=0.2,
                objectives=_point(2.0, 6.0),
            ),
            _candidate(
                "c2",
                native_rank=2,
                frozen_score=0.1,
                objectives=_point(6.0, 2.0),
            ),
            _candidate(
                "c3",
                native_rank=3,
                frozen_score=0.9,
                objectives=_point(5.0, 5.0),
            ),
        ),
    )


def test_exact_hypervolume_matches_hand_computed_values() -> None:
    reference = (10.0, 10.0)
    assert exact_hypervolume(((4.0, 4.0),), reference) == 36.0
    assert exact_hypervolume(
        ((4.0, 4.0), (2.0, 6.0)),
        reference,
    ) == 44.0
    assert exact_hypervolume(
        ((4.0, 4.0), (2.0, 6.0), (6.0, 2.0)),
        reference,
    ) == 52.0
    # Dominated points contribute nothing; out-of-region points drop.
    assert exact_hypervolume(
        ((4.0, 4.0), (5.0, 5.0), (11.0, 1.0)),
        reference,
    ) == 36.0
    # Three-dimensional sanity.
    assert exact_hypervolume(
        ((0.0, 0.0, 0.0),),
        (1.0, 1.0, 1.0),
    ) == 1.0
    assert exact_hypervolume(
        ((0.5, 0.0, 0.0), (0.0, 0.5, 0.5)),
        (1.0, 1.0, 1.0),
    ) == pytest.approx(0.5 + 0.5 * 0.5 * 0.5)


def test_replay_regret_matches_hand_computed_tiny_market() -> None:
    record = _tiny_market()
    harness = SequentialMarketReplay()

    native = harness.run(
        record=record,
        policy=NativeRankRoundRobinReplayPolicy(),
        budget=2,
    )
    # Native rank picks c1 then c2: 8 + 8 = 16, the oracle pair.
    assert native.realized_gain == 16.0
    assert native.oracle_gain == 16.0
    assert native.regret == 0.0
    assert [value.marginal_gain for value in native.receipts] == [
        8.0,
        8.0,
    ]

    frozen = harness.run(
        record=record,
        policy=FrozenScoreTopKReplayPolicy(),
        budget=2,
    )
    # The stale frozen score prefers dominated c3 first: 0 + 8 = 8.
    assert frozen.receipts[0].action_sha256 == _sha("replay:c3")
    assert frozen.realized_gain == 8.0
    assert frozen.regret == 8.0

    assert set(native.oracle_subset) == {
        _sha("replay:c1"),
        _sha("replay:c2"),
    }


def test_uniform_random_policy_logs_exact_support_propensity() -> None:
    record = _tiny_market()
    result = SequentialMarketReplay().run(
        record=record,
        policy=UniformRandomReplayPolicy(seed=5),
        budget=2,
    )

    assert result.receipts[0].selection_propensity == 1.0 / 3.0
    assert result.receipts[1].selection_propensity == 1.0 / 2.0


def test_lane_heads_policy_reproduces_frozen_head_walk() -> None:
    record = MarketRecord(
        market_id="lane.heads",
        archive_points=(_point(4.0, 4.0),),
        hv_reference_point=_REFERENCE,
        candidates=(
            _candidate(
                "a1",
                engine_id="engine.a",
                native_rank=1,
                frozen_score=0.3,
                objectives=_point(2.0, 6.0),
            ),
            _candidate(
                "a2",
                engine_id="engine.a",
                native_rank=2,
                frozen_score=0.8,
                objectives=_point(6.0, 2.0),
            ),
            _candidate(
                "b1",
                engine_id="engine.b",
                native_rank=1,
                frozen_score=0.5,
                objectives=_point(3.0, 3.0),
            ),
        ),
    )
    result = SequentialMarketReplay().run(
        record=record,
        policy=LaneHeadsReplayPolicy(),
        budget=3,
    )

    # Engine order by best frozen score: a (0.8) then b (0.5); heads
    # first, then depth two of the walk.
    assert result.selected_action_sha256s == (
        _sha("replay:a2"),
        _sha("replay:b1"),
        _sha("replay:a1"),
    )


def _v8lite_policy() -> V8LiteAllocationPolicy:
    return V8LiteAllocationPolicy(
        archive_gain_utility=ExactHypervolumeGainPort(_REFERENCE),
        config=V8LiteAllocationConfig(
            diagnostic_slots=2,
            reference_gain_scale=0.001,
            reference_gain_evidence_sha256=_sha("prior-evidence"),
            random_seed=17,
        ),
    )


_WIDE_MARKET_SHAPE: tuple[tuple[str, str, int, float], ...] = (
    ("a1", "engine.a", 1, 0.9),
    ("a2", "engine.a", 2, 0.6),
    ("a3", "engine.a", 3, 0.5),
    ("a4", "engine.a", 4, 0.4),
    ("b1", "engine.b", 1, 0.7),
    ("b2", "engine.b", 2, 0.3),
)

_WIDE_MARKET_POINTS: dict[str, tuple[tuple[str, float], ...]] = {
    "a1": _point(2.0, 6.0),
    "a2": _point(3.0, 5.0),
    "a3": _point(9.0, 9.0),
    "a4": _point(9.5, 9.5),
    "b1": _point(6.0, 2.0),
    "b2": _point(7.0, 3.0),
}


def _wide_market(
    *,
    swap_labels: tuple[str, str] | None = None,
) -> MarketRecord:
    points = dict(_WIDE_MARKET_POINTS)
    if swap_labels is not None:
        left, right = swap_labels
        points[left], points[right] = points[right], points[left]
    return MarketRecord(
        market_id="wide.market",
        archive_points=(_point(4.0, 4.0),),
        hv_reference_point=_REFERENCE,
        candidates=tuple(
            _candidate(
                label,
                engine_id=engine_id,
                native_rank=native_rank,
                frozen_score=frozen_score,
                objectives=points[label],
            )
            for label, engine_id, native_rank, frozen_score in (
                _WIDE_MARKET_SHAPE
            )
        ),
    )


def test_v8lite_adapter_runs_all_phases_within_the_boundary() -> None:
    record = _wide_market()
    result = SequentialMarketReplay().run(
        record=record,
        policy=V8LiteReplayPolicy(_v8lite_policy()),
        budget=4,
    )

    assert len(result.receipts) == 4
    assert len(set(result.selected_action_sha256s)) == 4
    assert all(
        0.0 < value.selection_propensity <= 1.0
        for value in result.receipts
    )
    assert result.regret == pytest.approx(
        result.oracle_gain - result.realized_gain
    )
    assert result.oracle_gain >= result.realized_gain


def test_permuting_unrevealed_outcomes_never_changes_selections() -> None:
    # The no-future-leak property: swapping the outcomes of candidates
    # the policy never selected must not change any selection.  The
    # swapped pair is chosen from the realized unselected support, so
    # the property is exercised regardless of what the policy picked.
    sha_to_label = {
        _sha(f"replay:{label}"): label
        for label, _engine, _rank, _score in _WIDE_MARKET_SHAPE
    }
    for policy_factory in (
        lambda: V8LiteReplayPolicy(_v8lite_policy()),
        lambda: UniformRandomReplayPolicy(seed=9),
        NativeRankRoundRobinReplayPolicy,
    ):
        base = SequentialMarketReplay().run(
            record=_wide_market(),
            policy=policy_factory(),
            budget=3,
        )
        unselected = tuple(
            sorted(
                label
                for sha, label in sha_to_label.items()
                if sha not in set(base.selected_action_sha256s)
            )
        )
        assert len(unselected) >= 2
        permuted = SequentialMarketReplay().run(
            record=_wide_market(
                swap_labels=(unselected[0], unselected[1])
            ),
            policy=policy_factory(),
            budget=3,
        )
        assert permuted.selected_action_sha256s == (
            base.selected_action_sha256s
        )


def test_unevaluated_selection_is_imputed_zero_when_unrestricted() -> None:
    record = MarketRecord(
        market_id="imputed.market",
        archive_points=(_point(4.0, 4.0),),
        hv_reference_point=_REFERENCE,
        candidates=(
            _candidate(
                "known",
                native_rank=1,
                frozen_score=0.1,
                objectives=_point(2.0, 6.0),
            ),
            _candidate(
                "unknown",
                native_rank=2,
                frozen_score=0.9,
                objectives=None,
                evaluated=False,
            ),
        ),
    )
    restricted = SequentialMarketReplay().run(
        record=record,
        policy=FrozenScoreTopKReplayPolicy(),
        budget=1,
    )
    assert restricted.receipts[0].action_sha256 == (
        _sha("replay:known")
    )

    unrestricted = SequentialMarketReplay().run(
        record=record,
        policy=FrozenScoreTopKReplayPolicy(),
        budget=1,
        restrict_to_evaluated=False,
    )
    receipt = unrestricted.receipts[0]
    assert receipt.action_sha256 == _sha("replay:unknown")
    assert receipt.imputed_zero_outcome is True
    assert receipt.marginal_gain == 0.0
    assert unrestricted.oracle_gain == restricted.oracle_gain


def test_oracle_refuses_beyond_the_enumeration_limit() -> None:
    record = _wide_market()
    # The positive non-dominated frontier is {a1, a2, b1}; choosing two
    # of three exceeds a limit of two subsets.
    with pytest.raises(ValueError, match="subset limit"):
        SequentialMarketReplay(oracle_subset_limit=2).oracle(
            record,
            2,
        )


def test_oracle_frontier_reduction_matches_brute_enumeration() -> None:
    record = _wide_market()
    harness = SequentialMarketReplay()
    evaluated = tuple(
        value
        for value in record.candidates
        if value.feasible and value.objectives is not None
    )
    base = record.hypervolume()
    for budget in (1, 2, 3):
        brute = max(
            record.hypervolume(
                tuple(value.objectives for value in subset)
            )
            - base
            for subset in itertools.combinations(
                evaluated,
                min(budget, len(evaluated)),
            )
        )
        reduced_gain, subset = harness.oracle(record, budget)
        assert reduced_gain == pytest.approx(brute, abs=1e-15)
        assert len(subset) <= budget


def test_archive_frontier_duplicate_yields_exact_zero_gain() -> None:
    # A candidate lying exactly on the archive frontier must produce an
    # exactly zero oracle and zero replay marginal, not float noise.
    record = MarketRecord(
        market_id="duplicate.market",
        archive_points=(_point(4.0, 4.0), _point(2.0, 6.0)),
        hv_reference_point=_REFERENCE,
        candidates=(
            _candidate(
                "duplicate",
                native_rank=1,
                frozen_score=0.9,
                objectives=_point(4.0, 4.0),
            ),
            _candidate(
                "dominated",
                native_rank=2,
                frozen_score=0.1,
                objectives=_point(5.0, 5.0),
            ),
        ),
    )
    oracle_gain, subset = SequentialMarketReplay().oracle(record, 2)
    assert oracle_gain == 0.0
    assert subset == ()
    result = SequentialMarketReplay().run(
        record=record,
        policy=FrozenScoreTopKReplayPolicy(),
        budget=2,
    )
    assert result.realized_gain == 0.0
    assert all(
        value.marginal_gain == 0.0 for value in result.receipts
    )
    assert result.regret == 0.0
    port = ExactHypervolumeGainPort(_REFERENCE)
    assert (
        port.marginal_archive_gain(
            record.archive_points,
            _point(4.0, 4.0),
        )
        == 0.0
    )


def test_corpus_loader_handles_real_schema_shapes() -> None:
    payload = {
        "market_id": "synthetic.market",
        "hv_reference_point": {
            "axes": [
                {
                    "metric_id": "m.x",
                    "goal": "min",
                    "ideal": 0.0,
                    "reference": 10.0,
                },
                {
                    "metric_id": "m.y",
                    "goal": "min",
                    "ideal": 0.0,
                    "reference": 10.0,
                },
            ]
        },
        "archive_at_market": {"points": [{"m.x": 4.0, "m.y": 4.0}]},
        "candidates": [
            {
                # lane_id present and wins over expert_id.
                "action_sha256": _sha("corpus:one"),
                "lane": {"lane_id": "engine.lane", "lane_size": 2},
                "expert_id": "engine.expert",
                "native_rank": 5,
                "frozen_scores": {"frozen_prior_score": 3.0},
                "evaluated": True,
                "valid": True,
                "objectives": {"m.x": 2.0, "m.y": 6.0},
                "forecast": [
                    {
                        "metric_id": "m.x",
                        "confidence": 0.8,
                        "p10_delta": -2.0,
                        "p50_delta": -1.0,
                        "p90_delta": 0.0,
                    },
                    {
                        "metric_id": "m.y",
                        "confidence": 0.6,
                        "p10_delta": -1.0,
                        "p50_delta": 0.0,
                        "p90_delta": 1.0,
                    },
                ],
                "parent": {
                    "candidate_id": "parent.one",
                    "objectives": {"m.x": 3.0, "m.y": 6.0},
                },
            },
            {
                # No action sha: derived id; expert_id fallback;
                # rank gap densified; invalid despite objectives.
                "candidate_id": "candidate.two",
                "expert_id": "engine.expert",
                "native_rank": 9,
                "frozen_scores": {"frozen_prior_score": 1.0},
                "evaluated": True,
                "valid": False,
                "objectives": {"m.x": 1.0, "m.y": 1.0},
                "forecast": [
                    {"direction": "decrease", "metric_id": "m.x"}
                ],
                "parent": None,
            },
            {
                # family fallback, no frozen score, unevaluated.
                "proposal_id": "proposal.three",
                "family": "engine.family",
                "native_rank": 1,
                "evaluated": False,
                "objectives": None,
            },
        ],
    }
    record = market_record_from_corpus(payload)

    by_engine = {
        value.engine_id: value for value in record.candidates
    }
    assert set(by_engine) == {
        "engine.lane",
        "engine.expert",
        "engine.family",
    }
    lane_one = by_engine["engine.lane"]
    assert lane_one.native_rank == 1
    # Frozen scores are min-max normalized: 3.0 -> 1.0, 1.0 -> 0.0.
    assert lane_one.frozen_score == 1.0
    assert by_engine["engine.expert"].frozen_score == 0.0
    assert by_engine["engine.family"].frozen_score is None
    # The invalid candidate is infeasible even though it recorded
    # objectives, and never enters the oracle.
    assert by_engine["engine.expert"].feasible is False
    assert by_engine["engine.expert"].objectives is None
    assert len(by_engine["engine.expert"].action_sha256) == 64
    # Delta forecast joined to parent objectives, normalized: parent
    # (3, 6) with p50 deltas (-1, 0) gives (2, 6) -> (0.2, 0.6).
    forecast = lane_one.forecast
    assert forecast is not None
    assert forecast.reliability == pytest.approx(0.7)
    assert forecast.point("p50") == _point(0.2, 0.6)
    # Direction-only forecast degrades to none.
    assert by_engine["engine.expert"].forecast is None
    oracle_gain, subset = SequentialMarketReplay().oracle(record, 2)
    assert subset == (lane_one.action_sha256,)
    assert oracle_gain == pytest.approx(8.0 / 100.0)


_CORPUS_DIR = pathlib.Path(
    "/home/yigit/repos/research_stuff/papers/agent_evolve_aaai_2027/"
    "research_artifacts/data/jul27_allocator_replay_corpus_v1"
)


@pytest.mark.skipif(
    not (_CORPUS_DIR / "boils_v70_qwen_support_horizon_b32.json").exists(),
    reason="jul27 replay corpus is not present on this machine",
)
def test_corpus_loader_reproduces_recorded_archive_hypervolume() -> None:
    payload = json.loads(
        (
            _CORPUS_DIR / "boils_v70_qwen_support_horizon_b32.json"
        ).read_text()
    )
    record = market_record_from_corpus(payload)

    assert record.hypervolume() == pytest.approx(
        payload["archive_hv"],
        rel=1e-9,
    )
    assert len(record.candidates) == len(payload["candidates"])
    lane_sizes: dict[str, int] = {}
    for value in record.candidates:
        lane_sizes[value.engine_id] = (
            lane_sizes.get(value.engine_id, 0) + 1
        )
    for value in record.candidates:
        assert 1 <= value.native_rank <= lane_sizes[value.engine_id]

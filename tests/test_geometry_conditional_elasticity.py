from __future__ import annotations

import hashlib
import math
from fractions import Fraction

from agent_evolve.application.geometry_conditional_elasticity import (
    ElasticSeatBidder,
    ElasticSeatConfig,
    LaneGeometryEvidence,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionOutcome,
)
from agent_evolve.application.region_conditional_credit import (
    RegionFeatures,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LiteAllocationConfig,
)
from agent_evolve.application.v9_candidate_policy import (
    V9CandidateConfig,
    V9CandidatePolicy,
)
from agent_evolve.domain.typed_json import thaw_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _point(x: float, y: float) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((("m.x", float(x)), ("m.y", float(y)))))


def _hv2(
    points: list[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    kept = sorted(
        {
            point
            for point in points
            if point[0] < reference[0] and point[1] < reference[1]
        }
    )
    skyline: list[tuple[float, float]] = []
    best_y = math.inf
    for point in kept:
        if point[1] < best_y:
            skyline.append(point)
            best_y = point[1]
    total = 0.0
    for index, (x, y) in enumerate(skyline):
        next_x = (
            skyline[index + 1][0]
            if index + 1 < len(skyline)
            else reference[0]
        )
        total += (next_x - x) * (reference[1] - y)
    return total


class _Hv2GainPort:
    utility_id = "test.exact_hv2"
    utility_version = 1
    definition_sha256 = _sha("test-exact-hv2")

    def __init__(self, reference: tuple[float, float]) -> None:
        self._reference = reference

    def marginal_archive_gain(
        self,
        archive_points: tuple[tuple[tuple[str, float], ...], ...],
        objective_point: tuple[tuple[str, float], ...],
    ) -> float:
        archive = [
            (dict(point)["m.x"], dict(point)["m.y"])
            for point in archive_points
        ]
        candidate = (
            dict(objective_point)["m.x"],
            dict(objective_point)["m.y"],
        )
        return float(
            _hv2([*archive, candidate], self._reference)
            - _hv2(archive, self._reference)
        )


_REFERENCE = _point(1.0, 1.0)
_ARCHIVE = (_point(0.4, 0.4),)


def _lane(
    engine_id: str,
    *,
    distances: tuple[float, ...] = (),
    predicted_dominated: tuple[bool, ...] = (),
    revealed_positive: tuple[bool, ...] = (),
) -> LaneGeometryEvidence:
    return LaneGeometryEvidence(
        engine_id=engine_id,
        parent_front_distances=distances,
        predicted_dominated=predicted_dominated,
        revealed_positive=revealed_positive,
    )


def test_far_front_parents_bid_below_near_front_parents() -> None:
    bidder = ElasticSeatBidder()
    bids = {
        value.engine_id: value
        for value in bidder.lane_bids(
            (
                _lane("engine.near", distances=(0.01, 0.02)),
                _lane("engine.far", distances=(0.5, 0.7)),
            )
        )
    }
    assert bids["engine.near"].bid > bids["engine.far"].bid
    assert bids["engine.far"].mean_parent_front_distance == 0.6
    assert bids["engine.near"].distance_factor > bids[
        "engine.far"
    ].distance_factor


def test_saturated_forecast_self_overlap_bids_low() -> None:
    bidder = ElasticSeatBidder()
    bids = {
        value.engine_id: value
        for value in bidder.lane_bids(
            (
                _lane(
                    "engine.novel",
                    predicted_dominated=(False, False, False),
                ),
                _lane(
                    "engine.saturated",
                    predicted_dominated=(True, True, True),
                ),
            )
        )
    }
    assert (
        bids["engine.novel"].bid > bids["engine.saturated"].bid
    )
    assert (
        bids["engine.saturated"].saturation_posterior
        > bids["engine.novel"].saturation_posterior
    )


def test_revealed_conversion_moves_bids() -> None:
    bidder = ElasticSeatBidder()
    bids = {
        value.engine_id: value
        for value in bidder.lane_bids(
            (
                _lane(
                    "engine.converting",
                    revealed_positive=(True, True),
                ),
                _lane(
                    "engine.dead",
                    revealed_positive=(False, False),
                ),
            )
        )
    }
    assert (
        bids["engine.converting"].bid > bids["engine.dead"].bid
    )


def test_global_exploration_floor_keeps_every_bid_positive() -> None:
    config = ElasticSeatConfig()
    bidder = ElasticSeatBidder(config=config)
    bids = bidder.lane_bids(
        (
            _lane(
                "engine.hopeless",
                distances=(5.0, 5.0),
                predicted_dominated=(True, True, True, True),
                revealed_positive=(False, False, False),
            ),
            _lane("engine.fine", distances=(0.0,)),
        )
    )
    by_engine = {value.engine_id: value for value in bids}
    assert (
        by_engine["engine.hopeless"].bid
        == config.exploration_floor_bid
    )
    assert by_engine["engine.hopeless"].floored
    assert not by_engine["engine.fine"].floored
    for value in bids:
        assert value.bid >= config.exploration_floor_bid > 0.0


def test_dhondt_walk_gives_far_lane_fewer_seats_floor_reached() -> None:
    config = ElasticSeatConfig()
    bidder = ElasticSeatBidder(config=config)
    bids = bidder.lane_bids(
        (
            _lane("engine.near", distances=(0.01, 0.02)),
            _lane("engine.far", distances=(0.5, 0.7)),
        )
    )
    awarded = {"engine.near": 0, "engine.far": 0}
    open_ids = frozenset(awarded)
    for _seat in range(4):
        chosen = bidder.choose_engine(
            bids=bids,
            seats_awarded=awarded,
            open_engine_ids=open_ids,
        )
        awarded[chosen] += 1
    # The near-front lane out-earns the far lane, which is NOT starved
    # forever: the positive floor bid eventually seats it.
    assert awarded["engine.near"] > awarded["engine.far"]
    assert awarded["engine.far"] >= 1


def test_dhondt_tie_breaks_by_ascending_engine_id() -> None:
    bidder = ElasticSeatBidder()
    bids = bidder.lane_bids(
        (_lane("engine.b"), _lane("engine.a"))
    )
    chosen = bidder.choose_engine(
        bids=bids,
        seats_awarded={},
        open_engine_ids=frozenset({"engine.a", "engine.b"}),
    )
    assert chosen == "engine.a"


def test_definition_sha_binds_config() -> None:
    default = ElasticSeatBidder()
    heavier = ElasticSeatBidder(
        config=ElasticSeatConfig(distance_weight=8.0),
    )
    assert default.policy_id == "geometry_conditional_elasticity"
    assert default.definition_sha256 != heavier.definition_sha256
    assert (
        default.definition_sha256
        == ElasticSeatBidder().definition_sha256
    )


_LANE_SIZE = 5
_SLOTS = 10


def _market() -> tuple[AdaptiveActionDescriptor, ...]:
    # Two lanes of five: at ten slots the r2 pilot width cap is
    # min(4, 9, max(2, ceil(10 / 3))) = 4, so a four-seat walk fits.
    values: list[AdaptiveActionDescriptor] = []
    for lane in ("engine.far", "engine.near"):
        for rank in range(1, _LANE_SIZE + 1):
            values.append(
                AdaptiveActionDescriptor(
                    action_sha256=_sha(f"action:{lane}:{rank}"),
                    phenotype_sha256=_sha(
                        f"phenotype:{lane}:{rank}"
                    ),
                    lane_id=lane,
                    operator_id=lane,
                    native_rank=rank,
                    lane_size=_LANE_SIZE,
                    prior_score=float(
                        _LANE_SIZE + 1 - rank
                    ) / float(_LANE_SIZE),
                    parent_generated_in_current_run=False,
                )
            )
    return tuple(values)


def _region_features() -> tuple[tuple[str, RegionFeatures], ...]:
    features: list[tuple[str, RegionFeatures]] = []
    for lane, parent in (
        ("engine.far", _point(0.9, 0.9)),
        ("engine.near", _point(0.41, 0.41)),
    ):
        for rank in range(1, _LANE_SIZE + 1):
            features.append(
                (
                    _sha(f"action:{lane}:{rank}"),
                    RegionFeatures(parent_point=parent, radius=2),
                )
            )
    return tuple(sorted(features))


def _v9(r3: bool = True) -> V9CandidatePolicy:
    return V9CandidatePolicy(
        archive_gain_utility=_Hv2GainPort((1.0, 1.0)),
        config=V9CandidateConfig(
            r3_geometry_conditional_elasticity=r3,
            base=V8LiteAllocationConfig(random_seed=5),
        ),
    )


def _walk_pilot(
    policy: V9CandidatePolicy,
    seats: int,
    *,
    revealed_gain: float,
) -> tuple[list[str], list[object]]:
    market = _market()
    by_action = {value.action_sha256: value for value in market}
    selected: tuple[str, ...] = ()
    outcomes: tuple[AdaptiveActionOutcome, ...] = ()
    lanes: list[str] = []
    decisions: list[object] = []
    for _seat in range(seats):
        decision = policy.design_pilot_seat(
            residual_request_sha256=_sha("request"),
            actions=market,
            evaluation_slots=_SLOTS,
            selected_action_sha256s=selected,
            outcomes=outcomes,
            archive_points=_ARCHIVE,
            reference_point=_REFERENCE,
            region_features=_region_features(),
        )
        chosen = decision.selected_action_sha256s[0]
        lanes.append(by_action[chosen].lane_id)
        decisions.append(decision)
        outcomes = (
            *outcomes,
            AdaptiveActionOutcome(
                action_sha256=chosen,
                evaluation_sha256=_sha(f"evaluation:{chosen}"),
                feasible=True,
                marginal_archive_gain=revealed_gain,
            ),
        )
        selected = tuple(sorted((*selected, chosen)))
    return lanes, decisions


def test_v9_r3_seats_favor_near_front_lane_over_coverage_floor() -> None:
    # Converting near-front lane: every early seat is earned by its
    # bid.  The fixed coverage floor would have seated the far lane by
    # seat two; the elastic walk never does.
    lanes, decisions = _walk_pilot(
        _v9(),
        seats=3,
        revealed_gain=0.01,
    )
    assert lanes == ["engine.near"] * 3
    for decision in decisions:
        evidence = thaw_json(decision.evidence)
        assert evidence["fixed_coverage_floor_used"] is False
        bids = {
            value["engine_id"]: float.fromhex(value["bid_hex"])
            for value in evidence["elastic_lane_bids"]
        }
        assert bids["engine.near"] > bids["engine.far"]


def test_v9_r3_floor_still_seats_the_far_lane_eventually() -> None:
    # Zero-gain reveals decay the near lane's posterior, and the far
    # lane's FLOORED bid wins seats through the D'Hondt divisors: fewer
    # seats than the near lane, but never starved out entirely.
    lanes, decisions = _walk_pilot(
        _v9(),
        seats=3,
        revealed_gain=0.0,
    )
    assert lanes.count("engine.near") > lanes.count("engine.far")
    assert "engine.far" in lanes
    for decision in decisions:
        evidence = thaw_json(decision.evidence)
        far = next(
            value
            for value in evidence["elastic_lane_bids"]
            if value["engine_id"] == "engine.far"
        )
        assert far["floored"] is True
        assert float.fromhex(far["bid_hex"]) == 0.0625


def test_v9_r3_seat_propensities_remain_exact_rationals() -> None:
    _lanes, decisions = _walk_pilot(
        _v9(),
        seats=3,
        revealed_gain=0.01,
    )
    for decision in decisions:
        evidence = thaw_json(decision.evidence)
        seat = evidence["pilot_seat"]
        total = Fraction(0)
        for value in seat["support_propensities"]:
            share = Fraction(
                value["propensity_numerator"],
                value["propensity_denominator"],
            )
            assert share > 0
            total += share
        assert total == Fraction(1)
        assert decision.selection_propensity == float(
            Fraction(
                next(
                    value["propensity_numerator"]
                    for value in seat["support_propensities"]
                    if value["action_sha256"]
                    == decision.selected_action_sha256s[0]
                ),
                next(
                    value["propensity_denominator"]
                    for value in seat["support_propensities"]
                    if value["action_sha256"]
                    == decision.selected_action_sha256s[0]
                ),
            )
        )

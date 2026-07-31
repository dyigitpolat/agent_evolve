from __future__ import annotations

import hashlib
from fractions import Fraction

import pytest

from agent_evolve.application.rank_balanced_causal_pilot import (
    DEFAULT_PILOT_BAND_WEIGHTS,
    PilotSeatObservation,
    RankBalancedCausalPilotPolicy,
    RankBalancedPilotCandidate,
    SequentialAdaptiveBandPilotPolicy,
    rank_band_index,
    rank_band_schedule,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _candidate(
    engine_id: str,
    native_rank: int,
    *,
    frozen_score: float | None,
) -> RankBalancedPilotCandidate:
    return RankBalancedPilotCandidate(
        action_sha256=_sha(f"pilot:{engine_id}:{native_rank}"),
        engine_id=engine_id,
        native_rank=native_rank,
        frozen_score=frozen_score,
    )


def _market(
    lane_sizes: dict[str, int],
    *,
    frozen: bool = True,
) -> tuple[RankBalancedPilotCandidate, ...]:
    values: list[RankBalancedPilotCandidate] = []
    for engine_id, lane_size in sorted(lane_sizes.items()):
        for rank in range(1, lane_size + 1):
            values.append(
                _candidate(
                    engine_id,
                    rank,
                    frozen_score=(
                        # A deliberately stale frozen order: the frozen
                        # score prefers the lane TAIL over the head.
                        float(rank) / lane_size if frozen else None
                    ),
                )
            )
    return tuple(values)


def test_validated_six_item_two_band_schedule_is_preserved() -> None:
    assert rank_band_schedule(6, 2) == (1, 4, 2, 5, 3, 6)


def test_three_band_schedule_interleaves_band_heads() -> None:
    assert rank_band_schedule(6, 3) == (1, 3, 5, 2, 4, 6)
    assert rank_band_schedule(5, 3) == (1, 3, 5, 2, 4)
    assert rank_band_schedule(2, 3) == (1, 2)
    assert rank_band_index(1, 6, 3) == 0
    assert rank_band_index(4, 6, 3) == 1
    assert rank_band_index(6, 6, 3) == 2


def test_weighted_schedule_concentrates_mass_on_top_bands() -> None:
    weighted = rank_band_schedule(6, 3, DEFAULT_PILOT_BAND_WEIGHTS)
    assert weighted == (1, 3, 2, 5, 4, 6)
    # Every band keeps a nonzero visitation floor.
    assert set(weighted) == {1, 2, 3, 4, 5, 6}
    with pytest.raises(ValueError, match="sum to exactly one"):
        rank_band_schedule(6, 3, (0.5, 0.3125, 0.125))


def test_every_engine_gets_a_seat_when_width_covers_engines() -> None:
    market = _market({"alpha": 6, "beta": 3, "gamma": 2})
    design = RankBalancedCausalPilotPolicy(
        random_seed=7
    ).design_pilot(
        residual_request_sha256=_sha("request"),
        candidates=market,
        pilot_width=4,
    )

    by_action = {value.action_sha256: value for value in market}
    engines = {
        by_action[value.selected_action_sha256].engine_id
        for value in design.seats
    }
    assert engines == {"alpha", "beta", "gamma"}
    # The extra D'Hondt seat goes to the widest engine.
    assert (
        sum(value.engine_id == "alpha" for value in design.seats) == 2
    )


def test_seat_propensities_are_exact_and_sum_to_one() -> None:
    market = _market({"alpha": 6, "beta": 3})
    design = RankBalancedCausalPilotPolicy(
        exploration_epsilon=0.125,
        random_seed=11,
    ).design_pilot(
        residual_request_sha256=_sha("request"),
        candidates=market,
        pilot_width=5,
    )

    for seat in design.seats:
        total = sum(
            (value.exact for value in seat.support_propensities),
            Fraction(0),
        )
        assert total == Fraction(1)
        for value in seat.support_propensities:
            assert value.propensity == float(value.exact)
        assert seat.selection_propensity > 0.0
    assert 0.0 < design.design_propensity <= 1.0


def test_block_randomization_splits_directed_mass_between_orders() -> None:
    market = _market({"alpha": 6})
    design = RankBalancedCausalPilotPolicy(
        exploration_epsilon=0.125,
        random_seed=3,
    ).design_pilot(
        residual_request_sha256=_sha("request"),
        candidates=market,
        pilot_width=2,
    )

    first_seat = design.seats[0]
    by_action = {value.action_sha256: value for value in market}
    # Band zero of a six-item three-band lane holds native ranks 1 and 2.
    # The stale frozen score prefers rank 2, so the two heads disagree and
    # the block-first seat must split its directed mass between them.
    halves = {
        by_action[value.action_sha256].native_rank: value.exact
        for value in first_seat.support_propensities
        if value.exact > Fraction(1, 8) / 6
    }
    expected = Fraction(7, 8) / 2 + Fraction(1, 8) / 6
    assert halves == {1: expected, 2: expected}
    # The block-second seat is deterministic given the logged block order.
    second_seat = design.seats[1]
    assert second_seat.directed_order.startswith("block_second")
    assert (
        max(
            value.exact
            for value in second_seat.support_propensities
        )
        == Fraction(7, 8) + Fraction(1, 8) / 5
    )


def test_band_schedule_purchases_top_and_interior_ranks() -> None:
    market = _market({"alpha": 6}, frozen=False)
    design = RankBalancedCausalPilotPolicy(
        exploration_epsilon=0.0,
        random_seed=0,
    ).design_pilot(
        residual_request_sha256=_sha("request"),
        candidates=market,
        pilot_width=6,
    )

    by_action = {value.action_sha256: value for value in market}
    ranks = tuple(
        by_action[value.selected_action_sha256].native_rank
        for value in design.seats
    )
    # Without frozen scores the directed schedule is the pure weighted
    # band walk: top-heavy, but every band retains its probe floor.
    assert ranks == (1, 3, 2, 5, 4, 6)
    assert ranks[:2] == (1, 3)
    assert {seat.effective_band_index for seat in design.seats} == {
        0,
        1,
        2,
    }


def test_first_lane_seat_resists_stale_frozen_rank_inversion() -> None:
    # The exact V70 failure: a stale frozen consequence score promoted
    # native rank 11 of a large lane over native ranks 1-5 even though
    # the engine's own top ranks held the positives.  Under the default
    # config the first seat of such a lane must place all of its
    # directed mass inside the TOP native-rank band; deep ranks keep
    # only the uniform exploration floor.
    lane_size = 12
    inverted = tuple(
        RankBalancedPilotCandidate(
            action_sha256=_sha(f"inverted:{rank}"),
            engine_id="interaction.lane",
            native_rank=rank,
            # Stale frozen order peaks at native rank 11.
            frozen_score=1.0 - abs(rank - 11) / lane_size,
        )
        for rank in range(1, lane_size + 1)
    )
    by_action = {value.action_sha256: value for value in inverted}
    policy_epsilon = 0.125
    top_band_ranks = {1, 2, 3, 4}
    directed_seen = False
    for seed in range(24):
        design = RankBalancedCausalPilotPolicy(
            random_seed=seed
        ).design_pilot(
            residual_request_sha256=_sha("v70-regression"),
            candidates=inverted,
            pilot_width=1,
        )
        seat = design.seats[0]
        assert seat.effective_band_index == 0
        for value in seat.support_propensities:
            if by_action[value.action_sha256].native_rank not in (
                top_band_ranks
            ):
                # Deep ranks carry exactly the exploration floor.
                assert value.exact == (
                    Fraction(policy_epsilon) / lane_size
                )
        if seat.branch == "directed":
            directed_seen = True
            assert by_action[
                seat.selected_action_sha256
            ].native_rank in top_band_ranks
    assert directed_seen


def test_design_is_deterministic_given_seed_and_market() -> None:
    market = _market({"alpha": 5, "beta": 4})
    policy = RankBalancedCausalPilotPolicy(random_seed=42)
    first = policy.design_pilot(
        residual_request_sha256=_sha("request"),
        candidates=market,
        pilot_width=4,
    )
    second = policy.design_pilot(
        residual_request_sha256=_sha("request"),
        candidates=tuple(reversed(market)),
        pilot_width=4,
    )
    other_seed = RankBalancedCausalPilotPolicy(
        random_seed=43
    ).design_pilot(
        residual_request_sha256=_sha("request"),
        candidates=market,
        pilot_width=4,
    )

    assert first.design_sha256 == second.design_sha256
    assert first.selected_action_sha256s == (
        second.selected_action_sha256s
    )
    assert other_seed.policy_definition_sha256 != (
        first.policy_definition_sha256
    )


def _seq_policy(**overrides: object) -> "SequentialAdaptiveBandPilotPolicy":
    values: dict[str, object] = {"random_seed": 5}
    values.update(overrides)
    return SequentialAdaptiveBandPilotPolicy(**values)


def _observation(
    action_sha256: str,
    gain: float,
) -> PilotSeatObservation:
    return PilotSeatObservation(
        action_sha256=action_sha256,
        feasible=True,
        marginal_archive_gain=float(gain),
    )


def _band_mass(
    seat,
    by_action,
    lane_size: int,
    band: int,
) -> Fraction:
    return sum(
        (
            value.exact
            for value in seat.support_propensities
            if rank_band_index(
                by_action[value.action_sha256].native_rank,
                lane_size,
                3,
            )
            == band
        ),
        Fraction(0),
    )


def test_sequential_pilot_concentrates_mass_when_heads_convert() -> None:
    market = _market({"alpha": 6, "beta": 6}, frozen=False)
    by_action = {value.action_sha256: value for value in market}
    policy = _seq_policy()
    request = _sha("heads-hot")
    alpha_head = next(
        value
        for value in market
        if value.engine_id == "alpha" and value.native_rank == 1
    )
    cold_seat = policy.design_seat(
        residual_request_sha256=request,
        candidates=market,
        selected_action_sha256s=(alpha_head.action_sha256,),
        observations=(
            _observation(alpha_head.action_sha256, 0.0),
        ),
        seat_ordinal=2,
    )
    hot_seat = policy.design_seat(
        residual_request_sha256=request,
        candidates=market,
        selected_action_sha256s=(alpha_head.action_sha256,),
        observations=(
            _observation(alpha_head.action_sha256, 0.02),
        ),
        seat_ordinal=2,
    )

    # Both seats sit on the unseated beta engine (coverage floor).
    assert cold_seat.engine_id == "beta"
    assert hot_seat.engine_id == "beta"
    cold_top = _band_mass(cold_seat, by_action, 6, 0)
    hot_top = _band_mass(hot_seat, by_action, 6, 0)
    cold_interior = _band_mass(cold_seat, by_action, 6, 1) + (
        _band_mass(cold_seat, by_action, 6, 2)
    )
    hot_interior = _band_mass(hot_seat, by_action, 6, 1) + (
        _band_mass(hot_seat, by_action, 6, 2)
    )
    # A revealed top-band success concentrates mass toward heads; a
    # revealed top-band zero preserves (raises) interior exploration.
    assert hot_top > cold_top
    assert cold_interior > hot_interior
    # The exploration floor keeps every candidate at nonzero mass.
    for seat in (cold_seat, hot_seat):
        total = sum(
            (value.exact for value in seat.support_propensities),
            Fraction(0),
        )
        assert total == Fraction(1)
        assert all(
            value.exact > 0
            for value in seat.support_propensities
        )


def test_sequential_pilot_floor_ignores_lane_width() -> None:
    # A wide engine must not out-rank narrow engines at zero evidence;
    # the floor orders unseated engines by posterior then engine id.
    market = _market({"aaa": 1, "bbb": 1, "zzz.wide": 6}, frozen=False)
    by_action = {value.action_sha256: value for value in market}
    seat = _seq_policy().design_seat(
        residual_request_sha256=_sha("width-blind"),
        candidates=market,
        selected_action_sha256s=(),
        observations=(),
        seat_ordinal=1,
    )
    assert seat.engine_id == "aaa"
    assert by_action[seat.selected_action_sha256].engine_id == "aaa"


def test_sequential_pilot_directs_extra_seats_by_posterior() -> None:
    # After the floor, extra seats go to the engine with the best
    # revealed conversion posterior, not the widest engine.
    market = _market({"alpha": 4, "beta": 4}, frozen=False)
    by_action = {value.action_sha256: value for value in market}
    heads = {
        engine: next(
            value
            for value in market
            if value.engine_id == engine and value.native_rank == 1
        )
        for engine in ("alpha", "beta")
    }
    selected = tuple(
        sorted(
            value.action_sha256 for value in heads.values()
        )
    )
    seat = _seq_policy().design_seat(
        residual_request_sha256=_sha("posterior-directed"),
        candidates=market,
        selected_action_sha256s=selected,
        observations=(
            _observation(heads["alpha"].action_sha256, 0.0),
            _observation(heads["beta"].action_sha256, 0.03),
        ),
        seat_ordinal=3,
    )
    assert seat.engine_id == "beta"


def test_mixed_frozen_score_presence_is_rejected_per_engine() -> None:
    market = (
        _candidate("alpha", 1, frozen_score=0.5),
        _candidate("alpha", 2, frozen_score=None),
    )
    with pytest.raises(ValueError, match="frozen scores for all"):
        RankBalancedCausalPilotPolicy().design_pilot(
            residual_request_sha256=_sha("request"),
            candidates=market,
            pilot_width=1,
        )


def test_pilot_width_cannot_exceed_the_market() -> None:
    market = _market({"alpha": 2})
    with pytest.raises(ValueError, match="fit the candidate market"):
        RankBalancedCausalPilotPolicy().design_pilot(
            residual_request_sha256=_sha("request"),
            candidates=market,
            pilot_width=3,
        )

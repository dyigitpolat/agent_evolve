"""The challenger's archive-geometry tie-break (v8lite r3).

The jul28 diagnosis measured that with no forecast supplied, every
probability input to the calibrated challenger is a per (engine, rank-band)
CELL constant: the score took 1.4 to 2.8 distinct values over markets of 37
to 64 members, 46-83% of every market tied with the argmax, and 23 of 28
live seats were decided by the TIE-BREAK rather than by the score.  The
tie-break in use was native-rank quality, a measured non-feature, and the
seats it bought were geometrically redundant.

These tests pin the repair where the decision is actually taken: among
candidates the score cannot separate, prefer the anchor farthest from what
this market already bought (provably submodular), then the anchor nearest
the current archive front.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    CalibratedPositiveGainOpportunityPolicy,
    ObservedConversionOutcome,
    PositiveGainCandidate,
    chebyshev_excess,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LITE_ALLOCATION_POLICY_VERSION_ID,
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R2,
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R3,
    V8LiteAllocationConfig,
    V8LiteAllocationPolicy,
)

METRIC_IDS = ("m.x", "m.y")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _point(x: float, y: float):
    return tuple(sorted((("m.x", float(x)), ("m.y", float(y)))))


class _NullGainPort:
    """Never consulted here: no candidate in these markets has a forecast."""

    utility_id = "test.null_gain"
    utility_version = 1
    definition_sha256 = _sha("test-null-gain")

    def marginal_archive_gain(self, archive_points, objective_point):
        return 0.0


def _policy(active: bool) -> CalibratedPositiveGainOpportunityPolicy:
    return CalibratedPositiveGainOpportunityPolicy(
        archive_gain_utility=_NullGainPort(),
        anchor_geometry_tie_break=active,
    )


def _candidate(label: str, anchor, *, engine_id="engine.a", rank=1, size=4):
    return PositiveGainCandidate(
        action_sha256=_sha(f"candidate:{label}"),
        engine_id=engine_id,
        native_rank=rank,
        lane_size=size,
        anchor_point=None if anchor is None else _point(*anchor),
    )


ARCHIVE = (_point(0.5, 0.5), _point(0.4, 0.7))
PRE_R3_SCORE_KEYS = {
    "schema_version",
    "action_sha256",
    "p_feasible_hex",
    "p_positive_archive_gain_hex",
    "expected_positive_gain_hex",
    "tail_risk_hex",
    "value_of_information_hex",
    "uncertainty_hex",
    "effective_sample_size_hex",
    "score_hex",
    "forecast_probability_hex",
    "forecast_magnitude_hex",
    "forecast_nondominated_fraction_hex",
    "conversion_probability_hex",
    "conversion_magnitude_hex",
    "frozen_score_probability_hex",
    "frozen_evidence_weight_hex",
    "abstained",
    "score_sha256",
}


def _rank(ranking, label):
    return ranking.ranked_action_sha256s.index(_sha(f"candidate:{label}"))


def test_tie_break_is_off_by_default_and_emits_no_new_bytes():
    off = _policy(False)
    assert off.anchor_geometry_tie_break is False
    ranking = off.score_market(
        candidates=(
            _candidate("a", (0.5, 0.5)),
            _candidate("b", (0.9, 0.9), rank=2),
        ),
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=1,
        horizon_total=2,
    )
    for card in ranking.scores:
        assert card.anchor_excess is None
        assert card.anchor_dispersion is None
        assert set(card.to_record()) == PRE_R3_SCORE_KEYS


def test_the_score_alone_cannot_separate_this_market_at_all():
    """The premise of the repair, asserted rather than assumed."""

    market = tuple(
        _candidate(label, anchor, rank=rank)
        for label, anchor, rank in (
            ("near", (0.45, 0.55), 1),
            ("mid", (0.7, 0.7), 2),
            ("far", (0.95, 0.95), 3),
            ("worst", (0.99, 0.99), 4),
        )
    )
    frozen = _policy(False).score_market(
        candidates=market,
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=4,
    )
    assert len({card.score for card in frozen.scores}) == 1
    assert len(
        {card.p_positive_archive_gain for card in frozen.scores}
    ) == 1


def test_untouched_market_breaks_ties_toward_the_front(  # noqa: D103
):
    market = tuple(
        _candidate(label, anchor, rank=rank)
        for label, anchor, rank in (
            ("near", (0.45, 0.55), 1),
            ("mid", (0.7, 0.7), 2),
            ("far", (0.95, 0.95), 3),
            ("worst", (0.99, 0.99), 4),
        )
    )
    fixed = _policy(True).score_market(
        candidates=market,
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=4,
    )
    assert fixed.ranked_action_sha256s == (
        _sha("candidate:near"),
        _sha("candidate:mid"),
        _sha("candidate:far"),
        _sha("candidate:worst"),
    )
    by_action = {card.action_sha256: card for card in fixed.scores}
    assert (
        by_action[_sha("candidate:near")].anchor_excess
        < by_action[_sha("candidate:far")].anchor_excess
    )
    # Nothing bought yet, so dispersion is infinite for everybody and the
    # front key alone decides.
    assert all(
        math.isinf(card.anchor_dispersion) for card in fixed.scores
    )


def test_a_bought_anchor_demotes_its_own_siblings_last():
    """Submodularity, stated as the property that actually matters.

    A second seat on an identical anchor has dispersion exactly zero, so it
    is taken after every candidate from a region this market has not paid
    to sample -- including candidates whose parents sit FURTHER from the
    front, which the untouched market ranked ahead of nothing.
    """

    market = tuple(
        _candidate(label, anchor, rank=rank)
        for label, anchor, rank in (
            ("sibling", (0.45, 0.55), 1),
            ("elsewhere", (0.70, 0.70), 2),
            ("distant", (0.95, 0.95), 3),
        )
    )
    policy = _policy(True)
    before = policy.score_market(
        candidates=market,
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=3,
    )
    assert before.ranked_action_sha256s[0] == _sha("candidate:sibling")

    after = policy.score_market(
        candidates=market,
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=1,
        horizon_total=3,
        covered_anchors=(_point(0.45, 0.55),),
    )
    assert _rank(after, "sibling") == 2
    assert _rank(after, "distant") < _rank(after, "sibling")
    by_action = {card.action_sha256: card for card in after.scores}
    assert by_action[_sha("candidate:sibling")].anchor_dispersion == 0.0
    assert by_action[_sha("candidate:elsewhere")].anchor_dispersion > 0.0
    # Buying a region cannot raise anyone's dispersion, and cannot lower
    # the dispersion of a candidate far from it below a near one's.
    assert (
        by_action[_sha("candidate:distant")].anchor_dispersion
        > by_action[_sha("candidate:elsewhere")].anchor_dispersion
    )


def test_dispersion_is_monotone_non_increasing_in_the_bought_set():
    """The formal submodularity property, over an arbitrary market."""

    market = tuple(
        _candidate(f"c{index}", (0.4 + 0.05 * index, 0.9 - 0.05 * index),
                   rank=index + 1, size=6)
        for index in range(6)
    )
    policy = _policy(True)

    def dispersions(covered):
        ranking = policy.score_market(
            candidates=market,
            archive_points=ARCHIVE,
            observed_outcomes=(),
            future_seats_remaining=1,
            horizon_total=6,
            covered_anchors=covered,
        )
        return {
            card.action_sha256: card.anchor_dispersion
            for card in ranking.scores
        }

    one = dispersions((_point(0.45, 0.85),))
    two = dispersions((_point(0.45, 0.85), _point(0.60, 0.70)))
    for action_sha256, value in two.items():
        assert value <= one[action_sha256]


def test_tie_break_is_invariant_to_rescaling_the_objective_frame():
    """It reads only ORDERS of geometry, so no unit or constant leaks."""

    def market(scale: float):
        return tuple(
            _candidate(label, (x * scale, y * scale), rank=rank, size=3)
            for label, x, y, rank in (
                ("a", 0.45, 0.55, 1),
                ("b", 0.70, 0.70, 2),
                ("c", 0.95, 0.95, 3),
            )
        )

    def archive(scale: float):
        return (
            _point(0.5 * scale, 0.5 * scale),
            _point(0.4 * scale, 0.7 * scale),
        )

    policy = _policy(True)
    orders = []
    for scale in (1.0, 1000.0):
        ranking = policy.score_market(
            candidates=market(scale),
            archive_points=archive(scale),
            observed_outcomes=(),
            future_seats_remaining=2,
            horizon_total=3,
            covered_anchors=(_point(0.45 * scale, 0.55 * scale),),
        )
        orders.append(ranking.ranked_action_sha256s)
    assert orders[0] == orders[1]


def test_anchorless_candidates_take_the_market_median_position():
    market = (
        _candidate("near", (0.45, 0.55), rank=1, size=4),
        _candidate("mid", (0.70, 0.70), rank=2, size=4),
        _candidate("far", (0.95, 0.95), rank=3, size=4),
        _candidate("anchorless", None, rank=4, size=4),
    )
    ranking = _policy(True).score_market(
        candidates=market,
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=4,
    )
    by_action = {card.action_sha256: card for card in ranking.scores}
    assert by_action[_sha("candidate:anchorless")].anchor_excess is None
    # Median of three anchors is the middle one, so the anchorless lane is
    # neither evicted to the back nor promoted to the front.
    assert _rank(ranking, "near") < _rank(ranking, "anchorless")
    assert _rank(ranking, "anchorless") < _rank(ranking, "far")


def test_the_score_still_dominates_the_tie_break():
    """A converting cell outranks a dead one whatever the geometry says."""

    market = (
        _candidate(
            "hot", (0.95, 0.95), engine_id="engine.hot", rank=1, size=1
        ),
        _candidate(
            "cold", (0.45, 0.55), engine_id="engine.cold", rank=1, size=1
        ),
    )
    outcomes = (
        ObservedConversionOutcome(
            observation_ordinal=1,
            engine_id="engine.hot",
            native_rank=1,
            lane_size=1,
            feasible=True,
            marginal_archive_gain=1.0e-2,
        ),
        ObservedConversionOutcome(
            observation_ordinal=2,
            engine_id="engine.cold",
            native_rank=1,
            lane_size=1,
            feasible=True,
            marginal_archive_gain=0.0,
        ),
    )
    ranking = _policy(True).score_market(
        candidates=market,
        archive_points=ARCHIVE,
        observed_outcomes=outcomes,
        future_seats_remaining=1,
        horizon_total=2,
    )
    assert ranking.ranked_action_sha256s[0] == _sha("candidate:hot")


def test_covered_anchors_must_be_exact_objective_points():
    with pytest.raises(TypeError):
        _policy(True).score_market(
            candidates=(_candidate("a", (0.5, 0.5)),),
            archive_points=ARCHIVE,
            observed_outcomes=(),
            future_seats_remaining=0,
            horizon_total=1,
            covered_anchors=[_point(0.5, 0.5)],
        )


def test_r1_and_r2_identities_are_preserved_and_r3_is_distinct():
    def build(version_id):
        return V8LiteAllocationPolicy(
            archive_gain_utility=_NullGainPort(),
            policy_version_id=version_id,
            config=V8LiteAllocationConfig(),
        )

    r1 = build(V8LITE_ALLOCATION_POLICY_VERSION_ID)
    r2 = build(V8LITE_ALLOCATION_POLICY_VERSION_ID_R2)
    r3 = build(V8LITE_ALLOCATION_POLICY_VERSION_ID_R3)
    assert len(
        {
            r1.definition_sha256,
            r2.definition_sha256,
            r3.definition_sha256,
        }
    ) == 3
    # r3 is r2 plus the tie-break, so it keeps the sequential pilot and the
    # frozen V7 terminal component byte-for-byte.
    assert r2.revision_2 and r3.revision_2 and not r1.revision_2
    assert r3.revision_3 and not r2.revision_3 and not r1.revision_3
    for key in ("pilot", "terminal_and_fallback"):
        assert (
            r3.identity_record()["components"][key]["definition_sha256"]
            == r2.identity_record()["components"][key]["definition_sha256"]
        )
    assert not r1.challenger_policy().anchor_geometry_tie_break
    assert not r2.challenger_policy().anchor_geometry_tie_break
    assert r3.challenger_policy().anchor_geometry_tie_break
    assert (
        r3.identity_record()["r3"]["tie_break_carries_no_weight_or_constant"]
        is True
    )


def test_chebyshev_excess_moved_but_still_importable_from_admission():
    from agent_evolve.application import front_proximity_admission

    assert front_proximity_admission.chebyshev_excess is chebyshev_excess
    assert chebyshev_excess(
        _point(0.6, 0.6), ARCHIVE, METRIC_IDS
    ) == pytest.approx(0.1)
    assert chebyshev_excess(_point(0.3, 0.3), ARCHIVE, METRIC_IDS) < 0.0

"""The three score-informativeness repairs (v8lite r4 and its wiring).

The jul28 diagnosis measured that the calibrated challenger's score carried
no per-candidate content: pooled Spearman with realised gain -0.0098, AUC
0.488, 1.4 to 2.8 distinct score values over markets of 37 to 64 members.
Three defects produced that:

1. the frozen cross-campaign prior -- the only input with measured signal --
   was gated shut because ``frozen_fit_training_run_count`` never reached
   the scorer;
2. the forecast branch never executed, because no adapter ever built a
   ``PositiveGainForecast`` even though the sealed proposal records carry
   per-metric scenario deltas against the candidate's parent; and
3. the conversion branch's tail risk ``(1 - p) * M`` is not a shortfall: it
   reduces the score to ``M * (2p - 1)``, inverting the magnitude channel
   for every candidate with ``p < 0.5``.

These tests pin all three, and pin that the r1, r2 and r3 identities are
byte-identical so the live pins still verify.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    CalibratedPositiveGainOpportunityPolicy,
    ObservedConversionOutcome,
    PositiveGainCandidate,
    PositiveGainForecast,
)
from agent_evolve.application.sequential_market_replay import (
    ExactHypervolumeGainPort,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionOutcome,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LITE_ALLOCATION_POLICY_VERSION_ID,
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R2,
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R3,
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R4,
    V8LiteAllocationConfig,
    V8LiteAllocationPolicy,
)

METRIC_IDS = ("m.x", "m.y")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _point(x: float, y: float):
    return tuple(sorted((("m.x", float(x)), ("m.y", float(y)))))


ARCHIVE = (_point(0.5, 0.5), _point(0.4, 0.7))


class _BoxGainPort:
    """Exact 2-D dominated-box gain against a unit reference."""

    utility_id = "test.box_gain"
    utility_version = 1
    definition_sha256 = _sha("test-box-gain")

    @staticmethod
    def _area(points) -> float:
        kept = sorted({p for p in points if p[0] < 1.0 and p[1] < 1.0})
        total, best_y = 0.0, 1.0
        for x, y in kept:
            if y < best_y:
                total += (1.0 - x) * (best_y - y)
                best_y = y
        return total

    def marginal_archive_gain(self, archive_points, objective_point):
        base = [tuple(v for _k, v in sorted(p)) for p in archive_points]
        added = tuple(v for _k, v in sorted(objective_point))
        return max(0.0, self._area([*base, added]) - self._area(base))


def _policy(**kwargs) -> CalibratedPositiveGainOpportunityPolicy:
    return CalibratedPositiveGainOpportunityPolicy(
        archive_gain_utility=_BoxGainPort(),
        **kwargs,
    )


def _candidate(label: str, *, engine_id="engine.a", rank=1, size=4, **kw):
    return PositiveGainCandidate(
        action_sha256=_sha(f"candidate:{label}"),
        engine_id=engine_id,
        native_rank=rank,
        lane_size=size,
        **kw,
    )


def _observed(ordinal: int, engine_id: str, gain: float, *, size=4):
    return ObservedConversionOutcome(
        observation_ordinal=ordinal,
        engine_id=engine_id,
        native_rank=1,
        lane_size=size,
        feasible=True,
        marginal_archive_gain=float(gain),
    )


# --- defect 3: the tail-risk sign -----------------------------------------


def _two_cell_market():
    """Two engines: one converts rarely but LARGE, one often but small.

    Both sit below the half-probability line where the old algebra inverts.
    """

    outcomes = (
        # engine.rare: 1 positive in 5, magnitude 1e-2.
        *(
            _observed(index, "engine.rare", 0.0)
            for index in range(1, 5)
        ),
        _observed(5, "engine.rare", 1.0e-2),
        # engine.often: 2 positives in 5, magnitude 1e-4.
        *(
            _observed(index, "engine.often", 0.0)
            for index in range(6, 9)
        ),
        _observed(9, "engine.often", 1.0e-4),
        _observed(10, "engine.often", 1.0e-4),
    )
    return (
        _candidate("rare", engine_id="engine.rare"),
        _candidate("often", engine_id="engine.often"),
    ), outcomes


def _scored(policy, candidates, outcomes):
    ranking = policy.score_market(
        candidates=candidates,
        archive_points=ARCHIVE,
        observed_outcomes=outcomes,
        future_seats_remaining=0,
        horizon_total=6,
    )
    return {
        value.action_sha256: value for value in ranking.scores
    }


def test_old_tail_risk_inverts_the_magnitude_channel() -> None:
    """The defect, pinned: below half-probability, bigger gains score lower."""

    candidates, outcomes = _two_cell_market()
    cards = _scored(_policy(), candidates, outcomes)
    rare = cards[_sha("candidate:rare")]
    often = cards[_sha("candidate:often")]

    assert rare.conversion_probability < 0.5
    assert often.conversion_probability < 0.5
    # The cell with the demonstrably larger gains ...
    assert rare.conversion_magnitude > often.conversion_magnitude
    # ... scores lower, and in fact scores below zero.
    assert rare.score < often.score
    assert rare.score < 0.0
    # Because the score is exactly M * (2p - 1) at lambda = 1, beta = 0.
    assert rare.score == pytest.approx(
        rare.conversion_magnitude
        * (2.0 * rare.conversion_probability - 1.0)
    )


def test_downside_shortfall_restores_the_magnitude_channel() -> None:
    """The repair.  This assertion FAILS on the old ``(1 - p) * M`` form."""

    candidates, outcomes = _two_cell_market()
    cards = _scored(
        _policy(downside_shortfall_tail_risk=True),
        candidates,
        outcomes,
    )
    rare = cards[_sha("candidate:rare")]
    often = cards[_sha("candidate:often")]

    assert rare.conversion_probability < 0.5
    assert rare.conversion_magnitude > often.conversion_magnitude
    assert rare.score > often.score
    assert rare.score > 0.0


def test_downside_shortfall_is_the_expected_shortfall_below_the_mean() -> None:
    """``p (1 - p) M`` is the forecast branch's own shortfall definition.

    ``sum_s w_s max(0, central - gain_s)`` over the conversion branch's own
    two-point scenario set -- gain ``M`` with probability ``p``, gain ``0``
    with probability ``1 - p``, central outcome the expected gain ``p M``.
    """

    candidates, outcomes = _two_cell_market()
    cards = _scored(
        _policy(downside_shortfall_tail_risk=True),
        candidates,
        outcomes,
    )
    for card in cards.values():
        probability = card.conversion_probability
        magnitude = card.conversion_magnitude
        central = probability * magnitude
        closed_form = probability * max(
            0.0, central - magnitude
        ) + (1.0 - probability) * max(0.0, central - 0.0)
        assert card.tail_risk == pytest.approx(closed_form)
        assert card.tail_risk == pytest.approx(
            probability * (1.0 - probability) * magnitude
        )


@pytest.mark.parametrize("positives", (0, 1, 2, 3, 4, 5))
def test_repaired_score_never_decreases_in_probability(positives) -> None:
    """Monotone in ``p`` at fixed magnitude, at every conversion rate."""

    scores = []
    for count in range(positives + 1):
        outcomes = tuple(
            _observed(
                index,
                "engine.a",
                1.0e-3 if index <= count else 0.0,
            )
            for index in range(1, 6)
        )
        card = _scored(
            _policy(downside_shortfall_tail_risk=True),
            (_candidate("probe"),),
            outcomes,
        )[_sha("candidate:probe")]
        scores.append((card.conversion_probability, card.score))
    ordered = sorted(scores)
    assert [value for _p, value in ordered] == sorted(
        value for _p, value in ordered
    )


def test_repaired_score_never_decreases_in_magnitude() -> None:
    """Monotone in ``M`` at fixed probability."""

    previous = None
    for magnitude in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        outcomes = (
            _observed(1, "engine.a", magnitude),
            _observed(2, "engine.a", 0.0),
            _observed(3, "engine.a", 0.0),
            _observed(4, "engine.a", 0.0),
            _observed(5, "engine.a", 0.0),
        )
        card = _scored(
            _policy(downside_shortfall_tail_risk=True),
            (_candidate("probe"),),
            outcomes,
        )[_sha("candidate:probe")]
        assert card.conversion_probability < 0.5
        if previous is not None:
            assert card.score > previous
        previous = card.score


def test_repaired_tail_risk_still_penalises_uncertainty() -> None:
    """The risk term is a penalty, not a no-op: lambda still bites."""

    candidates, outcomes = _two_cell_market()
    risky = _scored(
        _policy(downside_shortfall_tail_risk=True, lambda_=1.0),
        candidates,
        outcomes,
    )[_sha("candidate:rare")]
    neutral = _scored(
        _policy(downside_shortfall_tail_risk=True, lambda_=0.0),
        candidates,
        outcomes,
    )[_sha("candidate:rare")]
    assert risky.tail_risk > 0.0
    assert risky.score < neutral.score


# --- defect 1: the frozen prior's provenance gate --------------------------


def _frozen_market():
    return tuple(
        _candidate(
            f"frozen-{index}",
            engine_id="engine.a",
            rank=index,
            size=6,
            frozen_score=score,
        )
        for index, score in enumerate(
            (0.05, 0.25, 0.5, 0.75, 0.95), start=1
        )
    )


def test_frozen_gate_stays_shut_below_the_attested_floor() -> None:
    policy = _policy()
    ranking = policy.score_market(
        candidates=_frozen_market(),
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=6,
        frozen_fit_training_run_count=(
            policy.frozen_score_minimum_training_runs - 1
        ),
    )
    assert all(
        value.frozen_evidence_weight == 0.0 for value in ranking.scores
    )
    # This is the diagnosed collapse: one distinct score over five members.
    assert len({value.score for value in ranking.scores}) == 1


def test_frozen_gate_opens_at_the_attested_floor() -> None:
    policy = _policy()
    ranking = policy.score_market(
        candidates=_frozen_market(),
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=6,
        frozen_fit_training_run_count=(
            policy.frozen_score_minimum_training_runs
        ),
    )
    assert all(
        value.frozen_evidence_weight > 0.0 for value in ranking.scores
    )
    # The market is no longer one cell constant.
    assert len({value.score for value in ranking.scores}) == 5


def test_frozen_authority_is_bounded_by_its_normalized_share() -> None:
    """It can move the posterior; it can never own it."""

    policy = _policy()
    share = policy.frozen_score_weight / (
        (1.0 - policy.mixture_weight) + policy.frozen_score_weight
    )
    ranking = policy.score_market(
        candidates=_frozen_market(),
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=6,
        frozen_fit_training_run_count=(
            policy.frozen_score_minimum_training_runs
        ),
    )
    for value in ranking.scores:
        assert value.frozen_evidence_weight == pytest.approx(share)
        assert value.frozen_evidence_weight < 0.5
        # A log-odds convex combination lands strictly between its two
        # inputs, and strictly nearer the heavier one (the conversion
        # posterior), so the prior is never the sole authority.
        low = min(value.conversion_probability, value.frozen_score_probability)
        high = max(
            value.conversion_probability, value.frozen_score_probability
        )
        assert low <= value.p_positive_archive_gain <= high
        if value.frozen_score_probability != value.conversion_probability:
            assert abs(
                value.p_positive_archive_gain
                - value.conversion_probability
            ) < abs(
                value.p_positive_archive_gain
                - value.frozen_score_probability
            )


def test_frozen_prior_cannot_outrank_a_converting_cell() -> None:
    """A dead cell with a perfect prior stays behind a converting one."""

    policy = _policy()
    candidates = (
        _candidate(
            "dead-but-confident",
            engine_id="engine.dead",
            frozen_score=1.0,
        ),
        _candidate(
            "live-but-doubted",
            engine_id="engine.live",
            frozen_score=0.0,
        ),
    )
    outcomes = tuple(
        _observed(index, "engine.dead", 0.0) for index in range(1, 9)
    ) + tuple(
        _observed(index, "engine.live", 1.0e-3)
        for index in range(9, 17)
    )
    ranking = policy.score_market(
        candidates=candidates,
        archive_points=ARCHIVE,
        observed_outcomes=outcomes,
        future_seats_remaining=2,
        horizon_total=6,
        frozen_fit_training_run_count=(
            policy.frozen_score_minimum_training_runs
        ),
    )
    assert ranking.ranked_action_sha256s[0] == _sha(
        "candidate:live-but-doubted"
    )


def test_frozen_gate_opens_through_the_live_controller() -> None:
    """Live-shaped: the count reaches the scorer through ``select_next``."""

    market = tuple(
        AdaptiveActionDescriptor(
            action_sha256=_sha(f"action:{index}"),
            phenotype_sha256=_sha(f"phenotype:{index}"),
            lane_id="engine.a" if index % 2 else "engine.b",
            operator_id="operator.x",
            native_rank=(index + 1) // 2,
            lane_size=4,
            prior_score=0.05 + 0.1 * index,
            parent_generated_in_current_run=False,
        )
        for index in range(1, 9)
    )
    controller = V8LiteAllocationPolicy(
        archive_gain_utility=_BoxGainPort(),
        policy_version_id=V8LITE_ALLOCATION_POLICY_VERSION_ID_R4,
        config=V8LiteAllocationConfig(beta=0.0),
    )
    selected = tuple(sorted(value.action_sha256 for value in market[:3]))
    outcomes = tuple(
        AdaptiveActionOutcome(
            action_sha256=value,
            evaluation_sha256=_sha(f"evaluation:{value}"),
            feasible=True,
            marginal_archive_gain=0.0,
        )
        for value in selected
    )
    kwargs = {
        "residual_request_sha256": _sha("request"),
        "actions": market,
        "evaluation_slots": 8,
        "diagnostic_action_sha256s": selected,
        "diagnostic_joint_gain": 0.0,
        "selected_action_sha256s": selected,
        "outcomes": outcomes,
        "archive_points": ARCHIVE,
    }
    floor = (
        controller.challenger_policy().frozen_score_minimum_training_runs
    )
    unattested = controller.select_next(**kwargs)
    attested = controller.select_next(
        **kwargs, frozen_fit_training_run_count=floor
    )

    assert all(
        value.frozen_evidence_weight == 0.0
        for value in unattested.challenger_ranking.scores
    )
    assert all(
        value.frozen_evidence_weight > 0.0
        for value in attested.challenger_ranking.scores
    )
    assert len(
        {value.score for value in unattested.challenger_ranking.scores}
    ) < len({value.score for value in attested.challenger_ranking.scores})


# --- defect 2: the forecast branch -----------------------------------------


def test_forecast_from_parent_and_deltas_builds_canonical_scenarios() -> None:
    forecast = PositiveGainForecast.from_parent_and_deltas(
        parent_point=_point(0.60, 0.70),
        quantile_deltas=(
            ("m.x", (-0.20, -0.10, 0.0)),
            ("m.y", (-0.30, -0.15, 0.05)),
        ),
        reliability=0.75,
    )
    for scenario, expected in (
        ("p10", (0.40, 0.40)),
        ("p50", (0.50, 0.55)),
        ("p90", (0.60, 0.75)),
    ):
        point = dict(forecast.point(scenario))
        assert point["m.x"] == pytest.approx(expected[0])
        assert point["m.y"] == pytest.approx(expected[1])
    assert tuple(
        scenario for scenario, _point in forecast.quantile_points
    ) == ("p10", "p50", "p90")
    assert forecast.reliability == 0.75


def test_forecast_construction_commutes_with_frame_rescaling() -> None:
    """Pure translation: any affine renormalization commutes with it."""

    scale = {"m.x": 80.0, "m.y": 12000.0}
    parent = {"m.x": 29.0, "m.y": 7116.0}
    deltas = {"m.x": (-6.0, -4.0, -2.0), "m.y": (-510.0, -320.0, -150.0)}

    raw_then_scaled = PositiveGainForecast.from_parent_and_deltas(
        parent_point=tuple(sorted(parent.items())),
        quantile_deltas=tuple(sorted(deltas.items())),
    )
    scaled_then_built = PositiveGainForecast.from_parent_and_deltas(
        parent_point=tuple(
            sorted(
                (key, value / scale[key]) for key, value in parent.items()
            )
        ),
        quantile_deltas=tuple(
            sorted(
                (
                    key,
                    tuple(value / scale[key] for value in triple),
                )
                for key, triple in deltas.items()
            )
        ),
    )
    for scenario in ("p10", "p50", "p90"):
        raw = dict(raw_then_scaled.point(scenario))
        built = dict(scaled_then_built.point(scenario))
        for metric_id in ("m.x", "m.y"):
            assert built[metric_id] == pytest.approx(
                raw[metric_id] / scale[metric_id]
            )


def test_forecast_construction_requires_the_parent_frame() -> None:
    with pytest.raises(ValueError):
        PositiveGainForecast.from_parent_and_deltas(
            parent_point=_point(0.6, 0.7),
            quantile_deltas=(("m.x", (-0.1, -0.05, 0.0)),),
        )
    with pytest.raises(TypeError):
        PositiveGainForecast.from_parent_and_deltas(
            parent_point=_point(0.6, 0.7),
            quantile_deltas=(
                ("m.x", (-0.1, -0.05)),
                ("m.y", (-0.1, -0.05, 0.0)),
            ),
        )


def test_the_forecast_branch_actually_executes_once_wired() -> None:
    """The dead branch: zero invocations without a forecast, and content
    that separates the market with one."""

    port = _BoxGainPort()
    calls: list[tuple] = []
    original = port.marginal_archive_gain

    def counted(archive_points, objective_point):
        calls.append((archive_points, objective_point))
        return original(archive_points, objective_point)

    port.marginal_archive_gain = counted  # type: ignore[method-assign]
    policy = CalibratedPositiveGainOpportunityPolicy(
        archive_gain_utility=port
    )
    bare = tuple(
        _candidate(f"bare-{index}", rank=index, size=4)
        for index in range(1, 5)
    )
    ranking = policy.score_market(
        candidates=bare,
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=6,
    )
    assert calls == []
    assert all(
        value.forecast_probability is None for value in ranking.scores
    )
    assert len({value.score for value in ranking.scores}) == 1

    wired = tuple(
        PositiveGainCandidate(
            action_sha256=value.action_sha256,
            engine_id=value.engine_id,
            native_rank=value.native_rank,
            lane_size=value.lane_size,
            forecast=PositiveGainForecast.from_parent_and_deltas(
                parent_point=_point(0.60, 0.70),
                quantile_deltas=(
                    ("m.x", (-0.30 * index, -0.20 * index, -0.10 * index)),
                    ("m.y", (-0.30 * index, -0.20 * index, -0.10 * index)),
                ),
            ),
        )
        for index, value in enumerate(bare, start=1)
    )
    wired_ranking = policy.score_market(
        candidates=wired,
        archive_points=ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=6,
    )
    assert len(calls) == 3 * len(wired)
    assert all(
        value.forecast_probability is not None
        for value in wired_ranking.scores
    )
    assert len({value.score for value in wired_ranking.scores}) == len(wired)


# --- identity preservation --------------------------------------------------


# The live pins bind the gain port's own identity, so they can only be
# reproduced with the port the panels ran: an exact hypervolume utility over
# the unit-normalized frame the campaign declared.
LIVE_GAIN_PORT_METRIC_IDS = ("total_levels", "total_lut_count")
LIVE_PINS = {
    V8LITE_ALLOCATION_POLICY_VERSION_ID: (
        "da51bf6caab3bad894eda24f9647c53b1aa7ec43099cb3386feb7170dfe86771",
        "33e1034fad6a07102d816a29986e33f5726e22ee9811a30444f46c702f6263b3",
    ),
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R2: (
        "7214359193f0fc8963a91c26a29000979051337aa301ab8e816001b677f78d76",
        "98f4527645db2acb5d86974f2eb950dab1926d7a2c6421d05687ec7c79d444d1",
    ),
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R3: (
        "5c215eaad681c64075748afe6e0d7f87609adc2976f730b9b10665fa41dc64a0",
        "6928192dcbb60ad962a93b530d4efd18ee5e322161eb2b0ebbdad18e698fcbaa",
    ),
}


@pytest.mark.parametrize("version_id", sorted(LIVE_PINS))
def test_earlier_revisions_keep_their_definition_bytes(version_id) -> None:
    """The jul28 live runs pinned v8lite_r2 at ``7214359193f0...``."""

    policy = V8LiteAllocationPolicy(
        archive_gain_utility=ExactHypervolumeGainPort(
            tuple(
                sorted(
                    (metric_id, 1.0)
                    for metric_id in LIVE_GAIN_PORT_METRIC_IDS
                )
            )
        ),
        policy_version_id=version_id,
        config=V8LiteAllocationConfig(),
    )
    controller, challenger = LIVE_PINS[version_id]
    assert policy.definition_sha256 == controller
    assert policy.challenger_policy().definition_sha256 == challenger
    assert not policy.challenger_policy().downside_shortfall_tail_risk


def test_r4_is_r3_plus_the_repaired_tail_risk_only() -> None:
    r3 = V8LiteAllocationPolicy(
        archive_gain_utility=_BoxGainPort(),
        policy_version_id=V8LITE_ALLOCATION_POLICY_VERSION_ID_R3,
        config=V8LiteAllocationConfig(),
    )
    r4 = V8LiteAllocationPolicy(
        archive_gain_utility=_BoxGainPort(),
        policy_version_id=V8LITE_ALLOCATION_POLICY_VERSION_ID_R4,
        config=V8LiteAllocationConfig(),
    )
    assert r4.revision_2 and r4.revision_3 and r4.revision_4
    assert not r3.revision_4
    assert r4.definition_sha256 != r3.definition_sha256
    assert (
        r4.pilot_policy().definition_sha256
        == r3.pilot_policy().definition_sha256
    )
    assert (
        r4.terminal_policy().definition_sha256
        == r3.terminal_policy().definition_sha256
    )
    assert r4.challenger_policy().downside_shortfall_tail_risk
    assert r4.challenger_policy().anchor_geometry_tie_break
    # The repair carries no new constant.
    assert r4.config.to_record() == r3.config.to_record()


def test_r4_score_cards_keep_the_pre_r4_byte_layout() -> None:
    """The repair changes a value, never the record schema."""

    candidates, outcomes = _two_cell_market()
    before = _scored(_policy(), candidates, outcomes)
    after = _scored(
        _policy(downside_shortfall_tail_risk=True), candidates, outcomes
    )
    for action_sha256, card in before.items():
        assert set(card.to_record()) == set(
            after[action_sha256].to_record()
        )
        assert card.score_sha256 != after[action_sha256].score_sha256


def test_repaired_score_stays_finite_at_the_probability_extremes() -> None:
    for gain in (0.0, 1.0e-9, 1.0):
        outcomes = tuple(
            _observed(index, "engine.a", gain) for index in range(1, 21)
        )
        card = _scored(
            _policy(downside_shortfall_tail_risk=True),
            (_candidate("probe"),),
            outcomes,
        )[_sha("candidate:probe")]
        assert math.isfinite(card.score)
        assert card.tail_risk >= 0.0

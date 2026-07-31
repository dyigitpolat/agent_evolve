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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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
    """Exact two-metric minimization hypervolume gain for tests."""

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


def _point(x: float, y: float) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((("m.x", float(x)), ("m.y", float(y)))))


def _forecast(
    p10: tuple[float, float],
    p50: tuple[float, float],
    p90: tuple[float, float],
    *,
    reliability: float = 1.0,
) -> PositiveGainForecast:
    return PositiveGainForecast(
        quantile_points=(
            ("p10", _point(*p10)),
            ("p50", _point(*p50)),
            ("p90", _point(*p90)),
        ),
        reliability=reliability,
    )


def _candidate(
    label: str,
    *,
    engine_id: str = "engine.a",
    native_rank: int = 1,
    lane_size: int = 6,
    forecast: PositiveGainForecast | None = None,
) -> PositiveGainCandidate:
    return PositiveGainCandidate(
        action_sha256=_sha(f"candidate:{label}"),
        engine_id=engine_id,
        native_rank=native_rank,
        lane_size=lane_size,
        forecast=forecast,
    )


def _outcome(
    ordinal: int,
    *,
    engine_id: str,
    native_rank: int,
    lane_size: int = 6,
    gain: float,
    feasible: bool = True,
) -> ObservedConversionOutcome:
    return ObservedConversionOutcome(
        observation_ordinal=ordinal,
        engine_id=engine_id,
        native_rank=native_rank,
        lane_size=lane_size,
        feasible=feasible,
        marginal_archive_gain=float(gain),
    )


_ARCHIVE = (_point(4.0, 4.0),)
_REFERENCE = (10.0, 10.0)


def _policy(**overrides: object) -> CalibratedPositiveGainOpportunityPolicy:
    values: dict[str, object] = {
        "archive_gain_utility": _Hv2GainPort(_REFERENCE),
        "reference_gain_scale": 0.5,
    }
    values.update(overrides)
    return CalibratedPositiveGainOpportunityPolicy(**values)


def test_uncertain_positive_central_forecast_is_never_abstained() -> None:
    # Adverse quantile dominated by the archive, central and favorable
    # clearly positive: the old adverse gate abstained here.
    candidate = _candidate(
        "uncertain",
        forecast=_forecast((5.0, 5.0), (2.0, 6.0), (1.0, 5.0)),
    )
    ranking = _policy().score_market(
        candidates=(candidate,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=8,
    )

    score = ranking.score_for(candidate.action_sha256)
    assert ranking.ranked_action_sha256s == (candidate.action_sha256,)
    assert score.forecast_nondominated_fraction == pytest.approx(0.75)
    assert score.forecast_probability == pytest.approx(0.75)
    assert score.forecast_magnitude > 0.0
    assert score.score > 0.0
    record = ranking.to_record(include_scores=True)
    assert record["hard_abstention"] is False
    assert all(
        value["abstained"] is False for value in record["scores"]
    )


def test_parent_improvement_dominated_by_archive_has_zero_geometry() -> None:
    # Every quantile improves an implicit parent but stays dominated by
    # the archive point (4, 4); the forecast geometry must contribute
    # nothing because opportunity is archive-conditioned.
    dominated = _candidate(
        "dominated",
        forecast=_forecast((9.0, 9.0), (6.0, 6.0), (5.0, 5.0)),
    )
    ranking = _policy().score_market(
        candidates=(dominated,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=8,
    )

    score = ranking.score_for(dominated.action_sha256)
    assert score.forecast_nondominated_fraction == 0.0
    assert score.forecast_probability == 0.0
    assert score.forecast_magnitude == 0.0


def test_dominated_candidate_ranks_below_archive_complement() -> None:
    dominated = _candidate(
        "dominated",
        forecast=_forecast((9.0, 9.0), (6.0, 6.0), (5.0, 5.0)),
    )
    complement = _candidate(
        "complement",
        forecast=_forecast((6.0, 3.0), (5.0, 2.0), (4.0, 1.0)),
    )
    ranking = _policy().score_market(
        candidates=(dominated, complement),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=8,
    )

    assert ranking.ranked_action_sha256s[0] == (
        complement.action_sha256
    )
    assert ranking.score_for(
        complement.action_sha256
    ).score > ranking.score_for(dominated.action_sha256).score


def test_cell_with_no_observations_shrinks_to_engine_then_global() -> None:
    policy = _policy()
    outcomes = (
        _outcome(1, engine_id="engine.a", native_rank=1, gain=1.0),
        _outcome(2, engine_id="engine.a", native_rank=2, gain=1.0),
        _outcome(3, engine_id="engine.a", native_rank=1, gain=0.0),
        _outcome(4, engine_id="engine.b", native_rank=1, gain=0.0),
    )
    # engine.a band 2 (ranks five and six) has zero observations.
    thin_cell = _candidate(
        "thin-cell",
        engine_id="engine.a",
        native_rank=6,
    )
    # engine.c has zero observations anywhere.
    thin_engine = _candidate("thin-engine", engine_id="engine.c")
    ranking = policy.score_market(
        candidates=(thin_cell, thin_engine),
        archive_points=_ARCHIVE,
        observed_outcomes=outcomes,
        future_seats_remaining=2,
        horizon_total=8,
    )

    strength = policy.prior_strength
    p_global = (strength * policy.root_prior_probability + 2.0) / (
        strength + 4.0
    )
    p_engine_a = (strength * p_global + 2.0) / (strength + 3.0)
    thin_cell_score = ranking.score_for(thin_cell.action_sha256)
    assert thin_cell_score.conversion_probability == p_engine_a
    assert thin_cell_score.effective_sample_size == strength

    p_engine_c = (strength * p_global + 0.0) / (strength + 0.0)
    thin_engine_score = ranking.score_for(
        thin_engine.action_sha256
    )
    assert thin_engine_score.conversion_probability == p_engine_c
    assert p_engine_c == p_global


def test_conversion_evidence_moves_forecastless_candidates() -> None:
    strong = _candidate(
        "strong-engine",
        engine_id="engine.a",
        native_rank=1,
    )
    weak = _candidate(
        "weak-engine",
        engine_id="engine.b",
        native_rank=1,
    )
    outcomes = (
        _outcome(1, engine_id="engine.a", native_rank=1, gain=1.0),
        _outcome(2, engine_id="engine.a", native_rank=2, gain=2.0),
        _outcome(3, engine_id="engine.b", native_rank=1, gain=0.0),
        _outcome(4, engine_id="engine.b", native_rank=2, gain=0.0),
    )
    ranking = _policy().score_market(
        candidates=(strong, weak),
        archive_points=_ARCHIVE,
        observed_outcomes=outcomes,
        future_seats_remaining=2,
        horizon_total=8,
    )

    strong_score = ranking.score_for(strong.action_sha256)
    weak_score = ranking.score_for(weak.action_sha256)
    assert strong_score.forecast_probability is None
    assert strong_score.conversion_probability > (
        weak_score.conversion_probability
    )
    assert strong_score.score > weak_score.score
    assert ranking.ranked_action_sha256s[0] == strong.action_sha256


def test_value_of_information_is_zero_at_the_terminal_seat() -> None:
    candidate = _candidate(
        "terminal",
        forecast=_forecast((5.0, 5.0), (2.0, 6.0), (1.0, 5.0)),
    )
    live = _policy().score_market(
        candidates=(candidate,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=8,
    )
    terminal = _policy().score_market(
        candidates=(candidate,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=0,
        horizon_total=8,
    )

    live_score = live.score_for(candidate.action_sha256)
    terminal_score = terminal.score_for(candidate.action_sha256)
    assert live_score.value_of_information > 0.0
    assert terminal_score.value_of_information == 0.0
    assert terminal_score.score < live_score.score


def test_probability_mixture_blends_in_log_odds_space() -> None:
    candidate = _candidate(
        "blend",
        forecast=_forecast((5.0, 5.0), (2.0, 6.0), (1.0, 5.0)),
    )
    policy = _policy()
    ranking = policy.score_market(
        candidates=(candidate,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=1,
        horizon_total=8,
    )

    score = ranking.score_for(candidate.action_sha256)
    forecast_p = 0.75
    conversion_p = policy.root_prior_probability
    blended = 1.0 / (
        1.0
        + math.exp(
            -(
                0.5 * math.log(forecast_p / (1.0 - forecast_p))
                + 0.5
                * math.log(conversion_p / (1.0 - conversion_p))
            )
        )
    )
    assert score.p_positive_archive_gain == pytest.approx(blended)


def test_scores_may_be_negative_but_ranking_is_total() -> None:
    hopeless = _candidate("hopeless", engine_id="engine.a")
    outcomes = tuple(
        _outcome(
            ordinal,
            engine_id="engine.a",
            native_rank=1,
            gain=0.0,
        )
        for ordinal in range(1, 7)
    )
    ranking = _policy(beta=0.0).score_market(
        candidates=(hopeless,),
        archive_points=_ARCHIVE,
        observed_outcomes=outcomes,
        future_seats_remaining=2,
        horizon_total=8,
    )

    score = ranking.score_for(hopeless.action_sha256)
    assert score.score < 0.0
    assert ranking.ranked_action_sha256s == (
        hopeless.action_sha256,
    )


def test_frozen_score_is_inert_below_the_training_run_floor() -> None:
    # The V70 frozen fit was trained on six runs; below the default
    # ten-run floor the frozen score must change nothing.
    bare = _candidate("frozen-gate")
    frozen = PositiveGainCandidate(
        action_sha256=bare.action_sha256,
        engine_id=bare.engine_id,
        native_rank=bare.native_rank,
        lane_size=bare.lane_size,
        frozen_score=0.99,
    )
    policy = _policy()
    without = policy.score_market(
        candidates=(bare,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=8,
    ).score_for(bare.action_sha256)
    gated = policy.score_market(
        candidates=(frozen,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=8,
        frozen_fit_training_run_count=6,
    ).score_for(frozen.action_sha256)

    assert gated.frozen_evidence_weight == 0.0
    assert gated.p_positive_archive_gain == (
        without.p_positive_archive_gain
    )
    assert gated.score == without.score


def test_frozen_score_is_a_weak_feature_even_when_active() -> None:
    frozen = PositiveGainCandidate(
        action_sha256=_sha("candidate:frozen-weak"),
        engine_id="engine.a",
        native_rank=1,
        lane_size=6,
        frozen_score=0.96875,
    )
    policy = _policy()
    score = policy.score_market(
        candidates=(frozen,),
        archive_points=_ARCHIVE,
        observed_outcomes=(),
        future_seats_remaining=2,
        horizon_total=8,
        frozen_fit_training_run_count=12,
    ).score_for(frozen.action_sha256)

    assert policy.frozen_score_weight <= 0.15
    # Normalized share: 0.125 / (0.5 + 0.125).
    assert score.frozen_evidence_weight == pytest.approx(0.2)
    # The blend moves toward the frozen prior but stays dominated by
    # the native-rank conversion evidence.
    assert score.p_positive_archive_gain > (
        score.conversion_probability
    )
    assert abs(
        score.p_positive_archive_gain - score.conversion_probability
    ) < abs(score.p_positive_archive_gain - 0.96875)


def test_future_leak_is_structurally_rejected() -> None:
    with pytest.raises(ValueError, match="ascending ordinals"):
        _policy().score_market(
            candidates=(_candidate("leak"),),
            archive_points=_ARCHIVE,
            observed_outcomes=(
                _outcome(
                    2,
                    engine_id="engine.a",
                    native_rank=1,
                    gain=0.0,
                ),
                _outcome(
                    1,
                    engine_id="engine.a",
                    native_rank=2,
                    gain=0.0,
                ),
            ),
            future_seats_remaining=1,
            horizon_total=8,
        )

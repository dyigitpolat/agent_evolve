from __future__ import annotations

import hashlib
import math

import pytest

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    CalibratedPositiveGainOpportunityPolicy,
    PositiveGainCandidate,
    PositiveGainForecast,
)
from agent_evolve.application.region_conditional_credit import (
    RADIUS_CLASS_LONG,
    RADIUS_CLASS_MID,
    RADIUS_CLASS_NONE,
    RADIUS_CLASS_SHORT,
    REGION_NO_PARENT,
    RegionConditionalChallengerPolicy,
    RegionConditionalOutcome,
    RegionCreditConfig,
    RegionFeatures,
    RegionScoredCandidate,
    parent_front_distance,
    parent_front_region,
    radius_operator_class,
)


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
# Normalized-frame archive: one extreme point per objective and a knee.
_ARCHIVE = (
    _point(0.1, 0.8),
    _point(0.4, 0.4),
    _point(0.8, 0.1),
)


def _challenger(
    credit: RegionCreditConfig = RegionCreditConfig(),
) -> RegionConditionalChallengerPolicy:
    return RegionConditionalChallengerPolicy(
        base=CalibratedPositiveGainOpportunityPolicy(
            archive_gain_utility=_Hv2GainPort((1.0, 1.0)),
        ),
        credit=credit,
    )


def _candidate(
    label: str,
    *,
    engine_id: str = "engine.inter",
    native_rank: int = 1,
    lane_size: int = 4,
    forecast: PositiveGainForecast | None = None,
    features: RegionFeatures = RegionFeatures(),
) -> RegionScoredCandidate:
    return RegionScoredCandidate(
        candidate=PositiveGainCandidate(
            action_sha256=_sha(f"region:{label}"),
            engine_id=engine_id,
            native_rank=native_rank,
            lane_size=lane_size,
            forecast=forecast,
        ),
        features=features,
    )


def _outcome(
    ordinal: int,
    *,
    engine_id: str = "engine.inter",
    positive: bool = False,
    region_id: str | None = None,
    radius_class_id: str | None = None,
    predicted: bool | None = None,
    actual: bool | None = None,
) -> RegionConditionalOutcome:
    return RegionConditionalOutcome(
        observation_ordinal=ordinal,
        engine_id=engine_id,
        feasible=True,
        marginal_archive_gain=0.01 if positive else 0.0,
        region_id=region_id,
        radius_class_id=radius_class_id,
        forecast_predicted_positive=predicted,
        forecast_actual_positive=actual,
    )


def test_parent_front_distance_matches_hand_computed_epsilon() -> None:
    # (0.5, 0.5) sits 0.1 behind the knee (0.4, 0.4) in max-norm.
    assert parent_front_distance(
        _ARCHIVE,
        _point(0.5, 0.5),
    ) == pytest.approx(0.1)
    # A point on the archive is at distance exactly zero.
    assert parent_front_distance(_ARCHIVE, _point(0.4, 0.4)) == 0.0
    # A point beyond the front clamps at zero.
    assert parent_front_distance(_ARCHIVE, _point(0.2, 0.2)) == 0.0


def test_parent_front_region_classifies_band_and_wedge() -> None:
    kwargs = {
        "archive_points": _ARCHIVE,
        "reference_point": _REFERENCE,
        "near_front_epsilon": 0.125,
        "extreme_affinity_threshold": 0.25,
    }
    assert (
        parent_front_region(parent_point=None, **kwargs)
        == REGION_NO_PARENT
    )
    # Near the m.x extreme: distance 0, affinity on m.x is 0.
    assert (
        parent_front_region(parent_point=_point(0.1, 0.8), **kwargs)
        == "region.near.extreme:m.x"
    )
    # Near the knee: both affinities are (0.4-0.1)/0.9 = 1/3 > 1/4.
    assert (
        parent_front_region(parent_point=_point(0.4, 0.4), **kwargs)
        == "region.near.interior"
    )
    # Deep off-front interior parent.
    assert (
        parent_front_region(parent_point=_point(0.7, 0.7), **kwargs)
        == "region.far.interior"
    )
    # Deep off-front but hugging the m.y extreme axis: distance to the
    # nearest archive point (0.8, 0.1) is max(0.15, 0.2) = 0.2 > 1/8,
    # while the m.y affinity (0.3 - 0.1) / 0.9 stays under 1/4.
    assert (
        parent_front_region(parent_point=_point(0.95, 0.3), **kwargs)
        == "region.far.extreme:m.y"
    )


def test_radius_operator_class_uses_breakpoints_then_fallbacks() -> None:
    breakpoints = (1, 3)
    for radius, expected in (
        (0, RADIUS_CLASS_SHORT),
        (1, RADIUS_CLASS_SHORT),
        (2, RADIUS_CLASS_MID),
        (3, RADIUS_CLASS_MID),
        (4, RADIUS_CLASS_LONG),
        (20, RADIUS_CLASS_LONG),
    ):
        assert (
            radius_operator_class(
                radius=radius,
                operator_class=None,
                radius_breakpoints=breakpoints,
            )
            == expected
        )
    assert (
        radius_operator_class(
            radius=None,
            operator_class="interaction.swap",
            radius_breakpoints=breakpoints,
        )
        == "op.interaction.swap"
    )
    assert (
        radius_operator_class(
            radius=None,
            operator_class=None,
            radius_breakpoints=breakpoints,
        )
        == RADIUS_CLASS_NONE
    )


def test_single_win_moves_its_cell_not_siblings_to_certainty() -> None:
    # The D1 defect: one n=1 lane win propagated positive=1.0 to all 32
    # lane siblings.  Region-conditional credit must lift the winning
    # cell above siblings, and the sibling estimate must be EXACTLY the
    # engine-level shrinkage value, far below certainty.
    challenger = _challenger()
    near = RegionFeatures(parent_point=_point(0.4, 0.4), radius=1)
    far = RegionFeatures(parent_point=_point(0.7, 0.7), radius=4)
    near_region, near_radius = challenger.region_for(
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        features=near,
    )
    assert (near_region, near_radius) == (
        "region.near.interior",
        RADIUS_CLASS_SHORT,
    )
    win = (
        _outcome(
            1,
            positive=True,
            region_id=near_region,
            radius_class_id=near_radius,
        ),
    )
    same_cell = _candidate("same-cell", features=near)
    sibling = _candidate("sibling", native_rank=2, features=far)

    def _p(scored: RegionScoredCandidate, outcomes) -> float:
        ranking = challenger.score_market(
            candidates=(scored,),
            archive_points=_ARCHIVE,
            reference_point=_REFERENCE,
            observed_outcomes=outcomes,
            future_seats_remaining=2,
            horizon_total=6,
        )
        return ranking.scores[0].conversion_probability

    baseline = _p(sibling, ())
    sibling_after = _p(sibling, win)
    same_cell_after = _p(same_cell, win)

    assert baseline == 0.5
    # Global: (2*0.5 + 1)/3 = 2/3; engine: (2*(2/3) + 1)/3 = 7/9;
    # sibling leaf cell is empty so it stays at the engine value.
    assert sibling_after == pytest.approx(7.0 / 9.0)
    # The winning cell moves strictly above the sibling.
    assert same_cell_after == pytest.approx(23.0 / 27.0)
    assert same_cell_after > sibling_after
    # Bounded increment: nowhere near the n=1 certainty the defect
    # propagated, and the sibling moved less than the winning cell.
    assert sibling_after < same_cell_after < 1.0
    assert sibling_after - baseline < same_cell_after - baseline


def test_prior_market_evidence_never_reaches_leaf_cells() -> None:
    challenger = _challenger()
    prior = tuple(
        _outcome(ordinal, positive=True) for ordinal in (1, 2, 3)
    )
    near = _candidate(
        "near",
        features=RegionFeatures(
            parent_point=_point(0.4, 0.4),
            radius=1,
        ),
    )
    far = _candidate(
        "far",
        native_rank=2,
        features=RegionFeatures(
            parent_point=_point(0.7, 0.7),
            radius=4,
        ),
    )
    ranking = challenger.score_market(
        candidates=(near, far),
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        observed_outcomes=prior,
        future_seats_remaining=2,
        horizon_total=6,
    )
    probabilities = {
        value.conversion_probability for value in ranking.scores
    }
    # Region-less prior evidence lands at the engine level, so both
    # cells inherit the identical engine posterior.
    assert len(probabilities) == 1


def test_forecast_trust_demotes_after_measured_anti_calibration() -> None:
    challenger = _challenger()
    # Optimistic forecast: every quantile beats the archive knee.
    forecast = PositiveGainForecast(
        quantile_points=(
            ("p10", _point(0.35, 0.35)),
            ("p50", _point(0.3, 0.3)),
            ("p90", _point(0.2, 0.2)),
        ),
    )
    candidate = _candidate("forecasted", forecast=forecast)
    anti_calibrated = tuple(
        _outcome(
            ordinal,
            positive=False,
            region_id="region.far.interior",
            radius_class_id=RADIUS_CLASS_LONG,
            predicted=True,
            actual=False,
        )
        for ordinal in (1, 2, 3, 4)
    )
    direction_blind = tuple(
        _outcome(
            ordinal,
            positive=False,
            region_id="region.far.interior",
            radius_class_id=RADIUS_CLASS_LONG,
        )
        for ordinal in (1, 2, 3, 4)
    )
    trusted = challenger.forecast_trust_multiplier(
        observed_outcomes=direction_blind,
        engine_id="engine.inter",
    )
    demoted = challenger.forecast_trust_multiplier(
        observed_outcomes=anti_calibrated,
        engine_id="engine.inter",
    )
    # No direction evidence keeps the multiplier at exactly one.
    assert trusted == 1.0
    # Beta posterior (2*0.5 + 0)/(2 + 4) = 1/6 -> min(1, 2/6) = 1/3.
    assert demoted == pytest.approx(1.0 / 3.0)

    def _p_positive(outcomes) -> float:
        ranking = challenger.score_market(
            candidates=(candidate,),
            archive_points=_ARCHIVE,
            reference_point=_REFERENCE,
            observed_outcomes=outcomes,
            future_seats_remaining=2,
            horizon_total=6,
        )
        return ranking.scores[0].p_positive_archive_gain

    # Identical conversion evidence; only the measured anti-calibration
    # differs, and it demotes the optimistic forecast's influence.
    assert _p_positive(anti_calibrated) < _p_positive(
        direction_blind
    )


def test_calibrated_direction_evidence_keeps_full_trust() -> None:
    challenger = _challenger()
    calibrated = tuple(
        _outcome(
            ordinal,
            positive=True,
            region_id="region.near.interior",
            radius_class_id=RADIUS_CLASS_SHORT,
            predicted=True,
            actual=True,
        )
        for ordinal in (1, 2, 3, 4)
    )
    assert (
        challenger.forecast_trust_multiplier(
            observed_outcomes=calibrated,
            engine_id="engine.inter",
        )
        == 1.0
    )


def test_definition_sha_binds_base_and_credit_config() -> None:
    default = _challenger()
    tightened = _challenger(
        credit=RegionCreditConfig(near_front_epsilon=0.0625)
    )
    assert default.policy_id == "region_conditional_credit"
    assert (
        default.definition_sha256 != tightened.definition_sha256
    )
    assert default.definition_sha256 == _challenger().definition_sha256


def test_rankings_are_complete_and_never_abstain() -> None:
    challenger = _challenger()
    candidates = tuple(
        _candidate(f"c{index}", native_rank=index, lane_size=4)
        for index in (1, 2, 3, 4)
    )
    ranking = challenger.score_market(
        candidates=candidates,
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=6,
    )
    assert len(ranking.ranked_action_sha256s) == 4
    assert sorted(ranking.ranked_action_sha256s) == sorted(
        value.candidate.action_sha256 for value in candidates
    )
    for value in ranking.scores:
        assert math.isfinite(value.score)

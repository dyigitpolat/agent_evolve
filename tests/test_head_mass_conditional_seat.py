from __future__ import annotations

import hashlib
import math

import pytest

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    CalibratedPositiveGainOpportunityPolicy,
    CalibratedPositiveGainRanking,
    CalibratedPositiveGainScore,
    PositiveGainCandidate,
    PositiveGainForecast,
)
from agent_evolve.application.head_mass_conditional_seat import (
    HeadMassSeatAssessor,
    HeadMassSeatConfig,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
)
from agent_evolve.application.region_conditional_credit import (
    RegionConditionalChallengerPolicy,
    RegionScoredCandidate,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LITE_PHASE_PILOT,
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
_STRONG_FORECAST = PositiveGainForecast(
    quantile_points=(
        ("p10", _point(0.35, 0.35)),
        ("p50", _point(0.2, 0.2)),
        ("p90", _point(0.1, 0.1)),
    ),
)


def _ranking_with_one_forecast() -> CalibratedPositiveGainRanking:
    challenger = RegionConditionalChallengerPolicy(
        base=CalibratedPositiveGainOpportunityPolicy(
            archive_gain_utility=_Hv2GainPort((1.0, 1.0)),
        ),
    )
    candidates = tuple(
        RegionScoredCandidate(
            candidate=PositiveGainCandidate(
                action_sha256=_sha(f"head:{index}"),
                engine_id="engine.a",
                native_rank=index,
                lane_size=4,
                forecast=(
                    _STRONG_FORECAST if index == 2 else None
                ),
            ),
        )
        for index in (1, 2, 3, 4)
    )
    return challenger.score_market(
        candidates=candidates,
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        observed_outcomes=(),
        future_seats_remaining=3,
        horizon_total=4,
    )


def _flat_score(action_sha256: str) -> CalibratedPositiveGainScore:
    return CalibratedPositiveGainScore(
        action_sha256=action_sha256,
        p_feasible=0.5,
        p_positive_archive_gain=0.5,
        expected_positive_gain=0.0,
        tail_risk=0.0,
        value_of_information=0.0,
        uncertainty=0.0,
        effective_sample_size=2.0,
        score=0.0,
        forecast_probability=None,
        forecast_magnitude=None,
        forecast_nondominated_fraction=None,
        conversion_probability=0.5,
        conversion_magnitude=0.0,
        frozen_score_probability=None,
        frozen_evidence_weight=0.0,
    )


def _flat_ranking() -> CalibratedPositiveGainRanking:
    scores = tuple(
        sorted(
            (_flat_score(_sha(f"flat:{index}")) for index in (1, 2)),
            key=lambda value: value.action_sha256,
        )
    )
    return CalibratedPositiveGainRanking(
        policy_id="test.flat",
        policy_version=1,
        policy_definition_sha256=_sha("flat-definition"),
        archive_sha256=_sha("flat-archive"),
        future_seats_remaining=1,
        horizon_total=2,
        scores=scores,
        ranked_action_sha256s=tuple(
            value.action_sha256 for value in scores
        ),
    )


def test_trigger_fires_only_strictly_above_threshold() -> None:
    ranking = _ranking_with_one_forecast()
    fraction = HeadMassSeatAssessor().assess(ranking)
    assert 0.0 < fraction.head_mass_fraction < 1.0
    below = HeadMassSeatAssessor(
        config=HeadMassSeatConfig(
            head_mass_threshold=(
                fraction.head_mass_fraction / 2.0
            ),
        ),
    ).assess(ranking)
    above = HeadMassSeatAssessor(
        config=HeadMassSeatConfig(
            head_mass_threshold=(
                (1.0 + fraction.head_mass_fraction) / 2.0
            ),
        ),
    ).assess(ranking)
    at = HeadMassSeatAssessor(
        config=HeadMassSeatConfig(
            head_mass_threshold=fraction.head_mass_fraction,
        ),
    ).assess(ranking)
    assert below.fired
    assert not above.fired
    # Exactly at the threshold is NOT strictly above: no fire.
    assert not at.fired
    assert below.argmax_action_sha256 == _sha("head:2")


def test_uniform_mass_ties_never_fire() -> None:
    ranking = _flat_ranking()
    # Total mass is zero here (expected gain zero); also every mass
    # ties, so neither condition can fire even at a tiny threshold.
    assessment = HeadMassSeatAssessor(
        config=HeadMassSeatConfig(head_mass_threshold=0.015625),
    ).assess(ranking)
    assert assessment.head_mass_fraction == 0.0
    assert assessment.total_predicted_mass == 0.0
    assert not assessment.fired


def test_assessment_record_logs_trigger_condition() -> None:
    ranking = _ranking_with_one_forecast()
    assessment = HeadMassSeatAssessor(
        config=HeadMassSeatConfig(head_mass_threshold=0.25),
    ).assess(ranking)
    record = assessment.to_record()
    assert record["fired"] is assessment.fired
    assert record["trigger_condition"] == (
        "head_mass_fraction_strictly_above_threshold"
    )
    assert record["deterministic_seat_propensity_hex"] == (1.0).hex()
    assert float.fromhex(record["head_mass_threshold_hex"]) == 0.25


def _market() -> tuple[AdaptiveActionDescriptor, ...]:
    values: list[AdaptiveActionDescriptor] = []
    for lane in ("coverage", "interaction", "local"):
        for rank in range(1, 4):
            values.append(
                AdaptiveActionDescriptor(
                    action_sha256=_sha(f"action:{lane}:{rank}"),
                    phenotype_sha256=_sha(
                        f"phenotype:{lane}:{rank}"
                    ),
                    lane_id=lane,
                    operator_id=lane,
                    native_rank=rank,
                    lane_size=3,
                    prior_score=float(4 - rank) / 3.0,
                    parent_generated_in_current_run=False,
                )
            )
    return tuple(values)


def _v9(
    *,
    r2: bool = True,
    threshold: float = 0.5,
) -> V9CandidatePolicy:
    return V9CandidatePolicy(
        archive_gain_utility=_Hv2GainPort((1.0, 1.0)),
        config=V9CandidateConfig(
            r2_head_mass_conditional_seat=r2,
            head=HeadMassSeatConfig(head_mass_threshold=threshold),
            base=V8LiteAllocationConfig(random_seed=3),
        ),
    )


def test_v9_first_seat_becomes_exact_point_mass_when_fired() -> None:
    market = _market()
    strong_action = _sha("action:interaction:2")
    policy = _v9(threshold=0.25)
    decision = policy.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(),
        outcomes=(),
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        forecasts=((strong_action, _STRONG_FORECAST),),
    )
    assert decision.phase == V8LITE_PHASE_PILOT
    assert decision.authority_policy_id == (
        "head_mass_conditional_seat"
    )
    assert decision.selected_action_sha256s == (strong_action,)
    # The deterministic branch is EXACTLY propensity one, logged.
    assert decision.selection_propensity == 1.0
    evidence = thaw_json(decision.evidence)
    seat = evidence["head_mass_seat"]
    assert seat["fired"] is True
    assert float.fromhex(seat["head_mass_fraction_hex"]) > 0.25
    assert evidence["deterministic_argmax_seat"] is True


def test_v9_first_seat_is_untouched_without_concentration() -> None:
    market = _market()
    policy = _v9(threshold=0.25)
    inner = policy.inner_policy()
    decision = policy.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(),
        outcomes=(),
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        forecasts=(),
    )
    reference = inner.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(),
        outcomes=(),
    )
    # No concentration: the seat is the inner stochastic pilot seat,
    # bit for bit (same selection, same exact propensity).
    assert decision.to_record() == reference.to_record()


def test_v9_later_seats_keep_the_stochastic_schedule() -> None:
    market = _market()
    strong_action = _sha("action:interaction:2")
    policy = _v9(threshold=0.25)
    inner = policy.inner_policy()
    first = policy.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(),
        outcomes=(),
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        forecasts=((strong_action, _STRONG_FORECAST),),
    )
    assert first.selected_action_sha256s == (strong_action,)
    from agent_evolve.application.outcome_adaptive_action_racing import (
        AdaptiveActionOutcome,
    )

    outcomes = (
        AdaptiveActionOutcome(
            action_sha256=strong_action,
            evaluation_sha256=_sha("evaluation:strong"),
            feasible=True,
            marginal_archive_gain=0.01,
        ),
    )
    second = policy.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(strong_action,),
        outcomes=outcomes,
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        forecasts=((strong_action, _STRONG_FORECAST),),
    )
    reference = inner.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(strong_action,),
        outcomes=outcomes,
    )
    assert second.to_record() == reference.to_record()
    assert second.selection_propensity < 1.0


def test_r2_disabled_never_intervenes() -> None:
    market = _market()
    strong_action = _sha("action:interaction:2")
    policy = _v9(r2=False, threshold=0.25)
    inner = policy.inner_policy()
    decision = policy.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(),
        outcomes=(),
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        forecasts=((strong_action, _STRONG_FORECAST),),
    )
    reference = inner.design_pilot_seat(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=6,
        selected_action_sha256s=(),
        outcomes=(),
    )
    assert decision.to_record() == reference.to_record()


def test_head_mass_fraction_matches_hand_computed_share() -> None:
    ranking = _ranking_with_one_forecast()
    masses = {
        value.action_sha256: (
            value.p_positive_archive_gain
            * value.expected_positive_gain
        )
        for value in ranking.scores
    }
    expected = max(masses.values()) / math.fsum(masses.values())
    assessment = HeadMassSeatAssessor().assess(ranking)
    assert assessment.head_mass_fraction == pytest.approx(expected)
    assert assessment.candidate_count == 4

from __future__ import annotations

import hashlib
import math

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    PositiveGainForecast,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionOutcome,
    AdaptiveActionWave,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LITE_PHASE_ADAPTIVE,
    V8LITE_PHASE_PILOT,
    V8LITE_PHASE_PROTECTED_FALLBACK,
    V8LITE_PHASE_TERMINAL,
    V8LiteAllocationConfig,
    V8LiteAllocationPolicy,
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


_ARCHIVE = (_point(4.0, 4.0),)


def _market() -> tuple[AdaptiveActionDescriptor, ...]:
    values: list[AdaptiveActionDescriptor] = []
    for lane in ("coverage", "interaction", "local"):
        for rank in range(1, 4):
            values.append(
                AdaptiveActionDescriptor(
                    action_sha256=_sha(f"action:{lane}:{rank}"),
                    phenotype_sha256=_sha(f"phenotype:{lane}:{rank}"),
                    lane_id=lane,
                    operator_id=f"{lane}.r{1 if rank < 3 else 2}",
                    native_rank=rank,
                    lane_size=3,
                    prior_score=float(4 - rank) / 3.0,
                    parent_generated_in_current_run=(rank == 2),
                    semantic_cell_ids=(
                        f"rank:{rank}",
                        f"source:{'generated' if rank == 2 else 'archive'}",
                    ),
                )
            )
    return tuple(values)


def _outcome(action_sha256: str, gain: float) -> AdaptiveActionOutcome:
    return AdaptiveActionOutcome(
        action_sha256=action_sha256,
        evaluation_sha256=_sha(f"evaluation:{action_sha256}"),
        feasible=True,
        marginal_archive_gain=float(gain),
    )


def _policy(**config_overrides: object) -> V8LiteAllocationPolicy:
    defaults: dict[str, object] = {
        "reference_gain_scale": 0.001,
        "reference_gain_evidence_sha256": _sha("prior-evidence"),
        "random_seed": 17,
    }
    defaults.update(config_overrides)
    return V8LiteAllocationPolicy(
        archive_gain_utility=_Hv2GainPort((10.0, 10.0)),
        config=V8LiteAllocationConfig(**defaults),
    )


def test_policy_identity_binds_components_and_version_id() -> None:
    policy = _policy()
    identity = policy.identity_record()

    assert identity["policy_version_id"] == "v8lite_r1"
    assert identity["v7_terminal_rule_altered"] is False
    assert set(identity["components"]) == {
        "pilot",
        "challenger",
        "terminal_and_fallback",
    }
    assert identity["definition_sha256"] == policy.definition_sha256
    assert (
        _policy(random_seed=18).definition_sha256
        != policy.definition_sha256
    )


def test_pilot_phase_uses_rank_balanced_causal_design() -> None:
    market = _market()
    decision = _policy().design_pilot(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
    )

    assert decision.phase == V8LITE_PHASE_PILOT
    assert decision.authority_policy_id == (
        "rank_balanced_causal_pilot"
    )
    assert len(decision.selected_action_sha256s) == 4
    by_action = {value.action_sha256: value for value in market}
    lanes = {
        by_action[value].lane_id
        for value in decision.selected_action_sha256s
    }
    assert lanes == {"coverage", "interaction", "local"}
    assert 0.0 < decision.selection_propensity <= 1.0
    seats = decision.pilot_design.seats
    assert all(
        seat.selection_propensity > 0.0 for seat in seats
    )


def test_positive_challenger_score_selects_adaptively() -> None:
    market = _market()
    policy = _policy()
    selected = tuple(
        sorted(value.action_sha256 for value in market[:3])
    )
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    strong_forecast_action = market[5].action_sha256
    decision = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
        forecasts=(
            (
                strong_forecast_action,
                PositiveGainForecast(
                    quantile_points=(
                        ("p10", _point(6.0, 3.0)),
                        ("p50", _point(5.0, 2.0)),
                        ("p90", _point(4.0, 1.0)),
                    ),
                ),
            ),
        ),
    )

    assert decision.phase == V8LITE_PHASE_ADAPTIVE
    assert decision.selected_action_sha256s == (
        strong_forecast_action,
    )
    assert decision.selection_propensity == 1.0
    ranking = decision.challenger_ranking
    assert ranking.ranked_action_sha256s[0] == strong_forecast_action
    assert ranking.score_for(strong_forecast_action).score > 0.0


def test_nonpositive_challenger_falls_back_to_protected_incumbent() -> None:
    market = _market()
    policy = _policy(beta=0.0)
    selected = tuple(
        sorted(value.action_sha256 for value in market[:3])
    )
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    incumbent = policy.terminal_policy().select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
    )
    decision = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
    )

    assert decision.phase == V8LITE_PHASE_PROTECTED_FALLBACK
    assert decision.selected_action_sha256s == (
        incumbent.selected_action_sha256s
    )
    assert decision.selection_propensity == (
        incumbent.selection_propensity
    )
    assert decision.delegated_decision.decision_sha256 == (
        incumbent.decision_sha256
    )
    top = decision.challenger_ranking.score_for(
        decision.challenger_ranking.ranked_action_sha256s[0]
    )
    assert top.score <= 0.0


def test_terminal_seat_delegates_exactly_to_v7() -> None:
    market = _market()
    by_lane_rank = {
        (value.lane_id, value.native_rank): value for value in market
    }
    diagnostic_actions = (
        by_lane_rank[("coverage", 1)],
        by_lane_rank[("interaction", 3)],
        by_lane_rank[("local", 1)],
    )
    directed_local = by_lane_rank[("local", 3)]
    selected = tuple(
        sorted(
            value.action_sha256
            for value in (*diagnostic_actions, directed_local)
        )
    )
    diagnostic = tuple(
        sorted(value.action_sha256 for value in diagnostic_actions)
    )
    outcomes = tuple(
        _outcome(
            value,
            (
                0.0005
                if value
                == by_lane_rank[("interaction", 3)].action_sha256
                else (
                    0.005
                    if value == directed_local.action_sha256
                    else 0.0
                )
            ),
        )
        for value in selected
    )
    policy = _policy(diagnostic_slots=3)
    raw_v7 = policy.terminal_policy().select_next(
        residual_request_sha256=_sha("v7-terminal"),
        actions=market,
        evaluation_slots=5,
        diagnostic_action_sha256s=diagnostic,
        diagnostic_joint_gain=0.0005,
        selected_action_sha256s=selected,
        outcomes=outcomes,
    )
    decision = policy.select_next(
        residual_request_sha256=_sha("v7-terminal"),
        actions=market,
        evaluation_slots=5,
        diagnostic_action_sha256s=diagnostic,
        diagnostic_joint_gain=0.0005,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
    )

    # The V7 terminal rule is preserved bit-for-bit by delegation.
    assert decision.phase == V8LITE_PHASE_TERMINAL
    assert decision.delegated_decision.decision_sha256 == (
        raw_v7.decision_sha256
    )
    assert decision.selected_action_sha256s == (
        raw_v7.selected_action_sha256s
    )
    evidence = raw_v7.to_record(include_evidence=True)["evidence"]
    # The terminal seat never buys an information-only audit when no
    # future optimization seat remains.
    assert raw_v7.wave is AdaptiveActionWave.ADAPTIVE
    assert evidence["authoritative_audit_due"] is True
    assert evidence["authoritative_audit_selected"] is False
    assert evidence["authoritative_audit_blocked_reason"] == (
        "terminal_hierarchical_exploitation"
    )
    assert evidence["terminal_hierarchical_allocation"] is True


def test_terminal_delegation_never_returns_a_randomized_audit() -> None:
    market = _market()
    policy = _policy()
    selected = tuple(
        sorted(value.action_sha256 for value in market[:7])
    )
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    decision = policy.select_next(
        residual_request_sha256=_sha("terminal-no-audit"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=tuple(sorted(selected[:4])),
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
    )

    assert decision.phase == V8LITE_PHASE_TERMINAL
    assert decision.delegated_decision.wave is (
        AdaptiveActionWave.ADAPTIVE
    )
    assert decision.selection_propensity == 1.0


def _r2_policy(**config_overrides: object) -> V8LiteAllocationPolicy:
    defaults: dict[str, object] = {
        "reference_gain_scale": 0.001,
        "reference_gain_evidence_sha256": _sha("prior-evidence"),
        "random_seed": 17,
    }
    defaults.update(config_overrides)
    return V8LiteAllocationPolicy(
        archive_gain_utility=_Hv2GainPort((10.0, 10.0)),
        policy_version_id="v8lite_r2",
        config=V8LiteAllocationConfig(**defaults),
    )


def test_r1_definition_sha_is_frozen_and_r2_differs() -> None:
    # Default-config r1 against the test gain port: pinned at the
    # value frozen before the r2 repair (verified identical at every
    # commit since the r1 release), so r2 cannot drift r1.
    r1_default = V8LiteAllocationPolicy(
        archive_gain_utility=_Hv2GainPort((10.0, 10.0))
    )
    assert r1_default.definition_sha256 == (
        "a6d86f0e97aecba9d6ac6c3ed57a3df0"
        "b1d8d0ad5690b4481f0ea009258e8f66"
    )
    r1 = _policy()
    r2 = _r2_policy()
    assert r2.definition_sha256 != r1.definition_sha256
    assert r2.policy_version_id == "v8lite_r2"
    identity = r2.identity_record()
    assert identity["phases"]["pilot"] == (
        "sequential_adaptive_band_pilot"
    )
    assert identity["r2"][
        "pilot_seats_sequential_with_revealed_outcomes"
    ] is True


def test_r2_pilot_width_cap_hands_seats_to_the_challenger() -> None:
    r1 = _policy()
    r2 = _r2_policy()
    # Three engines at budget eight: r1 claims four pilot seats, r2
    # caps at max(3, ceil(8/3)) = 3.
    assert (
        r1.pilot_width_for(evaluation_slots=8, engine_count=3) == 4
    )
    assert (
        r2.pilot_width_for(evaluation_slots=8, engine_count=3) == 3
    )
    # Many-engine narrow markets keep full coverage capacity.
    assert (
        r2.pilot_width_for(evaluation_slots=8, engine_count=7) == 4
    )


def test_r2_sequential_pilot_seats_have_exact_propensities() -> None:
    from fractions import Fraction

    market = _market()
    policy = _r2_policy()
    selected: tuple[str, ...] = ()
    outcomes: tuple[AdaptiveActionOutcome, ...] = ()
    for _seat in range(
        policy.pilot_width_for(evaluation_slots=8, engine_count=3)
    ):
        decision = policy.design_pilot_seat(
            residual_request_sha256=_sha("r2-pilot"),
            actions=market,
            evaluation_slots=8,
            selected_action_sha256s=selected,
            outcomes=outcomes,
        )
        assert decision.phase == V8LITE_PHASE_PILOT
        assert decision.policy_version_id == "v8lite_r2"
        chosen = decision.selected_action_sha256s[0]
        from agent_evolve.domain.typed_json import thaw_json

        seat = thaw_json(decision.evidence)["pilot_seat"]
        total = sum(
            Fraction(
                value["propensity_numerator"],
                value["propensity_denominator"],
            )
            for value in seat["support_propensities"]
        )
        assert total == Fraction(1)
        selected = tuple(sorted((*selected, chosen)))
        outcomes = (*outcomes, _outcome(chosen, 0.0))
    # The completed pilot rejects further seats.
    import pytest

    with pytest.raises(ValueError, match="already complete"):
        policy.design_pilot_seat(
            residual_request_sha256=_sha("r2-pilot"),
            actions=market,
            evaluation_slots=8,
            selected_action_sha256s=selected,
            outcomes=outcomes,
        )


def test_r2_terminal_seat_still_delegates_exactly_to_v7() -> None:
    market = _market()
    by_lane_rank = {
        (value.lane_id, value.native_rank): value for value in market
    }
    diagnostic_actions = (
        by_lane_rank[("coverage", 1)],
        by_lane_rank[("interaction", 3)],
        by_lane_rank[("local", 1)],
    )
    directed_local = by_lane_rank[("local", 3)]
    selected = tuple(
        sorted(
            value.action_sha256
            for value in (*diagnostic_actions, directed_local)
        )
    )
    diagnostic = tuple(
        sorted(value.action_sha256 for value in diagnostic_actions)
    )
    outcomes = tuple(
        _outcome(
            value,
            (
                0.0005
                if value
                == by_lane_rank[("interaction", 3)].action_sha256
                else (
                    0.005
                    if value == directed_local.action_sha256
                    else 0.0
                )
            ),
        )
        for value in selected
    )
    r2 = _r2_policy(diagnostic_slots=3)
    raw_v7 = r2.terminal_policy().select_next(
        residual_request_sha256=_sha("v7-terminal"),
        actions=market,
        evaluation_slots=5,
        diagnostic_action_sha256s=diagnostic,
        diagnostic_joint_gain=0.0005,
        selected_action_sha256s=selected,
        outcomes=outcomes,
    )
    decision = r2.select_next(
        residual_request_sha256=_sha("v7-terminal"),
        actions=market,
        evaluation_slots=5,
        diagnostic_action_sha256s=diagnostic,
        diagnostic_joint_gain=0.0005,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
    )

    assert decision.phase == V8LITE_PHASE_TERMINAL
    assert decision.delegated_decision.decision_sha256 == (
        raw_v7.decision_sha256
    )
    assert raw_v7.wave is AdaptiveActionWave.ADAPTIVE


def test_r2_challenger_breaks_cell_ties_by_native_rank() -> None:
    market = _market()
    selected = tuple(
        sorted(value.action_sha256 for value in market[:3])
    )
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    kwargs = dict(
        residual_request_sha256=_sha("tie-break"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
    )
    by_action = {value.action_sha256: value for value in market}
    r1_ranking = _policy().select_next(**kwargs).challenger_ranking
    r2_ranking = _r2_policy().select_next(**kwargs).challenger_ranking

    def _tied_block(ranking):
        scores = {
            value.action_sha256: value.score
            for value in ranking.scores
        }
        top = ranking.ranked_action_sha256s[0]
        return [
            action
            for action in ranking.ranked_action_sha256s
            if scores[action] == scores[top]
        ]

    r2_block = _tied_block(r2_ranking)
    if len(r2_block) > 1:
        qualities = [
            by_action[action].rank_quality for action in r2_block
        ]
        assert qualities == sorted(qualities, reverse=True)
    assert set(r1_ranking.ranked_action_sha256s) == set(
        r2_ranking.ranked_action_sha256s
    )


def test_warm_prior_conversion_evidence_shifts_the_challenger() -> None:
    from agent_evolve.application.calibrated_positive_gain_opportunity import (
        ObservedConversionOutcome,
    )

    market = _market()
    policy = _policy()
    selected = tuple(
        sorted(value.action_sha256 for value in market[:3])
    )
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    warm_evidence = tuple(
        ObservedConversionOutcome(
            observation_ordinal=ordinal,
            engine_id="interaction",
            native_rank=1,
            lane_size=3,
            feasible=True,
            marginal_archive_gain=0.02,
        )
        for ordinal in range(1, 9)
    )
    cold = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
    )
    warm = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
        prior_conversion_outcomes=warm_evidence,
    )

    by_action = {value.action_sha256: value for value in market}
    warm_ranking = warm.challenger_ranking
    cold_ranking = cold.challenger_ranking
    top_warm = by_action[warm_ranking.ranked_action_sha256s[0]]
    # Strong warm interaction evidence lifts that engine to the top of
    # the challenger ranking and raises its conversion posterior.
    assert top_warm.lane_id == "interaction"
    interaction_id = next(
        value.action_sha256
        for value in market
        if value.lane_id == "interaction"
        and value.action_sha256
        in set(warm_ranking.ranked_action_sha256s)
    )
    assert warm_ranking.score_for(
        interaction_id
    ).conversion_probability > cold_ranking.score_for(
        interaction_id
    ).conversion_probability


def test_frozen_prior_score_requires_training_history() -> None:
    market = _market()
    policy = _policy(beta=0.0)
    selected = tuple(
        sorted(value.action_sha256 for value in market[:3])
    )
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    sparse = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
        frozen_fit_training_run_count=6,
    )
    attested = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
        frozen_fit_training_run_count=12,
    )

    sparse_scores = sparse.challenger_ranking.scores
    assert all(
        value.frozen_evidence_weight == 0.0
        for value in sparse_scores
    )
    assert any(
        value.frozen_evidence_weight > 0.0
        for value in attested.challenger_ranking.scores
    )

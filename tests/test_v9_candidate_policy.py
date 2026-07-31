from __future__ import annotations

import hashlib

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    PositiveGainForecast,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionOutcome,
)
from agent_evolve.application.sequential_market_replay import (
    ExactHypervolumeGainPort,
    MarketCandidateRecord,
    MarketRecord,
    SequentialMarketReplay,
    V8LiteReplayPolicy,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LITE_PHASE_ADAPTIVE,
    V8LITE_PHASE_TERMINAL,
    V8LiteAllocationConfig,
    V8LiteAllocationPolicy,
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R2,
)
from agent_evolve.application.v9_candidate_policy import (
    V9CandidateConfig,
    V9CandidatePolicy,
    V9ReplayPolicy,
    v9_arm_version_id,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _point(x: float, y: float) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((("m.x", float(x)), ("m.y", float(y)))))


_REFERENCE = _point(10.0, 10.0)
_ARCHIVE = (_point(4.0, 4.0),)

#: label, engine, native rank, frozen score, objectives.
_MARKET_SHAPE = (
    ("a1", "engine.a", 1, 0.9, (2.0, 6.0)),
    ("a2", "engine.a", 2, 0.5, (5.0, 5.0)),
    ("a3", "engine.a", 3, 0.2, (6.0, 6.0)),
    ("b1", "engine.b", 1, 0.8, (6.0, 2.0)),
    ("b2", "engine.b", 2, 0.4, (7.0, 7.0)),
    ("b3", "engine.b", 3, 0.1, (3.0, 3.0)),
    ("c1", "engine.c", 1, 0.7, (8.0, 8.0)),
    ("c2", "engine.c", 2, 0.6, (1.0, 7.0)),
    ("c3", "engine.c", 3, 0.3, (9.0, 9.0)),
)


def _record(
    swap_labels: tuple[str, str] | None = None,
) -> MarketRecord:
    objectives_by_label = {
        label: objectives
        for label, _engine, _rank, _score, objectives in (
            _MARKET_SHAPE
        )
    }
    if swap_labels is not None:
        first, second = swap_labels
        objectives_by_label[first], objectives_by_label[second] = (
            objectives_by_label[second],
            objectives_by_label[first],
        )
    return MarketRecord(
        market_id="v9.market",
        archive_points=_ARCHIVE,
        hv_reference_point=_REFERENCE,
        candidates=tuple(
            MarketCandidateRecord(
                action_sha256=_sha(f"v9:{label}"),
                engine_id=engine,
                native_rank=rank,
                frozen_score=score,
                forecast=None,
                evaluated=True,
                feasible=True,
                objectives=_point(*objectives_by_label[label]),
            )
            for label, engine, rank, score, _objectives in (
                _MARKET_SHAPE
            )
        ),
    )


def _port() -> ExactHypervolumeGainPort:
    return ExactHypervolumeGainPort(_REFERENCE)


def _v9(
    *,
    r1: bool = False,
    r2: bool = False,
    r3: bool = False,
    seed: int = 0,
) -> V9CandidatePolicy:
    return V9CandidatePolicy(
        archive_gain_utility=_port(),
        config=V9CandidateConfig(
            r1_region_conditional_credit=r1,
            r2_head_mass_conditional_seat=r2,
            r3_geometry_conditional_elasticity=r3,
            base=V8LiteAllocationConfig(random_seed=seed),
        ),
    )


def _v8(seed: int = 0) -> V8LiteAllocationPolicy:
    return V8LiteAllocationPolicy(
        archive_gain_utility=_port(),
        policy_version_id=V8LITE_ALLOCATION_POLICY_VERSION_ID_R2,
        config=V8LiteAllocationConfig(random_seed=seed),
    )


_ARMS = (
    {"r1": False, "r2": False, "r3": False},
    {"r1": True, "r2": False, "r3": False},
    {"r1": False, "r2": True, "r3": False},
    {"r1": False, "r2": False, "r3": True},
    {"r1": True, "r2": True, "r3": True},
)


def test_arm_version_ids_and_definition_shas_are_distinct() -> None:
    shas = {}
    for flags in _ARMS:
        policy = _v9(**flags)
        expected = v9_arm_version_id(**flags)
        assert policy.policy_version_id == expected
        identity = policy.identity_record()
        assert identity["flags"] == {
            "r1": flags["r1"],
            "r2": flags["r2"],
            "r3": flags["r3"],
        }
        assert identity["definition_sha256"] == (
            policy.definition_sha256
        )
        assert identity["inner"]["policy_version_id"] == "v8lite_r2"
        assert identity["v7_terminal_rule_altered"] is False
        assert identity["live_authority"] is False
        shas[expected] = policy.definition_sha256
    assert len(set(shas.values())) == len(_ARMS)


def test_base_arm_replay_is_bit_identical_to_v8lite_r2() -> None:
    record = _record()
    harness = SequentialMarketReplay()
    for seed in (0, 3, 11):
        reference = harness.run(
            record=record,
            policy=V8LiteReplayPolicy(_v8(seed)),
            budget=5,
        )
        base = harness.run(
            record=record,
            policy=V9ReplayPolicy(_v9(seed=seed)),
            budget=5,
        )
        assert (
            base.selected_action_sha256s
            == reference.selected_action_sha256s
        )
        assert [
            value.selection_propensity for value in base.receipts
        ] == [
            value.selection_propensity
            for value in reference.receipts
        ]
        assert base.realized_gain == reference.realized_gain


def test_every_arm_replays_within_the_boundary() -> None:
    record = _record()
    harness = SequentialMarketReplay()
    for flags in _ARMS:
        result = harness.run(
            record=record,
            policy=V9ReplayPolicy(_v9(**flags, seed=2)),
            budget=5,
        )
        assert len(result.receipts) == 5
        assert len(set(result.selected_action_sha256s)) == 5
        assert all(
            0.0 < value.selection_propensity <= 1.0
            for value in result.receipts
        )
        assert result.oracle_gain >= result.realized_gain


def test_terminal_seat_delegates_exactly_to_v7_for_every_arm() -> None:
    market = tuple(
        V8LiteReplayPolicy(_v8())._descriptors(
            _record(),
            tuple(
                sorted(
                    value.action_sha256
                    for value in _record().candidates
                )
            ),
        )
    )
    selected = tuple(
        sorted(value.action_sha256 for value in market[:4])
    )
    outcomes = tuple(
        AdaptiveActionOutcome(
            action_sha256=value,
            evaluation_sha256=_sha(f"evaluation:{value}"),
            feasible=True,
            marginal_archive_gain=0.0,
        )
        for value in selected
    )
    inner_decision = _v8().select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=5,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
    )
    assert inner_decision.phase == V8LITE_PHASE_TERMINAL
    for flags in _ARMS:
        decision = _v9(**flags).select_next(
            residual_request_sha256=_sha("request"),
            actions=market,
            evaluation_slots=5,
            diagnostic_action_sha256s=selected,
            diagnostic_joint_gain=0.0,
            selected_action_sha256s=selected,
            outcomes=outcomes,
            archive_points=_ARCHIVE,
            reference_point=_REFERENCE,
        )
        # The terminal seat is the inner v8lite decision object,
        # which itself is the EXACT V7 delegation: bit-identical
        # records for every arm.
        assert decision.to_record() == inner_decision.to_record()


def test_permuting_unrevealed_outcomes_never_changes_selections() -> None:
    # The no-future-leak property for the composed arm: swapping the
    # outcomes of candidates the policy never selected must not change
    # any selection.
    sha_to_label = {
        _sha(f"v9:{label}"): label
        for label, _engine, _rank, _score, _objectives in (
            _MARKET_SHAPE
        )
    }
    harness = SequentialMarketReplay()
    for flags in _ARMS:
        base = harness.run(
            record=_record(),
            policy=V9ReplayPolicy(_v9(**flags, seed=9)),
            budget=4,
        )
        unselected = tuple(
            sorted(
                label
                for sha, label in sha_to_label.items()
                if sha not in set(base.selected_action_sha256s)
            )
        )
        assert len(unselected) >= 2
        permuted = harness.run(
            record=_record(
                swap_labels=(unselected[0], unselected[1])
            ),
            policy=V9ReplayPolicy(_v9(**flags, seed=9)),
            budget=4,
        )
        assert (
            permuted.selected_action_sha256s
            == base.selected_action_sha256s
        )


def test_r1_adaptive_seat_carries_region_authority() -> None:
    market = tuple(
        V8LiteReplayPolicy(_v8())._descriptors(
            _record(),
            tuple(
                sorted(
                    value.action_sha256
                    for value in _record().candidates
                )
            ),
        )
    )
    selected = tuple(
        sorted(value.action_sha256 for value in market[:3])
    )
    outcomes = tuple(
        AdaptiveActionOutcome(
            action_sha256=value,
            evaluation_sha256=_sha(f"evaluation:{value}"),
            feasible=True,
            marginal_archive_gain=0.0,
        )
        for value in selected
    )
    strong_action = market[5].action_sha256
    forecast = PositiveGainForecast(
        quantile_points=(
            ("p10", _point(6.0, 3.0)),
            ("p50", _point(5.0, 2.0)),
            ("p90", _point(4.0, 1.0)),
        ),
    )
    decision = _v9(r1=True).select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=selected,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
        archive_points=_ARCHIVE,
        reference_point=_REFERENCE,
        forecasts=((strong_action, forecast),),
    )
    assert decision.phase == V8LITE_PHASE_ADAPTIVE
    assert decision.authority_policy_id == (
        "region_conditional_credit"
    )
    assert decision.selected_action_sha256s == (strong_action,)
    assert decision.selection_propensity == 1.0
    assert decision.policy_version_id == "v9_r1"


def test_r1_protected_fallback_matches_the_frozen_incumbent() -> None:
    market = tuple(
        V8LiteReplayPolicy(_v8())._descriptors(
            _record(),
            tuple(
                sorted(
                    value.action_sha256
                    for value in _record().candidates
                )
            ),
        )
    )
    selected = tuple(
        sorted(value.action_sha256 for value in market[:3])
    )
    outcomes = tuple(
        AdaptiveActionOutcome(
            action_sha256=value,
            evaluation_sha256=_sha(f"evaluation:{value}"),
            feasible=True,
            marginal_archive_gain=0.0,
        )
        for value in selected
    )
    policy = V9CandidatePolicy(
        archive_gain_utility=_port(),
        config=V9CandidateConfig(
            r1_region_conditional_credit=True,
            base=V8LiteAllocationConfig(beta=0.0),
        ),
    )
    incumbent = policy.inner_policy().terminal_policy().select_next(
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
        reference_point=_REFERENCE,
    )
    assert decision.phase == "protected_fallback"
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

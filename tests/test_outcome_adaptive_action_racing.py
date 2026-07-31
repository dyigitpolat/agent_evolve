from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionFactorCell,
    AdaptiveActionOutcome,
    AdaptiveActionSetOutcome,
    AdaptiveActionWave,
    OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION,
    OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION,
    OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION,
    OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION,
    OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION,
    OutcomeAdaptiveActionRacingPolicy,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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


def _factor_market() -> tuple[AdaptiveActionDescriptor, ...]:
    values = []
    for branch in ("branch_a", "branch_b"):
        for role in ("coverage", "interaction", "local"):
            lane = f"{branch}.{role}"
            for rank in (1, 2):
                values.append(
                    AdaptiveActionDescriptor(
                        action_sha256=_sha(
                            f"factor-action:{branch}:{role}:{rank}"
                        ),
                        phenotype_sha256=_sha(
                            f"factor-phenotype:{branch}:{role}:{rank}"
                        ),
                        lane_id=lane,
                        operator_id=f"{role}.r{rank}",
                        native_rank=rank,
                        lane_size=2,
                        prior_score=float(3 - rank) / 2.0,
                        parent_generated_in_current_run=False,
                        semantic_cell_ids=(f"role:{role}",),
                        factor_cells=tuple(
                            sorted(
                                (
                                    AdaptiveActionFactorCell(
                                        family_id="evolutionary_role",
                                        level_id=role,
                                    ),
                                    AdaptiveActionFactorCell(
                                        family_id=(
                                            "materialized_rank_layer"
                                        ),
                                        level_id=f"layer{rank - 1}",
                                    ),
                                    AdaptiveActionFactorCell(
                                        family_id="source_branch",
                                        level_id=branch,
                                    ),
                                    AdaptiveActionFactorCell(
                                        family_id="source_branch_role",
                                        level_id=f"{branch}:{role}",
                                    ),
                                )
                            )
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


def _set_outcome(
    *,
    prior_action_sha256s: tuple[str, ...],
    current_action_sha256s: tuple[str, ...],
    prior_gain: float,
    fixed_gain: float,
    conditional_gain: float,
) -> AdaptiveActionSetOutcome:
    return AdaptiveActionSetOutcome(
        prior_action_evaluation_bindings=tuple(
            sorted(
                (
                    value,
                    _sha(f"evaluation:{value}"),
                )
                for value in prior_action_sha256s
            )
        ),
        current_action_evaluation_bindings=tuple(
            sorted(
                (
                    value,
                    _sha(f"evaluation:{value}"),
                )
                for value in current_action_sha256s
            )
        ),
        prior_selected_set_gain=float(prior_gain),
        current_wave_fixed_set_gain=float(fixed_gain),
        augmented_selected_set_gain=float(
            prior_gain + conditional_gain
        ),
        conditional_set_gain=float(conditional_gain),
    )


def _policy(**overrides: object) -> OutcomeAdaptiveActionRacingPolicy:
    values: dict[str, object] = {
        "diagnostic_slots": 4,
        "randomized_audit_slots": 1,
        "reference_gain_scale": 0.001,
        "reference_gain_evidence_sha256": _sha("prior-evidence"),
        "random_seed": 17,
    }
    values.update(overrides)
    return OutcomeAdaptiveActionRacingPolicy(**values)


def _v7_policy(**overrides: object) -> OutcomeAdaptiveActionRacingPolicy:
    values: dict[str, object] = {
        "policy_version": (
            OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
        ),
        "trace_alternative_count": 9,
        "randomized_audit_after_directed_steps": 1,
        "minimum_post_audit_optimization_slots": 1,
        "terminal_hierarchical_slots": 1,
        "native_rank_strength": 0.25,
    }
    values.update(overrides)
    return _policy(**values)


def test_diagnostic_pilot_covers_every_lane_before_maximin_fill() -> None:
    market = _market()
    policy = _policy()
    decision = policy.design_diagnostic_pilot(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
    )

    assert decision.wave is AdaptiveActionWave.DIAGNOSTIC
    assert decision.candidate_outcomes_observed is False
    assert len(decision.selected_action_sha256s) == 4
    by_action = {value.action_sha256: value for value in market}
    selected = [by_action[value] for value in decision.selected_action_sha256s]
    assert {value.lane_id for value in selected} == {
        "coverage",
        "interaction",
        "local",
    }
    assert sum(value.native_rank == 1 for value in selected) == 3
    assert decision.to_record(include_evidence=True)["evidence"][
        "lane_head_count"
    ] == 3


def test_diagnostic_design_is_deterministic_and_outcome_blind() -> None:
    market = _market()
    policy = _policy()
    first = policy.design_diagnostic_pilot(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
    )
    second = policy.design_diagnostic_pilot(
        residual_request_sha256=_sha("request"),
        actions=tuple(reversed(market)),
        evaluation_slots=8,
    )

    assert first.selected_action_sha256s == second.selected_action_sha256s
    assert first.decision_sha256 == second.decision_sha256
    assert first.observed_outcome_sha256s == ()


def test_diagnostic_pilot_preserves_authenticated_fixed_constraints() -> None:
    market = _market()
    policy = _policy()
    required = tuple(
        sorted(
            (
                market[2].action_sha256,
                market[5].action_sha256,
            )
        )
    )
    decision = policy.design_diagnostic_pilot(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        required_action_sha256s=required,
    )

    assert set(required).issubset(decision.selected_action_sha256s)
    evidence = decision.to_record(include_evidence=True)["evidence"]
    assert evidence["fixed_constraint_count"] == 2
    assert evidence["required_action_sha256s"] == list(required)
    assert decision.candidate_outcomes_observed is False


def test_continuation_rejects_any_outcome_outside_selected_prefix() -> None:
    market = _market()
    policy = _policy()
    pilot = policy.design_diagnostic_pilot(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
    )
    extra = next(
        value
        for value in market
        if value.action_sha256 not in pilot.selected_action_sha256s
    )
    outcomes = tuple(
        _outcome(value, 0.0) for value in pilot.selected_action_sha256s
    ) + (_outcome(extra.action_sha256, 999.0),)

    with pytest.raises(
        ValueError,
        match="exactly cover all previously selected",
    ):
        policy.select_next(
            residual_request_sha256=_sha("request"),
            actions=market,
            evaluation_slots=8,
            diagnostic_action_sha256s=pilot.selected_action_sha256s,
            diagnostic_joint_gain=0.0,
            selected_action_sha256s=pilot.selected_action_sha256s,
            outcomes=outcomes,
        )


def test_positive_pilot_changes_adaptive_evidence_without_unsealed_input() -> None:
    market = _market()
    policy = _policy()
    pilot = policy.design_diagnostic_pilot(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
    )
    by_action = {value.action_sha256: value for value in market}
    positive_id = next(
        value
        for value in pilot.selected_action_sha256s
        if by_action[value].lane_id == "coverage"
    )
    outcomes = tuple(
        _outcome(value, 0.01 if value == positive_id else 0.0)
        for value in pilot.selected_action_sha256s
    )
    decision = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.01,
        selected_action_sha256s=pilot.selected_action_sha256s,
        outcomes=outcomes,
    )

    assert decision.wave is AdaptiveActionWave.ADAPTIVE
    assert decision.candidate_outcomes_observed is True
    assert decision.selection_propensity == 1.0
    assert decision.observed_outcome_sha256s == tuple(
        sorted(value.outcome_sha256 for value in outcomes)
    )
    record = decision.to_record(include_evidence=True)
    assert record["evidence"]["unobserved_candidate_outcomes_available"] is False
    assert float.fromhex(
        record["evidence"]["selected_components"]["ucb_index"]
    ) > 0.0


def test_last_seat_is_randomized_over_logged_exploration_pool() -> None:
    market = _market()
    policy = _policy(exploration_pool_size=3)
    selected = tuple(
        sorted(value.action_sha256 for value in market[:7])
    )
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    decision = policy.select_next(
        residual_request_sha256=_sha("request"),
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=tuple(sorted(selected[:4])),
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
    )

    assert decision.wave is AdaptiveActionWave.RANDOMIZED_AUDIT
    evidence = decision.to_record(include_evidence=True)["evidence"]
    pool = evidence["randomized_exploration_pool_action_sha256s"]
    assert decision.selected_action_sha256s[0] in pool
    assert decision.selection_propensity == 1.0 / len(pool)


def test_market_rejects_duplicate_phenotypes() -> None:
    market = list(_market())
    original = market[0]
    duplicate = market[1]
    market[1] = AdaptiveActionDescriptor(
        action_sha256=duplicate.action_sha256,
        phenotype_sha256=original.phenotype_sha256,
        lane_id=duplicate.lane_id,
        operator_id=duplicate.operator_id,
        native_rank=duplicate.native_rank,
        lane_size=duplicate.lane_size,
        prior_score=duplicate.prior_score,
        parent_generated_in_current_run=(
            duplicate.parent_generated_in_current_run
        ),
        semantic_cell_ids=duplicate.semantic_cell_ids,
    )

    with pytest.raises(ValueError, match="unique phenotype"):
        _policy().design_diagnostic_pilot(
            residual_request_sha256=_sha("request"),
            actions=tuple(market),
            evaluation_slots=8,
        )


def test_v2_definition_remains_compatible_with_the_v51_trace() -> None:
    policy = OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=4,
        randomized_audit_slots=1,
        reference_gain_scale=float.fromhex(
            "0x1.5d3bee284c000p-15"
        ),
        reference_gain_evidence_sha256=(
            "72ffedd18d166d697efed5b269ec8847e"
            "e997b7c053bf4eff6857c6932ac36de"
        ),
        prior_strength=2.0,
        ucb_strength=1.0,
        counterfactual_strength=0.5,
        diversity_strength=0.25,
        exploration_pool_size=4,
        random_seed=260726,
    )

    assert policy.definition_sha256 == (
        "49f3793e44805c78ee3db5f01d0075d7"
        "30482f1fca6cf9b2aeb8da501ebbe590"
    )


def test_set_aware_controls_require_an_auditable_v3_policy() -> None:
    with pytest.raises(ValueError, match="require policy version 3"):
        _policy(positive_redundancy_strength=0.25)

    with pytest.raises(ValueError, match="requires auditable alternatives"):
        _policy(
            policy_version=(
                OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION
            )
        )


def test_v3_penalizes_repeating_a_realized_positive_neighborhood() -> None:
    market = _market()
    request_sha256 = _sha("request")
    v2 = _policy()
    pilot = v2.design_diagnostic_pilot(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
    )
    by_action = {value.action_sha256: value for value in market}
    positive_id = next(
        value
        for value in pilot.selected_action_sha256s
        if by_action[value].lane_id == "coverage"
        and by_action[value].native_rank == 1
    )
    outcomes = tuple(
        _outcome(value, 0.01 if value == positive_id else 0.0)
        for value in pilot.selected_action_sha256s
    )
    baseline = v2.select_next(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.01,
        selected_action_sha256s=pilot.selected_action_sha256s,
        outcomes=outcomes,
    )
    set_aware = _policy(
        policy_version=(
            OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION
        ),
        positive_redundancy_strength=0.25,
        trace_alternative_count=5,
    ).select_next(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.01,
        selected_action_sha256s=pilot.selected_action_sha256s,
        outcomes=outcomes,
    )

    baseline_action = by_action[baseline.selected_action_sha256s[0]]
    set_aware_action = by_action[set_aware.selected_action_sha256s[0]]
    assert baseline_action.lane_id == "coverage"
    assert set_aware_action.lane_id != "coverage"
    evidence = set_aware.to_record(include_evidence=True)["evidence"]
    assert evidence["candidate_score_card_count"] == 5
    assert evidence["chosen_action_present_in_score_cards"] is True
    assert sum(
        bool(value["selected"])
        for value in evidence["candidate_score_cards"]
    ) == 1
    assert float.fromhex(
        evidence["selected_components"]["redundancy_penalty"]
    ) > 0.0


def test_v4_uses_real_conditional_lift_and_moves_audit_midwave() -> None:
    market = _market()
    request_sha256 = _sha("causal-request")
    policy = _policy(
        policy_version=(
            OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION
        ),
        conditional_saturation_strength=1.0,
        conditional_synergy_strength=0.25,
        causal_prior_strength=1.0,
        randomized_audit_after_directed_steps=1,
        trace_alternative_count=5,
    )
    pilot = policy.design_diagnostic_pilot(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
    )
    outcomes = tuple(
        _outcome(value, 0.01 if ordinal == 0 else 0.0)
        for ordinal, value in enumerate(
            pilot.selected_action_sha256s
        )
    )
    diagnostic = _set_outcome(
        prior_action_sha256s=(),
        current_action_sha256s=pilot.selected_action_sha256s,
        prior_gain=0.0,
        fixed_gain=0.01,
        conditional_gain=0.01,
    )

    directed = policy.select_next(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.01,
        selected_action_sha256s=pilot.selected_action_sha256s,
        outcomes=outcomes,
        set_outcomes=(diagnostic,),
    )
    assert directed.wave is AdaptiveActionWave.ADAPTIVE
    directed_action = directed.selected_action_sha256s[0]
    directed_outcome = _outcome(directed_action, 0.005)
    saturated = _set_outcome(
        prior_action_sha256s=pilot.selected_action_sha256s,
        current_action_sha256s=(directed_action,),
        prior_gain=0.01,
        fixed_gain=0.005,
        conditional_gain=0.0,
    )
    selected_after_directed = tuple(
        sorted((*pilot.selected_action_sha256s, directed_action))
    )

    audit = policy.select_next(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.01,
        selected_action_sha256s=selected_after_directed,
        outcomes=(*outcomes, directed_outcome),
        set_outcomes=(diagnostic, saturated),
    )
    assert audit.wave is AdaptiveActionWave.RANDOMIZED_AUDIT
    assert audit.observed_set_outcome_sha256s == tuple(
        sorted(
            (
                diagnostic.set_outcome_sha256,
                saturated.set_outcome_sha256,
            )
        )
    )
    audit_evidence = audit.to_record(include_evidence=True)["evidence"]
    assert audit_evidence[
        "causal_set_outcomes_used_for_selection"
    ] is True
    assert float.fromhex(
        audit_evidence["selected_components"][
            "posterior_saturation"
        ]
    ) > 0.0

    audit_action = audit.selected_action_sha256s[0]
    audit_outcome = _outcome(audit_action, 0.004)
    informative_audit = _set_outcome(
        prior_action_sha256s=selected_after_directed,
        current_action_sha256s=(audit_action,),
        prior_gain=0.01,
        fixed_gain=0.004,
        conditional_gain=0.004,
    )
    selected_after_audit = tuple(
        sorted((*selected_after_directed, audit_action))
    )
    post_audit = policy.select_next(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.01,
        selected_action_sha256s=selected_after_audit,
        outcomes=(*outcomes, directed_outcome, audit_outcome),
        set_outcomes=(
            diagnostic,
            saturated,
            informative_audit,
        ),
    )
    assert post_audit.wave is AdaptiveActionWave.ADAPTIVE
    post_evidence = post_audit.to_record(
        include_evidence=True
    )["evidence"]
    assert float.fromhex(
        post_evidence["selected_components"][
            "causal_evidence_count"
        ]
    ) == 2.0
    assert post_evidence[
        "directed_steps_completed_before_decision"
    ] == 2


def _v5_policy(
    *,
    random_seed: int,
    audit_after: int,
    epsilon: float = 0.25,
) -> OutcomeAdaptiveActionRacingPolicy:
    return _policy(
        policy_version=(
            OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
        ),
        conditional_saturation_strength=1.0,
        conditional_synergy_strength=0.0,
        causal_prior_strength=0.5,
        randomized_audit_after_directed_steps=audit_after,
        audit_exploration_probability=epsilon,
        trace_alternative_count=9,
        exploration_pool_size=4,
        random_seed=random_seed,
    )


def test_v5_requires_a_positive_risk_controlled_audit_epsilon() -> None:
    with pytest.raises(ValueError, match="positive epsilon"):
        _policy(
            policy_version=(
                OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
            ),
            conditional_saturation_strength=1.0,
            causal_prior_strength=0.5,
            trace_alternative_count=9,
        )


def test_v5_seed_affects_neither_diagnostic_nor_directed_choice() -> None:
    market = _market()
    request_sha256 = _sha("v5-seed-invariance")
    policies = tuple(
        _v5_policy(random_seed=value, audit_after=2)
        for value in (0, 17, 999)
    )
    pilots = tuple(
        policy.design_diagnostic_pilot(
            residual_request_sha256=request_sha256,
            actions=market,
            evaluation_slots=8,
        )
        for policy in policies
    )

    assert len(
        {value.selected_action_sha256s for value in pilots}
    ) == 1
    pilot = pilots[0]
    outcomes = tuple(
        _outcome(value, 0.01 if ordinal == 0 else 0.0)
        for ordinal, value in enumerate(
            pilot.selected_action_sha256s
        )
    )
    diagnostic = _set_outcome(
        prior_action_sha256s=(),
        current_action_sha256s=pilot.selected_action_sha256s,
        prior_gain=0.0,
        fixed_gain=0.01,
        conditional_gain=0.01,
    )
    decisions = tuple(
        policy.select_next(
            residual_request_sha256=request_sha256,
            actions=market,
            evaluation_slots=8,
            diagnostic_action_sha256s=pilot.selected_action_sha256s,
            diagnostic_joint_gain=0.01,
            selected_action_sha256s=pilot.selected_action_sha256s,
            outcomes=outcomes,
            set_outcomes=(diagnostic,),
        )
        for policy in policies
    )

    assert all(
        value.wave is AdaptiveActionWave.ADAPTIVE
        for value in decisions
    )
    assert len(
        {value.selected_action_sha256s for value in decisions}
    ) == 1


def test_v5_logs_exact_epsilon_greedy_audit_propensities() -> None:
    market = _market()
    request_sha256 = _sha("v5-audit-propensities")
    template = _v5_policy(random_seed=0, audit_after=0)
    pilot = template.design_diagnostic_pilot(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
    )
    outcomes = tuple(
        _outcome(value, 0.01 if ordinal == 0 else 0.0)
        for ordinal, value in enumerate(
            pilot.selected_action_sha256s
        )
    )
    diagnostic = _set_outcome(
        prior_action_sha256s=(),
        current_action_sha256s=pilot.selected_action_sha256s,
        prior_gain=0.0,
        fixed_gain=0.01,
        conditional_gain=0.01,
    )

    exploitation = None
    exploratory_non_anchor = None
    for seed in range(10_000):
        decision = _v5_policy(
            random_seed=seed,
            audit_after=0,
        ).select_next(
            residual_request_sha256=request_sha256,
            actions=market,
            evaluation_slots=8,
            diagnostic_action_sha256s=pilot.selected_action_sha256s,
            diagnostic_joint_gain=0.01,
            selected_action_sha256s=pilot.selected_action_sha256s,
            outcomes=outcomes,
            set_outcomes=(diagnostic,),
        )
        evidence = decision.to_record(include_evidence=True)[
            "evidence"
        ]
        if evidence["audit_exploration_branch"] is False:
            exploitation = (decision, evidence)
        if (
            evidence["audit_exploration_branch"] is True
            and decision.selected_action_sha256s[0]
            != evidence["audit_anchor_action_sha256"]
        ):
            exploratory_non_anchor = (decision, evidence)
        if exploitation is not None and exploratory_non_anchor is not None:
            break

    assert exploitation is not None
    assert exploratory_non_anchor is not None
    exploitation_decision, exploitation_evidence = exploitation
    exploratory_decision, exploratory_evidence = exploratory_non_anchor
    pool_size = len(
        exploitation_evidence[
            "randomized_exploration_pool_action_sha256s"
        ]
    )
    assert pool_size == 4
    assert exploitation_decision.selected_action_sha256s[0] == (
        exploitation_evidence["audit_anchor_action_sha256"]
    )
    assert exploitation_decision.selection_propensity == pytest.approx(
        1.0 - 0.25 + 0.25 / pool_size
    )
    assert exploratory_decision.selection_propensity == pytest.approx(
        0.25 / pool_size
    )
    assert exploitation_evidence["seed_affects_only_audit_draw"] is True


def _v6_policy(*, random_seed: int) -> OutcomeAdaptiveActionRacingPolicy:
    return _policy(
        policy_version=(
            OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
        ),
        audit_exploration_probability=0.1,
        trace_alternative_count=12,
        exploration_pool_size=4,
        stratified_audit_coverage_family_ids=(
            "source_branch_role",
        ),
        stratified_audit_stratum_family_ids=(
            "evolutionary_role",
            "materialized_rank_layer",
            "source_branch",
        ),
        random_seed=random_seed,
    )


def test_v6_preserves_legacy_anchor_and_logs_stratified_propensity() -> None:
    market = _factor_market()
    by_lane_rank = {
        (value.lane_id, value.native_rank): value for value in market
    }
    selected = tuple(
        sorted(
            (
                by_lane_rank[("branch_a.coverage", 1)].action_sha256,
                by_lane_rank[("branch_a.coverage", 2)].action_sha256,
                by_lane_rank[("branch_a.interaction", 1)].action_sha256,
                by_lane_rank[("branch_a.interaction", 2)].action_sha256,
                by_lane_rank[("branch_a.local", 1)].action_sha256,
                by_lane_rank[("branch_b.coverage", 1)].action_sha256,
                by_lane_rank[("branch_b.interaction", 1)].action_sha256,
            )
        )
    )
    diagnostic = tuple(sorted(selected[:4]))
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    request_sha256 = _sha("v6-stratified-audit")
    expected_support = {
        by_lane_rank[("branch_b.local", rank)].action_sha256
        for rank in (1, 2)
    }
    exploitation = None
    exploratory_non_anchor = None
    for seed in range(10_000):
        legacy = _policy(
            policy_version=(
                OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION
            ),
            trace_alternative_count=12,
            exploration_pool_size=4,
            random_seed=seed,
        ).select_next(
            residual_request_sha256=request_sha256,
            actions=market,
            evaluation_slots=8,
            diagnostic_action_sha256s=diagnostic,
            diagnostic_joint_gain=0.0,
            selected_action_sha256s=selected,
            outcomes=outcomes,
        )
        decision = _v6_policy(random_seed=seed).select_next(
            residual_request_sha256=request_sha256,
            actions=market,
            evaluation_slots=8,
            diagnostic_action_sha256s=diagnostic,
            diagnostic_joint_gain=0.0,
            selected_action_sha256s=selected,
            outcomes=outcomes,
        )
        evidence = decision.to_record(include_evidence=True)["evidence"]
        assert evidence["legacy_audit_anchor_action_sha256"] == (
            legacy.selected_action_sha256s[0]
        )
        assert set(
            evidence["randomized_exploration_pool_action_sha256s"]
        ) == expected_support
        assert evidence["audit_max_uncovered_factor_count"] == 1
        if evidence["audit_exploration_branch"] is False:
            exploitation = (decision, evidence)
        if (
            evidence["audit_exploration_branch"] is True
            and decision.selected_action_sha256s[0]
            != evidence["legacy_audit_anchor_action_sha256"]
        ):
            exploratory_non_anchor = (decision, evidence)
        if exploitation is not None and exploratory_non_anchor is not None:
            break

    assert exploitation is not None
    assert exploratory_non_anchor is not None
    for decision, evidence in (exploitation, exploratory_non_anchor):
        selected_action = decision.selected_action_sha256s[0]
        exploration_propensity = 0.0
        for stratum in evidence["audit_strata"]:
            if selected_action in stratum["action_sha256s"]:
                exploration_propensity = (
                    0.1
                    / len(evidence["audit_strata"])
                    / len(stratum["action_sha256s"])
                )
                break
        expected = exploration_propensity
        if selected_action == evidence["legacy_audit_anchor_action_sha256"]:
            expected += 0.9
        assert decision.selection_propensity == pytest.approx(expected)
        assert evidence["candidate_factor_cells_outcome_blind"] is True


def test_v7_terminal_horizon_blocks_audit_and_allocates_engine_then_action() -> None:
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
    decision = _v7_policy(
        diagnostic_slots=3,
        reference_gain_scale=0.001,
    ).select_next(
        residual_request_sha256=_sha("v7-terminal"),
        actions=market,
        evaluation_slots=5,
        diagnostic_action_sha256s=diagnostic,
        diagnostic_joint_gain=0.0005,
        selected_action_sha256s=selected,
        outcomes=outcomes,
    )

    selected_action = next(
        value
        for value in market
        if value.action_sha256 == decision.selected_action_sha256s[0]
    )
    evidence = decision.to_record(include_evidence=True)["evidence"]
    assert decision.wave is AdaptiveActionWave.ADAPTIVE
    assert selected_action.lane_id == "interaction"
    assert selected_action.native_rank in {1, 2}
    assert evidence["authoritative_audit_due"] is True
    assert evidence["authoritative_audit_selected"] is False
    assert evidence["authoritative_audit_blocked_reason"] == (
        "terminal_hierarchical_exploitation"
    )
    assert evidence["terminal_hierarchical_allocation"] is True
    assert evidence["hierarchical_engine_order"][0] == "interaction"
    assert evidence["selected_engine_id"] == "interaction"
    assert float.fromhex(
        evidence["selected_components"]["native_rank_bonus"]
    ) > 0.0
    assert evidence["engine_returns_capped_only_for_allocation"] is True
    assert evidence["authoritative_archive_utility_capped"] is False


def test_v7_allows_early_audit_only_with_future_optimization_horizon() -> None:
    market = _market()
    policy = _v7_policy()
    request_sha256 = _sha("v7-early-audit")
    pilot = policy.design_diagnostic_pilot(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
    )
    pilot_outcomes = tuple(
        _outcome(value, 0.0) for value in pilot.selected_action_sha256s
    )
    directed = policy.select_next(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=pilot.selected_action_sha256s,
        outcomes=pilot_outcomes,
    )
    assert directed.wave is AdaptiveActionWave.ADAPTIVE
    selected = tuple(
        sorted(
            (
                *pilot.selected_action_sha256s,
                directed.selected_action_sha256s[0],
            )
        )
    )
    outcomes = (
        *pilot_outcomes,
        _outcome(directed.selected_action_sha256s[0], 0.0),
    )
    audit = policy.select_next(
        residual_request_sha256=request_sha256,
        actions=market,
        evaluation_slots=8,
        diagnostic_action_sha256s=pilot.selected_action_sha256s,
        diagnostic_joint_gain=0.0,
        selected_action_sha256s=selected,
        outcomes=outcomes,
    )

    evidence = audit.to_record(include_evidence=True)["evidence"]
    assert audit.wave is AdaptiveActionWave.RANDOMIZED_AUDIT
    assert evidence["authoritative_audit_due"] is True
    assert evidence["authoritative_audit_selected"] is True
    assert evidence["future_optimization_slots_after_decision"] == 2
    assert evidence["terminal_hierarchical_allocation"] is False

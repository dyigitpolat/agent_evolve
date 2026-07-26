"""Randomized subset retrieval and causal-credit policy contracts."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest

from agent_evolve.domain.ids import CandidateId, InsightId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.policies.memory.randomized_subset import (
    EpsilonGreedySubsetSelector,
    InsightSelectionDecision,
    InsightSelectionMode,
    InsightTrial,
    estimate_marginal_effect,
    estimate_pair_synergy,
)

CONTEXT_A = "a" * 64
CONTEXT_B = "b" * 64
A = InsightRef(InsightId("insight_a"), 1)
B = InsightRef(InsightId("insight_b"), 1)
C = InsightRef(InsightId("insight_c"), 2)
D = InsightRef(InsightId("insight_d"), 1)
ELIGIBLE = (A, B, C, D)
SCORES = {A: 4.0, B: 3.0, C: 2.0, D: 1.0}
REWARD = "c" * 64


class ScriptedRandom:
    def __init__(self, *, draw=0, sample=()):
        self.draw = draw
        self.sampled = list(sample)
        self.randrange_calls = 0
        self.randrange_stops = []
        self.sample_calls = 0

    def randrange(self, stop):
        self.randrange_calls += 1
        self.randrange_stops.append(stop)
        assert type(stop) is int and stop > 0
        return self.draw

    def sample(self, population, k):
        self.sample_calls += 1
        assert set(self.sampled).issubset(population)
        assert len(self.sampled) == k
        return list(self.sampled)


def _uniform_decision(selected, *, context=CONTEXT_A):
    return EpsilonGreedySubsetSelector(Fraction(1)).select(
        context_hash=context,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=ScriptedRandom(sample=selected),
    )


def _trial(index, decision, reward, *, reward_hash=REWARD, candidates=1):
    return InsightTrial(
        OperatorInvocationId(f"operator_credit_{index}"),
        tuple(
            CandidateId(f"candidate_credit_{index}_{candidate_index}")
            for candidate_index in range(candidates)
        ),
        reward_hash,
        decision,
        float(reward),
    )


def test_exploitation_and_exploration_record_exact_assignment_probabilities():
    selector = EpsilonGreedySubsetSelector(Fraction(1, 4))
    exploit_rng = ScriptedRandom(draw=3)
    exploit = selector.select(
        context_hash=CONTEXT_A,
        eligible=(D, B, A, C),
        scores=SCORES,
        subset_size=2,
        rng=exploit_rng,
    )
    assert exploit.mode is InsightSelectionMode.EXPLOIT
    assert exploit.eligible == ELIGIBLE
    assert exploit.selected == (A, B)
    assert exploit.exploitation_subset == (A, B)
    assert exploit.selected_subset_probability == Fraction(19, 24)
    assert exploit.inclusion_probability(A) == Fraction(7, 8)
    assert exploit.inclusion_probability(C) == Fraction(1, 8)
    assert exploit.joint_cell_probability(A, C, True, True) == Fraction(1, 24)
    assert exploit.joint_cell_probability(A, C, True, False) == Fraction(5, 6)
    assert exploit.joint_cell_probability(A, C, False, True) == Fraction(1, 12)
    assert exploit.joint_cell_probability(A, C, False, False) == Fraction(1, 24)
    assert exploit.credit_identifiable
    assert exploit_rng.randrange_calls == 1
    assert exploit_rng.sample_calls == 0

    explore_rng = ScriptedRandom(draw=0, sample=(D, C))
    explore = selector.select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=explore_rng,
    )
    assert explore.mode is InsightSelectionMode.EXPLORE_UNIFORM
    assert explore.selected == (C, D)
    assert explore.selected_subset_probability == Fraction(1, 24)
    assert explore_rng.randrange_calls == 1
    assert explore_rng.sample_calls == 1


@pytest.mark.parametrize(
    "epsilon,draw,expected_mode",
    [
        (Fraction(1, 3), 0, InsightSelectionMode.EXPLORE_UNIFORM),
        (Fraction(1, 3), 1, InsightSelectionMode.EXPLOIT),
        (Fraction(1, 2**54), 0, InsightSelectionMode.EXPLORE_UNIFORM),
        (Fraction(1, 2**54), 1, InsightSelectionMode.EXPLOIT),
    ],
)
def test_exploration_branch_uses_exact_rational_integer_law(
    epsilon, draw, expected_mode
):
    rng = ScriptedRandom(draw=draw, sample=(C, D))
    decision = EpsilonGreedySubsetSelector(epsilon).select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=rng,
    )
    assert decision.mode is expected_mode
    assert decision.exploration_probability == epsilon
    assert rng.randrange_stops == [epsilon.denominator]
    assert rng.sample_calls == int(expected_mode is InsightSelectionMode.EXPLORE_UNIFORM)


@pytest.mark.parametrize("draw", [True, 0.0, -1, 3])
def test_exploration_rejects_invalid_exact_integer_draws(draw):
    with pytest.raises((TypeError, ValueError), match="randrange"):
        EpsilonGreedySubsetSelector(Fraction(1, 3)).select(
            context_hash=CONTEXT_A,
            eligible=ELIGIBLE,
            scores=SCORES,
            subset_size=2,
            rng=ScriptedRandom(draw=draw, sample=(C, D)),
        )


def test_score_ties_have_stable_id_version_tie_breaking_and_no_rng_when_deterministic():
    tied = {reference: 0 for reference in ELIGIBLE}
    rng = ScriptedRandom()
    decision = EpsilonGreedySubsetSelector(Fraction(0)).select(
        context_hash=CONTEXT_A,
        eligible=(D, C, B, A),
        scores=tied,
        subset_size=2,
        rng=rng,
    )
    assert decision.selected == (A, B)
    assert decision.selected_subset_probability == 1
    assert decision.inclusion_probability(A) == 1
    assert decision.inclusion_probability(D) == 0
    assert not decision.credit_identifiable
    assert rng.randrange_calls == 0
    assert rng.sample_calls == 0


def test_selecting_every_eligible_insight_is_explicitly_nonidentifiable():
    decision = EpsilonGreedySubsetSelector(Fraction(1, 2)).select(
        context_hash=CONTEXT_A,
        eligible=(A, B),
        scores={A: 1.0, B: 0.0},
        subset_size=2,
        rng=ScriptedRandom(draw=0, sample=(A, B)),
    )
    assert decision.selected == (A, B)
    assert decision.selected_subset_probability == 1
    assert decision.inclusion_probability(A) == 1
    assert not decision.credit_identifiable


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"eligible": (A, A)}, "duplicates"),
        ({"scores": {A: 1.0, B: 2.0}}, "exactly"),
        ({"scores": {A: 1.0, B: 2.0, C: float("nan"), D: 4.0}}, "finite"),
        ({"subset_size": 5}, "cannot exceed"),
        ({"context_hash": "not-a-hash"}, "SHA-256"),
    ],
)
def test_selector_rejects_ambiguous_or_invalid_inputs(kwargs, match):
    arguments = {
        "context_hash": CONTEXT_A,
        "eligible": ELIGIBLE,
        "scores": SCORES,
        "subset_size": 2,
        "rng": ScriptedRandom(draw=3),
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        EpsilonGreedySubsetSelector(Fraction(1, 4)).select(**arguments)


def test_decision_rejects_forged_probability_or_exploitation_subset():
    valid = _uniform_decision((C, D))
    values = {
        field: getattr(valid, field)
        for field in valid.__dataclass_fields__
    }
    values["selected_subset_probability"] = Fraction(1, 5)
    with pytest.raises(ValueError, match="does not match"):
        InsightSelectionDecision(**values)

    values["selected_subset_probability"] = valid.selected_subset_probability
    values["exploitation_subset"] = (C, D)
    with pytest.raises(ValueError, match="recorded scores"):
        InsightSelectionDecision(**values)

    deterministic = EpsilonGreedySubsetSelector(Fraction(0)).select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=ScriptedRandom(),
    )
    deterministic_values = {
        field: getattr(deterministic, field)
        for field in deterministic.__dataclass_fields__
    }
    deterministic_values["mode"] = InsightSelectionMode.EXPLORE_UNIFORM
    with pytest.raises(ValueError, match="zero exploration"):
        InsightSelectionDecision(**deterministic_values)


def test_marginal_credit_recovers_randomized_selected_minus_unselected_effect():
    trials = []
    for index, selected in enumerate(itertools.combinations(ELIGIBLE, 2), start=1):
        reward = 1.0 if A in selected else 0.0
        trials.append(_trial(f"marginal_{index}", _uniform_decision(selected), reward))
    estimate = estimate_marginal_effect(trials, A)
    assert estimate.identified
    assert estimate.effect == pytest.approx(1.0)
    assert estimate.treated_mean == pytest.approx(1.0)
    assert estimate.control_mean == pytest.approx(0.0)
    assert estimate.treated_trials == estimate.control_trials == 3
    assert estimate.treated_effective_sample_size == pytest.approx(3.0)
    assert estimate.control_effective_sample_size == pytest.approx(3.0)
    assert estimate.eligible_trials == estimate.overlap_trials == 6
    assert estimate.context_hash == CONTEXT_A
    assert estimate.subset_size == 2
    assert estimate.exploration_probability == 1
    assert estimate.reward_definition_hash == REWARD
    assert estimate.policy_id == "epsilon_greedy_uniform_k_subset"
    assert estimate.policy_version == 1


def test_context_filter_and_deterministic_trials_do_not_create_false_credit():
    randomized_selected = _trial("context_1", _uniform_decision((A, B)), 1.0)
    randomized_control = _trial("context_2", _uniform_decision((C, D)), 0.0)
    other_context = _trial(
        "context_3", _uniform_decision((C, D), context=CONTEXT_B), 99.0
    )
    deterministic = EpsilonGreedySubsetSelector(Fraction(0)).select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=ScriptedRandom(),
    )
    deterministic_trial = _trial("context_4", deterministic, -100.0)
    estimate = estimate_marginal_effect(
        [randomized_selected, randomized_control, other_context, deterministic_trial],
        A,
        context_hash=CONTEXT_A,
    )
    assert estimate.effect == pytest.approx(1.0)
    assert estimate.eligible_trials == 3
    assert estimate.overlap_trials == 2
    assert estimate.treated_trials == estimate.control_trials == 1
    assert estimate.context_hash == CONTEXT_A
    assert estimate.subset_size == 2
    assert estimate.exploration_probability == 1
    assert estimate.reward_definition_hash == REWARD
    assert estimate.policy_id == "epsilon_greedy_uniform_k_subset"
    assert estimate.policy_version == 1


def test_one_observation_cannot_penalize_unseen_insights():
    one_trial = _trial("one_trial", _uniform_decision((B, D)), 1.0, candidates=4)
    for reference in ELIGIBLE:
        estimate = estimate_marginal_effect([one_trial], reference)
        assert not estimate.identified
        assert estimate.effect is None


def test_pair_credit_recovers_known_interaction_and_fails_closed_without_all_cells():
    trials = []
    # Repeat the complete uniform design so every cell has variance support.
    index = 0
    for repeat in range(2):
        for selected in itertools.combinations(ELIGIBLE, 2):
            index += 1
            reward = 1.0 if A in selected and B in selected else 0.0
            trials.append(_trial(f"pair_{index}", _uniform_decision(selected), reward))
    estimate = estimate_pair_synergy(trials, A, B)
    assert estimate.identified
    assert estimate.synergy == pytest.approx(1.0)
    assert dict(estimate.cell_means) == pytest.approx(
        {"11": 1.0, "10": 0.0, "01": 0.0, "00": 0.0}
    )
    assert dict(estimate.cell_trials) == {"11": 2, "10": 4, "01": 4, "00": 2}
    assert estimate.context_hash == CONTEXT_A
    assert estimate.subset_size == 2
    assert estimate.exploration_probability == 1
    assert estimate.reward_definition_hash == REWARD
    assert estimate.policy_id == "epsilon_greedy_uniform_k_subset"
    assert estimate.policy_version == 1

    k_one = EpsilonGreedySubsetSelector(Fraction(1)).select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=1,
        rng=ScriptedRandom(sample=(A,)),
    )
    no_joint_support = estimate_pair_synergy(
        [_trial("pair_k1", k_one, 1.0)], A, B
    )
    assert not no_joint_support.identified
    assert no_joint_support.overlap_trials == 0


def test_credit_rejects_duplicate_candidate_units_and_invalid_rewards():
    decision = _uniform_decision((A, B))
    shared_operator = OperatorInvocationId("operator_credit_duplicate")
    duplicate = [
        InsightTrial(
            shared_operator,
            (CandidateId("candidate_duplicate_1"),),
            REWARD,
            decision,
            0.0,
        ),
        InsightTrial(
            shared_operator,
            (CandidateId("candidate_duplicate_2"),),
            REWARD,
            decision,
            1.0,
        ),
    ]
    with pytest.raises(ValueError, match="operator invocation"):
        estimate_marginal_effect(duplicate, A)
    with pytest.raises(TypeError, match="canonical float"):
        InsightTrial(
            OperatorInvocationId("operator_credit_bad_reward"),
            (CandidateId("candidate_bad_reward"),),
            REWARD,
            decision,
            math.nan,
        )


def test_credit_unit_aggregates_a_batch_and_rejects_pseudoreplication_or_mixed_rewards():
    decision = _uniform_decision((A, B))
    batch = _trial("batch", decision, 0.5, candidates=3)
    assert len(batch.candidate_ids) == 3

    repeated_candidate = InsightTrial(
        OperatorInvocationId("operator_credit_other"),
        (batch.candidate_ids[0],),
        REWARD,
        decision,
        0.6,
    )
    with pytest.raises(ValueError, match="candidate may appear"):
        estimate_marginal_effect([batch, repeated_candidate], A)

    mixed_reward = _trial("mixed_reward", decision, 0.4, reward_hash="d" * 64)
    with pytest.raises(ValueError, match="mix reward definitions"):
        estimate_marginal_effect([batch, mixed_reward], A)


def test_zero_child_invocation_remains_a_randomized_credit_unit():
    no_child = _trial("zero_child", _uniform_decision((A, B)), -1.0, candidates=0)
    control = _trial("zero_child_control", _uniform_decision((C, D)), 0.0)
    assert no_child.candidate_ids == ()
    estimate = estimate_marginal_effect([no_child, control], A)
    assert estimate.effect == pytest.approx(-1.0)
    assert estimate.treated_trials == estimate.control_trials == 1


def test_credit_rejects_silent_context_or_subset_size_pooling():
    context_a = _trial("stratum_context_a", _uniform_decision((A, B)), 1.0)
    context_b = _trial(
        "stratum_context_b",
        _uniform_decision((C, D), context=CONTEXT_B),
        0.0,
    )
    with pytest.raises(ValueError, match="context strata"):
        estimate_marginal_effect([context_a, context_b], A)

    k_one_decision = EpsilonGreedySubsetSelector(Fraction(1)).select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=1,
        rng=ScriptedRandom(sample=(A,)),
    )
    k_one = _trial("stratum_k_one", k_one_decision, 1.0)
    with pytest.raises(ValueError, match="subset-size estimands"):
        estimate_marginal_effect([context_a, k_one], A)

    different_epsilon_decision = EpsilonGreedySubsetSelector(Fraction(1, 2)).select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=ScriptedRandom(draw=0, sample=(C, D)),
    )
    different_epsilon = _trial(
        "stratum_epsilon", different_epsilon_decision, 0.0
    )
    with pytest.raises(ValueError, match="exploration-policy strata"):
        estimate_marginal_effect([context_a, different_epsilon], A)


def test_extreme_exact_propensity_is_numerically_stable():
    epsilon = Fraction(1, 10**400)
    selector = EpsilonGreedySubsetSelector(epsilon)
    explored = selector.select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=ScriptedRandom(draw=0, sample=(C, D)),
    )
    exploited = selector.select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=ScriptedRandom(draw=1),
    )
    estimate = estimate_marginal_effect(
        [
            _trial("tiny_propensity_selected", explored, 1.0),
            _trial("tiny_propensity_control", exploited, 0.0),
        ],
        C,
    )
    assert estimate.effect == pytest.approx(1.0)
    assert math.isfinite(estimate.treated_effective_sample_size)
    assert math.isfinite(estimate.control_effective_sample_size)


def test_extreme_finite_rewards_do_not_overflow_weighted_means():
    selected = _trial("huge_selected", _uniform_decision((A, B)), 1e308)
    control = _trial("huge_control", _uniform_decision((C, D)), 1e308)
    estimate = estimate_marginal_effect([selected, control], A)
    assert estimate.treated_mean == 1e308
    assert estimate.control_mean == 1e308
    assert estimate.effect == 0.0

    opposite = _trial("huge_opposite", _uniform_decision((C, D)), -1e308)
    with pytest.raises(ValueError, match="finite float"):
        estimate_marginal_effect([selected, opposite], A)


def test_joint_weight_reward_dynamic_range_does_not_double_underflow():
    epsilon = Fraction(1, 10**400)
    selector = EpsilonGreedySubsetSelector(epsilon)
    non_top_scores = {A: 0.0, B: 4.0, C: 3.0, D: 2.0}
    rare_selected = selector.select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=non_top_scores,
        subset_size=2,
        rng=ScriptedRandom(draw=0, sample=(A, D)),
    )
    common_selected = selector.select(
        context_hash=CONTEXT_A,
        eligible=ELIGIBLE,
        scores=SCORES,
        subset_size=2,
        rng=ScriptedRandom(draw=1),
    )
    estimate = estimate_marginal_effect(
        [
            _trial("joint_range_rare", rare_selected, 1e-300),
            _trial("joint_range_common", common_selected, 1e308),
        ],
        A,
    )
    assert estimate.treated_mean == pytest.approx(5e-93, rel=1e-12)
    assert math.isfinite(estimate.treated_mean)

"""Provider-free contracts for wave-sealed v6 causal memory."""

from __future__ import annotations

import itertools
import math
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from agent_evolve.domain.ids import CandidateId, InsightId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    DeterministicMemoryControlPolicy,
    FrozenDiagnosticMemoryWave,
    IncompleteMemoryWaveError,
    MemoryAssignmentArm,
    MemoryAssignmentReceipt,
    MemoryCheckpointClosureStatus,
    MemoryTrialTerminalStatus,
    ResolvedInsightAssignment,
    StaleMemorySnapshotError,
    WaveSealedCheckpointBuilder,
    insight_selection_decision_sha256,
)


CONTEXT = "a" * 64
STRATUM = "b" * 64
PROMPT_SHAPE = "c" * 64
REWARD_DEFINITION = "d" * 64
A = InsightRef(InsightId("insight_a"), 1)
B = InsightRef(InsightId("insight_b"), 1)
C = InsightRef(InsightId("insight_c"), 1)
D = InsightRef(InsightId("insight_d"), 1)


def _policy(**kwargs) -> CausalSearchScorePolicy:
    return CausalSearchScorePolicy(**kwargs)


def _genesis(
    references=(A, B),
    *,
    priors=None,
    policy=None,
):
    score_policy = policy or _policy()
    prior_scores = priors or {reference: 0.0 for reference in references}
    return score_policy.genesis(
        exact_context_hash=CONTEXT,
        estimand_stratum_hash=STRATUM,
        priors=prior_scores,
    )


def _diagnostic_assignment(
    snapshot,
    *,
    index: int,
    subset_rank: int,
    subset_size: int = 1,
):
    decision = DeterministicMemoryControlPolicy().uniform(
        snapshot=snapshot,
        subset_size=subset_size,
        subset_rank=subset_rank,
    )
    return ResolvedInsightAssignment.resolve(
        credit_unit_id=OperatorInvocationId(f"operator_trial_{index}"),
        snapshot=snapshot,
        expected_snapshot_sha256=snapshot.snapshot_sha256,
        block_id=f"diagnostic_block_{index}",
        arm=MemoryAssignmentArm.DIAGNOSTIC,
        selection_decision=decision,
        prompt_shape_sha256=PROMPT_SHAPE,
    )


def _wave(snapshot, assignments, *, no_yield=-2.0):
    return FrozenDiagnosticMemoryWave(
        wave_id="diagnostic_wave_1",
        prior_snapshot=snapshot,
        assignments=tuple(
            sorted(assignments, key=lambda value: value.assignment_sha256)
        ),
        reward_definition_hash=REWARD_DEFINITION,
        no_yield_reward=float(no_yield),
    )


def _success(assignment, reward, *, index: int):
    return MemoryAssignmentReceipt(
        assignment_sha256=assignment.assignment_sha256,
        credit_unit_id=assignment.credit_unit_id,
        status=MemoryTrialTerminalStatus.SUCCEEDED,
        candidate_ids=(CandidateId(f"candidate_trial_{index}"),),
        observed_reward=float(reward),
    )


def _failure(assignment, status, *, index: int):
    candidate_ids = (
        (CandidateId(f"candidate_failed_{index}"),)
        if status is MemoryTrialTerminalStatus.CANDIDATE_FAILURE
        else ()
    )
    return MemoryAssignmentReceipt(
        assignment_sha256=assignment.assignment_sha256,
        credit_unit_id=assignment.credit_unit_id,
        status=status,
        candidate_ids=candidate_ids,
    )


def _seal(policy, snapshot, assignments, receipts, *, no_yield=-2.0):
    return WaveSealedCheckpointBuilder(policy).close(
        _wave(snapshot, assignments, no_yield=no_yield),
        receipts,
    )


def test_resolved_assignment_has_stable_complete_plan_record_and_digest() -> None:
    policy = _policy()
    first_snapshot = _genesis(
        references=(A, B, C),
        priors={C: -1.0, A: 3.0, B: 1.0},
        policy=policy,
    )
    second_snapshot = _genesis(
        references=(C, A, B),
        priors={B: 1.0, C: -1.0, A: 3.0},
        policy=policy,
    )
    first = _diagnostic_assignment(first_snapshot, index=1, subset_rank=2)
    second = _diagnostic_assignment(second_snapshot, index=1, subset_rank=2)

    assert first_snapshot.snapshot_sha256 == second_snapshot.snapshot_sha256
    assert first.assignment_sha256 == second.assignment_sha256
    assert first.to_record() == second.to_record()
    assert first.selection_decision_sha256 == insight_selection_decision_sha256(
        first.selection_decision
    )
    assert first.to_record()["selection_decision"]["selected"] == [
        {"insight_id": "insight_c", "version": 1}
    ]
    assert first.to_record()["score_snapshot_sha256"] == (
        first_snapshot.snapshot_sha256
    )
    with pytest.raises(FrozenInstanceError):
        first.block_id = "mutated"


def test_assignment_rejects_context_or_decision_digest_tampering() -> None:
    snapshot = _genesis()
    decision = DeterministicMemoryControlPolicy().uniform(
        snapshot=snapshot,
        subset_size=1,
        subset_rank=0,
    )
    arguments = {
        "credit_unit_id": OperatorInvocationId("operator_tamper"),
        "exact_context_hash": CONTEXT,
        "estimand_stratum_hash": STRATUM,
        "block_id": "tamper_block",
        "arm": MemoryAssignmentArm.DIAGNOSTIC,
        "selection_decision": decision,
        "selection_decision_sha256": insight_selection_decision_sha256(decision),
        "score_snapshot_sha256": snapshot.snapshot_sha256,
        "prompt_shape_sha256": PROMPT_SHAPE,
    }

    with pytest.raises(ValueError, match="does not match"):
        ResolvedInsightAssignment(
            **{**arguments, "selection_decision_sha256": "0" * 64}
        )
    with pytest.raises(ValueError, match="context"):
        ResolvedInsightAssignment(**{**arguments, "exact_context_hash": "e" * 64})


def test_assignment_resolution_fails_closed_on_stale_or_mismatched_scores() -> None:
    snapshot = _genesis(references=(A, B, C), priors={A: 3.0, B: 2.0, C: 1.0})
    controls = DeterministicMemoryControlPolicy()
    decision = controls.adaptive(snapshot=snapshot, subset_size=1)

    with pytest.raises(StaleMemorySnapshotError, match="differs"):
        ResolvedInsightAssignment.resolve(
            credit_unit_id=OperatorInvocationId("operator_stale"),
            snapshot=snapshot,
            expected_snapshot_sha256="f" * 64,
            block_id="stale_block",
            arm=MemoryAssignmentArm.ADAPTIVE,
            selection_decision=decision,
            prompt_shape_sha256=PROMPT_SHAPE,
        )

    shuffled = controls.score_shuffled(
        snapshot=snapshot,
        subset_size=1,
        permutation_rank=5,
    )
    with pytest.raises(ValueError, match="scores differ"):
        ResolvedInsightAssignment.resolve(
            credit_unit_id=OperatorInvocationId("operator_wrong_scores"),
            snapshot=snapshot,
            expected_snapshot_sha256=snapshot.snapshot_sha256,
            block_id="wrong_scores_block",
            arm=MemoryAssignmentArm.ADAPTIVE,
            selection_decision=shuffled,
            prompt_shape_sha256=PROMPT_SHAPE,
        )


def test_checkpoint_is_invisible_until_complete_wave_seal_and_order_invariant() -> None:
    policy = _policy()
    snapshot = _genesis(policy=policy)
    assignments = (
        _diagnostic_assignment(snapshot, index=0, subset_rank=0),
        _diagnostic_assignment(snapshot, index=1, subset_rank=1),
    )
    receipts = (
        _success(assignments[0], 4.0, index=0),
        _success(assignments[1], 0.0, index=1),
    )
    wave = _wave(snapshot, assignments)
    builder = WaveSealedCheckpointBuilder(policy)

    with pytest.raises(IncompleteMemoryWaveError, match="receipt set differs"):
        builder.close(wave, receipts[:1])
    assert snapshot.checkpoint_index == 0
    assert snapshot.observations == ()
    forward = builder.close(wave, receipts)
    reverse = builder.close(wave, tuple(reversed(receipts)))

    assert forward.status is MemoryCheckpointClosureStatus.SEALED
    assert forward.snapshot is not None
    assert forward.snapshot.checkpoint_index == 1
    assert forward.snapshot.parent_snapshot_sha256 == snapshot.snapshot_sha256
    assert forward.snapshot.snapshot_sha256 == reverse.snapshot.snapshot_sha256
    assert forward.receipts == reverse.receipts
    assert forward.observations == reverse.observations
    assert snapshot.observations == ()  # The parent checkpoint remains immutable.


def test_model_and_candidate_failures_are_no_yield_itt_observations() -> None:
    policy = _policy(prior_effective_sample_size=1.0)
    snapshot = _genesis(policy=policy)
    assignments = tuple(
        _diagnostic_assignment(snapshot, index=index, subset_rank=index % 2)
        for index in range(4)
    )
    receipts = (
        _success(assignments[0], 10.0, index=0),
        _failure(assignments[1], MemoryTrialTerminalStatus.MODEL_FAILURE, index=1),
        _failure(assignments[2], MemoryTrialTerminalStatus.CANDIDATE_FAILURE, index=2),
        _success(assignments[3], 0.0, index=3),
    )
    closure = _seal(policy, snapshot, assignments, receipts, no_yield=-2.0)

    assert closure.snapshot is not None
    observations = {
        value.assignment.credit_unit_id: value for value in closure.observations
    }
    assert observations[assignments[1].credit_unit_id].credited_reward == -2.0
    assert observations[assignments[1].credit_unit_id].candidate_ids == ()
    assert observations[assignments[1].credit_unit_id].reward_was_imputed
    assert observations[assignments[2].credit_unit_id].credited_reward == -2.0
    assert observations[assignments[2].credit_unit_id].candidate_ids
    assert observations[assignments[2].credit_unit_id].reward_was_imputed
    assert not observations[assignments[0].credit_unit_id].reward_was_imputed
    assert all(entry.treated_trials == 2 for entry in closure.snapshot.entries)
    assert all(entry.control_trials == 2 for entry in closure.snapshot.entries)


def test_any_infrastructure_failure_invalidates_wave_without_memory_evidence() -> None:
    policy = _policy()
    snapshot = _genesis(policy=policy)
    assignments = (
        _diagnostic_assignment(snapshot, index=0, subset_rank=0),
        _diagnostic_assignment(snapshot, index=1, subset_rank=1),
    )
    receipts = (
        _success(assignments[0], 99.0, index=0),
        _failure(
            assignments[1],
            MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE,
            index=1,
        ),
    )
    closure = _seal(policy, snapshot, assignments, receipts)

    assert closure.status is (MemoryCheckpointClosureStatus.INVALIDATED_INFRASTRUCTURE)
    assert closure.snapshot is None
    assert closure.observations == ()
    assert snapshot.observations == ()


def test_synthetic_b_and_d_signal_becomes_the_top_causal_memory() -> None:
    policy = _policy(
        prior_effective_sample_size=1.0,
        uncertainty_scale=0.0,
        exploration_weight=0.0,
    )
    snapshot = _genesis(references=(A, B, C, D), policy=policy)
    subsets = tuple(itertools.combinations((A, B, C, D), 2))
    assignments = tuple(
        _diagnostic_assignment(
            snapshot,
            index=index,
            subset_rank=index,
            subset_size=2,
        )
        for index in range(len(subsets))
    )
    receipts = tuple(
        _success(
            assignment,
            2.0 * len({B, D}.intersection(subsets[index])),
            index=index,
        )
        for index, assignment in enumerate(assignments)
    )
    closure = _seal(policy, snapshot, assignments, receipts)

    assert closure.snapshot is not None
    scores = {entry.reference: entry for entry in closure.snapshot.entries}
    assert scores[B].effect_estimate == pytest.approx(4.0 / 3.0)
    assert scores[D].effect_estimate == pytest.approx(4.0 / 3.0)
    assert scores[A].effect_estimate == pytest.approx(-4.0 / 3.0)
    assert scores[C].effect_estimate == pytest.approx(-4.0 / 3.0)
    assert scores[B].retrieval_score == scores[D].retrieval_score
    assert scores[B].retrieval_score > scores[A].retrieval_score
    assert scores[D].retrieval_score > scores[C].retrieval_score
    assert all(
        entry.effective_support == pytest.approx(3.0) for entry in scores.values()
    )


def test_min_arm_ess_controls_shrinkage_and_uncertainty_formula() -> None:
    policy = _policy(
        prior_effective_sample_size=3.0,
        uncertainty_scale=2.0,
        exploration_weight=0.5,
    )
    snapshot = _genesis(policy=policy)
    assignments = (
        _diagnostic_assignment(snapshot, index=0, subset_rank=0),
        _diagnostic_assignment(snapshot, index=1, subset_rank=1),
    )
    closure = _seal(
        policy,
        snapshot,
        assignments,
        (
            _success(assignments[0], 4.0, index=0),
            _success(assignments[1], 0.0, index=1),
        ),
    )

    assert closure.snapshot is not None
    scores = {entry.reference: entry for entry in closure.snapshot.entries}
    assert scores[A].treated_effective_sample_size == pytest.approx(1.0)
    assert scores[A].control_effective_sample_size == pytest.approx(1.0)
    assert scores[A].effective_support == pytest.approx(1.0)
    assert scores[A].shrinkage == pytest.approx(0.25)
    assert scores[A].effect_estimate == pytest.approx(4.0)
    assert scores[A].posterior_mean == pytest.approx(1.0)
    assert scores[A].uncertainty_bonus == pytest.approx(1.0)
    assert scores[A].retrieval_score == pytest.approx(1.5)
    assert scores[B].posterior_mean == pytest.approx(-1.0)
    assert scores[B].retrieval_score == pytest.approx(-0.5)


def test_one_sided_support_does_not_manufacture_an_effect_or_penalty() -> None:
    policy = _policy(
        prior_effective_sample_size=4.0,
        uncertainty_scale=2.0,
        exploration_weight=0.5,
    )
    snapshot = _genesis(references=(A, B), priors={A: 0.2, B: -0.3}, policy=policy)
    assignment = _diagnostic_assignment(snapshot, index=0, subset_rank=0)
    closure = _seal(
        policy,
        snapshot,
        (assignment,),
        (_success(assignment, 100.0, index=0),),
    )

    assert closure.snapshot is not None
    for entry in closure.snapshot.entries:
        assert entry.effect_estimate is None
        assert entry.effective_support == 0.0
        assert entry.shrinkage == 0.0
        assert entry.posterior_mean == entry.prior_score
        assert entry.uncertainty_bonus == pytest.approx(1.0)
        assert entry.retrieval_score == pytest.approx(entry.prior_score + 0.5)


def test_score_shuffle_is_exact_deterministic_rank_law_and_preserves_multiset() -> None:
    snapshot = _genesis(
        references=(A, B, C),
        priors={A: 3.0, B: 2.0, C: 1.0},
    )
    controls = DeterministicMemoryControlPolicy()
    source = tuple(entry.retrieval_score for entry in snapshot.entries)
    realizations = []
    for rank in range(math.factorial(3)):
        first = controls.score_shuffled(
            snapshot=snapshot,
            subset_size=1,
            permutation_rank=rank,
        )
        second = controls.score_shuffled(
            snapshot=snapshot,
            subset_size=1,
            permutation_rank=rank,
        )
        observed = tuple(score for _, score in first.score_snapshot)
        assert first == second
        assert sorted(value.hex() for value in observed) == sorted(
            value.hex() for value in source
        )
        assignment = ResolvedInsightAssignment.resolve(
            credit_unit_id=OperatorInvocationId(f"operator_shuffle_{rank}"),
            snapshot=snapshot,
            expected_snapshot_sha256=snapshot.snapshot_sha256,
            block_id=f"shuffle_block_{rank}",
            arm=MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL,
            selection_decision=first,
            prompt_shape_sha256=PROMPT_SHAPE,
        )
        assignment.validate_against_snapshot(snapshot)
        realizations.append(tuple(value.hex() for value in observed))

    assert len(set(realizations)) == math.factorial(3)
    with pytest.raises(ValueError, match="permutation_rank"):
        controls.score_shuffled(
            snapshot=snapshot,
            subset_size=1,
            permutation_rank=math.factorial(3),
        )


def test_uniform_control_rank_enumerates_exact_k_subsets() -> None:
    snapshot = _genesis(
        references=(A, B, C),
        priors={A: 3.0, B: 2.0, C: 1.0},
    )
    controls = DeterministicMemoryControlPolicy()
    decisions = tuple(
        controls.uniform(snapshot=snapshot, subset_size=2, subset_rank=rank)
        for rank in range(math.comb(3, 2))
    )

    assert {decision.selected for decision in decisions} == set(
        itertools.combinations((A, B, C), 2)
    )
    assert all(
        decision.selected_subset_probability == Fraction(1, 3) for decision in decisions
    )
    assignment = ResolvedInsightAssignment.resolve(
        credit_unit_id=OperatorInvocationId("operator_uniform_control"),
        snapshot=snapshot,
        expected_snapshot_sha256=snapshot.snapshot_sha256,
        block_id="uniform_control_block",
        arm=MemoryAssignmentArm.UNIFORM_CONTROL,
        selection_decision=decisions[1],
        prompt_shape_sha256=PROMPT_SHAPE,
    )
    assert assignment.selection_decision.selected == (A, C)
    with pytest.raises(ValueError, match="subset_rank"):
        controls.uniform(snapshot=snapshot, subset_size=2, subset_rank=3)


def test_wave_rejects_stale_assignments_mixed_laws_and_duplicate_receipts() -> None:
    policy = _policy()
    snapshot = _genesis(policy=policy)
    assignment_a = _diagnostic_assignment(snapshot, index=0, subset_rank=0)
    assignment_b = _diagnostic_assignment(snapshot, index=1, subset_rank=1)
    wave = _wave(snapshot, (assignment_a, assignment_b))
    receipt = _success(assignment_a, 1.0, index=0)

    with pytest.raises(IncompleteMemoryWaveError, match="more than one"):
        WaveSealedCheckpointBuilder(policy).close(wave, (receipt, receipt))

    different_snapshot = _genesis(
        references=(A, B), priors={A: 1.0, B: 0.0}, policy=policy
    )
    stale_assignment = _diagnostic_assignment(
        different_snapshot, index=2, subset_rank=0
    )
    with pytest.raises(StaleMemorySnapshotError, match="different"):
        _wave(snapshot, (stale_assignment,))

    three_way_snapshot = _genesis(references=(A, B, C), policy=policy)
    one_of_three = _diagnostic_assignment(
        three_way_snapshot, index=3, subset_rank=0, subset_size=1
    )
    two_of_three = _diagnostic_assignment(
        three_way_snapshot, index=4, subset_rank=0, subset_size=2
    )
    with pytest.raises(ValueError, match="assignment-law strata"):
        _wave(three_way_snapshot, (one_of_three, two_of_three))


def test_candidate_ids_cannot_be_reused_as_independent_causal_units() -> None:
    policy = _policy()
    snapshot = _genesis(policy=policy)
    assignments = (
        _diagnostic_assignment(snapshot, index=0, subset_rank=0),
        _diagnostic_assignment(snapshot, index=1, subset_rank=1),
    )
    repeated_candidate = CandidateId("candidate_repeated")
    receipts = tuple(
        MemoryAssignmentReceipt(
            assignment_sha256=assignment.assignment_sha256,
            credit_unit_id=assignment.credit_unit_id,
            status=MemoryTrialTerminalStatus.SUCCEEDED,
            candidate_ids=(repeated_candidate,),
            observed_reward=float(index),
        )
        for index, assignment in enumerate(assignments)
    )

    with pytest.raises(ValueError, match="candidate may appear"):
        _seal(policy, snapshot, assignments, receipts)

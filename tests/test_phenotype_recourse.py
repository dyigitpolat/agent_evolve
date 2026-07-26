"""Provider-free contracts for phenotype coalescing and bounded recourse."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields

import pytest

from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.selection.phenotype_recourse import (
    BoundedEvaluationRecoursePolicy,
    EvaluationOccurrenceRole,
    EvaluationOccurrenceStatus,
    EvaluationRecourseDecision,
    EvaluationRecoursePolicy,
    PhenotypeOccurrence,
    PhenotypeOccurrenceLedger,
    PresealedRecoursePool,
    RecourseBudgetSnapshot,
    RecourseDecisionReason,
    RecoursePoolCandidate,
    SemanticProjectionPhenotypeIdentityPolicy,
    TypedConfigurationPhenotypeIdentityPolicy,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


TYPED = TypedConfigurationPhenotypeIdentityPolicy()


def _occurrence(
    label: str,
    configuration: object | None,
    *,
    policy=TYPED,
    role: EvaluationOccurrenceRole = EvaluationOccurrenceRole.PRIMARY,
    status: EvaluationOccurrenceStatus = EvaluationOccurrenceStatus.SUCCESS,
) -> PhenotypeOccurrence:
    if configuration is None:
        return PhenotypeOccurrence(
            trial_id=OperatorInvocationId(f"operator_{label}"),
            role=role,
            status=status,
            candidate_id=None,
            phenotype=None,
        )
    return PhenotypeOccurrence(
        trial_id=OperatorInvocationId(f"operator_{label}"),
        role=role,
        status=status,
        candidate_id=CandidateId(f"candidate_{label}"),
        phenotype=policy.identify(configuration),
    )


def _ledger(*occurrences: PhenotypeOccurrence, policy=TYPED):
    return PhenotypeOccurrenceLedger.build(occurrences, identity_policy=policy)


def _pool(*configurations: object, policy=TYPED, labels: tuple[str, ...] | None = None):
    if labels is None:
        labels = tuple(f"pool_{index}" for index in range(len(configurations)))
    assert len(labels) == len(configurations)
    return PresealedRecoursePool.seal(
        pool_id="test_pool",
        seal_context_sha256=_hash("pre-outcome-plan-cutoff"),
        candidates=tuple(
            RecoursePoolCandidate.freeze(label, configuration)
            for label, configuration in zip(labels, configurations, strict=True)
        ),
        identity_policy=policy,
    )


def _budget(
    maximum: int = 20,
    used: int = 0,
    reserved: int = 0,
    protected: int = 0,
) -> RecourseBudgetSnapshot:
    return RecourseBudgetSnapshot(maximum, used, reserved, protected)


def test_default_identity_is_type_sensitive_and_mapping_order_invariant() -> None:
    ordered = TYPED.identify({"x": 1, "flag": True})
    reordered = TYPED.identify({"flag": True, "x": 1})
    integer_flag = TYPED.identify({"x": 1, "flag": 1})
    floating_x = TYPED.identify({"x": 1.0, "flag": True})

    assert ordered == reordered
    assert ordered.identity_sha256 == reordered.identity_sha256
    assert ordered != integer_flag
    assert ordered != floating_x


def test_injected_semantic_identity_groups_physical_eval_but_keeps_trials() -> None:
    def evaluator_projection(configuration):
        raw = thaw_json(configuration)
        assert isinstance(raw, dict)
        return {"decoded_x": raw["x"]}

    semantic = SemanticProjectionPhenotypeIdentityPolicy(
        policy_id="test_decoded_x",
        policy_version=1,
        projector=evaluator_projection,
    )
    left = _occurrence("semantic_left", {"x": 3, "comment": "a"}, policy=semantic)
    right = _occurrence("semantic_right", {"comment": "b", "x": 3}, policy=semantic)
    ledger = _ledger(right, left, policy=semantic)

    assert left.trial_id != right.trial_id
    assert left.phenotype == right.phenotype
    assert len(ledger.occurrences) == 2
    assert len(ledger.clusters) == 1
    assert len(ledger.clusters[0].occurrences) == 2
    assert ledger.successful_primary_collision_credit == 1


def test_same_phenotype_pair_has_explicit_zero_identity_contrast() -> None:
    left = _occurrence("contrast_left", {"x": 1})
    right = _occurrence("contrast_right", {"x": 1})
    other = _occurrence("contrast_other", {"x": 2})
    ledger = _ledger(left, right, other)

    assert ledger.is_zero_identity_contrast(left.trial_id, right.trial_id)
    assert not ledger.is_zero_identity_contrast(left.trial_id, other.trial_id)
    assert (
        ledger.clusters[0].zero_identity_contrast_pairs
        or ledger.clusters[1].zero_identity_contrast_pairs
    )


def test_bounded_formula_protects_reservations_and_recombination_budget() -> None:
    ledger = _ledger(
        _occurrence("duplicate_a", {"x": 1}),
        _occurrence("duplicate_b", {"x": 1}),
        _occurrence("duplicate_c", {"x": 1}),
    )
    pool = _pool(
        {"x": 10},
        {"x": 11},
        {"x": 12},
        labels=("first", "second", "third"),
    )
    # 10 - 4 used - 1 other reservation - 3 protected recombinations = 2.
    budget = _budget(maximum=10, used=4, reserved=1, protected=3)
    decision = BoundedEvaluationRecoursePolicy(max_recourse=9).decide(
        ledger=ledger,
        pool=pool,
        budget=budget,
    )

    assert ledger.successful_primary_collision_credit == 2
    assert budget.available_for_recourse == 2
    assert decision.slots == min(9, 2, 2, 3)
    assert decision.selected_entry_ids == ("first", "second")
    assert decision.reason is RecourseDecisionReason.SELECTED
    assert (
        budget.used_unique_evaluations
        + budget.reserved_non_recourse_evaluations
        + budget.protected_recombination_evaluations
        + decision.slots
        <= budget.max_unique_evaluations
    )


def test_recourse_policy_surface_cannot_receive_objectives_rewards_or_scores() -> None:
    protocol_parameters = set(
        inspect.signature(EvaluationRecoursePolicy.decide).parameters
    )
    concrete_parameters = set(
        inspect.signature(BoundedEvaluationRecoursePolicy.decide).parameters
    )
    assert protocol_parameters == {"self", "ledger", "pool", "budget"}
    assert concrete_parameters == protocol_parameters

    visible_fields = {
        field.name
        for data_type in (
            PhenotypeOccurrence,
            PhenotypeOccurrenceLedger,
            RecourseBudgetSnapshot,
            EvaluationRecourseDecision,
        )
        for field in fields(data_type)
    }
    assert not visible_fields.intersection(
        {"objective", "objectives", "reward", "rewards", "score", "scores"}
    )


def test_no_successful_collision_means_no_recourse() -> None:
    ledger = _ledger(
        _occurrence("distinct_a", {"x": 1}),
        _occurrence("distinct_b", {"x": 2}),
    )
    decision = BoundedEvaluationRecoursePolicy(3).decide(
        ledger=ledger,
        pool=_pool({"x": 10}),
        budget=_budget(),
    )

    assert decision.slots == 0
    assert decision.reason is RecourseDecisionReason.NO_SUCCESSFUL_PRIMARY_COLLISION


def test_failure_semantics_keep_causal_yield_and_block_validity_distinct() -> None:
    duplicate_a = _occurrence("failure_duplicate_a", {"x": 1})
    duplicate_b = _occurrence("failure_duplicate_b", {"x": 1})
    model_failure = _occurrence(
        "failure_model",
        None,
        status=EvaluationOccurrenceStatus.MODEL_FAILURE,
    )
    candidate_failure = _occurrence(
        "failure_candidate",
        {"x": 2},
        status=EvaluationOccurrenceStatus.CANDIDATE_FAILURE,
    )
    unmaterialized_candidate_failure = _occurrence(
        "failure_candidate_unmaterialized",
        None,
        status=EvaluationOccurrenceStatus.CANDIDATE_FAILURE,
    )
    valid = _ledger(
        duplicate_a,
        duplicate_b,
        model_failure,
        candidate_failure,
        unmaterialized_candidate_failure,
    )
    valid_decision = BoundedEvaluationRecoursePolicy(3).decide(
        ledger=valid,
        pool=_pool({"x": 10}),
        budget=_budget(),
    )
    assert valid.successful_primary_collision_credit == 1
    assert valid_decision.slots == 1

    infrastructure_failure = _occurrence(
        "failure_infrastructure",
        {"x": 4},
        status=EvaluationOccurrenceStatus.INFRASTRUCTURE_FAILURE,
    )
    invalid = _ledger(
        duplicate_a,
        duplicate_b,
        model_failure,
        candidate_failure,
        unmaterialized_candidate_failure,
        infrastructure_failure,
    )
    invalid_decision = BoundedEvaluationRecoursePolicy(3).decide(
        ledger=invalid,
        pool=_pool({"x": 10}),
        budget=_budget(),
    )
    assert invalid_decision.slots == 0
    assert invalid_decision.reason is RecourseDecisionReason.PRIMARY_BLOCK_INVALID
    assert invalid_decision.to_trace_record()["invalidating_primary_trial_ids"] == [
        "operator_failure_infrastructure"
    ]
    assert not invalid.primary_block_valid
    assert not invalid.experiment_block_valid


def test_occurrence_order_does_not_change_replay_and_pool_order_is_frozen() -> None:
    left = _occurrence("replay_left", {"x": 1})
    right = _occurrence("replay_right", {"x": 1})
    forward = _ledger(left, right)
    reverse = _ledger(right, left)
    pool = _pool(
        {"x": 10},
        {"x": 11},
        labels=("z_first_by_seal", "a_second_by_seal"),
    )
    policy = BoundedEvaluationRecoursePolicy(1)

    first = policy.decide(ledger=forward, pool=pool, budget=_budget())
    replay = policy.decide(ledger=reverse, pool=pool, budget=_budget())

    assert forward == reverse
    assert first == replay
    assert first.decision_sha256 == replay.decision_sha256
    assert first.selected_entry_ids == ("z_first_by_seal",)
    with pytest.raises(FrozenInstanceError):
        pool.entries = ()  # type: ignore[misc]


def test_recourse_successes_never_chain_into_additional_recourse() -> None:
    primary_a = _occurrence("chain_primary_a", {"x": 1})
    primary_b = _occurrence("chain_primary_b", {"x": 1})
    primary_only = _ledger(primary_a, primary_b)
    pool = _pool({"x": 10}, {"x": 11}, {"x": 12})
    policy = BoundedEvaluationRecoursePolicy(3)
    before = policy.decide(ledger=primary_only, pool=pool, budget=_budget())

    recourse_a = _occurrence(
        "chain_recourse_a",
        {"x": 10},
        role=EvaluationOccurrenceRole.RECOURSE,
    )
    recourse_b = _occurrence(
        "chain_recourse_b",
        {"x": 10},
        role=EvaluationOccurrenceRole.RECOURSE,
    )
    after_ledger = _ledger(primary_a, primary_b, recourse_a, recourse_b)
    after = policy.decide(ledger=after_ledger, pool=pool, budget=_budget())

    assert before.slots == after.slots == 1
    assert before.selected_entry_ids == after.selected_entry_ids == ("pool_0",)
    assert after_ledger.successful_primary_collision_credit == 1
    assert [item.value for item in after_ledger.ignored_recourse_trial_ids] == [
        "operator_chain_recourse_a",
        "operator_chain_recourse_b",
    ]


def test_late_infrastructure_failure_is_fatal_without_chaining() -> None:
    primary_a = _occurrence("late_failure_primary_a", {"x": 1})
    primary_b = _occurrence("late_failure_primary_b", {"x": 1})
    recourse_failure = _occurrence(
        "late_failure_recourse",
        {"x": 10},
        role=EvaluationOccurrenceRole.RECOURSE,
        status=EvaluationOccurrenceStatus.INFRASTRUCTURE_FAILURE,
    )
    ledger = _ledger(primary_a, primary_b, recourse_failure)
    decision = BoundedEvaluationRecoursePolicy(3).decide(
        ledger=ledger,
        pool=_pool({"x": 10}, {"x": 11}),
        budget=_budget(),
    )

    assert ledger.primary_block_valid
    assert not ledger.experiment_block_valid
    assert [item.value for item in ledger.invalidating_trial_ids] == [
        "operator_late_failure_recourse"
    ]
    assert ledger.successful_primary_collision_credit == 1
    assert decision.slots == 1


def test_pool_excludes_primary_phenotypes_and_rejects_semantic_duplicates() -> None:
    primary_a = _occurrence("occupied_a", {"x": 1})
    primary_b = _occurrence("occupied_b", {"x": 1})
    ledger = _ledger(primary_a, primary_b)
    pool = _pool({"x": 1}, {"x": 10}, labels=("occupied", "fresh"))
    decision = BoundedEvaluationRecoursePolicy(2).decide(
        ledger=ledger,
        pool=pool,
        budget=_budget(),
    )
    assert tuple(entry.entry_id for entry in decision.eligible_entries) == ("fresh",)
    assert decision.selected_entry_ids == ("fresh",)

    semantic = SemanticProjectionPhenotypeIdentityPolicy(
        "test_ignore_note",
        1,
        lambda configuration: {"x": thaw_json(configuration)["x"]},
    )
    with pytest.raises(ValueError, match="phenotypes must be unique"):
        _pool(
            {"x": 4, "note": "left"},
            {"x": 4, "note": "right"},
            policy=semantic,
        )


def test_budget_cap_property_holds_over_small_integer_grid() -> None:
    ledger = _ledger(
        *(_occurrence(f"property_{index}", {"x": 1}) for index in range(6))
    )
    pool = _pool(*({"x": 100 + index} for index in range(6)))

    for maximum in range(1, 8):
        for used in range(maximum + 1):
            for reserved in range(maximum - used + 1):
                for protected in range(maximum - used - reserved + 1):
                    budget = _budget(maximum, used, reserved, protected)
                    for max_recourse in range(5):
                        decision = BoundedEvaluationRecoursePolicy(max_recourse).decide(
                            ledger=ledger, pool=pool, budget=budget
                        )
                        expected = min(
                            max_recourse,
                            ledger.successful_primary_collision_credit,
                            budget.available_for_recourse,
                            len(pool.entries),
                        )
                        assert decision.slots == expected
                        assert used + reserved + protected + decision.slots <= maximum


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"used_unique_evaluations": 5}, "exceeds"),
        ({"reserved_non_recourse_evaluations": -1}, "non-negative"),
        (
            {
                "used_unique_evaluations": 2,
                "reserved_non_recourse_evaluations": 2,
                "protected_recombination_evaluations": 2,
            },
            "exceed",
        ),
    ],
)
def test_budget_snapshot_rejects_impossible_reservations(kwargs, match) -> None:
    arguments = {
        "max_unique_evaluations": 4,
        "used_unique_evaluations": 0,
        "reserved_non_recourse_evaluations": 0,
        "protected_recombination_evaluations": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        RecourseBudgetSnapshot(**arguments)

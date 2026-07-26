"""Golden reconciliation for the versioned RunCounters event projection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from agent_evolve.application.event_recorder import EventRecorder
from agent_evolve.application.projections import (
    RUN_COUNTERS_PROJECTION_VERSION,
    ProjectionError,
    project_run_counters,
)
from agent_evolve.domain.event import (
    CandidateAdmitted,
    CandidateProposed,
    DuplicateDetected,
    EvaluationCacheBypassed,
    EvaluationCacheHit,
    EvaluationCacheMiss,
    EvaluationCacheStored,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    EvaluationRetried,
    EvaluationStarted,
    GenerationCompleted,
    GenerationStarted,
    LLMCallCompleted,
    LLMCallRequested,
    LLMCallRetried,
    LLMCallStarted,
    OperatorSelected,
    ProviderAttemptCompleted,
    ProviderAttemptFailed,
    ProviderAttemptStarted,
    RunFinished,
    RunStarted,
    ValidationCompleted,
    ValidationStage,
    ValidationStarted,
    event_from_json,
    event_to_json,
)
from agent_evolve.domain.outcome import FailureCategory, FailureCode
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.infrastructure.events import InMemoryEventStore
from agent_evolve.infrastructure.ids import DeterministicIdFactory

HASH_A = "a" * 64
HASH_B = "b" * 64
OBJECTIVES = (("score", 7.0), ("cost", 2.0))


def _rich_trace():
    store = InMemoryEventStore()
    ids = DeterministicIdFactory("counters")
    run_id = ids.new_run_id()
    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=FakeClock(),
    )

    generation_id = ids.new_generation_id()
    operator_id = ids.new_operator_invocation_id()
    call_id = ids.new_llm_call_id()

    recorder.record(RunStarted(HASH_A))
    recorder.record(GenerationStarted(generation_id, 1))
    recorder.record(OperatorSelected(operator_id, generation_id, "semantic_mutation", "v1", 2))
    recorder.record(LLMCallRequested(call_id, "generate", "model", generation_id, operator_id))
    recorder.record(LLMCallStarted(call_id))

    provider_1 = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, provider_1, 1))
    recorder.record(
        ProviderAttemptFailed(
            call_id,
            provider_1,
            FailureCategory.INFRASTRUCTURE,
            FailureCode.TRANSIENT_EXTERNAL_SERVICE_FAILURE,
            True,
            "provider temporarily unavailable",
            latency_ns=5,
        )
    )
    recorder.record(LLMCallRetried(call_id, 2))
    provider_2 = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, provider_2, 2))
    recorder.record(
        ProviderAttemptCompleted(
            call_id,
            provider_2,
            "openrouter",
            "model-resolved",
            input_tokens=10,
            output_tokens=4,
            reasoning_tokens=2,
            cache_read_tokens=1,
            cache_write_tokens=3,
            cost_usd=Decimal("0.01"),
            latency_ns=7,
        )
    )
    recorder.record(LLMCallCompleted(call_id))

    candidate_1 = ids.new_candidate_id()
    candidate_2 = ids.new_candidate_id()
    recorder.record(CandidateProposed(candidate_1, generation_id, HASH_A, 0, operator_id, call_id))
    recorder.record(CandidateProposed(candidate_2, generation_id, HASH_A, 1, operator_id, call_id))
    recorder.record(DuplicateDetected(candidate_2, HASH_A, candidate_1))

    recorder.record(ValidationStarted(candidate_1, ValidationStage.SCHEMA))
    recorder.record(ValidationCompleted(candidate_1, ValidationStage.SCHEMA, True))
    recorder.record(ValidationStarted(candidate_1, ValidationStage.DETERMINISTIC_PRECHECK))
    recorder.record(
        ValidationCompleted(candidate_1, ValidationStage.DETERMINISTIC_PRECHECK, True)
    )

    evaluation_1 = ids.new_evaluation_id()
    attempt_1 = ids.new_evaluation_attempt_id()
    recorder.record(EvaluationRequested(evaluation_1, candidate_1, "full", 11, HASH_B))
    recorder.record(EvaluationCacheMiss(evaluation_1, "full"))
    recorder.record(EvaluationStarted(evaluation_1, attempt_1, "full", 1))
    recorder.record(
        EvaluationFailed(
            evaluation_1,
            "full",
            FailureCategory.INFRASTRUCTURE,
            FailureCode.TIMEOUT_OR_RESOURCE_FAILURE,
            True,
            False,
            "worker timed out",
            attempt_1,
            worker_time_ns=11,
        )
    )
    recorder.record(EvaluationRetried(evaluation_1, "full", 2))
    attempt_2 = ids.new_evaluation_attempt_id()
    recorder.record(EvaluationStarted(evaluation_1, attempt_2, "full", 2))
    recorder.record(EvaluationCompleted(evaluation_1, attempt_2, "full", OBJECTIVES, 13))
    recorder.record(EvaluationCacheStored(evaluation_1, "full", HASH_B))
    recorder.record(CandidateAdmitted(candidate_1, evaluation_1))

    candidate_3 = ids.new_candidate_id()
    evaluation_2 = ids.new_evaluation_id()
    recorder.record(CandidateProposed(candidate_3, generation_id, HASH_B, 2))
    recorder.record(EvaluationRequested(evaluation_2, candidate_3, "full", 11, HASH_B))
    recorder.record(
        EvaluationCacheHit(
            evaluation_2,
            candidate_3,
            "full",
            run_id,
            evaluation_1,
            OBJECTIVES,
        )
    )
    recorder.record(CandidateAdmitted(candidate_3, evaluation_2))

    candidate_4 = ids.new_candidate_id()
    recorder.record(CandidateProposed(candidate_4, generation_id, "c" * 64, 3))
    recorder.record(ValidationStarted(candidate_4, ValidationStage.SCHEMA))
    recorder.record(
        ValidationCompleted(
            candidate_4,
            ValidationStage.SCHEMA,
            False,
            FailureCategory.CANDIDATE,
            FailureCode.SCHEMA_INVALID,
        )
    )

    candidate_5 = ids.new_candidate_id()
    recorder.record(CandidateProposed(candidate_5, generation_id, "d" * 64, 4))
    recorder.record(ValidationStarted(candidate_5, ValidationStage.DETERMINISTIC_PRECHECK))
    recorder.record(
        ValidationCompleted(
            candidate_5,
            ValidationStage.DETERMINISTIC_PRECHECK,
            False,
            FailureCategory.CANDIDATE,
            FailureCode.DETERMINISTIC_PRECHECK_INFEASIBLE,
        )
    )

    candidate_6 = ids.new_candidate_id()
    evaluation_3 = ids.new_evaluation_id()
    attempt_3 = ids.new_evaluation_attempt_id()
    recorder.record(CandidateProposed(candidate_6, generation_id, "e" * 64, 5))
    recorder.record(EvaluationRequested(evaluation_3, candidate_6, "cheap", None, None))
    recorder.record(EvaluationCacheBypassed(evaluation_3, "cheap", "missing fingerprint"))
    recorder.record(EvaluationStarted(evaluation_3, attempt_3, "cheap", 1))
    recorder.record(
        EvaluationFailed(
            evaluation_3,
            "cheap",
            FailureCategory.CANDIDATE,
            FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
            False,
            True,
            "candidate is infeasible",
            attempt_3,
            worker_time_ns=17,
        )
    )

    candidate_7 = ids.new_candidate_id()
    evaluation_4 = ids.new_evaluation_id()
    attempt_4 = ids.new_evaluation_attempt_id()
    recorder.record(CandidateProposed(candidate_7, generation_id, "f" * 64, 6))
    recorder.record(EvaluationRequested(evaluation_4, candidate_7, "full", None, HASH_A))
    recorder.record(EvaluationCacheMiss(evaluation_4, "full"))
    recorder.record(EvaluationStarted(evaluation_4, attempt_4, "full", 1))
    recorder.record(
        EvaluationFailed(
            evaluation_4,
            "full",
            FailureCategory.SYSTEM,
            FailureCode.EVALUATOR_CONTRACT_VIOLATION,
            False,
            True,
            "evaluator omitted an objective",
            attempt_4,
            worker_time_ns=19,
        )
    )
    recorder.record(GenerationCompleted(generation_id, 1, 2))
    recorder.record(RunFinished("budget_exhausted"))
    return recorder.events()


def test_golden_counter_projection_reconciles_independent_resources():
    events = _rich_trace()
    counters = project_run_counters(events)

    assert counters.projection_version == RUN_COUNTERS_PROJECTION_VERSION
    assert counters.event_count == len(events) == 49
    assert (counters.runs_started, counters.runs_finished, counters.runs_aborted) == (1, 1, 0)
    assert (counters.generations_started, counters.generations_completed) == (1, 1)
    assert counters.operators_selected == 1

    assert counters.llm_logical_calls_requested == 1
    assert counters.llm_logical_calls_started == 1
    assert counters.llm_logical_calls_completed == 1
    assert counters.llm_logical_retries == 1
    assert counters.llm_provider_attempts_started == 2
    assert counters.llm_provider_attempts_completed == 1
    assert counters.llm_provider_attempts_failed == 1
    assert counters.llm_input_tokens == 10
    assert counters.llm_output_tokens == 4
    assert counters.llm_reasoning_tokens == 2
    assert counters.llm_cache_read_tokens == 1
    assert counters.llm_cache_write_tokens == 3
    assert counters.llm_cost_usd == Decimal("0.01")
    assert counters.llm_provider_latency_ns == 12

    assert counters.candidates_proposed == 7
    assert counters.duplicate_proposals == 1
    assert (counters.schema_validation_requests, counters.schema_validation_failures) == (2, 1)
    assert (
        counters.deterministic_precheck_requests,
        counters.deterministic_precheck_failures,
    ) == (2, 1)

    assert counters.evaluation_requests == 4
    assert counters.evaluation_attempts_started == 4
    assert counters.evaluation_attempts_completed == 4
    assert counters.evaluation_successes == 1
    assert counters.candidate_evaluation_failures == 1
    assert counters.infrastructure_evaluation_failures == 1
    assert counters.system_evaluation_failures == 1
    assert counters.evaluation_retries == 1

    assert counters.full_evaluator_requests == 3
    assert counters.full_evaluator_attempts_started == 3
    assert counters.full_evaluator_attempts_completed == 3
    assert counters.full_evaluator_successes == 1
    assert counters.full_candidate_failures == 0
    assert counters.full_infrastructure_failures == 1
    assert counters.full_system_failures == 1
    assert counters.full_evaluator_retries == 1

    assert (counters.cache_lookups, counters.cache_hits, counters.cache_misses) == (3, 1, 2)
    assert (counters.cache_bypasses, counters.cache_writes) == (1, 1)
    assert counters.successful_objective_vectors == 2
    assert counters.candidate_admissions == 2
    assert counters.evaluator_worker_time_ns == 60


def test_projection_is_replay_idempotent_and_counters_are_frozen():
    events = _rich_trace()
    first = project_run_counters(events)
    durable_round_trip = tuple(event_from_json(event_to_json(event)) for event in events)
    assert durable_round_trip == events
    second = project_run_counters(durable_round_trip)
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.event_count = 0


def test_projection_rejects_gap_or_mixed_run_stream():
    events = list(_rich_trace())
    with pytest.raises(ProjectionError, match="Expected event sequence 2"):
        project_run_counters([events[0], events[2]])

    other = events[1]
    from dataclasses import replace
    from agent_evolve.domain.ids import RunId

    with pytest.raises(ProjectionError, match="multiple runs"):
        project_run_counters([events[0], replace(other, run_id=RunId("run_other"))])

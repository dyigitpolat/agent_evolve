"""Pure, versioned projections derived from immutable events."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from decimal import Decimal
from typing import Iterable, Optional

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
    EventEnvelope,
    GenerationCompleted,
    GenerationStarted,
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallRequested,
    LLMCallRetried,
    LLMCallStarted,
    OperatorSelected,
    ProviderAttemptCompleted,
    ProviderAttemptFailed,
    ProviderAttemptStarted,
    RunAborted,
    RunFinished,
    RunStarted,
    ValidationCompleted,
    ValidationStage,
    ValidationStarted,
)
from agent_evolve.domain.ids import RunId
from agent_evolve.domain.outcome import FailureCategory

RUN_COUNTERS_PROJECTION_VERSION = 1


class ProjectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunCounters:
    """Named resource/accounting counters; no generic "evaluations" field."""

    projection_version: int = RUN_COUNTERS_PROJECTION_VERSION
    event_count: int = 0

    runs_started: int = 0
    runs_finished: int = 0
    runs_aborted: int = 0
    generations_started: int = 0
    generations_completed: int = 0
    operators_selected: int = 0

    llm_logical_calls_requested: int = 0
    llm_logical_calls_started: int = 0
    llm_logical_calls_completed: int = 0
    llm_logical_calls_failed: int = 0
    llm_logical_retries: int = 0
    llm_provider_attempts_started: int = 0
    llm_provider_attempts_completed: int = 0
    llm_provider_attempts_failed: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_reasoning_tokens: int = 0
    llm_cache_read_tokens: int = 0
    llm_cache_write_tokens: int = 0
    llm_cost_usd: Decimal = Decimal("0")
    llm_provider_latency_ns: int = 0

    candidates_proposed: int = 0
    duplicate_proposals: int = 0
    schema_validation_requests: int = 0
    schema_validation_failures: int = 0
    deterministic_precheck_requests: int = 0
    deterministic_precheck_failures: int = 0

    evaluation_requests: int = 0
    evaluation_attempts_started: int = 0
    evaluation_attempts_completed: int = 0
    evaluation_successes: int = 0
    candidate_evaluation_failures: int = 0
    infrastructure_evaluation_failures: int = 0
    system_evaluation_failures: int = 0
    evaluation_retries: int = 0

    full_evaluator_requests: int = 0
    full_evaluator_attempts_started: int = 0
    full_evaluator_attempts_completed: int = 0
    full_evaluator_successes: int = 0
    full_candidate_failures: int = 0
    full_infrastructure_failures: int = 0
    full_system_failures: int = 0
    full_evaluator_retries: int = 0

    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: int = 0
    cache_writes: int = 0
    successful_objective_vectors: int = 0
    candidate_admissions: int = 0
    evaluator_worker_time_ns: int = 0

    def __post_init__(self) -> None:
        if self.projection_version != RUN_COUNTERS_PROJECTION_VERSION:
            raise ValueError(
                f"Unsupported RunCounters projection version {self.projection_version}"
            )
        for field in fields(self):
            if field.name in ("projection_version", "llm_cost_usd"):
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Counter {field.name} must be a non-negative integer")
        if not isinstance(self.llm_cost_usd, Decimal) or not self.llm_cost_usd.is_finite():
            raise TypeError("llm_cost_usd must be a finite Decimal")
        if self.llm_cost_usd < 0:
            raise ValueError("llm_cost_usd must be non-negative")


def _increment(counters: RunCounters, **deltas: int | Decimal) -> RunCounters:
    updates = {}
    for name, delta in deltas.items():
        if not hasattr(counters, name):  # pragma: no cover - implementation guard.
            raise ProjectionError(f"Unknown counter {name!r}")
        updates[name] = getattr(counters, name) + delta
    return replace(counters, **updates)


def apply_counter_event(counters: RunCounters, event: EventEnvelope) -> RunCounters:
    """Apply one event to the immutable counter snapshot."""

    counters = _increment(counters, event_count=1)
    payload = event.payload

    if isinstance(payload, RunStarted):
        return _increment(counters, runs_started=1)
    if isinstance(payload, RunFinished):
        return _increment(counters, runs_finished=1)
    if isinstance(payload, RunAborted):
        return _increment(counters, runs_aborted=1)
    if isinstance(payload, GenerationStarted):
        return _increment(counters, generations_started=1)
    if isinstance(payload, GenerationCompleted):
        return _increment(counters, generations_completed=1)
    if isinstance(payload, OperatorSelected):
        return _increment(counters, operators_selected=1)

    if isinstance(payload, LLMCallRequested):
        return _increment(counters, llm_logical_calls_requested=1)
    if isinstance(payload, LLMCallStarted):
        return _increment(counters, llm_logical_calls_started=1)
    if isinstance(payload, LLMCallCompleted):
        return _increment(counters, llm_logical_calls_completed=1)
    if isinstance(payload, LLMCallFailed):
        return _increment(counters, llm_logical_calls_failed=1)
    if isinstance(payload, LLMCallRetried):
        return _increment(counters, llm_logical_retries=1)
    if isinstance(payload, ProviderAttemptStarted):
        return _increment(counters, llm_provider_attempts_started=1)
    if isinstance(payload, ProviderAttemptCompleted):
        return _increment(
            counters,
            llm_provider_attempts_completed=1,
            llm_input_tokens=payload.input_tokens,
            llm_output_tokens=payload.output_tokens,
            llm_reasoning_tokens=payload.reasoning_tokens,
            llm_cache_read_tokens=payload.cache_read_tokens,
            llm_cache_write_tokens=payload.cache_write_tokens,
            llm_cost_usd=payload.cost_usd,
            llm_provider_latency_ns=payload.latency_ns,
        )
    if isinstance(payload, ProviderAttemptFailed):
        return _increment(
            counters,
            llm_provider_attempts_failed=1,
            llm_provider_latency_ns=payload.latency_ns,
        )

    if isinstance(payload, CandidateProposed):
        return _increment(counters, candidates_proposed=1)
    if isinstance(payload, DuplicateDetected):
        return _increment(counters, duplicate_proposals=1)
    if isinstance(payload, ValidationStarted):
        if payload.stage is ValidationStage.SCHEMA:
            return _increment(counters, schema_validation_requests=1)
        return _increment(counters, deterministic_precheck_requests=1)
    if isinstance(payload, ValidationCompleted) and not payload.ok:
        if payload.stage is ValidationStage.SCHEMA:
            return _increment(counters, schema_validation_failures=1)
        return _increment(counters, deterministic_precheck_failures=1)

    if isinstance(payload, EvaluationRequested):
        deltas = {"evaluation_requests": 1}
        if payload.fidelity == "full":
            deltas["full_evaluator_requests"] = 1
        return _increment(counters, **deltas)
    if isinstance(payload, EvaluationCacheHit):
        return _increment(
            counters,
            cache_lookups=1,
            cache_hits=1,
            successful_objective_vectors=1,
        )
    if isinstance(payload, EvaluationCacheMiss):
        return _increment(counters, cache_lookups=1, cache_misses=1)
    if isinstance(payload, EvaluationCacheBypassed):
        return _increment(counters, cache_bypasses=1)
    if isinstance(payload, EvaluationCacheStored):
        return _increment(counters, cache_writes=1)
    if isinstance(payload, EvaluationStarted):
        deltas = {"evaluation_attempts_started": 1}
        if payload.fidelity == "full":
            deltas["full_evaluator_attempts_started"] = 1
        return _increment(counters, **deltas)
    if isinstance(payload, EvaluationCompleted):
        deltas = {
            "evaluation_attempts_completed": 1,
            "evaluation_successes": 1,
            "successful_objective_vectors": 1,
            "evaluator_worker_time_ns": payload.worker_time_ns,
        }
        if payload.fidelity == "full":
            deltas.update(
                full_evaluator_attempts_completed=1,
                full_evaluator_successes=1,
            )
        return _increment(counters, **deltas)
    if isinstance(payload, EvaluationFailed):
        deltas = {"evaluator_worker_time_ns": payload.worker_time_ns}
        # A failed evaluator attempt is still a completed physical attempt. A
        # failure without an attempt ID happened before an evaluator worker was
        # started and therefore must not close a nonexistent attempt.
        if payload.evaluation_attempt_id is not None:
            deltas["evaluation_attempts_completed"] = 1
        category_field = {
            FailureCategory.CANDIDATE: "candidate_evaluation_failures",
            FailureCategory.INFRASTRUCTURE: "infrastructure_evaluation_failures",
            FailureCategory.SYSTEM: "system_evaluation_failures",
        }[payload.category]
        deltas[category_field] = 1
        if payload.fidelity == "full":
            if payload.evaluation_attempt_id is not None:
                deltas["full_evaluator_attempts_completed"] = 1
            full_field = {
                FailureCategory.CANDIDATE: "full_candidate_failures",
                FailureCategory.INFRASTRUCTURE: "full_infrastructure_failures",
                FailureCategory.SYSTEM: "full_system_failures",
            }[payload.category]
            deltas[full_field] = 1
        return _increment(counters, **deltas)
    if isinstance(payload, EvaluationRetried):
        deltas = {"evaluation_retries": 1}
        if payload.fidelity == "full":
            deltas["full_evaluator_retries"] = 1
        return _increment(counters, **deltas)
    if isinstance(payload, CandidateAdmitted):
        return _increment(counters, candidate_admissions=1)
    return counters


def project_run_counters(events: Iterable[EventEnvelope]) -> RunCounters:
    """Fold one contiguous run stream into a version-1 counter projection."""

    counters = RunCounters()
    run_id: Optional[RunId] = None
    expected_sequence = 1
    for event in events:
        if run_id is None:
            run_id = event.run_id
        elif event.run_id != run_id:
            raise ProjectionError(
                f"Counter projection received multiple runs: {run_id} and {event.run_id}"
            )
        if event.sequence_number != expected_sequence:
            raise ProjectionError(
                f"Expected event sequence {expected_sequence}, got {event.sequence_number}"
            )
        counters = apply_counter_event(counters, event)
        expected_sequence += 1
    return counters

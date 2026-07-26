"""Production and deterministic ID factories."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from typing import DefaultDict, Type, TypeVar

from agent_evolve.domain.ids import (
    CandidateId,
    CorrelationId,
    EvaluationAttemptId,
    EvaluationId,
    EventId,
    GenerationId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
    ProviderAttemptId,
    RunId,
    StableId,
    validate_id_namespace,
)

IdT = TypeVar("IdT", bound=StableId)
class UuidIdFactory:
    """Allocate opaque UUID4-backed IDs.

    Ordering comes from the event sequence, so IDs do not encode wall time and do
    not risk clock-regression ambiguity.
    """

    @staticmethod
    def _new(id_type: Type[IdT]) -> IdT:
        return id_type(f"{id_type.PREFIX}_{uuid.uuid4().hex}")

    def new_run_id(self) -> RunId:
        return self._new(RunId)

    def new_event_id(self) -> EventId:
        return self._new(EventId)

    def new_generation_id(self) -> GenerationId:
        return self._new(GenerationId)

    def new_candidate_id(self) -> CandidateId:
        return self._new(CandidateId)

    def new_insight_id(self) -> InsightId:
        return self._new(InsightId)

    def new_operator_invocation_id(self) -> OperatorInvocationId:
        return self._new(OperatorInvocationId)

    def new_llm_call_id(self) -> LLMCallId:
        return self._new(LLMCallId)

    def new_provider_attempt_id(self) -> ProviderAttemptId:
        return self._new(ProviderAttemptId)

    def new_evaluation_id(self) -> EvaluationId:
        return self._new(EvaluationId)

    def new_evaluation_attempt_id(self) -> EvaluationAttemptId:
        return self._new(EvaluationAttemptId)

    def new_correlation_id(self) -> CorrelationId:
        return self._new(CorrelationId)


class DeterministicIdFactory:
    """Per-ID-type deterministic allocator for tests and recorded replay."""

    def __init__(self, namespace: str = "test") -> None:
        validate_id_namespace(namespace)
        self._namespace = namespace
        self._counters: DefaultDict[Type[StableId], int] = defaultdict(int)
        self._lock = threading.Lock()

    def _new(self, id_type: Type[IdT]) -> IdT:
        with self._lock:
            self._counters[id_type] += 1
            ordinal = self._counters[id_type]
        return id_type(f"{id_type.PREFIX}_{self._namespace}_{ordinal:06d}")

    def new_run_id(self) -> RunId:
        return self._new(RunId)

    def new_event_id(self) -> EventId:
        return self._new(EventId)

    def new_generation_id(self) -> GenerationId:
        return self._new(GenerationId)

    def new_candidate_id(self) -> CandidateId:
        return self._new(CandidateId)

    def new_insight_id(self) -> InsightId:
        return self._new(InsightId)

    def new_operator_invocation_id(self) -> OperatorInvocationId:
        return self._new(OperatorInvocationId)

    def new_llm_call_id(self) -> LLMCallId:
        return self._new(LLMCallId)

    def new_provider_attempt_id(self) -> ProviderAttemptId:
        return self._new(ProviderAttemptId)

    def new_evaluation_id(self) -> EvaluationId:
        return self._new(EvaluationId)

    def new_evaluation_attempt_id(self) -> EvaluationAttemptId:
        return self._new(EvaluationAttemptId)

    def new_correlation_id(self) -> CorrelationId:
        return self._new(CorrelationId)

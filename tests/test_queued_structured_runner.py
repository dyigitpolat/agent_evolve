from __future__ import annotations

import asyncio
import gc
import hashlib
import json
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from typing import Any, Awaitable, Deque, Iterable, TypeVar

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel, ConfigDict, create_model
from pydantic_ai import Agent
from pydantic_ai.exceptions import ContentFilterError, ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from agent_evolve.application.llm_task_queue import (
    AsyncLLMTaskQueue,
    LLMTaskQueueClosedError,
    LLMTaskQueueFullError,
)
from agent_evolve.application.concurrent_stage import gather_concurrent_stage
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.llm_task_queue import (
    AttemptStatus,
    AttemptRequestVariant,
    CancellationReason,
    LLMAttemptContext,
    NANOSECONDS_PER_SECOND,
    PartitionedRetryBudget,
    RetryAfter,
    RetryAfterSource,
    RetryClassification,
    RetryDisposition,
    RetryReason,
    SanitizedAttemptFailure,
    SanitizedValidationIssue,
    StructuredOutputFailureMode,
    TaskOutcomeStatus,
    ValidationIssueCategory,
    ValidationIssueReasonCode,
)
from agent_evolve.infrastructure.asyncio_runtime import (
    AsyncioRuntime,
    TransportAbortedTimeoutError,
)
from agent_evolve.infrastructure.clock import FakeClock, SystemClock
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    PydanticAIAgenticGenerator,
    REFLECTION_PROMPT_RENDERER_DEFINITION_SHA256,
    REFLECTION_PROMPT_RENDERER_ID,
    REFLECTION_PROMPT_RENDERER_REVISION,
    _candidate_proposal_type,
    render_reflection_prompt,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    PydanticAIStructuredGenerator,
    classify_generation_exception,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    CancelledOutcomePublicationError,
    ExactPayloadAttemptPolicy,
    ExactTransportSchemaRepairAttemptPolicy,
    MAX_SCHEMA_REPAIR_REQUIRED_PATHS,
    MAX_SCHEMA_REPAIR_SCHEMA_NODES,
    MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES,
    NonRepeatingStreamTransportRetryClassifier,
    OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier,
    OpaqueHTTP400AndSchemaRepairOnceRetryClassifier,
    OpaqueHTTP400OnceRetryClassifier,
    OutcomePublicationError,
    OutcomePublicationPolicy,
    QueuedStructuredGenerationError,
    QueuedStructuredGenerationRunner,
    SCHEMA_REPAIR_POLICY_MANIFEST,
    SCHEMA_REPAIR_PROMPT_RENDERER_ID,
    SCHEMA_REPAIR_PROMPT_RENDERER_REVISION,
    STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION,
    SchemaRepairAttemptPolicy,
    StructuredEvidencePublicationError,
    StructuredEvidencePublicationPolicy,
    StructuredEvidencePublicationStage,
    StructuredGenerationExecutor,
    StructuredGenerationRetryClassifier,
    TransportOnlyStructuredGenerationRetryClassifier,
    _SEMANTIC_REPAIR_GUIDANCE,
    create_production_queued_runner,
    structured_generation_outcome_record,
    validate_structured_generation_output_evidence_record,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.integrations.pydantic_ai.provider_attempt_join import (
    validate_structured_generation_outcome_record,
)
from agent_evolve.policies.llm_backoff import (
    ExponentialBackoff,
    FullJitter,
    NoJitter,
)
from agent_evolve.ports.generation_failure import GenerationFailureDisposition
from agent_evolve.ports.agentic_generator import (
    ReflectionGenerationRequest,
    VariationGenerationRequest,
)
from agent_evolve.ports.structured_generator import (
    CanonicalProviderErrorCode,
    GenerationFailureKind,
    IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256,
    IDENTITY_PROMPT_RENDERER_ID,
    IDENTITY_PROMPT_RENDERER_REVISION,
    StructuredGenerationError,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredOutputRepairLiteralSet,
    StructuredStreamCleanupTimeoutError,
    StructuredStreamLivenessPolicy,
    StructuredStreamTimeoutError,
    StructuredStreamTimeoutPhase,
    identity_prompt_lineage,
)


T = TypeVar("T")


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: int


class _CardOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    option_id: str


class _HostileModelDumpOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: int

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("instance model_dump override must not run")


class _RepairPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metric_id: str
    direction: str


class _RepairInsight(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim: str
    trigger: str
    mechanism: str
    affected_paths: list[str]
    evidence_summary: str
    evidence_contrast_ids: list[str]
    confidence: float
    effect_predictions: list[_RepairPrediction]
    recommended_option_families: list[str]
    action_template: str
    falsification_condition: str


class _RepairEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    insights: list[_RepairInsight]


def _request(number: int) -> StructuredGenerationRequest[_Output]:
    return StructuredGenerationRequest(
        call_id=LLMCallId(f"call_queued_{number:04d}"),
        operation="queued_test",
        prompt=f"Return answer {number}.",
        output_type=_Output,
        output_tool_name="return_answer",
        max_output_tokens=64,
        temperature=0.0,
    )


def _provider_payload_view(
    request: StructuredGenerationRequest[Any],
) -> StructuredGenerationRequest[Any]:
    """Remove queue-only physical-attempt identity from request equality."""

    return replace(request, provider_attempt_id=None)


def test_identity_prompt_lineage_cannot_forge_semantic_or_renderer_identity() -> None:
    request = _request(98)
    lineage = identity_prompt_lineage(request.prompt)
    assert replace(request, prompt_lineage=lineage).prompt_lineage is lineage

    forged_lineages = (
        replace(lineage, semantic_prompt_sha256="0" * 64),
        replace(lineage, renderer_revision="identity_v2"),
        replace(lineage, renderer_definition_sha256="f" * 64),
    )
    for forged in forged_lineages:
        with pytest.raises(ValueError, match="does not authenticate"):
            replace(request, prompt_lineage=forged)


def test_exact_payload_attempt_policy_never_mutates_a_retry_treatment() -> None:
    request = _request(99)
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.MISSING,
                ("answer",),
            ),
        ),
    )
    context = LLMAttemptContext(
        task_id=request.call_id.value,
        attempt_number=2,
        attempt_timeout_ns=NANOSECONDS_PER_SECOND,
        previous_failure=failure,
        active_output_failure=failure,
    )

    prepared = ExactPayloadAttemptPolicy().request_for_attempt(
        request,
        context=context,
    )

    assert prepared.request is request
    assert prepared.evidence.variant is AttemptRequestVariant.ORIGINAL
    assert (
        prepared.evidence.prompt_sha256
        == hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
    )


def _response(request: StructuredGenerationRequest[_Output]):
    number = int(request.call_id.value.rsplit("_", 1)[1])
    return StructuredGenerationResponse(
        value=_Output(answer=number),
        requested_model="requested/model",
        resolved_model="resolved/model",
        resolved_provider="provider",
        provider_response_id=f"response-{number}",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=2,
        reasoning_tokens=1,
        cache_read_tokens=3,
        cache_write_tokens=0,
        cost_usd=Decimal("0.001"),
        latency_ns=50,
    )


def _response_for_value(
    request: StructuredGenerationRequest[Any],
    value: BaseModel,
) -> StructuredGenerationResponse[Any]:
    return StructuredGenerationResponse(
        value=value,
        requested_model="requested/model",
        resolved_model="resolved/model",
        resolved_provider="provider",
        provider_response_id=f"response-{request.call_id.value}",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=2,
        reasoning_tokens=1,
        cache_read_tokens=3,
        cache_write_tokens=0,
        cost_usd=Decimal("0.001"),
        latency_ns=50,
    )


def _error(
    kind: GenerationFailureKind,
    *,
    retryable: bool,
    retry_after_seconds: float | None = None,
    status_code: int | None = None,
    message: str = "sanitized provider failure",
    output_failure_mode: StructuredOutputFailureMode | None = None,
    validation_issues: tuple[SanitizedValidationIssue, ...] = (),
) -> StructuredGenerationError:
    return StructuredGenerationError(
        kind=kind,
        retryable=retryable,
        safe_message=message,
        status_code=status_code,
        retry_after_seconds=retry_after_seconds,
        output_failure_mode=output_failure_mode,
        validation_issues=validation_issues,
    )


def _sanitized_failure(
    kind: GenerationFailureKind,
    *,
    retryable: bool,
    retry_after_seconds: float | None = None,
    status_code: int | None = None,
    message: str = "sanitized provider failure",
    output_failure_mode: StructuredOutputFailureMode | None = None,
    validation_issues: tuple[SanitizedValidationIssue, ...] = (),
) -> SanitizedAttemptFailure:
    return SanitizedAttemptFailure(
        kind=kind.value,
        retryable=retryable,
        safe_message=message,
        status_code=status_code,
        retry_after_seconds=retry_after_seconds,
        output_failure_mode=output_failure_mode,
        validation_issues=validation_issues,
    )


def _pre_stream_read_failure() -> ModelAPIError:
    """Build the typed, content-safe shape observed in the R2 transport trace."""

    request = httpx.Request("POST", "https://example.invalid")
    read_error = httpx.ReadError("RAW_SECRET_FAKE_READ", request=request)
    connection_error = APIConnectionError(
        message="RAW_SECRET_FAKE_CONNECTION",
        request=request,
    )
    connection_error.__cause__ = read_error
    wrapped = ModelAPIError("offline/test-model", "RAW_SECRET_FAKE_WRAPPER")
    wrapped.__cause__ = connection_error
    return wrapped


class _ScriptedGenerator:
    def __init__(self, script: Iterable[object]) -> None:
        self.script: Deque[object] = deque(script)
        self.requests: list[StructuredGenerationRequest[_Output]] = []

    async def generate_once(self, request: StructuredGenerationRequest[_Output]):
        self.requests.append(request)
        item = self.script.popleft()
        if isinstance(item, BaseException):
            raise item
        if item == "response":
            return _response(request)
        return item


class _ContractAwareGenerator:
    def __init__(self) -> None:
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def generate_once(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        self.requests.append(request)
        if request.output_tool_name == "return_reflection_insights":
            value = request.output_type(insights=[])
        elif request.output_tool_name == "return_candidate_proposal":
            value = request.output_type(
                configuration={"answer": 17},
                design_rationale="Use the exact typed proposal.",
            )
        elif request.output_tool_name == "return_card":
            value = request.output_type(option_id="trim.other_allowed_card")
        elif request.output_tool_name == "return_hostile_output":
            value = request.output_type(answer=23)
        else:  # pragma: no cover - the focused evidence tests close this set.
            raise AssertionError("unexpected output tool")
        return _response_for_value(request, value)


class _DeterministicRuntime:
    def __init__(self, clock: FakeClock, *, timeout_calls: Iterable[int] = ()) -> None:
        self.clock = clock
        self.timeout_calls = frozenset(timeout_calls)
        self.wait_calls: list[int] = []
        self.sleep_calls: list[int] = []

    async def sleep(self, delay_ns: int) -> None:
        self.sleep_calls.append(delay_ns)
        self.clock.advance_ns(delay_ns)
        await asyncio.sleep(0)

    async def wait_for(self, awaitable: Awaitable[T], timeout_ns: int) -> T:
        self.wait_calls.append(timeout_ns)
        if len(self.wait_calls) not in self.timeout_calls:
            return await awaitable
        child = asyncio.create_task(awaitable)
        await asyncio.sleep(0)
        self.clock.advance_ns(timeout_ns)
        child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        raise TimeoutError("queue-owned deterministic timeout")


def _make_runner(
    generator: Any,
    *,
    max_in_flight: int = 1,
    max_pending: int = 2,
    max_attempts: int = 3,
    retry_budget: PartitionedRetryBudget | None = None,
    attempt_timeout_ns: int = 100,
    base_backoff_ns: int = 0,
    max_backoff_ns: int = 0,
    runtime: _DeterministicRuntime | None = None,
    outcome_sink=None,
    outcome_publication_policy: OutcomePublicationPolicy = (
        OutcomePublicationPolicy.BEST_EFFORT
    ),
    request_evidence_sink=None,
    output_evidence_sink=None,
    evidence_publication_policy: StructuredEvidencePublicationPolicy = (
        StructuredEvidencePublicationPolicy.BEST_EFFORT
    ),
    attempt_request_policy=None,
    retry_classifier=None,
):
    clock = runtime.clock if runtime is not None else FakeClock()
    runtime = runtime or _DeterministicRuntime(clock)
    queue = AsyncLLMTaskQueue(
        executor=StructuredGenerationExecutor(
            generator,
            attempt_request_policy=attempt_request_policy,
        ),
        retry_classifier=retry_classifier or StructuredGenerationRetryClassifier(),
        backoff_policy=ExponentialBackoff(
            base_backoff_ns,
            max_backoff_ns,
            NoJitter(),
        ),
        clock=clock,
        max_in_flight=max_in_flight,
        max_pending=max_pending,
        attempt_timeout_ns=attempt_timeout_ns,
        runtime=runtime,
    )
    return (
        QueuedStructuredGenerationRunner(
            queue=queue,
            max_attempts=max_attempts,
            retry_budget=retry_budget,
            outcome_sink=outcome_sink,
            outcome_publication_policy=outcome_publication_policy,
            request_evidence_sink=request_evidence_sink,
            output_evidence_sink=output_evidence_sink,
            evidence_publication_policy=evidence_publication_policy,
        ),
        runtime,
    )


def test_transport_only_exact_payload_queue_integration() -> None:
    terminal_outcomes = []
    invalid_generator = _ScriptedGenerator(
        [_error(GenerationFailureKind.OUTPUT_INVALID, retryable=True), "response"]
    )
    invalid_runner, _ = _make_runner(
        invalid_generator,
        max_attempts=2,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
        outcome_sink=terminal_outcomes.append,
    )

    async def invalid_scenario() -> None:
        try:
            with pytest.raises(QueuedStructuredGenerationError):
                await invalid_runner(_request(70))
        finally:
            await invalid_runner.aclose()

    asyncio.run(invalid_scenario())
    assert len(invalid_generator.requests) == 1
    assert len(terminal_outcomes) == 1
    assert len(terminal_outcomes[0].telemetry.attempts) == 1
    assert (
        terminal_outcomes[0].telemetry.attempts[0].request_evidence.variant
        is AttemptRequestVariant.ORIGINAL
    )

    terminal_outcomes.clear()
    transient_generator = _ScriptedGenerator(
        [
            _error(GenerationFailureKind.PROVIDER_UNAVAILABLE, retryable=True),
            "response",
        ]
    )
    transient_runner, _ = _make_runner(
        transient_generator,
        max_attempts=2,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
        outcome_sink=terminal_outcomes.append,
    )

    async def transient_scenario() -> None:
        try:
            await transient_runner(_request(71))
        finally:
            await transient_runner.aclose()

    asyncio.run(transient_scenario())
    assert len(transient_generator.requests) == 2
    first, second = transient_generator.requests
    assert first.prompt == second.prompt
    assert first.output_type is second.output_type
    assert first.output_tool_name == second.output_tool_name
    assert first.max_output_tokens == second.max_output_tokens
    assert first.temperature == second.temperature
    attempts = terminal_outcomes[0].telemetry.attempts
    assert len(attempts) == 2
    assert all(
        attempt.request_evidence.variant is AttemptRequestVariant.ORIGINAL
        for attempt in attempts
    )


def test_opaque_http_400_queue_retry_preserves_provider_payload_exactly() -> None:
    observed = []
    generator = _ScriptedGenerator([_opaque_http_400(), "response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=OpaqueHTTP400OnceRetryClassifier(),
        outcome_sink=observed.append,
    )
    original = _request(74)

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(original)
        finally:
            await runner.aclose()

    response = asyncio.run(scenario())

    assert response.attempt_count == 2
    assert len(generator.requests) == 2
    assert [_provider_payload_view(item) for item in generator.requests] == [
        original,
        original,
    ]
    attempts = observed[0].telemetry.attempts
    assert [item.status for item in attempts] == [
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.SUCCEEDED,
    ]
    assert attempts[0].classification.reason is RetryReason.TRANSIENT
    assert all(
        item.request_evidence.variant is AttemptRequestVariant.ORIGINAL
        for item in attempts
    )


def test_composite_recovery_repairs_one_output_invalid_then_succeeds() -> None:
    observed = []
    generator = _ScriptedGenerator(
        [
            _error(
                GenerationFailureKind.OUTPUT_INVALID,
                retryable=True,
                output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
                validation_issues=(
                    SanitizedValidationIssue(
                        ValidationIssueCategory.LITERAL_OR_ENUM,
                        ("answer",),
                    ),
                ),
            ),
            "response",
        ]
    )
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        attempt_request_policy=ExactTransportSchemaRepairAttemptPolicy(),
        retry_classifier=OpaqueHTTP400AndSchemaRepairOnceRetryClassifier(),
        outcome_sink=observed.append,
    )

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(_request(75))
        finally:
            await runner.aclose()

    response = asyncio.run(scenario())

    assert response.attempt_count == 2
    assert len(generator.requests) == 2
    assert [
        item.request_evidence.variant for item in observed[0].telemetry.attempts
    ] == [AttemptRequestVariant.ORIGINAL, AttemptRequestVariant.SCHEMA_REPAIR_V3]
    assert generator.requests[1].prompt.startswith(generator.requests[0].prompt)
    assert "STRUCTURED_OUTPUT_SCHEMA_REPAIR_V3" in generator.requests[1].prompt


def test_composite_recovery_keeps_second_output_invalid_terminal() -> None:
    observed = []
    invalid = _error(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.LITERAL_OR_ENUM,
                ("answer",),
            ),
        ),
    )
    generator = _ScriptedGenerator([invalid, invalid, "response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        attempt_request_policy=ExactTransportSchemaRepairAttemptPolicy(),
        retry_classifier=OpaqueHTTP400AndSchemaRepairOnceRetryClassifier(),
        outcome_sink=observed.append,
    )

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(_request(76))
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.TERMINAL_FAILURE
    assert len(generator.requests) == 2
    assert [
        item.request_evidence.variant for item in observed[0].telemetry.attempts
    ] == [AttemptRequestVariant.ORIGINAL, AttemptRequestVariant.SCHEMA_REPAIR_V3]
    assert observed[0].telemetry.attempts[1].classification.disposition is (
        RetryDisposition.FAIL
    )


def test_bounded_recovery_resamples_schema_repair_until_attempt_budget() -> None:
    observed = []
    invalid = _error(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.MALFORMED_ARGUMENTS,
                ("root",),
            ),
        ),
    )
    generator = _ScriptedGenerator([invalid, invalid, "response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        attempt_request_policy=ExactTransportSchemaRepairAttemptPolicy(),
        retry_classifier=(
            OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier()
        ),
        outcome_sink=observed.append,
    )

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(_request(761))
        finally:
            await runner.aclose()

    response = asyncio.run(scenario())

    assert response.attempt_count == 3
    assert len(generator.requests) == 3
    assert [
        item.request_evidence.variant for item in observed[0].telemetry.attempts
    ] == [
        AttemptRequestVariant.ORIGINAL,
        AttemptRequestVariant.SCHEMA_REPAIR_V3,
        AttemptRequestVariant.SCHEMA_REPAIR_V3,
    ]
    assert generator.requests[1].prompt != generator.requests[2].prompt
    assert "Repair pass: 1" in generator.requests[1].prompt
    assert "Repair pass: 2" in generator.requests[2].prompt
    assert "FINAL BOUNDED REPAIR PASS" in generator.requests[2].prompt
    assert [item.status for item in observed[0].telemetry.attempts] == [
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.SUCCEEDED,
    ]


def test_partitioned_budget_replays_escalated_repair_after_transport_failure() -> None:
    """Reproduce the OSS-20B failure topology without spending a live call."""

    observed = []
    invalid = _error(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.LITERAL_OR_ENUM,
                ("answer",),
            ),
        ),
    )
    unavailable = _error(
        GenerationFailureKind.PROVIDER_UNAVAILABLE,
        retryable=True,
    )
    generator = _ScriptedGenerator([invalid, invalid, unavailable, "response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=5,
        retry_budget=PartitionedRetryBudget(
            output_invalid_retries=2,
            transport_retries=2,
        ),
        attempt_request_policy=ExactTransportSchemaRepairAttemptPolicy(),
        retry_classifier=OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier(),
        outcome_sink=observed.append,
    )

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(_request(762))
        finally:
            await runner.aclose()

    response = asyncio.run(scenario())

    assert response.attempt_count == 4
    assert [row.status for row in observed[0].telemetry.attempts] == [
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.SUCCEEDED,
    ]
    first, repair_one, repair_two, transport_replay = generator.requests
    assert "Repair pass: 1" in repair_one.prompt
    assert "Repair pass: 2" in repair_two.prompt
    assert "FINAL BOUNDED REPAIR PASS" in repair_two.prompt
    assert repair_two.prompt == transport_replay.prompt
    assert repair_two.provider_attempt_id != transport_replay.provider_attempt_id
    assert first.prompt != repair_one.prompt != repair_two.prompt
    attempts = observed[0].telemetry.attempts
    assert (
        attempts[2].request_evidence.prompt_sha256
        == attempts[3].request_evidence.prompt_sha256
    )


def test_composite_recovery_preserves_opaque_replay_before_schema_repair() -> None:
    observed = []
    generator = _ScriptedGenerator(
        [
            _opaque_http_400(),
            _error(
                GenerationFailureKind.OUTPUT_INVALID,
                retryable=True,
                output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
                validation_issues=(
                    SanitizedValidationIssue(
                        ValidationIssueCategory.LITERAL_OR_ENUM,
                        ("answer",),
                    ),
                ),
            ),
            "response",
        ]
    )
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        attempt_request_policy=ExactTransportSchemaRepairAttemptPolicy(),
        retry_classifier=OpaqueHTTP400AndSchemaRepairOnceRetryClassifier(),
        outcome_sink=observed.append,
    )
    original = _request(77)

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(original)
        finally:
            await runner.aclose()

    response = asyncio.run(scenario())

    assert response.attempt_count == 3
    first, replay, repair = generator.requests
    assert _provider_payload_view(first) == original
    assert _provider_payload_view(replay) == original
    assert repair.prompt.startswith(original.prompt)
    assert [
        item.request_evidence.variant for item in observed[0].telemetry.attempts
    ] == [
        AttemptRequestVariant.ORIGINAL,
        AttemptRequestVariant.ORIGINAL,
        AttemptRequestVariant.SCHEMA_REPAIR_V3,
    ]


def test_stream_liveness_timeout_retries_as_exact_transport_failure() -> None:
    observed = []
    generator = _ScriptedGenerator(
        [
            StructuredStreamTimeoutError(StructuredStreamTimeoutPhase.IDLE),
            "response",
        ]
    )
    runner, _ = _make_runner(
        generator,
        max_attempts=2,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
        outcome_sink=observed.append,
    )

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(_request(72))
        finally:
            await runner.aclose()

    response = asyncio.run(scenario())

    assert response.attempt_count == 2
    assert len(generator.requests) == 2
    assert [_provider_payload_view(item) for item in generator.requests] == [
        _request(72),
        _request(72),
    ]
    assert (
        generator.requests[0].provider_attempt_id
        != generator.requests[1].provider_attempt_id
    )
    assert [row.status for row in observed[0].telemetry.attempts] == [
        AttemptStatus.TIMED_OUT,
        AttemptStatus.SUCCEEDED,
    ]
    evidence_ids = [
        row.request_evidence.provider_attempt_id
        for row in observed[0].telemetry.attempts
    ]
    assert evidence_ids == [
        request.provider_attempt_id for request in generator.requests
    ]
    assert len(set(evidence_ids)) == 2
    first_failure = observed[0].telemetry.attempts[0].classification.sanitized_failure
    assert first_failure is not None
    assert first_failure.stream_timeout_phase is StructuredStreamTimeoutPhase.IDLE
    assert first_failure.safe_message == (
        "provider stream stopped producing progress before completion"
    )


def test_stream_cleanup_timeout_is_terminal_and_never_overlaps_a_retry() -> None:
    observed = []
    generator = _ScriptedGenerator(
        [
            StructuredStreamCleanupTimeoutError(StructuredStreamTimeoutPhase.IDLE),
            "response-that-must-never-run",
        ]
    )
    runner, _ = _make_runner(
        generator,
        max_attempts=2,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
        outcome_sink=observed.append,
    )

    async def scenario() -> None:
        try:
            with pytest.raises(QueuedStructuredGenerationError):
                await runner(_request(73))
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    assert len(generator.requests) == 1
    attempts = observed[0].telemetry.attempts
    assert len(attempts) == 1
    classification = attempts[0].classification
    assert classification is not None
    assert classification.disposition is RetryDisposition.FAIL
    assert classification.sanitized_failure is not None
    assert classification.sanitized_failure.retryable is False
    assert classification.sanitized_failure.stream_timeout_phase is (
        StructuredStreamTimeoutPhase.IDLE
    )


def test_success_propagates_attempt_count_and_publishes_exact_terminal_outcome() -> (
    None
):
    observed = []
    generator = _ScriptedGenerator(
        [
            _error(GenerationFailureKind.OUTPUT_INVALID, retryable=True),
            _error(GenerationFailureKind.PROVIDER_UNAVAILABLE, retryable=True),
            "response",
        ]
    )
    runner, runtime = _make_runner(generator, outcome_sink=observed.append)
    original = _request(1)

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(original)
        finally:
            await runner.aclose()

    result = asyncio.run(scenario())

    assert type(result) is AttemptedStructuredGenerationResponse
    assert result.response.value == _Output(answer=1)
    assert result.attempt_count == 3
    assert len(generator.requests) == 3
    assert runtime.sleep_calls == [0, 0]
    assert len(observed) == 1
    assert observed[0].status is TaskOutcomeStatus.SUCCEEDED
    assert observed[0].response is result.response
    assert [attempt.status for attempt in observed[0].telemetry.attempts] == [
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.SUCCEEDED,
    ]
    assert [
        attempt.classification.reason for attempt in observed[0].telemetry.attempts[:-1]
    ] == [RetryReason.OUTPUT_INVALID, RetryReason.TRANSIENT]
    first, repaired, repaired_after_transient = generator.requests
    assert _provider_payload_view(first) == original
    assert repaired.prompt != original.prompt
    assert _provider_payload_view(repaired_after_transient) == (
        _provider_payload_view(repaired)
    )
    assert repaired_after_transient.provider_attempt_id != repaired.provider_attempt_id
    assert [
        attempt.request_evidence.variant for attempt in observed[0].telemetry.attempts
    ] == [
        AttemptRequestVariant.ORIGINAL,
        AttemptRequestVariant.SCHEMA_REPAIR_V3,
        AttemptRequestVariant.SCHEMA_REPAIR_V3,
    ]
    assert (
        observed[0].telemetry.attempts[1].request_evidence.prompt_sha256
        == observed[0].telemetry.attempts[2].request_evidence.prompt_sha256
    )


def test_output_invalid_retry_uses_only_sanitized_schema_repair_guidance() -> None:
    issue = SanitizedValidationIssue(
        ValidationIssueCategory.MISSING,
        ("answer",),
    )
    generator = _ScriptedGenerator(
        [
            _error(
                GenerationFailureKind.OUTPUT_INVALID,
                retryable=True,
                message="sanitized diagnostic DO_NOT_COPY_THIS_TEXT",
                output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
                validation_issues=(issue,),
            ),
            "response",
        ]
    )
    observed = []
    runner, runtime = _make_runner(
        generator,
        max_attempts=2,
        outcome_sink=observed.append,
    )
    original_without_lineage = _request(101)
    original = replace(
        original_without_lineage,
        prompt_lineage=identity_prompt_lineage(original_without_lineage.prompt),
    )

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(original)
        finally:
            await runner.aclose()

    result = asyncio.run(scenario())

    assert result.attempt_count == 2
    assert runtime.sleep_calls == [0]
    assert len(generator.requests) == 2
    first, repaired = generator.requests
    assert _provider_payload_view(first) == original
    assert repaired is not original
    assert repaired.prompt.startswith(original.prompt)
    suffix = repaired.prompt[len(original.prompt) :]
    assert "STRUCTURED_OUTPUT_SCHEMA_REPAIR_V3" in suffix
    assert "schema_validation" in suffix
    assert "field paths from the trusted local output contract" in suffix
    assert '["/answer"]' in suffix
    assert "missing at answer" in suffix
    assert "return_answer" in suffix
    assert "Keep every field concise so the tool call completes." in suffix
    assert "DO_NOT_COPY_THIS_TEXT" not in suffix
    assert len(suffix.encode("utf-8")) <= MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES
    assert repaired.call_id == original.call_id
    assert repaired.operation == original.operation
    assert repaired.output_type is original.output_type
    assert repaired.output_tool_name == original.output_tool_name
    assert repaired.max_output_tokens == original.max_output_tokens
    assert repaired.temperature == original.temperature
    assert repaired.prompt_lineage is not None
    assert (
        repaired.prompt_lineage.semantic_prompt_sha256
        == hashlib.sha256(original.prompt.encode("utf-8")).hexdigest()
    )
    assert repaired.prompt_lineage.renderer_id == SCHEMA_REPAIR_PROMPT_RENDERER_ID
    assert (
        repaired.prompt_lineage.renderer_revision
        == SCHEMA_REPAIR_PROMPT_RENDERER_REVISION
    )
    evidence = [attempt.request_evidence for attempt in observed[0].telemetry.attempts]
    assert [item.variant for item in evidence] == [
        AttemptRequestVariant.ORIGINAL,
        AttemptRequestVariant.SCHEMA_REPAIR_V3,
    ]
    assert [item.prompt_sha256 for item in evidence] == [
        hashlib.sha256(original.prompt.encode("utf-8")).hexdigest(),
        hashlib.sha256(repaired.prompt.encode("utf-8")).hexdigest(),
    ]


def test_schema_repair_v3_lists_all_nested_required_fields_beyond_issue_cap() -> None:
    issue_field_names = tuple(list(_RepairInsight.model_fields)[:8])
    issues = tuple(
        SanitizedValidationIssue(
            ValidationIssueCategory.MISSING,
            ("insights", "item", field_name),
        )
        for field_name in issue_field_names
    )
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=issues,
    )
    request = StructuredGenerationRequest(
        call_id=LLMCallId("call_queued_nested_repair"),
        operation="generic_nested_repair_test",
        prompt="Return the typed output.",
        output_type=_RepairEnvelope,
        output_tool_name="return_nested_output",
        max_output_tokens=128,
        temperature=0.0,
    )

    prepared = SchemaRepairAttemptPolicy().request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=2,
            attempt_timeout_ns=100,
            active_output_failure=failure,
        ),
    )

    assert prepared.evidence.variant is AttemptRequestVariant.SCHEMA_REPAIR_V3
    suffix = prepared.request.prompt[len(request.prompt) :]
    expected_paths = {
        "/insights",
        *{f"/insights/*/{field_name}" for field_name in _RepairInsight.model_fields},
        "/insights/*/effect_predictions/*/metric_id",
        "/insights/*/effect_predictions/*/direction",
    }
    for path in expected_paths:
        assert json.dumps(path) in suffix
    # These late fields were hidden by the eight-issue diagnostic boundary but
    # are independently recovered from the trusted local output schema.
    assert "missing at insights.item.action_template" not in suffix
    assert json.dumps("/insights/*/action_template") in suffix
    assert "missing at insights.item.falsification_condition" not in suffix
    assert json.dumps("/insights/*/falsification_condition") in suffix
    assert len(suffix.encode("utf-8")) <= MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES


def test_semantic_reason_code_safely_guides_same_schema_repair_retry_variant() -> None:
    reason_code = ValidationIssueReasonCode.NO_FEASIBLE_DISJOINT_PORTFOLIO
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.SEMANTIC_CONSTRAINT,
                ("root",),
                reason_code,
            ),
        ),
    )
    request = _request(102)

    prepared = SchemaRepairAttemptPolicy().request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=2,
            attempt_timeout_ns=100,
            active_output_failure=failure,
        ),
    )

    assert prepared.evidence.variant is AttemptRequestVariant.SCHEMA_REPAIR_V3
    suffix = prepared.request.prompt[len(request.prompt) :]
    assert "semantic_constraint at root" in suffix
    assert f"reason={reason_code.value}" in suffix
    assert "pairwise changed-path" in suffix


def test_proposal_support_repair_cannot_crash_attempt_construction() -> None:
    reason_code = ValidationIssueReasonCode.PROPOSAL_SUPPORT_OPTION_OMITTED
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.SEMANTIC_CONSTRAINT,
                ("root",),
                reason_code,
            ),
        ),
    )
    request = _request(104)

    prepared = SchemaRepairAttemptPolicy().request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=2,
            attempt_timeout_ns=100,
            active_output_failure=failure,
        ),
    )

    assert prepared.evidence.variant is AttemptRequestVariant.SCHEMA_REPAIR_V3
    suffix = prepared.request.prompt[len(request.prompt) :]
    assert f"reason={reason_code.value}" in suffix
    assert "every engine-reserved proposal-support option" in suffix
    assert "copy each reserved option ID exactly" in suffix


def test_finite_option_repair_restates_complete_trusted_closed_set_and_escalates() -> None:
    allowed = tuple(f"option.family_{index:03d}" for index in range(145))
    literal_set = StructuredOutputRepairLiteralSet(
        field_path=("members", "*", "option_id"),
        allowed_literals=allowed,
    )
    request = StructuredGenerationRequest(
        call_id=LLMCallId("call_queued_finite_literal_repair"),
        operation="generic_finite_literal_repair_test",
        prompt="Return a typed portfolio from the sealed options.",
        output_type=_CardOutput,
        output_tool_name="return_finite_portfolio",
        max_output_tokens=128,
        temperature=0.0,
        repair_literal_sets=(literal_set,),
    )
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.SEMANTIC_CONSTRAINT,
                ("members", "item"),
                ValidationIssueReasonCode.FINITE_OPTION_OUT_OF_CONTRACT,
            ),
        ),
    )
    policy = SchemaRepairAttemptPolicy()

    first_repair = policy.request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=2,
            attempt_timeout_ns=100,
            previous_failure=failure,
            active_output_failure=failure,
        ),
    )
    final_repair = policy.request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=3,
            attempt_timeout_ns=100,
            previous_failure=failure,
            active_output_failure=failure,
        ),
    )

    literal_json = json.dumps(allowed, ensure_ascii=True, separators=(",", ":"))
    first_suffix = first_repair.request.prompt[len(request.prompt) :]
    final_suffix = final_repair.request.prompt[len(request.prompt) :]
    assert first_repair.evidence.variant is AttemptRequestVariant.SCHEMA_REPAIR_V3
    assert final_repair.evidence.variant is AttemptRequestVariant.SCHEMA_REPAIR_V3
    assert f"- /members/*/option_id={literal_json}" in first_suffix
    assert first_suffix.count(literal_json) == 1
    assert "Repair pass: 1" in first_suffix
    assert "FINAL BOUNDED REPAIR PASS" not in first_suffix
    assert f"- /members/*/option_id={literal_json}" in final_suffix
    assert "Repair pass: 2" in final_suffix
    assert "FINAL BOUNDED REPAIR PASS" in final_suffix
    assert first_repair.evidence.prompt_sha256 != final_repair.evidence.prompt_sha256
    assert len(final_suffix.encode("utf-8")) <= MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES


@pytest.mark.parametrize(
    ("reason_code", "expected_guidance"),
    (
        (
            ValidationIssueReasonCode.REFLECTION_METRIC_CONTRACT_VIOLATION,
            "each required metric exactly once",
        ),
        (
            ValidationIssueReasonCode.REFLECTION_ACTION_CONTRACT_VIOLATION,
            "use only request-listed values",
        ),
        (
            ValidationIssueReasonCode.REFLECTION_SEMANTIC_CONTRACT_VIOLATION,
            "use only the request-listed insight kind",
        ),
        (
            ValidationIssueReasonCode.REFLECTION_DIRECTION_OR_ANCHOR_VIOLATION,
            "adjudicable non-unknown direction",
        ),
    ),
)
def test_reflection_reason_codes_produce_actionable_generic_repair_guidance(
    reason_code: ValidationIssueReasonCode,
    expected_guidance: str,
) -> None:
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.SEMANTIC_CONSTRAINT,
                ("insights", "item"),
                reason_code,
            ),
        ),
    )
    request = _request(103)

    prepared = SchemaRepairAttemptPolicy().request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=2,
            attempt_timeout_ns=100,
            active_output_failure=failure,
        ),
    )

    suffix = prepared.request.prompt[len(request.prompt) :]
    assert f"reason={reason_code.value}" in suffix
    assert expected_guidance in suffix


def test_assigned_memory_card_repair_explains_the_closed_semantic_obligation() -> None:
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
        validation_issues=(
            SanitizedValidationIssue(
                ValidationIssueCategory.SEMANTIC_CONSTRAINT,
                ("root",),
                ValidationIssueReasonCode.ASSIGNED_MEMORY_CARD_OMITTED,
            ),
        ),
    )
    request = _request(102)

    prepared = SchemaRepairAttemptPolicy().request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=2,
            attempt_timeout_ns=100,
            active_output_failure=failure,
        ),
    )

    suffix = prepared.request.prompt[len(request.prompt) :]
    assert "every prospectively assigned memory-card key" in suffix
    assert "supporting_card_keys" in suffix
    assert "compatibility and dose bounds" in suffix


def test_schema_repair_never_emits_a_partial_over_bound_required_path_map() -> None:
    output_type = create_model(
        "OverBoundRepairOutput",
        __config__=ConfigDict(extra="forbid", strict=True),
        **{
            f"field_{index:03d}": (str, ...)
            for index in range(MAX_SCHEMA_REPAIR_REQUIRED_PATHS + 1)
        },
    )
    request = replace(
        _request(202),
        output_type=output_type,
        output_tool_name="return_over_bound_output",
    )
    failure = _sanitized_failure(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
    )

    prepared = SchemaRepairAttemptPolicy().request_for_attempt(
        request,
        context=LLMAttemptContext(
            task_id=request.call_id.value,
            attempt_number=2,
            attempt_timeout_ns=100,
            active_output_failure=failure,
        ),
    )

    assert prepared.request is request
    assert prepared.evidence.variant is AttemptRequestVariant.ORIGINAL
    assert "STRUCTURED_OUTPUT_SCHEMA_REPAIR" not in prepared.request.prompt


def test_non_output_retries_reuse_exact_original_request_across_three_attempts() -> (
    None
):
    generator = _ScriptedGenerator(
        [
            _error(GenerationFailureKind.PROVIDER_UNAVAILABLE, retryable=True),
            _error(GenerationFailureKind.RATE_LIMITED, retryable=True),
            "response",
        ]
    )
    observed = []
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        outcome_sink=observed.append,
    )
    original = _request(102)

    async def scenario() -> None:
        try:
            await runner(original)
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    assert [_provider_payload_view(item) for item in generator.requests] == [
        original,
        original,
        original,
    ]
    assert len({item.provider_attempt_id for item in generator.requests}) == 3
    evidence = [attempt.request_evidence for attempt in observed[0].telemetry.attempts]
    assert all(item.variant is AttemptRequestVariant.ORIGINAL for item in evidence)
    assert {item.prompt_sha256 for item in evidence} == {
        hashlib.sha256(original.prompt.encode("utf-8")).hexdigest()
    }
    assert "STRUCTURED_OUTPUT_SCHEMA_REPAIR" not in original.prompt
    assert "RAW_SECRET" not in repr(observed[0])


def test_second_invalid_output_exhausts_without_local_replacement() -> None:
    failure = _error(
        GenerationFailureKind.OUTPUT_INVALID,
        retryable=True,
        output_failure_mode=StructuredOutputFailureMode.TYPED_OUTPUT_CONTRACT,
    )
    generator = _ScriptedGenerator([failure, failure])
    runner, _ = _make_runner(generator, max_attempts=2)
    original = _request(103)

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(original)
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.ATTEMPTS_EXHAUSTED
    assert (
        error.generation_failure_disposition
        is GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
    )
    assert error.outcome.response is None
    assert len(generator.requests) == 2
    assert _provider_payload_view(generator.requests[0]) == original
    assert "STRUCTURED_OUTPUT_SCHEMA_REPAIR_V3" in generator.requests[1].prompt
    assert [
        attempt.request_evidence.variant for attempt in error.telemetry.attempts
    ] == [
        AttemptRequestVariant.ORIGINAL,
        AttemptRequestVariant.SCHEMA_REPAIR_V3,
    ]


def test_content_filter_is_terminal_without_schema_repair_or_second_request() -> None:
    filtered = classify_generation_exception(
        ContentFilterError(
            "RAW_SECRET_FILTER_MESSAGE",
            body="RAW_SECRET_PARTIAL_RESPONSE",
        )
    )
    generator = _ScriptedGenerator([filtered, "must-not-run"])
    runner, _ = _make_runner(generator, max_attempts=3)
    original = _request(104)

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(original)
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.TERMINAL_FAILURE
    assert (
        error.generation_failure_disposition
        is GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
    )
    assert [_provider_payload_view(item) for item in generator.requests] == [original]
    attempt = error.telemetry.attempts[0]
    assert attempt.classification.reason is RetryReason.PERMANENT
    assert attempt.request_evidence.variant is AttemptRequestVariant.ORIGINAL
    assert (
        attempt.request_evidence.prompt_sha256
        == hashlib.sha256(original.prompt.encode("utf-8")).hexdigest()
    )
    assert "STRUCTURED_OUTPUT_SCHEMA_REPAIR" not in generator.requests[0].prompt
    assert "RAW_SECRET" not in repr(error.outcome)


def test_wrapped_api_connection_retries_with_exact_original_request() -> None:
    connection = APIConnectionError(
        message="RAW_SECRET_CONNECTION_MESSAGE",
        request=httpx.Request("POST", "https://example.invalid"),
    )
    wrapped = ModelAPIError("model", "RAW_SECRET_WRAPPER_MESSAGE")
    wrapped.__cause__ = connection
    translated = classify_generation_exception(wrapped)
    generator = _ScriptedGenerator([translated, "response"])
    observed = []
    runner, runtime = _make_runner(
        generator,
        max_attempts=2,
        outcome_sink=observed.append,
    )
    original = _request(105)

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(original)
        finally:
            await runner.aclose()

    result = asyncio.run(scenario())

    assert result.attempt_count == 2
    assert runtime.sleep_calls == [0]
    assert [_provider_payload_view(item) for item in generator.requests] == [
        original,
        original,
    ]
    assert len({item.provider_attempt_id for item in generator.requests}) == 2
    evidence = [attempt.request_evidence for attempt in observed[0].telemetry.attempts]
    assert [item.variant for item in evidence] == [
        AttemptRequestVariant.ORIGINAL,
        AttemptRequestVariant.ORIGINAL,
    ]
    assert {item.prompt_sha256 for item in evidence} == {
        hashlib.sha256(original.prompt.encode("utf-8")).hexdigest()
    }


def test_real_stream_adapter_and_queue_retry_typed_pre_stream_read_failure() -> None:
    """Exercise transport classification at the actual Agent/adapter/queue seam."""

    class RetryThenSucceedModel(TestModel):
        def __init__(self) -> None:
            super().__init__(
                custom_output_args=json.dumps({"answer": 17}),
                model_name="offline-pre-stream-retry",
            )
            self.stream_attempts = 0

        @asynccontextmanager
        async def request_stream(self, *args, **kwargs):
            self.stream_attempts += 1
            if self.stream_attempts < 3:
                raise _pre_stream_read_failure()
            async with super().request_stream(*args, **kwargs) as stream:
                yield stream

    model = RetryThenSucceedModel()
    generator = PydanticAIStructuredGenerator(
        agent=Agent(model, retries=0),
        requested_model="offline/pre-stream-retry",
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=NANOSECONDS_PER_SECOND,
            idle_timeout_ns=NANOSECONDS_PER_SECOND,
        ),
    )
    observed = []
    runner, runtime = _make_runner(
        generator,
        max_attempts=3,
        base_backoff_ns=4,
        max_backoff_ns=8,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
        outcome_sink=observed.append,
    )

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(_request(106))
        finally:
            await runner.aclose()

    result = asyncio.run(scenario())

    assert result.response.value == _Output(answer=17)
    assert result.attempt_count == 3
    assert model.stream_attempts == 3
    assert runtime.sleep_calls == [4, 8]
    attempts = observed[0].telemetry.attempts
    assert [attempt.status for attempt in attempts] == [
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.SUCCEEDED,
    ]
    for attempt in attempts[:2]:
        assert attempt.classification is not None
        assert attempt.classification.reason is RetryReason.TRANSIENT
        assert attempt.classification.sanitized_failure is not None
        assert (
            attempt.classification.sanitized_failure.kind
            == GenerationFailureKind.PROVIDER_UNAVAILABLE.value
        )
        assert attempt.classification.sanitized_failure.retryable is True
    assert "RAW_SECRET" not in repr(observed)


def test_concurrent_stage_drains_real_pydantic_pre_stream_handoff_tasks() -> None:
    """Regress the Pydantic 1.91 pre-handoff cancellation orphan race."""

    class AdversarialConcurrentModel(TestModel):
        def __init__(self) -> None:
            super().__init__(
                custom_output_args=json.dumps({"answer": 17}),
                model_name="offline-pre-stream-race",
            )
            self.sibling_entered = asyncio.Event()
            self.release_sibling = asyncio.Event()
            self.sibling_cancelled = False
            self.active_stream_requests = 0
            self.stream_attempts: dict[str, int] = {}

        @staticmethod
        def _prompt(messages: object) -> str:
            if type(messages) is not list:
                raise AssertionError("Pydantic model messages escaped exact list")
            for message in reversed(messages):
                if isinstance(message, ModelRequest):
                    for part in reversed(message.parts):
                        if isinstance(part, UserPromptPart) and type(part.content) is str:
                            return part.content
            raise AssertionError("no exact string user prompt reached the model")

        @asynccontextmanager
        async def request_stream(self, messages, *args, **kwargs):
            prompt = self._prompt(messages)
            self.stream_attempts[prompt] = self.stream_attempts.get(prompt, 0) + 1
            self.active_stream_requests += 1
            try:
                if prompt == _request(108).prompt:
                    self.sibling_entered.set()
                    try:
                        await self.release_sibling.wait()
                    except asyncio.CancelledError:
                        self.sibling_cancelled = True
                        # A real provider adapter translates the transport
                        # cause before the Pydantic wrap task terminates.
                        raise _pre_stream_read_failure()
                    raise _pre_stream_read_failure()
                if prompt == _request(107).prompt:
                    await self.sibling_entered.wait()
                    raise RuntimeError("RAW_SECRET_LOCAL_PROGRAMMING_FAILURE")
                raise AssertionError("unexpected adversarial test prompt")
                yield  # pragma: no cover - required by asynccontextmanager.
            finally:
                self.active_stream_requests -= 1

    async def scenario() -> tuple[list[object], list[tuple[object, object]], object]:
        model = AdversarialConcurrentModel()
        generator = PydanticAIStructuredGenerator(
            agent=Agent(model, retries=0),
            requested_model="offline/pre-stream-race",
            stream_liveness_policy=StructuredStreamLivenessPolicy(
                first_event_timeout_ns=NANOSECONDS_PER_SECOND,
                idle_timeout_ns=NANOSECONDS_PER_SECOND,
            ),
        )
        observed: list[object] = []
        runner, _ = _make_runner(
            generator,
            max_in_flight=2,
            max_attempts=3,
            attempt_request_policy=ExactPayloadAttemptPolicy(),
            retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
            outcome_sink=observed.append,
        )
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled: list[tuple[object, object]] = []

        def capture_unhandled(_loop, context) -> None:
            exception = context.get("exception")
            unhandled.append(
                (
                    context.get("message"),
                    None if exception is None else type(exception).__name__,
                )
            )

        loop.set_exception_handler(capture_unhandled)
        try:
            with pytest.raises(QueuedStructuredGenerationError):
                await gather_concurrent_stage(
                    (runner(_request(107)), runner(_request(108)))
                )
            # On the repaired framework the pre-stream wrap task has already
            # received cancellation and reached a retrieved terminal state.
            assert model.sibling_cancelled is True
            model.release_sibling.set()
            for _ in range(3):
                await asyncio.sleep(0)
                gc.collect()
            assert model.active_stream_requests == 0
            snapshot = await runner.snapshot()
            assert snapshot.in_flight == 0
            assert snapshot.pending == 0
        finally:
            model.release_sibling.set()
            await runner.aclose()
            loop.set_exception_handler(previous_handler)
        return observed, unhandled, model

    observed, unhandled, model = asyncio.run(scenario())

    by_task = {outcome.telemetry.task_id: outcome for outcome in observed}
    primary = by_task[_request(107).call_id.value]
    sibling = by_task[_request(108).call_id.value]
    assert primary.status is TaskOutcomeStatus.TERMINAL_FAILURE
    assert len(primary.telemetry.attempts) == 1
    primary_classification = primary.telemetry.attempts[0].classification
    assert primary_classification is not None
    assert primary_classification.disposition is RetryDisposition.FAIL
    assert primary_classification.sanitized_failure is not None
    assert primary_classification.sanitized_failure.kind == "unknown"
    assert primary_classification.sanitized_failure.retryable is False
    assert sibling.status is TaskOutcomeStatus.CANCELLED
    assert len(sibling.telemetry.attempts) == 1
    assert model.stream_attempts == {
        _request(107).prompt: 1,
        _request(108).prompt: 1,
    }
    assert unhandled == []
    assert "RAW_SECRET" not in repr(observed)


def test_detailed_validation_failure_remains_inside_queue_retry_attempts() -> None:
    class _DetailedCandidate(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        width: int

    output_type = _candidate_proposal_type(_DetailedCandidate, "typed_mutation")
    assert "$defs" not in output_type.model_json_schema()
    generator = PydanticAIStructuredGenerator(
        agent=Agent(
            TestModel(
                custom_output_args={
                    "configuration": {"width": "not-an-integer"},
                    "design_rationale": "wire-valid but locally invalid",
                }
            ),
            retries=0,
        ),
        requested_model="offline/invalid-detailed-output",
    )
    runner, runtime = _make_runner(generator, max_attempts=2)
    request = StructuredGenerationRequest(
        call_id=LLMCallId("call_queued_detailed_invalid"),
        operation="typed_mutation",
        prompt="Return the typed mutation.",
        output_type=output_type,
        output_tool_name="return_candidate_proposal",
        max_output_tokens=128,
        temperature=0.0,
    )

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(request)
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.ATTEMPTS_EXHAUSTED
    assert len(error.telemetry.attempts) == 2
    assert runtime.sleep_calls == [0]
    assert all(
        attempt.classification.reason is RetryReason.OUTPUT_INVALID
        and attempt.status is AttemptStatus.RETRYABLE_FAILURE
        for attempt in error.telemetry.attempts
    )
    failures = [
        attempt.classification.sanitized_failure for attempt in error.telemetry.attempts
    ]
    assert all(
        failure.output_failure_mode is StructuredOutputFailureMode.SCHEMA_VALIDATION
        for failure in failures
    )
    assert all(failure.validation_issues for failure in failures)


def test_retry_after_and_backoff_use_the_queue_max_rule_in_both_directions() -> None:
    generator = _ScriptedGenerator(
        [
            _error(
                GenerationFailureKind.RATE_LIMITED,
                retryable=True,
                retry_after_seconds=9,
            ),
            _error(
                GenerationFailureKind.PROVIDER_UNAVAILABLE,
                retryable=True,
                retry_after_seconds=1,
            ),
            "response",
        ]
    )
    runner, runtime = _make_runner(
        generator,
        base_backoff_ns=5 * NANOSECONDS_PER_SECOND,
        max_backoff_ns=20 * NANOSECONDS_PER_SECOND,
    )

    async def scenario() -> None:
        try:
            result = await runner(_request(2))
            assert result.attempt_count == 3
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    # Attempt one honors the longer Retry-After; attempt two honors the longer
    # exponential backoff. No other layer sleeps.
    assert runtime.sleep_calls == [
        9 * NANOSECONDS_PER_SECOND,
        10 * NANOSECONDS_PER_SECOND,
    ]


def test_terminal_capability_mismatch_is_not_retried_and_is_sanitized() -> None:
    observed = []
    generator = _ScriptedGenerator(
        [
            _error(
                GenerationFailureKind.CAPABILITY_MISMATCH,
                retryable=False,
                retry_after_seconds=99,
                message="capability mismatch TOP_SECRET_PROVIDER_TEXT",
            )
        ]
    )
    runner, runtime = _make_runner(generator, outcome_sink=observed.append)

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(_request(3))
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.TERMINAL_FAILURE
    assert (
        error.generation_failure_disposition
        is GenerationFailureDisposition.INFRASTRUCTURE_FAILURE
    )
    assert len(error.telemetry.attempts) == 1
    attempt = error.telemetry.attempts[0]
    assert attempt.classification == RetryClassification(
        RetryDisposition.FAIL,
        RetryReason.PERMANENT,
        sanitized_failure=_sanitized_failure(
            GenerationFailureKind.CAPABILITY_MISMATCH,
            retryable=False,
            retry_after_seconds=99,
            message="capability mismatch TOP_SECRET_PROVIDER_TEXT",
        ),
    )
    assert attempt.retry_after_ns == 0
    assert attempt.will_retry is False
    assert len(generator.requests) == 1
    assert runtime.sleep_calls == []
    assert observed == [error.outcome]
    assert "TOP_SECRET" not in str(error)
    assert "TOP_SECRET" not in repr(error)


def test_retryable_failures_exhaust_exact_budget_with_terminal_telemetry() -> None:
    generator = _ScriptedGenerator(
        [
            _error(
                GenerationFailureKind.PROVIDER_UNAVAILABLE,
                retryable=True,
                retry_after_seconds=2,
            )
            for _ in range(3)
        ]
    )
    runner, runtime = _make_runner(generator, max_attempts=3)

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(_request(4))
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.ATTEMPTS_EXHAUSTED
    assert len(generator.requests) == 3
    assert len(error.telemetry.attempts) == 3
    assert runtime.sleep_calls == [
        2 * NANOSECONDS_PER_SECOND,
        2 * NANOSECONDS_PER_SECOND,
    ]
    final_attempt = error.telemetry.attempts[-1]
    assert final_attempt.status is AttemptStatus.RETRYABLE_FAILURE
    assert final_attempt.will_retry is False
    assert final_attempt.retry_after_ns == 2 * NANOSECONDS_PER_SECOND
    assert final_attempt.scheduled_delay_ns == 0


def test_queue_timeout_is_classified_and_retried_by_the_queue() -> None:
    clock = FakeClock()
    runtime = _DeterministicRuntime(clock, timeout_calls=[1])
    generator = _ScriptedGenerator(["response", "response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=2,
        attempt_timeout_ns=77,
        runtime=runtime,
    )

    async def scenario() -> AttemptedStructuredGenerationResponse[_Output]:
        try:
            return await runner(_request(5))
        finally:
            await runner.aclose()

    result = asyncio.run(scenario())

    assert result.attempt_count == 2
    assert runtime.wait_calls == [77, 77]
    assert runtime.sleep_calls == [0]
    assert len(generator.requests) == 2


def test_transport_aborted_timeout_is_terminal_and_sanitized() -> None:
    classification = StructuredGenerationRetryClassifier().classify(
        TransportAbortedTimeoutError(),
        context=LLMAttemptContext("hard-timeout", 1, 10),
    )

    assert classification.disposition is RetryDisposition.FAIL
    assert classification.reason is RetryReason.TIMEOUT
    assert classification.retry_after is None
    assert classification.sanitized_failure == SanitizedAttemptFailure(
        kind="timeout",
        retryable=False,
        safe_message=(
            "provider attempt exceeded its hard deadline; the owned transport "
            "was closed and the attempt was drained"
        ),
    )


def test_classifier_maps_retry_after_with_ceiling_and_ignores_it_when_terminal() -> (
    None
):
    classifier = StructuredGenerationRetryClassifier()
    context = LLMAttemptContext("classifier", 1, 10)

    retryable = classifier.classify(
        _error(
            GenerationFailureKind.RATE_LIMITED,
            retryable=True,
            retry_after_seconds=0.0000000001,
        ),
        context=context,
    )
    assert retryable == RetryClassification(
        RetryDisposition.RETRY,
        RetryReason.RATE_LIMIT,
        RetryAfter(1, RetryAfterSource.DELAY_SECONDS),
        _sanitized_failure(
            GenerationFailureKind.RATE_LIMITED,
            retryable=True,
            retry_after_seconds=0.0000000001,
        ),
    )

    terminal = classifier.classify(
        _error(
            GenerationFailureKind.AUTHENTICATION,
            retryable=False,
            retry_after_seconds=12,
        ),
        context=context,
    )
    assert terminal == RetryClassification(
        RetryDisposition.FAIL,
        RetryReason.PERMANENT,
        sanitized_failure=_sanitized_failure(
            GenerationFailureKind.AUTHENTICATION,
            retryable=False,
            retry_after_seconds=12,
        ),
    )
    assert classifier.classify(TimeoutError(), context=context) == RetryClassification(
        RetryDisposition.RETRY,
        RetryReason.TIMEOUT,
    )
    assert classifier.classify(RuntimeError("raw secret"), context=context) == (
        RetryClassification(RetryDisposition.FAIL, RetryReason.INTERNAL)
    )


def test_classifier_distinguishes_output_invalid_from_provider_unavailable() -> None:
    classifier = StructuredGenerationRetryClassifier()
    context = LLMAttemptContext("typed-output", 1, 10)

    assert classifier.classify(
        _error(GenerationFailureKind.OUTPUT_INVALID, retryable=True),
        context=context,
    ) == RetryClassification(
        RetryDisposition.RETRY,
        RetryReason.OUTPUT_INVALID,
        sanitized_failure=_sanitized_failure(
            GenerationFailureKind.OUTPUT_INVALID,
            retryable=True,
        ),
    )
    assert classifier.classify(
        _error(GenerationFailureKind.PROVIDER_UNAVAILABLE, retryable=True),
        context=context,
    ) == RetryClassification(
        RetryDisposition.RETRY,
        RetryReason.TRANSIENT,
        sanitized_failure=_sanitized_failure(
            GenerationFailureKind.PROVIDER_UNAVAILABLE,
            retryable=True,
        ),
    )
    assert classifier.classify(
        _error(
            GenerationFailureKind.OUTPUT_INVALID,
            retryable=False,
            retry_after_seconds=12,
        ),
        context=context,
    ) == RetryClassification(
        RetryDisposition.FAIL,
        RetryReason.OUTPUT_INVALID,
        sanitized_failure=_sanitized_failure(
            GenerationFailureKind.OUTPUT_INVALID,
            retryable=False,
            retry_after_seconds=12,
        ),
    )
    assert RetryReason.OUTPUT_INVALID.value == "output_invalid"


@pytest.mark.parametrize("status", range(400, 600))
def test_transport_only_classifier_enforces_exact_http_retry_matrix(
    status: int,
) -> None:
    classifier = TransportOnlyStructuredGenerationRetryClassifier()
    context = LLMAttemptContext("http-matrix", 1, 2)
    expected = (
        RetryDisposition.RETRY
        if status in {408, 429} or 500 <= status <= 599
        else RetryDisposition.FAIL
    )

    # Deliberately mislabel every status as each transient kind. In
    # particular, no other 4xx can obtain a retry merely from kind/retryable.
    for kind in (
        GenerationFailureKind.RATE_LIMITED,
        GenerationFailureKind.TIMEOUT,
        GenerationFailureKind.PROVIDER_UNAVAILABLE,
    ):
        classification = classifier.classify(
            _error(kind, retryable=True, status_code=status),
            context=context,
        )
        assert classification.disposition is expected, (status, kind)


def test_transport_only_classifier_rejects_statusless_rate_limit_label() -> None:
    classification = TransportOnlyStructuredGenerationRetryClassifier().classify(
        _error(GenerationFailureKind.RATE_LIMITED, retryable=True),
        context=LLMAttemptContext("statusless-rate-limit", 1, 2),
    )

    assert classification.disposition is RetryDisposition.FAIL
    assert classification.reason is RetryReason.RATE_LIMIT


@pytest.mark.parametrize(
    "phase",
    [StructuredStreamTimeoutPhase.FIRST_EVENT, StructuredStreamTimeoutPhase.IDLE],
)
def test_non_repeating_stream_classifier_never_retries_owned_stream_timeout(
    phase: StructuredStreamTimeoutPhase,
) -> None:
    classification = NonRepeatingStreamTransportRetryClassifier().classify(
        StructuredStreamTimeoutError(phase),
        context=LLMAttemptContext("owned-stream", 1, 3),
    )

    assert classification.disposition is RetryDisposition.FAIL
    assert classification.reason is RetryReason.TIMEOUT
    assert classification.sanitized_failure is not None
    assert classification.sanitized_failure.stream_timeout_phase is phase


def _opaque_http_400() -> StructuredGenerationError:
    return StructuredGenerationError(
        kind=GenerationFailureKind.INVALID_REQUEST,
        retryable=False,
        safe_message="provider rejected invalid request parameters",
        status_code=400,
        provider_error_envelope_sha256="3" * 64,
    )


def test_opaque_http_400_policy_retries_only_the_first_opaque_failure() -> None:
    classifier = OpaqueHTTP400OnceRetryClassifier()
    first = classifier.classify(
        _opaque_http_400(),
        context=LLMAttemptContext("opaque-400", 1, 3),
    )
    second = classifier.classify(
        _opaque_http_400(),
        context=LLMAttemptContext(
            "opaque-400",
            2,
            3,
            previous_failure=first.sanitized_failure,
        ),
    )

    assert first.disposition is RetryDisposition.RETRY
    assert first.reason is RetryReason.TRANSIENT
    assert first.sanitized_failure is not None
    assert first.sanitized_failure.retryable is False
    assert second.disposition is RetryDisposition.FAIL
    assert second.reason is RetryReason.PERMANENT


def test_opaque_http_400_policy_keeps_typed_or_unfingerprinted_400_terminal() -> None:
    classifier = OpaqueHTTP400OnceRetryClassifier()
    typed = StructuredGenerationError(
        kind=GenerationFailureKind.INVALID_REQUEST,
        retryable=False,
        safe_message="provider rejected invalid request parameters",
        status_code=400,
        provider_error_code=CanonicalProviderErrorCode.INVALID_REQUEST,
        provider_error_envelope_sha256="4" * 64,
    )
    unfingerprinted = StructuredGenerationError(
        kind=GenerationFailureKind.INVALID_REQUEST,
        retryable=False,
        safe_message="provider rejected invalid request parameters",
        status_code=400,
    )

    for error in (typed, unfingerprinted):
        result = classifier.classify(
            error,
            context=LLMAttemptContext("terminal-400", 1, 3),
        )
        assert result.disposition is RetryDisposition.FAIL
        assert result.reason is RetryReason.PERMANENT


@pytest.mark.parametrize("terminal", [False, True])
def test_outcome_sink_failure_is_isolated_from_generation_semantics(
    terminal: bool,
) -> None:
    def failing_sink(_outcome) -> None:
        raise RuntimeError("trace writer unavailable")

    script = (
        [_error(GenerationFailureKind.INVALID_REQUEST, retryable=False)]
        if terminal
        else ["response"]
    )
    runner, _ = _make_runner(
        _ScriptedGenerator(script),
        max_attempts=1,
        outcome_sink=failing_sink,
    )

    async def scenario():
        try:
            if terminal:
                with pytest.raises(QueuedStructuredGenerationError) as caught:
                    await runner(_request(6))
                return caught.value
            return await runner(_request(6))
        finally:
            await runner.aclose()

    result = asyncio.run(scenario())

    if terminal:
        assert result.status is TaskOutcomeStatus.TERMINAL_FAILURE
    else:
        assert result.response.value == _Output(answer=6)


def test_required_outcome_publication_precedes_downstream_response_access() -> None:
    order: list[str] = []

    def sink(outcome) -> None:
        order.append("published")
        assert outcome.status is TaskOutcomeStatus.SUCCEEDED
        assert outcome.response.cost_usd == Decimal("0.001")

    generator = _ScriptedGenerator(["response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=1,
        outcome_sink=sink,
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
    )

    async def scenario() -> None:
        try:
            result = await runner(_request(7))
            order.append("downstream")
            assert result.response.value == _Output(answer=7)
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    assert order == ["published", "downstream"]
    assert len(generator.requests) == 1


def test_required_outcome_publication_fails_closed_without_provider_retry() -> None:
    def failing_sink(_outcome) -> None:
        raise OSError("durable writer unavailable TOP_SECRET_PATH")

    generator = _ScriptedGenerator(["response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        outcome_sink=failing_sink,
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
    )

    async def scenario() -> OutcomePublicationError:
        try:
            with pytest.raises(OutcomePublicationError) as caught:
                await runner(_request(8))
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.outcome.status is TaskOutcomeStatus.SUCCEEDED
    assert error.status is TaskOutcomeStatus.SUCCEEDED
    assert error.telemetry is error.outcome.telemetry
    assert error.outcome.response.value == _Output(answer=8)
    assert len(generator.requests) == 1
    assert "TOP_SECRET" not in str(error)
    assert error.__cause__ is None


def test_required_outcome_publication_requires_a_sink() -> None:
    with pytest.raises(ValueError, match="needs an outcome_sink"):
        _make_runner(
            _ScriptedGenerator([]),
            outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        )


def test_required_content_evidence_binds_wire_prompt_schema_and_typed_output() -> None:
    request_records: list[dict[str, object]] = []
    output_records: list[dict[str, object]] = []
    generator = _ContractAwareGenerator()
    runner, _ = _make_runner(
        generator,
        max_attempts=1,
        request_evidence_sink=request_records.append,
        output_evidence_sink=output_records.append,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    adapter = PydanticAIAgenticGenerator(runner)
    proposal_prompt = "Semantic proposal prompt remains byte-exact."
    reflection_prompt = "Semantic reflection prompt is rendered at the wire boundary."

    async def scenario() -> None:
        try:
            proposal = await adapter.propose(
                VariationGenerationRequest(
                    call_id=LLMCallId("call_evidence_proposal"),
                    operation="mutation",
                    prompt=proposal_prompt,
                    candidate_model=_Output,
                    max_output_tokens=64,
                    temperature=0.0,
                )
            )
            assert proposal.draft.configuration == {"answer": 17}
            reflection = await adapter.reflect(
                ReflectionGenerationRequest(
                    call_id=LLMCallId("call_evidence_reflection"),
                    operation="extract_insights",
                    prompt=reflection_prompt,
                    min_insights=0,
                    max_insights=1,
                    max_output_tokens=64,
                    temperature=0.0,
                )
            )
            assert reflection.insights == ()
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    assert len(generator.requests) == len(request_records) == len(output_records) == 2
    proposal_record, reflection_record = request_records
    assert proposal_record["call_id"] == "call_evidence_proposal"
    assert (
        proposal_record["prompt_sha256"]
        == hashlib.sha256(proposal_prompt.encode("utf-8")).hexdigest()
    )
    assert proposal_record["wire_prompt_sha256"] == proposal_record["prompt_sha256"]
    assert (
        proposal_record["semantic_prompt_sha256"]
        == proposal_record["wire_prompt_sha256"]
    )
    assert proposal_record["prompt_renderer_id"] == IDENTITY_PROMPT_RENDERER_ID
    assert (
        proposal_record["prompt_renderer_revision"] == IDENTITY_PROMPT_RENDERER_REVISION
    )
    assert (
        proposal_record["prompt_renderer_definition_sha256"]
        == IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256
    )
    rendered_reflection = render_reflection_prompt(reflection_prompt)
    assert rendered_reflection != reflection_prompt
    assert reflection_record["call_id"] == "call_evidence_reflection"
    assert (
        reflection_record["prompt_sha256"]
        == hashlib.sha256(rendered_reflection.encode("utf-8")).hexdigest()
    )
    assert (
        reflection_record["prompt_sha256"]
        != hashlib.sha256(reflection_prompt.encode("utf-8")).hexdigest()
    )
    assert reflection_record["wire_prompt_sha256"] == reflection_record["prompt_sha256"]
    assert (
        reflection_record["semantic_prompt_sha256"]
        == hashlib.sha256(reflection_prompt.encode("utf-8")).hexdigest()
    )
    assert reflection_record["prompt_renderer_id"] == (REFLECTION_PROMPT_RENDERER_ID)
    assert reflection_record["prompt_renderer_revision"] == (
        REFLECTION_PROMPT_RENDERER_REVISION
    )
    assert reflection_record["prompt_renderer_definition_sha256"] == (
        REFLECTION_PROMPT_RENDERER_DEFINITION_SHA256
    )
    for request_record, output_record in zip(
        request_records,
        output_records,
        strict=True,
    ):
        assert (
            output_record["request_evidence_sha256"]
            == request_record["request_evidence_sha256"]
        )
        assert (
            output_record["output_schema_sha256"]
            == request_record["output_schema_sha256"]
        )
        assert output_record["provider_response_id"] == (
            f"response-{request_record['call_id']}"
        )
        assert len(str(output_record["typed_output_sha256"])) == 64
        assert (
            validate_structured_generation_request_evidence_record(request_record)
            == request_record
        )
        assert (
            validate_structured_generation_output_evidence_record(
                output_record,
                request_evidence=request_record,
            )
            == output_record
        )

    tampered_output = dict(output_records[0])
    tampered_output["typed_output"] = {"configuration": {"answer": 99}}
    with pytest.raises(ValueError, match="does not authenticate the output"):
        validate_structured_generation_output_evidence_record(tampered_output)
    with pytest.raises(ValueError, match="does not join"):
        validate_structured_generation_output_evidence_record(
            output_records[0],
            request_evidence=request_records[1],
        )


def test_wrong_card_typed_output_is_durable_before_downstream_rejection() -> None:
    request_records: list[dict[str, object]] = []
    output_records: list[dict[str, object]] = []
    generator = _ContractAwareGenerator()
    runner, _ = _make_runner(
        generator,
        max_attempts=1,
        request_evidence_sink=request_records.append,
        output_evidence_sink=output_records.append,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    request = StructuredGenerationRequest(
        call_id=LLMCallId("call_wrong_card"),
        operation="extract_insights",
        prompt="Select one globally allowed card.",
        output_type=_CardOutput,
        output_tool_name="return_card",
        max_output_tokens=64,
        temperature=0.0,
    )

    async def scenario() -> None:
        try:
            response = await runner(request)
            with pytest.raises(ValueError, match="source action"):
                if response.response.value.option_id != "trim.executed_source_card":
                    raise ValueError("card differs from the exact source action")
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    assert len(request_records) == len(output_records) == 1
    assert output_records[0]["typed_output"] == {"option_id": "trim.other_allowed_card"}
    canonical = json.dumps(
        output_records[0]["typed_output"],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert (
        output_records[0]["typed_output_sha256"]
        == hashlib.sha256(canonical).hexdigest()
    )


def test_typed_output_evidence_bypasses_hostile_model_dump_override() -> None:
    output_records: list[dict[str, object]] = []
    runner, _ = _make_runner(
        _ContractAwareGenerator(),
        max_attempts=1,
        request_evidence_sink=lambda _record: None,
        output_evidence_sink=output_records.append,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    request = StructuredGenerationRequest(
        call_id=LLMCallId("call_hostile_model_dump"),
        operation="typed_evidence_test",
        prompt="Return one hostile typed output.",
        output_type=_HostileModelDumpOutput,
        output_tool_name="return_hostile_output",
        max_output_tokens=64,
        temperature=0.0,
    )

    async def scenario() -> None:
        try:
            response = await runner(request)
            assert response.response.value.answer == 23
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    assert output_records[0]["typed_output"] == {"answer": 23}


@pytest.mark.parametrize(
    ("failure_stage", "expected_provider_calls"),
    (
        (StructuredEvidencePublicationStage.REQUEST, 0),
        (StructuredEvidencePublicationStage.OUTPUT, 1),
    ),
)
def test_required_content_evidence_sink_failure_is_fail_closed_without_retry(
    failure_stage: StructuredEvidencePublicationStage,
    expected_provider_calls: int,
) -> None:
    request_records: list[dict[str, object]] = []

    def failing_sink(_record: dict[str, object]) -> None:
        raise OSError("durable evidence writer unavailable SECRET_PATH")

    generator = _ScriptedGenerator(["response"])
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        request_evidence_sink=(
            failing_sink
            if failure_stage is StructuredEvidencePublicationStage.REQUEST
            else request_records.append
        ),
        output_evidence_sink=(
            failing_sink
            if failure_stage is StructuredEvidencePublicationStage.OUTPUT
            else lambda _record: None
        ),
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )

    async def scenario() -> StructuredEvidencePublicationError:
        try:
            with pytest.raises(StructuredEvidencePublicationError) as caught:
                await runner(_request(81))
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.stage is failure_stage
    assert error.call_id == "call_queued_0081"
    assert len(generator.requests) == expected_provider_calls
    assert "SECRET_PATH" not in str(error)
    assert error.__cause__ is None
    if failure_stage is StructuredEvidencePublicationStage.OUTPUT:
        assert error.outcome is not None
        assert error.outcome.status is TaskOutcomeStatus.SUCCEEDED
        assert len(request_records) == 1
    else:
        assert error.outcome is None
        assert request_records == []


def test_required_content_evidence_requires_both_sinks() -> None:
    with pytest.raises(ValueError, match="needs both sinks"):
        _make_runner(
            _ScriptedGenerator([]),
            request_evidence_sink=lambda _record: None,
            evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
        )


def test_structured_outcome_record_retains_response_telemetry_not_content() -> None:
    observed = []
    generator = _ScriptedGenerator(
        [
            _error(
                GenerationFailureKind.OUTPUT_INVALID,
                retryable=True,
                message="model output violated the typed response contract",
                output_failure_mode=StructuredOutputFailureMode.SCHEMA_VALIDATION,
                validation_issues=(
                    SanitizedValidationIssue(
                        ValidationIssueCategory.SEMANTIC_CONSTRAINT,
                        ("root",),
                        ValidationIssueReasonCode.NO_FEASIBLE_DISJOINT_PORTFOLIO,
                    ),
                ),
            ),
            "response",
        ]
    )
    runner, _ = _make_runner(
        generator,
        max_attempts=2,
        outcome_sink=lambda outcome: observed.append(
            structured_generation_outcome_record(outcome)
        ),
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
    )

    async def scenario() -> None:
        try:
            await runner(_request(9))
        finally:
            await runner.aclose()

    asyncio.run(scenario())

    assert len(observed) == 1
    record = observed[0]
    assert record["schema_version"] == STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION
    assert record["task_id"] == "call_queued_0009"
    assert record["status"] == "succeeded"
    assert [attempt["request_evidence"] for attempt in record["attempts"]] == [
        {
            "variant": "original",
            "prompt_sha256": hashlib.sha256(
                generator.requests[0].prompt.encode("utf-8")
            ).hexdigest(),
            "provider_attempt_id": (generator.requests[0].provider_attempt_id.value),
        },
        {
            "variant": "schema_repair_v3",
            "prompt_sha256": hashlib.sha256(
                generator.requests[1].prompt.encode("utf-8")
            ).hexdigest(),
            "provider_attempt_id": (generator.requests[1].provider_attempt_id.value),
        },
    ]
    assert [attempt["failure"] for attempt in record["attempts"]] == [
        {
            "kind": "output_invalid",
            "retryable": True,
            "safe_message": "model output violated the typed response contract",
            "status_code": None,
            "retry_after_seconds": None,
            "provider_error_code": None,
            "provider_error_envelope_sha256": None,
            "exception_provenance": None,
            "stream_timeout_phase": None,
            "output_failure_mode": "schema_validation",
            "validation_issues": [
                {
                    "category": "semantic_constraint",
                    "location": ["root"],
                    "reason_code": "no_feasible_disjoint_portfolio",
                }
            ],
        },
        None,
    ]
    assert record["response"] == {
        "requested_model": "requested/model",
        "resolved_model": "resolved/model",
        "resolved_provider": "provider",
        "provider_response_id": "response-9",
        "finish_reason": "stop",
        "input_tokens": 10,
        "output_tokens": 2,
        "reasoning_tokens": 1,
        "cache_read_tokens": 3,
        "cache_write_tokens": 0,
        "cost_usd": "0.001",
        "latency_ns": 50,
    }
    serialized = repr(record)
    assert "Return answer" not in serialized
    assert "answer=9" not in serialized


def test_unknown_exception_provenance_survives_strict_journal_round_trip() -> None:
    secret = "OPENROUTER_API_KEY=sk-journal-secret https://journal.example/body"
    cause = LookupError(secret)
    raw = RuntimeError(secret)
    raw.__cause__ = cause
    classified = classify_generation_exception(raw)
    assert classified.kind is GenerationFailureKind.UNKNOWN
    assert classified.retryable is False
    assert classified.exception_provenance is not None
    observed: list[dict[str, object]] = []
    generator = _ScriptedGenerator([classified])
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        outcome_sink=lambda outcome: observed.append(
            structured_generation_outcome_record(outcome)
        ),
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
    )

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(_request(91))
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.TERMINAL_FAILURE
    assert len(error.telemetry.attempts) == 1
    assert error.telemetry.attempts[0].will_retry is False
    record = observed[0]
    assert record["schema_version"] == 8
    assert validate_structured_generation_outcome_record(record) == record
    provenance = record["attempts"][0]["failure"]["exception_provenance"]
    assert provenance["truncated"] is False
    assert [node["link"] for node in provenance["nodes"]] == ["root", "cause"]
    retained = json.dumps(record, allow_nan=False, sort_keys=True)
    for forbidden in (secret, "sk-journal-secret", "journal.example", "body"):
        assert forbidden not in retained


def test_structured_outcome_record_traces_all_failures_and_sanitizes_404() -> None:
    raw_capability_error = ModelHTTPError(
        404,
        "deepseek/deepseek-v4-pro",
        {
            "error": {
                "message": (
                    "No endpoints found that can handle requested parameters "
                    "RAW_PROVIDER_BODY_OPENROUTER_API_KEY=sk-secret"
                ),
                "metadata": {
                    "error_type": "invalid_request_error",
                    "raw": "RAW_PROVIDER_UPSTREAM_BODY_sk-secret",
                },
            }
        },
    )
    capability_error = classify_generation_exception(raw_capability_error)
    observed: list[dict[str, object]] = []
    generator = _ScriptedGenerator(
        [
            _error(
                GenerationFailureKind.RATE_LIMITED,
                retryable=True,
                retry_after_seconds=0.25,
                status_code=429,
                message="provider rate limit",
            ),
            capability_error,
        ]
    )
    runner, _ = _make_runner(
        generator,
        max_attempts=3,
        outcome_sink=lambda outcome: observed.append(
            structured_generation_outcome_record(outcome)
        ),
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
    )

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(_request(10))
            return caught.value
        finally:
            await runner.aclose()

    error = asyncio.run(scenario())

    assert error.status is TaskOutcomeStatus.TERMINAL_FAILURE
    assert len(observed) == 1
    record = observed[0]
    assert record["schema_version"] == STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION
    assert record["status"] == "terminal_failure"
    assert record["response"] is None
    assert [attempt["attempt_number"] for attempt in record["attempts"]] == [1, 2]
    original_prompt_hash = hashlib.sha256(
        _request(10).prompt.encode("utf-8")
    ).hexdigest()
    assert [attempt["request_evidence"] for attempt in record["attempts"]] == [
        {
            "variant": "original",
            "prompt_sha256": original_prompt_hash,
            "provider_attempt_id": generator.requests[0].provider_attempt_id.value,
        },
        {
            "variant": "original",
            "prompt_sha256": original_prompt_hash,
            "provider_attempt_id": generator.requests[1].provider_attempt_id.value,
        },
    ]
    assert [attempt["failure"] for attempt in record["attempts"]] == [
        {
            "kind": "rate_limited",
            "retryable": True,
            "safe_message": "provider rate limit",
            "status_code": 429,
            "retry_after_seconds": 0.25,
                "provider_error_code": None,
                "provider_error_envelope_sha256": None,
                "exception_provenance": None,
                "stream_timeout_phase": None,
            "output_failure_mode": None,
            "validation_issues": [],
        },
        {
            "kind": "invalid_request",
            "retryable": False,
            "safe_message": "provider rejected invalid request parameters",
            "status_code": 404,
            "retry_after_seconds": None,
            "provider_error_code": "invalid_request_error",
                "provider_error_envelope_sha256": (
                    capability_error.provider_error_envelope_sha256
                ),
                "exception_provenance": None,
                "stream_timeout_phase": None,
            "output_failure_mode": None,
            "validation_issues": [],
        },
    ]
    serialized = json.dumps(record, sort_keys=True)
    assert "RAW_PROVIDER_BODY" not in serialized
    assert "sk-secret" not in serialized
    assert "Return answer 10" not in serialized


class _GateGenerator:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.releases: dict[str, asyncio.Event] = {}
        self.cancelled: list[str] = []
        self.active = 0
        self.max_active = 0

    def release(self, call_id: str) -> None:
        self.releases.setdefault(call_id, asyncio.Event()).set()

    async def generate_once(self, request: StructuredGenerationRequest[_Output]):
        call_id = request.call_id.value
        self.started.append(call_id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.releases.setdefault(call_id, asyncio.Event()).wait()
            return _response(request)
        except asyncio.CancelledError:
            self.cancelled.append(call_id)
            raise
        finally:
            self.active -= 1


async def _eventually(predicate, *, turns: int = 100) -> None:
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


async def _eventually_snapshot(
    runner: QueuedStructuredGenerationRunner,
    *,
    in_flight: int,
    pending: int,
) -> None:
    for _ in range(100):
        snapshot = await runner.snapshot()
        if snapshot.in_flight == in_flight and snapshot.pending == pending:
            return
        await asyncio.sleep(0)
    raise AssertionError("queue did not reach expected occupancy")


def test_runner_preserves_hard_in_flight_and_pending_admission_bounds() -> None:
    async def scenario() -> None:
        generator = _GateGenerator()
        runner, _ = _make_runner(
            generator,
            max_in_flight=1,
            max_pending=1,
            max_attempts=1,
        )
        first = asyncio.create_task(runner(_request(10)))
        await _eventually(lambda: len(generator.started) == 1)
        second = asyncio.create_task(runner(_request(11)))
        await _eventually_snapshot(runner, in_flight=1, pending=1)

        with pytest.raises(LLMTaskQueueFullError):
            await runner(_request(12))

        generator.release("call_queued_0010")
        await _eventually(lambda: "call_queued_0011" in generator.started)
        generator.release("call_queued_0011")
        results = await asyncio.gather(first, second)
        assert [result.attempt_count for result in results] == [1, 1]
        assert generator.max_active == 1
        await runner.aclose()

    asyncio.run(scenario())


def test_close_cancels_active_and_pending_work_and_closes_admission() -> None:
    async def scenario() -> None:
        observed = []
        generator = _GateGenerator()
        runner, _ = _make_runner(
            generator,
            max_in_flight=1,
            max_pending=1,
            max_attempts=2,
            outcome_sink=observed.append,
        )
        active = asyncio.create_task(runner(_request(20)))
        await _eventually(lambda: len(generator.started) == 1)
        pending = asyncio.create_task(runner(_request(21)))
        await _eventually_snapshot(runner, in_flight=1, pending=1)

        await runner.aclose()
        results = await asyncio.gather(active, pending, return_exceptions=True)

        assert all(type(item) is QueuedStructuredGenerationError for item in results)
        outcomes = [item.outcome for item in results]
        assert all(
            outcome.status is TaskOutcomeStatus.CANCELLED for outcome in outcomes
        )
        assert all(
            outcome.cancellation_reason is CancellationReason.QUEUE_CLOSED
            for outcome in outcomes
        )
        assert sorted(len(outcome.telemetry.attempts) for outcome in outcomes) == [0, 1]
        assert generator.cancelled == ["call_queued_0020"]
        assert sorted(observed, key=lambda item: item.telemetry.task_id) == sorted(
            outcomes,
            key=lambda item: item.telemetry.task_id,
        )
        assert (await runner.snapshot()).closed is True
        with pytest.raises(LLMTaskQueueClosedError):
            await runner(_request(22))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "publication_policy",
    [OutcomePublicationPolicy.BEST_EFFORT, OutcomePublicationPolicy.REQUIRED],
)
def test_runner_publishes_submitter_cancelled_pending_and_active_outcomes_once(
    publication_policy: OutcomePublicationPolicy,
) -> None:
    async def scenario() -> None:
        observed = []
        generator = _GateGenerator()
        runner, _ = _make_runner(
            generator,
            max_in_flight=1,
            max_pending=1,
            max_attempts=1,
            outcome_sink=observed.append,
            outcome_publication_policy=publication_policy,
        )
        active = asyncio.create_task(runner(_request(30)))
        await _eventually(lambda: generator.started == ["call_queued_0030"])
        pending = asyncio.create_task(runner(_request(31)))
        await _eventually_snapshot(runner, in_flight=1, pending=1)

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert [item.telemetry.task_id for item in observed] == [
            "call_queued_0031"
        ]

        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert generator.cancelled == ["call_queued_0030"]
        by_id = {item.telemetry.task_id: item for item in observed}
        assert set(by_id) == {"call_queued_0030", "call_queued_0031"}
        assert all(
            item.status is TaskOutcomeStatus.CANCELLED for item in by_id.values()
        )
        assert all(
            item.cancellation_reason is CancellationReason.SUBMITTER_CANCELLED
            for item in by_id.values()
        )
        assert by_id["call_queued_0031"].telemetry.attempts == ()
        assert [
            attempt.status
            for attempt in by_id["call_queued_0030"].telemetry.attempts
        ] == [AttemptStatus.CANCELLED]

        await runner.aclose()
        assert len(observed) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize(
    ("publication_policy", "expected_error"),
    [
        (OutcomePublicationPolicy.BEST_EFFORT, asyncio.CancelledError),
        (
            OutcomePublicationPolicy.REQUIRED,
            CancelledOutcomePublicationError,
        ),
    ],
)
def test_cancelled_outcome_sink_failure_preserves_cancellation_semantics(
    pending: bool,
    publication_policy: OutcomePublicationPolicy,
    expected_error: type[BaseException],
) -> None:
    async def scenario() -> None:
        generator = _GateGenerator()
        target_id = "call_queued_0041" if pending else "call_queued_0040"
        target_sink_calls = 0

        def sink(outcome) -> None:
            nonlocal target_sink_calls
            if outcome.telemetry.task_id == target_id:
                target_sink_calls += 1
                raise OSError("durable writer unavailable SECRET_PATH")

        runner, _ = _make_runner(
            generator,
            max_in_flight=1,
            max_pending=1,
            max_attempts=1,
            outcome_sink=sink,
            outcome_publication_policy=publication_policy,
        )
        blocker = None
        if pending:
            blocker = asyncio.create_task(runner(_request(42)))
            await _eventually(lambda: generator.started == ["call_queued_0042"])
            target = asyncio.create_task(runner(_request(41)))
            await _eventually_snapshot(runner, in_flight=1, pending=1)
        else:
            target = asyncio.create_task(runner(_request(40)))
            await _eventually(lambda: generator.started == [target_id])

        target.cancel()
        with pytest.raises(expected_error) as caught:
            await target
        assert target.cancelled()
        assert target_sink_calls == 1
        if publication_policy is OutcomePublicationPolicy.REQUIRED:
            error = caught.value
            assert isinstance(error, CancelledOutcomePublicationError)
            assert error.status is TaskOutcomeStatus.CANCELLED
            assert (
                error.outcome.cancellation_reason
                is CancellationReason.SUBMITTER_CANCELLED
            )
            assert "SECRET_PATH" not in str(error)

        if blocker is not None:
            generator.release("call_queued_0042")
            await blocker
        await runner.aclose()
        assert target_sink_calls == 1

    asyncio.run(scenario())


class _NeverCalledAgent:
    async def run(self, *_args, **_kwargs):
        raise AssertionError("production factory test must not make a provider call")


class _CloseClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _AbortCloseClient:
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


class _CancellationBlockedAgent:
    """Simulate provider cleanup shielded until its transport is closed."""

    def __init__(self, client: _AbortCloseClient) -> None:
        self.client = client
        self.cancellation_seen = asyncio.Event()
        self.finished = asyncio.Event()
        self.run_calls = 0

    async def run(self, *_args, **_kwargs):
        self.run_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellation_seen.set()
            await self.client.closed.wait()
            raise
        finally:
            self.finished.set()


class _ZeroRandom:
    def randrange(self, stop: int) -> int:
        assert stop > 0
        return 0


def test_production_factory_uses_real_runtime_full_jitter_and_owned_close() -> None:
    client = _CloseClient()
    generator = PydanticAIStructuredGenerator(
        agent=_NeverCalledAgent(),
        requested_model="offline/factory-test",
        owned_openai_client=client,
    )
    observed = []
    attempt_policy = SchemaRepairAttemptPolicy()
    retry_classifier = TransportOnlyStructuredGenerationRetryClassifier()
    runner = create_production_queued_runner(
        generator=generator,
        max_in_flight=2,
        max_pending=3,
        max_attempts=4,
        attempt_timeout_ns=123,
        base_backoff_ns=7,
        max_backoff_ns=19,
        random_source=_ZeroRandom(),
        outcome_sink=observed.append,
        attempt_request_policy=attempt_policy,
        retry_classifier=retry_classifier,
    )

    async def scenario() -> None:
        async with runner as entered:
            assert entered is runner
            snapshot = await runner.snapshot()
            assert (snapshot.max_in_flight, snapshot.max_pending) == (2, 3)
            assert type(runner._queue._clock) is SystemClock
            assert type(runner._queue._runtime) is AsyncioRuntime
            assert type(runner._queue._backoff_policy) is ExponentialBackoff
            assert type(runner._queue._backoff_policy.jitter) is FullJitter
            assert runner._queue._backoff_policy.base_delay_ns == 7
            assert runner._queue._backoff_policy.max_delay_ns == 19
            assert runner._queue._backoff_policy.rate_limit_floor_ns == 0
            assert runner._queue._executor.attempt_request_policy is attempt_policy
            assert runner._queue._retry_classifier is retry_classifier
            assert runner.max_attempts == 4
        await runner.aclose()

    asyncio.run(scenario())

    assert client.close_calls == 1
    assert observed == []


def test_progress_aware_factory_forbids_a_competing_fixed_queue_cutoff() -> None:
    generator = PydanticAIStructuredGenerator(
        agent=_NeverCalledAgent(),
        requested_model="offline/progress-aware-factory-test",
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=10,
            idle_timeout_ns=20,
        ),
    )

    with pytest.raises(ValueError, match="attempt_timeout_ns=None"):
        create_production_queued_runner(generator=generator)

    runner = create_production_queued_runner(
        generator=generator,
        attempt_timeout_ns=None,
    )
    assert runner._queue._attempt_timeout_ns is None
    assert runner._queue._runtime._timeout_abort is None
    asyncio.run(runner.aclose())


def test_production_hard_timeout_closes_transport_publishes_and_leaves_no_attempt() -> (
    None
):
    async def scenario() -> None:
        client = _AbortCloseClient()
        agent = _CancellationBlockedAgent(client)
        generator = PydanticAIStructuredGenerator(
            agent=agent,
            requested_model="offline/hard-timeout-test",
            owned_openai_client=client,
        )
        observed = []
        runner = create_production_queued_runner(
            generator=generator,
            max_in_flight=1,
            max_pending=1,
            max_attempts=2,
            attempt_timeout_ns=1_000_000,
            base_backoff_ns=0,
            max_backoff_ns=0,
            random_source=_ZeroRandom(),
            outcome_sink=observed.append,
            outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        )

        async with runner:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(_request(30))
            assert (await runner.snapshot()).closed is True

        error = caught.value
        assert error.status is TaskOutcomeStatus.TERMINAL_FAILURE
        assert len(error.telemetry.attempts) == 1
        attempt = error.telemetry.attempts[0]
        assert attempt.status is AttemptStatus.TIMED_OUT
        assert attempt.error_type == "TransportAbortedTimeoutError"
        assert attempt.will_retry is False
        assert attempt.classification is not None
        assert attempt.classification.reason is RetryReason.TIMEOUT
        assert attempt.classification.disposition is RetryDisposition.FAIL
        assert attempt.classification.sanitized_failure is not None
        assert attempt.classification.sanitized_failure.kind == "timeout"
        assert attempt.classification.sanitized_failure.retryable is False
        assert observed == [error.outcome]
        record = structured_generation_outcome_record(observed[0])
        assert record["status"] == "terminal_failure"
        assert record["attempts"][0]["failure"] == {
            "kind": "timeout",
            "retryable": False,
            "safe_message": (
                "provider attempt exceeded its hard deadline; the owned "
                "transport was closed and the attempt was drained"
            ),
            "status_code": None,
            "retry_after_seconds": None,
                "provider_error_code": None,
                "provider_error_envelope_sha256": None,
                "exception_provenance": None,
                "stream_timeout_phase": None,
            "output_failure_mode": None,
            "validation_issues": [],
        }
        assert record["response"] is None
        assert "Return answer 30" not in json.dumps(record, sort_keys=True)
        assert client.close_calls == 1
        assert agent.run_calls == 1
        assert agent.cancellation_seen.is_set()
        assert agent.finished.is_set()
        current = asyncio.current_task()
        assert all(task is current or task.done() for task in asyncio.all_tasks())

    asyncio.run(scenario())


def test_schema_repair_policy_manifest_is_immutable_and_content_addressed() -> None:
    manifest = SCHEMA_REPAIR_POLICY_MANIFEST

    assert SchemaRepairAttemptPolicy.manifest is manifest
    assert manifest.policy_id == "structured_output_schema_repair"
    assert manifest.policy_version == 3
    assert manifest.max_suffix_utf8_bytes == MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES
    assert manifest.max_schema_nodes == MAX_SCHEMA_REPAIR_SCHEMA_NODES
    assert manifest.max_required_paths == MAX_SCHEMA_REPAIR_REQUIRED_PATHS
    assert manifest.template_sha256 == (
        "b075cad4590f938fcf7624d463a747eb1dda02033747ff15a86891c13c293f00"
    )
    assert manifest.semantic_guidance_sha256 == (
        "2e09b7e66f56a23ef1bf40b2788b7b767499b094a297f77735db3dd457f4324f"
    )
    assert manifest.policy_sha256 == (
        "dbda83d79337fb8d09c5f82fc24d747b3a2371fd1a807c4672852c2ede83425e"
    )
    assert manifest.to_trace_record() == {
        "max_required_paths": 256,
        "max_schema_nodes": 4_096,
        "max_suffix_utf8_bytes": 24_576,
        "policy_id": "structured_output_schema_repair",
        "policy_version": 3,
        "semantic_guidance_sha256": (
            "2e09b7e66f56a23ef1bf40b2788b7b767499b094a297f77735db3dd457f4324f"
        ),
        "template_sha256": (
            "b075cad4590f938fcf7624d463a747eb1dda02033747ff15a86891c13c293f00"
        ),
        "policy_sha256": (
            "dbda83d79337fb8d09c5f82fc24d747b3a2371fd1a807c4672852c2ede83425e"
        ),
    }
    assert set(_SEMANTIC_REPAIR_GUIDANCE) == set(ValidationIssueReasonCode)
    assert AttemptRequestVariant("schema_repair_v1") is (
        AttemptRequestVariant.SCHEMA_REPAIR_V1
    )
    with pytest.raises(ValueError, match="does not authenticate"):
        replace(manifest, policy_version=4)
    with pytest.raises(FrozenInstanceError):
        manifest.policy_version = 2

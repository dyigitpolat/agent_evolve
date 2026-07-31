from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import ClassVar, Literal

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
)
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from pydantic_core import PydanticCustomError
from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ContentFilterError,
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.providers.openrouter import OpenRouterProvider

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.domain.llm_task_queue import (
    AttemptStatus,
    CanonicalProviderErrorCode,
    SanitizedValidationIssue,
    StructuredOutputFailureMode,
    TaskOutcomeStatus,
    ValidationIssueCategory,
    ValidationIssueReasonCode,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
    PydanticAIStructuredGenerator,
    PROVIDER_ERROR_ENVELOPE_FINGERPRINT_ALGORITHM,
    _http_failure,
    classify_generation_exception,
)
from agent_evolve.integrations.pydantic_ai.validated_openrouter_model import (
    InvalidOpenRouterStreamItemError,
    ValidatedOpenRouterModel,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    ExactPayloadAttemptPolicy,
    OutcomePublicationPolicy,
    QueuedStructuredGenerationError,
    TransportOnlyStructuredGenerationRetryClassifier,
    create_production_queued_runner,
)
from agent_evolve.policies.llm_backoff import NoJitter
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
    StructuredGenerationRequest,
    StructuredStreamChannel,
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgressKind,
    StructuredStreamTimeoutError,
    StructuredStreamTimeoutPhase,
)


class _SmokeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acknowledgement: str
    number: int


class _DiagnosticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    required_number: int
    exact_choice: Literal["allowed"]
    required_text: str


class _SemanticDiagnosticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reason_code: ClassVar[str] = "duplicate_finite_options"
    value: int

    @model_validator(mode="after")
    def _reject(self) -> "_SemanticDiagnosticOutput":
        raise PydanticCustomError(
            type(self).reason_code,
            "RAW_SECRET_SEMANTIC_VALIDATION_MESSAGE",
        )


def _assert_exception_graph_and_traceback_locals_omit(
    root: BaseException,
    *forbidden: str,
) -> None:
    """Inspect every linked exception and direct traceback-frame local."""

    pending: deque[BaseException] = deque((root,))
    seen: set[int] = set()
    rendered: list[str] = []
    while pending:
        current = pending.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend(
            (
                repr(current),
                repr(current.args),
                repr(current.__dict__),
            )
        )
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        traceback = current.__traceback__
        while traceback is not None:
            rendered.extend(
                repr(value) for value in traceback.tb_frame.f_locals.values()
            )
            traceback = traceback.tb_next

    retained = "\n".join(rendered)
    for value in forbidden:
        assert value not in retained


def _request(**changes) -> StructuredGenerationRequest[_SmokeOutput]:
    values = {
        "call_id": LLMCallId("call_async_test_000001"),
        "operation": "transport_smoke",
        "prompt": "Return a typed smoke response.",
        "output_type": _SmokeOutput,
        "output_tool_name": "return_smoke",
        "max_output_tokens": 128,
        "temperature": 0.0,
    }
    values.update(changes)
    return StructuredGenerationRequest(**values)


@pytest.mark.parametrize(
    ("code", "error_type", "expected_kind", "expected_retryable"),
    (
        (400, "invalid_request", GenerationFailureKind.INVALID_REQUEST, False),
        (429, "rate_limit_exceeded", GenerationFailureKind.RATE_LIMITED, True),
        (
            502,
            "provider_unavailable",
            GenerationFailureKind.PROVIDER_UNAVAILABLE,
            True,
        ),
        (
            503,
            "provider_overloaded",
            GenerationFailureKind.PROVIDER_UNAVAILABLE,
            True,
        ),
    ),
)
def test_openai_in_band_sse_integer_code_uses_redacted_http_semantics(
    code: int,
    error_type: str,
    expected_kind: GenerationFailureKind,
    expected_retryable: bool,
) -> None:
    secret = "OPENROUTER_API_KEY=sk-in-band-secret https://private.example"
    raw = APIError(
        secret,
        httpx.Request("POST", "https://private.example/v1/chat"),
        body={
            "code": code,
            "message": secret,
            "metadata": {"error_type": error_type, "raw": secret},
        },
    )

    classified = classify_generation_exception(raw)

    assert classified.kind is expected_kind
    assert classified.retryable is expected_retryable
    assert classified.status_code == code
    assert classified.provider_error_code is CanonicalProviderErrorCode(error_type)
    assert classified.exception_provenance is None
    retained = repr(classified.__dict__)
    assert secret not in retained
    assert "sk-in-band-secret" not in retained
    assert "private.example" not in retained


@pytest.mark.parametrize(
    "body",
    (
        {"code": "503", "message": "RAW_SECRET"},
        {"code": 200, "message": "RAW_SECRET"},
        {"code": True, "message": "RAW_SECRET"},
        [503, "RAW_SECRET"],
    ),
)
def test_untrusted_in_band_sse_body_stays_terminal_unknown_with_provenance(
    body: object,
) -> None:
    secret = "OPENROUTER_API_KEY=sk-hostile-secret https://hostile.example"
    raw = APIError(
        secret,
        httpx.Request("POST", "https://hostile.example/v1/chat"),
        body=body,
    )

    classified = classify_generation_exception(raw)

    assert classified.kind is GenerationFailureKind.UNKNOWN
    assert classified.retryable is False
    assert classified.status_code is None
    assert classified.exception_provenance is not None
    assert classified.exception_provenance.nodes[0].family.value == "openai"
    retained = repr(classified.__dict__)
    for forbidden in (secret, "sk-hostile-secret", "hostile.example", "RAW_SECRET"):
        assert forbidden not in retained


def test_api_error_subclass_cannot_spoof_in_band_http_semantics() -> None:
    class _DistinctAPIErrorSubclass(APIError):
        pass

    secret = "OPENROUTER_API_KEY=sk-subclass-secret https://subclass.example"
    raw = _DistinctAPIErrorSubclass(
        secret,
        httpx.Request("POST", "https://subclass.example/v1/chat"),
        body={
            "code": 503,
            "message": secret,
            "metadata": {"error_type": "provider_overloaded", "raw": secret},
        },
    )

    classified = classify_generation_exception(raw)

    assert classified.kind is GenerationFailureKind.UNKNOWN
    assert classified.retryable is False
    assert classified.status_code is None
    assert classified.provider_error_code is None
    assert classified.exception_provenance is not None
    retained = repr(classified.__dict__)
    assert secret not in retained
    assert "sk-subclass-secret" not in retained
    assert "subclass.example" not in retained


def test_real_openrouter_stream_path_classifies_official_in_band_503() -> None:
    """Regress the exact SDK seam that made both BOiLS R3 calls UNKNOWN."""

    provider_secret = (
        "OPENROUTER_API_KEY=sk-offline-provider-secret "
        "https://private-provider.example"
    )
    sse_payload = (
        "data: "
        + json.dumps(
            {
                "id": "gen-offline-in-band-error",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "deepseek/deepseek-v4-pro",
                "provider": "StreamLake",
                "error": {
                    "code": 503,
                    "message": provider_secret,
                    "metadata": {
                        "error_type": "provider_overloaded",
                        "provider_code": provider_secret,
                        "raw": provider_secret,
                    },
                },
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": "error",
                    }
                ],
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n\n"
    ).encode("ascii")

    class _OfficialErrorStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            yield sse_payload

        async def aclose(self) -> None:
            self.closed = True

    error_stream = _OfficialErrorStream()
    observed_wire_bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        parsed = json.loads(request.content)
        assert type(parsed) is dict
        observed_wire_bodies.append(parsed)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=error_stream,
            request=request,
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=httpx.Timeout(connect=90.0, pool=90.0, write=90.0, read=None),
    )
    openai_client = AsyncOpenAI(
        base_url="https://offline-openrouter.invalid/api/v1",
        api_key="sk-offline-unit-test-only",
        max_retries=0,
        http_client=http_client,
    )
    model = ValidatedOpenRouterModel(
        "deepseek/deepseek-v4-pro",
        provider=OpenRouterProvider(openai_client=openai_client),
    )
    progress = []
    generator = PydanticAIStructuredGenerator(
        agent=Agent(model, retries=0),
        requested_model="deepseek/deepseek-v4-pro",
        provider_options={"only": ["streamlake"], "allow_fallbacks": False},
        reasoning_config=OpenRouterReasoningConfig(effort="xhigh"),
        owned_openai_client=openai_client,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=10_000_000_000,
            idle_timeout_ns=10_000_000_000,
            absolute_timeout_ns=20_000_000_000,
        ),
        stream_progress_sink=progress.append,
    )
    request = _request(
        call_id=LLMCallId("call_boils_r3_offline_in_band_repro"),
        operation="select_portfolio",
        prompt="Provider-free BOiLS R3 transport regression.",
        output_tool_name="propose_calibrated_portfolio_slate",
        max_output_tokens=384_000,
        temperature=0.2,
    )

    async def scenario() -> tuple[StructuredGenerationError, list[str]]:
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled: list[str] = []

        def capture_unhandled(_loop, context) -> None:
            exception = context.get("exception")
            unhandled.append(
                str(context.get("message"))
                + ":"
                + ("none" if exception is None else type(exception).__name__)
            )

        loop.set_exception_handler(capture_unhandled)
        try:
            with pytest.raises(StructuredGenerationError) as caught:
                await generator.generate_once(request)
            return caught.value, unhandled
        finally:
            await generator.aclose()
            await asyncio.sleep(0)
            loop.set_exception_handler(previous_handler)

    failure, unhandled = asyncio.run(scenario())

    assert failure.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE
    assert failure.retryable is True
    assert failure.status_code == 503
    assert (
        failure.provider_error_code
        is CanonicalProviderErrorCode.PROVIDER_OVERLOADED
    )
    assert failure.provider_error_envelope_sha256 is not None
    assert failure.exception_provenance is None
    assert progress == []
    assert unhandled == []
    assert error_stream.closed is True
    assert http_client.is_closed is True

    assert len(observed_wire_bodies) == 1
    wire = observed_wire_bodies[0]
    assert wire["model"] == "deepseek/deepseek-v4-pro"
    assert wire["reasoning"] == {"effort": "xhigh"}
    assert wire["provider"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert wire["stream"] is True
    assert wire["stream_options"] == {"include_usage": True}
    assert wire["max_completion_tokens"] == 384_000
    assert wire["tool_choice"] == "required"
    assert wire["usage"] == {"include": True}
    assert len(wire["tools"]) == 1
    assert wire["tools"][0]["function"]["name"] == (
        "propose_calibrated_portfolio_slate"
    )

    retained = repr(failure.__dict__)
    for forbidden in (
        provider_secret,
        "sk-offline-provider-secret",
        "private-provider.example",
        "sk-offline-unit-test-only",
        "offline-openrouter.invalid",
    ):
        assert forbidden not in retained


@pytest.mark.parametrize(
    "later_item",
    (
        None,
        [],
        "OPENROUTER_API_KEY=sk-malformed-stream-item-secret",
        17,
        True,
    ),
    ids=("null", "list", "string", "number", "boolean"),
)
def test_real_openrouter_non_chunk_later_item_is_narrow_retryable_failure(
    later_item: object,
) -> None:
    """Exercise AsyncOpenAI -> OpenRouterModel -> Agent, not a test double."""

    payload_secret = "OPENROUTER_API_KEY=sk-malformed-stream-item-secret"
    valid_empty_first_item = {
        "id": "chatcmpl-valid-empty-first-item",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek/deepseek-v4-pro",
        "choices": [],
    }
    sse_payload = b"".join(
        (
            "data: "
            + json.dumps(item, ensure_ascii=True, separators=(",", ":"))
            + "\n\n"
        ).encode("ascii")
        for item in (valid_empty_first_item, later_item)
    ) + b"data: [DONE]\n\n"

    class _MalformedLaterItemStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            yield sse_payload

        async def aclose(self) -> None:
            self.closed = True

    malformed_stream = _MalformedLaterItemStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=malformed_stream,
            request=request,
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=httpx.Timeout(connect=90.0, pool=90.0, write=90.0, read=None),
    )
    openai_client = AsyncOpenAI(
        base_url="https://malformed-openrouter.invalid/api/v1",
        api_key="sk-malformed-client-secret",
        max_retries=0,
        http_client=http_client,
    )
    model = ValidatedOpenRouterModel(
        "deepseek/deepseek-v4-pro",
        provider=OpenRouterProvider(openai_client=openai_client),
    )
    progress = []
    generator = PydanticAIStructuredGenerator(
        agent=Agent(model, retries=0),
        requested_model="deepseek/deepseek-v4-pro",
        provider_options={"only": ["streamlake"], "allow_fallbacks": False},
        owned_openai_client=openai_client,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=10_000_000_000,
            idle_timeout_ns=10_000_000_000,
            absolute_timeout_ns=20_000_000_000,
        ),
        stream_progress_sink=progress.append,
    )

    async def scenario() -> StructuredGenerationError:
        try:
            with pytest.raises(StructuredGenerationError) as caught:
                await generator.generate_once(_request())
            return caught.value
        finally:
            await generator.aclose()
            await asyncio.sleep(0)

    failure = asyncio.run(scenario())

    assert failure.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE
    assert failure.retryable is True
    assert failure.safe_message == "provider stream returned an invalid item"
    # This is a local protocol classification of an HTTP-200 stream, not a
    # fabricated provider status.
    assert failure.status_code is None
    assert failure.retry_after_seconds is None
    assert failure.exception_provenance is None
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert progress == []
    assert malformed_stream.closed is True
    assert http_client.is_closed is True
    retained = repr(failure.__dict__)
    for forbidden in (
        payload_secret,
        "sk-malformed-stream-item-secret",
        "sk-malformed-client-secret",
        "malformed-openrouter.invalid",
    ):
        assert forbidden not in retained
    _assert_exception_graph_and_traceback_locals_omit(
        failure,
        payload_secret,
        "sk-malformed-stream-item-secret",
        "sk-malformed-client-secret",
        "malformed-openrouter.invalid",
    )


def test_real_openrouter_valid_first_chunk_and_typed_output_are_unchanged() -> None:
    tool_arguments = json.dumps(
        {"acknowledgement": "accepted", "number": 29},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    chunks = (
        {
            "id": "chatcmpl-valid-boundary",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek/deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_valid_boundary",
                                "type": "function",
                                "function": {
                                    "name": "return_smoke",
                                    "arguments": tool_arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-valid-boundary",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek/deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 6,
                "total_tokens": 15,
            },
        },
    )
    sse_payload = b"".join(
        (
            "data: "
            + json.dumps(chunk, ensure_ascii=True, separators=(",", ":"))
            + "\n\n"
        ).encode("ascii")
        for chunk in chunks
    ) + b"data: [DONE]\n\n"

    class _ValidStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            yield sse_payload

        async def aclose(self) -> None:
            self.closed = True

    valid_stream = _ValidStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=valid_stream,
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    openai_client = AsyncOpenAI(
        base_url="https://valid-openrouter.invalid/api/v1",
        api_key="sk-valid-client-secret",
        max_retries=0,
        http_client=http_client,
    )
    model = ValidatedOpenRouterModel(
        "deepseek/deepseek-v4-pro",
        provider=OpenRouterProvider(openai_client=openai_client),
    )
    progress = []
    generator = PydanticAIStructuredGenerator(
        agent=Agent(model, retries=0),
        requested_model="deepseek/deepseek-v4-pro",
        owned_openai_client=openai_client,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=10_000_000_000,
            idle_timeout_ns=10_000_000_000,
            absolute_timeout_ns=20_000_000_000,
        ),
        stream_progress_sink=progress.append,
    )

    async def scenario():
        try:
            return await generator.generate_once(_request())
        finally:
            await generator.aclose()
            await asyncio.sleep(0)

    response = asyncio.run(scenario())

    assert response.value == _SmokeOutput(acknowledgement="accepted", number=29)
    assert response.requested_model == "deepseek/deepseek-v4-pro"
    assert response.resolved_model == "deepseek/deepseek-v4-pro"
    assert response.input_tokens == 9
    assert response.output_tokens == 6
    assert progress
    assert valid_stream.closed is True
    assert http_client.is_closed is True


def test_real_openrouter_guard_delegates_stream_close_during_cancellation() -> None:
    first_chunk = {
        "id": "chatcmpl-cancel-boundary",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek/deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "partial"},
                "finish_reason": None,
            }
        ],
    }
    first_sse_event = (
        "data: "
        + json.dumps(first_chunk, ensure_ascii=True, separators=(",", ":"))
        + "\n\n"
    ).encode("ascii")

    async def scenario() -> tuple[bool, bool, list[object], list[str]]:
        never_release = asyncio.Event()
        first_progress = asyncio.Event()

        class _BlockingValidStream(httpx.AsyncByteStream):
            def __init__(self) -> None:
                self.closed = False

            async def __aiter__(self):
                yield first_sse_event
                await never_release.wait()

            async def aclose(self) -> None:
                self.closed = True

        blocking_stream = _BlockingValidStream()

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=blocking_stream,
                request=request,
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        openai_client = AsyncOpenAI(
            base_url="https://cancel-openrouter.invalid/api/v1",
            api_key="sk-cancel-client-secret",
            max_retries=0,
            http_client=http_client,
        )
        model = ValidatedOpenRouterModel(
            "deepseek/deepseek-v4-pro",
            provider=OpenRouterProvider(openai_client=openai_client),
        )
        progress: list[object] = []

        def progress_sink(item: object) -> None:
            progress.append(item)
            first_progress.set()

        generator = PydanticAIStructuredGenerator(
            agent=Agent(model, retries=0),
            requested_model="deepseek/deepseek-v4-pro",
            owned_openai_client=openai_client,
            stream_liveness_policy=StructuredStreamLivenessPolicy(
                first_event_timeout_ns=10_000_000_000,
                idle_timeout_ns=10_000_000_000,
                absolute_timeout_ns=20_000_000_000,
            ),
            stream_progress_sink=progress_sink,
        )
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled: list[str] = []

        def capture_unhandled(_loop, context) -> None:
            exception = context.get("exception")
            unhandled.append(
                str(context.get("message"))
                + ":"
                + ("none" if exception is None else type(exception).__name__)
            )

        loop.set_exception_handler(capture_unhandled)
        try:
            task = asyncio.create_task(generator.generate_once(_request()))
            await asyncio.wait_for(first_progress.wait(), timeout=2.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await generator.aclose()
            await asyncio.sleep(0)
            loop.set_exception_handler(previous_handler)
        return blocking_stream.closed, http_client.is_closed, progress, unhandled

    stream_closed, client_closed, progress, unhandled = asyncio.run(scenario())

    assert progress
    assert stream_closed is True
    assert client_closed is True
    assert unhandled == []


def test_queue_retries_malformed_later_item_once_with_exact_payload() -> None:
    """Keep the queue as the sole owner of a physical exact-payload retry."""

    tool_arguments = json.dumps(
        {"acknowledgement": "recovered", "number": 31},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    valid_chunks = (
        {
            "id": "chatcmpl-recovered-boundary",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek/deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_recovered_boundary",
                                "type": "function",
                                "function": {
                                    "name": "return_smoke",
                                    "arguments": tool_arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-recovered-boundary",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek/deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
            },
        },
    )
    valid_payload = b"".join(
        (
            "data: "
            + json.dumps(chunk, ensure_ascii=True, separators=(",", ":"))
            + "\n\n"
        ).encode("ascii")
        for chunk in valid_chunks
    ) + b"data: [DONE]\n\n"
    valid_empty_first_item = {
        "id": "chatcmpl-retry-empty-first-item",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek/deepseek-v4-pro",
        "choices": [],
    }
    malformed_later_payload = (
        "data: "
        + json.dumps(
            valid_empty_first_item,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n\ndata: null\n\ndata: [DONE]\n\n"
    ).encode("ascii")
    response_payloads = deque((malformed_later_payload, valid_payload))
    observed_wire_bodies: list[dict[str, object]] = []
    streams: list[httpx.AsyncByteStream] = []

    class _AttemptStream(httpx.AsyncByteStream):
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.closed = False

        async def __aiter__(self):
            yield self.payload

        async def aclose(self) -> None:
            self.closed = True

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert type(body) is dict
        observed_wire_bodies.append(body)
        stream = _AttemptStream(response_payloads.popleft())
        streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    openai_client = AsyncOpenAI(
        base_url="https://retry-openrouter.invalid/api/v1",
        api_key="sk-retry-client-secret",
        max_retries=0,
        http_client=http_client,
    )
    model = ValidatedOpenRouterModel(
        "deepseek/deepseek-v4-pro",
        provider=OpenRouterProvider(openai_client=openai_client),
    )
    progress = []
    generator = PydanticAIStructuredGenerator(
        agent=Agent(model, retries=0),
        requested_model="deepseek/deepseek-v4-pro",
        provider_options={"only": ["streamlake"], "allow_fallbacks": False},
        owned_openai_client=openai_client,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=10_000_000_000,
            idle_timeout_ns=10_000_000_000,
            absolute_timeout_ns=20_000_000_000,
        ),
        stream_progress_sink=progress.append,
    )
    outcomes = []
    runner = create_production_queued_runner(
        generator=generator,
        max_in_flight=1,
        max_pending=0,
        max_attempts=2,
        attempt_timeout_ns=None,
        base_backoff_ns=0,
        max_backoff_ns=0,
        jitter_policy=NoJitter(),
        close_generator=True,
        outcome_sink=outcomes.append,
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
    )

    async def scenario():
        try:
            return await runner(_request())
        finally:
            await runner.aclose()
            await asyncio.sleep(0)

    attempted = asyncio.run(scenario())

    assert attempted.attempt_count == 2
    assert attempted.response.value == _SmokeOutput(
        acknowledgement="recovered",
        number=31,
    )
    assert response_payloads == deque()
    assert len(observed_wire_bodies) == 2
    # Physical attempt identity is queue-only: provider payload, prompt, tool
    # schema, settings, and route are byte-semantically identical.
    assert observed_wire_bodies[0] == observed_wire_bodies[1]
    assert len(outcomes) == 1
    assert outcomes[0].response is attempted.response
    attempts = outcomes[0].telemetry.attempts
    assert [attempt.status for attempt in attempts] == [
        AttemptStatus.RETRYABLE_FAILURE,
        AttemptStatus.SUCCEEDED,
    ]
    assert len(
        {
            attempt.request_evidence.provider_attempt_id
            for attempt in attempts
        }
    ) == 2
    assert len(
        {attempt.request_evidence.prompt_sha256 for attempt in attempts}
    ) == 1
    first_failure = attempts[0].classification.sanitized_failure
    assert first_failure.kind == GenerationFailureKind.PROVIDER_UNAVAILABLE.value
    assert first_failure.retryable is True
    assert first_failure.status_code is None
    assert first_failure.exception_provenance is None
    first_attempt_id = attempts[0].request_evidence.provider_attempt_id
    second_attempt_id = attempts[1].request_evidence.provider_attempt_id
    assert first_attempt_id is not None
    assert second_attempt_id is not None
    assert not any(
        row.provider_attempt_id == first_attempt_id.value for row in progress
    )
    assert progress
    assert {
        row.provider_attempt_id for row in progress
    } == {second_attempt_id.value}
    assert all(getattr(stream, "closed") for stream in streams)
    assert http_client.is_closed is True


def test_queue_never_replays_invalid_stream_item_after_semantic_progress() -> None:
    """A partial semantic generation is ambiguous and must fail closed."""

    semantic_chunk = {
        "id": "chatcmpl-semantic-then-invalid",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek/deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "partial semantic model output",
                },
                "finish_reason": None,
            }
        ],
    }
    payload = (
        "data: "
        + json.dumps(semantic_chunk, ensure_ascii=True, separators=(",", ":"))
        + "\n\ndata: null\n\ndata: [DONE]\n\n"
    ).encode("ascii")
    observed_wire_bodies: list[dict[str, object]] = []
    streams: list[httpx.AsyncByteStream] = []

    class _SemanticThenInvalidStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            yield payload

        async def aclose(self) -> None:
            self.closed = True

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert type(body) is dict
        observed_wire_bodies.append(body)
        stream = _SemanticThenInvalidStream()
        streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    openai_client = AsyncOpenAI(
        base_url="https://nonreplay-openrouter.invalid/api/v1",
        api_key="sk-nonreplay-client-secret",
        max_retries=0,
        http_client=http_client,
    )
    model = ValidatedOpenRouterModel(
        "deepseek/deepseek-v4-pro",
        provider=OpenRouterProvider(openai_client=openai_client),
    )
    progress = []
    generator = PydanticAIStructuredGenerator(
        agent=Agent(model, retries=0),
        requested_model="deepseek/deepseek-v4-pro",
        provider_options={"only": ["streamlake"], "allow_fallbacks": False},
        owned_openai_client=openai_client,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=10_000_000_000,
            idle_timeout_ns=10_000_000_000,
            absolute_timeout_ns=20_000_000_000,
        ),
        stream_progress_sink=progress.append,
    )
    outcomes = []
    runner = create_production_queued_runner(
        generator=generator,
        max_in_flight=1,
        max_pending=0,
        max_attempts=2,
        attempt_timeout_ns=None,
        base_backoff_ns=0,
        max_backoff_ns=0,
        jitter_policy=NoJitter(),
        close_generator=True,
        outcome_sink=outcomes.append,
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        attempt_request_policy=ExactPayloadAttemptPolicy(),
        retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
    )

    async def scenario() -> QueuedStructuredGenerationError:
        try:
            with pytest.raises(QueuedStructuredGenerationError) as caught:
                await runner(_request())
            return caught.value
        finally:
            await runner.aclose()
            await asyncio.sleep(0)

    failure = asyncio.run(scenario())

    assert failure.status is TaskOutcomeStatus.TERMINAL_FAILURE
    assert len(observed_wire_bodies) == 1
    assert len(streams) == 1
    assert streams[0].closed is True
    assert http_client.is_closed is True
    assert progress
    assert any(row.event_content_utf8_bytes > 0 for row in progress)
    assert all(
        row.kind is not StructuredStreamProgressKind.STREAM_COMPLETED
        for row in progress
    )
    assert outcomes == [failure.outcome]
    attempts = failure.telemetry.attempts
    assert len(attempts) == 1
    assert attempts[0].status is AttemptStatus.TERMINAL_FAILURE
    sanitized = attempts[0].classification.sanitized_failure
    assert sanitized.kind == GenerationFailureKind.PROVIDER_UNAVAILABLE.value
    assert sanitized.retryable is False
    assert sanitized.safe_message == (
        "provider stream returned an invalid item after semantic progress"
    )


def test_invalid_stream_item_exception_maps_only_by_exact_bounded_type() -> None:
    secret = "OPENROUTER_API_KEY=sk-topology-secret"
    wrapped = ExceptionGroup(
        secret,
        [RuntimeError(secret), InvalidOpenRouterStreamItemError()],
    )
    classified = classify_generation_exception(wrapped)

    assert classified.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE
    assert classified.retryable is True
    assert classified.status_code is None
    assert classified.exception_provenance is None
    assert secret not in repr(classified.__dict__)

    post_progress = classify_generation_exception(
        InvalidOpenRouterStreamItemError(),
        semantic_progress_observed=True,
    )
    assert post_progress.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE
    assert post_progress.retryable is False
    assert post_progress.safe_message == (
        "provider stream returned an invalid item after semantic progress"
    )
    assert post_progress.status_code is None
    assert post_progress.exception_provenance is None

    class _HostileLookalike(InvalidOpenRouterStreamItemError):
        pass

    with pytest.raises(ValidationError) as validation_caught:
        _SmokeOutput.model_validate({"acknowledgement": secret})

    for hostile in (
        AttributeError(secret),
        validation_caught.value,
        _HostileLookalike(),
    ):
        unknown = classify_generation_exception(hostile)
        assert unknown.kind is GenerationFailureKind.UNKNOWN
        assert unknown.retryable is False
        assert unknown.exception_provenance is not None
        assert secret not in repr(unknown.__dict__)


def test_testmodel_executes_one_typed_attempt_with_telemetry() -> None:
    agent = Agent(
        TestModel(
            custom_output_args={"acknowledgement": "ok", "number": 42},
            model_name="offline-test-model",
        ),
        retries=0,
    )
    generator = PydanticAIStructuredGenerator(
        agent=agent,
        requested_model="offline/test-model",
    )

    response = asyncio.run(generator.generate_once(_request()))

    assert response.value == _SmokeOutput(acknowledgement="ok", number=42)
    assert response.requested_model == "offline/test-model"
    assert response.resolved_model == "offline-test-model"
    assert response.input_tokens >= 0
    assert response.output_tokens >= 0
    assert response.latency_ns > 0
    assert response.cost_usd is None


def test_default_stream_supervisor_retires_and_closes_owned_transport() -> None:
    class _OwnedClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        client = _OwnedClient()
        agent = _RaisingAgent(AssertionError("retired generator must not dispatch"))
        generator = PydanticAIStructuredGenerator(
            agent=agent,
            requested_model="offline/retirement-test",
            owned_openai_client=client,
            stream_liveness_policy=StructuredStreamLivenessPolicy(
                first_event_timeout_ns=1_000_000,
                idle_timeout_ns=1_000_000,
            ),
        )
        supervisor = generator._stream_supervisor
        assert supervisor is not None
        await supervisor._retirement_operation()

        assert client.close_calls == 1
        assert generator._owned_openai_client is None
        with pytest.raises(StructuredGenerationError) as caught:
            await generator.generate_once(_request())
        assert caught.value.kind is GenerationFailureKind.CANCELLED
        assert caught.value.retryable is False
        assert agent.calls == 0

    asyncio.run(scenario())


def test_owned_stream_transport_close_is_bounded_when_close_resists_cancel() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        close_started = asyncio.Event()
        close_cancelled = asyncio.Event()
        close_settled = asyncio.Event()

        class _ResistantClient:
            async def close(self) -> None:
                close_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    close_cancelled.set()
                    await release.wait()
                finally:
                    close_settled.set()

        generator = PydanticAIStructuredGenerator(
            agent=_RaisingAgent(AssertionError("must not dispatch")),
            requested_model="offline/bounded-close-test",
            owned_openai_client=_ResistantClient(),
            stream_liveness_policy=StructuredStreamLivenessPolicy(
                first_event_timeout_ns=1_000_000,
                idle_timeout_ns=1_000_000,
                cleanup_policy=StructuredStreamCleanupPolicy(
                    cancel_drain_timeout_ns=5_000_000,
                    transport_retire_timeout_ns=5_000_000,
                ),
            ),
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        await generator.aclose()
        elapsed = loop.time() - started

        assert close_started.is_set()
        await asyncio.sleep(0)
        assert close_cancelled.is_set()
        assert generator._owned_openai_client is None
        assert elapsed < 0.2
        release.set()
        await asyncio.wait_for(close_settled.wait(), timeout=1.0)

    asyncio.run(scenario())


class _RaisingAgent:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def run(self, *_args, **_kwargs):
        self.calls += 1
        raise self.error


class _CapturingAgent:
    def __init__(
        self,
        delegate: Agent,
        *,
        streamed_tool_args: dict[str, object] | None = None,
    ) -> None:
        self.delegate = delegate
        self.streamed_tool_args = streamed_tool_args
        self.model_settings: dict[str, object] | None = None
        self.event_stream_handler = None
        self.output_type = None
        self.usage_limits = None
        self.returned = False

    async def run(self, *args, **kwargs):
        self.returned = False
        self.model_settings = kwargs["model_settings"]
        self.event_stream_handler = kwargs.get("event_stream_handler")
        self.output_type = kwargs["output_type"]
        self.usage_limits = kwargs["usage_limits"]
        if (
            self.event_stream_handler is not None
            and self.streamed_tool_args is not None
        ):
            await _emit_string_tool_stream(
                self.event_stream_handler,
                self.streamed_tool_args,
            )
            kwargs = dict(kwargs)
            kwargs.pop("event_stream_handler")
        result = await self.delegate.run(*args, **kwargs)
        self.returned = True
        return result


async def _emit_string_tool_stream(handler, output: dict[str, object]) -> None:
    arguments = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    started_part = ToolCallPart(
        tool_name="return_smoke",
        args="",
        tool_call_id="offline_tool_call_000001",
    )
    completed_part = ToolCallPart(
        tool_name=started_part.tool_name,
        args=arguments,
        tool_call_id=started_part.tool_call_id,
    )

    async def events():
        yield PartStartEvent(index=0, part=started_part)
        # This faithfully reproduces the live v1 ordering: Pydantic-AI selects
        # the output tool before its argument deltas have finished streaming.
        yield FinalResultEvent(
            tool_name=started_part.tool_name,
            tool_call_id=started_part.tool_call_id,
        )
        yield PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(args_delta=arguments),
        )
        yield PartEndEvent(index=0, part=completed_part)

    await handler(None, events())


@pytest.mark.parametrize(
    ("reasoning_config", "expected"),
    [
        (OpenRouterReasoningConfig(effort="low"), {"effort": "low"}),
        (OpenRouterReasoningConfig(max_tokens=4_096), {"max_tokens": 4_096}),
    ],
)
def test_reasoning_config_is_forwarded_exactly(reasoning_config, expected) -> None:
    delegate = Agent(
        TestModel(
            custom_output_args={"acknowledgement": "ok", "number": 42},
            model_name="offline-test-model",
        ),
        retries=0,
    )
    agent = _CapturingAgent(delegate)
    generator = PydanticAIStructuredGenerator(
        agent=agent,
        requested_model="offline/test-model",
        reasoning_config=reasoning_config,
    )

    asyncio.run(generator.generate_once(_request()))

    assert agent.model_settings is not None
    assert agent.model_settings["openrouter_reasoning"] == expected
    assert set(agent.model_settings) == {
        "max_tokens",
        "openrouter_provider",
        "openrouter_reasoning",
        "openrouter_usage",
        "temperature",
    }


def test_streaming_preserves_request_settings_and_emits_content_blind_progress() -> (
    None
):
    delegate = Agent(
        TestModel(
            custom_output_args={"acknowledgement": "RAW_SECRET_OK", "number": 42},
            model_name="offline-test-model",
        ),
        retries=0,
    )
    agent = _CapturingAgent(
        delegate,
        streamed_tool_args={
            "acknowledgement": "RAW_SECRET_OK",
            "number": 42,
        },
    )
    progress = []

    def capture_progress(row) -> None:
        if row.kind is StructuredStreamProgressKind.STREAM_COMPLETED:
            assert agent.returned is True
        else:
            assert agent.returned is False
        progress.append(row)

    generator = PydanticAIStructuredGenerator(
        agent=agent,
        requested_model="offline/test-model",
        reasoning_config=OpenRouterReasoningConfig(effort="xhigh"),
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=1_000_000_000,
            idle_timeout_ns=1_000_000_000,
        ),
        stream_progress_sink=capture_progress,
    )

    provider_attempt_id = ProviderAttemptId(
        "provider_attempt_async_stream_binding_000001"
    )
    response = asyncio.run(
        generator.generate_once(
            _request(
                max_output_tokens=384_000,
                provider_attempt_id=provider_attempt_id,
            )
        )
    )

    assert response.value == _SmokeOutput(
        acknowledgement="RAW_SECRET_OK",
        number=42,
    )
    assert callable(agent.event_stream_handler)
    assert agent.model_settings == {
        "max_tokens": 384_000,
        "openrouter_provider": {"allow_fallbacks": True},
        "openrouter_reasoning": {"effort": "xhigh"},
        "openrouter_usage": {"include": True},
        "temperature": 0.0,
    }
    assert agent.usage_limits.output_tokens_limit == 384_000
    assert progress
    assert [row.sequence for row in progress] == list(range(1, len(progress) + 1))
    assert progress[0].kind is StructuredStreamProgressKind.PART_STARTED
    assert [row.kind for row in progress] == [
        StructuredStreamProgressKind.PART_STARTED,
        StructuredStreamProgressKind.OUTPUT_SELECTED,
        StructuredStreamProgressKind.PART_DELTA,
        StructuredStreamProgressKind.PART_ENDED,
        StructuredStreamProgressKind.STREAM_COMPLETED,
    ]
    assert all(type(row.channel) is StructuredStreamChannel for row in progress)
    assert progress[0].event_content_utf8_bytes > 0
    assert progress[2].event_content_utf8_bytes > 0
    assert progress[-1].event_content_utf8_bytes == 0
    assert progress[-1].cumulative_content_utf8_bytes > (
        progress[0].event_content_utf8_bytes
    )
    assert progress[-1].rolling_content_sha256 == progress[-2].rolling_content_sha256
    assert all(len(row.rolling_content_sha256) == 64 for row in progress)
    assert {row.provider_attempt_id for row in progress} == {provider_attempt_id.value}
    assert "RAW_SECRET" not in repr(progress)


def test_dictionary_tool_delta_fails_instead_of_inventing_original_bytes() -> None:
    generator = PydanticAIStructuredGenerator(
        agent=Agent(
            TestModel(
                custom_output_args={"acknowledgement": "ok", "number": 42},
                model_name="offline-test-model",
            ),
            retries=0,
        ),
        requested_model="offline/test-model",
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=1_000_000_000,
            idle_timeout_ns=1_000_000_000,
        ),
    )

    with pytest.raises(StructuredGenerationError) as caught:
        asyncio.run(generator.generate_once(_request()))

    assert caught.value.kind is GenerationFailureKind.UNKNOWN
    assert caught.value.retryable is False
    assert caught.value.safe_message == (
        "stream semantic content cannot be projected exactly"
    )


class _PartialStreamingAgent:
    def __init__(self) -> None:
        self.cancelled = False

    async def run(self, *_args, **kwargs):
        handler = kwargs["event_stream_handler"]

        async def events():
            yield PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta="RAW_SECRET_PARTIAL_REASONING"),
            )

        await handler(None, events())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_partial_stream_timeout_never_returns_a_scientific_response_or_content() -> (
    None
):
    agent = _PartialStreamingAgent()
    progress = []
    generator = PydanticAIStructuredGenerator(
        agent=agent,
        requested_model="offline/test-model",
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=1_000_000_000,
            idle_timeout_ns=10_000_000,
        ),
        stream_progress_sink=progress.append,
    )

    with pytest.raises(StructuredStreamTimeoutError) as caught:
        asyncio.run(generator.generate_once(_request()))

    assert caught.value.phase is StructuredStreamTimeoutPhase.IDLE
    assert agent.cancelled is True
    assert len(progress) == 1
    assert progress[0].kind is StructuredStreamProgressKind.PART_DELTA
    assert progress[0].channel is StructuredStreamChannel.THINKING
    assert "RAW_SECRET" not in repr(progress)
    assert "RAW_SECRET" not in repr(caught.value.__dict__)


class _MixedConcurrentStreamingAgent:
    def __init__(self, delegate: Agent) -> None:
        self.delegate = delegate
        self.stalled_cancelled = False

    async def run(self, prompt, **kwargs):
        if prompt != "Stall this request stream.":
            handler = kwargs.pop("event_stream_handler")
            await _emit_string_tool_stream(
                handler,
                {"acknowledgement": "peer-ok", "number": 7},
            )
            return await self.delegate.run(prompt, **kwargs)
        handler = kwargs["event_stream_handler"]

        async def events():
            yield PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(
                    args_delta="RAW_SECRET_PARTIAL_ARGUMENTS",
                ),
            )

        await handler(None, events())
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.stalled_cancelled = True
            raise


def test_one_idle_stream_is_cancelled_locally_while_peer_completes() -> None:
    delegate = Agent(
        TestModel(
            custom_output_args={"acknowledgement": "peer-ok", "number": 7},
            model_name="offline-test-model",
        ),
        retries=0,
    )
    agent = _MixedConcurrentStreamingAgent(delegate)
    progress = []
    generator = PydanticAIStructuredGenerator(
        agent=agent,
        requested_model="offline/test-model",
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=1_000_000_000,
            idle_timeout_ns=50_000_000,
        ),
        stream_progress_sink=progress.append,
    )
    stalled = _request(
        call_id=LLMCallId("call_async_test_stalled_000001"),
        prompt="Stall this request stream.",
    )
    peer = _request(
        call_id=LLMCallId("call_async_test_peer_000001"),
        prompt="Complete this peer request.",
    )

    async def scenario():
        stalled_task = asyncio.create_task(generator.generate_once(stalled))
        peer_task = asyncio.create_task(generator.generate_once(peer))
        return await asyncio.gather(
            stalled_task,
            peer_task,
            return_exceptions=True,
        )

    stalled_result, peer_result = asyncio.run(scenario())

    assert type(stalled_result) is StructuredStreamTimeoutError
    assert stalled_result.phase is StructuredStreamTimeoutPhase.IDLE
    assert agent.stalled_cancelled is True
    assert peer_result.value == _SmokeOutput(acknowledgement="peer-ok", number=7)
    assert {row.call_id for row in progress} == {
        stalled.call_id.value,
        peer.call_id.value,
    }
    assert "RAW_SECRET" not in repr(progress)


@pytest.mark.parametrize(
    "effort",
    ["xhigh", "high", "medium", "low", "minimal", "none"],
)
def test_all_supported_reasoning_efforts_are_accepted(effort) -> None:
    config = OpenRouterReasoningConfig(effort=effort)

    assert config.to_model_setting() == {"effort": effort}


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"effort": "low", "max_tokens": 1},
        {"effort": "unsupported"},
        {"effort": 1},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_tokens": -1},
    ],
)
def test_reasoning_config_rejects_ambiguous_or_invalid_values(changes) -> None:
    with pytest.raises(ValueError):
        OpenRouterReasoningConfig(**changes)


def test_reasoning_config_is_frozen_and_generator_rejects_raw_mapping() -> None:
    config = OpenRouterReasoningConfig(effort="low")
    with pytest.raises(FrozenInstanceError):
        config.effort = "high"  # type: ignore[misc]

    with pytest.raises(TypeError):
        PydanticAIStructuredGenerator(
            agent=object(),
            requested_model="offline/test-model",
            reasoning_config={"effort": "low"},  # type: ignore[arg-type]
        )


def test_adapter_does_not_retry_and_sanitizes_capability_mismatch() -> None:
    raw = RuntimeError("raw transport")
    raw.response = SimpleNamespace(headers={"retry-after": "19"})  # type: ignore[attr-defined]
    error = ModelHTTPError(
        404,
        "deepseek/deepseek-v4-pro",
        {
            "message": "RAW_SECRET_CAPABILITY_MESSAGE",
            "metadata": {"error_type": "no_endpoints_found"},
        },
    )
    error.__cause__ = raw
    agent = _RaisingAgent(error)
    generator = PydanticAIStructuredGenerator(
        agent=agent,
        requested_model="deepseek/deepseek-v4-pro",
    )

    with pytest.raises(StructuredGenerationError) as caught:
        asyncio.run(generator.generate_once(_request()))

    assert agent.calls == 1
    assert caught.value.kind is GenerationFailureKind.CAPABILITY_MISMATCH
    assert caught.value.retryable is False
    assert caught.value.status_code == 404
    assert caught.value.retry_after_seconds == 19
    assert "RAW_SECRET" not in repr(caught.value.__dict__)


@pytest.mark.parametrize(
    "body",
    (
        {"metadata": {"error_type": "no_endpoints_found"}},
        {"error_type": "capability_mismatch"},
        {"type": "no_compatible_endpoint"},
        {"code": "unsupported_parameters"},
        {"error": {"metadata": {"error_type": "no_endpoints_found"}}},
        {"error": {"error_type": "capability_mismatch"}},
        {"error": {"type": "no_compatible_endpoint"}},
        {"error": {"code": "unsupported_parameter"}},
    ),
)
def test_http_404_capability_mismatch_accepts_only_fixed_typed_paths(body) -> None:
    body["message"] = "OPENROUTER_API_KEY=RAW_SECRET_MUST_NOT_SURVIVE"
    translated = classify_generation_exception(ModelHTTPError(404, "model", body))

    assert translated.kind is GenerationFailureKind.CAPABILITY_MISMATCH
    assert translated.retryable is False
    assert translated.status_code == 404
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_http_404_message_content_cannot_select_capability_classification() -> None:
    translated = classify_generation_exception(
        ModelHTTPError(
            404,
            "model",
            {
                "message": (
                    "No endpoints found that can handle requested parameters "
                    "OPENROUTER_API_KEY=RAW_SECRET"
                )
            },
        )
    )

    assert translated.kind is GenerationFailureKind.INVALID_REQUEST
    assert translated.retryable is False
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_http_failure_never_renders_an_adversarial_body() -> None:
    class RaisingStringBody:
        def __str__(self) -> str:
            raise AssertionError("provider body rendering was attempted")

    translated = _http_failure(404, RaisingStringBody(), RuntimeError("offline"))

    assert translated.kind is GenerationFailureKind.INVALID_REQUEST
    assert translated.retryable is False
    assert translated.status_code == 404


def test_conflicting_typed_capability_fields_fail_closed() -> None:
    translated = classify_generation_exception(
        ModelHTTPError(
            404,
            "model",
            {
                "metadata": {"error_type": "no_endpoints_found"},
                "type": "invalid_request_error",
            },
        )
    )

    assert translated.kind is GenerationFailureKind.INVALID_REQUEST
    assert translated.retryable is False


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (429, GenerationFailureKind.RATE_LIMITED, True),
        (500, GenerationFailureKind.PROVIDER_UNAVAILABLE, True),
        (501, GenerationFailureKind.PROVIDER_UNAVAILABLE, True),
        (503, GenerationFailureKind.PROVIDER_UNAVAILABLE, True),
        (504, GenerationFailureKind.TIMEOUT, True),
        (599, GenerationFailureKind.PROVIDER_UNAVAILABLE, True),
        (408, GenerationFailureKind.TIMEOUT, True),
        (409, GenerationFailureKind.PROVIDER_UNAVAILABLE, False),
        (425, GenerationFailureKind.PROVIDER_UNAVAILABLE, False),
        (401, GenerationFailureKind.AUTHENTICATION, False),
        (402, GenerationFailureKind.PAYMENT_REQUIRED, False),
        (400, GenerationFailureKind.INVALID_REQUEST, False),
    ],
)
def test_http_failure_classification(status, kind, retryable) -> None:
    translated = classify_generation_exception(
        ModelHTTPError(status, "model", {"message": "untrusted detail"})
    )
    assert translated.kind is kind
    assert translated.retryable is retryable
    assert translated.status_code == status
    assert "untrusted detail" not in translated.safe_message


def test_http_error_telemetry_admits_only_closed_structured_codes() -> None:
    bodies = (
        {
            "error": {
                "message": "RAW_SECRET_MESSAGE_ONE",
                "metadata": {
                    "error_type": "invalid_request_error",
                    "raw": "OPENROUTER_API_KEY=sk-secret-one",
                },
            }
        },
        {
            "error": {
                "message": "RAW_SECRET_MESSAGE_TWO",
                "metadata": {
                    "error_type": "invalid_request_error",
                    "raw": "OPENROUTER_API_KEY=sk-secret-two",
                },
            }
        },
    )
    first, second = (
        classify_generation_exception(ModelHTTPError(400, "model", body))
        for body in bodies
    )

    assert first.provider_error_code is CanonicalProviderErrorCode.INVALID_REQUEST_ERROR
    assert second.provider_error_code is first.provider_error_code
    assert first.provider_error_envelope_sha256 == (
        second.provider_error_envelope_sha256
    )
    assert len(first.provider_error_envelope_sha256 or "") == 64
    assert (
        PROVIDER_ERROR_ENVELOPE_FINGERPRINT_ALGORITHM
        == "sha256_domain_and_canonical_redacted_structure_v1"
    )
    retained = repr((first.__dict__, second.__dict__))
    assert "RAW_SECRET" not in retained
    assert "sk-secret" not in retained


def test_pinned_openai_to_pydantic_error_chain_uses_direct_error_object() -> None:
    """Mirror the installed SDK unwrapping before Pydantic-AI translation."""

    from pydantic_ai.models.openai import _map_api_errors

    wire_body = {
        "error": {
            "code": 400,
            "message": "RAW_SECRET_PROVIDER_MESSAGE",
            "metadata": {
                "error_type": "invalid_request",
                "raw": "OPENROUTER_API_KEY=sk-secret-chain",
            },
        }
    }
    response = httpx.Response(
        400,
        request=httpx.Request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
    )
    client = OpenAI(
        api_key="offline-placeholder-not-a-credential",
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        sdk_error = client._make_status_error(
            "offline HTTP 400",
            body=wire_body,
            response=response,
        )
    finally:
        client.close()

    assert sdk_error.body == wire_body["error"]
    with pytest.raises(ModelHTTPError) as caught:
        with _map_api_errors("deepseek/deepseek-v4-pro"):
            raise sdk_error
    assert caught.value.body == wire_body["error"]

    translated = classify_generation_exception(caught.value)
    assert translated.kind is GenerationFailureKind.INVALID_REQUEST
    assert translated.retryable is False
    assert translated.status_code == 400
    assert translated.provider_error_code is CanonicalProviderErrorCode.INVALID_REQUEST
    same_shape_different_content = classify_generation_exception(
        ModelHTTPError(
            400,
            "model",
            {
                "code": 400,
                "message": "RAW_SECRET_DIFFERENT_MESSAGE",
                "metadata": {
                    "error_type": "invalid_request",
                    "raw": "OPENROUTER_API_KEY=sk-secret-different",
                },
            },
        )
    )
    assert translated.provider_error_envelope_sha256 == (
        same_shape_different_content.provider_error_envelope_sha256
    )
    retained = repr((translated.__dict__, same_shape_different_content.__dict__))
    assert "RAW_SECRET" not in retained
    assert "sk-secret" not in retained


def test_provider_error_code_vocabulary_covers_current_openrouter_types() -> None:
    official_current = {
        "authentication",
        "content_policy_violation",
        "context_length_exceeded",
        "image_download_failed",
        "image_not_found",
        "image_too_large",
        "image_too_small",
        "invalid_image",
        "invalid_prompt",
        "invalid_request",
        "max_tokens_exceeded",
        "not_found",
        "payload_too_large",
        "payment_required",
        "permission_denied",
        "precondition_failed",
        "provider_overloaded",
        "provider_unavailable",
        "rate_limit_exceeded",
        "refusal",
        "server",
        "string_too_long",
        "timeout",
        "token_limit_exceeded",
        "unmapped",
        "unprocessable",
        "unsupported_image_format",
    }

    assert official_current <= {code.value for code in CanonicalProviderErrorCode}


def test_direct_top_level_error_type_is_admitted_without_affecting_retry_policy() -> (
    None
):
    translated = classify_generation_exception(
        ModelHTTPError(
            400,
            "model",
            {
                "error_type": "timeout",
                "message": "RAW_SECRET_TIMEOUT_MESSAGE",
            },
        )
    )

    assert translated.provider_error_code is CanonicalProviderErrorCode.TIMEOUT
    assert translated.kind is GenerationFailureKind.INVALID_REQUEST
    assert translated.retryable is False
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_http_error_telemetry_drops_unknown_and_conflicting_codes() -> None:
    unknown_bodies = (
        {
            "error": {
                "metadata": {"error_type": "RAW_SECRET_UNKNOWN_ONE"},
                "message": "RAW_SECRET_MESSAGE_ONE",
            }
        },
        {
            "error": {
                "metadata": {"error_type": "RAW_SECRET_UNKNOWN_TWO"},
                "message": "RAW_SECRET_MESSAGE_TWO",
            }
        },
    )
    unknown_first, unknown_second = (
        classify_generation_exception(ModelHTTPError(400, "model", body))
        for body in unknown_bodies
    )
    conflicting = classify_generation_exception(
        ModelHTTPError(
            400,
            "model",
            {
                "error": {
                    "type": "rate_limit_error",
                    "metadata": {"error_type": "invalid_request_error"},
                }
            },
        )
    )

    assert unknown_first.provider_error_code is None
    assert unknown_second.provider_error_code is None
    assert unknown_first.provider_error_envelope_sha256 == (
        unknown_second.provider_error_envelope_sha256
    )
    assert conflicting.provider_error_code is None
    assert "RAW_SECRET" not in repr(
        (unknown_first.__dict__, unknown_second.__dict__, conflicting.__dict__)
    )


def test_provider_error_diagnostics_require_closed_values_and_http_status() -> None:
    common = {
        "kind": GenerationFailureKind.INVALID_REQUEST,
        "retryable": False,
        "safe_message": "sanitized provider failure",
    }
    with pytest.raises(TypeError, match="CanonicalProviderErrorCode"):
        StructuredGenerationError(
            **common,
            status_code=400,
            provider_error_code="invalid_request_error",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="require status_code"):
        StructuredGenerationError(
            **common,
            provider_error_code=CanonicalProviderErrorCode.INVALID_REQUEST_ERROR,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        StructuredGenerationError(
            **common,
            status_code=400,
            provider_error_envelope_sha256="RAW_SECRET",
        )


@pytest.mark.parametrize("status", range(400, 600))
def test_http_failure_retryability_is_the_exact_closed_status_matrix(
    status: int,
) -> None:
    translated = classify_generation_exception(
        ModelHTTPError(status, "model", {"message": "RAW_SECRET"})
    )

    assert translated.retryable is (status in {408, 429} or 500 <= status <= 599)
    assert translated.status_code == status
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_invalid_structured_output_is_retryable_by_outer_owner() -> None:
    translated = classify_generation_exception(
        UnexpectedModelBehavior("invalid output with raw model text")
    )
    assert translated.kind is GenerationFailureKind.OUTPUT_INVALID
    assert translated.retryable is True
    assert (
        translated.output_failure_mode
        is StructuredOutputFailureMode.TYPED_OUTPUT_CONTRACT
    )
    assert translated.validation_issues == ()
    assert "raw model text" not in translated.safe_message


def test_validation_diagnostics_are_closed_bounded_and_drop_model_content() -> None:
    with pytest.raises(ValidationError) as caught:
        _DiagnosticOutput.model_validate(
            {
                "required_number": "RAW_SECRET_INPUT",
                "exact_choice": "RAW_SECRET_LITERAL",
                "RAW_SECRET_EXTRA_KEY": "RAW_SECRET_EXTRA_VALUE",
            }
        )
    raw = UnexpectedModelBehavior("RAW_SECRET_FRAMEWORK_MESSAGE")
    raw.__cause__ = caught.value

    translated = classify_generation_exception(raw, output_type=_DiagnosticOutput)

    assert (
        translated.output_failure_mode is StructuredOutputFailureMode.SCHEMA_VALIDATION
    )
    assert {
        (issue.category, issue.location) for issue in translated.validation_issues
    } == {
        (ValidationIssueCategory.WRONG_TYPE, ("required_number",)),
        (ValidationIssueCategory.LITERAL_OR_ENUM, ("exact_choice",)),
        (ValidationIssueCategory.MISSING, ("required_text",)),
        (ValidationIssueCategory.EXTRA_FIELD, ("unknown_field",)),
    }
    retained = repr(
        (
            translated.safe_message,
            translated.output_failure_mode,
            translated.validation_issues,
        )
    )
    assert "RAW_SECRET" not in retained


@pytest.mark.parametrize("reason_code", list(ValidationIssueReasonCode))
def test_closed_semantic_reason_codes_are_admitted_from_error_type_only(
    reason_code: ValidationIssueReasonCode,
) -> None:
    _SemanticDiagnosticOutput.reason_code = reason_code.value
    with pytest.raises(ValidationError) as caught:
        _SemanticDiagnosticOutput.model_validate(
            {"value": 1},
            strict=True,
        )
    raw = UnexpectedModelBehavior("RAW_SECRET_FRAMEWORK_MESSAGE")
    raw.__cause__ = caught.value

    translated = classify_generation_exception(
        raw,
        output_type=_SemanticDiagnosticOutput,
    )

    assert translated.validation_issues == (
        SanitizedValidationIssue(
            category=ValidationIssueCategory.SEMANTIC_CONSTRAINT,
            location=("root",),
            reason_code=reason_code,
        ),
    )
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_unrecognized_custom_error_type_is_not_retained_as_a_reason_code() -> None:
    _SemanticDiagnosticOutput.reason_code = "RAW_SECRET_UNTRUSTED_REASON"
    with pytest.raises(ValidationError) as caught:
        _SemanticDiagnosticOutput.model_validate({"value": 1}, strict=True)
    raw = UnexpectedModelBehavior("RAW_SECRET_FRAMEWORK_MESSAGE")
    raw.__cause__ = caught.value

    translated = classify_generation_exception(
        raw,
        output_type=_SemanticDiagnosticOutput,
    )

    assert translated.validation_issues == (
        SanitizedValidationIssue(
            category=ValidationIssueCategory.OTHER_VALIDATION,
            location=("root",),
        ),
    )
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_incomplete_tool_call_is_classified_by_type_without_parsing_text() -> None:
    translated = classify_generation_exception(
        IncompleteToolCall(
            "RAW_SECRET_TOKEN_LIMIT_MESSAGE",
            body="RAW_SECRET_PARTIAL_TOOL_ARGUMENTS",
        ),
        output_type=_DiagnosticOutput,
    )

    assert translated.kind is GenerationFailureKind.OUTPUT_INVALID
    assert translated.retryable is True
    assert (
        translated.output_failure_mode
        is StructuredOutputFailureMode.INCOMPLETE_TOOL_CALL
    )
    assert translated.validation_issues == ()
    assert "RAW_SECRET" not in repr(translated.__dict__)

    text_only = classify_generation_exception(
        UnexpectedModelBehavior(
            "IncompleteToolCall token limit RAW_SECRET_LOOKALIKE_TEXT"
        )
    )
    assert (
        text_only.output_failure_mode
        is StructuredOutputFailureMode.TYPED_OUTPUT_CONTRACT
    )


def test_content_filter_is_terminal_and_not_misclassified_as_invalid_output() -> None:
    translated = classify_generation_exception(
        ContentFilterError(
            "RAW_SECRET_FILTER_MESSAGE",
            body="RAW_SECRET_FILTER_BODY",
        ),
        output_type=_DiagnosticOutput,
    )

    assert translated.kind is GenerationFailureKind.CONTENT_REJECTED
    assert translated.retryable is False
    assert translated.safe_message == (
        "provider content filter rejected the model response"
    )
    assert translated.output_failure_mode is None
    assert translated.validation_issues == ()
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_model_api_connection_cause_is_retryable_and_sanitized() -> None:
    request = httpx.Request("POST", "https://example.invalid")
    connection = APIConnectionError(
        message="RAW_SECRET_CONNECTION_MESSAGE",
        request=request,
    )
    wrapped = ModelAPIError("model", "RAW_SECRET_WRAPPER_MESSAGE")
    wrapped.__cause__ = connection

    translated = classify_generation_exception(wrapped)

    assert translated.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE
    assert translated.retryable is True
    assert translated.safe_message == "provider API transport unavailable"
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_model_api_timeout_cause_takes_precedence_over_connection_base() -> None:
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
    wrapped = ModelAPIError("model", "RAW_SECRET_WRAPPER_MESSAGE")
    wrapped.__cause__ = timeout

    translated = classify_generation_exception(wrapped)

    assert translated.kind is GenerationFailureKind.TIMEOUT
    assert translated.retryable is True
    assert translated.safe_message == "provider API transport timed out"
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_other_model_api_failure_uses_closed_terminal_fallback() -> None:
    translated = classify_generation_exception(
        ModelAPIError("model", "RAW_SECRET_UNCLASSIFIED_API_MESSAGE")
    )

    assert translated.kind is GenerationFailureKind.UNKNOWN
    assert translated.retryable is False
    assert translated.safe_message == "unclassified provider API failure"
    assert "RAW_SECRET" not in repr(translated.__dict__)


@pytest.mark.parametrize(
    "raw",
    [
        httpx.ConnectError(
            "RAW_SECRET_CONNECT",
            request=httpx.Request("POST", "https://example.invalid"),
        ),
        ConnectionError("RAW_SECRET_BUILTIN_CONNECTION"),
    ],
)
def test_typed_raw_connection_failures_are_retryable(raw: BaseException) -> None:
    translated = classify_generation_exception(raw)

    assert translated.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE
    assert translated.retryable is True
    assert translated.safe_message == "provider transport unavailable"
    assert "RAW_SECRET" not in repr(translated.__dict__)


@pytest.mark.parametrize(
    "raw",
    [
        httpx.RemoteProtocolError("RAW_SECRET_REMOTE_PROTOCOL"),
        RuntimeError("RAW_SECRET_WRAPPER"),
    ],
)
def test_typed_remote_protocol_failures_are_retryable(raw: BaseException) -> None:
    if type(raw) is RuntimeError:
        import httpcore

        raw.__cause__ = httpcore.RemoteProtocolError(
            "RAW_SECRET_HTTPCORE_REMOTE_PROTOCOL"
        )

    translated = classify_generation_exception(raw)

    assert translated.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE
    assert translated.retryable is True
    assert translated.safe_message == (
        "provider response stream was interrupted remotely"
    )
    assert "RAW_SECRET" not in repr(translated.__dict__)


def test_local_protocol_failure_remains_terminal_and_fail_closed() -> None:
    translated = classify_generation_exception(
        httpx.LocalProtocolError("RAW_SECRET_LOCAL_PROTOCOL")
    )

    assert translated.kind is GenerationFailureKind.UNKNOWN
    assert translated.retryable is False
    assert translated.safe_message == "unclassified generation adapter failure"
    assert "RAW_SECRET" not in repr(translated.__dict__)


@pytest.mark.parametrize(
    "raw",
    [
        type("NetworkLookalikeError", (RuntimeError,), {})("RAW_SECRET"),
        type("TimeoutLookalikeError", (RuntimeError,), {})("RAW_SECRET"),
    ],
)
def test_exception_name_lookalikes_are_not_transport_retryable(
    raw: BaseException,
) -> None:
    translated = classify_generation_exception(raw)

    assert translated.kind is GenerationFailureKind.UNKNOWN
    assert translated.retryable is False
    assert translated.safe_message == "unclassified generation adapter failure"
    assert "RAW_SECRET" not in repr(translated.__dict__)


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "Not Allowed"},
        {"prompt": ""},
        {"output_tool_name": "bad tool"},
        {"max_output_tokens": 0},
        {"temperature": float("nan")},
    ],
)
def test_request_boundary_rejects_invalid_values(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        _request(**changes)

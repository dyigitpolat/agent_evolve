"""Offline exact-stack tests for redacted pre-transport request evidence."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from collections.abc import Awaitable, Callable
from importlib.metadata import version

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.providers.openrouter import OpenRouterProvider

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai import outbound_request_manifest as manifest
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    OpenRouterOutboundRequestManifestError,
    OpenRouterOutboundRequestManifestPublisher,
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationError,
    StructuredGenerationRequest,
    StructuredStreamLivenessPolicy,
)


class _SecretOutputContract(BaseModel):
    """SECRET_TOOL_DESCRIPTION_MUST_NOT_BE_RETAINED."""

    model_config = ConfigDict(extra="forbid", strict=True)

    secret_schema_marker: str
    score: int


def _request(
    *,
    suffix: str = "a",
    prompt: str = "SECRET_PROMPT_MUST_NOT_BE_RETAINED",
    max_output_tokens: int = 384_000,
) -> StructuredGenerationRequest[_SecretOutputContract]:
    return StructuredGenerationRequest(
        call_id=LLMCallId(f"call_outbound_manifest_{suffix}_000001"),
        operation="outbound_manifest_probe",
        prompt=prompt,
        output_type=_SecretOutputContract,
        output_tool_name="return_probe",
        max_output_tokens=max_output_tokens,
        provider_attempt_id=ProviderAttemptId(
            f"provider_attempt_outbound_manifest_{suffix}_000001"
        ),
    )


def _error_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": "SECRET_PROVIDER_ERROR_MUST_NOT_BE_RETAINED",
                "type": "invalid_request_error",
            }
        },
    )


def _build_generator(
    *,
    model_name: str,
    provider: dict[str, object],
    reasoning: OpenRouterReasoningConfig,
    rows: list[dict[str, object]],
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
    stream: bool = True,
    supports_forced_tool_choice: bool = True,
) -> PydanticAIStructuredGenerator:
    publisher = OpenRouterOutboundRequestManifestPublisher(rows.append)
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [publisher.httpx_request_hook]},
    )
    openai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="SECRET_API_KEY_MUST_NOT_BE_RETAINED",
        max_retries=0,
        http_client=http_client,
    )
    profile = OpenRouterProvider.model_profile(model_name)
    assert profile is not None
    if not supports_forced_tool_choice:
        profile = replace(
            profile,
            openai_supports_tool_choice_required=False,
        )
    model = OpenRouterModel(
        model_name,
        provider=OpenRouterProvider(openai_client=openai_client),
        profile=profile,
    )
    return PydanticAIStructuredGenerator(
        agent=Agent(model, retries=0),
        requested_model=model_name,
        provider_options=provider,
        reasoning_config=reasoning,
        supports_forced_tool_choice=supports_forced_tool_choice,
        owned_openai_client=openai_client,
        stream_liveness_policy=(
            StructuredStreamLivenessPolicy(
                first_event_timeout_ns=5_000_000_000,
                idle_timeout_ns=5_000_000_000,
            )
            if stream
            else None
        ),
        stream_progress_sink=(lambda _row: None) if stream else None,
        outbound_request_manifest_publisher=publisher,
    )


def _run_failing_attempt(
    generator: PydanticAIStructuredGenerator,
    request: StructuredGenerationRequest[_SecretOutputContract],
) -> StructuredGenerationError:
    async def scenario() -> StructuredGenerationError:
        try:
            await generator.generate_once(request)
        except StructuredGenerationError as exc:
            return exc
        finally:
            await generator.aclose()
        raise AssertionError("mock HTTP 400 unexpectedly succeeded")

    return asyncio.run(scenario())


def test_deepseek_max_reasoning_manifest_is_published_before_http_400() -> None:
    rows: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert len(rows) == 1, "manifest sink must return before transport"
        return _error_response(request)

    generator = _build_generator(
        model_name="deepseek/deepseek-v4-pro",
        provider={
            "only": ["streamlake"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        reasoning=OpenRouterReasoningConfig(max_tokens=384_000),
        rows=rows,
        handler=handler,
    )
    error = _run_failing_attempt(generator, _request())

    assert error.status_code == 400
    assert len(rows) == 1
    row = validate_openrouter_outbound_request_manifest_record(rows[0])
    assert row["provider_attempt_id"] == (
        "provider_attempt_outbound_manifest_a_000001"
    )
    assert row["settings"] == {
        "model": "deepseek/deepseek-v4-pro",
        "provider": {
            "only": ["streamlake"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "reasoning": {"max_tokens": 384_000},
        "usage": {"include": True},
        "stream": True,
        "stream_options": {"include_usage": True},
        "tool_choice": "required",
        "max_completion_tokens": 384_000,
        "temperature_hex": None,
        "response_format": None,
        "output_mode": "tool",
    }
    assert row["framework_versions"] == {
        name: version(name)
        for name in ("httpx", "openai", "pydantic", "pydantic-ai")
    }
    assert row["tool"]["requested_strict"] is False
    assert row["tool"]["wire_strict"] is None
    serialized = json.dumps(row, sort_keys=True)
    for forbidden in (
        "SECRET_PROMPT_MUST_NOT_BE_RETAINED",
        "SECRET_TOOL_DESCRIPTION_MUST_NOT_BE_RETAINED",
        "secret_schema_marker",
        "SECRET_PROVIDER_ERROR_MUST_NOT_BE_RETAINED",
        "SECRET_API_KEY_MUST_NOT_BE_RETAINED",
        "authorization",
    ):
        assert forbidden not in serialized


def test_standard_gpt_sol_manifest_has_xhigh_only_without_mode_or_pro() -> None:
    rows: list[dict[str, object]] = []
    generator = _build_generator(
        model_name="openai/gpt-5.6-sol",
        provider={
            "only": ["openai"],
            "allow_fallbacks": False,
        },
        reasoning=OpenRouterReasoningConfig(effort="xhigh"),
        rows=rows,
        handler=_error_response,
    )
    _run_failing_attempt(
        generator,
        _request(suffix="gpt", max_output_tokens=128_000),
    )

    row = validate_openrouter_outbound_request_manifest_record(rows[0])
    assert row["settings"]["model"] == "openai/gpt-5.6-sol"
    assert row["settings"]["reasoning"] == {"effort": "xhigh"}
    assert row["settings"]["max_completion_tokens"] == 128_000
    assert row["settings"]["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
    }
    assert all(row["forbidden_fields_absent"].values())
    assert "mode" not in row["settings"]
    assert "pro" not in row["settings"]
    assert "reasoning_effort" not in row["settings"]
    assert "mode" not in row["settings"]["reasoning"]
    assert "pro" not in row["settings"]["reasoning"]


def test_reasoning_route_without_tool_forcing_emits_authenticated_auto_choice() -> None:
    rows: list[dict[str, object]] = []
    generator = _build_generator(
        model_name="deepseek/deepseek-v4-pro",
        provider={"only": ["streamlake"], "allow_fallbacks": False},
        reasoning=OpenRouterReasoningConfig(effort="xhigh"),
        rows=rows,
        handler=_error_response,
        supports_forced_tool_choice=False,
    )

    _run_failing_attempt(generator, _request(suffix="auto"))

    row = validate_openrouter_outbound_request_manifest_record(rows[0])
    assert row["settings"]["tool_choice"] == "auto"
    assert row["settings"]["reasoning"] == {"effort": "xhigh"}


def test_contextvars_keep_concurrent_physical_attempts_joined() -> None:
    rows: list[dict[str, object]] = []
    arrived = 0
    both_arrived = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_arrived.set()
        await both_arrived.wait()
        return _error_response(request)

    generator = _build_generator(
        model_name="deepseek/deepseek-v4-pro",
        provider={"only": ["streamlake"], "allow_fallbacks": False},
        reasoning=OpenRouterReasoningConfig(max_tokens=384_000),
        rows=rows,
        handler=handler,
    )

    first = _request(suffix="one", prompt="concurrent-prompt-one")
    second = _request(suffix="two", prompt="concurrent-prompt-two")

    async def scenario() -> None:
        outcomes = await asyncio.gather(
            generator.generate_once(first),
            generator.generate_once(second),
            return_exceptions=True,
        )
        assert all(isinstance(value, StructuredGenerationError) for value in outcomes)
        await generator.aclose()

    asyncio.run(scenario())

    assert len(rows) == 2
    by_attempt = {row["provider_attempt_id"]: row for row in rows}
    assert by_attempt[first.provider_attempt_id.value]["message"][
        "content_sha256"
    ] == hashlib.sha256(first.prompt.encode()).hexdigest()
    assert by_attempt[second.provider_attempt_id.value]["message"][
        "content_sha256"
    ] == hashlib.sha256(second.prompt.encode()).hexdigest()


def test_duplicate_dispatch_of_same_physical_attempt_fails_before_transport() -> None:
    rows: list[dict[str, object]] = []
    transport_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        return _error_response(request)

    generator = _build_generator(
        model_name="deepseek/deepseek-v4-pro",
        provider={"only": ["streamlake"], "allow_fallbacks": False},
        reasoning=OpenRouterReasoningConfig(max_tokens=384_000),
        rows=rows,
        handler=handler,
        stream=False,
    )
    request = _request(suffix="duplicate")

    async def scenario() -> list[StructuredGenerationError]:
        errors = []
        for _ in range(2):
            with pytest.raises(StructuredGenerationError) as caught:
                await generator.generate_once(request)
            errors.append(caught.value)
        await generator.aclose()
        return errors

    errors = asyncio.run(scenario())
    assert len(errors) == 2
    assert transport_calls == 1
    assert len(rows) == 1


def _manual_body(
    request: StructuredGenerationRequest[_SecretOutputContract],
    *,
    prompt: str | None = None,
    mutate_schema: bool = False,
    extra_field: tuple[str, object] | None = None,
) -> dict[str, object]:
    logical = request.output_type.model_json_schema(mode="validation")
    schema = OpenAIJsonSchemaTransformer(copy.deepcopy(logical), strict=False).walk()
    if mutate_schema:
        schema["additionalProperties"] = True
    body: dict[str, object] = {
        "messages": [{"role": "user", "content": prompt or request.prompt}],
        "model": "deepseek/deepseek-v4-pro",
        "max_completion_tokens": request.max_output_tokens,
        "stream": False,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": request.output_tool_name,
                    "description": "description need not be retained",
                    "parameters": schema,
                },
            }
        ],
        "provider": {"only": ["streamlake"], "allow_fallbacks": False},
        "reasoning": {"max_tokens": 384_000},
        "usage": {"include": True},
    }
    if extra_field is not None:
        body[extra_field[0]] = extra_field[1]
    return body


def _resign_manifest(row: dict[str, object]) -> None:
    row.pop("outbound_request_manifest_sha256", None)
    row.pop("public_projection_sha256", None)
    row["public_projection_sha256"] = manifest._domain_sha256(
        manifest._PUBLIC_PROJECTION_DOMAIN,
        row,
    )
    row["outbound_request_manifest_sha256"] = manifest._domain_sha256(
        manifest._MANIFEST_DOMAIN,
        row,
    )


@pytest.mark.parametrize(
    ("body_changes", "message"),
    [
        ({"prompt": "different prompt"}, "prompt"),
        ({"mutate_schema": True}, "schema"),
        ({"extra_field": ("models", ["rogue/model"])}, "top-level"),
    ],
)
def test_prompt_schema_and_unknown_field_drift_fail_before_transport(
    body_changes: dict[str, object],
    message: str,
) -> None:
    suffix = {"prompt": "dp", "schema": "ds", "top-level": "dt"}[message]
    request = _request(suffix=suffix)
    rows: list[dict[str, object]] = []
    publisher = OpenRouterOutboundRequestManifestPublisher(rows.append)
    transport_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("invalid request reached transport")

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            event_hooks={"request": [publisher.httpx_request_hook]},
        ) as client:
            with publisher.bind(
                request,
                requested_model="deepseek/deepseek-v4-pro",
                provider={"only": ["streamlake"], "allow_fallbacks": False},
                reasoning={"max_tokens": 384_000},
                stream=False,
            ):
                with pytest.raises(OpenRouterOutboundRequestManifestError):
                    await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=_manual_body(request, **body_changes),
                    )

    asyncio.run(scenario())
    assert transport_calls == 0
    assert rows == []


def test_missing_context_and_tampered_persisted_record_fail_closed() -> None:
    publisher = OpenRouterOutboundRequestManifestPublisher(lambda _row: None)
    transport_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("unbound request reached transport")

    async def missing_context() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            event_hooks={"request": [publisher.httpx_request_hook]},
        ) as client:
            with pytest.raises(OpenRouterOutboundRequestManifestError):
                await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={},
                )

    asyncio.run(missing_context())
    assert transport_calls == 0

    rows: list[dict[str, object]] = []
    generator = _build_generator(
        model_name="deepseek/deepseek-v4-pro",
        provider={"only": ["streamlake"], "allow_fallbacks": False},
        reasoning=OpenRouterReasoningConfig(max_tokens=384_000),
        rows=rows,
        handler=_error_response,
    )
    _run_failing_attempt(generator, _request(suffix="tamper"))
    tampered = copy.deepcopy(rows[0])
    tampered["settings"]["model"] = "attacker/model"
    with pytest.raises(ValueError, match="self hash"):
        validate_openrouter_outbound_request_manifest_record(tampered)
    injected = copy.deepcopy(rows[0])
    injected["raw_body"] = "SECRET"
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_openrouter_outbound_request_manifest_record(injected)

    semantically_unsafe = copy.deepcopy(rows[0])
    semantically_unsafe["settings"]["provider"]["only"] = ["SECRET/route"]
    _resign_manifest(semantically_unsafe)
    with pytest.raises(ValueError, match="unsafe provider/reasoning"):
        validate_openrouter_outbound_request_manifest_record(semantically_unsafe)


def test_unsafe_bindings_and_duplicate_json_keys_fail_before_publication() -> None:
    rows: list[dict[str, object]] = []
    publisher = OpenRouterOutboundRequestManifestPublisher(rows.append)
    request = _request(suffix="unsafe")

    with pytest.raises(
        OpenRouterOutboundRequestManifestError,
        match="closed safe projection",
    ):
        with publisher.bind(
            request,
            requested_model="deepseek/deepseek-v4-pro",
            provider={
                "only": ["streamlake"],
                "allow_fallbacks": False,
                "api_key": "SECRET",
            },
            reasoning={"max_tokens": 384_000},
            stream=False,
        ):
            raise AssertionError("unsafe provider binding was admitted")

    raw_duplicate_body = (
        b'{"model":"deepseek/deepseek-v4-pro",'
        b'"model":"deepseek/deepseek-v4-pro"}'
    )
    with publisher.bind(
        request,
        requested_model="deepseek/deepseek-v4-pro",
        provider={"only": ["streamlake"], "allow_fallbacks": False},
        reasoning={"max_tokens": 384_000},
        stream=False,
    ):
        with pytest.raises(
            OpenRouterOutboundRequestManifestError,
            match="UTF-8 JSON object",
        ):
            asyncio.run(
                publisher.httpx_request_hook(
                    httpx.Request(
                        "POST",
                        "https://openrouter.ai/api/v1/chat/completions",
                        content=raw_duplicate_body,
                    )
                )
            )
    assert rows == []

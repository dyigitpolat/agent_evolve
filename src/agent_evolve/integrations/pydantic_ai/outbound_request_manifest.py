"""Content-free evidence for exact OpenRouter requests at the HTTP boundary.

Logical request evidence is not enough to diagnose provider rejections: both
Pydantic-AI and the OpenAI SDK transform model settings, messages, and tool
schemas before HTTP transport.  This module observes the fully serialized JSON
body in an HTTPX request hook, verifies that it still represents the bound
``StructuredGenerationRequest``, and synchronously publishes a redacted record
before the transport is allowed to send any bytes.

The record deliberately contains no headers, query text, prompt, tool
description, JSON schema, or raw body.  Exact byte and canonical semantic
commitments make those omitted values auditable without retaining them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.ports.structured_generator import (
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_UTF8_BYTES,
    StructuredGenerationRequest,
)


OPENROUTER_OUTBOUND_REQUEST_MANIFEST_SCHEMA_VERSION = 3
OPENROUTER_OUTBOUND_REQUEST_BOUNDARY_ID = (
    "openrouter_chat_completions_httpx_pre_transport_v3"
)
OPENROUTER_OUTBOUND_REQUEST_MANIFEST_HASH_ALGORITHM = (
    "sha256_domain_and_canonical_redacted_projection_v3"
)
OPENROUTER_OUTBOUND_PUBLIC_PROJECTION_HASH_ALGORITHM = (
    "sha256_domain_and_canonical_public_projection_v3"
)
OPENROUTER_OUTBOUND_TRANSPORT_CONTRACT_HASH_ALGORITHM = (
    "sha256_domain_and_canonical_transport_contract_v3"
)

_FRAMEWORK_PACKAGES = ("httpx", "openai", "pydantic", "pydantic-ai")
_TRANSPORT_CONTRACT = {
    "contract_version": 3,
    "hook_phase": "httpx_request_event_before_transport",
    "method": "POST",
    "scheme": "https",
    "host": "openrouter.ai",
    "port": None,
    "path": "/api/v1/chat/completions",
    "query_required_absent": True,
    "openai_sdk_retries": 0,
    "pydantic_ai_agent_retries": 0,
    "pydantic_ai_output_retries": 0,
    "pydantic_ai_request_limit": 1,
    "output_mode": "profile_selected_tool_or_native_json_schema",
    "sink_timing": "synchronous_before_transport_send",
    "raw_sensitive_values_persisted": False,
}
_FORBIDDEN_TOP_LEVEL_FIELDS = (
    "extra_body",
    "mode",
    "models",
    "plugins",
    "preset",
    "pro",
    "reasoning_effort",
    "reasoning_mode",
    "transforms",
    "web_search_options",
)
_FORBIDDEN_REASONING_FIELDS = (
    "enabled",
    "exclude",
    "mode",
    "pro",
)
_BASE_BODY_FIELDS = frozenset(
    {
        "max_completion_tokens",
        "messages",
        "model",
        "provider",
        "stream",
        "usage",
    }
)
_STRUCTURED_OUTPUT_MODES = frozenset({"tool", "native_json_schema"})
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPERATION_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_TOOL_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_PROVIDER_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MODEL_SLUG = re.compile(
    r"^[a-z0-9][a-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$"
)
_VERSION_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$")
_REASONING_EFFORTS = frozenset(
    {"xhigh", "high", "medium", "low", "minimal", "none"}
)
_MAX_WIRE_BODY_BYTES = 8 * 1_048_576
_MANIFEST_DOMAIN = b"agent-evolve:openrouter-outbound-manifest:v3\x00"
_PUBLIC_PROJECTION_DOMAIN = (
    b"agent-evolve:openrouter-outbound-public-projection:v3\x00"
)
_TRANSPORT_CONTRACT_DOMAIN = (
    b"agent-evolve:openrouter-outbound-transport-contract:v3\x00"
)


OpenRouterOutboundRequestManifestSink = Callable[[dict[str, object]], None]


class OpenRouterOutboundRequestManifestError(RuntimeError):
    """The exact outbound request could not be safely authenticated/published."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _canonical_json_copy(value: object, *, label: str) -> object:
    try:
        encoded = _canonical_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise OpenRouterOutboundRequestManifestError(
            f"{label} is not finite canonical JSON"
        ) from exc


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _framework_versions() -> dict[str, str]:
    """Project the closed dependency set; launch qualification binds exact values."""

    observed = {name: version(name) for name in _FRAMEWORK_PACKAGES}
    if any(
        type(value) is not str or _VERSION_TOKEN.fullmatch(value) is None
        for value in observed.values()
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound framework version is outside the closed token grammar"
        )
    return observed


def _closed_provider(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("provider routing must be an exact object")
    fields = frozenset(value)
    if fields not in (
        frozenset({"only", "allow_fallbacks"}),
        frozenset({"only", "allow_fallbacks", "require_parameters"}),
    ):
        raise ValueError("provider routing has unexpected fields")
    only = value["only"]
    if (
        type(only) is not list
        or not 1 <= len(only) <= 16
        or any(
            type(item) is not str or _PROVIDER_SLUG.fullmatch(item) is None
            for item in only
        )
        or len(set(only)) != len(only)
        or value["allow_fallbacks"] is not False
        or (
            "require_parameters" in value
            and value["require_parameters"] is not True
        )
    ):
        raise ValueError("provider routing violates the closed no-fallback policy")
    return dict(value)


def _closed_reasoning(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if type(value) is not dict or len(value) != 1:
        raise ValueError("reasoning must contain exactly one closed control")
    if frozenset(value) == {"effort"}:
        effort = value["effort"]
        if type(effort) is not str or effort not in _REASONING_EFFORTS:
            raise ValueError("reasoning effort is outside the closed enum")
    elif frozenset(value) == {"max_tokens"}:
        max_tokens = value["max_tokens"]
        if (
            type(max_tokens) is not int
            or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError("reasoning max_tokens is outside the generic bound")
    else:
        raise ValueError("reasoning contains an unsupported control")
    return dict(value)


def _transport_contract_sha256() -> str:
    return _domain_sha256(_TRANSPORT_CONTRACT_DOMAIN, _TRANSPORT_CONTRACT)


def _wire_output_schema(
    output_type: type[Any],
    *,
    json_schema_transformer: type[Any] | None = None,
    strict: bool = False,
) -> tuple[int, str, int, str]:
    """Return logical and pinned OpenAI-wire schema commitments.

    ``OpenRouterModel`` delegates schema projection to its resolved model
    profile.  Different model families can therefore serialize the same
    logical Pydantic type differently (for example, OpenAI-style references
    versus inline definitions).  Invoking the exact profile transformer is
    intentional: a dependency/source/model-profile change must be qualified
    rather than approximated by an ad-hoc title stripper.

    The default preserves the historical helper contract for callers that do
    not own a resolved model profile.  Production publishers always inject the
    selected OpenRouter profile's transformer.
    """

    try:
        from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
        from pydantic_ai.tools import GenerateToolJsonSchema

        transformer = (
            OpenAIJsonSchemaTransformer
            if json_schema_transformer is None
            else json_schema_transformer
        )
        if not isinstance(transformer, type):
            raise TypeError("json schema transformer must be a class")
        # The logical commitment must be byte-identical to the prequeue
        # request-evidence boundary.  That boundary deliberately records
        # Pydantic's ordinary validation schema, before any transport-specific
        # projection.
        logical = output_type.model_json_schema(
            mode="validation",
        )
        if type(logical) is not dict:
            raise TypeError("logical schema is not an exact object")
        logical_bytes = _canonical_bytes(logical)
        # ``ObjectOutputProcessor`` independently uses Pydantic-AI's
        # tool-specific generator before the model-profile transformer.  In
        # particular it removes property titles.  Reproduce that maintained
        # path for the wire commitment without conflating it with the logical
        # request schema above.
        transport_logical = output_type.model_json_schema(
            mode="validation",
            schema_generator=GenerateToolJsonSchema,
        )
        if type(transport_logical) is not dict:
            raise TypeError("transport logical schema is not an exact object")
        # Detach because schema transformers are allowed to mutate their input.
        # Preserve insertion order: strict transformers derive ordered
        # ``required`` arrays from property order, whereas replaying canonical
        # (key-sorted) JSON would authenticate a different wire schema.
        detached = deepcopy(transport_logical)
        # ObjectOutputProcessor promotes the root schema description (normally
        # the model docstring) to the function-tool description before the
        # model-specific schema transformer runs.
        detached.pop("description", None)
        if type(strict) is not bool:
            raise TypeError("strict must be an exact bool")
        wire = transformer(detached, strict=strict).walk()
        if type(wire) is not dict:
            raise TypeError("wire schema is not an exact object")
        wire_bytes = _canonical_bytes(wire)
    except Exception as exc:
        raise OpenRouterOutboundRequestManifestError(
            "structured output type cannot produce the pinned wire schema"
        ) from exc
    return (
        len(logical_bytes),
        _sha256(logical_bytes),
        len(wire_bytes),
        _sha256(wire_bytes),
    )


@dataclass(frozen=True, slots=True)
class _OutboundExpectation:
    publisher_identity: object
    call_id: str
    operation: str
    provider_attempt_id: str
    requested_model: str
    prompt_utf8_bytes: int
    prompt_sha256: str
    output_tool_name: str
    logical_schema_utf8_bytes: int
    logical_schema_sha256: str
    wire_schema_utf8_bytes: int
    wire_schema_sha256: str
    max_completion_tokens: int
    requested_temperature_hex: str | None
    provider: dict[str, object]
    reasoning: dict[str, object] | None
    stream: bool
    output_mode: str
    output_strict: bool
    expected_tool_choice: str


_BOUND_OUTBOUND_EXPECTATION: ContextVar[_OutboundExpectation | None] = ContextVar(
    "agent_evolve_openrouter_outbound_expectation",
    default=None,
)


class OpenRouterOutboundRequestManifestPublisher:
    """Concurrency-safe HTTPX hook and task-local request expectation owner."""

    def __init__(
        self,
        sink: OpenRouterOutboundRequestManifestSink,
        *,
        json_schema_transformer: type[Any] | None = None,
    ) -> None:
        if not callable(sink):
            raise TypeError("outbound request manifest sink must be callable")
        if json_schema_transformer is not None and not isinstance(
            json_schema_transformer, type
        ):
            raise TypeError("json_schema_transformer must be a class or None")
        self._sink = sink
        self._json_schema_transformer = json_schema_transformer
        self._identity = object()
        self._publication_lock = threading.Lock()
        self._published_provider_attempt_ids: set[str] = set()

    @contextmanager
    def bind(
        self,
        request: StructuredGenerationRequest[Any],
        *,
        requested_model: str,
        provider: Mapping[str, object],
        reasoning: Mapping[str, object] | None,
        stream: bool,
        output_mode: str = "tool",
        output_strict: bool = False,
        expected_tool_choice: str = "required",
    ) -> Iterator[None]:
        """Bind one exact physical-attempt identity to the current async task."""

        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(request)
        if request.provider_attempt_id is None:
            raise OpenRouterOutboundRequestManifestError(
                "outbound evidence requires a physical provider_attempt_id"
            )
        if (
            type(requested_model) is not str
            or _MODEL_SLUG.fullmatch(requested_model) is None
        ):
            raise ValueError("requested_model must be an OpenRouter model slug")
        if not isinstance(provider, Mapping):
            raise TypeError("provider must be a mapping")
        if reasoning is not None and not isinstance(reasoning, Mapping):
            raise TypeError("reasoning must be a mapping or None")
        if type(stream) is not bool:
            raise TypeError("stream must be an exact bool")
        if type(output_strict) is not bool:
            raise TypeError("output_strict must be an exact bool")
        if (
            type(expected_tool_choice) is not str
            or expected_tool_choice not in {"required", "auto"}
        ):
            raise ValueError("expected_tool_choice must be required or auto")
        if (
            type(output_mode) is not str
            or output_mode not in _STRUCTURED_OUTPUT_MODES
        ):
            raise ValueError("output_mode is outside the closed transport modes")
        if _BOUND_OUTBOUND_EXPECTATION.get() is not None:
            raise OpenRouterOutboundRequestManifestError(
                "nested outbound request evidence contexts are forbidden"
            )

        try:
            provider_json = _canonical_json_copy(dict(provider), label="provider")
            reasoning_json = (
                None
                if reasoning is None
                else _canonical_json_copy(dict(reasoning), label="reasoning")
            )
            provider_copy = _closed_provider(provider_json)
            reasoning_copy = _closed_reasoning(reasoning_json)
        except (TypeError, ValueError) as exc:
            raise OpenRouterOutboundRequestManifestError(
                "provider/reasoning settings violate the closed safe projection"
            ) from exc
        prompt_bytes = request.prompt.encode("utf-8", errors="strict")
        (
            logical_schema_utf8_bytes,
            logical_schema_sha256,
            wire_schema_utf8_bytes,
            wire_schema_sha256,
        ) = _wire_output_schema(
            request.output_type,
            json_schema_transformer=self._json_schema_transformer,
            strict=output_strict,
        )
        expectation = _OutboundExpectation(
            publisher_identity=self._identity,
            call_id=request.call_id.value,
            operation=request.operation,
            provider_attempt_id=request.provider_attempt_id.value,
            requested_model=requested_model,
            prompt_utf8_bytes=len(prompt_bytes),
            prompt_sha256=_sha256(prompt_bytes),
            output_tool_name=request.output_tool_name,
            logical_schema_utf8_bytes=logical_schema_utf8_bytes,
            logical_schema_sha256=logical_schema_sha256,
            wire_schema_utf8_bytes=wire_schema_utf8_bytes,
            wire_schema_sha256=wire_schema_sha256,
            max_completion_tokens=request.max_output_tokens,
            requested_temperature_hex=(
                None
                if request.temperature is None
                else float(request.temperature).hex()
            ),
            provider=provider_copy,
            reasoning=reasoning_copy,
            stream=stream,
            output_mode=output_mode,
            output_strict=output_strict,
            expected_tool_choice=expected_tool_choice,
        )
        token = _BOUND_OUTBOUND_EXPECTATION.set(expectation)
        try:
            yield
        finally:
            _BOUND_OUTBOUND_EXPECTATION.reset(token)

    async def httpx_request_hook(self, request: Any) -> None:
        """Validate and publish the final serialized body before HTTP transport."""

        expectation = _BOUND_OUTBOUND_EXPECTATION.get()
        if expectation is None or expectation.publisher_identity is not self._identity:
            raise OpenRouterOutboundRequestManifestError(
                "OpenRouter dispatch has no matching task-local attempt context"
            )
        record = await _manifest_record_from_httpx_request(request, expectation)
        provider_attempt_id = expectation.provider_attempt_id
        with self._publication_lock:
            if provider_attempt_id in self._published_provider_attempt_ids:
                raise OpenRouterOutboundRequestManifestError(
                    "duplicate transport dispatch for one provider_attempt_id"
                )
            # Reserve before publication. A sink that writes and then raises must
            # never cause the same physical identity to be published/sent again.
            self._published_provider_attempt_ids.add(provider_attempt_id)
            self._sink(record)


async def _manifest_record_from_httpx_request(
    request: Any,
    expectation: _OutboundExpectation,
) -> dict[str, object]:
    if getattr(request, "method", None) != _TRANSPORT_CONTRACT["method"]:
        raise OpenRouterOutboundRequestManifestError(
            "outbound request method violates the transport contract"
        )
    url = getattr(request, "url", None)
    if (
        url is None
        or getattr(url, "scheme", None) != _TRANSPORT_CONTRACT["scheme"]
        or getattr(url, "host", None) != _TRANSPORT_CONTRACT["host"]
        or getattr(url, "port", None) is not None
        or getattr(url, "path", None) != _TRANSPORT_CONTRACT["path"]
        or getattr(url, "query", b"") not in (b"", "")
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound URL violates the closed OpenRouter transport contract"
        )
    try:
        wire_body = await request.aread()
    except Exception as exc:
        raise OpenRouterOutboundRequestManifestError(
            "outbound JSON body could not be read before transport"
        ) from exc
    if (
        type(wire_body) is not bytes
        or not wire_body
        or len(wire_body) > _MAX_WIRE_BODY_BYTES
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound JSON body is empty or outside its evidence bound"
        )
    try:
        body = json.loads(
            wire_body,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OpenRouterOutboundRequestManifestError(
            "outbound body is not one UTF-8 JSON object"
        ) from exc
    if type(body) is not dict:
        raise OpenRouterOutboundRequestManifestError(
            "outbound body must be one exact JSON object"
        )

    expected_fields = set(_BASE_BODY_FIELDS)
    if expectation.output_mode == "tool":
        expected_fields.update(("tools", "tool_choice"))
    else:
        expected_fields.add("response_format")
    if expectation.reasoning is not None:
        expected_fields.add("reasoning")
    if expectation.requested_temperature_hex is not None:
        # Pydantic-AI intentionally drops sampling parameters for model
        # profiles that identify an active reasoning model. Absence is therefore
        # admitted and recorded; any present value must still match exactly.
        if "temperature" in body:
            expected_fields.add("temperature")
    if expectation.stream:
        expected_fields.add("stream_options")
    if frozenset(body) != frozenset(expected_fields):
        raise OpenRouterOutboundRequestManifestError(
            "outbound body has missing or unexpected top-level fields"
        )
    if any(name in body for name in _FORBIDDEN_TOP_LEVEL_FIELDS):
        raise OpenRouterOutboundRequestManifestError(
            "outbound body contains a forbidden routing/behavior field"
        )

    model = body["model"]
    if type(model) is not str or model != expectation.requested_model:
        raise OpenRouterOutboundRequestManifestError(
            "outbound model does not match the bound request"
        )
    max_completion_tokens = body["max_completion_tokens"]
    if (
        type(max_completion_tokens) is not int
        or max_completion_tokens != expectation.max_completion_tokens
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound completion budget does not match the bound request"
        )

    messages = body["messages"]
    if (
        type(messages) is not list
        or len(messages) != 1
        or type(messages[0]) is not dict
        or frozenset(messages[0]) != {"content", "role"}
        or messages[0]["role"] != "user"
        or type(messages[0]["content"]) is not str
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound messages violate the single-prompt contract"
        )
    prompt_bytes = messages[0]["content"].encode("utf-8", errors="strict")
    if (
        len(prompt_bytes) != expectation.prompt_utf8_bytes
        or _sha256(prompt_bytes) != expectation.prompt_sha256
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound prompt does not match the bound request"
        )

    output_count: int
    output_kind: str
    output_name: str
    output_description: str
    wire_strict: bool | None
    tool_choice: str | None
    response_format_kind: str | None
    if expectation.output_mode == "tool":
        tools = body["tools"]
        if (
            type(tools) is not list
            or len(tools) != 1
            or type(tools[0]) is not dict
            or frozenset(tools[0]) != {"function", "type"}
            or tools[0]["type"] != "function"
            or type(tools[0]["function"]) is not dict
        ):
            raise OpenRouterOutboundRequestManifestError(
                "outbound tools violate the single-function contract"
            )
        function = tools[0]["function"]
        expected_function_fields = {"description", "name", "parameters"}
        if expectation.output_strict:
            expected_function_fields.add("strict")
        if frozenset(function) != expected_function_fields:
            raise OpenRouterOutboundRequestManifestError(
                "outbound function tool has unexpected fields"
            )
        if (
            type(function["name"]) is not str
            or function["name"] != expectation.output_tool_name
            or type(function["description"]) is not str
            or type(function["parameters"]) is not dict
        ):
            raise OpenRouterOutboundRequestManifestError(
                "outbound function tool does not match the bound contract"
            )
        output_count = 1
        output_kind = "function"
        output_name = function["name"]
        output_description = function["description"]
        schema = function["parameters"]
        wire_strict = function.get("strict")
        tool_choice = body["tool_choice"]
        response_format_kind = None
        if tool_choice != expectation.expected_tool_choice:
            raise OpenRouterOutboundRequestManifestError(
                "outbound tool choice differs from the bound route capability"
            )
        if wire_strict != (True if expectation.output_strict else None):
            raise OpenRouterOutboundRequestManifestError(
                "outbound function strictness differs from the bound contract"
            )
    else:
        response_format = body["response_format"]
        if (
            type(response_format) is not dict
            or frozenset(response_format) != {"json_schema", "type"}
            or response_format["type"] != "json_schema"
            or type(response_format["json_schema"]) is not dict
        ):
            raise OpenRouterOutboundRequestManifestError(
                "outbound native response format violates the JSON-schema contract"
            )
        native = response_format["json_schema"]
        if frozenset(native) not in (
            frozenset({"name", "schema", "strict"}),
            frozenset({"description", "name", "schema", "strict"}),
        ):
            raise OpenRouterOutboundRequestManifestError(
                "outbound native JSON schema has unexpected fields"
            )
        if (
            type(native["name"]) is not str
            or native["name"] != expectation.output_tool_name
            or type(native["schema"]) is not dict
            or native["strict"] is not expectation.output_strict
            or (
                "description" in native
                and type(native["description"]) is not str
            )
        ):
            raise OpenRouterOutboundRequestManifestError(
                "outbound native JSON schema does not match the bound contract"
            )
        output_count = 0
        output_kind = "native_json_schema"
        output_name = native["name"]
        output_description = native.get("description", "")
        schema = native["schema"]
        wire_strict = expectation.output_strict
        tool_choice = None
        response_format_kind = "json_schema"

    description_bytes = output_description.encode("utf-8", errors="strict")
    schema_bytes = _canonical_bytes(schema)
    if (
        len(schema_bytes) != expectation.wire_schema_utf8_bytes
        or _sha256(schema_bytes) != expectation.wire_schema_sha256
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound structured schema does not match the bound output type: "
            f"expected={expectation.wire_schema_sha256}/"
            f"{expectation.wire_schema_utf8_bytes}, "
            f"observed={_sha256(schema_bytes)}/{len(schema_bytes)}"
        )

    if type(body["provider"]) is not dict or body["provider"] != expectation.provider:
        raise OpenRouterOutboundRequestManifestError(
            "outbound provider routing differs from the bound route"
        )
    reasoning = body.get("reasoning")
    if reasoning != expectation.reasoning or (
        reasoning is not None and type(reasoning) is not dict
    ):
        raise OpenRouterOutboundRequestManifestError(
            "outbound reasoning differs from the bound configuration"
        )
    if reasoning is not None:
        expected_reasoning_fields = (
            {"effort"} if "effort" in reasoning else {"max_tokens"}
        )
        if (
            frozenset(reasoning) != expected_reasoning_fields
            or any(name in reasoning for name in _FORBIDDEN_REASONING_FIELDS)
        ):
            raise OpenRouterOutboundRequestManifestError(
                "outbound reasoning has unexpected or forbidden fields"
            )
    if body["usage"] != {"include": True}:
        raise OpenRouterOutboundRequestManifestError(
            "outbound usage accounting is not explicitly enabled"
        )
    if type(body["stream"]) is not bool or body["stream"] is not expectation.stream:
        raise OpenRouterOutboundRequestManifestError(
            "outbound stream mode differs from the bound execution path"
        )
    stream_options = body.get("stream_options")
    if stream_options != ({"include_usage": True} if expectation.stream else None):
        raise OpenRouterOutboundRequestManifestError(
            "outbound stream usage options violate the closed contract"
        )

    temperature = body.get("temperature")
    temperature_hex: str | None = None
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
        ):
            raise OpenRouterOutboundRequestManifestError(
                "outbound temperature is not a finite number"
            )
        temperature_hex = float(temperature).hex()
        if temperature_hex != expectation.requested_temperature_hex:
            raise OpenRouterOutboundRequestManifestError(
                "outbound temperature differs from the bound request"
            )

    canonical_body = _canonical_bytes(body)
    record: dict[str, object] = {
        "schema_version": OPENROUTER_OUTBOUND_REQUEST_MANIFEST_SCHEMA_VERSION,
        "boundary_id": OPENROUTER_OUTBOUND_REQUEST_BOUNDARY_ID,
        "call_id": expectation.call_id,
        "operation": expectation.operation,
        "provider_attempt_id": expectation.provider_attempt_id,
        "transport": dict(_TRANSPORT_CONTRACT),
        "transport_contract_sha256": _transport_contract_sha256(),
        "framework_versions": _framework_versions(),
        "body": {
            "wire_utf8_bytes": len(wire_body),
            "wire_sha256": _sha256(wire_body),
            "canonical_utf8_bytes": len(canonical_body),
            "canonical_sha256": _sha256(canonical_body),
            "top_level_keys": sorted(body),
        },
        "message": {
            "count": 1,
            "role": "user",
            "content_utf8_bytes": len(prompt_bytes),
            "content_sha256": _sha256(prompt_bytes),
        },
        "tool": {
            "count": output_count,
            "type": output_kind,
            "name": output_name,
            "description_utf8_bytes": len(description_bytes),
            "description_sha256": _sha256(description_bytes),
            "schema_utf8_bytes": len(schema_bytes),
            "schema_sha256": _sha256(schema_bytes),
            "requested_strict": expectation.output_strict,
            "wire_strict": wire_strict,
        },
        "request_contract": {
            "logical_output_schema_utf8_bytes": (
                expectation.logical_schema_utf8_bytes
            ),
            "logical_output_schema_sha256": expectation.logical_schema_sha256,
            "wire_output_schema_utf8_bytes": expectation.wire_schema_utf8_bytes,
            "wire_output_schema_sha256": expectation.wire_schema_sha256,
            "requested_temperature_hex": expectation.requested_temperature_hex,
        },
        "settings": {
            "model": model,
            "provider": body["provider"],
            "reasoning": reasoning,
            "usage": body["usage"],
            "stream": body["stream"],
            "stream_options": stream_options,
            "output_mode": expectation.output_mode,
            "tool_choice": tool_choice,
            "max_completion_tokens": max_completion_tokens,
            "temperature_hex": temperature_hex,
            "response_format": response_format_kind,
        },
        "forbidden_fields_absent": {
            **{name: name not in body for name in _FORBIDDEN_TOP_LEVEL_FIELDS},
            **{
                f"reasoning.{name}": reasoning is None or name not in reasoning
                for name in _FORBIDDEN_REASONING_FIELDS
            },
        },
        "hash_algorithms": {
            "manifest": OPENROUTER_OUTBOUND_REQUEST_MANIFEST_HASH_ALGORITHM,
            "public_projection": (
                OPENROUTER_OUTBOUND_PUBLIC_PROJECTION_HASH_ALGORITHM
            ),
            "transport_contract": (
                OPENROUTER_OUTBOUND_TRANSPORT_CONTRACT_HASH_ALGORITHM
            ),
            "content": "sha256_exact_bytes_or_canonical_json_v1",
        },
    }
    record["public_projection_sha256"] = _domain_sha256(
        _PUBLIC_PROJECTION_DOMAIN,
        record,
    )
    record["outbound_request_manifest_sha256"] = _domain_sha256(
        _MANIFEST_DOMAIN,
        record,
    )
    return validate_openrouter_outbound_request_manifest_record(record)


_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "boundary_id",
        "call_id",
        "operation",
        "provider_attempt_id",
        "transport",
        "transport_contract_sha256",
        "framework_versions",
        "body",
        "message",
        "tool",
        "request_contract",
        "settings",
        "forbidden_fields_absent",
        "hash_algorithms",
        "public_projection_sha256",
        "outbound_request_manifest_sha256",
    }
)


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_temperature_hex(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{label} must be a hexadecimal float or None")
    try:
        decoded = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a hexadecimal float") from exc
    if (
        not math.isfinite(decoded)
        or not 0 <= decoded <= 2
        or decoded.hex() != value
    ):
        raise ValueError(f"{label} is outside the canonical temperature range")
    return value


def validate_openrouter_outbound_request_manifest_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate, authenticate, and detach one persisted manifest row."""

    if not isinstance(record, Mapping):
        raise TypeError("outbound request manifest must be a mapping")
    try:
        canonical = json.loads(_canonical_bytes(dict(record)))
    except (TypeError, ValueError) as exc:
        raise ValueError("outbound request manifest is not canonical JSON") from exc
    if type(canonical) is not dict or frozenset(canonical) != _MANIFEST_FIELDS:
        raise ValueError("outbound request manifest has unexpected fields")
    if (
        type(canonical["schema_version"]) is not int
        or canonical["schema_version"]
        != OPENROUTER_OUTBOUND_REQUEST_MANIFEST_SCHEMA_VERSION
        or canonical["boundary_id"] != OPENROUTER_OUTBOUND_REQUEST_BOUNDARY_ID
    ):
        raise ValueError("unsupported outbound request manifest contract")
    LLMCallId(canonical["call_id"])
    ProviderAttemptId(canonical["provider_attempt_id"])
    if (
        type(canonical["operation"]) is not str
        or _OPERATION_TOKEN.fullmatch(canonical["operation"]) is None
    ):
        raise ValueError("operation is outside the closed token grammar")
    if canonical["transport"] != _TRANSPORT_CONTRACT:
        raise ValueError("transport contract projection drifted")
    frameworks = canonical["framework_versions"]
    if (
        type(frameworks) is not dict
        or tuple(sorted(frameworks)) != tuple(sorted(_FRAMEWORK_PACKAGES))
        or any(
            type(value) is not str or _VERSION_TOKEN.fullmatch(value) is None
            for value in frameworks.values()
        )
    ):
        raise ValueError("framework version projection is outside its closed schema")
    if (
        _require_sha256(
            canonical["transport_contract_sha256"],
            label="transport_contract_sha256",
        )
        != _transport_contract_sha256()
    ):
        raise ValueError("transport_contract_sha256 does not authenticate contract")

    exact_nested_fields = {
        "body": {
            "wire_utf8_bytes",
            "wire_sha256",
            "canonical_utf8_bytes",
            "canonical_sha256",
            "top_level_keys",
        },
        "message": {
            "count",
            "role",
            "content_utf8_bytes",
            "content_sha256",
        },
        "tool": {
            "count",
            "type",
            "name",
            "description_utf8_bytes",
            "description_sha256",
            "schema_utf8_bytes",
            "schema_sha256",
            "requested_strict",
            "wire_strict",
        },
        "request_contract": {
            "logical_output_schema_utf8_bytes",
            "logical_output_schema_sha256",
            "wire_output_schema_utf8_bytes",
            "wire_output_schema_sha256",
            "requested_temperature_hex",
        },
        "settings": {
            "model",
            "provider",
            "reasoning",
            "usage",
            "stream",
            "stream_options",
            "output_mode",
            "tool_choice",
            "max_completion_tokens",
            "temperature_hex",
            "response_format",
        },
        "hash_algorithms": {
            "manifest",
            "public_projection",
            "transport_contract",
            "content",
        },
    }
    for name, fields in exact_nested_fields.items():
        value = canonical[name]
        if type(value) is not dict or frozenset(value) != fields:
            raise ValueError(f"{name} projection has unexpected fields")
    for container, names in (
        (canonical["body"], ("wire_sha256", "canonical_sha256")),
        (canonical["message"], ("content_sha256",)),
        (canonical["tool"], ("description_sha256", "schema_sha256")),
        (
            canonical["request_contract"],
            ("logical_output_schema_sha256", "wire_output_schema_sha256"),
        ),
    ):
        for name in names:
            _require_sha256(container[name], label=name)
    for container, names in (
        (canonical["body"], ("wire_utf8_bytes", "canonical_utf8_bytes")),
        (canonical["message"], ("content_utf8_bytes",)),
        (
            canonical["tool"],
            ("description_utf8_bytes", "schema_utf8_bytes"),
        ),
        (
            canonical["request_contract"],
            ("logical_output_schema_utf8_bytes", "wire_output_schema_utf8_bytes"),
        ),
    ):
        if any(type(container[name]) is not int or container[name] < 0 for name in names):
            raise ValueError("manifest byte lengths must be non-negative integers")
    body = canonical["body"]
    message = canonical["message"]
    tool = canonical["tool"]
    request_contract = canonical["request_contract"]
    settings = canonical["settings"]
    if (
        not 1 <= body["wire_utf8_bytes"] <= _MAX_WIRE_BODY_BYTES
        or not 1 <= body["canonical_utf8_bytes"] <= _MAX_WIRE_BODY_BYTES
        or not 1 <= message["content_utf8_bytes"] <= MAX_PROMPT_UTF8_BYTES
        or tool["description_utf8_bytes"] > MAX_PROMPT_UTF8_BYTES
        or not 1 <= tool["schema_utf8_bytes"] <= 1_048_576
        or not 1 <= request_contract["logical_output_schema_utf8_bytes"] <= 1_048_576
        or not 1 <= request_contract["wire_output_schema_utf8_bytes"] <= 1_048_576
    ):
        raise ValueError("manifest byte length exceeds its generic bound")
    output_mode = settings["output_mode"]
    if type(output_mode) is not str or output_mode not in _STRUCTURED_OUTPUT_MODES:
        raise ValueError("manifest output mode is outside the closed transport modes")
    tool_mode_valid = (
        output_mode == "tool"
        and tool["count"] == 1
        and tool["type"] == "function"
        and tool["wire_strict"]
        == (True if tool["requested_strict"] else None)
        and settings["tool_choice"] in {"required", "auto"}
        and settings["response_format"] is None
    )
    native_mode_valid = (
        output_mode == "native_json_schema"
        and tool["count"] == 0
        and tool["type"] == "native_json_schema"
        and tool["wire_strict"] is tool["requested_strict"]
        and settings["tool_choice"] is None
        and settings["response_format"] == "json_schema"
    )
    if (
        message["count"] != 1
        or message["role"] != "user"
        or not (tool_mode_valid or native_mode_valid)
        or type(tool["name"]) is not str
        or _TOOL_TOKEN.fullmatch(tool["name"]) is None
        or type(tool["requested_strict"]) is not bool
        or settings["usage"] != {"include": True}
    ):
        raise ValueError("manifest projections violate the structured request contract")
    if (
        tool["schema_sha256"] != request_contract["wire_output_schema_sha256"]
        or tool["schema_utf8_bytes"]
        != request_contract["wire_output_schema_utf8_bytes"]
    ):
        raise ValueError("tool schema projection does not join its request contract")

    model = settings["model"]
    if type(model) is not str or _MODEL_SLUG.fullmatch(model) is None:
        raise ValueError("model is outside the closed OpenRouter slug grammar")
    try:
        provider = _closed_provider(settings["provider"])
        reasoning = _closed_reasoning(settings["reasoning"])
    except ValueError as exc:
        raise ValueError("settings contain unsafe provider/reasoning values") from exc
    if provider != settings["provider"] or reasoning != settings["reasoning"]:
        raise ValueError("provider/reasoning settings are not canonical")
    if (
        type(settings["stream"]) is not bool
        or settings["stream_options"]
        != ({"include_usage": True} if settings["stream"] else None)
        or type(settings["max_completion_tokens"]) is not int
        or not 1 <= settings["max_completion_tokens"] <= MAX_OUTPUT_TOKENS
    ):
        raise ValueError("stream or completion settings are outside the closed contract")
    requested_temperature = _validated_temperature_hex(
        request_contract["requested_temperature_hex"],
        label="requested_temperature_hex",
    )
    wire_temperature = _validated_temperature_hex(
        settings["temperature_hex"],
        label="temperature_hex",
    )
    if wire_temperature is not None and wire_temperature != requested_temperature:
        raise ValueError("wire temperature differs from requested temperature")

    expected_top_level_fields = set(_BASE_BODY_FIELDS)
    if output_mode == "tool":
        expected_top_level_fields.update(("tools", "tool_choice"))
    else:
        expected_top_level_fields.add("response_format")
    if reasoning is not None:
        expected_top_level_fields.add("reasoning")
    if wire_temperature is not None:
        expected_top_level_fields.add("temperature")
    if settings["stream"]:
        expected_top_level_fields.add("stream_options")
    top_level_keys = body["top_level_keys"]
    if (
        type(top_level_keys) is not list
        or any(type(name) is not str for name in top_level_keys)
        or top_level_keys != sorted(expected_top_level_fields)
    ):
        raise ValueError("top-level key projection disagrees with settings")
    forbidden = canonical["forbidden_fields_absent"]
    expected_forbidden_keys = {
        *_FORBIDDEN_TOP_LEVEL_FIELDS,
        *(f"reasoning.{name}" for name in _FORBIDDEN_REASONING_FIELDS),
    }
    if (
        type(forbidden) is not dict
        or frozenset(forbidden) != expected_forbidden_keys
        or any(value is not True for value in forbidden.values())
    ):
        raise ValueError("forbidden-field absence projection is invalid")
    if canonical["hash_algorithms"] != {
        "manifest": OPENROUTER_OUTBOUND_REQUEST_MANIFEST_HASH_ALGORITHM,
        "public_projection": OPENROUTER_OUTBOUND_PUBLIC_PROJECTION_HASH_ALGORITHM,
        "transport_contract": (
            OPENROUTER_OUTBOUND_TRANSPORT_CONTRACT_HASH_ALGORITHM
        ),
        "content": "sha256_exact_bytes_or_canonical_json_v1",
    }:
        raise ValueError("manifest hash algorithm projection drifted")

    supplied_manifest = _require_sha256(
        canonical["outbound_request_manifest_sha256"],
        label="outbound_request_manifest_sha256",
    )
    without_manifest = dict(canonical)
    del without_manifest["outbound_request_manifest_sha256"]
    if supplied_manifest != _domain_sha256(_MANIFEST_DOMAIN, without_manifest):
        raise ValueError("outbound manifest self hash does not authenticate record")
    supplied_public = _require_sha256(
        canonical["public_projection_sha256"],
        label="public_projection_sha256",
    )
    public_projection = dict(without_manifest)
    del public_projection["public_projection_sha256"]
    if supplied_public != _domain_sha256(
        _PUBLIC_PROJECTION_DOMAIN,
        public_projection,
    ):
        raise ValueError("public projection hash does not authenticate record")
    return canonical


__all__ = [
    "OPENROUTER_OUTBOUND_PUBLIC_PROJECTION_HASH_ALGORITHM",
    "OPENROUTER_OUTBOUND_REQUEST_BOUNDARY_ID",
    "OPENROUTER_OUTBOUND_REQUEST_MANIFEST_HASH_ALGORITHM",
    "OPENROUTER_OUTBOUND_REQUEST_MANIFEST_SCHEMA_VERSION",
    "OPENROUTER_OUTBOUND_TRANSPORT_CONTRACT_HASH_ALGORITHM",
    "OpenRouterOutboundRequestManifestError",
    "OpenRouterOutboundRequestManifestPublisher",
    "OpenRouterOutboundRequestManifestSink",
    "validate_openrouter_outbound_request_manifest_record",
]

"""One-attempt Pydantic-AI/OpenRouter implementation of the generator port.

The OpenAI-compatible SDK is constructed with ``max_retries=0``.  This adapter
never sleeps and never retries; the application scheduler is the sole retry owner.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Any, Literal

from agent_evolve.domain.llm_task_queue import (
    CanonicalProviderErrorCode,
    MAX_VALIDATION_ISSUES,
    MAX_VALIDATION_LOCATION_DEPTH,
    SanitizedValidationIssue,
    StructuredOutputFailureMode,
    ValidationIssueCategory,
    ValidationIssueReasonCode,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    OutputT,
    StructuredGenerationError,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgressKind,
    StructuredStreamProgressSink,
)
from agent_evolve.infrastructure.stream_liveness import (
    AsyncioContentBlindStreamSupervisor,
    ContentBlindStreamSupervisor,
    StreamProgressMarker,
)
from agent_evolve.infrastructure.exception_provenance import (
    sanitized_exception_provenance,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    OpenRouterOutboundRequestManifestPublisher,
    OpenRouterOutboundRequestManifestSink,
)
from agent_evolve.integrations.pydantic_ai.json_schema_dialect import (
    OpenRouterJsonSchemaDialect,
    json_schema_transformer_for_dialect,
)


_RETRYABLE_HTTP_STATUS = frozenset({408, 429})
_RETRY_AFTER_MAX_SECONDS = 3_600.0
_MAX_EXCEPTION_NODES = 16
_MAX_SCHEMA_NODES = 256
_MAX_SCHEMA_PROPERTIES = 256
_SAFE_SCHEMA_PROPERTY = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_STREAM_CONTENT_IDENTITY_DOMAIN = (
    b"agent-evolve:structured-stream-semantic-content:v1\x00"
)
_PROVIDER_ERROR_ENVELOPE_DOMAIN = (
    b"agent-evolve:provider-error-redacted-envelope:v1\x00"
)
PROVIDER_ERROR_ENVELOPE_FINGERPRINT_ALGORITHM = (
    "sha256_domain_and_canonical_redacted_structure_v1"
)
PROVIDER_ERROR_ENVELOPE_DOMAIN_SHA256 = hashlib.sha256(
    _PROVIDER_ERROR_ENVELOPE_DOMAIN
).hexdigest()
STREAM_CONTENT_IDENTITY_ALGORITHM = "sha256_domain_and_length_framed_semantic_utf8_v1"
STREAM_CONTENT_IDENTITY_DOMAIN_SHA256 = hashlib.sha256(
    _STREAM_CONTENT_IDENTITY_DOMAIN
).hexdigest()


def _consume_close_task(task: "asyncio.Task[None]") -> None:
    with suppress(BaseException):
        task.exception()


OpenRouterReasoningEffort = Literal[
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
]
_OPENROUTER_REASONING_EFFORTS = frozenset(
    {"xhigh", "high", "medium", "low", "minimal", "none"}
)


class OpenRouterStructuredOutputMode(str, Enum):
    """Provider-capability-owned transport for one typed response."""

    TOOL = "tool"
    NATIVE_JSON_SCHEMA = "native_json_schema"
_MISSING_ERRORS = frozenset(
    {
        "missing",
        "missing_argument",
        "missing_keyword_only_argument",
        "missing_positional_only_argument",
    }
)
_MALFORMED_ARGUMENT_ERRORS = frozenset(
    {
        "arguments_type",
        "json_invalid",
        "json_type",
        "model_attributes_type",
        "model_type",
    }
)
_BOUND_ERRORS = frozenset(
    {
        "bytes_too_long",
        "bytes_too_short",
        "decimal_max_digits",
        "decimal_max_places",
        "decimal_whole_digits",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "multiple_of",
        "string_pattern_mismatch",
        "string_too_long",
        "string_too_short",
        "too_long",
        "too_short",
    }
)
_VALIDATION_REASON_ERROR_TYPES = frozenset(
    reason.value for reason in ValidationIssueReasonCode
)
_ABSENT = object()
_CAPABILITY_MISMATCH_ERROR_TYPES = frozenset(
    {
        "capability_mismatch",
        "no_compatible_endpoint",
        "no_endpoints_found",
        "unsupported_parameter",
        "unsupported_parameters",
    }
)


def _exact_dict_value(value: object, key: str) -> object:
    """Read one fixed key without accepting arbitrary mapping behavior."""

    if type(value) is not dict:
        return _ABSENT
    return value.get(key, _ABSENT)


def _redacted_value_kind(value: object) -> str:
    """Return a closed JSON-shape label without coercing or rendering a value."""

    if value is _ABSENT:
        return "absent"
    if value is None:
        return "null"
    if type(value) is dict:
        return "object"
    if type(value) is list:
        return "array"
    if type(value) is str:
        return "string"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    return "other"


def _fixed_provider_error_fields(body: object) -> tuple[tuple[str, object], ...]:
    """Read only fixed direct/wrapped typed fields from exact dictionaries."""

    wrapped_error = _exact_dict_value(body, "error")
    direct_metadata = _exact_dict_value(body, "metadata")
    wrapped_metadata = _exact_dict_value(wrapped_error, "metadata")
    return (
        (
            "body.metadata.error_type",
            _exact_dict_value(direct_metadata, "error_type"),
        ),
        ("body.error_type", _exact_dict_value(body, "error_type")),
        ("body.type", _exact_dict_value(body, "type")),
        ("body.code", _exact_dict_value(body, "code")),
        (
            "body.error.metadata.error_type",
            _exact_dict_value(wrapped_metadata, "error_type"),
        ),
        (
            "body.error.error_type",
            _exact_dict_value(wrapped_error, "error_type"),
        ),
        ("body.error.type", _exact_dict_value(wrapped_error, "type")),
        ("body.error.code", _exact_dict_value(wrapped_error, "code")),
    )


def _is_structured_capability_mismatch(body: object) -> bool:
    """Admit only an unambiguous finite typed capability error.

    Numeric HTTP codes and absent fields are ignored. If multiple typed string
    fields are present, every one must name the same closed capability family;
    a conflicting or unfamiliar typed value fails closed. Messages, metadata
    ``raw`` values, arbitrary keys, mappings, and object rendering are never
    inspected.
    """

    typed_values = tuple(
        value for _, value in _fixed_provider_error_fields(body) if type(value) is str
    )
    return bool(typed_values) and all(
        value in _CAPABILITY_MISMATCH_ERROR_TYPES for value in typed_values
    )


def _provider_error_diagnostics(
    status_code: int,
    body: object,
) -> tuple[CanonicalProviderErrorCode | None, str]:
    """Extract finite HTTP diagnostics without retaining provider content.

    Fixed direct and wrapped error-object paths are considered in
    priority-neutral fashion. OpenAI SDK status exceptions normally carry the
    direct error object after unwrapping a wire ``{"error": ...}`` response;
    other adapters can retain the wrapper. An unfamiliar string, a non-string,
    or conflicting admitted values yields no canonical code. The fingerprint
    authenticates only a redacted envelope of status and value *kinds*;
    provider values, arbitrary keys, messages, and raw payloads never enter its
    preimage.
    """

    wrapped_error = _exact_dict_value(body, "error")
    direct_metadata = _exact_dict_value(body, "metadata")
    wrapped_metadata = _exact_dict_value(wrapped_error, "metadata")
    fields = _fixed_provider_error_fields(body)
    admitted: list[tuple[str, CanonicalProviderErrorCode]] = []
    for source, raw_value in fields:
        if type(raw_value) is not str:
            continue
        try:
            code = CanonicalProviderErrorCode(raw_value)
        except ValueError:
            continue
        admitted.append((source, code))

    unique_codes = {code for _, code in admitted}
    provider_error_code = next(iter(unique_codes)) if len(unique_codes) == 1 else None
    redacted_envelope = {
        "schema_version": 1,
        "status_code": status_code,
        "body_kind": _redacted_value_kind(body),
        "direct_metadata_kind": _redacted_value_kind(direct_metadata),
        "wrapped_error_kind": _redacted_value_kind(wrapped_error),
        "wrapped_metadata_kind": _redacted_value_kind(wrapped_metadata),
        "fixed_field_kinds": {
            source: _redacted_value_kind(value) for source, value in fields
        },
        "admitted_provider_error_code": (
            None if provider_error_code is None else provider_error_code.value
        ),
        "admitted_sources": sorted(
            source for source, code in admitted if code is provider_error_code
        ),
        "conflicting_admitted_codes": len(unique_codes) > 1,
    }
    canonical = json.dumps(
        redacted_envelope,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    fingerprint = hashlib.sha256(
        _PROVIDER_ERROR_ENVELOPE_DOMAIN + canonical
    ).hexdigest()
    return provider_error_code, fingerprint


def _bounded_retry_after(value: object) -> float | None:
    if type(value) is not str:
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > 128:
        return None
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(stripped)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        seconds = (parsed - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return min(max(seconds, 0.0), _RETRY_AFTER_MAX_SECONDS)


def _retry_after_from_exception(exc: BaseException) -> float | None:
    """Read only the standard retry header from a bounded cause/context chain."""

    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                value = headers.get("retry-after")
            except Exception:
                value = None
            parsed = _bounded_retry_after(value)
            if parsed is not None:
                return parsed
        current = current.__cause__ or current.__context__
    return None


def _http_failure(
    status_code: int,
    body: object,
    exc: BaseException,
) -> StructuredGenerationError:
    provider_error_code, provider_error_envelope_sha256 = _provider_error_diagnostics(
        status_code, body
    )
    if status_code == 429:
        kind = GenerationFailureKind.RATE_LIMITED
        message = "provider rate limit"
    elif status_code == 408:
        kind = GenerationFailureKind.TIMEOUT
        message = "provider request timed out"
    elif 500 <= status_code <= 599:
        kind = (
            GenerationFailureKind.TIMEOUT
            if status_code == 504
            else GenerationFailureKind.PROVIDER_UNAVAILABLE
        )
        message = (
            "provider request timed out"
            if status_code == 504
            else "provider temporarily unavailable"
        )
    elif status_code in {409, 425}:
        kind = GenerationFailureKind.PROVIDER_UNAVAILABLE
        message = "provider request conflict is terminal"
    elif status_code == 404 and _is_structured_capability_mismatch(body):
        kind = GenerationFailureKind.CAPABILITY_MISMATCH
        message = "no model endpoint supports the requested capability set"
    elif status_code == 401:
        kind = GenerationFailureKind.AUTHENTICATION
        message = "provider authentication failed"
    elif status_code == 402:
        kind = GenerationFailureKind.PAYMENT_REQUIRED
        message = "provider payment or credit requirement failed"
    elif status_code == 403:
        kind = GenerationFailureKind.CONTENT_REJECTED
        message = "provider rejected the request"
    elif status_code in {400, 404, 422}:
        kind = GenerationFailureKind.INVALID_REQUEST
        message = "provider rejected invalid request parameters"
    else:
        kind = GenerationFailureKind.UNKNOWN
        message = "unclassified provider HTTP failure"
    return StructuredGenerationError(
        kind=kind,
        retryable=(status_code in _RETRYABLE_HTTP_STATUS or 500 <= status_code <= 599),
        safe_message=message,
        status_code=status_code,
        retry_after_seconds=_retry_after_from_exception(exc),
        provider_error_code=provider_error_code,
        provider_error_envelope_sha256=provider_error_envelope_sha256,
    )


def _validation_category(error_type: object) -> ValidationIssueCategory:
    if type(error_type) is not str:
        return ValidationIssueCategory.OTHER_VALIDATION
    if error_type in _MISSING_ERRORS:
        return ValidationIssueCategory.MISSING
    if error_type == "extra_forbidden":
        return ValidationIssueCategory.EXTRA_FIELD
    if error_type in {"enum", "literal_error"}:
        return ValidationIssueCategory.LITERAL_OR_ENUM
    if error_type in _MALFORMED_ARGUMENT_ERRORS:
        return ValidationIssueCategory.MALFORMED_ARGUMENTS
    if error_type in _BOUND_ERRORS:
        return ValidationIssueCategory.BOUNDS_OR_LENGTH
    if error_type in {"assertion_error", "value_error"} or (
        error_type in _VALIDATION_REASON_ERROR_TYPES
    ):
        return ValidationIssueCategory.SEMANTIC_CONSTRAINT
    if error_type.endswith(("_type", "_parsing")) or error_type in {
        "is_instance_of",
        "is_subclass_of",
    }:
        return ValidationIssueCategory.WRONG_TYPE
    return ValidationIssueCategory.OTHER_VALIDATION


def _validation_reason_code(
    error_type: object,
) -> ValidationIssueReasonCode | None:
    """Admit only a closed trusted validator error type.

    In particular, this function does not inspect Pydantic's free-form
    ``msg``, ``ctx``, or ``input`` fields, any of which can contain model data.
    """

    if type(error_type) is not str:
        return None
    try:
        return ValidationIssueReasonCode(error_type)
    except ValueError:
        return None


def _safe_schema_properties(output_type: type[Any] | None) -> frozenset[str]:
    """Collect bounded schema property names; values and descriptions are ignored."""

    if output_type is None:
        return frozenset()
    try:
        schema = output_type.model_json_schema()
    except Exception:
        return frozenset()
    if type(schema) is not dict:
        return frozenset()

    properties: set[str] = set()
    stack: list[tuple[object, int]] = [(schema, 0)]
    visited = 0
    while stack and visited < _MAX_SCHEMA_NODES:
        value, depth = stack.pop()
        visited += 1
        if depth > MAX_VALIDATION_LOCATION_DEPTH:
            continue
        if type(value) is dict:
            declared = value.get("properties")
            if type(declared) is dict:
                for name in declared:
                    if len(properties) >= _MAX_SCHEMA_PROPERTIES:
                        break
                    if type(name) is str and _SAFE_SCHEMA_PROPERTY.fullmatch(name):
                        properties.add(name)
            for child in value.values():
                if visited + len(stack) >= _MAX_SCHEMA_NODES:
                    break
                if type(child) in {dict, list}:
                    stack.append((child, depth + 1))
        elif type(value) is list:
            for child in value:
                if visited + len(stack) >= _MAX_SCHEMA_NODES:
                    break
                if type(child) in {dict, list}:
                    stack.append((child, depth + 1))
    return frozenset(properties)


def _safe_validation_location(
    raw_location: object,
    *,
    schema_properties: frozenset[str],
) -> tuple[str, ...]:
    if type(raw_location) not in {tuple, list}:
        return ("unknown_field",)
    safe: list[str] = []
    for segment in raw_location[:MAX_VALIDATION_LOCATION_DEPTH]:
        if type(segment) is str and segment in schema_properties:
            safe.append(segment)
        elif type(segment) is int and segment >= 0:
            safe.append("item")
        else:
            # Extra-field locations may contain arbitrary model output. Never
            # preserve those strings merely because Pydantic calls them a loc.
            safe.append("unknown_field")
    return tuple(safe) if safe else ("root",)


def _bounded_exception_nodes(exc: BaseException) -> tuple[BaseException, ...]:
    pending: deque[BaseException] = deque([exc])
    result: list[BaseException] = []
    seen: set[int] = set()
    while pending and len(result) < _MAX_EXCEPTION_NODES:
        current = pending.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException) and id(linked) not in seen:
                pending.append(linked)
        if type(current).__module__ == "builtins" and type(current).__name__ in {
            "BaseExceptionGroup",
            "ExceptionGroup",
        }:
            grouped = getattr(current, "exceptions", ())
            if type(grouped) is tuple:
                for linked in grouped[:_MAX_EXCEPTION_NODES]:
                    if isinstance(linked, BaseException) and id(linked) not in seen:
                        pending.append(linked)
    return tuple(result)


def _structured_output_diagnostics(
    exc: BaseException,
    *,
    output_type: type[Any] | None,
) -> tuple[StructuredOutputFailureMode, tuple[SanitizedValidationIssue, ...]]:
    from pydantic import ValidationError
    from pydantic_ai.exceptions import ToolRetryError

    schema_properties = _safe_schema_properties(output_type)
    details: list[object] = []
    validation_seen = False
    for current in _bounded_exception_nodes(exc):
        if isinstance(current, ValidationError):
            validation_seen = True
            try:
                details.extend(
                    current.errors(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    )[:MAX_VALIDATION_ISSUES]
                )
            except Exception:
                continue
        elif isinstance(current, ToolRetryError):
            content = current.tool_retry.content
            if type(content) is list:
                validation_seen = True
                details.extend(content[:MAX_VALIDATION_ISSUES])

    issues: list[SanitizedValidationIssue] = []
    seen_issues: set[
        tuple[
            ValidationIssueCategory,
            tuple[str, ...],
            ValidationIssueReasonCode | None,
        ]
    ] = set()
    for detail in details:
        if len(issues) >= MAX_VALIDATION_ISSUES:
            break
        if type(detail) is not dict:
            continue
        error_type = detail.get("type")
        issue = SanitizedValidationIssue(
            category=_validation_category(error_type),
            location=_safe_validation_location(
                detail.get("loc"),
                schema_properties=schema_properties,
            ),
            reason_code=_validation_reason_code(error_type),
        )
        identity = (issue.category, issue.location, issue.reason_code)
        if identity not in seen_issues:
            seen_issues.add(identity)
            issues.append(issue)

    mode = (
        StructuredOutputFailureMode.SCHEMA_VALIDATION
        if validation_seen
        else StructuredOutputFailureMode.TYPED_OUTPUT_CONTRACT
    )
    return mode, tuple(issues)


def _model_api_failure(exc: BaseException) -> StructuredGenerationError:
    """Classify a non-HTTP model API failure from typed, bounded causes only."""

    try:
        from openai import APIConnectionError, APITimeoutError
    except ImportError:  # pragma: no cover - OpenRouter installs the OpenAI SDK.
        APIConnectionError = ()  # type: ignore[assignment,misc]
        APITimeoutError = ()  # type: ignore[assignment,misc]

    nodes = _bounded_exception_nodes(exc)
    # OpenAI's SDK wraps exceptions raised by HTTPX request hooks in
    # ``APIConnectionError``.  Our pre-transport evidence hook is local and has
    # not sent provider bytes, so it must outrank that outer transport wrapper:
    # retrying cannot help and calling it provider unavailability hides the
    # actionable integrity failure.
    from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
        OpenRouterOutboundRequestManifestError,
    )

    if any(
        type(node) is OpenRouterOutboundRequestManifestError for node in nodes
    ):
        return StructuredGenerationError(
            kind=GenerationFailureKind.INVALID_REQUEST,
            retryable=False,
            safe_message="local outbound request evidence contract failed",
        )
    if any(isinstance(node, APITimeoutError) for node in nodes):
        return StructuredGenerationError(
            kind=GenerationFailureKind.TIMEOUT,
            retryable=True,
            safe_message="provider API transport timed out",
            retry_after_seconds=_retry_after_from_exception(exc),
        )
    if any(isinstance(node, APIConnectionError) for node in nodes):
        return StructuredGenerationError(
            kind=GenerationFailureKind.PROVIDER_UNAVAILABLE,
            retryable=True,
            safe_message="provider API transport unavailable",
            retry_after_seconds=_retry_after_from_exception(exc),
        )
    in_band_failure = _in_band_openai_api_failure(exc)
    if in_band_failure is not None:
        return in_band_failure
    return StructuredGenerationError(
        kind=GenerationFailureKind.UNKNOWN,
        retryable=False,
        safe_message="unclassified provider API failure",
        retry_after_seconds=_retry_after_from_exception(exc),
        exception_provenance=sanitized_exception_provenance(exc),
    )


def _in_band_openai_api_failure(
    exc: BaseException,
) -> StructuredGenerationError | None:
    """Admit only an exact integer status from an OpenAI SSE error body.

    OpenAI-compatible streams can raise the base ``openai.APIError`` for an
    error event delivered after the HTTP response has already become a stream.
    Such an exception has no ``APIStatusError.status_code``. OpenRouter's
    documented in-band envelope instead carries ``body.code``. We accept that
    code only from an exact dictionary and only in the HTTP error range, then
    reuse the ordinary value-redacting HTTP classifier. Messages and all
    unfamiliar/free-form bodies continue to fail closed as UNKNOWN.
    """

    try:
        from openai import APIError
    except ImportError:  # pragma: no cover - OpenRouter installs the OpenAI SDK.
        return None
    for node in _bounded_exception_nodes(exc):
        # The SDK's SSE error-event path raises the exact base APIError.
        # Subclasses such as APIStatusError and APIResponseValidationError have
        # distinct authoritative semantics; an in-body code must not override
        # them even when a hostile body contradicts their typed state.
        if type(node) is not APIError:
            continue
        try:
            body = BaseException.__getattribute__(node, "body")
        except BaseException:
            continue
        code = _exact_dict_value(body, "code")
        if type(code) is int and 400 <= code <= 599:
            return _http_failure(code, body, exc)
    return None


def classify_generation_exception(
    exc: BaseException,
    *,
    output_type: type[Any] | None = None,
    semantic_progress_observed: bool = False,
) -> StructuredGenerationError:
    """Translate framework/provider failures without retaining raw provider text."""

    if type(semantic_progress_observed) is not bool:
        raise TypeError("semantic_progress_observed must be an exact bool")

    # Imports stay local so importing the provider-neutral port remains cheap.
    from pydantic_ai.exceptions import (
        ContentFilterError,
        IncompleteToolCall,
        ModelAPIError,
        ModelHTTPError,
        UnexpectedModelBehavior,
        UsageLimitExceeded,
        UserError,
    )

    if isinstance(exc, StructuredGenerationError):
        return exc
    # Pydantic-AI 1.107.1 assumes every decoded OpenAI SSE value is a
    # ChatCompletionChunk before validating its runtime type.  Admit only our
    # exact, payload-free compatibility-boundary exception, including bounded
    # framework wrapping.
    # Generic AttributeError, ValidationError, and name lookalikes remain
    # terminal UNKNOWN failures.
    from agent_evolve.integrations.pydantic_ai.validated_openrouter_model import (
        InvalidOpenRouterStreamItemError,
    )

    if any(
        type(node) is InvalidOpenRouterStreamItemError
        for node in _bounded_exception_nodes(exc)
    ):
        return StructuredGenerationError(
            kind=GenerationFailureKind.PROVIDER_UNAVAILABLE,
            retryable=not semantic_progress_observed,
            safe_message=(
                "provider stream returned an invalid item"
                if not semantic_progress_observed
                else (
                    "provider stream returned an invalid item after "
                    "semantic progress"
                )
            ),
        )
    if isinstance(exc, ModelHTTPError):
        return _http_failure(exc.status_code, exc.body, exc)
    if isinstance(exc, ModelAPIError):
        return _model_api_failure(exc)
    if isinstance(exc, ContentFilterError):
        return StructuredGenerationError(
            kind=GenerationFailureKind.CONTENT_REJECTED,
            retryable=False,
            safe_message="provider content filter rejected the model response",
        )
    if isinstance(exc, IncompleteToolCall):
        return StructuredGenerationError(
            kind=GenerationFailureKind.OUTPUT_INVALID,
            retryable=True,
            safe_message="model stopped while emitting the typed output tool call",
            output_failure_mode=StructuredOutputFailureMode.INCOMPLETE_TOOL_CALL,
        )
    if isinstance(exc, UnexpectedModelBehavior):
        output_failure_mode, validation_issues = _structured_output_diagnostics(
            exc,
            output_type=output_type,
        )
        return StructuredGenerationError(
            kind=GenerationFailureKind.OUTPUT_INVALID,
            retryable=True,
            safe_message="model output violated the typed response contract",
            output_failure_mode=output_failure_mode,
            validation_issues=validation_issues,
        )
    if isinstance(exc, UsageLimitExceeded):
        return StructuredGenerationError(
            kind=GenerationFailureKind.OUTPUT_INVALID,
            retryable=False,
            safe_message="logical call exceeded its frozen usage limit",
        )
    if isinstance(exc, UserError):
        return StructuredGenerationError(
            kind=GenerationFailureKind.INVALID_REQUEST,
            retryable=False,
            safe_message="invalid Pydantic-AI request configuration",
        )

    in_band_failure = _in_band_openai_api_failure(exc)
    if in_band_failure is not None:
        return in_band_failure

    try:
        import httpx
    except ImportError:  # pragma: no cover - OpenRouter installs HTTPX.
        httpx_timeout_types: tuple[type[BaseException], ...] = ()
        httpx_network_types: tuple[type[BaseException], ...] = ()
    else:
        httpx_timeout_types = (httpx.TimeoutException,)
        httpx_network_types = (httpx.NetworkError,)

    nodes = _bounded_exception_nodes(exc)
    if any(isinstance(node, (TimeoutError, *httpx_timeout_types)) for node in nodes):
        return StructuredGenerationError(
            kind=GenerationFailureKind.TIMEOUT,
            retryable=True,
            safe_message="provider transport timed out",
            retry_after_seconds=_retry_after_from_exception(exc),
        )
    if any(isinstance(node, (ConnectionError, *httpx_network_types)) for node in nodes):
        return StructuredGenerationError(
            kind=GenerationFailureKind.PROVIDER_UNAVAILABLE,
            retryable=True,
            safe_message="provider transport unavailable",
            retry_after_seconds=_retry_after_from_exception(exc),
        )
    return StructuredGenerationError(
        kind=GenerationFailureKind.UNKNOWN,
        retryable=False,
        safe_message="unclassified generation adapter failure",
        exception_provenance=sanitized_exception_provenance(exc),
    )


def _usage_detail(details: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = details.get(name)
        if type(value) is int and value >= 0:
            return value
    return 0


def _cost(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not result.is_finite() or result < 0:
        return None
    return result


@dataclass(frozen=True, slots=True)
class OpenRouterReasoningConfig:
    """Validated OpenRouter reasoning control owned by the composition root.

    OpenRouter accepts either a qualitative effort level or an explicit reasoning
    token budget.  Keeping that choice in a frozen value object prevents arbitrary
    provider request fields from leaking through benchmark adapters.
    """

    effort: OpenRouterReasoningEffort | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        has_effort = self.effort is not None
        has_max_tokens = self.max_tokens is not None
        if has_effort == has_max_tokens:
            raise ValueError("exactly one of effort or max_tokens must be supplied")
        if has_effort and (
            type(self.effort) is not str
            or self.effort not in _OPENROUTER_REASONING_EFFORTS
        ):
            raise ValueError("effort is not a supported OpenRouter reasoning level")
        if has_max_tokens and (
            type(self.max_tokens) is not int or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")

    def to_model_setting(self) -> dict[str, object]:
        """Return the closed provider payload; no caller-owned mapping is reused."""

        if self.effort is not None:
            return {"effort": self.effort}
        return {"max_tokens": self.max_tokens}


class PydanticAIStructuredGenerator:
    """Reusable async Pydantic-AI agent that executes one attempt per call."""

    def __init__(
        self,
        *,
        agent: Any,
        requested_model: str,
        provider_options: Mapping[str, object] | None = None,
        reasoning_config: OpenRouterReasoningConfig | None = None,
        structured_output_mode: OpenRouterStructuredOutputMode = (
            OpenRouterStructuredOutputMode.TOOL
        ),
        structured_output_strict: bool = False,
        supports_forced_tool_choice: bool = True,
        owned_openai_client: Any | None = None,
        stream_liveness_policy: StructuredStreamLivenessPolicy | None = None,
        stream_progress_sink: StructuredStreamProgressSink | None = None,
        stream_supervisor: ContentBlindStreamSupervisor | None = None,
        outbound_request_manifest_publisher: (
            OpenRouterOutboundRequestManifestPublisher | None
        ) = None,
    ) -> None:
        if type(requested_model) is not str or not requested_model.strip():
            raise ValueError("requested_model must be non-empty")
        if (
            reasoning_config is not None
            and type(reasoning_config) is not OpenRouterReasoningConfig
        ):
            raise TypeError("reasoning_config must be an OpenRouterReasoningConfig")
        if type(structured_output_mode) is not OpenRouterStructuredOutputMode:
            raise TypeError(
                "structured_output_mode must be an exact "
                "OpenRouterStructuredOutputMode"
            )
        if type(structured_output_strict) is not bool:
            raise TypeError("structured_output_strict must be an exact bool")
        if type(supports_forced_tool_choice) is not bool:
            raise TypeError("supports_forced_tool_choice must be an exact bool")
        if stream_liveness_policy is not None and (
            type(stream_liveness_policy) is not StructuredStreamLivenessPolicy
        ):
            raise TypeError(
                "stream_liveness_policy must be a StructuredStreamLivenessPolicy"
            )
        if stream_progress_sink is not None and not callable(stream_progress_sink):
            raise TypeError("stream_progress_sink must be callable or None")
        if stream_liveness_policy is None and (
            stream_progress_sink is not None or stream_supervisor is not None
        ):
            raise ValueError(
                "stream progress dependencies require a stream liveness policy"
            )
        if stream_supervisor is not None and not isinstance(
            stream_supervisor,
            ContentBlindStreamSupervisor,
        ):
            raise TypeError(
                "stream_supervisor must implement ContentBlindStreamSupervisor"
            )
        if (
            outbound_request_manifest_publisher is not None
            and type(outbound_request_manifest_publisher)
            is not OpenRouterOutboundRequestManifestPublisher
        ):
            raise TypeError(
                "outbound_request_manifest_publisher must be an exact "
                "OpenRouterOutboundRequestManifestPublisher or None"
            )
        self._agent = agent
        self.requested_model = requested_model
        self._provider_options = dict(provider_options or {"allow_fallbacks": True})
        self._reasoning_config = reasoning_config
        self._structured_output_mode = structured_output_mode
        self._structured_output_strict = structured_output_strict
        self._supports_forced_tool_choice = supports_forced_tool_choice
        self._owned_openai_client = owned_openai_client
        self._stream_liveness_policy = stream_liveness_policy
        self._stream_progress_sink = stream_progress_sink
        self._outbound_request_manifest_publisher = outbound_request_manifest_publisher
        self._transport_retired = False
        self._stream_supervisor = (
            AsyncioContentBlindStreamSupervisor(
                retirement_operation=self._retire_owned_transport,
            )
            if stream_liveness_policy is not None and stream_supervisor is None
            else stream_supervisor
        )

    @property
    def stream_liveness_policy(self) -> StructuredStreamLivenessPolicy | None:
        """Return the immutable content-blind policy, if streaming is enabled."""

        return self._stream_liveness_policy

    @classmethod
    def openrouter(
        cls,
        *,
        api_key: str,
        model_name: str,
        max_connections: int,
        timeout_seconds: float = 90.0,
        provider_options: Mapping[str, object] | None = None,
        reasoning_config: OpenRouterReasoningConfig | None = None,
        structured_output_mode: OpenRouterStructuredOutputMode = (
            OpenRouterStructuredOutputMode.TOOL
        ),
        structured_output_strict: bool = False,
        supports_forced_tool_choice: bool = True,
        json_schema_dialect: OpenRouterJsonSchemaDialect = (
            OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT
        ),
        app_title: str = "AgentEvolve research",
        stream_liveness_policy: StructuredStreamLivenessPolicy | None = None,
        stream_progress_sink: StructuredStreamProgressSink | None = None,
        outbound_request_manifest_sink: (
            OpenRouterOutboundRequestManifestSink | None
        ) = None,
    ) -> "PydanticAIStructuredGenerator":
        """Build the production adapter with SDK retries explicitly disabled."""

        if type(api_key) is not str or not api_key:
            raise ValueError("api_key must be supplied at the composition root")
        if type(model_name) is not str or "/" not in model_name:
            raise ValueError("model_name must be an OpenRouter model slug")
        if type(max_connections) is not int or not 1 <= max_connections <= 256:
            raise ValueError("max_connections must lie in [1,256]")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or not 1 <= float(timeout_seconds) <= 600
        ):
            raise ValueError("timeout_seconds must lie in [1,600]")
        if stream_liveness_policy is not None and (
            type(stream_liveness_policy) is not StructuredStreamLivenessPolicy
        ):
            raise TypeError(
                "stream_liveness_policy must be a StructuredStreamLivenessPolicy"
            )
        if stream_progress_sink is not None and not callable(stream_progress_sink):
            raise TypeError("stream_progress_sink must be callable or None")
        if outbound_request_manifest_sink is not None and not callable(
            outbound_request_manifest_sink
        ):
            raise TypeError("outbound_request_manifest_sink must be callable or None")
        if type(structured_output_mode) is not OpenRouterStructuredOutputMode:
            raise TypeError(
                "structured_output_mode must be an exact "
                "OpenRouterStructuredOutputMode"
            )
        if type(structured_output_strict) is not bool:
            raise TypeError("structured_output_strict must be an exact bool")
        if type(supports_forced_tool_choice) is not bool:
            raise TypeError("supports_forced_tool_choice must be an exact bool")
        if type(json_schema_dialect) is not OpenRouterJsonSchemaDialect:
            raise TypeError(
                "json_schema_dialect must be an exact "
                "OpenRouterJsonSchemaDialect"
            )

        import httpx
        from openai import AsyncOpenAI
        from pydantic_ai import Agent
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        from agent_evolve.integrations.pydantic_ai.validated_openrouter_model import (
            ValidatedOpenRouterModel,
        )

        # A streamed response's read liveness is owned by the content-blind
        # first-event/idle supervisor.  Connect, pool, and write operations
        # retain bounded SDK timeouts, while reads have no competing fixed
        # total boundary that could censor a healthy long generation.
        transport_timeout = (
            httpx.Timeout(
                connect=float(timeout_seconds),
                pool=float(timeout_seconds),
                write=float(timeout_seconds),
                read=None,
            )
            if stream_liveness_policy is not None
            else httpx.Timeout(float(timeout_seconds))
        )
        resolved_profile = OpenRouterProvider.model_profile(model_name)
        if resolved_profile is None:
            raise ValueError("OpenRouter model has no resolvable execution profile")
        if (
            structured_output_mode
            is OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
        ):
            resolved_profile = replace(
                resolved_profile,
                supports_json_schema_output=True,
            )
        if not supports_forced_tool_choice:
            resolved_profile = replace(
                resolved_profile,
                openai_supports_tool_choice_required=False,
            )
        resolved_profile = replace(
            resolved_profile,
            json_schema_transformer=json_schema_transformer_for_dialect(
                resolved_profile.json_schema_transformer,
                json_schema_dialect,
            ),
        )
        outbound_publisher = (
            None
            if outbound_request_manifest_sink is None
            else OpenRouterOutboundRequestManifestPublisher(
                outbound_request_manifest_sink,
                json_schema_transformer=(
                    resolved_profile.json_schema_transformer
                ),
            )
        )
        http_client = httpx.AsyncClient(
            timeout=transport_timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            event_hooks=(
                None
                if outbound_publisher is None
                else {"request": [outbound_publisher.httpx_request_hook]}
            ),
        )
        openai_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,
            http_client=http_client,
            default_headers={"X-Title": app_title},
        )
        provider = OpenRouterProvider(openai_client=openai_client)
        model = ValidatedOpenRouterModel(
            model_name,
            provider=provider,
            profile=resolved_profile,
        )
        # One integer freezes both tool and output retry budgets at zero on the
        # maintained Pydantic-AI v1 API.  The application queue remains the
        # only retry owner.
        agent = Agent(model, retries=0)
        return cls(
            agent=agent,
            requested_model=model_name,
            provider_options=provider_options,
            reasoning_config=reasoning_config,
            structured_output_mode=structured_output_mode,
            structured_output_strict=structured_output_strict,
            supports_forced_tool_choice=supports_forced_tool_choice,
            owned_openai_client=openai_client,
            stream_liveness_policy=stream_liveness_policy,
            stream_progress_sink=stream_progress_sink,
            outbound_request_manifest_publisher=outbound_publisher,
        )

    async def aclose(self) -> None:
        self._transport_retired = True
        if self._owned_openai_client is not None:
            client = self._owned_openai_client
            self._owned_openai_client = None
            if self._stream_liveness_policy is None:
                await client.close()
                return
            close_task = asyncio.create_task(client.close())
            done, _ = await asyncio.wait(
                (close_task,),
                timeout=(
                    self._stream_liveness_policy.cleanup_policy.transport_retire_timeout_ns
                    / 1_000_000_000
                ),
                return_when=asyncio.ALL_COMPLETED,
            )
            if close_task in done:
                # Preserve an immediate close failure for explicit shutdown.
                await close_task
                return
            close_task.cancel()
            close_task.add_done_callback(_consume_close_task)

    async def _retire_owned_transport(self) -> None:
        """Irreversibly reject new calls, then close the detached owned client."""

        self._transport_retired = True
        if self._owned_openai_client is not None:
            client = self._owned_openai_client
            self._owned_openai_client = None
            await client.close()

    async def __aenter__(self) -> "PydanticAIStructuredGenerator":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def generate_once(
        self, request: StructuredGenerationRequest[OutputT]
    ) -> StructuredGenerationResponse[OutputT]:
        from pydantic_ai import NativeOutput, ToolOutput
        from pydantic_ai.messages import ModelResponse
        from pydantic_ai.usage import UsageLimits

        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(request)
        if self._transport_retired:
            raise StructuredGenerationError(
                kind=GenerationFailureKind.CANCELLED,
                retryable=False,
                safe_message=(
                    "provider transport is retired after incomplete stream cleanup"
                ),
            )
        settings: dict[str, object] = {
            "max_tokens": request.max_output_tokens,
            "openrouter_provider": dict(self._provider_options),
            "openrouter_usage": {"include": True},
        }
        if request.temperature is not None:
            settings["temperature"] = float(request.temperature)
        if self._reasoning_config is not None:
            settings["openrouter_reasoning"] = self._reasoning_config.to_model_setting()

        started = time.perf_counter_ns()
        sanitized_failure: StructuredGenerationError | None = None
        semantic_progress_observed = False
        try:
            output_type = (
                ToolOutput(
                    request.output_type,
                    name=request.output_tool_name,
                    strict=self._structured_output_strict,
                )
                if self._structured_output_mode
                is OpenRouterStructuredOutputMode.TOOL
                else NativeOutput(
                    request.output_type,
                    name=request.output_tool_name,
                    strict=self._structured_output_strict,
                )
            )
            usage_limits = UsageLimits(
                request_limit=1,
                output_tokens_limit=request.max_output_tokens,
            )
            if self._stream_liveness_policy is None:
                if self._outbound_request_manifest_publisher is None:
                    result = await self._agent.run(
                        request.prompt,
                        output_type=output_type,
                        model_settings=settings,
                        usage_limits=usage_limits,
                    )
                else:
                    with self._outbound_request_manifest_publisher.bind(
                        request,
                        requested_model=self.requested_model,
                        provider=self._provider_options,
                        reasoning=(
                            None
                            if self._reasoning_config is None
                            else self._reasoning_config.to_model_setting()
                        ),
                        stream=False,
                        output_mode=self._structured_output_mode.value,
                        output_strict=self._structured_output_strict,
                        expected_tool_choice=(
                            "required"
                            if self._supports_forced_tool_choice
                            else "auto"
                        ),
                    ):
                        result = await self._agent.run(
                            request.prompt,
                            output_type=output_type,
                            model_settings=settings,
                            usage_limits=usage_limits,
                        )
            else:
                assert self._stream_supervisor is not None
                content_hasher = hashlib.sha256(_STREAM_CONTENT_IDENTITY_DOMAIN)
                cumulative_content_utf8_bytes = 0

                async def streamed_operation(mark_progress: StreamProgressMarker):
                    async def handle_events(_context: Any, events: Any) -> None:
                        nonlocal cumulative_content_utf8_bytes
                        nonlocal semantic_progress_observed
                        async for event in events:
                            projection = _stream_progress_projection(event)
                            if projection is not None:
                                kind, channel, fragments = projection
                                event_content_utf8_bytes = 0
                                for field, content in fragments:
                                    field_bytes = field.encode("ascii", errors="strict")
                                    content_bytes = content.encode(
                                        "utf-8", errors="strict"
                                    )
                                    event_content_utf8_bytes += len(content_bytes)
                                    content_hasher.update(
                                        len(field_bytes).to_bytes(2, "big")
                                    )
                                    content_hasher.update(field_bytes)
                                    content_hasher.update(
                                        len(content_bytes).to_bytes(8, "big")
                                    )
                                    content_hasher.update(content_bytes)
                                cumulative_content_utf8_bytes += (
                                    event_content_utf8_bytes
                                )
                                mark_progress(
                                    kind,
                                    channel,
                                    event_content_utf8_bytes=(event_content_utf8_bytes),
                                    cumulative_content_utf8_bytes=(
                                        cumulative_content_utf8_bytes
                                    ),
                                    rolling_content_sha256=(content_hasher.hexdigest()),
                                )
                                # This flag is deliberately attempt-local and
                                # content-blind.  Once any supported semantic
                                # model event has been durably projected, an
                                # invalid later decoded SSE item must not cause
                                # an exact-payload replay of an ambiguous
                                # partial generation.
                                semantic_progress_observed = True

                    if self._outbound_request_manifest_publisher is None:
                        result = await self._agent.run(
                            request.prompt,
                            output_type=output_type,
                            model_settings=settings,
                            usage_limits=usage_limits,
                            event_stream_handler=handle_events,
                        )
                    else:
                        with self._outbound_request_manifest_publisher.bind(
                            request,
                            requested_model=self.requested_model,
                            provider=self._provider_options,
                            reasoning=(
                                None
                                if self._reasoning_config is None
                                else self._reasoning_config.to_model_setting()
                            ),
                            stream=True,
                            output_mode=self._structured_output_mode.value,
                            output_strict=self._structured_output_strict,
                            expected_tool_choice=(
                                "required"
                                if self._supports_forced_tool_choice
                                else "auto"
                            ),
                        ):
                            result = await self._agent.run(
                                request.prompt,
                                output_type=output_type,
                                model_settings=settings,
                                usage_limits=usage_limits,
                                event_stream_handler=handle_events,
                            )
                    # Pydantic-AI's FinalResultEvent means that the output tool
                    # has been selected; tool argument deltas may follow it.
                    # Only the return from Agent.run proves the stream is done
                    # and a typed output is now available.  Keep this local
                    # marker inside liveness supervision so completion itself
                    # remains subject to the idle/absolute policy.
                    _ = result.output
                    mark_progress(
                        StructuredStreamProgressKind.STREAM_COMPLETED,
                        StructuredStreamChannel.OTHER,
                        event_content_utf8_bytes=0,
                        cumulative_content_utf8_bytes=(cumulative_content_utf8_bytes),
                        rolling_content_sha256=content_hasher.hexdigest(),
                    )
                    return result

                result = await self._stream_supervisor.run(
                    streamed_operation,
                    call_id=request.call_id.value,
                    provider_attempt_id=(
                        None
                        if request.provider_attempt_id is None
                        else request.provider_attempt_id.value
                    ),
                    policy=self._stream_liveness_policy,
                    progress_sink=self._stream_progress_sink,
                )
        except Exception as exc:
            sanitized_failure = classify_generation_exception(
                exc,
                output_type=request.output_type,
                semantic_progress_observed=semantic_progress_observed,
            )
            # ``raise ... from None`` suppresses display of an active exception,
            # but Python still retains it in ``__context__`` together with its
            # traceback-frame locals.  A malformed decoded stream item can be
            # arbitrary provider content, so detach the admitted sanitized
            # failure while still inside the classification boundary and raise
            # it only after the raw exception scope has ended.  Clearing the
            # returned failure as well covers an already-sanitized exception
            # that ``classify_generation_exception`` returned unchanged.
            sanitized_failure.__traceback__ = None
            sanitized_failure.__cause__ = None
            sanitized_failure.__context__ = None
        if sanitized_failure is not None:
            raise sanitized_failure from None
        latency_ns = time.perf_counter_ns() - started

        response = result.response
        if not isinstance(
            response, ModelResponse
        ):  # pragma: no cover - framework guard.
            raise StructuredGenerationError(
                kind=GenerationFailureKind.UNKNOWN,
                retryable=False,
                safe_message="Pydantic-AI returned no terminal model response",
            )
        # ``usage`` became a property late in Pydantic-AI v1.  Admit the
        # maintained API while retaining compatibility with older injected
        # test doubles that exposed the historical method.
        usage = result.usage
        if not hasattr(usage, "input_tokens") and callable(usage):
            usage = usage()
        details = usage.details if isinstance(usage.details, Mapping) else {}
        provider_details = (
            response.provider_details
            if isinstance(response.provider_details, Mapping)
            else {}
        )
        resolved_provider = provider_details.get("downstream_provider")
        if type(resolved_provider) is not str or not resolved_provider.strip():
            resolved_provider = response.provider_name or "unknown"
        resolved_model = response.model_name or self.requested_model
        return StructuredGenerationResponse(
            value=result.output,
            requested_model=self.requested_model,
            resolved_model=resolved_model,
            resolved_provider=resolved_provider,
            provider_response_id=response.provider_response_id,
            finish_reason=response.finish_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=_usage_detail(
                details,
                "reasoning_tokens",
                "reasoning",
                "completion_tokens_details.reasoning_tokens",
            ),
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=_cost(provider_details.get("cost")),
            latency_ns=latency_ns,
        )


def _stream_progress_projection(
    event: object,
) -> (
    tuple[
        StructuredStreamProgressKind,
        StructuredStreamChannel,
        tuple[tuple[str, str], ...],
    ]
    | None
):
    """Project exact semantic fragments from supported Pydantic stream events.

    This is deliberately not a wire-byte projection: Pydantic-AI does not
    expose the original SSE framing here.  For its closed text, thinking, and
    string tool-call fields, however, it exposes exact semantic string
    fragments. Unsupported model-response parts, including dictionary tool
    argument deltas whose original serialization is unknowable, fail closed.
    Agent workflow events outside the model-response stream return ``None``.
    """

    from pydantic_ai.messages import (
        FinalResultEvent,
        PartDeltaEvent,
        PartEndEvent,
        PartStartEvent,
        TextPart,
        TextPartDelta,
        ThinkingPart,
        ThinkingPartDelta,
        ToolCallPart,
        ToolCallPartDelta,
    )

    def exact_text(value: object, *, field: str) -> tuple[str, str] | None:
        if value is None:
            return None
        if type(value) is not str:
            raise StructuredGenerationError(
                kind=GenerationFailureKind.UNKNOWN,
                retryable=False,
                safe_message=("stream semantic content cannot be projected exactly"),
            )
        if not value:
            return None
        return field, value

    if type(event) is PartStartEvent:
        part = event.part
        if type(part) is TextPart:
            fragment = exact_text(part.content, field="text")
            return (
                StructuredStreamProgressKind.PART_STARTED,
                StructuredStreamChannel.TEXT,
                () if fragment is None else (fragment,),
            )
        if type(part) is ThinkingPart:
            fragment = exact_text(part.content, field="thinking")
            return (
                StructuredStreamProgressKind.PART_STARTED,
                StructuredStreamChannel.THINKING,
                () if fragment is None else (fragment,),
            )
        if type(part) is ToolCallPart:
            tool_name = exact_text(part.tool_name, field="tool_name")
            tool_args = exact_text(part.args, field="tool_args")
            return (
                StructuredStreamProgressKind.PART_STARTED,
                StructuredStreamChannel.TOOL_CALL,
                tuple(
                    fragment
                    for fragment in (tool_name, tool_args)
                    if fragment is not None
                ),
            )
        raise StructuredGenerationError(
            kind=GenerationFailureKind.UNKNOWN,
            retryable=False,
            safe_message="unsupported streamed model-response part",
        )

    if type(event) is PartDeltaEvent:
        delta = event.delta
        if type(delta) is TextPartDelta:
            fragment = exact_text(delta.content_delta, field="text")
            return (
                StructuredStreamProgressKind.PART_DELTA,
                StructuredStreamChannel.TEXT,
                () if fragment is None else (fragment,),
            )
        if type(delta) is ThinkingPartDelta:
            fragment = exact_text(delta.content_delta, field="thinking")
            return (
                StructuredStreamProgressKind.PART_DELTA,
                StructuredStreamChannel.THINKING,
                () if fragment is None else (fragment,),
            )
        if type(delta) is ToolCallPartDelta:
            tool_name = exact_text(delta.tool_name_delta, field="tool_name")
            tool_args = exact_text(delta.args_delta, field="tool_args")
            return (
                StructuredStreamProgressKind.PART_DELTA,
                StructuredStreamChannel.TOOL_CALL,
                tuple(
                    fragment
                    for fragment in (tool_name, tool_args)
                    if fragment is not None
                ),
            )
        raise StructuredGenerationError(
            kind=GenerationFailureKind.UNKNOWN,
            retryable=False,
            safe_message="unsupported streamed model-response delta",
        )

    if type(event) is PartEndEvent:
        part = event.part
        if type(part) is TextPart:
            channel = StructuredStreamChannel.TEXT
        elif type(part) is ThinkingPart:
            channel = StructuredStreamChannel.THINKING
        elif type(part) is ToolCallPart:
            channel = StructuredStreamChannel.TOOL_CALL
        else:
            raise StructuredGenerationError(
                kind=GenerationFailureKind.UNKNOWN,
                retryable=False,
                safe_message="unsupported completed model-response part",
            )
        return StructuredStreamProgressKind.PART_ENDED, channel, ()

    if type(event) is FinalResultEvent:
        return (
            # Despite its framework name, this event only announces which
            # output/tool Pydantic-AI selected. Its arguments can still stream.
            StructuredStreamProgressKind.OUTPUT_SELECTED,
            StructuredStreamChannel.OTHER,
            (),
        )
    return None


__all__ = [
    "OpenRouterReasoningConfig",
    "OpenRouterReasoningEffort",
    "OpenRouterStructuredOutputMode",
    "PydanticAIStructuredGenerator",
    "STREAM_CONTENT_IDENTITY_ALGORITHM",
    "STREAM_CONTENT_IDENTITY_DOMAIN_SHA256",
    "classify_generation_exception",
]

"""Provider-neutral port for one typed LLM generation attempt.

Retrying, queueing, and experiment policy deliberately live outside this port.  An
adapter executes exactly one provider attempt and either returns typed telemetry or
raises :class:`StructuredGenerationError` with a closed, non-secret classification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.domain.llm_task_queue import (
    CanonicalProviderErrorCode,
    MAX_VALIDATION_ISSUES,
    SanitizedExceptionProvenance,
    SanitizedValidationIssue,
    StreamTimeoutPhase,
    StructuredOutputFailureMode,
)
from agent_evolve.ports.generation_failure import GenerationFailureDisposition


OutputT = TypeVar("OutputT")
_OPERATION_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_TOOL_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPAIR_PATH_TOKEN = re.compile(r"^(?:\*|[A-Za-z_][A-Za-z0-9_-]{0,63})$")
MAX_PROMPT_UTF8_BYTES = 512_000
# Provider capabilities are injected by composition policies; this is only a
# language/runtime safety envelope for the integer carried across the generic
# port.  In particular, it must not encode one model route's completion cap.
MAX_OUTPUT_TOKENS = 2_147_483_647
MAX_STRUCTURED_REPAIR_LITERAL_SETS = 8
MAX_STRUCTURED_REPAIR_LITERALS = 1_024
MAX_STRUCTURED_REPAIR_LITERAL_UTF8_BYTES = 256
MAX_STRUCTURED_REPAIR_CONTEXT_UTF8_BYTES = 16_384
DEFAULT_STREAM_CANCEL_DRAIN_TIMEOUT_NS = 5_000_000_000
DEFAULT_STREAM_TRANSPORT_RETIRE_TIMEOUT_NS = 5_000_000_000
STREAM_CLEANUP_POLICY_ID = "bounded_cancel_drain"
STREAM_CLEANUP_POLICY_VERSION = 1
STREAM_CLEANUP_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:bounded-stream-cleanup:v1:"
    b"cancel-on-liveness-expiry;wait-bounded-cancel-drain;"
    b"if-unsettled-retire-supervisor-and-owned-transport-bounded;"
    b"publish-terminal-nonretryable-cleanup-timeout;never-overlap-retry"
).hexdigest()
_STREAM_CLEANUP_CONFIGURATION_DOMAIN = (
    b"agent-evolve:bounded-stream-cleanup-configuration:v1\x00"
)
IDENTITY_PROMPT_RENDERER_ID = "agent_evolve.identity_prompt"
IDENTITY_PROMPT_RENDERER_REVISION = "identity_v1"
IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:prompt-renderer:identity:v1\x00"
    b"wire-prompt-utf8-bytes-equal-semantic-prompt-utf8-bytes"
).hexdigest()


class GenerationFailureKind(str, Enum):
    """Closed failure classes used by scheduler and trace projections."""

    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CAPABILITY_MISMATCH = "capability_mismatch"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PAYMENT_REQUIRED = "payment_required"
    CONTENT_REJECTED = "content_rejected"
    OUTPUT_INVALID = "output_invalid"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class StructuredStreamProgressKind(str, Enum):
    """Closed, content-free model-stream lifecycle events.

    The part and output-selection values project provider-framework events.
    ``STREAM_COMPLETED`` is different: the adapter emits it locally only after
    the supervised agent call has returned a typed result.  No text,
    reasoning, tool arguments, or candidate data may cross this boundary.
    """

    PART_STARTED = "part_started"
    PART_DELTA = "part_delta"
    PART_ENDED = "part_ended"
    OUTPUT_SELECTED = "output_selected"
    STREAM_COMPLETED = "stream_completed"


class StructuredStreamChannel(str, Enum):
    """Closed stream channel classification safe for experiment journals."""

    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    OTHER = "other"


StructuredStreamTimeoutPhase = StreamTimeoutPhase


@dataclass(frozen=True, slots=True)
class StructuredStreamCleanupPolicy:
    """Bounded cleanup/retirement policy after a liveness deadline expires.

    This is not a healthy-generation deadline. It begins only after first-event,
    idle, absolute, or caller cancellation has already terminated the attempt.
    """

    cancel_drain_timeout_ns: int = DEFAULT_STREAM_CANCEL_DRAIN_TIMEOUT_NS
    transport_retire_timeout_ns: int = (
        DEFAULT_STREAM_TRANSPORT_RETIRE_TIMEOUT_NS
    )
    policy_id: str = STREAM_CLEANUP_POLICY_ID
    policy_version: int = STREAM_CLEANUP_POLICY_VERSION
    definition_sha256: str = STREAM_CLEANUP_POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        for name in ("cancel_drain_timeout_ns", "transport_retire_timeout_ns"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.policy_id) is not str or _OPERATION_TOKEN.fullmatch(
            self.policy_id
        ) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        if (
            type(self.definition_sha256) is not str
            or _LOWER_SHA256.fullmatch(self.definition_sha256) is None
        ):
            raise ValueError("definition_sha256 must be lowercase SHA-256")

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        payload = json.dumps(
            {
                "cancel_drain_timeout_ns": self.cancel_drain_timeout_ns,
                "definition_sha256": self.definition_sha256,
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "transport_retire_timeout_ns": self.transport_retire_timeout_ns,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(
            _STREAM_CLEANUP_CONFIGURATION_DOMAIN + payload
        ).hexdigest()

    def to_manifest_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "configuration_sha256": self.configuration_sha256,
            "cancel_drain_timeout_ns": self.cancel_drain_timeout_ns,
            "transport_retire_timeout_ns": self.transport_retire_timeout_ns,
        }


@dataclass(frozen=True, slots=True)
class StructuredStreamLivenessPolicy:
    """Content-blind liveness policy for one streamed provider attempt.

    ``first_event_timeout_ns`` bounds an attempt that never starts producing
    model-stream events. ``idle_timeout_ns`` is reset by every meaningful
    event. ``absolute_timeout_ns`` is an optional operational fail-safe; it is
    intentionally ``None`` by default so a healthy, progressing generation is
    not truncated merely because it is long. ``cleanup_policy`` starts only
    after one of those liveness boundaries (or caller cancellation) has fired.
    """

    first_event_timeout_ns: int
    idle_timeout_ns: int
    absolute_timeout_ns: int | None = None
    cleanup_policy: StructuredStreamCleanupPolicy = field(
        default_factory=StructuredStreamCleanupPolicy
    )

    def __post_init__(self) -> None:
        for name in ("first_event_timeout_ns", "idle_timeout_ns"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.absolute_timeout_ns is not None and (
            type(self.absolute_timeout_ns) is not int
            or self.absolute_timeout_ns <= 0
        ):
            raise ValueError(
                "absolute_timeout_ns must be a positive exact integer or None"
            )
        if type(self.cleanup_policy) is not StructuredStreamCleanupPolicy:
            raise TypeError(
                "cleanup_policy must be an exact StructuredStreamCleanupPolicy"
            )
        self.cleanup_policy.__post_init__()


@dataclass(frozen=True, slots=True)
class StructuredStreamProgress:
    """One content-free, monotonically sequenced stream observation.

    The byte counters and rolling digest authenticate the adapter's exact
    semantic UTF-8 fragments without retaining those fragments.  They are not
    transport/SSE byte counts.  Lifecycle events legitimately carry a zero
    delta and repeat the preceding rolling digest.
    """

    call_id: str
    sequence: int
    kind: StructuredStreamProgressKind
    channel: StructuredStreamChannel
    elapsed_ns: int
    event_content_utf8_bytes: int
    cumulative_content_utf8_bytes: int
    rolling_content_sha256: str
    provider_attempt_id: str | None = None

    def __post_init__(self) -> None:
        LLMCallId(self.call_id)
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("sequence must be a positive exact integer")
        if type(self.kind) is not StructuredStreamProgressKind:
            raise TypeError("kind must be a StructuredStreamProgressKind")
        if type(self.channel) is not StructuredStreamChannel:
            raise TypeError("channel must be a StructuredStreamChannel")
        if type(self.elapsed_ns) is not int or self.elapsed_ns < 0:
            raise ValueError("elapsed_ns must be a non-negative exact integer")
        if (
            type(self.event_content_utf8_bytes) is not int
            or self.event_content_utf8_bytes < 0
        ):
            raise ValueError(
                "event_content_utf8_bytes must be a non-negative exact integer"
            )
        if self.kind in {
            StructuredStreamProgressKind.OUTPUT_SELECTED,
            StructuredStreamProgressKind.STREAM_COMPLETED,
        } and (
            self.channel is not StructuredStreamChannel.OTHER
            or self.event_content_utf8_bytes != 0
        ):
            raise ValueError(
                "output selection and stream completion are content-free other events"
            )
        if (
            type(self.cumulative_content_utf8_bytes) is not int
            or self.cumulative_content_utf8_bytes < self.event_content_utf8_bytes
        ):
            raise ValueError(
                "cumulative_content_utf8_bytes must include the event byte count"
            )
        if (
            type(self.rolling_content_sha256) is not str
            or _LOWER_SHA256.fullmatch(self.rolling_content_sha256) is None
        ):
            raise ValueError(
                "rolling_content_sha256 must be a lowercase SHA-256 digest"
            )
        if self.provider_attempt_id is not None:
            ProviderAttemptId(self.provider_attempt_id)


StructuredStreamProgressSink = Callable[[StructuredStreamProgress], None]


class StructuredGenerationError(RuntimeError):
    """A sanitized one-attempt failure safe to retain in experiment traces."""

    def __init__(
        self,
        *,
        kind: GenerationFailureKind,
        retryable: bool,
        safe_message: str,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        output_failure_mode: StructuredOutputFailureMode | None = None,
        validation_issues: tuple[SanitizedValidationIssue, ...] = (),
        provider_error_code: CanonicalProviderErrorCode | None = None,
        provider_error_envelope_sha256: str | None = None,
        exception_provenance: SanitizedExceptionProvenance | None = None,
    ) -> None:
        if type(kind) is not GenerationFailureKind:
            raise TypeError("kind must be a GenerationFailureKind")
        if type(retryable) is not bool:
            raise TypeError("retryable must be bool")
        if type(safe_message) is not str or not safe_message.strip():
            raise ValueError("safe_message must be non-empty")
        if len(safe_message.encode("utf-8", errors="strict")) > 512:
            raise ValueError("safe_message is too large for inline telemetry")
        if status_code is not None and (
            type(status_code) is not int or not 100 <= status_code <= 599
        ):
            raise ValueError("status_code must be an HTTP status or None")
        if retry_after_seconds is not None and (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, (int, float))
            or not math.isfinite(float(retry_after_seconds))
            or float(retry_after_seconds) < 0
        ):
            raise ValueError("retry_after_seconds must be finite and non-negative")
        if (
            output_failure_mode is not None
            and type(output_failure_mode) is not StructuredOutputFailureMode
        ):
            raise TypeError(
                "output_failure_mode must be a StructuredOutputFailureMode or None"
            )
        if (
            output_failure_mode is not None
            and kind is not GenerationFailureKind.OUTPUT_INVALID
        ):
            raise ValueError(
                "only output_invalid failures may carry output diagnostics"
            )
        if type(validation_issues) is not tuple:
            raise TypeError("validation_issues must be an exact tuple")
        if len(validation_issues) > MAX_VALIDATION_ISSUES:
            raise ValueError(
                f"validation_issues cannot exceed {MAX_VALIDATION_ISSUES} entries"
            )
        if any(
            type(issue) is not SanitizedValidationIssue for issue in validation_issues
        ):
            raise TypeError(
                "validation_issues must contain exact SanitizedValidationIssue values"
            )
        if output_failure_mode is None and validation_issues:
            raise ValueError("validation issues require an output failure mode")
        if (
            validation_issues
            and output_failure_mode is not StructuredOutputFailureMode.SCHEMA_VALIDATION
        ):
            raise ValueError("validation issues require schema_validation mode")
        if provider_error_code is not None and (
            type(provider_error_code) is not CanonicalProviderErrorCode
        ):
            raise TypeError(
                "provider_error_code must be a CanonicalProviderErrorCode or None"
            )
        if provider_error_envelope_sha256 is not None and (
            type(provider_error_envelope_sha256) is not str
            or _LOWER_SHA256.fullmatch(provider_error_envelope_sha256) is None
        ):
            raise ValueError(
                "provider_error_envelope_sha256 must be lowercase SHA-256 or None"
            )
        if (
            provider_error_code is not None
            or provider_error_envelope_sha256 is not None
        ) and status_code is None:
            raise ValueError("provider HTTP diagnostics require status_code")
        if exception_provenance is not None and (
            type(exception_provenance) is not SanitizedExceptionProvenance
        ):
            raise TypeError(
                "exception_provenance must be SanitizedExceptionProvenance or None"
            )
        if exception_provenance is not None:
            SanitizedExceptionProvenance.__post_init__(exception_provenance)
            if kind is not GenerationFailureKind.UNKNOWN:
                raise ValueError(
                    "exception provenance is retained only for unknown failures"
                )
        super().__init__(safe_message)
        self.kind = kind
        self.retryable = retryable
        self.safe_message = safe_message
        self.status_code = status_code
        self.retry_after_seconds = (
            None if retry_after_seconds is None else float(retry_after_seconds)
        )
        self.output_failure_mode = output_failure_mode
        self.validation_issues = validation_issues
        self.provider_error_code = provider_error_code
        self.provider_error_envelope_sha256 = provider_error_envelope_sha256
        self.exception_provenance = exception_provenance

    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition:
        if self.kind in {
            GenerationFailureKind.OUTPUT_INVALID,
            GenerationFailureKind.CONTENT_REJECTED,
        }:
            return GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
        return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE


class StructuredStreamTimeoutError(StructuredGenerationError, TimeoutError):
    """A content-blind stream liveness boundary expired before completion."""

    def __init__(self, phase: StructuredStreamTimeoutPhase) -> None:
        if type(phase) is not StructuredStreamTimeoutPhase:
            raise TypeError("phase must be a StructuredStreamTimeoutPhase")
        messages = {
            StructuredStreamTimeoutPhase.FIRST_EVENT: (
                "provider stream produced no first event before its liveness deadline"
            ),
            StructuredStreamTimeoutPhase.IDLE: (
                "provider stream stopped producing progress before completion"
            ),
            StructuredStreamTimeoutPhase.ABSOLUTE: (
                "provider stream exceeded its absolute operational fail-safe"
            ),
        }
        super().__init__(
            kind=GenerationFailureKind.TIMEOUT,
            retryable=True,
            safe_message=messages[phase],
        )
        self.phase = phase


class StructuredStreamCleanupTimeoutError(StructuredGenerationError, TimeoutError):
    """Cancellation did not settle; the attempt and transport are retired.

    Unlike an ordinary liveness timeout this failure is terminal. Retrying could
    overlap the still-running provider attempt whose cancellation was resisted.
    """

    def __init__(self, phase: StructuredStreamTimeoutPhase) -> None:
        if type(phase) is not StructuredStreamTimeoutPhase:
            raise TypeError("phase must be a StructuredStreamTimeoutPhase")
        super().__init__(
            kind=GenerationFailureKind.TIMEOUT,
            retryable=False,
            safe_message=(
                "provider stream resisted bounded cancellation; its transport "
                "was retired and retry is forbidden"
            ),
        )
        self.phase = phase


class StructuredStreamRetiredError(StructuredGenerationError):
    """A supervisor/transport retired after an abandoned provider attempt."""

    def __init__(self) -> None:
        super().__init__(
            kind=GenerationFailureKind.CANCELLED,
            retryable=False,
            safe_message=(
                "provider stream supervisor is retired after incomplete cleanup"
            ),
        )


@dataclass(frozen=True, slots=True)
class StructuredPromptLineage:
    """Content-free commitment joining a semantic prompt to its wire renderer."""

    semantic_prompt_sha256: str
    renderer_id: str
    renderer_revision: str
    renderer_definition_sha256: str

    def __post_init__(self) -> None:
        for name in ("semantic_prompt_sha256", "renderer_definition_sha256"):
            value = getattr(self, name)
            if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("renderer_id", "renderer_revision"):
            value = getattr(self, name)
            if type(value) is not str or _OPERATION_TOKEN.fullmatch(value) is None:
                raise ValueError(
                    f"{name} must use the closed lowercase token grammar"
                )


def identity_prompt_lineage(prompt: str) -> StructuredPromptLineage:
    """Commit an exact prompt to the generic byte-identity renderer."""

    if type(prompt) is not str or not prompt.strip():
        raise ValueError("prompt must be a non-empty exact string")
    prompt_bytes = prompt.encode("utf-8", errors="strict")
    if len(prompt_bytes) > MAX_PROMPT_UTF8_BYTES:
        raise ValueError("prompt exceeds MAX_PROMPT_UTF8_BYTES")
    return StructuredPromptLineage(
        semantic_prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        renderer_id=IDENTITY_PROMPT_RENDERER_ID,
        renderer_revision=IDENTITY_PROMPT_RENDERER_REVISION,
        renderer_definition_sha256=IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256,
    )


@dataclass(frozen=True, slots=True)
class StructuredOutputRepairLiteralSet:
    """Provider-visible closed literals for a bounded structured-output repair.

    This is a generic output-contract hint, not workload advice.  A caller may
    expose values that were already visible in the original provider request so
    a retry can restate a large local closed set without weakening exact local
    validation.  No free-form text is accepted at this boundary.
    """

    field_path: tuple[str, ...]
    allowed_literals: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.field_path) is not tuple or not 1 <= len(self.field_path) <= 8:
            raise ValueError("field_path must contain between one and eight segments")
        if any(
            type(segment) is not str
            or _REPAIR_PATH_TOKEN.fullmatch(segment) is None
            for segment in self.field_path
        ):
            raise ValueError("field_path segments violate the closed repair grammar")
        if type(self.allowed_literals) is not tuple or not (
            1 <= len(self.allowed_literals) <= MAX_STRUCTURED_REPAIR_LITERALS
        ):
            raise ValueError(
                "allowed_literals must be a non-empty bounded exact tuple"
            )
        if len(set(self.allowed_literals)) != len(self.allowed_literals):
            raise ValueError("allowed_literals must be distinct")
        for literal in self.allowed_literals:
            if type(literal) is not str or not literal:
                raise ValueError("allowed_literals must contain non-empty strings")
            if (
                len(literal.encode("utf-8", errors="strict"))
                > MAX_STRUCTURED_REPAIR_LITERAL_UTF8_BYTES
            ):
                raise ValueError("one repair literal exceeds its UTF-8 byte bound")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "field_path": list(self.field_path),
            "allowed_literals": list(self.allowed_literals),
        }


@dataclass(frozen=True, slots=True)
class StructuredGenerationRequest(Generic[OutputT]):
    """Complete application request for one typed logical LLM call."""

    call_id: LLMCallId
    operation: str
    prompt: str
    output_type: type[OutputT]
    output_tool_name: str
    max_output_tokens: int = 2_048
    temperature: float | None = None
    provider_attempt_id: ProviderAttemptId | None = None
    prompt_lineage: StructuredPromptLineage | None = None
    repair_literal_sets: tuple[StructuredOutputRepairLiteralSet, ...] = ()

    def __post_init__(self) -> None:
        if type(self.call_id) is not LLMCallId:
            raise TypeError("call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.call_id)
        if (
            type(self.operation) is not str
            or _OPERATION_TOKEN.fullmatch(self.operation) is None
        ):
            raise ValueError("operation must use the closed lowercase token grammar")
        if type(self.prompt) is not str or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty exact string")
        if len(self.prompt.encode("utf-8", errors="strict")) > MAX_PROMPT_UTF8_BYTES:
            raise ValueError("prompt exceeds MAX_PROMPT_UTF8_BYTES")
        if not isinstance(self.output_type, type):
            raise TypeError("output_type must be a runtime type")
        if (
            type(self.output_tool_name) is not str
            or _TOOL_TOKEN.fullmatch(self.output_tool_name) is None
        ):
            raise ValueError("output_tool_name must use the closed tool token grammar")
        if (
            type(self.max_output_tokens) is not int
            or not 1 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError(f"max_output_tokens must lie in [1, {MAX_OUTPUT_TOKENS}]")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0 <= float(self.temperature) <= 2
        ):
            raise ValueError("temperature must be finite in [0,2] or None")
        if self.provider_attempt_id is not None and (
            type(self.provider_attempt_id) is not ProviderAttemptId
        ):
            raise TypeError(
                "provider_attempt_id must be a ProviderAttemptId or None"
            )
        if self.prompt_lineage is not None:
            if type(self.prompt_lineage) is not StructuredPromptLineage:
                raise TypeError(
                    "prompt_lineage must be an exact StructuredPromptLineage or None"
                )
            StructuredPromptLineage.__post_init__(self.prompt_lineage)
            if self.prompt_lineage.renderer_id == IDENTITY_PROMPT_RENDERER_ID:
                expected_semantic_sha256 = hashlib.sha256(
                    self.prompt.encode("utf-8", errors="strict")
                ).hexdigest()
                if (
                    self.prompt_lineage.renderer_revision
                    != IDENTITY_PROMPT_RENDERER_REVISION
                    or self.prompt_lineage.renderer_definition_sha256
                    != IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256
                    or self.prompt_lineage.semantic_prompt_sha256
                    != expected_semantic_sha256
                ):
                    raise ValueError(
                        "identity prompt lineage does not authenticate the exact "
                        "wire prompt"
                    )
        if type(self.repair_literal_sets) is not tuple or len(
            self.repair_literal_sets
        ) > MAX_STRUCTURED_REPAIR_LITERAL_SETS:
            raise ValueError("repair_literal_sets must be a bounded exact tuple")
        if any(
            type(item) is not StructuredOutputRepairLiteralSet
            for item in self.repair_literal_sets
        ):
            raise TypeError(
                "repair_literal_sets must contain exact "
                "StructuredOutputRepairLiteralSet values"
            )
        for item in self.repair_literal_sets:
            item.__post_init__()
        paths = tuple(item.field_path for item in self.repair_literal_sets)
        if len(set(paths)) != len(paths):
            raise ValueError("repair_literal_sets cannot repeat a field path")
        repair_context_bytes = json.dumps(
            [item.to_record() for item in self.repair_literal_sets],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(repair_context_bytes) > MAX_STRUCTURED_REPAIR_CONTEXT_UTF8_BYTES:
            raise ValueError("repair_literal_sets exceed the aggregate byte bound")


@dataclass(frozen=True, slots=True)
class StructuredGenerationResponse(Generic[OutputT]):
    """Typed output plus the telemetry available from exactly one attempt."""

    value: OutputT
    requested_model: str
    resolved_model: str
    resolved_provider: str
    provider_response_id: str | None
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: Decimal | None
    latency_ns: int

    def __post_init__(self) -> None:
        for name in ("requested_model", "resolved_model", "resolved_provider"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty exact string")
        for name in ("provider_response_id", "finish_reason"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"{name} must be a non-empty exact string or None")
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "latency_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.cost_usd is not None and (
            type(self.cost_usd) is not Decimal
            or not self.cost_usd.is_finite()
            or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be a finite non-negative Decimal or None")


@runtime_checkable
class StructuredGenerator(Protocol):
    """Execute one provider attempt; an outer scheduler owns all retries."""

    async def generate_once(
        self, request: StructuredGenerationRequest[OutputT]
    ) -> StructuredGenerationResponse[OutputT]: ...


__all__ = [
    "DEFAULT_STREAM_CANCEL_DRAIN_TIMEOUT_NS",
    "DEFAULT_STREAM_TRANSPORT_RETIRE_TIMEOUT_NS",
    "GenerationFailureKind",
    "IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256",
    "IDENTITY_PROMPT_RENDERER_ID",
    "IDENTITY_PROMPT_RENDERER_REVISION",
    "MAX_OUTPUT_TOKENS",
    "MAX_PROMPT_UTF8_BYTES",
    "STREAM_CLEANUP_POLICY_DEFINITION_SHA256",
    "STREAM_CLEANUP_POLICY_ID",
    "STREAM_CLEANUP_POLICY_VERSION",
    "StructuredGenerationError",
    "StructuredGenerationRequest",
    "StructuredGenerationResponse",
    "StructuredGenerator",
    "StructuredPromptLineage",
    "StructuredStreamChannel",
    "StructuredStreamCleanupPolicy",
    "StructuredStreamCleanupTimeoutError",
    "StructuredStreamLivenessPolicy",
    "StructuredStreamProgress",
    "StructuredStreamProgressKind",
    "StructuredStreamProgressSink",
    "StructuredStreamRetiredError",
    "StructuredStreamTimeoutError",
    "StructuredStreamTimeoutPhase",
    "identity_prompt_lineage",
]

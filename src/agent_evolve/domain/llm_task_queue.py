"""Provider-neutral values for asynchronous LLM task scheduling.

The values in this module contain no provider request, SDK exception, URL, or
transport type.  Provider integrations classify their own exceptions through
the port defined in :mod:`agent_evolve.ports.llm_task_queue`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Generic, Optional, Tuple, TypeVar

from agent_evolve.domain.ids import ProviderAttemptId


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")

NANOSECONDS_PER_SECOND = 1_000_000_000
MAX_ATTEMPTS = 100
MAX_TASK_ID_LENGTH = 128
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RETRY_AFTER_SECONDS = re.compile(r"^(?:0|[1-9][0-9]*)$")
_FAILURE_KIND = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VALIDATION_LOCATION_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_VALIDATION_ISSUES = 8
MAX_VALIDATION_LOCATION_DEPTH = 8
MAX_EXCEPTION_PROVENANCE_NODES = 16


def _require_nonnegative_ns(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer number of nanoseconds")


def _require_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_ns(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer number of nanoseconds")


class RetryDisposition(str, Enum):
    RETRY = "retry"
    FAIL = "fail"


class RetryReason(str, Enum):
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    OUTPUT_INVALID = "output_invalid"
    PERMANENT = "permanent"
    INTERNAL = "internal"


class RetryBudgetPartition(str, Enum):
    """Independent retry allowances beneath the hard physical-attempt cap."""

    OUTPUT_INVALID = "output_invalid"
    TRANSPORT = "transport"


def retry_budget_partition(reason: RetryReason) -> RetryBudgetPartition:
    """Map every retryable reason to its provider-neutral budget partition.

    ``PERMANENT`` and ``INTERNAL`` classifications are terminal by definition;
    admitting either as a retry would hide a broken classifier, so this helper
    rejects them instead of silently charging an unrelated allowance.
    """

    if type(reason) is not RetryReason:
        raise TypeError("reason must be a RetryReason")
    if reason is RetryReason.OUTPUT_INVALID:
        return RetryBudgetPartition.OUTPUT_INVALID
    if reason in {
        RetryReason.RATE_LIMIT,
        RetryReason.TRANSIENT,
        RetryReason.TIMEOUT,
    }:
        return RetryBudgetPartition.TRANSPORT
    raise ValueError(f"{reason.value} is not a retryable budget reason")


@dataclass(frozen=True, slots=True)
class PartitionedRetryBudget:
    """Bound semantic repair and transport replay independently.

    Values count *additional attempts admitted after failures*, while
    :attr:`LLMTask.max_attempts` remains the final bound on physical calls.
    This prevents a transient provider failure from consuming the allowance
    reserved for repairing invalid structured output, and vice versa.
    """

    output_invalid_retries: int
    transport_retries: int

    def __post_init__(self) -> None:
        for name, value in (
            ("output_invalid_retries", self.output_invalid_retries),
            ("transport_retries", self.transport_retries),
        ):
            _require_nonnegative_int(value, name)
            if value >= MAX_ATTEMPTS:
                raise ValueError(f"{name} must be less than {MAX_ATTEMPTS}")

    def limit(self, partition: RetryBudgetPartition) -> int:
        if type(partition) is not RetryBudgetPartition:
            raise TypeError("partition must be a RetryBudgetPartition")
        if partition is RetryBudgetPartition.OUTPUT_INVALID:
            return self.output_invalid_retries
        return self.transport_retries


@dataclass(frozen=True, slots=True)
class RetryBudgetUsage:
    """Immutable count of retry admissions before one physical attempt."""

    output_invalid_retries: int = 0
    transport_retries: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("output_invalid_retries", self.output_invalid_retries),
            ("transport_retries", self.transport_retries),
        ):
            _require_nonnegative_int(value, name)
            if value >= MAX_ATTEMPTS:
                raise ValueError(f"{name} must be less than {MAX_ATTEMPTS}")

    def used(self, partition: RetryBudgetPartition) -> int:
        if type(partition) is not RetryBudgetPartition:
            raise TypeError("partition must be a RetryBudgetPartition")
        if partition is RetryBudgetPartition.OUTPUT_INVALID:
            return self.output_invalid_retries
        return self.transport_retries

    def consume(self, partition: RetryBudgetPartition) -> "RetryBudgetUsage":
        if type(partition) is not RetryBudgetPartition:
            raise TypeError("partition must be a RetryBudgetPartition")
        if partition is RetryBudgetPartition.OUTPUT_INVALID:
            return RetryBudgetUsage(
                output_invalid_retries=self.output_invalid_retries + 1,
                transport_retries=self.transport_retries,
            )
        return RetryBudgetUsage(
            output_invalid_retries=self.output_invalid_retries,
            transport_retries=self.transport_retries + 1,
        )


class RetryAfterSource(str, Enum):
    DELAY_SECONDS = "delay_seconds"
    HTTP_DATE = "http_date"


class StructuredOutputFailureMode(str, Enum):
    """Closed, non-content-bearing modes for typed-output failures."""

    SCHEMA_VALIDATION = "schema_validation"
    TYPED_OUTPUT_CONTRACT = "typed_output_contract"
    INCOMPLETE_TOOL_CALL = "incomplete_tool_call"


class CanonicalProviderErrorCode(str, Enum):
    """Closed provider error codes safe to retain in durable telemetry.

    Provider adapters may admit a value only from a structured error field;
    messages, response bodies, and free-form provider metadata are never
    coerced into this enum.  The deliberately finite vocabulary covers the
    OpenAI-compatible error types currently exposed by OpenRouter and common
    upstream providers while failing closed for an unfamiliar value.
    """

    # Current OpenRouter typed error vocabulary.
    AUTHENTICATION = "authentication"
    CONTENT_POLICY_VIOLATION = "content_policy_violation"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    IMAGE_DOWNLOAD_FAILED = "image_download_failed"
    IMAGE_NOT_FOUND = "image_not_found"
    IMAGE_TOO_LARGE = "image_too_large"
    IMAGE_TOO_SMALL = "image_too_small"
    INVALID_IMAGE = "invalid_image"
    INVALID_PROMPT = "invalid_prompt"
    INVALID_REQUEST = "invalid_request"
    MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"
    NOT_FOUND = "not_found"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    PAYMENT_REQUIRED = "payment_required"
    PERMISSION_DENIED = "permission_denied"
    PRECONDITION_FAILED = "precondition_failed"
    PROVIDER_OVERLOADED = "provider_overloaded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    REFUSAL = "refusal"
    SERVER = "server"
    STRING_TOO_LONG = "string_too_long"
    TIMEOUT = "timeout"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    UNMAPPED = "unmapped"
    UNPROCESSABLE = "unprocessable"
    UNSUPPORTED_IMAGE_FORMAT = "unsupported_image_format"

    # OpenAI SDK and upstream compatibility codes observed on compatible APIs.
    API_ERROR = "api_error"
    AUTHENTICATION_ERROR = "authentication_error"
    BAD_REQUEST = "bad_request"
    CONFLICT_ERROR = "conflict_error"
    CONTENT_FILTER_ERROR = "content_filter_error"
    INSUFFICIENT_QUOTA = "insufficient_quota"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    INVALID_REQUEST_ERROR = "invalid_request_error"
    MODEL_NOT_FOUND = "model_not_found"
    NOT_FOUND_ERROR = "not_found_error"
    OVERLOADED_ERROR = "overloaded_error"
    PERMISSION_ERROR = "permission_error"
    PROVIDER_ERROR = "provider_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    REQUEST_TIMEOUT = "request_timeout"
    SERVER_ERROR = "server_error"
    SERVICE_UNAVAILABLE_ERROR = "service_unavailable_error"
    TOOL_ERROR = "tool_error"
    UNPROCESSABLE_ENTITY_ERROR = "unprocessable_entity_error"


class StreamTimeoutPhase(str, Enum):
    """Closed progress-aware stream boundary that expired."""

    FIRST_EVENT = "first_event"
    IDLE = "idle"
    ABSOLUTE = "absolute"


class ExceptionOriginFamily(str, Enum):
    """Closed runtime families safe to expose in durable diagnostics.

    Exact exception type identity is carried separately as a domain-separated
    fingerprint.  Keeping this value closed prevents an attacker-controlled
    module or class name from becoming durable journal text.
    """

    BUILTINS = "builtins"
    ASYNCIO = "asyncio"
    ANYIO = "anyio"
    HTTPX = "httpx"
    HTTPCORE = "httpcore"
    OPENAI = "openai"
    PYDANTIC = "pydantic"
    PYDANTIC_AI = "pydantic_ai"
    AGENT_EVOLVE = "agent_evolve"
    OTHER = "other"


class ExceptionProvenanceLink(str, Enum):
    """Closed relationship from one sanitized exception node to its parent."""

    ROOT = "root"
    CAUSE = "cause"
    CONTEXT = "context"
    GROUP_MEMBER = "group_member"


@dataclass(frozen=True, slots=True)
class SanitizedExceptionProvenanceNode:
    """One content-free exception-type identity in a bounded exception graph."""

    parent_index: Optional[int]
    link: ExceptionProvenanceLink
    family: ExceptionOriginFamily
    type_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.link) is not ExceptionProvenanceLink:
            raise TypeError("link must be an ExceptionProvenanceLink")
        if type(self.family) is not ExceptionOriginFamily:
            raise TypeError("family must be an ExceptionOriginFamily")
        if (
            type(self.type_identity_sha256) is not str
            or _SHA256.fullmatch(self.type_identity_sha256) is None
        ):
            raise ValueError("type_identity_sha256 must be lowercase SHA-256")
        if self.link is ExceptionProvenanceLink.ROOT:
            if self.parent_index is not None:
                raise ValueError("root exception provenance cannot have a parent")
        elif type(self.parent_index) is not int or self.parent_index < 0:
            raise ValueError(
                "non-root exception provenance requires a non-negative parent index"
            )


@dataclass(frozen=True, slots=True)
class SanitizedExceptionProvenance:
    """Bounded, topology-preserving provenance with no exception text.

    Nodes contain only closed families, SHA-256 fingerprints of bounded type
    identity projections, and integer graph edges.  Messages, reprs, response
    objects, URLs, payloads, tracebacks, and arbitrary metadata are absent by
    construction.
    """

    nodes: Tuple[SanitizedExceptionProvenanceNode, ...]
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.nodes) is not tuple or not self.nodes:
            raise ValueError("exception provenance nodes must be a non-empty tuple")
        if len(self.nodes) > MAX_EXCEPTION_PROVENANCE_NODES:
            raise ValueError(
                "exception provenance exceeds the bounded node allowance"
            )
        if any(
            type(node) is not SanitizedExceptionProvenanceNode
            for node in self.nodes
        ):
            raise TypeError(
                "exception provenance must contain exact sanitized nodes"
            )
        if type(self.truncated) is not bool:
            raise TypeError("exception provenance truncated must be bool")
        for index, node in enumerate(self.nodes):
            SanitizedExceptionProvenanceNode.__post_init__(node)
            if index == 0:
                if node.link is not ExceptionProvenanceLink.ROOT:
                    raise ValueError("first exception provenance node must be root")
            elif (
                node.link is ExceptionProvenanceLink.ROOT
                or node.parent_index is None
                or node.parent_index >= index
            ):
                raise ValueError(
                    "exception provenance parents must precede their child nodes"
                )


class ValidationIssueCategory(str, Enum):
    """Coarse validation categories safe to expose to a repair attempt."""

    MISSING = "missing"
    EXTRA_FIELD = "extra_field"
    LITERAL_OR_ENUM = "literal_or_enum"
    WRONG_TYPE = "wrong_type"
    BOUNDS_OR_LENGTH = "bounds_or_length"
    SEMANTIC_CONSTRAINT = "semantic_constraint"
    MALFORMED_ARGUMENTS = "malformed_arguments"
    OTHER_VALIDATION = "other_validation"


class ValidationIssueReasonCode(str, Enum):
    """Closed semantic-output reasons safe for durable diagnostics.

    Integrations may populate this field only from a trusted validator error
    *type*.  Free-form validation messages, model output, and Pydantic context
    values are never parsed into these codes.
    """

    DUPLICATE_FINITE_OPTIONS = "duplicate_finite_options"
    FINITE_OPTION_OUT_OF_CONTRACT = "finite_option_out_of_contract"
    ASSIGNED_MEMORY_CARD_OMITTED = "assigned_memory_card_omitted"
    PROPOSAL_SUPPORT_OPTION_OMITTED = "proposal_support_option_omitted"
    NO_FEASIBLE_DISJOINT_PORTFOLIO = "no_feasible_disjoint_portfolio"
    PORTFOLIO_MEMORY_DOSE_VIOLATION = "portfolio_memory_dose_violation"
    REFLECTION_METRIC_CONTRACT_VIOLATION = (
        "reflection_metric_contract_violation"
    )
    REFLECTION_ACTION_CONTRACT_VIOLATION = (
        "reflection_action_contract_violation"
    )
    REFLECTION_SEMANTIC_CONTRACT_VIOLATION = (
        "reflection_semantic_contract_violation"
    )
    REFLECTION_DIRECTION_OR_ANCHOR_VIOLATION = (
        "reflection_direction_or_anchor_violation"
    )
    RESIDUAL_RADIUS_CONTRACT_VIOLATION = (
        "residual_radius_contract_violation"
    )
    RESIDUAL_OPTION_CONTRACT_VIOLATION = (
        "residual_option_contract_violation"
    )
    RESIDUAL_METRIC_CONTRACT_VIOLATION = (
        "residual_metric_contract_violation"
    )
    RESIDUAL_QUANTILE_ORDER_VIOLATION = (
        "residual_quantile_order_violation"
    )
    RESIDUAL_PLAN_DIVERSITY_VIOLATION = (
        "residual_plan_diversity_violation"
    )


class AttemptRequestVariant(str, Enum):
    """Closed variants for the exact prompt used by one provider attempt."""

    ORIGINAL = "original"
    # Retained so historical telemetry remains decodable. New attempts use v4.
    SCHEMA_REPAIR_V1 = "schema_repair_v1"
    SCHEMA_REPAIR_V2 = "schema_repair_v2"
    SCHEMA_REPAIR_V3 = "schema_repair_v3"
    SCHEMA_REPAIR_V4 = "schema_repair_v4"


@dataclass(frozen=True, slots=True)
class AttemptRequestEvidence:
    """Content-free identity evidence for the request sent by one attempt."""

    variant: AttemptRequestVariant
    prompt_sha256: str
    provider_attempt_id: Optional[ProviderAttemptId] = None

    def __post_init__(self) -> None:
        if type(self.variant) is not AttemptRequestVariant:
            raise TypeError("variant must be an AttemptRequestVariant")
        if (
            type(self.prompt_sha256) is not str
            or _SHA256.fullmatch(self.prompt_sha256) is None
        ):
            raise ValueError("prompt_sha256 must be a lowercase SHA-256 digest")
        if self.provider_attempt_id is not None and (
            type(self.provider_attempt_id) is not ProviderAttemptId
        ):
            raise TypeError("provider_attempt_id must be a ProviderAttemptId or None")


@dataclass(frozen=True, slots=True)
class SanitizedValidationIssue:
    """One bounded issue with no message, input value, or provider content."""

    category: ValidationIssueCategory
    location: Tuple[str, ...]
    reason_code: Optional[ValidationIssueReasonCode] = None

    def __post_init__(self) -> None:
        if type(self.category) is not ValidationIssueCategory:
            raise TypeError("category must be a ValidationIssueCategory")
        if type(self.location) is not tuple:
            raise TypeError("location must be an exact tuple")
        if not 1 <= len(self.location) <= MAX_VALIDATION_LOCATION_DEPTH:
            raise ValueError(
                "location must contain between one and "
                f"{MAX_VALIDATION_LOCATION_DEPTH} safe segments"
            )
        for segment in self.location:
            if (
                type(segment) is not str
                or _VALIDATION_LOCATION_TOKEN.fullmatch(segment) is None
            ):
                raise ValueError("location segments must use the safe token grammar")
        if self.reason_code is not None and (
            type(self.reason_code) is not ValidationIssueReasonCode
        ):
            raise TypeError("reason_code must be a ValidationIssueReasonCode or None")
        if (
            self.reason_code is not None
            and self.category is not ValidationIssueCategory.SEMANTIC_CONSTRAINT
        ):
            raise ValueError(
                "reason_code is valid only for a semantic_constraint issue"
            )


@dataclass(frozen=True, slots=True)
class RetryAfter:
    """A validated Retry-After delay; sleeping remains the queue's job."""

    delay_ns: int
    source: RetryAfterSource

    def __post_init__(self) -> None:
        _require_nonnegative_ns(self.delay_ns, "delay_ns")
        if type(self.source) is not RetryAfterSource:
            raise TypeError("source must be a RetryAfterSource")


def parse_retry_after(value: object, *, now_utc: datetime) -> Optional[RetryAfter]:
    """Parse an HTTP Retry-After value without sleeping or provider coupling.

    RFC delay-seconds and timezone-aware HTTP dates are accepted. Invalid,
    non-ASCII, fractional, or unbounded values return ``None``. Past dates map
    to a zero delay. Date resolution is rounded up to the next nanosecond so a
    parsed server delay is never shortened.
    """

    if type(value) is not str or not value or len(value) > 128 or not value.isascii():
        return None
    text = value.strip()
    if _RETRY_AFTER_SECONDS.fullmatch(text):
        seconds = int(text)
        if seconds > (2**63 - 1) // NANOSECONDS_PER_SECOND:
            return None
        return RetryAfter(
            delay_ns=seconds * NANOSECONDS_PER_SECOND,
            source=RetryAfterSource.DELAY_SECONDS,
        )

    if type(now_utc) is not datetime or now_utc.utcoffset() is None:
        raise ValueError("now_utc must be a timezone-aware datetime")
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None or parsed.utcoffset() is None:
        return None
    delta_seconds = (
        parsed.astimezone(timezone.utc) - now_utc.astimezone(timezone.utc)
    ).total_seconds()
    if not math.isfinite(delta_seconds):
        return None
    delay_ns = max(0, math.ceil(delta_seconds * NANOSECONDS_PER_SECOND))
    if delay_ns > 2**63 - 1:
        return None
    return RetryAfter(delay_ns=delay_ns, source=RetryAfterSource.HTTP_DATE)


@dataclass(frozen=True, slots=True)
class SanitizedAttemptFailure:
    """Bounded failure evidence safe to retain in durable attempt telemetry.

    The queue never derives this value from ``str(error)``.  Integrations may
    attach it only after replacing provider text and response bodies with a
    closed kind and an explicitly sanitized message.
    """

    kind: str
    retryable: bool
    safe_message: str
    status_code: Optional[int] = None
    retry_after_seconds: Optional[float] = None
    output_failure_mode: Optional[StructuredOutputFailureMode] = None
    validation_issues: Tuple[SanitizedValidationIssue, ...] = ()
    stream_timeout_phase: Optional[StreamTimeoutPhase] = None
    provider_error_code: Optional[CanonicalProviderErrorCode] = None
    provider_error_envelope_sha256: Optional[str] = None
    exception_provenance: Optional[SanitizedExceptionProvenance] = None

    def __post_init__(self) -> None:
        if type(self.kind) is not str or _FAILURE_KIND.fullmatch(self.kind) is None:
            raise ValueError("kind must use the closed lowercase token grammar")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be bool")
        if type(self.safe_message) is not str or not self.safe_message.strip():
            raise ValueError("safe_message must be non-empty")
        if len(self.safe_message.encode("utf-8", errors="strict")) > 512:
            raise ValueError("safe_message is too large for inline telemetry")
        if self.status_code is not None and (
            type(self.status_code) is not int or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be an HTTP status or None")
        if self.retry_after_seconds is not None and (
            isinstance(self.retry_after_seconds, bool)
            or not isinstance(self.retry_after_seconds, (int, float))
            or not math.isfinite(float(self.retry_after_seconds))
            or float(self.retry_after_seconds) < 0
        ):
            raise ValueError("retry_after_seconds must be finite and non-negative")
        if (
            self.output_failure_mode is not None
            and type(self.output_failure_mode) is not StructuredOutputFailureMode
        ):
            raise TypeError(
                "output_failure_mode must be a StructuredOutputFailureMode or None"
            )
        if self.output_failure_mode is not None and self.kind != "output_invalid":
            raise ValueError(
                "only output_invalid failures may carry output diagnostics"
            )
        if type(self.validation_issues) is not tuple:
            raise TypeError("validation_issues must be an exact tuple")
        if len(self.validation_issues) > MAX_VALIDATION_ISSUES:
            raise ValueError(
                f"validation_issues cannot exceed {MAX_VALIDATION_ISSUES} entries"
            )
        if any(
            type(issue) is not SanitizedValidationIssue
            for issue in self.validation_issues
        ):
            raise TypeError(
                "validation_issues must contain exact SanitizedValidationIssue values"
            )
        if self.output_failure_mode is None and self.validation_issues:
            raise ValueError("validation issues require an output failure mode")
        if (
            self.validation_issues
            and self.output_failure_mode
            is not StructuredOutputFailureMode.SCHEMA_VALIDATION
        ):
            raise ValueError("validation issues require schema_validation mode")
        if self.stream_timeout_phase is not None and (
            type(self.stream_timeout_phase) is not StreamTimeoutPhase
        ):
            raise TypeError("stream_timeout_phase must be a StreamTimeoutPhase or None")
        if self.stream_timeout_phase is not None and self.kind != "timeout":
            raise ValueError("only timeout failures may carry a stream timeout phase")
        if self.provider_error_code is not None and (
            type(self.provider_error_code) is not CanonicalProviderErrorCode
        ):
            raise TypeError(
                "provider_error_code must be a CanonicalProviderErrorCode or None"
            )
        if self.provider_error_envelope_sha256 is not None and (
            type(self.provider_error_envelope_sha256) is not str
            or _SHA256.fullmatch(self.provider_error_envelope_sha256) is None
        ):
            raise ValueError(
                "provider_error_envelope_sha256 must be lowercase SHA-256 or None"
            )
        if (
            self.provider_error_code is not None
            or self.provider_error_envelope_sha256 is not None
        ) and self.status_code is None:
            raise ValueError("provider HTTP diagnostics require status_code")
        if self.exception_provenance is not None and (
            type(self.exception_provenance) is not SanitizedExceptionProvenance
        ):
            raise TypeError(
                "exception_provenance must be SanitizedExceptionProvenance or None"
            )
        if self.exception_provenance is not None:
            SanitizedExceptionProvenance.__post_init__(self.exception_provenance)
            if self.kind != "unknown":
                raise ValueError(
                    "exception provenance is retained only for unknown failures"
                )


@dataclass(frozen=True, slots=True)
class RetryClassification:
    disposition: RetryDisposition
    reason: RetryReason
    retry_after: Optional[RetryAfter] = None
    sanitized_failure: Optional[SanitizedAttemptFailure] = None

    def __post_init__(self) -> None:
        if type(self.disposition) is not RetryDisposition:
            raise TypeError("disposition must be a RetryDisposition")
        if type(self.reason) is not RetryReason:
            raise TypeError("reason must be a RetryReason")
        if self.retry_after is not None and type(self.retry_after) is not RetryAfter:
            raise TypeError("retry_after must be a RetryAfter or None")
        if (
            self.sanitized_failure is not None
            and type(self.sanitized_failure) is not SanitizedAttemptFailure
        ):
            raise TypeError(
                "sanitized_failure must be a SanitizedAttemptFailure or None"
            )
        if self.disposition is RetryDisposition.FAIL and self.retry_after is not None:
            raise ValueError("a terminal classification cannot carry Retry-After")


@dataclass(frozen=True, slots=True)
class LLMTask(Generic[RequestT]):
    task_id: str
    request: RequestT
    max_attempts: int
    attempt_timeout_ns: Optional[int] = None
    retry_budget: Optional[PartitionedRetryBudget] = None

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or _TASK_ID.fullmatch(self.task_id) is None:
            raise ValueError(
                "task_id must be a non-secret ASCII identifier of at most "
                f"{MAX_TASK_ID_LENGTH} characters"
            )
        if (
            type(self.max_attempts) is not int
            or not 1 <= self.max_attempts <= MAX_ATTEMPTS
        ):
            raise ValueError(f"max_attempts must be an integer in [1, {MAX_ATTEMPTS}]")
        if self.attempt_timeout_ns is not None:
            _require_positive_ns(self.attempt_timeout_ns, "attempt_timeout_ns")
        if self.retry_budget is not None and (
            type(self.retry_budget) is not PartitionedRetryBudget
        ):
            raise TypeError(
                "retry_budget must be a PartitionedRetryBudget or None"
            )
        if self.retry_budget is not None:
            PartitionedRetryBudget.__post_init__(self.retry_budget)


@dataclass(frozen=True, slots=True)
class LLMAttemptContext:
    task_id: str
    attempt_number: int
    attempt_timeout_ns: Optional[int]
    previous_failure: Optional[SanitizedAttemptFailure] = None
    active_output_failure: Optional[SanitizedAttemptFailure] = None
    retry_budget_usage: Optional[RetryBudgetUsage] = None

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or _TASK_ID.fullmatch(self.task_id) is None:
            raise ValueError("task_id violates the queue identifier policy")
        if type(self.attempt_number) is not int or self.attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        if self.attempt_timeout_ns is not None:
            _require_positive_ns(self.attempt_timeout_ns, "attempt_timeout_ns")
        if (
            self.previous_failure is not None
            and type(self.previous_failure) is not SanitizedAttemptFailure
        ):
            raise TypeError(
                "previous_failure must be a SanitizedAttemptFailure or None"
            )
        if (
            self.active_output_failure is not None
            and type(self.active_output_failure) is not SanitizedAttemptFailure
        ):
            raise TypeError(
                "active_output_failure must be a SanitizedAttemptFailure or None"
            )
        if self.active_output_failure is not None and (
            self.active_output_failure.kind != "output_invalid"
            or not self.active_output_failure.retryable
        ):
            raise ValueError(
                "active_output_failure must be a retryable output_invalid failure"
            )
        if self.retry_budget_usage is not None and (
            type(self.retry_budget_usage) is not RetryBudgetUsage
        ):
            raise TypeError(
                "retry_budget_usage must be a RetryBudgetUsage or None"
            )
        if self.retry_budget_usage is not None:
            RetryBudgetUsage.__post_init__(self.retry_budget_usage)
            admitted_retries = (
                self.retry_budget_usage.output_invalid_retries
                + self.retry_budget_usage.transport_retries
            )
            if admitted_retries != self.attempt_number - 1:
                raise ValueError(
                    "partitioned retry usage must account for every prior attempt"
                )
        if self.attempt_number == 1 and (
            self.previous_failure is not None or self.active_output_failure is not None
        ):
            raise ValueError("attempt one cannot carry prior failure state")


class AttemptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AttemptTelemetry:
    """One provider attempt's wait, execution, and retry scheduling evidence.

    ``wait_time_ns`` is initial scheduler queue time for attempt one and the
    actual inter-attempt wait for later attempts. ``service_time_ns`` covers
    only execution inside the timeout owner.
    """

    attempt_number: int
    status: AttemptStatus
    wait_time_ns: int
    service_time_ns: int
    will_retry: bool
    policy_backoff_ns: int = 0
    retry_after_ns: int = 0
    scheduled_delay_ns: int = 0
    classification: Optional[RetryClassification] = None
    error_type: Optional[str] = None
    request_evidence: Optional[AttemptRequestEvidence] = None

    def __post_init__(self) -> None:
        if type(self.attempt_number) is not int or self.attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        if type(self.status) is not AttemptStatus:
            raise TypeError("status must be an AttemptStatus")
        for name, value in (
            ("wait_time_ns", self.wait_time_ns),
            ("service_time_ns", self.service_time_ns),
            ("policy_backoff_ns", self.policy_backoff_ns),
            ("retry_after_ns", self.retry_after_ns),
            ("scheduled_delay_ns", self.scheduled_delay_ns),
        ):
            _require_nonnegative_ns(value, name)
        if type(self.will_retry) is not bool:
            raise TypeError("will_retry must be bool")
        if (
            self.classification is not None
            and type(self.classification) is not RetryClassification
        ):
            raise TypeError("classification must be a RetryClassification or None")
        if self.error_type is not None and (
            type(self.error_type) is not str
            or not self.error_type
            or len(self.error_type) > 256
        ):
            raise ValueError("error_type must be a bounded non-empty string or None")
        if (
            self.request_evidence is not None
            and type(self.request_evidence) is not AttemptRequestEvidence
        ):
            raise TypeError(
                "request_evidence must be an AttemptRequestEvidence or None"
            )

        if self.status is AttemptStatus.SUCCEEDED:
            if (
                self.will_retry
                or self.classification is not None
                or self.error_type is not None
                or self.policy_backoff_ns
                or self.retry_after_ns
                or self.scheduled_delay_ns
            ):
                raise ValueError("successful attempt telemetry carries failure fields")
            return

        if self.status is AttemptStatus.CANCELLED:
            if self.will_retry or self.classification is not None:
                raise ValueError("cancelled attempt cannot schedule a retry")
            return

        if self.classification is None or self.error_type is None:
            raise ValueError(
                "failed attempt telemetry requires classification and error type"
            )
        expected_retry_after = (
            self.classification.retry_after.delay_ns
            if self.classification.retry_after is not None
            else 0
        )
        if self.retry_after_ns != expected_retry_after:
            raise ValueError("retry_after_ns disagrees with the classification")
        if self.will_retry:
            if self.classification.disposition is not RetryDisposition.RETRY:
                raise ValueError("will_retry requires a retry classification")
            if self.scheduled_delay_ns != max(
                self.policy_backoff_ns,
                self.retry_after_ns,
            ):
                raise ValueError(
                    "scheduled delay must honor policy backoff and Retry-After"
                )
        elif self.policy_backoff_ns or self.scheduled_delay_ns:
            raise ValueError("a terminal attempt cannot carry a scheduled backoff")


class TaskOutcomeStatus(str, Enum):
    SUCCEEDED = "succeeded"
    TERMINAL_FAILURE = "terminal_failure"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    CANCELLED = "cancelled"


class CancellationReason(str, Enum):
    QUEUE_CLOSED = "queue_closed"
    EXECUTOR_RETIRED = "executor_retired"
    SUBMITTER_CANCELLED = "submitter_cancelled"


@dataclass(frozen=True, slots=True)
class TaskTelemetry:
    """Whole-task timing.

    Queue time ends at the first attempt start. Service time runs from that
    start through the terminal outcome and therefore includes retry waits.
    """

    task_id: str
    queue_time_ns: int
    service_time_ns: int
    total_time_ns: int
    attempts: Tuple[AttemptTelemetry, ...]

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or _TASK_ID.fullmatch(self.task_id) is None:
            raise ValueError("task_id violates the queue identifier policy")
        for name, value in (
            ("queue_time_ns", self.queue_time_ns),
            ("service_time_ns", self.service_time_ns),
            ("total_time_ns", self.total_time_ns),
        ):
            _require_nonnegative_ns(value, name)
        if type(self.attempts) is not tuple or any(
            type(attempt) is not AttemptTelemetry for attempt in self.attempts
        ):
            raise TypeError(
                "attempts must be an exact tuple of AttemptTelemetry values"
            )
        if [attempt.attempt_number for attempt in self.attempts] != list(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("attempt telemetry numbers must be contiguous")
        if self.queue_time_ns + self.service_time_ns != self.total_time_ns:
            raise ValueError("total_time_ns must equal queue plus service time")


@dataclass(frozen=True, slots=True)
class LLMTaskOutcome(Generic[ResponseT]):
    status: TaskOutcomeStatus
    telemetry: TaskTelemetry
    response: Optional[ResponseT] = None
    cancellation_reason: Optional[CancellationReason] = None

    def __post_init__(self) -> None:
        if type(self.status) is not TaskOutcomeStatus:
            raise TypeError("status must be a TaskOutcomeStatus")
        if type(self.telemetry) is not TaskTelemetry:
            raise TypeError("telemetry must be a TaskTelemetry")
        if self.status is TaskOutcomeStatus.SUCCEEDED:
            if self.cancellation_reason is not None:
                raise ValueError("a successful task cannot carry cancellation state")
            if not self.telemetry.attempts or (
                self.telemetry.attempts[-1].status is not AttemptStatus.SUCCEEDED
            ):
                raise ValueError(
                    "a successful task requires a successful final attempt"
                )
        elif self.status is TaskOutcomeStatus.CANCELLED:
            if type(self.cancellation_reason) is not CancellationReason:
                raise ValueError("a cancelled task requires a cancellation reason")
            if self.response is not None:
                raise ValueError("a cancelled task cannot carry a response")
        else:
            if self.response is not None or self.cancellation_reason is not None:
                raise ValueError(
                    "a failed task cannot carry response or cancellation state"
                )
            if not self.telemetry.attempts:
                raise ValueError("a failed task requires at least one attempt")


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    max_in_flight: int
    max_pending: int
    in_flight: int
    pending: int
    closed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("max_in_flight", self.max_in_flight),
            ("max_pending", self.max_pending),
            ("in_flight", self.in_flight),
            ("pending", self.pending),
        ):
            _require_nonnegative_int(value, name)
        if self.max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        if self.in_flight > self.max_in_flight or self.pending > self.max_pending:
            raise ValueError("queue snapshot exceeds configured bounds")
        if type(self.closed) is not bool:
            raise TypeError("closed must be bool")

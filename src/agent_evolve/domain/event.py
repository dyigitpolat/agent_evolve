"""Immutable event schema and canonical JSON codec.

The event vocabulary is intentionally small but covers the accounting boundaries
needed by the current loop. Large or mutable values are represented by artifact
IDs; objective vectors use immutable tuples.
"""

from __future__ import annotations

import json
import math
import re
import types
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from agent_evolve.domain.artifact import ArtifactRef, ArtifactRole
from agent_evolve.domain.ids import (
    ArtifactId,
    CandidateId,
    CorrelationId,
    EvaluationAttemptId,
    EvaluationId,
    EventId,
    GenerationId,
    ID_TYPES,
    LLMCallId,
    OperatorInvocationId,
    ProviderAttemptId,
    RunId,
    StableId,
)
from agent_evolve.domain.inline_text import InlineTextPolicy, validate_inline_text
from agent_evolve.domain.outcome import (
    FailureCategory,
    FailureCode,
    FailureRecord,
    validate_failure_pair,
)

CURRENT_EVENT_SCHEMA_VERSION = 1
ObjectiveValues = Tuple[Tuple[str, float], ...]
_POLICY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_POLICY_CREDENTIAL_SHAPE = re.compile(
    r"(?:(?<![A-Za-z0-9])(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])or-v1-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])(?:ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9]))",
    re.IGNORECASE,
)
_DECIMAL_TEXT = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_MAX_DECIMAL_DIGITS = 64
_MAX_DECIMAL_ABS_EXPONENT = 64
_MAX_DECIMAL_TEXT_LENGTH = 128
_DECODE_FAILED = object()


class EventCodecError(ValueError):
    """An event record does not conform to the supported durable schema."""


class ValidationStage(str, Enum):
    SCHEMA = "schema"
    DETERMINISTIC_PRECHECK = "deterministic_precheck"


class EventPayload:
    """Marker base for frozen payload value objects."""

    EVENT_TYPE: ClassVar[str]


@dataclass(frozen=True, slots=True)
class ArtifactRegistered(EventPayload):
    """A post-write journal record for one verified, sanitized artifact."""

    EVENT_TYPE: ClassVar[str] = "ArtifactRegistered"

    artifact_id: ArtifactId
    sha256_hex: str
    size_bytes: int
    media_type: str
    role: ArtifactRole
    minimization_policy_id: str
    minimization_policy_version: str
    minimization_policy_config_sha256: str
    sanitization_policy_id: str
    sanitization_policy_version: str

    def __post_init__(self) -> None:
        # Reuse the exact ArtifactRef invariants rather than maintaining a
        # second, subtly different metadata schema in the event vocabulary.
        ArtifactRef(
            artifact_id=self.artifact_id,
            sha256_hex=self.sha256_hex,
            size_bytes=self.size_bytes,
            media_type=self.media_type,
        )
        if not isinstance(self.role, ArtifactRole):
            raise TypeError("role must be an ArtifactRole")
        for name, value in (
            ("minimization_policy_id", self.minimization_policy_id),
            ("minimization_policy_version", self.minimization_policy_version),
            ("sanitization_policy_id", self.sanitization_policy_id),
            ("sanitization_policy_version", self.sanitization_policy_version),
        ):
            if (
                not isinstance(value, str)
                or _POLICY_COMPONENT.fullmatch(value) is None
                or _POLICY_CREDENTIAL_SHAPE.search(value)
            ):
                raise ValueError(
                    f"{name} must be a non-secret, storage-safe policy component"
                )
        _require_hash(
            self.minimization_policy_config_sha256,
            "minimization_policy_config_sha256",
        )

    @property
    def artifact_ref(self) -> ArtifactRef:
        """Reconstruct the complete reference claimed by this event."""

        return ArtifactRef(
            artifact_id=self.artifact_id,
            sha256_hex=self.sha256_hex,
            size_bytes=self.size_bytes,
            media_type=self.media_type,
        )


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_nonnegative(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_hash(value: str, name: str) -> None:
    _require_nonempty(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _is_bounded_finite_decimal(value: Any) -> bool:
    if type(value) is not Decimal or not value.is_finite():
        return False
    decimal_tuple = value.as_tuple()
    return (
        len(decimal_tuple.digits) <= _MAX_DECIMAL_DIGITS
        and -_MAX_DECIMAL_ABS_EXPONENT
        <= decimal_tuple.exponent
        <= _MAX_DECIMAL_ABS_EXPONENT
    )


def _require_bounded_finite_decimal(value: Any, name: str) -> None:
    if not _is_bounded_finite_decimal(value):
        raise TypeError(f"{name} must be a bounded finite Decimal")


def _validate_objective_values(values: ObjectiveValues) -> None:
    if type(values) is not tuple:
        raise TypeError("objective_values must be an immutable tuple")
    if not values:
        raise ValueError("objective_values must contain at least one declared objective")
    names = set()
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("each objective entry must be a (name, value) tuple")
        name, value = item
        _require_nonempty(name, "objective name")
        validate_inline_text(
            name,
            field_name="objective name",
            policy=InlineTextPolicy.METADATA_TOKEN,
        )
        if name in names:
            raise ValueError(f"duplicate objective value {name!r}")
        names.add(name)
        if type(value) not in (int, float):
            raise TypeError(f"objective {name!r} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"objective {name!r} must be finite")


@dataclass(frozen=True, slots=True)
class RunStarted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "RunStarted"

    experiment_spec_hash: str
    manifest_artifact_id: Optional[ArtifactId] = None

    def __post_init__(self) -> None:
        _require_hash(self.experiment_spec_hash, "experiment_spec_hash")


@dataclass(frozen=True, slots=True)
class RunFinished(EventPayload):
    EVENT_TYPE: ClassVar[str] = "RunFinished"

    stop_reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.stop_reason, "stop_reason")


@dataclass(frozen=True, slots=True)
class RunAborted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "RunAborted"

    category: FailureCategory
    code: FailureCode
    message: str
    exception_type: Optional[str] = None
    diagnostics_artifact_id: Optional[ArtifactId] = None

    def __post_init__(self) -> None:
        FailureRecord(
            category=self.category,
            code=self.code,
            message=self.message,
            exception_type=self.exception_type,
            diagnostics_artifact_id=self.diagnostics_artifact_id,
        )
        if self.category is FailureCategory.CANDIDATE:
            raise ValueError("candidate-attributable failures cannot abort a run")


@dataclass(frozen=True, slots=True)
class GenerationStarted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "GenerationStarted"

    generation_id: GenerationId
    ordinal: int

    def __post_init__(self) -> None:
        _require_positive(self.ordinal, "ordinal")


@dataclass(frozen=True, slots=True)
class GenerationCompleted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "GenerationCompleted"

    generation_id: GenerationId
    ordinal: int
    pareto_size: int

    def __post_init__(self) -> None:
        _require_positive(self.ordinal, "ordinal")
        _require_nonnegative(self.pareto_size, "pareto_size")


@dataclass(frozen=True, slots=True)
class OperatorSelected(EventPayload):
    EVENT_TYPE: ClassVar[str] = "OperatorSelected"

    operator_invocation_id: OperatorInvocationId
    generation_id: GenerationId
    operator_name: str
    operator_version: str
    requested_offspring: int

    def __post_init__(self) -> None:
        _require_nonempty(self.operator_name, "operator_name")
        _require_nonempty(self.operator_version, "operator_version")
        _require_nonnegative(self.requested_offspring, "requested_offspring")


@dataclass(frozen=True, slots=True)
class LLMCallRequested(EventPayload):
    EVENT_TYPE: ClassVar[str] = "LLMCallRequested"

    call_id: LLMCallId
    operation: str
    requested_model: str
    generation_id: Optional[GenerationId] = None
    operator_invocation_id: Optional[OperatorInvocationId] = None

    def __post_init__(self) -> None:
        _require_nonempty(self.operation, "operation")
        _require_nonempty(self.requested_model, "requested_model")


@dataclass(frozen=True, slots=True)
class LLMRequestArtifactLinked(EventPayload):
    """Link one sanitized request artifact to a logical LLM call."""

    EVENT_TYPE: ClassVar[str] = "LLMRequestArtifactLinked"

    call_id: LLMCallId
    request_artifact_id: ArtifactId


@dataclass(frozen=True, slots=True)
class LLMCallStarted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "LLMCallStarted"

    call_id: LLMCallId


@dataclass(frozen=True, slots=True)
class LLMCallCompleted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "LLMCallCompleted"

    call_id: LLMCallId
    response_artifact_id: Optional[ArtifactId] = None


@dataclass(frozen=True, slots=True)
class LLMCallFailed(EventPayload):
    EVENT_TYPE: ClassVar[str] = "LLMCallFailed"

    call_id: LLMCallId
    category: FailureCategory
    code: FailureCode
    retryable: bool
    message: str
    exception_type: Optional[str] = None
    diagnostics_artifact_id: Optional[ArtifactId] = None

    def __post_init__(self) -> None:
        FailureRecord(
            category=self.category,
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            exception_type=self.exception_type,
            diagnostics_artifact_id=self.diagnostics_artifact_id,
        )
        if self.category is FailureCategory.CANDIDATE:
            raise ValueError("LLM call failures cannot be candidate-attributable")


@dataclass(frozen=True, slots=True)
class LLMCallRetried(EventPayload):
    EVENT_TYPE: ClassVar[str] = "LLMCallRetried"

    call_id: LLMCallId
    next_attempt_number: int

    def __post_init__(self) -> None:
        _require_positive(self.next_attempt_number, "next_attempt_number")
        if self.next_attempt_number < 2:
            raise ValueError("a retry attempt number must be at least 2")


@dataclass(frozen=True, slots=True)
class ProviderAttemptStarted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "ProviderAttemptStarted"

    call_id: LLMCallId
    provider_attempt_id: ProviderAttemptId
    attempt_number: int

    def __post_init__(self) -> None:
        _require_positive(self.attempt_number, "attempt_number")


@dataclass(frozen=True, slots=True)
class ProviderAttemptCompleted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "ProviderAttemptCompleted"

    call_id: LLMCallId
    provider_attempt_id: ProviderAttemptId
    resolved_provider: str
    resolved_model: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    latency_ns: int = 0

    def __post_init__(self) -> None:
        _require_nonempty(self.resolved_provider, "resolved_provider")
        _require_nonempty(self.resolved_model, "resolved_model")
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "latency_ns",
        ):
            _require_nonnegative(getattr(self, name), name)
        _require_bounded_finite_decimal(self.cost_usd, "cost_usd")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")


@dataclass(frozen=True, slots=True)
class ProviderAttemptFailed(EventPayload):
    EVENT_TYPE: ClassVar[str] = "ProviderAttemptFailed"

    call_id: LLMCallId
    provider_attempt_id: ProviderAttemptId
    category: FailureCategory
    code: FailureCode
    retryable: bool
    message: str
    latency_ns: int = 0
    exception_type: Optional[str] = None
    diagnostics_artifact_id: Optional[ArtifactId] = None

    def __post_init__(self) -> None:
        FailureRecord(
            category=self.category,
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            exception_type=self.exception_type,
            diagnostics_artifact_id=self.diagnostics_artifact_id,
        )
        if self.category is FailureCategory.CANDIDATE:
            raise ValueError("provider failures cannot be candidate-attributable")
        _require_nonnegative(self.latency_ns, "latency_ns")


@dataclass(frozen=True, slots=True)
class CandidateProposed(EventPayload):
    EVENT_TYPE: ClassVar[str] = "CandidateProposed"

    candidate_id: CandidateId
    generation_id: GenerationId
    content_hash: str
    proposal_index: int
    operator_invocation_id: Optional[OperatorInvocationId] = None
    generator_call_id: Optional[LLMCallId] = None
    configuration_artifact_id: Optional[ArtifactId] = None

    def __post_init__(self) -> None:
        _require_hash(self.content_hash, "content_hash")
        _require_nonnegative(self.proposal_index, "proposal_index")


@dataclass(frozen=True, slots=True)
class DuplicateDetected(EventPayload):
    EVENT_TYPE: ClassVar[str] = "DuplicateDetected"

    candidate_id: CandidateId
    content_hash: str
    canonical_candidate_id: Optional[CandidateId] = None
    reason: str = "same_content"

    def __post_init__(self) -> None:
        _require_hash(self.content_hash, "content_hash")
        _require_nonempty(self.reason, "reason")
        if self.canonical_candidate_id == self.candidate_id:
            raise ValueError("a duplicate cannot identify itself as its canonical occurrence")


@dataclass(frozen=True, slots=True)
class ValidationStarted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "ValidationStarted"

    candidate_id: CandidateId
    stage: ValidationStage


@dataclass(frozen=True, slots=True)
class ValidationCompleted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "ValidationCompleted"

    candidate_id: CandidateId
    stage: ValidationStage
    ok: bool
    failure_category: Optional[FailureCategory] = None
    failure_code: Optional[FailureCode] = None

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")
        if self.ok:
            if self.failure_category is not None or self.failure_code is not None:
                raise ValueError("successful validation cannot carry failure fields")
        elif self.failure_category is None or self.failure_code is None:
            raise ValueError("failed validation requires a failure category and code")
        else:
            validate_failure_pair(self.failure_category, self.failure_code)


@dataclass(frozen=True, slots=True)
class EvaluationRequested(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationRequested"

    evaluation_id: EvaluationId
    candidate_id: CandidateId
    fidelity: str
    seed: Optional[int] = None
    cache_key_hash: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        if self.seed is not None and type(self.seed) is not int:
            raise TypeError("seed must be an integer or None")
        if self.cache_key_hash is not None:
            _require_hash(self.cache_key_hash, "cache_key_hash")


@dataclass(frozen=True, slots=True)
class EvaluationCacheHit(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationCacheHit"

    evaluation_id: EvaluationId
    candidate_id: CandidateId
    fidelity: str
    source_run_id: RunId
    source_evaluation_id: EvaluationId
    objective_values: ObjectiveValues

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        _validate_objective_values(self.objective_values)


@dataclass(frozen=True, slots=True)
class EvaluationCacheMiss(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationCacheMiss"

    evaluation_id: EvaluationId
    fidelity: str

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")


@dataclass(frozen=True, slots=True)
class EvaluationCacheBypassed(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationCacheBypassed"

    evaluation_id: EvaluationId
    fidelity: str
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        _require_nonempty(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class EvaluationCacheStored(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationCacheStored"

    evaluation_id: EvaluationId
    fidelity: str
    cache_key_hash: str

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        _require_hash(self.cache_key_hash, "cache_key_hash")


@dataclass(frozen=True, slots=True)
class EvaluationStarted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationStarted"

    evaluation_id: EvaluationId
    evaluation_attempt_id: EvaluationAttemptId
    fidelity: str
    attempt_number: int

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        _require_positive(self.attempt_number, "attempt_number")


@dataclass(frozen=True, slots=True)
class EvaluationCompleted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationCompleted"

    evaluation_id: EvaluationId
    evaluation_attempt_id: EvaluationAttemptId
    fidelity: str
    objective_values: ObjectiveValues
    worker_time_ns: int = 0

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        _validate_objective_values(self.objective_values)
        _require_nonnegative(self.worker_time_ns, "worker_time_ns")


@dataclass(frozen=True, slots=True)
class EvaluationFailed(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationFailed"

    evaluation_id: EvaluationId
    fidelity: str
    category: FailureCategory
    code: FailureCode
    retryable: bool
    terminal: bool
    message: str
    evaluation_attempt_id: Optional[EvaluationAttemptId] = None
    worker_time_ns: int = 0
    exception_type: Optional[str] = None
    diagnostics_artifact_id: Optional[ArtifactId] = None

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        FailureRecord(
            category=self.category,
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            exception_type=self.exception_type,
            diagnostics_artifact_id=self.diagnostics_artifact_id,
        )
        if not isinstance(self.retryable, bool) or not isinstance(self.terminal, bool):
            raise TypeError("retryable and terminal must be bool")
        if not self.terminal and not self.retryable:
            raise ValueError("a non-terminal failure must be retryable")
        if self.category is FailureCategory.CANDIDATE and not self.terminal:
            raise ValueError("candidate-attributable evaluation failures must be terminal")
        _require_nonnegative(self.worker_time_ns, "worker_time_ns")
        if self.evaluation_attempt_id is None and self.worker_time_ns:
            raise ValueError("worker_time_ns requires an evaluation_attempt_id")


@dataclass(frozen=True, slots=True)
class EvaluationRetried(EventPayload):
    EVENT_TYPE: ClassVar[str] = "EvaluationRetried"

    evaluation_id: EvaluationId
    fidelity: str
    next_attempt_number: int

    def __post_init__(self) -> None:
        _require_nonempty(self.fidelity, "fidelity")
        _require_positive(self.next_attempt_number, "next_attempt_number")
        if self.next_attempt_number < 2:
            raise ValueError("a retry attempt number must be at least 2")


@dataclass(frozen=True, slots=True)
class CandidateAdmitted(EventPayload):
    EVENT_TYPE: ClassVar[str] = "CandidateAdmitted"

    candidate_id: CandidateId
    evaluation_id: EvaluationId
    archive_name: str = "all_time"

    def __post_init__(self) -> None:
        _require_nonempty(self.archive_name, "archive_name")


EVENT_PAYLOAD_TYPES: Tuple[Type[EventPayload], ...] = (
    ArtifactRegistered,
    RunStarted,
    RunFinished,
    RunAborted,
    GenerationStarted,
    GenerationCompleted,
    OperatorSelected,
    LLMCallRequested,
    LLMRequestArtifactLinked,
    LLMCallStarted,
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallRetried,
    ProviderAttemptStarted,
    ProviderAttemptCompleted,
    ProviderAttemptFailed,
    CandidateProposed,
    DuplicateDetected,
    ValidationStarted,
    ValidationCompleted,
    EvaluationRequested,
    EvaluationCacheHit,
    EvaluationCacheMiss,
    EvaluationCacheBypassed,
    EvaluationCacheStored,
    EvaluationStarted,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRetried,
    CandidateAdmitted,
)


_INLINE_TEXT_POLICIES: Dict[Type[EventPayload], Dict[str, InlineTextPolicy]] = {
    ArtifactRegistered: {
        "sha256_hex": InlineTextPolicy.SHA256,
        "media_type": InlineTextPolicy.MEDIA_TYPE,
        "minimization_policy_id": InlineTextPolicy.POLICY_COMPONENT,
        "minimization_policy_version": InlineTextPolicy.POLICY_COMPONENT,
        "minimization_policy_config_sha256": InlineTextPolicy.SHA256,
        "sanitization_policy_id": InlineTextPolicy.POLICY_COMPONENT,
        "sanitization_policy_version": InlineTextPolicy.POLICY_COMPONENT,
    },
    RunStarted: {"experiment_spec_hash": InlineTextPolicy.SHA256},
    RunFinished: {"stop_reason": InlineTextPolicy.SAFE_SUMMARY},
    RunAborted: {
        "message": InlineTextPolicy.SAFE_SUMMARY,
        "exception_type": InlineTextPolicy.EXCEPTION_TYPE,
    },
    OperatorSelected: {
        "operator_name": InlineTextPolicy.METADATA_TOKEN,
        "operator_version": InlineTextPolicy.METADATA_TOKEN,
    },
    LLMCallRequested: {
        "operation": InlineTextPolicy.METADATA_TOKEN,
        "requested_model": InlineTextPolicy.ROUTING_LABEL,
    },
    LLMCallFailed: {
        "message": InlineTextPolicy.SAFE_SUMMARY,
        "exception_type": InlineTextPolicy.EXCEPTION_TYPE,
    },
    ProviderAttemptCompleted: {
        "resolved_provider": InlineTextPolicy.ROUTING_LABEL,
        "resolved_model": InlineTextPolicy.ROUTING_LABEL,
    },
    ProviderAttemptFailed: {
        "message": InlineTextPolicy.SAFE_SUMMARY,
        "exception_type": InlineTextPolicy.EXCEPTION_TYPE,
    },
    CandidateProposed: {"content_hash": InlineTextPolicy.SHA256},
    DuplicateDetected: {
        "content_hash": InlineTextPolicy.SHA256,
        "reason": InlineTextPolicy.METADATA_TOKEN,
    },
    EvaluationRequested: {
        "fidelity": InlineTextPolicy.METADATA_TOKEN,
        "cache_key_hash": InlineTextPolicy.SHA256,
    },
    EvaluationCacheHit: {"fidelity": InlineTextPolicy.METADATA_TOKEN},
    EvaluationCacheMiss: {"fidelity": InlineTextPolicy.METADATA_TOKEN},
    EvaluationCacheBypassed: {
        "fidelity": InlineTextPolicy.METADATA_TOKEN,
        "reason": InlineTextPolicy.SAFE_SUMMARY,
    },
    EvaluationCacheStored: {
        "fidelity": InlineTextPolicy.METADATA_TOKEN,
        "cache_key_hash": InlineTextPolicy.SHA256,
    },
    EvaluationStarted: {"fidelity": InlineTextPolicy.METADATA_TOKEN},
    EvaluationCompleted: {"fidelity": InlineTextPolicy.METADATA_TOKEN},
    EvaluationFailed: {
        "fidelity": InlineTextPolicy.METADATA_TOKEN,
        "message": InlineTextPolicy.SAFE_SUMMARY,
        "exception_type": InlineTextPolicy.EXCEPTION_TYPE,
    },
    EvaluationRetried: {"fidelity": InlineTextPolicy.METADATA_TOKEN},
    CandidateAdmitted: {"archive_name": InlineTextPolicy.METADATA_TOKEN},
}


def _annotation_is_optional_or_required_string(annotation: Any) -> bool:
    if annotation is str:
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin in (Union, types.UnionType) and set(args) == {str, type(None)}


def _annotation_contains_string(annotation: Any) -> bool:
    if annotation is str:
        return True
    return any(_annotation_contains_string(arg) for arg in get_args(annotation))


def _annotation_contains_type(annotation: Any, expected: Any) -> bool:
    if annotation is expected:
        return True
    return any(
        _annotation_contains_type(argument, expected)
        for argument in get_args(annotation)
    )


def _annotation_contains_tuple(annotation: Any) -> bool:
    if get_origin(annotation) is tuple:
        return True
    return any(_annotation_contains_tuple(argument) for argument in get_args(annotation))


def _validate_supported_value_annotation(annotation: Any) -> None:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        if len(arguments) != 2 or type(None) not in arguments:
            raise RuntimeError("event unions must be a scalar optional value")
        for argument in arguments:
            if argument is not type(None):
                _validate_supported_value_annotation(argument)
        return
    if origin is tuple:
        for argument in arguments:
            if argument is not Ellipsis:
                _validate_supported_value_annotation(argument)
        return
    if origin is not None:
        raise RuntimeError("event containers require an explicit codec design")
    if annotation in (str, bool, int, float, Decimal, type(None)):
        return
    if annotation in ID_TYPES:
        return
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return
    raise RuntimeError("event field annotation has no durable codec policy")


_NESTED_STRING_FIELDS = {
    (EvaluationCacheHit, "objective_values"),
    (EvaluationCompleted, "objective_values"),
}
_CONTAINER_FIELDS = frozenset(_NESTED_STRING_FIELDS)
_DECIMAL_TEXT_FIELDS = {(ProviderAttemptCompleted, "cost_usd")}


def validate_event_value_schema() -> None:
    """Audit every payload path that can serialize to durable JSON text."""

    for payload_type in EVENT_PAYLOAD_TYPES:
        try:
            validate_inline_text(
                payload_type.EVENT_TYPE,
                field_name="event type",
                policy=InlineTextPolicy.METADATA_TOKEN,
            )
        except (TypeError, ValueError):
            valid_event_type = False
        else:
            valid_event_type = True
        if not valid_event_type:
            raise RuntimeError("event type violates the durable metadata policy")

        hints = get_type_hints(payload_type)
        for field in fields(payload_type):
            annotation = hints[field.name]
            _validate_supported_value_annotation(annotation)
            path = (payload_type, field.name)
            contains_nested_string = (
                _annotation_contains_string(annotation)
                and not _annotation_is_optional_or_required_string(annotation)
            )
            if contains_nested_string != (path in _NESTED_STRING_FIELDS):
                raise RuntimeError(
                    "non-scalar event string fields require an explicit policy"
                )
            if _annotation_contains_tuple(annotation) != (path in _CONTAINER_FIELDS):
                raise RuntimeError(
                    "event container fields require an explicit immutable policy"
                )
            contains_decimal = _annotation_contains_type(annotation, Decimal)
            if contains_decimal != (path in _DECIMAL_TEXT_FIELDS):
                raise RuntimeError(
                    "Decimal event fields require an explicit bounded-text policy"
                )

            for nested_type in _iter_annotation_leaf_types(annotation):
                if not (
                    isinstance(nested_type, type)
                    and issubclass(nested_type, Enum)
                ):
                    continue
                for member in nested_type:
                    try:
                        validate_inline_text(
                            member.value,
                            field_name="enum value",
                            policy=InlineTextPolicy.ENUM_VALUE,
                        )
                    except (TypeError, ValueError):
                        valid_member = False
                    else:
                        valid_member = True
                    if not valid_member:
                        raise RuntimeError(
                            "event enum value violates the durable metadata policy"
                        )


def _iter_annotation_leaf_types(annotation: Any) -> Tuple[Any, ...]:
    arguments = get_args(annotation)
    if not arguments:
        return (annotation,)
    leaves = []
    for argument in arguments:
        if argument is not Ellipsis:
            leaves.extend(_iter_annotation_leaf_types(argument))
    return tuple(leaves)


def validate_inline_text_schema() -> None:
    """Require an explicit storage policy for every direct event string field."""

    validate_event_value_schema()
    payload_types = set(EVENT_PAYLOAD_TYPES)
    if set(_INLINE_TEXT_POLICIES) - payload_types:
        raise RuntimeError("inline-text policy references an unregistered payload")
    for payload_type in EVENT_PAYLOAD_TYPES:
        hints = get_type_hints(payload_type)
        nested_string_fields = {
            field.name
            for field in fields(payload_type)
            if _annotation_contains_string(hints[field.name])
            and not _annotation_is_optional_or_required_string(hints[field.name])
        }
        allowed_nested = {
            field_name
            for declared_type, field_name in _NESTED_STRING_FIELDS
            if declared_type is payload_type
        }
        if nested_string_fields != allowed_nested:
            raise RuntimeError(
                "non-scalar event string fields require an explicit storage policy"
            )
        string_fields = {
            field.name
            for field in fields(payload_type)
            if _annotation_is_optional_or_required_string(hints[field.name])
        }
        declared = _INLINE_TEXT_POLICIES.get(payload_type, {})
        if string_fields != set(declared):
            raise RuntimeError(
                "event inline-text fields and policies do not match for "
                f"{payload_type.EVENT_TYPE}"
            )
        if any(not isinstance(policy, InlineTextPolicy) for policy in declared.values()):
            raise RuntimeError("event inline-text policies must use InlineTextPolicy")


def _validate_payload_inline_text(payload: EventPayload) -> None:
    policies = _INLINE_TEXT_POLICIES.get(type(payload), {})
    for field_name, policy in policies.items():
        value = getattr(payload, field_name)
        if value is None:
            continue
        validate_inline_text(value, field_name=field_name, policy=policy)


validate_inline_text_schema()
_PAYLOAD_BY_EVENT_TYPE: Dict[str, Type[EventPayload]] = {
    payload_type.EVENT_TYPE: payload_type for payload_type in EVENT_PAYLOAD_TYPES
}
if len(_PAYLOAD_BY_EVENT_TYPE) != len(EVENT_PAYLOAD_TYPES):  # pragma: no cover
    raise RuntimeError("Event payload types must have unique EVENT_TYPE values")


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    schema_version: int
    event_id: EventId
    run_id: RunId
    sequence_number: int
    event_type: str
    wall_timestamp_utc: datetime
    monotonic_offset_ns: int
    correlation_id: Optional[CorrelationId]
    causation_event_id: Optional[EventId]
    payload: EventPayload

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        if self.schema_version != CURRENT_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported event schema version {self.schema_version!r}; "
                f"supported={CURRENT_EVENT_SCHEMA_VERSION}"
            )
        _require_positive(self.sequence_number, "sequence_number")
        _require_nonnegative(self.monotonic_offset_ns, "monotonic_offset_ns")
        if type(self.event_id) is not EventId or not _payload_field_round_trip_is_valid(
            EventId,
            self.event_id,
            "event_id",
        ):
            raise TypeError("event_id must be an EventId")
        if type(self.run_id) is not RunId or not _payload_field_round_trip_is_valid(
            RunId,
            self.run_id,
            "run_id",
        ):
            raise TypeError("run_id must be a RunId")
        if self.correlation_id is not None and (
            type(self.correlation_id) is not CorrelationId
            or not _payload_field_round_trip_is_valid(
                CorrelationId,
                self.correlation_id,
                "correlation_id",
            )
        ):
            raise TypeError("correlation_id must be a CorrelationId or None")
        if self.causation_event_id is not None and (
            type(self.causation_event_id) is not EventId
            or not _payload_field_round_trip_is_valid(
                EventId,
                self.causation_event_id,
                "causation_event_id",
            )
        ):
            raise TypeError("causation_event_id must be an EventId or None")
        if type(self.payload) not in EVENT_PAYLOAD_TYPES:
            raise TypeError("payload must be a registered EventPayload")
        if type(self.event_type) is not str:
            raise TypeError("event_type must be an exact string")
        validate_inline_text(
            self.event_type,
            field_name="event_type",
            policy=InlineTextPolicy.METADATA_TOKEN,
        )
        payload_event_type = type(self.payload).EVENT_TYPE
        if type(payload_event_type) is not str:
            raise TypeError("payload event type must be an exact string")
        validate_inline_text(
            payload_event_type,
            field_name="payload event type",
            policy=InlineTextPolicy.METADATA_TOKEN,
        )
        _validate_payload_inline_text(self.payload)
        hints = get_type_hints(type(self.payload))
        for field in fields(self.payload):
            # Encoding followed by typed decoding is also the runtime type check
            # for direct (non-JSON) construction. It prevents, for example, a
            # RunId from being stored in a CandidateId field.
            if not _payload_field_round_trip_is_valid(
                hints[field.name],
                getattr(self.payload, field.name),
                field.name,
            ):
                raise TypeError(f"Invalid payload field {field.name}")
        if self.event_type != payload_event_type:
            raise ValueError("event_type does not match payload type")
        if type(self.wall_timestamp_utc) is not datetime:
            raise TypeError("wall_timestamp_utc must be an exact datetime")
        if self.wall_timestamp_utc.tzinfo is not timezone.utc:
            raise ValueError("wall_timestamp_utc must use the canonical UTC timezone")


_ENVELOPE_FIELDS = {
    "schema_version",
    "event_id",
    "run_id",
    "sequence_number",
    "event_type",
    "wall_timestamp_utc",
    "monotonic_offset_ns",
    "correlation_id",
    "causation_event_id",
    "payload",
}


def _encode_value(value: Any) -> Any:
    if isinstance(value, StableId):
        return value.value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        if not _is_bounded_finite_decimal(value):
            raise EventCodecError("Decimal exceeds the durable numeric-text policy")
        encoded = format(value, "f")
        if (
            len(encoded) > _MAX_DECIMAL_TEXT_LENGTH
            or _DECIMAL_TEXT.fullmatch(encoded) is None
        ):
            raise EventCodecError("Decimal exceeds the durable numeric-text policy")
        return encoded
    if isinstance(value, tuple):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise EventCodecError(f"Unsupported event payload value {type(value).__name__}")


def _payload_field_round_trip_is_valid(
    annotation: Any,
    value: Any,
    field_name: str,
) -> bool:
    try:
        _decode_value(annotation, _encode_value(value), field_name)
    except Exception:
        return False
    return True


def _decode_value(annotation: Any, value: Any, field_name: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) != 1:
            raise EventCodecError(f"Unsupported union annotation for {field_name}")
        return _decode_value(non_none[0], value, field_name)

    if origin is tuple:
        if type(value) is not list:
            raise EventCodecError(f"{field_name} must be a JSON array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode_value(args[0], item, field_name) for item in value)
        if len(args) != len(value):
            raise EventCodecError(f"{field_name} has the wrong tuple length")
        return tuple(
            _decode_value(item_type, item, field_name)
            for item_type, item in zip(args, value)
        )

    if isinstance(annotation, type) and issubclass(annotation, StableId):
        if type(value) is not str:
            raise EventCodecError(f"{field_name} must be a string ID")
        try:
            decoded_id = annotation(value)
        except (TypeError, ValueError):
            decoded_id = _DECODE_FAILED
        if decoded_id is _DECODE_FAILED:
            raise EventCodecError(f"Invalid durable ID for {field_name}")
        return decoded_id
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            decoded_enum = annotation(value)
        except Exception:
            decoded_enum = _DECODE_FAILED
        if decoded_enum is _DECODE_FAILED:
            raise EventCodecError(f"Invalid enum value for {field_name}")
        return decoded_enum
    if annotation is Decimal:
        if (
            type(value) is not str
            or len(value) > _MAX_DECIMAL_TEXT_LENGTH
            or _DECIMAL_TEXT.fullmatch(value) is None
        ):
            raise EventCodecError(f"{field_name} Decimal must be encoded as a string")
        try:
            decoded_decimal = Decimal(value)
        except Exception:
            decoded_decimal = _DECODE_FAILED
        if (
            decoded_decimal is _DECODE_FAILED
            or not _is_bounded_finite_decimal(decoded_decimal)
            or format(decoded_decimal, "f") != value
        ):
            raise EventCodecError(f"Invalid bounded Decimal for {field_name}")
        return decoded_decimal
    if annotation is str:
        if type(value) is not str:
            raise EventCodecError(f"{field_name} must be a string")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise EventCodecError(f"{field_name} must be bool")
        return value
    if annotation is int:
        if type(value) is not int:
            raise EventCodecError(f"{field_name} must be an integer")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise EventCodecError(f"{field_name} must be numeric")
        return float(value)
    raise EventCodecError(f"Unsupported annotation {annotation!r} for {field_name}")


def _payload_to_record(payload: EventPayload) -> Dict[str, Any]:
    return {field.name: _encode_value(getattr(payload, field.name)) for field in fields(payload)}


def _payload_from_record(event_type: str, record: Any) -> EventPayload:
    payload_type = _PAYLOAD_BY_EVENT_TYPE.get(event_type)
    if payload_type is None:
        raise EventCodecError("Unsupported event_type")
    if type(record) is not dict:
        raise EventCodecError("payload must be a JSON object")
    if any(type(key) is not str for key in record):
        raise EventCodecError("payload keys must be exact strings")
    expected = {field.name for field in fields(payload_type)}
    if set(record) != expected:
        raise EventCodecError("Invalid payload fields")
    hints = get_type_hints(payload_type)
    try:
        kwargs = {
            name: _decode_value(hints[name], record[name], name)
            for name in expected
        }
    except Exception:
        kwargs = None
    if kwargs is None:
        raise EventCodecError("Invalid payload field encoding")
    try:
        payload = payload_type(**kwargs)
    except Exception:
        payload = None
    if payload is None:
        raise EventCodecError("Invalid event payload")
    return payload


def _try_snapshot_payload(
    event_type: str,
    payload: EventPayload,
) -> EventPayload | object:
    try:
        return _payload_from_record(event_type, _payload_to_record(payload))
    except Exception:
        # Return inside the handler so secret-bearing extension exceptions do
        # not become context on the public snapshot/codec error.
        return _DECODE_FAILED


def _try_construct_event_snapshot(
    event: EventEnvelope,
    payload: EventPayload,
) -> EventEnvelope | object:
    try:
        source_timestamp = event.wall_timestamp_utc
        timestamp = datetime(
            source_timestamp.year,
            source_timestamp.month,
            source_timestamp.day,
            source_timestamp.hour,
            source_timestamp.minute,
            source_timestamp.second,
            source_timestamp.microsecond,
            tzinfo=timezone.utc,
            fold=source_timestamp.fold,
        )
        return EventEnvelope(
            schema_version=event.schema_version,
            event_id=EventId(event.event_id.value),
            run_id=RunId(event.run_id.value),
            sequence_number=event.sequence_number,
            event_type=event.event_type,
            wall_timestamp_utc=timestamp,
            monotonic_offset_ns=event.monotonic_offset_ns,
            correlation_id=(
                CorrelationId(event.correlation_id.value)
                if event.correlation_id is not None
                else None
            ),
            causation_event_id=(
                EventId(event.causation_event_id.value)
                if event.causation_event_id is not None
                else None
            ),
            payload=payload,
        )
    except Exception:
        return _DECODE_FAILED


def validated_event_snapshot(event: EventEnvelope) -> EventEnvelope:
    """Return an independently reconstructed event safe for persistence.

    Frozen dataclasses can be mutated through low-level APIs, so adapters and
    serializers never trust a caller-owned instance merely because its original
    constructor ran.
    """

    if type(event) is not EventEnvelope:
        raise EventCodecError("event must be an exact EventEnvelope")
    if (
        type(event.schema_version) is not int
        or type(event.event_id) is not EventId
        or type(event.run_id) is not RunId
        or type(event.sequence_number) is not int
        or type(event.event_type) is not str
        or type(event.wall_timestamp_utc) is not datetime
        or event.wall_timestamp_utc.tzinfo is not timezone.utc
        or type(event.monotonic_offset_ns) is not int
        or (
            event.correlation_id is not None
            and type(event.correlation_id) is not CorrelationId
        )
        or (
            event.causation_event_id is not None
            and type(event.causation_event_id) is not EventId
        )
        or type(event.payload) not in EVENT_PAYLOAD_TYPES
    ):
        raise EventCodecError("event violates the canonical envelope policy")
    payload = _try_snapshot_payload(event.event_type, event.payload)
    if payload is _DECODE_FAILED:
        raise EventCodecError("event payload could not be canonically reconstructed")
    assert isinstance(payload, EventPayload)
    snapshot = _try_construct_event_snapshot(event, payload)
    if snapshot is _DECODE_FAILED:
        raise EventCodecError("event could not be canonically reconstructed")
    assert isinstance(snapshot, EventEnvelope)
    return snapshot


def event_to_record(event: EventEnvelope) -> Dict[str, Any]:
    """Return the schema-stable JSON-compatible representation of *event*."""

    snapshot = validated_event_snapshot(event)
    timestamp = snapshot.wall_timestamp_utc.isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": snapshot.schema_version,
        "event_id": snapshot.event_id.value,
        "run_id": snapshot.run_id.value,
        "sequence_number": snapshot.sequence_number,
        "event_type": snapshot.event_type,
        "wall_timestamp_utc": timestamp,
        "monotonic_offset_ns": snapshot.monotonic_offset_ns,
        "correlation_id": (
            snapshot.correlation_id.value
            if snapshot.correlation_id is not None
            else None
        ),
        "causation_event_id": (
            snapshot.causation_event_id.value
            if snapshot.causation_event_id is not None
            else None
        ),
        "payload": _payload_to_record(snapshot.payload),
    }


def event_from_record(record: Any) -> EventEnvelope:
    """Decode and validate one event record."""

    if type(record) is not dict:
        raise EventCodecError("event record must be a JSON object")
    if any(type(key) is not str for key in record):
        raise EventCodecError("event envelope keys must be exact strings")
    if set(record) != _ENVELOPE_FIELDS:
        raise EventCodecError("Invalid envelope fields")
    schema_version = record["schema_version"]
    if type(schema_version) is not int:
        raise EventCodecError("schema_version must be an integer")
    if schema_version != CURRENT_EVENT_SCHEMA_VERSION:
        raise EventCodecError(
            f"Unsupported event schema version {schema_version!r}; "
            f"supported={CURRENT_EVENT_SCHEMA_VERSION}"
        )
    event_type = record["event_type"]
    if type(event_type) is not str:
        raise EventCodecError("event_type must be a string")
    timestamp_text = record["wall_timestamp_utc"]
    if type(timestamp_text) is not str:
        raise EventCodecError("wall_timestamp_utc must be a string")
    try:
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError:
        timestamp = None
    if timestamp is None:
        raise EventCodecError("wall_timestamp_utc is not ISO-8601")
    try:
        payload = _payload_from_record(event_type, record["payload"])
        envelope = EventEnvelope(
            schema_version=schema_version,
            event_id=EventId(record["event_id"]),
            run_id=RunId(record["run_id"]),
            sequence_number=record["sequence_number"],
            event_type=event_type,
            wall_timestamp_utc=timestamp,
            monotonic_offset_ns=record["monotonic_offset_ns"],
            correlation_id=(
                CorrelationId(record["correlation_id"])
                if record["correlation_id"] is not None
                else None
            ),
            causation_event_id=(
                EventId(record["causation_event_id"])
                if record["causation_event_id"] is not None
                else None
            ),
            payload=payload,
        )
    except Exception:
        envelope = None
    if envelope is None:
        raise EventCodecError("Invalid event envelope")
    return envelope


def event_to_json(event: EventEnvelope) -> str:
    """Encode one event as canonical compact JSON (without a trailing newline)."""

    try:
        encoded = json.dumps(
            event_to_record(event),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except Exception:
        encoded = None
    if encoded is None:
        raise EventCodecError("Could not encode event")
    return encoded


def event_from_json(data: str) -> EventEnvelope:
    """Decode one JSON event line."""

    if type(data) is not str:
        raise EventCodecError("event JSON must be an exact string")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        record: Dict[str, Any] = {}
        for key, value in pairs:
            if key in record:
                raise EventCodecError("Duplicate JSON object key")
            record[key] = value
        return record

    def reject_nonstandard_number(value: str) -> Any:
        raise EventCodecError("Non-standard JSON number")

    try:
        record = json.loads(
            data,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonstandard_number,
        )
    except EventCodecError as exc:
        parse_failure = str(exc)
        record = None
    except Exception:
        parse_failure = "syntax"
        record = None
    else:
        parse_failure = None
    if parse_failure not in (None, "syntax"):
        raise EventCodecError(parse_failure)
    if parse_failure == "syntax":
        raise EventCodecError("Malformed event JSON")
    return event_from_record(record)

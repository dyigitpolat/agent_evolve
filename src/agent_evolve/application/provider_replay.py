"""Offline reconstruction and exact replay of recorded provider exchanges.

This module has no provider, HTTP, model, or environment dependency.  It reads
only validated events and sanitized canonical artifacts.
"""

from __future__ import annotations

import hashlib
from dataclasses import InitVar, dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

from agent_evolve.application.artifact_replay import (
    ArtifactRegistrationKey,
    verify_artifact_journal,
)
from agent_evolve.domain.artifact import ArtifactRef, ArtifactRole, artifact_ref_for_bytes
from agent_evolve.domain.event import (
    ArtifactRegistered,
    EventEnvelope,
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallRequested,
    LLMCallRetried,
    LLMCallStarted,
    LLMRequestArtifactLinked,
    ProviderAttemptCompleted,
    ProviderAttemptFailed,
    ProviderAttemptStarted,
    validated_event_snapshot,
)
from agent_evolve.domain.ids import EventId, LLMCallId, ProviderAttemptId, RunId
from agent_evolve.domain.inline_text import InlineTextPolicy, validate_inline_text
from agent_evolve.domain.outcome import (
    FailureCategory,
    FailureCode,
    validate_failure_pair,
)
from agent_evolve.ports.artifact_store import (
    JSON_MEDIA_TYPE,
    ArtifactStore,
    canonical_json_bytes,
    decode_json_bytes,
)

_READ_FAILED = object()
_STREAM_FAILED = object()
_CLONE_FAILED = object()
_BUILD_TOKEN = object()


class RecordedProviderReplayError(RuntimeError):
    """Recorded provider history is missing, inconsistent, or not replayable."""


class RecordedCallNotFoundError(RecordedProviderReplayError):
    pass


class RecordedRequestMismatchError(RecordedProviderReplayError):
    pass


class RecordedCallUnavailableError(RecordedProviderReplayError):
    pass


class RecordedCallStatus(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETED = "completed"
    FAILED = "failed"


class RecordedAttemptStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class _RegistrationOccurrence:
    sequence_number: int
    causation_event_id: Optional[EventId]


@dataclass(frozen=True, slots=True)
class RecordedProviderAttempt:
    provider_attempt_id: ProviderAttemptId
    attempt_number: int
    status: RecordedAttemptStatus
    resolved_provider: Optional[str] = None
    resolved_model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    latency_ns: int = 0
    failure_category: Optional[FailureCategory] = None
    failure_code: Optional[FailureCode] = None
    retryable: Optional[bool] = None
    failure_message: Optional[str] = None
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _BUILD_TOKEN:
            raise TypeError("recorded attempts are constructed only by verified replay")
        _validate_recorded_attempt(self)


@dataclass(frozen=True, slots=True)
class RecordedLLMCall:
    call_id: LLMCallId
    operation: str
    requested_model: str
    request_ref: ArtifactRef
    request_bytes: bytes = field(repr=False)
    status: RecordedCallStatus
    attempts: Tuple[RecordedProviderAttempt, ...]
    response_ref: Optional[ArtifactRef] = None
    response_bytes: Optional[bytes] = field(default=None, repr=False)
    failure_category: Optional[FailureCategory] = None
    failure_code: Optional[FailureCode] = None
    failure_message: Optional[str] = None
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _BUILD_TOKEN:
            raise TypeError("recorded calls are constructed only by verified replay")
        _validate_recorded_call(self)


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_canonical_json_ref(ref: object, content: object) -> bool:
    if type(ref) is not ArtifactRef or type(content) is not bytes:
        return False
    try:
        reconstructed = artifact_ref_for_bytes(content, media_type=JSON_MEDIA_TYPE)
        canonical = canonical_json_bytes(decode_json_bytes(content))
        validated_ref = ArtifactRef(
            artifact_id=ref.artifact_id,
            sha256_hex=ref.sha256_hex,
            size_bytes=ref.size_bytes,
            media_type=ref.media_type,
        )
    except Exception:
        return False
    return canonical == content and validated_ref == reconstructed


def _is_safe_inline(value: object, policy: InlineTextPolicy) -> bool:
    try:
        validate_inline_text(value, field_name="recorded metadata", policy=policy)
    except (TypeError, ValueError):
        return False
    return True


def _validate_recorded_attempt(attempt: RecordedProviderAttempt) -> None:
    if (
        type(attempt.provider_attempt_id) is not ProviderAttemptId
        or type(attempt.attempt_number) is not int
        or attempt.attempt_number < 1
        or type(attempt.status) is not RecordedAttemptStatus
        or any(
            not _is_nonnegative_int(value)
            for value in (
                attempt.input_tokens,
                attempt.output_tokens,
                attempt.reasoning_tokens,
                attempt.cache_read_tokens,
                attempt.cache_write_tokens,
                attempt.latency_ns,
            )
        )
        or type(attempt.cost_usd) is not Decimal
        or not attempt.cost_usd.is_finite()
        or attempt.cost_usd < 0
        or len(attempt.cost_usd.as_tuple().digits) > 64
        or abs(attempt.cost_usd.as_tuple().exponent) > 64
    ):
        raise TypeError("recorded attempt violates the immutable value contract")

    if attempt.status is RecordedAttemptStatus.STARTED:
        if (
            attempt.resolved_provider is not None
            or attempt.resolved_model is not None
            or any(
                value != 0
                for value in (
                    attempt.input_tokens,
                    attempt.output_tokens,
                    attempt.reasoning_tokens,
                    attempt.cache_read_tokens,
                    attempt.cache_write_tokens,
                    attempt.cost_usd,
                    attempt.latency_ns,
                )
            )
            or attempt.failure_category is not None
            or attempt.failure_code is not None
            or attempt.retryable is not None
            or attempt.failure_message is not None
        ):
            raise ValueError("started recorded attempt carries terminal fields")
        return

    if attempt.status is RecordedAttemptStatus.COMPLETED:
        if (
            not _is_safe_inline(
                attempt.resolved_provider,
                InlineTextPolicy.ROUTING_LABEL,
            )
            or not _is_safe_inline(
                attempt.resolved_model,
                InlineTextPolicy.ROUTING_LABEL,
            )
            or attempt.failure_category is not None
            or attempt.failure_code is not None
            or attempt.retryable is not None
            or attempt.failure_message is not None
        ):
            raise ValueError("completed recorded attempt has inconsistent fields")
        return

    if (
        attempt.resolved_provider is not None
        or attempt.resolved_model is not None
        or any(
            value != 0
            for value in (
                attempt.input_tokens,
                attempt.output_tokens,
                attempt.reasoning_tokens,
                attempt.cache_read_tokens,
                attempt.cache_write_tokens,
                attempt.cost_usd,
            )
        )
        or type(attempt.failure_category) is not FailureCategory
        or type(attempt.failure_code) is not FailureCode
        or type(attempt.retryable) is not bool
        or not _is_safe_inline(
            attempt.failure_message,
            InlineTextPolicy.SAFE_SUMMARY,
        )
    ):
        raise ValueError("failed recorded attempt has inconsistent fields")
    validate_failure_pair(attempt.failure_category, attempt.failure_code)


def _validate_recorded_call(call: RecordedLLMCall) -> None:
    if (
        type(call.call_id) is not LLMCallId
        or not _is_safe_inline(call.operation, InlineTextPolicy.METADATA_TOKEN)
        or not _is_safe_inline(call.requested_model, InlineTextPolicy.ROUTING_LABEL)
        or not _is_canonical_json_ref(call.request_ref, call.request_bytes)
        or type(call.status) is not RecordedCallStatus
        or type(call.attempts) is not tuple
        or any(type(attempt) is not RecordedProviderAttempt for attempt in call.attempts)
        or [attempt.attempt_number for attempt in call.attempts]
        != list(range(1, len(call.attempts) + 1))
        or len({attempt.provider_attempt_id for attempt in call.attempts})
        != len(call.attempts)
    ):
        raise TypeError("recorded call violates the immutable value contract")

    if call.status is RecordedCallStatus.COMPLETED:
        if (
            not call.attempts
            or call.attempts[-1].status is not RecordedAttemptStatus.COMPLETED
            or not _is_canonical_json_ref(call.response_ref, call.response_bytes)
            or call.failure_category is not None
            or call.failure_code is not None
            or call.failure_message is not None
        ):
            raise ValueError("completed recorded call has inconsistent fields")
        return

    if call.response_ref is not None or call.response_bytes is not None:
        raise ValueError("non-completed recorded call cannot carry a response")
    if call.status is RecordedCallStatus.INCOMPLETE:
        if (
            call.failure_category is not None
            or call.failure_code is not None
            or call.failure_message is not None
        ):
            raise ValueError("incomplete recorded call cannot carry failure fields")
        return
    if (
        type(call.failure_category) is not FailureCategory
        or type(call.failure_code) is not FailureCode
        or not _is_safe_inline(call.failure_message, InlineTextPolicy.SAFE_SUMMARY)
    ):
        raise ValueError("failed recorded call has inconsistent fields")
    validate_failure_pair(call.failure_category, call.failure_code)


def _clone_recorded_attempt(
    attempt: RecordedProviderAttempt,
) -> RecordedProviderAttempt:
    return RecordedProviderAttempt(
        provider_attempt_id=ProviderAttemptId(attempt.provider_attempt_id.value),
        attempt_number=attempt.attempt_number,
        status=attempt.status,
        resolved_provider=attempt.resolved_provider,
        resolved_model=attempt.resolved_model,
        input_tokens=attempt.input_tokens,
        output_tokens=attempt.output_tokens,
        reasoning_tokens=attempt.reasoning_tokens,
        cache_read_tokens=attempt.cache_read_tokens,
        cache_write_tokens=attempt.cache_write_tokens,
        cost_usd=Decimal(attempt.cost_usd),
        latency_ns=attempt.latency_ns,
        failure_category=attempt.failure_category,
        failure_code=attempt.failure_code,
        retryable=attempt.retryable,
        failure_message=attempt.failure_message,
        _construction_token=_BUILD_TOKEN,
    )


def _clone_recorded_call(call: RecordedLLMCall) -> RecordedLLMCall:
    request_bytes = bytes(call.request_bytes)
    response_bytes = (
        bytes(call.response_bytes) if call.response_bytes is not None else None
    )
    return RecordedLLMCall(
        call_id=LLMCallId(call.call_id.value),
        operation=call.operation,
        requested_model=call.requested_model,
        request_ref=artifact_ref_for_bytes(
            request_bytes,
            media_type=JSON_MEDIA_TYPE,
        ),
        request_bytes=request_bytes,
        status=call.status,
        attempts=tuple(_clone_recorded_attempt(item) for item in call.attempts),
        response_ref=(
            artifact_ref_for_bytes(response_bytes, media_type=JSON_MEDIA_TYPE)
            if response_bytes is not None
            else None
        ),
        response_bytes=response_bytes,
        failure_category=call.failure_category,
        failure_code=call.failure_code,
        failure_message=call.failure_message,
        _construction_token=_BUILD_TOKEN,
    )


def _try_clone_recorded_call(call: RecordedLLMCall) -> RecordedLLMCall | object:
    try:
        _validate_recorded_call(call)
        return _clone_recorded_call(call)
    except Exception:
        return _CLONE_FAILED


def _recorded_call_fingerprint(call: RecordedLLMCall) -> str:
    record = {
        "call_id": call.call_id.value,
        "operation": call.operation,
        "requested_model": call.requested_model,
        "request_sha256": hashlib.sha256(call.request_bytes).hexdigest(),
        "status": call.status.value,
        "attempts": [
            {
                "provider_attempt_id": attempt.provider_attempt_id.value,
                "attempt_number": attempt.attempt_number,
                "status": attempt.status.value,
                "resolved_provider": attempt.resolved_provider,
                "resolved_model": attempt.resolved_model,
                "input_tokens": attempt.input_tokens,
                "output_tokens": attempt.output_tokens,
                "reasoning_tokens": attempt.reasoning_tokens,
                "cache_read_tokens": attempt.cache_read_tokens,
                "cache_write_tokens": attempt.cache_write_tokens,
                "cost_usd": format(attempt.cost_usd, "f"),
                "latency_ns": attempt.latency_ns,
                "failure_category": (
                    attempt.failure_category.value
                    if attempt.failure_category is not None
                    else None
                ),
                "failure_code": (
                    attempt.failure_code.value
                    if attempt.failure_code is not None
                    else None
                ),
                "retryable": attempt.retryable,
                "failure_message": attempt.failure_message,
            }
            for attempt in call.attempts
        ],
        "response_sha256": (
            hashlib.sha256(call.response_bytes).hexdigest()
            if call.response_bytes is not None
            else None
        ),
        "failure_category": (
            call.failure_category.value
            if call.failure_category is not None
            else None
        ),
        "failure_code": (
            call.failure_code.value if call.failure_code is not None else None
        ),
        "failure_message": call.failure_message,
    }
    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


@dataclass(slots=True)
class _AttemptBuilder:
    provider_attempt_id: ProviderAttemptId
    attempt_number: int
    status: RecordedAttemptStatus = RecordedAttemptStatus.STARTED
    resolved_provider: Optional[str] = None
    resolved_model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    latency_ns: int = 0
    failure_category: Optional[FailureCategory] = None
    failure_code: Optional[FailureCode] = None
    retryable: Optional[bool] = None
    failure_message: Optional[str] = None
    terminal_sequence: Optional[int] = None
    terminal_event_id: Optional[EventId] = None

    def freeze(self) -> RecordedProviderAttempt:
        return RecordedProviderAttempt(
            provider_attempt_id=self.provider_attempt_id,
            attempt_number=self.attempt_number,
            status=self.status,
            resolved_provider=self.resolved_provider,
            resolved_model=self.resolved_model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            cost_usd=self.cost_usd,
            latency_ns=self.latency_ns,
            failure_category=self.failure_category,
            failure_code=self.failure_code,
            retryable=self.retryable,
            failure_message=self.failure_message,
            _construction_token=_BUILD_TOKEN,
        )


@dataclass(slots=True)
class _CallBuilder:
    call_id: LLMCallId
    operation: str
    requested_model: str
    requested_sequence: int
    request_ref: Optional[ArtifactRef] = None
    request_bytes: Optional[bytes] = None
    started: bool = False
    status: RecordedCallStatus = RecordedCallStatus.INCOMPLETE
    attempts: list[_AttemptBuilder] = field(default_factory=list)
    open_attempt_id: Optional[ProviderAttemptId] = None
    authorized_retry_number: Optional[int] = None
    response_ref: Optional[ArtifactRef] = None
    response_bytes: Optional[bytes] = None
    failure_category: Optional[FailureCategory] = None
    failure_code: Optional[FailureCode] = None
    failure_message: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.status is not RecordedCallStatus.INCOMPLETE


def _fail() -> RecordedProviderReplayError:
    return RecordedProviderReplayError(
        "recorded provider history violates the offline replay contract"
    )


def _try_snapshot_event_stream(
    events: Iterable[EventEnvelope],
) -> Tuple[EventEnvelope, ...] | object:
    try:
        return tuple(validated_event_snapshot(event) for event in events)
    except Exception:
        return _STREAM_FAILED


@dataclass(frozen=True, slots=True)
class RecordedProviderReplay:
    """Immutable offline provider boundary keyed by recorded logical call ID."""

    run_id: Optional[RunId]
    calls: Tuple[RecordedLLMCall, ...] = field(repr=False)
    _by_call_id: Mapping[LLMCallId, RecordedLLMCall] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _fingerprints: Mapping[LLMCallId, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _BUILD_TOKEN:
            raise TypeError("recorded replay is constructed only by verified replay")
        if type(self.calls) is not tuple:
            raise TypeError("calls must be an immutable tuple")
        if self.run_id is not None and type(self.run_id) is not RunId:
            raise TypeError("run_id must be a RunId or None")
        if any(type(call) is not RecordedLLMCall for call in self.calls):
            raise TypeError("calls must contain RecordedLLMCall values")
        try:
            public_calls = tuple(_try_clone_recorded_call(call) for call in self.calls)
            internal_calls = tuple(_try_clone_recorded_call(call) for call in self.calls)
            canonical_run_id = (
                RunId(self.run_id.value) if self.run_id is not None else None
            )
        except Exception:
            public_calls = None
            internal_calls = None
            canonical_run_id = None
        if public_calls is None or internal_calls is None:
            raise TypeError("recorded replay inputs failed canonical reconstruction")
        if any(call is _CLONE_FAILED for call in public_calls + internal_calls):
            raise TypeError("recorded replay inputs failed canonical reconstruction")
        assert all(isinstance(call, RecordedLLMCall) for call in public_calls)
        assert all(isinstance(call, RecordedLLMCall) for call in internal_calls)
        by_call_id = {call.call_id: call for call in internal_calls}
        if len(by_call_id) != len(internal_calls):
            raise ValueError("recorded calls must have unique call IDs")
        object.__setattr__(self, "run_id", canonical_run_id)
        object.__setattr__(self, "calls", public_calls)
        object.__setattr__(self, "_by_call_id", MappingProxyType(by_call_id))
        object.__setattr__(
            self,
            "_fingerprints",
            MappingProxyType(
                {
                    call_id: _recorded_call_fingerprint(call)
                    for call_id, call in by_call_id.items()
                }
            ),
        )

    def _internal_call(self, call_id: LLMCallId) -> RecordedLLMCall:
        if type(call_id) is not LLMCallId:
            raise TypeError("call_id must be an LLMCallId")
        try:
            canonical_call_id = LLMCallId(call_id.value)
            call = self._by_call_id[canonical_call_id]
        except (KeyError, TypeError, ValueError):
            raise RecordedCallNotFoundError(
                "recorded logical call was not found"
            ) from None
        validated = _try_clone_recorded_call(call)
        if (
            validated is _CLONE_FAILED
            or _recorded_call_fingerprint(call)
            != self._fingerprints.get(canonical_call_id)
        ):
            raise RecordedProviderReplayError(
                "recorded replay snapshot failed its immutable integrity check"
            )
        return call

    def call(self, call_id: LLMCallId) -> RecordedLLMCall:
        call = _try_clone_recorded_call(self._internal_call(call_id))
        if call is _CLONE_FAILED:  # pragma: no cover - guarded by _internal_call.
            raise RecordedProviderReplayError(
                "recorded replay snapshot failed its immutable integrity check"
            )
        assert isinstance(call, RecordedLLMCall)
        return call

    def replay_response(self, call_id: LLMCallId, *, request_bytes: bytes) -> bytes:
        """Return exact recorded response bytes only for the exact recorded request."""

        if type(request_bytes) is not bytes:
            raise TypeError("request_bytes must be immutable bytes")
        call = self._internal_call(call_id)
        if request_bytes != call.request_bytes:
            raise RecordedRequestMismatchError(
                "supplied request does not match the recorded sanitized request"
            )
        if (
            call.status is not RecordedCallStatus.COMPLETED
            or call.response_bytes is None
        ):
            raise RecordedCallUnavailableError(
                "recorded call has no completed response available for replay"
            )
        return call.response_bytes


def _read_registered_bytes(
    store: ArtifactStore,
    registrations: Dict[
        ArtifactRegistrationKey,
        tuple[ArtifactRef, list[_RegistrationOccurrence]],
    ],
    artifact_id,
    role: ArtifactRole,
) -> tuple[ArtifactRef, bytes, Tuple[_RegistrationOccurrence, ...]]:
    key = ArtifactRegistrationKey(artifact_id, role)
    registered = registrations.get(key)
    if registered is None:
        raise _fail()
    ref, registration_occurrences = registered
    content = _try_read_canonical_registered_bytes(store, ref)
    if content is _READ_FAILED:
        raise _fail()
    assert isinstance(content, bytes)
    return ref, content, tuple(registration_occurrences)


def _try_read_canonical_registered_bytes(
    store: ArtifactStore,
    ref: ArtifactRef,
) -> bytes | object:
    try:
        content = store.read_bytes(
            ref.artifact_id,
            expected_media_type=JSON_MEDIA_TYPE,
        )
        reread_ref = artifact_ref_for_bytes(content, media_type=JSON_MEDIA_TYPE)
        canonical = canonical_json_bytes(decode_json_bytes(content))
        if reread_ref != ref or canonical != content:
            return _READ_FAILED
        return content
    except Exception:
        # Return from the exception handler so the public replay error cannot
        # retain secret-bearing store/codec exception context.
        return _READ_FAILED


def _require_call(
    calls: Dict[LLMCallId, _CallBuilder],
    call_id: LLMCallId,
) -> _CallBuilder:
    call = calls.get(call_id)
    if call is None:
        raise _fail()
    return call


def build_recorded_provider_replay(
    events: Iterable[EventEnvelope],
    *,
    artifact_store: ArtifactStore,
) -> RecordedProviderReplay:
    """Validate and reconstruct an offline replay boundary from one run stream."""

    stream = _try_snapshot_event_stream(events)
    if stream is _STREAM_FAILED:
        raise _fail()
    assert isinstance(stream, tuple)
    report = verify_artifact_journal(stream, artifact_store=artifact_store)
    registrations: Dict[
        ArtifactRegistrationKey,
        tuple[ArtifactRef, list[_RegistrationOccurrence]],
    ] = {}
    calls: Dict[LLMCallId, _CallBuilder] = {}
    call_order: list[LLMCallId] = []
    attempts_by_id: Dict[ProviderAttemptId, tuple[LLMCallId, _AttemptBuilder]] = {}
    seen_event_ids: set[EventId] = set()

    for event in stream:
        if event.event_id in seen_event_ids or (
            event.causation_event_id is not None
            and event.causation_event_id not in seen_event_ids
        ):
            raise _fail()
        seen_event_ids.add(event.event_id)
        payload = event.payload
        if isinstance(payload, ArtifactRegistered):
            registration_key = ArtifactRegistrationKey(
                payload.artifact_id,
                payload.role,
            )
            registered = registrations.get(registration_key)
            if registered is None:
                registrations[registration_key] = (
                    payload.artifact_ref,
                    [
                        _RegistrationOccurrence(
                            sequence_number=event.sequence_number,
                            causation_event_id=event.causation_event_id,
                        )
                    ],
                )
            else:
                registered[1].append(
                    _RegistrationOccurrence(
                        sequence_number=event.sequence_number,
                        causation_event_id=event.causation_event_id,
                    )
                )
            continue

        if isinstance(payload, LLMCallRequested):
            if payload.call_id in calls:
                raise _fail()
            calls[payload.call_id] = _CallBuilder(
                call_id=payload.call_id,
                operation=payload.operation,
                requested_model=payload.requested_model,
                requested_sequence=event.sequence_number,
            )
            call_order.append(payload.call_id)
            continue

        if isinstance(payload, LLMRequestArtifactLinked):
            call = _require_call(calls, payload.call_id)
            if call.request_ref is not None or call.started or call.terminal:
                raise _fail()
            (
                call.request_ref,
                call.request_bytes,
                registration_occurrences,
            ) = _read_registered_bytes(
                artifact_store,
                registrations,
                payload.request_artifact_id,
                ArtifactRole.LLM_REQUEST,
            )
            if not any(
                occurrence.sequence_number < call.requested_sequence
                for occurrence in registration_occurrences
            ):
                raise _fail()
            continue

        if isinstance(payload, LLMCallStarted):
            call = _require_call(calls, payload.call_id)
            if call.started or call.terminal or call.request_ref is None:
                raise _fail()
            call.started = True
            continue

        if isinstance(payload, ProviderAttemptStarted):
            call = _require_call(calls, payload.call_id)
            expected_number = len(call.attempts) + 1
            if (
                not call.started
                or call.terminal
                or call.open_attempt_id is not None
                or payload.provider_attempt_id in attempts_by_id
                or payload.attempt_number != expected_number
                or (
                    expected_number > 1
                    and call.authorized_retry_number != expected_number
                )
            ):
                raise _fail()
            attempt = _AttemptBuilder(
                provider_attempt_id=payload.provider_attempt_id,
                attempt_number=payload.attempt_number,
            )
            call.attempts.append(attempt)
            call.open_attempt_id = payload.provider_attempt_id
            call.authorized_retry_number = None
            attempts_by_id[payload.provider_attempt_id] = (payload.call_id, attempt)
            continue

        if isinstance(payload, (ProviderAttemptCompleted, ProviderAttemptFailed)):
            call = _require_call(calls, payload.call_id)
            found = attempts_by_id.get(payload.provider_attempt_id)
            if (
                found is None
                or found[0] != payload.call_id
                or call.open_attempt_id != payload.provider_attempt_id
                or call.terminal
                or found[1].status is not RecordedAttemptStatus.STARTED
            ):
                raise _fail()
            attempt = found[1]
            if isinstance(payload, ProviderAttemptCompleted):
                attempt.status = RecordedAttemptStatus.COMPLETED
                attempt.resolved_provider = payload.resolved_provider
                attempt.resolved_model = payload.resolved_model
                attempt.input_tokens = payload.input_tokens
                attempt.output_tokens = payload.output_tokens
                attempt.reasoning_tokens = payload.reasoning_tokens
                attempt.cache_read_tokens = payload.cache_read_tokens
                attempt.cache_write_tokens = payload.cache_write_tokens
                attempt.cost_usd = payload.cost_usd
                attempt.latency_ns = payload.latency_ns
            else:
                attempt.status = RecordedAttemptStatus.FAILED
                attempt.failure_category = payload.category
                attempt.failure_code = payload.code
                attempt.retryable = payload.retryable
                attempt.failure_message = payload.message
                attempt.latency_ns = payload.latency_ns
            attempt.terminal_sequence = event.sequence_number
            attempt.terminal_event_id = event.event_id
            call.open_attempt_id = None
            continue

        if isinstance(payload, LLMCallRetried):
            call = _require_call(calls, payload.call_id)
            if (
                not call.started
                or call.terminal
                or call.open_attempt_id is not None
                or not call.attempts
                or call.attempts[-1].status is not RecordedAttemptStatus.FAILED
                or not call.attempts[-1].retryable
                or call.authorized_retry_number is not None
                or payload.next_attempt_number != len(call.attempts) + 1
            ):
                raise _fail()
            call.authorized_retry_number = payload.next_attempt_number
            continue

        if isinstance(payload, LLMCallCompleted):
            call = _require_call(calls, payload.call_id)
            if (
                not call.started
                or call.terminal
                or call.open_attempt_id is not None
                or not call.attempts
                or call.attempts[-1].status is not RecordedAttemptStatus.COMPLETED
                or payload.response_artifact_id is None
            ):
                raise _fail()
            (
                call.response_ref,
                call.response_bytes,
                registration_occurrences,
            ) = _read_registered_bytes(
                artifact_store,
                registrations,
                payload.response_artifact_id,
                ArtifactRole.LLM_RESPONSE,
            )
            if (
                call.attempts[-1].terminal_sequence is None
                or call.attempts[-1].terminal_event_id is None
                or not any(
                    occurrence.sequence_number
                    > call.attempts[-1].terminal_sequence
                    and occurrence.causation_event_id
                    == call.attempts[-1].terminal_event_id
                    for occurrence in registration_occurrences
                )
            ):
                raise _fail()
            call.status = RecordedCallStatus.COMPLETED
            continue

        if isinstance(payload, LLMCallFailed):
            call = _require_call(calls, payload.call_id)
            if (
                not call.started
                or call.terminal
                or call.open_attempt_id is not None
                or call.authorized_retry_number is not None
            ):
                raise _fail()
            call.status = RecordedCallStatus.FAILED
            call.failure_category = payload.category
            call.failure_code = payload.code
            call.failure_message = payload.message

    frozen_calls: list[RecordedLLMCall] = []
    for call_id in call_order:
        call = calls[call_id]
        if call.request_ref is None or call.request_bytes is None:
            raise _fail()
        frozen_calls.append(
            RecordedLLMCall(
                call_id=call.call_id,
                operation=call.operation,
                requested_model=call.requested_model,
                request_ref=call.request_ref,
                request_bytes=call.request_bytes,
                status=call.status,
                attempts=tuple(attempt.freeze() for attempt in call.attempts),
                response_ref=call.response_ref,
                response_bytes=call.response_bytes,
                failure_category=call.failure_category,
                failure_code=call.failure_code,
                failure_message=call.failure_message,
                _construction_token=_BUILD_TOKEN,
            )
        )
    return RecordedProviderReplay(
        run_id=report.run_id,
        calls=tuple(frozen_calls),
        _construction_token=_BUILD_TOKEN,
    )

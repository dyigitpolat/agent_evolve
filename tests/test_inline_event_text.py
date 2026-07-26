"""Closed inline event-string schema and adversarial secret-bypass tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest

import agent_evolve.domain.event as event_module
from agent_evolve.application.event_recorder import EventRecorder
from agent_evolve.domain.artifact import ArtifactRole
from agent_evolve.domain.event import (
    ArtifactRegistered,
    CURRENT_EVENT_SCHEMA_VERSION,
    EventEnvelope,
    EvaluationCompleted,
    EvaluationFailed,
    LLMCallFailed,
    LLMCallRequested,
    LLMCallStarted,
    LLMRequestArtifactLinked,
    ProviderAttemptCompleted,
    RunFinished,
    event_from_record,
    event_from_json,
    event_to_record,
    event_to_json,
    validate_event_value_schema,
    validate_inline_text_schema,
)
from agent_evolve.domain.ids import (
    ArtifactId,
    EvaluationAttemptId,
    EvaluationId,
    EventId,
    LLMCallId,
    ProviderAttemptId,
    RunId,
)
from agent_evolve.domain.outcome import FailureCategory, FailureCode
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.infrastructure.events import InMemoryEventStore
from agent_evolve.infrastructure.ids import DeterministicIdFactory


def _envelope(payload):
    return EventEnvelope(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        event_id=EventId("event_inline_text"),
        run_id=RunId("run_inline_text"),
        sequence_number=1,
        event_type=payload.EVENT_TYPE,
        wall_timestamp_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
        monotonic_offset_ns=0,
        correlation_id=None,
        causation_event_id=None,
        payload=payload,
    )


def test_request_artifact_link_event_is_typed_frozen_and_canonical():
    payload = LLMRequestArtifactLinked(
        call_id=LLMCallId("call_inline"),
        request_artifact_id=ArtifactId(f"artifact_{'a' * 64}"),
    )
    event = _envelope(payload)
    assert event_from_json(event_to_json(event)) == event


def test_complete_inline_string_vocabulary_has_explicit_policies(monkeypatch):
    validate_event_value_schema()
    validate_inline_text_schema()
    monkeypatch.delitem(
        event_module._INLINE_TEXT_POLICIES[LLMCallRequested],
        "operation",
    )
    with pytest.raises(RuntimeError, match="do not match"):
        validate_inline_text_schema()


@pytest.mark.parametrize(
    "payload, raw_secret",
    [
        (
            LLMCallRequested(
                LLMCallId("call_case_a"),
                "system prompt: optimize this candidate",
                "model",
            ),
            "system prompt",
        ),
        (
            LLMCallRequested(
                LLMCallId("call_case_b"),
                "generate",
                "sk-abcdefghijklmnopqrstuv",
            ),
            "sk-abcdefghijklmnopqrstuv",
        ),
        (
            LLMCallFailed(
                call_id=LLMCallId("call_case_c"),
                category=FailureCategory.SYSTEM,
                code=FailureCode.PARSER_FAILURE,
                retryable=False,
                message="Authorization: Bearer abcdefghijklmnop",
            ),
            "abcdefghijklmnop",
        ),
        (
            LLMCallFailed(
                call_id=LLMCallId("call_case_d"),
                category=FailureCategory.SYSTEM,
                code=FailureCode.PARSER_FAILURE,
                retryable=False,
                message="Cookie: sessionid=abcdefghijklmnop",
            ),
            "sessionid",
        ),
        (
            RunFinished('{"assistant_response":"raw output"}'),
            "raw output",
        ),
        (
            ProviderAttemptCompleted(
                call_id=LLMCallId("call_provider"),
                provider_attempt_id=ProviderAttemptId("provider_attempt_inline"),
                resolved_provider="openrouter",
                resolved_model="or-v1-abcdefghijklmnopqrstuv",
            ),
            "or-v1-abcdefghijklmnopqrstuv",
        ),
        (
            EvaluationFailed(
                evaluation_id=EvaluationId("evaluation_inline"),
                fidelity="full",
                category=FailureCategory.SYSTEM,
                code=FailureCode.INTERNAL_BUG,
                retryable=False,
                terminal=True,
                message="opaque_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            ),
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        ),
    ],
)
def test_event_envelope_rejects_prompt_response_diagnostic_or_credential_content(
    payload,
    raw_secret,
):
    with pytest.raises(ValueError, match="inline-text policy") as caught:
        _envelope(payload)
    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_non_utf8_inline_error_does_not_retain_secret_exception_context():
    raw = "TOP-SECRET-\ud800"
    payload = RunFinished(raw)
    with pytest.raises(ValueError) as caught:
        _envelope(payload)
    assert "TOP-SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_recorder_fails_before_event_store_append_for_unsafe_inline_content():
    store = InMemoryEventStore()
    ids = DeterministicIdFactory("inline_reject")
    run_id = ids.new_run_id()
    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=FakeClock(),
    )
    with pytest.raises(ValueError, match="inline-text policy"):
        recorder.record(
            LLMCallRequested(
                call_id=ids.new_llm_call_id(),
                operation="return JSON containing all tool output",
                requested_model="model",
            )
        )
    assert store.read(run_id) == ()
    assert recorder.last_sequence == 0


def test_short_classification_summary_remains_backward_compatible():
    event = _envelope(RunFinished("continued safely"))
    assert event.payload.stop_reason == "continued safely"


@pytest.mark.parametrize(
    "raw_value",
    [
        "call_sk-abcdefghijklmnopqrstuv",
        "call_system_prompt_optimize_candidate",
        "call_" + "x" * 129,
    ],
)
def test_stable_ids_reject_credential_content_and_unbounded_values_without_echo(
    raw_value,
):
    with pytest.raises(ValueError, match="durable storage") as caught:
        LLMCallId(raw_value)
    assert raw_value not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "namespace",
    [
        "sk-abcdefghijklmnopqrstuv",
        "system_prompt_payload",
        "n" * 49,
    ],
)
def test_deterministic_id_namespaces_are_bounded_non_content_metadata(namespace):
    with pytest.raises(ValueError, match="identifier policy") as caught:
        DeterministicIdFactory(namespace)
    assert namespace not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def _forged_llm_call_id(raw_value: str) -> LLMCallId:
    forged = object.__new__(LLMCallId)
    object.__setattr__(forged, "value", raw_value)
    return forged


def test_forged_secret_bearing_payload_id_fails_before_event_store_append():
    raw_secret = "call_sk-abcdefghijklmnopqrstuv"
    store = InMemoryEventStore()
    ids = DeterministicIdFactory("id_reject")
    run_id = ids.new_run_id()
    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=FakeClock(),
    )

    with pytest.raises(TypeError, match="Invalid payload field") as caught:
        recorder.record(LLMCallStarted(_forged_llm_call_id(raw_secret)))

    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert store.read(run_id) == ()
    assert recorder.last_sequence == 0


@pytest.mark.parametrize("location", ["payload", "event_id"])
def test_codec_rejects_secret_bearing_ids_without_echo_or_exception_context(location):
    raw_secret = "sk-abcdefghijklmnopqrstuv"
    record = event_to_record(_envelope(LLMCallStarted(LLMCallId("call_case_e"))))
    if location == "payload":
        record["payload"]["call_id"] = f"call_{raw_secret}"
    else:
        record["event_id"] = f"event_{raw_secret}"

    with pytest.raises(event_module.EventCodecError) as caught:
        event_from_record(record)

    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_codec_errors_do_not_echo_untrusted_enum_decimal_or_container_text():
    enum_secret = "sk-abcdefghijklmnopqrstuv"
    enum_record = event_to_record(
        _envelope(
            EvaluationFailed(
                evaluation_id=EvaluationId("evaluation_case_c"),
                fidelity="full",
                category=FailureCategory.SYSTEM,
                code=FailureCode.INTERNAL_BUG,
                retryable=False,
                terminal=True,
                message="internal classification failure",
            )
        )
    )
    enum_record["payload"]["category"] = enum_secret

    decimal_secret = "token_abcdefghijklmnopqrstuv"
    decimal_record = event_to_record(
        _envelope(
            ProviderAttemptCompleted(
                call_id=LLMCallId("call_case_f"),
                provider_attempt_id=ProviderAttemptId("provider_attempt_case_f"),
                resolved_provider="provider",
                resolved_model="provider/model",
                cost_usd=Decimal("0"),
            )
        )
    )
    decimal_record["payload"]["cost_usd"] = decimal_secret

    nested_secret = "system_prompt_optimize_candidate"
    nested_record = event_to_record(
        _envelope(
            EvaluationCompleted(
                evaluation_id=EvaluationId("evaluation_case_d"),
                evaluation_attempt_id=EvaluationAttemptId(
                    "evaluation_attempt_case_e"
                ),
                fidelity="full",
                objective_values=(("score", 1.0),),
            )
        )
    )
    nested_record["payload"]["objective_values"] = [[nested_secret, 1.0]]

    for record, raw_value in (
        (enum_record, enum_secret),
        (decimal_record, decimal_secret),
        (nested_record, nested_secret),
    ):
        with pytest.raises(event_module.EventCodecError) as caught:
            event_from_record(record)
        assert raw_value not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


def test_decimal_text_is_bounded_before_fixed_point_encoding():
    with pytest.raises(TypeError, match="bounded finite Decimal"):
        ProviderAttemptCompleted(
            call_id=LLMCallId("call_case_g"),
            provider_attempt_id=ProviderAttemptId("provider_attempt_case_g"),
            resolved_provider="provider",
            resolved_model="provider/model",
            cost_usd=Decimal("1e1000000"),
        )


@pytest.mark.parametrize("case", ["malformed", "duplicate_key"])
def test_json_parser_errors_discard_secret_bearing_input_context(case):
    raw_secret = "sk-abcdefghijklmnopqrstuv"
    if case == "malformed":
        data = f'{{"value":"{raw_secret}"'
    else:
        data = f'{{"{raw_secret}":1,"{raw_secret}":2}}'

    with pytest.raises(event_module.EventCodecError) as caught:
        event_from_json(data)

    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_hostile_event_type_string_subclass_cannot_spoof_payload_equality():
    raw_secret = "sk-abcdefghijklmnopqrstuv"

    class SpoofedEventType(str):
        def __ne__(self, other):
            return False

    with pytest.raises(TypeError, match="exact string") as caught:
        EventEnvelope(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            event_id=EventId("event_case_h"),
            run_id=RunId("run_case_h"),
            sequence_number=1,
            event_type=SpoofedEventType(raw_secret),
            wall_timestamp_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            monotonic_offset_ns=0,
            correlation_id=None,
            causation_event_id=None,
            payload=LLMCallStarted(LLMCallId("call_case_h")),
        )

    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_hostile_datetime_subclass_cannot_control_timestamp_serialization():
    class SpoofedDatetime(datetime):
        def isoformat(self, *args, **kwargs):
            return "sk-abcdefghijklmnopqrstuv"

    timestamp = SpoofedDatetime(2026, 7, 13, tzinfo=timezone.utc)
    with pytest.raises(TypeError, match="exact datetime"):
        EventEnvelope(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            event_id=EventId("event_case_i"),
            run_id=RunId("run_case_i"),
            sequence_number=1,
            event_type=LLMCallStarted.EVENT_TYPE,
            wall_timestamp_utc=timestamp,
            monotonic_offset_ns=0,
            correlation_id=None,
            causation_event_id=None,
            payload=LLMCallStarted(LLMCallId("call_case_i")),
        )


def test_stateful_zero_offset_tzinfo_cannot_shift_timestamp_during_snapshot():
    class StatefulTimezone(tzinfo):
        def __init__(self):
            self.calls = 0

        def utcoffset(self, value):
            self.calls += 1
            if self.calls == 1:
                return timedelta(0)
            return timedelta(hours=1)

        def dst(self, value):
            return timedelta(0)

    timestamp = datetime(2026, 7, 13, tzinfo=StatefulTimezone())
    with pytest.raises(ValueError, match="canonical UTC"):
        EventEnvelope(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            event_id=EventId("event_case_j"),
            run_id=RunId("run_case_j"),
            sequence_number=1,
            event_type=LLMCallStarted.EVENT_TYPE,
            wall_timestamp_utc=timestamp,
            monotonic_offset_ns=0,
            correlation_id=None,
            causation_event_id=None,
            payload=LLMCallStarted(LLMCallId("call_case_j")),
        )


def test_envelope_ids_and_integer_fields_require_exact_runtime_types():
    class EventIdSubclass(EventId):
        pass

    class IntegerSubclass(int):
        pass

    payload = LLMCallStarted(LLMCallId("call_case_k"))
    with pytest.raises(TypeError, match="event_id"):
        EventEnvelope(
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            event_id=EventIdSubclass("event_case_k"),
            run_id=RunId("run_case_k"),
            sequence_number=1,
            event_type=payload.EVENT_TYPE,
            wall_timestamp_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            monotonic_offset_ns=0,
            correlation_id=None,
            causation_event_id=None,
            payload=payload,
        )
    with pytest.raises(TypeError, match="schema_version"):
        EventEnvelope(
            schema_version=IntegerSubclass(CURRENT_EVENT_SCHEMA_VERSION),
            event_id=EventId("event_case_l"),
            run_id=RunId("run_case_l"),
            sequence_number=1,
            event_type=payload.EVENT_TYPE,
            wall_timestamp_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
            monotonic_offset_ns=0,
            correlation_id=None,
            causation_event_id=None,
            payload=payload,
        )


def test_serializer_revalidates_mutated_envelope_without_echo_or_context():
    raw_secret = "codec_secret_material_123"
    event = _envelope(LLMCallStarted(LLMCallId("call_case_m")))

    class ExplosiveTimestamp:
        def astimezone(self, zone):
            raise ValueError(raw_secret)

    object.__setattr__(event, "wall_timestamp_utc", ExplosiveTimestamp())
    with pytest.raises(event_module.EventCodecError) as caught:
        event_to_json(event)
    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_serializer_rejects_low_level_mutation_to_exact_noncanonical_timezone():
    event = _envelope(LLMCallStarted(LLMCallId("call_case_timezone")))
    object.__setattr__(
        event,
        "wall_timestamp_utc",
        datetime(2026, 7, 13, 12, tzinfo=timezone(timedelta(hours=5))),
    )

    with pytest.raises(event_module.EventCodecError) as caught:
        event_to_json(event)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "raw_parameter",
    [
        "prompt=ignore_previous_instructions",
        "raw_output=private_answer",
        "tool_output=private_answer",
        "response=private_answer",
    ],
)
def test_media_type_inline_policy_is_bounded_and_rejects_content_parameters(
    raw_parameter,
):
    common = {
        "artifact_id": ArtifactId(f"artifact_{'e' * 64}"),
        "sha256_hex": "f" * 64,
        "size_bytes": 2,
        "role": ArtifactRole.LLM_REQUEST,
        "minimization_policy_id": "allowlist",
        "minimization_policy_version": "1",
        "minimization_policy_config_sha256": "a" * 64,
        "sanitization_policy_id": "strict_json",
        "sanitization_policy_version": "1",
    }
    with pytest.raises(ValueError, match="metadata policy") as caught:
        ArtifactRegistered(
            media_type=f"application/json; {raw_parameter}",
            **common,
        )
    assert raw_parameter not in str(caught.value)


@pytest.mark.parametrize(
    "media_type",
    [
        "application/json",
        "application/octet-stream",
        "text/plain",
        "text/plain; charset=utf-8",
    ],
)
def test_media_type_inline_policy_accepts_only_needed_canonical_examples(media_type):
    ArtifactRegistered(
        artifact_id=ArtifactId(f"artifact_{'e' * 64}"),
        sha256_hex="f" * 64,
        size_bytes=2,
        media_type=media_type,
        role=ArtifactRole.LLM_REQUEST,
        minimization_policy_id="allowlist",
        minimization_policy_version="1",
        minimization_policy_config_sha256="a" * 64,
        sanitization_policy_id="strict_json",
        sanitization_policy_version="1",
    )


def test_media_type_inline_policy_rejects_oversized_metadata():
    common = {
        "artifact_id": ArtifactId(f"artifact_{'e' * 64}"),
        "sha256_hex": "f" * 64,
        "size_bytes": 2,
        "role": ArtifactRole.LLM_REQUEST,
        "minimization_policy_id": "allowlist",
        "minimization_policy_version": "1",
        "minimization_policy_config_sha256": "a" * 64,
        "sanitization_policy_id": "strict_json",
        "sanitization_policy_version": "1",
    }

    with pytest.raises(ValueError, match="metadata limit"):
        ArtifactRegistered(
            media_type="application/" + "x" * 300,
            **common,
        )

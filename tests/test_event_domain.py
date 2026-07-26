"""Typed IDs, frozen events, codec, and deterministic clock/factory contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from agent_evolve.domain.event import (
    CURRENT_EVENT_SCHEMA_VERSION,
    EventCodecError,
    EventEnvelope,
    EvaluationCompleted,
    EvaluationFailed,
    LLMCallRetried,
    ProviderAttemptCompleted,
    event_from_json,
    event_to_json,
)
from agent_evolve.domain.ids import (
    ArtifactId,
    CandidateId,
    EvaluationAttemptId,
    EvaluationId,
    EventId,
    InsightId,
    LLMCallId,
    ProviderAttemptId,
    RunId,
)
from agent_evolve.domain.outcome import FailureCategory, FailureCode, FailureRecord
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.infrastructure.ids import DeterministicIdFactory, UuidIdFactory


def _envelope(payload):
    return EventEnvelope(
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        event_id=EventId("event_test_000001"),
        run_id=RunId("run_test_000001"),
        sequence_number=1,
        event_type=payload.EVENT_TYPE,
        wall_timestamp_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
        monotonic_offset_ns=10,
        correlation_id=None,
        causation_event_id=None,
        payload=payload,
    )


def test_id_types_are_runtime_distinct_and_prefix_checked():
    run_id = RunId("run_x")
    candidate_id = CandidateId("candidate_x")
    assert type(run_id) is not type(candidate_id)
    assert run_id.value != candidate_id.value
    with pytest.raises(ValueError, match="must start"):
        CandidateId("run_x")
    with pytest.raises(ValueError, match="unsafe"):
        RunId("run_../../escape")


def test_deterministic_and_uuid_factories_cover_stable_id_kinds():
    ids = DeterministicIdFactory("fixture")
    assert ids.new_run_id() == RunId("run_fixture_000001")
    assert ids.new_run_id() == RunId("run_fixture_000002")
    # Counters are independent by runtime ID type.
    assert ids.new_event_id() == EventId("event_fixture_000001")
    assert ids.new_candidate_id() == CandidateId("candidate_fixture_000001")
    assert ids.new_insight_id() == InsightId("insight_fixture_000001")

    uuid_ids = UuidIdFactory()
    first, second = uuid_ids.new_event_id(), uuid_ids.new_event_id()
    assert first != second
    assert first.value.startswith("event_")


def test_fake_clock_advances_wall_and_monotonic_time_together():
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc), start_monotonic_ns=7)
    clock.advance_ns(2_000_000)
    assert clock.monotonic_ns() == 2_000_007
    assert clock.utc_now() == datetime(2026, 1, 1, 0, 0, 0, 2000, tzinfo=timezone.utc)


def test_event_and_payload_are_frozen_and_objectives_must_be_immutable():
    payload = EvaluationCompleted(
        evaluation_id=EvaluationId("evaluation_x"),
        evaluation_attempt_id=EvaluationAttemptId("evaluation_attempt_x"),
        fidelity="full",
        objective_values=(("score", 1.0),),
    )
    event = _envelope(payload)
    with pytest.raises(FrozenInstanceError):
        event.sequence_number = 2
    with pytest.raises(FrozenInstanceError):
        payload.fidelity = "cheap"
    with pytest.raises(TypeError, match="immutable tuple"):
        EvaluationCompleted(
            evaluation_id=EvaluationId("evaluation_y"),
            evaluation_attempt_id=EvaluationAttemptId("evaluation_attempt_y"),
            fidelity="full",
            objective_values=[["score", 1.0]],
        )

    class TupleSubclass(tuple):
        pass

    with pytest.raises(TypeError, match="immutable tuple"):
        EvaluationCompleted(
            evaluation_id=EvaluationId("evaluation_tuple_subclass"),
            evaluation_attempt_id=EvaluationAttemptId(
                "evaluation_attempt_tuple_subclass"
            ),
            fidelity="full",
            objective_values=TupleSubclass((("score", 1.0),)),
        )


def test_decimal_subclasses_cannot_override_durable_fixed_point_encoding():
    class DecimalSubclass(Decimal):
        def __format__(self, specification):
            return "123456789"

    with pytest.raises(TypeError, match="bounded finite Decimal"):
        ProviderAttemptCompleted(
            call_id=LLMCallId("call_decimal_subclass"),
            provider_attempt_id=ProviderAttemptId(
                "provider_attempt_decimal_subclass"
            ),
            resolved_provider="provider",
            resolved_model="provider/model",
            cost_usd=DecimalSubclass("1"),
        )


def test_event_json_round_trip_preserves_types_decimal_and_tuple():
    payload = ProviderAttemptCompleted(
        call_id=LLMCallId("call_x"),
        provider_attempt_id=ProviderAttemptId("provider_attempt_x"),
        resolved_provider="openrouter",
        resolved_model="deepseek/deepseek-v4",
        input_tokens=12,
        output_tokens=4,
        cost_usd=Decimal("0.000123"),
        latency_ns=99,
    )
    event = _envelope(payload)
    encoded = event_to_json(event)
    decoded = event_from_json(encoded)
    assert decoded == event
    assert isinstance(decoded.payload.cost_usd, Decimal)


def test_failure_events_preserve_safe_diagnostics_and_require_a_message():
    payload = EvaluationFailed(
        evaluation_id=EvaluationId("evaluation_case_a"),
        fidelity="full",
        category=FailureCategory.SYSTEM,
        code=FailureCode.PARSER_FAILURE,
        retryable=False,
        terminal=True,
        message="structured response could not be parsed",
        evaluation_attempt_id=EvaluationAttemptId("evaluation_attempt_case_a"),
        worker_time_ns=17,
        exception_type="OutputParserError",
        diagnostics_artifact_id=ArtifactId(f"artifact_{'d' * 64}"),
    )
    assert event_from_json(event_to_json(_envelope(payload))).payload == payload

    with pytest.raises(ValueError, match="non-empty string"):
        EvaluationFailed(
            evaluation_id=EvaluationId("evaluation_case_b"),
            fidelity="full",
            category=FailureCategory.SYSTEM,
            code=FailureCode.PARSER_FAILURE,
            retryable=False,
            terminal=True,
            message=" ",
        )
    with pytest.raises(ValueError, match="requires an evaluation_attempt_id"):
        EvaluationFailed(
            evaluation_id=EvaluationId("evaluation_missing_attempt"),
            fidelity="full",
            category=FailureCategory.INFRASTRUCTURE,
            code=FailureCode.PROCESS_START_FAILURE,
            retryable=True,
            terminal=False,
            message="worker launch failed",
            worker_time_ns=1,
        )


def test_codec_rejects_unknown_schema_and_envelope_payload_mismatch():
    payload = EvaluationCompleted(
        evaluation_id=EvaluationId("evaluation_x"),
        evaluation_attempt_id=EvaluationAttemptId("evaluation_attempt_x"),
        fidelity="full",
        objective_values=(("score", 1.0),),
    )
    encoded = event_to_json(_envelope(payload)).replace(
        '"schema_version":1', '"schema_version":999'
    )
    with pytest.raises(EventCodecError, match="Unsupported event schema"):
        event_from_json(encoded)
    with pytest.raises(ValueError, match="does not match payload"):
        EventEnvelope(
            schema_version=1,
            event_id=EventId("event_other"),
            run_id=RunId("run_other"),
            sequence_number=1,
            event_type="RunStarted",
            wall_timestamp_utc=datetime.now(timezone.utc),
            monotonic_offset_ns=0,
            correlation_id=None,
            causation_event_id=None,
            payload=payload,
        )

    with pytest.raises(TypeError, match="evaluation_id"):
        _envelope(
            EvaluationCompleted(
                evaluation_id=RunId("run_wrong_type"),
                evaluation_attempt_id=EvaluationAttemptId("evaluation_attempt_x"),
                fidelity="full",
                objective_values=(("score", 1.0),),
            )
        )


def test_codec_rejects_ambiguous_or_nonstandard_json():
    encoded = event_to_json(
        _envelope(
            EvaluationCompleted(
                evaluation_id=EvaluationId("evaluation_json"),
                evaluation_attempt_id=EvaluationAttemptId("evaluation_attempt_json"),
                fidelity="full",
                objective_values=(("score", 1.0),),
            )
        )
    )
    duplicate_key = encoded.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
    )
    with pytest.raises(EventCodecError, match="Duplicate JSON object key"):
        event_from_json(duplicate_key)
    with pytest.raises(EventCodecError, match="schema_version must be an integer"):
        event_from_json(encoded.replace('"schema_version":1', '"schema_version":true'))
    with pytest.raises(EventCodecError, match="Non-standard JSON number"):
        event_from_json(encoded.replace('"score",1.0', '"score",NaN'))


def test_failure_code_must_belong_to_category():
    with pytest.raises(ValueError, match="does not belong"):
        FailureRecord(
            category=FailureCategory.CANDIDATE,
            code=FailureCode.INTERNAL_BUG,
            message="wrong category",
        )


def test_retry_and_terminal_failure_invariants_are_validated():
    with pytest.raises(ValueError, match="at least 2"):
        LLMCallRetried(LLMCallId("call_x"), 1)
    with pytest.raises(ValueError, match="must be terminal"):
        EvaluationFailed(
            evaluation_id=EvaluationId("evaluation_x"),
            fidelity="full",
            category=FailureCategory.CANDIDATE,
            code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
            retryable=True,
            terminal=False,
            message="invalid candidate",
        )

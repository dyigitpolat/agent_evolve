"""Offline request/response reconstruction and provider replay contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from agent_evolve.application.artifact_journal import ArtifactJournal
from agent_evolve.application.event_recorder import EventRecorder
from agent_evolve.application.provider_replay import (
    RecordedAttemptStatus,
    RecordedCallStatus,
    RecordedCallUnavailableError,
    RecordedLLMCall,
    RecordedProviderAttempt,
    RecordedProviderReplay,
    RecordedProviderReplayError,
    RecordedRequestMismatchError,
    build_recorded_provider_replay,
)
from agent_evolve.domain.artifact import ArtifactRole
from agent_evolve.domain.event import (
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallRequested,
    LLMCallRetried,
    LLMCallStarted,
    LLMRequestArtifactLinked,
    ProviderAttemptCompleted,
    ProviderAttemptFailed,
    ProviderAttemptStarted,
)
from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.domain.outcome import FailureCategory, FailureCode
from agent_evolve.infrastructure.artifacts import InMemoryArtifactStore
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.infrastructure.events import InMemoryEventStore
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.infrastructure.sanitization import (
    StrictJsonSanitizer,
    TopLevelAllowlistMinimizer,
)
from agent_evolve.ports.artifact_store import canonical_json_bytes

RAW_SECRET = "runtime-provider-secret-12345"


def _setup():
    artifact_store = InMemoryArtifactStore()
    event_store = InMemoryEventStore()
    ids = DeterministicIdFactory("provider_replay")
    run_id = ids.new_run_id()
    recorder = EventRecorder(
        run_id=run_id,
        event_store=event_store,
        id_factory=ids,
        clock=FakeClock(),
    )
    journal = ArtifactJournal(
        artifact_store=artifact_store,
        event_recorder=recorder,
        minimizer=TopLevelAllowlistMinimizer(
            {
                ArtifactRole.LLM_REQUEST: {
                    "messages",
                    "model_parameters",
                    "response_schema",
                    "tools",
                },
                ArtifactRole.LLM_RESPONSE: {"output", "usage"},
                ArtifactRole.DIAGNOSTICS: {"message"},
            }
        ),
        sanitizer=StrictJsonSanitizer(exact_secret_values=(RAW_SECRET,)),
        max_size_bytes=32_000,
    )
    return artifact_store, event_store, ids, run_id, recorder, journal


def _request(recorder, journal, ids):
    request = journal.register_json(
        {
            "messages": [
                {"role": "user", "content": f"safe prefix {RAW_SECRET}"},
            ],
            "model_parameters": {"temperature": 0},
            "response_schema": {"type": "object"},
            "tools": [],
            "unneeded_headers": {"Authorization": RAW_SECRET},
        },
        role=ArtifactRole.LLM_REQUEST,
    )
    call_id = ids.new_llm_call_id()
    recorder.record(LLMCallRequested(call_id, "generate", "provider/model"))
    recorder.record(LLMRequestArtifactLinked(call_id, request.artifact_ref.artifact_id))
    return call_id, request


def _completed_trace():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    call_id, request = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(call_id))

    first_attempt = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, first_attempt, 1))
    recorder.record(
        ProviderAttemptFailed(
            call_id=call_id,
            provider_attempt_id=first_attempt,
            category=FailureCategory.INFRASTRUCTURE,
            code=FailureCode.TRANSIENT_EXTERNAL_SERVICE_FAILURE,
            retryable=True,
            message="transient provider failure",
            latency_ns=11,
        )
    )
    recorder.record(LLMCallRetried(call_id, 2))

    second_attempt = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, second_attempt, 2))
    successful_attempt = recorder.record(
        ProviderAttemptCompleted(
            call_id=call_id,
            provider_attempt_id=second_attempt,
            resolved_provider="openrouter",
            resolved_model="provider/model",
            input_tokens=17,
            output_tokens=5,
            reasoning_tokens=2,
            cost_usd=Decimal("0.0012"),
            latency_ns=22,
        )
    )
    response = journal.register_json(
        {"output": {"candidate": 7}, "usage": {"tokens": 5}},
        role=ArtifactRole.LLM_RESPONSE,
        causation_event_id=successful_attempt.event_id,
    )
    recorder.record(LLMCallCompleted(call_id, response.artifact_ref.artifact_id))
    return (
        artifact_store,
        event_store,
        run_id,
        call_id,
        request,
        response,
    )


def test_completed_recorded_call_replays_exact_response_without_provider_access():
    artifact_store, event_store, run_id, call_id, request, response = _completed_trace()

    class ReadOnlySpy:
        def __init__(self, delegate):
            self.delegate = delegate
            self.reads = 0

        def put_bytes(self, content, *, media_type):
            raise AssertionError("offline replay attempted to persist or contact a provider")

        def stat(self, artifact_id):
            return self.delegate.stat(artifact_id)

        def read_bytes(self, artifact_id, *, expected_media_type=None):
            self.reads += 1
            return self.delegate.read_bytes(
                artifact_id,
                expected_media_type=expected_media_type,
            )

    store = ReadOnlySpy(artifact_store)
    replay = build_recorded_provider_replay(
        event_store.read(run_id),
        artifact_store=store,
    )
    call = replay.call(call_id)

    assert replay.run_id == run_id
    assert len(replay.calls) == 1
    assert call.status is RecordedCallStatus.COMPLETED
    assert call.request_ref == request.artifact_ref
    assert RAW_SECRET.encode() not in call.request_bytes
    assert call.request_bytes == canonical_json_bytes(
        {
            "messages": [
                {"content": "safe prefix [REDACTED]", "role": "user"},
            ],
            "model_parameters": {"temperature": 0},
            "response_schema": {"type": "object"},
            "tools": [],
        }
    )
    assert [attempt.status for attempt in call.attempts] == [
        RecordedAttemptStatus.FAILED,
        RecordedAttemptStatus.COMPLETED,
    ]
    assert call.attempts[1].cost_usd == Decimal("0.0012")
    assert replay.replay_response(call_id, request_bytes=call.request_bytes) == (
        canonical_json_bytes({"output": {"candidate": 7}, "usage": {"tokens": 5}})
    )
    assert call.response_ref == response.artifact_ref
    assert store.reads > 0


def test_exact_request_match_is_mandatory_and_errors_do_not_echo_bytes():
    artifact_store, event_store, run_id, call_id, request, response = _completed_trace()
    replay = build_recorded_provider_replay(
        event_store.read(run_id),
        artifact_store=artifact_store,
    )
    supplied = b'{"secret":"must-not-echo"}'
    with pytest.raises(RecordedRequestMismatchError) as caught:
        replay.replay_response(call_id, request_bytes=supplied)
    assert "must-not-echo" not in str(caught.value)


def test_failed_and_incomplete_calls_are_reconstructed_but_not_replayable():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    failed_id, _ = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(failed_id))
    recorder.record(
        LLMCallFailed(
            call_id=failed_id,
            category=FailureCategory.SYSTEM,
            code=FailureCode.PARSER_FAILURE,
            retryable=False,
            message="structured output parse failed",
        )
    )
    incomplete_id, _ = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(incomplete_id))

    replay = build_recorded_provider_replay(
        event_store.read(run_id),
        artifact_store=artifact_store,
    )
    assert replay.call(failed_id).status is RecordedCallStatus.FAILED
    assert replay.call(incomplete_id).status is RecordedCallStatus.INCOMPLETE
    for call_id in (failed_id, incomplete_id):
        call = replay.call(call_id)
        with pytest.raises(RecordedCallUnavailableError):
            replay.replay_response(call_id, request_bytes=call.request_bytes)


def test_missing_or_late_request_link_fails_closed():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    request = journal.register_json(
        {"messages": [], "model_parameters": {}, "response_schema": {}, "tools": []},
        role=ArtifactRole.LLM_REQUEST,
    )
    call_id = ids.new_llm_call_id()
    recorder.record(LLMCallRequested(call_id, "generate", "provider/model"))
    recorder.record(LLMCallStarted(call_id))
    recorder.record(LLMRequestArtifactLinked(call_id, request.artifact_ref.artifact_id))

    with pytest.raises(RecordedProviderReplayError):
        build_recorded_provider_replay(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_completed_call_without_registered_response_fails_closed():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    call_id, _ = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(call_id))
    attempt_id = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, attempt_id, 1))
    recorder.record(
        ProviderAttemptCompleted(
            call_id=call_id,
            provider_attempt_id=attempt_id,
            resolved_provider="provider",
            resolved_model="provider/model",
        )
    )
    recorder.record(LLMCallCompleted(call_id))

    with pytest.raises(RecordedProviderReplayError):
        build_recorded_provider_replay(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_retry_requires_a_retryable_failed_attempt_and_exact_attempt_number():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    call_id, _ = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(call_id))
    attempt_id = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, attempt_id, 1))
    recorder.record(
        ProviderAttemptCompleted(
            call_id=call_id,
            provider_attempt_id=attempt_id,
            resolved_provider="provider",
            resolved_model="provider/model",
        )
    )
    recorder.record(LLMCallRetried(call_id, 2))

    with pytest.raises(RecordedProviderReplayError):
        build_recorded_provider_replay(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_request_registration_must_precede_logical_request_metadata():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    request_value = {
        "messages": [],
        "model_parameters": {},
        "response_schema": {},
        "tools": [],
    }
    response = journal.register_json(
        {"output": {"preseeded": True}, "usage": {}},
        role=ArtifactRole.LLM_RESPONSE,
    )
    call_id = ids.new_llm_call_id()
    recorder.record(LLMCallRequested(call_id, "generate", "provider/model"))
    # This request registration is too late: replay requires the durable request
    # to exist before the logical request metadata event.
    request = journal.register_json(request_value, role=ArtifactRole.LLM_REQUEST)
    recorder.record(LLMRequestArtifactLinked(call_id, request.artifact_ref.artifact_id))
    recorder.record(LLMCallStarted(call_id))
    attempt_id = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, attempt_id, 1))
    recorder.record(
        ProviderAttemptCompleted(
            call_id=call_id,
            provider_attempt_id=attempt_id,
            resolved_provider="provider",
            resolved_model="provider/model",
        )
    )
    recorder.record(LLMCallCompleted(call_id, response.artifact_ref.artifact_id))

    with pytest.raises(RecordedProviderReplayError):
        build_recorded_provider_replay(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_response_registration_must_follow_successful_provider_attempt():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    response = journal.register_json(
        {"output": {"preseeded": True}, "usage": {}},
        role=ArtifactRole.LLM_RESPONSE,
    )
    call_id, _ = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(call_id))
    attempt_id = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, attempt_id, 1))
    recorder.record(
        ProviderAttemptCompleted(
            call_id=call_id,
            provider_attempt_id=attempt_id,
            resolved_provider="provider",
            resolved_model="provider/model",
        )
    )
    recorder.record(LLMCallCompleted(call_id, response.artifact_ref.artifact_id))

    with pytest.raises(RecordedProviderReplayError):
        build_recorded_provider_replay(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_uncausal_duplicate_registration_cannot_launder_a_preseeded_response():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    response_value = {"output": {"same": True}, "usage": {}}
    preseeded = journal.register_json(
        response_value,
        role=ArtifactRole.LLM_RESPONSE,
    )
    call_id, _ = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(call_id))
    attempt_id = ids.new_provider_attempt_id()
    recorder.record(ProviderAttemptStarted(call_id, attempt_id, 1))
    recorder.record(
        ProviderAttemptCompleted(
            call_id=call_id,
            provider_attempt_id=attempt_id,
            resolved_provider="provider",
            resolved_model="provider/model",
        )
    )
    duplicate = journal.register_json(
        response_value,
        role=ArtifactRole.LLM_RESPONSE,
    )
    assert duplicate.artifact_ref == preseeded.artifact_ref
    recorder.record(LLMCallCompleted(call_id, duplicate.artifact_ref.artifact_id))

    with pytest.raises(RecordedProviderReplayError):
        build_recorded_provider_replay(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_causal_registration_allows_identical_responses_from_distinct_calls():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    response_value = {"output": {"deterministic": 1}, "usage": {}}
    calls = []
    responses = []

    for _ in range(2):
        call_id, request = _request(recorder, journal, ids)
        recorder.record(LLMCallStarted(call_id))
        attempt_id = ids.new_provider_attempt_id()
        recorder.record(ProviderAttemptStarted(call_id, attempt_id, 1))
        successful_attempt = recorder.record(
            ProviderAttemptCompleted(
                call_id=call_id,
                provider_attempt_id=attempt_id,
                resolved_provider="provider",
                resolved_model="provider/model",
            )
        )
        response = journal.register_json(
            response_value,
            role=ArtifactRole.LLM_RESPONSE,
            causation_event_id=successful_attempt.event_id,
        )
        recorder.record(LLMCallCompleted(call_id, response.artifact_ref.artifact_id))
        calls.append((call_id, request))
        responses.append(response)

    assert responses[0].artifact_ref == responses[1].artifact_ref
    replay = build_recorded_provider_replay(
        event_store.read(run_id),
        artifact_store=artifact_store,
    )
    expected_response = canonical_json_bytes(response_value)
    for call_id, request in calls:
        recorded = replay.call(call_id)
        assert recorded.request_ref == request.artifact_ref
        assert replay.replay_response(
            call_id,
            request_bytes=recorded.request_bytes,
        ) == expected_response


def test_replay_reread_failure_does_not_retain_store_secret_or_exception_context():
    artifact_store, event_store, ids, run_id, recorder, journal = _setup()
    call_id, request = _request(recorder, journal, ids)
    recorder.record(LLMCallStarted(call_id))

    class ChangingReadStore:
        def __init__(self, delegate):
            self.delegate = delegate
            self.request_reads = 0

        def put_bytes(self, content, *, media_type):
            raise AssertionError("offline replay attempted a write")

        def stat(self, artifact_id):
            return self.delegate.stat(artifact_id)

        def read_bytes(self, artifact_id, *, expected_media_type=None):
            if artifact_id == request.artifact_ref.artifact_id:
                self.request_reads += 1
                if self.request_reads == 2:
                    raise ValueError(RAW_SECRET)
            return self.delegate.read_bytes(
                artifact_id,
                expected_media_type=expected_media_type,
            )

    store = ChangingReadStore(artifact_store)
    with pytest.raises(RecordedProviderReplayError) as caught:
        build_recorded_provider_replay(
            event_store.read(run_id),
            artifact_store=store,
        )

    assert store.request_reads == 2
    assert RAW_SECRET not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_recorded_replay_snapshots_and_index_are_immutable():
    artifact_store, event_store, run_id, call_id, request, response = _completed_trace()
    replay = build_recorded_provider_replay(
        event_store.read(run_id),
        artifact_store=artifact_store,
    )
    call = replay.call(call_id)

    with pytest.raises(FrozenInstanceError):
        replay.calls = ()
    with pytest.raises(FrozenInstanceError):
        call.request_bytes = b"changed"
    with pytest.raises(FrozenInstanceError):
        call.attempts[0].latency_ns = 999
    with pytest.raises(TypeError):
        replay._by_call_id[call_id] = call


def test_recorded_replay_repr_excludes_request_and_response_content():
    artifact_store, event_store, run_id, call_id, request, response = _completed_trace()
    replay = build_recorded_provider_replay(
        event_store.read(run_id),
        artifact_store=artifact_store,
    )
    call = replay.call(call_id)

    for rendered in (repr(call), repr(replay)):
        assert "safe prefix" not in rendered
        assert "candidate" not in rendered


def test_recorded_snapshot_constructors_are_not_a_public_provenance_bypass():
    request_bytes = canonical_json_bytes({"request": "synthetic"})
    artifact_store = InMemoryArtifactStore()
    request_ref = artifact_store.put_bytes(
        request_bytes,
        media_type="application/json",
    )

    with pytest.raises(TypeError, match="verified replay"):
        RecordedProviderAttempt(
            provider_attempt_id=ProviderAttemptId("provider_attempt_case_z"),
            attempt_number=1,
            status=RecordedAttemptStatus.STARTED,
        )
    with pytest.raises(TypeError, match="verified replay"):
        RecordedLLMCall(
            call_id=LLMCallId("call_case_z"),
            operation="generate",
            requested_model="provider/model",
            request_ref=request_ref,
            request_bytes=request_bytes,
            status=RecordedCallStatus.COMPLETED,
            attempts=(),
            response_bytes=b'{"forged":true}',
        )
    with pytest.raises(TypeError, match="verified replay"):
        RecordedProviderReplay(run_id=None, calls=())


def test_low_level_mutation_of_public_snapshots_cannot_change_replay_output():
    artifact_store, event_store, run_id, call_id, request, response = _completed_trace()
    replay = build_recorded_provider_replay(
        event_store.read(run_id),
        artifact_store=artifact_store,
    )
    expected = canonical_json_bytes(
        {"output": {"candidate": 7}, "usage": {"tokens": 5}}
    )

    public_call = replay.calls[0]
    object.__setattr__(public_call, "response_bytes", b'{"forged":true}')
    returned_call = replay.call(call_id)
    object.__setattr__(returned_call, "response_bytes", b'{"forged":true}')
    object.__setattr__(replay, "calls", ())

    assert replay.replay_response(
        call_id,
        request_bytes=returned_call.request_bytes,
    ) == expected
    assert replay.call(call_id).response_bytes == expected

    internal_call = replay._by_call_id[call_id]
    object.__setattr__(internal_call, "response_bytes", b'{"forged":true}')
    with pytest.raises(RecordedProviderReplayError, match="integrity check"):
        replay.replay_response(
            call_id,
            request_bytes=returned_call.request_bytes,
        )

"""Provider-free contracts for sealed accepted-output continuation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, field_validator

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.domain.llm_task_queue import (
    AttemptRequestEvidence,
    AttemptRequestVariant,
    AttemptStatus,
    AttemptTelemetry,
    LLMTaskOutcome,
    RetryClassification,
    RetryDisposition,
    RetryReason,
    SanitizedAttemptFailure,
    TaskOutcomeStatus,
    TaskTelemetry,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    structured_generation_outcome_record,
    structured_generation_output_evidence_record,
    structured_generation_request_evidence_record,
)
from agent_evolve.integrations.pydantic_ai.sealed_output_replay import (
    SealedAcceptedOutputReplayError,
    SealedReplayJsonlFile,
    SealedReplayReceiptPublicationError,
    SealedReplayRequestDriftError,
    SealedReplayThenLiveStructuredRunner,
    SealedReplayTypedOutputError,
    load_sealed_accepted_output_replay_jsonl,
    validate_sealed_accepted_output_replay_source_receipt,
    validate_sealed_replay_decision_receipt,
)
from agent_evolve.ports.artifact_store import canonical_json_bytes
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


MODEL = "provider/model-v1"


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    answer: int


class _SameSchemaStricterOutput(BaseModel):
    """Trusted local semantics can tighten without changing JSON Schema."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    answer: int

    @field_validator("answer")
    @classmethod
    def reject_recorded_value(cls, value: int) -> int:
        if value == 7:
            raise ValueError("the current trusted contract rejects seven")
        return value

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        return _Output.model_json_schema(*args, **kwargs)


# Reproduce the same public type identity and JSON schema while retaining a
# different trusted core validator. Replay must still validate the typed JSON.
_SameSchemaStricterOutput.__module__ = _Output.__module__
_SameSchemaStricterOutput.__qualname__ = _Output.__qualname__


def _request(call_id: str, *, prompt: str = "solve exactly") -> StructuredGenerationRequest[_Output]:
    return StructuredGenerationRequest(
        call_id=LLMCallId(call_id),
        operation="select_action",
        prompt=prompt,
        output_type=_Output,
        output_tool_name="return_answer",
        max_output_tokens=8_192,
        temperature=0.2,
    )


def _response(answer: int, *, response_id: str) -> StructuredGenerationResponse[_Output]:
    return StructuredGenerationResponse(
        value=_Output(answer=answer),
        requested_model=MODEL,
        resolved_model=MODEL,
        resolved_provider="provider",
        provider_response_id=response_id,
        finish_reason="tool_call",
        input_tokens=20,
        output_tokens=3,
        reasoning_tokens=7,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.001"),
        latency_ns=123,
    )


def _successful_outcome(
    request: StructuredGenerationRequest[_Output],
    response: StructuredGenerationResponse[_Output],
) -> LLMTaskOutcome[StructuredGenerationResponse[_Output]]:
    attempt = AttemptTelemetry(
        attempt_number=1,
        status=AttemptStatus.SUCCEEDED,
        wait_time_ns=4,
        service_time_ns=123,
        will_retry=False,
        request_evidence=AttemptRequestEvidence(
            variant=AttemptRequestVariant.ORIGINAL,
            prompt_sha256=hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            provider_attempt_id=ProviderAttemptId(
                f"provider_attempt_{hashlib.sha256(request.call_id.value.encode()).hexdigest()}"
            ),
        ),
    )
    return LLMTaskOutcome(
        status=TaskOutcomeStatus.SUCCEEDED,
        telemetry=TaskTelemetry(
            task_id=request.call_id.value,
            queue_time_ns=4,
            service_time_ns=123,
            total_time_ns=127,
            attempts=(attempt,),
        ),
        response=response,
    )


def _failed_outcome(
    request: StructuredGenerationRequest[_Output],
) -> LLMTaskOutcome[StructuredGenerationResponse[_Output]]:
    failure = SanitizedAttemptFailure(
        kind="invalid_request",
        retryable=False,
        safe_message="request was rejected",
    )
    attempt = AttemptTelemetry(
        attempt_number=1,
        status=AttemptStatus.TERMINAL_FAILURE,
        wait_time_ns=2,
        service_time_ns=5,
        will_retry=False,
        classification=RetryClassification(
            disposition=RetryDisposition.FAIL,
            reason=RetryReason.PERMANENT,
            sanitized_failure=failure,
        ),
        error_type="StructuredGenerationError",
        request_evidence=AttemptRequestEvidence(
            variant=AttemptRequestVariant.ORIGINAL,
            prompt_sha256=hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            provider_attempt_id=ProviderAttemptId(
                f"provider_attempt_{hashlib.sha256(request.call_id.value.encode()).hexdigest()}"
            ),
        ),
    )
    return LLMTaskOutcome(
        status=TaskOutcomeStatus.TERMINAL_FAILURE,
        telemetry=TaskTelemetry(
            task_id=request.call_id.value,
            queue_time_ns=2,
            service_time_ns=5,
            total_time_ns=7,
            attempts=(attempt,),
        ),
    )


def _envelope(record: dict[str, object], sequence: int) -> dict[str, object]:
    return {
        "authenticated_record": record,
        "observation": {
            "monotonic_ns_since_execution_start": sequence,
            "observed_at_utc": "2026-07-17T00:00:00+00:00",
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> SealedReplayJsonlFile:
    payload = b"".join(
        canonical_json_bytes(_envelope(record, index)) + b"\n"
        for index, record in enumerate(records, start=1)
    )
    path.write_bytes(payload)
    return SealedReplayJsonlFile(path, hashlib.sha256(payload).hexdigest())


def _source_files(
    tmp_path: Path,
    *,
    include_failed_request: bool = False,
) -> tuple[SealedReplayJsonlFile, SealedReplayJsonlFile, SealedReplayJsonlFile]:
    accepted_request = _request("call_source_accepted")
    accepted_response = _response(7, response_id="response-source-1")
    accepted_outcome = _successful_outcome(accepted_request, accepted_response)
    request_records = [
        structured_generation_request_evidence_record(accepted_request)
    ]
    output_records = [
        structured_generation_output_evidence_record(
            accepted_request,
            accepted_outcome,
        )
    ]
    outcome_records = [structured_generation_outcome_record(accepted_outcome)]
    if include_failed_request:
        failed_request = _request(
            "call_source_failed",
            prompt="this provider call failed",
        )
        request_records.append(
            structured_generation_request_evidence_record(failed_request)
        )
        outcome_records.append(
            structured_generation_outcome_record(_failed_outcome(failed_request))
        )
    return (
        _write_jsonl(tmp_path / "requests.jsonl", request_records),
        _write_jsonl(tmp_path / "outputs.jsonl", output_records),
        _write_jsonl(tmp_path / "outcomes.jsonl", outcome_records),
    )


def _load(tmp_path: Path, *, include_failed_request: bool = False):
    requests, outputs, outcomes = _source_files(
        tmp_path,
        include_failed_request=include_failed_request,
    )
    return load_sealed_accepted_output_replay_jsonl(
        source_id="source_run_v1",
        request_evidence=requests,
        output_evidence=outputs,
        terminal_outcomes=outcomes,
    )


class _LiveRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    async def __call__(self, request: StructuredGenerationRequest[_Output]):
        self.calls.append(request.call_id.value)
        return _response(99, response_id="response-live")

    async def aclose(self) -> None:
        self.closed = True


def test_exact_new_call_id_replays_then_distinct_request_continues_live(
    tmp_path: Path,
) -> None:
    source = _load(tmp_path, include_failed_request=True)
    live = _LiveRunner()
    receipts: list[dict[str, object]] = []
    runner = SealedReplayThenLiveStructuredRunner(
        source=source,
        requested_model=MODEL,
        live_runner=live,
        decision_receipt_sink=receipts.append,
    )

    async def exercise() -> None:
        replayed = await runner(_request("call_continuation_accepted"))
        assert replayed.response.value == _Output(answer=7)
        assert replayed.response.provider_response_id == "response-source-1"
        assert live.calls == []

        # The failed source request had no accepted output and therefore did
        # not become an entry. Once the complete accepted prefix is consumed,
        # the matching logical request is a normal live continuation.
        live_result = await runner(
            _request("call_continuation_failed", prompt="this provider call failed")
        )
        assert live_result.value == _Output(answer=99)
        assert live.calls == ["call_continuation_failed"]
        await runner.aclose()

    asyncio.run(exercise())

    assert [receipt["decision"] for receipt in receipts] == [
        "replayed",
        "live_after_prefix",
    ]
    assert receipts[0]["source_call_id"] == "call_source_accepted"
    assert receipts[0]["current_call_id"] == "call_continuation_accepted"
    assert receipts[0]["typed_output_persisted_in_receipt"] is False
    assert all(
        receipt["provider_dispatched_by_replay_boundary"] is False
        for receipt in receipts
    )
    assert runner.remaining_entry_count == 0
    assert live.closed is True


def test_request_or_model_drift_cannot_fall_through_while_prefix_remains(
    tmp_path: Path,
) -> None:
    source = _load(tmp_path)
    live = _LiveRunner()
    receipts: list[dict[str, object]] = []
    runner = SealedReplayThenLiveStructuredRunner(
        source=source,
        requested_model=MODEL,
        live_runner=live,
        decision_receipt_sink=receipts.append,
    )

    async def exercise() -> None:
        with pytest.raises(SealedReplayRequestDriftError):
            await runner(_request("call_drift", prompt="prompt changed by one byte"))

    asyncio.run(exercise())
    assert live.calls == []
    assert runner.remaining_entry_count == 1
    assert receipts[0]["decision"] == "request_drift"

    with pytest.raises(ValueError, match="foreign requested model"):
        SealedReplayThenLiveStructuredRunner(
            source=source,
            requested_model="provider/other-model",
            live_runner=live,
            decision_receipt_sink=receipts.append,
        )


def test_required_receipt_failure_releases_neither_replay_nor_live(
    tmp_path: Path,
) -> None:
    source = _load(tmp_path)
    live = _LiveRunner()

    def fail_receipt(_: dict[str, object]) -> None:
        raise OSError("recorder unavailable")

    runner = SealedReplayThenLiveStructuredRunner(
        source=source,
        requested_model=MODEL,
        live_runner=live,
        decision_receipt_sink=fail_receipt,
    )

    async def exercise() -> None:
        with pytest.raises(SealedReplayReceiptPublicationError) as caught:
            await runner(_request("call_receipt_failure"))
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True

    asyncio.run(exercise())
    assert runner.remaining_entry_count == 1
    assert live.calls == []


def test_replayed_json_must_pass_current_trusted_core_validator(tmp_path: Path) -> None:
    source = _load(tmp_path)
    live = _LiveRunner()
    receipts: list[dict[str, object]] = []
    runner = SealedReplayThenLiveStructuredRunner(
        source=source,
        requested_model=MODEL,
        live_runner=live,
        decision_receipt_sink=receipts.append,
    )
    request = replace(
        _request("call_stricter_contract"),
        output_type=_SameSchemaStricterOutput,
    )

    async def exercise() -> None:
        with pytest.raises(SealedReplayTypedOutputError) as caught:
            await runner(request)
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True

    asyncio.run(exercise())
    assert runner.remaining_entry_count == 1
    assert receipts == []
    assert live.calls == []


def test_file_and_internal_record_tampering_fail_closed(tmp_path: Path) -> None:
    requests, outputs, outcomes = _source_files(tmp_path)
    outputs.path.write_bytes(outputs.path.read_bytes() + b"\n")
    with pytest.raises(SealedAcceptedOutputReplayError, match="digest drifted"):
        load_sealed_accepted_output_replay_jsonl(
            source_id="tampered_file",
            request_evidence=requests,
            output_evidence=outputs,
            terminal_outcomes=outcomes,
        )

    # Restore a canonical row, change typed output, and seal the changed file;
    # its inner evidence hash must still reject the mutation.
    row = json.loads(outputs.path.read_text().splitlines()[0])
    row["authenticated_record"]["typed_output"]["answer"] = 9
    payload = canonical_json_bytes(row) + b"\n"
    outputs.path.write_bytes(payload)
    resealed = SealedReplayJsonlFile(
        outputs.path,
        hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(
        SealedAcceptedOutputReplayError,
        match="strict record validation",
    ):
        load_sealed_accepted_output_replay_jsonl(
            source_id="tampered_record",
            request_evidence=requests,
            output_evidence=resealed,
            terminal_outcomes=outcomes,
        )


def test_output_evidence_joined_to_failed_outcome_is_never_replayable(
    tmp_path: Path,
) -> None:
    failed_request = _request("call_failed_with_output", prompt="failed call")
    fabricated_success = _successful_outcome(
        failed_request,
        _response(5, response_id="response-never-accepted"),
    )
    request_record = structured_generation_request_evidence_record(failed_request)
    output_record = structured_generation_output_evidence_record(
        failed_request,
        fabricated_success,
    )
    terminal_record = structured_generation_outcome_record(
        _failed_outcome(failed_request)
    )
    requests = _write_jsonl(tmp_path / "requests.jsonl", [request_record])
    outputs = _write_jsonl(tmp_path / "outputs.jsonl", [output_record])
    outcomes = _write_jsonl(tmp_path / "outcomes.jsonl", [terminal_record])

    with pytest.raises(
        SealedAcceptedOutputReplayError,
        match="successful provider outcome",
    ):
        load_sealed_accepted_output_replay_jsonl(
            source_id="failed_output",
            request_evidence=requests,
            output_evidence=outputs,
            terminal_outcomes=outcomes,
        )


def test_source_receipt_is_content_free_and_self_authenticating(tmp_path: Path) -> None:
    source = _load(tmp_path)
    receipt = source.source_receipt()
    assert source.accepted_output_count == 1
    assert source.requested_models == (MODEL,)
    assert receipt["accepted_output_count"] == 1
    assert receipt["source_identity_sha256"] == source.source_identity_sha256
    assert receipt["raw_prompt_or_output_persisted_in_receipt"] is False
    encoded = canonical_json_bytes(receipt)
    assert b"solve exactly" not in encoded
    assert b'"answer":7' not in encoded
    assert validate_sealed_accepted_output_replay_source_receipt(receipt) == receipt

    tampered = dict(receipt)
    tampered["accepted_output_count"] = 2
    with pytest.raises(ValueError):
        validate_sealed_accepted_output_replay_source_receipt(tampered)


def test_replay_decision_receipt_validator_rejects_tampering(tmp_path: Path) -> None:
    source = _load(tmp_path)
    receipts: list[dict[str, object]] = []
    runner = SealedReplayThenLiveStructuredRunner(
        source=source,
        requested_model=MODEL,
        live_runner=_LiveRunner(),
        decision_receipt_sink=receipts.append,
    )
    asyncio.run(runner(_request("call_receipt_validation")))
    receipt = receipts[0]
    assert validate_sealed_replay_decision_receipt(receipt) == receipt
    assert receipt["response_telemetry_origin"] == "historical_source_attempt"
    assert receipt["historical_attempt_count"] == 1

    tampered = dict(receipt)
    tampered["remaining_entry_count"] = 99
    with pytest.raises(ValueError, match="hash is invalid"):
        validate_sealed_replay_decision_receipt(tampered)

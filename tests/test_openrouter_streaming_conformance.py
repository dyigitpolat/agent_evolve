"""Provider-free tests for the one-call streaming-conformance harness."""

from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Decimal
import hashlib
from pathlib import Path

import pytest

from agent_evolve.domain.ids import ProviderAttemptId
from agent_evolve.domain.llm_task_queue import (
    AttemptRequestEvidence,
    AttemptRequestVariant,
    AttemptStatus,
    AttemptTelemetry,
    LLMTaskOutcome,
    TaskOutcomeStatus,
    TaskTelemetry,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.ports.artifact_store import decode_json_bytes
from agent_evolve.ports.structured_generator import (
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development import run_openrouter_streaming_conformance as conformance
from examples.development.durable_run_artifacts import (
    finalize_run_directory,
    read_jsonl,
    write_bytes_atomic,
    write_json_atomic,
)


def _config() -> conformance.ProgressAwareOpenRouterConfig:
    return conformance.build_config(
        first_event_timeout_seconds=180,
        idle_timeout_seconds=120,
        absolute_timeout_seconds=0,
    )


def _release_gate(tmp_path: Path) -> Path:
    root = tmp_path / "provider_free_release_gate"
    root.mkdir()
    report = root / "focused_tests.junit.xml"
    write_bytes_atomic(
        report,
        (
            b'<?xml version="1.0" encoding="utf-8"?>'
            b'<testsuites name="pytest tests">'
            b'<testsuite name="pytest" errors="0" failures="0" '
            b'skipped="0" tests="1">'
            b'<testcase classname="offline" name="test_provider_free" />'
            b"</testsuite></testsuites>"
        ),
    )
    source_before = conformance._source_identity()
    tests_before = conformance._focused_test_source_identity()
    gate = conformance.build_provider_free_release_gate(
        config=_config(),
        junit_report_path=report,
        pytest_exit_code=0,
        source_identity_before=source_before,
        focused_test_source_identity_before=tests_before,
    )
    write_json_atomic(root / "release_gate.json", gate)
    finalize_run_directory(root, status="provider_free_release_gate_passed")
    return root


def _response(**changes) -> StructuredGenerationResponse:
    values = {
        "value": conformance.StreamingConformanceOutput(
            nonce=conformance.CONFORMANCE_NONCE,
            acknowledgement=conformance.ACKNOWLEDGEMENT,
        ),
        "requested_model": conformance.MODEL,
        "resolved_model": conformance.CANONICAL_MODEL,
        "resolved_provider": conformance.RESOLVED_PROVIDER,
        "provider_response_id": "response_offline_conformance_000001",
        "finish_reason": "stop",
        "input_tokens": 80,
        "output_tokens": 24,
        "reasoning_tokens": 10,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": Decimal("0.001"),
        "latency_ns": 3_000_000_000,
    }
    values.update(changes)
    return StructuredGenerationResponse(**values)


def _outcome(response: StructuredGenerationResponse, *, call_id: str) -> LLMTaskOutcome:
    attempt_id = ProviderAttemptId(
        "provider_attempt_offline_conformance_000001"
    )
    evidence = AttemptRequestEvidence(
        variant=AttemptRequestVariant.ORIGINAL,
        prompt_sha256=hashlib.sha256(conformance.PROMPT.encode()).hexdigest(),
        provider_attempt_id=attempt_id,
    )
    attempt = AttemptTelemetry(
        attempt_number=1,
        status=AttemptStatus.SUCCEEDED,
        wait_time_ns=1,
        service_time_ns=2,
        will_retry=False,
        request_evidence=evidence,
    )
    return LLMTaskOutcome(
        status=TaskOutcomeStatus.SUCCEEDED,
        telemetry=TaskTelemetry(
            task_id=call_id,
            queue_time_ns=1,
            service_time_ns=2,
            total_time_ns=3,
            attempts=(attempt,),
        ),
        response=response,
    )


def _completed_progress_rows(call_id: str) -> list[dict[str, object]]:
    attempt_id = "provider_attempt_offline_conformance_000001"
    return [
        {
            "schema_version": 2,
            "call_id": call_id,
            "provider_attempt_id": attempt_id,
            "sequence": 1,
            "kind": "output_selected",
            "channel": "other",
            "elapsed_ns": 10,
            "event_content_utf8_bytes": 0,
            "cumulative_content_utf8_bytes": 0,
            "rolling_content_sha256": "c" * 64,
        },
        {
            "schema_version": 2,
            "call_id": call_id,
            "provider_attempt_id": attempt_id,
            "sequence": 2,
            "kind": "part_delta",
            "channel": "tool_call",
            "elapsed_ns": 20,
            "event_content_utf8_bytes": 40,
            "cumulative_content_utf8_bytes": 40,
            "rolling_content_sha256": "d" * 64,
        },
        {
            "schema_version": 2,
            "call_id": call_id,
            "provider_attempt_id": attempt_id,
            "sequence": 3,
            "kind": "part_ended",
            "channel": "tool_call",
            "elapsed_ns": 30,
            "event_content_utf8_bytes": 0,
            "cumulative_content_utf8_bytes": 40,
            "rolling_content_sha256": "d" * 64,
        },
        {
            "schema_version": 2,
            "call_id": call_id,
            "provider_attempt_id": attempt_id,
            "sequence": 4,
            "kind": "stream_completed",
            "channel": "other",
            "elapsed_ns": 40,
            "event_content_utf8_bytes": 0,
            "cumulative_content_utf8_bytes": 40,
            "rolling_content_sha256": "d" * 64,
        },
    ]
class _OfflineRunner:
    def __init__(self, *, progress_sink, outcome_sink) -> None:
        self.progress_sink = progress_sink
        self.outcome_sink = outcome_sink
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def __call__(self, request):
        self.request = request
        attempt_id = "provider_attempt_offline_conformance_000001"
        self.progress_sink(
            StructuredStreamProgress(
                call_id=request.call_id.value,
                sequence=1,
                kind=StructuredStreamProgressKind.PART_STARTED,
                channel=StructuredStreamChannel.TOOL_CALL,
                elapsed_ns=10,
                event_content_utf8_bytes=40,
                cumulative_content_utf8_bytes=40,
                rolling_content_sha256="a" * 64,
                provider_attempt_id=attempt_id,
            )
        )
        self.progress_sink(
            StructuredStreamProgress(
                call_id=request.call_id.value,
                sequence=2,
                kind=StructuredStreamProgressKind.OUTPUT_SELECTED,
                channel=StructuredStreamChannel.OTHER,
                elapsed_ns=20,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=40,
                rolling_content_sha256="a" * 64,
                provider_attempt_id=attempt_id,
            )
        )
        self.progress_sink(
            StructuredStreamProgress(
                call_id=request.call_id.value,
                sequence=3,
                kind=StructuredStreamProgressKind.PART_DELTA,
                channel=StructuredStreamChannel.TOOL_CALL,
                elapsed_ns=30,
                event_content_utf8_bytes=12,
                cumulative_content_utf8_bytes=52,
                rolling_content_sha256="b" * 64,
                provider_attempt_id=attempt_id,
            )
        )
        self.progress_sink(
            StructuredStreamProgress(
                call_id=request.call_id.value,
                sequence=4,
                kind=StructuredStreamProgressKind.PART_ENDED,
                channel=StructuredStreamChannel.TOOL_CALL,
                elapsed_ns=40,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=52,
                rolling_content_sha256="b" * 64,
                provider_attempt_id=attempt_id,
            )
        )
        self.progress_sink(
            StructuredStreamProgress(
                call_id=request.call_id.value,
                sequence=5,
                kind=StructuredStreamProgressKind.STREAM_COMPLETED,
                channel=StructuredStreamChannel.OTHER,
                elapsed_ns=50,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=52,
                rolling_content_sha256="b" * 64,
                provider_attempt_id=attempt_id,
            )
        )
        response = _response()
        self.outcome_sink(_outcome(response, call_id=request.call_id.value))
        return AttemptedStructuredGenerationResponse(
            response=response,
            attempt_count=1,
        )


def test_readiness_exercises_every_non_network_gate_without_credentials(
    tmp_path: Path,
) -> None:
    credentials = 0
    factories = 0

    def credential_loader() -> str:
        nonlocal credentials
        credentials += 1
        return "must-not-be-read"

    def runner_factory(**_):
        nonlocal factories
        factories += 1
        raise AssertionError("provider composition is unreachable in readiness")

    run_dir = tmp_path / "stream_conformance_readiness"
    release_gate_dir = _release_gate(tmp_path)
    summary = conformance.execute(
        mode="readiness",
        run_dir=run_dir,
        release_gate_dir=release_gate_dir,
        config=_config(),
        dependencies=conformance.ConformanceDependencies(
            credential_loader=credential_loader,
            runner_factory=runner_factory,
        ),
    )

    assert credentials == 0
    assert factories == 0
    assert summary["result"]["status"] == "ready_provider_not_called"
    assert summary["result"]["client_constructed"] is False
    assert summary["result"]["provider_call_attempted"] is False
    assert read_jsonl(run_dir / "stream_progress.jsonl") == ()
    assert read_jsonl(run_dir / "queue_outcomes.jsonl") == ()
    manifest = decode_json_bytes((run_dir / "manifest.json").read_bytes())
    assert manifest["request"]["max_output_tokens"] == 384_000
    assert manifest["route"]["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert manifest["composition"]["queue"]["attempt_timeout_ns"] is None
    assert manifest["composition"]["queue"]["attempt_request_policy"] == (
        "exact_payload"
    )
    assert manifest["progress_journal"]["max_unfsynced_rows"] == 64
    assert manifest["source_identity"]["file_count"] >= 9
    assert manifest["protocol"]["sha256"] == conformance.PREREGISTRATION_SHA256
    assert manifest["provider_free_release_gate"][
        "release_gate_commitment_sha256"
    ]
    finalized = decode_json_bytes((run_dir / "finalized.json").read_bytes())
    assert finalized["status"] == "ready_provider_not_called"
    assert finalized["files"]["stream_progress.jsonl"]["jsonl_rows"] == 0


def test_injected_live_path_produces_bound_progress_outcome_and_finalization(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def runner_factory(**kwargs):
        observed.update(kwargs)
        runner = _OfflineRunner(
            progress_sink=kwargs["progress_sink"],
            outcome_sink=kwargs["outcome_sink"],
        )
        observed["runner"] = runner
        return runner

    run_dir = tmp_path / conformance.FROZEN_LIVE_RUN_ID
    summary = conformance.execute(
        mode="live",
        run_dir=run_dir,
        release_gate_dir=_release_gate(tmp_path),
        config=_config(),
        dependencies=conformance.ConformanceDependencies(
            credential_loader=lambda: "offline-injected-key",
            runner_factory=runner_factory,
        ),
    )

    assert summary["result"]["status"] == "completed_conformance_only"
    assert summary["result"]["client_constructed"] is True
    assert summary["result"]["provider_call_attempted"] is True
    assert summary["result"]["scientific_result_eligible"] is False
    assert observed["config"].to_manifest_record()["queue"][
        "attempt_timeout_ns"
    ] is None
    runner = observed["runner"]
    assert runner.request.max_output_tokens == 384_000
    assert runner.request.provider_attempt_id is None
    progress = read_jsonl(run_dir / "stream_progress.jsonl")
    outcomes = read_jsonl(run_dir / "queue_outcomes.jsonl")
    assert {row["provider_attempt_id"] for row in progress} == {
        "provider_attempt_offline_conformance_000001"
    }
    assert [row["kind"] for row in progress] == [
        "part_started",
        "output_selected",
        "part_delta",
        "part_ended",
        "stream_completed",
    ]
    assert outcomes[0]["schema_version"] == (
        conformance.queued_runner.STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION
    )
    assert outcomes[0]["attempts"][0]["request_evidence"][
        "provider_attempt_id"
    ] == "provider_attempt_offline_conformance_000001"
    finalized = decode_json_bytes((run_dir / "finalized.json").read_bytes())
    assert finalized["status"] == "completed_conformance_only"


def test_completed_call_fails_closed_on_route_telemetry_nonce_or_progress() -> None:
    response = _response()
    attempted = AttemptedStructuredGenerationResponse(
        response=response,
        attempt_count=1,
    )
    outcome_record = conformance.queued_runner.structured_generation_outcome_record(
        _outcome(response, call_id="call_offline_validation_000001")
    )
    progress = _completed_progress_rows("call_offline_validation_000001")
    assert conformance.validate_completed_call(
        attempted,
        expected_call_id="call_offline_validation_000001",
        outcome_rows=[outcome_record],
        progress_rows=progress,
    )["successful_provider_attempt_id"] == (
        "provider_attempt_offline_conformance_000001"
    )

    bad_nonce = conformance.StreamingConformanceOutput.model_construct(
        nonce="wrong_nonce",
        acknowledgement=conformance.ACKNOWLEDGEMENT,
    )
    duplicate_completion = copy.deepcopy(progress)
    duplicate_completion.append(
        {
            **duplicate_completion[-1],
            "sequence": 5,
            "elapsed_ns": 50,
        }
    )
    progress_after_completion = copy.deepcopy(progress)
    progress_after_completion.append(
        {
            **progress_after_completion[-2],
            "sequence": 5,
            "elapsed_ns": 50,
            "kind": "part_ended",
            "channel": "tool_call",
        }
    )
    cases = [
        (
            AttemptedStructuredGenerationResponse(
                response=replace(response, resolved_provider="OtherProvider"),
                attempt_count=1,
            ),
            [outcome_record],
            progress,
        ),
        (
            AttemptedStructuredGenerationResponse(
                response=replace(response, provider_response_id=None),
                attempt_count=1,
            ),
            [outcome_record],
            progress,
        ),
        (
            AttemptedStructuredGenerationResponse(
                response=replace(response, value=bad_nonce),
                attempt_count=1,
            ),
            [outcome_record],
            progress,
        ),
        (attempted, [outcome_record], []),
        (attempted, [outcome_record], progress[:-1]),
        (attempted, [outcome_record], duplicate_completion),
        (attempted, [outcome_record], progress_after_completion),
        (attempted, [], progress),
    ]
    for bad_attempted, bad_outcomes, bad_progress in cases:
        with pytest.raises(conformance.ConformanceRunError):
            conformance.validate_completed_call(
                bad_attempted,
                expected_call_id="call_offline_validation_000001",
                outcome_rows=bad_outcomes,
                progress_rows=bad_progress,
            )


def test_completed_call_rejects_unbound_or_ambiguous_physical_attempts() -> None:
    call_id = "call_offline_binding_validation_000001"
    response = _response()
    attempted = AttemptedStructuredGenerationResponse(
        response=response,
        attempt_count=1,
    )
    outcome = conformance.queued_runner.structured_generation_outcome_record(
        _outcome(response, call_id=call_id)
    )
    progress = _completed_progress_rows(call_id)

    count_mismatch = replace(attempted, attempt_count=2)
    null_identity = copy.deepcopy(outcome)
    null_identity["attempts"][0]["request_evidence"][
        "provider_attempt_id"
    ] = None
    duplicate_identity = copy.deepcopy(outcome)
    duplicate_identity["attempts"] = [
        copy.deepcopy(duplicate_identity["attempts"][0]),
        copy.deepcopy(duplicate_identity["attempts"][0]),
    ]
    duplicate_identity["attempts"][0]["status"] = "retryable_failure"
    duplicate_identity["attempts"][1]["attempt_number"] = 2
    rogue_progress = copy.deepcopy(progress)
    rogue_progress[0]["provider_attempt_id"] = "provider_attempt_rogue_000001"
    wrong_task = copy.deepcopy(outcome)
    wrong_task["task_id"] = "call_wrong_binding_000001"
    wrong_progress_call = copy.deepcopy(progress)
    wrong_progress_call[0]["call_id"] = "call_wrong_binding_000001"

    cases = [
        (count_mismatch, [outcome], progress),
        (attempted, [null_identity], progress),
        (replace(attempted, attempt_count=2), [duplicate_identity], progress),
        (attempted, [outcome], rogue_progress),
        (attempted, [wrong_task], progress),
        (attempted, [outcome], wrong_progress_call),
    ]
    for bad_attempted, bad_outcomes, bad_progress in cases:
        with pytest.raises(conformance.ConformanceRunError):
            conformance.validate_completed_call(
                bad_attempted,
                expected_call_id=call_id,
                outcome_rows=bad_outcomes,
                progress_rows=bad_progress,
            )


def test_live_state_distinguishes_client_construction_from_call_boundary(
    tmp_path: Path,
) -> None:
    gate = _release_gate(tmp_path)
    factory_run = tmp_path / conformance.FROZEN_LIVE_RUN_ID

    def broken_factory(**_):
        raise RuntimeError("offline factory failure")

    with pytest.raises(conformance.ConformanceRunError):
        conformance.execute(
            mode="live",
            run_dir=factory_run,
            release_gate_dir=gate,
            config=_config(),
            dependencies=conformance.ConformanceDependencies(
                credential_loader=lambda: "offline-injected-key",
                runner_factory=broken_factory,
            ),
        )
    factory_result = decode_json_bytes((factory_run / "result.json").read_bytes())
    assert factory_result["client_constructed"] is False
    assert factory_result["provider_call_attempted"] is False


def test_runner_entry_failure_is_after_the_provider_call_boundary(
    tmp_path: Path,
) -> None:
    class EntryFailureRunner:
        async def __aenter__(self):
            raise RuntimeError("offline entry failure")

        async def __aexit__(self, *_):
            return None

        async def __call__(self, _):
            raise AssertionError("call is unreachable")

    run_dir = tmp_path / conformance.FROZEN_LIVE_RUN_ID
    with pytest.raises(conformance.ConformanceRunError):
        conformance.execute(
            mode="live",
            run_dir=run_dir,
            release_gate_dir=_release_gate(tmp_path),
            config=_config(),
            dependencies=conformance.ConformanceDependencies(
                credential_loader=lambda: "offline-injected-key",
                runner_factory=lambda **_: EntryFailureRunner(),
            ),
        )
    result = decode_json_bytes((run_dir / "result.json").read_bytes())
    assert result["client_constructed"] is True
    assert result["provider_call_attempted"] is True


def test_live_preflight_enforces_frozen_run_id_and_liveness_config(
    tmp_path: Path,
) -> None:
    gate = _release_gate(tmp_path)
    credentials = 0

    def credential_loader() -> str:
        nonlocal credentials
        credentials += 1
        return "must-not-be-read"

    dependencies = conformance.ConformanceDependencies(
        credential_loader=credential_loader,
        runner_factory=lambda **_: pytest.fail("runner is unreachable"),
    )
    wrong_id = tmp_path / "wrong_but_safe_live_run_id"
    with pytest.raises(conformance.ConformanceRunError):
        conformance.execute(
            mode="live",
            run_dir=wrong_id,
            release_gate_dir=gate,
            config=_config(),
            dependencies=dependencies,
        )
    wrong_config_dir = tmp_path / "config" / conformance.FROZEN_LIVE_RUN_ID
    with pytest.raises(conformance.ConformanceRunError):
        conformance.execute(
            mode="live",
            run_dir=wrong_config_dir,
            release_gate_dir=gate,
            config=conformance.build_config(
                first_event_timeout_seconds=181,
                idle_timeout_seconds=120,
                absolute_timeout_seconds=0,
            ),
            dependencies=dependencies,
        )
    assert credentials == 0
    assert not wrong_id.exists()
    assert not wrong_config_dir.exists()


def test_progress_is_sealed_when_terminal_outcome_is_published(
    tmp_path: Path,
) -> None:
    class ProgressAfterOutcomeRunner(_OfflineRunner):
        async def __call__(self, request):
            attempted = await super().__call__(request)
            self.progress_sink(
                StructuredStreamProgress(
                    call_id=request.call_id.value,
                    sequence=6,
                    kind=StructuredStreamProgressKind.PART_DELTA,
                    channel=StructuredStreamChannel.TEXT,
                    elapsed_ns=60,
                    event_content_utf8_bytes=1,
                    cumulative_content_utf8_bytes=53,
                    rolling_content_sha256="d" * 64,
                    provider_attempt_id=(
                        "provider_attempt_offline_conformance_000001"
                    ),
                )
            )
            return attempted

    run_dir = tmp_path / conformance.FROZEN_LIVE_RUN_ID
    with pytest.raises(conformance.ConformanceRunError):
        conformance.execute(
            mode="live",
            run_dir=run_dir,
            release_gate_dir=_release_gate(tmp_path),
            config=_config(),
            dependencies=conformance.ConformanceDependencies(
                credential_loader=lambda: "offline-injected-key",
                runner_factory=lambda **kwargs: ProgressAfterOutcomeRunner(
                    progress_sink=kwargs["progress_sink"],
                    outcome_sink=kwargs["outcome_sink"],
                ),
            ),
        )
    result = decode_json_bytes((run_dir / "result.json").read_bytes())
    assert result["status"] == "failed_conformance_only"
    assert result["client_constructed"] is True
    assert result["provider_call_attempted"] is True


def test_route_or_source_drift_fails_before_credential_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = 0

    def credential_loader() -> str:
        nonlocal credentials
        credentials += 1
        return "must-not-be-read"

    release_gate_dir = _release_gate(tmp_path)
    monkeypatch.setattr(conformance, "CAPABILITY_SNAPSHOT_SHA256", "0" * 64)
    with pytest.raises(conformance.ConformanceRunError):
        conformance.execute(
            mode="live",
            run_dir=tmp_path / conformance.FROZEN_LIVE_RUN_ID,
            release_gate_dir=release_gate_dir,
            config=_config(),
            dependencies=conformance.ConformanceDependencies(
                credential_loader=credential_loader,
                runner_factory=lambda **_: pytest.fail("runner is unreachable"),
            ),
        )
    assert credentials == 0
    assert not (tmp_path / conformance.FROZEN_LIVE_RUN_ID).exists()

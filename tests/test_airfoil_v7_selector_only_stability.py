from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.llm_task_queue import (
    LLMAttemptContext,
    RetryDisposition,
)
from agent_evolve.infrastructure.asyncio_runtime import (
    TransportAbortedTimeoutError,
)
from agent_evolve.integrations.pydantic_ai.execution_binding import (
    ExecutionIdRebindingRunner,
    StructuredScienceRequestBinding,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    PydanticAIPortfolioSelectionPolicy,
    _portfolio_output_type,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    TransportOnlyStructuredGenerationRetryClassifier,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from examples.development import (
    run_airfoil_v7_selector_only_stability as stability,
)


class _SyntheticOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


def _synthetic_request(call_id: str = "call_science") -> StructuredGenerationRequest:
    return StructuredGenerationRequest(
        call_id=LLMCallId(call_id),
        operation="synthetic",
        prompt="frozen synthetic prompt",
        output_type=_SyntheticOutput,
        output_tool_name="synthetic_output",
        max_output_tokens=99,
        temperature=0.0,
    )


def test_generic_execution_rebinding_changes_only_queue_identity() -> None:
    science = _synthetic_request()
    binding = StructuredScienceRequestBinding.from_request(science)
    observed: list[StructuredGenerationRequest[Any]] = []

    async def inner(request: StructuredGenerationRequest[Any]) -> object:
        observed.append(request)
        return "ok"

    runner = ExecutionIdRebindingRunner(
        runner=inner,
        expected=binding,
        execution_call_id=LLMCallId("call_execution_000001"),
    )
    assert asyncio.run(runner(science)) == "ok"
    assert observed[0].call_id.value == "call_execution_000001"
    rebound_binding = StructuredScienceRequestBinding.from_request(observed[0])
    assert rebound_binding.provider_fingerprint() == binding.provider_fingerprint()

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(runner(_synthetic_request("call_other_science")))


@pytest.mark.parametrize("kind", tuple(GenerationFailureKind))
def test_transport_only_retry_classifier_is_a_positive_allowlist(
    kind: GenerationFailureKind,
) -> None:
    classifier = TransportOnlyStructuredGenerationRetryClassifier()
    error = StructuredGenerationError(
        kind=kind,
        retryable=True,
        safe_message="sanitized fixture",
    )
    classified = classifier.classify(
        error,
        context=LLMAttemptContext("classifier", 1, 2),
    )
    allowed = {
        GenerationFailureKind.TIMEOUT,
        GenerationFailureKind.PROVIDER_UNAVAILABLE,
    }
    assert classified.disposition is (
        RetryDisposition.RETRY if kind in allowed else RetryDisposition.FAIL
    )
    if classified.sanitized_failure is not None:
        assert classified.sanitized_failure.retryable is True


def test_transport_only_classifier_handles_ordinary_and_hard_timeouts() -> None:
    classifier = TransportOnlyStructuredGenerationRetryClassifier()
    context = LLMAttemptContext("classifier", 1, 2)
    assert classifier.classify(
        TimeoutError(), context=context
    ).disposition is RetryDisposition.RETRY
    assert classifier.classify(
        TransportAbortedTimeoutError(), context=context
    ).disposition is RetryDisposition.FAIL


def test_slot_runtime_evidence_rejects_repair_and_nontransport_retry() -> None:
    slot = stability.SCHEDULE[0]
    request_row = {
        "record_type": "request",
        "absolute_slot": slot.absolute_slot,
        "view_id": slot.view_id,
        "execution_call_id": slot.execution_call_id,
        "operation": "select_portfolio",
        "prompt_sha256": stability.PROMPT_SHA256[slot.view_id],
        "prompt_utf8_bytes": stability.PROMPT_UTF8_BYTES[slot.view_id],
        "output_schema_sha256": stability.OUTPUT_SCHEMA_SHA256,
        "output_schema_utf8_bytes": stability.OUTPUT_SCHEMA_UTF8_BYTES,
        "output_tool_name": stability.PORTFOLIO_SELECTION_TOOL_NAME,
        "max_output_tokens": stability.MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }

    def attempt(number: int, *, variant: str = "original") -> dict[str, object]:
        return {
            "attempt_number": number,
            "request_evidence": {
                "variant": variant,
                "prompt_sha256": stability.PROMPT_SHA256[slot.view_id],
            },
            "wait_time_ns": 0,
            "service_time_ns": 1,
            "policy_backoff_ns": 0,
            "retry_after_ns": 0,
            "scheduled_delay_ns": 0,
            "will_retry": number == 1,
            "classification": (
                {"disposition": "retry", "reason": "transient"}
                if number == 1
                else None
            ),
        }

    base = {
        "task_id": slot.execution_call_id,
        "published_monotonic_ns": 1,
        "published_wall_utc": "2026-07-15T00:00:00+00:00",
        "status": "succeeded",
    }
    with pytest.raises(RuntimeError, match="changed or lost"):
        stability._validate_slot_runtime_evidence(
            slot,
            queue_row={**base, "attempts": [attempt(1, variant="schema_repair_v2")]},
            journal_rows=[request_row, {"record_type": "response"}],
            valid_response=True,
        )
    nontransport = attempt(1)
    nontransport["classification"] = {
        "disposition": "retry",
        "reason": "output_invalid",
    }
    with pytest.raises(RuntimeError, match="allowlisted transport"):
        stability._validate_slot_runtime_evidence(
            slot,
            queue_row={**base, "attempts": [nontransport, attempt(2)]},
            journal_rows=[request_row, {"record_type": "response"}],
            valid_response=True,
        )


def test_authentication_reconstructs_exact_frozen_bank_prompt_and_schema() -> None:
    bank = stability.authenticate_frozen_bank()
    assert bank.record["strict_card_count"] == 8
    bindings = {
        row["view_id"]: row for row in bank.record["low_level_bindings"]
    }
    assert {view: row["prompt_sha256"] for view, row in bindings.items()} == (
        stability.PROMPT_SHA256
    )
    assert {
        view: row["prompt_utf8_bytes"] for view, row in bindings.items()
    } == stability.PROMPT_UTF8_BYTES
    assert {
        row["output_schema_sha256"] for row in bindings.values()
    } == {stability.OUTPUT_SCHEMA_SHA256}


def _release_evidence(tmp_path: Path, *, passed_count: int = 1) -> Path:
    path = tmp_path / "release_evidence.json"
    stability.frozen.write_json_atomic(
        path,
        {
            "schema_version": 1,
            "kind": "airfoil_v7_selector_stability_release_gate_v1",
            "source_snapshot_sha256": stability.source_snapshot()["sha256"],
            "protocol_artifact_sha256": stability.PROTOCOL_ARTIFACT_SHA256,
            "commands": list(stability.RELEASE_GATE_COMMANDS),
            "required_test_names": list(stability.REQUIRED_RELEASE_TESTS),
            "exit_codes": [0, 0],
            "passed_count": passed_count,
            "failures": 0,
            "errors": 0,
            "warnings": 0,
            "provider_calls": 0,
            "recorded_at_utc": "2026-07-15T00:00:00+00:00",
        },
    )
    return path


def test_manifest_binds_serial_schedule_and_fresh_stack_policy(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "selector.manifest.json"
    output = tmp_path / "selector_test"
    release_evidence = _release_evidence(tmp_path)
    record = stability.write_manifest(
        manifest,
        run_id="selector_test",
        output_dir=output,
        release_evidence_path=release_evidence,
    )
    verified = stability.verify_manifest(manifest)

    assert verified.output_dir == output
    assert [row["view_id"] for row in record["schedule"]] == list("MPNPNMNMP")
    queue = record["provider_policy"]["queue"]
    assert queue["fresh_owned_stack_per_logical_slot"] is True
    assert queue["max_in_flight"] == 1
    assert queue["max_pending"] == 0
    assert queue["max_connections"] == 1
    assert queue["attempt_request_policy"] == "exact_payload"
    assert record["authorization"]["logical_selector_calls"] == 9
    assert not output.exists()

    for index, mutation in enumerate(("extra", "claim", "authorization")):
        tampered = dict(record)
        if mutation == "extra":
            tampered["unexpected"] = True
        elif mutation == "claim":
            tampered["claim_boundary"] = "broadened claim"
        else:
            tampered["authorization"] = {
                **tampered["authorization"],
                "logical_selector_calls": 10,
            }
        tampered.pop("manifest_sha256")
        tampered["manifest_sha256"] = stability.frozen._domain_sha256(
            tampered,
            stability.MANIFEST_FRAMING,
        )
        tampered_path = tmp_path / f"tampered_{index}.json"
        stability.frozen.write_json_atomic(tampered_path, tampered)
        with pytest.raises(RuntimeError):
            stability.verify_manifest(tampered_path)

    release_record = stability.json.loads(release_evidence.read_text())
    release_record["provider_calls"] = 1
    release_evidence.write_text(stability.json.dumps(release_record))
    with pytest.raises(RuntimeError, match="release-gate"):
        stability.verify_manifest(manifest)


def test_predispatch_rejects_provider_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "selector.manifest.json"
    output = tmp_path / "selector_policy_drift"
    stability.write_manifest(
        manifest,
        run_id="selector_policy_drift",
        output_dir=output,
        release_evidence_path=_release_evidence(tmp_path),
    )
    verified = stability.verify_manifest(manifest)
    original = stability.provider_policy_record

    def drifted() -> dict[str, object]:
        return {**original(), "provider": "drifted-provider"}

    monkeypatch.setattr(stability, "provider_policy_record", drifted)
    with pytest.raises(RuntimeError, match="provider route or queue policy"):
        stability.reverify_dispatch_inputs(verified)


def _typed_selector_response(
    request: StructuredGenerationRequest[Any],
) -> StructuredGenerationResponse[Any]:
    output_type = request.output_type
    member_type = output_type.model_fields["members"].annotation.__args__[0]
    option_ids = tuple(output_type.option_family_by_id)[:3]
    card_key = sorted(member_type.allowed_card_keys)[0]
    value = output_type.model_validate(
        {
            "members": [
                {
                    "option_id": option_id,
                    "supporting_card_keys": [card_key],
                    "effect_predictions": [
                        {"metric_id": metric_id, "direction": "unchanged"}
                        for metric_id in member_type.required_metric_ids
                    ],
                    "design_rationale": "Deterministic provider-free fixture.",
                }
                for option_id in option_ids
            ]
        },
        strict=True,
    )
    return StructuredGenerationResponse(
        value=value,
        requested_model=stability.MODEL,
        resolved_model=stability.MODEL,
        resolved_provider="StreamLake",
        provider_response_id=f"fixture-{request.call_id.value}",
        finish_reason="tool_call",
        input_tokens=100,
        output_tokens=100,
        reasoning_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=None,
        latency_ns=1,
    )


class _FakeSlotExecutor:
    def __init__(self, *, failed_slot: int | None = None) -> None:
        self.failed_slot = failed_slot
        self.slots: list[int] = []
        self.active = 0
        self.closed: set[int] = set()

    async def __call__(
        self,
        *,
        api_key: str,
        slot: stability.SelectorSlot,
        historical_request: Any,
        queue_sink: Any,
        journal_sink: Any,
        pre_dispatch: Any,
    ) -> Any:
        assert api_key == "provider-free-key"
        assert self.active == 0
        self.active += 1
        self.slots.append(slot.absolute_slot)

        async def low(request: StructuredGenerationRequest[Any]) -> object:
            queue_sink(
                {
                    "schema_version": 4,
                    "task_id": request.call_id.value,
                    "status": (
                        "terminal_failure"
                        if slot.absolute_slot == self.failed_slot
                        else "succeeded"
                    ),
                    "attempts": [
                        {
                            "attempt_number": 1,
                            "status": (
                                "failed"
                                if slot.absolute_slot == self.failed_slot
                                else "succeeded"
                            ),
                            "request_evidence": {
                                "variant": "original",
                                "prompt_sha256": stability.PROMPT_SHA256[
                                    slot.view_id
                                ],
                            },
                            "wait_time_ns": 0,
                            "service_time_ns": 1,
                            "will_retry": False,
                            "policy_backoff_ns": 0,
                            "retry_after_ns": 0,
                            "scheduled_delay_ns": 0,
                            "classification": (
                                {
                                    "disposition": "fail",
                                    "reason": "timeout",
                                }
                                if slot.absolute_slot == self.failed_slot
                                else None
                            ),
                        }
                    ],
                }
            )
            if slot.absolute_slot == self.failed_slot:
                raise RuntimeError("sanitized hard-timeout fixture")
            return _typed_selector_response(request)

        audited = stability.HistoricalScienceExecutionRunner(
            low,
            slot=slot,
            historical_request=historical_request,
            pre_dispatch=pre_dispatch,
            journal_sink=journal_sink,
        )
        selector = PydanticAIPortfolioSelectionPolicy(audited)
        try:
            return await selector.select(historical_request)
        finally:
            self.active -= 1
            self.closed.add(slot.absolute_slot)


class _MissingJournalExecutor(_FakeSlotExecutor):
    async def __call__(
        self,
        *,
        api_key: str,
        slot: stability.SelectorSlot,
        historical_request: Any,
        queue_sink: Any,
        journal_sink: Any,
        pre_dispatch: Any,
    ) -> Any:
        del journal_sink, pre_dispatch
        assert api_key == "provider-free-key"
        self.slots.append(slot.absolute_slot)
        self.active += 1
        output_type = _portfolio_output_type(historical_request)
        execution_request = StructuredGenerationRequest(
            call_id=LLMCallId(slot.execution_call_id),
            operation="select_portfolio",
            prompt=stability.render_portfolio_selection_prompt(historical_request),
            output_type=output_type,
            output_tool_name=stability.PORTFOLIO_SELECTION_TOOL_NAME,
            max_output_tokens=stability.MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        queue_sink(
            {
                "schema_version": 4,
                "task_id": slot.execution_call_id,
                "status": "succeeded",
                "attempts": [
                    {
                        "attempt_number": 1,
                        "status": "succeeded",
                        "request_evidence": {
                            "variant": "original",
                            "prompt_sha256": stability.PROMPT_SHA256[slot.view_id],
                        },
                        "wait_time_ns": 0,
                        "service_time_ns": 1,
                        "will_retry": False,
                        "policy_backoff_ns": 0,
                        "retry_after_ns": 0,
                        "scheduled_delay_ns": 0,
                        "classification": None,
                    }
                ],
            }
        )

        async def low(_: StructuredGenerationRequest[Any]) -> object:
            return _typed_selector_response(execution_request)

        try:
            return await PydanticAIPortfolioSelectionPolicy(low).select(
                historical_request
            )
        finally:
            self.active -= 1
            self.closed.add(slot.absolute_slot)


def _run_fake_study(
    tmp_path: Path,
    *,
    failed_slot: int | None,
    tick_step_ns: int = 5_000_000_000,
    executor: _FakeSlotExecutor | None = None,
) -> tuple[dict[str, object], _FakeSlotExecutor, Path, list[float]]:
    manifest = tmp_path / "selector.manifest.json"
    output = tmp_path / "selector_fake"
    stability.write_manifest(
        manifest,
        run_id="selector_fake",
        output_dir=output,
        release_evidence_path=_release_evidence(tmp_path),
    )
    executor = executor or _FakeSlotExecutor(failed_slot=failed_slot)
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        # Quiet begins only after the stack closed and disposition fsynced.
        disposition_count = len(
            (output / "slot_dispositions.jsonl").read_text().splitlines()
        )
        assert disposition_count in executor.closed
        sleeps.append(seconds)

    ticks = iter(range(0, tick_step_ns * 100, tick_step_ns))

    def credential_loader() -> str:
        assert (output / "authenticated_bank.json").is_file()
        assert (output / "schedule.json").is_file()
        source_rows = [
            stability.json.loads(line)
            for line in (output / "source_verifications.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [row["stage"] for row in source_rows] == [
            "post_run_directory_creation",
            "pre_credential_load",
        ]
        return "provider-free-key"

    result = stability.execute_with_dependencies(
        manifest,
        stability.LiveDependencies(
            credential_loader=credential_loader,
            slot_executor=executor,
            sleep=sleep,
            monotonic_ns=lambda: next(ticks),
            wall_time=lambda: "2026-07-15T00:00:00+00:00",
        ),
    )
    return result, executor, output, sleeps


@pytest.mark.parametrize("failed_slot", (1, 5, 9))
def test_serial_controller_continues_after_timeout_and_defers_analysis(
    tmp_path: Path,
    failed_slot: int,
) -> None:
    result, executor, output, sleeps = _run_fake_study(
        tmp_path,
        failed_slot=failed_slot,
    )
    assert executor.slots == list(range(1, 10))
    assert executor.closed == set(range(1, 10))
    assert sleeps == [5.0] * 8
    assert result["decision"] == "transport_incomplete"
    assert result["valid_response_count"] == 8
    assert result["scientific_analysis_performed"] is False
    assert not (output / "analysis.json").exists()
    assert len((output / "slot_dispositions.jsonl").read_text().splitlines()) == 9
    assert len((output / "provider_queue_outcomes.jsonl").read_text().splitlines()) == 9
    queue_rows = [
        stability.json.loads(line)
        for line in (output / "provider_queue_outcomes.jsonl").read_text().splitlines()
    ]
    assert all(type(row["published_monotonic_ns"]) is int for row in queue_rows)
    assert all(type(row["published_wall_utc"]) is str for row in queue_rows)


def test_complete_batch_analyzes_only_after_all_nine_stacks_close(
    tmp_path: Path,
) -> None:
    result, executor, output, sleeps = _run_fake_study(
        tmp_path,
        failed_slot=None,
    )
    assert executor.slots == list(range(1, 10))
    assert executor.closed == set(range(1, 10))
    assert sleeps == [5.0] * 8
    assert result["transport_complete"] is True
    assert result["scientific_analysis_performed"] is True
    assert result["decision"] == "do_not_advance"
    analysis = stability.json.loads((output / "analysis.json").read_text())
    assert analysis["stability_gate_passes"] is True
    assert analysis["scientific_gate_passes"] is False


def test_short_quiet_interval_fails_closed_and_seals(tmp_path: Path) -> None:
    output = tmp_path / "selector_fake"
    with pytest.raises(RuntimeError, match="shorter than five seconds"):
        _run_fake_study(
            tmp_path,
            failed_slot=None,
            tick_step_ns=1,
        )
    finalized = stability.json.loads((output / "finalized.json").read_text())
    assert finalized["status"] == "failed"
    assert len((output / "slot_dispositions.jsonl").read_text().splitlines()) == 1


def test_missing_journal_terminal_is_infrastructure_failure(tmp_path: Path) -> None:
    output = tmp_path / "selector_fake"
    with pytest.raises(RuntimeError, match="journal"):
        _run_fake_study(
            tmp_path,
            failed_slot=None,
            executor=_MissingJournalExecutor(),
        )
    finalized = stability.json.loads((output / "finalized.json").read_text())
    assert finalized["status"] == "failed"
    assert not (output / "summary.json").exists()

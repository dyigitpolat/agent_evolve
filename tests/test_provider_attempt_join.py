"""Offline contract tests for physical provider-attempt terminal joins."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    OpenRouterOutboundRequestManifestPublisher,
)
from agent_evolve.integrations.pydantic_ai.provider_attempt_join import (
    build_provider_attempt_terminal_join_receipt as _production_build_join,
    explicit_pre_transport_failure_evidence_record,
    validate_explicit_pre_transport_failure_evidence_record,
    validate_provider_attempt_terminal_join_receipt,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION,
    structured_generation_request_evidence_record,
)
from agent_evolve.ports.structured_generator import StructuredGenerationRequest


_JOIN_DOMAIN = b"agent-evolve:provider-attempt-terminal-join:v1\x00"
_PROMPT_SHA256 = hashlib.sha256(b"offline join probe").hexdigest()
_PROGRESS_SHA256 = hashlib.sha256(b"x").hexdigest()
_EXPECTED_TRANSPORT_KEYS = (
    "model",
    "provider",
    "reasoning",
    "usage",
    "stream",
    "stream_options",
    "tool_choice",
    "response_format",
)


class _JoinOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


class _DifferentJoinOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    different_value: str


def _structured_request(
    *,
    call_id: str,
    provider_attempt_id: str,
    operation: str = "provider_join_probe",
    prompt: str = "offline join probe",
    output_type: type[BaseModel] = _JoinOutput,
    output_tool_name: str = "return_join_probe",
    max_output_tokens: int = 64,
    temperature: float | None = None,
) -> StructuredGenerationRequest[BaseModel]:
    return StructuredGenerationRequest(
        call_id=LLMCallId(call_id),
        operation=operation,
        prompt=prompt,
        output_type=output_type,
        output_tool_name=output_tool_name,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        provider_attempt_id=ProviderAttemptId(provider_attempt_id),
    )


def _outbound_manifest(
    *,
    call_id: str,
    provider_attempt_id: str,
) -> dict[str, object]:
    """Publish one real manifest through the production pre-transport hook."""

    request = _structured_request(
        call_id=call_id,
        provider_attempt_id=provider_attempt_id,
    )
    logical_schema = copy.deepcopy(_JoinOutput.model_json_schema(mode="validation"))
    logical_schema.pop("description", None)
    wire_schema = OpenAIJsonSchemaTransformer(
        logical_schema,
        strict=False,
    ).walk()
    body = {
        "messages": [{"role": "user", "content": request.prompt}],
        "model": "offline/test-model",
        "max_completion_tokens": request.max_output_tokens,
        "stream": False,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": request.output_tool_name,
                    "description": "content-free offline fixture",
                    "parameters": wire_schema,
                },
            }
        ],
        "provider": {"only": ["offline"], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    rows: list[dict[str, object]] = []
    publisher = OpenRouterOutboundRequestManifestPublisher(rows.append)
    http_request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
        json=body,
    )
    with publisher.bind(
        request,
        requested_model="offline/test-model",
        provider={"only": ["offline"], "allow_fallbacks": False},
        reasoning=None,
        stream=False,
    ):
        asyncio.run(publisher.httpx_request_hook(http_request))
    assert len(rows) == 1
    return rows[0]


def _logical_request(
    *,
    call_id: str,
    operation: str = "provider_join_probe",
    prompt: str = "offline join probe",
    output_type: type[BaseModel] = _JoinOutput,
    output_tool_name: str = "return_join_probe",
    max_output_tokens: int = 64,
    temperature: float | None = None,
) -> dict[str, object]:
    return structured_generation_request_evidence_record(
        _structured_request(
            call_id=call_id,
            provider_attempt_id="provider_attempt_logical_fixture_000001",
            operation=operation,
            prompt=prompt,
            output_type=output_type,
            output_tool_name=output_tool_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    )


def _attempt(
    *,
    attempt_number: int,
    provider_attempt_id: str,
    status: str,
    status_code: int | None = None,
) -> dict[str, object]:
    failed = status != "succeeded"
    return {
        "attempt_number": attempt_number,
        "status": status,
        "wait_time_ns": 1,
        "service_time_ns": 2,
        "will_retry": status == "retryable_failure",
        "policy_backoff_ns": 0,
        "retry_after_ns": 0,
        "scheduled_delay_ns": 0,
        "error_type": "StructuredGenerationError" if failed else None,
        "request_evidence": {
            "variant": "original",
            "prompt_sha256": _PROMPT_SHA256,
            "provider_attempt_id": provider_attempt_id,
        },
        "classification": (
            None
            if not failed
            else {
                "disposition": ("retry" if status == "retryable_failure" else "fail"),
                "reason": (
                    "transient" if status == "retryable_failure" else "permanent"
                ),
            }
        ),
        "failure": (
            None
            if not failed
            else {
                "kind": "invalid_request",
                "retryable": status == "retryable_failure",
                "safe_message": "sanitized offline failure",
                "status_code": status_code,
                "retry_after_seconds": None,
                "provider_error_code": None,
                "provider_error_envelope_sha256": None,
                "exception_provenance": None,
                "stream_timeout_phase": None,
                "output_failure_mode": None,
                "validation_issues": [],
            }
        ),
    }


def _outcome(
    *,
    call_id: str,
    attempts: list[dict[str, object]],
    succeeded: bool,
) -> dict[str, object]:
    return {
        "schema_version": STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION,
        "task_id": call_id,
        "status": "succeeded" if succeeded else "terminal_failure",
        "cancellation_reason": None,
        "queue_time_ns": 1,
        "service_time_ns": 2,
        "total_time_ns": 3,
        "attempts": attempts,
        "response": (
            {
                "requested_model": "offline/test-model",
                "resolved_model": "offline/test-model",
                "resolved_provider": "offline",
                "provider_response_id": "response-offline-1",
                "finish_reason": "tool_call",
                "input_tokens": 8,
                "output_tokens": 2,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": "0.000001",
                "latency_ns": 2,
            }
            if succeeded
            else None
        ),
    }


def _progress(
    *,
    call_id: str,
    provider_attempt_id: str,
    sequence: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "call_id": call_id,
        "provider_attempt_id": provider_attempt_id,
        "sequence": sequence,
        "kind": "part_delta",
        "channel": "text",
        "elapsed_ns": sequence,
        "event_content_utf8_bytes": 1,
        "cumulative_content_utf8_bytes": sequence,
        "rolling_content_sha256": _PROGRESS_SHA256,
    }


def _success_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    call_id = "call_provider_join_success_000001"
    attempt_id = "provider_attempt_provider_join_success_000001"
    manifest = _outbound_manifest(
        call_id=call_id,
        provider_attempt_id=attempt_id,
    )
    outcome = _outcome(
        call_id=call_id,
        attempts=[
            _attempt(
                attempt_number=1,
                provider_attempt_id=attempt_id,
                status="succeeded",
            )
        ],
        succeeded=True,
    )
    progress = [
        _progress(
            call_id=call_id,
            provider_attempt_id=attempt_id,
            sequence=sequence,
        )
        for sequence in (1, 2)
    ]
    return manifest, outcome, progress


def _join_hash(record: dict[str, object]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(_JOIN_DOMAIN + payload).hexdigest()


def build_provider_attempt_terminal_join_receipt(
    *,
    logical_requests: list[dict[str, object]],
    outbound_manifests: list[dict[str, object]],
    terminal_outcomes: list[dict[str, object]],
    progress_rows: list[dict[str, object]],
    explicit_pre_transport_failures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    expectations: dict[str, object] = {}
    if outbound_manifests:
        manifest = outbound_manifests[0]
        frameworks = manifest["framework_versions"]
        settings = manifest["settings"]
        assert isinstance(frameworks, dict)
        assert isinstance(settings, dict)
        expectations = {
            "expected_framework_versions": frameworks,
            "expected_transport_settings": {
                key: settings[key] for key in _EXPECTED_TRANSPORT_KEYS
            },
        }
    return _production_build_join(
        logical_requests=logical_requests,
        outbound_manifests=outbound_manifests,
        terminal_outcomes=terminal_outcomes,
        progress_rows=progress_rows,
        explicit_pre_transport_failures=(
            []
            if explicit_pre_transport_failures is None
            else explicit_pre_transport_failures
        ),
        **expectations,
    )


def test_success_manifest_outcome_and_progress_form_one_valid_join() -> None:
    manifest, outcome, progress = _success_fixture()

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is True
    assert receipt["source_counts"] == {
        "logical_requests": 1,
        "outbound_manifests": 1,
        "terminal_outcomes": 1,
        "outcome_attempts_with_physical_ids": 1,
        "progress_rows": 2,
        "explicit_pre_transport_failures": 0,
    }
    assert validate_provider_attempt_terminal_join_receipt(receipt) == receipt


def test_outcome_with_extra_raw_provider_body_is_rejected_without_retaining_it() -> (
    None
):
    manifest, outcome, progress = _success_fixture()
    raw_marker = "raw-provider-body-must-not-cross-join-boundary"
    outcome["raw_provider_body"] = raw_marker

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["malformed_row_indices"]["terminal_outcomes"] == [0]
    assert receipt["invariants"]["raw_provider_content_persisted"] is False
    assert raw_marker not in json.dumps(receipt, sort_keys=True)


def test_succeeded_attempt_with_non_null_failure_is_rejected() -> None:
    manifest, outcome, progress = _success_fixture()
    contradictory_failure = _attempt(
        attempt_number=1,
        provider_attempt_id=manifest["provider_attempt_id"],
        status="terminal_failure",
        status_code=400,
    )["failure"]
    outcome["attempts"][0]["failure"] = contradictory_failure

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["malformed_row_indices"]["terminal_outcomes"] == [0]


def test_progress_with_extra_raw_field_is_rejected_without_retaining_it() -> None:
    manifest, outcome, progress = _success_fixture()
    raw_marker = "raw-stream-delta-must-not-cross-join-boundary"
    progress[0]["raw_provider_delta"] = raw_marker

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["malformed_row_indices"]["progress_rows"] == [0]
    assert receipt["invariants"]["raw_provider_content_persisted"] is False
    assert raw_marker not in json.dumps(receipt, sort_keys=True)


def test_http_400_is_terminal_provider_evidence_and_joins_its_manifest() -> None:
    call_id = "call_provider_join_http400_000001"
    attempt_id = "provider_attempt_provider_join_http400_000001"
    manifest = _outbound_manifest(
        call_id=call_id,
        provider_attempt_id=attempt_id,
    )
    outcome = _outcome(
        call_id=call_id,
        attempts=[
            _attempt(
                attempt_number=1,
                provider_attempt_id=attempt_id,
                status="terminal_failure",
                status_code=400,
            )
        ],
        succeeded=False,
    )

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=[],
    )

    assert receipt["join_valid"] is True
    assert receipt["provider_attempt_ids"]["provider_status_response_or_progress"] == [
        attempt_id
    ]


def test_schema_v7_closed_semantic_reason_code_survives_strict_join_validation() -> (
    None
):
    call_id = "call_provider_join_semantic_reason_000001"
    attempt_id = "provider_attempt_provider_join_semantic_reason_000001"
    manifest = _outbound_manifest(
        call_id=call_id,
        provider_attempt_id=attempt_id,
    )
    attempt = _attempt(
        attempt_number=1,
        provider_attempt_id=attempt_id,
        status="terminal_failure",
    )
    attempt["classification"] = {
        "disposition": "fail",
        "reason": "output_invalid",
    }
    attempt["failure"] = {
        "kind": "output_invalid",
        "retryable": True,
        "safe_message": "model output violated the typed response contract",
        "status_code": None,
        "retry_after_seconds": None,
        "provider_error_code": None,
        "provider_error_envelope_sha256": None,
        "stream_timeout_phase": None,
        "output_failure_mode": "schema_validation",
        "validation_issues": [
            {
                "category": "semantic_constraint",
                "location": ["root"],
                "reason_code": "duplicate_finite_options",
            }
        ],
    }
    outcome = _outcome(
        call_id=call_id,
        attempts=[attempt],
        succeeded=False,
    )
    # This fixture explicitly exercises the prior durable schema.
    outcome["schema_version"] = 7

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=[
            _progress(
                call_id=call_id,
                provider_attempt_id=attempt_id,
                sequence=1,
            )
        ],
    )

    assert receipt["join_valid"] is True
    assert receipt["provider_attempt_ids"]["provider_status_response_or_progress"] == [
        attempt_id
    ]

    legacy_outcome = copy.deepcopy(outcome)
    legacy_outcome["schema_version"] = 6
    legacy_failure = legacy_outcome["attempts"][0]["failure"]
    del legacy_failure["validation_issues"][0]["reason_code"]
    legacy_receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[manifest],
        terminal_outcomes=[legacy_outcome],
        progress_rows=[
            _progress(
                call_id=call_id,
                provider_attempt_id=attempt_id,
                sequence=1,
            )
        ],
    )
    assert legacy_receipt["join_valid"] is True

    unsafe_outcome = copy.deepcopy(outcome)
    unsafe_marker = "RAW_SECRET_UNTRUSTED_REASON"
    unsafe_outcome["attempts"][0]["failure"]["validation_issues"][0]["reason_code"] = (
        unsafe_marker
    )
    unsafe_receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[manifest],
        terminal_outcomes=[unsafe_outcome],
        progress_rows=[
            _progress(
                call_id=call_id,
                provider_attempt_id=attempt_id,
                sequence=1,
            )
        ],
    )
    assert unsafe_receipt["join_valid"] is False
    assert unsafe_marker not in json.dumps(unsafe_receipt, sort_keys=True)


def test_framework_version_expectation_mismatch_is_invalid() -> None:
    manifest, outcome, progress = _success_fixture()
    frameworks = dict(manifest["framework_versions"])
    frameworks["httpx"] = "999.invalid"
    settings = manifest["settings"]
    assert isinstance(settings, dict)

    receipt = _production_build_join(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
        expected_framework_versions=frameworks,
        expected_transport_settings={
            key: settings[key] for key in _EXPECTED_TRANSPORT_KEYS
        },
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["framework_version_mismatch_attempt_ids"] == [
        manifest["provider_attempt_id"]
    ]


def test_selected_profile_transport_expectation_mismatch_is_invalid() -> None:
    manifest, outcome, progress = _success_fixture()
    frameworks = manifest["framework_versions"]
    settings = manifest["settings"]
    assert isinstance(frameworks, dict)
    assert isinstance(settings, dict)
    expected_settings = {key: settings[key] for key in _EXPECTED_TRANSPORT_KEYS}
    expected_settings["model"] = "different/model"

    receipt = _production_build_join(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
        expected_framework_versions=frameworks,
        expected_transport_settings=expected_settings,
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["transport_settings_mismatch_attempt_ids"] == [
        manifest["provider_attempt_id"]
    ]


def test_manifest_without_logical_request_is_invalid() -> None:
    manifest, outcome, progress = _success_fixture()

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["missing_logical_request_attempt_ids"] == [
        manifest["provider_attempt_id"]
    ]


def test_duplicate_logical_request_for_one_call_is_invalid() -> None:
    manifest, outcome, progress = _success_fixture()
    logical = _logical_request(call_id=manifest["call_id"])

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[logical, copy.deepcopy(logical)],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["duplicate_logical_request_call_ids"] == [
        manifest["call_id"]
    ]


@pytest.mark.parametrize(
    "logical_overrides",
    [
        {"operation": "provider_join_different_operation"},
        {"prompt": "offline join probe with different bytes"},
        {"output_type": _DifferentJoinOutput},
        {"output_tool_name": "return_different_join_probe"},
        {"max_output_tokens": 65},
        {"temperature": 0.5},
    ],
)
def test_authentic_logical_and_physical_request_mismatch_is_invalid(
    logical_overrides: dict[str, object],
) -> None:
    manifest, outcome, progress = _success_fixture()
    logical = _logical_request(
        call_id=manifest["call_id"],
        **logical_overrides,
    )

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[logical],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["logical_physical_mismatch_attempt_ids"] == [
        manifest["provider_attempt_id"]
    ]


def test_missing_manifest_is_invalid_and_never_inferred_as_pretransport() -> None:
    call_id = "call_provider_join_missing_000001"
    attempt_id = "provider_attempt_provider_join_missing_000001"
    outcome = _outcome(
        call_id=call_id,
        attempts=[
            _attempt(
                attempt_number=1,
                provider_attempt_id=attempt_id,
                status="terminal_failure",
                status_code=None,
            )
        ],
        succeeded=False,
    )

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[],
        terminal_outcomes=[outcome],
        progress_rows=[],
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["missing_manifest_attempt_ids"] == [attempt_id]
    assert receipt["provider_attempt_ids"]["explicit_pre_transport_failures"] == []


def test_progress_without_manifest_or_outcome_is_invalid() -> None:
    call_id = "call_provider_join_progress_only_000001"
    attempt_id = "provider_attempt_provider_join_progress_only_000001"

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[],
        outbound_manifests=[],
        terminal_outcomes=[],
        progress_rows=[
            _progress(
                call_id=call_id,
                provider_attempt_id=attempt_id,
                sequence=1,
            )
        ],
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["progress_without_manifest_attempt_ids"] == [attempt_id]
    assert receipt["defects"]["progress_without_outcome_attempt_ids"] == [attempt_id]


def test_duplicate_manifest_for_one_physical_attempt_is_invalid() -> None:
    manifest, outcome, progress = _success_fixture()

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest, copy.deepcopy(manifest)],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["duplicate_outbound_manifest_attempt_ids"] == [
        manifest["provider_attempt_id"]
    ]


def test_conflicting_progress_call_ids_are_invalid() -> None:
    manifest, outcome, progress = _success_fixture()
    progress[1]["call_id"] = "call_provider_join_conflict_000002"

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False
    assert receipt["defects"]["call_id_mismatch_attempt_ids"] == [
        manifest["provider_attempt_id"]
    ]


@pytest.mark.parametrize("sequences", [(1, 1), (1, 3)])
def test_progress_sequences_must_be_exactly_contiguous(
    sequences: tuple[int, int],
) -> None:
    manifest, outcome, progress = _success_fixture()
    for row, sequence in zip(progress, sequences, strict=True):
        row["sequence"] = sequence

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=manifest["call_id"])],
        outbound_manifests=[manifest],
        terminal_outcomes=[outcome],
        progress_rows=progress,
    )

    assert receipt["join_valid"] is False


def test_outcome_attempt_numbers_must_start_at_one_and_be_contiguous() -> None:
    call_id = "call_provider_join_attempt_numbers_000001"
    first_id = "provider_attempt_provider_join_attempt_numbers_000001"
    second_id = "provider_attempt_provider_join_attempt_numbers_000002"
    outcome = _outcome(
        call_id=call_id,
        attempts=[
            _attempt(
                attempt_number=1,
                provider_attempt_id=first_id,
                status="retryable_failure",
                status_code=429,
            ),
            _attempt(
                attempt_number=3,
                provider_attempt_id=second_id,
                status="terminal_failure",
                status_code=400,
            ),
        ],
        succeeded=False,
    )

    receipt = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[
            _outbound_manifest(
                call_id=call_id,
                provider_attempt_id=first_id,
            ),
            _outbound_manifest(
                call_id=call_id,
                provider_attempt_id=second_id,
            ),
        ],
        terminal_outcomes=[outcome],
        progress_rows=[],
    )

    assert receipt["join_valid"] is False


def test_explicit_pretransport_exemption_requires_authentic_evidence() -> None:
    call_id = "call_provider_join_pretransport_000001"
    attempt_id = "provider_attempt_provider_join_pretransport_000001"
    outcome = _outcome(
        call_id=call_id,
        attempts=[
            _attempt(
                attempt_number=1,
                provider_attempt_id=attempt_id,
                status="terminal_failure",
                status_code=None,
            )
        ],
        succeeded=False,
    )
    evidence = explicit_pre_transport_failure_evidence_record(
        call_id=call_id,
        provider_attempt_id=attempt_id,
        failure_stage="httpx.pre_transport",
    )
    assert validate_explicit_pre_transport_failure_evidence_record(evidence) == evidence

    valid = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[],
        terminal_outcomes=[outcome],
        progress_rows=[],
        explicit_pre_transport_failures=[evidence],
    )
    assert valid["join_valid"] is True

    tampered = copy.deepcopy(evidence)
    tampered["failure_stage"] = "httpx.different_stage"
    with pytest.raises(ValueError, match="hash"):
        validate_explicit_pre_transport_failure_evidence_record(tampered)
    invalid = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[],
        terminal_outcomes=[outcome],
        progress_rows=[],
        explicit_pre_transport_failures=[tampered],
    )
    assert invalid["join_valid"] is False
    assert invalid["malformed_row_indices"]["explicit_pre_transport_failures"] == [0]
    assert invalid["defects"]["missing_manifest_attempt_ids"] == [attempt_id]


def test_recomputed_self_hash_cannot_turn_an_invalid_join_green() -> None:
    call_id = "call_provider_join_forged_000001"
    attempt_id = "provider_attempt_provider_join_forged_000001"
    invalid = build_provider_attempt_terminal_join_receipt(
        logical_requests=[_logical_request(call_id=call_id)],
        outbound_manifests=[],
        terminal_outcomes=[
            _outcome(
                call_id=call_id,
                attempts=[
                    _attempt(
                        attempt_number=1,
                        provider_attempt_id=attempt_id,
                        status="terminal_failure",
                        status_code=None,
                    )
                ],
                succeeded=False,
            )
        ],
        progress_rows=[],
    )
    forged = copy.deepcopy(invalid)
    forged["join_valid"] = True
    for values in forged["defects"].values():
        assert isinstance(values, list)
        values.clear()
    for name in forged["invariants"]:
        forged["invariants"][name] = name != "raw_provider_content_persisted"
    forged.pop("join_receipt_sha256")
    forged["join_receipt_sha256"] = _join_hash(forged)

    with pytest.raises(ValueError, match="semantic|inconsistent|derived"):
        validate_provider_attempt_terminal_join_receipt(forged)

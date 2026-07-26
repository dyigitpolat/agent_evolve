"""Benchmark-neutral terminal joins for physical provider-attempt evidence.

An outbound manifest proves that a request reached the HTTPX pre-transport
boundary.  Queue outcomes and stream progress prove how that physical attempt
terminated.  This module joins those independently durable channels without
retaining prompts, schemas, response bodies, or stream content.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.domain.llm_task_queue import (
    AttemptRequestEvidence,
    AttemptRequestVariant,
    AttemptStatus,
    AttemptTelemetry,
    CancellationReason,
    CanonicalProviderErrorCode,
    ExceptionOriginFamily,
    ExceptionProvenanceLink,
    LLMTaskOutcome,
    RetryAfter,
    RetryAfterSource,
    RetryClassification,
    RetryDisposition,
    RetryReason,
    SanitizedAttemptFailure,
    SanitizedExceptionProvenance,
    SanitizedExceptionProvenanceNode,
    SanitizedValidationIssue,
    StreamTimeoutPhase,
    StructuredOutputFailureMode,
    TaskOutcomeStatus,
    TaskTelemetry,
    ValidationIssueCategory,
    ValidationIssueReasonCode,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    SUPPORTED_STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSIONS,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.ports.structured_generator import (
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)


PROVIDER_ATTEMPT_TERMINAL_JOIN_SCHEMA_VERSION = 1
PROVIDER_ATTEMPT_TERMINAL_JOIN_CONTRACT_ID = (
    "provider_attempt_outbound_outcome_progress_join_v1"
)
PRE_TRANSPORT_FAILURE_EVIDENCE_SCHEMA_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAILURE_STAGE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_JOIN_DOMAIN = b"agent-evolve:provider-attempt-terminal-join:v1\x00"
_PRE_TRANSPORT_DOMAIN = (
    b"agent-evolve:provider-attempt-explicit-pre-transport-failure:v1\x00"
)
_COLLECTION_DOMAIN = b"agent-evolve:provider-attempt-join-collection:v1\x00"
_ATTEMPT_STATUSES = frozenset(
    {
        "succeeded",
        "retryable_failure",
        "terminal_failure",
        "timed_out",
        "cancelled",
    }
)
_TASK_STATUSES = frozenset(
    {"succeeded", "terminal_failure", "attempts_exhausted", "cancelled"}
)
_OUTCOME_RESPONSE_FIELDS = frozenset(
    {
        "requested_model",
        "resolved_model",
        "resolved_provider",
        "provider_response_id",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
        "latency_ns",
    }
)
_OUTCOME_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "status",
        "cancellation_reason",
        "queue_time_ns",
        "service_time_ns",
        "total_time_ns",
        "attempts",
        "response",
    }
)
_OUTCOME_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_number",
        "status",
        "wait_time_ns",
        "service_time_ns",
        "will_retry",
        "policy_backoff_ns",
        "retry_after_ns",
        "scheduled_delay_ns",
        "error_type",
        "request_evidence",
        "classification",
        "failure",
    }
)
_ATTEMPT_REQUEST_EVIDENCE_FIELDS = frozenset(
    {"variant", "prompt_sha256", "provider_attempt_id"}
)
_ATTEMPT_CLASSIFICATION_FIELDS = frozenset({"disposition", "reason"})
_ATTEMPT_FAILURE_FIELDS_V5 = frozenset(
    {
        "kind",
        "retryable",
        "safe_message",
        "status_code",
        "retry_after_seconds",
        "stream_timeout_phase",
        "output_failure_mode",
        "validation_issues",
    }
)
_ATTEMPT_FAILURE_FIELDS_V6 = _ATTEMPT_FAILURE_FIELDS_V5 | frozenset(
    {"provider_error_code", "provider_error_envelope_sha256"}
)
_ATTEMPT_FAILURE_FIELDS_V8 = _ATTEMPT_FAILURE_FIELDS_V6 | frozenset(
    {"exception_provenance"}
)
_EXCEPTION_PROVENANCE_FIELDS_V8 = frozenset({"nodes", "truncated"})
_EXCEPTION_PROVENANCE_NODE_FIELDS_V8 = frozenset(
    {"parent_index", "link", "family", "type_identity_sha256"}
)
_VALIDATION_ISSUE_FIELDS_V6 = frozenset({"category", "location"})
_VALIDATION_ISSUE_FIELDS_V7 = _VALIDATION_ISSUE_FIELDS_V6 | frozenset({"reason_code"})
_PROGRESS_FIELDS = frozenset(
    {
        "schema_version",
        "call_id",
        "provider_attempt_id",
        "sequence",
        "kind",
        "channel",
        "elapsed_ns",
        "event_content_utf8_bytes",
        "cumulative_content_utf8_bytes",
        "rolling_content_sha256",
    }
)
_FRAMEWORK_VERSION_FIELDS = frozenset({"httpx", "openai", "pydantic", "pydantic-ai"})
_EXPECTED_TRANSPORT_SETTING_FIELDS = frozenset(
    {
        "model",
        "provider",
        "reasoning",
        "usage",
        "stream",
        "stream_options",
        "tool_choice",
        "response_format",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "logical_requests",
        "outbound_manifests",
        "terminal_outcomes",
        "progress_rows",
        "explicit_pre_transport_failures",
    }
)
_SOURCE_COUNT_FIELDS = frozenset(
    {*_SOURCE_FIELDS, "outcome_attempts_with_physical_ids"}
)
_PROVIDER_ATTEMPT_ID_FIELDS = frozenset(
    {
        "dispatched",
        "terminal_outcomes",
        "progress",
        "provider_status_response_or_progress",
        "explicit_pre_transport_failures",
    }
)
_DEFECT_ATTEMPT_ID_FIELDS = frozenset(
    {
        "duplicate_outbound_manifest_attempt_ids",
        "duplicate_outcome_attempt_ids",
        "duplicate_pre_transport_attempt_ids",
        "missing_logical_request_attempt_ids",
        "logical_physical_mismatch_attempt_ids",
        "framework_version_mismatch_attempt_ids",
        "transport_settings_mismatch_attempt_ids",
        "missing_manifest_attempt_ids",
        "orphan_manifest_attempt_ids",
        "progress_without_manifest_attempt_ids",
        "progress_without_outcome_attempt_ids",
        "progress_duplicate_sequence_attempt_ids",
        "progress_noncontiguous_sequence_attempt_ids",
        "provider_evidence_without_manifest_attempt_ids",
        "pre_transport_with_manifest_attempt_ids",
        "pre_transport_with_provider_evidence_attempt_ids",
        "pre_transport_without_outcome_attempt_ids",
        "call_id_mismatch_attempt_ids",
    }
)
_DEFECT_CALL_ID_FIELDS = frozenset({"duplicate_logical_request_call_ids"})
_DEFECT_FIELDS = _DEFECT_ATTEMPT_ID_FIELDS | _DEFECT_CALL_ID_FIELDS
_INVARIANT_FIELDS = frozenset(
    {
        "logical_request_exactly_once_per_dispatched_attempt",
        "logical_physical_request_fields_exact",
        "framework_versions_join_qualification_exact",
        "transport_settings_join_selected_profile_exact",
        "manifest_exactly_once_per_dispatched_attempt",
        "outcome_attempt_identity_exactly_once",
        "every_physical_outcome_attempt_accounted_for",
        "every_dispatched_attempt_has_terminal_outcome",
        "provider_evidence_always_has_manifest",
        "progress_joins_manifest_and_outcome",
        "progress_sequence_unique_and_contiguous",
        "explicit_pre_transport_evidence_consistent",
        "explicit_pre_transport_is_affirmative_not_inferred",
        "call_id_join_exact",
        "raw_provider_content_persisted",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _canonical_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        detached = json.loads(_canonical_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain canonical JSON values") from exc
    if type(detached) is not dict:
        raise ValueError(f"{label} must encode one exact object")
    return detached


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def explicit_pre_transport_failure_evidence_record(
    *,
    call_id: str,
    provider_attempt_id: str,
    failure_stage: str,
) -> dict[str, object]:
    """Build content-free affirmative evidence that HTTP dispatch never began."""

    LLMCallId(call_id)
    ProviderAttemptId(provider_attempt_id)
    if (
        type(failure_stage) is not str
        or _FAILURE_STAGE.fullmatch(failure_stage) is None
    ):
        raise ValueError("failure_stage is outside the closed token grammar")
    record: dict[str, object] = {
        "schema_version": PRE_TRANSPORT_FAILURE_EVIDENCE_SCHEMA_VERSION,
        "call_id": call_id,
        "provider_attempt_id": provider_attempt_id,
        "failure_stage": failure_stage,
        "http_transport_dispatch_started": False,
        "raw_failure_content_persisted": False,
    }
    record["evidence_sha256"] = _domain_sha256(_PRE_TRANSPORT_DOMAIN, record)
    return record


def validate_explicit_pre_transport_failure_evidence_record(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = _canonical_mapping(value, label="pre-transport failure evidence")
    if frozenset(record) != {
        "schema_version",
        "call_id",
        "provider_attempt_id",
        "failure_stage",
        "http_transport_dispatch_started",
        "raw_failure_content_persisted",
        "evidence_sha256",
    }:
        raise ValueError("pre-transport failure evidence has unexpected fields")
    if (
        record["schema_version"] != PRE_TRANSPORT_FAILURE_EVIDENCE_SCHEMA_VERSION
        or record["http_transport_dispatch_started"] is not False
        or record["raw_failure_content_persisted"] is not False
    ):
        raise ValueError("pre-transport failure evidence contract drifted")
    LLMCallId(record["call_id"])
    ProviderAttemptId(record["provider_attempt_id"])
    stage = record["failure_stage"]
    if type(stage) is not str or _FAILURE_STAGE.fullmatch(stage) is None:
        raise ValueError("pre-transport failure stage is invalid")
    supplied = _require_sha256(record["evidence_sha256"], label="evidence_sha256")
    authenticated = dict(record)
    del authenticated["evidence_sha256"]
    if supplied != _domain_sha256(_PRE_TRANSPORT_DOMAIN, authenticated):
        raise ValueError("pre-transport failure evidence hash is invalid")
    return record


def _collection_sha256(rows: Sequence[object]) -> str:
    digest = hashlib.sha256(_COLLECTION_DOMAIN)
    for index, row in enumerate(rows):
        try:
            payload = _canonical_bytes(row)
        except (TypeError, ValueError):
            payload = f"noncanonical-row:{index}".encode("ascii")
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_outcome_response(value: object) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != _OUTCOME_RESPONSE_FIELDS:
        raise ValueError("provider outcome response has unexpected fields")
    for name in ("requested_model", "resolved_model", "resolved_provider"):
        if type(value[name]) is not str or not value[name].strip():
            raise ValueError(f"provider outcome response {name} is invalid")
    for name in ("provider_response_id", "finish_reason"):
        item = value[name]
        if item is not None and (type(item) is not str or not item.strip()):
            raise ValueError(f"provider outcome response {name} is invalid")
    for name in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "latency_ns",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError(f"provider outcome response {name} is invalid")
    cost = value["cost_usd"]
    if cost is not None:
        if type(cost) is not str:
            raise ValueError("provider outcome response cost_usd is invalid")
        try:
            parsed = Decimal(cost)
        except InvalidOperation as exc:
            raise ValueError("provider outcome response cost_usd is invalid") from exc
        if not parsed.is_finite() or parsed < 0 or str(parsed) != cost:
            raise ValueError("provider outcome response cost_usd is not canonical")
    return value


def _validate_attempt_request_evidence(
    value: object,
) -> AttemptRequestEvidence | None:
    if value is None:
        return None
    if type(value) is not dict or frozenset(value) != _ATTEMPT_REQUEST_EVIDENCE_FIELDS:
        raise ValueError("provider attempt request evidence has unexpected fields")
    variant = value["variant"]
    prompt_sha256 = value["prompt_sha256"]
    attempt_id = value["provider_attempt_id"]
    if type(variant) is not str or type(prompt_sha256) is not str:
        raise ValueError("provider attempt request evidence is invalid")
    physical_id = None
    if attempt_id is not None:
        if type(attempt_id) is not str:
            raise ValueError("provider attempt physical identity is invalid")
        physical_id = ProviderAttemptId(attempt_id)
    return AttemptRequestEvidence(
        variant=AttemptRequestVariant(variant),
        prompt_sha256=prompt_sha256,
        provider_attempt_id=physical_id,
    )


def _validate_attempt_failure(
    value: object,
    *,
    schema_version: int,
) -> SanitizedAttemptFailure | None:
    if value is None:
        return None
    expected_fields = (
        _ATTEMPT_FAILURE_FIELDS_V8
        if schema_version >= 8
        else (
            _ATTEMPT_FAILURE_FIELDS_V6
            if schema_version >= 6
            else _ATTEMPT_FAILURE_FIELDS_V5
        )
    )
    if type(value) is not dict or frozenset(value) != expected_fields:
        raise ValueError("provider attempt failure has unexpected fields")

    issues_value = value["validation_issues"]
    if type(issues_value) is not list:
        raise ValueError("provider attempt validation issues must be a list")
    issues: list[SanitizedValidationIssue] = []
    for issue in issues_value:
        expected_issue_fields = (
            _VALIDATION_ISSUE_FIELDS_V7
            if schema_version >= 7
            else _VALIDATION_ISSUE_FIELDS_V6
        )
        if type(issue) is not dict or frozenset(issue) != expected_issue_fields:
            raise ValueError("provider attempt validation issue has unexpected fields")
        category = issue["category"]
        location = issue["location"]
        reason_code = issue.get("reason_code")
        if (
            type(category) is not str
            or type(location) is not list
            or (reason_code is not None and type(reason_code) is not str)
        ):
            raise ValueError("provider attempt validation issue is invalid")
        issues.append(
            SanitizedValidationIssue(
                category=ValidationIssueCategory(category),
                location=tuple(location),
                reason_code=(
                    None
                    if reason_code is None
                    else ValidationIssueReasonCode(reason_code)
                ),
            )
        )

    output_mode = value["output_failure_mode"]
    timeout_phase = value["stream_timeout_phase"]
    provider_error_code = value["provider_error_code"] if schema_version >= 6 else None
    exception_provenance_value = (
        value["exception_provenance"] if schema_version >= 8 else None
    )
    for name, item in (
        ("output_failure_mode", output_mode),
        ("stream_timeout_phase", timeout_phase),
        ("provider_error_code", provider_error_code),
    ):
        if item is not None and type(item) is not str:
            raise ValueError(f"provider attempt failure {name} is invalid")
    exception_provenance: SanitizedExceptionProvenance | None = None
    if exception_provenance_value is not None:
        if (
            type(exception_provenance_value) is not dict
            or frozenset(exception_provenance_value)
            != _EXCEPTION_PROVENANCE_FIELDS_V8
            or type(exception_provenance_value["nodes"]) is not list
            or type(exception_provenance_value["truncated"]) is not bool
        ):
            raise ValueError("provider attempt exception provenance is invalid")
        provenance_nodes: list[SanitizedExceptionProvenanceNode] = []
        for raw_node in exception_provenance_value["nodes"]:
            if (
                type(raw_node) is not dict
                or frozenset(raw_node) != _EXCEPTION_PROVENANCE_NODE_FIELDS_V8
            ):
                raise ValueError(
                    "provider attempt exception provenance node is invalid"
                )
            parent_index = raw_node["parent_index"]
            link = raw_node["link"]
            family = raw_node["family"]
            fingerprint = raw_node["type_identity_sha256"]
            if (
                (parent_index is not None and type(parent_index) is not int)
                or type(link) is not str
                or type(family) is not str
                or type(fingerprint) is not str
            ):
                raise ValueError(
                    "provider attempt exception provenance node fields are invalid"
                )
            provenance_nodes.append(
                SanitizedExceptionProvenanceNode(
                    parent_index=parent_index,
                    link=ExceptionProvenanceLink(link),
                    family=ExceptionOriginFamily(family),
                    type_identity_sha256=fingerprint,
                )
            )
        exception_provenance = SanitizedExceptionProvenance(
            nodes=tuple(provenance_nodes),
            truncated=exception_provenance_value["truncated"],
        )
    return SanitizedAttemptFailure(
        kind=value["kind"],
        retryable=value["retryable"],
        safe_message=value["safe_message"],
        status_code=value["status_code"],
        retry_after_seconds=value["retry_after_seconds"],
        output_failure_mode=(
            None if output_mode is None else StructuredOutputFailureMode(output_mode)
        ),
        validation_issues=tuple(issues),
        stream_timeout_phase=(
            None if timeout_phase is None else StreamTimeoutPhase(timeout_phase)
        ),
        provider_error_code=(
            None
            if provider_error_code is None
            else CanonicalProviderErrorCode(provider_error_code)
        ),
        provider_error_envelope_sha256=(
            value["provider_error_envelope_sha256"] if schema_version >= 6 else None
        ),
        exception_provenance=exception_provenance,
    )


def _validate_outcome_attempt(
    value: object,
    *,
    schema_version: int,
) -> tuple[AttemptTelemetry, str | None, bool]:
    if type(value) is not dict or frozenset(value) != _OUTCOME_ATTEMPT_FIELDS:
        raise ValueError("provider outcome attempt has unexpected fields")
    status_value = value["status"]
    if type(status_value) is not str or status_value not in _ATTEMPT_STATUSES:
        raise ValueError("provider attempt status is invalid")
    evidence = _validate_attempt_request_evidence(value["request_evidence"])
    failure = _validate_attempt_failure(
        value["failure"],
        schema_version=schema_version,
    )
    classification_value = value["classification"]
    if classification_value is None:
        if failure is not None:
            raise ValueError("provider attempt failure lacks its classification")
        classification = None
    else:
        if (
            type(classification_value) is not dict
            or frozenset(classification_value) != _ATTEMPT_CLASSIFICATION_FIELDS
        ):
            raise ValueError("provider attempt classification has unexpected fields")
        disposition = classification_value["disposition"]
        reason = classification_value["reason"]
        if type(disposition) is not str or type(reason) is not str:
            raise ValueError("provider attempt classification is invalid")
        retry_after_ns = value["retry_after_ns"]
        if type(retry_after_ns) is not int or retry_after_ns < 0:
            raise ValueError("provider attempt retry_after_ns is invalid")
        classification = RetryClassification(
            disposition=RetryDisposition(disposition),
            reason=RetryReason(reason),
            retry_after=(
                None
                if retry_after_ns == 0
                else RetryAfter(
                    delay_ns=retry_after_ns,
                    # The durable projection intentionally omits the header's
                    # parsing source; either source has identical queue semantics.
                    source=RetryAfterSource.DELAY_SECONDS,
                )
            ),
            sanitized_failure=failure,
        )

    attempt = AttemptTelemetry(
        attempt_number=value["attempt_number"],
        status=AttemptStatus(status_value),
        wait_time_ns=value["wait_time_ns"],
        service_time_ns=value["service_time_ns"],
        will_retry=value["will_retry"],
        policy_backoff_ns=value["policy_backoff_ns"],
        retry_after_ns=value["retry_after_ns"],
        scheduled_delay_ns=value["scheduled_delay_ns"],
        classification=classification,
        error_type=value["error_type"],
        request_evidence=evidence,
    )
    if evidence is None or evidence.provider_attempt_id is None:
        if attempt.status is not AttemptStatus.CANCELLED:
            raise ValueError("non-cancelled attempt lacks physical request evidence")
        return attempt, None, False
    has_provider_evidence = failure is not None and failure.status_code is not None
    return attempt, evidence.provider_attempt_id.value, has_provider_evidence


def _outcome_projection(
    row: object,
) -> tuple[str, list[tuple[str, str, str, bool, str, str]]]:
    value = _canonical_mapping(row, label="provider outcome")
    if frozenset(value) != _OUTCOME_FIELDS:
        raise ValueError("provider outcome has unexpected fields")
    schema_version = value["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version not in SUPPORTED_STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSIONS
    ):
        raise ValueError("provider outcome schema version is unsupported")
    call_id = value["task_id"]
    if type(call_id) is not str:
        raise ValueError("provider outcome lacks task_id")
    LLMCallId(call_id)
    status = value["status"]
    cancellation_reason = value["cancellation_reason"]
    attempts = value["attempts"]
    response = value["response"]
    if type(status) is not str or status not in _TASK_STATUSES:
        raise ValueError("provider outcome status is invalid")
    if type(attempts) is not list:
        raise ValueError("provider outcome attempts/response shape is invalid")
    if response is not None:
        _validate_outcome_response(response)
    if status == "succeeded" and response is None:
        raise ValueError("successful provider outcome lacks its response")

    projected: list[tuple[str, str, str, bool, str, str]] = []
    telemetry_attempts: list[AttemptTelemetry] = []
    for raw_attempt in attempts:
        attempt, attempt_id, has_provider_evidence = _validate_outcome_attempt(
            raw_attempt,
            schema_version=schema_version,
        )
        telemetry_attempts.append(attempt)
        if attempt_id is not None:
            if attempt.request_evidence is None:
                raise ValueError("physical attempt lacks request evidence")
            projected.append(
                (
                    attempt_id,
                    call_id,
                    attempt.status.value,
                    has_provider_evidence or attempt.status is AttemptStatus.SUCCEEDED,
                    attempt.request_evidence.prompt_sha256,
                    attempt.request_evidence.variant.value,
                )
            )

    telemetry = TaskTelemetry(
        task_id=call_id,
        queue_time_ns=value["queue_time_ns"],
        service_time_ns=value["service_time_ns"],
        total_time_ns=value["total_time_ns"],
        attempts=tuple(telemetry_attempts),
    )
    status_enum = TaskOutcomeStatus(status)
    if cancellation_reason is not None and type(cancellation_reason) is not str:
        raise ValueError("provider outcome cancellation reason is invalid")
    LLMTaskOutcome(
        status=status_enum,
        telemetry=telemetry,
        response=response,
        cancellation_reason=(
            None
            if cancellation_reason is None
            else CancellationReason(cancellation_reason)
        ),
    )
    succeeded = sum(
        attempt.status is AttemptStatus.SUCCEEDED for attempt in telemetry_attempts
    )
    if status_enum is TaskOutcomeStatus.SUCCEEDED and succeeded != 1:
        raise ValueError("provider response does not join one succeeded attempt")
    if status_enum is not TaskOutcomeStatus.SUCCEEDED and succeeded:
        raise ValueError("non-successful provider outcome has a succeeded attempt")
    return call_id, projected


def validate_structured_generation_outcome_record(
    row: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate and detach one durable queue-outcome projection.

    This is the public, content-free decoder for records produced by
    :func:`structured_generation_outcome_record`.  It deliberately returns the
    canonical JSON projection rather than reconstructing a provider response:
    typed response content lives in the separately authenticated structured
    output-evidence channel.
    """

    value = _canonical_mapping(row, label="provider outcome")
    _outcome_projection(value)
    return value


def _progress_projection(row: object) -> tuple[str, str, int]:
    value = _canonical_mapping(row, label="provider progress")
    if frozenset(value) != _PROGRESS_FIELDS:
        raise ValueError("provider progress has unexpected fields")
    if value["schema_version"] != 1:
        raise ValueError("provider progress schema version is unsupported")
    call_id = value["call_id"]
    attempt_id = value["provider_attempt_id"]
    sequence = value["sequence"]
    if type(call_id) is not str or type(attempt_id) is not str:
        raise ValueError("provider progress lacks call/attempt identity")
    kind = value["kind"]
    channel = value["channel"]
    if type(kind) is not str or type(channel) is not str:
        raise ValueError("provider progress kind/channel is invalid")
    progress = StructuredStreamProgress(
        call_id=call_id,
        provider_attempt_id=attempt_id,
        sequence=sequence,
        kind=StructuredStreamProgressKind(kind),
        channel=StructuredStreamChannel(channel),
        elapsed_ns=value["elapsed_ns"],
        event_content_utf8_bytes=value["event_content_utf8_bytes"],
        cumulative_content_utf8_bytes=value["cumulative_content_utf8_bytes"],
        rolling_content_sha256=value["rolling_content_sha256"],
    )
    progress.__post_init__()
    return attempt_id, call_id, sequence


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count != 1)


def _logical_request_projection(row: object) -> dict[str, object]:
    value = validate_structured_generation_request_evidence_record(
        _canonical_mapping(row, label="logical request evidence")
    )
    return {
        "call_id": value["call_id"],
        "operation": value["operation"],
        "prompt_sha256": value["wire_prompt_sha256"],
        "prompt_utf8_bytes": value["prompt_utf8_bytes"],
        "output_tool_name": value["output_tool_name"],
        "logical_output_schema_sha256": value["output_schema_sha256"],
        "logical_output_schema_utf8_bytes": value["output_schema_utf8_bytes"],
        "max_completion_tokens": value["max_output_tokens"],
        "requested_temperature_hex": value["temperature_hex"],
        "request_evidence_sha256": value["request_evidence_sha256"],
    }


def _manifest_logical_projection(value: Mapping[str, object]) -> dict[str, object]:
    message = value["message"]
    tool = value["tool"]
    request_contract = value["request_contract"]
    settings = value["settings"]
    assert isinstance(message, Mapping)
    assert isinstance(tool, Mapping)
    assert isinstance(request_contract, Mapping)
    assert isinstance(settings, Mapping)
    return {
        "call_id": value["call_id"],
        "operation": value["operation"],
        "prompt_sha256": message["content_sha256"],
        "prompt_utf8_bytes": message["content_utf8_bytes"],
        "output_tool_name": tool["name"],
        "logical_output_schema_sha256": request_contract[
            "logical_output_schema_sha256"
        ],
        "logical_output_schema_utf8_bytes": request_contract[
            "logical_output_schema_utf8_bytes"
        ],
        "max_completion_tokens": settings["max_completion_tokens"],
        "requested_temperature_hex": request_contract["requested_temperature_hex"],
    }


def _validated_expected_framework_versions(
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    record = _canonical_mapping(value, label="expected framework versions")
    if frozenset(record) != _FRAMEWORK_VERSION_FIELDS or any(
        type(item) is not str or not item or item != item.strip()
        for item in record.values()
    ):
        raise ValueError("expected framework versions violate the closed schema")
    return record


def _validated_expected_transport_settings(
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    record = _canonical_mapping(value, label="expected transport settings")
    if frozenset(record) != _EXPECTED_TRANSPORT_SETTING_FIELDS:
        raise ValueError("expected transport settings violate the closed schema")
    return record


def build_provider_attempt_terminal_join_receipt(
    *,
    logical_requests: Sequence[Mapping[str, object]] = (),
    outbound_manifests: Sequence[Mapping[str, object]],
    terminal_outcomes: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    explicit_pre_transport_failures: Sequence[Mapping[str, object]] = (),
    expected_framework_versions: Mapping[str, object] | None = None,
    expected_transport_settings: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a redacted terminal join; mismatches yield ``join_valid=False``.

    Every queue attempt carrying a physical ID requires one outbound manifest,
    unless a separately authenticated record affirmatively proves it failed
    before HTTP dispatch. Missing manifests are never treated as such proof.
    """

    collections = (
        logical_requests,
        outbound_manifests,
        terminal_outcomes,
        progress_rows,
        explicit_pre_transport_failures,
    )
    if any(not isinstance(value, Sequence) for value in collections):
        raise TypeError("provider-attempt join inputs must be sequences")

    expected_frameworks = _validated_expected_framework_versions(
        expected_framework_versions
    )
    expected_settings = _validated_expected_transport_settings(
        expected_transport_settings
    )

    malformed_logical_requests: list[int] = []
    malformed_manifests: list[int] = []
    malformed_outcomes: list[int] = []
    malformed_progress: list[int] = []
    malformed_pretransport: list[int] = []
    logical_entries: list[dict[str, object]] = []
    manifest_entries: list[dict[str, object]] = []
    outcome_entries: list[tuple[str, str, str, bool, str, str]] = []
    progress_entries: list[tuple[str, str, int]] = []
    pretransport_entries: list[tuple[str, str, str]] = []

    for index, row in enumerate(logical_requests):
        try:
            logical_entries.append(_logical_request_projection(row))
        except (TypeError, ValueError):
            malformed_logical_requests.append(index)
    for index, row in enumerate(outbound_manifests):
        try:
            manifest = validate_openrouter_outbound_request_manifest_record(row)
            manifest_entries.append(manifest)
        except (TypeError, ValueError):
            malformed_manifests.append(index)
    for index, row in enumerate(terminal_outcomes):
        try:
            _, attempts = _outcome_projection(row)
            outcome_entries.extend(attempts)
        except (TypeError, ValueError):
            malformed_outcomes.append(index)
    for index, row in enumerate(progress_rows):
        try:
            progress_entries.append(_progress_projection(row))
        except (TypeError, ValueError):
            malformed_progress.append(index)
    for index, row in enumerate(explicit_pre_transport_failures):
        try:
            evidence = validate_explicit_pre_transport_failure_evidence_record(row)
            pretransport_entries.append(
                (
                    evidence["provider_attempt_id"],
                    evidence["call_id"],
                    evidence["evidence_sha256"],
                )
            )
        except (TypeError, ValueError):
            malformed_pretransport.append(index)

    logical_call_ids = [str(value["call_id"]) for value in logical_entries]
    manifest_ids = [str(value["provider_attempt_id"]) for value in manifest_entries]
    outcome_ids = [value[0] for value in outcome_entries]
    progress_ids = [value[0] for value in progress_entries]
    pretransport_ids = [value[0] for value in pretransport_entries]
    manifest_set = set(manifest_ids)
    outcome_set = set(outcome_ids)
    progress_set = set(progress_ids)
    pretransport_set = set(pretransport_ids)
    provider_evidence_set = {
        attempt_id
        for attempt_id, _, _, has_evidence, _, _ in outcome_entries
        if has_evidence
    } | progress_set
    affirmative_pretransport_set = pretransport_set - provider_evidence_set
    required_manifest_set = outcome_set - affirmative_pretransport_set

    logical_by_call: dict[str, list[dict[str, object]]] = defaultdict(list)
    for value in logical_entries:
        logical_by_call[str(value["call_id"])].append(value)
    attempt_request_by_id = {
        attempt_id: (prompt_sha256, variant)
        for attempt_id, _, _, _, prompt_sha256, variant in outcome_entries
    }
    missing_logical_requests: list[str] = []
    logical_physical_mismatches: list[str] = []
    framework_version_mismatches: list[str] = []
    transport_settings_mismatches: list[str] = []
    for manifest in manifest_entries:
        attempt_id = str(manifest["provider_attempt_id"])
        matching = logical_by_call.get(str(manifest["call_id"]), [])
        if not matching:
            missing_logical_requests.append(attempt_id)
        elif len(matching) == 1:
            observed = _manifest_logical_projection(manifest)
            expected = {
                name: item
                for name, item in matching[0].items()
                if name != "request_evidence_sha256"
            }
            attempt_request = attempt_request_by_id.get(attempt_id)
            if attempt_request is None:
                logical_physical_mismatches.append(attempt_id)
            else:
                attempt_prompt_sha256, variant = attempt_request
                if variant == AttemptRequestVariant.ORIGINAL.value:
                    mismatch = (
                        attempt_prompt_sha256 != expected["prompt_sha256"]
                        or observed != expected
                    )
                else:
                    # Schema-repair variants preserve every logical request
                    # field except the provider-facing prompt, whose exact
                    # digest is committed by the per-attempt queue evidence.
                    expected["prompt_sha256"] = attempt_prompt_sha256
                    observed.pop("prompt_utf8_bytes")
                    expected.pop("prompt_utf8_bytes")
                    mismatch = observed != expected
                if mismatch:
                    logical_physical_mismatches.append(attempt_id)
        if (
            expected_frameworks is None
            or manifest["framework_versions"] != expected_frameworks
        ):
            framework_version_mismatches.append(attempt_id)
        settings = manifest["settings"]
        if (
            expected_settings is None
            or type(settings) is not dict
            or any(
                settings[name] != expected_settings[name] for name in expected_settings
            )
        ):
            transport_settings_mismatches.append(attempt_id)

    call_ids_by_attempt: dict[str, set[str]] = defaultdict(set)
    for manifest in manifest_entries:
        call_ids_by_attempt[str(manifest["provider_attempt_id"])].add(
            str(manifest["call_id"])
        )
    for attempt_id, call_id, _, _, _, _ in outcome_entries:
        call_ids_by_attempt[attempt_id].add(call_id)
    for attempt_id, call_id, _ in progress_entries:
        call_ids_by_attempt[attempt_id].add(call_id)
    for attempt_id, call_id, _ in pretransport_entries:
        call_ids_by_attempt[attempt_id].add(call_id)

    progress_sequences: dict[str, list[int]] = defaultdict(list)
    for attempt_id, _, sequence in progress_entries:
        progress_sequences[attempt_id].append(sequence)
    canonical_progress_sequences = {
        attempt_id: sorted(sequences)
        for attempt_id, sequences in sorted(progress_sequences.items())
    }
    duplicate_progress_sequences = sorted(
        attempt_id
        for attempt_id, sequences in canonical_progress_sequences.items()
        if len(sequences) != len(set(sequences))
    )
    noncontiguous_progress_sequences = sorted(
        attempt_id
        for attempt_id, sequences in canonical_progress_sequences.items()
        if sequences != list(range(1, len(sequences) + 1))
    )

    shared_ids = manifest_set | outcome_set | progress_set | pretransport_set
    call_mismatches = sorted(
        attempt_id
        for attempt_id in shared_ids
        if len(call_ids_by_attempt[attempt_id]) > 1
    )

    duplicate_logical_call_ids = _duplicates(logical_call_ids)
    duplicate_manifest_ids = _duplicates(manifest_ids)
    duplicate_outcome_ids = _duplicates(outcome_ids)
    duplicate_pretransport_ids = _duplicates(pretransport_ids)
    missing_manifests = sorted(required_manifest_set - manifest_set)
    orphan_manifests = sorted(manifest_set - outcome_set)
    progress_without_manifest = sorted(progress_set - manifest_set)
    progress_without_outcome = sorted(progress_set - outcome_set)
    provider_evidence_without_manifest = sorted(provider_evidence_set - manifest_set)
    pretransport_with_manifest = sorted(pretransport_set & manifest_set)
    pretransport_with_provider_evidence = sorted(
        pretransport_set & provider_evidence_set
    )
    pretransport_without_outcome = sorted(pretransport_set - outcome_set)

    defects = (
        malformed_logical_requests,
        malformed_manifests,
        malformed_outcomes,
        malformed_progress,
        malformed_pretransport,
        duplicate_logical_call_ids,
        duplicate_manifest_ids,
        duplicate_outcome_ids,
        duplicate_pretransport_ids,
        missing_logical_requests,
        logical_physical_mismatches,
        framework_version_mismatches,
        transport_settings_mismatches,
        missing_manifests,
        orphan_manifests,
        progress_without_manifest,
        progress_without_outcome,
        duplicate_progress_sequences,
        noncontiguous_progress_sequences,
        provider_evidence_without_manifest,
        pretransport_with_manifest,
        pretransport_with_provider_evidence,
        pretransport_without_outcome,
        call_mismatches,
    )
    join_valid = not any(defects)
    record: dict[str, object] = {
        "schema_version": PROVIDER_ATTEMPT_TERMINAL_JOIN_SCHEMA_VERSION,
        "contract_id": PROVIDER_ATTEMPT_TERMINAL_JOIN_CONTRACT_ID,
        "source_counts": {
            "logical_requests": len(logical_requests),
            "outbound_manifests": len(outbound_manifests),
            "terminal_outcomes": len(terminal_outcomes),
            "outcome_attempts_with_physical_ids": len(outcome_entries),
            "progress_rows": len(progress_rows),
            "explicit_pre_transport_failures": len(explicit_pre_transport_failures),
        },
        "source_sha256": {
            "logical_requests": _collection_sha256(logical_requests),
            "outbound_manifests": _collection_sha256(outbound_manifests),
            "terminal_outcomes": _collection_sha256(terminal_outcomes),
            "progress_rows": _collection_sha256(progress_rows),
            "explicit_pre_transport_failures": _collection_sha256(
                explicit_pre_transport_failures
            ),
        },
        "expected_framework_versions": expected_frameworks,
        "expected_transport_settings": expected_settings,
        "logical_request_call_ids": sorted(set(logical_call_ids)),
        "provider_attempt_ids": {
            "dispatched": sorted(manifest_set),
            "terminal_outcomes": sorted(outcome_set),
            "progress": sorted(progress_set),
            "provider_status_response_or_progress": sorted(provider_evidence_set),
            "explicit_pre_transport_failures": sorted(pretransport_set),
        },
        "call_ids_by_provider_attempt": {
            attempt_id: sorted(call_ids_by_attempt[attempt_id])
            for attempt_id in sorted(shared_ids)
        },
        "progress_sequences_by_provider_attempt": canonical_progress_sequences,
        "malformed_row_indices": {
            "logical_requests": malformed_logical_requests,
            "outbound_manifests": malformed_manifests,
            "terminal_outcomes": malformed_outcomes,
            "progress_rows": malformed_progress,
            "explicit_pre_transport_failures": malformed_pretransport,
        },
        "defects": {
            "duplicate_logical_request_call_ids": duplicate_logical_call_ids,
            "duplicate_outbound_manifest_attempt_ids": duplicate_manifest_ids,
            "duplicate_outcome_attempt_ids": duplicate_outcome_ids,
            "duplicate_pre_transport_attempt_ids": duplicate_pretransport_ids,
            "missing_logical_request_attempt_ids": sorted(
                set(missing_logical_requests)
            ),
            "logical_physical_mismatch_attempt_ids": sorted(
                set(logical_physical_mismatches)
            ),
            "framework_version_mismatch_attempt_ids": sorted(
                set(framework_version_mismatches)
            ),
            "transport_settings_mismatch_attempt_ids": sorted(
                set(transport_settings_mismatches)
            ),
            "missing_manifest_attempt_ids": missing_manifests,
            "orphan_manifest_attempt_ids": orphan_manifests,
            "progress_without_manifest_attempt_ids": progress_without_manifest,
            "progress_without_outcome_attempt_ids": progress_without_outcome,
            "progress_duplicate_sequence_attempt_ids": (duplicate_progress_sequences),
            "progress_noncontiguous_sequence_attempt_ids": (
                noncontiguous_progress_sequences
            ),
            "provider_evidence_without_manifest_attempt_ids": (
                provider_evidence_without_manifest
            ),
            "pre_transport_with_manifest_attempt_ids": pretransport_with_manifest,
            "pre_transport_with_provider_evidence_attempt_ids": (
                pretransport_with_provider_evidence
            ),
            "pre_transport_without_outcome_attempt_ids": (pretransport_without_outcome),
            "call_id_mismatch_attempt_ids": call_mismatches,
        },
        "invariants": {
            "logical_request_exactly_once_per_dispatched_attempt": (
                not duplicate_logical_call_ids and not missing_logical_requests
            ),
            "logical_physical_request_fields_exact": (not logical_physical_mismatches),
            "framework_versions_join_qualification_exact": (
                not framework_version_mismatches
            ),
            "transport_settings_join_selected_profile_exact": (
                not transport_settings_mismatches
            ),
            "manifest_exactly_once_per_dispatched_attempt": (
                not duplicate_manifest_ids
            ),
            "outcome_attempt_identity_exactly_once": not duplicate_outcome_ids,
            "every_physical_outcome_attempt_accounted_for": not missing_manifests,
            "every_dispatched_attempt_has_terminal_outcome": not orphan_manifests,
            "provider_evidence_always_has_manifest": (
                not provider_evidence_without_manifest
            ),
            "progress_joins_manifest_and_outcome": (
                not progress_without_manifest and not progress_without_outcome
            ),
            "progress_sequence_unique_and_contiguous": (
                not duplicate_progress_sequences
                and not noncontiguous_progress_sequences
            ),
            "explicit_pre_transport_evidence_consistent": (
                not duplicate_pretransport_ids
                and not pretransport_with_manifest
                and not pretransport_with_provider_evidence
                and not pretransport_without_outcome
            ),
            "explicit_pre_transport_is_affirmative_not_inferred": True,
            "call_id_join_exact": not call_mismatches,
            # Accepted source rows have exact, content-safe durable schemas.
            # Rejected rows contribute only an index and a collection digest;
            # no source value is copied into this receipt projection.
            "raw_provider_content_persisted": False,
        },
        "join_valid": join_valid,
    }
    record["join_receipt_sha256"] = _domain_sha256(_JOIN_DOMAIN, record)
    return validate_provider_attempt_terminal_join_receipt(record)


def validate_provider_attempt_terminal_join_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = _canonical_mapping(value, label="provider-attempt terminal join")
    if frozenset(record) != {
        "schema_version",
        "contract_id",
        "source_counts",
        "source_sha256",
        "expected_framework_versions",
        "expected_transport_settings",
        "logical_request_call_ids",
        "provider_attempt_ids",
        "call_ids_by_provider_attempt",
        "progress_sequences_by_provider_attempt",
        "malformed_row_indices",
        "defects",
        "invariants",
        "join_valid",
        "join_receipt_sha256",
    }:
        raise ValueError("provider-attempt terminal join has unexpected fields")
    if (
        record["schema_version"] != PROVIDER_ATTEMPT_TERMINAL_JOIN_SCHEMA_VERSION
        or record["contract_id"] != PROVIDER_ATTEMPT_TERMINAL_JOIN_CONTRACT_ID
        or type(record["join_valid"]) is not bool
    ):
        raise ValueError("provider-attempt terminal join contract drifted")
    for name in (
        "source_counts",
        "source_sha256",
        "provider_attempt_ids",
        "call_ids_by_provider_attempt",
        "progress_sequences_by_provider_attempt",
        "malformed_row_indices",
        "defects",
        "invariants",
    ):
        if type(record[name]) is not dict:
            raise ValueError(f"{name} must be an exact object")
    source_counts = record["source_counts"]
    source_sha256 = record["source_sha256"]
    provider_attempt_ids = record["provider_attempt_ids"]
    malformed = record["malformed_row_indices"]
    defects = record["defects"]
    invariants = record["invariants"]
    call_ids_by_attempt = record["call_ids_by_provider_attempt"]
    progress_sequences = record["progress_sequences_by_provider_attempt"]
    assert type(source_counts) is dict
    assert type(source_sha256) is dict
    assert type(provider_attempt_ids) is dict
    assert type(malformed) is dict
    assert type(defects) is dict
    assert type(invariants) is dict
    assert type(call_ids_by_attempt) is dict
    assert type(progress_sequences) is dict

    if frozenset(source_counts) != _SOURCE_COUNT_FIELDS or any(
        type(item) is not int or item < 0 for item in source_counts.values()
    ):
        raise ValueError("provider-attempt join source counts are invalid")
    if frozenset(source_sha256) != _SOURCE_FIELDS:
        raise ValueError("provider-attempt join source hashes are incomplete")
    for name, digest in source_sha256.items():
        _require_sha256(digest, label=f"source_sha256.{name}")

    expected_frameworks = _validated_expected_framework_versions(
        record["expected_framework_versions"]
    )
    expected_settings = _validated_expected_transport_settings(
        record["expected_transport_settings"]
    )
    if expected_frameworks != record["expected_framework_versions"]:
        raise ValueError("expected framework versions are not canonical")
    if expected_settings != record["expected_transport_settings"]:
        raise ValueError("expected transport settings are not canonical")

    def exact_identity_list(
        item: object,
        *,
        label: str,
        constructor: object,
        unique: bool = True,
    ) -> list[str]:
        if type(item) is not list or any(type(entry) is not str for entry in item):
            raise ValueError(f"{label} must be a list of exact identifiers")
        values = list(item)
        if values != sorted(values) or (unique and len(values) != len(set(values))):
            raise ValueError(f"{label} must be canonical sorted identifiers")
        for entry in values:
            try:
                constructor(entry)  # type: ignore[operator]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} contains an invalid identifier") from exc
        return values

    logical_call_ids = exact_identity_list(
        record["logical_request_call_ids"],
        label="logical_request_call_ids",
        constructor=LLMCallId,
    )
    if frozenset(provider_attempt_ids) != _PROVIDER_ATTEMPT_ID_FIELDS:
        raise ValueError("provider_attempt_ids violates the closed schema")
    attempt_sets: dict[str, set[str]] = {}
    for name in sorted(_PROVIDER_ATTEMPT_ID_FIELDS):
        attempt_sets[name] = set(
            exact_identity_list(
                provider_attempt_ids[name],
                label=f"provider_attempt_ids.{name}",
                constructor=ProviderAttemptId,
            )
        )

    shared_ids = set().union(*attempt_sets.values())
    if set(call_ids_by_attempt) != shared_ids:
        raise ValueError("call ID summary does not cover exact attempt identities")
    normalized_call_summary: dict[str, list[str]] = {}
    for attempt_id, call_ids in call_ids_by_attempt.items():
        ProviderAttemptId(attempt_id)
        normalized_call_summary[attempt_id] = exact_identity_list(
            call_ids,
            label=f"call_ids_by_provider_attempt.{attempt_id}",
            constructor=LLMCallId,
        )
        if not normalized_call_summary[attempt_id]:
            raise ValueError("attempt call ID summary cannot be empty")

    progress_set = attempt_sets["progress"]
    if set(progress_sequences) != progress_set:
        raise ValueError("progress sequence summary does not cover progress attempts")
    normalized_sequences: dict[str, list[int]] = {}
    for attempt_id, sequences in progress_sequences.items():
        ProviderAttemptId(attempt_id)
        if (
            type(sequences) is not list
            or not sequences
            or any(type(sequence) is not int or sequence < 1 for sequence in sequences)
            or sequences != sorted(sequences)
        ):
            raise ValueError("progress sequence summary is invalid")
        normalized_sequences[attempt_id] = list(sequences)

    if frozenset(malformed) != _SOURCE_FIELDS:
        raise ValueError("malformed row indices violate the closed schema")
    for name in sorted(_SOURCE_FIELDS):
        indices = malformed[name]
        if (
            type(indices) is not list
            or any(type(index) is not int or index < 0 for index in indices)
            or indices != sorted(set(indices))
            or any(index >= source_counts[name] for index in indices)
        ):
            raise ValueError(f"malformed_row_indices.{name} is invalid")

    if frozenset(defects) != _DEFECT_FIELDS:
        raise ValueError("provider-attempt join defects violate the closed schema")
    normalized_defects: dict[str, list[str]] = {}
    for name in sorted(_DEFECT_ATTEMPT_ID_FIELDS):
        normalized_defects[name] = exact_identity_list(
            defects[name],
            label=f"defects.{name}",
            constructor=ProviderAttemptId,
        )
    for name in sorted(_DEFECT_CALL_ID_FIELDS):
        normalized_defects[name] = exact_identity_list(
            defects[name],
            label=f"defects.{name}",
            constructor=LLMCallId,
        )

    if frozenset(invariants) != _INVARIANT_FIELDS or any(
        type(item) is not bool for item in invariants.values()
    ):
        raise ValueError("provider-attempt join invariants violate the closed schema")

    dispatched = attempt_sets["dispatched"]
    outcomes = attempt_sets["terminal_outcomes"]
    provider_evidence = attempt_sets["provider_status_response_or_progress"]
    pretransport = attempt_sets["explicit_pre_transport_failures"]
    if not progress_set.issubset(provider_evidence) or not provider_evidence.issubset(
        outcomes | progress_set
    ):
        raise ValueError("provider evidence identity summary is inconsistent")
    if any(
        value not in dispatched
        for value in normalized_defects["missing_logical_request_attempt_ids"]
        + normalized_defects["logical_physical_mismatch_attempt_ids"]
        + normalized_defects["framework_version_mismatch_attempt_ids"]
        + normalized_defects["transport_settings_mismatch_attempt_ids"]
    ):
        raise ValueError("outbound defect names a non-dispatched attempt")
    if any(
        value not in progress_set
        for value in normalized_defects["progress_duplicate_sequence_attempt_ids"]
        + normalized_defects["progress_noncontiguous_sequence_attempt_ids"]
    ):
        raise ValueError("progress sequence defect names a non-progress attempt")
    if any(
        value not in shared_ids
        for value in normalized_defects["call_id_mismatch_attempt_ids"]
    ):
        raise ValueError("call mismatch names an unknown attempt")
    if any(
        value not in logical_call_ids
        for value in normalized_defects["duplicate_logical_request_call_ids"]
    ):
        raise ValueError("duplicate logical request names an unknown call")

    duplicate_progress = sorted(
        attempt_id
        for attempt_id, sequences in normalized_sequences.items()
        if len(sequences) != len(set(sequences))
    )
    noncontiguous_progress = sorted(
        attempt_id
        for attempt_id, sequences in normalized_sequences.items()
        if sequences != list(range(1, len(sequences) + 1))
    )
    call_mismatches = sorted(
        attempt_id
        for attempt_id, call_ids in normalized_call_summary.items()
        if len(call_ids) > 1
    )
    affirmative_pretransport = pretransport - provider_evidence
    derived_defects = {
        "missing_manifest_attempt_ids": sorted(
            (outcomes - affirmative_pretransport) - dispatched
        ),
        "orphan_manifest_attempt_ids": sorted(dispatched - outcomes),
        "progress_without_manifest_attempt_ids": sorted(progress_set - dispatched),
        "progress_without_outcome_attempt_ids": sorted(progress_set - outcomes),
        "progress_duplicate_sequence_attempt_ids": duplicate_progress,
        "progress_noncontiguous_sequence_attempt_ids": noncontiguous_progress,
        "provider_evidence_without_manifest_attempt_ids": sorted(
            provider_evidence - dispatched
        ),
        "pre_transport_with_manifest_attempt_ids": sorted(pretransport & dispatched),
        "pre_transport_with_provider_evidence_attempt_ids": sorted(
            pretransport & provider_evidence
        ),
        "pre_transport_without_outcome_attempt_ids": sorted(pretransport - outcomes),
        "call_id_mismatch_attempt_ids": call_mismatches,
    }
    if any(
        normalized_defects[name] != expected
        for name, expected in derived_defects.items()
    ):
        raise ValueError("provider-attempt relationship defects are inconsistent")

    expected_invariants = {
        "logical_request_exactly_once_per_dispatched_attempt": (
            not normalized_defects["duplicate_logical_request_call_ids"]
            and not normalized_defects["missing_logical_request_attempt_ids"]
        ),
        "logical_physical_request_fields_exact": not normalized_defects[
            "logical_physical_mismatch_attempt_ids"
        ],
        "framework_versions_join_qualification_exact": not normalized_defects[
            "framework_version_mismatch_attempt_ids"
        ],
        "transport_settings_join_selected_profile_exact": not normalized_defects[
            "transport_settings_mismatch_attempt_ids"
        ],
        "manifest_exactly_once_per_dispatched_attempt": not normalized_defects[
            "duplicate_outbound_manifest_attempt_ids"
        ],
        "outcome_attempt_identity_exactly_once": not normalized_defects[
            "duplicate_outcome_attempt_ids"
        ],
        "every_physical_outcome_attempt_accounted_for": not normalized_defects[
            "missing_manifest_attempt_ids"
        ],
        "every_dispatched_attempt_has_terminal_outcome": not normalized_defects[
            "orphan_manifest_attempt_ids"
        ],
        "provider_evidence_always_has_manifest": not normalized_defects[
            "provider_evidence_without_manifest_attempt_ids"
        ],
        "progress_joins_manifest_and_outcome": (
            not normalized_defects["progress_without_manifest_attempt_ids"]
            and not normalized_defects["progress_without_outcome_attempt_ids"]
        ),
        "progress_sequence_unique_and_contiguous": (
            not normalized_defects["progress_duplicate_sequence_attempt_ids"]
            and not normalized_defects["progress_noncontiguous_sequence_attempt_ids"]
        ),
        "explicit_pre_transport_evidence_consistent": (
            not normalized_defects["duplicate_pre_transport_attempt_ids"]
            and not normalized_defects["pre_transport_with_manifest_attempt_ids"]
            and not normalized_defects[
                "pre_transport_with_provider_evidence_attempt_ids"
            ]
            and not normalized_defects["pre_transport_without_outcome_attempt_ids"]
        ),
        "explicit_pre_transport_is_affirmative_not_inferred": True,
        "call_id_join_exact": not normalized_defects["call_id_mismatch_attempt_ids"],
        # The receipt validator is itself closed-schema and the builder copies
        # only the validated identity/counter projections authenticated above.
        "raw_provider_content_persisted": False,
    }
    if invariants != expected_invariants:
        raise ValueError("provider-attempt join invariant projection is inconsistent")

    malformed_present = any(malformed[name] for name in _SOURCE_FIELDS)
    defects_present = any(normalized_defects[name] for name in _DEFECT_FIELDS)
    expected_join_valid = not malformed_present and not defects_present
    if record["join_valid"] is not expected_join_valid:
        raise ValueError("provider-attempt join_valid contradicts its defects")
    if expected_join_valid:
        if expected_frameworks is None or expected_settings is None:
            # Empty runs may be finalized before qualification/route construction.
            if dispatched:
                raise ValueError(
                    "a green dispatched join lacks environment expectations"
                )
        if source_counts["outbound_manifests"] != len(dispatched):
            raise ValueError("green join outbound count is inconsistent")
        if source_counts["outcome_attempts_with_physical_ids"] != len(outcomes):
            raise ValueError("green join outcome-attempt count is inconsistent")
        if source_counts["logical_requests"] != len(logical_call_ids):
            raise ValueError("green join logical-request count is inconsistent")
        if source_counts["progress_rows"] != sum(
            len(sequences) for sequences in normalized_sequences.values()
        ):
            raise ValueError("green join progress-row count is inconsistent")
        if source_counts["explicit_pre_transport_failures"] != len(pretransport):
            raise ValueError("green join pre-transport count is inconsistent")

    supplied = _require_sha256(
        record["join_receipt_sha256"],
        label="join_receipt_sha256",
    )
    authenticated = dict(record)
    del authenticated["join_receipt_sha256"]
    if supplied != _domain_sha256(_JOIN_DOMAIN, authenticated):
        raise ValueError("provider-attempt terminal join hash is invalid")
    return record


__all__ = [
    "PRE_TRANSPORT_FAILURE_EVIDENCE_SCHEMA_VERSION",
    "PROVIDER_ATTEMPT_TERMINAL_JOIN_CONTRACT_ID",
    "PROVIDER_ATTEMPT_TERMINAL_JOIN_SCHEMA_VERSION",
    "build_provider_attempt_terminal_join_receipt",
    "explicit_pre_transport_failure_evidence_record",
    "validate_explicit_pre_transport_failure_evidence_record",
    "validate_provider_attempt_terminal_join_receipt",
    "validate_structured_generation_outcome_record",
]

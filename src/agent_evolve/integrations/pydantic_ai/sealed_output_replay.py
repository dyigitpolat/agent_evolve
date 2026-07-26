"""Fail-closed replay of accepted structured outputs before live continuation.

The replay boundary consumes the three durable channels produced by the
queued Pydantic-AI integration: logical request evidence, typed output
evidence, and terminal queue outcomes.  Exact file digests are supplied by an
external launch/finalization contract.  No provider, credential, environment,
or network dependency exists in this module.

Logical call IDs are deliberately excluded from replay matching because a
continuation run has a fresh identity namespace.  Every semantic request field
is retained, including the exact wire-prompt digest, complete output schema,
generation settings, and requested model.  While accepted source entries
remain, a non-matching request is drift and cannot fall through to the live
provider.  Live execution becomes possible only after the complete sealed
prefix has been consumed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Generic, cast

from pydantic import BaseModel

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.provider_attempt_join import (
    validate_structured_generation_outcome_record,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    structured_generation_request_evidence_record,
    validate_structured_generation_output_evidence_record,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.ports.artifact_store import canonical_json_bytes, decode_json_bytes
from agent_evolve.ports.structured_generator import (
    OutputT,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


SEALED_ACCEPTED_OUTPUT_REPLAY_SCHEMA_VERSION = 1
SEALED_REPLAY_DECISION_RECEIPT_SCHEMA_VERSION = 1
MAX_REPLAY_JSONL_BYTES = 256 * 1024 * 1024
MAX_REPLAY_JSONL_ROWS = 100_000

_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_IDENTITY_DOMAIN = b"agent-evolve:sealed-replay-request-identity:v1\x00"
_SOURCE_IDENTITY_DOMAIN = b"agent-evolve:sealed-replay-source-identity:v1\x00"
_DECISION_RECEIPT_DOMAIN = b"agent-evolve:sealed-replay-decision:v1\x00"
_BUILD_TOKEN = object()

_JOURNAL_ENVELOPE_FIELDS = frozenset({"authenticated_record", "observation"})
_OBSERVATION_FIELDS = frozenset(
    {"monotonic_ns_since_execution_start", "observed_at_utc"}
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
_SOURCE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "source_id",
        "files",
        "accepted_output_count",
        "entries",
        "raw_prompt_or_output_persisted_in_receipt",
        "source_identity_sha256",
    }
)
_SOURCE_FILE_ROLES = frozenset(
    {"request_evidence", "output_evidence", "terminal_outcomes"}
)
_SOURCE_FILE_FIELDS = frozenset({"sha256", "size_bytes", "row_count"})
_SOURCE_ENTRY_FIELDS = frozenset(
    {
        "request_identity_sha256",
        "requested_model",
        "source_call_id",
        "source_request_evidence_sha256",
        "source_output_evidence_sha256",
        "typed_output_sha256",
        "response_record_sha256",
        "attempt_count",
    }
)
_DECISION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "decision",
        "current_call_id",
        "request_identity_sha256",
        "requested_model",
        "source_id",
        "source_identity_sha256",
        "remaining_entry_count",
        "source_call_id",
        "source_request_evidence_sha256",
        "source_output_evidence_sha256",
        "typed_output_sha256",
        "historical_response_record_sha256",
        "historical_attempt_count",
        "response_telemetry_origin",
        "provider_dispatched_by_replay_boundary",
        "typed_output_persisted_in_receipt",
        "replay_decision_receipt_sha256",
    }
)


class SealedAcceptedOutputReplayError(RuntimeError):
    """A sealed replay source or decision violates its generic contract."""


class SealedReplayRequestDriftError(SealedAcceptedOutputReplayError):
    """A request failed to match while accepted prefix entries remained."""


class SealedReplayTypedOutputError(SealedAcceptedOutputReplayError):
    """Recorded typed JSON is not exact under the current trusted output type."""


class SealedReplayReceiptPublicationError(SealedAcceptedOutputReplayError):
    """The required replay decision receipt could not be published."""


@dataclass(frozen=True, slots=True)
class SealedReplayJsonlFile:
    """One immutable expected digest for a canonical durable JSONL channel."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if type(self.sha256) is not str or _LOWER_SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class _LoadedJsonl:
    sha256: str
    size_bytes: int
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _AcceptedReplayEntry:
    request_identity_sha256: str
    requested_model: str
    source_call_id: str
    source_request_evidence_sha256: str
    source_output_evidence_sha256: str
    typed_output_sha256: str
    typed_output_bytes: bytes = field(repr=False)
    response_record_bytes: bytes = field(repr=False)
    response_record_sha256: str
    attempt_count: int


@dataclass(frozen=True, slots=True)
class SealedAcceptedOutputReplaySource:
    """Verified accepted-output prefix detached from its source files."""

    source_id: str
    source_identity_sha256: str
    source_receipt_bytes: bytes = field(repr=False)
    _entries: tuple[_AcceptedReplayEntry, ...] = field(repr=False)
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _BUILD_TOKEN:
            raise TypeError("sealed replay sources are constructed only by verification")
        if type(self.source_id) is not str or _SOURCE_ID.fullmatch(self.source_id) is None:
            raise ValueError("source_id is outside the closed replay grammar")
        _require_sha256(self.source_identity_sha256, label="source_identity_sha256")
        if type(self.source_receipt_bytes) is not bytes:
            raise TypeError("source_receipt_bytes must be exact bytes")
        if type(self._entries) is not tuple or any(
            type(entry) is not _AcceptedReplayEntry for entry in self._entries
        ):
            raise TypeError("replay entries must be an immutable verified tuple")
        receipt = validate_sealed_accepted_output_replay_source_receipt(
            _canonical_mapping(
                decode_json_bytes(self.source_receipt_bytes),
                label="source receipt",
            )
        )
        if receipt.get("source_identity_sha256") != self.source_identity_sha256:
            raise ValueError("source receipt does not authenticate the source identity")
        if receipt.get("accepted_output_count") != len(self._entries):
            raise ValueError("source receipt accepted-output count drifted")

    @property
    def accepted_output_count(self) -> int:
        return len(self._entries)

    @property
    def requested_models(self) -> tuple[str, ...]:
        return tuple(sorted({entry.requested_model for entry in self._entries}))

    def source_receipt(self) -> dict[str, object]:
        """Return a detached JSON-safe receipt without typed output content."""

        return _canonical_mapping(
            decode_json_bytes(self.source_receipt_bytes),
            label="source receipt",
        )


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_sealed_accepted_output_replay_source_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly verify a content-free sealed replay-source receipt."""

    record = _canonical_mapping(value, label="sealed replay source receipt")
    if frozenset(record) != _SOURCE_RECEIPT_FIELDS:
        raise ValueError("sealed replay source receipt has unexpected fields")
    if record["schema_version"] != SEALED_ACCEPTED_OUTPUT_REPLAY_SCHEMA_VERSION:
        raise ValueError("sealed replay source receipt schema is unsupported")
    source_id = record["source_id"]
    if type(source_id) is not str or _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("sealed replay source ID is invalid")
    if record["raw_prompt_or_output_persisted_in_receipt"] is not False:
        raise ValueError("sealed replay source receipt crossed the content boundary")

    files = record["files"]
    if type(files) is not dict or frozenset(files) != _SOURCE_FILE_ROLES:
        raise ValueError("sealed replay source files have unexpected roles")
    for role in _SOURCE_FILE_ROLES:
        file_record = files[role]
        if type(file_record) is not dict or frozenset(file_record) != _SOURCE_FILE_FIELDS:
            raise ValueError("sealed replay source file receipt is invalid")
        _require_sha256(file_record["sha256"], label=f"{role}.sha256")
        for name in ("size_bytes", "row_count"):
            count = file_record[name]
            if type(count) is not int or count < 0:
                raise ValueError(f"{role}.{name} must be non-negative")

    entries = record["entries"]
    accepted_count = record["accepted_output_count"]
    if (
        type(entries) is not list
        or type(accepted_count) is not int
        or accepted_count < 0
        or accepted_count != len(entries)
    ):
        raise ValueError("sealed replay accepted-output count is invalid")
    identities: set[str] = set()
    for entry in entries:
        if type(entry) is not dict or frozenset(entry) != _SOURCE_ENTRY_FIELDS:
            raise ValueError("sealed replay source entry has unexpected fields")
        identity = _require_sha256(
            entry["request_identity_sha256"],
            label="request_identity_sha256",
        )
        if identity in identities:
            raise ValueError("sealed replay source receipt contains duplicate identities")
        identities.add(identity)
        _validate_requested_model(entry["requested_model"])
        LLMCallId(entry["source_call_id"])
        for name in (
            "source_request_evidence_sha256",
            "source_output_evidence_sha256",
            "typed_output_sha256",
            "response_record_sha256",
        ):
            _require_sha256(entry[name], label=name)
        if type(entry["attempt_count"]) is not int or entry["attempt_count"] < 1:
            raise ValueError("sealed replay source attempt count is invalid")

    supplied = _require_sha256(
        record["source_identity_sha256"],
        label="source_identity_sha256",
    )
    authenticated = dict(record)
    del authenticated["source_identity_sha256"]
    if supplied != _domain_sha256(_SOURCE_IDENTITY_DOMAIN, authenticated):
        raise ValueError("source_identity_sha256 does not authenticate the receipt")
    return record


def _canonical_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        detached = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain canonical JSON values") from None
    if type(detached) is not dict:
        raise ValueError(f"{label} must encode one exact object")
    return detached


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _validate_requested_model(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("requested_model must be a non-empty exact string")
    return value


def _request_identity_record(
    request_record: Mapping[str, object],
    *,
    requested_model: str,
) -> dict[str, object]:
    validated = validate_structured_generation_request_evidence_record(request_record)
    model = _validate_requested_model(requested_model)
    logical_request = dict(validated)
    del logical_request["call_id"]
    del logical_request["request_evidence_sha256"]
    return {
        "schema_version": SEALED_ACCEPTED_OUTPUT_REPLAY_SCHEMA_VERSION,
        "requested_model": model,
        "logical_request": logical_request,
    }


def structured_replay_request_identity_sha256(
    request: StructuredGenerationRequest[Any],
    *,
    requested_model: str,
) -> str:
    """Commit a current request independent of its fresh logical call ID."""

    if type(request) is not StructuredGenerationRequest:
        raise TypeError("request must be an exact StructuredGenerationRequest")
    StructuredGenerationRequest.__post_init__(request)
    identity = _request_identity_record(
        structured_generation_request_evidence_record(request),
        requested_model=requested_model,
    )
    return _domain_sha256(_REQUEST_IDENTITY_DOMAIN, identity)


def _record_request_identity_sha256(
    request_record: Mapping[str, object],
    *,
    requested_model: str,
) -> str:
    return _domain_sha256(
        _REQUEST_IDENTITY_DOMAIN,
        _request_identity_record(
            request_record,
            requested_model=requested_model,
        ),
    )


def _unwrap_durable_row(row: object) -> dict[str, object]:
    envelope = _canonical_mapping(row, label="durable JSONL row")
    if frozenset(envelope) != _JOURNAL_ENVELOPE_FIELDS:
        raise ValueError("durable JSONL row has unexpected envelope fields")
    observation = envelope["observation"]
    if type(observation) is not dict or frozenset(observation) != _OBSERVATION_FIELDS:
        raise ValueError("durable JSONL observation has unexpected fields")
    monotonic_ns = observation["monotonic_ns_since_execution_start"]
    observed_at = observation["observed_at_utc"]
    if type(monotonic_ns) is not int or monotonic_ns < 0:
        raise ValueError("durable JSONL monotonic observation is invalid")
    if type(observed_at) is not str or not observed_at:
        raise ValueError("durable JSONL UTC observation is invalid")
    return _canonical_mapping(
        envelope["authenticated_record"],
        label="authenticated durable record",
    )


def _load_jsonl(spec: SealedReplayJsonlFile) -> _LoadedJsonl:
    SealedReplayJsonlFile.__post_init__(spec)
    try:
        path = spec.path.expanduser().resolve(strict=True)
        payload = path.read_bytes()
    except OSError:
        raise SealedAcceptedOutputReplayError(
            "sealed replay JSONL could not be read"
        ) from None
    if len(payload) > MAX_REPLAY_JSONL_BYTES:
        raise SealedAcceptedOutputReplayError("sealed replay JSONL exceeds its bound")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != spec.sha256:
        raise SealedAcceptedOutputReplayError("sealed replay JSONL digest drifted")
    lines = payload.splitlines(keepends=True)
    if len(lines) > MAX_REPLAY_JSONL_ROWS:
        raise SealedAcceptedOutputReplayError("sealed replay JSONL row count exceeds bound")
    rows: list[dict[str, object]] = []
    try:
        for line in lines:
            if not line.endswith(b"\n") or line == b"\n":
                raise ValueError("JSONL rows must be non-empty and newline terminated")
            decoded = decode_json_bytes(line[:-1])
            if canonical_json_bytes(decoded) + b"\n" != line:
                raise ValueError("JSONL row is not canonical")
            rows.append(_unwrap_durable_row(decoded))
    except (TypeError, ValueError):
        raise SealedAcceptedOutputReplayError(
            "sealed replay JSONL violates its canonical envelope"
        ) from None
    return _LoadedJsonl(
        sha256=observed_sha256,
        size_bytes=len(payload),
        rows=tuple(rows),
    )


def _unique_by_call_id(
    records: Sequence[dict[str, object]],
    *,
    call_id_field: str,
    label: str,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for record in records:
        call_id = record.get(call_id_field)
        if type(call_id) is not str:
            raise SealedAcceptedOutputReplayError(f"{label} lacks an exact call ID")
        try:
            LLMCallId(call_id)
        except (TypeError, ValueError):
            raise SealedAcceptedOutputReplayError(
                f"{label} has an invalid call ID"
            ) from None
        if call_id in result:
            raise SealedAcceptedOutputReplayError(f"{label} has duplicate call IDs")
        result[call_id] = record
    return result


def _validated_success_response(
    outcome: Mapping[str, object],
    *,
    request_record: Mapping[str, object],
    output_record: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    if outcome["status"] != "succeeded" or outcome["cancellation_reason"] is not None:
        raise SealedAcceptedOutputReplayError(
            "typed output evidence does not join a successful provider outcome"
        )
    response = outcome["response"]
    attempts = outcome["attempts"]
    if type(response) is not dict or frozenset(response) != _OUTCOME_RESPONSE_FIELDS:
        raise SealedAcceptedOutputReplayError(
            "successful provider outcome response is unavailable"
        )
    if type(attempts) is not list or not attempts:
        raise SealedAcceptedOutputReplayError(
            "successful provider outcome has no physical attempt"
        )
    succeeded = [attempt for attempt in attempts if attempt.get("status") == "succeeded"]
    if len(succeeded) != 1 or succeeded[0] is not attempts[-1]:
        raise SealedAcceptedOutputReplayError(
            "successful provider outcome has ambiguous terminal attempts"
        )
    request_evidence = succeeded[0].get("request_evidence")
    if type(request_evidence) is not dict:
        raise SealedAcceptedOutputReplayError(
            "successful provider attempt lacks request evidence"
        )
    if (
        request_evidence.get("variant") != "original"
        or request_evidence.get("prompt_sha256")
        != request_record["wire_prompt_sha256"]
    ):
        raise SealedAcceptedOutputReplayError(
            "accepted output was not produced from the exact original wire prompt"
        )
    if response["provider_response_id"] != output_record["provider_response_id"]:
        raise SealedAcceptedOutputReplayError(
            "provider response identity does not join typed output evidence"
        )
    _validate_requested_model(response["requested_model"])
    return dict(response), len(attempts)


def _response_record_sha256(response: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(response)).hexdigest()


def load_sealed_accepted_output_replay_jsonl(
    *,
    source_id: str,
    request_evidence: SealedReplayJsonlFile,
    output_evidence: SealedReplayJsonlFile,
    terminal_outcomes: SealedReplayJsonlFile,
) -> SealedAcceptedOutputReplaySource:
    """Load one accepted prefix from exact, externally sealed JSONL files.

    Requests and failed outcomes without typed-output evidence are validated but
    never become replay entries.  Conversely, every output-evidence row must
    join one exact request and one successful outcome.  Duplicate replay
    identities are rejected because selecting among different accepted outputs
    for one logical request would be nondeterministic under concurrency.
    """

    if type(source_id) is not str or _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("source_id is outside the closed replay grammar")
    request_file = _load_jsonl(request_evidence)
    output_file = _load_jsonl(output_evidence)
    outcome_file = _load_jsonl(terminal_outcomes)

    try:
        requests = tuple(
            validate_structured_generation_request_evidence_record(row)
            for row in request_file.rows
        )
        outputs = tuple(
            validate_structured_generation_output_evidence_record(row)
            for row in output_file.rows
        )
        outcomes = tuple(
            validate_structured_generation_outcome_record(row)
            for row in outcome_file.rows
        )
    except (TypeError, ValueError):
        raise SealedAcceptedOutputReplayError(
            "sealed replay evidence failed strict record validation"
        ) from None

    requests_by_call = _unique_by_call_id(
        requests,
        call_id_field="call_id",
        label="request evidence",
    )
    _unique_by_call_id(
        outputs,
        call_id_field="call_id",
        label="output evidence",
    )
    outcomes_by_call = _unique_by_call_id(
        outcomes,
        call_id_field="task_id",
        label="provider outcomes",
    )

    entries: list[_AcceptedReplayEntry] = []
    seen_identities: set[str] = set()
    for output in outputs:
        call_id = cast(str, output["call_id"])
        request = requests_by_call.get(call_id)
        outcome = outcomes_by_call.get(call_id)
        if request is None or outcome is None:
            raise SealedAcceptedOutputReplayError(
                "typed output evidence lacks its request or terminal outcome"
            )
        try:
            output = validate_structured_generation_output_evidence_record(
                output,
                request_evidence=request,
            )
        except (TypeError, ValueError):
            raise SealedAcceptedOutputReplayError(
                "typed output evidence does not join its logical request"
            ) from None
        response, attempt_count = _validated_success_response(
            outcome,
            request_record=request,
            output_record=output,
        )
        requested_model = cast(str, response["requested_model"])
        request_identity_sha256 = _record_request_identity_sha256(
            request,
            requested_model=requested_model,
        )
        if request_identity_sha256 in seen_identities:
            raise SealedAcceptedOutputReplayError(
                "sealed accepted outputs contain a duplicate replay identity"
            )
        seen_identities.add(request_identity_sha256)
        typed_output_bytes = canonical_json_bytes(output["typed_output"])
        response_record_bytes = canonical_json_bytes(response)
        entries.append(
            _AcceptedReplayEntry(
                request_identity_sha256=request_identity_sha256,
                requested_model=requested_model,
                source_call_id=call_id,
                source_request_evidence_sha256=cast(
                    str,
                    request["request_evidence_sha256"],
                ),
                source_output_evidence_sha256=cast(
                    str,
                    output["output_evidence_sha256"],
                ),
                typed_output_sha256=cast(str, output["typed_output_sha256"]),
                typed_output_bytes=typed_output_bytes,
                response_record_bytes=response_record_bytes,
                response_record_sha256=_response_record_sha256(response),
                attempt_count=attempt_count,
            )
        )

    entry_projection = [
        {
            "request_identity_sha256": entry.request_identity_sha256,
            "requested_model": entry.requested_model,
            "source_call_id": entry.source_call_id,
            "source_request_evidence_sha256": (
                entry.source_request_evidence_sha256
            ),
            "source_output_evidence_sha256": entry.source_output_evidence_sha256,
            "typed_output_sha256": entry.typed_output_sha256,
            "response_record_sha256": entry.response_record_sha256,
            "attempt_count": entry.attempt_count,
        }
        for entry in entries
    ]
    receipt: dict[str, object] = {
        "schema_version": SEALED_ACCEPTED_OUTPUT_REPLAY_SCHEMA_VERSION,
        "source_id": source_id,
        "files": {
            "request_evidence": {
                "sha256": request_file.sha256,
                "size_bytes": request_file.size_bytes,
                "row_count": len(request_file.rows),
            },
            "output_evidence": {
                "sha256": output_file.sha256,
                "size_bytes": output_file.size_bytes,
                "row_count": len(output_file.rows),
            },
            "terminal_outcomes": {
                "sha256": outcome_file.sha256,
                "size_bytes": outcome_file.size_bytes,
                "row_count": len(outcome_file.rows),
            },
        },
        "accepted_output_count": len(entries),
        "entries": entry_projection,
        "raw_prompt_or_output_persisted_in_receipt": False,
    }
    receipt["source_identity_sha256"] = _domain_sha256(
        _SOURCE_IDENTITY_DOMAIN,
        receipt,
    )
    return SealedAcceptedOutputReplaySource(
        source_id=source_id,
        source_identity_sha256=cast(str, receipt["source_identity_sha256"]),
        source_receipt_bytes=canonical_json_bytes(receipt),
        _entries=tuple(entries),
        _construction_token=_BUILD_TOKEN,
    )


class SealedReplayDecision(str, Enum):
    REPLAYED = "replayed"
    LIVE_AFTER_PREFIX = "live_after_prefix"
    REQUEST_DRIFT = "request_drift"


ReplayDecisionReceiptSink = Callable[[dict[str, object]], None]
LowLevelStructuredRunner = Callable[
    [StructuredGenerationRequest[Any]],
    Awaitable[
        StructuredGenerationResponse[Any]
        | AttemptedStructuredGenerationResponse[Any]
    ],
]


def _decision_receipt(
    *,
    decision: SealedReplayDecision,
    current_call_id: str,
    request_identity_sha256: str,
    requested_model: str,
    source: SealedAcceptedOutputReplaySource,
    remaining_entry_count: int,
    entry: _AcceptedReplayEntry | None,
) -> dict[str, object]:
    if type(decision) is not SealedReplayDecision:
        raise TypeError("decision must be a SealedReplayDecision")
    record: dict[str, object] = {
        "schema_version": SEALED_REPLAY_DECISION_RECEIPT_SCHEMA_VERSION,
        "decision": decision.value,
        "current_call_id": current_call_id,
        "request_identity_sha256": request_identity_sha256,
        "requested_model": requested_model,
        "source_id": source.source_id,
        "source_identity_sha256": source.source_identity_sha256,
        "remaining_entry_count": remaining_entry_count,
        "source_call_id": None if entry is None else entry.source_call_id,
        "source_request_evidence_sha256": (
            None if entry is None else entry.source_request_evidence_sha256
        ),
        "source_output_evidence_sha256": (
            None if entry is None else entry.source_output_evidence_sha256
        ),
        "typed_output_sha256": None if entry is None else entry.typed_output_sha256,
        "historical_response_record_sha256": (
            None if entry is None else entry.response_record_sha256
        ),
        "historical_attempt_count": None if entry is None else entry.attempt_count,
        "response_telemetry_origin": (
            None if entry is None else "historical_source_attempt"
        ),
        "provider_dispatched_by_replay_boundary": False,
        "typed_output_persisted_in_receipt": False,
    }
    record["replay_decision_receipt_sha256"] = _domain_sha256(
        _DECISION_RECEIPT_DOMAIN,
        record,
    )
    return record


def validate_sealed_replay_decision_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly verify one content-free replay, live, or drift decision."""

    record = _canonical_mapping(value, label="sealed replay decision receipt")
    if frozenset(record) != _DECISION_RECEIPT_FIELDS:
        raise ValueError("sealed replay decision receipt has unexpected fields")
    if record["schema_version"] != SEALED_REPLAY_DECISION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("sealed replay decision receipt schema is unsupported")
    try:
        decision = SealedReplayDecision(record["decision"])
    except (TypeError, ValueError):
        raise ValueError("sealed replay decision is invalid") from None
    LLMCallId(record["current_call_id"])
    _require_sha256(
        record["request_identity_sha256"],
        label="request_identity_sha256",
    )
    _validate_requested_model(record["requested_model"])
    source_id = record["source_id"]
    if type(source_id) is not str or _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("sealed replay decision source ID is invalid")
    _require_sha256(
        record["source_identity_sha256"],
        label="source_identity_sha256",
    )
    remaining = record["remaining_entry_count"]
    if type(remaining) is not int or remaining < 0:
        raise ValueError("remaining_entry_count must be non-negative")
    if (
        record["provider_dispatched_by_replay_boundary"] is not False
        or record["typed_output_persisted_in_receipt"] is not False
    ):
        raise ValueError("sealed replay decision crossed its content/I/O boundary")

    source_fields = (
        "source_call_id",
        "source_request_evidence_sha256",
        "source_output_evidence_sha256",
        "typed_output_sha256",
        "historical_response_record_sha256",
        "historical_attempt_count",
        "response_telemetry_origin",
    )
    if decision is SealedReplayDecision.REPLAYED:
        LLMCallId(record["source_call_id"])
        for name in (
            "source_request_evidence_sha256",
            "source_output_evidence_sha256",
            "typed_output_sha256",
            "historical_response_record_sha256",
        ):
            _require_sha256(record[name], label=name)
        if (
            type(record["historical_attempt_count"]) is not int
            or record["historical_attempt_count"] < 1
            or record["response_telemetry_origin"] != "historical_source_attempt"
        ):
            raise ValueError("replayed historical telemetry provenance is invalid")
    elif any(record[name] is not None for name in source_fields):
        raise ValueError("non-replay decision carries a source output identity")

    supplied = _require_sha256(
        record["replay_decision_receipt_sha256"],
        label="replay_decision_receipt_sha256",
    )
    authenticated = dict(record)
    del authenticated["replay_decision_receipt_sha256"]
    if supplied != _domain_sha256(_DECISION_RECEIPT_DOMAIN, authenticated):
        raise ValueError("replay decision receipt hash is invalid")
    return record


def _parse_cost(value: object) -> Decimal | None:
    if value is None:
        return None
    if type(value) is not str:
        raise SealedReplayTypedOutputError("recorded response cost is invalid")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise SealedReplayTypedOutputError(
            "recorded response cost is invalid"
        ) from None
    if not result.is_finite() or result < 0 or str(result) != value:
        raise SealedReplayTypedOutputError("recorded response cost is not canonical")
    return result


def _materialize_replayed_response(
    entry: _AcceptedReplayEntry,
    request: StructuredGenerationRequest[OutputT],
) -> AttemptedStructuredGenerationResponse[OutputT]:
    output_type = request.output_type
    if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
        raise SealedReplayTypedOutputError(
            "replay requires a trusted Pydantic BaseModel output type"
        )
    try:
        value = output_type.model_validate_json(
            entry.typed_output_bytes,
            strict=True,
        )
        round_trip = canonical_json_bytes(
            BaseModel.model_dump(
                value,
                mode="json",
                by_alias=False,
                exclude_unset=False,
                exclude_defaults=False,
                exclude_none=False,
                exclude_computed_fields=True,
                round_trip=True,
                warnings="error",
                fallback=None,
                serialize_as_any=False,
            )
        )
    except Exception:
        raise SealedReplayTypedOutputError(
            "recorded typed output fails its current trusted output contract"
        ) from None
    if round_trip != entry.typed_output_bytes or (
        hashlib.sha256(round_trip).hexdigest() != entry.typed_output_sha256
    ):
        raise SealedReplayTypedOutputError(
            "recorded typed output is not an exact canonical round trip"
        )
    response = _canonical_mapping(
        decode_json_bytes(entry.response_record_bytes),
        label="historical response record",
    )
    if hashlib.sha256(entry.response_record_bytes).hexdigest() != (
        entry.response_record_sha256
    ):
        raise SealedReplayTypedOutputError("historical response record hash drifted")
    structured = StructuredGenerationResponse(
        value=value,
        requested_model=cast(str, response["requested_model"]),
        resolved_model=cast(str, response["resolved_model"]),
        resolved_provider=cast(str, response["resolved_provider"]),
        provider_response_id=cast(str | None, response["provider_response_id"]),
        finish_reason=cast(str | None, response["finish_reason"]),
        input_tokens=cast(int, response["input_tokens"]),
        output_tokens=cast(int, response["output_tokens"]),
        reasoning_tokens=cast(int, response["reasoning_tokens"]),
        cache_read_tokens=cast(int, response["cache_read_tokens"]),
        cache_write_tokens=cast(int, response["cache_write_tokens"]),
        cost_usd=_parse_cost(response["cost_usd"]),
        latency_ns=cast(int, response["latency_ns"]),
    )
    return AttemptedStructuredGenerationResponse(
        response=structured,
        attempt_count=entry.attempt_count,
    )


class SealedReplayThenLiveStructuredRunner(Generic[OutputT]):
    """Consume an exact accepted prefix, then delegate subsequent calls live."""

    def __init__(
        self,
        *,
        source: SealedAcceptedOutputReplaySource,
        requested_model: str,
        live_runner: LowLevelStructuredRunner,
        decision_receipt_sink: ReplayDecisionReceiptSink,
    ) -> None:
        if type(source) is not SealedAcceptedOutputReplaySource:
            raise TypeError("source must be a SealedAcceptedOutputReplaySource")
        model = _validate_requested_model(requested_model)
        if not callable(live_runner):
            raise TypeError("live_runner must be callable")
        if not callable(decision_receipt_sink):
            raise TypeError("decision_receipt_sink must be callable")
        foreign_models = tuple(
            source_model
            for source_model in source.requested_models
            if source_model != model
        )
        if foreign_models:
            raise ValueError("sealed replay source contains a foreign requested model")
        self.source = source
        self.requested_model = model
        self._live_runner = live_runner
        self._decision_receipt_sink = decision_receipt_sink
        self._remaining = {
            entry.request_identity_sha256: entry for entry in source._entries
        }
        self._decision_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def remaining_entry_count(self) -> int:
        return len(self._remaining)

    async def __call__(
        self,
        request: StructuredGenerationRequest[OutputT],
    ) -> AttemptedStructuredGenerationResponse[OutputT] | StructuredGenerationResponse[OutputT]:
        return await self.generate(request)

    async def generate(
        self,
        request: StructuredGenerationRequest[OutputT],
    ) -> AttemptedStructuredGenerationResponse[OutputT] | StructuredGenerationResponse[OutputT]:
        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(request)
        if self._closed:
            raise SealedAcceptedOutputReplayError("sealed replay runner is closed")
        identity = structured_replay_request_identity_sha256(
            request,
            requested_model=self.requested_model,
        )
        async with self._decision_lock:
            entry = self._remaining.get(identity)
            if entry is not None:
                replayed = _materialize_replayed_response(entry, request)
                receipt = _decision_receipt(
                    decision=SealedReplayDecision.REPLAYED,
                    current_call_id=request.call_id.value,
                    request_identity_sha256=identity,
                    requested_model=self.requested_model,
                    source=self.source,
                    remaining_entry_count=len(self._remaining) - 1,
                    entry=entry,
                )
                self._publish_receipt(receipt)
                del self._remaining[identity]
                return replayed

            if self._remaining:
                receipt = _decision_receipt(
                    decision=SealedReplayDecision.REQUEST_DRIFT,
                    current_call_id=request.call_id.value,
                    request_identity_sha256=identity,
                    requested_model=self.requested_model,
                    source=self.source,
                    remaining_entry_count=len(self._remaining),
                    entry=None,
                )
                self._publish_receipt(receipt)
                raise SealedReplayRequestDriftError(
                    "structured request drifted before sealed replay prefix exhaustion"
                )

            receipt = _decision_receipt(
                decision=SealedReplayDecision.LIVE_AFTER_PREFIX,
                current_call_id=request.call_id.value,
                request_identity_sha256=identity,
                requested_model=self.requested_model,
                source=self.source,
                remaining_entry_count=0,
                entry=None,
            )
            self._publish_receipt(receipt)

        result = await self._live_runner(request)
        if type(result) is StructuredGenerationResponse:
            StructuredGenerationResponse.__post_init__(result)
            if type(result.value) is not request.output_type:
                raise TypeError("live runner response violates the requested output type")
            return cast(StructuredGenerationResponse[OutputT], result)
        if type(result) is AttemptedStructuredGenerationResponse:
            AttemptedStructuredGenerationResponse.__post_init__(result)
            if type(result.response.value) is not request.output_type:
                raise TypeError("live runner response violates the requested output type")
            return cast(AttemptedStructuredGenerationResponse[OutputT], result)
        raise TypeError("live runner returned an unsupported structured response")

    def _publish_receipt(self, receipt: dict[str, object]) -> None:
        try:
            validated = validate_sealed_replay_decision_receipt(receipt)
            self._decision_receipt_sink(validated)
        except Exception:
            raise SealedReplayReceiptPublicationError(
                "required sealed replay decision receipt publication failed"
            ) from None

    async def snapshot(self) -> object:
        snapshot = getattr(self._live_runner, "snapshot", None)
        if not callable(snapshot):
            raise AttributeError("live runner does not expose snapshot")
        return await snapshot()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._live_runner, "aclose", None)
            if callable(close):
                await close()

    async def __aenter__(self) -> "SealedReplayThenLiveStructuredRunner[OutputT]":
        if self._closed:
            raise SealedAcceptedOutputReplayError("sealed replay runner is closed")
        enter = getattr(self._live_runner, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


__all__ = [
    "MAX_REPLAY_JSONL_BYTES",
    "MAX_REPLAY_JSONL_ROWS",
    "ReplayDecisionReceiptSink",
    "SEALED_ACCEPTED_OUTPUT_REPLAY_SCHEMA_VERSION",
    "SEALED_REPLAY_DECISION_RECEIPT_SCHEMA_VERSION",
    "SealedAcceptedOutputReplayError",
    "SealedAcceptedOutputReplaySource",
    "SealedReplayDecision",
    "SealedReplayJsonlFile",
    "SealedReplayReceiptPublicationError",
    "SealedReplayRequestDriftError",
    "SealedReplayThenLiveStructuredRunner",
    "SealedReplayTypedOutputError",
    "load_sealed_accepted_output_replay_jsonl",
    "structured_replay_request_identity_sha256",
    "validate_sealed_accepted_output_replay_source_receipt",
    "validate_sealed_replay_decision_receipt",
]

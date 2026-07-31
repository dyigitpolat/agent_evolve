"""Queue-owned retry composition for one-attempt structured generation.

The Pydantic-AI adapter remains a one-attempt executor.  This module gives the
application queue sole ownership of admission bounds, timeouts, retries,
backoff, and sleeps, then exposes the callable expected by the high-level
agentic adapter.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
from enum import Enum
from functools import partial
from random import SystemRandom
from typing import Any, Generic, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from agent_evolve.application.llm_task_queue import (
    AsyncLLMTaskQueue,
    LLMTaskQueueClosedError,
)
from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId
from agent_evolve.domain.llm_task_queue import (
    MAX_ATTEMPTS,
    NANOSECONDS_PER_SECOND,
    AttemptRequestEvidence,
    AttemptRequestVariant,
    LLMAttemptContext,
    LLMTask,
    LLMTaskOutcome,
    PartitionedRetryBudget,
    QueueSnapshot,
    RetryAfter,
    RetryAfterSource,
    RetryClassification,
    RetryDisposition,
    RetryReason,
    SanitizedAttemptFailure,
    StructuredOutputFailureMode,
    TaskOutcomeStatus,
    TaskTelemetry,
    ValidationIssueCategory,
    ValidationIssueReasonCode,
)
from agent_evolve.infrastructure.asyncio_runtime import (
    AsyncioRuntime,
    TransportAbortedTimeoutError,
)
from agent_evolve.infrastructure.clock import SystemClock
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    PydanticAIStructuredGenerator,
)
from agent_evolve.policies.llm_backoff import (
    ExponentialBackoff,
    FullJitter,
    JitterPolicy,
    RandomRange,
)
from agent_evolve.ports.llm_task_queue import (
    AsyncRuntime,
    BackoffPolicy,
    PreparedLLMAttempt,
    RetryClassifier,
)
from agent_evolve.ports.generation_failure import GenerationFailureDisposition
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256,
    IDENTITY_PROMPT_RENDERER_ID,
    IDENTITY_PROMPT_RENDERER_REVISION,
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_UTF8_BYTES,
    OutputT,
    StructuredGenerationError,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredGenerator,
    StructuredPromptLineage,
    StructuredStreamCleanupTimeoutError,
    StructuredStreamTimeoutError,
    StructuredStreamTimeoutPhase,
    identity_prompt_lineage,
)


DEFAULT_MAX_IN_FLIGHT = 8
DEFAULT_MAX_PENDING = 64
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_ATTEMPT_TIMEOUT_NS = 90 * NANOSECONDS_PER_SECOND
DEFAULT_BASE_BACKOFF_NS = NANOSECONDS_PER_SECOND // 2
DEFAULT_MAX_BACKOFF_NS = 30 * NANOSECONDS_PER_SECOND
_MAX_RETRY_AFTER_NS = 2**63 - 1
_PROVIDER_ATTEMPT_ID_DOMAIN = b"agent-evolve:provider-attempt-id:v1\x00"
_STRUCTURED_REQUEST_EVIDENCE_DOMAIN = b"agent-evolve:structured-request-evidence:v2\x00"
_STRUCTURED_OUTPUT_EVIDENCE_DOMAIN = b"agent-evolve:structured-output-evidence:v1\x00"
_POLICY_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_OPERATION = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_EVIDENCE_TOOL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
MAX_STRUCTURED_OUTPUT_SCHEMA_UTF8_BYTES = 1_048_576
MAX_STRUCTURED_OUTPUT_EVIDENCE_UTF8_BYTES = 4_194_304
STRUCTURED_REQUEST_EVIDENCE_SCHEMA_VERSION = 2
STRUCTURED_OUTPUT_EVIDENCE_SCHEMA_VERSION = 1
STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION = 8
SUPPORTED_STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSIONS = frozenset({5, 6, 7, 8})
_STRUCTURED_REQUEST_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "call_id",
        "operation",
        "prompt_sha256",
        "wire_prompt_sha256",
        "prompt_utf8_bytes",
        "semantic_prompt_sha256",
        "prompt_renderer_id",
        "prompt_renderer_revision",
        "prompt_renderer_definition_sha256",
        "output_tool_name",
        "output_type",
        "output_schema",
        "output_schema_sha256",
        "output_schema_utf8_bytes",
        "max_output_tokens",
        "temperature_hex",
        "request_evidence_sha256",
    }
)
_STRUCTURED_OUTPUT_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "call_id",
        "operation",
        "provider_response_id",
        "request_evidence_sha256",
        "output_tool_name",
        "output_schema_sha256",
        "typed_output",
        "typed_output_sha256",
        "typed_output_utf8_bytes",
        "output_evidence_sha256",
    }
)
SCHEMA_REPAIR_POLICY_ID = "structured_output_schema_repair"
SCHEMA_REPAIR_POLICY_VERSION = 4
SCHEMA_REPAIR_PROMPT_RENDERER_ID = "agent_evolve.schema_repair_prompt"
SCHEMA_REPAIR_PROMPT_RENDERER_REVISION = "schema_repair_v4"
MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES = 24_576
MAX_SCHEMA_REPAIR_SCHEMA_NODES = 4_096
MAX_SCHEMA_REPAIR_REQUIRED_PATHS = 256
_SCHEMA_REPAIR_TEMPLATE = (
    "\n\nSTRUCTURED_OUTPUT_SCHEMA_REPAIR_V{policy_version}\n"
    "The previous provider response did not satisfy the typed output contract. "
    "Failure mode: {failure_mode}. Repair pass: {repair_pass}.\n"
    "Schema-required field paths from the trusted local output contract "
    "(JSON Pointer; '*' marks each emitted collection item): "
    "{required_paths_json}\n"
    "Include every applicable path above. "
    "{issue_block}"
    "{literal_constraint_block}"
    "Call the {output_tool_name} output tool exactly once. Emit only fields "
    "declared by its schema, include every required field, and use exact schema "
    "literals, enums, and types. Do not emit commentary outside the tool call."
    "{completion_guidance}{escalation_guidance}"
)
_SEMANTIC_REPAIR_GUIDANCE: dict[ValidationIssueReasonCode, str] = {
    ValidationIssueReasonCode.DUPLICATE_FINITE_OPTIONS: (
        "Correction: every proposed finite option ID must be distinct."
    ),
    ValidationIssueReasonCode.FINITE_OPTION_OUT_OF_CONTRACT: (
        "Correction: every proposed option_id must exactly match one option_id "
        "from the request's sealed ordered_options list."
    ),
    ValidationIssueReasonCode.ASSIGNED_MEMORY_CARD_OMITTED: (
        "Correction: across the complete proposal, include every prospectively "
        "assigned memory-card key in at least one member's "
        "supporting_card_keys; also obey the supplied compatibility and dose "
        "bounds exactly."
    ),
    ValidationIssueReasonCode.PROPOSAL_SUPPORT_OPTION_OMITTED: (
        "Correction: include every engine-reserved proposal-support option in "
        "the complete proposal; copy each reserved option ID exactly from the "
        "trusted request and keep all proposal members distinct."
    ),
    ValidationIssueReasonCode.NO_FEASIBLE_DISJOINT_PORTFOLIO: (
        "Correction: the complete proposal must contain a subset satisfying "
        "the supplied evaluation-size, pairwise changed-path, and distinct-family "
        "constraints."
    ),
    ValidationIssueReasonCode.PORTFOLIO_MEMORY_DOSE_VIOLATION: (
        "Correction: obey every supplied memory-dose bound, cite only "
        "card-compatible options, include every assigned card, and preserve the "
        "required unattributed-member count."
    ),
    ValidationIssueReasonCode.REFLECTION_METRIC_CONTRACT_VIOLATION: (
        "Correction: in every insight, emit each required metric exactly once "
        "and emit no other metric."
    ),
    ValidationIssueReasonCode.REFLECTION_ACTION_CONTRACT_VIOLATION: (
        "Correction: use at least one allowed recommended option ID and family; "
        "use only request-listed values, remove duplicates, and keep every "
        "recommended ID, family, affected path, and capability mutually "
        "consistent with the cited observed action."
    ),
    ValidationIssueReasonCode.REFLECTION_SEMANTIC_CONTRACT_VIOLATION: (
        "Correction: use only the request-listed insight kind, consumer scope, "
        "affected path, and factor capability values; required set-like arrays "
        "must be nonempty where specified and contain no duplicates."
    ),
    ValidationIssueReasonCode.REFLECTION_DIRECTION_OR_ANCHOR_VIOLATION: (
        "Correction: every metric prediction must use an adjudicable non-unknown "
        "direction and an explicitly allowed comparison anchor; supply a source "
        "role only when that anchor kind requires one."
    ),
    ValidationIssueReasonCode.RESIDUAL_RADIUS_CONTRACT_VIOLATION: (
        "Correction: every residual member must contain exactly an allowed "
        "number of component_option_ids."
    ),
    ValidationIssueReasonCode.RESIDUAL_OPTION_CONTRACT_VIOLATION: (
        "Correction: within each residual member, use distinct option IDs that "
        "are all available for its selected parent; a two-option member must "
        "copy one of that parent's declared safe disjoint pairs."
    ),
    ValidationIssueReasonCode.RESIDUAL_METRIC_CONTRACT_VIOLATION: (
        "Correction: within every residual member, emit each required metric "
        "exactly once and emit no other metric."
    ),
    ValidationIssueReasonCode.RESIDUAL_QUANTILE_ORDER_VIOLATION: (
        "Correction: every metric forecast must contain finite raw deltas in "
        "nondecreasing order: p10_delta <= p50_delta <= p90_delta."
    ),
    ValidationIssueReasonCode.RESIDUAL_PLAN_DIVERSITY_VIOLATION: (
        "Correction: residual members must be distinct parent-relative plans "
        "and collectively cover at least the requested number of distinct "
        "parents."
    ),
}
_DEFAULT_SEMANTIC_REPAIR_GUIDANCE = (
    "Correction: rebuild the complete typed output and satisfy the named "
    "trusted semantic constraint."
)


def _semantic_repair_guidance_record() -> dict[str, object]:
    """Return the complete deterministic guidance contract.

    Validator reason codes and prompt guidance evolve in separate modules.  A
    total, content-addressed record prevents either enum drift or wording drift
    from changing retry behavior under an unchanged experiment identity.
    """

    missing = set(ValidationIssueReasonCode).difference(_SEMANTIC_REPAIR_GUIDANCE)
    extra = set(_SEMANTIC_REPAIR_GUIDANCE).difference(ValidationIssueReasonCode)
    if missing or extra:
        raise ValueError(
            "semantic repair guidance must cover every validation reason code "
            "exactly"
        )
    return {
        "default_guidance": _DEFAULT_SEMANTIC_REPAIR_GUIDANCE,
        "reason_guidance": {
            reason.value: _SEMANTIC_REPAIR_GUIDANCE[reason]
            for reason in sorted(
                ValidationIssueReasonCode,
                key=lambda item: item.value,
            )
        },
    }


@dataclass(frozen=True, slots=True)
class SchemaRepairPolicyManifest:
    """Immutable, self-authenticating schema-repair experiment contract."""

    policy_id: str
    policy_version: int
    max_suffix_utf8_bytes: int
    max_schema_nodes: int
    max_required_paths: int
    template_sha256: str
    semantic_guidance_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.policy_id) is not str
            or _POLICY_ID.fullmatch(self.policy_id) is None
        ):
            raise ValueError("policy_id must use the closed lowercase token grammar")
        if type(self.policy_version) is not int or self.policy_version < 1:
            raise ValueError("policy_version must be a positive integer")
        if (
            type(self.max_suffix_utf8_bytes) is not int
            or not 1 <= self.max_suffix_utf8_bytes <= MAX_PROMPT_UTF8_BYTES
        ):
            raise ValueError("max_suffix_utf8_bytes is outside the prompt boundary")
        if type(self.max_schema_nodes) is not int or self.max_schema_nodes < 1:
            raise ValueError("max_schema_nodes must be a positive integer")
        if type(self.max_required_paths) is not int or self.max_required_paths < 1:
            raise ValueError("max_required_paths must be a positive integer")
        for name, value in (
            ("template_sha256", self.template_sha256),
            ("semantic_guidance_sha256", self.semantic_guidance_sha256),
            ("policy_sha256", self.policy_sha256),
        ):
            if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        expected = hashlib.sha256(
            json.dumps(
                self._policy_record(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        if self.policy_sha256 != expected:
            raise ValueError("policy_sha256 does not authenticate the policy fields")

    def _policy_record(self) -> dict[str, object]:
        return {
            "max_required_paths": self.max_required_paths,
            "max_schema_nodes": self.max_schema_nodes,
            "max_suffix_utf8_bytes": self.max_suffix_utf8_bytes,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "semantic_guidance_sha256": self.semantic_guidance_sha256,
            "template_sha256": self.template_sha256,
        }

    def to_trace_record(self) -> dict[str, object]:
        """Return the complete JSON-safe contract for a launch manifest."""

        return {
            **self._policy_record(),
            "policy_sha256": self.policy_sha256,
        }


def _schema_repair_policy_manifest() -> SchemaRepairPolicyManifest:
    template_sha256 = hashlib.sha256(
        _SCHEMA_REPAIR_TEMPLATE.encode("utf-8")
    ).hexdigest()
    semantic_guidance_sha256 = hashlib.sha256(
        json.dumps(
            _semantic_repair_guidance_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    policy_record = {
        "max_required_paths": MAX_SCHEMA_REPAIR_REQUIRED_PATHS,
        "max_schema_nodes": MAX_SCHEMA_REPAIR_SCHEMA_NODES,
        "max_suffix_utf8_bytes": MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES,
        "policy_id": SCHEMA_REPAIR_POLICY_ID,
        "policy_version": SCHEMA_REPAIR_POLICY_VERSION,
        "semantic_guidance_sha256": semantic_guidance_sha256,
        "template_sha256": template_sha256,
    }
    policy_sha256 = hashlib.sha256(
        json.dumps(
            policy_record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return SchemaRepairPolicyManifest(
        **policy_record,
        policy_sha256=policy_sha256,
    )


SCHEMA_REPAIR_POLICY_MANIFEST = _schema_repair_policy_manifest()
OutcomeSink = Callable[
    [LLMTaskOutcome[StructuredGenerationResponse[Any]]],
    None,
]
StructuredRequestEvidenceSink = Callable[[dict[str, object]], None]
StructuredOutputEvidenceSink = Callable[[dict[str, object]], None]


class OutcomePublicationPolicy(str, Enum):
    """Control whether terminal-outcome publication is advisory or required.

    ``REQUIRED`` makes publication a synchronous fail-closed boundary: no
    successful response is returned to downstream validation or experiment
    policy unless the sink returns normally.  Actual durability remains the
    sink's responsibility (for example, flush and fsync before returning).
    Publication failure never causes a provider retry because the queue has
    already reached one terminal logical outcome.
    """

    BEST_EFFORT = "best_effort"
    REQUIRED = "required"


class StructuredEvidencePublicationPolicy(str, Enum):
    """Control publication of opt-in request/output content evidence.

    These records intentionally cross the privacy boundary that the sanitized
    terminal-outcome projection does not: request evidence authenticates the
    exact *wire* prompt and output contract, while successful-output evidence
    contains the canonical typed output itself.  Callers must opt in by supplying
    both sinks.  ``REQUIRED`` makes both synchronous durability barriers.
    """

    BEST_EFFORT = "best_effort"
    REQUIRED = "required"


class StructuredEvidencePublicationStage(str, Enum):
    REQUEST = "request"
    OUTPUT = "output"


def _canonical_evidence_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _output_schema_record(
    output_type: type[Any],
) -> tuple[dict[str, object], bytes, str]:
    try:
        schema = output_type.model_json_schema(mode="validation")
    except Exception as exc:
        raise TypeError("structured output type cannot render a JSON schema") from exc
    if type(schema) is not dict:
        raise TypeError("structured output schema must be an exact object")
    schema_bytes = _canonical_evidence_bytes(schema)
    if len(schema_bytes) > MAX_STRUCTURED_OUTPUT_SCHEMA_UTF8_BYTES:
        raise ValueError("structured output schema exceeds the evidence bound")
    return schema, schema_bytes, hashlib.sha256(schema_bytes).hexdigest()


def structured_generation_request_evidence_record(
    request: StructuredGenerationRequest[Any],
) -> dict[str, object]:
    """Authenticate one exact prequeue wire request without retaining its prompt.

    The high-level agentic adapter may render a semantic prompt into a different
    provider-facing prompt (for example by appending a reflection wire-contract
    note).  This boundary is downstream of that rendering, so ``prompt_sha256``
    names the bytes actually submitted to the queue rather than an upstream
    semantic-prompt commitment.
    """

    if type(request) is not StructuredGenerationRequest:
        raise TypeError("request must be an exact StructuredGenerationRequest")
    StructuredGenerationRequest.__post_init__(request)
    schema, schema_bytes, schema_sha256 = _output_schema_record(request.output_type)
    output_type = request.output_type
    prompt_bytes = request.prompt.encode("utf-8", errors="strict")
    wire_prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    prompt_lineage = request.prompt_lineage
    record: dict[str, object] = {
        "schema_version": STRUCTURED_REQUEST_EVIDENCE_SCHEMA_VERSION,
        "call_id": request.call_id.value,
        "operation": request.operation,
        # ``prompt_sha256`` is retained as an unambiguous compatibility alias
        # for existing journal consumers; new consumers should use the
        # explicitly named wire-prompt field.
        "prompt_sha256": wire_prompt_sha256,
        "wire_prompt_sha256": wire_prompt_sha256,
        "prompt_utf8_bytes": len(prompt_bytes),
        "semantic_prompt_sha256": (
            None if prompt_lineage is None else prompt_lineage.semantic_prompt_sha256
        ),
        "prompt_renderer_id": (
            None if prompt_lineage is None else prompt_lineage.renderer_id
        ),
        "prompt_renderer_revision": (
            None if prompt_lineage is None else prompt_lineage.renderer_revision
        ),
        "prompt_renderer_definition_sha256": (
            None
            if prompt_lineage is None
            else prompt_lineage.renderer_definition_sha256
        ),
        "output_tool_name": request.output_tool_name,
        "output_type": {
            "module": output_type.__module__,
            "qualname": output_type.__qualname__,
        },
        "output_schema": schema,
        "output_schema_sha256": schema_sha256,
        "output_schema_utf8_bytes": len(schema_bytes),
        "max_output_tokens": request.max_output_tokens,
        "temperature_hex": (
            None if request.temperature is None else float(request.temperature).hex()
        ),
    }
    record["request_evidence_sha256"] = hashlib.sha256(
        _STRUCTURED_REQUEST_EVIDENCE_DOMAIN + _canonical_evidence_bytes(record)
    ).hexdigest()
    return record


def structured_generation_output_evidence_record(
    request: StructuredGenerationRequest[Any],
    outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
    *,
    request_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Retain one bounded canonical typed output before downstream validation."""

    if type(request) is not StructuredGenerationRequest:
        raise TypeError("request must be an exact StructuredGenerationRequest")
    StructuredGenerationRequest.__post_init__(request)
    if type(outcome) is not LLMTaskOutcome:
        raise TypeError("outcome must be an exact LLMTaskOutcome")
    LLMTaskOutcome.__post_init__(outcome)
    if outcome.status is not TaskOutcomeStatus.SUCCEEDED:
        raise ValueError("typed output evidence requires a successful outcome")
    if outcome.telemetry.task_id != request.call_id.value:
        raise ValueError("request and successful outcome call identities differ")
    response = outcome.response
    if type(response) is not StructuredGenerationResponse:
        raise TypeError("successful outcome has no structured response")
    StructuredGenerationResponse.__post_init__(response)
    if type(response.value) is not request.output_type or not isinstance(
        response.value, BaseModel
    ):
        raise TypeError("successful typed output differs from its output contract")
    expected_request = structured_generation_request_evidence_record(request)
    if request_evidence is None:
        request_record = expected_request
    else:
        if type(request_evidence) is not dict or request_evidence != expected_request:
            raise ValueError("request evidence differs from the exact wire request")
        request_record = request_evidence
    typed_output = BaseModel.model_dump(
        response.value,
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
    output_bytes = _canonical_evidence_bytes(typed_output)
    if len(output_bytes) > MAX_STRUCTURED_OUTPUT_EVIDENCE_UTF8_BYTES:
        raise ValueError("typed output exceeds the evidence bound")
    record: dict[str, object] = {
        "schema_version": STRUCTURED_OUTPUT_EVIDENCE_SCHEMA_VERSION,
        "call_id": request.call_id.value,
        "operation": request.operation,
        "provider_response_id": response.provider_response_id,
        "request_evidence_sha256": request_record["request_evidence_sha256"],
        "output_tool_name": request.output_tool_name,
        "output_schema_sha256": request_record["output_schema_sha256"],
        "typed_output": typed_output,
        "typed_output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "typed_output_utf8_bytes": len(output_bytes),
    }
    record["output_evidence_sha256"] = hashlib.sha256(
        _STRUCTURED_OUTPUT_EVIDENCE_DOMAIN + _canonical_evidence_bytes(record)
    ).hexdigest()
    return record


def _canonical_evidence_mapping(
    record: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        encoded = _canonical_evidence_bytes(dict(record))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain canonical JSON values") from exc
    if type(decoded) is not dict:
        raise ValueError(f"{label} must encode one exact object")
    return decoded


def _validate_evidence_sha256(value: object, *, field_name: str) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _validate_evidence_identity_fields(record: Mapping[str, object]) -> None:
    call_id = record["call_id"]
    if type(call_id) is not str:
        raise ValueError("call_id must be an exact string")
    try:
        LLMCallId(call_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("call_id is outside the generic LLM identity domain") from exc
    operation = record["operation"]
    if type(operation) is not str or _EVIDENCE_OPERATION.fullmatch(operation) is None:
        raise ValueError("operation is outside the closed token grammar")
    tool_name = record["output_tool_name"]
    if type(tool_name) is not str or _EVIDENCE_TOOL.fullmatch(tool_name) is None:
        raise ValueError("output_tool_name is outside the closed tool grammar")


def validate_structured_generation_request_evidence_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Strictly verify and detach one persisted prequeue request record."""

    canonical = _canonical_evidence_mapping(
        record,
        label="structured request evidence",
    )
    if frozenset(canonical) != _STRUCTURED_REQUEST_EVIDENCE_FIELDS:
        raise ValueError("structured request evidence has unexpected fields")
    if (
        type(canonical["schema_version"]) is not int
        or canonical["schema_version"] != STRUCTURED_REQUEST_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported structured request evidence schema version")
    _validate_evidence_identity_fields(canonical)

    prompt_sha256 = _validate_evidence_sha256(
        canonical["prompt_sha256"],
        field_name="prompt_sha256",
    )
    wire_prompt_sha256 = _validate_evidence_sha256(
        canonical["wire_prompt_sha256"],
        field_name="wire_prompt_sha256",
    )
    if prompt_sha256 != wire_prompt_sha256:
        raise ValueError("prompt_sha256 must equal its wire compatibility alias")
    prompt_utf8_bytes = canonical["prompt_utf8_bytes"]
    if (
        type(prompt_utf8_bytes) is not int
        or not 1 <= prompt_utf8_bytes <= MAX_PROMPT_UTF8_BYTES
    ):
        raise ValueError("prompt_utf8_bytes is outside the generic prompt bound")

    lineage_values = (
        canonical["semantic_prompt_sha256"],
        canonical["prompt_renderer_id"],
        canonical["prompt_renderer_revision"],
        canonical["prompt_renderer_definition_sha256"],
    )
    if not all(value is None for value in lineage_values):
        if any(value is None for value in lineage_values):
            raise ValueError("prompt lineage fields must be all present or all absent")
        lineage = StructuredPromptLineage(
            semantic_prompt_sha256=cast(str, lineage_values[0]),
            renderer_id=cast(str, lineage_values[1]),
            renderer_revision=cast(str, lineage_values[2]),
            renderer_definition_sha256=cast(str, lineage_values[3]),
        )
        if lineage.renderer_id == IDENTITY_PROMPT_RENDERER_ID and (
            lineage.semantic_prompt_sha256 != wire_prompt_sha256
            or lineage.renderer_revision != IDENTITY_PROMPT_RENDERER_REVISION
            or lineage.renderer_definition_sha256
            != IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256
        ):
            raise ValueError("identity renderer lineage is inconsistent")

    output_type = canonical["output_type"]
    if type(output_type) is not dict or frozenset(output_type) != {
        "module",
        "qualname",
    }:
        raise ValueError("output_type must contain exact module and qualname fields")
    if any(
        type(output_type[name]) is not str or not output_type[name]
        for name in ("module", "qualname")
    ):
        raise ValueError("output_type identities must be non-empty exact strings")

    output_schema = canonical["output_schema"]
    if type(output_schema) is not dict:
        raise ValueError("output_schema must be an exact object")
    schema_bytes = _canonical_evidence_bytes(output_schema)
    if len(schema_bytes) > MAX_STRUCTURED_OUTPUT_SCHEMA_UTF8_BYTES:
        raise ValueError("output_schema exceeds the evidence bound")
    schema_utf8_bytes = canonical["output_schema_utf8_bytes"]
    if type(schema_utf8_bytes) is not int or schema_utf8_bytes != len(schema_bytes):
        raise ValueError("output_schema_utf8_bytes does not authenticate the schema")
    schema_sha256 = _validate_evidence_sha256(
        canonical["output_schema_sha256"],
        field_name="output_schema_sha256",
    )
    if schema_sha256 != hashlib.sha256(schema_bytes).hexdigest():
        raise ValueError("output_schema_sha256 does not authenticate the schema")

    max_output_tokens = canonical["max_output_tokens"]
    if (
        type(max_output_tokens) is not int
        or not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS
    ):
        raise ValueError("max_output_tokens is outside the generic port bound")
    temperature_hex = canonical["temperature_hex"]
    if temperature_hex is not None:
        if type(temperature_hex) is not str:
            raise ValueError("temperature_hex must be an exact string or None")
        try:
            temperature = float.fromhex(temperature_hex)
        except ValueError as exc:
            raise ValueError(
                "temperature_hex is not a finite hexadecimal float"
            ) from exc
        if (
            not math.isfinite(temperature)
            or not 0 <= temperature <= 2
            or temperature.hex() != temperature_hex
        ):
            raise ValueError("temperature_hex is outside the canonical range")

    supplied_sha256 = _validate_evidence_sha256(
        canonical["request_evidence_sha256"],
        field_name="request_evidence_sha256",
    )
    authenticated = dict(canonical)
    del authenticated["request_evidence_sha256"]
    expected_sha256 = hashlib.sha256(
        _STRUCTURED_REQUEST_EVIDENCE_DOMAIN + _canonical_evidence_bytes(authenticated)
    ).hexdigest()
    if supplied_sha256 != expected_sha256:
        raise ValueError("request_evidence_sha256 does not authenticate the record")
    return canonical


def validate_structured_generation_output_evidence_record(
    record: Mapping[str, object],
    *,
    request_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Strictly verify one typed-output record and its optional request join."""

    canonical = _canonical_evidence_mapping(
        record,
        label="structured output evidence",
    )
    if frozenset(canonical) != _STRUCTURED_OUTPUT_EVIDENCE_FIELDS:
        raise ValueError("structured output evidence has unexpected fields")
    if (
        type(canonical["schema_version"]) is not int
        or canonical["schema_version"] != STRUCTURED_OUTPUT_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported structured output evidence schema version")
    _validate_evidence_identity_fields(canonical)
    provider_response_id = canonical["provider_response_id"]
    if provider_response_id is not None and (
        type(provider_response_id) is not str or not provider_response_id
    ):
        raise ValueError("provider_response_id must be non-empty or None")
    for name in (
        "request_evidence_sha256",
        "output_schema_sha256",
        "typed_output_sha256",
        "output_evidence_sha256",
    ):
        _validate_evidence_sha256(canonical[name], field_name=name)

    typed_output = canonical["typed_output"]
    if type(typed_output) is not dict:
        raise ValueError("typed_output must be an exact JSON object")
    output_bytes = _canonical_evidence_bytes(typed_output)
    if len(output_bytes) > MAX_STRUCTURED_OUTPUT_EVIDENCE_UTF8_BYTES:
        raise ValueError("typed_output exceeds the evidence bound")
    output_utf8_bytes = canonical["typed_output_utf8_bytes"]
    if type(output_utf8_bytes) is not int or output_utf8_bytes != len(output_bytes):
        raise ValueError("typed_output_utf8_bytes does not authenticate the output")
    if canonical["typed_output_sha256"] != hashlib.sha256(output_bytes).hexdigest():
        raise ValueError("typed_output_sha256 does not authenticate the output")

    supplied_sha256 = canonical["output_evidence_sha256"]
    authenticated = dict(canonical)
    del authenticated["output_evidence_sha256"]
    expected_sha256 = hashlib.sha256(
        _STRUCTURED_OUTPUT_EVIDENCE_DOMAIN + _canonical_evidence_bytes(authenticated)
    ).hexdigest()
    if supplied_sha256 != expected_sha256:
        raise ValueError("output_evidence_sha256 does not authenticate the record")

    if request_evidence is not None:
        request_record = validate_structured_generation_request_evidence_record(
            request_evidence
        )
        joined_fields = (
            ("call_id", "call_id"),
            ("operation", "operation"),
            ("output_tool_name", "output_tool_name"),
            ("output_schema_sha256", "output_schema_sha256"),
            ("request_evidence_sha256", "request_evidence_sha256"),
        )
        if any(
            canonical[output_name] != request_record[request_name]
            for output_name, request_name in joined_fields
        ):
            raise ValueError("output evidence does not join its request evidence")
    return canonical


def structured_generation_outcome_record(
    outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
) -> dict[str, object]:
    """Project a terminal outcome to sanitized, JSON-compatible telemetry.

    The projection deliberately excludes prompts and typed output content.  A
    successful row retains the provider identity, usage, exact reported cost,
    latency, and response identifier that would otherwise be lost if a later
    experiment gate rejects the response. Schema version 2 added bounded,
    sanitized failure evidence to each attempt; schema version 3 added bounded
    structured-output diagnostics. Schema version 4 added the closed request
    variant and SHA-256 for each prepared provider attempt. Schema version 5
    binds deterministic physical-attempt identity and an optional closed stream
    timeout phase. Schema version 6 added a finite canonical provider-error
    code and a domain-separated fingerprint of a value-free redacted HTTP
    error envelope. Schema version 8 added bounded, privacy-safe exception
    provenance for otherwise-unknown adapter failures. The successful response
    projection is otherwise unchanged.
    """

    if type(outcome) is not LLMTaskOutcome:
        raise TypeError("outcome must be an exact LLMTaskOutcome")
    LLMTaskOutcome.__post_init__(outcome)

    attempts: list[dict[str, object]] = []
    for attempt in outcome.telemetry.attempts:
        classification = attempt.classification
        failure = None if classification is None else classification.sanitized_failure
        attempts.append(
            {
                "attempt_number": attempt.attempt_number,
                "status": attempt.status.value,
                "wait_time_ns": attempt.wait_time_ns,
                "service_time_ns": attempt.service_time_ns,
                "will_retry": attempt.will_retry,
                "policy_backoff_ns": attempt.policy_backoff_ns,
                "retry_after_ns": attempt.retry_after_ns,
                "scheduled_delay_ns": attempt.scheduled_delay_ns,
                "error_type": attempt.error_type,
                "request_evidence": (
                    None
                    if attempt.request_evidence is None
                    else {
                        "variant": attempt.request_evidence.variant.value,
                        "prompt_sha256": attempt.request_evidence.prompt_sha256,
                        "provider_attempt_id": (
                            None
                            if attempt.request_evidence.provider_attempt_id is None
                            else attempt.request_evidence.provider_attempt_id.value
                        ),
                    }
                ),
                "classification": (
                    None
                    if classification is None
                    else {
                        "disposition": classification.disposition.value,
                        "reason": classification.reason.value,
                    }
                ),
                "failure": (
                    None
                    if failure is None
                    else {
                        "kind": failure.kind,
                        "retryable": failure.retryable,
                        "safe_message": failure.safe_message,
                        "status_code": failure.status_code,
                        "retry_after_seconds": failure.retry_after_seconds,
                        "provider_error_code": (
                            None
                            if failure.provider_error_code is None
                            else failure.provider_error_code.value
                        ),
                        "provider_error_envelope_sha256": (
                            failure.provider_error_envelope_sha256
                        ),
                        "exception_provenance": (
                            None
                            if failure.exception_provenance is None
                            else {
                                "truncated": (
                                    failure.exception_provenance.truncated
                                ),
                                "nodes": [
                                    {
                                        "parent_index": node.parent_index,
                                        "link": node.link.value,
                                        "family": node.family.value,
                                        "type_identity_sha256": (
                                            node.type_identity_sha256
                                        ),
                                    }
                                    for node in failure.exception_provenance.nodes
                                ],
                            }
                        ),
                        "stream_timeout_phase": (
                            None
                            if failure.stream_timeout_phase is None
                            else failure.stream_timeout_phase.value
                        ),
                        "output_failure_mode": (
                            None
                            if failure.output_failure_mode is None
                            else failure.output_failure_mode.value
                        ),
                        "validation_issues": [
                            {
                                "category": issue.category.value,
                                "location": list(issue.location),
                                "reason_code": (
                                    None
                                    if issue.reason_code is None
                                    else issue.reason_code.value
                                ),
                            }
                            for issue in failure.validation_issues
                        ],
                    }
                ),
            }
        )

    response_record: dict[str, object] | None = None
    if outcome.status is TaskOutcomeStatus.SUCCEEDED:
        response = outcome.response
        if type(response) is not StructuredGenerationResponse:
            raise TypeError("successful outcome has no structured response")
        StructuredGenerationResponse.__post_init__(response)
        response_record = {
            "requested_model": response.requested_model,
            "resolved_model": response.resolved_model,
            "resolved_provider": response.resolved_provider,
            "provider_response_id": response.provider_response_id,
            "finish_reason": response.finish_reason,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "cache_read_tokens": response.cache_read_tokens,
            "cache_write_tokens": response.cache_write_tokens,
            "cost_usd": (None if response.cost_usd is None else str(response.cost_usd)),
            "latency_ns": response.latency_ns,
        }

    return {
        "schema_version": STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION,
        "task_id": outcome.telemetry.task_id,
        "status": outcome.status.value,
        "cancellation_reason": (
            None
            if outcome.cancellation_reason is None
            else outcome.cancellation_reason.value
        ),
        "queue_time_ns": outcome.telemetry.queue_time_ns,
        "service_time_ns": outcome.telemetry.service_time_ns,
        "total_time_ns": outcome.telemetry.total_time_ns,
        "attempts": attempts,
        "response": response_record,
    }


@runtime_checkable
class StructuredAttemptRequestPolicy(Protocol):
    """Derive one attempt request from bounded queue context."""

    def request_for_attempt(
        self,
        request: StructuredGenerationRequest[OutputT],
        *,
        context: LLMAttemptContext,
    ) -> "PreparedStructuredAttemptRequest[OutputT]": ...


@dataclass(frozen=True, slots=True)
class PreparedStructuredAttemptRequest(Generic[OutputT]):
    """Exact structured request paired with evidence derived from its prompt."""

    request: StructuredGenerationRequest[OutputT]
    evidence: AttemptRequestEvidence

    def __post_init__(self) -> None:
        if type(self.request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(self.request)
        if type(self.evidence) is not AttemptRequestEvidence:
            raise TypeError("evidence must be an AttemptRequestEvidence")
        expected = hashlib.sha256(
            self.request.prompt.encode("utf-8", errors="strict")
        ).hexdigest()
        if self.evidence.prompt_sha256 != expected:
            raise ValueError("request evidence does not match the exact prompt")
        if self.evidence.provider_attempt_id != self.request.provider_attempt_id:
            raise ValueError(
                "request evidence and prepared request attempt identities differ"
            )


def _provider_attempt_id(
    *,
    context: LLMAttemptContext,
    prompt_sha256: str,
) -> ProviderAttemptId:
    """Derive a content-free stable identity for one physical queue attempt."""

    fields = (
        context.task_id.encode("utf-8", errors="strict"),
        str(context.attempt_number).encode("ascii"),
        prompt_sha256.encode("ascii", errors="strict"),
    )
    digest = hashlib.sha256(_PROVIDER_ATTEMPT_ID_DOMAIN)
    for field in fields:
        digest.update(len(field).to_bytes(8, "big"))
        digest.update(field)
    return ProviderAttemptId(f"provider_attempt_{digest.hexdigest()}")


class ExactPayloadAttemptPolicy:
    """Replay the original structured request byte-for-byte on every attempt.

    This policy is useful for controlled replicates and transport-only retries
    where changing the prompt after a provider or validation failure would
    change the treatment.  Retry admission remains owned by the queue and its
    classifier; this policy only guarantees that every admitted attempt uses
    the original prompt, output type, tool contract, and generation settings.
    """

    def request_for_attempt(
        self,
        request: StructuredGenerationRequest[OutputT],
        *,
        context: LLMAttemptContext,
    ) -> PreparedStructuredAttemptRequest[OutputT]:
        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(request)
        if type(context) is not LLMAttemptContext:
            raise TypeError("context must be an exact LLMAttemptContext")
        LLMAttemptContext.__post_init__(context)
        return PreparedStructuredAttemptRequest(
            request=request,
            evidence=AttemptRequestEvidence(
                variant=AttemptRequestVariant.ORIGINAL,
                prompt_sha256=hashlib.sha256(
                    request.prompt.encode("utf-8", errors="strict")
                ).hexdigest(),
            ),
        )


class _SchemaRequiredPathMapUnavailable(ValueError):
    """The local schema cannot yield one bounded, complete required-path map."""


def _local_schema_reference(
    root: dict[str, Any],
    reference: object,
) -> dict[str, Any] | bool:
    if type(reference) is not str or not reference.startswith("#/"):
        raise _SchemaRequiredPathMapUnavailable(
            "schema-repair path maps permit only local references"
        )
    current: object = root
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if type(current) is not dict or token not in current:
            raise _SchemaRequiredPathMapUnavailable(
                "schema-repair path map contains an unresolved reference"
            )
        current = current[token]
    if type(current) not in {dict, bool}:
        raise _SchemaRequiredPathMapUnavailable(
            "schema-repair reference does not resolve to a schema"
        )
    return current


def _json_pointer(path: tuple[str, ...]) -> str:
    return "/" + "/".join(token.replace("~", "~0").replace("/", "~1") for token in path)


def _required_field_paths(output_type: type[Any]) -> tuple[str, ...]:
    """Enumerate all reachable ``required`` properties without partial output.

    The map is derived solely from the trusted local Pydantic output type. If a
    recursive, malformed, or over-large schema cannot be represented in the
    fixed repair budget, callers retain the original request instead of giving
    the model an incomplete and therefore misleading field list.
    """

    try:
        root = output_type.model_json_schema()
    except Exception as error:
        raise _SchemaRequiredPathMapUnavailable(
            "local output schema generation failed"
        ) from error
    if type(root) is not dict:
        raise _SchemaRequiredPathMapUnavailable("local output schema must be an object")

    required_paths: set[tuple[str, ...]] = set()
    visited_nodes = 0

    def visit(
        schema: object,
        path: tuple[str, ...],
        active_references: tuple[str, ...] = (),
    ) -> None:
        nonlocal visited_nodes
        if type(schema) is bool:
            return
        if type(schema) is not dict:
            raise _SchemaRequiredPathMapUnavailable(
                "local output schema contains a malformed child"
            )
        visited_nodes += 1
        if visited_nodes > MAX_SCHEMA_REPAIR_SCHEMA_NODES:
            raise _SchemaRequiredPathMapUnavailable(
                "local output schema exceeds the node bound"
            )

        if "$ref" in schema:
            reference = schema["$ref"]
            if type(reference) is not str or reference in active_references:
                raise _SchemaRequiredPathMapUnavailable(
                    "recursive or malformed local output reference"
                )
            visit(
                _local_schema_reference(root, reference),
                path,
                (*active_references, reference),
            )
            siblings = {key: value for key, value in schema.items() if key != "$ref"}
            if siblings:
                visit(siblings, path, active_references)
            return

        properties = schema.get("properties", {})
        if type(properties) is not dict:
            raise _SchemaRequiredPathMapUnavailable(
                "local output object properties are malformed"
            )
        required = schema.get("required", [])
        if type(required) is not list or not all(
            type(name) is str for name in required
        ):
            raise _SchemaRequiredPathMapUnavailable(
                "local output required fields are malformed"
            )
        for name in required:
            required_paths.add((*path, name))
            if len(required_paths) > MAX_SCHEMA_REPAIR_REQUIRED_PATHS:
                raise _SchemaRequiredPathMapUnavailable(
                    "local output schema exceeds the required-path bound"
                )
        for name, child in properties.items():
            if type(name) is not str:
                raise _SchemaRequiredPathMapUnavailable(
                    "local output property name is malformed"
                )
            visit(child, (*path, name), active_references)

        items = schema.get("items")
        if type(items) is list:
            for index, child in enumerate(items):
                visit(child, (*path, str(index)), active_references)
        elif items is not None:
            visit(items, (*path, "*"), active_references)
        prefix_items = schema.get("prefixItems")
        if prefix_items is not None:
            if type(prefix_items) is not list:
                raise _SchemaRequiredPathMapUnavailable(
                    "local output tuple items are malformed"
                )
            for index, child in enumerate(prefix_items):
                visit(child, (*path, str(index)), active_references)

        for keyword in ("allOf", "anyOf", "oneOf"):
            branches = schema.get(keyword)
            if branches is None:
                continue
            if type(branches) is not list:
                raise _SchemaRequiredPathMapUnavailable(
                    "local output composition is malformed"
                )
            for branch in branches:
                visit(branch, path, active_references)
        for keyword in ("if", "then", "else", "not"):
            branch = schema.get(keyword)
            if branch is not None:
                visit(branch, path, active_references)

        dependent_schemas = schema.get("dependentSchemas")
        if dependent_schemas is not None:
            if type(dependent_schemas) is not dict:
                raise _SchemaRequiredPathMapUnavailable(
                    "local output dependent schemas are malformed"
                )
            for branch in dependent_schemas.values():
                visit(branch, path, active_references)

        for keyword in ("patternProperties",):
            dynamic_schemas = schema.get(keyword)
            if dynamic_schemas is None:
                continue
            if type(dynamic_schemas) is not dict:
                raise _SchemaRequiredPathMapUnavailable(
                    "local output dynamic properties are malformed"
                )
            for child in dynamic_schemas.values():
                visit(child, (*path, "*"), active_references)
        for keyword in ("additionalProperties", "unevaluatedProperties"):
            child = schema.get(keyword)
            if type(child) is dict:
                visit(child, (*path, "*"), active_references)
        for keyword in ("contains", "unevaluatedItems"):
            child = schema.get(keyword)
            if child is not None:
                visit(child, (*path, "*"), active_references)

    visit(root, ())
    return tuple(sorted(_json_pointer(path) for path in required_paths))


def _schema_repair_prompt_lineage(
    request: StructuredGenerationRequest[Any],
) -> StructuredPromptLineage:
    upstream = request.prompt_lineage or identity_prompt_lineage(request.prompt)
    definition_record = {
        "schema_repair_policy_sha256": SCHEMA_REPAIR_POLICY_MANIFEST.policy_sha256,
        "upstream_renderer_id": upstream.renderer_id,
        "upstream_renderer_revision": upstream.renderer_revision,
        "upstream_renderer_definition_sha256": (upstream.renderer_definition_sha256),
    }
    definition_sha256 = hashlib.sha256(
        b"agent-evolve:schema-repair-prompt-renderer:v1\x00"
        + _canonical_evidence_bytes(definition_record)
    ).hexdigest()
    return StructuredPromptLineage(
        semantic_prompt_sha256=upstream.semantic_prompt_sha256,
        renderer_id=SCHEMA_REPAIR_PROMPT_RENDERER_ID,
        renderer_revision=SCHEMA_REPAIR_PROMPT_RENDERER_REVISION,
        renderer_definition_sha256=definition_sha256,
    )


def _repair_literal_constraint_block(
    request: StructuredGenerationRequest[Any],
    failure: SanitizedAttemptFailure,
) -> str:
    """Render only trusted, provider-visible closed sets relevant to the failure."""

    if not request.repair_literal_sets:
        return ""
    literal_failure = any(
        issue.category is ValidationIssueCategory.LITERAL_OR_ENUM
        or issue.reason_code
        is ValidationIssueReasonCode.FINITE_OPTION_OUT_OF_CONTRACT
        for issue in failure.validation_issues
    )
    if not literal_failure:
        return ""
    lines = [
        "Exact allowed string literals from the trusted local output contract "
        "(copy byte-for-byte; never synthesize or truncate an identifier):\n"
    ]
    for constraint in request.repair_literal_sets:
        path = _json_pointer(constraint.field_path)
        literals = json.dumps(
            constraint.allowed_literals,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        lines.append(f"- {path}={literals}\n")
    return "".join(lines)


class SchemaRepairAttemptPolicy:
    """Add bounded schema guidance only after a sanitized output failure."""

    manifest = SCHEMA_REPAIR_POLICY_MANIFEST

    @staticmethod
    def _location_text(location: tuple[str, ...]) -> str:
        return ".".join(location[:4])

    @staticmethod
    def _prepared(
        request: StructuredGenerationRequest[OutputT],
        variant: AttemptRequestVariant,
    ) -> PreparedStructuredAttemptRequest[OutputT]:
        evidence = AttemptRequestEvidence(
            variant=variant,
            prompt_sha256=hashlib.sha256(
                request.prompt.encode("utf-8", errors="strict")
            ).hexdigest(),
        )
        return PreparedStructuredAttemptRequest(request=request, evidence=evidence)

    def request_for_attempt(
        self,
        request: StructuredGenerationRequest[OutputT],
        *,
        context: LLMAttemptContext,
    ) -> PreparedStructuredAttemptRequest[OutputT]:
        failure = context.active_output_failure
        if (
            failure is None
            or not failure.retryable
            or failure.kind != GenerationFailureKind.OUTPUT_INVALID.value
        ):
            return self._prepared(request, AttemptRequestVariant.ORIGINAL)

        mode = failure.output_failure_mode or (
            StructuredOutputFailureMode.TYPED_OUTPUT_CONTRACT
        )
        try:
            required_paths = _required_field_paths(request.output_type)
        except _SchemaRequiredPathMapUnavailable:
            return self._prepared(request, AttemptRequestVariant.ORIGINAL)
        required_paths_json = json.dumps(
            required_paths,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        # Output-token pressure can surface as schema validation (for example,
        # a truncated object missing late fields), not only as the provider's
        # explicit incomplete-tool-call category. Keep this bounded guidance
        # active for every output-invalid repair without weakening the schema.
        completion_guidance = " Keep every field concise so the tool call completes."
        # A partitioned queue exposes semantic retry usage independently of
        # physical attempts.  Deriving escalation from that ledger keeps a
        # repair request byte-identical across intervening 429/5xx/timeouts.
        # The legacy fallback preserves behavior for callers without the new
        # budget contract.
        if context.retry_budget_usage is not None:
            repair_pass = min(
                2,
                max(1, context.retry_budget_usage.output_invalid_retries),
            )
        else:
            repair_pass = (
                2
                if context.attempt_number >= 3
                and context.previous_failure is not None
                and context.previous_failure.kind
                == GenerationFailureKind.OUTPUT_INVALID.value
                else 1
            )
        escalation_guidance = (
            ""
            if repair_pass == 1
            else (
                " FINAL BOUNDED REPAIR PASS: rebuild the complete tool call "
                "independently, then check every constrained string by exact "
                "equality against the trusted lists before emitting it."
            )
        )
        literal_constraint_block = _repair_literal_constraint_block(request, failure)

        def render(issue_lines: list[str]) -> str:
            issue_block = (
                "Validation issues:\n" + "".join(issue_lines) if issue_lines else ""
            )
            return _SCHEMA_REPAIR_TEMPLATE.format(
                policy_version=SCHEMA_REPAIR_POLICY_VERSION,
                failure_mode=mode.value,
                repair_pass=repair_pass,
                required_paths_json=required_paths_json,
                issue_block=issue_block,
                literal_constraint_block=literal_constraint_block,
                output_tool_name=request.output_tool_name,
                completion_guidance=completion_guidance,
                escalation_guidance=escalation_guidance,
            )

        if len(render([]).encode("utf-8", errors="strict")) > (
            MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES
        ):
            return self._prepared(request, AttemptRequestVariant.ORIGINAL)
        issue_lines: list[str] = []
        for issue in failure.validation_issues:
            reason = (
                ""
                if issue.reason_code is None
                else f"; reason={issue.reason_code.value}"
            )
            guidance = (
                ""
                if issue.reason_code is None
                else (
                    " "
                    + _SEMANTIC_REPAIR_GUIDANCE.get(
                        issue.reason_code,
                        _DEFAULT_SEMANTIC_REPAIR_GUIDANCE,
                    )
                )
            )
            line = (
                f"- {issue.category.value} at "
                f"{self._location_text(issue.location)}{reason}."
                f"{guidance}\n"
            )
            candidate = render([*issue_lines, line])
            if (
                len(candidate.encode("utf-8", errors="strict"))
                > MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES
            ):
                break
            issue_lines.append(line)
        suffix = render(issue_lines)
        if (
            len(suffix.encode("utf-8", errors="strict"))
            > MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES
        ):
            raise AssertionError("schema repair suffix exceeded its static bound")
        repaired_prompt = request.prompt + suffix
        if (
            len(repaired_prompt.encode("utf-8", errors="strict"))
            > MAX_PROMPT_UTF8_BYTES
        ):
            # A maximal original request remains a valid provider attempt. Do
            # not turn its retry into a local request-construction failure.
            return self._prepared(request, AttemptRequestVariant.ORIGINAL)
        repaired = replace(
            request,
            prompt=repaired_prompt,
            prompt_lineage=_schema_repair_prompt_lineage(request),
        )
        return self._prepared(repaired, AttemptRequestVariant.SCHEMA_REPAIR_V4)


class ExactTransportSchemaRepairAttemptPolicy:
    """Replay transport failures exactly and fail closed on repair derivation.

    The ordinary :class:`SchemaRepairAttemptPolicy` deliberately falls back to
    the original request when it cannot derive a bounded, complete repair
    suffix.  That is convenient in general-purpose applications, but it would
    turn a preregistered schema-repair attempt into an unlabelled additional
    sample.  This experiment-facing policy therefore requires the authenticated
    repair variant whenever an output-invalid failure activated repair.  All
    other admitted retries preserve the original request exactly.
    """

    manifest = SCHEMA_REPAIR_POLICY_MANIFEST

    def __init__(self) -> None:
        self._exact = ExactPayloadAttemptPolicy()
        self._repair = SchemaRepairAttemptPolicy()

    def request_for_attempt(
        self,
        request: StructuredGenerationRequest[OutputT],
        *,
        context: LLMAttemptContext,
    ) -> PreparedStructuredAttemptRequest[OutputT]:
        if context.active_output_failure is None:
            return self._exact.request_for_attempt(request, context=context)
        prepared = self._repair.request_for_attempt(request, context=context)
        if prepared.evidence.variant is not AttemptRequestVariant.SCHEMA_REPAIR_V4:
            raise StructuredGenerationError(
                kind=GenerationFailureKind.INVALID_REQUEST,
                retryable=False,
                safe_message=(
                    "bounded schema-repair guidance could not be derived locally"
                ),
            )
        return prepared


class StructuredGenerationExecutor:
    """Execute exactly one structured-provider attempt for the queue."""

    def __init__(
        self,
        generator: StructuredGenerator,
        *,
        attempt_request_policy: StructuredAttemptRequestPolicy | None = None,
    ) -> None:
        if not isinstance(generator, StructuredGenerator):
            raise TypeError("generator must implement StructuredGenerator")
        if attempt_request_policy is None:
            attempt_request_policy = SchemaRepairAttemptPolicy()
        if not isinstance(attempt_request_policy, StructuredAttemptRequestPolicy):
            raise TypeError(
                "attempt_request_policy must implement StructuredAttemptRequestPolicy"
            )
        self.generator = generator
        self.attempt_request_policy = attempt_request_policy

    def prepare_attempt(
        self,
        request: StructuredGenerationRequest[OutputT],
        *,
        context: LLMAttemptContext,
    ) -> PreparedLLMAttempt[StructuredGenerationResponse[OutputT]]:
        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(request)
        if type(context) is not LLMAttemptContext:
            raise TypeError("context must be an exact LLMAttemptContext")
        LLMAttemptContext.__post_init__(context)

        prepared_request = self.attempt_request_policy.request_for_attempt(
            request,
            context=context,
        )
        if type(prepared_request) is not PreparedStructuredAttemptRequest:
            raise TypeError("attempt request policy returned an invalid value")
        PreparedStructuredAttemptRequest.__post_init__(prepared_request)
        provider_attempt_id = _provider_attempt_id(
            context=context,
            prompt_sha256=prepared_request.evidence.prompt_sha256,
        )
        attempt_request = replace(
            prepared_request.request,
            provider_attempt_id=provider_attempt_id,
        )
        request_evidence = replace(
            prepared_request.evidence,
            provider_attempt_id=provider_attempt_id,
        )
        return PreparedLLMAttempt(
            execute_once=partial(
                self._execute_prepared,
                attempt_request,
            ),
            request_evidence=request_evidence,
        )

    async def execute(
        self,
        request: StructuredGenerationRequest[OutputT],
        *,
        context: LLMAttemptContext,
    ) -> StructuredGenerationResponse[OutputT]:
        prepared = self.prepare_attempt(request, context=context)
        return await prepared.execute_once()

    async def _execute_prepared(
        self,
        attempt_request: StructuredGenerationRequest[OutputT],
    ) -> StructuredGenerationResponse[OutputT]:

        # The queue still owns whether this attempt exists. The policy only
        # derives its request; the provider boundary never retries or sleeps.
        response = await self.generator.generate_once(attempt_request)
        if type(response) is not StructuredGenerationResponse:
            raise TypeError(
                "structured generator must return an exact StructuredGenerationResponse"
            )
        StructuredGenerationResponse.__post_init__(response)
        if type(response.value) is not attempt_request.output_type:
            raise TypeError("structured response value violates output_type")
        return response


def _retry_after(seconds: float | None) -> RetryAfter | None:
    if seconds is None:
        return None
    # StructuredGenerationError already establishes finite, non-negative input.
    # Decimal(str(...)) plus ceiling prevents a positive sub-nanosecond server
    # delay from being shortened to zero.
    nanoseconds = int(
        (Decimal(str(seconds)) * NANOSECONDS_PER_SECOND).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    return RetryAfter(
        delay_ns=min(nanoseconds, _MAX_RETRY_AFTER_NS),
        source=RetryAfterSource.DELAY_SECONDS,
    )


class StructuredGenerationRetryClassifier:
    """Translate sanitized structured failures into the queue's closed domain."""

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        if type(context) is not LLMAttemptContext:
            raise TypeError("context must be an exact LLMAttemptContext")
        LLMAttemptContext.__post_init__(context)

        if isinstance(error, TransportAbortedTimeoutError):
            return RetryClassification(
                disposition=RetryDisposition.FAIL,
                reason=RetryReason.TIMEOUT,
                sanitized_failure=SanitizedAttemptFailure(
                    kind="timeout",
                    retryable=False,
                    safe_message=(
                        "provider attempt exceeded its hard deadline; the owned "
                        "transport was closed and the attempt was drained"
                    ),
                ),
            )
        if not isinstance(error, StructuredGenerationError):
            if isinstance(error, TimeoutError):
                return RetryClassification(
                    disposition=RetryDisposition.RETRY,
                    reason=RetryReason.TIMEOUT,
                )
            return RetryClassification(
                disposition=RetryDisposition.FAIL,
                reason=RetryReason.INTERNAL,
            )

        sanitized_failure = SanitizedAttemptFailure(
            kind=error.kind.value,
            retryable=error.retryable,
            safe_message=error.safe_message,
            status_code=error.status_code,
            retry_after_seconds=error.retry_after_seconds,
            output_failure_mode=error.output_failure_mode,
            validation_issues=error.validation_issues,
            provider_error_code=error.provider_error_code,
            provider_error_envelope_sha256=(error.provider_error_envelope_sha256),
            exception_provenance=error.exception_provenance,
            stream_timeout_phase=(
                error.phase
                if isinstance(
                    error,
                    (
                        StructuredStreamTimeoutError,
                        StructuredStreamCleanupTimeoutError,
                    ),
                )
                else None
            ),
        )

        disposition = (
            RetryDisposition.RETRY if error.retryable else RetryDisposition.FAIL
        )
        if error.kind is GenerationFailureKind.RATE_LIMITED:
            reason = RetryReason.RATE_LIMIT
        elif error.kind is GenerationFailureKind.TIMEOUT:
            reason = RetryReason.TIMEOUT
        elif error.kind is GenerationFailureKind.OUTPUT_INVALID:
            reason = RetryReason.OUTPUT_INVALID
        elif error.kind is GenerationFailureKind.PROVIDER_UNAVAILABLE:
            reason = RetryReason.TRANSIENT
        elif error.retryable:
            reason = RetryReason.TRANSIENT
        else:
            reason = RetryReason.PERMANENT

        return RetryClassification(
            disposition=disposition,
            reason=reason,
            retry_after=(
                _retry_after(error.retry_after_seconds)
                if disposition is RetryDisposition.RETRY
                else None
            ),
            sanitized_failure=sanitized_failure,
        )


class TransportOnlyStructuredGenerationRetryClassifier:
    """Retry transient transport conditions but never invalid model output.

    The provider adapter may label incomplete or schema-invalid model output
    retryable for production repair workflows.  Controlled experiments often
    need those failures to be terminal so that a physical retry cannot become
    an unplanned extra sample. HTTP status is authoritative: only 408, 429,
    and 500--599 may retry. Any other 4xx carrying a misleading transient kind
    or ``retryable=True`` remains terminal. Connection failures and
    cooperative stream-liveness timeouts have no HTTP status and retain the
    base classifier's retry behavior.
    """

    def __init__(self) -> None:
        self._base = StructuredGenerationRetryClassifier()

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._base.classify(error, context=context)
        transport_condition = False
        if isinstance(error, StructuredGenerationError):
            if error.status_code is not None:
                transport_condition = (
                    error.status_code in {408, 429} or 500 <= error.status_code <= 599
                )
            elif isinstance(error, StructuredStreamCleanupTimeoutError):
                transport_condition = False
            else:
                # Status-free TIMEOUT covers cooperative stream-liveness and
                # typed transport timeouts. Status-free PROVIDER_UNAVAILABLE
                # is the adapter's closed representation of a typed
                # connection failure. RATE_LIMITED is deliberately excluded:
                # the admitted representation of rate limiting is HTTP 429.
                transport_condition = error.kind in {
                    GenerationFailureKind.TIMEOUT,
                    GenerationFailureKind.PROVIDER_UNAVAILABLE,
                }
        ordinary_timeout = isinstance(error, TimeoutError) and not isinstance(
            error, TransportAbortedTimeoutError
        )
        if classified.disposition is RetryDisposition.RETRY and not (
            transport_condition or ordinary_timeout
        ):
            return RetryClassification(
                disposition=RetryDisposition.FAIL,
                reason=classified.reason,
                sanitized_failure=classified.sanitized_failure,
            )
        return classified


class NonRepeatingStreamTransportRetryClassifier:
    """Retry transient pre-response transport failures, never an owned stream.

    Once a streamed attempt has crossed the provider boundary, a first-event or
    idle-liveness timeout has an uncertain provider-side completion and billing
    state.  Recovery/replay experiments therefore need a stricter policy than
    :class:`TransportOnlyStructuredGenerationRetryClassifier`: HTTP 408/429/5xx
    and typed connection failures may still retry, while every supervised
    stream timeout is terminal even when cancellation drained cleanly.
    """

    def __init__(self) -> None:
        self._transport_only = TransportOnlyStructuredGenerationRetryClassifier()

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._transport_only.classify(error, context=context)
        if isinstance(error, StructuredStreamTimeoutError) and (
            classified.disposition is RetryDisposition.RETRY
        ):
            return RetryClassification(
                disposition=RetryDisposition.FAIL,
                reason=classified.reason,
                sanitized_failure=classified.sanitized_failure,
            )
        return classified


class OpaqueHTTP400OnceRetryClassifier:
    """Retry one evidence-bearing but otherwise opaque HTTP 400 exactly once.

    Some OpenRouter routes occasionally reject a byte-valid request before a
    stream exists while returning only an opaque HTTP-400 envelope.  A later
    exact-payload replay can then succeed.  This policy is deliberately much
    narrower than treating HTTP 400 as transient:

    * only the first attempt is eligible;
    * the failure must be ``invalid_request`` with status 400;
    * a redacted envelope fingerprint must exist, while no typed provider code,
      output diagnostic, validation issue, or retry-after hint may exist; and
    * all ordinary non-repeating-stream transport rules remain unchanged.

    Typed/actionable 4xx responses therefore remain terminal.  Composition
    roots must also pair this classifier with an exact-payload attempt policy
    when request identity across the retry matters.
    """

    def __init__(self) -> None:
        self._non_repeating = NonRepeatingStreamTransportRetryClassifier()

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._non_repeating.classify(error, context=context)
        if (
            classified.disposition is RetryDisposition.FAIL
            and context.attempt_number == 1
            and context.previous_failure is None
            and isinstance(error, StructuredGenerationError)
            and error.kind is GenerationFailureKind.INVALID_REQUEST
            and error.status_code == 400
            and error.provider_error_code is None
            and error.provider_error_envelope_sha256 is not None
            and error.retry_after_seconds is None
            and error.output_failure_mode is None
            and not error.validation_issues
        ):
            return RetryClassification(
                disposition=RetryDisposition.RETRY,
                reason=RetryReason.TRANSIENT,
                sanitized_failure=classified.sanitized_failure,
            )
        return classified


class BoundedOpaqueHTTP400RetryClassifier:
    """Retry an identical opaque pre-stream HTTP 400 to the task budget.

    A provider can transiently reject several byte-identical, contract-valid
    requests before accepting the next replay.  This policy remains narrower
    than treating HTTP 400 as generally retryable:

    * the response must be an ``invalid_request`` status 400 with a redacted
      envelope fingerprint and no typed provider code or validation detail;
    * every preceding failure in the replay chain must have the same envelope
      fingerprint and the same closed failure shape; and
    * the queue's immutable ``LLMTask.max_attempts`` remains the hard bound.

    Composition roots must pair this classifier with an exact-payload attempt
    policy.  Actionable 4xx responses, post-content stream failures, and a
    changed opaque envelope remain terminal.
    """

    def __init__(self) -> None:
        self._non_repeating = NonRepeatingStreamTransportRetryClassifier()

    @staticmethod
    def _is_opaque_http_400(
        error: StructuredGenerationError,
    ) -> bool:
        return (
            error.kind is GenerationFailureKind.INVALID_REQUEST
            and error.status_code == 400
            and error.provider_error_code is None
            and error.provider_error_envelope_sha256 is not None
            and error.retry_after_seconds is None
            and error.output_failure_mode is None
            and not error.validation_issues
        )

    @staticmethod
    def _continues_same_chain(
        *,
        error: StructuredGenerationError,
        context: LLMAttemptContext,
    ) -> bool:
        if context.attempt_number == 1:
            return context.previous_failure is None
        previous = context.previous_failure
        return (
            previous is not None
            and previous.kind
            == GenerationFailureKind.INVALID_REQUEST.value
            and previous.status_code == 400
            and previous.provider_error_code is None
            and previous.provider_error_envelope_sha256
            == error.provider_error_envelope_sha256
            and previous.retry_after_seconds is None
            and previous.output_failure_mode is None
            and not previous.validation_issues
            and context.active_output_failure is None
        )

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._non_repeating.classify(
            error,
            context=context,
        )
        if not (
            classified.disposition is RetryDisposition.FAIL
            and isinstance(error, StructuredGenerationError)
            and self._is_opaque_http_400(error)
            and self._continues_same_chain(
                error=error,
                context=context,
            )
        ):
            return classified
        return RetryClassification(
            disposition=RetryDisposition.RETRY,
            reason=RetryReason.TRANSIENT,
            sanitized_failure=classified.sanitized_failure,
        )


class OpaqueHTTP400AndSchemaRepairOnceRetryClassifier:
    """Combine exact opaque-400 recovery with one strict output repair.

    Transport behavior is inherited unchanged from
    :class:`OpaqueHTTP400OnceRetryClassifier`, including terminal owned-stream
    timeouts and terminal typed/actionable 4xx responses.  A retryable typed
    output failure receives one repair opportunity only.  The queue's
    ``active_output_failure`` marker prevents a second invalid output from
    becoming another model sample.
    """

    def __init__(self) -> None:
        self._opaque_http_400 = OpaqueHTTP400OnceRetryClassifier()
        self._structured = StructuredGenerationRetryClassifier()

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._opaque_http_400.classify(error, context=context)
        if not (
            isinstance(error, StructuredGenerationError)
            and error.kind is GenerationFailureKind.OUTPUT_INVALID
            and error.retryable
            and context.active_output_failure is None
        ):
            return classified
        repair = self._structured.classify(error, context=context)
        if (
            repair.disposition is RetryDisposition.RETRY
            and repair.reason is RetryReason.OUTPUT_INVALID
        ):
            return repair
        return classified


class OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier:
    """Combine opaque-400 replay with repair resampling to the task budget.

    A typed output failure does not expose a valid candidate and therefore is
    not an optimization sample.  Retrying it cannot select among candidate
    outcomes.  This classifier admits another schema-repair attempt whenever
    the failure remains typed and retryable; the queue's immutable
    ``LLMTask.max_attempts`` is the sole hard bound.  Every physical attempt,
    exact repair prompt, and terminal response remains separately recorded.

    Transport behavior is inherited unchanged from
    :class:`OpaqueHTTP400OnceRetryClassifier`: an opaque pre-stream HTTP 400
    may replay once, owned-stream timeouts remain terminal, and actionable 4xx
    responses never retry.
    """

    def __init__(self) -> None:
        self._opaque_http_400 = OpaqueHTTP400OnceRetryClassifier()
        self._structured = StructuredGenerationRetryClassifier()

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._opaque_http_400.classify(error, context=context)
        if not (
            isinstance(error, StructuredGenerationError)
            and error.kind is GenerationFailureKind.OUTPUT_INVALID
            and error.retryable
        ):
            return classified
        repair = self._structured.classify(error, context=context)
        if (
            repair.disposition is RetryDisposition.RETRY
            and repair.reason is RetryReason.OUTPUT_INVALID
        ):
            return repair
        return classified


class FirstEventResilientBoundedSchemaRepairRetryClassifier:
    """Recover a content-blind first-event timeout inside one logical sample.

    This policy preserves the opaque-HTTP-400 and bounded schema-repair
    semantics of :class:`OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier`.
    It additionally retries a supervised ``FIRST_EVENT`` timeout because no
    provider content was observed and therefore no candidate outcome can be
    selected or discarded.  ``IDLE`` and ``ABSOLUTE`` timeouts remain
    terminal because they follow observable stream progress, as do cleanup
    timeouts whose underlying attempt may still be running.

    The queue's immutable attempt and partitioned retry budgets remain the
    hard bounds.  Composition roots must pair this classifier with an exact
    transport/schema-repair attempt policy so the retry continues the same
    recorded logical sample rather than silently changing its prompt.
    """

    def __init__(self) -> None:
        self._bounded_schema_repair = (
            OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier()
        )
        self._structured = StructuredGenerationRetryClassifier()

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._bounded_schema_repair.classify(error, context=context)
        if not (
            isinstance(error, StructuredStreamTimeoutError)
            and error.phase is StructuredStreamTimeoutPhase.FIRST_EVENT
        ):
            return classified
        retry = self._structured.classify(error, context=context)
        if (
            retry.disposition is RetryDisposition.RETRY
            and retry.reason is RetryReason.TIMEOUT
        ):
            return retry
        return classified


class BoundedPrestreamAndSchemaRepairRetryClassifier:
    """Bound opaque pre-stream recovery, schema repair, and first-event retry.

    This is the long-running campaign policy.  It preserves exact request
    bytes across opaque HTTP-400 and transport retries, admits bounded repair
    resampling only after typed invalid output, and retries only the
    content-blind first-event stream timeout.  Idle, absolute, and cleanup
    timeouts remain terminal.
    """

    def __init__(self) -> None:
        self._opaque_http_400 = BoundedOpaqueHTTP400RetryClassifier()
        self._structured = StructuredGenerationRetryClassifier()

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        classified = self._opaque_http_400.classify(
            error,
            context=context,
        )
        if (
            isinstance(error, StructuredGenerationError)
            and error.kind is GenerationFailureKind.OUTPUT_INVALID
            and error.retryable
        ):
            repair = self._structured.classify(
                error,
                context=context,
            )
            if (
                repair.disposition is RetryDisposition.RETRY
                and repair.reason is RetryReason.OUTPUT_INVALID
            ):
                return repair
        if not (
            isinstance(error, StructuredStreamTimeoutError)
            and error.phase is StructuredStreamTimeoutPhase.FIRST_EVENT
        ):
            return classified
        retry = self._structured.classify(error, context=context)
        if (
            retry.disposition is RetryDisposition.RETRY
            and retry.reason is RetryReason.TIMEOUT
        ):
            return retry
        return classified


class QueuedStructuredGenerationError(RuntimeError):
    """Sanitized non-success queue outcome with complete scheduling telemetry."""

    def __init__(
        self,
        outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
    ) -> None:
        if type(outcome) is not LLMTaskOutcome:
            raise TypeError("outcome must be an exact LLMTaskOutcome")
        LLMTaskOutcome.__post_init__(outcome)
        if outcome.status is TaskOutcomeStatus.SUCCEEDED:
            raise ValueError("a successful outcome is not a terminal error")
        messages = {
            TaskOutcomeStatus.TERMINAL_FAILURE: (
                "queued structured generation failed terminally"
            ),
            TaskOutcomeStatus.ATTEMPTS_EXHAUSTED: (
                "queued structured generation exhausted its attempt budget"
            ),
            TaskOutcomeStatus.CANCELLED: "queued structured generation was cancelled",
        }
        super().__init__(messages[outcome.status])
        self.outcome = outcome

    @property
    def status(self) -> TaskOutcomeStatus:
        return self.outcome.status

    @property
    def telemetry(self) -> TaskTelemetry:
        return self.outcome.telemetry

    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition:
        attempts = self.outcome.telemetry.attempts
        if not attempts:
            return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE
        classification = attempts[-1].classification
        failure = None if classification is None else classification.sanitized_failure
        if failure is not None and failure.kind in {
            "output_invalid",
            "content_rejected",
        }:
            return GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
        return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE


class OutcomePublicationError(RuntimeError):
    """Sanitized failure of a required terminal-outcome publication sink."""

    def __init__(
        self,
        outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
    ) -> None:
        if type(outcome) is not LLMTaskOutcome:
            raise TypeError("outcome must be an exact LLMTaskOutcome")
        LLMTaskOutcome.__post_init__(outcome)
        super().__init__("required queued outcome publication failed")
        self.outcome = outcome

    @property
    def status(self) -> TaskOutcomeStatus:
        return self.outcome.status

    @property
    def telemetry(self) -> TaskTelemetry:
        return self.outcome.telemetry

    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition:
        return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE


class CancelledOutcomePublicationError(asyncio.CancelledError):
    """Cancellation whose required terminal receipt could not be published.

    A submitter cancellation remains cancellation even when a required recorder
    fails: there is no provider response that can safely be released and no
    retry that can repair the recorder.  This typed, content-free cancellation
    surfaces that secondary failure without replacing caller cancellation with
    an ordinary exception.
    """

    def __init__(
        self,
        outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
    ) -> None:
        if type(outcome) is not LLMTaskOutcome:
            raise TypeError("outcome must be an exact LLMTaskOutcome")
        LLMTaskOutcome.__post_init__(outcome)
        super().__init__("required cancelled-outcome publication failed")
        self.outcome = outcome

    @property
    def status(self) -> TaskOutcomeStatus:
        return self.outcome.status

    @property
    def telemetry(self) -> TaskTelemetry:
        return self.outcome.telemetry


class StructuredEvidencePublicationError(RuntimeError):
    """Sanitized failure of a required request/output evidence sink."""

    def __init__(
        self,
        *,
        stage: StructuredEvidencePublicationStage,
        call_id: str,
        outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]] | None = None,
    ) -> None:
        if type(stage) is not StructuredEvidencePublicationStage:
            raise TypeError("stage must be a StructuredEvidencePublicationStage")
        if type(call_id) is not str or not call_id:
            raise ValueError("call_id must be a non-empty exact string")
        if outcome is not None:
            if type(outcome) is not LLMTaskOutcome:
                raise TypeError("outcome must be an exact LLMTaskOutcome or None")
            LLMTaskOutcome.__post_init__(outcome)
            if outcome.telemetry.task_id != call_id:
                raise ValueError("evidence failure outcome has a foreign call ID")
        super().__init__(
            f"required structured {stage.value} evidence publication failed"
        )
        self.stage = stage
        self.call_id = call_id
        self.outcome = outcome

    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition:
        return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE


class QueuedStructuredGenerationRunner:
    """Callable multi-attempt runner consumed by ``PydanticAIAgenticGenerator``."""

    def __init__(
        self,
        *,
        queue: AsyncLLMTaskQueue[
            StructuredGenerationRequest[Any],
            StructuredGenerationResponse[Any],
        ],
        max_attempts: int,
        retry_budget: PartitionedRetryBudget | None = None,
        owned_generator: PydanticAIStructuredGenerator | None = None,
        outcome_sink: OutcomeSink | None = None,
        outcome_publication_policy: OutcomePublicationPolicy = (
            OutcomePublicationPolicy.BEST_EFFORT
        ),
        request_evidence_sink: StructuredRequestEvidenceSink | None = None,
        output_evidence_sink: StructuredOutputEvidenceSink | None = None,
        evidence_publication_policy: StructuredEvidencePublicationPolicy = (
            StructuredEvidencePublicationPolicy.BEST_EFFORT
        ),
    ) -> None:
        if type(queue) is not AsyncLLMTaskQueue:
            raise TypeError("queue must be an exact AsyncLLMTaskQueue")
        if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ATTEMPTS:
            raise ValueError(f"max_attempts must lie in [1, {MAX_ATTEMPTS}]")
        if retry_budget is not None and type(retry_budget) is not PartitionedRetryBudget:
            raise TypeError(
                "retry_budget must be a PartitionedRetryBudget or None"
            )
        if retry_budget is not None:
            PartitionedRetryBudget.__post_init__(retry_budget)
        if owned_generator is not None and not isinstance(
            owned_generator, PydanticAIStructuredGenerator
        ):
            raise TypeError(
                "owned_generator must be a PydanticAIStructuredGenerator or None"
            )
        if outcome_sink is not None and not callable(outcome_sink):
            raise TypeError("outcome_sink must be callable or None")
        if type(outcome_publication_policy) is not OutcomePublicationPolicy:
            raise TypeError(
                "outcome_publication_policy must be an OutcomePublicationPolicy"
            )
        if (
            outcome_publication_policy is OutcomePublicationPolicy.REQUIRED
            and outcome_sink is None
        ):
            raise ValueError("required outcome publication needs an outcome_sink")
        for name, sink in (
            ("request_evidence_sink", request_evidence_sink),
            ("output_evidence_sink", output_evidence_sink),
        ):
            if sink is not None and not callable(sink):
                raise TypeError(f"{name} must be callable or None")
        if type(evidence_publication_policy) is not StructuredEvidencePublicationPolicy:
            raise TypeError(
                "evidence_publication_policy must be a "
                "StructuredEvidencePublicationPolicy"
            )
        if evidence_publication_policy is StructuredEvidencePublicationPolicy.REQUIRED:
            if request_evidence_sink is None or output_evidence_sink is None:
                raise ValueError(
                    "required structured evidence publication needs both sinks"
                )
        self._queue = queue
        self.max_attempts = max_attempts
        self.retry_budget = retry_budget
        self._owned_generator = owned_generator
        self._outcome_sink = outcome_sink
        self.outcome_publication_policy = outcome_publication_policy
        self._request_evidence_sink = request_evidence_sink
        self._output_evidence_sink = output_evidence_sink
        self.evidence_publication_policy = evidence_publication_policy
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def __call__(
        self,
        request: StructuredGenerationRequest[OutputT],
    ) -> AttemptedStructuredGenerationResponse[OutputT]:
        return await self.generate(request)

    async def generate(
        self,
        request: StructuredGenerationRequest[OutputT],
    ) -> AttemptedStructuredGenerationResponse[OutputT]:
        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(request)
        request_evidence = self._publish_request_evidence(request)
        outcome = await self._queue.submit(
            LLMTask(
                task_id=request.call_id.value,
                request=request,
                max_attempts=self.max_attempts,
                retry_budget=self.retry_budget,
            ),
            cancellation_outcome_sink=self._observe_cancelled_outcome,
        )
        if type(outcome) is not LLMTaskOutcome:
            raise TypeError("queue returned a non-outcome value")
        LLMTaskOutcome.__post_init__(outcome)
        self._observe_outcome(outcome)

        if outcome.status is not TaskOutcomeStatus.SUCCEEDED:
            raise QueuedStructuredGenerationError(outcome) from None
        response = outcome.response
        if type(response) is not StructuredGenerationResponse:
            raise TypeError("successful queue outcome has no structured response")
        StructuredGenerationResponse.__post_init__(response)
        self._publish_output_evidence(
            request,
            outcome,
            request_evidence=request_evidence,
        )
        return AttemptedStructuredGenerationResponse(
            response=response,
            attempt_count=len(outcome.telemetry.attempts),
        )

    def _observe_outcome(
        self,
        outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
    ) -> None:
        if self._outcome_sink is None:
            return
        try:
            self._outcome_sink(outcome)
        except Exception:
            if self.outcome_publication_policy is OutcomePublicationPolicy.REQUIRED:
                # The logical provider outcome is already terminal.  Fail
                # closed without retrying or exposing the response downstream.
                raise OutcomePublicationError(outcome) from None
            # Best-effort publication preserves the historical behavior: a
            # recorder failure does not change an already-terminal outcome.

    def _observe_cancelled_outcome(
        self,
        outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
    ) -> None:
        """Publish a cancelled await's terminal receipt without changing its law."""

        try:
            self._observe_outcome(outcome)
        except OutcomePublicationError as error:
            # Required publication failure is observable, but remains a
            # cancellation so concurrent-stage cleanup cannot misclassify the
            # cancelled sibling as the primary model/provider failure.
            raise CancelledOutcomePublicationError(error.outcome) from None

    def _publish_request_evidence(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> dict[str, object] | None:
        sink = self._request_evidence_sink
        if sink is None:
            return None
        try:
            record = structured_generation_request_evidence_record(request)
            sink(record)
            return record
        except Exception:
            if (
                self.evidence_publication_policy
                is StructuredEvidencePublicationPolicy.REQUIRED
            ):
                raise StructuredEvidencePublicationError(
                    stage=StructuredEvidencePublicationStage.REQUEST,
                    call_id=request.call_id.value,
                ) from None
            return None

    def _publish_output_evidence(
        self,
        request: StructuredGenerationRequest[Any],
        outcome: LLMTaskOutcome[StructuredGenerationResponse[Any]],
        *,
        request_evidence: dict[str, object] | None,
    ) -> None:
        sink = self._output_evidence_sink
        if sink is None:
            return
        try:
            record = structured_generation_output_evidence_record(
                request,
                outcome,
                request_evidence=request_evidence,
            )
            sink(record)
        except Exception:
            if (
                self.evidence_publication_policy
                is StructuredEvidencePublicationPolicy.REQUIRED
            ):
                raise StructuredEvidencePublicationError(
                    stage=StructuredEvidencePublicationStage.OUTPUT,
                    call_id=request.call_id.value,
                    outcome=outcome,
                ) from None

    async def snapshot(self) -> QueueSnapshot:
        return await self._queue.snapshot()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            try:
                await self._queue.aclose()
            finally:
                self._closed = True
                if self._owned_generator is not None:
                    generator = self._owned_generator
                    self._owned_generator = None
                    await generator.aclose()

    async def __aenter__(self) -> "QueuedStructuredGenerationRunner":
        if self._closed:
            raise LLMTaskQueueClosedError("the queued runner is closed")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _composed_runner(
    *,
    generator: StructuredGenerator,
    max_in_flight: int,
    max_pending: int,
    max_attempts: int,
    retry_budget: PartitionedRetryBudget | None,
    attempt_timeout_ns: int | None,
    backoff_policy: BackoffPolicy,
    runtime: AsyncRuntime,
    owned_generator: PydanticAIStructuredGenerator | None,
    outcome_sink: OutcomeSink | None,
    outcome_publication_policy: OutcomePublicationPolicy,
    request_evidence_sink: StructuredRequestEvidenceSink | None,
    output_evidence_sink: StructuredOutputEvidenceSink | None,
    evidence_publication_policy: StructuredEvidencePublicationPolicy,
    attempt_request_policy: StructuredAttemptRequestPolicy | None,
    retry_classifier: RetryClassifier,
) -> QueuedStructuredGenerationRunner:
    queue = AsyncLLMTaskQueue(
        executor=StructuredGenerationExecutor(
            generator,
            attempt_request_policy=attempt_request_policy,
        ),
        retry_classifier=retry_classifier,
        backoff_policy=backoff_policy,
        clock=SystemClock(),
        max_in_flight=max_in_flight,
        max_pending=max_pending,
        attempt_timeout_ns=attempt_timeout_ns,
        runtime=runtime,
    )
    return QueuedStructuredGenerationRunner(
        queue=queue,
        max_attempts=max_attempts,
        retry_budget=retry_budget,
        owned_generator=owned_generator,
        outcome_sink=outcome_sink,
        outcome_publication_policy=outcome_publication_policy,
        request_evidence_sink=request_evidence_sink,
        output_evidence_sink=output_evidence_sink,
        evidence_publication_policy=evidence_publication_policy,
    )


def create_production_queued_runner(
    *,
    generator: PydanticAIStructuredGenerator,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    max_pending: int = DEFAULT_MAX_PENDING,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_budget: PartitionedRetryBudget | None = None,
    attempt_timeout_ns: int | None = DEFAULT_ATTEMPT_TIMEOUT_NS,
    base_backoff_ns: int = DEFAULT_BASE_BACKOFF_NS,
    max_backoff_ns: int = DEFAULT_MAX_BACKOFF_NS,
    rate_limit_backoff_floor_ns: int = 0,
    random_source: RandomRange | None = None,
    jitter_policy: JitterPolicy | None = None,
    close_generator: bool = True,
    outcome_sink: OutcomeSink | None = None,
    outcome_publication_policy: OutcomePublicationPolicy = (
        OutcomePublicationPolicy.BEST_EFFORT
    ),
    request_evidence_sink: StructuredRequestEvidenceSink | None = None,
    output_evidence_sink: StructuredOutputEvidenceSink | None = None,
    evidence_publication_policy: StructuredEvidencePublicationPolicy = (
        StructuredEvidencePublicationPolicy.BEST_EFFORT
    ),
    attempt_request_policy: StructuredAttemptRequestPolicy | None = None,
    retry_classifier: RetryClassifier | None = None,
) -> QueuedStructuredGenerationRunner:
    """Build the real queue runtime around an already configured generator.

    The existing OpenRouter generator factory fixes SDK and Pydantic-AI retries
    at zero.  By default this factory takes ownership of that generator; pass
    ``close_generator=False`` only when its lifecycle is owned elsewhere.
    Experiments that require durable pre-validation telemetry should supply an
    fsync-on-return sink and select ``OutcomePublicationPolicy.REQUIRED``.
    ``attempt_request_policy`` lets an experiment bind the exact immutable
    schema-repair policy whose manifest is frozen with its launch contract.

    ``attempt_timeout_ns`` is the queue's absolute containment boundary, not a
    provider read-idle timeout.  Set it to ``None`` when the generator owns a
    :class:`StructuredStreamLivenessPolicy`; the content-blind first-event and
    idle watchdogs then supervise normal liveness without imposing a fixed
    total cutoff on a progressing stream.  An optional absolute fail-safe
    remains available in that policy. In this mode request cancellation stays
    local to its stream; the shared owned generator is closed only after queue
    shutdown has drained all active calls.
    """

    if not isinstance(generator, PydanticAIStructuredGenerator):
        raise TypeError("generator must be a PydanticAIStructuredGenerator")
    if generator.stream_liveness_policy is not None and attempt_timeout_ns is not None:
        raise ValueError(
            "progress-aware generators require attempt_timeout_ns=None; configure "
            "an absolute fail-safe on StructuredStreamLivenessPolicy instead"
        )
    if type(close_generator) is not bool:
        raise TypeError("close_generator must be bool")
    if retry_budget is not None and type(retry_budget) is not PartitionedRetryBudget:
        raise TypeError("retry_budget must be a PartitionedRetryBudget or None")
    if retry_budget is not None:
        PartitionedRetryBudget.__post_init__(retry_budget)
    if random_source is not None and jitter_policy is not None:
        raise ValueError("random_source and jitter_policy are mutually exclusive")
    if retry_classifier is None:
        retry_classifier = StructuredGenerationRetryClassifier()
    elif not isinstance(retry_classifier, RetryClassifier):
        raise TypeError("retry_classifier must implement RetryClassifier or be None")
    if jitter_policy is None:
        if random_source is None:
            random_source = SystemRandom()
        jitter_policy = FullJitter(random_source)
    elif not isinstance(jitter_policy, JitterPolicy):
        raise TypeError("jitter_policy must implement JitterPolicy or be None")
    backoff = ExponentialBackoff(
        base_delay_ns=base_backoff_ns,
        max_delay_ns=max_backoff_ns,
        jitter=jitter_policy,
        rate_limit_floor_ns=rate_limit_backoff_floor_ns,
    )
    return _composed_runner(
        generator=generator,
        max_in_flight=max_in_flight,
        max_pending=max_pending,
        max_attempts=max_attempts,
        retry_budget=retry_budget,
        attempt_timeout_ns=attempt_timeout_ns,
        backoff_policy=backoff,
        runtime=AsyncioRuntime(
            timeout_abort=(
                generator.aclose
                if close_generator and attempt_timeout_ns is not None
                else None
            ),
        ),
        owned_generator=generator if close_generator else None,
        outcome_sink=outcome_sink,
        outcome_publication_policy=outcome_publication_policy,
        request_evidence_sink=request_evidence_sink,
        output_evidence_sink=output_evidence_sink,
        evidence_publication_policy=evidence_publication_policy,
        attempt_request_policy=attempt_request_policy,
        retry_classifier=retry_classifier,
    )


__all__ = [
    "CancelledOutcomePublicationError",
    "ExactPayloadAttemptPolicy",
    "ExactTransportSchemaRepairAttemptPolicy",
    "MAX_SCHEMA_REPAIR_REQUIRED_PATHS",
    "MAX_SCHEMA_REPAIR_SCHEMA_NODES",
    "MAX_SCHEMA_REPAIR_SUFFIX_UTF8_BYTES",
    "MAX_STRUCTURED_OUTPUT_EVIDENCE_UTF8_BYTES",
    "MAX_STRUCTURED_OUTPUT_SCHEMA_UTF8_BYTES",
    "OutcomePublicationError",
    "OutcomePublicationPolicy",
    "QueuedStructuredGenerationError",
    "QueuedStructuredGenerationRunner",
    "OutcomeSink",
    "PreparedStructuredAttemptRequest",
    "SCHEMA_REPAIR_POLICY_ID",
    "SCHEMA_REPAIR_POLICY_MANIFEST",
    "SCHEMA_REPAIR_POLICY_VERSION",
    "SchemaRepairPolicyManifest",
    "SchemaRepairAttemptPolicy",
    "StructuredAttemptRequestPolicy",
    "StructuredEvidencePublicationError",
    "StructuredEvidencePublicationPolicy",
    "StructuredEvidencePublicationStage",
    "STRUCTURED_OUTPUT_EVIDENCE_SCHEMA_VERSION",
    "STRUCTURED_REQUEST_EVIDENCE_SCHEMA_VERSION",
    "STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSION",
    "SUPPORTED_STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSIONS",
    "StructuredGenerationExecutor",
    "StructuredGenerationRetryClassifier",
    "NonRepeatingStreamTransportRetryClassifier",
    "BoundedOpaqueHTTP400RetryClassifier",
    "BoundedPrestreamAndSchemaRepairRetryClassifier",
    "OpaqueHTTP400OnceRetryClassifier",
    "OpaqueHTTP400AndSchemaRepairOnceRetryClassifier",
    "OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier",
    "FirstEventResilientBoundedSchemaRepairRetryClassifier",
    "TransportOnlyStructuredGenerationRetryClassifier",
    "create_production_queued_runner",
    "structured_generation_output_evidence_record",
    "structured_generation_outcome_record",
    "structured_generation_request_evidence_record",
    "validate_structured_generation_output_evidence_record",
    "validate_structured_generation_request_evidence_record",
    "StructuredOutputEvidenceSink",
    "StructuredRequestEvidenceSink",
]

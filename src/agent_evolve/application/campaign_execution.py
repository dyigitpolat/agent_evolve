"""Live, workload-neutral execution ledger for prepared evolution campaigns.

Preparation and execution intentionally remain separate.  This module consumes
an authenticated :class:`PreparedEvolutionCampaign`, but delegates benchmark
state, portfolio/recombination work, reflection, and lifecycle effects to
small async ports.  The application layer owns only chronology, joins, budget
accounting, archive-utility cutoffs, reflection visibility, durable events,
and cleanup/finalization.

No provider call, model setting, objective, or workload action appears here.
Existing ``PortfolioEvolution`` and ``PortfolioRecombination`` services can be
bridged by one stateful ``CampaignStageRuntimePort`` implementation.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.evolution_campaign import (
    ArchiveUtilitySnapshot,
    CampaignGenerationKind,
    CampaignGenerationStep,
    CampaignPolicies,
    CampaignPromotionBarrier,
    CampaignReflectionWave,
    CampaignReflectionSupervisionPolicy,
    PreparedEvolutionCampaign,
    ReflectionFailureMode,
    ReflectionVisibility,
    freeze_archive_utility,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_STEP_DOMAIN = b"agent-evolve:campaign-execution-step:v1\x00"
_AUDIT_DOMAIN = b"agent-evolve:campaign-selector-audit:v1\x00"
_AUDIT_SET_DOMAIN = b"agent-evolve:campaign-selector-audit-set:v1\x00"
_START_DOMAIN = b"agent-evolve:campaign-execution-start:v1\x00"
_ARCHIVE_REQUEST_DOMAIN = b"agent-evolve:campaign-archive-cutoff-request:v1\x00"
_ARCHIVE_RECEIPT_DOMAIN = b"agent-evolve:campaign-archive-cutoff-receipt:v1\x00"
_STAGE_REQUEST_DOMAIN = b"agent-evolve:campaign-stage-request:v1\x00"
_STAGE_RECEIPT_DOMAIN = b"agent-evolve:campaign-stage-receipt:v1\x00"
_REFLECTION_REQUEST_DOMAIN = b"agent-evolve:campaign-reflection-request:v1\x00"
_REFLECTION_RECEIPT_DOMAIN = b"agent-evolve:campaign-reflection-receipt:v1\x00"
_REFLECTION_CANCELLATION_RECEIPT_DOMAIN = (
    b"agent-evolve:campaign-reflection-cancellation-receipt:v1\x00"
)
_TEST_ADMISSION_REQUEST_DOMAIN = (
    b"agent-evolve:campaign-reflection-test-admission-request:v1\x00"
)
_TEST_ADMISSION_RECEIPT_DOMAIN = (
    b"agent-evolve:campaign-reflection-test-admission-receipt:v1\x00"
)
_TAIL_DRAIN_DOMAIN = b"agent-evolve:campaign-tail-drain:v1\x00"
_COUNTERS_DOMAIN = b"agent-evolve:campaign-execution-counters:v1\x00"
_FINALIZATION_REQUEST_DOMAIN = (
    b"agent-evolve:campaign-execution-finalization-request:v1\x00"
)
_FINALIZATION_RECEIPT_DOMAIN = (
    b"agent-evolve:campaign-execution-finalization-receipt:v1\x00"
)
_CLEANUP_REQUEST_DOMAIN = b"agent-evolve:campaign-runtime-cleanup-request:v1\x00"
_CLEANUP_RECEIPT_DOMAIN = b"agent-evolve:campaign-runtime-cleanup-receipt:v1\x00"
_EVENT_DOMAIN = b"agent-evolve:campaign-execution-event:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:campaign-execution-result:v1\x00"
_AUDIT_TEXT_INLINE_UTF8_BYTES = 1_000_000
_AUDIT_TEXT_BASE64_CHUNK_CHARACTERS = 500_000
_AUDIT_TEXT_MAX_DECOMPRESSED_UTF8_BYTES = 64 * 1024 * 1024


class CampaignExecutionContractError(ValueError):
    """A runtime receipt violated the prepared campaign contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, record: object) -> str:
    return hashlib.sha256(domain + _canonical_json(record)).hexdigest()


def encode_selector_audit_text(name: str, value: str) -> dict[str, object]:
    """Represent exact audit text without violating typed-JSON scalar limits.

    Ordinary traces retain the original single-string representation byte for
    byte. Large structured decisions are losslessly compressed and split into
    bounded ASCII chunks; the receipt authenticates both compressed and
    decompressed bytes. This is representation-only and has no optimizer or
    model authority.
    """

    if name not in {"request_text", "response_text"}:
        raise ValueError("selector audit text name is unsupported")
    if type(value) is not str or not value:
        raise ValueError("selector audit text must be a non-empty exact string")
    raw = value.encode("utf-8", errors="strict")
    if len(raw) <= _AUDIT_TEXT_INLINE_UTF8_BYTES:
        return {name: value}
    if len(raw) > _AUDIT_TEXT_MAX_DECOMPRESSED_UTF8_BYTES:
        raise ValueError("selector audit text exceeds the lossless audit ceiling")
    compressed = zlib.compress(raw, level=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    chunks = [
        encoded[offset : offset + _AUDIT_TEXT_BASE64_CHUNK_CHARACTERS]
        for offset in range(0, len(encoded), _AUDIT_TEXT_BASE64_CHUNK_CHARACTERS)
    ]
    return {
        f"{name}_encoding": "zlib_base64_chunks_v1",
        f"{name}_base64_chunks": chunks,
        f"{name}_utf8_bytes": len(raw),
        f"{name}_sha256": hashlib.sha256(raw).hexdigest(),
        f"{name}_compressed_bytes": len(compressed),
        f"{name}_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
    }


def decode_selector_audit_text(audit: dict[str, object], name: str) -> str:
    """Validate and recover one inline or losslessly chunked audit field."""

    if type(audit) is not dict:
        raise TypeError("selector audit must be an exact object")
    if name not in {"request_text", "response_text"}:
        raise ValueError("selector audit text name is unsupported")
    inline = audit.get(name)
    if inline is not None:
        if type(inline) is not str or not inline:
            raise ValueError(f"plaintext audit requires non-empty {name}")
        encoded_keys = {
            f"{name}_encoding",
            f"{name}_base64_chunks",
            f"{name}_utf8_bytes",
            f"{name}_sha256",
            f"{name}_compressed_bytes",
            f"{name}_compressed_sha256",
        }
        if any(key in audit for key in encoded_keys):
            raise ValueError("inline selector audit text cannot also be encoded")
        return inline
    if audit.get(f"{name}_encoding") != "zlib_base64_chunks_v1":
        raise ValueError(f"plaintext audit requires non-empty {name}")
    chunks = audit.get(f"{name}_base64_chunks")
    if (
        type(chunks) is not list
        or not chunks
        or any(type(value) is not str or not value for value in chunks)
    ):
        raise ValueError("selector audit chunks must be non-empty exact strings")
    expected_utf8_bytes = audit.get(f"{name}_utf8_bytes")
    expected_compressed_bytes = audit.get(f"{name}_compressed_bytes")
    if (
        type(expected_utf8_bytes) is not int
        or not _AUDIT_TEXT_INLINE_UTF8_BYTES
        < expected_utf8_bytes
        <= _AUDIT_TEXT_MAX_DECOMPRESSED_UTF8_BYTES
        or type(expected_compressed_bytes) is not int
        or expected_compressed_bytes <= 0
    ):
        raise ValueError("selector audit encoded byte counts are invalid")
    try:
        compressed = base64.b64decode("".join(chunks), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise ValueError("selector audit chunks are not canonical base64") from error
    if len(compressed) != expected_compressed_bytes:
        raise ValueError("selector audit compressed byte count disagrees")
    if hashlib.sha256(compressed).hexdigest() != audit.get(
        f"{name}_compressed_sha256"
    ):
        raise ValueError("selector audit compressed digest disagrees")
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(compressed, expected_utf8_bytes + 1)
        raw += decompressor.flush()
    except zlib.error as error:
        raise ValueError("selector audit compressed text is invalid") from error
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or len(raw) != expected_utf8_bytes
    ):
        raise ValueError("selector audit decompressed byte count disagrees")
    if hashlib.sha256(raw).hexdigest() != audit.get(f"{name}_sha256"):
        raise ValueError("selector audit decompressed digest disagrees")
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("selector audit text is not valid UTF-8") from error
    if not value:
        raise ValueError(f"plaintext audit requires non-empty {name}")
    return value


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed execution-token grammar")


def _frozen(value: object, *, name: str) -> FrozenJsonObject:
    if type(value) is not FrozenJsonObject:
        raise TypeError(f"{name} must be an exact FrozenJsonObject")
    if freeze_json(value) is not value:
        raise TypeError(f"{name} must already be frozen typed JSON")
    return value


def _freeze_record(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed input.
        raise AssertionError("record did not freeze to an object")
    return frozen


def campaign_step_sha256(step: CampaignGenerationStep) -> str:
    if type(step) is not CampaignGenerationStep:
        raise TypeError("step must be an exact CampaignGenerationStep")
    CampaignGenerationStep.__post_init__(step)
    return _hash(_STEP_DOMAIN, step.to_record())


class SelectorAuditExecutionMode(str, Enum):
    FRESH = "fresh"


@dataclass(frozen=True, slots=True)
class CampaignSelectorAuditReceipt:
    """Fresh selector-call trace required for each portfolio parent slot."""

    generation: int
    parent_slot: int
    selector_call_id: str
    request_sha256: str
    decision_sha256: str
    trace_receipt_sha256: str
    plaintext_audit: FrozenJsonObject
    prior_audit_set_sha256: str
    execution_mode: SelectorAuditExecutionMode

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("selector audit generation must be positive")
        if type(self.parent_slot) is not int or self.parent_slot < 0:
            raise ValueError("selector audit parent_slot must be non-negative")
        _token(self.selector_call_id, name="selector_call_id")
        for name in (
            "request_sha256",
            "decision_sha256",
            "trace_receipt_sha256",
            "prior_audit_set_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _frozen(self.plaintext_audit, name="plaintext_audit")
        if typed_json_sha256(self.plaintext_audit) != self.trace_receipt_sha256:
            raise ValueError(
                "trace_receipt_sha256 does not authenticate plaintext audit"
            )
        audit = thaw_json(self.plaintext_audit)
        if type(audit) is not dict:  # pragma: no cover - exact object above.
            raise AssertionError("plaintext audit thawed to a non-object")
        required = {
            "selector_call_id": self.selector_call_id,
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
        }
        if any(audit.get(name) != value for name, value in required.items()):
            raise ValueError(
                "plaintext audit request/decision/call join is inconsistent"
            )
        for name in ("request_text", "response_text"):
            decode_selector_audit_text(audit, name)
        if self.execution_mode is not SelectorAuditExecutionMode.FRESH:
            raise ValueError("portfolio selector audit must record fresh execution")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "generation": self.generation,
            "parent_slot": self.parent_slot,
            "selector_call_id": self.selector_call_id,
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
            "trace_receipt_sha256": self.trace_receipt_sha256,
            "plaintext_audit": thaw_json(self.plaintext_audit),
            "prior_audit_set_sha256": self.prior_audit_set_sha256,
            "execution_mode": self.execution_mode.value,
        }

    @property
    def audit_sha256(self) -> str:
        return _hash(_AUDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "audit_sha256": self.audit_sha256}


def selector_audit_set_sha256(
    audits: tuple[CampaignSelectorAuditReceipt, ...],
) -> str:
    if type(audits) is not tuple or any(
        type(value) is not CampaignSelectorAuditReceipt for value in audits
    ):
        raise TypeError("audits must contain exact selector audit receipts")
    for value in audits:
        CampaignSelectorAuditReceipt.__post_init__(value)
    return _hash(
        _AUDIT_SET_DOMAIN,
        [value.audit_sha256 for value in audits],
    )


@dataclass(frozen=True, slots=True)
class CampaignSeedExecutionReceipt:
    """Exact seed admission/evaluation accounting produced during start."""

    seed_id: str
    configuration_sha256: str
    evaluated: bool
    unique_evaluation: bool
    valid: bool
    failure_type: str | None
    evidence: FrozenJsonObject

    def __post_init__(self) -> None:
        _token(self.seed_id, name="seed_id")
        require_sha256(self.configuration_sha256, "configuration_sha256")
        for name in ("evaluated", "unique_evaluation", "valid"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if not self.evaluated and self.unique_evaluation:
            raise ValueError("an unevaluated seed cannot be a unique evaluation")
        if not self.evaluated and self.valid:
            raise ValueError("an unevaluated seed cannot be valid")
        if self.valid:
            if not self.evaluated or self.failure_type is not None:
                raise ValueError("valid seed accounting cannot carry a failure")
        elif type(self.failure_type) is not str or not self.failure_type:
            raise ValueError("failed or invalid seed accounting requires failure_type")
        _frozen(self.evidence, name="evidence")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "seed_id": self.seed_id,
            "configuration_sha256": self.configuration_sha256,
            "evaluated": self.evaluated,
            "unique_evaluation": self.unique_evaluation,
            "valid": self.valid,
            "failure_type": self.failure_type,
            "evidence": thaw_json(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class CampaignExecutionStartReceipt:
    preparation_sha256: str
    runtime_preflight_receipt_sha256: str
    runtime_session_id: str
    seed_batch_sha256: str
    seed_receipts: tuple[CampaignSeedExecutionReceipt, ...]
    evidence: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_preflight_receipt_sha256,
            "runtime_preflight_receipt_sha256",
        )
        _token(self.runtime_session_id, name="runtime_session_id")
        require_sha256(self.seed_batch_sha256, "seed_batch_sha256")
        if type(self.seed_receipts) is not tuple or not self.seed_receipts:
            raise ValueError("runtime start requires exact seed receipts")
        if any(
            type(value) is not CampaignSeedExecutionReceipt
            for value in self.seed_receipts
        ):
            raise TypeError("seed_receipts must contain exact receipts")
        for value in self.seed_receipts:
            CampaignSeedExecutionReceipt.__post_init__(value)
        if len({value.seed_id for value in self.seed_receipts}) != len(
            self.seed_receipts
        ):
            raise ValueError("runtime start seed IDs must be unique")
        _frozen(self.evidence, name="evidence")

    @property
    def seed_occurrence_count(self) -> int:
        return len(self.seed_receipts)

    @property
    def seed_unique_evaluation_count(self) -> int:
        return sum(value.unique_evaluation for value in self.seed_receipts)

    @property
    def seed_failure_count(self) -> int:
        return sum(not value.valid for value in self.seed_receipts)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "runtime_preflight_receipt_sha256": (self.runtime_preflight_receipt_sha256),
            "runtime_session_id": self.runtime_session_id,
            "seed_batch_sha256": self.seed_batch_sha256,
            "seed_receipts": [value.to_record() for value in self.seed_receipts],
            "seed_accounting": {
                "occurrences": self.seed_occurrence_count,
                "unique_evaluations": self.seed_unique_evaluation_count,
                "failures": self.seed_failure_count,
            },
            "evidence": thaw_json(self.evidence),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_START_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignArchiveCutoffRequest:
    preparation_sha256: str
    runtime_start_receipt_sha256: str
    generation: int
    kind: CampaignGenerationKind
    step_sha256: str
    prior_stage_receipt_sha256: str | None

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_start_receipt_sha256,
            "runtime_start_receipt_sha256",
        )
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("archive cutoff generation must be positive")
        if type(self.kind) is not CampaignGenerationKind:
            raise TypeError("archive cutoff kind must be CampaignGenerationKind")
        require_sha256(self.step_sha256, "step_sha256")
        if self.prior_stage_receipt_sha256 is not None:
            require_sha256(
                self.prior_stage_receipt_sha256,
                "prior_stage_receipt_sha256",
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "generation": self.generation,
            "kind": self.kind.value,
            "step_sha256": self.step_sha256,
            "prior_stage_receipt_sha256": self.prior_stage_receipt_sha256,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_ARCHIVE_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignArchiveCutoffReceipt:
    request_sha256: str
    preparation_sha256: str
    generation: int
    archive: FrozenJsonObject
    evidence: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("archive cutoff generation must be positive")
        _frozen(self.archive, name="archive")
        _frozen(self.evidence, name="evidence")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "generation": self.generation,
            "archive": thaw_json(self.archive),
            "archive_sha256": typed_json_sha256(self.archive),
            "evidence": thaw_json(self.evidence),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_ARCHIVE_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignStageReceipt:
    """Opaque stage result plus exact accounting and selector trace evidence."""

    request_sha256: str
    preparation_sha256: str
    generation: int
    kind: CampaignGenerationKind
    candidate_occurrence_count: int
    unique_evaluation_count: int
    selector_audits: tuple[CampaignSelectorAuditReceipt, ...]
    result: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("stage receipt generation must be positive")
        if type(self.kind) is not CampaignGenerationKind:
            raise TypeError("stage receipt kind must be CampaignGenerationKind")
        if (
            type(self.candidate_occurrence_count) is not int
            or self.candidate_occurrence_count < 0
        ):
            raise ValueError("candidate_occurrence_count must be non-negative")
        if (
            type(self.unique_evaluation_count) is not int
            or not 0 <= self.unique_evaluation_count <= self.candidate_occurrence_count
        ):
            raise ValueError(
                "unique_evaluation_count must lie within candidate occurrences"
            )
        if type(self.selector_audits) is not tuple or any(
            type(value) is not CampaignSelectorAuditReceipt
            for value in self.selector_audits
        ):
            raise TypeError("selector_audits must contain exact audit receipts")
        for value in self.selector_audits:
            CampaignSelectorAuditReceipt.__post_init__(value)
        _frozen(self.result, name="result")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "generation": self.generation,
            "kind": self.kind.value,
            "candidate_occurrence_count": self.candidate_occurrence_count,
            "unique_evaluation_count": self.unique_evaluation_count,
            "selector_audits": [value.to_record() for value in self.selector_audits],
            "result": thaw_json(self.result),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_STAGE_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignStageRequest:
    preparation_sha256: str
    runtime_start_receipt_sha256: str
    step: CampaignGenerationStep
    archive_cutoff: CampaignArchiveCutoffReceipt
    archive_utility: ArchiveUtilitySnapshot
    source_portfolio: CampaignStageReceipt | None
    test_eligible_reflection_receipt_sha256s: tuple[str, ...]
    prior_selector_audit_set_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_start_receipt_sha256,
            "runtime_start_receipt_sha256",
        )
        if type(self.step) is not CampaignGenerationStep:
            raise TypeError("step must be an exact CampaignGenerationStep")
        CampaignGenerationStep.__post_init__(self.step)
        if type(self.archive_cutoff) is not CampaignArchiveCutoffReceipt:
            raise TypeError("archive_cutoff must be exact")
        CampaignArchiveCutoffReceipt.__post_init__(self.archive_cutoff)
        if type(self.archive_utility) is not ArchiveUtilitySnapshot:
            raise TypeError("archive_utility must be exact")
        ArchiveUtilitySnapshot.__post_init__(self.archive_utility)
        if self.archive_cutoff.generation != self.step.generation:
            raise ValueError("archive cutoff generation differs from stage")
        if (
            self.archive_utility.generation != self.step.generation
            or self.archive_utility.archive_sha256
            != typed_json_sha256(self.archive_cutoff.archive)
        ):
            raise ValueError("archive utility differs from the pre-stage cutoff")
        if self.step.kind is CampaignGenerationKind.PORTFOLIO:
            if self.source_portfolio is not None:
                raise ValueError("portfolio stage cannot carry a source portfolio")
        else:
            if type(self.source_portfolio) is not CampaignStageReceipt:
                raise ValueError("recombination requires its portfolio receipt")
            CampaignStageReceipt.__post_init__(self.source_portfolio)
            if (
                self.source_portfolio.kind is not CampaignGenerationKind.PORTFOLIO
                or self.source_portfolio.generation
                != self.step.source_portfolio_generation
                or self.source_portfolio.preparation_sha256 != self.preparation_sha256
            ):
                raise ValueError("recombination source portfolio is foreign")
        if type(self.test_eligible_reflection_receipt_sha256s) is not tuple or any(
            type(value) is not str
            for value in self.test_eligible_reflection_receipt_sha256s
        ):
            raise TypeError(
                "test-eligible reflection hashes must be an exact string tuple"
            )
        for value in self.test_eligible_reflection_receipt_sha256s:
            require_sha256(value, "test_eligible_reflection_receipt_sha256")
        if self.test_eligible_reflection_receipt_sha256s != tuple(
            sorted(set(self.test_eligible_reflection_receipt_sha256s))
        ):
            raise ValueError(
                "test-eligible reflection hashes must be unique and canonical"
            )
        require_sha256(
            self.prior_selector_audit_set_sha256,
            "prior_selector_audit_set_sha256",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "step": self.step.to_record(),
            "step_sha256": campaign_step_sha256(self.step),
            "archive_cutoff_receipt_sha256": self.archive_cutoff.receipt_sha256,
            "archive_utility_snapshot_sha256": self.archive_utility.snapshot_sha256,
            "source_portfolio_receipt_sha256": (
                None
                if self.source_portfolio is None
                else self.source_portfolio.receipt_sha256
            ),
            "test_eligible_reflection_receipt_sha256s": list(
                self.test_eligible_reflection_receipt_sha256s
            ),
            "prior_selector_audit_set_sha256": (self.prior_selector_audit_set_sha256),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_STAGE_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignReflectionRequest:
    """Reflection over one exact, already-sealed recombination cutoff.

    ``source_stage`` is the complete evidence boundary exposed by this
    application port.  A delayed admission barrier may occur later, but no
    later stage receipt is present in this request or passed to ``reflect``.
    """

    preparation_sha256: str
    runtime_start_receipt_sha256: str
    wave: CampaignReflectionWave
    source_stage: CampaignStageReceipt

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_start_receipt_sha256,
            "runtime_start_receipt_sha256",
        )
        if type(self.wave) is not CampaignReflectionWave:
            raise TypeError("wave must be an exact CampaignReflectionWave")
        CampaignReflectionWave.__post_init__(self.wave)
        if type(self.source_stage) is not CampaignStageReceipt:
            raise TypeError("source_stage must be exact")
        CampaignStageReceipt.__post_init__(self.source_stage)
        if (
            self.source_stage.preparation_sha256 != self.preparation_sha256
            or self.source_stage.generation != self.wave.source_generation
            or self.source_stage.kind is not CampaignGenerationKind.RECOMBINATION
        ):
            raise ValueError("reflection source stage is foreign")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "wave": self.wave.to_record(),
            "source_stage_receipt_sha256": self.source_stage.receipt_sha256,
        }
        barrier = self.wave.promotion_barrier_generation
        if barrier is not None and barrier > self.wave.source_generation:
            record["sealed_evidence_cutoff"] = {
                "generation": self.source_stage.generation,
                "stage_receipt_sha256": self.source_stage.receipt_sha256,
            }
            record["future_stage_evidence_permitted"] = False
        return record

    @property
    def request_sha256(self) -> str:
        return _hash(_REFLECTION_REQUEST_DOMAIN, self.to_record())


class CampaignReflectionStatus(str, Enum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CampaignReflectionReceipt:
    request_sha256: str
    preparation_sha256: str
    source_generation: int
    source_stage_receipt_sha256: str
    logical_agent_calls: int
    visibility: ReflectionVisibility
    status: CampaignReflectionStatus
    failure_type: str | None
    quarantined_result: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.source_generation) is not int or self.source_generation <= 0:
            raise ValueError("reflection source_generation must be positive")
        require_sha256(
            self.source_stage_receipt_sha256,
            "source_stage_receipt_sha256",
        )
        if type(self.logical_agent_calls) is not int or self.logical_agent_calls <= 0:
            raise ValueError("logical_agent_calls must be positive")
        if self.visibility is not ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE:
            raise ValueError("reflection result must remain quarantined")
        if type(self.status) is not CampaignReflectionStatus:
            raise TypeError("status must be CampaignReflectionStatus")
        if self.status is CampaignReflectionStatus.COMPLETED:
            if self.failure_type is not None:
                raise ValueError("completed reflection cannot carry failure_type")
        elif self.status is CampaignReflectionStatus.ABSTAINED:
            if self.failure_type is not None:
                raise ValueError("abstained reflection cannot carry failure_type")
            abstention = thaw_json(self.quarantined_result)
            if (
                abstention.get("status")
                != "abstained_no_identifiable_mutation_evidence"
                or abstention.get("evidence_tier") != "e0"
                or abstention.get("provider_calls") != 0
                or abstention.get("publishable_reflection_content") is not False
            ):
                raise ValueError("abstained reflection requires typed E0 evidence")
        elif type(self.failure_type) is not str or not self.failure_type:
            raise ValueError("failed reflection requires failure_type")
        _frozen(self.quarantined_result, name="quarantined_result")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "source_generation": self.source_generation,
            "source_stage_receipt_sha256": self.source_stage_receipt_sha256,
            "logical_agent_calls": self.logical_agent_calls,
            "visibility": self.visibility.value,
            "status": self.status.value,
            "failure_type": self.failure_type,
            "quarantined_result": thaw_json(self.quarantined_result),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_REFLECTION_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignReflectionCancellationReceipt:
    """Durable accounting for one scheduled reflection canceled on abort."""

    request_sha256: str
    preparation_sha256: str
    source_generation: int
    source_stage_receipt_sha256: str
    logical_agent_calls_reserved: int
    dispatched_to_runtime: bool
    cancellation_reason: str

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.source_generation) is not int or self.source_generation <= 0:
            raise ValueError("source_generation must be a positive exact integer")
        require_sha256(
            self.source_stage_receipt_sha256,
            "source_stage_receipt_sha256",
        )
        if (
            type(self.logical_agent_calls_reserved) is not int
            or self.logical_agent_calls_reserved <= 0
        ):
            raise ValueError("logical_agent_calls_reserved must be positive")
        if type(self.dispatched_to_runtime) is not bool:
            raise TypeError("dispatched_to_runtime must be an exact bool")
        if (
            type(self.cancellation_reason) is not str
            or _TOKEN.fullmatch(self.cancellation_reason) is None
        ):
            raise ValueError("cancellation_reason must use the campaign token grammar")

    @property
    def logical_agent_calls_dispatched_to_runtime(self) -> int:
        self.__post_init__()
        return self.logical_agent_calls_reserved if self.dispatched_to_runtime else 0

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "source_generation": self.source_generation,
            "source_stage_receipt_sha256": self.source_stage_receipt_sha256,
            "logical_agent_calls_reserved": self.logical_agent_calls_reserved,
            "logical_agent_calls_dispatched_to_runtime": (
                self.logical_agent_calls_dispatched_to_runtime
            ),
            "dispatched_to_runtime": self.dispatched_to_runtime,
            "cancellation_reason": self.cancellation_reason,
            "publishable_reflection_content": False,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(
            _REFLECTION_CANCELLATION_RECEIPT_DOMAIN,
            self._unsigned_record(),
        )

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignReflectionTestAdmissionRequest:
    """Admit quarantined evidence only to later controlled testing.

    For a delayed barrier, ``admission_cutoff_stage`` authenticates the later
    stage seal available to an admission-time shadow screen.  It is deliberately
    absent from :class:`CampaignReflectionRequest`, so it cannot alter or leak
    into the already-running reflection.
    """

    preparation_sha256: str
    runtime_start_receipt_sha256: str
    barrier: CampaignPromotionBarrier
    reflections: tuple[CampaignReflectionReceipt, ...]
    previously_test_eligible_reflection_receipt_sha256s: tuple[str, ...]
    admission_cutoff_stage: CampaignStageReceipt | None = None

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_start_receipt_sha256,
            "runtime_start_receipt_sha256",
        )
        if type(self.barrier) is not CampaignPromotionBarrier:
            raise TypeError("barrier must be an exact CampaignPromotionBarrier")
        CampaignPromotionBarrier.__post_init__(self.barrier)
        if type(self.reflections) is not tuple or any(
            type(value) is not CampaignReflectionReceipt for value in self.reflections
        ):
            raise TypeError("reflections must contain exact receipts")
        for value in self.reflections:
            CampaignReflectionReceipt.__post_init__(value)
        if tuple(value.source_generation for value in self.reflections) != (
            self.barrier.reflection_source_generations
        ):
            raise ValueError("test-admission reflections differ from barrier sources")
        if any(
            value.preparation_sha256 != self.preparation_sha256
            or value.status is not CampaignReflectionStatus.COMPLETED
            for value in self.reflections
        ):
            raise ValueError("test admission requires completed reflections")
        values = self.previously_test_eligible_reflection_receipt_sha256s
        if type(values) is not tuple or any(type(value) is not str for value in values):
            raise TypeError(
                "previously test-eligible hashes must be an exact string tuple"
            )
        for value in values:
            require_sha256(value, "previous_test_eligible_reflection_sha256")
        if values != tuple(sorted(set(values))):
            raise ValueError(
                "previously test-eligible hashes must be unique and canonical"
            )
        delayed = (
            self.barrier.generation
            > self.barrier.reflection_source_generations[-1]
        )
        if delayed:
            if type(self.admission_cutoff_stage) is not CampaignStageReceipt:
                raise ValueError(
                    "delayed admission requires its exact barrier-stage cutoff"
                )
            CampaignStageReceipt.__post_init__(self.admission_cutoff_stage)
            if (
                self.admission_cutoff_stage.preparation_sha256
                != self.preparation_sha256
                or self.admission_cutoff_stage.generation
                != self.barrier.generation
                or self.admission_cutoff_stage.kind
                is not CampaignGenerationKind.RECOMBINATION
            ):
                raise ValueError("delayed admission cutoff stage is foreign")
        elif self.admission_cutoff_stage is not None:
            raise ValueError(
                "source-closing admission cannot carry a later-stage cutoff"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "barrier": self.barrier.to_record(),
            "reflection_receipt_sha256s": [
                value.receipt_sha256 for value in self.reflections
            ],
            "previously_test_eligible_reflection_receipt_sha256s": list(
                self.previously_test_eligible_reflection_receipt_sha256s
            ),
            "admission_scope": "controlled_future_testing_only",
            "lifecycle_promotion": False,
        }
        if self.admission_cutoff_stage is not None:
            record["admission_cutoff_stage_receipt_sha256"] = (
                self.admission_cutoff_stage.receipt_sha256
            )
            record["reflection_cutoff_precedes_admission_cutoff"] = True
            record["future_evidence_visible_to_reflection"] = False
        return record

    @property
    def request_sha256(self) -> str:
        return _hash(_TEST_ADMISSION_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignReflectionTestAdmissionReceipt:
    request_sha256: str
    preparation_sha256: str
    barrier_generation: int
    admitted_reflection_receipt_sha256s: tuple[str, ...]
    test_eligible_reflection_receipt_sha256s: tuple[str, ...]
    lifecycle_promoted: bool
    evidence: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.barrier_generation) is not int or self.barrier_generation <= 0:
            raise ValueError("barrier_generation must be positive")
        for name in (
            "admitted_reflection_receipt_sha256s",
            "test_eligible_reflection_receipt_sha256s",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str for value in values
            ):
                raise TypeError(f"{name} must be an exact string tuple")
            for value in values:
                require_sha256(value, name[:-1])
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if self.lifecycle_promoted is not False:
            raise ValueError(
                "reflection barrier cannot lifecycle-promote untested insights"
            )
        _frozen(self.evidence, name="evidence")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "barrier_generation": self.barrier_generation,
            "admitted_reflection_receipt_sha256s": list(
                self.admitted_reflection_receipt_sha256s
            ),
            "test_eligible_reflection_receipt_sha256s": list(
                self.test_eligible_reflection_receipt_sha256s
            ),
            "lifecycle_promoted": False,
            "evidence": thaw_json(self.evidence),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_TEST_ADMISSION_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignTailDrainReceipt:
    preparation_sha256: str
    drained_reflections: tuple[tuple[int, str], ...]
    admitted_for_testing: bool = False
    lifecycle_promoted: bool = False

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.drained_reflections) is not tuple or not self.drained_reflections:
            raise ValueError("tail drain requires reflection receipts")
        generations: list[int] = []
        for item in self.drained_reflections:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or item[0] <= 0
                or type(item[1]) is not str
            ):
                raise TypeError(
                    "drained_reflections must contain generation/hash pairs"
                )
            require_sha256(item[1], "reflection_receipt_sha256")
            generations.append(item[0])
        if generations != sorted(set(generations)):
            raise ValueError("tail drain generations must be unique and canonical")
        if (
            self.admitted_for_testing is not False
            or self.lifecycle_promoted is not False
        ):
            raise ValueError(
                "incomplete-tail reflections remain quarantined and unadmitted"
            )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "drained_reflections": [
                {"source_generation": generation, "receipt_sha256": receipt}
                for generation, receipt in self.drained_reflections
            ],
            "admitted_for_testing": False,
            "lifecycle_promoted": False,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_TAIL_DRAIN_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignExecutionCounters:
    generations_completed: int
    candidate_occurrences: int
    unique_evaluations: int
    logical_agent_calls: int
    logical_agent_calls_dispatched_to_runtime: int = 0
    logical_agent_calls_succeeded: int = 0
    logical_agent_calls_abstained: int = 0
    logical_agent_calls_failed: int = 0
    logical_agent_calls_cancelled_before_dispatch: int = 0
    logical_agent_calls_cancelled_after_dispatch: int = 0

    def __post_init__(self) -> None:
        for name in (
            "generations_completed",
            "candidate_occurrences",
            "unique_evaluations",
            "logical_agent_calls",
            "logical_agent_calls_dispatched_to_runtime",
            "logical_agent_calls_succeeded",
            "logical_agent_calls_abstained",
            "logical_agent_calls_failed",
            "logical_agent_calls_cancelled_before_dispatch",
            "logical_agent_calls_cancelled_after_dispatch",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.logical_agent_calls_dispatched_to_runtime > self.logical_agent_calls:
            raise ValueError("dispatched logical calls exceed reserved calls")
        if (
            self.logical_agent_calls_dispatched_to_runtime
            + self.logical_agent_calls_cancelled_before_dispatch
            > self.logical_agent_calls
        ):
            raise ValueError(
                "dispatched and pre-dispatch-cancelled calls exceed reserve"
            )
        if (
            self.logical_agent_calls_succeeded
            + self.logical_agent_calls_abstained
            + self.logical_agent_calls_failed
            + self.logical_agent_calls_cancelled_after_dispatch
            > self.logical_agent_calls_dispatched_to_runtime
        ):
            raise ValueError("settled logical calls exceed dispatched calls")

    def to_record(self) -> dict[str, int]:
        self.__post_init__()
        return {
            "generations_completed": self.generations_completed,
            "candidate_occurrences": self.candidate_occurrences,
            "unique_evaluations": self.unique_evaluations,
            "logical_agent_calls": self.logical_agent_calls,
            "logical_agent_calls_reserved": self.logical_agent_calls,
            "logical_agent_calls_dispatched_to_runtime": (
                self.logical_agent_calls_dispatched_to_runtime
            ),
            "logical_agent_calls_succeeded": self.logical_agent_calls_succeeded,
            "logical_agent_calls_abstained": self.logical_agent_calls_abstained,
            "logical_agent_calls_failed": self.logical_agent_calls_failed,
            "logical_agent_calls_cancelled_before_dispatch": (
                self.logical_agent_calls_cancelled_before_dispatch
            ),
            "logical_agent_calls_cancelled_after_dispatch": (
                self.logical_agent_calls_cancelled_after_dispatch
            ),
        }

    @property
    def counters_sha256(self) -> str:
        return _hash(_COUNTERS_DOMAIN, self.to_record())


class CampaignExecutionStatus(str, Enum):
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CampaignFinalizationRequest:
    preparation_sha256: str
    runtime_start_receipt_sha256: str
    status: CampaignExecutionStatus
    counters: CampaignExecutionCounters
    stage_receipt_sha256s: tuple[str, ...]
    reflection_receipt_sha256s: tuple[str, ...]
    reflection_cancellation_receipt_sha256s: tuple[str, ...]
    test_admission_receipt_sha256s: tuple[str, ...]
    tail_drain_receipt_sha256: str | None
    last_durable_event_sha256: str | None
    failure_type: str | None

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_start_receipt_sha256,
            "runtime_start_receipt_sha256",
        )
        if type(self.status) is not CampaignExecutionStatus:
            raise TypeError("status must be CampaignExecutionStatus")
        if type(self.counters) is not CampaignExecutionCounters:
            raise TypeError("counters must be exact")
        CampaignExecutionCounters.__post_init__(self.counters)
        for name in (
            "stage_receipt_sha256s",
            "reflection_receipt_sha256s",
            "reflection_cancellation_receipt_sha256s",
            "test_admission_receipt_sha256s",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str for value in values
            ):
                raise TypeError(f"{name} must be an exact string tuple")
            for value in values:
                require_sha256(value, name[:-1])
        for name in (
            "tail_drain_receipt_sha256",
            "last_durable_event_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        if self.status is CampaignExecutionStatus.COMPLETED:
            if self.failure_type is not None:
                raise ValueError("completed finalization cannot carry failure_type")
        elif type(self.failure_type) is not str or not self.failure_type:
            raise ValueError("failed finalization requires failure_type")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "status": self.status.value,
            "counters": self.counters.to_record(),
            "stage_receipt_sha256s": list(self.stage_receipt_sha256s),
            "reflection_receipt_sha256s": list(self.reflection_receipt_sha256s),
            "reflection_cancellation_receipt_sha256s": list(
                self.reflection_cancellation_receipt_sha256s
            ),
            "test_admission_receipt_sha256s": list(self.test_admission_receipt_sha256s),
            "tail_drain_receipt_sha256": self.tail_drain_receipt_sha256,
            "last_durable_event_sha256": self.last_durable_event_sha256,
            "failure_type": self.failure_type,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_FINALIZATION_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignFinalizationReceipt:
    request_sha256: str
    preparation_sha256: str
    status: CampaignExecutionStatus
    evidence: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.status) is not CampaignExecutionStatus:
            raise TypeError("status must be CampaignExecutionStatus")
        _frozen(self.evidence, name="evidence")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "status": self.status.value,
            "evidence": thaw_json(self.evidence),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_FINALIZATION_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CampaignCleanupRequest:
    preparation_sha256: str
    runtime_start_receipt_sha256: str
    status: CampaignExecutionStatus
    finalization_receipt_sha256: str | None
    failure_type: str | None

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_start_receipt_sha256,
            "runtime_start_receipt_sha256",
        )
        if type(self.status) is not CampaignExecutionStatus:
            raise TypeError("status must be CampaignExecutionStatus")
        if self.finalization_receipt_sha256 is not None:
            require_sha256(
                self.finalization_receipt_sha256,
                "finalization_receipt_sha256",
            )
        if self.status is CampaignExecutionStatus.COMPLETED:
            if self.failure_type is not None:
                raise ValueError("completed cleanup cannot carry failure_type")
        elif type(self.failure_type) is not str or not self.failure_type:
            raise ValueError("failed cleanup requires failure_type")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "status": self.status.value,
            "finalization_receipt_sha256": self.finalization_receipt_sha256,
            "failure_type": self.failure_type,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_CLEANUP_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class CampaignCleanupReceipt:
    request_sha256: str
    preparation_sha256: str
    released: bool
    evidence: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.released) is not bool:
            raise TypeError("released must be an exact bool")
        _frozen(self.evidence, name="evidence")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "request_sha256": self.request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "released": self.released,
            "evidence": thaw_json(self.evidence),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_CLEANUP_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


class CampaignExecutionEventKind(str, Enum):
    EXECUTION_STARTED = "execution_started"
    ARCHIVE_UTILITY_FROZEN = "archive_utility_frozen"
    STAGE_SEALED = "stage_sealed"
    REFLECTION_LAUNCHED = "reflection_launched"
    REFLECTION_COMPLETED = "reflection_completed"
    REFLECTION_ABSTAINED = "reflection_abstained"
    REFLECTION_FAILED = "reflection_failed"
    REFLECTION_CANCELLED = "reflection_cancelled"
    REFLECTION_ADMITTED_FOR_TESTING = "reflection_admitted_for_testing"
    REFLECTION_TAIL_DRAINED = "reflection_tail_drained"
    EXECUTION_DEGRADED = "execution_degraded"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_FINALIZED = "execution_finalized"
    RUNTIME_CLEANED = "runtime_cleaned"


@dataclass(frozen=True, slots=True)
class CampaignExecutionEvent:
    preparation_sha256: str
    sequence: int
    kind: CampaignExecutionEventKind
    previous_event_sha256: str | None
    payload: FrozenJsonObject

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("event sequence must be positive")
        if type(self.kind) is not CampaignExecutionEventKind:
            raise TypeError("event kind must be CampaignExecutionEventKind")
        if self.previous_event_sha256 is not None:
            require_sha256(self.previous_event_sha256, "previous_event_sha256")
        if self.sequence == 1 and self.previous_event_sha256 is not None:
            raise ValueError("first event cannot name a predecessor")
        if self.sequence > 1 and self.previous_event_sha256 is None:
            raise ValueError("later event requires a predecessor")
        _frozen(self.payload, name="payload")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "previous_event_sha256": self.previous_event_sha256,
            "payload": thaw_json(self.payload),
        }

    @property
    def event_sha256(self) -> str:
        return _hash(_EVENT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "event_sha256": self.event_sha256}


@dataclass(frozen=True, slots=True)
class CampaignJournalAck:
    event_sha256: str
    durable: bool

    def __post_init__(self) -> None:
        require_sha256(self.event_sha256, "event_sha256")
        if self.durable is not True:
            raise ValueError("campaign journal acknowledgement must be durable")


@runtime_checkable
class CampaignStageRuntimePort(Protocol):
    async def snapshot_archive(
        self,
        request: CampaignArchiveCutoffRequest,
    ) -> CampaignArchiveCutoffReceipt: ...

    async def execute_stage(
        self,
        request: CampaignStageRequest,
    ) -> CampaignStageReceipt: ...


@runtime_checkable
class CampaignReflectionRuntimePort(Protocol):
    async def reflect(
        self,
        request: CampaignReflectionRequest,
    ) -> CampaignReflectionReceipt: ...

    async def admit_for_testing(
        self,
        request: CampaignReflectionTestAdmissionRequest,
    ) -> CampaignReflectionTestAdmissionReceipt: ...


@runtime_checkable
class CampaignExecutionLifecyclePort(Protocol):
    """Own runtime resources around one scheduler execution.

    ``start`` is transaction-like: if it raises before returning a valid start
    receipt, it must release anything it acquired.  After a valid receipt is
    returned, the scheduler calls ``finalize`` and ``cleanup`` on every success,
    stage failure, receipt rejection, cancellation, or durable-journal failure.
    ``cleanup`` must tolerate a failed ``finalize`` request.
    """

    async def start(
        self,
        prepared: PreparedEvolutionCampaign,
    ) -> CampaignExecutionStartReceipt: ...

    async def finalize(
        self,
        request: CampaignFinalizationRequest,
    ) -> CampaignFinalizationReceipt: ...

    async def cleanup(
        self,
        request: CampaignCleanupRequest,
    ) -> CampaignCleanupReceipt: ...


@runtime_checkable
class CampaignExecutionJournalPort(Protocol):
    async def append(self, event: CampaignExecutionEvent) -> CampaignJournalAck: ...


@dataclass(frozen=True, slots=True)
class CampaignExecutionResult:
    preparation_sha256: str
    runtime_start_receipt_sha256: str
    stage_receipts: tuple[CampaignStageReceipt, ...]
    reflection_receipts: tuple[CampaignReflectionReceipt, ...]
    reflection_cancellation_receipts: tuple[CampaignReflectionCancellationReceipt, ...]
    test_admission_receipts: tuple[CampaignReflectionTestAdmissionReceipt, ...]
    tail_drain_receipt: CampaignTailDrainReceipt | None
    counters: CampaignExecutionCounters
    finalization_receipt: CampaignFinalizationReceipt
    cleanup_receipt: CampaignCleanupReceipt
    durable_event_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.runtime_start_receipt_sha256,
            "runtime_start_receipt_sha256",
        )
        if type(self.stage_receipts) is not tuple or not self.stage_receipts:
            raise ValueError("execution result requires stage receipts")
        if any(
            type(value) is not CampaignStageReceipt for value in self.stage_receipts
        ):
            raise TypeError("stage_receipts must contain exact receipts")
        if type(self.reflection_receipts) is not tuple or any(
            type(value) is not CampaignReflectionReceipt
            for value in self.reflection_receipts
        ):
            raise TypeError("reflection_receipts must contain exact receipts")
        if type(self.reflection_cancellation_receipts) is not tuple or any(
            type(value) is not CampaignReflectionCancellationReceipt
            for value in self.reflection_cancellation_receipts
        ):
            raise TypeError(
                "reflection_cancellation_receipts must contain exact receipts"
            )
        if type(self.test_admission_receipts) is not tuple or any(
            type(value) is not CampaignReflectionTestAdmissionReceipt
            for value in self.test_admission_receipts
        ):
            raise TypeError("test_admission_receipts must contain exact receipts")
        if (
            self.tail_drain_receipt is not None
            and type(self.tail_drain_receipt) is not CampaignTailDrainReceipt
        ):
            raise TypeError("tail_drain_receipt must be exact or None")
        if type(self.counters) is not CampaignExecutionCounters:
            raise TypeError("counters must be exact")
        CampaignExecutionCounters.__post_init__(self.counters)
        if type(self.finalization_receipt) is not CampaignFinalizationReceipt:
            raise TypeError("finalization_receipt must be exact")
        if type(self.cleanup_receipt) is not CampaignCleanupReceipt:
            raise TypeError("cleanup_receipt must be exact")
        if not self.cleanup_receipt.released:
            raise ValueError("successful execution requires released runtime resources")
        if type(self.durable_event_sha256s) is not tuple or any(
            type(value) is not str for value in self.durable_event_sha256s
        ):
            raise TypeError("durable event hashes must be an exact string tuple")
        for value in self.durable_event_sha256s:
            require_sha256(value, "durable_event_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "stage_receipt_sha256s": [
                value.receipt_sha256 for value in self.stage_receipts
            ],
            "reflection_receipt_sha256s": [
                value.receipt_sha256 for value in self.reflection_receipts
            ],
            "reflection_cancellation_receipt_sha256s": [
                value.receipt_sha256 for value in self.reflection_cancellation_receipts
            ],
            "test_admission_receipt_sha256s": [
                value.receipt_sha256 for value in self.test_admission_receipts
            ],
            "tail_drain_receipt_sha256": (
                None
                if self.tail_drain_receipt is None
                else self.tail_drain_receipt.receipt_sha256
            ),
            "counters": self.counters.to_record(),
            "finalization_receipt_sha256": self.finalization_receipt.receipt_sha256,
            "cleanup_receipt_sha256": self.cleanup_receipt.receipt_sha256,
            "durable_event_sha256s": list(self.durable_event_sha256s),
        }

    @property
    def result_sha256(self) -> str:
        return _hash(_RESULT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "result_sha256": self.result_sha256}


@dataclass(slots=True)
class EvolutionCampaignScheduler:
    """Execute a prepared schedule while owning its scientific ledger."""

    prepared: PreparedEvolutionCampaign
    policies: CampaignPolicies
    stages: CampaignStageRuntimePort
    reflections: CampaignReflectionRuntimePort
    lifecycle: CampaignExecutionLifecyclePort
    journal: CampaignExecutionJournalPort
    _has_run: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if type(self.prepared) is not PreparedEvolutionCampaign:
            raise TypeError("prepared must be an exact PreparedEvolutionCampaign")
        PreparedEvolutionCampaign.__post_init__(self.prepared)
        if type(self.policies) is not CampaignPolicies:
            raise TypeError("policies must be an exact CampaignPolicies")
        CampaignPolicies.__post_init__(self.policies)
        if self.policies.policies_sha256 != self.prepared.policies_sha256:
            raise ValueError("execution policies differ from prepared policies")
        if not isinstance(self.stages, CampaignStageRuntimePort):
            raise TypeError("stages must implement CampaignStageRuntimePort")
        if not isinstance(self.reflections, CampaignReflectionRuntimePort):
            raise TypeError("reflections must implement CampaignReflectionRuntimePort")
        if not isinstance(self.lifecycle, CampaignExecutionLifecyclePort):
            raise TypeError("lifecycle must implement CampaignExecutionLifecyclePort")
        if not isinstance(self.journal, CampaignExecutionJournalPort):
            raise TypeError("journal must implement CampaignExecutionJournalPort")

    async def run(self) -> CampaignExecutionResult:
        """Run every prepared generation and always finalize and clean up."""

        self.__post_init__()
        if self._has_run:
            raise RuntimeError("campaign scheduler instances are one-shot")
        self._has_run = True
        preparation_sha256 = self.prepared.preparation_sha256
        start = await self.lifecycle.start(self.prepared)
        if type(start) is not CampaignExecutionStartReceipt:
            raise TypeError("lifecycle start must return CampaignExecutionStartReceipt")
        CampaignExecutionStartReceipt.__post_init__(start)

        event_sequence = 0
        last_event_sha256: str | None = None
        durable_event_sha256s: list[str] = []

        async def append_event(
            kind: CampaignExecutionEventKind,
            payload: dict[str, object],
        ) -> None:
            nonlocal event_sequence, last_event_sha256
            event = CampaignExecutionEvent(
                preparation_sha256=preparation_sha256,
                sequence=event_sequence + 1,
                kind=kind,
                previous_event_sha256=last_event_sha256,
                payload=_freeze_record(payload),
            )
            ack = await self.journal.append(event)
            if type(ack) is not CampaignJournalAck:
                raise TypeError("journal must return CampaignJournalAck")
            CampaignJournalAck.__post_init__(ack)
            if ack.event_sha256 != event.event_sha256:
                raise CampaignExecutionContractError(
                    "journal acknowledged a foreign event"
                )
            event_sequence = event.sequence
            last_event_sha256 = event.event_sha256
            durable_event_sha256s.append(event.event_sha256)

        counters = CampaignExecutionCounters(
            generations_completed=0,
            candidate_occurrences=0,
            unique_evaluations=0,
            logical_agent_calls=0,
        )
        stage_receipts: list[CampaignStageReceipt] = []
        reflection_receipts: dict[int, CampaignReflectionReceipt] = {}
        reflection_cancellations: list[CampaignReflectionCancellationReceipt] = []
        test_admission_receipts: list[CampaignReflectionTestAdmissionReceipt] = []
        tail_drain: CampaignTailDrainReceipt | None = None
        selector_audits: list[CampaignSelectorAuditReceipt] = []
        test_eligible_reflections: tuple[str, ...] = ()
        reflection_tasks: dict[int, asyncio.Task[CampaignReflectionReceipt]] = {}
        reflection_requests: dict[int, CampaignReflectionRequest] = {}
        reflection_runtime_dispatched: set[int] = set()
        degraded_reflection_sources: set[int] = set()
        abstained_reflection_sources: set[int] = set()
        finalization: CampaignFinalizationReceipt | None = None
        cleanup: CampaignCleanupReceipt | None = None
        primary_error: BaseException | None = None

        supervision = self.policies.reflection_supervision
        if type(supervision) is not CampaignReflectionSupervisionPolicy:
            raise TypeError("campaign has a foreign reflection supervision policy")
        CampaignReflectionSupervisionPolicy.__post_init__(supervision)

        def update_call_counters(
            *,
            reserved: int = 0,
            dispatched: int = 0,
            succeeded: int = 0,
            abstained: int = 0,
            failed: int = 0,
            cancelled_before_dispatch: int = 0,
            cancelled_after_dispatch: int = 0,
        ) -> None:
            nonlocal counters
            counters = CampaignExecutionCounters(
                generations_completed=counters.generations_completed,
                candidate_occurrences=counters.candidate_occurrences,
                unique_evaluations=counters.unique_evaluations,
                logical_agent_calls=counters.logical_agent_calls + reserved,
                logical_agent_calls_dispatched_to_runtime=(
                    counters.logical_agent_calls_dispatched_to_runtime + dispatched
                ),
                logical_agent_calls_succeeded=(
                    counters.logical_agent_calls_succeeded + succeeded
                ),
                logical_agent_calls_abstained=(
                    counters.logical_agent_calls_abstained + abstained
                ),
                logical_agent_calls_failed=(
                    counters.logical_agent_calls_failed + failed
                ),
                logical_agent_calls_cancelled_before_dispatch=(
                    counters.logical_agent_calls_cancelled_before_dispatch
                    + cancelled_before_dispatch
                ),
                logical_agent_calls_cancelled_after_dispatch=(
                    counters.logical_agent_calls_cancelled_after_dispatch
                    + cancelled_after_dispatch
                ),
            )

        def failed_reflection_receipt(
            source_generation: int,
            error: BaseException,
        ) -> CampaignReflectionReceipt:
            request = reflection_requests[source_generation]
            failure_type = type(error).__name__
            failure_digest = hashlib.sha256(
                type(error).__qualname__.encode("utf-8", errors="strict")
                + b"\x00"
                + str(error).encode("utf-8", errors="replace")
            ).hexdigest()
            return CampaignReflectionReceipt(
                request_sha256=request.request_sha256,
                preparation_sha256=preparation_sha256,
                source_generation=source_generation,
                source_stage_receipt_sha256=(request.source_stage.receipt_sha256),
                logical_agent_calls=request.wave.call_count,
                visibility=request.wave.visibility,
                status=CampaignReflectionStatus.FAILED,
                failure_type=failure_type,
                quarantined_result=_freeze_record(
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "failure_type": failure_type,
                        "failure_digest_sha256": failure_digest,
                        "approved_summary": (
                            "reflection execution failed without publishable "
                            "insight content"
                        ),
                        "publishable_reflection_content": False,
                    }
                ),
            )

        async def run_reflection(
            request: CampaignReflectionRequest,
            started: asyncio.Event,
        ) -> CampaignReflectionReceipt:
            source_generation = request.wave.source_generation
            reflection_runtime_dispatched.add(source_generation)
            started.set()
            try:
                return await self.reflections.reflect(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                return failed_reflection_receipt(source_generation, error)

        async def settle_reflection(
            source_generation: int,
            receipt: CampaignReflectionReceipt,
        ) -> CampaignReflectionReceipt:
            if source_generation in reflection_receipts:
                if reflection_receipts[source_generation] != receipt:
                    raise CampaignExecutionContractError(
                        "reflection source settled with two different receipts"
                    )
                return receipt
            request = reflection_requests[source_generation]
            if type(receipt) is not CampaignReflectionReceipt:
                raise TypeError("reflection runtime must return exact receipt")
            CampaignReflectionReceipt.__post_init__(receipt)
            if (
                receipt.request_sha256 != request.request_sha256
                or receipt.preparation_sha256 != preparation_sha256
                or receipt.source_generation != source_generation
                or receipt.source_stage_receipt_sha256
                != request.source_stage.receipt_sha256
                or receipt.logical_agent_calls != request.wave.call_count
            ):
                raise CampaignExecutionContractError(
                    "reflection receipt differs from its launch request"
                )
            reflection_receipts[source_generation] = receipt
            reflection_tasks.pop(source_generation, None)
            if receipt.status is CampaignReflectionStatus.COMPLETED:
                update_call_counters(succeeded=receipt.logical_agent_calls)
                kind = CampaignExecutionEventKind.REFLECTION_COMPLETED
            elif receipt.status is CampaignReflectionStatus.ABSTAINED:
                update_call_counters(abstained=receipt.logical_agent_calls)
                abstained_reflection_sources.add(source_generation)
                kind = CampaignExecutionEventKind.REFLECTION_ABSTAINED
            else:
                update_call_counters(failed=receipt.logical_agent_calls)
                kind = CampaignExecutionEventKind.REFLECTION_FAILED
            await append_event(
                kind,
                {"reflection_receipt": receipt.to_record()},
            )
            return receipt

        async def record_reflection_cancellation(
            source_generation: int,
            *,
            reason: str,
        ) -> CampaignReflectionCancellationReceipt:
            request = reflection_requests[source_generation]
            dispatched = source_generation in reflection_runtime_dispatched
            receipt = CampaignReflectionCancellationReceipt(
                request_sha256=request.request_sha256,
                preparation_sha256=preparation_sha256,
                source_generation=source_generation,
                source_stage_receipt_sha256=request.source_stage.receipt_sha256,
                logical_agent_calls_reserved=request.wave.call_count,
                dispatched_to_runtime=dispatched,
                cancellation_reason=reason,
            )
            reflection_cancellations.append(receipt)
            if dispatched:
                update_call_counters(cancelled_after_dispatch=request.wave.call_count)
            else:
                update_call_counters(cancelled_before_dispatch=request.wave.call_count)
            await append_event(
                CampaignExecutionEventKind.REFLECTION_CANCELLED,
                {"reflection_cancellation_receipt": receipt.to_record()},
            )
            return receipt

        async def settle_sources(
            sources: tuple[int, ...],
            *,
            wait_for_all: bool,
        ) -> tuple[CampaignReflectionReceipt, ...]:
            pending_sources = tuple(
                source
                for source in sources
                if source not in reflection_receipts
                and source in reflection_tasks
                and (wait_for_all or reflection_tasks[source].done())
            )
            if pending_sources:
                raw = await asyncio.gather(
                    *(reflection_tasks[source] for source in pending_sources),
                    return_exceptions=True,
                )
                for source, value in zip(pending_sources, raw, strict=True):
                    if isinstance(value, asyncio.CancelledError):
                        await record_reflection_cancellation(
                            source,
                            reason="scheduler_task_cancelled",
                        )
                        reflection_tasks.pop(source, None)
                    elif isinstance(value, BaseException):
                        await settle_reflection(
                            source,
                            failed_reflection_receipt(source, value),
                        )
                    else:
                        await settle_reflection(source, value)
            return tuple(
                reflection_receipts[source]
                for source in sources
                if source in reflection_receipts
            )

        def failed_sources(sources: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(
                source
                for source in sources
                if source in reflection_receipts
                and reflection_receipts[source].status
                is CampaignReflectionStatus.FAILED
            )

        def abstained_sources(sources: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(
                source
                for source in sources
                if source in reflection_receipts
                and reflection_receipts[source].status
                is CampaignReflectionStatus.ABSTAINED
            )

        async def settle_stage_boundary() -> None:
            # Give callbacks made runnable by the just-sealed stage one turn to
            # publish their terminal task state before the next expensive stage.
            await asyncio.sleep(0)
            sources = tuple(sorted(reflection_tasks))
            await settle_sources(sources, wait_for_all=False)
            failures = failed_sources(tuple(sorted(reflection_receipts)))
            if not failures:
                return
            if supervision.mode is ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY:
                raise CampaignExecutionContractError(
                    "reflection failed at the next durable stage boundary"
                )
            if supervision.mode is ReflectionFailureMode.BEST_EFFORT_DEGRADED:
                degraded_reflection_sources.update(failures)

        async def join_barrier(barrier: CampaignPromotionBarrier) -> None:
            nonlocal test_eligible_reflections
            sources = barrier.reflection_source_generations
            if any(
                source not in reflection_tasks and source not in reflection_receipts
                for source in sources
            ):
                raise CampaignExecutionContractError(
                    "test-admission barrier lacks a launched reflection"
                )
            await settle_sources(sources, wait_for_all=True)
            joined = tuple(reflection_receipts[source] for source in sources)
            failures = failed_sources(sources)
            if failures:
                for source in sources:
                    reflection_tasks.pop(source, None)
                    reflection_requests.pop(source, None)
                if supervision.mode is ReflectionFailureMode.BEST_EFFORT_DEGRADED:
                    degraded_reflection_sources.update(failures)
                    return
                raise CampaignExecutionContractError(
                    "reflection block closed with one or more failed receipts"
                )
            abstentions = abstained_sources(sources)
            if abstentions:
                # A block containing E0 evidence has no publishable reflection
                # content.  Preserve every receipt, but do not manufacture a
                # partial admission whose estimand differs from the prepared
                # barrier.
                for source in sources:
                    reflection_tasks.pop(source, None)
                    reflection_requests.pop(source, None)
                return
            request = CampaignReflectionTestAdmissionRequest(
                preparation_sha256=preparation_sha256,
                runtime_start_receipt_sha256=start.receipt_sha256,
                barrier=barrier,
                reflections=joined,
                previously_test_eligible_reflection_receipt_sha256s=(
                    test_eligible_reflections
                ),
                admission_cutoff_stage=(
                    stage_receipts[barrier.generation - 1]
                    if barrier.generation
                    > barrier.reflection_source_generations[-1]
                    else None
                ),
            )
            receipt = await self.reflections.admit_for_testing(request)
            if type(receipt) is not CampaignReflectionTestAdmissionReceipt:
                raise TypeError("test admission runtime must return exact receipt")
            CampaignReflectionTestAdmissionReceipt.__post_init__(receipt)
            admitted = tuple(sorted(value.receipt_sha256 for value in joined))
            expected_eligible = tuple(sorted((*test_eligible_reflections, *admitted)))
            if (
                receipt.request_sha256 != request.request_sha256
                or receipt.preparation_sha256 != preparation_sha256
                or receipt.barrier_generation != barrier.generation
                or receipt.admitted_reflection_receipt_sha256s != admitted
                or receipt.test_eligible_reflection_receipt_sha256s != expected_eligible
                or receipt.lifecycle_promoted
            ):
                raise CampaignExecutionContractError(
                    "test-admission receipt differs from its barrier"
                )
            test_admission_receipts.append(receipt)
            test_eligible_reflections = expected_eligible
            await append_event(
                CampaignExecutionEventKind.REFLECTION_ADMITTED_FOR_TESTING,
                {"test_admission_receipt": receipt.to_record()},
            )
            for source in sources:
                reflection_tasks.pop(source, None)
                reflection_requests.pop(source, None)

        try:
            if (
                start.preparation_sha256 != preparation_sha256
                or start.runtime_preflight_receipt_sha256
                != self.prepared.runtime_receipt.receipt_sha256
                or start.seed_batch_sha256 != self.prepared.seeds.batch_sha256
            ):
                raise CampaignExecutionContractError(
                    "runtime start receipt differs from preparation"
                )
            expected_seeds = tuple(
                (value.seed_id, value.configuration_sha256)
                for value in self.prepared.seeds.seeds
            )
            observed_seeds = tuple(
                (value.seed_id, value.configuration_sha256)
                for value in start.seed_receipts
            )
            if observed_seeds != expected_seeds:
                raise CampaignExecutionContractError(
                    "runtime seed accounting differs from prepared seed batch"
                )
            counters = CampaignExecutionCounters(
                generations_completed=0,
                candidate_occurrences=start.seed_occurrence_count,
                unique_evaluations=start.seed_unique_evaluation_count,
                logical_agent_calls=0,
            )
            if (
                counters.unique_evaluations
                > self.prepared.budget.max_unique_evaluations
            ):
                raise CampaignExecutionContractError(
                    "seed evaluations already exceed campaign budget"
                )
            await append_event(
                CampaignExecutionEventKind.EXECUTION_STARTED,
                {
                    "start_receipt": start.to_record(),
                    "initial_counters": counters.to_record(),
                },
            )
            if start.seed_failure_count or any(
                not value.evaluated or not value.valid for value in start.seed_receipts
            ):
                raise CampaignExecutionContractError(
                    "runtime start reported failed or unevaluated seeds"
                )
            barrier_by_generation = {
                value.generation: value
                for value in self.prepared.schedule.promotion_barriers
            }
            wave_by_generation = {
                value.source_generation: value
                for value in self.prepared.schedule.reflection_waves
            }
            prior_stage: CampaignStageReceipt | None = None

            for step in self.prepared.schedule.steps:
                if step.generation != counters.generations_completed + 1:
                    raise CampaignExecutionContractError(
                        "prepared schedule is not the next generation"
                    )
                expected_stage_agent_calls = (
                    step.parent_count
                    if step.kind is CampaignGenerationKind.PORTFOLIO
                    else 0
                )
                if (
                    counters.logical_agent_calls + expected_stage_agent_calls
                    > self.prepared.budget.max_logical_llm_calls
                ):
                    raise CampaignExecutionContractError(
                        "selector calls would exceed campaign budget"
                    )
                cutoff_request = CampaignArchiveCutoffRequest(
                    preparation_sha256=preparation_sha256,
                    runtime_start_receipt_sha256=start.receipt_sha256,
                    generation=step.generation,
                    kind=step.kind,
                    step_sha256=campaign_step_sha256(step),
                    prior_stage_receipt_sha256=(
                        None if prior_stage is None else prior_stage.receipt_sha256
                    ),
                )
                cutoff = await self.stages.snapshot_archive(cutoff_request)
                if type(cutoff) is not CampaignArchiveCutoffReceipt:
                    raise TypeError("stage runtime must return exact archive cutoff")
                CampaignArchiveCutoffReceipt.__post_init__(cutoff)
                if (
                    cutoff.request_sha256 != cutoff_request.request_sha256
                    or cutoff.preparation_sha256 != preparation_sha256
                    or cutoff.generation != step.generation
                ):
                    raise CampaignExecutionContractError(
                        "archive cutoff receipt differs from its request"
                    )
                utility = freeze_archive_utility(
                    self.policies.archive_utility,
                    benchmark=self.prepared.benchmark_session.benchmark,
                    generation=step.generation,
                    archive=cutoff.archive,
                )
                await append_event(
                    CampaignExecutionEventKind.ARCHIVE_UTILITY_FROZEN,
                    {
                        "archive_cutoff": cutoff.to_record(),
                        "archive_utility": utility.to_record(),
                    },
                )

                source_portfolio: CampaignStageReceipt | None = None
                if step.kind is CampaignGenerationKind.RECOMBINATION:
                    if (
                        prior_stage is None
                        or prior_stage.kind is not CampaignGenerationKind.PORTFOLIO
                        or prior_stage.generation != step.source_portfolio_generation
                    ):
                        raise CampaignExecutionContractError(
                            "recombination lacks its exact portfolio predecessor"
                        )
                    source_portfolio = prior_stage
                stage_request = CampaignStageRequest(
                    preparation_sha256=preparation_sha256,
                    runtime_start_receipt_sha256=start.receipt_sha256,
                    step=step,
                    archive_cutoff=cutoff,
                    archive_utility=utility,
                    source_portfolio=source_portfolio,
                    test_eligible_reflection_receipt_sha256s=(
                        test_eligible_reflections
                    ),
                    prior_selector_audit_set_sha256=selector_audit_set_sha256(
                        tuple(selector_audits)
                    ),
                )
                stage = await self.stages.execute_stage(stage_request)
                if type(stage) is not CampaignStageReceipt:
                    raise TypeError("stage runtime must return exact stage receipt")
                CampaignStageReceipt.__post_init__(stage)
                if (
                    stage.request_sha256 != stage_request.request_sha256
                    or stage.preparation_sha256 != preparation_sha256
                    or stage.generation != step.generation
                    or stage.kind is not step.kind
                ):
                    raise CampaignExecutionContractError(
                        "stage receipt is out-of-order or foreign"
                    )
                maximum_occurrences = step.planned_candidate_evaluations
                if step.kind is CampaignGenerationKind.PORTFOLIO:
                    if stage.candidate_occurrence_count != maximum_occurrences:
                        raise CampaignExecutionContractError(
                            "portfolio stage did not account for every planned child"
                        )
                    if len(stage.selector_audits) != step.parent_count:
                        raise CampaignExecutionContractError(
                            "portfolio stage lacks one fresh selector audit per parent"
                        )
                    expected_slots = tuple(range(step.parent_count))
                    if (
                        tuple(value.parent_slot for value in stage.selector_audits)
                        != expected_slots
                    ):
                        raise CampaignExecutionContractError(
                            "selector audits do not cover canonical parent slots"
                        )
                    prior_audit_hash = selector_audit_set_sha256(tuple(selector_audits))
                    if any(
                        value.generation != step.generation
                        or value.prior_audit_set_sha256 != prior_audit_hash
                        for value in stage.selector_audits
                    ):
                        raise CampaignExecutionContractError(
                            "selector audit is stale or names another generation"
                        )
                    existing_calls = {
                        value.selector_call_id for value in selector_audits
                    }
                    existing_requests = {
                        value.request_sha256 for value in selector_audits
                    }
                    existing_traces = {
                        value.trace_receipt_sha256 for value in selector_audits
                    }
                    if (
                        len({value.selector_call_id for value in stage.selector_audits})
                        != len(stage.selector_audits)
                        or len(
                            {value.request_sha256 for value in stage.selector_audits}
                        )
                        != len(stage.selector_audits)
                        or len(
                            {
                                value.trace_receipt_sha256
                                for value in stage.selector_audits
                            }
                        )
                        != len(stage.selector_audits)
                        or any(
                            value.selector_call_id in existing_calls
                            or value.request_sha256 in existing_requests
                            or value.trace_receipt_sha256 in existing_traces
                            for value in stage.selector_audits
                        )
                    ):
                        raise CampaignExecutionContractError(
                            "selector audit is duplicate rather than fresh"
                        )
                    selector_audits.extend(stage.selector_audits)
                else:
                    if stage.candidate_occurrence_count > maximum_occurrences:
                        raise CampaignExecutionContractError(
                            "recombination exceeded its planned child envelope"
                        )
                    if stage.selector_audits:
                        raise CampaignExecutionContractError(
                            "recombination stage cannot publish selector audits"
                        )
                next_unique = (
                    counters.unique_evaluations + stage.unique_evaluation_count
                )
                if next_unique > self.prepared.budget.max_unique_evaluations:
                    raise CampaignExecutionContractError(
                        "stage evaluations exceed campaign budget"
                    )
                counters = CampaignExecutionCounters(
                    generations_completed=step.generation,
                    candidate_occurrences=(
                        counters.candidate_occurrences
                        + stage.candidate_occurrence_count
                    ),
                    unique_evaluations=next_unique,
                    logical_agent_calls=(
                        counters.logical_agent_calls + expected_stage_agent_calls
                    ),
                    logical_agent_calls_dispatched_to_runtime=(
                        counters.logical_agent_calls_dispatched_to_runtime
                        + expected_stage_agent_calls
                    ),
                    logical_agent_calls_succeeded=(
                        counters.logical_agent_calls_succeeded
                        + expected_stage_agent_calls
                    ),
                    logical_agent_calls_abstained=(
                        counters.logical_agent_calls_abstained
                    ),
                    logical_agent_calls_failed=(counters.logical_agent_calls_failed),
                    logical_agent_calls_cancelled_before_dispatch=(
                        counters.logical_agent_calls_cancelled_before_dispatch
                    ),
                    logical_agent_calls_cancelled_after_dispatch=(
                        counters.logical_agent_calls_cancelled_after_dispatch
                    ),
                )
                stage_receipts.append(stage)
                prior_stage = stage
                await append_event(
                    CampaignExecutionEventKind.STAGE_SEALED,
                    {
                        "stage_receipt": stage.to_record(),
                        "counters": counters.to_record(),
                    },
                )
                await settle_stage_boundary()

                wave = wave_by_generation.get(step.generation)
                if wave is not None:
                    if step.kind is not CampaignGenerationKind.RECOMBINATION:
                        raise CampaignExecutionContractError(
                            "reflection wave follows a non-recombination stage"
                        )
                    if (
                        counters.logical_agent_calls + wave.call_count
                        > self.prepared.budget.max_logical_llm_calls
                    ):
                        raise CampaignExecutionContractError(
                            "reflection calls would exceed campaign budget"
                        )
                    reflection_request = CampaignReflectionRequest(
                        preparation_sha256=preparation_sha256,
                        runtime_start_receipt_sha256=start.receipt_sha256,
                        wave=wave,
                        source_stage=stage,
                    )
                    await append_event(
                        CampaignExecutionEventKind.REFLECTION_LAUNCHED,
                        {"reflection_request": reflection_request.to_record()},
                    )
                    reflection_requests[step.generation] = reflection_request
                    update_call_counters(reserved=wave.call_count)
                    started = asyncio.Event()
                    reflection_tasks[step.generation] = asyncio.create_task(
                        run_reflection(reflection_request, started)
                    )
                    await started.wait()
                    update_call_counters(dispatched=wave.call_count)
                    await settle_stage_boundary()

                barrier = barrier_by_generation.get(step.generation)
                if barrier is not None:
                    await join_barrier(barrier)

            if (
                counters.generations_completed != len(self.prepared.schedule.steps)
                or counters.generations_completed > self.prepared.budget.max_generations
            ):
                raise CampaignExecutionContractError(
                    "execution did not close the exact prepared generation count"
                )

            if reflection_requests:
                tail_sources = tuple(sorted(reflection_requests))
                if any(
                    reflection_requests[source].wave.promotion_barrier_generation
                    is not None
                    for source in tail_sources
                ):
                    raise CampaignExecutionContractError(
                        "a declared promotion barrier was not joined"
                    )
                await settle_sources(tail_sources, wait_for_all=True)
                drained = tuple(reflection_receipts[source] for source in tail_sources)
                failures = failed_sources(tail_sources)
                if failures and supervision.mode is not (
                    ReflectionFailureMode.BEST_EFFORT_DEGRADED
                ):
                    raise CampaignExecutionContractError(
                        "reflection tail closed with one or more failed receipts"
                    )
                degraded_reflection_sources.update(failures)
                tail_drain = CampaignTailDrainReceipt(
                    preparation_sha256=preparation_sha256,
                    drained_reflections=tuple(
                        (value.source_generation, value.receipt_sha256)
                        for value in drained
                    ),
                )
                await append_event(
                    CampaignExecutionEventKind.REFLECTION_TAIL_DRAINED,
                    {"tail_drain_receipt": tail_drain.to_record()},
                )
                reflection_tasks.clear()
                reflection_requests.clear()
        except BaseException as exc:
            primary_error = exc
        finally:
            if reflection_tasks:
                pending_sources = tuple(sorted(reflection_tasks))
                for task in reflection_tasks.values():
                    if not task.done():
                        task.cancel()
                raw = await asyncio.gather(
                    *(reflection_tasks[source] for source in pending_sources),
                    return_exceptions=True,
                )
                for source, value in zip(pending_sources, raw, strict=True):
                    try:
                        if isinstance(value, asyncio.CancelledError):
                            await record_reflection_cancellation(
                                source,
                                reason="campaign_abort",
                            )
                        elif isinstance(value, BaseException):
                            await settle_reflection(
                                source,
                                failed_reflection_receipt(source, value),
                            )
                        else:
                            await settle_reflection(source, value)
                    except BaseException as settlement_error:
                        if primary_error is None:
                            primary_error = settlement_error
                reflection_tasks.clear()
            reflection_requests.clear()

            if primary_error is not None:
                status = CampaignExecutionStatus.FAILED
                failure_type = type(primary_error).__name__
            elif degraded_reflection_sources:
                status = CampaignExecutionStatus.DEGRADED
                failure_type = "reflection_failure"
                try:
                    await append_event(
                        CampaignExecutionEventKind.EXECUTION_DEGRADED,
                        {
                            "approved_summary": (
                                "campaign completed with quarantined reflection "
                                "failures"
                            ),
                            "failed_reflection_sources": sorted(
                                degraded_reflection_sources
                            ),
                            "counters": counters.to_record(),
                        },
                    )
                except BaseException as degraded_event_error:
                    primary_error = degraded_event_error
                    status = CampaignExecutionStatus.FAILED
                    failure_type = type(degraded_event_error).__name__
            else:
                status = CampaignExecutionStatus.COMPLETED
                failure_type = None
            if primary_error is not None:
                failure_digest = hashlib.sha256(
                    type(primary_error).__qualname__.encode("utf-8", errors="strict")
                    + b"\x00"
                    + str(primary_error).encode("utf-8", errors="replace")
                ).hexdigest()
                try:
                    await append_event(
                        CampaignExecutionEventKind.EXECUTION_FAILED,
                        {
                            "failure_type": failure_type,
                            "failure_digest_sha256": failure_digest,
                            "approved_summary": "campaign execution failed",
                            "counters": counters.to_record(),
                        },
                    )
                except BaseException:
                    pass

            finalize_request = CampaignFinalizationRequest(
                preparation_sha256=preparation_sha256,
                runtime_start_receipt_sha256=start.receipt_sha256,
                status=status,
                counters=counters,
                stage_receipt_sha256s=tuple(
                    value.receipt_sha256 for value in stage_receipts
                ),
                reflection_receipt_sha256s=tuple(
                    reflection_receipts[source].receipt_sha256
                    for source in sorted(reflection_receipts)
                ),
                reflection_cancellation_receipt_sha256s=tuple(
                    value.receipt_sha256 for value in reflection_cancellations
                ),
                test_admission_receipt_sha256s=tuple(
                    value.receipt_sha256 for value in test_admission_receipts
                ),
                tail_drain_receipt_sha256=(
                    None if tail_drain is None else tail_drain.receipt_sha256
                ),
                last_durable_event_sha256=last_event_sha256,
                failure_type=failure_type,
            )
            try:
                candidate_finalization = await self.lifecycle.finalize(finalize_request)
                if type(candidate_finalization) is not CampaignFinalizationReceipt:
                    raise TypeError(
                        "lifecycle finalize must return CampaignFinalizationReceipt"
                    )
                CampaignFinalizationReceipt.__post_init__(candidate_finalization)
                if (
                    candidate_finalization.request_sha256
                    != finalize_request.request_sha256
                    or candidate_finalization.preparation_sha256 != preparation_sha256
                    or candidate_finalization.status is not status
                ):
                    raise CampaignExecutionContractError(
                        "finalization receipt differs from request"
                    )
                finalization = candidate_finalization
                await append_event(
                    CampaignExecutionEventKind.EXECUTION_FINALIZED,
                    {"finalization_receipt": finalization.to_record()},
                )
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
                    status = CampaignExecutionStatus.FAILED
                    failure_type = type(exc).__name__

            cleanup_request = CampaignCleanupRequest(
                preparation_sha256=preparation_sha256,
                runtime_start_receipt_sha256=start.receipt_sha256,
                status=status,
                finalization_receipt_sha256=(
                    None if finalization is None else finalization.receipt_sha256
                ),
                failure_type=failure_type,
            )
            try:
                cleanup = await self.lifecycle.cleanup(cleanup_request)
                if type(cleanup) is not CampaignCleanupReceipt:
                    raise TypeError(
                        "lifecycle cleanup must return CampaignCleanupReceipt"
                    )
                CampaignCleanupReceipt.__post_init__(cleanup)
                if (
                    cleanup.request_sha256 != cleanup_request.request_sha256
                    or cleanup.preparation_sha256 != preparation_sha256
                    or not cleanup.released
                ):
                    raise CampaignExecutionContractError(
                        "cleanup receipt differs from request or retained resources"
                    )
                await append_event(
                    CampaignExecutionEventKind.RUNTIME_CLEANED,
                    {"cleanup_receipt": cleanup.to_record()},
                )
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc

        if primary_error is not None:
            raise primary_error
        if finalization is None or cleanup is None:  # pragma: no cover - guarded above.
            raise AssertionError("successful execution lost lifecycle receipts")
        result = CampaignExecutionResult(
            preparation_sha256=preparation_sha256,
            runtime_start_receipt_sha256=start.receipt_sha256,
            stage_receipts=tuple(stage_receipts),
            reflection_receipts=tuple(
                reflection_receipts[source] for source in sorted(reflection_receipts)
            ),
            reflection_cancellation_receipts=tuple(reflection_cancellations),
            test_admission_receipts=tuple(test_admission_receipts),
            tail_drain_receipt=tail_drain,
            counters=counters,
            finalization_receipt=finalization,
            cleanup_receipt=cleanup,
            durable_event_sha256s=tuple(durable_event_sha256s),
        )
        CampaignExecutionResult.__post_init__(result)
        return result


__all__ = [
    "CampaignArchiveCutoffReceipt",
    "CampaignArchiveCutoffRequest",
    "CampaignCleanupReceipt",
    "CampaignCleanupRequest",
    "CampaignExecutionContractError",
    "CampaignExecutionCounters",
    "CampaignExecutionEvent",
    "CampaignExecutionEventKind",
    "CampaignExecutionJournalPort",
    "CampaignExecutionLifecyclePort",
    "CampaignExecutionResult",
    "CampaignExecutionStartReceipt",
    "CampaignExecutionStatus",
    "CampaignFinalizationReceipt",
    "CampaignFinalizationRequest",
    "CampaignJournalAck",
    "CampaignReflectionReceipt",
    "CampaignReflectionCancellationReceipt",
    "CampaignReflectionRequest",
    "CampaignReflectionRuntimePort",
    "CampaignReflectionStatus",
    "CampaignReflectionTestAdmissionReceipt",
    "CampaignReflectionTestAdmissionRequest",
    "CampaignSeedExecutionReceipt",
    "CampaignSelectorAuditReceipt",
    "CampaignStageReceipt",
    "CampaignStageRequest",
    "CampaignStageRuntimePort",
    "CampaignTailDrainReceipt",
    "EvolutionCampaignScheduler",
    "SelectorAuditExecutionMode",
    "campaign_step_sha256",
    "decode_selector_audit_text",
    "encode_selector_audit_text",
    "selector_audit_set_sha256",
]

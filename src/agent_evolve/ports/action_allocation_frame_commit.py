"""Immutable inputs and executions for audited frame-allocation commits.

This port is benchmark- and provider-neutral.  A treatment occurrence is bound
to the exact physical forecast request represented by an authenticated
allocation frame: the logical request for complete batches, or the block-call
request for partition blocks and subsets.  The terminal-provider-ledger digest
is deliberately modelled as a caller-supplied commitment.  Its presence does
not prove persistence; callers must materialize and fsync that ledger before
asking the application service to issue a phase commit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_allocation_frame import (
    AuditedFrameActionAllocationResult,
    FrameActionAllocationRequest,
    validate_frame_action_portfolio_decision,
)
from agent_evolve.ports.treatment_assignment import (
    ProspectiveTreatmentAssignmentReceipt,
    TreatmentOccurrence,
)


_COMMIT_INPUT_DOMAIN = b"agent-evolve:frame-allocation-commit-input:v1\x00"
_EXECUTION_DOMAIN = b"agent-evolve:frame-allocation-treatment-execution:v1\x00"
_TERMINAL_LEDGER_DURABILITY_SCOPE = (
    "external_materialization_and_fsync_required_before_commit"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def frame_source_call_and_request_identity(
    request: FrameActionAllocationRequest,
) -> tuple[str, str]:
    """Return the exact physical call ID and request digest behind a frame."""

    if type(request) is not FrameActionAllocationRequest:
        raise TypeError("request must be an exact frame allocation request")
    request.__post_init__()
    frame = request.frame
    if frame.block_request is None:
        return frame.request.call_id.value, frame.request.request_sha256
    return (
        frame.block_request.block_call_id.value,
        frame.block_request.block_request_sha256,
    )


def validate_treatment_occurrence_frame_request(
    occurrence: TreatmentOccurrence,
    request: FrameActionAllocationRequest,
) -> None:
    """Fail closed unless prospective call/request bindings name this frame."""

    if type(occurrence) is not TreatmentOccurrence:
        raise TypeError("occurrence must be an exact TreatmentOccurrence")
    occurrence.__post_init__()
    call_identity, request_identity_sha256 = frame_source_call_and_request_identity(
        request
    )
    if occurrence.call_identity != call_identity:
        raise ValueError("treatment occurrence names another frame source call")
    if occurrence.request_identity_sha256 != request_identity_sha256:
        raise ValueError("treatment occurrence names another frame source request")


@dataclass(frozen=True, slots=True, eq=False)
class FrameActionAllocationCommitInput:
    """Exact pre-allocation inputs bound by an ALLOCATE phase receipt."""

    upstream_input_sha256: str
    terminal_provider_ledger_commitment_sha256: str
    treatment_assignment: ProspectiveTreatmentAssignmentReceipt = field(
        repr=False,
        compare=False,
    )
    allocation_requests: tuple[FrameActionAllocationRequest, ...] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        require_sha256(self.upstream_input_sha256, "upstream_input_sha256")
        require_sha256(
            self.terminal_provider_ledger_commitment_sha256,
            "terminal_provider_ledger_commitment_sha256",
        )
        if type(self.treatment_assignment) is not (
            ProspectiveTreatmentAssignmentReceipt
        ):
            raise TypeError(
                "treatment_assignment must be an exact prospective receipt"
            )
        self.treatment_assignment.__post_init__()
        if type(self.allocation_requests) is not tuple or not (
            self.allocation_requests
        ) or any(
            type(value) is not FrameActionAllocationRequest
            for value in self.allocation_requests
        ):
            raise ValueError(
                "allocation_requests must be a non-empty exact request tuple"
            )
        if len(self.allocation_requests) != len(
            self.treatment_assignment.occurrence_input_order
        ):
            raise ValueError(
                "allocation requests must cover every treatment occurrence"
            )
        for occurrence, request in zip(
            self.treatment_assignment.occurrence_input_order,
            self.allocation_requests,
            strict=True,
        ):
            request.__post_init__()
            validate_treatment_occurrence_frame_request(occurrence, request)

    @property
    def terminal_provider_ledger_commitment_record(self) -> dict[str, object]:
        return {
            "commitment_sha256": (
                self.terminal_provider_ledger_commitment_sha256
            ),
            "durability_scope": _TERMINAL_LEDGER_DURABILITY_SCOPE,
            "digest_alone_proves_durability": False,
        }

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "upstream_input_sha256": self.upstream_input_sha256,
            "terminal_provider_ledger_commitment": (
                self.terminal_provider_ledger_commitment_record
            ),
            "treatment_assignment": self.treatment_assignment.to_record(),
            "treatment_occurrence_order": [
                value.to_record()
                for value in self.treatment_assignment.occurrence_input_order
            ],
            "ordered_allocation_inputs": [
                {
                    "treatment_occurrence_id": occurrence.occurrence_id.value,
                    "frame_receipt_sha256": request.frame.receipt_sha256,
                    "source_forecast_receipt_sha256": (
                        request.frame.source_forecast_receipt_sha256
                    ),
                    "allocation_request_sha256": request.request_sha256,
                    "eligible_options_sha256": request.eligible_options_sha256,
                }
                for occurrence, request in zip(
                    self.treatment_assignment.occurrence_input_order,
                    self.allocation_requests,
                    strict=True,
                )
            ],
        }

    @property
    def input_sha256(self) -> str:
        return _hash(_COMMIT_INPUT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {
            **record,
            "input_sha256": _hash(_COMMIT_INPUT_DOMAIN, record),
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FrameActionAllocationCommitInput
            and self.input_sha256 == other.input_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class FrameActionAllocationTreatmentExecution:
    """One treatment occurrence's exact audited allocation execution."""

    treatment_assignment: ProspectiveTreatmentAssignmentReceipt = field(
        repr=False,
        compare=False,
    )
    treatment_occurrence: TreatmentOccurrence
    request: FrameActionAllocationRequest = field(repr=False, compare=False)
    result: AuditedFrameActionAllocationResult = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.treatment_assignment) is not (
            ProspectiveTreatmentAssignmentReceipt
        ):
            raise TypeError(
                "treatment_assignment must be an exact prospective receipt"
            )
        self.treatment_assignment.__post_init__()
        if type(self.treatment_occurrence) is not TreatmentOccurrence:
            raise TypeError(
                "treatment_occurrence must be an exact TreatmentOccurrence"
            )
        self.treatment_occurrence.__post_init__()
        by_id = {
            value.occurrence_id: value
            for value in self.treatment_assignment.occurrence_input_order
        }
        expected = by_id.get(self.treatment_occurrence.occurrence_id)
        if expected is None or expected != self.treatment_occurrence:
            raise ValueError(
                "treatment occurrence differs from its prospective assignment"
            )
        if type(self.request) is not FrameActionAllocationRequest:
            raise TypeError("request must be an exact frame allocation request")
        self.request.__post_init__()
        validate_treatment_occurrence_frame_request(
            self.treatment_occurrence,
            self.request,
        )
        if type(self.result) is not AuditedFrameActionAllocationResult:
            raise TypeError("result must be an exact audited frame result")
        self.result.__post_init__()
        validate_frame_action_portfolio_decision(
            self.request,
            self.result.decision,
        )
        expected_candidate_counts = tuple(
            len(self.request.eligible_option_ids) - step
            for step in range(self.request.portfolio_size)
        )
        if tuple(
            value.candidate_count for value in self.result.audit.steps
        ) != expected_candidate_counts:
            raise ValueError(
                "allocation audit does not cover every greedy extension"
            )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "treatment_assignment_receipt_sha256": (
                self.treatment_assignment.receipt_sha256
            ),
            "treatment_occurrence": self.treatment_occurrence.to_record(),
            "frame": self.request.frame.to_record(),
            "allocation_request": self.request.to_record(),
            "decision": self.result.decision.to_record(),
            "audit": self.result.audit.to_record(),
            "audited_result_receipt_sha256": self.result.receipt_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_EXECUTION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {**record, "receipt_sha256": _hash(_EXECUTION_DOMAIN, record)}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FrameActionAllocationTreatmentExecution
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


__all__ = [
    "FrameActionAllocationCommitInput",
    "FrameActionAllocationTreatmentExecution",
    "frame_source_call_and_request_identity",
    "validate_treatment_occurrence_frame_request",
]

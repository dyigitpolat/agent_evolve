"""Durable fail-closed commits for audited frame allocation treatments."""

from __future__ import annotations

from agent_evolve.application.two_stage_action_evolution import (
    TwoStageActionPhase,
    TwoStageActionPhaseCommit,
    TwoStageActionPhaseReceipt,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.action_allocation_frame_commit import (
    FrameActionAllocationCommitInput,
    FrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.treatment_assignment import (
    ProspectiveTreatmentAssignmentReceipt,
)


_COMMIT_KIND = "audited_frame_action_allocation"


class FrameActionAllocationCommitRejected(RuntimeError):
    """No evaluator authority may open from these allocation executions."""


def _validated_execution_tuple(
    executions: tuple[FrameActionAllocationTreatmentExecution, ...],
    *,
    require_all_pass: bool,
) -> ProspectiveTreatmentAssignmentReceipt:
    if type(executions) is not tuple or not executions or any(
        type(value) is not FrameActionAllocationTreatmentExecution
        for value in executions
    ):
        raise ValueError("executions must be a non-empty exact execution tuple")
    for value in executions:
        value.__post_init__()
    assignment = executions[0].treatment_assignment
    if any(
        value.treatment_assignment.receipt_sha256 != assignment.receipt_sha256
        for value in executions
    ):
        raise ValueError("executions name different treatment assignments")
    observed_order = tuple(value.treatment_occurrence for value in executions)
    if observed_order != assignment.occurrence_input_order:
        raise ValueError(
            "executions must use exact treatment occurrence input order"
        )
    if require_all_pass and any(not value.result.audit.passes for value in executions):
        raise FrameActionAllocationCommitRejected(
            "every allocation-surface audit must pass before evaluator authority"
        )
    return assignment


def _frozen_payload(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:
        raise AssertionError("allocation commit payload must freeze as an object")
    return result


def build_frame_action_allocation_phase_commit(
    *,
    upstream_input_sha256: str,
    terminal_provider_ledger_commitment_sha256: str,
    executions: tuple[FrameActionAllocationTreatmentExecution, ...],
) -> TwoStageActionPhaseCommit:
    """Build an ALLOCATE commit only from complete passing audit records.

    The ledger digest is a commitment to a ledger the caller has already
    materialized and fsynced.  This pure builder cannot establish filesystem
    durability and its payload states that limitation explicitly.
    """

    require_sha256(upstream_input_sha256, "upstream_input_sha256")
    require_sha256(
        terminal_provider_ledger_commitment_sha256,
        "terminal_provider_ledger_commitment_sha256",
    )
    assignment = _validated_execution_tuple(executions, require_all_pass=True)
    commit_input = FrameActionAllocationCommitInput(
        upstream_input_sha256=upstream_input_sha256,
        terminal_provider_ledger_commitment_sha256=(
            terminal_provider_ledger_commitment_sha256
        ),
        treatment_assignment=assignment,
        allocation_requests=tuple(value.request for value in executions),
    )
    payload = _frozen_payload(
        {
            "schema_version": 1,
            "phase": TwoStageActionPhase.ALLOCATE.value,
            "commit_kind": _COMMIT_KIND,
            "allocation_input": commit_input.to_record(),
            "terminal_provider_ledger_commitment": (
                commit_input.terminal_provider_ledger_commitment_record
            ),
            "treatment_assignment_receipt_sha256": assignment.receipt_sha256,
            "treatment_occurrence_order": [
                value.to_record() for value in assignment.occurrence_input_order
            ],
            "treatment_executions": [value.to_record() for value in executions],
            "all_allocation_surface_audits_pass": True,
            "evaluator_authority_eligible": True,
        }
    )
    return TwoStageActionPhaseCommit(
        receipt=TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.ALLOCATE,
            input_sha256=commit_input.input_sha256,
            output_sha256=typed_json_sha256(payload),
        ),
        payload=payload,
    )


def validate_frame_action_allocation_phase_commit(
    executions: tuple[FrameActionAllocationTreatmentExecution, ...],
    phase_commit: TwoStageActionPhaseCommit,
) -> FrameActionAllocationCommitInput:
    """Validate payload equality and return its exact authenticated input."""

    assignment = _validated_execution_tuple(executions, require_all_pass=True)
    if type(phase_commit) is not TwoStageActionPhaseCommit:
        raise TypeError("phase_commit must be an exact TwoStageActionPhaseCommit")
    phase_commit.__post_init__()
    if phase_commit.receipt.phase is not TwoStageActionPhase.ALLOCATE:
        raise ValueError("only an ALLOCATE phase commit can authorize evaluation")
    payload = thaw_json(phase_commit.payload)
    if (
        type(payload) is not dict
        or payload.get("phase") != TwoStageActionPhase.ALLOCATE.value
        or payload.get("commit_kind") != _COMMIT_KIND
        or type(payload.get("allocation_input")) is not dict
    ):
        raise ValueError("frame-allocation phase payload is malformed")
    raw_input = payload["allocation_input"]
    assert type(raw_input) is dict
    upstream_input_sha256 = raw_input.get("upstream_input_sha256")
    raw_ledger = raw_input.get("terminal_provider_ledger_commitment")
    if type(upstream_input_sha256) is not str or type(raw_ledger) is not dict:
        raise ValueError("frame-allocation input commitment is malformed")
    ledger_sha256 = raw_ledger.get("commitment_sha256")
    if type(ledger_sha256) is not str:
        raise ValueError("terminal-provider-ledger commitment is malformed")
    require_sha256(upstream_input_sha256, "upstream_input_sha256")
    require_sha256(
        ledger_sha256,
        "terminal_provider_ledger_commitment_sha256",
    )
    expected = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=upstream_input_sha256,
        terminal_provider_ledger_commitment_sha256=ledger_sha256,
        executions=executions,
    )
    if (
        phase_commit.receipt.receipt_sha256
        != expected.receipt.receipt_sha256
        or phase_commit.payload != expected.payload
        or phase_commit.to_record() != expected.to_record()
    ):
        raise ValueError(
            "frame-allocation phase commit differs from exact execution payloads"
        )
    return FrameActionAllocationCommitInput(
        upstream_input_sha256=upstream_input_sha256,
        terminal_provider_ledger_commitment_sha256=ledger_sha256,
        treatment_assignment=assignment,
        allocation_requests=tuple(value.request for value in executions),
    )


__all__ = [
    "FrameActionAllocationCommitRejected",
    "build_frame_action_allocation_phase_commit",
    "validate_frame_action_allocation_phase_commit",
]

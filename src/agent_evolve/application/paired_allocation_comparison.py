"""Build and replay paired v2/v3 allocation authority before outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_evolve.application.action_allocation_frame_commit import (
    validate_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_allocation_frame_commit_v3 import (
    validate_operational_frame_action_allocation_phase_commit,
)
from agent_evolve.application.two_stage_action_evolution import (
    TwoStageActionPhaseCommit,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_allocation_frame import FrameActionAllocationRequest
from agent_evolve.ports.action_allocation_frame_commit import (
    FrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.action_allocation_frame_commit_v3 import (
    OperationalFrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.paired_allocation_comparison import (
    AllocationComparisonMethodKind,
    AllocationComparisonMethodReceipt,
    AllocationComparisonSelectedOption,
    PairedAllocationComparisonCommitment,
    frame_allocation_comparison_structure_sha256,
)


AllocationExecution = (
    FrameActionAllocationTreatmentExecution
    | OperationalFrameActionAllocationTreatmentExecution
)


@dataclass(frozen=True, slots=True)
class AllocationComparisonMethodWave:
    """One exact allocator method's already-durable treatment wave.

    ``schedule_binding_sha256`` is a caller-verifiable receipt for the common
    prospective schedule.  This builder binds it but does not claim that a
    bare digest proves materialization or chronology.
    """

    comparison_method_id: str
    schedule_binding_sha256: str
    executions: tuple[AllocationExecution, ...] = field(
        repr=False,
        compare=False,
    )
    phase_commit: TwoStageActionPhaseCommit = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        require_sha256(self.schedule_binding_sha256, "schedule_binding_sha256")
        if type(self.executions) is not tuple or not self.executions:
            raise ValueError("executions must be a non-empty exact tuple")
        first_type = type(self.executions[0])
        if first_type not in {
            FrameActionAllocationTreatmentExecution,
            OperationalFrameActionAllocationTreatmentExecution,
        } or any(type(value) is not first_type for value in self.executions):
            raise TypeError("a method wave must contain one exact execution type")
        if type(self.phase_commit) is not TwoStageActionPhaseCommit:
            raise TypeError("phase_commit must be exact")
        self.phase_commit.__post_init__()


def _base_request(execution: AllocationExecution) -> FrameActionAllocationRequest:
    if type(execution) is FrameActionAllocationTreatmentExecution:
        return execution.request
    if type(execution) is OperationalFrameActionAllocationTreatmentExecution:
        return execution.request.allocation
    raise TypeError("execution type is outside the paired comparison")


def _method_request_sha256(execution: AllocationExecution) -> str:
    if type(execution) is FrameActionAllocationTreatmentExecution:
        return execution.request.request_sha256
    if type(execution) is OperationalFrameActionAllocationTreatmentExecution:
        return execution.request.request_sha256
    raise TypeError("execution type is outside the paired comparison")


def _method_receipt(
    wave: AllocationComparisonMethodWave,
) -> AllocationComparisonMethodReceipt:
    wave.__post_init__()
    executions = wave.executions
    if type(executions[0]) is FrameActionAllocationTreatmentExecution:
        v2 = executions
        assert all(
            type(value) is FrameActionAllocationTreatmentExecution for value in v2
        )
        commit_input = validate_frame_action_allocation_phase_commit(
            v2,  # type: ignore[arg-type]
            wave.phase_commit,
        )
        kind = AllocationComparisonMethodKind.AUDITED_FRAME_V2
    else:
        v3 = executions
        assert all(
            type(value) is OperationalFrameActionAllocationTreatmentExecution
            for value in v3
        )
        commit_input = validate_operational_frame_action_allocation_phase_commit(
            v3,  # type: ignore[arg-type]
            wave.phase_commit,
        )
        kind = AllocationComparisonMethodKind.OPERATIONAL_FRAME_V3

    base_requests = tuple(_base_request(value) for value in executions)
    expected_utility = base_requests[0].utility.utility
    if any(value.utility.utility is not expected_utility for value in base_requests):
        raise ValueError("one method wave must share the same utility executable")
    structures = tuple(
        frame_allocation_comparison_structure_sha256(value)
        for value in base_requests
    )
    if len(set(structures)) != 1:
        raise ValueError(
            "one method wave mixes schedule/frame/eligibility/budget structure"
        )
    decisions = tuple(value.result.decision for value in executions)
    allocator_identities = {
        (
            value.allocator_policy_id,
            value.allocator_policy_version,
            value.allocator_definition_sha256,
            value.allocator_configuration_sha256,
        )
        for value in decisions
    }
    if len(allocator_identities) != 1:
        raise ValueError("one comparison method mixes allocator configurations")
    policy_id, policy_version, definition_sha256, configuration_sha256 = next(
        iter(allocator_identities)
    )
    selected = tuple(
        AllocationComparisonSelectedOption(
            treatment_occurrence_id=(
                execution.treatment_occurrence.occurrence_id.value
            ),
            treatment_id=execution.treatment_occurrence.treatment_id.value,
            rank=member.rank,
            option_id=member.option_id,
            option_identity_sha256=member.option_identity_sha256,
            child_configuration_sha256=member.child_configuration_sha256,
            family=member.family,
        )
        for execution in executions
        for member in execution.result.decision.members
    )
    assignment = executions[0].treatment_assignment
    return AllocationComparisonMethodReceipt(
        comparison_method_id=wave.comparison_method_id,
        method_kind=kind,
        schedule_binding_sha256=wave.schedule_binding_sha256,
        allocation_phase_commit_receipt_sha256=(
            wave.phase_commit.receipt.receipt_sha256
        ),
        allocation_input_sha256=commit_input.input_sha256,
        treatment_assignment_receipt_sha256=assignment.receipt_sha256,
        upstream_input_sha256=commit_input.upstream_input_sha256,
        terminal_provider_ledger_commitment_sha256=(
            commit_input.terminal_provider_ledger_commitment_sha256
        ),
        common_comparison_structure_sha256=structures[0],
        allocator_policy_id=policy_id,
        allocator_policy_version=policy_version,
        allocator_definition_sha256=definition_sha256,
        allocator_configuration_sha256=configuration_sha256,
        ordered_frame_receipt_sha256s=tuple(
            value.frame.receipt_sha256 for value in base_requests
        ),
        ordered_base_request_sha256s=tuple(
            value.request_sha256 for value in base_requests
        ),
        ordered_method_request_sha256s=tuple(
            _method_request_sha256(value) for value in executions
        ),
        ordered_decision_receipt_sha256s=tuple(
            value.receipt_sha256 for value in decisions
        ),
        selected_options=selected,
    )


def build_paired_allocation_comparison_commitment(
    methods: tuple[AllocationComparisonMethodWave, ...],
) -> PairedAllocationComparisonCommitment:
    """Validate both durable methods and bind their selected-option union."""

    if type(methods) is not tuple or len(methods) != 2 or any(
        type(value) is not AllocationComparisonMethodWave for value in methods
    ):
        raise ValueError("methods must contain exactly two exact method waves")
    for value in methods:
        value.__post_init__()
    all_requests = tuple(
        _base_request(execution)
        for method in methods
        for execution in method.executions
    )
    expected_utility = all_requests[0].utility.utility
    if any(value.utility.utility is not expected_utility for value in all_requests):
        raise ValueError("paired methods must share the same utility executable")
    receipts = tuple(
        sorted(
            (_method_receipt(value) for value in methods),
            key=lambda value: value.comparison_method_id,
        )
    )
    return PairedAllocationComparisonCommitment(methods=receipts)


def validate_paired_allocation_comparison_commitment(
    methods: tuple[AllocationComparisonMethodWave, ...],
    commitment: PairedAllocationComparisonCommitment,
) -> PairedAllocationComparisonCommitment:
    """Replay both phase commits and reject any post-commit object drift."""

    if type(commitment) is not PairedAllocationComparisonCommitment:
        raise TypeError("commitment must be exact")
    commitment.__post_init__()
    expected = build_paired_allocation_comparison_commitment(methods)
    if (
        commitment.commitment_sha256 != expected.commitment_sha256
        or commitment.to_record() != expected.to_record()
    ):
        raise ValueError("paired allocation commitment differs from exact methods")
    return expected


__all__ = [
    "AllocationComparisonMethodWave",
    "build_paired_allocation_comparison_commitment",
    "validate_paired_allocation_comparison_commitment",
]

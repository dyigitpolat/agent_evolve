"""Application adapter for wave-sealed causal-memory checkpoints.

The engine returns typed invocation outcomes and the optimizer publishes an
immutable generation receipt.  This module is the deliberately small adapter
that turns those application records into the provider-free causal-memory
records.  It never selects insights, changes scores, or reads an objective
archive; the already-frozen scalar reward is its only outcome input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from agent_evolve.application.agentic_evolution import (
    InvocationOutcome,
)
from agent_evolve.application.budgeted_optimizer import (
    GenerationReceipt,
    validate_generation_receipt_integrity,
)
from agent_evolve.policies.memory.staged_causal import (
    FrozenDiagnosticMemoryWave,
    MemoryAssignmentArm,
    MemoryAssignmentReceipt,
    MemoryCheckpointClosure,
    MemoryCheckpointClosureStatus,
    MemoryTrialTerminalStatus,
    WaveSealedCheckpointBuilder,
)


MemoryCheckpointTraceSink = Callable[[Mapping[str, object]], None]


def memory_assignment_receipt(
    outcome: InvocationOutcome,
) -> MemoryAssignmentReceipt:
    """Project one resolved diagnostic outcome into its exact ITT receipt."""

    if type(outcome) is not InvocationOutcome:
        raise TypeError("outcome must be an exact InvocationOutcome")
    assignment = outcome.prepared.plan.resolved_insight_assignment
    if assignment is None or assignment.arm is not MemoryAssignmentArm.DIAGNOSTIC:
        raise ValueError("outcome is not a resolved diagnostic assignment")
    if outcome.prepared.operator_invocation_id != assignment.credit_unit_id:
        raise ValueError("prepared invocation differs from assignment credit unit")

    if outcome.failure_stage is None:
        candidate = outcome.candidate
        if candidate is None:
            raise ValueError("successful diagnostic outcome has no candidate")
        return MemoryAssignmentReceipt(
            assignment_sha256=assignment.assignment_sha256,
            credit_unit_id=assignment.credit_unit_id,
            status=MemoryTrialTerminalStatus.SUCCEEDED,
            candidate_ids=(candidate.candidate_id,),
            observed_reward=outcome.reward,
        )
    if outcome.candidate is not None and outcome.failure_stage != "infrastructure":
        raise ValueError("failed diagnostic outcome unexpectedly carries a candidate")
    status = {
        "llm": MemoryTrialTerminalStatus.MODEL_FAILURE,
        "candidate": MemoryTrialTerminalStatus.CANDIDATE_FAILURE,
        "treatment_noncompliance": MemoryTrialTerminalStatus.CANDIDATE_FAILURE,
        "infrastructure": MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE,
        # Resolved prompt memory is forbidden on engine materializations.  Keep
        # this fail-closed mapping for corrupted/recovered external receipts.
        "materialization": MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE,
    }.get(outcome.failure_stage)
    if status is None:
        raise ValueError("diagnostic outcome has an unsupported failure stage")
    return MemoryAssignmentReceipt(
        assignment_sha256=assignment.assignment_sha256,
        credit_unit_id=assignment.credit_unit_id,
        status=status,
        candidate_ids=(
            () if outcome.candidate is None else (outcome.candidate.candidate_id,)
        ),
    )


@dataclass(frozen=True, slots=True)
class DiagnosticMemoryCheckpointService:
    """Close one frozen diagnostic wave from one optimizer receipt."""

    builder: WaveSealedCheckpointBuilder
    trace_sink: MemoryCheckpointTraceSink | None = None

    def __post_init__(self) -> None:
        if type(self.builder) is not WaveSealedCheckpointBuilder:
            raise TypeError("builder must be an exact WaveSealedCheckpointBuilder")
        if self.trace_sink is not None and not callable(self.trace_sink):
            raise TypeError("trace_sink must be callable")

    def publish_frozen_wave(self, wave: FrozenDiagnosticMemoryWave) -> None:
        """Publish the complete pre-call assignment cutoff to the trace sink."""

        if type(wave) is not FrozenDiagnosticMemoryWave:
            raise TypeError("wave must be an exact FrozenDiagnosticMemoryWave")
        if self.trace_sink is None:
            return
        self.trace_sink(
            {
                "event_type": "memory_wave_frozen",
                **wave.to_record(),
                "wave_sha256": wave.wave_sha256,
                "assignments": [
                    {
                        **assignment.to_record(),
                        "assignment_sha256": assignment.assignment_sha256,
                    }
                    for assignment in wave.assignments
                ],
            }
        )

    def close_generation(
        self,
        wave: FrozenDiagnosticMemoryWave,
        receipt: GenerationReceipt,
    ) -> MemoryCheckpointClosure:
        """Atomically close the diagnostic assignments in ``receipt``.

        Non-memory coverage or control slots may share the generation and are
        ignored.  Every resolved diagnostic assignment, however, must be an
        exact member of the frozen wave; omissions and extras fail closed.
        """

        if type(wave) is not FrozenDiagnosticMemoryWave:
            raise TypeError("wave must be an exact FrozenDiagnosticMemoryWave")
        if type(receipt) is not GenerationReceipt:
            raise TypeError("receipt must be an exact GenerationReceipt")
        validate_generation_receipt_integrity(receipt)
        if receipt.reward_definition_hash != wave.reward_definition_hash:
            raise ValueError("generation and diagnostic wave rewards differ")

        outcomes: dict[str, InvocationOutcome] = {}
        for result in receipt.slot_results:
            outcome = result.outcome
            assignment = outcome.prepared.plan.resolved_insight_assignment
            if (
                assignment is None
                or assignment.arm is not MemoryAssignmentArm.DIAGNOSTIC
            ):
                continue
            key = assignment.assignment_sha256
            if key in outcomes:
                raise ValueError("generation repeats a diagnostic assignment")
            outcomes[key] = outcome

        expected = {value.assignment_sha256 for value in wave.assignments}
        if set(outcomes) != expected:
            raise ValueError(
                "generation diagnostic assignments differ from the frozen wave"
            )
        terminal = tuple(
            memory_assignment_receipt(outcomes[key]) for key in sorted(outcomes)
        )
        closure = self.builder.close(wave, terminal)
        self._publish(closure, receipt)
        return closure

    def _publish(
        self,
        closure: MemoryCheckpointClosure,
        receipt: GenerationReceipt,
    ) -> None:
        if self.trace_sink is None:
            return
        common = {
            "wave_sha256": closure.wave_sha256,
            "generation": receipt.generation,
            "generation_receipt_hash": receipt.receipt_hash,
            "terminal_receipt_count": len(closure.receipts),
        }
        if closure.status is MemoryCheckpointClosureStatus.SEALED:
            snapshot = closure.snapshot
            assert snapshot is not None
            self.trace_sink(
                {
                    "event_type": "memory_wave_sealed",
                    **common,
                    "observation_count": len(closure.observations),
                }
            )
            self.trace_sink(
                {
                    "event_type": "memory_checkpoint_published",
                    **common,
                    "checkpoint_index": snapshot.checkpoint_index,
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "parent_snapshot_sha256": snapshot.parent_snapshot_sha256,
                }
            )
            return
        self.trace_sink(
            {
                "event_type": "memory_wave_invalidated",
                **common,
                "reason": "infrastructure_failure",
            }
        )


__all__ = [
    "DiagnosticMemoryCheckpointService",
    "MemoryCheckpointTraceSink",
    "memory_assignment_receipt",
]

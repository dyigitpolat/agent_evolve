"""Budgeted orchestration for explicit agentic evolution generations.

The :class:`AgenticEvolutionEngine` deliberately does not choose generations.
This module supplies the missing application boundary: a planner receives an
immutable history/archive cutoff, returns one ordered wave, and the optimizer
reserves hard budgets before either a model call or an evaluation can start.

The coordinator is domain- and provider-agnostic.  Model-authored and
engine-materialized slots execute concurrently, but candidates are appended to
history and published to the Pareto archive in the planner's frozen slot order.
Every wave uses one reward binding tied to the exact pre-wave archive cutoff.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Protocol

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    MaterializedInvocation,
    OperatorKind,
    ProposalAuthority,
    RewardPolicyBinding,
)
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackContext,
    GenerationFeedbackInterceptor,
    GenerationFeedbackReceipt,
    GenerationFeedbackReservation,
    GenerationFeedbackResult,
    seal_generation_feedback,
    validate_generation_feedback_receipt,
)
from agent_evolve.application.pareto_archive import (
    ParetoArchive,
    ParetoArchiveSnapshot,
    ParetoDecision,
    pareto_candidate_hash,
)
from agent_evolve.domain.patch import canonical_path_bytes, require_sha256
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.policies.variation.exact_parent_crossover import (
    exact_parent_import_exclusions_sha256,
)
from agent_evolve.ports.agentic_generator import AtomicMutationDraft, CandidateDraft


OptimizerTraceSink = Callable[[Mapping[str, object]], None]
_HASH_DOMAIN = b"agent-evolve:budgeted-agentic-optimizer:v1\x00"


class OptimizerContractError(ValueError):
    """A planner, budget, or optimizer input violated the frozen contract."""


class OptimizerBudgetExceeded(OptimizerContractError):
    """A complete wave could exceed a hard resource cap."""


class OptimizerPlanningError(RuntimeError):
    """The injected generation planner failed before a wave was admitted."""


class OptimizerExecutionError(RuntimeError):
    """An admitted wave failed outside the engine's typed outcome boundary."""


class OptimizerStopReason(str, Enum):
    GENERATION_LIMIT_REACHED = "generation_limit_reached"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _record_hash(kind: str, value: object) -> str:
    if type(kind) is not str or not kind:
        raise ValueError("hash record kind must be non-empty")
    return hashlib.sha256(
        _HASH_DOMAIN + kind.encode("ascii") + b"\x00" + _canonical_json(value)
    ).hexdigest()


def pareto_archive_snapshot_hash(snapshot: ParetoArchiveSnapshot) -> str:
    """Hash the complete immutable archive cutoff, including its decision ledger."""

    if type(snapshot) is not ParetoArchiveSnapshot:
        raise TypeError("snapshot must be an exact ParetoArchiveSnapshot")
    record = {
        "objectives": [
            {"name": objective.name, "goal": objective.goal}
            for objective in snapshot.objectives
        ],
        "front": [
            reference.to_trace_record() for reference in snapshot.front_references
        ],
        "decisions": [decision.to_trace_record() for decision in snapshot.decisions],
        "consideration_count": snapshot.consideration_count,
        "eligible_configuration_count": snapshot.eligible_configuration_count,
        "evidence_admission_policy": snapshot.evidence_admission_policy.value,
    }
    if not snapshot.objective_pareto_relation:
        record["outcome_relation_policy"] = {
            "policy_id": snapshot.outcome_relation_policy[0],
            "policy_version": snapshot.outcome_relation_policy[1],
            "definition_sha256": snapshot.outcome_relation_policy[2],
        }
    return _record_hash("pareto-archive-snapshot", record)


@dataclass(frozen=True, slots=True)
class OptimizerBudget:
    """Hard run-level caps; retries remain owned by the provider queue."""

    max_unique_evaluations: int
    max_logical_llm_calls: int
    max_generations: int

    def __post_init__(self) -> None:
        if (
            type(self.max_unique_evaluations) is not int
            or self.max_unique_evaluations <= 0
        ):
            raise ValueError("max_unique_evaluations must be a positive integer")
        if (
            type(self.max_logical_llm_calls) is not int
            or self.max_logical_llm_calls < 0
        ):
            raise ValueError("max_logical_llm_calls must be a non-negative integer")
        if type(self.max_generations) is not int or self.max_generations < 0:
            raise ValueError("max_generations must be a non-negative integer")

    def to_trace_record(self) -> dict[str, int]:
        return {
            "max_unique_evaluations": self.max_unique_evaluations,
            "max_logical_llm_calls": self.max_logical_llm_calls,
            "max_generations": self.max_generations,
        }

    @property
    def budget_hash(self) -> str:
        return _record_hash("budget", self.to_trace_record())


@dataclass(frozen=True, slots=True)
class FrozenWaveReward:
    """One reward policy frozen against one explicit pre-wave evidence cutoff."""

    binding: RewardPolicyBinding
    archive_snapshot_hash: str
    reward_snapshot_hash: str

    def __post_init__(self) -> None:
        if type(self.binding) is not RewardPolicyBinding:
            raise TypeError("binding must be an exact RewardPolicyBinding")
        RewardPolicyBinding.__post_init__(self.binding)
        require_sha256(self.archive_snapshot_hash, "archive_snapshot_hash")
        require_sha256(self.reward_snapshot_hash, "reward_snapshot_hash")


@dataclass(frozen=True, slots=True)
class SeedGateContext:
    """Run facts available to a domain-specific seed admission policy."""

    seed_index: int
    label: str
    requested_configuration_hash: str
    unique_evaluations_before: int
    unique_evaluations_after: int

    def __post_init__(self) -> None:
        if type(self.seed_index) is not int or self.seed_index < 0:
            raise ValueError("seed_index must be non-negative")
        if type(self.label) is not str or not self.label:
            raise ValueError("label must be non-empty")
        require_sha256(
            self.requested_configuration_hash,
            "requested_configuration_hash",
        )
        for name in ("unique_evaluations_before", "unique_evaluations_after"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.unique_evaluations_after < self.unique_evaluations_before:
            raise ValueError("seed evaluation counters cannot decrease")


@dataclass(frozen=True, slots=True)
class SeedGateDecision:
    """Versioned seed identity/objective/provenance admission evidence."""

    admitted: bool
    policy_id: str
    policy_version: int
    reason: str
    evidence: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool:
            raise TypeError("admitted must be bool")
        if (
            type(self.policy_id) is not str
            or not self.policy_id
            or self.policy_id != self.policy_id.strip()
        ):
            raise ValueError("policy_id must be canonical non-empty text")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if type(self.evidence) is not tuple:
            raise TypeError("evidence must be an exact tuple")
        for item in self.evidence:
            if (
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not str for value in item)
            ):
                raise TypeError("evidence must contain exact string pairs")
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise ValueError("evidence must be unique and canonically sorted")

    def to_trace_record(self) -> dict[str, object]:
        record = {
            "admitted": self.admitted,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "reason": self.reason,
            "evidence": [list(item) for item in self.evidence],
        }
        return {
            **record,
            "decision_hash": _record_hash("seed-gate-decision", record),
        }


class SeedAdmissionPolicy(Protocol):
    """Domain gate for exact seed identity, objectives, and external provenance."""

    def assess(
        self,
        candidate: EvolutionCandidate,
        context: SeedGateContext,
    ) -> SeedGateDecision: ...


class ValidSeedAdmissionPolicy:
    """Default gate: require a valid evaluated seed with a complete objective vector."""

    policy_id = "valid_evaluated_seed"
    policy_version = 1

    def assess(
        self,
        candidate: EvolutionCandidate,
        context: SeedGateContext,
    ) -> SeedGateDecision:
        del context
        complete = candidate.valid and len(candidate.objectives) > 0
        return SeedGateDecision(
            admitted=complete,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            reason=(
                "seed is valid with a non-empty objective vector"
                if complete
                else "seed is invalid or lacks an objective vector"
            ),
            evidence=(
                ("candidate_hash", pareto_candidate_hash(candidate)),
                ("valid", str(candidate.valid).lower()),
            ),
        )


@dataclass(frozen=True, slots=True)
class OptimizerSlot:
    """One ordered generation slot with explicit proposal authority."""

    slot_id: str
    role: str
    proposal_authority: ProposalAuthority
    plan: InvocationPlan
    materialized: MaterializedInvocation | None = None

    def __post_init__(self) -> None:
        for name in ("slot_id", "role"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be canonical non-empty text")
        if type(self.proposal_authority) is not ProposalAuthority:
            raise TypeError("proposal_authority must be a ProposalAuthority")
        if type(self.plan) is not InvocationPlan:
            raise TypeError("plan must be an exact InvocationPlan")
        InvocationPlan.__post_init__(self.plan)
        if self.proposal_authority is ProposalAuthority.ENGINE:
            if type(self.materialized) is not MaterializedInvocation:
                raise ValueError("engine authority requires a materialized invocation")
            MaterializedInvocation.__post_init__(self.materialized)
            if self.materialized.plan != self.plan:
                raise ValueError("slot plan and materialized invocation plan differ")
        elif self.materialized is not None:
            raise ValueError("only engine authority accepts a materialized invocation")
        if self.proposal_authority is ProposalAuthority.MODEL:
            if self.plan.operator_kind is OperatorKind.REPRODUCTION:
                raise ValueError("reproduction is not a model-authored proposal")
        elif self.proposal_authority is ProposalAuthority.REPRODUCTION:
            if self.plan.operator_kind is not OperatorKind.REPRODUCTION:
                raise ValueError("reproduction authority requires a reproduction plan")
        elif self.plan.operator_kind is OperatorKind.REPRODUCTION:
            raise ValueError("reproduction plans require reproduction authority")

    @classmethod
    def model(
        cls,
        *,
        slot_id: str,
        role: str,
        plan: InvocationPlan,
    ) -> "OptimizerSlot":
        return cls(slot_id, role, ProposalAuthority.MODEL, plan)

    @classmethod
    def engine(
        cls,
        *,
        slot_id: str,
        role: str,
        invocation: MaterializedInvocation,
    ) -> "OptimizerSlot":
        if type(invocation) is not MaterializedInvocation:
            raise TypeError("invocation must be an exact MaterializedInvocation")
        return cls(
            slot_id,
            role,
            ProposalAuthority.ENGINE,
            invocation.plan,
            invocation,
        )

    @classmethod
    def reproduction(
        cls,
        *,
        slot_id: str,
        role: str,
        plan: InvocationPlan,
    ) -> "OptimizerSlot":
        return cls(slot_id, role, ProposalAuthority.REPRODUCTION, plan)

    @property
    def logical_llm_call_reservation(self) -> int:
        return int(self.proposal_authority is ProposalAuthority.MODEL)

    @property
    def unique_evaluation_reservation(self) -> int:
        # Reproduction is guaranteed to reuse the exact parent configuration.
        # Every other slot may require one new physical evaluation; reserving the
        # upper bound is what makes concurrent admission safe.
        return int(self.proposal_authority is not ProposalAuthority.REPRODUCTION)


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """An immutable, ordered wave returned by a generation policy."""

    generation: int
    slots: tuple[OptimizerSlot, ...]
    reward: FrozenWaveReward
    planner_policy_id: str
    planner_policy_version: int
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive integer")
        if type(self.slots) is not tuple or any(
            type(slot) is not OptimizerSlot for slot in self.slots
        ):
            raise TypeError("slots must contain exact OptimizerSlot values")
        if len({slot.slot_id for slot in self.slots}) != len(self.slots):
            raise ValueError("generation slot IDs must be unique")
        if any(slot.plan.generation != self.generation for slot in self.slots):
            raise ValueError("every invocation must target the plan generation")
        if type(self.reward) is not FrozenWaveReward:
            raise TypeError("reward must be an exact FrozenWaveReward")
        FrozenWaveReward.__post_init__(self.reward)
        if (
            type(self.planner_policy_id) is not str
            or not self.planner_policy_id
            or self.planner_policy_id != self.planner_policy_id.strip()
        ):
            raise ValueError("planner_policy_id must be canonical non-empty text")
        if (
            type(self.planner_policy_version) is not int
            or self.planner_policy_version <= 0
        ):
            raise ValueError("planner_policy_version must be positive")
        if type(self.metadata) is not tuple:
            raise TypeError("metadata must be an exact tuple")
        for item in self.metadata:
            if (
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not str for value in item)
            ):
                raise TypeError("metadata must contain exact string pairs")
        if self.metadata != tuple(sorted(set(self.metadata))):
            raise ValueError("metadata must be unique and canonically sorted")

    @property
    def logical_llm_call_reservation(self) -> int:
        return sum(slot.logical_llm_call_reservation for slot in self.slots)

    @property
    def unique_evaluation_reservation(self) -> int:
        return sum(slot.unique_evaluation_reservation for slot in self.slots)


@dataclass(frozen=True, slots=True)
class OptimizerState:
    """Immutable planner view after an entire generation has been published."""

    generation: int
    candidates: tuple[EvolutionCandidate, ...]
    archive: ParetoArchiveSnapshot
    archive_snapshot_hash: str
    unique_evaluations: int
    logical_llm_calls: int
    generation_receipts: tuple["GenerationReceipt", ...] = ()
    feedback_receipts: tuple[GenerationFeedbackReceipt, ...] = ()

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be non-negative")
        if type(self.candidates) is not tuple or any(
            type(candidate) is not EvolutionCandidate for candidate in self.candidates
        ):
            raise TypeError("candidates must contain exact EvolutionCandidate values")
        if len({candidate.candidate_id for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("candidate history contains duplicate occurrence IDs")
        if type(self.archive) is not ParetoArchiveSnapshot:
            raise TypeError("archive must be an exact ParetoArchiveSnapshot")
        require_sha256(self.archive_snapshot_hash, "archive_snapshot_hash")
        if self.archive_snapshot_hash != pareto_archive_snapshot_hash(self.archive):
            raise ValueError("archive_snapshot_hash does not identify archive")
        for name in ("unique_evaluations", "logical_llm_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.generation_receipts) is not tuple or any(
            type(receipt) is not GenerationReceipt
            for receipt in self.generation_receipts
        ):
            raise TypeError(
                "generation_receipts must contain exact GenerationReceipt values"
            )
        for receipt in self.generation_receipts:
            validate_generation_receipt_integrity(receipt)
        receipt_generations = tuple(
            receipt.generation for receipt in self.generation_receipts
        )
        if receipt_generations != tuple(range(1, len(receipt_generations) + 1)):
            raise ValueError("generation_receipts must be contiguous and ordered")
        if self.generation_receipts and receipt_generations[-1] > self.generation:
            raise ValueError("generation_receipts cannot be newer than planner state")
        if type(self.feedback_receipts) is not tuple or any(
            type(receipt) is not GenerationFeedbackReceipt
            for receipt in self.feedback_receipts
        ):
            raise TypeError(
                "feedback_receipts must contain exact GenerationFeedbackReceipt values"
            )
        for receipt in self.feedback_receipts:
            validate_generation_feedback_receipt(receipt)
        feedback_generations = tuple(
            receipt.generation for receipt in self.feedback_receipts
        )
        if feedback_generations != tuple(range(1, len(feedback_generations) + 1)):
            raise ValueError("feedback_receipts must be contiguous and ordered")
        if self.feedback_receipts and feedback_generations[-1] > self.generation:
            raise ValueError("feedback_receipts cannot be newer than planner state")
        if len(self.feedback_receipts) > len(self.generation_receipts):
            raise ValueError("feedback receipts cannot outnumber generation receipts")
        for feedback_receipt in self.feedback_receipts:
            generation_receipt = self.generation_receipts[
                feedback_receipt.generation - 1
            ]
            if (
                feedback_receipt.generation_receipt_hash
                != generation_receipt.receipt_hash
            ):
                raise ValueError(
                    "feedback receipt does not identify its generation receipt"
                )
        if self.feedback_receipts and (
            self.feedback_receipts[-1].logical_llm_calls_after > self.logical_llm_calls
        ):
            raise ValueError("feedback receipt exceeds planner logical-call state")


class GenerationPlanner(Protocol):
    """Injected, deterministic generation policy."""

    def plan(
        self,
        state: OptimizerState,
        budget: OptimizerBudget,
    ) -> GenerationPlan: ...


@dataclass(frozen=True, slots=True)
class SeedReceipt:
    label: str
    candidate: EvolutionCandidate
    gate_decision: SeedGateDecision
    archive_decisions: tuple[ParetoDecision, ...]
    unique_evaluations_before: int
    unique_evaluations_after: int
    archive_snapshot_hash: str
    receipt_hash: str


@dataclass(frozen=True, slots=True)
class SlotResult:
    slot: OptimizerSlot
    outcome: InvocationOutcome
    archive_decisions: tuple[ParetoDecision, ...]


@dataclass(frozen=True, slots=True)
class GenerationReceipt:
    generation: int
    plan_hash: str
    pre_archive_snapshot_hash: str
    post_archive_snapshot_hash: str
    reward_definition_hash: str
    reward_snapshot_hash: str
    logical_llm_calls_before: int
    logical_llm_calls_after: int
    unique_evaluations_before: int
    unique_evaluations_after: int
    reserved_logical_llm_calls: int
    reserved_unique_evaluations: int
    slot_results: tuple[SlotResult, ...]
    receipt_hash: str


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    budget: OptimizerBudget
    final_state: OptimizerState
    seed_receipts: tuple[SeedReceipt, ...]
    generation_receipts: tuple[GenerationReceipt, ...]
    stop_reason: OptimizerStopReason
    result_hash: str
    feedback_receipts: tuple[GenerationFeedbackReceipt, ...] = ()


def _fraction_record(value: Fraction | None) -> object:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _candidate_identity(candidate: EvolutionCandidate | None) -> object:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id.value,
        "candidate_hash": pareto_candidate_hash(candidate),
        "configuration_hash": candidate.occurrence.configuration_hash,
    }


def _invocation_plan_record(plan: InvocationPlan) -> dict[str, object]:
    contract = plan.mutation_contract
    record: dict[str, object] = {
        "operator_kind": plan.operator_kind.value,
        "generation": plan.generation,
        "label": plan.label,
        "parents": [_candidate_identity(parent) for parent in plan.parents],
        "common_ancestor": _candidate_identity(plan.common_ancestor),
        "allowed_top_level": list(plan.allowed_top_level),
        "phase": plan.phase,
        "use_memory": plan.use_memory,
        "memory_subset_size": plan.memory_subset_size,
        "memory_exploration_probability": _fraction_record(
            plan.memory_exploration_probability
        ),
        "memory_score_phase": plan.memory_score_phase,
        "mutation_response_mode": plan.mutation_response_mode.value,
        "mutation_contract": (
            None
            if contract is None
            else {
                "editable_path_hashes": [
                    hashlib.sha256(canonical_path_bytes(path)).hexdigest()
                    for path in contract.editable_paths
                ],
                "max_changed_paths": contract.max_changed_paths,
                "max_operations": contract.max_operations,
                "allow_abstention": contract.allow_abstention,
            }
        ),
        "atomic_replacement_option_hashes": [
            typed_json_sha256(option) for option in plan.atomic_replacement_options
        ],
        "quarantine_test_insights": [
            {"insight_id": ref.insight_id.value, "version": ref.version}
            for ref in plan.quarantine_test_insights
        ],
        "resolved_insight_assignment": (
            None
            if plan.resolved_insight_assignment is None
            else {
                **plan.resolved_insight_assignment.to_record(),
                "assignment_sha256": (
                    plan.resolved_insight_assignment.assignment_sha256
                ),
            }
        ),
        "insight_treatment_requirement": (
            None
            if plan.insight_treatment_requirement is None
            else {
                **plan.insight_treatment_requirement.to_record(),
                "requirement_sha256": (
                    plan.insight_treatment_requirement.requirement_sha256
                ),
            }
        ),
        "compiled_hypothesis_treatment": (
            None
            if plan.compiled_hypothesis_treatment is None
            else {
                **plan.compiled_hypothesis_treatment.to_record(),
                "binding_sha256": (plan.compiled_hypothesis_treatment.binding_sha256),
            }
        ),
        "compiled_hypothesis_eligibility": [
            {
                **value.to_record(),
                "binding_sha256": value.binding_sha256,
            }
            for value in plan.compiled_hypothesis_eligibility
        ],
    }
    # The contract identity binds the exact parent, ordered palette, prompt
    # semantics, and every sealed child. Omitting the key for legacy modes keeps
    # their historical plan and generation receipt hashes byte-compatible.
    if plan.finite_variation_contract is not None:
        record["finite_variation_contract"] = (
            plan.finite_variation_contract.evidence_record()
        )
    if plan.finite_action_set_authority is not None:
        record["finite_action_set_authority"] = {
            **plan.finite_action_set_authority.to_record(),
            "authority_sha256": plan.finite_action_set_authority.authority_sha256,
        }
    # Keep the historical record shape byte-stable for the default/full
    # crossover representation.  Exact parent import is a distinct model
    # action space, so its mode, complete machine contract, and independently
    # recomputable contract identity must all be authenticated by the
    # generation-plan hash.
    if plan.exact_parent_crossover_contract is not None:
        exact_contract = plan.exact_parent_crossover_contract
        record["crossover_response_mode"] = plan.crossover_response_mode.value
        record["exact_parent_crossover_contract"] = exact_contract.to_record()
        record["exact_parent_crossover_contract_sha256"] = (
            exact_contract.contract_sha256
        )
        record["forbidden_exact_parent_import_sets"] = [
            list(value) for value in plan.forbidden_exact_parent_import_sets
        ]
        record["exact_parent_import_exclusions_sha256"] = (
            exact_parent_import_exclusions_sha256(
                exact_contract,
                plan.forbidden_exact_parent_import_sets,
            )
        )
    return record


def _materialized_draft_record(
    draft: CandidateDraft | AtomicMutationDraft,
) -> dict[str, object]:
    if type(draft) is CandidateDraft:
        return {
            "kind": "candidate_draft",
            "configuration_hash": typed_json_sha256(freeze_json(draft.configuration)),
            "design_rationale_sha256": hashlib.sha256(
                draft.design_rationale.encode("utf-8")
            ).hexdigest(),
            "intended_changes": list(draft.intended_changes),
            "source_attribution": [
                {"path": item.path, "source": item.source}
                for item in draft.source_attribution
            ],
            "claimed_insight_ids": list(draft.claimed_insight_ids),
            "claimed_preservation_obligation_ids": list(
                draft.claimed_preservation_obligation_ids
            ),
            "conflict_resolutions": [
                {
                    "relation_id": item.relation_id,
                    "choice": item.choice,
                    "explanation_sha256": hashlib.sha256(
                        item.explanation.encode("utf-8")
                    ).hexdigest(),
                }
                for item in draft.conflict_resolutions
            ],
        }
    if type(draft) is AtomicMutationDraft:
        return {
            "kind": "atomic_mutation_draft",
            "path_hash": hashlib.sha256(canonical_path_bytes(draft.path)).hexdigest(),
            "replacement_hash": typed_json_sha256(draft.replacement),
            "design_rationale_sha256": hashlib.sha256(
                draft.design_rationale.encode("utf-8")
            ).hexdigest(),
            "claimed_insight_ids": list(draft.claimed_insight_ids),
        }
    raise TypeError("unsupported materialized draft")


def _slot_record(slot: OptimizerSlot) -> dict[str, object]:
    materialized = slot.materialized
    return {
        "slot_id": slot.slot_id,
        "role": slot.role,
        "proposal_authority": slot.proposal_authority.value,
        "logical_llm_call_reservation": slot.logical_llm_call_reservation,
        "unique_evaluation_reservation": slot.unique_evaluation_reservation,
        "invocation": _invocation_plan_record(slot.plan),
        "materialized": (
            None
            if materialized is None
            else {
                "candidate_id": materialized.candidate_id.value,
                "policy_id": materialized.materialization_policy_id,
                "policy_version": materialized.materialization_policy_version,
                "receipt_hash": materialized.materialization_receipt_hash,
                "draft": _materialized_draft_record(materialized.draft),
            }
        ),
    }


def _generation_plan_record(
    plan: GenerationPlan,
    *,
    budget_hash: str,
) -> dict[str, object]:
    return {
        "generation": plan.generation,
        "planner_policy_id": plan.planner_policy_id,
        "planner_policy_version": plan.planner_policy_version,
        "metadata": [list(item) for item in plan.metadata],
        "budget_hash": budget_hash,
        "pre_archive_snapshot_hash": plan.reward.archive_snapshot_hash,
        "reward_definition_hash": plan.reward.binding.definition_hash,
        "reward_failure_score_hex": plan.reward.binding.failure_score.hex(),
        "reward_binding_sha256": plan.reward.binding.binding_sha256,
        "reward_snapshot_hash": plan.reward.reward_snapshot_hash,
        "logical_llm_call_reservation": plan.logical_llm_call_reservation,
        "unique_evaluation_reservation": plan.unique_evaluation_reservation,
        "slots": [_slot_record(slot) for slot in plan.slots],
    }


def _generation_receipt_record(
    *,
    generation: int,
    plan_hash: str,
    pre_archive_snapshot_hash: str,
    post_archive_snapshot_hash: str,
    reward_definition_hash: str,
    reward_snapshot_hash: str,
    logical_llm_calls_before: int,
    logical_llm_calls_after: int,
    unique_evaluations_before: int,
    unique_evaluations_after: int,
    reserved_logical_llm_calls: int,
    reserved_unique_evaluations: int,
    slot_results: tuple[SlotResult, ...],
) -> dict[str, object]:
    """Project the exact immutable fields authenticated by a receipt hash."""

    if type(generation) is not int or generation <= 0:
        raise ValueError("receipt generation must be a positive exact integer")
    for name in (
        "plan_hash",
        "pre_archive_snapshot_hash",
        "post_archive_snapshot_hash",
        "reward_definition_hash",
        "reward_snapshot_hash",
    ):
        require_sha256(locals()[name], name)
    for name in (
        "logical_llm_calls_before",
        "logical_llm_calls_after",
        "unique_evaluations_before",
        "unique_evaluations_after",
        "reserved_logical_llm_calls",
        "reserved_unique_evaluations",
    ):
        value = locals()[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative exact integer")
    if logical_llm_calls_after < logical_llm_calls_before:
        raise ValueError("logical LLM-call counters cannot decrease")
    if unique_evaluations_after < unique_evaluations_before:
        raise ValueError("unique-evaluation counters cannot decrease")
    if type(slot_results) is not tuple:
        raise TypeError("slot_results must be an exact tuple")
    if any(type(result) is not SlotResult for result in slot_results):
        raise TypeError("slot_results must contain exact SlotResult values")
    if any(type(result.slot) is not OptimizerSlot for result in slot_results):
        raise TypeError("receipt results must contain exact OptimizerSlot values")
    if any(type(result.outcome) is not InvocationOutcome for result in slot_results):
        raise TypeError("receipt results must contain exact InvocationOutcome values")
    if any(
        type(result.archive_decisions) is not tuple
        or any(
            type(decision) is not ParetoDecision
            for decision in result.archive_decisions
        )
        for result in slot_results
    ):
        raise TypeError(
            "receipt archive_decisions must contain exact ParetoDecision values"
        )
    if len({result.slot.slot_id for result in slot_results}) != len(slot_results):
        raise OptimizerContractError("generation receipt repeats a slot ID")
    for result in slot_results:
        prepared = result.outcome.prepared
        InvocationOutcome.__post_init__(result.outcome)
        if result.slot.plan != prepared.plan:
            raise OptimizerContractError(
                "generation receipt slot plan differs from its prepared outcome"
            )
        if result.slot.proposal_authority is not prepared.proposal_authority:
            raise OptimizerContractError(
                "generation receipt slot authority differs from its outcome"
            )
        if result.slot.plan.generation != generation:
            raise OptimizerContractError(
                "generation receipt slot targets a different generation"
            )
        if prepared.variation_case.reward_definition_hash != reward_definition_hash:
            raise OptimizerContractError(
                "generation receipt outcome has a different reward definition"
            )

    expected_logical_reservation = sum(
        result.slot.logical_llm_call_reservation for result in slot_results
    )
    expected_unique_reservation = sum(
        result.slot.unique_evaluation_reservation for result in slot_results
    )
    if reserved_logical_llm_calls != expected_logical_reservation:
        raise OptimizerContractError(
            "generation receipt logical reservation differs from its slots"
        )
    if reserved_unique_evaluations != expected_unique_reservation:
        raise OptimizerContractError(
            "generation receipt evaluation reservation differs from its slots"
        )
    if logical_llm_calls_after - logical_llm_calls_before != (
        reserved_logical_llm_calls
    ):
        raise OptimizerContractError(
            "generation receipt logical-call counters differ from its reservation"
        )
    if unique_evaluations_after - unique_evaluations_before > (
        reserved_unique_evaluations
    ):
        raise OptimizerContractError(
            "generation receipt physical evaluations exceed its reservation"
        )

    return {
        "generation": generation,
        "plan_hash": plan_hash,
        "pre_archive_snapshot_hash": pre_archive_snapshot_hash,
        "post_archive_snapshot_hash": post_archive_snapshot_hash,
        "reward_definition_hash": reward_definition_hash,
        "reward_snapshot_hash": reward_snapshot_hash,
        "logical_llm_calls_before": logical_llm_calls_before,
        "logical_llm_calls_after": logical_llm_calls_after,
        "unique_evaluations_before": unique_evaluations_before,
        "unique_evaluations_after": unique_evaluations_after,
        "reserved_logical_llm_calls": reserved_logical_llm_calls,
        "reserved_unique_evaluations": reserved_unique_evaluations,
        "slots": [
            {
                "slot": _slot_record(result.slot),
                "operator_invocation_id": (
                    result.outcome.prepared.operator_invocation_id.value
                ),
                "call_id": (
                    None
                    if result.outcome.prepared.call_id is None
                    else result.outcome.prepared.call_id.value
                ),
                "reserved_candidate_id": result.outcome.prepared.candidate_id.value,
                "proposal_sequence": result.outcome.prepared.proposal_sequence,
                "prepared_reward_definition_hash": (
                    result.outcome.prepared.variation_case.reward_definition_hash
                ),
                "prepared_selected_insights": [
                    {
                        "insight_id": reference.insight_id.value,
                        "version": reference.version,
                    }
                    for reference in (
                        result.outcome.prepared.variation_case.selected_insights
                    )
                ],
                "prepared_assignment_sha256": (
                    None
                    if result.outcome.prepared.plan.resolved_insight_assignment is None
                    else result.outcome.prepared.plan.resolved_insight_assignment.assignment_sha256
                ),
                "candidate": _candidate_identity(result.outcome.candidate),
                "reward": result.outcome.reward.hex(),
                "call_failure_type": result.outcome.call_failure_type,
                "failure_stage": result.outcome.failure_stage,
                "finite_action_decision": (
                    None
                    if result.outcome.finite_action_decision is None
                    else {
                        **result.outcome.finite_action_decision.to_record(),
                        "decision_sha256": (
                            result.outcome.finite_action_decision.decision_sha256
                        ),
                    }
                ),
                "treatment_admission": (
                    None
                    if result.outcome.treatment_admission_receipt is None
                    else {
                        **result.outcome.treatment_admission_receipt.to_record(),
                        "receipt_sha256": (
                            result.outcome.treatment_admission_receipt.receipt_sha256
                        ),
                    }
                ),
                "dominates_any_parent": result.outcome.dominates_any_parent,
                "better_than_any_parent": result.outcome.better_than_any_parent,
                "archive_decisions": [
                    decision.to_trace_record() for decision in result.archive_decisions
                ],
            }
            for result in slot_results
        ],
    }


def generation_receipt_hash(receipt: GenerationReceipt) -> str:
    """Recompute the canonical identity of a published generation receipt.

    This is intentionally public: replay and downstream evidence adapters must
    verify a receipt before trusting its nested outcomes.  Frozen dataclasses
    prevent in-place mutation but do not by themselves authenticate values
    reconstructed from durable storage or made with ``dataclasses.replace``.
    """

    if type(receipt) is not GenerationReceipt:
        raise TypeError("receipt must be an exact GenerationReceipt")
    return _record_hash(
        "generation-receipt",
        _generation_receipt_record(
            generation=receipt.generation,
            plan_hash=receipt.plan_hash,
            pre_archive_snapshot_hash=receipt.pre_archive_snapshot_hash,
            post_archive_snapshot_hash=receipt.post_archive_snapshot_hash,
            reward_definition_hash=receipt.reward_definition_hash,
            reward_snapshot_hash=receipt.reward_snapshot_hash,
            logical_llm_calls_before=receipt.logical_llm_calls_before,
            logical_llm_calls_after=receipt.logical_llm_calls_after,
            unique_evaluations_before=receipt.unique_evaluations_before,
            unique_evaluations_after=receipt.unique_evaluations_after,
            reserved_logical_llm_calls=receipt.reserved_logical_llm_calls,
            reserved_unique_evaluations=receipt.reserved_unique_evaluations,
            slot_results=receipt.slot_results,
        ),
    )


def validate_generation_receipt_integrity(receipt: GenerationReceipt) -> None:
    """Fail closed unless ``receipt_hash`` authenticates the exact projection."""

    if type(receipt) is not GenerationReceipt:
        raise TypeError("receipt must be an exact GenerationReceipt")
    require_sha256(receipt.receipt_hash, "receipt_hash")
    if generation_receipt_hash(receipt) != receipt.receipt_hash:
        raise OptimizerContractError(
            "generation receipt hash does not authenticate its contents"
        )


def seed_receipt_hash(receipt: SeedReceipt) -> str:
    """Recompute the canonical identity of one seed-admission receipt."""

    if type(receipt) is not SeedReceipt:
        raise TypeError("receipt must be an exact SeedReceipt")
    record = {
        "label": receipt.label,
        "candidate": _candidate_identity(receipt.candidate),
        "candidate_objectives": [
            [name, float(value).hex()] for name, value in receipt.candidate.objectives
        ],
        "candidate_configuration_artifact_hash": (
            receipt.candidate.occurrence.configuration_artifact_hash
        ),
        "requested_configuration_hash": (
            receipt.candidate.occurrence.configuration_hash
        ),
        "gate": receipt.gate_decision.to_trace_record(),
        "archive_decision_sequences": [
            decision.decision_sequence for decision in receipt.archive_decisions
        ],
        "unique_evaluations_before": receipt.unique_evaluations_before,
        "unique_evaluations_after": receipt.unique_evaluations_after,
        "archive_snapshot_hash": receipt.archive_snapshot_hash,
    }
    if receipt.candidate.objective_resolution_receipt is not None:
        record["objective_resolution_receipt_sha256"] = (
            receipt.candidate.objective_resolution_receipt.receipt_sha256
        )
    return _record_hash("seed-receipt", record)


def validate_seed_receipt_integrity(receipt: SeedReceipt) -> None:
    """Fail closed unless a seed receipt authenticates its full projection."""

    if type(receipt) is not SeedReceipt:
        raise TypeError("receipt must be an exact SeedReceipt")
    require_sha256(receipt.receipt_hash, "receipt_hash")
    if seed_receipt_hash(receipt) != receipt.receipt_hash:
        raise OptimizerContractError(
            "seed receipt hash does not authenticate its contents"
        )


def optimizer_result_hash(result: OptimizerResult) -> str:
    """Recompute the public terminal optimizer-result commitment."""

    if type(result) is not OptimizerResult:
        raise TypeError("result must be an exact OptimizerResult")
    return _record_hash(
        "optimizer-result",
        {
            "budget_hash": result.budget.budget_hash,
            "stop_reason": result.stop_reason.value,
            "generation": result.final_state.generation,
            "unique_evaluations": result.final_state.unique_evaluations,
            "logical_llm_calls": result.final_state.logical_llm_calls,
            "archive_snapshot_hash": result.final_state.archive_snapshot_hash,
            "seed_receipt_hashes": [item.receipt_hash for item in result.seed_receipts],
            "generation_receipt_hashes": [
                item.receipt_hash for item in result.generation_receipts
            ],
            "feedback_receipt_hashes": [
                item.receipt_hash for item in result.feedback_receipts
            ],
        },
    )


def validate_optimizer_result_integrity(result: OptimizerResult) -> None:
    """Authenticate a complete result and every receipt it directly owns."""

    if type(result) is not OptimizerResult:
        raise TypeError("result must be an exact OptimizerResult")
    OptimizerState.__post_init__(result.final_state)
    if result.generation_receipts != result.final_state.generation_receipts:
        raise OptimizerContractError(
            "result generation receipts differ from its final state"
        )
    if result.feedback_receipts != result.final_state.feedback_receipts:
        raise OptimizerContractError(
            "result feedback receipts differ from its final state"
        )
    for receipt in result.seed_receipts:
        validate_seed_receipt_integrity(receipt)
    for receipt in result.generation_receipts:
        validate_generation_receipt_integrity(receipt)
    for receipt in result.feedback_receipts:
        validate_generation_feedback_receipt(receipt)
    require_sha256(result.result_hash, "result_hash")
    if optimizer_result_hash(result) != result.result_hash:
        raise OptimizerContractError(
            "optimizer result hash does not authenticate its contents"
        )


class BudgetedAgenticOptimizer:
    """Compose a planner, evolution engine, and archive under hard run budgets."""

    def __init__(
        self,
        *,
        engine: AgenticEvolutionEngine,
        archive: ParetoArchive,
        planner: GenerationPlanner,
        budget: OptimizerBudget,
        seed_admission_policy: SeedAdmissionPolicy | None = None,
        feedback_interceptor: GenerationFeedbackInterceptor | None = None,
        trace_sink: OptimizerTraceSink | None = None,
    ) -> None:
        if not isinstance(engine, AgenticEvolutionEngine):
            raise TypeError("engine must be an AgenticEvolutionEngine")
        if type(archive) is not ParetoArchive:
            raise TypeError("archive must be an exact ParetoArchive")
        if tuple(archive.objectives) != tuple(engine.objectives):
            raise ValueError("engine and archive objective contracts differ")
        if (
            archive.outcome_relation_binding.identity
            != engine.outcome_relation_binding.identity
        ):
            raise ValueError("engine and archive outcome relation bindings differ")
        if not callable(getattr(planner, "plan", None)):
            raise TypeError("planner must implement plan(state, budget)")
        if type(budget) is not OptimizerBudget:
            raise TypeError("budget must be an exact OptimizerBudget")
        if feedback_interceptor is not None and not isinstance(
            feedback_interceptor,
            GenerationFeedbackInterceptor,
        ):
            raise TypeError(
                "feedback_interceptor must implement reserve and after_generation"
            )
        if trace_sink is not None and not callable(trace_sink):
            raise TypeError("trace_sink must be callable")
        gate = (
            ValidSeedAdmissionPolicy()
            if seed_admission_policy is None
            else seed_admission_policy
        )
        if not callable(getattr(gate, "assess", None)):
            raise TypeError("seed_admission_policy must implement assess")
        self.engine = engine
        self.archive = archive
        self.planner = planner
        self.budget = budget
        self.seed_admission_policy = gate
        self.feedback_interceptor = feedback_interceptor
        self._trace_sink = trace_sink
        self._started = False
        self._trace_sequence = 0
        self._trace_origin_ns = time.monotonic_ns()

    def _emit(self, event_type: str, **payload: object) -> None:
        if self._trace_sink is None:
            return
        self._trace_sequence += 1
        self._trace_sink(
            {
                "optimizer_trace_sequence": self._trace_sequence,
                "event_type": event_type,
                "optimizer_monotonic_offset_ns": (
                    time.monotonic_ns() - self._trace_origin_ns
                ),
                **payload,
            }
        )

    async def _evaluation_misses(self) -> int:
        snapshot = await self.engine.evaluation_cache_snapshot()
        misses = snapshot["misses"]
        in_flight = snapshot["in_flight"]
        if type(misses) is not int or type(in_flight) is not int:
            raise RuntimeError("engine evaluation cache returned invalid counters")
        if in_flight != 0:
            raise OptimizerContractError(
                "engine has evaluations in flight at an optimizer checkpoint"
            )
        return misses

    def _state(
        self,
        *,
        generation: int,
        candidates: tuple[EvolutionCandidate, ...],
        unique_evaluations: int,
        logical_llm_calls: int,
        generation_receipts: tuple[GenerationReceipt, ...],
        feedback_receipts: tuple[GenerationFeedbackReceipt, ...],
    ) -> OptimizerState:
        snapshot = self.archive.snapshot()
        return OptimizerState(
            generation=generation,
            candidates=candidates,
            archive=snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
            unique_evaluations=unique_evaluations,
            logical_llm_calls=logical_llm_calls,
            generation_receipts=generation_receipts,
            feedback_receipts=feedback_receipts,
        )

    def _validate_plan(self, plan: GenerationPlan, state: OptimizerState) -> None:
        if plan.generation != state.generation + 1:
            raise OptimizerContractError(
                "planner generation is not the next unpublished generation"
            )
        if plan.reward.archive_snapshot_hash != state.archive_snapshot_hash:
            raise OptimizerContractError(
                "wave reward is not bound to the exact pre-wave archive cutoff"
            )
        known = {candidate.candidate_id: candidate for candidate in state.candidates}
        materialized_ids = set()
        for slot in plan.slots:
            for parent in slot.plan.parents:
                if known.get(parent.candidate_id) != parent:
                    raise OptimizerContractError(
                        f"slot {slot.slot_id!r} refers to an unknown or altered parent"
                    )
            ancestor = slot.plan.common_ancestor
            if ancestor is not None and known.get(ancestor.candidate_id) != ancestor:
                raise OptimizerContractError(
                    f"slot {slot.slot_id!r} refers to an unknown or altered ancestor"
                )
            if slot.materialized is not None:
                candidate_id = slot.materialized.candidate_id
                if candidate_id in known or candidate_id in materialized_ids:
                    raise OptimizerContractError(
                        "materialized candidate occurrence IDs must be new and unique"
                    )
                materialized_ids.add(candidate_id)

    def _reserve_plan(
        self,
        plan: GenerationPlan,
        state: OptimizerState,
        feedback_reservation: GenerationFeedbackReservation | None,
    ) -> None:
        auxiliary_calls = (
            0
            if feedback_reservation is None
            else feedback_reservation.logical_llm_calls
        )
        if (
            state.logical_llm_calls
            + plan.logical_llm_call_reservation
            + auxiliary_calls
            > self.budget.max_logical_llm_calls
        ):
            raise OptimizerBudgetExceeded(
                "generation exceeds the logical LLM-call budget"
            )
        if (
            state.unique_evaluations + plan.unique_evaluation_reservation
            > self.budget.max_unique_evaluations
        ):
            raise OptimizerBudgetExceeded(
                "generation could exceed the unique-evaluation budget"
            )

    async def _execute_plan(
        self,
        plan: GenerationPlan,
    ) -> tuple[InvocationOutcome, ...]:
        direct_slots = tuple(
            slot
            for slot in plan.slots
            if slot.proposal_authority is not ProposalAuthority.ENGINE
        )
        engine_slots = tuple(
            slot
            for slot in plan.slots
            if slot.proposal_authority is ProposalAuthority.ENGINE
        )

        async def direct() -> tuple[InvocationOutcome, ...]:
            if not direct_slots:
                return ()
            return await self.engine.run_invocations(
                tuple(slot.plan for slot in direct_slots),
                reward_binding=plan.reward.binding,
            )

        async def materialized() -> tuple[InvocationOutcome, ...]:
            if not engine_slots:
                return ()
            return await self.engine.run_materialized_invocations(
                tuple(slot.materialized for slot in engine_slots),  # type: ignore[arg-type]
                reward_binding=plan.reward.binding,
            )

        direct_outcomes, engine_outcomes = await asyncio.gather(
            direct(), materialized()
        )
        by_slot: dict[str, InvocationOutcome] = {}
        for slot, outcome in zip(direct_slots, direct_outcomes, strict=True):
            by_slot[slot.slot_id] = outcome
        for slot, outcome in zip(engine_slots, engine_outcomes, strict=True):
            by_slot[slot.slot_id] = outcome
        if set(by_slot) != {slot.slot_id for slot in plan.slots}:
            raise RuntimeError("engine returned an incomplete generation")
        ordered = tuple(by_slot[slot.slot_id] for slot in plan.slots)
        for slot, outcome in zip(plan.slots, ordered, strict=True):
            prepared = outcome.prepared
            if prepared.plan != slot.plan:
                raise RuntimeError("engine outcome differs from its admitted slot plan")
            if prepared.proposal_authority is not slot.proposal_authority:
                raise RuntimeError("engine outcome has the wrong proposal authority")
            if (
                prepared.variation_case.reward_definition_hash
                != plan.reward.binding.definition_hash
            ):
                raise RuntimeError("engine outcome has the wrong reward identity")
            if not math.isfinite(outcome.reward):
                raise RuntimeError("engine outcome reward must be finite")
        return ordered

    async def run(
        self,
        seed_configs: Sequence[dict[str, object]],
    ) -> OptimizerResult:
        """Evaluate seeds and execute exactly ``max_generations`` planner waves."""

        if self._started:
            raise OptimizerContractError("an optimizer instance is single-use")
        self._started = True
        if self.archive.snapshot().consideration_count != 0:
            raise OptimizerContractError("optimizer requires a fresh empty archive")
        seeds = tuple(seed_configs)
        if not seeds:
            raise OptimizerContractError("at least one seed configuration is required")
        if any(type(config) is not dict for config in seeds):
            raise TypeError("seed configurations must be exact dictionaries")
        seed_hashes = tuple(typed_json_sha256(freeze_json(config)) for config in seeds)
        if len(set(seed_hashes)) != len(seed_hashes):
            raise OptimizerContractError("seed configurations must be unique")
        if len(seeds) > self.budget.max_unique_evaluations:
            raise OptimizerBudgetExceeded(
                "seed gate could exceed the unique-evaluation budget"
            )

        initial_misses = await self._evaluation_misses()
        self._emit(
            "optimizer_started",
            budget=self.budget.to_trace_record(),
            budget_hash=self.budget.budget_hash,
            initial_engine_evaluation_misses=initial_misses,
            seed_configuration_hashes=list(seed_hashes),
        )
        candidates: tuple[EvolutionCandidate, ...] = ()
        seed_receipts: list[SeedReceipt] = []
        prior_unique = 0
        for index, config in enumerate(seeds):
            label = f"seed_{index}"
            candidate = await self.engine.register_seed(config, label=label)
            current_unique = (await self._evaluation_misses()) - initial_misses
            if not 0 <= current_unique <= self.budget.max_unique_evaluations:
                raise OptimizerExecutionError(
                    "seed evaluation counters exceeded budget"
                )
            gate_context = SeedGateContext(
                seed_index=index,
                label=label,
                requested_configuration_hash=seed_hashes[index],
                unique_evaluations_before=prior_unique,
                unique_evaluations_after=current_unique,
            )
            try:
                gate_decision = self.seed_admission_policy.assess(
                    candidate,
                    gate_context,
                )
            except Exception as exc:
                self._emit(
                    "optimizer_seed_gate_failed",
                    label=label,
                    candidate=_candidate_identity(candidate),
                    requested_configuration_hash=seed_hashes[index],
                    unique_evaluations_before=prior_unique,
                    unique_evaluations_after=current_unique,
                    failure_type=type(exc).__name__,
                )
                raise OptimizerExecutionError(
                    f"seed admission policy failed for {label}"
                ) from exc
            if type(gate_decision) is not SeedGateDecision:
                raise OptimizerContractError(
                    "seed admission policy must return an exact SeedGateDecision"
                )
            SeedGateDecision.__post_init__(gate_decision)
            if gate_decision.admitted and not candidate.valid:
                raise OptimizerContractError(
                    "a seed admission policy cannot override engine invalidity"
                )
            decisions = (
                self.archive.consider(candidate) if gate_decision.admitted else ()
            )
            snapshot_hash = pareto_archive_snapshot_hash(self.archive.snapshot())
            seed_record = {
                "label": label,
                "candidate": _candidate_identity(candidate),
                "candidate_objectives": [
                    [name, float(value).hex()] for name, value in candidate.objectives
                ],
                "candidate_configuration_artifact_hash": (
                    candidate.occurrence.configuration_artifact_hash
                ),
                "requested_configuration_hash": seed_hashes[index],
                "gate": gate_decision.to_trace_record(),
                "archive_decision_sequences": [
                    decision.decision_sequence for decision in decisions
                ],
                "unique_evaluations_before": prior_unique,
                "unique_evaluations_after": current_unique,
                "archive_snapshot_hash": snapshot_hash,
            }
            if candidate.objective_resolution_receipt is not None:
                seed_record["objective_resolution_receipt_sha256"] = (
                    candidate.objective_resolution_receipt.receipt_sha256
                )
            receipt = SeedReceipt(
                label=label,
                candidate=candidate,
                gate_decision=gate_decision,
                archive_decisions=decisions,
                unique_evaluations_before=prior_unique,
                unique_evaluations_after=current_unique,
                archive_snapshot_hash=snapshot_hash,
                receipt_hash=_record_hash("seed-receipt", seed_record),
            )
            seed_receipts.append(receipt)
            candidates = (*candidates, candidate)
            prior_unique = current_unique
            self._emit(
                "optimizer_seed_completed",
                **seed_record,
                receipt_hash=receipt.receipt_hash,
                valid=candidate.valid,
            )
            if not gate_decision.admitted:
                raise OptimizerExecutionError(
                    "seed gate rejected candidate "
                    f"{candidate.candidate_id.value}: {gate_decision.reason}"
                )

        state = self._state(
            generation=0,
            candidates=candidates,
            unique_evaluations=prior_unique,
            logical_llm_calls=0,
            generation_receipts=(),
            feedback_receipts=(),
        )
        generation_receipts: list[GenerationReceipt] = []
        feedback_receipts: list[GenerationFeedbackReceipt] = []

        while state.generation < self.budget.max_generations:
            try:
                plan = self.planner.plan(state, self.budget)
            except Exception as exc:
                self._emit(
                    "optimizer_planning_failed",
                    generation=state.generation + 1,
                    failure_type=type(exc).__name__,
                )
                raise OptimizerPlanningError(
                    f"planner failed for generation {state.generation + 1}"
                ) from exc
            if type(plan) is not GenerationPlan:
                raise OptimizerContractError(
                    "planner must return an exact GenerationPlan"
                )
            GenerationPlan.__post_init__(plan)
            self._validate_plan(plan, state)
            feedback_reservation: GenerationFeedbackReservation | None = None
            if self.feedback_interceptor is not None:
                try:
                    feedback_reservation = self.feedback_interceptor.reserve(
                        state=state,
                        plan=plan,
                    )
                except Exception as exc:
                    self._emit(
                        "optimizer_generation_feedback_reservation_failed",
                        generation=plan.generation,
                        failure_type=type(exc).__name__,
                    )
                    raise OptimizerPlanningError(
                        "generation feedback reservation failed for "
                        f"generation {plan.generation}"
                    ) from exc
                if type(feedback_reservation) is not GenerationFeedbackReservation:
                    raise OptimizerContractError(
                        "feedback interceptor must return an exact reservation"
                    )
                GenerationFeedbackReservation.__post_init__(feedback_reservation)
            plan_record = _generation_plan_record(
                plan,
                budget_hash=self.budget.budget_hash,
            )
            plan_hash = _record_hash("generation-plan", plan_record)
            try:
                self._reserve_plan(plan, state, feedback_reservation)
            except OptimizerBudgetExceeded:
                self._emit(
                    "optimizer_generation_rejected",
                    plan_hash=plan_hash,
                    feedback_reservation=(
                        None
                        if feedback_reservation is None
                        else {
                            **feedback_reservation.to_record(),
                            "reservation_hash": (feedback_reservation.reservation_hash),
                        }
                    ),
                    **plan_record,
                )
                raise
            if feedback_reservation is not None:
                self._emit(
                    "optimizer_generation_feedback_reserved",
                    generation=plan.generation,
                    plan_hash=plan_hash,
                    **feedback_reservation.to_record(),
                    reservation_hash=feedback_reservation.reservation_hash,
                )
            self._emit(
                "optimizer_generation_planned",
                plan_hash=plan_hash,
                feedback_reservation_hash=(
                    None
                    if feedback_reservation is None
                    else feedback_reservation.reservation_hash
                ),
                reserved_feedback_logical_llm_calls=(
                    0
                    if feedback_reservation is None
                    else feedback_reservation.logical_llm_calls
                ),
                **plan_record,
            )
            if (
                _record_hash(
                    "generation-plan",
                    _generation_plan_record(
                        plan,
                        budget_hash=self.budget.budget_hash,
                    ),
                )
                != plan_hash
            ):
                raise OptimizerContractError(
                    "generation plan changed after its admission receipt"
                )

            before_unique = state.unique_evaluations
            before_calls = state.logical_llm_calls
            try:
                outcomes = await self._execute_plan(plan)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit(
                    "optimizer_generation_execution_failed",
                    generation=plan.generation,
                    plan_hash=plan_hash,
                    failure_type=type(exc).__name__,
                )
                raise OptimizerExecutionError(
                    f"generation {plan.generation} execution failed"
                ) from exc
            if (
                _record_hash(
                    "generation-plan",
                    _generation_plan_record(
                        plan,
                        budget_hash=self.budget.budget_hash,
                    ),
                )
                != plan_hash
            ):
                raise OptimizerExecutionError(
                    "generation plan changed while its slots were executing"
                )

            observed_calls = sum(
                outcome.prepared.call_id is not None for outcome in outcomes
            )
            if observed_calls != plan.logical_llm_call_reservation:
                raise OptimizerExecutionError(
                    "engine logical-call identities differ from reservations"
                )
            after_calls = before_calls + observed_calls
            after_unique = (await self._evaluation_misses()) - initial_misses
            if (
                after_calls > self.budget.max_logical_llm_calls
                or after_unique > self.budget.max_unique_evaluations
                or after_unique < before_unique
            ):
                raise OptimizerExecutionError("engine counters violated hard budgets")

            slot_results: list[SlotResult] = []
            next_candidates = list(state.candidates)
            for slot, outcome in zip(plan.slots, outcomes, strict=True):
                candidate = outcome.candidate
                decisions: tuple[ParetoDecision, ...] = ()
                if candidate is not None:
                    if any(
                        existing.candidate_id == candidate.candidate_id
                        for existing in next_candidates
                    ):
                        raise OptimizerExecutionError(
                            "engine reused a candidate occurrence ID"
                        )
                    next_candidates.append(candidate)
                    decisions = self.archive.consider(candidate)
                slot_results.append(SlotResult(slot, outcome, decisions))

            post_snapshot = self.archive.snapshot()
            post_archive_hash = pareto_archive_snapshot_hash(post_snapshot)
            receipt_record = _generation_receipt_record(
                generation=plan.generation,
                plan_hash=plan_hash,
                pre_archive_snapshot_hash=state.archive_snapshot_hash,
                post_archive_snapshot_hash=post_archive_hash,
                reward_definition_hash=plan.reward.binding.definition_hash,
                reward_snapshot_hash=plan.reward.reward_snapshot_hash,
                logical_llm_calls_before=before_calls,
                logical_llm_calls_after=after_calls,
                unique_evaluations_before=before_unique,
                unique_evaluations_after=after_unique,
                reserved_logical_llm_calls=plan.logical_llm_call_reservation,
                reserved_unique_evaluations=plan.unique_evaluation_reservation,
                slot_results=tuple(slot_results),
            )
            receipt_hash = _record_hash("generation-receipt", receipt_record)
            receipt = GenerationReceipt(
                generation=plan.generation,
                plan_hash=plan_hash,
                pre_archive_snapshot_hash=state.archive_snapshot_hash,
                post_archive_snapshot_hash=post_archive_hash,
                reward_definition_hash=plan.reward.binding.definition_hash,
                reward_snapshot_hash=plan.reward.reward_snapshot_hash,
                logical_llm_calls_before=before_calls,
                logical_llm_calls_after=after_calls,
                unique_evaluations_before=before_unique,
                unique_evaluations_after=after_unique,
                reserved_logical_llm_calls=plan.logical_llm_call_reservation,
                reserved_unique_evaluations=plan.unique_evaluation_reservation,
                slot_results=tuple(slot_results),
                receipt_hash=receipt_hash,
            )
            generation_receipts.append(receipt)
            self._emit(
                "optimizer_generation_completed",
                **receipt_record,
                receipt_hash=receipt_hash,
            )
            state = self._state(
                generation=plan.generation,
                candidates=tuple(next_candidates),
                unique_evaluations=after_unique,
                logical_llm_calls=after_calls,
                generation_receipts=tuple(generation_receipts),
                feedback_receipts=tuple(feedback_receipts),
            )
            if self.feedback_interceptor is not None:
                assert feedback_reservation is not None
                context = GenerationFeedbackContext(
                    state=state,
                    plan=plan,
                    generation_receipt=receipt,
                    reservation=feedback_reservation,
                )
                self._emit(
                    "optimizer_generation_feedback_started",
                    generation=plan.generation,
                    generation_receipt_hash=receipt.receipt_hash,
                    reservation_hash=feedback_reservation.reservation_hash,
                )
                try:
                    feedback_result = await self.feedback_interceptor.after_generation(
                        context
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._emit(
                        "optimizer_generation_feedback_failed",
                        generation=plan.generation,
                        generation_receipt_hash=receipt.receipt_hash,
                        reservation_hash=feedback_reservation.reservation_hash,
                        failure_type=type(exc).__name__,
                    )
                    raise OptimizerExecutionError(
                        f"generation feedback failed for generation {plan.generation}"
                    ) from exc
                if type(feedback_result) is not GenerationFeedbackResult:
                    raise OptimizerContractError(
                        "feedback interceptor must return an exact result"
                    )
                try:
                    feedback_receipt = seal_generation_feedback(
                        context=context,
                        result=feedback_result,
                    )
                except (TypeError, ValueError) as exc:
                    raise OptimizerContractError(
                        "feedback result differs from its admitted reservation"
                    ) from exc
                if (
                    feedback_receipt.logical_llm_calls_after
                    > self.budget.max_logical_llm_calls
                ):
                    raise OptimizerExecutionError(
                        "feedback logical-call counters violated the hard budget"
                    )
                feedback_receipts.append(feedback_receipt)
                self._emit(
                    "optimizer_generation_feedback_completed",
                    generation=plan.generation,
                    policy_id=feedback_receipt.policy_id,
                    policy_version=feedback_receipt.policy_version,
                    generation_receipt_hash=receipt.receipt_hash,
                    reservation_hash=feedback_receipt.reservation_hash,
                    reserved_logical_llm_calls=(
                        feedback_receipt.reserved_logical_llm_calls
                    ),
                    used_logical_llm_calls=feedback_receipt.used_logical_llm_calls,
                    logical_llm_calls_before=(
                        feedback_receipt.logical_llm_calls_before
                    ),
                    logical_llm_calls_after=(feedback_receipt.logical_llm_calls_after),
                    result_metadata=[
                        list(item) for item in feedback_receipt.result_metadata
                    ],
                    feedback_receipt_hash=feedback_receipt.receipt_hash,
                )
                state = self._state(
                    generation=plan.generation,
                    candidates=tuple(next_candidates),
                    unique_evaluations=after_unique,
                    logical_llm_calls=(feedback_receipt.logical_llm_calls_after),
                    generation_receipts=tuple(generation_receipts),
                    feedback_receipts=tuple(feedback_receipts),
                )

        stop_reason = OptimizerStopReason.GENERATION_LIMIT_REACHED
        result_record = {
            "budget_hash": self.budget.budget_hash,
            "stop_reason": stop_reason.value,
            "generation": state.generation,
            "unique_evaluations": state.unique_evaluations,
            "logical_llm_calls": state.logical_llm_calls,
            "archive_snapshot_hash": state.archive_snapshot_hash,
            "seed_receipt_hashes": [item.receipt_hash for item in seed_receipts],
            "generation_receipt_hashes": [
                item.receipt_hash for item in generation_receipts
            ],
            "feedback_receipt_hashes": [
                item.receipt_hash for item in feedback_receipts
            ],
        }
        result_hash = _record_hash("optimizer-result", result_record)
        self._emit("optimizer_completed", **result_record, result_hash=result_hash)
        return OptimizerResult(
            budget=self.budget,
            final_state=state,
            seed_receipts=tuple(seed_receipts),
            generation_receipts=tuple(generation_receipts),
            feedback_receipts=tuple(feedback_receipts),
            stop_reason=stop_reason,
            result_hash=result_hash,
        )


__all__ = [
    "BudgetedAgenticOptimizer",
    "FrozenWaveReward",
    "GenerationPlan",
    "GenerationPlanner",
    "GenerationReceipt",
    "OptimizerBudget",
    "OptimizerBudgetExceeded",
    "OptimizerContractError",
    "OptimizerExecutionError",
    "OptimizerPlanningError",
    "OptimizerResult",
    "OptimizerSlot",
    "OptimizerState",
    "OptimizerStopReason",
    "SeedReceipt",
    "SeedAdmissionPolicy",
    "SeedGateContext",
    "SeedGateDecision",
    "SlotResult",
    "ValidSeedAdmissionPolicy",
    "generation_receipt_hash",
    "optimizer_result_hash",
    "pareto_archive_snapshot_hash",
    "seed_receipt_hash",
    "validate_generation_receipt_integrity",
    "validate_optimizer_result_integrity",
    "validate_seed_receipt_integrity",
]

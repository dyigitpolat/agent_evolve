"""Generic propose-once, evaluate-pilot, choose-branch residual evolution.

Every proposal, score, semantic cell, pilot action, and complete continuation
branch is sealed before the first current candidate is evaluated.  A typed
pilot projector then exposes only real marginal archive contribution to a
small gate.  The chosen pre-existing branch is evaluated without re-querying a
model or regenerating the universe.

This service is workload-, objective-, model-, provider-, and prompt-blind.
Those concerns remain behind proposal experts, the pilot-outcome projector,
and the already established archive/evaluator ports.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionBrokerRequest,
    MaterializedSlateFeasibilityPort,
    MaterializedSlateValuePort,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.residual_portfolio_evolution import (
    DISJOINT_ACTION_EVALUATION_WAVES_V1,
    MaterializedActionEvaluation,
    MaterializedActionEvaluationBatch,
    MaterializedActionProposalBatch,
    MaterializedActionProposalExpertPort,
    ResidualPortfolioDecisionRequest,
    ResidualPortfolioEvolutionResult,
    evaluate_materialized_action_subset,
    propose_materialized_action_batches,
)
from agent_evolve.application.sequential_lineage_allocation import (
    AnyPositiveSequentialLineageGate,
    SequentialAllocationGateDecisionPort,
    SequentialAllocationGatePort,
    SequentialAllocationPlanPort,
    SequentialLineageAllocationPlannerPort,
    SequentialPilotOutcomeBatch,
    SequentialPilotOutcomeProjectorPort,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_ID = (
    "sequential_residual_portfolio_evolution"
)
SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_VERSION = 4
SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:sequential-residual-portfolio-evolution:v4;"
    b"proposal-universe=sealed-once-before-current-outcomes;"
    b"branches=one-or-more-complete-slates-frozen-before-pilot;"
    b"pilot=planner-authenticated-action-subset;"
    b"gate=injected-precommitted-branch-decision-port;"
    b"continuation=no-regeneration-no-rescoring-no-provider-call;"
    b"execution=pilot-then-chosen-remaining-actions;"
    b"exactly-once-boundary=disjoint-actions-not-proposal-batches;"
    b"failed-wave-reservation=fail-closed;"
    b"archive-publication=combined-final-slate-only;"
    b"durability=optional-or-required-plan-pilot-evaluated-adjudicated-final-commit-port;"
    b"workload-objective-model-provider-prompt-branches=false"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_PHASE_RECEIPT_DOMAIN = (
    b"agent-evolve:sequential-residual-phase-receipt:v1\x00"
)
_PHASE_ACK_DOMAIN = b"agent-evolve:sequential-residual-phase-ack:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:sequential-residual-result:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _planner_identity(
    value: SequentialLineageAllocationPlannerPort,
) -> tuple[str, int, str]:
    if not isinstance(value, SequentialLineageAllocationPlannerPort):
        raise TypeError("planner must implement its application port")
    identity = (
        getattr(value, "planner_id", None),
        getattr(value, "planner_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="planner_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("planner_version must be positive")
    require_sha256(identity[2], "planner definition_sha256")
    return identity  # type: ignore[return-value]


def _projector_identity(
    value: SequentialPilotOutcomeProjectorPort,
) -> tuple[str, int, str]:
    if not isinstance(value, SequentialPilotOutcomeProjectorPort):
        raise TypeError("pilot projector must implement its application port")
    identity = (
        getattr(value, "projector_id", None),
        getattr(value, "projector_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="projector_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("projector_version must be positive")
    require_sha256(identity[2], "projector definition_sha256")
    return identity  # type: ignore[return-value]


def _gate_identity(
    value: SequentialAllocationGatePort,
) -> tuple[str, int, str]:
    if not isinstance(value, SequentialAllocationGatePort):
        raise TypeError("gate must implement its application port")
    identity = (
        getattr(value, "gate_id", None),
        getattr(value, "gate_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="gate_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("gate_version must be positive")
    require_sha256(identity[2], "gate definition_sha256")
    return identity  # type: ignore[return-value]


class SequentialResidualPhase(str, Enum):
    PLAN_FROZEN = "plan_frozen"
    PILOT_EVALUATED = "pilot_evaluated"
    PILOT_ADJUDICATED = "pilot_adjudicated"
    FINALIZED = "finalized"


@dataclass(frozen=True, slots=True)
class SequentialResidualPhaseReceipt:
    """One hash-bound interception boundary in a sequential stage."""

    phase: SequentialResidualPhase
    residual_request_sha256: str
    plan_sha256: str
    product_sha256s: tuple[str, ...]
    evidence: FrozenJsonObject
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.phase) is not SequentialResidualPhase:
            raise TypeError("phase must be an exact SequentialResidualPhase")
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        require_sha256(self.plan_sha256, "plan_sha256")
        if (
            type(self.product_sha256s) is not tuple
            or not self.product_sha256s
        ):
            raise ValueError("product_sha256s must be a non-empty exact tuple")
        for value in self.product_sha256s:
            require_sha256(value, "product_sha256")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("phase evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_PHASE_RECEIPT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": self.phase.value,
            "residual_request_sha256": self.residual_request_sha256,
            "plan_sha256": self.plan_sha256,
            "product_sha256s": list(self.product_sha256s),
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "receipt_sha256": self.receipt_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@dataclass(frozen=True, slots=True)
class SequentialResidualPhaseCommitAck:
    """Proof that a supplied interception boundary accepted one receipt."""

    committer_id: str
    committer_version: int
    committer_definition_sha256: str
    phase_receipt_sha256: str
    durable: bool
    evidence: FrozenJsonObject
    ack_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.committer_id, name="committer_id")
        if type(self.committer_version) is not int or self.committer_version <= 0:
            raise ValueError("committer_version must be positive")
        require_sha256(
            self.committer_definition_sha256,
            "committer_definition_sha256",
        )
        require_sha256(
            self.phase_receipt_sha256,
            "phase_receipt_sha256",
        )
        if type(self.durable) is not bool:
            raise TypeError("durable must be an exact bool")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("commit evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "ack_sha256",
            _hash(_PHASE_ACK_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "committer": {
                "committer_id": self.committer_id,
                "committer_version": self.committer_version,
                "definition_sha256": self.committer_definition_sha256,
            },
            "phase_receipt_sha256": self.phase_receipt_sha256,
            "durable": self.durable,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "ack_sha256": self.ack_sha256}
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class SequentialResidualPhaseCommitPort(Protocol):
    """Durably intercept a sequential phase before later evaluation authority."""

    committer_id: str
    committer_version: int
    definition_sha256: str

    async def commit(
        self,
        receipt: SequentialResidualPhaseReceipt,
    ) -> SequentialResidualPhaseCommitAck: ...


def _committer_identity(
    value: SequentialResidualPhaseCommitPort,
) -> tuple[str, int, str]:
    if not isinstance(value, SequentialResidualPhaseCommitPort):
        raise TypeError("phase committer must implement its application port")
    identity = (
        getattr(value, "committer_id", None),
        getattr(value, "committer_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="committer_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("committer_version must be positive")
    require_sha256(identity[2], "committer definition_sha256")
    return identity  # type: ignore[return-value]


def _flatten_evaluations(
    batches: tuple[MaterializedActionEvaluationBatch, ...],
) -> tuple[MaterializedActionEvaluation, ...]:
    by_action = {
        value.action.action_sha256: value
        for batch in batches
        for value in batch.evaluations
    }
    return tuple(by_action[value] for value in sorted(by_action))


def _merge_evaluation_waves(
    proposals: tuple[MaterializedActionProposalBatch, ...],
    pilot: tuple[MaterializedActionEvaluationBatch, ...],
    continuation: tuple[MaterializedActionEvaluationBatch, ...],
) -> tuple[MaterializedActionEvaluationBatch, ...]:
    """Merge per-wave expert batches without erasing their source receipts."""

    proposal_by_expert = {value.expert_id: value for value in proposals}
    batches_by_expert: dict[
        str, list[MaterializedActionEvaluationBatch]
    ] = {}
    for batch in (*pilot, *continuation):
        batches_by_expert.setdefault(batch.expert_id, []).append(batch)
    merged: list[MaterializedActionEvaluationBatch] = []
    for expert_id in sorted(batches_by_expert):
        source = batches_by_expert[expert_id]
        proposal = proposal_by_expert[expert_id]
        evaluations = tuple(
            sorted(
                (
                    value
                    for batch in source
                    for value in batch.evaluations
                ),
                key=lambda value: value.action.action_sha256,
            )
        )
        action_ids = tuple(
            value.action.action_sha256 for value in evaluations
        )
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("pilot and continuation repeat an evaluation")
        merged.append(
            MaterializedActionEvaluationBatch(
                proposal_sha256=proposal.proposal_sha256,
                expert_id=proposal.expert_id,
                expert_version=proposal.expert_version,
                expert_definition_sha256=(
                    proposal.expert_definition_sha256
                ),
                selected_action_sha256s=action_ids,
                evaluations=evaluations,
                evidence=freeze_json(
                    {
                        "sequential_wave_batch_sha256s": [
                            value.batch_sha256 for value in source
                        ],
                        "pilot_then_continuation": True,
                        "candidate_outcomes_reused": False,
                    }
                ),
            )
        )
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class SequentialResidualPortfolioEvolutionResult:
    """Full sequential trace plus the standard combined downstream result."""

    result: ResidualPortfolioEvolutionResult
    allocation_plan: SequentialAllocationPlanPort
    pilot_evaluation_batches: tuple[
        MaterializedActionEvaluationBatch, ...
    ]
    pilot_outcomes: SequentialPilotOutcomeBatch
    gate_decision: SequentialAllocationGateDecisionPort
    continuation_evaluation_batches: tuple[
        MaterializedActionEvaluationBatch, ...
    ]
    phase_receipts: tuple[SequentialResidualPhaseReceipt, ...]
    phase_commit_acks: tuple[SequentialResidualPhaseCommitAck, ...]
    method_definition_sha256: str = (
        SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_DEFINITION_SHA256
    )
    sequential_result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.result) is not ResidualPortfolioEvolutionResult:
            raise TypeError("result must be exact")
        self.result.__post_init__()
        if not isinstance(
            self.allocation_plan,
            SequentialAllocationPlanPort,
        ):
            raise TypeError(
                "allocation_plan must implement its application port"
            )
        if (
            self.allocation_plan.residual_request_sha256
            != self.result.request.request_sha256
            or self.allocation_plan.proposal_sha256s
            != tuple(
                sorted(
                    value.proposal_sha256 for value in self.result.proposals
                )
            )
        ):
            raise ValueError("allocation plan differs from the combined result")
        for batches, name in (
            (self.pilot_evaluation_batches, "pilot_evaluation_batches"),
            (
                self.continuation_evaluation_batches,
                "continuation_evaluation_batches",
            ),
        ):
            if type(batches) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            for value in batches:
                if type(value) is not MaterializedActionEvaluationBatch:
                    raise TypeError(f"{name} must contain exact batches")
                value.__post_init__()
        if type(self.pilot_outcomes) is not SequentialPilotOutcomeBatch:
            raise TypeError("pilot_outcomes must be exact")
        self.pilot_outcomes.__post_init__()
        if not isinstance(
            self.gate_decision,
            SequentialAllocationGateDecisionPort,
        ):
            raise TypeError(
                "gate_decision must implement its application port"
            )
        if (
            self.pilot_outcomes.plan_sha256
            != self.allocation_plan.plan_sha256
            or self.gate_decision.plan_sha256
            != self.allocation_plan.plan_sha256
            or self.gate_decision.pilot_outcome_batch_sha256
            != self.pilot_outcomes.batch_sha256
        ):
            raise ValueError("pilot outcome/gate trace does not join the plan")
        selected_requirement = self.allocation_plan.requirement_for(
            self.gate_decision.selected_branch_id
        )
        if (
            self.gate_decision.selected_requirement_sha256
            != selected_requirement.requirement_sha256
            or self.result.broker_decision.allocation_requirement
            != selected_requirement
        ):
            raise ValueError("combined broker result differs from gate branch")
        pilot_ids = {
            value.action.action_sha256
            for value in _flatten_evaluations(
                self.pilot_evaluation_batches
            )
        }
        continuation_ids = {
            value.action.action_sha256
            for value in _flatten_evaluations(
                self.continuation_evaluation_batches
            )
        }
        if pilot_ids != set(self.allocation_plan.pilot_action_sha256s):
            raise ValueError("pilot evaluations differ from the plan")
        if pilot_ids & continuation_ids:
            raise ValueError("continuation reevaluates a pilot")
        if pilot_ids | continuation_ids != set(
            selected_requirement.required_action_sha256s
        ):
            raise ValueError("two evaluation waves do not close the branch")
        if (
            type(self.phase_receipts) is not tuple
            or tuple(value.phase for value in self.phase_receipts)
            != tuple(SequentialResidualPhase)
        ):
            raise ValueError("phase receipts must cover canonical phase order")
        for value in self.phase_receipts:
            value.__post_init__()
            if value.plan_sha256 != self.allocation_plan.plan_sha256:
                raise ValueError("phase receipt names another plan")
        if type(self.phase_commit_acks) is not tuple:
            raise TypeError("phase_commit_acks must be an exact tuple")
        for value in self.phase_commit_acks:
            value.__post_init__()
        if self.phase_commit_acks and tuple(
            value.phase_receipt_sha256
            for value in self.phase_commit_acks
        ) != tuple(value.receipt_sha256 for value in self.phase_receipts):
            raise ValueError("phase commit acknowledgements do not close")
        require_sha256(
            self.method_definition_sha256,
            "method_definition_sha256",
        )
        object.__setattr__(
            self,
            "sequential_result_sha256",
            _hash(_RESULT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": {
                "method_id": SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_ID,
                "method_version": (
                    SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_VERSION
                ),
                "definition_sha256": self.method_definition_sha256,
            },
            "combined_result_sha256": self.result.result_sha256,
            "allocation_plan_sha256": self.allocation_plan.plan_sha256,
            "pilot_evaluation_batch_sha256s": [
                value.batch_sha256
                for value in self.pilot_evaluation_batches
            ],
            "pilot_outcome_batch_sha256": self.pilot_outcomes.batch_sha256,
            "gate_decision_sha256": self.gate_decision.decision_sha256,
            "continuation_evaluation_batch_sha256s": [
                value.batch_sha256
                for value in self.continuation_evaluation_batches
            ],
            "phase_receipt_sha256s": [
                value.receipt_sha256 for value in self.phase_receipts
            ],
            "phase_commit_ack_sha256s": [
                value.ack_sha256 for value in self.phase_commit_acks
            ],
            "archive_credit_in_combined_result": False,
            "workload_objective_model_provider_prompt_branches": False,
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "allocation_plan": self.allocation_plan.to_record(
                include_evidence=include_evidence
            ),
            "pilot_evaluation_batches": [
                value.to_record(include_evidence=include_evidence)
                for value in self.pilot_evaluation_batches
            ],
            "pilot_outcomes": self.pilot_outcomes.to_record(
                include_evidence=include_evidence
            ),
            "gate_decision": self.gate_decision.to_record(
                include_evidence=include_evidence
            ),
            "continuation_evaluation_batches": [
                value.to_record(include_evidence=include_evidence)
                for value in self.continuation_evaluation_batches
            ],
            "phase_receipts": [
                value.to_record(include_evidence=include_evidence)
                for value in self.phase_receipts
            ],
            "phase_commit_acks": [
                value.to_record(include_evidence=include_evidence)
                for value in self.phase_commit_acks
            ],
            "combined_result": self.result.to_record(),
            "sequential_result_sha256": self.sequential_result_sha256,
        }


@dataclass(frozen=True, slots=True)
class SequentialResidualPortfolioEvolution:
    """Execute one proposal universe through a K-pilot plus K-rest gate."""

    experts: tuple[MaterializedActionProposalExpertPort, ...]
    broker: RegretBrokeredMaterializedActionPolicy
    planner: SequentialLineageAllocationPlannerPort = field(
        repr=False,
        compare=False,
    )
    pilot_outcome_projector: SequentialPilotOutcomeProjectorPort = field(
        repr=False,
        compare=False,
    )
    slate_value: MaterializedSlateValuePort = field(
        repr=False,
        compare=False,
    )
    slate_feasibility: MaterializedSlateFeasibilityPort = field(
        repr=False,
        compare=False,
    )
    gate: SequentialAllocationGatePort = field(
        default_factory=AnyPositiveSequentialLineageGate,
        repr=False,
        compare=False,
    )
    phase_committer: SequentialResidualPhaseCommitPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    require_durable_phase_commits: bool = False

    def __post_init__(self) -> None:
        if type(self.experts) is not tuple or not self.experts:
            raise ValueError("experts must be a non-empty exact tuple")
        if any(
            getattr(value, "evaluation_wave_semantics", None)
            != DISJOINT_ACTION_EVALUATION_WAVES_V1
            for value in self.experts
        ):
            raise ValueError(
                "sequential experts must support disjoint action "
                "evaluation waves"
            )
        if type(self.broker) is not RegretBrokeredMaterializedActionPolicy:
            raise TypeError("broker must be an exact regret-broker policy")
        self.broker.__post_init__()
        _planner_identity(self.planner)
        _projector_identity(self.pilot_outcome_projector)
        if not isinstance(self.slate_value, MaterializedSlateValuePort):
            raise TypeError("slate_value must implement its application port")
        if not isinstance(
            self.slate_feasibility,
            MaterializedSlateFeasibilityPort,
        ):
            raise TypeError(
                "slate_feasibility must implement its application port"
            )
        require_sha256(
            self.slate_value.definition_sha256,
            "slate value definition_sha256",
        )
        require_sha256(
            self.slate_feasibility.definition_sha256,
            "slate feasibility definition_sha256",
        )
        _gate_identity(self.gate)
        if self.phase_committer is not None:
            _committer_identity(self.phase_committer)
        if type(self.require_durable_phase_commits) is not bool:
            raise TypeError("require_durable_phase_commits must be exact")
        if (
            self.require_durable_phase_commits
            and self.phase_committer is None
        ):
            raise ValueError(
                "required durable phase commits need a committer"
            )

    async def _commit(
        self,
        receipt: SequentialResidualPhaseReceipt,
    ) -> SequentialResidualPhaseCommitAck | None:
        if self.phase_committer is None:
            return None
        identity = _committer_identity(self.phase_committer)
        ack = await self.phase_committer.commit(receipt)
        if type(ack) is not SequentialResidualPhaseCommitAck:
            raise TypeError("phase committer returned a foreign ack")
        ack.__post_init__()
        if (
            ack.committer_id,
            ack.committer_version,
            ack.committer_definition_sha256,
        ) != identity:
            raise ValueError("phase commit ack has a foreign identity")
        if ack.phase_receipt_sha256 != receipt.receipt_sha256:
            raise ValueError("phase commit ack targets another receipt")
        if self.require_durable_phase_commits and not ack.durable:
            raise ValueError("required phase commit was not durable")
        return ack

    async def run(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> SequentialResidualPortfolioEvolutionResult:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        proposals = await propose_materialized_action_batches(
            experts=self.experts,
            request=request,
        )
        plan = await self.planner.plan(request, proposals)
        if not isinstance(plan, SequentialAllocationPlanPort):
            raise TypeError(
                "planner returned a value outside the plan port"
            )
        plan_receipt = SequentialResidualPhaseReceipt(
            phase=SequentialResidualPhase.PLAN_FROZEN,
            residual_request_sha256=request.request_sha256,
            plan_sha256=plan.plan_sha256,
            product_sha256s=(
                plan.pilot_requirement.requirement_sha256,
                *(
                    value.requirement_sha256
                    for value in plan.frozen_requirements
                ),
            ),
            evidence=freeze_json(
                {
                    "proposal_sha256s": list(plan.proposal_sha256s),
                    "complete_branch_count": len(
                        plan.frozen_requirements
                    ),
                    "all_complete_branches_frozen": True,
                    "current_candidate_outcomes_observed": False,
                    "allocation_plan": plan.to_record(
                        include_evidence=True
                    ),
                }
            ),
        )
        phase_receipts = [plan_receipt]
        phase_acks: list[SequentialResidualPhaseCommitAck] = []
        plan_ack = await self._commit(plan_receipt)
        if plan_ack is not None:
            phase_acks.append(plan_ack)

        pilot_batches: tuple[MaterializedActionEvaluationBatch, ...]
        if plan.pilot_action_sha256s:
            pilot_batches = await evaluate_materialized_action_subset(
                experts=self.experts,
                proposals=proposals,
                selected_action_sha256s=plan.pilot_action_sha256s,
            )
        else:
            pilot_batches = ()
        pilot_evaluations = _flatten_evaluations(pilot_batches)
        pilot_outcomes = self.pilot_outcome_projector.project(
            plan,
            pilot_evaluations,
        )
        pilot_evaluated_receipt = SequentialResidualPhaseReceipt(
            phase=SequentialResidualPhase.PILOT_EVALUATED,
            residual_request_sha256=request.request_sha256,
            plan_sha256=plan.plan_sha256,
            product_sha256s=(
                *(
                    value.batch_sha256 for value in pilot_batches
                ),
                pilot_outcomes.batch_sha256,
            ),
            evidence=freeze_json(
                {
                    "pilot_action_sha256s": list(
                        plan.pilot_action_sha256s
                    ),
                    "pilot_evaluation_count": len(
                        pilot_evaluations
                    ),
                    "later_candidate_outcomes_observed": False,
                    "gate_decision_observed": False,
                    "pilot_evaluation_batches": [
                        value.to_record(include_evidence=True)
                        for value in pilot_batches
                    ],
                    "pilot_outcomes": pilot_outcomes.to_record(
                        include_evidence=True
                    ),
                }
            ),
        )
        phase_receipts.append(pilot_evaluated_receipt)
        pilot_evaluated_ack = await self._commit(
            pilot_evaluated_receipt
        )
        if pilot_evaluated_ack is not None:
            phase_acks.append(pilot_evaluated_ack)

        gate_decision = self.gate.decide(plan, pilot_outcomes)
        if not isinstance(
            gate_decision,
            SequentialAllocationGateDecisionPort,
        ):
            raise TypeError("gate returned a value outside its decision port")
        selected_requirement = plan.requirement_for(
            gate_decision.selected_branch_id
        )
        if (
            gate_decision.plan_sha256 != plan.plan_sha256
            or gate_decision.pilot_outcome_batch_sha256
            != pilot_outcomes.batch_sha256
            or gate_decision.selected_requirement_sha256
            != selected_requirement.requirement_sha256
        ):
            raise ValueError("gate decision does not close the frozen plan")
        pilot_receipt = SequentialResidualPhaseReceipt(
            phase=SequentialResidualPhase.PILOT_ADJUDICATED,
            residual_request_sha256=request.request_sha256,
            plan_sha256=plan.plan_sha256,
            product_sha256s=(
                pilot_evaluated_receipt.receipt_sha256,
                gate_decision.decision_sha256,
                selected_requirement.requirement_sha256,
            ),
            evidence=freeze_json(
                {
                    "pilot_action_sha256s": list(
                        plan.pilot_action_sha256s
                    ),
                    "positive_pilot_count": (
                        gate_decision.positive_pilot_count
                    ),
                    "selected_branch": (
                        gate_decision.selected_branch_id
                    ),
                    "later_candidate_outcomes_observed": False,
                    "pilot_evaluated_receipt_sha256": (
                        pilot_evaluated_receipt.receipt_sha256
                    ),
                    "gate_decision": gate_decision.to_record(
                        include_evidence=True
                    ),
                    "selected_requirement": (
                        selected_requirement.to_record(
                            include_evidence=True
                        )
                    ),
                }
            ),
        )
        phase_receipts.append(pilot_receipt)
        pilot_ack = await self._commit(pilot_receipt)
        if pilot_ack is not None:
            phase_acks.append(pilot_ack)

        pilot_ids = set(plan.pilot_action_sha256s)
        continuation_ids = tuple(
            value
            for value in selected_requirement.required_action_sha256s
            if value not in pilot_ids
        )
        if not continuation_ids:
            raise ValueError("sequential branch leaves no continuation wave")
        continuation_batches = await evaluate_materialized_action_subset(
            experts=self.experts,
            proposals=proposals,
            selected_action_sha256s=continuation_ids,
        )
        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )
        final_decision = self.broker.select(
            MaterializedActionBrokerRequest(
                actions=actions,
                evaluation_slots=request.evaluation_slots,
                slate_value=self.slate_value,
                slate_feasibility=self.slate_feasibility,
                reference_escrow_slots=request.reference_escrow_slots,
                allocation_requirement=selected_requirement,
            )
        )
        if tuple(
            value.action_sha256 for value in final_decision.selected_actions
        ) != selected_requirement.required_action_sha256s:
            raise ValueError("final broker did not honor the complete branch")
        merged = _merge_evaluation_waves(
            proposals,
            pilot_batches,
            continuation_batches,
        )
        combined = ResidualPortfolioEvolutionResult(
            request=request,
            proposals=proposals,
            broker_decision=final_decision,
            evaluation_batches=merged,
            slate_value_definition_sha256=self.slate_value.definition_sha256,
            slate_feasibility_definition_sha256=(
                self.slate_feasibility.definition_sha256
            ),
        )
        final_receipt = SequentialResidualPhaseReceipt(
            phase=SequentialResidualPhase.FINALIZED,
            residual_request_sha256=request.request_sha256,
            plan_sha256=plan.plan_sha256,
            product_sha256s=(
                *(
                    value.batch_sha256
                    for value in continuation_batches
                ),
                final_decision.decision_sha256,
                combined.result_sha256,
            ),
            evidence=freeze_json(
                {
                    "selected_branch": (
                        gate_decision.selected_branch_id
                    ),
                    "selected_action_sha256s": list(
                        selected_requirement.required_action_sha256s
                    ),
                    "pilot_evaluation_count": len(pilot_evaluations),
                    "continuation_evaluation_count": len(
                        _flatten_evaluations(continuation_batches)
                    ),
                    "continuation_evaluation_batches": [
                        value.to_record(include_evidence=True)
                        for value in continuation_batches
                    ],
                    "final_broker_decision": final_decision.to_record(),
                    "real_evaluation_budget_preserved": True,
                }
            ),
        )
        phase_receipts.append(final_receipt)
        final_ack = await self._commit(final_receipt)
        if final_ack is not None:
            phase_acks.append(final_ack)
        return SequentialResidualPortfolioEvolutionResult(
            result=combined,
            allocation_plan=plan,
            pilot_evaluation_batches=pilot_batches,
            pilot_outcomes=pilot_outcomes,
            gate_decision=gate_decision,
            continuation_evaluation_batches=continuation_batches,
            phase_receipts=tuple(phase_receipts),
            phase_commit_acks=tuple(phase_acks),
        )


__all__ = [
    "SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_DEFINITION_SHA256",
    "SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_ID",
    "SEQUENTIAL_RESIDUAL_PORTFOLIO_EVOLUTION_VERSION",
    "SequentialResidualPhase",
    "SequentialResidualPhaseCommitAck",
    "SequentialResidualPhaseCommitPort",
    "SequentialResidualPhaseReceipt",
    "SequentialResidualPortfolioEvolution",
    "SequentialResidualPortfolioEvolutionResult",
]

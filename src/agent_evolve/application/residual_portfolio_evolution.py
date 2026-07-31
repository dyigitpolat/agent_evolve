"""Generic propose -> broker -> evaluate orchestration for residual evolution.

Proposal experts own workload knowledge, candidate legality, and exact
materialization.  This application service receives only already-materialized
actions, gives every expert access to the same authenticated prior cutoff, and
lets the workload-blind action broker choose the expensive evaluations.

The service deliberately stops before archive credit.  A downstream utility
policy must turn the jointly evaluated slate into conserved immediate credit,
and a lineage resolver may later append finite-horizon return.  Keeping those
steps outside proposal execution prevents an expert from grading itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionBrokerDecision,
    MaterializedActionBrokerRequest,
    MaterializedActionAllocationRequirement,
    MaterializedActionDescriptor,
    MaterializedSlateFeasibilityPort,
    MaterializedSlateValuePort,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


RESIDUAL_PORTFOLIO_EVOLUTION_ID = "conserved_residual_portfolio_evolution"
RESIDUAL_PORTFOLIO_EVOLUTION_VERSION = 2
RESIDUAL_PORTFOLIO_EVOLUTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:conserved-residual-portfolio-evolution:v2;"
    b"boundary=authenticated-prior-cutoff-and-opaque-proposal-context;"
    b"experts=workload-owned-sealed-materialized-action-batches;"
    b"allocation=optional-async-outcome-blind-policy-over-sealed-union;"
    b"broker=workload-model-provider-blind-materialized-action-selection;"
    b"reference=conservative-escrow;"
    b"execution=selected-actions-only-grouped-by-owning-expert-concurrent;"
    b"credit=downstream-conserved-archive-return-not-expert-self-score;"
    b"lineage=downstream-append-only-resolved-return;"
    b"core-workload-model-provider-branches=false"
).hexdigest()
DISJOINT_ACTION_EVALUATION_WAVES_V1 = "disjoint_action_subsets_v1"

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_REQUEST_DOMAIN = b"agent-evolve:residual-portfolio-request:v1\x00"
_PROPOSAL_DOMAIN = b"agent-evolve:residual-portfolio-proposal:v1\x00"
_EVALUATION_DOMAIN = b"agent-evolve:residual-portfolio-evaluation:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:residual-portfolio-result:v1\x00"


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


def _expert_capacity(
    values: tuple[tuple[str, int], ...],
    expert_id: str,
) -> int:
    try:
        return dict(values)[expert_id]
    except KeyError as error:
        raise ValueError("proposal expert is absent from the request") from error


def _candidate_record(candidate: EvolutionCandidate) -> dict[str, object]:
    if type(candidate) is not EvolutionCandidate:
        raise TypeError("evaluations must contain exact EvolutionCandidate values")
    EvolutionCandidate.__post_init__(candidate)
    detailed = candidate.detailed_evaluation
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "generation": candidate.generation,
        "valid": candidate.valid,
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "objectives": [
            {"metric_id": metric_id, "value_hex": value.hex()}
            for metric_id, value in candidate.objectives
        ],
        "detailed_evaluation_sha256": (
            None if detailed is None else detailed.evidence_sha256
        ),
    }


@dataclass(frozen=True, slots=True)
class ResidualPortfolioDecisionRequest:
    """Authenticated common cutoff presented to every proposal expert.

    ``proposal_context`` is intentionally opaque to the core.  A workload
    adapter can place typed, replayable state in that object, while broker
    behavior remains independent of its fields.
    """

    campaign_scope_sha256: str
    prior_state_sha256: str
    decision_index: int
    phase: SearchPhase
    remaining_decisions: int
    remaining_evaluations: int
    evaluation_slots: int
    expert_proposal_slots: tuple[tuple[str, int], ...]
    proposal_context: FrozenJsonObject
    reference_escrow_slots: int = 1
    request_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(self.prior_state_sha256, "prior_state_sha256")
        if type(self.decision_index) is not int or self.decision_index <= 0:
            raise ValueError("decision_index must be a positive exact integer")
        if type(self.phase) is not SearchPhase:
            raise TypeError("phase must be an exact SearchPhase")
        for name in ("remaining_decisions", "remaining_evaluations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            type(self.evaluation_slots) is not int
            or self.evaluation_slots <= 0
            or self.evaluation_slots > self.remaining_evaluations
        ):
            raise ValueError("evaluation_slots must fit remaining evaluations")
        if (
            type(self.expert_proposal_slots) is not tuple
            or not self.expert_proposal_slots
        ):
            raise ValueError("expert_proposal_slots must be a non-empty exact tuple")
        for expert_id, slots in self.expert_proposal_slots:
            _require_token(expert_id, name="expert_id")
            if type(slots) is not int or slots <= 0:
                raise ValueError("expert proposal slots must be positive")
        expert_ids = tuple(value[0] for value in self.expert_proposal_slots)
        if expert_ids != tuple(sorted(set(expert_ids))):
            raise ValueError("expert proposal capacities must be unique and canonical")
        if sum(value[1] for value in self.expert_proposal_slots) < (
            self.evaluation_slots
        ):
            raise ValueError("proposal capacity cannot cover evaluation capacity")
        if (
            type(self.reference_escrow_slots) is not int
            or not 0 <= self.reference_escrow_slots <= self.evaluation_slots
        ):
            raise ValueError("reference escrow must fit evaluation capacity")
        if (
            type(self.proposal_context) is not FrozenJsonObject
            or freeze_json(self.proposal_context) is not self.proposal_context
        ):
            raise TypeError("proposal_context must be an exact frozen JSON object")
        object.__setattr__(
            self,
            "request_sha256",
            _hash(_REQUEST_DOMAIN, self._unsigned_record()),
        )

    def proposal_slots_for(self, expert_id: str) -> int:
        _require_token(expert_id, name="expert_id")
        return _expert_capacity(self.expert_proposal_slots, expert_id)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "prior_state_sha256": self.prior_state_sha256,
            "decision_index": self.decision_index,
            "phase": self.phase.value,
            "remaining_decisions": self.remaining_decisions,
            "remaining_evaluations": self.remaining_evaluations,
            "evaluation_slots": self.evaluation_slots,
            "expert_proposal_slots": [
                {"expert_id": expert_id, "proposal_slots": slots}
                for expert_id, slots in self.expert_proposal_slots
            ],
            "proposal_context_sha256": typed_json_sha256(self.proposal_context),
            "reference_escrow_slots": self.reference_escrow_slots,
        }

    def to_record(self, *, include_proposal_context: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "request_sha256": self.request_sha256}
        if include_proposal_context:
            record["proposal_context"] = thaw_json(self.proposal_context)
        return record


@dataclass(frozen=True, slots=True)
class MaterializedActionProposalBatch:
    """One expert's sealed, unevaluated action batch."""

    request_sha256: str
    expert_id: str
    expert_version: int
    expert_definition_sha256: str
    actions: tuple[MaterializedActionDescriptor, ...]
    evidence: FrozenJsonObject
    proposal_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        _require_token(self.expert_id, name="expert_id")
        if type(self.expert_version) is not int or self.expert_version <= 0:
            raise ValueError("expert_version must be positive")
        require_sha256(
            self.expert_definition_sha256,
            "expert_definition_sha256",
        )
        if type(self.actions) is not tuple or not self.actions:
            raise ValueError("proposal batch must contain at least one action")
        for action in self.actions:
            if type(action) is not MaterializedActionDescriptor:
                raise TypeError("proposal actions must be exact descriptors")
            action.__post_init__()
            if action.expert_id != self.expert_id:
                raise ValueError("proposal action names another expert")
        if tuple(value.native_rank for value in self.actions) != tuple(
            range(1, len(self.actions) + 1)
        ):
            raise ValueError("expert actions must retain contiguous native ranks")
        for values, name in (
            (
                tuple(value.action_sha256 for value in self.actions),
                "action identities",
            ),
            (
                tuple(value.target_candidate_id for value in self.actions),
                "target candidate IDs",
            ),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"proposal batch repeats {name}")
        contexts = {
            (
                value.context.campaign_scope_sha256,
                value.context.decision_index,
                value.context.phase,
            )
            for value in self.actions
        }
        if len(contexts) != 1:
            raise ValueError("one proposal batch cannot mix decision cutoffs")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("proposal evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "proposal_sha256",
            _hash(_PROPOSAL_DOMAIN, self._unsigned_record()),
        )

    def require_request(self, request: ResidualPortfolioDecisionRequest) -> None:
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if self.request_sha256 != request.request_sha256:
            raise ValueError("proposal targets another residual decision")
        if len(self.actions) > request.proposal_slots_for(self.expert_id):
            raise ValueError("expert exceeded its proposal capacity")
        for action in self.actions:
            context = action.context
            if (
                context.campaign_scope_sha256 != request.campaign_scope_sha256
                or context.decision_index != request.decision_index
                or context.phase is not request.phase
                or context.remaining_decisions != request.remaining_decisions
                or context.remaining_evaluations != request.remaining_evaluations
            ):
                raise ValueError("proposal action context differs from the request")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "expert": {
                "expert_id": self.expert_id,
                "expert_version": self.expert_version,
                "definition_sha256": self.expert_definition_sha256,
            },
            "actions": [value.to_record() for value in self.actions],
            "evidence_sha256": typed_json_sha256(self.evidence),
            "evaluation_performed": False,
        }

    def to_record(
        self,
        *,
        include_configurations: bool = False,
        include_evidence: bool = False,
    ) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "actions": [
                value.to_record(include_configuration=include_configurations)
                for value in self.actions
            ],
            "proposal_sha256": self.proposal_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@dataclass(frozen=True, slots=True)
class DisjointActionEvaluationWave:
    """One fail-closed reservation of previously unevaluated actions.

    Sequential allocation can evaluate a pilot and a continuation from the
    same sealed proposal.  The action—not the proposal batch—is therefore the
    exactly-once boundary.  A reservation is recorded before invoking the
    authoritative evaluator so an interrupted wave cannot be retried
    ambiguously through the same live expert instance.
    """

    proposal_sha256: str
    wave_index: int
    previously_attempted_action_count: int
    selected_action_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.proposal_sha256, "proposal_sha256")
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be a positive exact integer")
        if (
            type(self.previously_attempted_action_count) is not int
            or self.previously_attempted_action_count < 0
        ):
            raise ValueError(
                "previously_attempted_action_count must be non-negative"
            )
        if (
            type(self.selected_action_sha256s) is not tuple
            or not self.selected_action_sha256s
            or self.selected_action_sha256s
            != tuple(sorted(set(self.selected_action_sha256s)))
        ):
            raise ValueError(
                "selected_action_sha256s must be non-empty and canonical"
            )
        for value in self.selected_action_sha256s:
            require_sha256(value, "selected_action_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "semantics": DISJOINT_ACTION_EVALUATION_WAVES_V1,
            "wave_index": self.wave_index,
            "previously_attempted_action_count": (
                self.previously_attempted_action_count
            ),
            "selected_action_sha256s": list(
                self.selected_action_sha256s
            ),
            "selected_subset_frozen_before_wave": True,
            "selected_actions_disjoint_from_previous_waves": True,
            "repeated_action_evaluation_forbidden": True,
        }


@dataclass(slots=True)
class DisjointActionEvaluationLedger:
    """Reserve disjoint evaluation waves over any sealed proposal batch."""

    _attempted_by_proposal: dict[str, set[str]] = field(
        init=False,
        default_factory=dict,
    )
    _wave_count_by_proposal: dict[str, int] = field(
        init=False,
        default_factory=dict,
    )

    def reserve(
        self,
        proposal: MaterializedActionProposalBatch,
        selected_action_sha256s: tuple[str, ...],
    ) -> DisjointActionEvaluationWave:
        if type(proposal) is not MaterializedActionProposalBatch:
            raise TypeError("proposal must be exact")
        proposal.__post_init__()
        if (
            type(selected_action_sha256s) is not tuple
            or not selected_action_sha256s
            or selected_action_sha256s
            != tuple(sorted(set(selected_action_sha256s)))
        ):
            raise ValueError(
                "selected action hashes must be non-empty and canonical"
            )
        available = {
            value.action_sha256 for value in proposal.actions
        }
        selected = set(selected_action_sha256s)
        if not selected.issubset(available):
            raise ValueError("selected action is outside the sealed proposal")
        attempted = self._attempted_by_proposal.setdefault(
            proposal.proposal_sha256,
            set(),
        )
        overlap = tuple(sorted(selected.intersection(attempted)))
        if overlap:
            raise ValueError(
                "an action cannot be evaluated in more than one wave: "
                + ",".join(overlap)
            )
        wave_index = (
            self._wave_count_by_proposal.get(
                proposal.proposal_sha256,
                0,
            )
            + 1
        )
        reservation = DisjointActionEvaluationWave(
            proposal_sha256=proposal.proposal_sha256,
            wave_index=wave_index,
            previously_attempted_action_count=len(attempted),
            selected_action_sha256s=selected_action_sha256s,
        )
        # Reserve before any evaluator await.  Failure is deliberately
        # fail-closed: a fresh run may replay from durable evidence, while this
        # live expert cannot accidentally double-spend evaluator budget.
        attempted.update(selected)
        self._wave_count_by_proposal[proposal.proposal_sha256] = wave_index
        return reservation


@dataclass(frozen=True, slots=True)
class MaterializedActionEvaluation:
    """Exact join between one selected action and its real candidate."""

    action: MaterializedActionDescriptor
    candidate: EvolutionCandidate
    evaluator_receipt_sha256: str
    evaluation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.action) is not MaterializedActionDescriptor:
            raise TypeError("action must be an exact descriptor")
        self.action.__post_init__()
        if type(self.candidate) is not EvolutionCandidate:
            raise TypeError("candidate must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(self.candidate)
        require_sha256(
            self.evaluator_receipt_sha256,
            "evaluator_receipt_sha256",
        )
        if self.candidate.candidate_id != self.action.target_candidate_id:
            raise ValueError("evaluated candidate ID differs from its action")
        if (
            self.candidate.occurrence.configuration_hash
            != self.action.configuration_sha256
        ):
            raise ValueError("evaluated configuration differs from its action")
        object.__setattr__(
            self,
            "evaluation_sha256",
            _hash(_EVALUATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action.action_sha256,
            "evaluator_receipt_sha256": self.evaluator_receipt_sha256,
            "candidate": _candidate_record(self.candidate),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "evaluation_sha256": self.evaluation_sha256,
        }


@dataclass(frozen=True, slots=True)
class MaterializedActionEvaluationBatch:
    """One expert's exact response for its broker-selected subset."""

    proposal_sha256: str
    expert_id: str
    expert_version: int
    expert_definition_sha256: str
    selected_action_sha256s: tuple[str, ...]
    evaluations: tuple[MaterializedActionEvaluation, ...]
    evidence: FrozenJsonObject
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.proposal_sha256, "proposal_sha256")
        _require_token(self.expert_id, name="expert_id")
        if type(self.expert_version) is not int or self.expert_version <= 0:
            raise ValueError("expert_version must be positive")
        require_sha256(
            self.expert_definition_sha256,
            "expert_definition_sha256",
        )
        if (
            type(self.selected_action_sha256s) is not tuple
            or not self.selected_action_sha256s
        ):
            raise ValueError("selected action hashes must be non-empty")
        for value in self.selected_action_sha256s:
            require_sha256(value, "selected_action_sha256")
        if self.selected_action_sha256s != tuple(
            sorted(set(self.selected_action_sha256s))
        ):
            raise ValueError("selected action hashes must be unique and canonical")
        if (
            type(self.evaluations) is not tuple
            or len(self.evaluations) != len(self.selected_action_sha256s)
        ):
            raise ValueError("evaluations must exactly cover selected actions")
        for value in self.evaluations:
            if type(value) is not MaterializedActionEvaluation:
                raise TypeError("evaluations must contain exact values")
            value.__post_init__()
            if value.action.expert_id != self.expert_id:
                raise ValueError("evaluation action names another expert")
        if tuple(value.action.action_sha256 for value in self.evaluations) != (
            self.selected_action_sha256s
        ):
            raise ValueError("evaluation order differs from selected action order")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("evaluation evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "batch_sha256",
            _hash(_EVALUATION_DOMAIN, self._unsigned_record()),
        )

    def require_proposal(self, proposal: MaterializedActionProposalBatch) -> None:
        if type(proposal) is not MaterializedActionProposalBatch:
            raise TypeError("proposal must be exact")
        proposal.__post_init__()
        if (
            self.proposal_sha256 != proposal.proposal_sha256
            or (
                self.expert_id,
                self.expert_version,
                self.expert_definition_sha256,
            )
            != (
                proposal.expert_id,
                proposal.expert_version,
                proposal.expert_definition_sha256,
            )
        ):
            raise ValueError("evaluation batch differs from its proposal identity")
        action_by_sha256 = {
            value.action_sha256: value for value in proposal.actions
        }
        try:
            expected = tuple(
                action_by_sha256[value] for value in self.selected_action_sha256s
            )
        except KeyError as error:
            raise ValueError(
                "evaluation selected outside the sealed proposal"
            ) from error
        if tuple(value.action for value in self.evaluations) != expected:
            raise ValueError("evaluation descriptors differ from sealed actions")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "proposal_sha256": self.proposal_sha256,
            "expert": {
                "expert_id": self.expert_id,
                "expert_version": self.expert_version,
                "definition_sha256": self.expert_definition_sha256,
            },
            "selected_action_sha256s": list(self.selected_action_sha256s),
            "evaluations": [value.to_record() for value in self.evaluations],
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "batch_sha256": self.batch_sha256}
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class MaterializedActionProposalExpertPort(Protocol):
    """Workload-owned proposal and evaluation boundary.

    ``evaluate`` may be called more than once for one proposal only when the
    requested action subsets are disjoint.  Implementations used by a
    sequential runtime expose
    ``evaluation_wave_semantics=DISJOINT_ACTION_EVALUATION_WAVES_V1`` and
    fail closed on any repeated action.
    """

    expert_id: str
    expert_version: int
    definition_sha256: str

    async def propose(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> MaterializedActionProposalBatch: ...

    async def evaluate(
        self,
        proposal: MaterializedActionProposalBatch,
        selected_action_sha256s: tuple[str, ...],
    ) -> MaterializedActionEvaluationBatch: ...


@runtime_checkable
class MaterializedActionCounterfactualEvaluationPort(Protocol):
    """Optional workload capability for excluded real-evaluation assays.

    This method is deliberately separate from authoritative ``evaluate``.
    Implementations must execute the same physical evaluator while avoiding
    authoritative budget registration, archive admission, and any state that
    can influence later action selection.  A runtime requesting a paired
    counterfactual fails closed when the owning expert does not expose this
    capability.
    """

    async def evaluate_counterfactual(
        self,
        proposal: MaterializedActionProposalBatch,
        selected_action_sha256s: tuple[str, ...],
    ) -> MaterializedActionEvaluationBatch: ...


@runtime_checkable
class MaterializedActionAllocationPolicyPort(Protocol):
    """Select authenticated required actions from the sealed proposal union.

    The policy is asynchronous so an LLM, learned ranker, or committee can
    inspect the complete population without blocking the event loop.  It sees
    the prior-cutoff request and unevaluated proposal batches only; real
    candidate evaluations do not exist at this boundary.
    """

    policy_id: str
    policy_version: int
    definition_sha256: str

    async def require(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionAllocationRequirement: ...


def _expert_identity(
    expert: MaterializedActionProposalExpertPort,
) -> tuple[str, int, str]:
    if not isinstance(expert, MaterializedActionProposalExpertPort):
        raise TypeError("proposal expert must implement its runtime port")
    identity = (
        getattr(expert, "expert_id", None),
        getattr(expert, "expert_version", None),
        getattr(expert, "definition_sha256", None),
    )
    _require_token(identity[0], name="expert_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("proposal expert version must be positive")
    require_sha256(identity[2], "proposal expert definition_sha256")
    return identity  # type: ignore[return-value]


def _allocation_policy_identity(
    policy: MaterializedActionAllocationPolicyPort,
) -> tuple[str, int, str]:
    if not isinstance(policy, MaterializedActionAllocationPolicyPort):
        raise TypeError(
            "allocation policy must implement "
            "MaterializedActionAllocationPolicyPort"
        )
    identity = (
        getattr(policy, "policy_id", None),
        getattr(policy, "policy_version", None),
        getattr(policy, "definition_sha256", None),
    )
    _require_token(identity[0], name="allocation policy_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("allocation policy version must be positive")
    require_sha256(identity[2], "allocation policy definition_sha256")
    return identity  # type: ignore[return-value]


async def propose_materialized_action_batches(
    *,
    experts: tuple[MaterializedActionProposalExpertPort, ...],
    request: ResidualPortfolioDecisionRequest,
) -> tuple[MaterializedActionProposalBatch, ...]:
    """Run one sealed proposal wave and validate its complete expert join."""

    if type(request) is not ResidualPortfolioDecisionRequest:
        raise TypeError("request must be an exact residual portfolio request")
    request.__post_init__()
    if type(experts) is not tuple or not experts:
        raise ValueError("experts must be a non-empty exact tuple")
    identities = tuple(_expert_identity(value) for value in experts)
    expert_ids = tuple(value[0] for value in identities)
    if expert_ids != tuple(sorted(set(expert_ids))):
        raise ValueError("experts must use canonical unique IDs")
    requested_expert_ids = tuple(
        value[0] for value in request.expert_proposal_slots
    )
    if requested_expert_ids != expert_ids:
        raise ValueError("request expert capacities differ from runtime experts")
    raw = await asyncio.gather(
        *(expert.propose(request) for expert in experts)
    )
    proposals = tuple(raw)
    for identity, proposal in zip(identities, proposals, strict=True):
        if type(proposal) is not MaterializedActionProposalBatch:
            raise TypeError("proposal expert returned a foreign batch")
        proposal.__post_init__()
        proposal.require_request(request)
        if (
            proposal.expert_id,
            proposal.expert_version,
            proposal.expert_definition_sha256,
        ) != identity:
            raise ValueError("proposal batch has a foreign expert identity")
    actions = tuple(
        action for proposal in proposals for action in proposal.actions
    )
    if len(actions) < request.evaluation_slots:
        raise ValueError("realized proposal union cannot fill evaluation capacity")
    if len({value.action_sha256 for value in actions}) != len(actions):
        raise ValueError("proposal union repeats an action identity")
    if len({value.target_candidate_id for value in actions}) != len(actions):
        raise ValueError("proposal union repeats a target candidate ID")
    return proposals


async def _evaluate_materialized_action_subset(
    *,
    experts: tuple[MaterializedActionProposalExpertPort, ...],
    proposals: tuple[MaterializedActionProposalBatch, ...],
    selected_action_sha256s: tuple[str, ...],
    counterfactual: bool,
) -> tuple[MaterializedActionEvaluationBatch, ...]:
    """Evaluate and authenticate one exact subset of a sealed universe."""

    if (
        type(selected_action_sha256s) is not tuple
        or not selected_action_sha256s
        or selected_action_sha256s
        != tuple(sorted(set(selected_action_sha256s)))
    ):
        raise ValueError(
            "selected_action_sha256s must be non-empty and canonical"
        )
    identities = tuple(_expert_identity(value) for value in experts)
    expert_ids = tuple(value[0] for value in identities)
    if expert_ids != tuple(sorted(set(expert_ids))):
        raise ValueError("experts must use canonical unique IDs")
    if (
        type(proposals) is not tuple
        or tuple(value.expert_id for value in proposals) != expert_ids
    ):
        raise ValueError("proposals must exactly cover runtime experts")
    proposal_by_expert = {value.expert_id: value for value in proposals}
    action_by_sha256 = {
        action.action_sha256: action
        for proposal in proposals
        for action in proposal.actions
    }
    if not set(selected_action_sha256s).issubset(action_by_sha256):
        raise ValueError("selected action is outside the proposal universe")
    selected_by_expert = {
        expert_id: tuple(
            value
            for value in selected_action_sha256s
            if action_by_sha256[value].expert_id == expert_id
        )
        for expert_id in expert_ids
    }
    experts_by_id = {
        identity[0]: expert
        for identity, expert in zip(identities, experts, strict=True)
    }
    active = tuple(
        value for value in expert_ids if selected_by_expert[value]
    )

    async def evaluate_one(
        expert_id: str,
    ) -> MaterializedActionEvaluationBatch:
        expert = experts_by_id[expert_id]
        proposal = proposal_by_expert[expert_id]
        selected = selected_by_expert[expert_id]
        if not counterfactual:
            return await expert.evaluate(proposal, selected)
        if not isinstance(
            expert,
            MaterializedActionCounterfactualEvaluationPort,
        ):
            raise TypeError(
                "paired counterfactual evaluation requires the owning "
                "expert to implement "
                "MaterializedActionCounterfactualEvaluationPort"
            )
        return await expert.evaluate_counterfactual(proposal, selected)

    raw = await asyncio.gather(
        *(evaluate_one(expert_id) for expert_id in active)
    )
    batches = tuple(raw)
    for expert_id, batch in zip(active, batches, strict=True):
        if type(batch) is not MaterializedActionEvaluationBatch:
            raise TypeError("proposal expert returned a foreign evaluation batch")
        batch.__post_init__()
        batch.require_proposal(proposal_by_expert[expert_id])
        if batch.selected_action_sha256s != selected_by_expert[expert_id]:
            raise ValueError("expert evaluated a different selected subset")
        identity = _expert_identity(experts_by_id[expert_id])
        if (
            batch.expert_id,
            batch.expert_version,
            batch.expert_definition_sha256,
        ) != identity:
            raise ValueError("evaluation batch has a foreign expert identity")
    evaluated = tuple(
        sorted(
            action_sha256
            for batch in batches
            for action_sha256 in batch.selected_action_sha256s
        )
    )
    if evaluated != selected_action_sha256s:
        raise ValueError("evaluation batches do not close the selected subset")
    return batches


async def evaluate_materialized_action_subset(
    *,
    experts: tuple[MaterializedActionProposalExpertPort, ...],
    proposals: tuple[MaterializedActionProposalBatch, ...],
    selected_action_sha256s: tuple[str, ...],
) -> tuple[MaterializedActionEvaluationBatch, ...]:
    """Evaluate an authoritative subset of one sealed proposal universe."""

    return await _evaluate_materialized_action_subset(
        experts=experts,
        proposals=proposals,
        selected_action_sha256s=selected_action_sha256s,
        counterfactual=False,
    )


async def evaluate_materialized_action_counterfactual_subset(
    *,
    experts: tuple[MaterializedActionProposalExpertPort, ...],
    proposals: tuple[MaterializedActionProposalBatch, ...],
    selected_action_sha256s: tuple[str, ...],
) -> tuple[MaterializedActionEvaluationBatch, ...]:
    """Evaluate an excluded real arm without authoritative side effects."""

    return await _evaluate_materialized_action_subset(
        experts=experts,
        proposals=proposals,
        selected_action_sha256s=selected_action_sha256s,
        counterfactual=True,
    )


@dataclass(frozen=True, slots=True)
class ResidualPortfolioEvolutionResult:
    """Authenticated execution evidence before downstream archive credit."""

    request: ResidualPortfolioDecisionRequest
    proposals: tuple[MaterializedActionProposalBatch, ...]
    broker_decision: MaterializedActionBrokerDecision
    evaluation_batches: tuple[MaterializedActionEvaluationBatch, ...]
    slate_value_definition_sha256: str
    slate_feasibility_definition_sha256: str
    method_definition_sha256: str = (
        RESIDUAL_PORTFOLIO_EVOLUTION_DEFINITION_SHA256
    )
    result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        self.request.__post_init__()
        if type(self.proposals) is not tuple or not self.proposals:
            raise ValueError("result must contain proposal batches")
        for value in self.proposals:
            if type(value) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            value.__post_init__()
            value.require_request(self.request)
        proposal_ids = tuple(value.expert_id for value in self.proposals)
        if proposal_ids != tuple(sorted(set(proposal_ids))):
            raise ValueError("proposal batches must be expert-canonical")
        if type(self.broker_decision) is not MaterializedActionBrokerDecision:
            raise TypeError("broker_decision must be exact")
        self.broker_decision.__post_init__()
        if len(self.broker_decision.selected_actions) != self.request.evaluation_slots:
            raise ValueError("broker decision does not fill evaluation capacity")
        action_by_sha256 = {
            action.action_sha256: action
            for proposal in self.proposals
            for action in proposal.actions
        }
        if len(action_by_sha256) != sum(
            len(value.actions) for value in self.proposals
        ):
            raise ValueError("proposal union repeats an action identity")
        for action in self.broker_decision.selected_actions:
            if action_by_sha256.get(action.action_sha256) != action:
                raise ValueError("broker selected outside the proposal union")
        if type(self.evaluation_batches) is not tuple:
            raise TypeError("evaluation_batches must be an exact tuple")
        proposal_by_expert = {value.expert_id: value for value in self.proposals}
        for value in self.evaluation_batches:
            if type(value) is not MaterializedActionEvaluationBatch:
                raise TypeError("evaluation_batches must contain exact values")
            value.__post_init__()
            try:
                proposal = proposal_by_expert[value.expert_id]
            except KeyError as error:
                raise ValueError(
                    "evaluation names an absent proposal expert"
                ) from error
            value.require_proposal(proposal)
        evaluation_expert_ids = tuple(
            value.expert_id for value in self.evaluation_batches
        )
        if evaluation_expert_ids != tuple(sorted(set(evaluation_expert_ids))):
            raise ValueError("evaluation batches must be expert-canonical")
        selected = tuple(
            value.action_sha256 for value in self.broker_decision.selected_actions
        )
        evaluated = tuple(
            sorted(
                action_sha256
                for batch in self.evaluation_batches
                for action_sha256 in batch.selected_action_sha256s
            )
        )
        if tuple(sorted(selected)) != evaluated:
            raise ValueError("real evaluations do not exactly cover the broker slate")
        for name in (
            "slate_value_definition_sha256",
            "slate_feasibility_definition_sha256",
            "method_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "result_sha256",
            _hash(_RESULT_DOMAIN, self._unsigned_record()),
        )

    @property
    def evaluations(self) -> tuple[MaterializedActionEvaluation, ...]:
        by_action = {
            value.action.action_sha256: value
            for batch in self.evaluation_batches
            for value in batch.evaluations
        }
        return tuple(
            by_action[value.action_sha256]
            for value in self.broker_decision.selected_actions
        )

    @property
    def candidates(self) -> tuple[EvolutionCandidate, ...]:
        return tuple(value.candidate for value in self.evaluations)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": {
                "method_id": RESIDUAL_PORTFOLIO_EVOLUTION_ID,
                "method_version": RESIDUAL_PORTFOLIO_EVOLUTION_VERSION,
                "definition_sha256": self.method_definition_sha256,
            },
            "request_sha256": self.request.request_sha256,
            "proposal_sha256s": [value.proposal_sha256 for value in self.proposals],
            "broker_decision_sha256": self.broker_decision.decision_sha256,
            "evaluation_batch_sha256s": [
                value.batch_sha256 for value in self.evaluation_batches
            ],
            "selected_action_sha256s": [
                value.action_sha256
                for value in self.broker_decision.selected_actions
            ],
            "candidate_records": [
                _candidate_record(value) for value in self.candidates
            ],
            "slate_value_definition_sha256": (
                self.slate_value_definition_sha256
            ),
            "slate_feasibility_definition_sha256": (
                self.slate_feasibility_definition_sha256
            ),
            "archive_credit_included": False,
            "lineage_return_resolution_included": False,
        }

    def to_record(
        self,
        *,
        include_allocation_evidence: bool = False,
    ) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "broker_decision": self.broker_decision.to_record(
                include_allocation_evidence=include_allocation_evidence,
            ),
            "proposals": [value.to_record() for value in self.proposals],
            "evaluation_batches": [
                value.to_record() for value in self.evaluation_batches
            ],
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResidualPortfolioEvolution:
    """Execute one workload-blind residual expert market."""

    experts: tuple[MaterializedActionProposalExpertPort, ...]
    broker: RegretBrokeredMaterializedActionPolicy
    slate_value: MaterializedSlateValuePort
    slate_feasibility: MaterializedSlateFeasibilityPort
    allocation_policy: MaterializedActionAllocationPolicyPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.experts) is not tuple or not self.experts:
            raise ValueError("experts must be a non-empty exact tuple")
        identities = tuple(_expert_identity(value) for value in self.experts)
        expert_ids = tuple(value[0] for value in identities)
        if expert_ids != tuple(sorted(set(expert_ids))):
            raise ValueError("experts must use canonical unique IDs")
        if type(self.broker) is not RegretBrokeredMaterializedActionPolicy:
            raise TypeError("broker must be an exact regret-broker policy")
        self.broker.__post_init__()
        if not isinstance(self.slate_value, MaterializedSlateValuePort):
            raise TypeError("slate_value must implement its port")
        if not isinstance(self.slate_feasibility, MaterializedSlateFeasibilityPort):
            raise TypeError("slate_feasibility must implement its port")
        require_sha256(
            self.slate_value.definition_sha256,
            "slate value definition_sha256",
        )
        require_sha256(
            self.slate_feasibility.definition_sha256,
            "slate feasibility definition_sha256",
        )
        if self.allocation_policy is not None:
            _allocation_policy_identity(self.allocation_policy)

    async def run(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> ResidualPortfolioEvolutionResult:
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be an exact residual portfolio request")
        request.__post_init__()
        proposals = await propose_materialized_action_batches(
            experts=self.experts,
            request=request,
        )

        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )

        allocation_requirement = None
        if self.allocation_policy is not None:
            allocation_requirement = await self.allocation_policy.require(
                request,
                proposals,
            )
            if (
                type(allocation_requirement)
                is not MaterializedActionAllocationRequirement
            ):
                raise TypeError(
                    "allocation policy returned a foreign requirement"
                )
            allocation_requirement.__post_init__()
            policy_identity = _allocation_policy_identity(
                self.allocation_policy
            )
            if (
                allocation_requirement.policy_id,
                allocation_requirement.policy_version,
                allocation_requirement.policy_definition_sha256,
            ) != policy_identity:
                raise ValueError(
                    "allocation requirement differs from its policy"
                )
            if (
                allocation_requirement.residual_request_sha256
                != request.request_sha256
            ):
                raise ValueError(
                    "allocation requirement targets another residual request"
                )
            proposal_sha256s = tuple(
                sorted(value.proposal_sha256 for value in proposals)
            )
            if (
                allocation_requirement.proposal_sha256s
                != proposal_sha256s
            ):
                raise ValueError(
                    "allocation requirement targets another proposal universe"
                )
            action_by_sha256 = {
                value.action_sha256: value for value in actions
            }
            required_action_sha256s = (
                allocation_requirement.required_action_sha256s
            )
            if not set(required_action_sha256s).issubset(action_by_sha256):
                raise ValueError(
                    "allocation policy required an action outside the "
                    "sealed proposal union"
                )
            if len(required_action_sha256s) > request.evaluation_slots:
                raise ValueError(
                    "allocation requirement exceeds evaluation capacity"
                )
            required_phenotypes = {
                action_by_sha256[value].phenotype_identity_sha256
                for value in required_action_sha256s
            }
            if len(required_phenotypes) != len(required_action_sha256s):
                raise ValueError(
                    "allocation requirement repeats a materialized phenotype"
                )

        broker_decision = self.broker.select(
            MaterializedActionBrokerRequest(
                actions=actions,
                evaluation_slots=request.evaluation_slots,
                slate_value=self.slate_value,
                slate_feasibility=self.slate_feasibility,
                reference_escrow_slots=request.reference_escrow_slots,
                allocation_requirement=allocation_requirement,
            )
        )
        evaluation_batches = await evaluate_materialized_action_subset(
            experts=self.experts,
            proposals=proposals,
            selected_action_sha256s=tuple(
                value.action_sha256
                for value in broker_decision.selected_actions
            ),
        )

        return ResidualPortfolioEvolutionResult(
            request=request,
            proposals=proposals,
            broker_decision=broker_decision,
            evaluation_batches=evaluation_batches,
            slate_value_definition_sha256=self.slate_value.definition_sha256,
            slate_feasibility_definition_sha256=(
                self.slate_feasibility.definition_sha256
            ),
        )


__all__ = [
    "RESIDUAL_PORTFOLIO_EVOLUTION_DEFINITION_SHA256",
    "RESIDUAL_PORTFOLIO_EVOLUTION_ID",
    "RESIDUAL_PORTFOLIO_EVOLUTION_VERSION",
    "MaterializedActionEvaluation",
    "MaterializedActionEvaluationBatch",
    "MaterializedActionAllocationPolicyPort",
    "MaterializedActionCounterfactualEvaluationPort",
    "MaterializedActionProposalBatch",
    "MaterializedActionProposalExpertPort",
    "ResidualPortfolioDecisionRequest",
    "ResidualPortfolioEvolution",
    "ResidualPortfolioEvolutionResult",
    "evaluate_materialized_action_counterfactual_subset",
    "evaluate_materialized_action_subset",
    "propose_materialized_action_batches",
]

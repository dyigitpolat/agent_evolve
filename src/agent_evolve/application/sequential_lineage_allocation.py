"""Outcome-blind branch planning and an authenticated lineage pilot gate.

The planner freezes three requirements against one sealed proposal universe:

* a small recursive-lineage pilot;
* a complete slate whose ordinary score lanes exclude recursive actions; and
* a complete slate whose ordinary score lanes explicitly admit them.

Both complete slates exist before a pilot is evaluated.  The only
outcome-conditioned operation is a typed gate choosing one already-frozen
branch from authenticated pilot marginal utility.  No workload, objective,
model, provider, configuration field, or prompt content enters this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.candidate_archive_consequence import (
    CandidateArchiveConsequenceUtilityPort,
    validate_candidate_archive_consequence_utility,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionAllocationPolicyPort,
    MaterializedActionEvaluation,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


SEQUENTIAL_LINEAGE_PLANNER_ID = "sequential_lineage_branch_planner"
SEQUENTIAL_LINEAGE_PLANNER_VERSION = 1
ANY_POSITIVE_LINEAGE_GATE_ID = "any_positive_lineage_pilot_gate"
ANY_POSITIVE_LINEAGE_GATE_VERSION = 1
MARGINAL_PILOT_OUTCOME_PROJECTOR_ID = (
    "candidate_archive_marginal_pilot_outcome"
)
MARGINAL_PILOT_OUTCOME_PROJECTOR_VERSION = 1

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_PLAN_DOMAIN = b"agent-evolve:sequential-lineage-plan:v1\x00"
_PILOT_OUTCOME_DOMAIN = b"agent-evolve:lineage-pilot-outcome:v1\x00"
_PILOT_BATCH_DOMAIN = b"agent-evolve:lineage-pilot-outcome-batch:v1\x00"
_GATE_DOMAIN = b"agent-evolve:lineage-pilot-gate:v1\x00"
_PLANNER_DEFINITION_DOMAIN = (
    b"agent-evolve:sequential-lineage-planner-definition:v1\x00"
)
_PROJECTOR_DEFINITION_DOMAIN = (
    b"agent-evolve:marginal-pilot-projector-definition:v1\x00"
)
_PILOT_REQUIREMENT_DEFINITION_DOMAIN = (
    b"agent-evolve:sequential-lineage-pilot-requirement:v1\x00"
)


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


def _policy_identity(
    policy: MaterializedActionAllocationPolicyPort,
) -> tuple[str, int, str]:
    if not isinstance(policy, MaterializedActionAllocationPolicyPort):
        raise TypeError(
            "allocation policy must implement its application port"
        )
    identity = (
        getattr(policy, "policy_id", None),
        getattr(policy, "policy_version", None),
        getattr(policy, "definition_sha256", None),
    )
    _require_token(identity[0], name="allocation policy_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("allocation policy_version must be positive")
    require_sha256(identity[2], "allocation policy definition_sha256")
    return identity  # type: ignore[return-value]


def _requirement_trace(
    requirement: MaterializedActionAllocationRequirement,
) -> tuple[dict[str, object], ...]:
    requirement.__post_init__()
    record = thaw_json(requirement.evidence)
    if type(record) is not dict:
        raise TypeError("allocation evidence must have an object root")
    raw_trace = record.get("selection_trace")
    if type(raw_trace) is not list:
        raise ValueError("allocation evidence lacks a selection trace")
    result: list[dict[str, object]] = []
    for raw in raw_trace:
        if type(raw) is not dict:
            raise TypeError("selection trace rows must be objects")
        action_sha256 = raw.get("action_sha256")
        allocation_kind = raw.get("allocation_kind")
        if type(action_sha256) is not str or type(allocation_kind) is not str:
            raise ValueError("selection trace row is malformed")
        require_sha256(action_sha256, "trace action_sha256")
        result.append(dict(raw))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SequentialLineageAllocationPlan:
    """Two frozen complete slates plus their common lineage pilot."""

    planner_id: str
    planner_version: int
    planner_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    evaluation_slots: int
    pilot_requirement: MaterializedActionAllocationRequirement
    locked_requirement: MaterializedActionAllocationRequirement
    unlocked_requirement: MaterializedActionAllocationRequirement
    candidate_outcomes_observed: bool
    evidence: FrozenJsonObject
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.planner_id, name="planner_id")
        if type(self.planner_version) is not int or self.planner_version <= 0:
            raise ValueError("planner_version must be positive")
        require_sha256(
            self.planner_definition_sha256,
            "planner_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if (
            type(self.proposal_sha256s) is not tuple
            or not self.proposal_sha256s
            or self.proposal_sha256s
            != tuple(sorted(set(self.proposal_sha256s)))
        ):
            raise ValueError("proposal_sha256s must be non-empty and canonical")
        for value in self.proposal_sha256s:
            require_sha256(value, "proposal_sha256")
        if type(self.evaluation_slots) is not int or self.evaluation_slots <= 0:
            raise ValueError("evaluation_slots must be positive")
        for requirement in (
            self.pilot_requirement,
            self.locked_requirement,
            self.unlocked_requirement,
        ):
            if type(requirement) is not MaterializedActionAllocationRequirement:
                raise TypeError("plan requirements must be exact")
            requirement.__post_init__()
            if (
                requirement.residual_request_sha256
                != self.residual_request_sha256
                or requirement.proposal_sha256s != self.proposal_sha256s
            ):
                raise ValueError("plan requirement targets another universe")
        pilot = set(self.pilot_requirement.required_action_sha256s)
        locked = set(self.locked_requirement.required_action_sha256s)
        unlocked = set(self.unlocked_requirement.required_action_sha256s)
        if (
            len(locked) != self.evaluation_slots
            or len(unlocked) != self.evaluation_slots
        ):
            raise ValueError("both plan branches must fill evaluation capacity")
        if not pilot.issubset(locked) or not pilot.issubset(unlocked):
            raise ValueError("both complete branches must retain every pilot")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be exact")
        if self.candidate_outcomes_observed:
            raise ValueError("a sequential plan cannot observe candidate outcomes")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("plan evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "plan_sha256",
            _hash(_PLAN_DOMAIN, self._unsigned_record()),
        )

    @property
    def pilot_action_sha256s(self) -> tuple[str, ...]:
        return self.pilot_requirement.required_action_sha256s

    @property
    def frozen_branch_ids(self) -> tuple[str, ...]:
        return tuple(value.value for value in SequentialLineageBranch)

    @property
    def frozen_requirements(
        self,
    ) -> tuple[MaterializedActionAllocationRequirement, ...]:
        return (self.locked_requirement, self.unlocked_requirement)

    def requirement_for(
        self,
        branch_id: str,
    ) -> MaterializedActionAllocationRequirement:
        _require_token(branch_id, name="branch_id")
        try:
            branch = SequentialLineageBranch(branch_id)
        except ValueError as error:
            raise ValueError("branch_id is not frozen in this plan") from error
        return self.branch_requirement(branch)

    def branch_requirement(
        self,
        branch: "SequentialLineageBranch",
    ) -> MaterializedActionAllocationRequirement:
        if type(branch) is not SequentialLineageBranch:
            raise TypeError("branch must be an exact SequentialLineageBranch")
        return (
            self.unlocked_requirement
            if branch is SequentialLineageBranch.UNLOCKED
            else self.locked_requirement
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "planner": {
                "planner_id": self.planner_id,
                "planner_version": self.planner_version,
                "definition_sha256": self.planner_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "evaluation_slots": self.evaluation_slots,
            "pilot_requirement_sha256": (
                self.pilot_requirement.requirement_sha256
            ),
            "locked_requirement_sha256": (
                self.locked_requirement.requirement_sha256
            ),
            "unlocked_requirement_sha256": (
                self.unlocked_requirement.requirement_sha256
            ),
            "candidate_outcomes_observed": self.candidate_outcomes_observed,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "pilot_requirement": self.pilot_requirement.to_record(
                include_evidence=include_evidence
            ),
            "locked_requirement": self.locked_requirement.to_record(
                include_evidence=include_evidence
            ),
            "unlocked_requirement": self.unlocked_requirement.to_record(
                include_evidence=include_evidence
            ),
            "plan_sha256": self.plan_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class SequentialAllocationPlanPort(Protocol):
    """A pilot plus one or more complete slates frozen before outcomes."""

    planner_id: str
    planner_version: int
    planner_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    evaluation_slots: int
    pilot_requirement: MaterializedActionAllocationRequirement
    candidate_outcomes_observed: bool
    evidence: FrozenJsonObject
    plan_sha256: str

    @property
    def pilot_action_sha256s(self) -> tuple[str, ...]: ...

    @property
    def frozen_branch_ids(self) -> tuple[str, ...]: ...

    @property
    def frozen_requirements(
        self,
    ) -> tuple[MaterializedActionAllocationRequirement, ...]: ...

    def requirement_for(
        self,
        branch_id: str,
    ) -> MaterializedActionAllocationRequirement: ...

    def to_record(
        self,
        *,
        include_evidence: bool = False,
    ) -> dict[str, object]: ...


@runtime_checkable
class SequentialLineageAllocationPlannerPort(Protocol):
    """Freeze a pilot and complete slates before current outcomes exist."""

    planner_id: str
    planner_version: int
    definition_sha256: str

    async def plan(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> SequentialAllocationPlanPort: ...


@dataclass(frozen=True, slots=True)
class FrozenBranchSequentialLineagePlanner:
    """Compose locked and unlocked outcome-blind allocation policies."""

    locked_policy: MaterializedActionAllocationPolicyPort = field(
        repr=False,
        compare=False,
    )
    unlocked_policy: MaterializedActionAllocationPolicyPort = field(
        repr=False,
        compare=False,
    )
    pilot_allocation_kind: str = "recursive_lineage_cell"
    planner_id: str = SEQUENTIAL_LINEAGE_PLANNER_ID
    planner_version: int = SEQUENTIAL_LINEAGE_PLANNER_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        locked = _policy_identity(self.locked_policy)
        unlocked = _policy_identity(self.unlocked_policy)
        _require_token(self.pilot_allocation_kind, name="pilot_allocation_kind")
        _require_token(self.planner_id, name="planner_id")
        if self.planner_version != SEQUENTIAL_LINEAGE_PLANNER_VERSION:
            raise ValueError("planner_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _PLANNER_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "planner_id": self.planner_id,
                    "planner_version": self.planner_version,
                    "locked_policy": {
                        "policy_id": locked[0],
                        "policy_version": locked[1],
                        "definition_sha256": locked[2],
                    },
                    "unlocked_policy": {
                        "policy_id": unlocked[0],
                        "policy_version": unlocked[1],
                        "definition_sha256": unlocked[2],
                    },
                    "pilot_allocation_kind": self.pilot_allocation_kind,
                    "branches_frozen_before_outcomes": True,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    async def plan(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> SequentialLineageAllocationPlan:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
        locked, unlocked = await asyncio.gather(
            self.locked_policy.require(request, proposals),
            self.unlocked_policy.require(request, proposals),
        )
        for requirement in (locked, unlocked):
            if type(requirement) is not MaterializedActionAllocationRequirement:
                raise TypeError("branch policy returned a foreign requirement")
            requirement.__post_init__()
        locked_trace = _requirement_trace(locked)
        unlocked_trace = _requirement_trace(unlocked)
        locked_pilot = tuple(
            str(value["action_sha256"])
            for value in locked_trace
            if value["allocation_kind"] == self.pilot_allocation_kind
        )
        unlocked_pilot = tuple(
            str(value["action_sha256"])
            for value in unlocked_trace
            if value["allocation_kind"] == self.pilot_allocation_kind
        )
        if locked_pilot != unlocked_pilot:
            raise ValueError("branch policies disagree on the frozen pilot")
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        pilot_definition_sha256 = _hash(
            _PILOT_REQUIREMENT_DEFINITION_DOMAIN,
            {
                "planner_definition_sha256": self.definition_sha256,
                "pilot_allocation_kind": self.pilot_allocation_kind,
            },
        )
        pilot_requirement = MaterializedActionAllocationRequirement(
            policy_id="sequential_lineage_pilot",
            policy_version=1,
            policy_definition_sha256=pilot_definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=tuple(sorted(locked_pilot)),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "planner_definition_sha256": self.definition_sha256,
                    "locked_requirement_sha256": locked.requirement_sha256,
                    "unlocked_requirement_sha256": unlocked.requirement_sha256,
                    "pilot_trace": [
                        value
                        for value in locked_trace
                        if value["allocation_kind"]
                        == self.pilot_allocation_kind
                    ],
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )
        locked_identity = _policy_identity(self.locked_policy)
        unlocked_identity = _policy_identity(self.unlocked_policy)
        return SequentialLineageAllocationPlan(
            planner_id=self.planner_id,
            planner_version=self.planner_version,
            planner_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            evaluation_slots=request.evaluation_slots,
            pilot_requirement=pilot_requirement,
            locked_requirement=locked,
            unlocked_requirement=unlocked,
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "locked_policy": {
                        "policy_id": locked_identity[0],
                        "policy_version": locked_identity[1],
                        "definition_sha256": locked_identity[2],
                        "requirement_sha256": locked.requirement_sha256,
                    },
                    "unlocked_policy": {
                        "policy_id": unlocked_identity[0],
                        "policy_version": unlocked_identity[1],
                        "definition_sha256": unlocked_identity[2],
                        "requirement_sha256": unlocked.requirement_sha256,
                    },
                    "pilot_action_sha256s": list(sorted(locked_pilot)),
                    "both_complete_branches_frozen_before_pilot": True,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SequentialPilotActionOutcome:
    """One pilot's real marginal archive contribution against the prior."""

    action_sha256: str
    evaluation_sha256: str
    feasible: bool
    marginal_archive_gain: float
    positive_marginal_utility: bool
    outcome_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        require_sha256(self.evaluation_sha256, "evaluation_sha256")
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be an exact bool")
        if (
            type(self.marginal_archive_gain) is not float
            or not math.isfinite(self.marginal_archive_gain)
            or self.marginal_archive_gain < 0.0
        ):
            raise ValueError(
                "marginal_archive_gain must be finite and non-negative"
            )
        if type(self.positive_marginal_utility) is not bool:
            raise TypeError("positive_marginal_utility must be an exact bool")
        if self.positive_marginal_utility != (
            self.marginal_archive_gain > 0.0
        ):
            raise ValueError("positive verdict differs from marginal gain")
        if not self.feasible and (
            self.marginal_archive_gain != 0.0
            or self.positive_marginal_utility
        ):
            raise ValueError("an infeasible pilot cannot have positive return")
        object.__setattr__(
            self,
            "outcome_sha256",
            _hash(_PILOT_OUTCOME_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "feasible": self.feasible,
            "marginal_archive_gain_hex": self.marginal_archive_gain.hex(),
            "positive_marginal_utility": self.positive_marginal_utility,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "outcome_sha256": self.outcome_sha256,
        }


@dataclass(frozen=True, slots=True)
class SequentialPilotOutcomeBatch:
    """Authenticated complete outcomes for one plan's pilot action set."""

    projector_id: str
    projector_version: int
    projector_definition_sha256: str
    plan_sha256: str
    residual_request_sha256: str
    outcomes: tuple[SequentialPilotActionOutcome, ...]
    evidence: FrozenJsonObject
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.projector_id, name="projector_id")
        if type(self.projector_version) is not int or self.projector_version <= 0:
            raise ValueError("projector_version must be positive")
        require_sha256(
            self.projector_definition_sha256,
            "projector_definition_sha256",
        )
        require_sha256(self.plan_sha256, "plan_sha256")
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if type(self.outcomes) is not tuple:
            raise TypeError("outcomes must be an exact tuple")
        for value in self.outcomes:
            if type(value) is not SequentialPilotActionOutcome:
                raise TypeError("outcomes must contain exact values")
            value.__post_init__()
        action_ids = tuple(value.action_sha256 for value in self.outcomes)
        if action_ids != tuple(sorted(set(action_ids))):
            raise ValueError("pilot outcomes must be action-canonical")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("outcome evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "batch_sha256",
            _hash(_PILOT_BATCH_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "projector": {
                "projector_id": self.projector_id,
                "projector_version": self.projector_version,
                "definition_sha256": self.projector_definition_sha256,
            },
            "plan_sha256": self.plan_sha256,
            "residual_request_sha256": self.residual_request_sha256,
            "outcomes": [value.to_record() for value in self.outcomes],
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "batch_sha256": self.batch_sha256}
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class SequentialPilotOutcomeProjectorPort(Protocol):
    """Project only the real pilot evidence needed by a branch gate."""

    projector_id: str
    projector_version: int
    definition_sha256: str

    def project(
        self,
        plan: SequentialAllocationPlanPort,
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> SequentialPilotOutcomeBatch: ...


@dataclass(frozen=True, slots=True)
class CandidateArchiveMarginalPilotOutcomeProjector:
    """Use an injected archive utility to adjudicate pilot positivity."""

    prior_candidates: tuple[EvolutionCandidate, ...]
    utility: CandidateArchiveConsequenceUtilityPort = field(
        repr=False,
        compare=False,
    )
    projector_id: str = MARGINAL_PILOT_OUTCOME_PROJECTOR_ID
    projector_version: int = MARGINAL_PILOT_OUTCOME_PROJECTOR_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.prior_candidates) is not tuple
            or any(
                type(value) is not EvolutionCandidate
                for value in self.prior_candidates
            )
        ):
            raise TypeError("prior_candidates must be an exact candidate tuple")
        for value in self.prior_candidates:
            value.__post_init__()
        utility_identity = validate_candidate_archive_consequence_utility(
            self.utility
        )
        _require_token(self.projector_id, name="projector_id")
        if self.projector_version != MARGINAL_PILOT_OUTCOME_PROJECTOR_VERSION:
            raise ValueError("projector_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _PROJECTOR_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_id": self.projector_id,
                    "projector_version": self.projector_version,
                    "utility": {
                        "utility_id": utility_identity[0],
                        "utility_version": utility_identity[1],
                        "definition_sha256": utility_identity[2],
                    },
                    "prior_candidate_ids": [
                        value.candidate_id.value
                        for value in self.prior_candidates
                    ],
                    "admission": (
                        "valid_and_operator_and_evidence_compliant"
                    ),
                    "gate_currency": "positive_individual_marginal_utility",
                },
            ),
        )

    def project(
        self,
        plan: SequentialAllocationPlanPort,
        evaluations: tuple[MaterializedActionEvaluation, ...],
    ) -> SequentialPilotOutcomeBatch:
        self.__post_init__()
        if not isinstance(plan, SequentialAllocationPlanPort):
            raise TypeError("plan must implement its application port")
        if (
            type(evaluations) is not tuple
            or any(
                type(value) is not MaterializedActionEvaluation
                for value in evaluations
            )
        ):
            raise TypeError("evaluations must be an exact tuple")
        by_action: dict[str, MaterializedActionEvaluation] = {}
        for value in evaluations:
            value.__post_init__()
            if value.action.action_sha256 in by_action:
                raise ValueError("pilot evaluations repeat an action")
            by_action[value.action.action_sha256] = value
        if set(by_action) != set(plan.pilot_action_sha256s):
            raise ValueError("evaluations do not exactly cover the pilot")
        outcomes: list[SequentialPilotActionOutcome] = []
        for action_sha256 in sorted(by_action):
            evaluation = by_action[action_sha256]
            candidate = evaluation.candidate
            feasible = bool(
                candidate.valid
                and candidate.operator_compliant
                and candidate.evidence_compliant
            )
            gain = (
                0.0
                if not feasible
                else self.utility.marginal_utility(
                    self.prior_candidates,
                    candidate.objective_map,
                )
            )
            if (
                type(gain) is not float
                or not math.isfinite(gain)
                or gain < 0.0
            ):
                raise ValueError(
                    "pilot utility returned a non-finite or negative gain"
                )
            outcomes.append(
                SequentialPilotActionOutcome(
                    action_sha256=action_sha256,
                    evaluation_sha256=evaluation.evaluation_sha256,
                    feasible=feasible,
                    marginal_archive_gain=float(gain),
                    positive_marginal_utility=gain > 0.0,
                )
            )
        return SequentialPilotOutcomeBatch(
            projector_id=self.projector_id,
            projector_version=self.projector_version,
            projector_definition_sha256=self.definition_sha256,
            plan_sha256=plan.plan_sha256,
            residual_request_sha256=plan.residual_request_sha256,
            outcomes=tuple(outcomes),
            evidence=freeze_json(
                {
                    "prior_candidate_count": len(self.prior_candidates),
                    "utility_definition_sha256": (
                        self.utility.definition_sha256
                    ),
                    "individual_marginal_against_common_prior": True,
                    "only_pilot_outcomes_observed": True,
                    "workload_model_provider_branches": False,
                }
            ),
        )


class SequentialLineageBranch(str, Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"


@dataclass(frozen=True, slots=True)
class SequentialLineageGateDecision:
    """Outcome-conditioned choice between two pre-existing branch receipts."""

    gate_id: str
    gate_version: int
    gate_definition_sha256: str
    plan_sha256: str
    pilot_outcome_batch_sha256: str
    branch: SequentialLineageBranch
    selected_requirement_sha256: str
    positive_pilot_count: int
    pilot_count: int
    evidence: FrozenJsonObject
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.gate_id, name="gate_id")
        if type(self.gate_version) is not int or self.gate_version <= 0:
            raise ValueError("gate_version must be positive")
        require_sha256(
            self.gate_definition_sha256,
            "gate_definition_sha256",
        )
        for value, name in (
            (self.plan_sha256, "plan_sha256"),
            (
                self.pilot_outcome_batch_sha256,
                "pilot_outcome_batch_sha256",
            ),
            (
                self.selected_requirement_sha256,
                "selected_requirement_sha256",
            ),
        ):
            require_sha256(value, name)
        if type(self.branch) is not SequentialLineageBranch:
            raise TypeError("branch must be an exact SequentialLineageBranch")
        if (
            type(self.positive_pilot_count) is not int
            or type(self.pilot_count) is not int
            or not 0
            <= self.positive_pilot_count
            <= self.pilot_count
        ):
            raise ValueError("pilot counts are inconsistent")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("gate evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_GATE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "gate": {
                "gate_id": self.gate_id,
                "gate_version": self.gate_version,
                "definition_sha256": self.gate_definition_sha256,
            },
            "plan_sha256": self.plan_sha256,
            "pilot_outcome_batch_sha256": (
                self.pilot_outcome_batch_sha256
            ),
            "branch": self.branch.value,
            "selected_requirement_sha256": (
                self.selected_requirement_sha256
            ),
            "positive_pilot_count": self.positive_pilot_count,
            "pilot_count": self.pilot_count,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    @property
    def selected_branch_id(self) -> str:
        return self.branch.value

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "decision_sha256": self.decision_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class SequentialAllocationGateDecisionPort(Protocol):
    """Authenticated selection of one branch frozen in a sequential plan."""

    gate_id: str
    gate_version: int
    gate_definition_sha256: str
    plan_sha256: str
    pilot_outcome_batch_sha256: str
    selected_requirement_sha256: str
    positive_pilot_count: int
    pilot_count: int
    evidence: FrozenJsonObject
    decision_sha256: str

    @property
    def selected_branch_id(self) -> str: ...

    def to_record(
        self,
        *,
        include_evidence: bool = False,
    ) -> dict[str, object]: ...


@runtime_checkable
class SequentialAllocationGatePort(Protocol):
    """Choose one already-frozen complete slate from pilot evidence."""

    gate_id: str
    gate_version: int
    definition_sha256: str

    def decide(
        self,
        plan: SequentialAllocationPlanPort,
        outcomes: SequentialPilotOutcomeBatch,
    ) -> SequentialAllocationGateDecisionPort: ...


@dataclass(frozen=True, slots=True)
class AnyPositiveSequentialLineageGate:
    """Unlock recursive score-lane competition after any positive pilot."""

    gate_id: str = ANY_POSITIVE_LINEAGE_GATE_ID
    gate_version: int = ANY_POSITIVE_LINEAGE_GATE_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.gate_id, name="gate_id")
        if self.gate_version != ANY_POSITIVE_LINEAGE_GATE_VERSION:
            raise ValueError("gate_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                b"agent-evolve:any-positive-lineage-pilot-gate:v1;"
                b"unlock=any-pilot-positive-individual-marginal-utility;"
                b"zero-pilot=locked;branches-frozen-before-outcomes=true;"
                b"workload-model-provider-branches=false"
            ).hexdigest(),
        )

    def decide(
        self,
        plan: SequentialLineageAllocationPlan,
        outcomes: SequentialPilotOutcomeBatch,
    ) -> SequentialLineageGateDecision:
        self.__post_init__()
        if type(plan) is not SequentialLineageAllocationPlan:
            raise TypeError("plan must be exact")
        plan.__post_init__()
        if type(outcomes) is not SequentialPilotOutcomeBatch:
            raise TypeError("outcomes must be exact")
        outcomes.__post_init__()
        if (
            outcomes.plan_sha256 != plan.plan_sha256
            or outcomes.residual_request_sha256
            != plan.residual_request_sha256
            or tuple(value.action_sha256 for value in outcomes.outcomes)
            != plan.pilot_action_sha256s
        ):
            raise ValueError("pilot outcomes differ from their plan")
        positive_count = sum(
            value.positive_marginal_utility for value in outcomes.outcomes
        )
        branch = (
            SequentialLineageBranch.UNLOCKED
            if positive_count > 0
            else SequentialLineageBranch.LOCKED
        )
        selected = plan.branch_requirement(branch)
        return SequentialLineageGateDecision(
            gate_id=self.gate_id,
            gate_version=self.gate_version,
            gate_definition_sha256=self.definition_sha256,
            plan_sha256=plan.plan_sha256,
            pilot_outcome_batch_sha256=outcomes.batch_sha256,
            branch=branch,
            selected_requirement_sha256=selected.requirement_sha256,
            positive_pilot_count=positive_count,
            pilot_count=len(outcomes.outcomes),
            evidence=freeze_json(
                {
                    "rule": "unlock_if_any_pilot_positive",
                    "only_pilot_outcomes_observed": True,
                    "selected_branch_frozen_before_pilot": True,
                    "selected_requirement_sha256": (
                        selected.requirement_sha256
                    ),
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "ANY_POSITIVE_LINEAGE_GATE_ID",
    "ANY_POSITIVE_LINEAGE_GATE_VERSION",
    "AnyPositiveSequentialLineageGate",
    "CandidateArchiveMarginalPilotOutcomeProjector",
    "FrozenBranchSequentialLineagePlanner",
    "MARGINAL_PILOT_OUTCOME_PROJECTOR_ID",
    "MARGINAL_PILOT_OUTCOME_PROJECTOR_VERSION",
    "SEQUENTIAL_LINEAGE_PLANNER_ID",
    "SEQUENTIAL_LINEAGE_PLANNER_VERSION",
    "SequentialAllocationGateDecisionPort",
    "SequentialAllocationGatePort",
    "SequentialAllocationPlanPort",
    "SequentialLineageAllocationPlan",
    "SequentialLineageAllocationPlannerPort",
    "SequentialLineageBranch",
    "SequentialLineageGateDecision",
    "SequentialPilotActionOutcome",
    "SequentialPilotOutcomeBatch",
    "SequentialPilotOutcomeProjectorPort",
]

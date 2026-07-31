"""Outcome-safe racing among complete, precommitted portfolio continuations.

The module generalizes the binary lineage gate without knowing anything about
workloads, objectives, models, providers, prompts, or configuration fields.
Allocation policies first construct several complete slates from one sealed
proposal market.  A small, lane-stratified pilot is then forced into every
slate before any current outcome exists.  Real pilot consequences update only
the predeclared lane evidence, and a deterministic gate selects one already
frozen completion.

This is deliberately a *portfolio* race rather than candidate-wise Bayesian
optimization.  It preserves exact-K evaluation accounting, avoids
outcome-conditioned regeneration, and leaves the authoritative evaluator and
archive utility behind existing injected ports.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionDescriptor,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionAllocationPolicyPort,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.sequential_lineage_allocation import (
    SequentialAllocationGateDecisionPort,
    SequentialAllocationPlanPort,
    SequentialPilotOutcomeBatch,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

PRECOMMITTED_PORTFOLIO_RACE_PLANNER_ID = (
    "precommitted_portfolio_race_planner"
)
PRECOMMITTED_PORTFOLIO_RACE_PLANNER_VERSION = 4
EVIDENCE_ADAPTIVE_PORTFOLIO_RACE_GATE_ID = (
    "evidence_adaptive_portfolio_race_gate"
)
EVIDENCE_ADAPTIVE_PORTFOLIO_RACE_GATE_VERSION = 2
TRACE_FIELD_PORTFOLIO_LANE_PROJECTION_ID = (
    "trace_field_portfolio_lane_projection"
)
TRACE_FIELD_PORTFOLIO_LANE_PROJECTION_VERSION = 2
PHASE_CONDITIONED_PORTFOLIO_RACE_PRIOR_ID = (
    "phase_conditioned_portfolio_race_prior"
)
PHASE_CONDITIONED_PORTFOLIO_RACE_PRIOR_VERSION = 1
SYMMETRIC_DIFFERENCE_PORTFOLIO_RACE_POLICY_ID = (
    "symmetric_difference_portfolio_race_policy"
)
SYMMETRIC_DIFFERENCE_PORTFOLIO_RACE_POLICY_VERSION = 2

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_BRANCH_DOMAIN = b"agent-evolve:precommitted-race-branch:v1\x00"
_PLAN_DOMAIN = b"agent-evolve:precommitted-portfolio-race-plan:v1\x00"
_PLANNER_DEFINITION_DOMAIN = (
    b"agent-evolve:precommitted-portfolio-race-planner-definition:v1\x00"
)
_LANE_PROJECTION_DEFINITION_DOMAIN = (
    b"agent-evolve:portfolio-race-lane-projection-definition:v1\x00"
)
_PRIOR_PROJECTION_DEFINITION_DOMAIN = (
    b"agent-evolve:portfolio-race-prior-projection-definition:v1\x00"
)
_GATE_DEFINITION_DOMAIN = (
    b"agent-evolve:evidence-adaptive-portfolio-race-gate-definition:v1\x00"
)
_GATE_DECISION_DOMAIN = (
    b"agent-evolve:evidence-adaptive-portfolio-race-decision:v1\x00"
)
_PILOT_REQUIREMENT_DEFINITION_DOMAIN = (
    b"agent-evolve:precommitted-portfolio-race-pilot:v1\x00"
)
_BRANCH_REQUIREMENT_DEFINITION_DOMAIN = (
    b"agent-evolve:precommitted-portfolio-race-branch-requirement:v1\x00"
)
_DISAGREEMENT_POLICY_DEFINITION_DOMAIN = (
    b"agent-evolve:portfolio-race-disagreement-policy-definition:v1\x00"
)
_DISAGREEMENT_DESIGN_DOMAIN = (
    b"agent-evolve:portfolio-race-disagreement-design:v1\x00"
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
        raise TypeError("allocation policy must implement its application port")
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
    evidence = thaw_json(requirement.evidence)
    if type(evidence) is not dict:
        raise TypeError("allocation evidence must have an object root")
    raw_trace = evidence.get("selection_trace")
    if type(raw_trace) is not list:
        raise ValueError("allocation evidence lacks a selection trace")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for fallback_ordinal, raw in enumerate(raw_trace, start=1):
        if type(raw) is not dict:
            raise TypeError("selection trace rows must be objects")
        row = dict(raw)
        action_sha256 = row.get("action_sha256")
        if type(action_sha256) is not str:
            raise TypeError("selection trace omits action_sha256")
        require_sha256(action_sha256, "selection trace action_sha256")
        if action_sha256 in seen:
            raise ValueError("selection trace repeats an action")
        seen.add(action_sha256)
        ordinal = row.get("ordinal", fallback_ordinal)
        if type(ordinal) is not int or ordinal <= 0:
            raise ValueError("selection trace ordinal must be positive")
        row["ordinal"] = ordinal
        rows.append(row)
    required = set(requirement.required_action_sha256s)
    if seen != required:
        raise ValueError(
            "selection trace must exactly cover required action identities"
        )
    return tuple(
        sorted(
            rows,
            key=lambda value: (
                int(value["ordinal"]),
                str(value["action_sha256"]),
            ),
        )
    )


@runtime_checkable
class PortfolioLaneProjectionPort(Protocol):
    """Project an outcome-blind trace/action pair into a portable lane."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    def project(
        self,
        trace_row: dict[str, object],
        action: MaterializedActionDescriptor,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TraceFieldPortfolioLaneProjection:
    """Use portable trace fields, falling back to source/operator identity."""

    trace_fields: tuple[str, ...] = (
        "score_lane",
        "allocation_kind",
    )
    include_expert_id: bool = True
    include_operator_id: bool = False
    projection_id: str = TRACE_FIELD_PORTFOLIO_LANE_PROJECTION_ID
    projection_version: int = (
        TRACE_FIELD_PORTFOLIO_LANE_PROJECTION_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.trace_fields) is not tuple
            or not self.trace_fields
            or self.trace_fields != tuple(dict.fromkeys(self.trace_fields))
        ):
            raise ValueError(
                "trace_fields must be a non-empty ordered unique tuple"
            )
        for value in self.trace_fields:
            _require_token(value, name="trace field")
        if type(self.include_expert_id) is not bool:
            raise TypeError("include_expert_id must be exact")
        if type(self.include_operator_id) is not bool:
            raise TypeError("include_operator_id must be exact")
        _require_token(self.projection_id, name="projection_id")
        if (
            self.projection_version
            != TRACE_FIELD_PORTFOLIO_LANE_PROJECTION_VERSION
        ):
            raise ValueError("projection_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _LANE_PROJECTION_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projection_id": self.projection_id,
                    "projection_version": self.projection_version,
                    "trace_fields": list(self.trace_fields),
                    "include_expert_id": self.include_expert_id,
                    "include_operator_id": self.include_operator_id,
                    "fallback": "allocation_kind_then_action_source",
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def project(
        self,
        trace_row: dict[str, object],
        action: MaterializedActionDescriptor,
    ) -> str:
        self.__post_init__()
        if type(trace_row) is not dict:
            raise TypeError("trace_row must be an exact dict")
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be exact")
        action.__post_init__()
        parts: list[str] = []
        for field_name in self.trace_fields:
            value = trace_row.get(field_name)
            if type(value) is str and _TOKEN.fullmatch(value) is not None:
                parts.append(f"{field_name}:{value}")
                break
        if not parts:
            parts.append(f"source:{action.expert_id}")
        if self.include_expert_id:
            parts.append(f"expert:{action.expert_id}")
        if self.include_operator_id:
            parts.append(f"operator:{action.operator_id}")
        lane_id = ".".join(parts)
        _require_token(lane_id, name="projected lane_id")
        return lane_id


def _lane_projection_identity(
    value: PortfolioLaneProjectionPort,
) -> tuple[str, int, str]:
    if not isinstance(value, PortfolioLaneProjectionPort):
        raise TypeError("lane projection must implement its application port")
    identity = (
        getattr(value, "projection_id", None),
        getattr(value, "projection_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="lane projection_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("lane projection_version must be positive")
    require_sha256(identity[2], "lane projection definition_sha256")
    return identity  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PortfolioRacePolicyBinding:
    """One named continuation policy and its outcome-blind prior mean."""

    branch_id: str
    policy: MaterializedActionAllocationPolicyPort = field(
        repr=False,
        compare=False,
    )
    prior_mean: float = 0.5

    def __post_init__(self) -> None:
        _require_token(self.branch_id, name="branch_id")
        _policy_identity(self.policy)
        if (
            type(self.prior_mean) is not float
            or not math.isfinite(self.prior_mean)
            or not 0.0 <= self.prior_mean <= 1.0
        ):
            raise ValueError("prior_mean must be finite and lie in [0, 1]")

    def identity_record(self) -> dict[str, object]:
        identity = _policy_identity(self.policy)
        return {
            "branch_id": self.branch_id,
            "policy_id": identity[0],
            "policy_version": identity[1],
            "policy_definition_sha256": identity[2],
            "prior_mean_hex": self.prior_mean.hex(),
        }


@runtime_checkable
class PortfolioRacePriorProjectionPort(Protocol):
    """Project frozen search context into one outcome-blind branch prior."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        binding: PortfolioRacePolicyBinding,
        source_requirement: MaterializedActionAllocationRequirement,
    ) -> float: ...


def _prior_projection_identity(
    value: PortfolioRacePriorProjectionPort,
) -> tuple[str, int, str]:
    if not isinstance(value, PortfolioRacePriorProjectionPort):
        raise TypeError(
            "prior projection must implement its application port"
        )
    identity = (
        getattr(value, "projection_id", None),
        getattr(value, "projection_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="prior projection_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("prior projection_version must be positive")
    require_sha256(identity[2], "prior projection definition_sha256")
    return identity  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PhaseConditionedPortfolioRacePrior:
    """Frozen method-branch priors indexed only by generic search phase.

    The core never knows workload, objective, model, provider, prompt, or
    configuration fields.  A training pipeline may fit these bounded means
    from earlier authenticated branch consequences and inject the resulting
    immutable table.  Missing cells retain each binding's declared prior.
    """

    prior_means: tuple[tuple[SearchPhase, str, float], ...]
    projection_id: str = PHASE_CONDITIONED_PORTFOLIO_RACE_PRIOR_ID
    projection_version: int = (
        PHASE_CONDITIONED_PORTFOLIO_RACE_PRIOR_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.prior_means) is not tuple or not self.prior_means:
            raise ValueError("prior_means must be a non-empty exact tuple")
        canonical = tuple(
            sorted(
                self.prior_means,
                key=lambda value: (value[0].value, value[1]),
            )
        )
        if self.prior_means != canonical:
            raise ValueError("phase-conditioned priors must be canonical")
        cells: list[tuple[str, str]] = []
        for phase, branch_id, mean in self.prior_means:
            if type(phase) is not SearchPhase:
                raise TypeError("prior phase must be an exact SearchPhase")
            _require_token(branch_id, name="prior branch_id")
            if (
                type(mean) is not float
                or not math.isfinite(mean)
                or not 0.0 <= mean <= 1.0
            ):
                raise ValueError("branch prior mean must lie in [0, 1]")
            cells.append((phase.value, branch_id))
        if len(cells) != len(set(cells)):
            raise ValueError("phase-conditioned prior cells must be unique")
        _require_token(self.projection_id, name="prior projection_id")
        if (
            self.projection_version
            != PHASE_CONDITIONED_PORTFOLIO_RACE_PRIOR_VERSION
        ):
            raise ValueError("prior projection_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _PRIOR_PROJECTION_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projection_id": self.projection_id,
                    "projection_version": self.projection_version,
                    "prior_means": [
                        {
                            "phase": phase.value,
                            "branch_id": branch_id,
                            "mean_hex": mean.hex(),
                        }
                        for phase, branch_id, mean in self.prior_means
                    ],
                    "missing_cell": "binding_prior_mean",
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        binding: PortfolioRacePolicyBinding,
        source_requirement: MaterializedActionAllocationRequirement,
    ) -> float:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(binding) is not PortfolioRacePolicyBinding:
            raise TypeError("binding must be exact")
        binding.__post_init__()
        if (
            type(source_requirement)
            is not MaterializedActionAllocationRequirement
        ):
            raise TypeError("source_requirement must be exact")
        source_requirement.__post_init__()
        if (
            source_requirement.residual_request_sha256
            != request.request_sha256
        ):
            raise ValueError("prior source targets another request")
        matches = tuple(
            mean
            for phase, branch_id, mean in self.prior_means
            if phase is request.phase and branch_id == binding.branch_id
        )
        if len(matches) > 1:
            raise AssertionError("validated prior table repeated a cell")
        return binding.prior_mean if not matches else matches[0]

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "projection_id": self.projection_id,
            "projection_version": self.projection_version,
            "definition_sha256": self.definition_sha256,
            "prior_means": [
                {
                    "phase": phase.value,
                    "branch_id": branch_id,
                    "mean_hex": mean.hex(),
                }
                for phase, branch_id, mean in self.prior_means
            ],
            "missing_cell": "binding_prior_mean",
            "candidate_outcomes_observed": False,
            "workload_model_provider_branches": False,
        }


@dataclass(frozen=True, slots=True)
class PortfolioRacePilotLaneBinding:
    action_sha256: str
    lane_id: str

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "pilot action_sha256")
        _require_token(self.lane_id, name="pilot lane_id")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "lane_id": self.lane_id,
        }


@dataclass(frozen=True, slots=True)
class FrozenPortfolioRaceBranch:
    """One complete slate and its pre-pilot lane exposure."""

    branch_id: str
    source_requirement_sha256: str
    requirement: MaterializedActionAllocationRequirement
    completion_lane_exposure: tuple[tuple[str, int], ...]
    prior_mean: float
    branch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.branch_id, name="branch_id")
        require_sha256(
            self.source_requirement_sha256,
            "source_requirement_sha256",
        )
        if type(self.requirement) is not MaterializedActionAllocationRequirement:
            raise TypeError("branch requirement must be exact")
        self.requirement.__post_init__()
        if (
            type(self.completion_lane_exposure) is not tuple
            or not self.completion_lane_exposure
            or self.completion_lane_exposure
            != tuple(sorted(self.completion_lane_exposure))
        ):
            raise ValueError(
                "completion_lane_exposure must be non-empty and canonical"
            )
        lane_ids = tuple(value[0] for value in self.completion_lane_exposure)
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("completion lane IDs must be unique")
        for lane_id, count in self.completion_lane_exposure:
            _require_token(lane_id, name="completion lane_id")
            if type(count) is not int or count <= 0:
                raise ValueError("completion lane counts must be positive")
        if (
            type(self.prior_mean) is not float
            or not math.isfinite(self.prior_mean)
            or not 0.0 <= self.prior_mean <= 1.0
        ):
            raise ValueError("prior_mean must be finite and lie in [0, 1]")
        object.__setattr__(
            self,
            "branch_sha256",
            _hash(_BRANCH_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "branch_id": self.branch_id,
            "source_requirement_sha256": self.source_requirement_sha256,
            "requirement_sha256": self.requirement.requirement_sha256,
            "completion_lane_exposure": [
                {"lane_id": lane_id, "count": count}
                for lane_id, count in self.completion_lane_exposure
            ],
            "prior_mean_hex": self.prior_mean.hex(),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "requirement": self.requirement.to_record(
                include_evidence=include_evidence
            ),
            "branch_sha256": self.branch_sha256,
        }


@dataclass(frozen=True, slots=True)
class PrecommittedPortfolioRacePlan:
    """A shared diagnostic pilot and N complete frozen continuations."""

    planner_id: str
    planner_version: int
    planner_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    evaluation_slots: int
    pilot_requirement: MaterializedActionAllocationRequirement
    pilot_lane_bindings: tuple[PortfolioRacePilotLaneBinding, ...]
    branches: tuple[FrozenPortfolioRaceBranch, ...]
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
        if type(self.pilot_requirement) is not MaterializedActionAllocationRequirement:
            raise TypeError("pilot_requirement must be exact")
        self.pilot_requirement.__post_init__()
        if (
            self.pilot_requirement.residual_request_sha256
            != self.residual_request_sha256
            or self.pilot_requirement.proposal_sha256s
            != self.proposal_sha256s
        ):
            raise ValueError("pilot requirement targets another proposal market")
        if (
            type(self.pilot_lane_bindings) is not tuple
            or tuple(
                value.action_sha256 for value in self.pilot_lane_bindings
            )
            != self.pilot_requirement.required_action_sha256s
        ):
            raise ValueError(
                "pilot lane bindings must exactly cover canonical pilot actions"
            )
        for value in self.pilot_lane_bindings:
            if type(value) is not PortfolioRacePilotLaneBinding:
                raise TypeError("pilot lane bindings must be exact")
            value.__post_init__()
        if (
            type(self.branches) is not tuple
            or not self.branches
            or tuple(value.branch_id for value in self.branches)
            != tuple(sorted({value.branch_id for value in self.branches}))
        ):
            raise ValueError(
                "branches must contain canonical unique IDs"
            )
        pilot = set(self.pilot_action_sha256s)
        for branch in self.branches:
            if type(branch) is not FrozenPortfolioRaceBranch:
                raise TypeError("branches must be exact")
            branch.__post_init__()
            requirement = branch.requirement
            if (
                requirement.residual_request_sha256
                != self.residual_request_sha256
                or requirement.proposal_sha256s != self.proposal_sha256s
                or len(requirement.required_action_sha256s)
                != self.evaluation_slots
                or not pilot.issubset(
                    requirement.required_action_sha256s
                )
            ):
                raise ValueError(
                    "every branch must be a complete pilot-preserving slate"
                )
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be exact")
        if self.candidate_outcomes_observed:
            raise ValueError("a race plan cannot observe current outcomes")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("race-plan evidence must be a frozen object")
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
        return tuple(value.branch_id for value in self.branches)

    @property
    def frozen_requirements(
        self,
    ) -> tuple[MaterializedActionAllocationRequirement, ...]:
        return tuple(value.requirement for value in self.branches)

    def requirement_for(
        self,
        branch_id: str,
    ) -> MaterializedActionAllocationRequirement:
        _require_token(branch_id, name="branch_id")
        matches = tuple(
            value.requirement
            for value in self.branches
            if value.branch_id == branch_id
        )
        if len(matches) != 1:
            raise ValueError("branch_id is not frozen in this plan")
        return matches[0]

    def branch_for(self, branch_id: str) -> FrozenPortfolioRaceBranch:
        _require_token(branch_id, name="branch_id")
        matches = tuple(
            value for value in self.branches if value.branch_id == branch_id
        )
        if len(matches) != 1:
            raise ValueError("branch_id is not frozen in this plan")
        return matches[0]

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
            "pilot_lane_bindings": [
                value.to_record() for value in self.pilot_lane_bindings
            ],
            "branch_sha256s": [
                value.branch_sha256 for value in self.branches
            ],
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
            "branches": [
                value.to_record(include_evidence=include_evidence)
                for value in self.branches
            ],
            "plan_sha256": self.plan_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


def _action_market(
    proposals: tuple[MaterializedActionProposalBatch, ...],
) -> dict[str, MaterializedActionDescriptor]:
    market: dict[str, MaterializedActionDescriptor] = {}
    candidate_ids: set[object] = set()
    for proposal in proposals:
        if type(proposal) is not MaterializedActionProposalBatch:
            raise TypeError("proposal market must contain exact batches")
        proposal.__post_init__()
        for action in proposal.actions:
            if action.action_sha256 in market:
                raise ValueError("proposal market repeats an action identity")
            if action.target_candidate_id in candidate_ids:
                raise ValueError("proposal market repeats a target candidate")
            market[action.action_sha256] = action
            candidate_ids.add(action.target_candidate_id)
    return market


def _trace_by_action(
    requirement: MaterializedActionAllocationRequirement,
) -> dict[str, dict[str, object]]:
    return {
        str(value["action_sha256"]): value
        for value in _requirement_trace(requirement)
    }


def _ordered_action_ids(
    requirement: MaterializedActionAllocationRequirement,
) -> tuple[str, ...]:
    return tuple(
        str(value["action_sha256"])
        for value in _requirement_trace(requirement)
    )


def _stratified_pilot_ids(
    *,
    requirement: MaterializedActionAllocationRequirement,
    market: dict[str, MaterializedActionDescriptor],
    lane_projection: PortfolioLaneProjectionPort,
    pilot_slots: int,
) -> tuple[tuple[str, str], ...]:
    trace = _requirement_trace(requirement)
    rows = [
        (
            str(value["action_sha256"]),
            lane_projection.project(
                value,
                market[str(value["action_sha256"])],
            ),
        )
        for value in trace
    ]
    by_lane: dict[str, list[str]] = {}
    for action_sha256, lane_id in rows:
        by_lane.setdefault(lane_id, []).append(action_sha256)
    lane_order = tuple(by_lane)
    selected: list[tuple[str, str]] = []
    selected_actions: set[str] = set()
    selected_phenotypes: set[str] = set()
    depth = 0
    while len(selected) < pilot_slots:
        added = False
        for lane_id in lane_order:
            members = by_lane[lane_id]
            if depth >= len(members):
                continue
            action_sha256 = members[depth]
            action = market[action_sha256]
            if (
                action_sha256 in selected_actions
                or action.phenotype_identity_sha256 in selected_phenotypes
            ):
                continue
            selected.append((action_sha256, lane_id))
            selected_actions.add(action_sha256)
            selected_phenotypes.add(action.phenotype_identity_sha256)
            added = True
            if len(selected) == pilot_slots:
                break
        if not added and all(depth + 1 >= len(by_lane[key]) for key in lane_order):
            break
        depth += 1
    if len(selected) != pilot_slots:
        raise ValueError("pilot policy cannot provide the requested unique slots")
    return tuple(sorted(selected))


def _branch_stratified_pilot_ids(
    *,
    branch_bindings: tuple[PortfolioRacePolicyBinding, ...],
    branch_requirements: tuple[
        MaterializedActionAllocationRequirement, ...
    ],
    market: dict[str, MaterializedActionDescriptor],
    pilot_slots: int,
) -> tuple[tuple[str, str], ...]:
    """Select branch-labelled diagnostics before composing shared slates."""

    if pilot_slots < len(branch_bindings):
        raise ValueError(
            "branch-stratified pilots require at least one slot per branch"
        )
    ordered_by_branch = {
        binding.branch_id: _ordered_action_ids(requirement)
        for binding, requirement in zip(
            branch_bindings,
            branch_requirements,
            strict=True,
        )
    }
    selected: list[tuple[str, str]] = []
    selected_actions: set[str] = set()
    selected_phenotypes: set[str] = set()
    depth = 0
    while len(selected) < pilot_slots:
        added = False
        for binding in branch_bindings:
            candidates = ordered_by_branch[binding.branch_id]
            candidate_index = depth
            while candidate_index < len(candidates):
                action_sha256 = candidates[candidate_index]
                action = market[action_sha256]
                candidate_index += 1
                if (
                    action_sha256 in selected_actions
                    or action.phenotype_identity_sha256
                    in selected_phenotypes
                ):
                    continue
                selected.append(
                    (action_sha256, f"branch:{binding.branch_id}")
                )
                selected_actions.add(action_sha256)
                selected_phenotypes.add(
                    action.phenotype_identity_sha256
                )
                added = True
                break
            if len(selected) == pilot_slots:
                break
        if not added:
            break
        depth += 1
    if len(selected) != pilot_slots:
        raise ValueError(
            "branch policies cannot provide the requested unique pilots"
        )
    return tuple(sorted(selected))


def _project_branch_pilot_lanes(
    *,
    pilot_pairs: tuple[tuple[str, str], ...],
    branch_bindings: tuple[PortfolioRacePolicyBinding, ...],
    branch_requirements: tuple[
        MaterializedActionAllocationRequirement, ...
    ],
    market: dict[str, MaterializedActionDescriptor],
    lane_projection: PortfolioLaneProjectionPort,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str, str], ...],
]:
    """Replace synthetic branch labels with source-trace exchangeability cells.

    Branch identity is useful for choosing disagreement pilots, but it is not
    evidence that every later action from that branch is exchangeable with the
    pilot.  The returned pilot bindings therefore use the same portable,
    outcome-blind trace projection as continuation actions.  A separate
    provenance tuple retains the originating branch without allowing it to
    masquerade as a statistical lane.
    """

    source_by_branch = {
        binding.branch_id: requirement
        for binding, requirement in zip(
            branch_bindings,
            branch_requirements,
            strict=True,
        )
    }
    trace_by_branch = {
        branch_id: _trace_by_action(requirement)
        for branch_id, requirement in source_by_branch.items()
    }
    projected: list[tuple[str, str]] = []
    provenance: list[tuple[str, str, str]] = []
    for action_sha256, source_lane in pilot_pairs:
        prefix = "branch:"
        if not source_lane.startswith(prefix):
            raise ValueError(
                "branch-stratified pilot lacks source-branch provenance"
            )
        branch_id = source_lane.removeprefix(prefix)
        if branch_id not in source_by_branch:
            raise ValueError("pilot names a branch outside the active race")
        row = trace_by_branch[branch_id].get(action_sha256)
        if row is None:
            raise ValueError(
                "branch-stratified pilot is absent from its source trace"
            )
        lane_id = lane_projection.project(
            row,
            market[action_sha256],
        )
        projected.append((action_sha256, lane_id))
        provenance.append((action_sha256, branch_id, lane_id))
    return (
        tuple(sorted(projected)),
        tuple(sorted(provenance)),
    )


def _compose_branch_action_ids(
    *,
    pilot_action_ids: tuple[str, ...],
    source_requirement: MaterializedActionAllocationRequirement,
    market: dict[str, MaterializedActionDescriptor],
    evaluation_slots: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    phenotypes: set[str] = set()

    def consider(action_sha256: str) -> None:
        if len(selected) >= evaluation_slots or action_sha256 in selected:
            return
        action = market[action_sha256]
        if action.phenotype_identity_sha256 in phenotypes:
            return
        selected.append(action_sha256)
        phenotypes.add(action.phenotype_identity_sha256)

    for value in pilot_action_ids:
        consider(value)
    for value in _ordered_action_ids(source_requirement):
        consider(value)
    if len(selected) < evaluation_slots:
        for action in sorted(
            market.values(),
            key=lambda value: (
                value.native_rank,
                value.expert_id,
                value.operator_id,
                value.action_sha256,
            ),
        ):
            consider(action.action_sha256)
    if len(selected) != evaluation_slots:
        raise ValueError("proposal market cannot fill a unique-phenotype branch")
    return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class PortfolioRaceDisagreementDesign:
    """Outcome-blind pilot design over behaviorally distinct complete slates."""

    pilot_pairs: tuple[tuple[str, str], ...]
    branch_action_sha256s: tuple[tuple[str, tuple[str, ...]], ...]
    evidence: FrozenJsonObject
    design_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.pilot_pairs) is not tuple
            or self.pilot_pairs
            != tuple(sorted(self.pilot_pairs))
        ):
            raise ValueError("pilot_pairs must be canonical")
        pilot_ids: list[str] = []
        for action_sha256, lane_id in self.pilot_pairs:
            require_sha256(action_sha256, "pilot action_sha256")
            _require_token(lane_id, name="pilot lane_id")
            pilot_ids.append(action_sha256)
        if len(pilot_ids) != len(set(pilot_ids)):
            raise ValueError("pilot_pairs repeat an action")
        if (
            type(self.branch_action_sha256s) is not tuple
            or not self.branch_action_sha256s
            or self.branch_action_sha256s
            != tuple(
                sorted(
                    self.branch_action_sha256s,
                    key=lambda value: value[0],
                )
            )
        ):
            raise ValueError(
                "branch_action_sha256s must be non-empty and canonical"
            )
        branch_ids: list[str] = []
        pilot_set = set(pilot_ids)
        for branch_id, action_sha256s in self.branch_action_sha256s:
            _require_token(branch_id, name="designed branch_id")
            branch_ids.append(branch_id)
            if (
                type(action_sha256s) is not tuple
                or not action_sha256s
                or action_sha256s
                != tuple(sorted(set(action_sha256s)))
            ):
                raise ValueError(
                    "designed branch actions must be non-empty and canonical"
                )
            for action_sha256 in action_sha256s:
                require_sha256(action_sha256, "designed action_sha256")
            if not pilot_set.issubset(action_sha256s):
                raise ValueError("every designed branch must retain each pilot")
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("designed branch IDs must be unique")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("design evidence must be a frozen object")
        object.__setattr__(
            self,
            "design_sha256",
            _hash(
                _DISAGREEMENT_DESIGN_DOMAIN,
                {
                    "schema_version": 1,
                    "pilot_pairs": [
                        {
                            "action_sha256": action_sha256,
                            "lane_id": lane_id,
                        }
                        for action_sha256, lane_id in self.pilot_pairs
                    ],
                    "branch_action_sha256s": [
                        {
                            "branch_id": branch_id,
                            "action_sha256s": list(action_sha256s),
                        }
                        for branch_id, action_sha256s
                        in self.branch_action_sha256s
                    ],
                    "evidence_sha256": typed_json_sha256(self.evidence),
                },
            ),
        )

    def actions_for(self, branch_id: str) -> tuple[str, ...]:
        _require_token(branch_id, name="branch_id")
        matches = tuple(
            action_sha256s
            for candidate_id, action_sha256s
            in self.branch_action_sha256s
            if candidate_id == branch_id
        )
        if len(matches) != 1:
            raise ValueError("branch is absent from disagreement design")
        return matches[0]

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "pilot_pairs": [
                {
                    "action_sha256": action_sha256,
                    "lane_id": lane_id,
                }
                for action_sha256, lane_id in self.pilot_pairs
            ],
            "branch_action_sha256s": [
                {
                    "branch_id": branch_id,
                    "action_sha256s": list(action_sha256s),
                }
                for branch_id, action_sha256s
                in self.branch_action_sha256s
            ],
            "evidence": thaw_json(self.evidence),
            "design_sha256": self.design_sha256,
        }


@runtime_checkable
class PortfolioRaceDisagreementPolicyPort(Protocol):
    """Design actionable pilots and distinct slates without current outcomes."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def design(
        self,
        *,
        branch_bindings: tuple[PortfolioRacePolicyBinding, ...],
        branch_requirements: tuple[
            MaterializedActionAllocationRequirement,
            ...,
        ],
        market: dict[str, MaterializedActionDescriptor],
        evaluation_slots: int,
        maximum_pilot_slots: int,
    ) -> PortfolioRaceDisagreementDesign: ...


def _disagreement_policy_identity(
    value: PortfolioRaceDisagreementPolicyPort,
) -> tuple[str, int, str]:
    if not isinstance(value, PortfolioRaceDisagreementPolicyPort):
        raise TypeError(
            "disagreement policy must implement its application port"
        )
    identity = (
        getattr(value, "policy_id", None),
        getattr(value, "policy_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="disagreement policy_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("disagreement policy_version must be positive")
    require_sha256(identity[2], "disagreement policy definition_sha256")
    return identity  # type: ignore[return-value]


def _representative_binding(
    bindings: tuple[PortfolioRacePolicyBinding, ...],
) -> PortfolioRacePolicyBinding:
    if not bindings:
        raise ValueError("an equivalence class needs a representative")
    return min(
        bindings,
        key=lambda value: (-value.prior_mean, value.branch_id),
    )


def _equivalence_classes(
    *,
    bindings: tuple[PortfolioRacePolicyBinding, ...],
    actions_by_branch: dict[str, tuple[str, ...]],
) -> tuple[
    tuple[
        PortfolioRacePolicyBinding,
        tuple[PortfolioRacePolicyBinding, ...],
    ],
    ...,
]:
    groups: dict[
        tuple[str, ...],
        list[PortfolioRacePolicyBinding],
    ] = {}
    for binding in bindings:
        groups.setdefault(
            actions_by_branch[binding.branch_id],
            [],
        ).append(binding)
    classes = []
    for members in groups.values():
        canonical_members = tuple(
            sorted(members, key=lambda value: value.branch_id)
        )
        classes.append(
            (
                _representative_binding(canonical_members),
                canonical_members,
            )
        )
    return tuple(
        sorted(classes, key=lambda value: value[0].branch_id)
    )


def _equivalence_class_records(
    classes: tuple[
        tuple[
            PortfolioRacePolicyBinding,
            tuple[PortfolioRacePolicyBinding, ...],
        ],
        ...,
    ],
    *,
    actions_by_branch: dict[str, tuple[str, ...]],
) -> list[dict[str, object]]:
    return [
        {
            "representative_branch_id": representative.branch_id,
            "member_branch_ids": [
                value.branch_id for value in members
            ],
            "action_sha256s": list(
                actions_by_branch[representative.branch_id]
            ),
        }
        for representative, members in classes
    ]


def _outcome_blind_route_resolution(
    *,
    source_classes: tuple[
        tuple[
            PortfolioRacePolicyBinding,
            tuple[PortfolioRacePolicyBinding, ...],
        ],
        ...,
    ],
    final_classes: tuple[
        tuple[
            PortfolioRacePolicyBinding,
            tuple[PortfolioRacePolicyBinding, ...],
        ],
        ...,
    ],
) -> list[dict[str, object]]:
    """Authenticate source-to-retained aliases through exact slate equality."""

    post_pilot_representative: dict[str, str] = {}
    for representative, members in final_classes:
        for member in members:
            post_pilot_representative[member.branch_id] = (
                representative.branch_id
            )
    rows: list[dict[str, object]] = []
    for source_representative, source_members in source_classes:
        retained = post_pilot_representative.get(
            source_representative.branch_id
        )
        if retained is None:
            continue
        for member in source_members:
            path: list[str] = []
            if member.branch_id != source_representative.branch_id:
                path.append(
                    "identical_complete_action_set_before_pilot"
                )
            if source_representative.branch_id != retained:
                path.append(
                    "identical_complete_action_set_after_frozen_pilot"
                )
            if not path:
                path.append("identity")
            rows.append(
                {
                    "source_branch_id": member.branch_id,
                    "representative_branch_id": retained,
                    "equivalence_path": path,
                }
            )
    return sorted(rows, key=lambda value: str(value["source_branch_id"]))


def _symmetric_difference_pilot_ids(
    *,
    branch_bindings: tuple[PortfolioRacePolicyBinding, ...],
    branch_requirements: dict[
        str,
        MaterializedActionAllocationRequirement,
    ],
    complete_action_ids: dict[str, tuple[str, ...]],
    market: dict[str, MaterializedActionDescriptor],
    pilot_slots: int,
) -> tuple[tuple[str, str], ...]:
    """Choose rare branch-disagreement actions, never common actions."""

    if type(pilot_slots) is not int or pilot_slots <= 0:
        raise ValueError("pilot_slots must be positive")
    action_sets = tuple(
        set(complete_action_ids[value.branch_id])
        for value in branch_bindings
    )
    common = set.intersection(*action_sets)
    frequency: dict[str, int] = {}
    for action_set in action_sets:
        for action_sha256 in action_set:
            frequency[action_sha256] = frequency.get(action_sha256, 0) + 1
    order_by_branch: dict[str, dict[str, int]] = {}
    for binding in branch_bindings:
        source_order = _ordered_action_ids(
            branch_requirements[binding.branch_id]
        )
        ordinal = {
            action_sha256: index
            for index, action_sha256 in enumerate(source_order)
        }
        order_by_branch[binding.branch_id] = ordinal
    ranked_by_branch: dict[str, tuple[str, ...]] = {}
    for binding in branch_bindings:
        branch_id = binding.branch_id
        ranked_by_branch[branch_id] = tuple(
            sorted(
                (
                    action_sha256
                    for action_sha256
                    in complete_action_ids[branch_id]
                    if action_sha256 not in common
                ),
                key=lambda action_sha256: (
                    frequency[action_sha256],
                    order_by_branch[branch_id].get(
                        action_sha256,
                        len(market),
                    ),
                    action_sha256,
                ),
            )
        )
    selection_order = tuple(
        sorted(
            branch_bindings,
            key=lambda value: (-value.prior_mean, value.branch_id),
        )
    )
    selected: list[tuple[str, str]] = []
    selected_actions: set[str] = set()
    selected_phenotypes: set[str] = set()
    depth = 0
    while len(selected) < pilot_slots:
        added = False
        for binding in selection_order:
            candidates = ranked_by_branch[binding.branch_id]
            candidate_index = depth
            while candidate_index < len(candidates):
                action_sha256 = candidates[candidate_index]
                candidate_index += 1
                action = market[action_sha256]
                if (
                    action_sha256 in selected_actions
                    or action.phenotype_identity_sha256
                    in selected_phenotypes
                ):
                    continue
                selected.append(
                    (
                        action_sha256,
                        f"branch:{binding.branch_id}",
                    )
                )
                selected_actions.add(action_sha256)
                selected_phenotypes.add(
                    action.phenotype_identity_sha256
                )
                added = True
                break
            if len(selected) == pilot_slots:
                break
        if not added:
            break
        depth += 1
    if len(selected) != pilot_slots:
        raise ValueError(
            "distinct branches cannot provide the requested disagreement pilots"
        )
    return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class SymmetricDifferencePortfolioRacePolicy:
    """Collapse equivalent slates and race only actionable disagreement.

    ``maximum_pilot_slots`` remains owned by the planner.  This policy treats
    it as a ceiling, searches downward for the largest diagnostic pilot that
    leaves at least two distinct complete continuations, and emits a zero-pilot
    deterministic plan when every source slate is behaviorally equivalent.
    """

    policy_id: str = SYMMETRIC_DIFFERENCE_PORTFOLIO_RACE_POLICY_ID
    policy_version: int = SYMMETRIC_DIFFERENCE_PORTFOLIO_RACE_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="disagreement policy_id")
        if (
            self.policy_version
            != SYMMETRIC_DIFFERENCE_PORTFOLIO_RACE_POLICY_VERSION
        ):
            raise ValueError("disagreement policy_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DISAGREEMENT_POLICY_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "source_equivalence": (
                        "identical_complete_action_sha256_set"
                    ),
                    "representative": (
                        "higher_prior_mean_then_branch_id"
                    ),
                    "pilot_support": "source_symmetric_difference_only",
                    "pilot_cardinality": (
                        "largest_at_most_configured_maximum_preserving_"
                        "multiple_complete_continuations"
                    ),
                    "no_actionable_disagreement": (
                        "zero_pilot_deterministic_representative"
                    ),
                    "collapsed_route_resolution": (
                        "authenticated_exact_action_set_equivalence_chain"
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def design(
        self,
        *,
        branch_bindings: tuple[PortfolioRacePolicyBinding, ...],
        branch_requirements: tuple[
            MaterializedActionAllocationRequirement,
            ...,
        ],
        market: dict[str, MaterializedActionDescriptor],
        evaluation_slots: int,
        maximum_pilot_slots: int,
    ) -> PortfolioRaceDisagreementDesign:
        self.__post_init__()
        if (
            type(branch_bindings) is not tuple
            or len(branch_bindings) < 2
            or len(branch_bindings) != len(branch_requirements)
        ):
            raise ValueError("disagreement design needs aligned branches")
        if (
            type(evaluation_slots) is not int
            or type(maximum_pilot_slots) is not int
            or not 0 < maximum_pilot_slots < evaluation_slots
        ):
            raise ValueError(
                "maximum_pilot_slots must leave continuation capacity"
            )
        source_by_branch = {
            binding.branch_id: requirement
            for binding, requirement in zip(
                branch_bindings,
                branch_requirements,
                strict=True,
            )
        }
        raw_actions = {
            binding.branch_id: _compose_branch_action_ids(
                pilot_action_ids=(),
                source_requirement=source_by_branch[binding.branch_id],
                market=market,
                evaluation_slots=evaluation_slots,
            )
            for binding in branch_bindings
        }
        source_classes = _equivalence_classes(
            bindings=branch_bindings,
            actions_by_branch=raw_actions,
        )
        active = tuple(value[0] for value in source_classes)
        source_class_records = _equivalence_class_records(
            source_classes,
            actions_by_branch=raw_actions,
        )
        attempts: list[dict[str, object]] = []
        if len(active) == 1:
            representative = active[0]
            retained_classes = (
                (representative, (representative,)),
            )
            return PortfolioRaceDisagreementDesign(
                pilot_pairs=(),
                branch_action_sha256s=(
                    (
                        representative.branch_id,
                        raw_actions[representative.branch_id],
                    ),
                ),
                evidence=freeze_json(
                    {
                        "policy_definition_sha256": self.definition_sha256,
                        "decision": "equivalent_source_bypass",
                        "source_equivalence_classes": source_class_records,
                        "retained_equivalence_classes": (
                            _equivalence_class_records(
                                retained_classes,
                                actions_by_branch=raw_actions,
                            )
                        ),
                        "outcome_blind_route_resolution": (
                            _outcome_blind_route_resolution(
                                source_classes=source_classes,
                                final_classes=retained_classes,
                            )
                        ),
                        "configured_maximum_pilot_slots": (
                            maximum_pilot_slots
                        ),
                        "effective_pilot_slots": 0,
                        "candidate_outcomes_observed": False,
                        "workload_model_provider_branches": False,
                    }
                ),
            )

        while len(active) >= 2:
            largest = min(
                maximum_pilot_slots,
                len(active),
                evaluation_slots - 1,
            )
            retained: tuple[PortfolioRacePolicyBinding, ...] | None = None
            retained_pairs: tuple[tuple[str, str], ...] = ()
            retained_actions: dict[str, tuple[str, ...]] = {}
            for requested_slots in range(largest, 0, -1):
                try:
                    pilot_pairs = _symmetric_difference_pilot_ids(
                        branch_bindings=active,
                        branch_requirements=source_by_branch,
                        complete_action_ids=raw_actions,
                        market=market,
                        pilot_slots=requested_slots,
                    )
                except ValueError:
                    attempts.append(
                        {
                            "active_branch_ids": [
                                value.branch_id for value in active
                            ],
                            "requested_pilot_slots": requested_slots,
                            "pilot_action_sha256s": [],
                            "distinct_complete_continuations": None,
                            "reason": (
                                "insufficient_unique_disagreement_phenotypes"
                            ),
                        }
                    )
                    continue
                pilot_ids = tuple(value[0] for value in pilot_pairs)
                composed = {
                    binding.branch_id: _compose_branch_action_ids(
                        pilot_action_ids=pilot_ids,
                        source_requirement=source_by_branch[
                            binding.branch_id
                        ],
                        market=market,
                        evaluation_slots=evaluation_slots,
                    )
                    for binding in active
                }
                final_classes = _equivalence_classes(
                    bindings=active,
                    actions_by_branch=composed,
                )
                attempts.append(
                    {
                        "active_branch_ids": [
                            value.branch_id for value in active
                        ],
                        "requested_pilot_slots": requested_slots,
                        "pilot_action_sha256s": list(pilot_ids),
                        "distinct_complete_continuations": len(
                            final_classes
                        ),
                    }
                )
                if len(final_classes) >= 2:
                    retained = tuple(
                        value[0] for value in final_classes
                    )
                    retained_pairs = pilot_pairs
                    retained_actions = {
                        representative.branch_id: composed[
                            representative.branch_id
                        ]
                        for representative in retained
                    }
                    break
            if retained is None:
                break
            return PortfolioRaceDisagreementDesign(
                pilot_pairs=retained_pairs,
                branch_action_sha256s=tuple(
                    sorted(retained_actions.items())
                ),
                evidence=freeze_json(
                    {
                        "policy_definition_sha256": (
                            self.definition_sha256
                        ),
                        "decision": "actionable_disagreement_race",
                        "source_equivalence_classes": (
                            source_class_records
                        ),
                        "retained_equivalence_classes": (
                            _equivalence_class_records(
                                final_classes,
                                actions_by_branch=composed,
                            )
                        ),
                        "outcome_blind_route_resolution": (
                            _outcome_blind_route_resolution(
                                source_classes=source_classes,
                                final_classes=final_classes,
                            )
                        ),
                        "pilot_search_attempts": attempts,
                        "configured_maximum_pilot_slots": (
                            maximum_pilot_slots
                        ),
                        "effective_pilot_slots": len(
                            retained_pairs
                        ),
                        "retained_branch_ids": [
                            value.branch_id for value in retained
                        ],
                        "candidate_outcomes_observed": False,
                        "workload_model_provider_branches": False,
                    }
                ),
            )

        representative = _representative_binding(active)
        representative_source_classes = tuple(
            value
            for value in source_classes
            if value[0].branch_id == representative.branch_id
        )
        retained_classes = ((representative, (representative,)),)
        return PortfolioRaceDisagreementDesign(
            pilot_pairs=(),
            branch_action_sha256s=(
                (
                    representative.branch_id,
                    raw_actions[representative.branch_id],
                ),
            ),
            evidence=freeze_json(
                {
                    "policy_definition_sha256": self.definition_sha256,
                    "decision": "no_actionable_pilot_bypass",
                    "source_equivalence_classes": source_class_records,
                    "retained_equivalence_classes": (
                        _equivalence_class_records(
                            retained_classes,
                            actions_by_branch=raw_actions,
                        )
                    ),
                    "outcome_blind_route_resolution": (
                        _outcome_blind_route_resolution(
                            source_classes=representative_source_classes,
                            final_classes=retained_classes,
                        )
                    ),
                    "pilot_search_attempts": attempts,
                    "configured_maximum_pilot_slots": maximum_pilot_slots,
                    "effective_pilot_slots": 0,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PrecommittedPortfolioRacePlanner:
    """Freeze a lane-stratified pilot and N complete portfolio branches."""

    branch_bindings: tuple[PortfolioRacePolicyBinding, ...]
    pilot_policy: MaterializedActionAllocationPolicyPort | None = field(
        repr=False,
        compare=False,
    )
    pilot_slots: int
    disagreement_policy: PortfolioRaceDisagreementPolicyPort | None = (
        field(default=None, repr=False, compare=False)
    )
    prior_projection: PortfolioRacePriorProjectionPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    lane_projection: PortfolioLaneProjectionPort = field(
        default_factory=TraceFieldPortfolioLaneProjection,
        repr=False,
        compare=False,
    )
    planner_id: str = PRECOMMITTED_PORTFOLIO_RACE_PLANNER_ID
    planner_version: int = PRECOMMITTED_PORTFOLIO_RACE_PLANNER_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.branch_bindings) is not tuple
            or len(self.branch_bindings) < 2
        ):
            raise ValueError("a portfolio race needs at least two branches")
        for value in self.branch_bindings:
            if type(value) is not PortfolioRacePolicyBinding:
                raise TypeError("branch bindings must be exact")
            value.__post_init__()
        branch_ids = tuple(value.branch_id for value in self.branch_bindings)
        if branch_ids != tuple(sorted(set(branch_ids))):
            raise ValueError("branch bindings must use canonical unique IDs")
        pilot_identity = (
            None
            if self.pilot_policy is None
            else _policy_identity(self.pilot_policy)
        )
        disagreement_identity = (
            None
            if self.disagreement_policy is None
            else _disagreement_policy_identity(
                self.disagreement_policy
            )
        )
        prior_identity = (
            None
            if self.prior_projection is None
            else _prior_projection_identity(self.prior_projection)
        )
        if (
            self.disagreement_policy is not None
            and self.pilot_policy is not None
        ):
            raise ValueError(
                "disagreement-aware design replaces an injected pilot policy"
            )
        if type(self.pilot_slots) is not int or self.pilot_slots <= 0:
            raise ValueError("pilot_slots must be positive")
        lane_identity = _lane_projection_identity(self.lane_projection)
        _require_token(self.planner_id, name="planner_id")
        if self.planner_version != PRECOMMITTED_PORTFOLIO_RACE_PLANNER_VERSION:
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
                    "branch_policies": [
                        value.identity_record()
                        for value in self.branch_bindings
                    ],
                    "pilot_policy": {
                        "mode": (
                            "branch_stratified"
                            if pilot_identity is None
                            else "injected_policy_lane_stratified"
                        ),
                        "policy_id": (
                            None
                            if pilot_identity is None
                            else pilot_identity[0]
                        ),
                        "policy_version": (
                            None
                            if pilot_identity is None
                            else pilot_identity[1]
                        ),
                        "definition_sha256": (
                            None
                            if pilot_identity is None
                            else pilot_identity[2]
                        ),
                    },
                    "pilot_slots": self.pilot_slots,
                    "disagreement_policy": (
                        None
                        if disagreement_identity is None
                        else {
                            "policy_id": disagreement_identity[0],
                            "policy_version": disagreement_identity[1],
                            "definition_sha256": disagreement_identity[2],
                        }
                    ),
                    "prior_projection": (
                        None
                        if prior_identity is None
                        else {
                            "projection_id": prior_identity[0],
                            "projection_version": prior_identity[1],
                            "definition_sha256": prior_identity[2],
                        }
                    ),
                    "prior_projection_timing": (
                        "before_disagreement_equivalence_resolution"
                    ),
                    "pilot_provenance_projection_scope": (
                        "full_frozen_source_market_before_branch_collapse"
                    ),
                    "lane_projection": {
                        "projection_id": lane_identity[0],
                        "projection_version": lane_identity[1],
                        "definition_sha256": lane_identity[2],
                    },
                    "all_complete_branches_frozen_before_pilot": True,
                    "outcome_conditioned_regeneration": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    async def plan(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> SequentialAllocationPlanPort:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if not 0 < self.pilot_slots < request.evaluation_slots:
            raise ValueError("pilot_slots must leave a continuation wave")
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
        if self.pilot_policy is None:
            branch_sources = tuple(
                await asyncio.gather(
                    *(
                        value.policy.require(request, proposals)
                        for value in self.branch_bindings
                    )
                )
            )
            requirements = branch_sources
            pilot_source = None
        else:
            requirements = tuple(
                await asyncio.gather(
                    self.pilot_policy.require(request, proposals),
                    *(
                        value.policy.require(request, proposals)
                        for value in self.branch_bindings
                    ),
                )
            )
            pilot_source = requirements[0]
            branch_sources = requirements[1:]
        for requirement in requirements:
            if type(requirement) is not MaterializedActionAllocationRequirement:
                raise TypeError("allocation policy returned a foreign requirement")
            requirement.__post_init__()
            if (
                requirement.residual_request_sha256
                != request.request_sha256
            ):
                raise ValueError("allocation policy targeted another request")
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        if any(
            value.proposal_sha256s != proposal_sha256s
            for value in requirements
        ):
            raise ValueError("allocation policies saw different proposal markets")
        market = _action_market(proposals)
        source_by_branch = {
            binding.branch_id: source
            for binding, source in zip(
                self.branch_bindings,
                branch_sources,
                strict=True,
            )
        }
        projected_prior_by_branch = {
            binding.branch_id: (
                binding.prior_mean
                if self.prior_projection is None
                else self.prior_projection.project(
                    request,
                    binding,
                    source_by_branch[binding.branch_id],
                )
            )
            for binding in self.branch_bindings
        }
        effective_branch_bindings = tuple(
            PortfolioRacePolicyBinding(
                branch_id=binding.branch_id,
                policy=binding.policy,
                prior_mean=projected_prior_by_branch[binding.branch_id],
            )
            for binding in self.branch_bindings
        )
        disagreement_design = (
            None
            if self.disagreement_policy is None
            else self.disagreement_policy.design(
                branch_bindings=effective_branch_bindings,
                branch_requirements=branch_sources,
                market=market,
                evaluation_slots=request.evaluation_slots,
                maximum_pilot_slots=self.pilot_slots,
            )
        )
        if (
            disagreement_design is not None
            and type(disagreement_design)
            is not PortfolioRaceDisagreementDesign
        ):
            raise TypeError(
                "disagreement policy returned a foreign design"
            )
        if disagreement_design is not None:
            disagreement_design.__post_init__()
        if disagreement_design is None:
            active_bindings = effective_branch_bindings
            active_sources = branch_sources
            raw_pilot_pairs = (
                _branch_stratified_pilot_ids(
                    branch_bindings=active_bindings,
                    branch_requirements=active_sources,
                    market=market,
                    pilot_slots=self.pilot_slots,
                )
                if pilot_source is None
                else _stratified_pilot_ids(
                    requirement=pilot_source,
                    market=market,
                    lane_projection=self.lane_projection,
                    pilot_slots=self.pilot_slots,
                )
            )
        else:
            designed_branch_ids = {
                value[0]
                for value in disagreement_design.branch_action_sha256s
            }
            active_bindings = tuple(
                value
                for value in effective_branch_bindings
                if value.branch_id in designed_branch_ids
            )
            active_sources = tuple(
                source_by_branch[value.branch_id]
                for value in active_bindings
            )
            raw_pilot_pairs = disagreement_design.pilot_pairs
        if pilot_source is None:
            projection_bindings = (
                effective_branch_bindings
                if disagreement_design is not None
                else active_bindings
            )
            projection_sources = (
                branch_sources
                if disagreement_design is not None
                else active_sources
            )
            pilot_pairs, pilot_source_attribution = (
                _project_branch_pilot_lanes(
                    pilot_pairs=raw_pilot_pairs,
                    branch_bindings=projection_bindings,
                    branch_requirements=projection_sources,
                    market=market,
                    lane_projection=self.lane_projection,
                )
            )
        else:
            pilot_pairs = raw_pilot_pairs
            pilot_source_attribution = tuple(
                (
                    action_sha256,
                    "injected_pilot_policy",
                    lane_id,
                )
                for action_sha256, lane_id in pilot_pairs
            )
        prior_by_branch = {
            binding.branch_id: binding.prior_mean
            for binding in active_bindings
        }
        pilot_ids = tuple(value[0] for value in pilot_pairs)
        pilot_source_sha256s = (
            tuple(
                value.requirement_sha256 for value in branch_sources
            )
            if pilot_source is None
            else (pilot_source.requirement_sha256,)
        )
        pilot_definition_sha256 = _hash(
            _PILOT_REQUIREMENT_DEFINITION_DOMAIN,
            {
                "planner_definition_sha256": self.definition_sha256,
                "source_requirement_sha256s": list(
                    pilot_source_sha256s
                ),
                "mode": (
                    "symmetric_difference_adaptive"
                    if disagreement_design is not None
                    else (
                        "branch_stratified"
                        if pilot_source is None
                        else "injected_policy_lane_stratified"
                    )
                ),
                "configured_maximum_pilot_slots": self.pilot_slots,
                "effective_pilot_slots": len(pilot_pairs),
                "pilot_lane_bindings": [
                    {"action_sha256": action, "lane_id": lane}
                    for action, lane in pilot_pairs
                ],
            },
        )
        pilot_requirement = MaterializedActionAllocationRequirement(
            policy_id="precommitted_portfolio_race_pilot",
            policy_version=1,
            policy_definition_sha256=pilot_definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=pilot_ids,
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "selection_trace": [
                        {
                            "ordinal": ordinal,
                            "action_sha256": action_sha256,
                            "allocation_kind": "portfolio_race_pilot",
                            "source_lane": lane_id,
                            "source_branch_id": next(
                                source_branch_id
                                for (
                                    source_action_sha256,
                                    source_branch_id,
                                    _projected_lane_id,
                                ) in pilot_source_attribution
                                if source_action_sha256 == action_sha256
                            ),
                            "candidate_outcomes_observed": False,
                        }
                        for ordinal, (action_sha256, lane_id) in enumerate(
                            pilot_pairs,
                            start=1,
                        )
                    ],
                    "source_requirement_sha256s": list(
                        pilot_source_sha256s
                    ),
                    "mode": (
                        "symmetric_difference_adaptive"
                        if disagreement_design is not None
                        else (
                            "branch_stratified"
                            if pilot_source is None
                            else "injected_policy_lane_stratified"
                        )
                    ),
                    "lane_stratified": bool(pilot_pairs),
                    "configured_maximum_pilot_slots": (
                        self.pilot_slots
                    ),
                    "effective_pilot_slots": len(pilot_pairs),
                    "disagreement_design_sha256": (
                        None
                        if disagreement_design is None
                        else disagreement_design.design_sha256
                    ),
                    "pilot_source_attribution": [
                        {
                            "action_sha256": action_sha256,
                            "source_branch_id": source_branch_id,
                            "projected_lane_id": lane_id,
                        }
                        for (
                            action_sha256,
                            source_branch_id,
                            lane_id,
                        ) in pilot_source_attribution
                    ],
                    "branch_identity_used_as_exchangeability_lane": False,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )

        branches: list[FrozenPortfolioRaceBranch] = []
        for binding, source in zip(
            active_bindings,
            active_sources,
            strict=True,
        ):
            source_trace = _trace_by_action(source)
            final_ids = (
                _compose_branch_action_ids(
                    pilot_action_ids=pilot_ids,
                    source_requirement=source,
                    market=market,
                    evaluation_slots=request.evaluation_slots,
                )
                if disagreement_design is None
                else disagreement_design.actions_for(
                    binding.branch_id
                )
            )
            completion_ids = tuple(
                value for value in final_ids if value not in set(pilot_ids)
            )
            lane_by_action: dict[str, str] = {}
            for action_sha256 in completion_ids:
                row = source_trace.get(
                    action_sha256,
                    {
                        "action_sha256": action_sha256,
                        "allocation_kind": "deterministic_market_refill",
                    },
                )
                lane_by_action[action_sha256] = (
                    self.lane_projection.project(
                        row,
                        market[action_sha256],
                    )
                )
            exposure: dict[str, int] = {}
            for lane_id in lane_by_action.values():
                exposure[lane_id] = exposure.get(lane_id, 0) + 1
            branch_definition_sha256 = _hash(
                _BRANCH_REQUIREMENT_DEFINITION_DOMAIN,
                {
                    "planner_definition_sha256": self.definition_sha256,
                    "branch_id": binding.branch_id,
                    "source_requirement_sha256": (
                        source.requirement_sha256
                    ),
                    "pilot_requirement_sha256": (
                        pilot_requirement.requirement_sha256
                    ),
                    "completion_lane_exposure": [
                        {"lane_id": lane_id, "count": count}
                        for lane_id, count in sorted(exposure.items())
                    ],
                },
            )
            branch_requirement = MaterializedActionAllocationRequirement(
                policy_id="precommitted_portfolio_race_branch",
                policy_version=1,
                policy_definition_sha256=branch_definition_sha256,
                residual_request_sha256=request.request_sha256,
                proposal_sha256s=proposal_sha256s,
                required_action_sha256s=final_ids,
                candidate_outcomes_observed=False,
                evidence=freeze_json(
                    {
                        "branch_id": binding.branch_id,
                        "source_requirement_sha256": (
                            source.requirement_sha256
                        ),
                        "selection_trace": [
                            {
                                "ordinal": ordinal,
                                "action_sha256": action_sha256,
                                "allocation_kind": (
                                    "portfolio_race_common_pilot"
                                    if action_sha256 in set(pilot_ids)
                                    else "portfolio_race_completion"
                                ),
                                "source_lane": (
                                    next(
                                        lane
                                        for action, lane in pilot_pairs
                                        if action == action_sha256
                                    )
                                    if action_sha256 in set(pilot_ids)
                                    else lane_by_action[action_sha256]
                                ),
                                "candidate_outcomes_observed": False,
                            }
                            for ordinal, action_sha256 in enumerate(
                                final_ids,
                                start=1,
                            )
                        ],
                        "completion_lane_exposure": [
                            {"lane_id": lane_id, "count": count}
                            for lane_id, count in sorted(exposure.items())
                        ],
                        "candidate_outcomes_observed": False,
                        "workload_model_provider_branches": False,
                    }
                ),
            )
            branches.append(
                FrozenPortfolioRaceBranch(
                    branch_id=binding.branch_id,
                    source_requirement_sha256=source.requirement_sha256,
                    requirement=branch_requirement,
                    completion_lane_exposure=tuple(sorted(exposure.items())),
                    prior_mean=prior_by_branch[binding.branch_id],
                )
            )
        return PrecommittedPortfolioRacePlan(
            planner_id=self.planner_id,
            planner_version=self.planner_version,
            planner_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            evaluation_slots=request.evaluation_slots,
            pilot_requirement=pilot_requirement,
            pilot_lane_bindings=tuple(
                PortfolioRacePilotLaneBinding(
                    action_sha256=action_sha256,
                    lane_id=lane_id,
                )
                for action_sha256, lane_id in pilot_pairs
            ),
            branches=tuple(sorted(branches, key=lambda value: value.branch_id)),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "pilot_source_requirement_sha256s": list(
                        pilot_source_sha256s
                    ),
                    "pilot_mode": (
                        "symmetric_difference_adaptive"
                        if disagreement_design is not None
                        else (
                            "branch_stratified"
                            if pilot_source is None
                            else "injected_policy_lane_stratified"
                        )
                    ),
                    "all_branch_source_requirement_sha256s": [
                        value.source_requirement_sha256
                        for value in sorted(
                            branches,
                            key=lambda item: item.branch_id,
                        )
                    ],
                    "complete_branch_count": len(branches),
                    "configured_maximum_pilot_slots": self.pilot_slots,
                    "effective_pilot_slots": len(pilot_pairs),
                    "pilot_source_attribution": [
                        {
                            "action_sha256": action_sha256,
                            "source_branch_id": source_branch_id,
                            "projected_lane_id": lane_id,
                        }
                        for (
                            action_sha256,
                            source_branch_id,
                            lane_id,
                        ) in pilot_source_attribution
                    ],
                    "pilot_and_completion_lanes_share_projection": True,
                    "branch_identity_used_as_exchangeability_lane": False,
                    "branch_prior_projection": (
                        {
                            "mode": "binding_prior_mean",
                            "prior_means": [
                                {
                                    "branch_id": branch_id,
                                    "mean_hex": mean.hex(),
                                }
                                for branch_id, mean in sorted(
                                    projected_prior_by_branch.items()
                                )
                            ],
                            "applied_before_disagreement_equivalence": True,
                            "candidate_outcomes_observed": False,
                        }
                        if self.prior_projection is None
                        else {
                            "mode": "injected_outcome_blind_projection",
                            "projection_id": (
                                self.prior_projection.projection_id
                            ),
                            "projection_version": (
                                self.prior_projection.projection_version
                            ),
                            "definition_sha256": (
                                self.prior_projection.definition_sha256
                            ),
                            "request_phase": request.phase.value,
                            "prior_means": [
                                {
                                    "branch_id": branch_id,
                                    "mean_hex": mean.hex(),
                                }
                                for branch_id, mean in sorted(
                                    projected_prior_by_branch.items()
                                )
                            ],
                            "applied_before_disagreement_equivalence": True,
                            "candidate_outcomes_observed": False,
                        }
                    ),
                    "disagreement_design": (
                        None
                        if disagreement_design is None
                        else disagreement_design.to_record()
                    ),
                    "all_complete_branches_frozen_before_pilot": True,
                    "outcome_conditioned_regeneration": False,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PortfolioRaceBranchScore:
    branch_id: str
    score: float
    observed_completion_fraction: float
    lane_posteriors: tuple[tuple[str, float, int], ...]

    def __post_init__(self) -> None:
        _require_token(self.branch_id, name="branch_id")
        for value, name in (
            (self.score, "score"),
            (
                self.observed_completion_fraction,
                "observed_completion_fraction",
            ),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if (
            type(self.lane_posteriors) is not tuple
            or not self.lane_posteriors
            or self.lane_posteriors
            != tuple(sorted(self.lane_posteriors))
        ):
            raise ValueError("lane_posteriors must be non-empty and canonical")
        for lane_id, posterior, count in self.lane_posteriors:
            _require_token(lane_id, name="lane posterior ID")
            if (
                type(posterior) is not float
                or not math.isfinite(posterior)
                or not 0.0 <= posterior <= 1.0
            ):
                raise ValueError("lane posterior must lie in [0, 1]")
            if type(count) is not int or count < 0:
                raise ValueError("lane observation count must be non-negative")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "branch_id": self.branch_id,
            "score_hex": self.score.hex(),
            "observed_completion_fraction_hex": (
                self.observed_completion_fraction.hex()
            ),
            "lane_posteriors": [
                {
                    "lane_id": lane_id,
                    "posterior_mean_hex": posterior.hex(),
                    "observation_count": count,
                }
                for lane_id, posterior, count in self.lane_posteriors
            ],
        }


@dataclass(frozen=True, slots=True)
class PortfolioRaceGateDecision:
    gate_id: str
    gate_version: int
    gate_definition_sha256: str
    plan_sha256: str
    pilot_outcome_batch_sha256: str
    selected_branch_id: str
    selected_requirement_sha256: str
    positive_pilot_count: int
    pilot_count: int
    branch_scores: tuple[PortfolioRaceBranchScore, ...]
    evidence: FrozenJsonObject
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.gate_id, name="gate_id")
        if type(self.gate_version) is not int or self.gate_version <= 0:
            raise ValueError("gate_version must be positive")
        for value, name in (
            (self.gate_definition_sha256, "gate_definition_sha256"),
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
        _require_token(self.selected_branch_id, name="selected_branch_id")
        if (
            type(self.positive_pilot_count) is not int
            or type(self.pilot_count) is not int
            or not 0 <= self.positive_pilot_count <= self.pilot_count
        ):
            raise ValueError("pilot counts are inconsistent")
        if (
            type(self.branch_scores) is not tuple
            or not self.branch_scores
            or tuple(value.branch_id for value in self.branch_scores)
            != tuple(sorted({value.branch_id for value in self.branch_scores}))
        ):
            raise ValueError("branch_scores must be non-empty and canonical")
        for value in self.branch_scores:
            if type(value) is not PortfolioRaceBranchScore:
                raise TypeError("branch_scores must be exact")
            value.__post_init__()
        if self.selected_branch_id not in {
            value.branch_id for value in self.branch_scores
        }:
            raise ValueError("selected branch is absent from scores")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("gate evidence must be a frozen object")
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_GATE_DECISION_DOMAIN, self._unsigned_record()),
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
            "selected_branch_id": self.selected_branch_id,
            "selected_requirement_sha256": (
                self.selected_requirement_sha256
            ),
            "positive_pilot_count": self.positive_pilot_count,
            "pilot_count": self.pilot_count,
            "branch_scores": [
                value.to_record() for value in self.branch_scores
            ],
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "decision_sha256": self.decision_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@dataclass(frozen=True, slots=True)
class EvidenceAdaptivePortfolioRaceGate:
    """Select a frozen slate only when pilot-to-completion transfer is covered.

    A pilot may update only its predeclared projected lane.  Adaptive branch
    choice is additionally disabled unless *every* branch has sufficient
    completion exposure in observed lanes.  The fail-closed decision then uses
    frozen branch priors, preventing a single non-exchangeable pilot from
    deciding an entire heterogeneous continuation.
    """

    prior_strength: float = 1.0
    minimum_observed_completion_fraction: float = 0.5
    gate_id: str = EVIDENCE_ADAPTIVE_PORTFOLIO_RACE_GATE_ID
    gate_version: int = EVIDENCE_ADAPTIVE_PORTFOLIO_RACE_GATE_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.prior_strength) is not float
            or not math.isfinite(self.prior_strength)
            or self.prior_strength <= 0.0
        ):
            raise ValueError("prior_strength must be finite and positive")
        if (
            type(self.minimum_observed_completion_fraction) is not float
            or not math.isfinite(
                self.minimum_observed_completion_fraction
            )
            or not 0.0
            < self.minimum_observed_completion_fraction
            <= 1.0
        ):
            raise ValueError(
                "minimum_observed_completion_fraction must lie in (0, 1]"
            )
        _require_token(self.gate_id, name="gate_id")
        if (
            self.gate_version
            != EVIDENCE_ADAPTIVE_PORTFOLIO_RACE_GATE_VERSION
        ):
            raise ValueError("gate_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _GATE_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "gate_id": self.gate_id,
                    "gate_version": self.gate_version,
                    "prior_strength_hex": self.prior_strength.hex(),
                    "minimum_observed_completion_fraction_hex": (
                        self.minimum_observed_completion_fraction.hex()
                    ),
                    "pilot_normalization": "max_positive_marginal_gain",
                    "lane_update": "bounded_consequence_posterior_mean",
                    "branch_score": "completion_exposure_weighted_lane_mean",
                    "adaptive_qualification": (
                        "all_branches_meet_minimum_observed_completion_"
                        "fraction"
                    ),
                    "unqualified_fallback": (
                        "higher_frozen_prior_then_branch_id"
                    ),
                    "tie_break": "higher_score_then_branch_id",
                    "only_pilot_outcomes_observed": True,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def decide(
        self,
        plan: SequentialAllocationPlanPort,
        outcomes: SequentialPilotOutcomeBatch,
    ) -> SequentialAllocationGateDecisionPort:
        self.__post_init__()
        if type(plan) is not PrecommittedPortfolioRacePlan:
            raise TypeError("portfolio-race gate requires its exact plan")
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
            raise ValueError("pilot outcomes differ from their race plan")
        lane_by_action = {
            value.action_sha256: value.lane_id
            for value in plan.pilot_lane_bindings
        }
        maximum_gain = max(
            (
                value.marginal_archive_gain
                for value in outcomes.outcomes
            ),
            default=0.0,
        )
        normalized_by_lane: dict[str, list[float]] = {}
        for outcome in outcomes.outcomes:
            normalized = (
                0.0
                if maximum_gain == 0.0
                else outcome.marginal_archive_gain / maximum_gain
            )
            normalized_by_lane.setdefault(
                lane_by_action[outcome.action_sha256],
                [],
            ).append(normalized)
        scores: list[PortfolioRaceBranchScore] = []
        for branch in plan.branches:
            weighted = 0.0
            observed_exposure = 0
            total_exposure = sum(
                count for _lane_id, count in branch.completion_lane_exposure
            )
            posteriors: list[tuple[str, float, int]] = []
            for lane_id, exposure in branch.completion_lane_exposure:
                observations = normalized_by_lane.get(lane_id, [])
                posterior = (
                    branch.prior_mean * self.prior_strength
                    + math.fsum(observations)
                ) / (self.prior_strength + len(observations))
                weighted += exposure * posterior
                if observations:
                    observed_exposure += exposure
                posteriors.append((lane_id, posterior, len(observations)))
            scores.append(
                PortfolioRaceBranchScore(
                    branch_id=branch.branch_id,
                    score=weighted / total_exposure,
                    observed_completion_fraction=(
                        observed_exposure / total_exposure
                    ),
                    lane_posteriors=tuple(sorted(posteriors)),
                )
            )
        adaptive_selection_qualified = bool(outcomes.outcomes) and all(
            value.observed_completion_fraction
            >= self.minimum_observed_completion_fraction
            for value in scores
        )
        if adaptive_selection_qualified:
            selected_score = min(
                scores,
                key=lambda value: (-value.score, value.branch_id),
            )
            selected = plan.branch_for(selected_score.branch_id)
            selection_mode = "coverage_qualified_lane_adaptation"
        else:
            selected = min(
                plan.branches,
                key=lambda value: (-value.prior_mean, value.branch_id),
            )
            selection_mode = "insufficient_coverage_frozen_prior_fallback"
        positive_count = sum(
            value.positive_marginal_utility for value in outcomes.outcomes
        )
        return PortfolioRaceGateDecision(
            gate_id=self.gate_id,
            gate_version=self.gate_version,
            gate_definition_sha256=self.definition_sha256,
            plan_sha256=plan.plan_sha256,
            pilot_outcome_batch_sha256=outcomes.batch_sha256,
            selected_branch_id=selected.branch_id,
            selected_requirement_sha256=(
                selected.requirement.requirement_sha256
            ),
            positive_pilot_count=positive_count,
            pilot_count=len(outcomes.outcomes),
            branch_scores=tuple(
                sorted(scores, key=lambda value: value.branch_id)
            ),
            evidence=freeze_json(
                {
                    "maximum_pilot_marginal_gain_hex": maximum_gain.hex(),
                    "minimum_observed_completion_fraction_hex": (
                        self.minimum_observed_completion_fraction.hex()
                    ),
                    "adaptive_selection_qualified": (
                        adaptive_selection_qualified
                    ),
                    "selection_mode": selection_mode,
                    "pilot_to_completion_transfer": (
                        "same_predeclared_projected_lane_only"
                    ),
                    "unobserved_completion_lanes_remain_at_frozen_prior": True,
                    "branch_identity_used_as_exchangeability_lane": False,
                    "only_pilot_outcomes_observed": True,
                    "selected_branch_frozen_before_pilot": True,
                    "all_branch_scores": [
                        value.to_record()
                        for value in sorted(
                            scores,
                            key=lambda item: item.branch_id,
                        )
                    ],
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "EVIDENCE_ADAPTIVE_PORTFOLIO_RACE_GATE_ID",
    "EVIDENCE_ADAPTIVE_PORTFOLIO_RACE_GATE_VERSION",
    "PHASE_CONDITIONED_PORTFOLIO_RACE_PRIOR_ID",
    "PHASE_CONDITIONED_PORTFOLIO_RACE_PRIOR_VERSION",
    "PRECOMMITTED_PORTFOLIO_RACE_PLANNER_ID",
    "PRECOMMITTED_PORTFOLIO_RACE_PLANNER_VERSION",
    "SYMMETRIC_DIFFERENCE_PORTFOLIO_RACE_POLICY_ID",
    "SYMMETRIC_DIFFERENCE_PORTFOLIO_RACE_POLICY_VERSION",
    "TRACE_FIELD_PORTFOLIO_LANE_PROJECTION_ID",
    "TRACE_FIELD_PORTFOLIO_LANE_PROJECTION_VERSION",
    "EvidenceAdaptivePortfolioRaceGate",
    "FrozenPortfolioRaceBranch",
    "PhaseConditionedPortfolioRacePrior",
    "PortfolioLaneProjectionPort",
    "PortfolioRacePriorProjectionPort",
    "PortfolioRaceBranchScore",
    "PortfolioRaceDisagreementDesign",
    "PortfolioRaceDisagreementPolicyPort",
    "PortfolioRaceGateDecision",
    "PortfolioRacePilotLaneBinding",
    "PortfolioRacePolicyBinding",
    "PrecommittedPortfolioRacePlan",
    "PrecommittedPortfolioRacePlanner",
    "SymmetricDifferencePortfolioRacePolicy",
    "TraceFieldPortfolioLaneProjection",
]

"""Outcome-blind factorial pilots for heterogeneous proposal branches.

Protecting only a generator branch is insufficient when useful proposals can
occur in different evolutionary roles and at different positions in an
uncalibrated within-expert ordering.  This module constructs a diagnostic
pilot over portable, pre-evaluation factors:

* an opaque source branch supplied by the composition root;
* the proposal expert and evolutionary role;
* a quantile layer of the materialized within-expert rank;
* parent lineage and operator identity; and
* exact phenotype identity and normalized evaluation cost.

The bounded design never observes candidate outcomes, workload identity,
model/provider names, objective values, prompts, or configuration contents.
It returns an allocation requirement; the ordinary outcome-adaptive broker
continues to fill every unprotected evaluation seat.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionDescriptor,
)
from agent_evolve.application.protected_branch_pilot import (
    ProtectedBranchBinding,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json


FACTORIAL_BRANCH_PILOT_POLICY_ID = "factorial_branch_pilot"
FACTORIAL_BRANCH_PILOT_POLICY_VERSION = 1
_DEFINITION_DOMAIN = b"agent-evolve:factorial-branch-pilot-definition:v1\x00"
_SELECTION_DOMAIN = b"agent-evolve:factorial-branch-pilot-selection:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _digest(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class FactorialPilotCandidate:
    """Portable pre-evaluation view of one materialized proposal."""

    action_sha256: str
    phenotype_identity_sha256: str
    branch_id: str
    expert_id: str
    role_id: str
    operator_id: str
    materialized_lane_rank: int
    lane_support_count: int
    rank_layer_index: int
    parent_ids: tuple[str, ...]
    normalized_evaluation_cost: float

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        for name in ("branch_id", "expert_id", "role_id", "operator_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty exact string")
        for name in ("materialized_lane_rank", "lane_support_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.materialized_lane_rank > self.lane_support_count:
            raise ValueError("materialized lane rank exceeds lane support")
        if (
            type(self.rank_layer_index) is not int
            or self.rank_layer_index < 0
        ):
            raise ValueError("rank_layer_index must be a non-negative integer")
        if (
            type(self.parent_ids) is not tuple
            or len(self.parent_ids) > 8
            or len(self.parent_ids) != len(set(self.parent_ids))
            or any(type(value) is not str or not value for value in self.parent_ids)
        ):
            raise ValueError("parent_ids must be a unique exact string tuple")
        if (
            type(self.normalized_evaluation_cost) is not float
            or not math.isfinite(self.normalized_evaluation_cost)
            or not 0.0 <= self.normalized_evaluation_cost <= 1.0
        ):
            raise ValueError(
                "normalized_evaluation_cost must be a finite float in [0, 1]"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "branch_id": self.branch_id,
            "expert_id": self.expert_id,
            "role_id": self.role_id,
            "operator_id": self.operator_id,
            "materialized_lane_rank": self.materialized_lane_rank,
            "lane_support_count": self.lane_support_count,
            "rank_layer_index": self.rank_layer_index,
            "parent_ids": list(self.parent_ids),
            "normalized_evaluation_cost_hex": (
                self.normalized_evaluation_cost.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class FactorialPilotSelection:
    """One deterministic bounded factorial design."""

    selected: tuple[FactorialPilotCandidate, ...]
    branch_quotas: tuple[tuple[str, int], ...]
    rank_layer_count: int
    beam_width: int
    decision_scope_sha256: str
    decision_index: int
    anchor_action_sha256s: tuple[str, ...] = ()
    selection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.selected) is not tuple or not self.selected:
            raise ValueError("selected must be a non-empty exact tuple")
        for value in self.selected:
            if type(value) is not FactorialPilotCandidate:
                raise TypeError(
                    "selected must contain exact FactorialPilotCandidate values"
                )
            value.__post_init__()
        require_sha256(self.decision_scope_sha256, "decision_scope_sha256")
        if type(self.decision_index) is not int or self.decision_index <= 0:
            raise ValueError("decision_index must be a positive exact integer")
        if (
            type(self.rank_layer_count) is not int
            or self.rank_layer_count < 2
        ):
            raise ValueError("rank_layer_count must be at least two")
        if type(self.beam_width) is not int or self.beam_width <= 0:
            raise ValueError("beam_width must be a positive exact integer")
        if (
            type(self.branch_quotas) is not tuple
            or not self.branch_quotas
            or tuple(value[0] for value in self.branch_quotas)
            != tuple(sorted({value[0] for value in self.branch_quotas}))
            or any(
                type(branch_id) is not str
                or not branch_id
                or type(quota) is not int
                or quota <= 0
                for branch_id, quota in self.branch_quotas
            )
        ):
            raise ValueError("branch_quotas must be positive and canonical")
        expected = dict(self.branch_quotas)
        observed = Counter(value.branch_id for value in self.selected)
        if observed != expected:
            raise ValueError("selected candidates do not close branch quotas")
        if len(
            {value.phenotype_identity_sha256 for value in self.selected}
        ) != len(self.selected):
            raise ValueError("selected candidates must have unique phenotypes")
        if any(
            not 0 <= value.rank_layer_index < self.rank_layer_count
            for value in self.selected
        ):
            raise ValueError("selected rank layer lies outside the design")
        if (
            type(self.anchor_action_sha256s) is not tuple
            or self.anchor_action_sha256s
            != tuple(sorted(set(self.anchor_action_sha256s)))
            or not set(self.anchor_action_sha256s)
            <= {value.action_sha256 for value in self.selected}
        ):
            raise ValueError(
                "anchor_action_sha256s must be a canonical selected subset"
            )
        for value in self.anchor_action_sha256s:
            require_sha256(value, "anchor_action_sha256s")
        object.__setattr__(
            self,
            "selection_sha256",
            _digest(self._unsigned_record(), domain=_SELECTION_DOMAIN),
        )

    def _coverage_record(self) -> dict[str, object]:
        values = self.selected
        return {
            "branch_count": len({value.branch_id for value in values}),
            "expert_count": len({value.expert_id for value in values}),
            "role_count": len({value.role_id for value in values}),
            "rank_layer_count": len(
                {value.rank_layer_index for value in values}
            ),
            "operator_count": len({value.operator_id for value in values}),
            "parent_count": len(
                {parent for value in values for parent in value.parent_ids}
            ),
            "branch_role_cell_count": len(
                {(value.branch_id, value.role_id) for value in values}
            ),
            "branch_rank_layer_cell_count": len(
                {
                    (value.branch_id, value.rank_layer_index)
                    for value in values
                }
            ),
            "role_rank_layer_cell_count": len(
                {
                    (value.role_id, value.rank_layer_index)
                    for value in values
                }
            ),
        }

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "decision_scope_sha256": self.decision_scope_sha256,
            "decision_index": self.decision_index,
            "branch_quotas": [
                {"branch_id": branch_id, "pilot_slots": quota}
                for branch_id, quota in self.branch_quotas
            ],
            "rank_layer_count": self.rank_layer_count,
            "beam_width": self.beam_width,
            "anchor_action_sha256s": list(self.anchor_action_sha256s),
            "selected": [value.to_record() for value in self.selected],
            "coverage": self._coverage_record(),
            "candidate_outcomes_observed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "selection_sha256": self.selection_sha256,
        }


def factorial_candidates_from_materialized_actions(
    actions: Iterable[MaterializedActionDescriptor],
    *,
    branch_bindings: tuple[ProtectedBranchBinding, ...],
    rank_layer_count: int,
) -> tuple[FactorialPilotCandidate, ...]:
    """Project materialized actions into quantile-layered factor candidates."""

    if (
        type(branch_bindings) is not tuple
        or len(branch_bindings) < 2
        or any(type(value) is not ProtectedBranchBinding for value in branch_bindings)
    ):
        raise TypeError("branch_bindings must contain at least two bindings")
    for value in branch_bindings:
        value.__post_init__()
    if type(rank_layer_count) is not int or rank_layer_count < 2:
        raise ValueError("rank_layer_count must be at least two")
    expert_to_branch: dict[str, str] = {}
    for binding in branch_bindings:
        for expert_id in binding.expert_ids:
            if expert_id in expert_to_branch:
                raise ValueError("one expert cannot belong to two branches")
            expert_to_branch[expert_id] = binding.branch_id

    values = tuple(actions)
    if not values:
        raise ValueError("actions must be non-empty")
    for value in values:
        if type(value) is not MaterializedActionDescriptor:
            raise TypeError("actions must contain exact descriptors")
        value.__post_init__()
    if len({value.action_sha256 for value in values}) != len(values):
        raise ValueError("actions repeat an action identity")

    lanes: dict[str, list[MaterializedActionDescriptor]] = {}
    for value in values:
        if value.expert_id in expert_to_branch:
            lanes.setdefault(value.expert_id, []).append(value)
    absent = set(expert_to_branch) - set(lanes)
    if absent:
        raise ValueError(
            "protected branch names absent proposal experts: "
            + ",".join(sorted(absent))
        )

    projected: list[FactorialPilotCandidate] = []
    for expert_id, lane_values in sorted(lanes.items()):
        lane = tuple(
            sorted(
                lane_values,
                key=lambda value: (value.native_rank, value.action_sha256),
            )
        )
        for position, value in enumerate(lane):
            layer = min(
                rank_layer_count - 1,
                (position * rank_layer_count) // len(lane),
            )
            projected.append(
                FactorialPilotCandidate(
                    action_sha256=value.action_sha256,
                    phenotype_identity_sha256=(
                        value.phenotype_identity_sha256
                    ),
                    branch_id=expert_to_branch[expert_id],
                    expert_id=expert_id,
                    role_id=value.role_id,
                    operator_id=value.operator_id,
                    materialized_lane_rank=position + 1,
                    lane_support_count=len(lane),
                    rank_layer_index=layer,
                    parent_ids=tuple(parent.value for parent in value.parent_ids),
                    normalized_evaluation_cost=(
                        value.normalized_evaluation_cost
                    ),
                )
            )
    return tuple(
        sorted(projected, key=lambda value: value.action_sha256)
    )


def _design_rank_key(
    selected: tuple[FactorialPilotCandidate, ...],
    *,
    decision_scope_sha256: str,
    decision_index: int,
) -> tuple[object, ...]:
    """Return an ascending key for a high-coverage, balanced design."""

    branch_role = {(value.branch_id, value.role_id) for value in selected}
    branch_layer = {
        (value.branch_id, value.rank_layer_index) for value in selected
    }
    role_layer = {
        (value.role_id, value.rank_layer_index) for value in selected
    }
    role_counts = Counter(value.role_id for value in selected)
    layer_counts = Counter(value.rank_layer_index for value in selected)
    parents = {parent for value in selected for parent in value.parent_ids}
    normalized_rank_depth = math.fsum(
        (
            (value.materialized_lane_rank - 1)
            / (value.lane_support_count - 1)
            if value.lane_support_count > 1
            else 0.0
        )
        for value in selected
    )
    tie_sha = _digest(
        {
            "decision_scope_sha256": decision_scope_sha256,
            "decision_index": decision_index,
            "selected_action_sha256s": sorted(
                value.action_sha256 for value in selected
            ),
        },
        domain=_SELECTION_DOMAIN,
    )
    return (
        -len(branch_role),
        -len(branch_layer),
        -len(role_layer),
        -len({value.expert_id for value in selected}),
        -len({value.rank_layer_index for value in selected}),
        -len(parents),
        -len({value.operator_id for value in selected}),
        -normalized_rank_depth,
        sum(value * value for value in role_counts.values()),
        sum(value * value for value in layer_counts.values()),
        math.fsum(value.normalized_evaluation_cost for value in selected),
        tie_sha,
        tuple(sorted(value.action_sha256 for value in selected)),
    )


def select_factorial_pilot_candidates(
    candidates: tuple[FactorialPilotCandidate, ...],
    *,
    branch_quotas: tuple[tuple[str, int], ...],
    rank_layer_count: int,
    beam_width: int,
    decision_scope_sha256: str,
    decision_index: int,
    anchor_action_sha256s: tuple[str, ...] = (),
) -> FactorialPilotSelection:
    """Select or extend a deterministic factorial pilot under exact quotas."""

    require_sha256(decision_scope_sha256, "decision_scope_sha256")
    if type(decision_index) is not int or decision_index <= 0:
        raise ValueError("decision_index must be a positive exact integer")
    if type(candidates) is not tuple or not candidates:
        raise ValueError("candidates must be a non-empty exact tuple")
    for value in candidates:
        if type(value) is not FactorialPilotCandidate:
            raise TypeError("candidates must contain exact candidate views")
        value.__post_init__()
    if len({value.action_sha256 for value in candidates}) != len(candidates):
        raise ValueError("candidate action identities must be unique")
    if (
        type(branch_quotas) is not tuple
        or not branch_quotas
        or tuple(value[0] for value in branch_quotas)
        != tuple(sorted({value[0] for value in branch_quotas}))
        or any(
            type(branch_id) is not str
            or not branch_id
            or type(quota) is not int
            or quota <= 0
            for branch_id, quota in branch_quotas
        )
    ):
        raise ValueError("branch_quotas must be positive and canonical")
    if type(rank_layer_count) is not int or rank_layer_count < 2:
        raise ValueError("rank_layer_count must be at least two")
    if type(beam_width) is not int or beam_width <= 0:
        raise ValueError("beam_width must be a positive exact integer")
    quota_by_branch = dict(branch_quotas)
    candidate_branches = {value.branch_id for value in candidates}
    if candidate_branches != set(quota_by_branch):
        raise ValueError("candidate and quota branch sets must match exactly")
    for branch_id, quota in branch_quotas:
        unique = {
            value.phenotype_identity_sha256
            for value in candidates
            if value.branch_id == branch_id
        }
        if len(unique) < quota:
            raise ValueError(
                f"branch {branch_id} lacks a phenotype-unique quota witness"
            )

    if (
        type(anchor_action_sha256s) is not tuple
        or anchor_action_sha256s
        != tuple(sorted(set(anchor_action_sha256s)))
    ):
        raise ValueError("anchor_action_sha256s must be a canonical tuple")
    candidate_by_action = {
        value.action_sha256: value for value in candidates
    }
    if not set(anchor_action_sha256s) <= set(candidate_by_action):
        raise ValueError("factorial pilot anchor is absent from candidates")
    anchors = tuple(
        candidate_by_action[value] for value in anchor_action_sha256s
    )
    if len(
        {value.phenotype_identity_sha256 for value in anchors}
    ) != len(anchors):
        raise ValueError("factorial pilot anchors repeat a phenotype")
    anchor_counts = Counter(value.branch_id for value in anchors)
    if any(
        anchor_counts.get(branch_id, 0) > quota
        for branch_id, quota in branch_quotas
    ):
        raise ValueError("factorial pilot anchors exceed a branch quota")
    tickets = tuple(
        branch_id
        for ordinal in range(
            max(
                quota_by_branch[branch_id]
                - anchor_counts.get(branch_id, 0)
                for branch_id in quota_by_branch
            )
        )
        for branch_id, quota in branch_quotas
        if ordinal < quota - anchor_counts.get(branch_id, 0)
    )
    beam: tuple[tuple[FactorialPilotCandidate, ...], ...] = (anchors,)
    candidate_order = tuple(
        sorted(candidates, key=lambda value: value.action_sha256)
    )
    for branch_id in tickets:
        expanded: dict[
            tuple[str, ...],
            tuple[FactorialPilotCandidate, ...],
        ] = {}
        for state in beam:
            used_actions = {value.action_sha256 for value in state}
            used_phenotypes = {
                value.phenotype_identity_sha256 for value in state
            }
            for candidate in candidate_order:
                if (
                    candidate.branch_id != branch_id
                    or candidate.action_sha256 in used_actions
                    or candidate.phenotype_identity_sha256 in used_phenotypes
                ):
                    continue
                successor = (*state, candidate)
                key = tuple(
                    sorted(value.action_sha256 for value in successor)
                )
                expanded.setdefault(key, successor)
        if not expanded:
            raise ValueError(
                "factorial pilot has no cross-branch unique-phenotype witness"
            )
        beam = tuple(
            sorted(
                expanded.values(),
                key=lambda value: _design_rank_key(
                    value,
                    decision_scope_sha256=decision_scope_sha256,
                    decision_index=decision_index,
                ),
            )[:beam_width]
        )

    selected = min(
        beam,
        key=lambda value: _design_rank_key(
            value,
            decision_scope_sha256=decision_scope_sha256,
            decision_index=decision_index,
        ),
    )
    return FactorialPilotSelection(
        selected=selected,
        branch_quotas=branch_quotas,
        rank_layer_count=rank_layer_count,
        beam_width=beam_width,
        decision_scope_sha256=decision_scope_sha256,
        decision_index=decision_index,
        anchor_action_sha256s=anchor_action_sha256s,
    )


@dataclass(frozen=True, slots=True)
class FactorialBranchPilotPolicy:
    """Reserve a portable factorial diagnostic pilot before adaptation."""

    branch_bindings: tuple[ProtectedBranchBinding, ...]
    rank_layer_count: int = 3
    beam_width: int = 1024
    policy_id: str = FACTORIAL_BRANCH_PILOT_POLICY_ID
    policy_version: int = FACTORIAL_BRANCH_PILOT_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.branch_bindings) is not tuple
            or len(self.branch_bindings) < 2
            or any(
                type(value) is not ProtectedBranchBinding
                for value in self.branch_bindings
            )
        ):
            raise TypeError(
                "branch_bindings must contain at least two exact bindings"
            )
        for value in self.branch_bindings:
            value.__post_init__()
        branch_ids = tuple(value.branch_id for value in self.branch_bindings)
        if branch_ids != tuple(sorted(set(branch_ids))):
            raise ValueError("branch bindings must be unique and canonical")
        experts = tuple(
            expert
            for binding in self.branch_bindings
            for expert in binding.expert_ids
        )
        if len(experts) != len(set(experts)):
            raise ValueError("one proposal expert cannot belong to two branches")
        if type(self.rank_layer_count) is not int or self.rank_layer_count < 2:
            raise ValueError("rank_layer_count must be at least two")
        if type(self.beam_width) is not int or self.beam_width <= 0:
            raise ValueError("beam_width must be a positive exact integer")
        if (
            self.policy_id != FACTORIAL_BRANCH_PILOT_POLICY_ID
            or self.policy_version != FACTORIAL_BRANCH_PILOT_POLICY_VERSION
        ):
            raise ValueError("factorial branch policy identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _digest(
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "branch_bindings": [
                        value.to_record() for value in self.branch_bindings
                    ],
                    "rank_layer_count": self.rank_layer_count,
                    "beam_width": self.beam_width,
                    "design": (
                        "exact-branch-quota-round-robin-bounded-beam;"
                        "lexicographic-coverage=branch-role,branch-rank-layer,"
                        "role-rank-layer,expert,rank-layer,parent,operator;"
                        "tail-sentinel=max-normalized-rank-depth-within-equal-"
                        "coverage;balance=role-and-rank-layer;"
                        "global-phenotype-unique"
                    ),
                    "rank_semantics": (
                        "quantile-layer-of-materialized-within-expert-order"
                    ),
                    "candidate_outcomes_observed": False,
                    "interpreted_fields": [
                        "branch_binding",
                        "expert_id",
                        "role_id",
                        "operator_id",
                        "native_rank_as_materialized_lane_order",
                        "parent_ids",
                        "phenotype_identity_sha256",
                        "normalized_evaluation_cost",
                        "action_sha256",
                    ],
                    "forbidden_interpretation": [
                        "workload",
                        "model",
                        "provider",
                        "objective",
                        "prompt",
                        "configuration",
                    ],
                },
                domain=_DEFINITION_DOMAIN,
            ),
        )

    @staticmethod
    def _validate_cutoff(
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> None:
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be an exact residual request")
        request.__post_init__()
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
        expert_ids = tuple(value.expert_id for value in proposals)
        if expert_ids != tuple(sorted(set(expert_ids))):
            raise ValueError("proposal batches must use canonical expert IDs")

    async def require(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionAllocationRequirement:
        self.__post_init__()
        self._validate_cutoff(request, proposals)
        pilot_slots = sum(
            value.pilot_slots for value in self.branch_bindings
        )
        if pilot_slots > request.evaluation_slots:
            raise ValueError("factorial pilot floors exceed evaluation capacity")
        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )
        if len({value.action_sha256 for value in actions}) != len(actions):
            raise ValueError("proposal union repeats an action identity")
        unique_market_phenotypes = {
            value.phenotype_identity_sha256 for value in actions
        }
        if len(unique_market_phenotypes) < request.evaluation_slots:
            raise ValueError(
                "sealed proposal market lacks a global unique-phenotype "
                "completion witness"
            )
        candidates = factorial_candidates_from_materialized_actions(
            actions,
            branch_bindings=self.branch_bindings,
            rank_layer_count=self.rank_layer_count,
        )
        selection = select_factorial_pilot_candidates(
            candidates,
            branch_quotas=tuple(
                (value.branch_id, value.pilot_slots)
                for value in self.branch_bindings
            ),
            rank_layer_count=self.rank_layer_count,
            beam_width=self.beam_width,
            decision_scope_sha256=request.request_sha256,
            decision_index=request.decision_index,
        )
        selected_phenotypes = {
            value.phenotype_identity_sha256 for value in selection.selected
        }
        residual_seats = request.evaluation_slots - len(selection.selected)
        remaining_unique_phenotypes = (
            unique_market_phenotypes - selected_phenotypes
        )
        if len(remaining_unique_phenotypes) < residual_seats:
            raise ValueError(
                "factorial pilots leave no global phenotype completion witness"
            )
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        selected_action_sha256s = tuple(
            sorted(value.action_sha256 for value in selection.selected)
        )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "candidate_outcomes_observed": False,
                "residual_request_sha256": request.request_sha256,
                "proposal_sha256s": list(proposal_sha256s),
                "factorial_selection": selection.to_record(),
                "market_capacity_certificate": {
                    "evaluation_slots": request.evaluation_slots,
                    "action_count": len(actions),
                    "unique_phenotype_count": len(unique_market_phenotypes),
                    "factorial_pilot_count": len(selection.selected),
                    "residual_seats": residual_seats,
                    "remaining_unique_phenotype_count": len(
                        remaining_unique_phenotypes
                    ),
                    "global_completion_witness": True,
                },
                "ordinary_broker_fills_residual_seats": True,
                "provider_native_rank_available": False,
                "materialized_lane_rank_named_explicitly": True,
                "workload_model_provider_objective_prompt_interpreted": False,
            }
        )
        require_sha256(self.definition_sha256, "definition_sha256")
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=selected_action_sha256s,
            candidate_outcomes_observed=False,
            evidence=evidence,
        )


__all__ = [
    "FACTORIAL_BRANCH_PILOT_POLICY_ID",
    "FACTORIAL_BRANCH_PILOT_POLICY_VERSION",
    "FactorialBranchPilotPolicy",
    "FactorialPilotCandidate",
    "FactorialPilotSelection",
    "factorial_candidates_from_materialized_actions",
    "select_factorial_pilot_candidates",
]

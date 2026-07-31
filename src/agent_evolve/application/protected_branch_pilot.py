"""Outcome-blind pilot floors for opaque proposal-source branches.

The policy protects a small number of materialized actions from each configured
source branch before the ordinary broker fills the rest of the expensive
evaluation slate.  A branch is only an opaque group of proposal-expert IDs.
The policy never interprets a workload, model, provider, objective, prompt, or
configuration field.

This is deliberately a *pilot* primitive rather than a complete learned
router.  It provides the smallest prospective test of a heterogeneous branch
portfolio: preserve source support long enough to obtain real outcomes, then
let the existing broker allocate every unprotected seat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionDescriptor,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json


PROTECTED_BRANCH_PILOT_POLICY_ID = "protected_branch_pilot"
PROTECTED_BRANCH_PILOT_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = b"agent-evolve:protected-branch-pilot-definition:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


@dataclass(frozen=True, slots=True)
class ProtectedBranchBinding:
    """One opaque source branch and its outcome-blind pilot floor."""

    branch_id: str
    expert_ids: tuple[str, ...]
    pilot_slots: int

    def __post_init__(self) -> None:
        _require_token(self.branch_id, name="branch_id")
        if (
            type(self.expert_ids) is not tuple
            or not self.expert_ids
            or self.expert_ids != tuple(sorted(set(self.expert_ids)))
        ):
            raise ValueError("expert_ids must be a non-empty canonical tuple")
        for value in self.expert_ids:
            _require_token(value, name="expert_id")
        if type(self.pilot_slots) is not int or self.pilot_slots <= 0:
            raise ValueError("pilot_slots must be a positive exact integer")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "branch_id": self.branch_id,
            "expert_ids": list(self.expert_ids),
            "pilot_slots": self.pilot_slots,
        }


@dataclass(frozen=True, slots=True)
class ProtectedBranchPilotPolicy:
    """Reserve deterministic, phenotype-unique pilots from sealed branches.

    Candidate outcomes do not exist at this boundary.  Within a branch,
    actions are ordered by native-rank layer and then expert identity, which
    prevents one expert from consuming every pilot merely because its batch is
    concatenated first.  Exact phenotype identity is the only cross-branch
    collision rule.
    """

    branch_bindings: tuple[ProtectedBranchBinding, ...]
    policy_id: str = PROTECTED_BRANCH_PILOT_POLICY_ID
    policy_version: int = PROTECTED_BRANCH_PILOT_POLICY_VERSION
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
        bound_expert_ids = tuple(
            expert_id
            for binding in self.branch_bindings
            for expert_id in binding.expert_ids
        )
        if len(bound_expert_ids) != len(set(bound_expert_ids)):
            raise ValueError("one proposal expert cannot belong to two branches")
        if (
            self.policy_id != PROTECTED_BRANCH_PILOT_POLICY_ID
            or self.policy_version != PROTECTED_BRANCH_PILOT_POLICY_VERSION
        ):
            raise ValueError("protected branch policy identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                _DEFINITION_DOMAIN
                + _canonical_json(
                    {
                        "schema_version": 1,
                        "policy_id": self.policy_id,
                        "policy_version": self.policy_version,
                        "branch_bindings": [
                            value.to_record() for value in self.branch_bindings
                        ],
                        "selection": (
                            "per-branch-native-rank-layer-then-expert-id;"
                            "global-phenotype-unique;ordinary-broker-fills-"
                            "unprotected-seats"
                        ),
                        "capacity_certificate": (
                            "protected-pilot-witness-plus-global-unique-"
                            "phenotype-completion-witness"
                        ),
                        "candidate_outcomes_observed": False,
                        "interpreted_fields": [
                            "expert_id",
                            "native_rank",
                            "phenotype_identity_sha256",
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
                    }
                )
            ).hexdigest(),
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
        proposal_expert_ids = tuple(value.expert_id for value in proposals)
        if proposal_expert_ids != tuple(sorted(set(proposal_expert_ids))):
            raise ValueError("proposal batches must use canonical expert IDs")

    @staticmethod
    def _market(
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> tuple[MaterializedActionDescriptor, ...]:
        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )
        if len({value.action_sha256 for value in actions}) != len(actions):
            raise ValueError("proposal union repeats an action identity")
        return actions

    @staticmethod
    def _branch_order_key(
        action: MaterializedActionDescriptor,
    ) -> tuple[int, str, float, str]:
        return (
            action.native_rank,
            action.expert_id,
            action.normalized_evaluation_cost,
            action.action_sha256,
        )

    async def require(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionAllocationRequirement:
        self.__post_init__()
        self._validate_cutoff(request, proposals)
        if sum(value.pilot_slots for value in self.branch_bindings) > (
            request.evaluation_slots
        ):
            raise ValueError("protected pilot floors exceed evaluation capacity")

        actions = self._market(proposals)
        action_expert_ids = {value.expert_id for value in actions}
        configured_expert_ids = {
            expert_id
            for binding in self.branch_bindings
            for expert_id in binding.expert_ids
        }
        absent = configured_expert_ids - action_expert_ids
        if absent:
            raise ValueError(
                "protected branch names absent proposal experts: "
                + ",".join(sorted(absent))
            )

        unique_market_phenotypes = {
            value.phenotype_identity_sha256 for value in actions
        }
        if len(unique_market_phenotypes) < request.evaluation_slots:
            raise ValueError(
                "sealed proposal market lacks a global unique-phenotype "
                "completion witness"
            )

        selected: list[MaterializedActionDescriptor] = []
        used_phenotypes: set[str] = set()
        branch_rows: list[dict[str, object]] = []
        for binding in self.branch_bindings:
            branch_candidates = sorted(
                (
                    value
                    for value in actions
                    if value.expert_id in binding.expert_ids
                ),
                key=self._branch_order_key,
            )
            branch_selected: list[MaterializedActionDescriptor] = []
            for action in branch_candidates:
                if action.phenotype_identity_sha256 in used_phenotypes:
                    continue
                branch_selected.append(action)
                used_phenotypes.add(action.phenotype_identity_sha256)
                selected.append(action)
                if len(branch_selected) == binding.pilot_slots:
                    break
            if len(branch_selected) != binding.pilot_slots:
                raise ValueError(
                    f"branch {binding.branch_id} lacks a phenotype-unique "
                    "pilot witness under the preceding protected floors"
                )
            branch_rows.append(
                {
                    "branch_id": binding.branch_id,
                    "expert_ids": list(binding.expert_ids),
                    "pilot_slots": binding.pilot_slots,
                    "available_action_count": len(branch_candidates),
                    "available_unique_phenotype_count": len(
                        {
                            value.phenotype_identity_sha256
                            for value in branch_candidates
                        }
                    ),
                    "selected": [
                        {
                            "action_sha256": value.action_sha256,
                            "phenotype_identity_sha256": (
                                value.phenotype_identity_sha256
                            ),
                            "expert_id": value.expert_id,
                            "native_rank": value.native_rank,
                        }
                        for value in branch_selected
                    ],
                }
            )

        remaining_unique_phenotypes = unique_market_phenotypes - used_phenotypes
        residual_seats = request.evaluation_slots - len(selected)
        if len(remaining_unique_phenotypes) < residual_seats:
            raise ValueError(
                "protected pilots leave no global phenotype completion witness"
            )

        selected_action_sha256s = tuple(
            sorted(value.action_sha256 for value in selected)
        )
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "candidate_outcomes_observed": False,
                "residual_request_sha256": request.request_sha256,
                "proposal_sha256s": list(proposal_sha256s),
                "market_capacity_certificate": {
                    "evaluation_slots": request.evaluation_slots,
                    "action_count": len(actions),
                    "unique_phenotype_count": len(unique_market_phenotypes),
                    "protected_pilot_count": len(selected),
                    "residual_seats": residual_seats,
                    "remaining_unique_phenotype_count": len(
                        remaining_unique_phenotypes
                    ),
                    "global_completion_witness": True,
                },
                "branches": branch_rows,
                "unbound_expert_ids": sorted(
                    action_expert_ids - configured_expert_ids
                ),
                "selection_trace": [
                    {
                        "ordinal": ordinal,
                        "allocation_kind": "protected_branch_pilot",
                        "action_sha256": value.action_sha256,
                        "phenotype_identity_sha256": (
                            value.phenotype_identity_sha256
                        ),
                        "expert_id": value.expert_id,
                        "native_rank": value.native_rank,
                        "candidate_outcomes_observed": False,
                    }
                    for ordinal, value in enumerate(
                        sorted(
                            selected,
                            key=lambda item: item.action_sha256,
                        ),
                        start=1,
                    )
                ],
                "ordinary_broker_fills_residual_seats": True,
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
    "PROTECTED_BRANCH_PILOT_POLICY_ID",
    "PROTECTED_BRANCH_PILOT_POLICY_VERSION",
    "ProtectedBranchBinding",
    "ProtectedBranchPilotPolicy",
]

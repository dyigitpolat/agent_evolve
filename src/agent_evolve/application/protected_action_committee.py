"""Protected action-level aggregation of outcome-blind allocation policies.

Complete-slate policies are treated as ballots over a shared sealed action
market. The committee preserves a configurable floor from one forecast-neutral
ballot, samples a precommitted uniform disagreement audit with exact marginal
propensities, and fills the remaining exact-K slate by weighted ballot
consensus. It never observes current candidate outcomes.

This is intentionally a generic application policy. It knows no workload,
objective, model, provider, prompt, or simulator field.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
import math

from agent_evolve.domain.typed_json import freeze_json

from .materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionDescriptor,
    MaterializedSlateFeasibilityPort,
)
from .residual_portfolio_evolution import (
    MaterializedActionAllocationPolicyPort,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)


PROTECTED_ACTION_COMMITTEE_POLICY_ID = "protected_action_committee"
PROTECTED_ACTION_COMMITTEE_POLICY_VERSION = 1
_DEFINITION_DOMAIN = b"agent-evolve:protected-action-committee:v1\x00"
_AUDIT_PRIORITY_DOMAIN = b"agent-evolve:protected-action-committee-audit:v1\x00"
_AUDIT_SUBSET_PRIORITY_DOMAIN = (
    b"agent-evolve:protected-action-committee-audit-subset:v1\x00"
)
_MAX_FEASIBILITY_REPAIR_TRIALS = 100_000


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_sha256(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class ActionCommitteeArmBinding:
    """One outcome-blind complete-slate ballot and its prior trust weight."""

    arm_id: str
    policy: MaterializedActionAllocationPolicyPort = field(
        repr=False,
        compare=False,
    )
    weight: float
    behavior_definition_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or not self.arm_id:
            raise ValueError("arm_id must be non-empty")
        if not isinstance(self.policy, MaterializedActionAllocationPolicyPort):
            raise TypeError("policy must implement its allocation port")
        if type(self.weight) is not float or not math.isfinite(self.weight):
            raise TypeError("weight must be a finite exact float")
        if self.weight <= 0.0:
            raise ValueError("weight must be positive")
        if self.behavior_definition_sha256 is not None:
            _require_sha256(
                self.behavior_definition_sha256,
                "behavior_definition_sha256",
            )

    def identity_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "arm_id": self.arm_id,
            "weight_hex": self.weight.hex(),
            "behavior_definition_sha256": (
                self.policy.definition_sha256
                if self.behavior_definition_sha256 is None
                else self.behavior_definition_sha256
            ),
            "policy": {
                "policy_id": self.policy.policy_id,
                "policy_version": self.policy.policy_version,
                "definition_sha256": self.policy.definition_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class _CommitteeAction:
    action: MaterializedActionDescriptor
    arm_ids: tuple[str, ...]
    normalized_weighted_support: float
    effective_support: float
    support_count: int
    protected_member: bool

    def consensus_key(self) -> tuple[float, float, int, str]:
        return (
            -self.normalized_weighted_support,
            -self.effective_support,
            self.action.native_rank,
            self.action.action_sha256,
        )


@dataclass(frozen=True, slots=True)
class ProtectedActionCommitteePolicy:
    """Aggregate policy ballots into one protected, auditable exact-K slate.

    The randomized audit is uniform over feasible audit subsets. Without an
    additional slate constraint, every audit-pool representative has exact
    marginal inclusion probability ``audit_slots / audit_pool_size``.
    Constraint-filtered subset marginals are enumerated and logged exactly.
    The precommitted seed makes the draw replayable while preserving the
    declared randomization distribution.
    """

    arm_bindings: tuple[ActionCommitteeArmBinding, ...]
    protected_arm_id: str
    protected_slots: int
    audit_slots: int
    audit_seed_sha256: str
    slate_feasibility: MaterializedSlateFeasibilityPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    policy_id: str = PROTECTED_ACTION_COMMITTEE_POLICY_ID
    policy_version: int = PROTECTED_ACTION_COMMITTEE_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.arm_bindings) is not tuple
            or len(self.arm_bindings) < 2
            or any(
                type(value) is not ActionCommitteeArmBinding
                for value in self.arm_bindings
            )
        ):
            raise TypeError(
                "arm_bindings must contain at least two exact bindings"
            )
        for value in self.arm_bindings:
            value.__post_init__()
        arm_ids = tuple(value.arm_id for value in self.arm_bindings)
        if arm_ids != tuple(sorted(set(arm_ids))):
            raise ValueError("arm bindings must be unique and canonical")
        if (
            type(self.protected_arm_id) is not str
            or self.protected_arm_id not in arm_ids
        ):
            raise ValueError("protected_arm_id must name one bound arm")
        for value, name in (
            (self.protected_slots, "protected_slots"),
            (self.audit_slots, "audit_slots"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        _require_sha256(self.audit_seed_sha256, "audit_seed_sha256")
        if self.slate_feasibility is not None:
            if not isinstance(
                self.slate_feasibility,
                MaterializedSlateFeasibilityPort,
            ):
                raise TypeError(
                    "slate_feasibility must implement its application port"
                )
            _require_sha256(
                self.slate_feasibility.definition_sha256,
                "slate_feasibility definition_sha256",
            )
        if (
            self.policy_id != PROTECTED_ACTION_COMMITTEE_POLICY_ID
            or self.policy_version
            != PROTECTED_ACTION_COMMITTEE_POLICY_VERSION
        ):
            raise ValueError("committee policy identity is immutable")
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
                        "arms": [
                            value.identity_record()
                            for value in self.arm_bindings
                        ],
                        "protected_arm_id": self.protected_arm_id,
                        "protected_slots": self.protected_slots,
                        "audit_slots": self.audit_slots,
                        "audit_seed_sha256": self.audit_seed_sha256,
                        "slate_feasibility_definition_sha256": (
                            None
                            if self.slate_feasibility is None
                            else self.slate_feasibility.definition_sha256
                        ),
                        "consensus": (
                            "jaccard_redundancy_adjusted_normalized_weighted_"
                            "ballot_membership_then_effective_support_then_"
                            "native_rank"
                        ),
                        "audit": (
                            "uniform_over_feasible_disagreement_phenotype_"
                            "subsets_with_exact_marginals"
                        ),
                        "candidate_outcomes_observed": False,
                        "workload_model_provider_branches": False,
                    }
                )
            ).hexdigest(),
        )

    @staticmethod
    def _action_market(
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> dict[str, MaterializedActionDescriptor]:
        result: dict[str, MaterializedActionDescriptor] = {}
        for proposal in proposals:
            for action in proposal.actions:
                if action.action_sha256 in result:
                    raise ValueError("proposal union repeats an action identity")
                result[action.action_sha256] = action
        return result

    @staticmethod
    def _phenotype_representatives(
        values: list[_CommitteeAction],
        *,
        normalized_weight_by_arm: dict[str, float],
        effective_vote_by_arm: dict[str, float],
    ) -> list[_CommitteeAction]:
        by_phenotype: dict[str, list[_CommitteeAction]] = {}
        for value in values:
            by_phenotype.setdefault(
                value.action.phenotype_identity_sha256,
                [],
            ).append(value)
        result: list[_CommitteeAction] = []
        for phenotype_values in by_phenotype.values():
            representative = min(
                phenotype_values,
                key=lambda item: item.consensus_key(),
            )
            arm_ids = tuple(
                sorted(
                    {
                        arm_id
                        for value in phenotype_values
                        for arm_id in value.arm_ids
                    }
                )
            )
            result.append(
                _CommitteeAction(
                    action=representative.action,
                    arm_ids=arm_ids,
                    normalized_weighted_support=math.fsum(
                        normalized_weight_by_arm[arm_id]
                        for arm_id in arm_ids
                    ),
                    effective_support=math.fsum(
                        effective_vote_by_arm[arm_id]
                        for arm_id in arm_ids
                    ),
                    support_count=len(arm_ids),
                    protected_member=any(
                        value.protected_member
                        for value in phenotype_values
                    ),
                )
            )
        return result

    def _audit_priority(
        self,
        *,
        request_sha256: str,
        phenotype_identity_sha256: str,
    ) -> str:
        return hashlib.sha256(
            _AUDIT_PRIORITY_DOMAIN
            + bytes.fromhex(self.audit_seed_sha256)
            + bytes.fromhex(request_sha256)
            + bytes.fromhex(phenotype_identity_sha256)
        ).hexdigest()

    def _audit_subset_priority(
        self,
        *,
        request_sha256: str,
        values: tuple[_CommitteeAction, ...],
    ) -> str:
        return hashlib.sha256(
            _AUDIT_SUBSET_PRIORITY_DOMAIN
            + bytes.fromhex(self.audit_seed_sha256)
            + bytes.fromhex(request_sha256)
            + b"".join(
                bytes.fromhex(
                    value.action.phenotype_identity_sha256
                )
                for value in sorted(
                    values,
                    key=lambda item: (
                        item.action.phenotype_identity_sha256,
                        item.action.action_sha256,
                    ),
                )
            )
        ).hexdigest()

    async def require(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionAllocationRequirement:
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
        if self.protected_slots + self.audit_slots > request.evaluation_slots:
            raise ValueError(
                "protected and audit slots exceed the evaluation budget"
            )

        requirements = await asyncio.gather(
            *(
                binding.policy.require(request, proposals)
                for binding in self.arm_bindings
            )
        )
        market = self._action_market(proposals)
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        requirement_by_arm: dict[
            str,
            MaterializedActionAllocationRequirement,
        ] = {}
        for binding, requirement in zip(
            self.arm_bindings,
            requirements,
            strict=True,
        ):
            if type(requirement) is not MaterializedActionAllocationRequirement:
                raise TypeError("committee arm returned a foreign requirement")
            requirement.__post_init__()
            if (
                requirement.policy_id,
                requirement.policy_version,
                requirement.policy_definition_sha256,
            ) != (
                binding.policy.policy_id,
                binding.policy.policy_version,
                binding.policy.definition_sha256,
            ):
                raise ValueError(
                    "committee arm requirement changed its policy identity"
                )
            if (
                requirement.residual_request_sha256
                != request.request_sha256
                or requirement.proposal_sha256s != proposal_sha256s
            ):
                raise ValueError("committee arm requirement changed its cutoff")
            if not set(requirement.required_action_sha256s) <= set(market):
                raise ValueError("committee arm selected outside the market")
            if not requirement.required_action_sha256s:
                raise ValueError("committee arm returned an empty ballot")
            requirement_by_arm[binding.arm_id] = requirement

        phenotype_ballot_by_arm = {
            binding.arm_id: frozenset(
                market[action_id].phenotype_identity_sha256
                for action_id in requirement_by_arm[
                    binding.arm_id
                ].required_action_sha256s
            )
            for binding in self.arm_bindings
        }

        def ballot_similarity(left_arm: str, right_arm: str) -> float:
            left = phenotype_ballot_by_arm[left_arm]
            right = phenotype_ballot_by_arm[right_arm]
            union = left | right
            if not union:
                raise RuntimeError("committee arm has an empty phenotype ballot")
            return len(left & right) / len(union)

        similarity_by_arm = {
            binding.arm_id: {
                other.arm_id: ballot_similarity(
                    binding.arm_id,
                    other.arm_id,
                )
                for other in self.arm_bindings
            }
            for binding in self.arm_bindings
        }
        redundancy_by_arm = {
            binding.arm_id: math.fsum(
                similarity_by_arm[binding.arm_id].values()
            )
            for binding in self.arm_bindings
        }
        effective_weight_by_arm = {
            binding.arm_id: (
                binding.weight / redundancy_by_arm[binding.arm_id]
            )
            for binding in self.arm_bindings
        }
        effective_vote_by_arm = {
            binding.arm_id: 1.0 / redundancy_by_arm[binding.arm_id]
            for binding in self.arm_bindings
        }
        total_weight = math.fsum(effective_weight_by_arm.values())
        normalized_weight_by_arm = {
            arm_id: weight / total_weight
            for arm_id, weight in effective_weight_by_arm.items()
        }
        actions: list[_CommitteeAction] = []
        for action_id, action in sorted(market.items()):
            memberships = tuple(
                binding.arm_id
                for binding in self.arm_bindings
                if action_id
                in requirement_by_arm[
                    binding.arm_id
                ].required_action_sha256s
            )
            if not memberships:
                continue
            weighted_support = math.fsum(
                effective_weight_by_arm[arm_id]
                for arm_id in memberships
            )
            effective_support = math.fsum(
                effective_vote_by_arm[arm_id]
                for arm_id in memberships
            )
            actions.append(
                _CommitteeAction(
                    action=action,
                    arm_ids=memberships,
                    normalized_weighted_support=(
                        weighted_support / total_weight
                    ),
                    effective_support=effective_support,
                    support_count=len(memberships),
                    protected_member=(
                        self.protected_arm_id in memberships
                    ),
                )
            )

        representatives = self._phenotype_representatives(
            actions,
            normalized_weight_by_arm=normalized_weight_by_arm,
            effective_vote_by_arm=effective_vote_by_arm,
        )
        if len(representatives) < request.evaluation_slots:
            raise ValueError(
                "committee ballot union has too few unique phenotypes"
            )
        selected: list[_CommitteeAction] = []
        selected_phenotypes: set[str] = set()
        selection_kind: dict[str, str] = {}

        def append(value: _CommitteeAction, kind: str) -> None:
            phenotype = value.action.phenotype_identity_sha256
            if phenotype in selected_phenotypes:
                raise RuntimeError("committee selected one phenotype twice")
            selected.append(value)
            selected_phenotypes.add(phenotype)
            selection_kind[value.action.action_sha256] = kind

        protected = sorted(
            (
                value
                for value in representatives
                if value.protected_member
            ),
            key=lambda value: value.consensus_key(),
        )
        if len(protected) < self.protected_slots:
            raise ValueError("protected ballot cannot satisfy its floor")
        for value in protected[: self.protected_slots]:
            append(value, "protected_floor")

        for value in sorted(
            representatives,
            key=lambda item: item.consensus_key(),
        ):
            if len(selected) == request.evaluation_slots:
                break
            if (
                value.action.phenotype_identity_sha256
                in selected_phenotypes
            ):
                continue
            append(value, "weighted_consensus")
        if len(selected) != request.evaluation_slots:
            raise RuntimeError(
                "committee failed to construct its deterministic baseline"
            )
        feasibility_replacements: list[dict[str, object]] = []

        def slate_permitted(values: list[_CommitteeAction]) -> bool:
            return (
                self.slate_feasibility is None
                or self.slate_feasibility.permits(
                    tuple(value.action for value in values)
                )
            )

        if not slate_permitted(selected):
            repairable = tuple(
                value
                for value in selected
                if selection_kind[value.action.action_sha256]
                != "protected_floor"
            )
            repair_candidates = tuple(
                sorted(
                    (
                        value
                        for value in representatives
                        if value.action.phenotype_identity_sha256
                        not in selected_phenotypes
                    ),
                    key=lambda value: value.consensus_key(),
                )
            )
            repaired: tuple[
                tuple[_CommitteeAction, ...],
                tuple[_CommitteeAction, ...],
                list[_CommitteeAction],
            ] | None = None
            repair_trials = 0
            for swap_count in range(
                1,
                min(len(repairable), len(repair_candidates)) + 1,
            ):
                for removed in combinations(repairable, swap_count):
                    removed_ids = {
                        value.action.action_sha256 for value in removed
                    }
                    fixed = [
                        value
                        for value in selected
                        if value.action.action_sha256 not in removed_ids
                    ]
                    fixed_phenotypes = {
                        value.action.phenotype_identity_sha256
                        for value in fixed
                    }
                    eligible_candidates = tuple(
                        value
                        for value in repair_candidates
                        if value.action.phenotype_identity_sha256
                        not in fixed_phenotypes
                    )
                    for inserted in combinations(
                        eligible_candidates,
                        swap_count,
                    ):
                        repair_trials += 1
                        if (
                            repair_trials
                            > _MAX_FEASIBILITY_REPAIR_TRIALS
                        ):
                            raise ValueError(
                                "committee feasibility repair exceeded "
                                "its declared trial bound"
                            )
                        trial = [*fixed, *inserted]
                        if len(
                            {
                                value.action.phenotype_identity_sha256
                                for value in trial
                            }
                        ) != request.evaluation_slots:
                            continue
                        if slate_permitted(trial):
                            repaired = (removed, inserted, trial)
                            break
                    if repaired is not None:
                        break
                if repaired is not None:
                    break
            if repaired is None:
                raise ValueError(
                    "committee cannot satisfy final slate feasibility "
                    "without displacing its protected floor"
                )
            removed, inserted, selected = repaired
            for value in removed:
                selection_kind.pop(value.action.action_sha256)
            for value in inserted:
                selection_kind[
                    value.action.action_sha256
                ] = "feasibility_repair"
            selected_phenotypes = {
                value.action.phenotype_identity_sha256
                for value in selected
            }
            feasibility_replacements.append(
                {
                    "removed_action_sha256s": sorted(
                        value.action.action_sha256
                        for value in removed
                    ),
                    "inserted_action_sha256s": sorted(
                        value.action.action_sha256
                        for value in inserted
                    ),
                    "swap_count": len(removed),
                    "trials": repair_trials,
                }
            )
        baseline_action_sha256s = tuple(
            sorted(value.action.action_sha256 for value in selected)
        )

        audit_pool = sorted(
            (
                value
                for value in representatives
                if (
                    value.action.phenotype_identity_sha256
                    not in selected_phenotypes
                    and value.support_count < len(self.arm_bindings)
                )
            ),
            key=lambda value: value.action.action_sha256,
        )
        replaceable = [
            value
            for value in selected
            if selection_kind[value.action.action_sha256]
            != "protected_floor"
        ]
        effective_audit_slots = min(
            self.audit_slots,
            len(audit_pool),
            len(replaceable),
        )
        audit_priority = {
            value.action.action_sha256: self._audit_priority(
                request_sha256=request.request_sha256,
                phenotype_identity_sha256=(
                    value.action.phenotype_identity_sha256
                ),
            )
            for value in audit_pool
        }
        sampled_audit: tuple[_CommitteeAction, ...] = ()
        replaced_baseline: list[_CommitteeAction] = []
        valid_audit_subset_count = 1
        audit_marginal_by_action = {
            value.action.action_sha256: 0.0 for value in audit_pool
        }
        sampled_subset_priority = self._audit_subset_priority(
            request_sha256=request.request_sha256,
            values=(),
        )
        while True:
            replaced_baseline = sorted(
                replaceable,
                key=lambda item: item.consensus_key(),
                reverse=True,
            )[:effective_audit_slots]
            replaced_ids_for_trial = {
                value.action.action_sha256
                for value in replaced_baseline
            }
            fixed_for_trial = [
                value
                for value in selected
                if value.action.action_sha256
                not in replaced_ids_for_trial
            ]
            if self.slate_feasibility is None:
                valid_audit_subset_count = math.comb(
                    len(audit_pool),
                    effective_audit_slots,
                )
                sampled_audit = tuple(
                    sorted(
                        audit_pool,
                        key=lambda item: (
                            audit_priority[
                                item.action.action_sha256
                            ],
                            item.action.action_sha256,
                        ),
                    )[:effective_audit_slots]
                )
                marginal = (
                    0.0
                    if not audit_pool
                    else effective_audit_slots / len(audit_pool)
                )
                audit_marginal_by_action = {
                    value.action.action_sha256: marginal
                    for value in audit_pool
                }
                sampled_subset_priority = (
                    self._audit_subset_priority(
                        request_sha256=request.request_sha256,
                        values=sampled_audit,
                    )
                )
                break

            subset_count = math.comb(
                len(audit_pool),
                effective_audit_slots,
            )
            if subset_count > _MAX_FEASIBILITY_REPAIR_TRIALS:
                raise ValueError(
                    "feasibility-constrained audit subset count exceeds "
                    "its declared exact-enumeration bound"
                )
            valid_subsets = tuple(
                subset
                for subset in combinations(
                    audit_pool,
                    effective_audit_slots,
                )
                if slate_permitted(
                    [*fixed_for_trial, *subset]
                )
            )
            if valid_subsets:
                valid_audit_subset_count = len(valid_subsets)
                sampled_audit = min(
                    valid_subsets,
                    key=lambda subset: (
                        self._audit_subset_priority(
                            request_sha256=request.request_sha256,
                            values=subset,
                        ),
                        tuple(
                            value.action.action_sha256
                            for value in subset
                        ),
                    ),
                )
                sampled_subset_priority = (
                    self._audit_subset_priority(
                        request_sha256=request.request_sha256,
                        values=sampled_audit,
                    )
                )
                audit_marginal_by_action = {
                    value.action.action_sha256: (
                        sum(
                            value in subset
                            for subset in valid_subsets
                        )
                        / valid_audit_subset_count
                    )
                    for value in audit_pool
                }
                break
            if effective_audit_slots == 0:
                raise RuntimeError(
                    "feasible deterministic baseline became infeasible"
                )
            effective_audit_slots -= 1

        audit_probabilities = set(
            audit_marginal_by_action.values()
        )
        common_audit_probability = (
            next(iter(audit_probabilities))
            if len(audit_probabilities) == 1
            else None
        )
        replaced_ids = {
            value.action.action_sha256 for value in replaced_baseline
        }
        selected = [
            value
            for value in selected
            if value.action.action_sha256 not in replaced_ids
        ]
        for value in replaced_baseline:
            selected_phenotypes.remove(
                value.action.phenotype_identity_sha256
            )
            selection_kind.pop(value.action.action_sha256)
        for value in sampled_audit:
            append(value, "randomized_disagreement_audit")
        if len(selected) != request.evaluation_slots:
            raise RuntimeError("committee failed to fill the exact-K slate")
        if not slate_permitted(selected):
            raise RuntimeError("committee emitted an infeasible final slate")

        selected_ids = tuple(
            sorted(value.action.action_sha256 for value in selected)
        )
        selected_order = {
            value.action.action_sha256: ordinal
            for ordinal, value in enumerate(selected, start=1)
        }
        evidence = freeze_json(
            {
                "schema_version": 1,
                "candidate_outcomes_observed": False,
                "workload_model_provider_branches": False,
                "selection_trace": [
                    {
                        "ordinal": selected_order[
                            value.action.action_sha256
                        ],
                        "action_sha256": value.action.action_sha256,
                        "allocation_kind": selection_kind[
                            value.action.action_sha256
                        ],
                        "support_count": value.support_count,
                        "arm_ids": list(value.arm_ids),
                        "audit_marginal_inclusion_probability_hex": (
                            audit_marginal_by_action[
                                value.action.action_sha256
                            ].hex()
                            if value in audit_pool
                            else (0.0).hex()
                        ),
                        "candidate_outcomes_observed": False,
                    }
                    for value in sorted(
                        selected,
                        key=lambda item: selected_order[
                            item.action.action_sha256
                        ],
                    )
                ],
                "committee": {
                    "protected_arm_id": self.protected_arm_id,
                    "configured_protected_slots": self.protected_slots,
                    "configured_audit_slots": self.audit_slots,
                    "effective_audit_slots": effective_audit_slots,
                    "audit_pool_size": len(audit_pool),
                    "audit_marginal_inclusion_probability_hex": (
                        None
                        if common_audit_probability is None
                        else common_audit_probability.hex()
                    ),
                    "valid_audit_subset_count": (
                        valid_audit_subset_count
                    ),
                    "audit_seed_sha256": self.audit_seed_sha256,
                    "sampled_audit_subset_priority_sha256": (
                        sampled_subset_priority
                    ),
                    "audit_randomization_distribution": (
                        "uniform_over_feasible_unselected_disagreement_"
                        "phenotype_subsets_with_fixed_baseline_swaps"
                    ),
                    "deterministic_baseline_action_sha256s": list(
                        baseline_action_sha256s
                    ),
                    "feasibility_replacements": (
                        feasibility_replacements
                    ),
                    "slate_feasibility_definition_sha256": (
                        None
                        if self.slate_feasibility is None
                        else self.slate_feasibility.definition_sha256
                    ),
                    "replaced_baseline_action_sha256s": sorted(replaced_ids),
                    "exact_action_set_propensity": (
                        1.0 / valid_audit_subset_count
                    ).hex(),
                },
                "arms": [
                    {
                        **binding.identity_record(),
                        "phenotype_ballot_size": len(
                            phenotype_ballot_by_arm[binding.arm_id]
                        ),
                        "behavioral_redundancy_hex": (
                            redundancy_by_arm[binding.arm_id].hex()
                        ),
                        "effective_weight_hex": (
                            effective_weight_by_arm[binding.arm_id].hex()
                        ),
                        "pairwise_jaccard_similarity_hex": {
                            other_arm_id: similarity.hex()
                            for other_arm_id, similarity in sorted(
                                similarity_by_arm[
                                    binding.arm_id
                                ].items()
                            )
                        },
                        "requirement_sha256": requirement_by_arm[
                            binding.arm_id
                        ].requirement_sha256,
                        "required_action_sha256s": list(
                            requirement_by_arm[
                                binding.arm_id
                            ].required_action_sha256s
                        ),
                    }
                    for binding in self.arm_bindings
                ],
                "ballot_union": [
                    {
                        "action_sha256": value.action.action_sha256,
                        "phenotype_identity_sha256": (
                            value.action.phenotype_identity_sha256
                        ),
                        "arm_ids": list(value.arm_ids),
                        "normalized_weighted_support_hex": (
                            value.normalized_weighted_support.hex()
                        ),
                        "effective_support_hex": (
                            value.effective_support.hex()
                        ),
                        "support_count": value.support_count,
                        "protected_member": value.protected_member,
                        "audit_eligible": (
                            value in audit_pool
                            and audit_marginal_by_action[
                                value.action.action_sha256
                            ]
                            > 0.0
                        ),
                        "audit_priority_sha256": audit_priority.get(
                            value.action.action_sha256
                        ),
                        "audit_marginal_inclusion_probability_hex": (
                            audit_marginal_by_action[
                                value.action.action_sha256
                            ].hex()
                            if value in audit_pool
                            else (0.0).hex()
                        ),
                        "final_marginal_inclusion_probability_hex": (
                            audit_marginal_by_action[
                                value.action.action_sha256
                            ].hex()
                            if value in audit_pool
                            else (
                                0.0
                                if value.action.action_sha256 in replaced_ids
                                else (
                                    1.0
                                    if value.action.action_sha256
                                    in baseline_action_sha256s
                                    else 0.0
                                )
                            ).hex()
                        ),
                        "selected": (
                            value.action.action_sha256 in selected_order
                        ),
                        "selected_ordinal": selected_order.get(
                            value.action.action_sha256
                        ),
                        "selection_kind": selection_kind.get(
                            value.action.action_sha256
                        ),
                    }
                    for value in sorted(
                        representatives,
                        key=lambda item: item.action.action_sha256,
                    )
                ],
                "exact_k": request.evaluation_slots,
                "unique_phenotype_selection": True,
                "unselected_candidate_outcomes_observed": False,
            }
        )
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=selected_ids,
            candidate_outcomes_observed=False,
            evidence=evidence,
        )


__all__ = [
    "ActionCommitteeArmBinding",
    "PROTECTED_ACTION_COMMITTEE_POLICY_ID",
    "PROTECTED_ACTION_COMMITTEE_POLICY_VERSION",
    "ProtectedActionCommitteePolicy",
]

"""Low-discrepancy protected exploration for the residual action market."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionBrokerRequest,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    MaterializedActionExplorationRequirement,
)
from agent_evolve.application.stratified_cold_start_allocation import (
    StratifiedColdStartAllocationRequest,
    SupportProportionalLowDiscrepancyStratifiedAllocator,
    stratified_proposal_from_materialized_action,
)
from agent_evolve.domain.typed_json import freeze_json


PREQUENTIAL_RESIDUAL_EXPLORATION_ID = (
    "prequential_low_discrepancy_residual_exploration"
)
PREQUENTIAL_RESIDUAL_EXPLORATION_VERSION = 2
PREQUENTIAL_RESIDUAL_EXPLORATION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:prequential-low-discrepancy-residual-exploration:v2;"
    b"inputs=current-materialized-actions-and-strictly-prior-campaign-local-"
    b"outcome-count;"
    b"reference-actions-and-required-reference-phenotypes=excluded-and-left-"
    b"to-conservative-escrow;"
    b"capacity-aware=preserve-one-unreserved-challenger-slot-by-default;"
    b"cold-start=explore-only-capacity-above-unreserved-floor;"
    b"continuing=one-protected-challenger-within-explorable-capacity;"
    b"terminal=zero-information-only-slots-including-cold-start;"
    b"allocation=opaque-expert-support-dhondt-plus-base-two-rank-cycle;"
    b"duplicates=one-deterministic-representative-per-phenotype;"
    b"variation-scale=max-one-parent-arity;"
    b"structural-cell=generic-residual-frontier-cell;"
    b"forbidden-inputs=workload-model-provider-objective-prompt-outcome-value"
).hexdigest()

_SCOPE_DOMAIN = b"agent-evolve:prequential-residual-exploration-scope:v2\x00"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _scope_sha256(
    request: MaterializedActionBrokerRequest,
    actions: tuple[MaterializedActionDescriptor, ...],
) -> str:
    first = actions[0].context
    return hashlib.sha256(
        _SCOPE_DOMAIN
        + _canonical_bytes(
            {
                "schema_version": 1,
                "campaign_scope_sha256": first.campaign_scope_sha256,
                "decision_index": first.decision_index,
                "evaluation_slots": request.evaluation_slots,
                "action_sha256s": [
                    value.action_sha256
                    for value in sorted(
                        request.actions,
                        key=lambda item: item.action_sha256,
                    )
                ],
            }
        )
    ).hexdigest()


def _deduplicate_phenotypes(
    actions: tuple[MaterializedActionDescriptor, ...],
) -> tuple[
    tuple[MaterializedActionDescriptor, ...],
    tuple[tuple[str, tuple[str, ...]], ...],
]:
    grouped: dict[str, list[MaterializedActionDescriptor]] = {}
    for action in actions:
        grouped.setdefault(action.phenotype_identity_sha256, []).append(action)
    retained: list[MaterializedActionDescriptor] = []
    exclusions: list[tuple[str, tuple[str, ...]]] = []
    for phenotype_sha256 in sorted(grouped):
        values = tuple(
            sorted(
                grouped[phenotype_sha256],
                key=lambda value: (
                    value.native_rank,
                    value.expert_id,
                    value.action_sha256,
                ),
            )
        )
        retained.append(values[0])
        exclusions.append(
            (
                phenotype_sha256,
                tuple(value.action_sha256 for value in values[1:]),
            )
        )
    return (
        tuple(
            sorted(
                retained,
                key=lambda value: (
                    value.expert_id,
                    value.native_rank,
                    value.action_sha256,
                ),
            )
        ),
        tuple(exclusions),
    )


@dataclass(frozen=True, slots=True)
class PrequentialLowDiscrepancyResidualExploration:
    """Protect rank coverage without allowing outcomes into the proposal wave.

    Exploration is limited to the capacity remaining after both conservative
    reference escrow and an explicit unreserved challenger floor.  This keeps
    a small K-slate from degenerating into reference-plus-mandatory-exploration
    with no decision left for the learned broker. Terminal decisions never buy
    information because no later decision can use it.
    """

    continuing_exploration_slots: int = 1
    minimum_unreserved_challenger_slots: int = 1
    policy_id: str = PREQUENTIAL_RESIDUAL_EXPLORATION_ID
    policy_version: int = PREQUENTIAL_RESIDUAL_EXPLORATION_VERSION
    definition_sha256: str = (
        PREQUENTIAL_RESIDUAL_EXPLORATION_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if (
            type(self.continuing_exploration_slots) is not int
            or self.continuing_exploration_slots < 0
        ):
            raise ValueError(
                "continuing_exploration_slots must be a non-negative integer"
            )
        if (
            type(self.minimum_unreserved_challenger_slots) is not int
            or self.minimum_unreserved_challenger_slots < 0
        ):
            raise ValueError(
                "minimum_unreserved_challenger_slots must be a "
                "non-negative integer"
            )

    def require(
        self,
        request: MaterializedActionBrokerRequest,
        ledger: MaterializedActionEvidenceLedger,
        required_reference_action_sha256s: tuple[str, ...],
    ) -> MaterializedActionExplorationRequirement:
        if type(request) is not MaterializedActionBrokerRequest:
            raise TypeError("request must be an exact broker request")
        request.__post_init__()
        if type(ledger) is not MaterializedActionEvidenceLedger:
            raise TypeError("ledger must be an exact evidence ledger")
        if required_reference_action_sha256s != tuple(
            sorted(set(required_reference_action_sha256s))
        ):
            raise ValueError(
                "required reference hashes must be unique and canonical"
            )
        action_by_sha256 = {
            value.action_sha256: value for value in request.actions
        }
        try:
            required_references = tuple(
                action_by_sha256[value]
                for value in required_reference_action_sha256s
            )
        except KeyError as error:
            raise ValueError(
                "required reference is outside the broker request"
            ) from error
        if any(not value.reference_action for value in required_references):
            raise ValueError(
                "required reference hashes must identify reference actions"
            )
        decision_index = request.actions[0].context.decision_index
        campaign_scope_sha256 = (
            request.actions[0].context.campaign_scope_sha256
        )
        prior_outcome_count = sum(
            value.action.context.campaign_scope_sha256
            == campaign_scope_sha256
            and value.action.context.decision_index < decision_index
            for value in ledger.outcomes
        )
        cold_start = prior_outcome_count == 0
        references = tuple(
            value for value in request.actions if value.reference_action
        )
        reference_slots = len(required_references)
        challenger_capacity = request.evaluation_slots - reference_slots
        unreserved_challenger_slots = min(
            self.minimum_unreserved_challenger_slots,
            challenger_capacity,
        )
        explorable_capacity = max(
            0,
            challenger_capacity - unreserved_challenger_slots,
        )
        required_reference_phenotypes = {
            value.phenotype_identity_sha256
            for value in required_references
        }
        challengers = tuple(
            value
            for value in request.actions
            if not value.reference_action
            and value.phenotype_identity_sha256
            not in required_reference_phenotypes
        )
        eligible, exclusions = _deduplicate_phenotypes(challengers)
        terminal = (
            request.actions[0].context.phase.value
            == "terminal_conversion"
        )
        desired_slots = (
            0
            if terminal
            else (
                explorable_capacity
                if cold_start
                else min(
                    self.continuing_exploration_slots,
                    explorable_capacity,
                )
            )
        )
        selected_slots = min(desired_slots, len(eligible))
        allocation = None
        action_by_proposal_id: dict[str, MaterializedActionDescriptor] = {}
        if selected_slots:
            projections = []
            for ordinal, action in enumerate(eligible, start=1):
                proposal_id = f"proposal_{ordinal:06d}"
                action_by_proposal_id[proposal_id] = action
                projections.append(
                    stratified_proposal_from_materialized_action(
                        action,
                        proposal_id=proposal_id,
                        variation_scale=max(1, len(action.parent_ids)),
                        structural_cell=(
                            action.context.residual_frontier_cell
                        ),
                    )
                )
            allocation = (
                SupportProportionalLowDiscrepancyStratifiedAllocator().select(
                    StratifiedColdStartAllocationRequest(
                        decision_scope_sha256=_scope_sha256(
                            request,
                            eligible,
                        ),
                        decision_index=decision_index,
                        proposals=tuple(projections),
                        evaluation_slots=selected_slots,
                    )
                )
            )
            required = tuple(
                sorted(
                    action_by_proposal_id[value].action_sha256
                    for value in allocation.selected_proposal_ids
                )
            )
        else:
            required = ()
        return MaterializedActionExplorationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            required_action_sha256s=required,
            prior_outcome_count=prior_outcome_count,
            cold_start=cold_start,
            evidence=freeze_json(
                {
                    "schema_version": 1,
                    "decision_index": decision_index,
                    "reference_action_count": len(references),
                    "reserved_reference_slots": reference_slots,
                    "required_reference_action_sha256s": list(
                        required_reference_action_sha256s
                    ),
                    "required_reference_phenotype_count": len(
                        required_reference_phenotypes
                    ),
                    "challenger_capacity": challenger_capacity,
                    "minimum_unreserved_challenger_slots": (
                        self.minimum_unreserved_challenger_slots
                    ),
                    "unreserved_challenger_slots": (
                        unreserved_challenger_slots
                    ),
                    "explorable_capacity": explorable_capacity,
                    "challenger_action_count": len(challengers),
                    "eligible_unique_phenotype_count": len(eligible),
                    "duplicate_phenotype_exclusions": [
                        {
                            "phenotype_identity_sha256": phenotype_sha256,
                            "excluded_action_sha256s": list(excluded),
                        }
                        for phenotype_sha256, excluded in exclusions
                        if excluded
                    ],
                    "desired_exploration_slots": desired_slots,
                    "selected_exploration_slots": selected_slots,
                    "terminal_information_purchase_suppressed": (
                        terminal
                    ),
                    "allocation": (
                        None if allocation is None else allocation.to_record()
                    ),
                    "outcome_values_consulted": False,
                    "workload_model_provider_fields_present": False,
                }
            ),
        )


__all__ = [
    "PREQUENTIAL_RESIDUAL_EXPLORATION_DEFINITION_SHA256",
    "PREQUENTIAL_RESIDUAL_EXPLORATION_ID",
    "PREQUENTIAL_RESIDUAL_EXPLORATION_VERSION",
    "PrequentialLowDiscrepancyResidualExploration",
]

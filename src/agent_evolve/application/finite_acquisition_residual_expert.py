"""Finite-acquisition adapter for the residual materialized-action market."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.campaign_capacity_recourse import (
    CampaignCapacityRecourseRequest,
)
from agent_evolve.application.finite_acquisition_capacity_recourse import (
    FiniteAcquisitionCapacityProposal,
    FiniteAcquisitionCapacityRecourse,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.residual_portfolio_evolution import (
    DISJOINT_ACTION_EVALUATION_WAVES_V1,
    DisjointActionEvaluationLedger,
    MaterializedActionEvaluation,
    MaterializedActionEvaluationBatch,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json, thaw_json


FINITE_ACQUISITION_RESIDUAL_EXPERT_ID = "numerical_acquisition"
FINITE_ACQUISITION_RESIDUAL_EXPERT_VERSION = 2
FINITE_ACQUISITION_CONTEXT_PROJECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-acquisition-residual-context:v1;"
    b"frontier-cell=global-acquisition;"
    b"parent-cell=global-no-parent;"
    b"archive-relation=unknown-before-evaluation;"
    b"structural-signature=phenotype-identity;"
    b"patch-compatibility=not-applicable;"
    b"forecast-calibration=not-applicable;"
    b"source-distance=zero;memory-dose=zero;"
    b"workload-model-provider-fields=false"
).hexdigest()

_DEFINITION_DOMAIN = b"agent-evolve:finite-acquisition-residual-expert:def:v2\x00"
_EVALUATOR_RECEIPT_DOMAIN = (
    b"agent-evolve:finite-acquisition-residual-expert:evaluation:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _definition_sha256(recourse: FiniteAcquisitionCapacityRecourse) -> str:
    return hashlib.sha256(
        _DEFINITION_DOMAIN
        + _canonical_json(
            {
                "schema_version": 1,
                "expert_id": FINITE_ACQUISITION_RESIDUAL_EXPERT_ID,
                "expert_version": FINITE_ACQUISITION_RESIDUAL_EXPERT_VERSION,
                "recourse_policy": {
                    "policy_id": recourse.policy_id,
                    "policy_version": recourse.policy_version,
                    "definition_sha256": recourse.definition_sha256,
                },
                "context_projection_definition_sha256": (
                    FINITE_ACQUISITION_CONTEXT_PROJECTION_DEFINITION_SHA256
                ),
                "source_role": "reference_backbone",
                "operator_id": "numerical_acquisition",
                "evaluation_cost": "one_normalized_real_evaluation",
                "selection_authority": "downstream_materialized_action_broker",
                "evaluation_wave_semantics": (
                    DISJOINT_ACTION_EVALUATION_WAVES_V1
                ),
                "exactly_once_boundary": "materialized_action",
            }
        )
    ).hexdigest()


def _context(
    request: ResidualPortfolioDecisionRequest,
    *,
    structural_signature_sha256: str,
) -> MaterializedActionContext:
    require_sha256(structural_signature_sha256, "structural_signature_sha256")
    return MaterializedActionContext(
        campaign_scope_sha256=request.campaign_scope_sha256,
        decision_index=request.decision_index,
        phase=request.phase,
        remaining_decisions=request.remaining_decisions,
        remaining_evaluations=request.remaining_evaluations,
        residual_frontier_cell="global_acquisition",
        parent_position_cell="global_no_parent",
        archive_relation_cell="unknown_before_evaluation",
        structural_signature_sha256=structural_signature_sha256,
        patch_compatibility_cell="not_applicable",
        forecast_calibration_cell="not_applicable",
        source_distance_bin=0,
        memory_dose_bin=0,
    )


def _evaluation_receipt_sha256(
    *,
    source_proposal_sha256: str,
    action: MaterializedActionDescriptor,
    candidate: EvolutionCandidate,
) -> str:
    require_sha256(source_proposal_sha256, "source_proposal_sha256")
    if type(action) is not MaterializedActionDescriptor:
        raise TypeError("action must be an exact materialized descriptor")
    action.__post_init__()
    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be an exact EvolutionCandidate")
    EvolutionCandidate.__post_init__(candidate)
    if (
        candidate.candidate_id != action.target_candidate_id
        or candidate.occurrence.configuration_hash != action.configuration_sha256
    ):
        raise ValueError("candidate differs from the materialized action")
    detailed = candidate.detailed_evaluation
    return hashlib.sha256(
        _EVALUATOR_RECEIPT_DOMAIN
        + _canonical_json(
            {
                "schema_version": 1,
                "source_proposal_sha256": source_proposal_sha256,
                "action_sha256": action.action_sha256,
                "candidate_id": candidate.candidate_id.value,
                "configuration_sha256": (
                    candidate.occurrence.configuration_hash
                ),
                "valid": candidate.valid,
                "objectives": [
                    {"metric_id": key, "value_hex": value.hex()}
                    for key, value in candidate.objectives
                ],
                "detailed_evaluation_sha256": (
                    None if detailed is None else detailed.evidence_sha256
                ),
            }
        )
    ).hexdigest()


@dataclass(slots=True)
class FiniteAcquisitionResidualExpert:
    """Expose an existing finite-acquisition recourse as a reference expert.

    The adapter is bound to one exact campaign-capacity request and one common
    campaign cutoff.  Every competing expert receives the same
    ``prior_state_sha256`` even though each expert may also have a distinct
    private source-request hash.  The workload-blind broker parses neither.
    """

    recourse: FiniteAcquisitionCapacityRecourse
    capacity_request: CampaignCapacityRecourseRequest
    prior_state_sha256: str
    expert_id: str = field(
        init=False,
        default=FINITE_ACQUISITION_RESIDUAL_EXPERT_ID,
    )
    expert_version: int = field(
        init=False,
        default=FINITE_ACQUISITION_RESIDUAL_EXPERT_VERSION,
    )
    definition_sha256: str = field(init=False, default="")
    _source_proposals: dict[str, FiniteAcquisitionCapacityProposal] = field(
        init=False,
        default_factory=dict,
    )
    evaluation_wave_semantics: str = field(
        init=False,
        default=DISJOINT_ACTION_EVALUATION_WAVES_V1,
    )
    _evaluation_ledger: DisjointActionEvaluationLedger = field(
        init=False,
        default_factory=DisjointActionEvaluationLedger,
    )

    def __post_init__(self) -> None:
        if type(self.recourse) is not FiniteAcquisitionCapacityRecourse:
            raise TypeError("recourse must be an exact finite-acquisition policy")
        self.recourse.__post_init__()
        if type(self.capacity_request) is not CampaignCapacityRecourseRequest:
            raise TypeError("capacity_request must be exact")
        self.capacity_request.__post_init__()
        require_sha256(self.prior_state_sha256, "prior_state_sha256")
        object.__setattr__(self, "definition_sha256", _definition_sha256(self.recourse))

    def _validate_request(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> int:
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be an exact residual request")
        request.__post_init__()
        expected_slots = request.proposal_slots_for(self.expert_id)
        capacity = self.capacity_request
        if request.campaign_scope_sha256 != capacity.campaign_scope_sha256:
            raise ValueError("residual and capacity requests cross campaign scopes")
        if request.prior_state_sha256 != self.prior_state_sha256:
            raise ValueError(
                "residual prior-state identity differs from the common cutoff"
            )
        if expected_slots != capacity.missing_candidate_occurrences:
            raise ValueError(
                "residual proposal capacity differs from acquisition capacity"
            )
        return expected_slots

    async def propose(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> MaterializedActionProposalBatch:
        expected_slots = self._validate_request(request)
        source = self.recourse.propose(self.capacity_request)
        if len(source.members) != expected_slots:
            raise RuntimeError("finite acquisition did not fill proposal capacity")
        actions = tuple(
            MaterializedActionDescriptor(
                context=_context(
                    request,
                    structural_signature_sha256=(
                        member.phenotype_identity_sha256
                    ),
                ),
                configuration=member.configuration,
                phenotype_identity_sha256=member.phenotype_identity_sha256,
                expert_id=self.expert_id,
                native_rank=member.native_rank,
                parent_ids=(),
                operator_id="numerical_acquisition",
                target_candidate_id=member.target_candidate_id,
                role_id="reference_backbone",
                normalized_evaluation_cost=1.0,
                reference_action=True,
            )
            for member in source.members
        )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "source_capacity_request_sha256": (
                    self.capacity_request.request_sha256
                ),
                "common_prior_state_sha256": self.prior_state_sha256,
                "source_proposal_sha256": source.proposal_sha256,
                "source_policy_definition_sha256": (
                    source.policy_definition_sha256
                ),
                "source_proposal_evidence_sha256": hashlib.sha256(
                    _canonical_json(thaw_json(source.evidence))
                ).hexdigest(),
                "strictly_prior_outcomes_only": True,
                "evaluation_deferred": True,
            }
        )
        batch = MaterializedActionProposalBatch(
            request_sha256=request.request_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            actions=actions,
            evidence=evidence,
        )
        if batch.proposal_sha256 in self._source_proposals:
            raise RuntimeError("finite-acquisition residual proposal was repeated")
        self._source_proposals[batch.proposal_sha256] = source
        return batch

    async def evaluate(
        self,
        proposal: MaterializedActionProposalBatch,
        selected_action_sha256s: tuple[str, ...],
    ) -> MaterializedActionEvaluationBatch:
        if type(proposal) is not MaterializedActionProposalBatch:
            raise TypeError("proposal must be exact")
        proposal.__post_init__()
        if (
            proposal.expert_id,
            proposal.expert_version,
            proposal.expert_definition_sha256,
        ) != (self.expert_id, self.expert_version, self.definition_sha256):
            raise ValueError("proposal is bound to another expert")
        try:
            source = self._source_proposals[proposal.proposal_sha256]
        except KeyError as error:
            raise ValueError("proposal was not issued by this live expert") from error
        if (
            type(selected_action_sha256s) is not tuple
            or not selected_action_sha256s
            or selected_action_sha256s
            != tuple(sorted(set(selected_action_sha256s)))
        ):
            raise ValueError("selected action hashes must be non-empty and canonical")
        action_by_sha256 = {
            value.action_sha256: value for value in proposal.actions
        }
        try:
            selected_actions = tuple(
                action_by_sha256[value] for value in selected_action_sha256s
            )
        except KeyError as error:
            raise ValueError(
                "broker selected outside the acquisition proposal"
            ) from error
        wave = self._evaluation_ledger.reserve(
            proposal,
            selected_action_sha256s,
        )
        native_actions = tuple(
            sorted(selected_actions, key=lambda value: value.native_rank)
        )
        candidates = await self.recourse.evaluate_members(
            source,
            tuple(value.target_candidate_id for value in native_actions),
        )
        candidate_by_id = {value.candidate_id: value for value in candidates}
        evaluations = tuple(
            MaterializedActionEvaluation(
                action=action,
                candidate=candidate_by_id[action.target_candidate_id],
                evaluator_receipt_sha256=_evaluation_receipt_sha256(
                    source_proposal_sha256=source.proposal_sha256,
                    action=action,
                    candidate=candidate_by_id[action.target_candidate_id],
                ),
            )
            for action in selected_actions
        )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "source_proposal_sha256": source.proposal_sha256,
                "selected_native_ranks": [
                    value.native_rank for value in native_actions
                ],
                "real_evaluation_count": len(evaluations),
                "broker_selected_subset_only": True,
                "evaluation_wave": wave.to_record(),
            }
        )
        batch = MaterializedActionEvaluationBatch(
            proposal_sha256=proposal.proposal_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            selected_action_sha256s=selected_action_sha256s,
            evaluations=evaluations,
            evidence=evidence,
        )
        return batch


__all__ = [
    "FINITE_ACQUISITION_CONTEXT_PROJECTION_DEFINITION_SHA256",
    "FINITE_ACQUISITION_RESIDUAL_EXPERT_ID",
    "FINITE_ACQUISITION_RESIDUAL_EXPERT_VERSION",
    "FiniteAcquisitionResidualExpert",
]

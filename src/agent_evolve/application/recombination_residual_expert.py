"""Replay-safe recombination adapter for the residual action market."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.application.agentic_evolution import RewardPolicyBinding
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.portfolio_recombination import (
    EvaluatedPreparedRecombinationMember,
    PortfolioRecombination,
    PortfolioRecombinationWaveRequest,
    PreparedPortfolioRecombinationWave,
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
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.policies.selection.phenotype_recourse import PhenotypeIdentity


RECOMBINATION_RESIDUAL_EXPERT_ID = "recombination"
RECOMBINATION_RESIDUAL_EXPERT_VERSION = 2
RECOMBINATION_RESIDUAL_EXPERT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:recombination-residual-expert:v2;"
    b"proposal=replay-safe-disjoint-exploit-and-coverage-unions;"
    b"materialization=engine-owned-three-way-patch-replay;"
    b"evaluation=none-before-downstream-broker;"
    b"parents=two-observed-source-branches-plus-common-ancestor-proof;"
    b"phenotype=engine-configured-benchmark-facing-identity-port;"
    b"operator=typed-disjoint-recombination;"
    b"selected-subset-waves=real-concurrent-engine-evaluation;"
    b"exactly-once-boundary=materialized-action;"
    b"reservation=fail-closed-before-authoritative-evaluator-await;"
    b"no-pair=exclude-expert-before-market-composition;"
    b"workload-model-provider-branches=false"
).hexdigest()
RECOMBINATION_RESIDUAL_CONTEXT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:recombination-residual-context:v1;"
    b"residual-frontier=pair-policy-nominated;"
    b"parent-position=observed-source-pair;"
    b"archive-relation=unknown-before-evaluation;"
    b"structural-signature=phenotype-identity;"
    b"patch-compatibility=exact-disjoint-union;"
    b"forecast-calibration=not-applicable;"
    b"source-distance=two-parent;memory-dose=zero;"
    b"workload-model-provider-fields=false"
).hexdigest()

_EVALUATOR_RECEIPT_DOMAIN = (
    b"agent-evolve:recombination-residual-evaluation:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _phenotype_identity(
    service: PortfolioRecombination,
    configuration: object,
) -> PhenotypeIdentity:
    identify = getattr(service.engine, "identify_phenotype", None)
    if not callable(identify):
        raise TypeError(
            "recombination residual expert requires the engine's "
            "benchmark-facing phenotype identity boundary"
        )
    identity = identify(configuration)
    if type(identity) is not PhenotypeIdentity:
        raise TypeError("portfolio engine must return an exact PhenotypeIdentity")
    PhenotypeIdentity.__post_init__(identity)
    return identity


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
        residual_frontier_cell="pair_policy_nominated",
        parent_position_cell="observed_source_pair",
        archive_relation_cell="unknown_before_evaluation",
        structural_signature_sha256=structural_signature_sha256,
        patch_compatibility_cell="exact_disjoint_union",
        forecast_calibration_cell="not_applicable",
        source_distance_bin=2,
        memory_dose_bin=0,
    )


def _evaluation_receipt_sha256(
    *,
    prepared: PreparedPortfolioRecombinationWave,
    evaluated: EvaluatedPreparedRecombinationMember,
    action: MaterializedActionDescriptor,
) -> str:
    if type(prepared) is not PreparedPortfolioRecombinationWave:
        raise TypeError("prepared must be exact")
    prepared.__post_init__()
    if type(evaluated) is not EvaluatedPreparedRecombinationMember:
        raise TypeError("evaluated must be exact")
    evaluated.__post_init__()
    if type(action) is not MaterializedActionDescriptor:
        raise TypeError("action must be exact")
    action.__post_init__()
    candidate = evaluated.outcome.candidate
    if (
        candidate is None
        or candidate.candidate_id != action.target_candidate_id
        or candidate.occurrence.configuration_hash != action.configuration_sha256
    ):
        raise ValueError("evaluated recombination differs from residual action")
    detailed = candidate.detailed_evaluation
    return hashlib.sha256(
        _EVALUATOR_RECEIPT_DOMAIN
        + _canonical_json(
            {
                "schema_version": 1,
                "prepared_recombination_sha256": prepared.preparation_sha256,
                "prepared_member_evaluation_sha256": (
                    evaluated.evaluation_sha256
                ),
                "action_sha256": action.action_sha256,
                "candidate_id": candidate.candidate_id.value,
                "configuration_sha256": candidate.occurrence.configuration_hash,
                "valid": candidate.valid,
                "detailed_evaluation_sha256": (
                    None if detailed is None else detailed.evidence_sha256
                ),
            }
        )
    ).hexdigest()


@dataclass(slots=True)
class RecombinationResidualExpert:
    """Expose prepared disjoint unions before any expensive evaluation.

    Construct this expert only when ``available_proposal_count`` is positive.
    ``try_create`` is the convenient fail-soft composition boundary for stages
    whose source branches contain no replay-safe disjoint pair.
    """

    recombination: PortfolioRecombination
    wave: PortfolioRecombinationWaveRequest
    campaign_scope_sha256: str
    prior_state_sha256: str
    reward_binding: RewardPolicyBinding | None = None
    expert_id: str = field(init=False, default=RECOMBINATION_RESIDUAL_EXPERT_ID)
    expert_version: int = field(
        init=False,
        default=RECOMBINATION_RESIDUAL_EXPERT_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=RECOMBINATION_RESIDUAL_EXPERT_DEFINITION_SHA256,
    )
    _prepared: PreparedPortfolioRecombinationWave = field(init=False)
    _proposal: MaterializedActionProposalBatch | None = field(
        init=False,
        default=None,
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
        if type(self.recombination) is not PortfolioRecombination:
            raise TypeError("recombination must be an exact PortfolioRecombination")
        self.recombination.__post_init__()
        if type(self.wave) is not PortfolioRecombinationWaveRequest:
            raise TypeError("wave must be an exact recombination request")
        PortfolioRecombinationWaveRequest.__post_init__(self.wave)
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(self.prior_state_sha256, "prior_state_sha256")
        if self.reward_binding is not None:
            if type(self.reward_binding) is not RewardPolicyBinding:
                raise TypeError("reward_binding must be exact or None")
            RewardPolicyBinding.__post_init__(self.reward_binding)
        prepared = self.recombination.prepare(self.wave)
        if not prepared.target_candidate_ids:
            raise ValueError(
                "recombination expert has no replay-safe proposal; omit it "
                "from this residual market"
            )
        for invocation in prepared.invocations:
            _phenotype_identity(self.recombination, invocation.draft.configuration)
        self._prepared = prepared

    @classmethod
    def try_create(
        cls,
        **kwargs: object,
    ) -> "RecombinationResidualExpert | None":
        try:
            return cls(**kwargs)  # type: ignore[arg-type]
        except ValueError as error:
            if "has no replay-safe proposal" not in str(error):
                raise
            return None

    @property
    def available_proposal_count(self) -> int:
        return len(self._prepared.invocations)

    def _validate_request(self, request: ResidualPortfolioDecisionRequest) -> None:
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be an exact residual request")
        request.__post_init__()
        if request.campaign_scope_sha256 != self.campaign_scope_sha256:
            raise ValueError("residual request crosses campaign scopes")
        if request.prior_state_sha256 != self.prior_state_sha256:
            raise ValueError(
                "residual prior-state identity differs from the common cutoff"
            )
        if request.proposal_slots_for(self.expert_id) != self.available_proposal_count:
            raise ValueError(
                "residual proposal capacity differs from prepared recombinations"
            )

    async def propose(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> MaterializedActionProposalBatch:
        self._validate_request(request)
        if self._proposal is not None:
            raise ValueError("one recombination expert can propose only once")
        actions: list[MaterializedActionDescriptor] = []
        member_evidence: list[dict[str, object]] = []
        for rank, ((role, pair_ids), invocation, materialization) in enumerate(
            zip(
                self._prepared.selected_roles_and_pairs,
                self._prepared.invocations,
                self._prepared.materializations,
                strict=True,
            ),
            start=1,
        ):
            identity = _phenotype_identity(
                self.recombination,
                invocation.draft.configuration,
            )
            action = MaterializedActionDescriptor(
                context=_context(
                    request,
                    structural_signature_sha256=identity.identity_sha256,
                ),
                configuration=freeze_json(invocation.draft.configuration),
                phenotype_identity_sha256=identity.identity_sha256,
                expert_id=self.expert_id,
                native_rank=rank,
                parent_ids=pair_ids,
                operator_id="typed_disjoint_recombination",
                target_candidate_id=invocation.candidate_id,
                role_id=f"{role}_challenger",
                normalized_evaluation_cost=1.0,
                reference_action=False,
            )
            actions.append(action)
            member_evidence.append(
                {
                    "action_sha256": action.action_sha256,
                    "selection_role": role,
                    "pair_ids": [value.value for value in pair_ids],
                    "materialization_receipt_sha256": (
                        materialization.receipt_sha256
                    ),
                }
            )
        proposal = MaterializedActionProposalBatch(
            request_sha256=request.request_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            actions=tuple(actions),
            evidence=freeze_json(
                {
                    "schema_version": 1,
                    "prepared_recombination_sha256": (
                        self._prepared.preparation_sha256
                    ),
                    "common_prior_state_sha256": self.prior_state_sha256,
                    "members": member_evidence,
                    "real_evaluations": 0,
                    "broker_selected_subset_only": True,
                }
            ),
        )
        self._proposal = proposal
        return proposal

    async def evaluate(
        self,
        proposal: MaterializedActionProposalBatch,
        selected_action_sha256s: tuple[str, ...],
    ) -> MaterializedActionEvaluationBatch:
        if self._proposal != proposal:
            raise ValueError("proposal was not issued by this recombination expert")
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
                "broker selected outside recombination proposal"
            ) from error
        wave = self._evaluation_ledger.reserve(
            proposal,
            selected_action_sha256s,
        )
        evaluated = await self.recombination.evaluate_prepared_members(
            self._prepared,
            tuple(value.target_candidate_id for value in selected_actions),
            reward_binding=self.reward_binding,
        )
        evaluated_by_id = {
            value.invocation.candidate_id: value for value in evaluated
        }
        evaluations: list[MaterializedActionEvaluation] = []
        for action in selected_actions:
            member = evaluated_by_id[action.target_candidate_id]
            candidate = member.outcome.candidate
            if candidate is None:  # pragma: no cover - prepared join closes this.
                raise AssertionError("evaluated recombination lost its candidate")
            evaluations.append(
                MaterializedActionEvaluation(
                    action=action,
                    candidate=candidate,
                    evaluator_receipt_sha256=_evaluation_receipt_sha256(
                        prepared=self._prepared,
                        evaluated=member,
                        action=action,
                    ),
                )
            )
        return MaterializedActionEvaluationBatch(
            proposal_sha256=proposal.proposal_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            selected_action_sha256s=selected_action_sha256s,
            evaluations=tuple(evaluations),
            evidence=freeze_json(
                {
                    "schema_version": 1,
                    "prepared_recombination_sha256": (
                        self._prepared.preparation_sha256
                    ),
                    "real_evaluation_count": len(evaluations),
                    "broker_selected_subset_only": True,
                    "evaluation_wave": wave.to_record(),
                }
            ),
        )


__all__ = [
    "RECOMBINATION_RESIDUAL_CONTEXT_DEFINITION_SHA256",
    "RECOMBINATION_RESIDUAL_EXPERT_DEFINITION_SHA256",
    "RECOMBINATION_RESIDUAL_EXPERT_ID",
    "RECOMBINATION_RESIDUAL_EXPERT_VERSION",
    "RecombinationResidualExpert",
]

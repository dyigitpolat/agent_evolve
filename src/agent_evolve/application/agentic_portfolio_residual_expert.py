"""Agentic ranked-portfolio adapter for the residual action market.

The adapter reuses the established LLM portfolio selector and exact finite
option materializer, but stops before real evaluation.  Its sealed candidates
can therefore compete with acquisition, restart, or recombination experts
under the workload/model/provider-blind materialized-action broker.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    RewardPolicyBinding,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.portfolio_evolution import (
    EvaluatedPreparedPortfolioMember,
    PortfolioEvolution,
    PortfolioVariationWaveRequest,
    PreparedPortfolioVariationWave,
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


AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_ID = "agentic_portfolio"
AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_VERSION = 2
AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:agentic-portfolio-residual-expert:v2;"
    b"proposal=one-existing-ranked-finite-portfolio-call;"
    b"materialization=engine-owned-exact-sealed-finite-options;"
    b"evaluation=none-before-downstream-broker;"
    b"phenotype=engine-configured-benchmark-facing-identity-port;"
    b"context=generic-conservative-action-cells;"
    b"operator-arm=finite-option-family;"
    b"memory=selector-visible-cards-with-action-attribution;"
    b"legacy-aggregate-memory-self-credit=forbidden;"
    b"matched-memory-control=forbidden;"
    b"selected-subset-waves=real-concurrent-engine-evaluation;"
    b"exactly-once-boundary=materialized-action;"
    b"reservation=fail-closed-before-authoritative-evaluator-await;"
    b"workload-model-provider-branches=false"
).hexdigest()
AGENTIC_PORTFOLIO_CONTEXT_PROJECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:agentic-portfolio-residual-context:v1;"
    b"residual-frontier=unknown-before-evaluation;"
    b"parent-position=selected-valid-parent;"
    b"archive-relation=unknown-before-evaluation;"
    b"structural-signature=phenotype-identity;"
    b"patch-compatibility=exact-parent-relative-materialization;"
    b"forecast-calibration=unidentified;"
    b"source-distance=zero-until-identified;"
    b"memory-dose=number-of-cited-cards-clipped-to-15;"
    b"workload-model-provider-fields=false"
).hexdigest()

_EVALUATOR_RECEIPT_DOMAIN = (
    b"agent-evolve:agentic-portfolio-residual-evaluation:v1\x00"
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
    service: PortfolioEvolution,
    configuration: object,
) -> PhenotypeIdentity:
    identify = getattr(service.engine, "identify_phenotype", None)
    if not callable(identify):
        raise TypeError(
            "agentic residual expert requires the portfolio engine's "
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
    phenotype_identity_sha256: str,
    supporting_card_count: int,
) -> MaterializedActionContext:
    require_sha256(
        phenotype_identity_sha256,
        "phenotype_identity_sha256",
    )
    if type(supporting_card_count) is not int or supporting_card_count < 0:
        raise ValueError("supporting_card_count must be non-negative")
    return MaterializedActionContext(
        campaign_scope_sha256=request.campaign_scope_sha256,
        decision_index=request.decision_index,
        phase=request.phase,
        remaining_decisions=request.remaining_decisions,
        remaining_evaluations=request.remaining_evaluations,
        residual_frontier_cell="unknown_before_evaluation",
        parent_position_cell="selected_valid_parent",
        archive_relation_cell="unknown_before_evaluation",
        structural_signature_sha256=phenotype_identity_sha256,
        patch_compatibility_cell="exact_parent_relative_materialization",
        forecast_calibration_cell="unidentified",
        source_distance_bin=0,
        memory_dose_bin=min(supporting_card_count, 15),
    )


def _evaluation_receipt_sha256(
    *,
    prepared: PreparedPortfolioVariationWave,
    evaluated: EvaluatedPreparedPortfolioMember,
    action: MaterializedActionDescriptor,
) -> str:
    if type(prepared) is not PreparedPortfolioVariationWave:
        raise TypeError("prepared must be exact")
    prepared.__post_init__()
    if type(evaluated) is not EvaluatedPreparedPortfolioMember:
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
        raise ValueError("evaluated member differs from residual action")
    detailed = candidate.detailed_evaluation
    return hashlib.sha256(
        _EVALUATOR_RECEIPT_DOMAIN
        + _canonical_json(
            {
                "schema_version": 1,
                "prepared_portfolio_sha256": prepared.preparation_sha256,
                "action_sha256": action.action_sha256,
                "materialization_receipt_sha256": (
                    evaluated.materialization.receipt_sha256
                ),
                "portfolio_outcome_sha256": evaluated.receipt.outcome_sha256,
                "candidate_id": candidate.candidate_id.value,
                "configuration_sha256": (
                    candidate.occurrence.configuration_hash
                ),
                "valid": candidate.valid,
                "detailed_evaluation_sha256": (
                    None if detailed is None else detailed.evidence_sha256
                ),
            }
        )
    ).hexdigest()


@dataclass(slots=True)
class AgenticPortfolioResidualExpert:
    """Expose one existing agentic portfolio wave as unevaluated proposals.

    The injected ``prior_state_sha256`` is the common campaign cutoff shared
    with every competing expert.  The distinct portfolio-selection request
    hash still binds this expert's finite action catalog, cards, parent,
    selector context, and LLM-call identity in proposal evidence.
    """

    portfolio: PortfolioEvolution
    wave: PortfolioVariationWaveRequest
    campaign_scope_sha256: str
    prior_state_sha256: str
    reward_binding: RewardPolicyBinding | None = None
    expert_id: str = field(
        init=False,
        default=AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_ID,
    )
    expert_version: int = field(
        init=False,
        default=AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_DEFINITION_SHA256,
    )
    _prepared_by_proposal: dict[str, PreparedPortfolioVariationWave] = field(
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
        if type(self.portfolio) is not PortfolioEvolution:
            raise TypeError("portfolio must be an exact PortfolioEvolution")
        self.portfolio.__post_init__()
        if type(self.wave) is not PortfolioVariationWaveRequest:
            raise TypeError("wave must be an exact PortfolioVariationWaveRequest")
        PortfolioVariationWaveRequest.__post_init__(self.wave)
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(self.prior_state_sha256, "prior_state_sha256")
        if (
            self.wave.memory_credit is not None
            or self.wave.matched_memory_control is not None
        ):
            raise ValueError(
                "residual brokerage forbids legacy wave-level memory credit; "
                "credit selected actions downstream"
            )
        if self.reward_binding is not None:
            if type(self.reward_binding) is not RewardPolicyBinding:
                raise TypeError("reward_binding must be exact or None")
            RewardPolicyBinding.__post_init__(self.reward_binding)
        # Fail during composition, not after one paid selector call.
        _phenotype_identity(self.portfolio, self.wave.parent.configuration)

    def _validate_request(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> int:
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be an exact residual request")
        request.__post_init__()
        expected_slots = request.proposal_slots_for(self.expert_id)
        if request.campaign_scope_sha256 != self.campaign_scope_sha256:
            raise ValueError("residual request crosses campaign scopes")
        if request.prior_state_sha256 != self.prior_state_sha256:
            raise ValueError(
                "residual prior-state identity differs from the common cutoff"
            )
        if expected_slots != self.wave.selection_request.portfolio_size:
            raise ValueError(
                "residual proposal capacity differs from ranked portfolio size"
            )
        return expected_slots

    async def propose(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> MaterializedActionProposalBatch:
        expected_slots = self._validate_request(request)
        prepared = await self.portfolio.prepare(self.wave)
        if len(prepared.invocations) != expected_slots:
            raise RuntimeError("agentic portfolio did not fill proposal capacity")
        actions: list[MaterializedActionDescriptor] = []
        member_evidence: list[dict[str, object]] = []
        for selected, invocation in zip(
            prepared.selection.decision.members,
            prepared.invocations,
            strict=True,
        ):
            identity = _phenotype_identity(
                self.portfolio,
                invocation.draft.configuration,
            )
            action = MaterializedActionDescriptor(
                context=_context(
                    request,
                    phenotype_identity_sha256=identity.identity_sha256,
                    supporting_card_count=len(selected.supporting_card_keys),
                ),
                configuration=freeze_json(invocation.draft.configuration),
                phenotype_identity_sha256=identity.identity_sha256,
                expert_id=self.expert_id,
                native_rank=selected.rank,
                parent_ids=tuple(
                    value.candidate_id for value in invocation.plan.parents
                ),
                operator_id=selected.family,
                target_candidate_id=invocation.candidate_id,
                role_id="llm_ranked_challenger",
                normalized_evaluation_cost=1.0,
                reference_action=False,
            )
            actions.append(action)
            member_evidence.append(
                {
                    "action_sha256": action.action_sha256,
                    "rank": selected.rank,
                    "option_id": selected.option_id,
                    "option_identity_sha256": (
                        selected.option_identity_sha256
                    ),
                    "family": selected.family,
                    "supporting_card_keys": list(
                        selected.supporting_card_keys
                    ),
                    "effect_predictions": [
                        {
                            "metric_id": value.metric_id,
                            "direction": value.direction.value,
                        }
                        for value in selected.effect_predictions
                    ],
                    "design_rationale_sha256": hashlib.sha256(
                        selected.design_rationale.encode(
                            "utf-8",
                            errors="strict",
                        )
                    ).hexdigest(),
                }
            )
        supplemental = prepared.selection.supplemental_audit
        evidence = freeze_json(
            {
                "schema_version": 1,
                "prepared_portfolio_sha256": prepared.preparation_sha256,
                "source_request_sha256": (
                    self.wave.selection_request.request_sha256
                ),
                "common_prior_state_sha256": self.prior_state_sha256,
                "source_decision_sha256": (
                    prepared.selection.decision.decision_sha256
                ),
                "source_selection_policy": {
                    "policy_id": prepared.selection.decision.policy_id,
                    "policy_version": (
                        prepared.selection.decision.policy_version
                    ),
                    "definition_sha256": (
                        prepared.selection.decision.policy_definition_sha256
                    ),
                },
                "selection_telemetry_sha256": (
                    prepared.selection_telemetry_sha256
                ),
                "supplemental_audit_sha256": (
                    None if supplemental is None else supplemental.audit_sha256
                ),
                "context_projection_definition_sha256": (
                    AGENTIC_PORTFOLIO_CONTEXT_PROJECTION_DEFINITION_SHA256
                ),
                "members": member_evidence,
                "strictly_prior_outcomes_only": True,
                "evaluation_deferred": True,
                "legacy_memory_self_credit": False,
            }
        )
        batch = MaterializedActionProposalBatch(
            request_sha256=request.request_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            actions=tuple(actions),
            evidence=evidence,
        )
        if batch.proposal_sha256 in self._prepared_by_proposal:
            raise RuntimeError("agentic residual proposal was repeated")
        self._prepared_by_proposal[batch.proposal_sha256] = prepared
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
            prepared = self._prepared_by_proposal[proposal.proposal_sha256]
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
                "broker selected outside the agentic proposal"
            ) from error
        wave = self._evaluation_ledger.reserve(
            proposal,
            selected_action_sha256s,
        )
        evaluated = await self.portfolio.evaluate_prepared_members(
            prepared,
            tuple(value.target_candidate_id for value in selected_actions),
            reward_binding=self.reward_binding,
        )
        evaluations: list[MaterializedActionEvaluation] = []
        for action, member in zip(
            selected_actions,
            evaluated,
            strict=True,
        ):
            candidate = member.outcome.candidate
            if type(candidate) is not EvolutionCandidate:
                raise ValueError("agentic portfolio evaluation omitted its candidate")
            evaluations.append(
                MaterializedActionEvaluation(
                    action=action,
                    candidate=candidate,
                    evaluator_receipt_sha256=_evaluation_receipt_sha256(
                        prepared=prepared,
                        evaluated=member,
                        action=action,
                    ),
                )
            )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "prepared_portfolio_sha256": prepared.preparation_sha256,
                "selected_native_ranks": [
                    value.native_rank for value in selected_actions
                ],
                "selected_candidate_ids": [
                    value.target_candidate_id.value
                    for value in selected_actions
                ],
                "real_evaluation_count": len(evaluations),
                "broker_selected_subset_only": True,
                "downstream_archive_credit_pending": True,
                "evaluation_wave": wave.to_record(),
            }
        )
        batch = MaterializedActionEvaluationBatch(
            proposal_sha256=proposal.proposal_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            selected_action_sha256s=selected_action_sha256s,
            evaluations=tuple(evaluations),
            evidence=evidence,
        )
        return batch


@dataclass(slots=True)
class AgenticPortfolioBatchResidualExpert:
    """Aggregate concurrent parent-lane portfolios into one proposal expert.

    A campaign commonly owns several parent lanes but the broker should learn
    one transferable ``agentic_portfolio`` arm, not one arm per lane.  This
    adapter prepares every lane concurrently, assigns one contiguous expert
    rank over their union, and later evaluates only the broker-selected
    occurrences grouped by source lane.
    """

    portfolio: PortfolioEvolution
    waves: tuple[PortfolioVariationWaveRequest, ...]
    campaign_scope_sha256: str
    prior_state_sha256: str
    reward_binding: RewardPolicyBinding | None = None
    expert_id: str = field(
        init=False,
        default=AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_ID,
    )
    expert_version: int = field(
        init=False,
        default=AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_DEFINITION_SHA256,
    )
    _prepared_by_proposal: dict[
        str,
        tuple[PreparedPortfolioVariationWave, ...],
    ] = field(init=False, default_factory=dict)
    evaluation_wave_semantics: str = field(
        init=False,
        default=DISJOINT_ACTION_EVALUATION_WAVES_V1,
    )
    _evaluation_ledger: DisjointActionEvaluationLedger = field(
        init=False,
        default_factory=DisjointActionEvaluationLedger,
    )

    def __post_init__(self) -> None:
        if type(self.portfolio) is not PortfolioEvolution:
            raise TypeError("portfolio must be an exact PortfolioEvolution")
        self.portfolio.__post_init__()
        if type(self.waves) is not tuple or not self.waves:
            raise ValueError("waves must be a non-empty exact tuple")
        for wave in self.waves:
            if type(wave) is not PortfolioVariationWaveRequest:
                raise TypeError("waves must contain exact portfolio requests")
            PortfolioVariationWaveRequest.__post_init__(wave)
            if (
                wave.memory_credit is not None
                or wave.matched_memory_control is not None
            ):
                raise ValueError(
                    "residual brokerage forbids legacy wave-level memory credit"
                )
        request_ids = tuple(
            value.selection_request.request_sha256 for value in self.waves
        )
        parent_ids = tuple(value.parent.candidate_id for value in self.waves)
        if (
            len(set(request_ids)) != len(request_ids)
            or len(set(parent_ids)) != len(parent_ids)
        ):
            raise ValueError("batch waves must have unique requests and parents")
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(self.prior_state_sha256, "prior_state_sha256")
        if self.reward_binding is not None:
            if type(self.reward_binding) is not RewardPolicyBinding:
                raise TypeError("reward_binding must be exact or None")
            RewardPolicyBinding.__post_init__(self.reward_binding)
        _phenotype_identity(self.portfolio, self.waves[0].parent.configuration)

    @property
    def proposal_capacity(self) -> int:
        return sum(
            value.selection_request.portfolio_size for value in self.waves
        )

    def _validate_request(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> None:
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be an exact residual request")
        request.__post_init__()
        if request.campaign_scope_sha256 != self.campaign_scope_sha256:
            raise ValueError("residual request crosses campaign scopes")
        if request.prior_state_sha256 != self.prior_state_sha256:
            raise ValueError(
                "residual prior-state identity differs from the common cutoff"
            )
        if request.proposal_slots_for(self.expert_id) != self.proposal_capacity:
            raise ValueError(
                "residual proposal capacity differs from batched portfolio size"
            )

    async def propose(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> MaterializedActionProposalBatch:
        self._validate_request(request)
        raw_prepared = await asyncio.gather(
            *(self.portfolio.prepare(value) for value in self.waves)
        )
        prepared = tuple(raw_prepared)
        if sum(len(value.invocations) for value in prepared) != self.proposal_capacity:
            raise RuntimeError("agentic portfolio batch did not fill capacity")
        actions: list[MaterializedActionDescriptor] = []
        wave_evidence: list[dict[str, object]] = []
        global_rank = 0
        for lane_index, value in enumerate(prepared, start=1):
            lane_members: list[dict[str, object]] = []
            for selected, invocation in zip(
                value.selection.decision.members,
                value.invocations,
                strict=True,
            ):
                global_rank += 1
                identity = _phenotype_identity(
                    self.portfolio,
                    invocation.draft.configuration,
                )
                action = MaterializedActionDescriptor(
                    context=_context(
                        request,
                        phenotype_identity_sha256=identity.identity_sha256,
                        supporting_card_count=len(
                            selected.supporting_card_keys
                        ),
                    ),
                    configuration=freeze_json(invocation.draft.configuration),
                    phenotype_identity_sha256=identity.identity_sha256,
                    expert_id=self.expert_id,
                    native_rank=global_rank,
                    parent_ids=tuple(
                        parent.candidate_id
                        for parent in invocation.plan.parents
                    ),
                    operator_id=selected.family,
                    target_candidate_id=invocation.candidate_id,
                    role_id="llm_ranked_challenger",
                    normalized_evaluation_cost=1.0,
                    reference_action=False,
                )
                actions.append(action)
                lane_members.append(
                    {
                        "action_sha256": action.action_sha256,
                        "expert_native_rank": global_rank,
                        "lane_native_rank": selected.rank,
                        "option_id": selected.option_id,
                        "option_identity_sha256": (
                            selected.option_identity_sha256
                        ),
                        "family": selected.family,
                        "supporting_card_keys": list(
                            selected.supporting_card_keys
                        ),
                        "effect_predictions": [
                            {
                                "metric_id": prediction.metric_id,
                                "direction": prediction.direction.value,
                            }
                            for prediction in selected.effect_predictions
                        ],
                        "design_rationale_sha256": hashlib.sha256(
                            selected.design_rationale.encode(
                                "utf-8",
                                errors="strict",
                            )
                        ).hexdigest(),
                    }
                )
            supplemental = value.selection.supplemental_audit
            wave_evidence.append(
                {
                    "lane_index": lane_index,
                    "parent_candidate_id": value.wave.parent.candidate_id.value,
                    "source_request_sha256": (
                        value.wave.selection_request.request_sha256
                    ),
                    "source_decision_sha256": (
                        value.selection.decision.decision_sha256
                    ),
                    "prepared_portfolio_sha256": value.preparation_sha256,
                    "selection_telemetry_sha256": (
                        value.selection_telemetry_sha256
                    ),
                    "supplemental_audit_sha256": (
                        None
                        if supplemental is None
                        else supplemental.audit_sha256
                    ),
                    "members": lane_members,
                }
            )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "common_prior_state_sha256": self.prior_state_sha256,
                "context_projection_definition_sha256": (
                    AGENTIC_PORTFOLIO_CONTEXT_PROJECTION_DEFINITION_SHA256
                ),
                "source_waves": wave_evidence,
                "concurrent_parent_lane_preparation": True,
                "strictly_prior_outcomes_only": True,
                "evaluation_deferred": True,
                "legacy_memory_self_credit": False,
            }
        )
        batch = MaterializedActionProposalBatch(
            request_sha256=request.request_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            actions=tuple(actions),
            evidence=evidence,
        )
        if batch.proposal_sha256 in self._prepared_by_proposal:
            raise RuntimeError("agentic residual batch proposal was repeated")
        self._prepared_by_proposal[batch.proposal_sha256] = prepared
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
            prepared = self._prepared_by_proposal[proposal.proposal_sha256]
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
                "broker selected outside the agentic batch proposal"
            ) from error
        wave = self._evaluation_ledger.reserve(
            proposal,
            selected_action_sha256s,
        )

        prepared_by_candidate = {
            invocation.candidate_id: (lane_index, value)
            for lane_index, value in enumerate(prepared)
            for invocation in value.invocations
        }
        lane_actions: dict[int, list[MaterializedActionDescriptor]] = {}
        for action in selected_actions:
            try:
                lane_index, _ = prepared_by_candidate[action.target_candidate_id]
            except KeyError as error:
                raise ValueError(
                    "selected action lost its prepared source lane"
                ) from error
            lane_actions.setdefault(lane_index, []).append(action)
        lane_indices = tuple(sorted(lane_actions))
        raw_evaluated = await asyncio.gather(
            *(
                self.portfolio.evaluate_prepared_members(
                    prepared[lane_index],
                    tuple(
                        value.target_candidate_id
                        for value in lane_actions[lane_index]
                    ),
                    reward_binding=self.reward_binding,
                )
                for lane_index in lane_indices
            )
        )
        evaluated_by_candidate = {
            value.materialization.candidate_id: value
            for lane_result in raw_evaluated
            for value in lane_result
        }
        evaluations: list[MaterializedActionEvaluation] = []
        for action in selected_actions:
            try:
                member = evaluated_by_candidate[action.target_candidate_id]
            except KeyError as error:
                raise RuntimeError(
                    "selected agentic action was not evaluated"
                ) from error
            candidate = member.outcome.candidate
            if type(candidate) is not EvolutionCandidate:
                raise ValueError("agentic batch evaluation omitted its candidate")
            _, source_prepared = prepared_by_candidate[
                action.target_candidate_id
            ]
            evaluations.append(
                MaterializedActionEvaluation(
                    action=action,
                    candidate=candidate,
                    evaluator_receipt_sha256=_evaluation_receipt_sha256(
                        prepared=source_prepared,
                        evaluated=member,
                        action=action,
                    ),
                )
            )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "source_preparation_sha256s": [
                    value.preparation_sha256 for value in prepared
                ],
                "selected_native_ranks": [
                    value.native_rank for value in selected_actions
                ],
                "selected_parent_lane_count": len(lane_indices),
                "real_evaluation_count": len(evaluations),
                "broker_selected_subset_only": True,
                "downstream_archive_credit_pending": True,
                "evaluation_wave": wave.to_record(),
            }
        )
        batch = MaterializedActionEvaluationBatch(
            proposal_sha256=proposal.proposal_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            selected_action_sha256s=selected_action_sha256s,
            evaluations=tuple(evaluations),
            evidence=evidence,
        )
        return batch


__all__ = [
    "AGENTIC_PORTFOLIO_CONTEXT_PROJECTION_DEFINITION_SHA256",
    "AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_DEFINITION_SHA256",
    "AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_ID",
    "AGENTIC_PORTFOLIO_RESIDUAL_EXPERT_VERSION",
    "AgenticPortfolioBatchResidualExpert",
    "AgenticPortfolioResidualExpert",
]

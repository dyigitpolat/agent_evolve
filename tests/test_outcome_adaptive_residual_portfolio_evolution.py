from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass, field, replace

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionBrokerRequest,
    MaterializedActionContext,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.materialized_action_constraints import (
    UniquePhenotypeMaterializedSlateFeasibility,
    ZeroMaterializedSlateValue,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionWave,
    OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION,
    OutcomeAdaptiveActionRacingPolicy,
)
from agent_evolve.application.outcome_adaptive_residual_portfolio_evolution import (
    CandidateArchiveAdaptiveActionOutcomeProjector,
    FactorStratifiedAdaptiveMarketProjector,
    OutcomeAdaptiveResidualPhase,
    OutcomeAdaptiveResidualPhaseCommitAck,
    OutcomeAdaptiveResidualPhaseReceipt,
    OutcomeAdaptiveResidualPortfolioEvolution,
    OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256,
    OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_VERSION,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_VERSION,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_VERSION,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256,
    OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_VERSION,
    PortableMaterializedAdaptiveMarketProjector,
    ScoreEnsembleMaterializedAdaptiveMarketProjector,
)
from agent_evolve.application.current_prefix_forecast_opportunity import (
    CurrentPrefixForecastOpportunityPolicy,
)
from agent_evolve.application.calibrated_current_prefix_forecast_opportunity import (
    CalibratedCurrentPrefixForecastOpportunityPolicy,
)
from agent_evolve.application.forecast_geometry_portfolio import (
    ForecastGeometryScenario,
    MaterializedForecastGeometryBatch,
    MaterializedForecastGeometryMember,
)
from agent_evolve.application.protected_current_prefix_forecast_opportunity import (
    ProtectedCurrentPrefixForecastOpportunityChallenger,
)
from agent_evolve.application.same_prefix_paired_audit import (
    FactorStratifiedSamePrefixPairedAuditDesigner,
    ForecastOpportunitySamePrefixShadowDesigner,
    ForecastStratifiedSamePrefixAuditDesigner,
    SamePrefixPairedAuditArm,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
)
from agent_evolve.application.prequential_archive_opportunity_calibration import (
    ArchiveOpportunityActionContext,
    ArchiveOpportunityCalibrationEvidenceRole,
    ArchiveOpportunityCalibrationObservation,
    ArchiveOpportunityCalibrationRequest,
    HierarchicalPrequentialArchiveOpportunityCalibration,
)
from agent_evolve.application.residual_portfolio_evolution import (
    DISJOINT_ACTION_EVALUATION_WAVES_V1,
    MaterializedActionEvaluation,
    MaterializedActionEvaluationBatch,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.outcome_adaptive_phase_journal import (
    DurableJsonlOutcomeAdaptivePhaseCommitter,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass
class _Expert:
    proposal: MaterializedActionProposalBatch
    events: list[str]
    expert_id: str = "test_adaptive_expert"
    expert_version: int = 1
    definition_sha256: str = _sha("test-adaptive-expert")
    evaluation_wave_semantics: str = (
        DISJOINT_ACTION_EVALUATION_WAVES_V1
    )
    propose_count: int = 0
    evaluation_waves: list[tuple[str, ...]] = field(default_factory=list)

    async def propose(self, request):
        assert request.request_sha256 == self.proposal.request_sha256
        self.propose_count += 1
        self.events.append("propose")
        return self.proposal

    async def evaluate(self, proposal, selected_action_sha256s):
        assert proposal == self.proposal
        self.events.append(f"evaluate:{len(self.evaluation_waves) + 1}")
        self.evaluation_waves.append(selected_action_sha256s)
        by_action = {
            value.action_sha256: value for value in proposal.actions
        }
        evaluations = []
        for action_sha256 in selected_action_sha256s:
            action = by_action[action_sha256]
            candidate = EvolutionCandidate(
                occurrence=CandidateOccurrence(
                    candidate_id=action.target_candidate_id,
                    configuration_hash=action.configuration_sha256,
                    configuration_artifact_hash=hashlib.sha256(
                        canonical_typed_json_bytes(action.configuration)
                    ).hexdigest(),
                    proposal_sequence=action.native_rank,
                ),
                configuration=action.configuration,
                objectives=(("cost", float(action.native_rank)),),
                valid=True,
                generation=action.context.decision_index,
                label=f"adaptive_{action.native_rank}",
            )
            evaluations.append(
                MaterializedActionEvaluation(
                    action=action,
                    candidate=candidate,
                    evaluator_receipt_sha256=_sha(
                        f"adaptive-evaluation:{action_sha256}"
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
            evidence=freeze_json({"real_evaluation": True}),
        )

    async def evaluate_counterfactual(
        self,
        proposal,
        selected_action_sha256s,
    ):
        batch = await self.evaluate(proposal, selected_action_sha256s)
        return MaterializedActionEvaluationBatch(
            proposal_sha256=batch.proposal_sha256,
            expert_id=batch.expert_id,
            expert_version=batch.expert_version,
            expert_definition_sha256=batch.expert_definition_sha256,
            selected_action_sha256s=batch.selected_action_sha256s,
            evaluations=batch.evaluations,
            evidence=freeze_json(
                {
                    "real_evaluation": True,
                    "authoritative_budget_registration": False,
                    "counterfactual_assay": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _Utility:
    utility_id: str = "test_adaptive_archive_utility"
    utility_version: int = 1
    definition_sha256: str = _sha("test-adaptive-archive-utility")

    def utility(self, candidates):
        return float(
            sum(
                1.0 / (1.0 + value.objective_map["cost"])
                for value in candidates
                if value.valid
            )
        )

    def marginal_utility(self, candidates, objective_point):
        del candidates
        return float(1.0 / (1.0 + objective_point["cost"]))

    def portfolio_marginal_utility(self, candidates, objective_points):
        del candidates
        return float(
            sum(
                1.0 / (1.0 + value["cost"])
                for value in objective_points
            )
        )


@dataclass(frozen=True, slots=True)
class _SaturatingUtility:
    """Set utility with exact overlap for a causal-outcome unit assay."""

    utility_id: str = "test_saturating_archive_utility"
    utility_version: int = 1
    definition_sha256: str = _sha("test-saturating-archive-utility")

    @staticmethod
    def _quality(cost: float) -> float:
        return float(1.0 / (1.0 + cost))

    def utility(self, candidates):
        return float(
            max(
                (
                    self._quality(value.objective_map["cost"])
                    for value in candidates
                    if value.valid
                ),
                default=0.0,
            )
        )

    def marginal_utility(self, candidates, objective_point):
        base = self.utility(candidates)
        return float(
            max(
                self._quality(objective_point["cost"]) - base,
                0.0,
            )
        )

    def portfolio_marginal_utility(self, candidates, objective_points):
        base = self.utility(candidates)
        augmented = max(
            (
                self._quality(value["cost"])
                for value in objective_points
            ),
            default=base,
        )
        return float(max(augmented - base, 0.0))


@dataclass(frozen=True, slots=True)
class _ForecastGeometryProjection:
    """Test-only sealed geometry with no access to candidate outcomes."""

    cost_offset: float = 0.0
    projection_id: str = "test_adaptive_forecast_geometry"
    projection_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(
                f"test-adaptive-forecast-geometry:{self.cost_offset}"
            ),
        )

    async def project(self, request, proposals):
        members = []
        for proposal in proposals:
            for action in proposal.actions:
                # Every scenario remains better than the imported cost=10
                # parent, so the opportunity policy has authority to act.
                central = float(self.cost_offset + action.native_rank)
                members.append(
                    MaterializedForecastGeometryMember(
                        action_sha256=action.action_sha256,
                        phenotype_identity_sha256=(
                            action.phenotype_identity_sha256
                        ),
                        reliability=1.0,
                        scenarios=(
                            ForecastGeometryScenario(
                                scenario_id="adverse",
                                objective_point=(
                                    ("cost", central + 0.25),
                                ),
                            ),
                            ForecastGeometryScenario(
                                scenario_id="favorable",
                                objective_point=(
                                    ("cost", central - 0.25),
                                ),
                            ),
                            ForecastGeometryScenario(
                                scenario_id="median",
                                objective_point=(("cost", central),),
                            ),
                        ),
                        source_evidence_sha256=_sha(
                            f"test-forecast:{action.action_sha256}"
                        ),
                    )
                )
        return MaterializedForecastGeometryBatch(
            projection_id=self.projection_id,
            projection_version=self.projection_version,
            projection_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            members=tuple(
                sorted(members, key=lambda value: value.action_sha256)
            ),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "candidate_outcomes_observed": False,
                    "test_projection": True,
                }
            ),
        )


@dataclass
class _Committer:
    events: list[str]
    committer_id: str = "test_adaptive_committer"
    committer_version: int = 1
    definition_sha256: str = _sha("test-adaptive-committer")
    phases: list[OutcomeAdaptiveResidualPhase] = field(
        default_factory=list
    )

    async def commit(
        self,
        receipt: OutcomeAdaptiveResidualPhaseReceipt,
    ) -> OutcomeAdaptiveResidualPhaseCommitAck:
        self.phases.append(receipt.phase)
        self.events.append(f"commit:{receipt.phase.value}")
        return OutcomeAdaptiveResidualPhaseCommitAck(
            committer_id=self.committer_id,
            committer_version=self.committer_version,
            committer_definition_sha256=self.definition_sha256,
            phase_receipt_sha256=receipt.receipt_sha256,
            durable=True,
            evidence=freeze_json({"fsync_completed": True}),
        )


@dataclass(frozen=True, slots=True)
class _Scorer:
    scorer_id: str
    reverse: bool = False
    scorer_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"adaptive-scorer:{self.scorer_id}:{self.reverse}"),
        )

    async def score(self, request, proposals):
        actions = tuple(
            value
            for proposal in proposals
            for value in proposal.actions
        )
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            scores=tuple(
                sorted(
                    (
                        MaterializedActionScore(
                            action_sha256=value.action_sha256,
                            value=float(
                                -value.native_rank
                                if self.reverse
                                else value.native_rank
                            ),
                        )
                        for value in actions
                    ),
                    key=lambda value: value.action_sha256,
                )
            ),
            candidate_outcomes_observed=False,
            evidence_sha256=_sha(
                f"adaptive-score-evidence:{self.scorer_id}:"
                f"{request.request_sha256}"
            ),
        )


@dataclass(frozen=True, slots=True)
class _FixedDiagnosticAllocation:
    native_ranks: tuple[int, ...]
    policy_id: str = "test_fixed_diagnostic_allocation"
    policy_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"fixed-diagnostic:{self.native_ranks}"),
        )

    async def require(self, request, proposals):
        actions = {
            value.native_rank: value
            for proposal in proposals
            for value in proposal.actions
        }
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            required_action_sha256s=tuple(
                sorted(
                    actions[value].action_sha256
                    for value in self.native_ranks
                )
            ),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "candidate_outcomes_observed": False,
                    "test_fixed_diagnostic": True,
                }
            ),
        )


def _fixture():
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("adaptive-campaign"),
        prior_state_sha256=_sha("adaptive-prior-state"),
        decision_index=2,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=8,
        evaluation_slots=4,
        expert_proposal_slots=(("test_adaptive_expert", 6),),
        proposal_context=freeze_json({"test": "adaptive-runtime"}),
        reference_escrow_slots=0,
    )
    context = MaterializedActionContext(
        campaign_scope_sha256=request.campaign_scope_sha256,
        decision_index=request.decision_index,
        phase=request.phase,
        remaining_decisions=request.remaining_decisions,
        remaining_evaluations=request.remaining_evaluations,
        residual_frontier_cell="test.frontier",
        parent_position_cell="test.parent",
        archive_relation_cell="unknown_pre_eval",
        structural_signature_sha256=_sha("adaptive-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.trace",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    prior_id = CandidateId("candidate_adaptive_prior")
    actions = tuple(
        MaterializedActionDescriptor(
            context=context,
            configuration=freeze_json({"native_rank": rank}),
            phenotype_identity_sha256=_sha(
                f"adaptive-phenotype:{rank}"
            ),
            expert_id="test_adaptive_expert",
            native_rank=rank,
            parent_ids=((prior_id,) if rank <= 3 else ()),
            operator_id="test_mutation",
            target_candidate_id=CandidateId(
                f"candidate_adaptive_{rank}"
            ),
            role_id="local_exploit",
            normalized_evaluation_cost=1.0,
            reference_action=False,
        )
        for rank in range(1, 7)
    )
    proposal = MaterializedActionProposalBatch(
        request_sha256=request.request_sha256,
        expert_id="test_adaptive_expert",
        expert_version=1,
        expert_definition_sha256=_sha("test-adaptive-expert"),
        actions=actions,
        evidence=freeze_json(
            {
                "candidate_outcomes_observed": False,
                "sealed_once": True,
            }
        ),
    )
    prior_configuration = freeze_json({"prior": True})
    prior = EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=prior_id,
            configuration_hash=typed_json_sha256(prior_configuration),
            configuration_artifact_hash=hashlib.sha256(
                canonical_typed_json_bytes(prior_configuration)
            ).hexdigest(),
            proposal_sequence=0,
        ),
        configuration=prior_configuration,
        objectives=(("cost", 10.0),),
        valid=True,
        generation=0,
        label="adaptive_prior",
    )
    events: list[str] = []
    return request, _Expert(proposal, events), prior, prior_id, events


async def _exercise_live_adaptive_runtime() -> None:
    request, expert, prior, prior_id, events = _fixture()
    committer = _Committer(events)
    result = await OutcomeAdaptiveResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            MaterializedActionEvidenceLedger()
        ),
        racing_policy=OutcomeAdaptiveActionRacingPolicy(
            diagnostic_slots=2,
            randomized_audit_slots=1,
            reference_gain_scale=0.1,
            reference_gain_evidence_sha256=_sha("prior-gain-scale"),
            random_seed=7,
        ),
        market_projector=PortableMaterializedAdaptiveMarketProjector(
            current_run_parent_ids=(prior_id,)
        ),
        outcome_projector=(
            CandidateArchiveAdaptiveActionOutcomeProjector(
                prior_candidates=(prior,),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=(
            UniquePhenotypeMaterializedSlateFeasibility()
        ),
        phase_committer=committer,
        require_durable_phase_commits=True,
    ).run(request)

    assert expert.propose_count == 1
    assert [len(value) for value in expert.evaluation_waves] == [2, 1, 1]
    assert len(
        {
            action
            for wave in expert.evaluation_waves
            for action in wave
        }
    ) == 4
    assert committer.phases == [
        OutcomeAdaptiveResidualPhase.MARKET_FROZEN,
        OutcomeAdaptiveResidualPhase.DIAGNOSTIC_EVALUATED,
        OutcomeAdaptiveResidualPhase.ADAPTIVE_STEP_EVALUATED,
        OutcomeAdaptiveResidualPhase.ADAPTIVE_STEP_EVALUATED,
        OutcomeAdaptiveResidualPhase.FINALIZED,
    ]
    assert events == [
        "propose",
        "commit:market_frozen",
        "evaluate:1",
        "commit:diagnostic_evaluated",
        "evaluate:2",
        "commit:adaptive_step_evaluated",
        "evaluate:3",
        "commit:adaptive_step_evaluated",
        "commit:finalized",
    ]
    assert len(result.result.candidates) == 4
    assert len(result.outcomes) == 4
    assert len(result.set_outcomes) == 3
    assert (
        result.set_outcomes[0].prior_action_evaluation_bindings
        == ()
    )
    assert all(
        value.prior_conditioned_redundancy == 0.0
        and value.prior_conditioned_synergy == 0.0
        for value in result.set_outcomes
    )
    assert result.diagnostic_joint_gain > 0.0
    assert result.continuation_decisions[-1].wave is (
        AdaptiveActionWave.RANDOMIZED_AUDIT
    )
    assert result.allocation_directive.candidate_outcomes_observed is True
    assert result.result.broker_decision.allocation_requirement is (
        result.allocation_directive
    )
    assert len(result.phase_commit_acks) == len(result.phase_receipts)
    assert any(
        value.parent_generated_in_current_run
        for value in result.adaptive_actions
    )
    record = result.to_record(include_evidence=True)
    assert record[
        "current_outcomes_used_only_after_interception"
    ] is True
    assert len(record["set_outcomes"]) == 3
    assert record["schema_version"] == 4
    assert record["method"] == {
        "method_id": "outcome_adaptive_residual_portfolio_evolution",
        "method_version": OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_VERSION,
        "definition_sha256": (
            OUTCOME_ADAPTIVE_RESIDUAL_EVOLUTION_DEFINITION_SHA256
        ),
    }
    assert "paired_audit_executions" not in record
    assert "paired_audit_execution_sha256s" not in record
    assert all(
        "paired_audit_execution"
        not in receipt.to_record(include_evidence=True)["evidence"]
        for receipt in result.phase_receipts
    )


def test_live_adaptive_runtime_is_causal_durable_and_end_to_end():
    asyncio.run(_exercise_live_adaptive_runtime())


def test_protected_forecast_opportunity_is_frozen_and_causal_end_to_end():
    request, expert, prior, prior_id, events = _fixture()
    committer = _Committer(events)
    racing = OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=2,
        randomized_audit_slots=1,
        reference_gain_scale=0.1,
        reference_gain_evidence_sha256=_sha(
            "forecast-prior-gain-scale"
        ),
        random_seed=7,
    )
    challenger = ProtectedCurrentPrefixForecastOpportunityChallenger(
        prior_candidates=(prior,),
        opportunity_policy=CurrentPrefixForecastOpportunityPolicy(
            archive_utility=_Utility(),
            risk_aversion=1.0,
        ),
        geometry_projection=_ForecastGeometryProjection(),
        fallback_policy_id=racing.policy_id,
        fallback_policy_version=racing.policy_version,
        fallback_policy_definition_sha256=racing.definition_sha256,
    )
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=racing,
            market_projector=PortableMaterializedAdaptiveMarketProjector(
                current_run_parent_ids=(prior_id,)
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_Utility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            phase_committer=committer,
            forecast_opportunity_challenger=challenger,
            require_durable_phase_commits=True,
        ).run(request)
    )

    assert [len(value) for value in expert.evaluation_waves] == [2, 1, 1]
    assert (
        result.forecast_opportunity_challenger_definition_sha256
        == challenger.definition_sha256
    )
    assert result.method_definition_sha256 == (
        OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256
    )
    assert all(
        value.policy_id == challenger.challenger_id
        and value.policy_version == challenger.challenger_version
        and value.policy_definition_sha256
        == challenger.definition_sha256
        for value in result.continuation_decisions
    )
    market = result.phase_receipts[0].to_record(
        include_evidence=True
    )["evidence"]
    assert market[
        "forecast_geometry_frozen_before_diagnostic_outcomes"
    ] is True
    assert market["current_candidate_outcomes_observed"] is False
    assert market["forecast_geometry"][
        "candidate_outcomes_observed"
    ] is False
    for decision in result.continuation_decisions:
        evidence = decision.to_record(include_evidence=True)["evidence"]
        assert evidence["selection_source"] == (
            "current_prefix_forecast_opportunity"
        )
        assert evidence[
            "eligible_candidate_outcomes_observed"
        ] is False
        assert evidence["opportunity_ranking"][
            "eligible_candidate_outcomes_observed"
        ] is False
        assert evidence["fallback_decision"][
            "candidate_outcomes_observed"
        ] is True
    record = result.to_record(include_evidence=True)
    assert record["schema_version"] == 6
    assert record["method"] == {
        "method_id": "outcome_adaptive_residual_portfolio_evolution",
        "method_version": (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_VERSION
        ),
        "definition_sha256": (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_OPPORTUNITY_EVOLUTION_DEFINITION_SHA256
        ),
    }
    assert record[
        "current_prefix_candidate_outcomes_used_for_selection"
    ] is True
    assert record["eligible_candidate_outcomes_observed"] is False
    assert record["fallback_preserved_on_abstention"] is True


def test_protected_forecast_opportunity_preserves_fallback_on_abstention():
    request, expert, prior, prior_id, _events = _fixture()
    racing = OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=2,
        randomized_audit_slots=1,
        reference_gain_scale=0.1,
        reference_gain_evidence_sha256=_sha(
            "abstention-prior-gain-scale"
        ),
        random_seed=7,
    )
    challenger = ProtectedCurrentPrefixForecastOpportunityChallenger(
        prior_candidates=(prior,),
        opportunity_policy=CurrentPrefixForecastOpportunityPolicy(
            archive_utility=_SaturatingUtility(),
            risk_aversion=1.0,
        ),
        geometry_projection=_ForecastGeometryProjection(
            cost_offset=20.0
        ),
        fallback_policy_id=racing.policy_id,
        fallback_policy_version=racing.policy_version,
        fallback_policy_definition_sha256=racing.definition_sha256,
    )
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=racing,
            market_projector=PortableMaterializedAdaptiveMarketProjector(
                current_run_parent_ids=(prior_id,)
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_SaturatingUtility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            forecast_opportunity_challenger=challenger,
        ).run(request)
    )

    for decision in result.continuation_decisions:
        evidence = decision.to_record(include_evidence=True)["evidence"]
        assert evidence["selection_source"] == "protected_fallback"
        assert decision.selected_action_sha256s == tuple(
            evidence["fallback_decision"][
                "selected_action_sha256s"
            ]
        )
        assert evidence["opportunity_ranking"][
            "recommended_action_sha256s"
        ] == []


def test_protected_forecast_opportunity_intersects_projected_market():
    request, expert, prior, prior_id, _events = _fixture()
    actions = list(expert.proposal.actions)
    actions[-1] = replace(
        actions[-1],
        phenotype_identity_sha256=(
            actions[-2].phenotype_identity_sha256
        ),
    )
    expert.proposal = replace(
        expert.proposal,
        actions=tuple(actions),
    )
    racing = OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=2,
        randomized_audit_slots=1,
        reference_gain_scale=0.1,
        reference_gain_evidence_sha256=_sha(
            "projected-market-prior-gain-scale"
        ),
        random_seed=7,
    )
    challenger = ProtectedCurrentPrefixForecastOpportunityChallenger(
        prior_candidates=(prior,),
        opportunity_policy=CurrentPrefixForecastOpportunityPolicy(
            archive_utility=_Utility(),
            risk_aversion=1.0,
        ),
        geometry_projection=_ForecastGeometryProjection(),
        fallback_policy_id=racing.policy_id,
        fallback_policy_version=racing.policy_version,
        fallback_policy_definition_sha256=racing.definition_sha256,
    )
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=racing,
            market_projector=PortableMaterializedAdaptiveMarketProjector(
                current_run_parent_ids=(prior_id,)
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_Utility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            forecast_opportunity_challenger=challenger,
        ).run(request)
    )

    assert len(result.adaptive_actions) == 5
    assert [len(value) for value in expert.evaluation_waves] == [2, 1, 1]
    for decision in result.continuation_decisions:
        evidence = decision.to_record(include_evidence=True)["evidence"]
        assert evidence["geometry_member_count"] == 6
        assert evidence["adaptive_market_action_count"] == 5
        assert len(
            evidence[
                "ignored_out_of_market_forecast_action_sha256s"
            ]
        ) == 1
        assert set(
            evidence["opportunity_ranking"]["eligible_action_sha256s"]
        ) <= {
            value.action_sha256 for value in result.adaptive_actions
        }


def test_calibrated_forecast_opportunity_uses_generic_policy_port():
    request, expert, prior, prior_id, _events = _fixture()
    observations = tuple(
        sorted(
            (
                ArchiveOpportunityCalibrationObservation(
                    request=ArchiveOpportunityCalibrationRequest(
                        context=ArchiveOpportunityActionContext(
                            action_sha256=_sha(
                                f"calibration-action:{index}"
                            ),
                            decision_index=2,
                            lane_id="prior.residual_local_exploit",
                            operator_id="test_mutation",
                            native_rank=index,
                            lane_size=6,
                            prior_score=float((7 - index) / 7),
                            parent_generated_in_current_run=True,
                        ),
                        forecast_reliability=1.0,
                        raw_adverse_gain=float(0.08 * index),
                        raw_central_gain=float(0.1 * index),
                        raw_favorable_gain=float(0.12 * index),
                        raw_acquisition_value=float(0.1 * index),
                        prefix_gain=0.2,
                        prefix_action_count=2,
                    ),
                    realized_conditional_gain=float(0.05 * index),
                    decision_sha256=_sha(
                        f"calibration-decision:{index}"
                    ),
                    outcome_sha256=_sha(
                        f"calibration-outcome:{index}"
                    ),
                    evidence_cutoff_ordinal=index,
                )
                for index in range(1, 7)
            ),
            key=lambda value: value.observation_sha256,
        )
    )
    calibration = (
        HierarchicalPrequentialArchiveOpportunityCalibration(
            observations=observations,
            maximum_evidence_cutoff_ordinal=6,
            maximum_support_log_distance=100.0,
        )
    )
    calibrated_policy = (
        CalibratedCurrentPrefixForecastOpportunityPolicy(
            base_policy=CurrentPrefixForecastOpportunityPolicy(
                archive_utility=_Utility(),
                risk_aversion=1.0,
            ),
            calibration=calibration,
        )
    )
    racing = OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=2,
        randomized_audit_slots=1,
        reference_gain_scale=0.1,
        reference_gain_evidence_sha256=_sha(
            "calibrated-prior-gain-scale"
        ),
        random_seed=7,
    )
    challenger = ProtectedCurrentPrefixForecastOpportunityChallenger(
        prior_candidates=(prior,),
        opportunity_policy=calibrated_policy,
        geometry_projection=_ForecastGeometryProjection(),
        fallback_policy_id=racing.policy_id,
        fallback_policy_version=racing.policy_version,
        fallback_policy_definition_sha256=racing.definition_sha256,
    )
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=racing,
            market_projector=PortableMaterializedAdaptiveMarketProjector(
                current_run_parent_ids=(prior_id,)
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_Utility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            forecast_opportunity_challenger=challenger,
        ).run(request)
    )

    assert len(result.continuation_decisions) == 2
    for decision in result.continuation_decisions:
        evidence = decision.to_record(include_evidence=True)["evidence"]
        ranking = evidence["opportunity_ranking"]
        assert ranking["policy"]["policy_id"] == (
            calibrated_policy.policy_id
        )
        assert ranking["eligible_candidate_outcomes_observed"] is False
        assert all(
            value["schema_version"] == 2
            and "raw_acquisition_value_hex" in value
            and value["calibration_result"][
                "eligible_candidate_outcomes_observed"
            ]
            is False
            for value in ranking["scores"]
        )


def test_forecast_opportunity_shadow_measures_exact_fallback_regret():
    request, expert, prior, prior_id, events = _fixture()
    committer = _Committer(events)
    racing = OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=2,
        randomized_audit_slots=1,
        reference_gain_scale=0.1,
        reference_gain_evidence_sha256=_sha(
            "forecast-shadow-prior-gain-scale"
        ),
        random_seed=0,
    )
    challenger = ProtectedCurrentPrefixForecastOpportunityChallenger(
        prior_candidates=(prior,),
        opportunity_policy=CurrentPrefixForecastOpportunityPolicy(
            archive_utility=_Utility(),
            risk_aversion=1.0,
        ),
        geometry_projection=_ForecastGeometryProjection(),
        fallback_policy_id=racing.policy_id,
        fallback_policy_version=racing.policy_version,
        fallback_policy_definition_sha256=racing.definition_sha256,
    )
    shadow = ForecastOpportunitySamePrefixShadowDesigner()
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=racing,
            market_projector=PortableMaterializedAdaptiveMarketProjector(
                current_run_parent_ids=(prior_id,)
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_Utility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            phase_committer=committer,
            forecast_opportunity_challenger=challenger,
            forecast_opportunity_shadow_designer=shadow,
            require_durable_phase_commits=True,
        ).run(request)
    )

    assert [len(value) for value in expert.evaluation_waves] == [2, 1, 1, 1]
    assert len(result.paired_audit_executions) == 1
    execution = result.paired_audit_executions[0]
    assert execution.plan.designer_id == shadow.designer_id
    assert execution.plan.authoritative_arm.value == "exploration"
    final_decision = result.continuation_decisions[-1]
    final_evidence = final_decision.to_record(
        include_evidence=True
    )["evidence"]
    assert execution.plan.exploration_action_sha256 == (
        final_decision.selected_action_sha256s[0]
    )
    assert execution.plan.legacy_action_sha256 == (
        final_evidence["fallback_decision"]["selected_action_sha256s"][0]
    )
    assert execution.plan.common_prefix_action_sha256s == (
        final_decision.prior_selected_action_sha256s
    )
    assert events.index("commit:paired_audit_frozen") < events.index(
        "evaluate:3"
    )
    authoritative_actions = {
        value.action.action_sha256
        for batch in result.result.evaluation_batches
        for value in batch.evaluations
    }
    assert execution.plan.exploration_action_sha256 in authoritative_actions
    assert execution.plan.legacy_action_sha256 not in authoritative_actions
    assert execution.observation.conditional_gain_delta == (
        execution.observation.exploration_set_outcome.conditional_set_gain
        - execution.observation.legacy_set_outcome.conditional_set_gain
    )
    assert len(
        result.forecast_opportunity_calibration_observations
    ) == 1
    calibration_observation = (
        result.forecast_opportunity_calibration_observations[0]
    )
    assert calibration_observation.request.context.action_sha256 == (
        execution.plan.exploration_action_sha256
    )
    assert calibration_observation.realized_conditional_gain == (
        execution.observation.exploration_set_outcome.conditional_set_gain
    )
    assert calibration_observation.request.prefix_gain == (
        execution.observation.exploration_set_outcome
        .prior_selected_set_gain
    )
    assert calibration_observation.request.prefix_action_count == len(
        execution.plan.common_prefix_action_sha256s
    )
    record = result.to_record(include_evidence=True)
    assert record["schema_version"] == 7
    assert record["method"] == {
        "method_id": "outcome_adaptive_residual_portfolio_evolution",
        "method_version": (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_VERSION
        ),
        "definition_sha256": (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_SHADOW_EVOLUTION_DEFINITION_SHA256
        ),
    }
    assert record[
        "paired_audit_physical_extra_real_evaluation_count"
    ] == 1
    assert record[
        "paired_audit_union_admitted_to_authoritative_archive"
    ] is False
    assert len(
        record["forecast_opportunity_calibration_observations"]
    ) == 1


def test_rotated_forecast_stratum_audit_learns_during_abstention_and_quarantines():
    request, expert, prior, prior_id, events = _fixture()
    committer = _Committer(events)
    racing = OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=2,
        randomized_audit_slots=1,
        reference_gain_scale=0.1,
        reference_gain_evidence_sha256=_sha(
            "forecast-stratum-prior-gain-scale"
        ),
        random_seed=0,
    )
    challenger = ProtectedCurrentPrefixForecastOpportunityChallenger(
        prior_candidates=(prior,),
        opportunity_policy=CurrentPrefixForecastOpportunityPolicy(
            archive_utility=_Utility(),
            # Force a raw lower-opportunity collapse while preserving positive
            # central/favorable forecast support for active auditing.
            risk_aversion=100.0,
        ),
        geometry_projection=_ForecastGeometryProjection(),
        fallback_policy_id=racing.policy_id,
        fallback_policy_version=racing.policy_version,
        fallback_policy_definition_sha256=racing.definition_sha256,
    )
    audit = ForecastStratifiedSamePrefixAuditDesigner(random_seed=1)
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=racing,
            market_projector=PortableMaterializedAdaptiveMarketProjector(
                current_run_parent_ids=(prior_id,)
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_Utility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            phase_committer=committer,
            forecast_opportunity_challenger=challenger,
            forecast_opportunity_shadow_designer=audit,
            require_durable_phase_commits=True,
        ).run(request)
    )

    assert [len(value) for value in expert.evaluation_waves] == [2, 1, 1, 1]
    assert len(result.paired_audit_executions) == 1
    execution = result.paired_audit_executions[0]
    assert thaw_json(
        execution.counterfactual_evaluation_batches[0].evidence
    )["authoritative_budget_registration"] is False
    plan_evidence = thaw_json(execution.plan.evidence)
    assert plan_evidence["target_adaptive_step"] == 1
    assert plan_evidence["adaptive_step"] == 1
    assert plan_evidence["selection_source"] == "protected_fallback"
    assert execution.plan.authoritative_arm is SamePrefixPairedAuditArm.LEGACY
    assert execution.plan.exploration_selection_propensity <= 1.0
    assert events.index("commit:paired_audit_frozen") < events.index(
        "evaluate:2"
    )

    authoritative_actions = {
        value.action.action_sha256
        for batch in result.result.evaluation_batches
        for value in batch.evaluations
    }
    assert execution.plan.legacy_action_sha256 in authoritative_actions
    assert execution.plan.exploration_action_sha256 not in authoritative_actions
    assert execution.plan.exploration_action_sha256 not in (
        result.continuation_decisions[-1].selected_action_sha256s
    )

    assert len(result.forecast_opportunity_calibration_observations) == 1
    observation = result.forecast_opportunity_calibration_observations[0]
    assert observation.evidence_role is (
        ArchiveOpportunityCalibrationEvidenceRole
        .SAME_PREFIX_PAIRED_COUNTERFACTUAL
    )
    assert observation.sampling_propensity == (
        execution.plan.exploration_selection_propensity
    )
    assert observation.paired_observation_sha256 == (
        execution.observation.observation_sha256
    )
    assert observation.to_record()["schema_version"] == 2

    record = result.to_record(include_evidence=True)
    assert record["schema_version"] == 8
    assert record["method"] == {
        "method_id": "outcome_adaptive_residual_portfolio_evolution",
        "method_version": (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_VERSION
        ),
        "definition_sha256": (
            OUTCOME_ADAPTIVE_RESIDUAL_FORECAST_STRATIFIED_AUDIT_EVOLUTION_DEFINITION_SHA256
        ),
    }
    assert record["counterfactual_action_quarantine_enforced"] is True
    assert record["counterfactual_quarantined_action_sha256s"] == [
        execution.plan.exploration_action_sha256
    ]


def test_same_prefix_paired_audit_is_frozen_before_extra_real_arm():
    request, expert, prior, prior_id, events = _fixture()
    committer = _Committer(events)
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=OutcomeAdaptiveActionRacingPolicy(
                diagnostic_slots=2,
                randomized_audit_slots=1,
                reference_gain_scale=0.1,
                reference_gain_evidence_sha256=_sha(
                    "paired-prior-gain-scale"
                ),
                trace_alternative_count=6,
                exploration_pool_size=4,
                audit_exploration_probability=0.1,
                stratified_audit_coverage_family_ids=(
                    "source_branch_role",
                ),
                stratified_audit_stratum_family_ids=(
                    "evolutionary_role",
                    "materialized_rank_layer",
                    "source_branch",
                ),
                random_seed=7,
                policy_version=(
                    OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
                ),
            ),
            market_projector=FactorStratifiedAdaptiveMarketProjector(
                delegate=PortableMaterializedAdaptiveMarketProjector(
                    current_run_parent_ids=(prior_id,)
                ),
                source_branch_by_expert=(
                    ("test_adaptive_expert", "branch_a"),
                ),
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_Utility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            phase_committer=committer,
            paired_audit_designer=(
                FactorStratifiedSamePrefixPairedAuditDesigner(
                    random_seed=41
                )
            ),
            require_durable_phase_commits=True,
        ).run(request)
    )

    assert [len(value) for value in expert.evaluation_waves] == [2, 1, 1, 1]
    assert committer.phases == [
        OutcomeAdaptiveResidualPhase.MARKET_FROZEN,
        OutcomeAdaptiveResidualPhase.DIAGNOSTIC_EVALUATED,
        OutcomeAdaptiveResidualPhase.ADAPTIVE_STEP_EVALUATED,
        OutcomeAdaptiveResidualPhase.PAIRED_AUDIT_FROZEN,
        OutcomeAdaptiveResidualPhase.ADAPTIVE_STEP_EVALUATED,
        OutcomeAdaptiveResidualPhase.FINALIZED,
    ]
    assert events.index("commit:paired_audit_frozen") < events.index(
        "evaluate:3"
    )
    assert len(result.result.candidates) == 4
    assert len(result.outcomes) == 4
    assert len(result.paired_audit_executions) == 1
    execution = result.paired_audit_executions[0]
    selected_action_sha256s = {
        value.action.action_sha256
        for batch in result.result.evaluation_batches
        for value in batch.evaluations
    }
    assert execution.plan.authoritative_action_sha256 in (
        selected_action_sha256s
    )
    assert execution.counterfactual_action_sha256 not in (
        selected_action_sha256s
    )
    assert len(result.phase_commit_acks) == len(result.phase_receipts)
    record = result.to_record(include_evidence=True)
    assert record[
        "paired_audit_physical_extra_real_evaluation_count"
    ] == 1
    assert record[
        "paired_audit_union_admitted_to_authoritative_archive"
    ] is False


def test_outcome_blind_allocation_can_protect_the_diagnostic_pilot():
    request, expert, prior, prior_id, _events = _fixture()
    protected = expert.proposal.actions[4:6]
    result = asyncio.run(
        OutcomeAdaptiveResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            racing_policy=OutcomeAdaptiveActionRacingPolicy(
                diagnostic_slots=2,
                randomized_audit_slots=1,
                reference_gain_scale=0.1,
                reference_gain_evidence_sha256=_sha("prior-gain-scale"),
                random_seed=7,
            ),
            market_projector=PortableMaterializedAdaptiveMarketProjector(
                current_run_parent_ids=(prior_id,)
            ),
            outcome_projector=(
                CandidateArchiveAdaptiveActionOutcomeProjector(
                    prior_candidates=(prior,),
                    utility=_Utility(),
                )
            ),
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            diagnostic_allocation_policy=(
                _FixedDiagnosticAllocation((5, 6))
            ),
        ).run(request)
    )

    assert set(expert.evaluation_waves[0]) == {
        value.action_sha256 for value in protected
    }
    assert set(result.diagnostic_decision.selected_action_sha256s) == {
        value.action_sha256 for value in protected
    }
    market_evidence = result.phase_receipts[0].to_record(
        include_evidence=True
    )["evidence"]
    assert set(market_evidence["fixed_action_sha256s"]) == {
        value.action_sha256 for value in protected
    }
    prior = market_evidence["prior_broker_decision"]
    assert (
        prior["allocation_requirement"]["policy"]["policy_id"]
        == "test_fixed_diagnostic_allocation"
    )


def test_set_outcome_separates_fixed_opportunity_from_local_saturation():
    _request, expert, prior, _prior_id, _events = _fixture()
    first_action, second_action = expert.proposal.actions[:2]
    first_batch = asyncio.run(
        expert.evaluate(
            expert.proposal,
            (first_action.action_sha256,),
        )
    )
    second_batch = asyncio.run(
        expert.evaluate(
            expert.proposal,
            (second_action.action_sha256,),
        )
    )
    first = first_batch.evaluations[0]
    second = second_batch.evaluations[0]
    projector = CandidateArchiveAdaptiveActionOutcomeProjector(
        prior_candidates=(prior,),
        utility=_SaturatingUtility(),
    )

    observation = projector.project_set_outcome(
        (first,),
        (second,),
    )

    expected_prior = (1.0 / 2.0) - (1.0 / 11.0)
    expected_fixed = (1.0 / 3.0) - (1.0 / 11.0)
    assert math.isclose(
        observation.prior_selected_set_gain,
        expected_prior,
    )
    assert math.isclose(
        observation.current_wave_fixed_set_gain,
        expected_fixed,
    )
    assert math.isclose(
        observation.augmented_selected_set_gain,
        expected_prior,
    )
    assert observation.conditional_set_gain == 0.0
    assert math.isclose(
        observation.prior_conditioned_redundancy,
        expected_fixed,
    )
    assert observation.prior_conditioned_synergy == 0.0
    assert observation.saturation_fraction == 1.0
    assert observation.to_record()["set_outcome_sha256"] == (
        observation.set_outcome_sha256
    )


def test_adaptive_jsonl_journal_persists_and_rejects_duplicate(tmp_path):
    receipt = OutcomeAdaptiveResidualPhaseReceipt(
        phase=OutcomeAdaptiveResidualPhase.MARKET_FROZEN,
        phase_ordinal=1,
        residual_request_sha256=_sha("journal-request"),
        diagnostic_decision_sha256=_sha("journal-diagnostic"),
        product_sha256s=(_sha("journal-product"),),
        evidence=freeze_json({"sealed_market": True}),
    )
    path = tmp_path / "adaptive_phases.jsonl"
    committer = DurableJsonlOutcomeAdaptivePhaseCommitter(path)
    ack = asyncio.run(committer.commit(receipt))

    assert ack.durable is True
    assert path.read_bytes().endswith(b"\n")
    reloaded = DurableJsonlOutcomeAdaptivePhaseCommitter(path)
    assert reloaded._commit_count == 1
    try:
        asyncio.run(reloaded.commit(receipt))
    except ValueError as error:
        assert "already durably committed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("duplicate receipt must fail closed")


def test_score_ensemble_market_projection_is_outcome_blind_and_bounded():
    request, expert, _prior, prior_id, _events = _fixture()
    broker = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger()
    )
    actions = expert.proposal.actions
    prior_decision = broker.select(
        MaterializedActionBrokerRequest(
            actions=actions,
            evaluation_slots=request.evaluation_slots,
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            reference_escrow_slots=0,
        )
    )
    scorers = (
        _Scorer("authority_a"),
        _Scorer("authority_b", reverse=True),
    )
    projection = asyncio.run(
        ScoreEnsembleMaterializedAdaptiveMarketProjector(
            scorers=scorers,
            scorer_weights=(
                ("authority_a", 0.75),
                ("authority_b", 0.25),
            ),
            current_run_parent_ids=(prior_id,),
        ).project(
            request,
            (expert.proposal,),
            actions,
            prior_decision.scores,
            (),
        )
    )

    assert len(projection) == len(actions)
    assert all(0.0 <= value.prior_score <= 1.0 for value in projection)
    assert len({value.prior_score for value in projection}) > 1
    assert any(value.parent_generated_in_current_run for value in projection)


def test_factor_stratified_market_projection_uses_only_explicit_portable_cells():
    request, expert, _prior, prior_id, _events = _fixture()
    broker = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger()
    )
    actions = expert.proposal.actions
    prior_decision = broker.select(
        MaterializedActionBrokerRequest(
            actions=actions,
            evaluation_slots=request.evaluation_slots,
            slate_value=ZeroMaterializedSlateValue(),
            slate_feasibility=(
                UniquePhenotypeMaterializedSlateFeasibility()
            ),
            reference_escrow_slots=0,
        )
    )
    projector = FactorStratifiedAdaptiveMarketProjector(
        delegate=PortableMaterializedAdaptiveMarketProjector(
            current_run_parent_ids=(prior_id,)
        ),
        source_branch_by_expert=(
            (expert.expert_id, "branch_a"),
            ("unused_deduplicated_expert", "branch_b"),
        ),
        rank_layer_count=3,
    )
    projection = asyncio.run(
        projector.project(
            request,
            (expert.proposal,),
            actions,
            prior_decision.scores,
            (),
        )
    )

    assert len(projection) == len(actions)
    for value in projection:
        cells = {
            cell.family_id: cell.level_id for cell in value.factor_cells
        }
        assert cells["source_branch"] == "branch_a"
        assert cells["evolutionary_role"]
        assert cells["materialized_rank_layer"] in {
            "layer0",
            "layer1",
            "layer2",
        }
        assert cells["source_branch_role"].startswith("cell:")
        assert len(cells["source_branch_role"]) == 69
    assert projector.definition_sha256
    assert projector.state_sha256

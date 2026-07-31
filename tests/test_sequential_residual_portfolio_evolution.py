from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field, replace

import pytest

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.causal_opportunity_portfolio_gate import (
    CausalOpportunityPortfolioRaceGate,
)
from agent_evolve.application.earned_lineage import (
    CandidateProposalProvenance,
    EarnedLineageLedger,
    ProposalLineageRole,
)
from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.forecast_geometry_portfolio import (
    ForecastGeometryPortfolioMode,
    ForecastGeometryPortfolioPolicy,
    ForecastGeometryScenario,
    MaterializedForecastGeometryBatch,
    MaterializedForecastGeometryMember,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.materialized_action_constraints import (
    UniquePhenotypeMaterializedSlateFeasibility,
    ZeroMaterializedSlateValue,
)
from agent_evolve.application.precommitted_portfolio_racing import (
    EvidenceAdaptivePortfolioRaceGate,
    PhaseConditionedPortfolioRacePrior,
    PortfolioRaceDisagreementDesign,
    PortfolioRacePolicyBinding,
    PrecommittedPortfolioRacePlan,
    PrecommittedPortfolioRacePlanner,
    SymmetricDifferencePortfolioRacePolicy,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
)
from agent_evolve.application.residual_campaign_runtime import (
    ResidualArchiveTransitionCommit,
    ResidualArchiveTransitionPreparation,
)
from agent_evolve.application.residual_learning_transaction import (
    ResidualLearningState,
    TransactionalResidualLearningStore,
)
from agent_evolve.application.residual_portfolio_evolution import (
    DISJOINT_ACTION_EVALUATION_WAVES_V1,
    MaterializedActionEvaluation,
    MaterializedActionEvaluationBatch,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.residual_stage_credit import (
    ResidualStageCreditProjector,
)
from agent_evolve.application.semantic_coverage_score_portfolio import (
    MaterializedActionSemanticCell,
    MaterializedActionSemanticCellBatch,
    SemanticCoverageScorePortfolioPolicy,
)
from agent_evolve.application.sequential_lineage_allocation import (
    CandidateArchiveMarginalPilotOutcomeProjector,
    FrozenBranchSequentialLineagePlanner,
    SequentialLineageBranch,
)
from agent_evolve.application.sequential_residual_campaign_runtime import (
    SequentialResidualPortfolioCampaignStageRuntime,
)
from agent_evolve.application.sequential_residual_portfolio_evolution import (
    SequentialResidualPhase,
    SequentialResidualPhaseCommitAck,
    SequentialResidualPhaseReceipt,
    SequentialResidualPortfolioEvolution,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _RankScore:
    scorer_id: str
    scorer_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"sequential-score:{self.scorer_id}"),
        )

    async def score(self, request, proposals):
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
                            action_sha256=action.action_sha256,
                            value=float(10 - action.native_rank),
                        )
                        for proposal in proposals
                        for action in proposal.actions
                    ),
                    key=lambda value: value.action_sha256,
                )
            ),
            candidate_outcomes_observed=False,
            evidence_sha256=_sha(
                f"sequential-score-evidence:{self.scorer_id}:"
                f"{request.request_sha256}"
            ),
        )


@dataclass(frozen=True, slots=True)
class _Cells:
    projection_id: str = "test_sequential_lineage_cells"
    projection_version: int = 1
    definition_sha256: str = _sha("test-sequential-lineage-cells")

    async def project(self, request, proposals):
        return MaterializedActionSemanticCellBatch(
            projection_id=self.projection_id,
            projection_version=self.projection_version,
            projection_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            cells=tuple(
                sorted(
                    (
                        MaterializedActionSemanticCell(
                            action_sha256=action.action_sha256,
                            direction_signature=(("cost", "decrease"),),
                            recursive_lineage=bool(action.parent_ids),
                        )
                        for proposal in proposals
                        for action in proposal.actions
                    ),
                    key=lambda value: value.action_sha256,
                )
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
class _Expert:
    proposal: MaterializedActionProposalBatch
    expert_id: str = "test_sequential_expert"
    expert_version: int = 1
    definition_sha256: str = _sha("test-sequential-expert")
    evaluation_wave_semantics: str = (
        DISJOINT_ACTION_EVALUATION_WAVES_V1
    )
    propose_count: int = 0
    evaluation_waves: list[tuple[str, ...]] = field(default_factory=list)

    async def propose(self, request):
        assert request.request_sha256 == self.proposal.request_sha256
        self.propose_count += 1
        return self.proposal

    async def evaluate(self, proposal, selected_action_sha256s):
        assert proposal == self.proposal
        self.evaluation_waves.append(selected_action_sha256s)
        actions = {
            value.action_sha256: value for value in proposal.actions
        }
        evaluations = []
        for action_sha256 in selected_action_sha256s:
            action = actions[action_sha256]
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
                label=f"sequential_{action.native_rank}",
            )
            evaluations.append(
                MaterializedActionEvaluation(
                    action=action,
                    candidate=candidate,
                    evaluator_receipt_sha256=_sha(
                        f"sequential-evaluation:{action_sha256}"
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


@dataclass(frozen=True, slots=True)
class _Utility:
    utility_id: str = "test_sequential_archive_utility"
    utility_version: int = 1
    definition_sha256: str = _sha("test-sequential-archive-utility")

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


@dataclass(frozen=True, slots=True)
class _MappedPilotUtility:
    gain_by_cost: tuple[tuple[float, float], ...]
    utility_id: str = "test_mapped_pilot_utility"
    utility_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"mapped-pilot-utility:{self.gain_by_cost}"),
        )

    def utility(self, candidates):
        del candidates
        return 0.0

    def marginal_utility(self, candidates, objective_point):
        del candidates
        return dict(self.gain_by_cost).get(
            float(objective_point["cost"]),
            0.0,
        )


@dataclass(frozen=True, slots=True)
class _JointUtility:
    utility_id: str = "test_forecast_geometry_joint_utility"
    utility_version: int = 1
    definition_sha256: str = _sha("test-forecast-geometry-joint-utility")

    def utility(self, candidates):
        del candidates
        return 0.0

    def marginal_utility(self, candidates, objective_point):
        return self.portfolio_marginal_utility(
            candidates,
            (objective_point,),
        )

    def portfolio_marginal_utility(self, candidates, objective_points):
        del candidates
        return float(
            max(
                (value["axis_x"] for value in objective_points),
                default=0.0,
            )
            + max(
                (value["axis_y"] for value in objective_points),
                default=0.0,
            )
        )


@dataclass(frozen=True, slots=True)
class _ForecastGeometry:
    projection_id: str = "test_forecast_geometry"
    projection_version: int = 1
    definition_sha256: str = _sha("test-forecast-geometry")

    async def project(self, request, proposals):
        points = {
            1: (1.0, 0.0, 1.0),
            2: (0.0, 1.0, 1.0),
            3: (0.8, 0.8, 0.0),
        }
        members = []
        for proposal in proposals:
            for action in proposal.actions:
                axis_x, axis_y, reliability = points[action.native_rank]
                members.append(
                    MaterializedForecastGeometryMember(
                        action_sha256=action.action_sha256,
                        phenotype_identity_sha256=(
                            action.phenotype_identity_sha256
                        ),
                        reliability=reliability,
                        scenarios=(
                            ForecastGeometryScenario(
                                scenario_id="central",
                                objective_point=(
                                    ("axis_x", axis_x),
                                    ("axis_y", axis_y),
                                ),
                            ),
                        ),
                        source_evidence_sha256=_sha(
                            f"forecast-geometry:{action.action_sha256}"
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
    committer_id: str = "test_sequential_committer"
    committer_version: int = 1
    definition_sha256: str = _sha("test-sequential-committer")
    phases: list[SequentialResidualPhase] = field(default_factory=list)

    async def commit(
        self,
        receipt: SequentialResidualPhaseReceipt,
    ) -> SequentialResidualPhaseCommitAck:
        self.phases.append(receipt.phase)
        return SequentialResidualPhaseCommitAck(
            committer_id=self.committer_id,
            committer_version=self.committer_version,
            committer_definition_sha256=self.definition_sha256,
            phase_receipt_sha256=receipt.receipt_sha256,
            durable=True,
            evidence=freeze_json({"fsync_completed": True}),
        )


@dataclass(frozen=True, slots=True)
class _FailAfterPilotGate:
    gate_id: str = "test_fail_after_pilot_gate"
    gate_version: int = 1
    definition_sha256: str = _sha("test-fail-after-pilot-gate")

    def decide(self, plan, outcomes):
        del plan, outcomes
        raise RuntimeError("intentional post-pilot gate failure")


@dataclass(frozen=True, slots=True)
class _DroppedSourcePilotPolicy:
    policy_id: str = "test_dropped_source_pilot_policy"
    policy_version: int = 1
    definition_sha256: str = _sha("test-dropped-source-pilot-policy")

    def design(
        self,
        *,
        branch_bindings,
        branch_requirements,
        market,
        evaluation_slots,
        maximum_pilot_slots,
    ):
        del maximum_pilot_slots
        source = {
            binding.branch_id: requirement
            for binding, requirement in zip(
                branch_bindings,
                branch_requirements,
                strict=True,
            )
        }
        pilot = source["gamma"].required_action_sha256s[0]

        def complete(branch_id):
            selected = [pilot]
            phenotypes = {market[pilot].phenotype_identity_sha256}
            for action_sha256 in (
                *source[branch_id].required_action_sha256s,
                *tuple(sorted(market)),
            ):
                phenotype = market[
                    action_sha256
                ].phenotype_identity_sha256
                if (
                    action_sha256 in selected
                    or phenotype in phenotypes
                ):
                    continue
                selected.append(action_sha256)
                phenotypes.add(phenotype)
                if len(selected) == evaluation_slots:
                    break
            return tuple(sorted(selected))

        return PortfolioRaceDisagreementDesign(
            pilot_pairs=((pilot, "branch:gamma"),),
            branch_action_sha256s=(
                ("alpha", complete("alpha")),
                ("beta", complete("beta")),
            ),
            evidence=freeze_json(
                {
                    "test": "pilot_provenance_outlives_branch_collapse",
                    "candidate_outcomes_observed": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _AdditiveJointSnapshot:
    def joint_gain(self, points):
        return float(0.1 * len(points))


@dataclass(frozen=True, slots=True)
class _StageArchiveUtility:
    definition_sha256: str = _sha("test-sequential-stage-utility")

    def require_snapshot(self, value):
        del value
        return _AdditiveJointSnapshot()


@dataclass
class _StageArchive:
    archive_id: str = "test_sequential_archive"
    archive_version: int = 1
    definition_sha256: str = _sha("test-sequential-stage-archive")
    committed: bool = False
    aborted: bool = False

    async def prepare(self, result):
        common = {
            "utility_id": "test_sequential_stage_utility",
            "utility_version": 1,
            "definition_sha256": _sha(
                "test-sequential-stage-utility"
            ),
            "generation": result.request.decision_index,
            "benchmark_sha256": _sha("test-sequential-benchmark"),
        }
        return ResidualArchiveTransitionPreparation(
            archive_id=self.archive_id,
            archive_version=self.archive_version,
            archive_definition_sha256=self.definition_sha256,
            residual_result_sha256=result.result_sha256,
            pre_snapshot=ArchiveUtilitySnapshot(
                **common,
                archive_sha256=_sha("test-sequential-pre-archive"),
                snapshot_receipt=freeze_json(
                    {"base_utility_hex": float(0.2).hex()}
                ),
                scalar_utility_hex=float(0.2).hex(),
            ),
            post_snapshot=ArchiveUtilitySnapshot(
                **common,
                archive_sha256=_sha("test-sequential-post-archive"),
                snapshot_receipt=freeze_json(
                    {"base_utility_hex": float(0.5).hex()}
                ),
                scalar_utility_hex=float(0.5).hex(),
            ),
            evidence=freeze_json({"preview_only": True}),
        )

    def commit(self, preparation):
        self.committed = True
        return ResidualArchiveTransitionCommit(
            preparation_sha256=preparation.preparation_sha256,
            committed_archive_sha256=(
                preparation.post_snapshot.archive_sha256
            ),
            evidence=freeze_json({"published": True}),
        )

    def abort(self, preparation):
        del preparation
        self.aborted = True


def _prior_candidate() -> EvolutionCandidate:
    configuration = freeze_json({"prior": True})
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId("candidate_sequential_prior"),
            configuration_hash=typed_json_sha256(configuration),
            configuration_artifact_hash=hashlib.sha256(
                canonical_typed_json_bytes(configuration)
            ).hexdigest(),
            proposal_sequence=0,
        ),
        configuration=configuration,
        objectives=(("cost", 10.0),),
        valid=True,
        generation=0,
        label="sequential_prior",
    )


def _prior_provenance() -> CandidateProposalProvenance:
    prior = _prior_candidate()
    return CandidateProposalProvenance(
        candidate_id=prior.candidate_id,
        configuration_sha256=prior.occurrence.configuration_hash,
        generation=prior.generation,
        source_role=ProposalLineageRole.BACKBONE,
        proposal_expert_id="test_prior_archive",
        proposal_expert_version=1,
        proposal_expert_definition_sha256=_sha("test-prior-archive"),
        operator_id="archive_seed",
        parent_candidate_ids=(),
        decision_cutoff_sha256=_sha("test-prior-cutoff"),
        source_receipt_sha256=_sha("test-prior-receipt"),
    )


def _sequential_fixture(
    *,
    evaluation_slots: int = 3,
    action_count: int = 4,
):
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("sequential-campaign"),
        prior_state_sha256=_sha("sequential-prior-state"),
        decision_index=2,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=max(6, evaluation_slots),
        evaluation_slots=evaluation_slots,
        expert_proposal_slots=(
            ("test_sequential_expert", action_count),
        ),
        proposal_context=freeze_json({"test": "sequential-runtime"}),
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
        structural_signature_sha256=_sha("sequential-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.trace",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    actions = tuple(
        MaterializedActionDescriptor(
            context=context,
            configuration=freeze_json({"native_rank": rank}),
            phenotype_identity_sha256=_sha(
                f"sequential-phenotype:{rank}"
            ),
            expert_id="test_sequential_expert",
            native_rank=rank,
            parent_ids=(
                (CandidateId("candidate_sequential_prior"),)
                if rank <= 2
                else ()
            ),
            operator_id="test_mutation",
            target_candidate_id=CandidateId(
                f"candidate_sequential_{rank}"
            ),
            role_id="local_exploit",
            normalized_evaluation_cost=1.0,
            reference_action=False,
        )
        for rank in range(1, action_count + 1)
    )
    proposal = MaterializedActionProposalBatch(
        request_sha256=request.request_sha256,
        expert_id="test_sequential_expert",
        expert_version=1,
        expert_definition_sha256=_sha("test-sequential-expert"),
        actions=actions,
        evidence=freeze_json(
            {
                "candidate_outcomes_observed": False,
                "sealed_once": True,
            }
        ),
    )
    primary = _RankScore("primary_rank")
    transport = _RankScore("transport_rank")

    def allocation(*, spillover: bool):
        return SemanticCoverageScorePortfolioPolicy(
            scorers=(primary, transport),
            scorer_capacity_fractions=(
                ("primary_rank", 2.0 / 3.0),
                ("transport_rank", 0.0),
            ),
            lineage_scorer_id="transport_rank",
            lineage_member_scorer_id="primary_rank",
            lineage_deficit_refill_scorer_id="primary_rank",
            lineage_capacity_fraction=1.0 / 3.0,
            semantic_projection=_Cells(),
            coverage_strength=1.0 / 3.0,
            allow_recursive_score_lane_spillover=spillover,
        )

    locked = allocation(spillover=False)
    unlocked = allocation(spillover=True)
    expert = _Expert(proposal)
    return request, expert, locked, unlocked


async def _exercise_sequential_runtime() -> None:
    request, expert, locked, unlocked = _sequential_fixture()
    committer = _Committer()
    result = await SequentialResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exploration_policy=None,
        ),
        planner=FrozenBranchSequentialLineagePlanner(
            locked_policy=locked,
            unlocked_policy=unlocked,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        phase_committer=committer,
        require_durable_phase_commits=True,
    ).run(request)

    assert expert.propose_count == 1
    assert tuple(len(value) for value in expert.evaluation_waves) == (1, 2)
    assert result.gate_decision.branch is SequentialLineageBranch.UNLOCKED
    assert result.gate_decision.positive_pilot_count == 1
    assert len(result.result.evaluations) == 3
    assert sum(
        bool(value.action.parent_ids) for value in result.result.evaluations
    ) == 2
    assert committer.phases == list(SequentialResidualPhase)
    assert len(result.phase_commit_acks) == 4
    result.__post_init__()


def test_sequential_runtime_freezes_branches_and_uses_real_pilot_gate() -> None:
    asyncio.run(_exercise_sequential_runtime())


async def _exercise_pilot_checkpoint_precedes_gate() -> None:
    request, expert, locked, unlocked = _sequential_fixture()
    committer = _Committer()
    runtime = SequentialResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exploration_policy=None,
        ),
        planner=FrozenBranchSequentialLineagePlanner(
            locked_policy=locked,
            unlocked_policy=unlocked,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        gate=_FailAfterPilotGate(),
        phase_committer=committer,
        require_durable_phase_commits=True,
    )
    with pytest.raises(
        RuntimeError,
        match="intentional post-pilot gate failure",
    ):
        await runtime.run(request)

    assert committer.phases == [
        SequentialResidualPhase.PLAN_FROZEN,
        SequentialResidualPhase.PILOT_EVALUATED,
    ]
    assert tuple(len(value) for value in expert.evaluation_waves) == (1,)


def test_pilot_evidence_is_durable_before_gate_adjudication() -> None:
    asyncio.run(_exercise_pilot_checkpoint_precedes_gate())


async def _exercise_three_branch_portfolio_race() -> None:
    request, expert, locked, unlocked = _sequential_fixture()
    third = _sequential_fixture()[3]
    result = await SequentialResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exploration_policy=None,
        ),
        planner=PrecommittedPortfolioRacePlanner(
            branch_bindings=(
                PortfolioRacePolicyBinding(
                    branch_id="locked",
                    policy=locked,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="unlocked",
                    policy=unlocked,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="wide",
                    policy=third,
                ),
            ),
            pilot_policy=unlocked,
            pilot_slots=1,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        gate=EvidenceAdaptivePortfolioRaceGate(),
    ).run(request)

    assert type(result.allocation_plan) is PrecommittedPortfolioRacePlan
    assert result.allocation_plan.frozen_branch_ids == (
        "locked",
        "unlocked",
        "wide",
    )
    assert expert.propose_count == 1
    assert tuple(len(value) for value in expert.evaluation_waves) == (1, 2)
    assert len(result.result.evaluations) == request.evaluation_slots
    assert result.gate_decision.selected_branch_id in {
        "locked",
        "unlocked",
        "wide",
    }
    assert result.gate_decision.positive_pilot_count == 1
    assert len(result.gate_decision.branch_scores) == 3
    result.__post_init__()


def test_sequential_runtime_executes_three_branch_precommitted_race() -> None:
    asyncio.run(_exercise_three_branch_portfolio_race())


async def _exercise_causal_opportunity_routes() -> None:
    request, expert, locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    plan = await PrecommittedPortfolioRacePlanner(
        branch_bindings=(
            PortfolioRacePolicyBinding(
                branch_id="exploration",
                policy=locked,
            ),
            PortfolioRacePolicyBinding(
                branch_id="productive",
                policy=unlocked,
            ),
            PortfolioRacePolicyBinding(
                branch_id="renewal",
                policy=unlocked,
            ),
        ),
        pilot_policy=None,
        pilot_slots=3,
    ).plan(request, (expert.proposal,))
    batch = await expert.evaluate(
        expert.proposal,
        plan.pilot_action_sha256s,
    )
    evaluations = batch.evaluations
    pilot_costs = tuple(
        value.candidate.objective_map["cost"] for value in evaluations
    )

    def outcomes(gains):
        return CandidateArchiveMarginalPilotOutcomeProjector(
            prior_candidates=(_prior_candidate(),),
            utility=_MappedPilotUtility(
                tuple(zip(pilot_costs, gains, strict=True))
            ),
        ).project(plan, evaluations)

    gate = CausalOpportunityPortfolioRaceGate(
        renewal_branch_id="renewal",
        exploration_branch_id="exploration",
        reference_gain_scale=1.0,
        reference_gain_evidence_sha256=_sha(
            "authenticated-prior-positive-gain-scale"
        ),
        minimum_peak_gain_ratio=0.5,
        minimum_positive_fraction=0.5,
        minimum_pilot_count=2,
    )
    weak = gate.decide(plan, outcomes((0.0, 0.0, 0.0)))
    sparse = gate.decide(plan, outcomes((1.0, 0.0, 0.0)))
    productive = gate.decide(plan, outcomes((1.0, 1.0, 1.0)))

    assert weak.selected_branch_id == "renewal"
    assert (
        thaw_json(weak.evidence)["route"]
        == "weak_opportunity_renewal"
    )
    assert sparse.selected_branch_id == "exploration"
    assert (
        thaw_json(sparse.evidence)["route"]
        == "sparse_opportunity_exploration"
    )
    productive_evidence = thaw_json(productive.evidence)
    assert (
        productive_evidence["route"]
        == "productive_market_lane_adaptation"
    )
    assert productive.selected_branch_id in plan.frozen_branch_ids
    assert (
        productive_evidence["base_gate_decision"][
            "selected_branch_id"
        ]
        == productive.selected_branch_id
    )
    for decision in (weak, sparse, productive):
        assert (
            decision.selected_requirement_sha256
            == plan.requirement_for(
                decision.selected_branch_id
            ).requirement_sha256
        )
        decision.__post_init__()


def test_causal_opportunity_gate_routes_weak_sparse_and_productive_markets():
    asyncio.run(_exercise_causal_opportunity_routes())


async def _exercise_causal_route_equivalence_resolution() -> None:
    request, expert, locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    plan = await PrecommittedPortfolioRacePlanner(
        branch_bindings=(
            PortfolioRacePolicyBinding(
                branch_id="exploration",
                policy=locked,
            ),
            PortfolioRacePolicyBinding(
                branch_id="productive",
                policy=unlocked,
            ),
            PortfolioRacePolicyBinding(
                branch_id="renewal",
                policy=unlocked,
            ),
        ),
        pilot_policy=None,
        pilot_slots=3,
        disagreement_policy=SymmetricDifferencePortfolioRacePolicy(),
    ).plan(request, (expert.proposal,))
    assert "renewal" not in plan.frozen_branch_ids
    assert "productive" in plan.frozen_branch_ids
    batch = await expert.evaluate(
        expert.proposal,
        plan.pilot_action_sha256s,
    )
    costs = tuple(
        value.candidate.objective_map["cost"]
        for value in batch.evaluations
    )
    outcomes = CandidateArchiveMarginalPilotOutcomeProjector(
        prior_candidates=(_prior_candidate(),),
        utility=_MappedPilotUtility(
            tuple((value, 0.0) for value in costs)
        ),
    ).project(plan, batch.evaluations)
    decision = CausalOpportunityPortfolioRaceGate(
        renewal_branch_id="renewal",
        exploration_branch_id="exploration",
        reference_gain_scale=1.0,
        reference_gain_evidence_sha256=_sha(
            "authenticated-equivalence-resolution-scale"
        ),
        minimum_pilot_count=1,
    ).decide(plan, outcomes)
    evidence = thaw_json(decision.evidence)

    assert decision.selected_branch_id == "productive"
    assert evidence["route"] == "weak_opportunity_renewal"
    assert evidence["requested_branch_id"] == "renewal"
    assert evidence["resolved_branch_id"] == "productive"
    assert evidence["route_equivalence_path"] == [
        "identical_complete_action_set_before_pilot"
    ]
    decision.__post_init__()


def test_causal_gate_resolves_only_authenticated_equivalent_routes() -> None:
    asyncio.run(_exercise_causal_route_equivalence_resolution())


async def _exercise_dropped_source_pilot_provenance() -> None:
    request, expert, locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    plan = await PrecommittedPortfolioRacePlanner(
        branch_bindings=(
            PortfolioRacePolicyBinding(
                branch_id="alpha",
                policy=locked,
            ),
            PortfolioRacePolicyBinding(
                branch_id="beta",
                policy=unlocked,
            ),
            PortfolioRacePolicyBinding(
                branch_id="gamma",
                policy=unlocked,
            ),
        ),
        pilot_policy=None,
        pilot_slots=3,
        disagreement_policy=_DroppedSourcePilotPolicy(),
    ).plan(request, (expert.proposal,))
    evidence = thaw_json(plan.evidence)

    assert plan.frozen_branch_ids == ("alpha", "beta")
    assert len(plan.pilot_action_sha256s) == 1
    assert evidence["pilot_source_attribution"][0][
        "source_branch_id"
    ] == "gamma"
    assert evidence["pilot_source_attribution"][0][
        "action_sha256"
    ] == plan.pilot_action_sha256s[0]
    plan.__post_init__()


def test_pilot_lane_projection_uses_full_precollapse_source_market() -> None:
    asyncio.run(_exercise_dropped_source_pilot_provenance())


async def _exercise_branch_stratified_portfolio_race() -> None:
    request, expert, locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    third = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )[3]
    result = await SequentialResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exploration_policy=None,
        ),
        planner=PrecommittedPortfolioRacePlanner(
            branch_bindings=(
                PortfolioRacePolicyBinding(
                    branch_id="locked",
                    policy=locked,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="unlocked",
                    policy=unlocked,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="wide",
                    policy=third,
                ),
            ),
            pilot_policy=None,
            pilot_slots=3,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        gate=EvidenceAdaptivePortfolioRaceGate(),
    ).run(request)

    assert tuple(len(value) for value in expert.evaluation_waves) == (3, 1)
    assert len(result.result.evaluations) == 4
    assert {
        value.lane_id
        for value in result.allocation_plan.pilot_lane_bindings
    } == {
        "score_lane:primary_rank.expert:test_sequential_expert",
        "score_lane:transport_rank.expert:test_sequential_expert",
    }
    assert all(
        not value.lane_id.startswith("branch:")
        for value in result.allocation_plan.pilot_lane_bindings
    )
    assert all(
        value.observed_completion_fraction == 1.0
        for value in result.gate_decision.branch_scores
    )
    result.__post_init__()


def test_portfolio_race_uses_one_diagnostic_pilot_per_branch() -> None:
    asyncio.run(_exercise_branch_stratified_portfolio_race())


async def _exercise_incomplete_lane_coverage_falls_back_to_prior() -> None:
    request, expert, locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    result = await SequentialResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exploration_policy=None,
        ),
        planner=PrecommittedPortfolioRacePlanner(
            branch_bindings=(
                PortfolioRacePolicyBinding(
                    branch_id="alpha_observed",
                    policy=locked,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="beta_unobserved",
                    policy=unlocked,
                ),
            ),
            pilot_policy=None,
            pilot_slots=2,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        gate=EvidenceAdaptivePortfolioRaceGate(),
    ).run(request)

    observed_lane = result.allocation_plan.pilot_lane_bindings[0].lane_id
    branches = tuple(
        replace(
            branch,
            prior_mean=(
                0.4
                if branch.branch_id == "alpha_observed"
                else 0.6
            ),
            completion_lane_exposure=(
                ((observed_lane, 2),)
                if branch.branch_id == "alpha_observed"
                else (("score_lane:held_out.expert:other", 2),)
            ),
        )
        for branch in result.allocation_plan.branches
    )
    counterfactual_plan = replace(
        result.allocation_plan,
        branches=branches,
    )
    counterfactual_outcomes = replace(
        result.pilot_outcomes,
        plan_sha256=counterfactual_plan.plan_sha256,
    )
    decision = EvidenceAdaptivePortfolioRaceGate().decide(
        counterfactual_plan,
        counterfactual_outcomes,
    )
    evidence = thaw_json(decision.evidence)

    assert evidence["adaptive_selection_qualified"] is False
    assert (
        evidence["selection_mode"]
        == "insufficient_coverage_frozen_prior_fallback"
    )
    assert decision.selected_branch_id == "beta_unobserved"
    assert {
        value.branch_id: value.observed_completion_fraction
        for value in decision.branch_scores
    } == {
        "alpha_observed": 1.0,
        "beta_unobserved": 0.0,
    }


def test_portfolio_race_does_not_extrapolate_one_pilot_to_unseen_lanes() -> None:
    asyncio.run(_exercise_incomplete_lane_coverage_falls_back_to_prior())


async def _exercise_phase_conditioned_prior_projection() -> None:
    request, expert, locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    prior = PhaseConditionedPortfolioRacePrior(
        prior_means=(
            (SearchPhase.COMPOSITION, "locked", 0.2),
            (SearchPhase.COMPOSITION, "unlocked", 0.8),
        )
    )
    plan = await PrecommittedPortfolioRacePlanner(
        branch_bindings=(
            PortfolioRacePolicyBinding(
                branch_id="locked",
                policy=locked,
            ),
            PortfolioRacePolicyBinding(
                branch_id="unlocked",
                policy=unlocked,
            ),
        ),
        pilot_policy=None,
        pilot_slots=2,
        prior_projection=prior,
    ).plan(request, (expert.proposal,))
    evidence = thaw_json(plan.evidence)

    assert {
        value.branch_id: value.prior_mean for value in plan.branches
    } == {
        "locked": 0.2,
        "unlocked": 0.8,
    }
    assert evidence["branch_prior_projection"] == {
        "mode": "injected_outcome_blind_projection",
        "projection_id": prior.projection_id,
        "projection_version": prior.projection_version,
        "definition_sha256": prior.definition_sha256,
        "request_phase": "composition",
        "prior_means": [
            {"branch_id": "locked", "mean_hex": float(0.2).hex()},
            {"branch_id": "unlocked", "mean_hex": float(0.8).hex()},
        ],
        "applied_before_disagreement_equivalence": True,
        "candidate_outcomes_observed": False,
    }


def test_portfolio_race_prior_is_injected_through_generic_phase_port() -> None:
    asyncio.run(_exercise_phase_conditioned_prior_projection())


async def _exercise_projected_prior_precedes_equivalence_resolution() -> None:
    request, expert, _locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    prior = PhaseConditionedPortfolioRacePrior(
        prior_means=(
            (SearchPhase.COMPOSITION, "alpha", 0.2),
            (SearchPhase.COMPOSITION, "beta", 0.8),
        )
    )
    plan = await PrecommittedPortfolioRacePlanner(
        branch_bindings=(
            PortfolioRacePolicyBinding(
                branch_id="alpha",
                policy=unlocked,
                prior_mean=0.9,
            ),
            PortfolioRacePolicyBinding(
                branch_id="beta",
                policy=unlocked,
                prior_mean=0.1,
            ),
        ),
        pilot_policy=None,
        pilot_slots=3,
        disagreement_policy=SymmetricDifferencePortfolioRacePolicy(),
        prior_projection=prior,
    ).plan(request, (expert.proposal,))
    evidence = thaw_json(plan.evidence)

    assert plan.frozen_branch_ids == ("beta",)
    assert plan.branches[0].prior_mean == 0.8
    assert plan.pilot_action_sha256s == ()
    assert evidence["branch_prior_projection"]["prior_means"] == [
        {"branch_id": "alpha", "mean_hex": float(0.2).hex()},
        {"branch_id": "beta", "mean_hex": float(0.8).hex()},
    ]
    assert (
        evidence["branch_prior_projection"][
            "applied_before_disagreement_equivalence"
        ]
        is True
    )
    design_evidence = evidence["disagreement_design"]["evidence"]
    assert design_evidence["decision"] == "equivalent_source_bypass"
    assert design_evidence["source_equivalence_classes"] == [
        {
            "representative_branch_id": "beta",
            "member_branch_ids": ["alpha", "beta"],
            "action_sha256s": list(
                plan.branches[0].requirement.required_action_sha256s
            ),
        }
    ]


def test_projected_prior_is_label_invariant_before_equivalence_collapse() -> None:
    asyncio.run(_exercise_projected_prior_precedes_equivalence_resolution())


async def _exercise_forecast_geometry_policy_arms() -> None:
    request, expert, _locked, _unlocked = _sequential_fixture(
        evaluation_slots=2,
        action_count=3,
    )
    common = {
        "prior_candidates": (_prior_candidate(),),
        "projection": _ForecastGeometry(),
        "archive_utility": _JointUtility(),
        "scenario_id": "central",
    }
    central = await ForecastGeometryPortfolioPolicy(
        **common,
        mode=ForecastGeometryPortfolioMode.SCENARIO,
    ).require(request, (expert.proposal,))
    reliable = await ForecastGeometryPortfolioPolicy(
        **common,
        mode=(
            ForecastGeometryPortfolioMode.RELIABILITY_ADJUSTED_SCENARIO
        ),
    ).require(request, (expert.proposal,))
    action_by_rank = {
        value.native_rank: value.action_sha256
        for value in expert.proposal.actions
    }

    assert set(central.required_action_sha256s) == {
        action_by_rank[1],
        action_by_rank[3],
    }
    assert set(reliable.required_action_sha256s) == {
        action_by_rank[1],
        action_by_rank[2],
    }
    for requirement in (central, reliable):
        evidence = thaw_json(requirement.evidence)
        assert evidence["candidate_outcomes_observed"] is False
        assert evidence["joint_set_value_not_member_score_sum"] is True
        assert evidence["candidate_portfolio_evaluations"] == 5
        assert len(evidence["selection_trace"]) == 2


def test_forecast_geometry_arms_share_one_generic_joint_value_port() -> None:
    asyncio.run(_exercise_forecast_geometry_policy_arms())


async def _exercise_equivalent_branch_bypass() -> None:
    request, expert, _locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    result = await SequentialResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exploration_policy=None,
        ),
        planner=PrecommittedPortfolioRacePlanner(
            branch_bindings=(
                PortfolioRacePolicyBinding(
                    branch_id="alpha",
                    policy=unlocked,
                    prior_mean=0.4,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="beta",
                    policy=unlocked,
                    prior_mean=0.9,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="gamma",
                    policy=unlocked,
                    prior_mean=0.6,
                ),
            ),
            pilot_policy=None,
            pilot_slots=3,
            disagreement_policy=(
                SymmetricDifferencePortfolioRacePolicy()
            ),
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        gate=EvidenceAdaptivePortfolioRaceGate(),
    ).run(request)

    assert result.allocation_plan.frozen_branch_ids == ("beta",)
    assert result.allocation_plan.pilot_action_sha256s == ()
    assert tuple(len(value) for value in expert.evaluation_waves) == (4,)
    assert result.gate_decision.selected_branch_id == "beta"
    assert result.gate_decision.pilot_count == 0
    assert result.gate_decision.positive_pilot_count == 0
    assert len(result.gate_decision.branch_scores) == 1
    plan_evidence = thaw_json(result.allocation_plan.evidence)
    assert (
        plan_evidence["disagreement_design"]["evidence"]["decision"]
        == "equivalent_source_bypass"
    )
    assert len(result.result.evaluations) == request.evaluation_slots
    result.__post_init__()


def test_disagreement_policy_bypasses_equivalent_branches_without_pilots() -> None:
    asyncio.run(_exercise_equivalent_branch_bypass())


async def _exercise_disagreement_preserving_variable_pilot() -> None:
    request, expert, locked, unlocked = _sequential_fixture(
        evaluation_slots=4,
        action_count=6,
    )
    result = await SequentialResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exploration_policy=None,
        ),
        planner=PrecommittedPortfolioRacePlanner(
            branch_bindings=(
                PortfolioRacePolicyBinding(
                    branch_id="locked",
                    policy=locked,
                ),
                PortfolioRacePolicyBinding(
                    branch_id="unlocked",
                    policy=unlocked,
                ),
            ),
            pilot_policy=None,
            pilot_slots=3,
            disagreement_policy=(
                SymmetricDifferencePortfolioRacePolicy()
            ),
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        gate=EvidenceAdaptivePortfolioRaceGate(),
    ).run(request)

    assert result.allocation_plan.frozen_branch_ids == (
        "locked",
        "unlocked",
    )
    assert len(result.allocation_plan.pilot_action_sha256s) == 1
    branch_sets = tuple(
        set(value.required_action_sha256s)
        for value in result.allocation_plan.frozen_requirements
    )
    assert branch_sets[0] != branch_sets[1]
    assert set(result.allocation_plan.pilot_action_sha256s).issubset(
        set.intersection(*branch_sets)
    )
    assert tuple(len(value) for value in expert.evaluation_waves) == (1, 3)
    design_evidence = thaw_json(
        result.allocation_plan.evidence
    )["disagreement_design"]["evidence"]
    assert design_evidence["decision"] == "actionable_disagreement_race"
    assert design_evidence["configured_maximum_pilot_slots"] == 3
    assert design_evidence["effective_pilot_slots"] == 1
    assert [
        value["distinct_complete_continuations"]
        for value in design_evidence["pilot_search_attempts"]
    ] == [1, 2]
    assert len(result.gate_decision.branch_scores) == 2
    assert len(result.result.evaluations) == request.evaluation_slots
    result.__post_init__()


def test_disagreement_policy_reduces_pilot_until_choice_remains() -> None:
    asyncio.run(_exercise_disagreement_preserving_variable_pilot())


async def _exercise_sequential_campaign_runtime() -> None:
    request, expert, locked, unlocked = _sequential_fixture()
    archive = _StageArchive()
    lineage = EarnedLineageLedger()
    lineage.register((_prior_provenance(),))
    learning = TransactionalResidualLearningStore(
        projector=ResidualStageCreditProjector(
            archive_utility=_StageArchiveUtility()
        ),
        _state=ResidualLearningState(
            broker_evidence=MaterializedActionEvidenceLedger(),
            earned_lineage=lineage,
            current_generation=0,
            revision=0,
        ),
    )
    committer = _Committer()
    receipt = await SequentialResidualPortfolioCampaignStageRuntime(
        experts=(expert,),
        archive=archive,
        learning=learning,
        planner=FrozenBranchSequentialLineagePlanner(
            locked_policy=locked,
            unlocked_policy=unlocked,
        ),
        pilot_outcome_projector=(
            CandidateArchiveMarginalPilotOutcomeProjector(
                prior_candidates=(_prior_candidate(),),
                utility=_Utility(),
            )
        ),
        slate_value=ZeroMaterializedSlateValue(),
        slate_feasibility=UniquePhenotypeMaterializedSlateFeasibility(),
        exploration_policy=None,
        phase_committer=committer,
        require_durable_phase_commits=True,
    ).run(request)

    assert archive.committed
    assert not archive.aborted
    assert expert.propose_count == 1
    assert tuple(len(value) for value in expert.evaluation_waves) == (1, 2)
    assert (
        receipt.sequential_result.gate_decision.branch
        is SequentialLineageBranch.UNLOCKED
    )
    assert receipt.campaign_stage.result.result_sha256 == (
        receipt.sequential_result.result.result_sha256
    )
    assert receipt.campaign_stage.learning_commit.committed_state_sha256 == (
        learning.state.state_sha256
    )
    assert len(learning.state.broker_evidence.outcomes) == 3
    assert learning.state.earned_lineage.version == 3
    assert len(receipt.sequential_result.phase_commit_acks) == 4
    assert committer.phases == list(SequentialResidualPhase)
    assert (
        receipt.to_record()[
            "workload_objective_model_provider_prompt_fields_present"
        ]
        is False
    )
    receipt.__post_init__()


def test_sequential_campaign_runtime_publishes_archive_and_learning_once() -> None:
    asyncio.run(_exercise_sequential_campaign_runtime())

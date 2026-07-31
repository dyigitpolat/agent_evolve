from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from decimal import Decimal
import hashlib
import math

import pytest

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.core.optimization_semantics import MetricSense
from agent_evolve.application.frozen_hurdle_score import (
    FrozenHurdleMaterializedActionScorer,
    FrozenHurdleScoreKind,
    FrozenStandardizedLinearModel,
    MaterializedActionFeatureBatch,
    MaterializedActionFeatureVector,
    WinsorizedFrozenHurdleMaterializedActionScorer,
)
from agent_evolve.application.support_guarded_hurdle_score import (
    FrozenFeatureSupportGroup,
    SupportGuardedFrozenHurdleMaterializedActionScorer,
)
from agent_evolve.application.action_score_authorities import (
    NativeRankMaterializedActionScorer,
    TargetEmpiricalReturnMaterializedActionScorer,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
    ReliabilityAdaptiveScorePortfolioPolicy,
)
from agent_evolve.application.materialized_action_constraints import (
    UniquePhenotypeMaterializedSlateFeasibility,
    ZeroMaterializedSlateValue,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    MaterializedActionOutcome,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionEvaluation,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.semantic_coverage_score_portfolio import (
    MaterializedActionSemanticCell,
    MaterializedActionSemanticCellBatch,
    SemanticCoverageScorePortfolioPolicy,
)
from agent_evolve.application.sequential_lineage_allocation import (
    AnyPositiveSequentialLineageGate,
    CandidateArchiveMarginalPilotOutcomeProjector,
    FrozenBranchSequentialLineagePlanner,
    SequentialLineageBranch,
)
from agent_evolve.application.residual_reachability import (
    HierarchicalResidualPlan,
    ParentFiniteVariationBinding,
    ResidualProposalRole,
    bind_cross_parent_finite_action_schema,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.integrations.pydantic_ai.materialized_hierarchical_residual_expert import (
    HierarchicalResidualExpertSpec,
    PydanticAIMaterializedHierarchicalResidualExpert,
    ResidualParentActionContext,
)
from agent_evolve.integrations.pydantic_ai.portable_residual_consequence_features import (
    PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS,
    PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES,
    PortableResidualConsequenceFeatureProjection,
)
from agent_evolve.integrations.pydantic_ai.residual_reachability import (
    HierarchicalResidualMetricForecast,
    HierarchicalResidualProposalSelection,
)
from agent_evolve.integrations.pydantic_ai.residual_forecast_geometry import (
    PydanticAIResidualForecastGeometryProjection,
    ResidualForecastEvidenceCoverageMode,
)
from agent_evolve.integrations.pydantic_ai.residual_semantic_cells import (
    PydanticAIResidualSemanticCellProjection,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityDecision,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class _ExactRejectedConfiguration:
    rejected_configuration_sha256: str
    policy_id: str = field(
        init=False,
        default="test_exact_rejected_configuration",
    )
    policy_version: int = field(init=False, default=1)
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"hard-feasibility:{self.rejected_configuration_sha256}"),
        )

    def assess(
        self,
        request: HardFeasibilityRequest,
    ) -> HardFeasibilityDecision:
        request.__post_init__()
        configuration_sha256 = typed_json_sha256(request.configuration)
        rejected = (
            configuration_sha256 == self.rejected_configuration_sha256
        )
        return HardFeasibilityDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            verdict=(
                HardFeasibilityVerdict.INFEASIBLE
                if rejected
                else HardFeasibilityVerdict.UNKNOWN
            ),
            proof=freeze_json(
                {
                    "configuration_sha256": configuration_sha256,
                    "exact_rejection_witness": (
                        "registered_test_identity"
                        if rejected
                        else None
                    ),
                }
            ),
        )


def _contract(parent: dict[str, int]) -> FiniteVariationContract:
    frozen = freeze_json(parent)
    parent_sha256 = typed_json_sha256(frozen)
    return FiniteVariationContract(
        catalog_id="generic_test_catalog",
        catalog_version=1,
        catalog_definition_sha256=_sha("catalog"),
        parent_configuration=frozen,
        options=(
            FiniteVariationOption(
                option_id="field.a.one",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(
                    {"a": 1, "b": parent["b"]}
                ),
                family="replace",
                description=f"Set a from {parent['a']} to one.",
                metadata=(("locus", "a"), ("target", "one")),
            ),
            FiniteVariationOption(
                option_id="field.b.one",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(
                    {"a": parent["a"], "b": 1}
                ),
                family="replace",
                description="Set b to one.",
                metadata=(("locus", "b"), ("target", "one")),
            ),
        ),
    )


def _schema():
    first = ParentFiniteVariationBinding(
        CandidateId("candidate_parent_a"),
        _contract({"a": 0, "b": 0}),
    )
    second = ParentFiniteVariationBinding(
        CandidateId("candidate_parent_b"),
        _contract({"a": 2, "b": 0}),
    )
    return bind_cross_parent_finite_action_schema((first, second))


@dataclass
class _Policy:
    async def select(self, request):
        plans = tuple(
            HierarchicalResidualPlan(
                parent_candidate_id=binding.parent_candidate_id,
                parent_contract_sha256=binding.contract.identity_sha256,
                action_schema_sha256=request.action_schema.schema_sha256,
                component_option_ids=(option_id,),
                role=ResidualProposalRole.LOCAL_EXPLOIT,
                expert_id=request.expert_id,
                expert_definition_sha256=(
                    request.expert_definition_sha256
                ),
                native_rank=rank,
                decision_receipt_sha256=_sha(
                    f"{request.request_sha256}:{rank}"
                ),
            )
            for rank, (binding, option_id) in enumerate(
                zip(
                    request.action_schema.bindings,
                    ("field.a.one", "field.b.one"),
                    strict=True,
                ),
                start=1,
            )
        )
        forecast = (
            HierarchicalResidualMetricForecast(
                metric_id="cost",
                p10_delta=-2.0,
                p50_delta=-1.0,
                p90_delta=0.0,
                confidence=0.75,
            ),
        )
        return HierarchicalResidualProposalSelection(
            request_sha256=request.request_sha256,
            decision_sha256=_sha(f"decision:{request.request_sha256}"),
            plans=plans,
            rationales=(
                "Test the first exact parent-bound move.",
                "Test the second exact parent-bound move.",
            ),
            probability_valid=(0.9, 0.8),
            effect_predictions=(forecast, forecast),
            slate_rationale="Cover both retained parent states.",
            telemetry=AgenticCallTelemetry(
                requested_model="provider/model",
                resolved_model="provider/model",
                resolved_provider="provider",
                provider_response_id="response",
                finish_reason="tool_call",
                input_tokens=100,
                output_tokens=20,
                reasoning_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0.001"),
                latency_ns=1000,
            ),
        )


@dataclass(frozen=True)
class _PhenotypeProjection:
    projection_id: str = "typed_json_phenotype"
    projection_version: int = 1
    definition_sha256: str = _sha("typed-json-phenotype")

    def project(self, configuration: FrozenJsonObject) -> str:
        return typed_json_sha256(configuration)


@dataclass
class _Evaluator:
    evaluator_id: str = "test_authoritative_evaluator"
    evaluator_version: int = 1
    definition_sha256: str = _sha("test-evaluator")
    calls: list[str] = field(default_factory=list)

    async def evaluate(self, action):
        self.calls.append(action.action_sha256)
        candidate = EvolutionCandidate(
            occurrence=CandidateOccurrence(
                candidate_id=action.target_candidate_id,
                configuration_hash=action.configuration_sha256,
                configuration_artifact_hash=hashlib.sha256(
                    canonical_typed_json_bytes(action.configuration)
                ).hexdigest(),
                proposal_sequence=len(self.calls),
            ),
            configuration=action.configuration,
            objectives=(("cost", float(len(self.calls))),),
            valid=True,
            generation=action.context.decision_index,
            label=f"evaluated_{len(self.calls)}",
        )
        return MaterializedActionEvaluation(
            action=action,
            candidate=candidate,
            evaluator_receipt_sha256=_sha(
                f"evaluation:{action.action_sha256}"
            ),
        )


@dataclass(frozen=True)
class _FeatureProjection:
    projection_id: str = "test_prequential_features"
    projection_version: int = 1
    definition_sha256: str = _sha("test-prequential-features")
    feature_names: tuple[str, ...] = ("rank_signal",)

    async def project(self, request, proposals):
        return MaterializedActionFeatureBatch(
            projection_id=self.projection_id,
            projection_version=self.projection_version,
            projection_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            feature_names=self.feature_names,
            vectors=tuple(
                sorted(
                    (
                        MaterializedActionFeatureVector(
                            action_sha256=action.action_sha256,
                            values=(
                                1.0
                                if action.native_rank == 1
                                else -1.0,
                            ),
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
                    "strictly_prior": True,
                    "candidate_outcomes_observed": False,
                }
            ),
        )


@dataclass(frozen=True)
class _OutOfSupportFeatureProjection:
    projection_id: str = "test_out_of_support_features"
    projection_version: int = 1
    definition_sha256: str = _sha("test-out-of-support-features")
    feature_names: tuple[str, ...] = ("rank_signal",)

    async def project(self, request, proposals):
        return MaterializedActionFeatureBatch(
            projection_id=self.projection_id,
            projection_version=self.projection_version,
            projection_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            feature_names=self.feature_names,
            vectors=tuple(
                sorted(
                    (
                        MaterializedActionFeatureVector(
                            action_sha256=action.action_sha256,
                            values=(
                                100.0
                                if action.native_rank == 1
                                else -1.0,
                            ),
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
                    "strictly_prior": True,
                    "candidate_outcomes_observed": False,
                }
            ),
        )


@dataclass(frozen=True)
class _StaticScoreAuthority:
    scorer_id: str
    scorer_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"static-score:{self.scorer_id}"),
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
                            value=float(4 - action.native_rank),
                        )
                        for proposal in proposals
                        for action in proposal.actions
                    ),
                    key=lambda value: value.action_sha256,
                )
            ),
            candidate_outcomes_observed=False,
            evidence_sha256=_sha(
                f"static-score-evidence:{self.scorer_id}:"
                f"{request.request_sha256}"
            ),
        )


@dataclass(frozen=True)
class _DirectionSemanticProjection:
    projection_id: str = "test_direction_semantic_cells"
    projection_version: int = 1
    definition_sha256: str = _sha("test-direction-semantic-cells")

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
                            direction_signature=(
                                (
                                    "cost",
                                    (
                                        "decrease"
                                        if action.native_rank <= 2
                                        else "increase"
                                    ),
                                ),
                            ),
                            recursive_lineage=False,
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


@dataclass(frozen=True)
class _ArchiveUtility:
    utility_id: str = "test_additive_archive_utility"
    utility_version: int = 1
    definition_sha256: str = _sha("test-additive-archive-utility")

    def utility(self, candidates):
        return float(
            sum(
                1.0 / (1.0 + candidate.objective_map["cost"])
                for candidate in candidates
                if candidate.valid
            )
        )

    def marginal_utility(self, candidates, objective_point):
        del candidates
        return float(1.0 / (1.0 + objective_point["cost"]))


def _parent_candidate(candidate_id: str, cost: float) -> EvolutionCandidate:
    configuration = freeze_json({"parent": candidate_id})
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(candidate_id),
            configuration_hash=typed_json_sha256(configuration),
            configuration_artifact_hash=hashlib.sha256(
                canonical_typed_json_bytes(configuration)
            ).hexdigest(),
            proposal_sequence=0,
        ),
        configuration=configuration,
        objectives=(("cost", cost),),
        valid=True,
        generation=0,
        label=candidate_id,
    )


def _spec() -> HierarchicalResidualExpertSpec:
    return HierarchicalResidualExpertSpec(
        expert_id="generic_local",
        expert_version=1,
        expert_definition_sha256=_sha("generic-local"),
        instruction="Propose two exact parent-bound local moves.",
        proposal_count=2,
        allowed_radii=(1,),
        allowed_roles=(ResidualProposalRole.LOCAL_EXPLOIT,),
        required_metric_ids=("cost",),
        minimum_distinct_parents=2,
        max_output_tokens=4096,
        temperature=0.0,
    )


def _request(
    *,
    decision_index: int = 3,
    evaluation_slots: int = 1,
) -> ResidualPortfolioDecisionRequest:
    return ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("campaign"),
        prior_state_sha256=_sha(f"prior:{decision_index}"),
        decision_index=decision_index,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=4,
        evaluation_slots=evaluation_slots,
        expert_proposal_slots=(("generic_local", 2),),
        proposal_context=freeze_json(
            {
                "scientific_semantics": {
                    "metric_ids": ["cost"],
                    "directions": ["minimize"],
                },
                "raw_trace_memory": [],
            }
        ),
        reference_escrow_slots=0,
    )


def _semantic_test_request_and_proposal():
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("semantic-campaign"),
        prior_state_sha256=_sha("semantic-prior"),
        decision_index=2,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=4,
        evaluation_slots=2,
        expert_proposal_slots=(("semantic_test", 4),),
        proposal_context=freeze_json({"test": "semantic-coverage"}),
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
        structural_signature_sha256=_sha("semantic-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.trace",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    actions = tuple(
        MaterializedActionDescriptor(
            context=context,
            configuration=freeze_json({"choice": native_rank}),
            phenotype_identity_sha256=_sha(
                f"semantic-phenotype:{native_rank}"
            ),
            expert_id="semantic_test",
            native_rank=native_rank,
            parent_ids=(),
            operator_id="test_mutation",
            target_candidate_id=CandidateId(
                f"candidate_semantic_{native_rank}"
            ),
            role_id="local_exploit",
            normalized_evaluation_cost=1.0,
            reference_action=False,
        )
        for native_rank in range(1, 5)
    )
    proposal = MaterializedActionProposalBatch(
        request_sha256=request.request_sha256,
        expert_id="semantic_test",
        expert_version=1,
        expert_definition_sha256=_sha("semantic-test-expert"),
        actions=actions,
        evidence=freeze_json(
            {
                "candidate_outcomes_observed": False,
                "test_proposal": True,
            }
        ),
    )
    return request, proposal


def _expert(*, observed: tuple[str, ...] = ()):
    schema = _schema()
    evaluator = _Evaluator()
    selections: list[dict[str, object]] = []
    expert = PydanticAIMaterializedHierarchicalResidualExpert(
        spec=_spec(),
        policy=_Policy(),
        action_schema=schema,
        phenotype_projection=_PhenotypeProjection(),
        evaluator=evaluator,
        parent_contexts=tuple(
            ResidualParentActionContext(
                parent_candidate_id=binding.parent_candidate_id,
                position_cell=(
                    "parent.frontier"
                    if index == 0
                    else "parent.reachability"
                ),
            )
            for index, binding in enumerate(schema.bindings)
        ),
        observed_phenotype_sha256s=observed,
        memory_dose_bin=2,
        selection_sink=selections.append,
    )
    return expert, evaluator, selections


def test_generic_expert_materializes_then_evaluates_selected_subset() -> None:
    asyncio.run(_exercise_selected_only_evaluation())


async def _exercise_selected_only_evaluation() -> None:
    expert, evaluator, selections = _expert()
    proposal = await expert.propose(_request())

    assert len(proposal.actions) == 2
    assert len(expert.action_evidence_by_sha256) == 2
    assert len(selections) == 1
    assert selections[0]["candidate_outcomes_observed"] is False
    assert tuple(action.native_rank for action in proposal.actions) == (1, 2)
    assert {
        action.context.parent_position_cell for action in proposal.actions
    } == {"parent.frontier", "parent.reachability"}
    assert all(
        action.context.memory_dose_bin == 2
        for action in proposal.actions
    )

    selected = (min(action.action_sha256 for action in proposal.actions),)
    batch = await expert.evaluate(proposal, selected)

    assert evaluator.calls == list(selected)
    assert batch.selected_action_sha256s == selected
    assert len(batch.evaluations) == 1
    assert batch.evaluations[0].candidate.candidate_id == (
        batch.evaluations[0].action.target_candidate_id
    )


def test_generic_expert_supports_disjoint_pilot_and_continuation_waves() -> None:
    asyncio.run(_exercise_disjoint_evaluation_waves())


async def _exercise_disjoint_evaluation_waves() -> None:
    expert, evaluator, _selections = _expert()
    proposal = await expert.propose(_request())
    first, second = tuple(
        (value.action_sha256,)
        for value in sorted(
            proposal.actions,
            key=lambda value: value.action_sha256,
        )
    )

    pilot = await expert.evaluate(proposal, first)
    continuation = await expert.evaluate(proposal, second)

    assert evaluator.calls == [first[0], second[0]]
    pilot_wave = thaw_json(pilot.evidence)["evaluation_wave"]
    continuation_wave = thaw_json(
        continuation.evidence
    )["evaluation_wave"]
    assert pilot_wave["wave_index"] == 1
    assert pilot_wave["previously_attempted_action_count"] == 0
    assert continuation_wave["wave_index"] == 2
    assert continuation_wave["previously_attempted_action_count"] == 1
    with pytest.raises(
        ValueError,
        match="cannot be evaluated in more than one wave",
    ):
        await expert.evaluate(proposal, first)


def test_generic_expert_filters_observed_phenotype_and_recloses_rank() -> None:
    asyncio.run(_exercise_observed_filter())


async def _exercise_observed_filter() -> None:
    schema = _schema()
    observed = typed_json_sha256(
        schema.bindings[0].contract.resolve(
            "field.a.one"
        ).child_configuration
    )
    expert, _evaluator, _selections = _expert(observed=(observed,))
    proposal = await expert.propose(_request())

    assert len(proposal.actions) == 1
    assert proposal.actions[0].native_rank == 1
    evidence = next(iter(expert.action_evidence_by_sha256.values()))
    assert evidence.provider_rank == 2
    assert evidence.materialized_rank == 1


def test_generic_slate_policies_enforce_phenotype_uniqueness() -> None:
    asyncio.run(_exercise_generic_slate_policies())


async def _exercise_generic_slate_policies() -> None:
    expert, _evaluator, _selections = _expert()
    proposal = await expert.propose(_request())
    first, second = proposal.actions
    feasibility = UniquePhenotypeMaterializedSlateFeasibility()
    value = ZeroMaterializedSlateValue()

    assert feasibility.permits((first, second))
    duplicate = type(first)(
        context=first.context,
        configuration=first.configuration,
        phenotype_identity_sha256=first.phenotype_identity_sha256,
        expert_id=first.expert_id,
        native_rank=3,
        parent_ids=first.parent_ids,
        operator_id=first.operator_id,
        target_candidate_id=CandidateId("candidate_duplicate"),
        role_id=first.role_id,
        normalized_evaluation_cost=1.0,
        reference_action=False,
    )
    assert not feasibility.permits((first, duplicate))
    assert value.value((first, second)) == 0.0


def test_frozen_hurdle_scorers_are_generic_and_outcome_blind() -> None:
    asyncio.run(_exercise_frozen_hurdle_scorers())


async def _exercise_frozen_hurdle_scorers() -> None:
    expert, _evaluator, _selections = _expert()
    request = _request()
    proposal = await expert.propose(request)
    projection = _FeatureProjection()
    positive_model = FrozenStandardizedLinearModel(
        model_id="test_positive",
        family="logistic",
        feature_names=projection.feature_names,
        means=(0.0,),
        scales=(1.0,),
        coefficients=(0.0, 1.0),
    )
    magnitude_model = FrozenStandardizedLinearModel(
        model_id="test_magnitude",
        family="log1p_ridge",
        feature_names=projection.feature_names,
        means=(0.0,),
        scales=(1.0,),
        coefficients=(math.log(2.0), 0.0),
    )
    positive = FrozenHurdleMaterializedActionScorer(
        scorer_id="test_positive_probability",
        projection=projection,
        positive_model=positive_model,
        magnitude_model=magnitude_model,
        score_kind=FrozenHurdleScoreKind.POSITIVE_PROBABILITY,
        source_fit_sha256=_sha("fit"),
    )
    magnitude = FrozenHurdleMaterializedActionScorer(
        scorer_id="test_expected_magnitude",
        projection=projection,
        positive_model=positive_model,
        magnitude_model=magnitude_model,
        score_kind=FrozenHurdleScoreKind.EXPECTED_POSITIVE_MAGNITUDE,
        source_fit_sha256=_sha("fit"),
    )

    positive_batch, magnitude_batch = await asyncio.gather(
        positive.score(request, (proposal,)),
        magnitude.score(request, (proposal,)),
    )

    assert not positive_batch.candidate_outcomes_observed
    assert not magnitude_batch.candidate_outcomes_observed
    positive_by_action = {
        value.action_sha256: value.value
        for value in positive_batch.scores
    }
    magnitude_by_action = {
        value.action_sha256: value.value
        for value in magnitude_batch.scores
    }
    first, second = proposal.actions
    assert positive_by_action[first.action_sha256] > (
        positive_by_action[second.action_sha256]
    )
    assert magnitude_by_action[first.action_sha256] == pytest.approx(
        positive_by_action[first.action_sha256]
    )
    assert magnitude_by_action[second.action_sha256] == pytest.approx(
        positive_by_action[second.action_sha256]
    )
    winsorized = WinsorizedFrozenHurdleMaterializedActionScorer(
        scorer_id="test_winsorized_probability",
        projection=_OutOfSupportFeatureProjection(),
        positive_model=positive_model,
        magnitude_model=FrozenStandardizedLinearModel(
            model_id="test_independent_magnitude_standardizer",
            family="log1p_ridge",
            feature_names=projection.feature_names,
            means=(10.0,),
            scales=(2.0,),
            coefficients=(math.log(2.0), 0.0),
        ),
        score_kind=FrozenHurdleScoreKind.POSITIVE_PROBABILITY,
        source_fit_sha256=_sha("winsorized-fit"),
        winsorization_limit=3.0,
    )
    winsorized_batch = await winsorized.score(request, (proposal,))
    winsorized_by_action = {
        value.action_sha256: value.value
        for value in winsorized_batch.scores
    }
    assert winsorized_by_action[first.action_sha256] == pytest.approx(
        1.0 / (1.0 + math.exp(-3.0))
    )
    assert winsorized_by_action[second.action_sha256] == pytest.approx(
        1.0 / (1.0 + math.exp(1.0))
    )


def test_support_guard_and_independent_score_authorities_are_prequential() -> None:
    asyncio.run(_exercise_support_guard_and_score_authorities())


async def _exercise_support_guard_and_score_authorities() -> None:
    expert, _evaluator, _selections = _expert()
    request = _request(evaluation_slots=2)
    proposal = await expert.propose(request)
    projection = _OutOfSupportFeatureProjection()
    positive_model = FrozenStandardizedLinearModel(
        model_id="test_guarded_positive",
        family="logistic",
        feature_names=projection.feature_names,
        means=(0.0,),
        scales=(1.0,),
        coefficients=(0.0, 1.0),
    )
    magnitude_model = FrozenStandardizedLinearModel(
        model_id="test_guarded_magnitude",
        family="log1p_ridge",
        feature_names=projection.feature_names,
        means=(0.0,),
        scales=(1.0,),
        coefficients=(math.log(2.0), 0.0),
    )
    guarded = SupportGuardedFrozenHurdleMaterializedActionScorer(
        scorer_id="test_guarded_transfer",
        projection=projection,
        positive_model=positive_model,
        magnitude_model=magnitude_model,
        score_kind=FrozenHurdleScoreKind.POSITIVE_PROBABILITY,
        source_fit_sha256=_sha("guarded-fit"),
        support_groups=(
            FrozenFeatureSupportGroup(
                group_id="structural",
                feature_names=("rank_signal",),
            ),
        ),
        support_radius=3.0,
        winsorization_limit=3.0,
    )
    guarded_batch = await guarded.score(request, (proposal,))
    support = guarded.evidence_for(request.request_sha256)
    assert support is not None
    assert support.group_rows == (
        ("structural", 100.0, pytest.approx(0.03)),
    )
    guarded_scores = {
        value.action_sha256: value.value
        for value in guarded_batch.scores
    }
    first, second = proposal.actions
    assert guarded_scores[first.action_sha256] > (
        guarded_scores[second.action_sha256]
    )
    assert guarded_scores[first.action_sha256] < 0.55

    native = NativeRankMaterializedActionScorer()
    native_batch = await native.score(request, (proposal,))
    native_scores = {
        value.action_sha256: value.value for value in native_batch.scores
    }
    assert native_scores[first.action_sha256] == 1.0
    assert native_scores[second.action_sha256] == 0.5

    ledger = MaterializedActionEvidenceLedger()
    empirical = TargetEmpiricalReturnMaterializedActionScorer(
        broker=RegretBrokeredMaterializedActionPolicy(ledger=ledger),
    )
    cold_batch = await empirical.score(request, (proposal,))
    cold_scores = {
        value.action_sha256: value.value for value in cold_batch.scores
    }
    assert cold_scores[first.action_sha256] > cold_scores[
        second.action_sha256
    ]
    adaptive = ReliabilityAdaptiveScorePortfolioPolicy(
        scorers=(native, empirical, guarded),
        primary_scorer_id=guarded.scorer_id,
        primary_reliability=guarded,
        reliability_component_ids=("structural",),
        minimum_primary_fraction=0.5,
    )
    requirement = await adaptive.require(request, (proposal,))
    assert len(requirement.required_action_sha256s) == 2
    adaptive_evidence = thaw_json(requirement.evidence)
    assert adaptive_evidence["effective_quotas"] == {
        "provider_native_rank": 1,
        "target_empirical_return": 0,
        "test_guarded_transfer": 1,
    }
    assert float.fromhex(
        adaptive_evidence["primary_reliability"][
            "overall_reliability_hex"
        ]
    ) == pytest.approx(0.03)
    in_support = SupportGuardedFrozenHurdleMaterializedActionScorer(
        scorer_id="test_guarded_in_support",
        projection=_FeatureProjection(),
        positive_model=positive_model,
        magnitude_model=magnitude_model,
        score_kind=FrozenHurdleScoreKind.POSITIVE_PROBABILITY,
        source_fit_sha256=_sha("guarded-fit"),
        support_groups=(
            FrozenFeatureSupportGroup(
                group_id="structural",
                feature_names=("rank_signal",),
            ),
        ),
        support_radius=3.0,
        winsorization_limit=3.0,
    )
    full_authority = ReliabilityAdaptiveScorePortfolioPolicy(
        scorers=(native, empirical, in_support),
        primary_scorer_id=in_support.scorer_id,
        primary_reliability=in_support,
        reliability_component_ids=("structural",),
        minimum_primary_fraction=0.5,
    )
    full_requirement = await full_authority.require(request, (proposal,))
    full_evidence = thaw_json(full_requirement.evidence)
    assert full_evidence["effective_quotas"] == {
        "provider_native_rank": 0,
        "target_empirical_return": 0,
        "test_guarded_in_support": 2,
    }
    assert float.fromhex(
        full_evidence["primary_reliability"][
            "overall_reliability_hex"
        ]
    ) == 1.0
    ledger.append_outcome(
        MaterializedActionOutcome(
            action=first,
            realized=True,
            feasible=True,
            normalized_archive_gain=0.25,
            positive_marginal_utility=True,
        )
    )

    next_expert, _next_evaluator, _next_selections = _expert()
    next_request = _request(decision_index=4, evaluation_slots=2)
    next_proposal = await next_expert.propose(next_request)
    warm_batch = await empirical.score(next_request, (next_proposal,))
    assert all(
        value.value >= 0.25 for value in warm_batch.scores
    )
    next_first = next_proposal.actions[0]
    ledger.append_outcome(
        MaterializedActionOutcome(
            action=next_first,
            realized=True,
            feasible=True,
            normalized_archive_gain=0.1,
            positive_marginal_utility=True,
        )
    )
    with pytest.raises(ValueError, match="crosses the decision cutoff"):
        await empirical.score(next_request, (next_proposal,))


def test_adaptive_portfolio_exact_feasibility_refills_same_authority() -> None:
    asyncio.run(_exercise_exact_feasibility_refill())


async def _exercise_exact_feasibility_refill() -> None:
    expert, _evaluator, _selections = _expert()
    request = _request(evaluation_slots=1)
    proposal = await expert.propose(request)
    first, second = proposal.actions
    projection = _FeatureProjection()
    positive_model = FrozenStandardizedLinearModel(
        model_id="test_refill_positive",
        family="logistic",
        feature_names=projection.feature_names,
        means=(0.0,),
        scales=(1.0,),
        coefficients=(0.0, 1.0),
    )
    magnitude_model = FrozenStandardizedLinearModel(
        model_id="test_refill_magnitude",
        family="log1p_ridge",
        feature_names=projection.feature_names,
        means=(0.0,),
        scales=(1.0,),
        coefficients=(math.log(2.0), 0.0),
    )
    primary = SupportGuardedFrozenHurdleMaterializedActionScorer(
        scorer_id="test_refill_primary",
        projection=projection,
        positive_model=positive_model,
        magnitude_model=magnitude_model,
        score_kind=FrozenHurdleScoreKind.POSITIVE_PROBABILITY,
        source_fit_sha256=_sha("refill-fit"),
        support_groups=(
            FrozenFeatureSupportGroup(
                group_id="structural",
                feature_names=("rank_signal",),
            ),
        ),
    )
    ledger = MaterializedActionEvidenceLedger()
    native = NativeRankMaterializedActionScorer()
    empirical = TargetEmpiricalReturnMaterializedActionScorer(
        broker=RegretBrokeredMaterializedActionPolicy(ledger=ledger),
    )
    hard_feasibility = _ExactRejectedConfiguration(
        rejected_configuration_sha256=first.configuration_sha256,
    )
    policy = ReliabilityAdaptiveScorePortfolioPolicy(
        scorers=(native, empirical, primary),
        primary_scorer_id=primary.scorer_id,
        primary_reliability=primary,
        reliability_component_ids=("structural",),
        minimum_primary_fraction=1.0,
        hard_feasibility=hard_feasibility,
    )

    requirement = await policy.require(request, (proposal,))

    assert requirement.required_action_sha256s == (
        second.action_sha256,
    )
    evidence = thaw_json(requirement.evidence)
    feasibility = evidence["hard_feasibility"]
    assert feasibility["enabled"] is True
    assert feasibility["candidate_outcomes_observed"] is False
    assert feasibility["unknown_actions_remain_eligible"] is True
    assert feasibility["refill"] == "same_authority_frozen_ranking"
    assert feasibility["verdict_counts"] == {
        "feasible": 0,
        "infeasible": 1,
        "unknown": 1,
    }
    assert feasibility["rejected_actions"][0]["action_sha256"] == (
        first.action_sha256
    )
    selected = evidence["selected_by_scorer"]["test_refill_primary"]
    assert selected == [
        {
            "action_sha256": second.action_sha256,
            "rank_position": 2,
            "score_hex": selected[0]["score_hex"],
        }
    ]


def test_semantic_coverage_policy_recovers_a_distinct_tradeoff_cell() -> None:
    asyncio.run(_exercise_semantic_coverage_policy())


async def _exercise_semantic_coverage_policy() -> None:
    request, proposal = _semantic_test_request_and_proposal()
    primary = _StaticScoreAuthority("primary_rank")
    transport = _StaticScoreAuthority("transport_rank")
    policy = SemanticCoverageScorePortfolioPolicy(
        scorers=(primary, transport),
        scorer_capacity_fractions=(
            ("primary_rank", 0.5),
            ("transport_rank", 0.0),
        ),
        lineage_scorer_id="transport_rank",
        lineage_member_scorer_id="transport_rank",
        lineage_deficit_refill_scorer_id="primary_rank",
        lineage_capacity_fraction=0.5,
        semantic_projection=_DirectionSemanticProjection(),
        coverage_strength=1.0 / 3.0,
    )

    requirement = await policy.require(request, (proposal,))

    first, _second, third, _fourth = proposal.actions
    assert requirement.required_action_sha256s == tuple(
        sorted((first.action_sha256, third.action_sha256))
    )
    evidence = thaw_json(requirement.evidence)
    assert evidence["nominal_capacity_quotas"] == {
        "primary_rank": 1,
        "recursive_lineage": 1,
        "transport_rank": 0,
    }
    assert evidence["realized_lineage_count"] == 0
    assert evidence["lineage_deficit_refill_count"] == 1
    assert evidence["candidate_outcomes_observed"] is False
    assert [
        value["direction_signature"][0]["direction"]
        for value in evidence["selection_trace"]
    ] == ["decrease", "increase"]
    assert float.fromhex(
        evidence["selection_trace"][1]["direction_novelty_hex"]
    ) == 1.0


def test_pydantic_semantic_projection_uses_p50_fallback_and_lineage() -> None:
    asyncio.run(_exercise_pydantic_semantic_projection())


def test_semantic_policy_enforces_exact_lineage_partition() -> None:
    asyncio.run(_exercise_exact_lineage_partition())


async def _exercise_exact_lineage_partition() -> None:
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("exact-lineage-campaign"),
        prior_state_sha256=_sha("exact-lineage-prior"),
        decision_index=2,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=8,
        evaluation_slots=4,
        expert_proposal_slots=(
            ("nonrecursive_test", 2),
            ("recursive_a", 2),
            ("recursive_b", 2),
        ),
        proposal_context=freeze_json({"test": "exact-lineage-partition"}),
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
        structural_signature_sha256=_sha("exact-lineage-structure"),
        patch_compatibility_cell="test.compatible",
        forecast_calibration_cell="test.trace",
        source_distance_bin=1,
        memory_dose_bin=0,
    )

    def proposal(
        expert_id: str,
        native_ranks: tuple[int, int],
        *,
        recursive: bool,
    ) -> MaterializedActionProposalBatch:
        actions = tuple(
            MaterializedActionDescriptor(
                context=context,
                configuration=freeze_json(
                    {"expert": expert_id, "native_rank": native_rank}
                ),
                phenotype_identity_sha256=_sha(
                    f"exact-lineage-phenotype:{expert_id}:{native_rank}"
                ),
                expert_id=expert_id,
                native_rank=native_rank,
                parent_ids=(
                    (CandidateId(f"candidate_generated_{expert_id}"),)
                    if recursive
                    else ()
                ),
                operator_id="test_mutation",
                target_candidate_id=CandidateId(
                    f"candidate_exact_{expert_id}_{native_rank}"
                ),
                role_id="local_exploit",
                normalized_evaluation_cost=1.0,
                reference_action=False,
            )
            for native_rank in native_ranks
        )
        return MaterializedActionProposalBatch(
            request_sha256=request.request_sha256,
            expert_id=expert_id,
            expert_version=1,
            expert_definition_sha256=_sha(
                f"exact-lineage-expert:{expert_id}"
            ),
            actions=actions,
            evidence=freeze_json(
                {
                    "candidate_outcomes_observed": False,
                    "test_proposal": True,
                }
            ),
        )

    proposals = (
        proposal("nonrecursive_test", (1, 2), recursive=False),
        proposal("recursive_a", (1, 2), recursive=True),
        proposal("recursive_b", (1, 2), recursive=True),
    )

    @dataclass(frozen=True)
    class _ExactLineageProjection:
        projection_id: str = "test_exact_lineage_cells"
        projection_version: int = 1
        definition_sha256: str = _sha("test-exact-lineage-cells")

        async def project(self, projected_request, projected_proposals):
            return MaterializedActionSemanticCellBatch(
                projection_id=self.projection_id,
                projection_version=self.projection_version,
                projection_definition_sha256=self.definition_sha256,
                residual_request_sha256=projected_request.request_sha256,
                proposal_sha256s=tuple(
                    sorted(
                        value.proposal_sha256
                        for value in projected_proposals
                    )
                ),
                cells=tuple(
                    sorted(
                        (
                            MaterializedActionSemanticCell(
                                action_sha256=action.action_sha256,
                                direction_signature=(
                                    ("cost", "decrease"),
                                ),
                                recursive_lineage=bool(action.parent_ids),
                            )
                            for value in projected_proposals
                            for action in value.actions
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

    primary = _StaticScoreAuthority("primary_rank")
    transport = _StaticScoreAuthority("transport_rank")
    policy = SemanticCoverageScorePortfolioPolicy(
        scorers=(primary, transport),
        scorer_capacity_fractions=(
            ("primary_rank", 0.5),
            ("transport_rank", 0.0),
        ),
        lineage_scorer_id="transport_rank",
        lineage_member_scorer_id="transport_rank",
        lineage_deficit_refill_scorer_id="primary_rank",
        lineage_capacity_fraction=0.5,
        semantic_projection=_ExactLineageProjection(),
        coverage_strength=1.0 / 3.0,
    )

    requirement = await policy.require(request, proposals)
    selected = set(requirement.required_action_sha256s)
    recursive_actions = {
        action.action_sha256
        for value in proposals
        for action in value.actions
        if action.parent_ids
    }
    recursive_native_champions = {
        value.actions[0].action_sha256 for value in proposals[1:]
    }
    nonrecursive_actions = {
        action.action_sha256 for action in proposals[0].actions
    }
    assert selected & recursive_actions == recursive_native_champions
    assert selected & nonrecursive_actions == nonrecursive_actions
    assert len(selected & recursive_actions) == 2
    evidence = thaw_json(requirement.evidence)
    assert evidence["realized_lineage_count"] == 2
    assert evidence["lineage_deficit_refill_count"] == 0
    assert evidence["lineage_member_scorer_id"] == "transport_rank"
    assert evidence["lineage_partition"] == (
        "exact_maximum_then_nonrecursive_score_lanes"
    )

    unlocked_policy = SemanticCoverageScorePortfolioPolicy(
        scorers=(primary, transport),
        scorer_capacity_fractions=(
            ("primary_rank", 0.5),
            ("transport_rank", 0.0),
        ),
        lineage_scorer_id="transport_rank",
        lineage_member_scorer_id="transport_rank",
        lineage_deficit_refill_scorer_id="primary_rank",
        lineage_capacity_fraction=0.5,
        semantic_projection=_ExactLineageProjection(),
        coverage_strength=1.0 / 3.0,
        allow_recursive_score_lane_spillover=True,
    )
    plan = await FrozenBranchSequentialLineagePlanner(
        locked_policy=policy,
        unlocked_policy=unlocked_policy,
    ).plan(request, proposals)
    assert set(plan.pilot_action_sha256s) == recursive_native_champions
    assert set(plan.pilot_action_sha256s).issubset(
        plan.locked_requirement.required_action_sha256s
    )
    assert set(plan.pilot_action_sha256s).issubset(
        plan.unlocked_requirement.required_action_sha256s
    )
    assert (
        thaw_json(plan.unlocked_requirement.evidence)[
            "lineage_partition"
        ]
        == "pilot_floor_then_recursive_score_lane_competition"
    )

    action_by_sha256 = {
        action.action_sha256: action
        for value in proposals
        for action in value.actions
    }
    evaluator = _Evaluator()
    pilot_evaluations = tuple(
        [
            await evaluator.evaluate(action_by_sha256[action_sha256])
            for action_sha256 in plan.pilot_action_sha256s
        ]
    )
    outcomes = CandidateArchiveMarginalPilotOutcomeProjector(
        prior_candidates=(_parent_candidate("candidate_pilot_prior", 10.0),),
        utility=_ArchiveUtility(),
    ).project(plan, pilot_evaluations)
    gate = AnyPositiveSequentialLineageGate().decide(plan, outcomes)
    assert gate.branch is SequentialLineageBranch.UNLOCKED
    assert gate.positive_pilot_count == len(plan.pilot_action_sha256s)
    assert (
        gate.selected_requirement_sha256
        == plan.unlocked_requirement.requirement_sha256
    )


async def _exercise_pydantic_semantic_projection() -> None:
    expert, _evaluator, _selections = _expert()
    request = _request()
    proposal = await expert.propose(request)
    parents = (
        CandidateId("candidate_parent_a"),
        CandidateId("candidate_parent_b"),
    )
    projection = PydanticAIResidualSemanticCellProjection(
        initial_candidate_ids=parents,
        evidence_sources=(expert,),
    )
    batch = await projection.project(request, (proposal,))

    assert all(
        value.direction_signature == (("cost", "decrease"),)
        for value in batch.cells
    )
    assert not any(value.recursive_lineage for value in batch.cells)
    evidence = thaw_json(batch.evidence)
    assert evidence["typed_residual_evidence_count"] == 2
    assert evidence["candidate_outcomes_observed"] is False

    recursive_projection = PydanticAIResidualSemanticCellProjection(
        initial_candidate_ids=(parents[0],),
        evidence_sources=(expert,),
    )
    recursive_batch = await recursive_projection.project(
        request,
        (proposal,),
    )
    by_action = {
        value.action_sha256: value for value in recursive_batch.cells
    }
    assert not by_action[proposal.actions[0].action_sha256].recursive_lineage
    assert by_action[proposal.actions[1].action_sha256].recursive_lineage


def test_portable_feature_groups_partition_the_feature_abi() -> None:
    grouped = tuple(
        feature_name
        for _group_id, feature_names in (
            PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS
        )
        for feature_name in feature_names
    )
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(
        PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES
    )


def test_portable_consequence_projection_uses_generic_prior_geometry() -> None:
    asyncio.run(_exercise_portable_consequence_projection())


def test_residual_forecast_geometry_is_sense_aware_and_outcome_blind() -> None:
    asyncio.run(_exercise_residual_forecast_geometry_projection())


def test_residual_forecast_geometry_ignores_ineligible_empty_objective_ledger_rows() -> (
    None
):
    asyncio.run(
        _exercise_residual_forecast_geometry_projection(
            include_ineligible_ledger_row=True
        )
    )


def test_residual_forecast_geometry_rejects_eligible_incomplete_objectives() -> (
    None
):
    expert, _evaluator, _selections = _expert()
    incomplete = replace(
        _parent_candidate("candidate_incomplete_eligible", 5.0),
        objectives=(("different_metric", 5.0),),
    )
    with pytest.raises(
        ValueError,
        match="eligible prior candidate objectives differ from senses",
    ):
        PydanticAIResidualForecastGeometryProjection(
            prior_candidates=(
                _parent_candidate("candidate_parent_a", 3.0),
                incomplete,
            ),
            evidence_sources=(expert,),
            objective_senses=(("cost", MetricSense.MINIMIZE),),
        )


def test_residual_forecast_geometry_can_project_heterogeneous_partial_evidence() -> (
    None
):
    asyncio.run(_exercise_partial_residual_forecast_geometry_projection())


async def _exercise_partial_residual_forecast_geometry_projection() -> None:
    expert, _evaluator, _selections = _expert()
    request = replace(
        _request(),
        expert_proposal_slots=(
            ("generic_local", 2),
            ("numerical_test", 1),
        ),
    )
    residual_proposal = await expert.propose(request)
    numerical_action = replace(
        residual_proposal.actions[0],
        expert_id="numerical_test",
        native_rank=1,
        phenotype_identity_sha256=_sha("numerical-test-phenotype"),
        operator_id="numerical_acquisition",
        role_id="numerical_baseline",
        target_candidate_id=CandidateId("candidate_numerical_test"),
    )
    numerical_proposal = MaterializedActionProposalBatch(
        request_sha256=request.request_sha256,
        expert_id="numerical_test",
        expert_version=1,
        expert_definition_sha256=_sha("numerical-test-expert"),
        actions=(numerical_action,),
        evidence=freeze_json(
            {
                "candidate_outcomes_observed": False,
                "forecast_evidence_authored": False,
                "test_numerical_proposal": True,
            }
        ),
    )
    parents = (
        _parent_candidate("candidate_parent_a", 3.0),
        _parent_candidate("candidate_parent_b", 4.0),
    )
    complete = PydanticAIResidualForecastGeometryProjection(
        prior_candidates=parents,
        evidence_sources=(expert,),
        objective_senses=(("cost", MetricSense.MINIMIZE),),
    )
    with pytest.raises(
        ValueError,
        match="exactly one forecast evidence row",
    ):
        await complete.project(
            request,
            (residual_proposal, numerical_proposal),
        )

    available = PydanticAIResidualForecastGeometryProjection(
        prior_candidates=parents,
        evidence_sources=(expert,),
        objective_senses=(("cost", MetricSense.MINIMIZE),),
        coverage_mode=(
            ResidualForecastEvidenceCoverageMode.AVAILABLE_ONLY
        ),
    )
    batch = await available.project(
        request,
        (residual_proposal, numerical_proposal),
    )
    assert len(batch.members) == 2
    assert {
        value.action_sha256 for value in batch.members
    } == {
        value.action_sha256 for value in residual_proposal.actions
    }
    evidence = thaw_json(batch.evidence)
    assert evidence["coverage_mode"] == "available_forecast_evidence"
    assert evidence["proposal_action_count"] == 3
    assert evidence["complete_action_coverage"] is False
    assert evidence["unprojected_action_sha256s"] == [
        numerical_action.action_sha256
    ]
    assert evidence["unprojected_actions_remain_fallback_eligible"] is True
    assert evidence["candidate_outcomes_observed"] is False
    assert available.definition_sha256 != complete.definition_sha256


async def _exercise_residual_forecast_geometry_projection(
    *,
    include_ineligible_ledger_row: bool = False,
) -> None:
    expert, _evaluator, _selections = _expert()
    request = _request()
    proposal = await expert.propose(request)
    parents: tuple[EvolutionCandidate, ...] = (
        _parent_candidate("candidate_parent_a", 3.0),
        _parent_candidate("candidate_parent_b", 4.0),
    )
    if include_ineligible_ledger_row:
        parents = (
            *parents,
            replace(
                _parent_candidate("candidate_invalid_ledger_row", 5.0),
                objectives=(),
                valid=False,
            ),
        )
    projection = PydanticAIResidualForecastGeometryProjection(
        prior_candidates=parents,
        evidence_sources=(expert,),
        objective_senses=(("cost", MetricSense.MINIMIZE),),
    )

    batch = await projection.project(request, (proposal,))
    assert batch.candidate_outcomes_observed is False
    assert len(batch.members) == 2
    parent_by_id = {
        value.candidate_id: value for value in parents
    }
    for member in batch.members:
        evidence = expert.evidence_for(member.action_sha256)
        assert evidence is not None
        parent_cost = parent_by_id[
            evidence.plan.parent_candidate_id
        ].objective_map["cost"]
        assert member.scenario("lower_numeric").as_mapping() == {
            "cost": parent_cost - 2.0
        }
        assert member.scenario("median").as_mapping() == {
            "cost": parent_cost - 1.0
        }
        assert member.scenario("upper_numeric").as_mapping() == {
            "cost": parent_cost
        }
        assert member.scenario("favorable").as_mapping() == (
            member.scenario("lower_numeric").as_mapping()
        )
        assert member.scenario("adverse").as_mapping() == (
            member.scenario("upper_numeric").as_mapping()
        )
        assert member.reliability == (
            evidence.probability_valid * 0.75
        )
    cached = await projection.project(request, (proposal,))
    assert cached is batch


async def _exercise_portable_consequence_projection() -> None:
    expert, _evaluator, _selections = _expert()
    request = _request()
    proposal = await expert.propose(request)
    parents = (
        _parent_candidate("candidate_parent_a", 3.0),
        _parent_candidate("candidate_parent_b", 4.0),
    )
    projection = PortableResidualConsequenceFeatureProjection(
        prior_candidates=parents,
        initial_candidate_ids=tuple(
            sorted(
                (candidate.candidate_id for candidate in parents),
                key=lambda value: value.value,
            )
        ),
        evidence_sources=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
        ),
        archive_utility=_ArchiveUtility(),
    )

    batch = await projection.project(request, (proposal,))

    assert batch.feature_names == (
        PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES
    )
    assert len(batch.vectors) == 2
    assert all(
        len(value.values)
        == len(PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES)
        for value in batch.vectors
    )
    assert batch.candidate_outcomes_observed is False
    opportunity_index = batch.feature_names.index(
        "parent_leave_one_out_contribution"
    )
    assert all(
        value.values[opportunity_index] > 0.0
        for value in batch.vectors
    )
    cached = await projection.project(request, (proposal,))
    assert cached is batch

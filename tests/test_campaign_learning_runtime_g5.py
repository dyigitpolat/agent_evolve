"""Provider-free G1--G5 proof of reflection, diagnostics, and robust quarantine."""

from __future__ import annotations

import asyncio
import hashlib
from fractions import Fraction

import pytest

from agent_evolve.agentic import (
    AgenticBenchmark,
    DeterministicIdFactory,
    InsightDraft,
    InsightMemoryBank,
    PortfolioCard,
    PortfolioSelectionRequest,
    TypedConfigurationPhenotypeIdentityPolicy,
    compose_portfolio_evolution,
)
from agent_evolve.application.budgeted_optimizer import OptimizerBudget
from agent_evolve.application.campaign_execution import (
    CampaignExecutionEvent,
    CampaignJournalAck,
    EvolutionCampaignScheduler,
)
from agent_evolve.application.campaign_learning import (
    CampaignInsightPromotionPolicy,
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.campaign_evidence_registry import (
    CampaignEvidenceRegistry,
)
from agent_evolve.application.campaign_generation_audit import (
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.campaign_learning_runtime import (
    CampaignReflectionLearningRecord,
    CampaignReflectionLearningRecordCodec,
    ClosedLoopCampaignLearningRuntime,
    StructuredCampaignReflectionLearningProjector,
)
from agent_evolve.application.finite_action_hypothesis_semantics import (
    PortableFiniteActionHypothesisMatcher,
    PortableFiniteActionInsightSemanticCompiler,
)
from agent_evolve.application.evolution_campaign import (
    AlternatingPortfolioRecombinationCadence,
    ArchiveUtilitySnapshot,
    CampaignAgentRuntimeReceipt,
    CampaignConcurrency,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    CampaignSeed,
    EvolutionCampaign,
)
from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightLifecycleState,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CAMPAIGN_IDENTIFIABLE_REFLECTION_BINDING_KEY,
    AgenticPortfolioCampaignRuntime,
    ArchiveReservoirCampaignParentSelector,
    CampaignIdentifiableReflectionInput,
    CampaignPortfolioWaveContext,
    CommittedRegistryIdentifiableReflectionEvidenceSource,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryCreditPlan,
    PortfolioRewardAggregationBinding,
    PortfolioVariationWaveRequest,
)
from agent_evolve.application.portfolio_hypothesis_observations import (
    FinitePortfolioActionSemanticsCompiler,
    ObjectiveDeltaMetricEffectProjector,
)
from agent_evolve.application.portfolio_projection import (
    admit_portfolio_card_sources,
    portfolio_card_from_insight_entry,
)
from agent_evolve.campaign_workload import (
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.memory.global_falsification import (
    GLOBAL_FALSIFICATION_POLICY_ID,
    GLOBAL_FALSIFICATION_POLICY_VERSION,
    HypothesisAuditScope,
)
from agent_evolve.policies.memory.randomized_subset import (
    InsightSelectionDecision,
    InsightSelectionMode,
)
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    MemoryAssignmentArm,
    ResolvedInsightAssignment,
)
from agent_evolve.policies.selection.random_portfolio import (
    DeterministicRandomFeasiblePortfolioPolicy,
)
from agent_evolve.ports.agentic_generator import (
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionEvidenceCatalog,
    ReflectionInsightContract,
    ReflectionInsightKind,
)
from examples.benchmarks.boils_abc.actions import (
    DEFAULT_ACTION_SEQUENCE,
    config_sha256,
    normalize_candidate,
)
from examples.benchmarks.boils_abc.budgeted_v5_support import PARENT_C_SEQUENCE
from examples.benchmarks.boils_abc.evaluator import (
    AbcEvaluatorSettings,
    BoilsEvaluation,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


class _Evaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, config: object) -> BoilsEvaluation:
        self.calls += 1
        sequence = normalize_candidate(config)
        weighted = sum(
            (index + 1) * (sum(action.encode("ascii")) % 23)
            for index, action in enumerate(sequence)
        )
        return BoilsEvaluation(
            configuration_sha256=config_sha256(config),
            sequence=sequence,
            abc_binary_sha256=_sha("provider-free-g5-abc"),
            lut_inputs=6,
            circuit_results=(),
            total_lut_count=8_000 + weighted % 2_000,
            total_levels=50 + (weighted // 7) % 30,
            max_levels=50 + (weighted // 7) % 30,
            elapsed_s=0.0,
            affinity_queue_wait_s=0.0,
            cpu_affinity=None,
        )


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - materialized path.
        raise AssertionError(request)

    async def reflect(self, request):  # pragma: no cover - external executor.
        raise AssertionError(request)


class _Evidence:
    def initialize_memory(self, benchmark, session, seeds):
        del benchmark, session
        return _object({"seed_ids": [value.seed_id for value in seeds.seeds]})

    def context(self, benchmark, session, parent, variation, memory):
        del benchmark, session, parent, variation, memory
        # An exact shared estimand context lets the two parent lanes identify a
        # selected-vs-unselected card contrast without benchmark-specific state.
        return _object({"estimand": "provider_free_closed_loop_g5"})

    def cards(self, benchmark, session, parent, variation, memory):
        del benchmark, session, parent, variation, memory
        return (_object({"evidence": "sealed_provider_free_fixture"}),)


def _reflection_contract() -> ReflectionInsightContract:
    return ReflectionInsightContract(
        required_metric_ids=("total_levels", "total_lut_count"),
        allowed_option_families=("sequence_rewrite",),
        allowed_decision_paths=("$.sequence",),
        allowed_insight_kinds=(ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,),
        allowed_consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        allowed_comparison_anchor_kinds=(MetricComparisonAnchorKind.CURRENT_PARENT,),
        allowed_factor_capabilities=("sequence_rewrite",),
    )


def _reflection_drafts(
    generation: int,
    contrast_ids: tuple[str, ...],
) -> tuple[InsightDraft, ...]:
    if len(contrast_ids) < 2:
        raise ValueError("provider-free reflection fixture requires two contrasts")
    return tuple(
        InsightDraft(
            claim=f"Generation {generation} typed hypothesis {label}",
            trigger="A typed sequence mutation is available.",
            mechanism="The mutation may improve the sealed quality endpoint.",
            affected_paths=("$.sequence",),
            evidence_summary="A sealed recombination contrast motivated the test.",
            confidence=0.5,
            evidence_contrast_ids=(contrast_id,),
            effect_predictions=(
                MetricEffectPrediction(
                    metric_id="total_levels",
                    direction=MetricEffectDirection.DECREASE,
                    comparison_anchor=MetricComparisonAnchor(
                        MetricComparisonAnchorKind.CURRENT_PARENT
                    ),
                ),
                MetricEffectPrediction(
                    metric_id="total_lut_count",
                    direction=MetricEffectDirection.DECREASE,
                    comparison_anchor=MetricComparisonAnchor(
                        MetricComparisonAnchorKind.CURRENT_PARENT
                    ),
                ),
            ),
            recommended_option_families=("sequence_rewrite",),
            action_template="Apply one sealed typed sequence mutation.",
            falsification_condition="The quality endpoint does not increase.",
            insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
            factor_capabilities=("sequence_rewrite",),
        )
        for label, contrast_id in zip(("a", "b"), contrast_ids, strict=False)
    )


def _reflection_learning_record(
    *,
    generation: int,
    source_stage_receipt_sha256: str,
    call_id: LLMCallId,
    source_operator_invocation_ids: tuple[OperatorInvocationId, ...],
    source_candidate_ids: tuple[CandidateId, ...],
    contrast_ids: tuple[str, ...],
    origin_cutoff_event_index: int | None = None,
    reflection_generation_request_sha256: str | None = None,
) -> CampaignReflectionLearningRecord:
    catalog = ReflectionEvidenceCatalog.from_contrast_ids(contrast_ids)
    return CampaignReflectionLearningRecord(
        reflection_generation_request_sha256=(
            _sha(f"provider-free-reflection-request:{generation}:{call_id.value}")
            if reflection_generation_request_sha256 is None
            else reflection_generation_request_sha256
        ),
        reflection_call_id=call_id,
        source_generation=generation,
        source_stage_receipt_sha256=source_stage_receipt_sha256,
        origin_cutoff_event_index=(
            generation
            if origin_cutoff_event_index is None
            else origin_cutoff_event_index
        ),
        source_operator_invocation_ids=source_operator_invocation_ids,
        source_candidate_ids=source_candidate_ids,
        evidence_catalog=catalog,
        insight_contract=_reflection_contract(),
        insights=_reflection_drafts(generation, contrast_ids),
        finite_action_bindings=(),
        empirical_evidence=tuple(
            EmpiricalEvidenceSnapshot(
                contrast_id=contrast_id,
                fact_schema_id="provider_free_recombination_contrast",
                fact_schema_version=1,
                fact_schema_definition_sha256=_sha(
                    "provider-free-recombination-contrast-schema-v1"
                ),
                facts=_object(
                    {
                        "source_outcome_sha256": contrast_id,
                        "provider_calls": 0,
                    }
                ),
            )
            for contrast_id in contrast_ids
        ),
    )


class _ReflectionExecutor:
    def __init__(self) -> None:
        self.generations: list[int] = []
        self.records: list[FrozenJsonObject] = []
        self.inputs: list[CampaignIdentifiableReflectionInput] = []

    async def reflect(self, reflection_input):
        assert type(reflection_input) is CampaignIdentifiableReflectionInput
        self.inputs.append(reflection_input)
        query = reflection_input.query
        evidence = reflection_input.evidence
        generation = query.wave.source_generation
        self.generations.append(generation)
        contrasts = evidence.contrasts
        record = _reflection_learning_record(
            generation=generation,
            source_stage_receipt_sha256=query.source_stage_receipt_sha256,
            call_id=LLMCallId(f"call_runtime_reflection_g{generation:02d}"),
            source_operator_invocation_ids=tuple(
                sorted({value.operator_invocation_id for value in contrasts})
            ),
            source_candidate_ids=tuple(
                sorted(
                    {
                        candidate_id
                        for value in contrasts
                        for candidate_id in (
                            value.parent_candidate_id,
                            value.child_candidate_id,
                        )
                    }
                )
            ),
            contrast_ids=tuple(value.contrast_id for value in contrasts),
            origin_cutoff_event_index=(
                query.sealed_cutoff_event_index_inclusive
            ),
            reflection_generation_request_sha256=reflection_input.input_sha256,
        )
        result = CampaignReflectionLearningRecordCodec.encode(record)
        self.records.append(result)
        return result


def _standalone_learning_result() -> FrozenJsonObject:
    contrasts = tuple(
        sorted((_sha("standalone-contrast-a"), _sha("standalone-contrast-b")))
    )
    record = _reflection_learning_record(
        generation=2,
        source_stage_receipt_sha256=_sha("standalone-source-stage"),
        call_id=LLMCallId("call_standalone_reflection"),
        source_operator_invocation_ids=(
            OperatorInvocationId("operator_standalone_reflection"),
        ),
        source_candidate_ids=(CandidateId("candidate_standalone_reflection"),),
        contrast_ids=contrasts,
    )
    return CampaignReflectionLearningRecordCodec.encode(record)


def test_campaign_reflection_learning_record_round_trips_semantic_v3() -> None:
    result = _standalone_learning_result()
    decoded = CampaignReflectionLearningRecordCodec.decode(result)

    assert decoded.insight_contract.is_semantic_v3
    assert len(decoded.insights) == 2
    assert all(value.has_semantic_contract for value in decoded.insights)
    assert all(
        value.empirical_evidence
        for value in (decoded.lineage_for(draft) for draft in decoded.insights)
    )
    assert CampaignReflectionLearningRecordCodec.encode(decoded) == result


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda record: record.pop("empirical_evidence"),
            "canonical schema",
        ),
        (
            lambda record: record["insights"][0].__setitem__(
                "affected_paths", ["$.foreign_path"]
            ),
            "decision-path vocabulary",
        ),
        (
            lambda record: record.__setitem__("foreign_field", True),
            "canonical schema",
        ),
    ),
)
def test_campaign_reflection_learning_record_rejects_tampering(
    mutation,
    error: str,
) -> None:
    raw = thaw_json(_standalone_learning_result())
    record = raw["campaign_reflection_learning"]
    mutation(record)
    tampered = freeze_json(raw)
    assert type(tampered) is FrozenJsonObject

    with pytest.raises((TypeError, ValueError), match=error):
        CampaignReflectionLearningRecordCodec.decode(tampered)


def _provider_free_lane_endpoint(outcomes) -> float:
    """One predeclared fixture endpoint applied identically to every trial."""

    labels = tuple(
        value.candidate.label for value in outcomes if value.candidate is not None
    )
    assert labels
    if all("_p01" in value for value in labels):
        return 1.0
    if all("_p02" in value for value in labels):
        return 0.0
    raise AssertionError("one portfolio trial crossed parent-lane labels")


class _WaveFactory:
    def __init__(self, composition, learning_runtime, seed_card) -> None:
        self.composition = composition
        self.learning_runtime = learning_runtime
        self.seed_card = seed_card
        self.normal_consumed: list = []
        self.diagnostic_assignments: list[tuple[int, tuple]] = []

    def _source_card(self, entry, source_receipt_sha256, generation):
        payload = _object(
            {
                "claim": entry.draft.claim,
                "generation": generation,
            }
        )
        return portfolio_card_from_insight_entry(
            entry,
            card_key=(
                f"card.g{generation:02d}.{entry.reference.insight_id.value[-8:]}"
            ),
            prompt_payload=payload,
            evidence_sha256=source_receipt_sha256,
            source_receipt_sha256=source_receipt_sha256,
            assigned_score=0.0,
        )

    def _request(self, context, cards, source_registry=None):
        return PortfolioSelectionRequest(
            call_id=self.composition.id_factory.new_llm_call_id(),
            operation="select_portfolio",
            instruction="Select two sealed path-disjoint options.",
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=cards,
            portfolio_size=context.stage_request.step.offspring_per_parent,
            required_metric_ids=("total_levels", "total_lut_count"),
            min_distinct_families=None,
            require_supporting_cards=False,
            max_output_tokens=1,
            temperature=None,
            source_registry=source_registry,
        )

    def _diagnostic_wave(self, context, exposure, selected_reference):
        entries = self.composition.memory.entries_for(exposure.references)
        entry = next(
            value for value in entries if value.reference == selected_reference
        )
        card = self._source_card(
            entry,
            exposure.receipt_sha256,
            context.stage_request.step.generation,
        )
        registry = admit_portfolio_card_sources((entry,), (card,))
        request = self._request(context, (card,), registry)
        eligible = exposure.references
        decision = InsightSelectionDecision(
            context_hash=typed_json_sha256(context.evidence_context),
            eligible=eligible,
            selected=(selected_reference,),
            exploitation_subset=(eligible[0],),
            score_snapshot=tuple((value, 0.0) for value in eligible),
            subset_size=1,
            exploration_probability=Fraction(1, 1),
            mode=InsightSelectionMode.EXPLORE_UNIFORM,
            selected_subset_probability=Fraction(1, len(eligible)),
        )
        snapshot = CausalSearchScorePolicy(
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        ).genesis(
            exact_context_hash=decision.context_hash,
            estimand_stratum_hash=_sha("provider-free-g5-estimand"),
            priors=dict(decision.score_snapshot),
        )
        credit_unit_id = self.composition.id_factory.new_operator_invocation_id()
        assignment = ResolvedInsightAssignment.resolve(
            credit_unit_id=credit_unit_id,
            snapshot=snapshot,
            expected_snapshot_sha256=snapshot.snapshot_sha256,
            block_id=(
                f"runtime_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            arm=MemoryAssignmentArm.DIAGNOSTIC,
            selection_decision=decision,
            prompt_shape_sha256=request.card_snapshot_sha256,
        )
        # The provider-free fixture uses a predeclared lane endpoint solely to
        # make the delayed causal promotion deterministic; it is not a result.
        aggregation = PortfolioRewardAggregationBinding(
            aggregate=_provider_free_lane_endpoint,
            aggregation_id="provider_free_lane_endpoint",
            aggregation_version=1,
            definition_sha256=_sha("provider-free-lane-endpoint-v1"),
        )
        self.diagnostic_assignments.append(
            (context.stage_request.step.generation, decision.selected)
        )
        return PortfolioVariationWaveRequest(
            selection_request=request,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"closed_loop_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase="provider_free_closed_loop",
            memory_credit=PortfolioMemoryCreditPlan(
                decision=decision,
                credit_unit_id=credit_unit_id,
                aggregation=aggregation,
                card_snapshot_sha256=request.card_snapshot_sha256,
                score_snapshot=snapshot,
                assignment=assignment,
                card_source_registry_sha256=registry.registry_sha256,
                quarantine_admission=exposure.memory_admission,
                context_projection=(
                    PortfolioMemoryContextProjectionBinding.exact_identity(
                        decision.context_hash
                    )
                ),
            ),
        )

    def build(self, context: CampaignPortfolioWaveContext):
        generation = context.stage_request.step.generation
        if generation == 1:
            request = self._request(context, (self.seed_card,))
            return PortfolioVariationWaveRequest(
                selection_request=request,
                parent=context.parent,
                generation=generation,
                label_prefix=f"closed_loop_g01_p{context.parent_slot + 1:02d}",
                phase="provider_free_closed_loop",
            )
        exposures = self.learning_runtime.diagnostic_exposures(
            context.stage_request.test_eligible_reflection_receipt_sha256s
        )
        latest = max(exposures, key=lambda value: value.barrier_generation)
        if generation == 3:
            selected = latest.references[context.parent_slot]
            return self._diagnostic_wave(
                context,
                latest,
                selected,
            )
        if context.parent_slot == 0:
            normal = self.learning_runtime.normal_references(
                operator_kind="typed_mutation",
                editable_paths=("$.sequence",),
                consumer_scope=ReflectionConsumerScope.MUTATION_SELECTION,
                factor_capabilities=("sequence_rewrite",),
            )
            promoted = tuple(
                value.reference
                for value in self.composition.memory.entries_for(normal)
                if value.lifecycle_state is InsightLifecycleState.PROMOTED
            )
            if promoted:
                assert len(promoted) == 1
                entry = self.composition.memory.entries_for(promoted)[0]
                prior = min(exposures, key=lambda value: value.barrier_generation)
                card = self._source_card(entry, prior.receipt_sha256, generation)
                registry = admit_portfolio_card_sources((entry,), (card,))
                request = self._request(context, (card,), registry)
                self.normal_consumed.extend(promoted)
                return PortfolioVariationWaveRequest(
                    selection_request=request,
                    parent=context.parent,
                    generation=generation,
                    label_prefix="closed_loop_g05_normal",
                    phase="provider_free_closed_loop",
                )
        # One diagnostic generation cannot pass the robust block-sign policy.
        # Continue testing the newest quarantine block instead of pretending the
        # first block was eligible for normal retrieval.
        return self._diagnostic_wave(
            context,
            latest,
            latest.references[context.parent_slot],
        )


class _ArchiveUtility:
    utility_id = "provider_free_g5_archive_utility"
    utility_version = 1
    definition_sha256 = _sha("provider-free-g5-archive-utility")

    def freeze(self, *, benchmark, generation, archive):
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object({"cutoff": generation}),
        )


class _PreparationRuntime:
    def prepare(self, request):
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="provider_free_g5_runtime",
            runtime_version=1,
            definition_sha256=_sha("provider-free-g5-runtime"),
            accepted=True,
            evidence=_object({"provider_calls": 0}),
        )


class _PreparationJournal:
    def append(self, record):
        assert type(record) is FrozenJsonObject


class _ExecutionJournal:
    def __init__(self) -> None:
        self.events: list[CampaignExecutionEvent] = []

    async def append(self, event):
        self.events.append(event)
        return CampaignJournalAck(event.event_sha256, True)


def _binding(name: str, implementation: object) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"campaign-policy:{name}"),
    )


def test_real_closed_loop_runtime_reflects_tests_deprecates_and_retains_by_g5():
    evaluator = _Evaluator()
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(
            AbcEvaluatorSettings.current_circuit_panel(circuit_names=("log2",)),
            evaluator=evaluator,
        ),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    ids = DeterministicIdFactory("closed_loop_runtime_g5")
    memory = InsightMemoryBank(id_factory=ids)
    seed_entry = memory.extend(
        (
            InsightDraft(
                claim="Use sealed finite mutations during the bootstrap stage.",
                trigger="A finite mutation catalog is available.",
                mechanism="The catalog provides provider-free candidate diversity.",
                affected_paths=("$.sequence",),
                evidence_summary="Provider-free bootstrap prior.",
                confidence=0.5,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )[0]
    seed_card = PortfolioCard(
        card_key="card.bootstrap",
        reference=seed_entry.reference,
        content_sha256=seed_entry.draft.content_sha256,
        evidence_sha256=_sha("bootstrap-evidence"),
        prompt_payload=_object({"prior": "sealed_finite_mutations"}),
        assigned_score=0.0,
    )
    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=DeterministicRandomFeasiblePortfolioPolicy(seed=20260717),
        seed=20260717,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=2,
        max_output_tokens=1,
        temperature=None,
    )
    coordinator = ClosedLoopCampaignLearning(
        memory=memory,
        promotion_policy=CampaignInsightPromotionPolicy(
            minimum_treated_trials=1,
            minimum_control_trials=1,
            minimum_effective_support=1.0,
            minimum_effect=0.0,
        ),
    )
    scope = HypothesisAuditScope(
        workload_instance_sha256s=(_sha("runtime-workload"),),
        evaluator_contract_sha256=_sha("runtime-evaluator"),
        metric_adjudicator_definition_sha256=_sha("runtime-adjudicator"),
        campaign_sha256s=(_sha("runtime-campaign"),),
    )
    evidence_registry = CampaignEvidenceRegistry()
    generation_auditor = TransactionalPortfolioGenerationAuditor(
        evidence_registry=evidence_registry,
        campaign_sha256=_sha("runtime-campaign"),
        workload_instance_sha256=_sha("runtime-workload"),
        evaluator_contract_sha256=_sha("runtime-evaluator"),
        metric_projector=ObjectiveDeltaMetricEffectProjector(
            _sha("runtime-adjudicator")
        ),
        action_semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
        hypothesis_matcher=PortableFiniteActionHypothesisMatcher(),
    )
    learning_runtime = ClosedLoopCampaignLearningRuntime(
        learning=coordinator,
        reflection_projection=StructuredCampaignReflectionLearningProjector(
            semantic_compiler=PortableFiniteActionInsightSemanticCompiler(),
            scope=scope,
            applicable_operator_kinds=("typed_mutation",),
            diagnostic_operator_kind="typed_mutation",
            diagnostic_editable_paths=("$.sequence",),
            initial_score=0.0,
            minimum_support_clusters=1,
            minimum_support_instances=1,
        ),
        generation_auditor=generation_auditor,
    )
    evidence = _Evidence()
    config = AgenticCampaignWorkloadConfig(
        workload_id="closed-loop-runtime-g5",
        workload_version=1,
        definition_sha256=_sha("closed-loop-runtime-g5"),
        benchmark=benchmark,
        seeds=(
            CampaignSeed(
                "seed_default",
                _object({"sequence": list(DEFAULT_ACTION_SEQUENCE)}),
            ),
            CampaignSeed(
                "seed_parent_c",
                _object({"sequence": list(PARENT_C_SEQUENCE)}),
            ),
        ),
        finite_catalog_id=FINITE_CATALOG_ID,
        evaluator_concurrency_cap=2,
        evaluator_preflight_receipt=_object({"qualified": True}),
        resource_lease_receipt=_object({"lease": "provider_free_cpu"}),
        evidence=AgenticCampaignEvidenceProjections(
            projection_id="closed_loop_runtime_evidence",
            projection_version=1,
            definition_sha256=_sha("closed-loop-runtime-evidence"),
            initialize_memory=evidence.initialize_memory,
            context=evidence.context,
            cards=evidence.cards,
        ),
    )
    parent_selector = ArchiveReservoirCampaignParentSelector(reservoir_limit=8)
    reflection = _ReflectionExecutor()
    wave_factory = _WaveFactory(composition, learning_runtime, seed_card)
    policies = CampaignPolicies(
        cadence=AlternatingPortfolioRecombinationCadence(),
        parent_selection=_binding("archive_reservoir", parent_selector),
        memory_assignment=_binding("closed_loop_memory", learning_runtime),
        portfolio_selection=_binding(
            "provider_free_random",
            composition.portfolio.selector,
        ),
        recombination=_binding("disjoint_patch_union", object()),
        reflection=_binding("provider_free_reflection", reflection),
        archive_utility=_ArchiveUtility(),
    )
    protocol = CampaignProtocol(
        protocol_id="provider_free_closed_loop_g5",
        protocol_version=1,
        definition_sha256=_sha("provider-free-closed-loop-g5"),
        outer_seed=20260717,
        generation_count=5,
        required_seed_count=2,
        parents_per_portfolio_generation=2,
        portfolio_width=2,
        recombinations_per_parent=1,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=1,
    )
    workload_ports = config.build_ports()
    prepared = EvolutionCampaign(
        protocol=protocol,
        workload=workload_ports,
        policies=policies,
        runtime=_PreparationRuntime(),
        budget=OptimizerBudget(
            max_unique_evaluations=18,
            max_logical_llm_calls=8,
            max_generations=5,
        ),
        concurrency=CampaignConcurrency(
            evaluator_concurrency=2,
            agent_concurrency=2,
            agent_queue_capacity=4,
        ),
        journals=(_PreparationJournal(),),
    ).prepare()
    runtime = AgenticPortfolioCampaignRuntime(
        prepared=prepared,
        workload_config=config,
        workload_ports=workload_ports,
        composition=composition,
        parent_selector=parent_selector,
        wave_factory=wave_factory,
        task_sha256=_sha("provider-free-closed-loop-task"),
        learning_lifecycle=learning_runtime,
        identifiable_reflection_executor=reflection,
        identifiable_reflection_evidence_source=(
            CommittedRegistryIdentifiableReflectionEvidenceSource(
                registry=evidence_registry,
                campaign_sha256=_sha("runtime-campaign"),
                workload_instance_sha256=_sha("runtime-workload"),
                evaluator_contract_sha256=_sha("runtime-evaluator"),
            )
        ),
    )
    result = asyncio.run(
        EvolutionCampaignScheduler(
            prepared=prepared,
            policies=policies,
            stages=runtime,
            reflections=runtime,
            lifecycle=runtime,
            journal=_ExecutionJournal(),
        ).run()
    )

    assert result.counters.generations_completed == 5
    assert reflection.generations == [2, 4]
    assert tuple(
        (
            value.query.prior_cutoff_event_index_exclusive,
            value.query.sealed_cutoff_event_index_inclusive,
        )
        for value in reflection.inputs
    ) == ((0, 1), (1, 3))
    assert all(
        not hasattr(value, "source_stage")
        and not hasattr(value, "recombination_results")
        for value in reflection.inputs
    )
    expected_action_compiler = FinitePortfolioActionSemanticsCompiler()
    assert all(
        (
            contrast.action_semantics_compiler_id,
            contrast.action_semantics_compiler_version,
            contrast.action_semantics_definition_sha256,
        )
        == (
            expected_action_compiler.compiler_id,
            expected_action_compiler.compiler_version,
            expected_action_compiler.definition_sha256,
        )
        for reflection_input in reflection.inputs
        for contrast in reflection_input.evidence.contrasts
    )
    assert tuple(
        thaw_json(receipt.quarantined_result)[
            CAMPAIGN_IDENTIFIABLE_REFLECTION_BINDING_KEY
        ]["input_sha256"]
        for receipt in result.reflection_receipts
    ) == tuple(value.input_sha256 for value in reflection.inputs)
    decoded_reflections = tuple(
        CampaignReflectionLearningRecordCodec.decode(value)
        for value in reflection.records
    )
    assert tuple(value.source_generation for value in decoded_reflections) == (2, 4)
    assert tuple(
        value.origin_cutoff_event_index for value in decoded_reflections
    ) == (1, 3)
    assert all(value.insight_contract.is_semantic_v3 for value in decoded_reflections)
    assert all(value.source_operator_invocation_ids for value in decoded_reflections)
    assert all(value.source_candidate_ids for value in decoded_reflections)
    assert all(
        value.evidence_catalog.contrast_ids
        == tuple(item.contrast_id for item in value.empirical_evidence)
        for value in decoded_reflections
    )
    assert len(result.test_admission_receipts) == 2
    assert len(memory.trials) == 4
    assert len(evidence_registry.observations) == 12
    assert {value.event_index for value in evidence_registry.observations} == {1, 3, 5}
    assert all(
        value.provenance.value == "direct_mutation"
        for value in evidence_registry.observations
    )
    g1_record = thaw_json(result.stage_receipts[0].result)
    g1_audit = g1_record["closed_loop_learning"]["evidence"][
        "generation_audit_preparation"
    ]
    assert g1_audit["status"] == ("evidence_append_prepared_no_diagnostic_assignment")
    assert g1_audit["projection"] is None
    g2_entries = tuple(
        value for value in memory.entries if "Generation 2" in value.draft.claim
    )
    assert all(
        value.lifecycle_state
        in {InsightLifecycleState.QUARANTINED, InsightLifecycleState.DEPRECATED}
        for value in g2_entries
    )
    assert wave_factory.normal_consumed == []
    g4_entries = tuple(
        value for value in memory.entries if "Generation 4" in value.draft.claim
    )
    assert all(
        value.lifecycle_state
        in {InsightLifecycleState.QUARANTINED, InsightLifecycleState.DEPRECATED}
        for value in g4_entries
    )
    assert wave_factory.diagnostic_assignments == [
        (3, (g2_entries[0].reference,)),
        (3, (g2_entries[1].reference,)),
        (5, (g4_entries[0].reference,)),
        (5, (g4_entries[1].reference,)),
    ]
    g5_record = thaw_json(result.stage_receipts[4].result)
    assert g5_record["closed_loop_learning"]["evidence"]["status"] == (
        "prepared_closed_loop_learning_real_gate"
    )
    audit_record = g5_record["closed_loop_learning"]["evidence"][
        "generation_audit_preparation"
    ]
    assert audit_record["status"] == "evidence_append_and_real_gate_prepared"
    memory_attribution = audit_record["memory_attribution_audit"]
    assert memory_attribution["causal_card_effect_identified"] is False
    assert memory_attribution["causal_action_effect_identified"] is False
    assert memory_attribution["online_score_update_allowed"] is False
    assert len(memory_attribution["candidate_contributions"]) == 4
    assert memory_attribution["card_performance"] == []
    projection = audit_record["projection"]
    assert (
        projection["registry_snapshot_sha256"]
        == (audit_record["prospective_registry_snapshot_sha256"])
    )
    assert all(
        binding["context_authority"] == "pre_outcome_resolved_diagnostic_assignment"
        for binding in projection["context_bindings"]
    )
    assert all(
        audit["receipt"]["audit_policy"]["policy_id"] == GLOBAL_FALSIFICATION_POLICY_ID
        and audit["receipt"]["audit_policy"]["policy_version"]
        == GLOBAL_FALSIFICATION_POLICY_VERSION
        for audit in projection["audits"]
    )
    assert len(projection["audits"]) == 2
    assert all(
        audit["request"]["origin_cutoff_event_index"] == 3
        and audit["request"]["audit_cutoff_event_index"] == 5
        and audit["request"]["registry_snapshot_sha256"]
        == projection["registry_snapshot_sha256"]
        and len(audit["receipt"]["decisions"]) == 12
        for audit in projection["audits"]
    )
    assert g5_record["memory_credit_batch"]["credit_count"] == 2
    assert evaluator.calls == result.counters.unique_evaluations

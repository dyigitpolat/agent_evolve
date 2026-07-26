#!/usr/bin/env python3
"""Provider-free multi-generation proof for the BOiLS campaign bridge.

This executable is deliberately not an efficacy experiment.  It replaces only
provider transport with deterministic local policies while retaining the real,
hash-pinned Berkeley ABC evaluator.  It exercises the production-generic
campaign path end to end:

* two BOiLS seeds and the real 200-option parent-local catalog;
* alternating ranked portfolio mutation and disjoint recombination;
* canonical semantic-v3 reflection records;
* quarantine admission and randomized diagnostic card assignment;
* transactional memory credit, authenticated finite-action evidence, and the
  real global falsification gate.

The composition points used here are the same ports a paid selector,
reflection model, and selector replace in a paid campaign.  This executable
does not make an OpenRouter call and is not an optimizer-efficacy experiment.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.agentic import (  # noqa: E402
    AgenticBenchmark,
    DeterministicIdFactory,
    InsightDraft,
    InsightMemoryBank,
    PortfolioCard,
    PortfolioSelectionRequest,
    TypedConfigurationPhenotypeIdentityPolicy,
    compose_portfolio_evolution,
)
from agent_evolve.application.budgeted_optimizer import OptimizerBudget  # noqa: E402
from agent_evolve.application.campaign_evidence_registry import (  # noqa: E402
    CampaignEvidenceRegistry,
)
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignExecutionEvent,
    CampaignExecutionResult,
    CampaignJournalAck,
    EvolutionCampaignScheduler,
)
from agent_evolve.application.campaign_generation_audit import (  # noqa: E402
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.campaign_learning import (  # noqa: E402
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.campaign_learning_runtime import (  # noqa: E402
    CampaignReflectionLearningRecord,
    CampaignReflectionLearningRecordCodec,
    ClosedLoopCampaignLearningRuntime,
    StructuredCampaignReflectionLearningProjector,
)
from agent_evolve.application.evolution_campaign import (  # noqa: E402
    AlternatingPortfolioRecombinationCadence,
    ArchiveUtilitySnapshot,
    CampaignAgentRuntimeReceipt,
    CampaignConcurrency,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    EvolutionCampaign,
)
from agent_evolve.application.finite_action_hypothesis_semantics import (  # noqa: E402
    PortableFiniteActionHypothesisMatcher,
    PortableFiniteActionInsightSemanticCompiler,
)
from agent_evolve.application.insight_memory import (  # noqa: E402
    EmpiricalEvidenceSnapshot,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    AgenticPortfolioCampaignRuntime,
    ArchiveReservoirCampaignParentSelector,
    CampaignPortfolioWaveContext,
)
from agent_evolve.application.portfolio_evolution import (  # noqa: E402
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryCreditPlan,
    PortfolioRewardAggregationBinding,
    PortfolioVariationWaveRequest,
)
from agent_evolve.application.portfolio_hypothesis_observations import (  # noqa: E402
    FinitePortfolioActionSemanticsCompiler,
    ObjectiveDeltaMetricEffectProjector,
)
from agent_evolve.application.portfolio_projection import (  # noqa: E402
    admit_portfolio_card_sources,
    portfolio_card_from_insight_entry,
)
from agent_evolve.domain.ids import LLMCallId  # noqa: E402
from agent_evolve.domain.insight import InsightRef  # noqa: E402
from agent_evolve.domain.typed_json import (  # noqa: E402
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.policies.memory.global_falsification import (  # noqa: E402
    HypothesisAuditScope,
)
from agent_evolve.policies.memory.balanced_subset_blocks import (  # noqa: E402
    BalancedSubsetBlockPlan,
    BalancedSubsetBlockPlanner,
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.memory.staged_causal import (  # noqa: E402
    CausalSearchScorePolicy,
    MemoryAssignmentArm,
    ResolvedInsightAssignment,
)
from agent_evolve.policies.selection.random_portfolio import (  # noqa: E402
    DeterministicRandomFeasiblePortfolioPolicy,
)
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionEvidenceCatalog,
    ReflectionInsightContract,
    ReflectionInsightKind,
)
from examples.benchmarks.boils_abc.campaign_workload import (  # noqa: E402
    WORKLOAD_DEFINITION_SHA256,
    compose_boils_campaign_workload,
)
from examples.benchmarks.boils_abc.detailed_evaluation import (  # noqa: E402
    boils_evaluator_identity,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (  # noqa: E402
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import (  # noqa: E402
    BoilsAbcProblem,
)
from examples.benchmarks.boils_abc.variation_catalog import (  # noqa: E402
    ACTION_FAMILIES,
)


OUTER_SEED = 20260717
OBJECTIVE_IDS = ("total_levels", "total_lut_count")
REFLECTION_PATHS = tuple(sorted(f"$.sequence[{index}]" for index in range(20)))
REFLECTION_FAMILIES = tuple(sorted(set(ACTION_FAMILIES.values())))
_METRIC_ADJUDICATOR_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-objective-delta-adjudicator:v1"
).hexdigest()
_PORTFOLIO_ENDPOINT_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-provider-free-negative-log-objective-endpoint:v1"
).hexdigest()
_REFLECTION_FACT_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-provider-free-recombination-contrast:v1"
).hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("BOiLS provider-free record is not an object")
    return frozen


class _TimedBoilsEvaluator:
    """Record wall timing while delegating to the real pinned ABC evaluator."""

    def __init__(self, settings: AbcEvaluatorSettings) -> None:
        self.delegate = BoilsAbcEvaluator(settings)
        self.elapsed_s: list[float] = []
        self.reported_elapsed_s: list[float] = []

    @property
    def calls(self) -> int:
        return len(self.elapsed_s)

    def evaluate(self, config: object) -> BoilsEvaluation:
        started = time.perf_counter()
        result = self.delegate.evaluate(config)
        self.elapsed_s.append(time.perf_counter() - started)
        self.reported_elapsed_s.append(result.elapsed_s)
        return result


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - materialized only.
        raise AssertionError(
            f"materialized BOiLS campaign invoked propose: {request}"
        )

    async def reflect(self, request):  # pragma: no cover - external executor.
        raise AssertionError(f"campaign invoked engine reflection: {request}")


def _reflection_contract() -> ReflectionInsightContract:
    return ReflectionInsightContract(
        required_metric_ids=OBJECTIVE_IDS,
        allowed_option_families=REFLECTION_FAMILIES,
        allowed_decision_paths=REFLECTION_PATHS,
        allowed_insight_kinds=(ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,),
        allowed_consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        allowed_comparison_anchor_kinds=(MetricComparisonAnchorKind.CURRENT_PARENT,),
        allowed_factor_capabilities=REFLECTION_FAMILIES,
    )


def _prediction(metric_id: str, direction: MetricEffectDirection):
    return MetricEffectPrediction(
        metric_id=metric_id,
        direction=direction,
        comparison_anchor=MetricComparisonAnchor(
            MetricComparisonAnchorKind.CURRENT_PARENT
        ),
    )


def _reflection_drafts(
    generation: int,
    contrast_ids: tuple[str, ...],
) -> tuple[InsightDraft, ...]:
    if len(contrast_ids) < 2:
        raise ValueError("BOiLS reflection proof requires two source contrasts")
    return (
        InsightDraft(
            claim=(
                f"Generation {generation}: an early AIG-balance replacement can "
                "reduce mapped depth and LUT demand."
            ),
            trigger="An early parent-local AIG-balance option is available.",
            mechanism=(
                "Early balancing can shorten critical paths before later rewriting "
                "and mapping under the frozen ABC protocol."
            ),
            affected_paths=("$.sequence[0]",),
            evidence_summary="One authenticated recombination contrast motivated testing.",
            confidence=0.5,
            evidence_contrast_ids=(contrast_ids[0],),
            effect_predictions=(
                _prediction("total_levels", MetricEffectDirection.DECREASE),
                _prediction("total_lut_count", MetricEffectDirection.DECREASE),
            ),
            recommended_option_families=("aig_balance",),
            action_template="Apply one sealed early AIG-balance finite action.",
            falsification_condition=(
                "A held-out exact early AIG-balance action violates a predicted "
                "metric direction."
            ),
            insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
            factor_capabilities=("aig_balance",),
        ),
        InsightDraft(
            claim=(
                f"Generation {generation}: an early AIG-refactor replacement can "
                "reduce mapped LUT demand and depth."
            ),
            trigger="An early parent-local AIG-refactor option is available.",
            mechanism=(
                "Early algebraic refactoring can expose a smaller logic structure "
                "to later rewriting and mapping under the frozen ABC protocol."
            ),
            affected_paths=("$.sequence[1]",),
            evidence_summary="One authenticated recombination contrast motivated testing.",
            confidence=0.5,
            evidence_contrast_ids=(contrast_ids[1],),
            effect_predictions=(
                _prediction("total_levels", MetricEffectDirection.DECREASE),
                _prediction("total_lut_count", MetricEffectDirection.DECREASE),
            ),
            recommended_option_families=("aig_refactor",),
            action_template="Apply one sealed early AIG-refactor finite action.",
            falsification_condition=(
                "A held-out exact early AIG-refactor action violates a predicted metric "
                "direction."
            ),
            insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
            factor_capabilities=("aig_refactor",),
        ),
    )


class _ReflectionExecutor:
    """Engine-authored canonical envelope replacing only the provider call."""

    def __init__(self) -> None:
        self.generations: list[int] = []
        self.records: list[FrozenJsonObject] = []

    async def reflect(self, request, source_results):
        generation = request.wave.source_generation
        members = tuple(
            member for result in source_results for member in result.receipt.members
        )
        contrast_ids = tuple(sorted(member.outcome_sha256 for member in members))
        catalog = ReflectionEvidenceCatalog.from_contrast_ids(contrast_ids)
        record = CampaignReflectionLearningRecord(
            reflection_generation_request_sha256=_sha(
                f"boils-provider-free-reflection:{request.request_sha256}"
            ),
            reflection_call_id=LLMCallId(
                f"call_boils_provider_free_reflection_g{generation:02d}"
            ),
            source_generation=generation,
            source_stage_receipt_sha256=request.source_stage.receipt_sha256,
            origin_cutoff_event_index=generation,
            source_operator_invocation_ids=tuple(
                sorted(member.operator_invocation_id for member in members)
            ),
            source_candidate_ids=tuple(
                sorted(member.target_candidate_id for member in members)
            ),
            evidence_catalog=catalog,
            insight_contract=_reflection_contract(),
            insights=_reflection_drafts(generation, contrast_ids),
            finite_action_bindings=(),
            empirical_evidence=tuple(
                EmpiricalEvidenceSnapshot(
                    contrast_id=contrast_id,
                    fact_schema_id="boils_recombination_contrast",
                    fact_schema_version=1,
                    fact_schema_definition_sha256=_REFLECTION_FACT_SCHEMA_SHA256,
                    facts=_object(
                        {
                            "source_outcome_sha256": contrast_id,
                            "provider_calls": 0,
                            "abc_evaluation_source": "authenticated_engine_receipt",
                        }
                    ),
                )
                for contrast_id in contrast_ids
            ),
        )
        encoded = CampaignReflectionLearningRecordCodec.encode(record)
        self.generations.append(generation)
        self.records.append(encoded)
        return encoded


def _portfolio_quality(outcomes) -> float:
    """Predeclared provider-free endpoint over actual evaluated objectives."""

    candidates = tuple(
        outcome.candidate for outcome in outcomes if outcome.candidate is not None
    )
    if not candidates:
        raise ValueError("BOiLS portfolio endpoint requires a valid candidate")
    utilities = tuple(
        -sum(math.log(value) for value in candidate.objective_map.values())
        for candidate in candidates
    )
    return float(max(utilities))


class _WaveFactory:
    def __init__(self, composition, learning_runtime, seed_card) -> None:
        self.composition = composition
        self.learning_runtime = learning_runtime
        self.seed_card = seed_card
        self.diagnostic_assignments: list[tuple[int, int, tuple[InsightRef, ...]]] = []
        self.assignment_plans: dict[tuple[int, str], BalancedSubsetBlockPlan] = {}

    def _request(self, context, cards, source_registry=None):
        return PortfolioSelectionRequest(
            call_id=self.composition.id_factory.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Select a diverse ranked portfolio from the sealed BOiLS finite "
                "options using only the authenticated context and cards."
            ),
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=cards,
            portfolio_size=context.stage_request.step.offspring_per_parent,
            required_metric_ids=OBJECTIVE_IDS,
            min_distinct_families=None,
            require_supporting_cards=False,
            temperature=None,
            source_registry=source_registry,
        )

    def _assignment_plan(self, context, exposure, projection):
        generation = context.stage_request.step.generation
        key = (generation, exposure.receipt_sha256)
        existing = self.assignment_plans.get(key)
        if existing is not None:
            if existing.snapshot.exact_context_hash != (
                projection.estimand_context_sha256
            ):
                raise RuntimeError("BOiLS diagnostic estimand changed across lanes")
            return existing
        snapshot = CausalSearchScorePolicy(
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        ).genesis(
            exact_context_hash=projection.estimand_context_sha256,
            estimand_stratum_hash=_sha("boils-provider-free-memory-estimand"),
            priors={reference: 0.0 for reference in exposure.references},
        )
        units = tuple(
            StableMemoryAssignmentUnit(
                unit_key=f"boils_g{generation:02d}_p{slot + 1:02d}",
                generation=generation,
                lane_id=f"parent_{slot + 1:02d}",
            )
            for slot in range(2)
        )
        permutation_rank = (
            int(
                _sha(
                    f"{OUTER_SEED}:{generation}:{exposure.receipt_sha256}:"
                    "balanced-subset-permutation"
                ),
                16,
            )
            % 2
        )
        plan = BalancedSubsetBlockPlanner().plan(
            snapshot=snapshot,
            ordered_units=units,
            subset_size=1,
            full_block_permutation_ranks=(permutation_rank,),
        )
        self.assignment_plans[key] = plan
        return plan

    def _diagnostic_wave(self, context, exposure):
        projection = PortfolioMemoryContextProjectionBinding.from_selector_context(
            context.evidence_context
        )
        plan = self._assignment_plan(context, exposure, projection)
        assignment_slot = plan.assignment_for(
            context.stage_request.step.generation,
            f"parent_{context.parent_slot + 1:02d}",
        )
        decision = assignment_slot.decision
        selected_reference = decision.selected[0]
        entry = next(
            value
            for value in self.composition.memory.entries_for(exposure.references)
            if value.reference == selected_reference
        )
        payload = _object(
            {
                "claim": entry.draft.claim,
                "source_generation": exposure.barrier_generation - 1,
                "test_generation": context.stage_request.step.generation,
            }
        )
        card = portfolio_card_from_insight_entry(
            entry,
            card_key=(
                "card.boils."
                f"{context.stage_request.step.generation:02d}."
                f"{entry.reference.insight_id.value[-8:]}"
            ),
            prompt_payload=payload,
            evidence_sha256=exposure.receipt_sha256,
            source_receipt_sha256=exposure.receipt_sha256,
            assigned_score=0.0,
        )
        registry = admit_portfolio_card_sources((entry,), (card,))
        request = self._request(context, (card,), registry)
        snapshot = plan.snapshot
        credit_unit_id = self.composition.id_factory.new_operator_invocation_id()
        assignment = ResolvedInsightAssignment.resolve(
            credit_unit_id=credit_unit_id,
            snapshot=snapshot,
            expected_snapshot_sha256=snapshot.snapshot_sha256,
            block_id=f"boils_g{context.stage_request.step.generation:02d}",
            arm=MemoryAssignmentArm.DIAGNOSTIC,
            selection_decision=decision,
            prompt_shape_sha256=request.card_snapshot_sha256,
        )
        self.diagnostic_assignments.append(
            (
                context.stage_request.step.generation,
                context.parent_slot,
                decision.selected,
            )
        )
        return PortfolioVariationWaveRequest(
            selection_request=request,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"boils_closed_loop_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase="boils_provider_free_closed_loop",
            memory_credit=PortfolioMemoryCreditPlan(
                decision=decision,
                credit_unit_id=credit_unit_id,
                aggregation=PortfolioRewardAggregationBinding(
                    aggregate=_portfolio_quality,
                    aggregation_id="boils_provider_free_quality",
                    aggregation_version=1,
                    definition_sha256=_PORTFOLIO_ENDPOINT_SHA256,
                ),
                card_snapshot_sha256=request.card_snapshot_sha256,
                score_snapshot=snapshot,
                assignment=assignment,
                card_source_registry_sha256=registry.registry_sha256,
                quarantine_admission=exposure.memory_admission,
                context_projection=projection,
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
                label_prefix=f"boils_closed_loop_g01_p{context.parent_slot + 1:02d}",
                phase="boils_provider_free_closed_loop",
            )
        exposures = self.learning_runtime.diagnostic_exposures(
            context.stage_request.test_eligible_reflection_receipt_sha256s
        )
        exposure = max(exposures, key=lambda value: value.barrier_generation)
        return self._diagnostic_wave(context, exposure)


class _ArchiveUtility:
    utility_id = "boils_provider_free_archive_trace"
    utility_version = 1
    definition_sha256 = _sha("boils-provider-free-archive-trace-v1")

    def freeze(self, *, benchmark, generation, archive):
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object(
                {"generation": generation, "role": "provider_free_trace_only"}
            ),
        )


class _PreparationRuntime:
    def prepare(self, request):
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="boils_provider_free_runtime",
            runtime_version=1,
            definition_sha256=_sha("boils-provider-free-runtime-v1"),
            accepted=True,
            evidence=_object({"provider_calls": 0, "real_abc_evaluator": True}),
        )


class _PreparationJournal:
    def append(self, record):
        if type(record) is not FrozenJsonObject:
            raise TypeError("campaign preparation journal requires a frozen record")


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
        definition_sha256=_sha(f"boils-campaign-policy:{name}"),
    )


@dataclass(slots=True)
class ProviderFreeBoilsCampaignRun:
    execution: CampaignExecutionResult
    evaluator: _TimedBoilsEvaluator
    memory: InsightMemoryBank
    evidence_registry: CampaignEvidenceRegistry
    reflection_executor: _ReflectionExecutor
    wave_factory: _WaveFactory
    journal: _ExecutionJournal
    wall_time_s: float
    cpu_affinity_sets: tuple[tuple[int, ...], ...]
    final_front: tuple[object, ...]

    def summary(self) -> dict[str, object]:
        decoded = tuple(
            CampaignReflectionLearningRecordCodec.decode(value)
            for value in self.reflection_executor.records
        )
        elapsed = tuple(self.evaluator.elapsed_s)
        stage_candidate_counts = tuple(
            receipt.candidate_occurrence_count
            for receipt in self.execution.stage_receipts
        )
        return {
            "status": self.execution.finalization_receipt.status.value,
            "generations_completed": self.execution.counters.generations_completed,
            "candidate_occurrences": self.execution.counters.candidate_occurrences,
            "unique_evaluations": self.execution.counters.unique_evaluations,
            "evaluator_calls": self.evaluator.calls,
            "stage_candidate_counts": list(stage_candidate_counts),
            "mutation_candidates": sum(stage_candidate_counts[0::2]),
            "recombination_candidates": sum(stage_candidate_counts[1::2]),
            "reflection_generations": list(self.reflection_executor.generations),
            "canonical_reflection_records": len(decoded),
            "memory_entries": len(self.memory.entries),
            "memory_trials": len(self.memory.trials),
            "authenticated_action_observations": len(
                self.evidence_registry.observations
            ),
            "diagnostic_assignments": [
                {
                    "generation": generation,
                    "parent_slot": parent_slot,
                    "references": [
                        {
                            "insight_id": reference.insight_id.value,
                            "version": reference.version,
                        }
                        for reference in references
                    ],
                }
                for generation, parent_slot, references in (
                    self.wave_factory.diagnostic_assignments
                )
            ],
            "balanced_assignment_plan_receipts": [
                plan.receipt_sha256
                for _, plan in sorted(self.wave_factory.assignment_plans.items())
            ],
            "evaluator_timing_s": {
                "count": len(elapsed),
                "minimum": min(elapsed),
                "median": statistics.median(elapsed),
                "mean": statistics.fmean(elapsed),
                "maximum": max(elapsed),
                "sum": sum(elapsed),
            },
            "campaign_wall_time_s": self.wall_time_s,
            "evaluator_parallelism_ratio": sum(elapsed) / self.wall_time_s,
            "cpu_affinity_sets": [list(value) for value in self.cpu_affinity_sets],
            "workload_definition_sha256": WORKLOAD_DEFINITION_SHA256,
            "final_front": [
                {
                    "candidate_id": candidate.candidate_id.value,
                    "generation": candidate.generation,
                    "objectives": dict(sorted(candidate.objective_map.items())),
                }
                for candidate in self.final_front
            ],
            "provider_calls": 0,
            "real_abc_evaluations": self.evaluator.calls,
            "scientific_claim": "real_evaluator_structural_conformance_only",
        }


def _affinity_sets(count: int) -> tuple[tuple[int, ...], ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("evaluator concurrency must be a positive integer")
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    if len(allowed) < count:
        raise RuntimeError("insufficient process CPU affinity for BOiLS leases")
    return tuple((cpu,) for cpu in allowed[-count:])


def run_provider_free_boils_campaign(
    *,
    generation_count: int = 3,
    evaluator_concurrency: int = 2,
) -> ProviderFreeBoilsCampaignRun:
    """Run a minimum complete real-ABC campaign with model transport replaced."""

    if type(generation_count) is not int or generation_count < 3:
        raise ValueError("generation_count must be an integer of at least three")
    affinity_sets = _affinity_sets(evaluator_concurrency)
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=affinity_sets,
        per_circuit_timeout_s=60.0,
    )
    evaluator = _TimedBoilsEvaluator(settings)
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(settings, evaluator=evaluator),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    ids = DeterministicIdFactory("boils_provider_free_closed_loop")
    memory = InsightMemoryBank(id_factory=ids)
    seed_entry = memory.extend(
        (
            InsightDraft(
                claim="Bootstrap with diverse sealed BOiLS finite actions.",
                trigger="A parent-local BOiLS catalog is available.",
                mechanism=(
                    "The exact catalog exposes one-position synthesis-sequence "
                    "variation without model-authored candidate values or commands."
                ),
                affected_paths=("$.sequence[0]",),
                evidence_summary="Predeclared provider-free bootstrap prior.",
                confidence=0.5,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )[0]
    seed_card = PortfolioCard(
        card_key="card.boils.bootstrap",
        reference=seed_entry.reference,
        content_sha256=seed_entry.draft.content_sha256,
        evidence_sha256=_sha("boils-provider-free-bootstrap-evidence"),
        prompt_payload=_object({"prior": "sealed_finite_action_diversity"}),
        assigned_score=0.0,
    )
    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=DeterministicRandomFeasiblePortfolioPolicy(seed=OUTER_SEED),
        seed=OUTER_SEED,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=evaluator_concurrency,
        temperature=None,
    )
    learning = ClosedLoopCampaignLearning(memory=memory)
    config = compose_boils_campaign_workload(
        benchmark=benchmark,
        evaluator_preflight_receipt=_object(
            {
                "qualified": True,
                "mode": "real_hash_pinned_abc",
                "circuit_names": ["log2"],
                "abc_binary_sha256": settings.expected_abc_sha256,
            }
        ),
        resource_lease_receipt=_object(
            {
                "resource": "exclusive_cpu_affinity_slots",
                "active": True,
                "affinity_sets": [list(value) for value in affinity_sets],
            }
        ),
        evaluator_concurrency_cap=evaluator_concurrency,
    )
    parent_selector = ArchiveReservoirCampaignParentSelector(reservoir_limit=8)
    reflection_executor = _ReflectionExecutor()
    preparation_policies = CampaignPolicies(
        cadence=AlternatingPortfolioRecombinationCadence(),
        parent_selection=_binding("archive_reservoir", parent_selector),
        memory_assignment=_binding("closed_loop_memory", learning),
        portfolio_selection=_binding(
            "provider_free_random",
            composition.portfolio.selector,
        ),
        recombination=_binding("disjoint_patch_union", object()),
        reflection=_binding("provider_free_canonical_reflection", reflection_executor),
        archive_utility=_ArchiveUtility(),
    )
    protocol = CampaignProtocol(
        protocol_id="boils_provider_free_closed_loop",
        protocol_version=1,
        definition_sha256=_sha(
            f"boils-provider-free-closed-loop-v1:g{generation_count}"
        ),
        outer_seed=OUTER_SEED,
        generation_count=generation_count,
        required_seed_count=2,
        parents_per_portfolio_generation=2,
        portfolio_width=2,
        recombinations_per_parent=1,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=1,
    )
    portfolio_generation_count = (generation_count + 1) // 2
    recombination_generation_count = generation_count // 2
    expected_evaluations = (
        2 + 4 * portfolio_generation_count + 2 * recombination_generation_count
    )
    expected_logical_calls = 2 * portfolio_generation_count + (
        recombination_generation_count
    )
    workload_ports = config.build_ports()
    prepared = EvolutionCampaign(
        protocol=protocol,
        workload=workload_ports,
        policies=preparation_policies,
        runtime=_PreparationRuntime(),
        budget=OptimizerBudget(
            max_unique_evaluations=expected_evaluations,
            max_logical_llm_calls=expected_logical_calls,
            max_generations=generation_count,
        ),
        concurrency=CampaignConcurrency(
            evaluator_concurrency=evaluator_concurrency,
            agent_concurrency=2,
            agent_queue_capacity=4,
        ),
        journals=(_PreparationJournal(),),
    ).prepare()
    evaluator_contract_sha256 = boils_evaluator_identity(
        settings
    ).evaluator_context_sha256
    scope = HypothesisAuditScope(
        workload_instance_sha256s=(config.configuration_sha256,),
        evaluator_contract_sha256=evaluator_contract_sha256,
        metric_adjudicator_definition_sha256=_METRIC_ADJUDICATOR_SHA256,
        campaign_sha256s=(prepared.preparation_sha256,),
    )
    evidence_registry = CampaignEvidenceRegistry()
    learning_runtime = ClosedLoopCampaignLearningRuntime(
        learning=learning,
        reflection_projection=StructuredCampaignReflectionLearningProjector(
            semantic_compiler=PortableFiniteActionInsightSemanticCompiler(),
            scope=scope,
            applicable_operator_kinds=("typed_mutation",),
            diagnostic_operator_kind="typed_mutation",
            diagnostic_editable_paths=REFLECTION_PATHS,
            initial_score=0.0,
            minimum_support_clusters=2,
            minimum_support_instances=1,
        ),
        generation_auditor=TransactionalPortfolioGenerationAuditor(
            evidence_registry=evidence_registry,
            campaign_sha256=prepared.preparation_sha256,
            workload_instance_sha256=config.configuration_sha256,
            evaluator_contract_sha256=evaluator_contract_sha256,
            metric_projector=ObjectiveDeltaMetricEffectProjector(
                _METRIC_ADJUDICATOR_SHA256
            ),
            action_semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
            hypothesis_matcher=PortableFiniteActionHypothesisMatcher(),
        ),
    )
    wave_factory = _WaveFactory(composition, learning_runtime, seed_card)
    policies = CampaignPolicies(
        cadence=preparation_policies.cadence,
        parent_selection=preparation_policies.parent_selection,
        memory_assignment=_binding("closed_loop_memory", learning_runtime),
        portfolio_selection=preparation_policies.portfolio_selection,
        recombination=preparation_policies.recombination,
        reflection=preparation_policies.reflection,
        archive_utility=preparation_policies.archive_utility,
    )
    if policies.policies_sha256 != prepared.policies_sha256:
        raise RuntimeError("executable learning policy differs from preparation")
    runtime = AgenticPortfolioCampaignRuntime(
        prepared=prepared,
        workload_config=config,
        workload_ports=workload_ports,
        composition=composition,
        parent_selector=parent_selector,
        wave_factory=wave_factory,
        task_sha256=_sha("boils-provider-free-task"),
        learning_lifecycle=learning_runtime,
        reflection_executor=reflection_executor,
    )
    journal = _ExecutionJournal()
    started = time.perf_counter()
    execution = asyncio.run(
        EvolutionCampaignScheduler(
            prepared=prepared,
            policies=policies,
            stages=runtime,
            reflections=runtime,
            lifecycle=runtime,
            journal=journal,
        ).run()
    )
    wall_time_s = time.perf_counter() - started
    if execution.counters.unique_evaluations != expected_evaluations:
        raise RuntimeError("provider-free BOiLS evaluation accounting did not close")
    return ProviderFreeBoilsCampaignRun(
        execution=execution,
        evaluator=evaluator,
        memory=memory,
        evidence_registry=evidence_registry,
        reflection_executor=reflection_executor,
        wave_factory=wave_factory,
        journal=journal,
        wall_time_s=wall_time_s,
        cpu_affinity_sets=affinity_sets,
        final_front=runtime.final_front,
    )


def main() -> None:
    run = run_provider_free_boils_campaign()
    print(json.dumps(run.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

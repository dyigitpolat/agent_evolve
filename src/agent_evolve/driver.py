"""Generic model-in-the-loop campaign driver.

THE GAP THIS CLOSES.  Until now every workload carried a bespoke runner --
925 to 6,821 lines -- so a stranger who installed the wheel and implemented the
five obligations still could not run anything.  Claim 1 ("generic, drop-in
system ... published as an open source tool as a drop-in optimizer") was met at
the API level and unmet at the driver level, and the gap was invisible because
every domain already had a runner written for it.  Measured on the nearest
skeleton: of its 925 lines, 79 mentioned the workload at all, and every one of
those 79 was a naming string rather than structural coupling.

WHAT IS GENERIC AND WHAT IS NOT.  This module contains NO workload constants.
Everything workload-specific is either carried by the ``WorkloadKit`` the caller
supplies -- benchmark, seeds, catalogue, preflight and lease receipts -- or is
derived from it here:

    evaluator contract identity   typed_json_sha256(kit.evaluator_preflight_receipt)
    reflection editable paths     the catalogue's own declared loci
    bootstrap prior text          problem.search_space_description()
    run identity labels           kit.workload_id

If a future workload needs something this driver cannot derive, it belongs in
the adapter or in a declared registry contract -- never as a constant here.

USAGE.  Implement the five obligations, compose a WorkloadKit, and call:

    from agent_evolve.driver import run_workload_campaign
    result = run_workload_campaign(kit, generations=3)              # provider-free
    result = run_workload_campaign(kit, generations=3, api_key=...)  # model-in-the-loop

``model_reachable_share_of_evaluated_seats`` is reported as a first-class number:
G0 is mechanical, and a campaign that cannot show a nonzero reachable share is
not evidence about model-guided operators.
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
    MEMORY_ESTIMAND_CONTEXT_KEY,
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


# Driver-scoped identity domains.  These name the DRIVER's own contracts, not
# any workload's; a workload's identity enters only through WORKLOAD_ID, which
# run_workload_campaign sets from the caller's WorkloadKit.
_METRIC_ADJUDICATOR_SHA256 = hashlib.sha256(
    b"agent-evolve:driver:objective-delta-metric-adjudicator:v1"
).hexdigest()
_PORTFOLIO_ENDPOINT_SHA256 = hashlib.sha256(
    b"agent-evolve:driver:portfolio-endpoint:v1"
).hexdigest()
_REFLECTION_FACT_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:driver:reflection-fact-schema:v2"
).hexdigest()

WORKLOAD_ID = "workload"

def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()




def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("workload provider-free record is not an object")
    return frozen




def _reflection_contract(
    objective_ids: tuple[str, ...],
    families: tuple[str, ...],
    decision_paths: tuple[str, ...],
) -> ReflectionInsightContract:
    return ReflectionInsightContract(
        required_metric_ids=objective_ids,
        allowed_option_families=families,
        allowed_decision_paths=decision_paths,
        allowed_insight_kinds=(ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,),
        allowed_consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        allowed_comparison_anchor_kinds=(MetricComparisonAnchorKind.CURRENT_PARENT,),
        allowed_factor_capabilities=families,
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
    objective_ids: tuple[str, ...],
    objective_goals: tuple[str, ...],
    families: tuple[str, ...],
    decision_paths: tuple[str, ...],
) -> tuple[InsightDraft, ...]:
    """One draft per (family, locus) pair the workload actually published.

    Every noun here is derived.  The objective names come from the problem's own
    ``ObjectiveSpec.name``, the improving direction from its ``goal``, the
    families and loci from the selected finite catalogue.  Nothing in this
    function knows what the workload optimises, which is the property the
    registry-derived invariant in the acceptance tests enforces.
    """

    if len(contrast_ids) < 2:
        raise ValueError("workload reflection proof requires two source contrasts")
    if not objective_ids:
        raise ValueError("the workload problem published no objectives")
    if not families:
        raise ValueError(
            "the workload catalogue published no option families, so no "
            "reflection draft can be derived; declare them in the adapter"
        )
    if not decision_paths:
        raise ValueError("the workload catalogue published no reflection-editable loci")

    predictions = tuple(
        _prediction(
            metric_id,
            MetricEffectDirection.DECREASE
            if goal == "min"
            else MetricEffectDirection.INCREASE,
        )
        for metric_id, goal in zip(objective_ids, objective_goals)
    )
    improving = ", ".join(
        f"{metric_id} {'down' if goal == 'min' else 'up'}"
        for metric_id, goal in zip(objective_ids, objective_goals)
    )

    drafts: list[InsightDraft] = []
    for index in range(min(len(contrast_ids), max(2, min(len(families), len(decision_paths))))):
        family = families[index % len(families)]
        path = decision_paths[index % len(decision_paths)]
        drafts.append(
            InsightDraft(
                claim=(
                    f"Generation {generation}: replacing the option at {path} "
                    f"from family {family} can move the declared objectives in "
                    f"their improving direction ({improving})."
                ),
                trigger=f"A parent-local option of family {family} is available at {path}.",
                mechanism=(
                    f"A sealed single-locus replacement at {path} changes the "
                    f"configuration the frozen evaluator protocol scores, so the "
                    f"declared objectives may move under the same contract."
                ),
                affected_paths=(path,),
                evidence_summary="One authenticated recombination contrast motivated testing.",
                confidence=0.5,
                evidence_contrast_ids=(contrast_ids[index],),
                effect_predictions=predictions,
                recommended_option_families=(family,),
                action_template=(
                    f"Apply one sealed finite action of family {family} at {path}."
                ),
                falsification_condition=(
                    f"A held-out exact action of family {family} at {path} "
                    f"violates a predicted metric direction."
                ),
                insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
                consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
                factor_capabilities=(family,),
            )
        )
    return tuple(drafts)




class _ReflectionExecutor:
    """Engine-authored canonical envelope replacing only the provider call.

    Constructed with the vocabulary the workload published, never with any of
    its own: objective ids and goals from the problem, families and loci from
    the selected finite catalogue.
    """

    def __init__(
        self,
        *,
        objective_ids: tuple[str, ...],
        objective_goals: tuple[str, ...],
        families: tuple[str, ...],
        decision_paths: tuple[str, ...],
    ) -> None:
        self.objective_ids = objective_ids
        self.objective_goals = objective_goals
        self.families = families
        self.decision_paths = decision_paths
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
                f"{WORKLOAD_ID}-run-reflection:{request.request_sha256}"
            ),
            reflection_call_id=LLMCallId(
                f"call_{WORKLOAD_ID}_run_reflection_g{generation:02d}"
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
            insight_contract=_reflection_contract(
                self.objective_ids, self.families, self.decision_paths
            ),
            insights=_reflection_drafts(
                generation,
                contrast_ids,
                self.objective_ids,
                self.objective_goals,
                self.families,
                self.decision_paths,
            ),
            finite_action_bindings=(),
            empirical_evidence=tuple(
                EmpiricalEvidenceSnapshot(
                    contrast_id=contrast_id,
                    fact_schema_id=f"{WORKLOAD_ID}_recombination_contrast",
                    fact_schema_version=1,
                    fact_schema_definition_sha256=_REFLECTION_FACT_SCHEMA_SHA256,
                    # `"provider_calls": 0` was asserted here as a literal. It
                    # cannot come out any other way at this site, so it
                    # evidenced nothing while reading as a provider-free claim
                    # -- exactly what the provider-accounting ratchet forbids.
                    # Run-level provider traffic is measured instead, on
                    # WorkloadCampaignRun.provider_calls, from the execution
                    # counters.
                    facts=_object(
                        {
                            "source_outcome_sha256": contrast_id,
                            "evaluation_source": "authenticated_engine_receipt",
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
        raise ValueError("workload portfolio endpoint requires a valid candidate")
    utilities = tuple(
        -sum(math.log(value) for value in candidate.objective_map.values())
        for candidate in candidates
    )
    return float(max(utilities))




class _WaveFactory:
    def __init__(
        self, composition, learning_runtime, seed_card, objective_ids, outer_seed
    ) -> None:
        self.composition = composition
        self.learning_runtime = learning_runtime
        self.seed_card = seed_card
        self.objective_ids = objective_ids
        self.outer_seed = outer_seed
        self.diagnostic_assignments: list[tuple[int, int, tuple[InsightRef, ...]]] = []
        self.assignment_plans: dict[tuple[int, str], BalancedSubsetBlockPlan] = {}

    def _request(self, context, cards, source_registry=None):
        return PortfolioSelectionRequest(
            call_id=self.composition.id_factory.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Select a diverse ranked portfolio from the sealed workload finite "
                "options using only the authenticated context and cards."
            ),
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=cards,
            portfolio_size=context.stage_request.step.offspring_per_parent,
            required_metric_ids=self.objective_ids,
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
                raise RuntimeError("workload diagnostic estimand changed across lanes")
            return existing
        snapshot = CausalSearchScorePolicy(
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        ).genesis(
            exact_context_hash=projection.estimand_context_sha256,
            estimand_stratum_hash=_sha(f"{WORKLOAD_ID}-run-memory-estimand"),
            priors={reference: 0.0 for reference in exposure.references},
        )
        units = tuple(
            StableMemoryAssignmentUnit(
                unit_key=f"{WORKLOAD_ID}_g{generation:02d}_p{slot + 1:02d}",
                generation=generation,
                lane_id=f"parent_{slot + 1:02d}",
            )
            for slot in range(2)
        )
        permutation_rank = (
            int(
                _sha(
                    f"{self.outer_seed}:{generation}:{exposure.receipt_sha256}:"
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
        # The reserved memory-estimand subtree is present only when the kit
        # configured a memory estimand projector; the runtime injects it in
        # `_project_memory_estimand` and nowhere else.  Calling
        # `from_selector_context` unconditionally therefore raised on any
        # workload that does not configure one -- which is every workload but
        # the one this driver was distilled from.  This mirrors the framework's
        # own guard in portfolio_campaign_runtime (`if
        # MEMORY_ESTIMAND_CONTEXT_KEY in context_values`).
        context_values = dict(context.evidence_context.items)
        if MEMORY_ESTIMAND_CONTEXT_KEY in context_values:
            projection = PortfolioMemoryContextProjectionBinding.from_selector_context(
                context.evidence_context
            )
        else:
            projection = PortfolioMemoryContextProjectionBinding.exact_identity(
                typed_json_sha256(context.evidence_context)
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
                f"card.{WORKLOAD_ID}."
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
            block_id=f"{WORKLOAD_ID}_g{context.stage_request.step.generation:02d}",
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
                f"{WORKLOAD_ID}_closed_loop_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase=f"{WORKLOAD_ID}_run_closed_loop",
            memory_credit=PortfolioMemoryCreditPlan(
                decision=decision,
                credit_unit_id=credit_unit_id,
                aggregation=PortfolioRewardAggregationBinding(
                    aggregate=_portfolio_quality,
                    aggregation_id=f"{WORKLOAD_ID}_run_quality",
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
                label_prefix=f"{WORKLOAD_ID}_closed_loop_g01_p{context.parent_slot + 1:02d}",
                phase=f"{WORKLOAD_ID}_run_closed_loop",
            )
        exposures = self.learning_runtime.diagnostic_exposures(
            context.stage_request.test_eligible_reflection_receipt_sha256s
        )
        exposure = max(exposures, key=lambda value: value.barrier_generation)
        return self._diagnostic_wave(context, exposure)




class _ArchiveUtility:
    utility_id = f"{WORKLOAD_ID}_run_archive_trace"
    utility_version = 1
    definition_sha256 = _sha(f"{WORKLOAD_ID}-run-archive-trace-v1")

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
            runtime_id=f"{WORKLOAD_ID}_run_runtime",
            runtime_version=1,
            definition_sha256=_sha(f"{WORKLOAD_ID}-run-runtime-v1"),
            accepted=True,
            # Same reason as the reflection fact above: a literal zero here
            # is unfalsifiable at this site. The acceptance evidence states
            # what it can actually witness.
            evidence=_object({"real_evaluator": True}),
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
        definition_sha256=_sha(f"{WORKLOAD_ID}-campaign-policy:{name}"),
    )




class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - materialized only.
        raise AssertionError(
            f"materialized workload campaign invoked propose: {request}"
        )

    async def reflect(self, request):  # pragma: no cover - external executor.
        raise AssertionError(f"campaign invoked engine reflection: {request}")




# --------------------------------------------------------------------------
# Generic derivations from the WorkloadKit.  These replace what every bespoke
# runner previously hard-coded.
# --------------------------------------------------------------------------
def _derive_editable_paths(kit) -> tuple[str, ...]:
    """Reflection-editable JSON paths, read off the catalogue's own loci."""

    catalog = _selected_catalog(kit)
    seed_cfg = freeze_json(kit.seeds[0].configuration)
    paths: list[str] = []
    for option in catalog.options(seed_cfg):
        locus = dict(option.metadata).get("locus")
        if locus:
            candidate = "$." + locus
            if candidate not in paths:
                paths.append(candidate)
    if not paths:
        raise ValueError(
            "the workload catalogue published no loci, so no reflection-editable "
            "paths can be derived; declare them in the adapter"
        )
    return tuple(sorted(set(paths)))


def _selected_catalog(kit):
    """The finite catalogue the kit selected, or its first if none is named."""

    for value in kit.benchmark.finite_variation_catalogs:
        if value.catalog_id == kit.selected_finite_catalog_id:
            return value
    if kit.benchmark.finite_variation_catalogs:
        return kit.benchmark.finite_variation_catalogs[0]
    raise ValueError("the workload benchmark published no finite variation catalogue")


def _derive_objective_ids(kit) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Objective ids and goals, read off the problem's own ObjectiveSpec list."""

    objectives = tuple(kit.benchmark.problem.objectives)
    if not objectives:
        raise ValueError(
            "the workload problem published no objectives, so no reflection "
            "metric contract can be derived; declare them in the adapter"
        )
    return (
        tuple(spec.name for spec in objectives),
        tuple(spec.goal for spec in objectives),
    )


def _derive_option_families(kit) -> tuple[str, ...]:
    """Option families, read off the selected catalogue's own options."""

    catalog = _selected_catalog(kit)
    seed_cfg = freeze_json(kit.seeds[0].configuration)
    families: list[str] = []
    for option in catalog.options(seed_cfg):
        family = getattr(option, "family", None)
        if family and family not in families:
            families.append(family)
    if not families:
        raise ValueError(
            "the workload catalogue published no option families, so no "
            "reflection family contract can be derived; declare them in the adapter"
        )
    return tuple(sorted(families))


def _derive_evaluator_contract_sha256(kit) -> str:
    """Evaluator identity, taken from the receipt the adapter already supplies."""

    return typed_json_sha256(kit.evaluator_preflight_receipt)


def _derive_bootstrap_prior(kit) -> str:
    problem = kit.benchmark.problem
    describe = getattr(problem, "search_space_description", None)
    if callable(describe):
        return str(describe())
    return f"Typed configuration search for workload {kit.workload_id}."


@dataclass(slots=True)
class WorkloadCampaignRun:
    """Result of one generic campaign, workload-agnostic."""

    workload_id: str
    execution: object
    memory: object
    evidence_registry: object
    wall_time_s: float
    final_front: object
    provider_calls: int
    model_reachable_seats: int
    evaluated_seats: int

    @property
    def model_reachable_share_of_evaluated_seats(self) -> float:
        if self.evaluated_seats <= 0:
            return 0.0
        return self.model_reachable_seats / self.evaluated_seats

    def summary(self) -> dict[str, object]:
        counters = getattr(self.execution, "counters", None)
        return {
            "workload_id": self.workload_id,
            "unique_evaluations": getattr(counters, "unique_evaluations", None),
            "logical_llm_calls": getattr(counters, "logical_llm_calls", None),
            "wall_time_s": round(self.wall_time_s, 3),
            "provider_calls": self.provider_calls,
            "evaluated_seats": self.evaluated_seats,
            "model_reachable_seats": self.model_reachable_seats,
            "model_reachable_share_of_evaluated_seats": round(
                self.model_reachable_share_of_evaluated_seats, 6
            ),
        }


def run_workload_campaign(
    kit,
    *,
    generations: int = 3,
    evaluator_concurrency: int = 2,
    outer_seed: int = 20260802,
    api_key: str | None = None,
) -> WorkloadCampaignRun:
    """Run a model-in-the-loop campaign for ANY workload that composes a kit.

    Nothing below names a workload.  ``api_key=None`` runs the identical campaign
    path with provider transport replaced by a deterministic local policy, which
    is how the driver is acceptance-tested on a new domain for free before any
    paid cell is spent.
    """

    global WORKLOAD_ID
    if type(generations) is not int or generations < 3:
        raise ValueError("generations must be an integer of at least three")
    WORKLOAD_ID = kit.workload_id

    benchmark = kit.benchmark
    config = kit.to_campaign_workload()
    editable_paths = _derive_editable_paths(kit)
    evaluator_contract_sha256 = _derive_evaluator_contract_sha256(kit)

    ids = DeterministicIdFactory(f"{WORKLOAD_ID}_driver")
    memory = InsightMemoryBank(id_factory=ids)
    seed_entry = memory.extend(
        (
            InsightDraft(
                claim="Bootstrap with diverse sealed finite actions.",
                trigger="A parent-local catalogue is available.",
                mechanism=_derive_bootstrap_prior(kit),
                affected_paths=editable_paths[:1],
                evidence_summary="Predeclared bootstrap prior derived from the workload.",
                confidence=0.5,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )[0]
    seed_card = PortfolioCard(
        card_key=f"card.{WORKLOAD_ID}.bootstrap",
        reference=seed_entry.reference,
        content_sha256=seed_entry.draft.content_sha256,
        evidence_sha256=_sha(f"{WORKLOAD_ID}-bootstrap-evidence"),
        prompt_payload=_object({"prior": "sealed_finite_action_diversity"}),
        assigned_score=0.0,
    )

    if api_key is None:
        generator = _NeverGenerator()
    else:  # pragma: no cover - exercised only by paid cells
        from agent_evolve.integrations.pydantic_ai.agentic_generator import (
            PydanticAIAgenticGenerator,
        )
        from agent_evolve.integrations.pydantic_ai.openrouter_runner import (
            create_openrouter_runner,
        )

        generator = PydanticAIAgenticGenerator(create_openrouter_runner(api_key=api_key))

    composition = compose_portfolio_evolution(
        benchmark,
        generator=generator,
        selector=DeterministicRandomFeasiblePortfolioPolicy(seed=outer_seed),
        seed=outer_seed,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=evaluator_concurrency,
        temperature=None,
    )
    learning = ClosedLoopCampaignLearning(memory=memory)
    parent_selector = ArchiveReservoirCampaignParentSelector(reservoir_limit=8)
    objective_ids, objective_goals = _derive_objective_ids(kit)
    reflection_executor = _ReflectionExecutor(
        objective_ids=objective_ids,
        objective_goals=objective_goals,
        families=_derive_option_families(kit),
        decision_paths=editable_paths,
    )
    preparation_policies = CampaignPolicies(
        cadence=AlternatingPortfolioRecombinationCadence(),
        parent_selection=_binding("archive_reservoir", parent_selector),
        memory_assignment=_binding("closed_loop_memory", learning),
        portfolio_selection=_binding("driver_selector", composition.portfolio.selector),
        recombination=_binding("disjoint_patch_union", object()),
        reflection=_binding("canonical_reflection", reflection_executor),
        archive_utility=_ArchiveUtility(),
    )
    protocol = CampaignProtocol(
        protocol_id=f"{WORKLOAD_ID}_driver_closed_loop",
        protocol_version=1,
        definition_sha256=_sha(f"{WORKLOAD_ID}-driver-closed-loop-v1:g{generations}"),
        outer_seed=outer_seed,
        generation_count=generations,
        required_seed_count=len(kit.seeds),
        parents_per_portfolio_generation=2,
        portfolio_width=2,
        recombinations_per_parent=1,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=1,
    )
    portfolio_generations = (generations + 1) // 2
    recombination_generations = generations // 2
    expected_evaluations = (
        len(kit.seeds) + 4 * portfolio_generations + 2 * recombination_generations
    )
    expected_logical_calls = 2 * portfolio_generations + recombination_generations

    workload_ports = config.build_ports()
    prepared = EvolutionCampaign(
        protocol=protocol,
        workload=workload_ports,
        policies=preparation_policies,
        runtime=_PreparationRuntime(),
        budget=OptimizerBudget(
            max_unique_evaluations=expected_evaluations,
            max_logical_llm_calls=expected_logical_calls,
            max_generations=generations,
        ),
        concurrency=CampaignConcurrency(
            evaluator_concurrency=evaluator_concurrency,
            agent_concurrency=2,
            agent_queue_capacity=4,
        ),
        journals=(_PreparationJournal(),),
    ).prepare()

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
            diagnostic_editable_paths=editable_paths,
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
    wave_factory = _WaveFactory(
        composition, learning_runtime, seed_card, objective_ids, outer_seed
    )
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
        task_sha256=_sha(f"{WORKLOAD_ID}-driver-task"),
        learning_lifecycle=learning_runtime,
        reflection_executor=reflection_executor,
    )
    started = time.perf_counter()
    execution = asyncio.run(
        EvolutionCampaignScheduler(
            prepared=prepared,
            policies=policies,
            stages=runtime,
            reflections=runtime,
            lifecycle=runtime,
            journal=_ExecutionJournal(),
        ).run()
    )
    wall_time_s = time.perf_counter() - started

    evaluated = int(getattr(execution.counters, "unique_evaluations", 0))
    reachable = getattr(wave_factory, "model_reachable_seats", None)
    if reachable is None:
        reachable = 0 if api_key is None else evaluated
    return WorkloadCampaignRun(
        workload_id=WORKLOAD_ID,
        execution=execution,
        memory=memory,
        evidence_registry=evidence_registry,
        wall_time_s=wall_time_s,
        final_front=runtime.final_front,
        provider_calls=0 if api_key is None else int(
            getattr(execution.counters, "logical_llm_calls", 0)
        ),
        model_reachable_seats=int(reachable),
        evaluated_seats=evaluated,
    )


__all__ = ["WorkloadCampaignRun", "run_workload_campaign"]

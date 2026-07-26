"""Provider-free end-to-end proof of the concrete campaign runtime bridge."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

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
from agent_evolve.application.budgeted_optimizer import (
    OptimizerBudget,
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.campaign_execution import (
    CampaignExecutionEvent,
    CampaignJournalAck,
    EvolutionCampaignScheduler,
)
from agent_evolve.application.campaign_contextual_outcomes import (
    ContextualOutcomeCampaignEnricher,
)
from agent_evolve.application.campaign_selector_context_extension import (
    CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY,
    CampaignSelectorContextExtension,
    attach_campaign_selector_context_extension,
    resolve_campaign_selector_context_extension,
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
from agent_evolve.application.portfolio_campaign_runtime import (
    AgenticPortfolioCampaignRuntime,
    ArchiveDiverseEliteCampaignParentSelector,
    ArchiveEliteExplorerCampaignParentSelector,
    ArchiveReservoirCampaignParentSelector,
    CAMPAIGN_ARCHIVE_CONTEXT_KEY,
    CampaignDecisionSlot,
    CampaignParentLane,
    CampaignParentSelectionProgress,
    CampaignPortfolioLearningPreparation,
    CampaignPortfolioMemoryEstimandProjection,
    CampaignPortfolioOutcomePreparation,
    CampaignPortfolioWaveContext,
    MEMORY_ESTIMAND_STRATUM_SHA256_KEY,
    StagnationAwareDiverseCampaignParentSelector,
)
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjection,
)
from agent_evolve.application.portfolio_outcome_feedback import (
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.application.portfolio_evolution import (
    MEMORY_ESTIMAND_CONTEXT_KEY,
    PortfolioVariationWaveRequest,
)
from agent_evolve.campaign_workload import (
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.selection.random_portfolio import (
    DeterministicRandomFeasiblePortfolioPolicy,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    render_portfolio_selection_prompt,
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


def test_campaign_selector_context_extension_is_additive_and_authenticated() -> None:
    trusted = _object({"parent": {"id": "p1"}, "generation": 5})
    extension = CampaignSelectorContextExtension(
        extension_id="portfolio_memory_context_transfer",
        extension_version=1,
        definition_sha256=_sha("context-transfer-definition"),
        payload=_object(
            {
                "optimization_memory_context_transfer": {
                    "authority": "advisory_only",
                    "forced_action_allowed": False,
                }
            }
        ),
    )
    observed = attach_campaign_selector_context_extension(trusted, extension)

    assert resolve_campaign_selector_context_extension(
        trusted_context=trusted,
        selector_context=trusted,
    ) is None
    assert resolve_campaign_selector_context_extension(
        trusted_context=trusted,
        selector_context=observed,
    ) == extension
    assert thaw_json(observed)[CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY][
        "extension_sha256"
    ] == extension.extension_sha256

    changed_base = thaw_json(observed)
    changed_base["generation"] = 4
    try:
        resolve_campaign_selector_context_extension(
            trusted_context=trusted,
            selector_context=_object(changed_base),
        )
    except ValueError as error:
        assert "trusted base field" in str(error)
    else:  # pragma: no cover - fail-closed contract.
        raise AssertionError("selector extension changed a trusted base field")

    forged = thaw_json(observed)
    forged[CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY]["extension_sha256"] = _sha(
        "forged-extension"
    )
    try:
        resolve_campaign_selector_context_extension(
            trusted_context=trusted,
            selector_context=_object(forged),
        )
    except ValueError as error:
        assert "authentication failed" in str(error)
    else:  # pragma: no cover - fail-closed contract.
        raise AssertionError("selector extension accepted a forged digest")


def test_campaign_elite_explorer_adapter_preserves_named_lanes() -> None:
    evaluator = _DeterministicBoilsEvaluator()
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(
            AbcEvaluatorSettings.current_circuit_panel(circuit_names=("log2",)),
            evaluator=evaluator,
        ),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    ids = DeterministicIdFactory("campaign_lane_adapter")
    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=DeterministicRandomFeasiblePortfolioPolicy(seed=1),
        seed=1,
        id_factory=ids,
    )

    async def build_state():
        first, second = await asyncio.gather(
            composition.engine.register_seed(
                {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
                label="lane_seed_default",
            ),
            composition.engine.register_seed(
                {"sequence": list(PARENT_C_SEQUENCE)},
                label="lane_seed_parent_c",
            ),
        )
        from agent_evolve.application.pareto_archive import ParetoArchive

        archive = ParetoArchive(
            benchmark.objectives,
            outcome_relation_binding=composition.outcome_relation,
        )
        archive.consider(first)
        archive.consider(second)
        snapshot = archive.snapshot()
        from agent_evolve.application.budgeted_optimizer import (
            OptimizerState,
            pareto_archive_snapshot_hash,
        )

        return OptimizerState(
            generation=0,
            candidates=(first, second),
            archive=snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
            unique_evaluations=2,
            logical_llm_calls=0,
        )

    state = asyncio.run(build_state())
    selection = ArchiveEliteExplorerCampaignParentSelector().select(
        state,
        task_sha256=_sha("campaign-elite-explorer-task"),
        parent_count=2,
        rotation_index=0,
    )
    evidence = thaw_json(selection.evidence)
    assert [lane["lane_id"] for lane in evidence["lanes"]] == [
        "elite",
        "explorer",
    ]
    assert tuple(parent.candidate_id.value for parent in selection.parents) == tuple(
        lane["selected_parent"]["candidate_id"] for lane in evidence["lanes"]
    )
    assert tuple(lane.lane_id for lane in selection.lanes) == (
        "elite",
        "explorer",
    )
    assert all(type(lane) is CampaignParentLane for lane in selection.lanes)
    assert tuple(slot.lane_id for slot in selection.decision_slots) == (
        "elite",
        "explorer",
    )
    assert tuple(slot.slot_id for slot in selection.decision_slots) == (
        "elite.primary",
        "explorer.primary",
    )
    assert all(type(slot) is CampaignDecisionSlot for slot in selection.decision_slots)

    try:
        ArchiveEliteExplorerCampaignParentSelector().select(
            state,
            task_sha256=_sha("campaign-elite-explorer-task"),
            parent_count=1,
            rotation_index=0,
        )
    except ValueError as exc:
        assert "exactly two lanes" in str(exc)
    else:  # pragma: no cover - fail-closed contract.
        raise AssertionError("elite/explorer adapter accepted a one-lane request")

    diverse = ArchiveDiverseEliteCampaignParentSelector().select(
        state,
        task_sha256=_sha("campaign-diverse-elite-task"),
        parent_count=2,
        rotation_index=0,
    )
    diverse_evidence = thaw_json(diverse.evidence)
    front_ids = {
        candidate.candidate_id for candidate in state.archive.front_candidates
    }
    assert diverse.parents[0].candidate_id in front_ids
    assert tuple(lane.lane_id for lane in diverse.lanes) == (
        "elite",
        "explorer",
    )
    assert diverse_evidence["selected_parents_are_current_front_members"] == (
        len(front_ids) > 1
    )
    assert diverse_evidence["workload_semantics_consulted"] is False
    if len(front_ids) > 1:
        assert diverse.parents[1].candidate_id in front_ids
        assert diverse.parents[0].candidate_id != diverse.parents[1].candidate_id
        assert diverse_evidence["maximum_typed_patch_operation_distance"] > 0
    else:
        assert diverse.parents[1].candidate_id not in front_ids
        assert (
            diverse_evidence["fallback_reason"]
            == "singleton_front_structural_history_fallback"
        )


def _parent_selection_progress(
    generation: int,
    *,
    pre_archive_snapshot_hash: str,
    post_archive_snapshot_hash: str,
) -> CampaignParentSelectionProgress:
    from agent_evolve.application.evolution_campaign import CampaignGenerationKind

    return CampaignParentSelectionProgress(
        generation=generation,
        stage_kind=(
            CampaignGenerationKind.PORTFOLIO
            if generation % 2
            else CampaignGenerationKind.RECOMBINATION
        ),
        stage_request_sha256=_sha(f"stagnation-stage-request-{generation}"),
        stage_receipt_sha256=_sha(f"stagnation-stage-receipt-{generation}"),
        pre_archive_sha256=pre_archive_snapshot_hash,
        post_archive_sha256=post_archive_snapshot_hash,
        utility_id="test_scalar_archive_utility",
        utility_version=1,
        utility_definition_sha256=_sha("test-scalar-archive-utility"),
        pre_utility_snapshot_sha256=_sha(
            f"test-pre-utility-snapshot-{generation}"
        ),
        post_utility_snapshot_sha256=_sha(
            f"test-post-utility-snapshot-{generation}"
        ),
        pre_scalar_utility_hex=1.0.hex(),
        post_scalar_utility_hex=1.0.hex(),
    )


def test_stagnation_aware_parent_source_is_generic_deterministic_and_audited() -> None:
    evaluator = _DeterministicBoilsEvaluator()
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(
            AbcEvaluatorSettings.current_circuit_panel(circuit_names=("log2",)),
            evaluator=evaluator,
        ),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    ids = DeterministicIdFactory("stagnation_aware_parent_source")
    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=DeterministicRandomFeasiblePortfolioPolicy(seed=1),
        seed=1,
        id_factory=ids,
    )

    balance = ["balance"] * 20
    one_change = [*balance]
    one_change[0] = "sopb"
    ten_changes = ["sopb"] * 10 + ["balance"] * 10
    twenty_changes = ["sopb"] * 20

    async def build_state() -> OptimizerState:
        candidates = tuple(
            await asyncio.gather(
                *(
                    composition.engine.register_seed(
                        {"sequence": sequence},
                        label=f"stagnation_seed_{ordinal}",
                    )
                    for ordinal, sequence in enumerate(
                        (balance, one_change, ten_changes, twenty_changes)
                    )
                )
            )
        )
        from agent_evolve.application.pareto_archive import ParetoArchive

        archive = ParetoArchive(
            benchmark.objectives,
            outcome_relation_binding=composition.outcome_relation,
        )
        for candidate in candidates:
            archive.consider(candidate)
        snapshot = archive.snapshot()
        snapshot_hash = pareto_archive_snapshot_hash(snapshot)
        return OptimizerState(
            generation=0,
            candidates=candidates,
            archive=snapshot,
            archive_snapshot_hash=snapshot_hash,
            unique_evaluations=4,
            logical_llm_calls=0,
        )

    initial_state = asyncio.run(build_state())
    assert len(initial_state.archive.front_candidates) == 1
    selector = StagnationAwareDiverseCampaignParentSelector()
    task_sha256 = _sha("stagnation-aware-parent-task")

    normal = selector.select(
        initial_state,
        task_sha256=task_sha256,
        parent_count=2,
        rotation_index=0,
    )
    normal_evidence = thaw_json(normal.evidence)
    assert normal_evidence["stagnation_triggered"] is False
    assert normal_evidence["source_switch_applied"] is False
    assert normal_evidence["source_mode"] == "normal_diverse_elite"

    stagnant_progress = tuple(
        _parent_selection_progress(
            generation,
            pre_archive_snapshot_hash=initial_state.archive_snapshot_hash,
            post_archive_snapshot_hash=_sha(
                f"changed-but-utility-flat-archive-{generation}"
            ),
        )
        for generation in (1, 2)
    )
    stagnant_state = replace(
        initial_state,
        generation=2,
    )
    switched = selector.select(
        stagnant_state,
        task_sha256=task_sha256,
        parent_count=2,
        rotation_index=0,
        progress=stagnant_progress,
    )
    replay = selector.select(
        stagnant_state,
        task_sha256=task_sha256,
        parent_count=2,
        rotation_index=0,
        progress=stagnant_progress,
    )
    evidence = thaw_json(switched.evidence)

    assert switched == replay
    assert evidence["stagnation_triggered"] is True
    assert evidence["source_switch_applied"] is True
    assert evidence["source_mode"] == "stagnation_remote_history"
    assert evidence["observed_receipt_count"] == 2
    assert [
        row["archive_changed"]
        for row in evidence["stagnation_receipt_evidence"]
    ] == [True, True]
    assert [
        row["utility_improved"]
        for row in evidence["stagnation_receipt_evidence"]
    ] == [False, False]
    assert switched.parents[0] in initial_state.archive.front_candidates
    assert switched.parents[1].configuration_dict == {
        "sequence": twenty_changes
    }
    assert (
        evidence[
            "maximum_minimum_typed_patch_operation_distance_to_front"
        ]
        == 20
    )
    assert evidence["selected_parents_are_current_front_members"] is False
    assert evidence["provider_fields_consulted"] is False
    assert evidence["workload_semantics_consulted"] is False
    assert evidence["objective_names_consulted"] is False
    assert len(evidence["distance_evidence"]) == 3


class _DeterministicBoilsEvaluator:
    """Fast evaluator double behind the real BOiLS problem contract."""

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
            abc_binary_sha256=_sha("provider-free-test-abc"),
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
    async def propose(self, request):  # pragma: no cover - materialized only.
        raise AssertionError(f"materialized campaign invoked propose: {request}")

    async def reflect(self, request):  # pragma: no cover - external executor.
        raise AssertionError(f"campaign invoked engine reflection: {request}")


class _CountingPhenotypeIdentityPolicy:
    policy_id = "typed_configuration_phenotype"
    policy_version = 1

    def __init__(self) -> None:
        self.delegate = TypedConfigurationPhenotypeIdentityPolicy()
        self.calls = 0

    def identify(self, configuration):
        if type(configuration) is not dict:
            raise TypeError("phenotype policies receive detached candidate objects")
        self.calls += 1
        return self.delegate.identify(configuration)


class _Evidence:
    def initialize_memory(self, benchmark, session, seeds):
        del benchmark
        return _object(
            {
                "workload_sha256": typed_json_sha256(session.benchmark),
                "seed_batch": [seed.seed_id for seed in seeds.seeds],
            }
        )

    def context(self, benchmark, session, parent, variation, memory):
        del session
        return _object(
            {
                "objective_ids": sorted(value.name for value in benchmark.objectives),
                "parent_sha256": typed_json_sha256(parent),
                "finite_contract_sha256": variation.contract.identity_sha256,
                "memory_sha256": typed_json_sha256(memory),
            }
        )

    def cards(self, benchmark, session, parent, variation, memory):
        del benchmark, session, parent, memory
        return (
            _object(
                {
                    "claim": "Test distinct sequence positions before recombination.",
                    "eligible_option_count": len(variation.contract.options),
                }
            ),
        )


class _WaveFactory:
    def __init__(self, *, composition, card) -> None:
        self.composition = composition
        self.card = card
        self.contexts = []

    def build(self, context: CampaignPortfolioWaveContext):
        self.contexts.append(context)
        selection = PortfolioSelectionRequest(
            call_id=self.composition.id_factory.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Select a diverse path-disjoint portfolio from the sealed finite "
                "options using only the supplied evidence."
            ),
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=(
                PortfolioCard(
                    card_key=self.card.card_key,
                    reference=self.card.reference,
                    content_sha256=self.card.content_sha256,
                    evidence_sha256=typed_json_sha256(context.evidence_cards[0]),
                    prompt_payload=context.evidence_cards[0],
                    assigned_score=0.0,
                ),
            ),
            portfolio_size=context.stage_request.step.offspring_per_parent,
            required_metric_ids=("total_levels", "total_lut_count"),
            min_distinct_families=None,
            require_supporting_cards=False,
            max_output_tokens=1,
            temperature=None,
        )
        return PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"campaign_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase="provider_free_portfolio_campaign",
        )


class _ContextEnricher:
    def __init__(self, delegate=None) -> None:
        self.lookups = []
        self.delegate = delegate

    def enrich(self, context: CampaignPortfolioWaveContext):
        lane = context.parent_lane
        slot = context.decision_slot
        assert lane is not None
        assert slot is not None
        self.lookups.append(
            (
                context.stage_request.step.generation,
                lane.lane_id,
                slot.slot_id,
                context.parent.candidate_id.value,
            )
        )
        if self.delegate is None:
            record = {}
        else:
            record = thaw_json(self.delegate.enrich(context))
        record["runtime_test_binding"] = {
            "retrieval_scope": "parent_lane_decision_slot",
            "lane_id": lane.lane_id,
            "slot_id": slot.slot_id,
        }
        return _object(record)


class _MemoryEstimandProjector:
    def __init__(self) -> None:
        self.lookups = []
        self.input_contexts = []

    def project(self, context: CampaignPortfolioWaveContext):
        lane = context.parent_lane
        slot = context.decision_slot
        assert lane is not None
        assert slot is not None
        input_record = thaw_json(context.evidence_context)
        assert "campaign_contextual_history" in input_record
        self.input_contexts.append(context.evidence_context)
        self.lookups.append(
            (
                context.stage_request.step.generation,
                lane.lane_id,
                slot.slot_id,
            )
        )
        estimand_context = _object(
            {
                "schema_version": 1,
                "generation": context.stage_request.step.generation,
                "lane_id": lane.lane_id,
                "decision_slot_id": slot.slot_id,
            }
        )
        return CampaignPortfolioMemoryEstimandProjection(
            estimand_context=estimand_context,
            estimand_stratum_sha256=typed_json_sha256(estimand_context),
        )


class _ArchiveContextProjector:
    projector_id = "runtime_test_archive_context"
    projector_version = 1
    definition_sha256 = _sha("runtime-test-archive-context")

    def __init__(self) -> None:
        self.lookups = []

    def project(self, *, archive_utility, parent):
        self.lookups.append(
            (
                archive_utility.generation,
                archive_utility.snapshot_sha256,
                parent.candidate_id.value,
            )
        )
        return CampaignPortfolioArchiveContextProjection(
            projector_id=self.projector_id,
            projector_version=self.projector_version,
            definition_sha256=self.definition_sha256,
            archive_utility_snapshot_sha256=archive_utility.snapshot_sha256,
            parent_configuration_sha256=parent.occurrence.configuration_hash,
            payload=_object(
                {
                    "schema_version": 1,
                    "generation": archive_utility.generation,
                    "lower_is_better": True,
                }
            ),
        )


class _Reflection:
    def __init__(self) -> None:
        self.requests = []

    async def reflect(self, request, source_results):
        self.requests.append(request)
        return _object(
            {
                "source_generation": request.wave.source_generation,
                "source_wave_receipts": [
                    value.receipt.receipt_sha256 for value in source_results
                ],
                "insights": [
                    {
                        "claim": "Recombine non-overlapping successful branches.",
                        "status": "quarantined_unpromoted",
                    }
                ],
                "provider_calls": 0,
            }
        )


class _OutcomeUpdater:
    def __init__(self) -> None:
        self.generations = []

    async def prepare_update(self, request, waves, results, prior_memory):
        record = thaw_json(prior_memory)
        record["last_portfolio_generation"] = request.step.generation
        record["evaluated_wave_receipts"] = [
            result.receipt.receipt_sha256 for result in results
        ]
        record["parent_count"] = len(waves)
        return CampaignPortfolioOutcomePreparation(
            request_sha256=request.request_sha256,
            generation=request.step.generation,
            wave_request_sha256s=tuple(
                wave.selection_request.request_sha256 for wave in waves
            ),
            result_receipt_sha256s=tuple(
                result.receipt.receipt_sha256 for result in results
            ),
            prior_memory_sha256=typed_json_sha256(prior_memory),
            updated_memory=_object(record),
            evidence=_object({"prepared": True}),
        )

    def commit_update(self, preparation):
        self.generations.append(preparation.generation)

    def abort_update(self, preparation):
        del preparation


class _LearningLifecycle:
    def __init__(self) -> None:
        self.reflection_generations = []
        self.admission_generations = []
        self.portfolio_generations = []

    def reflection_completed(self, request, receipt, result):
        self.reflection_generations.append(request.wave.source_generation)
        return _object(
            {
                "reflection_receipt_sha256": receipt.receipt_sha256,
                "result_sha256": typed_json_sha256(result),
                "registered_quarantine": True,
            }
        )

    def reflections_admitted(self, request, contents):
        self.admission_generations.append(request.barrier.generation)
        return _object(
            {
                "campaign_admission_request_sha256": request.request_sha256,
                "content_count": len(contents),
                "diagnostic_only": True,
            }
        )

    async def prepare_portfolio_generation_close(
        self,
        request,
        waves,
        results,
        memory_credit_preparation,
    ):
        return CampaignPortfolioLearningPreparation(
            request_sha256=request.request_sha256,
            generation=request.step.generation,
            wave_request_sha256s=tuple(
                wave.selection_request.request_sha256 for wave in waves
            ),
            result_receipt_sha256s=tuple(
                result.receipt.receipt_sha256 for result in results
            ),
            memory_credit_preparation_sha256=(
                memory_credit_preparation.preparation_sha256
            ),
            evidence=_object(
                {
                    "generation": request.step.generation,
                    "wave_count": len(waves),
                    "result_count": len(results),
                    "memory_credit_batch": (
                        memory_credit_preparation.batch_receipt is not None
                    ),
                }
            ),
        )

    def commit_portfolio_generation_close(self, preparation):
        self.portfolio_generations.append(preparation.generation)

    def abort_portfolio_generation_close(self, preparation):
        del preparation


class _ArchiveUtility:
    utility_id = "campaign_test_archive_utility"
    utility_version = 1
    definition_sha256 = _sha("campaign-test-archive-utility")

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
            runtime_id="provider_free_campaign_runtime",
            runtime_version=1,
            definition_sha256=_sha("provider-free-campaign-runtime"),
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


def test_provider_free_g3_executes_real_portfolio_recombination_and_evaluator() -> None:
    evaluator = _DeterministicBoilsEvaluator()
    phenotype_identity = _CountingPhenotypeIdentityPolicy()
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(
            AbcEvaluatorSettings.current_circuit_panel(circuit_names=("log2",)),
            evaluator=evaluator,
        ),
        phenotype_identity=phenotype_identity,
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    ids = DeterministicIdFactory("campaign_bridge")
    memory = InsightMemoryBank(id_factory=ids)
    entry = memory.extend(
        (
            InsightDraft(
                claim="Test distinct action positions before recombination.",
                trigger="A finite path-disjoint portfolio is available.",
                mechanism="Independent positions can later be unioned exactly.",
                affected_paths=("$.sequence",),
                evidence_summary="Provider-free runtime integration prior.",
                confidence=0.5,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )[0]
    seed_card = PortfolioCard(
        card_key="card.runtime_prior",
        reference=entry.reference,
        content_sha256=entry.draft.content_sha256,
        evidence_sha256=_sha("runtime-prior-evidence"),
        prompt_payload=_object({"prior": "independent_positions"}),
        assigned_score=0.0,
    )
    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=DeterministicRandomFeasiblePortfolioPolicy(seed=20260716),
        seed=20260716,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=2,
        max_output_tokens=1,
        temperature=None,
    )
    evidence = _Evidence()
    first = list(DEFAULT_ACTION_SEQUENCE)
    second = list(PARENT_C_SEQUENCE)
    config = AgenticCampaignWorkloadConfig(
        workload_id="boils-campaign-runtime-test",
        workload_version=1,
        definition_sha256=_sha("boils-campaign-runtime-test"),
        benchmark=benchmark,
        seeds=(
            CampaignSeed("seed_default", _object({"sequence": first})),
            CampaignSeed("seed_parent_c", _object({"sequence": second})),
        ),
        finite_catalog_id=FINITE_CATALOG_ID,
        evaluator_concurrency_cap=2,
        evaluator_preflight_receipt=_object({"qualified": True}),
        resource_lease_receipt=_object({"lease": "test_cpu_pool"}),
        evidence=AgenticCampaignEvidenceProjections(
            projection_id="campaign_runtime_test_evidence",
            projection_version=1,
            definition_sha256=_sha("campaign-runtime-test-evidence"),
            initialize_memory=evidence.initialize_memory,
            context=evidence.context,
            cards=evidence.cards,
        ),
    )
    parent_selector = ArchiveReservoirCampaignParentSelector(reservoir_limit=8)
    wave_factory = _WaveFactory(composition=composition, card=seed_card)
    context_enricher = _ContextEnricher(
        ContextualOutcomeCampaignEnricher(
            ledger=PortfolioOutcomeFeedbackLedger(),
            max_actions=8,
        )
    )
    memory_estimand_projector = _MemoryEstimandProjector()
    archive_context_projector = _ArchiveContextProjector()
    reflection = _Reflection()
    outcome_updater = _OutcomeUpdater()
    learning_lifecycle = _LearningLifecycle()
    policies = CampaignPolicies(
        cadence=AlternatingPortfolioRecombinationCadence(),
        parent_selection=_binding("archive_reservoir", parent_selector),
        memory_assignment=_binding("runtime_evidence", evidence),
        portfolio_selection=_binding(
            "provider_free_random",
            composition.portfolio.selector,
        ),
        recombination=_binding("disjoint_patch_union", object()),
        reflection=_binding("provider_free_quarantine", reflection),
        archive_utility=_ArchiveUtility(),
    )
    protocol = CampaignProtocol(
        protocol_id="provider_free_runtime_g3",
        protocol_version=1,
        definition_sha256=_sha("provider-free-runtime-g3"),
        outer_seed=20260716,
        generation_count=3,
        required_seed_count=2,
        parents_per_portfolio_generation=1,
        portfolio_width=2,
        recombinations_per_parent=1,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=1,
    )
    budget = OptimizerBudget(
        max_unique_evaluations=7,
        max_logical_llm_calls=3,
        max_generations=3,
    )
    workload_ports = config.build_ports()
    prepared = EvolutionCampaign(
        protocol=protocol,
        workload=workload_ports,
        policies=policies,
        runtime=_PreparationRuntime(),
        budget=budget,
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
        task_sha256=_sha("provider-free-boils-runtime-task"),
        archive_context_projector=archive_context_projector,
        context_enricher=context_enricher,
        memory_estimand_projector=memory_estimand_projector,
        learning_lifecycle=learning_lifecycle,
        reflection_executor=reflection,
        outcome_updater=outcome_updater,
    )
    journal = _ExecutionJournal()
    result = asyncio.run(
        EvolutionCampaignScheduler(
            prepared=prepared,
            policies=policies,
            stages=runtime,
            reflections=runtime,
            lifecycle=runtime,
            journal=journal,
        ).run()
    )

    assert result.counters.generations_completed == 3
    assert result.counters.candidate_occurrences == 7
    assert result.counters.unique_evaluations == evaluator.calls == 7
    assert result.counters.logical_agent_calls == 3
    assert tuple(
        value.candidate_occurrence_count for value in result.stage_receipts
    ) == (
        2,
        1,
        2,
    )
    assert len(runtime.history) == 7
    assert runtime.final_front
    assert len(wave_factory.contexts) == 2
    assert len(context_enricher.lookups) == 2
    assert len(memory_estimand_projector.lookups) == 2
    assert len(archive_context_projector.lookups) == 2
    assert tuple(value[0] for value in memory_estimand_projector.lookups) == (1, 3)
    assert tuple(value[0] for value in context_enricher.lookups) == (1, 3)
    assert all(value[1] == "reservoir_0001" for value in context_enricher.lookups)
    assert all(
        value[2] == "reservoir_0001.primary" for value in context_enricher.lookups
    )
    contextual_histories = tuple(
        thaw_json(context.evidence_context)["campaign_contextual_history"]
        for context in wave_factory.contexts
    )
    projected_contexts = tuple(
        thaw_json(context.evidence_context) for context in wave_factory.contexts
    )
    assert tuple(
        projected[CAMPAIGN_ARCHIVE_CONTEXT_KEY]["payload"]["generation"]
        for projected in projected_contexts
    ) == (1, 3)
    assert all(
        projected["objective_ids"] == ["total_levels", "total_lut_count"]
        and "parent_sha256" in projected
        and "finite_contract_sha256" in projected
        and "memory_sha256" in projected
        for projected in projected_contexts
    )
    for projected in projected_contexts:
        estimand_context = projected[MEMORY_ESTIMAND_CONTEXT_KEY]
        frozen_estimand_context = freeze_json(estimand_context)
        assert type(frozen_estimand_context) is FrozenJsonObject
        assert projected[MEMORY_ESTIMAND_STRATUM_SHA256_KEY] == typed_json_sha256(
            frozen_estimand_context
        )
    for source, projected in zip(
        memory_estimand_projector.input_contexts,
        projected_contexts,
        strict=True,
    ):
        without_projection = dict(projected)
        without_projection.pop(MEMORY_ESTIMAND_CONTEXT_KEY)
        without_projection.pop(MEMORY_ESTIMAND_STRATUM_SHA256_KEY)
        assert without_projection == thaw_json(source)
    assert tuple(
        value["cutoff_wave_index_exclusive"] for value in contextual_histories
    ) == (1, 3)
    assert all(value["actions"] == [] for value in contextual_histories)
    assert all(
        value["epistemic_status"]
        == "observational_predictive_history_not_causal_credit"
        for value in contextual_histories
    )
    assert all(
        value["runtime_test_binding"]
        == {
            "retrieval_scope": "parent_lane_decision_slot",
            "lane_id": "reservoir_0001",
            "slot_id": "reservoir_0001.primary",
        }
        for value in contextual_histories
    )
    assert all(
        context.parent_lane is not None
        and context.parent_lane.lane_id == "reservoir_0001"
        and context.decision_slot is not None
        and context.decision_slot.slot_id == "reservoir_0001.primary"
        for context in wave_factory.contexts
    )
    assert len(reflection.requests) == 1
    assert learning_lifecycle.reflection_generations == [2]
    assert learning_lifecycle.admission_generations == [2]
    assert learning_lifecycle.portfolio_generations == [1, 3]
    assert len(wave_factory.contexts[1].test_eligible_reflections) == 1
    assert outcome_updater.generations == [1, 3]
    assert typed_json_sha256(wave_factory.contexts[0].memory) != typed_json_sha256(
        wave_factory.contexts[1].memory
    )
    assert result.tail_drain_receipt is None
    assert len(result.test_admission_receipts) == 1
    assert result.test_admission_receipts[0].lifecycle_promoted is False
    assert len(result.stage_receipts[0].selector_audits) == 1
    audit = thaw_json(result.stage_receipts[0].selector_audits[0].plaintext_audit)
    first_wave = runtime._portfolio_waves[1][0][0]
    first_build = wave_factory.contexts[0]
    assert first_wave.selection_request.context == first_build.evidence_context
    extension = CampaignSelectorContextExtension(
        extension_id="runtime_test",
        extension_version=1,
        definition_sha256=_sha("runtime-test-selector-extension"),
        payload=_object({"advisory": {"forced_action_allowed": False}}),
    )
    extended_wave = replace(
        first_wave,
        selection_request=replace(
            first_wave.selection_request,
            context=attach_campaign_selector_context_extension(
                first_build.evidence_context,
                extension,
            ),
        ),
    )
    runtime._validate_wave(wave=extended_wave, build=first_build)
    assert audit["request_text"] == render_portfolio_selection_prompt(
        first_wave.selection_request
    )
    assert "RANKED PORTFOLIO SELECTION CONTRACT" in audit["request_text"]
    assert "design_rationale" in audit["response_text"]
    assert any(
        "recombination_wave_receipts" in thaw_json(value.result)
        for value in result.stage_receipts
    )
    first_stage_record = thaw_json(result.stage_receipts[0].result)
    assert first_stage_record["parent_lanes"][0]["lane_id"] == "reservoir_0001"
    assert first_stage_record["decision_slots"][0]["slot_id"] == (
        "reservoir_0001.primary"
    )
    assert first_stage_record["context_enrichment_applied"] is True
    assert first_stage_record["archive_context_projection"] == {
        "reserved_key": CAMPAIGN_ARCHIVE_CONTEXT_KEY,
        "projector_id": archive_context_projector.projector_id,
        "projector_version": archive_context_projector.projector_version,
        "definition_sha256": archive_context_projector.definition_sha256,
    }
    for preparation, context in zip(
        runtime.wave_preparation_receipts,
        wave_factory.contexts,
        strict=True,
    ):
        projection_identity = thaw_json(preparation.context_projection_identity)
        archive_projection = dict(context.evidence_context.items)[
            CAMPAIGN_ARCHIVE_CONTEXT_KEY
        ]
        assert projection_identity["archive_context_projection"] == {
            "reserved_key": CAMPAIGN_ARCHIVE_CONTEXT_KEY,
            "projection_sha256": typed_json_sha256(archive_projection),
        }
    assert first_stage_record["memory_credit_batch"] is None
    assert result.cleanup_receipt.released is True
    assert thaw_json(result.cleanup_receipt.evidence)["resource_cleanup"] == {
        "adapter_owned_resource_count": 0,
        "external_resource_close_required_by_caller": True,
        "ownership": "external_to_adapter",
    }
    assert journal.events[-1].kind.value == "runtime_cleaned"

    # The second portfolio stage has already revisited the earlier history;
    # only terminal-generation configurations remain absent from the cache.
    history_sha256s = {
        candidate.occurrence.configuration_hash for candidate in runtime.history
    }
    preterminal_sha256s = {
        candidate.occurrence.configuration_hash
        for candidate in runtime.history
        if candidate.generation < 3
    }
    assert len(preterminal_sha256s) == 5
    assert len(history_sha256s) == 7
    assert set(runtime._phenotype_identity_cache) == preterminal_sha256s
    calls_before = phenotype_identity.calls
    expected_new = len(history_sha256s - preterminal_sha256s)
    assert expected_new == 2
    first_known = runtime._known_phenotypes()
    assert phenotype_identity.calls - calls_before == expected_new
    calls_after_first = phenotype_identity.calls
    assert runtime._known_phenotypes() == first_known
    assert phenotype_identity.calls == calls_after_first
    assert set(runtime._phenotype_identity_cache) == history_sha256s

    tampered_context = thaw_json(first_build.evidence_context)
    tampered_context[MEMORY_ESTIMAND_STRATUM_SHA256_KEY] = _sha("tampered-stratum")
    tampered_wave = replace(
        first_wave,
        selection_request=replace(
            first_wave.selection_request,
            context=_object(tampered_context),
        ),
    )
    try:
        runtime._validate_wave(wave=tampered_wave, build=first_build)
    except ValueError as error:
        assert "escaped its trusted campaign inputs" in str(error)
    else:  # pragma: no cover - fail-closed contract.
        raise AssertionError("wave factory was allowed to alter trusted enrichment")

    class _ForeignProjector:
        def project(self, context):
            del context
            return _object({"estimand_context": {}})

    runtime.memory_estimand_projector = _ForeignProjector()
    try:
        runtime._project_memory_estimand(first_build)
    except TypeError as error:
        assert "must return an exact projection or None" in str(error)
    else:  # pragma: no cover - fail-closed contract.
        raise AssertionError("runtime accepted a foreign estimand projection")

    forged_projection = object.__new__(CampaignPortfolioMemoryEstimandProjection)
    object.__setattr__(forged_projection, "estimand_context", {"not": "frozen"})
    object.__setattr__(
        forged_projection,
        "estimand_stratum_sha256",
        _sha("forged-estimand"),
    )

    class _ForgedProjector:
        def project(self, context):
            del context
            return forged_projection

    runtime.memory_estimand_projector = _ForgedProjector()
    try:
        runtime._project_memory_estimand(first_build)
    except TypeError as error:
        assert "estimand_context must be an exact frozen object" in str(error)
    else:  # pragma: no cover - fail-closed contract.
        raise AssertionError("runtime accepted a non-frozen estimand projection")

    try:
        CampaignPortfolioMemoryEstimandProjection(
            estimand_context=_object({"schema_version": 1}),
            estimand_stratum_sha256=_sha("different-estimand"),
        )
    except ValueError as error:
        assert "must identify the exact estimand context" in str(error)
    else:  # pragma: no cover - fail-closed contract.
        raise AssertionError("projection accepted a mismatched estimand digest")

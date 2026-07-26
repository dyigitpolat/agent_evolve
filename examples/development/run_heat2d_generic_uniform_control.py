#!/usr/bin/env python3
"""Prepare or run the matched provider-free Heat2D G6 uniform control.

The workload, seeds, evaluator, cadence, elite/explorer parent policy,
semantic novelty boundary, recombination materializer, archive-aware pair
utility, and affine archive analysis match ``run_heat2d_generic_campaign``.
The treatment is replaced by an outcome-blind conditional-uniform portfolio
policy, frozen unused memory, and a zero-provider inert reflection receipt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.agentic import (  # noqa: E402
    DeterministicIdFactory,
    InsightMemoryBank,
    LLMCallId,
    PortfolioCard,
    PortfolioSelectionRequest,
    PortfolioVariationWaveRequest,
    compose_portfolio_evolution,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.application.budgeted_optimizer import OptimizerBudget  # noqa: E402
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignArchiveCutoffReceipt,
    CampaignExecutionEvent,
    CampaignJournalAck,
    CampaignStageRequest,
    EvolutionCampaignScheduler,
)
from agent_evolve.application.evolution_campaign import (  # noqa: E402
    CampaignAgentRuntimeReceipt,
    CampaignConcurrency,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    CampaignReflectionSupervisionPolicy,
    CampaignSeed,
    CampaignWorkloadPorts,
    EvolutionCampaign,
    PreparedEvolutionCampaign,
    ReflectionFailureMode,
    SealedCutoffDelayedAdmissionCadence,
    TerminalReflectionPolicy,
)
from agent_evolve.application.evaluation_accounting import (  # noqa: E402
    CampaignEvaluationAccounting,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    AgenticPortfolioCampaignRuntime,
    ArchiveEliteExplorerCampaignParentSelector,
    CampaignPortfolioWaveContext,
)
from agent_evolve.application.parent_measurement import (  # noqa: E402
    attach_parent_measurement_to_context,
    bind_parent_measurement,
)
from agent_evolve.campaign_workload import (  # noqa: E402
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, thaw_json  # noqa: E402
from agent_evolve.policies.reward.affine_hypervolume import (  # noqa: E402
    AffineHypervolumeArchiveUtility,
)
from agent_evolve.ports.parent_measurement import (  # noqa: E402
    ParentMeasurementProjection,
)
from examples.benchmarks.heat2d_constructive.artifact_boundary import (  # noqa: E402
    artifact_scripts_dir,
)
from examples.benchmarks.heat2d_constructive.candidate import (  # noqa: E402
    seed_layouts,
)
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (  # noqa: E402
    CATALOG_ID,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (  # noqa: E402
    FORMULATION_DEFINITION_SHA256,
    WORKLOAD_ID,
)
from examples.benchmarks.heat2d_constructive.problem_def import (  # noqa: E402
    DirectV3Evaluator,
)
from examples.development import run_heat2d_generic_campaign as agentic  # noqa: E402
from examples.development.durable_run_artifacts import (  # noqa: E402
    DurableJsonlJournal,
    finalize_run_directory,
    source_identity,
    write_json_atomic,
)
from examples.development.uniform_feasible_portfolio_control import (  # noqa: E402
    MAX_REJECTION_DRAWS,
    POLICY_DEFINITION_SHA256,
    TaskKeyedConditionalUniformPortfolioPolicy,
    analyze_grouped_feasible_slate_space,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/engibench_heat2d/generic_campaign_control"
)
PROTOCOL_ID = agentic.PROTOCOL_ID
# Reuse the treatment's generic environment-resolved seed boundary.  A shared
# control is model-free, but it must still be paired to the exact workload
# replicate selected by the systematic study.
CONTROL_REPLICATE_SEED = agentic.OUTER_SEED
CONTROL_ID_NAMESPACE = "heat_calibrated_g6_control_r1"
MINIMUM_ACCEPTANCE_PROBABILITY = Fraction(1, 100)
TARGET_UNIQUE_EVALUATIONS = agentic.PLANNED_UNIQUE_EVALUATIONS
SCHEDULER_LOCAL_DECISION_OPERATIONS = agentic.PLANNED_LOGICAL_LLM_CALLS
ACTUAL_LLM_CALLS = 0
PROVIDER_CALLS = 0
MAX_PDE_WALL_S = 45.0
MAX_PDE_PEAK_RSS_BYTES = 3 * 1024**3


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expected_stage_widths() -> list[int]:
    """Derive the matched control budget from the treatment scale contract."""

    return [
        agentic.PARENTS_PER_PORTFOLIO
        * (
            agentic.PORTFOLIO_WIDTH
            if generation in agentic.PORTFOLIO_GENERATIONS
            else agentic.RECOMBINATIONS_PER_PARENT
        )
        for generation in range(1, agentic.GENERATION_COUNT + 1)
    ]


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:
        raise TypeError("expected a frozen typed-JSON object")
    return result


def _source_paths() -> tuple[Path, ...]:
    scripts = artifact_scripts_dir()
    numeric = tuple(
        scripts / name
        for name in (
            "calibration_harness.py",
            "heat2d_constructive_layout.py",
            "heat2d_integrity_adapter.py",
            "heat2d_exact_volume_contract.py",
            "run_heat2d_direct_v1.py",
            "heat2d_direct_v1_container.py",
            "run_heat2d_direct_v3.py",
            "heat2d_direct_v3_container.py",
        )
    )
    core = tuple(sorted((AGENT_EVOLVE_ROOT / "src/agent_evolve").rglob("*.py")))
    heat = tuple(
        sorted(
            (AGENT_EVOLVE_ROOT / "examples/benchmarks/heat2d_constructive").glob("*.py")
        )
    )
    return (
        Path(__file__),
        Path(agentic.__file__).resolve(),
        AGENT_EVOLVE_ROOT / "examples/development/heat2d_campaign_reflection.py",
        AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
        AGENT_EVOLVE_ROOT
        / "examples/development/uniform_feasible_portfolio_control.py",
        AGENT_EVOLVE_ROOT / "pyproject.toml",
        AGENT_EVOLVE_ROOT / "uv.lock",
        *core,
        *heat,
        *numeric,
    )


def _snapshot_sources(
    run_dir: Path,
    paths: tuple[Path, ...],
) -> dict[str, object]:
    """Use the treatment runner's exact source-byte snapshot boundary."""

    return agentic._snapshot_sources(run_dir, paths)


@dataclass(frozen=True, slots=True)
class _ControlPreparationRuntime:
    source_closure_sha256: str

    def prepare(self, request):
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="provider_free_uniform_control_runtime",
            runtime_version=1,
            definition_sha256=_sha("provider-free-uniform-control-runtime-v1"),
            accepted=True,
            evidence=_object(
                {
                    "provider_calls": 0,
                    "actual_llm_calls": 0,
                    "credential_read": False,
                    "source_closure_sha256": self.source_closure_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _PolicyTag:
    name: str


def _control_binding(name: str, implementation: object) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"heat2d-generic-uniform-control-policy:{name}:v1"),
    )


class _PreparationJournal:
    def __init__(self, journal: DurableJsonlJournal) -> None:
        self.journal = journal

    def append(self, record) -> None:
        self.journal.append(thaw_json(record))


@dataclass(slots=True)
class _TimedExecutionJournal:
    journal: DurableJsonlJournal
    execution_started_ns: int

    async def append(self, event: CampaignExecutionEvent):
        self.journal.append(
            {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - self.execution_started_ns
                    ),
                    "observed_at_utc": _utc_now(),
                },
                "authenticated_record": event.to_record(),
            }
        )
        return CampaignJournalAck(event.event_sha256, True)


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - materialized only.
        raise AssertionError(f"uniform control invoked propose: {request}")

    async def reflect(self, request):  # pragma: no cover - local executor.
        raise AssertionError(f"uniform control invoked generator reflection: {request}")


def _portfolio_cards(
    memory: InsightMemoryBank,
    evidence_cards: tuple[FrozenJsonObject, ...],
) -> tuple[PortfolioCard, ...]:
    entry_by_reference = {
        (entry.reference.insight_id.value, entry.reference.version): entry
        for entry in memory.entries
    }
    cards: list[PortfolioCard] = []
    for ordinal, payload in enumerate(evidence_cards, start=1):
        record = thaw_json(payload)
        reference_key = (record["insight_id"], record["version"])
        entry = entry_by_reference[reference_key]
        cards.append(
            PortfolioCard(
                card_key=f"card.control.{ordinal:02d}",
                reference=entry.reference,
                content_sha256=entry.draft.content_sha256,
                evidence_sha256=typed_json_sha256(payload),
                prompt_payload=payload,
                assigned_score=0.0,
            )
        )
    return tuple(cards)


@dataclass(slots=True)
class _ControlWaveFactory:
    ids: DeterministicIdFactory
    memory: InsightMemoryBank
    utility: AffineHypervolumeArchiveUtility
    records: list[dict[str, object]]
    _analysis_by_space: dict[tuple[str, int, int | None, bool], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def build(self, context: CampaignPortfolioWaveContext):
        generation = context.stage_request.step.generation
        selection = PortfolioSelectionRequest(
            call_id=self.ids.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Apply the preregistered outcome-blind conditional-uniform control "
                "to select four sealed one-locus options covering all three option "
                "families with pairwise disjoint parent-relative patch paths."
            ),
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=_portfolio_cards(self.memory, context.evidence_cards),
            portfolio_size=agentic.PORTFOLIO_WIDTH,
            required_metric_ids=agentic.OBJECTIVE_IDS,
            min_distinct_families=3,
            require_supporting_cards=False,
            require_pairwise_disjoint_parent_patches=True,
            max_output_tokens=1,
            temperature=None,
        )
        analysis_key = (
            selection.finite_variation_contract.identity_sha256,
            selection.portfolio_size,
            selection.min_distinct_families,
            selection.require_pairwise_disjoint_parent_patches,
        )
        analysis = self._analysis_by_space.get(analysis_key)
        analysis_cache_hit = analysis is not None
        if analysis is None:
            analysis = analyze_grouped_feasible_slate_space(selection)
            self._analysis_by_space[analysis_key] = analysis
        probability = analysis.acceptance_probability
        if analysis.feasible_unordered_slate_count <= 0:
            raise RuntimeError("eligible finite space has no feasible control slate")
        if probability < MINIMUM_ACCEPTANCE_PROBABILITY:
            raise RuntimeError(
                "eligible finite space violates the preregistered one-percent "
                "conditional-uniform acceptance gate"
            )
        snapshot = self.utility.require_snapshot(context.stage_request.archive_utility)
        self.records.append(
            {
                "generation": generation,
                "parent_slot": context.parent_slot,
                "parent_candidate_id": context.parent.candidate_id.value,
                "selection_request_sha256": selection.request_sha256,
                "finite_contract_identity_sha256": (
                    selection.finite_variation_contract.identity_sha256
                ),
                "feasible_slate_space": analysis.to_record(),
                "feasible_slate_analysis_cache_hit": analysis_cache_hit,
                "parent_measurement_binding_sha256": (
                    None
                    if context.parent_measurement is None
                    else context.parent_measurement.binding_sha256
                ),
                "outcome_fields_read_by_policy": [],
                "memory_or_reflection_fields_read_by_policy": [],
                "parent_measurement_fields_read_by_policy": [],
                "affine_snapshot": snapshot.to_record(),
            }
        )
        return PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=generation,
            label_prefix=(
                f"heatcontrolg{generation:02d}p{context.parent_slot + 1:02d}"
            ),
            phase="generic_heat2d_uniform_control",
            memory_credit=None,
        )


@dataclass(slots=True)
class _InertReflection:
    records: list[dict[str, object]]

    async def reflect(self, request, source_results):
        record = {
            "source_generation": request.wave.source_generation,
            "source_wave_count": len(source_results),
            "mode": "provider_free_inert_control_receipt",
            "outcome_content_inspected": False,
            "insights": [],
            "quarantined": True,
            "lifecycle_promoted": False,
            "provider_calls": 0,
            "actual_llm_calls": 0,
        }
        self.records.append(record)
        return _object(record)


@dataclass(frozen=True, slots=True)
class _OwnedLocalResources:
    async def close(self):
        return _object(
            {
                "ownership": "campaign_runtime",
                "provider_queue_owned": False,
                "provider_queue_closed": True,
                "credential_read": False,
            }
        )


@dataclass(frozen=True, slots=True)
class _Bundle:
    benchmark: Any
    config: AgenticCampaignWorkloadConfig
    workload_ports: CampaignWorkloadPorts
    policies: CampaignPolicies
    prepared: PreparedEvolutionCampaign
    memory: InsightMemoryBank
    memory_plan: Any
    utility: AffineHypervolumeArchiveUtility
    ids: DeterministicIdFactory
    parent_measurement_projection: ParentMeasurementProjection


def _prepare_bundle(
    *,
    run_dir: Path,
    preparation_journal: DurableJsonlJournal,
    source_closure_sha256: str,
) -> _Bundle:
    settings = agentic._evaluator_settings(run_dir / "pde")
    benchmark = agentic._scientific_benchmark(settings)
    preflight = benchmark.problem.preflight()
    ids = DeterministicIdFactory(CONTROL_ID_NAMESPACE)
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=Fraction(1, 1),
    )
    agentic._seed_memory(memory)
    memory_plan = agentic._memory_plan(memory)
    evidence = agentic._Evidence(memory, memory_plan)
    seeds = tuple(
        CampaignSeed(
            f"seed_{ordinal}",
            _object(value.model_dump(mode="python")),
        )
        for ordinal, value in enumerate(seed_layouts(), start=1)
    )
    config = AgenticCampaignWorkloadConfig(
        workload_id="heat2d_pareto_v1_generic_campaign",
        workload_version=1,
        definition_sha256=FORMULATION_DEFINITION_SHA256,
        benchmark=benchmark,
        seeds=seeds,
        finite_catalog_id=CATALOG_ID,
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=_object(preflight),
        resource_lease_receipt=_object(
            {
                "cpu_set": "8",
                "external_concurrency": 1,
                "lease_scope": "serialized_direct_v3_campaign",
            }
        ),
        evidence=AgenticCampaignEvidenceProjections(
            projection_id="heat2d_generic_campaign_evidence",
            projection_version=1,
            definition_sha256=_sha("heat2d-generic-campaign-evidence-v1"),
            initialize_memory=evidence.initialize_memory,
            context=evidence.context,
            cards=evidence.cards,
        ),
    )
    workload_ports = config.build_ports()
    utility = AffineHypervolumeArchiveUtility(agentic._affine_spec())
    parent_selector = ArchiveEliteExplorerCampaignParentSelector()
    policy = TaskKeyedConditionalUniformPortfolioPolicy(
        task_sha256=agentic.TASK_SHA256,
        replicate_seed=CONTROL_REPLICATE_SEED,
    )
    policies = CampaignPolicies(
        cadence=SealedCutoffDelayedAdmissionCadence(),
        parent_selection=agentic._binding("archive_elite_explorer", parent_selector),
        memory_assignment=_control_binding("frozen_ignored_memory", memory_plan),
        portfolio_selection=_control_binding("conditional_uniform_feasible", policy),
        recombination=agentic._binding(
            "archive_aware_disjoint_union",
            _PolicyTag("recombination"),
        ),
        reflection=_control_binding(
            "provider_free_inert_quarantine",
            _PolicyTag("reflection"),
        ),
        reflection_supervision=CampaignReflectionSupervisionPolicy(
            ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY
        ),
        archive_utility=utility,
    )
    protocol = CampaignProtocol(
        protocol_id=PROTOCOL_ID,
        protocol_version=1,
        definition_sha256=_sha("heat2d-generic-calibrated-g6-v1"),
        outer_seed=agentic.OUTER_SEED,
        generation_count=agentic.GENERATION_COUNT,
        required_seed_count=2,
        parents_per_portfolio_generation=agentic.PARENTS_PER_PORTFOLIO,
        portfolio_width=agentic.PORTFOLIO_WIDTH,
        recombinations_per_parent=agentic.RECOMBINATIONS_PER_PARENT,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=1,
        terminal_reflection_policy=(
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ),
    )
    prepared = EvolutionCampaign(
        protocol=protocol,
        workload=workload_ports,
        policies=policies,
        runtime=_ControlPreparationRuntime(source_closure_sha256),
        budget=OptimizerBudget(
            max_unique_evaluations=TARGET_UNIQUE_EVALUATIONS,
            max_logical_llm_calls=SCHEDULER_LOCAL_DECISION_OPERATIONS,
            max_generations=agentic.GENERATION_COUNT,
        ),
        concurrency=CampaignConcurrency(
            evaluator_concurrency=agentic.EVALUATOR_CONCURRENCY,
            agent_concurrency=agentic.AGENT_CONCURRENCY,
            agent_queue_capacity=agentic.AGENT_QUEUE_CAPACITY,
        ),
        journals=(_PreparationJournal(preparation_journal),),
    ).prepare()
    parent_measurement_projection = agentic._parent_measurement_projection(
        prepared,
        benchmark,
    )
    return _Bundle(
        benchmark,
        config,
        workload_ports,
        policies,
        prepared,
        memory,
        memory_plan,
        utility,
        ids,
        parent_measurement_projection,
    )


def _eligibility_probe(bundle: _Bundle) -> dict[str, object]:
    session = bundle.prepared.benchmark_session
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    known = tuple(
        sorted(
            bundle.benchmark.phenotype_identity.identify(
                thaw_json(seed.configuration)
            ).value_sha256
            for seed in bundle.prepared.seeds.seeds
        )
    )
    known_cpu_s = time.process_time() - cpu_started
    known_wall_s = time.perf_counter() - wall_started
    memory = bundle.workload_ports.evidence.initialize_memory(
        session,
        bundle.prepared.seeds,
    )
    rows: list[dict[str, object]] = []
    for ordinal, seed in enumerate(bundle.prepared.seeds.seeds, start=1):
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        variation = bundle.workload_ports.catalog.bind(
            session.benchmark,
            seed.configuration,
            known,
        )
        first_bind_cpu_s = time.process_time() - cpu_started
        first_bind_wall_s = time.perf_counter() - wall_started
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        context = bundle.workload_ports.evidence.context(
            session,
            seed.configuration,
            variation,
            memory,
        )
        cards = bundle.workload_ports.evidence.cards(
            session,
            seed.configuration,
            variation,
            memory,
        )
        evidence_cpu_s = time.process_time() - cpu_started
        evidence_wall_s = time.perf_counter() - wall_started
        receipt = variation.eligibility_receipt
        assert receipt is not None
        analysis_request = PortfolioSelectionRequest(
            call_id=LLMCallId(f"call_control_readiness_seed_{ordinal}"),
            operation="select_portfolio",
            instruction="Provider-free exact control-space readiness analysis.",
            context=context,
            finite_variation_contract=variation.contract,
            cards=_portfolio_cards(bundle.memory, cards),
            portfolio_size=agentic.PORTFOLIO_WIDTH,
            required_metric_ids=agentic.OBJECTIVE_IDS,
            min_distinct_families=3,
            require_supporting_cards=False,
            require_pairwise_disjoint_parent_patches=True,
            max_output_tokens=1,
            temperature=None,
        )
        slate_analysis = analyze_grouped_feasible_slate_space(analysis_request)
        rows.append(
            {
                "seed_id": seed.seed_id,
                "first_bind_wall_s": first_bind_wall_s,
                "first_bind_process_cpu_s": first_bind_cpu_s,
                "context_cards_wall_s": evidence_wall_s,
                "context_cards_process_cpu_s": evidence_cpu_s,
                "raw_option_count": len(receipt.option_phenotypes),
                "eligible_option_count": len(receipt.eligible_option_ids),
                "known_excluded_option_count": len(receipt.known_excluded_option_ids),
                "semantic_alias_count": len(receipt.alias_excluded_option_ids),
                "context_sha256": typed_json_sha256(context),
                "card_sha256s": [typed_json_sha256(value) for value in cards],
                "feasible_slate_space": slate_analysis.to_record(),
                "acceptance_gate_pass": (
                    slate_analysis.acceptance_probability
                    >= MINIMUM_ACCEPTANCE_PROBABILITY
                ),
            }
        )
    return {
        "status": "provider_and_pde_free_semantic_readiness",
        "provider_calls": 0,
        "actual_llm_calls": 0,
        "pde_solves": 0,
        "resolution": 1001,
        "known_seed_identity_wall_s": known_wall_s,
        "known_seed_identity_process_cpu_s": known_cpu_s,
        "parents": rows,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gate_under_60_process_cpu_s_each": all(
            row["first_bind_process_cpu_s"] < 60 for row in rows
        ),
        "wall_under_60_s_each_observed": all(
            row["first_bind_wall_s"] < 60 for row in rows
        ),
        "all_acceptance_gates_pass": all(row["acceptance_gate_pass"] for row in rows),
    }


async def _control_all_wave_probe(bundle: _Bundle) -> dict[str, object]:
    """Select every G6 control slate without a provider call or PDE solve.

    The probe constructs the same six portfolio requests that the alternating
    schedule requires and runs the real conditional-uniform policy.  Synthetic
    parent measurements are contract-construction evidence only; no reported
    scientific objective originates here.
    """

    session = bundle.prepared.benchmark_session
    parents = tuple(
        agentic._construction_parent(
            ordinal=ordinal,
            configuration=seed.configuration,
            benchmark=bundle.benchmark,
        )
        for ordinal, seed in enumerate(bundle.prepared.seeds.seeds, start=1)
    )
    known = tuple(
        sorted(
            bundle.benchmark.phenotype_identity.identify(
                thaw_json(parent.configuration)
            ).value_sha256
            for parent in parents
        )
    )
    memory_projection = bundle.workload_ports.evidence.initialize_memory(
        session,
        bundle.prepared.seeds,
    )
    archive = _object(
        {
            "front_candidates": [
                {
                    "objectives": [
                        {"metric_id": name, "value_hex": value.hex()}
                        for name, value in parent.objectives
                    ]
                }
                for parent in parents
            ],
            "preparation_only": True,
        }
    )
    records: list[dict[str, object]] = []
    policy = TaskKeyedConditionalUniformPortfolioPolicy(
        task_sha256=agentic.TASK_SHA256,
        replicate_seed=CONTROL_REPLICATE_SEED,
    )
    factory = _ControlWaveFactory(
        ids=DeterministicIdFactory(f"{CONTROL_ID_NAMESPACE}_prepare_probe"),
        memory=bundle.memory,
        utility=bundle.utility,
        records=records,
    )
    rows: list[dict[str, object]] = []
    step_by_generation = {
        step.generation: step for step in bundle.prepared.schedule.steps
    }
    for generation in agentic.PORTFOLIO_GENERATIONS:
        step = step_by_generation[generation]
        utility = bundle.utility.freeze(
            benchmark=session.benchmark,
            generation=generation,
            archive=archive,
        )
        cutoff = CampaignArchiveCutoffReceipt(
            request_sha256=_sha(f"heat-control-prepare-cutoff:{generation}"),
            preparation_sha256=bundle.prepared.preparation_sha256,
            generation=generation,
            archive=archive,
            evidence=_object({"preparation_only": True, "generation": generation}),
        )
        stage_request = CampaignStageRequest(
            preparation_sha256=bundle.prepared.preparation_sha256,
            runtime_start_receipt_sha256=_sha("heat-control-prepare-runtime-start"),
            step=step,
            archive_cutoff=cutoff,
            archive_utility=utility,
            source_portfolio=None,
            test_eligible_reflection_receipt_sha256s=(),
            prior_selector_audit_set_sha256=_sha(
                f"heat-control-prepare-prior-audit:{generation}"
            ),
        )
        for parent_slot, parent in enumerate(parents):
            variation = bundle.workload_ports.catalog.bind(
                session.benchmark,
                parent.configuration,
                known,
            )
            evidence_context = bundle.workload_ports.evidence.context(
                session,
                parent.configuration,
                variation,
                memory_projection,
            )
            parent_measurement = bind_parent_measurement(
                candidate=parent,
                variation=variation,
                projection=bundle.parent_measurement_projection,
            )
            evidence_context = attach_parent_measurement_to_context(
                evidence_context,
                parent_measurement,
            )
            evidence_cards = bundle.workload_ports.evidence.cards(
                session,
                parent.configuration,
                variation,
                memory_projection,
            )
            wave = factory.build(
                CampaignPortfolioWaveContext(
                    prepared=bundle.prepared,
                    stage_request=stage_request,
                    parent_slot=parent_slot,
                    parent=parent,
                    variation=variation,
                    evidence_context=evidence_context,
                    evidence_cards=evidence_cards,
                    memory=memory_projection,
                    parent_measurement=parent_measurement,
                )
            )
            selected = await policy.select(wave.selection_request)
            rows.append(
                {
                    "generation": generation,
                    "lane_id": ("elite", "explorer")[parent_slot],
                    "request_sha256": wave.selection_request.request_sha256,
                    "decision_sha256": selected.decision.decision_sha256,
                    "parent_measurement_binding_sha256": (
                        parent_measurement.binding_sha256
                    ),
                    "selected_option_ids": [
                        member.option_id for member in selected.decision.members
                    ],
                    "selected_family_count": len(
                        {member.family for member in selected.decision.members}
                    ),
                    "evaluation_width": len(selected.decision.members),
                    "eligible_option_count": len(
                        wave.selection_request.finite_variation_contract.options
                    ),
                    "resolved_provider": selected.telemetry.resolved_provider,
                    "input_tokens": selected.telemetry.input_tokens,
                    "output_tokens": selected.telemetry.output_tokens,
                    "reasoning_tokens": selected.telemetry.reasoning_tokens,
                }
            )

    request_hashes = tuple(row["request_sha256"] for row in rows)
    decision_hashes = tuple(row["decision_sha256"] for row in rows)
    stage_widths = [
        step.planned_candidate_evaluations for step in bundle.prepared.schedule.steps
    ]
    expected_parent_binding = agentic._binding(
        "archive_elite_explorer",
        ArchiveEliteExplorerCampaignParentSelector(),
    ).to_record()
    expected_recombination_binding = agentic._binding(
        "archive_aware_disjoint_union",
        _PolicyTag("recombination"),
    ).to_record()
    expected_resolution = agentic._objective_resolution().to_record()
    observed_resolution = bundle.benchmark.objective_resolution.to_record()
    exact_expected_wave_count = len(rows) == (
        len(agentic.PORTFOLIO_GENERATIONS) * agentic.PARENTS_PER_PORTFOLIO
    )
    return {
        "status": "provider_and_pde_free_all_wave_control_selection",
        "provider_calls": 0,
        "actual_llm_calls": 0,
        "credential_read": False,
        "pde_solves": 0,
        "scientific_values": "synthetic_contract_construction_only",
        "portfolio_generations": list(agentic.PORTFOLIO_GENERATIONS),
        "planned_reflections": len(agentic.REFLECTION_SOURCE_GENERATIONS),
        "planned_stage_unique_evaluation_counts": stage_widths,
        "constructed_wave_count": len(rows),
        "selected_wave_count": len(decision_hashes),
        "exact_expected_wave_count": exact_expected_wave_count,
        "all_request_hashes_unique": (len(set(request_hashes)) == len(request_hashes)),
        "all_decision_hashes_unique": (
            len(set(decision_hashes)) == len(decision_hashes)
        ),
        "all_slate_selections_feasible": all(
            row["evaluation_width"] == agentic.PORTFOLIO_WIDTH
            and row["selected_family_count"] >= 3
            for row in rows
        ),
        "parent_measurement_bound_every_wave": all(
            row["parent_measurement_binding_sha256"] is not None for row in rows
        ),
        "policy_read_sets_empty": all(
            not record["outcome_fields_read_by_policy"]
            and not record["memory_or_reflection_fields_read_by_policy"]
            and not record["parent_measurement_fields_read_by_policy"]
            for record in records
        ),
        "parent_measurement_projection": (
            bundle.parent_measurement_projection.to_record()
        ),
        "all_acceptance_gates_pass": all(
            record["feasible_slate_space"]["acceptance_probability_float"]
            >= float(MINIMUM_ACCEPTANCE_PROBABILITY)
            for record in records
        ),
        "exact_stage_widths": stage_widths == _expected_stage_widths(),
        "objective_resolution_matches_treatment": (
            observed_resolution == expected_resolution
        ),
        "objective_resolution": observed_resolution,
        "parent_policy_matches_treatment": (
            bundle.policies.parent_selection.to_record() == expected_parent_binding
        ),
        "parent_policy": bundle.policies.parent_selection.to_record(),
        "recombination_policy_matches_treatment": (
            bundle.policies.recombination.to_record() == expected_recombination_binding
        ),
        "archive_utility_matches_treatment": (
            bundle.utility.definition_sha256 == agentic._affine_spec().definition_sha256
        ),
        "archive_utility": {
            "utility_id": bundle.utility.utility_id,
            "utility_version": bundle.utility.utility_version,
            "definition_sha256": bundle.utility.definition_sha256,
        },
        "rows": rows,
    }


def _manifest(
    *,
    run_id: str,
    mode: str,
    source: dict[str, object],
    source_snapshot: dict[str, object],
    preregistration: Path | None,
) -> dict[str, object]:
    invocation_observation = DirectV3Evaluator(
        agentic._evaluator_settings(ARTIFACT_ROOT / run_id / "pde")
    ).invocation_observation()
    prereg_record = None
    if preregistration is not None:
        prereg_record = {
            "path": preregistration.relative_to(WORKSPACE_ROOT).as_posix(),
            "sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "claim_boundary": {
            "matched_uninformed_floor_only": True,
            "paper_ready_result": False,
            "strong_baselines_still_required": True,
            "single_control_replicate": True,
        },
        "preregistration": prereg_record,
        "one_factor_match": {
            "task_sha256": agentic.TASK_SHA256,
            "workload_id": WORKLOAD_ID,
            "formulation_definition_sha256": FORMULATION_DEFINITION_SHA256,
            "evaluator": "direct_v3",
            "resolution": 1001,
            "cpu_set": "8",
            "external_concurrency": 1,
            "seeds": ["SEED_LAYOUT_A", "SEED_LAYOUT_B"],
            "cadence": (
                "portfolio_g1_recombination_g2_portfolio_g3_"
                "recombination_g4_portfolio_g5_recombination_g6"
            ),
            "parent_policy": "archive_elite_explorer_v1",
            "portfolio_width": agentic.PORTFOLIO_WIDTH,
            "minimum_distinct_families": 3,
            "recombinations_per_source": 2,
            "target_unique_evaluations": TARGET_UNIQUE_EVALUATIONS,
            "archive_utility": agentic._affine_spec().to_record(),
            "objective_resolution": agentic._objective_resolution().to_record(),
            "reflection_supervision": CampaignReflectionSupervisionPolicy(
                ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY
            ).to_record(),
        },
        "control_treatment": {
            "policy": "TaskKeyedConditionalUniformPortfolioPolicy",
            "policy_definition_sha256": POLICY_DEFINITION_SHA256,
            "replicate_seed": CONTROL_REPLICATE_SEED,
            "id_namespace": CONTROL_ID_NAMESPACE,
            "rejection_cap": MAX_REJECTION_DRAWS,
            "minimum_runtime_acceptance_probability": "1/100",
            "memory": "same preregistered cards present but never consumed or credited",
            "reflection": "one delayed inert local quarantined receipt",
            "actual_llm_calls": 0,
            "provider_calls": 0,
            "credential_read": False,
        },
        "parent_measurement": {
            "enabled_for_live_parent_selection": True,
            "raw_scientific_and_decision_values_separate": True,
            "current_wave_outcomes_included": False,
            "uniform_policy_fields_read": [],
        },
        "analysis": {
            "equal_evaluation": (
                f"fixed {agentic.PLANNED_UNIQUE_EVALUATIONS} unique evaluations"
            ),
            "equal_wall": (
                "compare authenticated anytime archives at the minimum of the two "
                "complete-run wall times; carry the final archive forward after a "
                "method finishes"
            ),
            "total_wall_to_planned_budget_reported_separately": True,
        },
        "utility_reference_qualification": {
            "all_void_thermal_term": agentic.QUALIFIED_ALL_VOID_THERMAL_TERM,
            "all_void_thermal_term_hex": (
                agentic.QUALIFIED_ALL_VOID_THERMAL_TERM.hex()
            ),
            "source_manifest_sha256": (agentic.QUALIFIED_ALL_VOID_MANIFEST_SHA256),
            "thermal_reference": agentic.THERMAL_AFFINE_REFERENCE,
            "material_reference": agentic.MATERIAL_AFFINE_REFERENCE,
        },
        "scientific_resource_gates": {
            "maximum_direct_v3_elapsed_s_hex": MAX_PDE_WALL_S.hex(),
            "maximum_peak_rss_bytes": MAX_PDE_PEAK_RSS_BYTES,
            "required_manifest_count": TARGET_UNIQUE_EVALUATIONS,
        },
        "source_identity": source,
        "source_snapshot": source_snapshot,
        "evaluator_process_invocation_observation": invocation_observation,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "python_executable_invoked": os.path.abspath(sys.executable),
            "python_executable_resolved": str(
                Path(sys.executable).resolve(strict=True)
            ),
            "python_prefix": os.path.abspath(sys.prefix),
            "python_base_prefix": os.path.abspath(sys.base_prefix),
            "numpy_module_origin": (
                None
                if importlib.util.find_spec("numpy") is None
                else importlib.util.find_spec("numpy").origin
            ),
            "resolved_package_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pydantic", "pydantic-ai", "openai")
            },
        },
    }


async def _live(
    *,
    bundle: _Bundle,
    run_dir: Path,
    journals: dict[str, DurableJsonlJournal],
    expected_source_aggregate_sha256: str,
) -> dict[str, object]:
    wave_records: list[dict[str, object]] = []
    reflection_records: list[dict[str, object]] = []
    policy = TaskKeyedConditionalUniformPortfolioPolicy(
        task_sha256=agentic.TASK_SHA256,
        replicate_seed=CONTROL_REPLICATE_SEED,
    )
    execution_started_ns = time.perf_counter_ns()

    def engine_trace_sink(record: dict[str, object]) -> None:
        journals["engine"].append(
            {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - execution_started_ns
                    ),
                    "observed_at_utc": _utc_now(),
                },
                "authenticated_record": dict(record),
            }
        )

    composition = compose_portfolio_evolution(
        bundle.benchmark,
        generator=_NeverGenerator(),
        selector=policy,
        seed=agentic.OUTER_SEED,
        id_factory=bundle.ids,
        memory=bundle.memory,
        evaluator_concurrency=1,
        engine_trace_sink=engine_trace_sink,
        max_output_tokens=1,
        temperature=None,
    )
    parent_selector = ArchiveEliteExplorerCampaignParentSelector()
    runtime = AgenticPortfolioCampaignRuntime(
        prepared=bundle.prepared,
        workload_config=bundle.config,
        workload_ports=bundle.workload_ports,
        composition=composition,
        parent_selector=parent_selector,
        wave_factory=_ControlWaveFactory(
            bundle.ids,
            bundle.memory,
            bundle.utility,
            wave_records,
        ),
        task_sha256=agentic.TASK_SHA256,
        parent_measurement_projection=bundle.parent_measurement_projection,
        reflection_executor=_InertReflection(reflection_records),
        outcome_updater=None,
        recombination_utility_binder=agentic._RecombinationUtilityBinder(
            bundle.utility
        ),
        owned_resources=_OwnedLocalResources(),
    )
    scheduler_started = time.perf_counter()
    result = await EvolutionCampaignScheduler(
        prepared=bundle.prepared,
        policies=bundle.policies,
        stages=runtime,
        reflections=runtime,
        lifecycle=runtime,
        journal=_TimedExecutionJournal(
            journals["campaign"],
            execution_started_ns,
        ),
    ).run()
    wall_s = time.perf_counter() - scheduler_started
    cache = await composition.engine.evaluation_cache_snapshot()
    stage_counts = [value.candidate_occurrence_count for value in result.stage_receipts]
    stage_unique = [value.unique_evaluation_count for value in result.stage_receipts]
    front = runtime.final_front
    history = [agentic._candidate_history_record(value) for value in runtime.history]
    stage_records = [thaw_json(value.result) for value in result.stage_receipts]
    spec = agentic._affine_spec()
    seed_front = [value for value in history if value["generation"] == 0]
    trajectory = [
        agentic._archive_trajectory_record(
            label="g0_seed_cutoff",
            generation=0,
            front_candidates=seed_front,
            spec=spec,
        )
    ]
    expected_initial_hv_hex = str(trajectory[0]["normalized_hypervolume_hex"])
    initial_hypervolumes = {
        record["affine_snapshot"]["base_hypervolume_hex"]
        for record in wave_records
        if record["generation"] == 1
    }
    for generation, stage_record in enumerate(stage_records, start=1):
        archive_after = stage_record.get("archive_after")
        if (
            type(archive_after) is not dict
            or type(archive_after.get("front_candidates")) is not list
        ):
            raise RuntimeError("completed control stage omitted archive front")
        trajectory.append(
            agentic._archive_trajectory_record(
                label=f"g{generation}_archive_after",
                generation=generation,
                front_candidates=list(archive_after["front_candidates"]),
                spec=spec,
            )
        )
    unique_configurations = {
        value.occurrence.configuration_hash for value in runtime.history
    }
    reference_violations = []
    for candidate in runtime.history:
        normalized = spec.normalize(candidate.objective_map)
        if any(value >= 1.0 for value in normalized):
            reference_violations.append(
                {
                    "candidate_id": candidate.candidate_id.value,
                    "objectives": candidate.objective_map,
                    "normalized": [value.hex() for value in normalized],
                }
            )
    evaluation_accounting = CampaignEvaluationAccounting(
        planned_candidate_occurrences=TARGET_UNIQUE_EVALUATIONS,
        seed_occurrences=(
            result.counters.candidate_occurrences - sum(stage_counts)
        ),
        seed_unique_evaluations=(
            result.counters.unique_evaluations - sum(stage_unique)
        ),
        stage_occurrences=tuple(stage_counts),
        stage_unique_evaluations=tuple(stage_unique),
        candidate_occurrences=result.counters.candidate_occurrences,
        unique_evaluations=result.counters.unique_evaluations,
    )
    resource_evidence = agentic._pde_evidence_record(
        run_dir,
        expected_physical_evaluations=evaluation_accounting.unique_evaluations,
    )
    postrun_source = source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)
    source_closure_unchanged = (
        postrun_source["aggregate_sha256"] == expected_source_aggregate_sha256
    )
    health = {
        "exact_generations": (
            result.counters.generations_completed == agentic.GENERATION_COUNT
        ),
        "exact_occurrences": (
            result.counters.candidate_occurrences == TARGET_UNIQUE_EVALUATIONS
        ),
        "exact_unique_evaluations": (
            result.counters.unique_evaluations == TARGET_UNIQUE_EVALUATIONS
        ),
        "exact_evaluation_accounting": True,
        "exact_scheduler_local_decision_operations": (
            result.counters.logical_agent_calls == SCHEDULER_LOCAL_DECISION_OPERATIONS
        ),
        "zero_actual_llm_calls": ACTUAL_LLM_CALLS == 0,
        "zero_provider_calls": PROVIDER_CALLS == 0,
        "exact_stage_occurrences": stage_counts == _expected_stage_widths(),
        "exact_stage_unique": stage_unique == _expected_stage_widths(),
        "all_candidates_valid": all(value.valid for value in runtime.history),
        "all_configurations_unique": (
            len(unique_configurations) == TARGET_UNIQUE_EVALUATIONS
        ),
        "exact_cache_misses": cache["misses"] == TARGET_UNIQUE_EVALUATIONS,
        "no_cache_hits": cache["hits"] == 0,
        "cache_drained": cache["in_flight"] == 0,
        "exact_uniform_policy_calls": len(wave_records) == 6,
        "parent_measurement_bound_every_selector_wave": all(
            record["parent_measurement_binding_sha256"] is not None
            for record in wave_records
        ),
        "uniform_policy_read_sets_empty": all(
            not record["outcome_fields_read_by_policy"]
            and not record["memory_or_reflection_fields_read_by_policy"]
            and not record["parent_measurement_fields_read_by_policy"]
            for record in wave_records
        ),
        "all_acceptance_gates_pass": all(
            record["feasible_slate_space"]["acceptance_probability_float"] >= 0.01
            for record in wave_records
        ),
        "frozen_memory_no_trials": len(bundle.memory.trials) == 0,
        "one_delayed_inert_reflection_receipt": len(reflection_records) == 1,
        "nonempty_final_front": bool(front),
        "cleanup_released": result.cleanup_receipt.released,
        "affine_reference_contains_every_candidate": not reference_violations,
        "qualified_initial_hypervolume_reproduced": initial_hypervolumes
        == {expected_initial_hv_hex},
        "archive_aware_recombination_enabled": all(
            stage_records[index].get("archive_aware_source_utility") is True
            for index in (1, 3, 5)
        ),
        "physical_direct_v3_manifests_match_accounting": resource_evidence[
            "manifest_count_matches_physical_evaluations"
        ],
        "all_direct_v3_scientific_contracts_pass": resource_evidence[
            "all_scientific_contracts_pass"
        ],
        "all_pde_evaluations_under_45_s_and_3_gib": resource_evidence[
            "all_under_45_s_and_3_gib"
        ],
        "all_candidates_use_fixed_grid_objectives": all(
            candidate.objective_resolution_receipt is not None
            for candidate in runtime.history
        ),
        "all_candidates_have_raw_detailed_evidence": all(
            candidate.detailed_evaluation is not None for candidate in runtime.history
        ),
        "source_closure_unchanged": source_closure_unchanged,
    }
    health_pass = all(health.values())
    return {
        "schema_version": 1,
        "status": "completed_healthy" if health_pass else "completed_unhealthy",
        "health_pass": health_pass,
        "health": health,
        "campaign_result": result.to_record(),
        "counters": result.counters.to_record(),
        "accounting_semantics": {
            "scheduler_logical_agent_calls_field": (
                SCHEDULER_LOCAL_DECISION_OPERATIONS
            ),
            "local_uniform_selection_operations": 6,
            "local_inert_reflection_operations": 1,
            "actual_llm_calls": 0,
            "provider_calls": 0,
            "credential_read": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": "0",
        },
        "wall_s": wall_s,
        "wall_s_to_planned_unique_evaluations": wall_s,
        "stage_occurrence_counts": stage_counts,
        "stage_unique_evaluation_counts": stage_unique,
        "evaluation_accounting": evaluation_accounting.to_record(),
        "cache": cache,
        "candidate_count": len(runtime.history),
        "unique_configuration_count": len(unique_configurations),
        "archive_hypervolume_trajectory": trajectory,
        "expected_initial_hypervolume_hex": expected_initial_hv_hex,
        "final_normalized_hypervolume_hex": trajectory[-1][
            "normalized_hypervolume_hex"
        ],
        "final_raw_oriented_hypervolume_hex": trajectory[-1][
            "raw_oriented_hypervolume_hex"
        ],
        "candidate_history": history,
        "final_front": [
            {
                "candidate_id": value.candidate_id.value,
                "configuration_sha256": value.occurrence.configuration_hash,
                "objectives": value.objective_map,
                "generation": value.generation,
                "operator_kind": (
                    None if value.operator_kind is None else value.operator_kind.value
                ),
                "parent_ids": [item.value for item in value.parent_ids],
            }
            for value in front
        ],
        "uniform_wave_records": wave_records,
        "reflection_records": reflection_records,
        "pde_evidence": resource_evidence,
        "postrun_source_identity": postrun_source,
        "source_closure": {
            "launch_aggregate_sha256": expected_source_aggregate_sha256,
            "postrun_aggregate_sha256": postrun_source["aggregate_sha256"],
            "unchanged": source_closure_unchanged,
            "postrun_file_count": postrun_source["file_count"],
        },
        "reference_violations": reference_violations,
        "memory": {
            "seed_entry_count": len(bundle.memory.entries),
            "trial_count": len(bundle.memory.trials),
            "adaptive_score_consumption": False,
        },
        "parent_measurement_projection": (
            bundle.parent_measurement_projection.to_record()
        ),
    }


async def _main_async(args: argparse.Namespace) -> int:
    preregistration = None
    if args.prereg is not None:
        preregistration = Path(args.prereg).expanduser().resolve(strict=True)
        if WORKSPACE_ROOT not in preregistration.parents:
            raise RuntimeError("preregistration must live inside the workspace")
    if args.mode == "live" and preregistration is None:
        raise RuntimeError("live mode requires --prereg")

    run_dir = (ARTIFACT_ROOT / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    preparation: DurableJsonlJournal | None = None
    try:
        source_paths = _source_paths()
        source = source_identity(source_paths, relative_to=WORKSPACE_ROOT)
        source_snapshot = _snapshot_sources(run_dir, source_paths)
        if (
            source_snapshot["aggregate_sha256"] != source["aggregate_sha256"]
            or source_snapshot["files"] != source["files"]
        ):
            raise RuntimeError("source-byte snapshot differs from launch identity")
        write_json_atomic(
            run_dir / "manifest.json",
            _manifest(
                run_id=args.run_id,
                mode=args.mode,
                source=source,
                source_snapshot=source_snapshot,
                preregistration=preregistration,
            ),
        )
        preparation = DurableJsonlJournal(run_dir / "preparation.jsonl")
        bundle = _prepare_bundle(
            run_dir=run_dir,
            preparation_journal=preparation,
            source_closure_sha256=str(source["aggregate_sha256"]),
        )
        readiness = _eligibility_probe(bundle)
        write_json_atomic(run_dir / "readiness.json", readiness)
        if not readiness["gate_under_60_process_cpu_s_each"]:
            raise RuntimeError(
                "semantic binding readiness exceeded 60 process-CPU seconds"
            )
        if not readiness["all_acceptance_gates_pass"]:
            raise RuntimeError(
                "provider-free finite spaces failed the one-percent acceptance gate"
            )
        all_wave_probe = await _control_all_wave_probe(bundle)
        write_json_atomic(run_dir / "control_all_wave_probe.json", all_wave_probe)
        if not all(
            all_wave_probe[name]
            for name in (
                "exact_expected_wave_count",
                "all_request_hashes_unique",
                "all_decision_hashes_unique",
                "all_slate_selections_feasible",
                "parent_measurement_bound_every_wave",
                "policy_read_sets_empty",
                "all_acceptance_gates_pass",
                "exact_stage_widths",
                "objective_resolution_matches_treatment",
                "parent_policy_matches_treatment",
                "recombination_policy_matches_treatment",
                "archive_utility_matches_treatment",
            )
        ):
            raise RuntimeError("conditional-uniform G6 all-wave gate failed")
        if args.mode == "prepare":
            summary = {
                "schema_version": 1,
                "status": "prepared_provider_and_pde_free",
                "provider_calls": 0,
                "actual_llm_calls": 0,
                "credential_read": False,
                "pde_solves": 0,
                "preparation": bundle.prepared.to_record(),
                "memory_plan": bundle.memory_plan.to_record(),
                "parent_measurement_projection": (
                    bundle.parent_measurement_projection.to_record()
                ),
                "readiness": readiness,
                "control_all_wave_probe": all_wave_probe,
                "source_snapshot": source_snapshot,
            }
            write_json_atomic(run_dir / "summary.json", summary)
            preparation.close()
            finalize_run_directory(run_dir, status=str(summary["status"]))
            print(
                json.dumps(summary, ensure_ascii=True, allow_nan=False, sort_keys=True)
            )
            return 0

        journals = {
            "engine": DurableJsonlJournal(run_dir / "engine_events.jsonl"),
            "campaign": DurableJsonlJournal(run_dir / "campaign_events.jsonl"),
        }
        try:
            summary = await _live(
                bundle=bundle,
                run_dir=run_dir,
                journals=journals,
                expected_source_aggregate_sha256=str(source["aggregate_sha256"]),
            )
            write_json_atomic(run_dir / "summary.json", summary)
        finally:
            for journal in journals.values():
                journal.close()
        preparation.close()
        finalize_run_directory(run_dir, status=str(summary["status"]))
        print(json.dumps(summary, ensure_ascii=True, allow_nan=False, sort_keys=True))
        return 0
    except BaseException as exc:
        if preparation is not None:
            preparation.close()
        if not (run_dir / "summary.json").exists():
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_digest_sha256": hashlib.sha256(
                        type(exc).__qualname__.encode("utf-8")
                        + b"\x00"
                        + str(exc).encode("utf-8", errors="replace")
                    ).hexdigest(),
                    "provider_calls": 0,
                    "actual_llm_calls": 0,
                    "credential_read": False,
                },
            )
        if not (run_dir / "finalized.json").exists():
            finalize_run_directory(run_dir, status="failed")
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "live"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prereg")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

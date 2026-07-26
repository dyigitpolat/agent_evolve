#!/usr/bin/env python3
"""Run the preregistered provider-free BOiLS portfolio-Q random control."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.agentic import (  # noqa: E402
    AgenticBenchmark,
    DeterministicIdFactory,
    InsightDraft,
    InsightMemoryBank,
    OptimizerState,
    ParetoArchive,
    PortfolioCard,
    PortfolioRecombination,
    PortfolioRecombinationWaveRequest,
    PortfolioSelectionRequest,
    PortfolioVariationWaveRequest,
    TaskKeyedArchiveReservoirParentPolicy,
    compose_portfolio_evolution,
    eligible_finite_variation_view,
    exact_configuration_phenotype_bindings,
    freeze_json,
)
from agent_evolve.application.agentic_evolution import EvolutionCandidate  # noqa: E402
from agent_evolve.application.budgeted_optimizer import (  # noqa: E402
    pareto_archive_snapshot_hash,
)
from agent_evolve.policies.selection.random_portfolio import (  # noqa: E402
    POLICY_DEFINITION_SHA256,
    DeterministicRandomFeasiblePortfolioPolicy,
)
from examples.benchmarks.boils_abc.actions import (  # noqa: E402
    DEFAULT_ACTION_SEQUENCE,
)
from examples.benchmarks.boils_abc.budgeted_v5_support import (  # noqa: E402
    PARENT_C_SEQUENCE,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (  # noqa: E402
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem  # noqa: E402
from examples.development.durable_run_artifacts import (  # noqa: E402
    DurableJsonlJournal,
    finalize_run_directory,
    source_identity,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/portfolio_q"
)
PREREGISTRATION = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts"
    / "181_boils_portfolio_q_matched_random_control_preregistration.md"
)
DEFAULT_RUN_ID = "boilsq_control_random_seed20260716_r1"
CONTROL_SEED = 20_260_716
TASK_SHA256 = hashlib.sha256(b"agent-evolve:boils-log2-portfolio-q-task:v1").hexdigest()
METRIC_IDS = ("total_levels", "total_lut_count")
PORTFOLIO_SIZE = 4
PARENTS_PER_CYCLE = 2
CYCLES = 3
RESERVOIR_LIMIT = 6
HYPERVOLUME_REFERENCE = (80, 12_000)


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - control forbids calls.
        raise AssertionError(f"random control invoked propose: {request}")

    async def reflect(self, request):  # pragma: no cover - control forbids calls.
        raise AssertionError(f"random control invoked reflect: {request}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _id_namespace(run_id: str) -> str:
    return f"boilsr_{_sha_text(run_id)[:16]}"


def _available_affinity_sets(limit: int = 16) -> tuple[tuple[int, ...], ...]:
    available = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    return tuple((cpu,) for cpu in available[: max(1, min(limit, len(available)))])


def _benchmark() -> tuple[AgenticBenchmark, tuple[tuple[int, ...], ...]]:
    affinities = _available_affinity_sets()
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=affinities,
        per_circuit_timeout_s=60.0,
    )
    return (
        AgenticBenchmark(
            problem=BoilsAbcProblem(settings),
            finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
        ),
        affinities,
    )


def _frozen_object(value: Mapping[str, object]):
    frozen = freeze_json(dict(value))
    if type(frozen).__name__ != "FrozenJsonObject":
        raise TypeError("expected a frozen typed-JSON object")
    return frozen


def _seed_inert_memory(memory: InsightMemoryBank) -> tuple[Any, PortfolioCard]:
    entry = memory.extend(
        (
            InsightDraft(
                claim="This inert control card carries no action recommendation.",
                trigger="A request schema requires a non-empty card tuple.",
                mechanism="The random policy ignores every card payload and score.",
                affected_paths=("$.sequence",),
                evidence_summary="No empirical evidence is supplied to the control.",
                confidence=0.0,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )[0]
    return entry, PortfolioCard(
        card_key="card.control",
        reference=entry.reference,
        content_sha256=entry.draft.content_sha256,
        evidence_sha256=_sha_text("boils-portfolio-q-inert-control-card-v1"),
        prompt_payload=_frozen_object({"control": "inert_no_action_information"}),
        assigned_score=0.0,
    )


def _candidate_record(candidate: EvolutionCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "configuration": candidate.configuration_dict,
        "objectives": candidate.objective_map,
        "valid": candidate.valid,
        "generation": candidate.generation,
        "label": candidate.label,
        "operator_kind": (
            None if candidate.operator_kind is None else candidate.operator_kind.value
        ),
        "parent_ids": [value.value for value in candidate.parent_ids],
        "common_ancestor_id": (
            None
            if candidate.common_ancestor_id is None
            else candidate.common_ancestor_id.value
        ),
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "parent_patch_hashes": list(candidate.parent_patch_hashes),
        "preservation_verified": candidate.preservation_verified,
    }


def _manifest(
    *,
    run_id: str,
    affinities: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    sources = (
        Path(__file__),
        PREREGISTRATION,
        AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/selection/random_portfolio.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/application/portfolio_evolution.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/application/portfolio_recombination.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/selection/archive_elite.py",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/finite_variation_catalog.py",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "mode": "provider_free_matched_random_control",
        "preregistration": str(PREREGISTRATION.relative_to(WORKSPACE_ROOT)),
        "preregistration_sha256": hashlib.sha256(
            PREREGISTRATION.read_bytes()
        ).hexdigest(),
        "claim_boundary": {
            "workflow_development_only": True,
            "paper_ready_result": False,
            "sota_claim": False,
            "single_post_treatment_control": True,
        },
        "estimand": (
            "finite-budget information advantage of agentic portfolio selection "
            "and adaptive memory over outcome-blind feasible random selection"
        ),
        "workload": {
            "id": "boils_abc_log2_portfolio_q",
            "circuit_panel": ["log2"],
            "objective_ids": list(METRIC_IDS),
            "evaluation_affinity_sets": [list(value) for value in affinities],
        },
        "schedule": {
            "cycles": CYCLES,
            "parents_per_cycle": PARENTS_PER_CYCLE,
            "portfolio_size": PORTFOLIO_SIZE,
            "mutation_candidates": 24,
            "recombination_candidates": 12,
            "seed_candidates": 2,
            "target_unique_evaluations": 38,
            "generation_numbers": list(range(1, 2 * CYCLES + 1)),
        },
        "selection_control": {
            "base_seed": CONTROL_SEED,
            "policy": "DeterministicRandomFeasiblePortfolioPolicy",
            "policy_definition_sha256": POLICY_DEFINITION_SHA256,
            "minimum_distinct_families": 3,
            "pairwise_disjoint_patch_paths": True,
            "outcome_blind": True,
            "provider_calls": 0,
        },
        "memory": {
            "mode": "frozen_inert_schema_card",
            "adaptive_selection": False,
            "credit_trials": 0,
            "reflections": 0,
        },
        "genericity": {
            "composition": "agent_evolve.agentic.compose_portfolio_evolution",
            "workload_injection": "AgenticBenchmark plus finite variation catalog",
            "candidate_materialization": "engine_exact",
            "parent_selection": "generic_task_keyed_archive_reservoir",
            "recombination": "generic_disjoint_parent_patch_union",
        },
        "hypervolume_reference": {
            "total_levels": HYPERVOLUME_REFERENCE[0],
            "total_lut_count": HYPERVOLUME_REFERENCE[1],
        },
        "source_identity": source_identity(sources, relative_to=WORKSPACE_ROOT),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
        },
    }


async def _register_seeds(composition: Any) -> tuple[EvolutionCandidate, ...]:
    return tuple(
        await asyncio.gather(
            composition.engine.register_seed(
                {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
                label="seed_default_q0",
            ),
            composition.engine.register_seed(
                {"sequence": list(PARENT_C_SEQUENCE)},
                label="seed_parent_c",
            ),
        )
    )


def _make_wave(
    *,
    composition: Any,
    parent: EvolutionCandidate,
    card: PortfolioCard,
    generation: int,
    cycle: int,
    parent_ordinal: int,
    known_configuration_sha256s: tuple[str, ...],
) -> PortfolioVariationWaveRequest:
    base_contract = composition.bind_finite_variation(
        FINITE_CATALOG_ID,
        parent.configuration,
    )
    eligibility = eligible_finite_variation_view(
        contract=base_contract,
        option_phenotypes=exact_configuration_phenotype_bindings(base_contract),
        known_phenotype_sha256s=known_configuration_sha256s,
    )
    selection = PortfolioSelectionRequest(
        call_id=composition.id_factory.new_llm_call_id(),
        operation="select_portfolio",
        instruction=(
            "Select exactly four sealed options under the frozen outcome-blind "
            "random-control policy. No objective or memory information is used."
        ),
        context=_frozen_object(
            {
                "control": "outcome_blind_random_feasible",
                "portfolio_size": PORTFOLIO_SIZE,
            }
        ),
        finite_variation_contract=eligibility.contract,
        cards=(card,),
        portfolio_size=PORTFOLIO_SIZE,
        required_metric_ids=METRIC_IDS,
        min_distinct_families=3,
        require_supporting_cards=False,
        max_output_tokens=1,
        temperature=None,
    )
    return PortfolioVariationWaveRequest(
        selection_request=selection,
        parent=parent,
        generation=generation,
        label_prefix=f"boilsr.c{cycle:02d}.p{parent_ordinal:02d}",
        phase="portfolio_random_control",
        memory_credit=None,
    )


def _hypervolume(front: Sequence[EvolutionCandidate]) -> int:
    reference_x, reference_y = HYPERVOLUME_REFERENCE
    points = sorted(
        (
            int(candidate.objective_map["total_levels"]),
            int(candidate.objective_map["total_lut_count"]),
        )
        for candidate in front
    )
    previous_y = reference_y
    volume = 0
    for x_value, y_value in points:
        if x_value >= reference_x or y_value >= previous_y:
            continue
        volume += (reference_x - x_value) * (previous_y - y_value)
        previous_y = y_value
    return volume


async def _execute(
    *,
    run_id: str,
    engine_journal: DurableJsonlJournal,
    planner_journal: DurableJsonlJournal,
    wave_journal: DurableJsonlJournal,
) -> dict[str, object]:
    benchmark, affinities = _benchmark()
    ids = DeterministicIdFactory(_id_namespace(run_id))
    memory = InsightMemoryBank(id_factory=ids)
    inert_entry, card = _seed_inert_memory(memory)
    selector = DeterministicRandomFeasiblePortfolioPolicy(seed=CONTROL_SEED)
    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=selector,
        seed=CONTROL_SEED,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=len(affinities),
        engine_trace_sink=engine_journal.append,
        max_output_tokens=1,
        temperature=None,
    )
    archive = ParetoArchive(
        benchmark.objectives,
        outcome_relation_binding=composition.outcome_relation,
    )
    recombiner = PortfolioRecombination(
        engine=composition.engine,
        ids=composition.id_factory,
    )
    parent_policy = TaskKeyedArchiveReservoirParentPolicy()
    history: list[EvolutionCandidate] = []
    variation_results: list[Any] = []
    recombination_results: list[Any] = []
    parent_records: list[dict[str, object]] = []
    archive_cursor = 0
    run_started = time.perf_counter()
    seed_started = time.perf_counter()
    seeds = await _register_seeds(composition)
    seed_wall_s = time.perf_counter() - seed_started
    if len(seeds) != 2 or not all(seed.valid for seed in seeds):
        raise RuntimeError("random control seeds were not both valid")
    for seed in seeds:
        history.append(seed)
        archive.consider(seed)

    for cycle in range(1, CYCLES + 1):
        mutation_generation = 2 * cycle - 1
        recombination_generation = 2 * cycle
        cache_before = await composition.engine.evaluation_cache_snapshot()
        snapshot = archive.snapshot()
        state = OptimizerState(
            generation=2 * (cycle - 1),
            candidates=tuple(history),
            archive=snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
            unique_evaluations=int(cache_before["misses"] or 0),
            logical_llm_calls=0,
        )
        parent_selection = parent_policy.select(
            state,
            task_sha256=TASK_SHA256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            reservoir_limit=RESERVOIR_LIMIT,
            parent_count=PARENTS_PER_CYCLE,
            rotation_index=cycle - 1,
        )
        parent_record = {
            "event_type": "parent_reservoir_selection",
            "cycle": cycle,
            "receipt": parent_selection.receipt.to_trace_record(),
        }
        parent_records.append(parent_record)
        planner_journal.append(parent_record)
        known = tuple(
            sorted({candidate.occurrence.configuration_hash for candidate in history})
        )
        waves = tuple(
            _make_wave(
                composition=composition,
                parent=parent,
                card=card,
                generation=mutation_generation,
                cycle=cycle,
                parent_ordinal=ordinal,
                known_configuration_sha256s=known,
            )
            for ordinal, parent in enumerate(parent_selection.parents, start=1)
        )
        for ordinal, wave in enumerate(waves, start=1):
            wave_journal.append(
                {
                    "event_type": "random_portfolio_wave_prepared",
                    "cycle": cycle,
                    "parent_ordinal": ordinal,
                    "parent_candidate_id": wave.parent.candidate_id.value,
                    "parent_objectives": wave.parent.objective_map,
                    "selection_request": wave.selection_request.to_record(),
                    "policy_definition_sha256": POLICY_DEFINITION_SHA256,
                }
            )
        mutation_started = time.perf_counter()
        cycle_variations = tuple(
            await asyncio.gather(*(composition.portfolio.run(wave) for wave in waves))
        )
        mutation_wall_s = time.perf_counter() - mutation_started
        for wave, result in zip(waves, cycle_variations, strict=True):
            variation_results.append(result)
            planner_journal.append(
                {
                    "event_type": "portfolio_wave_completed",
                    "cycle": cycle,
                    "wave_request_sha256": wave.selection_request.request_sha256,
                    "receipt": result.receipt.to_record(),
                }
            )
            for candidate in result.candidates:
                history.append(candidate)
                archive.consider(candidate)

        recombination_requests = tuple(
            PortfolioRecombinationWaveRequest(
                source_wave=wave,
                source_result=result,
                ancestor=wave.parent,
                generation=recombination_generation,
                label_prefix=f"boilsr.r{cycle:02d}.p{ordinal:02d}",
                phase="portfolio_recombination",
            )
            for ordinal, (wave, result) in enumerate(
                zip(waves, cycle_variations, strict=True),
                start=1,
            )
        )
        recombination_started = time.perf_counter()
        cycle_recombinations = tuple(
            await asyncio.gather(
                *(recombiner.run(request) for request in recombination_requests)
            )
        )
        recombination_wall_s = time.perf_counter() - recombination_started
        for request, result in zip(
            recombination_requests,
            cycle_recombinations,
            strict=True,
        ):
            recombination_results.append(result)
            planner_journal.append(
                {
                    "event_type": "recombination_wave_completed",
                    "cycle": cycle,
                    "source_request_sha256": (
                        request.source_wave.selection_request.request_sha256
                    ),
                    "receipt": result.receipt.to_record(),
                }
            )
            for candidate in result.candidates:
                history.append(candidate)
                archive.consider(candidate)

        new_archive_decisions = archive.decisions[archive_cursor:]
        archive_cursor = len(archive.decisions)
        planner_journal.append(
            {
                "event_type": "cycle_closed",
                "cycle": cycle,
                "mutation_generation": mutation_generation,
                "recombination_generation": recombination_generation,
                "mutation_wall_s": mutation_wall_s,
                "recombination_wall_s": recombination_wall_s,
                "mutation_candidate_count": sum(
                    len(result.candidates) for result in cycle_variations
                ),
                "recombination_candidate_count": sum(
                    len(result.candidates) for result in cycle_recombinations
                ),
                "archive_decisions": [
                    value.to_trace_record() for value in new_archive_decisions
                ],
                "archive": archive.snapshot().to_trace_record(),
                "cache": await composition.engine.evaluation_cache_snapshot(),
            }
        )

    cache = await composition.engine.evaluation_cache_snapshot()
    final_archive = archive.snapshot()
    unique_configurations = {
        candidate.occurrence.configuration_hash for candidate in history
    }
    recombination_counts = [len(result.candidates) for result in recombination_results]
    health_checks = {
        "both_seeds_valid": len(seeds) == 2 and all(seed.valid for seed in seeds),
        "six_portfolio_waves": len(variation_results) == 6,
        "every_portfolio_has_four_candidates": all(
            len(result.candidates) == PORTFOLIO_SIZE for result in variation_results
        ),
        "six_recombination_waves": len(recombination_results) == 6,
        "two_recombinations_per_wave": recombination_counts == [2] * 6,
        "exact_candidate_count": len(history) == 38,
        "all_configurations_unique": len(unique_configurations) == 38,
        "exact_cache_misses": cache["misses"] == 38,
        "no_cache_hits": cache["hits"] == 0,
        "cache_drained": cache["in_flight"] == 0,
        "memory_frozen": len(memory.entries) == 1 and len(memory.trials) == 0,
        "no_provider_or_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "status": "completed" if all(health_checks.values()) else "completed_unhealthy",
        "health_pass": all(health_checks.values()),
        "health_checks": health_checks,
        "claim_boundary": {
            "workflow_development_only": True,
            "paper_ready_result": False,
            "sota_claim": False,
            "single_random_replicate": True,
        },
        "run_wall_s": time.perf_counter() - run_started,
        "seed_batch_wall_s": seed_wall_s,
        "provider_calls": 0,
        "credential_read": False,
        "logical_llm_calls": 0,
        "local_selection_policy_calls": len(variation_results),
        "reflection_calls": 0,
        "memory_credit_trials": len(memory.trials),
        "random_policy": {
            "base_seed": CONTROL_SEED,
            "definition_sha256": POLICY_DEFINITION_SHA256,
        },
        "cache": cache,
        "candidate_count": len(history),
        "unique_configuration_count": len(unique_configurations),
        "candidates": [_candidate_record(value) for value in history],
        "final_front": [
            _candidate_record(value) for value in final_archive.front_candidates
        ],
        "final_hypervolume": _hypervolume(final_archive.front_candidates),
        "hypervolume_reference": {
            "total_levels": HYPERVOLUME_REFERENCE[0],
            "total_lut_count": HYPERVOLUME_REFERENCE[1],
        },
        "archive": final_archive.to_trace_record(),
        "parent_selections": parent_records,
        "portfolio_wave_count": len(variation_results),
        "portfolio_receipts": [
            result.receipt.to_record() for result in variation_results
        ],
        "recombination_wave_count": len(recombination_results),
        "recombination_candidate_counts": recombination_counts,
        "recombination_receipts": [
            result.receipt.to_record() for result in recombination_results
        ],
        "inert_memory_reference": {
            "insight_id": inert_entry.reference.insight_id.value,
            "version": inert_entry.reference.version,
        },
    }


async def _run(run_id: str) -> tuple[Path, dict[str, object]]:
    if not PREREGISTRATION.is_file():
        raise RuntimeError("frozen preregistration is missing")
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _unused_benchmark, affinities = _benchmark()
    del _unused_benchmark
    write_json_atomic(
        run_dir / "manifest.json", _manifest(run_id=run_id, affinities=affinities)
    )
    journals = (
        DurableJsonlJournal(run_dir / "engine_events.jsonl"),
        DurableJsonlJournal(run_dir / "planner_events.jsonl"),
        DurableJsonlJournal(run_dir / "wave_requests.jsonl"),
        DurableJsonlJournal(run_dir / "provider_outcomes.jsonl"),
    )
    failure: Exception | None = None
    try:
        result = await _execute(
            run_id=run_id,
            engine_journal=journals[0],
            planner_journal=journals[1],
            wave_journal=journals[2],
        )
        write_json_atomic(run_dir / "summary.json", result)
    except Exception as exc:
        write_json_atomic(
            run_dir / "failed.json",
            {
                "status": "failed",
                "failure_type": type(exc).__name__,
                "safe_message": str(exc)[:2_000],
                "failed_at_utc": _utc_now(),
                "provider_calls": 0,
                "credential_read": False,
            },
        )
        failure = exc
    finally:
        for journal in journals:
            journal.close()
    if failure is not None:
        finalize_run_directory(run_dir, status="failed")
        raise failure
    finalize_run_directory(run_dir, status=str(result["status"]))
    return run_dir, result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir, result = asyncio.run(_run(args.run_id))
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "status": result["status"],
                "health_pass": result["health_pass"],
                "candidate_count": result["candidate_count"],
                "provider_calls": result["provider_calls"],
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

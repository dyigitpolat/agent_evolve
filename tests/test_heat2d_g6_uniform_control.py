"""Provider/PDE-free qualification of the matched Heat2D G6 control."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from agent_evolve.agentic import PhenotypeIdentity, typed_json_sha256
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (
    create_multiobjective_benchmark,
)
from examples.development import run_heat2d_generic_campaign as agentic
from examples.development import run_heat2d_generic_uniform_control as control
from examples.development.durable_run_artifacts import DurableJsonlJournal


class _NoPDEEvaluator:
    def __init__(self) -> None:
        self.preflight_calls = 0
        self.evaluate_calls = 0

    def preflight(self) -> dict[str, object]:
        self.preflight_calls += 1
        return {
            "schema_version": 1,
            "status": "test_provider_and_pde_free",
            "pde_solves": 0,
        }

    def evaluate(self, config: object) -> object:  # pragma: no cover - fail path.
        del config
        self.evaluate_calls += 1
        raise AssertionError("the control preparation probe attempted a PDE solve")


class _FastTestPhenotypeIdentity:
    """Cheap test projection; production keeps the exact dense-field identity."""

    policy_id = "heat2d_test_typed_configuration"
    policy_version = 1

    def identify(self, configuration: object) -> PhenotypeIdentity:
        return PhenotypeIdentity(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            value_sha256=typed_json_sha256(configuration),
        )


def test_v11_parent_selector_conformance_is_checked_before_live_work() -> None:
    selector = agentic.StagnationAwareDiverseCampaignParentSelector()
    profile = SimpleNamespace(
        method_version=11,
        parent_selection=SimpleNamespace(implementation=selector),
    )

    assert agentic._reference_parent_selector(profile) is selector


def test_g6_treatment_preparation_constructs_all_reflections_without_provider_or_pde(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluator = _NoPDEEvaluator()

    def benchmark_factory(settings):
        benchmark = create_multiobjective_benchmark(
            settings,
            evaluator=evaluator,
        )
        return replace(
            benchmark,
            phenotype_identity=_FastTestPhenotypeIdentity(),
        )

    monkeypatch.setattr(
        agentic,
        "create_multiobjective_benchmark",
        benchmark_factory,
    )
    journal = DurableJsonlJournal(tmp_path / "treatment_preparation.jsonl")
    try:
        bundle = agentic._prepare_bundle(
            run_dir=tmp_path / "treatment_run",
            run_id="heat-treatment-preparation-test",
            preparation_journal=journal,
            source_closure_sha256="a" * 64,
        )
        readiness = agentic._eligibility_probe(bundle)
        probe = agentic._calibrated_all_wave_probe(bundle)
    finally:
        journal.close()

    assert evaluator.preflight_calls == 1
    assert evaluator.evaluate_calls == 0
    assert readiness["gate_under_60_process_cpu_s_each"] is True
    assert all(
        row["first_bind_process_cpu_s"] >= 0.0
        and row["first_bind_wall_s"] >= 0.0
        for row in readiness["parents"]
    )
    assert probe["provider_calls"] == 0
    assert probe["credential_read"] is False
    assert probe["pde_solves"] == 0
    assert probe["all_reflection_construction_gates_pass"] is True
    reflection = probe["reflection_construction_probe"]
    assert reflection["constructed_reflection_request_count"] == 1
    assert reflection["planned_source_generations"] == [2]
    assert reflection["sealed_source_portfolio_generations"] == [1]
    assert reflection["promotion_barrier_generations"] == [4]
    assert reflection["first_consumer_generation"] == 5
    assert reflection["terminal_reflection"] is False
    assert reflection["all_acceptance_gates_pass"] is True
    assert all(
        row["full_contrast_ids_exposed_to_model"] is False
        and row["source_portfolio_generation"] == 1
        for row in reflection["rows"]
    )
    schedule = bundle.prepared.schedule
    assert tuple(step.planned_agent_calls for step in schedule.steps) == (
        2,
        1,
        2,
        0,
        2,
        0,
    )
    assert schedule.planned_agent_calls == 7
    assert tuple(
        (wave.source_generation, wave.promotion_barrier_generation)
        for wave in schedule.reflection_waves
    ) == ((2, 4),)
    assert tuple(
        (barrier.generation, barrier.reflection_source_generations)
        for barrier in schedule.promotion_barriers
    ) == ((4, (2,)),)


def test_g6_control_prepares_and_selects_all_waves_without_provider_or_pde(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluator = _NoPDEEvaluator()

    def benchmark_factory(settings):
        benchmark = create_multiobjective_benchmark(
            settings,
            evaluator=evaluator,
        )
        return replace(
            benchmark,
            phenotype_identity=_FastTestPhenotypeIdentity(),
        )

    monkeypatch.setattr(
        agentic,
        "create_multiobjective_benchmark",
        benchmark_factory,
    )
    journal = DurableJsonlJournal(tmp_path / "preparation.jsonl")
    try:
        bundle = control._prepare_bundle(
            run_dir=tmp_path / "run",
            preparation_journal=journal,
            source_closure_sha256="a" * 64,
        )
        readiness = control._eligibility_probe(bundle)
        probe = asyncio.run(control._control_all_wave_probe(bundle))
    finally:
        journal.close()

    assert evaluator.preflight_calls == 1
    assert evaluator.evaluate_calls == 0
    assert readiness["provider_calls"] == 0
    assert readiness["pde_solves"] == 0
    assert readiness["gate_under_60_process_cpu_s_each"] is True
    assert readiness["all_acceptance_gates_pass"] is True

    assert probe["provider_calls"] == 0
    assert probe["actual_llm_calls"] == 0
    assert probe["credential_read"] is False
    assert probe["pde_solves"] == 0
    assert probe["constructed_wave_count"] == 6
    assert probe["selected_wave_count"] == 6
    assert probe["exact_expected_wave_count"] is True
    assert probe["all_request_hashes_unique"] is True
    assert probe["all_decision_hashes_unique"] is True
    assert probe["all_slate_selections_feasible"] is True
    assert probe["parent_measurement_bound_every_wave"] is True
    assert probe["policy_read_sets_empty"] is True
    assert probe["all_acceptance_gates_pass"] is True
    assert probe["planned_stage_unique_evaluation_counts"] == [16, 4, 16, 4, 16, 4]
    assert probe["exact_stage_widths"] is True
    assert probe["objective_resolution_matches_treatment"] is True
    assert probe["parent_policy_matches_treatment"] is True
    assert probe["recombination_policy_matches_treatment"] is True
    assert probe["archive_utility_matches_treatment"] is True
    assert all(
        row["resolved_provider"] == "local-deterministic-control"
        for row in probe["rows"]
    )
    assert all(row["evaluation_width"] == 8 for row in probe["rows"])

    schedule = bundle.prepared.schedule
    assert schedule.portfolio_generations == agentic.PORTFOLIO_GENERATIONS
    assert schedule.paired_recombination_generations == (
        agentic.RECOMBINATION_GENERATIONS
    )
    assert (
        schedule.planned_candidate_evaluations + len(bundle.prepared.seeds.seeds) == 62
    )
    assert (
        bundle.policies.parent_selection.to_record()
        == agentic._binding(
            "archive_elite_explorer",
            control.ArchiveEliteExplorerCampaignParentSelector(),
        ).to_record()
    )
    assert bundle.utility.definition_sha256 == agentic._affine_spec().definition_sha256
    assert (
        bundle.benchmark.objective_resolution.to_record()
        == agentic._objective_resolution().to_record()
    )
    assert (
        bundle.parent_measurement_projection
        == agentic._parent_measurement_projection(
            bundle.prepared,
            bundle.benchmark,
        )
    )


def test_control_stage_width_contract_tracks_common_pool_evaluation_width(
    monkeypatch,
) -> None:
    monkeypatch.setattr(agentic, "PORTFOLIO_WIDTH", 4)

    assert control._expected_stage_widths() == [8, 4, 8, 4, 8, 4]

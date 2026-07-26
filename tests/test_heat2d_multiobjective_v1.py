"""Provider- and PDE-free tests for the Heat2D Pareto-v1 formulation."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from agent_evolve.agentic import AgenticBenchmark
from examples.benchmarks.heat2d_constructive.agentic_benchmark import (
    benchmark as historical_scalar_benchmark,
)
from examples.benchmarks.heat2d_constructive.candidate import (
    SEED_LAYOUT_A,
    SEED_LAYOUT_B,
    normalize_candidate,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (
    FORMULATION_DEFINITION_SHA256,
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
    WORKLOAD_ID,
    Heat2DMultiObjectiveV1Problem,
    create_multiobjective_benchmark,
    prepare_multiobjective_benchmark,
)
from examples.benchmarks.heat2d_constructive.problem_def import (
    EVALUATOR_ID,
    OBJECTIVE_NAME,
    DirectV3ContractError,
    Heat2DConstructiveProblem,
    Heat2DDirectV3Evaluation,
    Heat2DDirectV3Settings,
)


class _ProviderFreeDirectV3:
    evaluator_id = EVALUATOR_ID
    evaluator_concurrency = 1

    def __init__(self, settings: Heat2DDirectV3Settings) -> None:
        self.settings = settings
        self.evaluate_calls = 0
        self.preflight_calls = 0
        self.exact_identity_matches = True

    def preflight(self) -> dict[str, object]:
        self.preflight_calls += 1
        return {
            "runner_sha256": "1" * 64,
            "resolution": self.settings.resolution,
            "external_concurrency": 1,
            "provider_free": True,
        }

    def evaluate(self, config: object) -> Heat2DDirectV3Evaluation:
        self.evaluate_calls += 1
        candidate = normalize_candidate(config)
        cells = self.settings.resolution - 1
        mesh_denominator = 6 * cells**2
        denominator = mesh_denominator * (1 << 1074)
        fraction = Fraction.from_float(candidate.material_fraction)
        scaled = fraction * denominator
        assert scaled.denominator == 1
        numerator = scaled.numerator
        exact_material = float(Fraction(numerator, denominator))
        thermal = 0.0006 - 0.0005 * candidate.material_fraction
        regularization = 0.000001
        combined = thermal + regularization
        digest = hashlib.sha256(
            json.dumps(
                candidate.model_dump(mode="python"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        manifest = {
            "schema_version": 3,
            "evaluator_id": EVALUATOR_ID,
            "all_checks_pass": True,
            "full_pde_solve_count": 1,
            "checks": {
                "exact_cross_runtime_fe_volume_identity_matches": (
                    self.exact_identity_matches
                )
            },
            "volume_agreement": {
                "exact_identity_matches": self.exact_identity_matches,
                "admission_policy": ("unit_square_cg1_binary64_exact_volume_identity"),
                "admission_policy_version": 1,
            },
            "container_result": {
                "result": {
                    OBJECTIVE_NAME: combined,
                    THERMAL_OBJECTIVE_NAME: thermal,
                    "regularization_term": regularization,
                    "decomposition_residual": 0.0,
                },
                "exact_volume_contract": {
                    "schema_version": 1,
                    "policy_id": ("unit_square_cg1_binary64_exact_volume_identity"),
                    "policy_version": 1,
                    "resolution": self.settings.resolution,
                    "cells_per_axis": cells,
                    "binary64_common_denominator_exponent": 1074,
                    "mesh_mass_denominator": mesh_denominator,
                    "exact_scaled_numerator_decimal": str(numerator),
                    "exact_scaled_numerator_sha256": "2" * 64,
                    "contract_sha256": "3" * 64,
                },
            },
        }
        return Heat2DDirectV3Evaluation(
            objective_values={OBJECTIVE_NAME: combined},
            output_dir=Path("/tmp/provider-free-heat2d-pareto"),
            genotype_sha256=digest,
            phenotype_sha256=digest,
            raw_array_sha256=digest,
            representation_spec_sha256=digest,
            finite_element_volume=exact_material,
            grayness=0.1,
            gray_fraction_005_095=0.2,
            adapter_elapsed_s=0.0,
            evaluator_elapsed_s=0.0,
            elapsed_inside_container_s=0.0,
            queue_wait_s=0.0,
            peak_rss_bytes=1,
            manifest=manifest,
        )


def _fixture() -> tuple[
    Heat2DDirectV3Settings,
    _ProviderFreeDirectV3,
    AgenticBenchmark,
]:
    settings = Heat2DDirectV3Settings(
        output_root=Path("/tmp/provider-free-heat2d-pareto"),
        resolution=41,
    )
    evaluator = _ProviderFreeDirectV3(settings)
    benchmark = create_multiobjective_benchmark(settings, evaluator=evaluator)
    return settings, evaluator, benchmark


def test_new_formulation_is_two_objective_and_does_not_mutate_scalar_v3() -> None:
    _, _, benchmark = _fixture()
    assert type(benchmark.problem) is Heat2DMultiObjectiveV1Problem
    assert benchmark.problem.workload_id == WORKLOAD_ID
    assert len(FORMULATION_DEFINITION_SHA256) == 64
    assert tuple((item.name, item.goal) for item in benchmark.objectives) == (
        (THERMAL_OBJECTIVE_NAME, "min"),
        (MATERIAL_OBJECTIVE_NAME, "min"),
    )
    assert tuple(
        (item.name, item.goal) for item in historical_scalar_benchmark.objectives
    ) == ((OBJECTIVE_NAME, "min"),)
    assert benchmark.problem.settings.external_concurrency == 1


def test_two_realistic_seeds_form_a_thermal_material_tradeoff() -> None:
    _, evaluator, benchmark = _fixture()
    first = benchmark.problem.evaluate(SEED_LAYOUT_A)
    second = benchmark.problem.evaluate(SEED_LAYOUT_B)
    assert evaluator.evaluate_calls == 2
    assert first[THERMAL_OBJECTIVE_NAME] < second[THERMAL_OBJECTIVE_NAME]
    assert first[MATERIAL_OBJECTIVE_NAME] > second[MATERIAL_OBJECTIVE_NAME]
    assert first[MATERIAL_OBJECTIVE_NAME] == 0.45
    assert second[MATERIAL_OBJECTIVE_NAME] == 0.38


def test_projection_fails_closed_when_exact_volume_identity_is_absent() -> None:
    settings, evaluator, _ = _fixture()
    evaluator.exact_identity_matches = False
    problem = Heat2DMultiObjectiveV1Problem(settings, evaluator=evaluator)
    with pytest.raises(DirectV3ContractError, match="exact cross-runtime"):
        problem.evaluate(SEED_LAYOUT_A)


def test_prepare_is_provider_and_pde_free_and_binds_two_diverse_seeds() -> None:
    pytest.importorskip("numpy")
    _, evaluator, benchmark = _fixture()
    receipt = prepare_multiobjective_benchmark(benchmark)
    assert receipt["status"] == "provider_and_pde_free_prepared"
    assert receipt["provider_calls"] == 0
    assert receipt["pde_solves"] == 0
    assert evaluator.preflight_calls == 1
    assert evaluator.evaluate_calls == 0
    assert len(receipt["receipt_sha256"]) == 64
    seeds = receipt["seeds"]
    assert type(seeds) is list and len(seeds) == 2
    assert {seed["requested_material_fraction"] for seed in seeds} == {0.38, 0.45}
    assert len({seed["candidate_key"] for seed in seeds}) == 2
    assert min(seed["finite_option_count"] for seed in seeds) >= 136


def test_candidate_identity_is_formulation_scoped_and_catalog_is_workload_local() -> (
    None
):
    settings, evaluator, benchmark = _fixture()
    pareto_problem = benchmark.problem
    scalar_problem = Heat2DConstructiveProblem(settings, evaluator=evaluator)
    assert pareto_problem.candidate_key(SEED_LAYOUT_A) != scalar_problem.candidate_key(
        SEED_LAYOUT_A
    )
    contract = benchmark.bind_finite_variation(
        "heat2d_constructive_scalar_grid", SEED_LAYOUT_A
    )
    assert len(contract.options) >= 136
    assert "Pareto co-optimization" in pareto_problem.search_space_description()

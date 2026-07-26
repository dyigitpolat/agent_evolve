"""Provider-free conformance for the public Timeloop v2 benchmark port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog
from examples.benchmarks.timeloop_codesign.v2.agentic_benchmark import (
    benchmark,
    finite_variation_catalog,
)
from examples.benchmarks.timeloop_codesign.v2.candidate import DEFAULT_CANDIDATE
from examples.benchmarks.timeloop_codesign.v2.evaluator import TimeloopV2Settings
from examples.benchmarks.timeloop_codesign.v2.finite_variation_catalog import CATALOG_ID
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import frozen_network_panel
from examples.benchmarks.timeloop_codesign.v2.problem_def import (
    TimeloopV2CoDesignProblem,
)


@dataclass(frozen=True)
class _FakeEvaluation:
    objective_values: dict[str, float]


class _FakeEvaluator:
    def evaluate(self, config: object) -> _FakeEvaluation:
        return _FakeEvaluation(
            objective_values={
                "energy_joules": 1.0,
                "latency_seconds": 2.0,
                "area_square_meters": 3.0,
            }
        )


def test_public_benchmark_binds_actual_panel_objectives_and_61_moves() -> None:
    assert isinstance(benchmark, AgenticBenchmark)
    assert isinstance(finite_variation_catalog, FiniteVariationCatalog)
    assert tuple((item.name, item.goal) for item in benchmark.objectives) == (
        ("energy_joules", "min"),
        ("latency_seconds", "min"),
        ("area_square_meters", "min"),
    )
    contract = benchmark.bind_finite_variation(CATALOG_ID, DEFAULT_CANDIDATE)
    assert len(contract.options) == 61
    assert len({item.child_configuration_sha256 for item in contract.options}) == 61
    description = benchmark.problem.search_space_description()
    assert "695,784,701,952" in description
    assert "resnet50 (calibration)" in description


def test_problem_accepts_an_injected_evaluator_without_docker(tmp_path: Path) -> None:
    problem = TimeloopV2CoDesignProblem(
        TimeloopV2Settings(output_root=tmp_path),
        frozen_network_panel("resnet50"),
        evaluator=_FakeEvaluator(),
    )
    assert problem.validate(DEFAULT_CANDIDATE) is True
    assert problem.evaluate(DEFAULT_CANDIDATE) == {
        "energy_joules": 1.0,
        "latency_seconds": 2.0,
        "area_square_meters": 3.0,
    }
    rendered = problem.render_candidate(DEFAULT_CANDIDATE)
    assert "m0=row_stationary/M/medium/balanced/channel_then_spatial" in rendered
    assert "400 valid mappings" in problem.search_space_description()

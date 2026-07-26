"""Provider-free conformance for the Timeloop co-design benchmark."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_evolve.agentic import AgenticBenchmark, FiniteVariationCatalog, thaw_json
from examples.benchmarks.timeloop_codesign.agentic_benchmark import (
    benchmark,
    finite_variation_catalog,
)
from examples.benchmarks.timeloop_codesign.candidate import (
    DEFAULT_CANDIDATE,
    CandidateConfig,
    candidate_sha256,
    normalize_candidate,
    seed_candidates,
)
from examples.benchmarks.timeloop_codesign.container_runner import (
    ASSET_SHA256,
    EVALUATOR_ID,
    MAPPER_ALGORITHM,
    MAPPER_THREADS,
    OPTIMIZATION_METRICS,
    SEARCH_SIZE,
    _candidate_sha256 as runner_candidate_sha256,
    _validate_candidate as runner_validate_candidate,
)
from examples.benchmarks.timeloop_codesign.finite_variation_catalog import (
    CATALOG_DEFINITION_SHA256,
    CATALOG_ID,
    FIELD_GRIDS,
)
from examples.benchmarks.timeloop_codesign.problem_def import (
    OBJECTIVE_NAMES,
    TimeloopCoDesignProblem,
    TimeloopContractError,
    TimeloopDockerEvaluator,
    TimeloopEvaluation,
    TimeloopSettings,
)


def test_candidate_schema_is_closed_strict_and_schema_scoped() -> None:
    default, compute_heavy, memory_heavy = seed_candidates()
    assert default.model_dump(mode="python") == DEFAULT_CANDIDATE
    assert len({candidate_sha256(item) for item in seed_candidates()}) == 3
    assert compute_heavy.pe_mesh_x > default.pe_mesh_x > memory_heavy.pe_mesh_x

    invalid = (
        {**DEFAULT_CANDIDATE, "pe_mesh_x": "8"},
        {**DEFAULT_CANDIDATE, "pe_mesh_x": True},
        {**DEFAULT_CANDIDATE, "global_buffer_depth": 768},
        {**DEFAULT_CANDIDATE, "register_enabled": 1},
        {**DEFAULT_CANDIDATE, "yaml": "!include /etc/passwd"},
    )
    for value in invalid:
        with pytest.raises(ValidationError):
            normalize_candidate(value)


def test_container_and_host_candidate_boundaries_agree() -> None:
    candidate = normalize_candidate(DEFAULT_CANDIDATE)
    payload = candidate.model_dump(mode="python")
    assert runner_validate_candidate(payload) == payload
    assert runner_candidate_sha256(payload) == candidate_sha256(candidate)
    with pytest.raises(ValueError, match="missing or extra"):
        runner_validate_candidate({**payload, "command": "rm -rf /"})


def test_public_benchmark_binds_three_objectives_and_parent_catalog() -> None:
    assert isinstance(benchmark, AgenticBenchmark)
    assert isinstance(finite_variation_catalog, FiniteVariationCatalog)
    assert tuple((item.name, item.goal) for item in benchmark.objectives) == tuple(
        (name, "min") for name in OBJECTIVE_NAMES
    )
    assert benchmark.finite_variation_catalog_identities == (
        (CATALOG_ID, 1, CATALOG_DEFINITION_SHA256),
    )

    contract = benchmark.bind_finite_variation(CATALOG_ID, DEFAULT_CANDIDATE)
    assert len(contract.options) == 10
    assert len({item.child_configuration_sha256 for item in contract.options}) == 10
    counts = Counter(dict(item.metadata)["field"] for item in contract.options)
    assert counts == {
        field: len(values) - 1 for field, values, _ in FIELD_GRIDS
    }
    parent = normalize_candidate(DEFAULT_CANDIDATE).model_dump(mode="python")
    for option in contract.options:
        child = normalize_candidate(thaw_json(option.child_configuration))
        changed = {
            field
            for field, value in child.model_dump(mode="python").items()
            if value != parent[field]
        }
        assert changed == {dict(option.metadata)["field"]}
    assert (
        benchmark.bind_finite_variation(CATALOG_ID, DEFAULT_CANDIDATE).identity_sha256
        == contract.identity_sha256
    )


class _FakeEvaluator:
    def evaluate(self, config: object) -> TimeloopEvaluation:
        candidate = normalize_candidate(config)
        digest = candidate_sha256(candidate)
        return TimeloopEvaluation(
            objective_values={
                "energy_joules": float(candidate.global_buffer_depth),
                "latency_seconds": 1.0 / float(candidate.pe_mesh_x),
                "area_square_meters": float(candidate.global_buffer_width),
            },
            output_dir=Path("/tmp/provider-free-timeloop"),
            candidate_sha256=digest,
            mapping_sha256="a" * 64,
            evaluator_elapsed_s=1.0,
            elapsed_inside_container_s=0.9,
            queue_wait_s=0.0,
            cycles=1,
            computes=1,
            manifest={},
        )


def test_problem_accepts_an_injected_evaluator_without_docker(tmp_path: Path) -> None:
    problem = TimeloopCoDesignProblem(
        TimeloopSettings(output_root=tmp_path),
        evaluator=_FakeEvaluator(),
    )
    assert problem.validate(DEFAULT_CANDIDATE) is True
    assert problem.evaluate(DEFAULT_CANDIDATE) == {
        "energy_joules": 512.0,
        "latency_seconds": 0.125,
        "area_square_meters": 128.0,
    }
    assert "2,000 valid mappings" in problem.search_space_description()


def _valid_manifest(evaluator: TimeloopDockerEvaluator) -> dict[str, object]:
    candidate = normalize_candidate(DEFAULT_CANDIDATE)
    return {
        "schema_version": 1,
        "evaluator_id": EVALUATOR_ID,
        "candidate": candidate.model_dump(mode="python"),
        "candidate_sha256": candidate_sha256(candidate),
        "objectives": {
            "energy_joules": 0.063,
            "latency_seconds": 1.07,
            "area_square_meters": 6.7e-8,
        },
        "diagnostics": {
            "elapsed_s": 14.0,
            "cycles": 1_073_741_824,
            "computes": 8_589_934_592,
            "mapping_sha256": "b" * 64,
        },
        "protocol": {
            "search_size": SEARCH_SIZE,
            "mapper_threads": MAPPER_THREADS,
            "mapper_algorithm": MAPPER_ALGORITHM,
            "optimization_metrics": list(OPTIMIZATION_METRICS),
        },
        "provenance": {
            "asset_sha256": ASSET_SHA256,
            "runner_sha256": evaluator._runner_sha256,
        },
    }


def test_result_validation_fails_closed_on_protocol_or_provenance_drift(
    tmp_path: Path,
) -> None:
    evaluator = TimeloopDockerEvaluator(TimeloopSettings(output_root=tmp_path))
    candidate = CandidateConfig()
    manifest = _valid_manifest(evaluator)
    result = evaluator._validate_result(
        candidate,
        tmp_path,
        manifest,
        evaluator_elapsed_s=14.1,
        queue_wait_s=0.0,
    )
    assert result.objective_values["energy_joules"] == 0.063

    altered = _valid_manifest(evaluator)
    altered["protocol"] = {**altered["protocol"], "search_size": 1_999}
    with pytest.raises(TimeloopContractError, match="protocol drift"):
        evaluator._validate_result(
            candidate,
            tmp_path,
            altered,
            evaluator_elapsed_s=14.1,
            queue_wait_s=0.0,
        )

    altered = _valid_manifest(evaluator)
    altered["provenance"] = {**altered["provenance"], "runner_sha256": "0" * 64}
    with pytest.raises(TimeloopContractError, match="runner provenance drift"):
        evaluator._validate_result(
            candidate,
            tmp_path,
            altered,
            evaluator_elapsed_s=14.1,
            queue_wait_s=0.0,
        )

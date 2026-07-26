"""Provider- and PDE-free conformance for Heat's narrow WorkloadKit adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent_evolve.application.evolution_campaign import BenchmarkSessionRequest
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from examples.benchmarks.heat2d_constructive.campaign_workload import (
    CAMPAIGN_WORKLOAD_ID,
    compose_heat2d_pareto_campaign_workload,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (
    create_multiobjective_benchmark,
)
from examples.benchmarks.heat2d_constructive.problem_def import (
    EVALUATOR_ID,
    Heat2DDirectV3Settings,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


class _ForbiddenEvaluator:
    evaluator_id = EVALUATOR_ID
    evaluator_concurrency = 1

    def __init__(self, settings: Heat2DDirectV3Settings) -> None:
        self.settings = settings
        self.preflight_calls = 0
        self.evaluate_calls = 0

    def preflight(self):
        self.preflight_calls += 1
        raise AssertionError("workload composition must not run Heat preflight")

    def evaluate(self, candidate):
        del candidate
        self.evaluate_calls += 1
        raise AssertionError("workload composition must not run a Heat PDE solve")


def _fixture(tmp_path: Path):
    settings = Heat2DDirectV3Settings(
        output_root=tmp_path,
        resolution=41,
        external_concurrency=1,
    )
    evaluator = _ForbiddenEvaluator(settings)
    benchmark = create_multiobjective_benchmark(settings, evaluator=evaluator)
    kit = compose_heat2d_pareto_campaign_workload(
        benchmark=benchmark,
        evaluator_preflight_receipt=_object(
            {"qualified": True, "pde_solves": 0, "provider_calls": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "serialized_heat2d", "active": True}
        ),
    )
    return evaluator, kit


def test_heat_workload_kit_builds_real_catalog_without_pde_or_provider(
    tmp_path: Path,
) -> None:
    evaluator, kit = _fixture(tmp_path)
    config = kit.to_campaign_workload()
    ports = config.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("heat-workload-kit-protocol"),
            budget_sha256=_sha("heat-workload-kit-budget"),
            outer_seed=20260719,
            requested_evaluator_concurrency=1,
        )
    )
    seeds = ports.seeds.load(session)
    receipt = thaw_json(kit.integration_receipt())

    assert kit.workload_id == CAMPAIGN_WORKLOAD_ID
    assert len(seeds.seeds) == 2
    assert config.selected_catalog_identity[0] == (
        "heat2d_constructive_scalar_grid"
    )
    assert receipt["required_obligation_count"] == 4
    assert receipt["uses_default_schema_evidence"] is True
    assert evaluator.preflight_calls == 0
    assert evaluator.evaluate_calls == 0


def test_heat_workload_materializes_real_catalog_provider_and_pde_free(
    tmp_path: Path,
) -> None:
    pytest.importorskip("numpy")
    evaluator, kit = _fixture(tmp_path)
    config = kit.to_campaign_workload()
    ports = config.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("heat-workload-kit-catalog-protocol"),
            budget_sha256=_sha("heat-workload-kit-catalog-budget"),
            outer_seed=20260719,
            requested_evaluator_concurrency=1,
        )
    )
    seeds = ports.seeds.load(session)
    parent = seeds.seeds[0].configuration
    variation = ports.catalog.bind(session.benchmark, parent, ())
    memory = ports.evidence.initialize_memory(session, seeds)
    context = thaw_json(
        ports.evidence.context(session, parent, variation, memory)
    )

    assert len(variation.contract.options) >= 136
    assert context["projection_id"] == "generic_schema_evidence"
    assert context["finite_variation"]["eligible_option_count"] >= 136
    assert evaluator.preflight_calls == 0
    assert evaluator.evaluate_calls == 0


def test_heat_workload_rejects_concurrency_beyond_qualified_cap(
    tmp_path: Path,
) -> None:
    evaluator, kit = _fixture(tmp_path)
    with pytest.raises(ValueError, match="qualified Heat2D external cap"):
        compose_heat2d_pareto_campaign_workload(
            benchmark=kit.benchmark,
            evaluator_preflight_receipt=kit.evaluator_preflight_receipt,
            resource_lease_receipt=kit.resource_lease_receipt,
            evaluator_concurrency_cap=2,
        )
    assert evaluator.preflight_calls == 0
    assert evaluator.evaluate_calls == 0

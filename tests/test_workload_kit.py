"""Provider-free conformance for the narrow workload integration API."""

from __future__ import annotations

import hashlib

import pytest

from agent_evolve import WorkloadKit, campaign_seed
from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.evolution_campaign import BenchmarkSessionRequest
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.evaluator import AbcEvaluatorSettings
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem
from examples.benchmarks.timeloop_codesign.v2.campaign_workload import (
    compose_timeloop_v2_campaign_workload,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _boils_kit() -> tuple[BoilsAbcProblem, WorkloadKit]:
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=((0,),),
        per_circuit_timeout_s=60.0,
    )
    problem = BoilsAbcProblem(settings)
    benchmark = AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    kit = WorkloadKit(
        workload_id="boils_kit_conformance",
        workload_version=1,
        benchmark=benchmark,
        seeds=(
            campaign_seed(
                "seed_default",
                {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
            ),
        ),
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=_object(
            {"qualified": True, "provider_calls": 0, "abc_executions": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "one_pinned_cpu_affinity", "active": True}
        ),
    )
    return problem, kit


def _exercise_provider_free_kit(kit: WorkloadKit) -> dict[str, object]:
    config = kit.to_campaign_workload()
    ports = config.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha(f"protocol:{kit.workload_id}"),
            budget_sha256=_sha(f"budget:{kit.workload_id}"),
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
    assert ports.evidence.cards(session, parent, variation, memory) == ()
    return context


def test_workload_kit_compiles_boils_with_four_required_obligations() -> None:
    problem, kit = _boils_kit()

    context = _exercise_provider_free_kit(kit)
    receipt = thaw_json(kit.integration_receipt())

    assert kit.selected_finite_catalog_id == FINITE_CATALOG_ID
    assert context["projection_id"] == "generic_schema_evidence"
    assert context["workload_id"] == kit.workload_id
    assert context["finite_variation"]["eligible_option_count"] == 200
    assert tuple(value["name"] for value in context["objectives"]) == (
        "total_lut_count",
        "total_levels",
    )
    assert receipt["required_obligation_count"] == 4
    assert receipt["uses_default_schema_evidence"] is True
    assert receipt["uses_custom_evidence_projection"] is False
    assert receipt["uses_optional_prompt_extension"] is False
    assert problem._evaluator is None


def test_workload_kit_compiles_existing_timeloop_benchmark_without_custom_evidence(
) -> None:
    existing = compose_timeloop_v2_campaign_workload(
        evaluator_preflight_receipt=_object(
            {"qualified": True, "timeloop_executions": 0, "provider_calls": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "serialized_timeloop", "active": True}
        ),
    )
    kit = WorkloadKit(
        workload_id="timeloop_kit_conformance",
        workload_version=1,
        benchmark=existing.benchmark,
        seeds=existing.seeds,
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=existing.evaluator_preflight_receipt,
        resource_lease_receipt=existing.resource_lease_receipt,
    )

    context = _exercise_provider_free_kit(kit)

    assert context["workload_id"] == kit.workload_id
    assert context["finite_variation"]["eligible_option_count"] > 0
    assert len(context["objectives"]) >= 2


def test_workload_kit_requires_explicit_catalog_when_selection_is_ambiguous() -> None:
    problem, base = _boils_kit()
    duplicate = BoilsFiniteVariationCatalog()
    # The benchmark itself rejects duplicate IDs before the WorkloadKit can
    # select one, preserving one unambiguous catalog authority.
    with pytest.raises(ValueError, match="catalog IDs must be unique"):
        AgenticBenchmark(
            problem=problem,
            finite_variation_catalogs=(duplicate, duplicate),
        )

    with pytest.raises(ValueError, match="zero or multiple catalogs"):
        WorkloadKit(
            workload_id="missing_catalog",
            workload_version=1,
            benchmark=AgenticBenchmark(problem=problem),
            seeds=base.seeds,
            evaluator_concurrency_cap=1,
            evaluator_preflight_receipt=base.evaluator_preflight_receipt,
            resource_lease_receipt=base.resource_lease_receipt,
        )

"""Provider/evaluator-free conformance for BOiLS's campaign boundary."""

from __future__ import annotations

import hashlib

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.evolution_campaign import BenchmarkSessionRequest
from agent_evolve.application.portfolio_evolution import MEMORY_ESTIMAND_CONTEXT_KEY
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from examples.benchmarks.boils_abc.campaign_workload import (
    WORKLOAD_ID,
    compose_boils_campaign_workload,
)
from examples.benchmarks.boils_abc.actions import ACTION_IDS, SEQUENCE_LENGTH
from examples.benchmarks.boils_abc.evaluator import AbcEvaluatorSettings
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def test_boils_composes_campaign_ports_without_abc_or_provider() -> None:
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=((0,), (1,)),
        per_circuit_timeout_s=60.0,
    )
    problem = BoilsAbcProblem(settings)
    benchmark = AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    assert problem._evaluator is None
    config = compose_boils_campaign_workload(
        benchmark=benchmark,
        evaluator_preflight_receipt=_object(
            {
                "qualified": True,
                "panel": ["log2"],
                "provider_calls": 0,
                "abc_executions": 0,
            }
        ),
        resource_lease_receipt=_object(
            {"resource": "two_disjoint_cpu_affinity_slots", "active": True}
        ),
        evaluator_concurrency_cap=2,
    )
    assert config.workload_id == WORKLOAD_ID
    assert config.finite_catalog_id == FINITE_CATALOG_ID
    assert config.evaluator_concurrency_cap == 2
    ports = config.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("boils-provider-free-protocol"),
            budget_sha256=_sha("boils-provider-free-budget"),
            outer_seed=20260717,
            requested_evaluator_concurrency=2,
        )
    )
    seeds = ports.seeds.load(session)
    assert tuple(value.seed_id for value in seeds.seeds) == (
        "seed_default",
        "seed_parent_c",
    )

    parent = seeds.seeds[0].configuration
    variation = ports.catalog.bind(session.benchmark, parent, ())
    assert len(variation.contract.options) == SEQUENCE_LENGTH * (len(ACTION_IDS) - 1)
    memory = ports.evidence.initialize_memory(session, seeds)
    context = ports.evidence.context(session, parent, variation, memory)
    cards = ports.evidence.cards(session, parent, variation, memory)
    context_record = thaw_json(context)
    assert context_record["workload_id"] == WORKLOAD_ID
    assert context_record["evaluator_panel"]["circuit_names"] == ["log2"]
    assert context_record["finite_variation"]["eligible_option_count"] == 200
    assert context_record[MEMORY_ESTIMAND_CONTEXT_KEY] == {
        "schema_version": 1,
        "estimand_id": "boils_abc_reflected_card_intent_to_treat",
        "workload_id": WORKLOAD_ID,
        "evaluator_context_sha256": context_record["evaluator_panel"]["identity"][
            "evaluator_context_sha256"
        ],
        "finite_catalog_id": FINITE_CATALOG_ID,
        "assignment_unit": "one_parent_lane_portfolio_wave",
        "treatment": "one_selected_reflected_insight_card",
        "endpoint_authority": (
            "authenticated_portfolio_reward_aggregation_binding"
        ),
    }
    assert tuple(value["metric_id"] for value in context_record["objectives"]) == (
        "total_lut_count",
        "total_levels",
    )
    assert len(cards) == 3
    assert all(thaw_json(value)["workload_id"] == WORKLOAD_ID for value in cards)
    assert problem._evaluator is None


def test_boils_campaign_rejects_concurrency_beyond_pinned_affinities() -> None:
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=((0,),),
    )
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(settings),
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    try:
        compose_boils_campaign_workload(
            benchmark=benchmark,
            evaluator_preflight_receipt=_object({"qualified": True}),
            resource_lease_receipt=_object({"active": True}),
            evaluator_concurrency_cap=2,
        )
    except ValueError as error:
        assert "affinity" in str(error)
    else:  # pragma: no cover - fail-closed assertion.
        raise AssertionError("foreign evaluator concurrency was accepted")

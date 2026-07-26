"""Provider-free conformance for Timeloop v2's generic campaign boundary."""

from __future__ import annotations

import hashlib

from agent_evolve.application.evolution_campaign import BenchmarkSessionRequest
from agent_evolve.application.portfolio_evolution import MEMORY_ESTIMAND_CONTEXT_KEY
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from examples.benchmarks.timeloop_codesign.v2.campaign_workload import (
    WORKLOAD_ID,
    compose_timeloop_v2_campaign_workload,
)
from examples.benchmarks.timeloop_codesign.v2.finite_variation_catalog import CATALOG_ID
from examples.benchmarks.timeloop_codesign.v2.problem_def import problem


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def test_timeloop_v2_composes_campaign_ports_without_docker_or_provider() -> None:
    assert problem._evaluator is None
    config = compose_timeloop_v2_campaign_workload(
        evaluator_preflight_receipt=_object(
            {
                "qualified": True,
                "protocol": "three_medoid_400_valid_mappings",
                "provider_calls": 0,
            }
        ),
        resource_lease_receipt=_object(
            {"resource": "serialized_timeloop_container_slot", "active": True}
        ),
    )
    assert config.workload_id == WORKLOAD_ID
    assert config.finite_catalog_id == CATALOG_ID
    assert config.evaluator_concurrency_cap == 1
    ports = config.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("timeloop-v2-provider-free-protocol"),
            budget_sha256=_sha("timeloop-v2-provider-free-budget"),
            outer_seed=20260717,
            requested_evaluator_concurrency=1,
        )
    )
    seeds = ports.seeds.load(session)
    assert tuple(value.seed_id for value in seeds.seeds) == (
        "seed_default",
        "seed_pe_mesh_16",
    )
    assert thaw_json(seeds.seeds[0].configuration)["pe_mesh_x"] == 8
    assert thaw_json(seeds.seeds[1].configuration)["pe_mesh_x"] == 16

    parent = seeds.seeds[0].configuration
    variation = ports.catalog.bind(session.benchmark, parent, ())
    assert len(variation.contract.options) == 61
    memory = ports.evidence.initialize_memory(session, seeds)
    context = ports.evidence.context(session, parent, variation, memory)
    cards = ports.evidence.cards(session, parent, variation, memory)
    context_record = thaw_json(context)
    assert context_record["workload_id"] == WORKLOAD_ID
    assert context_record["network_panel"]["network_id"] == "resnet50"
    assert context_record["network_panel"]["medoid_multiplicities"] == [19, 14, 20]
    assert context_record["finite_variation"]["eligible_option_count"] == 61
    assert context_record[MEMORY_ESTIMAND_CONTEXT_KEY] == {
        "schema_version": 1,
        "estimand_id": "timeloop_v2_reflected_card_intent_to_treat",
        "workload_id": WORKLOAD_ID,
        "network_panel_sha256": context_record["network_panel"]["panel_sha256"],
        "finite_catalog_id": CATALOG_ID,
        "assignment_unit": "one_parent_lane_portfolio_wave",
        "treatment": "one_selected_reflected_insight_card",
        "endpoint_authority": ("authenticated_portfolio_reward_aggregation_binding"),
    }
    assert tuple(value["metric_id"] for value in context_record["objectives"]) == (
        "energy_joules",
        "latency_seconds",
        "area_square_meters",
    )
    assert len(cards) == 3
    assert all(thaw_json(value)["workload_id"] == WORKLOAD_ID for value in cards)
    assert problem._evaluator is None

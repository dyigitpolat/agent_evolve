"""Campaign-workload composition for the qualified Timeloop v2 benchmark.

This module owns only Timeloop facts: the two seed configurations, frozen
ResNet50 calibration panel, exact finite catalog, evaluator admission facts,
and scientific prompt evidence.  Campaign chronology, model transport,
memory assignment, portfolio selection, and recombination remain injected by
the generic AgentEvolve application layer.
"""

from __future__ import annotations

import hashlib
import json

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.evolution_campaign import CampaignSeed
from agent_evolve.application.portfolio_evolution import MEMORY_ESTIMAND_CONTEXT_KEY
from agent_evolve.campaign_workload import (
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

from .agentic_benchmark import benchmark as default_benchmark
from .candidate import DEFAULT_CANDIDATE
from .finite_variation_catalog import CATALOG_ID
from .frozen_panels import frozen_network_panel
from .network_panel import panel_sha256


WORKLOAD_ID = "timeloop-codesign-v2-resnet50"
WORKLOAD_VERSION = 2
EVIDENCE_PROJECTION_ID = "timeloop_v2_scientific_evidence"
EVIDENCE_PROJECTION_VERSION = 2


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _definition_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("Timeloop campaign evidence did not freeze to an object")
    return frozen


_PANEL = frozen_network_panel("resnet50")
_PANEL_SHA256 = panel_sha256(_PANEL)
_PRIOR_INSIGHTS: tuple[dict[str, object], ...] = (
    {
        "card_id": "timeloop_v2.parallelism_tradeoff",
        "claim": (
            "Increasing PE parallelism can reduce latency while increasing chip area; "
            "energy depends on utilization and data movement."
        ),
        "trigger": "The parent leaves spatial parallelism or utilization headroom.",
        "mechanism": (
            "A wider physical mesh exposes more concurrent MACs, but only a compatible "
            "mapping policy converts the extra area into useful work."
        ),
        "affected_paths": ["$.pe_mesh_x", "$.policy_cluster_*"],
        "status": "prior_hypothesis_to_test",
    },
    {
        "card_id": "timeloop_v2.buffer_residency_coupling",
        "claim": (
            "Global-buffer geometry and residency policy should be varied jointly when "
            "off-chip movement dominates energy."
        ),
        "trigger": "A medoid policy repeatedly reloads a dominant tensor.",
        "mechanism": (
            "Capacity, width, and tensor residency jointly determine reuse and bandwidth; "
            "changing one without the others can move rather than remove a bottleneck."
        ),
        "affected_paths": [
            "$.global_buffer_depth",
            "$.global_buffer_width",
            "$.policy_cluster_*",
        ],
        "status": "prior_hypothesis_to_test",
    },
    {
        "card_id": "timeloop_v2.medoid_heterogeneity",
        "claim": (
            "The three outcome-blind network medoids may require different mapping "
            "policies under one shared architecture."
        ),
        "trigger": "Medoid shapes differ in channel or spatial extent.",
        "mechanism": (
            "Stationarity, spatial axis, and loop order expose shape-specific reuse, so "
            "one policy copied across all medoids can waste the available architecture."
        ),
        "affected_paths": [
            "$.policy_cluster_0",
            "$.policy_cluster_1",
            "$.policy_cluster_2",
        ],
        "status": "prior_hypothesis_to_test",
    },
)

EVIDENCE_PROJECTION_DEFINITION_SHA256 = _definition_hash(
    b"agent-evolve:timeloop-v2-campaign-evidence:v2\x00",
    {
        "projection_id": EVIDENCE_PROJECTION_ID,
        "projection_version": EVIDENCE_PROJECTION_VERSION,
        "panel_sha256": _PANEL_SHA256,
        "prior_insights": _PRIOR_INSIGHTS,
        "context_fields": [
            "network_panel",
            "objectives",
            "parent",
            "finite_variation",
            MEMORY_ESTIMAND_CONTEXT_KEY,
            "memory_sha256",
        ],
    },
)

WORKLOAD_DEFINITION_SHA256 = _definition_hash(
    b"agent-evolve:timeloop-v2-campaign-workload:v2\x00",
    {
        "workload_id": WORKLOAD_ID,
        "workload_version": WORKLOAD_VERSION,
        "panel_sha256": _PANEL_SHA256,
        "finite_catalog_id": CATALOG_ID,
        "seed_law": "default_and_single_pe_mesh_8_to_16_tradeoff",
        "evidence_projection_definition_sha256": (
            EVIDENCE_PROJECTION_DEFINITION_SHA256
        ),
        "evaluator_concurrency_cap": 1,
    },
)


def _initialize_memory(benchmark, session, seeds) -> FrozenJsonObject:
    del session
    return _object(
        {
            "schema_version": 1,
            "workload_id": WORKLOAD_ID,
            "network_panel_sha256": _PANEL_SHA256,
            "objective_names": [value.name for value in benchmark.objectives],
            "seed_candidate_keys": [
                benchmark.problem.candidate_key(thaw_json(seed.configuration))
                for seed in seeds.seeds
            ],
            "insights": list(_PRIOR_INSIGHTS),
        }
    )


def _context(benchmark, session, parent, variation, memory) -> FrozenJsonObject:
    del session
    return _object(
        {
            "schema_version": 1,
            "workload_id": WORKLOAD_ID,
            "network_panel": {
                "panel_id": _PANEL.panel_id,
                "network_id": _PANEL.network_id,
                "role": _PANEL.role,
                "panel_sha256": _PANEL_SHA256,
                "supported_conv_layer_count": _PANEL.supported_conv_layer_count,
                "medoid_multiplicities": [
                    value.multiplicity for value in _PANEL.medoids()
                ],
            },
            "objectives": [
                {"metric_id": value.name, "goal": value.goal}
                for value in benchmark.objectives
            ],
            "search_space": benchmark.problem.search_space_description(),
            "parent": {
                "configuration_sha256": variation.parent_configuration_sha256,
                "rendered": benchmark.problem.render_candidate(thaw_json(parent)),
            },
            "finite_variation": {
                "catalog_id": variation.contract.catalog_id,
                "contract_identity_sha256": variation.contract.identity_sha256,
                "eligible_option_count": len(variation.contract.options),
            },
            # Parent- and generation-local evidence remains in the full selector
            # context above.  Memory experiments use only this core-reserved,
            # workload-authored subtree as their repeated estimand stratum.  The
            # exact reward law is still authenticated independently by every
            # PortfolioRewardAggregationBinding, so campaign policies must hold
            # that binding fixed across one diagnostic cohort.
            MEMORY_ESTIMAND_CONTEXT_KEY: {
                "schema_version": 1,
                "estimand_id": "timeloop_v2_reflected_card_intent_to_treat",
                "workload_id": WORKLOAD_ID,
                "network_panel_sha256": _PANEL_SHA256,
                "finite_catalog_id": CATALOG_ID,
                "assignment_unit": "one_parent_lane_portfolio_wave",
                "treatment": "one_selected_reflected_insight_card",
                "endpoint_authority": (
                    "authenticated_portfolio_reward_aggregation_binding"
                ),
            },
            "memory_sha256": typed_json_sha256(memory),
        }
    )


def _cards(
    benchmark, session, parent, variation, memory
) -> tuple[FrozenJsonObject, ...]:
    del benchmark, session, parent
    record = thaw_json(memory)
    insights = record.get("insights")
    if type(insights) is not list or not insights:
        raise ValueError("Timeloop campaign memory omitted its scientific priors")
    return tuple(
        _object(
            {
                **insight,
                "workload_id": WORKLOAD_ID,
                "network_panel_sha256": _PANEL_SHA256,
                "finite_contract_identity_sha256": variation.contract.identity_sha256,
            }
        )
        for insight in insights
        if type(insight) is dict
    )


def compose_timeloop_v2_campaign_workload(
    *,
    evaluator_preflight_receipt: FrozenJsonObject,
    resource_lease_receipt: FrozenJsonObject,
    benchmark: AgenticBenchmark = default_benchmark,
) -> AgenticCampaignWorkloadConfig:
    """Compose the workload after external evaluator/resource admission."""

    second_seed = json.loads(_canonical_bytes(DEFAULT_CANDIDATE))
    second_seed["pe_mesh_x"] = 16
    default_seed = _object(json.loads(_canonical_bytes(DEFAULT_CANDIDATE)))
    parallel_seed = _object(second_seed)
    return AgenticCampaignWorkloadConfig(
        workload_id=WORKLOAD_ID,
        workload_version=WORKLOAD_VERSION,
        definition_sha256=WORKLOAD_DEFINITION_SHA256,
        benchmark=benchmark,
        seeds=(
            CampaignSeed("seed_default", default_seed),
            CampaignSeed("seed_pe_mesh_16", parallel_seed),
        ),
        finite_catalog_id=CATALOG_ID,
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=evaluator_preflight_receipt,
        resource_lease_receipt=resource_lease_receipt,
        evidence=AgenticCampaignEvidenceProjections(
            projection_id=EVIDENCE_PROJECTION_ID,
            projection_version=EVIDENCE_PROJECTION_VERSION,
            definition_sha256=EVIDENCE_PROJECTION_DEFINITION_SHA256,
            initialize_memory=_initialize_memory,
            context=_context,
            cards=_cards,
        ),
    )


__all__ = [
    "EVIDENCE_PROJECTION_DEFINITION_SHA256",
    "EVIDENCE_PROJECTION_ID",
    "EVIDENCE_PROJECTION_VERSION",
    "WORKLOAD_DEFINITION_SHA256",
    "WORKLOAD_ID",
    "WORKLOAD_VERSION",
    "compose_timeloop_v2_campaign_workload",
]

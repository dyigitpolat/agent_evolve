"""Campaign-workload composition for pinned BOiLS/ABC panels.

This module owns BOiLS facts only: the two seed sequences, pinned evaluator
panel identity, exact finite catalog, and scientific prompt evidence.  Campaign
chronology, model transport, memory assignment, portfolio selection,
recombination, reflection, and lifecycle learning remain injected through the
generic AgentEvolve campaign application layer.

Construction and port binding perform no evaluator or provider work.
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

from .actions import DEFAULT_ACTION_SEQUENCE
from .budgeted_v5_support import PARENT_C_SEQUENCE
from .detailed_evaluation import (
    boils_evaluator_context_record,
    boils_evaluator_identity,
)
from .finite_variation_catalog import FINITE_CATALOG_ID
from .problem_def import BoilsAbcProblem


WORKLOAD_ID = "boils-abc-pinned-panel"
WORKLOAD_VERSION = 1
EVIDENCE_PROJECTION_ID = "boils_abc_scientific_evidence"
EVIDENCE_PROJECTION_VERSION = 1


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
        raise AssertionError("BOiLS campaign evidence did not freeze to an object")
    return frozen


_PRIOR_INSIGHTS: tuple[dict[str, object], ...] = (
    {
        "card_id": "boils_abc.early_structural_cleanup",
        "claim": (
            "Early structural cleanup or functional reduction can improve mapped "
            "depth, sometimes at an area trade-off."
        ),
        "trigger": "The parent remains depth-limited after its early stages.",
        "mechanism": (
            "Earlier simplification can expose shorter logic paths to later mapping "
            "stages."
        ),
        "affected_paths": ["$.sequence"],
        "status": "prior_hypothesis_to_test",
    },
    {
        "card_id": "boils_abc.late_local_restructuring",
        "claim": (
            "Late technology-aware or resubstitution transforms can improve mapped "
            "area while preserving depth."
        ),
        "trigger": "The parent retains redundant late-stage structure.",
        "mechanism": (
            "Late local restructuring can remove LUT demand without undoing global "
            "depth structure established earlier."
        ),
        "affected_paths": ["$.sequence"],
        "status": "prior_hypothesis_to_test",
    },
    {
        "card_id": "boils_abc.mid_sequence_rebalance",
        "claim": (
            "A middle-stage balance or refactor after rewriting can reduce mapped "
            "depth."
        ),
        "trigger": (
            "A first-half rewrite is followed by a long interval without explicit "
            "depth balancing."
        ),
        "mechanism": (
            "Rebalancing the rewritten network can shorten critical paths before "
            "later area-oriented transforms remap them."
        ),
        "affected_paths": ["$.sequence"],
        "status": "prior_hypothesis_to_test",
    },
)

EVIDENCE_PROJECTION_DEFINITION_SHA256 = _definition_hash(
    b"agent-evolve:boils-abc-campaign-evidence:v1\x00",
    {
        "projection_id": EVIDENCE_PROJECTION_ID,
        "projection_version": EVIDENCE_PROJECTION_VERSION,
        "prior_insights": _PRIOR_INSIGHTS,
        "context_fields": [
            "evaluator_panel",
            "objectives",
            "parent",
            "finite_variation",
            MEMORY_ESTIMAND_CONTEXT_KEY,
            "memory_sha256",
        ],
    },
)

WORKLOAD_DEFINITION_SHA256 = _definition_hash(
    b"agent-evolve:boils-abc-campaign-workload:v1\x00",
    {
        "workload_id": WORKLOAD_ID,
        "workload_version": WORKLOAD_VERSION,
        "finite_catalog_id": FINITE_CATALOG_ID,
        "seed_law": "default_and_parent_c_sequence",
        "evidence_projection_definition_sha256": (
            EVIDENCE_PROJECTION_DEFINITION_SHA256
        ),
        "evaluator_concurrency": "injected_bounded_by_pinned_affinity_leases",
    },
)


def shared_initial_design_workload_definition_sha256(
    *, seed_design_sha256: str, seed_count: int
) -> str:
    """Bind the BOiLS workload identity to one external initial design."""

    if (
        type(seed_design_sha256) is not str
        or len(seed_design_sha256) != 64
        or any(value not in "0123456789abcdef" for value in seed_design_sha256)
    ):
        raise ValueError("seed_design_sha256 must be a lowercase SHA-256 identity")
    if type(seed_count) is not int or seed_count <= 0:
        raise ValueError("seed_count must be a positive exact integer")
    return _definition_hash(
        b"agent-evolve:boils-abc-campaign-workload-shared-initial-design:v1\x00",
        {
            "base_workload_definition_sha256": WORKLOAD_DEFINITION_SHA256,
            "seed_law": "external_sealed_outcome_blind_initial_design",
            "seed_design_sha256": seed_design_sha256,
            "seed_count": seed_count,
        },
    )


def _initialize_memory(benchmark, session, seeds) -> FrozenJsonObject:
    del session
    problem = benchmark.problem
    if type(problem) is not BoilsAbcProblem:
        raise TypeError("BOiLS evidence requires an exact BoilsAbcProblem")
    return _object(
        {
            "schema_version": 1,
            "workload_id": WORKLOAD_ID,
            "evaluator_identity": boils_evaluator_identity(
                problem.settings
            ).to_record(),
            "objective_names": [value.name for value in benchmark.objectives],
            "seed_candidate_keys": [
                problem.candidate_key(thaw_json(seed.configuration))
                for seed in seeds.seeds
            ],
            "insights": list(_PRIOR_INSIGHTS),
        }
    )


def _context(benchmark, session, parent, variation, memory) -> FrozenJsonObject:
    del session
    problem = benchmark.problem
    if type(problem) is not BoilsAbcProblem:
        raise TypeError("BOiLS evidence requires an exact BoilsAbcProblem")
    evaluator_context = boils_evaluator_context_record(problem.settings)
    evaluator_identity = boils_evaluator_identity(problem.settings)
    return _object(
        {
            "schema_version": 1,
            "workload_id": WORKLOAD_ID,
            "evaluator_panel": {
                "identity": evaluator_identity.to_record(),
                "context_sha256": typed_json_sha256(_object(evaluator_context)),
                "circuit_names": [
                    circuit.name for circuit in problem.settings.circuits
                ],
                "per_circuit_timeout_s_hex": float(
                    problem.settings.per_circuit_timeout_s
                ).hex(),
            },
            "objectives": [
                {"metric_id": value.name, "goal": value.goal}
                for value in benchmark.objectives
            ],
            "search_space": problem.search_space_description(),
            "parent": {
                "configuration_sha256": variation.parent_configuration_sha256,
                "rendered": problem.render_candidate(thaw_json(parent)),
            },
            "finite_variation": {
                "catalog_id": variation.contract.catalog_id,
                "contract_identity_sha256": variation.contract.identity_sha256,
                "eligible_option_count": len(variation.contract.options),
            },
            # Parent-, lane-, generation-, and archive-local facts remain in the
            # full selector context.  Reflected-card experiments pool only this
            # explicit, workload-authored intent-to-treat stratum.  The reward
            # aggregation binding is authenticated independently by each trial.
            MEMORY_ESTIMAND_CONTEXT_KEY: {
                "schema_version": 1,
                "estimand_id": "boils_abc_reflected_card_intent_to_treat",
                "workload_id": WORKLOAD_ID,
                "evaluator_context_sha256": (
                    evaluator_identity.evaluator_context_sha256
                ),
                "finite_catalog_id": FINITE_CATALOG_ID,
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
        raise ValueError("BOiLS campaign memory omitted its scientific priors")
    return tuple(
        _object(
            {
                **insight,
                "workload_id": WORKLOAD_ID,
                "finite_contract_identity_sha256": variation.contract.identity_sha256,
            }
        )
        for insight in insights
        if type(insight) is dict
    )


def compose_boils_campaign_workload(
    *,
    benchmark: AgenticBenchmark,
    evaluator_preflight_receipt: FrozenJsonObject,
    resource_lease_receipt: FrozenJsonObject,
    evaluator_concurrency_cap: int,
    seeds: tuple[CampaignSeed, ...] | None = None,
    seed_design_sha256: str | None = None,
) -> AgenticCampaignWorkloadConfig:
    """Compose a pinned BOiLS panel behind the generic campaign API.

    The caller acquires and supplies evaluator/resource admission receipts.
    Merely composing this object neither verifies the filesystem nor evaluates
    a sequence.  ``seeds`` is an experiment-level initial-design interceptor:
    omitting it preserves the two historical benchmark seeds, while supplying
    it requires a sealed design identity.  Candidate validation and evaluation
    remain owned by the generic campaign boundary.
    """

    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    benchmark.validate_binding()
    if type(benchmark.problem) is not BoilsAbcProblem:
        raise TypeError("benchmark problem must be an exact BoilsAbcProblem")
    if type(evaluator_concurrency_cap) is not int or evaluator_concurrency_cap <= 0:
        raise ValueError("evaluator_concurrency_cap must be positive")
    settings = benchmark.problem.settings
    available_evaluator_slots = max(1, len(settings.affinity_sets))
    if evaluator_concurrency_cap > available_evaluator_slots:
        raise ValueError("evaluator concurrency exceeds the pinned affinity leases")
    if seeds is None:
        if seed_design_sha256 is not None:
            raise ValueError("seed_design_sha256 requires an injected seed design")
        resolved_seeds = (
            CampaignSeed(
                "seed_default",
                _object({"sequence": list(DEFAULT_ACTION_SEQUENCE)}),
            ),
            CampaignSeed(
                "seed_parent_c",
                _object({"sequence": list(PARENT_C_SEQUENCE)}),
            ),
        )
        workload_definition_sha256 = WORKLOAD_DEFINITION_SHA256
    else:
        if type(seeds) is not tuple or not seeds:
            raise ValueError("injected seeds must be a non-empty exact tuple")
        if any(type(seed) is not CampaignSeed for seed in seeds):
            raise TypeError("injected seeds must contain exact CampaignSeed values")
        resolved_seeds = seeds
        workload_definition_sha256 = (
            shared_initial_design_workload_definition_sha256(
                seed_design_sha256=seed_design_sha256,
                seed_count=len(resolved_seeds),
            )
        )
    return AgenticCampaignWorkloadConfig(
        workload_id=WORKLOAD_ID,
        workload_version=WORKLOAD_VERSION,
        definition_sha256=workload_definition_sha256,
        benchmark=benchmark,
        seeds=resolved_seeds,
        finite_catalog_id=FINITE_CATALOG_ID,
        evaluator_concurrency_cap=evaluator_concurrency_cap,
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
    "compose_boils_campaign_workload",
    "shared_initial_design_workload_definition_sha256",
]

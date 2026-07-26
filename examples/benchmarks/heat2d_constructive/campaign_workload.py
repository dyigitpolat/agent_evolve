"""Narrow, provider/PDE-free campaign integration for Heat2D Pareto-v1."""

from __future__ import annotations

from agent_evolve.agentic import (
    AgenticBenchmark,
    AgenticCampaignEvidenceProjections,
    FrozenJsonObject,
    WorkloadKit,
    WorkloadPromptExtensionView,
    campaign_seed,
)

from .candidate import seed_layouts
from .finite_variation_catalog import CATALOG_ID
from .multiobjective_v1 import (
    WORKLOAD_VERSION,
    Heat2DMultiObjectiveV1Problem,
)


CAMPAIGN_WORKLOAD_ID = "heat2d_pareto_v1_campaign"


def compose_heat2d_pareto_campaign_workload(
    *,
    benchmark: AgenticBenchmark,
    evaluator_preflight_receipt: FrozenJsonObject,
    resource_lease_receipt: FrozenJsonObject,
    evaluator_concurrency_cap: int = 1,
    evidence: AgenticCampaignEvidenceProjections | None = None,
    prompt_extension: WorkloadPromptExtensionView | None = None,
) -> WorkloadKit:
    """Declare Heat's workload facts behind the shared four-obligation API.

    Evaluator qualification and resource acquisition happen before this call.
    Composition validates immutable facts and canonical seeds, but never starts
    a PDE solve or contacts a model provider.
    """

    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    benchmark.validate_binding()
    if type(benchmark.problem) is not Heat2DMultiObjectiveV1Problem:
        raise TypeError(
            "benchmark problem must be an exact Heat2DMultiObjectiveV1Problem"
        )
    if type(evaluator_concurrency_cap) is not int or evaluator_concurrency_cap <= 0:
        raise ValueError("evaluator_concurrency_cap must be positive")
    if evaluator_concurrency_cap > benchmark.problem.settings.external_concurrency:
        raise ValueError(
            "evaluator concurrency exceeds the qualified Heat2D external cap"
        )
    seeds = tuple(
        campaign_seed(f"seed_{ordinal}", candidate)
        for ordinal, candidate in enumerate(seed_layouts(), start=1)
    )
    return WorkloadKit(
        workload_id=CAMPAIGN_WORKLOAD_ID,
        workload_version=WORKLOAD_VERSION,
        benchmark=benchmark,
        seeds=seeds,
        finite_catalog_id=CATALOG_ID,
        evaluator_concurrency_cap=evaluator_concurrency_cap,
        evaluator_preflight_receipt=evaluator_preflight_receipt,
        resource_lease_receipt=resource_lease_receipt,
        evidence=evidence,
        prompt_extension=prompt_extension,
    )


__all__ = [
    "CAMPAIGN_WORKLOAD_ID",
    "compose_heat2d_pareto_campaign_workload",
]

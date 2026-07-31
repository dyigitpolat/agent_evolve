"""Narrow, provider- and solver-free WorkloadKit composition for scip_miplib.

Evaluator qualification and resource acquisition happen before this call; the
caller supplies the admission receipts.  Composing the kit validates seeds and
catalog identity but never launches a SCIP subprocess or a model provider.
"""

from __future__ import annotations

from agent_evolve.agentic import (
    AgenticBenchmark,
    AgenticCampaignEvidenceProjections,
    FrozenJsonObject,
    WorkloadKit,
    WorkloadPromptExtensionView,
    campaign_seed,
)

from .finite_variation_catalog import CATALOG_ID
from .problem_def import (
    SEED_AGGRESSIVE,
    SEED_DEFAULT,
    SEED_NO_CUTS_NO_HEUR,
    ScipMiplibProblem,
)


CAMPAIGN_WORKLOAD_ID = "scip_miplib_config_v1_campaign"
WORKLOAD_VERSION = 1


def compose_scip_miplib_campaign_workload(
    *,
    benchmark: AgenticBenchmark,
    evaluator_preflight_receipt: FrozenJsonObject,
    resource_lease_receipt: FrozenJsonObject,
    evaluator_concurrency_cap: int = 1,
    evidence: AgenticCampaignEvidenceProjections | None = None,
    prompt_extension: WorkloadPromptExtensionView | None = None,
) -> WorkloadKit:
    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    benchmark.validate_binding()
    if type(benchmark.problem) is not ScipMiplibProblem:
        raise TypeError("benchmark problem must be an exact ScipMiplibProblem")
    if type(evaluator_concurrency_cap) is not int or evaluator_concurrency_cap <= 0:
        raise ValueError("evaluator_concurrency_cap must be positive")
    if (
        evaluator_concurrency_cap
        > benchmark.problem.settings.external_concurrency
    ):
        raise ValueError(
            "evaluator concurrency exceeds the qualified scip_miplib cap"
        )
    seeds = (
        campaign_seed("seed_default", SEED_DEFAULT),
        campaign_seed("seed_no_cuts_no_heur", SEED_NO_CUTS_NO_HEUR),
        campaign_seed("seed_aggressive", SEED_AGGRESSIVE),
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
    "WORKLOAD_VERSION",
    "compose_scip_miplib_campaign_workload",
]

"""Narrow, provider- and solver-free WorkloadKit composition for
pybamm_fastcharge.

The caller supplies externally acquired admission receipts; composing the kit
validates the seed and catalog identity but never launches a PyBaMM
subprocess or a model provider.
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
from .problem_def import SEED_BASELINE_1C, PybammFastChargeProblem


CAMPAIGN_WORKLOAD_ID = "pybamm_fastcharge_v1_campaign"
WORKLOAD_VERSION = 1


def compose_pybamm_fastcharge_campaign_workload(
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
    if type(benchmark.problem) is not PybammFastChargeProblem:
        raise TypeError(
            "benchmark problem must be an exact PybammFastChargeProblem"
        )
    if type(evaluator_concurrency_cap) is not int or evaluator_concurrency_cap <= 0:
        raise ValueError("evaluator_concurrency_cap must be positive")
    if (
        evaluator_concurrency_cap
        > benchmark.problem.settings.external_concurrency
    ):
        raise ValueError(
            "evaluator concurrency exceeds the qualified pybamm_fastcharge cap"
        )
    return WorkloadKit(
        workload_id=CAMPAIGN_WORKLOAD_ID,
        workload_version=WORKLOAD_VERSION,
        benchmark=benchmark,
        seeds=(campaign_seed("seed_baseline_1c", SEED_BASELINE_1C),),
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
    "compose_pybamm_fastcharge_campaign_workload",
]

"""Narrow, provider- and simulator-free WorkloadKit composition for analog_sizing.

Evaluator qualification and resource acquisition happen before this call; the
caller supplies the admission receipts.  Composing the kit validates seeds and
catalog identity but never launches ngspice or a model provider.
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
    SEED_ANALOGGYM_DEFAULT,
    SEED_HEAVIER_COMPENSATION,
    SEED_WIDE_INPUT,
    AnalogSizingProblem,
)


CAMPAIGN_WORKLOAD_ID = "analog_sizing_nmcf_v1_campaign"
WORKLOAD_VERSION = 1


def compose_analog_sizing_campaign_workload(
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
    if type(benchmark.problem) is not AnalogSizingProblem:
        raise TypeError("benchmark problem must be an exact AnalogSizingProblem")
    if type(evaluator_concurrency_cap) is not int or evaluator_concurrency_cap <= 0:
        raise ValueError("evaluator_concurrency_cap must be positive")
    # Seeds span the measured fom_s/phase_margin conflict; each is one
    # documented engineering move from AnalogGym's own shipped default.
    seeds = (
        campaign_seed("seed_analoggym_default", SEED_ANALOGGYM_DEFAULT),
        campaign_seed("seed_heavier_compensation", SEED_HEAVIER_COMPENSATION),
        campaign_seed("seed_wide_input", SEED_WIDE_INPUT),
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
    "compose_analog_sizing_campaign_workload",
]

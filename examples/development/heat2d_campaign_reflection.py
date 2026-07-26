"""Heat2D composition adapter for generic identifiable campaign reflection.

This module owns no scheduling, provider, memory, or campaign runtime logic.
It contributes only Heat2D's closed metric, decision-path, and finite-action
vocabulary, then delegates request construction and learning projection to the
generic application layer.
"""

from __future__ import annotations

from agent_evolve.application.identifiable_reflection_learning import (
    build_identifiable_campaign_reflection_learning_envelope,
)
from agent_evolve.application.identifiable_reflection_request import (
    bind_reflection_contract_to_evidence_actions,
    build_identifiable_reflection_generation_request,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignIdentifiableReflectionInput,
)
from agent_evolve.core.optimization_semantics import OptimizationSemantics
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import FrozenJsonObject
from agent_evolve.ports.agentic_generator import (
    MetricComparisonAnchorKind,
    ReflectionConsumerScope,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    ReflectionInsightKind,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (
    LOCUS_GRIDS,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
)


HEAT2D_REFLECTION_METRIC_IDS = tuple(
    sorted((MATERIAL_OBJECTIVE_NAME, THERMAL_OBJECTIVE_NAME))
)
HEAT2D_REFLECTION_DECISION_PATHS = tuple(
    sorted(f"$.{locus}" for locus, _values, _family in LOCUS_GRIDS)
)
HEAT2D_REFLECTION_OPTION_FAMILIES = tuple(
    sorted({family for _locus, _values, family in LOCUS_GRIDS})
)


def _validate_heat2d_optimization_semantics(
    optimization_semantics: OptimizationSemantics,
) -> None:
    if type(optimization_semantics) is not OptimizationSemantics:
        raise TypeError("optimization_semantics must be exact")
    OptimizationSemantics.__post_init__(optimization_semantics)
    decision_metrics = DecisionMetricProjection.from_optimization_semantics(
        optimization_semantics
    )
    if decision_metrics.metric_ids != HEAT2D_REFLECTION_METRIC_IDS:
        raise ValueError(
            "optimization semantics decision metrics differ from Heat2D objectives"
        )


def heat2d_reflection_contract(
    optimization_semantics: OptimizationSemantics,
    allowed_option_families: tuple[str, ...] = HEAT2D_REFLECTION_OPTION_FAMILIES,
) -> ReflectionInsightContract:
    """Bind the generic empirical-rule contract to the closed Heat2D palette."""

    _validate_heat2d_optimization_semantics(optimization_semantics)
    if (
        type(allowed_option_families) is not tuple
        or not allowed_option_families
        or allowed_option_families
        != tuple(sorted(set(allowed_option_families)))
        or not set(allowed_option_families).issubset(
            HEAT2D_REFLECTION_OPTION_FAMILIES
        )
    ):
        raise ValueError(
            "allowed_option_families must be a canonical Heat2D subset"
        )
    return ReflectionInsightContract(
        required_metric_ids=HEAT2D_REFLECTION_METRIC_IDS,
        allowed_option_families=allowed_option_families,
        allowed_decision_paths=HEAT2D_REFLECTION_DECISION_PATHS,
        allowed_insight_kinds=(
            ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        ),
        allowed_consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        allowed_comparison_anchor_kinds=(
            MetricComparisonAnchorKind.CURRENT_PARENT,
        ),
        allowed_factor_capabilities=allowed_option_families,
    )


def build_heat2d_identifiable_reflection_request(
    *,
    call_id: LLMCallId,
    reflection_input: CampaignIdentifiableReflectionInput,
    optimization_semantics: OptimizationSemantics,
    max_output_tokens: int,
    temperature: float | None,
    allowed_option_families: tuple[str, ...] = (
        HEAT2D_REFLECTION_OPTION_FAMILIES
    ),
    min_insights: int = 1,
    max_insights: int = 2,
) -> ReflectionGenerationRequest:
    """Construct the provider-neutral request from sealed G1 mutations only."""

    if type(reflection_input) is not CampaignIdentifiableReflectionInput:
        raise TypeError("reflection_input must be exact")
    CampaignIdentifiableReflectionInput.__post_init__(reflection_input)
    contract = bind_reflection_contract_to_evidence_actions(
        heat2d_reflection_contract(
            optimization_semantics,
            allowed_option_families,
        ),
        reflection_input.evidence,
    )
    return build_identifiable_reflection_generation_request(
        call_id=call_id,
        evidence=reflection_input.evidence,
        insight_contract=contract,
        optimization_semantics=optimization_semantics,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        min_insights=min_insights,
        max_insights=max_insights,
    )


def build_heat2d_identifiable_reflection_learning_envelope(
    *,
    reflection_input: CampaignIdentifiableReflectionInput,
    request: ReflectionGenerationRequest,
    result: ReflectionGenerationResult,
    optimization_semantics: OptimizationSemantics,
) -> FrozenJsonObject:
    """Delegate occurrence lineage and finite attribution to the generic codec."""

    _validate_heat2d_optimization_semantics(optimization_semantics)
    contract = request.insight_contract
    if contract is None:
        raise ValueError("identifiable Heat2D request lost its semantic contract")
    expected = bind_reflection_contract_to_evidence_actions(
        heat2d_reflection_contract(
            optimization_semantics,
            contract.allowed_option_families,
        ),
        reflection_input.evidence,
    )
    if contract != expected:
        raise ValueError("reflection request carries a foreign Heat2D contract")
    return build_identifiable_campaign_reflection_learning_envelope(
        reflection_input=reflection_input,
        request=request,
        result=result,
        optimization_semantics=optimization_semantics,
    )


__all__ = [
    "HEAT2D_REFLECTION_DECISION_PATHS",
    "HEAT2D_REFLECTION_METRIC_IDS",
    "HEAT2D_REFLECTION_OPTION_FAMILIES",
    "build_heat2d_identifiable_reflection_learning_envelope",
    "build_heat2d_identifiable_reflection_request",
    "heat2d_reflection_contract",
]

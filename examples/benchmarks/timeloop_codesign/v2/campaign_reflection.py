"""Timeloop-owned vocabulary for generic identifiable reflection.

The benchmark owns only metric meanings and the closed finite-action
vocabulary.  Evidence projection, cutoff enforcement, provider prompting,
lineage replay, and the campaign-learning envelope remain in AgentEvolve's
workload-neutral application layer.
"""

from __future__ import annotations

import math

from agent_evolve.agentic import (
    MetricComparisonAnchorKind,
    OptimizationSemantics,
    ReflectionConsumerScope,
    ReflectionGenerationRequest,
    ReflectionInsightContract,
    ReflectionInsightKind,
)
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
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import FrozenJsonObject
from agent_evolve.ports.agentic_generator import ReflectionGenerationResult
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection

from .candidate import (
    ARCHITECTURE_FIELD_GRIDS,
    POLICY_BLOCK_FIELDS,
    POLICY_FIELD_GRIDS,
)
from .evaluator import OBJECTIVE_NAMES


OBJECTIVE_IDS = tuple(sorted(OBJECTIVE_NAMES))
REFLECTION_OPTION_FAMILIES = tuple(
    sorted(
        {value[2] for value in ARCHITECTURE_FIELD_GRIDS}
        | {value[2] for value in POLICY_FIELD_GRIDS}
    )
)
REFLECTION_DECISION_PATHS = tuple(
    sorted(
        {
            *(f"$.{field}" for field, _values, _family in ARCHITECTURE_FIELD_GRIDS),
            *(
                f"$.{block}.{field}"
                for block in POLICY_BLOCK_FIELDS
                for field, _values, _family in POLICY_FIELD_GRIDS
            ),
        }
    )
)


def timeloop_v2_reflection_contract() -> ReflectionInsightContract:
    """Bind empirical predictions to the complete Timeloop v2 action palette."""

    return ReflectionInsightContract(
        required_metric_ids=OBJECTIVE_IDS,
        allowed_option_families=REFLECTION_OPTION_FAMILIES,
        allowed_decision_paths=REFLECTION_DECISION_PATHS,
        allowed_insight_kinds=(ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,),
        allowed_consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        allowed_comparison_anchor_kinds=(
            MetricComparisonAnchorKind.CURRENT_PARENT,
        ),
        allowed_factor_capabilities=REFLECTION_OPTION_FAMILIES,
    )


def _validate_semantics(optimization_semantics: OptimizationSemantics) -> None:
    if type(optimization_semantics) is not OptimizationSemantics:
        raise TypeError("optimization_semantics must be exact")
    OptimizationSemantics.__post_init__(optimization_semantics)
    projection = DecisionMetricProjection.from_optimization_semantics(
        optimization_semantics
    )
    if projection.metric_ids != OBJECTIVE_IDS:
        raise ValueError(
            "optimization semantics decision metrics differ from Timeloop objectives"
        )


def build_timeloop_v2_identifiable_reflection_request(
    *,
    call_id: LLMCallId,
    reflection_input: CampaignIdentifiableReflectionInput,
    optimization_semantics: OptimizationSemantics,
    max_output_tokens: int,
    temperature: float | None,
    min_insights: int = 1,
    max_insights: int = 2,
) -> ReflectionGenerationRequest:
    """Join Timeloop semantics to the generic sealed request builder."""

    if type(reflection_input) is not CampaignIdentifiableReflectionInput:
        raise TypeError("reflection_input must be exact")
    CampaignIdentifiableReflectionInput.__post_init__(reflection_input)
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if temperature is not None and (
        type(temperature) is not float or not math.isfinite(temperature)
    ):
        raise ValueError("temperature must be a finite exact float or None")
    _validate_semantics(optimization_semantics)
    exact_contract = bind_reflection_contract_to_evidence_actions(
        timeloop_v2_reflection_contract(),
        reflection_input.evidence,
    )
    return build_identifiable_reflection_generation_request(
        call_id=call_id,
        evidence=reflection_input.evidence,
        insight_contract=exact_contract,
        optimization_semantics=optimization_semantics,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        min_insights=min_insights,
        max_insights=max_insights,
    )


def build_timeloop_v2_identifiable_learning_envelope(
    *,
    reflection_input: CampaignIdentifiableReflectionInput,
    request: ReflectionGenerationRequest,
    result: ReflectionGenerationResult,
    optimization_semantics: OptimizationSemantics,
) -> FrozenJsonObject:
    """Delegate canonical occurrence lineage to the generic learning adapter."""

    _validate_semantics(optimization_semantics)
    if type(request) is not ReflectionGenerationRequest:
        raise TypeError("request must be exact")
    ReflectionGenerationRequest.__post_init__(request)
    exact_contract = bind_reflection_contract_to_evidence_actions(
        timeloop_v2_reflection_contract(),
        reflection_input.evidence,
    )
    if request.insight_contract != exact_contract:
        raise ValueError("reflection request carries a foreign Timeloop contract")
    return build_identifiable_campaign_reflection_learning_envelope(
        reflection_input=reflection_input,
        request=request,
        result=result,
        optimization_semantics=optimization_semantics,
    )


__all__ = [
    "OBJECTIVE_IDS",
    "REFLECTION_DECISION_PATHS",
    "REFLECTION_OPTION_FAMILIES",
    "build_timeloop_v2_identifiable_learning_envelope",
    "build_timeloop_v2_identifiable_reflection_request",
    "timeloop_v2_reflection_contract",
]

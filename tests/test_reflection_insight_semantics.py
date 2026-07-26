from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
    REFLECTION_SEMANTIC_PROMPT_RENDERER_DEFINITION_SHA256,
    REFLECTION_SEMANTIC_WIRE_CONTRACT_REVISION,
    _reflection_output_type,
    render_reflection_prompt,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionGenerationRequest,
    ReflectionInsightContract,
    ReflectionInsightKind,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


_CONTRAST_ID = "1" * 64


def _contract() -> ReflectionInsightContract:
    return ReflectionInsightContract(
        required_metric_ids=("material_fraction", "thermal_term"),
        allowed_option_families=("additive_geometry", "material_fraction"),
        allowed_decision_paths=("$.geometry.left_lobe.radius_x",),
        allowed_insight_kinds=(
            ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            ReflectionInsightKind.MECHANISTIC_CONJECTURE,
        ),
        allowed_consumer_scopes=(ReflectionConsumerScope.RECOMBINATION_SELECTION,),
        allowed_comparison_anchor_kinds=(
            MetricComparisonAnchorKind.COMMON_ANCESTOR,
            MetricComparisonAnchorKind.NAMED_SOURCE_ROLE,
        ),
        allowed_factor_capabilities=(
            "additive_geometry",
            "material_fraction_control",
        ),
        allowed_source_role_ids=("geometry_source", "material_source"),
    )


def _draft() -> InsightDraft:
    return InsightDraft(
        claim="A transferred lobe can improve the child relative to its ancestor.",
        trigger="Recombine a material controller with an additive lobe source.",
        mechanism="The lobe adds a connected conduction path.",
        affected_paths=("$.geometry.left_lobe.radius_x",),
        evidence_summary="The cited child improved in the held comparison.",
        confidence=0.7,
        evidence_contrast_ids=(_CONTRAST_ID,),
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="material_fraction",
                direction=MetricEffectDirection.UNCHANGED,
                comparison_anchor=MetricComparisonAnchor(
                    MetricComparisonAnchorKind.NAMED_SOURCE_ROLE,
                    "material_source",
                ),
            ),
            MetricEffectPrediction(
                metric_id="thermal_term",
                direction=MetricEffectDirection.DECREASE,
                comparison_anchor=MetricComparisonAnchor(
                    MetricComparisonAnchorKind.COMMON_ANCESTOR,
                ),
            ),
        ),
        recommended_option_families=("additive_geometry",),
        action_template="Pair the controller with a connected lobe source.",
        falsification_condition="The held child fails to improve thermal_term.",
        insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        consumer_scopes=(ReflectionConsumerScope.RECOMBINATION_SELECTION,),
        factor_capabilities=("additive_geometry", "material_fraction_control"),
    )


def test_comparison_anchor_has_closed_role_semantics() -> None:
    with pytest.raises(ValueError, match="require a canonical source_role_id"):
        MetricComparisonAnchor(MetricComparisonAnchorKind.NAMED_SOURCE_ROLE)
    with pytest.raises(ValueError, match="only for named_source_role"):
        MetricComparisonAnchor(
            MetricComparisonAnchorKind.CURRENT_PARENT,
            "material_source",
        )
    with pytest.raises(TypeError, match="exact MetricComparisonAnchorKind"):
        MetricComparisonAnchor("current_parent")  # type: ignore[arg-type]


def test_v3_contract_is_explicit_bounded_and_preserves_v2_records() -> None:
    legacy = ReflectionInsightContract(
        required_metric_ids=("material_fraction", "thermal_term"),
        allowed_option_families=("additive_geometry", "material_fraction"),
    )
    assert legacy.to_record()["schema_version"] == 2
    assert "allowed_decision_paths" not in legacy.to_record()

    contract = _contract()
    record = contract.to_record()
    assert contract.is_semantic_v3
    assert record["schema_version"] == 3
    assert record["allowed_decision_paths"] == ["$.geometry.left_lobe.radius_x"]
    assert record["allowed_insight_kinds"] == [
        "empirical_predictive_rule",
        "mechanistic_conjecture",
    ]
    assert "contract_invariant" not in record["allowed_insight_kinds"]
    assert record["allowed_factor_capabilities"] == [
        "additive_geometry",
        "material_fraction_control",
    ]
    assert len(contract.identity_sha256) == 64

    with pytest.raises(ValueError, match="cannot admit search_heuristic"):
        replace(
            contract,
            allowed_insight_kinds=(
                ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
                ReflectionInsightKind.SEARCH_HEURISTIC,
            ),
        )
    with pytest.raises(ValueError, match="cannot admit contract_invariant"):
        replace(
            contract,
            allowed_insight_kinds=(
                ReflectionInsightKind.CONTRACT_INVARIANT,
                ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            ),
        )
    with pytest.raises(ValueError, match="source role IDs"):
        replace(contract, allowed_source_role_ids=())
    with pytest.raises(ValueError, match="SEMANTIC_VOCABULARY"):
        replace(
            contract,
            allowed_decision_paths=tuple(
                sorted(f"$.field_{index:04d}" for index in range(257))
            ),
        )


def test_provider_free_validation_separates_paths_metrics_and_capabilities() -> None:
    contract = _contract()
    draft = _draft()
    validate_reflection_insight_draft(draft, contract)

    for foreign_path in ("$.thermal_term", "$.additive_geometry"):
        with pytest.raises(ValueError, match="decision-path vocabulary"):
            validate_reflection_insight_draft(
                replace(draft, affected_paths=(foreign_path,)),
                contract,
            )
    with pytest.raises(ValueError, match="capability vocabulary"):
        validate_reflection_insight_draft(
            replace(draft, factor_capabilities=("thermal_term",)),
            contract,
        )
    with pytest.raises(ValueError, match="consumer scopes"):
        validate_reflection_insight_draft(
            replace(
                draft,
                consumer_scopes=(ReflectionConsumerScope.PARENT_SELECTION,),
            ),
            contract,
        )
    unknown_prediction = replace(
        draft.effect_predictions[0],
        direction=MetricEffectDirection.UNKNOWN,
    )
    with pytest.raises(ValueError, match="adjudicable metric directions"):
        validate_reflection_insight_draft(
            replace(
                draft,
                effect_predictions=(unknown_prediction, draft.effect_predictions[1]),
            ),
            contract,
        )


def test_v3_rejects_missing_or_foreign_metric_comparison_anchors() -> None:
    contract = _contract()
    draft = _draft()
    legacy_prediction = replace(
        draft.effect_predictions[0],
        comparison_anchor=None,
    )
    with pytest.raises(ValueError, match="require comparison anchors"):
        replace(
            draft,
            effect_predictions=(legacy_prediction, draft.effect_predictions[1]),
        )

    foreign_role_prediction = replace(
        draft.effect_predictions[0],
        comparison_anchor=MetricComparisonAnchor(
            MetricComparisonAnchorKind.NAMED_SOURCE_ROLE,
            "foreign_source",
        ),
    )
    with pytest.raises(ValueError, match="source role escapes"):
        validate_reflection_insight_draft(
            replace(
                draft,
                effect_predictions=(
                    foreign_role_prediction,
                    draft.effect_predictions[1],
                ),
            ),
            contract,
        )


def test_search_heuristic_cannot_self_certify_causal_predictions() -> None:
    with pytest.raises(ValueError, match="cannot carry causal metric predictions"):
        replace(
            _draft(),
            insight_kind=ReflectionInsightKind.SEARCH_HEURISTIC,
        )


def test_v3_validator_rejects_model_authored_contract_invariant_defensively() -> None:
    draft = replace(
        _draft(),
        insight_kind=ReflectionInsightKind.CONTRACT_INVARIANT,
    )
    with pytest.raises(ValueError, match="cannot assert contract_invariant"):
        validate_reflection_insight_draft(draft, _contract())


def test_v3_pydantic_schema_is_closed_before_provider_execution() -> None:
    contract = _contract()
    output_type = _reflection_output_type(
        1,
        (_CONTRAST_ID,),
        contract,
        min_insights=1,
    )
    properties = output_type.model_json_schema()["properties"]["insights"]["items"][
        "properties"
    ]
    assert properties["affected_paths"]["items"]["enum"] == [
        "$.geometry.left_lobe.radius_x"
    ]
    assert properties["insight_kind"]["enum"] == [
        "empirical_predictive_rule",
        "mechanistic_conjecture",
    ]
    assert properties["consumer_scopes"]["items"]["enum"] == ["recombination_selection"]
    prediction = properties["effect_predictions"]["items"]
    assert "comparison_anchor" in prediction["required"]
    assert prediction["properties"]["comparison_anchor"]["properties"]["kind"][
        "enum"
    ] == ["common_ancestor", "named_source_role"]

    payload = _wire_payload()
    output_type.model_validate(payload, strict=True)
    with pytest.raises(ValidationError, match="decision-path vocabulary"):
        output_type.model_validate(
            {
                "insights": [
                    {
                        **payload["insights"][0],
                        "affected_paths": ["$.thermal_term"],
                    }
                ]
            },
            strict=True,
        )
    unknown_payload = _wire_payload()
    unknown_payload["insights"][0]["effect_predictions"][0]["direction"] = "unknown"
    with pytest.raises(ValidationError, match="adjudicable metric direction"):
        output_type.model_validate(unknown_payload, strict=True)


def _wire_payload() -> dict[str, Any]:
    return {
        "insights": [
            {
                "claim": "A transferred lobe can improve the child.",
                "trigger": "Recombine controller and lobe sources.",
                "mechanism": "The lobe adds a connected path.",
                "affected_paths": ["$.geometry.left_lobe.radius_x"],
                "evidence_summary": "The cited held comparison improved.",
                "evidence_contrast_ids": [_CONTRAST_ID],
                "confidence": 0.7,
                "effect_predictions": [
                    {
                        "metric_id": "material_fraction",
                        "direction": "unchanged",
                        "comparison_anchor": {
                            "kind": "named_source_role",
                            "source_role_id": "material_source",
                        },
                    },
                    {
                        "metric_id": "thermal_term",
                        "direction": "decrease",
                        "comparison_anchor": {
                            "kind": "common_ancestor",
                            "source_role_id": None,
                        },
                    },
                ],
                "recommended_option_families": ["additive_geometry"],
                "action_template": "Pair the controller with the lobe source.",
                "falsification_condition": "The child fails to improve.",
                "insight_kind": "empirical_predictive_rule",
                "consumer_scopes": ["recombination_selection"],
                "factor_capabilities": [
                    "additive_geometry",
                    "material_fraction_control",
                ],
            }
        ]
    }


def test_v3_codec_and_prompt_round_trip_typed_semantics() -> None:
    captured: list[StructuredGenerationRequest[Any]] = []

    async def generate_once(
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        captured.append(request)
        value = request.output_type.model_validate(_wire_payload(), strict=True)
        return StructuredGenerationResponse(
            value=value,
            requested_model="test/model",
            resolved_model="test/model",
            resolved_provider="fake",
            provider_response_id="response-1",
            finish_reason="tool_call",
            input_tokens=10,
            output_tokens=20,
            reasoning_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.001"),
            latency_ns=100,
        )

    contract = _contract()
    request = ReflectionGenerationRequest(
        call_id=LLMCallId("call_reflection_semantics_v3_0001"),
        operation="reflect_evaluation",
        prompt="Extract bounded reusable insights.",
        max_insights=1,
        min_insights=1,
        max_output_tokens=1024,
        available_contrast_ids=(_CONTRAST_ID,),
        insight_contract=contract,
    )
    result = asyncio.run(PydanticAIAgenticGenerator(generate_once).reflect(request))
    assert len(result.insights) == 1
    decoded = result.insights[0]
    validate_reflection_insight_draft(decoded, contract)
    assert decoded.insight_kind is ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE
    assert decoded.consumer_scopes == (ReflectionConsumerScope.RECOMBINATION_SELECTION,)
    assert decoded.factor_capabilities == (
        "additive_geometry",
        "material_fraction_control",
    )
    assert decoded.effect_predictions[0].comparison_anchor == MetricComparisonAnchor(
        MetricComparisonAnchorKind.NAMED_SOURCE_ROLE,
        "material_source",
    )
    assert captured[0].prompt_lineage.renderer_revision == (
        REFLECTION_SEMANTIC_WIRE_CONTRACT_REVISION
    )
    assert captured[0].prompt_lineage.renderer_definition_sha256 == (
        REFLECTION_SEMANTIC_PROMPT_RENDERER_DEFINITION_SHA256
    )
    assert "affected_paths contains only candidate decision paths" in captured[0].prompt
    assert '"allowed_decision_paths":["$.geometry.left_lobe.radius_x"]' in (
        captured[0].prompt
    )
    assert (
        render_reflection_prompt(
            "Extract bounded reusable insights.",
            insight_contract=contract,
        )
        == captured[0].prompt
    )

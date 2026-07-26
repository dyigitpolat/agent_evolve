from __future__ import annotations

import asyncio
import json
from decimal import Decimal

from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    render_portfolio_selection_prompt,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    resolve_ranked_portfolio_decision,
)
from examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_development import (
    BASE_DEVELOPMENT_DESIGN_ID,
    DEVELOPMENT_DESIGN_ID,
    DirectoryDevelopmentRecordSink,
    EXPECTED_SHARD_MAPPING_SHA256,
    STAGE_A_MAX_OUTPUT_TOKENS,
    execute_provider_ready_stage_a,
)


def _telemetry(kind: str) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="offline",
        provider_response_id=f"response-{kind}",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=10,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _ConcurrentCardGenerator:
    def __init__(self) -> None:
        self.calls: list[ReflectionGenerationRequest] = []
        self.active = 0
        self.max_active = 0

    async def propose(self, request: VariationGenerationRequest):  # pragma: no cover
        raise AssertionError(request)

    async def reflect(
        self, request: ReflectionGenerationRequest
    ) -> ReflectionGenerationResult:
        self.calls.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        contract = request.insight_contract
        assert contract is not None
        option_id = contract.allowed_option_ids[0]
        family = contract.allowed_option_families[0]
        insight = InsightDraft(
            claim=f"Use one sealed action from {family}.",
            trigger="The supplied parent matches this sealed adaptation context.",
            mechanism="The action changes a bounded Airfoil design coordinate.",
            affected_paths=("$.alpha_deg",),
            evidence_summary="The cited sealed contrast determines the direction.",
            confidence=0.75,
            evidence_contrast_ids=(request.available_contrast_ids[0],),
            effect_predictions=tuple(
                MetricEffectPrediction(
                    metric_id=metric_id,
                    direction=MetricEffectDirection.DECREASE,
                )
                for metric_id in contract.required_metric_ids
            ),
            recommended_option_families=(family,),
            recommended_option_ids=(option_id,),
            action_template=f"Select exact sealed option {option_id}.",
            falsification_condition="Reject if either predicted metric increases.",
        )
        return ReflectionGenerationResult(
            insights=(insight,), telemetry=_telemetry("reflection")
        )


class _ConcurrentSelector:
    def __init__(self) -> None:
        self.calls: list[PortfolioSelectionRequest] = []
        self.active = 0
        self.max_active = 0

    async def select(
        self, request: PortfolioSelectionRequest
    ) -> PortfolioSelectionResult:
        self.calls.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        first_payload = thaw_json(request.cards[0].prompt_payload)
        if "origin_evidence" in first_payload["selector_evidence_bundle"]:
            option_ids = (
                "trim.p050.p050.p050",
                "trim.p025.p050.p050",
                "trim.p050.p025.p050",
            )
        else:
            option_ids = tuple(
                option.option_id
                for option in request.finite_variation_contract.options[-3:]
            )
        card_key = request.cards[0].card_key
        predictions = tuple(
            MetricEffectPrediction(
                metric_id=metric_id,
                direction=MetricEffectDirection.DECREASE,
            )
            for metric_id in request.required_metric_ids
        )
        drafts = tuple(
            PortfolioMemberDraft(
                option_id=option_id,
                supporting_card_keys=(card_key,),
                effect_predictions=predictions,
                design_rationale=f"Offline selection of {option_id}.",
            )
            for option_id in option_ids
        )
        decision = resolve_ranked_portfolio_decision(
            request,
            drafts,
            policy_id="offline_test_selector",
            policy_version=1,
            policy_definition_sha256="f" * 64,
        )
        return PortfolioSelectionResult(decision=decision, telemetry=None)


def test_sealed_oracle_stage_a_runs_end_to_end_with_injected_concurrency(
    tmp_path,
) -> None:
    generator = _ConcurrentCardGenerator()
    selector = _ConcurrentSelector()
    sink = DirectoryDevelopmentRecordSink(tmp_path)

    result = asyncio.run(
        execute_provider_ready_stage_a(
            generator=generator,
            selector=selector,
            id_factory=DeterministicIdFactory("oracle_portfolio_test"),
            sink=sink,
        )
    )

    assert result["design_id"] == DEVELOPMENT_DESIGN_ID
    assert result["base_method_design_id"] == BASE_DEVELOPMENT_DESIGN_ID
    assert result["execution_revision_class"] == (
        "pre_treatment_provider_grammar_repair"
    )
    assert result["mechanism_revision_ordinal"] == 0
    assert len(generator.calls) == 8
    assert generator.max_active == 8
    assert {call.max_output_tokens for call in generator.calls} == {
        STAGE_A_MAX_OUTPUT_TOKENS
    }
    assert sorted(len(call.available_contrast_ids) for call in generator.calls) == [
        4,
        4,
        4,
        4,
        16,
        16,
        16,
        16,
    ]
    assert len(selector.calls) == 3
    assert selector.max_active == 3
    assert {call.max_output_tokens for call in selector.calls} == {
        STAGE_A_MAX_OUTPUT_TOKENS
    }
    assert {row["view_id"] for row in result["views"]} == {"M", "P", "N"}
    assert result["engine_baselines"]["E"]["score"]["action_ranks"] == [1, 2, 3]

    plan = json.loads((tmp_path / "development_plan.json").read_text())
    reflection = json.loads((tmp_path / "reflection_results.json").read_text())
    durable_result = json.loads((tmp_path / "selector_results.json").read_text())
    assert plan["credentials_read"] is False
    assert plan["provider_calls_observed"] == 0
    assert (
        plan["shard_design"]["mapping_sha256"]
        == EXPECTED_SHARD_MAPPING_SHA256
    )
    assert len(reflection["cards"]) == 8
    assert all(
        len(source["score_components"]) == 6
        for source in reflection["card_projection_sources"]
    )
    assert all(
        "source_shard_id" not in source["evidence_bundle"]
        and "source_family" not in source["evidence_bundle"]
        for source in reflection["card_projection_sources"]
    )
    selector_views = {
        row["view_id"]: row for row in reflection["selector_views"]["views"]
    }
    source_by_key = {
        source["card_key"]: source
        for source in reflection["card_projection_sources"]
    }
    card_by_view_and_key = {
        view_id: {card["card_key"]: card for card in view["cards"]}
        for view_id, view in selector_views.items()
    }
    p_source_by_target = plan["selector_view_design"][
        "P_precommitted_bundle_sources"
    ]
    for target_key, source in source_by_key.items():
        m_card = card_by_view_and_key["M"][target_key]
        p_card = card_by_view_and_key["P"][target_key]
        n_card = card_by_view_and_key["N"][target_key]
        rotated_source = source_by_key[p_source_by_target[target_key]]

        assert m_card["payload"]["action_binding"] == source["action_binding"]
        assert p_card["payload"]["action_binding"] == source["action_binding"]
        assert m_card["payload"]["selector_evidence_bundle"] == source[
            "evidence_bundle"
        ]
        assert p_card["payload"]["selector_evidence_bundle"] == rotated_source[
            "evidence_bundle"
        ]
        assert m_card["evidence_sha256"] == source["evidence_sha256"]
        assert p_card["evidence_sha256"] == rotated_source["evidence_sha256"]
        assert m_card["content_sha256"] == p_card["content_sha256"] == n_card[
            "content_sha256"
        ]
        assert m_card["insight_id"] == p_card["insight_id"] == n_card[
            "insight_id"
        ]
        assert m_card["insight_version"] == p_card["insight_version"] == n_card[
            "insight_version"
        ]
        assert [
            {
                **{
                    key: value
                    for key, value in component.items()
                    if key != "value_hex"
                },
                "value": float.fromhex(component["value_hex"]),
            }
            for component in rotated_source["score_components"]
        ] == p_card["score_components"]
        assert n_card["score_components"] == []
        assert n_card["payload"]["action_binding"]["recommended_option_ids"] == []
        assert set(n_card["payload"]["selector_evidence_bundle"]) == {
            "metric_ids",
            "score_component_ids",
        }

    forbidden_treatment_labels = (
        "correct",
        "within_family_rotated",
        "evidence_redacted",
        '"view"',
    )
    for view in selector_views.values():
        model_visible = render_portfolio_selection_prompt(
            next(
                call
                for call in selector.calls
                if call.call_id.value == view["request"]["call_id"]
            )
        )
        assert not any(label in model_visible for label in forbidden_treatment_labels)
        assert all(
            set(card["payload"]) == {
                "schema_version",
                "action_binding",
                "selector_evidence_bundle",
            }
            for card in view["cards"]
        )
    rendered_by_view = {
        view_id: render_portfolio_selection_prompt(
            next(
                call
                for call in selector.calls
                if call.call_id.value == view["request"]["call_id"]
            )
        )
        for view_id, view in selector_views.items()
    }
    assert len(rendered_by_view["M"].encode()) == len(
        rendered_by_view["P"].encode()
    )
    assert [
        tuple(card["card_key"] for card in selector_views[view_id]["cards"])
        for view_id in ("M", "P", "N")
    ] == [
        tuple(card["card_key"] for card in selector_views["M"]["cards"])
    ] * 3
    assert [
        tuple(
            card["payload"]["action_binding"]["family"]
            for card in selector_views[view_id]["cards"]
        )
        for view_id in ("M", "P", "N")
    ] == [
        tuple(
            card["payload"]["action_binding"]["family"]
            for card in selector_views["M"]["cards"]
        )
    ] * 3
    assert durable_result == result

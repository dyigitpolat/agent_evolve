from __future__ import annotations

import asyncio
from decimal import Decimal
import hashlib

import pytest
from pydantic import ValidationError

from agent_evolve.application.residual_reachability import (
    ParentFiniteVariationBinding,
    ResidualProposalRole,
    bind_cross_parent_finite_action_schema,
    materialize_hierarchical_residual_plan,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.integrations.pydantic_ai.reconciled_residual_reachability import (
    PydanticAIReconciledHierarchicalResidualProposalPolicy,
    postcompile_semantic_regrounding_output_type,
    reconciled_residual_preference_output_type,
)
from agent_evolve.integrations.pydantic_ai.residual_reachability import (
    HierarchicalResidualProposalRequest,
    hierarchical_residual_output_type,
)
from agent_evolve.ports.structured_generator import StructuredGenerationResponse


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _contract() -> FiniteVariationContract:
    parent = freeze_json({"a": 0, "b": 0})
    parent_sha256 = typed_json_sha256(parent)
    return FiniteVariationContract(
        catalog_id="reconciliation_test",
        catalog_version=1,
        catalog_definition_sha256=_sha("catalog"),
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="field.a.one",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": 1, "b": 0}),
                family="replace",
                description="Set field a to one.",
                metadata=(("replacement", "one"),),
            ),
            FiniteVariationOption(
                option_id="field.a.two",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": 2, "b": 0}),
                family="replace",
                description="Set field a to two.",
                metadata=(("replacement", "two"),),
            ),
            FiniteVariationOption(
                option_id="field.b.one",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": 0, "b": 1}),
                family="replace",
                description="Set field b to one.",
                metadata=(("replacement", "one"),),
            ),
        ),
    )


def _request() -> HierarchicalResidualProposalRequest:
    parent_id = CandidateId("candidate_parent")
    schema = bind_cross_parent_finite_action_schema(
        (ParentFiniteVariationBinding(parent_id, _contract()),)
    )
    return HierarchicalResidualProposalRequest(
        call_id=LLMCallId("call_reconciled"),
        operation="propose_residual",
        instruction="Propose one interaction.",
        context=freeze_json({"archive": []}),
        action_schema=schema,
        proposal_count=1,
        allowed_radii=(2,),
        allowed_roles=(ResidualProposalRole.INTERACTION,),
        required_metric_ids=("m1",),
        minimum_distinct_parents=1,
        expert_id="interaction_expert",
        expert_definition_sha256=_sha("expert"),
        max_output_tokens=1024,
        temperature=0.0,
    )


def _invalid_preference() -> dict[str, object]:
    return {
        "members": [
            {
                "parent_candidate_id": "candidate_parent",
                "component_option_ids": [
                    "field.a.one",
                    "field.a.two",
                ],
                "role": "interaction",
                "probability_valid": 0.8,
                "effect_predictions": [
                    {
                        "metric_id": "m1",
                        "p10_delta": 2.0,
                        "p50_delta": 0.0,
                        "p90_delta": -1.0,
                        "confidence": 0.7,
                    }
                ],
                "interaction_rationale": (
                    "Prefer a strong same-region interaction."
                ),
            }
        ],
        "slate_rationale": "One preference for trusted reconciliation.",
    }


def test_structural_preference_accepts_member_local_semantic_mistakes() -> None:
    request = _request()
    structural = reconciled_residual_preference_output_type(request)
    assert structural.model_validate(_invalid_preference())
    strict = hierarchical_residual_output_type(request)
    with pytest.raises(ValidationError):
        strict.model_validate(_invalid_preference())


def test_policy_reconciles_one_member_without_retrying_the_slate() -> None:
    asyncio.run(_exercise_policy())


async def _exercise_policy() -> None:
    request = _request()
    provider_calls = 0
    call_ids: list[str] = []
    receipts: list[dict[str, object]] = []

    async def generate_once(structured_request):
        nonlocal provider_calls
        provider_calls += 1
        call_ids.append(structured_request.call_id.value)
        if structured_request.operation == "propose_residual":
            payload = _invalid_preference()
        else:
            assert (
                structured_request.operation
                == "postcompile_semantic_regrounding"
            )
            assert "Claims written for the rejected actions have no authority" in (
                structured_request.prompt
            )
            payload = {
                "members": [
                    {
                        "member_index": 1,
                        "semantic_fidelity_acknowledgement": (
                            "exact_compiled_action"
                        ),
                        "probability_valid": 0.93,
                        "effect_predictions": [
                            {
                                "metric_id": "m1",
                                "p10_delta": -3.0,
                                "p50_delta": -2.0,
                                "p90_delta": 0.5,
                                "confidence": 0.85,
                            }
                        ],
                        "interaction_rationale": (
                            "The exact compiled a/b action edits disjoint fields."
                        ),
                    }
                ],
                "regrounding_rationale": (
                    "All claims refer only to the exact compiled action."
                ),
            }
        value = structured_request.output_type.model_validate(payload)
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider/model",
            resolved_model="provider/model",
            resolved_provider="provider",
            provider_response_id="response",
            finish_reason="tool_call",
            input_tokens=10,
            output_tokens=5,
            reasoning_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.001"),
            latency_ns=100,
        )

    policy = PydanticAIReconciledHierarchicalResidualProposalPolicy(
        generate_once,
        reconciliation_sink=receipts.append,
    )
    selection = await policy.select(request)

    assert provider_calls == 2
    assert call_ids == ["call_reconciled", "call_reconciled_reground"]
    assert len(selection.plans) == 1
    assert "field.b.one" in selection.plans[0].component_option_ids
    assert len(set(selection.plans[0].component_option_ids)) == 2
    forecast = selection.effect_predictions[0][0]
    assert (forecast.p10_delta, forecast.p50_delta, forecast.p90_delta) == (
        -3.0,
        -2.0,
        0.5,
    )
    assert forecast.confidence == 0.85
    assert selection.probability_valid == (0.93,)
    assert "exact compiled a/b action" in selection.rationales[0]
    assert "All claims refer only to the exact compiled action." in (
        selection.slate_rationale
    )
    proposal = materialize_hierarchical_residual_plan(
        schema=request.action_schema,
        plan=selection.plans[0],
        target_candidate_id=CandidateId("candidate_child"),
    )
    assert proposal.configuration
    assert receipts[0]["projected_member_count"] == 1
    assert receipts[0]["unordered_forecast_count"] == 1
    assert receipts[0]["whole_slate_retry_avoided"] is True
    assert receipts[0]["schema_version"] == 2
    assert receipts[0]["postcompile_regrounding"]["status"] == "regrounded"
    member = receipts[0]["members"][0]
    assert member["semantic_fidelity"] == "projected_claims_regrounded"
    assert (
        member["original_claims_sha256"]
        != member["allocation_claims_sha256"]
    )


def test_postcompile_schema_cannot_change_or_reorder_action_identity() -> None:
    output_type = postcompile_semantic_regrounding_output_type(
        projected_member_indices=(1, 3),
        required_metric_ids=("m1",),
    )
    member = {
        "semantic_fidelity_acknowledgement": "exact_compiled_action",
        "probability_valid": 0.5,
        "effect_predictions": [
            {
                "metric_id": "m1",
                "p10_delta": -1.0,
                "p50_delta": 0.0,
                "p90_delta": 1.0,
                "confidence": 0.5,
            }
        ],
        "interaction_rationale": "Grounded only in the exact compiled action.",
    }
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                "members": [
                    {"member_index": 3, **member},
                    {"member_index": 1, **member},
                ],
                "regrounding_rationale": "Reordered output is invalid.",
            }
        )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                "members": [
                    {
                        "member_index": 1,
                        "component_option_ids": ["forged.option"],
                        **member,
                    },
                    {"member_index": 3, **member},
                ],
                "regrounding_rationale": "Executable edits are forbidden.",
            }
        )


def test_regrounding_failure_quarantines_claims_without_losing_action() -> None:
    asyncio.run(_exercise_fallback())


async def _exercise_fallback() -> None:
    request = _request()
    receipts: list[dict[str, object]] = []

    async def generate_once(structured_request):
        if structured_request.operation == "postcompile_semantic_regrounding":
            raise RuntimeError("fixture failure")
        value = structured_request.output_type.model_validate(
            _invalid_preference()
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider/model",
            resolved_model="provider/model",
            resolved_provider="provider",
            provider_response_id="response",
            finish_reason="tool_call",
            input_tokens=10,
            output_tokens=5,
            reasoning_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.001"),
            latency_ns=100,
        )

    policy = PydanticAIReconciledHierarchicalResidualProposalPolicy(
        generate_once,
        reconciliation_sink=receipts.append,
    )
    selection = await policy.select(request)

    assert len(selection.plans) == 1
    assert "field.b.one" in selection.plans[0].component_option_ids
    assert selection.probability_valid == (0.5,)
    forecast = selection.effect_predictions[0][0]
    assert (
        forecast.p10_delta,
        forecast.p50_delta,
        forecast.p90_delta,
        forecast.confidence,
    ) == (0.0, 0.0, 0.0, 0.0)
    assert "original semantic claims are quarantined" in (
        selection.rationales[0]
    )
    regrounding = receipts[0]["postcompile_regrounding"]
    assert regrounding["status"] == "quarantined_fallback"
    assert regrounding["failure_kind"] == "RuntimeError"

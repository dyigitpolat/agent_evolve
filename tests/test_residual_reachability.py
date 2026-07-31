from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal

import pytest
from pydantic import ValidationError

from agent_evolve.application.residual_reachability import (
    HierarchicalResidualPlan,
    ParentFiniteVariationBinding,
    ReachabilityAdmissionReason,
    ReachabilityCandidate,
    ResidualProposalRole,
    ResidualReachabilityBasisPolicy,
    bind_cross_parent_finite_action_schema,
    materialize_hierarchical_residual_plan,
    select_residual_reachability_basis,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.llm_task_queue import ValidationIssueReasonCode
from agent_evolve.domain.typed_json import freeze_json, thaw_json, typed_json_sha256
from agent_evolve.integrations.pydantic_ai.residual_reachability import (
    PydanticAIHierarchicalResidualProposalPolicy,
    HierarchicalResidualProposalRequest,
    hierarchical_residual_output_type,
)
from agent_evolve.integrations.pydantic_ai.semantic_decision_replay import (
    SealedHierarchicalResidualProposalReplayThenLivePolicy,
    hierarchical_residual_proposal_selection_from_record,
)
from agent_evolve.ports.structured_generator import StructuredGenerationResponse


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _candidate(
    ordinal: int,
    *,
    quality: bool = False,
    initial: bool = False,
    earned: bool = False,
    cell: str | None = None,
) -> ReachabilityCandidate:
    configuration = freeze_json({"a": ordinal, "b": 0})
    return ReachabilityCandidate(
        candidate_id=CandidateId(f"candidate_{ordinal:03d}"),
        configuration=configuration,
        phenotype_identity_sha256=typed_json_sha256(configuration),
        evaluation_ordinal=ordinal,
        structural_cell=cell or f"cell.{ordinal:03d}",
        quality_archive_member=quality,
        initial_design_member=initial,
        earned_positive_lineage=earned,
    )


def _contract(parent: dict[str, int], *, suffix: str) -> FiniteVariationContract:
    frozen_parent = freeze_json(parent)
    parent_sha = typed_json_sha256(frozen_parent)
    return FiniteVariationContract(
        catalog_id="test_catalog",
        catalog_version=1,
        catalog_definition_sha256=_sha("test-catalog"),
        parent_configuration=frozen_parent,
        options=(
            FiniteVariationOption(
                option_id="field.a.one",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json({"a": 1, "b": parent["b"]}),
                family="replace",
                description="Set field a to one.",
            ),
            FiniteVariationOption(
                option_id="field.b.one",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json({"a": parent["a"], "b": 1}),
                family="replace",
                description="Set field b to one.",
                metadata=(("compiled_plan_sha256", _sha(suffix)),),
            ),
        ),
    )


def test_dual_archive_retains_dominated_initial_anchor() -> None:
    candidates = (
        _candidate(1, quality=True, cell="cell.front"),
        _candidate(2, initial=True, cell="cell.seed"),
        _candidate(3, earned=True, cell="cell.lineage"),
        _candidate(4, cell="cell.cover"),
    )
    basis = select_residual_reachability_basis(
        candidates,
        ResidualReachabilityBasisPolicy(
            maximum_parents=4,
            maximum_quality_archive_parents=1,
            maximum_initial_design_parents=1,
            maximum_earned_lineage_parents=1,
            maximum_structural_cover_parents=1,
        ),
    )

    reasons = {
        member.candidate.candidate_id.value: set(member.admission_reasons)
        for member in basis.members
    }
    assert ReachabilityAdmissionReason.QUALITY_ARCHIVE in reasons["candidate_001"]
    assert ReachabilityAdmissionReason.INITIAL_DESIGN in reasons["candidate_002"]
    assert ReachabilityAdmissionReason.EARNED_LINEAGE in reasons["candidate_003"]
    assert ReachabilityAdmissionReason.STRUCTURAL_COVER in reasons["candidate_004"]


def test_hierarchical_radius_two_plan_is_engine_materialized() -> None:
    parent_id = CandidateId("candidate_parent")
    contract = _contract({"a": 0, "b": 0}, suffix="same")
    schema = bind_cross_parent_finite_action_schema(
        (ParentFiniteVariationBinding(parent_id, contract),)
    )
    plan = HierarchicalResidualPlan(
        parent_candidate_id=parent_id,
        parent_contract_sha256=contract.identity_sha256,
        action_schema_sha256=schema.schema_sha256,
        component_option_ids=("field.a.one", "field.b.one"),
        role=ResidualProposalRole.INTERACTION,
        expert_id="test_interaction_expert",
        expert_definition_sha256=_sha("expert"),
        native_rank=1,
        decision_receipt_sha256=_sha("decision"),
    )

    proposal = materialize_hierarchical_residual_plan(
        schema=schema,
        plan=plan,
        target_candidate_id=CandidateId("candidate_child"),
    )

    assert thaw_json(proposal.configuration) == {"a": 1, "b": 1}
    assert proposal.plan.radius == 2
    assert proposal.to_record()["configuration_sha256"] == typed_json_sha256(
        proposal.configuration
    )


def test_dynamic_output_accepts_safe_disjoint_radius_two_pair() -> None:
    parent_id = CandidateId("candidate_parent")
    contract = _contract({"a": 0, "b": 0}, suffix="same")
    schema = bind_cross_parent_finite_action_schema(
        (ParentFiniteVariationBinding(parent_id, contract),)
    )
    request = HierarchicalResidualProposalRequest(
        call_id=LLMCallId("call_test_residual"),
        operation="propose_residual",
        instruction="Select one safe interaction.",
        context=freeze_json({"front": []}),
        action_schema=schema,
        proposal_count=1,
        allowed_radii=(2,),
        allowed_roles=(ResidualProposalRole.INTERACTION,),
        required_metric_ids=("m1", "m2"),
        minimum_distinct_parents=1,
        expert_id="interaction_expert",
        expert_definition_sha256=_sha("expert"),
        max_output_tokens=1024,
        temperature=0.0,
    )
    output_type = hierarchical_residual_output_type(request)
    value = output_type.model_validate(
        {
            "members": [
                {
                    "parent_candidate_id": parent_id.value,
                    "component_option_ids": ["field.a.one", "field.b.one"],
                    "role": ResidualProposalRole.INTERACTION.value,
                    "probability_valid": 0.8,
                    "effect_predictions": [
                        {
                            "metric_id": "m1",
                            "p10_delta": -3.0,
                            "p50_delta": -2.0,
                            "p90_delta": -1.0,
                            "confidence": 0.7,
                        },
                        {
                            "metric_id": "m2",
                            "p10_delta": -1.0,
                            "p50_delta": 0.0,
                            "p90_delta": 1.0,
                            "confidence": 0.4,
                        },
                    ],
                    "interaction_rationale": "Combine disjoint field changes.",
                }
            ],
            "slate_rationale": "One safe interaction.",
        }
    )
    assert value.members[0].component_option_ids == [
        "field.a.one",
        "field.b.one",
    ]

    invalid_quantiles = value.model_dump(mode="python")
    invalid_quantiles["members"][0]["effect_predictions"][0].update(
        {
            "p10_delta": 1.0,
            "p50_delta": 0.0,
            "p90_delta": -1.0,
        }
    )
    with pytest.raises(ValidationError) as quantile_error:
        output_type.model_validate(invalid_quantiles)
    assert quantile_error.value.errors(include_url=False)[0]["type"] == (
        ValidationIssueReasonCode.RESIDUAL_QUANTILE_ORDER_VIOLATION.value
    )

    invalid_pair = value.model_dump(mode="python")
    invalid_pair["members"][0]["component_option_ids"] = [
        "field.a.one",
        "field.a.one",
    ]
    with pytest.raises(ValidationError) as option_error:
        output_type.model_validate(invalid_pair)
    assert option_error.value.errors(include_url=False)[0]["type"] == (
        ValidationIssueReasonCode.RESIDUAL_OPTION_CONTRACT_VIOLATION.value
    )


def test_cross_parent_schema_keeps_parent_specific_digests_in_contracts() -> None:
    first = ParentFiniteVariationBinding(
        CandidateId("candidate_parent_001"),
        _contract({"a": 0, "b": 0}, suffix="first"),
    )
    second = ParentFiniteVariationBinding(
        CandidateId("candidate_parent_002"),
        _contract({"a": 2, "b": 2}, suffix="second"),
    )

    schema = bind_cross_parent_finite_action_schema((first, second))

    assert len(schema.action_prompt_records) == 2
    evidence_keys = {
        value["option_id"]: value["parent_bound_evidence_metadata_keys"]
        for value in schema.action_prompt_records
    }
    assert evidence_keys == {
        "field.a.one": [],
        "field.b.one": ["compiled_plan_sha256"],
    }
    assert first.contract.identity_sha256 != second.contract.identity_sha256


def test_semantic_decision_replay_preserves_repaired_typed_selection() -> None:
    asyncio.run(_exercise_semantic_decision_replay())


async def _exercise_semantic_decision_replay() -> None:
    parent_id = CandidateId("candidate_parent")
    contract = _contract({"a": 0, "b": 0}, suffix="same")
    schema = bind_cross_parent_finite_action_schema(
        (ParentFiniteVariationBinding(parent_id, contract),)
    )

    def request(call_id: str) -> HierarchicalResidualProposalRequest:
        return HierarchicalResidualProposalRequest(
            call_id=LLMCallId(call_id),
            operation="propose_residual",
            instruction="Select one safe interaction.",
            context=freeze_json({"front": []}),
            action_schema=schema,
            proposal_count=1,
            allowed_radii=(2,),
            allowed_roles=(ResidualProposalRole.INTERACTION,),
            required_metric_ids=("m1", "m2"),
            minimum_distinct_parents=1,
            expert_id="interaction_expert",
            expert_definition_sha256=_sha("expert"),
            max_output_tokens=1024,
            temperature=0.0,
        )

    calls: list[str] = []

    async def generate_once(structured_request):
        calls.append(structured_request.call_id.value)
        output_type = structured_request.output_type
        value = output_type.model_validate(
            {
                "members": [
                    {
                        "parent_candidate_id": parent_id.value,
                        "component_option_ids": [
                            "field.a.one",
                            "field.b.one",
                        ],
                        "role": ResidualProposalRole.INTERACTION.value,
                        "probability_valid": 0.8,
                        "effect_predictions": [
                            {
                                "metric_id": "m1",
                                "p10_delta": -3.0,
                                "p50_delta": -2.0,
                                "p90_delta": -1.0,
                                "confidence": 0.7,
                            },
                            {
                                "metric_id": "m2",
                                "p10_delta": -1.0,
                                "p50_delta": 0.0,
                                "p90_delta": 1.0,
                                "confidence": 0.4,
                            },
                        ],
                        "interaction_rationale": (
                            "Combine disjoint field changes."
                        ),
                    }
                ],
                "slate_rationale": "One safe interaction.",
            }
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

    live = PydanticAIHierarchicalResidualProposalPolicy(generate_once)
    sealed_request = request("call_sealed")
    original = await live.select(sealed_request)
    assert calls == ["call_sealed"]
    restored = hierarchical_residual_proposal_selection_from_record(
        original.to_record(),
        telemetry=original.telemetry,
    )
    receipts: list[dict[str, object]] = []
    replay_then_live = (
        SealedHierarchicalResidualProposalReplayThenLivePolicy(
            source_id="test_source",
            source_identity_sha256=_sha("source"),
            sealed_selections=(restored,),
            live_policy=live,
            allowed_live_call_ids=("call_live",),
            decision_receipt_sink=receipts.append,
        )
    )

    replayed = await replay_then_live.select(sealed_request)
    assert replayed.to_record() == original.to_record()
    assert calls == ["call_sealed"]
    assert receipts[0]["decision"] == "replayed"
    replay_then_live.assert_consumed()

    live_result = await replay_then_live.select(request("call_live"))
    assert live_result.request_sha256 == request("call_live").request_sha256
    assert calls == ["call_sealed", "call_live"]
    assert receipts[1]["decision"] == "live_after_prefix"

    tampered = original.to_record()
    tampered["unexpected"] = True
    with pytest.raises(ValueError, match="canonical"):
        hierarchical_residual_proposal_selection_from_record(
            tampered,
            telemetry=original.telemetry,
        )

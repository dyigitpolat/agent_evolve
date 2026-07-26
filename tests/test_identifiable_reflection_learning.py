"""Adversarial provider-free tests for identifiable reflection learning."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json

import pytest

import agent_evolve.application as application_api
from agent_evolve.application.campaign_learning_runtime import (
    CampaignReflectionLearningRecord,
    CampaignReflectionLearningRecordCodec,
)
from agent_evolve.application.evolution_campaign import (
    CampaignReflectionWave,
    ReflectionLaunchMode,
    ReflectionVisibility,
)
from agent_evolve.application.identifiable_reflection_evidence import (
    project_identifiable_reflection_evidence,
)
from agent_evolve.application.identifiable_reflection_learning import (
    build_identifiable_campaign_reflection_learning_envelope,
    build_identifiable_campaign_reflection_learning_record,
)
from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
    build_identifiable_reflection_generation_request,
    identifiable_reflection_request_construction_record,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignIdentifiableReflectionEvidenceProjection,
    CampaignIdentifiableReflectionEvidenceQuery,
    CampaignIdentifiableReflectionInput,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.domain.typed_json import freeze_json, thaw_json
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceProvenance,
    InterventionIdentifiability,
    ObservedMetricEffect,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    ReflectionInsightKind,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _observation(
    name: str,
    *,
    action_semantics_definition_sha256: str | None = None,
) -> AuthenticatedHypothesisObservation:
    offset = 1 if name == "a" else 2
    action_definition = (
        _action_semantics_sha()
        if action_semantics_definition_sha256 is None
        else action_semantics_definition_sha256
    )
    parent = freeze_json({"x": offset, "untouched": [1, 2, 3]})
    child = freeze_json({"x": offset + 1, "untouched": [1, 2, 3]})
    return AuthenticatedHypothesisObservation(
        source_evidence_id=_sha(f"source:{name}"),
        event_index=1,
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        campaign_sha256=_sha("campaign"),
        parent_candidate_id=CandidateId(f"candidate_parent_{name}"),
        child_candidate_id=CandidateId(f"candidate_child_{name}"),
        operator_invocation_id=OperatorInvocationId(f"operator_{name}"),
        finite_contract_identity_sha256=_sha(f"finite-contract:{name}"),
        provenance=EvidenceProvenance.DIRECT_MUTATION,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha(f"wave:{name}"),
            estimand_unit=CausalEstimandUnit.WAVE,
        ),
        parent_configuration=parent,
        child_configuration=child,
        parent_configuration_sha256=(
            AuthenticatedHypothesisObservation.configuration_sha256(parent)
        ),
        child_configuration_sha256=(
            AuthenticatedHypothesisObservation.configuration_sha256(child)
        ),
        parent_outcome_sha256=_sha(f"parent-outcome:{name}"),
        child_outcome_sha256=_sha(f"child-outcome:{name}"),
        operator_family="typed_mutation",
        affected_paths=("$.x",),
        observed_action=freeze_json(
            {
                "schema_version": 2,
                "option_id": f"option.raise_x_{name}",
                "option_identity_sha256": _sha(f"option:{name}"),
                "finite_contract_identity_sha256": _sha(
                    f"finite-contract:{name}"
                ),
                "option_family": "coordinate",
                "operator_family": "typed_mutation",
                "changed_paths": ["$.x"],
                "compiler": {
                    "compiler_id": "test_action_semantics",
                    "compiler_version": 1,
                    "definition_sha256": action_definition,
                },
            }
        ),
        action_semantics_compiler_id="test_action_semantics",
        action_semantics_compiler_version=1,
        action_semantics_definition_sha256=action_definition,
        intervention_identifiability=InterventionIdentifiability.EXACT_SINGLE,
        metrics=(
            ObservedMetricEffect(
                metric_id="loss",
                direction=MetricEffectDirection.DECREASE,
                delta=-0.25 * offset,
                adjudicator_definition_sha256=_sha("metric-adjudicator"),
            ),
        ),
        lineage_cluster_sha256=_sha(f"cluster:{name}"),
        factorial_block_sha256=_sha(f"block:{name}"),
    )


def _semantics(*, semantics_id: str = "identifiable_learning_test") -> OptimizationSemantics:
    return OptimizationSemantics(
        semantics_id=semantics_id,
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:loss",
                name="loss",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Lower deterministic loss is better.",
                aggregation="One sealed evaluator result.",
                witness_interpretation="Raw child minus current-parent loss.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=("objective:loss",),
            description="Minimize the published objective.",
            equivalence="Exact binary64 equality.",
            policy_id="identifiable_test_relation",
            policy_version=1,
            definition_sha256=_sha("outcome-relation"),
        ),
    )


def _contract() -> ReflectionInsightContract:
    return ReflectionInsightContract(
        required_metric_ids=("loss",),
        allowed_option_families=("coordinate",),
        allowed_decision_paths=("$.x",),
        allowed_insight_kinds=(
            ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        ),
        allowed_consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        allowed_comparison_anchor_kinds=(
            MetricComparisonAnchorKind.CURRENT_PARENT,
        ),
        allowed_factor_capabilities=("coordinate",),
    )


def _input(
    *,
    second_action_semantics_definition_sha256: str | None = None,
) -> CampaignIdentifiableReflectionInput:
    snapshot = project_identifiable_reflection_evidence(
        (
            _observation("a"),
            _observation(
                "b",
                action_semantics_definition_sha256=(
                    second_action_semantics_definition_sha256
                ),
            ),
        ),
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
    )
    query = CampaignIdentifiableReflectionEvidenceQuery(
        reflection_request_sha256=_sha("campaign-reflection-request"),
        preparation_sha256=_sha("campaign-preparation"),
        runtime_start_receipt_sha256=_sha("runtime-start"),
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        wave=CampaignReflectionWave(
            source_generation=2,
            call_count=1,
            launch_mode=ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
            visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            promotion_barrier_generation=2,
        ),
        source_stage_receipt_sha256=_sha("source-stage:g2"),
        source_portfolio_generation=1,
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
    )
    projection = CampaignIdentifiableReflectionEvidenceProjection(
        query_sha256=query.query_sha256,
        registry_snapshot_sha256=_sha("registry-at-g1"),
        registry_captured_through_event_index=1,
        evidence=snapshot,
    )
    return CampaignIdentifiableReflectionInput(query=query, source=projection)


def _request(reflection_input: CampaignIdentifiableReflectionInput):
    return build_identifiable_reflection_generation_request(
        call_id=LLMCallId("call_identifiable_learning"),
        evidence=reflection_input.evidence,
        insight_contract=_contract(),
        optimization_semantics=_semantics(),
        max_output_tokens=384_000,
        temperature=None,
        min_insights=1,
        max_insights=2,
    )


def _draft(contrast_ids: tuple[str, ...]) -> InsightDraft:
    return InsightDraft(
        claim="Increasing x by one predicted lower loss in both observations.",
        trigger="A coordinate finite action raises $.x by one.",
        mechanism="Prospective rationale: the local step may improve the objective.",
        affected_paths=("$.x",),
        evidence_summary="Two direct single-mutation observations lowered loss.",
        confidence=0.7,
        evidence_contrast_ids=contrast_ids,
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="loss",
                direction=MetricEffectDirection.DECREASE,
                comparison_anchor=MetricComparisonAnchor(
                    kind=MetricComparisonAnchorKind.CURRENT_PARENT,
                ),
            ),
        ),
        recommended_option_families=("coordinate",),
        action_template="Select one finite coordinate action affecting $.x.",
        falsification_condition="A later matched action does not decrease loss.",
        insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        factor_capabilities=("coordinate",),
    )


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="provider-free-model",
        resolved_model="provider-free-model",
        resolved_provider="provider-free",
        provider_response_id=None,
        finish_reason="stop",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=0,
    )


def _result(reflection_input: CampaignIdentifiableReflectionInput):
    request = _request(reflection_input)
    return ReflectionGenerationResult(
        insights=(_draft(request.available_contrast_ids),),
        telemetry=_telemetry(),
        evidence_catalog_identity_sha256=(
            request.evidence_catalog.catalog_identity_sha256
            if request.evidence_catalog is not None
            else None
        ),
    )


def _action_semantics_sha() -> str:
    return _sha("trusted-campaign-action-semantics-compiler")


def test_projects_exact_input_request_and_result_to_canonical_learning() -> None:
    reflection_input = _input()
    request = _request(reflection_input)
    result = _result(reflection_input)
    envelope = build_identifiable_campaign_reflection_learning_envelope(
        reflection_input=reflection_input,
        request=request,
        result=result,
        optimization_semantics=_semantics(),
    )
    record = CampaignReflectionLearningRecordCodec.decode(envelope)
    construction = identifiable_reflection_request_construction_record(
        request,
        reflection_input.evidence,
    )

    assert record.reflection_generation_request_sha256 == (
        construction["request_identity_sha256"]
    )
    assert record.reflection_call_id == request.call_id
    assert record.source_generation == 2
    assert record.source_stage_receipt_sha256 == _sha("source-stage:g2")
    assert record.origin_cutoff_event_index == 1
    assert record.source_operator_invocation_ids == tuple(
        sorted(value.operator_invocation_id for value in reflection_input.evidence.contrasts)
    )
    assert record.source_candidate_ids == tuple(
        sorted(
            candidate_id
            for value in reflection_input.evidence.contrasts
            for candidate_id in (value.parent_candidate_id, value.child_candidate_id)
        )
    )
    assert record.evidence_catalog == request.evidence_catalog
    assert record.insight_contract == request.insight_contract
    assert record.insights == result.insights
    assert tuple(value.contrast_id for value in record.finite_action_bindings) == (
        request.available_contrast_ids
    )
    assert tuple(value.contrast_id for value in record.empirical_evidence) == (
        request.available_contrast_ids
    )
    lineage = record.lineage_for(record.insights[0])
    assert lineage.source_operator_invocation_ids == (
        record.source_operator_invocation_ids
    )
    assert lineage.source_candidate_ids == record.source_candidate_ids
    assert lineage.available_contrast_ids == request.available_contrast_ids
    assert tuple(value.contrast_id for value in lineage.finite_action_bindings) == (
        request.available_contrast_ids
    )
    assert tuple(value.contrast_id for value in lineage.empirical_evidence) == (
        request.available_contrast_ids
    )
    assert CampaignReflectionLearningRecordCodec.encode(record) == envelope


def test_empirical_facts_are_direct_single_generic_and_fully_bound() -> None:
    reflection_input = _input()
    request = _request(reflection_input)
    result = _result(reflection_input)
    record = build_identifiable_campaign_reflection_learning_record(
        reflection_input=reflection_input,
        request=request,
        result=result,
        optimization_semantics=_semantics(),
    )
    decision_identity = DecisionMetricProjection.from_optimization_semantics(
        _semantics()
    ).definition_sha256
    construction_identity = identifiable_reflection_request_construction_record(
        request,
        reflection_input.evidence,
    )["request_identity_sha256"]
    contrasts = {
        value.contrast_id: value for value in reflection_input.evidence.contrasts
    }
    for snapshot, binding in zip(
        record.empirical_evidence,
        record.finite_action_bindings,
        strict=True,
    ):
        contrast = contrasts[snapshot.contrast_id]
        facts = thaw_json(snapshot.facts)
        assert snapshot.fact_schema_id == IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID
        assert snapshot.fact_schema_definition_sha256 == (
            IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256
        )
        assert snapshot.optimization_semantics_definition_sha256 == (
            _semantics().definition_sha256
        )
        assert snapshot.action_semantics_definition_sha256 == (
            _action_semantics_sha()
        )
        assert facts["design_kind"] == "direct_single_mutation"
        assert facts["mechanism_identifying_design"] is False
        assert facts["request_binding"][
            "decision_metric_projection_definition_sha256"
        ] == decision_identity
        assert facts["request_binding"][
            "identifiable_reflection_request_identity_sha256"
        ] == construction_identity
        assert facts["occurrence_lineage"] == {
            "parent_candidate_id": contrast.parent_candidate_id.value,
            "child_candidate_id": contrast.child_candidate_id.value,
            "operator_invocation_id": contrast.operator_invocation_id.value,
        }
        assert binding.contract_identity_sha256 == (
            contrast.finite_contract_identity_sha256
        )
        assert binding.option_identity_sha256 == contrast.option_identity_sha256

    serialized = json.dumps(thaw_json(
        build_identifiable_campaign_reflection_learning_envelope(
            reflection_input=reflection_input,
            request=request,
            result=result,
            optimization_semantics=_semantics(),
        )
    )).casefold()
    assert "recombination" not in serialized
    assert "observational_association" not in serialized


def test_rejects_handcrafted_request_and_foreign_optimization_semantics() -> None:
    reflection_input = _input()
    request = _request(reflection_input)
    result = _result(reflection_input)
    with pytest.raises(ValueError, match="canonical identifiable construction"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=replace(request, prompt=request.prompt + " "),
            result=result,
            optimization_semantics=_semantics(),
        )
    with pytest.raises(ValueError, match="canonical identifiable construction"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=result,
            optimization_semantics=_semantics(semantics_id="foreign_semantics"),
        )


def test_rejects_foreign_catalog_citation_and_non_empirical_result() -> None:
    reflection_input = _input()
    request = _request(reflection_input)
    result = _result(reflection_input)
    foreign_catalog = replace(
        result,
        evidence_catalog_identity_sha256=_sha("foreign-catalog"),
    )
    with pytest.raises(ValueError, match="foreign evidence catalog"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=foreign_catalog,
            optimization_semantics=_semantics(),
        )

    foreign_citation = replace(
        result,
        insights=(
            replace(result.insights[0], evidence_contrast_ids=(_sha("foreign"),)),
        ),
    )
    with pytest.raises(ValueError, match="citations escaped"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=foreign_citation,
            optimization_semantics=_semantics(),
        )

    non_empirical = replace(
        result,
        insights=(
            replace(
                result.insights[0],
                insight_kind=ReflectionInsightKind.MECHANISTIC_CONJECTURE,
            ),
        ),
    )
    with pytest.raises(ValueError, match="insight kind escapes"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=non_empirical,
            optimization_semantics=_semantics(),
        )


def test_rejects_empty_batch_and_count_escape() -> None:
    reflection_input = _input()
    request = _request(reflection_input)
    result = _result(reflection_input)
    with pytest.raises(ValueError, match="abstention is unsupported"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=replace(result, insights=()),
            optimization_semantics=_semantics(),
        )
    with pytest.raises(ValueError, match="count escaped"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=replace(result, insights=result.insights * 3),
            optimization_semantics=_semantics(),
        )


def test_rejects_mixed_authenticated_action_semantics_compilers() -> None:
    reflection_input = _input(
        second_action_semantics_definition_sha256=_sha(
            "different-trusted-action-compiler"
        )
    )
    request = _request(reflection_input)
    result = _result(reflection_input)
    with pytest.raises(ValueError, match="mixes action-semantics compiler"):
        build_identifiable_campaign_reflection_learning_record(
            reflection_input=reflection_input,
            request=request,
            result=result,
            optimization_semantics=_semantics(),
        )


def test_identifiable_reflection_application_api_exports_canonical_boundary() -> None:
    expected = {
        "CampaignIdentifiableReflectionEvidenceProjection": (
            CampaignIdentifiableReflectionEvidenceProjection
        ),
        "CampaignIdentifiableReflectionEvidenceQuery": (
            CampaignIdentifiableReflectionEvidenceQuery
        ),
        "CampaignIdentifiableReflectionInput": CampaignIdentifiableReflectionInput,
        "CampaignReflectionLearningRecordCodec": (
            CampaignReflectionLearningRecordCodec
        ),
        "CampaignReflectionLearningRecord": CampaignReflectionLearningRecord,
        "build_identifiable_campaign_reflection_learning_envelope": (
            build_identifiable_campaign_reflection_learning_envelope
        ),
        "build_identifiable_campaign_reflection_learning_record": (
            build_identifiable_campaign_reflection_learning_record
        ),
        "build_identifiable_reflection_generation_request": (
            build_identifiable_reflection_generation_request
        ),
        "identifiable_reflection_request_construction_record": (
            identifiable_reflection_request_construction_record
        ),
    }
    assert set(expected).issubset(application_api.__all__)
    for name, value in expected.items():
        assert getattr(application_api, name) is value

"""Provider/evaluator-free checks for BOiLS semantic-v3 reflection."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
import hashlib
import json

import pytest

from agent_evolve.application.campaign_learning_runtime import (
    CampaignReflectionLearningRecordCodec,
)
from agent_evolve.application.evolution_campaign import (
    CampaignReflectionWave,
    ReflectionLaunchMode,
    ReflectionVisibility,
)
from agent_evolve.application.identifiable_reflection_evidence import (
    ReflectionFalsificationFeedback,
    project_identifiable_reflection_evidence,
)
from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignIdentifiableReflectionEvidenceProjection,
    CampaignIdentifiableReflectionEvidenceQuery,
    CampaignIdentifiableReflectionInput,
)
from agent_evolve.application.portfolio_hypothesis_observations import (
    FinitePortfolioActionSemanticsCompiler,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.infrastructure.artifacts.in_memory import InMemoryArtifactStore
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
    ReflectionInsightKind,
)
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.campaign_reflection import (
    BoilsReflectionContrast,
    OBJECTIVE_IDS,
    REFLECTION_DECISION_PATHS,
    REFLECTION_OPTION_FAMILIES,
    boils_reflection_contract,
    boils_reflection_request_construction_record,
    build_boils_identifiable_reflection_learning_envelope,
    build_boils_identifiable_reflection_request,
    build_boils_reflection_generation_request,
)
from examples.benchmarks.boils_abc.detailed_evaluation import (
    create_current_sqrt_workload,
)
from examples.benchmarks.boils_abc.variation_catalog import ACTION_FAMILIES


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _contrast(ordinal: int) -> BoilsReflectionContrast:
    return BoilsReflectionContrast(
        contrast_id=_sha(f"boils-reflection-test-{ordinal}"),
        wave_ordinal=ordinal,
        selection_role=f"test_role_{ordinal}",
        source_option_ids=(f"boils_abc.p{ordinal:02d}.rewrite",),
        source_families=(REFLECTION_OPTION_FAMILIES[ordinal - 1],),
        source_parent_objectives=(
            _object({"total_levels": 71.0, "total_lut_count": 8028.0}),
            _object({"total_levels": 69.0, "total_lut_count": 7944.0}),
        ),
        target_objectives=_object(
            {
                "total_levels": float(69 - ordinal),
                "total_lut_count": float(7930 - ordinal),
            }
        ),
        reward_hex=(ordinal / 100).hex(),
        dominates_any_parent=ordinal == 1,
        better_than_any_parent=True,
    )


def _optimization_semantics():
    workload = create_current_sqrt_workload(
        artifact_store=InMemoryArtifactStore(),
    )
    semantics = workload.benchmark.optimization_semantics
    assert semantics is not None
    return semantics


def _feedback() -> ReflectionFalsificationFeedback:
    return ReflectionFalsificationFeedback(
        insight_content_sha256=_sha("deprecated-boils-insight"),
        applicable_workload_instance_sha256s=(_sha("boils-workload-instance"),),
        evaluator_contract_sha256=_sha("boils-evaluator-contract"),
        applicable_campaign_sha256s=(_sha("boils-campaign"),),
        audit_scope_sha256=_sha("boils-audit-scope"),
        available_event_index=1,
        affected_paths=("$.sequence[0]",),
        predictions=tuple(
            (metric_id, MetricEffectDirection.DECREASE)
            for metric_id in OBJECTIVE_IDS
        ),
        counterexample_source_evidence_ids=(_sha("boils-counterexample"),),
        semantic_audit_receipt_sha256=_sha("boils-semantic-audit"),
        lifecycle_decision_receipt_sha256=_sha("boils-lifecycle-decision"),
        deprecation_reason="Later direct mutation contradicted the prediction.",
    )


def _identifiable_input() -> CampaignIdentifiableReflectionInput:
    parent_sequence = list(DEFAULT_ACTION_SEQUENCE)
    child_sequence = list(DEFAULT_ACTION_SEQUENCE)
    child_sequence[0] = "rewrite"
    parent = freeze_json({"sequence": parent_sequence})
    child = freeze_json({"sequence": child_sequence})
    compiler = FinitePortfolioActionSemanticsCompiler()
    finite_contract_sha256 = _sha("boils-finite-contract")
    option_identity_sha256 = _sha("boils-finite-option")
    observation = AuthenticatedHypothesisObservation(
        source_evidence_id=_sha("boils-direct-mutation-source"),
        event_index=1,
        workload_instance_sha256=_sha("boils-workload-instance"),
        evaluator_contract_sha256=_sha("boils-evaluator-contract"),
        campaign_sha256=_sha("boils-campaign"),
        parent_candidate_id=CandidateId("candidate_boils_parent"),
        child_candidate_id=CandidateId("candidate_boils_child"),
        operator_invocation_id=OperatorInvocationId("operator_boils_mutation"),
        finite_contract_identity_sha256=finite_contract_sha256,
        provenance=EvidenceProvenance.DIRECT_MUTATION,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha("boils-wave"),
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
        parent_outcome_sha256=_sha("boils-parent-outcome"),
        child_outcome_sha256=_sha("boils-child-outcome"),
        operator_family="typed_mutation",
        affected_paths=("$.sequence[0]",),
        observed_action=freeze_json(
            {
                "schema_version": 2,
                "option_id": "boils_abc.sequence_00.rewrite",
                "option_identity_sha256": option_identity_sha256,
                "finite_contract_identity_sha256": finite_contract_sha256,
                "option_family": ACTION_FAMILIES["rewrite"],
                "operator_family": "typed_mutation",
                "changed_paths": ["$.sequence[0]"],
                "compiler": {
                    "compiler_id": compiler.compiler_id,
                    "compiler_version": compiler.compiler_version,
                    "definition_sha256": compiler.definition_sha256,
                },
            }
        ),
        action_semantics_compiler_id=compiler.compiler_id,
        action_semantics_compiler_version=compiler.compiler_version,
        action_semantics_definition_sha256=compiler.definition_sha256,
        intervention_identifiability=InterventionIdentifiability.EXACT_SINGLE,
        metrics=(
            ObservedMetricEffect(
                metric_id=OBJECTIVE_IDS[0],
                direction=MetricEffectDirection.DECREASE,
                delta=-2.0,
                adjudicator_definition_sha256=_sha("boils-metric-adjudicator"),
            ),
            ObservedMetricEffect(
                metric_id=OBJECTIVE_IDS[1],
                direction=MetricEffectDirection.DECREASE,
                delta=-100.0,
                adjudicator_definition_sha256=_sha("boils-metric-adjudicator"),
            ),
        ),
        lineage_cluster_sha256=_sha("boils-lineage-cluster"),
        factorial_block_sha256=_sha("boils-factorial-block"),
    )
    evidence = project_identifiable_reflection_evidence(
        (observation,),
        campaign_sha256=_sha("boils-campaign"),
        workload_instance_sha256=_sha("boils-workload-instance"),
        evaluator_contract_sha256=_sha("boils-evaluator-contract"),
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
        prior_falsifications=(_feedback(),),
    )
    query = CampaignIdentifiableReflectionEvidenceQuery(
        reflection_request_sha256=_sha("boils-campaign-reflection-request"),
        preparation_sha256=_sha("boils-campaign-preparation"),
        runtime_start_receipt_sha256=_sha("boils-runtime-start"),
        campaign_sha256=_sha("boils-campaign"),
        workload_instance_sha256=_sha("boils-workload-instance"),
        evaluator_contract_sha256=_sha("boils-evaluator-contract"),
        wave=CampaignReflectionWave(
            source_generation=2,
            call_count=1,
            launch_mode=ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
            visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            promotion_barrier_generation=2,
        ),
        source_stage_receipt_sha256=_sha("boils-source-stage"),
        source_portfolio_generation=1,
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
    )
    source = CampaignIdentifiableReflectionEvidenceProjection(
        query_sha256=query.query_sha256,
        registry_snapshot_sha256=_sha("boils-registry-snapshot"),
        registry_captured_through_event_index=1,
        evidence=evidence,
    )
    return CampaignIdentifiableReflectionInput(query=query, source=source)


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="provider-free",
        resolved_model="provider-free",
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


def _draft(
    contrast_ids: tuple[str, ...],
    *,
    insight_kind: ReflectionInsightKind = (
        ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE
    ),
) -> InsightDraft:
    return InsightDraft(
        claim="Replacing the first action with rewrite predicted lower BOiLS costs.",
        trigger="The finite selector can replace $.sequence[0] with rewrite.",
        mechanism=(
            "Prospective rationale: early rewriting may expose a smaller mapped AIG."
        ),
        affected_paths=("$.sequence[0]",),
        evidence_summary="One direct single-mutation observation lowered both metrics.",
        confidence=0.6,
        evidence_contrast_ids=contrast_ids,
        effect_predictions=tuple(
            MetricEffectPrediction(
                metric_id=metric_id,
                direction=MetricEffectDirection.DECREASE,
                comparison_anchor=MetricComparisonAnchor(
                    kind=MetricComparisonAnchorKind.CURRENT_PARENT,
                ),
            )
            for metric_id in OBJECTIVE_IDS
        ),
        recommended_option_families=(ACTION_FAMILIES["rewrite"],),
        recommended_option_ids=("boils_abc.sequence_00.rewrite",),
        action_template="Select one finite rewrite action at $.sequence[0].",
        falsification_condition=(
            "A later matched rewrite action fails to decrease either metric."
        ),
        insight_kind=insight_kind,
        consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        factor_capabilities=(ACTION_FAMILIES["rewrite"],),
    )


class _ProviderFreeGenerator:
    def __init__(
        self,
        *,
        insight_kind: ReflectionInsightKind = (
            ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE
        ),
        foreign_citation: bool = False,
    ) -> None:
        self.insight_kind = insight_kind
        self.foreign_citation = foreign_citation
        self.requests = []

    async def reflect(self, request):
        self.requests.append(request)
        contrast_ids = (
            (_sha("foreign-boils-contrast"),)
            if self.foreign_citation
            else request.available_contrast_ids
        )
        catalog = request.evidence_catalog
        assert catalog is not None
        return ReflectionGenerationResult(
            insights=(
                _draft(contrast_ids, insight_kind=self.insight_kind),
            ),
            telemetry=_telemetry(),
            evidence_catalog_identity_sha256=catalog.catalog_identity_sha256,
        )


def test_boils_reflection_request_is_closed_semantic_v3() -> None:
    contrasts = (_contrast(1), _contrast(2))
    with pytest.warns(DeprecationWarning, match="recombination-derived"):
        request = build_boils_reflection_generation_request(
            call_id=LLMCallId("call_boils_reflection_test"),
            contrasts=contrasts,
            allowed_option_families=tuple(
                sorted({value.source_families[0] for value in contrasts})
            ),
            max_output_tokens=384_000,
            temperature=0.2,
        )
    contract = request.insight_contract
    catalog = request.evidence_catalog
    assert contract is not None and contract.is_semantic_v3
    assert catalog is not None
    assert contract.required_metric_ids == OBJECTIVE_IDS
    assert contract.allowed_decision_paths == REFLECTION_DECISION_PATHS
    assert contract.allowed_insight_kinds == (
        ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
    )
    assert request.max_output_tokens == 384_000
    assert request.available_contrast_ids == tuple(
        sorted(value.contrast_id for value in contrasts)
    )

    with pytest.warns(DeprecationWarning, match="recombination-derived"):
        record = boils_reflection_request_construction_record(request)
    assert record["exact_evidence_citation_mapping"] is True
    assert record["no_legacy_evidence_key"] is True
    assert record["insight_contract_identity_sha256"] == contract.identity_sha256
    assert [value["evidence_citation_key"] for value in record["evidence_citation_mapping"]] == [
        "e0001",
        "e0002",
    ]


def test_boils_reflection_contract_rejects_foreign_family() -> None:
    with pytest.raises(ValueError, match="BOiLS subset"):
        boils_reflection_contract(("foreign_family",))


def test_identifiable_boils_path_returns_generic_direct_learning_envelope() -> None:
    reflection_input = _identifiable_input()
    semantics = _optimization_semantics()
    request = build_boils_identifiable_reflection_request(
        call_id=LLMCallId("call_boils_identifiable_reflection"),
        reflection_input=reflection_input,
        optimization_semantics=semantics,
        max_output_tokens=384_000,
        temperature=None,
    )
    generator = _ProviderFreeGenerator()
    result = asyncio.run(generator.reflect(request))
    envelope = build_boils_identifiable_reflection_learning_envelope(
        reflection_input=reflection_input,
        request=request,
        result=result,
        optimization_semantics=semantics,
    )
    learning = CampaignReflectionLearningRecordCodec.decode(envelope)
    assert "recombination" not in json.dumps(thaw_json(envelope)).casefold()

    assert generator.requests == [request]
    assert request.insight_contract is not None
    assert request.insight_contract.allowed_option_ids == (
        "boils_abc.sequence_00.rewrite",
    )
    assert request.operation == "extract_identifiable_insights"
    prompt = json.loads(request.prompt)
    assert "identifiable_mutation_contrasts" in prompt
    assert "contrasts" not in prompt
    assert "recombination" not in request.prompt.casefold()
    assert prompt["prior_falsifications"][0]["affected_paths"] == [
        "$.sequence[0]"
    ]
    evidence_row = prompt["identifiable_mutation_contrasts"][0]
    assert evidence_row["affected_path"] == "$.sequence[0]"
    assert evidence_row["option_family"] == ACTION_FAMILIES["rewrite"]
    assert tuple(
        value["metric_id"] for value in evidence_row["metric_effects"]
    ) == OBJECTIVE_IDS
    assert all(
        value["direction"] == MetricEffectDirection.DECREASE.value
        for value in evidence_row["metric_effects"]
    )
    assert reflection_input.evidence.contrasts[0].contrast_id not in request.prompt

    contrast = reflection_input.evidence.contrasts[0]
    assert learning.source_generation == 2
    assert learning.source_stage_receipt_sha256 == _sha("boils-source-stage")
    assert learning.origin_cutoff_event_index == 1
    assert learning.source_candidate_ids == tuple(
        sorted((contrast.parent_candidate_id, contrast.child_candidate_id))
    )
    assert learning.source_operator_invocation_ids == (
        contrast.operator_invocation_id,
    )
    assert learning.insights[0].falsification_condition == (
        "A later matched rewrite action fails to decrease either metric."
    )
    assert tuple(
        prediction.direction for prediction in learning.insights[0].effect_predictions
    ) == tuple(MetricEffectDirection.DECREASE for _ in OBJECTIVE_IDS)
    assert learning.finite_action_bindings[0].contract_identity_sha256 == (
        contrast.finite_contract_identity_sha256
    )
    assert learning.finite_action_bindings[0].option_identity_sha256 == (
        contrast.option_identity_sha256
    )
    empirical = learning.empirical_evidence[0]
    facts = thaw_json(empirical.facts)
    compiler = FinitePortfolioActionSemanticsCompiler()
    assert empirical.fact_schema_id == IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID
    assert empirical.action_semantics_definition_sha256 == compiler.definition_sha256
    assert facts["design_kind"] == "direct_single_mutation"
    assert facts["request_binding"]["action_semantics_compiler"] == {
        "compiler_id": compiler.compiler_id,
        "compiler_version": compiler.compiler_version,
        "definition_sha256": compiler.definition_sha256,
    }
    assert facts["observed_metric_effects"] == [
        value.to_record() for value in contrast.metrics
    ]


@pytest.mark.parametrize(
    ("generator", "message"),
    (
        (
            _ProviderFreeGenerator(
                insight_kind=ReflectionInsightKind.MECHANISTIC_CONJECTURE
            ),
            "insight kind escapes",
        ),
        (_ProviderFreeGenerator(foreign_citation=True), "citations escaped"),
    ),
)
def test_identifiable_boils_learning_rejects_provider_epistemic_escape(
    generator: _ProviderFreeGenerator,
    message: str,
) -> None:
    reflection_input = _identifiable_input()
    semantics = _optimization_semantics()
    request = build_boils_identifiable_reflection_request(
        call_id=LLMCallId("call_boils_adversarial_reflection"),
        reflection_input=reflection_input,
        optimization_semantics=semantics,
        max_output_tokens=384_000,
        temperature=None,
    )
    result = asyncio.run(generator.reflect(request))
    with pytest.raises(ValueError, match=message):
        build_boils_identifiable_reflection_learning_envelope(
            reflection_input=reflection_input,
            request=request,
            result=result,
            optimization_semantics=semantics,
        )


def test_identifiable_boils_adapter_rejects_foreign_metric_semantics() -> None:
    semantics = _optimization_semantics()
    first, second = semantics.metrics
    foreign_first = replace(
        first,
        metric_id="objective:foreign_cost",
        name="foreign_cost",
    )
    foreign_ordering = replace(
        semantics.outcome_ordering,
        metric_priority=(foreign_first.metric_id, second.metric_id),
    )
    foreign = replace(
        semantics,
        semantics_id="foreign_boils_semantics",
        metrics=(foreign_first, second),
        outcome_ordering=foreign_ordering,
    )
    with pytest.raises(ValueError, match="differ from BOiLS objectives"):
        build_boils_identifiable_reflection_request(
            call_id=LLMCallId("call_boils_foreign_semantics"),
            reflection_input=_identifiable_input(),
            optimization_semantics=foreign,
            max_output_tokens=384_000,
            temperature=None,
        )

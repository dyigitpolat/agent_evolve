from __future__ import annotations

import hashlib

from agent_evolve.application.finite_action_hypothesis_semantics import (
    PortableFiniteActionHypothesisMatcher,
    PortableFiniteActionInsightSemanticCompiler,
)
from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
)
from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightEvidenceLineage,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.domain.insight import InsightId, InsightRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceProvenance,
    GlobalEvidenceRegistrySnapshot,
    GlobalHypothesisAuditRequest,
    GlobalHypothesisFalsificationGate,
    GlobalHypothesisVerdict,
    HypothesisAuditScope,
    HypothesisClaimStrength,
    HypothesisMetricPrediction,
    InterventionIdentifiability,
    InterventionMatch,
    ObservedMetricEffect,
    TriggerMatch,
    TypedInterventionSignature,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionInsightContract,
    ReflectionInsightKind,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _direct_snapshot() -> EmpiricalEvidenceSnapshot:
    facts = _object(
        {
            "schema_version": 1,
            "design_kind": "direct_single_mutation",
            "comparison_anchor": "current_parent",
            "mechanism_identifying_design": False,
            "permitted_insight_kinds": ["empirical_predictive_rule"],
            "request_binding": {
                "campaign_identifiable_reflection_input_sha256": _sha("input"),
                "identifiable_reflection_request_identity_sha256": _sha("request"),
                "evidence_snapshot_sha256": _sha("window"),
                "evidence_catalog_identity_sha256": _sha("catalog"),
                "insight_contract_identity_sha256": _sha("contract"),
                "decision_metric_projection_definition_sha256": _sha("metrics"),
                "action_semantics_compiler": {
                    "compiler_id": "test_action_semantics",
                    "compiler_version": 1,
                    "definition_sha256": _sha("action-semantics-definition"),
                },
            },
            "source_scope": {
                "source_observation_sha256": _sha("source-observation"),
                "source_evidence_id": _sha("source-evidence"),
                "event_index": 1,
                "workload_instance_sha256": _sha("workload"),
                "evaluator_contract_sha256": _sha("evaluator"),
                "campaign_sha256": _sha("campaign"),
                "evidence_citation_key": "e0001",
            },
            "occurrence_lineage": {
                "parent_candidate_id": "candidate_source_parent",
                "child_candidate_id": "candidate_source_child",
                "operator_invocation_id": "operator_source",
            },
            "finite_action": {
                "option_id": "alpha.x1",
                "option_identity_sha256": _sha("source-option"),
                "option_family": "alpha",
                "finite_contract_identity_sha256": _sha("source-contract"),
            },
            "local_intervention": {
                "affected_path": "$.x",
                "parent_value": 1,
                "child_value": 2,
            },
            "configuration_lineage": {
                "parent_configuration_sha256": _sha("source-parent-config"),
                "child_configuration_sha256": _sha("source-child-config"),
            },
            "outcome_lineage": {
                "parent_outcome_sha256": _sha("source-parent-outcome"),
                "child_outcome_sha256": _sha("source-child-outcome"),
            },
            "observed_metric_effects": [],
        }
    )
    return EmpiricalEvidenceSnapshot(
        contrast_id=_sha("contrast"),
        fact_schema_id=IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
        fact_schema_version=IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
        fact_schema_definition_sha256=(
            IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256
        ),
        facts=facts,
        optimization_semantics_definition_sha256=_sha("optimization-semantics"),
        action_semantics_definition_sha256=_sha("action-semantics-definition"),
    )


def _compiled(*, parent_conditioned: bool = False):
    draft = InsightDraft(
        claim="Family alpha should increase quality.",
        trigger="A parent can accept the alpha action.",
        mechanism="The alpha edit exposes a useful implementation choice.",
        affected_paths=("$.x",),
        evidence_summary="One source-stage contrast motivated a future test.",
        confidence=0.5,
        evidence_contrast_ids=(_sha("contrast"),),
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="quality",
                direction=MetricEffectDirection.INCREASE,
                comparison_anchor=MetricComparisonAnchor(
                    MetricComparisonAnchorKind.CURRENT_PARENT
                ),
            ),
        ),
        recommended_option_families=("alpha",),
        recommended_option_ids=("alpha.x1",),
        action_template="Apply one alpha finite action.",
        falsification_condition="Quality does not increase.",
        insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        factor_capabilities=("alpha",),
    )
    contract = ReflectionInsightContract(
        required_metric_ids=("quality",),
        allowed_option_families=("alpha", "beta"),
        allowed_decision_paths=("$.x", "$.y"),
        allowed_insight_kinds=(ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,),
        allowed_consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        allowed_comparison_anchor_kinds=(MetricComparisonAnchorKind.CURRENT_PARENT,),
        allowed_factor_capabilities=("alpha", "beta"),
    )
    lineage = InsightEvidenceLineage(
        reflection_call_id=LLMCallId("call_portable_semantics"),
        source_operator_invocation_ids=(
            OperatorInvocationId("operator_portable_semantics"),
        ),
        source_candidate_ids=(CandidateId("candidate_portable_semantics"),),
        available_contrast_ids=(_sha("contrast"),),
        cited_contrast_ids=(_sha("contrast"),),
        empirical_evidence=((_direct_snapshot(),) if parent_conditioned else ()),
    )
    compiled = PortableFiniteActionInsightSemanticCompiler().compile(
        draft=draft,
        insight_contract=contract,
        evidence_lineage=lineage,
    )
    return draft, compiled


def _observation(
    *,
    family: str,
    delta: float,
    source: str,
    parent_x: int = 0,
    child_x: int | None = None,
):
    parent = _object({"x": parent_x, "y": 0})
    child = _object(
        {"x": (1 if delta > 0 else -1) if child_x is None else child_x, "y": 0}
    )
    direction = (
        MetricEffectDirection.INCREASE if delta > 0 else MetricEffectDirection.DECREASE
    )
    return AuthenticatedHypothesisObservation(
        source_evidence_id=_sha(source),
        event_index=3,
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        campaign_sha256=_sha("campaign"),
        parent_candidate_id=CandidateId(f"candidate_parent_{source}"),
        child_candidate_id=CandidateId(f"candidate_child_{source}"),
        operator_invocation_id=OperatorInvocationId(f"operator_{source}"),
        finite_contract_identity_sha256=_sha("finite-contract"),
        provenance=EvidenceProvenance.DIRECT_MUTATION,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha(f"wave:{source}"),
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
        parent_outcome_sha256=_sha(f"parent:{source}"),
        child_outcome_sha256=_sha(f"child:{source}"),
        operator_family="typed_mutation",
        affected_paths=("$.x",),
        observed_action=_object(
            {
                "option_id": f"{family}.x1",
                "option_family": family,
                "operator_family": "typed_mutation",
                "changed_paths": ["$.x"],
            }
        ),
        action_semantics_compiler_id="test_action_semantics",
        action_semantics_compiler_version=1,
        action_semantics_definition_sha256=_sha("action-semantics-definition"),
        intervention_identifiability=InterventionIdentifiability.EXACT_SINGLE,
        metrics=(
            ObservedMetricEffect(
                metric_id="quality",
                direction=direction,
                delta=delta,
                adjudicator_definition_sha256=_sha("adjudicator"),
            ),
        ),
        lineage_cluster_sha256=_sha(f"lineage:{source}"),
        factorial_block_sha256=_sha(f"block:{source}"),
    )


def test_portable_compiler_and_real_gate_support_then_counterexample() -> None:
    draft, compiled = _compiled()
    support = _observation(family="alpha", delta=1.0, source="support")
    counterexample = _observation(family="alpha", delta=-1.0, source="counterexample")
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=3,
        observations=(support, counterexample),
    )
    request = GlobalHypothesisAuditRequest(
        reference=InsightRef(InsightId("insight_portable"), 1),
        draft_content_sha256=draft.content_sha256,
        trigger=compiled.trigger,
        intervention=TypedInterventionSignature(
            affected_paths=draft.affected_paths,
            old_value_predicate=compiled.old_value_predicate,
            new_action=compiled.new_action,
            admissible_operator_families=("typed_mutation",),
        ),
        predictions=(
            HypothesisMetricPrediction(
                metric_id="quality",
                direction=MetricEffectDirection.INCREASE,
            ),
        ),
        claim_strength=HypothesisClaimStrength(),
        scope=HypothesisAuditScope(
            workload_instance_sha256s=(_sha("workload"),),
            evaluator_contract_sha256=_sha("evaluator"),
            metric_adjudicator_definition_sha256=_sha("adjudicator"),
            campaign_sha256s=(_sha("campaign"),),
        ),
        matcher_definition_sha256=compiled.matcher_definition_sha256,
        origin_cutoff_event_index=2,
        audit_cutoff_event_index=3,
        registry_snapshot_sha256=registry.snapshot_sha256,
        minimum_support_clusters=1,
        minimum_support_instances=1,
    )
    receipt = GlobalHypothesisFalsificationGate().audit(
        request=request,
        registry=registry,
        matcher=PortableFiniteActionHypothesisMatcher(),
    )
    assert receipt.verdict is GlobalHypothesisVerdict.COUNTEREXAMPLE
    assert receipt.support_ids == (support.source_evidence_id,)
    assert receipt.counterexample_ids == (counterexample.source_evidence_id,)


def test_foreign_family_is_near_not_support_or_counterexample() -> None:
    draft, compiled = _compiled()
    observation = _observation(family="beta", delta=-1.0, source="beta")
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=3,
        observations=(observation,),
    )
    request = GlobalHypothesisAuditRequest(
        reference=InsightRef(InsightId("insight_portable"), 1),
        draft_content_sha256=draft.content_sha256,
        trigger=compiled.trigger,
        intervention=TypedInterventionSignature(
            affected_paths=draft.affected_paths,
            old_value_predicate=compiled.old_value_predicate,
            new_action=compiled.new_action,
            admissible_operator_families=("typed_mutation",),
        ),
        predictions=(
            HypothesisMetricPrediction(
                metric_id="quality",
                direction=MetricEffectDirection.INCREASE,
            ),
        ),
        claim_strength=HypothesisClaimStrength(),
        scope=HypothesisAuditScope(
            workload_instance_sha256s=(_sha("workload"),),
            evaluator_contract_sha256=_sha("evaluator"),
            metric_adjudicator_definition_sha256=_sha("adjudicator"),
            campaign_sha256s=(_sha("campaign"),),
        ),
        matcher_definition_sha256=compiled.matcher_definition_sha256,
        origin_cutoff_event_index=2,
        audit_cutoff_event_index=3,
        registry_snapshot_sha256=registry.snapshot_sha256,
        minimum_support_clusters=1,
        minimum_support_instances=1,
    )
    receipt = GlobalHypothesisFalsificationGate().audit(
        request=request,
        registry=registry,
        matcher=PortableFiniteActionHypothesisMatcher(),
    )
    assert receipt.verdict is GlobalHypothesisVerdict.INSUFFICIENT
    assert receipt.support_ids == ()
    assert receipt.counterexample_ids == ()


def test_authenticated_transition_is_parent_conditioned_and_child_exact() -> None:
    draft, compiled = _compiled(parent_conditioned=True)
    observations = (
        _observation(
            family="alpha",
            delta=1.0,
            source="matching-transition",
            parent_x=1,
            child_x=2,
        ),
        _observation(
            family="alpha",
            delta=1.0,
            source="off-trigger-transition",
            parent_x=3,
            child_x=2,
        ),
        _observation(
            family="alpha",
            delta=1.0,
            source="wrong-child-transition",
            parent_x=1,
            child_x=3,
        ),
    )
    registry = GlobalEvidenceRegistrySnapshot.seal(
        captured_through_event_index=3,
        observations=observations,
    )
    request = GlobalHypothesisAuditRequest(
        reference=InsightRef(InsightId("insight_parent_conditioned"), 1),
        draft_content_sha256=draft.content_sha256,
        trigger=compiled.trigger,
        intervention=TypedInterventionSignature(
            affected_paths=draft.affected_paths,
            old_value_predicate=compiled.old_value_predicate,
            new_action=compiled.new_action,
            admissible_operator_families=("typed_mutation",),
        ),
        predictions=(
            HypothesisMetricPrediction(
                metric_id="quality",
                direction=MetricEffectDirection.INCREASE,
            ),
        ),
        claim_strength=HypothesisClaimStrength(),
        scope=HypothesisAuditScope(
            workload_instance_sha256s=(_sha("workload"),),
            evaluator_contract_sha256=_sha("evaluator"),
            metric_adjudicator_definition_sha256=_sha("adjudicator"),
            campaign_sha256s=(_sha("campaign"),),
        ),
        matcher_definition_sha256=compiled.matcher_definition_sha256,
        origin_cutoff_event_index=0,
        audit_cutoff_event_index=3,
        registry_snapshot_sha256=registry.snapshot_sha256,
        minimum_support_clusters=1,
        minimum_support_instances=1,
    )
    matcher = PortableFiniteActionHypothesisMatcher()
    receipts = tuple(matcher.classify(request, value) for value in observations)
    assert (
        receipts[0].trigger_match,
        receipts[0].intervention_match,
    ) == (TriggerMatch.EXACT, InterventionMatch.EXACT)
    assert (
        receipts[1].trigger_match,
        receipts[1].intervention_match,
    ) == (TriggerMatch.OFF_TRIGGER, InterventionMatch.EXACT)
    assert (
        receipts[2].trigger_match,
        receipts[2].intervention_match,
    ) == (TriggerMatch.EXACT, InterventionMatch.NEAR)

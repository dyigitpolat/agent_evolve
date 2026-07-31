"""Provider-free tests for mutation-only, falsification-aware reflection input."""

from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.identifiable_reflection_evidence import (
    NoIdentifiableMutationEvidenceError,
    ReflectionEvidenceExclusionReason,
    ReflectionFalsificationFeedback,
    cluster_identifiable_mutation_reflection_hypotheses,
    project_identifiable_reflection_evidence,
)
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceProvenance,
    InterventionIdentifiability,
    ObservedMetricEffect,
)
from agent_evolve.ports.agentic_generator import (
    MetricEffectDirection,
    ReflectionInsightKind,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _observation(
    *,
    name: str,
    event_index: int,
    path: str = "$.x",
    provenance: EvidenceProvenance = EvidenceProvenance.DIRECT_MUTATION,
    identifiability: InterventionIdentifiability = (
        InterventionIdentifiability.EXACT_SINGLE
    ),
    action_paths: tuple[str, ...] | None = None,
    campaign: str = "campaign",
    parent_configuration: dict[str, object] | None = None,
    child_configuration: dict[str, object] | None = None,
    observed_compiler_id: str = "finite_portfolio_action_semantics",
    observed_compiler_version: int = 2,
    observed_compiler_definition_sha256: str | None = None,
    action_name: str | None = None,
) -> AuthenticatedHypothesisObservation:
    parent = freeze_json(
        {"x": 1, "y": 1}
        if parent_configuration is None
        else parent_configuration
    )
    child = freeze_json(
        {"x": 2, "y": 1}
        if child_configuration is None
        else child_configuration
    )
    paths = (path,) if action_paths is None else action_paths
    trusted_compiler_definition = _sha("action-semantics-definition")
    observed_compiler_definition = (
        trusted_compiler_definition
        if observed_compiler_definition_sha256 is None
        else observed_compiler_definition_sha256
    )
    action = name if action_name is None else action_name
    return AuthenticatedHypothesisObservation(
        source_evidence_id=_sha(f"source-{name}"),
        event_index=event_index,
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        campaign_sha256=_sha(campaign),
        parent_candidate_id=CandidateId(f"candidate_parent_{name}"),
        child_candidate_id=CandidateId(f"candidate_child_{name}"),
        operator_invocation_id=OperatorInvocationId(f"operator_{name}"),
        finite_contract_identity_sha256=_sha("finite-contract"),
        provenance=provenance,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha(f"wave-{name}"),
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
        parent_outcome_sha256=_sha(f"parent-{name}"),
        child_outcome_sha256=_sha(f"child-{name}"),
        operator_family="typed_mutation",
        affected_paths=(path,),
        observed_action=freeze_json(
            {
                "schema_version": 2,
                "option_id": f"option.{action}",
                "option_identity_sha256": _sha(f"option-{action}"),
                "finite_contract_identity_sha256": _sha("finite-contract"),
                "option_family": "coordinate",
                "operator_family": "typed_mutation",
                "changed_paths": list(paths),
                "compiler": {
                    "compiler_id": observed_compiler_id,
                    "compiler_version": observed_compiler_version,
                    "definition_sha256": observed_compiler_definition,
                },
            }
        ),
        action_semantics_compiler_id="finite_portfolio_action_semantics",
        action_semantics_compiler_version=2,
        action_semantics_definition_sha256=trusted_compiler_definition,
        intervention_identifiability=identifiability,
        metrics=(
            ObservedMetricEffect(
                metric_id="loss",
                direction=MetricEffectDirection.DECREASE,
                delta=-1.0,
                adjudicator_definition_sha256=_sha("delta-adjudicator"),
            ),
        ),
        lineage_cluster_sha256=_sha(f"cluster-{name}"),
        factorial_block_sha256=_sha(f"block-{name}"),
        mechanism_identifying_design=False,
    )


def _feedback(
    *,
    workload: str = "workload",
    evaluator: str = "evaluator",
    campaign: str = "campaign",
    available_event_index: int = 1,
) -> ReflectionFalsificationFeedback:
    return ReflectionFalsificationFeedback(
        insight_content_sha256=_sha("prior-insight"),
        applicable_workload_instance_sha256s=(_sha(workload),),
        evaluator_contract_sha256=_sha(evaluator),
        applicable_campaign_sha256s=(_sha(campaign),),
        audit_scope_sha256=_sha("audit-scope"),
        available_event_index=available_event_index,
        affected_paths=("$.x",),
        predictions=(("loss", MetricEffectDirection.DECREASE),),
        counterexample_source_evidence_ids=tuple(
            sorted((_sha("counterexample-a"), _sha("counterexample-b")))
        ),
        semantic_audit_receipt_sha256=_sha("semantic-audit"),
        lifecycle_decision_receipt_sha256=_sha("lifecycle-decision"),
        deprecation_reason="Global semantic audit found a counterexample.",
    )


def test_projection_uses_only_new_exact_single_mutations_at_sealed_cutoff() -> None:
    observations = (
        _observation(name="old", event_index=1),
        _observation(name="accepted", event_index=2),
        _observation(
            name="joint",
            event_index=2,
            identifiability=InterventionIdentifiability.JOINT_WITHOUT_ABLATION,
        ),
        _observation(
            name="recombined",
            event_index=2,
            provenance=EvidenceProvenance.OBSERVATIONAL_ASSOCIATION,
        ),
        _observation(name="future", event_index=3),
        _observation(name="foreign", event_index=2, campaign="foreign"),
    )
    snapshot = project_identifiable_reflection_evidence(
        observations,
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        prior_cutoff_event_index_exclusive=1,
        sealed_cutoff_event_index_inclusive=2,
    )
    assert len(snapshot.contrasts) == 1
    contrast = snapshot.contrasts[0]
    assert contrast.parent_candidate_id == CandidateId("candidate_parent_accepted")
    assert contrast.child_candidate_id == CandidateId("candidate_child_accepted")
    assert contrast.operator_invocation_id == OperatorInvocationId(
        "operator_accepted"
    )
    assert contrast.finite_contract_identity_sha256 == _sha("finite-contract")
    assert (
        contrast.action_semantics_compiler_id
        == "finite_portfolio_action_semantics"
    )
    assert contrast.action_semantics_compiler_version == 2
    assert contrast.action_semantics_definition_sha256 == _sha(
        "action-semantics-definition"
    )
    assert contrast.to_record()["action_semantics_compiler"] == {
        "compiler_id": "finite_portfolio_action_semantics",
        "compiler_version": 2,
        "definition_sha256": _sha("action-semantics-definition"),
    }
    assert contrast.option_id == "option.accepted"
    assert contrast.affected_path == "$.x"
    assert contrast.permitted_insight_kinds == (
        ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
    )
    assert contrast.to_prompt_record(evidence_citation_key="evidence.01") == {
        "evidence_citation_key": "evidence.01",
        "event_index": 2,
        "option_id": "option.accepted",
        "option_family": "coordinate",
        "affected_path": "$.x",
        "local_intervention": {
            "parent_value": 1,
            "child_value": 2,
        },
        "metric_effects": [
            {
                "metric_id": "loss",
                "direction": "decrease",
                "delta_decimal": "-1.0",
                "delta_hex": "-0x1.0000000000000p+0",
            }
        ],
        "permitted_insight_kinds": ["empirical_predictive_rule"],
        "comparison_anchor": "current_parent",
    }
    exclusions = dict(snapshot.exclusions)
    assert exclusions == {
        ReflectionEvidenceExclusionReason.AFTER_SEALED_CUTOFF: 1,
        ReflectionEvidenceExclusionReason.BEFORE_OR_AT_PRIOR_CUTOFF: 1,
        ReflectionEvidenceExclusionReason.FOREIGN_SCOPE: 1,
        ReflectionEvidenceExclusionReason.NON_MUTATION_PROVENANCE: 1,
        ReflectionEvidenceExclusionReason.NON_SINGLE_INTERVENTION: 1,
    }


def test_repeated_interventions_cluster_as_one_empirical_hypothesis() -> None:
    snapshot = project_identifiable_reflection_evidence(
        (
            _observation(name="repeat-a", action_name="shared", event_index=2),
            _observation(name="repeat-b", action_name="shared", event_index=2),
            _observation(
                name="different-local-anchor",
                action_name="shared",
                event_index=2,
                parent_configuration={"x": 0, "y": 1},
                child_configuration={"x": 2, "y": 1},
            ),
        ),
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        prior_cutoff_event_index_exclusive=1,
        sealed_cutoff_event_index_inclusive=2,
    )

    clusters = cluster_identifiable_mutation_reflection_hypotheses(
        snapshot.contrasts
    )
    assert len(clusters) == 2
    repeated = next(value for value in clusters if len(value.contrasts) == 2)
    assert {contrast.parent_candidate_id.value for contrast in repeated.contrasts} == {
        "candidate_parent_repeat-a",
        "candidate_parent_repeat-b",
    }
    assert repeated.contrast_ids == tuple(sorted(repeated.contrast_ids))
    assert repeated.to_record()["observation_count"] == 2
    assert repeated.to_record()["hypothesis_signature"]["finite_action"][
        "option_id"
    ] == "option.shared"


def test_projection_excludes_oversized_local_intervention_values() -> None:
    snapshot = project_identifiable_reflection_evidence(
        (
            _observation(name="accepted", event_index=2),
            _observation(
                name="oversized",
                event_index=2,
                path="$.blob",
                parent_configuration={"blob": "a" * 5_000},
                child_configuration={"blob": "b" * 5_000},
            ),
        ),
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        prior_cutoff_event_index_exclusive=1,
        sealed_cutoff_event_index_inclusive=2,
    )
    assert len(snapshot.contrasts) == 1
    assert dict(snapshot.exclusions) == {
        ReflectionEvidenceExclusionReason.LOCAL_INTERVENTION_TOO_LARGE: 1,
    }


def test_action_semantics_path_mismatch_is_not_identifiable() -> None:
    with pytest.raises(
        NoIdentifiableMutationEvidenceError,
        match="no identifiable mutation evidence",
    ) as captured:
        project_identifiable_reflection_evidence(
            (_observation(name="mismatch", event_index=2, action_paths=("$.y",)),),
            campaign_sha256=_sha("campaign"),
            workload_instance_sha256=_sha("workload"),
            evaluator_contract_sha256=_sha("evaluator"),
            prior_cutoff_event_index_exclusive=1,
            sealed_cutoff_event_index_inclusive=2,
        )
    assert captured.value.to_record() == {
        "schema_version": 1,
        "evidence_tier": "e0",
        "status": "abstained_no_identifiable_mutation_evidence",
        "observation_count": 1,
        "identifiable_contrast_count": 0,
        "exclusions": [
            {
                "reason": "malformed_action_semantics",
                "count": 1,
            }
        ],
        "provider_calls": 0,
        "publishable_reflection_content": False,
    }


@pytest.mark.parametrize(
    "compiler_override",
    (
        {"observed_compiler_id": "foreign_action_semantics"},
        {"observed_compiler_version": 3},
        {"observed_compiler_definition_sha256": _sha("foreign-definition")},
    ),
)
def test_action_semantics_compiler_mismatch_is_not_identifiable(
    compiler_override: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="no identifiable mutation evidence"):
        project_identifiable_reflection_evidence(
            (
                _observation(
                    name="compiler-mismatch",
                    event_index=2,
                    **compiler_override,
                ),
            ),
            campaign_sha256=_sha("campaign"),
            workload_instance_sha256=_sha("workload"),
            evaluator_contract_sha256=_sha("evaluator"),
            prior_cutoff_event_index_exclusive=1,
            sealed_cutoff_event_index_inclusive=2,
        )


def test_prior_counterexample_feedback_is_hash_bound_and_prompt_bounded() -> None:
    feedback = _feedback()
    snapshot = project_identifiable_reflection_evidence(
        (_observation(name="accepted", event_index=2),),
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        prior_cutoff_event_index_exclusive=1,
        sealed_cutoff_event_index_inclusive=2,
        prior_falsifications=(feedback,),
    )
    assert snapshot.prior_falsifications == (feedback,)
    prompt = feedback.to_prompt_record()
    assert prompt["counterexample_count"] == 2
    assert "counterexample_source_evidence_ids" not in prompt
    assert "Do not repeat" in str(prompt["instruction"])


@pytest.mark.parametrize(
    "feedback",
    (
        _feedback(workload="foreign-workload"),
        _feedback(evaluator="foreign-evaluator"),
        _feedback(campaign="foreign-campaign"),
        _feedback(available_event_index=3),
    ),
)
def test_prior_feedback_cannot_escape_scope_or_sealed_cutoff(
    feedback: ReflectionFalsificationFeedback,
) -> None:
    with pytest.raises(ValueError, match="falsification escapes"):
        project_identifiable_reflection_evidence(
            (_observation(name="accepted", event_index=2),),
            campaign_sha256=_sha("campaign"),
            workload_instance_sha256=_sha("workload"),
            evaluator_contract_sha256=_sha("evaluator"),
            prior_cutoff_event_index_exclusive=1,
            sealed_cutoff_event_index_inclusive=2,
            prior_falsifications=(feedback,),
        )

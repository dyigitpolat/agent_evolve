"""Provider-free tests for the generic identifiable-reflection request."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from agent_evolve.application.identifiable_reflection_evidence import (
    project_identifiable_reflection_evidence,
)
from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
    IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION,
    bind_reflection_contract_to_evidence_actions,
    build_identifiable_reflection_generation_request,
    identifiable_reflection_request_construction_record,
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
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceProvenance,
    InterventionIdentifiability,
    ObservedMetricEffect,
)
from agent_evolve.ports.agentic_generator import (
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    ReflectionConsumerScope,
    ReflectionInsightContract,
    ReflectionInsightKind,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot():
    parent = freeze_json({"x": 1, "unrelated": [1, 2, 3]})
    child = freeze_json({"x": 2, "unrelated": [1, 2, 3]})
    observation = AuthenticatedHypothesisObservation(
        source_evidence_id=_sha("source"),
        event_index=1,
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        campaign_sha256=_sha("campaign"),
        parent_candidate_id=CandidateId("candidate_parent"),
        child_candidate_id=CandidateId("candidate_child"),
        operator_invocation_id=OperatorInvocationId("operator_mutation"),
        finite_contract_identity_sha256=_sha("finite-contract"),
        provenance=EvidenceProvenance.DIRECT_MUTATION,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha("wave"),
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
        parent_outcome_sha256=_sha("parent-outcome"),
        child_outcome_sha256=_sha("child-outcome"),
        operator_family="typed_mutation",
        affected_paths=("$.x",),
        observed_action=freeze_json(
            {
                "schema_version": 2,
                "option_id": "option.raise_x",
                "option_identity_sha256": _sha("option"),
                "finite_contract_identity_sha256": _sha("finite-contract"),
                "option_family": "coordinate",
                "operator_family": "typed_mutation",
                "changed_paths": ["$.x"],
                "compiler": {
                    "compiler_id": "finite_portfolio_action_semantics",
                    "compiler_version": 2,
                    "definition_sha256": _sha("action-semantics-definition"),
                },
            }
        ),
        action_semantics_compiler_id="finite_portfolio_action_semantics",
        action_semantics_compiler_version=2,
        action_semantics_definition_sha256=_sha("action-semantics-definition"),
        intervention_identifiability=InterventionIdentifiability.EXACT_SINGLE,
        metrics=(
            ObservedMetricEffect(
                metric_id="loss",
                direction=MetricEffectDirection.DECREASE,
                delta=-0.25,
                adjudicator_definition_sha256=_sha("metric-adjudicator"),
            ),
        ),
        lineage_cluster_sha256=_sha("cluster"),
        factorial_block_sha256=_sha("block"),
    )
    return project_identifiable_reflection_evidence(
        (observation,),
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
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


def _semantics() -> OptimizationSemantics:
    return OptimizationSemantics(
        semantics_id="test_identifiable_reflection",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:loss",
                name="loss",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Lower loss is better.",
                aggregation="One deterministic evaluator result.",
                witness_interpretation="Child minus current parent.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=("objective:loss",),
            description="Minimize the objective without scalar side channels.",
            equivalence="Exact binary64 equality.",
            policy_id="test_pareto_relation",
            policy_version=1,
            definition_sha256=_sha("relation"),
        ),
    )


def test_request_contains_only_bounded_local_intervention_and_short_citation() -> None:
    snapshot = _snapshot()
    request = build_identifiable_reflection_generation_request(
        call_id=LLMCallId("call_identifiable_reflection"),
        evidence=snapshot,
        insight_contract=_contract(),
        optimization_semantics=_semantics(),
        max_output_tokens=384_000,
        temperature=0.2,
    )
    assert request.available_contrast_ids == (
        snapshot.contrasts[0].contrast_id,
    )
    assert snapshot.contrasts[0].contrast_id not in request.prompt
    prompt = json.loads(request.prompt)
    assert prompt["identifiable_mutation_contrasts"] == [
        {
            "affected_path": "$.x",
            "comparison_anchor": "current_parent",
            "evidence_citation_key": "e0001",
            "event_index": 1,
            "local_intervention": {"child_value": 2, "parent_value": 1},
            "metric_effects": [
                {
                    "delta_decimal": "-0.25",
                    "delta_hex": "-0x1.0000000000000p-2",
                    "direction": "decrease",
                    "metric_id": "loss",
                }
            ],
            "option_family": "coordinate",
            "option_id": "option.raise_x",
            "permitted_insight_kinds": ["empirical_predictive_rule"],
        }
    ]
    assert "unrelated" not in request.prompt
    assert prompt["optimization_semantics"]["metrics"][0]["sense"] == "minimize"
    assert "delta_decimal is child minus parent" in prompt["task"]
    assert "authoritative text for numerical magnitude" in prompt["task"]
    assert "must agree on affected path" in prompt["task"]
    assert "not an established mechanistic or causal conclusion" in prompt["task"]


def test_exact_action_contract_is_derived_only_from_authenticated_evidence() -> None:
    snapshot = _snapshot()
    exact_contract = bind_reflection_contract_to_evidence_actions(
        _contract(),
        snapshot,
    )
    assert exact_contract.allowed_option_ids == ("option.raise_x",)

    request = build_identifiable_reflection_generation_request(
        call_id=LLMCallId("call_exact_action_reflection"),
        evidence=snapshot,
        insight_contract=exact_contract,
        optimization_semantics=_semantics(),
        max_output_tokens=384_000,
        temperature=None,
    )
    prompt = json.loads(request.prompt)
    assert prompt["action_vocabulary"]["allowed_option_ids"] == [
        "option.raise_x"
    ]


def test_exact_action_binding_rejects_foreign_adapter_whitelist() -> None:
    contract = replace(
        _contract(),
        allowed_option_ids=("option.lower_x",),
    )
    with pytest.raises(ValueError, match="escape the adapter vocabulary"):
        bind_reflection_contract_to_evidence_actions(contract, _snapshot())


def test_construction_record_replays_snapshot_contract_and_mapping() -> None:
    snapshot = _snapshot()
    request = build_identifiable_reflection_generation_request(
        call_id=LLMCallId("call_identifiable_reflection"),
        evidence=snapshot,
        insight_contract=_contract(),
        optimization_semantics=_semantics(),
        max_output_tokens=384_000,
        temperature=None,
    )
    record = identifiable_reflection_request_construction_record(request, snapshot)
    assert record["schema_version"] == 2
    assert record["request_builder"] == {
        "builder_id": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
        "builder_version": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION,
        "definition_sha256": (
            IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
        ),
    }
    assert record["evidence_snapshot_sha256"] == snapshot.snapshot_sha256
    assert record["full_contrast_ids_exposed_to_model"] is False
    assert record["evidence_citation_mapping"] == [
        {
            "citation_key": "e0001",
            "contrast_id": snapshot.contrasts[0].contrast_id,
        }
    ]
    assert len(str(record["request_identity_sha256"])) == 64


def test_request_rejects_mechanistic_contract_and_typed_abstention_gap() -> None:
    mechanistic = replace(
        _contract(),
        allowed_insight_kinds=(
            ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            ReflectionInsightKind.MECHANISTIC_CONJECTURE,
        ),
    )
    with pytest.raises(ValueError, match="empirical rules only"):
        build_identifiable_reflection_generation_request(
            call_id=LLMCallId("call_identifiable_reflection"),
            evidence=_snapshot(),
            insight_contract=mechanistic,
            optimization_semantics=_semantics(),
            max_output_tokens=384_000,
            temperature=None,
        )
    with pytest.raises(ValueError, match="typed abstention"):
        build_identifiable_reflection_generation_request(
            call_id=LLMCallId("call_identifiable_reflection"),
            evidence=_snapshot(),
            insight_contract=_contract(),
            optimization_semantics=_semantics(),
            max_output_tokens=384_000,
            temperature=None,
            min_insights=0,
        )


def test_request_rejects_foreign_action_vocabulary() -> None:
    contract = replace(_contract(), allowed_option_families=("other",))
    with pytest.raises(ValueError, match="family escaped"):
        build_identifiable_reflection_generation_request(
            call_id=LLMCallId("call_identifiable_reflection"),
            evidence=_snapshot(),
            insight_contract=contract,
            optimization_semantics=_semantics(),
            max_output_tokens=384_000,
            temperature=None,
        )

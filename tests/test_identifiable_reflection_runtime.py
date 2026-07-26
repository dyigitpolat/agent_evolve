"""Fail-closed runtime binding for committed identifiable reflection evidence."""

from __future__ import annotations

import hashlib

import pytest

import agent_evolve
from agent_evolve.application.campaign_evidence_registry import (
    CampaignEvidenceRegistry,
)
from agent_evolve.application.evolution_campaign import (
    CampaignReflectionWave,
    ReflectionLaunchMode,
    ReflectionVisibility,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignIdentifiableReflectionEvidenceQuery,
    CampaignIdentifiableReflectionInput,
    CommittedRegistryIdentifiableReflectionEvidenceSource,
)
from agent_evolve.application.identifiable_reflection_evidence import (
    ReflectionFalsificationFeedback,
)
from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceProvenance,
    InterventionIdentifiability,
    ObservedMetricEffect,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _observation(name: str, event_index: int) -> AuthenticatedHypothesisObservation:
    parent = freeze_json({"x": event_index})
    child = freeze_json({"x": event_index + 1})
    return AuthenticatedHypothesisObservation(
        source_evidence_id=_sha(f"source:{name}"),
        event_index=event_index,
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        campaign_sha256=_sha("campaign"),
        parent_candidate_id=CandidateId(f"candidate_parent_{name}"),
        child_candidate_id=CandidateId(f"candidate_child_{name}"),
        operator_invocation_id=OperatorInvocationId(f"operator_{name}"),
        finite_contract_identity_sha256=_sha("finite-contract"),
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
                "option_id": f"coordinate.{name}",
                "option_identity_sha256": _sha(f"option:{name}"),
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
                delta=-1.0,
                adjudicator_definition_sha256=_sha("adjudicator"),
            ),
        ),
        lineage_cluster_sha256=_sha(f"lineage:{name}"),
        factorial_block_sha256=_sha(f"block:{name}"),
    )


def _query(
    *,
    source_generation: int,
    source_portfolio_generation: int,
    prior_cutoff: int,
    campaign_sha256: str | None = None,
) -> CampaignIdentifiableReflectionEvidenceQuery:
    return CampaignIdentifiableReflectionEvidenceQuery(
        reflection_request_sha256=_sha(
            f"reflection-request:{source_generation}"
        ),
        preparation_sha256=_sha("preparation"),
        runtime_start_receipt_sha256=_sha("runtime-start"),
        campaign_sha256=(
            _sha("campaign") if campaign_sha256 is None else campaign_sha256
        ),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        wave=CampaignReflectionWave(
            source_generation=source_generation,
            call_count=1,
            launch_mode=ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
            visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            promotion_barrier_generation=source_generation,
        ),
        source_stage_receipt_sha256=_sha(
            f"source-stage:{source_generation}"
        ),
        source_portfolio_generation=source_portfolio_generation,
        prior_cutoff_event_index_exclusive=prior_cutoff,
        sealed_cutoff_event_index_inclusive=source_portfolio_generation,
    )


def _feedback(available_event_index: int) -> ReflectionFalsificationFeedback:
    return ReflectionFalsificationFeedback(
        insight_content_sha256=_sha("deprecated-insight"),
        applicable_workload_instance_sha256s=(_sha("workload"),),
        evaluator_contract_sha256=_sha("evaluator"),
        applicable_campaign_sha256s=(_sha("campaign"),),
        audit_scope_sha256=_sha("audit-scope"),
        available_event_index=available_event_index,
        affected_paths=("$.x",),
        predictions=(("loss", MetricEffectDirection.DECREASE),),
        counterexample_source_evidence_ids=(_sha("counterexample"),),
        semantic_audit_receipt_sha256=_sha("semantic-audit"),
        lifecycle_decision_receipt_sha256=_sha("lifecycle-decision"),
        deprecation_reason="contradicted_by_later_direct_mutation",
    )


class _MutableFalsificationSource:
    def __init__(self) -> None:
        self.feedback: tuple[ReflectionFalsificationFeedback, ...] = ()
        self.cutoffs: list[int] = []

    def available(self, query):
        self.cutoffs.append(query.sealed_cutoff_event_index_inclusive)
        return self.feedback


def test_committed_source_replays_historical_cutoff_after_registry_advances() -> None:
    registry = CampaignEvidenceRegistry()
    first = _observation("first", 1)
    registry.commit_append(registry.prepare_append((first,)))
    source = CommittedRegistryIdentifiableReflectionEvidenceSource(
        registry=registry,
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
    )
    first_query = _query(
        source_generation=2,
        source_portfolio_generation=1,
        prior_cutoff=0,
    )
    before = source.project(first_query)
    first_input = CampaignIdentifiableReflectionInput(first_query, before)

    third = _observation("third", 3)
    registry.commit_append(registry.prepare_append((third,)))
    after = source.project(first_query)

    assert after == before
    assert after.registry_snapshot_sha256 == before.registry_snapshot_sha256
    assert first_input.evidence.contrasts[0].parent_candidate_id == (
        first.parent_candidate_id
    )
    assert first_input.evidence.contrasts[0].child_candidate_id == (
        first.child_candidate_id
    )
    assert first_input.evidence.contrasts[0].operator_invocation_id == (
        first.operator_invocation_id
    )
    assert first_input.evidence.contrasts[0].action_semantics_definition_sha256 == (
        first.action_semantics_definition_sha256
    )
    record = first_input.to_record()
    assert record["source_stage_payload_exposed"] is False
    assert record["recombination_results_exposed"] is False

    next_query = _query(
        source_generation=4,
        source_portfolio_generation=3,
        prior_cutoff=1,
    )
    next_input = CampaignIdentifiableReflectionInput(
        next_query,
        source.project(next_query),
    )
    assert tuple(
        value.source_evidence_id for value in next_input.evidence.contrasts
    ) == (third.source_evidence_id,)
    assert {
        value.source_evidence_id for value in first_input.evidence.contrasts
    }.isdisjoint(
        value.source_evidence_id for value in next_input.evidence.contrasts
    )


def test_cutoff_aware_falsification_source_refreshes_between_reflections() -> None:
    registry = CampaignEvidenceRegistry()
    registry.commit_append(registry.prepare_append((_observation("first", 1),)))
    dynamic = _MutableFalsificationSource()
    source = CommittedRegistryIdentifiableReflectionEvidenceSource(
        registry=registry,
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        falsification_source=dynamic,
    )
    first = source.project(
        _query(
            source_generation=2,
            source_portfolio_generation=1,
            prior_cutoff=0,
        )
    )
    assert first.evidence.prior_falsifications == ()

    registry.commit_append(registry.prepare_append((_observation("third", 3),)))
    feedback = _feedback(2)
    dynamic.feedback = (feedback,)
    second = source.project(
        _query(
            source_generation=4,
            source_portfolio_generation=3,
            prior_cutoff=1,
        )
    )
    assert second.evidence.prior_falsifications == (feedback,)
    assert dynamic.cutoffs == [1, 3]

    dynamic.feedback = (_feedback(4),)
    with pytest.raises(ValueError, match="foreign or future feedback"):
        source.project(
            _query(
                source_generation=4,
                source_portfolio_generation=3,
                prior_cutoff=1,
            )
        )


def test_source_rejects_foreign_scope_and_uncommitted_cutoff() -> None:
    registry = CampaignEvidenceRegistry()
    registry.commit_append(registry.prepare_append((_observation("first", 1),)))
    source = CommittedRegistryIdentifiableReflectionEvidenceSource(
        registry=registry,
        campaign_sha256=_sha("campaign"),
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
    )

    with pytest.raises(ValueError, match="foreign evidence scope"):
        source.project(
            _query(
                source_generation=2,
                source_portfolio_generation=1,
                prior_cutoff=0,
                campaign_sha256=_sha("foreign-campaign"),
            )
        )
    with pytest.raises(RuntimeError, match="has not reached the cutoff"):
        source.project(
            _query(
                source_generation=4,
                source_portfolio_generation=3,
                prior_cutoff=1,
            )
        )


def test_identifiable_runtime_symbols_are_public() -> None:
    for name in (
        "CAMPAIGN_IDENTIFIABLE_REFLECTION_BINDING_KEY",
        "CampaignIdentifiableReflectionEvidenceProjection",
        "CampaignIdentifiableReflectionEvidenceQuery",
        "CampaignIdentifiableReflectionEvidenceSource",
        "CampaignIdentifiableReflectionInput",
        "CampaignLegacyRecombinationReflectionExecutor",
        "CampaignReflectionFalsificationSource",
        "CampaignReflectionExecutor",
        "CommittedRegistryIdentifiableReflectionEvidenceSource",
    ):
        assert name in agent_evolve.__all__
        assert getattr(agent_evolve, name) is not None

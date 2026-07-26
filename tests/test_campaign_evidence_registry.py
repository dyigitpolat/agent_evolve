from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.campaign_evidence_registry import (
    CampaignEvidenceRegistry,
)
from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
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


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _observation(label: str, event_index: int) -> AuthenticatedHypothesisObservation:
    parent = _object({"x": event_index})
    child = _object({"x": event_index + 1})
    return AuthenticatedHypothesisObservation(
        source_evidence_id=_sha(f"source:{label}"),
        event_index=event_index,
        workload_instance_sha256=_sha("workload"),
        evaluator_contract_sha256=_sha("evaluator"),
        campaign_sha256=_sha("campaign"),
        parent_candidate_id=CandidateId(f"candidate_parent_{label}"),
        child_candidate_id=CandidateId(f"candidate_child_{label}"),
        operator_invocation_id=OperatorInvocationId(f"operator_{label}"),
        finite_contract_identity_sha256=_sha("finite-contract"),
        provenance=EvidenceProvenance.DIRECT_MUTATION,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha(f"wave:{label}"),
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
        parent_outcome_sha256=_sha(f"parent-outcome:{label}"),
        child_outcome_sha256=_sha(f"child-outcome:{label}"),
        operator_family="typed_mutation",
        affected_paths=("$.x",),
        observed_action=_object({"path": "$.x", "delta": 1}),
        action_semantics_compiler_id="test_action_semantics",
        action_semantics_compiler_version=1,
        action_semantics_definition_sha256=_sha("action-semantics-definition"),
        intervention_identifiability=InterventionIdentifiability.EXACT_SINGLE,
        metrics=(
            ObservedMetricEffect(
                metric_id="quality",
                direction=MetricEffectDirection.INCREASE,
                delta=1.0,
                adjudicator_definition_sha256=_sha("adjudicator"),
            ),
        ),
        lineage_cluster_sha256=_sha(f"lineage:{label}"),
        factorial_block_sha256=_sha(f"block:{label}"),
    )


def test_prepare_is_pure_and_commit_publishes_exact_snapshot() -> None:
    registry = CampaignEvidenceRegistry()
    first = _observation("first", 1)
    preparation = registry.prepare_append((first,))

    assert registry.observations == ()
    assert preparation.prior_observation_count == 0
    assert preparation.prospective_snapshot.observations == (first,)

    committed = registry.commit_append(preparation)
    assert registry.observations == (first,)
    assert committed.snapshot_sha256 == preparation.prospective_snapshot.snapshot_sha256


def test_abort_leaves_registry_unchanged_and_retry_is_possible() -> None:
    registry = CampaignEvidenceRegistry()
    observation = _observation("retry", 2)
    preparation = registry.prepare_append((observation,))
    registry.abort_append(preparation)

    assert registry.observations == ()
    retried = registry.prepare_append((observation,))
    registry.commit_append(retried)
    assert registry.observations == (observation,)


def test_duplicate_or_nonmonotone_evidence_fails_closed() -> None:
    registry = CampaignEvidenceRegistry()
    first = _observation("first", 3)
    registry.commit_append(registry.prepare_append((first,)))

    with pytest.raises(ValueError, match="already committed"):
        registry.prepare_append((first,))
    with pytest.raises(ValueError, match="follow the committed event cutoff"):
        registry.prepare_append((_observation("late-id-old-event", 3),))


def test_stale_parallel_preparation_cannot_commit_over_new_state() -> None:
    registry = CampaignEvidenceRegistry()
    first = registry.prepare_append((_observation("first", 1),))
    second = registry.prepare_append((_observation("second", 2),))
    registry.commit_append(first)

    with pytest.raises(RuntimeError, match="changed after preparation"):
        registry.commit_append(second)


def test_empty_no_yield_append_advances_cutoff_without_fabricating_evidence() -> None:
    registry = CampaignEvidenceRegistry()
    preparation = registry.prepare_append(
        (),
        captured_through_event_index=1,
    )

    assert preparation.observations == ()
    assert preparation.prior_observation_count == 0
    assert preparation.prospective_snapshot.observations == ()
    assert preparation.prospective_snapshot.captured_through_event_index == 1
    committed = registry.commit_append(preparation)
    assert committed.observations == registry.observations == ()
    assert committed.captured_through_event_index == 1

    with pytest.raises(ValueError, match="requires an explicit event cutoff"):
        CampaignEvidenceRegistry().prepare_append(())

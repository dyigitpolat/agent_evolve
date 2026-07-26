"""Application-level contracts for phase-aware insight retrieval."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightEvidenceLineage,
    InsightLifecycleChangeRequest,
    InsightLifecycleState,
    InsightMemoryBank,
    InsightOrigin,
    InsightRelation,
    InsightRelationKind,
    compose_epistemic_prompt_payload,
)
from agent_evolve.domain.ids import InsightId, LLMCallId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.randomized_subset import InsightSelectionMode
from agent_evolve.ports.agentic_generator import InsightDraft


DISCOVERY_CONTEXT = "d" * 64
EXPLOITATION_CONTEXT = "e" * 64
REWARD_DEFINITION_HASH = "f" * 64


class _SampleOnlyRandom:
    def __init__(self, selected) -> None:
        self._selected = selected

    def randrange(self, stop: int) -> int:  # pragma: no cover - epsilon is exact 0/1.
        raise AssertionError(f"unexpected random branch draw with stop={stop}")

    def sample(self, population, k: int):
        assert self._selected in population
        assert k == 1
        return [self._selected]


class _NoRandom:
    def randrange(self, stop: int) -> int:  # pragma: no cover - exploitation only.
        raise AssertionError(f"unexpected random branch draw with stop={stop}")

    def sample(self, population, k: int):  # pragma: no cover - exploitation only.
        raise AssertionError("unexpected uniform sample")


def _insight(claim: str, confidence: float) -> InsightDraft:
    return InsightDraft(
        claim=claim,
        trigger="the corresponding design choice is available",
        mechanism="the choice changes the evaluated trade-off",
        affected_paths=("$.runtime",),
        evidence_summary="controlled development evidence",
        confidence=confidence,
    )


def test_selection_can_exploit_discovery_scores_in_a_distinct_credit_phase() -> None:
    ids = DeterministicIdFactory("memory_phase")
    memory = InsightMemoryBank(id_factory=ids)
    initially_high, _ = memory.add(_insight("Initially preferred", 0.9))
    initially_low, _ = memory.add(_insight("Initially disfavored", 0.1))

    high_trial = memory.select(
        context_hash=DISCOVERY_CONTEXT,
        subset_size=1,
        rng=_SampleOnlyRandom(initially_high.reference),
        exploration_probability=Fraction(1, 1),
    )
    memory.record_trial(
        credit_unit_id=ids.new_operator_invocation_id(),
        candidate_ids=(ids.new_candidate_id(),),
        reward_definition_hash=REWARD_DEFINITION_HASH,
        decision=high_trial,
        reward=-10.0,
    )
    low_trial = memory.select(
        context_hash=DISCOVERY_CONTEXT,
        subset_size=1,
        rng=_SampleOnlyRandom(initially_low.reference),
        exploration_probability=Fraction(1, 1),
    )
    memory.record_trial(
        credit_unit_id=ids.new_operator_invocation_id(),
        candidate_ids=(ids.new_candidate_id(),),
        reward_definition_hash=REWARD_DEFINITION_HASH,
        decision=low_trial,
        reward=10.0,
    )

    discovery_scores = memory.score_snapshot(DISCOVERY_CONTEXT)
    assert discovery_scores[initially_low.reference] > discovery_scores[initially_high.reference]

    phase_aware = memory.select(
        context_hash=EXPLOITATION_CONTEXT,
        score_context_hash=DISCOVERY_CONTEXT,
        subset_size=1,
        rng=_NoRandom(),
        exploration_probability=Fraction(0, 1),
    )
    phase_local = memory.select(
        context_hash=EXPLOITATION_CONTEXT,
        subset_size=1,
        rng=_NoRandom(),
        exploration_probability=Fraction(0, 1),
    )

    assert phase_aware.context_hash == EXPLOITATION_CONTEXT
    assert phase_aware.mode is InsightSelectionMode.EXPLOIT
    assert phase_aware.exploration_probability == Fraction(0, 1)
    assert phase_aware.selected == (initially_low.reference,)
    assert dict(phase_aware.score_snapshot) == discovery_scores
    assert phase_local.selected == (initially_high.reference,)


def test_explicit_neutral_initial_score_overrides_confidence_prior() -> None:
    ids = DeterministicIdFactory("memory_neutral")
    memory = InsightMemoryBank(id_factory=ids)
    neutral, added = memory.add(
        _insight("New reflected hypothesis", 0.99),
        initial_score=0.0,
    )

    assert added is True
    assert neutral.initial_score == 0.0
    assert memory.score_snapshot(DISCOVERY_CONTEXT)[neutral.reference] == 0.0

    with pytest.raises(ValueError, match="initial_score must be finite"):
        memory.add(_insight("Invalid prior", 0.5), initial_score=float("nan"))


def test_memory_entry_record_is_complete_deterministic_and_detached() -> None:
    ids = DeterministicIdFactory("memory_entry_record")
    memory = InsightMemoryBank(id_factory=ids)
    seed, _ = memory.add(_insight("Seed comparison card", 0.5))
    contrast_id = "a" * 64
    lineage = InsightEvidenceLineage(
        reflection_call_id=ids.new_llm_call_id(),
        source_operator_invocation_ids=(ids.new_operator_invocation_id(),),
        source_candidate_ids=(ids.new_candidate_id(),),
        available_contrast_ids=(contrast_id,),
        cited_contrast_ids=(contrast_id,),
    )
    reflected, _ = memory.add(
        _insight("Reflected durable card", 0.75),
        initial_score=-0.25,
        applicable_operator_kinds=("mutation", "recombination"),
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=lineage,
        relations=(
            InsightRelation(
                kind=InsightRelationKind.CONTRADICTS,
                target=seed.reference,
                note="The observed operating regimes differ.",
            ),
        ),
    )

    record = reflected.to_record()
    assert record == {
        "schema_version": 1,
        "reference": {
            "insight_id": reflected.reference.insight_id.value,
            "version": 1,
        },
        "draft": reflected.draft.content_record(),
        "draft_content_sha256": reflected.draft.content_sha256,
        "draft_hypothesis_sha256": reflected.draft.hypothesis_sha256,
        "initial_score_hex": (-0.25).hex(),
        "applicable_operator_kinds": ["mutation", "recombination"],
        "lifecycle_state": "quarantined",
        "retrievable": False,
        "origin": "reflection",
        "evidence_lineage": lineage.to_record(),
        "relations": [
            {
                "kind": "contradicts",
                "target": {
                    "insight_id": seed.reference.insight_id.value,
                    "version": 1,
                },
                "note": "The observed operating regimes differ.",
            }
        ],
    }
    encoded = json.dumps(record, allow_nan=False, sort_keys=True)
    assert encoded == json.dumps(
        reflected.to_record(),
        allow_nan=False,
        sort_keys=True,
    )

    record["draft"]["claim"] = "mutated detached projection"  # type: ignore[index]
    record["relations"].clear()  # type: ignore[union-attr]
    assert reflected.to_record()["draft"] == reflected.draft.content_record()
    assert reflected.to_record()["relations"] != []


def _empirical_snapshot(
    contrast_id: str = "a" * 64,
) -> EmpiricalEvidenceSnapshot:
    facts = freeze_json(
        {
            "valid": True,
            "observed_metric_deltas": {
                "objective:cost": -0.25,
                "violation:constraint": 0.5,
            },
            "evaluation_receipt_sha256": "b" * 64,
        }
    )
    assert type(facts) is FrozenJsonObject
    return EmpiricalEvidenceSnapshot(
        contrast_id=contrast_id,
        fact_schema_id="metric_delta_receipt",
        fact_schema_version=1,
        fact_schema_definition_sha256="c" * 64,
        facts=facts,
        optimization_semantics_definition_sha256="d" * 64,
        action_semantics_definition_sha256="e" * 64,
    )


def test_empirical_snapshot_identity_binds_facts_schema_and_semantics() -> None:
    snapshot = _empirical_snapshot()
    same = _empirical_snapshot()
    assert snapshot.snapshot_sha256 == same.snapshot_sha256
    assert snapshot.to_record()["snapshot_sha256"] == snapshot.snapshot_sha256

    changed_facts = freeze_json(
        {
            "valid": True,
            "observed_metric_deltas": {"objective:cost": -0.20},
            "evaluation_receipt_sha256": "b" * 64,
        }
    )
    assert type(changed_facts) is FrozenJsonObject
    assert (
        replace(snapshot, facts=changed_facts).snapshot_sha256
        != snapshot.snapshot_sha256
    )
    assert (
        replace(snapshot, action_semantics_definition_sha256="f" * 64).snapshot_sha256
        != snapshot.snapshot_sha256
    )


def test_v2_lineage_requires_exact_empirical_coverage_and_preserves_legacy_identity() -> None:
    contrast_id = "a" * 64
    legacy = InsightEvidenceLineage(
        reflection_call_id=LLMCallId("call_memory_legacy"),
        source_operator_invocation_ids=(),
        source_candidate_ids=(),
        available_contrast_ids=(contrast_id,),
        cited_contrast_ids=(contrast_id,),
    )
    explicit_empty = replace(legacy, empirical_evidence=())
    assert legacy.identity_sha256 == explicit_empty.identity_sha256
    assert legacy.to_record()["schema_version"] == 1
    assert "empirical_evidence" not in legacy.to_record()

    snapshot = _empirical_snapshot(contrast_id)
    v2 = replace(legacy, empirical_evidence=(snapshot,))
    assert v2.to_record()["schema_version"] == 2
    assert v2.identity_sha256 != legacy.identity_sha256
    assert v2.to_record()["empirical_evidence"] == [snapshot.to_record()]

    with pytest.raises(ValueError, match="exactly cover every cited contrast"):
        replace(legacy, empirical_evidence=(_empirical_snapshot("9" * 64),))


def test_epistemic_prompt_structurally_separates_facts_from_hypothesis() -> None:
    draft = _insight("A testable but unverified mechanism", 0.7)
    legacy_content_sha256 = draft.content_sha256
    hypothesis = freeze_json(draft.hypothesis_record())
    assert type(hypothesis) is FrozenJsonObject
    payload = compose_epistemic_prompt_payload(
        empirical_evidence=(_empirical_snapshot(),),
        hypothesis=hypothesis,
    )
    record = thaw_json(payload)

    assert record["empirical_facts"][0]["facts"]["valid"] is True
    assert record["hypothesis"]["epistemic_status"] == "unverified_hypothesis"
    assert record["hypothesis"]["mechanism_hypothesis"] == draft.mechanism
    assert record["hypothesis"]["evidence_interpretation"] == draft.evidence_summary
    assert record["interpretation_policy"] == {
        "empirical_facts_are_observations": True,
        "hypothesis_is_observation": False,
        "mechanism_requires_independent_validation": True,
    }
    assert draft.content_sha256 == legacy_content_sha256
    assert draft.hypothesis_sha256 != draft.content_sha256

    mislabeled = freeze_json({"epistemic_status": "observation", "claim": "bad"})
    assert type(mislabeled) is FrozenJsonObject
    with pytest.raises(ValueError, match="unverified_hypothesis"):
        compose_epistemic_prompt_payload(
            empirical_evidence=(_empirical_snapshot(),),
            hypothesis=mislabeled,
        )


def test_lifecycle_batch_is_atomic_and_canonical() -> None:
    ids = DeterministicIdFactory("memory_lifecycle_batch")
    memory = InsightMemoryBank(id_factory=ids)
    first, _ = memory.add(
        _insight("First quarantined hypothesis", 0.5),
        origin=InsightOrigin.MANUAL,
    )
    second, _ = memory.add(
        _insight("Second quarantined hypothesis", 0.5),
        origin=InsightOrigin.MANUAL,
    )
    before = memory.entries
    with pytest.raises(ValueError, match="foreign"):
        memory.apply_lifecycle_batch(
            (
                InsightLifecycleChangeRequest(
                    reference=first.reference,
                    new_state=InsightLifecycleState.PROMOTED,
                    reason="validated in a later diagnostic wave",
                    supporting_evidence=("a" * 64,),
                ),
                InsightLifecycleChangeRequest(
                    reference=InsightRef(InsightId("insight_foreign"), 1),
                    new_state=InsightLifecycleState.DEPRECATED,
                    reason="synthetic foreign request",
                ),
            )
        )
    assert memory.entries == before
    assert memory.transitions == ()

    updated = memory.apply_lifecycle_batch(
        (
            InsightLifecycleChangeRequest(
                reference=second.reference,
                new_state=InsightLifecycleState.DEPRECATED,
                reason="global replay found a counterexample",
                supporting_evidence=("c" * 64,),
            ),
            InsightLifecycleChangeRequest(
                reference=first.reference,
                new_state=InsightLifecycleState.PROMOTED,
                reason="validated in a later diagnostic wave",
                supporting_evidence=("b" * 64,),
            ),
        )
    )
    assert tuple(value.reference for value in updated) == tuple(
        sorted((first.reference, second.reference))
    )
    assert tuple(value.reference for value in memory.transitions) == tuple(
        sorted((first.reference, second.reference))
    )
    assert memory.entries_for((first.reference,))[0].lifecycle_state is (
        InsightLifecycleState.PROMOTED
    )
    assert memory.entries_for((second.reference,))[0].lifecycle_state is (
        InsightLifecycleState.DEPRECATED
    )

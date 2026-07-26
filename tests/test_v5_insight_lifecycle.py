"""Focused contracts for the v5 quarantine/promote/deprecate lifecycle."""

from __future__ import annotations

from fractions import Fraction

import pytest

from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryBank,
    InsightOrigin,
    InsightRelation,
    InsightRelationKind,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import InsightDraft


CONTEXT = "c" * 64


class _NoRandom:
    def randrange(self, stop: int) -> int:  # pragma: no cover - exploitation only.
        raise AssertionError(f"unexpected random draw with stop={stop}")

    def sample(self, population, k: int):  # pragma: no cover - exploitation only.
        raise AssertionError("unexpected random sample")


def _draft(claim: str) -> InsightDraft:
    return InsightDraft(
        claim=claim,
        trigger="the corresponding coordinate is editable",
        mechanism="the edit changes the evaluated objective trade-off",
        affected_paths=("$.runtime.threads",),
        evidence_summary="controlled held-out evidence",
        confidence=0.7,
    )


def _lineage(ids: DeterministicIdFactory) -> InsightEvidenceLineage:
    contrast = "a" * 64
    return InsightEvidenceLineage(
        reflection_call_id=ids.new_llm_call_id(),
        source_operator_invocation_ids=(ids.new_operator_invocation_id(),),
        source_candidate_ids=(ids.new_candidate_id(),),
        available_contrast_ids=(contrast,),
        cited_contrast_ids=(contrast,),
    )


def test_reflection_is_quarantined_until_recorded_promotion_then_deprecation() -> None:
    ids = DeterministicIdFactory("v5_lifecycle")
    memory = InsightMemoryBank(id_factory=ids)
    seed, _ = memory.add(_draft("Seed prior remains eligible."))
    reflected, _ = memory.add(
        _draft("Reflected hypothesis starts quarantined."),
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=_lineage(ids),
    )

    assert seed.lifecycle_state is InsightLifecycleState.SEED
    assert seed.retrievable is True
    assert reflected.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert reflected.retrievable is False
    assert memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.threads",),
    ) == (seed.reference,)
    assert tuple(memory.score_snapshot(CONTEXT)) == (seed.reference,)
    with pytest.raises(ValueError, match="lifecycle-ineligible"):
        memory.select(
            context_hash=CONTEXT,
            subset_size=1,
            rng=_NoRandom(),
            exploration_probability=Fraction(0),
            eligible_references=(reflected.reference,),
        )

    promoted = memory.promote(
        reflected.reference,
        reason="Held-out randomized block improved regret.",
        supporting_evidence=("experiment:heldout-001",),
    )
    promotion_history = memory.transitions

    assert reflected.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert promoted.lifecycle_state is InsightLifecycleState.PROMOTED
    assert promoted.retrievable is True
    assert memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.threads",),
    ) == tuple(sorted((seed.reference, reflected.reference)))
    assert promotion_history[0].prior_state is InsightLifecycleState.QUARANTINED
    assert promotion_history[0].new_state is InsightLifecycleState.PROMOTED
    assert promotion_history[0].supporting_evidence == ("experiment:heldout-001",)

    deprecated = memory.deprecate(
        reflected.reference,
        reason="Later counterevidence invalidated the trigger.",
        supporting_evidence=("experiment:counterexample-002",),
    )

    assert deprecated.lifecycle_state is InsightLifecycleState.DEPRECATED
    assert deprecated.retrievable is False
    assert len(promotion_history) == 1  # tuple snapshot remains immutable.
    assert [transition.sequence for transition in memory.transitions] == [1, 2]
    assert memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.threads",),
    ) == (seed.reference,)


def test_lifecycle_rejects_bypass_and_invalid_transitions() -> None:
    ids = DeterministicIdFactory("v5_lifecycle_invalid")
    memory = InsightMemoryBank(id_factory=ids)
    seed, _ = memory.add(_draft("Seed cannot be promoted."))
    reflected, _ = memory.add(
        _draft("Quarantined evidence needs a promotion record."),
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=_lineage(ids),
    )

    with pytest.raises(ValueError, match="only a quarantined"):
        memory.promote(
            seed.reference,
            reason="Seed is already eligible.",
            supporting_evidence=("experiment:irrelevant",),
        )
    with pytest.raises(ValueError, match="at least one"):
        memory.promote(
            reflected.reference,
            reason="Unsubstantiated promotion.",
            supporting_evidence=(),
        )
    with pytest.raises(ValueError, match="must start in quarantined"):
        memory.add(
            _draft("Reflection cannot bypass quarantine."),
            origin=InsightOrigin.REFLECTION,
            lifecycle_state=InsightLifecycleState.PROMOTED,
            evidence_lineage=_lineage(ids),
        )

    memory.deprecate(reflected.reference, reason="Malformed evidence hypothesis.")
    with pytest.raises(ValueError, match="cannot transition again"):
        memory.deprecate(reflected.reference, reason="Duplicate deprecation.")
    with pytest.raises(ValueError, match="only a quarantined"):
        memory.promote(
            reflected.reference,
            reason="Deprecated entries cannot be resurrected.",
            supporting_evidence=("experiment:late",),
        )


def test_declared_semantic_relations_and_versions_are_not_inferred() -> None:
    ids = DeterministicIdFactory("v5_semantic_lineage")
    memory = InsightMemoryBank(id_factory=ids)
    seed, _ = memory.add(_draft("More threads help this workload."))
    contrary, added = memory.add(
        _draft("More threads hurt memory-bound workloads."),
        origin=InsightOrigin.MANUAL,
        relations=(
            InsightRelation(
                InsightRelationKind.CONTRADICTS,
                seed.reference,
                "Triggers describe disjoint workload regimes.",
            ),
        ),
    )

    assert added is True
    assert contrary.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert contrary.relations[0].kind is InsightRelationKind.CONTRADICTS
    assert len(memory.entries) == 2

    revision = memory.add_revision(
        contrary.reference,
        _draft("More threads hurt when memory bandwidth is saturated."),
        relations=(
            InsightRelation(
                InsightRelationKind.DUPLICATES,
                seed.reference,
                "Overlaps only outside the new saturation trigger.",
            ),
        ),
    )

    assert revision.reference.insight_id == contrary.reference.insight_id
    assert revision.reference.version == contrary.reference.version + 1
    assert revision.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert {(relation.kind, relation.target) for relation in revision.relations} == {
        (InsightRelationKind.REVISES, contrary.reference),
        (InsightRelationKind.DUPLICATES, seed.reference),
    }
    assert memory.entries[1].lifecycle_state is InsightLifecycleState.QUARANTINED
    with pytest.raises(ValueError, match="latest version"):
        memory.add_revision(contrary.reference, _draft("Stale branch revision."))


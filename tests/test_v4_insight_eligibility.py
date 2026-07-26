"""Focused contracts for v4 structural insight eligibility."""

from __future__ import annotations

from fractions import Fraction

import pytest

from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.domain.insight import InsightRef
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    ReflectionConsumerScope,
    ReflectionInsightKind,
)


CONTEXT = "a" * 64


class _NoRandom:
    def randrange(self, stop: int) -> int:  # pragma: no cover - exploitation only.
        raise AssertionError(f"unexpected random draw with stop={stop}")

    def sample(self, population, k: int):  # pragma: no cover - exploitation only.
        raise AssertionError("unexpected random sample")


def _insight(
    claim: str,
    *,
    affected_paths: tuple[str, ...],
    confidence: float = 0.9,
) -> InsightDraft:
    return InsightDraft(
        claim=claim,
        trigger="a natural-language condition that eligibility does not interpret",
        mechanism="the affected component changes an evaluated trade-off",
        affected_paths=affected_paths,
        evidence_summary="development evidence",
        confidence=confidence,
    )


def _semantic_heuristic(
    claim: str,
    *,
    scope: ReflectionConsumerScope,
    capabilities: tuple[str, ...],
) -> InsightDraft:
    return InsightDraft(
        claim=claim,
        trigger="the declared factor capabilities are available",
        mechanism="the heuristic directs a bounded evolutionary decision",
        affected_paths=("$.runtime",),
        evidence_summary="search-outcome evidence awaiting a separate schema",
        confidence=0.5,
        insight_kind=ReflectionInsightKind.SEARCH_HEURISTIC,
        consumer_scopes=(scope,),
        factor_capabilities=capabilities,
    )


def test_confidence_at_creation_is_not_an_implicit_utility_prior() -> None:
    memory = InsightMemoryBank(id_factory=DeterministicIdFactory("v4_neutral"))
    high, _ = memory.add(
        _insight("High-confidence claim", affected_paths=("$.runtime",)),
    )
    low, _ = memory.add(
        _insight(
            "Low-confidence claim",
            affected_paths=("$.runtime",),
            confidence=0.01,
        ),
    )
    explicit, _ = memory.add(
        _insight("Explicit prior", affected_paths=("$.runtime",)),
        initial_score=0.25,
    )

    scores = memory.score_snapshot(CONTEXT)
    assert high.initial_score == 0.0
    assert low.initial_score == 0.0
    assert scores[high.reference] == scores[low.reference] == 0.0
    assert scores[explicit.reference] == 0.25


def test_eligibility_filters_only_by_explicit_operator_tokens() -> None:
    memory = InsightMemoryBank(id_factory=DeterministicIdFactory("v4_operator"))
    mutation, _ = memory.add(
        _insight("Mutation only", affected_paths=("$.runtime",)),
        applicable_operator_kinds=("typed_mutation",),
    )
    repair, _ = memory.add(
        _insight("Repair only", affected_paths=("$.runtime",)),
        applicable_operator_kinds=("repair",),
    )
    universal, _ = memory.add(
        _insight("All operators", affected_paths=("$.runtime",)),
    )

    eligible = memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.threads",),
    )

    assert eligible == tuple(sorted((mutation.reference, universal.reference)))
    assert repair.reference not in eligible
    assert mutation.applicable_operator_kinds == ("typed_mutation",)
    assert universal.applicable_operator_kinds == ()


def test_path_eligibility_includes_exact_ancestor_and_descendant_overlap() -> None:
    memory = InsightMemoryBank(id_factory=DeterministicIdFactory("v4_paths"))
    ancestor, _ = memory.add(
        _insight("Runtime ancestor", affected_paths=("$.runtime",)),
    )
    exact, _ = memory.add(
        _insight(
            "Exact prefetch leaf",
            affected_paths=("$.runtime.prefetch_distance",),
        ),
    )
    descendant, _ = memory.add(
        _insight(
            "Array descendant",
            affected_paths=("$.runtime.buffers[0].capacity",),
        ),
    )
    unrelated, _ = memory.add(
        _insight("Compiler only", affected_paths=("$.compiler",)),
    )

    leaf_edit = memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.prefetch_distance",),
    )
    parent_edit = memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.buffers",),
    )

    assert leaf_edit == tuple(sorted((ancestor.reference, exact.reference)))
    assert parent_edit == tuple(sorted((ancestor.reference, descendant.reference)))
    assert unrelated.reference not in leaf_edit
    assert unrelated.reference not in parent_edit


def test_v3_eligibility_honors_consumer_scope_and_required_capabilities() -> None:
    memory = InsightMemoryBank(id_factory=DeterministicIdFactory("v3_semantics"))
    mutation, _ = memory.add(
        _semantic_heuristic(
            "Mutation heuristic",
            scope=ReflectionConsumerScope.MUTATION_SELECTION,
            capabilities=("discrete_sequence", "latency_model"),
        ),
        applicable_operator_kinds=("typed_mutation",),
    )
    recombination, _ = memory.add(
        _semantic_heuristic(
            "Recombination heuristic",
            scope=ReflectionConsumerScope.RECOMBINATION_SELECTION,
            capabilities=("discrete_sequence",),
        ),
        applicable_operator_kinds=("typed_mutation",),
    )

    assert memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime",),
    ) == tuple(sorted((mutation.reference, recombination.reference)))
    assert memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime",),
        consumer_scope=ReflectionConsumerScope.MUTATION_SELECTION,
        factor_capabilities=("discrete_sequence", "latency_model", "simulator"),
    ) == (mutation.reference,)
    assert memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime",),
        consumer_scope=ReflectionConsumerScope.MUTATION_SELECTION,
        factor_capabilities=("discrete_sequence",),
    ) == ()


def test_semantic_prompt_record_survives_without_intervention_contract() -> None:
    memory = InsightMemoryBank(
        id_factory=DeterministicIdFactory("v3_card_semantics")
    )
    entry, _ = memory.add(
        _semantic_heuristic(
            "Scoped search heuristic",
            scope=ReflectionConsumerScope.PARENT_SELECTION,
            capabilities=("archive_geometry",),
        )
    )

    record = memory.prompt_records((entry.reference,))[0]
    assert record["insight_kind"] == "search_heuristic"
    assert record["consumer_scopes"] == ["parent_selection"]
    assert record["factor_capabilities"] == ["archive_geometry"]
    assert "effect_predictions" not in record


def test_v3_eligibility_rejects_invalid_semantic_query_types() -> None:
    memory = InsightMemoryBank(id_factory=DeterministicIdFactory("v3_bad_query"))
    memory.add(
        _semantic_heuristic(
            "Typed query",
            scope=ReflectionConsumerScope.MUTATION_SELECTION,
            capabilities=("latency_model",),
        )
    )
    with pytest.raises(TypeError, match="exact ReflectionConsumerScope"):
        memory.eligible_references(
            operator_kind="typed_mutation",
            consumer_scope="mutation_selection",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="sequence of strings"):
        memory.eligible_references(
            operator_kind="typed_mutation",
            factor_capabilities="latency_model",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("method_name", ["score_snapshot", "select"])
@pytest.mark.parametrize("bad_subset_kind", ["duplicate", "foreign"])
def test_explicit_eligible_subsets_reject_duplicate_and_foreign_references(
    method_name: str,
    bad_subset_kind: str,
) -> None:
    memory = InsightMemoryBank(id_factory=DeterministicIdFactory("v4_subset"))
    entry, _ = memory.add(
        _insight("Owned insight", affected_paths=("$.runtime",)),
    )
    subset: tuple[InsightRef, ...]
    if bad_subset_kind == "duplicate":
        subset = (entry.reference, entry.reference)
    else:
        foreign_memory = InsightMemoryBank(
            id_factory=DeterministicIdFactory("v4_foreign")
        )
        foreign, _ = foreign_memory.add(
            _insight("Foreign insight", affected_paths=("$.runtime",)),
        )
        subset = (foreign.reference,)

    with pytest.raises(ValueError, match=bad_subset_kind):
        if method_name == "score_snapshot":
            memory.score_snapshot(CONTEXT, eligible_references=subset)
        else:
            memory.select(
                context_hash=CONTEXT,
                subset_size=1,
                rng=_NoRandom(),
                exploration_probability=Fraction(0),
                eligible_references=subset,
            )


def test_snapshot_and_selection_log_only_the_explicit_eligible_subset() -> None:
    memory = InsightMemoryBank(id_factory=DeterministicIdFactory("v4_exact_subset"))
    excluded, _ = memory.add(
        _insight("Excluded despite high score", affected_paths=("$.compiler",)),
        initial_score=100.0,
    )
    first, _ = memory.add(
        _insight("Eligible lower score", affected_paths=("$.runtime",)),
        initial_score=1.0,
    )
    second, _ = memory.add(
        _insight("Eligible higher score", affected_paths=("$.runtime",)),
        initial_score=2.0,
    )
    requested = (second.reference, first.reference)

    snapshot = memory.score_snapshot(CONTEXT, eligible_references=requested)
    decision = memory.select(
        context_hash=CONTEXT,
        subset_size=1,
        rng=_NoRandom(),
        exploration_probability=Fraction(0),
        eligible_references=requested,
    )

    canonical = tuple(sorted(requested))
    assert tuple(snapshot) == canonical
    assert decision.eligible == canonical
    assert tuple(reference for reference, _ in decision.score_snapshot) == canonical
    assert decision.selected == (second.reference,)
    assert excluded.reference not in snapshot

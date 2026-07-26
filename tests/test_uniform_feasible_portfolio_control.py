from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent_evolve.agentic import (
    FiniteVariationContract,
    FiniteVariationOption,
    InsightId,
    InsightRef,
    LLMCallId,
    PortfolioCard,
    PortfolioSelectionRequest,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from examples.development.uniform_feasible_portfolio_control import (
    MAX_REJECTION_DRAWS,
    TaskKeyedConditionalUniformPortfolioPolicy,
    analyze_grouped_feasible_slate_space,
)


def _sha(character: str) -> str:
    return character * 64


def _object(value: dict[str, object]):
    frozen = freeze_json(value)
    if type(frozen).__name__ != "FrozenJsonObject":
        raise AssertionError("fixture did not freeze as an object")
    return frozen


def _contract() -> FiniteVariationContract:
    parent = _object({"a": 0, "b": 0, "c": 0, "d": 0})
    parent_sha256 = typed_json_sha256(parent)
    rows = (
        ("alpha.a1", "alpha", {"a": 1, "b": 0, "c": 0, "d": 0}),
        ("alpha.a2", "alpha", {"a": 2, "b": 0, "c": 0, "d": 0}),
        ("beta.b1", "beta", {"a": 0, "b": 1, "c": 0, "d": 0}),
        ("gamma.c1", "gamma", {"a": 0, "b": 0, "c": 1, "d": 0}),
        ("delta.d1", "delta", {"a": 0, "b": 0, "c": 0, "d": 1}),
    )
    return FiniteVariationContract(
        catalog_id="uniform_control_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("3"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object(child),
                family=family,
                description=f"Fixture option {option_id}.",
            )
            for option_id, family, child in rows
        ),
    )


def _request(payload: str = "payload one") -> PortfolioSelectionRequest:
    card = PortfolioCard(
        card_key="card.control",
        reference=InsightRef(InsightId("insight_uniform_control"), 1),
        content_sha256=_sha("1"),
        evidence_sha256=_sha("2"),
        prompt_payload=_object({"ignored": payload}),
        assigned_score=0.0,
    )
    return PortfolioSelectionRequest(
        call_id=LLMCallId("call_uniform_control_0001"),
        operation="select_portfolio",
        instruction=f"Ignored instruction {payload}.",
        context=_object({"ignored": payload}),
        finite_variation_contract=_contract(),
        cards=(card,),
        portfolio_size=3,
        required_metric_ids=("first", "second"),
        min_distinct_families=3,
        require_supporting_cards=False,
        require_pairwise_disjoint_parent_patches=True,
        max_output_tokens=1,
    )


def test_exact_grouped_acceptance_count_and_outcome_blind_replay() -> None:
    request = _request()
    analysis = analyze_grouped_feasible_slate_space(request)
    assert analysis.feasible_unordered_slate_count == 7
    assert analysis.total_unordered_slate_count == 10
    assert analysis.acceptance_probability.numerator == 7
    assert analysis.acceptance_probability.denominator == 10
    assert analysis.rejection_cap == MAX_REJECTION_DRAWS

    policy = TaskKeyedConditionalUniformPortfolioPolicy(
        task_sha256=_sha("a"),
        replicate_seed=20_260_716,
    )
    first = asyncio.run(policy.select(request))
    changed_payload = asyncio.run(policy.select(_request("different hidden outcome")))
    assert tuple(member.option_id for member in first.decision.members) == tuple(
        member.option_id for member in changed_payload.decision.members
    )
    assert first.telemetry is not None
    assert first.telemetry.input_tokens == first.telemetry.output_tokens == 0
    assert first.telemetry.reasoning_tokens == 0
    assert first.telemetry.cost_usd == 0

    prose_changed_contract = replace(
        request.finite_variation_contract,
        options=tuple(
            replace(
                option,
                description=f"Different ignored prose for {option.option_id}.",
                metadata=(("ignored_note", "also not admitted to entropy"),),
            )
            for option in request.finite_variation_contract.options
        ),
    )
    assert (
        prose_changed_contract.identity_sha256
        != request.finite_variation_contract.identity_sha256
    )
    prose_changed = asyncio.run(
        policy.select(
            replace(request, finite_variation_contract=prose_changed_contract)
        )
    )
    assert tuple(member.option_id for member in first.decision.members) == tuple(
        member.option_id for member in prose_changed.decision.members
    )

    changed_keys: list[str] = []
    for member in first.decision.members:
        child = thaw_json(_contract().resolve(member.option_id).child_configuration)
        changed_keys.extend(key for key, value in child.items() if value != 0)
    assert len(changed_keys) == len(set(changed_keys)) == 3
    assert len({member.family for member in first.decision.members}) == 3


def test_impossible_grouped_space_fails_closed_at_analysis_and_selection() -> None:
    parent = _object({"x": 0})
    parent_sha256 = typed_json_sha256(parent)
    contract = FiniteVariationContract(
        catalog_id="impossible_uniform_control",
        catalog_version=1,
        catalog_definition_sha256=_sha("4"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=f"family_{ordinal}.x",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object({"x": ordinal}),
                family=f"family_{ordinal}",
                description="Conflicting fixture option.",
            )
            for ordinal in (1, 2, 3)
        ),
    )
    backward_compatible = replace(
        _request(),
        finite_variation_contract=contract,
        portfolio_size=2,
        min_distinct_families=2,
        require_pairwise_disjoint_parent_patches=False,
    )
    analysis = analyze_grouped_feasible_slate_space(backward_compatible)
    assert analysis.feasible_unordered_slate_count == 3
    assert analysis.total_unordered_slate_count == 3
    result = asyncio.run(
        TaskKeyedConditionalUniformPortfolioPolicy(
            task_sha256=_sha("b"),
            replicate_seed=1,
        ).select(backward_compatible)
    )
    assert len(result.decision.members) == 2

    with pytest.raises(ValueError, match="no feasible pairwise-disjoint"):
        replace(
            backward_compatible,
            require_pairwise_disjoint_parent_patches=True,
        )


def test_general_overlap_graph_is_counted_exactly_without_group_assumption() -> None:
    parent = _object({"a": 0, "b": 0, "c": 0, "d": 0})
    parent_sha256 = typed_json_sha256(parent)
    rows = (
        ("alpha.a", "alpha", {"a": 1, "b": 0, "c": 0, "d": 0}),
        ("beta.b", "beta", {"a": 0, "b": 1, "c": 0, "d": 0}),
        ("gamma.ab", "gamma", {"a": 1, "b": 1, "c": 0, "d": 0}),
        ("gamma.c", "gamma", {"a": 0, "b": 0, "c": 1, "d": 0}),
        ("delta.d", "delta", {"a": 0, "b": 0, "c": 0, "d": 1}),
    )
    contract = FiniteVariationContract(
        catalog_id="general_overlap_uniform_control",
        catalog_version=1,
        catalog_definition_sha256=_sha("5"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object(child),
                family=family,
                description="General overlap fixture option.",
            )
            for option_id, family, child in rows
        ),
    )
    request = replace(
        _request(),
        finite_variation_contract=contract,
        portfolio_size=3,
        min_distinct_families=3,
    )

    analysis = analyze_grouped_feasible_slate_space(request)

    assert analysis.analysis_law == "exact_compatibility_clique_count_v1"
    assert analysis.feasible_unordered_slate_count == 4
    assert analysis.total_unordered_slate_count == 10
    assert analysis.compatibility_edge_count == 8
    selected = asyncio.run(
        TaskKeyedConditionalUniformPortfolioPolicy(
            task_sha256=_sha("c"),
            replicate_seed=2,
        ).select(request)
    )
    assert len(selected.decision.members) == 3

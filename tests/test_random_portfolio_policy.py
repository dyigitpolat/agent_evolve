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
    PortfolioSelectionPolicy,
    PortfolioSelectionRequest,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.selection.random_portfolio import (
    DeterministicRandomFeasiblePortfolioPolicy,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection


def _object(value: dict[str, object]):
    frozen = freeze_json(value)
    if type(frozen).__name__ != "FrozenJsonObject":
        raise AssertionError("fixture did not freeze as an object")
    return frozen


def _card(claim: str) -> PortfolioCard:
    return PortfolioCard(
        card_key="card.control",
        reference=InsightRef(InsightId("insight_random_control"), 1),
        content_sha256="1" * 64,
        evidence_sha256="2" * 64,
        prompt_payload=_object({"claim": claim}),
        assigned_score=0.0,
    )


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
        catalog_id="random_control_fixture",
        catalog_version=1,
        catalog_definition_sha256="3" * 64,
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object(child),
                family=family,
                description=f"Apply sealed fixture option {option_id}.",
            )
            for option_id, family, child in rows
        ),
    )


def _request(*, claim: str = "payload one") -> PortfolioSelectionRequest:
    return PortfolioSelectionRequest(
        call_id=LLMCallId("call_random_control_0001"),
        operation="select_portfolio",
        instruction="Select an outcome-blind feasible random portfolio.",
        context=_object({"hidden_outcome_like_value": claim}),
        finite_variation_contract=_contract(),
        cards=(_card(claim),),
        portfolio_size=3,
        required_metric_ids=("cost", "quality"),
        min_distinct_families=3,
        require_supporting_cards=False,
        max_output_tokens=16,
    )


def test_random_policy_is_replayable_path_disjoint_and_provider_free() -> None:
    policy = DeterministicRandomFeasiblePortfolioPolicy(seed=20_260_716)
    assert isinstance(policy, PortfolioSelectionPolicy)
    first = asyncio.run(policy.select(_request()))
    second = asyncio.run(policy.select(_request()))
    assert first.decision.decision_sha256 == second.decision.decision_sha256
    assert len(first.decision.members) == 3
    assert len({member.family for member in first.decision.members}) >= 3

    changed_keys: list[str] = []
    for member in first.decision.members:
        child = thaw_json(_contract().resolve(member.option_id).child_configuration)
        changed_keys.extend(key for key, value in child.items() if value != 0)
        assert member.supporting_card_keys == ()
        assert all(
            prediction.direction is MetricEffectDirection.UNKNOWN
            for prediction in member.effect_predictions
        )
    assert len(changed_keys) == len(set(changed_keys)) == 3
    assert first.telemetry is not None
    assert first.telemetry.resolved_provider == "local-deterministic-control"
    assert first.telemetry.input_tokens == first.telemetry.output_tokens == 0
    assert first.telemetry.reasoning_tokens == 0


def test_payload_and_context_cannot_change_selected_option_ids() -> None:
    policy = DeterministicRandomFeasiblePortfolioPolicy(seed=7)
    first = asyncio.run(policy.select(_request(claim="first payload")))
    second = asyncio.run(policy.select(_request(claim="different payload")))
    assert first.decision.request_sha256 != second.decision.request_sha256
    assert tuple(member.option_id for member in first.decision.members) == tuple(
        member.option_id for member in second.decision.members
    )


def test_impossible_path_disjoint_request_fails_closed() -> None:
    parent = _object({"x": 0})
    parent_sha256 = typed_json_sha256(parent)
    contract = FiniteVariationContract(
        catalog_id="impossible_random_control",
        catalog_version=1,
        catalog_definition_sha256="4" * 64,
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="alpha.x1",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object({"x": 1}),
                family="alpha",
                description="Set x to one.",
            ),
            FiniteVariationOption(
                option_id="beta.x2",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_object({"x": 2}),
                family="beta",
                description="Set x to two.",
            ),
        ),
    )
    request = replace(
        _request(),
        finite_variation_contract=contract,
        portfolio_size=2,
        min_distinct_families=2,
    )
    with pytest.raises(ValueError, match="no path-disjoint portfolio"):
        asyncio.run(DeterministicRandomFeasiblePortfolioPolicy(seed=1).select(request))

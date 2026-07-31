"""Pins for the composition portfolio policy.

Each test locks a property the evidence requires, not an implementation detail:

* it makes no provider call (telemetry is None, always);
* it spans the proposal sources before repeating one;
* its composition comes from injected receipts, never a hard-coded mix;
* it is deterministic from the request identity, so receipts reproduce it;
* it honours engine-required options rather than overriding them;
* it satisfies the port's Protocol.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_evolve.application.composition_portfolio_selection import (
    COMPOSITION_PORTFOLIO_POLICY_DEFINITION_SHA256,
    COMPOSITION_PORTFOLIO_POLICY_ID,
    CompositionPortfolioSelectionPolicy,
    FrozenCompositionObservations,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import InsightId, LLMCallId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioSelectionPolicy,
    PortfolioSelectionRequest,
)


def _card() -> PortfolioCard:
    payload = freeze_json({"claim": "a card the policy never reads"})
    from agent_evolve.domain.typed_json import typed_json_sha256

    digest = typed_json_sha256(payload)
    return PortfolioCard(
        card_key="card.01",
        reference=InsightRef(
            insight_id=InsightId("insight_composition_0001"), version=1
        ),
        prompt_payload=payload,
        content_sha256=digest,
        evidence_sha256=digest,
    )


def _contract(counts: dict[str, int]) -> FiniteVariationContract:
    parent = freeze_json({"x": 0})
    from agent_evolve.domain.typed_json import typed_json_sha256

    parent_sha = typed_json_sha256(parent)
    options = []
    for family, count in sorted(counts.items()):
        for index in range(count):
            options.append(
                FiniteVariationOption(
                    option_id=f"{family}.v{index:02d}",
                    parent_configuration_sha256=parent_sha,
                    child_configuration=freeze_json({"x": index + 1, "f": family}),
                    family=family,
                    description=f"set {family} to variant {index}",
                )
            )
    return FiniteVariationContract(
        catalog_id="test.catalog",
        catalog_version=1,
        catalog_definition_sha256="0" * 64,
        parent_configuration=parent,
        options=tuple(options),
    )


def _request(contract, size, *, call="call_composition_0001", required=()):
    return PortfolioSelectionRequest(
        call_id=LLMCallId(call),
        operation="select_portfolio",
        instruction="select a portfolio",
        context=freeze_json({}),
        finite_variation_contract=contract,
        cards=(_card(),),
        portfolio_size=size,
        required_metric_ids=("objective_a",),
        require_supporting_cards=False,
        candidate_pool_required_option_ids=tuple(required),
    )


def _select(policy, request):
    return asyncio.run(policy.select(request))


def test_policy_satisfies_the_port_protocol():
    assert isinstance(CompositionPortfolioSelectionPolicy(), PortfolioSelectionPolicy)


def test_makes_no_provider_call():
    contract = _contract({"alpha": 4, "beta": 4})
    result = _select(CompositionPortfolioSelectionPolicy(), _request(contract, 4))
    assert result.telemetry is None, "the composition policy must not call a provider"


def test_spans_the_sources_before_repeating_one():
    contract = _contract({"alpha": 4, "beta": 4, "gamma": 4, "delta": 4})
    result = _select(CompositionPortfolioSelectionPolicy(), _request(contract, 4))
    families = {
        contract.resolve(member.option_id).family
        for member in result.decision.members
    }
    assert len(families) == 4, "four seats over four families must span all four"


def test_spanning_is_used_when_no_observations_are_supplied():
    contract = _contract({"alpha": 8, "beta": 1})
    result = _select(CompositionPortfolioSelectionPolicy(), _request(contract, 2))
    families = {
        contract.resolve(member.option_id).family
        for member in result.decision.members
    }
    assert families == {"alpha", "beta"}, (
        "with no observations the policy must span, not follow catalogue mass"
    )


def test_composition_follows_injected_receipts_not_a_hard_coded_mix():
    contract = _contract({"alpha": 8, "beta": 8})
    skewed = FrozenCompositionObservations(shares=(("alpha", 3.0), ("beta", 1.0)))
    result = _select(
        CompositionPortfolioSelectionPolicy(observations=skewed),
        _request(contract, 4),
    )
    families = [
        contract.resolve(m.option_id).family for m in result.decision.members
    ]
    assert families.count("alpha") == 3 and families.count("beta") == 1, (
        "the budget must be apportioned to the injected shares"
    )
    # and the opposite skew must give the opposite split: nothing is baked in
    flipped = FrozenCompositionObservations(shares=(("alpha", 1.0), ("beta", 3.0)))
    result = _select(
        CompositionPortfolioSelectionPolicy(observations=flipped),
        _request(contract, 4),
    )
    families = [
        contract.resolve(m.option_id).family for m in result.decision.members
    ]
    assert families.count("beta") == 3 and families.count("alpha") == 1


def test_is_deterministic_from_the_request_identity():
    contract = _contract({"alpha": 8, "beta": 8})
    policy = CompositionPortfolioSelectionPolicy()
    first = _select(policy, _request(contract, 4))
    second = _select(policy, _request(contract, 4))
    assert [m.option_id for m in first.decision.members] == [
        m.option_id for m in second.decision.members
    ], "receipts must reproduce the decision exactly"


def test_a_different_call_id_may_draw_differently():
    contract = _contract({"alpha": 16, "beta": 16})
    policy = CompositionPortfolioSelectionPolicy()
    first = _select(policy, _request(contract, 4, call="call_composition_0001"))
    second = _select(policy, _request(contract, 4, call="call_composition_0002"))
    # the composition is fixed; only the member drawn inside it may vary
    families_first = sorted(
        contract.resolve(m.option_id).family for m in first.decision.members
    )
    families_second = sorted(
        contract.resolve(m.option_id).family for m in second.decision.members
    )
    assert families_first == families_second


def test_engine_required_options_are_honoured():
    contract = _contract({"alpha": 8, "beta": 8})
    required = ("alpha.v03",)
    result = _select(
        CompositionPortfolioSelectionPolicy(),
        _request(contract, 4, required=required),
    )
    assert "alpha.v03" in {m.option_id for m in result.decision.members}


def test_policy_identity_is_pinned():
    assert COMPOSITION_PORTFOLIO_POLICY_ID == "composition_spanning_portfolio"
    assert len(COMPOSITION_PORTFOLIO_POLICY_DEFINITION_SHA256) == 64
    assert set(COMPOSITION_PORTFOLIO_POLICY_DEFINITION_SHA256) <= set("0123456789abcdef")


def test_rejects_a_non_request():
    with pytest.raises(TypeError):
        _select(CompositionPortfolioSelectionPolicy(), object())


def _colliding_contract() -> FiniteVariationContract:
    """Two options per family that patch the SAME parent path, so they collide."""

    from agent_evolve.domain.typed_json import typed_json_sha256

    parent = freeze_json({"a": 0, "b": 0, "c": 0, "d": 0})
    parent_sha = typed_json_sha256(parent)
    options = []
    for family, path in (("alpha", "a"), ("beta", "b"), ("gamma", "c"), ("delta", "d")):
        for index in range(2):
            child = dict({"a": 0, "b": 0, "c": 0, "d": 0})
            child[path] = index + 1
            options.append(
                FiniteVariationOption(
                    option_id=f"{family}.v{index:02d}",
                    parent_configuration_sha256=parent_sha,
                    child_configuration=freeze_json(child),
                    family=family,
                    description=f"set {path} to {index + 1}",
                )
            )
    return FiniteVariationContract(
        catalog_id="test.catalog",
        catalog_version=1,
        catalog_definition_sha256="0" * 64,
        parent_configuration=parent,
        options=tuple(options),
    )


def test_emits_a_pairwise_disjoint_slate_when_the_request_requires_it():
    """The crash this fixes: a composition slate the port rejects."""

    contract = _colliding_contract()
    request = PortfolioSelectionRequest(
        call_id=LLMCallId("call_composition_legal_0001"),
        operation="select_portfolio",
        instruction="select a portfolio",
        context=freeze_json({}),
        finite_variation_contract=contract,
        cards=(_card(),),
        portfolio_size=4,
        required_metric_ids=("objective_a",),
        require_supporting_cards=False,
        require_pairwise_disjoint_parent_patches=True,
    )
    result = _select(CompositionPortfolioSelectionPolicy(), request)
    chosen = [m.option_id for m in result.decision.members]
    assert len(chosen) == 4
    patched = [contract.resolve(o).family for o in chosen]
    assert len(set(patched)) == 4, (
        "a legal slate here must take one option from each colliding group"
    )


def test_legality_projection_is_off_when_the_request_does_not_require_it():
    contract = _contract({"alpha": 4, "beta": 4})
    request = _request(contract, 4)
    assert request.require_pairwise_disjoint_parent_patches is False
    result = _select(CompositionPortfolioSelectionPolicy(), request)
    assert len(result.decision.members) == 4


def test_asserts_provider_free_rather_than_leaving_telemetry_absent():
    contract = _contract({"alpha": 4, "beta": 4})
    result = _select(CompositionPortfolioSelectionPolicy(), _request(contract, 4))
    assert result.telemetry is None
    assert result.provider_free is True, (
        "absence must be asserted, not inferred from a missing field"
    )


def test_result_rejects_unasserted_absence():
    """A bare None telemetry with no assertion is 'unrecorded', not 'zero'."""

    from agent_evolve.ports.portfolio_selection import PortfolioSelectionResult

    contract = _contract({"alpha": 4, "beta": 4})
    good = _select(CompositionPortfolioSelectionPolicy(), _request(contract, 4))
    with pytest.raises(ValueError, match="absence must be asserted"):
        PortfolioSelectionResult(
            decision=good.decision, telemetry=None, provider_free=False
        )


def test_result_rejects_provider_free_that_also_carries_telemetry():
    from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
    from agent_evolve.ports.portfolio_selection import PortfolioSelectionResult

    contract = _contract({"alpha": 4, "beta": 4})
    good = _select(CompositionPortfolioSelectionPolicy(), _request(contract, 4))
    telemetry = AgenticCallTelemetry(
        requested_model="m", resolved_model="m", resolved_provider="p",
        provider_response_id=None, finish_reason="stop",
        input_tokens=1, output_tokens=1, reasoning_tokens=0,
        cache_read_tokens=0, cache_write_tokens=0, cost_usd=None, latency_ns=1,
    )
    with pytest.raises(ValueError, match="cannot also carry call telemetry"):
        PortfolioSelectionResult(
            decision=good.decision, telemetry=telemetry, provider_free=True
        )

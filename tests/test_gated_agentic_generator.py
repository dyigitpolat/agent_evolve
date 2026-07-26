from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal

import pytest

from agent_evolve.application.gated_agentic_generator import (
    AgenticTelemetryPolicy,
    AgenticTelemetryRejected,
    TelemetryGatedAgenticGenerator,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
)


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="deepseek/deepseek-v4-pro",
        resolved_model="deepseek/deepseek-v4-pro",
        resolved_provider="DeepSeek",
        provider_response_id="response-1",
        finish_reason="stop",
        input_tokens=1_000,
        output_tokens=100,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.000522"),
        latency_ns=2,
        attempt_count=1,
    )


def _policy() -> AgenticTelemetryPolicy:
    return AgenticTelemetryPolicy(
        requested_model="deepseek/deepseek-v4-pro",
        allowed_resolved_models=(
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-pro-20260423",
        ),
        allowed_resolved_providers=("DeepSeek",),
        max_cost_usd=Decimal("0.006"),
        max_input_tokens=10_000,
        max_output_tokens=640,
        max_reasoning_tokens=640,
        max_attempt_count=2,
    )


class _Generator:
    def __init__(self, telemetry: AgenticCallTelemetry) -> None:
        self.telemetry = telemetry

    async def propose(self, request):
        return VariationGenerationResult(
            CandidateDraft({"x": 2}, "one candidate"), self.telemetry
        )

    async def reflect(self, request):
        return ReflectionGenerationResult((), self.telemetry)


def test_policy_identity_is_stable_and_gate_passes_both_call_kinds() -> None:
    policy = _policy()
    assert policy.policy_sha256 == _policy().policy_sha256
    assert policy.policy_sha256 == (
        "c910d285eec0c40d63c81b8634553c61fdbb8d8a111cd31e9dcf52a562117b6d"
    )
    assert policy.to_trace_record()["policy_version"] == 3
    assert policy.to_trace_record()["reasoning_token_accounting"] == (
        "included_in_output_tokens"
    )
    gated = TelemetryGatedAgenticGenerator(_Generator(_telemetry()), policy)
    proposal = asyncio.run(
        gated.propose(
            VariationGenerationRequest(
                call_id=LLMCallId("call_test_1"),
                operation="typed_mutation",
                prompt="prompt",
                candidate_model=dict,
            )
        )
    )
    reflection = asyncio.run(
        gated.reflect(
            ReflectionGenerationRequest(
                call_id=LLMCallId("call_test_2"),
                operation="reflection",
                prompt="prompt",
            )
        )
    )
    assert proposal.telemetry == _telemetry()
    assert reflection.telemetry == _telemetry()
    policy.validate(
        replace(
            _telemetry(),
            resolved_model="deepseek/deepseek-v4-pro-20260423",
        )
    )
    assert (
        policy.policy_sha256
        != replace(
            policy,
            allowed_resolved_models=("deepseek/deepseek-v4-pro",),
        ).policy_sha256
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"requested_model": "other/model"}, "requested_model"),
        ({"resolved_model": "other/model"}, "resolved_model"),
        ({"resolved_provider": "Other"}, "resolved_provider"),
        ({"input_tokens": 10_001}, "input_tokens"),
        ({"output_tokens": 641}, "output_tokens"),
        ({"attempt_count": 3}, "attempt_count"),
        ({"cost_usd": None}, "missing_cost"),
        ({"cost_usd": Decimal("0.006001")}, "cost_ceiling"),
    ],
)
def test_policy_rejects_route_or_resource_drift_before_return(change, reason) -> None:
    telemetry = replace(_telemetry(), **change)
    gated = TelemetryGatedAgenticGenerator(_Generator(telemetry), _policy())
    with pytest.raises(AgenticTelemetryRejected, match=reason):
        asyncio.run(
            gated.propose(
                VariationGenerationRequest(
                    call_id=LLMCallId("call_test_reject"),
                    operation="typed_mutation",
                    prompt="prompt",
                    candidate_model=dict,
                )
            )
        )


def test_configured_reasoning_ceiling_remains_a_hard_gate() -> None:
    policy = replace(_policy(), max_reasoning_tokens=64)
    policy.validate(replace(_telemetry(), reasoning_tokens=64))

    with pytest.raises(AgenticTelemetryRejected, match="reasoning_tokens"):
        policy.validate(replace(_telemetry(), reasoning_tokens=65))


def test_absent_reasoning_ceiling_accepts_usage_within_aggregate_output() -> None:
    policy = replace(
        _policy(),
        max_output_tokens=384_000,
        max_reasoning_tokens=None,
    )
    observed = replace(
        _telemetry(),
        output_tokens=5_527,
        reasoning_tokens=4_565,
    )

    policy.validate(observed)

    record = policy.to_trace_record()
    assert record["max_reasoning_tokens"] is None
    assert policy.policy_sha256 != _policy().policy_sha256


def test_reasoning_usage_must_be_included_in_aggregate_output() -> None:
    policy = replace(
        _policy(),
        max_output_tokens=384_000,
        max_reasoning_tokens=None,
    )

    with pytest.raises(
        AgenticTelemetryRejected,
        match="reasoning_output_accounting",
    ):
        policy.validate(
            replace(
                _telemetry(),
                output_tokens=4_564,
                reasoning_tokens=4_565,
            )
        )


@pytest.mark.parametrize("value", [True, -1, 1.5, "64"])
def test_reasoning_ceiling_rejects_invalid_configured_values(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="max_reasoning_tokens must be a non-negative integer or None",
    ):
        replace(_policy(), max_reasoning_tokens=value)  # type: ignore[arg-type]


def test_per_call_ceiling_composes_with_call_budget_without_timing_order() -> None:
    policy = _policy()
    assert policy.max_cost_usd * 5 == Decimal("0.030")

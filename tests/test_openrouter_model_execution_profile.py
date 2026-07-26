"""Workload-neutral conformance for the cross-model execution matrix."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH,
    DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
    GPT_5_6_SOL_OPENAI_XHIGH,
    GPT_OSS_120B_GROQ_HIGH,
    GPT_OSS_20B_GROQ_HIGH,
    GPT_OSS_20B_GROQ_HIGH_SERIAL,
    MISTRAL_LARGE_3_MISTRAL,
    OPENROUTER_MODEL_EXECUTION_PROFILES,
    OPENROUTER_MODEL_EXECUTION_PROFILE_VARIANTS,
    QWEN_3_7_MAX_ALIBABA_XHIGH,
    QWEN_3_7_MAX_ALIBABA_XHIGH_RATE_SAFE,
    OpenRouterModelExecutionProfile,
    openrouter_model_execution_profile,
)
from agent_evolve.integrations.pydantic_ai.json_schema_dialect import (
    OpenRouterJsonSchemaDialect,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry


EXPECTED_PROFILE_NAMES = (
    "deepseek",
    "gpt_oss_120b",
    "gpt_oss_20b",
    "gpt_sol",
    "mistral",
    "qwen",
)


def _telemetry(
    profile: OpenRouterModelExecutionProfile,
    *,
    resolved_model: str | None = None,
    reasoning_tokens: int,
) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model=profile.requested_model,
        resolved_model=resolved_model or profile.requested_model,
        resolved_provider=profile.accepted_resolved_providers[0],
        provider_response_id="response-test",
        finish_reason=profile.accepted_finish_reasons[0],
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.001"),
        latency_ns=1,
    )


def test_six_requested_profiles_are_closed_unique_and_workload_neutral() -> None:
    assert tuple(OPENROUTER_MODEL_EXECUTION_PROFILES) == EXPECTED_PROFILE_NAMES
    profiles = tuple(OPENROUTER_MODEL_EXECUTION_PROFILES.values())
    assert len({profile.profile_sha256 for profile in profiles}) == len(profiles)
    assert all(
        profile.to_record()["workload_specific_fields"] == []
        for profile in profiles
    )
    assert all(
        openrouter_model_execution_profile(name) is profile
        for name, profile in OPENROUTER_MODEL_EXECUTION_PROFILES.items()
    )


def test_transport_repair_variant_is_explicit_and_does_not_change_roster() -> None:
    profile = DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON
    assert OPENROUTER_MODEL_EXECUTION_PROFILE_VARIANTS == {
        "deepseek_json": profile,
        "gpt_oss_20b_serial": GPT_OSS_20B_GROQ_HIGH_SERIAL,
        "qwen_rate_safe": QWEN_3_7_MAX_ALIBABA_XHIGH_RATE_SAFE,
    }
    assert openrouter_model_execution_profile("deepseek_json") is profile
    assert profile.requested_model == DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH.requested_model
    assert profile.structured_output_mode.value == "native_json_schema"
    assert profile.profile_sha256 != DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH.profile_sha256
    assert "deepseek_json" not in OPENROUTER_MODEL_EXECUTION_PROFILES


def test_route_capacity_variant_serializes_without_changing_the_model_route() -> None:
    original = GPT_OSS_20B_GROQ_HIGH
    serial = GPT_OSS_20B_GROQ_HIGH_SERIAL

    assert original.effective_max_connections(default=3) == 3
    assert serial.effective_max_connections(default=3) == 1
    assert serial.requested_model == original.requested_model
    assert serial.provider_only == original.provider_only
    assert serial.to_record()["route_concurrency_cap"] == 1
    assert serial.profile_sha256 != original.profile_sha256
    assert openrouter_model_execution_profile("gpt_oss_20b_serial") is serial
    with pytest.raises(ValueError, match="route_concurrency_cap"):
        replace(serial, route_concurrency_cap=0)


def test_qwen_rate_safe_variant_binds_cap_and_rate_limit_floor() -> None:
    original = QWEN_3_7_MAX_ALIBABA_XHIGH
    repaired = QWEN_3_7_MAX_ALIBABA_XHIGH_RATE_SAFE

    assert repaired.requested_model == original.requested_model
    assert repaired.provider_only == original.provider_only
    assert repaired.effective_max_connections(default=3) == 2
    assert repaired.rate_limit_backoff_floor_ns == 15_000_000_000
    assert repaired.to_record()["rate_limit_backoff_floor_ns"] == 15_000_000_000
    assert repaired.profile_sha256 != original.profile_sha256
    assert openrouter_model_execution_profile("qwen_rate_safe") is repaired
    with pytest.raises(ValueError, match="rate_limit_backoff_floor_ns"):
        replace(repaired, rate_limit_backoff_floor_ns=-1)


def test_non_reasoning_mistral_profile_omits_reasoning_and_accepts_zero_tokens() -> None:
    profile = MISTRAL_LARGE_3_MISTRAL
    assert profile.reasoning_config is None
    assert profile.outbound_reasoning_setting is None
    assert profile.accepts_reasoning_tokens(0) is True
    assert profile.accepts_reasoning_tokens(-1) is False
    profile.validate_telemetry(_telemetry(profile, reasoning_tokens=0))


def test_reasoning_profiles_require_positive_tokens() -> None:
    for name in EXPECTED_PROFILE_NAMES:
        profile = openrouter_model_execution_profile(name)
        if profile.require_positive_reasoning_tokens:
            assert profile.outbound_reasoning_setting is not None
            assert profile.accepts_reasoning_tokens(1) is True
            assert profile.accepts_reasoning_tokens(0) is False
            with pytest.raises(ValueError, match="reasoning-token contract"):
                profile.validate_telemetry(_telemetry(profile, reasoning_tokens=0))


def test_gpt_sol_accepts_the_authenticated_versioned_resolution() -> None:
    profile = GPT_5_6_SOL_OPENAI_XHIGH
    profile.validate_telemetry(
        _telemetry(
            profile,
            resolved_model="openai/gpt-5.6-sol-20260709",
            reasoning_tokens=1,
        )
    )
    with pytest.raises(ValueError, match="foreign resolved model"):
        profile.validate_telemetry(
            _telemetry(
                profile,
                resolved_model="openai/gpt-5.6-sol-foreign",
                reasoning_tokens=1,
            )
        )


def test_non_reasoning_profile_cannot_demand_positive_reasoning_tokens() -> None:
    with pytest.raises(ValueError, match="non-reasoning profile"):
        replace(
            MISTRAL_LARGE_3_MISTRAL,
            require_positive_reasoning_tokens=True,
        )


def test_route_capabilities_own_tool_forcing_and_schema_dialect() -> None:
    assert DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH.supports_forced_tool_choice is False
    assert (
        DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH.to_record()[
            "supports_forced_tool_choice"
        ]
        is False
    )
    for profile in (GPT_OSS_120B_GROQ_HIGH, GPT_OSS_20B_GROQ_HIGH):
        assert (
            profile.json_schema_dialect
            is OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT
        )
        assert "OPENAI_STRICT_BOUNDED_TEXT" not in repr(profile.to_record())


def test_route_capability_fields_reject_non_boolean_values() -> None:
    with pytest.raises(TypeError, match="supports_forced_tool_choice"):
        replace(
            DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH,
            supports_forced_tool_choice=1,  # type: ignore[arg-type]
        )

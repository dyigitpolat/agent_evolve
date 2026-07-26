"""Authenticated, workload-neutral OpenRouter model execution profiles.

Workload adapters describe candidates and evaluators.  They must not decide
provider routing, reasoning controls, output ceilings, or terminal telemetry
requirements.  This module keeps those model-facing choices in one immutable
composition object that can be injected into any AgentEvolve campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
    OpenRouterStructuredOutputMode,
)
from agent_evolve.integrations.pydantic_ai.json_schema_dialect import (
    OpenRouterJsonSchemaDialect,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry


_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MODEL = re.compile(r"^[a-z0-9][a-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PROVIDER_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROFILE_DOMAIN = b"agent-evolve:openrouter-model-execution-profile:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _canonical_strings(
    values: tuple[str, ...],
    *,
    name: str,
    pattern: re.Pattern[str] | None = None,
) -> None:
    if type(values) is not tuple or not values or any(
        type(value) is not str
        or not value
        or (pattern is not None and pattern.fullmatch(value) is None)
        for value in values
    ):
        raise TypeError(f"{name} must be a non-empty exact string tuple")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


@dataclass(frozen=True, slots=True, eq=False)
class OpenRouterModelExecutionProfile:
    """One complete model/provider contract independent of any workload."""

    profile_id: str
    profile_version: int
    requested_model: str
    accepted_resolved_models: tuple[str, ...]
    provider_only: tuple[str, ...]
    accepted_resolved_providers: tuple[str, ...]
    max_output_tokens: int
    reasoning_effort: str | None
    temperature: float | None
    structured_output_mode: OpenRouterStructuredOutputMode = (
        OpenRouterStructuredOutputMode.TOOL
    )
    structured_output_strict: bool = False
    json_schema_dialect: OpenRouterJsonSchemaDialect = (
        OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT
    )
    require_positive_reasoning_tokens: bool = True
    provider_require_parameters: bool = True
    supports_forced_tool_choice: bool = True
    route_concurrency_cap: int | None = None
    rate_limit_backoff_floor_ns: int = 0

    def __post_init__(self) -> None:
        if type(self.profile_id) is not str or _TOKEN.fullmatch(self.profile_id) is None:
            raise ValueError("profile_id must use the closed lowercase token grammar")
        if type(self.profile_version) is not int or self.profile_version <= 0:
            raise ValueError("profile_version must be a positive exact integer")
        if (
            type(self.requested_model) is not str
            or _MODEL.fullmatch(self.requested_model) is None
        ):
            raise ValueError("requested_model must be an OpenRouter model slug")
        _canonical_strings(
            self.accepted_resolved_models,
            name="accepted_resolved_models",
            pattern=_MODEL,
        )
        if self.requested_model not in self.accepted_resolved_models:
            raise ValueError("requested_model must be an accepted resolved model")
        _canonical_strings(
            self.provider_only,
            name="provider_only",
            pattern=_PROVIDER_SLUG,
        )
        _canonical_strings(
            self.accepted_resolved_providers,
            name="accepted_resolved_providers",
        )
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive exact integer")
        if self.reasoning_effort is not None:
            if type(self.reasoning_effort) is not str:
                raise TypeError("reasoning_effort must be an exact string or None")
            OpenRouterReasoningConfig(effort=self.reasoning_effort).__post_init__()
        if self.temperature is not None and (
            type(self.temperature) is not float
            or not math.isfinite(self.temperature)
            or not 0.0 <= self.temperature <= 2.0
        ):
            raise ValueError("temperature must be None or a finite float in [0,2]")
        if type(self.structured_output_mode) is not OpenRouterStructuredOutputMode:
            raise TypeError(
                "structured_output_mode must be an exact "
                "OpenRouterStructuredOutputMode"
            )
        if type(self.structured_output_strict) is not bool:
            raise TypeError("structured_output_strict must be an exact bool")
        if type(self.json_schema_dialect) is not OpenRouterJsonSchemaDialect:
            raise TypeError(
                "json_schema_dialect must be an exact "
                "OpenRouterJsonSchemaDialect"
            )
        if type(self.require_positive_reasoning_tokens) is not bool:
            raise TypeError("require_positive_reasoning_tokens must be exact bool")
        if self.reasoning_effort is None and self.require_positive_reasoning_tokens:
            raise ValueError(
                "a non-reasoning profile cannot require positive reasoning tokens"
            )
        if type(self.provider_require_parameters) is not bool:
            raise TypeError("provider_require_parameters must be exact bool")
        if type(self.supports_forced_tool_choice) is not bool:
            raise TypeError("supports_forced_tool_choice must be exact bool")
        if self.route_concurrency_cap is not None and (
            type(self.route_concurrency_cap) is not int
            or self.route_concurrency_cap <= 0
        ):
            raise ValueError(
                "route_concurrency_cap must be a positive exact integer or None"
            )
        if (
            type(self.rate_limit_backoff_floor_ns) is not int
            or self.rate_limit_backoff_floor_ns < 0
        ):
            raise ValueError(
                "rate_limit_backoff_floor_ns must be a non-negative exact integer"
            )

    @property
    def reasoning_config(self) -> OpenRouterReasoningConfig | None:
        self.__post_init__()
        if self.reasoning_effort is None:
            return None
        return OpenRouterReasoningConfig(effort=self.reasoning_effort)

    @property
    def outbound_reasoning_setting(self) -> dict[str, object] | None:
        """Return the exact OpenRouter reasoning field expected on the wire."""

        config = self.reasoning_config
        return None if config is None else config.to_model_setting()

    def accepts_reasoning_tokens(self, value: object) -> bool:
        """Validate telemetry without assuming that every model reasons."""

        self.__post_init__()
        return type(value) is int and (
            value > 0 if self.require_positive_reasoning_tokens else value >= 0
        )

    def effective_max_connections(self, *, default: int) -> int:
        """Resolve the authenticated route cap against a composition default."""

        self.__post_init__()
        if type(default) is not int or default <= 0:
            raise ValueError("default must be a positive exact integer")
        return (
            default
            if self.route_concurrency_cap is None
            else min(default, self.route_concurrency_cap)
        )

    @property
    def accepted_finish_reasons(self) -> tuple[str, ...]:
        """Return the closed terminal-reason contract for the wire mode."""

        self.__post_init__()
        if self.structured_output_mode is OpenRouterStructuredOutputMode.TOOL:
            return ("tool_call",)
        if (
            self.structured_output_mode
            is OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
        ):
            return ("stop",)
        raise AssertionError("unreachable structured output mode")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "schema_version": 2,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "requested_model": self.requested_model,
            "accepted_resolved_models": list(self.accepted_resolved_models),
            "provider_options": {
                "only": list(self.provider_only),
                "allow_fallbacks": False,
                "require_parameters": self.provider_require_parameters,
            },
            "accepted_resolved_providers": list(
                self.accepted_resolved_providers
            ),
            "max_output_tokens": self.max_output_tokens,
            "reasoning": self.outbound_reasoning_setting,
            "structured_output_mode": self.structured_output_mode.value,
            "structured_output_strict": self.structured_output_strict,
            "json_schema_dialect": self.json_schema_dialect.value,
            "supports_forced_tool_choice": self.supports_forced_tool_choice,
            "temperature_hex": (
                None if self.temperature is None else self.temperature.hex()
            ),
            "terminal_telemetry": {
                "requested_model_exact": True,
                "resolved_model_allowlist": True,
                "resolved_provider_allowlist": True,
                "accepted_finish_reasons": list(self.accepted_finish_reasons),
                "positive_reasoning_tokens_required": (
                    self.require_positive_reasoning_tokens
                ),
            },
            "workload_specific_fields": [],
        }
        if self.route_concurrency_cap is not None:
            record["schema_version"] = 3
            record["route_concurrency_cap"] = self.route_concurrency_cap
        if self.rate_limit_backoff_floor_ns:
            record["schema_version"] = 4
            record["rate_limit_backoff_floor_ns"] = (
                self.rate_limit_backoff_floor_ns
            )
        return record

    @property
    def profile_sha256(self) -> str:
        return hashlib.sha256(
            _PROFILE_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "profile_sha256": self.profile_sha256}

    def validate_telemetry(self, telemetry: AgenticCallTelemetry) -> None:
        """Fail closed when a terminal response escapes the frozen route."""

        self.__post_init__()
        if type(telemetry) is not AgenticCallTelemetry:
            raise TypeError("telemetry must be exact AgenticCallTelemetry")
        AgenticCallTelemetry.__post_init__(telemetry)
        if telemetry.requested_model != self.requested_model:
            raise ValueError("terminal telemetry has a foreign requested model")
        if telemetry.resolved_model not in self.accepted_resolved_models:
            raise ValueError("terminal telemetry has a foreign resolved model")
        if telemetry.resolved_provider not in self.accepted_resolved_providers:
            raise ValueError("terminal telemetry has a foreign resolved provider")
        if telemetry.finish_reason not in self.accepted_finish_reasons:
            raise ValueError("terminal telemetry has a foreign finish reason")
        if not self.accepts_reasoning_tokens(telemetry.reasoning_tokens):
            raise ValueError("terminal telemetry violates the reasoning-token contract")

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is OpenRouterModelExecutionProfile
            and self.profile_sha256 == other.profile_sha256
        )

    __hash__ = None


DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH = OpenRouterModelExecutionProfile(
    profile_id="deepseek_v4_pro_streamlake_xhigh",
    profile_version=2,
    requested_model="deepseek/deepseek-v4-pro",
    accepted_resolved_models=("deepseek/deepseek-v4-pro",),
    provider_only=("streamlake",),
    accepted_resolved_providers=("StreamLake",),
    max_output_tokens=384_000,
    reasoning_effort="xhigh",
    temperature=0.2,
    # StreamLake intermittently rejects ``tool_choice=required`` when xhigh
    # thinking is active.  The one-tool AgentEvolve contract remains closed,
    # while Pydantic-AI emits ``auto`` and the semantic prompt requires that
    # sole output tool.  Terminal typed validation is unchanged.
    supports_forced_tool_choice=False,
    provider_require_parameters=False,
)


# Prospective transport repair for providers that cannot reliably combine
# long reasoning with a forced output tool.  Keep the legacy tool profile
# addressable so completed studies remain replayable; promotion of this
# response-format variant requires its own canary and registry identity.
DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON = OpenRouterModelExecutionProfile(
    profile_id="deepseek_v4_pro_streamlake_xhigh_native_json",
    profile_version=1,
    requested_model="deepseek/deepseek-v4-pro",
    accepted_resolved_models=("deepseek/deepseek-v4-pro",),
    provider_only=("streamlake",),
    accepted_resolved_providers=("StreamLake",),
    max_output_tokens=384_000,
    reasoning_effort="xhigh",
    temperature=0.2,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    structured_output_strict=False,
    json_schema_dialect=OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT,
    provider_require_parameters=False,
    supports_forced_tool_choice=False,
)


QWEN_3_7_MAX_ALIBABA_XHIGH = OpenRouterModelExecutionProfile(
    profile_id="qwen_3_7_max_alibaba_xhigh",
    profile_version=2,
    requested_model="qwen/qwen3.7-max",
    accepted_resolved_models=("qwen/qwen3.7-max",),
    provider_only=("alibaba",),
    accepted_resolved_providers=("Alibaba",),
    max_output_tokens=65_536,
    reasoning_effort="xhigh",
    temperature=0.2,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    json_schema_dialect=OpenRouterJsonSchemaDialect.ALIBABA_NATIVE_V1,
    # Alibaba advertises ``max_tokens`` while Pydantic-AI's OpenAI-compatible
    # transport serializes ``max_completion_tokens``. OpenRouter can bridge
    # that spelling only when its capability prefilter is disabled. Routing
    # remains Alibaba-only with fallbacks forbidden, and AgentEvolve still
    # authenticates the exact wire budget and validates the typed response.
    provider_require_parameters=False,
)


# Prospective provider-systems repair after an otherwise healthy Qwen/BOiLS
# campaign launched reflection plus two selectors concurrently.  Alibaba
# accepted the earlier two-selector wave but returned repeated HTTP 429s for
# the three-call wave.  The model, provider, reasoning, schema, temperature,
# and completion budget remain identical.  This explicit variant caps the
# route at the highest observed healthy concurrency and prevents full jitter
# from reducing an admitted rate-limit retry to a sub-second delay.
QWEN_3_7_MAX_ALIBABA_XHIGH_RATE_SAFE = OpenRouterModelExecutionProfile(
    profile_id="qwen_3_7_max_alibaba_xhigh_rate_safe",
    profile_version=1,
    requested_model="qwen/qwen3.7-max",
    accepted_resolved_models=("qwen/qwen3.7-max",),
    provider_only=("alibaba",),
    accepted_resolved_providers=("Alibaba",),
    max_output_tokens=65_536,
    reasoning_effort="xhigh",
    temperature=0.2,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    json_schema_dialect=OpenRouterJsonSchemaDialect.ALIBABA_NATIVE_V1,
    provider_require_parameters=False,
    route_concurrency_cap=2,
    rate_limit_backoff_floor_ns=15_000_000_000,
)


MISTRAL_LARGE_3_MISTRAL = OpenRouterModelExecutionProfile(
    profile_id="mistral_large_3_mistral",
    profile_version=3,
    requested_model="mistralai/mistral-large-2512",
    accepted_resolved_models=("mistralai/mistral-large-2512",),
    provider_only=("mistral",),
    accepted_resolved_providers=("Mistral",),
    max_output_tokens=131_072,
    reasoning_effort=None,
    temperature=0.2,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    structured_output_strict=True,
    require_positive_reasoning_tokens=False,
    # Mistral advertises ``max_tokens`` while the maintained OpenAI-compatible
    # transport emits ``max_completion_tokens``.  Keep the sole Mistral route
    # pinned and fallbacks forbidden while allowing OpenRouter's field bridge.
    provider_require_parameters=False,
)


GPT_5_6_SOL_OPENAI_XHIGH = OpenRouterModelExecutionProfile(
    profile_id="gpt_5_6_sol_openai_xhigh",
    profile_version=2,
    requested_model="openai/gpt-5.6-sol",
    accepted_resolved_models=(
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol-20260709",
    ),
    provider_only=("openai",),
    accepted_resolved_providers=("OpenAI",),
    max_output_tokens=128_000,
    reasoning_effort="xhigh",
    # The pinned first-party endpoint does not advertise temperature.
    temperature=None,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    structured_output_strict=True,
    json_schema_dialect=(
        OpenRouterJsonSchemaDialect.OPENAI_STRICT_BOUNDED_TEXT_V1
    ),
    # The endpoint advertises ``max_tokens`` while the maintained transport
    # emits ``max_completion_tokens``. Routing is still pinned to OpenAI with
    # fallbacks forbidden; this only permits OpenRouter's spelling bridge.
    provider_require_parameters=False,
)


GPT_OSS_120B_GROQ_HIGH = OpenRouterModelExecutionProfile(
    profile_id="gpt_oss_120b_groq_high",
    profile_version=7,
    requested_model="openai/gpt-oss-120b",
    accepted_resolved_models=("openai/gpt-oss-120b",),
    provider_only=("groq",),
    accepted_resolved_providers=("Groq",),
    # The endpoint's 131,072-token context is shared by input and output.  A
    # 65,536-token completion ceiling preserves half the window for evolving
    # prompts, memory, and schemas while remaining far above observed outputs.
    max_output_tokens=65_536,
    reasoning_effort="high",
    temperature=0.2,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    structured_output_strict=True,
    # Groq accepts native length bounds but rejects the OpenAI-dialect rewrite
    # ``^.{1,16384}$`` because its regex engine caps repeat counts.  The base
    # transformer removes unsupported bounds while local Pydantic validation
    # remains authoritative.
    json_schema_dialect=OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT,
    # Groq advertises the OpenAI-compatible ``max_tokens`` capability,
    # while the Pydantic-AI transport emits ``max_completion_tokens``.  Keep
    # the route pinned and fallbacks forbidden, but let OpenRouter translate
    # the equivalent field spelling instead of prefiltering the sole route.
    provider_require_parameters=False,
)


GPT_OSS_20B_GROQ_HIGH = OpenRouterModelExecutionProfile(
    profile_id="gpt_oss_20b_groq_high",
    profile_version=7,
    requested_model="openai/gpt-oss-20b",
    accepted_resolved_models=("openai/gpt-oss-20b",),
    provider_only=("groq",),
    accepted_resolved_providers=("Groq",),
    max_output_tokens=65_536,
    reasoning_effort="high",
    temperature=0.2,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    structured_output_strict=True,
    json_schema_dialect=OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT,
    provider_require_parameters=False,
)


# Prospective deployment repair after a two-connection reference cell received
# two HTTP 429 responses followed by a long-stream HTTP 502 while its sibling
# occupied the other Groq connection.  The requested model, provider, schema,
# reasoning, and completion budget are unchanged; only queue admission is
# serialized.  Keep the original profile addressable for exact replay.
GPT_OSS_20B_GROQ_HIGH_SERIAL = OpenRouterModelExecutionProfile(
    profile_id="gpt_oss_20b_groq_high_serial",
    profile_version=1,
    requested_model="openai/gpt-oss-20b",
    accepted_resolved_models=("openai/gpt-oss-20b",),
    provider_only=("groq",),
    accepted_resolved_providers=("Groq",),
    max_output_tokens=65_536,
    reasoning_effort="high",
    temperature=0.2,
    structured_output_mode=(
        OpenRouterStructuredOutputMode.NATIVE_JSON_SCHEMA
    ),
    structured_output_strict=True,
    json_schema_dialect=OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT,
    provider_require_parameters=False,
    route_concurrency_cap=1,
)


OPENROUTER_MODEL_EXECUTION_PROFILES = {
    "deepseek": DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH,
    "gpt_oss_120b": GPT_OSS_120B_GROQ_HIGH,
    "gpt_oss_20b": GPT_OSS_20B_GROQ_HIGH,
    "gpt_sol": GPT_5_6_SOL_OPENAI_XHIGH,
    "mistral": MISTRAL_LARGE_3_MISTRAL,
    "qwen": QWEN_3_7_MAX_ALIBABA_XHIGH,
}

# Transport variants are deliberately excluded from the six-model scientific
# roster above.  They are separately named so a registry cannot silently
# reinterpret an already completed model cell after a route-capability repair.
OPENROUTER_MODEL_EXECUTION_PROFILE_VARIANTS = {
    "deepseek_json": DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
    "gpt_oss_20b_serial": GPT_OSS_20B_GROQ_HIGH_SERIAL,
    "qwen_rate_safe": QWEN_3_7_MAX_ALIBABA_XHIGH_RATE_SAFE,
}


def openrouter_model_execution_profile(
    name: str,
) -> OpenRouterModelExecutionProfile:
    """Resolve one closed built-in profile without workload-dependent logic."""

    if type(name) is not str:
        raise TypeError("model profile name must be an exact string")
    matches = tuple(
        registry[name]
        for registry in (
            OPENROUTER_MODEL_EXECUTION_PROFILES,
            OPENROUTER_MODEL_EXECUTION_PROFILE_VARIANTS,
        )
        if name in registry
    )
    if len(matches) != 1:
        raise ValueError(f"unknown OpenRouter model execution profile: {name}")
    return matches[0]


__all__ = [
    "DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH",
    "DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON",
    "GPT_5_6_SOL_OPENAI_XHIGH",
    "GPT_OSS_120B_GROQ_HIGH",
    "GPT_OSS_20B_GROQ_HIGH",
    "GPT_OSS_20B_GROQ_HIGH_SERIAL",
    "MISTRAL_LARGE_3_MISTRAL",
    "OPENROUTER_MODEL_EXECUTION_PROFILES",
    "OPENROUTER_MODEL_EXECUTION_PROFILE_VARIANTS",
    "OpenRouterModelExecutionProfile",
    "QWEN_3_7_MAX_ALIBABA_XHIGH",
    "QWEN_3_7_MAX_ALIBABA_XHIGH_RATE_SAFE",
    "openrouter_model_execution_profile",
]

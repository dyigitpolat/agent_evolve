"""Fail-closed telemetry policy before a successful agentic call can be used.

The provider queue durably publishes terminal metadata first.  This decorator
then verifies the scientific route and per-call resource envelope before an
``AgenticEvolutionEngine`` may materialize or physically evaluate the proposal.
Per-call aggregate ceilings avoid completion-order-dependent selective
acceptance: with a separate hard logical-call cap, their product is a
deterministic run ceiling.  Reasoning usage is part of aggregate output usage;
an independent reasoning ceiling is enforced only when the selected route
actually guarantees one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AgenticGenerator,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
)


class AgenticTelemetryRejected(RuntimeError):
    """A sanitized rejection that never includes prompts or model output."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("rejection reason must be non-empty")
        super().__init__(f"agentic response telemetry rejected: {reason}")
        self.reason = reason


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class AgenticTelemetryPolicy:
    """Exact provider/model identity and one-call token/cost bounds."""

    requested_model: str
    allowed_resolved_models: tuple[str, ...]
    allowed_resolved_providers: tuple[str, ...]
    max_cost_usd: Decimal
    max_input_tokens: int
    max_output_tokens: int
    max_reasoning_tokens: int | None
    max_attempt_count: int

    policy_id = "exact_agentic_telemetry_gate"
    policy_version = 3
    reasoning_token_accounting = "included_in_output_tokens"

    def __post_init__(self) -> None:
        for name in ("requested_model",):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be canonical non-empty text")
        for name in ("allowed_resolved_models", "allowed_resolved_providers"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or any(
                    type(value) is not str or not value or value != value.strip()
                    for value in values
                )
            ):
                raise ValueError(f"{name} must be non-empty canonical text")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} cannot contain duplicates")
        if any("/" not in value for value in self.allowed_resolved_models):
            raise ValueError("allowed_resolved_models must contain model slugs")
        if type(self.max_cost_usd) is not Decimal:
            raise TypeError("max_cost_usd must be an exact Decimal")
        if not self.max_cost_usd.is_finite() or self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be finite and non-negative")
        for name in (
            "max_input_tokens",
            "max_output_tokens",
            "max_attempt_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_reasoning_tokens is not None and (
            type(self.max_reasoning_tokens) is not int
            or self.max_reasoning_tokens < 0
        ):
            raise ValueError(
                "max_reasoning_tokens must be a non-negative integer or None"
            )
        if self.max_attempt_count == 0:
            raise ValueError("max_attempt_count must be positive")

    def to_trace_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "requested_model": self.requested_model,
            "allowed_resolved_models": list(self.allowed_resolved_models),
            "allowed_resolved_providers": list(self.allowed_resolved_providers),
            "max_cost_usd": str(self.max_cost_usd),
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_reasoning_tokens": self.max_reasoning_tokens,
            "reasoning_token_accounting": self.reasoning_token_accounting,
            "max_attempt_count": self.max_attempt_count,
        }

    @property
    def policy_sha256(self) -> str:
        return hashlib.sha256(
            b"agent-evolve:agentic-telemetry-policy:v3\x00"
            + _canonical_json(self.to_trace_record())
        ).hexdigest()

    def validate(self, telemetry: AgenticCallTelemetry) -> None:
        if type(telemetry) is not AgenticCallTelemetry:
            raise TypeError("telemetry must be an exact AgenticCallTelemetry")
        AgenticCallTelemetry.__post_init__(telemetry)
        if telemetry.reasoning_tokens > telemetry.output_tokens:
            raise AgenticTelemetryRejected("reasoning_output_accounting")
        checks = (
            (telemetry.requested_model == self.requested_model, "requested_model"),
            (
                telemetry.resolved_model in self.allowed_resolved_models,
                "resolved_model",
            ),
            (
                telemetry.resolved_provider in self.allowed_resolved_providers,
                "resolved_provider",
            ),
            (telemetry.input_tokens <= self.max_input_tokens, "input_tokens"),
            (telemetry.output_tokens <= self.max_output_tokens, "output_tokens"),
            (telemetry.attempt_count <= self.max_attempt_count, "attempt_count"),
        )
        for accepted, reason in checks:
            if not accepted:
                raise AgenticTelemetryRejected(reason)
        if (
            self.max_reasoning_tokens is not None
            and telemetry.reasoning_tokens > self.max_reasoning_tokens
        ):
            raise AgenticTelemetryRejected("reasoning_tokens")
        cost = telemetry.cost_usd
        if cost is None:
            raise AgenticTelemetryRejected("missing_cost")
        if type(cost) is not Decimal:
            raise AgenticTelemetryRejected("non_decimal_cost")
        if not cost.is_finite() or cost < 0:
            raise AgenticTelemetryRejected("invalid_cost")
        if cost > self.max_cost_usd:
            raise AgenticTelemetryRejected("cost_ceiling")


class TelemetryGatedAgenticGenerator:
    """Transparent generator decorator with a pre-evaluation telemetry gate."""

    def __init__(self, generator: AgenticGenerator, policy: AgenticTelemetryPolicy):
        if not isinstance(generator, AgenticGenerator):
            raise TypeError("generator must implement AgenticGenerator")
        if type(policy) is not AgenticTelemetryPolicy:
            raise TypeError("policy must be an exact AgenticTelemetryPolicy")
        AgenticTelemetryPolicy.__post_init__(policy)
        self.generator = generator
        self.policy = policy

    async def propose(
        self, request: VariationGenerationRequest
    ) -> VariationGenerationResult:
        result = await self.generator.propose(request)
        if type(result) is not VariationGenerationResult:
            raise TypeError("generator returned a non-variation result")
        self.policy.validate(result.telemetry)
        return result

    async def reflect(
        self, request: ReflectionGenerationRequest
    ) -> ReflectionGenerationResult:
        result = await self.generator.reflect(request)
        if type(result) is not ReflectionGenerationResult:
            raise TypeError("generator returned a non-reflection result")
        self.policy.validate(result.telemetry)
        return result


__all__ = [
    "AgenticTelemetryPolicy",
    "AgenticTelemetryRejected",
    "TelemetryGatedAgenticGenerator",
]

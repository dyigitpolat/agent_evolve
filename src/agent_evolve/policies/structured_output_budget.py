"""Immutable structured-output budget policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from agent_evolve.ports.structured_generator import MAX_OUTPUT_TOKENS
from agent_evolve.ports.structured_output_budget import (
    StructuredOutputRequestKind,
    validate_structured_output_operation,
)


def _validate_budget(value: int, name: str) -> None:
    if type(value) is not int or not 1 <= value <= MAX_OUTPUT_TOKENS:
        raise ValueError(f"{name} must lie in [1, {MAX_OUTPUT_TOKENS}]")


@dataclass(frozen=True, slots=True)
class FixedStructuredOutputBudgetPolicy:
    """Assign immutable output limits by agentic request kind.

    The operation argument is deliberately retained by the policy port.  This
    fixed implementation ignores its value after validation; experiments that
    need per-operator allocation can inject another implementation without
    changing the engine or a benchmark adapter.
    """

    proposal_max_output_tokens: int = 2_048
    reflection_max_output_tokens: int = 2_048

    policy_id: ClassVar[str] = "fixed_by_agentic_request_kind"
    policy_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        _validate_budget(
            self.proposal_max_output_tokens,
            "proposal_max_output_tokens",
        )
        _validate_budget(
            self.reflection_max_output_tokens,
            "reflection_max_output_tokens",
        )

    def max_output_tokens(
        self,
        *,
        request_kind: StructuredOutputRequestKind,
        operation: str,
    ) -> int:
        if type(request_kind) is not StructuredOutputRequestKind:
            raise TypeError(
                "request_kind must be an exact StructuredOutputRequestKind"
            )
        validate_structured_output_operation(operation)
        if request_kind is StructuredOutputRequestKind.PROPOSAL:
            return self.proposal_max_output_tokens
        return self.reflection_max_output_tokens


__all__ = ["FixedStructuredOutputBudgetPolicy"]

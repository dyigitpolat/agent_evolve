"""Provider-neutral policy port for structured-output token budgets.

The evolutionary engine emits two materially different typed workloads:
candidate proposals and evidence reflections.  This port lets an experiment
allocate their output budgets independently without teaching the engine about
any benchmark or model provider.  The operation token remains available to
custom policies for finer-grained, benchmark-neutral scheduling.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.ports.structured_generator import MAX_OUTPUT_TOKENS


_POLICY_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPERATION_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")


class StructuredOutputRequestKind(str, Enum):
    """Closed agentic request classes with potentially different workloads."""

    PROPOSAL = "proposal"
    REFLECTION = "reflection"


@runtime_checkable
class StructuredOutputBudgetPolicy(Protocol):
    """Resolve an output-token limit for one typed agentic operation."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> int: ...

    def max_output_tokens(
        self,
        *,
        request_kind: StructuredOutputRequestKind,
        operation: str,
    ) -> int: ...


def validate_structured_output_operation(operation: str) -> None:
    """Validate the operation grammar shared with structured generation."""

    if type(operation) is not str or _OPERATION_TOKEN.fullmatch(operation) is None:
        raise ValueError("operation must use the closed lowercase token grammar")


def structured_output_budget_policy_metadata(
    policy: StructuredOutputBudgetPolicy,
) -> tuple[str, int]:
    """Validate and return stable scientific identity for a budget policy."""

    if not isinstance(policy, StructuredOutputBudgetPolicy):
        raise TypeError(
            "structured_output_budget_policy must implement "
            "StructuredOutputBudgetPolicy"
        )
    policy_id = policy.policy_id
    policy_version = policy.policy_version
    if type(policy_id) is not str or _POLICY_ID.fullmatch(policy_id) is None:
        raise ValueError("structured-output budget policy_id is not canonical")
    if type(policy_version) is not int or policy_version <= 0:
        raise ValueError("structured-output budget policy_version must be positive")
    return policy_id, policy_version


def resolve_structured_output_budget(
    policy: StructuredOutputBudgetPolicy,
    *,
    request_kind: StructuredOutputRequestKind,
    operation: str,
) -> int:
    """Resolve a deterministic, bounded budget without trusting policy state.

    The policy is called twice and its identity is checked around the calls.
    This makes a mutable or time-varying experimental policy fail closed before
    a provider request can escape the application boundary.
    """

    if type(request_kind) is not StructuredOutputRequestKind:
        raise TypeError("request_kind must be an exact StructuredOutputRequestKind")
    validate_structured_output_operation(operation)
    metadata = structured_output_budget_policy_metadata(policy)
    first = policy.max_output_tokens(
        request_kind=request_kind,
        operation=operation,
    )
    second = policy.max_output_tokens(
        request_kind=request_kind,
        operation=operation,
    )
    for value in (first, second):
        if type(value) is not int or not 1 <= value <= MAX_OUTPUT_TOKENS:
            raise ValueError(
                "structured-output max_output_tokens must lie in "
                f"[1, {MAX_OUTPUT_TOKENS}]"
            )
    if first != second:
        raise ValueError("structured-output budget policy must be deterministic")
    if structured_output_budget_policy_metadata(policy) != metadata:
        raise ValueError(
            "structured-output budget policy metadata changed during resolution"
        )
    return first


__all__ = [
    "StructuredOutputBudgetPolicy",
    "StructuredOutputRequestKind",
    "resolve_structured_output_budget",
    "structured_output_budget_policy_metadata",
    "validate_structured_output_operation",
]

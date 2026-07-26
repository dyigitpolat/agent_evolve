"""Domain-neutral separation of scientific and execution call identities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any

from pydantic import BaseModel

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.ports.structured_generator import StructuredGenerationRequest


def _schema_bytes(output_type: type[object]) -> bytes:
    if not issubclass(output_type, BaseModel):
        raise TypeError("structured output type must be a Pydantic BaseModel")
    return json.dumps(
        output_type.model_json_schema(mode="validation"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class StructuredScienceRequestBinding:
    """Fingerprint one scientific request independently of its execution ID."""

    science_call_id: str
    operation: str
    prompt_utf8_bytes: int
    prompt_sha256: str
    output_schema_utf8_bytes: int
    output_schema_sha256: str
    output_tool_name: str
    max_output_tokens: int
    temperature_hex: str | None

    @classmethod
    def from_request(
        cls,
        request: StructuredGenerationRequest[Any],
    ) -> "StructuredScienceRequestBinding":
        if type(request) is not StructuredGenerationRequest:
            raise TypeError("request must be an exact StructuredGenerationRequest")
        StructuredGenerationRequest.__post_init__(request)
        prompt = request.prompt.encode("utf-8", errors="strict")
        schema = _schema_bytes(request.output_type)
        return cls(
            science_call_id=request.call_id.value,
            operation=request.operation,
            prompt_utf8_bytes=len(prompt),
            prompt_sha256=hashlib.sha256(prompt).hexdigest(),
            output_schema_utf8_bytes=len(schema),
            output_schema_sha256=hashlib.sha256(schema).hexdigest(),
            output_tool_name=request.output_tool_name,
            max_output_tokens=request.max_output_tokens,
            temperature_hex=(
                None
                if request.temperature is None
                else float(request.temperature).hex()
            ),
        )

    def provider_fingerprint(self) -> tuple[object, ...]:
        return (
            self.operation,
            self.prompt_utf8_bytes,
            self.prompt_sha256,
            self.output_schema_utf8_bytes,
            self.output_schema_sha256,
            self.output_tool_name,
            self.max_output_tokens,
            self.temperature_hex,
        )


def rebind_structured_execution_request(
    request: StructuredGenerationRequest[Any],
    *,
    expected: StructuredScienceRequestBinding,
    execution_call_id: LLMCallId,
) -> StructuredGenerationRequest[Any]:
    """Replace only queue identity after verifying the scientific payload."""

    if type(expected) is not StructuredScienceRequestBinding:
        raise TypeError("expected must be a StructuredScienceRequestBinding")
    if type(execution_call_id) is not LLMCallId:
        raise TypeError("execution_call_id must be an exact LLMCallId")
    LLMCallId.__post_init__(execution_call_id)
    observed = StructuredScienceRequestBinding.from_request(request)
    if observed != expected:
        raise ValueError("structured scientific request does not match its binding")
    rebound = replace(request, call_id=execution_call_id)
    rebound_binding = StructuredScienceRequestBinding.from_request(rebound)
    if rebound_binding.provider_fingerprint() != expected.provider_fingerprint():
        raise AssertionError("execution-ID rebinding changed provider-visible fields")
    return rebound


@dataclass(slots=True)
class ExecutionIdRebindingRunner:
    """Adapt a frozen scientific request to a fresh queue execution ID."""

    runner: Callable[[StructuredGenerationRequest[Any]], Awaitable[object]]
    expected: StructuredScienceRequestBinding
    execution_call_id: LLMCallId

    def __post_init__(self) -> None:
        if not callable(self.runner):
            raise TypeError("runner must be callable")
        if type(self.expected) is not StructuredScienceRequestBinding:
            raise TypeError("expected must be a StructuredScienceRequestBinding")
        if type(self.execution_call_id) is not LLMCallId:
            raise TypeError("execution_call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.execution_call_id)

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> object:
        rebound = rebind_structured_execution_request(
            request,
            expected=self.expected,
            execution_call_id=self.execution_call_id,
        )
        return await self.runner(rebound)


__all__ = [
    "ExecutionIdRebindingRunner",
    "StructuredScienceRequestBinding",
    "rebind_structured_execution_request",
]

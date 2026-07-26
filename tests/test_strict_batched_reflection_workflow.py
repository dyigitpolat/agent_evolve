from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from agent_evolve.application.reflection_workflow import (
    PlannedReflectionBatchCall,
    ReflectionCardContractError,
    ReflectionPromptShard,
    ReflectionWorkflowRequest,
    StrictBatchedReflectionWorkflow,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    ReflectionGenerationResult,
)


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="batch-response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


def _draft(contrast_id: str, suffix: str) -> InsightDraft:
    return InsightDraft(
        claim=f"Observed action {suffix} changed the objective.",
        trigger=f"Action {suffix} is legal for the current parent.",
        mechanism="The exact measured intervention changed the evaluated structure.",
        affected_paths=(f"$.position_{suffix}",),
        evidence_summary="One exact system-derived intervention contrast.",
        confidence=0.5,
        evidence_contrast_ids=(contrast_id,),
    )


class _BatchGenerator:
    def __init__(self, insights: tuple[InsightDraft, ...]) -> None:
        self.insights = insights
        self.requests = []

    async def propose(self, request):  # pragma: no cover - protocol-only method.
        raise AssertionError("batched reflection must not propose")

    async def reflect(self, request):
        self.requests.append(request)
        return ReflectionGenerationResult(
            insights=self.insights,
            telemetry=_telemetry(),
        )


def _request() -> ReflectionWorkflowRequest:
    contrast_ids = ("a" * 64, "b" * 64)
    return ReflectionWorkflowRequest(
        operation="extract_insights",
        shards=tuple(
            ReflectionPromptShard(contrast_id=item, prompt=f"singleton {item}")
            for item in contrast_ids
        ),
        max_output_tokens=384_000,
        batch_prompt="one exact prompt containing both contrasts",
    )


def test_strict_batch_uses_one_call_and_returns_complete_canonical_cards() -> None:
    request = _request()
    generator = _BatchGenerator(
        (
            _draft("b" * 64, "b"),
            _draft("a" * 64, "a"),
        )
    )
    planned = []

    result = asyncio.run(
        StrictBatchedReflectionWorkflow().run(
            request,
            generator=generator,
            id_factory=DeterministicIdFactory("strict_batch"),
            call_planned_sink=planned.append,
        )
    )

    assert len(generator.requests) == 1
    sent = generator.requests[0]
    assert sent.prompt == request.batch_prompt
    assert sent.min_insights == sent.max_insights == 2
    assert sent.available_contrast_ids == ("a" * 64, "b" * 64)
    assert sent.max_output_tokens == 384_000
    assert len(planned) == 1 and type(planned[0]) is PlannedReflectionBatchCall
    assert result.logical_llm_calls_used == 1
    assert len(result.call_ids) == 1
    assert tuple(item.contrast_id for item in result.shards) == (
        "a" * 64,
        "b" * 64,
    )
    assert len({item.call_id for item in result.shards}) == 1


def test_strict_batch_rejects_duplicate_or_missing_contrast_coverage() -> None:
    generator = _BatchGenerator(
        (
            _draft("a" * 64, "a1"),
            _draft("a" * 64, "a2"),
        )
    )

    with pytest.raises(
        ReflectionCardContractError,
        match="foreign or duplicate contrast coverage",
    ):
        asyncio.run(
            StrictBatchedReflectionWorkflow().run(
                _request(),
                generator=generator,
                id_factory=DeterministicIdFactory("strict_batch_bad"),
            )
        )


def test_strict_batch_requires_the_engine_supplied_combined_prompt() -> None:
    request = _request()
    without_batch = ReflectionWorkflowRequest(
        operation=request.operation,
        shards=request.shards,
        max_output_tokens=request.max_output_tokens,
    )
    generator = _BatchGenerator(
        (_draft("a" * 64, "a"), _draft("b" * 64, "b"))
    )

    with pytest.raises(ValueError, match="requires batch_prompt"):
        asyncio.run(
            StrictBatchedReflectionWorkflow().run(
                without_batch,
                generator=generator,
                id_factory=DeterministicIdFactory("strictbatchnoprompt"),
            )
        )
    assert generator.requests == []

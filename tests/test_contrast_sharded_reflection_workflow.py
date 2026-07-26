from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    MutationContract,
    OperatorKind,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.reflection_workflow import (
    ContrastShardedReflectionWorkflow,
    StrictBatchedReflectionWorkflow,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.patch import JsonPath, ObjectKey
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.feedback.held_out_asn import build_reflected_card_batch
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int
    y: int


class _Problem:
    candidate_model = _Config
    objectives = (ObjectiveSpec("score", "max"),)

    @staticmethod
    def search_space_description() -> str:
        return "Two independent integer coordinates."

    @staticmethod
    def validate(configuration) -> bool:
        _Config.model_validate(configuration, strict=True)
        return True

    @staticmethod
    def evaluate(configuration) -> dict[str, float]:
        return {"score": float(configuration["x"] + configuration["y"])}


def _telemetry(response_id: str) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id=response_id,
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _ShardedGenerator:
    def __init__(self, *, duplicate_claims: bool = False) -> None:
        self.duplicate_claims = duplicate_claims
        self.reflection_requests = []
        self._both_started = asyncio.Event()

    async def propose(self, request):
        parents = json.loads(
            request.prompt.split("PARENTS\n", 1)[1].split(
                "\n\nSELECTED MEMORY", 1
            )[0]
        )
        configuration = copy.deepcopy(parents[0]["configuration"])
        path = "$.x" if '"editable_paths":["$.x"]' in request.prompt else "$.y"
        configuration[path[-1]] += 1
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale=f"Change only {path}.",
                source_attribution=(SourceAttribution(path, "mutation"),),
            ),
            telemetry=_telemetry("proposal"),
        )

    async def reflect(self, request):
        self.reflection_requests.append(request)
        if len(self.reflection_requests) == 2:
            self._both_started.set()
        await asyncio.wait_for(self._both_started.wait(), timeout=1)
        rows = json.loads(
            request.prompt.split("EVALUATED TRACE\n", 1)[1].split(
                "\n\nReturn exactly 1 insight.", 1
            )[0]
        )
        assert len(rows) == 1
        contrast, = rows[0]["machine_derived_contrasts"]
        operation, = contrast["system_derived_operations"]
        path = operation["path"]
        claim = (
            "The same normalized claim."
            if self.duplicate_claims
            else f"Changing {path} improved the observed score."
        )
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim=claim,
                    trigger=f"{path} is editable",
                    mechanism="the observed score increased",
                    affected_paths=(path,),
                    evidence_summary="One exact scalar contrast.",
                    confidence=0.5,
                    evidence_contrast_ids=(contrast["contrast_id"],),
                ),
            ),
            telemetry=_telemetry(path),
        )


class _StrictBatchGenerator(_ShardedGenerator):
    async def reflect(self, request):
        self.reflection_requests.append(request)
        rows = json.loads(
            request.prompt.split("EVALUATED TRACE\n", 1)[1].split(
                "\n\nReturn exactly 2 insights.", 1
            )[0]
        )
        assert len(rows) == 2
        insights = []
        for row in reversed(rows):
            contrast, = row["machine_derived_contrasts"]
            operation, = contrast["system_derived_operations"]
            path = operation["path"]
            insights.append(
                InsightDraft(
                    claim=(
                        "The same normalized claim."
                        if self.duplicate_claims
                        else f"Changing {path} improved the observed score."
                    ),
                    trigger=f"{path} is editable",
                    mechanism="the observed score increased",
                    affected_paths=(path,),
                    evidence_summary="One exact scalar contrast.",
                    confidence=0.5,
                    evidence_contrast_ids=(contrast["contrast_id"],),
                )
            )
        return ReflectionGenerationResult(
            insights=tuple(insights),
            telemetry=_telemetry("strict-batch"),
        )


async def _run(
    *,
    duplicate_claims: bool = False,
    memory_observer: list[InsightMemoryBank] | None = None,
):
    ids = DeterministicIdFactory("sharded_reflection")
    generator = _ShardedGenerator(duplicate_claims=duplicate_claims)
    memory = InsightMemoryBank(id_factory=ids)
    if memory_observer is not None:
        memory_observer.append(memory)
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=_Problem(),
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=1,
        trace_sink=traces.append,
        reflection_workflow=ContrastShardedReflectionWorkflow(),
    )
    first = await engine.register_seed({"x": 1, "y": 10}, label="first")
    outcomes = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (first,),
                generation=1,
                label="x",
                allowed_top_level=("x",),
                mutation_contract=MutationContract(
                    (JsonPath((ObjectKey("x"),)),)
                ),
            ),
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (first,),
                generation=1,
                label="y",
                allowed_top_level=("y",),
                mutation_contract=MutationContract(
                    (JsonPath((ObjectKey("y"),)),)
                ),
            ),
        )
    )
    added = await engine.reflect(outcomes, label="sharded", min_insights=2)
    return added, memory, generator, traces, outcomes


async def _run_batched(*, duplicate_claims: bool = False):
    ids = DeterministicIdFactory("batched_reflection")
    generator = _StrictBatchGenerator(duplicate_claims=duplicate_claims)
    memory = InsightMemoryBank(id_factory=ids)
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=_Problem(),
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=1,
        trace_sink=traces.append,
        reflection_workflow=StrictBatchedReflectionWorkflow(),
    )
    parent = await engine.register_seed({"x": 1, "y": 10}, label="first")
    outcomes = await engine.run_invocations(
        tuple(
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (parent,),
                generation=1,
                label=coordinate,
                allowed_top_level=(coordinate,),
                mutation_contract=MutationContract(
                    (JsonPath((ObjectKey(coordinate),)),)
                ),
            )
            for coordinate in ("x", "y")
        )
    )
    added = await engine.reflect(
        outcomes,
        label="strict-batched",
        min_insights=2,
        max_insights=2,
    )
    return added, memory, generator, traces


def test_sharded_reflection_is_concurrent_deterministic_and_singleton_bound() -> None:
    added, memory, generator, traces, outcomes = asyncio.run(_run())

    assert len(added) == len(memory.entries) == 2
    requests = generator.reflection_requests
    assert len(requests) == 2
    assert [request.call_id.value for request in requests] == [
        "call_sharded_reflection_000003",
        "call_sharded_reflection_000004",
    ]
    contrast_ids = [request.available_contrast_ids[0] for request in requests]
    assert contrast_ids == sorted(contrast_ids)
    assert all(request.min_insights == request.max_insights == 1 for request in requests)
    assert all(
        entry.evidence_lineage is not None
        and entry.evidence_lineage.available_contrast_ids
        == entry.evidence_lineage.cited_contrast_ids
        == entry.draft.evidence_contrast_ids
        and len(entry.evidence_lineage.available_contrast_ids) == 1
        for entry in added
    )
    batch = build_reflected_card_batch(
        outcomes=tuple(
            replace(outcome, reward=reward)
            for outcome, reward in zip(outcomes, (1.0, -1.0), strict=True)
        ),
        entries=added,
        reflection_logical_calls=2,
    )
    assert batch.reflection_logical_calls == 2
    assert [
        event["event_type"] for event in traces if "reflection" in event["event_type"]
    ] == [
        "reflection_requested",
        "reflection_requested",
        "reflection_completed",
        "reflection_completed",
        "reflection_batch_completed",
    ]


def test_sharded_reflection_duplicate_batch_is_not_partially_published() -> None:
    async def scenario():
        observed: list[InsightMemoryBank] = []
        with pytest.raises(ValueError, match="duplicate normalized claims"):
            await _run(duplicate_claims=True, memory_observer=observed)
        assert observed[0].entries == ()

    asyncio.run(scenario())


def test_engine_strict_batch_publishes_two_cards_from_one_logical_call() -> None:
    added, memory, generator, traces = asyncio.run(_run_batched())

    assert len(generator.reflection_requests) == 1
    request = generator.reflection_requests[0]
    assert request.min_insights == request.max_insights == 2
    assert len(added) == len(memory.entries) == 2
    assert len({entry.evidence_lineage.reflection_call_id for entry in added}) == 1
    batch_event, = (
        event
        for event in traces
        if event["event_type"] == "reflection_batch_completed"
    )
    assert batch_event["logical_llm_calls_used"] == 1
    assert len(batch_event["call_ids"]) == 1
    assert len(batch_event["contrast_ids"]) == 2


def test_engine_strict_batch_rejects_before_any_memory_publication() -> None:
    async def scenario():
        ids = DeterministicIdFactory("batchatomic")
        generator = _StrictBatchGenerator(duplicate_claims=True)
        memory = InsightMemoryBank(id_factory=ids)
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=generator,
            id_factory=ids,
            memory=memory,
            seed=1,
            reflection_workflow=StrictBatchedReflectionWorkflow(),
        )
        parent = await engine.register_seed({"x": 1, "y": 10}, label="first")
        outcomes = await engine.run_invocations(
            tuple(
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (parent,),
                    generation=1,
                    label=coordinate,
                    allowed_top_level=(coordinate,),
                    mutation_contract=MutationContract(
                        (JsonPath((ObjectKey(coordinate),)),)
                    ),
                )
                for coordinate in ("x", "y")
            )
        )
        with pytest.raises(ValueError, match="duplicate normalized claims"):
            await engine.reflect(
                outcomes,
                label="strict-batched",
                min_insights=2,
                max_insights=2,
            )
        assert memory.entries == ()

    asyncio.run(scenario())

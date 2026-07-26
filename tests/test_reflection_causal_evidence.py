from __future__ import annotations

import asyncio
import copy
import json
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    MutationContract,
    OperatorKind,
    ReflectionCallExecutionError,
    ReflectionCallStatus,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.patch import JsonPath, ObjectKey
from agent_evolve.domain.typed_json import typed_json_sha256
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
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


class _CausalGenerator:
    async def propose(self, request):
        parents = json.loads(
            request.prompt.split("PARENTS\n", 1)[1].split(
                "\n\nSELECTED MEMORY", 1
            )[0]
        )
        configuration = copy.deepcopy(parents[0]["configuration"])
        if '"editable_paths":["$.x"]' in request.prompt:
            configuration["x"] += 1
            path = "$.x"
        else:
            assert '"editable_paths":["$.y"]' in request.prompt
            configuration["y"] += 2
            path = "$.y"
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale=f"Change only {path}.",
                source_attribution=(SourceAttribution(path, "mutation"),),
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        rows = json.loads(
            request.prompt.split("EVALUATED TRACE\n", 1)[1].split(
                "\n\nReturn at most", 1
            )[0]
        )
        contrasts_by_path = {}
        for row in rows:
            for contrast in row["machine_derived_contrasts"]:
                operation, = contrast["system_derived_operations"]
                contrasts_by_path[operation["path"]] = contrast["contrast_id"]
        foreign = "f" * 64
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim="Incrementing x improved the observed score.",
                    trigger="x is locally editable",
                    mechanism="the score increases with x",
                    affected_paths=("$.x",),
                    evidence_summary="One exact scalar contrast.",
                    confidence=0.6,
                    evidence_contrast_ids=(contrasts_by_path["$.x"],),
                ),
                InsightDraft(
                    claim="Incrementing y improved the observed score.",
                    trigger="y is locally editable",
                    mechanism="the score increases with y",
                    affected_paths=("$.y",),
                    evidence_summary="One exact scalar contrast.",
                    confidence=0.6,
                    evidence_contrast_ids=(contrasts_by_path["$.y"],),
                ),
                InsightDraft(
                    claim="This unsupported insight must not enter memory.",
                    trigger="a foreign citation is supplied",
                    mechanism="none observed",
                    affected_paths=("$.x",),
                    evidence_summary="Foreign evidence only.",
                    confidence=0.1,
                    evidence_contrast_ids=(foreign,),
                ),
            ),
            telemetry=_telemetry(),
        )


def test_reflection_projects_exact_scalar_operations_and_narrows_lineage_by_citation() -> None:
    async def scenario():
        ids = DeterministicIdFactory("causal_reflection")
        traces: list[dict[str, object]] = []
        memory = InsightMemoryBank(id_factory=ids)
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=_CausalGenerator(),
            id_factory=ids,
            memory=memory,
            seed=1,
            trace_sink=traces.append,
        )
        parent_x = await engine.register_seed({"x": 1, "y": 10}, label="px")
        parent_y = await engine.register_seed({"x": 100, "y": 20}, label="py")
        outcomes = await engine.run_invocations(
            (
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (parent_x,),
                    generation=1,
                    label="x",
                    allowed_top_level=("x",),
                    mutation_contract=MutationContract(
                        (JsonPath((ObjectKey("x"),)),)
                    ),
                ),
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (parent_y,),
                    generation=1,
                    label="y",
                    allowed_top_level=("y",),
                    mutation_contract=MutationContract(
                        (JsonPath((ObjectKey("y"),)),)
                    ),
                ),
            )
        )
        added = await engine.reflect(outcomes, label="causal", max_insights=3)
        return parent_x, parent_y, outcomes, added, traces, memory

    parent_x, parent_y, outcomes, added, traces, memory = asyncio.run(scenario())
    assert len(added) == 2
    assert len(memory.entries) == 2

    reflection_event = next(
        event for event in traces if event["event_type"] == "reflection_requested"
    )
    rows = json.loads(
        reflection_event["prompt"].split("EVALUATED TRACE\n", 1)[1].split(
            "\n\nReturn at most", 1
        )[0]
    )
    projections = {
        operation["path"]: operation
        for row in rows
        for contrast in row["machine_derived_contrasts"]
        for operation in contrast["system_derived_operations"]
    }
    assert projections["$.x"] == {
        "operation_kind": "replace_scalar",
        "path": "$.x",
        "old_value": 1,
        "new_value": 2,
        "old_value_hash": typed_json_sha256(1),
        "new_value_hash": typed_json_sha256(2),
    }
    assert projections["$.y"] == {
        "operation_kind": "replace_scalar",
        "path": "$.y",
        "old_value": 20,
        "new_value": 22,
        "old_value_hash": typed_json_sha256(20),
        "new_value_hash": typed_json_sha256(22),
    }
    contrasts = {
        contrast["system_derived_operations"][0]["path"]: contrast
        for row in rows
        for contrast in row["machine_derived_contrasts"]
    }
    assert contrasts["$.x"]["parent_configuration_hash"] == (
        parent_x.occurrence.configuration_hash
    )
    assert contrasts["$.x"]["child_configuration_hash"] == (
        outcomes[0].candidate.occurrence.configuration_hash
    )
    assert contrasts["$.x"]["derived_patch_hash"] == derive_patch(
        parent_x.configuration,
        outcomes[0].candidate.configuration,
        base_candidate_id=parent_x.candidate_id,
        target_candidate_id=outcomes[0].candidate.candidate_id,
    ).patch_hash

    by_path = {entry.draft.affected_paths[0]: entry for entry in added}
    x_lineage = by_path["$.x"].evidence_lineage
    y_lineage = by_path["$.y"].evidence_lineage
    assert x_lineage is not None and y_lineage is not None
    assert x_lineage.source_operator_invocation_ids == (
        outcomes[0].prepared.operator_invocation_id,
    )
    assert y_lineage.source_operator_invocation_ids == (
        outcomes[1].prepared.operator_invocation_id,
    )
    assert set(x_lineage.source_candidate_ids) == {
        parent_x.candidate_id,
        outcomes[0].candidate.candidate_id,
    }
    assert set(y_lineage.source_candidate_ids) == {
        parent_y.candidate_id,
        outcomes[1].candidate.candidate_id,
    }
    assert set(x_lineage.source_candidate_ids).isdisjoint(
        y_lineage.source_candidate_ids
    )
    assert x_lineage.available_contrast_ids == y_lineage.available_contrast_ids
    assert len(x_lineage.available_contrast_ids) == 2
    assert len(x_lineage.cited_contrast_ids) == 1
    assert len(y_lineage.cited_contrast_ids) == 1

    filtered = next(
        event
        for event in traces
        if event["event_type"] == "reflection_evidence_contrast_ids_filtered"
    )
    assert filtered["rejected_contrast_ids"] == ["f" * 64]
    rejected = next(
        event
        for event in traces
        if event["event_type"] == "reflection_insight_rejected"
    )
    assert rejected["reason"] == "no_accepted_evidence_contrast_ids"
    assert rejected["submitted_contrast_ids"] == ["f" * 64]


def test_nonempty_all_rejected_reflection_is_failure_not_abstention() -> None:
    class AllRejectedGenerator(_CausalGenerator):
        async def reflect(self, request):
            del request
            return ReflectionGenerationResult(
                insights=(
                    InsightDraft(
                        claim="A foreign-only card must not resemble abstention.",
                        trigger="a foreign citation is supplied",
                        mechanism="no admissible evidence supports this card",
                        affected_paths=("$.x",),
                        evidence_summary="Foreign evidence only.",
                        confidence=0.1,
                        evidence_contrast_ids=("f" * 64,),
                    ),
                ),
                telemetry=_telemetry(),
            )

    async def scenario():
        ids = DeterministicIdFactory("all_rejected_reflection")
        traces: list[dict[str, object]] = []
        memory = InsightMemoryBank(id_factory=ids)
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=AllRejectedGenerator(),
            id_factory=ids,
            memory=memory,
            seed=1,
            trace_sink=traces.append,
        )
        parent = await engine.register_seed({"x": 1, "y": 10}, label="parent")
        outcomes = await engine.run_invocations(
            (
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (parent,),
                    generation=1,
                    label="x",
                    allowed_top_level=("x",),
                    mutation_contract=MutationContract(
                        (JsonPath((ObjectKey("x"),)),)
                    ),
                ),
            )
        )
        with pytest.raises(ReflectionCallExecutionError) as caught:
            await engine.reflect_with_receipt(
                outcomes,
                label="all_rejected",
                min_insights=0,
                max_insights=1,
            )
        return caught.value, traces, memory, engine

    error, traces, memory, engine = asyncio.run(scenario())
    assert error.failure_type == "ReflectionCardContractError"
    assert error.receipt.status is ReflectionCallStatus.FAILED
    assert error.receipt.telemetry is not None
    assert error.receipt.publications == ()
    assert memory.entries == ()
    assert engine.reflection_call_receipts == (error.receipt,)
    failed = next(
        event for event in traces if event["event_type"] == "reflection_failed"
    )
    assert failed["reason"] == "all_submitted_insights_rejected"
    assert failed["submitted_insight_count"] == 1
    assert failed["rejected_insight_count"] == 1


def test_empty_reflection_remains_completed_abstention() -> None:
    class AbstainingGenerator(_CausalGenerator):
        async def reflect(self, request):
            del request
            return ReflectionGenerationResult(insights=(), telemetry=_telemetry())

    async def scenario():
        ids = DeterministicIdFactory("empty_reflection")
        memory = InsightMemoryBank(id_factory=ids)
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=AbstainingGenerator(),
            id_factory=ids,
            memory=memory,
            seed=1,
        )
        result = await engine.reflect_with_receipt(
            (),
            label="true_abstention",
            min_insights=0,
            max_insights=1,
        )
        return result, memory, engine

    result, memory, engine = asyncio.run(scenario())
    assert result.entries == ()
    assert result.receipt.status is ReflectionCallStatus.COMPLETED
    assert result.receipt.publications == ()
    assert memory.entries == ()
    assert engine.reflection_call_receipts == (result.receipt,)

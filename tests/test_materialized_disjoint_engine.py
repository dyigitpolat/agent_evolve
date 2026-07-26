from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    OperatorKind,
    ProposalAuthority,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchRecombiner,
)


class _NoCallGenerator:
    """Fail closed if an engine-owned materialization reaches an LLM port."""

    def __init__(self) -> None:
        self.requests: list[object] = []

    async def propose(self, request):
        self.requests.append(request)
        raise AssertionError("materialized variation must not invoke propose")

    async def reflect(self, request):
        self.requests.append(request)
        raise AssertionError("materialized variation must not invoke reflect")


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    left: int
    right: int
    same: int


class _Problem:
    candidate_model = _Config
    objectives = (ObjectiveSpec("sum", "min"), ObjectiveSpec("spread", "min"))

    @staticmethod
    def search_space_description() -> str:
        return "Three integer coordinates for deterministic recombination."

    @staticmethod
    def validate(configuration: object) -> bool:
        _Config.model_validate(configuration, strict=True)
        return True

    @staticmethod
    def evaluate(configuration: dict[str, object]) -> dict[str, float]:
        values = [int(configuration[name]) for name in ("left", "right", "same")]
        return {
            "sum": float(sum(values)),
            "spread": float(max(values) - min(values)),
        }


def test_disjoint_policy_materialization_enters_normal_engine_lineage_without_llm() -> (
    None
):
    async def scenario():
        ids = DeterministicIdFactory("materialized_disjoint_engine")
        traces: list[dict[str, object]] = []
        generator = _NoCallGenerator()
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=1,
            trace_sink=traces.append,
        )
        ancestor = await engine.register_seed(
            {"left": 0, "right": 0, "same": 0}, label="ancestor"
        )
        left = await engine.register_seed(
            {"left": 3, "right": 0, "same": 1}, label="left"
        )
        right = await engine.register_seed(
            {"left": 0, "right": 4, "same": 1}, label="right"
        )
        target_id = ids.new_candidate_id()
        materialization = DisjointPatchRecombiner().materialize(
            ancestor=ancestor.configuration,
            ancestor_candidate_id=ancestor.candidate_id,
            left=left.configuration,
            left_candidate_id=left.candidate_id,
            right=right.configuration,
            right_candidate_id=right.candidate_id,
            target_candidate_id=target_id,
        )
        item = materialized_disjoint_invocation(
            plan=InvocationPlan(
                OperatorKind.THREE_WAY_RECOMBINATION,
                (left, right),
                generation=1,
                label="engine_union",
                common_ancestor=ancestor,
            ),
            materialization=materialization,
        )
        (outcome,) = await engine.run_materialized_invocations((item,))
        return generator, traces, materialization, outcome

    generator, traces, materialization, outcome = asyncio.run(scenario())
    candidate = outcome.candidate
    assert candidate is not None
    assert candidate.candidate_id == materialization.union_patch.target_candidate_id
    assert candidate.configuration == materialization.configuration
    assert candidate.valid and candidate.operator_compliant
    assert candidate.preservation_verified is True
    assert candidate.evidence_compliant is True
    assert candidate.call_telemetry is None
    assert outcome.prepared.call_id is None
    assert outcome.prepared.proposal_authority is ProposalAuthority.ENGINE
    assert outcome.prepared.materialized_finite_action_authority is None
    assert outcome.prepared.materialized_finite_action_decision is None
    assert generator.requests == []
    assert not any(event["event_type"].startswith("llm_call") for event in traces)
    evaluated = next(
        event for event in traces if event["event_type"] == "candidate_evaluated"
    )
    assert evaluated["source_attribution_provenance"] == "engine_materialized"
    completed = next(
        event for event in traces if event["event_type"] == "invocation_completed"
    )
    assert completed["proposal_authority"] == "engine"
    assert completed["materialization_policy_id"] == "disjoint_patch_union"
    assert completed["materialization_receipt_hash"] == (materialization.receipt_sha256)
    assert completed["materialized_finite_action_authority"] is None
    assert completed["materialized_finite_action_decision"] is None


def test_adapter_rejects_matching_occurrence_id_with_different_configuration() -> None:
    async def scenario():
        ids = DeterministicIdFactory("materialized_disjoint_endpoint_hash")
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=_NoCallGenerator(),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=1,
        )
        ancestor = await engine.register_seed(
            {"left": 0, "right": 0, "same": 0}, label="ancestor"
        )
        left = await engine.register_seed(
            {"left": 3, "right": 0, "same": 1}, label="left"
        )
        right = await engine.register_seed(
            {"left": 0, "right": 4, "same": 1}, label="right"
        )
        materialization = DisjointPatchRecombiner().materialize(
            ancestor=ancestor.configuration,
            ancestor_candidate_id=ancestor.candidate_id,
            left=left.configuration,
            left_candidate_id=left.candidate_id,
            right=right.configuration,
            right_candidate_id=right.candidate_id,
            target_candidate_id=ids.new_candidate_id(),
        )
        impostor = await engine.register_seed(
            {"left": 9, "right": 0, "same": 2}, label="impostor"
        )
        forged_left = replace(
            impostor,
            occurrence=replace(
                impostor.occurrence,
                candidate_id=left.candidate_id,
            ),
        )
        return materialization, InvocationPlan(
            OperatorKind.THREE_WAY_RECOMBINATION,
            (forged_left, right),
            generation=1,
            label="forged_endpoint",
            common_ancestor=ancestor,
        )

    materialization, plan = asyncio.run(scenario())
    with pytest.raises(ValueError, match="endpoint configurations differ"):
        materialized_disjoint_invocation(
            plan=plan,
            materialization=materialization,
        )

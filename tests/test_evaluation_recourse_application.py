from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    OperatorKind,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    FrozenWaveReward,
    GenerationPlan,
    OptimizerBudget,
    OptimizerContractError,
    OptimizerSlot,
)
from agent_evolve.application.evaluation_recourse import (
    EvaluationRecourseApplicationService,
    phenotype_ledger_from_generation,
    phenotype_occurrence,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.phenotype_recourse import (
    BoundedEvaluationRecoursePolicy,
    EvaluationOccurrenceRole,
    EvaluationOccurrenceStatus,
    PresealedRecoursePool,
    RecourseBudgetSnapshot,
    RecoursePoolCandidate,
    TypedConfigurationPhenotypeIdentityPolicy,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int
    y: int


class _Problem:
    candidate_model = _Config
    objectives = (ObjectiveSpec("x", "min"), ObjectiveSpec("y", "min"))

    @staticmethod
    def search_space_description() -> str:
        return "Minimize two bounded integer coordinates."

    @staticmethod
    def validate(configuration: object) -> bool:
        value = _Config.model_validate(configuration, strict=True)
        return 0 <= value.x <= 4 and 0 <= value.y <= 4

    @staticmethod
    def evaluate(configuration: dict[str, object]) -> dict[str, float]:
        return {"x": float(configuration["x"]), "y": float(configuration["y"])}


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="duplicate-response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class _DuplicateGenerator:
    async def propose(self, request):
        del request
        await asyncio.sleep(0)
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration={"x": 1, "y": 2},
                design_rationale="Independent model occurrence with same phenotype.",
                intended_changes=("$.x", "$.y"),
                source_attribution=(
                    SourceAttribution("$.x", "mutation"),
                    SourceAttribution("$.y", "mutation"),
                ),
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


class _DuplicatePlanner:
    def __init__(self, engine: AgenticEvolutionEngine) -> None:
        self.engine = engine

    def plan(self, state, budget):
        del budget
        parent = state.candidates[0]
        plans = tuple(
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (parent,),
                generation=1,
                label=f"primary_{index}",
                allowed_top_level=("x", "y"),
                phase="phenotype_recourse_primary",
            )
            for index in range(2)
        )
        return GenerationPlan(
            generation=1,
            slots=tuple(
                OptimizerSlot.model(
                    slot_id=f"P{index}",
                    role="primary_model",
                    plan=plan,
                )
                for index, plan in enumerate(plans)
            ),
            reward=FrozenWaveReward(
                binding=self.engine.reward_binding,
                archive_snapshot_hash=state.archive_snapshot_hash,
                reward_snapshot_hash=_hash(
                    f"recourse-test:{state.archive_snapshot_hash}"
                ),
            ),
            planner_policy_id="duplicate_recourse_application_test",
            planner_policy_version=1,
        )


def test_duplicate_occurrences_create_objective_blind_bounded_recourse() -> None:
    async def scenario():
        ids = DeterministicIdFactory("recourse_application")
        problem = _Problem()
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=_DuplicateGenerator(),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=4,
        )
        optimizer = BudgetedAgenticOptimizer(
            engine=engine,
            archive=ParetoArchive(problem.objectives),
            planner=_DuplicatePlanner(engine),
            budget=OptimizerBudget(4, 2, 1),
        )
        result = await optimizer.run(({"x": 3, "y": 3},))
        return engine, result

    engine, result = asyncio.run(scenario())
    receipt = result.generation_receipts[0]
    identity_policy = TypedConfigurationPhenotypeIdentityPolicy()
    pool = PresealedRecoursePool.seal(
        pool_id="recourse.pool.1",
        seal_context_sha256=_hash("preoutcome-primary-plan"),
        candidates=(
            RecoursePoolCandidate.freeze("coverage_x", {"x": 0, "y": 3}),
            RecoursePoolCandidate.freeze("coverage_y", {"x": 3, "y": 0}),
        ),
        identity_policy=identity_policy,
    )
    traces: list[dict[str, object]] = []
    service = EvaluationRecourseApplicationService(
        identity_policy=identity_policy,
        recourse_policy=BoundedEvaluationRecoursePolicy(max_recourse=2),
        trace_sink=traces.append,
    )
    decision = service.decide(
        primary_receipt=receipt,
        pool=pool,
        budget=RecourseBudgetSnapshot(
            max_unique_evaluations=4,
            used_unique_evaluations=result.final_state.unique_evaluations,
            reserved_non_recourse_evaluations=0,
            protected_recombination_evaluations=1,
        ),
    )

    ledger = decision.ledger
    assert len(ledger.primary_occurrences) == 2
    assert len({item.candidate_id for item in ledger.primary_occurrences}) == 2
    assert len(ledger.clusters) == 1
    assert ledger.successful_primary_collision_credit == 1
    assert decision.slots == 1
    assert decision.selected_entry_ids == ("coverage_x",)
    assert result.final_state.unique_evaluations == 2
    cache = asyncio.run(engine.evaluation_cache_snapshot())
    assert cache["misses"] == 2  # one seed plus one shared primary phenotype
    assert cache["coalesced"] == 1
    collision = next(
        value for value in traces if value["event_type"] == "phenotype_collision"
    )
    assert collision["collision_credit"] == 1
    decision_trace = next(
        value
        for value in traces
        if value["event_type"] == "evaluation_recourse_decided"
    )
    assert "reward" not in repr(decision_trace).lower()
    assert "objective" not in repr(decision_trace).lower()


def test_generation_projection_can_select_an_explicit_primary_slot_subset() -> None:
    async def scenario():
        ids = DeterministicIdFactory("recourse_application_subset")
        problem = _Problem()
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=_DuplicateGenerator(),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=5,
        )
        result = await BudgetedAgenticOptimizer(
            engine=engine,
            archive=ParetoArchive(problem.objectives),
            planner=_DuplicatePlanner(engine),
            budget=OptimizerBudget(4, 2, 1),
        ).run(({"x": 3, "y": 3},))
        return result

    result = asyncio.run(scenario())
    ledger = phenotype_ledger_from_generation(
        result.generation_receipts[0],
        role=EvaluationOccurrenceRole.PRIMARY,
        identity_policy=TypedConfigurationPhenotypeIdentityPolicy(),
        included_slot_ids=("P1",),
    )

    assert len(ledger.occurrences) == 1
    assert ledger.successful_primary_collision_credit == 0


def test_infrastructure_outcome_invalidates_recourse_and_stale_receipt_is_rejected() -> (
    None
):
    async def scenario():
        ids = DeterministicIdFactory("recourse_application_infrastructure")
        problem = _Problem()
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=_DuplicateGenerator(),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=6,
        )
        return await BudgetedAgenticOptimizer(
            engine=engine,
            archive=ParetoArchive(problem.objectives),
            planner=_DuplicatePlanner(engine),
            budget=OptimizerBudget(4, 2, 1),
        ).run(({"x": 3, "y": 3},))

    result = asyncio.run(scenario())
    receipt = result.generation_receipts[0]
    successful = receipt.slot_results[0]
    infrastructure = replace(
        successful.outcome,
        candidate=None,
        reward=-1.0,
        call_failure_type="TimeoutError",
        failure_stage="infrastructure",
        dominates_any_parent=False,
        better_than_any_parent=False,
    )
    occurrence = phenotype_occurrence(
        infrastructure,
        role=EvaluationOccurrenceRole.PRIMARY,
        identity_policy=TypedConfigurationPhenotypeIdentityPolicy(),
    )
    assert occurrence.status is EvaluationOccurrenceStatus.INFRASTRUCTURE_FAILURE

    stale_hash_tamper = replace(
        receipt,
        slot_results=(
            replace(successful, outcome=infrastructure),
            *receipt.slot_results[1:],
        ),
    )
    with pytest.raises(OptimizerContractError, match="does not authenticate"):
        phenotype_ledger_from_generation(
            stale_hash_tamper,
            role=EvaluationOccurrenceRole.PRIMARY,
            identity_policy=TypedConfigurationPhenotypeIdentityPolicy(),
        )

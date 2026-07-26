"""Focused provider-free tests for generic terminal reflection."""

from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    MaterializedInvocation,
    OperatorKind,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    FrozenWaveReward,
    GenerationPlan,
    OptimizerBudget,
    OptimizerSlot,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightOrigin,
)
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.application.post_evolution_reflection import (
    PostEvolutionReflectionFactory,
    PostEvolutionReflectionInterceptor,
    PostEvolutionReflectionSource,
    PostEvolutionReflectionSourceScope,
    PostEvolutionReflectionSpec,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationResult,
    ReflectionInsightContract,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int


class _Problem:
    candidate_model = _Config
    objectives = (ObjectiveSpec("x", "min"),)

    @staticmethod
    def search_space_description() -> str:
        return "Minimize one bounded integer coordinate."

    @staticmethod
    def validate(configuration: object) -> bool:
        candidate = _Config.model_validate(configuration, strict=True)
        return 0 <= candidate.x <= 9

    @staticmethod
    def evaluate(configuration: dict[str, object]) -> dict[str, float]:
        return {"x": float(configuration["x"])}


def _telemetry(index: int) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="provider-free",
        provider_response_id=f"reflection-{index}",
        finish_reason="stop",
        input_tokens=20,
        output_tokens=10,
        reasoning_tokens=3,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=index + 1,
        attempt_count=1,
    )


class _ReflectionGenerator:
    def __init__(self, *, fail: bool = False, abstain: bool = False) -> None:
        self.fail = fail
        self.abstain = abstain
        self.proposal_requests = []
        self.reflection_requests = []

    async def propose(self, request):
        self.proposal_requests.append(request)
        raise AssertionError("the test planner materializes every variation")

    async def reflect(self, request):
        self.reflection_requests.append(request)
        if self.fail:
            raise RuntimeError("injected terminal reflection failure")
        if self.abstain:
            insights = ()
        else:
            insights = (
                InsightDraft(
                    claim="A smaller coordinate improved both sealed descendants.",
                    trigger="When a parent has positive x.",
                    mechanism="Reducing x directly lowers the benchmark objective.",
                    affected_paths=("$.x",),
                    evidence_summary="Both selected terminal-ledger contrasts reduced x.",
                    confidence=0.8,
                    evidence_contrast_ids=request.available_contrast_ids,
                    effect_predictions=(
                        MetricEffectPrediction(
                            metric_id="x",
                            direction=MetricEffectDirection.DECREASE,
                        ),
                    ),
                    recommended_option_families=("coordinate",),
                    action_template="Decrease x while preserving all other fields.",
                    falsification_condition="A lower x fails to lower objective x.",
                ),
            )
        return ReflectionGenerationResult(
            insights=insights,
            telemetry=_telemetry(len(self.reflection_requests)),
        )


def _reward(state_hash: str) -> FrozenWaveReward:
    def score(child, parents, objectives):
        del objectives
        if not child.valid or not child.operator_compliant:
            return -1.0
        return float(parents[0].objective_map["x"] - child.objective_map["x"])

    return FrozenWaveReward(
        binding=RewardPolicyBinding(
            score,
            _hash("post-evolution-reflection-test-reward-v1"),
        ),
        archive_snapshot_hash=state_hash,
        reward_snapshot_hash=_hash(f"reward-snapshot:{state_hash}"),
    )


class _Planner:
    """Reuse a slot name while freezing memory selection only at G2."""

    def __init__(
        self,
        *,
        benchmark,
        engine: AgenticEvolutionEngine,
        ids: DeterministicIdFactory,
        memory: InsightMemoryBank,
        predecessor,
    ) -> None:
        self.benchmark = benchmark
        self.engine = engine
        self.ids = ids
        self.memory = memory
        self.predecessor = predecessor
        self.selected_predecessor = None

    def plan(self, state, budget):
        del budget
        generation = state.generation + 1
        if generation == 2:
            # This models a benchmark planner sealing its terminal memory
            # assignment before feedback reservation is requested.
            self.selected_predecessor = self.predecessor
        parent = state.candidates[-1]
        plan = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=generation,
            label=f"g{generation}_materialized_mutation",
            allowed_top_level=("x",),
        )
        materialized = MaterializedInvocation(
            plan=plan,
            draft=CandidateDraft(
                configuration={"x": 5 - generation},
                design_rationale="Deterministic provider-free mutation.",
                intended_changes=("$.x",),
            ),
            candidate_id=self.ids.new_candidate_id(),
            materialization_policy_id="terminal_reflection_test_materializer",
            materialization_policy_version=1,
            materialization_receipt_hash=_hash(f"materialized-g{generation}"),
        )
        return GenerationPlan(
            generation=generation,
            slots=(
                OptimizerSlot.engine(
                    slot_id="mutant",
                    role="deterministic_descent",
                    invocation=materialized,
                ),
            ),
            reward=_reward(state.archive_snapshot_hash),
            planner_policy_id="generic_terminal_reflection_test_planner",
            planner_policy_version=1,
        )


class _Resolver:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, planner):
        self.calls += 1
        if planner.selected_predecessor is None:
            raise AssertionError(
                "predecessor resolved before terminal memory selection"
            )
        return planner.selected_predecessor


def _seed_insight() -> InsightDraft:
    return InsightDraft(
        claim="Smaller x may improve the objective.",
        trigger="When x is positive.",
        mechanism="The objective directly reports x.",
        affected_paths=("$.x",),
        evidence_summary="Seed hypothesis awaiting evolutionary evidence.",
        confidence=0.4,
    )


def _spec() -> PostEvolutionReflectionSpec:
    return PostEvolutionReflectionSpec(
        terminal_generation=2,
        source_scope=PostEvolutionReflectionSourceScope(
            sources=(
                PostEvolutionReflectionSource(1, "mutant"),
                PostEvolutionReflectionSource(2, "mutant"),
            )
        ),
        insight_contract=ReflectionInsightContract(
            required_metric_ids=("x",),
            allowed_option_families=("coordinate",),
        ),
    )


def _run(*, fail: bool = False, abstain: bool = False):
    ids = DeterministicIdFactory(
        f"post_evolution_reflection_{int(fail)}_{int(abstain)}"
    )
    memory = InsightMemoryBank(id_factory=ids)
    predecessor_entry, added = memory.add(_seed_insight())
    assert added
    problem = _Problem()
    benchmark = SimpleNamespace(problem=problem)
    generator = _ReflectionGenerator(fail=fail, abstain=abstain)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=7,
    )
    planner = _Planner(
        benchmark=benchmark,
        engine=engine,
        ids=ids,
        memory=memory,
        predecessor=predecessor_entry.reference,
    )
    resolver = _Resolver()
    factory = PostEvolutionReflectionFactory(
        spec=_spec(),
        predecessor_resolver=resolver,
    )
    interceptor = factory.build(
        benchmark=benchmark,
        engine=engine,
        id_factory=ids,
        memory=memory,
        planner=planner,
    )
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=ParetoArchive(
            problem.objectives,
            evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
        ),
        planner=planner,
        budget=OptimizerBudget(
            max_unique_evaluations=3,
            max_logical_llm_calls=1,
            max_generations=2,
        ),
        feedback_interceptor=interceptor,
    )
    result = asyncio.run(optimizer.run(({"x": 5},)))
    return SimpleNamespace(
        result=result,
        ids=ids,
        memory=memory,
        problem=problem,
        benchmark=benchmark,
        generator=generator,
        engine=engine,
        planner=planner,
        resolver=resolver,
        factory=factory,
        interceptor=interceptor,
        predecessor=predecessor_entry,
    )


def test_terminal_reflection_is_one_receipt_bound_revision_after_memory_selection() -> (
    None
):
    run = _run()
    result = run.result
    interceptor = run.interceptor

    assert type(interceptor) is PostEvolutionReflectionInterceptor
    assert run.resolver.calls == 1
    assert interceptor.invoked_generations == [1, 2]
    assert [
        receipt.reserved_logical_llm_calls for receipt in result.feedback_receipts
    ] == [0, 1]
    assert [receipt.used_logical_llm_calls for receipt in result.feedback_receipts] == [
        0,
        1,
    ]
    assert result.final_state.logical_llm_calls == 1
    assert run.generator.proposal_requests == []
    assert len(run.generator.reflection_requests) == 1

    request = run.generator.reflection_requests[0]
    assert request.min_insights == 0
    assert request.max_insights == 1
    assert request.insight_contract is interceptor.spec.insight_contract
    assert len(request.available_contrast_ids) == 2

    assert interceptor.reflection_authority is not None
    assert interceptor.reflection_receipt is not None
    assert interceptor.reflection_result is not None
    authority = interceptor.reflection_authority
    receipt = interceptor.reflection_receipt
    call_request = receipt.call_receipt.request
    assert authority.sources == interceptor.spec.source_scope.sources
    assert authority.generation_receipt_sha256s == tuple(
        value.receipt_hash for value in result.generation_receipts
    )
    assert call_request.source_receipt_sha256s == authority.generation_receipt_sha256s
    assert call_request.source_operator_invocation_ids == tuple(
        generation.slot_results[0].outcome.prepared.operator_invocation_id
        for generation in result.generation_receipts
    )
    assert receipt.reflection_status == "sealed_complete"
    assert receipt.publication_outcome == "completed_revision"

    assert len(interceptor.reflected_entries) == 1
    revision = interceptor.reflected_entries[0]
    assert revision.reference.insight_id == run.predecessor.reference.insight_id
    assert revision.reference.version == run.predecessor.reference.version + 1
    assert revision.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert revision.origin is InsightOrigin.REFLECTION
    assert revision.initial_score == 0.0
    assert interceptor.reflection_result.entries == (revision,)
    assert len(run.memory.entries) == 2

    metadata = dict(result.feedback_receipts[-1].result_metadata)
    assert metadata["reflection_status"] == "sealed_complete"
    assert metadata["reflection_publication_outcome"] == "completed_revision"
    assert metadata["reflection_authority_sha256"] == authority.authority_sha256
    assert metadata["reflection_receipt_sha256"] == receipt.receipt_sha256


@pytest.mark.parametrize(
    ("fail", "abstain", "status", "outcome", "failure_type"),
    (
        (False, True, "sealed_complete", "completed_abstention", None),
        (True, False, "incomplete", "failed", "RuntimeError"),
    ),
)
def test_terminal_abstention_and_failure_preserve_sealed_optimization(
    fail: bool,
    abstain: bool,
    status: str,
    outcome: str,
    failure_type: str | None,
) -> None:
    run = _run(fail=fail, abstain=abstain)
    interceptor = run.interceptor
    result = run.result

    assert result.final_state.generation == 2
    assert [
        candidate.configuration_dict for candidate in result.final_state.candidates
    ] == [
        {"x": 5},
        {"x": 4},
        {"x": 3},
    ]
    assert result.final_state.logical_llm_calls == 1
    assert interceptor.reflected_entries == ()
    assert interceptor.reflection_receipt is not None
    assert interceptor.reflection_receipt.reflection_status == status
    assert interceptor.reflection_receipt.publication_outcome == outcome
    assert interceptor.reflection_failure_type == failure_type
    metadata = dict(result.feedback_receipts[-1].result_metadata)
    assert metadata["reflection_status"] == status
    assert metadata["reflection_publication_outcome"] == outcome
    if failure_type is None:
        assert interceptor.reflection_result is not None
        assert "reflection_failure_type" not in metadata
    else:
        assert interceptor.reflection_result is None
        assert metadata["reflection_failure_type"] == failure_type


def test_source_scope_uses_generation_slot_pairs_and_rejects_missing_scope() -> None:
    run = _run(abstain=True)
    receipts = run.result.generation_receipts
    reverse = PostEvolutionReflectionSourceScope(
        sources=(
            PostEvolutionReflectionSource(2, "mutant"),
            PostEvolutionReflectionSource(1, "mutant"),
        )
    )

    selected = reverse.select(receipts)
    assert tuple(outcome.prepared.plan.generation for outcome in selected) == (2, 1)
    with pytest.raises(ValueError, match="resolve each generation/slot pair"):
        PostEvolutionReflectionSourceScope(
            sources=(PostEvolutionReflectionSource(1, "absent"),)
        ).select(receipts)
    with pytest.raises(ValueError, match="cannot repeat"):
        PostEvolutionReflectionSourceScope(
            sources=(
                PostEvolutionReflectionSource(1, "mutant"),
                PostEvolutionReflectionSource(1, "mutant"),
            )
        )
    with pytest.raises(ValueError, match="follow the terminal generation"):
        PostEvolutionReflectionSpec(
            terminal_generation=1,
            source_scope=PostEvolutionReflectionSourceScope(
                sources=(PostEvolutionReflectionSource(2, "mutant"),)
            ),
            insight_contract=ReflectionInsightContract(
                required_metric_ids=("x",),
                allowed_option_families=("coordinate",),
            ),
        )


def test_factory_binds_exact_runtime_identities_and_is_single_use() -> None:
    run = _run(abstain=True)
    assert run.factory.runtime_identities == (
        run.benchmark,
        run.engine,
        run.ids,
        run.memory,
        run.planner,
    )
    with pytest.raises(RuntimeError, match="single-use"):
        run.factory.build(
            benchmark=run.benchmark,
            engine=run.engine,
            id_factory=run.ids,
            memory=run.memory,
            planner=run.planner,
        )

    foreign_ids = DeterministicIdFactory("foreign_terminal_reflection")
    foreign_factory = PostEvolutionReflectionFactory(
        spec=_spec(),
        predecessor_resolver=_Resolver(),
    )
    with pytest.raises(ValueError, match="foreign engine identities"):
        foreign_factory.build(
            benchmark=run.benchmark,
            engine=run.engine,
            id_factory=foreign_ids,
            memory=run.memory,
            planner=run.planner,
        )

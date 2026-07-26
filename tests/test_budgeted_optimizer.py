from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    CrossoverResponseMode,
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
    OptimizerBudgetExceeded,
    OptimizerContractError,
    OptimizerExecutionError,
    OptimizerSlot,
    OptimizerStopReason,
    SeedGateDecision,
    _generation_plan_record,
    _invocation_plan_record,
    _record_hash,
)
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackReservation,
    GenerationFeedbackResult,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.variation.exact_parent_crossover import (
    derive_exact_parent_crossover_contract,
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

    def __init__(self) -> None:
        self.evaluations: list[tuple[int, int]] = []

    @staticmethod
    def search_space_description() -> str:
        return "Minimize two bounded integer coordinates."

    @staticmethod
    def validate(configuration: object) -> bool:
        candidate = _Config.model_validate(configuration, strict=True)
        return 0 <= candidate.x <= 9 and 0 <= candidate.y <= 9

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        x, y = int(configuration["x"]), int(configuration["y"])
        self.evaluations.append((x, y))
        return {"x": float(x), "y": float(y)}


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="response-1",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class _Generator:
    def __init__(self) -> None:
        self.requests = []

    async def propose(self, request):
        self.requests.append(request)
        await asyncio.sleep(0)
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration={"x": 2, "y": 5},
                design_rationale="Test a smaller first coordinate.",
                intended_changes=("$.x",),
                source_attribution=(SourceAttribution("$.x", "mutation"),),
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def _reward(state_hash: str) -> FrozenWaveReward:
    def score(child, parents, objectives):
        del parents, objectives
        if not child.valid or not child.operator_compliant:
            return -1.0
        return 1.0 / (1.0 + child.objective_map["x"] + child.objective_map["y"])

    return FrozenWaveReward(
        binding=RewardPolicyBinding(score, _hash("test-frozen-wave-reward-v1")),
        archive_snapshot_hash=state_hash,
        reward_snapshot_hash=_hash(f"reward-snapshot:{state_hash}"),
    )


class _TwoGenerationPlanner:
    def __init__(self, ids: DeterministicIdFactory) -> None:
        self.ids = ids
        self.observed_state_hashes: list[str] = []
        self.observed_receipt_counts: list[int] = []

    def plan(self, state, budget):
        del budget
        self.observed_state_hashes.append(state.archive_snapshot_hash)
        self.observed_receipt_counts.append(len(state.generation_receipts))
        seed = state.candidates[0]
        reward = _reward(state.archive_snapshot_hash)
        if state.generation == 0:
            model_plan = InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (seed,),
                generation=1,
                label="g1_model_x",
                allowed_top_level=("x",),
            )
            engine_plan = InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (seed,),
                generation=1,
                label="g1_engine_y",
                allowed_top_level=("y",),
            )
            materialized = MaterializedInvocation(
                plan=engine_plan,
                draft=CandidateDraft(
                    configuration={"x": 5, "y": 2},
                    design_rationale="Engine-owned coverage edit.",
                    intended_changes=("$.y",),
                    source_attribution=(SourceAttribution("$.y", "mutation"),),
                ),
                candidate_id=self.ids.new_candidate_id(),
                materialization_policy_id="test_coverage_policy",
                materialization_policy_version=1,
                materialization_receipt_hash=_hash("g1-engine-materialization"),
            )
            return GenerationPlan(
                generation=1,
                slots=(
                    OptimizerSlot.model(
                        slot_id="G1-A", role="objective_extreme", plan=model_plan
                    ),
                    OptimizerSlot.engine(
                        slot_id="G1-X",
                        role="representation_coverage",
                        invocation=materialized,
                    ),
                ),
                reward=reward,
                planner_policy_id="two_generation_test_planner",
                planner_policy_version=1,
                metadata=(("wave", "atomic"),),
            )

        model_branch = state.candidates[1]
        union_plan = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (model_branch,),
            generation=2,
            label="g2_engine_refinement",
            allowed_top_level=("y",),
        )
        union = MaterializedInvocation(
            plan=union_plan,
            draft=CandidateDraft(
                configuration={"x": 2, "y": 1},
                design_rationale="Engine-owned second-wave composition.",
                intended_changes=("$.y",),
                source_attribution=(SourceAttribution("$.y", "mutation"),),
            ),
            candidate_id=self.ids.new_candidate_id(),
            materialization_policy_id="test_composition_policy",
            materialization_policy_version=1,
            materialization_receipt_hash=_hash("g2-engine-materialization"),
        )
        return GenerationPlan(
            generation=2,
            slots=(
                OptimizerSlot.engine(
                    slot_id="G2-E", role="exploit_composition", invocation=union
                ),
            ),
            reward=reward,
            planner_policy_id="two_generation_test_planner",
            planner_policy_version=1,
            metadata=(("wave", "composition"),),
        )


def _optimizer(
    *,
    planner,
    budget: OptimizerBudget,
    seed_gate=None,
    feedback_interceptor=None,
):
    ids = planner.ids
    problem = _Problem()
    generator = _Generator()
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=3,
        trace_sink=traces.append,
    )
    archive = ParetoArchive(
        problem.objectives,
        evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
    )
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=archive,
        planner=planner,
        budget=budget,
        seed_admission_policy=seed_gate,
        feedback_interceptor=feedback_interceptor,
        trace_sink=traces.append,
    )
    return optimizer, problem, generator, traces


def test_two_generation_mixed_authority_run_is_budgeted_frozen_and_auditable() -> None:
    ids = DeterministicIdFactory("budgeted_optimizer")
    planner = _TwoGenerationPlanner(ids)
    optimizer, problem, generator, traces = _optimizer(
        planner=planner,
        budget=OptimizerBudget(4, 1, 2),
    )

    result = asyncio.run(optimizer.run(({"x": 5, "y": 5},)))

    assert result.stop_reason is OptimizerStopReason.GENERATION_LIMIT_REACHED
    assert result.final_state.generation == 2
    assert result.final_state.unique_evaluations == 4
    assert result.final_state.logical_llm_calls == 1
    assert result.final_state.generation_receipts == result.generation_receipts
    assert planner.observed_receipt_counts == [0, 1]
    assert len(result.final_state.candidates) == 4
    assert len(generator.requests) == 1
    assert problem.evaluations[0] == (5, 5)
    assert set(problem.evaluations[1:3]) == {(2, 5), (5, 2)}
    assert problem.evaluations[3] == (2, 1)
    assert [
        candidate.configuration_dict
        for candidate in result.final_state.archive.front_candidates
    ] == [{"x": 2, "y": 1}]

    first, second = result.generation_receipts
    assert first.pre_archive_snapshot_hash == planner.observed_state_hashes[0]
    assert second.pre_archive_snapshot_hash == planner.observed_state_hashes[1]
    assert first.pre_archive_snapshot_hash != second.pre_archive_snapshot_hash
    assert first.reward_definition_hash == second.reward_definition_hash
    assert first.reward_snapshot_hash != second.reward_snapshot_hash
    assert first.reserved_logical_llm_calls == 1
    assert second.reserved_logical_llm_calls == 0
    assert [item.slot.slot_id for item in first.slot_results] == ["G1-A", "G1-X"]
    assert all(
        item.outcome.prepared.variation_case.reward_definition_hash
        == first.reward_definition_hash
        for item in first.slot_results
    )
    assert len(result.result_hash) == 64
    assert all(len(item.receipt_hash) == 64 for item in result.generation_receipts)
    assert result.seed_receipts[0].gate_decision.admitted is True

    planned_index = next(
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "optimizer_generation_planned"
    )
    llm_index = next(
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "llm_call_completed"
    )
    assert planned_index < llm_index


class _FeedbackAwarePlanner(_TwoGenerationPlanner):
    def __init__(self, ids: DeterministicIdFactory) -> None:
        super().__init__(ids)
        self.observed_feedback = []

    def plan(self, state, budget):
        self.observed_feedback.append(state.feedback_receipts)
        return super().plan(state, budget)


class _FeedbackInterceptor:
    def __init__(self) -> None:
        self.contexts = []

    def reserve(self, *, state, plan):
        del state
        return GenerationFeedbackReservation(
            policy_id="test_trace_reflection",
            policy_version=1,
            logical_llm_calls=int(plan.generation == 1),
            metadata=(("phase", f"generation_{plan.generation}"),),
        )

    async def after_generation(self, context):
        self.contexts.append(context)
        await asyncio.sleep(0)
        return GenerationFeedbackResult(
            logical_llm_calls_used=context.reservation.logical_llm_calls,
            metadata=(("insight_ref", f"feedback-{context.plan.generation}"),),
        )


def test_feedback_runs_after_sealed_receipt_and_conditions_next_planner_state() -> None:
    planner = _FeedbackAwarePlanner(
        DeterministicIdFactory("optimizer_generation_feedback")
    )
    feedback = _FeedbackInterceptor()
    optimizer, _, generator, traces = _optimizer(
        planner=planner,
        budget=OptimizerBudget(4, 2, 2),
        feedback_interceptor=feedback,
    )

    result = asyncio.run(optimizer.run(({"x": 5, "y": 5},)))

    assert result.final_state.logical_llm_calls == 2
    assert len(generator.requests) == 1
    assert len(feedback.contexts) == 2
    assert len(result.feedback_receipts) == 2
    assert result.feedback_receipts[0].reserved_logical_llm_calls == 1
    assert result.feedback_receipts[1].reserved_logical_llm_calls == 0
    assert planner.observed_feedback[0] == ()
    assert planner.observed_feedback[1] == (result.feedback_receipts[0],)
    assert planner.observed_feedback[1][0].result_metadata == (
        ("insight_ref", "feedback-1"),
    )
    assert result.generation_receipts[1].logical_llm_calls_before == 2

    generation_completed = next(
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "optimizer_generation_completed"
        and event["generation"] == 1
    )
    feedback_completed = next(
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "optimizer_generation_feedback_completed"
        and event["generation"] == 1
    )
    next_generation_planned = next(
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "optimizer_generation_planned"
        and event["generation"] == 2
    )
    assert generation_completed < feedback_completed < next_generation_planned


def test_feedback_reservation_is_rejected_before_generation_provider_calls() -> None:
    planner = _FeedbackAwarePlanner(
        DeterministicIdFactory("optimizer_feedback_over_budget")
    )
    feedback = _FeedbackInterceptor()
    optimizer, _, generator, traces = _optimizer(
        planner=planner,
        budget=OptimizerBudget(4, 1, 2),
        feedback_interceptor=feedback,
    )

    with pytest.raises(OptimizerBudgetExceeded):
        asyncio.run(optimizer.run(({"x": 5, "y": 5},)))

    assert generator.requests == []
    assert feedback.contexts == []
    rejected = [
        event
        for event in traces
        if event["event_type"] == "optimizer_generation_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["feedback_reservation"]["logical_llm_calls"] == 1


class _OverBudgetPlanner:
    def __init__(self, ids: DeterministicIdFactory) -> None:
        self.ids = ids

    def plan(self, state, budget):
        del budget
        parent = state.candidates[0]
        slots = tuple(
            OptimizerSlot.model(
                slot_id=f"G1-{index}",
                role="duplicate_test_role",
                plan=InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (parent,),
                    generation=1,
                    label=f"over_budget_{index}",
                    allowed_top_level=("x",),
                ),
            )
            for index in range(2)
        )
        return GenerationPlan(
            generation=1,
            slots=slots,
            reward=_reward(state.archive_snapshot_hash),
            planner_policy_id="over_budget_test",
            planner_policy_version=1,
        )


def test_whole_wave_is_rejected_before_calls_when_llm_reservation_exceeds_cap() -> None:
    planner = _OverBudgetPlanner(DeterministicIdFactory("optimizer_over_budget"))
    optimizer, problem, generator, traces = _optimizer(
        planner=planner,
        budget=OptimizerBudget(5, 1, 1),
    )

    with pytest.raises(OptimizerBudgetExceeded, match="LLM-call budget"):
        asyncio.run(optimizer.run(({"x": 5, "y": 5},)))

    assert generator.requests == []
    assert problem.evaluations == [(5, 5)]
    rejected = next(
        event
        for event in traces
        if event["event_type"] == "optimizer_generation_rejected"
    )
    assert rejected["logical_llm_call_reservation"] == 2
    assert len(rejected["plan_hash"]) == 64


class _StaleRewardPlanner:
    def __init__(self, ids: DeterministicIdFactory) -> None:
        self.ids = ids

    def plan(self, state, budget):
        del budget
        parent = state.candidates[0]
        stale = FrozenWaveReward(
            binding=_reward(state.archive_snapshot_hash).binding,
            archive_snapshot_hash="f" * 64,
            reward_snapshot_hash=_hash("stale-reward"),
        )
        return GenerationPlan(
            generation=1,
            slots=(
                OptimizerSlot.model(
                    slot_id="G1-stale",
                    role="stale_reward_test",
                    plan=InvocationPlan(
                        OperatorKind.TYPED_MUTATION,
                        (parent,),
                        generation=1,
                        label="stale_reward",
                        allowed_top_level=("x",),
                    ),
                ),
            ),
            reward=stale,
            planner_policy_id="stale_reward_test",
            planner_policy_version=1,
        )


def test_stale_reward_cutoff_fails_closed_before_model_or_child_evaluation() -> None:
    planner = _StaleRewardPlanner(DeterministicIdFactory("optimizer_stale_reward"))
    optimizer, problem, generator, _ = _optimizer(
        planner=planner,
        budget=OptimizerBudget(2, 1, 1),
    )

    with pytest.raises(OptimizerContractError, match="pre-wave archive cutoff"):
        asyncio.run(optimizer.run(({"x": 5, "y": 5},)))

    assert generator.requests == []
    assert problem.evaluations == [(5, 5)]


def test_zero_generation_budget_returns_seed_only_and_optimizer_is_single_use() -> None:
    ids = DeterministicIdFactory("optimizer_seed_only")
    planner = _TwoGenerationPlanner(ids)
    optimizer, problem, generator, _ = _optimizer(
        planner=planner,
        budget=OptimizerBudget(1, 0, 0),
    )

    result = asyncio.run(optimizer.run(({"x": 5, "y": 5},)))

    assert result.final_state.generation == 0
    assert result.final_state.unique_evaluations == 1
    assert result.generation_receipts == ()
    assert planner.observed_state_hashes == []
    assert generator.requests == []
    assert problem.evaluations == [(5, 5)]
    with pytest.raises(OptimizerContractError, match="single-use"):
        asyncio.run(optimizer.run(({"x": 4, "y": 4},)))


class _ExactSeedGate:
    def __init__(self, *, expected_x: float, provenance_receipt: str) -> None:
        self.expected_x = expected_x
        self.provenance_receipt = provenance_receipt

    def assess(self, candidate, context):
        identity_ok = (
            candidate.occurrence.configuration_hash
            == context.requested_configuration_hash
        )
        objectives_ok = candidate.objective_map.get("x") == self.expected_x
        admitted = candidate.valid and identity_ok and objectives_ok
        return SeedGateDecision(
            admitted=admitted,
            policy_id="exact_seed_identity_objectives_and_provenance",
            policy_version=1,
            reason="all exact seed gates passed" if admitted else "objective mismatch",
            evidence=tuple(
                sorted(
                    (
                        ("identity_ok", str(identity_ok).lower()),
                        ("objectives_ok", str(objectives_ok).lower()),
                        ("provenance_receipt_sha256", self.provenance_receipt),
                    )
                )
            ),
        )


def test_injected_exact_seed_gate_records_provenance_and_aborts_before_planning() -> (
    None
):
    ids = DeterministicIdFactory("optimizer_exact_seed_gate")
    planner = _TwoGenerationPlanner(ids)
    provenance = _hash("evaluator-source-and-cec-receipt")
    optimizer, problem, generator, traces = _optimizer(
        planner=planner,
        budget=OptimizerBudget(2, 1, 1),
        seed_gate=_ExactSeedGate(expected_x=4.0, provenance_receipt=provenance),
    )

    with pytest.raises(OptimizerExecutionError, match="seed gate rejected"):
        asyncio.run(optimizer.run(({"x": 5, "y": 5},)))

    assert problem.evaluations == [(5, 5)]
    assert generator.requests == []
    assert planner.observed_state_hashes == []
    completed = next(
        event for event in traces if event["event_type"] == "optimizer_seed_completed"
    )
    assert completed["gate"]["admitted"] is False
    assert ["provenance_receipt_sha256", provenance] in completed["gate"]["evidence"]
    assert completed["unique_evaluations_after"] == 1
    assert not any(
        event["event_type"] == "optimizer_generation_planned" for event in traces
    )


def test_generation_plan_hash_binds_exact_crossover_mode_and_full_contract() -> None:
    async def register_parents():
        ids = DeterministicIdFactory("exact_crossover_plan_hash")
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=_Generator(),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=3,
        )
        base = await engine.register_seed({"x": 5, "y": 5}, label="base")
        donor = await engine.register_seed({"x": 2, "y": 3}, label="donor")
        return base, donor

    base, donor = asyncio.run(register_parents())
    default_contract = derive_exact_parent_crossover_contract(
        base=base.configuration,
        donor=donor.configuration,
    )
    wide_contract = derive_exact_parent_crossover_contract(
        base=base.configuration,
        donor=donor.configuration,
        max_loci=4_096,
    )
    assert default_contract.contract_sha256 != wide_contract.contract_sha256

    def invocation(contract=None, forbidden=()) -> InvocationPlan:
        return InvocationPlan(
            operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
            parents=(base, donor),
            generation=1,
            label="exact_crossover_plan_hash",
            crossover_response_mode=(
                CrossoverResponseMode.FULL_CONFIGURATION
                if contract is None
                else CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
            ),
            exact_parent_crossover_contract=contract,
            forbidden_exact_parent_import_sets=forbidden,
        )

    full = invocation()
    exact_default = invocation(default_contract)
    exact_wide = invocation(wide_contract)
    exact_excluded = invocation(default_contract, (("locus_0001",),))

    full_invocation_record = _invocation_plan_record(full)
    default_invocation_record = _invocation_plan_record(exact_default)
    wide_invocation_record = _invocation_plan_record(exact_wide)
    excluded_invocation_record = _invocation_plan_record(exact_excluded)

    # Default/full plans retain the historical record shape.  Exact plans bind
    # both the complete replay contract and its domain-separated identity.
    assert "crossover_response_mode" not in full_invocation_record
    assert "exact_parent_crossover_contract" not in full_invocation_record
    assert "exact_parent_crossover_contract_sha256" not in full_invocation_record
    assert default_invocation_record["crossover_response_mode"] == (
        "exact_parent_import_v1"
    )
    assert default_invocation_record["exact_parent_crossover_contract"] == (
        default_contract.to_record()
    )
    assert (
        default_invocation_record["exact_parent_crossover_contract_sha256"]
        == default_contract.contract_sha256
    )
    assert wide_invocation_record["exact_parent_crossover_contract"] == (
        wide_contract.to_record()
    )
    assert (
        wide_invocation_record["exact_parent_crossover_contract_sha256"]
        == wide_contract.contract_sha256
    )
    assert default_invocation_record["forbidden_exact_parent_import_sets"] == []
    assert excluded_invocation_record["forbidden_exact_parent_import_sets"] == [
        ["locus_0001"]
    ]
    assert (
        default_invocation_record["exact_parent_import_exclusions_sha256"]
        != (excluded_invocation_record["exact_parent_import_exclusions_sha256"])
    )

    reward = _reward(_hash("exact-crossover-pre-archive"))
    budget_hash = OptimizerBudget(3, 3, 1).budget_hash

    def generation_plan_hash(plan: InvocationPlan) -> str:
        generation_plan = GenerationPlan(
            generation=1,
            slots=(
                OptimizerSlot.model(
                    slot_id="G1-X",
                    role="exact_crossover_hash_probe",
                    plan=plan,
                ),
            ),
            reward=reward,
            planner_policy_id="exact_crossover_hash_probe",
            planner_policy_version=1,
        )
        return _record_hash(
            "generation-plan",
            _generation_plan_record(generation_plan, budget_hash=budget_hash),
        )

    full_hash = generation_plan_hash(full)
    default_hash = generation_plan_hash(exact_default)
    wide_hash = generation_plan_hash(exact_wide)
    excluded_hash = generation_plan_hash(exact_excluded)

    assert full_hash != default_hash
    assert default_hash != wide_hash
    assert default_hash != excluded_hash

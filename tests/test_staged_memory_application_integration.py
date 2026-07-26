"""Provider-free integration tests for the staged causal-memory boundary."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    OperatorKind,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    FrozenWaveReward,
    GenerationPlan,
    OptimizerBudget,
    OptimizerContractError,
    OptimizerSlot,
    generation_receipt_hash,
    validate_generation_receipt_integrity,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    context_stratum_hash,
)
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.application.staged_memory import (
    DiagnosticMemoryCheckpointService,
    memory_assignment_receipt,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.ids import InsightId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.llm_task_queue import (
    AttemptStatus,
    AttemptTelemetry,
    CancellationReason,
    LLMTaskOutcome,
    RetryClassification,
    RetryDisposition,
    RetryReason,
    SanitizedAttemptFailure,
    TaskOutcomeStatus,
    TaskTelemetry,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    OutcomePublicationError,
    QueuedStructuredGenerationError,
)
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    DeterministicMemoryControlPolicy,
    FrozenDiagnosticMemoryWave,
    MemoryAssignmentArm,
    MemoryCheckpointClosure,
    MemoryCheckpointClosureStatus,
    MemoryScoreSnapshot,
    MemoryTrialTerminalStatus,
    ResolvedInsightAssignment,
    WaveSealedCheckpointBuilder,
)
from agent_evolve.policies.memory.randomized_subset import InsightSelectionDecision
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


REWARD_DEFINITION_HASH = _hash("v6-application-integration-reward-v1")
NO_YIELD_REWARD = -3.0


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
        candidate = _Config.model_validate(configuration, strict=True)
        return 0 <= candidate.x <= 9 and 0 <= candidate.y <= 9

    @staticmethod
    def evaluate(configuration: dict[str, object]) -> dict[str, float]:
        return {
            "x": float(configuration["x"]),
            "y": float(configuration["y"]),
        }


def _telemetry(index: int) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="provider-free",
        provider_response_id=f"response-{index}",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=index + 1,
        attempt_count=1,
    )


class _MixedTerminalGenerator:
    """Produce success/model-failure/candidate-failure/success/success."""

    def __init__(
        self,
        memory: InsightMemoryBank,
        model_failure: Exception,
    ) -> None:
        self.memory = memory
        self.model_failure = model_failure
        self.requests = []
        self.mutable_trial_counts: list[int] = []

    async def propose(self, request):
        index = len(self.requests)
        self.requests.append(request)
        self.mutable_trial_counts.append(len(self.memory.trials))
        await asyncio.sleep(0.004 if index == 0 else 0)
        if index == 1:
            raise self.model_failure

        selected_ids = tuple(
            sorted(set(re.findall(r'"insight_id":"([^"]+)"', request.prompt)))
        )
        if index == 2:
            # CandidateDraft itself is well formed, but the engine's typed-JSON
            # trust boundary must reject the opaque value before evaluation.
            configuration = {"x": object(), "y": 5}
        else:
            configuration = {"x": (2, 3, 4)[min(index, 4) - 2], "y": 5}
            if index == 0:
                configuration = {"x": 1, "y": 5}
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale="Exercise the resolved causal-memory boundary.",
                intended_changes=("$.x",),
                source_attribution=(SourceAttribution("$.x", "mutation"),),
                claimed_insight_ids=selected_ids,
            ),
            telemetry=_telemetry(index),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry(99))


def _score(child, parents, objectives) -> float:
    del objectives
    if not child.valid or not child.operator_compliant:
        return -1.0
    parent_x = parents[0].objective_map["x"]
    return float(parent_x - child.objective_map["x"])


@dataclass(frozen=True, slots=True)
class _AssignmentSpec:
    arm: MemoryAssignmentArm
    decision: InsightSelectionDecision
    index: int


class _Planner:
    def __init__(
        self,
        *,
        engine: AgenticEvolutionEngine,
        ids: DeterministicIdFactory,
        snapshot: MemoryScoreSnapshot,
        specs: tuple[_AssignmentSpec, ...],
        score: Callable,
        service: DiagnosticMemoryCheckpointService,
    ) -> None:
        self.engine = engine
        self.ids = ids
        self.snapshot = snapshot
        self.specs = specs
        self.score = score
        self.service = service
        self.assignments: tuple[ResolvedInsightAssignment, ...] = ()
        self.wave: FrozenDiagnosticMemoryWave | None = None

    def plan(self, state, budget) -> GenerationPlan:
        del budget
        assert state.generation == 0
        parent = state.candidates[0]
        plans = []
        assignments = []
        for spec in self.specs:
            base_plan = InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (parent,),
                generation=1,
                label=f"resolved_{spec.arm.value}_{spec.index}",
                allowed_top_level=("x",),
                phase="v6_diagnostic_integration",
            )
            prompt_shape = self.engine.prompt_shape_commitment(
                base_plan,
                selected_insight_count=len(spec.decision.selected),
                reward_definition_hash=REWARD_DEFINITION_HASH,
            )
            assignment = ResolvedInsightAssignment.resolve(
                credit_unit_id=self.ids.new_operator_invocation_id(),
                snapshot=self.snapshot,
                expected_snapshot_sha256=self.snapshot.snapshot_sha256,
                block_id=f"v6_block_{spec.index}",
                arm=spec.arm,
                selection_decision=spec.decision,
                prompt_shape_sha256=prompt_shape,
            )
            assignments.append(assignment)
            plans.append(replace(base_plan, resolved_insight_assignment=assignment))
        self.assignments = tuple(assignments)
        diagnostic = tuple(
            sorted(
                (
                    assignment
                    for assignment in self.assignments
                    if assignment.arm is MemoryAssignmentArm.DIAGNOSTIC
                ),
                key=lambda value: value.assignment_sha256,
            )
        )
        self.wave = FrozenDiagnosticMemoryWave(
            wave_id="v6_integration_diagnostic_wave",
            prior_snapshot=self.snapshot,
            assignments=diagnostic,
            reward_definition_hash=REWARD_DEFINITION_HASH,
            no_yield_reward=NO_YIELD_REWARD,
        )
        self.service.publish_frozen_wave(self.wave)
        slots = tuple(
            OptimizerSlot.model(
                slot_id=f"G1-{index}",
                role=assignment.arm.value,
                plan=plan,
            )
            for index, (assignment, plan) in enumerate(
                zip(self.assignments, plans, strict=True)
            )
        )
        binding = RewardPolicyBinding(self.score, REWARD_DEFINITION_HASH)
        return GenerationPlan(
            generation=1,
            slots=slots,
            reward=FrozenWaveReward(
                binding=binding,
                archive_snapshot_hash=state.archive_snapshot_hash,
                reward_snapshot_hash=_hash(
                    f"v6-reward-snapshot:{state.archive_snapshot_hash}"
                ),
            ),
            planner_policy_id="v6_staged_memory_integration",
            planner_policy_version=1,
        )


@dataclass(frozen=True, slots=True)
class _Scenario:
    engine: AgenticEvolutionEngine
    generator: _MixedTerminalGenerator
    memory: InsightMemoryBank
    score_policy: CausalSearchScorePolicy
    prior_snapshot: MemoryScoreSnapshot
    diagnostic_assignments: tuple[ResolvedInsightAssignment, ...]
    later_assignments: tuple[ResolvedInsightAssignment, ...]
    wave: FrozenDiagnosticMemoryWave
    service: DiagnosticMemoryCheckpointService
    result: object
    closure: MemoryCheckpointClosure
    traces: list[dict[str, object]]


def _model_output_failure() -> StructuredGenerationError:
    return StructuredGenerationError(
        kind=GenerationFailureKind.OUTPUT_INVALID,
        retryable=False,
        safe_message="closed provider-free model output failure",
    )


def _queued_failure(
    kind: str,
    *,
    status: TaskOutcomeStatus = TaskOutcomeStatus.TERMINAL_FAILURE,
) -> QueuedStructuredGenerationError:
    reasons = {
        "output_invalid": RetryReason.OUTPUT_INVALID,
        "content_rejected": RetryReason.PERMANENT,
        "rate_limited": RetryReason.RATE_LIMIT,
        "timeout": RetryReason.TIMEOUT,
        "provider_unavailable": RetryReason.TRANSIENT,
        "authentication": RetryReason.PERMANENT,
        "payment_required": RetryReason.PERMANENT,
    }
    failure = SanitizedAttemptFailure(
        kind=kind,
        retryable=False,
        safe_message="closed queued terminal failure",
    )
    classification = RetryClassification(
        disposition=RetryDisposition.FAIL,
        reason=reasons[kind],
        sanitized_failure=failure,
    )
    attempt = AttemptTelemetry(
        attempt_number=1,
        status=AttemptStatus.TERMINAL_FAILURE,
        wait_time_ns=0,
        service_time_ns=1,
        will_retry=False,
        classification=classification,
        error_type="StructuredGenerationError",
    )
    telemetry = TaskTelemetry(
        task_id="v6_failure_classification",
        queue_time_ns=0,
        service_time_ns=1,
        total_time_ns=1,
        attempts=(attempt,),
    )
    return QueuedStructuredGenerationError(
        LLMTaskOutcome(status=status, telemetry=telemetry)
    )


def _queued_cancellation() -> QueuedStructuredGenerationError:
    attempt = AttemptTelemetry(
        attempt_number=1,
        status=AttemptStatus.CANCELLED,
        wait_time_ns=0,
        service_time_ns=1,
        will_retry=False,
        error_type="CancelledError",
    )
    telemetry = TaskTelemetry(
        task_id="v6_cancelled_generation",
        queue_time_ns=0,
        service_time_ns=1,
        total_time_ns=1,
        attempts=(attempt,),
    )
    return QueuedStructuredGenerationError(
        LLMTaskOutcome(
            status=TaskOutcomeStatus.CANCELLED,
            telemetry=telemetry,
            cancellation_reason=CancellationReason.QUEUE_CLOSED,
        )
    )


def _run_scenario(
    *,
    prompt_shape: str = "v6-prompt-shape-a",
    model_failure: Exception | None = None,
    problem: _Problem | None = None,
    score: Callable = _score,
) -> _Scenario:
    ids = DeterministicIdFactory("v6_app_integration")
    memory = InsightMemoryBank(id_factory=ids)
    entries = memory.extend(
        (
            InsightDraft(
                claim="Reducing x can improve the first objective.",
                trigger="x is editable",
                mechanism="the evaluator minimizes x",
                affected_paths=("$.x",),
                evidence_summary="provider-free integration fixture A",
                confidence=0.5,
            ),
            InsightDraft(
                claim="A second independent x heuristic may help.",
                trigger="x is editable",
                mechanism="it proposes an alternate bounded x value",
                affected_paths=("$.x",),
                evidence_summary="provider-free integration fixture B",
                confidence=0.5,
            ),
        )
    )
    references = tuple(entry.reference for entry in entries)
    traces: list[dict[str, object]] = []
    generator = _MixedTerminalGenerator(
        memory,
        _model_output_failure() if model_failure is None else model_failure,
    )
    engine = AgenticEvolutionEngine(
        problem=_Problem() if problem is None else problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=7,
        trace_sink=traces.append,
        max_output_tokens=(768 if prompt_shape == "v6-prompt-shape-a" else 769),
    )
    context_hash = context_stratum_hash(
        problem_id=engine.problem_id,
        operator_kind=OperatorKind.TYPED_MUTATION.value,
        phase="v6_diagnostic_integration",
    )
    score_policy = CausalSearchScorePolicy(
        prior_effective_sample_size=1.0,
        uncertainty_scale=0.0,
        exploration_weight=0.0,
    )
    snapshot = score_policy.genesis(
        exact_context_hash=context_hash,
        estimand_stratum_hash=_hash("v6-integration-estimand"),
        priors={references[0]: 2.0, references[1]: 0.0},
    )
    controls = DeterministicMemoryControlPolicy()
    specs = tuple(
        _AssignmentSpec(
            MemoryAssignmentArm.DIAGNOSTIC,
            controls.uniform(
                snapshot=snapshot,
                subset_size=1,
                subset_rank=rank,
            ),
            index,
        )
        for index, rank in enumerate((0, 1, 0))
    ) + (
        _AssignmentSpec(
            MemoryAssignmentArm.ADAPTIVE,
            controls.adaptive(snapshot=snapshot, subset_size=1),
            3,
        ),
        _AssignmentSpec(
            MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL,
            controls.score_shuffled(
                snapshot=snapshot,
                subset_size=1,
                permutation_rank=1,
            ),
            4,
        ),
    )
    service = DiagnosticMemoryCheckpointService(
        WaveSealedCheckpointBuilder(score_policy),
        trace_sink=traces.append,
    )
    planner = _Planner(
        engine=engine,
        ids=ids,
        snapshot=snapshot,
        specs=specs,
        score=score,
        service=service,
    )
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=ParetoArchive(
            engine.objectives,
            evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
        ),
        planner=planner,
        budget=OptimizerBudget(
            max_unique_evaluations=6,
            max_logical_llm_calls=5,
            max_generations=1,
        ),
        trace_sink=traces.append,
    )
    result = asyncio.run(optimizer.run(({"x": 5, "y": 5},)))
    if planner.wave is None:
        raise AssertionError("planner did not publish its frozen diagnostic wave")
    wave = planner.wave
    closure = service.close_generation(wave, result.generation_receipts[0])
    diagnostic = tuple(
        assignment
        for assignment in planner.assignments
        if assignment.arm is MemoryAssignmentArm.DIAGNOSTIC
    )
    later = tuple(
        assignment
        for assignment in planner.assignments
        if assignment.arm is not MemoryAssignmentArm.DIAGNOSTIC
    )
    return _Scenario(
        engine=engine,
        generator=generator,
        memory=memory,
        score_policy=score_policy,
        prior_snapshot=snapshot,
        diagnostic_assignments=diagnostic,
        later_assignments=later,
        wave=wave,
        service=service,
        result=result,
        closure=closure,
        traces=traces,
    )


def test_real_optimizer_receipt_seals_success_and_failure_itt_without_live_credit() -> (
    None
):
    scenario = _run_scenario()
    receipt = scenario.result.generation_receipts[0]
    outcomes = tuple(result.outcome for result in receipt.slot_results)

    assert tuple(value.failure_stage for value in outcomes) == (
        None,
        "llm",
        "candidate",
        None,
        None,
    )
    assert scenario.generator.mutable_trial_counts == [0, 0, 0, 0, 0]
    assert scenario.memory.trials == ()
    assert scenario.prior_snapshot.checkpoint_index == 0
    assert scenario.prior_snapshot.observations == ()

    closure = scenario.closure
    assert closure.status is MemoryCheckpointClosureStatus.SEALED
    assert closure.snapshot is not None
    assert closure.snapshot.checkpoint_index == 1
    assert closure.snapshot.parent_snapshot_sha256 == (
        scenario.prior_snapshot.snapshot_sha256
    )
    observations = {
        value.assignment.credit_unit_id: value for value in closure.observations
    }
    assert tuple(
        observations[assignment.credit_unit_id].status
        for assignment in scenario.diagnostic_assignments
    ) == (
        MemoryTrialTerminalStatus.SUCCEEDED,
        MemoryTrialTerminalStatus.MODEL_FAILURE,
        MemoryTrialTerminalStatus.CANDIDATE_FAILURE,
    )
    assert (
        observations[scenario.diagnostic_assignments[0].credit_unit_id].credited_reward
        == outcomes[0].reward
    )
    assert all(
        observations[assignment.credit_unit_id].credited_reward == NO_YIELD_REWARD
        for assignment in scenario.diagnostic_assignments[1:]
    )
    assert len(closure.receipts) == len(scenario.diagnostic_assignments)
    assert {value.assignment.assignment_sha256 for value in closure.observations} == {
        value.assignment_sha256 for value in scenario.diagnostic_assignments
    }

    # Later matched arms consume resolved memory but are not diagnostic evidence.
    assert scenario.later_assignments[0].selection_decision.selected != (
        scenario.later_assignments[1].selection_decision.selected
    )
    assert all(outcome.candidate is not None for outcome in outcomes[3:])
    assert all(
        assignment.assignment_sha256
        not in {value.assignment.assignment_sha256 for value in closure.observations}
        for assignment in scenario.later_assignments
    )
    with pytest.raises(ValueError, match="not a resolved diagnostic"):
        memory_assignment_receipt(outcomes[3])


@pytest.mark.parametrize(
    "failure",
    (
        _queued_failure("output_invalid"),
        _queued_failure(
            "output_invalid",
            status=TaskOutcomeStatus.ATTEMPTS_EXHAUSTED,
        ),
        _queued_failure("content_rejected"),
    ),
)
def test_explicit_queued_model_or_schema_failures_enter_no_yield_itt(
    failure: Exception,
) -> None:
    scenario = _run_scenario(model_failure=failure)
    outcome = scenario.result.generation_receipts[0].slot_results[1].outcome

    assert outcome.failure_stage == "llm"
    assert scenario.closure.status is MemoryCheckpointClosureStatus.SEALED
    observation = next(
        value
        for value in scenario.closure.observations
        if value.assignment == scenario.diagnostic_assignments[1]
    )
    assert observation.status is MemoryTrialTerminalStatus.MODEL_FAILURE
    assert observation.credited_reward == NO_YIELD_REWARD
    terminal = next(
        event
        for event in scenario.traces
        if event.get("event_type") == "trial_terminal"
        and event.get("assignment_sha256")
        == scenario.diagnostic_assignments[1].assignment_sha256
    )
    assert terminal["terminal_status"] == "model_or_schema_failure"
    assert terminal["reward_disposition"] == "impute_wave_no_yield_at_seal"


def _structured_failure(kind: GenerationFailureKind) -> StructuredGenerationError:
    return StructuredGenerationError(
        kind=kind,
        retryable=False,
        safe_message="closed structured infrastructure failure",
    )


@pytest.mark.parametrize(
    "failure",
    (
        *(
            _queued_failure(kind)
            for kind in (
                "rate_limited",
                "timeout",
                "provider_unavailable",
                "authentication",
                "payment_required",
            )
        ),
        _queued_cancellation(),
        OutcomePublicationError(_queued_failure("output_invalid").outcome),
        RuntimeError("untyped failure must fail closed"),
        *(
            _structured_failure(kind)
            for kind in (
                GenerationFailureKind.CAPABILITY_MISMATCH,
                GenerationFailureKind.INVALID_REQUEST,
                GenerationFailureKind.UNKNOWN,
            )
        ),
    ),
)
def test_provider_scheduler_and_untyped_failures_invalidate_diagnostic_wave(
    failure: Exception,
) -> None:
    scenario = _run_scenario(model_failure=failure)
    outcome = scenario.result.generation_receipts[0].slot_results[1].outcome

    assert outcome.failure_stage == "infrastructure"
    terminal_receipt = memory_assignment_receipt(outcome)
    assert terminal_receipt.status is MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE
    assert scenario.closure.status is (
        MemoryCheckpointClosureStatus.INVALIDATED_INFRASTRUCTURE
    )
    assert scenario.closure.snapshot is None
    assert scenario.closure.observations == ()
    terminal = next(
        event
        for event in scenario.traces
        if event.get("event_type") == "trial_terminal"
        and event.get("assignment_sha256")
        == scenario.diagnostic_assignments[1].assignment_sha256
    )
    assert terminal["terminal_status"] == "infrastructure_failure"
    assert terminal["reward_disposition"] == "invalidates_block"


class _EvaluatorFailureProblem(_Problem):
    @staticmethod
    def evaluate(configuration: dict[str, object]) -> dict[str, float]:
        if configuration["x"] == 1:
            raise RuntimeError("closed evaluator infrastructure failure")
        return _Problem.evaluate(configuration)


def _reward_failure(child, parents, objectives) -> float:
    if child.objective_map["x"] == 1:
        raise RuntimeError("closed reward-policy infrastructure failure")
    return _score(child, parents, objectives)


@pytest.mark.parametrize(
    ("problem", "score"),
    (
        (_EvaluatorFailureProblem(), _score),
        (_Problem(), _reward_failure),
    ),
)
def test_evaluator_and_reward_exceptions_become_infrastructure_receipts(
    problem: _Problem,
    score: Callable,
) -> None:
    scenario = _run_scenario(problem=problem, score=score)
    outcome = scenario.result.generation_receipts[0].slot_results[0].outcome

    assert outcome.failure_stage == "infrastructure"
    assert memory_assignment_receipt(outcome).status is (
        MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE
    )
    assert scenario.closure.status is (
        MemoryCheckpointClosureStatus.INVALIDATED_INFRASTRUCTURE
    )
    assert scenario.result.final_state.generation == 1


def test_assignment_is_generation_plan_hash_bound_and_replay_stable() -> None:
    first = _run_scenario(prompt_shape="v6-prompt-shape-a")
    replay = _run_scenario(prompt_shape="v6-prompt-shape-a")
    changed = _run_scenario(prompt_shape="v6-prompt-shape-b")

    first_receipt = first.result.generation_receipts[0]
    replay_receipt = replay.result.generation_receipts[0]
    changed_receipt = changed.result.generation_receipts[0]
    assert first_receipt.plan_hash == replay_receipt.plan_hash
    assert first_receipt.plan_hash != changed_receipt.plan_hash
    planned = next(
        event
        for event in first.traces
        if event["event_type"] == "optimizer_generation_planned"
    )
    assert planned["plan_hash"] == first_receipt.plan_hash
    assert [
        slot["invocation"]["resolved_insight_assignment"]["assignment_sha256"]
        for slot in planned["slots"]
    ] == [
        value.assignment_sha256
        for value in (*first.diagnostic_assignments, *first.later_assignments)
    ]


def test_duplicate_foreign_and_context_mismatched_assignments_fail_closed() -> None:
    scenario = _run_scenario()
    receipt = scenario.result.generation_receipts[0]
    assert generation_receipt_hash(receipt) == receipt.receipt_hash
    validate_generation_receipt_integrity(receipt)

    successful = receipt.slot_results[0]
    altered_outcome = replace(
        successful.outcome,
        reward=successful.outcome.reward + 100.0,
    )
    stale_hash_tamper = replace(
        receipt,
        slot_results=(
            replace(successful, outcome=altered_outcome),
            *receipt.slot_results[1:],
        ),
    )
    assert generation_receipt_hash(stale_hash_tamper) != receipt.receipt_hash
    with pytest.raises(OptimizerContractError, match="does not authenticate"):
        scenario.service.close_generation(scenario.wave, stale_hash_tamper)
    with pytest.raises(OptimizerContractError, match="does not authenticate"):
        replace(
            scenario.result.final_state,
            generation_receipts=(stale_hash_tamper,),
        )

    altered_plan = replace(
        successful.slot.plan,
        resolved_insight_assignment=scenario.diagnostic_assignments[1],
    )
    altered_prepared = replace(successful.outcome.prepared, plan=altered_plan)
    plan_tampered_result = replace(
        successful,
        slot=replace(successful.slot, plan=altered_plan),
        outcome=replace(successful.outcome, prepared=altered_prepared),
    )
    stale_plan_tamper = replace(
        receipt,
        slot_results=(plan_tampered_result, *receipt.slot_results[1:]),
    )
    assert generation_receipt_hash(stale_plan_tamper) != receipt.receipt_hash
    with pytest.raises(OptimizerContractError, match="does not authenticate"):
        scenario.service.close_generation(scenario.wave, stale_plan_tamper)

    inconsistent_plan_tamper = replace(
        receipt,
        slot_results=(
            replace(
                successful,
                outcome=replace(successful.outcome, prepared=altered_prepared),
            ),
            *receipt.slot_results[1:],
        ),
    )
    with pytest.raises(OptimizerContractError, match="slot plan differs"):
        validate_generation_receipt_integrity(inconsistent_plan_tamper)

    diagnostic_result = receipt.slot_results[0]
    duplicate_result = replace(
        diagnostic_result,
        slot=replace(diagnostic_result.slot, slot_id="G1-duplicate-assignment"),
    )
    duplicated_with_stale_hash = replace(
        receipt,
        logical_llm_calls_after=receipt.logical_llm_calls_after + 1,
        reserved_logical_llm_calls=receipt.reserved_logical_llm_calls + 1,
        reserved_unique_evaluations=receipt.reserved_unique_evaluations + 1,
        slot_results=(*receipt.slot_results, duplicate_result),
    )
    duplicated = replace(
        duplicated_with_stale_hash,
        receipt_hash=generation_receipt_hash(duplicated_with_stale_hash),
    )
    with pytest.raises(ValueError, match="repeats a diagnostic assignment"):
        scenario.service.close_generation(scenario.wave, duplicated)

    controls = DeterministicMemoryControlPolicy()
    foreign_assignment = ResolvedInsightAssignment.resolve(
        credit_unit_id=OperatorInvocationId("operator_foreign_receipt"),
        snapshot=scenario.prior_snapshot,
        expected_snapshot_sha256=scenario.prior_snapshot.snapshot_sha256,
        block_id="v6_foreign_receipt",
        arm=MemoryAssignmentArm.DIAGNOSTIC,
        selection_decision=controls.uniform(
            snapshot=scenario.prior_snapshot,
            subset_size=1,
            subset_rank=1,
        ),
        prompt_shape_sha256=_hash("v6-prompt-shape-a"),
    )
    foreign_wave = FrozenDiagnosticMemoryWave(
        wave_id="v6_foreign_wave",
        prior_snapshot=scenario.prior_snapshot,
        assignments=(foreign_assignment,),
        reward_definition_hash=REWARD_DEFINITION_HASH,
        no_yield_reward=NO_YIELD_REWARD,
    )
    with pytest.raises(ValueError, match="differ from the frozen wave"):
        scenario.service.close_generation(foreign_wave, receipt)

    parent = scenario.result.final_state.candidates[0]
    call_count = len(scenario.generator.requests)
    repeated_plan = InvocationPlan(
        OperatorKind.TYPED_MUTATION,
        (parent,),
        generation=2,
        label="duplicate_assignment",
        allowed_top_level=("x",),
        phase="v6_diagnostic_integration",
        resolved_insight_assignment=scenario.diagnostic_assignments[0],
    )
    with pytest.raises(ValueError, match="already reserved"):
        asyncio.run(scenario.engine.run_invocations((repeated_plan,)))
    assert len(scenario.generator.requests) == call_count

    wrong_context_snapshot = scenario.score_policy.genesis(
        exact_context_hash=_hash("wrong-exact-context"),
        estimand_stratum_hash=_hash("wrong-context-estimand"),
        priors={
            entry.reference: entry.prior_score
            for entry in scenario.prior_snapshot.entries
        },
    )
    wrong_context = ResolvedInsightAssignment.resolve(
        credit_unit_id=OperatorInvocationId("operator_wrong_context"),
        snapshot=wrong_context_snapshot,
        expected_snapshot_sha256=wrong_context_snapshot.snapshot_sha256,
        block_id="v6_wrong_context",
        arm=MemoryAssignmentArm.DIAGNOSTIC,
        selection_decision=controls.uniform(
            snapshot=wrong_context_snapshot,
            subset_size=1,
            subset_rank=0,
        ),
        prompt_shape_sha256=_hash("v6-prompt-shape-a"),
    )
    with pytest.raises(ValueError, match="context differs"):
        asyncio.run(
            scenario.engine.run_invocations(
                (
                    replace(
                        repeated_plan,
                        label="wrong_context",
                        resolved_insight_assignment=wrong_context,
                    ),
                )
            )
        )
    assert len(scenario.generator.requests) == call_count

    foreign_references = (
        InsightRef(InsightId("insight_foreign_memory_a"), 1),
        InsightRef(InsightId("insight_foreign_memory_b"), 1),
    )
    foreign_snapshot = scenario.score_policy.genesis(
        exact_context_hash=scenario.prior_snapshot.exact_context_hash,
        estimand_stratum_hash=_hash("foreign-insight-estimand"),
        priors={reference: 0.0 for reference in foreign_references},
    )
    foreign_insight = ResolvedInsightAssignment.resolve(
        credit_unit_id=OperatorInvocationId("operator_foreign_memory"),
        snapshot=foreign_snapshot,
        expected_snapshot_sha256=foreign_snapshot.snapshot_sha256,
        block_id="v6_foreign_memory",
        arm=MemoryAssignmentArm.DIAGNOSTIC,
        selection_decision=controls.uniform(
            snapshot=foreign_snapshot,
            subset_size=1,
            subset_rank=0,
        ),
        prompt_shape_sha256=_hash("v6-prompt-shape-a"),
    )
    with pytest.raises(ValueError, match="unavailable or structurally inapplicable"):
        asyncio.run(
            scenario.engine.run_invocations(
                (
                    replace(
                        repeated_plan,
                        label="foreign_insight",
                        resolved_insight_assignment=foreign_insight,
                    ),
                )
            )
        )
    assert len(scenario.generator.requests) == call_count


def test_staged_trace_records_freeze_commit_terminal_seal_and_publish() -> None:
    scenario = _run_scenario()
    event_types = [event["event_type"] for event in scenario.traces]

    assert event_types.count("memory_wave_frozen") == 1
    assert event_types.count("assignment_committed") == 5
    assert event_types.count("trial_terminal") == 5
    assert event_types.count("memory_wave_sealed") == 1
    assert event_types.count("memory_checkpoint_published") == 1

    committed = [
        event
        for event in scenario.traces
        if event["event_type"] == "assignment_committed"
    ]
    terminal = [
        event for event in scenario.traces if event["event_type"] == "trial_terminal"
    ]
    expected = {
        value.assignment_sha256
        for value in (*scenario.diagnostic_assignments, *scenario.later_assignments)
    }
    assert {event["assignment_sha256"] for event in committed} == expected
    assert {event["assignment_sha256"] for event in terminal} == expected
    assert all(event["prompt_shape_commitment_verified"] is True for event in committed)
    assert {event["prompt_shape_sha256"] for event in committed} == {
        assignment.prompt_shape_sha256
        for assignment in (
            *scenario.diagnostic_assignments,
            *scenario.later_assignments,
        )
    }
    assert [event["failure_stage"] for event in terminal] == [
        None,
        "llm",
        "candidate",
        None,
        None,
    ]
    frozen_index = event_types.index("memory_wave_frozen")
    first_commit_index = event_types.index("assignment_committed")
    first_terminal_index = event_types.index("trial_terminal")
    sealed_index = event_types.index("memory_wave_sealed")
    published_index = event_types.index("memory_checkpoint_published")
    assert frozen_index < first_commit_index < first_terminal_index
    assert first_terminal_index < sealed_index < published_index

"""Reusable provider-neutral v6 closed-loop development scenario.

The scenario is intentionally synthetic and tiny.  It exists to exercise the
orchestration mechanism through production application boundaries; it is not a
benchmark and cannot support quality, SOTA, or wall-clock claims.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
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
    OptimizerState,
)
from agent_evolve.application.evaluation_recourse import (
    EvaluationRecourseApplicationService,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    context_stratum_hash,
)
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
)
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.application.staged_memory import (
    DiagnosticMemoryCheckpointService,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    DeterministicMemoryControlPolicy,
    FrozenDiagnosticMemoryWave,
    MemoryAssignmentArm,
    MemoryCheckpointClosure,
    MemoryCheckpointClosureStatus,
    ResolvedInsightAssignment,
    WaveSealedCheckpointBuilder,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    BoundedEvaluationRecoursePolicy,
    EvaluationRecourseDecision,
    PhenotypeOccurrenceLedger,
    PresealedRecoursePool,
    RecourseBudgetSnapshot,
    RecoursePoolCandidate,
    TypedConfigurationPhenotypeIdentityPolicy,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchRecombiner,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AgenticGenerator,
    CandidateDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def canonical_record_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


REWARD_DEFINITION_HASH = sha256_text("v6-closed-loop-parent-a-improvement-v1")
PHASE = "v6_closed_loop_memory"
ID_NAMESPACE = "v6_closed_loop_development_probe_v1"
MAX_OUTPUT_TOKENS = 768
TEMPERATURE = 0.2
MODEL_WAVE_WIDTH = 2
FULL_WAVE_WIDTH = 4
OPTIMIZER_BUDGET = OptimizerBudget(
    max_unique_evaluations=7,
    max_logical_llm_calls=4,
    max_generations=4,
)


class ClosedLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    a: int
    b: int


class ClosedLoopProblem:
    candidate_model = ClosedLoopConfig
    objectives = (ObjectiveSpec("a", "min"), ObjectiveSpec("b", "min"))

    def __init__(self, *, evaluation_delay_seconds: float = 0.025) -> None:
        if (
            isinstance(evaluation_delay_seconds, bool)
            or not isinstance(evaluation_delay_seconds, (int, float))
            or evaluation_delay_seconds < 0
        ):
            raise ValueError("evaluation_delay_seconds must be non-negative")
        self.evaluation_delay_seconds = float(evaluation_delay_seconds)
        self.evaluated: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    @staticmethod
    def search_space_description() -> str:
        return (
            "Engineering-only synthetic mechanism probe. Minimize the bounded "
            "integer coordinates a and b in [0,4]. A selected memory hypothesis "
            "must change only a and should follow its explicit target value."
        )

    @staticmethod
    def validate(configuration: object) -> bool:
        value = ClosedLoopConfig.model_validate(configuration, strict=True)
        return 0 <= value.a <= 4 and 0 <= value.b <= 4

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        value = ClosedLoopConfig.model_validate(configuration, strict=True)
        if self.evaluation_delay_seconds:
            time.sleep(self.evaluation_delay_seconds)
        with self._lock:
            self.evaluated.append((value.a, value.b))
        return {"a": float(value.a), "b": float(value.b)}


def closed_loop_reward(
    child: EvolutionCandidate,
    parents: tuple[EvolutionCandidate, ...],
    objectives,
) -> float:
    del objectives
    if not (child.valid and child.operator_compliant and child.evidence_compliant):
        return -10.0
    return float(parents[0].objective_map["a"] - child.objective_map["a"])


def _telemetry(index: int) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/v6-closed-loop",
        resolved_model="offline/v6-closed-loop",
        resolved_provider="provider-free",
        provider_response_id=f"offline-v6-response-{index}",
        finish_reason="fixture",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=0,
        attempt_count=1,
    )


class OfflineInsightConditionedGenerator:
    """Exact provider-free proposal policy for mechanism regression tests."""

    def __init__(self, memory: InsightMemoryBank, *, a_id: str, b_id: str) -> None:
        self.memory = memory
        self.a_id = a_id
        self.b_id = b_id
        self.selected_ids: list[tuple[str, ...]] = []
        self.mutable_trial_counts: list[int] = []

    async def propose(self, request):
        selected = tuple(
            sorted(set(re.findall(r'"insight_id":"([^"]+)"', request.prompt)))
        )
        if len(selected) != 1 or selected[0] not in {self.a_id, self.b_id}:
            raise AssertionError("offline fixture requires one resolved A/B insight")
        index = len(self.selected_ids)
        self.selected_ids.append(selected)
        self.mutable_trial_counts.append(len(self.memory.trials))
        await asyncio.sleep(0)
        a = 3 if selected[0] == self.a_id else 1
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration={"a": a, "b": 4},
                design_rationale="Apply exactly the assigned a-coordinate target.",
                intended_changes=("$.a",),
                source_attribution=(SourceAttribution("$.a", "mutation"),),
                claimed_insight_ids=selected,
            ),
            telemetry=_telemetry(index),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry(99))


def _resolve_assignment(
    *,
    ids: DeterministicIdFactory,
    snapshot,
    controls: DeterministicMemoryControlPolicy,
    arm: MemoryAssignmentArm,
    block_id: str,
    prompt_shape_sha256: str,
    uniform_rank: int | None = None,
    permutation_rank: int | None = None,
) -> ResolvedInsightAssignment:
    if arm is MemoryAssignmentArm.DIAGNOSTIC:
        if uniform_rank is None or permutation_rank is not None:
            raise ValueError("diagnostic assignment requires one uniform rank")
        decision = controls.uniform(
            snapshot=snapshot,
            subset_size=1,
            subset_rank=uniform_rank,
        )
    elif arm is MemoryAssignmentArm.ADAPTIVE:
        if uniform_rank is not None or permutation_rank is not None:
            raise ValueError("adaptive assignment accepts no randomization rank")
        decision = controls.adaptive(snapshot=snapshot, subset_size=1)
    elif arm is MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL:
        if uniform_rank is not None or permutation_rank is None:
            raise ValueError("score-shuffled assignment requires a permutation rank")
        decision = controls.score_shuffled(
            snapshot=snapshot,
            subset_size=1,
            permutation_rank=permutation_rank,
        )
    else:
        raise ValueError("unsupported closed-loop memory arm")
    return ResolvedInsightAssignment.resolve(
        credit_unit_id=ids.new_operator_invocation_id(),
        snapshot=snapshot,
        expected_snapshot_sha256=snapshot.snapshot_sha256,
        block_id=block_id,
        arm=arm,
        selection_decision=decision,
        prompt_shape_sha256=prompt_shape_sha256,
    )


def _engine_mutation(
    *,
    ids: DeterministicIdFactory,
    parent: EvolutionCandidate,
    generation: int,
    label: str,
    configuration: dict[str, object],
    changed_paths: tuple[str, ...],
    policy_id: str,
    receipt_material: str,
) -> MaterializedInvocation:
    allowed = tuple(
        sorted(path.removeprefix("$.").split(".", 1)[0] for path in changed_paths)
    )
    plan = InvocationPlan(
        OperatorKind.TYPED_MUTATION,
        (parent,),
        generation=generation,
        label=label,
        allowed_top_level=allowed,
        phase="v6_engine_materialization",
    )
    return MaterializedInvocation(
        plan=plan,
        draft=CandidateDraft(
            configuration=configuration,
            design_rationale="Precommitted engine-authored coverage candidate.",
            intended_changes=changed_paths,
            source_attribution=tuple(
                SourceAttribution(path, "mutation") for path in changed_paths
            ),
        ),
        candidate_id=ids.new_candidate_id(),
        materialization_policy_id=policy_id,
        materialization_policy_version=1,
        materialization_receipt_hash=sha256_text(receipt_material),
    )


class ClosedLoopPlanner:
    policy_id = "v6_provider_free_closed_loop"
    policy_version = 2

    def __init__(
        self,
        *,
        ids: DeterministicIdFactory,
        engine: AgenticEvolutionEngine,
        binding: RewardPolicyBinding,
        genesis,
        checkpoint_service: DiagnosticMemoryCheckpointService,
        controls: DeterministicMemoryControlPolicy,
        recourse_pool: PresealedRecoursePool,
        recourse_service: EvaluationRecourseApplicationService,
        max_output_tokens: int,
        temperature: float | None,
    ) -> None:
        self.ids = ids
        self.engine = engine
        self.binding = binding
        self.genesis = genesis
        self.checkpoint_service = checkpoint_service
        self.controls = controls
        self.recourse_pool = recourse_pool
        self.recourse_service = recourse_service
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        self.prompt_shape_sha256: str | None = None
        self.diagnostic_assignments: tuple[ResolvedInsightAssignment, ...] = ()
        self.wave: FrozenDiagnosticMemoryWave | None = None
        self.closure: MemoryCheckpointClosure | None = None
        self.adaptive_assignment: ResolvedInsightAssignment | None = None
        self.control_assignment: ResolvedInsightAssignment | None = None
        self.recourse_decision: EvaluationRecourseDecision | None = None
        self.combined_ledger: PhenotypeOccurrenceLedger | None = None

    def _reward(self, state: OptimizerState, generation: int) -> FrozenWaveReward:
        return FrozenWaveReward(
            binding=self.binding,
            archive_snapshot_hash=state.archive_snapshot_hash,
            reward_snapshot_hash=sha256_text(
                f"v6-g{generation}:{state.archive_snapshot_hash}"
            ),
        )

    def plan(self, state: OptimizerState, budget: OptimizerBudget) -> GenerationPlan:
        generation = state.generation + 1
        if generation == 1:
            return self._diagnostic_plan(state)
        if generation == 2:
            return self._matched_plan(state)
        if generation == 3:
            return self._recourse_plan(state, budget)
        if generation == 4:
            return self._recombination_plan(state)
        raise ValueError("closed-loop probe has exactly four generations")

    def _diagnostic_plan(self, state: OptimizerState) -> GenerationPlan:
        if self.wave is not None or self.diagnostic_assignments:
            raise RuntimeError("diagnostic wave was already frozen")
        parent = state.candidates[0]
        base_plans = tuple(
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (parent,),
                generation=1,
                label=f"G1_diagnostic_{index}",
                allowed_top_level=("a",),
                phase=PHASE,
            )
            for index in range(2)
        )
        commitments = tuple(
            self.engine.prompt_shape_commitment(
                plan,
                selected_insight_count=1,
                reward_definition_hash=self.binding.definition_hash,
            )
            for plan in base_plans
        )
        if len(set(commitments)) != 1:
            raise RuntimeError("matched diagnostic plans have different prompt shapes")
        self.prompt_shape_sha256 = commitments[0]
        self.diagnostic_assignments = tuple(
            _resolve_assignment(
                ids=self.ids,
                snapshot=self.genesis,
                controls=self.controls,
                arm=MemoryAssignmentArm.DIAGNOSTIC,
                block_id=f"v6_diagnostic_{name}",
                prompt_shape_sha256=self.prompt_shape_sha256,
                uniform_rank=index,
            )
            for index, name in enumerate(("a", "b"))
        )
        self.wave = FrozenDiagnosticMemoryWave(
            wave_id="v6_closed_loop_diagnostic_wave",
            prior_snapshot=self.genesis,
            assignments=tuple(
                sorted(
                    self.diagnostic_assignments,
                    key=lambda item: item.assignment_sha256,
                )
            ),
            reward_definition_hash=self.binding.definition_hash,
            no_yield_reward=-10.0,
        )
        self.checkpoint_service.publish_frozen_wave(self.wave)
        slots = tuple(
            OptimizerSlot.model(
                slot_id=f"G1-diagnostic-{index}",
                role="diagnostic_memory",
                plan=replace(
                    base_plans[index],
                    resolved_insight_assignment=assignment,
                ),
            )
            for index, assignment in enumerate(self.diagnostic_assignments)
        )
        return GenerationPlan(
            generation=1,
            slots=slots,
            reward=self._reward(state, 1),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=(
                ("diagnostic_wave_sha256", self.wave.wave_sha256),
                ("prompt_shape_sha256", self.prompt_shape_sha256),
            ),
        )

    def _matched_plan(self, state: OptimizerState) -> GenerationPlan:
        if self.wave is None or self.prompt_shape_sha256 is None:
            raise RuntimeError("diagnostic wave is unavailable")
        self.closure = self.checkpoint_service.close_generation(
            self.wave,
            state.generation_receipts[0],
        )
        if self.closure.status is not MemoryCheckpointClosureStatus.SEALED:
            raise RuntimeError("diagnostic memory wave did not seal")
        snapshot = self.closure.snapshot
        if snapshot is None:
            raise RuntimeError("sealed memory wave has no checkpoint")
        parent = state.candidates[0]
        adaptive_base = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=2,
            label="G2-adaptive",
            allowed_top_level=("a",),
            phase=PHASE,
        )
        control_base = replace(adaptive_base, label="G2-control")
        matched_commitments = (
            self.engine.prompt_shape_commitment(
                adaptive_base,
                selected_insight_count=1,
                reward_definition_hash=self.binding.definition_hash,
            ),
            self.engine.prompt_shape_commitment(
                control_base,
                selected_insight_count=1,
                reward_definition_hash=self.binding.definition_hash,
            ),
        )
        if set(matched_commitments) != {self.prompt_shape_sha256}:
            raise RuntimeError("G1/G2 matched prompt-shape commitment drifted")
        self.adaptive_assignment = _resolve_assignment(
            ids=self.ids,
            snapshot=snapshot,
            controls=self.controls,
            arm=MemoryAssignmentArm.ADAPTIVE,
            block_id="v6_matched_block",
            prompt_shape_sha256=self.prompt_shape_sha256,
        )
        self.control_assignment = _resolve_assignment(
            ids=self.ids,
            snapshot=snapshot,
            controls=self.controls,
            arm=MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL,
            block_id="v6_matched_block",
            prompt_shape_sha256=self.prompt_shape_sha256,
            permutation_rank=1,
        )
        model_slots = (
            OptimizerSlot.model(
                slot_id="G2-adaptive",
                role="adaptive_memory",
                plan=replace(
                    adaptive_base,
                    resolved_insight_assignment=self.adaptive_assignment,
                ),
            ),
            OptimizerSlot.model(
                slot_id="G2-control",
                role="score_shuffled_control",
                plan=replace(
                    control_base,
                    resolved_insight_assignment=self.control_assignment,
                ),
            ),
        )
        duplicate_slots = tuple(
            OptimizerSlot.engine(
                slot_id=f"G2-coverage-{index}",
                role="duplicate_coverage_primary",
                invocation=_engine_mutation(
                    ids=self.ids,
                    parent=parent,
                    generation=2,
                    label=f"G2_coverage_{index}",
                    configuration={"a": 2, "b": 2},
                    changed_paths=("$.a", "$.b"),
                    policy_id="v6_precommitted_duplicate_coverage",
                    receipt_material=f"v6-coverage-duplicate-{index}",
                ),
            )
            for index in range(2)
        )
        return GenerationPlan(
            generation=2,
            slots=(*model_slots, *duplicate_slots),
            reward=self._reward(state, 2),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        ("memory_snapshot_sha256", snapshot.snapshot_sha256),
                        ("prompt_shape_sha256", self.prompt_shape_sha256),
                        ("recourse_pool_sha256", self.recourse_pool.pool_sha256),
                    )
                )
            ),
        )

    def _recourse_plan(
        self,
        state: OptimizerState,
        budget: OptimizerBudget,
    ) -> GenerationPlan:
        self.recourse_decision = self.recourse_service.decide(
            primary_receipt=state.generation_receipts[1],
            pool=self.recourse_pool,
            budget=RecourseBudgetSnapshot(
                max_unique_evaluations=budget.max_unique_evaluations,
                used_unique_evaluations=state.unique_evaluations,
                reserved_non_recourse_evaluations=0,
                protected_recombination_evaluations=1,
            ),
        )
        if self.recourse_decision.selected_entry_ids != ("orthogonal_b",):
            raise RuntimeError("bounded recourse did not select the frozen target")
        selected = self.recourse_decision.selected_entries[0]
        configuration = thaw_json(selected.candidate.configuration)
        if type(configuration) is not dict:
            raise TypeError("recourse configuration is not an object")
        invocation = _engine_mutation(
            ids=self.ids,
            parent=state.candidates[0],
            generation=3,
            label="G3_recourse_orthogonal_b",
            configuration=configuration,
            changed_paths=("$.b",),
            policy_id="v6_presealed_objective_blind_recourse",
            receipt_material=(
                self.recourse_pool.pool_sha256
                + self.recourse_decision.decision_sha256
                + selected.entry_id
            ),
        )
        return GenerationPlan(
            generation=3,
            slots=(
                OptimizerSlot.engine(
                    slot_id="G3-recourse",
                    role="bounded_evaluation_recourse",
                    invocation=invocation,
                ),
            ),
            reward=self._reward(state, 3),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        (
                            "recourse_decision_sha256",
                            self.recourse_decision.decision_sha256,
                        ),
                        ("recourse_pool_sha256", self.recourse_pool.pool_sha256),
                    )
                )
            ),
        )

    def _recombination_plan(self, state: OptimizerState) -> GenerationPlan:
        if self.recourse_decision is None:
            raise RuntimeError("recourse decision is unavailable")
        self.combined_ledger = self.recourse_service.append_recourse_receipt(
            self.recourse_decision.ledger,
            state.generation_receipts[2],
            recourse_slot_ids=("G3-recourse",),
        )
        adaptive = state.generation_receipts[1].slot_results[0].outcome.candidate
        recourse = state.generation_receipts[2].slot_results[0].outcome.candidate
        if adaptive is None or recourse is None:
            raise RuntimeError("recombination requires adaptive and recourse children")
        ancestor = state.candidates[0]
        materialization = DisjointPatchRecombiner().materialize(
            ancestor=ancestor.configuration,
            ancestor_candidate_id=ancestor.candidate_id,
            left=adaptive.configuration,
            left_candidate_id=adaptive.candidate_id,
            right=recourse.configuration,
            right_candidate_id=recourse.candidate_id,
            target_candidate_id=self.ids.new_candidate_id(),
        )
        plan = InvocationPlan(
            OperatorKind.THREE_WAY_RECOMBINATION,
            (adaptive, recourse),
            generation=4,
            label="G4_disjoint_recombination",
            common_ancestor=ancestor,
            phase="v6_disjoint_recombination",
        )
        invocation = materialized_disjoint_invocation(
            plan=plan,
            materialization=materialization,
        )
        return GenerationPlan(
            generation=4,
            slots=(
                OptimizerSlot.engine(
                    slot_id="G4-recombine",
                    role="engine_disjoint_recombination",
                    invocation=invocation,
                ),
            ),
            reward=self._reward(state, 4),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=(
                ("recombination_receipt_sha256", materialization.receipt_sha256),
            ),
        )


GeneratorFactory = Callable[
    [InsightMemoryBank, InsightRef, InsightRef],
    AgenticGenerator,
]


@dataclass(frozen=True, slots=True)
class ProbeComposition:
    problem: ClosedLoopProblem
    memory: InsightMemoryBank
    a_ref: InsightRef
    b_ref: InsightRef
    generator: AgenticGenerator
    engine: AgenticEvolutionEngine
    archive: ParetoArchive
    planner: ClosedLoopPlanner
    optimizer: BudgetedAgenticOptimizer
    recourse_pool: PresealedRecoursePool


def compose_probe(
    *,
    generator_factory: GeneratorFactory,
    trace_sink: Callable[[Mapping[str, object]], None] | None = None,
    evaluation_delay_seconds: float = 0.025,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float | None = TEMPERATURE,
) -> ProbeComposition:
    if not callable(generator_factory):
        raise TypeError("generator_factory must be callable")
    ids = DeterministicIdFactory(ID_NAMESPACE)
    memory = InsightMemoryBank(id_factory=ids)
    entries = memory.extend(
        (
            InsightDraft(
                claim="Set a to exactly 3 and preserve b exactly.",
                trigger="The parent is {a:4,b:4} and only a is editable.",
                mechanism=(
                    "This conservative mutation reduces a by one; emit the "
                    "complete configuration {a:3,b:4}."
                ),
                affected_paths=("$.a",),
                evidence_summary="Engineering hypothesis A; not benchmark evidence.",
                confidence=0.5,
            ),
            InsightDraft(
                claim="Set a to exactly 1 and preserve b exactly.",
                trigger="The parent is {a:4,b:4} and only a is editable.",
                mechanism=(
                    "This aggressive mutation reduces a by three; emit the "
                    "complete configuration {a:1,b:4}."
                ),
                affected_paths=("$.a",),
                evidence_summary="Engineering hypothesis B; not benchmark evidence.",
                confidence=0.5,
            ),
        )
    )
    a_ref, b_ref = tuple(sorted(entry.reference for entry in entries))
    generator = generator_factory(memory, a_ref, b_ref)
    problem = ClosedLoopProblem(evaluation_delay_seconds=evaluation_delay_seconds)
    identity_policy = TypedConfigurationPhenotypeIdentityPolicy()
    binding = RewardPolicyBinding(closed_loop_reward, REWARD_DEFINITION_HASH)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=17,
        evaluator_concurrency=FULL_WAVE_WIDTH,
        trace_sink=trace_sink,
        reward_policy=closed_loop_reward,
        reward_definition_hash=REWARD_DEFINITION_HASH,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        phenotype_identity_policy=identity_policy,
    )
    score_policy = CausalSearchScorePolicy(
        prior_effective_sample_size=1.0,
        uncertainty_scale=0.0,
        exploration_weight=0.0,
    )
    genesis = score_policy.genesis(
        exact_context_hash=context_stratum_hash(
            problem_id=engine.problem_id,
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            phase=PHASE,
        ),
        estimand_stratum_hash=sha256_text("v6-closed-loop-singleton-itt-estimand"),
        priors={a_ref: 0.0, b_ref: 0.0},
    )
    checkpoint_service = DiagnosticMemoryCheckpointService(
        WaveSealedCheckpointBuilder(score_policy),
        trace_sink=trace_sink,
    )
    recourse_pool = PresealedRecoursePool.seal(
        pool_id="v6.presealed.orthogonal.coverage",
        seal_context_sha256=sha256_text("v6-pre-outcome-recourse-pool-contract"),
        candidates=(RecoursePoolCandidate.freeze("orthogonal_b", {"a": 4, "b": 1}),),
        identity_policy=identity_policy,
    )
    recourse_service = EvaluationRecourseApplicationService(
        identity_policy=identity_policy,
        recourse_policy=BoundedEvaluationRecoursePolicy(max_recourse=1),
        trace_sink=trace_sink,
    )
    planner = ClosedLoopPlanner(
        ids=ids,
        engine=engine,
        binding=binding,
        genesis=genesis,
        checkpoint_service=checkpoint_service,
        controls=DeterministicMemoryControlPolicy(),
        recourse_pool=recourse_pool,
        recourse_service=recourse_service,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    archive = ParetoArchive(problem.objectives)
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=archive,
        planner=planner,
        budget=OPTIMIZER_BUDGET,
        trace_sink=trace_sink,
    )
    return ProbeComposition(
        problem=problem,
        memory=memory,
        a_ref=a_ref,
        b_ref=b_ref,
        generator=generator,
        engine=engine,
        archive=archive,
        planner=planner,
        optimizer=optimizer,
        recourse_pool=recourse_pool,
    )


def offline_generator_factory(
    memory: InsightMemoryBank,
    a_ref: InsightRef,
    b_ref: InsightRef,
) -> OfflineInsightConditionedGenerator:
    return OfflineInsightConditionedGenerator(
        memory,
        a_id=a_ref.insight_id.value,
        b_id=b_ref.insight_id.value,
    )


__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopPlanner",
    "ClosedLoopProblem",
    "FULL_WAVE_WIDTH",
    "ID_NAMESPACE",
    "MAX_OUTPUT_TOKENS",
    "MODEL_WAVE_WIDTH",
    "OPTIMIZER_BUDGET",
    "OfflineInsightConditionedGenerator",
    "PHASE",
    "ProbeComposition",
    "REWARD_DEFINITION_HASH",
    "TEMPERATURE",
    "canonical_record_sha256",
    "closed_loop_reward",
    "compose_probe",
    "offline_generator_factory",
    "sha256_text",
]

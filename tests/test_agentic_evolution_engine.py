from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import random
import re
import sys
import threading
import time
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    CrossoverResponseMode,
    InvocationPlan,
    OperatorKind,
    REWARD_DEFINITION_HASH,
    RewardPolicyBinding,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightOrigin,
    InsightMemoryBank,
    context_stratum_hash,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.randomized_subset import (
    EpsilonGreedySubsetSelector,
)
from agent_evolve.policies.memory.prompt_shape import (
    DefaultEvidencePromptShapePolicyV1,
)
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    MemoryAssignmentReceipt,
    MemoryAssignmentArm,
    MemoryTrialTerminalStatus,
    ResolvedInsightAssignment,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    derive_exact_parent_crossover_contract,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ExactParentCrossoverDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
)


def _load_pipeline_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "pipeline_codesign"
        / "problem_def.py"
    )
    name = "_agent_evolve_test_pipeline_codesign"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dag_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "dag_dispatch_codesign"
        / "problem_def.py"
    )
    name = "_agent_evolve_test_dag_dispatch_codesign_engine"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake-provider",
        provider_response_id="fake-response",
        finish_reason="stop",
        input_tokens=100,
        output_tokens=50,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1_000,
        attempt_count=1,
    )


class _FakeAgenticGenerator:
    def __init__(self, module) -> None:
        self.module = module
        self.active = 0
        self.max_active = 0
        self.propose_calls = 0

    async def propose(self, request):
        self.propose_calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.005)
            prompt = request.prompt
            insight_ids = tuple(
                sorted(set(re.findall(r'"insight_id":"([^"]+)"', prompt)))
            )
            if "OPERATOR: two_parent_crossover" in prompt:
                config = self.module.DEVELOPMENT_BRANCH_LEFT
                rationale = "Copied only the left parent as an ignored-parent control."
            elif "OPERATOR: typed_mutation" in prompt:
                config = self.module.DEVELOPMENT_BRANCH_LEFT
                rationale = "Changed compiler components despite a runtime-only scope."
            else:
                config = self.module.DEVELOPMENT_RECOMBINATION_TARGET
                rationale = (
                    "Preserved the compiler branch and runtime branch innovations."
                )
            obligation_ids = tuple(
                sorted(set(re.findall(r'"obligation_id":"([0-9a-f]{64})"', prompt)))
            )
            draft = CandidateDraft(
                configuration={
                    "passes": list(config["passes"]),
                    "frontend": dict(config["frontend"]),
                    "backend": dict(config["backend"]),
                    "runtime": dict(config["runtime"]),
                },
                design_rationale=rationale,
                intended_changes=("compose branch innovations",),
                source_attribution=(
                    SourceAttribution(path="$.frontend", source="left"),
                    SourceAttribution(path="$.runtime", source="right"),
                ),
                claimed_insight_ids=insight_ids,
                claimed_preservation_obligation_ids=obligation_ids,
            )
            return VariationGenerationResult(draft=draft, telemetry=_telemetry())
        finally:
            self.active -= 1

    async def reflect(self, request):
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim="Combine disjoint compiler and runtime improvements.",
                    trigger="two branches modify separate top-level components",
                    mechanism="their typed patches can be composed without overwriting either innovation",
                    affected_paths=("$.frontend", "$.runtime"),
                    evidence_summary=(
                        "The recombined child dominated its compiler parent; "
                        "an unrelated prose prefix is "
                        f"{request.available_contrast_ids[-1][:8]}."
                    ),
                    confidence=0.8,
                    evidence_contrast_ids=request.available_contrast_ids[:1],
                ),
            ),
            telemetry=_telemetry(),
        )


class _SlowEvaluationProblem:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.candidate_model = delegate.candidate_model
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    @property
    def objectives(self):
        return self.delegate.objectives

    def search_space_description(self):
        return self.delegate.search_space_description()

    def validate(self, config):
        return self.delegate.validate(config)

    def evaluate(self, config):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            return self.delegate.evaluate(config)
        finally:
            with self._lock:
                self.active -= 1


class _StaticCandidateGenerator:
    def __init__(self, configuration) -> None:
        self.configuration = configuration

    async def propose(self, request):
        obligation_ids = tuple(
            sorted(set(re.findall(r'"obligation_id":"([0-9a-f]{64})"', request.prompt)))
        )
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=copy.deepcopy(self.configuration),
                design_rationale="Adversarially preserved only one left edit.",
                source_attribution=(
                    SourceAttribution(path="$.assignments", source="left"),
                    SourceAttribution(path="$.dispatch_order", source="right"),
                ),
                claimed_preservation_obligation_ids=obligation_ids,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def _seed_insights(memory: InsightMemoryBank) -> None:
    memory.extend(
        (
            InsightDraft(
                claim="SOA layout helps vectorized code.",
                trigger="vectorize is enabled",
                mechanism="contiguous lanes reduce gather overhead",
                affected_paths=("$.runtime.data_layout", "$.passes"),
                evidence_summary="development prior for randomized testing",
                confidence=0.6,
            ),
            InsightDraft(
                claim="Prefetch distance four helps multi-threaded execution.",
                trigger="threads are at least two",
                mechanism="moderate lookahead hides memory latency",
                affected_paths=("$.runtime.prefetch_distance",),
                evidence_summary="development prior for randomized testing",
                confidence=0.55,
            ),
        )
    )


async def _resolved_trace_failure_case(
    *,
    name: str,
    generator_factory,
    problem_factory=None,
    reward_policy=None,
    failure_score: float = -1.0,
):
    module = _load_pipeline_module()
    ids = DeterministicIdFactory(name)
    memory = InsightMemoryBank(id_factory=ids)
    _seed_insights(memory)
    traces: list[dict[str, object]] = []
    problem = (
        module.PipelineCoDesignProblem()
        if problem_factory is None
        else problem_factory(module)
    )
    arguments = {
        "problem": problem,
        "generator": generator_factory(module),
        "id_factory": ids,
        "memory": memory,
        "seed": 17,
        "trace_sink": traces.append,
        "failure_score": failure_score,
    }
    if reward_policy is not None:
        arguments.update(
            {
                "reward_policy": reward_policy,
                "reward_definition_hash": hashlib.sha256(
                    f"test:{name}:reward:v1".encode("ascii")
                ).hexdigest(),
            }
        )
    engine = AgenticEvolutionEngine(**arguments)
    seed = await engine.register_seed(module.BASE_CONFIG, label="base")
    phase = f"{name}_phase"
    context = context_stratum_hash(
        problem_id=engine.problem_id,
        operator_kind=OperatorKind.TYPED_MUTATION.value,
        phase=phase,
    )
    eligible = memory.eligible_references(
        operator_kind=OperatorKind.TYPED_MUTATION.value,
        editable_paths=("$.runtime",),
    )
    snapshot = CausalSearchScorePolicy().genesis(
        exact_context_hash=context,
        estimand_stratum_hash=hashlib.sha256(
            f"test:{name}:estimand:v1".encode("ascii")
        ).hexdigest(),
        priors={reference: 0.0 for reference in eligible},
    )
    decision = EpsilonGreedySubsetSelector(Fraction(1, 2)).select(
        context_hash=context,
        eligible=eligible,
        scores=snapshot.retrieval_scores,
        subset_size=1,
        rng=random.Random(9),
    )
    unassigned_plan = InvocationPlan(
        OperatorKind.TYPED_MUTATION,
        (seed,),
        generation=1,
        label=name,
        allowed_top_level=("runtime",),
        phase=phase,
    )
    assignment = ResolvedInsightAssignment.resolve(
        credit_unit_id=ids.new_operator_invocation_id(),
        snapshot=snapshot,
        expected_snapshot_sha256=snapshot.snapshot_sha256,
        block_id=f"{name}.block.1",
        arm=MemoryAssignmentArm.DIAGNOSTIC,
        selection_decision=decision,
        prompt_shape_sha256=engine.prompt_shape_commitment(
            unassigned_plan,
            selected_insight_count=len(decision.selected),
        ),
    )
    outcome = (
        await engine.run_invocations(
            (
                replace(
                    unassigned_plan,
                    resolved_insight_assignment=assignment,
                ),
            )
        )
    )[0]
    return memory, traces, assignment, outcome


def test_resolved_causal_assignment_is_plan_bound_and_credit_is_wave_deferred() -> None:
    async def scenario():
        module = _load_pipeline_module()
        ids = DeterministicIdFactory("resolved_causal_engine")
        memory = InsightMemoryBank(id_factory=ids)
        _seed_insights(memory)
        traces: list[dict[str, object]] = []
        generator = _FakeAgenticGenerator(module)
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=generator,
            id_factory=ids,
            memory=memory,
            seed=17,
            trace_sink=traces.append,
        )
        seed = await engine.register_seed(module.BASE_CONFIG, label="base")
        phase = "resolved_causal_engine_test"
        context = context_stratum_hash(
            problem_id=engine.problem_id,
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            phase=phase,
        )
        eligible = memory.eligible_references(
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            editable_paths=("$.runtime",),
        )
        snapshot = CausalSearchScorePolicy().genesis(
            exact_context_hash=context,
            estimand_stratum_hash=hashlib.sha256(
                b"resolved-causal-engine-estimand-v1"
            ).hexdigest(),
            priors={reference: 0.0 for reference in eligible},
        )
        decision = EpsilonGreedySubsetSelector(Fraction(1, 2)).select(
            context_hash=context,
            eligible=eligible,
            scores=snapshot.retrieval_scores,
            subset_size=1,
            rng=random.Random(9),
        )
        unassigned_plan = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (seed,),
            generation=1,
            label="resolved_diagnostic",
            allowed_top_level=("runtime",),
            phase=phase,
        )
        credit_unit_id = ids.new_operator_invocation_id()
        assignment = ResolvedInsightAssignment.resolve(
            credit_unit_id=credit_unit_id,
            snapshot=snapshot,
            expected_snapshot_sha256=snapshot.snapshot_sha256,
            block_id="diagnostic.block.1",
            arm=MemoryAssignmentArm.DIAGNOSTIC,
            selection_decision=decision,
            prompt_shape_sha256=engine.prompt_shape_commitment(
                unassigned_plan,
                selected_insight_count=len(decision.selected),
            ),
        )
        reservation_before = frozenset(engine._reserved_operator_invocation_ids)
        sequence_before = engine._proposal_sequence
        bad_assignment = replace(assignment, prompt_shape_sha256="0" * 64)
        with pytest.raises(ValueError, match="prompt-shape commitment differs"):
            engine.prepare_invocations(
                (
                    replace(
                        unassigned_plan,
                        resolved_insight_assignment=bad_assignment,
                    ),
                )
            )
        assert frozenset(engine._reserved_operator_invocation_ids) == reservation_before
        assert engine._proposal_sequence == sequence_before
        assert generator.propose_calls == 0
        outcome = (
            await engine.run_invocations(
                (
                    replace(
                        unassigned_plan,
                        resolved_insight_assignment=assignment,
                    ),
                )
            )
        )[0]
        return memory, traces, assignment, outcome

    memory, traces, assignment, outcome = asyncio.run(scenario())

    assert outcome.prepared.operator_invocation_id == assignment.credit_unit_id
    assert outcome.prepared.selection_decision == assignment.selection_decision
    assert outcome.prepared.plan.resolved_insight_assignment == assignment
    assert memory.trials == ()
    prepared = next(
        event for event in traces if event["event_type"] == "invocation_prepared"
    )
    assert prepared["resolved_insight_assignment_sha256"] == (
        assignment.assignment_sha256
    )
    deferred = next(
        event for event in traces if event["event_type"] == "insight_credit_deferred"
    )
    assert deferred["assignment_sha256"] == assignment.assignment_sha256
    assert deferred["failure_stage"] is None
    assert deferred["reward"] == outcome.reward
    committed = next(
        event for event in traces if event["event_type"] == "assignment_committed"
    )
    assert committed["assignment_sha256"] == assignment.assignment_sha256
    assert committed["assignment"] == assignment.to_record()
    assert committed["operator_invocation_id"] == assignment.credit_unit_id.value
    assert (
        committed["prepared_prompt_sha256"]
        == hashlib.sha256(outcome.prepared.prompt.encode("utf-8")).hexdigest()
    )
    assert committed["prompt_shape_commitment_verified"] is True
    assert committed["prompt_shape_policy"] == {
        "policy_id": "treatment_blinded_prompt_shape",
        "policy_version": 3,
        "renderer_policy_id": "default_evidence_prompt",
        "renderer_policy_version": 3,
    }
    terminal = next(
        event for event in traces if event["event_type"] == "trial_terminal"
    )
    assert terminal["assignment_sha256"] == assignment.assignment_sha256
    assert terminal["terminal_status"] == "succeeded"
    assert terminal["observed_reward"] == outcome.reward
    assert terminal["engine_terminal_reward"] == outcome.reward
    assert terminal["reward_disposition"] == "observed"
    assert terminal["candidate_ids"] == [outcome.candidate.candidate_id.value]
    # The staged-memory receipt constructor is the authoritative status/candidate
    # contract.  Constructing it here prevents the trace adapter from silently
    # drifting away from checkpoint closure semantics.
    receipt = MemoryAssignmentReceipt(
        assignment_sha256=terminal["assignment_sha256"],
        credit_unit_id=assignment.credit_unit_id,
        status=MemoryTrialTerminalStatus(terminal["terminal_status"]),
        candidate_ids=(outcome.candidate.candidate_id,),
        observed_reward=terminal["observed_reward"],
    )
    assert receipt.status is MemoryTrialTerminalStatus.SUCCEEDED
    completed = next(
        event for event in traces if event["event_type"] == "invocation_completed"
    )
    assert completed["insight_credit_status"] == "deferred_wave_sealed_itt"
    event_types = [event["event_type"] for event in traces]
    assert (
        event_types.index("invocation_prepared")
        < event_types.index("assignment_committed")
        < event_types.index("llm_call_completed")
    )
    assert event_types.index("trial_terminal") < event_types.index(
        "invocation_completed"
    )


@pytest.mark.parametrize(
    ("error", "expected_stage", "expected_status", "expected_disposition"),
    (
        (
            StructuredGenerationError(
                kind=GenerationFailureKind.OUTPUT_INVALID,
                retryable=False,
                safe_message="typed model output failed validation",
            ),
            "llm",
            "model_or_schema_failure",
            "impute_wave_no_yield_at_seal",
        ),
        (
            RuntimeError("untyped provider or application failure"),
            "infrastructure",
            "infrastructure_failure",
            "invalidates_block",
        ),
        (
            StructuredGenerationError(
                kind=GenerationFailureKind.RATE_LIMITED,
                retryable=False,
                safe_message="provider request budget was exhausted",
                status_code=429,
            ),
            "infrastructure",
            "infrastructure_failure",
            "invalidates_block",
        ),
        (
            StructuredGenerationError(
                kind=GenerationFailureKind.CONTENT_REJECTED,
                retryable=False,
                safe_message="model rejected the treatment-bearing prompt",
            ),
            "llm",
            "model_or_schema_failure",
            "impute_wave_no_yield_at_seal",
        ),
    ),
)
def test_resolved_generation_failures_use_explicit_provider_neutral_taxonomy(
    error, expected_stage, expected_status, expected_disposition
) -> None:
    class FailingGenerator(_FakeAgenticGenerator):
        async def propose(self, request):
            del request
            raise error

    memory, traces, assignment, outcome = asyncio.run(
        _resolved_trace_failure_case(
            name=(
                "resolved_failure_"
                + hashlib.sha256(
                    f"{expected_status}:{error!r}".encode("utf-8")
                ).hexdigest()[:8]
            ),
            generator_factory=FailingGenerator,
        )
    )

    assert memory.trials == ()
    assert outcome.candidate is None
    assert outcome.failure_stage == expected_stage
    terminal = next(
        event for event in traces if event["event_type"] == "trial_terminal"
    )
    assert terminal["assignment_sha256"] == assignment.assignment_sha256
    assert terminal["terminal_status"] == expected_status
    assert terminal["observed_reward"] is None
    assert terminal["engine_terminal_reward"] == -1.0
    assert terminal["reward_disposition"] == expected_disposition
    assert terminal["candidate_ids"] == []
    assert any(event["event_type"] == "llm_call_failed" for event in traces) is (
        expected_stage == "llm"
    )
    assert any(
        event["event_type"] == "infrastructure_boundary_failed" for event in traces
    ) is (expected_stage == "infrastructure")


def test_failure_reward_is_owned_by_the_active_reward_binding() -> None:
    class FailingGenerator(_FakeAgenticGenerator):
        async def propose(self, request):
            del request
            raise StructuredGenerationError(
                kind=GenerationFailureKind.OUTPUT_INVALID,
                retryable=False,
                safe_message="typed model output failed validation",
            )

    memory, traces, assignment, outcome = asyncio.run(
        _resolved_trace_failure_case(
            name="resolved_custom_failure_score",
            generator_factory=FailingGenerator,
            failure_score=-2.0,
        )
    )

    assert memory.trials == ()
    assert outcome.reward == -2.0
    terminal = next(
        event for event in traces if event["event_type"] == "trial_terminal"
    )
    assert terminal["assignment_sha256"] == assignment.assignment_sha256
    assert terminal["engine_terminal_reward"] == -2.0
    binding = next(
        event for event in traces if event["event_type"] == "reward_binding_committed"
    )
    assert binding["reward_binding"]["failure_score_hex"] == (-2.0).hex()
    assert len(binding["reward_binding_sha256"]) == 64


def test_resolved_evaluator_and_reward_failures_emit_infrastructure_terminals() -> None:
    class SecondEvaluationFails:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.candidate_model = delegate.candidate_model
            self.evaluations = 0

        @property
        def objectives(self):
            return self.delegate.objectives

        def search_space_description(self):
            return self.delegate.search_space_description()

        def validate(self, configuration):
            return self.delegate.validate(configuration)

        def evaluate(self, configuration):
            self.evaluations += 1
            if self.evaluations > 1:
                raise RuntimeError("offline evaluator infrastructure failed")
            return self.delegate.evaluate(configuration)

    evaluator_result = asyncio.run(
        _resolved_trace_failure_case(
            name="resolved_evaluator_infrastructure",
            generator_factory=_FakeAgenticGenerator,
            problem_factory=lambda module: SecondEvaluationFails(
                module.PipelineCoDesignProblem()
            ),
        )
    )

    def broken_reward(child, parents, objectives):
        del child, parents, objectives
        raise RuntimeError("offline reward infrastructure failed")

    reward_result = asyncio.run(
        _resolved_trace_failure_case(
            name="resolved_reward_infrastructure",
            generator_factory=_FakeAgenticGenerator,
            reward_policy=broken_reward,
        )
    )

    for index, (_, traces, assignment, outcome) in enumerate(
        (evaluator_result, reward_result)
    ):
        assert outcome.failure_stage == "infrastructure"
        assert outcome.reward == -1.0
        terminal = next(
            event for event in traces if event["event_type"] == "trial_terminal"
        )
        assert terminal["assignment_sha256"] == assignment.assignment_sha256
        assert terminal["terminal_status"] == "infrastructure_failure"
        assert terminal["reward_disposition"] == "invalidates_block"
        assert terminal["observed_reward"] is None
        assert outcome.candidate is None
        if index == 0:
            assert terminal["candidate_ids"] == []
        else:
            assert len(terminal["candidate_ids"]) == 1


def test_custom_prompt_renderer_requires_an_explicit_nondefault_shape_pairing() -> None:
    def custom_prompt(problem_description, prepared, selected_records):
        del problem_description, prepared, selected_records
        return "custom prompt"

    async def scenario():
        module = _load_pipeline_module()
        ids = DeterministicIdFactory("shape_pair_a")
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=_FakeAgenticGenerator(module),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=17,
            prompt_builder=custom_prompt,
        )
        parent = await engine.register_seed(module.BASE_CONFIG, label="base")
        plan = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=1,
            label="custom_prompt",
            allowed_top_level=("runtime",),
            phase="custom_prompt_shape",
        )
        with pytest.raises(ValueError, match="requires an explicit"):
            engine.prompt_shape_commitment(plan, selected_insight_count=1)

    asyncio.run(scenario())

    module = _load_pipeline_module()
    ids = DeterministicIdFactory("shape_pair_b")
    with pytest.raises(ValueError, match="renderer identity do not match"):
        AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=_FakeAgenticGenerator(module),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=17,
            prompt_builder=custom_prompt,
            prompt_shape_commitment_policy=DefaultEvidencePromptShapePolicyV1(),
        )


def test_exact_crossover_prompt_shape_binds_contract_identity_end_to_end() -> None:
    async def scenario() -> tuple[str, str, str]:
        module = _load_pipeline_module()
        ids = DeterministicIdFactory("exact_xover_shape")
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=_FakeAgenticGenerator(module),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=17,
        )
        base = await engine.register_seed(
            module.DEVELOPMENT_BRANCH_LEFT,
            label="base",
        )
        donor = await engine.register_seed(
            module.DEVELOPMENT_BRANCH_RIGHT,
            label="donor",
        )
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

        def plan(contract=None) -> InvocationPlan:
            return InvocationPlan(
                operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
                parents=(base, donor),
                generation=1,
                label="exact_crossover_prompt_shape",
                phase="exact_crossover_prompt_shape",
                crossover_response_mode=(
                    CrossoverResponseMode.FULL_CONFIGURATION
                    if contract is None
                    else CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
                ),
                exact_parent_crossover_contract=contract,
            )

        return (
            engine.prompt_shape_commitment(
                plan(),
                selected_insight_count=0,
            ),
            engine.prompt_shape_commitment(
                plan(default_contract),
                selected_insight_count=0,
            ),
            engine.prompt_shape_commitment(
                plan(wide_contract),
                selected_insight_count=0,
            ),
        )

    full, exact_default, exact_wide = asyncio.run(scenario())

    assert full != exact_default
    assert exact_default != exact_wide


def test_exact_crossover_accepts_permutation_parent_effects_from_core_receipt() -> None:
    class PermutationConfiguration(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

        arr: list[int]
        other: int

    class PermutationProblem:
        candidate_model = PermutationConfiguration
        constraints_description = "Integer arrays and one independent scalar."

        @property
        def objectives(self):
            return (ObjectiveSpec("score", "min"),)

        def search_space_description(self):
            return "Choose integer array entries and an independent scalar."

        def validate(self, configuration):
            PermutationConfiguration.model_validate(configuration, strict=True)
            return True

        def evaluate(self, configuration):
            parsed = PermutationConfiguration.model_validate(
                configuration,
                strict=True,
            )
            return {"score": float(sum(parsed.arr) + parsed.other)}

    class ExactSelector:
        import_locus_id: str | None = None

        async def propose(self, request):
            contract = request.exact_parent_crossover_contract
            assert contract is not None
            assert self.import_locus_id in contract.locus_ids
            return VariationGenerationResult(
                draft=ExactParentCrossoverDraft(
                    contract_identity_sha256=contract.contract_identity_sha256,
                    import_locus_ids=(self.import_locus_id,),
                ),
                telemetry=_telemetry(),
            )

        async def reflect(self, request):
            del request
            return ReflectionGenerationResult(insights=(), telemetry=_telemetry())

    async def scenario():
        ids = DeterministicIdFactory("permutation_exact_crossover")
        traces: list[dict[str, object]] = []
        selector = ExactSelector()
        engine = AgenticEvolutionEngine(
            problem=PermutationProblem(),
            generator=selector,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=17,
            trace_sink=traces.append,
        )
        base = await engine.register_seed(
            {"arr": [1, 2, 3], "other": 0},
            label="base",
        )
        donor = await engine.register_seed(
            {"arr": [3, 2, 1], "other": 1},
            label="donor",
        )
        contract = derive_exact_parent_crossover_contract(
            base=base.configuration,
            donor=donor.configuration,
        )
        selector.import_locus_id = next(
            locus.locus_id
            for locus in contract.loci
            if locus.path_text == '$["arr"][0]'
        )
        outcome = (
            await engine.run_invocations(
                (
                    InvocationPlan(
                        operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
                        parents=(base, donor),
                        generation=1,
                        label="permutation_exact_crossover",
                        phase="permutation_exact_crossover",
                        crossover_response_mode=(
                            CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
                        ),
                        exact_parent_crossover_contract=contract,
                    ),
                )
            )
        )[0]
        return outcome, traces, contract, selector.import_locus_id

    outcome, traces, contract, imported_locus_id = asyncio.run(scenario())

    assert outcome.failure_stage is None, (
        outcome.call_failure_type,
        traces[-5:],
    )
    assert outcome.candidate is not None
    assert outcome.candidate.configuration_dict == {
        "arr": [3, 2, 3],
        "other": 0,
    }
    assert outcome.candidate.operator_compliant is True
    assert outcome.candidate.evidence_compliant is True
    assert {value.source for value in outcome.candidate.source_attribution} == {
        "left",
        "right",
    }
    assert tuple(
        (value.path, value.source) for value in outcome.candidate.source_attribution
    ) == tuple(
        (
            locus.path_text,
            "right" if locus.locus_id == imported_locus_id else "left",
        )
        for locus in contract.loci
    )
    evaluated = next(
        row for row in traces if row["event_type"] == "candidate_evaluated"
    )
    assert evaluated["source_attribution_provenance"] == (
        "engine_derived_exact_parent_import"
    )


def test_prepare_invocations_is_no_io_and_exactly_matches_live_preparation() -> None:
    class CountingProblem:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.candidate_model = delegate.candidate_model
            self.evaluations = 0

        @property
        def objectives(self):
            return self.delegate.objectives

        def search_space_description(self):
            return self.delegate.search_space_description()

        def validate(self, config):
            return self.delegate.validate(config)

        def evaluate(self, config):
            self.evaluations += 1
            return self.delegate.evaluate(config)

    async def scenario():
        module = _load_pipeline_module()

        def build():
            ids = DeterministicIdFactory("prepare_invocations_equivalence")
            problem = CountingProblem(module.PipelineCoDesignProblem())
            generator = _FakeAgenticGenerator(module)
            engine = AgenticEvolutionEngine(
                problem=problem,
                generator=generator,
                id_factory=ids,
                memory=InsightMemoryBank(id_factory=ids),
                seed=17,
            )
            return engine, problem, generator

        preview_engine, preview_problem, preview_generator = build()
        preview_seed = await preview_engine.register_seed(
            module.BASE_CONFIG,
            label="base",
        )
        preview_plan = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (preview_seed,),
            generation=1,
            label="prepared_mutation",
            allowed_top_level=("passes", "frontend", "backend"),
            phase="prepare_equivalence",
        )
        prepared, binding = preview_engine.prepare_invocations((preview_plan,))
        assert preview_problem.evaluations == 1
        assert preview_generator.propose_calls == 0

        live_engine, live_problem, live_generator = build()
        live_seed = await live_engine.register_seed(module.BASE_CONFIG, label="base")
        live_plan = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (live_seed,),
            generation=1,
            label="prepared_mutation",
            allowed_top_level=("passes", "frontend", "backend"),
            phase="prepare_equivalence",
        )
        outcome = (await live_engine.run_invocations((live_plan,)))[0]
        return (
            prepared[0],
            binding,
            outcome.prepared,
            preview_problem,
            preview_generator,
            live_problem,
            live_generator,
        )

    (
        preview,
        binding,
        live,
        preview_problem,
        preview_generator,
        live_problem,
        live_generator,
    ) = asyncio.run(scenario())
    assert binding.definition_hash == REWARD_DEFINITION_HASH
    assert preview.plan == live.plan
    assert preview.operator_invocation_id == live.operator_invocation_id
    assert preview.call_id == live.call_id
    assert preview.prompt == live.prompt
    assert preview.variation_case == live.variation_case
    assert preview_problem.evaluations == 1
    assert preview_generator.propose_calls == 0
    assert live_problem.evaluations == 2
    assert live_generator.propose_calls == 1


def test_custom_reward_requires_and_propagates_its_own_definition_hash() -> None:
    async def scenario():
        module = _load_pipeline_module()
        custom_hash = hashlib.sha256(
            b"test:frozen-archive-marginal-reward:v1"
        ).hexdigest()

        def custom_reward(child, parents, objectives):
            del child, parents, objectives
            return 0.25

        def build(**overrides):
            ids = DeterministicIdFactory("custom_reward_identity")
            arguments = {
                "problem": module.PipelineCoDesignProblem(),
                "generator": _FakeAgenticGenerator(module),
                "id_factory": ids,
                "memory": InsightMemoryBank(id_factory=ids),
                "seed": 3,
            }
            arguments.update(overrides)
            return AgenticEvolutionEngine(**arguments)

        with pytest.raises(ValueError, match="same reward semantics"):
            build(reward_policy=custom_reward)
        with pytest.raises(ValueError, match="same reward semantics"):
            build(reward_definition_hash=custom_hash)
        with pytest.raises(ValueError, match="SHA-256"):
            build(
                reward_policy=custom_reward,
                reward_definition_hash="not-a-hash",
            )

        ids = DeterministicIdFactory("custom_reward_propagation")
        traces = []
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=_FakeAgenticGenerator(module),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=3,
            trace_sink=traces.append,
            reward_policy=custom_reward,
            reward_definition_hash=custom_hash,
        )
        seed = await engine.register_seed(module.BASE_CONFIG, label="base")
        outcome = (
            await engine.run_invocations(
                (
                    InvocationPlan(
                        OperatorKind.REPRODUCTION,
                        (seed,),
                        generation=1,
                        label="custom_reward_control",
                    ),
                )
            )
        )[0]
        default_ids = DeterministicIdFactory("per_wave_reward_override")
        override_traces = []
        default_engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=_FakeAgenticGenerator(module),
            id_factory=default_ids,
            memory=InsightMemoryBank(id_factory=default_ids),
            seed=4,
            trace_sink=override_traces.append,
        )
        default_seed = await default_engine.register_seed(
            module.BASE_CONFIG,
            label="default_base",
        )
        override_outcome = (
            await default_engine.run_invocations(
                (
                    InvocationPlan(
                        OperatorKind.REPRODUCTION,
                        (default_seed,),
                        generation=1,
                        label="wave_override",
                    ),
                ),
                reward_binding=RewardPolicyBinding(custom_reward, custom_hash),
            )
        )[0]
        return (
            custom_hash,
            engine,
            traces,
            outcome,
            default_engine,
            override_traces,
            override_outcome,
        )

    (
        custom_hash,
        engine,
        traces,
        outcome,
        default_engine,
        override_traces,
        override_outcome,
    ) = asyncio.run(scenario())
    assert custom_hash != REWARD_DEFINITION_HASH
    assert engine.reward_definition_hash == custom_hash
    assert outcome.reward == 0.25
    assert outcome.prepared.variation_case.reward_definition_hash == custom_hash
    completed = next(
        event for event in traces if event["event_type"] == "invocation_completed"
    )
    assert completed["scalar_reward_definition_sha256"] == custom_hash
    assert default_engine.reward_definition_hash == REWARD_DEFINITION_HASH
    assert override_outcome.reward == 0.25
    assert (
        override_outcome.prepared.variation_case.reward_definition_hash == custom_hash
    )
    override_completed = next(
        event
        for event in override_traces
        if event["event_type"] == "invocation_completed"
    )
    assert override_completed["scalar_reward_definition_sha256"] == custom_hash


def test_paired_operator_batch_verifies_lineage_memory_and_ignored_parent() -> None:
    async def scenario():
        module = _load_pipeline_module()
        ids = DeterministicIdFactory("agentic_engine")
        memory = InsightMemoryBank(
            id_factory=ids,
            exploration_probability=Fraction(1, 2),
        )
        _seed_insights(memory)
        generator = _FakeAgenticGenerator(module)
        traces = []
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=generator,
            id_factory=ids,
            memory=memory,
            seed=7,
            trace_sink=traces.append,
        )
        base, left, right = await asyncio.gather(
            engine.register_seed(module.BASE_CONFIG, label="base"),
            engine.register_seed(module.DEVELOPMENT_BRANCH_LEFT, label="left"),
            engine.register_seed(module.DEVELOPMENT_BRANCH_RIGHT, label="right"),
        )
        outcomes = await engine.run_invocations(
            (
                InvocationPlan(
                    OperatorKind.TWO_PARENT_CROSSOVER,
                    (left, right),
                    generation=1,
                    label="ignored_parent",
                ),
                InvocationPlan(
                    OperatorKind.THREE_WAY_RECOMBINATION,
                    (left, right),
                    generation=1,
                    label="recombine_no_memory",
                    common_ancestor=base,
                ),
                InvocationPlan(
                    OperatorKind.THREE_WAY_RECOMBINATION,
                    (left, right),
                    generation=1,
                    label="recombine_memory",
                    common_ancestor=base,
                    use_memory=True,
                    memory_subset_size=1,
                    memory_exploration_probability=Fraction(1, 1),
                    memory_score_phase="discovery_generation",
                    phase="scored_generation",
                ),
                InvocationPlan(
                    OperatorKind.REPRODUCTION,
                    (left,),
                    generation=1,
                    label="reproduction",
                ),
            )
        )
        added = await engine.reflect(
            outcomes,
            label="post_probe",
            max_insights=2,
        )
        return module, generator, memory, traces, (base, left, right), outcomes, added

    module, generator, memory, traces, seeds, outcomes, added = asyncio.run(scenario())
    base, left, right = seeds
    ignored, no_memory, with_memory, reproduction = outcomes

    assert generator.propose_calls == 3
    assert generator.max_active >= 2
    assert all(
        "INVOCATION LABEL:" not in outcome.prepared.prompt for outcome in outcomes
    )
    assert "ignored_parent" not in ignored.prepared.prompt
    assert "recombine_no_memory" not in no_memory.prepared.prompt
    assert "recombine_memory" not in with_memory.prepared.prompt
    for outcome in outcomes:
        assert (
            "preservation claims, and conflict resolutions in their typed fields"
            not in outcome.prepared.prompt
        )
        assert (
            "Emit only fields present in the supplied output schema"
            in outcome.prepared.prompt
        )
    for outcome in (ignored, reproduction):
        assert "PRESERVATION OBLIGATIONS" not in outcome.prepared.prompt
        assert "conflict_resolutions" not in outcome.prepared.prompt
        assert "Do not echo preservation IDs" not in outcome.prepared.prompt
    for outcome in (no_memory, with_memory):
        assert "PRESERVATION OBLIGATIONS" in outcome.prepared.prompt
        assert "conflict_resolutions MUST be an empty list" in outcome.prepared.prompt
        assert "Do not echo preservation IDs" in outcome.prepared.prompt
    assert ignored.candidate is None
    assert ignored.failure_stage == "candidate"
    assert ignored.call_failure_type == "ValueError"
    assert ignored.reward == -1.0

    for outcome in (no_memory, with_memory):
        assert outcome.candidate is not None
        assert (
            outcome.candidate.configuration_dict
            == module.DEVELOPMENT_RECOMBINATION_TARGET
        )
        assert outcome.candidate.valid
        assert outcome.candidate.operator_compliant
        assert outcome.candidate.preservation_verified is True
        assert outcome.reward > 0
        assert outcome.dominates_any_parent
        assert len(outcome.prepared.variation_case.preservation_obligations) == 8

    assert len(memory.trials) == 1
    assert (
        memory.trials[0].credit_unit_id == with_memory.prepared.operator_invocation_id
    )
    assert memory.trials[0].candidate_ids == (with_memory.candidate.candidate_id,)
    problem_id = (
        f"{module.PipelineCoDesignProblem.__module__}."
        f"{module.PipelineCoDesignProblem.__qualname__}"
    )
    assert memory.trials[0].decision.context_hash == context_stratum_hash(
        problem_id=problem_id,
        operator_kind=OperatorKind.THREE_WAY_RECOMBINATION.value,
        phase="scored_generation",
    )
    assert memory.trials[0].decision.exploration_probability == Fraction(1, 1)
    assert len(with_memory.candidate.selected_insight_ids) == 1

    assert reproduction.candidate is not None
    assert reproduction.candidate.configuration_dict == left.configuration_dict
    assert reproduction.candidate.candidate_id != left.candidate_id
    assert reproduction.candidate.operator_compliant
    assert reproduction.reward == 0.0

    event_types = [event["event_type"] for event in traces]
    assert event_types.count("invocation_prepared") == 4
    assert event_types.count("candidate_evaluated") == 3
    assert event_types.count("candidate_boundary_failed") == 1
    assert event_types.count("invocation_completed") == 4
    assert "insight_credit_updated" in event_types
    memory_prepared = next(
        event
        for event in traces
        if event["event_type"] == "invocation_prepared"
        and event["label"] == "recombine_memory"
    )
    assert memory_prepared["score_context_hash"] == context_stratum_hash(
        problem_id=problem_id,
        operator_kind=OperatorKind.THREE_WAY_RECOMBINATION.value,
        phase="discovery_generation",
    )
    assert memory_prepared["exploration_probability"] == {
        "numerator": 1,
        "denominator": 1,
    }
    decision = memory.trials[0].decision
    assert memory_prepared["selection_decision"] == {
        "context_hash": decision.context_hash,
        "eligible": [
            {
                "insight_id": reference.insight_id.value,
                "version": reference.version,
            }
            for reference in decision.eligible
        ],
        "selected": [
            {
                "insight_id": reference.insight_id.value,
                "version": reference.version,
            }
            for reference in decision.selected
        ],
        "exploitation_subset": [
            {
                "insight_id": reference.insight_id.value,
                "version": reference.version,
            }
            for reference in decision.exploitation_subset
        ],
        "score_snapshot": [
            {
                "insight_id": reference.insight_id.value,
                "version": reference.version,
                "score": score,
            }
            for reference, score in decision.score_snapshot
        ],
        "subset_size": decision.subset_size,
        "exploration_probability": {"numerator": 1, "denominator": 1},
        "mode": decision.mode.value,
        "selected_subset_probability": {
            "numerator": decision.selected_subset_probability.numerator,
            "denominator": decision.selected_subset_probability.denominator,
        },
        "policy_id": decision.policy_id,
        "policy_version": decision.policy_version,
    }
    no_memory_prepared = next(
        event
        for event in traces
        if event["event_type"] == "invocation_prepared"
        and event["label"] == "recombine_no_memory"
    )
    assert no_memory_prepared["selection_decision"] is None
    reflection_prompt = next(
        event["prompt"]
        for event in traces
        if event["event_type"] == "reflection_requested"
    )
    assert '"parents"' in reflection_prompt
    assert '"scalar_reward"' in reflection_prompt
    assert '"dominates_any_parent"' in reflection_prompt
    assert all(event["sequence"] == index for index, event in enumerate(traces, 1))

    reflection_completed = next(
        event for event in traces if event["event_type"] == "reflection_completed"
    )
    assert {
        name: reflection_completed[name]
        for name in (
            "requested_model",
            "resolved_model",
            "resolved_provider",
            "provider_response_id",
            "finish_reason",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cost_usd",
            "provider_latency_ns",
            "attempt_count",
        )
    } == {
        "requested_model": "offline/fake",
        "resolved_model": "offline/fake",
        "resolved_provider": "fake-provider",
        "provider_response_id": "fake-response",
        "finish_reason": "stop",
        "input_tokens": 100,
        "output_tokens": 50,
        "reasoning_tokens": 0,
        "cost_usd": "0",
        "provider_latency_ns": 1_000,
        "attempt_count": 1,
    }

    assert len(added) == 1
    assert added[0].initial_score == 0.0
    assert added[0].origin is InsightOrigin.REFLECTION
    assert added[0].lifecycle_state is InsightLifecycleState.QUARANTINED
    assert added[0].retrievable is False
    assert added[0].evidence_lineage is not None
    assert (
        added[0].evidence_lineage.reflection_call_id.value
        == (reflection_completed["call_id"])
    )
    cited_contrast = added[0].evidence_lineage.cited_contrast_ids[0]
    cited_outcome = next(
        outcome
        for outcome in outcomes
        if any(
            hashlib.sha256(
                b"agent-evolve:reflection-contrast:v1\x00"
                + outcome.prepared.operator_invocation_id.value.encode("ascii")
                + b"\x00"
                + parent.candidate_id.value.encode("ascii")
            ).hexdigest()
            == cited_contrast
            for parent in outcome.prepared.plan.parents
        )
    )
    assert added[0].evidence_lineage.source_operator_invocation_ids == (
        cited_outcome.prepared.operator_invocation_id,
    )
    assert len(added[0].evidence_lineage.available_contrast_ids) == sum(
        len(outcome.prepared.plan.parents)
        for outcome in outcomes
        if outcome.candidate is not None
    )
    assert added[0].evidence_lineage.cited_contrast_ids == (
        added[0].evidence_lineage.available_contrast_ids[0],
    )
    assert added[0].draft.evidence_contrast_ids == (
        added[0].evidence_lineage.available_contrast_ids[0],
    )
    assert reflection_completed["insights"][0]["evidence_contrast_ids"] == [
        added[0].evidence_lineage.available_contrast_ids[0]
    ]
    assert reflection_completed["insights"][0]["lifecycle_state"] == "quarantined"
    assert reflection_completed["insights"][0]["retrievable"] is False
    assert len(memory.entries) == 3


def test_typed_mutation_scope_escape_is_evaluated_but_not_credited() -> None:
    async def scenario():
        module = _load_pipeline_module()
        ids = DeterministicIdFactory("scope_engine")
        memory = InsightMemoryBank(id_factory=ids)
        generator = _FakeAgenticGenerator(module)
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=generator,
            id_factory=ids,
            memory=memory,
            seed=11,
        )
        base = await engine.register_seed(module.BASE_CONFIG, label="base")
        return (
            await engine.run_invocations(
                (
                    InvocationPlan(
                        OperatorKind.TYPED_MUTATION,
                        (base,),
                        generation=1,
                        label="scope_escape",
                        allowed_top_level=("runtime",),
                    ),
                )
            )
        )[0]

    outcome = asyncio.run(scenario())
    assert outcome.candidate is not None
    assert outcome.candidate.valid
    assert outcome.candidate.operator_compliant is False
    assert (
        outcome.candidate.operator_failure
        == "mutation escaped its declared top-level scope"
    )
    assert outcome.reward == -1.0
    assert "PRESERVATION OBLIGATIONS" not in outcome.prepared.prompt
    assert "conflict_resolutions" not in outcome.prepared.prompt
    assert "Do not echo preservation IDs" not in outcome.prepared.prompt


def test_candidate_evaluations_remain_concurrent_after_generation() -> None:
    class DistinctRuntimeGenerator(_FakeAgenticGenerator):
        def __init__(self, module) -> None:
            super().__init__(module)
            self._runtimes = iter(
                (
                    {"threads": 2, "prefetch_distance": 0, "data_layout": "aos"},
                    {"threads": 4, "prefetch_distance": 4, "data_layout": "blocked"},
                    {"threads": 8, "prefetch_distance": 4, "data_layout": "soa"},
                    {"threads": 1, "prefetch_distance": 2, "data_layout": "soa"},
                )
            )

        async def propose(self, request):
            del request
            config = copy.deepcopy(self.module.BASE_CONFIG)
            config["runtime"] = next(self._runtimes)
            return VariationGenerationResult(
                draft=CandidateDraft(
                    configuration=config,
                    design_rationale="Distinct scoped candidate for concurrency coverage.",
                    source_attribution=(
                        SourceAttribution(path="$.runtime", source="mutation"),
                    ),
                ),
                telemetry=_telemetry(),
            )

    async def scenario():
        module = _load_pipeline_module()
        problem = _SlowEvaluationProblem(module.PipelineCoDesignProblem())
        ids = DeterministicIdFactory("evaluation_concurrency")
        memory = InsightMemoryBank(id_factory=ids)
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=DistinctRuntimeGenerator(module),
            id_factory=ids,
            memory=memory,
            seed=17,
            evaluator_concurrency=4,
        )
        parent = await engine.register_seed(module.BASE_CONFIG, label="base")
        problem.max_active = 0
        outcomes = await engine.run_invocations(
            tuple(
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (parent,),
                    generation=1,
                    label=f"mutation_{index}",
                    allowed_top_level=("runtime",),
                )
                for index in range(4)
            )
        )
        return problem.max_active, outcomes

    max_active, outcomes = asyncio.run(scenario())
    assert max_active >= 2
    assert all(outcome.candidate is not None for outcome in outcomes)


def test_duplicate_candidate_evaluations_reuse_run_local_cache() -> None:
    async def scenario():
        module = _load_pipeline_module()
        problem = _SlowEvaluationProblem(module.PipelineCoDesignProblem())
        ids = DeterministicIdFactory("evaluation_cache_engine")
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=_FakeAgenticGenerator(module),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=17,
            evaluator_concurrency=4,
        )
        parent = await engine.register_seed(module.BASE_CONFIG, label="base")
        problem.max_active = 0
        outcomes = await engine.run_invocations(
            tuple(
                InvocationPlan(
                    OperatorKind.REPRODUCTION,
                    (parent,),
                    generation=1,
                    label=f"reproduction_{index}",
                )
                for index in range(4)
            )
        )
        return problem.max_active, outcomes, await engine.evaluation_cache_snapshot()

    max_active, outcomes, snapshot = asyncio.run(scenario())
    assert max_active == 0
    assert all(outcome.candidate is not None for outcome in outcomes)
    assert snapshot == {
        "capacity": None,
        "cached_entries": 1,
        "in_flight": 0,
        "hits": 4,
        "misses": 1,
        "coalesced": 0,
        "evictions": 0,
    }


def test_reflection_failure_is_traced_and_propagated() -> None:
    class FailingReflectionGenerator(_FakeAgenticGenerator):
        async def reflect(self, request):
            del request
            raise RuntimeError("offline reflection failure")

    async def scenario():
        module = _load_pipeline_module()
        ids = DeterministicIdFactory("reflection_failure")
        traces = []
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=FailingReflectionGenerator(module),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=19,
            trace_sink=traces.append,
        )
        with pytest.raises(RuntimeError, match="offline reflection failure"):
            await engine.reflect((), label="must_fail")
        return traces

    traces = asyncio.run(scenario())
    assert [event["event_type"] for event in traces] == [
        "reflection_requested",
        "reflection_failed",
    ]
    assert traces[-1]["failure_type"] == "RuntimeError"


def test_reflection_filters_foreign_structured_contrast_ids() -> None:
    foreign_id = "f" * 64

    class ForeignCitationGenerator(_FakeAgenticGenerator):
        async def reflect(self, request):
            assert request.available_contrast_ids == ()
            return ReflectionGenerationResult(
                insights=(
                    InsightDraft(
                        claim="A custom generator supplied a foreign citation.",
                        trigger="foreign evidence is submitted",
                        mechanism="the engine intersects against its evidence boundary",
                        affected_paths=("$.runtime.threads",),
                        evidence_summary=f"Narrative mentions {foreign_id}.",
                        confidence=0.5,
                        evidence_contrast_ids=(foreign_id,),
                    ),
                ),
                telemetry=_telemetry(),
            )

    async def scenario():
        module = _load_pipeline_module()
        ids = DeterministicIdFactory("foreign_reflection_citation")
        traces = []
        engine = AgenticEvolutionEngine(
            problem=module.PipelineCoDesignProblem(),
            generator=ForeignCitationGenerator(module),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=23,
            trace_sink=traces.append,
        )
        added = await engine.reflect((), label="foreign_citation")
        return traces, added

    traces, added = asyncio.run(scenario())
    assert len(added) == 1
    assert added[0].draft.evidence_contrast_ids == ()
    assert added[0].evidence_lineage is not None
    assert added[0].evidence_lineage.cited_contrast_ids == ()
    filtered = next(
        event
        for event in traces
        if event["event_type"] == "reflection_evidence_contrast_ids_filtered"
    )
    assert filtered["submitted_contrast_ids"] == [foreign_id]
    assert filtered["accepted_contrast_ids"] == []
    assert filtered["rejected_contrast_ids"] == [foreign_id]
    completed = next(
        event for event in traces if event["event_type"] == "reflection_completed"
    )
    assert completed["insights"][0]["evidence_contrast_ids"] == []


def test_recombination_must_preserve_every_feasible_branch_operation() -> None:
    async def scenario():
        module = _load_dag_module()
        partial = copy.deepcopy(module.BASE_CONFIG)
        partial["assignments"][1]["worker"] = "gpu"
        partial["dispatch_order"] = copy.deepcopy(
            module.DEVELOPMENT_BRANCH_RIGHT["dispatch_order"]
        )
        ids = DeterministicIdFactory("all_branch_effects")
        memory = InsightMemoryBank(id_factory=ids)
        engine = AgenticEvolutionEngine(
            problem=module.DagDispatchCoDesignProblem(),
            generator=_StaticCandidateGenerator(partial),
            id_factory=ids,
            memory=memory,
            seed=23,
        )
        base, left, right = await asyncio.gather(
            engine.register_seed(module.BASE_CONFIG, label="base"),
            engine.register_seed(module.DEVELOPMENT_BRANCH_LEFT, label="left"),
            engine.register_seed(module.DEVELOPMENT_BRANCH_RIGHT, label="right"),
        )
        outcome = (
            await engine.run_invocations(
                (
                    InvocationPlan(
                        OperatorKind.THREE_WAY_RECOMBINATION,
                        (left, right),
                        generation=1,
                        label="partial_branch_attack",
                        common_ancestor=base,
                    ),
                )
            )
        )[0]
        return outcome

    outcome = asyncio.run(scenario())
    assert len(outcome.prepared.variation_case.preservation_obligations) == 4
    assert outcome.candidate is not None and outcome.candidate.valid
    assert outcome.candidate.objective_map == {
        "makespan_ms": 62.0,
        "energy_mj": 74.8,
        "peak_worker_load_ms": 41.0,
    }
    assert outcome.candidate.operator_compliant is False
    assert outcome.candidate.preservation_verified is False
    assert outcome.candidate.operator_failure == (
        "child failed exact two-branch preservation verification"
    )
    assert outcome.reward == -1.0

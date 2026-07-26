from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal

import pytest
from pydantic import BaseModel

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    OperatorKind,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    OptimizerBudget,
)
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluationPayload,
    EvaluationCheck,
    EvaluationCheckStatus,
    EvaluatorIdentity,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.outcome_relation import (
    OutcomeRelation,
    OutcomeRelationPolicyBinding,
    objective_pareto_outcome_binding,
)
from agent_evolve.application.pareto_archive import (
    ParetoArchive,
    ParetoDecisionReason,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.artifact import artifact_ref_for_bytes
from agent_evolve.domain.outcome import FailureCategory, FailureCode, FailureRecord
from agent_evolve.domain.typed_json import freeze_json, thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.phenotype_recourse import (
    SemanticProjectionPhenotypeIdentityPolicy,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


class _Configuration(BaseModel):
    x: int
    alias: str


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="offline",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _Problem:
    candidate_model = _Configuration

    def __init__(self, objectives: tuple[ObjectiveSpec, ...]) -> None:
        self.objectives = objectives
        self.legacy_calls = 0
        self.raw_detailed_calls = 0

    @staticmethod
    def search_space_description() -> str:
        return "An integer design variable and evaluator-inert alias."

    def evaluate(self, configuration):
        del configuration
        self.legacy_calls += 1
        raise AssertionError("the evidence adapter must replace legacy evaluate")

    def evaluate_detailed(self, configuration):
        """An intentionally colliding legacy domain method returning a raw receipt."""

        self.raw_detailed_calls += 1
        return {"raw_domain_receipt": configuration}


def _receipt(configuration: dict[str, object], *, prefix: str):
    content = json.dumps(
        {"prefix": prefix, "x": configuration["x"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return artifact_ref_for_bytes(content, media_type="application/json")


class _ConstrainedEvidenceAdapter:
    evaluator_identity = EvaluatorIdentity(
        "constrained_probe",
        1,
        hashlib.sha256(b"constrained-context").hexdigest(),
    )

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        self.calls += 1
        x = configuration["x"]
        assert type(x) is int
        receipt = _receipt(configuration, prefix="constrained")
        if x < 0:
            category = (
                FailureCategory.CANDIDATE
                if x == -1
                else FailureCategory.INFRASTRUCTURE
            )
            code = (
                FailureCode.EVALUATOR_DECLARED_INFEASIBLE
                if category is FailureCategory.CANDIDATE
                else FailureCode.TIMEOUT_OR_RESOURCE_FAILURE
            )
            return DetailedEvaluationPayload(
                failure=FailureRecord(
                    category,
                    code,
                    "typed evaluator failure",
                    retryable=category is FailureCategory.INFRASTRUCTURE,
                ),
                objectives=(),
                violations=(),
                checks=(
                    EvaluationCheck(
                        "solver",
                        EvaluationCheckStatus.FAIL,
                        freeze_json({"completed": False}),
                        "$.checks.solver",
                    ),
                ),
                receipt=receipt,
                evaluator=self.evaluator_identity,
            )
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(("cost", float(x)),),
            violations=(("normalized_constraint", float(max(0, 2 - 2 * x))),),
            checks=(
                EvaluationCheck(
                    "geometry",
                    EvaluationCheckStatus.PASS,
                    freeze_json({"finite": True}),
                    "$.checks.geometry",
                ),
                EvaluationCheck(
                    "solver",
                    EvaluationCheckStatus.PASS,
                    freeze_json({"residual": 0.0001}),
                    "$.checks.solver",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
            active_wall_seconds=0.001,
            resource_queue_wall_seconds=0.002,
        )


class _UnconstrainedEvidenceAdapter:
    evaluator_identity = EvaluatorIdentity(
        "unconstrained_probe",
        1,
        hashlib.sha256(b"unconstrained-context").hexdigest(),
    )

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        self.calls += 1
        x = configuration["x"]
        assert type(x) is int
        receipt = _receipt(configuration, prefix="unconstrained")
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(
                ("levels", float(20 - x)),
                ("lut", float(10 - x)),
            ),
            violations=(),
            checks=(
                EvaluationCheck(
                    "equivalence",
                    EvaluationCheckStatus.PASS,
                    freeze_json({"return_code": 0}),
                    "$.checks.equivalence",
                ),
            ),
            receipt=receipt,
            evaluator=self.evaluator_identity,
            active_wall_seconds=0.003,
            resource_queue_wall_seconds=0.004,
        )


class _CapturingGenerator:
    def __init__(self, child: dict[str, object] | None = None) -> None:
        self.child = child
        self.reflection_prompt: str | None = None

    async def propose(self, request):
        del request
        if self.child is None:
            raise AssertionError("seed-only test must not propose")
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=dict(self.child),
                design_rationale="Reduce the measured normalized violation.",
                intended_changes=("$.x",),
                source_attribution=(SourceAttribution("$.x", "mutation"),),
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        self.reflection_prompt = request.prompt
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def _violation_first_relation() -> OutcomeRelationPolicyBinding:
    def compare(left, right):
        left_violation = dict(left.violations)["normalized_constraint"]
        right_violation = dict(right.violations)["normalized_constraint"]
        if left_violation < right_violation:
            return OutcomeRelation.BETTER
        if left_violation > right_violation:
            return OutcomeRelation.WORSE
        left_cost = dict(left.objectives)["cost"]
        right_cost = dict(right.objectives)["cost"]
        if left_cost < right_cost:
            return OutcomeRelation.BETTER
        if left_cost > right_cost:
            return OutcomeRelation.WORSE
        return OutcomeRelation.EQUIVALENT

    return OutcomeRelationPolicyBinding(
        compare=compare,
        policy_id="violation_first_test",
        policy_version=1,
        definition_sha256=hashlib.sha256(
            b"test:violation-first-then-cost:v1"
        ).hexdigest(),
    )


def _engine(
    *,
    problem: _Problem,
    adapter,
    generator: _CapturingGenerator,
    relation: OutcomeRelationPolicyBinding,
    reward_policy=None,
) -> AgenticEvolutionEngine:
    ids = DeterministicIdFactory("detailed_evaluation")
    kwargs = {}
    if reward_policy is not None:
        kwargs = {
            "reward_policy": reward_policy,
            "reward_definition_hash": hashlib.sha256(
                b"test:detailed-reward:v1"
            ).hexdigest(),
        }
    return AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=3,
        detailed_evaluator=adapter,
        outcome_relation_binding=relation,
        **kwargs,
    )


def test_airfoil_style_constrained_evidence_drives_relation_archive_and_reflection() -> None:
    objectives = (ObjectiveSpec("cost", "min"),)
    problem = _Problem(objectives)
    adapter = _ConstrainedEvidenceAdapter()
    generator = _CapturingGenerator({"x": 1, "alias": "same"})
    relation = _violation_first_relation()

    def reward(child, parents, declared_objectives):
        assert tuple(declared_objectives) == objectives
        assert child.detailed_evaluation is not None
        assert parents[0].detailed_evaluation is not None
        return 1.0 if relation.relate(
            child.detailed_evaluation,
            parents[0].detailed_evaluation,
        ) is OutcomeRelation.BETTER else -1.0

    async def scenario():
        engine = _engine(
            problem=problem,
            adapter=adapter,
            generator=generator,
            relation=relation,
            reward_policy=reward,
        )
        parent = await engine.register_seed(
            {"x": 0, "alias": "same"},
            label="parent",
        )
        outcomes = await engine.run_invocations(
            (
                InvocationPlan(
                    operator_kind=OperatorKind.TYPED_MUTATION,
                    parents=(parent,),
                    generation=1,
                    label="child",
                    allowed_top_level=("x",),
                ),
            )
        )
        await engine.reflect(outcomes, label="reflection")
        return engine, parent, outcomes[0]

    engine, parent, outcome = asyncio.run(scenario())
    child = outcome.candidate
    assert child is not None and child.detailed_evaluation is not None
    assert outcome.parent_relations == (OutcomeRelation.BETTER,)
    assert outcome.dominates_any_parent is False
    assert outcome.reward == 1.0
    assert dict(parent.objectives)["cost"] < dict(child.objectives)["cost"]
    assert dict(child.detailed_evaluation.violations) == {
        "normalized_constraint": 0.0
    }
    assert child.detailed_evaluation.timings.active_wall_seconds == 0.001
    assert child.detailed_evaluation.timings.resource_queue_wall_seconds == 0.002
    assert child.detailed_evaluation.timings.total_wall_seconds >= 0.0

    archive = ParetoArchive(
        objectives,
        outcome_relation_binding=relation,
    )
    archive.consider(parent)
    decisions = archive.consider(child)
    assert decisions[0].reasons == (
        ParetoDecisionReason.ADMITTED_RELATION_FRONT,
    )
    assert decisions[0].outcome_relations[0].relation is OutcomeRelation.BETTER
    assert archive.front == (child,)
    assert archive.outcome_relation_binding.identity == (
        engine.outcome_relation_binding.identity
    )

    class _UnusedPlanner:
        @staticmethod
        def plan(state, budget):
            del state, budget
            raise AssertionError("constructor-only binding check")

    with pytest.raises(ValueError, match="outcome relation bindings differ"):
        BudgetedAgenticOptimizer(
            engine=engine,
            archive=ParetoArchive(objectives),
            planner=_UnusedPlanner(),
            budget=OptimizerBudget(2, 1, 1),
        )

    assert generator.reflection_prompt is not None
    assert '"child_outcome_relation":"better"' in generator.reflection_prompt
    assert '"normalized_constraint":0.0' in generator.reflection_prompt
    assert '"evaluator_context_sha256"' in generator.reflection_prompt
    assert '"evidence_sha256"' in generator.reflection_prompt
    assert "Never rename a generic BETTER relation as Pareto dominance" in (
        generator.reflection_prompt
    )
    assert problem.legacy_calls == 0
    assert problem.raw_detailed_calls == 0


def test_boils_style_unconstrained_evidence_is_cached_by_phenotype_and_context() -> None:
    objectives = (
        ObjectiveSpec("lut", "min"),
        ObjectiveSpec("levels", "min"),
    )
    problem = _Problem(objectives)
    adapter = _UnconstrainedEvidenceAdapter()
    generator = _CapturingGenerator()
    relation = objective_pareto_outcome_binding(objectives)

    def project(configuration):
        value = thaw_json(configuration)
        assert type(value) is dict
        return {"x": value["x"]}

    ids = DeterministicIdFactory("detailed_evaluation_unconstrained")
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=4,
        detailed_evaluator=adapter,
        outcome_relation_binding=relation,
        phenotype_identity_policy=SemanticProjectionPhenotypeIdentityPolicy(
            policy_id="unconstrained_semantics",
            policy_version=1,
            projector=project,
        ),
    )

    async def scenario():
        left = await engine.register_seed({"x": 3, "alias": "three"}, label="left")
        right = await engine.register_seed({"x": 3, "alias": "III"}, label="right")
        return left, right, await engine.evaluation_cache_snapshot()

    left, right, snapshot = asyncio.run(scenario())
    assert adapter.calls == 1
    assert snapshot["misses"] == 1
    assert snapshot["hits"] == 1
    assert left.occurrence.configuration_hash != right.occurrence.configuration_hash
    assert left.detailed_evaluation is right.detailed_evaluation
    detailed = left.detailed_evaluation
    assert detailed is not None
    assert detailed.violations == ()
    assert tuple(name for name, _ in detailed.objectives) == ("lut", "levels")
    assert tuple(check.name for check in detailed.checks) == ("equivalence",)
    record = detailed.to_record()
    assert "candidate_key" not in record
    assert record["receipt"]["media_type"] == "application/json"
    assert problem.legacy_calls == 0
    assert problem.raw_detailed_calls == 0


def test_negative_normalized_violation_fails_closed() -> None:
    adapter = _ConstrainedEvidenceAdapter()
    payload = adapter.evaluate_evidence({"x": 0, "alias": "same"})
    with pytest.raises(ValueError, match="finite non-negative"):
        DetailedEvaluationPayload(
            failure=payload.failure,
            objectives=payload.objectives,
            violations=(("normalized_constraint", -0.1),),
            checks=payload.checks,
            receipt=payload.receipt,
            evaluator=payload.evaluator,
        )


def test_typed_candidate_failure_is_invalid_but_infrastructure_is_not_cached() -> None:
    objectives = (ObjectiveSpec("cost", "min"),)
    problem = _Problem(objectives)
    adapter = _ConstrainedEvidenceAdapter()
    engine = _engine(
        problem=problem,
        adapter=adapter,
        generator=_CapturingGenerator(),
        relation=_violation_first_relation(),
    )

    async def scenario():
        invalid = await engine.register_seed(
            {"x": -1, "alias": "candidate"},
            label="invalid",
        )
        for _ in range(2):
            with pytest.raises(RuntimeError, match="terminal failure evidence"):
                await engine.register_seed(
                    {"x": -2, "alias": "infrastructure"},
                    label="terminal",
                )
        return invalid, await engine.evaluation_cache_snapshot()

    invalid, snapshot = asyncio.run(scenario())
    assert invalid.valid is False
    assert invalid.detailed_evaluation is not None
    assert invalid.detailed_evaluation.failure is not None
    assert (
        invalid.detailed_evaluation.failure.category is FailureCategory.CANDIDATE
    )
    assert adapter.calls == 3
    assert snapshot["misses"] == 3
    assert snapshot["cached_entries"] == 1

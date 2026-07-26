from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

import agent_evolve.agentic as agentic
from agent_evolve.agentic import (
    ActionAxisSemantics,
    ActionSpaceSemantics,
    AgenticBenchmark,
    AgenticCallTelemetry,
    DetailedEvaluationPayload,
    DeterministicIdFactory,
    EvaluatorIdentity,
    FiniteVariationContract,
    FiniteVariationOption,
    FiniteVariationSelectionDraft,
    FixedStructuredOutputBudgetPolicy,
    FrozenJsonObject,
    FrozenWaveReward,
    GenerationPlan,
    InvocationPlan,
    JsonPath,
    MutationContract,
    MutationResponseMode,
    MetricRole,
    MetricSemantics,
    MetricSense,
    ObjectiveSpec,
    OptimizationSemantics,
    ObjectKey,
    OperatorKind,
    OptimizerBudget,
    OptimizerPlanningError,
    OptimizerSlot,
    OutcomeRelation,
    OutcomeRelationPolicyBinding,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
    ReflectionGenerationResult,
    ReflectionRowProjectionBinding,
    RewardPolicyBinding,
    SemanticProjectionPhenotypeIdentityPolicy,
    StructuredOutputBudgetPolicy,
    StructuredOutputRequestKind,
    VariationGenerationResult,
    compose_agentic_optimizer,
    freeze_json,
    resolve_structured_output_budget,
    thaw_json,
    typed_json_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int
    alias: str


class _DetailedProblem:
    candidate_model = _Configuration
    objectives = (ObjectiveSpec("score", "min"),)

    @staticmethod
    def search_space_description() -> str:
        return "An integer design variable with an evaluator-inert alias."

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        del configuration
        raise AssertionError("detailed evidence must replace legacy evaluation")


class _EvidenceAdapter:
    evaluator_identity = EvaluatorIdentity(
        "public_api_fixture",
        1,
        _sha("public-api-evaluator-context"),
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
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(("score", float(x)),),
            violations=(("normalized_constraint", 0.0),),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


def _relation() -> OutcomeRelationPolicyBinding:
    def compare(left, right):
        left_score = dict(left.objectives)["score"]
        right_score = dict(right.objectives)["score"]
        if left_score < right_score:
            return OutcomeRelation.BETTER
        if left_score > right_score:
            return OutcomeRelation.WORSE
        return OutcomeRelation.EQUIVALENT

    return OutcomeRelationPolicyBinding(
        compare=compare,
        policy_id="public_api_score_order",
        policy_version=1,
        definition_sha256=_sha("public-api-score-order-v1"),
    )


def _reward() -> RewardPolicyBinding:
    def score(child, parents, objectives):
        del objectives
        if not child.valid:
            return -1.0
        parent_scores = tuple(
            parent.objective_map["score"] for parent in parents if parent.valid
        )
        if not parent_scores:
            return 0.0
        return float(min(parent_scores) - child.objective_map["score"])

    return RewardPolicyBinding(score, _sha("public-api-parent-score-reward-v1"))


def _optimization_semantics() -> OptimizationSemantics:
    relation = _relation()
    return OptimizationSemantics(
        semantics_id="public_api_score_semantics",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:score",
                name="score",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="score is the evaluated integer design coordinate",
                aggregation="one scalar evaluation",
                witness_interpretation="lower score is better",
                tolerance=0.0,
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.LEXICOGRAPHIC,
            metric_priority=("objective:score",),
            description="Lower score is better.",
            equivalence="Equal scores are equivalent.",
            policy_id=relation.policy_id,
            policy_version=relation.policy_version,
            definition_sha256=relation.definition_sha256,
        ),
    )


def _semantic_identity() -> SemanticProjectionPhenotypeIdentityPolicy:
    def project(configuration):
        value = thaw_json(configuration)
        assert type(value) is dict
        return {"x": value["x"]}

    return SemanticProjectionPhenotypeIdentityPolicy(
        policy_id="public_api_semantic_x",
        policy_version=1,
        projector=project,
    )


class _Catalog:
    catalog_id = "public_api_fixture"
    catalog_version = 1
    definition_sha256 = _sha("public-api-catalog-v1")
    option_families = ("coordinate_and_label",)

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        assert type(parent) is dict
        x = parent["x"]
        assert type(x) is int
        return (
            FiniteVariationOption(
                option_id="coordinate.increment_and_relabel",
                parent_configuration_sha256=typed_json_sha256(parent_configuration),
                child_configuration=freeze_json({"x": x + 1, "alias": "catalog_child"}),
                family="coordinate_and_label",
                description="Increment the coordinate and assign its canonical label.",
                metadata=(("step", "positive_one"),),
            ),
        )


class _SecondCatalog:
    catalog_id = "public_api_alternate"
    catalog_version = 2
    definition_sha256 = _sha("public-api-catalog-alternate-v2")
    option_families = ("coordinate_and_label",)

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        assert type(parent) is dict
        x = parent["x"]
        assert type(x) is int
        return (
            FiniteVariationOption(
                option_id="coordinate.decrement_and_relabel",
                parent_configuration_sha256=typed_json_sha256(parent_configuration),
                child_configuration=freeze_json(
                    {"x": x - 1, "alias": "alternate_child"}
                ),
                family="coordinate_and_label",
                description="Decrement the coordinate and assign another label.",
            ),
        )


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/public-api",
        resolved_model="offline/public-api",
        resolved_provider="fixture",
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


class _SelectingGenerator:
    def __init__(self) -> None:
        self.contracts: list[FiniteVariationContract] = []
        self.proposal_requests = []
        self.reflection_requests = []

    async def propose(self, request):
        self.proposal_requests.append(request)
        contract = request.finite_variation_contract
        assert type(contract) is FiniteVariationContract
        self.contracts.append(contract)
        option = contract.options[0]
        return VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=contract.identity_sha256,
                design_rationale="Use the benchmark-sealed coordinated option.",
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        self.reflection_requests.append(request)
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


class _NeverPlanner:
    def plan(self, state, budget):
        del state, budget
        raise AssertionError("a zero-generation run must not invoke the planner")


class _WrongPlanner:
    def plan(self, state, budget):
        del state, budget
        return object()


class _FinitePlanner:
    def __init__(self, benchmark: AgenticBenchmark) -> None:
        self.benchmark = benchmark

    def plan(self, state, budget):
        del budget
        parent = state.candidates[0]
        contract = self.benchmark.bind_finite_variation(
            "public_api_fixture",
            parent.configuration,
        )
        invocation = InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(parent,),
            generation=state.generation + 1,
            label="public_api_finite_selection",
            allowed_top_level=("alias", "x"),
            mutation_contract=MutationContract(
                editable_paths=(
                    JsonPath((ObjectKey("alias"),)),
                    JsonPath((ObjectKey("x"),)),
                ),
                max_changed_paths=2,
                max_operations=2,
            ),
            mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
            finite_variation_contract=contract,
        )
        return GenerationPlan(
            generation=state.generation + 1,
            slots=(
                OptimizerSlot.model(
                    slot_id="G1-F",
                    role="sealed_finite_selection",
                    plan=invocation,
                ),
            ),
            reward=FrozenWaveReward(
                binding=self.benchmark.reward,
                archive_snapshot_hash=state.archive_snapshot_hash,
                reward_snapshot_hash=_sha(
                    f"public-api-reward:{state.archive_snapshot_hash}"
                ),
            ),
            planner_policy_id="public_api_finite_planner",
            planner_policy_version=1,
        )


def _benchmark(adapter: _EvidenceAdapter) -> AgenticBenchmark:
    return AgenticBenchmark(
        problem=_DetailedProblem(),
        reward=_reward(),
        detailed_evaluator=adapter,
        outcome_relation=_relation(),
        optimization_semantics=_optimization_semantics(),
        phenotype_identity=_semantic_identity(),
        finite_variation_catalogs=(_Catalog(), _SecondCatalog()),
    )


def _action_semantics(
    *,
    families: tuple[str, ...] = ("coordinate_and_label",),
    alternate_definition_sha256: str | None = None,
) -> ActionSpaceSemantics:
    return ActionSpaceSemantics(
        semantics_id="public_api_fixture_action_space",
        semantics_version=1,
        catalog_identities=(
            (
                _SecondCatalog.catalog_id,
                _SecondCatalog.catalog_version,
                (
                    _SecondCatalog.definition_sha256
                    if alternate_definition_sha256 is None
                    else alternate_definition_sha256
                ),
            ),
            (
                _Catalog.catalog_id,
                _Catalog.catalog_version,
                _Catalog.definition_sha256,
            ),
        ),
        axes=(
            ActionAxisSemantics(
                axis_id="coordinated_fixture_edit",
                configuration_paths=("$.alias", "$.x"),
                option_families=families,
                definition=(
                    "One sealed edit changes the integer coordinate and its label."
                ),
                independence=(
                    "The coordinate and label changes are coupled within each "
                    "catalog option."
                ),
                excluded_interpretations=(
                    "The label is not an independently selectable action.",
                ),
            ),
        ),
    )


def _run(awaitable):
    """Run with an owned evaluator pool and leave no threads behind."""

    async def run_with_heartbeat():
        execution = asyncio.create_task(awaitable)
        while not execution.done():
            await asyncio.sleep(0.01)
        return await execution

    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="agentic_public_api_evaluator",
    )
    loop.set_default_executor(executor)
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_with_heartbeat())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        loop.close()
        asyncio.set_event_loop(None)


def test_every_declared_facade_symbol_is_importable() -> None:
    assert agentic.__all__
    assert len(agentic.__all__) == len(set(agentic.__all__))
    for symbol in agentic.__all__:
        assert not symbol.startswith("_")
        assert getattr(agentic, symbol) is not None


def test_fixed_structured_output_budget_is_immutable_and_route_capable() -> None:
    policy = FixedStructuredOutputBudgetPolicy(
        proposal_max_output_tokens=384_000,
        reflection_max_output_tokens=384_000,
    )

    assert isinstance(policy, StructuredOutputBudgetPolicy)
    assert (
        resolve_structured_output_budget(
            policy,
            request_kind=StructuredOutputRequestKind.PROPOSAL,
            operation="typed_mutation",
        )
        == 384_000
    )
    assert (
        resolve_structured_output_budget(
            policy,
            request_kind=StructuredOutputRequestKind.REFLECTION,
            operation="extract_insights",
        )
        == 384_000
    )
    with pytest.raises(FrozenInstanceError):
        policy.proposal_max_output_tokens = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="proposal_max_output_tokens"):
        FixedStructuredOutputBudgetPolicy(
            proposal_max_output_tokens=0,
            reflection_max_output_tokens=1,
        )


def test_composition_routes_split_budgets_and_preserves_legacy_uniform_limit() -> None:
    def composition(
        *,
        generator: _SelectingGenerator,
        id_namespace: str,
        policy: FixedStructuredOutputBudgetPolicy | None = None,
        legacy_limit: int = 2_048,
        engine_events: list[dict[str, object]] | None = None,
    ):
        return compose_agentic_optimizer(
            _benchmark(_EvidenceAdapter()),
            generator=generator,
            planner=_NeverPlanner(),
            budget=OptimizerBudget(4, 2, 2),
            seed=23,
            id_factory=DeterministicIdFactory(id_namespace),
            engine_trace_sink=(
                None
                if engine_events is None
                else lambda event: engine_events.append(dict(event))
            ),
            max_output_tokens=legacy_limit,
            structured_output_budget_policy=policy,
        )

    def finite_plan(composed, parent):
        contract = composed.bind_finite_variation(
            "public_api_fixture",
            parent.configuration,
        )
        return InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(parent,),
            generation=1,
            label="public_api_output_budget",
            allowed_top_level=("alias", "x"),
            mutation_contract=MutationContract(
                editable_paths=(
                    JsonPath((ObjectKey("alias"),)),
                    JsonPath((ObjectKey("x"),)),
                ),
                max_changed_paths=2,
                max_operations=2,
            ),
            mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
            finite_variation_contract=contract,
        )

    async def exercise(composed):
        parent = await composed.engine.register_seed(
            {"x": 1, "alias": "seed"},
            label="seed",
        )
        plan = finite_plan(composed, parent)
        prompt_shape = composed.engine.prompt_shape_commitment(
            plan,
            selected_insight_count=0,
        )
        outcome = (await composed.engine.run_invocations((plan,)))[0]
        await composed.engine.reflect(
            (outcome,),
            label="public_api_output_budget",
        )
        return prompt_shape, plan

    split_generator = _SelectingGenerator()
    split_events: list[dict[str, object]] = []
    split = composition(
        generator=split_generator,
        id_namespace="public_api_split_budget",
        policy=FixedStructuredOutputBudgetPolicy(
            proposal_max_output_tokens=1_536,
            reflection_max_output_tokens=4_096,
        ),
        engine_events=split_events,
    )
    first, shared_plan = _run(exercise(split))
    assert split_generator.proposal_requests[0].max_output_tokens == 1_536
    assert split_generator.reflection_requests[0].max_output_tokens == 4_096
    semantics = split.engine.optimization_semantics
    assert type(semantics) is OptimizationSemantics
    canonical_semantics = json.dumps(
        semantics.to_record(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    proposal_prompt = split_generator.proposal_requests[0].prompt
    reflection_prompt = split_generator.reflection_requests[0].prompt
    assert proposal_prompt.count(canonical_semantics) == 1
    assert reflection_prompt.count(canonical_semantics) == 1
    semantics_record = semantics.to_record()
    assert (
        next(
            event
            for event in split_events
            if event["event_type"] == "invocation_prepared"
        )["optimization_semantics"]
        == semantics_record
    )
    assert (
        next(
            event
            for event in split_events
            if event["event_type"] == "reflection_requested"
        )["optimization_semantics"]
        == semantics_record
    )

    legacy_generator = _SelectingGenerator()
    legacy = composition(
        generator=legacy_generator,
        id_namespace="public_api_legacy_budget",
        legacy_limit=777,
    )
    _run(exercise(legacy))
    assert legacy_generator.proposal_requests[0].max_output_tokens == 777
    assert legacy_generator.reflection_requests[0].max_output_tokens == 777

    reflection_only_change = composition(
        generator=_SelectingGenerator(),
        id_namespace="shape_pair_reflect",
        policy=FixedStructuredOutputBudgetPolicy(1_536, 384_000),
    ).engine.prompt_shape_commitment(
        shared_plan,
        selected_insight_count=0,
    )
    proposal_change = composition(
        generator=_SelectingGenerator(),
        id_namespace="shape_pair_propose",
        policy=FixedStructuredOutputBudgetPolicy(1_537, 4_096),
    ).engine.prompt_shape_commitment(
        shared_plan,
        selected_insight_count=0,
    )
    assert first == reflection_only_change
    assert first != proposal_change


def test_feedback_inversion_contract_is_available_without_internal_imports() -> None:
    for symbol in (
        "GenerationFeedbackContext",
        "GenerationFeedbackInterceptor",
        "GenerationFeedbackReceipt",
        "GenerationFeedbackReservation",
        "GenerationFeedbackResult",
        "generation_feedback_receipt_hash",
        "seal_generation_feedback",
        "validate_generation_feedback_receipt",
    ):
        assert symbol in agentic.__all__
        assert getattr(agentic, symbol) is not None


def test_exact_parent_crossover_is_available_on_the_generic_facade() -> None:
    for symbol in (
        "CrossoverResponseMode",
        "ExactParentCrossoverContract",
        "ExactParentCrossoverLocus",
        "ExactParentCrossoverMaterialization",
        "ExactParentCrossoverReceipt",
        "ExactParentImportPlan",
        "ExactParentLocusAttribution",
        "ExactParentSource",
        "build_exact_parent_import_plan",
        "derive_exact_parent_crossover_contract",
        "materialize_exact_parent_crossover",
        "replay_exact_parent_crossover",
    ):
        assert symbol in agentic.__all__
        assert getattr(agentic, symbol) is not None


def test_g3_curation_and_engine_reflection_receipts_are_public() -> None:
    for symbol in (
        "G3CurationSourceScope",
        "G3PostsealCurationAuthority",
        "G3PostsealCurationFactory",
        "G3PostsealCurationInterceptor",
        "G3PostsealCurationReceipt",
        "G3PostsealCurationSpec",
        "G3_POSTSEAL_CURATION_DEFINITION_SHA256",
        "ReflectionCallExecutionError",
        "ReflectionCallReceipt",
        "ReflectionCallRequest",
        "ReflectionCallStatus",
        "ReflectionPublication",
        "ReflectionPublicationResult",
        "build_g3_postseal_curation_reservation",
    ):
        assert symbol in agentic.__all__
        assert getattr(agentic, symbol) is not None


def test_detailed_mode_fails_closed_without_a_complete_binding() -> None:
    problem = _DetailedProblem()
    adapter = _EvidenceAdapter()
    with pytest.raises(ValueError, match="explicit outcome_relation"):
        AgenticBenchmark(problem=problem, detailed_evaluator=adapter)
    with pytest.raises(ValueError, match="requires a detailed_evaluator"):
        AgenticBenchmark(problem=problem, outcome_relation=_relation())


def test_catalog_registry_rejects_duplicates_unknown_ids_and_identity_drift() -> None:
    adapter = _EvidenceAdapter()
    with pytest.raises(ValueError, match="catalog IDs must be unique"):
        AgenticBenchmark(
            problem=_DetailedProblem(),
            reward=_reward(),
            detailed_evaluator=adapter,
            outcome_relation=_relation(),
            finite_variation_catalogs=(_Catalog(), _Catalog()),
        )

    mutable_catalog = _Catalog()
    benchmark = AgenticBenchmark(
        problem=_DetailedProblem(),
        reward=_reward(),
        detailed_evaluator=adapter,
        outcome_relation=_relation(),
        finite_variation_catalogs=(mutable_catalog, _SecondCatalog()),
    )
    assert tuple(
        identity[0] for identity in benchmark.finite_variation_catalog_identities
    ) == ("public_api_fixture", "public_api_alternate")
    parent = {"x": 1, "alias": "seed"}
    primary = benchmark.bind_finite_variation("public_api_fixture", parent)
    alternate = benchmark.bind_finite_variation("public_api_alternate", parent)
    assert primary.catalog_id == "public_api_fixture"
    assert alternate.catalog_id == "public_api_alternate"
    assert primary.identity_sha256 != alternate.identity_sha256
    with pytest.raises(KeyError, match="unknown finite variation catalog"):
        benchmark.bind_finite_variation(
            "missing_catalog",
            {"x": 1, "alias": "seed"},
        )
    mutable_catalog.definition_sha256 = _sha("mutated-definition")
    with pytest.raises(ValueError, match="identity changed"):
        benchmark.validate_binding()


def test_benchmark_action_semantics_bind_full_catalog_and_family_vocabularies() -> None:
    primary = _Catalog()
    alternate = _SecondCatalog()
    semantics = _action_semantics()
    benchmark = AgenticBenchmark(
        problem=_DetailedProblem(),
        finite_variation_catalogs=(primary, alternate),
        action_semantics=semantics,
    )

    assert benchmark.action_semantics is semantics
    benchmark.validate_binding()

    with pytest.raises(ValueError, match="catalog identities differ"):
        AgenticBenchmark(
            problem=_DetailedProblem(),
            finite_variation_catalogs=(primary, alternate),
            action_semantics=_action_semantics(
                alternate_definition_sha256=_sha("wrong-catalog-definition"),
            ),
        )
    with pytest.raises(ValueError, match="option-family coverage"):
        AgenticBenchmark(
            problem=_DetailedProblem(),
            finite_variation_catalogs=(primary, alternate),
            action_semantics=_action_semantics(families=("invented_family",)),
        )

    primary.option_families = ("mutated_family",)
    with pytest.raises(ValueError, match="family vocabulary changed"):
        benchmark.validate_binding()


def test_planner_wrong_return_type_fails_at_public_boundary() -> None:
    problem = _LegacyProblem()
    composition = compose_agentic_optimizer(
        AgenticBenchmark(problem=problem),
        generator=_SelectingGenerator(),
        planner=_WrongPlanner(),
        budget=OptimizerBudget(1, 0, 1),
        seed=3,
        id_factory=DeterministicIdFactory("public_api_wrong_planner"),
    )
    with pytest.raises(OptimizerPlanningError, match="planner failed"):
        _run(composition.optimizer.run(({"x": 1, "alias": "wrong_planner"},)))


def test_deferred_runtime_factories_receive_exact_composed_identities() -> None:
    planner = _NeverPlanner()
    feedback = object()
    planner_bindings: dict[str, object] = {}
    feedback_bindings: dict[str, object] = {}

    class PlannerFactory:
        calls = 0

        def build(self, *, benchmark, engine, id_factory, memory):
            self.calls += 1
            planner_bindings.update(
                benchmark=benchmark,
                engine=engine,
                id_factory=id_factory,
                memory=memory,
            )
            return planner

    class Feedback:
        def reserve(self, *, state, plan):  # pragma: no cover - identity test only.
            del state, plan
            raise AssertionError("zero-generation identity test cannot reserve")

        async def after_generation(self, context):  # pragma: no cover
            del context
            raise AssertionError("zero-generation identity test cannot run feedback")

    feedback = Feedback()

    class FeedbackFactory:
        calls = 0

        def build(self, *, benchmark, engine, id_factory, memory, planner):
            self.calls += 1
            feedback_bindings.update(
                benchmark=benchmark,
                engine=engine,
                id_factory=id_factory,
                memory=memory,
                planner=planner,
            )
            return feedback

    planner_factory = PlannerFactory()
    feedback_factory = FeedbackFactory()
    benchmark = AgenticBenchmark(problem=_LegacyProblem())
    ids = DeterministicIdFactory("public_api_deferred_factory")
    composition = compose_agentic_optimizer(
        benchmark,
        generator=_SelectingGenerator(),
        planner_factory=planner_factory,
        feedback_interceptor_factory=feedback_factory,
        budget=OptimizerBudget(1, 0, 0),
        seed=29,
        id_factory=ids,
    )

    assert planner_factory.calls == 1
    assert feedback_factory.calls == 1
    assert composition.planner is planner
    assert composition.feedback_interceptor is feedback
    assert composition.optimizer.planner.delegate is planner
    assert composition.optimizer.feedback_interceptor is feedback
    for bindings in (planner_bindings, feedback_bindings):
        assert bindings["benchmark"] is composition.benchmark
        assert bindings["engine"] is composition.engine
        assert bindings["id_factory"] is composition.id_factory
        assert bindings["memory"] is composition.memory
    assert feedback_bindings["planner"] is composition.planner


def test_composition_requires_xor_planner_boundary_and_valid_factory_results() -> None:
    benchmark = AgenticBenchmark(problem=_LegacyProblem())
    common = {
        "benchmark": benchmark,
        "generator": _SelectingGenerator(),
        "budget": OptimizerBudget(1, 0, 0),
        "seed": 31,
    }

    with pytest.raises(ValueError, match="exactly one"):
        compose_agentic_optimizer(**common)

    class PlannerFactory:
        def build(self, *, benchmark, engine, id_factory, memory):
            del benchmark, engine, id_factory, memory
            return _NeverPlanner()

    with pytest.raises(ValueError, match="exactly one"):
        compose_agentic_optimizer(
            **common,
            planner=_NeverPlanner(),
            planner_factory=PlannerFactory(),
        )
    with pytest.raises(TypeError, match="planner_factory"):
        compose_agentic_optimizer(**common, planner_factory=object())

    class BrokenPlannerFactory:
        def build(self, *, benchmark, engine, id_factory, memory):
            del benchmark, engine, id_factory, memory
            return object()

    with pytest.raises(TypeError, match="must return a planner"):
        compose_agentic_optimizer(
            **common,
            planner_factory=BrokenPlannerFactory(),
        )


def test_deferred_factory_cannot_drift_benchmark_or_forge_runtime_identities() -> None:
    benchmark = AgenticBenchmark(problem=_LegacyProblem())

    class MutatingPlannerFactory:
        def build(self, *, benchmark, engine, id_factory, memory):
            del engine, id_factory, memory
            benchmark.problem.objectives = (ObjectiveSpec("drifted", "min"),)
            return _NeverPlanner()

    with pytest.raises(ValueError, match="objectives changed"):
        compose_agentic_optimizer(
            benchmark,
            generator=_SelectingGenerator(),
            planner_factory=MutatingPlannerFactory(),
            budget=OptimizerBudget(1, 0, 0),
            seed=37,
        )

    composition = compose_agentic_optimizer(
        AgenticBenchmark(problem=_LegacyProblem()),
        generator=_SelectingGenerator(),
        planner=_NeverPlanner(),
        budget=OptimizerBudget(1, 0, 0),
        seed=41,
        id_factory=DeterministicIdFactory("public_api_identity_forge"),
    )
    with pytest.raises(ValueError, match="different ID factory"):
        replace(
            composition,
            id_factory=DeterministicIdFactory("public_api_wrong_ids"),
        )
    with pytest.raises(ValueError, match="different memory bank"):
        replace(
            composition,
            memory=agentic.InsightMemoryBank(
                id_factory=DeterministicIdFactory("public_api_wrong_memory")
            ),
        )
    forged_reward = RewardPolicyBinding(
        composition.benchmark.reward.score,
        composition.benchmark.reward.definition_hash,
        failure_score=-2.0,
    )
    with pytest.raises(ValueError, match="total reward binding"):
        replace(
            composition,
            benchmark=AgenticBenchmark(
                problem=composition.benchmark.problem,
                reward=forged_reward,
            ),
        )


def test_composition_does_not_mask_problem_objective_drift() -> None:
    adapter = _EvidenceAdapter()
    benchmark = _benchmark(adapter)
    benchmark.problem.objectives = (ObjectiveSpec("changed", "min"),)
    with pytest.raises(ValueError, match="objectives changed"):
        compose_agentic_optimizer(
            benchmark,
            generator=_SelectingGenerator(),
            planner=_NeverPlanner(),
            budget=OptimizerBudget(1, 0, 0),
            seed=1,
        )


def test_binding_detects_evaluator_and_phenotype_identity_drift() -> None:
    adapter = _EvidenceAdapter()
    benchmark = _benchmark(adapter)
    adapter.evaluator_identity = EvaluatorIdentity(
        "public_api_fixture",
        2,
        _sha("changed-evaluator-context"),
    )
    with pytest.raises(ValueError, match="evaluator identity changed"):
        benchmark.validate_binding()

    adapter = _EvidenceAdapter()
    benchmark = _benchmark(adapter)
    object.__setattr__(benchmark.phenotype_identity, "policy_version", 2)
    with pytest.raises(ValueError, match="phenotype policy identity changed"):
        benchmark.validate_binding()


def test_composition_shares_relation_and_semantic_cache_identity() -> None:
    adapter = _EvidenceAdapter()
    benchmark = _benchmark(adapter)
    composition = compose_agentic_optimizer(
        benchmark,
        generator=_SelectingGenerator(),
        planner=_NeverPlanner(),
        budget=OptimizerBudget(2, 0, 0),
        seed=11,
        id_factory=DeterministicIdFactory("public_api_cache"),
    )

    assert composition.outcome_relation is benchmark.outcome_relation
    assert (
        composition.engine.outcome_relation_binding
        is composition.archive.outcome_relation_binding
    )
    assert composition.optimizer.engine is composition.engine
    assert composition.optimizer.archive is composition.archive
    assert composition.engine.reward_binding == benchmark.reward
    assert composition.engine.optimization_semantics is (
        benchmark.optimization_semantics
    )

    async def scenario():
        first = await composition.engine.register_seed(
            {"x": 4, "alias": "first"},
            label="first_alias",
        )
        second = await composition.engine.register_seed(
            {"x": 4, "alias": "second"},
            label="second_alias",
        )
        return first, second, await composition.engine.evaluation_cache_snapshot()

    first, second, cache = _run(scenario())
    assert first.candidate_id != second.candidate_id
    assert first.detailed_evaluation == second.detailed_evaluation
    assert adapter.calls == 1
    assert cache["misses"] == 1
    assert cache["hits"] == 1


def test_bound_catalog_materializes_a_finite_selection_through_engine() -> None:
    adapter = _EvidenceAdapter()
    benchmark = _benchmark(adapter)
    generator = _SelectingGenerator()
    composition = compose_agentic_optimizer(
        benchmark,
        generator=generator,
        planner=_FinitePlanner(benchmark),
        budget=OptimizerBudget(2, 1, 1),
        seed=19,
        id_factory=DeterministicIdFactory("public_api_finite"),
    )

    parent = _run(
        composition.engine.register_seed(
            {"x": 1, "alias": "seed"},
            label="seed",
        )
    )
    contract = composition.bind_finite_variation(
        "public_api_fixture",
        parent.configuration,
    )
    plan = InvocationPlan(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=1,
        label="public_api_finite_selection",
        allowed_top_level=("alias", "x"),
        mutation_contract=MutationContract(
            editable_paths=(
                JsonPath((ObjectKey("alias"),)),
                JsonPath((ObjectKey("x"),)),
            ),
            max_changed_paths=2,
            max_operations=2,
        ),
        mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
        finite_variation_contract=contract,
    )
    (outcome,) = _run(composition.engine.run_invocations((plan,)))

    assert adapter.calls == 2
    assert len(generator.contracts) == 1
    assert (
        generator.contracts[0].identity_sha256
        == composition.bind_finite_variation(
            "public_api_fixture",
            {"x": 1, "alias": "seed"},
        ).identity_sha256
    )
    child = outcome.candidate
    assert child is not None
    assert child.configuration_dict == {"x": 2, "alias": "catalog_child"}
    assert child.valid and child.operator_compliant


_PROJECTION_UNSET = object()


def _non_airfoil_reflection_prompt(
    projection: ReflectionRowProjectionBinding | None | object = _PROJECTION_UNSET,
) -> str:
    benchmark = AgenticBenchmark(
        problem=_LegacyProblem(),
        finite_variation_catalogs=(_Catalog(),),
    )
    generator = _SelectingGenerator()
    projection_kwargs = (
        {}
        if projection is _PROJECTION_UNSET
        else {"reflection_row_projection": projection}
    )
    composition = compose_agentic_optimizer(
        benchmark,
        generator=generator,
        planner=_FinitePlanner(benchmark),
        budget=OptimizerBudget(2, 1, 1),
        seed=29,
        id_factory=DeterministicIdFactory("public_api_reflection_projection"),
        **projection_kwargs,
    )
    result = _run(composition.optimizer.run(({"x": 1, "alias": "seed"},)))
    outcome = result.generation_receipts[0].slot_results[0].outcome
    _run(composition.engine.reflect((outcome,), label="public_api_reflection"))
    assert len(generator.reflection_requests) == 1
    return generator.reflection_requests[0].prompt


def test_non_airfoil_default_projection_preserves_legacy_prompt_bytes() -> None:
    implicit = _non_airfoil_reflection_prompt()
    explicit = _non_airfoil_reflection_prompt(None)

    assert implicit.encode("utf-8") == explicit.encode("utf-8")
    assert "REFLECTION EVIDENCE PROJECTION" not in implicit


def test_non_airfoil_projection_cannot_mutate_nested_machine_evidence() -> None:
    def malicious_projector(row: Mapping[str, object]) -> Mapping[str, object]:
        assert type(row) is dict
        contrasts = row["machine_derived_contrasts"]
        assert type(contrasts) is list and contrasts
        first = contrasts[0]
        assert type(first) is dict
        first["forged_by_projector"] = True
        return row

    binding = ReflectionRowProjectionBinding(
        project=malicious_projector,
        policy_id="public_api_hostile_nested_mutator",
        policy_version=1,
        definition_sha256=_sha("public-api-hostile-nested-mutator-v1"),
    )

    with pytest.raises(ValueError, match="machine-derived contrasts"):
        _non_airfoil_reflection_prompt(binding)


class _LegacyProblem:
    candidate_model = _Configuration
    objectives = (ObjectiveSpec("score", "min"),)

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        self.calls += 1
        return {"score": float(configuration["x"])}


def test_objective_only_legacy_mode_uses_one_shared_default_relation() -> None:
    problem = _LegacyProblem()
    benchmark = AgenticBenchmark(problem=problem)
    composition = compose_agentic_optimizer(
        benchmark,
        generator=_SelectingGenerator(),
        planner=_NeverPlanner(),
        budget=OptimizerBudget(1, 0, 0),
        seed=23,
        id_factory=DeterministicIdFactory("public_api_legacy"),
    )

    candidate = _run(
        composition.engine.register_seed(
            {"x": 3, "alias": "legacy"},
            label="legacy_seed",
        )
    )
    assert candidate.valid
    assert candidate.detailed_evaluation is None
    assert problem.calls == 1
    assert (
        composition.engine.outcome_relation_binding
        is composition.archive.outcome_relation_binding
    )

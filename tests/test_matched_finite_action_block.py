"""Provider-free integration tests for the generic matched finite-action block."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, replace
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    OperatorKind,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    OptimizerBudget,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    InsightOrigin,
)
from agent_evolve.application.matched_finite_action_block import (
    MatchedFiniteActionBlockPlanner,
)
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_action_set import FiniteActionSourceMode
from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.treatment_compliance import (
    TreatmentActionBinding,
)
from agent_evolve.policies.selection.finite_action import (
    TaskKeyedUniformFiniteActionPolicy,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    VariationGenerationResult,
)
from agent_evolve.ports.executable_hypothesis import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
)
from agent_evolve.ports.finite_action_set import (
    FiniteActionSetCompilationRequest,
    FiniteActionSetDraft,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    value: int
    held: int


class _Problem:
    candidate_model = _Config
    objectives = (ObjectiveSpec("score", "min"),)

    def __init__(self) -> None:
        self.evaluations: list[int] = []

    @staticmethod
    def search_space_description() -> str:
        return "Choose one bounded integer while holding a second field fixed."

    @staticmethod
    def validate(configuration: object) -> bool:
        candidate = _Config.model_validate(configuration, strict=True)
        return -4 <= candidate.value <= 4 and candidate.held == 7

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        value = int(configuration["value"])
        self.evaluations.append(value)
        # Leave enough overlap for identical concurrent A/U requests to reach
        # the cache's single-flight path deterministically.
        time.sleep(0.03)
        return {"score": float(abs(value - 3))}


class _K8Catalog:
    catalog_id = "fixture_k8_local"
    catalog_version = 1
    definition_sha256 = _sha("fixture K8 catalog v1")

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        assert type(parent) is dict
        parent_sha256 = typed_json_sha256(parent_configuration)
        return tuple(
            FiniteVariationOption(
                option_id=f"local.value_{'n' if value < 0 else 'p'}{abs(value)}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"value": value, "held": 7}),
                family="local_value",
                description=f"Set the local integer to {value}.",
            )
            for value in (-4, -3, -2, -1, 1, 2, 3, 4)
        )


class _ExactAnchorCompiler:
    policy_id = "fixture_exact_anchor"
    policy_version = 1
    definition_sha256 = _sha("fixture exact K8 anchor v1")

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        option = request.finite_contract.resolve(
            request.insight.recommended_option_ids[0]
        )
        spec = ExecutableHypothesisTestSpec(
            request_sha256=request.request_sha256,
            reference=request.reference,
            insight_content_sha256=request.insight.content_sha256,
            source_evidence_sha256=request.source_evidence_sha256,
            requested_operator_kind=request.requested_operator_kind,
            source_operator_kinds=request.source_operator_kinds,
            executable_operator_kinds=(request.requested_operator_kind,),
            parent_candidate_id=request.parent_candidate_id,
            parent_configuration_sha256=request.parent_configuration_sha256,
            finite_contract_sha256=request.finite_contract.identity_sha256,
            context_projection_sha256=request.context_projection_sha256,
            endpoint_definition_sha256=request.endpoint_definition_sha256,
            allowed_actions=(
                TreatmentActionBinding(option.option_id, option.identity_sha256),
            ),
            recommended_option_families=(option.family,),
            affected_paths=("$.value",),
            held_fixed_paths=("$.held",),
            effect_predictions=request.insight.effect_predictions,
            falsification_condition=str(request.insight.falsification_condition),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
        )
        return HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.APPLICABLE,
            reason_codes=(),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
            spec=spec,
        )


class _K8SupportCompiler:
    policy_id = "fixture_k8_support"
    policy_version = 1
    definition_sha256 = _sha("fixture K8 support compiler v1")

    def compile(
        self,
        request: FiniteActionSetCompilationRequest,
    ) -> FiniteActionSetDraft:
        option_ids = tuple(
            option.option_id for option in request.finite_contract.options
        )
        assert len(option_ids) == request.required_cardinality == 8
        return FiniteActionSetDraft(
            request_sha256=request.request_sha256,
            ordered_option_ids=option_ids,
            anchor_option_id=request.anchor_option_id,
            presentation_policy_id="fixture_k8_presentation",
            presentation_policy_version=1,
            presentation_definition_sha256=_sha("fixture K8 presentation v1"),
            prompt_shape_sha256=_sha("fixture K8 prompt shape v1"),
        )


def _reward() -> RewardPolicyBinding:
    def score(child, parents, objectives):
        del objectives
        if not child.valid or not child.operator_compliant:
            return -1.0
        return parents[0].objective_map["score"] - child.objective_map["score"]

    return RewardPolicyBinding(score, _sha("fixture K8 reward v1"))


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fixture",
        provider_response_id="fixture-k8-response",
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


class _SelectingGenerator:
    def __init__(self, *, forged_claim: bool = False) -> None:
        self.planner: MatchedFiniteActionBlockPlanner | None = None
        self.forged_claim = forged_claim
        self.requests = []

    async def propose(self, request):
        self.requests.append(request)
        planner = self.planner
        assert planner is not None and planner.uniform_decision is not None
        contract = request.finite_variation_contract
        assert contract is not None
        option = contract.options[planner.uniform_decision.selected_ordinal]
        claimed_id = (
            "insight_forged_claim"
            if self.forged_claim
            else planner.card_reference.insight_id.value
        )
        await asyncio.sleep(0)
        return VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=contract.identity_sha256,
                design_rationale="Choose one opaque ID from the exact K8 authority.",
                claimed_insight_ids=(claimed_id,),
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


class _FiniteActionCitationReflector:
    def __init__(self, *, recommended_option_ids: tuple[str, ...] = ()) -> None:
        self.recommended_option_ids = recommended_option_ids
        self.requests = []

    async def propose(self, request):  # pragma: no cover - reflection-only test.
        del request
        raise AssertionError("finite-action citation reflector cannot propose")

    async def reflect(self, request):
        self.requests.append(request)
        advanced: dict[str, object] = {}
        if request.insight_contract is not None:
            advanced = {
                "effect_predictions": (
                    MetricEffectPrediction(
                        metric_id="objective:score",
                        direction=MetricEffectDirection.DECREASE,
                    ),
                ),
                "recommended_option_families": ("local_value",),
                "recommended_option_ids": self.recommended_option_ids,
                "action_template": (
                    "Select only the exact bounded local action supported by "
                    "the cited model and engine contrasts."
                ),
                "falsification_condition": (
                    "The held-out score fails to decrease under that exact action."
                ),
            }
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim="The cited bounded finite action may transfer.",
                    trigger="A held-out parent exposes the same bounded coordinate.",
                    mechanism=(
                        "Replay only the exact finite action observed in the cited "
                        "model and prospective-engine outcomes."
                    ),
                    affected_paths=("$.value",),
                    evidence_summary=(
                        "Grounded jointly in every cited authenticated contrast."
                    ),
                    confidence=0.6,
                    evidence_contrast_ids=request.available_contrast_ids,
                    **advanced,
                ),
            ),
            telemetry=_telemetry(),
        )


@dataclass(slots=True)
class _Fixture:
    problem: _Problem
    generator: _SelectingGenerator
    planner: MatchedFiniteActionBlockPlanner
    engine: AgenticEvolutionEngine
    optimizer: BudgetedAgenticOptimizer
    traces: list[dict[str, object]]


def _fixture(*, forged_claim: bool = False) -> _Fixture:
    ids = DeterministicIdFactory(
        "matched_finite_forged" if forged_claim else "matched_finite_alias"
    )
    memory = InsightMemoryBank(id_factory=ids)
    entry, added = memory.add(
        InsightDraft(
            claim="A nearby signed integer intervention should improve the score.",
            trigger="The parent is at the neutral integer value.",
            mechanism="Moving the integer toward the target reduces absolute error.",
            affected_paths=("$.value",),
            evidence_summary="A prior diagnostic identified the local coordinate.",
            confidence=0.8,
            effect_predictions=(
                MetricEffectPrediction(
                    metric_id="objective:score",
                    direction=MetricEffectDirection.DECREASE,
                ),
            ),
            recommended_option_families=("local_value",),
            recommended_option_ids=("local.value_p2",),
            action_template="Select the exact signed local intervention.",
            falsification_condition="The score does not improve.",
        ),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        origin=InsightOrigin.MANUAL,
    )
    assert added
    problem = _Problem()
    reward = _reward()
    benchmark = AgenticBenchmark(
        problem=problem,
        reward=reward,
        finite_variation_catalogs=(_K8Catalog(),),
        hypothesis_compiler=_ExactAnchorCompiler(),
        finite_action_set_compiler=_K8SupportCompiler(),
    )
    traces: list[dict[str, object]] = []
    generator = _SelectingGenerator(forged_claim=forged_claim)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=11,
        evaluator_concurrency=2,
        trace_sink=traces.append,
        reward_policy=reward.score,
        reward_definition_hash=reward.definition_hash,
    )
    planner = MatchedFiniteActionBlockPlanner(
        benchmark=benchmark,
        engine=engine,
        ids=ids,
        memory=memory,
        card_reference=entry.reference,
        catalog_id=_K8Catalog.catalog_id,
        required_cardinality=8,
        context_projection_sha256=_sha("fixture K8 context"),
        endpoint_definition_sha256=reward.definition_hash,
        task_sha256=_sha("fixture K8 task"),
        pre_outcome_phase_commit_sha256=_sha("fixture K8 pre-outcome commit"),
        uniform_policy=TaskKeyedUniformFiniteActionPolicy(
            schedule_seed_sha256=_sha("fixture K8 prospective schedule")
        ),
        source_mode=FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
        phase="fixture_matched_k8",
    )
    generator.planner = planner
    archive = ParetoArchive(
        problem.objectives,
        evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
    )
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=archive,
        planner=planner,
        budget=OptimizerBudget(
            max_unique_evaluations=3,
            max_logical_llm_calls=1,
            max_generations=1,
        ),
        trace_sink=traces.append,
    )
    return _Fixture(problem, generator, planner, engine, optimizer, traces)


def test_k8_model_and_uniform_arms_share_support_and_alias_without_resampling() -> None:
    fixture = _fixture()
    result = asyncio.run(fixture.optimizer.run(({"value": 0, "held": 7},)))

    planner = fixture.planner
    authority = planner.authority
    uniform = planner.uniform_decision
    assert authority is not None and uniform is not None
    assert authority.support.cardinality == 8
    assert authority.current_outcome_access is False
    receipt = result.generation_receipts[0]
    adaptive, uninformed = receipt.slot_results
    assert adaptive.slot.slot_id == "A"
    assert uninformed.slot.slot_id == "U"
    assert adaptive.slot.plan.finite_action_set_authority is authority
    assert adaptive.slot.plan.finite_variation_contract == (
        authority.support.support_contract
    )
    assert planner.uniform_rank is not None
    assert planner.uniform_rank.current_outcome_access is False
    assert uninformed.slot.materialized is not None
    assert uninformed.slot.materialized.materialization_receipt_hash == (
        uniform.decision_sha256
    )

    model = adaptive.outcome.finite_action_decision
    assert model is not None
    assert model.authority_sha256 == uniform.authority_sha256
    assert model.support_sha256 == uniform.support_sha256
    assert model.option_id == uniform.option_id
    assert adaptive.outcome.candidate is not None
    assert uninformed.outcome.candidate is not None
    assert adaptive.outcome.candidate.candidate_id != (
        uninformed.outcome.candidate.candidate_id
    )
    assert adaptive.outcome.candidate.occurrence.configuration_hash == (
        uninformed.outcome.candidate.occurrence.configuration_hash
    )

    # Seed plus one physical child: the A=U collision remains two causal
    # occurrences but is neither resampled nor physically evaluated twice.
    cache = asyncio.run(fixture.engine.evaluation_cache_snapshot())
    assert result.final_state.unique_evaluations == 2
    selected_child = thaw_json(
        authority.support.options[model.selected_ordinal].option.child_configuration
    )
    assert type(selected_child) is dict
    assert fixture.problem.evaluations == [0, int(selected_child["value"])]
    assert cache["misses"] == 2
    assert cache["coalesced"] == 1

    sealed = next(
        event
        for event in fixture.traces
        if event["event_type"] == "finite_action_decision_sealed"
    )
    model_evaluated = next(
        event
        for event in fixture.traces
        if event["event_type"] == "candidate_evaluated"
        and event["candidate_id"] == adaptive.outcome.candidate.candidate_id.value
    )
    assert sealed["evaluator_entered"] is False
    assert sealed["authority_sha256"] == authority.authority_sha256
    assert sealed["sequence"] < model_evaluated["sequence"]


def test_engine_materialized_finite_action_contrast_is_citable() -> None:
    fixture = _fixture()
    result = asyncio.run(fixture.optimizer.run(({"value": 0, "held": 7},)))

    planner = fixture.planner
    authority = planner.authority
    decision = planner.uniform_decision
    assert authority is not None and decision is not None
    _, uniform = result.generation_receipts[0].slot_results
    invocation = uniform.slot.materialized
    assert invocation is not None
    assert invocation.materialized_finite_action_authority is authority
    assert invocation.materialized_finite_action_decision is decision
    assert uniform.outcome.prepared.materialized_finite_action_authority is authority
    assert uniform.outcome.prepared.materialized_finite_action_decision is decision
    # Engine provenance is distinct from the model-only outcome decision field.
    assert uniform.outcome.finite_action_decision is None

    reflector = _FiniteActionCitationReflector()
    fixture.engine.generator = reflector
    added = asyncio.run(
        fixture.engine.reflect(
            (uniform.outcome,),
            label="engine_finite_action_citation",
            min_insights=1,
            max_insights=1,
        )
    )

    assert len(added) == 1
    lineage = added[0].evidence_lineage
    assert lineage is not None
    assert lineage.cited_contrast_ids == lineage.available_contrast_ids
    (binding,) = lineage.finite_action_bindings
    assert binding.contrast_id == lineage.cited_contrast_ids[0]
    assert binding.option_id == decision.option_id
    assert binding.option_identity_sha256 == decision.option_identity_sha256
    assert binding.contract_identity_sha256 == (
        authority.support.support_contract.identity_sha256
    )
    prompt = reflector.requests[0].prompt
    assert '"finite_variation_option"' in prompt
    assert f'"option_id":"{decision.option_id}"' in prompt

    prepared_event = next(
        event
        for event in fixture.traces
        if event["event_type"] == "invocation_prepared"
        and event["candidate_id"] == invocation.candidate_id.value
    )
    assert prepared_event["materialized_finite_action_authority"][
        "authority_sha256"
    ] == authority.authority_sha256
    assert prepared_event["materialized_finite_action_decision"][
        "decision_sha256"
    ] == decision.decision_sha256


def test_engine_materialized_finite_action_provenance_is_exact_and_paired() -> None:
    fixture = _fixture()
    result = asyncio.run(fixture.optimizer.run(({"value": 0, "held": 7},)))

    adaptive, uniform = result.generation_receipts[0].slot_results
    invocation = uniform.slot.materialized
    model_decision = adaptive.outcome.finite_action_decision
    assert invocation is not None and model_decision is not None

    with pytest.raises(ValueError, match="authority and decision must be paired"):
        replace(invocation, materialized_finite_action_decision=None)
    with pytest.raises(ValueError, match="requires an engine selector"):
        replace(invocation, materialized_finite_action_decision=model_decision)
    wrong_child = replace(
        invocation.draft,
        configuration={"value": 0, "held": 7},
    )
    with pytest.raises(ValueError, match="bound to a different child"):
        replace(invocation, draft=wrong_child)


def test_exact_action_revision_citing_model_and_engine_actions_is_admitted() -> None:
    fixture = _fixture()
    result = asyncio.run(fixture.optimizer.run(({"value": 0, "held": 7},)))

    authority = fixture.planner.authority
    engine_decision = fixture.planner.uniform_decision
    assert authority is not None and engine_decision is not None
    adaptive, uniform = result.generation_receipts[0].slot_results
    model_decision = adaptive.outcome.finite_action_decision
    assert model_decision is not None
    exact_option_ids = tuple(
        sorted({model_decision.option_id, engine_decision.option_id})
    )
    reflector = _FiniteActionCitationReflector(
        recommended_option_ids=exact_option_ids
    )
    fixture.engine.generator = reflector
    contract = ReflectionInsightContract(
        required_metric_ids=("objective:score",),
        allowed_option_families=("local_value",),
        allowed_option_ids=tuple(
            sorted(
                option.option_id
                for option in authority.support.support_contract.options
            )
        ),
    )

    added = asyncio.run(
        fixture.engine.reflect(
            (adaptive.outcome, uniform.outcome),
            label="model_engine_exact_action_revision",
            min_insights=1,
            max_insights=1,
            insight_contract=contract,
            revision_predecessors=(fixture.planner.card_reference,),
        )
    )

    assert len(added) == 1
    revision = added[0]
    assert revision.reference.insight_id == fixture.planner.card_reference.insight_id
    assert revision.reference.version == fixture.planner.card_reference.version + 1
    assert revision.draft.recommended_option_ids == exact_option_ids
    lineage = revision.evidence_lineage
    assert lineage is not None
    assert len(lineage.cited_contrast_ids) == 2
    assert len(lineage.finite_action_bindings) == 2
    assert {binding.contrast_id for binding in lineage.finite_action_bindings} == set(
        lineage.cited_contrast_ids
    )
    assert {binding.option_id for binding in lineage.finite_action_bindings} == set(
        exact_option_ids
    )
    prompt = reflector.requests[0].prompt
    assert prompt.count('"finite_variation_option"') == 2


def test_forged_card_claim_is_rejected_before_the_model_candidate_evaluator() -> None:
    fixture = _fixture(forged_claim=True)
    result = asyncio.run(fixture.optimizer.run(({"value": 0, "held": 7},)))

    adaptive, uninformed = result.generation_receipts[0].slot_results
    assert adaptive.outcome.failure_stage == "candidate"
    assert adaptive.outcome.call_failure_type == "ValueError"
    assert adaptive.outcome.candidate is None
    assert adaptive.outcome.finite_action_decision is not None
    assert uninformed.outcome.candidate is not None
    uninformed_value = int(
        uninformed.outcome.candidate.configuration_dict["value"]
    )
    assert fixture.problem.evaluations == [0, uninformed_value]
    assert not any(
        event["event_type"] == "candidate_evaluated"
        and event["candidate_id"] == adaptive.outcome.prepared.candidate_id.value
        for event in fixture.traces
    )
    assert not any(
        event["event_type"] == "finite_action_decision_sealed"
        for event in fixture.traces
    )
    cache = asyncio.run(fixture.engine.evaluation_cache_snapshot())
    assert cache["misses"] == 2
    assert cache["coalesced"] == 0

"""Provider-free executable G0→G3 reference for the generic causal screen.

The fixture is intentionally small but not a shortcut around the production
boundaries.  It uses the public benchmark bundle, parent-bound finite catalogs,
the authenticated hypothesis compiler seam, the real agentic engine/evaluation
cache, the budgeted optimizer, strict treatment admission, wave-sealed causal
memory, engine-owned mate/recombination, and post-G3 reflection.  Only the model
provider and expensive evaluator are replaced by deterministic in-process fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_evolve.agentic import (
    AgenticBenchmark,
    AgenticOptimizerComposition,
    G3CurationSourceScope,
    G3PostsealCurationFactory,
    G3PostsealCurationInterceptor,
    G3PostsealCurationSpec,
    compose_agentic_optimizer,
)
from agent_evolve.application.agentic_evolution import (
    OperatorKind,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    OptimizerResult,
)
from agent_evolve.application.g3_causal_screen import (
    G1_DIAGNOSTIC_SLOT_IDS,
    G2_SLOT_IDS,
    G3_SLOT_IDS,
    G3CausalScreenPlanner,
    G3_SCREEN_BUDGET,
    FrozenDiagnosticPermutation,
    ParentBoundActionChoice,
    PreparedHypothesisMatrix,
)
from agent_evolve.application.g3_causal_validation import (
    G3CausalScreenResultValidationReceipt,
    validate_g3_causal_screen_result,
)
from agent_evolve.application.executable_hypothesis import (
    registered_source_evidence_sha256,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    InsightMemoryEntry,
    InsightOrigin,
    context_stratum_hash,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
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
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.executable_hypothesis import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii", errors="strict")).hexdigest()


MODEL_CATALOG_ID = "g3_fixture_model_axis"
MATE_CATALOG_ID = "g3_fixture_engine_axis"
ENDPOINT_SHA256 = _sha("g3 fixture absolute negative-cost endpoint v1")
P_D = {"a": 8, "b": 8}
P_H = {"a": 6, "b": 6}


class FixtureCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    a: int
    b: int


class FixtureProblem:
    candidate_model = FixtureCandidate
    objectives = (ObjectiveSpec("cost", "min"),)

    def __init__(self) -> None:
        self.evaluations: list[tuple[int, int]] = []

    @staticmethod
    def search_space_description() -> str:
        return "Minimize a+b over bounded integer coordinates a and b."

    @staticmethod
    def validate(configuration: object) -> bool:
        value = FixtureCandidate.model_validate(configuration, strict=True)
        return 0 <= value.a <= 20 and 0 <= value.b <= 20

    def evaluate(self, configuration: dict[str, Any]) -> dict[str, float]:
        value = FixtureCandidate.model_validate(configuration, strict=True)
        self.evaluations.append((value.a, value.b))
        return {"cost": float(value.a + value.b)}


class ModelAxisCatalog:
    catalog_id = MODEL_CATALOG_ID
    catalog_version = 1
    definition_sha256 = _sha("g3 fixture model-axis catalog v1")

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        if type(parent) is not dict:
            raise TypeError("model-axis parent must thaw to an exact object")
        parent_sha256 = typed_json_sha256(parent_configuration)
        a = int(parent["a"])
        b = int(parent["b"])
        return (
            FiniteVariationOption(
                option_id="model.aggressive",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": a - 3, "b": b}),
                family="model_only",
                description="Reduce a by three units.",
            ),
            FiniteVariationOption(
                option_id="model.conservative",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": a - 1, "b": b}),
                family="model_only",
                description="Reduce a by one unit.",
            ),
            FiniteVariationOption(
                option_id="model.neutral",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": a + 1, "b": b}),
                family="model_only",
                description="Neutral placebo action on the shared model axis.",
            ),
        )


class EngineAxisCatalog:
    catalog_id = MATE_CATALOG_ID
    catalog_version = 1
    definition_sha256 = _sha("g3 fixture engine-axis catalog v1")

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        if type(parent) is not dict:
            raise TypeError("engine-axis parent must thaw to an exact object")
        parent_sha256 = typed_json_sha256(parent_configuration)
        return (
            FiniteVariationOption(
                option_id="engine.mate",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(
                    {"a": int(parent["a"]), "b": int(parent["b"]) - 2}
                ),
                family="engine_only",
                description="Reduce the disjoint engine coordinate b by two.",
            ),
        )


class FixtureHypothesisCompiler:
    policy_id = "g3_fixture_hypothesis_compiler"
    policy_version = 1
    definition_sha256 = _sha("g3 fixture hypothesis compiler v1")

    def __init__(self, option_by_insight_id: dict[str, str]) -> None:
        self.option_by_insight_id = dict(option_by_insight_id)
        self.requests: list[HypothesisCompilationRequest] = []

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        self.requests.append(request)
        option_id = self.option_by_insight_id[request.reference.insight_id.value]
        option = request.finite_contract.resolve(option_id)
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
            recommended_option_families=("model_only",),
            affected_paths=("$.a",),
            held_fixed_paths=("$.b",),
            effect_predictions=request.insight.effect_predictions,
            falsification_condition=(
                request.insight.falsification_condition
                or "The held-out cost does not change as predicted."
            ),
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


def _telemetry(response_id: str) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="provider-free/g3-fixture",
        resolved_model="provider-free/g3-fixture",
        resolved_provider="in-process",
        provider_response_id=response_id,
        finish_reason="fixture",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class FixtureGenerator:
    def __init__(self, option_by_insight_id: dict[str, str]) -> None:
        self.option_by_insight_id = dict(option_by_insight_id)
        self.proposal_requests: list[VariationGenerationRequest] = []
        self.reflection_requests: list[ReflectionGenerationRequest] = []

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        self.proposal_requests.append(request)
        contract = request.finite_variation_contract
        if contract is None:
            raise AssertionError("G3 fixture expects finite option selection")
        selected_id = next(
            insight_id
            for insight_id in self.option_by_insight_id
            if f'"insight_id":"{insight_id}"' in request.prompt
        )
        option = contract.resolve(self.option_by_insight_id[selected_id])
        return VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=contract.identity_sha256,
                design_rationale="Execute the assigned sealed finite treatment.",
                claimed_insight_ids=(selected_id,),
            ),
            telemetry=_telemetry(request.call_id.value),
        )

    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        self.reflection_requests.append(request)
        if not request.available_contrast_ids:
            insights: tuple[InsightDraft, ...] = ()
        else:
            insights = (
                InsightDraft(
                    claim="One sealed batch association merits an independent retest.",
                    trigger="A future parent admits the same bounded action family.",
                    mechanism="The current evidence is associational and remains quarantined.",
                    affected_paths=("$.a",),
                    evidence_summary="One exact sealed contrast is cited without causal promotion.",
                    confidence=0.1,
                    evidence_contrast_ids=(request.available_contrast_ids[0],),
                    effect_predictions=(
                        MetricEffectPrediction(
                            metric_id="objective:cost",
                            direction=MetricEffectDirection.DECREASE,
                        ),
                    ),
                    recommended_option_families=("model_only",),
                    action_template="Retest one bounded model-axis action.",
                    falsification_condition="The association does not replicate.",
                ),
            )
        return ReflectionGenerationResult(
            insights=insights,
            telemetry=_telemetry(request.call_id.value),
        )


class FailingCurationFixtureGenerator(FixtureGenerator):
    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        self.reflection_requests.append(request)
        raise RuntimeError("provider-free injected curation transport failure")


class OverproducingCurationFixtureGenerator(FixtureGenerator):
    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        result = await super().reflect(request)
        if len(result.insights) != 1:
            raise RuntimeError("overproduction fixture requires one base card")
        return ReflectionGenerationResult(
            insights=(
                result.insights[0],
                replace(
                    result.insights[0],
                    claim="A second illicit revision exceeds the frozen request.",
                ),
            ),
            telemetry=result.telemetry,
        )


def _active_draft(name: str) -> InsightDraft:
    return InsightDraft(
        claim=f"The {name} model-axis action should reduce absolute cost.",
        trigger="The parent admits a bounded reduction of coordinate a.",
        mechanism="Absolute cost contains coordinate a additively.",
        affected_paths=("$.a",),
        evidence_summary="Frozen developmental evidence motivates a transfer test.",
        confidence=0.5,
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="objective:cost",
                direction=MetricEffectDirection.DECREASE,
            ),
        ),
        recommended_option_families=("model_only",),
        recommended_option_ids=(f"source.{name}",),
        action_template="Select the matching parent-bound model-axis action.",
        falsification_condition="Held-out absolute cost does not decrease.",
    )


def _neutral_draft() -> InsightDraft:
    return InsightDraft(
        claim="The evidence-free sham makes no directional claim.",
        trigger="The held-out parent admits the preselected sham action.",
        mechanism="This card is a structural placebo only.",
        affected_paths=("$.a",),
        evidence_summary="No empirical evidence; causal credit is forbidden.",
        confidence=0.0,
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="objective:cost",
                direction=MetricEffectDirection.UNKNOWN,
            ),
        ),
        recommended_option_families=("model_only",),
        recommended_option_ids=("model.neutral",),
        action_template="Select only the sealed neutral option.",
        falsification_condition="Not applicable to an evidence-free placebo.",
    )


def _prepared_matrix(
    *,
    role: str,
    parent: dict[str, int],
    entries: tuple[InsightMemoryEntry, InsightMemoryEntry],
    benchmark: AgenticBenchmark,
    compiler: FixtureHypothesisCompiler,
    context_sha256: str,
) -> PreparedHypothesisMatrix:
    contract = benchmark.bind_finite_variation(MODEL_CATALOG_ID, parent)
    parent_id_suffix = {
        "diagnostic_parent": "d",
        "hypothesis_parent": "h",
    }[role]
    requests = tuple(
        HypothesisCompilationRequest(
            reference=entry.reference,
            insight=entry.draft,
            source_evidence_sha256=registered_source_evidence_sha256(entry),
            requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
            source_operator_kinds=entry.applicable_operator_kinds,
            parent_candidate_id=CandidateId(
                f"candidate_prepared_{parent_id_suffix}"
            ),
            parent_configuration_sha256=contract.parent_configuration_sha256,
            finite_contract=contract,
            context_projection_sha256=context_sha256,
            endpoint_definition_sha256=ENDPOINT_SHA256,
        )
        for entry in entries
    )
    receipts = tuple(compiler.compile(request) for request in requests)
    return PreparedHypothesisMatrix(
        parent_role=role,
        requests=requests,
        receipts=receipts,
    )


class FixtureG3PlannerFactory:
    """Deferred fixture policy built against public-composition identities."""

    def __init__(
        self,
        *,
        reward: RewardPolicyBinding,
        compiler: FixtureHypothesisCompiler,
        active_entries: tuple[InsightMemoryEntry, InsightMemoryEntry],
        neutral_entry: InsightMemoryEntry,
        trace_sink,
    ) -> None:
        self.reward = reward
        self.compiler = compiler
        self.active_entries = active_entries
        self.neutral_entry = neutral_entry
        self.trace_sink = trace_sink
        self.planner: G3CausalScreenPlanner | None = None
        self.runtime_identities: tuple[object, object, object, object] | None = None

    def build(self, *, benchmark, engine, id_factory, memory):
        if self.planner is not None:
            raise RuntimeError("fixture planner factory may be invoked only once")
        self.runtime_identities = (benchmark, engine, id_factory, memory)
        context_sha256 = context_stratum_hash(
            problem_id=engine.problem_id,
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            phase="g3_causal_screen",
        )
        prepared = (
            _prepared_matrix(
                role="diagnostic_parent",
                parent=P_D,
                entries=self.active_entries,
                benchmark=benchmark,
                compiler=self.compiler,
                context_sha256=context_sha256,
            ),
            _prepared_matrix(
                role="hypothesis_parent",
                parent=P_H,
                entries=self.active_entries,
                benchmark=benchmark,
                compiler=self.compiler,
                context_sha256=context_sha256,
            ),
        )
        p_h_model_contract = benchmark.bind_finite_variation(MODEL_CATALOG_ID, P_H)
        p_h_mate_contract = benchmark.bind_finite_variation(MATE_CATALOG_ID, P_H)
        choice_definition = _sha("g3 fixture public hash choices v1")
        active_references = tuple(
            value.reference for value in self.active_entries
        )
        self.planner = G3CausalScreenPlanner(
            benchmark=benchmark,
            engine=engine,
            ids=id_factory,
            memory=memory,
            reward_binding=self.reward,
            active_references=active_references,
            neutral_reference=self.neutral_entry.reference,
            diagnostic_permutation=FrozenDiagnosticPermutation(
                active_references=active_references,
                permutation_rank=1,
                randomization_policy_id="public_uniform_permutation",
                randomization_policy_version=1,
                randomization_definition_sha256=_sha(
                    "g3 fixture external uniform permutation sampler v1"
                ),
            ),
            prepared_hypothesis_matrices=prepared,
            model_catalog_id=MODEL_CATALOG_ID,
            neutral_choice=ParentBoundActionChoice.seal(
                role="neutral_sham",
                contract=p_h_model_contract,
                option_id="model.neutral",
                selection_policy_id="public_hash_neutral",
                selection_policy_version=1,
                selection_policy_definition_sha256=choice_definition,
            ),
            mate_choice=ParentBoundActionChoice.seal(
                role="orthogonal_mate",
                contract=p_h_mate_contract,
                option_id="engine.mate",
                selection_policy_id="public_hash_mate",
                selection_policy_version=1,
                selection_policy_definition_sha256=choice_definition,
            ),
            diagnostic_parent_configuration_sha256=typed_json_sha256(
                freeze_json(P_D)
            ),
            hypothesis_parent_configuration_sha256=typed_json_sha256(
                freeze_json(P_H)
            ),
            endpoint_definition_sha256=ENDPOINT_SHA256,
            estimand_stratum_sha256=_sha("g3 fixture transfer estimand v1"),
            no_yield_reward=self.reward.failure_score,
            trace_sink=self.trace_sink,
        )
        return self.planner


@dataclass(frozen=True, slots=True)
class ProviderFreeG3Run:
    result: OptimizerResult
    planner: G3CausalScreenPlanner
    generator: FixtureGenerator
    problem: FixtureProblem
    compiler: FixtureHypothesisCompiler
    curation: G3PostsealCurationInterceptor
    composition: AgenticOptimizerComposition
    planner_factory: FixtureG3PlannerFactory
    curation_factory: G3PostsealCurationFactory
    validation_receipt: G3CausalScreenResultValidationReceipt
    evaluation_cache_snapshot: dict[str, int | None]
    traces: tuple[dict[str, object], ...]


async def run_provider_free_g3(
    *,
    fail_curation: bool = False,
    overproduce_curation: bool = False,
) -> ProviderFreeG3Run:
    if type(fail_curation) is not bool or type(overproduce_curation) is not bool:
        raise TypeError("curation failure controls must be exact bools")
    if fail_curation and overproduce_curation:
        raise ValueError("curation failure controls are mutually exclusive")
    ids = DeterministicIdFactory("g3_provider_free")
    memory = InsightMemoryBank(id_factory=ids)
    aggressive, aggressive_added = memory.add(
        _active_draft("aggressive"),
        applicable_operator_kinds=("mutation",),
        origin=InsightOrigin.MANUAL,
    )
    conservative, conservative_added = memory.add(
        _active_draft("conservative"),
        applicable_operator_kinds=("mutation",),
        origin=InsightOrigin.MANUAL,
    )
    neutral, neutral_added = memory.add(
        _neutral_draft(),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        origin=InsightOrigin.MANUAL,
    )
    if not (aggressive_added and conservative_added and neutral_added):
        raise RuntimeError("fixture memory entries were not newly registered")
    active_entries = tuple(
        sorted((aggressive, conservative), key=lambda value: value.reference)
    )
    active_references = tuple(value.reference for value in active_entries)
    option_by_insight_id = {
        aggressive.reference.insight_id.value: "model.aggressive",
        conservative.reference.insight_id.value: "model.conservative",
        neutral.reference.insight_id.value: "model.neutral",
    }
    compiler = FixtureHypothesisCompiler(option_by_insight_id)
    problem = FixtureProblem()

    def absolute_reward(child, parents, objectives) -> float:
        del parents, objectives
        return -float(child.objective_map["cost"])

    reward = RewardPolicyBinding(
        score=absolute_reward,
        definition_hash=ENDPOINT_SHA256,
        failure_score=-100.0,
    )
    benchmark = AgenticBenchmark(
        problem=problem,
        reward=reward,
        finite_variation_catalogs=(ModelAxisCatalog(), EngineAxisCatalog()),
        hypothesis_compiler=compiler,
    )
    generator: FixtureGenerator
    if fail_curation:
        generator = FailingCurationFixtureGenerator(option_by_insight_id)
    elif overproduce_curation:
        generator = OverproducingCurationFixtureGenerator(option_by_insight_id)
    else:
        generator = FixtureGenerator(option_by_insight_id)
    traces: list[dict[str, object]] = []
    planner_factory = FixtureG3PlannerFactory(
        reward=reward,
        compiler=compiler,
        active_entries=active_entries,
        neutral_entry=neutral,
        trace_sink=traces.append,
    )
    curation_spec = G3PostsealCurationSpec(
        insight_contract=ReflectionInsightContract(
            required_metric_ids=("objective:cost",),
            allowed_option_families=("model_only",),
        ),
        source_scope=G3CurationSourceScope(
            policy_id="all_g1_g3_outcomes",
            policy_version=1,
            policy_definition_sha256=_sha(
                "provider-free fixture ordered all G1-G3 outcome slots v1"
            ),
            slot_ids=(
                *G1_DIAGNOSTIC_SLOT_IDS,
                *G2_SLOT_IDS,
                *G3_SLOT_IDS,
            ),
        ),
    )
    curation_factory = G3PostsealCurationFactory(spec=curation_spec)
    composition = compose_agentic_optimizer(
        benchmark,
        generator=generator,
        planner_factory=planner_factory,
        budget=G3_SCREEN_BUDGET,
        seed=7,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=4,
        engine_trace_sink=traces.append,
        optimizer_trace_sink=traces.append,
        feedback_interceptor_factory=curation_factory,
        max_output_tokens=384_000,
    )
    planner = composition.planner
    curation = composition.feedback_interceptor
    if type(planner) is not G3CausalScreenPlanner:
        raise RuntimeError("public composition returned the wrong G3 planner")
    if type(curation) is not G3PostsealCurationInterceptor:
        raise RuntimeError("public composition returned the wrong curation policy")
    result = await composition.optimizer.run((P_D, P_H))
    evaluation_cache_snapshot = await composition.engine.evaluation_cache_snapshot()
    if curation.curation_authority is None or curation.curation_receipt is None:
        raise RuntimeError("post-G3 curation did not publish typed evidence")
    validation_receipt = validate_g3_causal_screen_result(
        result,
        planner=planner,
        evaluation_cache_snapshot=evaluation_cache_snapshot,
        curation_spec=curation.spec,
        curation_authority=curation.curation_authority,
        curation_receipt=curation.curation_receipt,
    )
    return ProviderFreeG3Run(
        result=result,
        planner=planner,
        generator=generator,
        problem=problem,
        compiler=compiler,
        curation=curation,
        composition=composition,
        planner_factory=planner_factory,
        curation_factory=curation_factory,
        validation_receipt=validation_receipt,
        evaluation_cache_snapshot=evaluation_cache_snapshot,
        traces=tuple(traces),
    )


def main() -> None:
    run = asyncio.run(run_provider_free_g3())
    result = run.result
    print(
        {
            "result_hash": result.result_hash,
            "logical_llm_calls": result.final_state.logical_llm_calls,
            "unique_evaluations": result.final_state.unique_evaluations,
            "candidate_occurrences": len(result.final_state.candidates),
            "generation_slots": [
                [value.slot.slot_id for value in receipt.slot_results]
                for receipt in result.generation_receipts
            ],
            "curated_entries": len(run.curation.curated_entries),
            "curation_status": run.validation_receipt.curation_status,
            "terminal_validation_receipt": (
                run.validation_receipt.terminal_state_receipt_sha256
            ),
            "mechanism_decision": (
                run.curation.terminal_validation_receipt.mechanism_decision.to_record()
                if run.curation.terminal_validation_receipt is not None
                else None
            ),
        }
    )


if __name__ == "__main__":
    main()

"""Provider-free E2E checks for compiled parent-bound treatments."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
)
from agent_evolve.application.budgeted_optimizer import _invocation_plan_record
from agent_evolve.application.executable_hypothesis import (
    compile_registered_hypothesis_treatment,
    registered_source_evidence_sha256,
)
from agent_evolve.application.insight_memory import InsightMemoryBank, InsightOrigin
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import JsonPath, ObjectKey
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, typed_json_sha256
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.treatment_compliance import TreatmentActionBinding
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.executable_hypothesis import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
)


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    x: int
    y: int


class _Problem:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec("score", "min"),)

    def __init__(self) -> None:
        self.evaluations = 0

    @staticmethod
    def search_space_description() -> str:
        return "Minimize two independent integer coordinates."

    @staticmethod
    def validate(configuration: object) -> bool:
        _Candidate.model_validate(configuration, strict=True)
        return True

    def evaluate(self, configuration: dict[str, Any]) -> dict[str, float]:
        self.evaluations += 1
        return {"score": float(configuration["x"] + configuration["y"])}


class _Catalog:
    catalog_id = "fixture_compiled_treatment"
    catalog_version = 1
    definition_sha256 = hashlib.sha256(b"fixture compiled catalog v1").hexdigest()

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent_sha = typed_json_sha256(parent_configuration)
        return (
            FiniteVariationOption(
                option_id="shape.raise_x",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json({"x": 6, "y": 5}),
                family="shape_only",
                description="Raise the x coordinate by one.",
            ),
            FiniteVariationOption(
                option_id="shape.lower_x",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json({"x": 4, "y": 5}),
                family="shape_only",
                description="Lower the x coordinate by one.",
            ),
            FiniteVariationOption(
                option_id="engine.lower_y",
                parent_configuration_sha256=parent_sha,
                child_configuration=freeze_json({"x": 5, "y": 3}),
                family="engine_only",
                description="Lower the independent y coordinate.",
            ),
        )


class _Compiler:
    policy_id = "fixture_parent_hypothesis_compiler"
    policy_version = 1
    definition_sha256 = hashlib.sha256(b"fixture hypothesis compiler v1").hexdigest()

    def __init__(self) -> None:
        self.compile_count = 0

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        self.compile_count += 1
        option = request.finite_contract.resolve("shape.lower_x")
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
            recommended_option_families=("shape_only",),
            affected_paths=("$.x",),
            held_fixed_paths=("$.y",),
            effect_predictions=request.insight.effect_predictions,
            falsification_condition=request.insight.falsification_condition or "missing",
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


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/compiled-treatment",
        resolved_model="offline/compiled-treatment",
        resolved_provider="provider-free",
        provider_response_id="fixture-response",
        finish_reason="fixture",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _Generator:
    def __init__(self, insight_id: str) -> None:
        self.insight_id = insight_id
        self.requests: list[VariationGenerationRequest] = []

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        self.requests.append(request)
        contract = request.finite_variation_contract
        assert contract is not None
        option = contract.resolve("shape.lower_x")
        return VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=contract.identity_sha256,
                design_rationale="Apply the exact compiled hypothesis treatment.",
                claimed_insight_ids=(self.insight_id,),
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request: object) -> ReflectionGenerationResult:
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def _insight() -> InsightDraft:
    return InsightDraft(
        claim="Lowering x should lower the total objective.",
        trigger="The parent x coordinate remains reducible.",
        mechanism="The objective adds x directly.",
        affected_paths=("$.x",),
        evidence_summary="A registered source snapshot supports this hypothesis.",
        confidence=0.8,
        effect_predictions=(
            MetricEffectPrediction(
                metric_id="objective:score",
                direction=MetricEffectDirection.DECREASE,
            ),
        ),
        recommended_option_families=("shape_only",),
        # This ID belongs to the source parent.  The compiler maps it to the
        # exact current-parent option while preserving the immutable family.
        recommended_option_ids=("shape.historical_parent",),
        action_template="Select a parent-bound x-lowering shape action.",
        falsification_condition="The objective does not decrease.",
    )


async def _run_compiled_treatment():
    ids = DeterministicIdFactory("compiled_treatment_e2e")
    memory = InsightMemoryBank(id_factory=ids)
    entry, added = memory.add(
        _insight(),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        origin=InsightOrigin.MANUAL,
    )
    assert added
    compiler = _Compiler()
    problem = _Problem()
    benchmark = AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(_Catalog(),),
        hypothesis_compiler=compiler,
    )
    generator = _Generator(entry.reference.insight_id.value)
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=3,
        trace_sink=traces.append,
    )
    parent = await engine.register_seed({"x": 5, "y": 5}, label="parent")
    compiled = benchmark.compile_registered_hypothesis_treatment(
        catalog_id=_Catalog.catalog_id,
        parent_candidate_id=parent.candidate_id,
        parent_configuration=parent.configuration,
        entry=entry,
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        context_projection_sha256="c" * 64,
        endpoint_definition_sha256="d" * 64,
    )
    plan = InvocationPlan(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=1,
        label="compiled_treatment",
        allowed_top_level=("x", "y"),
        mutation_contract=MutationContract(
            editable_paths=(
                JsonPath((ObjectKey("x"),)),
                JsonPath((ObjectKey("y"),)),
            ),
            max_changed_paths=1,
            max_operations=1,
        ),
        mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
        finite_variation_contract=compiled.request.finite_contract,
        quarantine_test_insights=(entry.reference,),
        insight_treatment_requirement=compiled.requirement,
        compiled_hypothesis_treatment=compiled,
        phase="compiled_treatment_e2e",
    )
    (outcome,) = await engine.run_invocations((plan,))
    return benchmark, compiler, problem, entry, compiled, plan, outcome, traces, generator


def test_compiled_parent_action_survives_prompt_preflight_and_evaluation() -> None:
    (
        _,
        compiler,
        problem,
        entry,
        compiled,
        plan,
        outcome,
        traces,
        generator,
    ) = asyncio.run(_run_compiled_treatment())

    assert compiler.compile_count == 1
    assert problem.evaluations == 2  # seed plus one fresh treatment child
    assert outcome.candidate is not None and outcome.failure_stage is None
    assert dict(outcome.candidate.objectives) == {"score": 9.0}
    preflight = outcome.prepared.treatment_preflight_receipt
    assert preflight is not None and preflight.passed
    assert tuple(item.option_id for item in preflight.compatible_actions) == (
        "shape.lower_x",
    )
    assert compiled.treatment_evidence.recommended_option_ids == ("shape.lower_x",)
    assert entry.draft.recommended_option_ids == ("shape.historical_parent",)
    assert compiled.request.source_evidence_sha256 == (
        registered_source_evidence_sha256(entry)
    )
    prompt = generator.requests[0].prompt
    assert compiled.binding_sha256 in prompt
    assert '"treatment_binding_kind":"compiled_hypothesis_v1"' in prompt
    record = _invocation_plan_record(plan)
    assert record["compiled_hypothesis_treatment"]["binding_sha256"] == (
        compiled.binding_sha256
    )
    event = next(
        item for item in traces if item["event_type"] == "treatment_preflight_completed"
    )
    assert event["requirement"]["compiled_hypothesis_treatment"][
        "binding_sha256"
    ] == compiled.binding_sha256


def test_compiled_binding_cannot_be_dropped_or_swapped_without_plan_change() -> None:
    _, _, _, _, compiled, plan, _, _, _ = asyncio.run(_run_compiled_treatment())
    old_record = _invocation_plan_record(plan)
    legacy_record = _invocation_plan_record(
        replace(plan, compiled_hypothesis_treatment=None)
    )
    assert old_record != legacy_record
    assert legacy_record["compiled_hypothesis_treatment"] is None

    # The receipt cannot authenticate the forged request, before a plan exists.
    with pytest.raises(ValueError, match="different request"):
        replace(
            compiled,
            request=replace(
                compiled.request,
                context_projection_sha256="e" * 64,
            ),
        )
    assert old_record["compiled_hypothesis_treatment"]["binding_sha256"] == (
        compiled.binding_sha256
    )


def test_benchmark_rejects_compiler_identity_swap_before_compile() -> None:
    ids = DeterministicIdFactory("compiled_treatment_swap")
    memory = InsightMemoryBank(id_factory=ids)
    entry, _ = memory.add(
        _insight(),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        origin=InsightOrigin.MANUAL,
    )
    compiler = _Compiler()
    benchmark = AgenticBenchmark(
        problem=_Problem(),
        finite_variation_catalogs=(_Catalog(),),
        hypothesis_compiler=compiler,
    )
    compiler.definition_sha256 = "f" * 64

    with pytest.raises(ValueError, match="identity changed"):
        benchmark.compile_registered_hypothesis_treatment(
            catalog_id=_Catalog.catalog_id,
            parent_candidate_id=CandidateId("candidate_parent"),
            parent_configuration={"x": 5, "y": 5},
            entry=entry,
            requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
            context_projection_sha256="c" * 64,
            endpoint_definition_sha256="d" * 64,
        )
    assert compiler.compile_count == 0


def test_builder_rejects_source_entry_injection() -> None:
    _, _, _, entry, compiled, _, _, _, _ = asyncio.run(_run_compiled_treatment())
    injected = replace(entry, initial_score=1.0)
    compiler = _Compiler()

    with pytest.raises(ValueError, match="registered source evidence"):
        compile_registered_hypothesis_treatment(
            entry=injected,
            request=compiled.request,
            compiler=compiler,
        )
    assert compiler.compile_count == 0


def test_high_level_builder_rejects_compiler_claiming_another_identity() -> None:
    class _ForeignReceiptCompiler(_Compiler):
        definition_sha256 = "f" * 64

        def compile(
            self,
            request: HypothesisCompilationRequest,
        ) -> HypothesisCompilationReceipt:
            self.compile_count += 1
            return _Compiler().compile(request)

    _, _, _, entry, compiled, _, _, _, _ = asyncio.run(_run_compiled_treatment())
    compiler = _ForeignReceiptCompiler()

    with pytest.raises(ValueError, match="frozen compiler identity"):
        compile_registered_hypothesis_treatment(
            entry=entry,
            request=compiled.request,
            compiler=compiler,
        )
    assert compiler.compile_count == 1


def test_high_level_builder_rejects_identity_mutation_during_compile() -> None:
    class _MutatingCompiler(_Compiler):
        def compile(
            self,
            request: HypothesisCompilationRequest,
        ) -> HypothesisCompilationReceipt:
            receipt = super().compile(request)
            self.definition_sha256 = "f" * 64
            return receipt

    _, _, _, entry, compiled, _, _, _, _ = asyncio.run(_run_compiled_treatment())
    compiler = _MutatingCompiler()

    with pytest.raises(ValueError, match="compiler identity changed"):
        compile_registered_hypothesis_treatment(
            entry=entry,
            request=compiled.request,
            compiler=compiler,
        )
    assert compiler.compile_count == 1


def test_high_level_builder_rejects_change_and_claim_identity_toctou() -> None:
    class _ChangeAndClaimCompiler(_Compiler):
        def compile(
            self,
            request: HypothesisCompilationRequest,
        ) -> HypothesisCompilationReceipt:
            self.definition_sha256 = "f" * 64
            return super().compile(request)

    _, _, _, entry, compiled, _, _, _, _ = asyncio.run(_run_compiled_treatment())
    compiler = _ChangeAndClaimCompiler()

    with pytest.raises(ValueError, match="identity changed during"):
        compile_registered_hypothesis_treatment(
            entry=entry,
            request=compiled.request,
            compiler=compiler,
        )
    assert compiler.compile_count == 1


def test_high_level_builder_rejects_request_and_source_mutation_toctou() -> None:
    class _RequestMutatingCompiler(_Compiler):
        def compile(
            self,
            request: HypothesisCompilationRequest,
        ) -> HypothesisCompilationReceipt:
            object.__setattr__(
                request.insight,
                "claim",
                "Hostile compiler rewrote the immutable registered claim.",
            )
            return super().compile(request)

    _, _, _, entry, compiled, _, _, _, _ = asyncio.run(_run_compiled_treatment())
    compiler = _RequestMutatingCompiler()

    with pytest.raises(ValueError, match="request changed during"):
        compile_registered_hypothesis_treatment(
            entry=entry,
            request=compiled.request,
            compiler=compiler,
        )
    assert compiler.compile_count == 1

"""Generic contracts for pre-evaluator insight-treatment administration."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_evolve.agentic import (
    AgenticEvolutionEngine,
    FiniteTreatmentAction,
    FiniteVariationContract,
    FiniteVariationOption,
    FiniteVariationSelectionDraft,
    InsightDraft,
    InsightMemoryBank,
    InsightTreatmentRequirement,
    InvocationPlan,
    JsonPath,
    MetricEffectDirection,
    MetricEffectPrediction,
    MutationContract,
    MutationResponseMode,
    ObjectKey,
    OperatorKind,
    StrictTreatmentCompliancePolicy,
    TreatmentAdmissionRequest,
    TreatmentActionBinding,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentComplianceViolation,
    TreatmentInsightEvidence,
    TreatmentPreflightRequest,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.application.budgeted_optimizer import _invocation_plan_record
from agent_evolve.application.insight_memory import InsightOrigin
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.domain.ids import InsightId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    ReflectionGenerationResult,
)


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int
    y: int


class _CountingProblem:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec("score", "min"),)

    def __init__(self) -> None:
        self.evaluations = 0

    @staticmethod
    def search_space_description() -> str:
        return "A generic two-coordinate co-optimization fixture."

    @staticmethod
    def validate(configuration: object) -> bool:
        _Candidate.model_validate(configuration, strict=True)
        return True

    def evaluate(self, configuration: dict[str, Any]) -> dict[str, float]:
        self.evaluations += 1
        return {"score": float(configuration["x"] + configuration["y"])}


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/treatment-fixture",
        resolved_model="offline/treatment-fixture",
        resolved_provider="provider-free",
        provider_response_id="offline-treatment-response",
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


class _SelectingGenerator:
    def __init__(self, *, option_id: str, claims: tuple[str, ...]) -> None:
        self.option_id = option_id
        self.claims = claims
        self.requests: list[VariationGenerationRequest] = []

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        self.requests.append(request)
        contract = request.finite_variation_contract
        if contract is None:  # pragma: no cover - fixture contract.
            raise AssertionError("treatment fixture requires a finite contract")
        option = contract.resolve(self.option_id)
        return VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=contract.identity_sha256,
                design_rationale="Instantiate the assigned bounded treatment.",
                claimed_insight_ids=self.claims,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request: object) -> ReflectionGenerationResult:
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


async def _run_treatment(
    *,
    option_id: str,
    claims: str,
    claim_mode: TreatmentClaimMode = TreatmentClaimMode.EXACT_REQUIRED,
):
    ids = DeterministicIdFactory(
        f"treatment_{option_id}_{claims or 'none'}_{claim_mode.value}"
    )
    memory = InsightMemoryBank(id_factory=ids)
    entry, added = memory.add(
        InsightDraft(
            claim="A bounded x action is the assigned transferable mechanism.",
            trigger="The current parent admits the same sealed action family.",
            mechanism="Select a sealed shape-only action that changes x.",
            affected_paths=("$.x",),
            evidence_summary="Provider-free treatment fixture evidence.",
            confidence=0.5,
            effect_predictions=(
                MetricEffectPrediction(
                    metric_id="objective:score",
                    direction=MetricEffectDirection.INCREASE,
                ),
            ),
            recommended_option_families=("shape_only",),
            recommended_option_ids=("shape.raise_x",),
            action_template="Select one sealed shape-only action.",
            falsification_condition="The selected action does not change x.",
        ),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        origin=InsightOrigin.MANUAL,
    )
    assert added
    parent_configuration = freeze_json({"x": 0, "y": 0})
    parent_sha256 = typed_json_sha256(parent_configuration)
    contract = FiniteVariationContract(
        catalog_id="generic_treatment_fixture",
        catalog_version=1,
        catalog_definition_sha256="a" * 64,
        parent_configuration=parent_configuration,
        options=(
            FiniteVariationOption(
                option_id="shape.raise_x",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 1, "y": 0}),
                family="shape_only",
                description="Raise x by one.",
            ),
            FiniteVariationOption(
                option_id="shape.lower_x",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": -1, "y": 0}),
                family="shape_only",
                description="Lower x by one.",
            ),
            FiniteVariationOption(
                option_id="control.raise_y",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 0, "y": 1}),
                family="control_only",
                description="Raise y by one.",
            ),
        ),
    )
    claimed = () if not claims else (entry.reference.insight_id.value,)
    generator = _SelectingGenerator(option_id=option_id, claims=claimed)
    problem = _CountingProblem()
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=7,
        trace_sink=traces.append,
    )
    parent = await engine.register_seed({"x": 0, "y": 0}, label="parent")
    evidence = TreatmentInsightEvidence(
        reference=entry.reference,
        insight_content_sha256=entry.draft.content_sha256,
        applicable_operator_kinds=entry.applicable_operator_kinds,
        affected_paths=entry.draft.affected_paths,
        recommended_option_families=entry.draft.recommended_option_families,
        recommended_option_ids=entry.draft.recommended_option_ids,
    )
    required_option = contract.resolve("shape.raise_x")
    requirement = InsightTreatmentRequirement(
        insight_bindings=(evidence.binding(),),
        finite_contract_sha256=contract.identity_sha256,
        allowed_actions=(
            TreatmentActionBinding(
                option_id=required_option.option_id,
                option_identity_sha256=required_option.identity_sha256,
            ),
        ),
        claim_mode=claim_mode,
        assignment_role=TreatmentAssignmentRole.ACTIVE,
    )
    plan = InvocationPlan(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=1,
        label="generic_treatment",
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
        finite_variation_contract=contract,
        quarantine_test_insights=(entry.reference,),
        insight_treatment_requirement=requirement,
        phase="generic_treatment_test",
    )
    (outcome,) = await engine.run_invocations((plan,))
    return problem, generator, traces, plan, outcome


def test_incompatible_treatment_is_distinct_no_yield_before_evaluator() -> None:
    problem, _, traces, _, outcome = asyncio.run(
        _run_treatment(option_id="shape.lower_x", claims="exact")
    )

    assert problem.evaluations == 1  # The seed only.
    assert outcome.candidate is None
    assert outcome.reward == -1.0
    assert outcome.failure_stage == "treatment_noncompliance"
    assert outcome.call_failure_type == "TreatmentComplianceRejected"
    assert outcome.terminal_evaluation is None
    receipt = outcome.treatment_admission_receipt
    assert receipt is not None and not receipt.passed
    assert receipt.evaluator_entered is False
    assert receipt.violations == (
        TreatmentComplianceViolation.SELECTED_ACTION_INCOMPATIBLE,
    )
    assert receipt.selected_action.family == "shape_only"
    assert receipt.selected_action.changed_paths == ("$.x",)
    assert type(hash(receipt)) is int
    admissions = [
        event
        for event in traces
        if event["event_type"] == "treatment_admission_completed"
    ]
    assert len(admissions) == 1
    assert admissions[0]["evaluator_entered"] is False
    assert admissions[0]["admission"]["receipt_sha256"] == receipt.receipt_sha256
    assert not any(
        event["event_type"] == "candidate_evaluated" for event in traces
    )
    rejected = next(
        event
        for event in traces
        if event["event_type"] == "treatment_compliance_rejected"
    )
    assert rejected["evaluator_entered"] is False
    completed = next(
        event for event in traces if event["event_type"] == "invocation_completed"
    )
    assert completed["failure_stage"] == "treatment_noncompliance"
    assert completed["treatment_admission"]["receipt_sha256"] == (
        receipt.receipt_sha256
    )


def test_valid_treatment_has_typed_receipt_and_enters_evaluator_once() -> None:
    problem, generator, traces, plan, outcome = asyncio.run(
        _run_treatment(option_id="shape.raise_x", claims="exact")
    )

    assert problem.evaluations == 2
    assert outcome.failure_stage is None
    assert outcome.candidate is not None
    receipt = outcome.treatment_admission_receipt
    assert receipt is not None and receipt.passed
    assert receipt.evaluator_entered is False
    preflight = outcome.prepared.treatment_preflight_receipt
    assert preflight is not None and preflight.passed
    assert tuple(action.option_id for action in preflight.compatible_actions) == (
        "shape.raise_x",
    )
    assert receipt.preflight_receipt_sha256 == preflight.receipt_sha256
    event_types = [event["event_type"] for event in traces]
    assert event_types.index("treatment_admission_completed") < event_types.index(
        "candidate_evaluated"
    )
    prompt = generator.requests[0].prompt
    assert "ASSIGNED INSIGHT TREATMENT CONTRACT" in prompt
    assert "claimed_insight_ids must equal this exact set" in prompt
    assert "shape.raise_x" in prompt and "control.raise_y" in prompt
    plan_record = _invocation_plan_record(plan)
    assert plan_record["insight_treatment_requirement"][
        "requirement_sha256"
    ] == plan.insight_treatment_requirement.requirement_sha256


def test_optional_treatment_allows_an_empty_honest_claim_subset() -> None:
    problem, _, _, _, outcome = asyncio.run(
        _run_treatment(
            option_id="shape.raise_x",
            claims="",
            claim_mode=TreatmentClaimMode.OPTIONAL_SUBSET,
        )
    )

    assert problem.evaluations == 2
    assert outcome.failure_stage is None
    receipt = outcome.treatment_admission_receipt
    assert receipt is not None and receipt.passed
    assert receipt.claimed_insight_ids == ()


def test_preflight_requires_one_joint_family_and_path_compatible_action() -> None:
    references = (
        InsightRef(InsightId("insight_joint_a"), 1),
        InsightRef(InsightId("insight_joint_b"), 1),
    )
    action = FiniteTreatmentAction(
        option_id="joint.change_x_y",
        option_identity_sha256="b" * 64,
        family="joint_edit",
        changed_paths=("$.x", "$.y"),
    )

    def evidence(families: tuple[str, str]) -> tuple[TreatmentInsightEvidence, ...]:
        return tuple(
            TreatmentInsightEvidence(
                reference=reference,
                insight_content_sha256=("d" if index == 0 else "e") * 64,
                applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
                affected_paths=(path,),
                recommended_option_families=(families[index],),
                recommended_option_ids=(action.option_id,),
            )
            for index, (reference, path) in enumerate(
                zip(references, ("$.x", "$.y"), strict=True)
            )
        )

    def preflight(
        insights: tuple[TreatmentInsightEvidence, ...],
    ):
        requirement = InsightTreatmentRequirement(
            insight_bindings=tuple(item.binding() for item in insights),
            finite_contract_sha256="c" * 64,
            allowed_actions=(action.binding(),),
            claim_mode=TreatmentClaimMode.EXACT_REQUIRED,
        )
        return StrictTreatmentCompliancePolicy().preflight(
            TreatmentPreflightRequest(
                requirement=requirement,
                operator_kind=OperatorKind.TYPED_MUTATION.value,
                editable_paths=("$.x", "$.y"),
                insights=insights,
                finite_contract_sha256="c" * 64,
                actions=(action,),
            )
        )

    disjoint_family = preflight(evidence(("shape_only", "control_only")))
    assert disjoint_family.violations == (
        TreatmentComplianceViolation.NO_COMPATIBLE_FINITE_ACTION,
    )

    shared_family = preflight(evidence(("joint_edit", "joint_edit")))
    assert shared_family.passed
    assert shared_family.compatible_actions == (action,)


def test_exact_claim_policy_rejects_duplicate_and_foreign_claims() -> None:
    _, _, _, plan, outcome = asyncio.run(
        _run_treatment(option_id="shape.raise_x", claims="exact")
    )
    requirement = plan.insight_treatment_requirement
    preflight = outcome.prepared.treatment_preflight_receipt
    assert requirement is not None and preflight is not None
    required_id = requirement.required_insights[0].insight_id.value
    receipt = StrictTreatmentCompliancePolicy().assess(
        TreatmentAdmissionRequest(
            requirement=requirement,
            preflight=preflight,
            claimed_insight_ids=(required_id, required_id, "insight_foreign"),
            selected_action=preflight.compatible_actions[0],
            operator_compliant=True,
        )
    )

    assert set(receipt.violations) == {
        TreatmentComplianceViolation.DUPLICATE_CLAIM,
        TreatmentComplianceViolation.FOREIGN_CLAIM,
        TreatmentComplianceViolation.EXACT_CLAIM_MISMATCH,
    }
    assert receipt.evaluator_entered is False

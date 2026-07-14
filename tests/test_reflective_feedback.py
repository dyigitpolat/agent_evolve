from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
)
from agent_evolve.application.budgeted_optimizer import (
    GenerationReceipt,
    OptimizerSlot,
    OptimizerState,
    SlotResult,
    generation_receipt_hash,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackContext,
    seal_generation_feedback,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightOrigin,
)
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.policies.feedback.held_out_asn import (
    G1ReflectionFeedbackInterceptor,
    HeldOutASNPlannerAdapter,
    HeldOutAssignmentUnavailable,
    HeldOutAssignmentUnavailableReason,
    ReflectedCardMailbox,
    ReflectiveFeedbackContractError,
    reflection_contrast_id,
    register_neutral_sham_card,
)
from agent_evolve.policies.memory.treatment_compliance import (
    TreatmentAssignmentRole,
    TreatmentClaimMode,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.patch import JsonPath, ObjectKey
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationResult,
    ReflectionInsightContract,
)
from agent_evolve.ports.generation_failure import GenerationFailureDisposition


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int
    y: int


class _NoEvaluationProblem:
    candidate_model = _Config
    objectives = (ObjectiveSpec("x", "min"), ObjectiveSpec("y", "min"))

    @staticmethod
    def search_space_description() -> str:
        return "Offline reflective-feedback contract; minimize x and y."

    @staticmethod
    def validate(configuration: object) -> bool:
        _Config.model_validate(configuration, strict=True)
        return True

    @staticmethod
    def evaluate(configuration):  # pragma: no cover - forbidden by the test.
        del configuration
        raise AssertionError("reflective-feedback contract test must not evaluate")


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/reflection",
        resolved_model="offline/reflection",
        resolved_provider="provider-free",
        provider_response_id="offline-reflection-1",
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


class _ExactCitationReflector:
    def __init__(
        self,
        *,
        duplicate_first_citation: bool = False,
        force_legacy_insights: bool = False,
    ) -> None:
        self.duplicate_first_citation = duplicate_first_citation
        self.force_legacy_insights = force_legacy_insights
        self.calls = 0
        self.requests = []
        self.option_id_by_contrast: dict[str, str] = {}

    async def propose(self, request):  # pragma: no cover - forbidden by the test.
        del request
        raise AssertionError("reflective-feedback contract test must not propose")

    async def reflect(self, request):
        self.calls += 1
        self.requests.append(request)
        first, second = request.available_contrast_ids
        if self.duplicate_first_citation:
            second = first
        def advanced(contrast_id: str) -> dict[str, object]:
            if request.insight_contract is None or self.force_legacy_insights:
                return {}
            record: dict[str, object] = {
                "effect_predictions": tuple(
                    MetricEffectPrediction(
                        metric_id=metric_id,
                        direction=(
                            MetricEffectDirection.DECREASE
                            if metric_id
                            == request.insight_contract.required_metric_ids[0]
                            else MetricEffectDirection.UNCHANGED
                        ),
                    )
                    for metric_id in request.insight_contract.required_metric_ids
                ),
                "recommended_option_families": (
                    request.insight_contract.allowed_option_families[0],
                ),
                "action_template": (
                    "Choose one bounded option from the recommended family."
                ),
                "falsification_condition": (
                    "Falsify if the predicted held-out metric direction fails."
                ),
            }
            if request.insight_contract.allowed_option_ids:
                record["recommended_option_ids"] = (
                    self.option_id_by_contrast[contrast_id],
                )
            return record
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim="The first exact intervention may transfer.",
                    trigger="The held-out parent admits the same bounded x edit.",
                    mechanism="Apply the first frozen x intervention template.",
                    affected_paths=("$.x",),
                    evidence_summary="Supported only by the cited first contrast.",
                    confidence=0.5,
                    evidence_contrast_ids=(first,),
                    **advanced(first),
                ),
                InsightDraft(
                    claim="The second exact intervention may transfer.",
                    trigger="The held-out parent admits the same bounded x edit.",
                    mechanism="Apply the second frozen x intervention template.",
                    affected_paths=("$.x",),
                    evidence_summary="Supported only by the cited second contrast.",
                    confidence=0.5,
                    evidence_contrast_ids=(second,),
                    **advanced(second),
                ),
            ),
            telemetry=_telemetry(),
        )


class _EmptyReflectionCapture:
    def __init__(self) -> None:
        self.requests = []

    async def propose(self, request):  # pragma: no cover - forbidden by the test.
        del request
        raise AssertionError("finite attribution test must not propose")

    async def reflect(self, request):
        self.requests.append(request)
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def _candidate(
    ids: DeterministicIdFactory,
    *,
    sequence: int,
    generation: int,
    x: int,
    y: int,
) -> EvolutionCandidate:
    configuration = freeze_json({"x": x, "y": y})
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=ids.new_candidate_id(),
            configuration_hash=typed_json_sha256(configuration),
            configuration_artifact_hash=hashlib.sha256(
                canonical_typed_json_bytes(configuration)
            ).hexdigest(),
            proposal_sequence=sequence,
        ),
        configuration=configuration,
        objectives=(("x", float(x)), ("y", float(y))),
        valid=True,
        generation=generation,
        label=f"candidate-{sequence}",
    )


def _sealed_g1_case(
    *,
    duplicate_first_citation: bool = False,
    force_legacy_insights: bool = False,
):
    ids = DeterministicIdFactory(
        "reflective_duplicate" if duplicate_first_citation else "reflective_valid"
    )
    memory = InsightMemoryBank(id_factory=ids)
    reflector = _ExactCitationReflector(
        duplicate_first_citation=duplicate_first_citation,
        force_legacy_insights=force_legacy_insights,
    )
    problem = _NoEvaluationProblem()
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=reflector,
        id_factory=ids,
        memory=memory,
        seed=7,
    )
    parent = _candidate(ids, sequence=0, generation=0, x=5, y=5)
    held_out_parent = _candidate(ids, sequence=1, generation=0, x=5, y=4)
    diagnostic_options = (
        FiniteVariationOption(
            option_id="bounded.decrease_x",
            parent_configuration_sha256=typed_json_sha256(parent.configuration),
            child_configuration=freeze_json({"x": 4, "y": 5}),
            family="bounded_coordinate",
            description="Decrease x by one unit.",
        ),
        FiniteVariationOption(
            option_id="bounded.increase_x",
            parent_configuration_sha256=typed_json_sha256(parent.configuration),
            child_configuration=freeze_json({"x": 6, "y": 5}),
            family="bounded_coordinate",
            description="Increase x by one unit.",
        ),
    )
    plans = (
        InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=1,
            label="diagnostic_efficiency",
            allowed_top_level=("x",),
            mutation_contract=MutationContract(
                editable_paths=(JsonPath((ObjectKey("x"),)),),
            ),
            mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
            finite_variation_contract=FiniteVariationContract(
                catalog_id="diagnostic_decrease_fixture",
                catalog_version=1,
                catalog_definition_sha256=_sha("diagnostic decrease fixture"),
                parent_configuration=parent.configuration,
                options=(diagnostic_options[0],),
            ),
        ),
        InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=1,
            label="diagnostic_constraint",
            allowed_top_level=("x",),
            mutation_contract=MutationContract(
                editable_paths=(JsonPath((ObjectKey("x"),)),),
            ),
            mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
            finite_variation_contract=FiniteVariationContract(
                catalog_id="diagnostic_increase_fixture",
                catalog_version=1,
                catalog_definition_sha256=_sha("diagnostic increase fixture"),
                parent_configuration=parent.configuration,
                options=(diagnostic_options[1],),
            ),
        ),
    )
    prepared, _ = engine.prepare_invocations(plans)
    children = (
        _candidate(ids, sequence=2, generation=1, x=4, y=5),
        _candidate(ids, sequence=3, generation=1, x=6, y=5),
    )
    outcomes = (
        InvocationOutcome(prepared[0], children[0], 1.0),
        InvocationOutcome(prepared[1], children[1], -1.0),
    )
    reflector.option_id_by_contrast = {
        reflection_contrast_id(outcome): option.option_id
        for outcome, option in zip(outcomes, diagnostic_options, strict=True)
    }
    slots = (
        OptimizerSlot.model(slot_id="D-S", role="diagnostic", plan=plans[0]),
        OptimizerSlot.model(slot_id="D-T", role="diagnostic", plan=plans[1]),
    )
    slot_results = tuple(
        SlotResult(slot, outcome, ())
        for slot, outcome in zip(slots, outcomes, strict=True)
    )
    provisional = GenerationReceipt(
        generation=1,
        plan_hash=_sha("reflective-g1-plan"),
        pre_archive_snapshot_hash=_sha("reflective-pre-archive"),
        post_archive_snapshot_hash=_sha("reflective-post-archive"),
        reward_definition_hash=prepared[0].variation_case.reward_definition_hash,
        reward_snapshot_hash=_sha("reflective-reward-snapshot"),
        logical_llm_calls_before=0,
        logical_llm_calls_after=2,
        unique_evaluations_before=0,
        unique_evaluations_after=2,
        reserved_logical_llm_calls=2,
        reserved_unique_evaluations=2,
        slot_results=slot_results,
        receipt_hash="0" * 64,
    )
    receipt = replace(provisional, receipt_hash=generation_receipt_hash(provisional))
    archive = ParetoArchive(problem.objectives).snapshot()
    state = OptimizerState(
        generation=1,
        candidates=(parent, held_out_parent, *children),
        archive=archive,
        archive_snapshot_hash=pareto_archive_snapshot_hash(archive),
        unique_evaluations=2,
        logical_llm_calls=2,
        generation_receipts=(receipt,),
    )
    plan = SimpleNamespace(generation=1)
    mailbox = ReflectedCardMailbox()
    interceptor = G1ReflectionFeedbackInterceptor(engine, mailbox)
    reservation = interceptor.reserve(state=state, plan=plan)
    context = GenerationFeedbackContext(state, plan, receipt, reservation)
    return (
        ids,
        memory,
        reflector,
        interceptor,
        mailbox,
        context,
        state,
        held_out_parent,
    )


def test_exact_two_card_reflection_binds_correct_swap_and_evidence_free_sham() -> None:
    (
        _,
        memory,
        reflector,
        interceptor,
        mailbox,
        context,
        state,
        held_out_parent,
    ) = _sealed_g1_case()
    insight_contract = ReflectionInsightContract(
        required_metric_ids=("objective:x", "objective:y"),
        allowed_option_families=("bounded_coordinate",),
        allowed_option_ids=("bounded.decrease_x", "bounded.increase_x"),
    )
    interceptor = G1ReflectionFeedbackInterceptor(
        interceptor.engine,
        mailbox,
        required_metric_ids=insight_contract.required_metric_ids,
        allowed_option_families=insight_contract.allowed_option_families,
        allowed_option_ids=insight_contract.allowed_option_ids,
    )

    result = asyncio.run(interceptor.after_generation(context))
    feedback_receipt = seal_generation_feedback(context=context, result=result)
    planner_state = replace(
        state,
        logical_llm_calls=3,
        feedback_receipts=(feedback_receipt,),
    )
    sham_reference = register_neutral_sham_card(
        memory=memory,
        affected_paths=("$.x",),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        insight_contract=ReflectionInsightContract(
            required_metric_ids=insight_contract.required_metric_ids,
            allowed_option_families=insight_contract.allowed_option_families,
            allowed_option_ids=("bounded.decrease_x",),
        ),
    )
    adapter = HeldOutASNPlannerAdapter(mailbox, memory, sham_reference)
    finite_child = freeze_json({"x": 4, "y": 4})
    finite_options = (
        FiniteVariationOption(
            option_id="bounded.decrease_x",
            parent_configuration_sha256=typed_json_sha256(
                held_out_parent.configuration
            ),
            child_configuration=finite_child,
            family="bounded_coordinate",
            description="Decrease the bounded x coordinate by one unit.",
        ),
        FiniteVariationOption(
            option_id="bounded.increase_x",
            parent_configuration_sha256=typed_json_sha256(
                held_out_parent.configuration
            ),
            child_configuration=freeze_json({"x": 6, "y": 4}),
            family="bounded_coordinate",
            description="Increase the bounded x coordinate by one unit.",
        ),
    )
    finite_contract = FiniteVariationContract(
        catalog_id="held_out_bounded_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("held-out bounded fixture definition"),
        parent_configuration=held_out_parent.configuration,
        options=finite_options,
    )
    bases = tuple(
        InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (held_out_parent,),
            generation=2,
            label=label,
            allowed_top_level=("x",),
            phase="held_out_transfer",
            mutation_contract=MutationContract(
                editable_paths=(JsonPath((ObjectKey("x"),)),),
            ),
            mutation_response_mode=(
                MutationResponseMode.FINITE_OPTION_SELECTION_V1
            ),
            finite_variation_contract=finite_contract,
        )
        for label in ("A-adaptive", "S-score-swapped", "N-sham")
    )

    bound = adapter.bind_plans(
        planner_state,
        adaptive_base=bases[0],
        score_swapped_base=bases[1],
        sham_base=bases[2],
    )

    assert reflector.calls == 1
    assert result.logical_llm_calls_used == 1
    assert 2 + result.logical_llm_calls_used + 3 == 6
    assert bound.assignments.adaptive.origin_transfer_score == 1
    assert bound.assignments.score_swapped.origin_transfer_score == -1
    assert (
        bound.assignments.adaptive.reference
        != bound.assignments.score_swapped.reference
    )
    assert bound.assignments.score_swapped.assigned_selection_score == 1
    commitment = bound.assignment_commitment
    commitment_record = commitment.to_record()
    assert commitment_record["assignment_sha256"] == commitment.assignment_sha256
    assert commitment_record["selector_policy_id"] == (
        "held_out_asn_origin_score_swap"
    )
    assert commitment_record["selector_policy_version"] == 1
    assert commitment_record["common_score_multiset"] == [-1, 1]
    assert len(commitment_record["true_score_map"]) == 2
    assert len(commitment_record["score_swapped_map"]) == 2
    assert {
        item["assigned_selection_score"]
        for item in commitment_record["true_score_map"]
    } == {-1, 1}
    assert {
        item["assigned_selection_score"]
        for item in commitment_record["score_swapped_map"]
    } == {-1, 1}
    assert commitment_record["chosen_references"]["sham"] == {
        "insight_id": sham_reference.insight_id.value,
        "insight_version": sham_reference.version,
    }
    assert bound.adaptive.quarantine_test_insights == (
        bound.assignments.adaptive.reference,
    )
    assert bound.score_swapped.quarantine_test_insights == (
        bound.assignments.score_swapped.reference,
    )
    assert bound.sham.quarantine_test_insights == (sham_reference,)
    requirements = tuple(
        plan.insight_treatment_requirement
        for plan in (bound.adaptive, bound.score_swapped, bound.sham)
    )
    assert all(requirement is not None for requirement in requirements)
    assert tuple(
        requirement.claim_mode for requirement in requirements if requirement
    ) == (
        TreatmentClaimMode.EXACT_REQUIRED,
        TreatmentClaimMode.EXACT_REQUIRED,
        TreatmentClaimMode.EXACT_REQUIRED,
    )
    assert tuple(
        requirement.assignment_role for requirement in requirements if requirement
    ) == (
        TreatmentAssignmentRole.ACTIVE,
        TreatmentAssignmentRole.ACTIVE,
        TreatmentAssignmentRole.SHAM_CONTROL,
    )
    assert not any(
        plan.use_memory for plan in (bound.adaptive, bound.score_swapped, bound.sham)
    )
    sham_entry = next(
        entry for entry in memory.entries if entry.reference == sham_reference
    )
    assert sham_entry.origin is InsightOrigin.MANUAL
    assert sham_entry.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert sham_entry.evidence_lineage is None
    assert sham_entry.draft.evidence_contrast_ids == ()


def test_reflected_exact_option_must_equal_its_cited_executed_action() -> None:
    _, _, reflector, legacy, mailbox, context, _, _ = _sealed_g1_case()
    exact_ids = ("bounded.decrease_x", "bounded.increase_x")
    reflector.option_id_by_contrast = {
        contrast_id: exact_ids[index]
        for index, contrast_id in enumerate(
            reversed(tuple(reflector.option_id_by_contrast))
        )
    }
    interceptor = G1ReflectionFeedbackInterceptor(
        legacy.engine,
        mailbox,
        required_metric_ids=("objective:x", "objective:y"),
        allowed_option_families=("bounded_coordinate",),
        allowed_option_ids=exact_ids,
    )

    result = asyncio.run(interceptor.after_generation(context))

    assert dict(result.metadata)["status"] == "reflection_rejected"
    assert not mailbox._batches


class _ReflectionBoundaryFailure(RuntimeError):
    def __init__(self, disposition: GenerationFailureDisposition) -> None:
        self._disposition = disposition
        super().__init__(disposition.value)

    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition:
        return self._disposition


class _FailingReflectionEngine:
    def __init__(self, delegate, failure: BaseException) -> None:
        self.delegate = delegate
        self.failure = failure

    def identify_phenotype(self, configuration):
        return self.delegate.identify_phenotype(configuration)

    async def reflect(self, *args, **kwargs):
        del args, kwargs
        raise self.failure


def test_only_typed_model_or_schema_reflection_failure_is_clean_no_card() -> None:
    _, _, _, interceptor, mailbox, context, state, _ = _sealed_g1_case()
    typed = _ReflectionBoundaryFailure(
        GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
    )
    interceptor.engine = _FailingReflectionEngine(interceptor.engine, typed)

    result = asyncio.run(interceptor.after_generation(context))
    assert result.logical_llm_calls_used == 1
    assert dict(result.metadata) == {
        "card_count": "0",
        "reason": "model_or_schema_failure",
        "schema": "v7-reflected-card-batch-v2",
        "source_generation": "1",
        "status": "reflection_failed",
    }
    receipt = seal_generation_feedback(context=context, result=result)
    planner_state = replace(
        state,
        logical_llm_calls=3,
        feedback_receipts=(receipt,),
    )
    with pytest.raises(HeldOutAssignmentUnavailable) as captured:
        mailbox.read_verified(state=planner_state)
    assert captured.value.reason is (
        HeldOutAssignmentUnavailableReason.REFLECTED_CARD_BATCH_UNAVAILABLE
    )


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("credential/source/queue/programming failure"),
        _ReflectionBoundaryFailure(
            GenerationFailureDisposition.INFRASTRUCTURE_FAILURE
        ),
    ],
)
def test_untyped_or_infrastructure_reflection_failure_propagates(
    failure: BaseException,
) -> None:
    _, _, _, interceptor, _, context, _, _ = _sealed_g1_case()
    interceptor.engine = _FailingReflectionEngine(interceptor.engine, failure)

    with pytest.raises(type(failure), match=str(failure)):
        asyncio.run(interceptor.after_generation(context))


def test_structural_inapplicability_is_the_only_assignment_error_translated() -> None:
    (
        _,
        memory,
        _,
        interceptor,
        _,
        context,
        state,
        held_out_parent,
    ) = _sealed_g1_case()
    result = asyncio.run(interceptor.after_generation(context))
    feedback_receipt = seal_generation_feedback(context=context, result=result)
    planner_state = replace(
        state,
        logical_llm_calls=3,
        feedback_receipts=(feedback_receipt,),
    )
    sham_reference = register_neutral_sham_card(
        memory=memory,
        affected_paths=("$.x",),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
    )
    adapter = HeldOutASNPlannerAdapter(
        interceptor.mailbox,
        memory,
        sham_reference,
    )

    def bases(path: str) -> tuple[InvocationPlan, ...]:
        return tuple(
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (held_out_parent,),
                generation=2,
                label=label,
                allowed_top_level=(path,),
                phase="held_out_transfer",
            )
            for label in ("A-adaptive", "S-score-swapped", "N-sham")
        )

    structurally_inapplicable = bases("y")
    with pytest.raises(HeldOutAssignmentUnavailable) as captured:
        adapter.bind_plans(
            planner_state,
            adaptive_base=structurally_inapplicable[0],
            score_swapped_base=structurally_inapplicable[1],
            sham_base=structurally_inapplicable[2],
        )
    assert captured.value.reason is (
        HeldOutAssignmentUnavailableReason.STRUCTURALLY_INAPPLICABLE_ASSIGNMENT
    )

    adaptive_reference = adapter.resolve(planner_state).adaptive.reference
    memory.promote(
        adaptive_reference,
        reason="deliberate lifecycle-drift fixture",
        supporting_evidence=("experiment:fixture",),
    )
    applicable = bases("x")
    with pytest.raises(ValueError, match="only quarantined") as fatal:
        adapter.bind_plans(
            planner_state,
            adaptive_base=applicable[0],
            score_swapped_base=applicable[1],
            sham_base=applicable[2],
        )
    assert not isinstance(fatal.value, HeldOutAssignmentUnavailable)


def test_duplicate_origin_citation_is_consumed_but_not_published() -> None:
    _, _, reflector, interceptor, mailbox, context, state, _ = _sealed_g1_case(
        duplicate_first_citation=True
    )

    result = asyncio.run(interceptor.after_generation(context))
    feedback_receipt = seal_generation_feedback(context=context, result=result)
    planner_state = replace(
        state,
        logical_llm_calls=3,
        feedback_receipts=(feedback_receipt,),
    )

    assert reflector.calls == 1
    assert result.logical_llm_calls_used == 1
    assert dict(result.metadata)["status"] == "reflection_rejected"
    with pytest.raises(HeldOutAssignmentUnavailable) as captured:
        mailbox.read_verified(state=planner_state)
    assert captured.value.reason is (
        HeldOutAssignmentUnavailableReason.REFLECTED_CARD_BATCH_UNAVAILABLE
    )


def test_missing_batch_without_authenticated_feedback_receipt_is_fatal_drift() -> None:
    _, _, _, _, mailbox, _, state, _ = _sealed_g1_case()

    with pytest.raises(
        ReflectiveFeedbackContractError,
        match="one exact source feedback receipt",
    ):
        mailbox.read_verified(state=state)


def test_g1_advanced_cards_and_sham_retain_prompt_visible_typed_fields() -> None:
    (
        _,
        memory,
        reflector,
        legacy_interceptor,
        _,
        context,
        _,
        _,
    ) = _sealed_g1_case()
    contract = ReflectionInsightContract(
        required_metric_ids=("objective:cost", "violation:capacity"),
        allowed_option_families=("control_only", "joint_edit"),
    )
    mailbox = ReflectedCardMailbox()
    interceptor = G1ReflectionFeedbackInterceptor(
        legacy_interceptor.engine,
        mailbox,
        required_metric_ids=contract.required_metric_ids,
        allowed_option_families=contract.allowed_option_families,
    )

    result = asyncio.run(interceptor.after_generation(context))
    assert dict(result.metadata)["status"] == "ready"
    assert reflector.requests[0].insight_contract == contract
    assert reflector.requests[0].min_insights == 2
    assert reflector.requests[0].max_insights == 2
    batch = mailbox._batches[1]
    adaptive_records = memory.prompt_records(
        tuple(card.reference for card in batch.cards)
    )
    for record in adaptive_records:
        assert record["effect_predictions"] == [
            {"metric_id": "objective:cost", "direction": "decrease"},
            {"metric_id": "violation:capacity", "direction": "unchanged"},
        ]
        assert record["recommended_option_families"] == ["control_only"]
        assert record["action_template"]
        assert record["falsification_condition"]

    sham_reference = register_neutral_sham_card(
        memory=memory,
        affected_paths=("$.x",),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        insight_contract=contract,
    )
    sham_record = memory.prompt_records((sham_reference,))[0]
    assert sham_record["effect_predictions"] == [
        {"metric_id": "objective:cost", "direction": "unknown"},
        {"metric_id": "violation:capacity", "direction": "unknown"},
    ]
    assert sham_record["recommended_option_families"] == [
        "control_only",
        "joint_edit",
    ]
    assert sham_record["action_template"]
    assert sham_record["falsification_condition"]


def test_g1_advanced_contract_rejects_legacy_free_form_cards() -> None:
    (
        _,
        _,
        reflector,
        legacy_interceptor,
        _,
        context,
        _,
        _,
    ) = _sealed_g1_case(force_legacy_insights=True)
    mailbox = ReflectedCardMailbox()
    interceptor = G1ReflectionFeedbackInterceptor(
        legacy_interceptor.engine,
        mailbox,
        required_metric_ids=("objective:cost", "violation:capacity"),
        allowed_option_families=("control_only", "joint_edit"),
    )

    result = asyncio.run(interceptor.after_generation(context))
    assert reflector.calls == 1
    assert result.logical_llm_calls_used == 1
    assert dict(result.metadata)["status"] == "reflection_rejected"
    assert not mailbox._batches


def test_reflection_contrast_attributes_exact_finite_option_and_family() -> None:
    ids = DeterministicIdFactory("finite_reflection_attribution")
    memory = InsightMemoryBank(id_factory=ids)
    generator = _EmptyReflectionCapture()
    engine = AgenticEvolutionEngine(
        problem=_NoEvaluationProblem(),
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=7,
    )
    parent = _candidate(ids, sequence=0, generation=0, x=5, y=5)
    child = _candidate(ids, sequence=1, generation=1, x=4, y=5)
    option = FiniteVariationOption(
        option_id="bounded.decrease_x",
        parent_configuration_sha256=typed_json_sha256(parent.configuration),
        child_configuration=child.configuration,
        family="bounded_coordinate",
        description="Decrease the first bounded coordinate by one unit.",
    )
    contract = FiniteVariationContract(
        catalog_id="bounded_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("bounded fixture definition"),
        parent_configuration=parent.configuration,
        options=(option,),
    )
    plan = InvocationPlan(
        OperatorKind.TYPED_MUTATION,
        (parent,),
        generation=1,
        label="finite_diagnostic",
        allowed_top_level=("x",),
        mutation_contract=MutationContract(
            editable_paths=(JsonPath((ObjectKey("x"),)),),
        ),
        mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
        finite_variation_contract=contract,
    )
    prepared, _ = engine.prepare_invocations((plan,))
    outcome = InvocationOutcome(prepared[0], child, 1.0)

    asyncio.run(engine.reflect((outcome,), label="finite_attribution"))

    prompt = generator.requests[0].prompt
    assert '"option_id":"bounded.decrease_x"' in prompt
    assert '"family":"bounded_coordinate"' in prompt
    assert option.identity_sha256 in prompt
    assert contract.identity_sha256 in prompt


def test_inactive_intervention_contract_preserves_legacy_reflection_prompt_bytes() -> (
    None
):
    ids = DeterministicIdFactory("legacy_bytes")
    generator = _EmptyReflectionCapture()
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=_NoEvaluationProblem(),
        generator=generator,
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=7,
        trace_sink=traces.append,
    )

    asyncio.run(engine.reflect((), label="legacy_bytes"))

    expected = "\n".join(
        [
            "Extract a small set of falsifiable optimization insights from the evaluated evidence below.",
            "Each insight must state a conditional trigger, a mechanism, affected JSON paths, and the exact evidence. "
            "Do not restate a candidate, invent unobserved causality, or give generic advice. Counterexamples should lower confidence. "
            "The scalar reward is not a Pareto-dominance claim; use the explicit dominance and validity fields when describing quality.",
            "Use the machine_derived_contrasts as the evidence boundary. A single_operation contrast may support a one-operation effect hypothesis. "
            "A no_change contrast is an abstention/control, not evidence that an unexecuted edit caused an outcome. A joint_intervention contrast supports only the joint association: do not assign its outcome to one coordinate or invent a benefit for a coordinate without an ablation. "
            "Put every supporting full 64-character contrast_id in evidence_contrast_ids; use evidence_summary only for a human-readable account of that evidence. Every affected path must be canonical and begin with $., and duplicated claims should be consolidated instead of re-added. "
            "For a multi-operation association, state a concrete one-coordinate falsification/ablation in the trigger or evidence summary.",
            "",
            "PROBLEM",
            "Offline reflective-feedback contract; minimize x and y.",
            "",
            "EVALUATED TRACE",
            "[]",
            "",
            "Return at most 4 insights.",
        ]
    )
    assert generator.requests[0].prompt.encode("utf-8") == expected.encode("utf-8")
    requested = next(
        event for event in traces if event["event_type"] == "reflection_requested"
    )
    completed = next(
        event for event in traces if event["event_type"] == "reflection_completed"
    )
    assert "insight_contract" not in requested
    assert "insight_contract" not in completed

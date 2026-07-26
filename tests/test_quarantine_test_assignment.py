"""Audit contracts for test-only assignment of quarantined insights."""

from __future__ import annotations

import asyncio
import copy
from decimal import Decimal

import pytest

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InsightAssignmentKind,
    InvocationPlan,
    MutationContract,
    OperatorKind,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightOrigin,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.patch import JsonPath, ObjectKey
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


_PARENT = {"runtime": {"threads": 4, "prefetch": 4}}
_CHILD = {"runtime": {"threads": 8, "prefetch": 4}}


def _draft(claim: str, path: str = "$.runtime.threads") -> InsightDraft:
    return InsightDraft(
        claim=claim,
        trigger="the named coordinate is editable",
        mechanism="the coordinate can change the measured score",
        affected_paths=(path,),
        evidence_summary="a discovery contrast awaiting an isolated test",
        confidence=0.6,
    )


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="response",
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
    objectives = (ObjectiveSpec("score", "max"),)
    candidate_model = dict

    @staticmethod
    def search_space_description() -> str:
        return "Choose a runtime thread count."

    @staticmethod
    def validate(configuration) -> bool:
        return type(configuration) is dict

    @staticmethod
    def evaluate(configuration) -> dict[str, float]:
        return {"score": float(configuration["runtime"]["threads"])}


class _CapturingGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def propose(self, request):
        self.prompts.append(request.prompt)
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=copy.deepcopy(_CHILD),
                design_rationale="Tested the assigned hypothesis atomically.",
                source_attribution=(
                    SourceAttribution("$.runtime.threads", "mutation"),
                ),
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        self.prompts.append(request.prompt)
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def _path() -> JsonPath:
    return JsonPath((ObjectKey("runtime"), ObjectKey("threads")))


def test_public_validation_is_quarantine_only_structural_and_detached() -> None:
    ids = DeterministicIdFactory("quarantine_validation")
    memory = InsightMemoryBank(id_factory=ids)
    assigned, _ = memory.add(
        _draft("Eight threads may improve throughput."),
        applicable_operator_kinds=("typed_mutation",),
        origin=InsightOrigin.MANUAL,
    )

    validated = memory.validate_quarantine_test_assignment(
        (assigned.reference,),
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.threads",),
    )
    records = memory.prompt_records(validated)

    assert validated == (assigned.reference,)
    assert assigned.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert memory.eligible_references(
        operator_kind="typed_mutation",
        editable_paths=("$.runtime.threads",),
    ) == ()
    assert records[0]["lifecycle_state"] == "quarantined"
    assert records[0]["evidence_summary"] == (
        "a discovery contrast awaiting an isolated test"
    )
    records[0]["claim"] = "caller mutation"
    records[0]["affected_paths"].append("$.runtime.prefetch")
    fresh = memory.prompt_records(validated)
    assert fresh[0]["claim"] == "Eight threads may improve throughput."
    assert fresh[0]["affected_paths"] == ["$.runtime.threads"]


@pytest.mark.parametrize(
    ("bad_kind", "expected"),
    [
        ("seed", "only quarantined"),
        ("promoted", "only quarantined"),
        ("deprecated", "only quarantined"),
        ("foreign", "foreign"),
        ("duplicate", "duplicates"),
        ("operator", "structurally inapplicable"),
        ("path", "structurally inapplicable"),
    ],
)
def test_quarantine_assignment_rejects_wrong_state_identity_or_scope(
    bad_kind: str,
    expected: str,
) -> None:
    ids = DeterministicIdFactory(f"quarantine_reject_{bad_kind}")
    memory = InsightMemoryBank(id_factory=ids)
    seed, _ = memory.add(_draft("Seed prior."))
    quarantined, _ = memory.add(
        _draft("Quarantined hypothesis."),
        applicable_operator_kinds=("typed_mutation",),
        origin=InsightOrigin.MANUAL,
    )
    reference = quarantined.reference
    references = (reference,)
    operator = "typed_mutation"
    paths = ("$.runtime.threads",)

    if bad_kind == "seed":
        references = (seed.reference,)
    elif bad_kind == "promoted":
        memory.promote(
            reference,
            reason="held-out test passed",
            supporting_evidence=("experiment:test-1",),
        )
    elif bad_kind == "deprecated":
        memory.deprecate(reference, reason="held-out test failed")
    elif bad_kind == "foreign":
        other = InsightMemoryBank(
            id_factory=DeterministicIdFactory("quarantine_foreign")
        )
        foreign, _ = other.add(
            _draft("Foreign quarantine."),
            origin=InsightOrigin.MANUAL,
        )
        references = (foreign.reference,)
    elif bad_kind == "duplicate":
        references = (reference, reference)
    elif bad_kind == "operator":
        operator = "repair"
    elif bad_kind == "path":
        paths = ("$.runtime.prefetch",)

    with pytest.raises(ValueError, match=expected):
        memory.validate_quarantine_test_assignment(
            references,
            operator_kind=operator,
            editable_paths=paths,
        )


def test_quarantine_lane_is_prompted_and_traced_without_retrieval_credit() -> None:
    async def scenario():
        ids = DeterministicIdFactory("quarantine_lane")
        memory = InsightMemoryBank(id_factory=ids)
        entry, _ = memory.add(
            _draft("Eight threads may improve throughput."),
            applicable_operator_kinds=("typed_mutation",),
            origin=InsightOrigin.MANUAL,
        )
        traces: list[dict[str, object]] = []
        generator = _CapturingGenerator()
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=generator,
            id_factory=ids,
            memory=memory,
            seed=2,
            trace_sink=traces.append,
        )
        parent = await engine.register_seed(_PARENT, label="parent")
        plan = InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(parent,),
            generation=1,
            label="quarantine_test",
            allowed_top_level=("runtime",),
            mutation_contract=MutationContract((_path(),)),
            quarantine_test_insights=(entry.reference,),
        )
        outcome, = await engine.run_invocations((plan,))
        await engine.reflect((outcome,), label="test_result", max_insights=1)
        return memory, entry, outcome, traces, generator

    memory, entry, outcome, traces, generator = asyncio.run(scenario())

    assert outcome.prepared.selection_decision is None
    assert outcome.prepared.insight_assignment_kind is (
        InsightAssignmentKind.QUARANTINE_TEST
    )
    assert outcome.prepared.variation_case.selected_insights == (entry.reference,)
    assert "QUARANTINED TEST HYPOTHESES" in outcome.prepared.prompt
    assert '"assignment_kind":"quarantine_test"' in outcome.prepared.prompt
    assert outcome.candidate is not None
    assert outcome.candidate.selected_insight_refs == (entry.reference,)
    assert outcome.candidate.insight_assignment_kind is (
        InsightAssignmentKind.QUARANTINE_TEST
    )
    assert entry.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert memory.entries[0].lifecycle_state is InsightLifecycleState.QUARANTINED
    assert memory.trials == ()

    prepared = next(
        event for event in traces if event["event_type"] == "invocation_prepared"
    )
    evaluated = next(
        event for event in traces if event["event_type"] == "candidate_evaluated"
    )
    completed = next(
        event for event in traces if event["event_type"] == "invocation_completed"
    )
    exact_ref = {
        "insight_id": entry.reference.insight_id.value,
        "version": entry.reference.version,
    }
    for event in (prepared, evaluated, completed):
        assert event["assignment_kind"] == "quarantine_test"
        assert event["selected_insights"] == [exact_ref]
    assert prepared["selection_decision"] is None
    assert prepared["selected_insight_records"][0]["assignment_kind"] == (
        "quarantine_test"
    )
    assert evaluated["selected_insight_records"][0]["lifecycle_state"] == (
        "quarantined"
    )
    assert completed["insight_credit_status"] == (
        "test_only_no_retrieval_credit"
    )
    assert not any(
        event["event_type"].startswith("insight_credit_") for event in traces
    )
    reflection_prompt = generator.prompts[-1]
    assert '"assignment_kind":"quarantine_test"' in reflection_prompt
    assert f'"version":{entry.reference.version}' in reflection_prompt


def test_invocation_plan_forbids_mixed_or_duplicate_assignment() -> None:
    async def parent_and_reference():
        ids = DeterministicIdFactory("quarantine_plan")
        memory = InsightMemoryBank(id_factory=ids)
        entry, _ = memory.add(
            _draft("Test-only hypothesis."),
            origin=InsightOrigin.MANUAL,
        )
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=_CapturingGenerator(),
            id_factory=ids,
            memory=memory,
            seed=1,
        )
        return await engine.register_seed(_PARENT, label="parent"), entry.reference

    parent, reference = asyncio.run(parent_and_reference())
    common = dict(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=1,
        label="invalid",
        allowed_top_level=("runtime",),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        InvocationPlan(
            **common,
            use_memory=True,
            quarantine_test_insights=(reference,),
        )
    with pytest.raises(ValueError, match="duplicates"):
        InvocationPlan(
            **common,
            quarantine_test_insights=(reference, reference),
        )

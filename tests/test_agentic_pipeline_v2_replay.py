"""Offline replay of sanitized candidate drafts from the pipeline v2 probe."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    OperatorKind,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ConflictResolutionDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "agentic_pipeline_v2_replay.json"
)
_OBLIGATION_ID = re.compile(r'"obligation_id":"([0-9a-f]{64})"')


def _load_pipeline_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "pipeline_codesign"
        / "problem_def.py"
    )
    name = "_agent_evolve_test_pipeline_v2_replay"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/canned-replay",
        resolved_model="offline/canned-replay",
        resolved_provider="offline",
        provider_response_id=None,
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


def _load_fixture() -> dict[str, object]:
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    assert value["schema_version"] == 1
    return value


class _CannedDraftGenerator:
    """Serve fixture drafts through the public agentic-generator boundary."""

    def __init__(self, fixture: dict[str, object]) -> None:
        configurations = fixture["configurations"]
        candidates = fixture["candidates"]
        assert type(configurations) is dict and type(candidates) is list
        self._configurations = configurations
        self._records = [
            record
            for record in candidates
            if record["operator"] != OperatorKind.REPRODUCTION.value
        ]
        self._next = 0

    @property
    def exhausted(self) -> bool:
        return self._next == len(self._records)

    async def propose(self, request):
        record = self._records[self._next]
        self._next += 1
        assert request.operation == record["operator"]

        configuration_name = record["configuration"]
        configuration = copy.deepcopy(self._configurations[configuration_name])
        source_attribution = tuple(
            SourceAttribution(path=item["path"], source=item["source"])
            for item in record.get("source_attribution", ())
        )
        conflict_resolutions = tuple(
            ConflictResolutionDraft(
                relation_id=item["relation_id"],
                choice=item["choice"],
                explanation=item["explanation"],
            )
            for item in record.get("conflict_resolutions", ())
        )
        obligation_ids = (
            tuple(sorted(set(_OBLIGATION_ID.findall(request.prompt))))
            if record.get("claim_prepared_obligations", False)
            else ()
        )
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale=f"Sanitized canned replay for {record['label']}.",
                intended_changes=("Replay the observed candidate configuration.",),
                source_attribution=source_attribution,
                claimed_preservation_obligation_ids=obligation_ids,
                conflict_resolutions=conflict_resolutions,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def _plan_for_record(record, *, base, left, right, generation_two_parent):
    operator = OperatorKind(record["operator"])
    generation = record["generation"]
    if operator in {
        OperatorKind.TWO_PARENT_CROSSOVER,
        OperatorKind.THREE_WAY_RECOMBINATION,
    }:
        parents = (left, right)
    elif generation == 1:
        parents = (left,)
    else:
        assert generation_two_parent is not None
        parents = (generation_two_parent,)
    return InvocationPlan(
        operator,
        parents,
        generation=generation,
        label=record["label"],
        common_ancestor=(
            base
            if operator is OperatorKind.THREE_WAY_RECOMBINATION
            else None
        ),
        allowed_top_level=(
            ("runtime",)
            if operator is OperatorKind.TYPED_MUTATION
            else ()
        ),
    )


def test_fixture_is_sanitized_and_repository_local() -> None:
    text = _FIXTURE_PATH.read_text(encoding="utf-8")
    assert "development-only" in text
    for forbidden in (
        "provider_response_id",
        "OPENROUTER_API_KEY",
        "requested_model",
        "cost_usd",
        "research_artifacts",
    ):
        assert forbidden not in text


def test_v2_candidate_drafts_replay_with_design_and_evidence_separated() -> None:
    fixture = _load_fixture()
    pipeline = _load_pipeline_module()

    async def scenario():
        ids = DeterministicIdFactory("pipeline_v2_replay")
        generator = _CannedDraftGenerator(fixture)
        traces = []
        engine = AgenticEvolutionEngine(
            problem=pipeline.PipelineCoDesignProblem(),
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=20260713,
            trace_sink=traces.append,
        )
        base, left, right = await asyncio.gather(
            engine.register_seed(pipeline.BASE_CONFIG, label="base"),
            engine.register_seed(
                pipeline.DEVELOPMENT_BRANCH_LEFT, label="left_branch"
            ),
            engine.register_seed(
                pipeline.DEVELOPMENT_BRANCH_RIGHT, label="right_branch"
            ),
        )
        outcomes = []
        generation_two_parent = None
        for record in fixture["candidates"]:
            plan = _plan_for_record(
                record,
                base=base,
                left=left,
                right=right,
                generation_two_parent=generation_two_parent,
            )
            outcome = (await engine.run_invocations((plan,)))[0]
            outcomes.append(outcome)
            if record["label"] == "g1_recombine_no_memory":
                assert outcome.candidate is not None
                generation_two_parent = outcome.candidate
        return generator, traces, outcomes

    generator, traces, outcomes = asyncio.run(scenario())
    records = fixture["candidates"]
    configurations = fixture["configurations"]
    assert generator.exhausted
    assert len(outcomes) == len(records) == 10

    by_label = {}
    for record, outcome in zip(records, outcomes, strict=True):
        expected = record["expected"]
        candidate = outcome.candidate
        by_label[record["label"]] = outcome
        if record["label"] == "g1_crossover_no_memory":
            assert candidate is None
            assert outcome.failure_stage == "candidate"
            assert outcome.call_failure_type == "ValueError"
            assert outcome.reward == -1.0
            continue
        assert candidate is not None
        assert candidate.valid is expected["valid"]
        assert candidate.operator_compliant is expected["operator_compliant"]
        assert candidate.operator_failure == expected["operator_failure"]
        assert candidate.evidence_compliant is expected["evidence_compliant"]
        assert candidate.evidence_failure == expected["evidence_failure"]
        assert candidate.objective_map == expected["objectives"]
        assert outcome.reward == pytest.approx(expected["reward"])
        assert outcome.dominates_any_parent is expected["dominates_any_parent"]
        target_match = (
            candidate.configuration_dict
            == configurations["known_recombination_target"]
        )
        assert target_match is expected["known_target_match"]

    structurally_compliant = [
        outcome
        for outcome in outcomes
        if outcome.candidate is not None
        and outcome.candidate.operator_compliant
    ]
    assert len(structurally_compliant) == 8
    structural_failures = [
        outcome.prepared.plan.label
        for outcome in outcomes
        if outcome.candidate is not None
        and not outcome.candidate.operator_compliant
    ]
    assert structural_failures == ["g1_recombine_random_memory"]

    evidence_compliant = [
        outcome
        for outcome in outcomes
        if outcome.candidate is not None
        and outcome.candidate.evidence_compliant
    ]
    assert len(evidence_compliant) == 3

    # Mutation annotations remain soft evidence: malformed explanations stay
    # visible without overwriting design quality. Crossover source attribution
    # is instead an operator admission obligation and failed before evaluation.
    strong_annotation_failures = {
        "g1_scoped_mutation_random_memory",
        "g2_mutation_random_memory_1",
        "g2_mutation_random_memory_2",
        "g2_mutation_random_memory_3",
    }
    for label in strong_annotation_failures:
        outcome = by_label[label]
        assert outcome.candidate is not None
        assert outcome.candidate.operator_compliant
        assert not outcome.candidate.evidence_compliant
        assert outcome.reward > 0
        assert outcome.dominates_any_parent

    memory_recombination = by_label["g1_recombine_random_memory"]
    assert memory_recombination.candidate is not None
    assert memory_recombination.candidate.objective_map["speedup"] == 3.42
    assert not memory_recombination.candidate.operator_compliant
    assert memory_recombination.reward == -1.0

    completed = [
        event for event in traces if event["event_type"] == "invocation_completed"
    ]
    assert len(completed) == 10
    assert sum(event["operator_compliant"] is True for event in completed) == 8
    assert sum(event["evidence_compliant"] is True for event in completed) == 3
    assert sum(event["failure_stage"] == "candidate" for event in completed) == 1

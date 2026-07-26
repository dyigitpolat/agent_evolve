"""Offline end-to-end gate for the real-evaluator BOiLS pilot workflow."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import threading
from decimal import Decimal
from pathlib import Path

from agent_evolve import ObjectiveSpec
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)
from examples.benchmarks.boils_abc.actions import ACTION_IDS, CandidateConfig
from examples.development import run_boils_agentic_pilot as pilot


_EDITABLE_PATH = re.compile(r'"editable_paths":\["(\$\.sequence\[(\d+)\])"\]')
_INSIGHT_ID = re.compile(r'"insight_id":"([^"]+)"')
_CONTRAST_ID = re.compile(r'"contrast_id":"([0-9a-f]{64})"')


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/boils-pilot",
        resolved_model="offline/boils-pilot",
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


def _parents(prompt: str) -> list[dict[str, object]]:
    payload = prompt.split("\nPARENTS\n", 1)[1]
    for delimiter in (
        "\nQUARANTINED TEST HYPOTHESES\n",
        "\nSELECTED MEMORY HYPOTHESES\n",
    ):
        if delimiter in payload:
            payload = payload.split(delimiter, 1)[0]
            break
    return json.loads(payload)


class _OfflineBoilsProblem:
    candidate_model = CandidateConfig

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.evaluation_calls = 0

    @property
    def objectives(self):
        return [
            ObjectiveSpec("total_lut_count", "min"),
            ObjectiveSpec("total_levels", "min"),
        ]

    def validate(self, configuration):
        CandidateConfig.model_validate(configuration)
        return True

    def evaluate(self, configuration):
        candidate = CandidateConfig.model_validate(configuration)
        with self._lock:
            self.evaluation_calls += 1
        ranks = {action: index for index, action in enumerate(ACTION_IDS)}
        weighted_forward = sum(
            (index + 1) * ranks[action]
            for index, action in enumerate(candidate.sequence)
        )
        weighted_reverse = sum(
            (len(candidate.sequence) - index) * ranks[action]
            for index, action in enumerate(candidate.sequence)
        )
        return {
            "total_lut_count": float(20_000 - weighted_forward),
            "total_levels": float(20_000 - weighted_reverse),
        }

    def search_space_description(self):
        return "Offline deterministic BOiLS length-20 action-sequence fixture."


class _ContractFollowingGenerator:
    def __init__(self) -> None:
        self.propose_calls = 0
        self.reflect_calls = 0

    async def propose(self, request):
        self.propose_calls += 1
        parents = _parents(request.prompt)
        if request.operation in {
            "three_way_recombination",
            "two_parent_crossover",
        }:
            configuration = copy.deepcopy(
                pilot.SEEDS["expected_composition_c"]
            )
            attribution = (
                SourceAttribution(path="$.sequence[4]", source="left"),
                SourceAttribution(path="$.sequence[15]", source="right"),
            )
            claimed = ()
        else:
            configuration = copy.deepcopy(parents[0]["configuration"])
            match = _EDITABLE_PATH.search(request.prompt)
            assert match is not None
            path, index_text = match.groups()
            index = int(index_text)
            previous = configuration["sequence"][index]
            previous_rank = ACTION_IDS.index(previous)
            configuration["sequence"][index] = ACTION_IDS[
                (previous_rank + 1) % len(ACTION_IDS)
            ]
            attribution = (SourceAttribution(path=path, source="mutation"),)
            claimed = (
                tuple(sorted(set(_INSIGHT_ID.findall(request.prompt))))
                if "QUARANTINED TEST HYPOTHESES" in request.prompt
                else ()
            )
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale="Follow the frozen offline operator contract.",
                intended_changes=("one bounded intervention",),
                source_attribution=attribution,
                claimed_insight_ids=claimed,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        self.reflect_calls += 1
        contrasts = _CONTRAST_ID.findall(request.prompt)
        assert contrasts
        assert "single_operation" in request.prompt
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim="At index 1, advance one action token for this parent state.",
                    trigger="$.sequence[1] is editable in a matched local test",
                    mechanism="the next allowlisted action improved both fixture objectives",
                    affected_paths=("$.sequence[1]",),
                    evidence_summary=(
                        f"Atomic association in full contrast {contrasts[0]}; "
                        "requires a separate paired test."
                    ),
                    confidence=0.6,
                    evidence_contrast_ids=(contrasts[0],),
                ),
            ),
            telemetry=_telemetry(),
        )


def test_offline_boils_pilot_exercises_full_agentic_workflow(tmp_path: Path) -> None:
    problem = _OfflineBoilsProblem()
    generator = _ContractFollowingGenerator()
    event_path = tmp_path / "events.jsonl"
    writer = pilot.DurableJsonlWriter(event_path)
    try:
        summary = asyncio.run(
            pilot.run_workflow(
                problem=problem,
                generator=generator,
                id_seed=20260713,
                event_writer=writer,
                evaluator_concurrency=4,
                max_output_tokens=2_400,
                temperature=0.2,
            )
        )
    finally:
        writer.close()

    assert generator.propose_calls == 8
    assert generator.reflect_calls == 1
    assert summary["provider_calls"]["expected_logical_calls"] == 9
    assert summary["provider_calls"]["successful_logical_calls"] == 9
    assert summary["gates"]["all_three_seeds_valid"] is True
    assert summary["gates"]["reproduction_exact"] is True
    assert summary["gates"]["ancestor_aware_recombination_exact_c"] is True
    assert summary["gates"]["ordinary_crossover_operator_compliant"] is True
    assert summary["gates"]["all_atomic_candidates_operator_compliant"] is True
    assert summary["gates"]["atomic_nonabstention_enforced"] is True
    assert summary["gates"]["reflection_entries_all_quarantined"] is True
    assert summary["gates"]["reflection_entries_all_nonretrievable"] is True
    assert summary["gates"]["quarantine_test_assignment_recorded"] is True
    assert summary["gates"]["quarantine_test_created_no_retrieval_credit"] is True
    assert summary["gates"]["tested_insight_not_auto_promoted"] is True
    assert summary["memory"]["trial_count_before_pair"] == 0
    assert summary["memory"]["trial_count_after_pair"] == 0
    assert (
        summary["evaluation_cache"]["hits"]
        + summary["evaluation_cache"]["coalesced"]
        >= 2
    )
    assert problem.evaluation_calls <= 11

    pair = summary["generation_three_quarantine_pair"]
    assert pair[0]["assignment_kind"] is None
    assert pair[1]["assignment_kind"] == "quarantine_test"
    assert len(pair[1]["selected_insight_refs"]) == 1

    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event_type"] == "pareto_archive_decision" for event in events)
    assert any(event["event_type"] == "parent_selected" for event in events)
    crossover_prepared = next(
        event
        for event in events
        if event["event_type"] == "invocation_prepared"
        and event["operator_kind"] == "two_parent_crossover"
    )
    assert "MACHINE-DERIVED BOILS CROSSOVER DIFF" in crossover_prepared["prompt"]
    assert '"path":"$.sequence[4]"' in crossover_prepared["prompt"]
    assert '"path":"$.sequence[15]"' in crossover_prepared["prompt"]
    assert any(
        event["event_type"] == "invocation_prepared"
        and event["assignment_kind"] == "quarantine_test"
        for event in events
    )
    assert any(
        event["event_type"] == "invocation_completed"
        and event["insight_credit_status"] == "test_only_no_retrieval_credit"
        for event in events
    )
    assert not any(event["event_type"] == "insight_credit_updated" for event in events)

"""Offline integration coverage for the three-generation development probe."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    InsightDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


def _load_probe_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "run_agentic_probe.py"
    )
    name = "_agent_evolve_test_run_agentic_probe_offline"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PROBE = _load_probe_module()
PIPELINE = _PROBE.PIPELINE
JsonlWriter = _PROBE.JsonlWriter
_canonical_json = _PROBE._canonical_json
_run_domain = _PROBE._run_domain


_SELECTED_INSIGHT_ID = re.compile(r'"insight_id":"([^"]+)"')


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/probe-integration",
        resolved_model="offline/probe-integration",
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


def _parents_from_prompt(prompt: str) -> list[dict[str, object]]:
    payload = prompt.split("\nPARENTS\n", 1)[1].split(
        "\nSELECTED MEMORY HYPOTHESES", 1
    )[0]
    parents = json.loads(payload)
    assert type(parents) is list
    return parents


class _DeterministicProbeGenerator:
    """Exercise the complete runner policy without a provider or task queue."""

    _MUTATION_RUNTIMES = (
        # Generation two: the no-memory arm discovers the local runtime optimum;
        # randomized-memory arms then provide heterogeneous score evidence.
        {"threads": 8, "prefetch_distance": 4, "data_layout": "soa"},
        {"threads": 8, "prefetch_distance": 8, "data_layout": "soa"},
        {"threads": 4, "prefetch_distance": 4, "data_layout": "soa"},
        {"threads": 4, "prefetch_distance": 16, "data_layout": "soa"},
        {"threads": 2, "prefetch_distance": 4, "data_layout": "soa"},
        {"threads": 8, "prefetch_distance": 2, "data_layout": "soa"},
        {"threads": 8, "prefetch_distance": 4, "data_layout": "blocked"},
        # Generation three: every arm is a real scoped mutation of the optimum.
        {"threads": 8, "prefetch_distance": 16, "data_layout": "soa"},
        {"threads": 4, "prefetch_distance": 4, "data_layout": "soa"},
        {"threads": 8, "prefetch_distance": 8, "data_layout": "soa"},
        {"threads": 2, "prefetch_distance": 4, "data_layout": "soa"},
        {"threads": 8, "prefetch_distance": 4, "data_layout": "blocked"},
    )

    def __init__(self) -> None:
        self.propose_operations: list[str] = []
        self.reflection_requests = []
        self._mutation_index = 0

    async def propose(self, request):
        self.propose_operations.append(request.operation)
        selected_ids = tuple(
            sorted(set(_SELECTED_INSIGHT_ID.findall(request.prompt)))
        )

        if request.operation == "typed_mutation":
            parent = _parents_from_prompt(request.prompt)[0]
            configuration = copy.deepcopy(parent["configuration"])
            configuration["runtime"] = dict(
                self._MUTATION_RUNTIMES[self._mutation_index]
            )
            self._mutation_index += 1
            source_attribution = (
                SourceAttribution(path="$.runtime", source="mutation"),
            )
        else:
            configuration = copy.deepcopy(
                PIPELINE.known_recombination_target
            )
            source_attribution = (
                SourceAttribution(path="$.passes", source="left"),
                SourceAttribution(path="$.frontend", source="left"),
                SourceAttribution(path="$.backend", source="left"),
                SourceAttribution(path="$.runtime", source="right"),
            )

        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale=(
                    "Deterministic offline candidate for runner integration."
                ),
                intended_changes=("Exercise the declared operator.",),
                source_attribution=source_attribution,
                claimed_insight_ids=selected_ids,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        self.reflection_requests.append(request)
        index = len(self.reflection_requests)
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim=f"Offline reflected runtime hypothesis {index}.",
                    trigger="runtime mutation evidence is available",
                    mechanism="a controlled prefetch change can alter throughput",
                    affected_paths=("$.runtime.prefetch_distance",),
                    evidence_summary=(
                        "Offline integration fixture; no empirical claim."
                    ),
                    confidence=0.9,
                    evidence_contrast_ids=request.available_contrast_ids[:1],
                ),
            ),
            telemetry=_telemetry(),
        )


def test_run_domain_exercises_three_generations_and_score_phase(tmp_path) -> None:
    generator = _DeterministicProbeGenerator()
    event_path = tmp_path / "events.jsonl"
    writer = JsonlWriter(event_path)
    try:
        summary = asyncio.run(
            _run_domain(
                PIPELINE,
                generator=generator,
                seed=20260713,
                event_writer=writer,
                max_output_tokens=1_600,
                temperature=0.2,
            )
        )
    finally:
        writer.close()

    # The returned report must remain directly serializable by the runner's
    # canonical JSON boundary, not merely by a permissive test serializer.
    assert json.loads(_canonical_json(summary))["domain"] == PIPELINE.name
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]

    assert Counter(generator.propose_operations) == {
        "two_parent_crossover": 1,
        "three_way_recombination": 1,
        "typed_mutation": 12,
    }
    assert len(generator.reflection_requests) == 2
    assert [
        request.prompt.count('"operator_invocation_id":')
        for request in generator.reflection_requests
    ] == [3, 12]

    assert len(summary["generation_one"]) == 3
    assert len(summary["generation_two"]) == 7
    assert len(summary["generation_three"]) == 5
    assert summary["counts"] == {
        "logical_variation_invocations": 15,
        "candidate_outputs": 15,
        "valid_candidates": 15,
        "operator_compliant_candidates": 15,
        "evidence_compliant_candidates": 15,
        "positive_reward_candidates": 7,
        "verified_recombination_target_matches": 1,
    }
    assert summary["memory"]["entry_count"] == 6
    assert summary["memory"]["trial_count"] == 10
    assert summary["provider_calls"] == {
        "expected_logical_calls": 16,
        "successful_logical_calls": 16,
        "failed_logical_calls": 0,
        "successful_attempts_reported": 16,
        "input_tokens_successful_responses": 0,
        "output_tokens_successful_responses": 0,
        "reasoning_tokens_successful_responses": 0,
        "cost_usd_successful_responses": "0",
        "responses_without_reported_cost": 0,
        "cost_scope": (
            "Successful responses only; failed attempts can be billable and are "
            "not included unless the provider returned usage telemetry."
        ),
        "requested_models": {"offline/probe-integration": 16},
        "resolved_models": {"offline/probe-integration": 16},
        "resolved_providers": {"offline": 16},
        "failure_types": {},
    }

    by_label = {
        event["label"]: event
        for event in events
        if event["event_type"] == "invocation_prepared"
    }
    g2_memory = [
        by_label[f"g2_mutation_random_memory_{index}"]
        for index in range(1, 7)
    ]
    g2_score_contexts = {event["score_context_hash"] for event in g2_memory}
    assert len(g2_score_contexts) == 1
    policy_snapshot = summary["memory"][
        "policy_snapshot_before_generation_three"
    ]
    assert policy_snapshot["context_hash"] in g2_score_contexts
    assert policy_snapshot["entry_count"] == 5
    assert len(policy_snapshot["score_evidence"]) == 5
    assert {event["selection_mode"] for event in g2_memory} == {
        "explore_uniform"
    }

    exploit = [
        by_label[f"g3_mutation_score_exploit_{index}"]
        for index in range(1, 3)
    ]
    holdout = [
        by_label[f"g3_mutation_uniform_holdout_{index}"]
        for index in range(1, 3)
    ]
    assert {event["score_context_hash"] for event in (*exploit, *holdout)} == (
        g2_score_contexts
    )
    assert {
        (event["selection_mode"], event["exploration_probability"]["numerator"])
        for event in exploit
    } == {("exploit", 0)}
    assert {
        (event["selection_mode"], event["exploration_probability"]["numerator"])
        for event in holdout
    } == {("explore_uniform", 1)}

    # Both exploitation calls use the explicit immutable snapshot captured
    # after G2 and before G3, so later reflection cannot rewrite the comparison.
    reflection_events = [
        event for event in events if event["event_type"] == "reflection_completed"
    ]
    assert len(reflection_events) == 2
    assert all(event["insights"] for event in reflection_events)
    scored = policy_snapshot["score_evidence"]
    top_two = {
        item["insight_id"]
        for item in sorted(
            scored,
            key=lambda item: (
                -item["retrieval_score"],
                item["insight_id"],
                item["version"],
            ),
        )[:2]
    }
    assert all(set(event["selected_insight_ids"]) == top_two for event in exploit)

    generation_one_recombination = next(
        item
        for item in summary["generation_one"]
        if item["label"] == "g1_recombine_no_memory"
    )
    assert generation_one_recombination["known_recombination_target_match"]
    assert generation_one_recombination["candidate"]["preservation_verified"]
    generation_two_control = next(
        item
        for item in summary["generation_two"]
        if item["label"] == "g2_mutation_no_memory"
    )
    assert summary["mutation_parent_id"] == generation_one_recombination[
        "candidate"
    ]["candidate_id"]
    assert summary["exploitation_parent_id"] == generation_two_control[
        "candidate"
    ]["candidate_id"]

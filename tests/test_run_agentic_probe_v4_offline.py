"""Offline end-to-end gate for the v4 atomic-memory development workflow."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import re
import sys
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


def _load_v4_probe():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "run_agentic_probe_v4.py"
    )
    name = "_agent_evolve_test_run_agentic_probe_v4"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4 = _load_v4_probe()
_INSIGHT_ID = re.compile(r'"insight_id":"([^"]+)"')


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/v4",
        resolved_model="offline/v4",
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


def _parent_configuration(prompt: str) -> dict[str, object]:
    payload = prompt.split("\nPARENTS\n", 1)[1].split(
        "\nSELECTED MEMORY HYPOTHESES", 1
    )[0]
    parents = json.loads(payload)
    return copy.deepcopy(parents[0]["configuration"])


class _ContractFollowingGenerator:
    def __init__(self) -> None:
        self.propose_calls = 0
        self.reflect_calls = 0

    async def propose(self, request):
        self.propose_calls += 1
        if request.operation == "three_way_recombination":
            configuration = copy.deepcopy(
                V4.pipeline_problem.DEVELOPMENT_RECOMBINATION_TARGET
            )
            attribution = (
                SourceAttribution(path="$.passes", source="left"),
                SourceAttribution(path="$.frontend", source="left"),
                SourceAttribution(path="$.backend", source="left"),
                SourceAttribution(path="$.runtime", source="right"),
            )
            claimed = ()
        else:
            configuration = _parent_configuration(request.prompt)
            before = copy.deepcopy(configuration)
            selected = tuple(sorted(set(_INSIGHT_ID.findall(request.prompt))))
            claimed = selected
            if "maximum allowed value 16" in request.prompt:
                configuration["runtime"]["prefetch_distance"] = 16
            elif "prefetch_distance to 4" in request.prompt:
                configuration["runtime"]["prefetch_distance"] = 4
            elif "data_layout to soa" in request.prompt:
                configuration["runtime"]["data_layout"] = "soa"
            elif "data_layout to blocked" in request.prompt:
                configuration["runtime"]["data_layout"] = "blocked"
            elif "threads to the maximum allowed value 8" in request.prompt:
                configuration["runtime"]["threads"] = 8
            elif "threads to the moderate value 4" in request.prompt:
                configuration["runtime"]["threads"] = 4
            else:
                # The no-memory transport control follows the obvious atomic
                # vector-layout move.  It is a control, not evidence of memory
                # superiority.
                configuration["runtime"]["data_layout"] = "soa"

            changed = [
                name
                for name in ("threads", "prefetch_distance", "data_layout")
                if configuration["runtime"][name] != before["runtime"][name]
            ]
            attribution = tuple(
                SourceAttribution(
                    path=f"$.runtime.{name}",
                    source="mutation",
                )
                for name in changed
            )

        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale="Follow the selected atomic action or abstain if already satisfied.",
                intended_changes=("one machine-bounded coordinate",),
                source_attribution=attribution,
                claimed_insight_ids=claimed,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        self.reflect_calls += 1
        assert "single_operation" in request.prompt
        assert '"contrast_scope":"joint_intervention"' not in request.prompt
        contrast = re.search(r'"contrast_id":"([0-9a-f]{64})"', request.prompt)
        assert contrast is not None
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim="A prefetch distance of four outperformed maximum lookahead in the observed parent state.",
                    trigger="threads exceed one; retest after a material parent-state change",
                    mechanism="moderate lookahead avoids excess traffic",
                    affected_paths=("$.runtime.prefetch_distance",),
                    evidence_summary=f"Supported by atomic contrast {contrast.group(1)}.",
                    confidence=0.7,
                    evidence_contrast_ids=(contrast.group(1),),
                ),
            ),
            telemetry=_telemetry(),
        )


def test_v4_atomic_workflow_reaches_scoped_oracle_and_records_causal_boundaries(
    tmp_path,
) -> None:
    generator = _ContractFollowingGenerator()
    event_path = tmp_path / "events.jsonl"
    writer = V4.support.JsonlWriter(event_path)
    try:
        summary = asyncio.run(
            V4._run_pipeline_v4(
                generator=generator,
                id_seed=20260713,
                assignment_seed=4,
                event_writer=writer,
                max_output_tokens=2_400,
                temperature=0.2,
            )
        )
    finally:
        writer.close()

    assert generator.propose_calls == 9
    assert generator.reflect_calls == 1
    assert summary["final_parent_id"]
    assert summary["assignment_design"]["preflight"] == summary[
        "assignment_design"
    ]["observed"]
    assert all(
        value is True
        for key, value in summary["gates"].items()
        if key
        not in {
            "discovery_nonabstaining_count",
            "discovery_nonabstaining_unique_count",
        }
    )
    assert summary["gates"]["discovery_nonabstaining_count"] == 3
    assert summary["gates"]["discovery_nonabstaining_unique_count"] == 3
    assert summary["evaluation_cache"]["hits"] >= 4
    assert summary["evaluation_cache"]["coalesced"] >= 1
    assert summary["provider_calls"]["expected_logical_calls"] == 10
    assert summary["provider_calls"]["successful_logical_calls"] == 10
    assert summary["memory"]["confidence_used_as_utility_prior"] is False
    assert summary["memory"]["reflected_entries_status"] == "quarantined_untested"

    transport = summary["generation_three_transport_test"]
    exploit = next(
        row for row in transport if row["label"] == "g3_layout_score_transport"
    )
    assert exploit["scoped_optimum_match"] is True
    assert exploit["action_adherence"]["hypothesis"] == "layout_soa"
    assert exploit["action_adherence"]["action_satisfied"] is True
    assert exploit["candidate"]["configuration"] == (
        V4.pipeline_problem.DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM
    )

    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    prepared = [
        event for event in events if event["event_type"] == "invocation_prepared"
    ]
    atomic = [event for event in prepared if event["mutation_contract"] is not None]
    assert len(atomic) == 8
    assert all(event["mutation_contract"]["max_operations"] == 1 for event in atomic)
    assert all(event["mutation_contract"]["allow_abstention"] for event in atomic)
    assert any(
        event["event_type"] == "evaluation_cache_event"
        and event["cache_event_type"] == "coalesced"
        for event in events
    )

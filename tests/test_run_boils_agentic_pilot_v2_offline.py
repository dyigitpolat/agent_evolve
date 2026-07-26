"""Offline end-to-end gates for the frozen BOiLS patch-native v2 runner."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from decimal import Decimal
from pathlib import Path

from agent_evolve import ObjectiveSpec
from agent_evolve.domain.patch import ArrayIndex
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AtomicMutationDraft,
    InsightDraft,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
)
from examples.benchmarks.boils_abc.actions import CandidateConfig, config_sha256
from examples.development import run_boils_agentic_pilot_v2 as pilot


REPLACEMENTS = {
    1: "rewrite_z",
    7: "resub_z",
    12: "refactor_z",
    18: "resub",
}
OBJECTIVES = {
    (1, "rewrite_z"): (7_935.0, 69.0),
    (7, "resub_z"): (7_940.0, 68.0),
    (12, "refactor_z"): (7_931.0, 69.0),
    (18, "resub"): (7_956.0, 69.0),
}


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model=pilot.MODEL,
        resolved_model=pilot.MODEL,
        resolved_provider="Together",
        provider_response_id="offline-fixture",
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


class _OfflineBoilsProblem:
    candidate_model = CandidateConfig

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.evaluation_calls = 0
        self.seed_was_first = False

    @property
    def objectives(self):
        return (
            ObjectiveSpec("total_lut_count", "min"),
            ObjectiveSpec("total_levels", "min"),
        )

    @staticmethod
    def validate(configuration):
        CandidateConfig.model_validate(
            configuration,
            strict=True,
            by_alias=False,
            by_name=True,
        )
        return True

    def evaluate(self, configuration):
        candidate = CandidateConfig.model_validate(
            configuration,
            strict=True,
            by_alias=False,
            by_name=True,
        )
        candidate_dict = candidate.model_dump(mode="python")
        with self._lock:
            self.evaluation_calls += 1
            call_number = self.evaluation_calls
        if candidate_dict == pilot.PARENT_C:
            if call_number != 1:
                raise AssertionError(
                    "frozen C must be the first and sole seed evaluation"
                )
            self.seed_was_first = True
            return dict(pilot.EXPECTED_SEED_OBJECTIVES)
        differences = [
            index
            for index, (parent, child) in enumerate(
                zip(
                    pilot.PARENT_C["sequence"],
                    candidate_dict["sequence"],
                    strict=True,
                )
            )
            if parent != child
        ]
        if len(differences) != 1:
            raise AssertionError("offline v2 fixture accepts one scalar edit only")
        index = differences[0]
        lut_count, levels = OBJECTIVES[(index, candidate_dict["sequence"][index])]
        return {"total_lut_count": lut_count, "total_levels": levels}

    @staticmethod
    def search_space_description():
        return "Offline BOiLS length-20 patch-native fixture."


class _AtomicContractGenerator:
    def __init__(self, problem: _OfflineBoilsProblem) -> None:
        self.problem = problem
        self.proposal_requests: list[VariationGenerationRequest] = []
        self.reflection_requests: list[ReflectionGenerationRequest] = []
        self.propose_calls = 0
        self.reflect_calls = 0
        self.active_proposals = 0
        self.max_active_proposals = 0
        self._all_proposals_started = asyncio.Event()

    async def propose(self, request: VariationGenerationRequest):
        assert self.problem.seed_was_first
        assert self.problem.evaluation_calls == 1
        contract = request.atomic_mutation_contract
        assert contract is not None
        assert contract.parent_configuration == freeze_json(pilot.PARENT_C)
        final_segment = contract.editable_path.segments[-1]
        assert type(final_segment) is ArrayIndex
        index = final_segment.value
        assert index in pilot.MUTATION_INDICES
        assert request.operation == "typed_mutation"
        assert request.max_output_tokens == pilot.MAX_OUTPUT_TOKENS
        assert request.temperature == pilot.TEMPERATURE

        self.proposal_requests.append(request)
        self.propose_calls += 1
        self.active_proposals += 1
        self.max_active_proposals = max(
            self.max_active_proposals,
            self.active_proposals,
        )
        if self.propose_calls == len(pilot.MUTATION_INDICES):
            self._all_proposals_started.set()
        await asyncio.wait_for(self._all_proposals_started.wait(), timeout=2.0)
        await asyncio.sleep(0)
        self.active_proposals -= 1
        return VariationGenerationResult(
            draft=AtomicMutationDraft(
                path=contract.editable_path,
                replacement=REPLACEMENTS[index],
                design_rationale=f"Replace only frozen sequence index {index}.",
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request: ReflectionGenerationRequest):
        self.reflect_calls += 1
        self.reflection_requests.append(request)
        assert request.operation == "extract_insights"
        assert request.max_insights == 3
        assert request.max_output_tokens == pilot.MAX_OUTPUT_TOKENS
        assert request.temperature == pilot.TEMPERATURE
        assert len(request.available_contrast_ids) == 4
        assert tuple(sorted(request.available_contrast_ids)) == (
            request.available_contrast_ids
        )
        assert all(len(value) == 64 for value in request.available_contrast_ids)
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim="The four local scalar substitutions have context-bound effects.",
                    trigger="The exact frozen parent C and one of the four cited paths.",
                    mechanism=(
                        "Action position changes the downstream ABC transformation context."
                    ),
                    affected_paths=tuple(
                        f"$.sequence[{index}]" for index in pilot.MUTATION_INDICES
                    ),
                    evidence_summary=(
                        "The four full structured contrasts are associated observations, "
                        "not a context-free transfer rule."
                    ),
                    confidence=0.55,
                    evidence_contrast_ids=request.available_contrast_ids,
                ),
            ),
            telemetry=_telemetry(),
        )


class _OneArmFailureGenerator(_AtomicContractGenerator):
    async def propose(self, request: VariationGenerationRequest):
        assert self.problem.seed_was_first
        contract = request.atomic_mutation_contract
        assert contract is not None
        final_segment = contract.editable_path.segments[-1]
        assert type(final_segment) is ArrayIndex
        index = final_segment.value
        self.proposal_requests.append(request)
        self.propose_calls += 1
        if index == 7:
            raise StructuredGenerationError(
                kind=GenerationFailureKind.OUTPUT_INVALID,
                retryable=False,
                safe_message="offline model-output failure",
            )
        return VariationGenerationResult(
            draft=AtomicMutationDraft(
                path=contract.editable_path,
                replacement=REPLACEMENTS[index],
                design_rationale=f"Replace only frozen sequence index {index}.",
            ),
            telemetry=_telemetry(),
        )


def _run(tmp_path: Path, generator_type=_AtomicContractGenerator):
    problem = _OfflineBoilsProblem()
    generator = generator_type(problem)
    event_path = tmp_path / "events.jsonl"
    writer = pilot.v1.DurableJsonlWriter(event_path)
    try:
        summary = asyncio.run(
            pilot.run_workflow(
                problem=problem,
                generator=generator,
                id_seed=20260714,
                event_writer=writer,
                evaluator_concurrency=4,
            )
        )
    finally:
        writer.close()
    events = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    return problem, generator, summary, events


def test_offline_v2_exercises_exact_patch_and_reflection_boundaries(
    tmp_path: Path,
) -> None:
    problem, generator, summary, events = _run(tmp_path)

    assert problem.evaluation_calls == 5
    assert generator.propose_calls == 4
    assert generator.max_active_proposals == 4
    assert generator.reflect_calls == 1
    assert summary["acceptance_passed"] is True
    assert all(summary["gates"].values())
    assert summary["fixed_mutation_order"] == [1, 7, 12, 18]
    assert summary["pareto_archive"]["consideration_count"] == 5
    assert summary["memory"] == {
        "entry_count": 1,
        "trial_count": 0,
        "lifecycle_transition_count": 0,
    }
    calls = summary["provider_calls"]
    assert calls["preregistered_logical_calls"] == 5
    assert calls["executed_logical_calls"] == 5
    assert calls["successful_logical_calls"] == 5
    assert calls["total_attempts_for_successful_logical_calls"] == 5
    assert "successful_attempts_reported" not in calls

    outcomes = summary["generation_one_patch_native"]
    assert len(outcomes) == 4
    for index, outcome in zip(pilot.MUTATION_INDICES, outcomes, strict=True):
        candidate = outcome["candidate"]
        legal = outcome["legal_child"]
        assert outcome["mutation_response_mode"] == "atomic_scalar_replacement_v1"
        assert outcome["atomic_trace_gate"] is True
        assert legal["index"] == index
        assert legal["replacement"] == REPLACEMENTS[index]
        assert candidate["configuration"]["sequence"][index] == REPLACEMENTS[index]
        assert (
            candidate["typed_json_configuration_sha256"]
            == legal["typed_json_configuration_sha256"]
        )
        assert (
            candidate["boils_schema_configuration_sha256"]
            == legal["boils_configuration_sha256"]
        )
        assert candidate["source_attribution"] == [
            {"path": f"$.sequence[{index}]", "source": "mutation"}
        ]

    reflection_request = generator.reflection_requests[0]
    reflection_event = next(
        event for event in events if event["event_type"] == "reflection_requested"
    )
    assert reflection_request.prompt == reflection_event["prompt"]
    rows = summary["reflection"]["machine_derived_rows"]
    assert len(rows) == 4
    contrast_ids: list[str] = []
    for index, row, outcome in zip(
        pilot.MUTATION_INDICES,
        rows,
        outcomes,
        strict=True,
    ):
        contrast = row["machine_derived_contrasts"][0]
        operation = contrast["system_derived_operations"][0]
        contrast_ids.append(contrast["contrast_id"])
        assert contrast["parent_configuration_hash"] == (
            pilot.EXPECTED_PARENT_TYPED_SHA256
        )
        assert (
            contrast["child_configuration_hash"]
            == outcome["candidate"]["typed_json_configuration_sha256"]
        )
        candidate_event = next(
            event
            for event in events
            if event["event_type"] == "candidate_evaluated"
            and event["candidate_id"] == outcome["candidate"]["candidate_id"]
        )
        assert (
            contrast["derived_patch_hash"] == candidate_event["materialized_patch_hash"]
        )
        assert candidate_event["parent_patch_hashes"] == [
            contrast["derived_patch_hash"]
        ]
        assert contrast["contrast_scope"] == "single_operation"
        assert contrast["changed_paths"] == [f"$.sequence[{index}]"]
        assert operation == {
            "operation_kind": "replace_scalar",
            "path": f"$.sequence[{index}]",
            "old_value": pilot.PARENT_C["sequence"][index],
            "new_value": REPLACEMENTS[index],
            "old_value_hash": typed_json_sha256(
                freeze_json(pilot.PARENT_C["sequence"][index])
            ),
            "new_value_hash": typed_json_sha256(freeze_json(REPLACEMENTS[index])),
        }
    assert tuple(sorted(contrast_ids)) == reflection_request.available_contrast_ids
    entry = summary["reflection"]["entries"][0]
    assert entry["evidence_contrast_ids"] == list(
        reflection_request.available_contrast_ids
    )
    assert entry["evidence_lineage"]["cited_contrast_ids"] == list(
        reflection_request.available_contrast_ids
    )
    assert entry["lifecycle_state"] == "quarantined"
    assert entry["retrievable"] is False

    assert not any(
        event["event_type"]
        in {
            "insight_credit_updated",
            "insight_credit_censored",
            "insight_lifecycle_transition",
            "reflection_evidence_contrast_ids_filtered",
        }
        for event in events
    )
    assert {
        event["atomic_submitted_path"]
        for event in events
        if event["event_type"] == "candidate_evaluated"
        and event.get("atomic_submitted_path") is not None
    } == {f"$.sequence[{index}]" for index in pilot.MUTATION_INDICES}
    first_archive_decisions = {
        event["consideration_sequence"]: event["candidate_id"]
        for event in events
        if event["event_type"] == "pareto_archive_decision"
        and event["action"] in {"admitted", "rejected"}
    }
    assert [first_archive_decisions[sequence] for sequence in range(2, 6)] == [
        outcome["candidate"]["candidate_id"] for outcome in outcomes
    ]


def test_legal_universe_and_factorial_fixture_replay_exactly() -> None:
    rows = pilot.legal_child_rows()
    assert len(rows) == 40
    assert pilot._sha256_bytes(pilot.LEGAL_CHILD_BYTES) == (
        pilot.EXPECTED_LEGAL_FILE_SHA256
    )
    for row in rows:
        child = copy.deepcopy(pilot.PARENT_C)
        child["sequence"][row["index"]] = row["replacement"]
        assert config_sha256(child) == row["boils_configuration_sha256"]
        assert (
            typed_json_sha256(freeze_json(child))
            == row["typed_json_configuration_sha256"]
        )

    replay = pilot.factorial_replay()
    assert replay["index_1_effect"] == (-9, 0)
    assert replay["index_12_effect"] == (-13, 0)
    assert replay["observed_joint_effect"] == (-26, 1)
    assert replay["additive_expected_joint_effect"] == (-22, 0)
    assert replay["interaction"] == (-4, 1)
    for cell in replay["cells"].values():
        assert len(cell["boils_configuration_sha256"]) == 64


def test_missing_fixed_arm_skips_reflection_without_selective_rerun(
    tmp_path: Path,
) -> None:
    problem, generator, summary, events = _run(tmp_path, _OneArmFailureGenerator)

    assert generator.propose_calls == 4
    assert generator.reflect_calls == 0
    assert problem.evaluation_calls == 4
    assert summary["acceptance_passed"] is False
    assert summary["reflection"] == {
        "attempted": False,
        "skipped": True,
        "failure_type": None,
        "max_insights": 3,
        "entries": [],
        "machine_derived_rows": [],
    }
    assert summary["provider_calls"]["executed_logical_calls"] == 4
    assert summary["provider_calls"]["reflection_skipped_calls"] == 1
    assert summary["provider_calls"]["failed_logical_calls"] == 1
    assert summary["memory"]["trial_count"] == 0
    skipped = next(
        event for event in events if event["event_type"] == "reflection_skipped"
    )
    assert skipped["missing_indices"] == [7]
    assert not any(event["event_type"] == "reflection_requested" for event in events)

"""Offline end-to-end gates for the frozen BOiLS recombination-v3 runner."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import threading
import time

import pytest
from pydantic import ValidationError

from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.ports.structured_generator import StructuredGenerationResponse
from examples.benchmarks.boils_abc.actions import config_sha256
from examples.benchmarks.boils_abc.evaluator import (
    AbcEvaluationError,
    BoilsEvaluation,
    BoilsEvaluationFailure,
    CircuitDiagnostics,
    CircuitEvaluation,
)
from examples.development import run_boils_recombination_v3 as block
from examples.development.corpus_paths import resolve_corpus_path


NEW_OBJECTIVES = {
    "AD": (7_900, 68),
    "BD": (7_890, 70),
    "ABD": (7_860, 68),
}


def _diagnostics(cpu: int) -> CircuitDiagnostics:
    return CircuitDiagnostics(
        status="passed",
        returncode=0,
        elapsed_s=0.01,
        timeout_s=float(block.PER_CANDIDATE_TIMEOUT_SECONDS),
        equivalent=True,
        error_signatures=(),
        stdout_excerpt="offline fixture",
        stderr_excerpt="",
        stdout_sha256="0" * 64,
        stderr_sha256="0" * 64,
        abc_program="offline fixture",
        argv=("offline-abc",),
        cpu_affinity=(cpu,),
    )


class _OfflineEvaluator:
    def __init__(
        self,
        observer,
        *,
        wrong_seed: bool = False,
        bad_abc: bool = False,
        invalid_arm: str | None = None,
        invalid_status: str = "timeout",
        new_objectives=None,
    ):
        self._observer = observer
        self._wrong_seed = wrong_seed
        self._bad_abc = bad_abc
        self._invalid_arm = invalid_arm
        self._invalid_status = invalid_status
        self._new_objectives = dict(NEW_OBJECTIVES if new_objectives is None else new_objectives)
        self._by_hash = {
            arm.boils_configuration_sha256: arm for arm in block.CUBE
        }
        self._barrier = threading.Barrier(block.CHILD_WORKERS)
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.seed_done = False
        self.active_children = 0
        self.max_active_children = 0

    def provenance(self):
        return {
            "abc_binary_sha256": (
                "f" * 64 if self._bad_abc else block.EXPECTED_ABC_SHA256
            ),
            "circuits": [
                {
                    "name": "log2",
                    "sha256": block.EXPECTED_CIRCUIT_SHA256,
                }
            ],
            "lut_inputs": 6,
            "per_circuit_timeout_s": float(block.PER_CANDIDATE_TIMEOUT_SECONDS),
            "affinity_sets": [[cpu] for cpu in block.PHYSICAL_CPUS],
        }

    def evaluate(self, configuration):
        digest = config_sha256(configuration)
        arm = self._by_hash[digest]
        label = arm.label
        with self._lock:
            self.calls.append(label)
            if label == "C":
                assert self.calls == ["C"]
            else:
                assert self.seed_done
                self.active_children += 1
                self.max_active_children = max(
                    self.max_active_children, self.active_children
                )
        if label != "C":
            self._barrier.wait(timeout=3.0)
            # Make completion differ from fixed report order.
            time.sleep(0.001 * (3 - block.NEW_ARM_ORDER.index(label)))
        try:
            if label == "C":
                lut_count, levels = block.EXPECTED_SEED_OBJECTIVES
                if self._wrong_seed:
                    lut_count += 1
                cpu = 8
            else:
                lut_count, levels = self._new_objectives[label]
                cpu = 8 + block.NEW_ARM_ORDER.index(label)
            if label == self._invalid_arm:
                diagnostics = CircuitDiagnostics(
                    status=self._invalid_status,
                    returncode=None,
                    elapsed_s=0.01,
                    timeout_s=float(block.PER_CANDIDATE_TIMEOUT_SECONDS),
                    equivalent=False,
                    error_signatures=(),
                    stdout_excerpt="offline fixture failure",
                    stderr_excerpt="",
                    stdout_sha256="0" * 64,
                    stderr_sha256="0" * 64,
                    abc_program="offline fixture",
                    argv=("offline-abc",),
                    cpu_affinity=(cpu,),
                )
                failure = BoilsEvaluationFailure(
                    configuration_sha256=arm.boils_configuration_sha256,
                    sequence=arm.sequence,
                    abc_binary_sha256=block.EXPECTED_ABC_SHA256,
                    failed_circuit_name="log2",
                    completed_circuit_results=(),
                    diagnostics=diagnostics,
                    elapsed_s=0.01,
                    affinity_queue_wait_s=0.001,
                    cpu_affinity=(cpu,),
                )
                self._observer(failure)
                raise AbcEvaluationError("log2", diagnostics)
            diagnostics = _diagnostics(cpu)
            circuit = CircuitEvaluation(
                circuit_name="log2",
                circuit_sha256=block.EXPECTED_CIRCUIT_SHA256,
                inputs=32,
                outputs=32,
                lut_count=lut_count,
                edge_count=lut_count * 2,
                aig_count=lut_count * 3,
                levels=levels,
                diagnostics=diagnostics,
            )
            result = BoilsEvaluation(
                configuration_sha256=arm.boils_configuration_sha256,
                sequence=arm.sequence,
                abc_binary_sha256=block.EXPECTED_ABC_SHA256,
                lut_inputs=6,
                circuit_results=(circuit,),
                total_lut_count=lut_count,
                total_levels=levels,
                max_levels=levels,
                elapsed_s=0.01,
                affinity_queue_wait_s=0.001,
                cpu_affinity=(cpu,),
            )
            self._observer(result)
            if label == "C":
                self.seed_done = True
            return result
        finally:
            if label != "C":
                with self._lock:
                    self.active_children -= 1


def _prediction(ranking=("ABD", "AD", "BD")) -> block.RecombinationPrediction:
    return block.RecombinationPrediction(
        ranking=list(ranking),
        AD=block.ArmPrediction(
            total_lut_count=block.DirectionProbabilities(
                decrease=0.7, same=0.1, increase=0.2
            ),
            total_levels=block.DirectionProbabilities(
                decrease=0.5, same=0.4, increase=0.1
            ),
        ),
        BD=block.ArmPrediction(
            total_lut_count=block.DirectionProbabilities(
                decrease=0.6, same=0.2, increase=0.2
            ),
            total_levels=block.DirectionProbabilities(
                decrease=0.2, same=0.3, increase=0.5
            ),
        ),
        ABD=block.ArmPrediction(
            total_lut_count=block.DirectionProbabilities(
                decrease=0.8, same=0.1, increase=0.1
            ),
            total_levels=block.DirectionProbabilities(
                decrease=0.4, same=0.5, increase=0.1
            ),
        ),
    )


class _OfflinePredictor:
    def __init__(self, evaluator: _OfflineEvaluator, *, ranking=("ABD", "AD", "BD")):
        self.evaluator = evaluator
        self.ranking = ranking
        self.requests = []
        self.returned = False

    async def __call__(self, request):
        assert self.evaluator.seed_done
        assert self.evaluator.calls == ["C"]
        assert request.output_type is block.RecombinationPrediction
        assert request.operation == "recombination_prediction"
        assert request.max_output_tokens == block.MAX_OUTPUT_TOKENS
        assert request.temperature == block.TEMPERATURE
        assert "SEALED_AND_NOT_EVALUATED" in request.prompt
        assert "expected marginal search value" in request.prompt
        assert "whose front is {D,AB}" in request.prompt
        assert "hypervolume is 213" in request.prompt
        assert "full_oracle" not in request.prompt
        assert '"hypervolume":700' not in request.prompt
        assert "descending marginal hypervolume" in request.prompt
        assert "lower Pareto layer" in request.prompt
        assert all(f'"arm":"{label}"' in request.prompt for label in block.ALL_ARM_ORDER)
        self.requests.append(request)
        response = StructuredGenerationResponse(
            value=_prediction(self.ranking),
            requested_model=block.MODEL,
            resolved_model=block.MODEL,
            resolved_provider=block.RESOLVED_PROVIDER,
            provider_response_id="offline-fixture",
            finish_reason="tool_call",
            input_tokens=100,
            output_tokens=40,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.001"),
            latency_ns=10,
        )
        self.returned = True
        return AttemptedStructuredGenerationResponse(response=response, attempt_count=1)


def _writers(tmp_path: Path, evaluator_options=None):
    event_writer = block.v1.DurableJsonlWriter(tmp_path / "events.jsonl")
    evaluation_writer = block.v1.DurableJsonlWriter(tmp_path / "evaluations.jsonl")
    trace = block.oracle.TraceRecorder(event_writer)
    recorder = block.oracle.EvaluationPublicationRecorder(
        evaluation_writer,
        trace,
        schedule=block.PHYSICAL_SCHEDULE,
    )
    evaluator = _OfflineEvaluator(recorder, **(evaluator_options or {}))
    predictor = _OfflinePredictor(evaluator)
    return event_writer, evaluation_writer, trace, recorder, evaluator, predictor


def _run(tmp_path: Path, *, evaluator_options=None, clock_ns=time.perf_counter_ns):
    values = _writers(tmp_path, evaluator_options)
    event_writer, evaluation_writer, trace, recorder, evaluator, predictor = values
    try:
        summary = asyncio.run(
            block.run_block(
                evaluator=evaluator,
                recorder=recorder,
                trace=trace,
                predictor=predictor,
                evidence_bundle=block.verify_evidence_bundle(),
                clock_ns=clock_ns,
            )
        )
    finally:
        event_writer.close()
        evaluation_writer.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return evaluator, predictor, summary, events


def test_engine_materializes_and_replays_exact_cube_identities() -> None:
    assert block.support._sha256(block.PREREGISTRATION_PATH) == (
        block.EXPECTED_PREREGISTRATION_SHA256
    )
    assert block.support._sha256(block.CORRECTION_PATH) == (
        block.EXPECTED_CORRECTION_SHA256
    )
    assert tuple(arm.label for arm in block.CUBE) == block.ALL_ARM_ORDER
    for arm in block.CUBE:
        assert config_sha256(arm.configuration) == arm.boils_configuration_sha256
        assert typed_json_sha256(freeze_json(arm.configuration)) == (
            arm.typed_json_configuration_sha256
        )
        patch = arm.patch_record
        assert patch is not None
        assert patch["replay_verified"] is True
        assert patch["target_hash"] == arm.typed_json_configuration_sha256
        if arm.label == "C":
            assert patch["operation_count"] == 0
            assert patch["operations"] == []
            assert patch["materialization_kind"] == "identity_reproduction"
            continue
        assert patch["operation_count"] == len(arm.branches)
        assert all(
            operation["operation_kind"] == "replace_scalar"
            and operation["attribution_provenance"]
            == "system_derived_from_frozen_path"
            for operation in patch["operations"]
        )


def test_fixed_wave_ignores_model_ranking_and_computes_full_cube(tmp_path: Path) -> None:
    evaluator, predictor, summary, events = _run(tmp_path)

    assert len(predictor.requests) == 1
    assert evaluator.calls[0] == "C"
    assert set(evaluator.calls[1:]) == set(block.NEW_ARM_ORDER)
    assert evaluator.max_active_children == 3
    assert summary["protocol_acceptance_passed"] is True
    assert all(summary["gates"].values())
    assert summary["model_prediction"]["ranking"] == ["ABD", "AD", "BD"]
    assert summary["decision"]["llm_affected_physical_selection_or_order"] is False

    submitted = [
        event["arm"]
        for event in events
        if event["event_type"] == "candidate_submitted"
    ]
    assert submitted == ["C", "AD", "BD", "ABD"]
    completed_position = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "prediction_completed"
    )
    first_child_position = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "candidate_submitted" and event["arm"] == "AD"
    )
    assert completed_position < first_child_position

    interactions = summary["interactions"]
    assert (interactions["I_AB"]["total_lut_count"], interactions["I_AB"]["total_levels"]) == (-4, 1)
    assert (interactions["I_AD"]["total_lut_count"], interactions["I_AD"]["total_levels"]) == (-16, -1)
    assert (interactions["I_BD"]["total_lut_count"], interactions["I_BD"]["total_levels"]) == (-22, 1)
    assert (interactions["I_ABD"]["total_lut_count"], interactions["I_ABD"]["total_levels"]) == (-1, -2)
    arithmetic = summary["triple_prediction_arithmetic"]
    assert arithmetic["additive_main_effect_prediction"] == [7903, 69]
    assert arithmetic["main_plus_pair_effect_prediction"] == [7861, 70]
    assert arithmetic["main_plus_pair_error_observed_minus_predicted"] == [-1, -2]
    assert arithmetic["third_order_residual_equals_main_plus_pair_error"] is True
    assert summary["pareto"]["triple_enters_combined_development_front"] is True
    assert summary["pareto"]["primary_comparison_archive"] == [
        "C",
        "A",
        "B",
        "D",
        "AB",
    ]
    assert summary["pareto"]["preblock_front"] == ["D", "AB"]
    assert summary["hypervolume"]["preblock"] == 213
    assert summary["hypervolume"]["terminal"] > summary["hypervolume"]["preblock"]
    assert summary["full_oracle_sensitivity_post_prediction_only"][
        "preblock_hypervolume"
    ] == 700
    assert summary["full_oracle_sensitivity_post_prediction_only"][
        "primary_decision_uses_this"
    ] is False
    assert summary["model_categorical_calibration"]["cell_count"] == 6
    assert summary["model_rank_calibration"]["predicted_ranking"] == [
        "ABD",
        "AD",
        "BD",
    ]


def test_prediction_contract_has_no_selection_or_configuration_surface() -> None:
    schema = block.RecombinationPrediction.model_json_schema()
    assert schema["additionalProperties"] is False
    properties = set(schema["properties"])
    assert properties == {"ranking", "AD", "BD", "ABD"}
    ranking_schema = schema["properties"]["ranking"]
    assert ranking_schema["minItems"] == ranking_schema["maxItems"] == 3
    assert ranking_schema["uniqueItems"] is True
    assert set(ranking_schema["items"]["enum"]) == set(block.NEW_ARM_ORDER)
    assert properties.isdisjoint({"selection", "configuration", "rationale", "confidence"})
    with pytest.raises(ValidationError):
        block.RecombinationPrediction.model_validate(
            {**_prediction().model_dump(), "selection": "ABD"}, strict=True
        )
    with pytest.raises(ValidationError):
        block.RecombinationPrediction.model_validate(
            {**_prediction().model_dump(), "ranking": ["ABD", "ABD", "AD"]},
            strict=True,
        )
    with pytest.raises(ValidationError):
        block.DirectionProbabilities(decrease=0.4, same=0.4, increase=0.4)

    valid_response = StructuredGenerationResponse(
        value=_prediction(),
        requested_model=block.MODEL,
        resolved_model=block.MODEL,
        resolved_provider=block.RESOLVED_PROVIDER,
        provider_response_id="offline-fixture",
        finish_reason="tool_call",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.001"),
        latency_ns=1,
    )
    with pytest.raises(RuntimeError, match="Together"):
        block._provider_record(replace(valid_response, resolved_provider="Parasail"))
    with pytest.raises(RuntimeError, match="cost gate"):
        block._provider_record(replace(valid_response, cost_usd=Decimal("0.011")))


def test_duplicate_front_point_alone_does_not_advance(tmp_path: Path) -> None:
    objectives = {
        "AD": block.KNOWN_OBJECTIVES["D"],
        "BD": (8_000, 70),
        "ABD": (8_001, 70),
    }
    _, _, summary, _ = _run(
        tmp_path,
        evaluator_options={"new_objectives": objectives},
    )
    decisions = {
        row["arm"]: row for row in summary["pareto"]["new_arm_decisions"]
    }
    assert decisions["AD"]["enters_combined_front"] is True
    assert decisions["AD"]["unique_objective_vector_on_combined_cube_front"] is False
    assert decisions["AD"]["marginal_fixed_reference_hv_gain"] == 0
    assert decisions["AD"]["contributes_search_value"] is False
    assert all(row["contributes_search_value"] is False for row in decisions.values())
    assert summary["decision"][
        "deterministic_disjoint_recombination_advances"
    ] is False


def test_candidate_local_invalid_retains_partial_negative_summary(tmp_path: Path) -> None:
    evaluator, _, summary, events = _run(
        tmp_path,
        evaluator_options={"invalid_arm": "BD", "invalid_status": "timeout"},
    )
    assert set(evaluator.calls) == {"C", "AD", "BD", "ABD"}
    assert summary["status"] == "partial_candidate_local_invalid"
    assert summary["protocol_acceptance_passed"] is True
    assert summary["partial_negative_record"] == {
        "present": True,
        "invalid_arms": [
            {"arm": "BD", "candidate_local_failure_status": "timeout"}
        ],
        "fixed_arms_consumed_without_replacement": True,
    }
    assert summary["interactions"]["I_AB"]["available"] is True
    assert summary["interactions"]["I_AD"]["available"] is True
    assert summary["interactions"]["I_BD"]["available"] is False
    assert summary["interactions"]["I_ABD"]["available"] is False
    assert summary["triple_prediction_arithmetic"]["available"] is False
    assert summary["decision"]["interaction_recording_advances"] is False
    assert summary["model_categorical_calibration"]["cell_count"] == 4
    assert summary["model_rank_calibration"]["missing_arms"] == ["BD"]
    assert summary["resources"]["physical_evaluations"] == 4
    assert summary["resources"]["candidate_local_invalid_arms"] == 1
    assert sum(
        event["event_type"] == "candidate_submitted"
        and event.get("arm") in block.NEW_ARM_ORDER
        for event in events
    ) == 3


def test_cec_failure_is_block_fatal_after_all_fixed_arms_run(tmp_path: Path) -> None:
    values = _writers(
        tmp_path,
        {"invalid_arm": "BD", "invalid_status": "cec_failed_or_missing"},
    )
    event_writer, evaluation_writer, trace, recorder, evaluator, predictor = values
    try:
        with pytest.raises(RuntimeError, match="mandatory CEC"):
            asyncio.run(
                block.run_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    predictor=predictor,
                    evidence_bundle=block.verify_evidence_bundle(),
                )
            )
    finally:
        event_writer.close()
        evaluation_writer.close()
    assert set(evaluator.calls) == {"C", "AD", "BD", "ABD"}


def test_bad_seed_aborts_before_prediction_or_children(tmp_path: Path) -> None:
    values = _writers(tmp_path, {"wrong_seed": True})
    event_writer, evaluation_writer, trace, recorder, evaluator, predictor = values
    try:
        with pytest.raises(block.oracle.SeedGateError):
            asyncio.run(
                block.run_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    predictor=predictor,
                    evidence_bundle=block.verify_evidence_bundle(),
                )
            )
    finally:
        event_writer.close()
        evaluation_writer.close()
    assert evaluator.calls == ["C"]
    assert predictor.requests == []


def test_oracle_json_loader_runs_only_after_durable_prediction(tmp_path: Path) -> None:
    values = _writers(tmp_path)
    event_writer, evaluation_writer, trace, recorder, evaluator, predictor = values
    sentinel_calls = 0

    def deferred_oracle_sentinel():
        nonlocal sentinel_calls
        sentinel_calls += 1
        assert predictor.returned is True
        assert evaluator.calls == ["C"]
        durable_events = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert durable_events[-1]["event_type"] == "prediction_completed"
        assert not any(
            event["event_type"] == "candidate_submitted"
            and event.get("arm") in block.NEW_ARM_ORDER
            for event in durable_events
        )
        return block.verify_deferred_oracle_evidence()

    try:
        summary = asyncio.run(
            block.run_block(
                evaluator=evaluator,
                recorder=recorder,
                trace=trace,
                predictor=predictor,
                evidence_bundle=block.verify_evidence_bundle(),
                deferred_oracle_loader=deferred_oracle_sentinel,
            )
        )
    finally:
        event_writer.close()
        evaluation_writer.close()
    assert sentinel_calls == 1
    assert summary["deferred_oracle_verification"]["verified"] is True


def test_identity_provenance_and_source_binding_fail_closed(tmp_path: Path) -> None:
    identities = copy.deepcopy(block.EXPECTED_IDENTITIES)
    identities["AD"] = ("0" * 64, identities["AD"][1])
    with pytest.raises(RuntimeError, match="AD failed its frozen identity gate"):
        block.materialize_cube(identities)

    corrupted = tmp_path / "preregistration.md"
    corrupted.write_bytes(resolve_corpus_path(block.PREREGISTRATION_PATH).read_bytes() + b"\n")
    sources = dict(block.EVIDENCE_SOURCES)
    sources["preregistration"] = (
        corrupted,
        block.EXPECTED_PREREGISTRATION_SHA256,
    )
    with pytest.raises(RuntimeError, match="preregistration"):
        block.verify_evidence_bundle(sources)

    corrupted_correction = tmp_path / "protocol_correction.md"
    corrupted_correction.write_bytes(resolve_corpus_path(block.CORRECTION_PATH).read_bytes() + b"\n")
    correction_sources = dict(block.EVIDENCE_SOURCES)
    correction_sources["protocol_correction"] = (
        corrupted_correction,
        block.EXPECTED_CORRECTION_SHA256,
    )
    with pytest.raises(RuntimeError, match="protocol_correction"):
        block.verify_evidence_bundle(correction_sources)

    forged_bundle = block.verify_evidence_bundle()
    forged_bundle["preregistration"]["sha256"] = "0" * 64
    forged_dir = tmp_path / "forged-bundle"
    forged_dir.mkdir()
    values = _writers(forged_dir)
    event_writer, evaluation_writer, trace, recorder, evaluator, predictor = values
    try:
        with pytest.raises(RuntimeError, match="bundle identity"):
            asyncio.run(
                block.run_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    predictor=predictor,
                    evidence_bundle=forged_bundle,
                )
            )
    finally:
        event_writer.close()
        evaluation_writer.close()
    assert evaluator.calls == []

    bad_dir = tmp_path / "bad-provenance"
    bad_dir.mkdir()
    values = _writers(bad_dir, {"bad_abc": True})
    event_writer, evaluation_writer, trace, recorder, evaluator, predictor = values
    try:
        with pytest.raises(RuntimeError, match="ABC provenance"):
            asyncio.run(
                block.run_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    predictor=predictor,
                    evidence_bundle=block.verify_evidence_bundle(),
                )
            )
    finally:
        event_writer.close()
        evaluation_writer.close()
    assert evaluator.calls == []


def test_latest_safe_start_and_terminal_seal(tmp_path: Path) -> None:
    class _DeadlineClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return 0 if self.calls == 1 else 241_000_000_000

    run_dir = tmp_path / "deadline"
    run_dir.mkdir()
    values = _writers(run_dir)
    event_writer, evaluation_writer, trace, recorder, evaluator, predictor = values
    try:
        with pytest.raises(RuntimeError, match="hard-deadline budget"):
            asyncio.run(
                block.run_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    predictor=predictor,
                    evidence_bundle=block.verify_evidence_bundle(),
                    clock_ns=_DeadlineClock(),
                )
            )
    finally:
        event_writer.close()
        evaluation_writer.close()
    assert evaluator.calls == ["C"]
    assert len(predictor.requests) == 1

    class _QualityHorizonClock:
        def __init__(self):
            self.values = iter((0, 130_000_000_000, 130_000_000_000))

        def __call__(self):
            return next(self.values)

    horizon_dir = tmp_path / "quality-horizon"
    horizon_dir.mkdir()
    horizon_evaluator, _, horizon_summary, _ = _run(
        horizon_dir,
        clock_ns=_QualityHorizonClock(),
    )
    assert set(horizon_evaluator.calls) == {"C", "AD", "BD", "ABD"}
    assert horizon_summary["protocol_acceptance_passed"] is True
    assert horizon_summary["resources"]["quality_horizon_met"] is False
    assert horizon_summary["resources"]["quality_horizon_failure"] is True
    assert horizon_summary["scientific_completeness"]["quality_horizon_met"] is False

    seal_dir = tmp_path / "seal"
    seal_dir.mkdir()
    (seal_dir / "preregistration.md").write_bytes(resolve_corpus_path(block.PREREGISTRATION_PATH).read_bytes())
    (seal_dir / "protocol_correction.md").write_bytes(
        resolve_corpus_path(block.CORRECTION_PATH).read_bytes()
    )
    (seal_dir / "runner_source.py").write_text("fixture\n", encoding="utf-8")
    (seal_dir / "events.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    block._finalize(seal_dir, "failed")
    finalized = json.loads((seal_dir / "finalized.json").read_text(encoding="utf-8"))
    assert finalized["status"] == "failed"
    assert finalized["preregistration_sha256"] == (
        block.EXPECTED_PREREGISTRATION_SHA256
    )
    assert finalized["protocol_correction_sha256"] == (
        block.EXPECTED_CORRECTION_SHA256
    )
    assert finalized["files"]["preregistration.md"]["sha256"] == (
        block.EXPECTED_PREREGISTRATION_SHA256
    )
    assert finalized["files"]["events.jsonl"]["lines"] == 1
    assert finalized["files"]["protocol_correction.md"]["sha256"] == (
        block.EXPECTED_CORRECTION_SHA256
    )

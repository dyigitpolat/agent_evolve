"""Offline gates for the engine-only BOiLS recombination continuation."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import threading
import time

import pytest

from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from examples.benchmarks.boils_abc.actions import config_sha256
from examples.benchmarks.boils_abc.evaluator import (
    AbcEvaluationError,
    BoilsEvaluation,
    BoilsEvaluationFailure,
    CircuitDiagnostics,
    CircuitEvaluation,
)
from examples.development import run_boils_recombination_engine_v4 as engine


NEW_OBJECTIVES = {
    "AD": (7_900, 68),
    "BD": (7_890, 70),
    "ABD": (7_860, 68),
}


def _diagnostics(cpu: int, *, status: str = "passed") -> CircuitDiagnostics:
    return CircuitDiagnostics(
        status=status,
        returncode=0 if status != "timeout" else None,
        elapsed_s=0.01,
        timeout_s=float(engine.PER_CANDIDATE_TIMEOUT_SECONDS),
        equivalent=status == "passed",
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
        invalid_arm: str | None = None,
        invalid_status: str = "timeout",
        bad_provenance: bool = False,
        duplicate_affinity: bool = False,
        omit_publication_arm: str | None = None,
        objectives=None,
    ) -> None:
        self._observer = observer
        self._invalid_arm = invalid_arm
        self._invalid_status = invalid_status
        self._bad_provenance = bad_provenance
        self._duplicate_affinity = duplicate_affinity
        self._omit_publication_arm = omit_publication_arm
        self._objectives = dict(NEW_OBJECTIVES if objectives is None else objectives)
        self._by_hash = {
            arm.boils_configuration_sha256: arm
            for arm in engine.CUBE
            if arm.label in engine.NEW_ARM_ORDER
        }
        self._barrier = threading.Barrier(engine.CHILD_WORKERS)
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0

    def provenance(self):
        return {
            "abc_binary_sha256": (
                "f" * 64 if self._bad_provenance else engine.EXPECTED_ABC_SHA256
            ),
            "circuits": [
                {"name": "log2", "sha256": engine.EXPECTED_CIRCUIT_SHA256}
            ],
            "lut_inputs": 6,
            "per_circuit_timeout_s": float(engine.PER_CANDIDATE_TIMEOUT_SECONDS),
            "affinity_sets": [[cpu] for cpu in engine.ENGINE_CPUS],
        }

    def evaluate(self, configuration):
        arm = self._by_hash[config_sha256(configuration)]
        label = arm.label
        with self._lock:
            self.calls.append(label)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self._barrier.wait(timeout=3.0)
        time.sleep(0.001 * (3 - engine.NEW_ARM_ORDER.index(label)))
        cpu = engine.ENGINE_CPUS[engine.NEW_ARM_ORDER.index(label)]
        if self._duplicate_affinity and label == "ABD":
            cpu = engine.ENGINE_CPUS[0]
        try:
            if label == self._invalid_arm:
                diagnostics = _diagnostics(cpu, status=self._invalid_status)
                failure = BoilsEvaluationFailure(
                    configuration_sha256=arm.boils_configuration_sha256,
                    sequence=arm.sequence,
                    abc_binary_sha256=engine.EXPECTED_ABC_SHA256,
                    failed_circuit_name="log2",
                    completed_circuit_results=(),
                    diagnostics=diagnostics,
                    elapsed_s=0.01,
                    affinity_queue_wait_s=0.001,
                    cpu_affinity=(cpu,),
                )
                if label != self._omit_publication_arm:
                    self._observer(failure)
                raise AbcEvaluationError("log2", diagnostics)
            lut_count, levels = self._objectives[label]
            diagnostics = _diagnostics(cpu)
            circuit = CircuitEvaluation(
                circuit_name="log2",
                circuit_sha256=engine.EXPECTED_CIRCUIT_SHA256,
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
                abc_binary_sha256=engine.EXPECTED_ABC_SHA256,
                lut_inputs=6,
                circuit_results=(circuit,),
                total_lut_count=lut_count,
                total_levels=levels,
                max_levels=levels,
                elapsed_s=0.01,
                affinity_queue_wait_s=0.001,
                cpu_affinity=(cpu,),
            )
            if label != self._omit_publication_arm:
                self._observer(result)
            return result
        finally:
            with self._lock:
                self.active -= 1


def _fixture_deferred_oracle() -> dict[str, object]:
    return engine.v3.verify_deferred_oracle_evidence()


def _fixture_admission() -> dict[str, object]:
    samples = iter(_proc_stat_sample(step) for step in range(4))
    return engine.sample_cpu_admission(
        reader=lambda: next(samples), sleeper=lambda _: None
    )


def _writers(tmp_path: Path, evaluator_options=None):
    event_writer = engine.v1.DurableJsonlWriter(tmp_path / "events.jsonl")
    evaluation_writer = engine.v1.DurableJsonlWriter(tmp_path / "evaluations.jsonl")
    trace = engine.v3.oracle.TraceRecorder(event_writer)
    recorder = engine.v3.oracle.EvaluationPublicationRecorder(
        evaluation_writer,
        trace,
        schedule=engine.CHILD_SCHEDULE,
    )
    evaluator = _OfflineEvaluator(recorder, **(evaluator_options or {}))
    return event_writer, evaluation_writer, trace, recorder, evaluator


def _run(
    tmp_path: Path,
    *,
    evaluator_options=None,
    clock_ns=time.perf_counter_ns,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    event_writer, evaluation_writer, trace, recorder, evaluator = _writers(
        tmp_path, evaluator_options
    )
    failed_v3 = engine.verify_failed_v3_bundle()
    preregistration = engine.verify_preregistration()
    deferred_calls: list[int] = []

    def deferred_loader():
        deferred_calls.append(len(recorder.records()))
        return _fixture_deferred_oracle()

    try:
        summary = asyncio.run(
            engine.run_engine_block(
                evaluator=evaluator,
                recorder=recorder,
                trace=trace,
                failed_v3=failed_v3,
                preregistration=preregistration,
                admission=_fixture_admission(),
                deferred_oracle_loader=deferred_loader,
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
    evaluations = [
        json.loads(line)
        for line in (tmp_path / "evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return evaluator, summary, events, evaluations, deferred_calls


def _proc_stat_sample(step: int, *, busy_cpu: int | None = None) -> str:
    rows = ["cpu  1 1 1 1 1 1 1 1 1 1"]
    for cpu in range(12):
        user_delta = 5
        idle_delta = 95
        if busy_cpu == cpu:
            user_delta = 20
            idle_delta = 80
        rows.append(
            f"cpu{cpu} {1000 + user_delta * step} 0 0 "
            f"{1000 + idle_delta * step} 0 0 0 0 0 0"
        )
    return "\n".join(rows) + "\n"


def _threshold_proc_stat_sample(step: int) -> str:
    rows = ["cpu  1 1 1 1 1 1"]
    for cpu in range(12):
        rows.append(f"cpu{cpu} {1000 + 10 * step} 0 0 {1000 + 90 * step} 0")
    return "\n".join(rows) + "\n"


def test_preregistration_cube_and_failed_v3_closure_are_exact() -> None:
    preregistration = engine.verify_preregistration()
    failed = engine.verify_failed_v3_bundle()

    assert preregistration == {
        "source": str(engine.PREREGISTRATION_PATH),
        "bytes": 7_638,
        "sha256": engine.EXPECTED_PREREGISTRATION_SHA256,
    }
    assert failed["status"] == "failed"
    assert failed["semantic_closure"]["physical_evaluations"] == 1
    assert failed["semantic_closure"]["only_arm"] == "C"
    assert failed["semantic_closure"]["prediction_completed"] is False
    assert failed["semantic_closure"]["new_child_submissions"] == 0
    assert failed["semantic_closure"]["provider_result_or_calibration_available"] is False
    assert failed["sealed_c"]["objectives"] == {
        "total_lut_count": 7_944,
        "total_levels": 69,
    }
    assert failed["unseen_child_scan"]["all_three_unseen"] is True
    assert tuple(arm.label for arm in engine.CUBE) == engine.ALL_ARM_ORDER
    assert tuple(spec.label for spec in engine.CHILD_SCHEDULE) == engine.NEW_ARM_ORDER
    for label in engine.NEW_ARM_ORDER:
        arm = engine.CUBE_BY_LABEL[label]
        assert config_sha256(arm.configuration) == arm.boils_configuration_sha256
        assert typed_json_sha256(freeze_json(arm.configuration)) == (
            arm.typed_json_configuration_sha256
        )
        assert arm.patch_record["replay_verified"] is True


def test_unseen_scan_rejects_any_prior_child_evaluation(tmp_path: Path) -> None:
    nested = tmp_path / "old_run"
    nested.mkdir()
    digest = engine.CUBE_BY_LABEL["AD"].boils_configuration_sha256
    (nested / "evaluations.jsonl").write_text(
        json.dumps({"candidate": {"boils_configuration_sha256": digest}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no longer an unseen"):
        engine.scan_unseen_children(tmp_path)


def test_proc_stat_parser_and_three_window_idle_admission() -> None:
    samples = iter(_proc_stat_sample(step) for step in range(4))
    sleeps: list[float] = []
    admission = engine.sample_cpu_admission(
        reader=lambda: next(samples),
        sleeper=lambda seconds: sleeps.append(seconds),
    )

    assert sleeps == [1.0, 1.0, 1.0]
    assert admission["selected_cpus"] == [8, 9, 11]
    assert len(admission["samples"]) == 4
    assert len(admission["windows"]) == 3
    assert all(
        row["busy_fraction"] == pytest.approx(0.05)
        for window in admission["windows"]
        for row in window["cpus"]
    )
    assert admission["passed"] is True


def test_exact_ten_percent_cpu_boundary_is_admitted() -> None:
    samples = iter(_threshold_proc_stat_sample(step) for step in range(4))
    admission = engine.sample_cpu_admission(
        reader=lambda: next(samples), sleeper=lambda _: None
    )
    assert all(
        row["busy_delta"] * 10 == row["total_delta"]
        for window in admission["windows"]
        for row in window["cpus"]
    )


def test_idle_admission_fails_without_creating_run_directory(tmp_path: Path) -> None:
    proposed = tmp_path / "must-not-exist"
    samples = iter(
        (
            _proc_stat_sample(0),
            _proc_stat_sample(1, busy_cpu=9),
            _proc_stat_sample(2),
            _proc_stat_sample(3),
        )
    )
    with pytest.raises(RuntimeError, match="cpu9 failed idle admission"):
        engine.prepare_launch_admission(
            run_dir=proposed,
            preregistration_loader=lambda: engine.verify_preregistration(),
            failed_v3_loader=lambda: engine.verify_failed_v3_bundle(),
            cpu_reader=lambda: next(samples),
            sleeper=lambda _: None,
        )
    assert not proposed.exists()


@pytest.mark.parametrize(
    "samples,match",
    [
        (["cpu8 1 2 3 4 5\n"], "missing selected CPUs"),
        (
            [_proc_stat_sample(0), _proc_stat_sample(0)],
            "total delta is not positive",
        ),
    ],
)
def test_idle_admission_rejects_malformed_or_zero_delta(samples, match) -> None:
    values = iter(samples)
    with pytest.raises(RuntimeError, match=match):
        engine.sample_cpu_admission(
            reader=lambda: next(values),
            sleeper=lambda _: None,
        )


def test_fixed_engine_wave_is_concurrent_persistent_and_model_free(tmp_path: Path) -> None:
    evaluator, summary, events, evaluations, deferred_calls = _run(tmp_path)

    assert len(evaluator.calls) == 3
    assert set(evaluator.calls) == set(engine.NEW_ARM_ORDER)
    assert evaluator.max_active == 3
    assert deferred_calls == [3]
    assert len(evaluations) == 3
    assert {row["candidate"]["label"] for row in evaluations} == set(
        engine.NEW_ARM_ORDER
    )
    submitted = [
        row["arm"] for row in events if row["event_type"] == "candidate_submitted"
    ]
    assert submitted == ["AD", "BD", "ABD"]
    durable_index = next(
        index
        for index, row in enumerate(events)
        if row["event_type"] == "fixed_child_wave_durable"
    )
    oracle_index = next(
        index
        for index, row in enumerate(events)
        if row["event_type"] == "sealed_local_oracle_verified"
    )
    assert durable_index < oracle_index

    assert summary["status"] == "succeeded"
    assert summary["protocol_acceptance_passed"] is True
    assert all(summary["protocol_gates"].values())
    assert summary["model"] == {
        "logical_llm_calls": 0,
        "result_reported": False,
        "calibration_reported": False,
        "reason": "engine-only continuation",
    }
    assert "model_rank_calibration" not in summary
    assert "model_categorical_calibration" not in summary
    assert summary["resources"]["physical_evaluations"] == 3
    assert summary["resources"]["retries"] == 0
    assert summary["resources"]["replacements"] == 0
    assert {tuple(value) for value in summary["resources"]["child_cpu_affinities"]} == {
        (8,),
        (9,),
        (11,),
    }


def test_run_block_revalidates_bound_inputs_and_cpu_admission(tmp_path: Path) -> None:
    event_writer, evaluation_writer, trace, recorder, evaluator = _writers(tmp_path)
    failed_v3 = engine.verify_failed_v3_bundle()
    failed_v3["files"]["failure.json"]["sha256"] = "0" * 64
    try:
        with pytest.raises(RuntimeError, match="not exactly bound"):
            asyncio.run(
                engine.run_engine_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    failed_v3=failed_v3,
                    preregistration=engine.verify_preregistration(),
                    admission=_fixture_admission(),
                )
            )
        assert evaluator.calls == []
    finally:
        event_writer.close()
        evaluation_writer.close()

    second = tmp_path / "admission"
    second.mkdir()
    event_writer, evaluation_writer, trace, recorder, evaluator = _writers(second)
    admission = _fixture_admission()
    admission["windows"][0]["cpus"][0]["busy_fraction"] = 0.2
    try:
        with pytest.raises(RuntimeError, match="admission row"):
            asyncio.run(
                engine.run_engine_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    failed_v3=engine.verify_failed_v3_bundle(),
                    preregistration=engine.verify_preregistration(),
                    admission=admission,
                )
            )
        assert evaluator.calls == []
    finally:
        event_writer.close()
        evaluation_writer.close()


def test_deferred_oracle_hash_binding_occurs_after_three_publications(
    tmp_path: Path,
) -> None:
    event_writer, evaluation_writer, trace, recorder, evaluator = _writers(tmp_path)

    def tampered_loader():
        assert len(recorder.records()) == 3
        record = _fixture_deferred_oracle()
        record["source_sha256"]["oracle_summary"] = "0" * 64
        return record

    try:
        with pytest.raises(RuntimeError, match="not exactly bound"):
            asyncio.run(
                engine.run_engine_block(
                    evaluator=evaluator,
                    recorder=recorder,
                    trace=trace,
                    failed_v3=engine.verify_failed_v3_bundle(),
                    preregistration=engine.verify_preregistration(),
                    admission=_fixture_admission(),
                    deferred_oracle_loader=tampered_loader,
                )
            )
        assert len(evaluator.calls) == 3
        assert len(recorder.records()) == 3
    finally:
        event_writer.close()
        evaluation_writer.close()


def test_exact_cube_analysis_interactions_hv_and_operator_rule(tmp_path: Path) -> None:
    _, summary, _, _, _ = _run(tmp_path)

    assert set(summary["pareto"]["preblock_front"]) == {"D", "AB"}
    assert summary["hypervolume"]["preblock"] == 213
    assert summary["interactions"]["I_AB"] == {
        "available": True,
        "missing_arms": [],
        "total_lut_count": -4,
        "total_levels": 1,
        "sign_interpretation": (
            "negative=favorable synergy; positive=antagonism for minimized objective"
        ),
    }
    assert summary["interactions"]["I_AD"]["total_lut_count"] == -16
    assert summary["interactions"]["I_AD"]["total_levels"] == -1
    assert summary["interactions"]["I_BD"]["total_lut_count"] == -22
    assert summary["interactions"]["I_BD"]["total_levels"] == 1
    assert summary["triple_prediction_arithmetic"]["available"] is True
    assert summary["triple_prediction_arithmetic"][
        "third_order_residual_matches_I_ABD"
    ] is True
    decisions = {
        row["arm"]: row for row in summary["pareto"]["new_arm_decisions"]
    }
    assert decisions["ABD"]["unique_objective_vector_on_combined_cube_front"] is True
    assert decisions["ABD"]["marginal_fixed_reference_hv_gain"] > 0
    assert decisions["ABD"]["contributes_search_value"] is True
    assert summary["decision"]["deterministic_disjoint_recombination_advances"] is True
    assert summary["decision"]["interaction_recording_advances"] is True
    assert summary["sealed_local_oracle_sensitivity"]["primary_decision_uses_this"] is False
    assert summary["sealed_local_oracle_sensitivity"]["preblock_hypervolume"] == 700


def test_candidate_local_timeout_is_consumed_as_partial_negative(tmp_path: Path) -> None:
    evaluator, summary, _, evaluations, deferred_calls = _run(
        tmp_path, evaluator_options={"invalid_arm": "BD", "invalid_status": "timeout"}
    )

    assert evaluator.calls.count("BD") == 1
    assert len(evaluations) == 3
    assert deferred_calls == [3]
    assert summary["status"] == "partial_candidate_local_invalid"
    assert summary["protocol_acceptance_passed"] is True
    assert summary["partial_negative_record"] == {
        "present": True,
        "invalid_arms": [
            {"arm": "BD", "candidate_local_failure_status": "timeout"}
        ],
        "fixed_arms_consumed_without_retry_or_replacement": True,
    }
    assert summary["interactions"]["I_BD"]["available"] is False
    assert summary["interactions"]["I_ABD"]["available"] is False
    assert summary["decision"]["interaction_recording_advances"] is False
    assert summary["resources"]["retries"] == 0
    assert summary["resources"]["replacements"] == 0


def test_mandatory_cec_failure_invalidates_block(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="mandatory CEC failed"):
        _run(
            tmp_path,
            evaluator_options={
                "invalid_arm": "AD",
                "invalid_status": "cec_failed_or_missing",
            },
        )


def test_wrong_affinity_or_provenance_invalidates_before_decision(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exact distinct affinity"):
        _run(tmp_path / "affinity", evaluator_options={"duplicate_affinity": True})
    with pytest.raises(RuntimeError, match="provenance gate"):
        _run(tmp_path / "provenance", evaluator_options={"bad_provenance": True})


def test_missing_durable_publication_invalidates_block(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="durable publication callback"):
        _run(
            tmp_path,
            evaluator_options={"omit_publication_arm": "ABD"},
        )


def test_hard_cleanup_deadline_is_infrastructure_fatal(tmp_path: Path) -> None:
    ticks = iter((0, 0, 0, 0, 0, 181_000_000_001))
    with pytest.raises(RuntimeError, match="hard cleanup deadline was exceeded"):
        _run(tmp_path, clock_ns=lambda: next(ticks))


def test_quality_horizon_failure_does_not_erase_valid_arm(tmp_path: Path) -> None:
    evaluator, summary, _, _, _ = _run(tmp_path)
    del evaluator
    child_rows = [
        copy.deepcopy(row)
        for row in summary["cube_outcomes"]
        if row["arm"] in engine.NEW_ARM_ORDER
    ]
    child_rows[0]["submission_elapsed_ns"] = 0
    child_rows[0]["published_elapsed_ns"] = 61_000_000_000
    rescored = engine.analyze_engine_cube(
        child_outcomes=[
            {
                "label": row["arm"],
                **{
                    key: value
                    for key, value in row.items()
                    if key
                    in {
                        "boils_configuration_sha256",
                        "typed_json_configuration_sha256",
                        "valid",
                        "cec_passed",
                        "candidate_local_failure_status",
                        "objectives",
                        "publication_sequence",
                        "published_elapsed_ns",
                        "submission_elapsed_ns",
                        "cpu_affinity",
                    }
                },
            }
            for row in child_rows
        ],
        sealed_c=summary["failed_v3_source"]["sealed_c"],
        deferred_oracle=summary["deferred_oracle_verification"],
        evaluator_provenance=summary["evaluator_provenance"],
        failed_v3=summary["failed_v3_source"],
        preregistration=summary["preregistration"],
        admission=summary["resource_admission"],
        started_ns=0,
        completed_ns=61_000_000_000,
    )
    assert rescored["resources"]["quality_horizon_met"] is False
    assert rescored["resources"]["quality_horizon_failure"] is True
    assert rescored["cube_outcomes"][5]["valid"] is True
    assert rescored["decision"]["deterministic_disjoint_recombination_advances"] is True
    assert rescored["protocol_acceptance_passed"] is True

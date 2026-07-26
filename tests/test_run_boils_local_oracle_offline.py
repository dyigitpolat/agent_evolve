"""Offline end-to-end gates for the frozen BOiLS local-oracle runner."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path
import shutil
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
from examples.development import run_boils_local_oracle as oracle


INVALID_IDENTITY = (12, 9)


def _diagnostics(*, status: str, equivalent: bool, cpu: int) -> CircuitDiagnostics:
    return CircuitDiagnostics(
        status=status,
        returncode=0 if status == "passed" else None,
        elapsed_s=0.01,
        timeout_s=float(oracle.PER_CANDIDATE_TIMEOUT_SECONDS),
        equivalent=equivalent,
        error_signatures=(),
        stdout_excerpt="offline fixture",
        stderr_excerpt="",
        stdout_sha256="0" * 64,
        stderr_sha256="0" * 64,
        abc_program="offline fixture",
        argv=("offline-abc",),
        cpu_affinity=(cpu,),
    )


def _fixture_objectives(spec: oracle.CandidateSpec) -> tuple[int, int]:
    if spec.index is None:
        return oracle.EXPECTED_PARENT_OBJECTIVES
    assert spec.legal_ordinal is not None
    ordinal = spec.legal_ordinal
    if spec.index == 1:
        return (7_943 - 2 * ordinal, 69)
    if spec.index == 7:
        return (7_945 - ordinal, 68 if ordinal >= 4 else 69)
    if spec.index == 12:
        return (7_939 + ordinal, 70 - ordinal % 3)
    if spec.index == 18:
        return (7_950 - ordinal, 67 + ordinal % 3)
    raise AssertionError("unexpected frozen index")


class _ConcurrentOfflineEvaluator:
    """Observer-faithful fake with a barrier in each fixed four-child round."""

    def __init__(self, observer) -> None:
        self._observer = observer
        self._by_hash = {
            spec.boils_configuration_sha256: spec for spec in oracle.FROZEN_SCHEDULE
        }
        self._barriers = {
            ordinal: threading.Barrier(oracle.WORKER_COUNT)
            for ordinal in range(oracle.CHILD_ROUNDS)
        }
        self._lock = threading.Lock()
        self.calls: list[tuple[int | None, int | None]] = []
        self.completed: list[tuple[int | None, int | None]] = []
        self.active = 0
        self.max_active = 0
        self.active_round: int | None = None
        self.round_started_after_previous_completed = True

    def _success(
        self,
        spec: oracle.CandidateSpec,
        *,
        cpu: int,
    ) -> BoilsEvaluation:
        lut_count, levels = _fixture_objectives(spec)
        diagnostics = _diagnostics(status="passed", equivalent=True, cpu=cpu)
        circuit = CircuitEvaluation(
            circuit_name="log2",
            circuit_sha256=oracle.EXPECTED_CIRCUIT_SHA256,
            inputs=32,
            outputs=32,
            lut_count=lut_count,
            edge_count=lut_count * 2,
            aig_count=lut_count * 3,
            levels=levels,
            diagnostics=diagnostics,
        )
        return BoilsEvaluation(
            configuration_sha256=spec.boils_configuration_sha256,
            sequence=spec.sequence,
            abc_binary_sha256=oracle.EXPECTED_ABC_SHA256,
            lut_inputs=6,
            circuit_results=(circuit,),
            total_lut_count=lut_count,
            total_levels=levels,
            max_levels=levels,
            elapsed_s=0.01,
            affinity_queue_wait_s=0.001,
            cpu_affinity=(cpu,),
        )

    def evaluate(self, config: object) -> BoilsEvaluation:
        digest = config_sha256(config)
        spec = self._by_hash[digest]
        if spec.index is None:
            with self._lock:
                assert not self.calls
                assert self.active == 0
                self.calls.append((None, None))
            result = self._success(spec, cpu=oracle.ORACLE_CPUS[0])
            self._observer(result)
            with self._lock:
                self.completed.append((None, None))
            return result

        assert spec.legal_ordinal is not None
        ordinal = spec.legal_ordinal
        position = oracle.MUTATION_INDICES.index(spec.index)
        cpu = oracle.ORACLE_CPUS[position]
        with self._lock:
            if self.active == 0:
                previous = sum(
                    completed_ordinal == ordinal - 1
                    for completed_ordinal, completed_index in self.completed
                    if completed_index is not None
                )
                if ordinal > 0 and previous != oracle.WORKER_COUNT:
                    self.round_started_after_previous_completed = False
                self.active_round = ordinal
            elif self.active_round != ordinal:
                self.round_started_after_previous_completed = False
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((ordinal, spec.index))
        self._barriers[ordinal].wait(timeout=3.0)
        # Deliberately make completion order differ from fixed submission order.
        time.sleep(0.001 * (oracle.WORKER_COUNT - position))
        try:
            if (spec.index, ordinal) == INVALID_IDENTITY:
                diagnostics = _diagnostics(
                    status="timeout",
                    equivalent=False,
                    cpu=cpu,
                )
                failure = BoilsEvaluationFailure(
                    configuration_sha256=spec.boils_configuration_sha256,
                    sequence=spec.sequence,
                    abc_binary_sha256=oracle.EXPECTED_ABC_SHA256,
                    failed_circuit_name="log2",
                    completed_circuit_results=(),
                    diagnostics=diagnostics,
                    elapsed_s=0.01,
                    affinity_queue_wait_s=0.001,
                    cpu_affinity=(cpu,),
                )
                self._observer(failure)
                raise AbcEvaluationError("log2", diagnostics)
            result = self._success(spec, cpu=cpu)
            self._observer(result)
            return result
        finally:
            with self._lock:
                self.active -= 1
                self.completed.append((ordinal, spec.index))
                if self.active == 0:
                    self.active_round = None


class _WrongSeedEvaluator(_ConcurrentOfflineEvaluator):
    def evaluate(self, config: object) -> BoilsEvaluation:
        digest = config_sha256(config)
        spec = self._by_hash[digest]
        if spec.index is not None:
            raise AssertionError("children must not run after a bad seed")
        with self._lock:
            self.calls.append((None, None))
        result = self._success(spec, cpu=oracle.ORACLE_CPUS[0])
        result = BoilsEvaluation(
            configuration_sha256=result.configuration_sha256,
            sequence=result.sequence,
            abc_binary_sha256=result.abc_binary_sha256,
            lut_inputs=result.lut_inputs,
            circuit_results=result.circuit_results,
            total_lut_count=result.total_lut_count + 1,
            total_levels=result.total_levels,
            max_levels=result.max_levels,
            elapsed_s=result.elapsed_s,
            affinity_queue_wait_s=result.affinity_queue_wait_s,
            cpu_affinity=result.cpu_affinity,
        )
        self._observer(result)
        return result


def _run(tmp_path: Path, evaluator_type=_ConcurrentOfflineEvaluator):
    event_writer = oracle.v1.DurableJsonlWriter(tmp_path / "events.jsonl")
    evaluation_writer = oracle.v1.DurableJsonlWriter(tmp_path / "evaluations.jsonl")
    trace = oracle.TraceRecorder(event_writer)
    recorder = oracle.EvaluationPublicationRecorder(evaluation_writer, trace)
    evaluator = evaluator_type(recorder)
    try:
        summary = oracle.run_oracle(
            evaluator=evaluator,
            recorder=recorder,
            trace=trace,
            v2_choices=oracle.load_sealed_v2_choices(),
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
    return evaluator, summary, events, evaluations


def _independent_hypervolume(points: list[tuple[int, int]]) -> int:
    reference = oracle.REFERENCE_POINT
    area = 0
    for x_value in range(min(point[0] for point in points), reference[0]):
        covered_y = [
            point[1]
            for point in points
            if point[0] <= x_value and point[1] < reference[1]
        ]
        if covered_y:
            area += reference[1] - min(covered_y)
    return area


def _strictly_dominates(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left != right


def test_frozen_schedule_uses_every_legal_identity_once_in_round_order() -> None:
    assert oracle._sha256(oracle.PREREGISTRATION_PATH) == (
        oracle.EXPECTED_PREREGISTRATION_SHA256
    )
    assert len(oracle.FROZEN_SCHEDULE) == 41
    assert oracle.FROZEN_SCHEDULE[0].configuration == oracle.v2.PARENT_C
    assert oracle.FROZEN_SCHEDULE[0].boils_configuration_sha256 == (
        oracle.EXPECTED_PARENT_BOILS_SHA256
    )
    seen: set[str] = set()
    for round_index in range(oracle.CHILD_ROUNDS):
        block = oracle.FROZEN_SCHEDULE[
            1 + round_index * 4 : 1 + (round_index + 1) * 4
        ]
        assert tuple(spec.index for spec in block) == oracle.MUTATION_INDICES
        assert all(spec.legal_ordinal == round_index for spec in block)
        for spec in block:
            assert config_sha256(spec.configuration) == spec.boils_configuration_sha256
            assert typed_json_sha256(freeze_json(spec.configuration)) == (
                spec.typed_json_configuration_sha256
            )
            source_row = oracle.v2.LEGAL_CHILD_UNIVERSE["indices"][str(spec.index)][
                "legal_children"
            ][round_index]
            assert spec.replacement == source_row["replacement"]
            assert spec.boils_configuration_sha256 == source_row[
                "boils_configuration_sha256"
            ]
            seen.add(spec.boils_configuration_sha256)
    assert len(seen) == 40

    choices = oracle.load_sealed_v2_choices()
    assert tuple(choice.index for choice in choices) == oracle.MUTATION_INDICES
    assert tuple(choice.replacement for choice in choices) == (
        "refactor_z",
        "resub",
        "rewrite_z",
        "rewrite",
    )


def test_offline_oracle_exercises_rounds_metrics_and_exact_distribution(
    tmp_path: Path,
) -> None:
    evaluator, summary, events, evaluations = _run(tmp_path)

    assert evaluator.max_active == 4
    assert evaluator.round_started_after_previous_completed is True
    assert len(evaluator.calls) == 41
    assert evaluator.calls[0] == (None, None)
    assert len(evaluator.completed) == 41
    assert summary["status"] == "succeeded"
    assert summary["schedule"] == {
        "seed_alone_and_verified_before_children": True,
        "child_rounds": 10,
        "children_per_round": 4,
        "fixed_index_order": [1, 7, 12, 18],
        "physical_evaluations": 41,
        "empty_cache": True,
        "retries": 0,
        "replacements_after_outcomes": 0,
    }

    outcomes = summary["outcomes_frozen_order"]
    assert len(outcomes) == 41
    assert [outcome["frozen_order"] for outcome in outcomes] == list(range(41))
    for spec, outcome in zip(oracle.FROZEN_SCHEDULE, outcomes, strict=True):
        assert outcome["boils_configuration_sha256"] == (
            spec.boils_configuration_sha256
        )
        assert outcome["typed_json_configuration_sha256"] == (
            spec.typed_json_configuration_sha256
        )
    invalid = summary["invalid_outcomes"]
    assert len(invalid) == 1
    assert (invalid[0]["index"], invalid[0]["legal_ordinal"]) == INVALID_IDENTITY
    assert invalid[0]["candidate_local_failure_status"] == "timeout"

    assert len(evaluations) == 41
    assert [record["publication_sequence"] for record in evaluations] == list(
        range(1, 42)
    )
    assert Counter(record["status"] for record in evaluations) == {
        "succeeded": 40,
        "candidate_local_failure": 1,
    }
    submissions = [
        event for event in events if event["event_type"] == "candidate_submitted"
    ]
    assert [event["frozen_order"] for event in submissions] == list(range(41))
    for round_index in range(10):
        block = submissions[1 + round_index * 4 : 1 + (round_index + 1) * 4]
        assert [event["index"] for event in block] == [1, 7, 12, 18]
        assert all(event["legal_ordinal"] == round_index for event in block)

    valid = [outcome for outcome in outcomes if outcome["valid"]]
    independent_front = [
        outcome
        for outcome in valid
        if not any(
            other is not outcome
            and _strictly_dominates(
                (
                    other["objectives"]["total_lut_count"],
                    other["objectives"]["total_levels"],
                ),
                (
                    outcome["objectives"]["total_lut_count"],
                    outcome["objectives"]["total_levels"],
                ),
            )
            for other in valid
        )
    ]
    assert {
        row["boils_configuration_sha256"] for row in summary["pareto_front"]
    } == {row["boils_configuration_sha256"] for row in independent_front}

    points = [
        (
            outcome["objectives"]["total_lut_count"],
            outcome["objectives"]["total_levels"],
        )
        for outcome in valid
    ]
    expected_oracle_hv = _independent_hypervolume(points)
    hv = summary["hypervolume"]
    assert hv["parent_c"] == 168
    assert hv["terminal_local_oracle"] == expected_oracle_hv
    assert len(hv["hv_at_k_frozen_order"]) == 41
    assert hv["hv_at_k_frozen_order"][0] == {"k": 1, "hypervolume": 168}
    assert hv["hv_at_k_frozen_order"][-1]["hypervolume"] == expected_oracle_hv
    wall = hv["wall_clock_auc"]
    assert wall["publications_within_horizon"] == 41
    assert wall["terminal_hypervolume_at_horizon"] == expected_oracle_hv
    assert 0 < wall["mean_hypervolume"] <= expected_oracle_hv

    by_path = {
        index: [outcome for outcome in outcomes if outcome["index"] == index]
        for index in oracle.MUTATION_INDICES
    }
    expected_hvs: list[int] = []
    expected_strict = Counter()
    v2_rows = {
        row["sealed_choice"]["index"]: row
        for row in summary["v2"]["path_conditional"]
    }
    v2_targets = {
        index: (
            v2_rows[index]["oracle_reevaluation"]["objectives"]["total_lut_count"],
            v2_rows[index]["oracle_reevaluation"]["objectives"]["total_levels"],
        )
        for index in oracle.MUTATION_INDICES
    }
    parent_point = oracle.EXPECTED_PARENT_OBJECTIVES
    for ordinals in itertools.product(range(10), repeat=4):
        selected = [
            by_path[index][ordinal]
            for index, ordinal in zip(oracle.MUTATION_INDICES, ordinals, strict=True)
        ]
        policy_points = [
            parent_point,
            *(
                (
                    outcome["objectives"]["total_lut_count"],
                    outcome["objectives"]["total_levels"],
                )
                for outcome in selected
                if outcome["valid"]
            ),
        ]
        expected_hvs.append(_independent_hypervolume(policy_points))
        for index, target in v2_targets.items():
            expected_strict[index] += any(
                _strictly_dominates(point, target) for point in policy_points
            )

    distribution = summary["exact_random_policy_distribution"]
    assert distribution["policy_count"] == 10_000
    support = Counter(expected_hvs)
    reported_support = {
        row["hypervolume"]: row["count"]
        for row in distribution["hypervolume"]["complete_support"]
    }
    assert reported_support == dict(sorted(support.items()))
    expected_mean = Fraction(sum(expected_hvs), 10_000)
    assert distribution["hypervolume"]["mean"]["fraction"] == (
        f"{expected_mean.numerator}/{expected_mean.denominator}"
    )
    sealed_threshold = summary["v2"]["sealed_terminal_hypervolume"]
    matching = sum(value >= sealed_threshold for value in expected_hvs)
    assert distribution["comparison_to_sealed_v2"]["matching_or_exceeding"][
        "count"
    ] == matching
    for index in oracle.MUTATION_INDICES:
        assert distribution[
            "probability_policy_archive_strictly_dominates_v2_child"
        ][str(index)]["count"] == expected_strict[index]
    assert distribution["dominance_probability_archive_includes_parent_c"] is True

    for row in summary["v2"]["path_conditional"]:
        assert "path_pareto_status_among_ten_children" in row
        assert "path_pareto_status_with_parent_c" in row
        assert "path_pareto_layer_rank_among_ten_children" in row
        assert "path_pareto_layer_rank_with_parent_c" in row

    resources = summary["resources"]
    assert resources["physical_evaluations"] == 41
    assert resources["valid_evaluations"] == 40
    assert resources["candidate_local_invalids"] == 1
    assert resources["cec_passed_valid_evaluations"] == 40
    assert resources["affinity_publication_counts"] == {
        "[10]": 10,
        "[11]": 10,
        "[8]": 11,
        "[9]": 10,
    }


def test_seed_objective_mismatch_aborts_before_children(tmp_path: Path) -> None:
    event_writer = oracle.v1.DurableJsonlWriter(tmp_path / "events.jsonl")
    evaluation_writer = oracle.v1.DurableJsonlWriter(tmp_path / "evaluations.jsonl")
    trace = oracle.TraceRecorder(event_writer)
    recorder = oracle.EvaluationPublicationRecorder(evaluation_writer, trace)
    evaluator = _WrongSeedEvaluator(recorder)
    try:
        with pytest.raises(oracle.SeedGateError):
            oracle.run_oracle(
                evaluator=evaluator,
                recorder=recorder,
                trace=trace,
                v2_choices=oracle.load_sealed_v2_choices(),
            )
    finally:
        event_writer.close()
        evaluation_writer.close()
    assert evaluator.calls == [(None, None)]
    assert len(
        (tmp_path / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 1


def test_sealed_v2_terminal_hash_gate_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "sealed-v2"
    shutil.copytree(oracle.V2_RUN_DIR, copied)
    with (copied / "summary.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(RuntimeError, match="terminal hash/size mismatch"):
        oracle.load_sealed_v2_choices(copied)

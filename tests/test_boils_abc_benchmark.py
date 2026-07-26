"""Focused offline tests for the production BOiLS/ABC example adapter."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_evolve.core.problem import normalize_objective_values
from examples.benchmarks.boils_abc.actions import (
    ACTION_COMMANDS,
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    SEQUENCE_LENGTH,
    CandidateConfig,
    canonical_config_bytes,
    config_sha256,
    expand_abc_commands,
)
from examples.benchmarks.boils_abc.evaluator import (
    ABC_SOURCE_COMMIT,
    EPFL_SOURCE_COMMIT,
    LUT_INPUTS,
    AbcEvaluationError,
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
    BoilsEvaluationFailure,
    CircuitSpec,
    ProvenanceMismatchError,
    parse_abc_output,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_executable(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_abc(path: Path) -> Path:
    return _make_executable(
        path,
        """#!/usr/bin/env python3
from pathlib import Path
import sys

program = sys.argv[2]
assert sys.argv[1] == "-c"
assert Path("source.blif").is_file()
lut_count, levels = (int(value) for value in Path("source.blif").read_text().split())
Path("mapped.blif").write_text("mapped")
print("PROGRAM=" + program)
print(f"top : i/o = 8/ 4 nd = {lut_count} edge = 19 aig = 11 lev = {levels}")
print("Networks are equivalent.")
""",
    )


def _fake_taskset(path: Path) -> Path:
    return _make_executable(
        path,
        """#!/usr/bin/env python3
import os
import sys

assert sys.argv[1] == "--cpu-list"
os.execv(sys.argv[3], sys.argv[3:])
""",
    )


def _settings(
    tmp_path: Path,
    *,
    affinity_sets: tuple[tuple[int, ...], ...] = (),
) -> AbcEvaluatorSettings:
    # Delimiters and whitespace in filesystem paths are deliberate.  They must
    # never be interpolated into the ABC command language.
    binary = _fake_abc(tmp_path / "bin; unsafe" / "fake abc")
    first = tmp_path / "inputs" / "one; quit.blif"
    second = tmp_path / "inputs" / "two with spaces.blif"
    first.parent.mkdir(parents=True)
    first.write_text("5 7", encoding="utf-8")
    second.write_text("11 13", encoding="utf-8")
    taskset = _fake_taskset(tmp_path / "launcher with spaces")
    return AbcEvaluatorSettings(
        abc_binary=binary,
        expected_abc_sha256=_digest(binary),
        circuits=(
            CircuitSpec("first", first, _digest(first)),
            CircuitSpec("second", second, _digest(second)),
        ),
        per_circuit_timeout_s=5.0,
        work_root=tmp_path / "work root; unsafe",
        affinity_sets=affinity_sets,
        taskset_binary=taskset,
        max_diagnostic_chars=4_096,
    )


def test_candidate_is_strict_exact_length_and_defaults_to_twenty_actions():
    candidate = CandidateConfig()
    assert len(candidate.sequence) == SEQUENCE_LENGTH == 20
    assert tuple(candidate.sequence) == DEFAULT_ACTION_SEQUENCE
    assert tuple(ACTION_COMMANDS) == ACTION_IDS

    invalid = (
        {"sequence": ["balance"] * 19},
        {"sequence": ["balance"] * 21},
        {"sequence": ["balance"] * 19 + ["balance; quit"]},
        {"sequence": ["balance"] * 19 + ["rewrite -z"]},
        {"sequence": ["balance"] * 19 + [1]},
        {"sequence": ["balance"] * 20, "abc_command": "quit"},
        {"sequence": tuple(["balance"] * 20)},
    )
    for value in invalid:
        with pytest.raises(ValidationError):
            CandidateConfig.model_validate(value)


def test_composite_actions_expand_only_through_the_allowlist():
    sequence = ["balance"] * 17 + ["sopb", "blut", "dsdb"]
    commands = expand_abc_commands({"sequence": sequence})
    assert commands[-9:] == (
        "&get -n",
        "&sopb",
        "&put",
        "&get -n",
        "&blut",
        "&put",
        "&get -n",
        "&dsdb",
        "&put",
    )


def test_configuration_hash_is_canonical_schema_scoped_and_order_sensitive():
    mapping = {"sequence": list(DEFAULT_ACTION_SEQUENCE)}
    model = CandidateConfig.model_validate(mapping)
    assert config_sha256(mapping) == config_sha256(model)
    assert canonical_config_bytes(mapping) == canonical_config_bytes(model)

    changed = {"sequence": list(reversed(DEFAULT_ACTION_SEQUENCE))}
    assert config_sha256(mapping) != config_sha256(changed)
    assert len(config_sha256(mapping)) == 64


def test_parser_rejects_zero_exit_errors_ambiguous_stats_and_missing_cec():
    stats = "top: i/o = 8/ 4 nd = 5 edge = 19 aig = 11 lev = 7"
    passed = parse_abc_output(stats, "Networks are equivalent.", 0)
    assert passed.status == "passed"
    assert passed.stats == {
        "inputs": 8,
        "outputs": 4,
        "lut_count": 5,
        "edge_count": 19,
        "aig_count": 11,
        "levels": 7,
    }
    assert parse_abc_output(stats, "unknown command 'oops'", 0).status == (
        "abc_reported_error"
    )
    assert parse_abc_output(stats + "\n" + stats, "Networks are equivalent.", 0).status == (
        "stats_missing_or_ambiguous"
    )
    assert parse_abc_output(stats, "", 0).status == "cec_failed_or_missing"
    assert parse_abc_output(stats, "Networks are equivalent.", 2).status == (
        "abc_nonzero_exit"
    )


def test_evaluator_aggregates_raw_results_and_closes_path_injection_boundary(
    tmp_path: Path,
):
    settings = _settings(tmp_path, affinity_sets=((3, 4),))
    observed: list[BoilsEvaluation | BoilsEvaluationFailure] = []

    def observe(item: BoilsEvaluation | BoilsEvaluationFailure) -> None:
        assert settings.work_root is not None
        assert list(settings.work_root.iterdir()) == []
        observed.append(item)

    evaluator = BoilsAbcEvaluator(settings, observer=observe)
    result = evaluator.evaluate(CandidateConfig())

    assert observed == [result]
    assert observed[0] is result
    assert result.lut_inputs == LUT_INPUTS == 6
    assert result.total_lut_count == 16
    assert result.total_levels == 20
    assert result.max_levels == 13
    assert result.cpu_affinity == (3, 4)
    assert result.elapsed_s > 0
    assert result.affinity_queue_wait_s >= 0
    assert result.objective_values == {
        "total_lut_count": 16.0,
        "total_levels": 20.0,
    }
    assert [item.circuit_name for item in result.circuit_results] == [
        "first",
        "second",
    ]
    for circuit in result.circuit_results:
        diagnostics = circuit.diagnostics
        assert diagnostics.status == "passed"
        assert diagnostics.equivalent is True
        assert diagnostics.cpu_affinity == (3, 4)
        assert diagnostics.argv[1:3] == ("--cpu-list", "3,4")
        assert "read source.blif" in diagnostics.abc_program
        assert "write_blif mapped.blif" in diagnostics.abc_program
        assert "if -K 6" in diagnostics.abc_program
        assert "one; quit" not in diagnostics.abc_program
        assert "two with spaces" not in diagnostics.abc_program
        assert str(settings.work_root) not in diagnostics.abc_program

    # Temporary workspaces are deleted even though the stable work root is
    # retained for subsequent concurrent calls.
    assert settings.work_root is not None
    assert list(settings.work_root.iterdir()) == []
    assert evaluator.concurrency_capacity == 1
    provenance = evaluator.provenance()
    assert provenance["abc_binary_sha256"] == settings.expected_abc_sha256
    assert provenance["abc_source_identity"] is None
    assert provenance["circuit_suite_identity"] is None
    assert provenance["taskset_binary_sha256"] == _digest(settings.taskset_binary)


def test_async_boundary_and_problem_projection_keep_diagnostics_out_of_objectives(
    tmp_path: Path,
):
    settings = _settings(tmp_path)
    evaluator = BoilsAbcEvaluator(settings)
    detailed = asyncio.run(evaluator.evaluate_async(CandidateConfig()))
    assert detailed.total_lut_count == 16
    assert evaluator.concurrency_capacity is None

    problem = BoilsAbcProblem(settings, evaluator=evaluator)
    objective_values = problem.evaluate(CandidateConfig())
    assert objective_values == {
        "total_lut_count": 16.0,
        "total_levels": 20.0,
    }
    assert normalize_objective_values(objective_values, problem.objectives) == (
        objective_values
    )
    assert problem.validate(CandidateConfig()) is True
    assert problem.candidate_key(CandidateConfig()) == config_sha256(CandidateConfig())
    assert "stdout" not in objective_values
    assert "LUT-6" in problem.search_space_description()


def test_binary_and_circuit_provenance_fail_closed(tmp_path: Path):
    settings = _settings(tmp_path)
    bad_binary = AbcEvaluatorSettings(
        abc_binary=settings.abc_binary,
        expected_abc_sha256="0" * 64,
        circuits=settings.circuits,
        work_root=settings.work_root,
    )
    with pytest.raises(ProvenanceMismatchError, match="ABC binary SHA-256 mismatch"):
        BoilsAbcEvaluator(bad_binary)

    evaluator = BoilsAbcEvaluator(settings)
    settings.circuits[0].source.write_text("17 19", encoding="utf-8")
    with pytest.raises(ProvenanceMismatchError, match="changed after"):
        evaluator.evaluate(CandidateConfig())


def test_subprocess_timeout_is_bounded_and_retains_structured_diagnostics(
    tmp_path: Path,
):
    binary = _make_executable(
        tmp_path / "slow abc",
        """#!/usr/bin/env python3
import time
time.sleep(5)
""",
    )
    circuit = tmp_path / "circuit.blif"
    circuit.write_text("5 7", encoding="utf-8")
    work_root = tmp_path / "work"
    settings = AbcEvaluatorSettings(
        abc_binary=binary,
        expected_abc_sha256=_digest(binary),
        circuits=(CircuitSpec("slow", circuit, _digest(circuit)),),
        per_circuit_timeout_s=0.05,
        work_root=work_root,
    )
    observed: list[BoilsEvaluation | BoilsEvaluationFailure] = []

    def observe(item: BoilsEvaluation | BoilsEvaluationFailure) -> None:
        assert list(work_root.iterdir()) == []
        observed.append(item)

    evaluator = BoilsAbcEvaluator(settings, observer=observe)
    with pytest.raises(AbcEvaluationError) as caught:
        evaluator.evaluate(CandidateConfig())
    diagnostics = caught.value.diagnostics
    assert caught.value.circuit_name == "slow"
    assert diagnostics.status == "timeout"
    assert diagnostics.returncode is None
    assert diagnostics.equivalent is False
    assert diagnostics.elapsed_s < 1.0
    assert list(work_root.iterdir()) == []
    assert len(observed) == 1
    failure = observed[0]
    assert isinstance(failure, BoilsEvaluationFailure)
    assert failure.failed_circuit_name == "slow"
    assert failure.diagnostics is diagnostics
    assert failure.completed_circuit_results == ()
    assert failure.configuration_sha256 == config_sha256(CandidateConfig())
    assert failure.elapsed_s >= diagnostics.elapsed_s


def test_affinity_declarations_are_nonempty_disjoint_exact_cpu_sets(tmp_path: Path):
    base = _settings(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        AbcEvaluatorSettings(
            abc_binary=base.abc_binary,
            expected_abc_sha256=base.expected_abc_sha256,
            circuits=base.circuits,
            affinity_sets=((1, 2), (2, 3)),
        )
    with pytest.raises(ValueError, match="non-negative exact integers"):
        AbcEvaluatorSettings(
            abc_binary=base.abc_binary,
            expected_abc_sha256=base.expected_abc_sha256,
            circuits=base.circuits,
            affinity_sets=((True,),),
        )


def test_default_declaration_labels_only_the_qualified_local_regime():
    settings = AbcEvaluatorSettings.current_four_circuit(
        affinity_sets=((8,),),
    )
    assert settings.abc_source_identity == f"git:{ABC_SOURCE_COMMIT}"
    assert settings.circuit_suite_identity == f"git:{EPFL_SOURCE_COMMIT}"
    assert tuple(item.name for item in settings.circuits) == (
        "log2",
        "multiplier",
        "sin",
        "sqrt",
    )


def test_current_panel_subset_is_explicit_ordered_and_closed():
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("multiplier", "sin", "sqrt"),
        affinity_sets=((8,), (9,)),
    )
    assert tuple(item.name for item in settings.circuits) == (
        "multiplier",
        "sin",
        "sqrt",
    )
    assert settings.affinity_sets == ((8,), (9,))
    with pytest.raises(ValueError, match="duplicates"):
        AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=("sin", "sin")
        )
    with pytest.raises(ValueError, match="unknown"):
        AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=("adder",)
        )

"""Offline tests for the development-only DAG assignment/ordering problem."""

from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from agent_evolve.core.problem import normalize_objective_values
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import ObjectKey
from agent_evolve.domain.typed_json import typed_json_equal
from agent_evolve.policies.variation.typed_patch import (
    ThreeWayRelationKind,
    apply_patch,
    classify_three_way_patches,
    derive_patch,
)

_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "development"
    / "dag_dispatch_codesign"
    / "problem_def.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "development_dag_dispatch_codesign_problem_def",
    _SOURCE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BASE_CONFIG = _MODULE.BASE_CONFIG
CandidateConfig = _MODULE.CandidateConfig
DEPENDENCY_EDGES = _MODULE.DEPENDENCY_EDGES
DEVELOPMENT_BRANCH_LEFT = _MODULE.DEVELOPMENT_BRANCH_LEFT
DEVELOPMENT_BRANCH_RIGHT = _MODULE.DEVELOPMENT_BRANCH_RIGHT
DEVELOPMENT_ONLY_NOTICE = _MODULE.DEVELOPMENT_ONLY_NOTICE
DEVELOPMENT_RECOMBINATION_TARGET = _MODULE.DEVELOPMENT_RECOMBINATION_TARGET
TASKS = _MODULE.TASKS
problem = _MODULE.problem


BASE_ID = CandidateId("candidate_dev_dag_base")
LEFT_ID = CandidateId("candidate_dev_dag_left")
RIGHT_ID = CandidateId("candidate_dev_dag_right")
TARGET_ID = CandidateId("candidate_dev_dag_target")


def _dump(config: object) -> dict[str, object]:
    return CandidateConfig.model_validate(config).model_dump(mode="json")


def _root_path_names(patch) -> tuple[str, ...]:
    roots = []
    for operation in patch.operations:
        first = operation.path.segments[0]
        assert type(first) is ObjectKey
        roots.append(first.value)
    return tuple(roots)


def test_frozen_development_cohort_is_strict_valid_and_explicitly_nonbenchmark():
    for config in (
        BASE_CONFIG,
        DEVELOPMENT_BRANCH_LEFT,
        DEVELOPMENT_BRANCH_RIGHT,
        DEVELOPMENT_RECOMBINATION_TARGET,
    ):
        parsed = CandidateConfig.model_validate(config)
        assert type(parsed) is CandidateConfig
        assert problem.validate(config) is True

    assert "not a benchmark" in DEVELOPMENT_ONLY_NOTICE
    description = problem.search_space_description().lower()
    assert "development" in description
    assert "not a benchmark" in description
    assert "not measured wall-clock" in description


def test_candidate_model_forbids_coercion_unknown_fields_and_non_list_shapes():
    cases = []

    extra_root = deepcopy(BASE_CONFIG)
    extra_root["hidden"] = "forbidden"
    cases.append(extra_root)

    extra_assignment = deepcopy(BASE_CONFIG)
    extra_assignment["assignments"][0]["hidden"] = "forbidden"
    cases.append(extra_assignment)

    coerced_worker = deepcopy(BASE_CONFIG)
    coerced_worker["assignments"][0]["worker"] = 1
    cases.append(coerced_worker)

    tuple_order = deepcopy(BASE_CONFIG)
    tuple_order["dispatch_order"] = tuple(tuple_order["dispatch_order"])
    cases.append(tuple_order)

    for config in cases:
        with pytest.raises(ValidationError):
            CandidateConfig.model_validate(config)

    # Frozen models still contain a list object; the consuming boundary must
    # revalidate an instance rather than trusting its constructor history.
    mutated_instance = CandidateConfig.model_validate(BASE_CONFIG)
    mutated_instance.dispatch_order.append("acquire")
    with pytest.raises(ValidationError):
        problem.validate(mutated_instance)


def test_hard_graph_assignment_and_resource_constraints_fail_closed():
    noncanonical_assignments = deepcopy(BASE_CONFIG)
    noncanonical_assignments["assignments"][0:2] = reversed(
        noncanonical_assignments["assignments"][0:2]
    )

    duplicate_order = deepcopy(BASE_CONFIG)
    duplicate_order["dispatch_order"][-1] = "acquire"

    dependency_violation = deepcopy(BASE_CONFIG)
    dependency_violation["dispatch_order"] = [
        "acquire",
        "compress",
        "encode",
        "inspect",
        "classify",
        "audit",
        "package",
        "release",
    ]

    incompatible_worker = deepcopy(BASE_CONFIG)
    incompatible_worker["assignments"][0]["worker"] = "gpu"

    accelerator_overflow = deepcopy(DEVELOPMENT_BRANCH_LEFT)
    accelerator_overflow["assignments"][2]["worker"] = "gpu"

    worker_overflow = deepcopy(BASE_CONFIG)
    for assignment in worker_overflow["assignments"]:
        assignment["worker"] = "cpu_a"

    excessive_crossings = deepcopy(BASE_CONFIG)
    workers = (
        "cpu_a",
        "gpu",
        "cpu_b",
        "cpu_a",
        "npu",
        "cpu_a",
        "cpu_b",
        "cpu_a",
    )
    for assignment, worker in zip(
        excessive_crossings["assignments"],
        workers,
        strict=True,
    ):
        assignment["worker"] = worker

    expected_messages = (
        (noncanonical_assignments, "canonical TASKS order"),
        (duplicate_order, "exact permutation"),
        (dependency_violation, "violates dependency encode->compress"),
        (incompatible_worker, "cannot run on worker"),
        (accelerator_overflow, "working set"),
        (worker_overflow, "task limit"),
        (excessive_crossings, "cross-worker"),
    )
    for config, message in expected_messages:
        with pytest.raises(ValueError, match=message):
            problem.validate(config)


def test_evaluator_is_deterministic_finite_and_objective_complete():
    objective_names = tuple(spec.name for spec in problem.objectives)
    assert objective_names == (
        "makespan_ms",
        "energy_mj",
        "peak_worker_load_ms",
    )
    assert tuple(spec.goal for spec in problem.objectives) == ("min", "min", "min")

    for config in (
        BASE_CONFIG,
        DEVELOPMENT_BRANCH_LEFT,
        DEVELOPMENT_BRANCH_RIGHT,
        DEVELOPMENT_RECOMBINATION_TARGET,
    ):
        first = problem.evaluate(config)
        second = problem.evaluate(deepcopy(config))
        assert first == second
        assert tuple(first) == objective_names
        assert all(type(value) is float and math.isfinite(value) for value in first.values())
        assert normalize_objective_values(first, problem.objectives) == first


def test_branch_patches_are_disjoint_replayable_and_target_is_their_union():
    base = _dump(BASE_CONFIG)
    left = _dump(DEVELOPMENT_BRANCH_LEFT)
    right = _dump(DEVELOPMENT_BRANCH_RIGHT)
    target = _dump(DEVELOPMENT_RECOMBINATION_TARGET)

    left_patch = derive_patch(
        base,
        left,
        base_candidate_id=BASE_ID,
        target_candidate_id=LEFT_ID,
    )
    right_patch = derive_patch(
        base,
        right,
        base_candidate_id=BASE_ID,
        target_candidate_id=RIGHT_ID,
    )
    target_patch = derive_patch(
        base,
        target,
        base_candidate_id=BASE_ID,
        target_candidate_id=TARGET_ID,
    )

    assert typed_json_equal(apply_patch(base, left_patch), left)
    assert typed_json_equal(apply_patch(base, right_patch), right)
    assert typed_json_equal(apply_patch(base, target_patch), target)

    assert _root_path_names(left_patch) == ("assignments",) * 3
    assert _root_path_names(right_patch) == ("dispatch_order",)
    assert sorted(_root_path_names(target_patch)) == [
        "assignments",
        "assignments",
        "assignments",
        "dispatch_order",
    ]

    classification = classify_three_way_patches(base, left_patch, right_patch)
    classification.revalidate()
    assert classification.relations
    assert all(
        relation.kind is ThreeWayRelationKind.DISJOINT
        for relation in classification.relations
    )
    assert classification.of_kind(ThreeWayRelationKind.CONFLICT) == ()
    assert all(
        any(operation == target_operation for target_operation in target_patch.operations)
        for operation in left_patch.operations + right_patch.operations
    )


def test_recombination_preserves_both_branches_and_adds_nonlinear_value():
    base = _dump(BASE_CONFIG)
    left = _dump(DEVELOPMENT_BRANCH_LEFT)
    right = _dump(DEVELOPMENT_BRANCH_RIGHT)
    target = _dump(DEVELOPMENT_RECOMBINATION_TARGET)

    assert left["dispatch_order"] == base["dispatch_order"]
    assert right["assignments"] == base["assignments"]
    assert target["assignments"] == left["assignments"]
    assert target["dispatch_order"] == right["dispatch_order"]

    base_values = problem.evaluate(base)
    left_values = problem.evaluate(left)
    right_values = problem.evaluate(right)
    target_values = problem.evaluate(target)

    assert left_values["energy_mj"] < base_values["energy_mj"]
    assert right_values["makespan_ms"] < base_values["makespan_ms"]
    assert target_values["energy_mj"] < left_values["energy_mj"]
    for objective in problem.objectives:
        assert objective.goal == "min"
        assert target_values[objective.name] <= left_values[objective.name]
        assert target_values[objective.name] <= right_values[objective.name]

    left_trace = problem.analyze(left)
    right_trace = problem.analyze(right)
    target_trace = problem.analyze(target)
    assert left_trace.fused_edges == ()
    assert right_trace.fused_edges == ()
    assert target_trace.fused_edges == (("encode", "compress"),)
    assert target_trace.cross_worker_edge_count == left_trace.cross_worker_edge_count


def test_graph_contract_and_source_have_no_runtime_or_external_capabilities():
    assert set(DEPENDENCY_EDGES) == {
        ("acquire", "encode"),
        ("acquire", "inspect"),
        ("encode", "compress"),
        ("inspect", "classify"),
        ("compress", "package"),
        ("classify", "package"),
        ("classify", "audit"),
        ("package", "release"),
        ("audit", "release"),
    }
    assert len(TASKS) == 8

    tree = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots == {
        "__future__",
        "agent_evolve",
        "dataclasses",
        "pydantic",
        "typing",
    }

    forbidden_names = {
        "open",
        "input",
        "eval",
        "exec",
        "compile",
        "__import__",
    }
    forbidden_attributes = {
        "time",
        "perf_counter",
        "run",
        "system",
        "popen",
        "connect",
        "request",
        "getenv",
        "load_dotenv",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_names
        elif isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_attributes

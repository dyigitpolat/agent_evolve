"""Provider-free conformance for the Airfoil-v7 generic-port composition."""

from __future__ import annotations

import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path

from agent_evolve.agentic import (
    AgenticBenchmark,
    DetailedEvaluation,
    DetailedEvaluationAdapter,
    DetailedEvaluationPayload,
    EvaluationCheckStatus,
    EvaluationTimings,
    AgenticEvolutionEngine,
    FailureCategory,
    FailureCode,
    InsightMemoryBank,
    OutcomeRelation,
    freeze_json,
    thaw_json,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.phenotype_recourse import PhenotypeIdentity
from agent_evolve.ports.variation_catalog import (
    FiniteVariationCatalog,
    bind_finite_variation_catalog,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
    AirfoilConvergenceEvaluationError,
)
from examples.benchmarks.engibench_airfoil import v7_contract as contract_module
from examples.benchmarks.engibench_airfoil import v7_problem_def as problem_module
from examples.benchmarks.engibench_airfoil.problem_def import (
    AirfoilPanelEvaluation,
    candidate_sha256,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AIRFOIL_V7_REWARD_BINDING,
    ARCHIVE_DEFINITION_SHA256,
    LIFT_TARGET,
    NEUTRAL_DECODED_FLOAT32_LE_SHA256,
    NEUTRAL_POINT_DRAGS,
    REWARD_DEFINITION_SHA256,
    SHAPE_CONTROL_DECODED_FLOAT32_LE_SHA256,
    AirfoilV7PhenotypeIdentityPolicy,
    decoded_float32_le_bytes,
    local_delta_parent_feedback,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AIRFOIL_V7_ACTION_SEMANTICS,
    AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    TASK_SHA256,
    VIOLATION_NAME,
    AirfoilV7DetailedEvaluationAdapter,
    AirfoilV7Problem,
    benchmark,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7ShapeVariationCatalog,
    AirfoilV7TrimVariationCatalog,
    AirfoilV7UnionVariationCatalog,
    task_keyed_presentation_sha256,
)


NEUTRAL = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": [0.0] * 10,
    "lower_coefficients": [0.0] * 10,
    "alpha_deg": [2.5, 2.5, 2.5],
}


def _run(awaitable):
    """Run evaluator work with a heartbeat and an explicitly joined pool."""

    async def run_with_heartbeat():
        execution = asyncio.create_task(awaitable)
        while not execution.done():
            await asyncio.sleep(0.01)
        return await execution

    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="airfoil_v7_composition_evaluator",
    )
    loop.set_default_executor(executor)
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_with_heartbeat())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        loop.close()
        asyncio.set_event_loop(None)


def _success_record(candidate: dict) -> dict:
    points = []
    for index, cd in enumerate(NEUTRAL_POINT_DRAGS):
        witness = {
            "authoritative_status": {
                "solve_failed": False,
                "fatal_fail": False,
                "check_solution_failure": False,
            },
            "residual_evidence": {
                "free_stream_total_residual_reference": 2.0,
                "convergence_history": {
                    "history_rows": 12,
                    "final_total_minor_iters": 30,
                    "series": {
                        "linear_res": {"initial": 1.0, "final": 1e-5},
                        "resrho": {"initial": 0.0, "final": -9.1},
                        "resrhoe": {"initial": 0.0, "final": -9.0},
                    },
                },
            },
        }
        points.append(
            {
                "index": index,
                "cd": cd,
                "cl": LIFT_TARGET,
                "evaluator_evidence": {
                    "contract_id": EVIDENCE_CONTRACT_ID,
                    "evaluator_id": ADFLOW_EVALUATOR_ID,
                    "accepted": True,
                    "witness": witness,
                },
            }
        )
    return {
        "schema_version": 2,
        "evaluator_id": V2_EVALUATOR_ID,
        "status": "evaluated",
        "candidate_sha256": candidate_sha256(candidate),
        "task_sha256": TASK_SHA256,
        "evaluator_calls": 3,
        "decoder_audit": {
            "area_ratio": 1.0,
            "area_ratio_bounds": [0.8873697327569672, 1.2],
            "decoded_coords_sha256": NEUTRAL_DECODED_FLOAT32_LE_SHA256,
            "external_representation_not_upstream_ffd": True,
        },
        "points": points,
    }


class _SuccessfulRawProblem:
    def __init__(self, evaluation: AirfoilPanelEvaluation) -> None:
        self.evaluation = evaluation
        self.calls = 0

    def evaluate_raw(self, config: object) -> AirfoilPanelEvaluation:
        del config
        self.calls += 1
        return self.evaluation


class _FailingRawProblem:
    def __init__(self, error: AirfoilConvergenceEvaluationError) -> None:
        self.error = error

    def evaluate_raw(self, config: object) -> AirfoilPanelEvaluation:
        del config
        raise self.error


class _UnusedGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("seed-only conformance must not call propose")

    async def reflect(self, request):
        del request
        raise AssertionError("seed-only conformance must not call reflect")


def _detailed(f_value: float, v_value: float, label: str) -> DetailedEvaluation:
    payload = DetailedEvaluationPayload(
        failure=None,
        objectives=((OBJECTIVE_NAME, float(f_value)),),
        violations=((VIOLATION_NAME, float(v_value)),),
        checks=(),
        receipt=None,
        evaluator=EVALUATOR_IDENTITY,
    )
    return DetailedEvaluation(
        phenotype=PhenotypeIdentity(
            policy_id="airfoil_v7_test",
            policy_version=1,
            value_sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
        ),
        payload=payload,
        timings=EvaluationTimings(total_wall_seconds=0.0),
    )


def test_problem_exposes_only_normalized_drag_objective() -> None:
    problem = AirfoilV7Problem(raw_problem=object.__new__(_SuccessfulRawProblem))
    assert tuple((item.name, item.goal) for item in problem.objectives) == (
        (OBJECTIVE_NAME, "min"),
    )
    assert "max_lift_target_error" not in {item.name for item in problem.objectives}
    assert isinstance(benchmark, AgenticBenchmark)
    assert benchmark.optimization_semantics is AIRFOIL_V7_OPTIMIZATION_SEMANTICS
    assert benchmark.action_semantics is AIRFOIL_V7_ACTION_SEMANTICS
    action_record = AIRFOIL_V7_ACTION_SEMANTICS.to_record()
    assert action_record["declared_option_families"] == [
        "shape_only",
        "trim_only",
    ]
    assert [
        value["catalog_id"] for value in action_record["catalog_identities"]
    ] == ["airfoil_v7_shape", "airfoil_v7_trim", "airfoil_v7_union"]
    trim_axis = next(
        value for value in action_record["axes"]
        if value["axis_id"] == "three_point_trim"
    )
    assert [value["index"] for value in trim_axis["coordinates"]] == [0, 1, 2]
    assert "without broadcasting" in trim_axis["independence"]
    assert any(
        "spanwise or chordwise stations" in value
        for value in trim_axis["excluded_interpretations"]
    )
    semantic_record = AIRFOIL_V7_OPTIMIZATION_SEMANTICS.to_record()
    metric_records = {
        row["metric_id"]: row for row in semantic_record["metrics"]
    }
    lift = metric_records[f"violation:{VIOLATION_NAME}"]
    assert lift["reference_target"] == LIFT_TARGET
    assert "sum_i(abs(cl_i - lift_target)" in lift["definition"]
    assert "negative" in lift["witness_interpretation"]
    ordering = semantic_record["outcome_ordering"]
    assert ordering["metric_priority"] == [
        f"violation:{VIOLATION_NAME}",
        f"objective:{OBJECTIVE_NAME}",
    ]
    assert ordering["relation_policy"]["definition_sha256"] == (
        ARCHIVE_DEFINITION_SHA256
    )
    assert tuple(
        identity[0] for identity in benchmark.finite_variation_catalog_identities
    ) == ("airfoil_v7_shape", "airfoil_v7_trim", "airfoil_v7_union")
    assert (
        len(benchmark.bind_finite_variation("airfoil_v7_union", NEUTRAL).options) == 80
    )


def test_evidence_adapter_projects_success_and_sorted_generic_checks(tmp_path) -> None:
    record = _success_record(NEUTRAL)
    receipt_path = tmp_path / "success.json"
    receipt_bytes = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    receipt_path.write_bytes(receipt_bytes)
    raw = AirfoilPanelEvaluation(
        candidate_sha256=candidate_sha256(NEUTRAL),
        objective_values={
            "mean_drag_coefficient": sum(NEUTRAL_POINT_DRAGS) / 3.0,
            "max_lift_target_error": 0.0,
        },
        wall_seconds=20.0,
        record_path=receipt_path,
        record=record,
    )
    adapter = AirfoilV7DetailedEvaluationAdapter(_SuccessfulRawProblem(raw))
    assert isinstance(adapter, DetailedEvaluationAdapter)
    payload = adapter.evaluate_evidence(NEUTRAL)

    assert payload.failure is None
    assert payload.objectives == ((OBJECTIVE_NAME, 1.0),)
    assert payload.violations == ((VIOLATION_NAME, 0.0),)
    assert tuple(item.name for item in payload.checks) == (
        "area_bounds",
        "geometry",
        "point_0_convergence",
        "point_1_convergence",
        "point_2_convergence",
        "three_point_panel",
    )
    assert all(item.status is EvaluationCheckStatus.PASS for item in payload.checks)
    assert payload.receipt is not None
    assert payload.receipt.sha256_hex == hashlib.sha256(receipt_bytes).hexdigest()
    assert payload.active_wall_seconds == 20.0
    assert payload.resource_queue_wall_seconds is None


def test_engine_consumes_airfoil_only_through_generic_ports(tmp_path) -> None:
    record = _success_record(NEUTRAL)
    receipt_path = tmp_path / "engine-success.json"
    receipt_path.write_text(json.dumps(record), encoding="utf-8")
    raw_evaluation = AirfoilPanelEvaluation(
        candidate_sha256=candidate_sha256(NEUTRAL),
        objective_values={
            "mean_drag_coefficient": sum(NEUTRAL_POINT_DRAGS) / 3.0,
            "max_lift_target_error": 0.0,
        },
        wall_seconds=20.0,
        record_path=receipt_path,
        record=record,
    )
    raw_problem = _SuccessfulRawProblem(raw_evaluation)
    problem = AirfoilV7Problem(raw_problem=raw_problem)
    ids = DeterministicIdFactory("airfoil_v7_generic_ports")
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=_UnusedGenerator(),
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=7,
        detailed_evaluator=problem.detailed_evaluator,
        outcome_relation_binding=AIRFOIL_V7_ARCHIVE_RELATION,
        phenotype_identity_policy=AirfoilV7PhenotypeIdentityPolicy(),
        reward_policy=AIRFOIL_V7_REWARD_BINDING.score,
        reward_definition_hash=AIRFOIL_V7_REWARD_BINDING.definition_hash,
    )
    candidate = _run(engine.register_seed(NEUTRAL, label="neutral"))
    assert candidate.valid is True
    assert candidate.objective_map == {OBJECTIVE_NAME: 1.0}
    assert candidate.detailed_evaluation is not None
    assert candidate.detailed_evaluation.violations == ((VIOLATION_NAME, 0.0),)
    assert raw_problem.calls == 1
    evidence_failed = replace(
        candidate,
        evidence_compliant=False,
        evidence_failure="unsupported causal attribution",
    )
    assert (
        AIRFOIL_V7_REWARD_BINDING.score(
            evidence_failed,
            (candidate,),
            problem.objectives,
        )
        == -1.0
    )


def test_v7_benchmark_uses_only_the_public_agentic_facade() -> None:
    benchmark_dir = Path("examples/benchmarks/engibench_airfoil")
    for name in (
        "v7_contract.py",
        "v7_experiment_support.py",
        "v7_problem_def.py",
        "v7_readiness.py",
        "v7_variation_catalog.py",
    ):
        tree = ast.parse((benchmark_dir / name).read_text(encoding="utf-8"))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("agent_evolve")
        }
        assert modules == {"agent_evolve.agentic"}


def test_authoritative_solver_failure_maps_to_typed_candidate_evidence(
    tmp_path,
) -> None:
    record = {
        "schema_version": 2,
        "evaluator_id": V2_EVALUATOR_ID,
        "status": "candidate_invalid",
        "failure_classification": "authoritative_solver_failure",
        "evaluator_calls": 1,
        "failed_point_index": 0,
        "failure": {"type": "CandidateSolverOutcome", "message": "failed"},
        "evaluator_evidence": {
            "authoritative_status": {
                "solve_failed": True,
                "fatal_fail": False,
                "check_solution_failure": True,
            }
        },
    }
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    error = AirfoilConvergenceEvaluationError(
        "solver failed",
        candidate_invalid=True,
        record_path=path,
        record=record,
    )
    payload = AirfoilV7DetailedEvaluationAdapter(
        _FailingRawProblem(error)
    ).evaluate_evidence(NEUTRAL)

    assert payload.failure is not None
    assert payload.failure.category is FailureCategory.CANDIDATE
    assert payload.failure.code is FailureCode.NUMERICAL_NONCONVERGENCE
    assert payload.objectives == ()
    assert payload.violations == ()
    assert tuple(item.name for item in payload.checks) == ("point_0_convergence",)
    assert payload.checks[0].status is EvaluationCheckStatus.FAIL
    assert payload.receipt is not None


def test_success_projection_rejects_untrusted_status_and_digest_shapes(
    tmp_path,
) -> None:
    bad_records = []
    nonfalse = _success_record(NEUTRAL)
    nonfalse["points"][1]["evaluator_evidence"]["witness"]["authoritative_status"][
        "fatal_fail"
    ] = True
    bad_records.append(nonfalse)
    uppercase_digest = _success_record(NEUTRAL)
    uppercase_digest["decoder_audit"]["decoded_coords_sha256"] = "A" * 64
    bad_records.append(uppercase_digest)

    for index, record in enumerate(bad_records):
        path = tmp_path / f"bad-success-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        raw = AirfoilPanelEvaluation(
            candidate_sha256=candidate_sha256(NEUTRAL),
            objective_values={
                "mean_drag_coefficient": sum(NEUTRAL_POINT_DRAGS) / 3.0,
                "max_lift_target_error": 0.0,
            },
            wall_seconds=20.0,
            record_path=path,
            record=record,
        )
        payload = AirfoilV7DetailedEvaluationAdapter(
            _SuccessfulRawProblem(raw)
        ).evaluate_evidence(NEUTRAL)
        assert payload.failure is not None
        assert payload.failure.category is FailureCategory.SYSTEM
        assert payload.failure.code is FailureCode.EVALUATOR_CONTRACT_VIOLATION
        assert payload.objectives == ()
        assert payload.violations == ()


def test_evaluator_relation_and_reward_identities_bind_disjoint_semantics() -> None:
    evaluator_definition = dict(problem_module._EVALUATOR_CONTEXT_DEFINITION)
    evaluator_objective = dict(evaluator_definition["objective"])
    evaluator_objective["formula"] = "different_physical_projection"
    evaluator_definition["objective"] = evaluator_objective
    assert (
        problem_module._context_sha256(evaluator_definition)
        != EVALUATOR_IDENTITY.evaluator_context_sha256
    )

    archive_definition = dict(contract_module._ARCHIVE_DEFINITION)
    archive_definition["order"] = "different_transitive_order"
    assert (
        contract_module._canonical_hash(
            contract_module._ARCHIVE_HASH_DOMAIN,
            archive_definition,
        )
        != ARCHIVE_DEFINITION_SHA256
    )

    reward_definition = dict(contract_module._REWARD_DEFINITION)
    reward_definition["delta_f"] = reward_definition["delta_f"] * 2
    assert (
        contract_module._canonical_hash(
            contract_module._REWARD_HASH_DOMAIN,
            reward_definition,
        )
        != REWARD_DEFINITION_SHA256
    )
    evaluator_text = json.dumps(problem_module._EVALUATOR_CONTEXT_DEFINITION)
    assert "delta_f" not in evaluator_text
    assert "delta_v" not in evaluator_text
    assert (
        len(
            {
                EVALUATOR_IDENTITY.evaluator_context_sha256,
                ARCHIVE_DEFINITION_SHA256,
                REWARD_DEFINITION_SHA256,
            }
        )
        == 3
    )


def test_semantic_phenotype_reproduces_frozen_decoder_hashes() -> None:
    neutral_bytes = decoded_float32_le_bytes(NEUTRAL)
    assert (
        hashlib.sha256(neutral_bytes).hexdigest() == NEUTRAL_DECODED_FLOAT32_LE_SHA256
    )
    shape = {
        key: list(value) if isinstance(value, list) else value
        for key, value in NEUTRAL.items()
    }
    shape["upper_coefficients"][4] = 0.005
    shape["lower_coefficients"][4] = -0.005
    assert (
        hashlib.sha256(decoded_float32_le_bytes(shape)).hexdigest()
        == SHAPE_CONTROL_DECODED_FLOAT32_LE_SHA256
    )
    angle = {
        key: list(value) if isinstance(value, list) else value
        for key, value in NEUTRAL.items()
    }
    angle["alpha_deg"] = [2.75, 2.75, 2.75]
    policy = AirfoilV7PhenotypeIdentityPolicy()
    neutral_identity = policy.identify(freeze_json(NEUTRAL))
    assert neutral_identity == policy.identify(NEUTRAL)
    assert neutral_bytes == decoded_float32_le_bytes(angle)
    assert policy.identify(angle) != neutral_identity


def _task_keyed_option_ids(catalog_id, options, task_sha256=TASK_SHA256):
    return tuple(
        option.option_id
        for option in sorted(
            options,
            key=lambda option: (
                task_keyed_presentation_sha256(
                    task_sha256=task_sha256,
                    catalog_id=catalog_id,
                    family=option.family,
                    option_id=option.option_id,
                ),
                option.option_id,
            ),
        )
    )


def test_finite_catalogs_preserve_sets_with_task_keyed_presentation() -> None:
    parent = freeze_json(NEUTRAL)
    shape_catalog = AirfoilV7ShapeVariationCatalog()
    trim_catalog = AirfoilV7TrimVariationCatalog()
    union_catalog = AirfoilV7UnionVariationCatalog()
    assert isinstance(shape_catalog, FiniteVariationCatalog)
    assert isinstance(trim_catalog, FiniteVariationCatalog)
    assert isinstance(union_catalog, FiniteVariationCatalog)
    shape = bind_finite_variation_catalog(shape_catalog, parent)
    trim = bind_finite_variation_catalog(trim_catalog, parent)
    union = bind_finite_variation_catalog(union_catalog, parent)
    assert len(shape.options) == 16
    assert len(trim.options) == 64
    assert len(union.options) == 80
    expected_shape_ids = {
        f"shape.{mode}.{amplitude}"
        for mode in (
            "camber_front",
            "camber_aft",
            "thickness_front",
            "thickness_aft",
        )
        for amplitude in ("n0030", "n0015", "p0015", "p0030")
    }
    expected_trim_ids = {
        "trim." + ".".join(deltas)
        for deltas in product(("n050", "n025", "p025", "p050"), repeat=3)
    }
    shape_ids = tuple(item.option_id for item in shape.options)
    trim_ids = tuple(item.option_id for item in trim.options)
    union_ids = tuple(item.option_id for item in union.options)
    assert set(shape_ids) == expected_shape_ids
    assert set(trim_ids) == expected_trim_ids
    assert set(union_ids) == expected_shape_ids | expected_trim_ids
    assert shape_ids == _task_keyed_option_ids(shape.catalog_id, shape.options)
    assert trim_ids == _task_keyed_option_ids(trim.catalog_id, trim.options)
    assert union_ids == _task_keyed_option_ids(union.catalog_id, union.options)
    assert union_ids != shape_ids + trim_ids

    alternate_task = hashlib.sha256(b"alternate-airfoil-task").hexdigest()
    assert shape_ids != _task_keyed_option_ids(
        shape.catalog_id,
        shape.options,
        alternate_task,
    )
    assert (
        union.identity_sha256
        == bind_finite_variation_catalog(union_catalog, parent).identity_sha256
    )

    moved_parent = freeze_json({**NEUTRAL, "alpha_deg": [2.25, 2.5, 2.75]})
    moved_union = bind_finite_variation_catalog(union_catalog, moved_parent)
    assert tuple(item.option_id for item in moved_union.options) == union_ids
    assert moved_union.identity_sha256 != union.identity_sha256

    boundary_parent = {**NEUTRAL, "upper_coefficients": [0.025] * 10}
    boundary_options = AirfoilV7ShapeVariationCatalog().options(
        freeze_json(boundary_parent)
    )
    positive = next(
        item
        for item in boundary_options
        if item.option_id == "shape.camber_front.p0030"
    )
    materialized = thaw_json(positive.child_configuration)
    assert math.isclose(materialized["upper_coefficients"][1], 0.028)
    assert materialized["alpha_deg"] == boundary_parent["alpha_deg"]


def test_exact_archive_order_breaks_delta_cycle_and_local_reward_stays_thresholded() -> (
    None
):
    a = _detailed(2.0, 0.000, "a")
    b = _detailed(1.0, 0.004, "b")
    c = _detailed(0.0, 0.008, "c")
    assert AIRFOIL_V7_ARCHIVE_RELATION.relate(a, b) is OutcomeRelation.BETTER
    assert AIRFOIL_V7_ARCHIVE_RELATION.relate(b, c) is OutcomeRelation.BETTER
    assert AIRFOIL_V7_ARCHIVE_RELATION.relate(a, c) is OutcomeRelation.BETTER
    assert ARCHIVE_DEFINITION_SHA256 != REWARD_DEFINITION_SHA256

    # Within delta_V, the local artifact-95 feedback can reward resolved drag.
    assert local_delta_parent_feedback(b, a) == 1.0
    # Across delta_V, exact violation direction takes priority over drag.
    assert local_delta_parent_feedback(a, c) == 1.0
    unresolved = _detailed(2.0 - 0.0005, 0.000 + 0.002, "unresolved")
    assert local_delta_parent_feedback(unresolved, a) == 0.0

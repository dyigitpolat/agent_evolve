"""Conformance for the pybamm_fastcharge typed WorkloadKit adapter.

Provider-free composition/catalog/schema checks and the termination-validity
policy unit tests always run; the real-eval contract tests execute DFN/IDAKLU
solves through the isolated domain virtualenv at the 3-4 s mesh-factor-1 test
fidelity (production fidelity stays config-switchable via settings) and skip
when that environment is absent.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from agent_evolve.application.evolution_campaign import BenchmarkSessionRequest
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from examples.benchmarks.pybamm_fastcharge.agentic_benchmark import (
    create_benchmark,
)
from examples.benchmarks.pybamm_fastcharge.campaign_workload import (
    CAMPAIGN_WORKLOAD_ID,
    compose_pybamm_fastcharge_campaign_workload,
)
from examples.benchmarks.pybamm_fastcharge.evaluator import (
    OBJECTIVE_CHARGE_TIME,
    OBJECTIVE_TEMP_RISE,
    PybammEvaluationError,
    PybammEvaluatorSettings,
    admit_runner_result,
    default_domain_python,
)
from examples.benchmarks.pybamm_fastcharge.finite_variation_catalog import (
    CATALOG_ID,
    PybammFastChargeFiniteVariationCatalog,
)
from examples.benchmarks.pybamm_fastcharge.problem_def import (
    SEED_BASELINE_1C,
    PybammFastChargeProblem,
    default_settings,
    normalize_candidate,
)


requires_pybamm_env = pytest.mark.skipif(
    not default_domain_python().exists(),
    reason="PyBaMM domain virtualenv is not installed on this host",
)

_AGGRESSIVE_3C = {
    "stage1_c_rate": 3.0,
    "stage2_c_rate": 2.0,
    "stage3_c_rate": 1.0,
    "switch_v1": 3.9,
    "switch_v2": 4.1,
}
_VALID_RESULT = {
    "status": "ok",
    "termination": "event: C-rate cut-off [experiment]",
    "n_steps_solved": 4,
    "charge_time_s": 6505.571,
    "peak_temp_rise_K": 12.5055,
    "plating_proxy_min_V": 0.032559,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


class _ForbiddenEvaluator:
    evaluator_id = "pybamm-fastcharge-dfn-idaklu-v1"
    evaluator_concurrency = 1

    def __init__(self) -> None:
        self.calls = 0

    def preflight(self):
        self.calls += 1
        raise AssertionError("composition must not run PyBaMM preflight")

    def evaluate_protocol(self, protocol):
        del protocol
        self.calls += 1
        raise AssertionError("composition must not run a DFN solve")


def test_schema_accepts_seed_and_rejects_invalid_protocols() -> None:
    candidate = normalize_candidate(SEED_BASELINE_1C)
    assert candidate.model_dump(mode="python") == SEED_BASELINE_1C
    with pytest.raises(ValidationError):
        normalize_candidate({**SEED_BASELINE_1C, "stage1_c_rate": 6.0})
    with pytest.raises(ValidationError):
        normalize_candidate({**SEED_BASELINE_1C, "switch_v2": 4.3})
    with pytest.raises(ValidationError, match="switch_v1"):
        normalize_candidate(
            {**SEED_BASELINE_1C, "switch_v1": 4.16, "switch_v2": 4.15}
        )
    with pytest.raises(ValidationError):
        normalize_candidate({**SEED_BASELINE_1C, "extra_stage": 1.0})


def test_candidate_key_is_canonical_and_deduplicates() -> None:
    reordered = dict(reversed(list(SEED_BASELINE_1C.items())))
    assert PybammFastChargeProblem.candidate_key(SEED_BASELINE_1C) == (
        PybammFastChargeProblem.candidate_key(reordered)
    )
    assert PybammFastChargeProblem.candidate_key(SEED_BASELINE_1C) != (
        PybammFastChargeProblem.candidate_key(_AGGRESSIVE_3C)
    )


@pytest.mark.parametrize(
    ("parent", "expected_count"),
    [(SEED_BASELINE_1C, 28), (_AGGRESSIVE_3C, 30)],
    ids=["baseline_1c", "aggressive_3c"],
)
def test_catalog_every_action_materializes_a_valid_child(
    parent: dict[str, float], expected_count: int
) -> None:
    catalog = PybammFastChargeFiniteVariationCatalog()
    frozen_parent = _object(parent)
    parent_sha = typed_json_sha256(frozen_parent)
    options = catalog.options(frozen_parent)
    assert len(options) == expected_count
    assert len({option.option_id for option in options}) == len(options)
    for option in options:
        assert option.parent_configuration_sha256 == parent_sha
        child = normalize_candidate(thaw_json(option.child_configuration))
        assert child.switch_v1 < child.switch_v2
        assert option.child_configuration_sha256 != parent_sha
        assert option.family in catalog.option_families


def test_validity_policy_admits_only_complete_terminations() -> None:
    settings = default_settings()
    values = admit_runner_result(dict(_VALID_RESULT), settings)
    assert values == {
        OBJECTIVE_CHARGE_TIME: 6505.571,
        OBJECTIVE_TEMP_RISE: 12.5055,
    }
    # The measured IDAKLU conv-fail shape: early termination that would
    # otherwise pass as a plausible outcome.
    with pytest.raises(ValueError, match="terminated early"):
        admit_runner_result(
            {
                **_VALID_RESULT,
                "termination": "event: Voltage > 4.2 V",
                "n_steps_solved": 2,
                "charge_time_s": 2269.1,
            },
            settings,
        )
    with pytest.raises(ValueError, match="terminated early"):
        admit_runner_result({**_VALID_RESULT, "n_steps_solved": 3}, settings)
    with pytest.raises(ValueError, match="solve failed"):
        admit_runner_result(
            {"status": "solver_error", "error": "SolverError: IDA_CONV_FAIL"},
            settings,
        )
    with pytest.raises(ValueError, match="plating"):
        admit_runner_result(
            {**_VALID_RESULT, "plating_proxy_min_V": 0.002}, settings
        )
    with pytest.raises(PybammEvaluationError):
        admit_runner_result({**_VALID_RESULT, "charge_time_s": None}, settings)
    with pytest.raises(PybammEvaluationError):
        admit_runner_result({**_VALID_RESULT, "status": "surprise"}, settings)


def test_workload_kit_composes_provider_and_solver_free() -> None:
    evaluator = _ForbiddenEvaluator()
    problem = PybammFastChargeProblem(default_settings(), evaluator=evaluator)
    kit = compose_pybamm_fastcharge_campaign_workload(
        benchmark=create_benchmark(problem),
        evaluator_preflight_receipt=_object(
            {"qualified": True, "dfn_solves": 0, "provider_calls": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "serialized_pybamm_domain_venv", "active": True}
        ),
    )
    config = kit.to_campaign_workload()
    ports = config.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("pybamm-workload-kit-protocol"),
            budget_sha256=_sha("pybamm-workload-kit-budget"),
            outer_seed=20260727,
            requested_evaluator_concurrency=1,
        )
    )
    seeds = ports.seeds.load(session)
    parent = seeds.seeds[0].configuration
    variation = ports.catalog.bind(session.benchmark, parent, ())
    memory = ports.evidence.initialize_memory(session, seeds)
    context = thaw_json(ports.evidence.context(session, parent, variation, memory))
    receipt = thaw_json(kit.integration_receipt())

    assert kit.workload_id == CAMPAIGN_WORKLOAD_ID
    assert kit.selected_finite_catalog_id == CATALOG_ID
    assert len(seeds.seeds) == 1
    assert context["finite_variation"]["eligible_option_count"] == 28
    assert tuple(item["name"] for item in context["objectives"]) == (
        OBJECTIVE_CHARGE_TIME,
        OBJECTIVE_TEMP_RISE,
    )
    assert receipt["required_obligation_count"] == 4
    assert receipt["uses_default_schema_evidence"] is True
    assert evaluator.calls == 0


def _test_fidelity_settings(**overrides) -> PybammEvaluatorSettings:
    return PybammEvaluatorSettings(
        domain_python=default_domain_python(),
        mesh_factor=1,
        **overrides,
    )


@requires_pybamm_env
def test_real_evaluations_honor_the_bi_objective_contract() -> None:
    problem = PybammFastChargeProblem(_test_fidelity_settings())
    detailed = problem.evaluate_detailed(SEED_BASELINE_1C)
    values = detailed.objective_values
    assert set(values) == {OBJECTIVE_CHARGE_TIME, OBJECTIVE_TEMP_RISE}
    assert 6000.0 < values[OBJECTIVE_CHARGE_TIME] < 7000.0
    assert 5.0 < values[OBJECTIVE_TEMP_RISE] < 25.0
    assert detailed.result["n_steps_solved"] == 4
    assert "C-rate cut-off" in detailed.result["termination"]
    assert detailed.result["plating_proxy_min_V"] > 0.01
    assert detailed.policy["solver"] == "idaklu"
    assert detailed.policy["mesh_factor"] == 1

    # Objective-level determinism across fresh subprocesses (memo evidence:
    # ~1e-7 relative; asserted here at 1e-5).
    repeat = problem.evaluate(SEED_BASELINE_1C)
    assert repeat[OBJECTIVE_CHARGE_TIME] == pytest.approx(
        values[OBJECTIVE_CHARGE_TIME], rel=1.0e-5
    )

    # Third real evaluation: one catalog child (stage-1 current move).
    options = PybammFastChargeFiniteVariationCatalog().options(
        freeze_json(SEED_BASELINE_1C)
    )
    child_option = next(
        option
        for option in options
        if dict(option.metadata)["locus"] == "stage1_c_rate"
        and dict(option.metadata)["target_value"] == "2"
    )
    child_values = problem.evaluate(thaw_json(child_option.child_configuration))
    assert set(child_values) == {OBJECTIVE_CHARGE_TIME, OBJECTIVE_TEMP_RISE}
    assert child_values != values


@requires_pybamm_env
def test_real_timeout_returns_typed_solver_slow_outcome() -> None:
    problem = PybammFastChargeProblem(_test_fidelity_settings(timeout_s=0.5))
    with pytest.raises(ValueError, match="solver-slow"):
        problem.evaluate(SEED_BASELINE_1C)

"""Conformance for the scip_miplib typed WorkloadKit adapter.

Provider-free composition/catalog/schema checks always run; the real-eval
contract tests execute two SCIP panel evaluations through the isolated domain
virtualenv at a reduced fidelity (2-instance panel, 5 s limit) and skip when
that environment is absent.
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
from examples.benchmarks.scip_miplib.agentic_benchmark import create_benchmark
from examples.benchmarks.scip_miplib.campaign_workload import (
    CAMPAIGN_WORKLOAD_ID,
    compose_scip_miplib_campaign_workload,
)
from examples.benchmarks.scip_miplib.evaluator import (
    OBJECTIVE_GAP_TIME,
    OBJECTIVE_PD_INTEGRAL,
    ScipEvaluatorSettings,
    ScipSubprocessEvaluator,
)
from examples.benchmarks.scip_miplib.finite_variation_catalog import (
    CATALOG_ID,
    LOCUS_GRIDS,
    ScipMiplibFiniteVariationCatalog,
)
from examples.benchmarks.scip_miplib.problem_def import (
    SEED_AGGRESSIVE,
    SEED_DEFAULT,
    SEED_NO_CUTS_NO_HEUR,
    ScipMiplibProblem,
    default_domain_python,
    default_settings,
    development_panel,
    held_out_panel,
    load_instance_split,
    normalize_candidate,
)


requires_scip_env = pytest.mark.skipif(
    not default_domain_python().exists(),
    reason="SCIP domain virtualenv is not installed on this host",
)

_SEEDS = {
    "seed_default": SEED_DEFAULT,
    "seed_no_cuts_no_heur": SEED_NO_CUTS_NO_HEUR,
    "seed_aggressive": SEED_AGGRESSIVE,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


class _ForbiddenEvaluator:
    evaluator_id = "scip-miplib-panel-config-v1"
    evaluator_concurrency = 1

    def __init__(self) -> None:
        self.calls = 0

    def preflight(self):
        self.calls += 1
        raise AssertionError("composition must not run SCIP preflight")

    def evaluate_panel(self, **kwargs):
        del kwargs
        self.calls += 1
        raise AssertionError("composition must not run a SCIP solve")


def _kit(evaluator=None):
    problem = ScipMiplibProblem(default_settings(), evaluator=evaluator)
    benchmark = create_benchmark(problem)
    return compose_scip_miplib_campaign_workload(
        benchmark=benchmark,
        evaluator_preflight_receipt=_object(
            {"qualified": True, "scip_solves": 0, "provider_calls": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "serialized_scip_domain_venv", "active": True}
        ),
    )


def test_schema_accepts_seeds_and_rejects_invalid_configurations() -> None:
    for name, seed in _SEEDS.items():
        candidate = normalize_candidate(seed)
        assert candidate.model_dump(mode="python") == seed, name
    bad_extra = {**SEED_DEFAULT, "unknown_group": {}}
    with pytest.raises(ValidationError):
        normalize_candidate(bad_extra)
    bad_range = {
        **SEED_DEFAULT,
        "separating": {**SEED_DEFAULT["separating"], "maxcuts": -5},
    }
    with pytest.raises(ValidationError):
        normalize_candidate(bad_range)
    bad_type = {
        **SEED_DEFAULT,
        "branching": {**SEED_DEFAULT["branching"], "preferbinary": "yes"},
    }
    with pytest.raises(ValidationError):
        normalize_candidate(bad_type)
    bad_enum = {
        **SEED_DEFAULT,
        "lp_node": {**SEED_DEFAULT["lp_node"], "pricing": "x"},
    }
    with pytest.raises(ValidationError):
        normalize_candidate(bad_enum)


def test_candidate_key_is_canonical_and_deduplicates() -> None:
    reordered = {
        group: dict(reversed(list(fields.items())))
        for group, fields in reversed(list(SEED_DEFAULT.items()))
    }
    assert ScipMiplibProblem.candidate_key(SEED_DEFAULT) == (
        ScipMiplibProblem.candidate_key(reordered)
    )
    assert ScipMiplibProblem.candidate_key(SEED_DEFAULT) != (
        ScipMiplibProblem.candidate_key(SEED_AGGRESSIVE)
    )


def test_frozen_instance_split_is_disjoint_and_complete() -> None:
    split = load_instance_split()
    assert len(split["development"]) == 8
    assert len(split["held_out"]) == 4
    dev = development_panel()
    held = held_out_panel()
    assert len(dev) == 8 and len(held) == 4
    assert not {name for name, _ in dev} & {name for name, _ in held}


@pytest.mark.parametrize("seed_name", sorted(_SEEDS))
def test_catalog_every_action_materializes_a_valid_child(seed_name: str) -> None:
    catalog = ScipMiplibFiniteVariationCatalog()
    parent = _object(_SEEDS[seed_name])
    parent_sha = typed_json_sha256(parent)
    options = catalog.options(parent)
    assert 80 <= len(options) <= 1024
    assert len({option.option_id for option in options}) == len(options)
    for option in options:
        assert option.parent_configuration_sha256 == parent_sha
        child = normalize_candidate(thaw_json(option.child_configuration))
        assert option.child_configuration_sha256 != parent_sha
        assert option.family in catalog.option_families
        assert child.model_dump(mode="python") != _SEEDS[seed_name]


def test_catalog_cardinality_is_exact_for_an_on_grid_parent() -> None:
    expected = sum(len(grid) - 1 for _, grid, _ in LOCUS_GRIDS)
    options = ScipMiplibFiniteVariationCatalog().options(_object(SEED_DEFAULT))
    assert len(options) == expected == 84


def test_workload_kit_composes_provider_and_solver_free() -> None:
    evaluator = _ForbiddenEvaluator()
    kit = _kit(evaluator=evaluator)
    config = kit.to_campaign_workload()
    ports = config.build_ports()
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("scip-workload-kit-protocol"),
            budget_sha256=_sha("scip-workload-kit-budget"),
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
    assert len(seeds.seeds) == 3
    assert context["finite_variation"]["eligible_option_count"] == 84
    assert tuple(item["name"] for item in context["objectives"]) == (
        OBJECTIVE_PD_INTEGRAL,
        OBJECTIVE_GAP_TIME,
    )
    assert receipt["required_obligation_count"] == 4
    assert receipt["uses_default_schema_evidence"] is True
    assert evaluator.calls == 0


def _fast_real_settings() -> ScipEvaluatorSettings:
    by_name = {name.split(".")[0]: (name, sha) for name, sha in development_panel()}
    return ScipEvaluatorSettings(
        domain_python=default_settings().domain_python,
        instances_root=default_settings().instances_root,
        panel=(by_name["mas76"], by_name["binkar10_1"]),
        time_limit_s=5.0,
    )


@requires_scip_env
def test_real_preflight_verifies_domain_env_and_frozen_instances() -> None:
    evaluator = ScipSubprocessEvaluator(_fast_real_settings())
    record = evaluator.preflight()
    assert record["evaluator_id"] == "scip-miplib-panel-config-v1"
    assert len(record["instances"]) == 2
    assert record["time_limit_s"] == 5.0
    assert len(record["runner_sha256"]) == 64


@requires_scip_env
def test_real_evaluations_honor_the_bi_objective_contract() -> None:
    problem = ScipMiplibProblem(_fast_real_settings())
    detailed = problem.evaluate_detailed(SEED_DEFAULT)
    values = detailed.objective_values
    assert set(values) == {OBJECTIVE_PD_INTEGRAL, OBJECTIVE_GAP_TIME}
    assert all(value >= 0.0 for value in values.values())
    assert len(detailed.per_instance) == 2
    for record in detailed.per_instance:
        assert record["status"] in ("optimal", "timelimit")
    assert detailed.policy["time_limit_s"] == 5.0
    assert detailed.policy["random_seed_shift"] == 0

    # Second real evaluation: one catalog child (cuts+heuristics move).
    options = ScipMiplibFiniteVariationCatalog().options(_object(SEED_DEFAULT))
    child_option = next(
        option
        for option in options
        if dict(option.metadata)["locus"] == "separating.emphasis"
        and dict(option.metadata)["target_value"] == '"off"'
    )
    child_values = problem.evaluate(thaw_json(child_option.child_configuration))
    assert set(child_values) == {OBJECTIVE_PD_INTEGRAL, OBJECTIVE_GAP_TIME}
    assert all(value >= 0.0 for value in child_values.values())

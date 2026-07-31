"""Provider-free conformance for Timeloop's delayed identifiable G6 loop."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_evolve.agentic import (
    AgenticBenchmark,
    TypedConfigurationPhenotypeIdentityPolicy,
    freeze_json,
    objective_pareto_outcome_binding,
)
from agent_evolve.application.campaign_learning_runtime import (
    CampaignReflectionLearningRecordCodec,
)
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluationPayload,
    EvaluatorIdentity,
)
from agent_evolve.application.insight_memory import InsightLifecycleState
from agent_evolve.domain.outcome import (
    FailureCategory,
    FailureCode,
    FailureRecord,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, thaw_json
from agent_evolve.policies.memory.global_falsification import EvidenceProvenance
from examples.benchmarks.timeloop_codesign.v2.campaign_reflection import (
    OBJECTIVE_IDS,
    REFLECTION_DECISION_PATHS,
    REFLECTION_OPTION_FAMILIES,
)
from examples.benchmarks.timeloop_codesign.v2.detailed_evaluation import (
    timeloop_v2_optimization_semantics,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (
    OBJECTIVE_NAMES,
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.finite_variation_catalog import (
    TimeloopV2FiniteVariationCatalog,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (
    frozen_network_panel,
)
from examples.benchmarks.timeloop_codesign.v2.problem_def import (
    TimeloopV2CoDesignProblem,
)
from examples.development.run_timeloop_v2_provider_free_campaign import (
    PARENTS_PER_PORTFOLIO as PROVIDER_FREE_PARENTS_PER_PORTFOLIO,
    PORTFOLIO_GENERATIONS as PROVIDER_FREE_PORTFOLIO_GENERATIONS,
    _DeterministicEvaluator,
    _ProviderFreeCalibratedRunner,
    _family_bounded_evaluation_witness,
    _profile_portfolio_selection_implementation,
    run_provider_free_timeloop_campaign,
    run_timeloop_campaign,
)
from examples.development.run_timeloop_v2_frontier_probe_live import (
    PARENTS_PER_PORTFOLIO as LIVE_PARENTS_PER_PORTFOLIO,
    PORTFOLIO_GENERATIONS as LIVE_PORTFOLIO_GENERATIONS,
    _allocation_policy_identity,
    _construction_probe_contract,
    _experiment_profile_preregistration_fields,
    _g5_memory_path_audit,
    _candidate_universe_binding_valid,
    _portfolio_candidate_infeasible_count,
    _provider_free_schema_string_enum,
    _pre_simulator_infeasible_count,
    _selector_policy_binding_valid,
    _typed_candidate_infeasible_events,
)


def test_live_audit_imports_its_parent_cardinality_dependency() -> None:
    """Prevent a ten-minute construction traversal from ending in NameError."""

    assert LIVE_PARENTS_PER_PORTFOLIO == PROVIDER_FREE_PARENTS_PER_PORTFOLIO == 2
    assert LIVE_PORTFOLIO_GENERATIONS == PROVIDER_FREE_PORTFOLIO_GENERATIONS


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def test_provider_free_forecast_schema_resolves_single_and_multiple_codes() -> None:
    schema = {
        "$defs": {
            "one": {"type": "string", "const": "e0"},
            "many": {"type": "string", "enum": ["e0", "e1"]},
        }
    }

    assert _provider_free_schema_string_enum(schema, {"$ref": "#/$defs/one"}) == (
        "e0",
    )
    assert _provider_free_schema_string_enum(schema, {"$ref": "#/$defs/many"}) == (
        "e0",
        "e1",
    )


def test_live_audit_decodes_outcome_conditioned_nested_policy_receipt() -> None:
    from agent_evolve.application.global_wave_action_allocation import (
        GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_DEFINITION_SHA256,
        GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_ID,
        GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_VERSION,
    )
    from agent_evolve.application.outcome_conditioned_portfolio_selection import (
        OUTCOME_CONDITIONED_PORTFOLIO_POLICY_ID,
        OUTCOME_CONDITIONED_PORTFOLIO_POLICY_VERSION,
    )

    digest = "a" * 64
    payload = _object(
        {
            "schema_version": 2,
            "policy_definition_sha256": digest,
            "allocation": {
                "allocator_policy": {
                    "policy_id": "trusted_allocator",
                    "policy_version": 1,
                    "definition_sha256": "b" * 64,
                }
            },
            "global_wave_allocation": {
                "policy": {
                    "policy_id": GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_ID,
                    "policy_version": GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_VERSION,
                    "definition_sha256": (
                        GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_DEFINITION_SHA256
                    ),
                }
            },
            "forecast_universe_projection": {
                "mode": "authenticated_outcome_blind_candidate_pool",
                "source_contract_sha256": "c" * 64,
                "forecast_contract_sha256": "d" * 64,
                "common_candidate_pool_decision_sha256": "e" * 64,
                "outcomes_consulted": False,
                "model_or_provider_fields_consulted": False,
            },
            "proposal_topology": {
                "source_contract_sha256": "c" * 64,
                "proposal_contract_sha256": "d" * 64,
            },
        }
    )
    result = SimpleNamespace(
        decision=SimpleNamespace(
            policy_id=OUTCOME_CONDITIONED_PORTFOLIO_POLICY_ID,
            policy_version=OUTCOME_CONDITIONED_PORTFOLIO_POLICY_VERSION,
            policy_definition_sha256=digest,
        ),
        supplemental_audit=SimpleNamespace(
            audit_kind="outcome_conditioned_expert_portfolio",
            payload=payload,
        ),
    )
    proposal_support = SimpleNamespace(
        policy_id="unused_legacy_policy",
        policy_version=1,
        definition_sha256="f" * 64,
    )

    assert _selector_policy_binding_valid(result)
    assert _candidate_universe_binding_valid(
        result,
        proposal_support_policy=proposal_support,
    )


def test_live_audit_decodes_acquisition_certified_rich_policy_receipt() -> None:
    policy_id, policy_version, definition_sha256 = _allocation_policy_identity()
    allocator_identity = {
        "policy_id": policy_id,
        "policy_version": policy_version,
        "definition_sha256": definition_sha256,
    }
    decision_identity = {
        "policy_id": "pydantic_ai_acquisition_certified_residual_portfolio",
        "policy_version": 1,
        "policy_definition_sha256": _sha("acre-wrapper"),
    }
    result = SimpleNamespace(
        decision=SimpleNamespace(**decision_identity),
        supplemental_audit=SimpleNamespace(
            audit_kind="acquisition_certified_residual_portfolio_k8_to_k4",
            payload=_object(
                {
                    "schema_version": 5,
                    **decision_identity,
                    "allocator_policy": {
                        **allocator_identity,
                        "scorer": {"policy_id": "authenticated-extra-field"},
                    },
                    "allocation": {
                        "policy_id": policy_id,
                        "policy_version": policy_version,
                        "policy_definition_sha256": definition_sha256,
                        "certificate_scope": (
                            "frozen_strictly_prior_acquisition_not_unseen_outcome"
                        ),
                    },
                }
            ),
        ),
    )

    assert _selector_policy_binding_valid(result)


def test_live_audit_decodes_regret_bounded_rich_policy_receipt() -> None:
    policy_id, policy_version, definition_sha256 = _allocation_policy_identity()
    allocator_identity = {
        "policy_id": policy_id,
        "policy_version": policy_version,
        "definition_sha256": definition_sha256,
    }
    decision_identity = {
        "policy_id": "pydantic_ai_regret_bounded_information_portfolio",
        "policy_version": 1,
        "policy_definition_sha256": _sha("rbie-wrapper"),
    }
    result = SimpleNamespace(
        decision=SimpleNamespace(**decision_identity),
        supplemental_audit=SimpleNamespace(
            audit_kind="regret_bounded_information_portfolio_k8_to_k4",
            payload=_object(
                {
                    "schema_version": 5,
                    **decision_identity,
                    "allocator_policy": {
                        **allocator_identity,
                        "future_value_policy": {
                            "policy_id": "authenticated-extra-field"
                        },
                    },
                    "allocation": {
                        "schema_version": 1,
                        "policy_id": policy_id,
                        "policy_version": policy_version,
                        "policy_definition_sha256": definition_sha256,
                        "certificate_scope": (
                            "conditional_on_frozen_acquisition_calibration_not_sota"
                        ),
                        "reference_option_ids": ["anchor.1", "anchor.2"],
                        "selected_option_ids": ["anchor.1", "residual.1"],
                        "selected_future_value": {
                            "authority": "development_assay"
                        },
                    },
                }
            ),
        ),
    )

    assert _selector_policy_binding_valid(result)


def test_preregistration_contract_projects_bulky_ephemeral_transcript() -> None:
    stable_summary = {
        "status": "completed",
        "execution_mode": "deterministic_provider_free_calibrated_double",
        "scientific_claim": "structural_conformance_only",
        "generations_completed": 6,
        "candidate_occurrences": 38,
        "planned_candidate_occurrences": 38,
        "unique_evaluations": 38,
        "physical_evaluator_calls": 38,
        "evaluator_calls": 38,
        "provider_calls": 0,
        "docker_calls": 0,
        "logical_agent_calls": 7,
        "selector_calls": 6,
        "canonical_reflection_records": 1,
        "outcome_feedback_receipts": 6,
        "forecast_calibration_observations": 72,
        "authenticated_action_observations": 24,
        "bounded_g5_dose_request_count": 0,
        "bounded_g5_dose_result_count": 0,
        "bounded_g5_dose_assessments_pass": True,
        "typed_recourse_receipts": 1,
    }
    decision = {
        "request_sha256": "ephemeral-a",
        "decision": {
            "policy_id": "outcome_conditioned_expert_portfolio",
            "policy_version": 8,
            "members": [{}, {}, {}, {}],
        },
        "supplemental_audit": {
            "audit_kind": "outcome_conditioned_expert_portfolio",
            "payload": {
                "schema_version": 2,
                "phase": {
                    "campaign_generation": 1,
                    "portfolio_generation_ordinal": 1,
                    "remaining_portfolio_generations": 2,
                    "current_or_future_outcomes_consulted": False,
                },
                "allocation": {
                    "allocator_policy": {
                        "policy_id": "allocator",
                        "policy_version": 3,
                        "definition_sha256": "a" * 64,
                    }
                },
                "global_wave_allocation": {
                    "policy": {
                        "policy_id": "global_wave",
                        "policy_version": 6,
                        "definition_sha256": "b" * 64,
                    }
                },
                "forecast_universe_projection": {
                    "mode": "authenticated_outcome_blind_candidate_pool",
                    "outcomes_consulted": False,
                },
                "required_option_ids": [],
                "evidence_mode": "catalog_only",
                "physical_call_count": 7,
                "ephemeral_candidate_transcript": [
                    list(range(1_000)) for _ in range(51)
                ],
            },
        },
    }
    probe = {
        "schema_version": 1,
        "replicate_seed": 20260724,
        "all_gates_pass": True,
        "gates": {"six_generations": True},
        "archive_context_projections": [],
        "contextual_history_action_counts": [0, 0, 4, 4, 8, 8],
        "contextual_history_action_counts_by_cutoff": {"g1": 0, "g3": 4, "g5": 8},
        "g5_memory_path_audit": {"passes": True},
        "method": {"acquisition_mode": "horizon_bounded"},
        "protected_acquisition": {"enabled": True},
        "stage_candidate_occurrences": [8, 4, 8, 4, 8, 4],
        "recombination_candidate_occurrences": [4, 4, 4],
        "capacity_recourse_stages": [],
        "anchor_residual_identification": None,
        "summary": stable_summary,
        "selection_decisions": [decision],
        "outcome_conditioned_scope_probe": {
            "forecast_runner_calls": 42,
            "summary": stable_summary,
        },
    }

    with pytest.raises(ValueError, match="max_nodes"):
        freeze_json(probe)
    contract = _construction_probe_contract(probe)
    assert freeze_json(contract)
    assert "selection_decisions" not in contract
    assert contract["selection_construction"][0]["member_count"] == 4


def test_timeloop_preregistration_exposes_shared_method_identity() -> None:
    method_sha256 = "a" * 64
    experiment_sha256 = "b" * 64
    profile = {
        "method_definition_sha256": method_sha256,
        "experiment_definition_sha256": experiment_sha256,
        "method": {"workload_specific_fields": []},
    }
    conformance = {
        "pass": True,
        "method_definition_sha256": method_sha256,
        "experiment_definition_sha256": experiment_sha256,
    }

    assert _experiment_profile_preregistration_fields(
        {
            "summary": {
                "experiment_profile": profile,
                "experiment_profile_conformance": conformance,
            }
        }
    ) == {
        "experiment_profile": profile,
        "experiment_profile_conformance": conformance,
        "method_definition_sha256": method_sha256,
        "experiment_definition_sha256": experiment_sha256,
    }


def test_family_witness_is_jointly_embeddable_in_proposal_partition() -> None:
    options = tuple(
        SimpleNamespace(option_id=option_id, family=family)
        for option_id, family in (
            ("a2", "f3"),
            ("c0", "f0"),
            ("c1", "f1"),
            ("c2", "f2"),
            ("a3", "f4"),
            ("a4", "f5"),
            ("a0", "f6"),
            ("a1", "f7"),
        )
    )
    contract = SimpleNamespace(options=options, __post_init__=lambda: None)

    witness = _family_bounded_evaluation_witness(
        contract,
        tuple(option.option_id for option in options),
        portfolio_size=4,
        min_distinct_families=4,
        family_exposure_bounds=(),
        embedding_required_option_ids=("a0", "a1"),
        embedding_composite_option_ids=("c0", "c1", "c2"),
        embedding_composite_capacity=2,
        embedding_total_capacity=8,
    )

    assert witness == ("a2", "c0", "c1", "a3")


def test_runtime_selector_override_does_not_replace_profile_allocator() -> None:
    class _RuntimeSelector:
        def select(self) -> None:
            return None

    allocator = object()
    runtime_selector = _RuntimeSelector()

    bound = _profile_portfolio_selection_implementation(
        selected_allocator=allocator,  # type: ignore[arg-type]
        runtime_selector_override=runtime_selector,
    )

    assert bound is allocator


def test_reference_profile_rejects_missing_engine_allocator() -> None:
    with pytest.raises(
        ValueError,
        match="reference profile requires an engine-owned allocator",
    ):
        _profile_portfolio_selection_implementation(
            selected_allocator=None,
            runtime_selector_override=None,
        )


def test_operator_stratified_reference_transport_binds_exact_allocator() -> None:
    """Exercise the import-time production configuration in a clean process."""

    code = """
import json
from examples.development.run_timeloop_v2_provider_free_campaign import (
    run_provider_free_timeloop_campaign,
)
summary = run_provider_free_timeloop_campaign().summary()
profile = summary["experiment_profile"]
variation = summary["variation_trace"]
print(json.dumps({
    "status": summary["status"],
    "provider_calls": summary["provider_calls"],
    "docker_calls": summary["docker_calls"],
    "method_id": profile["method"]["method_id"],
    "method_version": profile["method"]["method_version"],
    "method_definition_sha256": profile["method_definition_sha256"],
    "portfolio_selection": profile["method"]["policies"]["portfolio_selection"],
    "composite_proposal_count": variation["composite_proposal_count"],
    "composite_evaluated_count": variation["composite_evaluated_count"],
    "exact_required_composite_call_rate": variation[
        "exact_required_composite_call_rate"
    ],
}, sort_keys=True))
"""
    environ = {
        **os.environ,
        "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": "task_keyed_common_pool",
        "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": "all",
        "AGENT_EVOLVE_ACQUISITION_MODE": "operator_stratified",
        "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": "authenticated_affine_v1",
        "AGENT_EVOLVE_VARIATION_TOPOLOGY": "hierarchical_r2",
        "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": "16",
        "AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS": "2",
        "AGENT_EVOLVE_OPERATOR_ASSAY_MINIMUM": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environ,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    record = json.loads(completed.stdout)

    assert record["status"] == "completed"
    assert record["provider_calls"] == 0
    assert record["docker_calls"] == 0
    assert record["method_id"] == "agent_evolve_operator_stratified_successor"
    assert record["method_version"] == 6
    assert record["method_definition_sha256"] == (
        "372d26c4f5d29fa0a7036e91b1fb18878ea4c9fe0e5c7c34ce2f352ab30c28a5"
    )
    assert record["portfolio_selection"]["policy_id"] == (
        "operator_stratified_hierarchical_k8_engine_k4"
    )
    assert record["composite_proposal_count"] == 12
    assert record["composite_evaluated_count"] == 6
    assert record["exact_required_composite_call_rate"] == 1.0


def test_horizon_bounded_reference_transport_uses_structural_recourse() -> None:
    """Prove the same finite-horizon method survives Timeloop's tighter topology."""

    code = """
import json
from agent_evolve.domain.typed_json import thaw_json
from examples.development.run_timeloop_v2_provider_free_campaign import (
    run_provider_free_timeloop_campaign,
)
run = run_provider_free_timeloop_campaign()
summary = run.summary()
profile = summary["experiment_profile"]
exposures = []
for request, result in run.selector.results:
    payload = thaw_json(result.supplemental_audit.payload)
    finite = payload["allocation"]["finite_horizon_exposure"]
    bound = finite["active_phase"]["bounds"][0]
    exposures.append({
        "wave_index": request.call_id.value,
        "requested_minimum": bound["minimum_evaluations"],
        "requested_maximum": bound["maximum_evaluations"],
        "applied": finite["applied_family_counts"]["composite_r2"],
        "violation": finite["family_exposure_violation_count"],
        "outcomes_consulted": finite["outcomes_consulted"],
    })
parent_sources = []
for receipt in run.execution.stage_receipts:
    selection = thaw_json(receipt.result).get("parent_selection")
    if selection is not None:
        parent_sources.append({
            "optimizer_generation": selection["optimizer_generation"],
            "stagnation_triggered": selection["stagnation_triggered"],
            "source_switch_applied": selection["source_switch_applied"],
            "source_mode": selection["source_mode"],
            "eligible_nonfront_history_count": (
                selection["eligible_nonfront_history_count"]
            ),
        })
print(json.dumps({
    "status": summary["status"],
    "provider_calls": summary["provider_calls"],
    "docker_calls": summary["docker_calls"],
    "method_id": profile["method"]["method_id"],
    "method_version": profile["method"]["method_version"],
    "method_definition_sha256": profile["method_definition_sha256"],
    "portfolio_selection": profile["method"]["policies"]["portfolio_selection"],
    "composite_proposal_count": summary["variation_trace"]["composite_proposal_count"],
    "composite_evaluated_count": summary["variation_trace"]["composite_evaluated_count"],
    "exposures": exposures,
    "parent_sources": parent_sources,
}, sort_keys=True))
"""
    environ = {
        **os.environ,
        "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": "task_keyed_common_pool",
        "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": "all",
        "AGENT_EVOLVE_ACQUISITION_MODE": "horizon_bounded",
        "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": "authenticated_affine_v1",
        "AGENT_EVOLVE_VARIATION_TOPOLOGY": "hierarchical_r2",
        "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": "16",
        "AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS": "2",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environ,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    record = json.loads(completed.stdout)

    assert record["status"] == "completed"
    assert record["provider_calls"] == 0
    assert record["docker_calls"] == 0
    assert record["method_id"] == "agent_evolve_stagnation_aware_successor"
    assert record["method_version"] == 8
    assert record["method_definition_sha256"] == (
        "2483021e3d8753916ca057924d92703a613bf65684e0ee60d6bd1eaee4576a4f"
    )
    assert record["portfolio_selection"]["policy_id"] == (
        "horizon_bounded_hierarchical_k8_engine_k4"
    )
    assert record["composite_proposal_count"] == 12
    assert record["composite_evaluated_count"] == 4
    assert [value["requested_minimum"] for value in record["exposures"]] == [
        2,
        2,
        2,
        2,
        0,
        0,
    ]
    assert [value["applied"] for value in record["exposures"]] == [1, 1, 1, 1, 0, 0]
    assert [value["violation"] for value in record["exposures"]] == [1, 1, 1, 1, 0, 0]
    assert all(value["outcomes_consulted"] is False for value in record["exposures"])
    assert [value["optimizer_generation"] for value in record["parent_sources"]] == [
        0,
        2,
        4,
    ]
    assert [value["stagnation_triggered"] for value in record["parent_sources"]] == [
        False,
        True,
        True,
    ]
    assert [value["source_switch_applied"] for value in record["parent_sources"]] == [
        False,
        True,
        True,
    ]
    assert [value["source_mode"] for value in record["parent_sources"]] == [
        "normal_diverse_elite",
        "stagnation_remote_history",
        "stagnation_remote_history",
    ]


def test_horizon_bounded_live_entrypoint_has_complete_protocol_identity() -> None:
    """Cover the production wrapper, not only its provider-free core."""

    code = """
import json
from examples.development.run_timeloop_v2_frontier_probe_live import (
    ACQUISITION_MODE,
    PARTITIONED_RETRY_BUDGET,
    PROTOCOL_ID,
    _acquisition_execution_label,
    _calibrated_allocator,
)
allocator = _calibrated_allocator()
print(json.dumps({
    "mode": ACQUISITION_MODE.value,
    "protocol_id": PROTOCOL_ID,
    "execution_label": _acquisition_execution_label(),
    "partitioned_retry": PARTITIONED_RETRY_BUDGET is not None,
    "allocator_policy_id": allocator.policy_id,
}, sort_keys=True))
"""
    environ = {
        **os.environ,
        "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": "task_keyed_common_pool",
        "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": "all",
        "AGENT_EVOLVE_ACQUISITION_MODE": "horizon_bounded",
        "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": "authenticated_affine_v1",
        "AGENT_EVOLVE_VARIATION_TOPOLOGY": "hierarchical_r2",
        "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": "16",
        "AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS": "2",
    }
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environ,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    record = json.loads(completed.stdout)
    assert record == {
        "allocator_policy_id": "horizon_bounded_calibrated_frontier_four_role_slate",
        "execution_label": "horizon_bounded_hierarchical_k8_to_k4",
        "mode": "horizon_bounded",
        "partitioned_retry": True,
        "protocol_id": "timeloop_v2_horizon_bounded_successor_g6_v16",
    }


@pytest.fixture(scope="module")
def completed_run():
    """Pay the full six-generation conformance cost only once per test module."""

    return run_provider_free_timeloop_campaign()


def test_timeloop_completes_the_delayed_g6_composition(completed_run) -> None:
    run = completed_run
    counters = run.execution.counters
    memory_path_audit = _g5_memory_path_audit(
        run.summary(),
        reflection_receipts=run.execution.reflection_receipts,
    )

    assert counters.generations_completed == 6
    assert counters.candidate_occurrences == 62
    assert counters.unique_evaluations == run.evaluator.calls == 60
    assert counters.logical_agent_calls == 7
    assert counters.logical_agent_calls_dispatched_to_runtime == 7
    assert counters.logical_agent_calls_succeeded == 7
    assert counters.logical_agent_calls_failed == 0
    assert len(run.selector.results) == 6
    assert run.calibrated_runner.calls == 6
    assert len(run.feedback_ledger.receipts) == 6
    assert len(run.feedback_ledger.observations) == 144

    assert run.reflection_executor.generations == [2]
    assert len(run.execution.reflection_receipts) == 1
    assert len(run.execution.test_admission_receipts) == 1
    assert run.execution.tail_drain_receipt is None
    assert len(run.memory.entries) == 9
    # One two-lane M/N block is experimental evidence, not an identified
    # card-level effect.  It must never enter the adaptive memory score path.
    assert len(run.memory.trials) == 0
    assert all(
        entry.lifecycle_state is not InsightLifecycleState.PROMOTED
        for entry in run.memory.entries[1:]
    )
    assert memory_path_audit == {
        "schema_version": 2,
        "reflection_receipt_count": 1,
        "completed_reflection_receipt_count": 1,
        "abstained_reflection_receipt_count": 0,
        "failed_reflection_receipt_count": 0,
        "typed_e0_receipt_authenticated": False,
        "e1_reflection_publication_valid": True,
        "active_neutral_assay_realized": True,
        "typed_no_shared_support_recourse_realized": False,
        "typed_e0_memory_free_recourse_realized": False,
        "reflection_path_valid": True,
        "workflow_path_valid": True,
        "expected_physical_reflection_provider_calls": 1,
        "memory_effect_claim_available": True,
    }

    assert [record["generation"] for record in run.wave_factory.wave_records] == [
        1,
        1,
        3,
        3,
        5,
        5,
    ]
    assert [record["status"] for record in run.wave_factory.wave_records] == [
        "bootstrap_prior",
        "bootstrap_prior",
        "delayed_reflection_not_yet_admitted",
        "delayed_reflection_not_yet_admitted",
        "applied_randomized_active_neutral_arm",
        "applied_randomized_active_neutral_arm",
    ]

    observations = run.evidence_registry.observations
    assert len(observations) == 48
    assert {value.event_index for value in observations} == {1, 3, 5}
    assert all(
        value.provenance is EvidenceProvenance.DIRECT_MUTATION for value in observations
    )
    assert sum(value.event_index == 1 for value in observations) == 16

    records = tuple(
        CampaignReflectionLearningRecordCodec.decode(value)
        for value in run.reflection_executor.records
    )
    assert len(records) == 1
    learning = records[0]
    assert learning.source_generation == 2
    assert learning.origin_cutoff_event_index == 1
    assert learning.insight_contract.is_semantic_v3
    assert learning.insight_contract.required_metric_ids == OBJECTIVE_IDS
    assert (
        learning.insight_contract.allowed_option_families == REFLECTION_OPTION_FAMILIES
    )
    assert learning.insight_contract.allowed_decision_paths == REFLECTION_DECISION_PATHS
    assert len(learning.empirical_evidence) == 16
    assert len(learning.finite_action_bindings) == 16
    assert learning.evidence_catalog.contrast_ids == tuple(
        item.contrast_id for item in learning.empirical_evidence
    )

    summary = run.summary()
    expected_summary_subset = {
        "status": "completed",
        "planned_candidate_occurrences": 62,
        "candidate_occurrences": 62,
        "unique_evaluations": 60,
        "physical_evaluator_calls": 60,
        "logical_agent_calls": 7,
        "provider_calls": 0,
        "docker_calls": 0,
        "scientific_claim": "structural_conformance_only",
    }
    assert {
        key: summary[key] for key in expected_summary_subset
    } == expected_summary_subset
    assert summary["outcome_feedback_receipts"] == 6
    assert summary["forecast_calibration_observations"] == 144
    assert summary["memory_trials"] == 0
    assert summary["diagnostic_compatibility_audits"] == []
    assert summary["diagnostic_memory_blocks"] == []
    assert summary["diagnostic_cohort_selections"] == []
    assert len(summary["matched_memory_support_resolutions"]) == 1
    support_resolution = summary["matched_memory_support_resolutions"][0]
    assert support_resolution["selected_card_key"] is not None
    assert len(support_resolution["selected_lane_supports"]) == 2
    assert support_resolution["provider_fields_consulted"] is False
    assert support_resolution["outcome_values_consulted"] is False
    assert support_resolution["card_vs_neutral_effect_identified"] is False
    assert len(summary["matched_memory_control_plans"]) == 1
    matched_plan = summary["matched_memory_control_plans"][0]
    assert {value["arm"] for value in matched_plan["assignments"]} == {"m", "n"}
    assert matched_plan["provider_and_outcome_blind_assignment"] is True
    assert matched_plan["single_block_card_effect_identified"] is False
    assert matched_plan["online_score_update_allowed"] is False

    portfolio_stage_records = tuple(
        thaw_json(receipt.result)
        for receipt in run.execution.stage_receipts
        if receipt.kind.value == "portfolio"
    )
    assert len(portfolio_stage_records) == 3
    assert all(
        value["context_enrichment_applied"] is True for value in portfolio_stage_records
    )
    assert all(
        value["memory_projection_updated"] is True
        and value["outcome_update_preparation"] is not None
        for value in portfolio_stage_records
    )
    assert [
        value["outcome_update_preparation"]["evidence"]["ledger_receipt_count_after"]
        for value in portfolio_stage_records
    ] == [2, 4, 6]
    g5_learning = portfolio_stage_records[2]["closed_loop_learning"]
    assert g5_learning["evidence"]["status"] == (
        "evidence_append_prepared_no_diagnostic_assignment"
    )
    g5_audit_projection = g5_learning["evidence"]["generation_audit_preparation"][
        "projection"
    ]
    assert g5_audit_projection is None
    assert portfolio_stage_records[2]["memory_credit_batch"] is None
    matched_outcomes = g5_learning["evidence"]["generation_audit_preparation"][
        "matched_memory_control_outcomes"
    ]
    assert len(matched_outcomes) == 1
    assert matched_outcomes[0]["single_block_card_effect_identified"] is False
    assert matched_outcomes[0]["online_score_update_allowed"] is False
    assert matched_outcomes[0]["analysis_scope"] == (
        "append_only_arm_aware_experimental_observation"
    )

    calibration_observation_counts = []
    calibration_observations_by_g5_arm = {}
    contextual_history_action_counts = []
    for request, result in run.selector.results:
        audit = result.supplemental_audit
        assert audit is not None
        payload = thaw_json(audit.payload)
        calibration_observation_counts.append(
            payload["allocation"]["request"]["calibration_snapshot"]["summary"][
                "observation_count"
            ]
        )
        if request.memory_dose_contract is not None:
            calibration_observations_by_g5_arm["active"] = (
                calibration_observation_counts[-1]
            )
        elif request.experimental_view_receipt is not None:
            calibration_observations_by_g5_arm["neutral"] = (
                calibration_observation_counts[-1]
            )
    for request, _ in run.selector.results:
        contextual_history = thaw_json(request.context)["campaign_contextual_history"]
        contextual_history_action_counts.append(len(contextual_history["actions"]))
        assert all(
            action["transfer_scope"] == "same_lineage"
            for action in contextual_history["actions"]
        )

    # Forecast calibration is prompt-treatment-specific.  The G5 neutral arm
    # retains the ordinary prompt definition and can consume its prior model-
    # behaviour observations; the active bounded-dose arm cold-starts its
    # distinct scope.  This is consequently a deployed-package contrast, not
    # a prose-only card effect.
    assert calibration_observation_counts[:4] == [0, 0, 48, 48]
    assert sorted(calibration_observation_counts[4:]) == [0, 96]
    assert calibration_observations_by_g5_arm == {"active": 0, "neutral": 96}
    assert contextual_history_action_counts == [0, 0, 8, 7, 8, 8]


def test_timeloop_reflection_excludes_ambient_and_recombination_data(
    completed_run,
) -> None:
    run = completed_run
    reflection_input = run.reflection_executor.inputs[0]
    input_record = reflection_input.to_record()
    prompt = run.reflection_executor.prompts[0]
    contrasts = reflection_input.evidence.contrasts
    observations = run.evidence_registry.observations

    assert input_record["source_stage_payload_exposed"] is False
    assert input_record["recombination_results_exposed"] is False
    assert reflection_input.query.wave.source_generation == 2
    assert reflection_input.query.wave.promotion_barrier_generation == 4
    assert reflection_input.query.prior_cutoff_event_index_exclusive == 0
    assert reflection_input.query.sealed_cutoff_event_index_inclusive == 1
    assert len(contrasts) == 16
    assert {value.event_index for value in contrasts} == {1}

    sealed_g1_hashes = {
        value.observation_sha256 for value in observations if value.event_index == 1
    }
    ambient_hashes = {
        value.observation_sha256
        for value in observations
        if value.event_index in {3, 5}
    }
    assert {value.source_observation_sha256 for value in contrasts} == (
        sealed_g1_hashes
    )
    assert sealed_g1_hashes.isdisjoint(ambient_hashes)
    assert "recombination" not in prompt.casefold()
    assert all(value.contrast_id not in prompt for value in contrasts)
    assert all(value not in prompt for value in ambient_hashes)

    learning = CampaignReflectionLearningRecordCodec.decode(
        run.reflection_executor.records[0]
    )
    assert set(learning.source_operator_invocation_ids) == {
        value.operator_invocation_id for value in contrasts
    }
    assert set(learning.source_candidate_ids) == {
        candidate_id
        for value in contrasts
        for candidate_id in (value.parent_candidate_id, value.child_candidate_id)
    }


def test_timeloop_randomizes_one_active_and_one_neutral_g5_lane(
    completed_run,
) -> None:
    run = completed_run
    proposal_records = run.calibrated_runner.records

    assert len(proposal_records) == 6
    assert all(value["proposal_width"] == 8 for value in proposal_records)
    bounded_proposals = tuple(
        value for value in proposal_records if value["bounded_memory_dose"]
    )
    assert len(bounded_proposals) == 1
    for value in bounded_proposals:
        supports = value["proposal_supporting_card_keys"]
        assert len(supports) == 8
        assert supports[0] == value["assigned_card_keys"]
        assert all(not member for member in supports[1:])

    assert run.wave_factory.matching_receipts == []
    assert len(run.wave_factory.recourse_receipts) == 0
    assert len(run.wave_factory.dose_contracts) == 1
    for dose in run.wave_factory.dose_contracts:
        assert dose.proposed_supported_member_bounds == (1, 1)
        assert dose.evaluated_supported_member_bounds == (1, 1)
        assert dose.minimum_unattributed_proposed_members == 7
        assert dose.minimum_unattributed_evaluated_members == 7
        assert dose.maximum_cards_per_member == 1
        assert dose.require_every_assigned_card

    bounded_results = tuple(
        result
        for request, result in run.selector.results
        if request.memory_dose_contract is not None
    )
    assert len(bounded_results) == 1
    for result in bounded_results:
        assessment = result.decision.memory_dose_assessment
        assert assessment is not None
        assert assessment.passed
        assert assessment.supported_member_ranks == (1,)
        assert assessment.unattributed_member_ranks == (2, 3, 4, 5, 6, 7, 8)
        assert assessment.proposal_assessment_sha256 is not None
        assert (
            assessment.to_record()["unattributed_members_are_blinded_controls"] is False
        )

    g5_records = tuple(
        record for record in run.wave_factory.wave_records if record["generation"] == 5
    )
    assert len(g5_records) == 2
    assert {record["evidence"]["experimental_arm"] for record in g5_records} == {
        "m",
        "n",
    }
    assert all(
        record["evidence"]["memory_credit_issued"] is False
        and record["evidence"]["matched_control_outcome_pending"] is True
        for record in g5_records
    )
    assert len(run.memory.trials) == 0
    assert run.wave_factory.compatibility_audits == []
    assert run.wave_factory.diagnostic_blocks == []
    assert len(run.wave_factory.matched_support_resolutions) == 1
    assert run.wave_factory.matched_support_resolutions[0].eligible
    assert len(run.wave_factory.matched_control_plans) == 1
    plan = run.wave_factory.matched_control_plans[0]
    assert {value.arm.value for value in plan.assignments} == {"m", "n"}


def test_timeloop_singleton_reflection_cohort_can_span_matched_control_lanes(
    monkeypatch,
) -> None:
    """One supported card is sufficient for a two-lane active/neutral block.

    Lane count controls randomized experimental units, not the minimum number
    of distinct hypotheses.  Numerical/global proposals can legitimately
    leave only one identifiable local contrast; rejecting that cohort would
    couple the memory lifecycle to a particular proposal mixture.
    """

    from examples.development import run_timeloop_v2_provider_free_campaign as module

    original_cluster = module.cluster_identifiable_mutation_reflection_hypotheses
    monkeypatch.setattr(
        module,
        "cluster_identifiable_mutation_reflection_hypotheses",
        lambda contrasts: original_cluster(contrasts)[:1],
    )
    run = run_provider_free_timeloop_campaign()

    assert len(run.reflection_executor.records) == 1
    reflection = CampaignReflectionLearningRecordCodec.decode(
        run.reflection_executor.records[0]
    )
    assert len(reflection.insights) == 1
    assert len(run.wave_factory.matched_control_plans) == 1
    assert len(run.wave_factory.recourse_receipts) == 0
    assert len(run.wave_factory.matched_support_resolutions) == 1
    assert run.wave_factory.matched_support_resolutions[0].eligible
    assert {
        value.arm.value
        for value in run.wave_factory.matched_control_plans[0].assignments
    } == {"m", "n"}
    assert {
        record["status"]
        for record in run.wave_factory.wave_records
        if record["generation"] == 5
    } == {"applied_randomized_active_neutral_arm"}


def test_provider_free_common_pool_preserves_required_memory_dose_action(
    completed_run,
) -> None:
    """The exact K8 pool may not place its hard-dose action at rank one."""

    request = next(
        request
        for request, _ in completed_run.selector.results
        if request.memory_dose_contract is not None
    )
    output_without_pool = SimpleNamespace(
        finite_variation_contract=request.finite_variation_contract,
        ordered_common_pool_option_ids=None,
        memory_dose_contract=request.memory_dose_contract,
    )
    baseline = _ProviderFreeCalibratedRunner._proposal_options(output_without_pool)
    rotated_pool = baseline[1:] + baseline[:1]
    output_with_pool = SimpleNamespace(
        finite_variation_contract=request.finite_variation_contract,
        ordered_common_pool_option_ids=tuple(
            option.option_id for option in rotated_pool
        ),
        memory_dose_contract=request.memory_dose_contract,
    )

    selected = _ProviderFreeCalibratedRunner._proposal_options(output_with_pool)

    support = request.memory_dose_contract.card_supports[0]
    assert {option.option_id for option in selected} == {
        option.option_id for option in rotated_pool
    }
    assert support.supports(
        selected[0].option_id,
        selected[0].identity_sha256,
    )


def test_timeloop_contradictory_reflection_fails_before_memory_recourse() -> None:
    # This fixture deliberately changes the declared option family while
    # retaining the authenticated action binding.  It is not a valid but
    # unsupported card: the semantic contract must reject it before G5.  The
    # generic valid-card/no-shared-support recourse is covered by Heat2D.
    run = run_provider_free_timeloop_campaign(incompatible_reflection_card=True)

    assert run.summary()["status"] == "degraded"
    assert len(run.execution.reflection_receipts) == 1
    receipt = run.execution.reflection_receipts[0]
    assert receipt.status.value == "failed"
    failure = thaw_json(receipt.quarantined_result)
    assert failure["status"] == "failed"
    assert failure["failure_type"] == "ValueError"
    assert failure["publishable_reflection_content"] is False
    assert run.wave_factory.matched_support_resolutions == []
    assert run.wave_factory.matched_control_plans == []
    assert run.wave_factory.recourse_receipts == []


class _OneCandidateInfeasibleDetailedEvaluator:
    """Inject one deterministic candidate failure through the public port."""

    evaluator_identity = EvaluatorIdentity(
        evaluator_id="timeloop_test_one_candidate_infeasible",
        evaluator_version=1,
        evaluator_context_sha256=_sha("timeloop-test-one-candidate-infeasible"),
    )

    def __init__(self, delegate: _DeterministicEvaluator) -> None:
        self.delegate = delegate
        self.calls = 0
        self.infeasible_calls = 0

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        self.calls += 1
        if self.calls == 3:
            self.infeasible_calls += 1
            return DetailedEvaluationPayload(
                failure=FailureRecord(
                    category=FailureCategory.CANDIDATE,
                    code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
                    message="deterministic Timeloop candidate-infeasibility probe",
                ),
                objectives=(),
                violations=(),
                checks=(),
                receipt=None,
                evaluator=self.evaluator_identity,
            )
        observation = self.delegate.evaluate(configuration)
        return DetailedEvaluationPayload(
            failure=None,
            objectives=tuple(
                (name, float(observation.objective_values[name]))
                for name in OBJECTIVE_NAMES
            ),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


def _run_candidate_infeasibility_probe(tmp_path: Path):
    panel = frozen_network_panel("resnet50")
    raw_evaluator = _DeterministicEvaluator()
    problem = TimeloopV2CoDesignProblem(
        TimeloopV2Settings(output_root=tmp_path / "unused"),
        panel,
        evaluator=raw_evaluator,
    )
    detailed = _OneCandidateInfeasibleDetailedEvaluator(raw_evaluator)
    benchmark = AgenticBenchmark(
        problem=problem,
        detailed_evaluator=detailed,
        outcome_relation=objective_pareto_outcome_binding(tuple(problem.objectives)),
        optimization_semantics=timeloop_v2_optimization_semantics(problem),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        finite_variation_catalogs=(TimeloopV2FiniteVariationCatalog(panel),),
    )
    return run_timeloop_campaign(
        benchmark=benchmark,
        evaluator=detailed,
        execution_mode="provider_free_candidate_infeasibility_probe",
        id_namespace="timeloop_v2_g6_infeasible",
        campaign_sha256=_sha("timeloop-g6-candidate-infeasible-campaign"),
        evaluator_contract_sha256=_sha("timeloop-g6-candidate-infeasible-evaluator"),
        protocol_id="timeloop_v2_provider_free_g6_candidate_recourse",
        protocol_definition_sha256=_sha(
            "timeloop-v2-provider-free-g6-candidate-recourse-v1"
        ),
        task_sha256=_sha("timeloop-v2-provider-free-g6-candidate-recourse-task"),
        evaluator_preflight_receipt=_object(
            {"qualified": True, "mode": "candidate_infeasibility_probe"}
        ),
        resource_lease_receipt=_object(
            {"resource": "provider_free_timeloop_test_slot", "active": True}
        ),
        docker_enabled=False,
        scientific_claim="candidate_recourse_conformance_only",
    )


def test_timeloop_candidate_infeasibility_does_not_abort_learning(
    tmp_path: Path,
) -> None:
    run = _run_candidate_infeasibility_probe(tmp_path)
    evaluator = run.evaluator

    assert run.summary()["status"] == "completed"
    assert run.execution.counters.generations_completed == 6
    assert run.execution.counters.candidate_occurrences == 62
    assert run.execution.counters.logical_agent_calls == 7
    assert run.execution.counters.unique_evaluations == evaluator.calls
    assert evaluator.infeasible_calls == 1
    assert len(run.evidence_registry.observations) == 47
    assert len(run.reflection_executor.inputs) == 1
    assert len(run.reflection_executor.inputs[0].evidence.contrasts) >= 1
    assert {
        value.event_index
        for value in run.reflection_executor.inputs[0].evidence.contrasts
    } == {1}
    dose_count = len(run.wave_factory.dose_contracts)
    assert dose_count in {0, 1}
    assert len(run.wave_factory.recourse_receipts) == (1 if dose_count == 0 else 0)
    assert run.summary()["bounded_g5_dose_assessments_pass"] is True
    assert _portfolio_candidate_infeasible_count(run) == 1


def test_static_candidate_infeasibility_is_terminal_without_a_simulator_call() -> None:
    event = {
        "event_type": "candidate_evaluated",
        "valid": False,
        "detailed_evaluation": {
            "failure": {
                "category": "candidate",
                "code": "evaluator_declared_infeasible",
            },
            "checks": [
                {
                    "name": "native_simulator_invocation",
                    "status": "not_applicable",
                    "observed_value": {"native_simulator_invoked": False},
                }
            ],
        },
    }

    events = _typed_candidate_infeasible_events([event])

    assert events == (event,)
    assert _pre_simulator_infeasible_count(events) == 1

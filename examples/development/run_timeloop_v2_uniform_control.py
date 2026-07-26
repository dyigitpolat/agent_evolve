#!/usr/bin/env python3
"""Prepare or execute a scale-matched provider-free Timeloop G6 control.

The control derives its candidate and portfolio budgets from the treatment,
then uses the same scheduler, seeds, parent policy, outcome-blind eligibility
boundary, serial pinned Docker evaluator, recombination implementation, and
affine endpoint.  At each portfolio wave it samples the treatment evaluation
width directly and conditionally uniformly from the complete eligible finite
contract.  It never reads an API credential or invokes a model; reflections
are retained as local diagnostics but are not exposed to the direct control
wave factory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.domain.ids import validate_id_namespace  # noqa: E402
from agent_evolve.domain.typed_json import thaw_json  # noqa: E402
from agent_evolve.infrastructure.artifacts.filesystem import (  # noqa: E402
    FileSystemArtifactStore,
)
from agent_evolve.policies.reward.affine_hypervolume_3d import (  # noqa: E402
    AffineHypervolumeArchiveUtility3D,
    AffineHypervolumeSnapshot3D,
)
from examples.benchmarks.timeloop_codesign.v2.detailed_evaluation import (  # noqa: E402
    compose_timeloop_v2_detailed_benchmark,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (  # noqa: E402
    PINNED_IMAGE_ID,
    PINNED_IMAGE_REF,
    TimeloopV2DockerEvaluator,
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (  # noqa: E402
    frozen_network_panel,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    source_identity,
    write_json_atomic,
)
from examples.development.run_timeloop_v2_frontier_probe_live import (  # noqa: E402
    CPU_SET,
    EVALUATOR_TIMEOUT_S,
    GENERATION_COUNT,
    PLANNED_CANDIDATE_OCCURRENCES,
    PLANNED_LOGICAL_CALLS,
    TASK_SHA256,
    _CountingEvaluator,
    _ExecutionJournal,
    _RecombinationUtilityBinder,
    _load_gate_a_baseline,
    _load_real_qualification_evidence,
    _object,
    _open_journals,
    _pareto_front,
    _require_source_closure,
    _snapshot_sources,
    _source_paths,
    _strictly_dominates,
    _utility_spec,
)
from examples.development.run_timeloop_v2_provider_free_campaign import (  # noqa: E402
    PORTFOLIO_WIDTH,
    run_provider_free_timeloop_campaign,
    run_timeloop_campaign,
)
from examples.development.uniform_feasible_portfolio_control import (  # noqa: E402
    POLICY_DEFINITION_SHA256,
    POLICY_ID,
    POLICY_VERSION,
    TaskKeyedConditionalUniformPortfolioPolicy,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/timeloop_codesign/full_support_g6/matched_uniform_control"
)
CONTROL_SCALE_ID = f"k{PORTFOLIO_WIDTH}_n{PLANNED_CANDIDATE_OCCURRENCES}"
DIRECT_SELECTION_COUNT = 6
AUTHENTICATED_ACTION_OBSERVATION_COUNT = DIRECT_SELECTION_COUNT * PORTFOLIO_WIDTH
PROTOCOL_ID = f"timeloop_v2_matched_conditional_uniform_g6_v7_{CONTROL_SCALE_ID}"
PROTOCOL_DEFINITION_SHA256 = hashlib.sha256(
    (
        "agent-evolve:timeloop-v2-matched-conditional-uniform-g6:v7;"
        "same-scheduler-seeds-parents-evaluator-recombination-budget;"
        f"direct-k{PORTFOLIO_WIDTH}-over-complete-eligible-finite-contract;"
        f"candidate-occurrences={PLANNED_CANDIDATE_OCCURRENCES};"
        "provider-free=true;learned-memory-exposure=false;"
        "reference-envelope-gate=true;prepare-live-id-namespace-parity=true;"
        "unique-normalized-reflection-claims=true;canonical-summary-keys=true"
    ).encode("ascii")
).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_sha256(replicate_seed: int) -> str:
    return hashlib.sha256(
        f"agent-evolve:timeloop-v2-matched-uniform-control:v7:{CONTROL_SCALE_ID}\x00".encode(
            "ascii"
        )
        + replicate_seed.to_bytes(16, "big", signed=True)
    ).hexdigest()


def _id_namespace(replicate_seed: int) -> str:
    value = f"tlv2_uniform_{replicate_seed}_{CONTROL_SCALE_ID}_v7"
    validate_id_namespace(value)
    return value


def _policy(replicate_seed: int) -> TaskKeyedConditionalUniformPortfolioPolicy:
    return TaskKeyedConditionalUniformPortfolioPolicy(
        task_sha256=TASK_SHA256,
        replicate_seed=replicate_seed,
    )


def _construction_probe(replicate_seed: int) -> dict[str, object]:
    utility = AffineHypervolumeArchiveUtility3D(_utility_spec())
    run = run_provider_free_timeloop_campaign(
        outer_seed=replicate_seed,
        id_namespace=_id_namespace(replicate_seed),
        direct_portfolio_selector=_policy(replicate_seed),
        archive_utility=utility,
        recombination_utility_binder=_RecombinationUtilityBinder(utility),
    )
    summary = run.summary()
    selected = [
        [member.option_id for member in result.decision.members]
        for _, result in run.selector.results
    ]
    gates = {
        "six_generations": summary["generations_completed"] == GENERATION_COUNT,
        "candidate_occurrences_exact": (
            summary["candidate_occurrences"] == PLANNED_CANDIDATE_OCCURRENCES
        ),
        "seven_scheduler_decisions": (
            summary["logical_agent_calls"] == PLANNED_LOGICAL_CALLS
        ),
        "six_direct_scale_matched_selections": (
            summary["selector_calls"] == DIRECT_SELECTION_COUNT
            and summary["direct_portfolio_selections"] == DIRECT_SELECTION_COUNT
            and summary["proposal_width"] == PORTFOLIO_WIDTH
            and summary["k8_typed_proposals"]
            == (DIRECT_SELECTION_COUNT if PORTFOLIO_WIDTH == 8 else 0)
            and len(selected) == DIRECT_SELECTION_COUNT
            and all(len(value) == PORTFOLIO_WIDTH for value in selected)
        ),
        "one_local_diagnostic_reflection": (summary["reflection_generations"] == [2]),
        "authenticated_mutation_count_exact": (
            summary["authenticated_action_observations"]
            == AUTHENTICATED_ACTION_OBSERVATION_COUNT
        ),
        "no_calibrated_feedback_or_memory_dose": (
            summary["outcome_feedback_receipts"] == 0
            and summary["forecast_calibration_observations"] == 0
            and summary["bounded_g5_dose_request_count"] == 0
            and summary["memory_trials"] == 0
        ),
        "provider_and_docker_free": (
            summary["provider_calls"] == 0 and summary["docker_calls"] == 0
        ),
    }
    return {
        "schema_version": 1,
        "replicate_seed": replicate_seed,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "summary": summary,
        "selected_option_ids": selected,
        "policy": {
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "policy_definition_sha256": POLICY_DEFINITION_SHA256,
        },
        "credential_read": False,
        "provider_calls": 0,
        "candidate_docker_evaluations": 0,
    }


def _preregistration(
    *,
    replicate_seed: int,
    source_sha256: str,
    probe: dict[str, object],
    baseline: dict[str, object],
    qualification: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": (
            f"timeloop_v2_matched_uniform_g6_seed_{replicate_seed}_"
            f"{CONTROL_SCALE_ID}_v7"
        ),
        "claim_boundary": (
            "prospective_matched_provider_free_uninformed_control;"
            "paired_efficacy_requires_sealed_treatment;not_sota"
        ),
        "replicate_seed": replicate_seed,
        "outer_seed": replicate_seed,
        "source_aggregate_sha256": source_sha256,
        "protocol_id": PROTOCOL_ID,
        "protocol_definition_sha256": PROTOCOL_DEFINITION_SHA256,
        "campaign_sha256": _campaign_sha256(replicate_seed),
        "task_sha256": TASK_SHA256,
        "schedule": {
            "generations": GENERATION_COUNT,
            "candidate_occurrences": PLANNED_CANDIDATE_OCCURRENCES,
            "scheduler_local_decisions": PLANNED_LOGICAL_CALLS,
            "actual_llm_calls": 0,
            "selector_calls": 6,
            "local_diagnostic_reflections": 1,
            "evaluation_width": PORTFOLIO_WIDTH,
            "portfolio_generations": [1, 3, 5],
            "recombination_generations": [2, 4, 6],
        },
        "selection_policy": {
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "definition_sha256": POLICY_DEFINITION_SHA256,
            "sampling_space": "complete_outcome_blind_eligible_finite_contract",
            "proposal": f"conditional_uniform_feasible_k{PORTFOLIO_WIDTH}",
            "forbidden": [
                "objective_values",
                "prior_outcomes",
                "option_prose",
                "memory_scores",
                "reflections",
            ],
        },
        "treatment_parity": {
            "same_two_seed_configurations": True,
            "same_outer_seed": True,
            "same_generation_and_candidate_budget": True,
            "same_parent_policy": True,
            "same_outcome_blind_known_phenotype_exclusion": True,
            "same_recombination_runtime": True,
            "same_serial_evaluator_and_cache": True,
            "same_affine_endpoint": True,
            "different_candidate_generation_and_memory_exposure": True,
        },
        "utility": _utility_spec().to_record(),
        "utility_definition_sha256": _utility_spec().definition_sha256,
        "evaluator": {
            "image_ref": PINNED_IMAGE_REF,
            "image_id": PINNED_IMAGE_ID,
            "cpu_set": CPU_SET,
            "timeout_s_hex": float(EVALUATOR_TIMEOUT_S).hex(),
            "network_panel": "resnet50",
            "external_concurrency": 1,
        },
        "health_gates": {
            "provider_free_construction_probe_passes": True,
            "reference_qualification_envelope_passes": True,
            "source_closure_unchanged": True,
            "pinned_docker_preflight_passes": True,
            "credential_reads_equal_zero": True,
            "provider_calls_equal_zero": True,
            "campaign_completes_six_generations": True,
            "candidate_occurrences_exact": True,
            "six_direct_scale_matched_selections": True,
            "all_candidate_outcomes_terminal_and_typed": True,
            "all_successful_objectives_inside_fixed_reference": True,
            "runtime_cleanup_released": True,
        },
        "candidate_quality_endpoints": {
            "primary": "final_all_evaluated_affine_3d_hypervolume",
            "paired_primary": "treatment_minus_control_final_affine_3d_hv",
            "secondary": [
                "anytime_hypervolume_by_unique_evaluation",
                "hypervolume_gain_over_two_seeds",
                "nondominated_front_size",
                "evaluation_wall_time",
            ],
        },
        "historical_gate_a": baseline,
        "real_qualification_evidence": qualification,
        "provider_free_probe_sha256": hashlib.sha256(
            json.dumps(
                probe,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
    }


def _validate_preregistration(
    *,
    path: Path,
    replicate_seed: int,
    source_sha256: str,
    probe: dict[str, object],
    baseline: dict[str, object],
    qualification: dict[str, object],
) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    payload = resolved.read_bytes()
    value = json.loads(payload.decode("utf-8", errors="strict"))
    expected = _preregistration(
        replicate_seed=replicate_seed,
        source_sha256=source_sha256,
        probe=probe,
        baseline=baseline,
        qualification=qualification,
    )
    if value != expected:
        raise RuntimeError("control preregistration differs from prepared contract")
    return {
        "path": resolved.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _execute_control(
    *,
    run_dir: Path,
    replicate_seed: int,
    source_sha256: str,
    docker_preflight: dict[str, object],
    reference_qualification_passed: bool,
) -> dict[str, object]:
    started_ns = time.perf_counter_ns()
    raw_journals, journals = _open_journals(run_dir, started_ns)
    try:
        settings = TimeloopV2Settings(
            output_root=run_dir / "evaluator_calls",
            cpu_set=CPU_SET,
            timeout_s=EVALUATOR_TIMEOUT_S,
        )
        panel = frozen_network_panel("resnet50")
        evaluator = _CountingEvaluator(
            TimeloopV2DockerEvaluator(settings, panel),
            journals["evaluations"],
        )
        benchmark = compose_timeloop_v2_detailed_benchmark(
            settings,
            panel,
            artifact_store=FileSystemArtifactStore(run_dir / "artifact_store"),
            evaluator=evaluator,
        )
        detailed = benchmark.detailed_evaluator
        if detailed is None:
            raise RuntimeError("Timeloop control omitted detailed evidence")
        utility = AffineHypervolumeArchiveUtility3D(_utility_spec())
        run = run_timeloop_campaign(
            benchmark=benchmark,
            evaluator=evaluator,
            execution_mode="real_docker_conditional_uniform_control_g6",
            id_namespace=_id_namespace(replicate_seed),
            campaign_sha256=_campaign_sha256(replicate_seed),
            evaluator_contract_sha256=(
                detailed.evaluator_identity.evaluator_context_sha256
            ),
            protocol_id=PROTOCOL_ID,
            protocol_definition_sha256=PROTOCOL_DEFINITION_SHA256,
            task_sha256=TASK_SHA256,
            evaluator_preflight_receipt=_object(
                {
                    "qualified": True,
                    "mode": "real_docker_conditional_uniform_control_g6",
                    "preflight": docker_preflight,
                }
            ),
            resource_lease_receipt=_object(
                {
                    "resource": "serial_timeloop_docker_cpu_8",
                    "active": True,
                    "evaluator_concurrency": 1,
                }
            ),
            docker_enabled=True,
            scientific_claim="prospective_matched_uninformed_control",
            outer_seed=replicate_seed,
            direct_portfolio_selector=_policy(replicate_seed),
            selector_policy_binding_id=(
                f"conditional_uniform_complete_finite_k{PORTFOLIO_WIDTH}"
            ),
            reflection_policy_binding_id="provider_free_control_diagnostic_only",
            provider_enabled=False,
            execution_journal=_ExecutionJournal(journals["campaign"]),
            engine_trace_sink=lambda value: journals["engine"].append(dict(value)),
            archive_utility=utility,
            recombination_utility_binder=_RecombinationUtilityBinder(utility),
        )
        base_summary = run.summary()
        for request, result in run.selector.results:
            journals["responses"].append(
                {
                    "record_kind": "provider_free_control_selection",
                    "request_sha256": request.request_sha256,
                    "decision": result.decision.to_audit_record(),
                    "telemetry": {
                        "requested_model": result.telemetry.requested_model,
                        "resolved_model": result.telemetry.resolved_model,
                        "resolved_provider": result.telemetry.resolved_provider,
                        "input_tokens": result.telemetry.input_tokens,
                        "output_tokens": result.telemetry.output_tokens,
                        "reasoning_tokens": result.telemetry.reasoning_tokens,
                        "cost_usd": str(result.telemetry.cost_usd),
                    },
                }
            )
        for record in run.reflection_executor.records:
            journals["reflections"].append(
                {
                    "record_kind": "provider_free_control_diagnostic_reflection",
                    "learning_envelope": thaw_json(record),
                }
            )

        successes = [
            dict(value["objectives"])
            for value in evaluator.observations
            if value["status"] == "passed"
        ]
        runtime_failures = [
            value
            for value in evaluator.observations
            if value["status"] == "runtime_failure"
        ]
        candidate_infeasible = [
            value
            for value in evaluator.observations
            if value["status"] == "candidate_infeasible"
        ]
        if len(successes) < 2:
            raise RuntimeError("control did not preserve two successful seeds")
        seed_points = successes[:2]
        seed_hv = AffineHypervolumeSnapshot3D.create(
            spec=_utility_spec(), archive_points=tuple(seed_points)
        ).base_hypervolume
        final_hv = AffineHypervolumeSnapshot3D.create(
            spec=_utility_spec(), archive_points=tuple(successes)
        ).base_hypervolume
        front = _pareto_front(successes)
        reference_contains_all = all(
            all(value < 1.0 for value in _utility_spec().normalize(point))
            for point in successes
        )
        dominates_seed = any(
            _strictly_dominates(point, seed)
            for point in successes[2:]
            for seed in seed_points
        )
        health = {
            "reference_qualification_envelope_passed": (
                reference_qualification_passed is True
            ),
            "six_generations": (
                base_summary["generations_completed"] == GENERATION_COUNT
            ),
            "candidate_occurrences_exact": (
                base_summary["candidate_occurrences"] == PLANNED_CANDIDATE_OCCURRENCES
            ),
            "seven_scheduler_local_decisions": (
                base_summary["logical_agent_calls"] == PLANNED_LOGICAL_CALLS
            ),
            "six_direct_scale_matched_selections": (
                len(run.selector.results) == DIRECT_SELECTION_COUNT
                and all(
                    len(result.decision.members) == PORTFOLIO_WIDTH
                    for _, result in run.selector.results
                )
            ),
            "one_local_diagnostic_reflection": (
                base_summary["reflection_generations"] == [2]
            ),
            "zero_provider_and_credential_activity": True,
            "no_runtime_evaluator_failures": not runtime_failures,
            "all_candidate_outcomes_terminal_and_typed": (
                len(successes) + len(candidate_infeasible) == evaluator.calls
            ),
            "fixed_reference_contains_successes": reference_contains_all,
            "cleanup_released": run.execution.cleanup_receipt.released,
            "source_closure_unchanged": (
                _require_source_closure(source_sha256)["aggregate_sha256"]
                == source_sha256
            ),
        }
        status = "completed_healthy" if all(health.values()) else "completed_unhealthy"
        gate_a = _load_gate_a_baseline()
        elapsed_values = sorted(
            float(value["evaluator_elapsed_s"])
            for value in evaluator.observations
            if value["status"] == "passed"
        )
        return {
            "schema_version": 1,
            "status": status,
            "replicate_seed": replicate_seed,
            "health": health,
            "campaign": base_summary,
            "wall_s": (time.perf_counter_ns() - started_ns) / 1e9,
            "provider": {
                "credential_reads": 0,
                "logical_calls": 0,
                "physical_attempts": 0,
                "total_cost_usd": "0",
            },
            "evaluator": {
                "physical_calls": evaluator.calls,
                "successful": len(successes),
                "candidate_infeasible": len(candidate_infeasible),
                "runtime_failures": len(runtime_failures),
                "median_evaluator_elapsed_s": (
                    elapsed_values[len(elapsed_values) // 2] if elapsed_values else None
                ),
            },
            "candidate_quality": {
                "utility_definition_sha256": _utility_spec().definition_sha256,
                "seed_hypervolume_hex": seed_hv.hex(),
                "final_hypervolume_hex": final_hv.hex(),
                "absolute_gain_hex": (final_hv - seed_hv).hex(),
                "relative_gain": (
                    None if seed_hv == 0 else (final_hv - seed_hv) / seed_hv
                ),
                "nondominated_front_size": len(front),
                "nondominated_front": front,
                "dominates_seed": dominates_seed,
                "historical_gate_a_hypervolume_hex": (gate_a["affine_hypervolume_hex"]),
                "historical_gate_a_role": gate_a["role"],
            },
            "selection_decision_count": len(run.selector.results),
            "source_closure_sha256": source_sha256,
            "claim_scope": (
                "prospective_matched_uninformed_control_not_standalone_efficacy"
            ),
        }
    finally:
        for journal in raw_journals.values():
            journal.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--replicate-seed", required=True, type=int)
    parser.add_argument("--prereg")
    args = parser.parse_args()

    run_dir = (ARTIFACT_ROOT / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        paths = _source_paths()
        source = source_identity(paths, relative_to=WORKSPACE_ROOT)
        snapshot = _snapshot_sources(run_dir, paths)
        if (
            source["aggregate_sha256"] != snapshot["aggregate_sha256"]
            or source["file_count"] != snapshot["file_count"]
        ):
            raise RuntimeError("source changed while sealing control launch")
        write_json_atomic(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "mode": args.mode,
                "replicate_seed": args.replicate_seed,
                "created_at_utc": _utc_now(),
                "source_identity": source,
                "source_snapshot": snapshot,
                "utility_definition_sha256": _utility_spec().definition_sha256,
                "provider_calls": 0,
                "credential_reads": 0,
            },
        )
        probe = _construction_probe(args.replicate_seed)
        write_json_atomic(run_dir / "provider_free_construction_probe.json", probe)
        if not probe["all_gates_pass"]:
            raise RuntimeError("provider-free Timeloop control construction failed")
        baseline = _load_gate_a_baseline()
        write_json_atomic(run_dir / "historical_gate_a_baseline.json", baseline)
        qualification = _load_real_qualification_evidence()
        write_json_atomic(run_dir / "real_qualification_evidence.json", qualification)
        source_sha256 = str(source["aggregate_sha256"])
        _require_source_closure(source_sha256)

        if args.mode == "prepare":
            if args.prereg is not None:
                raise RuntimeError("prepare mode does not accept --prereg")
            prereg = _preregistration(
                replicate_seed=args.replicate_seed,
                source_sha256=source_sha256,
                probe=probe,
                baseline=baseline,
                qualification=qualification,
            )
            write_json_atomic(run_dir / "preregistration_template.json", prereg)
            summary = {
                "schema_version": 1,
                "status": (
                    "prepared_without_credential_provider_or_candidate_docker_run"
                ),
                "replicate_seed": args.replicate_seed,
                "credential_read": False,
                "provider_calls": 0,
                "candidate_docker_evaluations": 0,
                "source_aggregate_sha256": source_sha256,
                "provider_free_construction_probe_passed": True,
            }
            write_json_atomic(run_dir / "summary.json", summary)
            final = finalize_run_directory(run_dir, status=str(summary["status"]))
            print(json.dumps({**summary, "finalization": final}, sort_keys=True))
            return 0

        if args.prereg is None:
            raise RuntimeError("run mode requires --prereg")
        prereg_identity = _validate_preregistration(
            path=Path(args.prereg),
            replicate_seed=args.replicate_seed,
            source_sha256=source_sha256,
            probe=probe,
            baseline=baseline,
            qualification=qualification,
        )
        write_json_atomic(run_dir / "preregistration_identity.json", prereg_identity)
        settings = TimeloopV2Settings(
            output_root=run_dir / "evaluator_calls",
            cpu_set=CPU_SET,
            timeout_s=EVALUATOR_TIMEOUT_S,
        )
        docker_preflight = TimeloopV2DockerEvaluator(
            settings, frozen_network_panel("resnet50")
        ).preflight()
        write_json_atomic(run_dir / "docker_preflight.json", docker_preflight)
        _require_source_closure(source_sha256)
        summary = _execute_control(
            run_dir=run_dir,
            replicate_seed=args.replicate_seed,
            source_sha256=source_sha256,
            docker_preflight=docker_preflight,
            reference_qualification_passed=(
                qualification["reference_envelope_audit"]["strictly_contains_all"]
                is True
            ),
        )
        write_json_atomic(run_dir / "summary.json", summary)
        final = finalize_run_directory(run_dir, status=str(summary["status"]))
        print(json.dumps({**summary, "finalization": final}, sort_keys=True))
        return 0 if summary["status"] == "completed_healthy" else 2
    except BaseException as error:
        if not (run_dir / "summary.json").exists():
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "schema_version": 1,
                    "status": "failed_before_completion",
                    "failure_type": type(error).__qualname__,
                    "failure_sha256": hashlib.sha256(
                        f"{type(error).__qualname__}\x00{error}".encode()
                    ).hexdigest(),
                },
            )
        if not (run_dir / "finalized.json").exists():
            finalize_run_directory(run_dir, status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

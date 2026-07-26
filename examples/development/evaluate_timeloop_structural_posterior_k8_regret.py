#!/usr/bin/env python3
"""Evaluate local K8-to-K4 regret in the sealed Timeloop v10 campaign.

This diagnostic reconstructs every exact K8 proposal slate from a completed
campaign and evaluates the 24 candidates that the structural-posterior K4
allocator did not select.  It then compares the recorded structural K4 with
the model's direct top four, all 70 uniform K4 subsets, the oracle K4, and the
full K8 support.

``prepare`` is provider- and Timeloop-free and seals the complete evaluation
plan. ``execute`` accepts only that sealed plan.  The result is a post-hoc,
fixed-parent/fixed-slate allocator diagnostic, not a campaign counterfactual.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
from statistics import fmean
import sys
import time
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.policies.selection.calibrated_slate_codec import (  # noqa: E402
    decode_slate_allocation_request_record,
)
from agent_evolve.policies.selection.structural_posterior_slate import (  # noqa: E402
    StructuralPosteriorSlatePolicy,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (  # noqa: E402
    OBJECTIVE_NAMES,
    TimeloopV2DockerEvaluator,
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (  # noqa: E402
    frozen_network_panel,
)
from examples.development import evaluate_timeloop_frontier_probe_k8 as support  # noqa: E402
from examples.development.durable_run_artifacts import (  # noqa: E402
    DurableJsonlJournal,
    finalize_run_directory,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
)
DEFAULT_SOURCE_RUN = (
    ARTIFACT_ROOT
    / "benchmark_q1/timeloop_codesign/full_support_g6/"
    "grid_timeloop_v2_deepseek_s20260761_v10_calfrontier_r1_live"
)
EXPECTED_SOURCE_FINALIZATION = (
    "dd95dffc0a58e0ac6fe3cb5baff1363f320684a1604f1e687c689245425f8169"
)
EXPECTED_SOURCE_CONTENT = (
    "57a4bfc93b553c0b817cd78f144bc7258bb1b23cce97e97b80dd9ecc4304fa82"
)
CPU_SET = "8"
TIMEOUT_S = 180.0
PORTFOLIO_SIZE = 4
SLATE_SIZE = 8
WAVE_COUNT = 6
UNSELECTED_OCCURRENCES = 24


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _allocation_from_audit(
    audit: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Decode the v10 structural allocation embedded in the selector audit."""

    plaintext = support._object(audit.get("plaintext_audit"), name="plaintext audit")
    response_text = plaintext.get("response_text")
    if type(response_text) is not str:
        raise RuntimeError("selector audit omitted trusted response text")
    response = support._object(json.loads(response_text), name="selector response")
    supplemental = support._object(
        response.get("supplemental_selector_audit"), name="supplemental audit"
    )
    if supplemental.get("decision_sha256") != audit.get(
        "decision_sha256"
    ) or supplemental.get("request_sha256") != audit.get("request_sha256"):
        raise RuntimeError("supplemental audit does not join its selector audit")
    payload = support._object(supplemental.get("payload"), name="supplemental payload")
    allocation = support._object(payload.get("allocation"), name="allocation")
    request = decode_slate_allocation_request_record(allocation.get("request"))
    if allocation.get("request_sha256") != request.request_sha256:
        raise RuntimeError("structural allocation does not bind its decoded request")
    return allocation, request


def _build_precommit(
    *, source_run: Path, events: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Reuse the sealed K8 reconstruction while selecting the v10 policy codec."""

    original_policy = support.FrontierProbeSlatePolicy
    original_decoder = support._allocation_from_audit
    support.FrontierProbeSlatePolicy = StructuralPosteriorSlatePolicy
    support._allocation_from_audit = _allocation_from_audit
    try:
        plan, waves, accounting = support._build_precommit(
            source_run=source_run,
            events=events,
        )
    finally:
        support.FrontierProbeSlatePolicy = original_policy
        support._allocation_from_audit = original_decoder
    for wave in waves:
        wave["structural_posterior_decision"] = wave.pop(
            "frontier_probe_decision"
        )
    return plan, waves, accounting


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        Path(support.__file__).resolve(strict=True),
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/structural_posterior_slate.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/calibrated_slate_codec.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/reward/affine_hypervolume_3d.py",
        AGENT_EVOLVE_ROOT
        / "examples/benchmarks/timeloop_codesign/v2/finite_variation_catalog.py",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/timeloop_codesign/v2/evaluator.py",
        AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
    )


def _validated_source_finalization(
    *,
    source_run: Path,
    expected_finalization_sha256: str | None,
    expected_content_sha256: str | None,
) -> dict[str, object]:
    source_seal = verify_finalized_run_directory(source_run)
    if source_seal.get("status") != "completed_healthy":
        raise RuntimeError("Timeloop K8 completion requires a healthy sealed source")
    if (
        expected_finalization_sha256 is not None
        and source_seal.get("finalization_sha256")
        != expected_finalization_sha256
    ):
        raise RuntimeError("Timeloop source finalization differs from expectation")
    if (
        expected_content_sha256 is not None
        and source_seal.get("recursive_content_sha256")
        != expected_content_sha256
    ):
        raise RuntimeError("Timeloop source content differs from expectation")
    return source_seal


def _prepare(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_seal = _validated_source_finalization(
        source_run=source_run,
        expected_finalization_sha256=args.expected_source_finalization_sha256,
        expected_content_sha256=args.expected_source_content_sha256,
    )
    plan, waves, accounting = _build_precommit(
        source_run=source_run,
        events=support._campaign_events(source_run),
    )
    policy = StructuralPosteriorSlatePolicy()
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "preparing_provider_and_timeloop_free",
        "diagnostic": "timeloop_generic_campaign_k8_support_completion",
        "source_run": {
            "path": support._relative(source_run),
            "finalization_sha256": source_seal["finalization_sha256"],
            "recursive_content_sha256": source_seal["recursive_content_sha256"],
            "mutated": False,
        },
        "policy": policy.to_record(),
        "accounting": accounting,
        "comparators": [
            "recorded_structural_posterior_k4",
            "direct_model_top4",
            "uniform_k4_exact_expectation_over_70_subsets",
            "oracle_k4",
            "full_k8_support",
        ],
        "claim_boundary": {
            "posthoc_diagnostic": True,
            "complete_evaluation_plan_sealed_before_new_outcomes": True,
            "fixed_parent_fixed_k8_local_allocator_diagnostic": True,
            "campaign_counterfactual": False,
            "paper_ready_efficacy": False,
            "provider_calls": 0,
            "api_key_reads": 0,
            "timeloop_evaluations_during_preparation": 0,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "source_identity": source_identity(
            _source_paths(), relative_to=WORKSPACE_ROOT
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    write_json_atomic(
        output_dir / "evaluation_plan.json",
        {"schema_version": 1, "accounting": accounting, "rows": plan},
    )
    write_json_atomic(
        output_dir / "waves.json",
        {"schema_version": 1, "wave_count": len(waves), "waves": waves},
    )
    precommit = {
        "schema_version": 1,
        "status": "prepared_provider_and_timeloop_free",
        "prepared_at_utc": _utc_now(),
        "source_run_mutated": False,
        "provider_calls": 0,
        "api_key_reads": 0,
        "timeloop_evaluations": 0,
        "accounting": accounting,
    }
    write_json_atomic(output_dir / "precommit.json", precommit)
    seal = finalize_run_directory(
        output_dir, status="prepared_provider_and_timeloop_free"
    )
    print(
        json.dumps(
            {
                **precommit,
                "output_dir": str(output_dir),
                "finalization_sha256": seal["finalization_sha256"],
                "recursive_content_sha256": seal["recursive_content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _member_outcomes(
    *,
    wave: dict[str, Any],
    evaluated: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, float] | None], dict[str, str]]:
    outcomes: dict[str, dict[str, float] | None] = {}
    statuses: dict[str, str] = {}
    for member_value in wave["members"]:
        member = support._object(member_value, name="K8 member")
        source = member.get("source_objectives")
        if source is not None:
            outcome = {
                key: float(value)
                for key, value in support._object(
                    source, name="source objectives"
                ).items()
            }
            status = "passed_source"
        else:
            record = evaluated.get(
                (wave["outer_request_sha256"], member["option_id"])
            )
            if record is None:
                raise RuntimeError("completed diagnostic omitted an outcome")
            status = str(record["status"])
            raw = record.get("objectives")
            outcome = (
                None
                if raw is None
                else {
                    key: float(value)
                    for key, value in support._object(
                        raw, name="evaluation objectives"
                    ).items()
                }
            )
        if outcome is not None and set(outcome) != set(OBJECTIVE_NAMES):
            raise RuntimeError("completed outcome differs from Timeloop metrics")
        outcomes[member["option_id"]] = outcome
        statuses[member["option_id"]] = status
    if len(outcomes) != SLATE_SIZE:
        raise RuntimeError("completed Timeloop wave differs from exact K8")
    return outcomes, statuses


def _full_support(
    *, option_ids: tuple[str, ...], outcomes: dict[str, dict[str, float] | None], snapshot: Any
) -> dict[str, Any]:
    points = tuple(
        value for option_id in option_ids if (value := outcomes[option_id]) is not None
    )
    gain = snapshot.joint_gain(points)
    return {
        "option_ids": list(option_ids),
        "successful_point_count": len(points),
        "gain": gain,
        "raw_oriented_gain": gain * snapshot.spec.raw_volume_scale,
        "augmented_hypervolume": snapshot.base_hypervolume + gain,
    }


def _summarize_members(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    grouped: dict[object, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result: list[dict[str, Any]] = []
    for value, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        known = sum(int(member["forecast_known"]) for member in members)
        correct = sum(int(member["forecast_correct"]) for member in members)
        result.append(
            {
                key: value,
                "occurrences": len(members),
                "selected_occurrences": sum(member["primary_selected"] for member in members),
                "passed": sum(member["outcomes"] is not None for member in members),
                "candidate_infeasible": sum(
                    member["outcomes"] is None for member in members
                ),
                "marginal_gain_sum": sum(member["marginal_gain"] for member in members),
                "mean_marginal_gain": fmean(
                    float(member["marginal_gain"]) for member in members
                ),
                "positive_marginal_gain_count": sum(
                    member["positive_marginal_gain"] for member in members
                ),
                "strict_parent_dominator_count": sum(
                    member["parent_relation"] == "strictly_dominates_parent"
                    for member in members
                ),
                "canonical_oracle_selection_count": sum(
                    member["canonical_oracle_selected"] for member in members
                ),
                "forecast_known": known,
                "forecast_correct": correct,
                "forecast_direction_accuracy": (
                    None if known == 0 else correct / known
                ),
            }
        )
    return result


def _analyze(
    *, waves: list[dict[str, Any]], evaluations: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluated = {
        (value["outer_request_sha256"], value["option_id"]): value
        for value in evaluations
    }
    if len(evaluated) != len(evaluations):
        raise RuntimeError("evaluation occurrence identities are not unique")

    wave_results: list[dict[str, Any]] = []
    all_members: list[dict[str, Any]] = []
    overall_total = 0
    overall_known = 0
    overall_correct = 0
    for wave in waves:
        snapshot = support._affine_snapshot(wave["archive_snapshot"])
        members = [support._object(value, name="K8 member") for value in wave["members"]]
        outcomes, statuses = _member_outcomes(wave=wave, evaluated=evaluated)
        option_order = tuple(member["option_id"] for member in members)
        subset_rows: list[dict[str, Any]] = []
        for subset in combinations(option_order, PORTFOLIO_SIZE):
            points = tuple(
                value
                for option_id in subset
                if (value := outcomes[option_id]) is not None
            )
            subset_rows.append(
                {
                    "option_ids": list(subset),
                    "successful_point_count": len(points),
                    "gain": snapshot.joint_gain(points),
                }
            )
        subset_rows.sort(key=lambda value: (-value["gain"], value["option_ids"]))
        gains = [float(value["gain"]) for value in subset_rows]
        primary = tuple(wave["primary_selected_option_ids"])
        direct = tuple(member["option_id"] for member in members[:PORTFOLIO_SIZE])
        if len(primary) != PORTFOLIO_SIZE or len(set(primary)) != PORTFOLIO_SIZE:
            raise RuntimeError("recorded structural selection differs from exact K4")

        decision = support._object(
            wave["structural_posterior_decision"], name="structural decision"
        )
        selected_roles = {
            selected["option_id"]: selected["role"]
            for selected in (
                support._object(value, name="allocated member")
                for value in support._array(decision.get("selected"), name="selected")
            )
        }
        if set(selected_roles) != set(primary):
            raise RuntimeError("role receipt differs from recorded structural K4")

        canonical_oracle = set(subset_rows[0]["option_ids"])
        member_results: list[dict[str, Any]] = []
        for member in members:
            option_id = member["option_id"]
            outcome = outcomes[option_id]
            parent = {
                key: float(value)
                for key, value in support._object(
                    member["parent_objectives"], name="parent objectives"
                ).items()
            }
            forecast_total = len(member["predictions"])
            forecast_known = 0
            forecast_correct = 0
            prediction_results: list[dict[str, Any]] = []
            for prediction_value in member["predictions"]:
                prediction = support._object(prediction_value, name="prediction")
                asserted = str(prediction["asserted_direction"])
                actual = (
                    None
                    if outcome is None
                    else support._actual_direction(
                        outcome[prediction["metric_id"]],
                        parent[prediction["metric_id"]],
                    )
                )
                known = asserted != "unknown" and actual is not None
                correct = known and asserted == actual
                forecast_known += int(known)
                forecast_correct += int(correct)
                prediction_results.append(
                    {
                        **prediction,
                        "actual_direction": actual,
                        "known_for_accuracy": known,
                        "correct": correct,
                    }
                )
            marginal_gain = 0.0 if outcome is None else snapshot.marginal_gain(outcome)
            row = {
                "model_rank": member["model_rank"],
                "option_id": option_id,
                "family": member["family"],
                "locus_key": member["locus_key"],
                "primary_selected": option_id in set(primary),
                "selected_role": selected_roles.get(option_id, "not_selected"),
                "direct_model_top4_selected": option_id in set(direct),
                "status": statuses[option_id],
                "outcomes": outcome,
                "marginal_gain": marginal_gain,
                "positive_marginal_gain": marginal_gain > 0.0,
                "parent_relation": support._parent_relation(outcome, parent),
                "canonical_oracle_selected": option_id in canonical_oracle,
                "forecast_total": forecast_total,
                "forecast_known": forecast_known,
                "forecast_correct": forecast_correct,
                "forecast_accuracy": (
                    None if forecast_known == 0 else forecast_correct / forecast_known
                ),
                "predictions": prediction_results,
            }
            member_results.append(row)
            all_members.append(row)
            overall_total += forecast_total
            overall_known += forecast_known
            overall_correct += forecast_correct

        structural = support._selection(
            option_ids=primary,
            outcomes=outcomes,
            snapshot=snapshot,
            subset_gains=gains,
        )
        model_top4 = support._selection(
            option_ids=direct,
            outcomes=outcomes,
            snapshot=snapshot,
            subset_gains=gains,
        )
        wave_results.append(
            {
                "wave_ordinal": wave["wave_ordinal"],
                "generation": wave["generation"],
                "parent_slot": wave["parent_slot"],
                "outer_request_sha256": wave["outer_request_sha256"],
                "base_hypervolume": snapshot.base_hypervolume,
                "structural_posterior_k4": structural,
                "direct_model_top4": model_top4,
                "uniform_k4": {
                    "support_size": len(gains),
                    "expected_gain": fmean(gains),
                    "minimum_gain": min(gains),
                    "maximum_gain": max(gains),
                    "zero_gain_fraction": sum(value == 0.0 for value in gains)
                    / len(gains),
                },
                "oracle_k4": subset_rows[0],
                "oracle_tie_count": sum(value == gains[0] for value in gains),
                "full_k8_support": _full_support(
                    option_ids=option_order,
                    outcomes=outcomes,
                    snapshot=snapshot,
                ),
                "structural_minus_model_top4_gain": (
                    structural["gain"] - model_top4["gain"]
                ),
                "members": member_results,
            }
        )

    structural = [value["structural_posterior_k4"]["gain"] for value in wave_results]
    direct = [value["direct_model_top4"]["gain"] for value in wave_results]
    uniform = [value["uniform_k4"]["expected_gain"] for value in wave_results]
    oracle = [value["oracle_k4"]["gain"] for value in wave_results]
    full = [value["full_k8_support"]["gain"] for value in wave_results]
    structural_sum = sum(structural)
    direct_sum = sum(direct)
    uniform_sum = sum(uniform)
    oracle_sum = sum(oracle)
    full_sum = sum(full)
    terminal = all(
        value["status"] in {"passed", "candidate_infeasible"}
        for value in evaluations
    )
    complete = (
        len(evaluations)
        + sum(
            member["source_objectives"] is not None
            for wave in waves
            for member in wave["members"]
        )
        == WAVE_COUNT * SLATE_SIZE
        and terminal
    )
    selected_members = [value for value in all_members if value["primary_selected"]]
    discarded_members = [value for value in all_members if not value["primary_selected"]]

    if structural_sum < uniform_sum:
        diagnosis = "structural_k4_underperforms_uniform_k4_on_fixed_slates"
    elif structural_sum < direct_sum:
        diagnosis = "structural_k4_underperforms_direct_model_top4"
    elif structural_sum < oracle_sum:
        diagnosis = "structural_k4_beats_controls_but_retains_selection_regret"
    else:
        diagnosis = "structural_k4_matches_oracle_k4"

    return {
        "schema_version": 1,
        "claim_scope": (
            "posthoc_precommitted_fixed_parent_fixed_k8_allocator_diagnostic_"
            "not_campaign_counterfactual_or_efficacy"
        ),
        "evidence_complete": complete,
        "diagnosis": diagnosis,
        "waves": wave_results,
        "rank_analysis": _summarize_members(all_members, key="model_rank"),
        "role_analysis": _summarize_members(all_members, key="selected_role"),
        "forecast_overall": {
            "total": overall_total,
            "known": overall_known,
            "correct": overall_correct,
            "exact_direction_accuracy": (
                None if overall_known == 0 else overall_correct / overall_known
            ),
        },
        "aggregate": {
            "wave_count": len(wave_results),
            "structural_posterior_k4_gain_sum": structural_sum,
            "direct_model_top4_gain_sum": direct_sum,
            "uniform_expected_k4_gain_sum": uniform_sum,
            "oracle_k4_gain_sum": oracle_sum,
            "full_k8_gain_sum": full_sum,
            "structural_minus_direct_model_top4_gain_sum": structural_sum - direct_sum,
            "structural_minus_uniform_expected_gain_sum": structural_sum - uniform_sum,
            "structural_regret_to_oracle_k4_gain_sum": oracle_sum - structural_sum,
            "model_top4_regret_to_oracle_k4_gain_sum": oracle_sum - direct_sum,
            "k4_structural_regret_to_full_k8_gain_sum": full_sum - structural_sum,
            "structural_multiple_of_uniform": (
                None if uniform_sum == 0.0 else structural_sum / uniform_sum
            ),
            "structural_fraction_of_oracle_k4": (
                None if oracle_sum == 0.0 else structural_sum / oracle_sum
            ),
            "structural_fraction_of_full_k8": (
                None if full_sum == 0.0 else structural_sum / full_sum
            ),
            "structural_wave_wins_vs_model_top4": sum(
                left > right for left, right in zip(structural, direct, strict=True)
            ),
            "structural_wave_wins_vs_uniform_expectation": sum(
                left > right for left, right in zip(structural, uniform, strict=True)
            ),
            "structural_oracle_ties": sum(
                left == right for left, right in zip(structural, oracle, strict=True)
            ),
            "selected_individual_marginal_gain_sum": sum(
                value["marginal_gain"] for value in selected_members
            ),
            "discarded_individual_marginal_gain_sum": sum(
                value["marginal_gain"] for value in discarded_members
            ),
            "selected_positive_marginal_count": sum(
                value["positive_marginal_gain"] for value in selected_members
            ),
            "discarded_positive_marginal_count": sum(
                value["positive_marginal_gain"] for value in discarded_members
            ),
        },
    }


def _execute(args: argparse.Namespace) -> int:
    preparation = args.preparation.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    preparation_seal = verify_finalized_run_directory(preparation)
    if preparation_seal.get("status") != "prepared_provider_and_timeloop_free":
        raise RuntimeError("execution requires the sealed provider/Timeloop-free plan")
    prepared_manifest = support._read_json(preparation / "manifest.json")
    policy = StructuralPosteriorSlatePolicy()
    if prepared_manifest.get("policy") != policy.to_record():
        raise RuntimeError("current structural-posterior policy differs from precommit")
    source_run = (
        WORKSPACE_ROOT
        / support._object(prepared_manifest.get("source_run"), name="source")["path"]
    ).resolve(strict=True)
    source_seal_before = verify_finalized_run_directory(source_run)
    if (
        source_seal_before.get("finalization_sha256")
        != prepared_manifest["source_run"]["finalization_sha256"]
        or source_seal_before.get("recursive_content_sha256")
        != prepared_manifest["source_run"]["recursive_content_sha256"]
    ):
        raise RuntimeError("sealed Timeloop source differs from preparation")
    plan_record = support._read_json(preparation / "evaluation_plan.json")
    waves_record = support._read_json(preparation / "waves.json")
    plan = [
        support._object(value, name="prepared evaluation")
        for value in support._array(plan_record.get("rows"), name="plan")
    ]
    waves = [
        support._object(value, name="prepared wave")
        for value in support._array(waves_record.get("waves"), name="waves")
    ]
    if len(plan) > UNSELECTED_OCCURRENCES or len(waves) != WAVE_COUNT:
        raise RuntimeError("sealed preparation differs from the exact K8 scale")

    output_dir.mkdir(parents=True, exist_ok=False)
    settings = TimeloopV2Settings(
        output_root=output_dir / "evaluator_calls",
        cpu_set=CPU_SET,
        timeout_s=TIMEOUT_S,
        external_concurrency=1,
    )
    evaluator = TimeloopV2DockerEvaluator(settings, frozen_network_panel("resnet50"))
    docker_preflight = evaluator.preflight()
    write_json_atomic(output_dir / "docker_preflight.json", docker_preflight)
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "running",
        "diagnostic": "timeloop_generic_campaign_k8_support_completion",
        "preparation": {
            "path": support._relative(preparation),
            "finalization_sha256": preparation_seal["finalization_sha256"],
            "recursive_content_sha256": preparation_seal["recursive_content_sha256"],
        },
        "source_run": prepared_manifest["source_run"],
        "policy": policy.to_record(),
        "workload": {
            "id": "timeloop-v2-resnet50-three-medoid-codesign",
            "cpu_set": CPU_SET,
            "external_concurrency": 1,
            "timeout_s": TIMEOUT_S,
            "planned_evaluation_occurrences": len(plan),
            "planned_unique_physical_evaluations": len(
                {value["configuration_sha256"] for value in plan}
            ),
        },
        "docker_preflight": docker_preflight,
        "claim_boundary": prepared_manifest["claim_boundary"],
    }
    write_json_atomic(output_dir / "manifest.json", manifest)

    started = time.perf_counter()
    evaluations: list[dict[str, Any]] = []
    by_configuration: dict[str, tuple[int, dict[str, Any]]] = {}
    try:
        with DurableJsonlJournal(output_dir / "evaluations.jsonl") as journal:
            for ordinal, row in enumerate(plan, start=1):
                configuration_sha256 = row["configuration_sha256"]
                prior = by_configuration.get(configuration_sha256)
                if prior is None:
                    record = support._evaluate(
                        row=row,
                        evaluator=evaluator,
                        output_dir=output_dir,
                    )
                    by_configuration[configuration_sha256] = (ordinal, record)
                else:
                    source_ordinal, source = prior
                    record = support._reuse_evaluation(
                        row=row,
                        source=source,
                        source_ordinal=source_ordinal,
                    )
                evaluations.append(record)
                journal.append(record)
                print(
                    json.dumps(
                        {
                            "progress": f"{ordinal}/{len(plan)}",
                            "wave_ordinal": row["wave_ordinal"],
                            "model_rank": row["model_rank"],
                            "option_id": row["option_id"],
                            "status": record["status"],
                            "physical": record["physical_timeloop_evaluation"],
                            "harness_elapsed_s": record["harness_elapsed_s"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except BaseException as error:
        write_json_atomic(
            output_dir / "failure.json",
            {
                "schema_version": 1,
                "failed_at_utc": _utc_now(),
                "failure_type": type(error).__qualname__,
                "failure_sha256": hashlib.sha256(
                    f"{type(error).__qualname__}\x00{error}".encode("utf-8")
                ).hexdigest(),
                "completed_evaluation_occurrences": len(evaluations),
                "provider_calls": 0,
                "api_key_reads": 0,
            },
        )
        manifest["status"] = "failed"
        write_json_atomic(output_dir / "manifest.json", manifest)
        finalize_run_directory(output_dir, status="failed")
        raise

    analysis = _analyze(waves=waves, evaluations=evaluations)
    wall_s = time.perf_counter() - started
    write_json_atomic(output_dir / "allocation_analysis.json", analysis)
    physical = [value for value in evaluations if value["physical_timeloop_evaluation"]]
    source_seal_after = verify_finalized_run_directory(source_run)
    if source_seal_after != source_seal_before:
        raise RuntimeError("source run changed during held-out execution")
    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "source_run_mutated": False,
        "accounting": {
            **prepared_manifest["accounting"],
            "completed_evaluation_occurrences": len(evaluations),
            "physical_timeloop_evaluations": len(physical),
            "identity_reuses": len(evaluations) - len(physical),
            "passed_unselected_occurrences": sum(
                value["status"] == "passed" for value in evaluations
            ),
            "candidate_infeasible_unselected_occurrences": sum(
                value["status"] == "candidate_infeasible" for value in evaluations
            ),
            "runtime_system_failures": 0,
        },
        "wall_s": wall_s,
        "mean_physical_evaluation_wall_s": (
            None
            if not physical
            else fmean(float(value["harness_elapsed_s"]) for value in physical)
        ),
        "diagnosis": analysis["diagnosis"],
        "aggregate": analysis["aggregate"],
        "forecast_overall": analysis["forecast_overall"],
        "provider_calls": 0,
        "api_key_reads": 0,
        "claim_scope": analysis["claim_scope"],
    }
    write_json_atomic(output_dir / "result.json", result)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = result["completed_at_utc"]
    manifest["result_sha256"] = support._sha256_file(output_dir / "result.json")
    write_json_atomic(output_dir / "manifest.json", manifest)
    seal = finalize_run_directory(output_dir, status="completed")
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(output_dir),
                "finalization_sha256": seal["finalization_sha256"],
                "recursive_content_sha256": seal["recursive_content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    prepare.add_argument("--expected-source-finalization-sha256")
    prepare.add_argument("--expected-source-content-sha256")
    prepare.add_argument("--output-dir", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--preparation", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "execute":
        return _execute(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())

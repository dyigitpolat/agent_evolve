#!/usr/bin/env python3
"""Replay the frontier-probe allocator on three completed development panels.

The three panels are BOiLS R8, the older BOiLS-Q campaign, and Heat2D.  Every
candidate outcome existed before this script was written.  This is therefore
mechanism-development evidence only: it authenticates inputs, performs zero
provider or evaluator calls, and emits a replayable result matrix before the
policy is exposed to the genuinely held-out Timeloop panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Callable


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.policies.selection.calibrated_slate_codec import (  # noqa: E402
    decode_slate_allocation_request_record,
)
from agent_evolve.policies.selection.frontier_probe_slate import (  # noqa: E402
    FrontierProbeSlateDecision,
    FrontierProbeSlatePolicy,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)
from examples.development.evaluate_boils_calibrated_slate_counterfactual import (  # noqa: E402
    _build_plan as build_boilsq_plan,
    _snapshot as boilsq_snapshot,
)
from examples.development.evaluate_boils_generic_campaign_k8 import (  # noqa: E402
    RAW_HYPERVOLUME_SCALE,
    _build_plan as build_boils_r8_plan,
    _campaign_events,
    _snapshot as boils_r8_snapshot,
)
from examples.development.evaluate_heat2d_structural_posterior_k8 import (  # noqa: E402
    _affine_snapshot,
    _describe_selection as describe_heat_selection,
)
from examples.development.replay_boils_structural_posterior_allocator import (  # noqa: E402
    DEFAULT_K8_DIAGNOSTIC as BOILS_R8_DIAGNOSTIC,
    DEFAULT_SOURCE_RUN as BOILS_R8_SOURCE,
    _allocation_requests_by_outer_request,
    _completed_outcome_waves,
    _describe_selection as describe_boils_selection,
)
from examples.development.replay_boilsq_structural_posterior_allocator import (  # noqa: E402
    DEFAULT_K8_DIAGNOSTIC as BOILSQ_DIAGNOSTIC,
    DEFAULT_SOURCE_RUN as BOILSQ_SOURCE,
)


HEAT_PRECOMMIT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/engibench_heat2d/generic_campaign"
    / "heat_structural_posterior_k8_precommit_v1_20260719"
)
HEAT_DIAGNOSTIC = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/engibench_heat2d/generic_campaign"
    / "heat_structural_posterior_k8_kill_test_v1_20260719"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "allocator_v2"
    / "frontier_probe_three_panel_development_replay_v1_20260719"
)


def _object(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeError(f"{name} must be an exact JSON object")
    return value


def _array(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeError(f"{name} must be an exact JSON array")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return _object(json.load(stream), name=path.name)


def _relative(path: Path) -> str:
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def _source_record(path: Path) -> dict[str, object]:
    finalization = verify_finalized_run_directory(path)
    return {
        "path": _relative(path),
        "finalization_sha256": finalization["finalization_sha256"],
        "content_sha256": finalization.get(
            "content_sha256",
            finalization.get("recursive_content_sha256"),
        ),
        "mutated": False,
    }


def _outcomes(
    wave: dict[str, Any],
) -> dict[str, dict[str, float]]:
    result = {
        row["option_id"]: _object(row.get("outcomes"), name="K8 outcome")
        for row in (
            _object(value, name="K8 outcome row")
            for value in _array(wave.get("all_k8_outcomes"), name="K8 outcomes")
        )
    }
    if len(result) != 8:
        raise RuntimeError("completed panel must contain eight distinct outcomes")
    return result


def _decision_summary(
    decision: FrontierProbeSlateDecision,
) -> dict[str, object]:
    decision.revalidate()
    return {
        "decision_sha256": decision.decision_sha256,
        "source_request_sha256": decision.source_request.request_sha256,
        "projected_request_sha256": decision.request.request_sha256,
        "projection": decision.projection.to_record(),
        "target_option_ids": list(decision.target_option_ids),
        "selected_option_ids": [value.option_id for value in decision.selected],
        "selected": [value.to_record() for value in decision.selected],
        "available_full_abstention_option_ids": list(
            decision.available_full_abstention_option_ids
        ),
        "selected_probe_option_id": decision.selected_probe_option_id,
        "ideal_target_feasible": decision.ideal_target_feasible,
        "feasible_subset_count": decision.feasible_subset_count,
        "distinct_family_count": decision.distinct_family_count,
        "distinct_locus_count": decision.distinct_locus_count,
        "distinct_phenotype_count": decision.distinct_phenotype_count,
        "administered_card_keys": list(decision.administered_card_keys),
        "memory_dose_assessment": (
            None
            if decision.memory_dose_assessment is None
            else decision.memory_dose_assessment.to_record()
        ),
        "prior_only": decision.prior_only,
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    historical_label: str,
    normalized_scale: float | None,
) -> dict[str, object]:
    policy = [float(row["frontier_probe_k4"]["gain"]) for row in rows]
    historical = [float(row[historical_label]["gain"]) for row in rows]
    direct = [float(row["direct_model_top4"]["gain"]) for row in rows]
    uniform = [float(row["uniform_k4"]["expected_gain"]) for row in rows]
    oracle = [float(row["oracle_k4"]["gain"]) for row in rows]
    policy_sum = sum(policy)
    historical_sum = sum(historical)
    direct_sum = sum(direct)
    uniform_sum = sum(uniform)
    oracle_sum = sum(oracle)
    return {
        "wave_count": len(rows),
        "frontier_probe_gain_sum": policy_sum,
        **(
            {}
            if normalized_scale is None
            else {
                "frontier_probe_normalized_gain_sum": (
                    policy_sum / normalized_scale
                )
            }
        ),
        "historical_comparator": historical_label,
        "historical_gain_sum": historical_sum,
        "direct_model_top4_gain_sum": direct_sum,
        "uniform_expected_gain_sum": uniform_sum,
        "oracle_gain_sum": oracle_sum,
        "frontier_probe_minus_historical_gain_sum": (
            policy_sum - historical_sum
        ),
        "frontier_probe_minus_uniform_expected_gain_sum": (
            policy_sum - uniform_sum
        ),
        "frontier_probe_multiple_of_historical": (
            None if historical_sum == 0.0 else policy_sum / historical_sum
        ),
        "frontier_probe_multiple_of_uniform": (
            None if uniform_sum == 0.0 else policy_sum / uniform_sum
        ),
        "frontier_probe_fraction_of_oracle": (
            None if oracle_sum == 0.0 else policy_sum / oracle_sum
        ),
        "frontier_probe_equals_direct_model_top4": policy_sum == direct_sum,
        "wins_vs_uniform_expectation": sum(
            left > right for left, right in zip(policy, uniform, strict=True)
        ),
        "oracle_ties": sum(
            left == right for left, right in zip(policy, oracle, strict=True)
        ),
        "full_abstention_probe_waves": sum(
            row["decision"]["selected_probe_option_id"] is not None
            for row in rows
        ),
        "mating_constraint_projection_waves": sum(
            row["decision"]["projection"]["removed_pairwise_constraint"]
            or row["decision"]["projection"][
                "removed_min_distinct_families"
            ]
            is not None
            for row in rows
        ),
        "ideal_target_feasible_waves": sum(
            row["decision"]["ideal_target_feasible"] for row in rows
        ),
        "mean_rank_min_among_70_subsets": fmean(
            float(row["frontier_probe_k4"]["rank_min"]) for row in rows
        ),
        "mean_rank_max_among_70_subsets": fmean(
            float(row["frontier_probe_k4"]["rank_max"]) for row in rows
        ),
    }


def _boils_r8_panel(policy: FrontierProbeSlatePolicy) -> dict[str, object]:
    events = _campaign_events(BOILS_R8_SOURCE)
    _, source_waves, _ = build_boils_r8_plan(events=events)
    requests = _allocation_requests_by_outer_request(events)
    completed = _completed_outcome_waves(BOILS_R8_DIAGNOSTIC)
    rows: list[dict[str, Any]] = []
    for source_wave in source_waves:
        key = source_wave["request_sha256"]
        request = requests.get(key)
        outcome_wave = completed.get(key)
        if request is None or outcome_wave is None:
            raise RuntimeError("BOiLS R8 source/request/outcome join failed")
        decision = policy.select(request)
        outcome_map = _outcomes(outcome_wave)
        if set(outcome_map) != {
            member.option_id for member in request.slate.members
        }:
            raise RuntimeError("BOiLS R8 outcome support differs from its K8")
        selection = describe_boils_selection(
            option_ids=tuple(value.option_id for value in decision.selected),
            outcomes=outcome_map,
            snapshot=boils_r8_snapshot(
                source_wave["archive_reward_snapshot"]
            ),
        )
        rows.append(
            {
                "wave_ordinal": source_wave["wave_ordinal"],
                "generation": source_wave["generation"],
                "parent_slot": (source_wave["wave_ordinal"] - 1) % 2,
                "outer_request_sha256": key,
                "decision": _decision_summary(decision),
                "frontier_probe_k4": selection,
                "historical_model_anchored_k4": outcome_wave["calibrated_k4"],
                "direct_model_top4": outcome_wave["direct_model_top4"],
                "uniform_k4": outcome_wave["uniform_k4"],
                "oracle_k4": outcome_wave["oracle_k4"],
            }
        )
    return {
        "panel_id": "boils_r8",
        "workload_id": "boils_abc_logic_synthesis",
        "metric_space": "raw_frozen_wave_hypervolume_gain",
        "sources": [
            _source_record(BOILS_R8_SOURCE),
            _source_record(BOILS_R8_DIAGNOSTIC),
        ],
        "waves": rows,
        "aggregate": _aggregate(
            rows,
            historical_label="historical_model_anchored_k4",
            normalized_scale=RAW_HYPERVOLUME_SCALE,
        ),
    }


def _boilsq_requests(
    summary: dict[str, Any],
) -> dict[str, Any]:
    requests: dict[str, Any] = {}
    for audit_value in _array(
        summary.get("portfolio_selection_audits"),
        name="BOiLS-Q selector audits",
    ):
        audit = _object(audit_value, name="BOiLS-Q selector audit")
        decision = _object(audit.get("decision"), name="BOiLS-Q decision")
        supplemental = _object(
            decision.get("supplemental_selector_audit"),
            name="BOiLS-Q supplemental audit",
        )
        payload = _object(
            supplemental.get("payload"),
            name="BOiLS-Q supplemental payload",
        )
        allocation = _object(
            payload.get("allocation"),
            name="BOiLS-Q allocation",
        )
        key = audit.get("wave_request_sha256")
        if type(key) is not str or key in requests:
            raise RuntimeError("BOiLS-Q request identity is invalid or duplicated")
        requests[key] = decode_slate_allocation_request_record(
            allocation.get("request")
        )
    if len(requests) != 6:
        raise RuntimeError("expected six authenticated BOiLS-Q requests")
    return requests


def _boilsq_panel(policy: FrontierProbeSlatePolicy) -> dict[str, object]:
    summary = _read_json(BOILSQ_SOURCE / "summary.json")
    _, source_waves = build_boilsq_plan(
        source_run=BOILSQ_SOURCE,
        summary=summary,
        wave_rows=read_jsonl(BOILSQ_SOURCE / "wave_requests.jsonl"),
    )
    requests = _boilsq_requests(summary)
    analysis = _read_json(BOILSQ_DIAGNOSTIC / "analysis.json")
    completed = {
        row["request_sha256"]: row
        for row in (
            _object(value, name="BOiLS-Q completed wave")
            for value in _array(
                analysis.get("waves"),
                name="BOiLS-Q completed waves",
            )
        )
    }
    if len(completed) != 6:
        raise RuntimeError("expected six completed BOiLS-Q waves")
    rows: list[dict[str, Any]] = []
    for source_wave in source_waves:
        key = source_wave["request_sha256"]
        request = requests.get(key)
        outcome_wave = completed.get(key)
        if request is None or outcome_wave is None:
            raise RuntimeError("BOiLS-Q source/request/outcome join failed")
        decision = policy.select(request)
        outcome_map = _outcomes(outcome_wave)
        selection = describe_boils_selection(
            option_ids=tuple(value.option_id for value in decision.selected),
            outcomes=outcome_map,
            snapshot=boilsq_snapshot(source_wave["archive_reward_snapshot"]),
        )
        rows.append(
            {
                "wave_ordinal": source_wave["wave_ordinal"],
                "generation": (
                    2 * ((source_wave["wave_ordinal"] - 1) // 2)
                )
                + 1,
                "parent_slot": (source_wave["wave_ordinal"] - 1) % 2,
                "outer_request_sha256": key,
                "decision": _decision_summary(decision),
                "frontier_probe_k4": selection,
                "historical_four_role_k4": outcome_wave["calibrated_k4"],
                "direct_model_top4": outcome_wave["direct_model_top4"],
                "uniform_k4": outcome_wave["uniform_k4"],
                "oracle_k4": outcome_wave["oracle_k4"],
            }
        )
    return {
        "panel_id": "boils_q",
        "workload_id": "boils_abc_logic_synthesis_older_portfolio_q",
        "metric_space": "raw_frozen_wave_hypervolume_gain",
        "sources": [
            _source_record(BOILSQ_SOURCE),
            _source_record(BOILSQ_DIAGNOSTIC),
        ],
        "waves": rows,
        "aggregate": _aggregate(
            rows,
            historical_label="historical_four_role_k4",
            normalized_scale=RAW_HYPERVOLUME_SCALE,
        ),
    }


def _heat_panel(policy: FrontierProbeSlatePolicy) -> dict[str, object]:
    source = _read_json(HEAT_PRECOMMIT / "waves.json")
    source_waves = [
        _object(value, name="Heat source wave")
        for value in _array(source.get("waves"), name="Heat source waves")
    ]
    analysis = _read_json(HEAT_DIAGNOSTIC / "allocation_analysis.json")
    completed = {
        row["outer_request_sha256"]: row
        for row in (
            _object(value, name="Heat completed wave")
            for value in _array(
                analysis.get("waves"),
                name="Heat completed waves",
            )
        )
    }
    if len(source_waves) != 6 or len(completed) != 6:
        raise RuntimeError("Heat panel must contain exactly six waves")
    rows: list[dict[str, Any]] = []
    for source_wave in source_waves:
        key = source_wave["outer_request_sha256"]
        outcome_wave = completed.get(key)
        if outcome_wave is None:
            raise RuntimeError("Heat source/outcome join failed")
        request = decode_slate_allocation_request_record(
            source_wave.get("slate_allocation_request")
        )
        decision = policy.select(request)
        outcome_map = _outcomes(outcome_wave)
        snapshot = _affine_snapshot(source_wave["archive_snapshot"])
        subset_gains = [
            max(
                0.0,
                snapshot.augmented_hypervolume(
                    tuple(outcome_map[option_id] for option_id in subset)
                )
                - snapshot.base_hypervolume,
            )
            for subset in combinations(tuple(outcome_map), 4)
        ]
        selection = describe_heat_selection(
            option_ids=tuple(value.option_id for value in decision.selected),
            outcomes=outcome_map,
            snapshot=snapshot,
            subset_gains=subset_gains,
        )
        rows.append(
            {
                "wave_ordinal": source_wave["wave_ordinal"],
                "generation": source_wave["generation"],
                "parent_slot": source_wave["parent_slot"],
                "outer_request_sha256": key,
                "decision": _decision_summary(decision),
                "frontier_probe_k4": selection,
                "historical_model_anchored_k4": outcome_wave[
                    "historical_model_anchored_k4"
                ],
                "direct_model_top4": outcome_wave["direct_model_top4"],
                "uniform_k4": outcome_wave["uniform_k4"],
                "oracle_k4": outcome_wave["oracle_k4"],
            }
        )
    return {
        "panel_id": "heat2d",
        "workload_id": "engibench_heat2d",
        "metric_space": "dimensionless_frozen_affine_hypervolume_gain",
        "sources": [
            _source_record(HEAT_PRECOMMIT),
            _source_record(HEAT_DIAGNOSTIC),
        ],
        "waves": rows,
        "aggregate": _aggregate(
            rows,
            historical_label="historical_model_anchored_k4",
            normalized_scale=None,
        ),
    }


def _matrix(panels: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for panel in panels:
        aggregate = _object(panel["aggregate"], name="panel aggregate")
        result.append(
            {
                "panel_id": panel["panel_id"],
                "workload_id": panel["workload_id"],
                "metric_space": panel["metric_space"],
                "frontier_probe_gain_sum": aggregate[
                    "frontier_probe_gain_sum"
                ],
                "historical_gain_sum": aggregate["historical_gain_sum"],
                "direct_model_top4_gain_sum": aggregate[
                    "direct_model_top4_gain_sum"
                ],
                "uniform_expected_gain_sum": aggregate[
                    "uniform_expected_gain_sum"
                ],
                "oracle_gain_sum": aggregate["oracle_gain_sum"],
                "frontier_probe_multiple_of_uniform": aggregate[
                    "frontier_probe_multiple_of_uniform"
                ],
                "frontier_probe_fraction_of_oracle": aggregate[
                    "frontier_probe_fraction_of_oracle"
                ],
                "full_abstention_probe_waves": aggregate[
                    "full_abstention_probe_waves"
                ],
                "ideal_target_feasible_waves": aggregate[
                    "ideal_target_feasible_waves"
                ],
            }
        )
    return result


def _run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    policy = FrontierProbeSlatePolicy()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "diagnostic": "frontier_probe_three_panel_development_replay",
        "policy": policy.to_record(),
        "panels": ["boils_r8", "boils_q", "heat2d"],
        "claim_boundary": {
            "retrospective_after_all_outcomes_observed": True,
            "policy_developed_using_these_panels": True,
            "genuinely_heldout_workload": False,
            "campaign_counterfactual": False,
            "provider_calls": 0,
            "candidate_evaluations": 0,
            "api_key_reads": 0,
            "paper_ready_efficacy": False,
        },
        "source_identity": source_identity(
            (
                Path(__file__),
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/selection/frontier_probe_slate.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/selection/calibrated_slate_codec.py",
                AGENT_EVOLVE_ROOT
                / "examples/development/replay_boils_structural_posterior_allocator.py",
                AGENT_EVOLVE_ROOT
                / "examples/development/replay_boilsq_structural_posterior_allocator.py",
                AGENT_EVOLVE_ROOT
                / "examples/development/evaluate_heat2d_structural_posterior_k8.py",
                AGENT_EVOLVE_ROOT
                / "examples/development/durable_run_artifacts.py",
            ),
            relative_to=WORKSPACE_ROOT,
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    builders: tuple[
        Callable[[FrontierProbeSlatePolicy], dict[str, object]], ...
    ] = (_boils_r8_panel, _boilsq_panel, _heat_panel)
    panels = [builder(policy) for builder in builders]
    matrix = _matrix(panels)
    write_json_atomic(
        output_dir / "replay.json",
        {
            "schema_version": 1,
            "policy": policy.to_record(),
            "panels": panels,
        },
    )
    result = {
        "schema_version": 1,
        "status": "completed_retrospective_development_diagnostic",
        "result_matrix": matrix,
        "cross_panel": {
            "panel_count": len(matrix),
            "wave_count": sum(
                int(
                    _object(panel["aggregate"], name="aggregate")[
                        "wave_count"
                    ]
                )
                for panel in panels
            ),
            "panels_above_uniform": sum(
                float(row["frontier_probe_multiple_of_uniform"]) > 1.0
                for row in matrix
            ),
            "panels_at_least_98_percent_oracle": sum(
                float(row["frontier_probe_fraction_of_oracle"]) >= 0.98
                for row in matrix
            ),
            "full_abstention_probe_waves": sum(
                int(row["full_abstention_probe_waves"]) for row in matrix
            ),
            "policy_changed_direct_top4_panel_count": sum(
                row["frontier_probe_gain_sum"]
                != row["direct_model_top4_gain_sum"]
                for row in matrix
            ),
        },
        "provider_calls": 0,
        "candidate_evaluations": 0,
        "api_key_reads": 0,
        "claim_scope": (
            "authenticated_retrospective_three_panel_allocator_development_"
            "diagnostic_not_heldout_or_campaign_efficacy"
        ),
    }
    write_json_atomic(output_dir / "result.json", result)
    finalization = finalize_run_directory(
        output_dir,
        status="completed_retrospective_development_diagnostic",
    )
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(output_dir),
                "finalization_sha256": finalization["finalization_sha256"],
                "content_sha256": finalization.get(
                    "content_sha256",
                    finalization.get("recursive_content_sha256"),
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    return _run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

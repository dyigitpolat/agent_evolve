#!/usr/bin/env python3
"""Replay the frozen structural-posterior policy on the older BOiLS-Q K8 panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
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
from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)
from examples.development.evaluate_boils_calibrated_slate_counterfactual import (  # noqa: E402
    _build_plan,
    _snapshot,
)
from examples.development.replay_boils_structural_posterior_allocator import (  # noqa: E402
    RAW_HYPERVOLUME_SCALE,
    _describe_selection,
)


DEFAULT_SOURCE_RUN = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/portfolio_q"
    / "boilsq_calibrated_g6_live_deepseek_v4_20260716"
)
DEFAULT_K8_DIAGNOSTIC = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/portfolio_q"
    / "boilsq_calibrated_k8_counterfactual_r2_20260716"
)


def _object(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeError(f"{name} must be an exact JSON object")
    return value


def _array(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeError(f"{name} must be an exact JSON array")
    return value


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return _object(json.load(stream), name=path.name)


def _run(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve(strict=True)
    diagnostic = args.k8_diagnostic.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_finalization = verify_finalized_run_directory(source_run)
    diagnostic_finalization = verify_finalized_run_directory(diagnostic)
    summary = _read(source_run / "summary.json")
    _, source_waves = _build_plan(
        source_run=source_run,
        summary=summary,
        wave_rows=read_jsonl(source_run / "wave_requests.jsonl"),
    )
    requests: dict[str, Any] = {}
    for audit_value in _array(
        summary.get("portfolio_selection_audits"), name="selector audits"
    ):
        audit = _object(audit_value, name="selector audit")
        decision = _object(audit.get("decision"), name="selector decision")
        supplemental = _object(
            decision.get("supplemental_selector_audit"),
            name="supplemental selector audit",
        )
        payload = _object(supplemental.get("payload"), name="supplemental payload")
        allocation = _object(payload.get("allocation"), name="allocation")
        outer_request_sha256 = audit.get("wave_request_sha256")
        if type(outer_request_sha256) is not str:
            raise RuntimeError("selector audit omitted outer request identity")
        requests[outer_request_sha256] = decode_slate_allocation_request_record(
            allocation.get("request")
        )
    if len(requests) != 6:
        raise RuntimeError("expected six authenticated BOiLS-Q allocation requests")
    analysis = _read(diagnostic / "analysis.json")
    completed = {
        item["request_sha256"]: item
        for item in (
            _object(value, name="completed K8 wave")
            for value in _array(analysis.get("waves"), name="completed K8 waves")
        )
    }
    if len(completed) != 6:
        raise RuntimeError("expected six completed BOiLS-Q K8 waves")
    policy = StructuralPosteriorSlatePolicy()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "diagnostic": "frozen_policy_older_boilsq_panel_retrospective_replay",
        "policy": policy.to_record(),
        "source_run": {
            "path": source_run.relative_to(WORKSPACE_ROOT).as_posix(),
            "finalization_sha256": source_finalization["finalization_sha256"],
            "mutated": False,
        },
        "completed_k8_diagnostic": {
            "path": diagnostic.relative_to(WORKSPACE_ROOT).as_posix(),
            "finalization_sha256": diagnostic_finalization["finalization_sha256"],
            "mutated": False,
        },
        "claim_boundary": {
            "policy_frozen_before_this_panel_was_replayed": True,
            "outcomes_already_existed_and_replay_is_retrospective": True,
            "same_workload_different_campaign_stack": True,
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
                / "examples/development/replay_boils_structural_posterior_allocator.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/selection/structural_posterior_slate.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/selection/calibrated_slate_codec.py",
                AGENT_EVOLVE_ROOT
                / "examples/development/evaluate_boils_calibrated_slate_counterfactual.py",
            ),
            relative_to=WORKSPACE_ROOT,
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    for source_wave in source_waves:
        key = source_wave["request_sha256"]
        request = requests.get(key)
        outcome_wave = completed.get(key)
        if request is None or outcome_wave is None:
            raise RuntimeError("source, request, and outcome waves do not join")
        decision = policy.select(request)
        outcomes = {
            item["option_id"]: _object(item.get("outcomes"), name="K8 outcome")
            for item in (
                _object(value, name="K8 outcome row")
                for value in _array(
                    outcome_wave.get("all_k8_outcomes"), name="all K8 outcomes"
                )
            )
        }
        if set(outcomes) != {member.option_id for member in request.slate.members}:
            raise RuntimeError("completed outcomes differ from authenticated K8 slate")
        selection = _describe_selection(
            option_ids=tuple(value.option_id for value in decision.selected),
            outcomes=outcomes,
            snapshot=_snapshot(source_wave["archive_reward_snapshot"]),
        )
        rows.append(
            {
                "wave_ordinal": source_wave["wave_ordinal"],
                "generation": (2 * ((source_wave["wave_ordinal"] - 1) // 2)) + 1,
                "parent_slot": (source_wave["wave_ordinal"] - 1) % 2,
                "outer_request_sha256": key,
                "slate_allocation_request_sha256": request.request_sha256,
                "policy_decision_sha256": decision.decision_sha256,
                "selected": [value.to_record() for value in decision.selected],
                "selection": selection,
                "historical_four_role": outcome_wave["calibrated_k4"],
                "direct_model_top4": outcome_wave["direct_model_top4"],
                "uniform_k4": outcome_wave["uniform_k4"],
                "oracle_k4": outcome_wave["oracle_k4"],
            }
        )
    policy_gains = [row["selection"]["gain"] for row in rows]
    historical = [row["historical_four_role"]["gain"] for row in rows]
    direct = [row["direct_model_top4"]["gain"] for row in rows]
    uniform = [row["uniform_k4"]["expected_gain"] for row in rows]
    oracle = [row["oracle_k4"]["gain"] for row in rows]
    gain_sum = sum(policy_gains)
    aggregate = {
        "wave_count": len(rows),
        "structural_posterior_gain_sum": gain_sum,
        "structural_posterior_normalized_gain_sum": (
            gain_sum / RAW_HYPERVOLUME_SCALE
        ),
        "historical_four_role_gain_sum": sum(historical),
        "direct_model_top4_gain_sum": sum(direct),
        "uniform_expected_gain_sum": sum(uniform),
        "oracle_gain_sum": sum(oracle),
        "structural_posterior_minus_uniform_expected_gain_sum": (
            gain_sum - sum(uniform)
        ),
        "structural_posterior_fraction_of_oracle": gain_sum / sum(oracle),
        "wins_vs_uniform_expectation": sum(
            left > right for left, right in zip(policy_gains, uniform)
        ),
        "mean_rank_min_among_70_subsets": fmean(
            row["selection"]["rank_min"] for row in rows
        ),
        "mean_rank_max_among_70_subsets": fmean(
            row["selection"]["rank_max"] for row in rows
        ),
    }
    write_json_atomic(
        output_dir / "replay.json",
        {"schema_version": 1, "policy": policy.to_record(), "waves": rows},
    )
    result = {
        "schema_version": 1,
        "status": "completed_retrospective_holdout_diagnostic",
        "aggregate": aggregate,
        "provider_calls": 0,
        "candidate_evaluations": 0,
        "api_key_reads": 0,
        "claim_scope": (
            "older_same_workload_panel_retrospective_policy_transfer_diagnostic_"
            "not_prospective_campaign_efficacy"
        ),
    }
    write_json_atomic(output_dir / "result.json", result)
    finalization = finalize_run_directory(
        output_dir,
        status="completed_retrospective_holdout_diagnostic",
    )
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(output_dir),
                "finalization_sha256": finalization["finalization_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--k8-diagnostic", type=Path, default=DEFAULT_K8_DIAGNOSTIC
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))

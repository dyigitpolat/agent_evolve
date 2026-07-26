#!/usr/bin/env python3
"""Replay the typed structural-posterior allocator on sealed BOiLS K8 slates.

This is an explicitly retrospective fixed-parent diagnostic.  It authenticates
the original campaign and its completed K8 outcome panel, decodes each persisted
``SlateAllocationRequest`` exactly, applies the versioned provider-free policy,
and reports local hypervolume.  It performs no model or candidate evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
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
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)
from examples.development.evaluate_boils_generic_campaign_k8 import (  # noqa: E402
    RAW_HYPERVOLUME_SCALE,
    _build_plan,
    _campaign_events,
    _snapshot,
)


DEFAULT_SOURCE_RUN = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/generic_campaign"
    / "boils_generic_g6_deepseek_live_unmatched_dev_r8_20260717"
)
DEFAULT_K8_DIAGNOSTIC = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/generic_campaign"
    / "boils_r8_complete_k8_diagnostic_r2_20260717"
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


def _allocation_requests_by_outer_request(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for event in events:
        if event.get("kind") != "stage_sealed":
            continue
        stage = _object(
            _object(event.get("payload"), name="stage payload").get(
                "stage_receipt"
            ),
            name="stage receipt",
        )
        generation = stage.get("generation")
        if type(generation) is not int or generation % 2 == 0:
            continue
        for audit_value in _array(stage.get("selector_audits"), name="selector audits"):
            audit = _object(audit_value, name="selector audit")
            plaintext = _object(
                audit.get("plaintext_audit"), name="plaintext selector audit"
            )
            response_text = plaintext.get("response_text")
            if type(response_text) is not str:
                raise RuntimeError("selector audit omitted its trusted response")
            response = _object(json.loads(response_text), name="selector response")
            supplemental = _object(
                response.get("supplemental_selector_audit"),
                name="supplemental selector audit",
            )
            payload = _object(
                supplemental.get("payload"), name="supplemental payload"
            )
            allocation = _object(payload.get("allocation"), name="allocation")
            outer_request_sha256 = audit.get("request_sha256")
            if type(outer_request_sha256) is not str:
                raise RuntimeError("selector audit omitted outer request identity")
            if outer_request_sha256 in result:
                raise RuntimeError("outer selector request identity is duplicated")
            result[outer_request_sha256] = decode_slate_allocation_request_record(
                allocation.get("request")
            )
    if len(result) != 6:
        raise RuntimeError("expected exactly six authenticated allocation requests")
    return result


def _completed_outcome_waves(diagnostic: Path) -> dict[str, dict[str, Any]]:
    analysis = _read_json(diagnostic / "allocation_analysis.json")
    waves = {
        value["request_sha256"]: value
        for value in (
            _object(item, name="completed K8 wave")
            for item in _array(analysis.get("waves"), name="completed K8 waves")
        )
    }
    if len(waves) != 6:
        raise RuntimeError("completed K8 diagnostic must contain six waves")
    return waves


def _describe_selection(
    *,
    option_ids: tuple[str, ...],
    outcomes: dict[str, dict[str, float]],
    snapshot: Any,
) -> dict[str, Any]:
    all_ids = tuple(outcomes)
    subset_rows: list[tuple[float, tuple[str, ...]]] = []
    for subset in combinations(all_ids, 4):
        augmented = snapshot.augmented_hypervolume(
            tuple(outcomes[option_id] for option_id in subset)
        )
        subset_rows.append(
            (max(0.0, augmented - snapshot.base_hypervolume), subset)
        )
    gain = max(
        0.0,
        snapshot.augmented_hypervolume(
            tuple(outcomes[option_id] for option_id in option_ids)
        )
        - snapshot.base_hypervolume,
    )
    gains = [value[0] for value in subset_rows]
    better = sum(value > gain for value in gains)
    ties = sum(value == gain for value in gains)
    return {
        "option_ids": list(option_ids),
        "gain": gain,
        "normalized_gain": gain / RAW_HYPERVOLUME_SCALE,
        "rank_min": better + 1,
        "rank_max": better + ties,
        "strictly_better_than_uniform_fraction": (
            sum(gain > value for value in gains) / len(gains)
        ),
        "uniform_support_size": len(gains),
    }


def _run(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve(strict=True)
    diagnostic = args.k8_diagnostic.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_finalization = verify_finalized_run_directory(source_run)
    diagnostic_finalization = verify_finalized_run_directory(diagnostic)
    events = _campaign_events(source_run)
    _, source_waves, accounting = _build_plan(events=events)
    requests = _allocation_requests_by_outer_request(events)
    completed_waves = _completed_outcome_waves(diagnostic)
    policy = StructuralPosteriorSlatePolicy()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "diagnostic": "typed_structural_posterior_allocator_retrospective_replay",
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
        "completed_support_accounting": accounting,
        "claim_boundary": {
            "retrospective_after_outcomes_observed": True,
            "fixed_parent_fixed_k8_local_mechanism_diagnostic": True,
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
                / "src/agent_evolve/policies/selection/structural_posterior_slate.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/selection/calibrated_slate_codec.py",
                AGENT_EVOLVE_ROOT
                / "examples/development/evaluate_boils_generic_campaign_k8.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/reward/frozen_wave_archive.py",
                AGENT_EVOLVE_ROOT
                / "examples/development/durable_run_artifacts.py",
            ),
            relative_to=WORKSPACE_ROOT,
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    for source_wave in source_waves:
        outer_request_sha256 = source_wave["request_sha256"]
        request = requests.get(outer_request_sha256)
        completed = completed_waves.get(outer_request_sha256)
        if request is None or completed is None:
            raise RuntimeError("source, request, and completed K8 waves do not join")
        decision = policy.select(request)
        completed_outcomes = {
            item["option_id"]: _object(item.get("outcomes"), name="K8 outcome")
            for item in (
                _object(value, name="K8 outcome row")
                for value in _array(
                    completed.get("all_k8_outcomes"), name="all K8 outcomes"
                )
            )
        }
        if set(completed_outcomes) != {
            member.option_id for member in request.slate.members
        }:
            raise RuntimeError("completed outcomes differ from authenticated K8 slate")
        snapshot = _snapshot(source_wave["archive_reward_snapshot"])
        selected_ids = tuple(value.option_id for value in decision.selected)
        selection = _describe_selection(
            option_ids=selected_ids,
            outcomes=completed_outcomes,
            snapshot=snapshot,
        )
        rows.append(
            {
                "wave_ordinal": source_wave["wave_ordinal"],
                "generation": source_wave["generation"],
                "parent_slot": (source_wave["wave_ordinal"] - 1) % 2,
                "outer_request_sha256": outer_request_sha256,
                "slate_allocation_request_sha256": request.request_sha256,
                "policy_decision_sha256": decision.decision_sha256,
                "prior_only": decision.prior_only,
                "selected": [value.to_record() for value in decision.selected],
                "selection": selection,
                "historical_model_anchored": completed["calibrated_k4"],
                "direct_model_top4": completed["direct_model_top4"],
                "uniform_k4": completed["uniform_k4"],
                "oracle_k4": completed["oracle_k4"],
            }
        )
    if len(rows) != 6:
        raise RuntimeError("typed replay must produce exactly six decisions")
    structural_gains = [value["selection"]["gain"] for value in rows]
    historical_gains = [value["historical_model_anchored"]["gain"] for value in rows]
    direct_gains = [value["direct_model_top4"]["gain"] for value in rows]
    uniform_gains = [value["uniform_k4"]["expected_gain"] for value in rows]
    oracle_gains = [value["oracle_k4"]["gain"] for value in rows]
    structural_sum = sum(structural_gains)
    historical_sum = sum(historical_gains)
    uniform_sum = sum(uniform_gains)
    oracle_sum = sum(oracle_gains)
    aggregate = {
        "wave_count": len(rows),
        "structural_posterior_gain_sum": structural_sum,
        "structural_posterior_normalized_gain_sum": (
            structural_sum / RAW_HYPERVOLUME_SCALE
        ),
        "historical_model_anchored_gain_sum": historical_sum,
        "direct_model_top4_gain_sum": sum(direct_gains),
        "uniform_expected_gain_sum": uniform_sum,
        "oracle_gain_sum": oracle_sum,
        "structural_posterior_minus_historical_gain_sum": (
            structural_sum - historical_sum
        ),
        "structural_posterior_minus_uniform_expected_gain_sum": (
            structural_sum - uniform_sum
        ),
        "structural_posterior_fraction_of_oracle": structural_sum / oracle_sum,
        "structural_posterior_multiple_of_historical": (
            structural_sum / historical_sum
        ),
        "wins_vs_uniform_expectation": sum(
            left > right for left, right in zip(structural_gains, uniform_gains)
        ),
        "oracle_ties": sum(
            left == right for left, right in zip(structural_gains, oracle_gains)
        ),
        "mean_rank_min_among_70_subsets": fmean(
            value["selection"]["rank_min"] for value in rows
        ),
        "mean_rank_max_among_70_subsets": fmean(
            value["selection"]["rank_max"] for value in rows
        ),
    }
    write_json_atomic(
        output_dir / "replay.json",
        {"schema_version": 1, "policy": policy.to_record(), "waves": rows},
    )
    result = {
        "schema_version": 1,
        "status": "completed_retrospective_diagnostic",
        "aggregate": aggregate,
        "provider_calls": 0,
        "candidate_evaluations": 0,
        "api_key_reads": 0,
        "claim_scope": (
            "authenticated_retrospective_fixed_parent_k8_allocator_diagnostic_"
            "not_prospective_campaign_efficacy"
        ),
    }
    write_json_atomic(output_dir / "result.json", result)
    finalization = finalize_run_directory(
        output_dir,
        status="completed_retrospective_diagnostic",
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
        "--k8-diagnostic",
        type=Path,
        default=DEFAULT_K8_DIAGNOSTIC,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    return _run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

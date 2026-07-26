#!/usr/bin/env python3
"""Freeze one portable T-RAP profile and all-panel meta-prior.

The development selection uses the closed artifact-343 grid.  Every candidate
setting is evaluated with leave-one-case-out meta-priors and one unchanged
setting across all held-out routes.  The precommitted winner is then fit once
to all 288 development outcomes and exported as serializable sufficient
statistics for prospective campaigns.

This script does not establish independent efficacy or SOTA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development import analyze_trap_prequential_k8 as trap  # noqa: E402
from examples.development.durable_run_artifacts import write_json_atomic  # noqa: E402
from agent_evolve.policies.selection.target_conditioned_prequential import (  # noqa: E402
    PrequentialLinearGaussianHead,
    TargetConditionedAcquisitionProfile,
    TargetConditionedMetaPrior,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts"
)
FREEZE_PATH = (
    ARTIFACT_ROOT
    / "345_trap_portable_profile_and_meta_prior_freeze_20260722.md"
)
DEFAULT_OUTPUT = ARTIFACT_ROOT / "data/trap_portable_profile_v1.json"
FROZEN_PARAMETERS = trap.Hyperparameters(
    alpha=10.0,
    meta_precision=0.25,
    direction_weight=0.5,
    uncertainty_weight=0.25,
)
FROZEN_MAXIMUM_REMAINING_HORIZON = 2
TRAINING_MANIFEST_DOMAIN = b"agent-evolve:trap-portable-training-manifest:v1\x00"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _prepared_panel() -> tuple[trap.PreparedWave, ...]:
    values: list[trap.PreparedWave] = []
    for spec in trap.gate._case_specs():  # noqa: SLF001
        waves, _ = trap._prepare_case(spec)  # noqa: SLF001
        values.extend(waves)
    return tuple(values)


def _grid_audit(
    all_waves: tuple[trap.PreparedWave, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_ids = tuple(sorted({value.wave.case_id for value in all_waves}))
    rows: list[dict[str, Any]] = []
    for parameters in trap._parameter_grid(trap.POLICIES[0]):  # noqa: SLF001
        cases: list[dict[str, Any]] = []
        for held_out in case_ids:
            simulated = trap._simulate_case(  # noqa: SLF001
                all_waves=all_waves,
                train_case_ids=tuple(
                    value for value in case_ids if value != held_out
                ),
                test_case_id=held_out,
                policy=trap.POLICIES[0],
                parameters=parameters,
            )
            first = simulated[0]
            cases.append(
                {
                    "case_id": held_out,
                    "workload": first["workload"],
                    "model": first["model"],
                    "oracle_fraction": trap._case_oracle_fraction(  # noqa: SLF001
                        simulated
                    ),
                }
            )
        rows.append(
            {
                "hyperparameters": parameters.as_dict(),
                "mean_case_oracle_fraction": fmean(
                    value["oracle_fraction"] for value in cases
                ),
                "case_summary": cases,
            }
        )
    winner = max(
        rows,
        key=lambda value: (
            float(value["mean_case_oracle_fraction"]),
            float(value["hyperparameters"]["alpha"]),
            float(value["hyperparameters"]["meta_precision"]),
            -float(value["hyperparameters"]["direction_weight"]),
            -float(value["hyperparameters"]["uncertainty_weight"]),
        ),
    )
    if winner["hyperparameters"] != FROZEN_PARAMETERS.as_dict():
        raise RuntimeError(
            "closed-grid winner drifted from artifact 345: "
            f"observed={winner['hyperparameters']}, "
            f"frozen={FROZEN_PARAMETERS.as_dict()}"
        )
    return rows, winner


def _portable_head(value: trap.SequentialRidgeHead) -> PrequentialLinearGaussianHead:
    return PrequentialLinearGaussianHead(
        feature_names=value.feature_names,
        means=tuple(float(item) for item in value.means),
        scales=tuple(float(item) for item in value.scales),
        precision=tuple(
            tuple(float(item) for item in row) for row in value.precision
        ),
        rhs=tuple(float(item) for item in value.rhs),
        residual_variance=float(value.residual_variance),
    )


def freeze() -> dict[str, Any]:
    all_waves = _prepared_panel()
    case_ids = tuple(sorted({value.wave.case_id for value in all_waves}))
    grid_rows, winner = _grid_audit(all_waves)
    marginal = _portable_head(
        trap._fit_prior(  # noqa: SLF001
            waves=all_waves,
            feature_names=trap.FULL_FEATURES,
            target_kind="marginal",
            alpha=FROZEN_PARAMETERS.alpha,
            meta_precision=FROZEN_PARAMETERS.meta_precision,
        )
    )
    direction = _portable_head(
        trap._fit_prior(  # noqa: SLF001
            waves=all_waves,
            feature_names=trap.FULL_FEATURES,
            target_kind="direction",
            alpha=FROZEN_PARAMETERS.alpha,
            meta_precision=FROZEN_PARAMETERS.meta_precision,
        )
    )
    profile = TargetConditionedAcquisitionProfile(
        direction_weight=FROZEN_PARAMETERS.direction_weight,
        uncertainty_weight=FROZEN_PARAMETERS.uncertainty_weight,
        maximum_remaining_horizon=FROZEN_MAXIMUM_REMAINING_HORIZON,
    )
    training_manifest = {
        "schema_version": 1,
        "source_panel_path": str(
            trap.SOURCE_PANEL_PATH.relative_to(WORKSPACE_ROOT)
        ),
        "source_panel_sha256": _sha256(trap.SOURCE_PANEL_PATH),
        "target_analysis_path": str(
            Path(trap.__file__).resolve().relative_to(WORKSPACE_ROOT)
        ),
        "target_analysis_sha256": _sha256(Path(trap.__file__).resolve()),
        "freeze_protocol_path": str(FREEZE_PATH.relative_to(WORKSPACE_ROOT)),
        "freeze_protocol_sha256": _sha256(FREEZE_PATH),
        "case_ids": list(case_ids),
        "wave_count": len(all_waves),
        "candidate_outcome_count": sum(
            len(value.wave.members) for value in all_waves
        ),
        "feature_names": list(trap.FULL_FEATURES),
        "ridge_alpha_hex": FROZEN_PARAMETERS.alpha.hex(),
        "meta_precision_hex": FROZEN_PARAMETERS.meta_precision.hex(),
        "labels": {
            "marginal": "normalized_singleton_archive_utility_gain",
            "direction": (
                "normalized_assigned_target_achievement_improvement_clipped"
            ),
        },
        "workload_model_provider_option_identifiers_enter_features": False,
    }
    training_data_sha256 = _hash(
        TRAINING_MANIFEST_DOMAIN, training_manifest
    )
    meta_prior = TargetConditionedMetaPrior(
        training_data_sha256=training_data_sha256,
        marginal_head=marginal,
        direction_head=direction,
    )
    return {
        "schema_version": 1,
        "artifact_id": "trap_portable_profile_v1",
        "claim_boundary": (
            "development-selected portable profile and all-panel prior; "
            "not independent efficacy or SOTA evidence"
        ),
        "freeze_protocol": {
            "path": str(FREEZE_PATH.relative_to(WORKSPACE_ROOT)),
            "sha256": _sha256(FREEZE_PATH),
            "prospective_outcomes_consulted": False,
        },
        "selection_protocol": {
            "kind": "one_setting_leave_one_case_out_closed_grid",
            "same_hyperparameters_across_all_held_out_cases": True,
            "grid": {
                "alpha": list(trap.ALPHA_GRID),
                "meta_precision": list(trap.META_PRECISION_GRID),
                "direction_weight": list(trap.DIRECTION_WEIGHT_GRID),
                "uncertainty_weight": list(trap.UNCERTAINTY_WEIGHT_GRID),
            },
            "winner": winner,
            "all_grid_rows": grid_rows,
        },
        "profile": profile.to_record(),
        "meta_prior": meta_prior.to_record(),
        "training_manifest": training_manifest,
        "feature_contract": {
            "feature_count": len(trap.FULL_FEATURES),
            "feature_names": list(trap.FULL_FEATURES),
            "missing_feature_policy": "fail_closed",
            "extra_feature_policy": "fail_closed",
            "nonfinite_feature_policy": "fail_closed",
            "prospective_projector_parity_required": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = freeze()
    write_json_atomic(arguments.output, result)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "profile_sha256": result["profile"]["profile_sha256"],
                "meta_prior_sha256": result["meta_prior"][
                    "meta_prior_sha256"
                ],
                "mean_case_oracle_fraction": result["selection_protocol"][
                    "winner"
                ]["mean_case_oracle_fraction"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

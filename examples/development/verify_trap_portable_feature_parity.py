#!/usr/bin/env python3
"""Replay all 288 development rows through the production T-RAP projector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development import analyze_trap_prequential_k8 as trap  # noqa: E402
from examples.development.durable_run_artifacts import write_json_atomic  # noqa: E402
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256  # noqa: E402
from agent_evolve.policies.selection.calibrated_slate import (  # noqa: E402
    CalibratedSlate,
    CalibratedSlateMember,
    MetricOptimizationGoal,
    SlateAllocationRequest,
    SlateMetricObjective,
    SlateRoleProposal,
    SlateStructuralEvidence,
)
from agent_evolve.policies.selection.forecast_calibration import (  # noqa: E402
    BetaCorrectnessPrior,
    ForecastCalibrationCell,
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
)
from agent_evolve.policies.selection.structural_posterior_slate import (  # noqa: E402
    StructuralPosteriorMemberScoreRow,
    StructuralPosteriorMetricScoreRow,
)
from agent_evolve.policies.selection.target_conditioned_features import (  # noqa: E402
    FEATURE_NAMES,
    PROJECTOR_DEFINITION_SHA256,
    TargetConditionedFeatureProjectionRequest,
    TargetConditionedPortableFeatureProjector,
    project_portable_transition,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection  # noqa: E402
from agent_evolve.ports.archive_context import (  # noqa: E402
    CampaignPortfolioArchiveContextProjection,
)
from agent_evolve.ports.frontier_target import (  # noqa: E402
    CampaignPortfolioFrontierTarget,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts"
)
DEFAULT_OUTPUT = ARTIFACT_ROOT / "data/trap_portable_feature_parity_v1.json"
ABSOLUTE_TOLERANCE = 1e-12


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hex(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be binary64 hexadecimal text")
    result = float.fromhex(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _score_row(raw: dict[str, Any]) -> StructuralPosteriorMemberScoreRow:
    metrics = []
    for metric in raw["metric_scores"]:
        cell = metric["calibration_cell"]
        prior = cell["prior"]
        metrics.append(
            StructuralPosteriorMetricScoreRow(
                metric_id=metric["metric_id"],
                goal=MetricOptimizationGoal(metric["goal"]),
                asserted_direction=MetricEffectDirection(
                    metric["asserted_direction"]
                ),
                confidence=ForecastConfidenceBin(metric["confidence"]),
                weight=_hex(metric["weight_hex"], name="metric weight"),
                calibration_cell=ForecastCalibrationCell(
                    metric_id=cell["metric_id"],
                    asserted_direction=MetricEffectDirection(
                        cell["asserted_direction"]
                    ),
                    confidence=ForecastConfidenceBin(cell["confidence"]),
                    family=cell["family"],
                    observation_count=int(cell["observation_count"]),
                    scorable_count=int(cell["scorable_count"]),
                    correct_count=int(cell["correct_count"]),
                    prior=BetaCorrectnessPrior(
                        alpha=_hex(prior["alpha_hex"], name="prior alpha"),
                        beta=_hex(prior["beta_hex"], name="prior beta"),
                    ),
                ),
                calibration_source=metric["calibration_source"],
                favorable_assertion=metric["favorable_assertion"],
                adverse_assertion=metric["adverse_assertion"],
                signed_exploitation_score=_hex(
                    metric["signed_exploitation_score_hex"],
                    name="signed exploitation",
                ),
                posterior_uncertainty_score=_hex(
                    metric["posterior_uncertainty_score_hex"],
                    name="posterior uncertainty",
                ),
                calibrated_disagreement_score=_hex(
                    metric["calibrated_disagreement_score_hex"],
                    name="calibrated disagreement",
                ),
                epistemic_score=_hex(
                    metric["epistemic_score_hex"], name="epistemic score"
                ),
                explicit_abstention=metric["explicit_abstention"],
            )
        )
    return StructuralPosteriorMemberScoreRow(
        option_id=raw["option_id"],
        option_identity_sha256=raw["option_identity_sha256"],
        model_rank=int(raw["model_rank"]),
        metric_scores=tuple(metrics),
        calibrated_exploitation_score=_hex(
            raw["calibrated_exploitation_score_hex"],
            name="calibrated exploitation",
        ),
        calibrated_frontier_score=_hex(
            raw["calibrated_frontier_score_hex"], name="calibrated frontier"
        ),
        raw_epistemic_score=_hex(
            raw["raw_epistemic_score_hex"], name="raw epistemic"
        ),
        structural_coverage_score=_hex(
            raw["structural_coverage_score_hex"], name="structural coverage"
        ),
        epistemic_structural_score=_hex(
            raw["epistemic_structural_score_hex"],
            name="epistemic structural",
        ),
        model_declared_assigned_card_keys=tuple(
            raw["model_declared_assigned_card_keys"]
        ),
    )


def _allocation_request(
    *, spec: trap.gate.CaseSpec, raw_wave: dict[str, Any]
) -> tuple[SlateAllocationRequest, tuple[StructuralPosteriorMemberScoreRow, ...]]:
    ordinal = int(raw_wave["wave_ordinal"])
    scope = ForecastCalibrationScope(
        model_profile_sha256=_sha(f"parity:{spec.case_id}:model"),
        prompt_definition_sha256=_sha(f"parity:{spec.case_id}:prompt"),
        selector_policy_definition_sha256=_sha(
            f"parity:{spec.case_id}:selector"
        ),
        benchmark_sha256=_sha(f"parity:{spec.workload}:benchmark"),
        session_sha256=_sha(f"parity:{spec.case_id}:session"),
    )
    selector_sha256 = _sha(f"parity:{spec.case_id}:{ordinal}:selector-decision")
    parent_configuration = freeze_json(raw_wave["parent_configuration"])
    parent_sha256 = typed_json_sha256(parent_configuration)
    request_members = trap.gate._request_member_by_id(  # noqa: SLF001
        spec.workload, raw_wave
    )
    score_records = trap.gate._score_row_by_id(  # noqa: SLF001
        spec.workload, raw_wave
    )
    raw_members = {
        value["option_id"]: value for value in raw_wave["members"]
    }
    members = []
    for raw_member in sorted(
        raw_wave["members"], key=lambda value: int(value["model_rank"])
    ):
        option_id = raw_member["option_id"]
        request_member = request_members[option_id]
        predictions = tuple(
            ForecastPredictionReceipt(
                scope=scope,
                wave_index=ordinal,
                selector_decision_sha256=selector_sha256,
                parent_candidate_identity_sha256=parent_sha256,
                option_id=option_id,
                option_identity_sha256=raw_member["option_identity_sha256"],
                family=raw_member["family"],
                metric_id=value["metric_id"],
                asserted_direction=MetricEffectDirection(
                    value["asserted_direction"]
                ),
                confidence=ForecastConfidenceBin(value["confidence"]),
            )
            for value in sorted(
                request_member["predictions"],
                key=lambda item: item["metric_id"],
            )
        )
        structural = request_member["structural_evidence"]
        members.append(
            CalibratedSlateMember(
                model_rank=int(raw_member["model_rank"]),
                option_id=option_id,
                option_identity_sha256=raw_member["option_identity_sha256"],
                family=raw_member["family"],
                locus_key=request_member["locus_key"],
                phenotype_identity_sha256=request_member.get(
                    "phenotype_identity_sha256",
                    _sha(f"parity:{spec.case_id}:{ordinal}:{option_id}:phenotype"),
                ),
                supporting_card_keys=tuple(
                    sorted(request_member.get("supporting_card_keys", ()))
                ),
                role_proposal=SlateRoleProposal(request_member["role_proposal"]),
                rationale_sha256=_sha(
                    f"parity:{spec.case_id}:{ordinal}:{option_id}:rationale"
                ),
                predictions=predictions,
                structural_evidence=SlateStructuralEvidence(
                    frozen_archive_snapshot_sha256=structural[
                        "frozen_archive_snapshot_sha256"
                    ],
                    evidence_receipt_sha256=structural[
                        "evidence_receipt_sha256"
                    ],
                    archive_novelty_score=_hex(
                        structural["archive_novelty_score_hex"],
                        name="archive novelty",
                    ),
                    structural_coverage_score=_hex(
                        structural["structural_coverage_score_hex"],
                        name="raw structural coverage",
                    ),
                ),
            )
        )
    first_scores = score_records[next(iter(raw_members))]["metric_scores"]
    objectives = tuple(
        SlateMetricObjective(
            metric_id=value["metric_id"],
            goal=MetricOptimizationGoal(value["goal"]),
            weight=_hex(value["weight_hex"], name="objective weight"),
            definition_sha256=_sha(
                f"parity:{spec.case_id}:{value['metric_id']}:objective"
            ),
        )
        for value in sorted(first_scores, key=lambda item: item["metric_id"])
    )
    assigned_cards = tuple(
        sorted(
            {
                card
                for member in members
                for card in member.supporting_card_keys
            }
        )
    )
    if not assigned_cards:
        assigned_cards = ("parity.empty",)
    allocation = SlateAllocationRequest(
        slate=CalibratedSlate(
            scope=scope,
            wave_index=ordinal,
            selector_decision_sha256=selector_sha256,
            parent_candidate_identity_sha256=parent_sha256,
            finite_contract_sha256=_sha(
                f"parity:{spec.case_id}:{ordinal}:finite-contract"
            ),
            members=tuple(members),
        ),
        portfolio_size=4,
        objectives=objectives,
        assigned_card_keys=assigned_cards,
    )
    score_rows = tuple(
        _score_row(score_records[option_id])
        for option_id in sorted(score_records)
    )
    return allocation, score_rows


def _projection_request(
    *,
    spec: trap.gate.CaseSpec,
    raw_wave: dict[str, Any],
    prepared: trap.PreparedWave,
) -> TargetConditionedFeatureProjectionRequest:
    allocation, scores = _allocation_request(spec=spec, raw_wave=raw_wave)
    raw_members = {
        value["option_id"]: value for value in raw_wave["members"]
    }
    parent_configuration = freeze_json(raw_wave["parent_configuration"])
    transitions = tuple(
        project_portable_transition(
            option_id=member.option_id,
            option_identity_sha256=member.option_identity_sha256,
            parent_configuration=parent_configuration,
            child_configuration=freeze_json(raw_members[member.option_id]["configuration"]),
        )
        for member in sorted(allocation.slate.members, key=lambda value: value.option_id)
    )
    metric_ids, archive, _axes = trap._geometry(  # noqa: SLF001
        spec.workload, raw_wave
    )
    directions = trap._reference_directions(len(metric_ids))  # noqa: SLF001
    parent_point = prepared.target.parent_point
    archive_sha256 = _sha(
        f"parity:{spec.case_id}:{prepared.wave.generation}:archive-utility"
    )
    base_hypervolume = prepared.wave.members[0].features[
        "archive_base_hypervolume"
    ]
    archive_context = CampaignPortfolioArchiveContextProjection(
        projector_id="parity.affine",
        projector_version=1,
        definition_sha256=_sha("parity affine context"),
        archive_utility_snapshot_sha256=archive_sha256,
        parent_configuration_sha256=(
            allocation.slate.parent_candidate_identity_sha256
        ),
        payload=freeze_json(
            {
                "optimization_frame": {
                    "axes": [
                        {"metric_id": metric_id} for metric_id in metric_ids
                    ],
                    "reference_directions": [
                        {
                            "direction_id": direction_id,
                            "normalized_importance_decimal": [
                                format(value, ".17g") for value in weights
                            ],
                        }
                        for direction_id, weights in directions
                    ],
                    "base_hypervolume_decimal": format(
                        base_hypervolume, ".17g"
                    ),
                },
                "archive": {
                    "normalized_points_decimal": [
                        [format(value, ".17g") for value in point]
                        for point in archive
                    ]
                },
                "parent": {
                    "normalized_point_decimal": [
                        format(value, ".17g") for value in parent_point
                    ]
                },
            }
        ),
    )
    target = prepared.target
    frontier_target = CampaignPortfolioFrontierTarget(
        allocator_id="parity",
        allocator_version=1,
        definition_sha256=_sha("parity target allocator"),
        archive_utility_snapshot_sha256=archive_sha256,
        lane_id=f"lane.{prepared.wave.parent_slot}",
        parent_configuration_sha256=(
            allocation.slate.parent_candidate_identity_sha256
        ),
        direction_id=target.direction_id,
        opportunity_rank=target.opportunity_rank,
        payload=freeze_json(
            {
                "target_direction": {
                    "normalized_weights_decimal": [
                        format(value, ".17g") for value in target.weights
                    ],
                    "opportunity_from_ideal_decimal": format(
                        target.opportunity_from_ideal, ".17g"
                    ),
                },
                "assigned_parent": {
                    "normalized_point_decimal": [
                        format(value, ".17g") for value in target.parent_point
                    ],
                    "achievement_decimal": format(
                        target.parent_achievement, ".17g"
                    ),
                    "regret_above_archive_best_decimal": format(
                        target.parent_regret, ".17g"
                    ),
                },
            }
        ),
    )
    return TargetConditionedFeatureProjectionRequest(
        allocation_request=allocation,
        structural_score_rows=scores,
        transition_receipts=transitions,
        archive_context=archive_context,
        frontier_target=frontier_target,
        campaign_generation=prepared.wave.generation,
        lane_slot=prepared.wave.parent_slot,
        remaining_proposal_horizon=target.remaining_proposal_horizon,
    )


def verify() -> dict[str, Any]:
    if FEATURE_NAMES != trap.FULL_FEATURES:
        raise RuntimeError("production and development feature names differ")
    projector = TargetConditionedPortableFeatureProjector()
    mismatches: list[dict[str, Any]] = []
    maximum_absolute_error = 0.0
    scalar_count = 0
    row_count = 0
    cases = []
    for spec in trap.gate._case_specs():  # noqa: SLF001
        raw_waves = trap.gate._reconstructed_waves(spec)  # noqa: SLF001
        raw_by_ordinal = {
            int(value["wave_ordinal"]): value for value in raw_waves
        }
        prepared_waves, _ = trap._prepare_case(spec)  # noqa: SLF001
        case_maximum = 0.0
        for prepared in prepared_waves:
            request = _projection_request(
                spec=spec,
                raw_wave=raw_by_ordinal[prepared.wave.wave_ordinal],
                prepared=prepared,
            )
            projected = projector.project(request)
            expected = {
                value.option_id: value.features for value in prepared.wave.members
            }
            for row in projected:
                row_count += 1
                observed = dict(zip(FEATURE_NAMES, row.values, strict=True))
                for name in FEATURE_NAMES:
                    scalar_count += 1
                    error = abs(observed[name] - expected[row.option_id][name])
                    maximum_absolute_error = max(maximum_absolute_error, error)
                    case_maximum = max(case_maximum, error)
                    if error > ABSOLUTE_TOLERANCE and len(mismatches) < 100:
                        mismatches.append(
                            {
                                "case_id": spec.case_id,
                                "wave_ordinal": prepared.wave.wave_ordinal,
                                "option_id": row.option_id,
                                "feature": name,
                                "expected": expected[row.option_id][name],
                                "observed": observed[name],
                                "absolute_error": error,
                            }
                        )
        cases.append(
            {
                "case_id": spec.case_id,
                "workload": spec.workload,
                "model": spec.model,
                "maximum_absolute_error": case_maximum,
            }
        )
    passed = not mismatches
    return {
        "schema_version": 1,
        "verification_id": "trap_portable_feature_parity_v1",
        "passed": passed,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "case_count": len(cases),
        "wave_count": 36,
        "candidate_row_count": row_count,
        "feature_count": len(FEATURE_NAMES),
        "scalar_comparison_count": scalar_count,
        "maximum_absolute_error": maximum_absolute_error,
        "mismatch_count_over_tolerance": len(mismatches),
        "first_mismatches": mismatches,
        "case_summary": cases,
        "projector": projector.to_record(),
        "provenance": {
            "source_panel_path": str(
                trap.SOURCE_PANEL_PATH.relative_to(WORKSPACE_ROOT)
            ),
            "source_panel_sha256": _sha256(trap.SOURCE_PANEL_PATH),
            "development_analysis_sha256": _sha256(Path(trap.__file__).resolve()),
            "projector_definition_sha256": PROJECTOR_DEFINITION_SHA256,
        },
        "claim_boundary": (
            "feature normalization parity only; not selector efficacy, "
            "campaign efficacy, or SOTA evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = verify()
    write_json_atomic(arguments.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "passed",
                    "candidate_row_count",
                    "scalar_comparison_count",
                    "maximum_absolute_error",
                    "mismatch_count_over_tolerance",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

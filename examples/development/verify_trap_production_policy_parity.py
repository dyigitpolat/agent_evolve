#!/usr/bin/env python3
"""Verify production T-RAP decisions against the frozen universal replay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from examples.development import verify_trap_portable_feature_parity as feature_parity  # noqa: E402
from examples.development.durable_run_artifacts import write_json_atomic  # noqa: E402
from agent_evolve.policies.selection.calibrated_slate import (  # noqa: E402
    SlateAllocationRequest,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (  # noqa: E402
    RealizablePortfolioProjector,
    StructuralScoreProjector,
    TargetConditionedAllocationContext,
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.structural_posterior_slate import (  # noqa: E402
    StructuralPosteriorMemberScoreRow,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (  # noqa: E402
    PrequentialLinearGaussianHead,
    RealizablePortfolioSet,
    TargetConditionedAcquisitionProfile,
    TargetConditionedAcquisitionState,
    TargetConditionedSelectedObservation,
    update_target_conditioned_state,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts"
)
DEFAULT_OUTPUT = ARTIFACT_ROOT / "data/trap_production_policy_parity_v1.json"
PARAMETERS = trap.Hyperparameters(10.0, 0.25, 0.5, 0.25)
PROFILE = TargetConditionedAcquisitionProfile(
    direction_weight=PARAMETERS.direction_weight,
    uncertainty_weight=PARAMETERS.uncertainty_weight,
    maximum_remaining_horizon=2,
)
REALIZABILITY_ID = "historical_executable_k4_parity"
REALIZABILITY_VERSION = 1
REALIZABILITY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:historical-executable-k4-parity:v1;"
    b"source=authenticated-development-wave-feasible-subsets;"
    b"purpose=production-policy-parity-only;prospective-use=false"
).hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head(value: trap.SequentialRidgeHead) -> PrequentialLinearGaussianHead:
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


@dataclass(frozen=True, slots=True)
class _ContextProvider:
    context: TargetConditionedAllocationContext
    provider_id: str = "historical_parity_context"
    provider_version: int = 1
    definition_sha256: str = hashlib.sha256(
        b"agent-evolve:historical-parity-context:v1"
    ).hexdigest()

    def context_for(
        self, request: SlateAllocationRequest
    ) -> TargetConditionedAllocationContext:
        self.context.require_request(request)
        return self.context


@dataclass(frozen=True, slots=True)
class _HistoricalRealizabilityProjector:
    option_id_sets: tuple[tuple[str, ...], ...]
    provider_id: str = REALIZABILITY_ID
    provider_version: int = REALIZABILITY_VERSION
    definition_sha256: str = REALIZABILITY_DEFINITION_SHA256

    def project(self, request: SlateAllocationRequest) -> RealizablePortfolioSet:
        request.revalidate()
        return RealizablePortfolioSet(
            source_request_sha256=request.request_sha256,
            projector_id=self.provider_id,
            projector_version=self.provider_version,
            projector_definition_sha256=self.definition_sha256,
            option_id_sets=self.option_id_sets,
        )


@dataclass(frozen=True, slots=True)
class _RecordedStructuralScoreProjector:
    rows: tuple[StructuralPosteriorMemberScoreRow, ...]
    provider_id: str = "recorded_structural_scores_parity"
    provider_version: int = 1
    definition_sha256: str = hashlib.sha256(
        b"agent-evolve:recorded-structural-scores-parity:v1;"
        b"purpose=production-policy-parity-only"
    ).hexdigest()

    def project(
        self, request: SlateAllocationRequest
    ) -> tuple[StructuralPosteriorMemberScoreRow, ...]:
        request.revalidate()
        return self.rows


def _initial_state(
    *,
    all_waves: tuple[trap.PreparedWave, ...],
    train_case_ids: tuple[str, ...],
    test_case_id: str,
) -> TargetConditionedAcquisitionState:
    marginal = trap._fit_prior(  # noqa: SLF001
        waves=tuple(
            value for value in all_waves if value.wave.case_id in train_case_ids
        ),
        feature_names=trap.FULL_FEATURES,
        target_kind="marginal",
        alpha=PARAMETERS.alpha,
        meta_precision=PARAMETERS.meta_precision,
    )
    direction = trap._fit_prior(  # noqa: SLF001
        waves=tuple(
            value for value in all_waves if value.wave.case_id in train_case_ids
        ),
        feature_names=trap.FULL_FEATURES,
        target_kind="direction",
        alpha=PARAMETERS.alpha,
        meta_precision=PARAMETERS.meta_precision,
    )
    return TargetConditionedAcquisitionState(
        campaign_scope_sha256=_sha(f"production-parity:{test_case_id}"),
        training_data_sha256=_sha(
            "production-parity-training:" + "|".join(train_case_ids)
        ),
        marginal_head=_head(marginal),
        direction_head=_head(direction),
    )


def verify() -> dict[str, Any]:
    prepared: list[trap.PreparedWave] = []
    raw_by_case_and_ordinal: dict[tuple[str, int], dict[str, Any]] = {}
    spec_by_case = {}
    for spec in trap.gate._case_specs():  # noqa: SLF001
        spec_by_case[spec.case_id] = spec
        for raw in trap.gate._reconstructed_waves(spec):  # noqa: SLF001
            raw_by_case_and_ordinal[(spec.case_id, int(raw["wave_ordinal"]))] = raw
        case_waves, _ = trap._prepare_case(spec)  # noqa: SLF001
        prepared.extend(case_waves)
    all_waves = tuple(prepared)
    case_ids = tuple(sorted(spec_by_case))
    selection_mismatches = []
    diagnostic_mismatches = []
    maximum_diagnostic_absolute_error = 0.0
    selection_match_count = 0
    diagnostic_scalar_count = 0
    case_summary = []
    all_observation_count = 0
    for case_id in case_ids:
        spec = spec_by_case[case_id]
        train_case_ids = tuple(value for value in case_ids if value != case_id)
        state = _initial_state(
            all_waves=all_waves,
            train_case_ids=train_case_ids,
            test_case_id=case_id,
        )
        expected_rows = trap._simulate_case(  # noqa: SLF001
            all_waves=all_waves,
            train_case_ids=train_case_ids,
            test_case_id=case_id,
            policy=trap.POLICIES[0],
            parameters=PARAMETERS,
        )
        expected_by_ordinal = {
            int(value["wave_ordinal"]): value for value in expected_rows
        }
        prepared_case = tuple(
            sorted(
                (value for value in all_waves if value.wave.case_id == case_id),
                key=lambda value: value.wave.wave_ordinal,
            )
        )
        gain = 0.0
        oracle_gain = 0.0
        for generation in (1, 3, 5):
            decisions = []
            observations = []
            for wave in (
                value for value in prepared_case if value.wave.generation == generation
            ):
                projection = feature_parity._projection_request(  # noqa: SLF001
                    spec=spec,
                    raw_wave=raw_by_case_and_ordinal[
                        (case_id, wave.wave.wave_ordinal)
                    ],
                    prepared=wave,
                )
                context = TargetConditionedAllocationContext(
                    finite_contract_sha256=(
                        projection.allocation_request.slate.finite_contract_sha256
                    ),
                    cutoff_receipt_sha256=_sha(
                        f"production-parity:{case_id}:{generation}:cutoff"
                    ),
                    archive_context=projection.archive_context,
                    frontier_target=projection.frontier_target,
                    state=state,
                    transition_receipts=projection.transition_receipts,
                    campaign_generation=generation,
                    lane_slot=wave.wave.parent_slot,
                    remaining_proposal_horizon=(
                        wave.target.remaining_proposal_horizon
                    ),
                )
                realizability: RealizablePortfolioProjector = (
                    _HistoricalRealizabilityProjector(wave.wave.feasible_subsets)
                )
                score_projector: StructuralScoreProjector = (
                    _RecordedStructuralScoreProjector(
                        projection.structural_score_rows
                    )
                )
                decision = TargetConditionedSlateAllocatorAdapter(
                    context_provider=_ContextProvider(context),
                    profile=PROFILE,
                    structural_score_projector=score_projector,
                    realizability_projector=realizability,
                ).select(projection.allocation_request)
                decisions.append(decision)
                selected = tuple(sorted(value.option_id for value in decision.selected))
                expected = expected_by_ordinal[wave.wave.wave_ordinal]
                expected_selected = tuple(expected["selected_option_ids"])
                if selected == expected_selected:
                    selection_match_count += 1
                else:
                    selection_mismatches.append(
                        {
                            "case_id": case_id,
                            "wave_ordinal": wave.wave.wave_ordinal,
                            "expected": list(expected_selected),
                            "observed": list(selected),
                        }
                    )
                gain += wave.wave.gain_by_subset[selected]
                oracle_gain += wave.wave.oracle_gain
                expected_diagnostics = expected["candidate_diagnostics"]
                for score in decision.score_rows:
                    expected_score = expected_diagnostics[score.option_id]
                    for name, observed in (
                        ("predicted_marginal", score.predicted_marginal),
                        ("predicted_direction", score.predicted_direction),
                        ("epistemic_uncertainty", score.epistemic_uncertainty),
                        ("selection_score", score.final_score),
                    ):
                        diagnostic_scalar_count += 1
                        error = abs(observed - float(expected_score[name]))
                        maximum_diagnostic_absolute_error = max(
                            maximum_diagnostic_absolute_error, error
                        )
                        if error > 1e-10 and len(diagnostic_mismatches) < 100:
                            diagnostic_mismatches.append(
                                {
                                    "case_id": case_id,
                                    "wave_ordinal": wave.wave.wave_ordinal,
                                    "option_id": score.option_id,
                                    "field": name,
                                    "expected": expected_score[name],
                                    "observed": observed,
                                    "absolute_error": error,
                                }
                            )
                features = {
                    value.option_id: value for value in decision.request.member_features
                }
                members = {value.option_id: value for value in wave.wave.members}
                for selected_member in decision.selected:
                    option_id = selected_member.option_id
                    feature = features[option_id]
                    observations.append(
                        TargetConditionedSelectedObservation(
                            decision_sha256=decision.decision_sha256,
                            campaign_generation=generation,
                            option_id=option_id,
                            option_identity_sha256=(
                                selected_member.option_identity_sha256
                            ),
                            feature_row_sha256=feature.feature_row_sha256,
                            feature_values=feature.values,
                            normalized_marginal_utility=(
                                members[option_id].normalized_marginal_gain
                            ),
                            normalized_target_improvement=(
                                wave.target_improvement_by_id[option_id]
                            ),
                            evaluator_receipt_sha256=_sha(
                                f"production-parity:{case_id}:"
                                f"{wave.wave.wave_ordinal}:{option_id}:evaluation"
                            ),
                        )
                    )
            update = update_target_conditioned_state(
                state,
                decisions=tuple(decisions),
                observations=tuple(observations),
            )
            all_observation_count += len(observations)
            state = update.next_state
        case_summary.append(
            {
                "case_id": case_id,
                "workload": spec.workload,
                "model": spec.model,
                "gain": gain,
                "oracle_gain": oracle_gain,
                "oracle_fraction": gain / oracle_gain,
                "final_state_sha256": state.state_sha256,
                "selected_observation_count": state.selected_observation_count,
            }
        )
    passed = not selection_mismatches and not diagnostic_mismatches
    return {
        "schema_version": 1,
        "verification_id": "trap_production_policy_parity_v1",
        "passed": passed,
        "profile": PROFILE.to_record(),
        "parameters": PARAMETERS.as_dict(),
        "case_count": len(case_summary),
        "workload_count": len({value["workload"] for value in case_summary}),
        "model_family_count": len({value["model"] for value in case_summary}),
        "wave_count": 36,
        "selection_match_count": selection_match_count,
        "selection_mismatch_count": len(selection_mismatches),
        "diagnostic_scalar_count": diagnostic_scalar_count,
        "maximum_diagnostic_absolute_error": maximum_diagnostic_absolute_error,
        "diagnostic_mismatch_count_over_1e_10": len(diagnostic_mismatches),
        "selected_observation_count": all_observation_count,
        "mean_case_oracle_fraction": fmean(
            value["oracle_fraction"] for value in case_summary
        ),
        "case_summary": case_summary,
        "selection_mismatches": selection_mismatches,
        "first_diagnostic_mismatches": diagnostic_mismatches,
        "provenance": {
            "development_analysis_sha256": _sha256(Path(trap.__file__).resolve()),
            "feature_parity_script_sha256": _sha256(
                Path(feature_parity.__file__).resolve()
            ),
            "source_panel_sha256": _sha256(trap.SOURCE_PANEL_PATH),
        },
        "claim_boundary": (
            "production-math parity on retrospective fixed slates; not "
            "prospective campaign efficacy or SOTA evidence"
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
                    "selection_match_count",
                    "selection_mismatch_count",
                    "diagnostic_scalar_count",
                    "maximum_diagnostic_absolute_error",
                    "diagnostic_mismatch_count_over_1e_10",
                    "mean_case_oracle_fraction",
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

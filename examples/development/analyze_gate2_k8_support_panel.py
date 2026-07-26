#!/usr/bin/env python3
"""Analyze the preregistered six-run Gate 2 complete-K8 support panel.

The script reconstructs each authenticated finite K8 slate from its immutable
source campaign, joins all real evaluator outcomes, verifies the published
current/direct/oracle utilities, and evaluates small workload-neutral ridge
policies under leave-run, leave-model, and leave-workload-out protocols.

Oracle outcomes are targets and evaluation endpoints only.  No test-fold
outcome enters a feature, fit, standardizer, uncertainty estimate, or
hyperparameter decision.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations, product
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Callable, Iterable, Sequence

import numpy as np


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development import evaluate_boils_generic_campaign_k8 as boils  # noqa: E402
from examples.development import evaluate_heat2d_structural_posterior_k8 as heat  # noqa: E402
from examples.development import (  # noqa: E402
    evaluate_timeloop_structural_posterior_k8_regret as timeloop,
)
from examples.development.durable_run_artifacts import write_json_atomic  # noqa: E402
from agent_evolve.campaign_variation_topology import (  # noqa: E402
    CampaignVariationTopology,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts"
)
LOG_ROOT = ARTIFACT_ROOT / "experiment_logs"
RAW_BOILS_HV_SCALE = 80.0 * 12000.0
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PAIR_ALPHA_GRID = (0.1, 10.0)
PAIR_WEIGHT_GRID = (0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    workload: str
    model: str
    source_run: Path
    preparation: Path | None
    analysis: Path


@dataclass(frozen=True, slots=True)
class Member:
    case_id: str
    workload: str
    model: str
    wave_ordinal: int
    option_id: str
    model_rank: int
    family: str
    locus_key: str
    transition_path: str
    features: dict[str, float]
    marginal_gain: float
    normalized_marginal_gain: float
    current_selected: bool
    direct_selected: bool
    oracle_selected: bool


@dataclass(frozen=True, slots=True)
class Wave:
    case_id: str
    workload: str
    model: str
    wave_ordinal: int
    generation: int
    parent_slot: int
    members: tuple[Member, ...]
    feasible_subsets: tuple[tuple[str, ...], ...]
    current_option_ids: tuple[str, ...]
    direct_option_ids: tuple[str, ...]
    oracle_option_ids: tuple[str, ...]
    current_gain: float
    direct_gain: float
    oracle_gain: float
    uniform_expected_gain: float
    published_literal_direct_gain: float
    published_oracle_gain: float
    published_uniform_expected_gain: float
    unconstrained_oracle_gain: float
    unconstrained_uniform_expected_gain: float
    gain_by_subset: dict[tuple[str, ...], float]


@dataclass(frozen=True, slots=True)
class RidgeModel:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    covariance: np.ndarray
    residual_variance: float

    def _row(self, features: dict[str, float]) -> np.ndarray:
        raw = np.asarray(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )
        return (raw - self.means) / self.scales

    def predict(self, features: dict[str, float]) -> float:
        row = self._row(features)
        return float(row @ self.coefficients)

    def uncertainty(self, features: dict[str, float]) -> float:
        row = self._row(features)
        leverage = max(0.0, float(row @ self.covariance @ row))
        return math.sqrt(leverage * max(self.residual_variance, 1e-12))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise RuntimeError(f"{path} is not an exact JSON object")
    return value


def _case_specs() -> tuple[CaseSpec, ...]:
    gate = LOG_ROOT / "gate2_support_completion"
    return (
        CaseSpec(
            case_id="boils_mistral_s20260770",
            workload="boils",
            model="mistral",
            source_run=(
                LOG_ROOT
                / "boils_abc/generic_campaign/"
                "grid_boils_abc_mistral_s20260770_v10r1_live"
            ),
            preparation=None,
            analysis=gate / "boils_mistral_s20260770_v1_live/allocation_analysis.json",
        ),
        CaseSpec(
            case_id="boils_qwen_s20260770",
            workload="boils",
            model="qwen",
            source_run=(
                LOG_ROOT
                / "boils_abc/generic_campaign/"
                "grid_boils_abc_qwen_s20260770_v11r1_live"
            ),
            preparation=None,
            analysis=(
                LOG_ROOT
                / "boils_abc/counterfactual/"
                "grid_boils_abc_qwen_s20260770_v11_k8_completion_r1/analysis.json"
            ),
        ),
        CaseSpec(
            case_id="heat2d_mistral_s20260770",
            workload="heat2d",
            model="mistral",
            source_run=(
                LOG_ROOT
                / "benchmark_q1/engibench_heat2d/generic_campaign/"
                "grid_heat2d_mistral_s20260770_v12r1_live"
            ),
            preparation=gate / "heat2d_mistral_s20260770_v1_prepare",
            analysis=gate / "heat2d_mistral_s20260770_v1_live/allocation_analysis.json",
        ),
        CaseSpec(
            case_id="heat2d_gpt_sol_s20260770",
            workload="heat2d",
            model="gpt_sol",
            source_run=(
                LOG_ROOT
                / "benchmark_q1/engibench_heat2d/generic_campaign/"
                "grid_heat2d_gpt_sol_s20260770_v18r1_live"
            ),
            preparation=gate / "heat2d_gpt_sol_s20260770_v1_prepare",
            analysis=gate / "heat2d_gpt_sol_s20260770_v1_live/allocation_analysis.json",
        ),
        CaseSpec(
            case_id="timeloop_deepseek_s20260770",
            workload="timeloop",
            model="deepseek",
            source_run=(
                LOG_ROOT
                / "benchmark_q1/timeloop_codesign/full_support_g6/"
                "grid_timeloop_v2_deepseek_s20260770_v18r1_live"
            ),
            preparation=gate / "timeloop_deepseek_s20260770_v1_prepare",
            analysis=(
                gate / "timeloop_deepseek_s20260770_v1_live/allocation_analysis.json"
            ),
        ),
        CaseSpec(
            case_id="timeloop_gpt_sol_s20260770",
            workload="timeloop",
            model="gpt_sol",
            source_run=(
                LOG_ROOT
                / "benchmark_q1/timeloop_codesign/full_support_g6/"
                "grid_timeloop_v2_gpt_sol_s20260770_v19r1_live"
            ),
            preparation=gate / "timeloop_gpt_sol_s20260770_v1_prepare",
            analysis=(
                gate / "timeloop_gpt_sol_s20260770_v1_live/allocation_analysis.json"
            ),
        ),
    )


def _float_hex(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise RuntimeError(f"{name} is not a hexadecimal float")
    return float.fromhex(value)


def _flatten(value: object, path: str = "$") -> dict[str, object]:
    if type(value) is dict:
        result: dict[str, object] = {}
        for key in sorted(value):
            result.update(_flatten(value[key], f"{path}.{key}"))
        return result
    if type(value) is list:
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{path}[{index}]"))
        return result
    return {path: value}


def _transition_features(
    parent: dict[str, Any], child: dict[str, Any]
) -> tuple[dict[str, float], str]:
    parent_leaves = _flatten(parent)
    child_leaves = _flatten(child)
    paths = sorted(set(parent_leaves) | set(child_leaves))
    changed = [
        path for path in paths if parent_leaves.get(path) != child_leaves.get(path)
    ]
    numeric_deltas: list[float] = []
    numeric_relative: list[float] = []
    numeric_signs: list[float] = []
    categorical = 0
    for path in changed:
        old = parent_leaves.get(path)
        new = child_leaves.get(path)
        if (
            type(old) in {int, float}
            and type(new) in {int, float}
            and type(old) is not bool
            and type(new) is not bool
        ):
            delta = float(new) - float(old)
            numeric_deltas.append(abs(delta))
            numeric_relative.append(abs(delta) / max(abs(float(old)), 1e-12))
            numeric_signs.append(0.0 if delta == 0.0 else math.copysign(1.0, delta))
        else:
            categorical += 1
    count = max(1, len(changed))
    path_depths = [path.count(".") + path.count("[") for path in changed]
    return (
        {
            "transition_change_count": float(len(changed)),
            "transition_numeric_fraction": len(numeric_deltas) / count,
            "transition_categorical_fraction": categorical / count,
            "transition_abs_numeric_delta": (
                0.0 if not numeric_deltas else fmean(numeric_deltas)
            ),
            "transition_relative_numeric_delta_log": (
                0.0
                if not numeric_relative
                else fmean(math.log1p(value) for value in numeric_relative)
            ),
            "transition_numeric_sign": (
                0.0 if not numeric_signs else fmean(numeric_signs)
            ),
            "transition_path_depth": 0.0 if not path_depths else fmean(path_depths),
        },
        "|".join(changed),
    )


def _parent_context_features(
    *, workload: str, raw_wave: dict[str, Any], base_hypervolume: float
) -> dict[str, float]:
    parent = raw_wave["parent_objectives"]
    desirability: list[float] = []
    if workload == "boils":
        reference = {
            metric_id: float.fromhex(encoded)
            for metric_id, encoded in raw_wave["archive_reward_snapshot"][
                "reference_point"
            ]
        }
        for metric_id, value in parent.items():
            desirability.append(1.0 - (float(value) / reference[metric_id]))
        archive_size = len(raw_wave["archive_reward_snapshot"]["archive_points"])
    else:
        for axis in raw_wave["archive_snapshot"]["spec"]["axes"]:
            value = float(parent[axis["metric_id"]])
            ideal = float.fromhex(axis["ideal_hex"])
            reference = float.fromhex(axis["reference_hex"])
            span = reference - ideal
            oriented = (
                (reference - value) / span
                if axis["goal"] == "min"
                else (value - reference) / (ideal - reference)
            )
            desirability.append(oriented)
        archive_size = len(raw_wave["archive_snapshot"]["raw_archive_points"])
    return {
        "parent_desirability_mean": fmean(desirability),
        "parent_desirability_min": min(desirability),
        "parent_desirability_max": max(desirability),
        "archive_base_hypervolume": base_hypervolume,
        "archive_point_count_log": math.log1p(archive_size),
    }


def _outcomes_by_wave(analysis: dict[str, Any]) -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, float]]] = {}
    for wave in analysis["waves"]:
        rows = wave.get("all_k8_outcomes", wave.get("members"))
        if type(rows) is not list or len(rows) != 8:
            raise RuntimeError("analysis wave does not contain eight outcomes")
        result[int(wave["wave_ordinal"])] = {
            row["option_id"]: {
                metric: float(value) for metric, value in row["outcomes"].items()
            }
            for row in rows
        }
    return result


def _analysis_selection(
    workload: str, wave: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], float, float, float, float]:
    if workload == "boils":
        current = wave["calibrated_k4"]
        direct = wave["direct_model_top4"]
        oracle = wave["oracle_k4"]
        uniform = wave["uniform_k4"]["expected_gain"] / RAW_BOILS_HV_SCALE
        scale = RAW_BOILS_HV_SCALE
    else:
        current = wave["structural_posterior_k4"]
        direct = wave["direct_model_top4"]
        oracle = wave["oracle_k4"]
        uniform = wave["uniform_k4"]["expected_gain"]
        scale = 1.0
    return (
        tuple(current["option_ids"]),
        tuple(direct["option_ids"]),
        tuple(oracle["option_ids"]),
        float(current["gain"]) / scale,
        float(direct["gain"]) / scale,
        float(oracle["gain"]) / scale,
        float(uniform),
    )


def _executable_subsets(
    workload: str,
    raw_wave: dict[str, Any],
    ordered_raw: Sequence[dict[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    if workload == "boils":
        return tuple(
            sorted(
                {
                    tuple(sorted(value))
                    for value in raw_wave["feasible_k4_option_id_sets"]
                }
            )
        )
    request = raw_wave["slate_allocation_request"]
    request_members = {
        value["option_id"]: value for value in request["slate"]["members"]
    }
    allowed_values = request.get("pairwise_disjoint_option_id_pairs")
    allowed_pairs = (
        None
        if allowed_values is None
        else {frozenset(value) for value in allowed_values}
    )
    min_families = request.get("min_distinct_families")
    option_ids = tuple(value["option_id"] for value in ordered_raw)
    feasible = []
    for subset in combinations(option_ids, 4):
        if allowed_pairs is not None and any(
            frozenset(pair) not in allowed_pairs for pair in combinations(subset, 2)
        ):
            continue
        if min_families is not None and len(
            {request_members[option_id]["family"] for option_id in subset}
        ) < int(min_families):
            continue
        feasible.append(tuple(sorted(subset)))
    if not feasible:
        raise RuntimeError("sealed K8 request has no executable performance K4")
    return tuple(sorted(set(feasible)))


def _reconstructed_waves(spec: CaseSpec) -> list[dict[str, Any]]:
    if spec.workload == "boils":
        manifest = _read_json(spec.source_run / "manifest.json")
        workload = manifest.get("workload")
        if type(workload) is not dict:
            raise RuntimeError("BOiLS source manifest omitted its workload")
        if workload.get("variation_topology") is None:
            if workload.get("finite_catalog_id") != "boils_abc_single_action":
                raise RuntimeError(
                    "legacy BOiLS source lacks an auditable atomic catalog"
                )
            variation_topology = CampaignVariationTopology()
        else:
            variation_topology = boils._variation_topology(  # noqa: SLF001
                spec.source_run
            )
        _, waves, _ = boils._build_plan(  # noqa: SLF001
            events=boils._campaign_events(spec.source_run),  # noqa: SLF001
            variation_topology=variation_topology,
        )
        return waves
    if spec.workload == "heat2d":
        _, waves, _ = heat._build_precommit(  # noqa: SLF001
            heat._campaign_events(spec.source_run)  # noqa: SLF001
        )
        return waves
    _, waves, _ = timeloop._build_precommit(  # noqa: SLF001
        source_run=spec.source_run,
        events=timeloop.support._campaign_events(spec.source_run),  # noqa: SLF001
    )
    return waves


def _gain_function(
    workload: str,
    raw_wave: dict[str, Any],
    outcomes: dict[str, dict[str, float]],
) -> Callable[[Sequence[str]], float]:
    if workload == "boils":
        snapshot = boils._snapshot(raw_wave["archive_reward_snapshot"])  # noqa: SLF001
        scale = RAW_BOILS_HV_SCALE
    elif workload == "heat2d":
        snapshot = heat._affine_snapshot(raw_wave["archive_snapshot"])  # noqa: SLF001
        scale = 1.0
    else:
        snapshot = timeloop.support._affine_snapshot(raw_wave["archive_snapshot"])
        scale = 1.0

    def gain(option_ids: Sequence[str]) -> float:
        augmented = snapshot.augmented_hypervolume(
            tuple(outcomes[option_id] for option_id in option_ids)
        )
        return max(0.0, float(augmented - snapshot.base_hypervolume)) / scale

    return gain


def _request_member_by_id(
    workload: str, raw_wave: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if workload == "boils":
        return {value["option_id"]: value for value in raw_wave["members"]}
    return {
        value["option_id"]: value
        for value in raw_wave["slate_allocation_request"]["slate"]["members"]
    }


def _score_row_by_id(
    workload: str, raw_wave: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if workload == "boils":
        return {
            value["option_id"]: value["structural_posterior_scores"]
            for value in raw_wave["members"]
        }
    return {
        value["option_id"]: value
        for value in raw_wave["structural_posterior_decision"]["score_rows"]
    }


def _member_features(
    *,
    spec: CaseSpec,
    raw_wave: dict[str, Any],
    raw_member: dict[str, Any],
    request_member: dict[str, Any],
    score_row: dict[str, Any],
    parent_features: dict[str, float],
    family_frequency: float,
) -> tuple[dict[str, float], str]:
    features: dict[str, float] = {"bias": 1.0, **parent_features}
    rank = int(raw_member["model_rank"])
    for value in range(1, 9):
        features[f"rank_{value}"] = float(rank == value)
    generation = int(raw_wave["generation"])
    for value in (1, 3, 5):
        features[f"generation_{value}"] = float(generation == value)
    parent_slot = int(raw_wave.get("parent_slot", raw_member.get("parent_slot", 0)))
    features["parent_slot_0"] = float(parent_slot == 0)
    features["parent_slot_1"] = float(parent_slot == 1)
    for value in ("exploit", "falsify", "coverage"):
        features[f"role_proposal_{value}"] = float(
            request_member.get("role_proposal") == value
        )
    for model in ("mistral", "qwen", "deepseek", "gpt_sol"):
        features[f"model_{model}"] = float(spec.model == model)
    features["family_frequency"] = family_frequency
    features["family_rarity"] = 1.0 / max(1.0, 8.0 * family_frequency)
    features["supporting_card_count"] = float(
        len(request_member.get("supporting_card_keys", ()))
    )
    structural = request_member["structural_evidence"]
    features["archive_novelty"] = _float_hex(
        structural["archive_novelty_score_hex"], name="archive novelty"
    )
    features["structural_coverage_raw"] = _float_hex(
        structural["structural_coverage_score_hex"], name="structural coverage"
    )
    for key in (
        "calibrated_exploitation_score",
        "calibrated_frontier_score",
        "raw_epistemic_score",
        "structural_coverage_score",
        "epistemic_structural_score",
    ):
        features[key] = _float_hex(score_row[f"{key}_hex"], name=key)
    metric_rows = score_row["metric_scores"]
    count = float(len(metric_rows))
    for confidence in ("low", "medium", "high"):
        features[f"confidence_{confidence}_fraction"] = (
            sum(value["confidence"] == confidence for value in metric_rows) / count
        )
    features["favorable_fraction"] = (
        sum(value["favorable_assertion"] for value in metric_rows) / count
    )
    features["adverse_fraction"] = (
        sum(value["adverse_assertion"] for value in metric_rows) / count
    )
    features["abstention_fraction"] = (
        sum(value["explicit_abstention"] for value in metric_rows) / count
    )
    features["posterior_correctness_mean"] = fmean(
        _float_hex(
            value["calibration_cell"]["posterior_correctness_hex"],
            name="posterior correctness",
        )
        for value in metric_rows
    )
    features["calibration_observation_count_log"] = math.log1p(
        sum(value["calibration_cell"]["observation_count"] for value in metric_rows)
    )
    transition, transition_path = _transition_features(
        raw_wave["parent_configuration"], raw_member["configuration"]
    )
    features.update(transition)
    return features, transition_path


def _load_case(
    spec: CaseSpec,
    *,
    reconstructed_waves: Sequence[dict[str, Any]] | None = None,
) -> tuple[Wave, ...]:
    analysis = _read_json(spec.analysis)
    analysis_waves = {
        int(value["wave_ordinal"]): value for value in analysis["waves"]
    }
    outcomes_by_wave = _outcomes_by_wave(analysis)
    raw_waves = list(
        _reconstructed_waves(spec)
        if reconstructed_waves is None
        else reconstructed_waves
    )
    if len(raw_waves) != 6:
        raise RuntimeError(f"{spec.case_id} did not reconstruct six waves")
    result: list[Wave] = []
    for raw_wave in raw_waves:
        ordinal = int(raw_wave["wave_ordinal"])
        analysis_wave = analysis_waves[ordinal]
        outcomes = outcomes_by_wave[ordinal]
        (
            current_ids,
            direct_ids,
            oracle_ids,
            published_current,
            published_direct,
            published_oracle,
            published_uniform,
        ) = _analysis_selection(spec.workload, analysis_wave)
        gain = _gain_function(spec.workload, raw_wave, outcomes)
        ordered_raw = sorted(raw_wave["members"], key=lambda value: value["model_rank"])
        ordered_ids = tuple(value["option_id"] for value in ordered_raw)
        all_subsets = tuple(
            tuple(sorted(value)) for value in combinations(ordered_ids, 4)
        )
        feasible = _executable_subsets(spec.workload, raw_wave, ordered_raw)
        gains = {subset: gain(subset) for subset in feasible}
        best_subset = min(feasible, key=lambda value: (-gains[value], value))
        all_gains = {subset: gain(subset) for subset in all_subsets}
        unconstrained_best = min(
            all_subsets, key=lambda value: (-all_gains[value], value)
        )
        current_key = tuple(sorted(current_ids))
        published_direct_key = tuple(sorted(direct_ids))
        published_oracle_key = tuple(sorted(oracle_ids))
        if current_key not in gains:
            raise RuntimeError("recorded structural K4 is not executable")
        rank_by_id = {
            value["option_id"]: int(value["model_rank"]) for value in ordered_raw
        }
        direct_key = min(
            feasible,
            key=lambda subset: (
                tuple(sorted(rank_by_id[option_id] for option_id in subset)),
                subset,
            ),
        )
        observed_published = (
            gain(current_key),
            gain(published_direct_key),
            gain(published_oracle_key),
        )
        expected_published = (published_current, published_direct, published_oracle)
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(
                observed_published, expected_published, strict=True
            )
        ):
            raise RuntimeError(
                f"{spec.case_id} wave {ordinal} utility reconstruction drift: "
                f"observed={observed_published}, expected={expected_published}"
            )
        request_by_id = _request_member_by_id(spec.workload, raw_wave)
        score_by_id = _score_row_by_id(spec.workload, raw_wave)
        family_counts: dict[str, int] = {}
        for value in ordered_raw:
            family_counts[value["family"]] = family_counts.get(value["family"], 0) + 1
        base_hv = float(analysis_wave["base_hypervolume"])
        if spec.workload == "boils":
            base_hv /= RAW_BOILS_HV_SCALE
        parent_features = _parent_context_features(
            workload=spec.workload,
            raw_wave=raw_wave,
            base_hypervolume=base_hv,
        )
        members: list[Member] = []
        for raw_member in ordered_raw:
            option_id = raw_member["option_id"]
            features, transition_path = _member_features(
                spec=spec,
                raw_wave=raw_wave,
                raw_member=raw_member,
                request_member=request_by_id[option_id],
                score_row=score_by_id[option_id],
                parent_features=parent_features,
                family_frequency=family_counts[raw_member["family"]] / 8.0,
            )
            marginal = gain((option_id,))
            members.append(
                Member(
                    case_id=spec.case_id,
                    workload=spec.workload,
                    model=spec.model,
                    wave_ordinal=ordinal,
                    option_id=option_id,
                    model_rank=int(raw_member["model_rank"]),
                    family=raw_member["family"],
                    locus_key=request_by_id[option_id]["locus_key"],
                    transition_path=transition_path,
                    features=features,
                    marginal_gain=marginal,
                    normalized_marginal_gain=(
                        0.0 if gains[best_subset] == 0.0 else marginal / gains[best_subset]
                    ),
                    current_selected=option_id in current_ids,
                    direct_selected=option_id in direct_key,
                    oracle_selected=option_id in best_subset,
                )
            )
        result.append(
            Wave(
                case_id=spec.case_id,
                workload=spec.workload,
                model=spec.model,
                wave_ordinal=ordinal,
                generation=int(raw_wave["generation"]),
                parent_slot=int(
                    raw_wave.get("parent_slot", ordered_raw[0].get("parent_slot", 0))
                ),
                members=tuple(members),
                feasible_subsets=feasible,
                current_option_ids=current_key,
                direct_option_ids=direct_key,
                oracle_option_ids=best_subset,
                current_gain=gains[current_key],
                direct_gain=gains[direct_key],
                oracle_gain=gains[best_subset],
                uniform_expected_gain=fmean(gains.values()),
                published_literal_direct_gain=published_direct,
                published_oracle_gain=published_oracle,
                published_uniform_expected_gain=published_uniform,
                unconstrained_oracle_gain=all_gains[unconstrained_best],
                unconstrained_uniform_expected_gain=fmean(all_gains.values()),
                gain_by_subset=gains,
            )
        )
    return tuple(result)


RANK_FEATURES = tuple(["bias", *[f"rank_{value}" for value in range(1, 9)]])
MODEL_FEATURES = tuple(
    f"model_{value}" for value in ("mistral", "qwen", "deepseek", "gpt_sol")
)
PORTABLE_FEATURES = (
    *RANK_FEATURES,
    "generation_1",
    "generation_3",
    "generation_5",
    "parent_slot_0",
    "parent_slot_1",
    "role_proposal_exploit",
    "role_proposal_falsify",
    "role_proposal_coverage",
    "family_frequency",
    "family_rarity",
    "supporting_card_count",
    "archive_novelty",
    "structural_coverage_raw",
    "calibrated_exploitation_score",
    "calibrated_frontier_score",
    "raw_epistemic_score",
    "structural_coverage_score",
    "epistemic_structural_score",
    "confidence_low_fraction",
    "confidence_medium_fraction",
    "confidence_high_fraction",
    "favorable_fraction",
    "adverse_fraction",
    "abstention_fraction",
    "posterior_correctness_mean",
    "calibration_observation_count_log",
    "transition_change_count",
    "transition_numeric_fraction",
    "transition_categorical_fraction",
    "transition_relative_numeric_delta_log",
    "transition_numeric_sign",
    "transition_path_depth",
    "parent_desirability_mean",
    "parent_desirability_min",
    "parent_desirability_max",
    "archive_base_hypervolume",
    "archive_point_count_log",
)
MODEL_AWARE_FEATURES = (*PORTABLE_FEATURES, *MODEL_FEATURES)
PAIR_FEATURES = (
    "bias",
    "same_family",
    "same_locus",
    "same_transition_path",
    "rank_distance",
    "rank_sum",
    "exploitation_distance",
    "frontier_distance",
    "epistemic_distance",
    "coverage_distance",
    "prediction_profile_distance",
)


def _fit_ridge(
    rows: Sequence[dict[str, float]],
    targets: Sequence[float],
    feature_names: tuple[str, ...],
    alpha: float,
) -> RidgeModel:
    if not rows or len(rows) != len(targets):
        raise ValueError("ridge fit requires aligned non-empty rows and targets")
    raw = np.asarray(
        [[float(row.get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=float,
    )
    target = np.asarray(targets, dtype=float)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    bias_index = feature_names.index("bias")
    means[bias_index] = 0.0
    scales[bias_index] = 1.0
    scales[scales < 1e-12] = 1.0
    design = (raw - means) / scales
    penalty = np.eye(len(feature_names), dtype=float) * float(alpha)
    penalty[bias_index, bias_index] = 0.0
    normal = design.T @ design + penalty
    covariance = np.linalg.pinv(normal)
    coefficients = covariance @ design.T @ target
    residual = target - design @ coefficients
    variance = float(np.mean(np.square(residual)))
    return RidgeModel(
        feature_names=feature_names,
        means=means,
        scales=scales,
        coefficients=coefficients,
        covariance=covariance,
        residual_variance=variance,
    )


def _pair_features(left: Member, right: Member) -> dict[str, float]:
    profile_keys = (
        "favorable_fraction",
        "adverse_fraction",
        "confidence_low_fraction",
        "confidence_medium_fraction",
        "confidence_high_fraction",
    )
    return {
        "bias": 1.0,
        "same_family": float(left.family == right.family),
        "same_locus": float(left.locus_key == right.locus_key),
        "same_transition_path": float(
            bool(left.transition_path)
            and left.transition_path == right.transition_path
        ),
        "rank_distance": abs(left.model_rank - right.model_rank) / 7.0,
        "rank_sum": (left.model_rank + right.model_rank) / 16.0,
        "exploitation_distance": abs(
            left.features["calibrated_exploitation_score"]
            - right.features["calibrated_exploitation_score"]
        ),
        "frontier_distance": abs(
            left.features["calibrated_frontier_score"]
            - right.features["calibrated_frontier_score"]
        ),
        "epistemic_distance": abs(
            left.features["raw_epistemic_score"]
            - right.features["raw_epistemic_score"]
        ),
        "coverage_distance": abs(
            left.features["structural_coverage_score"]
            - right.features["structural_coverage_score"]
        ),
        "prediction_profile_distance": fmean(
            abs(left.features[key] - right.features[key]) for key in profile_keys
        ),
    }


def _members(waves: Iterable[Wave]) -> list[Member]:
    return [member for wave in waves for member in wave.members]


def _pair_training_rows(
    waves: Iterable[Wave],
) -> tuple[list[dict[str, float]], list[float]]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    for wave in waves:
        by_id = {value.option_id: value for value in wave.members}
        for left_id, right_id in combinations(tuple(by_id), 2):
            left = by_id[left_id]
            right = by_id[right_id]
            pair_gain = _gain_for_ids(wave, (left_id, right_id))
            denominator = wave.oracle_gain
            residual = (
                0.0
                if denominator == 0.0
                else (
                    pair_gain - left.marginal_gain - right.marginal_gain
                )
                / denominator
            )
            rows.append(_pair_features(left, right))
            targets.append(residual)
    return rows, targets


def _gain_for_ids(wave: Wave, option_ids: Sequence[str]) -> float:
    key = tuple(sorted(option_ids))
    if len(key) == 4:
        return wave.gain_by_subset[key]
    # Inclusion-exclusion is not sufficient for arbitrary pairs.  Pair gains
    # are materialized as singleton-union values below when loading a cache.
    member_by_id = {value.option_id: value for value in wave.members}
    if len(key) == 1:
        return member_by_id[key[0]].marginal_gain
    pair_key = ("__pair__", *key)
    value = wave.gain_by_subset.get(pair_key)  # type: ignore[arg-type]
    if value is None:
        raise RuntimeError("pair gain cache is missing")
    return value


def _with_pair_gains(waves: Sequence[Wave], raw_by_case: dict[str, list[dict[str, Any]]]) -> tuple[Wave, ...]:
    result: list[Wave] = []
    for wave in waves:
        raw_wave = raw_by_case[wave.case_id][wave.wave_ordinal - 1]
        analysis = None
        spec = next(value for value in _case_specs() if value.case_id == wave.case_id)
        analysis_record = _read_json(spec.analysis)
        outcomes = _outcomes_by_wave(analysis_record)[wave.wave_ordinal]
        gain = _gain_function(wave.workload, raw_wave, outcomes)
        cached = dict(wave.gain_by_subset)
        for left, right in combinations(tuple(value.option_id for value in wave.members), 2):
            cached[("__pair__", *tuple(sorted((left, right))))] = gain((left, right))  # type: ignore[index]
        result.append(
            Wave(
                case_id=wave.case_id,
                workload=wave.workload,
                model=wave.model,
                wave_ordinal=wave.wave_ordinal,
                generation=wave.generation,
                parent_slot=wave.parent_slot,
                members=wave.members,
                feasible_subsets=wave.feasible_subsets,
                current_option_ids=wave.current_option_ids,
                direct_option_ids=wave.direct_option_ids,
                oracle_option_ids=wave.oracle_option_ids,
                current_gain=wave.current_gain,
                direct_gain=wave.direct_gain,
                oracle_gain=wave.oracle_gain,
                uniform_expected_gain=wave.uniform_expected_gain,
                published_literal_direct_gain=wave.published_literal_direct_gain,
                published_oracle_gain=wave.published_oracle_gain,
                published_uniform_expected_gain=(
                    wave.published_uniform_expected_gain
                ),
                unconstrained_oracle_gain=wave.unconstrained_oracle_gain,
                unconstrained_uniform_expected_gain=(
                    wave.unconstrained_uniform_expected_gain
                ),
                gain_by_subset=cached,
            )
        )
    return tuple(result)


def _fit_models(
    train_waves: Sequence[Wave],
    *,
    feature_names: tuple[str, ...],
    alpha: float,
    pair_alpha: float | None,
) -> tuple[RidgeModel, RidgeModel | None]:
    train_members = _members(train_waves)
    marginal = _fit_ridge(
        [value.features for value in train_members],
        [value.normalized_marginal_gain for value in train_members],
        feature_names,
        alpha,
    )
    if pair_alpha is None:
        return marginal, None
    rows, targets = _pair_training_rows(train_waves)
    return marginal, _fit_ridge(rows, targets, PAIR_FEATURES, pair_alpha)


def _select_subset(
    wave: Wave,
    marginal_model: RidgeModel,
    *,
    pair_model: RidgeModel | None,
    pair_weight: float,
    protected_exploration: bool,
) -> tuple[str, ...]:
    by_id = {value.option_id: value for value in wave.members}
    predicted = {
        option_id: marginal_model.predict(member.features)
        for option_id, member in by_id.items()
    }
    uncertainty = {
        option_id: marginal_model.uncertainty(member.features)
        for option_id, member in by_id.items()
    }
    protected: set[str] = set()
    if protected_exploration:
        protected = set(
            sorted(
                by_id,
                key=lambda option_id: (
                    -uncertainty[option_id],
                    by_id[option_id].model_rank,
                    option_id,
                ),
            )[:2]
        )

    def score(subset: tuple[str, ...]) -> float:
        value = sum(predicted[option_id] for option_id in subset)
        if pair_model is not None:
            for left_id, right_id in combinations(subset, 2):
                residual = pair_model.predict(
                    _pair_features(by_id[left_id], by_id[right_id])
                )
                value += pair_weight * min(0.0, residual)
        return value

    feasible = [
        subset
        for subset in wave.feasible_subsets
        if not protected or any(option_id in protected for option_id in subset)
    ]
    if not feasible:
        raise RuntimeError("protected exploration removed every feasible K4 subset")
    return min(feasible, key=lambda subset: (-score(subset), subset))


def _evaluate_policy(
    train_waves: Sequence[Wave],
    test_waves: Sequence[Wave],
    *,
    feature_names: tuple[str, ...],
    alpha: float,
    pair_alpha: float | None = None,
    pair_weight: float = 0.0,
    protected_exploration: bool = False,
) -> tuple[dict[str, Any], ...]:
    marginal, pair = _fit_models(
        train_waves,
        feature_names=feature_names,
        alpha=alpha,
        pair_alpha=pair_alpha,
    )
    rows: list[dict[str, Any]] = []
    for wave in test_waves:
        selected = _select_subset(
            wave,
            marginal,
            pair_model=pair,
            pair_weight=pair_weight,
            protected_exploration=protected_exploration,
        )
        rows.append(
            {
                "case_id": wave.case_id,
                "workload": wave.workload,
                "model": wave.model,
                "wave_ordinal": wave.wave_ordinal,
                "selected_option_ids": list(selected),
                "gain": wave.gain_by_subset[selected],
                "oracle_gain": wave.oracle_gain,
                "current_gain": wave.current_gain,
                "direct_gain": wave.direct_gain,
            }
        )
    return tuple(rows)


def _group_score(rows: Sequence[dict[str, Any]]) -> float:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    values = []
    for case_rows in by_case.values():
        oracle = sum(value["oracle_gain"] for value in case_rows)
        values.append(
            0.0 if oracle == 0.0 else sum(value["gain"] for value in case_rows) / oracle
        )
    return fmean(values)


def _inner_tune(
    train_waves: Sequence[Wave],
    *,
    feature_names: tuple[str, ...],
    protected_exploration: bool,
    joint: bool,
) -> dict[str, float | None]:
    case_ids = sorted({value.case_id for value in train_waves})
    if len(case_ids) < 2:
        raise RuntimeError("inner tuning requires at least two training cases")
    parameters = (
        tuple(
            (alpha, pair_alpha, pair_weight)
            for alpha, pair_alpha, pair_weight in product(
                ALPHA_GRID, PAIR_ALPHA_GRID, PAIR_WEIGHT_GRID
            )
        )
        if joint
        else tuple((alpha, None, 0.0) for alpha in ALPHA_GRID)
    )
    scored: list[tuple[float, float, float, float]] = []
    for alpha, pair_alpha, pair_weight in parameters:
        validation_rows: list[dict[str, Any]] = []
        for validation_case in case_ids:
            inner_train = [
                value for value in train_waves if value.case_id != validation_case
            ]
            inner_test = [
                value for value in train_waves if value.case_id == validation_case
            ]
            validation_rows.extend(
                _evaluate_policy(
                    inner_train,
                    inner_test,
                    feature_names=feature_names,
                    alpha=alpha,
                    pair_alpha=pair_alpha,
                    pair_weight=pair_weight,
                    protected_exploration=protected_exploration,
                )
            )
        scored.append(
            (
                _group_score(validation_rows),
                alpha,
                -1.0 if pair_alpha is None else pair_alpha,
                pair_weight,
            )
        )
    _, alpha, encoded_pair_alpha, pair_weight = max(
        scored,
        key=lambda value: (value[0], value[1], value[2], -value[3]),
    )
    return {
        "alpha": alpha,
        "pair_alpha": None if encoded_pair_alpha < 0.0 else encoded_pair_alpha,
        "pair_weight": pair_weight,
    }


POLICIES = {
    "rank_ridge": {
        "feature_names": RANK_FEATURES,
        "protected_exploration": False,
        "joint": False,
    },
    "portable_magnitude_ridge": {
        "feature_names": PORTABLE_FEATURES,
        "protected_exploration": False,
        "joint": False,
    },
    "portable_magnitude_with_explorer": {
        "feature_names": PORTABLE_FEATURES,
        "protected_exploration": True,
        "joint": False,
    },
    "portable_joint_redundancy": {
        "feature_names": PORTABLE_FEATURES,
        "protected_exploration": False,
        "joint": True,
    },
    "model_aware_magnitude_ridge": {
        "feature_names": MODEL_AWARE_FEATURES,
        "protected_exploration": False,
        "joint": False,
    },
}


def _folds(waves: Sequence[Wave], protocol: str) -> tuple[tuple[str, tuple[Wave, ...], tuple[Wave, ...]], ...]:
    if protocol == "leave_one_run_out":
        groups = {value.case_id for value in waves}
        key = lambda value: value.case_id
    elif protocol == "leave_one_workload_out":
        groups = {value.workload for value in waves}
        key = lambda value: value.workload
    elif protocol == "leave_one_model_out":
        groups = {value.model for value in waves}
        key = lambda value: value.model
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    result = []
    for group in sorted(groups):
        train = tuple(value for value in waves if key(value) != group)
        test = tuple(value for value in waves if key(value) == group)
        result.append((group, train, test))
    return tuple(result)


def _run_protocol(waves: Sequence[Wave], protocol: str) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy_id, configuration in POLICIES.items():
        output_rows: list[dict[str, Any]] = []
        fold_records: list[dict[str, Any]] = []
        for held_out, train, test in _folds(waves, protocol):
            parameters = _inner_tune(
                train,
                feature_names=configuration["feature_names"],
                protected_exploration=configuration["protected_exploration"],
                joint=configuration["joint"],
            )
            rows = _evaluate_policy(
                train,
                test,
                feature_names=configuration["feature_names"],
                alpha=float(parameters["alpha"]),
                pair_alpha=(
                    None
                    if parameters["pair_alpha"] is None
                    else float(parameters["pair_alpha"])
                ),
                pair_weight=float(parameters["pair_weight"]),
                protected_exploration=configuration["protected_exploration"],
            )
            output_rows.extend(rows)
            fold_records.append(
                {
                    "held_out": held_out,
                    "train_case_ids": sorted({value.case_id for value in train}),
                    "test_case_ids": sorted({value.case_id for value in test}),
                    "hyperparameters": parameters,
                    "test_mean_case_oracle_fraction": _group_score(rows),
                }
            )
        policies[policy_id] = {
            "mean_case_oracle_fraction": _group_score(output_rows),
            "folds": fold_records,
            "rows": output_rows,
        }
    return {"protocol": protocol, "policies": policies}


def _case_summary(waves: Sequence[Wave]) -> list[dict[str, Any]]:
    result = []
    for case_id in sorted({value.case_id for value in waves}):
        rows = [value for value in waves if value.case_id == case_id]
        oracle = sum(value.oracle_gain for value in rows)
        current = sum(value.current_gain for value in rows)
        direct = sum(value.direct_gain for value in rows)
        uniform = sum(value.uniform_expected_gain for value in rows)
        published_direct = sum(
            value.published_literal_direct_gain for value in rows
        )
        published_oracle = sum(value.published_oracle_gain for value in rows)
        published_uniform = sum(
            value.published_uniform_expected_gain for value in rows
        )
        unconstrained_oracle = sum(
            value.unconstrained_oracle_gain for value in rows
        )
        unconstrained_uniform = sum(
            value.unconstrained_uniform_expected_gain for value in rows
        )
        feasible_counts = [len(value.feasible_subsets) for value in rows]
        result.append(
            {
                "case_id": case_id,
                "workload": rows[0].workload,
                "model": rows[0].model,
                "wave_count": len(rows),
                "current_gain_sum": current,
                "direct_gain_sum": direct,
                "uniform_expected_gain_sum": uniform,
                "oracle_gain_sum": oracle,
                "published_literal_direct_gain_sum": published_direct,
                "published_oracle_gain_sum": published_oracle,
                "published_uniform_expected_gain_sum": published_uniform,
                "unconstrained_oracle_gain_sum": unconstrained_oracle,
                "unconstrained_uniform_expected_gain_sum": unconstrained_uniform,
                "feasible_k4_subset_count_min": min(feasible_counts),
                "feasible_k4_subset_count_max": max(feasible_counts),
                "feasible_k4_subset_count_mean": fmean(feasible_counts),
                "current_oracle_fraction": current / oracle,
                "direct_oracle_fraction": direct / oracle,
                "current_multiple_of_uniform": current / uniform,
                "direct_multiple_of_uniform": direct / uniform,
                "current_minus_direct": current - direct,
                "feasibility_oracle_loss": unconstrained_oracle - oracle,
                "feasibility_uniform_shift": uniform - unconstrained_uniform,
            }
        )
    return result


def _promotion_audit(
    case_summary: Sequence[dict[str, Any]],
    loro: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_workload: dict[str, dict[str, float]] = {}
    for workload in sorted({value["workload"] for value in case_summary}):
        rows = [value for value in case_summary if value["workload"] == workload]
        baseline_by_workload[workload] = {
            "current": fmean(value["current_oracle_fraction"] for value in rows),
            "direct": fmean(value["direct_oracle_fraction"] for value in rows),
        }
    audits: dict[str, Any] = {}
    for policy_id, policy in loro["policies"].items():
        by_workload: dict[str, float] = {}
        for workload in baseline_by_workload:
            rows = [value for value in policy["rows"] if value["workload"] == workload]
            by_case: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                by_case.setdefault(row["case_id"], []).append(row)
            by_workload[workload] = fmean(
                sum(value["gain"] for value in case_rows)
                / sum(value["oracle_gain"] for value in case_rows)
                for case_rows in by_case.values()
            )
        wins = [
            workload
            for workload, value in by_workload.items()
            if value > baseline_by_workload[workload]["current"]
            and value > baseline_by_workload[workload]["direct"]
        ]
        no_material_losses = all(
            value
            >= 0.95
            * max(
                baseline_by_workload[workload]["current"],
                baseline_by_workload[workload]["direct"],
            )
            for workload, value in by_workload.items()
        )
        baseline_mean = max(
            fmean(value["current"] for value in baseline_by_workload.values()),
            fmean(value["direct"] for value in baseline_by_workload.values()),
        )
        reduces_regret = fmean(by_workload.values()) > baseline_mean
        audits[policy_id] = {
            "workload_oracle_fraction": by_workload,
            "workloads_beating_both_baselines": wins,
            "beats_both_on_at_least_two_workloads": len(wins) >= 2,
            "no_more_than_five_percent_relative_loss_on_any_workload": (
                no_material_losses
            ),
            "reduces_mean_oracle_regret": reduces_regret,
            "promotion_gate_passed": (
                len(wins) >= 2 and no_material_losses and reduces_regret
            ),
        }
    return {
        "baseline_workload_oracle_fraction": baseline_by_workload,
        "policy_audits": audits,
    }


def _candidate_rows(waves: Sequence[Wave]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wave in waves:
        for member in wave.members:
            rows.append(
                {
                    "case_id": member.case_id,
                    "workload": member.workload,
                    "model": member.model,
                    "wave_ordinal": member.wave_ordinal,
                    "generation": wave.generation,
                    "parent_slot": wave.parent_slot,
                    "option_id": member.option_id,
                    "model_rank": member.model_rank,
                    "family": member.family,
                    "locus_key": member.locus_key,
                    "transition_path": member.transition_path,
                    "marginal_gain": member.marginal_gain,
                    "normalized_marginal_gain": member.normalized_marginal_gain,
                    "current_selected": member.current_selected,
                    "direct_selected": member.direct_selected,
                    "oracle_selected": member.oracle_selected,
                    **member.features,
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    specs = _case_specs()
    raw_by_case = {spec.case_id: _reconstructed_waves(spec) for spec in specs}
    loaded: list[Wave] = []
    for spec in specs:
        loaded.extend(_load_case(spec))
    waves = _with_pair_gains(tuple(loaded), raw_by_case)
    if len(waves) != 36 or len(_members(waves)) != 288:
        raise RuntimeError("Gate 2 panel must contain 36 waves and 288 candidates")
    cases = _case_summary(waves)
    protocols = {
        protocol: _run_protocol(waves, protocol)
        for protocol in (
            "leave_one_run_out",
            "leave_one_workload_out",
            "leave_one_model_out",
        )
    }
    result = {
        "schema_version": 1,
        "study": "gate2_cross_domain_complete_k8_support",
        "claim_scope": (
            "fixed_parent_fixed_slate_mechanism_development_not_campaign_"
            "counterfactual_efficacy_or_sota"
        ),
        "source_panel": [
            {
                "case_id": value.case_id,
                "workload": value.workload,
                "model": value.model,
                "source_run": value.source_run.relative_to(WORKSPACE_ROOT).as_posix(),
                "analysis": value.analysis.relative_to(WORKSPACE_ROOT).as_posix(),
            }
            for value in specs
        ],
        "accounting": {
            "case_count": len(specs),
            "workload_count": 3,
            "model_count": 4,
            "wave_count": len(waves),
            "candidate_outcome_count": len(_members(waves)),
            "executable_k4_subset_count_min": min(
                len(value.feasible_subsets) for value in waves
            ),
            "executable_k4_subset_count_max": max(
                len(value.feasible_subsets) for value in waves
            ),
            "executable_k4_subset_count_mean": fmean(
                len(value.feasible_subsets) for value in waves
            ),
            "unconstrained_k4_subsets_per_wave": 70,
            "provider_calls": 0,
        },
        "hyperparameter_grids": {
            "ridge_alpha": list(ALPHA_GRID),
            "pair_ridge_alpha": list(PAIR_ALPHA_GRID),
            "pair_weight": list(PAIR_WEIGHT_GRID),
            "selection_rule": "inner_training_fold_mean_case_oracle_fraction",
        },
        "case_summary": cases,
        "protocols": protocols,
        "promotion_audit": _promotion_audit(
            cases, protocols["leave_one_run_out"]
        ),
    }
    output_json = args.output_json.expanduser().resolve(strict=False)
    output_csv = args.output_csv.expanduser().resolve(strict=False)
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("Gate 2 analysis outputs must be fresh")
    write_json_atomic(output_json, result)
    _write_csv(output_csv, _candidate_rows(waves))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

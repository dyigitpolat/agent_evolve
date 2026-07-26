#!/usr/bin/env python3
"""Develop T-RAP on the authenticated complete-K8 evidence panel.

This is a fixed-parent, fixed-slate retrospective diagnostic.  It reconstructs
the six real Gate-2 campaigns, derives workload-neutral affine frontier targets
from information available before each proposal wave, and evaluates a
target-conditioned prequential K4 selector.  Rejected held-out outcomes never
enter a fit or posterior update.

The analysis protocol was frozen before inspecting these results in research
artifact 343.  This script must not be used to claim campaign efficacy or SOTA.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Iterable, Sequence

import numpy as np


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development import analyze_gate2_k8_support_panel as gate  # noqa: E402
from examples.development.durable_run_artifacts import write_json_atomic  # noqa: E402


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts"
)
DATA_ROOT = ARTIFACT_ROOT / "data"
PROTOCOL_PATH = (
    ARTIFACT_ROOT
    / "343_trap_prequential_complete_k8_development_protocol_20260722.md"
)
SOURCE_PANEL_PATH = DATA_ROOT / "gate2_k8_support_panel_v1.json"
DEFAULT_JSON = DATA_ROOT / "trap_prequential_complete_k8_development_v1.json"
DEFAULT_CSV = DATA_ROOT / "trap_prequential_complete_k8_development_v1.csv"

ALPHA_GRID = (0.1, 1.0, 10.0)
META_PRECISION_GRID = (0.25, 1.0)
DIRECTION_WEIGHT_GRID = (0.0, 0.25, 0.5)
UNCERTAINTY_WEIGHT_GRID = (0.0, 0.25)
CONFIDENCE_VALUE = {
    "unknown": 0.0,
    "low": 1.0 / 3.0,
    "medium": 2.0 / 3.0,
    "high": 1.0,
}


@dataclass(frozen=True, slots=True)
class FrontierTargetContext:
    """One prior-only lane target in normalized lower-is-better geometry."""

    metric_ids: tuple[str, ...]
    direction_id: str
    weights: tuple[float, ...]
    opportunity_rank: int
    archive_best_achievement: float
    opportunity_from_ideal: float
    parent_point: tuple[float, ...]
    parent_achievement: float
    parent_regret: float
    remaining_proposal_horizon: int


@dataclass(frozen=True, slots=True)
class PreparedWave:
    wave: gate.Wave
    target: FrontierTargetContext
    target_improvement_by_id: dict[str, float]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    policy_id: str
    target_features: bool
    direction_head: bool
    prequential_updates: bool


@dataclass(frozen=True, slots=True)
class Hyperparameters:
    alpha: float
    meta_precision: float
    direction_weight: float
    uncertainty_weight: float

    def as_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "meta_precision": self.meta_precision,
            "direction_weight": self.direction_weight,
            "uncertainty_weight": self.uncertainty_weight,
        }


@dataclass(frozen=True, slots=True)
class SequentialRidgeHead:
    """Linear-Gaussian head with immutable sufficient-statistic updates."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    precision: np.ndarray
    rhs: np.ndarray
    covariance: np.ndarray
    coefficients: np.ndarray
    residual_variance: float

    def _row(self, features: dict[str, float]) -> np.ndarray:
        raw = np.asarray(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=float,
        )
        return (raw - self.means) / self.scales

    def predict(self, features: dict[str, float]) -> float:
        return float(self._row(features) @ self.coefficients)

    def uncertainty(self, features: dict[str, float]) -> float:
        row = self._row(features)
        leverage = max(0.0, float(row @ self.covariance @ row))
        return math.sqrt(max(self.residual_variance, 1e-12) * leverage)

    def update(
        self,
        rows: Sequence[dict[str, float]],
        targets: Sequence[float],
    ) -> SequentialRidgeHead:
        if not rows or len(rows) != len(targets):
            raise ValueError("posterior update requires aligned non-empty rows")
        design = np.asarray([self._row(row) for row in rows], dtype=float)
        target = np.asarray(targets, dtype=float)
        precision = self.precision + design.T @ design
        rhs = self.rhs + design.T @ target
        covariance = np.linalg.pinv(precision)
        return replace(
            self,
            precision=precision,
            rhs=rhs,
            covariance=covariance,
            coefficients=covariance @ rhs,
        )


TARGET_FEATURES = (
    "target_favorable_fraction",
    "target_adverse_fraction",
    "target_abstention_fraction",
    "off_target_favorable_fraction",
    "off_target_adverse_fraction",
    "off_target_abstention_fraction",
    "target_declared_confidence",
    "target_posterior_correctness",
    "target_signed_evidence",
    "target_reliability_adjusted_evidence",
    "target_opportunity_from_ideal",
    "target_parent_achievement",
    "target_parent_regret",
    "target_active_axis_fraction",
    "target_zero_axis_fraction",
    *tuple(f"target_opportunity_rank_{value}" for value in range(1, 8)),
    "remaining_proposal_horizon",
    "remaining_proposal_horizon_fraction",
)
FULL_FEATURES = (*gate.PORTABLE_FEATURES, *TARGET_FEATURES)

POLICIES = (
    PolicySpec(
        policy_id="trap_prequential",
        target_features=True,
        direction_head=True,
        prequential_updates=True,
    ),
    PolicySpec(
        policy_id="target_static",
        target_features=True,
        direction_head=True,
        prequential_updates=False,
    ),
    PolicySpec(
        policy_id="prequential_no_target_features",
        target_features=False,
        direction_head=False,
        prequential_updates=True,
    ),
    PolicySpec(
        policy_id="prequential_no_direction_head",
        target_features=True,
        direction_head=False,
        prequential_updates=True,
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _achievement(point: Sequence[float], weights: Sequence[float]) -> float:
    active = tuple(
        (float(value), float(weight))
        for value, weight in zip(point, weights, strict=True)
        if weight > 0.0
    )
    if not active:
        raise ValueError("a frontier direction must activate an axis")
    maximum = max(weight * value for value, weight in active)
    weighted_mean = sum(weight * value for value, weight in active) / sum(
        weight for _, weight in active
    )
    return maximum + 0.05 * weighted_mean


def _reference_directions(dimension: int) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if dimension not in (2, 3):
        raise ValueError("T-RAP supports exact affine 2-D or 3-D geometry")
    values: list[tuple[str, tuple[float, ...]]] = [
        (
            f"axis_{index + 1}_extreme",
            tuple(1.0 if index == other else 0.0 for other in range(dimension)),
        )
        for index in range(dimension)
    ]
    if dimension == 3:
        values.extend(
            (
                f"axes_{left + 1}_{right + 1}_tradeoff",
                tuple(
                    0.5 if index in (left, right) else 0.0
                    for index in range(dimension)
                ),
            )
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
    values.append(
        (
            "balanced_tradeoff",
            tuple(1.0 / dimension for _ in range(dimension)),
        )
    )
    return tuple(
        (identifier, tuple(value / max(weights) for value in weights))
        for identifier, weights in values
    )


def _boils_pairs(value: object, *, name: str) -> dict[str, float]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list of metric/hex pairs")
    result: dict[str, float] = {}
    for pair in value:
        if (
            type(pair) is not list
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
        ):
            raise TypeError(f"{name} contains a malformed metric/hex pair")
        result[pair[0]] = float.fromhex(pair[1])
    return result


def _geometry(
    workload: str, raw_wave: dict[str, Any]
) -> tuple[
    tuple[str, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[str, float, float], ...],
]:
    if workload == "boils":
        snapshot = raw_wave["archive_reward_snapshot"]
        reference = _boils_pairs(snapshot["reference_point"], name="reference")
        metric_ids = tuple(reference)
        axes = tuple(("min", 0.0, reference[metric_id]) for metric_id in metric_ids)
        archive = tuple(
            tuple(
                _boils_pairs(point, name="archive point")[metric_id]
                / reference[metric_id]
                for metric_id in metric_ids
            )
            for point in snapshot["archive_points"]
        )
        return metric_ids, archive, axes
    snapshot = raw_wave["archive_snapshot"]
    raw_axes = snapshot["spec"]["axes"]
    metric_ids = tuple(value["metric_id"] for value in raw_axes)
    axes = tuple(
        (
            value["goal"],
            float.fromhex(value["ideal_hex"]),
            float.fromhex(value["reference_hex"]),
        )
        for value in raw_axes
    )
    archive = tuple(
        tuple(float.fromhex(cell) for cell in point)
        for point in snapshot["normalized_archive_points"]
    )
    return metric_ids, archive, axes


def _normalize_point(
    objectives: dict[str, float],
    metric_ids: Sequence[str],
    axes: Sequence[tuple[str, float, float]],
) -> tuple[float, ...]:
    result: list[float] = []
    for metric_id, (goal, ideal, reference) in zip(metric_ids, axes, strict=True):
        value = float(objectives[metric_id])
        if goal == "min":
            result.append((value - ideal) / (reference - ideal))
        elif goal == "max":
            result.append((ideal - value) / (ideal - reference))
        else:
            raise ValueError(f"unknown affine goal: {goal}")
    return tuple(result)


def _snapshot_record(workload: str, raw_wave: dict[str, Any]) -> object:
    return (
        raw_wave["archive_reward_snapshot"]
        if workload == "boils"
        else raw_wave["archive_snapshot"]
    )


def _lane_targets(
    *,
    workload: str,
    raw_pair: Sequence[dict[str, Any]],
) -> dict[int, FrontierTargetContext]:
    if len(raw_pair) != 2:
        raise ValueError("frontier allocation requires exactly two parent lanes")
    if _snapshot_record(workload, raw_pair[0]) != _snapshot_record(
        workload, raw_pair[1]
    ):
        raise RuntimeError("same-generation parent lanes use different archive cutoffs")
    metric_ids, archive, axes = _geometry(workload, raw_pair[0])
    if not archive:
        raise RuntimeError("frontier allocation requires a non-empty archive")
    directions: list[dict[str, object]] = []
    for direction_id, weights in _reference_directions(len(metric_ids)):
        archive_best = min(_achievement(point, weights) for point in archive)
        directions.append(
            {
                "direction_id": direction_id,
                "weights": weights,
                "archive_best": archive_best,
                "opportunity": max(0.0, archive_best),
            }
        )
    directions.sort(
        key=lambda value: (-float(value["opportunity"]), str(value["direction_id"]))
    )
    selected = directions[:2]
    rank_by_id = {
        str(value["direction_id"]): index
        for index, value in enumerate(directions, start=1)
    }
    remaining: dict[int, tuple[dict[str, Any], tuple[float, ...]]] = {}
    for raw_wave in raw_pair:
        slot = int(
            raw_wave.get("parent_slot", raw_wave["members"][0].get("parent_slot", 0))
        )
        if slot in remaining:
            raise RuntimeError("same-generation parent slots are not unique")
        remaining[slot] = (
            raw_wave,
            _normalize_point(raw_wave["parent_objectives"], metric_ids, axes),
        )
    assignments: dict[int, FrontierTargetContext] = {}
    generation = int(raw_pair[0]["generation"])
    remaining_horizon = {1: 2, 3: 1, 5: 0}[generation]
    for direction in selected:
        weights = direction["weights"]
        assert type(weights) is tuple
        slot = min(
            remaining,
            key=lambda value: (
                _achievement(remaining[value][1], weights),
                value,
            ),
        )
        _, parent_point = remaining.pop(slot)
        parent_achievement = _achievement(parent_point, weights)
        archive_best = float(direction["archive_best"])
        direction_id = str(direction["direction_id"])
        assignments[slot] = FrontierTargetContext(
            metric_ids=metric_ids,
            direction_id=direction_id,
            weights=weights,
            opportunity_rank=rank_by_id[direction_id],
            archive_best_achievement=archive_best,
            opportunity_from_ideal=float(direction["opportunity"]),
            parent_point=parent_point,
            parent_achievement=parent_achievement,
            parent_regret=max(0.0, parent_achievement - archive_best),
            remaining_proposal_horizon=remaining_horizon,
        )
    return assignments


def _weighted_metric_value(
    rows: dict[str, dict[str, Any]],
    metric_ids: Sequence[str],
    weights: Sequence[float],
    value,
) -> float:
    denominator = sum(weights)
    if denominator <= 0.0:
        return 0.0
    return sum(
        weight * float(value(rows[metric_id]))
        for metric_id, weight in zip(metric_ids, weights, strict=True)
    ) / denominator


def _target_features(
    *, score_row: dict[str, Any], target: FrontierTargetContext
) -> dict[str, float]:
    rows = {value["metric_id"]: value for value in score_row["metric_scores"]}
    if set(rows) != set(target.metric_ids):
        raise RuntimeError("prediction metrics differ from affine target axes")
    active = target.weights
    off_target = tuple(float(value == 0.0) for value in active)

    def weighted(weights: Sequence[float], key: str) -> float:
        return _weighted_metric_value(
            rows, target.metric_ids, weights, lambda row: bool(row[key])
        )

    correctness = lambda row: gate._float_hex(  # noqa: E731, SLF001
        row["calibration_cell"]["posterior_correctness_hex"],
        name="posterior correctness",
    )
    signed = lambda row: float(bool(row["favorable_assertion"])) - float(  # noqa: E731
        bool(row["adverse_assertion"])
    )
    result = {
        "target_favorable_fraction": weighted(active, "favorable_assertion"),
        "target_adverse_fraction": weighted(active, "adverse_assertion"),
        "target_abstention_fraction": weighted(active, "explicit_abstention"),
        "off_target_favorable_fraction": weighted(
            off_target, "favorable_assertion"
        ),
        "off_target_adverse_fraction": weighted(off_target, "adverse_assertion"),
        "off_target_abstention_fraction": weighted(
            off_target, "explicit_abstention"
        ),
        "target_declared_confidence": _weighted_metric_value(
            rows,
            target.metric_ids,
            active,
            lambda row: CONFIDENCE_VALUE[row["confidence"]],
        ),
        "target_posterior_correctness": _weighted_metric_value(
            rows, target.metric_ids, active, correctness
        ),
        "target_signed_evidence": _weighted_metric_value(
            rows, target.metric_ids, active, signed
        ),
        "target_reliability_adjusted_evidence": _weighted_metric_value(
            rows,
            target.metric_ids,
            active,
            lambda row: signed(row) * (2.0 * correctness(row) - 1.0),
        ),
        "target_opportunity_from_ideal": target.opportunity_from_ideal,
        "target_parent_achievement": target.parent_achievement,
        "target_parent_regret": target.parent_regret,
        "target_active_axis_fraction": sum(value > 0.0 for value in active)
        / len(active),
        "target_zero_axis_fraction": sum(value == 0.0 for value in active)
        / len(active),
        "remaining_proposal_horizon": float(target.remaining_proposal_horizon),
        "remaining_proposal_horizon_fraction": (
            target.remaining_proposal_horizon / 2.0
        ),
    }
    for rank in range(1, 8):
        result[f"target_opportunity_rank_{rank}"] = float(
            target.opportunity_rank == rank
        )
    return result


def _prepare_case(
    spec: gate.CaseSpec,
) -> tuple[tuple[PreparedWave, ...], list[dict[str, Any]]]:
    raw_waves = gate._reconstructed_waves(spec)  # noqa: SLF001
    waves = gate._load_case(  # noqa: SLF001
        spec, reconstructed_waves=raw_waves
    )
    analysis = gate._read_json(spec.analysis)  # noqa: SLF001
    outcomes = gate._outcomes_by_wave(analysis)  # noqa: SLF001
    raw_by_ordinal = {int(value["wave_ordinal"]): value for value in raw_waves}
    prepared: list[PreparedWave] = []
    target_rows: list[dict[str, Any]] = []
    for generation in (1, 3, 5):
        raw_pair = sorted(
            (value for value in raw_waves if int(value["generation"]) == generation),
            key=lambda value: int(
                value.get("parent_slot", value["members"][0].get("parent_slot", 0))
            ),
        )
        targets = _lane_targets(workload=spec.workload, raw_pair=raw_pair)
        for wave in sorted(
            (value for value in waves if value.generation == generation),
            key=lambda value: value.parent_slot,
        ):
            target = targets[wave.parent_slot]
            raw_wave = raw_by_ordinal[wave.wave_ordinal]
            score_by_id = gate._score_row_by_id(spec.workload, raw_wave)  # noqa: SLF001
            _, _, axes = _geometry(spec.workload, raw_wave)
            target_labels: dict[str, float] = {}
            enriched_members: list[gate.Member] = []
            denominator = max(
                target.opportunity_from_ideal,
                target.parent_achievement,
                1e-12,
            )
            for member in wave.members:
                point = _normalize_point(
                    outcomes[wave.wave_ordinal][member.option_id],
                    target.metric_ids,
                    axes,
                )
                candidate_achievement = _achievement(point, target.weights)
                target_improvement = max(
                    -1.0,
                    min(
                        1.0,
                        (target.parent_achievement - candidate_achievement)
                        / denominator,
                    ),
                )
                target_labels[member.option_id] = target_improvement
                features = {
                    **member.features,
                    **_target_features(
                        score_row=score_by_id[member.option_id], target=target
                    ),
                }
                enriched_members.append(replace(member, features=features))
                target_rows.append(
                    {
                        "case_id": spec.case_id,
                        "workload": spec.workload,
                        "model": spec.model,
                        "wave_ordinal": wave.wave_ordinal,
                        "generation": generation,
                        "parent_slot": wave.parent_slot,
                        "option_id": member.option_id,
                        "direction_id": target.direction_id,
                        "opportunity_rank": target.opportunity_rank,
                        "target_improvement": target_improvement,
                        "normalized_marginal_gain": member.normalized_marginal_gain,
                    }
                )
            prepared.append(
                PreparedWave(
                    wave=replace(wave, members=tuple(enriched_members)),
                    target=target,
                    target_improvement_by_id=target_labels,
                )
            )
    return tuple(sorted(prepared, key=lambda value: value.wave.wave_ordinal)), target_rows


def _all_members(waves: Iterable[PreparedWave]) -> list[gate.Member]:
    return [member for value in waves for member in value.wave.members]


def _fit_prior(
    *,
    waves: Sequence[PreparedWave],
    feature_names: tuple[str, ...],
    target_kind: str,
    alpha: float,
    meta_precision: float,
) -> SequentialRidgeHead:
    members = _all_members(waves)
    if not members:
        raise ValueError("a meta-prior requires training candidates")
    raw = np.asarray(
        [
            [float(member.features.get(name, 0.0)) for name in feature_names]
            for member in members
        ],
        dtype=float,
    )
    if target_kind == "marginal":
        target = np.asarray(
            [member.normalized_marginal_gain for member in members], dtype=float
        )
    elif target_kind == "direction":
        labels = {
            (value.wave.case_id, value.wave.wave_ordinal): value.target_improvement_by_id
            for value in waves
        }
        target = np.asarray(
            [
                labels[(member.case_id, member.wave_ordinal)][member.option_id]
                for member in members
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"unknown target kind: {target_kind}")
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    bias_index = feature_names.index("bias")
    means[bias_index] = 0.0
    scales[bias_index] = 1.0
    scales[scales < 1e-12] = 1.0
    design = (raw - means) / scales
    penalty = np.eye(len(feature_names), dtype=float) * alpha
    penalty[bias_index, bias_index] = 0.0
    precision = meta_precision * (design.T @ design) + penalty
    rhs = meta_precision * (design.T @ target)
    covariance = np.linalg.pinv(precision)
    coefficients = covariance @ rhs
    residual = target - design @ coefficients
    return SequentialRidgeHead(
        feature_names=feature_names,
        means=means,
        scales=scales,
        precision=precision,
        rhs=rhs,
        covariance=covariance,
        coefficients=coefficients,
        residual_variance=max(float(np.mean(np.square(residual))), 1e-12),
    )


def _z_scores(values: dict[str, float]) -> dict[str, float]:
    mean = fmean(values.values())
    scale = math.sqrt(fmean((value - mean) ** 2 for value in values.values()))
    if scale < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - mean) / scale for key, value in values.items()}


def _select(
    *,
    prepared: PreparedWave,
    marginal_head: SequentialRidgeHead,
    direction_head: SequentialRidgeHead | None,
    parameters: Hyperparameters,
) -> tuple[tuple[str, ...], dict[str, dict[str, float]]]:
    wave = prepared.wave
    by_id = {member.option_id: member for member in wave.members}
    marginal = {
        option_id: marginal_head.predict(member.features)
        for option_id, member in by_id.items()
    }
    direction = {
        option_id: (
            0.0
            if direction_head is None
            else direction_head.predict(member.features)
        )
        for option_id, member in by_id.items()
    }
    uncertainty = {
        option_id: marginal_head.uncertainty(member.features)
        for option_id, member in by_id.items()
    }
    marginal_z = _z_scores(marginal)
    direction_z = _z_scores(direction)
    uncertainty_z = _z_scores(uncertainty)
    horizon = prepared.target.remaining_proposal_horizon / 2.0
    score = {
        option_id: (
            marginal_z[option_id]
            + parameters.direction_weight * direction_z[option_id]
            + parameters.uncertainty_weight * horizon * uncertainty_z[option_id]
        )
        for option_id in by_id
    }
    selected = min(
        wave.feasible_subsets,
        key=lambda subset: (-sum(score[option_id] for option_id in subset), subset),
    )
    diagnostics = {
        option_id: {
            "predicted_marginal": marginal[option_id],
            "predicted_direction": direction[option_id],
            "epistemic_uncertainty": uncertainty[option_id],
            "selection_score": score[option_id],
        }
        for option_id in by_id
    }
    return selected, diagnostics


_PRIOR_CACHE: dict[
    tuple[tuple[str, ...], tuple[str, ...], str, float, float],
    SequentialRidgeHead,
] = {}


def _cached_prior(
    *,
    all_waves: Sequence[PreparedWave],
    train_case_ids: Sequence[str],
    feature_names: tuple[str, ...],
    target_kind: str,
    alpha: float,
    meta_precision: float,
) -> SequentialRidgeHead:
    key = (
        tuple(sorted(train_case_ids)),
        feature_names,
        target_kind,
        alpha,
        meta_precision,
    )
    cached = _PRIOR_CACHE.get(key)
    if cached is not None:
        return cached
    train_set = set(train_case_ids)
    value = _fit_prior(
        waves=tuple(value for value in all_waves if value.wave.case_id in train_set),
        feature_names=feature_names,
        target_kind=target_kind,
        alpha=alpha,
        meta_precision=meta_precision,
    )
    _PRIOR_CACHE[key] = value
    return value


def _simulate_case(
    *,
    all_waves: Sequence[PreparedWave],
    train_case_ids: Sequence[str],
    test_case_id: str,
    policy: PolicySpec,
    parameters: Hyperparameters,
) -> tuple[dict[str, Any], ...]:
    feature_names = FULL_FEATURES if policy.target_features else gate.PORTABLE_FEATURES
    marginal = _cached_prior(
        all_waves=all_waves,
        train_case_ids=train_case_ids,
        feature_names=feature_names,
        target_kind="marginal",
        alpha=parameters.alpha,
        meta_precision=parameters.meta_precision,
    )
    direction = (
        _cached_prior(
            all_waves=all_waves,
            train_case_ids=train_case_ids,
            feature_names=feature_names,
            target_kind="direction",
            alpha=parameters.alpha,
            meta_precision=parameters.meta_precision,
        )
        if policy.direction_head
        else None
    )
    frozen_marginal = marginal
    frozen_direction = direction
    test_waves = tuple(
        sorted(
            (value for value in all_waves if value.wave.case_id == test_case_id),
            key=lambda value: value.wave.wave_ordinal,
        )
    )
    rows: list[dict[str, Any]] = []
    for generation in (1, 3, 5):
        generation_waves = tuple(
            value for value in test_waves if value.wave.generation == generation
        )
        update_members: list[gate.Member] = []
        update_direction_targets: list[float] = []
        for prepared in generation_waves:
            selected, diagnostics = _select(
                prepared=prepared,
                marginal_head=marginal,
                direction_head=direction,
                parameters=parameters,
            )
            static_selected, _ = _select(
                prepared=prepared,
                marginal_head=frozen_marginal,
                direction_head=frozen_direction,
                parameters=parameters,
            )
            wave = prepared.wave
            by_id = {member.option_id: member for member in wave.members}
            selected_target = [
                prepared.target_improvement_by_id[option_id]
                for option_id in selected
            ]
            rows.append(
                {
                    "policy_id": policy.policy_id,
                    "case_id": wave.case_id,
                    "workload": wave.workload,
                    "model": wave.model,
                    "wave_ordinal": wave.wave_ordinal,
                    "generation": wave.generation,
                    "parent_slot": wave.parent_slot,
                    "direction_id": prepared.target.direction_id,
                    "opportunity_rank": prepared.target.opportunity_rank,
                    "remaining_proposal_horizon": (
                        prepared.target.remaining_proposal_horizon
                    ),
                    "selected_option_ids": list(selected),
                    "selected_model_ranks": [
                        by_id[option_id].model_rank for option_id in selected
                    ],
                    "selected_target_improvements": selected_target,
                    "selected_target_positive_count": sum(
                        value > 0.0 for value in selected_target
                    ),
                    "selected_target_improvement_mean": fmean(selected_target),
                    "gain": wave.gain_by_subset[selected],
                    "oracle_gain": wave.oracle_gain,
                    "current_gain": wave.current_gain,
                    "direct_gain": wave.direct_gain,
                    "uniform_expected_gain": wave.uniform_expected_gain,
                    "online_changed_vs_same_prior_static": selected
                    != static_selected,
                    "same_prior_static_option_ids": list(static_selected),
                    "same_prior_static_gain": wave.gain_by_subset[static_selected],
                    "gain_minus_same_prior_static": (
                        wave.gain_by_subset[selected]
                        - wave.gain_by_subset[static_selected]
                    ),
                    "candidate_diagnostics": diagnostics,
                }
            )
            update_members.extend(by_id[option_id] for option_id in selected)
            update_direction_targets.extend(selected_target)
        if policy.prequential_updates:
            marginal = marginal.update(
                [member.features for member in update_members],
                [member.normalized_marginal_gain for member in update_members],
            )
            if direction is not None:
                direction = direction.update(
                    [member.features for member in update_members],
                    update_direction_targets,
                )
    return tuple(rows)


def _case_oracle_fraction(rows: Sequence[dict[str, Any]]) -> float:
    oracle = sum(float(value["oracle_gain"]) for value in rows)
    if oracle == 0.0:
        return 0.0
    return sum(float(value["gain"]) for value in rows) / oracle


def _parameter_grid(policy: PolicySpec) -> tuple[Hyperparameters, ...]:
    direction_weights = DIRECTION_WEIGHT_GRID if policy.direction_head else (0.0,)
    return tuple(
        Hyperparameters(alpha, meta, direction, uncertainty)
        for alpha, meta, direction, uncertainty in product(
            ALPHA_GRID,
            META_PRECISION_GRID,
            direction_weights,
            UNCERTAINTY_WEIGHT_GRID,
        )
    )


def _tune(
    *,
    all_waves: Sequence[PreparedWave],
    outer_train_case_ids: Sequence[str],
    policy: PolicySpec,
) -> tuple[Hyperparameters, list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    for parameters in _parameter_grid(policy):
        fractions: list[float] = []
        for validation_case in sorted(outer_train_case_ids):
            meta_cases = tuple(
                value for value in outer_train_case_ids if value != validation_case
            )
            rows = _simulate_case(
                all_waves=all_waves,
                train_case_ids=meta_cases,
                test_case_id=validation_case,
                policy=policy,
                parameters=parameters,
            )
            fractions.append(_case_oracle_fraction(rows))
        scores.append(
            {
                "hyperparameters": parameters.as_dict(),
                "inner_mean_case_oracle_fraction": fmean(fractions),
                "inner_case_oracle_fractions": fractions,
            }
        )
    winner = max(
        scores,
        key=lambda value: (
            float(value["inner_mean_case_oracle_fraction"]),
            float(value["hyperparameters"]["alpha"]),
            float(value["hyperparameters"]["meta_precision"]),
            -float(value["hyperparameters"]["direction_weight"]),
            -float(value["hyperparameters"]["uncertainty_weight"]),
        ),
    )
    raw = winner["hyperparameters"]
    selected = Hyperparameters(
        alpha=float(raw["alpha"]),
        meta_precision=float(raw["meta_precision"]),
        direction_weight=float(raw["direction_weight"]),
        uncertainty_weight=float(raw["uncertainty_weight"]),
    )
    return selected, scores


def _baseline_cases(waves: Sequence[PreparedWave]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    case_ids = sorted({value.wave.case_id for value in waves})
    for case_id in case_ids:
        case = [value.wave for value in waves if value.wave.case_id == case_id]
        oracle = sum(value.oracle_gain for value in case)
        current = sum(value.current_gain for value in case)
        direct = sum(value.direct_gain for value in case)
        uniform = sum(value.uniform_expected_gain for value in case)
        result.append(
            {
                "case_id": case_id,
                "workload": case[0].workload,
                "model": case[0].model,
                "current_gain": current,
                "direct_gain": direct,
                "uniform_expected_gain": uniform,
                "oracle_gain": oracle,
                "current_oracle_fraction": current / oracle,
                "direct_oracle_fraction": direct / oracle,
                "uniform_oracle_fraction": uniform / oracle,
            }
        )
    return result


def _policy_case_summary(
    policy_id: str, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case_id in sorted({value["case_id"] for value in rows}):
        case = [value for value in rows if value["case_id"] == case_id]
        gain = sum(value["gain"] for value in case)
        oracle = sum(value["oracle_gain"] for value in case)
        selected_ranks = [
            rank for value in case for rank in value["selected_model_ranks"]
        ]
        target_values = [
            target
            for value in case
            for target in value["selected_target_improvements"]
        ]
        result.append(
            {
                "policy_id": policy_id,
                "case_id": case_id,
                "workload": case[0]["workload"],
                "model": case[0]["model"],
                "gain": gain,
                "oracle_gain": oracle,
                "oracle_fraction": gain / oracle,
                "target_improvement_rate": sum(value > 0.0 for value in target_values)
                / len(target_values),
                "target_improvement_mean": fmean(target_values),
                "selected_rank_mean": fmean(selected_ranks),
                "same_prior_static_gain": sum(
                    value["same_prior_static_gain"] for value in case
                ),
                "gain_minus_same_prior_static": sum(
                    value["gain_minus_same_prior_static"] for value in case
                ),
                "online_changed_later_wave_count": sum(
                    value["generation"] > 1
                    and value["online_changed_vs_same_prior_static"]
                    for value in case
                ),
            }
        )
    return result


def _workload_summary(
    baseline_cases: Sequence[dict[str, Any]],
    policy_cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for workload in sorted({value["workload"] for value in baseline_cases}):
        baseline = [value for value in baseline_cases if value["workload"] == workload]
        row: dict[str, Any] = {
            "workload": workload,
            "current_oracle_fraction": fmean(
                value["current_oracle_fraction"] for value in baseline
            ),
            "direct_oracle_fraction": fmean(
                value["direct_oracle_fraction"] for value in baseline
            ),
            "uniform_oracle_fraction": fmean(
                value["uniform_oracle_fraction"] for value in baseline
            ),
        }
        for policy_id in sorted({value["policy_id"] for value in policy_cases}):
            values = [
                value
                for value in policy_cases
                if value["policy_id"] == policy_id
                and value["workload"] == workload
            ]
            row[f"{policy_id}_oracle_fraction"] = fmean(
                value["oracle_fraction"] for value in values
            )
        result.append(row)
    return result


def _promotion_audit(workloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    policy_ids = tuple(value.policy_id for value in POLICIES)
    audits: dict[str, Any] = {}
    baseline_mean = max(
        fmean(value["current_oracle_fraction"] for value in workloads),
        fmean(value["direct_oracle_fraction"] for value in workloads),
    )
    for policy_id in policy_ids:
        key = f"{policy_id}_oracle_fraction"
        wins = [
            value["workload"]
            for value in workloads
            if value[key] > value["current_oracle_fraction"]
            and value[key] > value["direct_oracle_fraction"]
        ]
        no_material_losses = all(
            value[key]
            >= 0.95
            * max(
                value["current_oracle_fraction"], value["direct_oracle_fraction"]
            )
            for value in workloads
        )
        reduces_regret = fmean(value[key] for value in workloads) > baseline_mean
        audits[policy_id] = {
            "workload_oracle_fraction": {
                value["workload"]: value[key] for value in workloads
            },
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
        "baseline_mean_workload_oracle_fraction": baseline_mean,
        "policy_audits": audits,
    }


def _generation_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for policy_id in sorted({value["policy_id"] for value in rows}):
        for generation in (1, 3, 5):
            values = [
                value
                for value in rows
                if value["policy_id"] == policy_id
                and value["generation"] == generation
            ]
            gain = sum(value["gain"] for value in values)
            oracle = sum(value["oracle_gain"] for value in values)
            result.append(
                {
                    "policy_id": policy_id,
                    "generation": generation,
                    "gain": gain,
                    "oracle_gain": oracle,
                    "oracle_fraction": 0.0 if oracle == 0.0 else gain / oracle,
                    "online_changed_wave_count": sum(
                        value["online_changed_vs_same_prior_static"]
                        for value in values
                    ),
                    "same_prior_static_gain": sum(
                        value["same_prior_static_gain"] for value in values
                    ),
                    "gain_minus_same_prior_static": sum(
                        value["gain_minus_same_prior_static"] for value in values
                    ),
                }
            )
    return result


def _target_summary(target_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    directions: dict[str, int] = {}
    for value in target_rows:
        directions[value["direction_id"]] = directions.get(value["direction_id"], 0) + 1
    return {
        "candidate_target_label_count": len(target_rows),
        "direction_candidate_counts": directions,
        "positive_target_label_rate": sum(
            value["target_improvement"] > 0.0 for value in target_rows
        )
        / len(target_rows),
        "positive_marginal_label_rate": sum(
            value["normalized_marginal_gain"] > 0.0 for value in target_rows
        )
        / len(target_rows),
        "target_marginal_sign_agreement_rate": sum(
            (value["target_improvement"] > 0.0)
            == (value["normalized_marginal_gain"] > 0.0)
            for value in target_rows
        )
        / len(target_rows),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "policy_id",
        "case_id",
        "workload",
        "model",
        "wave_ordinal",
        "generation",
        "parent_slot",
        "direction_id",
        "opportunity_rank",
        "selected_option_ids",
        "selected_model_ranks",
        "gain",
        "oracle_gain",
        "current_gain",
        "direct_gain",
        "uniform_expected_gain",
        "selected_target_positive_count",
        "selected_target_improvement_mean",
        "online_changed_vs_same_prior_static",
        "same_prior_static_gain",
        "gain_minus_same_prior_static",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "|".join(str(value) for value in row[key])
                        if type(row[key]) is list
                        else row[key]
                    )
                    for key in fields
                }
            )


def analyze() -> dict[str, Any]:
    prepared: list[PreparedWave] = []
    target_rows: list[dict[str, Any]] = []
    for spec in gate._case_specs():  # noqa: SLF001
        case_waves, case_targets = _prepare_case(spec)
        prepared.extend(case_waves)
        target_rows.extend(case_targets)
    all_waves = tuple(prepared)
    case_ids = sorted({value.wave.case_id for value in all_waves})
    baseline_cases = _baseline_cases(all_waves)
    policy_records: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    all_policy_cases: list[dict[str, Any]] = []
    for policy in POLICIES:
        policy_rows: list[dict[str, Any]] = []
        folds: list[dict[str, Any]] = []
        for held_out in case_ids:
            train_cases = tuple(value for value in case_ids if value != held_out)
            parameters, tuning_rows = _tune(
                all_waves=all_waves,
                outer_train_case_ids=train_cases,
                policy=policy,
            )
            rows = _simulate_case(
                all_waves=all_waves,
                train_case_ids=train_cases,
                test_case_id=held_out,
                policy=policy,
                parameters=parameters,
            )
            policy_rows.extend(rows)
            folds.append(
                {
                    "held_out_case_id": held_out,
                    "meta_train_case_ids": list(train_cases),
                    "selected_hyperparameters": parameters.as_dict(),
                    "inner_selected_score": max(
                        value["inner_mean_case_oracle_fraction"]
                        for value in tuning_rows
                    ),
                    "held_out_oracle_fraction": _case_oracle_fraction(rows),
                    "closed_grid_results": tuning_rows,
                }
            )
        cases = _policy_case_summary(policy.policy_id, policy_rows)
        all_rows.extend(policy_rows)
        all_policy_cases.extend(cases)
        policy_records[policy.policy_id] = {
            "specification": {
                "target_features": policy.target_features,
                "direction_head": policy.direction_head,
                "prequential_updates": policy.prequential_updates,
            },
            "mean_case_oracle_fraction": fmean(
                value["oracle_fraction"] for value in cases
            ),
            "case_summary": cases,
            "folds": folds,
            "rows": policy_rows,
        }
    workload_summary = _workload_summary(baseline_cases, all_policy_cases)
    return {
        "schema_version": 1,
        "analysis_id": "trap_prequential_complete_k8_development_v1",
        "claim_boundary": (
            "retrospective fixed-parent/fixed-slate mechanism development; "
            "not campaign efficacy, model ranking, or SOTA evidence"
        ),
        "information_boundary": {
            "outer_protocol": "leave_one_run_out",
            "inner_protocol": "nested_leave_one_case_out",
            "held_out_updates": (
                "policy-selected prior-generation actions only; both lanes concurrent"
            ),
            "rejected_held_out_outcomes_enter_fit": False,
            "future_held_out_outcomes_enter_fit": False,
            "workload_model_provider_option_identifiers_enter_features": False,
            "historical_fixed_trace_score_state_retained": True,
            "historical_score_state_can_encode_deployed_prior_action_outcomes": True,
            "counterfactual_prompt_memory_calibration_state_reconstructed": False,
        },
        "provenance": {
            "protocol_path": str(PROTOCOL_PATH.relative_to(WORKSPACE_ROOT)),
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "source_panel_path": str(SOURCE_PANEL_PATH.relative_to(WORKSPACE_ROOT)),
            "source_panel_sha256": _sha256(SOURCE_PANEL_PATH),
            "source_case_ids": case_ids,
        },
        "evidence_counts": {
            "case_count": len(case_ids),
            "workload_count": len({value.wave.workload for value in all_waves}),
            "model_family_count": len({value.wave.model for value in all_waves}),
            "wave_count": len(all_waves),
            "real_candidate_outcome_count": sum(
                len(value.wave.members) for value in all_waves
            ),
            "selected_actions_per_policy": len(all_waves) * 4,
        },
        "closed_hyperparameter_grid": {
            "alpha": list(ALPHA_GRID),
            "meta_precision": list(META_PRECISION_GRID),
            "direction_weight": list(DIRECTION_WEIGHT_GRID),
            "uncertainty_weight": list(UNCERTAINTY_WEIGHT_GRID),
            "tie_break": (
                "larger alpha, larger meta precision, smaller direction weight, "
                "smaller uncertainty weight, canonical option IDs"
            ),
        },
        "feature_contract": {
            "portable_feature_count": len(gate.PORTABLE_FEATURES),
            "target_feature_count": len(TARGET_FEATURES),
            "full_feature_count": len(FULL_FEATURES),
            "portable_features": list(gate.PORTABLE_FEATURES),
            "target_features": list(TARGET_FEATURES),
        },
        "target_label_summary": _target_summary(target_rows),
        "baseline_case_summary": baseline_cases,
        "policies": policy_records,
        "workload_summary": workload_summary,
        "promotion_audit": _promotion_audit(workload_summary),
        "generation_summary": _generation_summary(all_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    arguments = parser.parse_args()
    result = analyze()
    write_json_atomic(arguments.output_json, result)
    rows = [
        row
        for policy in result["policies"].values()
        for row in policy["rows"]
    ]
    _write_csv(arguments.output_csv, rows)
    summary = {
        "output_json": str(arguments.output_json),
        "output_csv": str(arguments.output_csv),
        "promotion_audit": result["promotion_audit"],
        "workload_summary": result["workload_summary"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

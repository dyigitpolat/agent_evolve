"""Authenticated portable feature projection for T-RAP profile v1.

The projector converts typed, prior-only engine facts into the exact 70-column
development schema.  It is intentionally separate from the acquisition model:
workload adapters materialize configurations and archive snapshots, while this
module owns generic normalization, transition differencing, target alignment,
feature order, and cryptographic receipts.

Profile v1 is scale-qualified for an eight-member proposal slate, two parent
lanes, and proposal generations 1/3/5.  Those are explicit validation limits,
not hidden assumptions of the generic acquisition core.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar, Sequence

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.selection.calibrated_slate import SlateAllocationRequest
from agent_evolve.policies.selection.forecast_calibration import ForecastConfidenceBin
from agent_evolve.policies.selection.structural_posterior_slate import (
    StructuralPosteriorMemberScoreRow,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    TargetConditionedMemberFeatures,
)
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjection,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget


PROJECTOR_ID = "trap_portable_features_v1"
PROJECTOR_VERSION = 1
PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:trap-portable-features:v1;"
    b"schema=70-exact-columns;inputs=sealed-k8,prior-structural-score-rows,"
    b"typed-parent-child-configurations,authenticated-affine-archive-context,"
    b"authenticated-frontier-target,generation,lane-slot,remaining-horizon;"
    b"scale=k8,two-lanes,generations-1-3-5,horizon-2;"
    b"missing-extra-nonfinite=fail-closed;"
    b"workload-model-provider-option-name-current-outcome-features=false"
).hexdigest()

PORTABLE_FEATURES = (
    "bias",
    *tuple(f"rank_{value}" for value in range(1, 9)),
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
FEATURE_NAMES = (*PORTABLE_FEATURES, *TARGET_FEATURES)

_TRANSITION_DOMAIN = b"agent-evolve:trap-portable-transition:v1\x00"
_PROJECTION_REQUEST_DOMAIN = b"agent-evolve:trap-feature-request:v1\x00"
_CONFIDENCE_VALUE = {
    ForecastConfidenceBin.UNKNOWN: 0.0,
    ForecastConfidenceBin.LOW: 1.0 / 3.0,
    ForecastConfidenceBin.MEDIUM: 2.0 / 3.0,
    ForecastConfidenceBin.HIGH: 1.0,
}


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


def _finite(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be an exact finite float")


def _decimal(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be finite decimal text")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} is not finite decimal text") from error
    _finite(result, name=name)
    return result


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


@dataclass(frozen=True, slots=True, eq=False)
class PortableTransitionReceipt:
    """Generic structural delta between one parent and proposed child."""

    option_id: str
    option_identity_sha256: str
    parent_configuration_sha256: str
    child_configuration_sha256: str
    changed_paths: tuple[str, ...]
    change_count: float
    numeric_fraction: float
    categorical_fraction: float
    relative_numeric_delta_log: float
    numeric_sign: float
    path_depth: float

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        for name in (
            "option_identity_sha256",
            "parent_configuration_sha256",
            "child_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.changed_paths) is not tuple or any(
            type(value) is not str or not value for value in self.changed_paths
        ):
            raise TypeError("changed_paths must be an exact string tuple")
        if self.changed_paths != tuple(sorted(set(self.changed_paths))):
            raise ValueError("changed_paths must be unique and canonical")
        for name in (
            "change_count",
            "numeric_fraction",
            "categorical_fraction",
            "relative_numeric_delta_log",
            "numeric_sign",
            "path_depth",
        ):
            _finite(getattr(self, name), name=name)
        if self.change_count != float(len(self.changed_paths)):
            raise ValueError("change_count differs from changed_paths")
        for name in ("numeric_fraction", "categorical_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not -1.0 <= self.numeric_sign <= 1.0:
            raise ValueError("numeric_sign must lie in [-1, 1]")
        if self.relative_numeric_delta_log < 0.0 or self.path_depth < 0.0:
            raise ValueError("transition magnitudes must be non-negative")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "changed_paths": list(self.changed_paths),
            "features_hex": {
                "transition_change_count": self.change_count.hex(),
                "transition_numeric_fraction": self.numeric_fraction.hex(),
                "transition_categorical_fraction": self.categorical_fraction.hex(),
                "transition_relative_numeric_delta_log": (
                    self.relative_numeric_delta_log.hex()
                ),
                "transition_numeric_sign": self.numeric_sign.hex(),
                "transition_path_depth": self.path_depth.hex(),
            },
        }

    @property
    def transition_sha256(self) -> str:
        return _hash(_TRANSITION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "transition_sha256": self.transition_sha256,
        }

    def feature_map(self) -> dict[str, float]:
        self.__post_init__()
        return {
            "transition_change_count": self.change_count,
            "transition_numeric_fraction": self.numeric_fraction,
            "transition_categorical_fraction": self.categorical_fraction,
            "transition_relative_numeric_delta_log": (
                self.relative_numeric_delta_log
            ),
            "transition_numeric_sign": self.numeric_sign,
            "transition_path_depth": self.path_depth,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is PortableTransitionReceipt
            and self.transition_sha256 == other.transition_sha256
        )

    __hash__ = None


def project_portable_transition(
    *,
    option_id: str,
    option_identity_sha256: str,
    parent_configuration: FrozenJsonValue,
    child_configuration: FrozenJsonValue,
) -> PortableTransitionReceipt:
    """Compute the profile-v1 configuration delta without domain knowledge."""

    parent_sha256 = typed_json_sha256(parent_configuration)
    child_sha256 = typed_json_sha256(child_configuration)
    parent = _flatten(thaw_json(parent_configuration))
    child = _flatten(thaw_json(child_configuration))
    paths = sorted(set(parent) | set(child))
    changed = tuple(path for path in paths if parent.get(path) != child.get(path))
    numeric_relative: list[float] = []
    numeric_signs: list[float] = []
    categorical = 0
    for path in changed:
        old = parent.get(path)
        new = child.get(path)
        if (
            type(old) in {int, float}
            and type(new) in {int, float}
            and type(old) is not bool
            and type(new) is not bool
        ):
            delta = float(new) - float(old)
            numeric_relative.append(abs(delta) / max(abs(float(old)), 1e-12))
            numeric_signs.append(
                0.0 if delta == 0.0 else math.copysign(1.0, delta)
            )
        else:
            categorical += 1
    denominator = max(1, len(changed))
    depths = [path.count(".") + path.count("[") for path in changed]
    return PortableTransitionReceipt(
        option_id=option_id,
        option_identity_sha256=option_identity_sha256,
        parent_configuration_sha256=parent_sha256,
        child_configuration_sha256=child_sha256,
        changed_paths=changed,
        change_count=float(len(changed)),
        numeric_fraction=len(numeric_relative) / denominator,
        categorical_fraction=categorical / denominator,
        relative_numeric_delta_log=(
            0.0
            if not numeric_relative
            else fmean(math.log1p(value) for value in numeric_relative)
        ),
        numeric_sign=0.0 if not numeric_signs else fmean(numeric_signs),
        path_depth=0.0 if not depths else fmean(depths),
    )


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedFeatureProjectionRequest:
    allocation_request: SlateAllocationRequest
    structural_score_rows: tuple[StructuralPosteriorMemberScoreRow, ...]
    transition_receipts: tuple[PortableTransitionReceipt, ...]
    archive_context: CampaignPortfolioArchiveContextProjection
    frontier_target: CampaignPortfolioFrontierTarget
    campaign_generation: int
    lane_slot: int
    remaining_proposal_horizon: int

    def __post_init__(self) -> None:
        if type(self.allocation_request) is not SlateAllocationRequest:
            raise TypeError("allocation_request must be exact")
        self.allocation_request.revalidate()
        if type(self.structural_score_rows) is not tuple or any(
            type(value) is not StructuralPosteriorMemberScoreRow
            for value in self.structural_score_rows
        ):
            raise TypeError("structural_score_rows must contain exact rows")
        if type(self.transition_receipts) is not tuple or any(
            type(value) is not PortableTransitionReceipt
            for value in self.transition_receipts
        ):
            raise TypeError("transition_receipts must contain exact rows")
        for value in (*self.structural_score_rows, *self.transition_receipts):
            value.__post_init__()
        members = self.allocation_request.slate.members
        member_ids = {value.option_id for value in members}
        score_ids = tuple(value.option_id for value in self.structural_score_rows)
        transition_ids = tuple(value.option_id for value in self.transition_receipts)
        canonical = tuple(sorted(member_ids))
        if score_ids != canonical or transition_ids != canonical:
            raise ValueError("score and transition rows must exactly cover the slate")
        member_by_id = {value.option_id: value for value in members}
        for row in self.structural_score_rows:
            if (
                row.option_identity_sha256
                != member_by_id[row.option_id].option_identity_sha256
            ):
                raise ValueError("structural row names a foreign option")
        parent_sha256 = self.allocation_request.slate.parent_candidate_identity_sha256
        for row in self.transition_receipts:
            if (
                row.option_identity_sha256
                != member_by_id[row.option_id].option_identity_sha256
                or row.parent_configuration_sha256 != parent_sha256
            ):
                raise ValueError("transition row names a foreign parent or option")
        if type(self.archive_context) is not CampaignPortfolioArchiveContextProjection:
            raise TypeError("archive_context must be exact")
        self.archive_context.__post_init__()
        if type(self.frontier_target) is not CampaignPortfolioFrontierTarget:
            raise TypeError("frontier_target must be exact")
        self.frontier_target.__post_init__()
        if (
            self.archive_context.archive_utility_snapshot_sha256
            != self.frontier_target.archive_utility_snapshot_sha256
            or self.archive_context.parent_configuration_sha256 != parent_sha256
            or self.frontier_target.parent_configuration_sha256 != parent_sha256
        ):
            raise ValueError("archive context, target, and slate disagree")
        if type(self.campaign_generation) is not int or self.campaign_generation not in {
            1,
            3,
            5,
        }:
            raise ValueError("profile v1 supports campaign generations 1, 3, and 5")
        if type(self.lane_slot) is not int or self.lane_slot not in {0, 1}:
            raise ValueError("profile v1 supports exactly lane slots 0 and 1")
        expected_horizon = {1: 2, 3: 1, 5: 0}[self.campaign_generation]
        if (
            type(self.remaining_proposal_horizon) is not int
            or self.remaining_proposal_horizon != expected_horizon
        ):
            raise ValueError("remaining horizon differs from profile-v1 generation")
        if len(members) != 8:
            raise ValueError("profile v1 requires an exact K8 proposal slate")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "allocation_request_sha256": self.allocation_request.request_sha256,
            "structural_score_rows": [
                value.to_record() for value in self.structural_score_rows
            ],
            "transition_receipts": [
                value.to_record() for value in self.transition_receipts
            ],
            "archive_context": self.archive_context.to_record(),
            "frontier_target": self.frontier_target.to_record(),
            "campaign_generation": self.campaign_generation,
            "lane_slot": self.lane_slot,
            "remaining_proposal_horizon": self.remaining_proposal_horizon,
            "current_or_future_outcomes_consulted": False,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_PROJECTION_REQUEST_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedFeatureProjectionRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


def _weighted(
    rows: dict[str, object],
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


def _geometry(
    request: TargetConditionedFeatureProjectionRequest,
) -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
    float,
    tuple[float, ...],
    float,
    int,
]:
    archive = thaw_json(request.archive_context.payload)
    target = thaw_json(request.frontier_target.payload)
    if type(archive) is not dict or type(target) is not dict:
        raise TypeError("affine archive and target payloads must be objects")
    frame = archive.get("optimization_frame")
    archive_rows = archive.get("archive")
    parent = archive.get("parent")
    target_direction = target.get("target_direction")
    assigned_parent = target.get("assigned_parent")
    if any(
        type(value) is not dict
        for value in (frame, archive_rows, parent, target_direction, assigned_parent)
    ):
        raise ValueError("affine archive or target payload is incomplete")
    axes = frame.get("axes")
    directions = frame.get("reference_directions")
    points = archive_rows.get("normalized_points_decimal")
    parent_point_raw = parent.get("normalized_point_decimal")
    if (
        type(axes) is not list
        or len(axes) not in (2, 3)
        or type(directions) is not list
        or type(points) is not list
        or not points
        or type(parent_point_raw) is not list
        or len(parent_point_raw) != len(axes)
    ):
        raise ValueError("affine geometry has an unsupported shape")
    metric_ids = tuple(
        axis.get("metric_id") if type(axis) is dict else None for axis in axes
    )
    if any(type(value) is not str for value in metric_ids) or len(
        set(metric_ids)
    ) != len(metric_ids):
        raise ValueError("affine axes must use unique metric IDs")
    direction = next(
        (
            value
            for value in directions
            if type(value) is dict
            and value.get("direction_id") == request.frontier_target.direction_id
        ),
        None,
    )
    if type(direction) is not dict:
        raise ValueError("frontier target direction is absent from archive context")
    archive_weights_raw = direction.get("normalized_importance_decimal")
    target_weights_raw = target_direction.get("normalized_weights_decimal")
    if (
        type(archive_weights_raw) is not list
        or type(target_weights_raw) is not list
        or len(archive_weights_raw) != len(metric_ids)
        or len(target_weights_raw) != len(metric_ids)
    ):
        raise ValueError("frontier weights differ from affine dimension")
    archive_weights = tuple(
        _decimal(value, name="archive direction weight")
        for value in archive_weights_raw
    )
    maximum = max(archive_weights)
    if maximum <= 0.0 or any(value < 0.0 for value in archive_weights):
        raise ValueError("affine direction weights must be non-negative")
    expected_weights = tuple(value / maximum for value in archive_weights)
    weights = tuple(
        _decimal(value, name="target direction weight")
        for value in target_weights_raw
    )
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(weights, expected_weights, strict=True)
    ):
        raise ValueError("frontier target weights disagree with archive context")
    parent_point = tuple(
        _decimal(value, name="archive parent point") for value in parent_point_raw
    )
    assigned_point_raw = assigned_parent.get("normalized_point_decimal")
    if type(assigned_point_raw) is not list or len(assigned_point_raw) != len(
        parent_point
    ):
        raise ValueError("frontier target parent point is malformed")
    assigned_point = tuple(
        _decimal(value, name="target parent point") for value in assigned_point_raw
    )
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(parent_point, assigned_point, strict=True)
    ):
        raise ValueError("frontier target parent differs from archive context")
    opportunity = _decimal(
        target_direction.get("opportunity_from_ideal_decimal"),
        name="target opportunity",
    )
    parent_achievement = _decimal(
        assigned_parent.get("achievement_decimal"), name="parent achievement"
    )
    parent_regret = _decimal(
        assigned_parent.get("regret_above_archive_best_decimal"),
        name="parent regret",
    )
    base_hypervolume = _decimal(
        frame.get("base_hypervolume_decimal"), name="base hypervolume"
    )
    return (
        metric_ids,
        weights,
        tuple(float(value == 0.0) for value in weights),
        opportunity,
        parent_achievement,
        parent_regret,
        parent_point,
        base_hypervolume,
        len(points),
    )


@dataclass(frozen=True, slots=True)
class TargetConditionedPortableFeatureProjector:
    projector_id: ClassVar[str] = PROJECTOR_ID
    projector_version: ClassVar[int] = PROJECTOR_VERSION
    definition_sha256: ClassVar[str] = PROJECTOR_DEFINITION_SHA256
    feature_names: ClassVar[tuple[str, ...]] = FEATURE_NAMES

    def project(
        self, request: TargetConditionedFeatureProjectionRequest
    ) -> tuple[TargetConditionedMemberFeatures, ...]:
        if type(request) is not TargetConditionedFeatureProjectionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        (
            metric_ids,
            target_weights,
            off_target_weights,
            opportunity,
            parent_achievement,
            parent_regret,
            parent_point,
            base_hypervolume,
            archive_point_count,
        ) = _geometry(request)
        members = {
            value.option_id: value for value in request.allocation_request.slate.members
        }
        scores = {value.option_id: value for value in request.structural_score_rows}
        transitions = {
            value.option_id: value for value in request.transition_receipts
        }
        family_counts: dict[str, int] = {}
        for member in members.values():
            family_counts[member.family] = family_counts.get(member.family, 0) + 1
        desirability = tuple(1.0 - value for value in parent_point)
        rows = []
        for option_id in sorted(members):
            member = members[option_id]
            score = scores[option_id]
            metric_rows = {value.metric_id: value for value in score.metric_scores}
            if set(metric_rows) != set(metric_ids):
                raise ValueError("structural metrics differ from affine target axes")
            count = float(len(metric_rows))

            def weighted(weights: Sequence[float], value) -> float:
                return _weighted(metric_rows, metric_ids, weights, value)

            signed = lambda row: float(row.favorable_assertion) - float(  # noqa: E731
                row.adverse_assertion
            )
            features: dict[str, float] = {"bias": 1.0}
            for rank in range(1, 9):
                features[f"rank_{rank}"] = float(member.model_rank == rank)
            for generation in (1, 3, 5):
                features[f"generation_{generation}"] = float(
                    request.campaign_generation == generation
                )
            features["parent_slot_0"] = float(request.lane_slot == 0)
            features["parent_slot_1"] = float(request.lane_slot == 1)
            for role in ("exploit", "falsify", "coverage"):
                features[f"role_proposal_{role}"] = float(
                    member.role_proposal.value == role
                )
            family_frequency = family_counts[member.family] / 8.0
            features.update(
                {
                    "family_frequency": family_frequency,
                    "family_rarity": 1.0 / max(1.0, 8.0 * family_frequency),
                    "supporting_card_count": float(len(member.supporting_card_keys)),
                    "archive_novelty": (
                        member.structural_evidence.archive_novelty_score
                    ),
                    "structural_coverage_raw": (
                        member.structural_evidence.structural_coverage_score
                    ),
                    "calibrated_exploitation_score": (
                        score.calibrated_exploitation_score
                    ),
                    "calibrated_frontier_score": score.calibrated_frontier_score,
                    "raw_epistemic_score": score.raw_epistemic_score,
                    "structural_coverage_score": score.structural_coverage_score,
                    "epistemic_structural_score": score.epistemic_structural_score,
                    "confidence_low_fraction": sum(
                        value.confidence is ForecastConfidenceBin.LOW
                        for value in metric_rows.values()
                    )
                    / count,
                    "confidence_medium_fraction": sum(
                        value.confidence is ForecastConfidenceBin.MEDIUM
                        for value in metric_rows.values()
                    )
                    / count,
                    "confidence_high_fraction": sum(
                        value.confidence is ForecastConfidenceBin.HIGH
                        for value in metric_rows.values()
                    )
                    / count,
                    "favorable_fraction": sum(
                        value.favorable_assertion for value in metric_rows.values()
                    )
                    / count,
                    "adverse_fraction": sum(
                        value.adverse_assertion for value in metric_rows.values()
                    )
                    / count,
                    "abstention_fraction": sum(
                        value.explicit_abstention for value in metric_rows.values()
                    )
                    / count,
                    "posterior_correctness_mean": fmean(
                        value.calibration_cell.posterior_correctness
                        for value in metric_rows.values()
                    ),
                    "calibration_observation_count_log": math.log1p(
                        sum(
                            value.calibration_cell.observation_count
                            for value in metric_rows.values()
                        )
                    ),
                    **transitions[option_id].feature_map(),
                    "parent_desirability_mean": fmean(desirability),
                    "parent_desirability_min": min(desirability),
                    "parent_desirability_max": max(desirability),
                    "archive_base_hypervolume": base_hypervolume,
                    "archive_point_count_log": math.log1p(archive_point_count),
                    "target_favorable_fraction": weighted(
                        target_weights, lambda row: row.favorable_assertion
                    ),
                    "target_adverse_fraction": weighted(
                        target_weights, lambda row: row.adverse_assertion
                    ),
                    "target_abstention_fraction": weighted(
                        target_weights, lambda row: row.explicit_abstention
                    ),
                    "off_target_favorable_fraction": weighted(
                        off_target_weights, lambda row: row.favorable_assertion
                    ),
                    "off_target_adverse_fraction": weighted(
                        off_target_weights, lambda row: row.adverse_assertion
                    ),
                    "off_target_abstention_fraction": weighted(
                        off_target_weights, lambda row: row.explicit_abstention
                    ),
                    "target_declared_confidence": weighted(
                        target_weights, lambda row: _CONFIDENCE_VALUE[row.confidence]
                    ),
                    "target_posterior_correctness": weighted(
                        target_weights,
                        lambda row: row.calibration_cell.posterior_correctness,
                    ),
                    "target_signed_evidence": weighted(target_weights, signed),
                    "target_reliability_adjusted_evidence": weighted(
                        target_weights,
                        lambda row: signed(row)
                        * (2.0 * row.calibration_cell.posterior_correctness - 1.0),
                    ),
                    "target_opportunity_from_ideal": opportunity,
                    "target_parent_achievement": parent_achievement,
                    "target_parent_regret": parent_regret,
                    "target_active_axis_fraction": sum(
                        value > 0.0 for value in target_weights
                    )
                    / len(target_weights),
                    "target_zero_axis_fraction": sum(
                        value == 0.0 for value in target_weights
                    )
                    / len(target_weights),
                    "remaining_proposal_horizon": float(
                        request.remaining_proposal_horizon
                    ),
                    "remaining_proposal_horizon_fraction": (
                        request.remaining_proposal_horizon / 2.0
                    ),
                }
            )
            for rank in range(1, 8):
                features[f"target_opportunity_rank_{rank}"] = float(
                    request.frontier_target.opportunity_rank == rank
                )
            if set(features) != set(self.feature_names):
                raise RuntimeError("portable feature implementation drifted")
            values = tuple(float(features[name]) for name in self.feature_names)
            if any(not math.isfinite(value) for value in values):
                raise ValueError("portable features must all be finite")
            rows.append(
                TargetConditionedMemberFeatures(
                    option_id=option_id,
                    option_identity_sha256=member.option_identity_sha256,
                    feature_names=self.feature_names,
                    values=values,
                    projector_id=self.projector_id,
                    projector_version=self.projector_version,
                    projector_definition_sha256=self.definition_sha256,
                )
            )
        return tuple(rows)

    def to_record(self) -> dict[str, object]:
        return {
            "projector_id": self.projector_id,
            "projector_version": self.projector_version,
            "definition_sha256": self.definition_sha256,
            "feature_names": list(self.feature_names),
            "scale_qualification": {
                "proposal_slate_size": 8,
                "lane_slots": 2,
                "proposal_generations": [1, 3, 5],
                "maximum_remaining_horizon": 2,
            },
            "workload_model_provider_option_name_features": False,
            "current_or_future_outcomes_consulted": False,
        }


__all__ = [
    "FEATURE_NAMES",
    "PORTABLE_FEATURES",
    "PROJECTOR_DEFINITION_SHA256",
    "PROJECTOR_ID",
    "PROJECTOR_VERSION",
    "PortableTransitionReceipt",
    "TARGET_FEATURES",
    "TargetConditionedFeatureProjectionRequest",
    "TargetConditionedPortableFeatureProjector",
    "project_portable_transition",
]

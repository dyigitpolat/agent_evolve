"""Parent-reachable residual-cell targets for workload-neutral Pareto search.

The allocator reconstructs prior-only residual-frontier geometry, restricts it
to cells whose two anchors are exactly covered by the supplied parent lanes,
and chooses the largest remaining opportunity.  A globally matched parent
selector therefore retains the global-best behavior, while other generic
parent policies receive the best target they can actually act upon.  Each lane
gets the canonical affine direction that improves the coordinates in which its
anchor is worse than the cell midpoint, plus the shared aspiration and signed
transition in its authenticated payload.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import permutations

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.policies.selection.affine_frontier_target import (
    GloballyMatchedDirectionCoveredAffineFrontierTargetAllocator,
)
from agent_evolve.policies.selection.residual_frontier import (
    RESIDUAL_FRONTIER_POLICY_DEFINITION_SHA256,
    normalized_candidate_point,
    residual_frontier_geometry,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget


RESIDUAL_TARGET_ALLOCATOR_ID = "residual_hypervolume_frontier_target"
RESIDUAL_TARGET_ALLOCATOR_VERSION = 4
RESIDUAL_TARGET_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:residual-hypervolume-frontier-target:v4;"
    b"input=authenticated-affine-prior-archive-plus-two-evaluated-parent-lanes;"
    b"cell=largest-positive-pairwise-midpoint-hypervolume-residual-exactly-"
    b"covered-by-the-supplied-parent-lanes;"
    b"global-opportunity-rank=authenticated;"
    b"binding=exact-minimum-two-anchor-distance;"
    b"lane-direction=canonical-reference-direction-over-needed-improvement-axes;"
    b"payload=shared-aspiration-plus-signed-parent-transition-in-normalized-and-"
    b"objective-space-with-explicit-axis-orientation-and-target-realization-contract;"
    b"fallback=directional-affine-bootstrap-with-raw-target-when-no-positive-"
    b"covered-cell;"
    b"current-future-outcomes=false;workload-model-provider-action-fields=false"
).hexdigest()
DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_ID = (
    "directional_bootstrap_affine_frontier_target"
)
DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_VERSION = 1
DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:directional-bootstrap-affine-frontier-target:v1;"
    b"input=authenticated-affine-prior-archive-plus-evaluated-parent-lanes;"
    b"direction=globally-matched-phase-covered-affine-reference-direction;"
    b"active-axis-aspiration=ten-percent-parent-to-fixed-ideal-normalized;"
    b"inactive-axis-bound=fixed-reference;"
    b"payload=raw-parent-to-aspiration-axis-target-plus-identifiability-facts;"
    b"purpose=bootstrap-before-positive-covered-residual-cell-exists;"
    b"current-future-outcomes=false;workload-model-provider-action-fields=false"
).hexdigest()
_BOOTSTRAP_STRIDE = 0.10


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("residual target payload did not freeze to an object")
    return result


def _decimal(value: float) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("residual target values must be finite exact floats")
    return format(0.0 if value == 0.0 else value, ".17g")


def _achievement(point: tuple[float, ...], weights: tuple[float, ...]) -> float:
    active = tuple(
        (value, weight)
        for value, weight in zip(point, weights, strict=True)
        if weight > 0.0
    )
    if not active:
        raise ValueError("residual target direction must activate an axis")
    maximum = max(weight * value for value, weight in active)
    weighted_mean = sum(weight * value for value, weight in active) / sum(
        weight for _, weight in active
    )
    return maximum + 0.05 * weighted_mean


def _direction_for_transition(
    parent: tuple[float, ...],
    aspiration: tuple[float, ...],
) -> tuple[str, tuple[float, ...], tuple[int, ...]]:
    if len(parent) != len(aspiration) or len(parent) not in (2, 3):
        raise ValueError("residual transition requires equal 2-D or 3-D points")
    improvement_axes = tuple(
        index
        for index, (source, target) in enumerate(zip(parent, aspiration, strict=True))
        if target < source
    )
    if not improvement_axes:
        raise ValueError("positive residual anchor has no improving transition")
    weights = tuple(
        1.0 if index in improvement_axes else 0.0 for index in range(len(parent))
    )
    if len(improvement_axes) == 1:
        direction_id = f"axis_{improvement_axes[0] + 1}_extreme"
    elif len(improvement_axes) == 2 and len(parent) == 3:
        direction_id = (
            f"axes_{improvement_axes[0] + 1}_{improvement_axes[1] + 1}_tradeoff"
        )
    else:
        direction_id = "balanced_tradeoff"
    return direction_id, weights, improvement_axes


@dataclass(frozen=True, slots=True)
class DirectionalBootstrapAffineFrontierTargetAllocator(
    GloballyMatchedDirectionCoveredAffineFrontierTargetAllocator
):
    """Expose an identifiable raw target before a residual cell exists.

    A single nondominated point identifies affine directions but not a
    pairwise residual cell.  This allocator preserves direction coverage and
    exact lane matching, then turns each active direction into a bounded
    parent-to-ideal step.  Inactive axes may trade off only as far as the fixed
    reference.  The rule uses no workload semantics or outcome oracle.
    """

    allocator_id: str = DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_ID
    allocator_version: int = DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_VERSION
    definition_sha256: str = DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.allocator_id != DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_ID
            or self.allocator_version != DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_VERSION
            or self.definition_sha256
            != DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_DEFINITION_SHA256
        ):
            raise ValueError("directional bootstrap allocator identity drifted")

    def allocate(self, *, archive_utility, lanes):
        base_targets = (
            GloballyMatchedDirectionCoveredAffineFrontierTargetAllocator.allocate(
                self,
                archive_utility=archive_utility,
                lanes=lanes,
            )
        )
        geometry = residual_frontier_geometry(archive_utility)
        dimension = len(geometry.axes)
        results: list[CampaignPortfolioFrontierTarget] = []
        for base in base_targets:
            payload = thaw_json(base.payload)
            if type(payload) is not dict:
                raise TypeError("bootstrap base payload must be an object")
            direction = payload.get("target_direction")
            assigned_parent = payload.get("assigned_parent")
            if type(direction) is not dict or type(assigned_parent) is not dict:
                raise ValueError("bootstrap base target omitted affine evidence")
            raw_weights = direction.get("normalized_weights_decimal")
            raw_parent_point = assigned_parent.get("normalized_point_decimal")
            if (
                type(raw_weights) is not list
                or type(raw_parent_point) is not list
                or len(raw_weights) != dimension
                or len(raw_parent_point) != dimension
            ):
                raise ValueError("bootstrap affine evidence has invalid dimension")
            weights = tuple(float(value) for value in raw_weights)
            parent_point = tuple(float(value) for value in raw_parent_point)
            if (
                any(not math.isfinite(value) or value < 0.0 for value in weights)
                or max(weights) <= 0.0
                or any(not math.isfinite(value) for value in parent_point)
            ):
                raise ValueError("bootstrap affine evidence is invalid")

            aspiration_point = tuple(
                (
                    max(0.0, parent * (1.0 - _BOOTSTRAP_STRIDE))
                    if weight > 0.0 and parent > 0.0
                    else parent
                    if weight > 0.0
                    else 1.0
                )
                for parent, weight in zip(parent_point, weights, strict=True)
            )
            normalized_delta = tuple(
                target - parent
                for parent, target in zip(
                    parent_point,
                    aspiration_point,
                    strict=True,
                )
            )
            raw_parent = tuple(
                axis.denormalize(value)
                for axis, value in zip(
                    geometry.axes,
                    parent_point,
                    strict=True,
                )
            )
            raw_aspiration = tuple(
                axis.denormalize(value)
                for axis, value in zip(
                    geometry.axes,
                    aspiration_point,
                    strict=True,
                )
            )
            raw_delta = tuple(
                target - parent
                for parent, target in zip(
                    raw_parent,
                    raw_aspiration,
                    strict=True,
                )
            )
            improve_axes = tuple(
                index for index, value in enumerate(normalized_delta) if value < 0.0
            )
            tradeoff_axes = tuple(
                index for index, value in enumerate(normalized_delta) if value > 0.0
            )
            metric_ids = tuple(axis.metric_id for axis in geometry.axes)
            payload["schema_version"] = 2
            payload["frontier_bootstrap"] = {
                "target_kind": "directional_affine_bootstrap",
                "eligibility_reason": "no_positive_covered_residual_frontier_cell",
                "geometry_sha256": geometry.geometry_sha256,
                "active_axis_parent_to_ideal_stride_decimal": _decimal(
                    _BOOTSTRAP_STRIDE
                ),
                "inactive_axis_bound": "fixed_affine_reference",
                "normalized_aspiration_point_decimal": [
                    _decimal(value) for value in aspiration_point
                ],
                "residual_frontier_cell_asserted": False,
            }
            payload["objective_space_target"] = {
                "purpose": (
                    "make_directional_bootstrap_magnitude_explicit_without_"
                    "predicting_evaluator_outcomes"
                ),
                "axes": [
                    {
                        "metric_id": axis.metric_id,
                        "goal": axis.goal,
                        "ideal_decimal": _decimal(axis.ideal),
                        "reference_decimal": _decimal(axis.reference),
                        "parent_value_decimal": _decimal(parent_value),
                        "aspiration_value_decimal": _decimal(aspiration_value),
                        "signed_parent_to_aspiration_delta_decimal": _decimal(delta),
                        "improving_raw_delta_sign": (
                            "negative" if axis.goal == "min" else "positive"
                        ),
                    }
                    for axis, parent_value, aspiration_value, delta in zip(
                        geometry.axes,
                        raw_parent,
                        raw_aspiration,
                        raw_delta,
                        strict=True,
                    )
                ],
            }
            payload["lane_transition"] = {
                "normalized_signed_delta_decimal": [
                    _decimal(value) for value in normalized_delta
                ],
                "improve_metric_ids": [metric_ids[index] for index in improve_axes],
                "permitted_tradeoff_metric_ids": [
                    metric_ids[index] for index in tradeoff_axes
                ],
                "negative_delta_means_improvement": True,
            }
            instruction = payload.get("acquisition_instruction")
            if type(instruction) is not dict:
                raise ValueError("bootstrap base target omitted acquisition evidence")
            instruction.update(
                {
                    "objective": "close_the_directional_affine_bootstrap_target",
                    "target_realization_is_magnitude_sensitive": True,
                    "compare_action_magnitude_to_raw_parent_to_aspiration_deltas": (
                        True
                    ),
                    "direction_only_forecasts_are_insufficient": True,
                    "avoid_candidates_outside_the_fixed_reference_box": True,
                }
            )
            results.append(
                CampaignPortfolioFrontierTarget(
                    allocator_id=self.allocator_id,
                    allocator_version=self.allocator_version,
                    definition_sha256=self.definition_sha256,
                    archive_utility_snapshot_sha256=(
                        base.archive_utility_snapshot_sha256
                    ),
                    lane_id=base.lane_id,
                    parent_configuration_sha256=(base.parent_configuration_sha256),
                    direction_id=base.direction_id,
                    opportunity_rank=base.opportunity_rank,
                    payload=_object(payload),
                )
            )
        return tuple(sorted(results, key=lambda value: value.lane_id))


@dataclass(frozen=True, slots=True)
class ResidualHypervolumeFrontierTargetAllocator:
    """Bind two parent lanes to one maximum-residual frontier aspiration."""

    allocator_id: str = RESIDUAL_TARGET_ALLOCATOR_ID
    allocator_version: int = RESIDUAL_TARGET_ALLOCATOR_VERSION
    definition_sha256: str = RESIDUAL_TARGET_ALLOCATOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.allocator_id != RESIDUAL_TARGET_ALLOCATOR_ID
            or self.allocator_version != RESIDUAL_TARGET_ALLOCATOR_VERSION
            or self.definition_sha256 != RESIDUAL_TARGET_ALLOCATOR_DEFINITION_SHA256
        ):
            raise ValueError("residual frontier target allocator identity drifted")

    def allocate(self, *, archive_utility, lanes):
        from agent_evolve.application.agentic_evolution import EvolutionCandidate
        from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot

        self.__post_init__()
        if type(archive_utility) is not ArchiveUtilitySnapshot:
            raise TypeError("archive_utility must be an exact frozen snapshot")
        archive_utility.__post_init__()
        if type(lanes) is not tuple or len(lanes) != 2:
            raise ValueError("residual frontier targeting requires exactly two lanes")
        lane_ids = tuple(value[0] for value in lanes)
        if lane_ids != tuple(sorted(set(lane_ids))):
            raise ValueError("residual target lanes must be unique and canonical")
        if any(
            type(lane_id) is not str
            or not lane_id
            or type(parent) is not EvolutionCandidate
            for lane_id, parent in lanes
        ):
            raise TypeError("residual target lanes must contain exact candidates")
        for _, parent in lanes:
            parent.__post_init__()

        geometry = residual_frontier_geometry(archive_utility)
        if not geometry.cells:
            return DirectionalBootstrapAffineFrontierTargetAllocator().allocate(
                archive_utility=archive_utility,
                lanes=lanes,
            )

        parent_points = tuple(
            (lane_id, parent, normalized_candidate_point(geometry, parent))
            for lane_id, parent in lanes
        )
        tolerance = 64.0 * math.ulp(1.0)
        covered = []
        for opportunity_rank, candidate_cell in enumerate(geometry.cells, start=1):
            candidate_assignments = min(
                permutations(parent_points),
                key=lambda ordered: (
                    sum(
                        abs(value - target)
                        for (_, _, point), anchor in zip(
                            ordered,
                            candidate_cell.anchor_points,
                            strict=True,
                        )
                        for value, target in zip(point, anchor, strict=True)
                    ),
                    tuple(value[0] for value in ordered),
                ),
            )
            candidate_distance = sum(
                abs(value - target)
                for (_, _, point), anchor in zip(
                    candidate_assignments,
                    candidate_cell.anchor_points,
                    strict=True,
                )
                for value, target in zip(point, anchor, strict=True)
            )
            if candidate_distance <= tolerance:
                covered.append(
                    (
                        opportunity_rank,
                        candidate_cell,
                        candidate_assignments,
                        candidate_distance,
                    )
                )
        if not covered:
            return DirectionalBootstrapAffineFrontierTargetAllocator().allocate(
                archive_utility=archive_utility,
                lanes=lanes,
            )
        opportunity_rank, cell, assignments, binding_distance = covered[0]

        metric_ids = tuple(axis.metric_id for axis in geometry.axes)
        raw_aspiration = tuple(
            axis.denormalize(value)
            for axis, value in zip(
                geometry.axes,
                cell.aspiration_point,
                strict=True,
            )
        )
        results: list[CampaignPortfolioFrontierTarget] = []
        for lane_id, parent, point in assignments:
            direction_id, weights, improvement_axes = _direction_for_transition(
                point,
                cell.aspiration_point,
            )
            archive_best = min(
                _achievement(value, weights)
                for value in geometry.normalized_archive_points
            )
            parent_achievement = _achievement(point, weights)
            signed_delta = tuple(
                target - source
                for source, target in zip(
                    point,
                    cell.aspiration_point,
                    strict=True,
                )
            )
            tradeoff_axes = tuple(
                index for index, value in enumerate(signed_delta) if value > 0.0
            )
            raw_parent = tuple(
                axis.denormalize(value)
                for axis, value in zip(geometry.axes, point, strict=True)
            )
            raw_signed_delta = tuple(
                target - source
                for source, target in zip(
                    raw_parent,
                    raw_aspiration,
                    strict=True,
                )
            )
            results.append(
                CampaignPortfolioFrontierTarget(
                    allocator_id=self.allocator_id,
                    allocator_version=self.allocator_version,
                    definition_sha256=self.definition_sha256,
                    archive_utility_snapshot_sha256=archive_utility.snapshot_sha256,
                    lane_id=lane_id,
                    parent_configuration_sha256=(parent.occurrence.configuration_hash),
                    direction_id=direction_id,
                    opportunity_rank=opportunity_rank,
                    payload=_object(
                        {
                            "schema_version": 2,
                            "epistemic_cutoff": {
                                "generation": archive_utility.generation,
                                "archive_utility_snapshot_sha256": (
                                    archive_utility.snapshot_sha256
                                ),
                                "current_or_future_candidate_outcomes_consulted": False,
                            },
                            "normalized_orientation": ("lower_is_better_on_every_axis"),
                            "target_direction": {
                                "direction_id": direction_id,
                                "normalized_weights_decimal": [
                                    _decimal(value) for value in weights
                                ],
                                "opportunity_rank": opportunity_rank,
                                "archive_best_achievement_decimal": _decimal(
                                    archive_best
                                ),
                                "opportunity_from_ideal_decimal": _decimal(
                                    max(0.0, archive_best)
                                ),
                            },
                            "assigned_parent": {
                                "normalized_point_decimal": [
                                    _decimal(value) for value in point
                                ],
                                "achievement_decimal": _decimal(parent_achievement),
                                "regret_above_archive_best_decimal": _decimal(
                                    max(0.0, parent_achievement - archive_best)
                                ),
                            },
                            "residual_frontier_cell": {
                                "cell_sha256": cell.cell_sha256,
                                "geometry_sha256": geometry.geometry_sha256,
                                "policy_definition_sha256": (
                                    RESIDUAL_FRONTIER_POLICY_DEFINITION_SHA256
                                ),
                                "normalized_anchor_points_decimal": [
                                    [_decimal(value) for value in anchor]
                                    for anchor in cell.anchor_points
                                ],
                                "normalized_aspiration_point_decimal": [
                                    _decimal(value) for value in cell.aspiration_point
                                ],
                                "potential_hypervolume_gain_decimal": _decimal(
                                    cell.potential_hypervolume_gain
                                ),
                                "parent_anchor_binding_distance_decimal": _decimal(
                                    binding_distance
                                ),
                                "selection_scope": "supplied_parent_lane_anchors",
                                "global_opportunity_rank": opportunity_rank,
                            },
                            "objective_space_target": {
                                "purpose": (
                                    "make_target_realization_magnitude_explicit_"
                                    "without_predicting_evaluator_outcomes"
                                ),
                                "axes": [
                                    {
                                        "metric_id": axis.metric_id,
                                        "goal": axis.goal,
                                        "ideal_decimal": _decimal(axis.ideal),
                                        "reference_decimal": _decimal(axis.reference),
                                        "parent_value_decimal": _decimal(parent_value),
                                        "aspiration_value_decimal": _decimal(
                                            aspiration_value
                                        ),
                                        "signed_parent_to_aspiration_delta_decimal": (
                                            _decimal(raw_delta)
                                        ),
                                        "improving_raw_delta_sign": (
                                            "negative"
                                            if axis.goal == "min"
                                            else "positive"
                                        ),
                                    }
                                    for axis, parent_value, aspiration_value, raw_delta in zip(
                                        geometry.axes,
                                        raw_parent,
                                        raw_aspiration,
                                        raw_signed_delta,
                                        strict=True,
                                    )
                                ],
                            },
                            "lane_transition": {
                                "normalized_signed_delta_decimal": [
                                    _decimal(value) for value in signed_delta
                                ],
                                "improve_metric_ids": [
                                    metric_ids[index] for index in improvement_axes
                                ],
                                "permitted_tradeoff_metric_ids": [
                                    metric_ids[index] for index in tradeoff_axes
                                ],
                                "negative_delta_means_improvement": True,
                            },
                            "acquisition_instruction": {
                                "objective": (
                                    "approach_the_shared_residual_frontier_aspiration"
                                ),
                                "maximize_actual_fixed_reference_hypervolume_gain": True,
                                "avoid_candidates_dominated_by_the_prior_archive": True,
                                "tradeoffs_can_be_frontier_improving": True,
                                "simultaneous_improvement_on_every_axis_required": False,
                                "propose_semantically_distinct_finite_actions": True,
                                "target_realization_is_magnitude_sensitive": True,
                                "reserve_multiple_semantically_distinct_"
                                "target_bridge_hypotheses": True,
                                "compare_action_magnitude_to_raw_parent_to_"
                                "aspiration_deltas": True,
                                "direction_only_forecasts_are_insufficient": True,
                                "evaluator_outcomes_remain_unknown": True,
                            },
                            "achievement_scalar": {
                                "kind": "augmented_weighted_chebyshev",
                                "rho_decimal": "0.050000000000000003",
                                "weights_rescaled_to_max_one": True,
                            },
                            "workload_identifiers_consulted": False,
                            "model_or_provider_fields_consulted": False,
                        }
                    ),
                )
            )
        return tuple(sorted(results, key=lambda value: value.lane_id))


__all__ = [
    "DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_DEFINITION_SHA256",
    "DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_ID",
    "DIRECTIONAL_BOOTSTRAP_TARGET_ALLOCATOR_VERSION",
    "DirectionalBootstrapAffineFrontierTargetAllocator",
    "RESIDUAL_TARGET_ALLOCATOR_DEFINITION_SHA256",
    "RESIDUAL_TARGET_ALLOCATOR_ID",
    "RESIDUAL_TARGET_ALLOCATOR_VERSION",
    "ResidualHypervolumeFrontierTargetAllocator",
]

"""Coordinated, workload-neutral target allocation over affine Pareto geometry.

The policy is intentionally prospective.  It sees only the frozen archive and
already-evaluated parents.  It converts the reference directions exposed by the
authenticated affine frontier projector into distinct lane-level opportunities
using an augmented weighted-Chebyshev achievement scalar.  Evaluator outcomes
from the current or future wave, model/provider metadata, action names, and
workload identifiers are outside the interface.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import permutations

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.policies.selection.affine_frontier_context import (
    AuthenticatedAffineFrontierContextProjector,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget


ALLOCATOR_ID = "affine_frontier_opportunity_target"
ALLOCATOR_VERSION = 1
ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:affine-frontier-opportunity-target:v1;"
    b"input=frozen-affine-archive-cutoff,selected-evaluated-parent-lanes;"
    b"geometry=authenticated-normalized-lower-is-better-reference-directions;"
    b"achievement=normalized-weight-augmented-chebyshev-rho-0.05;"
    b"opportunity=nonnegative-distance-from-normalized-ideal;"
    b"direction-selection=descending-opportunity-distinct-before-cycle;"
    b"lane-matching=greedy-min-parent-achievement;"
    b"current-future-outcomes=false;workload-model-provider-action-fields=false"
).hexdigest()
DIRECTION_COVERED_ALLOCATOR_ID = "direction_covered_affine_frontier_target"
DIRECTION_COVERED_ALLOCATOR_VERSION = 2
DIRECTION_COVERED_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:direction-covered-affine-frontier-target:v2;"
    b"input=frozen-affine-archive-cutoff,selected-evaluated-parent-lanes;"
    b"geometry=authenticated-normalized-lower-is-better-reference-directions;"
    b"achievement=normalized-weight-augmented-chebyshev-rho-0.05;"
    b"opportunity=nonnegative-distance-from-normalized-ideal;"
    b"direction-selection=canonical-phase-stratified-coverage-before-repeat;"
    b"coverage-phase=portfolio-wave-index-derived-from-generation;"
    b"lane-matching=greedy-min-parent-achievement;"
    b"current-future-outcomes=false;workload-model-provider-action-fields=false;"
    b"legacy-opportunity-target-unchanged=true"
).hexdigest()
GLOBALLY_MATCHED_ALLOCATOR_ID = "globally_matched_direction_covered_frontier_target"
GLOBALLY_MATCHED_ALLOCATOR_VERSION = 3
GLOBALLY_MATCHED_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:globally-matched-direction-covered-frontier-target:v3;"
    b"input=frozen-affine-archive-cutoff,selected-evaluated-parent-lanes;"
    b"geometry=authenticated-normalized-lower-is-better-reference-directions;"
    b"achievement=normalized-weight-augmented-chebyshev-rho-0.05;"
    b"opportunity=nonnegative-distance-from-normalized-ideal;"
    b"direction-selection=canonical-phase-stratified-coverage-before-repeat;"
    b"coverage-phase=portfolio-wave-index-derived-from-generation;"
    b"lane-matching=exact-global-minimum-total-parent-achievement;"
    b"ties=canonical-lane-permutation;"
    b"current-future-outcomes=false;workload-model-provider-action-fields=false;"
    b"legacy-allocators-unchanged=true"
).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("frontier target payload did not freeze to an object")
    return result


def _finite_decimal(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be decimal text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _decimal(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("frontier target cannot render a non-finite value")
    normalized = 0.0 if value == 0.0 else value
    return format(normalized, ".17g")


def _achievement(point: tuple[float, ...], weights: tuple[float, ...]) -> float:
    active = tuple(
        (value, weight)
        for value, weight in zip(point, weights, strict=True)
        if weight > 0
    )
    if not active:
        raise ValueError("frontier target direction must activate an axis")
    maximum = max(weight * value for value, weight in active)
    weighted_mean = sum(weight * value for value, weight in active) / sum(
        weight for _, weight in active
    )
    return maximum + 0.05 * weighted_mean


@dataclass(frozen=True, slots=True)
class AuthenticatedAffineFrontierTargetAllocator:
    """Coordinate lane targets without benchmark-specific knowledge."""

    allocator_id: str = ALLOCATOR_ID
    allocator_version: int = ALLOCATOR_VERSION
    definition_sha256: str = ALLOCATOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.allocator_id != ALLOCATOR_ID
            or self.allocator_version != ALLOCATOR_VERSION
            or self.definition_sha256 != ALLOCATOR_DEFINITION_SHA256
        ):
            raise ValueError("affine frontier target allocator identity drifted")

    def _select_directions(
        self,
        directions: tuple[dict[str, object], ...],
        *,
        lane_count: int,
        generation: int,
    ) -> tuple[dict[str, object], ...]:
        del generation
        return tuple(
            directions[index % len(directions)] for index in range(lane_count)
        )

    def _match_directions_to_parents(
        self,
        *,
        selected: tuple[dict[str, object], ...],
        parents: dict[str, tuple[object, tuple[float, ...]]],
    ) -> tuple[tuple[str, object, tuple[float, ...], dict[str, object]], ...]:
        """Preserve the version-1 greedy matching law for replay stability."""

        assignments: list[
            tuple[str, object, tuple[float, ...], dict[str, object]]
        ] = []
        remaining = dict(parents)
        for direction in selected:
            weights = direction["weights"]
            assert type(weights) is tuple
            lane_id = min(
                remaining,
                key=lambda value: (
                    _achievement(remaining[value][1], weights),
                    value,
                ),
            )
            parent, point = remaining.pop(lane_id)
            assignments.append((lane_id, parent, point, direction))
        return tuple(assignments)

    def allocate(self, *, archive_utility, lanes):
        from agent_evolve.application.agentic_evolution import EvolutionCandidate
        from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot

        self.__post_init__()
        if type(archive_utility) is not ArchiveUtilitySnapshot:
            raise TypeError("archive_utility must be an exact frozen snapshot")
        ArchiveUtilitySnapshot.__post_init__(archive_utility)
        if type(lanes) is not tuple or not lanes:
            raise ValueError("lanes must be a non-empty exact tuple")
        for lane_id, parent in lanes:
            if type(lane_id) is not str or not lane_id:
                raise ValueError("lane IDs must be non-empty strings")
            if type(parent) is not EvolutionCandidate:
                raise TypeError("lane parents must be exact EvolutionCandidate values")
            EvolutionCandidate.__post_init__(parent)
        if tuple(value[0] for value in lanes) != tuple(
            sorted({value[0] for value in lanes})
        ):
            raise ValueError("lanes must be unique and canonical")

        projector = AuthenticatedAffineFrontierContextProjector()
        projected = tuple(
            (
                lane_id,
                parent,
                projector.project(archive_utility=archive_utility, parent=parent),
            )
            for lane_id, parent in lanes
        )
        payloads = tuple(thaw_json(value.payload) for _, _, value in projected)
        if any(type(value) is not dict for value in payloads):
            raise TypeError("affine frontier projection payload must be an object")
        first = payloads[0]
        frame = first.get("optimization_frame")
        archive = first.get("archive")
        if type(frame) is not dict or type(archive) is not dict:
            raise ValueError("affine frontier projection omitted its geometry")
        raw_points = archive.get("normalized_points_decimal")
        raw_directions = frame.get("reference_directions")
        if type(raw_points) is not list or not raw_points:
            raise ValueError("affine frontier target requires archive points")
        if type(raw_directions) is not list or not raw_directions:
            raise ValueError("affine frontier target requires reference directions")
        dimension = frame.get("dimension")
        if type(dimension) is not int or dimension not in (2, 3):
            raise ValueError("affine frontier target supports exact 2-D or 3-D")
        points = tuple(
            tuple(
                _finite_decimal(cell, name=f"archive[{row_index}][{axis_index}]")
                for axis_index, cell in enumerate(row)
            )
            for row_index, row in enumerate(raw_points)
            if type(row) is list and len(row) == dimension
        )
        if len(points) != len(raw_points):
            raise ValueError("affine archive point dimension is malformed")

        directions: list[dict[str, object]] = []
        for raw in raw_directions:
            if type(raw) is not dict or type(raw.get("direction_id")) is not str:
                raise ValueError("affine reference direction is malformed")
            raw_weights = raw.get("normalized_importance_decimal")
            if type(raw_weights) is not list or len(raw_weights) != dimension:
                raise ValueError("affine reference-direction dimension is malformed")
            weights = tuple(
                _finite_decimal(value, name="reference_direction_weight")
                for value in raw_weights
            )
            if any(value < 0 for value in weights) or max(weights) <= 0:
                raise ValueError("reference-direction weights must be nonnegative")
            maximum = max(weights)
            comparable = tuple(value / maximum for value in weights)
            archive_best = min(_achievement(point, comparable) for point in points)
            directions.append(
                {
                    "direction_id": raw["direction_id"],
                    "weights": comparable,
                    "archive_best": archive_best,
                    "opportunity": max(0.0, archive_best),
                }
            )
        directions.sort(
            key=lambda value: (
                -float(value["opportunity"]),
                str(value["direction_id"]),
            )
        )
        selected = self._select_directions(
            tuple(directions),
            lane_count=len(lanes),
            generation=archive_utility.generation,
        )

        parents: dict[str, tuple[object, tuple[float, ...]]] = {}
        shared_geometry_sha256 = None
        for (lane_id, parent, projection), payload in zip(
            projected, payloads, strict=True
        ):
            assert type(payload) is dict
            if (
                payload.get("optimization_frame") != frame
                or payload.get("archive") != archive
            ):
                raise ValueError(
                    "lane frontier projections disagree on archive geometry"
                )
            parent_record = payload.get("parent")
            if type(parent_record) is not dict:
                raise ValueError("affine frontier projection omitted its parent")
            raw_parent = parent_record.get("normalized_point_decimal")
            if type(raw_parent) is not list or len(raw_parent) != dimension:
                raise ValueError("affine parent point dimension is malformed")
            parents[lane_id] = (
                parent,
                tuple(
                    _finite_decimal(value, name="parent_normalized_point")
                    for value in raw_parent
                ),
            )
            geometry_sha256 = projection.archive_utility_snapshot_sha256
            if shared_geometry_sha256 is None:
                shared_geometry_sha256 = geometry_sha256
            elif shared_geometry_sha256 != geometry_sha256:
                raise ValueError("lane frontier projections use different cutoffs")

        assignments = self._match_directions_to_parents(
            selected=selected,
            parents=parents,
        )

        rank_by_direction = {
            str(value["direction_id"]): index
            for index, value in enumerate(directions, start=1)
        }
        results = []
        for lane_id, parent, point, direction in assignments:
            weights = direction["weights"]
            assert type(weights) is tuple
            parent_achievement = _achievement(point, weights)
            archive_best = float(direction["archive_best"])
            direction_id = str(direction["direction_id"])
            rank = rank_by_direction[direction_id]
            results.append(
                CampaignPortfolioFrontierTarget(
                    allocator_id=self.allocator_id,
                    allocator_version=self.allocator_version,
                    definition_sha256=self.definition_sha256,
                    archive_utility_snapshot_sha256=(archive_utility.snapshot_sha256),
                    lane_id=lane_id,
                    parent_configuration_sha256=(parent.occurrence.configuration_hash),
                    direction_id=direction_id,
                    opportunity_rank=rank,
                    payload=_object(
                        {
                            "schema_version": 1,
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
                                "opportunity_rank": rank,
                                "archive_best_achievement_decimal": _decimal(
                                    archive_best
                                ),
                                "opportunity_from_ideal_decimal": _decimal(
                                    float(direction["opportunity"])
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
                            "acquisition_instruction": {
                                "objective": (
                                    "reduce_the_assigned_augmented_chebyshev_achievement"
                                ),
                                "tradeoffs_can_be_frontier_improving": True,
                                "simultaneous_improvement_on_every_axis_required": False,
                                "propose_semantically_distinct_finite_actions": True,
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


@dataclass(frozen=True, slots=True)
class DirectionCoveredAffineFrontierTargetAllocator(
    AuthenticatedAffineFrontierTargetAllocator
):
    """Rotate canonical directions across waves before any direction repeats."""

    allocator_id: str = DIRECTION_COVERED_ALLOCATOR_ID
    allocator_version: int = DIRECTION_COVERED_ALLOCATOR_VERSION
    definition_sha256: str = DIRECTION_COVERED_ALLOCATOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.allocator_id != DIRECTION_COVERED_ALLOCATOR_ID
            or self.allocator_version != DIRECTION_COVERED_ALLOCATOR_VERSION
            or self.definition_sha256 != DIRECTION_COVERED_ALLOCATOR_DEFINITION_SHA256
        ):
            raise ValueError("direction-covered frontier allocator identity drifted")

    def _select_directions(
        self,
        directions: tuple[dict[str, object], ...],
        *,
        lane_count: int,
        generation: int,
    ) -> tuple[dict[str, object], ...]:
        canonical = tuple(
            sorted(directions, key=lambda value: str(value["direction_id"]))
        )
        portfolio_wave_index = max(0, (generation - 1) // 2)
        offset = (portfolio_wave_index * lane_count) % len(canonical)
        return tuple(
            canonical[(offset + index) % len(canonical)]
            for index in range(lane_count)
        )


@dataclass(frozen=True, slots=True)
class GloballyMatchedDirectionCoveredAffineFrontierTargetAllocator(
    DirectionCoveredAffineFrontierTargetAllocator
):
    """Cover directions and solve their lane binding as one exact assignment."""

    allocator_id: str = GLOBALLY_MATCHED_ALLOCATOR_ID
    allocator_version: int = GLOBALLY_MATCHED_ALLOCATOR_VERSION
    definition_sha256: str = GLOBALLY_MATCHED_ALLOCATOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if (
            self.allocator_id != GLOBALLY_MATCHED_ALLOCATOR_ID
            or self.allocator_version != GLOBALLY_MATCHED_ALLOCATOR_VERSION
            or self.definition_sha256 != GLOBALLY_MATCHED_ALLOCATOR_DEFINITION_SHA256
        ):
            raise ValueError("globally matched frontier allocator identity drifted")

    def _match_directions_to_parents(
        self,
        *,
        selected: tuple[dict[str, object], ...],
        parents: dict[str, tuple[object, tuple[float, ...]]],
    ) -> tuple[tuple[str, object, tuple[float, ...], dict[str, object]], ...]:
        if len(selected) != len(parents):
            raise ValueError("global frontier matching requires equal cardinality")
        lane_ids = tuple(sorted(parents))
        # Campaign lane counts are intentionally small.  Exact enumeration is
        # preferable here: it is dependency-free, deterministic, and leaves an
        # auditable optimum rather than another greedy repair boundary.
        best = min(
            permutations(lane_ids),
            key=lambda assignment: (
                sum(
                    _achievement(parents[lane_id][1], direction["weights"])
                    for lane_id, direction in zip(
                        assignment,
                        selected,
                        strict=True,
                    )
                ),
                assignment,
            ),
        )
        return tuple(
            (
                lane_id,
                parents[lane_id][0],
                parents[lane_id][1],
                direction,
            )
            for lane_id, direction in zip(best, selected, strict=True)
        )


__all__ = [
    "ALLOCATOR_DEFINITION_SHA256",
    "ALLOCATOR_ID",
    "ALLOCATOR_VERSION",
    "AuthenticatedAffineFrontierTargetAllocator",
    "DIRECTION_COVERED_ALLOCATOR_DEFINITION_SHA256",
    "DIRECTION_COVERED_ALLOCATOR_ID",
    "DIRECTION_COVERED_ALLOCATOR_VERSION",
    "DirectionCoveredAffineFrontierTargetAllocator",
    "GLOBALLY_MATCHED_ALLOCATOR_DEFINITION_SHA256",
    "GLOBALLY_MATCHED_ALLOCATOR_ID",
    "GLOBALLY_MATCHED_ALLOCATOR_VERSION",
    "GloballyMatchedDirectionCoveredAffineFrontierTargetAllocator",
]

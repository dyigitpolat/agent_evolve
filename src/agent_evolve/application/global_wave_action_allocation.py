"""Wave-level allocation across concurrent evolutionary parent lanes.

Lane-local selection duplicates protected exploration roles and can spend the
same wave on structurally equivalent actions from different parents.  This
module coordinates only after every lane has independently produced its sealed
finite action set and consequence forecasts.  It owns no workload adapter and
never sees evaluator outcomes from the current or a future wave.

The policy gives a nonterminal wave one global bridge and one global probe,
uses the remaining capacity for exploitation, preserves every lane's hard
portfolio contract, and compares both lane orderings with and without
cross-lane redundancy avoidance.  Existing deterministic allocators remain the
hard-feasibility authority for every candidate slate.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from agent_evolve.application.action_role_value import (
    RoleFactorizedActionPortfolioUtility,
)
from agent_evolve.application.action_structural_signature import (
    ActionStructuralSignature,
    action_structural_signatures_by_option,
)
from agent_evolve.application.contextual_action_allocation import (
    contextual_action_arm_count_constraints,
    contextual_allocation_realization,
)
from agent_evolve.application.target_conditioned_action_forecast import (
    TargetConditionedActionForecastPlan,
    allocate_target_conditioned_actions,
    build_target_conditioned_action_utility,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.ports.action_allocation import ActionAllocationResult
from agent_evolve.ports.action_forecast import ResolvedActionForecastBatch
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationContract,
)
from agent_evolve.ports.portfolio_selection import PortfolioSelectionRequest

GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_ID = "global_role_balanced_wave"
GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_VERSION = 7
GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:global-role-balanced-wave:v7;"
    b"scope=concurrent-parent-wave;"
    b"nonterminal-roles=one-bridge-one-probe-remainder-exploit;"
    b"protected-role-placement=exhaustive-small-finite-set;"
    b"memory-search=bounded-primary-then-exhaustive-feasibility-fallback;"
    b"eligible-universe=plan-authenticated-forecast-contract;"
    b"hard-feasibility=existing-lane-allocator-plus-contextual-arm-marginals;"
    b"contextual-count-recourse=minimum-l1-materialized-feasible-projection-"
    b"only-after-exact-search-exhaustion;"
    b"cross-lane-diversity=family-path-signature-plus-p50-objective-cell;"
    b"search=bounded-memory-pairings-times-role-placements-times-lane-rotations;"
    b"current-future-outcomes=false;workload-model-provider-branches=false"
).hexdigest()
_RECEIPT_DOMAIN = b"agent-evolve:global-wave-action-allocation:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("global allocation audit did not freeze to an object")
    return frozen


MemoryAssignment = tuple[tuple[str, str], ...]
RoleSlots = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class GlobalWaveActionAllocationLane:
    """One fully forecast lane submitted to the wave-level trusted policy."""

    generation: int
    request: PortfolioSelectionRequest
    plan: TargetConditionedActionForecastPlan
    forecasts: ResolvedActionForecastBatch
    utility_mode: str
    target_kind: str
    memory_assignments: tuple[MemoryAssignment, ...]
    risk_aversion: float
    diversity_weight: float
    beam_width: int
    contextual_allocation: ContextualPortfolioAllocationContract | None = None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if type(self.request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        self.request.__post_init__()
        if type(self.plan) is not TargetConditionedActionForecastPlan:
            raise TypeError("plan must be exact")
        self.plan.__post_init__()
        if type(self.forecasts) is not ResolvedActionForecastBatch:
            raise TypeError("forecasts must be exact")
        self.plan.assess(self.forecasts)
        if type(self.utility_mode) is not str or not self.utility_mode:
            raise ValueError("utility_mode must be non-empty")
        if type(self.target_kind) is not str or not self.target_kind:
            raise ValueError("target_kind must be non-empty")
        if type(self.memory_assignments) is not tuple or not self.memory_assignments:
            raise ValueError("memory_assignments must be a non-empty tuple")
        for assignment in self.memory_assignments:
            if (
                type(assignment) is not tuple
                or assignment != tuple(sorted(set(assignment)))
                or any(
                    type(item) is not tuple
                    or len(item) != 2
                    or any(type(value) is not str or not value for value in item)
                    for item in assignment
                )
            ):
                raise ValueError("memory assignments must be canonical string pairs")
        for name in ("risk_aversion", "diversity_weight"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.beam_width) is not int or self.beam_width <= 0:
            raise ValueError("beam_width must be a positive exact integer")
        if self.contextual_allocation is not None:
            if type(self.contextual_allocation) is not (
                ContextualPortfolioAllocationContract
            ):
                raise TypeError("contextual_allocation must be exact or None")
            self.contextual_allocation.__post_init__()
            if (
                self.contextual_allocation.campaign_generation != self.generation
                or self.contextual_allocation.slice_id
                != self.plan.campaign_target.lane_id
                or self.contextual_allocation.evaluation_slots
                != self.request.portfolio_size
            ):
                raise ValueError("contextual allocation differs from its wave lane")

    @property
    def lane_id(self) -> str:
        # The exact frozen lane is validated at construction and again at the
        # public policy/coordinator boundaries.  Revalidating its complete
        # forecast and finite contract on every identity lookup made the joint
        # counterfactual search repeatedly hash the same immutable graph.
        return self.plan.campaign_target.lane_id

    @property
    def wave_key(self) -> tuple[int, str]:
        # See ``lane_id``: this is a pure projection of already validated,
        # immutable fields, not a trust boundary.
        return (
            self.generation,
            self.plan.campaign_target.archive_utility_snapshot_sha256,
        )


@dataclass(frozen=True, slots=True)
class GlobalWaveActionAllocationLaneResult:
    """One lane projection of a jointly selected wave portfolio."""

    allocation: ActionAllocationResult
    memory_assignment: MemoryAssignment
    role_slots: RoleSlots | None
    eligible_option_ids: tuple[str, ...] | None
    global_receipt_sha256: str
    audit: FrozenJsonObject

    def __post_init__(self) -> None:
        if type(self.allocation) is not ActionAllocationResult:
            raise TypeError("allocation must be exact")
        self.allocation.__post_init__()
        if type(self.memory_assignment) is not tuple:
            raise TypeError("memory_assignment must be exact")
        if self.role_slots is not None and (
            type(self.role_slots) is not tuple
            or len(self.role_slots) != 3
            or any(type(value) is not int or value < 0 for value in self.role_slots)
        ):
            raise ValueError("role_slots must be non-negative exact integers")
        if self.eligible_option_ids is not None and (
            type(self.eligible_option_ids) is not tuple
            or self.eligible_option_ids != tuple(sorted(set(self.eligible_option_ids)))
        ):
            raise ValueError("eligible_option_ids must be canonical or None")
        if (
            type(self.global_receipt_sha256) is not str
            or len(self.global_receipt_sha256) != 64
        ):
            raise ValueError("global_receipt_sha256 must be a SHA-256 digest")
        if (
            type(self.audit) is not FrozenJsonObject
            or freeze_json(self.audit) is not self.audit
        ):
            raise TypeError("audit must be an exact frozen object")


@runtime_checkable
class GlobalWaveActionAllocationPolicy(Protocol):
    def allocate(
        self,
        lanes: tuple[GlobalWaveActionAllocationLane, ...],
    ) -> dict[str, GlobalWaveActionAllocationLaneResult]: ...


@runtime_checkable
class GlobalWaveActionAllocationCoordinator(Protocol):
    async def allocate(
        self,
        lane: GlobalWaveActionAllocationLane,
    ) -> GlobalWaveActionAllocationLaneResult: ...


@dataclass(frozen=True, slots=True)
class _Trial:
    allocations: tuple[tuple[str, ActionAllocationResult], ...]
    memory_assignments: tuple[tuple[str, MemoryAssignment], ...]
    role_slots: tuple[tuple[str, RoleSlots | None], ...]
    eligible_option_ids: tuple[tuple[str, tuple[str, ...] | None], ...]
    lane_order: tuple[str, ...]
    avoided_redundancy: bool
    structural_collisions: int
    near_outcome_collisions: int
    local_utility: float
    total_score: float
    contextual_source_l1_deviation: int = 0
    contextual_operator_l1_deviation: int = 0
    contextual_projection_count: int = 0

    @property
    def tie_break(self) -> tuple[str, ...]:
        return tuple(
            allocation.decision.receipt_sha256 for _, allocation in self.allocations
        )

    @property
    def contextual_l1_deviation(self) -> int:
        return (
            self.contextual_source_l1_deviation
            + self.contextual_operator_l1_deviation
        )


@dataclass(frozen=True, slots=True)
class _LaneAllocationAttempt:
    allocation: ActionAllocationResult
    source_l1_deviation: int
    operator_l1_deviation: int
    contextual_counts_projected: bool


def _memory_map(assignment: MemoryAssignment) -> dict[str, str]:
    return dict(assignment)


def _required_option_ids(
    lane: GlobalWaveActionAllocationLane,
    assignment: MemoryAssignment,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *lane.request.candidate_pool_required_option_ids,
                *(option_id for option_id, _ in assignment),
            }
        )
    )


def _p50_cells(
    lane: GlobalWaveActionAllocationLane,
) -> dict[str, tuple[tuple[str, float], ...]]:
    aliases = {
        value.target_metric_id: value.forecast_metric_id for value in lane.plan.aliases
    }
    result: dict[str, tuple[tuple[str, float], ...]] = {}
    for forecast in lane.forecasts.forecasts:
        metrics = {value.metric_id: value for value in forecast.metric_forecasts}
        result[forecast.option_id] = tuple(
            (
                axis.metric_id,
                axis.normalize(
                    axis.parent_value + metrics[aliases[axis.metric_id]].p50_delta
                ),
            )
            for axis in lane.plan.objective_target.axes
        )
    return result


def _cell_distance(
    left: tuple[tuple[str, float], ...],
    right: tuple[tuple[str, float], ...],
) -> float | None:
    if tuple(value[0] for value in left) != tuple(value[0] for value in right):
        return None
    return sum(abs(a[1] - b[1]) for a, b in zip(left, right, strict=True)) / len(left)


def _role_budgets(
    lanes: tuple[GlobalWaveActionAllocationLane, ...],
) -> tuple[dict[str, RoleSlots | None], ...]:
    if any(value.utility_mode != "role_factorized" for value in lanes):
        return ({value.lane_id: None for value in lanes},)
    results: list[dict[str, RoleSlots | None]] = []
    for bridge_lane in lanes:
        for probe_lane in lanes:
            budget: dict[str, RoleSlots | None] = {}
            feasible = True
            for lane in lanes:
                bridge = int(lane.lane_id == bridge_lane.lane_id)
                probe = int(lane.lane_id == probe_lane.lane_id)
                exploit = lane.request.portfolio_size - bridge - probe
                if exploit <= 0:
                    feasible = False
                    break
                budget[lane.lane_id] = (exploit, bridge, probe)
            if feasible and budget not in results:
                results.append(budget)
    if not results:
        raise ValueError("wave capacity cannot support one global bridge and probe")

    def screen(budget: dict[str, RoleSlots | None]) -> float:
        total = 0.0
        for lane in lanes:
            role_slots = budget[lane.lane_id]
            assert role_slots is not None
            binding = build_target_conditioned_action_utility(
                plan=lane.plan,
                forecasts=lane.forecasts,
                portfolio_size=lane.request.portfolio_size,
                utility_mode="role_factorized",
                role_slots=role_slots,
            )
            utility = binding.utility
            if type(utility) is not RoleFactorizedActionPortfolioUtility:
                raise AssertionError("role utility factory returned a foreign type")
            exploit_slots, bridge_slots, probe_slots = role_slots
            bridge_choices: tuple[tuple[str | None, float], ...] = (
                ((None, 0.0),)
                if bridge_slots == 0
                else tuple(
                    (row.option_id, row.bridge_rank_scores[1])
                    for row in utility.score_rows
                )
            )
            probe_choices: tuple[tuple[str | None, float], ...] = (
                ((None, 0.0),)
                if probe_slots == 0
                else tuple(
                    (row.option_id, row.probe_rank_score) for row in utility.score_rows
                )
            )
            rows = {value.option_id: value for value in utility.score_rows}
            best: float | None = None
            for bridge_id, bridge_score in bridge_choices:
                for probe_id, probe_score in probe_choices:
                    if bridge_id is not None and bridge_id == probe_id:
                        continue
                    excluded = {value for value in (bridge_id, probe_id) if value}
                    exploit = sorted(
                        (
                            value.exploit_rank_scores[1]
                            for value in rows.values()
                            if value.option_id not in excluded
                        ),
                        reverse=True,
                    )[:exploit_slots]
                    if len(exploit) != exploit_slots:
                        continue
                    value = bridge_score + probe_score + sum(exploit)
                    if best is None or value > best:
                        best = value
            if best is None:
                raise ValueError("role budget has no injective action assignment")
            total += best
        return total

    ranked = sorted(
        results,
        key=lambda value: (
            -screen(value),
            tuple((lane_id, value[lane_id]) for lane_id in sorted(value)),
        ),
    )
    # Screening orders the tiny protected-role placement set; it must not
    # discard alternatives before memory, patch, and family feasibility are
    # enforced by the authoritative lane allocator.  In particular, a
    # high-scoring bridge/probe placement may be incompatible with a required
    # memory action while another placement remains fully feasible.
    return tuple(ranked)


def _memory_combinations(
    lanes: tuple[GlobalWaveActionAllocationLane, ...],
    *,
    per_lane_limit: int,
    combination_limit: int,
) -> tuple[dict[str, MemoryAssignment], ...]:
    candidates = tuple(lane.memory_assignments[:per_lane_limit] for lane in lanes)
    indexed = itertools.product(*(tuple(enumerate(value)) for value in candidates))
    rows = sorted(
        indexed,
        key=lambda row: (
            sum(value[0] for value in row),
            tuple(value[0] for value in row),
        ),
    )[:combination_limit]
    return tuple(
        {
            lane.lane_id: indexed_assignment[1]
            for lane, indexed_assignment in zip(lanes, row, strict=True)
        }
        for row in rows
    )


@dataclass(slots=True)
class GlobalRoleBalancedWaveActionAllocationPolicy:
    """Bounded joint search with one wave-level exploration role budget."""

    structural_redundancy_penalty: float = 0.25
    near_outcome_redundancy_penalty: float = 0.15
    near_outcome_distance: float = 0.03
    memory_assignments_per_lane: int = 8
    memory_combination_limit: int = 4

    def __post_init__(self) -> None:
        for name in (
            "structural_redundancy_penalty",
            "near_outcome_redundancy_penalty",
            "near_outcome_distance",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("memory_assignments_per_lane", "memory_combination_limit"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")

    def _allocate_lane(
        self,
        lane: GlobalWaveActionAllocationLane,
        *,
        assignment: MemoryAssignment,
        role_slots: RoleSlots | None,
        eligible_option_ids: tuple[str, ...] | None,
    ) -> ActionAllocationResult:
        return allocate_target_conditioned_actions(
            plan=lane.plan,
            forecasts=lane.forecasts,
            portfolio_size=lane.request.portfolio_size,
            eligible_option_ids=eligible_option_ids,
            risk_aversion=lane.risk_aversion,
            diversity_weight=lane.diversity_weight,
            beam_width=lane.beam_width,
            utility_mode=lane.utility_mode,
            required_option_ids=_required_option_ids(lane, assignment),
            role_slots=role_slots,
            exact_arm_count_constraints=contextual_action_arm_count_constraints(
                # The forecast plan is the authenticated allocation universe.
                # The selection request may intentionally retain a much larger
                # source union for support auditing and must not be used to
                # reconstruct constraints over the screened forecast batch.
                finite_contract=lane.plan.request.finite_variation_contract,
                allocation=lane.contextual_allocation,
                portfolio_size=lane.request.portfolio_size,
            ),
            minimum_single_path_interventions=(
                0
                if lane.contextual_allocation is None
                else lane.contextual_allocation.minimum_single_path_interventions
            ),
            minimum_disjoint_parent_patch_pairs=(
                0
                if lane.contextual_allocation is None
                else lane.contextual_allocation.minimum_disjoint_parent_patch_pairs
            ),
        )

    @staticmethod
    def _count_vectors(
        target: tuple[tuple[str, int], ...],
        *,
        slots: int,
    ) -> tuple[tuple[tuple[str, int], ...], ...]:
        """Enumerate canonical exposure vectors by distance from one request."""

        arm_ids = tuple(value[0] for value in target)
        requested = dict(target)
        rows = tuple(
            tuple(zip(arm_ids, counts, strict=True))
            for counts in itertools.product(range(slots + 1), repeat=len(arm_ids))
            if sum(counts) == slots
        )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    sum(abs(count - requested[arm_id]) for arm_id, count in row),
                    row,
                ),
            )
        )

    def _allocate_lane_with_count_recourse(
        self,
        lane: GlobalWaveActionAllocationLane,
        *,
        assignment: MemoryAssignment,
        role_slots: RoleSlots | None,
        eligible_option_ids: tuple[str, ...] | None,
    ) -> _LaneAllocationAttempt:
        """Project an unrealizable count request onto the nearest legal slate.

        Role, memory, structural, phenotype, and evaluator constraints remain
        hard.  Only the controller's desired source/operator marginals may
        move, and the first successful L1 shell is exhausted so forecast
        utility breaks ties without permitting a more distant exposure vector.
        """

        original = lane.contextual_allocation
        if original is None:
            return _LaneAllocationAttempt(
                allocation=self._allocate_lane(
                    lane,
                    assignment=assignment,
                    role_slots=role_slots,
                    eligible_option_ids=eligible_option_ids,
                ),
                source_l1_deviation=0,
                operator_l1_deviation=0,
                contextual_counts_projected=False,
            )

        source_rows = self._count_vectors(
            original.source_target_counts,
            slots=lane.request.portfolio_size,
        )
        operator_rows = self._count_vectors(
            original.operator_target_counts,
            slots=lane.request.portfolio_size,
        )
        requested_source = dict(original.source_target_counts)
        requested_operator = dict(original.operator_target_counts)
        candidates = sorted(
            itertools.product(source_rows, operator_rows),
            key=lambda rows: (
                sum(
                    abs(count - requested_source[arm_id])
                    for arm_id, count in rows[0]
                )
                + sum(
                    abs(count - requested_operator[arm_id])
                    for arm_id, count in rows[1]
                ),
                rows,
            ),
        )
        best: tuple[ActionAllocationResult, int, int] | None = None
        best_distance: int | None = None
        failures: list[BaseException] = []
        for source_counts, operator_counts in candidates:
            source_l1 = sum(
                abs(count - requested_source[arm_id])
                for arm_id, count in source_counts
            )
            operator_l1 = sum(
                abs(count - requested_operator[arm_id])
                for arm_id, count in operator_counts
            )
            distance = source_l1 + operator_l1
            if best_distance is not None and distance > best_distance:
                break
            projected_contract = replace(
                original,
                source_target_counts=source_counts,
                operator_target_counts=operator_counts,
                feasibility_witness_option_identity_sha256s=(),
            )
            try:
                allocation = self._allocate_lane(
                    replace(lane, contextual_allocation=projected_contract),
                    assignment=assignment,
                    role_slots=role_slots,
                    eligible_option_ids=eligible_option_ids,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                failures.append(error)
                continue
            if best is None or (
                allocation.decision.final_score.total_utility,
                allocation.decision.receipt_sha256,
            ) > (
                best[0].decision.final_score.total_utility,
                best[0].decision.receipt_sha256,
            ):
                best = (allocation, source_l1, operator_l1)
                best_distance = distance
        if best is None:
            detail = "" if not failures else f"; {type(failures[0]).__name__}:{failures[0]}"
            raise ValueError(
                "contextual count projection found no hard-feasible lane slate"
                + detail
            )
        allocation, source_l1, operator_l1 = best
        selected_option_ids = tuple(
            value.option_id for value in allocation.decision.members
        )
        realization = contextual_allocation_realization(
            finite_contract=lane.plan.request.finite_variation_contract,
            allocation=original,
            selected_option_ids=selected_option_ids,
        )
        if realization is None:  # pragma: no cover - original is non-null.
            raise AssertionError("contextual projection lost its source contract")
        if (
            realization.source_l1_deviation != source_l1
            or realization.operator_l1_deviation != operator_l1
        ):
            raise RuntimeError("projected count receipt differs from selected slate")
        return _LaneAllocationAttempt(
            allocation=allocation,
            source_l1_deviation=source_l1,
            operator_l1_deviation=operator_l1,
            contextual_counts_projected=not realization.exact,
        )

    def _eligible_after_prior(
        self,
        lane: GlobalWaveActionAllocationLane,
        *,
        assignment: MemoryAssignment,
        prior: tuple[
            tuple[GlobalWaveActionAllocationLane, ActionAllocationResult], ...
        ],
        signatures: dict[str, dict[str, ActionStructuralSignature]],
        cells: dict[str, dict[str, tuple[tuple[str, float], ...]]],
    ) -> tuple[str, ...]:
        required = set(_required_option_ids(lane, assignment))
        blocked_signatures = {
            signatures[prior_lane.lane_id][member.option_id]
            for prior_lane, allocation in prior
            for member in allocation.decision.members
        }
        prior_cells = tuple(
            cells[prior_lane.lane_id][member.option_id]
            for prior_lane, allocation in prior
            for member in allocation.decision.members
        )
        eligible: list[str] = []
        for forecast in lane.forecasts.forecasts:
            option_id = forecast.option_id
            if option_id in required:
                eligible.append(option_id)
                continue
            if signatures[lane.lane_id][option_id] in blocked_signatures:
                continue
            cell = cells[lane.lane_id][option_id]
            if any(
                distance is not None and distance <= self.near_outcome_distance
                for distance in (_cell_distance(cell, value) for value in prior_cells)
            ):
                continue
            eligible.append(option_id)
        return tuple(sorted(eligible))

    def _collision_counts(
        self,
        allocations: dict[str, ActionAllocationResult],
        *,
        signatures: dict[str, dict[str, ActionStructuralSignature]],
        cells: dict[str, dict[str, tuple[tuple[str, float], ...]]],
    ) -> tuple[int, int]:
        structural = 0
        near = 0
        lane_ids = tuple(sorted(allocations))
        for left_index, left_id in enumerate(lane_ids):
            for right_id in lane_ids[left_index + 1 :]:
                for left in allocations[left_id].decision.members:
                    for right in allocations[right_id].decision.members:
                        if (
                            signatures[left_id][left.option_id]
                            == signatures[right_id][right.option_id]
                        ):
                            structural += 1
                        distance = _cell_distance(
                            cells[left_id][left.option_id],
                            cells[right_id][right.option_id],
                        )
                        if (
                            distance is not None
                            and distance <= self.near_outcome_distance
                        ):
                            near += 1
        return structural, near

    def allocate(
        self,
        lanes: tuple[GlobalWaveActionAllocationLane, ...],
    ) -> dict[str, GlobalWaveActionAllocationLaneResult]:
        self.__post_init__()
        if type(lanes) is not tuple or not lanes:
            raise ValueError("lanes must be a non-empty exact tuple")
        for lane in lanes:
            if type(lane) is not GlobalWaveActionAllocationLane:
                raise TypeError("lanes must contain exact lane requests")
            lane.__post_init__()
        lanes = tuple(sorted(lanes, key=lambda value: value.lane_id))
        if len({value.lane_id for value in lanes}) != len(lanes):
            raise ValueError("global wave cannot repeat a lane")
        if len({value.wave_key for value in lanes}) != 1:
            raise ValueError("global allocation lanes belong to different waves")
        signatures = {
            lane.lane_id: action_structural_signatures_by_option(
                lane.plan.request.finite_variation_contract
            )
            for lane in lanes
        }
        cells = {lane.lane_id: _p50_cells(lane) for lane in lanes}
        role_budgets = _role_budgets(lanes)
        maximum_memory_combinations = math.prod(
            min(len(lane.memory_assignments), self.memory_assignments_per_lane)
            for lane in lanes
        )
        all_memory_combinations = _memory_combinations(
            lanes,
            per_lane_limit=self.memory_assignments_per_lane,
            combination_limit=maximum_memory_combinations,
        )
        primary_memory_combinations = all_memory_combinations[
            : self.memory_combination_limit
        ]
        fallback_memory_combinations = all_memory_combinations[
            len(primary_memory_combinations) :
        ]
        memory_batches = (
            primary_memory_combinations,
            fallback_memory_combinations,
        )
        orders = tuple(
            (*lanes[index:], *lanes[:index]) for index in range(len(lanes))
        )
        best: _Trial | None = None
        projected_best: _Trial | None = None
        failures: list[str] = []
        projection_failures: list[str] = []
        trial_count = 0
        projection_trial_count = 0
        projection_candidates: list[
            tuple[
                dict[str, RoleSlots | None],
                dict[str, MemoryAssignment],
                tuple[GlobalWaveActionAllocationLane, ...],
                bool,
            ]
        ] = []
        allocation_cache: dict[
            tuple[
                str,
                MemoryAssignment,
                RoleSlots | None,
                tuple[str, ...] | None,
                bool,
            ],
            _LaneAllocationAttempt,
        ] = {}
        fallback_memory_search_used = False

        def attempt_trial(
            *,
            role_budget: dict[str, RoleSlots | None],
            memory: dict[str, MemoryAssignment],
            order: tuple[GlobalWaveActionAllocationLane, ...],
            avoid_redundancy: bool,
            permit_count_projection: bool,
        ) -> _Trial:
            allocations: dict[str, ActionAllocationResult] = {}
            eligible_by_lane: dict[str, tuple[str, ...] | None] = {}
            prior: list[
                tuple[GlobalWaveActionAllocationLane, ActionAllocationResult]
            ] = []
            source_l1 = 0
            operator_l1 = 0
            projection_count = 0
            for lane in order:
                eligible = None
                if avoid_redundancy and prior:
                    candidate_eligible = self._eligible_after_prior(
                        lane,
                        assignment=memory[lane.lane_id],
                        prior=tuple(prior),
                        signatures=signatures,
                        cells=cells,
                    )
                    if len(candidate_eligible) >= lane.request.portfolio_size:
                        eligible = candidate_eligible
                cache_key = (
                    lane.lane_id,
                    memory[lane.lane_id],
                    role_budget[lane.lane_id],
                    eligible,
                    permit_count_projection,
                )
                attempt = allocation_cache.get(cache_key)
                if attempt is None:
                    if permit_count_projection:
                        attempt = self._allocate_lane_with_count_recourse(
                            lane,
                            assignment=memory[lane.lane_id],
                            role_slots=role_budget[lane.lane_id],
                            eligible_option_ids=eligible,
                        )
                    else:
                        attempt = _LaneAllocationAttempt(
                            allocation=self._allocate_lane(
                                lane,
                                assignment=memory[lane.lane_id],
                                role_slots=role_budget[lane.lane_id],
                                eligible_option_ids=eligible,
                            ),
                            source_l1_deviation=0,
                            operator_l1_deviation=0,
                            contextual_counts_projected=False,
                        )
                    allocation_cache[cache_key] = attempt
                allocation = attempt.allocation
                allocations[lane.lane_id] = allocation
                eligible_by_lane[lane.lane_id] = eligible
                prior.append((lane, allocation))
                source_l1 += attempt.source_l1_deviation
                operator_l1 += attempt.operator_l1_deviation
                projection_count += attempt.contextual_counts_projected
            structural, near = self._collision_counts(
                allocations,
                signatures=signatures,
                cells=cells,
            )
            local = sum(
                value.decision.final_score.total_utility
                for value in allocations.values()
            )
            total = (
                local
                - self.structural_redundancy_penalty * structural
                - self.near_outcome_redundancy_penalty * near
            )
            return _Trial(
                allocations=tuple(sorted(allocations.items())),
                memory_assignments=tuple(sorted(memory.items())),
                role_slots=tuple(sorted(role_budget.items())),
                eligible_option_ids=tuple(sorted(eligible_by_lane.items())),
                lane_order=tuple(value.lane_id for value in order),
                avoided_redundancy=avoid_redundancy,
                structural_collisions=structural,
                near_outcome_collisions=near,
                local_utility=float(local),
                total_score=float(total),
                contextual_source_l1_deviation=source_l1,
                contextual_operator_l1_deviation=operator_l1,
                contextual_projection_count=projection_count,
            )

        for batch_index, memory_combinations in enumerate(memory_batches):
            if not memory_combinations:
                continue
            if batch_index == 1:
                fallback_memory_search_used = True
            for role_budget in role_budgets:
                for memory in memory_combinations:
                    for order in orders:
                        for avoid_redundancy in (False, True):
                            trial_count += 1
                            try:
                                trial = attempt_trial(
                                    role_budget=role_budget,
                                    memory=memory,
                                    order=order,
                                    avoid_redundancy=avoid_redundancy,
                                    permit_count_projection=False,
                                )
                            except (TypeError, ValueError, RuntimeError) as error:
                                failures.append(f"{type(error).__name__}:{error}")
                                projection_candidates.append(
                                    (
                                        role_budget,
                                        memory,
                                        order,
                                        avoid_redundancy,
                                    )
                                )
                                continue
                            if best is None or (
                                trial.total_score,
                                trial.tie_break,
                            ) > (
                                best.total_score,
                                best.tie_break,
                            ):
                                best = trial
            # The bounded primary set is a quality-prioritized search.  The
            # larger set is a feasibility recourse only; do not pay its cost
            # once any primary assignment produces a legal joint slate.
            if best is not None:
                break
        if best is None:
            for (
                role_budget,
                memory,
                order,
                avoid_redundancy,
            ) in projection_candidates:
                projection_trial_count += 1
                try:
                    projected = attempt_trial(
                        role_budget=role_budget,
                        memory=memory,
                        order=order,
                        avoid_redundancy=avoid_redundancy,
                        permit_count_projection=True,
                    )
                except (TypeError, ValueError, RuntimeError) as projected_error:
                    projection_failures.append(
                        f"{type(projected_error).__name__}:{projected_error}"
                    )
                else:
                    if projected_best is None or (
                        projected.contextual_l1_deviation,
                        -projected.total_score,
                        projected.tie_break,
                    ) < (
                        projected_best.contextual_l1_deviation,
                        -projected_best.total_score,
                        projected_best.tie_break,
                    ):
                        projected_best = projected
        contextual_count_recourse_used = best is None and projected_best is not None
        if best is None:
            best = projected_best
        if best is None:
            detail = "; ".join((*failures[:2], *projection_failures[:2]))
            raise ValueError(
                "global wave allocation found no hard-feasible joint slate"
                + ("; " + detail if detail else "")
            )
        record = {
            "schema_version": 1,
            "policy": {
                "policy_id": GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_ID,
                "policy_version": GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_VERSION,
                "definition_sha256": (
                    GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_DEFINITION_SHA256
                ),
            },
            "wave_key": {
                "generation": lanes[0].generation,
                "archive_utility_snapshot_sha256": lanes[0].wave_key[1],
            },
            "lane_order": list(best.lane_order),
            "avoided_redundancy": best.avoided_redundancy,
            "role_slots": {
                lane_id: None if slots is None else list(slots)
                for lane_id, slots in best.role_slots
            },
            "selected_option_ids": {
                lane_id: [value.option_id for value in allocation.decision.members]
                for lane_id, allocation in best.allocations
            },
            "eligible_option_ids": {
                lane_id: (None if option_ids is None else list(option_ids))
                for lane_id, option_ids in best.eligible_option_ids
            },
            "allocation_receipt_sha256s": {
                lane_id: allocation.decision.receipt_sha256
                for lane_id, allocation in best.allocations
            },
            "memory_assignments": {
                lane_id: dict(assignment)
                for lane_id, assignment in best.memory_assignments
            },
            "contextual_allocation_contract_sha256s": {
                lane.lane_id: (
                    None
                    if lane.contextual_allocation is None
                    else lane.contextual_allocation.contract_sha256
                )
                for lane in lanes
            },
            "source_finite_contract_sha256s": {
                lane.lane_id: lane.request.finite_variation_contract.identity_sha256
                for lane in lanes
            },
            "allocation_finite_contract_sha256s": {
                lane.lane_id: (
                    lane.plan.request.finite_variation_contract.identity_sha256
                )
                for lane in lanes
            },
            "structural_collisions": best.structural_collisions,
            "near_outcome_collisions": best.near_outcome_collisions,
            "near_outcome_distance_hex": self.near_outcome_distance.hex(),
            "local_utility_hex": best.local_utility.hex(),
            "structural_penalty_hex": self.structural_redundancy_penalty.hex(),
            "near_outcome_penalty_hex": self.near_outcome_redundancy_penalty.hex(),
            "total_score_hex": best.total_score.hex(),
            "trial_count": trial_count,
            "failed_trial_count": len(failures),
            "memory_combination_count": len(all_memory_combinations),
            "primary_memory_combination_count": len(primary_memory_combinations),
            "fallback_memory_search_used": fallback_memory_search_used,
            "contextual_count_recourse_used": contextual_count_recourse_used,
            "contextual_projection_deferred_until_exact_exhaustion": True,
            "contextual_projection_eligible_trial_count": len(
                projection_candidates
            ),
            "contextual_projection_trial_count": projection_trial_count,
            "contextual_projection_failure_count": len(projection_failures),
            "contextual_source_l1_deviation": (
                best.contextual_source_l1_deviation
            ),
            "contextual_operator_l1_deviation": (
                best.contextual_operator_l1_deviation
            ),
            "contextual_projection_count": best.contextual_projection_count,
        }
        receipt = hashlib.sha256(_RECEIPT_DOMAIN + _canonical_json(record)).hexdigest()
        audit = _object({**record, "receipt_sha256": receipt})
        allocation_by_id = dict(best.allocations)
        memory_by_id = dict(best.memory_assignments)
        roles_by_id = dict(best.role_slots)
        eligible_by_id = dict(best.eligible_option_ids)
        return {
            lane.lane_id: GlobalWaveActionAllocationLaneResult(
                allocation=allocation_by_id[lane.lane_id],
                memory_assignment=memory_by_id[lane.lane_id],
                role_slots=roles_by_id[lane.lane_id],
                eligible_option_ids=eligible_by_id[lane.lane_id],
                global_receipt_sha256=receipt,
                audit=audit,
            )
            for lane in lanes
        }


@dataclass(slots=True)
class BarrierGlobalWaveActionAllocationCoordinator:
    """Cancellation-safe rendezvous for independently forecast parent lanes."""

    policy: GlobalWaveActionAllocationPolicy
    expected_lane_count: int = 2
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _pending: dict[
        tuple[int, str],
        dict[
            str,
            tuple[
                GlobalWaveActionAllocationLane,
                asyncio.Future[GlobalWaveActionAllocationLaneResult],
            ],
        ],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, GlobalWaveActionAllocationPolicy):
            raise TypeError("policy must satisfy GlobalWaveActionAllocationPolicy")
        if type(self.expected_lane_count) is not int or self.expected_lane_count <= 0:
            raise ValueError("expected_lane_count must be a positive exact integer")

    async def allocate(
        self,
        lane: GlobalWaveActionAllocationLane,
    ) -> GlobalWaveActionAllocationLaneResult:
        self.__post_init__()
        if type(lane) is not GlobalWaveActionAllocationLane:
            raise TypeError("lane must be exact")
        lane.__post_init__()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GlobalWaveActionAllocationLaneResult] = (
            loop.create_future()
        )
        key = lane.wave_key
        async with self._lock:
            bucket = self._pending.setdefault(key, {})
            if lane.lane_id in bucket:
                raise ValueError("global allocation received the same lane twice")
            bucket[lane.lane_id] = (lane, future)
            if len(bucket) > self.expected_lane_count:
                bucket.pop(lane.lane_id)
                raise ValueError("global allocation exceeded expected lane count")
            if len(bucket) == self.expected_lane_count:
                entries = tuple(bucket.values())
                self._pending.pop(key)
                try:
                    results = self.policy.allocate(tuple(value[0] for value in entries))
                    if set(results) != {value[0].lane_id for value in entries}:
                        raise ValueError("global allocation result differs from cohort")
                except BaseException as error:  # noqa: BLE001 - release all waiters.
                    for _, pending_future in entries:
                        if not pending_future.done():
                            pending_future.set_exception(error)
                else:
                    for pending_lane, pending_future in entries:
                        pending_future.set_result(results[pending_lane.lane_id])
        try:
            return await asyncio.shield(future)
        except BaseException:
            async with self._lock:
                bucket = self._pending.get(key)
                if (
                    bucket is not None
                    and bucket.get(lane.lane_id, (None, None))[1] is future
                ):
                    bucket.pop(lane.lane_id)
                    if not bucket:
                        self._pending.pop(key)
            if not future.done():
                future.cancel()
            raise


__all__ = [
    "GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_DEFINITION_SHA256",
    "GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_ID",
    "GLOBAL_WAVE_ACTION_ALLOCATION_POLICY_VERSION",
    "BarrierGlobalWaveActionAllocationCoordinator",
    "GlobalRoleBalancedWaveActionAllocationPolicy",
    "GlobalWaveActionAllocationCoordinator",
    "GlobalWaveActionAllocationLane",
    "GlobalWaveActionAllocationLaneResult",
    "GlobalWaveActionAllocationPolicy",
    "MemoryAssignment",
    "RoleSlots",
]

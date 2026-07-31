"""Campaign bridge for prior-only stage-global contextual allocations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol, runtime_checkable

from agent_evolve.application.contextual_search_controller import (
    ContextualSearchLedger,
    ContextualSearchQuery,
    ContextualSearchStageAllocation,
    PhaseAwareContextualSearchController,
    slice_contextual_search_decision,
)
from agent_evolve.application.action_structural_signature import (
    parent_relative_changed_json_paths_by_option,
    parent_relative_path_sets_are_disjoint,
)
from agent_evolve.application.campaign_execution import CampaignStageRequest
from agent_evolve.application.evolution_campaign import (
    ParentVariationBinding,
    PreparedEvolutionCampaign,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.selection.affine_frontier_target import (
    AuthenticatedAffineFrontierTargetAllocator,
)
from agent_evolve.ports.contextual_search_allocation import (
    ContextualArmCountCapability,
    ContextualArmCountCapabilityWitness,
    ContextualJointCountVector,
    ContextualLaneJointCountCapability,
    ContextualPortfolioAllocationContract,
    ContextualPortfolioAllocationRealization,
)
from agent_evolve.ports.frontier_target import (
    CampaignPortfolioFrontierTarget,
    CampaignPortfolioFrontierTargetAllocator,
    objective_space_target_from_campaign_target,
)
from agent_evolve.ports.variation_source import (
    PRIMARY_VARIATION_SOURCE_ID,
    finite_variation_operator_id,
    finite_variation_source_ids,
    finite_variation_source_id,
    finite_variation_source_minimum_counts,
)
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
)


_PLAN_DOMAIN = b"agent-evolve:contextual-campaign-search-plan:v1\x00"
_JOINT_CONSTRAINT_DOMAIN = b"agent-evolve:finite-contract-joint-count-constraint:v3\x00"


@runtime_checkable
class CampaignContextualPlanningContext(Protocol):
    """Narrow structural view; avoids coupling the planner to one runtime."""

    prepared: PreparedEvolutionCampaign
    stage_request: CampaignStageRequest
    parent_lane: object
    variation: ParentVariationBinding

    def __post_init__(self) -> None: ...


@runtime_checkable
class CampaignContextualJointCapabilityProjector(Protocol):
    """Optional inverted API for current-wave finite structural capacity."""

    def project(
        self,
        context: CampaignContextualPlanningContext,
        *,
        evaluation_slots: int,
        source_arm_ids: tuple[str, ...],
        operator_arm_ids: tuple[str, ...],
    ) -> ContextualLaneJointCountCapability: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _project_archive_front_size(archive_record: dict[str, object]) -> int:
    """Resolve one generic frontier count from supported archive projections.

    Campaign adapters may expose the count directly, inside the authenticated
    Pareto summary, or implicitly through the serialized frontier.  Treat all
    present projections as redundant witnesses and fail closed on drift.
    """

    if type(archive_record) is not dict:
        raise TypeError("archive_record must be an exact object")
    witnesses: list[int] = []
    direct = archive_record.get("front_size")
    if direct is not None:
        if type(direct) is not int or direct <= 0:
            raise ValueError("campaign archive has an invalid direct front size")
        witnesses.append(direct)
    summary = archive_record.get("summary")
    if summary is not None:
        if type(summary) is not dict:
            raise TypeError("campaign archive summary must be an object")
        summarized = summary.get("front_size")
        if summarized is not None:
            if type(summarized) is not int or summarized <= 0:
                raise ValueError("campaign archive has an invalid summary front size")
            witnesses.append(summarized)
    front_candidates = archive_record.get("front_candidates")
    if front_candidates is not None:
        if type(front_candidates) is not list or not front_candidates:
            raise ValueError("campaign archive frontier must be a non-empty list")
        witnesses.append(len(front_candidates))
    if not witnesses:
        raise ValueError("campaign archive omitted a positive frontier witness")
    if len(set(witnesses)) != 1:
        raise ValueError("campaign archive frontier witnesses disagree")
    return witnesses[0]


@dataclass(frozen=True, slots=True)
class CampaignContextualSearchPlan:
    """One stage allocation and its request-local signed contracts."""

    campaign_generation: int
    stage_allocation: ContextualSearchStageAllocation
    contracts: tuple[ContextualPortfolioAllocationContract, ...]
    frontier_targets: tuple[CampaignPortfolioFrontierTarget, ...]
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.campaign_generation) is not int or self.campaign_generation <= 0:
            raise ValueError("campaign_generation must be positive")
        if type(self.stage_allocation) is not ContextualSearchStageAllocation:
            raise TypeError("stage_allocation must be exact")
        self.stage_allocation.__post_init__()
        if (
            type(self.contracts) is not tuple
            or not self.contracts
            or any(
                type(value) is not ContextualPortfolioAllocationContract
                for value in self.contracts
            )
        ):
            raise ValueError("contracts must contain exact allocation contracts")
        for value in self.contracts:
            value.__post_init__()
            if value.campaign_generation != self.campaign_generation:
                raise ValueError("allocation contract differs from the stage")
        if tuple(value.slice_id for value in self.contracts) != tuple(
            value.slice_id for value in self.stage_allocation.slices
        ):
            raise ValueError("allocation contracts differ from stage slices")
        if tuple(value.decision_sha256 for value in self.contracts) != tuple(
            self.stage_allocation.decision.decision_sha256 for _ in self.contracts
        ):
            raise ValueError("allocation contracts differ from the decision")
        if type(self.frontier_targets) is not tuple or any(
            type(value) is not CampaignPortfolioFrontierTarget
            for value in self.frontier_targets
        ):
            raise TypeError("frontier_targets must contain exact targets")
        for value in self.frontier_targets:
            value.__post_init__()
        if tuple(value.lane_id for value in self.frontier_targets) != tuple(
            value.slice_id for value in self.contracts
        ):
            raise ValueError("frontier targets must exactly cover allocation slices")
        object.__setattr__(
            self,
            "plan_sha256",
            hashlib.sha256(
                _PLAN_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "campaign_generation": self.campaign_generation,
            "stage_allocation_sha256": self.stage_allocation.allocation_sha256,
            "contract_sha256s": [value.contract_sha256 for value in self.contracts],
            "frontier_target_sha256s": [
                value.target_sha256 for value in self.frontier_targets
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "stage_allocation": self.stage_allocation.to_record(),
            "contracts": [value.to_record() for value in self.contracts],
            "frontier_targets": [value.to_record() for value in self.frontier_targets],
            "plan_sha256": self.plan_sha256,
        }

    def contract_for_slice(
        self, slice_id: str
    ) -> ContextualPortfolioAllocationContract:
        try:
            return next(value for value in self.contracts if value.slice_id == slice_id)
        except StopIteration as error:
            raise ValueError(
                "allocation plan has no contract for this slice"
            ) from error

    def frontier_target_for_lane(
        self,
        lane_id: str,
    ) -> CampaignPortfolioFrontierTarget:
        try:
            return next(
                value for value in self.frontier_targets if value.lane_id == lane_id
            )
        except StopIteration as error:
            raise ValueError("allocation plan has no target for this lane") from error


def _empirical_count_capability(
    realizations: tuple[ContextualPortfolioAllocationRealization, ...],
    *,
    kind: str,
    current_wave_index: int,
    evaluation_slots: int,
    arm_ids: tuple[str, ...],
) -> ContextualArmCountCapability | None:
    """Build a conservative prior-wave capability only after observed recourse."""

    grouped: dict[int, list[ContextualPortfolioAllocationRealization]] = {}
    for value in realizations:
        value.__post_init__()
        if value.controller_wave_index >= current_wave_index:
            continue
        grouped.setdefault(value.controller_wave_index, []).append(value)
    witnesses: list[ContextualArmCountCapabilityWitness] = []
    mismatch_observed = False
    requested_name = f"requested_{kind}_target_counts"
    realized_name = f"realized_{kind}_target_counts"
    for wave_index, values in sorted(grouped.items()):
        requested = {arm_id: 0 for arm_id in arm_ids}
        realized = {arm_id: 0 for arm_id in arm_ids}
        for value in values:
            requested_counts = getattr(value, requested_name)
            realized_counts = getattr(value, realized_name)
            if tuple(arm_id for arm_id, _ in requested_counts) != arm_ids:
                raise ValueError("allocation realization uses foreign capability arms")
            for arm_id, count in requested_counts:
                requested[arm_id] += count
            for arm_id, count in realized_counts:
                realized[arm_id] += count
        if sum(requested.values()) != evaluation_slots:
            # A different scale shape is evidence, but not a count vector in the
            # current stage polytope.  Keep it outside this bounded capability.
            continue
        mismatch_observed |= requested != realized
        witnesses.append(
            ContextualArmCountCapabilityWitness(
                controller_wave_index=wave_index,
                evaluation_slots=evaluation_slots,
                realized_target_counts=tuple(sorted(realized.items())),
                allocation_realization_sha256s=tuple(
                    sorted(value.realization_sha256 for value in values)
                ),
            )
        )
    if not mismatch_observed or not witnesses:
        return None
    return ContextualArmCountCapability(
        kind=kind,
        evaluation_slots=evaluation_slots,
        arm_ids=arm_ids,
        witnesses=tuple(sorted(witnesses, key=lambda value: value.witness_sha256)),
    )


def _bounded_compositions(
    total: int,
    maxima: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []

    def visit(index: int, remaining: int, prefix: tuple[int, ...]) -> None:
        if index == len(maxima) - 1:
            if 0 <= remaining <= maxima[index]:
                rows.append((*prefix, remaining))
            return
        lower = max(0, remaining - sum(maxima[index + 1 :]))
        upper = min(maxima[index], remaining)
        for value in range(lower, upper + 1):
            visit(index + 1, remaining - value, (*prefix, value))

    visit(0, total, ())
    return tuple(rows)


def _contingency_tables(
    row_counts: tuple[int, ...],
    column_counts: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if sum(row_counts) != sum(column_counts):
        raise ValueError("joint marginal counts must have equal capacity")
    tables: list[tuple[tuple[int, ...], ...]] = []

    def visit(
        row_index: int,
        remaining_columns: tuple[int, ...],
        rows: tuple[tuple[int, ...], ...],
    ) -> None:
        if row_index == len(row_counts) - 1:
            if sum(remaining_columns) == row_counts[row_index]:
                tables.append((*rows, remaining_columns))
            return
        for row in _bounded_compositions(
            row_counts[row_index],
            remaining_columns,
        ):
            visit(
                row_index + 1,
                tuple(
                    remaining - used
                    for remaining, used in zip(
                        remaining_columns,
                        row,
                        strict=True,
                    )
                ),
                (*rows, row),
            )

    visit(0, column_counts, ())
    return tuple(tables)


@dataclass(frozen=True, slots=True)
class FiniteContractContextualJointCapabilityProjector:
    """Derive exact current-wave arm marginals from sealed finite structure.

    This reusable projector knows only finite action metadata and structural
    selection constraints.  It does not receive objectives, model outputs,
    posterior scores, workload identifiers, or provider state.
    """

    min_distinct_families: int | None = None
    require_pairwise_disjoint_parent_patches: bool = False
    family_exposure_bounds: tuple[tuple[str, int, int], ...] = ()
    operator_exposure_bounds: tuple[tuple[str, int, int], ...] = ()
    minimum_single_path_interventions: int = 0
    protect_future_recombination_opportunities: bool = False
    require_declared_source_floor_options: bool = True

    def __post_init__(self) -> None:
        if self.min_distinct_families is not None and (
            type(self.min_distinct_families) is not int
            or self.min_distinct_families <= 0
        ):
            raise ValueError("min_distinct_families must be positive or None")
        if type(self.require_pairwise_disjoint_parent_patches) is not bool:
            raise TypeError(
                "require_pairwise_disjoint_parent_patches must be an exact bool"
            )
        if type(self.require_declared_source_floor_options) is not bool:
            raise TypeError(
                "require_declared_source_floor_options must be an exact bool"
            )
        if type(self.protect_future_recombination_opportunities) is not bool:
            raise TypeError(
                "protect_future_recombination_opportunities must be an exact bool"
            )
        if (
            type(self.minimum_single_path_interventions) is not int
            or self.minimum_single_path_interventions < 0
        ):
            raise ValueError(
                "minimum_single_path_interventions must be non-negative"
            )
        if type(self.family_exposure_bounds) is not tuple:
            raise TypeError("family_exposure_bounds must be an exact tuple")
        families: list[str] = []
        for value in self.family_exposure_bounds:
            if type(value) is not tuple or len(value) != 3:
                raise TypeError("family exposure bounds must be exact triples")
            family, minimum, maximum = value
            if type(family) is not str or not family:
                raise ValueError("family exposure bound needs a family")
            if (
                type(minimum) is not int
                or type(maximum) is not int
                or minimum < 0
                or maximum < minimum
            ):
                raise ValueError("family exposure bound is invalid")
            families.append(family)
        if families != sorted(set(families)):
            raise ValueError("family exposure bounds must be unique and canonical")
        if type(self.operator_exposure_bounds) is not tuple:
            raise TypeError("operator_exposure_bounds must be an exact tuple")
        operators: list[str] = []
        for value in self.operator_exposure_bounds:
            if type(value) is not tuple or len(value) != 3:
                raise TypeError("operator exposure bounds must be exact triples")
            operator, minimum, maximum = value
            if type(operator) is not str or not operator:
                raise ValueError("operator exposure bound needs an operator")
            if (
                type(minimum) is not int
                or type(maximum) is not int
                or minimum < 0
                or maximum < minimum
            ):
                raise ValueError("operator exposure bound is invalid")
            operators.append(operator)
        if operators != sorted(set(operators)):
            raise ValueError("operator exposure bounds must be unique and canonical")

    @property
    def structural_constraint_sha256(self) -> str:
        self.__post_init__()
        return hashlib.sha256(
            _JOINT_CONSTRAINT_DOMAIN
            + _canonical_json(
                {
                    "schema_version": 2,
                    "witness_solver": {
                        "policy_id": "exact_structural_witness_search",
                        "policy_version": 3,
                        "pairwise_equivalence": (
                            "source_operator_family_exact_changed_path_set"
                        ),
                        "pairwise_representative": "lexicographically_first_option",
                        "pairwise_search_order": (
                            "descending_compatibility_then_arm_family_option"
                        ),
                    },
                    "min_distinct_families": self.min_distinct_families,
                    "require_pairwise_disjoint_parent_patches": (
                        self.require_pairwise_disjoint_parent_patches
                    ),
                    "family_exposure_bounds": [
                        list(value) for value in self.family_exposure_bounds
                    ],
                    "operator_exposure_bounds": [
                        list(value) for value in self.operator_exposure_bounds
                    ],
                    "minimum_single_path_interventions": (
                        self.minimum_single_path_interventions
                    ),
                    "protect_future_recombination_opportunities": (
                        self.protect_future_recombination_opportunities
                    ),
                    "offspring_opportunity_policy": (
                        "reserve_planned_disjoint_parent_patch_pairs_when_consumed"
                    ),
                    "intervention_axis": (
                        "exact_parent_relative_changed_json_path_count"
                    ),
                    "require_declared_source_floor_options": (
                        self.require_declared_source_floor_options
                    ),
                    "source_floor_semantics": (
                        "minimum_count_per_source_not_fixed_representative"
                    ),
                    "source_axis": "sealed_finite_variation_source",
                    "operator_axis": (
                        "declared_evaluation_operator_or_composition_metadata"
                    ),
                    "diversity_axis": "finite_option_family",
                    "objective_values_consulted": False,
                    "workload_identifiers_consulted": False,
                }
            )
        ).hexdigest()

    def project(
        self,
        context: CampaignContextualPlanningContext,
        *,
        evaluation_slots: int,
        source_arm_ids: tuple[str, ...],
        operator_arm_ids: tuple[str, ...],
    ) -> ContextualLaneJointCountCapability:
        self.__post_init__()
        if not isinstance(context, CampaignContextualPlanningContext):
            raise TypeError("context must implement the planning context port")
        context.__post_init__()
        if type(evaluation_slots) is not int or evaluation_slots <= 0:
            raise ValueError("evaluation_slots must be positive")
        if self.min_distinct_families is not None and (
            self.min_distinct_families > evaluation_slots
        ):
            raise ValueError("minimum family diversity exceeds lane capacity")
        if self.minimum_single_path_interventions > evaluation_slots:
            raise ValueError("minimum single-path floor exceeds lane capacity")
        future_recombination_widths = (
            tuple(
                step.offspring_per_parent
                for step in context.prepared.schedule.steps
                if step.source_portfolio_generation
                == context.stage_request.step.generation
            )
            if self.protect_future_recombination_opportunities
            else ()
        )
        if len(future_recombination_widths) > 1:
            raise ValueError("portfolio generation has multiple recombination consumers")
        minimum_disjoint_parent_patch_pairs = (
            future_recombination_widths[0]
            if self.protect_future_recombination_opportunities
            and future_recombination_widths
            else 0
        )
        maximum_selected_pairs = evaluation_slots * (evaluation_slots - 1) // 2
        if minimum_disjoint_parent_patch_pairs > maximum_selected_pairs:
            raise ValueError(
                "planned recombination width exceeds the selected parent-pair universe"
            )
        contract = context.variation.contract
        contract.__post_init__()
        options = contract.options
        source_by_option = {
            value.option_id: finite_variation_source_id(value) for value in options
        }
        operator_by_option = {
            value.option_id: finite_variation_operator_id(value) for value in options
        }
        if not set(source_by_option.values()).issubset(source_arm_ids):
            raise ValueError("finite contract exposes a source outside planner arms")
        if not set(operator_by_option.values()).issubset(operator_arm_ids):
            raise ValueError("finite contract exposes an operator outside planner arms")
        if not {
            operator for operator, _, _ in self.operator_exposure_bounds
        }.issubset(operator_arm_ids):
            raise ValueError("operator exposure bound escapes planner arms")
        option_by_id = {value.option_id: value for value in options}
        changed_paths_by_option = parent_relative_changed_json_paths_by_option(
            contract
        )
        changed_path_keys_by_option = {
            option_id: tuple(path.schema_identity for path in paths)
            for option_id, paths in changed_paths_by_option.items()
        }
        single_path_option_ids = {
            option_id
            for option_id, paths in changed_paths_by_option.items()
            if len(paths) == 1
        }
        if len(single_path_option_ids) < self.minimum_single_path_interventions:
            raise ValueError(
                "finite contract cannot satisfy its single-path intervention floor"
            )
        source_minimum_counts = dict(
            finite_variation_source_minimum_counts(contract)
            if self.require_declared_source_floor_options
            else ()
        )
        if not set(source_minimum_counts).issubset(source_arm_ids):
            raise ValueError("declared source floor escapes planner arms")
        if sum(source_minimum_counts.values()) > evaluation_slots:
            raise ValueError("declared source floors exceed lane capacity")
        # Strict patch-disjoint feasibility depends only on source, operator,
        # family, and the exact parent-relative path set.  Options sharing all
        # four attributes are interchangeable to every constraint below and,
        # because their non-empty path sets overlap, at most one can occur in
        # a legal slate.  Collapse that exact equivalence before constructing
        # the compatibility graph.  This is lossless but prevents a catalogue
        # with many values per locus from turning a K-small witness query into
        # an O(N**K) traversal over phenotype multiplicity.
        pairwise_candidate_ids: tuple[str, ...] = ()
        if self.require_pairwise_disjoint_parent_patches:
            equivalence_groups: dict[
                tuple[str, str, str, tuple[str, ...]], list[str]
            ] = {}
            for option in options:
                option_id = option.option_id
                signature = (
                    source_by_option[option_id],
                    operator_by_option[option_id],
                    option.family,
                    changed_path_keys_by_option[option_id],
                )
                equivalence_groups.setdefault(signature, []).append(option_id)
            pairwise_candidate_ids = tuple(
                sorted(
                    min(option_ids)
                    for option_ids in equivalence_groups.values()
                )
            )
        allowed_pairs = (
            None
            if not self.require_pairwise_disjoint_parent_patches
            else {
                frozenset(value)
                for value in pairwise_disjoint_parent_patch_pairs(
                    contract,
                    pairwise_candidate_ids,
                )
            }
        )
        disjoint_pair_cache: dict[frozenset[str], bool] = {}

        def pair_is_disjoint(left: str, right: str) -> bool:
            if left == right:
                return False
            key = frozenset((left, right))
            cached = disjoint_pair_cache.get(key)
            if cached is not None:
                return cached
            value = parent_relative_path_sets_are_disjoint(
                changed_paths_by_option[left],
                changed_paths_by_option[right],
            )
            disjoint_pair_cache[key] = value
            return value

        def disjoint_pair_count(selected: tuple[str, ...]) -> int:
            return sum(
                pair_is_disjoint(left, right)
                for index, left in enumerate(selected)
                for right in selected[index + 1 :]
            )
        family_bounds = {
            family: (minimum, maximum)
            for family, minimum, maximum in self.family_exposure_bounds
        }
        operator_bounds = {
            operator: (minimum, maximum)
            for operator, minimum, maximum in self.operator_exposure_bounds
        }

        def structurally_valid(
            selected: tuple[str, ...],
            *,
            check_pairwise_patches: bool = True,
            check_offspring_opportunity: bool = True,
        ) -> bool:
            if len(selected) != evaluation_slots:
                return False
            if any(
                sum(source_by_option[value] == source_id for value in selected)
                < minimum
                for source_id, minimum in source_minimum_counts.items()
            ):
                return False
            if any(
                not minimum
                <= sum(operator_by_option[value] == operator for value in selected)
                <= maximum
                for operator, (minimum, maximum) in operator_bounds.items()
            ):
                return False
            if sum(
                value in single_path_option_ids for value in selected
            ) < self.minimum_single_path_interventions:
                return False
            if (
                check_offspring_opportunity
                and disjoint_pair_count(selected)
                < minimum_disjoint_parent_patch_pairs
            ):
                return False
            families = tuple(option_by_id[value].family for value in selected)
            if self.min_distinct_families is not None and len(set(families)) < (
                self.min_distinct_families
            ):
                return False
            if any(
                not minimum <= families.count(family) <= maximum
                for family, (minimum, maximum) in family_bounds.items()
            ):
                return False
            return (
                not check_pairwise_patches
                or allowed_pairs is None
                or all(
                    frozenset((left, right)) in allowed_pairs
                    for index, left in enumerate(selected)
                    for right in selected[index + 1 :]
                )
            )

        compatible_degree = (
            {}
            if allowed_pairs is None
            else {
                option_id: sum(
                    frozenset((option_id, other_id)) in allowed_pairs
                    for other_id in pairwise_candidate_ids
                    if other_id != option_id
                )
                for option_id in pairwise_candidate_ids
            }
        )
        pairwise_base_candidates = (
            ()
            if allowed_pairs is None
            else tuple(
                sorted(
                    pairwise_candidate_ids,
                    key=lambda option_id: (
                        -compatible_degree[option_id],
                        source_by_option[option_id],
                        operator_by_option[option_id],
                        option_by_id[option_id].family,
                        option_id,
                    ),
                )
            )
        )
        options_by_arm_cell = {
            (source_id, operator_id): tuple(
                option.option_id
                for option in options
                if source_by_option[option.option_id] == source_id
                and operator_by_option[option.option_id] == operator_id
            )
            for source_id in source_arm_ids
            for operator_id in operator_arm_ids
        }
        base_options_by_arm_cell: dict[tuple[str, str], tuple[str, ...]] = {}
        for arm_cell, group in options_by_arm_cell.items():
            if allowed_pairs is not None:
                base_options_by_arm_cell[arm_cell] = group
                continue
            # Family and exact intervention arity are the only option-level
            # axes used by a base witness. Keep K representatives for each
            # exact shape, interleaved across shapes so the first solution is
            # diverse. Offspring opportunity is subsequently repaired over
            # the complete changed-path equivalence domain.
            by_shape: dict[tuple[str, bool], tuple[str, ...]] = {}
            for family in sorted({option_by_id[value].family for value in group}):
                for single_path in (True, False):
                    members = tuple(
                        value
                        for value in group
                        if option_by_id[value].family == family
                        and (value in single_path_option_ids) is single_path
                    )
                    if members:
                        by_shape[(family, single_path)] = members[:evaluation_slots]
            reduced: list[str] = []
            for rank in range(evaluation_slots):
                reduced.extend(
                    members[rank]
                    for _, members in sorted(by_shape.items())
                    if rank < len(members)
                )
            base_options_by_arm_cell[arm_cell] = tuple(reduced)

        # Within one source/operator/family/arity signature, only the exact
        # changed-path pattern can affect offspring opportunity. Retain one
        # representative per pattern (up to K identical-path duplicates) so
        # the bounded repair is complete without exposing the full catalogue
        # multiplicity to the base marginal solver.
        groups: dict[tuple[str, str, str, bool], list[str]] = {}
        signature_by_option: dict[str, tuple[str, str, str, bool]] = {}
        for option in options:
            signature = (
                source_by_option[option.option_id],
                operator_by_option[option.option_id],
                option.family,
                option.option_id in single_path_option_ids,
            )
            groups.setdefault(signature, []).append(option.option_id)
            signature_by_option[option.option_id] = signature
        opportunity_domain_by_signature: dict[
            tuple[str, str, str, bool], tuple[str, ...]
        ] = {}
        if minimum_disjoint_parent_patch_pairs:
            for signature, option_ids in groups.items():
                by_paths: dict[tuple[str, ...], list[str]] = {}
                for option_id in sorted(option_ids):
                    by_paths.setdefault(
                        changed_path_keys_by_option[option_id], []
                    ).append(option_id)
                opportunity_domain_by_signature[signature] = tuple(
                    option_id
                    for paths in sorted(by_paths)
                    for option_id in by_paths[paths][:evaluation_slots]
                )

        def repair_offspring_opportunity(
            selected: tuple[str, ...],
        ) -> tuple[str, ...] | None:
            if (
                not minimum_disjoint_parent_patch_pairs
                or disjoint_pair_count(selected)
                >= minimum_disjoint_parent_patch_pairs
            ):
                return selected
            position_signatures = tuple(
                signature_by_option[value] for value in selected
            )

            def search(
                position: int,
                chosen: tuple[str, ...],
            ) -> tuple[str, ...] | None:
                remaining = len(position_signatures) - position
                observed_pairs = disjoint_pair_count(chosen)
                optimistic_pairs = (
                    observed_pairs
                    + len(chosen) * remaining
                    + remaining * (remaining - 1) // 2
                )
                if optimistic_pairs < minimum_disjoint_parent_patch_pairs:
                    return None
                if position == len(position_signatures):
                    return chosen if structurally_valid(chosen) else None
                domain = opportunity_domain_by_signature[
                    position_signatures[position]
                ]
                preferred = selected[position]
                ordered = tuple(
                    sorted(
                        domain,
                        key=lambda option_id: (
                            -sum(
                                pair_is_disjoint(option_id, prior)
                                for prior in chosen
                            ),
                            option_id != preferred,
                            changed_path_keys_by_option[option_id],
                            option_id,
                        ),
                    )
                )
                for option_id in ordered:
                    if option_id in chosen:
                        continue
                    resolved = search(position + 1, (*chosen, option_id))
                    if resolved is not None:
                        return resolved
                return None

            return search(0, ())

        def first_witness(
            source_counts: tuple[int, ...],
            operator_counts: tuple[int, ...],
        ) -> tuple[str, ...] | None:
            source_targets = dict(zip(source_arm_ids, source_counts, strict=True))
            operator_targets = dict(zip(operator_arm_ids, operator_counts, strict=True))
            required_source_counts = {value: 0 for value in source_arm_ids}
            required_operator_counts = {value: 0 for value in operator_arm_ids}
            if any(
                required_source_counts[value] > source_targets[value]
                for value in source_arm_ids
            ) or any(
                required_operator_counts[value] > operator_targets[value]
                for value in operator_arm_ids
            ):
                return None
            if allowed_pairs is not None:
                initial: tuple[str, ...] = ()
                remaining_source = {
                    value: source_targets[value] - required_source_counts[value]
                    for value in source_arm_ids
                }
                remaining_operator = {
                    value: operator_targets[value] - required_operator_counts[value]
                    for value in operator_arm_ids
                }
                candidates = pairwise_base_candidates

                def pairwise_search(
                    selected: tuple[str, ...],
                    available: tuple[str, ...],
                    source_remaining: dict[str, int],
                    operator_remaining: dict[str, int],
                ) -> tuple[str, ...] | None:
                    slots_remaining = evaluation_slots - len(selected)
                    if slots_remaining == 0:
                        if any(source_remaining.values()) or any(
                            operator_remaining.values()
                        ):
                            return None
                        return selected if structurally_valid(selected) else None
                    eligible = tuple(
                        value
                        for value in available
                        if source_remaining[source_by_option[value]] > 0
                        and operator_remaining[operator_by_option[value]] > 0
                    )
                    if len(eligible) < slots_remaining:
                        return None
                    if any(
                        sum(source_by_option[value] == arm_id for value in eligible)
                        < remaining
                        for arm_id, remaining in source_remaining.items()
                    ) or any(
                        sum(operator_by_option[value] == arm_id for value in eligible)
                        < remaining
                        for arm_id, remaining in operator_remaining.items()
                    ):
                        return None
                    selected_families = tuple(
                        option_by_id[value].family for value in selected
                    )
                    eligible_families = {
                        option_by_id[value].family for value in eligible
                    }
                    if self.min_distinct_families is not None and (
                        len(set(selected_families).union(eligible_families))
                        < self.min_distinct_families
                    ):
                        return None
                    if any(
                        selected_families.count(family)
                        + min(
                            slots_remaining,
                            sum(
                                option_by_id[value].family == family
                                for value in eligible
                            ),
                        )
                        < minimum
                        for family, (minimum, _) in family_bounds.items()
                    ):
                        return None
                    for index, option_id in enumerate(eligible):
                        family = option_by_id[option_id].family
                        maximum = family_bounds.get(
                            family,
                            (0, evaluation_slots),
                        )[1]
                        if selected_families.count(family) >= maximum:
                            continue
                        source_id = source_by_option[option_id]
                        operator_id = operator_by_option[option_id]
                        next_source = dict(source_remaining)
                        next_operator = dict(operator_remaining)
                        next_source[source_id] -= 1
                        next_operator[operator_id] -= 1
                        tail = tuple(
                            value
                            for value in eligible[index + 1 :]
                            if frozenset((option_id, value)) in allowed_pairs
                        )
                        resolved = pairwise_search(
                            (*selected, option_id),
                            tail,
                            next_source,
                            next_operator,
                        )
                        if resolved is not None:
                            return tuple(sorted(resolved))
                    return None

                return pairwise_search(
                    initial,
                    candidates,
                    remaining_source,
                    remaining_operator,
                )
            for table in _contingency_tables(source_counts, operator_counts):
                required_by_cell = {
                    (source_id, operator_id): sum(
                        source_by_option[value] == source_id
                        and operator_by_option[value] == operator_id
                        for value in ()
                    )
                    for source_id in source_arm_ids
                    for operator_id in operator_arm_ids
                }
                cell_rows: list[tuple[tuple[str, str], tuple[str, ...], int]] = []
                table_valid = True
                for source_index, source_id in enumerate(source_arm_ids):
                    for operator_index, operator_id in enumerate(operator_arm_ids):
                        target = table[source_index][operator_index]
                        required_count = required_by_cell[(source_id, operator_id)]
                        if required_count > target:
                            table_valid = False
                            break
                        group = base_options_by_arm_cell[(source_id, operator_id)]
                        needed = target - required_count
                        if needed > len(group):
                            table_valid = False
                            break
                        if needed:
                            cell_rows.append(((source_id, operator_id), group, needed))
                    if not table_valid:
                        break
                if not table_valid:
                    continue
                ordered_cells = tuple(
                    sorted(
                        cell_rows,
                        key=lambda value: (
                            len(value[1]),
                            value[0],
                        ),
                    )
                )
                initial: tuple[str, ...] = ()

                def search(
                    cell_index: int,
                    selected: tuple[str, ...],
                ) -> tuple[str, ...] | None:
                    if cell_index == len(ordered_cells):
                        if not structurally_valid(
                            selected,
                            check_offspring_opportunity=False,
                        ):
                            return None
                        return repair_offspring_opportunity(selected)
                    _, group, needed = ordered_cells[cell_index]
                    for choice in combinations(group, needed):
                        combined = (*selected, *choice)
                        if allowed_pairs is not None and any(
                            frozenset((left, right)) not in allowed_pairs
                            for index, left in enumerate(combined)
                            for right in combined[index + 1 :]
                        ):
                            continue
                        families = tuple(
                            option_by_id[value].family for value in combined
                        )
                        if any(
                            families.count(family) > maximum
                            for family, (_, maximum) in family_bounds.items()
                        ):
                            continue
                        if self.min_distinct_families is not None:
                            remaining_slots = evaluation_slots - len(combined)
                            remaining_families = {
                                option_by_id[value].family
                                for _, remaining_group, _ in ordered_cells[
                                    cell_index + 1 :
                                ]
                                for value in remaining_group
                            }
                            if (
                                len(set(families))
                                + min(
                                    remaining_slots,
                                    len(remaining_families.difference(families)),
                                )
                                < self.min_distinct_families
                            ):
                                continue
                        resolved = search(cell_index + 1, combined)
                        if resolved is not None:
                            return resolved
                    return None

                witness = search(0, initial)
                if witness is not None:
                    return tuple(sorted(witness))
            return None

        # Eagerly enumerating every K-combination of structural signatures is
        # O(G**K) in the number of exposed families. Large finite catalogues
        # can have hundreds of such families even though the controller needs
        # only the small source/operator marginal polytope. Solve those
        # marginals directly below; the exact path-equivalence repair remains
        # responsible for the registered offspring opportunity floor.

        vectors: list[ContextualJointCountVector] = []
        source_vectors = _bounded_compositions(
            evaluation_slots,
            tuple(evaluation_slots for _ in source_arm_ids),
        )
        operator_vectors = _bounded_compositions(
            evaluation_slots,
            tuple(evaluation_slots for _ in operator_arm_ids),
        )
        for source_counts in source_vectors:
            if any(
                source_counts[source_arm_ids.index(source_id)] < minimum
                for source_id, minimum in source_minimum_counts.items()
            ):
                continue
            for operator_counts in operator_vectors:
                witness = first_witness(source_counts, operator_counts)
                if witness is not None and allowed_pairs is None:
                    witness = repair_offspring_opportunity(witness)
                if witness is None:
                    continue
                vectors.append(
                    ContextualJointCountVector(
                        source_target_counts=tuple(
                            zip(source_arm_ids, source_counts, strict=True)
                        ),
                        operator_target_counts=tuple(
                            zip(operator_arm_ids, operator_counts, strict=True)
                        ),
                        feasibility_witness_option_identity_sha256s=tuple(
                            sorted(
                                option_by_id[value].identity_sha256 for value in witness
                            )
                        ),
                    )
                )
        if not vectors:
            raise ValueError("finite contract has no jointly feasible lane vector")
        return ContextualLaneJointCountCapability(
            slice_id=context.parent_lane.lane_id,
            finite_contract_identity_sha256=contract.identity_sha256,
            structural_constraint_sha256=self.structural_constraint_sha256,
            evaluation_slots=evaluation_slots,
            source_arm_ids=source_arm_ids,
            operator_arm_ids=operator_arm_ids,
            feasible_vectors=tuple(
                sorted(vectors, key=lambda value: value.vector_sha256)
            ),
            minimum_single_path_interventions=(
                self.minimum_single_path_interventions
            ),
            minimum_disjoint_parent_patch_pairs=(
                minimum_disjoint_parent_patch_pairs
            ),
        )


@dataclass(slots=True)
class CampaignContextualSearchPlanner:
    """Build prior-only decisions from framework facts shared by all workloads."""

    ledger: ContextualSearchLedger
    campaign_scope_sha256: str
    available_source_ids: tuple[str, ...] | None = None
    available_operator_ids: tuple[str, ...] = ("atomic", "composite")
    incumbent_source_id: str = PRIMARY_VARIATION_SOURCE_ID
    incumbent_operator_id: str = "atomic"
    composition_positive_atomic_threshold: int = 2
    require_objective_space_targets: bool = False
    joint_capability_projector: CampaignContextualJointCapabilityProjector | None = None
    controller: PhaseAwareContextualSearchController = field(
        default_factory=PhaseAwareContextualSearchController
    )
    frontier_target_allocator: CampaignPortfolioFrontierTargetAllocator = field(
        default_factory=AuthenticatedAffineFrontierTargetAllocator
    )
    plans: list[CampaignContextualSearchPlan] = field(
        init=False,
        default_factory=list,
    )

    def __post_init__(self) -> None:
        if type(self.ledger) is not ContextualSearchLedger:
            raise TypeError("ledger must be exact ContextualSearchLedger")
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.controller) is not PhaseAwareContextualSearchController:
            raise TypeError("controller must be exact")
        if self.available_source_ids is not None and (
            type(self.available_source_ids) is not tuple
            or not self.available_source_ids
            or self.available_source_ids
            != tuple(sorted(set(self.available_source_ids)))
        ):
            raise ValueError(
                "available_source_ids must be canonical when explicitly supplied"
            )
        if not isinstance(
            self.frontier_target_allocator,
            CampaignPortfolioFrontierTargetAllocator,
        ):
            raise TypeError("frontier_target_allocator must satisfy its port")
        if self.joint_capability_projector is not None and not isinstance(
            self.joint_capability_projector,
            CampaignContextualJointCapabilityProjector,
        ):
            raise TypeError("joint_capability_projector must satisfy its inverted API")
        if (
            type(self.composition_positive_atomic_threshold) is not int
            or self.composition_positive_atomic_threshold <= 0
        ):
            raise ValueError("composition threshold must be positive")
        if type(self.require_objective_space_targets) is not bool:
            raise TypeError("require_objective_space_targets must be exact bool")

    def plan(
        self,
        contexts: tuple[CampaignContextualPlanningContext, ...],
    ) -> CampaignContextualSearchPlan:
        self.__post_init__()
        if (
            type(contexts) is not tuple
            or not contexts
            or any(
                not isinstance(value, CampaignContextualPlanningContext)
                for value in contexts
            )
        ):
            raise ValueError("contexts must implement the planning context port")
        for value in contexts:
            value.__post_init__()
        preparations = {value.prepared.preparation_sha256 for value in contexts}
        generations = {value.stage_request.step.generation for value in contexts}
        if len(preparations) != 1 or len(generations) != 1:
            raise ValueError("contextual plan cannot mix campaigns or generations")
        generation = next(iter(generations))
        if generation % 2 != 1:
            raise ValueError("contextual planning requires a portfolio generation")
        ordered = tuple(sorted(contexts, key=lambda value: value.parent_lane.lane_id))
        slice_ids = tuple(value.parent_lane.lane_id for value in ordered)
        if slice_ids != tuple(sorted(set(slice_ids))):
            raise ValueError("parent lanes must be unique and canonical")
        prepared = ordered[0].prepared
        declared_source_ids = tuple(
            sorted(
                {
                    source_id
                    for value in ordered
                    for source_id in finite_variation_source_ids(
                        value.variation.contract
                    )
                }
            )
        )
        available_source_ids = (
            declared_source_ids
            if self.available_source_ids is None
            else self.available_source_ids
        )
        if not set(declared_source_ids).issubset(available_source_ids):
            raise ValueError(
                "contextual source arms omit a finite-contract variation source"
            )
        if self.incumbent_source_id not in available_source_ids:
            raise ValueError("incumbent source is absent from current source arms")
        portfolio_width = prepared.protocol.portfolio_width
        evaluation_slots = tuple(portfolio_width for _ in ordered)
        joint_count_capabilities = (
            ()
            if self.joint_capability_projector is None
            else tuple(
                self.joint_capability_projector.project(
                    value,
                    evaluation_slots=portfolio_width,
                    source_arm_ids=available_source_ids,
                    operator_arm_ids=self.available_operator_ids,
                )
                for value in ordered
            )
        )
        if (
            joint_count_capabilities
            and tuple(value.slice_id for value in joint_count_capabilities) != slice_ids
        ):
            raise ValueError("joint capability projection changed campaign lanes")
        wave_index = (generation + 1) // 2
        total_portfolio_waves = len(prepared.schedule.portfolio_generations)
        archive_record = thaw_json(ordered[0].stage_request.archive_cutoff.archive)
        if type(archive_record) is not dict:
            raise TypeError("campaign archive cutoff must thaw to an object")
        archive_front_size = _project_archive_front_size(archive_record)
        prior = tuple(
            value
            for value in self.ledger.observations
            if value.campaign_scope_sha256 == self.campaign_scope_sha256
            and value.wave_index < wave_index
        )
        composition_evidence_available = (
            sum(
                value.operator_id == "atomic" and value.positive_marginal_utility
                for value in prior
            )
            >= self.composition_positive_atomic_threshold
        )
        prior_realizations = tuple(
            value
            for value in self.ledger.allocation_realizations
            if value.campaign_scope_sha256 == self.campaign_scope_sha256
            and value.controller_wave_index < wave_index
        )
        source_count_capability = _empirical_count_capability(
            prior_realizations,
            kind="source",
            current_wave_index=wave_index,
            evaluation_slots=sum(evaluation_slots),
            arm_ids=available_source_ids,
        )
        operator_count_capability = _empirical_count_capability(
            prior_realizations,
            kind="operator",
            current_wave_index=wave_index,
            evaluation_slots=sum(evaluation_slots),
            arm_ids=self.available_operator_ids,
        )
        query = ContextualSearchQuery(
            campaign_scope_sha256=self.campaign_scope_sha256,
            wave_index=wave_index,
            total_portfolio_waves=total_portfolio_waves,
            real_evaluation_slots=sum(evaluation_slots),
            available_source_ids=available_source_ids,
            available_operator_ids=self.available_operator_ids,
            incumbent_source_id=self.incumbent_source_id,
            incumbent_operator_id=self.incumbent_operator_id,
            archive_front_size=archive_front_size,
            # Individual fixed-reference marginals overlap and are not an
            # authenticated stage-level archive delta. Keep this channel empty
            # until the runtime publishes an exact pre/post utility receipt.
            recent_normalized_archive_gains=(),
            composition_evidence_available=composition_evidence_available,
            source_count_capability=source_count_capability,
            operator_count_capability=operator_count_capability,
            joint_count_capabilities=joint_count_capabilities,
        )
        snapshot = self.ledger.snapshot(
            campaign_scope_sha256=self.campaign_scope_sha256,
            cutoff_wave_index_exclusive=wave_index,
            available_source_ids=available_source_ids,
            available_operator_ids=self.available_operator_ids,
        )
        decision = self.controller.decide(query, snapshot)
        stage_allocation = slice_contextual_search_decision(
            decision,
            slice_ids=slice_ids,
            evaluation_slots=evaluation_slots,
        )
        frontier_targets = self.frontier_target_allocator.allocate(
            archive_utility=ordered[0].stage_request.archive_utility,
            lanes=tuple((value.parent_lane.lane_id, value.parent) for value in ordered),
        )
        if self.require_objective_space_targets and any(
            objective_space_target_from_campaign_target(value) is None
            for value in frontier_targets
        ):
            raise ValueError(
                "contextual frontier target omits the required raw objective-space "
                "representation"
            )
        plan = CampaignContextualSearchPlan(
            campaign_generation=generation,
            stage_allocation=stage_allocation,
            contracts=tuple(
                value.to_contract(campaign_generation=generation)
                for value in stage_allocation.slices
            ),
            frontier_targets=frontier_targets,
        )
        existing = next(
            (value for value in self.plans if value.campaign_generation == generation),
            None,
        )
        if existing is not None:
            if existing != plan:
                raise RuntimeError("contextual stage was replanned differently")
            return existing
        self.plans.append(plan)
        return plan


__all__ = [
    "CampaignContextualJointCapabilityProjector",
    "CampaignContextualPlanningContext",
    "CampaignContextualSearchPlan",
    "CampaignContextualSearchPlanner",
    "FiniteContractContextualJointCapabilityProjector",
    "_empirical_count_capability",
]

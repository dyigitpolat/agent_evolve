"""Lower prior-only contextual decisions into generic allocator constraints.

This bridge deliberately knows only the sealed finite variation contract and
the generic source/operator attribution ports.  It is the single seam between
the adaptive search controller and any trusted action allocator; no workload,
model, provider, prompt, or objective branch is permitted here.
"""

from __future__ import annotations

from agent_evolve.application.action_structural_signature import (
    parent_relative_changed_paths_by_option,
)
from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.ports.action_allocation import ExactActionArmCountConstraint
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
)
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationContract,
    ContextualPortfolioAllocationRealization,
)
from agent_evolve.ports.variation_source import (
    finite_variation_operator_id,
    finite_variation_source_id,
)


def contextual_action_arm_count_constraints(
    *,
    finite_contract: FiniteVariationContract,
    allocation: ContextualPortfolioAllocationContract | None,
    portfolio_size: int,
) -> tuple[ExactActionArmCountConstraint, ...]:
    """Return exact source/operator marginals for one finite action request."""

    if type(finite_contract) is not FiniteVariationContract:
        raise TypeError("finite_contract must be an exact FiniteVariationContract")
    finite_contract.__post_init__()
    if type(portfolio_size) is not int or portfolio_size <= 0:
        raise ValueError("portfolio_size must be a positive exact integer")
    if allocation is None:
        return ()
    if type(allocation) is not ContextualPortfolioAllocationContract:
        raise TypeError("allocation must be an exact contextual contract or None")
    allocation.__post_init__()
    if allocation.evaluation_slots != portfolio_size:
        raise ValueError("contextual allocation differs from portfolio capacity")

    source_rows = tuple(
        sorted(
            (option.option_id, finite_variation_source_id(option))
            for option in finite_contract.options
        )
    )
    operator_rows = tuple(
        sorted(
            (option.option_id, finite_variation_operator_id(option))
            for option in finite_contract.options
        )
    )
    constraints = (
        ExactActionArmCountConstraint(
            constraint_id="operator",
            option_arm_ids=operator_rows,
            target_counts=allocation.operator_target_counts,
        ),
        ExactActionArmCountConstraint(
            constraint_id="source",
            option_arm_ids=source_rows,
            target_counts=allocation.source_target_counts,
        ),
    )
    return tuple(sorted(constraints, key=lambda value: value.constraint_id))


def contextual_allocation_realization(
    *,
    finite_contract: FiniteVariationContract,
    allocation: ContextualPortfolioAllocationContract | None,
    selected_option_ids: tuple[str, ...],
) -> ContextualPortfolioAllocationRealization | None:
    """Project selected sealed actions into objective-blind controller credit."""

    if type(finite_contract) is not FiniteVariationContract:
        raise TypeError("finite_contract must be an exact FiniteVariationContract")
    finite_contract.__post_init__()
    if allocation is None:
        return None
    if type(allocation) is not ContextualPortfolioAllocationContract:
        raise TypeError("allocation must be an exact contextual contract or None")
    allocation.__post_init__()
    if (
        type(selected_option_ids) is not tuple
        or len(selected_option_ids) != allocation.evaluation_slots
        or selected_option_ids != tuple(dict.fromkeys(selected_option_ids))
    ):
        raise ValueError("selected_option_ids must be an exact unique K-tuple")
    by_id = {value.option_id: value for value in finite_contract.options}
    if not set(selected_option_ids).issubset(by_id):
        raise ValueError("selected option escapes the finite variation contract")

    def realized(
        target: tuple[tuple[str, int], ...],
        projector,
    ) -> tuple[tuple[str, int], ...]:
        counts = {arm_id: 0 for arm_id, _ in target}
        for option_id in selected_option_ids:
            arm_id = projector(by_id[option_id])
            if arm_id not in counts:
                raise ValueError("selected option carries an unrequested arm")
            counts[arm_id] += 1
        return tuple(sorted(counts.items()))

    receipt = ContextualPortfolioAllocationRealization(
        campaign_scope_sha256=allocation.campaign_scope_sha256,
        query_sha256=allocation.query_sha256,
        decision_sha256=allocation.decision_sha256,
        contract_sha256=allocation.contract_sha256,
        controller_wave_index=allocation.controller_wave_index,
        slice_id=allocation.slice_id,
        requested_source_target_counts=allocation.source_target_counts,
        requested_operator_target_counts=allocation.operator_target_counts,
        realized_source_target_counts=realized(
            allocation.source_target_counts,
            finite_variation_source_id,
        ),
        realized_operator_target_counts=realized(
            allocation.operator_target_counts,
            finite_variation_operator_id,
        ),
        requested_minimum_single_path_interventions=(
            allocation.minimum_single_path_interventions
        ),
        realized_single_path_interventions=sum(
            len(paths) == 1
            for option_id, paths in parent_relative_changed_paths_by_option(
                finite_contract
            ).items()
            if option_id in selected_option_ids
        ),
        requested_minimum_disjoint_parent_patch_pairs=(
            allocation.minimum_disjoint_parent_patch_pairs
        ),
        realized_disjoint_parent_patch_pairs=len(
            pairwise_disjoint_parent_patch_pairs(
                finite_contract,
                selected_option_ids,
            )
        ),
    )
    receipt.require_contract(allocation)
    return receipt


def selected_variation_source_ids(
    *,
    finite_contract: FiniteVariationContract,
    selected_option_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return sealed proposal-source labels in evaluated member order."""

    if type(finite_contract) is not FiniteVariationContract:
        raise TypeError("finite_contract must be an exact FiniteVariationContract")
    finite_contract.__post_init__()
    if type(selected_option_ids) is not tuple:
        raise TypeError("selected_option_ids must be an exact tuple")
    by_id = {value.option_id: value for value in finite_contract.options}
    if not set(selected_option_ids).issubset(by_id):
        raise ValueError("selected option escapes the finite variation contract")
    return tuple(
        finite_variation_source_id(by_id[option_id])
        for option_id in selected_option_ids
    )


__all__ = [
    "contextual_action_arm_count_constraints",
    "contextual_allocation_realization",
    "selected_variation_source_ids",
]

"""Workload-neutral structural identities for parent-relative finite actions.

Option identifiers are intentionally opaque and need not remain stable across
parents. Search policies that learn or diversify across parent lanes therefore
need an identity derived from the action itself rather than from its label.
This module exposes exact changed JSON paths and the coarser ``(family, paths)``
signature from sealed configurations. It contains no workload names, objective
values, evaluator outcomes, or prompt text.
"""

from __future__ import annotations

from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.policies.variation.typed_patch import derive_patch


ActionStructuralSignature = tuple[str, tuple[str, ...]]


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported JSON-path segment")
    return "".join(parts)


def parent_relative_changed_paths_by_option(
    contract: FiniteVariationContract,
) -> dict[str, tuple[str, ...]]:
    """Return canonical changed-path sets for every action in ``contract``."""

    typed_paths = parent_relative_changed_json_paths_by_option(contract)
    return {
        option_id: tuple(sorted(_path_text(path) for path in paths))
        for option_id, paths in typed_paths.items()
    }


def parent_relative_changed_json_paths_by_option(
    contract: FiniteVariationContract,
) -> dict[str, tuple[JsonPath, ...]]:
    """Return exact typed parent-relative paths for every sealed action.

    Keeping this typed projection public lets structural controllers compare
    path overlap once without repeatedly revalidating a large finite contract
    or relying on ambiguous rendered JSON-path strings.
    """

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    base = CandidateId("candidate_action_signature_parent")
    target = CandidateId("candidate_action_signature_child")
    return {
        option.option_id: tuple(
            sorted(
                (
                    operation.path
                    for operation in derive_patch(
                        contract.parent_configuration,
                        option.child_configuration,
                        base_candidate_id=base,
                        target_candidate_id=target,
                    ).operations
                ),
                key=lambda path: path.schema_identity,
            )
        )
        for option in contract.options
    }


def parent_relative_path_sets_are_disjoint(
    left: tuple[JsonPath, ...],
    right: tuple[JsonPath, ...],
) -> bool:
    """Return whether two already-derived typed patch path sets are disjoint."""

    if type(left) is not tuple or type(right) is not tuple or not left or not right:
        raise ValueError("path sets must be non-empty exact tuples")
    if any(type(value) is not JsonPath for value in (*left, *right)):
        raise TypeError("path sets must contain exact JsonPath values")
    return not any(
        left_path.segments == right_path.segments[: len(left_path.segments)]
        or right_path.segments == left_path.segments[: len(right_path.segments)]
        for left_path in left
        for right_path in right
    )


def action_structural_signatures_by_option(
    contract: FiniteVariationContract,
) -> dict[str, ActionStructuralSignature]:
    """Return canonical family-plus-path identities for every sealed action."""

    paths = parent_relative_changed_paths_by_option(contract)
    return {
        option.option_id: (option.family, paths[option.option_id])
        for option in contract.options
    }


__all__ = [
    "ActionStructuralSignature",
    "action_structural_signatures_by_option",
    "parent_relative_changed_json_paths_by_option",
    "parent_relative_changed_paths_by_option",
    "parent_relative_path_sets_are_disjoint",
]

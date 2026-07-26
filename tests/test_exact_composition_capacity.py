from __future__ import annotations

import pytest

from agent_evolve.policies.variation.exact_composition_capacity import (
    project_exact_k_binary_composition,
)


def test_exact_composition_preserves_a_feasible_preference() -> None:
    projection = project_exact_k_binary_composition(
        proposal_size=8,
        preferred_composite_count=4,
        mandatory_atomic_count=3,
        mandatory_composite_count=1,
        selectable_atomic_count=8,
        selectable_composite_count=8,
    )

    assert projection.effective_composite_count == 4
    assert projection.capacity_projected is False
    assert projection.feasible_minimum_composite_count == 1
    assert projection.feasible_maximum_composite_count == 5


def test_exact_composition_projects_down_after_atomic_recourse() -> None:
    projection = project_exact_k_binary_composition(
        proposal_size=8,
        preferred_composite_count=4,
        mandatory_atomic_count=5,
        mandatory_composite_count=1,
        selectable_atomic_count=12,
        selectable_composite_count=12,
    )

    assert projection.feasible_minimum_composite_count == 1
    assert projection.feasible_maximum_composite_count == 3
    assert projection.effective_composite_count == 3
    assert projection.capacity_projected is True
    assert (
        projection.to_record()["workload_model_provider_identifiers_consulted"] is False
    )


def test_exact_composition_projects_up_when_atomic_capacity_is_sparse() -> None:
    projection = project_exact_k_binary_composition(
        proposal_size=8,
        preferred_composite_count=2,
        mandatory_atomic_count=2,
        mandatory_composite_count=2,
        selectable_atomic_count=3,
        selectable_composite_count=20,
    )

    assert projection.feasible_minimum_composite_count == 5
    assert projection.feasible_maximum_composite_count == 6
    assert projection.effective_composite_count == 5
    assert projection.capacity_projected is True


def test_exact_composition_rejects_an_oversubscribed_mandatory_partition() -> None:
    with pytest.raises(ValueError, match="mandatory set exceeds proposal size"):
        project_exact_k_binary_composition(
            proposal_size=8,
            preferred_composite_count=4,
            mandatory_atomic_count=5,
            mandatory_composite_count=4,
            selectable_atomic_count=10,
            selectable_composite_count=10,
        )

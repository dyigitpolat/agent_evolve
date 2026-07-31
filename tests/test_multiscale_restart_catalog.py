"""Contracts for the generic replay-safe multiscale restart source."""

from __future__ import annotations

import hashlib

from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.variation.multiscale_restart_catalog import (
    GENERIC_MULTISCALE_RESTART_FAMILY,
    GENERIC_MULTISCALE_RESTART_SOURCE_ID,
    GenericMultiscaleRestartFiniteVariationCatalog,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    SourceExposureFiniteVariationCatalog,
    SourceUnionFiniteVariationCatalog,
    required_source_evaluation_option_ids,
)
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from agent_evolve.ports.variation_source import (
    finite_variation_operator_id,
    finite_variation_source_id,
    finite_variation_source_minimum_counts,
)


class _EightAxisAtomicCatalog:
    catalog_id = "eight_axis_atomic_fixture"
    catalog_version = 1
    definition_sha256 = hashlib.sha256(b"eight-axis-atomic-fixture-v1").hexdigest()
    option_families = tuple(f"axis_{index}" for index in range(8))

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        parent_sha256 = typed_json_sha256(parent_configuration)
        result = []
        for index in range(8):
            for value in (1, 2):
                child = dict(parent)
                child[f"x{index}"] = value
                result.append(
                    FiniteVariationOption(
                        option_id=f"axis_{index}.value_{value}",
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=freeze_json(child),
                        family=f"axis_{index}",
                        description=f"Set axis {index} to {value}.",
                    )
                )
        return tuple(result)


def _parent() -> FrozenJsonObject:
    value = freeze_json({f"x{index}": 0 for index in range(8)})
    assert type(value) is FrozenJsonObject
    return value


def test_multiscale_restart_is_deterministic_replay_safe_and_source_typed() -> None:
    first = GenericMultiscaleRestartFiniteVariationCatalog(
        _EightAxisAtomicCatalog(),
        radii=(4, 8),
        restarts_per_radius=3,
    )
    second = GenericMultiscaleRestartFiniteVariationCatalog(
        _EightAxisAtomicCatalog(),
        radii=(4, 8),
        restarts_per_radius=3,
    )

    first_options = first.options(_parent())
    second_options = second.options(_parent())

    assert first.definition_sha256 == second.definition_sha256
    assert first_options == second_options
    assert len(first_options) == 6
    assert len({value.child_configuration_sha256 for value in first_options}) == 6
    for option in first_options:
        metadata = dict(option.metadata)
        radius = int(metadata["restart_radius"])
        child = thaw_json(option.child_configuration)
        assert sum(value != 0 for value in child.values()) == radius
        assert option.family == GENERIC_MULTISCALE_RESTART_FAMILY
        assert finite_variation_source_id(option) == (
            GENERIC_MULTISCALE_RESTART_SOURCE_ID
        )
        assert finite_variation_operator_id(option) == "composite"
        assert metadata["evaluation_source_minimum"] == "1"
        assert metadata["component_option_count"] == str(radius)


def test_multiscale_restart_composes_through_the_ordinary_source_union_port() -> None:
    atomic = _EightAxisAtomicCatalog()
    restart = GenericMultiscaleRestartFiniteVariationCatalog(
        atomic,
        radii=(4,),
        restarts_per_radius=2,
    )
    union = SourceUnionFiniteVariationCatalog(
        primary_catalog=atomic,
        source_catalogs=(restart,),
    )

    contract = bind_finite_variation_catalog(union, _parent())

    assert len(contract.options) == 18
    assert (
        sum(
            finite_variation_source_id(value) == GENERIC_MULTISCALE_RESTART_SOURCE_ID
            for value in contract.options
        )
        == 2
    )
    assert len({value.child_configuration_sha256 for value in contract.options}) == 18


def test_source_identity_can_remain_attributable_without_a_hard_floor() -> None:
    atomic = _EightAxisAtomicCatalog()
    restart = SourceExposureFiniteVariationCatalog(
        GenericMultiscaleRestartFiniteVariationCatalog(
            atomic,
            radii=(4,),
            restarts_per_radius=2,
        ),
        evaluation_source_minimum=None,
    )
    contract = bind_finite_variation_catalog(
        SourceUnionFiniteVariationCatalog(
            primary_catalog=atomic,
            source_catalogs=(restart,),
        ),
        _parent(),
    )

    restart_options = tuple(
        value
        for value in contract.options
        if finite_variation_source_id(value) == GENERIC_MULTISCALE_RESTART_SOURCE_ID
    )
    assert len(restart_options) == 2
    assert all(
        "evaluation_source_minimum" not in dict(value.metadata)
        for value in restart_options
    )
    assert finite_variation_source_minimum_counts(contract) == ()
    assert required_source_evaluation_option_ids(contract) == ()


def test_unavailable_radius_is_a_deterministic_sleeping_expert() -> None:
    restart = GenericMultiscaleRestartFiniteVariationCatalog(
        _EightAxisAtomicCatalog(),
        radii=(16,),
        restarts_per_radius=2,
    )

    assert restart.options(_parent()) == ()

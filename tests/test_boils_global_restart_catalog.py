"""Geometry and exposure contracts for optional BOiLS global proposals."""

from __future__ import annotations

from agent_evolve.agentic import (
    SourceUnionFiniteVariationCatalog,
    required_source_evaluation_option_ids,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from agent_evolve.ports.variation_source import (
    PRIMARY_VARIATION_SOURCE_ID,
    finite_variation_source_by_option,
    finite_variation_source_ids,
)
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.global_restart_catalog import (
    GLOBAL_RESTART_FAMILY,
    GLOBAL_RESTART_RADII,
    GLOBAL_RESTARTS_PER_RADIUS,
    BoilsGlobalRestartVariationCatalog,
)


def _parent() -> FrozenJsonObject:
    value = freeze_json({"sequence": list(DEFAULT_ACTION_SEQUENCE)})
    assert type(value) is FrozenJsonObject
    return value


def test_boils_global_restarts_are_exact_multiscale_outcome_blind_jumps() -> None:
    parent = _parent()
    parent_sequence = thaw_json(parent)["sequence"]
    catalog = BoilsGlobalRestartVariationCatalog()

    first = catalog.options(parent)
    second = catalog.options(parent)

    assert first == second
    assert len(first) == len(GLOBAL_RESTART_RADII) * GLOBAL_RESTARTS_PER_RADIUS
    assert all(value.family == GLOBAL_RESTART_FAMILY for value in first)
    observed_radii: list[int] = []
    for option in first:
        child_sequence = thaw_json(option.child_configuration)["sequence"]
        radius = sum(
            before != after
            for before, after in zip(
                parent_sequence,
                child_sequence,
                strict=True,
            )
        )
        observed_radii.append(radius)
        metadata = dict(option.metadata)
        assert int(metadata["changed_coordinate_count"]) == radius
        assert metadata["evaluation_source_minimum"] == "1"
    assert tuple(sorted(observed_radii)) == tuple(
        sorted(
            radius
            for radius in GLOBAL_RESTART_RADII
            for _ in range(GLOBAL_RESTARTS_PER_RADIUS)
        )
    )
    assert len({value.child_configuration_sha256 for value in first}) == len(first)


def test_source_union_preserves_local_port_and_binds_one_evaluation_witness() -> None:
    parent = _parent()
    local = BoilsFiniteVariationCatalog().options(parent)
    catalog = SourceUnionFiniteVariationCatalog(
        primary_catalog=BoilsFiniteVariationCatalog(),
        source_catalogs=(BoilsGlobalRestartVariationCatalog(),),
    )

    options = catalog.options(parent)
    contract = bind_finite_variation_catalog(catalog, parent)
    required = required_source_evaluation_option_ids(contract)

    assert options[: len(local)] == local
    assert len(options) == len(local) + 16
    assert catalog.catalog_id == BoilsFiniteVariationCatalog.catalog_id
    assert len(required) == 1
    assert contract.resolve(required[0]).family == GLOBAL_RESTART_FAMILY
    assert required == required_source_evaluation_option_ids(contract)
    source_by_option = finite_variation_source_by_option(contract)
    assert finite_variation_source_ids(contract) == (
        "global_restart",
        PRIMARY_VARIATION_SOURCE_ID,
    )
    assert all(
        source_by_option[value.option_id] == PRIMARY_VARIATION_SOURCE_ID
        for value in local
    )
    assert all(
        source_by_option[value.option_id] == "global_restart"
        for value in options[len(local) :]
    )

"""Ports for deterministic finite, parent-relative variation catalogs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.domain.variation_space import AtomicEditOption

if TYPE_CHECKING:
    from agent_evolve.application.agentic_evolution import EvolutionCandidate


@runtime_checkable
class AtomicVariationCatalog(Protocol):
    """Enumerate an immutable, deterministic option tuple for one parent."""

    def options(
        self,
        parent: "EvolutionCandidate",
    ) -> tuple[AtomicEditOption, ...]: ...


@runtime_checkable
class FiniteVariationCatalog(Protocol):
    """Enumerate sealed full-child choices for one immutable parent.

    The protocol intentionally receives configuration data rather than an
    application-layer candidate.  Benchmark adapters therefore need no import
    from the evolutionary engine and may coordinate arbitrarily many fields in
    one option.
    """

    catalog_id: str
    catalog_version: int
    definition_sha256: str

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]: ...


def bind_finite_variation_catalog(
    catalog: FiniteVariationCatalog,
    parent_configuration: FrozenJsonObject,
) -> FiniteVariationContract:
    """Seal one catalog result and its published semantics to one parent."""

    if not isinstance(catalog, FiniteVariationCatalog):
        raise TypeError("catalog must implement FiniteVariationCatalog")
    if type(parent_configuration) is not FrozenJsonObject:
        raise TypeError("parent_configuration must be an exact FrozenJsonObject")
    if freeze_json(parent_configuration) is not parent_configuration:
        raise TypeError("parent_configuration must already be frozen typed JSON")
    options = catalog.options(parent_configuration)
    if type(options) is not tuple:
        raise TypeError("finite variation catalog must return an exact tuple")
    return FiniteVariationContract(
        catalog_id=catalog.catalog_id,
        catalog_version=catalog.catalog_version,
        catalog_definition_sha256=catalog.definition_sha256,
        parent_configuration=parent_configuration,
        options=options,
    )


__all__ = [
    "AtomicVariationCatalog",
    "FiniteVariationCatalog",
    "bind_finite_variation_catalog",
]

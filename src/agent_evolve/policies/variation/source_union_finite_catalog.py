"""Outcome-blind union of local and optional global finite-action sources.

The optimizer sees one ordinary :class:`FiniteVariationCatalog`.  Workload
adapters may contribute global restarts, archive-gap macros, or other sealed
full-child sources without changing campaign orchestration.  Source options
may declare only a source ID and an evaluation exposure floor.  The floor is
resolved deterministically from the sealed finite contract, so objective
values and provider fields never enter source union or exposure selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.finite_variation import (
    MAX_FINITE_VARIATION_OPTIONS,
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.typed_json import FrozenJsonObject
from agent_evolve.ports.variation_catalog import FiniteVariationCatalog
from agent_evolve.ports.variation_source import (
    VARIATION_SOURCE_METADATA_KEY,
    VARIATION_SOURCE_MINIMUM_METADATA_KEY,
)


# Source-compatible aliases remain public while the authority lives at the
# inverted API boundary rather than in this concrete union policy.
EVALUATION_SOURCE_METADATA_KEY = VARIATION_SOURCE_METADATA_KEY
EVALUATION_SOURCE_MINIMUM_METADATA_KEY = VARIATION_SOURCE_MINIMUM_METADATA_KEY
SOURCE_UNION_POLICY_ID = "finite_variation_source_union"
SOURCE_UNION_POLICY_VERSION = 1
_DEFINITION_DOMAIN = b"agent-evolve:finite-variation-source-union:def:v1\x00"
_EXPOSURE_DOMAIN = b"agent-evolve:finite-variation-source-exposure:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _catalog_identity(catalog: FiniteVariationCatalog) -> tuple[str, int, str]:
    if not isinstance(catalog, FiniteVariationCatalog):
        raise TypeError("source union members must implement FiniteVariationCatalog")
    catalog_id = getattr(catalog, "catalog_id", None)
    catalog_version = getattr(catalog, "catalog_version", None)
    definition_sha256 = getattr(catalog, "definition_sha256", None)
    if type(catalog_id) is not str or not catalog_id:
        raise ValueError("source catalog must publish catalog_id")
    if type(catalog_version) is not int or catalog_version <= 0:
        raise ValueError("source catalog must publish a positive version")
    if (
        type(definition_sha256) is not str
        or len(definition_sha256) != 64
        or any(value not in "0123456789abcdef" for value in definition_sha256)
    ):
        raise ValueError("source catalog must publish a lowercase SHA-256")
    return catalog_id, catalog_version, definition_sha256


@dataclass(frozen=True, slots=True)
class SourceUnionFiniteVariationCatalog:
    """Merge one primary catalog with independently versioned source catalogs."""

    primary_catalog: FiniteVariationCatalog
    source_catalogs: tuple[FiniteVariationCatalog, ...]
    catalog_id: str = field(init=False)
    catalog_version: int = field(init=False)
    definition_sha256: str = field(init=False)
    option_families: tuple[str, ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        primary = _catalog_identity(self.primary_catalog)
        if type(self.source_catalogs) is not tuple or not self.source_catalogs:
            raise ValueError("source_catalogs must be a non-empty exact tuple")
        sources = tuple(_catalog_identity(value) for value in self.source_catalogs)
        if len({value[0] for value in sources}) != len(sources):
            raise ValueError("source catalog IDs must be unique")
        definition = {
            "schema_version": 1,
            "policy_id": SOURCE_UNION_POLICY_ID,
            "policy_version": SOURCE_UNION_POLICY_VERSION,
            "primary_catalog": {
                "catalog_id": primary[0],
                "catalog_version": primary[1],
                "definition_sha256": primary[2],
            },
            "source_catalogs": [
                {
                    "catalog_id": catalog_id,
                    "catalog_version": version,
                    "definition_sha256": definition_sha256,
                }
                for catalog_id, version, definition_sha256 in sources
            ],
            "merge_order": "primary_then_declared_source_order",
            "duplicate_option_ids": "reject",
            "duplicate_child_configurations": "reject",
            "outcomes_consulted": False,
            "workload_model_provider_identifiers_consulted": False,
        }
        object.__setattr__(self, "catalog_id", primary[0])
        object.__setattr__(
            self,
            "catalog_version",
            primary[1] + SOURCE_UNION_POLICY_VERSION,
        )
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                _DEFINITION_DOMAIN + _canonical_json(definition)
            ).hexdigest(),
        )
        family_sets = tuple(
            getattr(value, "option_families", None)
            for value in (self.primary_catalog, *self.source_catalogs)
        )
        if all(value is not None for value in family_sets):
            if any(
                type(value) is not tuple
                or not value
                or any(type(item) is not str or not item for item in value)
                for value in family_sets
            ):
                raise TypeError("source option_families must be non-empty tuples")
            object.__setattr__(
                self,
                "option_families",
                tuple(sorted({item for value in family_sets for item in value})),
            )

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        self.__post_init__()
        members = (self.primary_catalog, *self.source_catalogs)
        options = tuple(
            option
            for catalog in members
            for option in catalog.options(parent_configuration)
        )
        if not options:
            raise ValueError("source union cannot produce an empty finite catalog")
        if len(options) > MAX_FINITE_VARIATION_OPTIONS:
            raise ValueError("source union exceeds MAX_FINITE_VARIATION_OPTIONS")
        option_ids = tuple(value.option_id for value in options)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("source union contains duplicate option IDs")
        children = tuple(value.child_configuration_sha256 for value in options)
        if len(set(children)) != len(children):
            raise ValueError("source union contains duplicate child configurations")
        return options


def required_source_evaluation_option_ids(
    contract: FiniteVariationContract,
) -> tuple[str, ...]:
    """Select deterministic source-floor witnesses from one sealed contract.

    A downstream selector must retain these options in its provider-visible
    pool *and* its evaluated slate.  This is an exposure contract rather than
    a quality claim: the witness is selected without current or historical
    objective values and rotates automatically as the parent contract changes.
    """

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    by_source: dict[str, list[FiniteVariationOption]] = {}
    minimum_by_source: dict[str, int] = {}
    for option in contract.options:
        metadata = dict(option.metadata)
        source_id = metadata.get(EVALUATION_SOURCE_METADATA_KEY)
        raw_minimum = metadata.get(EVALUATION_SOURCE_MINIMUM_METADATA_KEY)
        if source_id is None and raw_minimum is None:
            continue
        if source_id is None or raw_minimum is None:
            raise ValueError("source option metadata must declare ID and minimum")
        if not raw_minimum.isascii() or not raw_minimum.isdigit():
            raise ValueError("evaluation source minimum must be decimal digits")
        minimum = int(raw_minimum)
        if not 1 <= minimum < 8:
            raise ValueError("evaluation source minimum must lie in [1, 8)")
        previous = minimum_by_source.setdefault(source_id, minimum)
        if previous != minimum:
            raise ValueError("one proposal source declares inconsistent minimums")
        by_source.setdefault(source_id, []).append(option)

    selected: list[str] = []
    contract_identity = bytes.fromhex(contract.identity_sha256)
    for source_id in sorted(by_source):
        source_options = by_source[source_id]
        minimum = minimum_by_source[source_id]
        if len(source_options) < minimum:
            raise ValueError("proposal source cannot satisfy its exposure minimum")
        ordered = sorted(
            source_options,
            key=lambda option: (
                hashlib.sha256(
                    _EXPOSURE_DOMAIN
                    + contract_identity
                    + source_id.encode("ascii", errors="strict")
                    + bytes.fromhex(option.identity_sha256)
                ).digest(),
                option.option_id,
            ),
        )
        selected.extend(option.option_id for option in ordered[:minimum])
    return tuple(sorted(selected))


__all__ = [
    "EVALUATION_SOURCE_METADATA_KEY",
    "EVALUATION_SOURCE_MINIMUM_METADATA_KEY",
    "SOURCE_UNION_POLICY_ID",
    "SOURCE_UNION_POLICY_VERSION",
    "SourceUnionFiniteVariationCatalog",
    "required_source_evaluation_option_ids",
]

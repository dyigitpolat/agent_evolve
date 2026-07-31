"""Generic multiscale restarts assembled from sealed atomic actions.

The policy gives every finite typed workload a global-coverage proposal source
without requiring benchmark-specific restart code.  It chooses bounded,
parent-keyed sets of pairwise-disjoint atomic actions at several radii and
materializes their union through the same replay-checked typed-patch machinery
used by crossover.  No outcomes, objective values, model fields, or workload
identifiers enter construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import FrozenJsonObject, typed_json_sha256
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchRecombinationError,
    DisjointPatchRecombiner,
)
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
)
from agent_evolve.ports.variation_catalog import FiniteVariationCatalog
from agent_evolve.ports.variation_source import (
    VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY,
    VARIATION_OPERATOR_METADATA_KEY,
    VARIATION_SOURCE_METADATA_KEY,
    VARIATION_SOURCE_MINIMUM_METADATA_KEY,
)


GENERIC_MULTISCALE_RESTART_CATALOG_ID = "generic_multiscale_restart"
GENERIC_MULTISCALE_RESTART_CATALOG_VERSION = 1
GENERIC_MULTISCALE_RESTART_SOURCE_ID = "multiscale_restart"
GENERIC_MULTISCALE_RESTART_FAMILY = "multiscale_restart"
GENERIC_MULTISCALE_RESTART_POLICY_ID = "sealed_atomic_multiscale_restart"
GENERIC_MULTISCALE_RESTART_POLICY_VERSION = 1
DEFAULT_RESTART_RADII = (4, 8, 16, 32)
DEFAULT_RESTARTS_PER_RADIUS = 4
_DEFINITION_DOMAIN = b"agent-evolve:generic-multiscale-restart:def:v1\x00"
_ORDER_DOMAIN = b"agent-evolve:generic-multiscale-restart:order:v1\x00"
_OPTION_DOMAIN = b"agent-evolve:generic-multiscale-restart:option:v1\x00"


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
        raise TypeError("atomic_catalog must implement FiniteVariationCatalog")
    catalog_id = getattr(catalog, "catalog_id", None)
    catalog_version = getattr(catalog, "catalog_version", None)
    definition_sha256 = getattr(catalog, "definition_sha256", None)
    if type(catalog_id) is not str or not catalog_id:
        raise ValueError("atomic_catalog must publish catalog_id")
    if type(catalog_version) is not int or catalog_version <= 0:
        raise ValueError("atomic_catalog must publish a positive version")
    if (
        type(definition_sha256) is not str
        or len(definition_sha256) != 64
        or any(value not in "0123456789abcdef" for value in definition_sha256)
    ):
        raise ValueError("atomic_catalog must publish a lowercase SHA-256")
    return catalog_id, catalog_version, definition_sha256


@dataclass(frozen=True, slots=True)
class GenericMultiscaleRestartFiniteVariationCatalog:
    """Construct replay-safe global jumps from any sealed atomic catalog."""

    atomic_catalog: FiniteVariationCatalog
    radii: tuple[int, ...] = DEFAULT_RESTART_RADII
    restarts_per_radius: int = DEFAULT_RESTARTS_PER_RADIUS
    evaluation_source_minimum: int = 1
    catalog_id: str = field(
        init=False,
        default=GENERIC_MULTISCALE_RESTART_CATALOG_ID,
    )
    catalog_version: int = field(
        init=False,
        default=GENERIC_MULTISCALE_RESTART_CATALOG_VERSION,
    )
    definition_sha256: str = field(init=False)
    option_families: tuple[str, ...] = field(
        init=False,
        default=(GENERIC_MULTISCALE_RESTART_FAMILY,),
    )

    def __post_init__(self) -> None:
        base_id, base_version, base_definition = _catalog_identity(self.atomic_catalog)
        if (
            type(self.radii) is not tuple
            or not self.radii
            or any(type(value) is not int or value < 3 for value in self.radii)
            or self.radii != tuple(sorted(set(self.radii)))
        ):
            raise ValueError("radii must be canonical unique integers of at least 3")
        if (
            type(self.restarts_per_radius) is not int
            or not 1 <= self.restarts_per_radius <= 16
        ):
            raise ValueError("restarts_per_radius must lie in [1, 16]")
        if (
            type(self.evaluation_source_minimum) is not int
            or not 1 <= self.evaluation_source_minimum < 8
        ):
            raise ValueError("evaluation_source_minimum must lie in [1, 8)")
        definition = {
            "schema_version": 1,
            "policy_id": GENERIC_MULTISCALE_RESTART_POLICY_ID,
            "policy_version": GENERIC_MULTISCALE_RESTART_POLICY_VERSION,
            "atomic_catalog": {
                "catalog_id": base_id,
                "catalog_version": base_version,
                "definition_sha256": base_definition,
            },
            "radii": list(self.radii),
            "restarts_per_radius": self.restarts_per_radius,
            "evaluation_source_minimum": self.evaluation_source_minimum,
            "option_order": "definition_parent_radius_slot_keyed_sha256",
            "group_construction": "greedy_pairwise_disjoint_clique",
            "materialization": "iterated_replay_checked_disjoint_patch_union",
            "failed_or_unavailable_radius": "deterministic_skip",
            "outcomes_consulted": False,
            "objective_values_consulted": False,
            "model_provider_fields_consulted": False,
            "workload_identifiers_consulted": False,
        }
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                _DEFINITION_DOMAIN + _canonical_json(definition)
            ).hexdigest(),
        )

    def _ordered_atomic_ids(
        self,
        *,
        options: tuple[FiniteVariationOption, ...],
        parent_sha256: str,
        radius: int,
        restart_slot: int,
    ) -> tuple[str, ...]:
        key = radius.to_bytes(4, "big") + restart_slot.to_bytes(4, "big")
        return tuple(
            value.option_id
            for value in sorted(
                options,
                key=lambda value: (
                    hashlib.sha256(
                        _ORDER_DOMAIN
                        + bytes.fromhex(self.definition_sha256)
                        + bytes.fromhex(parent_sha256)
                        + key
                        + bytes.fromhex(value.identity_sha256)
                    ).digest(),
                    value.option_id,
                ),
            )
        )

    @staticmethod
    def _disjoint_group(
        *,
        ordered_ids: tuple[str, ...],
        disjoint_pairs: frozenset[tuple[str, str]],
        radius: int,
    ) -> tuple[str, ...] | None:
        selected: list[str] = []
        for option_id in ordered_ids:
            if all(
                tuple(sorted((option_id, previous))) in disjoint_pairs
                for previous in selected
            ):
                selected.append(option_id)
                if len(selected) == radius:
                    return tuple(selected)
        return None

    @staticmethod
    def _materialize_group(
        *,
        parent_configuration: FrozenJsonObject,
        options_by_id: dict[str, FiniteVariationOption],
        option_ids: tuple[str, ...],
        radius: int,
        restart_slot: int,
    ) -> tuple[FrozenJsonObject, tuple[str, ...]] | None:
        current = options_by_id[option_ids[0]].child_configuration
        receipts: list[str] = []
        recombiner = DisjointPatchRecombiner()
        ancestor_id = CandidateId("candidate_multiscale_restart_parent")
        for step, option_id in enumerate(option_ids[1:], start=2):
            tag = f"r{radius:03d}_s{restart_slot:02d}_k{step:03d}"
            try:
                materialization = recombiner.materialize(
                    ancestor=parent_configuration,
                    ancestor_candidate_id=ancestor_id,
                    left=current,
                    left_candidate_id=CandidateId(
                        f"candidate_multiscale_restart_acc_{tag}"
                    ),
                    right=options_by_id[option_id].child_configuration,
                    right_candidate_id=CandidateId(
                        f"candidate_multiscale_restart_atom_{tag}"
                    ),
                    target_candidate_id=CandidateId(
                        f"candidate_multiscale_restart_union_{tag}"
                    ),
                )
            except DisjointPatchRecombinationError:
                return None
            current = materialization.configuration
            receipts.append(materialization.receipt_sha256)
        if type(current) is not FrozenJsonObject:  # pragma: no cover - closed root.
            raise AssertionError("multiscale restart materialized a non-object")
        return current, tuple(receipts)

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        self.__post_init__()
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("parent_configuration must be exact FrozenJsonObject")
        atomic_options = self.atomic_catalog.options(parent_configuration)
        if type(atomic_options) is not tuple or not atomic_options:
            raise ValueError("atomic_catalog must return a non-empty exact tuple")
        if any(type(value) is not FiniteVariationOption for value in atomic_options):
            raise TypeError("atomic_catalog returned a foreign option")
        contract = FiniteVariationContract(
            catalog_id=self.atomic_catalog.catalog_id,
            catalog_version=self.atomic_catalog.catalog_version,
            catalog_definition_sha256=self.atomic_catalog.definition_sha256,
            parent_configuration=parent_configuration,
            options=atomic_options,
        )
        options_by_id = {value.option_id: value for value in atomic_options}
        disjoint_pairs = frozenset(
            pairwise_disjoint_parent_patch_pairs(
                contract,
                tuple(options_by_id),
            )
        )
        parent_sha256 = typed_json_sha256(parent_configuration)
        child_hashes: set[str] = set()
        restarts: list[FiniteVariationOption] = []
        for radius in self.radii:
            for restart_slot in range(self.restarts_per_radius):
                ordered_ids = self._ordered_atomic_ids(
                    options=atomic_options,
                    parent_sha256=parent_sha256,
                    radius=radius,
                    restart_slot=restart_slot,
                )
                group = self._disjoint_group(
                    ordered_ids=ordered_ids,
                    disjoint_pairs=disjoint_pairs,
                    radius=radius,
                )
                if group is None:
                    continue
                materialized = self._materialize_group(
                    parent_configuration=parent_configuration,
                    options_by_id=options_by_id,
                    option_ids=group,
                    radius=radius,
                    restart_slot=restart_slot,
                )
                if materialized is None:
                    continue
                child, receipts = materialized
                child_sha256 = typed_json_sha256(child)
                if child_sha256 in child_hashes:
                    continue
                child_hashes.add(child_sha256)
                component_sha256 = hashlib.sha256(
                    _canonical_json(list(group))
                ).hexdigest()
                receipt_sha256 = hashlib.sha256(
                    _canonical_json(list(receipts))
                ).hexdigest()
                option_digest = hashlib.sha256(
                    _OPTION_DOMAIN
                    + bytes.fromhex(self.definition_sha256)
                    + bytes.fromhex(parent_sha256)
                    + radius.to_bytes(4, "big")
                    + restart_slot.to_bytes(4, "big")
                    + bytes.fromhex(component_sha256)
                    + bytes.fromhex(child_sha256)
                    + bytes.fromhex(receipt_sha256)
                ).hexdigest()
                restarts.append(
                    FiniteVariationOption(
                        option_id=(
                            f"generic_restart.r{radius:03d}."
                            f"s{restart_slot:02d}.{option_digest[:24]}"
                        ),
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=child,
                        family=GENERIC_MULTISCALE_RESTART_FAMILY,
                        description=(
                            f"Apply {radius} pairwise-disjoint sealed atomic "
                            "actions selected by an outcome-blind parent-keyed "
                            "multiscale restart policy."
                        ),
                        metadata=tuple(
                            sorted(
                                (
                                    ("component_option_count", str(radius)),
                                    (
                                        "component_option_ids_sha256",
                                        component_sha256,
                                    ),
                                    ("composition_radius", str(radius)),
                                    (
                                        "construction_receipts_sha256",
                                        receipt_sha256,
                                    ),
                                    (
                                        VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY,
                                        f"multiscale_restart.r{radius:03d}",
                                    ),
                                    (VARIATION_OPERATOR_METADATA_KEY, "composite"),
                                    (
                                        VARIATION_SOURCE_METADATA_KEY,
                                        GENERIC_MULTISCALE_RESTART_SOURCE_ID,
                                    ),
                                    (
                                        VARIATION_SOURCE_MINIMUM_METADATA_KEY,
                                        str(self.evaluation_source_minimum),
                                    ),
                                    (
                                        "restart_policy_definition_sha256",
                                        self.definition_sha256,
                                    ),
                                    ("restart_radius", str(radius)),
                                    ("restart_slot", str(restart_slot)),
                                )
                            )
                        ),
                    )
                )
        return tuple(restarts)


__all__ = [
    "DEFAULT_RESTART_RADII",
    "DEFAULT_RESTARTS_PER_RADIUS",
    "GENERIC_MULTISCALE_RESTART_CATALOG_ID",
    "GENERIC_MULTISCALE_RESTART_CATALOG_VERSION",
    "GENERIC_MULTISCALE_RESTART_FAMILY",
    "GENERIC_MULTISCALE_RESTART_POLICY_ID",
    "GENERIC_MULTISCALE_RESTART_POLICY_VERSION",
    "GENERIC_MULTISCALE_RESTART_SOURCE_ID",
    "GenericMultiscaleRestartFiniteVariationCatalog",
]

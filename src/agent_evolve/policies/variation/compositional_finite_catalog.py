"""Bounded workload-neutral composition of sealed finite variation options.

Benchmark adapters may expose simple atomic choices because they are easy to
validate and explain.  Restricting every model-ranked candidate to one atomic
edit, however, creates an unnecessarily shallow search topology.  This
decorator preserves every base option and samples a bounded, outcome-blind set
of apparently disjoint two-option unions.  A sampled union is admitted only
after the engine materializes it and verifies that the result re-diffs to the
exact classified source effects.  The model still selects only opaque finite
option IDs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum

from agent_evolve.domain.finite_variation import (
    MAX_FINITE_VARIATION_OPTIONS,
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import FrozenJsonObject, typed_json_sha256
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchRecombiner,
    DisjointPatchRecombinationError,
)
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
)
from agent_evolve.ports.variation_catalog import FiniteVariationCatalog


COMPOSITIONAL_FINITE_CATALOG_POLICY_ID = "bounded_disjoint_finite_composition"
COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION = 2
COMPOSITE_OPTION_FAMILY = "composite_r2"
HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_ID = (
    "bounded_hierarchical_disjoint_finite_composition"
)
HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION = 2
COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY = "composition_selection_exposure"
COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY = "composition_required_proposals"
COMPOSITION_PREFERRED_PROPOSALS_METADATA_KEY = "composition_preferred_proposals"
COMPOSITION_CAPACITY_PROJECTED_METADATA_KEY = "composition_capacity_projected"
COMPOSITION_LEFT_OPTION_METADATA_KEY = "left_option_id"
COMPOSITION_RIGHT_OPTION_METADATA_KEY = "right_option_id"
_DEFINITION_DOMAIN = b"agent-evolve:bounded-disjoint-finite-composition:def:v2\x00"
_HIERARCHICAL_DEFINITION_DOMAIN = (
    b"agent-evolve:bounded-hierarchical-disjoint-finite-composition:def:v2\x00"
)
_PAIR_ORDER_DOMAIN = b"agent-evolve:bounded-disjoint-finite-composition:pair:v1\x00"
_OPTION_ID_DOMAIN = b"agent-evolve:bounded-disjoint-finite-composition:option:v1\x00"
_NON_REPLAY_SAFE_UNION = (
    "replayed union does not re-diff to the exact classified effects"
)


class CompositionSelectionExposure(str, Enum):
    """Closed prompt/selection treatment for materialized compositions."""

    FLAT = "flat"
    HIERARCHICAL_RANKED_UNION = "hierarchical_ranked_union"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _catalog_identity(catalog: FiniteVariationCatalog) -> tuple[str, int, str]:
    catalog_id = getattr(catalog, "catalog_id", None)
    catalog_version = getattr(catalog, "catalog_version", None)
    definition_sha256 = getattr(catalog, "definition_sha256", None)
    if type(catalog_id) is not str or not catalog_id:
        raise ValueError("base catalog must publish catalog_id")
    if type(catalog_version) is not int or catalog_version <= 0:
        raise ValueError("base catalog must publish a positive catalog_version")
    if (
        type(definition_sha256) is not str
        or len(definition_sha256) != 64
        or any(value not in "0123456789abcdef" for value in definition_sha256)
    ):
        raise ValueError("base catalog must publish a lowercase definition SHA-256")
    return catalog_id, catalog_version, definition_sha256


@dataclass(frozen=True, slots=True)
class BoundedCompositionalFiniteVariationCatalog:
    """Decorate any finite catalog with bounded replay-safe radius-two options."""

    base_catalog: FiniteVariationCatalog
    max_composite_options: int = 128
    selection_exposure: CompositionSelectionExposure = CompositionSelectionExposure.FLAT
    required_composite_proposals: int = 2
    catalog_id: str = field(init=False)
    catalog_version: int = field(init=False)
    definition_sha256: str = field(init=False)
    option_families: tuple[str, ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.base_catalog, FiniteVariationCatalog):
            raise TypeError("base_catalog must implement FiniteVariationCatalog")
        base_id, base_version, base_definition = _catalog_identity(self.base_catalog)
        if (
            type(self.max_composite_options) is not int
            or self.max_composite_options <= 0
            or self.max_composite_options >= MAX_FINITE_VARIATION_OPTIONS
        ):
            raise ValueError(
                "max_composite_options must lie in [1, MAX_FINITE_VARIATION_OPTIONS)"
            )
        if type(self.selection_exposure) is not CompositionSelectionExposure:
            raise TypeError("selection_exposure must be exact and closed")
        if (
            type(self.required_composite_proposals) is not int
            or not 1 <= self.required_composite_proposals < 8
        ):
            raise ValueError("required_composite_proposals must lie in [1, 8)")
        definition = {
            "schema_version": 1,
            "policy_id": COMPOSITIONAL_FINITE_CATALOG_POLICY_ID,
            "policy_version": COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION,
            "base_catalog": {
                "catalog_id": base_id,
                "catalog_version": base_version,
                "definition_sha256": base_definition,
            },
            "radius": 2,
            "retain_all_base_options": True,
            "max_composite_options": self.max_composite_options,
            "pair_prefilter": "disjoint_parent_relative_typed_patch_paths",
            "pair_sampling": "parent_keyed_hash_then_balanced_source_exposure",
            "materialization_attempt_bound": self.max_composite_options,
            "pair_admission": "exact_replay_union_and_rediff_to_source_effects",
            "failed_materialization": "deterministic_skip_without_replacement",
            "outcomes_consulted": False,
            "provider_fields_consulted": False,
        }
        object.__setattr__(self, "catalog_id", base_id)
        if self.selection_exposure is CompositionSelectionExposure.FLAT:
            catalog_version = base_version + COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION
            definition_sha256 = hashlib.sha256(
                _DEFINITION_DOMAIN + _canonical_json(definition)
            ).hexdigest()
        else:
            hierarchical_definition = {
                **definition,
                "policy_id": HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_ID,
                "policy_version": (
                    HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION
                ),
                "selection_exposure": self.selection_exposure.value,
                "required_composite_proposals": (self.required_composite_proposals),
                "realized_proposal_mix": (
                    "nearest-configured-exact-k8-capacity-projection"
                ),
                "capacity_projection_inputs": (
                    "current-parent-atomic-and-materialized-composite-counts"
                ),
                "capacity_projection_excludes": [
                    "outcomes",
                    "objective_values",
                    "provider_fields",
                    "workload_identifiers",
                ],
                "ranked_union_schema": (
                    "atomic-option-or-authenticated-radius-two-tuple"
                ),
                "model_materialization_authority": False,
            }
            catalog_version = (
                base_version
                + COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION
                + HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION
            )
            definition_sha256 = hashlib.sha256(
                _HIERARCHICAL_DEFINITION_DOMAIN
                + _canonical_json(hierarchical_definition)
            ).hexdigest()
        object.__setattr__(self, "catalog_version", catalog_version)
        object.__setattr__(self, "definition_sha256", definition_sha256)
        base_families = getattr(self.base_catalog, "option_families", None)
        if base_families is not None:
            if type(base_families) is not tuple or not base_families:
                raise TypeError("base option_families must be a non-empty tuple")
            object.__setattr__(
                self,
                "option_families",
                tuple(sorted({*base_families, COMPOSITE_OPTION_FAMILY})),
            )

    @staticmethod
    def _balanced_pairs(
        ordered_pairs: tuple[tuple[str, str], ...],
        *,
        limit: int,
    ) -> tuple[tuple[str, str], ...]:
        """Bound source exposure before reusing an atomic option."""

        if limit <= 0 or not ordered_pairs:
            return ()
        exposure: dict[str, int] = {}
        selected: list[tuple[str, str]] = []
        selected_set: set[tuple[str, str]] = set()
        tier = 0
        while len(selected) < limit:
            progressed = False
            for pair in ordered_pairs:
                if pair in selected_set:
                    continue
                left, right = pair
                if exposure.get(left, 0) > tier or exposure.get(right, 0) > tier:
                    continue
                selected.append(pair)
                selected_set.add(pair)
                exposure[left] = exposure.get(left, 0) + 1
                exposure[right] = exposure.get(right, 0) + 1
                progressed = True
                if len(selected) == limit:
                    break
            if len(selected) == limit or len(selected_set) == len(ordered_pairs):
                break
            tier += 1
            if not progressed and tier > len(ordered_pairs):  # pragma: no cover
                raise RuntimeError("balanced pair selection failed to progress")
        return tuple(selected)

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        self.__post_init__()
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("parent_configuration must be an exact FrozenJsonObject")
        base_options = self.base_catalog.options(parent_configuration)
        if type(base_options) is not tuple or not base_options:
            raise ValueError("base catalog must return a non-empty exact tuple")
        base_contract = FiniteVariationContract(
            catalog_id=self.base_catalog.catalog_id,
            catalog_version=self.base_catalog.catalog_version,
            catalog_definition_sha256=self.base_catalog.definition_sha256,
            parent_configuration=parent_configuration,
            options=base_options,
        )
        remaining_capacity = MAX_FINITE_VARIATION_OPTIONS - len(base_options)
        composite_limit = min(self.max_composite_options, remaining_capacity)
        if composite_limit <= 0:
            return base_options

        option_by_id = {value.option_id: value for value in base_options}
        eligible_pairs = pairwise_disjoint_parent_patch_pairs(
            base_contract,
            tuple(option_by_id),
        )
        parent_sha256 = typed_json_sha256(parent_configuration)
        ordered_pairs = tuple(
            sorted(
                eligible_pairs,
                key=lambda pair: (
                    hashlib.sha256(
                        _PAIR_ORDER_DOMAIN
                        + bytes.fromhex(self.definition_sha256)
                        + bytes.fromhex(parent_sha256)
                        + bytes.fromhex(option_by_id[pair[0]].identity_sha256)
                        + bytes.fromhex(option_by_id[pair[1]].identity_sha256)
                    ).digest(),
                    pair,
                ),
            )
        )
        selected_pairs = self._balanced_pairs(
            ordered_pairs,
            limit=min(composite_limit, len(ordered_pairs)),
        )

        ancestor_id = CandidateId("candidate_composition_parent")
        recombiner = DisjointPatchRecombiner()
        composites: list[FiniteVariationOption] = []
        child_hashes = {value.child_configuration_sha256 for value in base_options}
        for ordinal, (left_id, right_id) in enumerate(selected_pairs, start=1):
            left = option_by_id[left_id]
            right = option_by_id[right_id]
            try:
                materialization = recombiner.materialize(
                    ancestor=parent_configuration,
                    ancestor_candidate_id=ancestor_id,
                    left=left.child_configuration,
                    left_candidate_id=CandidateId(
                        f"candidate_composition_left_{ordinal:04d}"
                    ),
                    right=right.child_configuration,
                    right_candidate_id=CandidateId(
                        f"candidate_composition_right_{ordinal:04d}"
                    ),
                    target_candidate_id=CandidateId(
                        f"candidate_composition_target_{ordinal:04d}"
                    ),
                )
            except DisjointPatchRecombinationError as error:
                # Repeated sequence values can make two locally disjoint
                # source patches interact under the canonical global diff.
                # Such a pair is not replay-safe even though it passed the
                # cheap path prefilter.  The bounded sample is outcome-blind,
                # so deterministic rejection needs neither repair nor refill.
                if str(error) != _NON_REPLAY_SAFE_UNION:
                    raise
                continue
            child_sha256 = typed_json_sha256(materialization.configuration)
            if child_sha256 in child_hashes:
                continue
            child_hashes.add(child_sha256)
            option_digest = hashlib.sha256(
                _OPTION_ID_DOMAIN
                + bytes.fromhex(self.definition_sha256)
                + bytes.fromhex(parent_sha256)
                + bytes.fromhex(left.identity_sha256)
                + bytes.fromhex(right.identity_sha256)
                + bytes.fromhex(materialization.receipt_sha256)
            ).hexdigest()
            composites.append(
                FiniteVariationOption(
                    option_id=f"compose.r02.{option_digest[:32]}",
                    parent_configuration_sha256=parent_sha256,
                    child_configuration=materialization.configuration,
                    family=COMPOSITE_OPTION_FAMILY,
                    description=(
                        "Apply both sealed parent-relative options "
                        f"{left_id} and {right_id}."
                    ),
                    metadata=tuple(
                        sorted(
                            (
                                (
                                    "composition_policy_definition_sha256",
                                    self.definition_sha256,
                                ),
                                ("composition_radius", "2"),
                                (COMPOSITION_LEFT_OPTION_METADATA_KEY, left_id),
                                ("left_option_identity_sha256", left.identity_sha256),
                                (COMPOSITION_RIGHT_OPTION_METADATA_KEY, right_id),
                                (
                                    "right_option_identity_sha256",
                                    right.identity_sha256,
                                ),
                            )
                            + (
                                ()
                                if self.selection_exposure
                                is CompositionSelectionExposure.FLAT
                                else (
                                    (
                                        COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY,
                                        self.selection_exposure.value,
                                    ),
                                    (
                                        COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY,
                                        str(self.required_composite_proposals),
                                    ),
                                )
                            )
                        )
                    ),
                )
            )
        if (
            self.selection_exposure
            is CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION
            and len(base_options) + len(composites) >= 8
        ):
            # The configured count is a search preference, not permission to
            # publish an impossible output schema.  Near a finite-domain
            # boundary a parent can expose fewer than ``8 - preferred`` legal
            # atomic moves.  Project to the closest exact-K8 composition using
            # only the current, engine-materialized action capacities.  This
            # is workload-, objective-, provider-, and outcome-independent.
            minimum_composites = max(0, 8 - len(base_options))
            maximum_composites = min(7, len(composites))
            effective_composites = max(
                minimum_composites,
                min(self.required_composite_proposals, maximum_composites),
            )
            projected = effective_composites != self.required_composite_proposals
            adjusted: list[FiniteVariationOption] = []
            for option in composites:
                metadata = dict(option.metadata)
                metadata[COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY] = str(
                    effective_composites
                )
                metadata[COMPOSITION_PREFERRED_PROPOSALS_METADATA_KEY] = str(
                    self.required_composite_proposals
                )
                metadata[COMPOSITION_CAPACITY_PROJECTED_METADATA_KEY] = (
                    "true" if projected else "false"
                )
                adjusted.append(
                    option
                    if tuple(sorted(metadata.items())) == option.metadata
                    else replace(option, metadata=tuple(sorted(metadata.items())))
                )
            composites = adjusted
        return (*base_options, *composites)


__all__ = [
    "COMPOSITION_LEFT_OPTION_METADATA_KEY",
    "COMPOSITION_CAPACITY_PROJECTED_METADATA_KEY",
    "COMPOSITION_PREFERRED_PROPOSALS_METADATA_KEY",
    "COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY",
    "COMPOSITION_RIGHT_OPTION_METADATA_KEY",
    "COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY",
    "COMPOSITIONAL_FINITE_CATALOG_POLICY_ID",
    "COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION",
    "COMPOSITE_OPTION_FAMILY",
    "HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_ID",
    "HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION",
    "BoundedCompositionalFiniteVariationCatalog",
    "CompositionSelectionExposure",
]

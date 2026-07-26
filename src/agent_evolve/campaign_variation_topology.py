"""One workload-neutral switch for atomic and radius-two search topology.

Workload adapters own their atomic finite-action catalog.  This configuration
owns only how AgentEvolve exposes that catalog to its selector: unchanged,
legacy flat radius-two options, or an explicit ranked union with a bounded
composite proposal stratum.  The same object is usable by every workload and
is safe to construct before any provider or evaluator access.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from agent_evolve.policies.variation.compositional_finite_catalog import (
    COMPOSITION_LEFT_OPTION_METADATA_KEY,
    COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY,
    COMPOSITION_RIGHT_OPTION_METADATA_KEY,
    COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY,
    BoundedCompositionalFiniteVariationCatalog,
    CompositionSelectionExposure,
)
from agent_evolve.ports.variation_catalog import FiniteVariationCatalog
from agent_evolve.domain.finite_variation import FiniteVariationContract


_POOL_SUPPORT_DOMAIN = b"agent-evolve:hierarchical-pool-support:v1\x00"


class CampaignVariationTopologyMode(str, Enum):
    """Closed search-topology treatments shared across workload runners."""

    ATOMIC = "atomic"
    FLAT_R2 = "flat_r2"
    HIERARCHICAL_R2 = "hierarchical_r2"


@dataclass(frozen=True, slots=True)
class CampaignVariationTopology:
    """Validated topology configuration applied behind the catalog port."""

    mode: CampaignVariationTopologyMode = CampaignVariationTopologyMode.ATOMIC
    max_composite_options: int = 0
    required_composite_proposals: int = 0

    def __post_init__(self) -> None:
        if type(self.mode) is not CampaignVariationTopologyMode:
            raise TypeError("mode must be an exact CampaignVariationTopologyMode")
        if type(self.max_composite_options) is not int:
            raise TypeError("max_composite_options must be an exact integer")
        if type(self.required_composite_proposals) is not int:
            raise TypeError("required_composite_proposals must be an exact integer")
        if self.mode is CampaignVariationTopologyMode.ATOMIC:
            if self.max_composite_options != 0:
                raise ValueError("atomic topology cannot materialize composites")
            if self.required_composite_proposals != 0:
                raise ValueError("atomic topology cannot require composites")
            return
        if not 1 <= self.max_composite_options < 1024:
            raise ValueError("radius-two composite bound must lie in [1, 1024)")
        if self.mode is CampaignVariationTopologyMode.FLAT_R2:
            if self.required_composite_proposals != 0:
                raise ValueError("flat radius-two topology has no proposal stratum")
            return
        if not 1 <= self.required_composite_proposals < 8:
            raise ValueError(
                "hierarchical required composite proposals must lie in [1, 8)"
            )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        *,
        default_mode: CampaignVariationTopologyMode = (
            CampaignVariationTopologyMode.ATOMIC
        ),
        default_max_composite_options: int = 128,
        default_required_composite_proposals: int = 2,
    ) -> "CampaignVariationTopology":
        """Parse one identical strict environment contract in every launcher."""

        if not isinstance(environment, Mapping):
            raise TypeError("environment must implement Mapping[str, str]")
        if type(default_mode) is not CampaignVariationTopologyMode:
            raise TypeError("default_mode must be exact")
        raw_max = environment.get("AGENT_EVOLVE_COMPOSITE_OPTION_COUNT")
        parsed_maximum: int | None = None
        if raw_max is not None:
            if not raw_max.isascii() or not raw_max.isdigit():
                raise ValueError(
                    "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT must be decimal digits"
                )
            parsed_maximum = int(raw_max)

        raw_required = environment.get("AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS")
        parsed_required: int | None = None
        if raw_required is not None:
            if not raw_required.isascii() or not raw_required.isdigit():
                raise ValueError(
                    "AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS must be decimal digits"
                )
            parsed_required = int(raw_required)

        legacy_exposure = environment.get("AGENT_EVOLVE_COMPOSITION_SELECTION_EXPOSURE")
        if legacy_exposure is not None:
            # Validate the legacy spelling before using it for compatibility
            # inference.  This keeps all launchers fail-closed on typos.
            CompositionSelectionExposure(legacy_exposure)

        raw_mode = environment.get("AGENT_EVOLVE_VARIATION_TOPOLOGY")
        if raw_mode is None:
            # Preserve legacy BOiLS launch commands while moving every runner
            # to the same explicit topology object.  A positive legacy bound
            # meant flat radius-two unless the old hierarchical exposure was
            # also selected.
            if legacy_exposure == (
                CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value
            ):
                mode = CampaignVariationTopologyMode.HIERARCHICAL_R2
            elif parsed_maximum is not None and parsed_maximum > 0:
                mode = CampaignVariationTopologyMode.FLAT_R2
            else:
                mode = default_mode
        else:
            mode = CampaignVariationTopologyMode(raw_mode)

        maximum = (
            parsed_maximum
            if parsed_maximum is not None
            else (
                0
                if mode is CampaignVariationTopologyMode.ATOMIC
                else default_max_composite_options
            )
        )
        required = (
            parsed_required
            if parsed_required is not None
            else (
                default_required_composite_proposals
                if mode is CampaignVariationTopologyMode.HIERARCHICAL_R2
                else 0
            )
        )
        if legacy_exposure is not None:
            expected = {
                CampaignVariationTopologyMode.ATOMIC: "flat",
                CampaignVariationTopologyMode.FLAT_R2: "flat",
                CampaignVariationTopologyMode.HIERARCHICAL_R2: (
                    CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value
                ),
            }[mode]
            if legacy_exposure != expected:
                raise ValueError(
                    "legacy composition exposure conflicts with variation topology"
                )
        return cls(
            mode=mode,
            max_composite_options=maximum,
            required_composite_proposals=required,
        )

    @property
    def hierarchical_composition_required_proposals(self) -> int | None:
        self.__post_init__()
        return (
            self.required_composite_proposals
            if self.mode is CampaignVariationTopologyMode.HIERARCHICAL_R2
            else None
        )

    @property
    def selection_exposure(self) -> CompositionSelectionExposure:
        self.__post_init__()
        return (
            CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION
            if self.mode is CampaignVariationTopologyMode.HIERARCHICAL_R2
            else CompositionSelectionExposure.FLAT
        )

    def decorate(self, catalog: FiniteVariationCatalog) -> FiniteVariationCatalog:
        """Return the selected topology behind the unchanged catalog protocol."""

        self.__post_init__()
        if not isinstance(catalog, FiniteVariationCatalog):
            raise TypeError("catalog must implement FiniteVariationCatalog")
        if self.mode is CampaignVariationTopologyMode.ATOMIC:
            return catalog
        return BoundedCompositionalFiniteVariationCatalog(
            catalog,
            max_composite_options=self.max_composite_options,
            selection_exposure=self.selection_exposure,
            required_composite_proposals=(
                self.required_composite_proposals
                if self.mode is CampaignVariationTopologyMode.HIERARCHICAL_R2
                else 1
            ),
        )

    def prompt_metadata_keys(
        self,
        base_keys: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return the minimal metadata projection needed by this topology."""

        self.__post_init__()
        if type(base_keys) is not tuple or any(
            type(value) is not str or not value for value in base_keys
        ):
            raise TypeError("base_keys must be an exact tuple of non-empty strings")
        if self.mode is not CampaignVariationTopologyMode.HIERARCHICAL_R2:
            return tuple(sorted(set(base_keys)))
        return tuple(
            sorted(
                {
                    *base_keys,
                    COMPOSITION_LEFT_OPTION_METADATA_KEY,
                    COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY,
                    COMPOSITION_RIGHT_OPTION_METADATA_KEY,
                    COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY,
                }
            )
        )

    def required_candidate_pool_option_ids(
        self,
        contract: FiniteVariationContract,
    ) -> tuple[str, ...]:
        """Reserve model-blind crossover support in a bounded common pool."""

        self.__post_init__()
        if type(contract) is not FiniteVariationContract:
            raise TypeError("contract must be exact FiniteVariationContract")
        contract.__post_init__()
        if self.mode is not CampaignVariationTopologyMode.HIERARCHICAL_R2:
            return ()
        composites = tuple(
            option.option_id
            for option in contract.options
            if dict(option.metadata).get(
                COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY
            )
            == CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value
        )
        if len(composites) < self.required_composite_proposals:
            raise ValueError("hierarchical contract lacks required crossover support")
        contract_sha256 = contract.identity_sha256
        return tuple(
            sorted(
                sorted(
                    composites,
                    key=lambda option_id: (
                        hashlib.sha256(
                            _POOL_SUPPORT_DOMAIN
                            + bytes.fromhex(contract_sha256)
                            + option_id.encode("ascii", errors="strict")
                        ).digest(),
                        option_id,
                    ),
                )[: self.required_composite_proposals]
            )
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "mode": self.mode.value,
            "max_composite_options": self.max_composite_options,
            "required_composite_proposals": self.required_composite_proposals,
            "required_composite_proposals_semantics": (
                "preferred_then_nearest_exact_k8_capacity_projection"
                if self.mode is CampaignVariationTopologyMode.HIERARCHICAL_R2
                else "not_applicable"
            ),
            "selection_exposure": self.selection_exposure.value,
            "provider_materialization_authority": False,
            "outcomes_consulted": False,
        }


__all__ = [
    "CampaignVariationTopology",
    "CampaignVariationTopologyMode",
]

"""Outcome-blind multiscale restart options for the BOiLS sequence space."""

from __future__ import annotations

import hashlib
import json

from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    EVALUATION_SOURCE_METADATA_KEY,
    EVALUATION_SOURCE_MINIMUM_METADATA_KEY,
)

from .actions import ACTION_IDS, SEQUENCE_LENGTH, CandidateConfig


GLOBAL_RESTART_CATALOG_ID = "boils_abc_global_restart"
GLOBAL_RESTART_CATALOG_VERSION = 1
GLOBAL_RESTART_FAMILY = "global_restart"
GLOBAL_RESTART_SOURCE_ID = "global_restart"
GLOBAL_RESTART_RADII = (5, 10, 15, 20)
GLOBAL_RESTARTS_PER_RADIUS = 4
_DEFINITION_DOMAIN = b"agent-evolve:boils-abc-global-restart:def:v1\x00"
_POSITION_DOMAIN = b"agent-evolve:boils-abc-global-restart:position:v1\x00"
_ACTION_DOMAIN = b"agent-evolve:boils-abc-global-restart:action:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


_DEFINITION = {
    "schema_version": 1,
    "catalog_id": GLOBAL_RESTART_CATALOG_ID,
    "catalog_version": GLOBAL_RESTART_CATALOG_VERSION,
    "source_id": GLOBAL_RESTART_SOURCE_ID,
    "sequence_length": SEQUENCE_LENGTH,
    "action_ids": list(ACTION_IDS),
    "hamming_radii": list(GLOBAL_RESTART_RADII),
    "restarts_per_radius": GLOBAL_RESTARTS_PER_RADIUS,
    "position_selection": "parent-keyed-hash-without-replacement",
    "replacement_selection": "parent-keyed-hash-over-non-parent-actions",
    "evaluation_source_minimum": 1,
    "outcomes_consulted": False,
    "known_incumbent_configurations_consulted": False,
}
GLOBAL_RESTART_CATALOG_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_DOMAIN + _canonical_json(_DEFINITION)
).hexdigest()


class BoilsGlobalRestartVariationCatalog:
    """Expose sealed whole-sequence jumps at four outcome-blind Hamming scales."""

    catalog_id = GLOBAL_RESTART_CATALOG_ID
    catalog_version = GLOBAL_RESTART_CATALOG_VERSION
    definition_sha256 = GLOBAL_RESTART_CATALOG_DEFINITION_SHA256
    option_families = (GLOBAL_RESTART_FAMILY,)

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("BOiLS restart catalog requires frozen typed JSON")
        parent = CandidateConfig.model_validate(
            thaw_json(parent_configuration),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent_digest = bytes.fromhex(parent_sha256)
        options: list[FiniteVariationOption] = []

        for radius in GLOBAL_RESTART_RADII:
            for restart_slot in range(GLOBAL_RESTARTS_PER_RADIUS):
                slot_bytes = radius.to_bytes(2, "big") + restart_slot.to_bytes(
                    2, "big"
                )
                positions = tuple(
                    sorted(
                        range(SEQUENCE_LENGTH),
                        key=lambda position: (
                            hashlib.sha256(
                                _POSITION_DOMAIN
                                + parent_digest
                                + slot_bytes
                                + position.to_bytes(2, "big")
                            ).digest(),
                            position,
                        ),
                    )[:radius]
                )
                child_sequence = list(parent.sequence)
                for position in positions:
                    alternatives = tuple(
                        action_id
                        for action_id in ACTION_IDS
                        if action_id != parent.sequence[position]
                    )
                    action_digest = hashlib.sha256(
                        _ACTION_DOMAIN
                        + parent_digest
                        + slot_bytes
                        + position.to_bytes(2, "big")
                    ).digest()
                    child_sequence[position] = alternatives[
                        int.from_bytes(action_digest[:8], "big") % len(alternatives)
                    ]
                child = CandidateConfig.model_validate(
                    {"sequence": child_sequence},
                    strict=True,
                    by_alias=False,
                    by_name=True,
                )
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise AssertionError("BOiLS restart did not freeze to an object")
                options.append(
                    FiniteVariationOption(
                        option_id=(
                            f"boils_abc.restart.r{radius:02d}.s{restart_slot:02d}"
                        ),
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=frozen_child,
                        family=GLOBAL_RESTART_FAMILY,
                        description=(
                            f"Outcome-blind global restart changing exactly {radius} "
                            "of 20 sequence positions."
                        ),
                        metadata=(
                            ("changed_coordinate_count", str(radius)),
                            (
                                EVALUATION_SOURCE_METADATA_KEY,
                                GLOBAL_RESTART_SOURCE_ID,
                            ),
                            (EVALUATION_SOURCE_MINIMUM_METADATA_KEY, "1"),
                            (
                                "restart_definition_sha256",
                                GLOBAL_RESTART_CATALOG_DEFINITION_SHA256,
                            ),
                            ("restart_scale", f"{radius:02d}_of_{SEQUENCE_LENGTH:02d}"),
                        ),
                    )
                )
        children = tuple(value.child_configuration_sha256 for value in options)
        if len(set(children)) != len(children):  # pragma: no cover - hash design.
            raise RuntimeError("BOiLS restart generator produced duplicate children")
        return tuple(options)


__all__ = [
    "GLOBAL_RESTART_CATALOG_DEFINITION_SHA256",
    "GLOBAL_RESTART_CATALOG_ID",
    "GLOBAL_RESTART_CATALOG_VERSION",
    "GLOBAL_RESTART_FAMILY",
    "GLOBAL_RESTART_RADII",
    "GLOBAL_RESTART_SOURCE_ID",
    "GLOBAL_RESTARTS_PER_RADIUS",
    "BoilsGlobalRestartVariationCatalog",
]

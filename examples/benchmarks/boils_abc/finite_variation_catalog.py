"""Public-API finite action choices for the BOiLS sequence benchmark.

Each option materializes one complete, typed length-20 child.  The catalog is
bound to an exact frozen parent: the model may select an option identifier, but
it never authors an ABC command or an unvalidated partial configuration.
"""

from __future__ import annotations

import hashlib
import json

from agent_evolve.agentic import (
    FiniteVariationOption,
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

from .actions import ACTION_COMMANDS, ACTION_IDS, SEQUENCE_LENGTH, CandidateConfig
from .variation_catalog import (
    ACTION_DEFINITION_SHA256,
    ACTION_FAMILIES,
    CATALOG_SOURCE_SHA256,
)


FINITE_CATALOG_ID = "boils_abc_single_action"
FINITE_CATALOG_VERSION = 1
FINITE_CATALOG_SCHEMA_ID = "boils_abc_finite_single_action_catalog_v1"
_DEFINITION_HASH_DOMAIN = b"agent-evolve:boils-abc-finite-catalog:v1\x00"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


_FINITE_CATALOG_DEFINITION = {
    "schema_id": FINITE_CATALOG_SCHEMA_ID,
    "catalog_id": FINITE_CATALOG_ID,
    "catalog_version": FINITE_CATALOG_VERSION,
    "sequence_length": SEQUENCE_LENGTH,
    "option_order": "position_major_then_action_ids_excluding_parent_value",
    "materialization": "exact_single_action_replacement_no_clipping",
    "source_catalog_sha256": CATALOG_SOURCE_SHA256,
    "actions": [
        {
            "action_id": action_id,
            "commands": list(ACTION_COMMANDS[action_id]),
            "definition_sha256": ACTION_DEFINITION_SHA256[action_id],
            "family": ACTION_FAMILIES[action_id],
        }
        for action_id in ACTION_IDS
    ],
}
FINITE_CATALOG_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_HASH_DOMAIN + _canonical_json_bytes(_FINITE_CATALOG_DEFINITION)
).hexdigest()


def _validated_parent(
    parent_configuration: FrozenJsonObject,
) -> tuple[CandidateConfig, str]:
    if type(parent_configuration) is not FrozenJsonObject:
        raise TypeError("BOiLS finite catalog requires an exact FrozenJsonObject")
    parsed = CandidateConfig.model_validate(
        thaw_json(parent_configuration),
        strict=True,
        by_alias=False,
        by_name=True,
    )
    return parsed, typed_json_sha256(parent_configuration)


class BoilsFiniteVariationCatalog:
    """Materialize every exact one-position action replacement.

    Enumeration is position-major and follows the frozen ``ACTION_IDS`` order
    within each position.  The current value is omitted, yielding exactly 200
    options for every valid 20-position parent.  There is no clipping,
    coercion, or repair step.
    """

    catalog_id = FINITE_CATALOG_ID
    catalog_version = FINITE_CATALOG_VERSION
    definition_sha256 = FINITE_CATALOG_DEFINITION_SHA256

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent, parent_sha256 = _validated_parent(parent_configuration)
        parent_sequence = tuple(parent.sequence)
        options: list[FiniteVariationOption] = []

        for position, current_action in enumerate(parent_sequence):
            for action_id in ACTION_IDS:
                if action_id == current_action:
                    continue
                child_sequence = list(parent_sequence)
                child_sequence[position] = action_id
                # Strict validation is intentionally repeated at the
                # materialization boundary.  It proves that the full child,
                # not merely the selected action, satisfies the benchmark
                # schema before the option reaches AgentEvolve.
                child = CandidateConfig.model_validate(
                    {"sequence": child_sequence},
                    strict=True,
                    by_alias=False,
                    by_name=True,
                )
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise RuntimeError("BOiLS candidate schema did not freeze as an object")

                commands_json = _canonical_json_bytes(
                    list(ACTION_COMMANDS[action_id])
                ).decode("ascii")
                options.append(
                    FiniteVariationOption(
                        option_id=f"boils_abc.p{position:02d}.{action_id}",
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=frozen_child,
                        family=ACTION_FAMILIES[action_id],
                        description=(
                            f"At position {position:02d}, replace {current_action} "
                            f"with {action_id}."
                        ),
                        metadata=(
                            ("abc_commands_json", commands_json),
                            (
                                "action_definition_sha256",
                                ACTION_DEFINITION_SHA256[action_id],
                            ),
                            (
                                "catalog_definition_sha256",
                                FINITE_CATALOG_DEFINITION_SHA256,
                            ),
                            ("position", f"{position:02d}"),
                            ("replacement_action", action_id),
                            ("source_catalog_sha256", CATALOG_SOURCE_SHA256),
                        ),
                    )
                )
        return tuple(options)


__all__ = [
    "FINITE_CATALOG_DEFINITION_SHA256",
    "FINITE_CATALOG_ID",
    "FINITE_CATALOG_SCHEMA_ID",
    "FINITE_CATALOG_VERSION",
    "BoilsFiniteVariationCatalog",
]

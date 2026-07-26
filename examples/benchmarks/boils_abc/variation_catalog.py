"""Finite atomic-variation catalog for the BOiLS action sequence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, canonical_path_bytes
from agent_evolve.domain.typed_json import canonical_typed_json_bytes
from agent_evolve.domain.variation_space import AtomicEditOption
from examples.benchmarks.boils_abc.actions import (
    ACTION_COMMANDS,
    ACTION_IDS,
    SEQUENCE_LENGTH,
    CandidateConfig,
)


CATALOG_SCHEMA_ID = "boils_abc_atomic_variation_catalog_v1"

# Source-derived, performance-neutral transformation families frozen in the
# BOiLS action-card source audit.  Families describe implementation semantics;
# they contain no observed objective information.
ACTION_FAMILIES: Mapping[str, str] = MappingProxyType(
    {
        "rewrite": "aig_rewrite",
        "rewrite_z": "aig_rewrite",
        "refactor": "aig_refactor",
        "refactor_z": "aig_refactor",
        "resub": "aig_resubstitute",
        "resub_z": "aig_resubstitute",
        "balance": "aig_balance",
        "fraig": "aig_functional_reduce",
        "sopb": "gia_sop_balance",
        "blut": "gia_lut_balance",
        "dsdb": "gia_dsd_balance",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _source_definition() -> dict[str, object]:
    return {
        "schema_id": CATALOG_SCHEMA_ID,
        "actions": [
            {
                "action_id": action_id,
                "commands": list(ACTION_COMMANDS[action_id]),
                "family": ACTION_FAMILIES[action_id],
            }
            for action_id in ACTION_IDS
        ],
    }


CATALOG_SOURCE_SHA256 = hashlib.sha256(
    _canonical_json_bytes(_source_definition())
).hexdigest()


def _action_definition_sha256(action_id: str) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_id": CATALOG_SCHEMA_ID,
                "action_id": action_id,
                "commands": list(ACTION_COMMANDS[action_id]),
                "family": ACTION_FAMILIES[action_id],
            }
        )
    ).hexdigest()


ACTION_DEFINITION_SHA256: Mapping[str, str] = MappingProxyType(
    {
        action_id: _action_definition_sha256(action_id)
        for action_id in ACTION_IDS
    }
)


def _option_binding_sha256(
    *,
    path: JsonPath,
    replacement: str,
    family: str,
    action_definition_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"boils-abc:atomic-option:v1\x00")
    digest.update(canonical_path_bytes(path))
    digest.update(canonical_typed_json_bytes(replacement))
    digest.update(family.encode("ascii"))
    digest.update(bytes.fromhex(action_definition_sha256))
    return digest.hexdigest()


class BoilsAtomicVariationCatalog:
    """Enumerate every one-coordinate action replacement in stable order.

    Options are path-major and follow ``ACTION_IDS`` within each path, omitting
    the parent's current value.  Each option ID contains a digest over its path,
    replacement, family, and exact ABC command definition; downstream evidence
    therefore cannot collide across positions or catalog revisions.
    """

    schema_id = CATALOG_SCHEMA_ID
    source_sha256 = CATALOG_SOURCE_SHA256

    def options(
        self,
        parent: EvolutionCandidate,
    ) -> tuple[AtomicEditOption, ...]:
        if type(parent) is not EvolutionCandidate:
            raise TypeError("parent must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(parent)
        parsed = CandidateConfig.model_validate(
            parent.configuration_dict,
            strict=True,
            by_alias=False,
            by_name=True,
        )
        sequence = tuple(parsed.sequence)
        if len(sequence) != SEQUENCE_LENGTH:  # pragma: no cover - model invariant.
            raise ValueError("BOiLS parent has the wrong sequence length")

        options: list[AtomicEditOption] = []
        for index, current_action in enumerate(sequence):
            path = JsonPath((ObjectKey("sequence"), ArrayIndex(index)))
            for action_id in ACTION_IDS:
                if action_id == current_action:
                    continue
                action_hash = ACTION_DEFINITION_SHA256[action_id]
                binding = _option_binding_sha256(
                    path=path,
                    replacement=action_id,
                    family=ACTION_FAMILIES[action_id],
                    action_definition_sha256=action_hash,
                )
                commands_json = _canonical_json_bytes(
                    list(ACTION_COMMANDS[action_id])
                ).decode("ascii")
                options.append(
                    AtomicEditOption(
                        option_id=(
                            f"boils_abc.sequence_{index:02d}.{action_id}.{binding}"
                        ),
                        path=path,
                        replacement=action_id,
                        family=ACTION_FAMILIES[action_id],
                        metadata=(
                            ("abc_commands_json", commands_json),
                            ("action_definition_sha256", action_hash),
                            ("catalog_source_sha256", CATALOG_SOURCE_SHA256),
                        ),
                    )
                )
        return tuple(options)


if tuple(ACTION_FAMILIES) != ACTION_IDS:  # pragma: no cover - import invariant.
    raise RuntimeError("BOiLS action-family order diverged from ACTION_IDS")
if tuple(ACTION_DEFINITION_SHA256) != ACTION_IDS:  # pragma: no cover
    raise RuntimeError("BOiLS action-definition order diverged from ACTION_IDS")


__all__ = [
    "ACTION_DEFINITION_SHA256",
    "ACTION_FAMILIES",
    "CATALOG_SCHEMA_ID",
    "CATALOG_SOURCE_SHA256",
    "BoilsAtomicVariationCatalog",
]

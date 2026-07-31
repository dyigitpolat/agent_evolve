"""Strict BOiLS action-space and candidate configuration boundary."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from types import MappingProxyType
from typing import Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field


SEQUENCE_LENGTH = 20
CONFIG_SCHEMA_ID = "boils_abc_action_sequence_v1"

ActionId: TypeAlias = Literal[
    "rewrite",
    "rewrite_z",
    "refactor",
    "refactor_z",
    "resub",
    "resub_z",
    "balance",
    "fraig",
    "sopb",
    "blut",
    "dsdb",
]

ACTION_IDS: tuple[str, ...] = get_args(ActionId)

# These are the exact BOiLS extended action semantics at commit
# ee6112d39d1a9e9703fecaf9057193e1ec9dae72.  Composite GIA actions retain the
# required &get/&put bridge.  Candidate text is never used as an ABC command.
ACTION_COMMANDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "rewrite": ("rewrite",),
        "rewrite_z": ("rewrite -z",),
        "refactor": ("refactor",),
        "refactor_z": ("refactor -z",),
        "resub": ("resub",),
        "resub_z": ("resub -z",),
        "balance": ("balance",),
        "fraig": ("fraig",),
        "sopb": ("&get -n", "&sopb", "&put"),
        "blut": ("&get -n", "&blut", "&put"),
        "dsdb": ("&get -n", "&dsdb", "&put"),
    }
)

_BOILS_Q0_SEQUENCE: tuple[ActionId, ...] = (
    "balance",
    "rewrite",
    "refactor",
    "balance",
    "rewrite",
    "rewrite_z",
    "balance",
    "refactor_z",
    "rewrite_z",
    "balance",
)
DEFAULT_ACTION_SEQUENCE: tuple[ActionId, ...] = (
    _BOILS_Q0_SEQUENCE + _BOILS_Q0_SEQUENCE
)


class CandidateConfig(BaseModel):
    """Exactly one length-20 sequence over the allowlisted BOiLS vocabulary."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
    )

    # A literal default rather than a factory. The two behave identically --
    # pydantic copies a mutable default per instance -- but a lambda is opaque
    # behavioural state that cannot be bound by a schema digest, so a candidate
    # model carrying one cannot be sealed for provider-free replay. The
    # generative path seals the schema it emitted against; this makes that
    # possible without changing a single candidate.
    sequence: list[ActionId] = Field(
        default=list(DEFAULT_ACTION_SEQUENCE),
        min_length=SEQUENCE_LENGTH,
        max_length=SEQUENCE_LENGTH,
        description=(
            "Exactly 20 BOiLS action IDs. Repetition and order are semantically "
            "meaningful; raw ABC commands are forbidden."
        ),
    )


def normalize_candidate(config: object) -> tuple[ActionId, ...]:
    """Revalidate and return an immutable candidate projection."""

    parsed = CandidateConfig.model_validate(config)
    return tuple(parsed.sequence)


def expand_abc_commands(config: object) -> tuple[str, ...]:
    """Expand a typed candidate exclusively through the frozen allowlist."""

    sequence = normalize_candidate(config)
    return tuple(
        command
        for action_id in sequence
        for command in ACTION_COMMANDS[action_id]
    )


def canonical_config_bytes(config: object) -> bytes:
    """Return a stable, schema-scoped JSON encoding for cache/provenance keys."""

    sequence = normalize_candidate(config)
    payload = {
        "schema_id": CONFIG_SCHEMA_ID,
        "sequence": list(sequence),
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def config_sha256(config: object) -> str:
    """Hash the canonical candidate representation deterministically."""

    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


if tuple(ACTION_COMMANDS) != ACTION_IDS:  # pragma: no cover - import invariant
    raise RuntimeError("BOiLS action literal and command allowlist orders diverged")
if len(DEFAULT_ACTION_SEQUENCE) != SEQUENCE_LENGTH:  # pragma: no cover
    raise RuntimeError("default BOiLS sequence has the wrong length")

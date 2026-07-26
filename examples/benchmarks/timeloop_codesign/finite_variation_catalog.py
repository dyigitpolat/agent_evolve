"""Parent-bound finite hardware moves for Timeloop co-design."""

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

from .candidate import CandidateConfig, normalize_candidate


CATALOG_ID = "timeloop_codesign_scalar_grid"
CATALOG_VERSION = 1
CATALOG_SCHEMA_ID = "timeloop_codesign_scalar_grid_v1"

FIELD_GRIDS: tuple[tuple[str, tuple[object, ...], str], ...] = (
    (
        "global_buffer_depth",
        (256, 512, 1024, 2048),
        "memory_capacity",
    ),
    (
        "global_buffer_width",
        (64, 128, 256),
        "memory_bandwidth",
    ),
    ("pe_mesh_x", (4, 8, 16, 32), "compute_parallelism"),
    ("datawidth_bits", (8, 16), "numeric_precision"),
    ("register_enabled", (False, True), "local_storage"),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


_DEFINITION = {
    "schema_id": CATALOG_SCHEMA_ID,
    "catalog_id": CATALOG_ID,
    "catalog_version": CATALOG_VERSION,
    "candidate_schema": CandidateConfig.model_json_schema(by_alias=False),
    "field_grids": [
        {"field": field, "values": list(values), "family": family}
        for field, values, family in FIELD_GRIDS
    ],
    "materialization": "exact_one_field_replacement_no_clipping_or_repair",
}
CATALOG_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:timeloop-codesign-finite-catalog:v1\x00"
    + _canonical_bytes(_DEFINITION)
).hexdigest()


def _prompt_value(value: object) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    raise TypeError("Timeloop catalog values must be exact integers or booleans")


class TimeloopFiniteVariationCatalog:
    """Enumerate every exact one-field alternative around a frozen parent."""

    catalog_id = CATALOG_ID
    catalog_version = CATALOG_VERSION
    definition_sha256 = CATALOG_DEFINITION_SHA256
    option_families = tuple(sorted({family for _, _, family in FIELD_GRIDS}))

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("Timeloop finite catalog requires a FrozenJsonObject")
        parent = normalize_candidate(thaw_json(parent_configuration))
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent_payload = parent.model_dump(mode="python")
        options: list[FiniteVariationOption] = []

        for field_ordinal, (field, values, family) in enumerate(FIELD_GRIDS):
            current = parent_payload[field]
            for value_ordinal, value in enumerate(values):
                if type(value) is type(current) and value == current:
                    continue
                child_payload = dict(parent_payload)
                child_payload[field] = value
                child = normalize_candidate(child_payload)
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise RuntimeError("Timeloop candidate did not freeze as an object")
                options.append(
                    FiniteVariationOption(
                        option_id=(
                            f"timeloop.f{field_ordinal:02d}.v{value_ordinal:02d}"
                        ),
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=frozen_child,
                        family=family,
                        description=(
                            f"Set {field} from {_prompt_value(current)} "
                            f"to {_prompt_value(value)}."
                        ),
                        metadata=(
                            ("catalog_definition_sha256", CATALOG_DEFINITION_SHA256),
                            ("field", field),
                            ("target_value", _prompt_value(value)),
                        ),
                    )
                )
        return tuple(options)


__all__ = [
    "CATALOG_DEFINITION_SHA256",
    "CATALOG_ID",
    "CATALOG_SCHEMA_ID",
    "CATALOG_VERSION",
    "FIELD_GRIDS",
    "TimeloopFiniteVariationCatalog",
]

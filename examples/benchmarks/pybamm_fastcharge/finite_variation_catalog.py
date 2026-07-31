"""Parent-relative atomic protocol perturbations for pybamm_fastcharge.

One option replaces exactly one protocol locus with one alternative grid
value; children that would violate threshold monotonicity are filtered by
re-validating through the exact candidate schema, so every published option
materializes to a valid child by construction.
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
from pydantic import ValidationError

from .problem_def import ChargingProtocolCandidate, normalize_candidate


CATALOG_ID = "pybamm_fastcharge_protocol_grid"
CATALOG_VERSION = 1
_C_RATE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

LOCUS_GRIDS: tuple[tuple[str, tuple[float, ...], str], ...] = (
    ("stage1_c_rate", _C_RATE_GRID, "stage_currents"),
    ("stage2_c_rate", _C_RATE_GRID, "stage_currents"),
    ("stage3_c_rate", _C_RATE_GRID, "stage_currents"),
    ("switch_v1", (3.85, 3.9, 3.95, 4.0, 4.05, 4.1), "switch_voltages"),
    ("switch_v2", (3.95, 4.0, 4.05, 4.1, 4.15, 4.18), "switch_voltages"),
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


_DEFINITION = {
    "schema_id": f"{CATALOG_ID}_v{CATALOG_VERSION}",
    "catalog_id": CATALOG_ID,
    "catalog_version": CATALOG_VERSION,
    "candidate_schema": ChargingProtocolCandidate.model_json_schema(
        by_alias=False
    ),
    "locus_grids": [
        {"locus": locus, "values": list(values), "family": family}
        for locus, values, family in LOCUS_GRIDS
    ],
    "materialization": (
        "exact_one_locus_replacement_schema_revalidated_monotonicity_filtered"
    ),
}
CATALOG_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pybamm-fastcharge-catalog:v1\x00"
    + _canonical_bytes(_DEFINITION)
).hexdigest()


class PybammFastChargeFiniteVariationCatalog:
    """Enumerate exact one-locus children of a charging protocol."""

    catalog_id = CATALOG_ID
    catalog_version = CATALOG_VERSION
    definition_sha256 = CATALOG_DEFINITION_SHA256
    option_families = tuple(sorted({family for _, _, family in LOCUS_GRIDS}))

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError(
                "pybamm_fastcharge catalog requires an exact FrozenJsonObject"
            )
        parent = normalize_candidate(thaw_json(parent_configuration))
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent_payload = parent.model_dump(mode="python")
        options: list[FiniteVariationOption] = []
        for locus_ordinal, (locus, grid, family) in enumerate(LOCUS_GRIDS):
            current = parent_payload[locus]
            emitted = 0
            for grid_ordinal, value in enumerate(grid):
                if value == current:
                    continue
                child_payload = dict(parent_payload)
                child_payload[locus] = value
                try:
                    child = normalize_candidate(child_payload)
                except ValidationError:
                    continue  # monotonicity-filtered relative to this parent
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise RuntimeError(
                        "protocol candidate did not freeze as an object"
                    )
                options.append(
                    FiniteVariationOption(
                        option_id=(
                            f"pybamm.l{locus_ordinal:02d}.v{grid_ordinal:02d}"
                        ),
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=frozen_child,
                        family=family,
                        description=(
                            f"Set {locus} from {float(current):.17g} "
                            f"to {value:.17g}."
                        ),
                        metadata=(
                            ("catalog_definition_sha256", CATALOG_DEFINITION_SHA256),
                            ("locus", locus),
                            ("target_value", f"{value:.17g}"),
                        ),
                    )
                )
                emitted += 1
            if family == "stage_currents" and emitted < len(grid) - 1:
                raise RuntimeError(
                    f"pybamm catalog emitted too few moves for locus {locus}"
                )
        return tuple(options)


__all__ = [
    "CATALOG_DEFINITION_SHA256",
    "CATALOG_ID",
    "CATALOG_VERSION",
    "LOCUS_GRIDS",
    "PybammFastChargeFiniteVariationCatalog",
]

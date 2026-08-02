"""Parent-relative atomic sizing moves for the analog_sizing workload.

One option replaces exactly one sizing locus with one alternative grid value.
Every grid value lies inside AnalogGym's own declared search box
(``RGNN_RL/ckt_graphs.py::GraphAMPNMCF``), and every emitted child re-validates
through the exact candidate schema before publication, so an out-of-box sizing
is unrepresentable rather than merely discouraged.

Widths and lengths are laid out geometrically because device behaviour is
governed by ratios, not by absolute microns; multipliers are laid out on a
coarse geometric ladder for the same reason.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_evolve.agentic import (
    FiniteVariationOption,
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)

from .problem_def import AnalogSizingCandidate, normalize_candidate


CATALOG_ID = "analog_sizing_nmcf_sizing_grid"
CATALOG_VERSION = 1

_W = (0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.0, 10.0)
_L = (0.5, 0.7, 1.0, 1.5, 2.2, 3.2, 4.5, 5.0)
_M = (1, 2, 4, 8, 16, 32, 50)
_M_OUT = (100, 150, 220, 320, 420, 500)
_C = (1.8228e-12, 3.6456e-12, 7.2912e-12, 1.45824e-11, 2.73420e-11, 5.4684e-11)
_IB = (1e-6, 2.5e-6, 5e-6, 1e-5, 2e-5, 3e-5)

_DEVICE_GROUPS = (
    ("bias_pmos", "bias"),
    ("input_pair", "input"),
    ("gm2_pmos", "gain"),
    ("bias_nmos", "bias"),
    ("load2_nmos", "gain"),
    ("gm3_nmos", "gain"),
)

LOCUS_GRIDS: tuple[tuple[str, tuple[Any, ...], str], ...] = tuple(
    [(f"{g}.{f}", grid, family)
     for g, family in _DEVICE_GROUPS
     for f, grid in (("w", _W), ("l", _L), ("m", _M))]
    + [("gmf2_output.w", _W, "output"),
       ("gmf2_output.l", _L, "output"),
       ("gmf2_output.m", _M_OUT, "output"),
       ("compensation.c0", _C, "compensation"),
       ("compensation.c1", _C, "compensation")]
) + (("bias_current", _IB, "bias"),)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


_DEFINITION = {
    "schema_id": f"{CATALOG_ID}_v{CATALOG_VERSION}",
    "catalog_id": CATALOG_ID,
    "catalog_version": CATALOG_VERSION,
    "candidate_schema": AnalogSizingCandidate.model_json_schema(by_alias=False),
    "locus_grids": [
        {"locus": locus, "values": list(values), "family": family}
        for locus, values, family in LOCUS_GRIDS
    ],
    "materialization": "exact_one_locus_replacement_schema_revalidated",
}
CATALOG_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:analog-sizing-catalog:v1\x00" + _canonical_bytes(_DEFINITION)
).hexdigest()


def _format_value(value: Any) -> str:
    if type(value) is float:
        return f"{value:.17g}"
    return json.dumps(value)


def _get(payload: dict, locus: str) -> Any:
    if "." in locus:
        group, field = locus.split(".")
        return payload[group][field]
    return payload[locus]


def _set(payload: dict, locus: str, value: Any) -> None:
    if "." in locus:
        group, field = locus.split(".")
        payload[group][field] = value
    else:
        payload[locus] = value


class AnalogSizingFiniteVariationCatalog:
    """Enumerate exact one-locus children of a typed amplifier sizing."""

    catalog_id = CATALOG_ID
    catalog_version = CATALOG_VERSION
    definition_sha256 = CATALOG_DEFINITION_SHA256
    option_families = tuple(sorted({family for _, _, family in LOCUS_GRIDS}))

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("analog_sizing catalog requires an exact FrozenJsonObject")
        parent = normalize_candidate(thaw_json(parent_configuration))
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent_payload = parent.model_dump(mode="python")
        options: list[FiniteVariationOption] = []
        for locus_ordinal, (locus, grid, family) in enumerate(LOCUS_GRIDS):
            current = _get(parent_payload, locus)
            for grid_ordinal, value in enumerate(grid):
                if type(value) is type(current) and value == current:
                    continue
                child_payload = json.loads(_canonical_bytes(parent_payload))
                _set(child_payload, locus, value)
                child = normalize_candidate(child_payload)
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise RuntimeError("analog candidate did not freeze as an object")
                options.append(
                    FiniteVariationOption(
                        option_id=f"analog.l{locus_ordinal:02d}.v{grid_ordinal:02d}",
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=frozen_child,
                        family=family,
                        description=(
                            f"Set {locus} from {_format_value(current)} "
                            f"to {_format_value(value)}."
                        ),
                        metadata=(
                            ("catalog_definition_sha256", CATALOG_DEFINITION_SHA256),
                            ("locus", locus),
                            ("target_value", _format_value(value)),
                        ),
                    )
                )
        return tuple(options)


__all__ = [
    "CATALOG_DEFINITION_SHA256",
    "CATALOG_ID",
    "CATALOG_VERSION",
    "LOCUS_GRIDS",
    "AnalogSizingFiniteVariationCatalog",
]

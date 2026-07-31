"""Parent-relative atomic parameter moves for the scip_miplib workload.

One option replaces exactly one curated locus (raw parameter or emphasis
meta-setting) with one alternative grid value.  Every grid value below was
dry-``setParam``-verified on the installed SCIP 10.0 build; every emitted
child re-validates through the exact candidate schema before publication.
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

from .problem_def import ScipConfigCandidate, normalize_candidate


CATALOG_ID = "scip_miplib_curated_param_grid"
CATALOG_VERSION = 1
_EMPHASIS = ("off", "default", "fast", "aggressive")

LOCUS_GRIDS: tuple[tuple[str, tuple[Any, ...], str], ...] = (
    ("separating.emphasis", _EMPHASIS, "separating"),
    ("separating.maxrounds", (-1, 0, 1, 3, 5, 10), "separating"),
    ("separating.maxroundsroot", (-1, 0, 5, 10, 20, 40), "separating"),
    ("separating.maxcuts", (0, 50, 100, 200, 500), "separating"),
    ("separating.maxcutsroot", (0, 500, 2000, 5000, 10000), "separating"),
    ("separating.gomory_freq", (-1, 0, 4, 10, 20), "separating"),
    ("separating.zerohalf_freq", (-1, 0, 10, 20), "separating"),
    ("separating.clique_freq", (-1, 0, 10, 20), "separating"),
    ("heuristics.emphasis", _EMPHASIS, "heuristics"),
    ("heuristics.rins_freq", (-1, 8, 15, 30, 60), "heuristics"),
    ("heuristics.rens_freq", (-1, 0, 10, 30), "heuristics"),
    ("heuristics.feaspump_freq", (-1, 10, 20, 40), "heuristics"),
    ("heuristics.shifting_freq", (-1, 3, 5, 10, 20), "heuristics"),
    ("heuristics.localbranching_freq", (-1, 0, 10, 30), "heuristics"),
    ("presolving.emphasis", _EMPHASIS, "presolving"),
    ("presolving.maxrounds", (-1, 0, 2, 5, 10), "presolving"),
    ("presolving.maxrestarts", (-1, 0, 1, 3), "presolving"),
    ("branching.scorefunc", ("s", "p", "q"), "branching"),
    ("branching.scorefac", (0.05, 0.167, 0.3, 0.5, 0.8), "branching"),
    ("branching.preferbinary", (False, True), "branching"),
    ("lp_node.pricing", ("l", "a", "f", "p", "s", "q", "d"), "lp_node"),
    ("lp_node.initalgorithm", ("s", "p", "d", "b", "c"), "lp_node"),
    ("lp_node.childsel", ("d", "u", "p", "i", "l", "r", "h"), "lp_node"),
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
    "candidate_schema": ScipConfigCandidate.model_json_schema(by_alias=False),
    "locus_grids": [
        {"locus": locus, "values": list(values), "family": family}
        for locus, values, family in LOCUS_GRIDS
    ],
    "materialization": "exact_one_locus_replacement_schema_revalidated",
}
CATALOG_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:scip-miplib-catalog:v1\x00" + _canonical_bytes(_DEFINITION)
).hexdigest()


def _format_value(value: Any) -> str:
    if type(value) is float:
        return f"{value:.17g}"
    return json.dumps(value)


class ScipMiplibFiniteVariationCatalog:
    """Enumerate exact one-locus children of a curated SCIP configuration."""

    catalog_id = CATALOG_ID
    catalog_version = CATALOG_VERSION
    definition_sha256 = CATALOG_DEFINITION_SHA256
    option_families = tuple(sorted({family for _, _, family in LOCUS_GRIDS}))

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("scip_miplib catalog requires an exact FrozenJsonObject")
        parent = normalize_candidate(thaw_json(parent_configuration))
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent_payload = parent.model_dump(mode="python")
        options: list[FiniteVariationOption] = []
        for locus_ordinal, (locus, grid, family) in enumerate(LOCUS_GRIDS):
            group, field = locus.split(".")
            current = parent_payload[group][field]
            emitted = 0
            for grid_ordinal, value in enumerate(grid):
                if type(value) is type(current) and value == current:
                    continue
                child_payload = json.loads(_canonical_bytes(parent_payload))
                child_payload[group][field] = value
                child = normalize_candidate(child_payload)
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise RuntimeError("scip candidate did not freeze as an object")
                options.append(
                    FiniteVariationOption(
                        option_id=f"scip.l{locus_ordinal:02d}.v{grid_ordinal:02d}",
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
                emitted += 1
            if emitted < len(grid) - 1:
                raise RuntimeError(
                    f"scip catalog emitted too few moves for locus {locus}"
                )
        return tuple(options)


__all__ = [
    "CATALOG_DEFINITION_SHA256",
    "CATALOG_ID",
    "CATALOG_VERSION",
    "LOCUS_GRIDS",
    "ScipMiplibFiniteVariationCatalog",
]

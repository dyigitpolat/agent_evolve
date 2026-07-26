"""Alias-checked parent-local finite variations for Timeloop v2."""

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

from .candidate import (
    ARCHITECTURE_FIELD_GRIDS,
    POLICY_BLOCK_FIELDS,
    POLICY_FIELD_GRIDS,
    CandidateConfig,
    normalize_candidate,
)
from .compiler import COMPILER_DEFINITION_SHA256, TimeloopV2Compiler
from .network_panel import NetworkLayerPanel, panel_sha256


CATALOG_ID = "timeloop_codesign_v2_scalar_grid"
CATALOG_VERSION = 1
CATALOG_SCHEMA_ID = "timeloop_codesign_v2_scalar_grid_v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _prompt_value(value: object) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int or type(value) is str:
        return str(value)
    raise TypeError("v2 catalog values must be exact integers, booleans, or strings")


def _read_locus(payload: dict[str, Any], locus: str) -> object:
    current: Any = payload
    for key in locus.split("."):
        current = current[key]
    return current


def _set_locus(payload: dict[str, Any], locus: str, value: object) -> None:
    path = locus.split(".")
    cursor: dict[str, Any] = payload
    for key in path[:-1]:
        nested = cursor[key]
        if type(nested) is not dict:
            raise RuntimeError(f"candidate locus is not an object: {locus}")
        cursor = nested
    cursor[path[-1]] = value


class TimeloopV2FiniteVariationCatalog:
    """Emit 61 exact one-field children and prove phenotype novelty locally."""

    catalog_id = CATALOG_ID
    catalog_version = CATALOG_VERSION

    def __init__(self, panel: NetworkLayerPanel) -> None:
        if type(panel) is not NetworkLayerPanel:
            raise TypeError("panel must be an exact NetworkLayerPanel")
        self._panel = panel
        definition = {
            "schema_id": CATALOG_SCHEMA_ID,
            "catalog_id": CATALOG_ID,
            "catalog_version": CATALOG_VERSION,
            "compiler_definition_sha256": COMPILER_DEFINITION_SHA256,
            "panel_sha256": panel_sha256(panel),
            "candidate_schema": CandidateConfig.model_json_schema(by_alias=False),
            "architecture_field_grids": [
                {"locus": field, "values": list(values), "family": family}
                for field, values, family in ARCHITECTURE_FIELD_GRIDS
            ],
            "policy_field_grids": [
                {"field": field, "values": list(values), "family": family}
                for field, values, family in POLICY_FIELD_GRIDS
            ],
            "policy_blocks": list(POLICY_BLOCK_FIELDS),
            "materialization": (
                "exact_one_field_replacement_no_clipping_no_repair_alias_checked"
            ),
        }
        self.definition_sha256 = hashlib.sha256(
            b"agent-evolve:timeloop-v2-finite-catalog:v1\x00"
            + _canonical_bytes(definition)
        ).hexdigest()
        self.option_families = tuple(
            sorted(
                {family for _, _, family in ARCHITECTURE_FIELD_GRIDS}
                | {family for _, _, family in POLICY_FIELD_GRIDS}
            )
        )

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        if type(parent_configuration) is not FrozenJsonObject:
            raise TypeError("v2 finite catalog requires an exact FrozenJsonObject")
        parent = normalize_candidate(thaw_json(parent_configuration))
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent_payload = parent.model_dump(mode="python")
        parent_plan_sha256 = TimeloopV2Compiler.compile(
            parent,
            self._panel,
        ).compiled_plan_sha256
        loci: list[tuple[str, tuple[object, ...], str]] = list(ARCHITECTURE_FIELD_GRIDS)
        for block in POLICY_BLOCK_FIELDS:
            loci.extend(
                (f"{block}.{field}", values, family)
                for field, values, family in POLICY_FIELD_GRIDS
            )

        options: list[FiniteVariationOption] = []
        compiled_children: set[str] = set()
        for locus_ordinal, (locus, values, family) in enumerate(loci):
            current = _read_locus(parent_payload, locus)
            for value_ordinal, value in enumerate(values):
                if type(value) is type(current) and value == current:
                    continue
                child_payload = json.loads(_canonical_bytes(parent_payload))
                _set_locus(child_payload, locus, value)
                child = normalize_candidate(child_payload)
                child_plan_sha256 = TimeloopV2Compiler.compile(
                    child,
                    self._panel,
                ).compiled_plan_sha256
                if child_plan_sha256 == parent_plan_sha256:
                    raise RuntimeError(
                        f"catalog locus compiled to parent alias: {locus}"
                    )
                if child_plan_sha256 in compiled_children:
                    raise RuntimeError(
                        f"two parent-local options compiled to one phenotype: {locus}"
                    )
                compiled_children.add(child_plan_sha256)
                frozen_child = freeze_json(child.model_dump(mode="python"))
                if type(frozen_child) is not FrozenJsonObject:  # pragma: no cover
                    raise RuntimeError("v2 candidate did not freeze as an object")
                options.append(
                    FiniteVariationOption(
                        option_id=f"timeloopv2.l{locus_ordinal:02d}.v{value_ordinal:02d}",
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=frozen_child,
                        family=family,
                        description=(
                            f"Set {locus} from {_prompt_value(current)} "
                            f"to {_prompt_value(value)}."
                        ),
                        metadata=(
                            ("catalog_definition_sha256", self.definition_sha256),
                            ("compiled_plan_sha256", child_plan_sha256),
                            ("locus", locus),
                            ("target_value", _prompt_value(value)),
                        ),
                    )
                )
        if len(options) != 61 or len(compiled_children) != 61:
            raise RuntimeError("v2 parent-local catalog must contain 61 phenotypes")
        return tuple(options)


__all__ = [
    "CATALOG_ID",
    "CATALOG_SCHEMA_ID",
    "CATALOG_VERSION",
    "TimeloopV2FiniteVariationCatalog",
]

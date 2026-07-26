"""Structural action semantics derived from any sealed finite contract.

Benchmarks may expose a richer, human-authored action glossary through the
public inverted API.  This module supplies the portable fallback: exact
parent-to-child patch paths and option-family membership are enough to state
what coordinates an action touches without inventing workload knowledge.
The option descriptions remain the authoritative prompt-visible semantics.
"""

from __future__ import annotations

import hashlib
import re

from agent_evolve.core.action_semantics import (
    ActionAxisSemantics,
    ActionSpaceSemantics,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.policies.variation.typed_patch import derive_patch


DERIVED_ACTION_SEMANTICS_ID = "derived_finite_contract_action_space"
DERIVED_ACTION_SEMANTICS_VERSION = 1
_SAFE_PATH_KEY = re.compile(r"^[^\.\[\]\s]+$")


def _render_path(path: JsonPath) -> str:
    path.__post_init__()
    if not path.segments:
        raise ValueError(
            "root-replacement finite actions require explicit benchmark action semantics"
        )
    rendered = "$"
    for segment in path.segments:
        if type(segment) is ObjectKey:
            if _SAFE_PATH_KEY.fullmatch(segment.value) is None:
                raise ValueError(
                    "finite-action path cannot use the canonical prompt path grammar; "
                    "provide explicit benchmark action semantics"
                )
            rendered += "." + segment.value
        elif type(segment) is ArrayIndex:
            rendered += f"[{segment.value}]"
        else:  # pragma: no cover - JsonPath validation closes the union.
            raise AssertionError("unsupported typed path segment")
    return rendered


def derive_action_space_semantics(
    contract: FiniteVariationContract,
) -> ActionSpaceSemantics:
    """Derive a hash-bound structural glossary for one parent-bound palette.

    The derivation reads sealed children only in trusted code.  Provider-visible
    output contains canonical coordinate paths and family membership, never a
    child configuration or an inferred causal interpretation.
    """

    validate_finite_variation_contract(contract)
    families_by_path: dict[str, set[str]] = {}
    base_id = CandidateId("candidate_derived_action_semantics_parent")
    child_id = CandidateId("candidate_derived_action_semantics_child")
    for option in contract.options:
        patch = derive_patch(
            contract.parent_configuration,
            option.child_configuration,
            base_candidate_id=base_id,
            target_candidate_id=child_id,
        )
        if not patch.operations:  # pragma: no cover - contract already forbids this.
            raise ValueError("finite action does not change its parent")
        for operation in patch.operations:
            families_by_path.setdefault(_render_path(operation.path), set()).add(
                option.family
            )
    axes = tuple(
        sorted(
            (
                ActionAxisSemantics(
                    axis_id=(
                        "coordinate_"
                        + hashlib.sha256(path.encode("utf-8")).hexdigest()[:20]
                    ),
                    configuration_paths=(path,),
                    option_families=tuple(sorted(families)),
                    definition=(
                        "A configuration coordinate changed by at least one sealed "
                        f"finite action: {path}. Exact option descriptions are "
                        "authoritative for its domain meaning."
                    ),
                    independence=(
                        "No coordinate independence is assumed; one sealed finite "
                        "action may jointly change multiple declared coordinates."
                    ),
                    excluded_interpretations=(
                        "Do not infer geometry, ordering, causality, or units from "
                        "the canonical path alone.",
                    ),
                )
                for path, families in families_by_path.items()
            ),
            key=lambda value: value.axis_id,
        )
    )
    return ActionSpaceSemantics(
        semantics_id=DERIVED_ACTION_SEMANTICS_ID,
        semantics_version=DERIVED_ACTION_SEMANTICS_VERSION,
        catalog_identities=(
            (
                contract.catalog_id,
                contract.catalog_version,
                contract.catalog_definition_sha256,
            ),
        ),
        axes=axes,
    )


__all__ = [
    "DERIVED_ACTION_SEMANTICS_ID",
    "DERIVED_ACTION_SEMANTICS_VERSION",
    "derive_action_space_semantics",
]

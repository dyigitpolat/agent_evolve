"""Outcome-blind K=8 local trim support for AgentEvolve Stage B."""

from __future__ import annotations

import hashlib
import json
from itertools import product

from agent_evolve.agentic import (
    FiniteActionSetCompilationRequest,
    FiniteActionSetDraft,
)


_COMPILER_DOMAIN = b"agent-evolve:airfoil-v7-stage-b-action-set:v1\x00"
_PRESENTATION_DOMAIN = b"agent-evolve:airfoil-v7-stage-b-presentation:v1\x00"
_DEFINITION = {
    "catalog_id": "airfoil_v7_trim",
    "anchor": "exact_compiled_trim_action",
    "support": "same_sign_pattern_all_025_050_magnitude_combinations",
    "cardinality": 8,
    "order": "source_contract_presentation_order",
    "current_outcome_access": False,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256 = hashlib.sha256(
    _COMPILER_DOMAIN + _canonical(_DEFINITION)
).hexdigest()
AIRFOIL_STAGE_B_PROMPT_SHAPE_SHA256 = hashlib.sha256(
    _PRESENTATION_DOMAIN
    + _canonical(
        {
            "card_record_count": 1,
            "option_record_count": 8,
            "option_fields": ["description", "family", "metadata", "option_id"],
            "selected_field": "option_id",
        }
    )
).hexdigest()


class AirfoilTrimLocalSupportCompiler:
    """Select all magnitudes within the anchor's three-coordinate sign pattern."""

    policy_id = "airfoil_v7_trim_local_support"
    policy_version = 1
    definition_sha256 = AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256

    def compile(
        self,
        request: FiniteActionSetCompilationRequest,
    ) -> FiniteActionSetDraft:
        if type(request) is not FiniteActionSetCompilationRequest:
            raise TypeError("request must be an exact FiniteActionSetCompilationRequest")
        FiniteActionSetCompilationRequest.__post_init__(request)
        if request.current_outcome_access:
            raise ValueError("Airfoil support compiler cannot access outcomes")
        if request.finite_contract.catalog_id != "airfoil_v7_trim":
            raise ValueError("Airfoil local support requires the trim catalog")
        if request.required_cardinality != 8:
            raise ValueError("Airfoil sign-pattern support has exactly eight options")
        parts = request.anchor_option_id.split(".")
        if len(parts) != 4 or parts[0] != "trim":
            raise ValueError("Airfoil anchor is not a three-coordinate trim option")
        signs: list[str] = []
        for token in parts[1:]:
            if len(token) != 4 or token[0] not in {"n", "p"} or token[1:] not in {
                "025",
                "050",
            }:
                raise ValueError("Airfoil trim anchor uses an unsupported delta token")
            signs.append(token[0])
        support_ids = {
            f"trim.{signs[0]}{first}.{signs[1]}{second}.{signs[2]}{third}"
            for first, second, third in product(("025", "050"), repeat=3)
        }
        ordered = tuple(
            option.option_id
            for option in request.finite_contract.options
            if option.option_id in support_ids
        )
        if len(ordered) != 8 or request.anchor_option_id not in set(ordered):
            raise ValueError("Airfoil source contract lacks the complete local support")
        return FiniteActionSetDraft(
            request_sha256=request.request_sha256,
            ordered_option_ids=ordered,
            anchor_option_id=request.anchor_option_id,
            presentation_policy_id="airfoil_v7_stage_b_local_presentation",
            presentation_policy_version=1,
            presentation_definition_sha256=self.definition_sha256,
            prompt_shape_sha256=AIRFOIL_STAGE_B_PROMPT_SHAPE_SHA256,
        )


__all__ = [
    "AIRFOIL_STAGE_B_ACTION_SET_DEFINITION_SHA256",
    "AIRFOIL_STAGE_B_PROMPT_SHAPE_SHA256",
    "AirfoilTrimLocalSupportCompiler",
]

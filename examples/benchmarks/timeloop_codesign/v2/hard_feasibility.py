"""Exact static mapspace-feasibility adapter for the generic screening port."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityDecision,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
)

from .compiler import COMPILER_DEFINITION_SHA256
from .evaluator import (
    analyze_static_mapspace_feasibility,
    build_evaluation_bundle,
)
from .network_panel import NetworkLayerPanel, panel_sha256


POLICY_ID = "timeloop_v2_exact_static_mapspace"
POLICY_VERSION = 1
_DEFINITION_DOMAIN = b"agent-evolve:timeloop-v2-hard-feasibility:def:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class TimeloopV2HardFeasibility:
    panel: NetworkLayerPanel
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.panel) is not NetworkLayerPanel:
            raise TypeError("panel must be an exact NetworkLayerPanel")
        definition = {
            "schema_version": 1,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "panel_sha256": panel_sha256(self.panel),
            "compiler_definition_sha256": COMPILER_DEFINITION_SHA256,
            "proof": "empty_primary_axis_integer_factor_range",
            "rejection_authority": "exact_infeasibility_only",
        }
        computed = hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(definition)
        ).hexdigest()
        if self.definition_sha256 not in ("", computed):
            raise ValueError("definition_sha256 does not authenticate the adapter")
        object.__setattr__(self, "definition_sha256", computed)

    def assess(
        self,
        request: HardFeasibilityRequest,
    ) -> HardFeasibilityDecision:
        self.__post_init__()
        if type(request) is not HardFeasibilityRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        bundle = build_evaluation_bundle(
            thaw_json(request.configuration),
            self.panel,
        )
        infeasible = analyze_static_mapspace_feasibility(bundle)
        if infeasible is None:
            verdict = HardFeasibilityVerdict.UNKNOWN
            proof_record = {
                "proof_kind": "no_empty_primary_axis_factor_range_found",
                "candidate_sha256": bundle.candidate_sha256,
                "compiled_plan_sha256": bundle.compiled_plan_sha256,
                "panel_sha256": bundle.panel_sha256,
                "simulator_feasibility_not_claimed": True,
            }
        else:
            verdict = HardFeasibilityVerdict.INFEASIBLE
            proof_record = {
                "proof_kind": "empty_primary_axis_integer_factor_range",
                "candidate_sha256": infeasible.candidate_sha256,
                "compiled_plan_sha256": infeasible.compiled_plan_sha256,
                "panel_sha256": infeasible.panel_sha256,
                "evaluation_bundle_sha256": infeasible.evaluation_bundle_sha256,
                "witnesses": [
                    {
                        "medoid_ordinal": value.medoid_ordinal,
                        "primary_axis": value.primary_axis,
                        "axis_extent": value.axis_extent,
                        "minimum_parallelism": value.minimum_parallelism,
                        "maximum_parallelism": value.maximum_parallelism,
                        "admissible_spatial_factors": list(
                            value.admissible_spatial_factors
                        ),
                    }
                    for value in infeasible.witnesses
                ],
            }
        proof = freeze_json(proof_record)
        if type(proof) is not FrozenJsonObject:  # pragma: no cover - closed root.
            raise AssertionError("feasibility proof did not freeze to an object")
        return HardFeasibilityDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            verdict=verdict,
            proof=proof,
        )


__all__ = [
    "POLICY_ID",
    "POLICY_VERSION",
    "TimeloopV2HardFeasibility",
]

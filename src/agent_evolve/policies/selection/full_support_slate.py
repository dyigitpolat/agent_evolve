"""Quality-first allocation that evaluates an entire calibrated slate.

The model-facing proposal contract and the evaluator-facing portfolio width
are deliberately separate concerns.  This policy closes that gap by selecting
every authenticated proposal in model order.  It is useful while a campaign is
still collecting enough outcome evidence to justify a narrower learned
allocator, and it avoids silently converting proposal quality into an
allocation failure.

The policy is workload neutral: it sees only :class:`SlateAllocationRequest`
and never inspects option names, workload identifiers, configurations, or
outcomes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agent_evolve.policies.selection.calibrated_slate import (
    SlateAllocationDecision,
    SlateAllocationMode,
    SlateAllocationRequest,
    TraceCalibratedSlatePolicy,
)


POLICY_ID = "full_support_calibrated_slate"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:full-support-calibrated-slate:v1;"
    b"select-entire-authenticated-slate=true;model-order-preserved=true;"
    b"outcome-blind=true;workload-identifiers-unused=true;"
    b"quality-first-evidence-acquisition=true"
).hexdigest()
CONFIGURATION_SHA256 = hashlib.sha256(
    b"agent-evolve:full-support-calibrated-slate-configuration:v1;"
    b"selection-scope=entire-authenticated-slate;"
    b"outcome-access=false;workload-identifier-access=false"
).hexdigest()


@dataclass(frozen=True, slots=True)
class FullSupportSlatePolicy:
    """Select every member of a finite calibrated slate in model order."""

    policy_id = POLICY_ID
    policy_version = POLICY_VERSION
    definition_sha256 = POLICY_DEFINITION_SHA256
    configuration_sha256 = CONFIGURATION_SHA256

    def __post_init__(self) -> None:
        if type(self) is not FullSupportSlatePolicy:
            raise TypeError("full-support policy must be exact")

    def revalidate(self) -> None:
        self.__post_init__()

    def select(self, request: SlateAllocationRequest) -> SlateAllocationDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        if request.portfolio_size != len(request.slate.members):
            raise ValueError(
                "full-support allocation requires portfolio_size equal to slate size"
            )
        return TraceCalibratedSlatePolicy(
            SlateAllocationMode.DIRECT_MODEL_TOP_K
        ).select(request)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "configuration_sha256": self.configuration_sha256,
            "selection_scope": "entire_authenticated_slate",
            "outcome_access": False,
            "workload_identifier_access": False,
        }


__all__ = [
    "CONFIGURATION_SHA256",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "FullSupportSlatePolicy",
]

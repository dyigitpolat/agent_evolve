"""Exact-K binary-composition capacity projection.

Cross-member recourse can make an otherwise feasible proposal composition
impossible after mandatory evaluation, memory, or structural obligations have
been bound.  This policy projects a preferred composite count onto the exact
interval realizable by the *current* mandatory and selectable sets.  It is
deliberately blind to workload identifiers, model/provider identities, and
objective values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


POLICY_ID = "nearest_exact_k_binary_composition_capacity_projection"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:nearest-exact-k-binary-composition-capacity-projection:v1;"
    b"inputs=proposal-size,preferred-count,mandatory-counts,selectable-counts;"
    b"feasible-interval=mandatory-and-selectable-capacity-intersection;"
    b"projection=nearest-clamped-preference;"
    b"objective-values-consulted=false;workload-model-provider-identifiers=false"
).hexdigest()
_RECEIPT_DOMAIN = b"agent-evolve:exact-k-composition-capacity-receipt:v1\x00"


@dataclass(frozen=True, slots=True)
class ExactKCompositionCapacityProjection:
    """Authenticated nearest feasible binary-composition count."""

    proposal_size: int
    preferred_composite_count: int
    mandatory_atomic_count: int
    mandatory_composite_count: int
    selectable_atomic_count: int
    selectable_composite_count: int
    feasible_minimum_composite_count: int
    feasible_maximum_composite_count: int
    effective_composite_count: int

    def __post_init__(self) -> None:
        for name in (
            "proposal_size",
            "preferred_composite_count",
            "mandatory_atomic_count",
            "mandatory_composite_count",
            "selectable_atomic_count",
            "selectable_composite_count",
            "feasible_minimum_composite_count",
            "feasible_maximum_composite_count",
            "effective_composite_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.proposal_size < 2:
            raise ValueError("proposal_size must be at least two")
        if not 1 <= self.preferred_composite_count < self.proposal_size:
            raise ValueError("preferred composite count must be internal to K")
        if self.mandatory_atomic_count > self.selectable_atomic_count:
            raise ValueError("mandatory atomic count exceeds selectable capacity")
        if self.mandatory_composite_count > self.selectable_composite_count:
            raise ValueError("mandatory composite count exceeds selectable capacity")
        if (
            self.mandatory_atomic_count + self.mandatory_composite_count
            > self.proposal_size
        ):
            raise ValueError("mandatory set exceeds proposal size")
        if (
            self.selectable_atomic_count + self.selectable_composite_count
            < self.proposal_size
        ):
            raise ValueError("selectable set cannot fill the proposal")
        if not (
            1
            <= self.feasible_minimum_composite_count
            <= self.feasible_maximum_composite_count
            < self.proposal_size
        ):
            raise ValueError(
                "composition capacity interval is not internal and ordered"
            )
        if not (
            self.feasible_minimum_composite_count
            <= self.effective_composite_count
            <= self.feasible_maximum_composite_count
        ):
            raise ValueError("effective composite count escapes its feasible interval")
        expected = min(
            max(
                self.preferred_composite_count,
                self.feasible_minimum_composite_count,
            ),
            self.feasible_maximum_composite_count,
        )
        if self.effective_composite_count != expected:
            raise ValueError("effective count is not the nearest interval projection")

    @property
    def capacity_projected(self) -> bool:
        return self.effective_composite_count != self.preferred_composite_count

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "policy_definition_sha256": POLICY_DEFINITION_SHA256,
            "proposal_size": self.proposal_size,
            "preferred_composite_count": self.preferred_composite_count,
            "mandatory_atomic_count": self.mandatory_atomic_count,
            "mandatory_composite_count": self.mandatory_composite_count,
            "selectable_atomic_count": self.selectable_atomic_count,
            "selectable_composite_count": self.selectable_composite_count,
            "feasible_minimum_composite_count": (self.feasible_minimum_composite_count),
            "feasible_maximum_composite_count": (self.feasible_maximum_composite_count),
            "effective_composite_count": self.effective_composite_count,
            "capacity_projected": self.capacity_projected,
            "objective_values_consulted": False,
            "workload_model_provider_identifiers_consulted": False,
        }

    @property
    def receipt_sha256(self) -> str:
        payload = json.dumps(
            self._unsigned_record(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(_RECEIPT_DOMAIN + payload).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


def project_exact_k_binary_composition(
    *,
    proposal_size: int,
    preferred_composite_count: int,
    mandatory_atomic_count: int,
    mandatory_composite_count: int,
    selectable_atomic_count: int,
    selectable_composite_count: int,
) -> ExactKCompositionCapacityProjection:
    """Project a preferred composite count onto the realizable exact-K interval."""

    for name, value in (
        ("proposal_size", proposal_size),
        ("preferred_composite_count", preferred_composite_count),
        ("mandatory_atomic_count", mandatory_atomic_count),
        ("mandatory_composite_count", mandatory_composite_count),
        ("selectable_atomic_count", selectable_atomic_count),
        ("selectable_composite_count", selectable_composite_count),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative exact integer")
    if proposal_size < 2 or not 1 <= preferred_composite_count < proposal_size:
        raise ValueError("preferred composition must be internal to an exact K >= 2")
    if mandatory_atomic_count > selectable_atomic_count:
        raise ValueError("mandatory atomic count exceeds selectable capacity")
    if mandatory_composite_count > selectable_composite_count:
        raise ValueError("mandatory composite count exceeds selectable capacity")
    if mandatory_atomic_count + mandatory_composite_count > proposal_size:
        raise ValueError("mandatory set exceeds proposal size")
    if selectable_atomic_count + selectable_composite_count < proposal_size:
        raise ValueError("selectable set cannot fill the proposal")

    lower = max(
        1,
        mandatory_composite_count,
        proposal_size - selectable_atomic_count,
    )
    upper = min(
        proposal_size - 1,
        selectable_composite_count,
        proposal_size - mandatory_atomic_count,
    )
    if lower > upper:
        raise ValueError(
            "mandatory and selectable counts have no feasible exact-K binary "
            "composition interval: "
            f"K={proposal_size}, preferred={preferred_composite_count}, "
            f"mandatory_atomic={mandatory_atomic_count}, "
            f"mandatory_composite={mandatory_composite_count}, "
            f"selectable_atomic={selectable_atomic_count}, "
            f"selectable_composite={selectable_composite_count}, "
            f"lower={lower}, upper={upper}"
        )
    effective = min(max(preferred_composite_count, lower), upper)
    return ExactKCompositionCapacityProjection(
        proposal_size=proposal_size,
        preferred_composite_count=preferred_composite_count,
        mandatory_atomic_count=mandatory_atomic_count,
        mandatory_composite_count=mandatory_composite_count,
        selectable_atomic_count=selectable_atomic_count,
        selectable_composite_count=selectable_composite_count,
        feasible_minimum_composite_count=lower,
        feasible_maximum_composite_count=upper,
        effective_composite_count=effective,
    )


__all__ = [
    "ExactKCompositionCapacityProjection",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "project_exact_k_binary_composition",
]

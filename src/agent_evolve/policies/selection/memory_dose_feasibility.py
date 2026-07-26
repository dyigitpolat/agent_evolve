"""Workload-neutral exact feasibility for bounded memory-dose attribution.

The portfolio reconciler must know whether a structural evaluation subset can
also carry the prospectively declared card dose.  This module answers only
that finite constraint question.  It never sees objective values, workload
identifiers, model/provider metadata, or natural-language card content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from itertools import combinations

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseStage,
)


POLICY_ID = "exact_bounded_memory_dose_attribution_feasibility"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:exact-bounded-memory-dose-attribution-feasibility:v1;"
    b"input=finite-option-identities,bounded-card-support-contract,stage;"
    b"constraints=compatibility,member-bounds,unattributed-floor,max-cards,card-cover;"
    b"objective=first-canonical-minimum-dose-witness;"
    b"objective-values=false;workload-model-provider-identifiers=false"
).hexdigest()
_DOMAIN = b"agent-evolve:memory-dose-attribution-feasibility-witness:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class MemoryDoseAttributionFeasibilityWitness:
    """One exact card assignment for a fixed finite member subset."""

    contract_sha256: str
    stage: PortfolioMemoryDoseStage
    member_option_identities: tuple[tuple[str, str], ...]
    attributions: tuple[tuple[str, tuple[str, ...]], ...]
    witness_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "contract_sha256")
        if type(self.stage) is not PortfolioMemoryDoseStage:
            raise TypeError("stage must be exact PortfolioMemoryDoseStage")
        if type(self.member_option_identities) is not tuple or any(
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not str
            or not value[0]
            or type(value[1]) is not str
            for value in self.member_option_identities
        ):
            raise TypeError("member_option_identities must contain exact pairs")
        for _, identity_sha256 in self.member_option_identities:
            require_sha256(identity_sha256, "option_identity_sha256")
        if self.member_option_identities != tuple(
            sorted(set(self.member_option_identities))
        ):
            raise ValueError(
                "member_option_identities must be unique and canonical"
            )
        option_ids = tuple(value[0] for value in self.member_option_identities)
        identity_sha256s = tuple(
            value[1] for value in self.member_option_identities
        )
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("member option IDs must be unique")
        if len(set(identity_sha256s)) != len(identity_sha256s):
            raise ValueError("member option identities must be unique")
        if type(self.attributions) is not tuple or any(
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not str
            or not value[0]
            or type(value[1]) is not tuple
            or any(type(card) is not str or not card for card in value[1])
            or value[1] != tuple(sorted(set(value[1])))
            for value in self.attributions
        ):
            raise TypeError("attributions must contain canonical exact values")
        if tuple(value[0] for value in self.attributions) != tuple(
            value[0] for value in self.member_option_identities
        ):
            raise ValueError("attributions differ from the witnessed members")
        object.__setattr__(
            self,
            "witness_sha256",
            hashlib.sha256(_DOMAIN + _canonical_json(self._unsigned_record())).hexdigest(),
        )

    @property
    def supported_member_count(self) -> int:
        return sum(bool(cards) for _, cards in self.attributions)

    @property
    def unattributed_member_count(self) -> int:
        return len(self.attributions) - self.supported_member_count

    @property
    def covered_card_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted({card for _, cards in self.attributions for card in cards})
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": POLICY_ID,
                "policy_version": POLICY_VERSION,
                "definition_sha256": POLICY_DEFINITION_SHA256,
            },
            "contract_sha256": self.contract_sha256,
            "stage": self.stage.value,
            "members": [
                {
                    "option_id": option_id,
                    "option_identity_sha256": identity_sha256,
                    "supporting_card_keys": list(cards),
                }
                for (option_id, identity_sha256), (_, cards) in zip(
                    self.member_option_identities,
                    self.attributions,
                    strict=True,
                )
            ],
            "supported_member_count": self.supported_member_count,
            "unattributed_member_count": self.unattributed_member_count,
            "covered_card_keys": list(self.covered_card_keys),
            "objective_values_consulted": False,
            "workload_model_provider_identifiers_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "witness_sha256": self.witness_sha256}


def find_memory_dose_attribution_feasibility_witness(
    contract: BoundedPortfolioMemoryDoseContract,
    *,
    stage: PortfolioMemoryDoseStage,
    member_option_identities: tuple[tuple[str, str], ...],
) -> MemoryDoseAttributionFeasibilityWitness | None:
    """Return the first canonical exact attribution, or ``None`` if impossible."""

    if type(contract) is not BoundedPortfolioMemoryDoseContract:
        raise TypeError("contract must be exact BoundedPortfolioMemoryDoseContract")
    contract.__post_init__()
    if type(stage) is not PortfolioMemoryDoseStage:
        raise TypeError("stage must be exact PortfolioMemoryDoseStage")
    if type(member_option_identities) is not tuple or any(
        type(value) is not tuple or len(value) != 2 for value in member_option_identities
    ):
        raise TypeError("member_option_identities must contain exact pairs")
    members = tuple(sorted(member_option_identities))
    # Construction performs the complete boundary validation once.
    provisional = MemoryDoseAttributionFeasibilityWitness(
        contract_sha256=contract.contract_sha256,
        stage=stage,
        member_option_identities=members,
        attributions=tuple((option_id, ()) for option_id, _ in members),
    )
    del provisional

    lower, upper = contract.bounds_for(stage)
    maximum_supported = min(
        upper,
        len(members) - contract.minimum_unattributed_for(stage),
    )
    if lower > maximum_supported:
        return None
    assigned_cards = contract.assigned_card_keys
    support_by_card = {value.card_key: value for value in contract.card_supports}
    choices: list[tuple[tuple[str, ...], ...]] = []
    for option_id, identity_sha256 in members:
        compatible = tuple(
            card_key
            for card_key in assigned_cards
            if support_by_card[card_key].supports(option_id, identity_sha256)
        )
        option_choices: list[tuple[str, ...]] = [()]
        for count in range(
            1,
            min(len(compatible), contract.maximum_cards_per_member) + 1,
        ):
            option_choices.extend(combinations(compatible, count))
        choices.append(tuple(option_choices))

    selected: list[tuple[str, ...]] = []
    result: MemoryDoseAttributionFeasibilityWitness | None = None

    def search(index: int, supported_count: int, covered: frozenset[str]) -> bool:
        nonlocal result
        remaining = len(members) - index
        if supported_count > maximum_supported:
            return False
        if supported_count + remaining < lower:
            return False
        if contract.require_every_assigned_card:
            future_cover = set(covered)
            for future_choices in choices[index:]:
                for candidate in future_choices:
                    future_cover.update(candidate)
            if not set(assigned_cards).issubset(future_cover):
                return False
        if index == len(members):
            if not lower <= supported_count <= maximum_supported:
                return False
            if contract.require_every_assigned_card and set(covered) != set(
                assigned_cards
            ):
                return False
            result = MemoryDoseAttributionFeasibilityWitness(
                contract_sha256=contract.contract_sha256,
                stage=stage,
                member_option_identities=members,
                attributions=tuple(
                    (option_id, cards)
                    for (option_id, _), cards in zip(members, selected, strict=True)
                ),
            )
            return True
        for candidate in choices[index]:
            selected.append(candidate)
            if search(
                index + 1,
                supported_count + bool(candidate),
                covered.union(candidate),
            ):
                return True
            selected.pop()
        return False

    search(0, 0, frozenset())
    return result


__all__ = [
    "MemoryDoseAttributionFeasibilityWitness",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "find_memory_dose_attribution_feasibility_witness",
]

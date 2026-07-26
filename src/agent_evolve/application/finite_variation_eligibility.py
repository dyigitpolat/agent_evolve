"""Outcome-blind phenotype eligibility for finite variation portfolios.

The optimizer may know that a phenotype has already been evaluated without
using the phenotype's objective values.  This module turns that identity-only
ledger into a sealed finite-contract view.  It also keeps at most one option
per phenotype in a wave, preventing semantic aliases from consuming multiple
evaluation slots.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.patch import require_sha256


FINITE_VARIATION_ELIGIBILITY_POLICY_ID = "outcome_blind_phenotype_exclusion"
FINITE_VARIATION_ELIGIBILITY_POLICY_VERSION = 1
FINITE_VARIATION_ELIGIBILITY_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:outcome-blind-phenotype-exclusion:v1:"
    b"exclude-known-phenotypes-and-canonicalize-wave-aliases"
).hexdigest()

_RECEIPT_DOMAIN = b"agent-evolve:finite-variation-eligibility:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class OptionPhenotypeBinding:
    """Bind one sealed option to its benchmark-supplied phenotype identity."""

    option_id: str
    option_identity_sha256: str
    phenotype_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty exact string")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(self.phenotype_identity_sha256, "phenotype_identity_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class FiniteVariationEligibilityReceipt:
    """Replayable identity-only admission decision for one option catalog."""

    base_contract_identity_sha256: str
    eligible_contract_identity_sha256: str
    known_phenotype_sha256s: tuple[str, ...]
    option_phenotypes: tuple[OptionPhenotypeBinding, ...]
    eligible_option_ids: tuple[str, ...]
    known_excluded_option_ids: tuple[str, ...]
    alias_excluded_option_ids: tuple[str, ...]
    policy_id: str = FINITE_VARIATION_ELIGIBILITY_POLICY_ID
    policy_version: int = FINITE_VARIATION_ELIGIBILITY_POLICY_VERSION
    policy_definition_sha256: str = (
        FINITE_VARIATION_ELIGIBILITY_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        require_sha256(
            self.base_contract_identity_sha256,
            "base_contract_identity_sha256",
        )
        require_sha256(
            self.eligible_contract_identity_sha256,
            "eligible_contract_identity_sha256",
        )
        if self.policy_id != FINITE_VARIATION_ELIGIBILITY_POLICY_ID:
            raise ValueError("eligibility receipt policy_id drifted")
        if self.policy_version != FINITE_VARIATION_ELIGIBILITY_POLICY_VERSION:
            raise ValueError("eligibility receipt policy_version drifted")
        if (
            self.policy_definition_sha256
            != FINITE_VARIATION_ELIGIBILITY_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("eligibility receipt policy definition drifted")
        for value in self.known_phenotype_sha256s:
            require_sha256(value, "known_phenotype_sha256")
        if self.known_phenotype_sha256s != tuple(
            sorted(set(self.known_phenotype_sha256s))
        ):
            raise ValueError("known phenotype identities must be unique and sorted")
        if type(self.option_phenotypes) is not tuple or any(
            type(value) is not OptionPhenotypeBinding
            for value in self.option_phenotypes
        ):
            raise TypeError("option_phenotypes must contain exact bindings")
        for value in self.option_phenotypes:
            value.__post_init__()
        option_ids = tuple(value.option_id for value in self.option_phenotypes)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("option phenotype bindings repeat an option")
        for name in (
            "eligible_option_ids",
            "known_excluded_option_ids",
            "alias_excluded_option_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str for value in values
            ):
                raise TypeError(f"{name} must be an exact tuple of strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        partitions = (
            set(self.eligible_option_ids),
            set(self.known_excluded_option_ids),
            set(self.alias_excluded_option_ids),
        )
        if any(
            left & right
            for index, left in enumerate(partitions)
            for right in partitions[index + 1 :]
        ):
            raise ValueError("eligibility receipt option partitions overlap")
        if set().union(*partitions) != set(option_ids):
            raise ValueError("eligibility receipt does not partition every option")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "base_contract_identity_sha256": self.base_contract_identity_sha256,
            "eligible_contract_identity_sha256": (
                self.eligible_contract_identity_sha256
            ),
            "known_phenotype_sha256s": list(self.known_phenotype_sha256s),
            "option_phenotypes": [
                value.to_record() for value in self.option_phenotypes
            ],
            "eligible_option_ids": list(self.eligible_option_ids),
            "known_excluded_option_ids": list(self.known_excluded_option_ids),
            "alias_excluded_option_ids": list(self.alias_excluded_option_ids),
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            _RECEIPT_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {**record, "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class EligibleFiniteVariationView:
    contract: FiniteVariationContract
    receipt: FiniteVariationEligibilityReceipt

    def __post_init__(self) -> None:
        validate_finite_variation_contract(self.contract)
        if type(self.receipt) is not FiniteVariationEligibilityReceipt:
            raise TypeError("receipt must be exact")
        self.receipt.__post_init__()
        if (
            self.contract.identity_sha256
            != self.receipt.eligible_contract_identity_sha256
        ):
            raise ValueError("eligible contract differs from its receipt")
        if tuple(option.option_id for option in self.contract.options) != (
            self.receipt.eligible_option_ids
        ):
            raise ValueError("eligible contract option order differs from its receipt")


def exact_configuration_phenotype_bindings(
    contract: FiniteVariationContract,
) -> tuple[OptionPhenotypeBinding, ...]:
    """Use typed child configuration identity as the conservative phenotype."""

    validate_finite_variation_contract(contract)
    return tuple(
        OptionPhenotypeBinding(
            option_id=option.option_id,
            option_identity_sha256=option.identity_sha256,
            phenotype_identity_sha256=option.child_configuration_sha256,
        )
        for option in contract.options
    )


def eligible_finite_variation_view(
    *,
    contract: FiniteVariationContract,
    option_phenotypes: tuple[OptionPhenotypeBinding, ...],
    known_phenotype_sha256s: tuple[str, ...],
) -> EligibleFiniteVariationView:
    """Seal options whose phenotypes are unknown, retaining one alias each.

    Admission uses identities only.  Objective values, rewards, ranks, and
    outcome relations are deliberately absent from the API.
    """

    validate_finite_variation_contract(contract)
    if type(option_phenotypes) is not tuple or any(
        type(value) is not OptionPhenotypeBinding for value in option_phenotypes
    ):
        raise TypeError("option_phenotypes must contain exact bindings")
    for value in option_phenotypes:
        value.__post_init__()
    expected = tuple(
        (option.option_id, option.identity_sha256) for option in contract.options
    )
    supplied = tuple(
        (value.option_id, value.option_identity_sha256)
        for value in option_phenotypes
    )
    if supplied != expected:
        raise ValueError("option phenotype bindings must match exact contract order")
    if type(known_phenotype_sha256s) is not tuple:
        raise TypeError("known_phenotype_sha256s must be an exact tuple")
    for value in known_phenotype_sha256s:
        require_sha256(value, "known_phenotype_sha256")
    known = tuple(sorted(set(known_phenotype_sha256s)))
    if known != known_phenotype_sha256s:
        raise ValueError("known phenotype identities must be unique and sorted")

    known_set = set(known)
    seen_phenotypes: set[str] = set()
    eligible_options = []
    eligible_ids: list[str] = []
    known_excluded: list[str] = []
    alias_excluded: list[str] = []
    for option, binding in zip(contract.options, option_phenotypes, strict=True):
        phenotype = binding.phenotype_identity_sha256
        if phenotype in known_set:
            known_excluded.append(option.option_id)
        elif phenotype in seen_phenotypes:
            alias_excluded.append(option.option_id)
        else:
            seen_phenotypes.add(phenotype)
            eligible_options.append(option)
            eligible_ids.append(option.option_id)
    if not eligible_options:
        raise ValueError("outcome-blind eligibility removed every finite option")

    eligible_contract = FiniteVariationContract(
        catalog_id="eligible_finite_variation",
        catalog_version=FINITE_VARIATION_ELIGIBILITY_POLICY_VERSION,
        catalog_definition_sha256=(
            FINITE_VARIATION_ELIGIBILITY_POLICY_DEFINITION_SHA256
        ),
        parent_configuration=contract.parent_configuration,
        options=tuple(eligible_options),
    )
    receipt = FiniteVariationEligibilityReceipt(
        base_contract_identity_sha256=contract.identity_sha256,
        eligible_contract_identity_sha256=eligible_contract.identity_sha256,
        known_phenotype_sha256s=known,
        option_phenotypes=option_phenotypes,
        eligible_option_ids=tuple(eligible_ids),
        known_excluded_option_ids=tuple(known_excluded),
        alias_excluded_option_ids=tuple(alias_excluded),
    )
    return EligibleFiniteVariationView(eligible_contract, receipt)


__all__ = [
    "FINITE_VARIATION_ELIGIBILITY_POLICY_DEFINITION_SHA256",
    "FINITE_VARIATION_ELIGIBILITY_POLICY_ID",
    "FINITE_VARIATION_ELIGIBILITY_POLICY_VERSION",
    "EligibleFiniteVariationView",
    "FiniteVariationEligibilityReceipt",
    "OptionPhenotypeBinding",
    "eligible_finite_variation_view",
    "exact_configuration_phenotype_bindings",
]

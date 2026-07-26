"""Outcome-blind diagnostic sampling from a sealed finite-action contract.

The first generation of an action-grounded experiment needs broad evidence,
but benchmark adapters must not quietly choose actions after seeing outcomes.
This policy balances the contract's declared action families and orders both
families and options with domain-separated hashes.  It consumes no evaluator,
metric, archive, or benchmark-specific parameter semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.patch import require_sha256


POLICY_ID = "hash_stratified_finite_diagnostics"
POLICY_VERSION = 1
_POLICY_DEFINITION = (
    b"agent-evolve:hash-stratified-finite-diagnostics:v1:"
    b"round-robin-declared-families;domain-separated-seed-and-contract-hash-order;"
    b"outcome-metric-and-evaluator-blind"
)
POLICY_DEFINITION_SHA256 = hashlib.sha256(_POLICY_DEFINITION).hexdigest()
_ORDER_DOMAIN = b"agent-evolve:diagnostic-action-order:v1\x00"
_SAMPLE_DOMAIN = b"agent-evolve:diagnostic-action-sample:v1\x00"
_DESIGN_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_SEED = (1 << 64) - 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _order_digest(
    *,
    seed: int,
    design_key: str,
    contract_sha256: str,
    purpose: str,
    identity: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_ORDER_DOMAIN)
    digest.update(seed.to_bytes(8, "big", signed=False))
    for value in (design_key, contract_sha256, purpose, identity):
        encoded = value.encode("ascii", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DiagnosticActionMember:
    """One exact finite option in its precommitted sampling order."""

    rank: int
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    family: str

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        if type(self.family) is not str or not self.family:
            raise ValueError("family must be non-empty")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(
            self.child_configuration_sha256,
            "child_configuration_sha256",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "rank": self.rank,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "family": self.family,
        }


@dataclass(frozen=True, slots=True, eq=False)
class DiagnosticActionSample:
    """Replay receipt for one outcome-independent G1 diagnostic set."""

    finite_contract_identity_sha256: str
    requested_size: int
    seed: int
    design_key: str
    members: tuple[DiagnosticActionMember, ...]
    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    policy_definition_sha256: str = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        if type(self.requested_size) is not int or self.requested_size <= 0:
            raise ValueError("requested_size must be a positive exact integer")
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            raise ValueError("seed must be an exact uint64 integer")
        if (
            type(self.design_key) is not str
            or _DESIGN_KEY.fullmatch(self.design_key) is None
        ):
            raise ValueError("design_key must use the closed lowercase token grammar")
        if type(self.members) is not tuple or len(self.members) != self.requested_size:
            raise ValueError("members must exactly match requested_size")
        if any(type(value) is not DiagnosticActionMember for value in self.members):
            raise TypeError("members must contain exact DiagnosticActionMember values")
        for member in self.members:
            member.__post_init__()
        if tuple(value.rank for value in self.members) != tuple(
            range(1, self.requested_size + 1)
        ):
            raise ValueError("member ranks must be contiguous and ordered")
        if len({value.option_id for value in self.members}) != len(self.members):
            raise ValueError("diagnostic sample cannot repeat an option")
        if len({value.option_identity_sha256 for value in self.members}) != len(
            self.members
        ):
            raise ValueError("diagnostic sample cannot repeat an option identity")
        if self.policy_id != POLICY_ID or self.policy_version != POLICY_VERSION:
            raise ValueError("diagnostic sample names an unsupported policy")
        if self.policy_definition_sha256 != POLICY_DEFINITION_SHA256:
            raise ValueError("diagnostic policy definition identity changed")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "requested_size": self.requested_size,
            "seed": self.seed,
            "design_key": self.design_key,
            "members": [value.to_record() for value in self.members],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            _SAMPLE_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is DiagnosticActionSample
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class HashStratifiedDiagnosticSampler:
    """Select a family-balanced diagnostic set without outcome information."""

    seed: int
    design_key: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            raise ValueError("seed must be an exact uint64 integer")
        if (
            type(self.design_key) is not str
            or _DESIGN_KEY.fullmatch(self.design_key) is None
        ):
            raise ValueError("design_key must use the closed lowercase token grammar")

    def sample(
        self,
        contract: FiniteVariationContract,
        *,
        sample_size: int,
    ) -> DiagnosticActionSample:
        """Return a deterministic round-robin sample over hashed family queues."""

        self.__post_init__()
        validate_finite_variation_contract(contract)
        if type(sample_size) is not int or sample_size <= 0:
            raise ValueError("sample_size must be a positive exact integer")
        if sample_size > len(contract.options):
            raise ValueError("sample_size exceeds the finite option count")

        contract_sha256 = contract.identity_sha256
        by_family: dict[str, list] = {}
        for option in contract.options:
            by_family.setdefault(option.family, []).append(option)
        family_order = sorted(
            by_family,
            key=lambda family: (
                _order_digest(
                    seed=self.seed,
                    design_key=self.design_key,
                    contract_sha256=contract_sha256,
                    purpose="family",
                    identity=family,
                ),
                family,
            ),
        )
        queues = {
            family: sorted(
                values,
                key=lambda option: (
                    _order_digest(
                        seed=self.seed,
                        design_key=self.design_key,
                        contract_sha256=contract_sha256,
                        purpose="option",
                        identity=option.identity_sha256,
                    ),
                    option.identity_sha256,
                    option.option_id,
                ),
            )
            for family, values in by_family.items()
        }

        selected = []
        offsets = {family: 0 for family in family_order}
        while len(selected) < sample_size:
            progressed = False
            for family in family_order:
                offset = offsets[family]
                if offset >= len(queues[family]):
                    continue
                selected.append(queues[family][offset])
                offsets[family] += 1
                progressed = True
                if len(selected) == sample_size:
                    break
            if not progressed:  # pragma: no cover - contract/sample bounds imply progress.
                raise AssertionError("finite diagnostic queues exhausted early")

        return DiagnosticActionSample(
            finite_contract_identity_sha256=contract_sha256,
            requested_size=sample_size,
            seed=self.seed,
            design_key=self.design_key,
            members=tuple(
                DiagnosticActionMember(
                    rank=rank,
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    child_configuration_sha256=(
                        option.child_configuration_sha256
                    ),
                    family=option.family,
                )
                for rank, option in enumerate(selected, start=1)
            ),
        )


def validate_diagnostic_action_sample(
    contract: FiniteVariationContract,
    sample: DiagnosticActionSample,
) -> None:
    """Rebind every receipt member to its exact sealed contract option."""

    validate_finite_variation_contract(contract)
    if type(sample) is not DiagnosticActionSample:
        raise TypeError("sample must be an exact DiagnosticActionSample")
    sample.__post_init__()
    if sample.finite_contract_identity_sha256 != contract.identity_sha256:
        raise ValueError("diagnostic sample is bound to a different contract")
    for member in sample.members:
        option = contract.resolve(member.option_id)
        if (
            member.option_identity_sha256 != option.identity_sha256
            or member.child_configuration_sha256
            != option.child_configuration_sha256
            or member.family != option.family
        ):
            raise ValueError("diagnostic member differs from its sealed option")


__all__ = [
    "DiagnosticActionMember",
    "DiagnosticActionSample",
    "HashStratifiedDiagnosticSampler",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "validate_diagnostic_action_sample",
]

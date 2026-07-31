"""Outcome-blind workload boundary for a finite acquisition reservoir.

The workload owns legality, materialization, and a normalized feature codec.
The search controller supplies only cutoff identity, exclusions, pool size,
and a deterministic seed.  Objective values never cross this boundary; they
are joined later by the generic acquisition-envelope application service.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_REQUEST_DOMAIN = b"agent-evolve:finite-acquisition-space-request:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionSpaceRequest:
    campaign_scope_sha256: str
    cutoff_index: int
    pool_size: int
    seed: int
    observed_configurations: tuple[FrozenJsonObject, ...]
    excluded_configuration_sha256s: tuple[str, ...]
    request_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.cutoff_index) is not int or self.cutoff_index < 1:
            raise ValueError("cutoff_index must be positive")
        if type(self.pool_size) is not int or self.pool_size < 1:
            raise ValueError("pool_size must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative exact integer")
        if type(self.observed_configurations) is not tuple or not (
            self.observed_configurations
        ):
            raise ValueError("observed_configurations must be a non-empty exact tuple")
        observed_identities: list[str] = []
        for value in self.observed_configurations:
            if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
                raise TypeError(
                    "observed configurations must be frozen typed-JSON objects"
                )
            observed_identities.append(typed_json_sha256(value))
        if tuple(observed_identities) != tuple(sorted(set(observed_identities))):
            raise ValueError("observed configurations must be unique and hash-sorted")
        if type(self.excluded_configuration_sha256s) is not tuple:
            raise TypeError("excluded configuration identities must be an exact tuple")
        for value in self.excluded_configuration_sha256s:
            require_sha256(value, "excluded_configuration_sha256")
        if self.excluded_configuration_sha256s != tuple(
            sorted(set(self.excluded_configuration_sha256s))
        ):
            raise ValueError("excluded configuration identities must be canonical")
        if not set(observed_identities) <= set(
            self.excluded_configuration_sha256s
        ):
            raise ValueError("observed configurations must be excluded from the pool")
        record = self._unsigned_record()
        computed = hashlib.sha256(
            _REQUEST_DOMAIN + _canonical_json(record)
        ).hexdigest()
        if self.request_sha256 not in ("", computed):
            raise ValueError("request_sha256 does not authenticate the request")
        object.__setattr__(self, "request_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "cutoff_index": self.cutoff_index,
            "pool_size": self.pool_size,
            "seed": self.seed,
            "observed_configuration_sha256s": [
                typed_json_sha256(value) for value in self.observed_configurations
            ],
            "excluded_configuration_sha256s": list(
                self.excluded_configuration_sha256s
            ),
            "outcome_access": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}


@runtime_checkable
class FiniteAcquisitionSpace(Protocol):
    """Enumerate legal configurations and encode any legal configuration."""

    space_id: str
    space_version: int
    definition_sha256: str

    def candidates(
        self,
        request: FiniteAcquisitionSpaceRequest,
    ) -> tuple[FrozenJsonObject, ...]: ...

    def features(self, configuration: FrozenJsonObject) -> tuple[float, ...]: ...


def validate_finite_acquisition_space_identity(
    space: FiniteAcquisitionSpace,
) -> tuple[str, int, str]:
    if not isinstance(space, FiniteAcquisitionSpace):
        raise TypeError("space must implement FiniteAcquisitionSpace")
    if type(space.space_id) is not str or _TOKEN.fullmatch(space.space_id) is None:
        raise ValueError("finite acquisition space_id has invalid syntax")
    if type(space.space_version) is not int or space.space_version <= 0:
        raise ValueError("finite acquisition space_version must be positive")
    require_sha256(space.definition_sha256, "finite acquisition space definition")
    return space.space_id, space.space_version, space.definition_sha256


def validate_finite_acquisition_space_candidates(
    *,
    request: FiniteAcquisitionSpaceRequest,
    candidates: tuple[FrozenJsonObject, ...],
) -> None:
    if type(request) is not FiniteAcquisitionSpaceRequest:
        raise TypeError("request must be exact")
    FiniteAcquisitionSpaceRequest.__post_init__(request)
    if type(candidates) is not tuple or len(candidates) != request.pool_size:
        raise ValueError("acquisition space must return the exact requested pool")
    excluded = set(request.excluded_configuration_sha256s)
    identities: list[str] = []
    for value in candidates:
        if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
            raise TypeError("acquisition candidates must be frozen typed-JSON objects")
        identity = typed_json_sha256(value)
        if identity in excluded:
            raise ValueError("acquisition space returned an excluded configuration")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("acquisition space returned duplicate configurations")


__all__ = [
    "FiniteAcquisitionSpace",
    "FiniteAcquisitionSpaceRequest",
    "validate_finite_acquisition_space_candidates",
    "validate_finite_acquisition_space_identity",
]

"""Benchmark-neutral exclusive-resource lease contracts.

Evaluators sometimes depend on process-external resources that an in-process
semaphore cannot protect: fixed Docker container names, licensed simulator
tokens, devices, or scratch workspaces.  This port lets a composition root
acquire one such resource before any costly work while keeping the optimizer
and benchmark semantics independent of the locking mechanism.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed resource-token grammar")


@dataclass(frozen=True, slots=True)
class ResourceConflictObservation:
    """One identity-bound, JSON-safe external-conflict observation."""

    probe_id: str
    probe_version: int
    conflict: bool
    facts: FrozenJsonObject

    def __post_init__(self) -> None:
        _token(self.probe_id, name="probe_id")
        if type(self.probe_version) is not int or self.probe_version <= 0:
            raise ValueError("probe_version must be a positive exact integer")
        if type(self.conflict) is not bool:
            raise TypeError("conflict must be an exact bool")
        if type(self.facts) is not FrozenJsonObject:
            raise TypeError("facts must be an exact FrozenJsonObject")
        if freeze_json(self.facts) is not self.facts:
            raise TypeError("facts must already be frozen typed JSON")

    def to_record(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "probe_version": self.probe_version,
            "conflict": self.conflict,
            "facts": thaw_json(self.facts),
        }


@dataclass(frozen=True, slots=True)
class ResourceLeaseReceipt:
    """Durable acquisition evidence returned before protected work starts."""

    resource_key: str
    owner_id: str
    acquisition_sha256: str
    lease_path: str
    process_id: int
    hostname: str
    acquired_at_utc: str
    owner_metadata: FrozenJsonObject
    conflict_observation: ResourceConflictObservation
    file_device: int
    file_inode: int

    def __post_init__(self) -> None:
        _token(self.resource_key, name="resource_key")
        if (
            type(self.owner_id) is not str
            or not self.owner_id
            or self.owner_id != self.owner_id.strip()
            or len(self.owner_id.encode("utf-8", errors="strict")) > 256
        ):
            raise ValueError("owner_id must be compact canonical text")
        if (
            type(self.acquisition_sha256) is not str
            or _SHA256.fullmatch(self.acquisition_sha256) is None
        ):
            raise ValueError("acquisition_sha256 must be a lowercase SHA-256")
        for name in ("lease_path", "hostname", "acquired_at_utc"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if type(self.process_id) is not int or self.process_id <= 0:
            raise ValueError("process_id must be a positive exact integer")
        if type(self.file_device) is not int or self.file_device < 0:
            raise ValueError("file_device must be a non-negative exact integer")
        if type(self.file_inode) is not int or self.file_inode <= 0:
            raise ValueError("file_inode must be a positive exact integer")
        if type(self.owner_metadata) is not FrozenJsonObject:
            raise TypeError("owner_metadata must be an exact FrozenJsonObject")
        if freeze_json(self.owner_metadata) is not self.owner_metadata:
            raise TypeError("owner_metadata must already be frozen typed JSON")
        if type(self.conflict_observation) is not ResourceConflictObservation:
            raise TypeError(
                "conflict_observation must be a ResourceConflictObservation"
            )
        ResourceConflictObservation.__post_init__(self.conflict_observation)
        if self.conflict_observation.conflict:
            raise ValueError("an acquired lease cannot carry a conflict")

    def to_record(self) -> dict[str, object]:
        return {
            "resource_key": self.resource_key,
            "owner_id": self.owner_id,
            "acquisition_sha256": self.acquisition_sha256,
            "lease_path": self.lease_path,
            "process_id": self.process_id,
            "hostname": self.hostname,
            "acquired_at_utc": self.acquired_at_utc,
            "owner_metadata": thaw_json(self.owner_metadata),
            "conflict_observation": self.conflict_observation.to_record(),
            "file_device": self.file_device,
            "file_inode": self.file_inode,
        }


@runtime_checkable
class ResourceConflictProbe(Protocol):
    """Observe conflicts from actors that may not honor the lease file."""

    def __call__(self) -> ResourceConflictObservation: ...


@runtime_checkable
class ExclusiveResourceLease(Protocol):
    """Nonblocking process-external lease acquired before costly work."""

    @property
    def active(self) -> bool: ...

    def acquire(self) -> ResourceLeaseReceipt: ...

    def release(
        self,
        *,
        outcome: str = "completed",
        failure_type: str | None = None,
    ) -> dict[str, object]: ...

    def __enter__(self) -> ResourceLeaseReceipt: ...

    def __exit__(self, exc_type, exc, traceback) -> bool: ...


__all__ = [
    "ExclusiveResourceLease",
    "ResourceConflictObservation",
    "ResourceConflictProbe",
    "ResourceLeaseReceipt",
]

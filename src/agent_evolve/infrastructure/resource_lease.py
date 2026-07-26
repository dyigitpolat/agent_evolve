"""Durable nonblocking ``flock`` implementation of the resource-lease port."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.ports.resource_lease import (
    ResourceConflictObservation,
    ResourceConflictProbe,
    ResourceLeaseReceipt,
)


_ACQUISITION_DOMAIN = b"agent-evolve:exclusive-resource-lease:v1\x00"
_MAX_LOCK_RECORD_BYTES = 65_536


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ResourceLeaseUnavailable(RuntimeError):
    """Another cooperating process already owns the resource lease."""

    def __init__(
        self,
        *,
        resource_key: str,
        lease_path: Path,
        holder_record_sha256: str,
    ) -> None:
        self.resource_key = resource_key
        self.lease_path = lease_path
        self.holder_record_sha256 = holder_record_sha256
        super().__init__(
            f"exclusive resource {resource_key!r} is already leased; "
            f"holder_record_sha256={holder_record_sha256}"
        )


class ResourceConflictDetected(RuntimeError):
    """A non-cooperating external actor conflicts with a newly locked resource."""

    def __init__(
        self,
        *,
        resource_key: str,
        observation: ResourceConflictObservation,
    ) -> None:
        self.resource_key = resource_key
        self.observation = observation
        super().__init__(
            f"external conflict probe {observation.probe_id!r} rejected "
            f"resource {resource_key!r}"
        )


class FileExclusiveResourceLease:
    """One-shot nonblocking filesystem lease with fsynced owner evidence.

    The lock descriptor is close-on-exec, so evaluator subprocesses cannot
    accidentally inherit ownership.  A crash releases the kernel lock; the
    next acquirer may overwrite the stale on-disk owner record only after it
    holds the lock and its external conflict probe passes.
    """

    def __init__(
        self,
        *,
        resource_key: str,
        owner_id: str,
        lease_path: Path,
        owner_metadata: Mapping[str, object],
        conflict_probe: ResourceConflictProbe | None = None,
    ) -> None:
        if type(resource_key) is not str:
            raise TypeError("resource_key must be an exact string")
        if (
            type(owner_id) is not str
            or not owner_id
            or owner_id != owner_id.strip()
            or len(owner_id.encode("utf-8", errors="strict")) > 256
        ):
            raise ValueError("owner_id must be compact canonical text")
        resolved = lease_path.expanduser().resolve()
        if resolved.name in {"", ".", ".."}:
            raise ValueError("lease_path must name one file")
        frozen_metadata = freeze_json(dict(owner_metadata))
        if type(frozen_metadata) is not FrozenJsonObject:
            raise TypeError("owner_metadata must freeze to a JSON object")
        if conflict_probe is not None and not isinstance(
            conflict_probe,
            ResourceConflictProbe,
        ):
            raise TypeError("conflict_probe must implement ResourceConflictProbe")

        # Reuse the receipt's closed validation law for resource tokens.
        ResourceConflictObservation(
            probe_id=resource_key,
            probe_version=1,
            conflict=False,
            facts=freeze_json({}),
        )
        self.resource_key = resource_key
        self.owner_id = owner_id
        self.lease_path = resolved
        self.owner_metadata = frozen_metadata
        self.conflict_probe = conflict_probe
        self._descriptor: int | None = None
        self._receipt: ResourceLeaseReceipt | None = None
        self._last_release_record: dict[str, object] | None = None
        self._state_lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._state_lock:
            return self._descriptor is not None

    @property
    def receipt(self) -> ResourceLeaseReceipt | None:
        with self._state_lock:
            return self._receipt

    @property
    def last_release_record(self) -> dict[str, object] | None:
        with self._state_lock:
            return (
                None
                if self._last_release_record is None
                else dict(self._last_release_record)
            )

    @staticmethod
    def _write_locked(descriptor: int, record: Mapping[str, object]) -> None:
        payload = _canonical_bytes(dict(record)) + b"\n"
        if len(payload) > _MAX_LOCK_RECORD_BYTES:
            raise ValueError("resource lease record exceeds its byte limit")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS boundary.
                raise OSError("resource lease record write made no progress")
            view = view[written:]
        os.fsync(descriptor)

    @staticmethod
    def _holder_sha256(descriptor: int) -> str:
        os.lseek(descriptor, 0, os.SEEK_SET)
        content = os.read(descriptor, _MAX_LOCK_RECORD_BYTES + 1)
        if len(content) > _MAX_LOCK_RECORD_BYTES:
            content = content[:_MAX_LOCK_RECORD_BYTES]
        return hashlib.sha256(content).hexdigest()

    def _close_unacquired(self, descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def acquire(self) -> ResourceLeaseReceipt:
        with self._state_lock:
            if self._descriptor is not None or self._receipt is not None:
                raise RuntimeError("resource lease objects are one-shot")
            self.lease_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.lease_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                holder_sha256 = self._holder_sha256(descriptor)
                os.close(descriptor)
                raise ResourceLeaseUnavailable(
                    resource_key=self.resource_key,
                    lease_path=self.lease_path,
                    holder_record_sha256=holder_sha256,
                ) from exc

            try:
                observation = (
                    ResourceConflictObservation(
                        probe_id="no_external_conflict_probe",
                        probe_version=1,
                        conflict=False,
                        facts=freeze_json({"probe_configured": False}),
                    )
                    if self.conflict_probe is None
                    else self.conflict_probe()
                )
                if type(observation) is not ResourceConflictObservation:
                    raise TypeError(
                        "conflict probe must return ResourceConflictObservation"
                    )
                ResourceConflictObservation.__post_init__(observation)
                if observation.conflict:
                    rejected = {
                        "schema_version": 1,
                        "status": "external_conflict_rejected",
                        "resource_key": self.resource_key,
                        "owner_id": self.owner_id,
                        "observed_at_utc": _utc_now(),
                        "conflict_observation": observation.to_record(),
                    }
                    self._write_locked(descriptor, rejected)
                    raise ResourceConflictDetected(
                        resource_key=self.resource_key,
                        observation=observation,
                    )

                stat = os.fstat(descriptor)
                acquired_at = _utc_now()
                unsigned = {
                    "schema_version": 1,
                    "status": "acquired",
                    "resource_key": self.resource_key,
                    "owner_id": self.owner_id,
                    "lease_path": str(self.lease_path),
                    "process_id": os.getpid(),
                    "hostname": socket.gethostname(),
                    "acquired_at_utc": acquired_at,
                    "owner_metadata": thaw_json(self.owner_metadata),
                    "conflict_observation": observation.to_record(),
                    "file_device": stat.st_dev,
                    "file_inode": stat.st_ino,
                }
                acquisition_sha256 = hashlib.sha256(
                    _ACQUISITION_DOMAIN + _canonical_bytes(unsigned)
                ).hexdigest()
                record = {**unsigned, "acquisition_sha256": acquisition_sha256}
                self._write_locked(descriptor, record)
                _directory_fsync(self.lease_path.parent)
                receipt = ResourceLeaseReceipt(
                    resource_key=self.resource_key,
                    owner_id=self.owner_id,
                    acquisition_sha256=acquisition_sha256,
                    lease_path=str(self.lease_path),
                    process_id=os.getpid(),
                    hostname=socket.gethostname(),
                    acquired_at_utc=acquired_at,
                    owner_metadata=self.owner_metadata,
                    conflict_observation=observation,
                    file_device=stat.st_dev,
                    file_inode=stat.st_ino,
                )
                self._descriptor = descriptor
                self._receipt = receipt
                return receipt
            except BaseException:
                self._close_unacquired(descriptor)
                raise

    def release(
        self,
        *,
        outcome: str = "completed",
        failure_type: str | None = None,
    ) -> dict[str, object]:
        if (
            type(outcome) is not str
            or not outcome
            or outcome != outcome.strip()
            or len(outcome.encode("utf-8", errors="strict")) > 128
        ):
            raise ValueError("outcome must be compact canonical text")
        if failure_type is not None and (
            type(failure_type) is not str
            or not failure_type
            or failure_type != failure_type.strip()
            or len(failure_type.encode("utf-8", errors="strict")) > 256
        ):
            raise ValueError("failure_type must be compact canonical text or None")
        with self._state_lock:
            descriptor = self._descriptor
            receipt = self._receipt
            if descriptor is None or receipt is None:
                raise RuntimeError("resource lease is not active")
            record = {
                "schema_version": 1,
                "status": "released",
                "resource_key": receipt.resource_key,
                "owner_id": receipt.owner_id,
                "acquisition_sha256": receipt.acquisition_sha256,
                "released_at_utc": _utc_now(),
                "outcome": outcome,
                "failure_type": failure_type,
                "process_id": os.getpid(),
                "hostname": socket.gethostname(),
            }
            pending: BaseException | None = None
            try:
                self._write_locked(descriptor, record)
            except BaseException as exc:
                pending = exc
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except BaseException as exc:
                if pending is None:
                    pending = exc
                else:
                    pending.add_note(
                        f"resource unlock also failed: {type(exc).__name__}"
                    )
            try:
                os.close(descriptor)
            except BaseException as exc:
                if pending is None:
                    pending = exc
                else:
                    pending.add_note(
                        f"resource descriptor close also failed: {type(exc).__name__}"
                    )
            self._descriptor = None
            self._last_release_record = record
            if pending is not None:
                raise pending
            return dict(record)

    def __enter__(self) -> ResourceLeaseReceipt:
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del traceback
        self.release(
            outcome="completed" if exc_type is None else "failed",
            failure_type=None if exc_type is None else exc_type.__name__,
        )
        return False


__all__ = [
    "FileExclusiveResourceLease",
    "ResourceConflictDetected",
    "ResourceLeaseUnavailable",
]

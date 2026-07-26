"""Atomic, durable filesystem implementation of the artifact-store port."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import agent_evolve.domain.artifact as artifact_domain
from agent_evolve.domain.artifact import ArtifactRef
from agent_evolve.domain.ids import ArtifactId
from agent_evolve.infrastructure.artifacts._verification import (
    require_artifact_id,
    require_content_and_media_type,
    verify_content,
    verify_expected_media_type,
)
from agent_evolve.ports.artifact_store import (
    ArtifactCollisionError,
    ArtifactNotFoundError,
    ArtifactSerializationError,
    ArtifactTypeError,
    CorruptArtifactError,
    canonical_json_bytes,
    decode_json_bytes,
)

try:  # pragma: no cover - exercised on Unix CI.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

_ARTIFACT_SCHEMA_VERSION = 2
_MAGIC_LINE = b"AGENT_EVOLVE_ARTIFACT_V2"
_FILE_SUFFIX = ".artifact"
_METADATA_KEYS = {
    "artifact_id",
    "media_type",
    "schema_version",
    "sha256_hex",
    "size_bytes",
}
_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}


def _process_lock_for_root(root: Path) -> threading.RLock:
    """Return one process-global lock for a canonical artifact-store root."""

    lock_key = os.path.normcase(os.fspath(root))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[lock_key] = lock
        return lock


class FileSystemArtifactStore:
    """Store one self-verifying ``<artifact_id>.artifact`` file per payload."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        requested_root = Path(root).absolute()
        try:
            canonical_parent = requested_root.parent.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Artifact-store parent directory must already exist: "
                f"{requested_root.parent}"
            ) from exc
        if not canonical_parent.is_dir():
            raise NotADirectoryError(
                f"Artifact-store parent is not a directory: {canonical_parent}"
            )
        canonical_root = canonical_parent / requested_root.name
        self._process_lock = _process_lock_for_root(canonical_root)
        with self._process_lock:
            if canonical_root.is_symlink():
                raise ValueError(
                    f"Artifact-store root must not be a symlink: {canonical_root}"
                )
            try:
                canonical_root.mkdir(mode=0o700)
            except FileExistsError:
                if canonical_root.is_symlink():
                    raise ValueError(
                        f"Artifact-store root must not be a symlink: {canonical_root}"
                    )
                if not canonical_root.is_dir():
                    raise NotADirectoryError(
                        f"Artifact-store root is not a directory: {canonical_root}"
                    )
            # This is deliberately unconditional: if another process won a
            # concurrent mkdir race, this constructor still establishes that the
            # directory entry is durable before returning.
            self._fsync_directory_path(canonical_parent)

        resolved_root = canonical_root.resolve(strict=True)
        if resolved_root != canonical_root:
            raise ValueError(
                f"Artifact-store root must resolve to itself: {canonical_root}"
            )
        self.root = resolved_root
        self._lock_path = self.root / ".artifact-store.lock"
        with self._locked():
            self._verify_all_unlocked()

    def path_for_artifact(self, artifact_id: ArtifactId) -> Path:
        require_artifact_id(artifact_id)
        return self.root / f"{artifact_id.value}{_FILE_SUFFIX}"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._process_lock:
            if fcntl is None:
                # This remains safe between store instances and threads in this
                # process. Cross-process use requires an available ``flock``.
                yield
                return
            lock_fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            locked = False
            try:
                # Failure is intentionally fail-closed: silently degrading here
                # would make a cross-process deployment appear durable when the
                # underlying filesystem does not support advisory locking.
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                locked = True
                yield
            finally:
                try:
                    if locked:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    @staticmethod
    def _metadata_record(ref: ArtifactRef) -> dict[str, Any]:
        return {
            "artifact_id": ref.artifact_id.value,
            "media_type": ref.media_type,
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "sha256_hex": ref.sha256_hex,
            "size_bytes": ref.size_bytes,
        }

    @classmethod
    def _encode_container(cls, ref: ArtifactRef, content: bytes) -> bytes:
        metadata = canonical_json_bytes(cls._metadata_record(ref))
        return _MAGIC_LINE + b"\n" + metadata + b"\n" + content

    @classmethod
    def _decode_container(
        cls,
        encoded: bytes,
        *,
        path: Path,
        expected_artifact_id: ArtifactId,
    ) -> tuple[ArtifactRef, bytes]:
        parts = encoded.split(b"\n", 2)
        if len(parts) != 3 or parts[0] != _MAGIC_LINE:
            raise CorruptArtifactError(f"Artifact file has an invalid header: {path}")
        raw_metadata, content = parts[1], parts[2]
        try:
            record = decode_json_bytes(raw_metadata)
        except (ArtifactSerializationError, ArtifactTypeError) as exc:
            raise CorruptArtifactError(
                f"Artifact file has malformed metadata: {path}"
            ) from exc
        if not isinstance(record, dict) or set(record) != _METADATA_KEYS:
            raise CorruptArtifactError(
                f"Artifact file has unsupported metadata fields: {path}"
            )
        try:
            canonical_metadata = canonical_json_bytes(record)
        except (ArtifactSerializationError, ArtifactTypeError) as exc:
            raise CorruptArtifactError(
                f"Artifact file metadata cannot be encoded canonically: {path}"
            ) from exc
        if canonical_metadata != raw_metadata:
            raise CorruptArtifactError(
                f"Artifact file metadata is not canonical JSON: {path}"
            )
        schema_version = record["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _ARTIFACT_SCHEMA_VERSION
        ):
            raise CorruptArtifactError(
                f"Artifact file has an unsupported schema version: {path}"
            )
        try:
            ref = ArtifactRef(
                artifact_id=ArtifactId(record["artifact_id"]),
                sha256_hex=record["sha256_hex"],
                size_bytes=record["size_bytes"],
                media_type=record["media_type"],
            )
        except (TypeError, ValueError) as exc:
            raise CorruptArtifactError(
                f"Artifact file metadata violates the schema: {path}"
            ) from exc
        if ref.artifact_id != expected_artifact_id:
            raise CorruptArtifactError(
                f"Artifact file name does not match its metadata: {path}"
            )
        verify_content(ref, content)
        return ref, content

    def _read_path_unlocked(
        self,
        path: Path,
        *,
        expected_artifact_id: ArtifactId,
    ) -> tuple[ArtifactRef, bytes]:
        if path.is_symlink() or not path.is_file():
            raise CorruptArtifactError(f"Artifact path is not a regular file: {path}")
        try:
            encoded = path.read_bytes()
        except OSError as exc:
            raise CorruptArtifactError(f"Artifact file cannot be read: {path}") from exc
        return self._decode_container(
            encoded,
            path=path,
            expected_artifact_id=expected_artifact_id,
        )

    def _verified_unlocked(self, artifact_id: ArtifactId) -> tuple[ArtifactRef, bytes]:
        require_artifact_id(artifact_id)
        path = self.path_for_artifact(artifact_id)
        if not path.exists() and not path.is_symlink():
            raise ArtifactNotFoundError(f"Artifact {artifact_id} was not found")
        return self._read_path_unlocked(path, expected_artifact_id=artifact_id)

    def _verify_all_unlocked(self) -> None:
        for path in sorted(self.root.glob(f"*{_FILE_SUFFIX}")):
            serialized_id = path.name[: -len(_FILE_SUFFIX)]
            try:
                artifact_id = ArtifactId(serialized_id)
            except (TypeError, ValueError) as exc:
                raise CorruptArtifactError(
                    f"Artifact file has an invalid content-addressed name: {path}"
                ) from exc
            self._read_path_unlocked(path, expected_artifact_id=artifact_id)

    @staticmethod
    def _write_all(fd: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:  # pragma: no cover - defensive OS failure guard.
                raise OSError("zero-byte write while persisting artifact")
            view = view[written:]

    @staticmethod
    def _fsync_directory_path(path: Path) -> None:
        if os.name != "posix":  # pragma: no cover - directory fsync is POSIX-only.
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _fsync_root_directory(self) -> None:
        self._fsync_directory_path(self.root)

    def _atomic_write_unlocked(self, path: Path, encoded: bytes) -> None:
        fd, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=".artifact-write-",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            self._write_all(fd, encoded)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary_path, path)
            self._fsync_root_directory()
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def put_bytes(self, content: bytes, *, media_type: str) -> ArtifactRef:
        require_content_and_media_type(content, media_type)
        ref = artifact_domain.artifact_ref_for_bytes(content, media_type=media_type)
        path = self.path_for_artifact(ref.artifact_id)
        with self._locked():
            if path.exists() or path.is_symlink():
                existing_ref, existing_content = self._read_path_unlocked(
                    path,
                    expected_artifact_id=ref.artifact_id,
                )
                if existing_content != content:
                    raise ArtifactCollisionError(
                        f"Different payload bytes resolved to {ref.artifact_id}"
                    )
                if existing_ref.media_type != media_type:
                    raise ArtifactCollisionError(
                        f"Different media types resolved to {ref.artifact_id}"
                    )
                # A previous attempt may have completed its atomic rename but
                # reported failure while syncing the directory. Re-syncing here
                # makes an idempotent retry a durability-recovery operation.
                self._fsync_root_directory()
                return existing_ref

            self._atomic_write_unlocked(path, self._encode_container(ref, content))
            persisted_ref, persisted_content = self._read_path_unlocked(
                path,
                expected_artifact_id=ref.artifact_id,
            )
            if persisted_ref != ref or persisted_content != content:
                raise CorruptArtifactError(
                    f"Artifact {ref.artifact_id} did not verify after persistence"
                )
            return persisted_ref

    def stat(self, artifact_id: ArtifactId) -> ArtifactRef:
        with self._locked():
            ref, _ = self._verified_unlocked(artifact_id)
            return ref

    def read_bytes(
        self,
        artifact_id: ArtifactId,
        *,
        expected_media_type: str | None = None,
    ) -> bytes:
        with self._locked():
            ref, content = self._verified_unlocked(artifact_id)
            verify_expected_media_type(ref, expected_media_type)
            return content

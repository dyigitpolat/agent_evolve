"""Small reusable durable-artifact helpers for development run harnesses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

from examples.development.corpus_paths import resolve_corpus_path
import threading

from agent_evolve.ports.artifact_store import canonical_json_bytes, decode_json_bytes


FINALIZATION_FRAMING = b"agent-evolve:development-run-finalization:v1\x00"


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Atomically publish exact bytes and fsync the file and directory entry."""

    if type(payload) is not bytes:
        raise TypeError("payload must be exact bytes")
    target = path.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        _directory_fsync(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically publish canonical JSON and fsync its directory entry."""

    write_bytes_atomic(path, canonical_json_bytes(value) + b"\n")


class DurableJsonlJournal:
    """Thread-safe canonical JSONL with fsync before ``append`` returns."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("xb")
        self._lock = threading.Lock()
        self._closed = False
        _directory_fsync(self.path.parent)

    def append(self, value: Mapping[str, object]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("journal value must be a mapping")
        payload = canonical_json_bytes(dict(value)) + b"\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("journal is closed")
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "DurableJsonlJournal":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class BatchedDurableJsonlJournal:
    """Bounded count-batched JSONL with an explicit durability barrier.

    Token-scale stream progress should not issue one ``fsync`` per event.  This
    writer syncs after at most ``max_unfsynced_rows`` appends and exposes
    :meth:`flush` so a required terminal-outcome sink can order progress
    durability before publishing the outcome.
    """

    def __init__(self, path: Path, *, max_unfsynced_rows: int) -> None:
        if type(max_unfsynced_rows) is not int or max_unfsynced_rows < 1:
            raise ValueError("max_unfsynced_rows must be a positive integer")
        self.path = path.expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("xb")
        self._max_unfsynced_rows = max_unfsynced_rows
        self._pending_rows = 0
        self._lock = threading.Lock()
        self._closed = False
        _directory_fsync(self.path.parent)

    def _flush_locked(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._pending_rows = 0

    def append(self, value: Mapping[str, object]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("journal value must be a mapping")
        payload = canonical_json_bytes(dict(value)) + b"\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("journal is closed")
            self._stream.write(payload)
            self._pending_rows += 1
            if self._pending_rows >= self._max_unfsynced_rows:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("journal is closed")
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._flush_locked()
            finally:
                self._stream.close()
                self._closed = True

    def __enter__(self) -> "BatchedDurableJsonlJournal":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = resolve_corpus_path(path).expanduser().resolve(strict=True)
    content = resolved.read_bytes()
    label = (
        resolved.name
        if relative_to is None
        else resolved.relative_to(relative_to.expanduser().resolve(strict=True)).as_posix()
    )
    return {
        "path": label,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def source_identity(
    paths: Sequence[Path],
    *,
    relative_to: Path,
) -> dict[str, object]:
    """Authenticate an ordered closed source set with length framing."""

    if not paths:
        raise ValueError("source identity requires at least one path")
    root = relative_to.expanduser().resolve(strict=True)
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256(b"agent-evolve:source-set:v1\x00")
    labels: set[str] = set()
    for path in paths:
        record = file_identity(path, relative_to=root)
        label = str(record["path"])
        if label in labels:
            raise ValueError("source identity paths must be unique")
        labels.add(label)
        content = path.expanduser().resolve(strict=True).read_bytes()
        label_bytes = label.encode("utf-8", errors="strict")
        aggregate.update(len(label_bytes).to_bytes(8, "big"))
        aggregate.update(label_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
        records.append(record)
    return {
        "schema_version": 1,
        "file_count": len(records),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }


def read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    content = path.expanduser().resolve(strict=True).read_bytes()
    if content and not content.endswith(b"\n"):
        raise RuntimeError("JSONL journal has a truncated final line")
    for raw_line in content.splitlines():
        value = decode_json_bytes(raw_line)
        if type(value) is not dict:
            raise RuntimeError("JSONL journal row is not an object")
        rows.append(value)
    return tuple(rows)


def finalize_run_directory(run_dir: Path, *, status: str) -> dict[str, object]:
    """Seal every published file below ``run_dir`` into one recursive identity."""

    if type(status) is not str or not status.strip():
        raise ValueError("status must be non-empty")
    root = run_dir.expanduser().resolve(strict=True)
    final_path = root / "finalized.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not path.name.endswith(".tmp")
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(FINALIZATION_FRAMING)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        files[relative] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            **(
                {"jsonl_rows": len(content.splitlines())}
                if path.suffix == ".jsonl"
                else {}
            ),
        }
        relative_bytes = relative.encode("utf-8", errors="strict")
        aggregate.update(len(relative_bytes).to_bytes(8, "big"))
        aggregate.update(relative_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    record: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "recursive_file_count": len(files),
        "recursive_content_sha256": aggregate.hexdigest(),
        "files": files,
    }
    record["finalization_sha256"] = hashlib.sha256(
        FINALIZATION_FRAMING + canonical_json_bytes(record)
    ).hexdigest()
    write_json_atomic(final_path, record)
    return record


def verify_finalized_run_directory(run_dir: Path) -> dict[str, object]:
    """Fail closed unless ``finalized.json`` authenticates the exact directory."""

    root = run_dir.expanduser().resolve(strict=True)
    final_path = root / "finalized.json"
    value = decode_json_bytes(final_path.read_bytes())
    if type(value) is not dict:
        raise RuntimeError("finalization record is not an exact object")
    record = dict(value)
    commitment = record.pop("finalization_sha256", None)
    if (
        type(commitment) is not str
        or len(commitment) != 64
        or any(character not in "0123456789abcdef" for character in commitment)
        or commitment
        != hashlib.sha256(
            FINALIZATION_FRAMING + canonical_json_bytes(record)
        ).hexdigest()
    ):
        raise RuntimeError("finalization commitment is invalid")
    files = value.get("files")
    if (
        value.get("schema_version") != 1
        or type(value.get("status")) is not str
        or type(value.get("finalized_at_utc")) is not str
        or type(files) is not dict
        or value.get("recursive_file_count") != len(files)
    ):
        raise RuntimeError("finalization record shape is invalid")
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path != final_path
            and not path.name.endswith(".tmp")
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if [path.relative_to(root).as_posix() for path in paths] != sorted(files):
        raise RuntimeError("finalized directory membership changed")
    aggregate = hashlib.sha256(FINALIZATION_FRAMING)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        expected = files.get(relative)
        content = path.read_bytes()
        if type(expected) is not dict:
            raise RuntimeError("finalized file identity is invalid")
        observed: dict[str, object] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            **(
                {"jsonl_rows": len(content.splitlines())}
                if path.suffix == ".jsonl"
                else {}
            ),
        }
        if expected != observed:
            raise RuntimeError("finalized file content changed")
        relative_bytes = relative.encode("utf-8", errors="strict")
        aggregate.update(len(relative_bytes).to_bytes(8, "big"))
        aggregate.update(relative_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    if value.get("recursive_content_sha256") != aggregate.hexdigest():
        raise RuntimeError("recursive finalization identity is invalid")
    return value


__all__ = [
    "BatchedDurableJsonlJournal",
    "DurableJsonlJournal",
    "file_identity",
    "finalize_run_directory",
    "read_jsonl",
    "source_identity",
    "verify_finalized_run_directory",
    "write_bytes_atomic",
    "write_json_atomic",
]

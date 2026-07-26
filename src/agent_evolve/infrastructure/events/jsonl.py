"""Durable append-only JSONL EventStore.

Correctness is favored over indexing in this first milestone: streams are
revalidated before reads and appends. Event payloads remain small because large
blobs will live in the artifact store added by the second half of M1.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from agent_evolve.domain.event import (
    EventCodecError,
    EventEnvelope,
    event_from_json,
    event_to_json,
)
from agent_evolve.domain.ids import EventId, RunId
from agent_evolve.infrastructure.events._validation import (
    prepare_event_for_append,
    validate_append,
    validate_loaded_stream,
)
from agent_evolve.ports.event_store import (
    CorruptEventLogError,
    DuplicateEventIdError,
    EventRunMismatchError,
    EventSequenceError,
)

try:  # pragma: no cover - exercised on Unix CI; fallback is thread-only.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class JsonlEventStore:
    """One canonical ``<run_id>.jsonl`` stream per run under *root*."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._lock_path = self.root / ".event-store.lock"
        # Fail early on malformed/tampered existing streams.
        with self._locked():
            self._load_all_unlocked()

    def path_for_run(self, run_id: RunId) -> Path:
        if type(run_id) is not RunId:
            raise TypeError("run_id must be an exact RunId")
        canonical_run_id = RunId(run_id.value)
        return self.root / f"{canonical_run_id.value}.jsonl"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            lock_fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def _read_file_unlocked(self, path: Path) -> Tuple[EventEnvelope, ...]:
        data = path.read_bytes()
        if not data:
            return ()
        if not data.endswith(b"\n"):
            raise CorruptEventLogError(f"Event log has a truncated final line: {path}")
        events: List[EventEnvelope] = []
        for line_number, raw_line in enumerate(data.splitlines(), 1):
            if not raw_line.strip():
                raise CorruptEventLogError(f"Blank event line at {path}:{line_number}")
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = None
            if line is None:
                raise CorruptEventLogError(
                    f"Malformed event at {path}:{line_number}"
                )
            try:
                event = event_from_json(line)
            except EventCodecError:
                event = None
            if event is None:
                raise CorruptEventLogError(
                    f"Malformed event at {path}:{line_number}"
                )
            events.append(event)
        return tuple(events)

    def _load_all_unlocked(
        self,
    ) -> tuple[Dict[RunId, Tuple[EventEnvelope, ...]], set[EventId]]:
        streams: Dict[RunId, Tuple[EventEnvelope, ...]] = {}
        event_ids: set[EventId] = set()
        for path in sorted(self.root.glob("*.jsonl")):
            events = self._read_file_unlocked(path)
            if not events:
                # Empty files have no authoritative run identity yet.
                continue
            run_id = events[0].run_id
            expected_name = f"{run_id.value}.jsonl"
            if path.name != expected_name:
                raise EventRunMismatchError(
                    f"Stream file {path.name!r} contains run {run_id}; "
                    f"expected file {expected_name!r}"
                )
            try:
                validate_loaded_stream(run_id, events)
            except (EventSequenceError, EventRunMismatchError, DuplicateEventIdError) as exc:
                raise CorruptEventLogError(f"Invalid event stream {path}: {exc}") from exc
            duplicate = event_ids.intersection(event.event_id for event in events)
            if duplicate:
                repeated = sorted(str(event_id) for event_id in duplicate)
                raise CorruptEventLogError(f"Duplicate event ID across streams: {repeated}")
            event_ids.update(event.event_id for event in events)
            streams[run_id] = events
        return streams, event_ids

    @staticmethod
    def _append_bytes(path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:  # pragma: no cover - defensive OS failure guard.
                    raise OSError("zero-byte write while appending event")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _fsync_root_directory(self) -> None:
        """Make creation of a new stream name durable on POSIX filesystems."""

        if os.name != "posix":  # pragma: no cover - directory fsync is POSIX-specific.
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(self.root, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def append(
        self,
        event: EventEnvelope,
        *,
        expected_previous_sequence: int,
    ) -> None:
        snapshot = prepare_event_for_append(event)
        encoded = (event_to_json(snapshot) + "\n").encode("utf-8")
        with self._locked():
            streams, event_ids = self._load_all_unlocked()
            existing = streams.get(snapshot.run_id, ())
            validate_append(
                existing,
                snapshot,
                expected_previous_sequence=expected_previous_sequence,
                known_event_ids=event_ids,
            )
            self._append_bytes(self.path_for_run(snapshot.run_id), encoded)
            self._fsync_root_directory()

    def read(
        self,
        run_id: RunId,
        *,
        after_sequence: int = 0,
    ) -> Tuple[EventEnvelope, ...]:
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be a non-negative integer")
        if type(run_id) is not RunId:
            raise TypeError("run_id must be an exact RunId")
        canonical_run_id = RunId(run_id.value)
        with self._locked():
            streams, _ = self._load_all_unlocked()
            return tuple(
                event
                for event in streams.get(canonical_run_id, ())
                if event.sequence_number > after_sequence
            )

"""Durable JSONL interception for sequential residual-evolution phases."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from agent_evolve.application.sequential_residual_portfolio_evolution import (
    SequentialResidualPhaseCommitAck,
    SequentialResidualPhaseReceipt,
)
from agent_evolve.domain.typed_json import freeze_json


DURABLE_SEQUENTIAL_PHASE_JOURNAL_ID = (
    "durable_jsonl_sequential_phase_journal"
)
DURABLE_SEQUENTIAL_PHASE_JOURNAL_VERSION = 1
DURABLE_SEQUENTIAL_PHASE_JOURNAL_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:durable-jsonl-sequential-phase-journal:v1;"
    b"record=full-receipt-evidence-plus-authenticated-ack;"
    b"write=posix-append-before-fsync;"
    b"first-create=parent-directory-fsync;"
    b"duplicate-phase-receipt=fail-closed"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":  # pragma: no cover - production target is POSIX.
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(slots=True)
class DurableJsonlSequentialPhaseCommitter:
    """Append and fsync every phase before the runtime may continue."""

    path: Path
    committer_id: str = field(
        init=False,
        default=DURABLE_SEQUENTIAL_PHASE_JOURNAL_ID,
    )
    committer_version: int = field(
        init=False,
        default=DURABLE_SEQUENTIAL_PHASE_JOURNAL_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=DURABLE_SEQUENTIAL_PHASE_JOURNAL_DEFINITION_SHA256,
    )
    _seen_receipt_sha256s: set[str] = field(
        init=False,
        default_factory=set,
    )
    _commit_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if self.path.exists() and not self.path.is_file():
            raise ValueError("phase journal path must be a regular file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return
        for line_number, raw in enumerate(
            self.path.read_bytes().splitlines(),
            start=1,
        ):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                receipt_sha256 = row["receipt"]["receipt_sha256"]
                sequence = row["sequence"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid sequential phase journal line {line_number}"
                ) from error
            if (
                type(receipt_sha256) is not str
                or len(receipt_sha256) != 64
                or type(sequence) is not int
                or sequence <= 0
            ):
                raise ValueError(
                    f"invalid sequential phase journal line {line_number}"
                )
            if receipt_sha256 in self._seen_receipt_sha256s:
                raise ValueError("phase journal repeats a receipt")
            self._seen_receipt_sha256s.add(receipt_sha256)
            self._commit_count = max(self._commit_count, sequence)

    async def commit(
        self,
        receipt: SequentialResidualPhaseReceipt,
    ) -> SequentialResidualPhaseCommitAck:
        if type(receipt) is not SequentialResidualPhaseReceipt:
            raise TypeError("receipt must be exact")
        receipt.__post_init__()
        if receipt.receipt_sha256 in self._seen_receipt_sha256s:
            raise ValueError("phase receipt was already durably committed")
        sequence = self._commit_count + 1
        ack = SequentialResidualPhaseCommitAck(
            committer_id=self.committer_id,
            committer_version=self.committer_version,
            committer_definition_sha256=self.definition_sha256,
            phase_receipt_sha256=receipt.receipt_sha256,
            durable=True,
            evidence=freeze_json(
                {
                    "backend": "posix_append_fsync",
                    "append_sequence": sequence,
                    "receipt_evidence_included": True,
                    "parent_directory_fsynced_on_create": True,
                }
            ),
        )
        row = {
            "schema_version": 1,
            "sequence": sequence,
            "receipt": receipt.to_record(include_evidence=True),
            "ack": ack.to_record(include_evidence=True),
        }
        payload = _canonical_json(row) + b"\n"
        created = not self.path.exists()
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:  # pragma: no cover - defensive OS boundary.
                    raise OSError("phase journal append made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            _fsync_directory(self.path.parent)
        self._seen_receipt_sha256s.add(receipt.receipt_sha256)
        self._commit_count = sequence
        return ack


__all__ = [
    "DURABLE_SEQUENTIAL_PHASE_JOURNAL_DEFINITION_SHA256",
    "DURABLE_SEQUENTIAL_PHASE_JOURNAL_ID",
    "DURABLE_SEQUENTIAL_PHASE_JOURNAL_VERSION",
    "DurableJsonlSequentialPhaseCommitter",
]

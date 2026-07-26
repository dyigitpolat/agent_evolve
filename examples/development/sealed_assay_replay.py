"""Provider-forbidden replay composition for finalized development assays.

The generic replay implementation deliberately accepts exact sealed files rather
than experiment directories.  This small composition helper bridges a finalized
development run into that port without teaching the core about any benchmark or
about the development artifact layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_evolve.integrations.pydantic_ai.sealed_output_replay import (
    SealedAcceptedOutputReplaySource,
    SealedReplayJsonlFile,
    SealedReplayThenLiveStructuredRunner,
    load_sealed_accepted_output_replay_jsonl,
)
from agent_evolve.ports.structured_generator import StructuredGenerationRequest
from examples.development.durable_run_artifacts import (
    verify_finalized_run_directory,
)


@dataclass(frozen=True, slots=True)
class ProviderForbiddenQueueSnapshot:
    """Queue-shaped zero-work snapshot used by a replay-only continuation."""

    closed: bool
    in_flight: int = 0
    pending: int = 0
    max_in_flight: int = 0
    max_pending: int = 0


class ProviderForbiddenStructuredRunner:
    """Fail closed if an allegedly replay-only assay reaches live execution."""

    def __init__(self) -> None:
        self._closed = False

    async def __call__(self, request: StructuredGenerationRequest[Any]) -> Any:
        del request
        raise RuntimeError("sealed replay exhausted before the assay forecast wave")

    async def aclose(self) -> None:
        self._closed = True

    async def snapshot(self) -> ProviderForbiddenQueueSnapshot:
        return ProviderForbiddenQueueSnapshot(closed=self._closed)


@dataclass(frozen=True, slots=True)
class FinalizedAssayReplay:
    """Verified replay source plus its provider-forbidden low-level runner."""

    source_run_dir: Path
    source_finalization_sha256: str
    source: SealedAcceptedOutputReplaySource
    runner: SealedReplayThenLiveStructuredRunner[Any]


def _sealed_file(
    *,
    source_run_dir: Path,
    files: dict[str, object],
    name: str,
) -> SealedReplayJsonlFile:
    record = files.get(name)
    if type(record) is not dict:
        raise RuntimeError(f"finalized replay source omits {name}")
    digest = record.get("sha256")
    if type(digest) is not str:
        raise RuntimeError(f"finalized replay source has no digest for {name}")
    return SealedReplayJsonlFile(source_run_dir / name, digest)


def build_finalized_assay_replay(
    *,
    source_run_dir: Path,
    requested_model: str,
    decision_receipt_sink: Callable[[dict[str, object]], None],
) -> FinalizedAssayReplay:
    """Verify a finalized run and expose only its accepted structured outputs."""

    root = source_run_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    files = finalization.get("files")
    if type(files) is not dict:
        raise RuntimeError("finalized replay source has no file map")
    source = load_sealed_accepted_output_replay_jsonl(
        source_id=root.name,
        request_evidence=_sealed_file(
            source_run_dir=root,
            files=files,
            name="request_evidence.jsonl",
        ),
        output_evidence=_sealed_file(
            source_run_dir=root,
            files=files,
            name="output_evidence.jsonl",
        ),
        terminal_outcomes=_sealed_file(
            source_run_dir=root,
            files=files,
            name="queue_outcomes.jsonl",
        ),
    )
    runner = SealedReplayThenLiveStructuredRunner(
        source=source,
        requested_model=requested_model,
        live_runner=ProviderForbiddenStructuredRunner(),
        decision_receipt_sink=decision_receipt_sink,
    )
    finalization_sha256 = finalization.get("finalization_sha256")
    if type(finalization_sha256) is not str:
        raise RuntimeError("finalized replay source has no finalization identity")
    return FinalizedAssayReplay(
        source_run_dir=root,
        source_finalization_sha256=finalization_sha256,
        source=source,
        runner=runner,
    )


__all__ = [
    "FinalizedAssayReplay",
    "ProviderForbiddenQueueSnapshot",
    "ProviderForbiddenStructuredRunner",
    "build_finalized_assay_replay",
]

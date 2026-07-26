"""Thread-safe in-memory content-addressed artifact store."""

from __future__ import annotations

import threading

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
)


class InMemoryArtifactStore:
    """Verified ephemeral store used by tests and non-durable runs."""

    def __init__(self) -> None:
        self._artifacts: dict[ArtifactId, tuple[ArtifactRef, bytes]] = {}
        self._lock = threading.RLock()

    def put_bytes(self, content: bytes, *, media_type: str) -> ArtifactRef:
        require_content_and_media_type(content, media_type)
        ref = artifact_domain.artifact_ref_for_bytes(content, media_type=media_type)
        with self._lock:
            existing = self._artifacts.get(ref.artifact_id)
            if existing is None:
                self._artifacts[ref.artifact_id] = (ref, content)
                return ref

            existing_ref, existing_content = existing
            verify_content(existing_ref, existing_content)
            if existing_content != content:
                raise ArtifactCollisionError(
                    f"Different payload bytes resolved to {ref.artifact_id}"
                )
            if existing_ref.media_type != media_type:
                raise ArtifactCollisionError(
                    f"Different media types resolved to {ref.artifact_id}"
                )
            return existing_ref

    def _verified(self, artifact_id: ArtifactId) -> tuple[ArtifactRef, bytes]:
        require_artifact_id(artifact_id)
        existing = self._artifacts.get(artifact_id)
        if existing is None:
            raise ArtifactNotFoundError(f"Artifact {artifact_id} was not found")
        ref, content = existing
        verify_content(ref, content)
        return ref, content

    def stat(self, artifact_id: ArtifactId) -> ArtifactRef:
        with self._lock:
            ref, _ = self._verified(artifact_id)
            return ref

    def read_bytes(
        self,
        artifact_id: ArtifactId,
        *,
        expected_media_type: str | None = None,
    ) -> bytes:
        with self._lock:
            ref, content = self._verified(artifact_id)
            verify_expected_media_type(ref, expected_media_type)
            return content

"""Ports for artifact minimization and deterministic secret sanitization."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_evolve.domain.artifact import ArtifactRole


class ArtifactMinimizationError(ValueError):
    """A value cannot be minimized under the selected role policy."""


class ArtifactSanitizationError(ValueError):
    """A value cannot be made safe under the selected sanitization policy."""


@runtime_checkable
class ArtifactMinimizer(Protocol):
    """Remove fields that are unnecessary for a role before redaction."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    @property
    def policy_config_sha256(self) -> str: ...

    def minimize_json(self, value: Any, *, role: ArtifactRole) -> Any: ...


@runtime_checkable
class ArtifactSanitizer(Protocol):
    """Return a detached JSON value safe to pass to canonical encoding."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    def sanitize_json(self, value: Any, *, role: ArtifactRole) -> Any: ...

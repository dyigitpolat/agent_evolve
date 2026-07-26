"""Conservative, deterministic sanitization for canonical JSON artifacts.

This adapter deliberately makes a bounded claim: it handles configured literal
secrets and a documented set of common credential forms, then rejects known
high-risk residual material.  It is not advertised as a general DLP system.
"""

from __future__ import annotations

import math
import re
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from agent_evolve.domain.artifact import ArtifactRole
from agent_evolve.ports.artifact_sanitizer import (
    ArtifactMinimizationError,
    ArtifactSanitizationError,
)

_REDACTED = "[REDACTED]"
_FAILED = object()
_POLICY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEY_NAMES = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "authtoken",
        "clientsecret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passwd",
        "privatekey",
        "proxyauthorization",
        "refreshtoken",
        "secret",
        "setcookie",
        "token",
    }
)
_SECRET_KEY_SUFFIXES = (
    "accesstoken",
    "accesskeyid",
    "apikey",
    "authtoken",
    "clientsecret",
    "credential",
    "credentials",
    "password",
    "passwd",
    "privatekey",
    "refreshtoken",
    "authorization",
    "cookie",
    "secret",
    "secretaccesskey",
    "secretkey",
    "sessiontoken",
    "signingkey",
    "token",
)

# Each expression replaces the complete credential-bearing match.  Patterns are
# ordered from contextual headers/assignments to standalone token signatures.
_CREDENTIAL_PATTERNS = (
    re.compile(
        r"\b(?:Bearer|Basic)[ \t]+[A-Za-z0-9._~+/=-]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9])(?:api[ _-]?key|access[ _-]?key(?:[ _-]?id)?|"
        r"secret[ _-]?access[ _-]?key|session[ _-]?token|"
        r"access[ _-]?token|auth[ _-]?token|refresh[ _-]?token|"
        r"client[ _-]?secret|authorization|password|passwd|"
        r"credential|token|secret)"
        r"[ \t]*[:=][ \t]*(?:\"[^\"\r\n]+\"|'[^'\r\n]+'|[^\s,;}{]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:Cookie|Set-Cookie)[ \t]*:[^\r\n]+", re.IGNORECASE),
    re.compile(
        r"(?<![A-Za-z0-9])(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9])or-v1-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9])(?:ghp_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])"
    ),
    re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"),
    re.compile(
        r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9])"
    ),
)
_PRIVATE_KEY_MATERIAL = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----", re.IGNORECASE
)
_CREDENTIAL_IN_URL = re.compile(
    # Bound the scheme scan so long punctuation-heavy nonmatches remain linear.
    r"\b[a-z][a-z0-9+.-]{0,31}://[^/\s:@]+:[^/\s@]+@", re.IGNORECASE
)


def _generic_minimization_failure() -> ArtifactMinimizationError:
    return ArtifactMinimizationError(
        "artifact minimization failed under the configured role policy"
    )


def _generic_sanitization_failure() -> ArtifactSanitizationError:
    return ArtifactSanitizationError(
        "artifact sanitization rejected unsafe or unsupported content"
    )


class TopLevelAllowlistMinimizer:
    """Keep only explicitly allowed top-level fields for each artifact role.

    Every role that may be used must be configured.  There is intentionally no
    implicit identity fallback: a missing policy is a fail-closed error.
    """

    __slots__ = ("_allowed", "_policy_config_sha256")

    def __init__(
        self,
        allowed_fields_by_role: Mapping[ArtifactRole, Iterable[str]],
    ) -> None:
        prepared = self._prepare(allowed_fields_by_role)
        if prepared is _FAILED:
            raise _generic_minimization_failure()
        assert isinstance(prepared, tuple)
        self._allowed, self._policy_config_sha256 = prepared

    @staticmethod
    def _prepare(
        allowed_fields_by_role: Mapping[ArtifactRole, Iterable[str]],
    ) -> tuple[MappingProxyType, str] | object:
        try:
            prepared: dict[ArtifactRole, frozenset[str]] = {}
            for role, fields in allowed_fields_by_role.items():
                if not isinstance(role, ArtifactRole) or isinstance(fields, str):
                    raise _generic_minimization_failure()
                validated_fields: list[str] = []
                for field in fields:
                    if (
                        type(field) is not str
                        or not field
                        or field != field.strip()
                    ):
                        raise _generic_minimization_failure()
                    field.encode("utf-8", errors="strict")
                    validated_fields.append(field)
                field_set = frozenset(validated_fields)
                prepared[role] = field_set
            policy_record = {
                role.value: sorted(field_set)
                for role, field_set in prepared.items()
            }
            encoded_policy = json.dumps(
                policy_record,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8", errors="strict")
            return (
                MappingProxyType(prepared),
                hashlib.sha256(encoded_policy).hexdigest(),
            )
        except Exception:
            return _FAILED

    @property
    def policy_id(self) -> str:
        return "top-level-allowlist"

    @property
    def policy_version(self) -> str:
        return "1"

    @property
    def policy_config_sha256(self) -> str:
        return self._policy_config_sha256

    def minimize_json(self, value: Any, *, role: ArtifactRole) -> Any:
        minimized = self._attempt_minimize(value, role)
        if minimized is _FAILED:
            raise _generic_minimization_failure()
        return minimized

    def _attempt_minimize(self, value: Any, role: ArtifactRole) -> Any:
        try:
            if not isinstance(role, ArtifactRole):
                raise _generic_minimization_failure()
            allowed = self._allowed.get(role)
            if allowed is None or type(value) is not dict:
                raise _generic_minimization_failure()
            # Sorting is not required by the canonical encoder, but makes the
            # intermediate result deterministic.  Iterating the small allowlist
            # also avoids touching or sorting fields that minimization discards.
            return {
                key: value[key]
                for key in sorted(allowed)
                if key in value
            }
        except Exception:
            return _FAILED


class StrictJsonSanitizer:
    """Recursively redact a strict JSON value under a versioned policy."""

    __slots__ = (
        "_exact_secrets",
        "_max_depth",
        "_max_nodes",
        "_max_string_bytes",
        "_max_total_string_bytes",
        "_policy_id",
        "_policy_version",
    )

    def __init__(
        self,
        *,
        exact_secret_values: Iterable[str] = (),
        policy_id: str = "strict-json-redaction",
        policy_version: str = "1",
        max_depth: int = 64,
        max_nodes: int = 100_000,
        max_string_bytes: int = 1_000_000,
        max_total_string_bytes: int = 4_000_000,
    ) -> None:
        prepared = self._prepare_configuration(
            exact_secret_values=exact_secret_values,
            policy_id=policy_id,
            policy_version=policy_version,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_string_bytes=max_string_bytes,
            max_total_string_bytes=max_total_string_bytes,
        )
        if prepared is _FAILED:
            raise _generic_sanitization_failure()
        (
            self._exact_secrets,
            self._policy_id,
            self._policy_version,
            self._max_depth,
            self._max_nodes,
            self._max_string_bytes,
            self._max_total_string_bytes,
        ) = prepared

    @staticmethod
    def _prepare_configuration(
        *,
        exact_secret_values: Iterable[str],
        policy_id: str,
        policy_version: str,
        max_depth: int,
        max_nodes: int,
        max_string_bytes: int,
        max_total_string_bytes: int,
    ) -> tuple[Any, ...] | object:
        try:
            if (
                not isinstance(policy_id, str)
                or _POLICY_COMPONENT.fullmatch(policy_id) is None
                or not isinstance(policy_version, str)
                or _POLICY_COMPONENT.fullmatch(policy_version) is None
            ):
                raise _generic_sanitization_failure()
            if (
                isinstance(max_depth, bool)
                or not isinstance(max_depth, int)
                or max_depth < 1
                or isinstance(max_nodes, bool)
                or not isinstance(max_nodes, int)
                or max_nodes < 1
                or isinstance(max_string_bytes, bool)
                or not isinstance(max_string_bytes, int)
                or max_string_bytes < 1
                or isinstance(max_total_string_bytes, bool)
                or not isinstance(max_total_string_bytes, int)
                or max_total_string_bytes < 1
            ):
                raise _generic_sanitization_failure()
            secrets: set[str] = set()
            for secret in exact_secret_values:
                if (
                    not isinstance(secret, str)
                    or len(secret) < 8
                    or secret in _REDACTED
                ):
                    raise _generic_sanitization_failure()
                encoded_secret = secret.encode("utf-8", errors="strict")
                if len(encoded_secret) > max_string_bytes:
                    raise _generic_sanitization_failure()
                secrets.add(secret)
            if any(
                secret in policy_id or secret in policy_version
                for secret in secrets
            ):
                raise _generic_sanitization_failure()
            if (
                _PRIVATE_KEY_MATERIAL.search(policy_id)
                or _PRIVATE_KEY_MATERIAL.search(policy_version)
                or _CREDENTIAL_IN_URL.search(policy_id)
                or _CREDENTIAL_IN_URL.search(policy_version)
                or any(
                    pattern.search(component)
                    for pattern in _CREDENTIAL_PATTERNS
                    for component in (policy_id, policy_version)
                )
            ):
                raise _generic_sanitization_failure()
            # Longest-first replacement prevents a shorter configured value from
            # exposing the unmatched tail of an overlapping longer value.
            exact_secrets = tuple(
                sorted(secrets, key=lambda item: (-len(item), item))
            )
            return (
                exact_secrets,
                policy_id,
                policy_version,
                max_depth,
                max_nodes,
                max_string_bytes,
                max_total_string_bytes,
            )
        except Exception:
            return _FAILED

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(policy_id={self._policy_id!r}, "
            f"policy_version={self._policy_version!r})"
        )

    @property
    def policy_id(self) -> str:
        return self._policy_id

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @staticmethod
    def _is_secret_key(key: str) -> bool:
        normalized = "".join(character.lower() for character in key if character.isalnum())
        return normalized in _SECRET_KEY_NAMES or normalized.endswith(
            _SECRET_KEY_SUFFIXES
        )

    def _sanitize_valid_string(self, value: str, *, is_key: bool = False) -> str:
        if is_key:
            # Renaming keys could create collisions or alter a schema silently.
            if any(secret in value for secret in self._exact_secrets):
                raise _generic_sanitization_failure()
            if (
                _PRIVATE_KEY_MATERIAL.search(value)
                or _CREDENTIAL_IN_URL.search(value)
                or any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS)
            ):
                raise _generic_sanitization_failure()
            return value

        sanitized = value
        for secret in self._exact_secrets:
            sanitized = sanitized.replace(secret, _REDACTED)
        for pattern in _CREDENTIAL_PATTERNS:
            sanitized = pattern.sub(_REDACTED, sanitized)

        # These forms are too structurally sensitive to persist after an
        # in-place rewrite; fail closed and let the caller supply a minimizer.
        if _PRIVATE_KEY_MATERIAL.search(sanitized) or _CREDENTIAL_IN_URL.search(
            sanitized
        ):
            raise _generic_sanitization_failure()
        if any(secret in sanitized for secret in self._exact_secrets):
            raise _generic_sanitization_failure()
        return sanitized

    def sanitize_json(self, value: Any, *, role: ArtifactRole) -> Any:
        sanitized = self._attempt_sanitize(value, role)
        if sanitized is _FAILED:
            raise _generic_sanitization_failure()
        return sanitized

    def _attempt_sanitize(self, value: Any, role: ArtifactRole) -> Any:
        ancestors: set[int] = set()
        nodes_seen = 0
        string_bytes_seen = 0

        def reserve_nodes(count: int = 1) -> None:
            nonlocal nodes_seen
            nodes_seen += count
            if nodes_seen > self._max_nodes:
                raise _generic_sanitization_failure()

        def sanitize_string(item: str, *, is_key: bool = False) -> str:
            nonlocal string_bytes_seen
            encoded = item.encode("utf-8", errors="strict")
            size = len(encoded)
            string_bytes_seen += size
            if (
                size > self._max_string_bytes
                or string_bytes_seen > self._max_total_string_bytes
            ):
                raise _generic_sanitization_failure()
            return self._sanitize_valid_string(item, is_key=is_key)

        def visit(item: Any, depth: int) -> Any:
            reserve_nodes()
            if depth > self._max_depth:
                raise _generic_sanitization_failure()

            if item is None or type(item) in (bool, int):
                return item
            if type(item) is float:
                if not math.isfinite(item):
                    raise _generic_sanitization_failure()
                return item
            if type(item) is str:
                return sanitize_string(item)
            if type(item) in (list, tuple):
                if len(item) > self._max_nodes - nodes_seen:
                    raise _generic_sanitization_failure()
                identity = id(item)
                if identity in ancestors:
                    raise _generic_sanitization_failure()
                ancestors.add(identity)
                try:
                    return [visit(child, depth + 1) for child in item]
                finally:
                    ancestors.remove(identity)
            if type(item) is dict:
                # Every entry accounts for one key and at least one value slot.
                # Gate this before sorting to avoid unbounded pre-check work.
                if len(item) * 2 > self._max_nodes - nodes_seen:
                    raise _generic_sanitization_failure()
                identity = id(item)
                if identity in ancestors:
                    raise _generic_sanitization_failure()
                ancestors.add(identity)
                try:
                    result: dict[str, Any] = {}
                    validated_keys: list[str] = []
                    for key in item:
                        reserve_nodes()  # object key
                        if type(key) is not str:
                            raise _generic_sanitization_failure()
                        validated_keys.append(sanitize_string(key, is_key=True))
                    for clean_key in sorted(validated_keys):
                        if self._is_secret_key(clean_key):
                            reserve_nodes()  # redacted value slot
                            result[clean_key] = _REDACTED
                        else:
                            result[clean_key] = visit(item[clean_key], depth + 1)
                    return result
                finally:
                    ancestors.remove(identity)
            raise _generic_sanitization_failure()

        try:
            if not isinstance(role, ArtifactRole):
                raise _generic_sanitization_failure()
            return visit(value, 0)
        except Exception:
            # Returning clears implicit exception context before the public
            # method constructs its generic error.
            return _FAILED

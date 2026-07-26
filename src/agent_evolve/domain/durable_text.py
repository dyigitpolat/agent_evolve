"""Shared detectors for text that must not enter durable inline metadata.

These bounded detectors are deliberately dependency-free so identifiers and
other event-text policies use exactly the same credential signatures. They are
defense in depth, not a general data-loss-prevention system.
"""

from __future__ import annotations

import re


_CREDENTIAL_SHAPE = re.compile(
    r"(?:\b(?:Bearer|Basic)[ \t]+\S+|"
    r"\b(?:Cookie|Set-Cookie)[ \t]*:[^\r\n]+|"
    r"(?<![A-Za-z0-9])(?:api[ _-]?key|access[ _-]?key(?:[ _-]?id)?|"
    r"secret[ _-]?access[ _-]?key|session[ _-]?token|access[ _-]?token|"
    r"auth[ _-]?token|refresh[ _-]?token|client[ _-]?secret|"
    r"authorization|password|passwd|credential|token|secret)"
    r"[ \t]*[:=][ \t]*\S+|"
    r"(?<![A-Za-z0-9])(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])or-v1-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])(?:ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{12,}(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])|"
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----|"
    r"\b[a-z][a-z0-9+.-]{0,31}://[^/\s:@]+:[^/\s@]+@)",
    re.IGNORECASE,
)
_INLINE_CONTENT_MARKER = re.compile(
    r"(?:</?(?:system|user|assistant)>|"
    r"(?:^|[^A-Za-z0-9])(?:system|user|assistant)[._ -]?"
    r"(?:prompt|message|response|content)(?:$|[^A-Za-z0-9])|"
    r"(?:^|[^A-Za-z0-9])return[._ -]?(?:json|yaml|xml)"
    r"(?:$|[^A-Za-z0-9]))",
    re.IGNORECASE,
)
_IDENTIFIER_CONTENT_MARKER = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:"
    r"(?:system|user|assistant)[._-]?(?:prompt|message|response|content)|"
    r"prompt|response|message|content|diagnostic|diagnostics|"
    r"(?:raw|tool)[._-]?(?:input|output)|"
    r"return[._-]?(?:json|yaml|xml)|"
    r"authorization|bearer|cookie|password|passwd|credential|secret|token|"
    r"(?:api|access|session|auth|refresh)[._-]?(?:key|token)|"
    r"client[._-]?secret"
    r")(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


def contains_credential_shape(value: str) -> bool:
    """Return whether *value* resembles one of the bounded credential forms."""

    return _CREDENTIAL_SHAPE.search(value) is not None


def contains_inline_content_marker(value: str) -> bool:
    """Return whether free text contains an obvious prompt/content marker."""

    return _INLINE_CONTENT_MARKER.search(value) is not None


def contains_identifier_content_marker(value: str) -> bool:
    """Return whether an ID embeds content- or credential-oriented tokens."""

    return _IDENTIFIER_CONTENT_MARKER.search(value) is not None

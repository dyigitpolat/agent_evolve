"""Fail-closed policies for small strings that may enter the durable event log.

Prompt, response, tool output, headers, and diagnostic bodies belong in the
sanitized artifact journal.  Event strings are limited to hashes, routing
labels, stable metadata tokens, exception type names, or short safe summaries.
"""

from __future__ import annotations

import re
from enum import Enum

from agent_evolve.domain.artifact import validate_media_type
from agent_evolve.domain.durable_text import (
    contains_credential_shape,
    contains_inline_content_marker,
)


class InlineTextPolicy(str, Enum):
    SHA256 = "sha256"
    MEDIA_TYPE = "media_type"
    POLICY_COMPONENT = "policy_component"
    ENUM_VALUE = "enum_value"
    METADATA_TOKEN = "metadata_token"
    ROUTING_LABEL = "routing_label"
    EXCEPTION_TYPE = "exception_type"
    SAFE_SUMMARY = "safe_summary"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POLICY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_METADATA_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ROUTING_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_EXCEPTION_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_OPAQUE_RUN = re.compile(
    r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_+/=-]{32,}(?![A-Za-z0-9_+/=-])"
)


def _fail(field_name: str, policy: InlineTextPolicy) -> ValueError:
    return ValueError(
        f"{field_name} violates the durable inline-text policy {policy.value}"
    )


def validate_inline_text(
    value: str,
    *,
    field_name: str,
    policy: InlineTextPolicy,
) -> None:
    """Validate a string without echoing its content in any error."""

    if not _is_valid_inline_text(value, policy):
        raise _fail(field_name, policy)


def _is_valid_inline_text(value: str, policy: InlineTextPolicy) -> bool:
    if type(value) is not str or not isinstance(policy, InlineTextPolicy):
        return False
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        return False
    if not value or value != value.strip() or contains_credential_shape(value):
        return False
    if policy in (
        InlineTextPolicy.MEDIA_TYPE,
        InlineTextPolicy.POLICY_COMPONENT,
        InlineTextPolicy.ENUM_VALUE,
        InlineTextPolicy.METADATA_TOKEN,
        InlineTextPolicy.ROUTING_LABEL,
        InlineTextPolicy.EXCEPTION_TYPE,
    ) and contains_inline_content_marker(value):
        return False

    if policy is InlineTextPolicy.SHA256:
        return _SHA256.fullmatch(value) is not None
    if policy is InlineTextPolicy.MEDIA_TYPE:
        try:
            validate_media_type(value)
        except (TypeError, ValueError):
            return False
        return True
    if policy is InlineTextPolicy.POLICY_COMPONENT:
        return _POLICY_COMPONENT.fullmatch(value) is not None
    if policy is InlineTextPolicy.ENUM_VALUE:
        return _METADATA_TOKEN.fullmatch(value) is not None
    if policy is InlineTextPolicy.METADATA_TOKEN:
        return _METADATA_TOKEN.fullmatch(value) is not None
    if policy is InlineTextPolicy.ROUTING_LABEL:
        return _ROUTING_LABEL.fullmatch(value) is not None
    if policy is InlineTextPolicy.EXCEPTION_TYPE:
        return _EXCEPTION_TYPE.fullmatch(value) is not None
    if policy is not InlineTextPolicy.SAFE_SUMMARY:  # pragma: no cover - enum guard.
        return False

    return not (
        len(encoded) > 160
        or any(ord(character) < 32 or ord(character) > 126 for character in value)
        or len(value.split()) > 24
        or _OPAQUE_RUN.search(value)
        or contains_inline_content_marker(value)
        or any(character in value for character in "{}[]\r\n\t")
    )

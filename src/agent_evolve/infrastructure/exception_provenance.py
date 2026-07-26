"""Privacy-safe, bounded exception-origin projection.

This module is framework-neutral.  It converts Python exception topology into
closed domain values without reading exception messages, reprs, arguments,
tracebacks, response objects, URLs, payloads, or arbitrary metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque

from agent_evolve.domain.llm_task_queue import (
    MAX_EXCEPTION_PROVENANCE_NODES,
    ExceptionOriginFamily,
    ExceptionProvenanceLink,
    SanitizedExceptionProvenance,
    SanitizedExceptionProvenanceNode,
)


EXCEPTION_TYPE_IDENTITY_FINGERPRINT_ALGORITHM = (
    "sha256_domain_and_bounded_module_qualname_projection_v1"
)
_EXCEPTION_TYPE_IDENTITY_DOMAIN = (
    b"agent-evolve:sanitized-exception-type-identity:v1\x00"
)
EXCEPTION_TYPE_IDENTITY_DOMAIN_SHA256 = hashlib.sha256(
    _EXCEPTION_TYPE_IDENTITY_DOMAIN
).hexdigest()
_MAX_TYPE_COMPONENT_CHARS = 512
_TYPE_COMPONENT_EDGE_CHARS = 128


def _runtime_type_text(exception_type: type[BaseException], name: str) -> str | None:
    """Read type metadata without invoking a custom metaclass override."""

    try:
        value = type.__getattribute__(exception_type, name)
    except BaseException:
        return None
    return value if type(value) is str else None


def _bounded_type_component(value: str | None) -> dict[str, object]:
    """Project type metadata with fixed memory independent of input length."""

    if value is None:
        return {"state": "absent", "length": 0, "prefix": "", "suffix": ""}
    length = len(value)
    if length <= _MAX_TYPE_COMPONENT_CHARS:
        return {
            "state": "complete",
            "length": length,
            "prefix": value,
            "suffix": "",
        }
    return {
        "state": "truncated",
        "length": length,
        "prefix": value[:_TYPE_COMPONENT_EDGE_CHARS],
        "suffix": value[-_TYPE_COMPONENT_EDGE_CHARS:],
    }


def exception_type_identity_sha256(exception_type: type[BaseException]) -> str:
    """Return a stable, redacted fingerprint for one runtime exception type."""

    if not isinstance(exception_type, type) or not issubclass(
        exception_type, BaseException
    ):
        raise TypeError("exception_type must be a BaseException type")
    record = {
        "schema_version": 1,
        "module": _bounded_type_component(
            _runtime_type_text(exception_type, "__module__")
        ),
        "qualname": _bounded_type_component(
            _runtime_type_text(exception_type, "__qualname__")
        ),
    }
    canonical = json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(_EXCEPTION_TYPE_IDENTITY_DOMAIN + canonical).hexdigest()


def _origin_family(exception_type: type[BaseException]) -> ExceptionOriginFamily:
    module = _runtime_type_text(exception_type, "__module__")
    if module is None:
        return ExceptionOriginFamily.OTHER
    roots = (
        ("builtins", ExceptionOriginFamily.BUILTINS),
        ("asyncio", ExceptionOriginFamily.ASYNCIO),
        ("anyio", ExceptionOriginFamily.ANYIO),
        ("httpx", ExceptionOriginFamily.HTTPX),
        ("httpcore", ExceptionOriginFamily.HTTPCORE),
        ("openai", ExceptionOriginFamily.OPENAI),
        ("pydantic_ai", ExceptionOriginFamily.PYDANTIC_AI),
        ("pydantic", ExceptionOriginFamily.PYDANTIC),
        ("agent_evolve", ExceptionOriginFamily.AGENT_EVOLVE),
    )
    for root, family in roots:
        if module == root or module.startswith(root + "."):
            return family
    return ExceptionOriginFamily.OTHER


def _base_exception_link(
    value: BaseException,
    attribute: str,
) -> BaseException | None:
    try:
        linked = BaseException.__getattribute__(value, attribute)
    except BaseException:
        return None
    return linked if isinstance(linked, BaseException) else None


def _group_members(value: BaseException) -> tuple[BaseException, ...]:
    if not isinstance(value, BaseExceptionGroup):
        return ()
    try:
        members = BaseExceptionGroup.exceptions.__get__(value, type(value))
    except BaseException:
        return ()
    if type(members) is not tuple:
        return ()
    # One extra member is enough to authenticate that the bounded graph was
    # truncated; never enqueue an attacker-sized group in full.
    return tuple(
        member
        for member in members[: MAX_EXCEPTION_PROVENANCE_NODES + 1]
        if isinstance(member, BaseException)
    )


def sanitized_exception_provenance(
    exc: BaseException,
) -> SanitizedExceptionProvenance:
    """Project root/cause/context/group topology into at most sixteen nodes."""

    if not isinstance(exc, BaseException):
        raise TypeError("exc must be a BaseException")
    pending: deque[tuple[BaseException, int | None, ExceptionProvenanceLink]] = deque(
        [(exc, None, ExceptionProvenanceLink.ROOT)]
    )
    nodes: list[SanitizedExceptionProvenanceNode] = []
    seen: set[int] = set()
    overflowed = False

    while pending and len(nodes) < MAX_EXCEPTION_PROVENANCE_NODES:
        current, parent_index, link = pending.popleft()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        node_index = len(nodes)
        exception_type = type(current)
        nodes.append(
            SanitizedExceptionProvenanceNode(
                parent_index=parent_index,
                link=link,
                family=_origin_family(exception_type),
                type_identity_sha256=exception_type_identity_sha256(exception_type),
            )
        )

        linked_values = (
            (
                _base_exception_link(current, "__cause__"),
                ExceptionProvenanceLink.CAUSE,
            ),
            (
                _base_exception_link(current, "__context__"),
                ExceptionProvenanceLink.CONTEXT,
            ),
        )
        children = [
            (linked, node_index, child_link)
            for linked, child_link in linked_values
            if linked is not None and id(linked) not in seen
        ]
        children.extend(
            (member, node_index, ExceptionProvenanceLink.GROUP_MEMBER)
            for member in _group_members(current)
            if id(member) not in seen
        )
        for child in children:
            # Bound pending storage as well as the emitted graph. The extra
            # slot only records that more topology existed.
            if len(nodes) + len(pending) < MAX_EXCEPTION_PROVENANCE_NODES:
                pending.append(child)
            else:
                overflowed = True

    if pending:
        overflowed = True
    return SanitizedExceptionProvenance(
        nodes=tuple(nodes),
        truncated=overflowed,
    )


__all__ = [
    "EXCEPTION_TYPE_IDENTITY_DOMAIN_SHA256",
    "EXCEPTION_TYPE_IDENTITY_FINGERPRINT_ALGORITHM",
    "exception_type_identity_sha256",
    "sanitized_exception_provenance",
]

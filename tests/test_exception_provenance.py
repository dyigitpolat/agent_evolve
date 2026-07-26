"""Privacy and boundedness contracts for exception-origin observability."""

from __future__ import annotations

import json

from agent_evolve.domain.llm_task_queue import (
    MAX_EXCEPTION_PROVENANCE_NODES,
    ExceptionOriginFamily,
    ExceptionProvenanceLink,
)
from agent_evolve.infrastructure.exception_provenance import (
    exception_type_identity_sha256,
    sanitized_exception_provenance,
)


class RAW_SECRET_EXCEPTION_TYPE_DO_NOT_PERSIST(RuntimeError):
    pass


def _safe_record(value) -> dict[str, object]:
    return {
        "truncated": value.truncated,
        "nodes": [
            {
                "parent_index": node.parent_index,
                "link": node.link.value,
                "family": node.family.value,
                "type_identity_sha256": node.type_identity_sha256,
            }
            for node in value.nodes
        ],
    }


def test_exception_provenance_is_bounded_and_retains_no_exception_text() -> None:
    secret = "OPENROUTER_API_KEY=sk-raw-secret https://private.example/payload"
    members = [
        RAW_SECRET_EXCEPTION_TYPE_DO_NOT_PERSIST(f"{secret}:{index}")
        for index in range(MAX_EXCEPTION_PROVENANCE_NODES * 4)
    ]
    group = ExceptionGroup(secret, members)
    root = RuntimeError(secret)
    root.__cause__ = group

    provenance = sanitized_exception_provenance(root)
    record = _safe_record(provenance)
    encoded = json.dumps(record, allow_nan=False, sort_keys=True)

    assert len(provenance.nodes) == MAX_EXCEPTION_PROVENANCE_NODES
    assert provenance.truncated is True
    assert provenance.nodes[0].link is ExceptionProvenanceLink.ROOT
    assert provenance.nodes[0].family is ExceptionOriginFamily.BUILTINS
    assert provenance.nodes[1].link is ExceptionProvenanceLink.CAUSE
    assert provenance.nodes[1].parent_index == 0
    assert any(
        node.link is ExceptionProvenanceLink.GROUP_MEMBER
        for node in provenance.nodes
    )
    for forbidden in (
        secret,
        "sk-raw-secret",
        "private.example",
        "RAW_SECRET_EXCEPTION_TYPE_DO_NOT_PERSIST",
        "payload",
    ):
        assert forbidden not in encoded


def test_exception_type_fingerprint_is_stable_and_type_specific() -> None:
    first = exception_type_identity_sha256(RuntimeError)
    assert first == exception_type_identity_sha256(RuntimeError)
    assert first != exception_type_identity_sha256(ValueError)
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_cyclic_cause_and_context_graph_remains_finite() -> None:
    left = RuntimeError("RAW_SECRET_LEFT")
    right = ValueError("RAW_SECRET_RIGHT")
    left.__cause__ = right
    right.__context__ = left

    provenance = sanitized_exception_provenance(left)

    assert len(provenance.nodes) == 2
    assert provenance.truncated is False
    assert [node.link for node in provenance.nodes] == [
        ExceptionProvenanceLink.ROOT,
        ExceptionProvenanceLink.CAUSE,
    ]

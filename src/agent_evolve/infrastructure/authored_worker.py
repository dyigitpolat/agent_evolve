"""Executes one authored artifact against a batch of calls, bounded.

Runs OUT of the optimizer's process, one process per batch, launched by
``AuthoredRuntime`` through the deny-by-default subprocess boundary. The
threat model is untrusted-BUGGY code, not adversarial code: resource limits
contain runaway loops and allocations, the import gate keeps an artifact from
reaching the filesystem or network by accident, and every failure becomes a
typed status the parent counts -- never an exception in the optimizer.

Order of operations matters and is deliberate: the request is read first
(this module's own imports must work unconstrained), THEN the resource
limits are applied, THEN the source is gated and compiled. Authored code
never runs a single instruction outside the limits.
"""

from __future__ import annotations

import ast
import json
import sys
import traceback

#: Modules an authored artifact may import. Enough for arithmetic, statistics
#: and seeded randomness; nothing that reaches the filesystem, the network,
#: or another process.
ALLOWED_IMPORTS = frozenset(
    {"math", "statistics", "random", "json", "itertools", "collections",
     "functools", "heapq", "bisect"}
)

_DETAIL_LIMIT = 500


def _forbidden_import(source: str) -> str | None:
    """The first disallowed root module the source imports, or ``None``."""

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return root
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                return root or "<relative>"
        elif isinstance(node, ast.Name) and node.id == "__import__":
            return "__import__"
    return None


def _apply_limits(limits: dict) -> None:
    """Best-effort rlimits: containment for buggy code, not a security wall."""

    try:
        import resource
    except ImportError:                      # non-POSIX: wall timeout remains
        return
    cpu = int(limits.get("cpu_seconds", 8))
    memory = int(limits.get("memory_bytes", 512 * 1024 * 1024))
    for kind, value in ((resource.RLIMIT_CPU, (cpu, cpu + 1)),
                        (resource.RLIMIT_AS, (memory, memory))):
        try:
            resource.setrlimit(kind, value)
        except (ValueError, OSError):
            pass


def _respond(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main(argv: list[str]) -> int:
    request_path, response_path = argv
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)

    _apply_limits(request.get("limits") or {})
    source = request["source"]
    entry_point = request["entry_point"]

    try:
        forbidden = _forbidden_import(source)
    except SyntaxError as error:
        _respond(response_path, {"status": "unparseable",
                                 "detail": str(error)[:_DETAIL_LIMIT]})
        return 0
    if forbidden is not None:
        _respond(response_path, {"status": "forbidden_import", "detail": forbidden})
        return 0

    namespace: dict = {"__name__": "authored_artifact"}
    # A HARNESS-WRITTEN prelude, executed into the same namespace before the
    # artifact and never gated: it is ours, not the model's, and it is how the
    # caller hands authored code machinery it should not have to re-derive
    # (see policies/emit_scaffold.py). Absent, nothing changes.
    prelude = request.get("prelude")
    if prelude:
        try:
            exec(compile(prelude, "<harness_prelude>", "exec"), namespace)  # noqa: S102
        except Exception:
            _respond(response_path, {
                "status": "crash",
                "detail": "harness prelude: "
                          + traceback.format_exc()[-_DETAIL_LIMIT:],
            })
            return 0

    try:
        exec(compile(source, "<authored>", "exec"), namespace)  # noqa: S102
    except SyntaxError as error:
        _respond(response_path, {"status": "unparseable",
                                 "detail": str(error)[:_DETAIL_LIMIT]})
        return 0
    except MemoryError:
        _respond(response_path, {"status": "memory", "detail": "during exec"})
        return 0
    except Exception:
        _respond(response_path, {"status": "crash",
                                 "detail": traceback.format_exc()[-_DETAIL_LIMIT:]})
        return 0

    fn = namespace.get(entry_point)
    if not callable(fn):
        _respond(response_path, {
            "status": "unparseable",
            "detail": f"source defines no callable named {entry_point!r}",
        })
        return 0

    results = []
    try:
        for call in request.get("calls", []):
            results.append(fn(*call))
    except MemoryError:
        _respond(response_path, {"status": "memory",
                                 "detail": f"after {len(results)} calls"})
        return 0
    except Exception:
        _respond(response_path, {"status": "crash",
                                 "detail": traceback.format_exc()[-_DETAIL_LIMIT:]})
        return 0

    payload = {"status": "ok", "results": results}
    notes = request.get("notes_global")
    if notes:
        # The prelude's own counters, returned alongside the results. Only a
        # JSON-serializable mapping crosses back, and a prelude that wrote
        # something else costs the caller its diagnostics, never the batch.
        value = namespace.get(notes)
        if isinstance(value, dict):
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = None
            if value is not None:
                payload["notes"] = value
    try:
        _respond(response_path, payload)
    except (TypeError, ValueError):
        _respond(response_path, {
            "status": "bad_shape",
            "detail": "results are not JSON-serializable",
        })
    return 0


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

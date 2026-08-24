"""Executes one authored artifact against a batch of calls, bounded.

Runs OUT of the optimizer's process, launched by ``AuthoredRuntime`` through
the deny-by-default subprocess boundary. The threat model is untrusted-BUGGY
code, not adversarial code: resource limits contain runaway loops and
allocations, the import gate keeps an artifact from reaching the filesystem or
network by accident, and every failure becomes a typed status the parent
counts -- never an exception in the optimizer.

Order of operations matters and is deliberate: the request is read first
(this module's own imports must work unconstrained), THEN the resource
limits are applied, THEN the source is gated and compiled. Authored code
never runs a single instruction outside the limits.

TWO TRANSPORTS, ONE DECISION PATH. The default is the file protocol: argv
carries a request path and a response path, one process per batch. Given
``--serve <limits-json>`` the same script instead applies its limits once and
then answers line-delimited JSON requests on stdin until stdin closes -- the
persistent worker, which exists because a spawn per fit/predict is what NAS
cells spend their wall on. Everything that DECIDES a status lives in
:func:`execute` and is shared verbatim; the transports only move bytes. That
is not tidiness, it is the only way the persistent path's output-identity
claim can be checked rather than asserted.
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


def _apply_serving_limits(limits: dict) -> None:
    """The same ceilings for a worker that will serve many requests.

    Memory is unchanged: an address-space cap is a property of the process and
    every request wants the identical one. CPU is not, and copying the one-shot
    call here would be a bug rather than a simplification. ``RLIMIT_CPU``
    charges the process's WHOLE life, so a single 8-second budget handed to a
    worker that answers thousands of batches would kill it partway through a
    run and turn every later batch into a spurious timeout. The soft limit is
    therefore renewed per request by :func:`_renew_cpu_budget`, which restores
    the one-shot meaning -- this many CPU-seconds for THIS batch -- and the
    hard limit is left where the parent had it, because a hard limit lowered
    to ``cpu + 1`` could never be raised again.
    """

    try:
        import resource
    except ImportError:                      # non-POSIX: wall timeout remains
        return
    memory = int(limits.get("memory_bytes", 512 * 1024 * 1024))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    except (ValueError, OSError):
        pass
    _renew_cpu_budget(limits)


def _renew_cpu_budget(limits: dict) -> None:
    """Give the next request its own CPU budget, measured from what is spent."""

    try:
        import resource
    except ImportError:
        return
    cpu = int(limits.get("cpu_seconds", 8))
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        spent = usage.ru_utime + usage.ru_stime
        # Rounded UP, which is the difference between a renewal and a slow
        # leak: `int()` here would hand a request that starts 0.9s in a budget
        # of 0.1s, and a worker serving sub-second batches would still die of
        # SIGXCPU at the original ceiling. Ceiling guarantees every request at
        # least the whole `cpu_seconds` the one-shot path gives it.
        soft = int(spent) + (1 if spent > int(spent) else 0) + cpu
        if hard != resource.RLIM_INFINITY:
            soft = min(soft, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
    except (ValueError, OSError):
        pass


def execute(request: dict) -> dict:
    """Gate, compile and call one artifact; return the payload to send back.

    The whole of a request's meaning: which status it earns, which detail
    string explains it, which results and prelude notes travel home. Both
    transports call exactly this, which is what makes "the persistent worker
    answers identically" a testable claim about one function rather than a
    hope about two copies.

    Resource limits are NOT applied here -- the file transport applies them
    per process before calling, the serving transport once at spawn plus a
    per-request CPU renewal -- because the request has to be read before the
    ceiling that a big request would hit.
    """

    source = request["source"]
    entry_point = request["entry_point"]

    try:
        forbidden = _forbidden_import(source)
    except SyntaxError as error:
        return {"status": "unparseable", "detail": str(error)[:_DETAIL_LIMIT]}
    if forbidden is not None:
        return {"status": "forbidden_import", "detail": forbidden}

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
            return {
                "status": "crash",
                "detail": "harness prelude: "
                          + traceback.format_exc()[-_DETAIL_LIMIT:],
            }

    try:
        exec(compile(source, "<authored>", "exec"), namespace)  # noqa: S102
    except SyntaxError as error:
        return {"status": "unparseable", "detail": str(error)[:_DETAIL_LIMIT]}
    except MemoryError:
        return {"status": "memory", "detail": "during exec"}
    except Exception:
        return {"status": "crash",
                "detail": traceback.format_exc()[-_DETAIL_LIMIT:]}

    fn = namespace.get(entry_point)
    if not callable(fn):
        return {
            "status": "unparseable",
            "detail": f"source defines no callable named {entry_point!r}",
        }

    results = []
    try:
        for call in request.get("calls", []):
            results.append(fn(*call))
    except MemoryError:
        return {"status": "memory", "detail": f"after {len(results)} calls"}
    except Exception:
        return {"status": "crash",
                "detail": traceback.format_exc()[-_DETAIL_LIMIT:]}

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
    return payload


def _serialize(payload: dict) -> str:
    """The exact bytes of a response, with the unserializable case typed.

    Results come back from model-written code and need not be JSON at all
    (a ``set`` was the first case seen in the wild). Encoding failure is a
    ``bad_shape`` batch, not a dead worker, and both transports get that
    verdict from this one function so their bytes cannot drift apart.
    """

    try:
        return json.dumps(payload)
    except (TypeError, ValueError):
        return json.dumps({
            "status": "bad_shape",
            "detail": "results are not JSON-serializable",
        })


def _respond(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_serialize(payload))


def serve(limits: dict, stdin=None, stdout=None) -> int:
    """Answer line-delimited JSON requests until stdin closes.

    THE PROTOCOL, stated once: one request per line on stdin, one response per
    line on stdout, each a single JSON object encoded with the default
    ``ensure_ascii`` -- so no encoded value can contain the newline that
    frames it, and no length prefix is needed to find the boundary. The
    request is byte-for-byte the object the file transport writes to
    ``request.json``; the response is byte-for-byte what it writes to
    ``response.json``. A request's ``limits`` field is ignored here because a
    process's rlimits were fixed at spawn -- the parent that spawned this
    worker owns them, and it uses the same limits for every request it sends.

    The parent enforces the wall clock, kills on overrun and respawns; nothing
    in this loop watches a clock.
    """

    stdin = sys.stdin.buffer if stdin is None else stdin
    stdout = sys.stdout.buffer if stdout is None else stdout
    _apply_serving_limits(limits)
    while True:
        line = stdin.readline()
        if not line:                      # parent closed stdin: orderly exit
            return 0
        if not line.strip():
            continue
        _renew_cpu_budget(limits)
        try:
            request = json.loads(line)
        except (ValueError, MemoryError) as error:
            payload = {
                "status": "bad_shape",
                "detail": f"unreadable request: {error}"[:_DETAIL_LIMIT],
            }
        else:
            payload = execute(request)
        stdout.write(_serialize(payload).encode("utf-8") + b"\n")
        stdout.flush()


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--serve":
        limits = json.loads(argv[1]) if len(argv) > 1 else {}
        return serve(limits)
    request_path, response_path = argv
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)

    _apply_limits(request.get("limits") or {})
    _respond(response_path, execute(request))
    return 0


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

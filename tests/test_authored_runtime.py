"""The sandbox must kill what needs killing and count what it killed.

Every failure mode of authored code becomes a typed, bounded outcome: a loop
is a timeout, an allocation is a memory kill, an exception is a crash, an
``import os`` never runs at all. Nothing here may raise into the caller --
a crashing artifact is an ordinary countable event.
"""

from __future__ import annotations

import time

from agent_evolve.core.authored import authored_artifact
from agent_evolve.infrastructure.authored_runtime import (
    AuthoredRuntime,
    RuntimeLimits,
)


def _artifact(source: str, kind: str = "operator"):
    return authored_artifact(kind, source, name="t", authored_by="rule")


def _runtime(**overrides):
    limits = RuntimeLimits(**{"wall_time_s": 10.0, "cpu_seconds": 8, **overrides})
    return AuthoredRuntime(limits=limits)


ARGS = [[{"x": 1}, {"x": 2}, ["x"], {"x": [1, 2]}, 7]]


def test_a_working_operator_returns_one_result_per_call():
    out = _runtime().call(_artifact(
        "def vary(a, b, loci, domains, seed):\n"
        "    return {'x': a['x']}\n"
    ), ARGS + ARGS)
    assert out.ok and out.results == ({"x": 1}, {"x": 1})


def test_an_infinite_loop_is_a_timeout_within_the_wall_bound():
    started = time.monotonic()
    out = _runtime(wall_time_s=1.5).call(_artifact(
        "def vary(a, b, loci, domains, seed):\n"
        "    while True:\n"
        "        pass\n"
    ), ARGS)
    assert out.status == "timeout", out
    assert time.monotonic() - started < 10.0, "the wall bound did not bind"


def test_a_runaway_allocation_is_a_memory_kill():
    out = _runtime(memory_bytes=128 * 1024 * 1024).call(_artifact(
        "def vary(a, b, loci, domains, seed):\n"
        "    return {'x': len('a' * (512 * 1024 * 1024))}\n"
    ), ARGS)
    assert out.status == "memory", out


def test_a_raising_artifact_is_a_crash_with_the_reason():
    out = _runtime().call(_artifact(
        "def vary(a, b, loci, domains, seed):\n"
        "    raise RuntimeError('boom')\n"
    ), ARGS)
    assert out.status == "crash" and "boom" in out.detail


def test_a_forbidden_import_never_runs_at_all():
    out = _runtime().call(_artifact(
        "import os\n"
        "def vary(a, b, loci, domains, seed):\n"
        "    return {'x': 1}\n"
    ), ARGS)
    assert out.status == "forbidden_import" and out.detail == "os"


def test_an_allowed_import_still_works():
    out = _runtime().call(_artifact(
        "import math\n"
        "def vary(a, b, loci, domains, seed):\n"
        "    return {'x': math.floor(1.9)}\n"
    ), ARGS)
    assert out.ok and out.results == ({"x": 1},)


def test_dunder_import_is_treated_as_an_import():
    out = _runtime().call(_artifact(
        "def vary(a, b, loci, domains, seed):\n"
        "    os = __import__('os')\n"
        "    return {'x': 1}\n"
    ), ARGS)
    assert out.status == "forbidden_import"


def test_a_syntax_error_is_unparseable():
    out = _runtime().call(_artifact("def vary(a, b:\n    pass\n"), ARGS)
    assert out.status == "unparseable"


def test_a_missing_entry_point_is_unparseable_with_the_name():
    out = _runtime().call(_artifact(
        "def mutate(a, b, loci, domains, seed):\n"
        "    return {}\n"
    ), ARGS)
    assert out.status == "unparseable" and "vary" in out.detail


def test_unserializable_results_are_bad_shape():
    out = _runtime().call(_artifact(
        "def vary(a, b, loci, domains, seed):\n"
        "    return {'x'}\n"
    ), ARGS)
    assert out.status == "bad_shape"


def test_an_empty_batch_spawns_nothing_and_is_ok():
    out = _runtime().call(_artifact(
        "def vary(a, b, loci, domains, seed):\n"
        "    return {}\n"
    ), [])
    assert out.ok and out.results == ()


def test_surrogate_contract_carries_its_own_entry_point():
    out = _runtime().call(authored_artifact(
        "surrogate",
        "def fit_predict(train_x, train_y, test_x):\n"
        "    return [{'y': 0.0} for _ in test_x]\n",
        name="s", authored_by="rule",
    ), [[[{"a": 1}], [{"y": 1.0}], [{"a": 2}, {"a": 3}]]])
    assert out.ok and out.results == ([{"y": 0.0}, {"y": 0.0}],)

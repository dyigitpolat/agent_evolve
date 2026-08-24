"""The persistent worker must be a wall-clock change and nothing else.

One sandboxed process serving many batches is worth 15-90 minutes of a NAS
cell -- the spawn per fit, per predict and per emit is what those cells burn.
It is worth exactly nothing if the answers move. So the property under test
here is not "persistent mode works": it is that for the same request, the two
transports return the SAME typed outcome with the SAME payload -- results,
detail text and prelude notes -- across the whole failure surface the sandbox
exists to type: a healthy fit, a healthy emit, a raise, a runaway loop, a
forbidden import, an allocation past the rlimit, a worker that dies mid-batch
and a reply that is not a reply.

Every case below runs through BOTH paths in the same test and is compared
field by field. A divergence here is a defect in the persistent transport,
never a tolerance to widen: the one-shot path is the definition.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pytest

from agent_evolve.core.authored import authored_artifact
from agent_evolve.infrastructure.authored_runtime import (
    AuthoredRuntime,
    RuntimeLimits,
    RuntimeOutcome,
)

DEFAULT = RuntimeLimits(wall_time_s=10.0, cpu_seconds=8)


@dataclass(frozen=True)
class Case:
    """One request, with the limits it needs, runnable through either transport."""

    label: str
    kind: str
    source: str
    calls: Sequence[Sequence[Any]]
    limits: RuntimeLimits = DEFAULT
    prelude: Optional[str] = None
    notes_global: Optional[str] = None
    #: Cases whose worker cannot be reused for the next request (it is killed,
    #: or it hit the address-space ceiling) are kept out of the reuse counts.
    ends_the_worker: bool = False
    tags: frozenset = field(default_factory=frozenset)

    def artifact(self):
        return authored_artifact(self.kind, self.source,
                                 name=self.label, authored_by="rule")

    def run(self, runtime: AuthoredRuntime) -> RuntimeOutcome:
        return runtime.call(self.artifact(), self.calls,
                            prelude=self.prelude, notes_global=self.notes_global)


VARY_ARGS = [[{"x": 1}, {"x": 2}, ["x"], {"x": [1, 2]}, 7]]
FIT_ARGS = [[[{"a": 1}, {"a": 2}], [{"y": 1.0}, {"y": 4.0}], [{"a": 3}, {"a": 4}]]]
PROPOSE_ARGS = [[[{"a": 1}], 3, {"a": [1, 2, 3]}, 11]]

#: The deterministic battery: same request, same answer, every time, in both
#: transports. These are also what the 50-request sequence cycles through.
DETERMINISTIC = (
    Case(
        label="surrogate_fit_predict",
        kind="surrogate",
        source=(
            "def fit_predict(train_x, train_y, test_x):\n"
            "    mean = sum(row['y'] for row in train_y) / len(train_y)\n"
            "    return [{'y': mean + row['a']} for row in test_x]\n"
        ),
        calls=FIT_ARGS,
    ),
    Case(
        label="generator_emit",
        kind="generator",
        source=(
            "def propose(archive, n, domains, seed):\n"
            "    values = domains['a']\n"
            "    return [{'a': values[(seed + i) % len(values)]}\n"
            "            for i in range(n)]\n"
        ),
        calls=PROPOSE_ARGS,
    ),
    Case(
        label="generator_emit_with_prelude_notes",
        kind="generator",
        source=(
            "def propose(archive, n, domains, seed):\n"
            "    return [scaffold_pick(domains, i) for i in range(n)]\n"
        ),
        calls=PROPOSE_ARGS,
        prelude=(
            "COUNTERS = {'picked': 0}\n"
            "def scaffold_pick(domains, i):\n"
            "    COUNTERS['picked'] += 1\n"
            "    return {'a': domains['a'][i % len(domains['a'])]}\n"
        ),
        notes_global="COUNTERS",
    ),
    Case(
        label="operator_with_allowed_import",
        kind="operator",
        source=(
            "import math\n"
            "def vary(a, b, loci, domains, seed):\n"
            "    return {'x': math.floor(1.9 + a['x'])}\n"
        ),
        calls=VARY_ARGS * 3,
    ),
    Case(
        label="source_that_raises",
        kind="operator",
        source=(
            "def vary(a, b, loci, domains, seed):\n"
            "    raise RuntimeError('boom')\n"
        ),
        calls=VARY_ARGS,
    ),
    Case(
        label="forbidden_import",
        kind="operator",
        source=(
            "import os\n"
            "def vary(a, b, loci, domains, seed):\n"
            "    return {'x': 1}\n"
        ),
        calls=VARY_ARGS,
    ),
    Case(
        label="syntax_error",
        kind="operator",
        source="def vary(a, b:\n    pass\n",
        calls=VARY_ARGS,
    ),
    Case(
        label="missing_entry_point",
        kind="operator",
        source="def mutate(a, b, loci, domains, seed):\n    return {}\n",
        calls=VARY_ARGS,
    ),
    Case(
        label="unserializable_result",
        kind="operator",
        source="def vary(a, b, loci, domains, seed):\n    return {'x'}\n",
        calls=VARY_ARGS,
    ),
    Case(
        label="empty_batch",
        kind="operator",
        source="def vary(a, b, loci, domains, seed):\n    return {}\n",
        calls=[],
    ),
)

#: The cases that need their own ceilings, and that end the worker serving
#: them: a wall overrun is killed, an interpreter that exits takes the loop
#: with it, and a batch that hit RLIMIT_AS is recycled on purpose.
BOUNDED = (
    Case(
        label="runaway_loop_times_out",
        kind="operator",
        source=(
            "def vary(a, b, loci, domains, seed):\n"
            "    while True:\n"
            "        pass\n"
        ),
        calls=VARY_ARGS,
        # The wall has to bind before the CPU rlimit, or the two transports
        # would be answering different questions (a SIGXCPU kill is reported
        # through the dead-worker branch, a wall overrun through the timeout
        # branch) -- and both would still be a "timeout" to the caller.
        limits=RuntimeLimits(wall_time_s=1.0, cpu_seconds=30),
        ends_the_worker=True,
    ),
    Case(
        label="allocation_past_the_rlimit",
        kind="operator",
        source=(
            "def vary(a, b, loci, domains, seed):\n"
            "    return {'x': len('a' * (512 * 1024 * 1024))}\n"
        ),
        calls=VARY_ARGS,
        limits=RuntimeLimits(wall_time_s=10.0, cpu_seconds=8,
                             memory_bytes=128 * 1024 * 1024),
        ends_the_worker=True,
    ),
    Case(
        label="worker_exits_under_the_batch",
        kind="operator",
        # SystemExit is not an Exception, so it walks straight out of the
        # worker's call loop and takes the process with it -- the closest
        # thing to a mid-batch death that authored code can reach without an
        # import, and the case the respawn path exists for.
        source=(
            "def vary(a, b, loci, domains, seed):\n"
            "    raise SystemExit(3)\n"
        ),
        calls=VARY_ARGS,
        ends_the_worker=True,
    ),
)


def _fields(outcome: RuntimeOutcome) -> tuple:
    return (outcome.status, outcome.results, outcome.detail, dict(outcome.notes))


def _assert_identical(case: Case, one_shot: RuntimeOutcome,
                      persistent: RuntimeOutcome) -> None:
    assert _fields(persistent) == _fields(one_shot), (
        f"{case.label}: persistent {_fields(persistent)!r} != "
        f"one-shot {_fields(one_shot)!r}")


@pytest.mark.parametrize("case", DETERMINISTIC + BOUNDED,
                         ids=lambda case: case.label)
def test_both_transports_return_the_same_outcome(case: Case) -> None:
    one_shot = case.run(AuthoredRuntime(limits=case.limits))
    with AuthoredRuntime(limits=case.limits, persistent=True) as runtime:
        persistent = case.run(runtime)
    _assert_identical(case, one_shot, persistent)


def test_the_battery_types_every_status_it_claims_to() -> None:
    """The comparison above is vacuous if both paths merely agree on 'crash'."""

    expected = {
        "surrogate_fit_predict": "ok",
        "generator_emit": "ok",
        "generator_emit_with_prelude_notes": "ok",
        "operator_with_allowed_import": "ok",
        "source_that_raises": "crash",
        "forbidden_import": "forbidden_import",
        "syntax_error": "unparseable",
        "missing_entry_point": "unparseable",
        "unserializable_result": "bad_shape",
        "empty_batch": "ok",
        "runaway_loop_times_out": "timeout",
        "allocation_past_the_rlimit": "memory",
        "worker_exits_under_the_batch": "crash",
    }
    with AuthoredRuntime(persistent=True) as runtime:
        for case in DETERMINISTIC:
            outcome = case.run(runtime)
            assert outcome.status == expected[case.label], (case.label, outcome)
    for case in BOUNDED:
        with AuthoredRuntime(limits=case.limits, persistent=True) as runtime:
            outcome = case.run(runtime)
            assert outcome.status == expected[case.label], (case.label, outcome)


def test_the_prelude_notes_survive_the_stdio_protocol() -> None:
    case = next(c for c in DETERMINISTIC if c.notes_global)
    with AuthoredRuntime(persistent=True) as runtime:
        outcome = case.run(runtime)
    assert outcome.ok and outcome.notes == {"picked": 3}


def test_fifty_mixed_requests_match_the_one_shot_path_in_order() -> None:
    """Request 50 must answer like request 1: no state crosses the boundary.

    One worker, fifty batches, the deterministic battery cycled -- healthy
    fits and emits interleaved with crashes, forbidden imports and syntax
    errors, which is the shape of a real screen. The one-shot list is the
    oracle and the comparison is positional.
    """

    sequence = [DETERMINISTIC[index % len(DETERMINISTIC)] for index in range(50)]
    one_shot_runtime = AuthoredRuntime()
    expected = [case.run(one_shot_runtime) for case in sequence]
    with AuthoredRuntime(persistent=True) as runtime:
        observed = [case.run(runtime) for case in sequence]
        pid = runtime.worker_pid
        respawns = runtime.respawns
    for case, want, got in zip(sequence, expected, observed):
        _assert_identical(case, want, got)
    assert respawns == 0, "nothing in the deterministic battery kills a worker"
    assert pid is not None


def test_one_worker_serves_the_whole_run() -> None:
    case = DETERMINISTIC[0]
    with AuthoredRuntime(persistent=True) as runtime:
        assert runtime.worker_pid is None, "no worker before the first request"
        case.run(runtime)
        first = runtime.worker_pid
        for _ in range(5):
            case.run(runtime)
        assert runtime.worker_pid == first
        assert runtime.respawns == 0
    # An empty batch is answered without a worker at all, in either mode.
    with AuthoredRuntime(persistent=True) as runtime:
        assert runtime.call(case.artifact(), []).ok
        assert runtime.worker_pid is None


def test_a_dead_worker_is_replaced_and_the_next_request_is_served() -> None:
    killer = next(c for c in BOUNDED if c.label == "worker_exits_under_the_batch")
    healthy = DETERMINISTIC[0]
    expected = healthy.run(AuthoredRuntime())
    with AuthoredRuntime(persistent=True) as runtime:
        healthy.run(runtime)
        first = runtime.worker_pid
        died = killer.run(runtime)
        assert died.status == "crash" and died.detail.startswith("rc 3:")
        after = healthy.run(runtime)
        second = runtime.worker_pid
        assert runtime.respawns == 1
    assert second not in (None, first), "the replacement is a new process"
    _assert_identical(healthy, expected, after)


def test_a_worker_killed_on_the_wall_is_replaced_and_keeps_serving() -> None:
    stuck = next(c for c in BOUNDED if c.label == "runaway_loop_times_out")
    healthy = Case(label="quick", kind="operator",
                   source="def vary(a, b, loci, domains, seed):\n"
                          "    return {'x': a['x']}\n",
                   calls=VARY_ARGS, limits=stuck.limits)
    with AuthoredRuntime(limits=stuck.limits, persistent=True) as runtime:
        started = time.monotonic()
        assert stuck.run(runtime).status == "timeout"
        assert time.monotonic() - started < 10.0, "the wall bound did not bind"
        assert runtime.worker_pid is None, "a killed worker is not kept"
        assert healthy.run(runtime).results == ({"x": 1},)
        assert runtime.respawns == 1


def test_a_worker_killed_from_outside_is_replaced_before_the_next_request() -> None:
    """A worker can die while nobody is asking it anything (the OOM killer).

    The next request must not inherit that corpse: it is reaped and replaced
    before the request is written, so the caller sees a served batch rather
    than a broken pipe.
    """

    case = DETERMINISTIC[0]
    expected = case.run(AuthoredRuntime())
    with AuthoredRuntime(persistent=True) as runtime:
        case.run(runtime)
        os.kill(runtime.worker_pid, 9)
        while runtime._process.poll() is None:      # the kill is asynchronous
            time.sleep(0.01)
        observed = case.run(runtime)
        assert runtime.respawns == 1
    _assert_identical(case, expected, observed)


def test_a_payload_larger_than_the_pipe_buffer_round_trips() -> None:
    """64KB is where a pipe stops taking bytes, and a real screen is bigger.

    A request that does not fit in the kernel's pipe buffer is written while
    the worker drains it, and the reply comes back in several reads. Neither
    end may lose or reorder a byte, so the comparison is against the file
    transport, which has no such seam.
    """

    rows = 2000
    case = Case(
        label="wide_screen", kind="surrogate",
        source=("def fit_predict(train_x, train_y, test_x):\n"
                "    return [{'y': float(row['a'])} for row in test_x]\n"),
        calls=[[[{"a": i, "pad": "x" * 40} for i in range(rows)],
                [{"y": float(i)} for i in range(rows)],
                [{"a": i, "pad": "y" * 40} for i in range(rows)]]],
    )
    expected = case.run(AuthoredRuntime())
    with AuthoredRuntime(persistent=True) as runtime:
        observed = case.run(runtime)
    assert expected.ok and len(expected.results[0]) == rows
    _assert_identical(case, expected, observed)


def test_an_allocation_kill_recycles_the_worker() -> None:
    oom = next(c for c in BOUNDED if c.label == "allocation_past_the_rlimit")
    healthy = Case(label="quick", kind="operator",
                   source="def vary(a, b, loci, domains, seed):\n"
                          "    return {'x': a['x']}\n",
                   calls=VARY_ARGS, limits=oom.limits)
    with AuthoredRuntime(limits=oom.limits, persistent=True) as runtime:
        assert oom.run(runtime).status == "memory"
        assert runtime.worker_pid is None
        assert healthy.run(runtime).results == ({"x": 1},)
        assert runtime.respawns == 1


def test_the_cpu_budget_is_charged_per_request_not_per_worker() -> None:
    """RLIMIT_CPU counts the whole process, so it has to be renewed.

    Copying the one-shot rlimit call into the serving worker would make the
    transport a slow bomb: each batch would eat the shared budget until some
    unrelated later batch died of SIGXCPU. Here the per-request budget is one
    second and the batches together spend several -- with renewal every batch
    is answered, without it the run dies partway.
    """

    budget_s = 1                      # the rlimit's granularity is a second
    iterations = _iterations_for(0.2)
    burn = Case(
        label="burn", kind="operator",
        source=("def vary(a, b, loci, domains, seed):\n"
                f"    return {{'x': sum(i * i for i in range({iterations})) > 0}}\n"),
        calls=VARY_ARGS,
        limits=RuntimeLimits(wall_time_s=30.0, cpu_seconds=budget_s),
    )
    # Driven by the clock rather than by a batch count, because the whole
    # question is cumulative: the loop runs until the worker has burned twice
    # the ONE-SHOT budget, which is where an un-renewed limit kills it.
    burned, batches = 0.0, 0
    with AuthoredRuntime(limits=burn.limits, persistent=True) as runtime:
        while burned < 2.0 * budget_s and batches < 200:
            started = time.monotonic()
            outcome = burn.run(runtime)
            burned += time.monotonic() - started
            batches += 1
            assert outcome.ok, (batches, burned, outcome)
        assert batches >= 3, "the batches were too coarse to prove anything"
        assert runtime.respawns == 0


def _iterations_for(seconds: float) -> int:
    """Loop length that costs about *seconds* of CPU on this machine."""

    started = time.process_time()
    sum(i * i for i in range(200_000))
    unit = max(time.process_time() - started, 1e-6)
    return max(200_000, min(int(200_000 * seconds / unit), 200_000_000))


_STUB_WORKER = '''\
"""A worker that answers with exactly one fixed reply, on either transport."""
import sys

REPLY = {reply!r}


def main(argv):
    if argv and argv[0] == "--serve":
        while True:
            if not sys.stdin.buffer.readline():
                return 0
            sys.stdout.buffer.write(REPLY.encode("utf-8") + b"\\n")
            sys.stdout.buffer.flush()
    _request_path, response_path = argv
    with open(response_path, "w", encoding="utf-8") as handle:
        handle.write(REPLY)
    return 0


raise SystemExit(main(sys.argv[1:]))
'''


@pytest.mark.parametrize("reply,detail", [
    ("not json", "unreadable response: Expecting value"),
    ('{"status": "weird"}', "unknown status 'weird'"),
    ('{"status": "ok", "results": []}', "0 results for 1 calls"),
])
def test_a_malformed_reply_is_bad_shape_identically(tmp_path, reply, detail) -> None:
    """A worker that answers nonsense is a bounded event, not an exception.

    The stub stands in for a worker whose reply the parent cannot use --
    truncated, mistyped, or the wrong length -- which the persistent transport
    can hit for a reason the one-shot transport cannot (a half-written line on
    a pipe). Both parents must reach the same verdict from the same bytes.
    """

    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB_WORKER.format(reply=reply), encoding="utf-8")
    case = DETERMINISTIC[0]

    one_shot = AuthoredRuntime()
    one_shot._launch = (str(stub),)
    expected = case.run(one_shot)

    with AuthoredRuntime(persistent=True) as runtime:
        runtime._launch = (str(stub),)
        observed = case.run(runtime)

    assert expected.status == "bad_shape" and detail in expected.detail
    _assert_identical(case, expected, observed)


def test_close_leaves_no_child_and_the_runtime_still_works() -> None:
    case = DETERMINISTIC[0]
    runtime = AuthoredRuntime(persistent=True)
    case.run(runtime)
    pid = runtime.worker_pid
    home = runtime._home.name
    runtime.close()

    assert runtime.worker_pid is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)                  # reaped, not merely signalled
    assert not os.path.exists(home), "the worker's scratch went with it"
    runtime.close()                      # idempotent
    assert case.run(runtime).ok, "a closed runtime spawns again on demand"
    runtime.close()


def test_the_child_environment_is_the_policy_and_nothing_else(monkeypatch) -> None:
    """Deny-by-default survives the transport change.

    The boundary port has no streaming form, so the serving spawn applies the
    policy itself. That is the kind of duplication that rots quietly, so the
    property is checked directly: the worker sees PYTHONPATH and no more, with
    an ambient secret in the parent's environment.
    """

    monkeypatch.setenv("AGENTEVOLVE_AMBIENT_MARKER", "leaked")
    probe = Case(
        label="environment_probe", kind="operator",
        # `os` is forbidden inside the sandbox by design, so the environment is
        # read by the harness prelude, which is ours rather than the model's.
        source="def vary(a, b, loci, domains, seed):\n    return NAMES\n",
        calls=VARY_ARGS,
        prelude="import os\nNAMES = sorted(os.environ)\n",
    )
    one_shot = probe.run(AuthoredRuntime())
    with AuthoredRuntime(persistent=True) as runtime:
        observed = probe.run(runtime)
    assert one_shot.ok, one_shot
    [names] = one_shot.results
    assert "PYTHONPATH" in names
    assert "AGENTEVOLVE_AMBIENT_MARKER" not in names
    # Whatever else the child interpreter puts in its own environ (CPython's
    # locale coercion writes LC_CTYPE, for one) it must put there identically
    # under both transports -- the point is that nothing the PARENT holds
    # crosses, and that the two spawns are the same spawn.
    _assert_identical(probe, one_shot, observed)


@pytest.mark.skipif(not os.environ.get("AGENTEVOLVE_BENCH"),
                    reason="wall benchmark; set AGENTEVOLVE_BENCH=1 to run")
def test_bench_two_hundred_mixed_requests() -> None:
    """The number the change exists for: wall of 200 batches, both transports."""

    sequence = [DETERMINISTIC[index % len(DETERMINISTIC)] for index in range(200)]

    started = time.monotonic()
    one_shot_runtime = AuthoredRuntime()
    for case in sequence:
        case.run(one_shot_runtime)
    one_shot_s = time.monotonic() - started

    started = time.monotonic()
    with AuthoredRuntime(persistent=True) as runtime:
        for case in sequence:
            case.run(runtime)
        respawns = runtime.respawns
    persistent_s = time.monotonic() - started

    ratio = one_shot_s / max(persistent_s, 1e-9)
    print(f"\n200 mixed batches: one-shot {one_shot_s:.2f}s, "
          f"persistent {persistent_s:.2f}s, ratio {ratio:.1f}x, "
          f"respawns {respawns}")
    assert ratio > 1.0, "the persistent transport must not be slower"

"""The launch record must be complete, honest, and behaviour-preserving.

These tests defend three properties that the campaign-provenance failure they
were written for depended on:

* the recording ``os.environ`` proxy is indistinguishable from the real thing,
  including for child processes -- instrumentation must not be able to change
  a campaign's result;
* the static surface scan finds environment names that no ``environ.get``
  grep can see, because they are bound to constants or handed to a shared
  parsing helper -- that blind spot is exactly what lost eight variables;
* credentials are recorded by identity, never by value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping, MutableMapping
from pathlib import Path

import pytest

from examples.development.launch_record import (
    LAUNCH_RECORD_FILENAME,
    EnvironmentReadLog,
    RecordingEnviron,
    build_launch_record,
    capture_process_environment,
    capture_referenced_python_environments,
    install_launch_recorder,
    record_campaign_launch,
    scan_declared_environment_surface,
)


@pytest.fixture()
def proxy() -> RecordingEnviron:
    return RecordingEnviron(os.environ, EnvironmentReadLog())


def test_proxy_is_a_mutable_mapping(proxy: RecordingEnviron) -> None:
    assert isinstance(proxy, Mapping)
    assert isinstance(proxy, MutableMapping)
    assert len(proxy) == len(os.environ)
    assert set(proxy.keys()) == set(os.environ.keys())
    assert type(proxy.copy()) is dict


def test_proxy_reads_are_transparent(proxy: RecordingEnviron) -> None:
    name = "AGENT_EVOLVE_LAUNCH_RECORD_PROBE"
    proxy[name] = "value"
    try:
        assert proxy[name] == "value"
        assert proxy.get(name) == "value"
        assert name in proxy
        assert proxy.get("AGENT_EVOLVE_ABSENT_PROBE", "fallback") == "fallback"
        with pytest.raises(KeyError):
            proxy["AGENT_EVOLVE_ABSENT_PROBE"]
    finally:
        del proxy[name]
    assert name not in proxy


def test_proxy_mutation_reaches_child_processes(proxy: RecordingEnviron) -> None:
    """``putenv``/``unsetenv`` must still fire, or subprocesses would change."""

    name = "AGENT_EVOLVE_LAUNCH_RECORD_CHILD"
    proxy[name] = "inherited"
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import os;print(os.environ.get({name!r}, 'MISSING'))"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "inherited"
    finally:
        del proxy[name]
    result = subprocess.run(
        [sys.executable, "-c", f"import os;print(os.environ.get({name!r}, 'MISSING'))"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "MISSING"


def test_proxy_records_what_was_read(proxy: RecordingEnviron) -> None:
    proxy.get("AGENT_EVOLVE_READ_ONE")
    proxy.get("AGENT_EVOLVE_READ_ONE")
    "AGENT_EVOLVE_READ_TWO" in proxy
    record = proxy._log.to_record()
    assert record["names"]["AGENT_EVOLVE_READ_ONE"]["reads"] == 2
    assert record["names"]["AGENT_EVOLVE_READ_ONE"]["present"] is False
    assert record["names"]["AGENT_EVOLVE_READ_TWO"]["operations"] == ["contains"]
    assert record["first_read_order"][:2] == [
        "AGENT_EVOLVE_READ_ONE",
        "AGENT_EVOLVE_READ_TWO",
    ]


def test_live_mode_is_not_instrumented() -> None:
    """The phase that produces results must run the unmodified process."""

    assert install_launch_recorder(argv=["live", "--run-id", "x"]) is None
    assert not isinstance(os.environ, RecordingEnviron)


def test_installation_in_a_child_process_sees_import_time_reads(
    tmp_path: Path,
) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, os, sys
            sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
            from examples.development.launch_record import install_launch_recorder
            recorder = install_launch_recorder(argv=["prepare"])
            os.environ.get("AGENT_EVOLVE_IMPORT_TIME_PROBE")
            print(json.dumps(recorder.environment_reads.to_record()["names"]))
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    )
    names = json.loads(result.stdout)
    assert "AGENT_EVOLVE_IMPORT_TIME_PROBE" in names


def test_credentials_are_recorded_by_identity_not_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-value")
    monkeypatch.setenv("AGENT_EVOLVE_ACQUISITION_MODE", "regret_bounded_information")
    record = capture_process_environment()
    variables = record["variables"]
    assert variables["OPENROUTER_API_KEY"]["redacted"] is True
    assert "value" not in variables["OPENROUTER_API_KEY"]
    assert variables["OPENROUTER_API_KEY"]["value_length"] == len("sk-secret-value")
    assert len(variables["OPENROUTER_API_KEY"]["value_sha256"]) == 64
    assert "sk-secret-value" not in json.dumps(record)
    # Configuration is never redacted: withholding it is what broke relaunch.
    entry = variables["AGENT_EVOLVE_ACQUISITION_MODE"]
    assert entry["redacted"] is False
    assert entry["value"] == "regret_bounded_information"


def _write(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_scan_finds_names_a_grep_for_environ_get_would_miss(tmp_path: Path) -> None:
    """The three spellings that hid the mapping campaign's configuration."""

    _write(
        tmp_path / "helpers.py",
        """
        import os

        RETENTION_ENV = "AGENT_EVOLVE_REGRET_MINIMUM_ACQUISITION_RETENTION_RATIO"

        def bounded_integer(name, default):
            return int(os.environ.get(name, str(default)))

        def retention(environ):
            return environ.get(RETENTION_ENV, "1.0")
        """,
    )
    _write(
        tmp_path / "runner.py",
        """
        import os
        from helpers import bounded_integer, retention

        MODE = os.environ.get("AGENT_EVOLVE_ACQUISITION_MODE", "full_support")
        POOL = bounded_integer("AGENT_EVOLVE_PROTECTED_ACQUISITION_POOL_SIZE", 8192)
        RATIO = retention(os.environ)
        """,
    )
    surface = scan_declared_environment_surface(
        sorted(tmp_path.glob("*.py")), relative_to=tmp_path
    )
    variables = surface["variables"]

    # 1. the obvious spelling
    assert "AGENT_EVOLVE_ACQUISITION_MODE" in variables
    assert variables["AGENT_EVOLVE_ACQUISITION_MODE"]["code_defaults"] == [
        "full_support"
    ]
    # 2. a literal handed to a shared parsing helper: no environ.get in sight
    assert "AGENT_EVOLVE_PROTECTED_ACQUISITION_POOL_SIZE" in variables
    assert (
        "namespace_literal"
        in variables["AGENT_EVOLVE_PROTECTED_ACQUISITION_POOL_SIZE"]["evidence"]
    )
    # 3. a name that only ever appears as a module-level constant
    assert (
        "AGENT_EVOLVE_REGRET_MINIMUM_ACQUISITION_RETENTION_RATIO" in variables
    )
    # and the helper's dynamic read is reported rather than silently dropped
    assert any(
        site.startswith("helpers.py")
        for site in surface["unresolved_dynamic_read_sites"]
    )


def test_assignment_is_not_reported_as_a_read(tmp_path: Path) -> None:
    _write(
        tmp_path / "writer.py",
        """
        import os
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
        """,
    )
    surface = scan_declared_environment_surface(
        [tmp_path / "writer.py"], relative_to=tmp_path
    )
    assert surface["variables"]["LD_LIBRARY_PATH"]["evidence"] == ["assignment"]


def test_launch_record_is_complete_and_self_describing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_EVOLVE_ACQUISITION_MODE", "regret_bounded_information")
    monkeypatch.setenv("PYTHONPATH", "/some/tree/src")
    source = _write(
        tmp_path / "runner.py",
        """
        import os
        MODE = os.environ.get("AGENT_EVOLVE_ACQUISITION_MODE", "full_support")
        """,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = build_launch_record(
        mode="prepare",
        run_id="probe",
        run_dir=run_dir,
        workspace_root=tmp_path,
        agent_evolve_root=tmp_path,
        source_paths=[source],
        source_closure={"aggregate_sha256": "0" * 64, "file_count": 1},
    )

    # The whole process environment, not a remembered subset.
    assert record["process_environment"]["variable_count"] == len(os.environ)
    assert (
        record["process_environment"]["variables"]["PYTHONPATH"]["value"]
        == "/some/tree/src"
    )
    # Enough to retype the command.
    for key in ("argv", "cwd", "reconstructed_shell_command", "workspace_root"):
        assert record["invocation"][key] is not None
    # Enough to know which code the interpreter would actually import.
    assert record["interpreter"]["sys_path"]
    assert record["interpreter"]["executable"] == sys.executable
    # The resolved configuration, joined across the layers.
    resolved = record["resolved_environment_inputs"]["AGENT_EVOLVE_ACQUISITION_MODE"]
    assert resolved["resolved_value"] == "regret_bounded_information"
    assert resolved["code_defaults"] == ["full_support"]
    assert resolved["resolved_from"] == "process_environment"


def test_python_environments_named_by_variables_are_identified(
    tmp_path: Path,
) -> None:
    """The numbers a campaign gets depend on packages it never imports."""

    environment = tmp_path / "scorer_env"
    site = environment / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("version = 3.12.3\n", encoding="utf-8")
    (environment / "bin").mkdir()
    (environment / "bin" / "python").symlink_to("/usr/bin/python3.12")
    for stem in ("torch-2.13.0", "botorch-0.18.1"):
        (site / f"{stem}.dist-info").mkdir()

    captured = capture_referenced_python_environments(
        {
            "AGENT_EVOLVE_BOTORCH_PYTHON": {
                "redacted": False,
                "value": str(environment / "bin" / "python"),
            },
            "OPENROUTER_API_KEY": {"redacted": True},
            "NOT_A_PATH": {"redacted": False, "value": "hierarchical_r2"},
        }
    )
    assert captured["count"] == 1
    identity = captured["by_environment_variable"]["AGENT_EVOLVE_BOTORCH_PYTHON"]
    assert identity["installed_distributions"] == {
        "torch": "2.13.0",
        "botorch": "0.18.1",
    }
    assert len(identity["installed_distributions_sha256"]) == 64


def test_recording_never_fails_a_campaign(tmp_path: Path) -> None:
    """A launch record is provenance, not a gate: it must never raise."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    written = record_campaign_launch(
        mode="prepare",
        run_id="probe",
        run_dir=run_dir,
        workspace_root=tmp_path,
        agent_evolve_root=tmp_path,
        source_paths=[Path(tmp_path / "does-not-exist.py")],
        dotenv_paths=(tmp_path / ".env",),
    )
    assert written == run_dir / LAUNCH_RECORD_FILENAME
    assert json.loads(written.read_text(encoding="utf-8"))["mode"] == "prepare"


# --- the reversible startup window ------------------------------------------
#
# `live` must stay byte-identical to the uninstrumented process while it
# produces results, but the evidence that a credential was never read is worth
# having for a live run too. Credentials are read at startup and the timed work
# happens later, so the window can be opened and closed again. These run in
# child processes: installing a recorder mutates process-global state, and the
# invariant under test is about a whole process, not a function call.

_STARTUP_WINDOW_PROBE = """
import json, os, sys
sys.path.insert(0, {root!r})
from examples.development.launch_record import (
    RecordingEnviron,
    active_recorder,
    instrument_startup_window,
    uninstall_launch_recorder,
)

recorder = instrument_startup_window(argv={argv!r})
if recorder is None:
    print(json.dumps({{"installed": False}}))
    raise SystemExit(0)

proxied_during_startup = isinstance(os.environ, RecordingEnviron)
os.environ.get("PROBE_STARTUP_API_KEY")

removed = uninstall_launch_recorder()
proxied_after = isinstance(os.environ, RecordingEnviron)
os.environ.get("PROBE_MEASURED_PHASE_ONLY")

reads = recorder.environment_reads.to_record()["names"]
print(json.dumps({{
    "installed": True,
    "proxied_during_startup": proxied_during_startup,
    "removed": removed,
    "proxied_after": proxied_after,
    "audit_hook_installed": recorder.audit_hook_installed,
    "reversible": recorder.reversible,
    "phases": list(recorder.instrumented_phases),
    "saw_startup_read": "PROBE_STARTUP_API_KEY" in reads,
    "saw_measured_read": "PROBE_MEASURED_PHASE_ONLY" in reads,
}}))
"""


def _run_probe(argv: list[str]) -> dict:
    root = str(Path(__file__).resolve().parents[1])
    source = _STARTUP_WINDOW_PROBE.format(root=root, argv=argv)
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_startup_window_closes_before_the_measured_phase() -> None:
    out = _run_probe(["live", "--run-id", "x"])

    assert out["installed"] is True, "a live campaign must still get the window"
    assert out["proxied_during_startup"] is True
    assert out["removed"] is True
    assert out["proxied_after"] is False, "the measured phase ran under a proxy"
    assert out["phases"] == ["startup"]


def test_the_window_records_startup_reads_and_nothing_after_it() -> None:
    """The evidence is captured; the measured phase is unobserved."""
    out = _run_probe(["live", "--run-id", "x"])

    assert out["saw_startup_read"] is True, "credential-read evidence was lost"
    assert out["saw_measured_read"] is False, "the recorder outlived its window"


def test_the_window_installs_nothing_irreversible() -> None:
    """sys.addaudithook cannot be undone, so the window must never add one."""
    out = _run_probe(["live", "--run-id", "x"])

    assert out["audit_hook_installed"] is False
    assert out["reversible"] is True


def test_the_window_ignores_an_invocation_that_names_no_campaign_mode() -> None:
    """Importing a runner under a test harness must instrument nothing."""
    assert _run_probe(["-q", "tests/"]) == {"installed": False}


def test_an_irreversible_recorder_refuses_to_report_a_clean_uninstall(
    monkeypatch,
) -> None:
    """A disarmed hook must never be mistaken for an absent one."""
    from examples.development import launch_record as module

    recorder = module.LaunchRecorder()
    recorder.audit_hook_installed = True
    monkeypatch.setattr(module, "_RECORDER", recorder)
    monkeypatch.setattr(module.os, "environ", dict(os.environ))

    assert module.uninstall_launch_recorder() is False
    assert recorder.reversible is False
    assert recorder.instrumented_phases == ("startup", "measured")


def test_the_launch_record_states_which_phases_were_instrumented(
    tmp_path: Path, monkeypatch
) -> None:
    from examples.development import launch_record as module

    recorder = module.LaunchRecorder()
    recorder.environment_proxy_installed = True
    monkeypatch.setattr(module, "_RECORDER", recorder)

    still_open = build_launch_record(
        mode="live", run_id="r", run_dir=tmp_path, workspace_root=tmp_path
    )["instrumentation"]
    assert still_open["instrumented_phases"] == ["startup", "measured"]
    assert still_open["environment_proxy_active"] is True

    recorder.uninstalled_at_utc = "2026-07-29T00:00:00+00:00"
    closed = build_launch_record(
        mode="live", run_id="r", run_dir=tmp_path, workspace_root=tmp_path
    )["instrumentation"]
    assert closed["instrumented_phases"] == ["startup"]
    assert closed["environment_proxy_active"] is False
    assert closed["reversible"] is True
    assert closed["uninstalled_at_utc"] == "2026-07-29T00:00:00+00:00"

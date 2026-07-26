"""Provider-free tests for the reusable explicit child-process boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from agent_evolve.agentic import (
    ChildProcessPolicy,
    ExplicitEnvironmentSubprocessBoundary,
)


def _policy() -> ChildProcessPolicy:
    return ChildProcessPolicy(
        policy_id="test_explicit_child",
        policy_version=1,
        inherited_environment_allowlist=("HOME",),
        fixed_environment=(("LANG", "C.UTF-8"), ("PATH", "/usr/bin:/bin")),
    )


def test_boundary_preserves_argv_pins_cwd_and_denies_ambient_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(
        "agent_evolve.infrastructure.subprocess_boundary.subprocess.run", fake_run
    )
    source_environment = {
        "HOME": "/nonsecret/home",
        "OPENROUTER_API_KEY": "must-not-cross",
        "SERVICE_KEY": "must-not-cross",
        "ACCESS_TOKEN": "must-not-cross",
        "CLIENT_SECRET": "must-not-cross",
        "DB_PASSWORD": "must-not-cross",
        "UNRELATED": "must-not-cross",
    }
    boundary = ExplicitEnvironmentSubprocessBoundary(
        policy=_policy(),
        working_directory=tmp_path.resolve(),
        source_environment=source_environment,
    )
    argv = ("/environment/bin/python", "runner.py", "--help")

    result = boundary.run(argv, timeout_s=3.5)

    assert result.returncode == 0
    assert captured["argv"] == argv
    assert captured["cwd"] == tmp_path.resolve()
    assert captured["timeout"] == 3.5
    assert captured["shell"] is False
    assert captured["env"] == {
        "HOME": "/nonsecret/home",
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    assert not {
        "OPENROUTER_API_KEY",
        "SERVICE_KEY",
        "ACCESS_TOKEN",
        "CLIENT_SECRET",
        "DB_PASSWORD",
    }.intersection(captured["env"])


def test_effective_inherited_values_change_identity_without_value_disclosure(
    tmp_path: Path,
) -> None:
    first = ExplicitEnvironmentSubprocessBoundary(
        policy=_policy(),
        working_directory=tmp_path.resolve(),
        source_environment={"HOME": "/private/first-home"},
    )
    second = ExplicitEnvironmentSubprocessBoundary(
        policy=_policy(),
        working_directory=tmp_path.resolve(),
        source_environment={"HOME": "/private/second-home"},
    )

    assert first.identity_sha256 != second.identity_sha256
    first_record = json.dumps(first.stable_record(), sort_keys=True)
    second_record = json.dumps(second.stable_record(), sort_keys=True)
    assert "/private/first-home" not in first_record
    assert "/private/second-home" not in second_record
    assert "effective_environment_sha256" in first.stable_record()


@pytest.mark.parametrize(
    "name",
    (
        "OPENROUTER_API_KEY",
        "SERVICE_KEY",
        "ACCESS_TOKEN",
        "CLIENT_SECRET",
        "DB_PASSWORD",
    ),
)
def test_policy_rejects_sensitive_environment_names(name: str) -> None:
    with pytest.raises(ValueError, match="sensitive environment"):
        ChildProcessPolicy(
            policy_id="unsafe",
            policy_version=1,
            inherited_environment_allowlist=(name,),
        )


def test_invocation_observation_preserves_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_bytes(b"python")
    target.chmod(0o700)
    invoked = tmp_path / "environment-python"
    invoked.symlink_to(target)
    boundary = ExplicitEnvironmentSubprocessBoundary(
        policy=_policy(),
        working_directory=tmp_path.resolve(),
        source_environment={"HOME": "/home/test"},
    )

    observation = boundary.invocation_observation(str(invoked))

    assert observation["invoked_path"] == str(invoked)
    assert observation["resolved_target"] == str(target)
    assert observation["is_symlink"] is True
    assert observation["process_boundary_identity_sha256"] == (
        boundary.identity_sha256
    )

"""The credential loader must never reach outside what the caller names.

Regression cover for a defect where ``AgentEvolveSettings.from_env()`` called
``dotenv.load_dotenv()`` with no argument. python-dotenv then walks *upward* --
from the directory of the calling module, or from the cwd when ``__main__`` has
no ``__file__`` -- until it finds any ``.env``. In a monorepo that meant the
library adopted seven unrelated provider keys from the repository root, and a
run launched as ``env -u OPENAI_API_KEY ...`` to prove it made no provider call
had the key handed straight back to it.

The tests below plant poisoned ``.env`` files in both places the old code would
have reached and assert nothing is picked up. They use fake keys only and never
touch the network.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_evolve import settings as settings_module
from agent_evolve.settings import AgentEvolveSettings

_SETTINGS_SOURCE = Path(settings_module.__file__)

# Fake values. Nothing here is a real credential and nothing here is ever sent.
_ANCESTOR_ENV = (
    "AE_DOTENV_SENTINEL=leaked-from-ancestor\n"
    "OPENAI_API_KEY=sk-fake-ancestor-must-not-be-read\n"
    "ANTHROPIC_API_KEY=sk-ant-fake-ancestor-must-not-be-read\n"
    "AGENTEVOLVE_MODEL=openai:ancestor-injected\n"
)
_CWD_ENV = (
    "AE_DOTENV_SENTINEL=leaked-from-cwd\n"
    "OPENAI_API_KEY=sk-fake-cwd-must-not-be-read\n"
    "AGENTEVOLVE_MODEL=openai:cwd-injected\n"
)

_LEAKABLE = ("AE_DOTENV_SENTINEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AGENTEVOLVE_MODEL")

_PROBE = """
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings

result = settings.AgentEvolveSettings.from_env(
    **({"dotenv_path": sys.argv[1]} if len(sys.argv) > 1 else {})
)
print(json.dumps({
    "env": {k: os.environ.get(k) for k in %r},
    "model": result.model,
    "dotenv_source": getattr(result, "dotenv_source", None),
}))
""" % (_LEAKABLE,)


def _plant(tmp_path: Path) -> dict:
    """Build a tree with a poisoned ``.env`` above the module and inside the cwd.

    Layout::

        root/.env                  <- ancestor poison (old bug reached this)
        root/work/pkg/settings.py  <- the real module, copied
        root/work/pkg/probe.py
        root/work/deep/.env        <- cwd poison (old bug reached this under -c)

    Both poisons sit under ``tmp_path`` so the walk can never escape into a real
    repository.
    """
    root = tmp_path / "root"
    pkg = root / "work" / "pkg"
    deep = root / "work" / "deep"
    home = tmp_path / "home"
    for d in (pkg, deep, home):
        d.mkdir(parents=True)

    (root / ".env").write_text(_ANCESTOR_ENV)
    (deep / ".env").write_text(_CWD_ENV)
    shutil.copy2(_SETTINGS_SOURCE, pkg / "settings.py")
    (pkg / "probe.py").write_text(_PROBE)

    return {"root": root, "pkg": pkg, "deep": deep, "home": home}


def _run(tree: dict, *, argv: list, cwd: Path) -> dict:
    """Run the probe in a subprocess whose environment carries no provider keys."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tree["home"]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, *argv],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_loader_never_walks_up_to_an_ancestor_dotenv(tmp_path):
    """The exact defect: a ``.env`` above the module's own directory."""
    tree = _plant(tmp_path)

    out = _run(tree, argv=[str(tree["pkg"] / "probe.py")], cwd=tree["deep"])

    leaked = {k: v for k, v in out["env"].items() if v is not None}
    assert leaked == {}, f"loader reached outside its caller's scope: {leaked}"
    assert out["dotenv_source"] is None
    assert out["model"] == "openai:gpt-4o", "model came from a .env nobody named"


def test_loader_never_falls_back_to_the_working_directory(tmp_path):
    """Under ``python -c`` python-dotenv searches from the cwd instead."""
    tree = _plant(tmp_path)

    inline = (
        "import json, os, sys; "
        f"sys.path.insert(0, {str(tree['pkg'])!r}); "
        "import settings; "
        "r = settings.AgentEvolveSettings.from_env(); "
        "print(json.dumps({"
        f"'env': {{k: os.environ.get(k) for k in {_LEAKABLE!r}}}, "
        "'model': r.model, 'dotenv_source': getattr(r, 'dotenv_source', None)}))"
    )
    out = _run(tree, argv=["-c", inline], cwd=tree["deep"])

    leaked = {k: v for k, v in out["env"].items() if v is not None}
    assert leaked == {}, f"loader adopted the working directory's .env: {leaked}"
    assert out["dotenv_source"] is None


def test_an_explicitly_named_dotenv_is_still_loaded(tmp_path):
    """Positive control: without this, the tests above could pass vacuously."""
    tree = _plant(tmp_path)
    named = tree["root"] / ".env"

    out = _run(tree, argv=[str(tree["pkg"] / "probe.py"), str(named)], cwd=tree["deep"])

    assert out["env"]["AE_DOTENV_SENTINEL"] == "leaked-from-ancestor"
    assert out["model"] == "openai:ancestor-injected"
    assert out["dotenv_source"] == str(named.resolve())


def test_from_env_never_calls_dotenv_without_an_explicit_path(monkeypatch, tmp_path):
    """Guard the mechanism directly, not just its observable effect."""
    dotenv = pytest.importorskip("dotenv")
    calls = []

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        dotenv,
        "find_dotenv",
        lambda *a, **k: pytest.fail("from_env() searched the filesystem for a .env"),
    )
    monkeypatch.delenv(settings_module.DOTENV_PATH_VAR, raising=False)

    AgentEvolveSettings.from_env()
    assert calls == [], "from_env() read a .env nobody named"

    named = tmp_path / ".env"
    named.write_text("AE_DOTENV_SENTINEL=explicit\n")
    AgentEvolveSettings.from_env(dotenv_path=str(named))
    assert len(calls) == 1
    assert calls[0][0][0] == named
    assert calls[0][1] == {"override": False}, "the process environment must outrank the file"


def test_dotenv_path_env_var_is_an_explicit_path(monkeypatch, tmp_path):
    named = tmp_path / "custom.env"
    named.write_text("AE_DOTENV_SENTINEL=from-env-var\n")
    monkeypatch.setenv(settings_module.DOTENV_PATH_VAR, str(named))
    monkeypatch.delenv("AE_DOTENV_SENTINEL", raising=False)

    resolved = AgentEvolveSettings.from_env()

    assert os.environ["AE_DOTENV_SENTINEL"] == "from-env-var"
    assert resolved.dotenv_source == str(named.resolve())


def test_a_named_dotenv_that_is_missing_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv(settings_module.DOTENV_PATH_VAR, raising=False)
    with pytest.raises(FileNotFoundError):
        AgentEvolveSettings.from_env(dotenv_path=str(tmp_path / "absent.env"))


def test_the_process_environment_outranks_a_named_dotenv(monkeypatch, tmp_path):
    """``env -u KEY`` / an exported value must not be overwritten by a stale file."""
    named = tmp_path / ".env"
    named.write_text("AGENTEVOLVE_MODEL=openai:from-file\n")
    monkeypatch.setenv("AGENTEVOLVE_MODEL", "openai:from-process")
    monkeypatch.delenv(settings_module.DOTENV_PATH_VAR, raising=False)

    assert AgentEvolveSettings.from_env(dotenv_path=str(named)).model == "openai:from-process"


def test_settings_source_contains_no_searching_dotenv_call():
    """A static ratchet: the searching form must never come back.

    Parsed rather than grepped so prose about the defect does not trip it.
    """
    tree = ast.parse(_SETTINGS_SOURCE.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "find_dotenv":
            pytest.fail("find_dotenv() searches upward by construction")
        if name == "load_dotenv" and not (node.args or node.keywords):
            pytest.fail("load_dotenv() with no path searches upward for a .env")

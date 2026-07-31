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

from agent_evolve.settings import _DEFAULT_MODEL

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
    assert out["model"] == _DEFAULT_MODEL, "model came from a .env nobody named"


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


def test_from_env_never_reads_a_file_nobody_named(monkeypatch, tmp_path):
    """Guard the mechanism directly, not just its observable effect."""
    dotenv = pytest.importorskip("dotenv")
    reads = []

    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: reads.append(a) or {})
    monkeypatch.setattr(
        dotenv,
        "find_dotenv",
        lambda *a, **k: pytest.fail("from_env() searched the filesystem for a .env"),
    )
    monkeypatch.setattr(
        dotenv,
        "load_dotenv",
        lambda *a, **k: pytest.fail("load_dotenv cannot honour a scrub list"),
    )
    monkeypatch.delenv(settings_module.DOTENV_PATH_VAR, raising=False)

    AgentEvolveSettings.from_env()
    assert reads == [], "from_env() read a .env nobody named"

    named = tmp_path / ".env"
    named.write_text("AE_DOTENV_SENTINEL=explicit\n")
    AgentEvolveSettings.from_env(dotenv_path=str(named))
    assert len(reads) == 1 and reads[0][0] == named


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


def test_settings_source_never_calls_load_dotenv_or_find_dotenv():
    """A static ratchet: neither unsafe primitive may come back.

    ``find_dotenv`` searches upward by construction. ``load_dotenv`` sets
    variables wholesale, so it cannot honour a scrub list no matter what
    arguments it is given. Parsed rather than grepped so prose about the defects
    does not trip it.
    """
    tree = ast.parse(_SETTINGS_SOURCE.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in {"find_dotenv", "load_dotenv"}:
            pytest.fail(f"{name}() cannot be used here; see this module's docstring")


# --- the re-injection defect -------------------------------------------------
#
# `load_dotenv(path, override=False)` restores a variable that `env -u` removed:
# override=False defers only to variables that are PRESENT, and a scrubbed one is
# absent. These tests pin the replacement behaviour.


def test_a_scrubbed_name_is_not_reintroduced_from_a_named_file(monkeypatch, tmp_path):
    """The defect itself, at the unit level."""
    named = tmp_path / ".env"
    named.write_text("FAKE_PROVIDER_API_KEY=sk-fake-must-not-come-back\nAE_OK=fine\n")
    monkeypatch.delenv("FAKE_PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("AE_OK", raising=False)
    monkeypatch.setenv(settings_module.SCRUBBED_VAR, "FAKE_PROVIDER_API_KEY")

    load = settings_module.load_credentials(named)

    assert "FAKE_PROVIDER_API_KEY" not in os.environ, "the scrubbed key was handed back"
    assert load.refused_scrubbed == ("FAKE_PROVIDER_API_KEY",)
    assert os.environ["AE_OK"] == "fine", "unrelated configuration must still load"
    assert load.introduced == ("AE_OK",)


def test_a_scrubbed_name_present_in_the_environment_is_removed(monkeypatch, tmp_path):
    """Declaring a scrub also revokes a value already exported into the shell."""
    named = tmp_path / ".env"
    named.write_text("AE_OK=fine\n")
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-fake-exported-earlier")
    monkeypatch.setenv(settings_module.SCRUBBED_VAR, "FAKE_PROVIDER_API_KEY")

    load = settings_module.load_credentials(named)

    assert "FAKE_PROVIDER_API_KEY" not in os.environ
    assert load.removed_from_environment == ("FAKE_PROVIDER_API_KEY",)


def test_a_scrub_outranks_override_true(monkeypatch, tmp_path):
    named = tmp_path / ".env"
    named.write_text("FAKE_PROVIDER_API_KEY=sk-fake-must-not-come-back\n")
    monkeypatch.delenv("FAKE_PROVIDER_API_KEY", raising=False)
    monkeypatch.setenv(settings_module.SCRUBBED_VAR, "FAKE_PROVIDER_API_KEY")

    settings_module.load_credentials(named, override=True)

    assert "FAKE_PROVIDER_API_KEY" not in os.environ


def test_a_run_can_declare_it_needs_no_credentials(monkeypatch, tmp_path):
    """A provider-free runner states that, and the file cannot override it."""
    named = tmp_path / ".env"
    named.write_text("FAKE_PROVIDER_API_KEY=sk-fake\nOTHER_TOKEN=t\nAE_MODEL_NAME=m\n")
    for name in ("FAKE_PROVIDER_API_KEY", "OTHER_TOKEN", "AE_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(settings_module.SCRUBBED_VAR, raising=False)

    load = settings_module.load_credentials(named, allow_credentials=())

    assert "FAKE_PROVIDER_API_KEY" not in os.environ
    assert "OTHER_TOKEN" not in os.environ
    assert load.refused_undeclared == ("FAKE_PROVIDER_API_KEY", "OTHER_TOKEN")
    assert os.environ["AE_MODEL_NAME"] == "m", "non-credential config is not gated"


def test_a_run_may_declare_exactly_the_credential_it_needs(monkeypatch, tmp_path):
    named = tmp_path / ".env"
    named.write_text("WANTED_API_KEY=sk-wanted\nUNWANTED_API_KEY=sk-unwanted\n")
    monkeypatch.delenv("WANTED_API_KEY", raising=False)
    monkeypatch.delenv("UNWANTED_API_KEY", raising=False)
    monkeypatch.delenv(settings_module.SCRUBBED_VAR, raising=False)

    load = settings_module.load_credentials(named, allow_credentials=("WANTED_API_KEY",))

    assert os.environ["WANTED_API_KEY"] == "sk-wanted"
    assert "UNWANTED_API_KEY" not in os.environ
    assert load.refused_undeclared == ("UNWANTED_API_KEY",)


def test_scrub_survives_a_real_subprocess_launch(tmp_path):
    """End to end, in the shape the operator actually types."""
    tree = _plant(tmp_path)
    named = tree["root"] / ".env"
    probe = (
        "import json, os, sys; "
        f"sys.path.insert(0, {str(tree['pkg'])!r}); "
        "import settings; "
        f"load = settings.load_credentials({str(named)!r}); "
        "print(json.dumps({"
        f"'env': {{k: os.environ.get(k) for k in {_LEAKABLE!r}}}, "
        "'refused': list(load.refused_scrubbed)}))"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tree["home"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "AGENTEVOLVE_SCRUBBED": "OPENAI_API_KEY,ANTHROPIC_API_KEY",
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tree["deep"]),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    assert out["env"]["OPENAI_API_KEY"] is None, "scrubbed key was re-injected"
    assert out["env"]["ANTHROPIC_API_KEY"] is None, "scrubbed key was re-injected"
    assert sorted(out["refused"]) == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    # The named file was genuinely read -- otherwise this proves nothing.
    assert out["env"]["AE_DOTENV_SENTINEL"] == "leaked-from-ancestor"


def test_no_runner_calls_python_dotenv_directly():
    """The whole repository must go through the safe loader.

    The audit that found the re-injection defect also found that it survived in
    the archive by luck: the one runner ever launched with a scrubbed key
    happened not to call ``load_dotenv``, while the module it imported did --
    inside a function rather than at import time. Hoisting that one call would
    have silently defeated every scrubbed launch. This ratchet removes the luck.
    """
    repo_root = _SETTINGS_SOURCE.parents[2]
    offenders = []
    for path in sorted((repo_root / "src").rglob("*.py")) + sorted(
        (repo_root / "examples").rglob("*.py")
    ):
        if path == _SETTINGS_SOURCE:
            continue  # the one module allowed to touch python-dotenv
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in {"load_dotenv", "find_dotenv", "dotenv_values"}:
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno} {name}()")

    assert offenders == [], (
        "these call python-dotenv directly and so cannot honour AGENTEVOLVE_SCRUBBED; "
        "use agent_evolve.settings.load_credentials instead:\n  "
        + "\n  ".join(offenders)
    )


def test_an_optional_dotenv_may_be_absent_but_still_enforces_the_scrub(monkeypatch, tmp_path):
    """Layering a workspace .env over a repository one must tolerate a gap."""
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-fake-exported-earlier")
    monkeypatch.setenv(settings_module.SCRUBBED_VAR, "FAKE_PROVIDER_API_KEY")

    load = settings_module.load_credentials(tmp_path / "absent.env", optional=True)

    assert load.dotenv_path is None and load.introduced == ()
    assert "FAKE_PROVIDER_API_KEY" not in os.environ, "a missing file must not skip the scrub"


def test_load_credentials_still_refuses_to_search(monkeypatch, tmp_path):
    """The first defect must not reappear through the new entry point."""
    dotenv = pytest.importorskip("dotenv")
    monkeypatch.setattr(
        dotenv,
        "find_dotenv",
        lambda *a, **k: pytest.fail("load_credentials() searched for a .env"),
    )
    with pytest.raises(FileNotFoundError):
        settings_module.load_credentials(tmp_path / "absent.env")

"""Provider-free tests for the generic content-addressed runtime manifest."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from agent_evolve.application.live_runtime_manifest import (
    LiveRuntimeManifestError,
    RuntimeManifestSection,
    build_live_runtime_manifest,
    capture_git_worktree_section,
    capture_runtime_source_closure,
    live_runtime_manifest_from_record,
    verify_runtime_source_closure,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True)


def test_role_indexed_source_closure_and_manifest_are_content_addressed(
    tmp_path: Path,
) -> None:
    core = tmp_path / "core.py"
    launcher = tmp_path / "launch.py"
    lock = tmp_path / "lock.txt"
    core.write_text("VALUE = 1\n", encoding="utf-8")
    launcher.write_text("from core import VALUE\n", encoding="utf-8")
    lock.write_text("dependency==1\n", encoding="utf-8")
    source = capture_runtime_source_closure(
        {
            "dependency_lock": {"lock.txt": lock},
            "generic_core": {"src/core.py": core},
            "launcher": {"launch.py": launcher, "src/core.py": core},
        }
    )
    sections = (
        RuntimeManifestSection.seal("experiment", {"budget": 6}),
        RuntimeManifestSection.seal(
            "provider_route",
            {"model": "provider/model", "fallbacks": False},
        ),
    )
    manifest = build_live_runtime_manifest(
        manifest_id="test_live_runtime",
        manifest_version=1,
        built_at_utc="2026-07-15T00:00:00Z",
        source_closure=source,
        sections=sections,
        required_section_ids=("experiment", "provider_route"),
    )

    record = manifest.to_record()
    assert record["claim_boundary"] == {
        "credentials_read": False,
        "provider_called": False,
        "physical_evaluator_called": False,
        "current_run_outcomes_observed": False,
        "meaning": "prospective provider-free runtime commitment only",
    }
    assert len(record["manifest_sha256"]) == 64
    assert record["source_closure"]["roles"]["launcher"] == [
        "launch.py",
        "src/core.py",
    ]
    assert len(source.files) == 3
    assert live_runtime_manifest_from_record(record) == manifest
    boolean_version = {**record, "manifest_version": True}
    with pytest.raises(LiveRuntimeManifestError, match="identity changed"):
        live_runtime_manifest_from_record(boolean_version)
    with pytest.raises(ValueError, match="init=False"):
        replace(manifest, manifest_sha256="0" * 64)

    verify_runtime_source_closure(source)
    core.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(LiveRuntimeManifestError, match="source closure drifted"):
        verify_runtime_source_closure(source)


def test_git_probe_binds_dirty_state_without_disclosing_patch_text(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "manifest@example.invalid")
    _git(repo, "config", "user.name", "Manifest Test")
    tracked = repo / "tracked.py"
    tracked.write_text("SECRET_SENTINEL = 'before'\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "initial")
    untracked = repo / "new.py"
    untracked.write_text("UNTRACKED_SENTINEL = 1\n", encoding="utf-8")
    tracked.write_text("SECRET_SENTINEL = 'after'\n", encoding="utf-8")
    source = capture_runtime_source_closure(
        {
            "generic_core": {
                "new.py": untracked,
                "tracked.py": tracked,
            }
        }
    )

    section = capture_git_worktree_section(repo, source_closure=source)
    record = section.to_record()
    payload = record["payload"]
    assert payload["dirty"] is True
    assert payload["content_disclosure"] == "hashes_only_no_patch_or_source_text"
    assert {row["git_state"] for row in payload["relevant_source_git_states"]} == {
        "tracked",
        "untracked",
    }
    serialized = str(record)
    assert "SECRET_SENTINEL" not in serialized
    assert "UNTRACKED_SENTINEL" not in serialized
    assert payload["unstaged_binary_diff_bytes"] > 0
    assert len(payload["porcelain_v2_sha256"]) == 64


def test_git_probe_ignores_unbound_run_outputs_but_detects_bound_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "manifest@example.invalid")
    _git(repo, "config", "user.name", "Manifest Test")
    bound = repo / "bound.py"
    bound.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "bound.py")
    _git(repo, "commit", "-m", "initial")
    source = capture_runtime_source_closure(
        {"generic_core": {"bound.py": bound}}
    )

    before = capture_git_worktree_section(repo, source_closure=source)
    output = repo / "run_outputs" / "progress.jsonl"
    output.parent.mkdir()
    output.write_text('{"event":"started"}\n', encoding="utf-8")
    after_output = capture_git_worktree_section(repo, source_closure=source)
    assert after_output == before
    assert after_output.to_record()["payload"]["dirty"] is False

    bound.write_text("VALUE = 2\n", encoding="utf-8")
    after_bound_drift = capture_git_worktree_section(repo, source_closure=source)
    assert after_bound_drift != before
    assert after_bound_drift.to_record()["payload"]["dirty"] is True

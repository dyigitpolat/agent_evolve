"""Provider-free failure/durability checks for the canonical G3 launcher."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evolve.application.live_runtime_manifest import (
    RuntimeManifestSection,
    build_live_runtime_manifest,
    capture_runtime_source_closure,
)
from examples.development.durable_run_artifacts import (
    verify_finalized_run_directory,
    write_json_atomic,
)
from examples.development import run_airfoil_v7_g3 as runner
from examples.benchmarks.engibench_airfoil import v7_g3_live as live


class _Receipt:
    def to_record(self) -> dict[str, object]:
        return {"resource_key": "fake_airfoil", "acquired": True}


class _Lease:
    def __init__(self) -> None:
        self._active = False
        self.acquired = 0
        self.released = 0

    @property
    def active(self) -> bool:
        return self._active

    def acquire(self):
        self.acquired += 1
        self._active = True
        return _Receipt()

    def release(self, *, outcome="completed", failure_type=None):
        self.released += 1
        self._active = False
        return {
            "resource_key": "fake_airfoil",
            "outcome": outcome,
            "failure_type": failure_type,
        }

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.release(outcome="failed")
        return False


def _synthetic_manifest(path: Path, *, run_id: str, source_path: Path) -> None:
    source = capture_runtime_source_closure(
        {"launcher": {"synthetic_launcher.py": source_path}}
    )
    experiment = RuntimeManifestSection.seal(
        "experiment",
        {
            "schema_version": 1,
            "run_id": run_id,
            "provider_profile_id": live.DEEPSEEK_G3_PROVIDER_PROFILE.profile_id,
        },
    )
    manifest = build_live_runtime_manifest(
        manifest_id="airfoil_v7_g3_live_runtime",
        manifest_version=1,
        built_at_utc="2026-07-15T00:00:00Z",
        source_closure=source,
        sections=(experiment,),
        required_section_ids=("experiment",),
    )
    write_json_atomic(path, manifest.to_record())


def test_malformed_manifest_rejects_before_lease_or_credentials(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "malformed.json"
    manifest.write_text("{}\n", encoding="utf-8")
    lease_calls = 0
    credential_calls = 0

    def lease_factory(run_id):
        nonlocal lease_calls
        del run_id
        lease_calls += 1
        return _Lease()

    def credential_loader():
        nonlocal credential_calls
        credential_calls += 1
        return "must-not-be-read"

    dependencies = runner.AirfoilG3RunnerDependencies(
        credential_loader=credential_loader,
        problem_factory=lambda run_id, run_dir: (_ for _ in ()).throw(
            AssertionError((run_id, run_dir))
        ),
        resource_lease_factory=lease_factory,
    )
    with pytest.raises(Exception):
        asyncio.run(runner.execute_live(manifest, dependencies=dependencies))
    assert lease_calls == credential_calls == 0


def test_failure_after_lease_releases_and_finalizes_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "provider_free_failure"
    run_root = tmp_path / "runs"
    work_root = tmp_path / "work"
    manifest_root = tmp_path / "manifests"
    monkeypatch.setattr(runner, "G3_RUN_ROOT", run_root)
    monkeypatch.setattr(runner, "G3_WORK_ROOT", work_root)
    monkeypatch.setattr(runner, "MANIFEST_ROOT", manifest_root)
    manifest_root.mkdir(parents=True)
    source = tmp_path / "synthetic_launcher.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = runner.canonical_manifest_path(run_id)
    _synthetic_manifest(manifest, run_id=run_id, source_path=source)

    lease = _Lease()
    credential_calls = 0

    def credential_loader():
        nonlocal credential_calls
        credential_calls += 1
        return "must-not-be-read"

    def fail_problem_factory(observed_run_id, observed_run_dir):
        assert observed_run_id == run_id
        assert observed_run_dir == run_root / run_id
        raise RuntimeError("injected benchmark-construction failure")

    dependencies = runner.AirfoilG3RunnerDependencies(
        credential_loader=credential_loader,
        problem_factory=fail_problem_factory,
        resource_lease_factory=lambda value: lease,
    )
    with pytest.raises(runner.AirfoilG3RunnerError, match="finalized run"):
        asyncio.run(runner.execute_live(manifest, dependencies=dependencies))

    run_dir = run_root / run_id
    final = verify_finalized_run_directory(run_dir)
    assert final["status"] == "failed"
    assert lease.acquired == lease.released == 1
    assert credential_calls == 0
    assert (run_dir / "resource_lease_acquired.json").is_file()
    assert (run_dir / "resource_lease_released.json").is_file()
    assert (run_dir / "failure.json").is_file()
    assert not (run_dir / "credential_access.json").exists()


def test_completed_optimizer_result_is_persisted_before_inventory_and_analysis() -> None:
    """Guard the deliberate crash-recovery ordering in the canonical runner."""

    source = inspect.getsource(runner.execute_live)
    persistence = source.index(
        'write_json_atomic(run_dir / "optimizer_result.json", result_record)'
    )
    checkpoint = source.index('run_dir / "optimizer_checkpoint.json"')
    inventory = source.index("raw_paths = _raw_receipt_inventory(")
    analysis = source.index("analysis = await analyze_airfoil_g3_live_result(")
    transport_close = source.index('stage = "provider_transport_close"')
    assert persistence < checkpoint < transport_close < inventory < analysis


def _freeze_test_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    freeze = tmp_path / "release" / "freeze.json"
    manifests = tmp_path / "manifests"
    runs = tmp_path / "runs"
    work = tmp_path / "work"
    monkeypatch.setattr(runner, "DEFAULT_FREEZE_RECEIPT_PATH", freeze)
    monkeypatch.setattr(runner, "MANIFEST_ROOT", manifests)
    monkeypatch.setattr(runner, "G3_RUN_ROOT", runs)
    monkeypatch.setattr(runner, "G3_WORK_ROOT", work)
    return freeze, manifests, runs, work


def test_prepare_freeze_rejects_an_existing_canonical_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, _, _, _ = _freeze_test_roots(tmp_path, monkeypatch)
    freeze.parent.mkdir(parents=True)
    freeze.write_text("already frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "prepare_release",
        lambda: (_ for _ in ()).throw(AssertionError("must stop before preparation")),
    )

    with pytest.raises(runner.AirfoilG3RunnerError, match="already exists"):
        runner.prepare_freeze(frozen_at_utc="2026-07-15T00:00:00Z")


@pytest.mark.parametrize("occupied_index", (1, 2, 3))
def test_prepare_freeze_rejects_each_prior_g3_state_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied_index: int,
) -> None:
    roots = _freeze_test_roots(tmp_path, monkeypatch)
    occupied = roots[occupied_index]
    occupied.mkdir(parents=True)
    (occupied / "prior-state").write_text("present\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "prepare_release",
        lambda: (_ for _ in ()).throw(AssertionError("must stop before preparation")),
    )

    with pytest.raises(runner.AirfoilG3RunnerError, match="prior-state root"):
        runner.prepare_freeze(frozen_at_utc="2026-07-15T00:00:00Z")


def test_prepare_freeze_publishes_once_from_empty_or_absent_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze, manifests, runs, work = _freeze_test_roots(tmp_path, monkeypatch)
    manifests.mkdir(parents=True)
    work.mkdir(parents=True)
    preparation = object()
    receipt = object()
    monkeypatch.setattr(runner, "prepare_release", lambda: preparation)

    def create(observed, *, frozen_at_utc):
        assert observed is preparation
        assert frozen_at_utc == "2026-07-15T00:00:00Z"
        return receipt

    def write(observed, *, path):
        assert observed is receipt
        assert path == freeze
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen\n", encoding="utf-8")
        return "0" * 64

    monkeypatch.setattr(runner, "create_prelaunch_freeze_receipt", create)
    monkeypatch.setattr(runner, "write_prelaunch_freeze_receipt", write)
    monkeypatch.setattr(
        runner,
        "load_prelaunch_freeze_receipt",
        lambda path: receipt if path == freeze else None,
    )

    assert runner.prepare_freeze(
        frozen_at_utc="2026-07-15T00:00:00Z"
    ) == freeze
    assert freeze.is_file()
    assert not runs.exists()
    with pytest.raises(runner.AirfoilG3RunnerError, match="already exists"):
        runner.prepare_freeze(frozen_at_utc="2026-07-15T00:00:01Z")


def test_prepare_freeze_rechecks_roots_after_release_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifests, _, _ = _freeze_test_roots(tmp_path, monkeypatch)
    preparation = object()
    receipt = object()
    monkeypatch.setattr(runner, "prepare_release", lambda: preparation)

    def create(observed, *, frozen_at_utc):
        assert observed is preparation
        assert frozen_at_utc == "2026-07-15T00:00:00Z"
        manifests.mkdir(parents=True)
        (manifests / "racing-manifest.json").write_text("{}\n", encoding="utf-8")
        return receipt

    monkeypatch.setattr(runner, "create_prelaunch_freeze_receipt", create)
    monkeypatch.setattr(
        runner,
        "write_prelaunch_freeze_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must stop before publication")
        ),
    )

    with pytest.raises(runner.AirfoilG3RunnerError, match="prior-state root"):
        runner.prepare_freeze(frozen_at_utc="2026-07-15T00:00:00Z")


@pytest.mark.parametrize(
    "built_at_utc",
    ("2026-07-15T00:00:00Z", "2026-07-14T23:59:59Z"),
)
def test_manifest_chronology_rejects_equal_or_earlier_time(
    monkeypatch: pytest.MonkeyPatch,
    built_at_utc: str,
) -> None:
    freeze_sha256 = "a" * 64
    monkeypatch.setattr(
        live,
        "load_prelaunch_freeze_receipt",
        lambda: SimpleNamespace(
            freeze_receipt_sha256=freeze_sha256,
            frozen_at_utc="2026-07-15T00:00:00Z",
        ),
    )

    with pytest.raises(live.AirfoilG3LiveError, match="strictly after"):
        live.verify_airfoil_g3_manifest_chronology(
            built_at_utc=built_at_utc,
            expected_freeze_receipt_sha256=freeze_sha256,
        )


def test_manifest_chronology_accepts_only_later_time_and_exact_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_sha256 = "a" * 64
    monkeypatch.setattr(
        live,
        "load_prelaunch_freeze_receipt",
        lambda: SimpleNamespace(
            freeze_receipt_sha256=freeze_sha256,
            frozen_at_utc="2026-07-15T00:00:00Z",
        ),
    )

    live.verify_airfoil_g3_manifest_chronology(
        built_at_utc="2026-07-15T00:00:01Z",
        expected_freeze_receipt_sha256=freeze_sha256,
    )
    with pytest.raises(live.AirfoilG3LiveError, match="foreign"):
        live.verify_airfoil_g3_manifest_chronology(
            built_at_utc="2026-07-15T00:00:01Z",
            expected_freeze_receipt_sha256="b" * 64,
        )

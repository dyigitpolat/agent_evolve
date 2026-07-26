"""Shared artifact-store contract plus filesystem durability checks."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import agent_evolve.domain.artifact as artifact_domain
import agent_evolve.infrastructure.artifacts.filesystem as filesystem_module
from agent_evolve.domain.artifact import ArtifactRef
from agent_evolve.domain.ids import ArtifactId, RunId
from agent_evolve.infrastructure.artifacts import (
    FileSystemArtifactStore,
    InMemoryArtifactStore,
)
from agent_evolve.ports.artifact_store import (
    ArtifactCollisionError,
    ArtifactNotFoundError,
    ArtifactSerializationError,
    ArtifactTypeError,
    CorruptArtifactError,
    canonical_json_bytes,
    decode_json_bytes,
    put_json,
    put_text,
    read_json,
    read_text,
)


def _put_for_multiprocess_collision(
    root,
    content,
    barrier,
    result_queue,
):
    """Fork-safe worker used to exercise the POSIX advisory-lock boundary."""

    try:
        store = FileSystemArtifactStore(root)
        barrier.wait(timeout=10)
        ref = store.put_bytes(content, media_type="application/octet-stream")
        result_queue.put(("ok", ref.artifact_id.value, content))
    except BaseException as exc:  # pragma: no cover - asserted in parent process.
        result_queue.put(("error", type(exc).__name__, str(exc)))


@pytest.fixture(params=["memory", "filesystem"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryArtifactStore()
    return FileSystemArtifactStore(tmp_path / "artifacts")


def test_store_derives_typed_identity_and_raw_checksum_and_returns_frozen_metadata(
    store,
):
    content = b"exact payload bytes\x00\xff"
    media_type = "application/octet-stream"
    payload_digest = hashlib.sha256(content).hexdigest()
    media_type_bytes = media_type.encode("ascii")
    identity_preimage = (
        b"agent-evolve:artifact-id:v2\x00"
        + len(media_type_bytes).to_bytes(8, "big")
        + media_type_bytes
        + len(content).to_bytes(8, "big")
        + content
    )
    identity_digest = hashlib.sha256(identity_preimage).hexdigest()
    ref = store.put_bytes(content, media_type="application/octet-stream")

    assert ref == ArtifactRef(
        artifact_id=ArtifactId(f"artifact_{identity_digest}"),
        sha256_hex=payload_digest,
        size_bytes=len(content),
        media_type=media_type,
    )
    assert store.stat(ref.artifact_id) == ref
    assert store.read_bytes(
        ref.artifact_id,
        expected_media_type="application/octet-stream",
    ) == content
    with pytest.raises(FrozenInstanceError):
        ref.media_type = "text/plain"


def test_store_is_idempotent_for_same_bytes_and_exact_type(store):
    first = store.put_bytes(b"same", media_type="application/octet-stream")
    second = store.put_bytes(b"same", media_type="application/octet-stream")
    assert second is first or second == first


@pytest.mark.parametrize(
    "media_types",
    [
        ("application/json", "text/plain"),
        ("text/plain", "application/json"),
    ],
)
def test_store_keeps_same_bytes_under_distinct_types_in_either_order(
    store,
    media_types,
):
    first = store.put_bytes(b"typed", media_type=media_types[0])
    second = store.put_bytes(b"typed", media_type=media_types[1])

    assert first.artifact_id != second.artifact_id
    assert first.sha256_hex == second.sha256_hex
    assert {first.media_type, second.media_type} == {
        "application/json",
        "text/plain",
    }
    assert store.read_bytes(
        first.artifact_id,
        expected_media_type=first.media_type,
    ) == b"typed"
    assert store.read_bytes(
        second.artifact_id,
        expected_media_type=second.media_type,
    ) == b"typed"
    with pytest.raises(ArtifactTypeError, match="has media type"):
        store.read_bytes(
            first.artifact_id,
            expected_media_type=second.media_type,
        )


def test_store_has_explicit_input_type_and_not_found_errors(store):
    with pytest.raises(ArtifactTypeError, match="immutable bytes"):
        store.put_bytes(bytearray(b"mutable"), media_type="application/octet-stream")
    with pytest.raises(ArtifactTypeError, match="type and subtype"):
        store.put_bytes(b"value", media_type="untyped")
    with pytest.raises(ArtifactTypeError, match="ArtifactId"):
        store.read_bytes(RunId("run_wrong_runtime_type"))
    with pytest.raises(ArtifactNotFoundError, match="was not found"):
        store.read_bytes(ArtifactId(f"artifact_{'a' * 64}"))
    with pytest.raises(ValueError, match="lowercase 64-hex"):
        ArtifactId("artifact_diagnostic")


def test_store_reports_a_digest_collision_without_overwriting(
    store,
    monkeypatch,
):
    forced_digest = "0" * 64
    monkeypatch.setattr(
        artifact_domain,
        "artifact_identity_sha256",
        lambda content, *, media_type: forced_digest,
    )
    first = store.put_bytes(b"alpha", media_type="application/octet-stream")
    with pytest.raises(ArtifactCollisionError, match="Different payload bytes"):
        store.put_bytes(b"bravo", media_type="application/octet-stream")
    assert store.read_bytes(first.artifact_id) == b"alpha"


def test_canonical_json_and_utf8_text_helpers_round_trip(store):
    first = put_json(store, {"b": [2, True], "a": "café"})
    second = put_json(store, {"a": "café", "b": [2, True]})
    assert first == second
    assert read_json(store, first.artifact_id) == {
        "a": "café",
        "b": [2, True],
    }

    text = put_text(store, "line one\nλ")
    assert read_text(store, text.artifact_id) == "line one\nλ"


def test_json_codec_rejects_ambiguous_or_nonstandard_values():
    with pytest.raises(ArtifactSerializationError, match="must be a string"):
        canonical_json_bytes({1: "ambiguous key coercion"})
    with pytest.raises(ArtifactSerializationError, match="Non-finite"):
        canonical_json_bytes({"score": float("nan")})
    with pytest.raises(ArtifactSerializationError, match="Duplicate"):
        decode_json_bytes(b'{"value":1,"value":2}')
    with pytest.raises(ArtifactSerializationError, match="Non-standard"):
        decode_json_bytes(b'{"value":NaN}')
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ArtifactSerializationError, match="Cyclic"):
        canonical_json_bytes(cyclic)
    with pytest.raises(ArtifactSerializationError, match="not strict UTF-8"):
        canonical_json_bytes("\ud800")
    with pytest.raises(ArtifactSerializationError, match="not strict UTF-8"):
        canonical_json_bytes({"\udfff": "invalid key"})
    with pytest.raises(ArtifactSerializationError, match="not strict UTF-8"):
        decode_json_bytes(b'"\\ud800"')
    with pytest.raises(ArtifactSerializationError, match="not strict UTF-8"):
        decode_json_bytes(b'{"\\udfff":1}')
    assert decode_json_bytes(b'"\\ud83d\\ude00"') == "😀"
    assert decode_json_bytes(canonical_json_bytes("scalar 😀")) == "scalar 😀"


def test_filesystem_store_reopens_and_reverifies_all_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    ref = store.put_bytes(b"durable", media_type="application/octet-stream")

    reopened = FileSystemArtifactStore(root)
    assert reopened.stat(ref.artifact_id) == ref
    assert reopened.read_bytes(ref.artifact_id) == b"durable"
    assert not tuple(root.glob(".artifact-write-*.tmp"))


def test_filesystem_store_rejects_pre_v2_container_magic(tmp_path):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    ref = store.put_bytes(b"legacy marker", media_type="application/octet-stream")
    path = store.path_for_artifact(ref.artifact_id)
    encoded = path.read_bytes()
    path.write_bytes(
        encoded.replace(
            b"AGENT_EVOLVE_ARTIFACT_V2",
            b"AGENT_EVOLVE_ARTIFACT_V1",
            1,
        )
    )

    with pytest.raises(CorruptArtifactError, match="invalid header"):
        FileSystemArtifactStore(root)


def test_filesystem_store_detects_payload_corruption_on_read_and_reopen(tmp_path):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    ref = store.put_bytes(b"original", media_type="application/octet-stream")
    path = store.path_for_artifact(ref.artifact_id)
    encoded = path.read_bytes()
    path.write_bytes(encoded[:-8] + b"tampered")

    with pytest.raises(CorruptArtifactError, match="SHA-256 digest"):
        store.read_bytes(ref.artifact_id)
    with pytest.raises(CorruptArtifactError, match="SHA-256 digest"):
        FileSystemArtifactStore(root)


def test_filesystem_store_detects_noncanonical_metadata_and_wrong_filename(tmp_path):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    ref = store.put_bytes(b"metadata", media_type="application/octet-stream")
    path = store.path_for_artifact(ref.artifact_id)
    magic, metadata, content = path.read_bytes().split(b"\n", 2)
    path.write_bytes(magic + b"\n" + b" " + metadata + b"\n" + content)
    with pytest.raises(CorruptArtifactError, match="not canonical JSON"):
        FileSystemArtifactStore(root)

    path.write_bytes(magic + b"\n" + metadata + b"\n" + content)
    wrong_path = root / f"artifact_{'f' * 64}.artifact"
    path.rename(wrong_path)
    with pytest.raises(CorruptArtifactError, match="file name"):
        FileSystemArtifactStore(root)


def test_filesystem_store_wraps_metadata_codec_failures_as_corruption(tmp_path):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    ref = store.put_bytes(b"metadata", media_type="application/octet-stream")
    path = store.path_for_artifact(ref.artifact_id)
    magic, metadata, content = path.read_bytes().split(b"\n", 2)
    invalid_metadata = metadata.replace(
        b"application/octet-stream",
        b"application/\\ud800",
    )
    path.write_bytes(magic + b"\n" + invalid_metadata + b"\n" + content)

    with pytest.raises(CorruptArtifactError, match="malformed metadata"):
        FileSystemArtifactStore(root)


def test_filesystem_store_wraps_canonical_encoder_failures_as_corruption(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    store.put_bytes(b"metadata", media_type="application/octet-stream")

    def fail_canonical_encoding(value):
        raise ArtifactSerializationError("injected canonical codec failure")

    monkeypatch.setattr(
        filesystem_module,
        "canonical_json_bytes",
        fail_canonical_encoding,
    )
    with pytest.raises(CorruptArtifactError, match="cannot be encoded canonically"):
        FileSystemArtifactStore(root)


def test_filesystem_store_detects_canonical_media_type_tampering(tmp_path):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    ref = store.put_bytes(b"metadata", media_type="application/octet-stream")
    path = store.path_for_artifact(ref.artifact_id)
    magic, metadata, content = path.read_bytes().split(b"\n", 2)
    record = decode_json_bytes(metadata)
    record["media_type"] = "application/x-tampered"
    path.write_bytes(
        magic + b"\n" + canonical_json_bytes(record) + b"\n" + content
    )

    with pytest.raises(CorruptArtifactError, match="typed payload"):
        FileSystemArtifactStore(root)


@pytest.mark.parametrize(
    "media_types",
    [
        ("application/json", "text/plain"),
        ("text/plain", "application/json"),
    ],
)
def test_two_filesystem_store_instances_are_type_order_independent(
    tmp_path,
    media_types,
):
    root = tmp_path / "artifacts"
    stores = (FileSystemArtifactStore(root), FileSystemArtifactStore(root))
    refs = [
        stores[index].put_bytes(b"shared bytes", media_type=media_type)
        for index, media_type in enumerate(media_types)
    ]

    assert len({ref.artifact_id for ref in refs}) == 2
    assert refs[0].sha256_hex == refs[1].sha256_hex
    reopened = FileSystemArtifactStore(root)
    for ref in refs:
        assert reopened.read_bytes(
            ref.artifact_id,
            expected_media_type=ref.media_type,
        ) == b"shared bytes"


def test_two_filesystem_store_instances_serialize_concurrent_writers(tmp_path):
    root = tmp_path / "artifacts"
    stores = (FileSystemArtifactStore(root), FileSystemArtifactStore(root))
    barrier = threading.Barrier(12)

    def put(index: int) -> ArtifactRef:
        barrier.wait(timeout=5)
        return stores[index % 2].put_bytes(
            b"concurrent payload",
            media_type="application/octet-stream",
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        refs = list(executor.map(put, range(12)))

    assert len({ref.artifact_id for ref in refs}) == 1
    assert FileSystemArtifactStore(root).read_bytes(refs[0].artifact_id) == (
        b"concurrent payload"
    )
    assert not tuple(root.glob(".artifact-write-*.tmp"))


@pytest.mark.skipif(
    os.name != "posix" or filesystem_module.fcntl is None,
    reason="requires POSIX fork and fcntl.flock",
)
def test_posix_multiprocess_flock_serializes_a_forced_identity_collision(
    tmp_path,
    monkeypatch,
):
    forced_digest = "0" * 64
    monkeypatch.setattr(
        artifact_domain,
        "artifact_identity_sha256",
        lambda content, *, media_type: forced_digest,
    )
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    root = tmp_path / "artifacts"
    processes = [
        context.Process(
            target=_put_for_multiprocess_collision,
            args=(str(root), content, barrier, result_queue),
        )
        for content in (b"alpha", b"bravo")
    ]

    try:
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result[0] for result in results) == ["error", "ok"]
    error = next(result for result in results if result[0] == "error")
    success = next(result for result in results if result[0] == "ok")
    assert error[1] == "ArtifactCollisionError"
    persisted = FileSystemArtifactStore(root).read_bytes(
        ArtifactId(success[1]),
    )
    assert persisted == success[2]


def test_missing_fcntl_uses_one_process_global_root_lock(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(filesystem_module, "fcntl", None)
    root = tmp_path / "artifacts"
    stores = (FileSystemArtifactStore(root), FileSystemArtifactStore(root))
    original_write = FileSystemArtifactStore._atomic_write_unlocked
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def observed_write(self, path, encoded):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.02)
            return original_write(self, path, encoded)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(
        FileSystemArtifactStore,
        "_atomic_write_unlocked",
        observed_write,
    )
    barrier = threading.Barrier(2)

    def put(store_index: int) -> ArtifactRef:
        barrier.wait(timeout=5)
        return stores[store_index].put_bytes(
            f"payload {store_index}".encode(),
            media_type="application/octet-stream",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = list(executor.map(put, range(2)))

    assert maximum_active == 1
    assert len({ref.artifact_id for ref in refs}) == 2


def test_runtime_flock_failure_is_fail_closed(tmp_path, monkeypatch):
    class UnsupportedFcntl:
        LOCK_EX = 1
        LOCK_UN = 2

        @staticmethod
        def flock(fd, operation):
            raise OSError("flock unsupported")

    monkeypatch.setattr(filesystem_module, "fcntl", UnsupportedFcntl())
    with pytest.raises(OSError, match="flock unsupported"):
        FileSystemArtifactStore(tmp_path / "artifacts")


def test_new_filesystem_root_requires_parent_and_fsyncs_parent(
    tmp_path,
    monkeypatch,
):
    with pytest.raises(FileNotFoundError, match="parent directory must already exist"):
        FileSystemArtifactStore(tmp_path / "missing" / "artifacts")

    calls: list[Path] = []
    original_fsync = FileSystemArtifactStore._fsync_directory_path

    def observed_fsync(path: Path) -> None:
        calls.append(path)
        original_fsync(path)

    monkeypatch.setattr(
        FileSystemArtifactStore,
        "_fsync_directory_path",
        staticmethod(observed_fsync),
    )
    FileSystemArtifactStore(tmp_path / "artifacts")
    assert calls == [tmp_path.resolve()]


def test_failed_atomic_replace_leaves_no_visible_or_temporary_artifact(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    expected_ref = artifact_domain.artifact_ref_for_bytes(
        b"interrupted",
        media_type="application/octet-stream",
    )
    original_replace = filesystem_module.os.replace

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(filesystem_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        store.put_bytes(b"interrupted", media_type="application/octet-stream")

    assert not store.path_for_artifact(expected_ref.artifact_id).exists()
    assert not tuple(root.glob(".artifact-write-*.tmp"))

    monkeypatch.setattr(filesystem_module.os, "replace", original_replace)
    assert store.put_bytes(
        b"interrupted",
        media_type="application/octet-stream",
    ) == expected_ref


def test_failed_payload_write_leaves_no_visible_or_temporary_artifact(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    expected_ref = artifact_domain.artifact_ref_for_bytes(
        b"partial write",
        media_type="application/octet-stream",
    )
    original_write = FileSystemArtifactStore._write_all

    def fail_write(fd, content):
        os.write(fd, content[:3])
        raise OSError("injected payload write failure")

    monkeypatch.setattr(
        FileSystemArtifactStore,
        "_write_all",
        staticmethod(fail_write),
    )
    with pytest.raises(OSError, match="injected payload write failure"):
        store.put_bytes(b"partial write", media_type="application/octet-stream")

    assert not store.path_for_artifact(expected_ref.artifact_id).exists()
    assert not tuple(root.glob(".artifact-write-*.tmp"))

    monkeypatch.setattr(
        FileSystemArtifactStore,
        "_write_all",
        staticmethod(original_write),
    )
    assert store.put_bytes(
        b"partial write",
        media_type="application/octet-stream",
    ) == expected_ref


def test_failed_payload_fsync_leaves_no_visible_or_temporary_artifact(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    expected_ref = artifact_domain.artifact_ref_for_bytes(
        b"unsynced payload",
        media_type="application/octet-stream",
    )
    original_fsync = filesystem_module.os.fsync
    call_count = 0

    def fail_first_fsync(fd):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("injected payload fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(filesystem_module.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="injected payload fsync failure"):
        store.put_bytes(b"unsynced payload", media_type="application/octet-stream")

    assert not store.path_for_artifact(expected_ref.artifact_id).exists()
    assert not tuple(root.glob(".artifact-write-*.tmp"))

    monkeypatch.setattr(filesystem_module.os, "fsync", original_fsync)
    assert store.put_bytes(
        b"unsynced payload",
        media_type="application/octet-stream",
    ) == expected_ref


def test_failed_root_fsync_is_recovered_by_an_idempotent_retry(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "artifacts"
    store = FileSystemArtifactStore(root)
    expected_ref = artifact_domain.artifact_ref_for_bytes(
        b"rename completed",
        media_type="application/octet-stream",
    )
    original_root_fsync = store._fsync_root_directory

    def fail_root_fsync():
        raise OSError("injected root fsync failure")

    monkeypatch.setattr(store, "_fsync_root_directory", fail_root_fsync)
    with pytest.raises(OSError, match="injected root fsync failure"):
        store.put_bytes(b"rename completed", media_type="application/octet-stream")

    assert store.path_for_artifact(expected_ref.artifact_id).is_file()
    recovery_calls = 0

    def recover_root_fsync():
        nonlocal recovery_calls
        recovery_calls += 1
        original_root_fsync()

    monkeypatch.setattr(store, "_fsync_root_directory", recover_root_fsync)
    assert store.put_bytes(
        b"rename completed",
        media_type="application/octet-stream",
    ) == expected_ref
    assert recovery_calls == 1
    assert FileSystemArtifactStore(root).read_bytes(expected_ref.artifact_id) == (
        b"rename completed"
    )

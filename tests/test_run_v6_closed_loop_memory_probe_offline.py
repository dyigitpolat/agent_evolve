"""Offline/readiness tests for the durable v6 engineering probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_probe_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "run_v6_closed_loop_memory_probe.py"
    )
    name = "_agent_evolve_test_run_v6_closed_loop_memory_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PROBE = _load_probe_module()


@pytest.fixture(autouse=True)
def _forbid_dotenv_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("an offline test attempted to load .env")

    monkeypatch.setattr(_PROBE, "load_dotenv", forbidden)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_recursive_finalization(run_dir: Path, *, status: str) -> None:
    finalized = _json(run_dir / "finalized.json")
    assert finalized["status"] == status
    assert finalized["recursive_file_count"] == len(finalized["files"])
    observed_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name != "finalized.json"
        and not path.name.endswith(".tmp")
    }
    assert set(finalized["files"]) == observed_paths
    for relative, record in finalized["files"].items():
        content = (run_dir / relative).read_bytes()
        assert record["bytes"] == len(content)
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
        if relative.endswith(".jsonl"):
            assert record["jsonl_lines"] == len(content.splitlines())
    assert not tuple(run_dir.rglob("*.tmp"))


def test_engine_readiness_commitments_are_computed_replayable_and_blinded() -> None:
    first = _PROBE.run_async_sync(_PROBE.prepare_readiness())
    replay = _PROBE.run_async_sync(_PROBE.prepare_readiness())

    assert first == replay
    assert first["provider_io_performed"] is False
    assert first["queue_started"] is False
    assert first["prompt_shape_source"] == (
        "AgenticEvolutionEngine.prompt_shape_commitment"
    )
    assert len(first["prompt_shape_sha256"]) == 64
    assignments = first["assignments_by_sha256"]
    assert len(assignments) == 2
    assert {
        row["assignment"]["prompt_shape_sha256"] for row in assignments.values()
    } == {first["prompt_shape_sha256"]}
    # Treatments differ, so rendered prompt hashes differ even though their
    # treatment-blinded non-treatment shape commitment is identical.
    assert len({row["prepared_prompt_sha256"] for row in assignments.values()}) == 2
    assert first["g1_model_wave_width"] == 2
    assert first["g2_model_wave_width"] == 2
    assert first["g2_full_wave_width"] == 4


def test_preview_is_default_and_never_composes_a_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ForbiddenStructuredFactory:
        @staticmethod
        def openrouter(**kwargs):
            del kwargs
            raise AssertionError("preview attempted to compose OpenRouter")

    monkeypatch.setattr(
        _PROBE,
        "PydanticAIStructuredGenerator",
        _ForbiddenStructuredFactory,
    )
    assert _PROBE.main(["--log-root", str(tmp_path), "--run-id", "preview"]) == 0
    run_dir = tmp_path / "preview"

    manifest = _json(run_dir / "manifest.json")
    readiness = _json(run_dir / "readiness.json")
    analysis = _json(run_dir / "mechanism_analysis.json")
    summary = _json(run_dir / "summary.json")
    assert manifest["mode"] == "preview"
    assert manifest["provider_io_authorized"] is False
    assert manifest["scientific_interpretation"] == {
        "causal_effect_identifiable": False,
        "fixed_assignment_ranks": {
            "control_score_permutation_rank": 1,
            "diagnostic_uniform_subset_ranks": [0, 1],
        },
        "fixture_contrast_only": True,
        "model_reasoning_quality_tested": False,
        "provider_integration_smoke_only_if_live": True,
        "scripted_fixture": True,
    }
    assert (
        manifest["external_freeze_commitments"]["all_provided_commitments_match"]
        is False
    )
    assert manifest["preregistration_authorization"] == {
        "reason": "not_live",
        "required_for_live": True,
        "validated": False,
    }
    assert manifest["route_snapshot_evidence"]["route_validated"] is True
    assert set(manifest["route_snapshot_evidence"]["snapshots"]) == {
        "capability",
        "pricing",
    }
    assert readiness["provider_io_performed"] is False
    assert analysis["overall_pass"] is None
    assert analysis["hypothesis_outcome"] == "not_executed_readiness_only"
    assert summary["status"] == "preview_ready"
    assert _jsonl(run_dir / "events.jsonl") == []
    assert _jsonl(run_dir / "queue_outcomes.jsonl") == []
    assert _jsonl(run_dir / "outcomes.jsonl") == [
        {
            "provider_io_performed": False,
            "readiness_sha256": readiness["readiness_sha256"],
            "record_type": "readiness_preview",
            "schema_version": 1,
        }
    ]
    source = _json(run_dir / "source_snapshot.json")
    assert source["file_count"] >= 16
    assert len(source["sha256"]) == 64
    snapshot_root = run_dir / "source_snapshot"
    snapshot_paths = {
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file()
    }
    assert snapshot_paths == set(source["files"])
    assert len(snapshot_paths) == source["file_count"]
    source_verifications = _jsonl(run_dir / "source_verifications.jsonl")
    assert len(source_verifications) == 1
    assert source_verifications[0]["stage"] == "post_preview_readiness"
    assert source_verifications[0]["verified"] is True
    _assert_recursive_finalization(run_dir, status="preview_ready")


def test_offline_mode_executes_full_probe_and_publishes_machine_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ForbiddenStructuredFactory:
        @staticmethod
        def openrouter(**kwargs):
            del kwargs
            raise AssertionError("offline mode attempted to compose OpenRouter")

    monkeypatch.setattr(
        _PROBE,
        "PydanticAIStructuredGenerator",
        _ForbiddenStructuredFactory,
    )
    assert (
        _PROBE.main(
            [
                "--offline",
                "--log-root",
                str(tmp_path),
                "--run-id",
                "offline",
            ]
        )
        == 0
    )
    run_dir = tmp_path / "offline"
    manifest = _json(run_dir / "manifest.json")
    analysis = _json(run_dir / "mechanism_analysis.json")
    summary = _json(run_dir / "summary.json")
    events = _jsonl(run_dir / "events.jsonl")
    outcomes = _jsonl(run_dir / "outcomes.jsonl")

    assert manifest["mode"] == "offline"
    assert manifest["development_only"] is True
    assert manifest["provider_io_authorized"] is False
    assert manifest["concurrency"] == {
        "engine_evaluator_concurrency": 4,
        "g1_model_wave_width": 2,
        "g2_full_wave_width": 4,
        "g2_model_wave_width": 2,
        "queue_max_in_flight": 4,
    }
    assert manifest["queue"]["backoff"] == {
        "base_delay_ns": 1_000_000_000,
        "kind": "exponential",
        "max_delay_ns": 8_000_000_000,
    }
    assert manifest["queue"]["jitter"] == {
        "domain": _PROBE.JITTER_DOMAIN,
        "kind": "task_keyed_sha256",
        "seed": _PROBE.JITTER_SEED,
    }
    assert manifest["queue"]["schema_repair_policy"] == (
        _PROBE.SCHEMA_REPAIR_POLICY_MANIFEST.to_trace_record()
    )
    assert len(manifest["queue"]["schema_repair_policy"]["policy_sha256"]) == 64
    exposure = manifest["cost_exposure"]
    assert exposure["accepted_success_response_cap_usd"] == "0.010"
    assert exposure["accepted_success_run_cap_usd"] == "0.040"
    assert exposure["max_attempts_per_logical_call"] == 2
    assert (
        exposure["conservative_declared_potentially_billable_run_exposure_usd"]
        == "0.080"
    )
    assert "not a mechanically guaranteed provider-spend cap" in exposure["caveat"]
    assert (
        exposure["pricing_derivation"]["derived_max_successful_response_cost_usd"]
        == "0.0078987648"
    )
    assert manifest["route_snapshot_evidence"]["route_validated"] is True
    assert analysis["overall_pass"] is True
    assert analysis["causal_effect_identifiable"] is False
    assert analysis["model_reasoning_quality_tested"] is False
    assert analysis["fixture_contrast_only"] is True
    assert analysis["hypothesis_outcome"] == "engineering_fixture_path_passed"
    assert all(check["passed"] is True for check in analysis["checks"].values())
    assert summary["status"] == "engineering_fixture_path_passed"
    assert summary["unique_evaluations"] == 6
    assert summary["logical_llm_calls"] == 4
    assert len(outcomes) == 9  # one seed plus eight engineering-fixture occurrences
    assert outcomes[-1]["configuration"] == {"a": 1, "b": 1}
    assert outcomes[-1]["proposal_authority"] == "engine"
    assert _jsonl(run_dir / "queue_outcomes.jsonl") == []
    assert sum(event["event_type"] == "assignment_committed" for event in events) == 4
    assert all(
        event["prompt_shape_commitment_verified"] is True
        for event in events
        if event["event_type"] == "assignment_committed"
    )
    paired = analysis["checks"]["adaptive_control_path_realized_expected_contrast"][
        "observed"
    ]
    assert paired["adaptive_slot_id"] == "G2-adaptive"
    assert paired["adaptive_slot_role"] == "adaptive_memory"
    assert paired["adaptive_proposal_authority"] == "model"
    assert paired["control_slot_id"] == "G2-control"
    assert paired["control_slot_role"] == "score_shuffled_control"
    assert paired["control_proposal_authority"] == "model"
    assert paired["adaptive_configuration"] == {"a": 1, "b": 4}
    assert paired["control_configuration"] == {"a": 3, "b": 4}
    assert float.fromhex(paired["adaptive_reward_hex"]) == 3.0
    assert float.fromhex(paired["control_reward_hex"]) == 1.0
    assert float.fromhex(paired["adaptive_minus_control_reward_hex"]) == 2.0
    assert paired["adaptive_reward_hex"] == paired["adaptive_recomputed_reward_hex"]
    assert paired["control_reward_hex"] == paired["control_recomputed_reward_hex"]
    assert paired["phenotype_equal"] is False
    assert paired["adaptive_phenotype_sha256"] != paired["control_phenotype_sha256"]
    g2_rows = {
        row["slot_id"]: row
        for row in outcomes
        if row.get("generation") == 2
        and row.get("slot_id") in {"G2-adaptive", "G2-control"}
    }
    assert g2_rows["G2-adaptive"]["configuration"] == paired["adaptive_configuration"]
    assert g2_rows["G2-adaptive"]["reward_hex"] == paired["adaptive_reward_hex"]
    assert g2_rows["G2-control"]["configuration"] == paired["control_configuration"]
    assert g2_rows["G2-control"]["reward_hex"] == paired["control_reward_hex"]
    verifications = _jsonl(run_dir / "source_verifications.jsonl")
    assert [(row["stage"], row["verified"]) for row in verifications] == [
        ("pre_offline_execution", True),
        ("post_offline_execution", True),
    ]
    _assert_recursive_finalization(
        run_dir,
        status="engineering_fixture_path_passed",
    )


def test_live_freeze_requirements_are_exact_and_pure() -> None:
    sha_a = "a" * 64
    sha_b = "b" * 64
    for kwargs in (
        {
            "run_id": None,
            "expected_source_sha256": sha_a,
            "expected_readiness_sha256": sha_b,
        },
        {
            "run_id": _PROBE.AUTHORIZED_LIVE_RUN_ID,
            "expected_source_sha256": None,
            "expected_readiness_sha256": sha_b,
        },
        {
            "run_id": _PROBE.AUTHORIZED_LIVE_RUN_ID,
            "expected_source_sha256": sha_a.upper(),
            "expected_readiness_sha256": sha_b,
        },
        {
            "run_id": _PROBE.AUTHORIZED_LIVE_RUN_ID,
            "expected_source_sha256": sha_a,
            "expected_readiness_sha256": "short",
        },
        {
            "run_id": "a-second-live-run-is-forbidden",
            "expected_source_sha256": sha_a,
            "expected_readiness_sha256": sha_b,
        },
    ):
        with pytest.raises(ValueError):
            _PROBE._validate_live_cli_freeze_requirements(
                mode="live",
                log_root=_PROBE.DEFAULT_LOG_ROOT,
                **kwargs,
            )

    _PROBE._validate_live_cli_freeze_requirements(
        mode="live",
        run_id=_PROBE.AUTHORIZED_LIVE_RUN_ID,
        expected_source_sha256=sha_a,
        expected_readiness_sha256=sha_b,
        log_root=_PROBE.DEFAULT_LOG_ROOT,
    )
    with pytest.raises(ValueError, match="canonical immutable log root"):
        _PROBE._validate_live_cli_freeze_requirements(
            mode="live",
            run_id=_PROBE.AUTHORIZED_LIVE_RUN_ID,
            expected_source_sha256=sha_a,
            expected_readiness_sha256=sha_b,
            log_root=Path("/tmp/v6-alternate-live-root"),
        )
    with pytest.raises(RuntimeError, match="source_snapshot_sha256"):
        _PROBE._external_freeze_commitments(
            mode="live",
            expected_source_sha256=sha_b,
            expected_readiness_sha256=sha_b,
            source_snapshot={"sha256": sha_a},
            readiness={"readiness_sha256": sha_b},
        )
    committed = _PROBE._external_freeze_commitments(
        mode="live",
        expected_source_sha256=sha_a,
        expected_readiness_sha256=sha_b,
        source_snapshot={"sha256": sha_a},
        readiness={"readiness_sha256": sha_b},
    )
    assert committed["all_provided_commitments_match"] is True


def test_live_preregistration_must_prospectively_bind_exact_freeze(
    tmp_path: Path,
) -> None:
    sha_a = "a" * 64
    sha_b = "b" * 64
    path = tmp_path / "86.md"
    exact_lines = "\n".join(
        (
            _PROBE.PREREGISTRATION_READY_MARKER,
            f"AUTHORIZED_LIVE_RUN_ID: `{_PROBE.AUTHORIZED_LIVE_RUN_ID}`",
            f"SOURCE_SNAPSHOT_SHA256: `{sha_a}`",
            f"READINESS_SHA256: `{sha_b}`",
        )
    )
    path.write_text(exact_lines + "\n", encoding="utf-8")

    evidence = _PROBE._preregistration_authorization_evidence(
        mode="live",
        run_id=_PROBE.AUTHORIZED_LIVE_RUN_ID,
        expected_source_sha256=sha_a,
        expected_readiness_sha256=sha_b,
        path=path,
    )

    content = path.read_bytes()
    assert evidence["validated"] is True
    assert evidence["sha256"] == hashlib.sha256(content).hexdigest()
    assert evidence["bytes"] == len(content)
    assert evidence["authorized_run_id"] == _PROBE.AUTHORIZED_LIVE_RUN_ID

    path.write_text(
        exact_lines.replace(_PROBE.PREREGISTRATION_READY_MARKER, "") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks exact executable"):
        _PROBE._preregistration_authorization_evidence(
            mode="live",
            run_id=_PROBE.AUTHORIZED_LIVE_RUN_ID,
            expected_source_sha256=sha_a,
            expected_readiness_sha256=sha_b,
            path=path,
        )


@pytest.mark.parametrize("run_id", [".", "..", "../escape", "/absolute", "bad/id"])
def test_run_id_is_one_safe_component(run_id: str) -> None:
    with pytest.raises(ValueError):
        _PROBE._validate_run_id(run_id)


def test_source_snapshot_copies_the_single_captured_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "changing.py"
    source.write_bytes(b"later on-disk bytes")
    captured = b"the one captured source read"
    digest = hashlib.sha256(captured).hexdigest()
    monkeypatch.setattr(
        _PROBE,
        "_read_source_payloads",
        lambda: (((source, "src/changing.py", captured, digest),), "f" * 64),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    snapshot = _PROBE._snapshot_sources(run_dir)

    assert (run_dir / "source_snapshot" / "src" / "changing.py").read_bytes() == (
        captured
    )
    assert source.read_bytes() == b"later on-disk bytes"
    assert snapshot["files"] == {"src/changing.py": digest}
    assert snapshot["sha256"] == "f" * 64


def test_source_drift_verification_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"files": {"src/a.py": "a" * 64}, "sha256": "b" * 64}
    monkeypatch.setattr(
        _PROBE,
        "_source_state",
        lambda: ({"src/a.py": "c" * 64}, "d" * 64),
    )
    with pytest.raises(RuntimeError, match="pre_queue_enter"):
        _PROBE._verify_source_snapshot(expected, stage="pre_queue_enter")


def test_failure_still_closes_and_recursively_finalizes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_readiness():
        raise RuntimeError("deliberate provider-free readiness failure")

    monkeypatch.setattr(_PROBE, "prepare_readiness", fail_readiness)
    with pytest.raises(RuntimeError, match="deliberate provider-free"):
        _PROBE.main(["--preview", "--log-root", str(tmp_path), "--run-id", "failed"])

    run_dir = tmp_path / "failed"
    assert _json(run_dir / "mechanism_analysis.json")["overall_pass"] is False
    assert _json(run_dir / "failure.json")["failure_type"] == "RuntimeError"
    for name in (
        "events.jsonl",
        "outcomes.jsonl",
        "queue_outcomes.jsonl",
        "source_verifications.jsonl",
    ):
        assert _jsonl(run_dir / name) == []
    _assert_recursive_finalization(run_dir, status="failed")


def test_run_directory_is_immutable_and_never_reuses_existing_owner(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "owned"
    run_dir.mkdir()
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("existing owner\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _PROBE.main(["--preview", "--log-root", str(tmp_path), "--run-id", "owned"])

    assert tuple(run_dir.iterdir()) == (sentinel,)
    assert sentinel.read_text(encoding="utf-8") == "existing owner\n"


def test_live_stack_is_exact_required_and_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class _StructuredFactory:
        @staticmethod
        def openrouter(**kwargs):
            calls["openrouter"] = kwargs
            return object()

    class _QueuedRunner:
        async def __call__(self, request):  # pragma: no cover - must stay unused.
            del request
            raise AssertionError("live runner was called")

    def queue_factory(**kwargs):
        calls["queue"] = kwargs
        return _QueuedRunner()

    monkeypatch.setattr(
        _PROBE,
        "PydanticAIStructuredGenerator",
        _StructuredFactory,
    )
    monkeypatch.setattr(_PROBE, "create_production_queued_runner", queue_factory)
    stack = _PROBE.create_live_stack(
        api_key="offline-placeholder",
        queue_sink=lambda outcome: None,
    )

    assert calls["openrouter"] == {
        "api_key": "offline-placeholder",
        "model_name": _PROBE.MODEL,
        "max_connections": 4,
        "timeout_seconds": 60.0,
        "provider_options": {"only": ["streamlake"]},
        "app_title": "AgentEvolve AAAI 2027 v6 engineering probe",
    }
    queue = calls["queue"]
    assert queue["max_in_flight"] == 4
    assert queue["max_pending"] == 8
    assert queue["max_attempts"] == 2
    assert queue["attempt_timeout_ns"] == 60_000_000_000
    assert queue["base_backoff_ns"] == 1_000_000_000
    assert queue["max_backoff_ns"] == 8_000_000_000
    assert queue["close_generator"] is True
    assert (
        queue["outcome_publication_policy"] is _PROBE.OutcomePublicationPolicy.REQUIRED
    )
    assert type(queue["attempt_request_policy"]) is _PROBE.SchemaRepairAttemptPolicy
    assert queue["jitter_policy"] == _PROBE.DeterministicHashJitter(
        seed=_PROBE.JITTER_SEED,
        domain=_PROBE.JITTER_DOMAIN,
    )
    assert stack.telemetry_policy == _PROBE.telemetry_policy()
    assert stack.telemetry_policy.requested_model == "deepseek/deepseek-v4-pro"
    assert stack.telemetry_policy.allowed_resolved_models == (
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-20260423",
    )
    assert stack.telemetry_policy.allowed_resolved_providers == ("StreamLake",)
    assert str(stack.telemetry_policy.max_cost_usd) == "0.010"

from __future__ import annotations

import asyncio
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import pytest

from agent_evolve.application.llm_task_queue import AsyncLLMTaskQueue
from agent_evolve.infrastructure.asyncio_runtime import AsyncioRuntime
from agent_evolve.infrastructure.clock import SystemClock
from agent_evolve.integrations.pydantic_ai.action_forecast import (
    plan_action_forecast_request,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (
    ProgressAwareRetryMode,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    OutcomePublicationPolicy,
    QueuedStructuredGenerationRunner,
    StructuredGenerationExecutor,
    TransportOnlyStructuredGenerationRetryClassifier,
)
from agent_evolve.policies.llm_backoff import ExponentialBackoff, NoJitter
from agent_evolve.ports.structured_generator import (
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development import run_airfoil_v7_two_stage_generation as launcher
from examples.development.airfoil_v7_two_stage_agent_evolution import (
    PreparedAirfoilTwoStageGeneration,
    prepare_airfoil_v7_two_stage_generation,
)
from examples.development.run_airfoil_v7_two_stage_generation import (
    CANONICAL_MODEL,
    FROZEN_LIVE_RUN_ID,
    MAX_ATTEMPTS,
    MAX_OUTPUT_TOKENS,
    MODEL,
    OPAQUE_FORECAST_CALL_IDS,
    RESOLVED_PROVIDER,
    AirfoilTwoStageRunError,
    FirewalledAirfoilPreparation,
    LiveDependencies,
    build_config,
    build_provider_free_preview,
    claim_live_run,
    execute_readiness,
    execute_live,
    finalize_claimed_live_abort,
    verify_readiness_gate,
)


@pytest.fixture(scope="module")
def preparation():
    return prepare_airfoil_v7_two_stage_generation()


@pytest.fixture(scope="module")
def bundle(preparation):
    return FirewalledAirfoilPreparation(
        preparation=preparation,
        predecision_firewall_record=preparation.evaluator.firewall_record(),
    )


def _junit(path: Path) -> Path:
    path.write_text(
        '<testsuite name="focused" tests="1" failures="0" errors="0" '
        'skipped="0"><testcase classname="offline" name="passed"/></testsuite>',
        encoding="ascii",
    )
    return path


def _focused_gate(path: Path) -> Path:
    path.mkdir()
    junit = _junit(path / "focused_tests.junit.xml")
    launcher.write_bytes_atomic(path / "pytest.stdout", b"")
    launcher.write_bytes_atomic(path / "pytest.stderr", b"")
    source = launcher.current_source_identity()
    record = launcher._focused_test_execution_record(
        run_dir=path,
        junit_path=junit,
        source_before=source,
        source_after=source,
        return_code=0,
        stdout=b"",
        stderr=b"",
    )
    launcher.write_json_atomic(path / "focused_test_execution.json", record)
    launcher.finalize_run_directory(path, status="focused_tests_passed")
    return path


class _SuccessfulTypedGenerator:
    def __init__(
        self,
        preparation: PreparedAirfoilTwoStageGeneration,
        progress_sink,
        run_dir: Path,
        state: dict[str, object],
    ) -> None:
        self._reflection_fixture = launcher._PreviewLowLevelRunner(preparation)
        self._progress_sink = progress_sink
        self._run_dir = run_dir
        self._state = state
        self._forecast_wave = asyncio.Event()
        self._lock = asyncio.Lock()
        self._active_forecasts = 0
        self._forecast_starts = 0

    @staticmethod
    def _forecast_value(request):
        batch_type = request.output_type
        payload = launcher._schema_driven_action_forecast_payload(batch_type)
        return batch_type.model_validate(payload)

    def _terminal_progress(self, request) -> None:
        attempt_id = request.provider_attempt_id
        assert attempt_id is not None
        digest = hashlib.sha256(b"").hexdigest()
        for sequence, kind in enumerate(
            (
                StructuredStreamProgressKind.OUTPUT_SELECTED,
                StructuredStreamProgressKind.STREAM_COMPLETED,
            ),
            start=1,
        ):
            self._progress_sink(
                StructuredStreamProgress(
                    call_id=request.call_id.value,
                    sequence=sequence,
                    kind=kind,
                    channel=StructuredStreamChannel.OTHER,
                    elapsed_ns=sequence,
                    event_content_utf8_bytes=0,
                    cumulative_content_utf8_bytes=0,
                    rolling_content_sha256=digest,
                    provider_attempt_id=attempt_id.value,
                )
            )

    async def generate_once(self, request):
        is_forecast = request.output_tool_name == "forecast_all_actions"
        if is_forecast:
            assert (self._run_dir / "planned_forecast_wave.json").is_file()
            assert len(
                (self._run_dir / "planned_calls.jsonl").read_text().splitlines()
            ) == 4
            async with self._lock:
                self._forecast_starts += 1
                self._active_forecasts += 1
                self._state["max_concurrent_forecasts"] = max(
                    int(self._state.get("max_concurrent_forecasts", 0)),
                    self._active_forecasts,
                )
                if self._forecast_starts == 3:
                    self._forecast_wave.set()
            await asyncio.wait_for(self._forecast_wave.wait(), timeout=5.0)
            try:
                value = self._forecast_value(request)
            finally:
                async with self._lock:
                    self._active_forecasts -= 1
        else:
            offline = await self._reflection_fixture(request)
            value = offline.value
        self._terminal_progress(request)
        return StructuredGenerationResponse(
            value=value,
            requested_model=MODEL,
            resolved_model=CANONICAL_MODEL,
            resolved_provider=RESOLVED_PROVIDER,
            provider_response_id=f"fake-{request.call_id.value}",
            finish_reason="tool_call",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.001"),
            latency_ns=1_000,
        )


class _SuccessfulRunnerFactory:
    def __init__(
        self,
        preparation: PreparedAirfoilTwoStageGeneration,
        run_dir: Path,
        state: dict[str, object],
    ) -> None:
        self._preparation = preparation
        self._run_dir = run_dir
        self._state = state

    def __call__(self, *, api_key, config, progress_sink, outcome_sink):
        assert api_key == "provider-free-test-key"
        assert config.to_manifest_record() == build_config().to_manifest_record()
        generator = _SuccessfulTypedGenerator(
            self._preparation,
            progress_sink,
            self._run_dir,
            self._state,
        )
        queue = AsyncLLMTaskQueue(
            executor=StructuredGenerationExecutor(generator),
            retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
            backoff_policy=ExponentialBackoff(0, 0, NoJitter()),
            clock=SystemClock(),
            max_in_flight=3,
            max_pending=3,
            attempt_timeout_ns=None,
            runtime=AsyncioRuntime(),
        )
        return QueuedStructuredGenerationRunner(
            queue=queue,
            max_attempts=2,
            outcome_sink=outcome_sink,
            outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        )


def test_focused_test_gate_runs_exact_credential_free_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    original_run = launcher.subprocess.run

    def fake_run(command, *args, **kwargs):
        if not any(
            type(value) is str and value.startswith("--junitxml=")
            for value in command
        ):
            return original_run(command, *args, **kwargs)
        observed.update(
            command=command,
            cwd=kwargs["cwd"],
            env=kwargs["env"],
            capture_output=kwargs["capture_output"],
            check=kwargs["check"],
        )
        junit_argument = next(
            value for value in command if value.startswith("--junitxml=")
        )
        _junit(Path(junit_argument.split("=", 1)[1]))
        return launcher.subprocess.CompletedProcess(
            command,
            0,
            stdout=b"focused pass\n",
            stderr=b"",
        )

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    gate = tmp_path / "focused_release_gate"
    result = launcher.execute_focused_test_gate(run_dir=gate)
    verified, finalization = launcher.verify_focused_test_gate(gate)

    assert result["execution"] == verified
    assert finalization["status"] == "focused_tests_passed"
    assert (gate / "pytest.stdout").read_bytes() == b"focused pass\n"
    assert (gate / "pytest.stderr").read_bytes() == b""
    assert observed["cwd"] == launcher.AGENT_EVOLVE_ROOT
    assert observed["capture_output"] is True
    assert observed["check"] is False
    environment = observed["env"]
    assert "OPENROUTER_API_KEY" not in environment
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_focused_test_command_preserves_import_capable_venv_executable(
    tmp_path: Path,
) -> None:
    command = launcher._focused_test_command(tmp_path / "focused.junit.xml")
    expected_executable = launcher.os.path.abspath(launcher.sys.executable)
    assert command[0] == expected_executable
    assert launcher._live_launch_command(("live", "--run-dir", "target"))[0] == (
        expected_executable
    )
    assert launcher.runtime_identity()["python_executable"] == expected_executable
    assert launcher.os.path.samefile(command[0], launcher.sys.executable)

    probe = launcher.subprocess.run(
        [command[0], "-c", "import pydantic_ai, pytest; print('venv-imports-ok')"],
        cwd=launcher.AGENT_EVOLVE_ROOT,
        env=launcher._focused_test_environment(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")
    assert probe.stdout == b"venv-imports-ok\n"


def test_scientific_transport_configuration_is_exact_and_transport_only() -> None:
    config = build_config()
    record = config.to_manifest_record()

    assert config.retry_mode is ProgressAwareRetryMode.TRANSPORT_ONLY
    assert config.reasoning_config is not None
    assert config.reasoning_config.to_model_setting() == {"effort": "high"}
    assert record["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert record["stream_liveness"] == {
        "first_event_timeout_ns": 180_000_000_000,
        "idle_timeout_ns": 120_000_000_000,
        "absolute_timeout_ns": None,
        "cleanup": {
            "policy_id": "bounded_cancel_drain",
            "policy_version": 1,
            "definition_sha256": (
                "7a5543befb8b6a1fcaf5388aac39747ec5e7e17c228a4a643b6874033e74329c"
            ),
            "configuration_sha256": (
                "aa4661cbdc1c8acae6ff2b6565b17d54bdd9d5b52b84bc4699061181227dac78"
            ),
            "cancel_drain_timeout_ns": 5_000_000_000,
            "transport_retire_timeout_ns": 5_000_000_000,
        },
    }
    assert record["queue"]["max_in_flight"] == 3
    assert record["queue"]["max_attempts"] == MAX_ATTEMPTS == 2
    assert record["queue"]["attempt_timeout_ns"] is None
    assert record["queue"]["attempt_request_policy"] == "exact_payload"
    assert record["queue"]["retry_classifier"] == "transport_only"
    assert record["queue"]["backoff"] == {
        "kind": "exponential_deterministic_task_keyed_full_jitter",
        "base_backoff_ns": 1_000_000_000,
        "max_backoff_ns": 30_000_000_000,
        "jitter_seed": 2_026_071_501,
        "jitter_domain": "airfoil-v7-generic-two-stage-v2",
    }


def test_provider_free_preview_builds_exact_four_call_schemas(preparation) -> None:
    preview = build_provider_free_preview(preparation)

    assert len(preview.structured_calls) == 4
    assert preview.structured_calls[0]["output_tool_name"] == (
        "return_reflection_insights"
    )
    assert [value["output_tool_name"] for value in preview.structured_calls[1:]] == [
        "forecast_all_actions",
        "forecast_all_actions",
        "forecast_all_actions",
    ]
    assert {value["max_output_tokens"] for value in preview.structured_calls} == {
        MAX_OUTPUT_TOKENS
    }
    assert len(preview.reflection.shards) == 8
    assert len(preview.arms.source_cards) == len(preview.arms.placebo_cards) == 8
    assert len(preview.arm_requests) == 3
    assert tuple(value.call_id for value in preview.arm_requests) == tuple(
        OPAQUE_FORECAST_CALL_IDS[arm]
        for arm in (
            preview.arms.memory_receipt.arm,
            preview.arms.placebo_receipt.arm,
            # N has no receipt; its precommitted control-plane mapping is last.
            next(
                arm
                for arm in OPAQUE_FORECAST_CALL_IDS
                if arm.value == "n"
            ),
        )
    )
    for request in preview.arm_requests:
        lowered = request.call_id.value.casefold()
        assert "memory" not in lowered
        assert "placebo" not in lowered
        assert "neutral" not in lowered
        assert not lowered.endswith(("_m", "_p", "_n"))

    forecast_plans = tuple(
        plan_action_forecast_request(request) for request in preview.arm_requests
    )
    launcher._validate_provider_boundary_blinding(forecast_plans)
    forbidden = (
        "permuted_placebo",
        "control_arm",
        "portfolioexperimentalarm",
        '"arm"',
    )
    for plan in forecast_plans:
        visible = "\n".join(
            (
                plan.call_id.value,
                plan.operation,
                plan.output_tool_name,
                plan.prompt,
                json.dumps(
                    plan.output_type.model_json_schema(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        ).casefold()
        assert not any(token in visible for token in forbidden)
    assert forecast_plans[0].operation == forecast_plans[1].operation
    assert forecast_plans[0].output_tool_name == forecast_plans[1].output_tool_name
    assert (
        forecast_plans[0].output_type.model_json_schema()
        == forecast_plans[1].output_type.model_json_schema()
    )


def test_readiness_creates_finalized_zero_provider_release(
    tmp_path: Path,
    bundle: FirewalledAirfoilPreparation,
) -> None:
    run_dir = tmp_path / "ae7_readiness_fixture"
    target = tmp_path / FROZEN_LIVE_RUN_ID

    result = execute_readiness(
        run_dir=run_dir,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    readiness = result["readiness"]

    assert readiness["status"] == "ready_provider_not_called"
    assert readiness["credential_read_attempted"] is False
    assert readiness["credentials_read"] is False
    assert readiness["client_constructed"] is False
    assert readiness["provider_call_attempted"] is False
    assert readiness["call_preview"]["logical_call_count"] == 4
    assert len((run_dir / "planned_calls.jsonl").read_text().splitlines()) == 4
    assert (run_dir / "stream_progress.jsonl").read_bytes() == b""
    assert (run_dir / "queue_outcomes.jsonl").read_bytes() == b""
    assert json.loads((run_dir / "finalized.json").read_text())[
        "status"
    ] == "ready_provider_not_called"

    verified, finalization = verify_readiness_gate(
        run_dir,
        target_live_run_dir=target,
    )
    assert verified == readiness
    assert finalization["status"] == "ready_provider_not_called"


def test_live_gate_rejects_existing_target_before_any_provider_boundary(
    tmp_path: Path,
    bundle: FirewalledAirfoilPreparation,
) -> None:
    gate = tmp_path / "ae7_readiness_existing_target"
    target = tmp_path / FROZEN_LIVE_RUN_ID
    execute_readiness(
        run_dir=gate,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    target.mkdir()

    with pytest.raises(AirfoilTwoStageRunError, match="not current"):
        verify_readiness_gate(gate, target_live_run_dir=target)


def test_live_claim_rejects_fresh_preparation_that_differs_from_gate(
    tmp_path: Path,
    bundle: FirewalledAirfoilPreparation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "ae7_readiness_bundle_swap"
    target = tmp_path / FROZEN_LIVE_RUN_ID
    execute_readiness(
        run_dir=gate,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    original = PreparedAirfoilTwoStageGeneration.to_record

    def swapped_record(self):
        record = original(self)
        return {**record, "oracle_swap_probe": True}

    monkeypatch.setattr(PreparedAirfoilTwoStageGeneration, "to_record", swapped_record)
    with pytest.raises(AirfoilTwoStageRunError, match="differs from readiness"):
        claim_live_run(
            run_dir=target,
            release_gate_dir=gate,
            launch_command=("python", "launcher.py", "live"),
            bundle=bundle,
        )
    assert not target.exists()


def test_precredential_failure_finalizes_while_writer_lock_is_held(
    tmp_path: Path,
    bundle: FirewalledAirfoilPreparation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "ae7_readiness_predispatch_abort"
    target = tmp_path / FROZEN_LIVE_RUN_ID
    execute_readiness(
        run_dir=gate,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    claim = claim_live_run(
        run_dir=target,
        release_gate_dir=gate,
        launch_command=("python", "launcher.py", "live"),
        bundle=bundle,
    )
    original_finalize = launcher.finalize_run_directory
    observed = {"lock_held": False}

    def finalize_with_lock_probe(path: Path, *, status: str):
        with (path / "writer.lock").open("rb") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed["lock_held"] = True
        return original_finalize(path, status=status)

    monkeypatch.setattr(launcher, "finalize_run_directory", finalize_with_lock_probe)
    terminal = finalize_claimed_live_abort(
        claim=claim,
        error=AirfoilTwoStageRunError("sanitized credential failure"),
        stage="credential_load",
        credential_read_attempted=True,
        credentials_read=False,
    )
    assert observed["lock_held"] is True
    assert claim.active is False
    assert terminal["result"]["status"] == "pre_dispatch_infrastructure_abort"
    assert terminal["result"]["provider_call_attempted"] is False
    assert terminal["result"]["client_constructed"] is False
    assert json.loads((target / "finalized.json").read_text())["status"] == (
        "pre_dispatch_infrastructure_abort"
    )


@pytest.mark.parametrize("drift_kind", ("source", "runtime"))
def test_claim_gap_identity_drift_and_one_shot_write_fault_are_sealed(
    tmp_path: Path,
    bundle: FirewalledAirfoilPreparation,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    gate = tmp_path / "ae7_readiness_claim_gap"
    target = tmp_path / FROZEN_LIVE_RUN_ID
    execute_readiness(
        run_dir=gate,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    original_source_identity = launcher.current_source_identity
    original_runtime_identity = launcher.runtime_identity
    original_write_json_atomic = launcher.write_json_atomic
    state = {"drift": False, "write_fault_armed": False, "write_fault_fired": False}

    def source_identity_with_post_lock_drift():
        identity = original_source_identity()
        if not state["drift"] or drift_kind != "source":
            return identity
        return {**identity, "aggregate_sha256": "f" * 64}

    def runtime_identity_with_post_lock_drift():
        identity = original_runtime_identity()
        if not state["drift"] or drift_kind != "runtime":
            return identity
        return {**identity, "python_version": "0.0-injected-drift"}

    def one_shot_post_lock_write_fault(path: Path, value: object):
        if state["write_fault_armed"] and not state["write_fault_fired"]:
            state["write_fault_fired"] = True
            raise OSError("injected sanitized post-lock write failure")
        return original_write_json_atomic(path, value)

    def open_claim_gap() -> None:
        state["drift"] = True
        state["write_fault_armed"] = True

    monkeypatch.setattr(
        launcher,
        "current_source_identity",
        source_identity_with_post_lock_drift,
    )
    monkeypatch.setattr(
        launcher,
        "runtime_identity",
        runtime_identity_with_post_lock_drift,
    )
    monkeypatch.setattr(launcher, "write_json_atomic", one_shot_post_lock_write_fault)
    with pytest.raises(AirfoilTwoStageRunError, match="sealed target artifacts"):
        claim_live_run(
            run_dir=target,
            release_gate_dir=gate,
            launch_command=("python", "launcher.py", "live"),
            bundle=bundle,
            post_lock_hook=open_claim_gap,
        )

    result = json.loads((target / "result.json").read_text())
    finalization = json.loads((target / "finalized.json").read_text())
    assert state["write_fault_fired"] is True
    assert result["status"] == "pre_dispatch_infrastructure_abort"
    assert result["credential_read_attempted"] is False
    assert result["credentials_read"] is False
    assert result["client_constructed"] is False
    assert result["provider_call_attempted"] is False
    assert finalization["status"] == "pre_dispatch_infrastructure_abort"
    with (target / "writer.lock").open("rb") as probe:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)


@pytest.mark.parametrize("drift_kind", ("source", "runtime"))
def test_post_claim_identity_drift_seals_before_runner_or_provider_planning(
    tmp_path: Path,
    bundle: FirewalledAirfoilPreparation,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    gate = tmp_path / f"ae7_readiness_post_claim_{drift_kind}"
    target = tmp_path / FROZEN_LIVE_RUN_ID
    execute_readiness(
        run_dir=gate,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    claim = claim_live_run(
        run_dir=target,
        release_gate_dir=gate,
        launch_command=("python", "launcher.py", "live"),
        bundle=bundle,
    )
    factory_calls = 0

    def forbidden_runner_factory(**_kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("runner construction crossed the identity barrier")

    if drift_kind == "source":
        original_source_identity = launcher.current_source_identity

        def drifted_source_identity():
            return {
                **original_source_identity(),
                "aggregate_sha256": "e" * 64,
            }

        monkeypatch.setattr(
            launcher,
            "current_source_identity",
            drifted_source_identity,
        )
    else:
        original_runtime_identity = launcher.runtime_identity

        def drifted_runtime_identity():
            return {
                **original_runtime_identity(),
                "python_version": "0.0-injected-post-claim-drift",
            }

        monkeypatch.setattr(
            launcher,
            "runtime_identity",
            drifted_runtime_identity,
        )

    with pytest.raises(AirfoilTwoStageRunError, match="inspect finalized artifacts"):
        execute_live(
            claim=claim,
            bundle=bundle,
            api_key="provider-free-test-key",
            dependencies=LiveDependencies(runner_factory=forbidden_runner_factory),
        )

    result = json.loads((target / "result.json").read_text())
    finalization = json.loads((target / "finalized.json").read_text())
    assert factory_calls == 0
    assert claim.active is False
    assert result["status"] == "pre_dispatch_infrastructure_abort"
    assert result["client_constructed"] is False
    assert result["provider_call_attempted"] is False
    assert result["planned_logical_call_count"] == 0
    assert finalization["status"] == "pre_dispatch_infrastructure_abort"
    assert not (target / "manifest.json").exists()
    assert not (target / "planned_calls.jsonl").exists()


def test_runner_marker_write_failure_closes_entered_runner_and_seals(
    tmp_path: Path,
    bundle: FirewalledAirfoilPreparation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "ae7_readiness_runner_marker_fault"
    target = tmp_path / FROZEN_LIVE_RUN_ID
    execute_readiness(
        run_dir=gate,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    claim = claim_live_run(
        run_dir=target,
        release_gate_dir=gate,
        launch_command=("python", "launcher.py", "live"),
        bundle=bundle,
    )
    state = {"factory": 0, "entered": 0, "exited": 0, "snapshots": 0, "calls": 0}

    class LifecycleProbeRunner:
        async def __aenter__(self):
            state["entered"] += 1
            return self

        async def __aexit__(self, *_args):
            state["exited"] += 1

        async def snapshot(self):
            state["snapshots"] += 1
            raise AssertionError("snapshot crossed the runner marker barrier")

        async def __call__(self, _request):
            state["calls"] += 1
            raise AssertionError("provider planning crossed the runner marker barrier")

    def runner_factory(**_kwargs):
        state["factory"] += 1
        return LifecycleProbeRunner()

    original_write_json_atomic = launcher.write_json_atomic

    def fail_runner_marker(path: Path, value: object):
        if path.name == "runner_constructed.json":
            raise OSError("injected runner marker write failure")
        return original_write_json_atomic(path, value)

    monkeypatch.setattr(launcher, "write_json_atomic", fail_runner_marker)
    with pytest.raises(AirfoilTwoStageRunError, match="inspect finalized artifacts"):
        execute_live(
            claim=claim,
            bundle=bundle,
            api_key="provider-free-test-key",
            dependencies=LiveDependencies(runner_factory=runner_factory),
        )

    result = json.loads((target / "result.json").read_text())
    finalization = json.loads((target / "finalized.json").read_text())
    assert state == {
        "factory": 1,
        "entered": 1,
        "exited": 1,
        "snapshots": 0,
        "calls": 0,
    }
    assert claim.active is False
    assert result["status"] == "pre_dispatch_infrastructure_abort"
    assert result["provider_call_attempted"] is False
    assert result["planned_logical_call_count"] == 0
    assert finalization["status"] == "pre_dispatch_infrastructure_abort"


def test_injected_live_run_executes_one_plus_three_and_finalizes_exact_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = prepare_airfoil_v7_two_stage_generation()
    bundle = FirewalledAirfoilPreparation(
        preparation=preparation,
        predecision_firewall_record=preparation.evaluator.firewall_record(),
    )
    gate = tmp_path / "ae7_readiness_full_fake_live"
    target = tmp_path / FROZEN_LIVE_RUN_ID
    execute_readiness(
        run_dir=gate,
        target_live_run_dir=target,
        bundle=bundle,
        focused_test_gate_dir=_focused_gate(tmp_path / "focused-test-gate"),
    )
    claim = claim_live_run(
        run_dir=target,
        release_gate_dir=gate,
        launch_command=("python", "launcher.py", "live"),
        bundle=bundle,
    )
    state: dict[str, object] = {}
    original_finalize = launcher.finalize_run_directory

    def finalize_with_lock_probe(path: Path, *, status: str):
        with (path / "writer.lock").open("rb") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state["finalize_saw_lock"] = True
        return original_finalize(path, status=status)

    monkeypatch.setattr(launcher, "finalize_run_directory", finalize_with_lock_probe)
    completed = execute_live(
        claim=claim,
        bundle=bundle,
        api_key="provider-free-test-key",
        dependencies=LiveDependencies(
            runner_factory=_SuccessfulRunnerFactory(preparation, target, state)
        ),
    )

    result = completed["result"]
    assert result["status"] == "completed_four_call_development_generation"
    assert result["provider_call_attempted"] is True
    assert result["planned_logical_call_count"] == 4
    assert result["terminal_queue_outcome_count"] == 4
    assert result["new_cfd_calls"] == 0
    assert result["terminal_ledger"] == {
        "logical_terminal_count": 4,
        "physical_attempt_count": 4,
        "scheduled_retry_count": 0,
        "extra_physical_attempt_count": 0,
        "scheduled_backoff_ns": [],
        "terminal_statuses": ["succeeded"] * 4,
    }
    assert state["max_concurrent_forecasts"] == 3
    assert state["finalize_saw_lock"] is True
    assert claim.active is False

    planned = [json.loads(line) for line in (target / "planned_calls.jsonl").read_text().splitlines()]
    assert len(planned) == 4
    assert planned[0]["mode"] == "live_exact_reflection_call"
    assert {row["mode"] for row in planned[1:]} == {
        "live_precommitted_forecast_wave"
    }
    outcomes = [json.loads(line) for line in (target / "queue_outcomes.jsonl").read_text().splitlines()]
    assert len(outcomes) == 4
    assert all(row["status"] == "succeeded" for row in outcomes)
    assert all(
        attempt["request_evidence"]["variant"] == "original"
        for row in outcomes
        for attempt in row["attempts"]
    )
    phases = [json.loads(line) for line in (target / "phase_commits.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in phases] == ["forecast", "allocate", "evaluate"]
    assert (target / "preallocation_terminal_barrier.json").is_file()
    assert (target / "g2_allocation_commitment.json").is_file()
    assert json.loads((target / "finalized.json").read_text())["status"] == (
        "completed_four_call_development_generation"
    )
    allocate_phase = json.loads((target / "phase_allocate.json").read_text())
    assert all(
        execution["decision"]["candidate_evaluations"] == 213
        for execution in allocate_phase["payload"]["arm_executions"]
    )

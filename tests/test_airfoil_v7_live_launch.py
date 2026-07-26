"""Provider/CFD-free proofs for the Airfoil-v7 live launch boundary."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evolve.agentic import (
    FileExclusiveResourceLease,
    StructuredOutputRequestKind,
)
from examples.benchmarks.engibench_airfoil import v7_launch as launch
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    AirfoilConvergenceEvaluationError,
    V2_EVALUATOR_ID,
)
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    compose_offline_experiment,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AirfoilV7DetailedEvaluationAdapter,
)


class _ExpectedStop(RuntimeError):
    pass


class _RunnerDouble:
    def __init__(self) -> None:
        self.enter_calls = 0

    async def __aenter__(self):
        self.enter_calls += 1
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


class _JournalDouble:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write(self, row: object) -> None:
        self.rows.append(dict(row))


def _lease_factory(
    lease_path: Path,
    leases: list[FileExclusiveResourceLease],
):
    def factory(run_id: str, phase: str) -> FileExclusiveResourceLease:
        lease = FileExclusiveResourceLease(
            resource_key="airfoil_v7_test_resource",
            owner_id=run_id,
            lease_path=lease_path,
            owner_metadata={"phase": phase},
        )
        leases.append(lease)
        return lease

    return factory


def test_manifest_uses_host_global_lease_and_invoked_python_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-target"
    target.write_bytes(b"fixture")
    invoked = tmp_path / "venv-python"
    invoked.symlink_to(target)

    assert launch._invoked_absolute_path(invoked) == invoked.absolute()
    assert launch._invoked_absolute_path(invoked) != invoked.resolve()
    record = launch._resource_lease_manifest_record(
        phase="seed_qualification"
    )
    assert record["resource_key"] == "engibench_airfoil_machaero"
    assert record["lease_path"] == (
        "/tmp/agent_evolve_resource_locks/"
        "engibench_airfoil_machaero.lock"
    )
    assert record["acquisition"] == (
        "nonblocking_before_benchmark_construction"
    )


def test_live_identifier_namespace_is_safe_and_rejected_during_manifest_build() -> None:
    namespace = launch._live_id_namespace("ae7_live_0714_2156")
    assert namespace == "ae7_ae7_live_0714_2156"
    assert ":" not in namespace

    with pytest.raises(ValueError, match="namespace violates"):
        launch._live_id_namespace("x" * 46)


def test_streamlake_snapshots_mechanically_bind_max_output_and_cost_gates() -> None:
    binding = launch._streamlake_route_snapshot_binding()
    route = binding["selected_route"]
    assert route == {
        "requested_model": "deepseek/deepseek-v4-pro",
        "canonical_model": "deepseek/deepseek-v4-pro-20260423",
        "provider_name": "StreamLake",
        "provider_request_slug": "streamlake",
        "endpoint_tag": "streamlake/fp8",
        "quantization": "fp8",
        "context_length": 1_024_000,
        "max_completion_tokens": 384_000,
        "prompt_usd_per_token": "0.0000007134",
        "completion_usd_per_token": "0.0000014268",
        "input_cache_read_usd_per_token": "0.00000005945",
    }
    assert binding["pricing_snapshot"]["sha256"] == (
        "5adea5e08d7aea5eb89de010e1750890fe6b7f70a3f7fe733a08996d0b8b7204"
    )
    assert binding["capability_snapshot"]["sha256"] == (
        "131d0fef27cb24350f9c067ea7407cd9279ddbe242eef77e29451390a750a671"
    )
    envelope = launch._cost_envelope_record(logical_calls=7)
    assert envelope["derived_max_billable_attempt"] == "0.5765641728"
    assert envelope["derived_max_accepted_run"] == "4.0359492096"
    assert envelope["derived_max_potentially_billable_run"] == (
        "8.0718984192"
    )
    assert envelope["raw_attempt_cap"] == 14
    assert envelope["ceiling_semantics"] == (
        "worst_case_gate_not_expected_usage"
    )
    telemetry = launch._airfoil_v7_telemetry_policy()
    assert telemetry.max_output_tokens == 384_000
    assert telemetry.max_reasoning_tokens == 4_096
    assert telemetry.max_cost_usd == Decimal("0.5765641728")


def test_prompt_preflight_binds_request_kind_policy_and_provider_ceiling() -> None:
    for request_kind in StructuredOutputRequestKind:
        record = launch.prompt_preflight(
            "provider-free prompt fixture",
            max_output_tokens=384_000,
            request_kind=request_kind,
        )
        assert record["request_kind"] == request_kind.value
        assert record["max_output_tokens"] == 384_000
        assert record["provider_max_completion_tokens"] == 384_000
        assert record["provider_context_length"] == 1_024_000
        assert record["cap_plausible"] is True

    with pytest.raises(RuntimeError, match="structured-output budget"):
        launch.prompt_preflight(
            "provider-free prompt fixture",
            max_output_tokens=383_999,
            request_kind=StructuredOutputRequestKind.REFLECTION,
        )


def test_provider_accounting_uses_the_snapshot_derived_attempt_gate() -> None:
    telemetry = {
        "requested_model": "deepseek/deepseek-v4-pro",
        "resolved_model": "deepseek/deepseek-v4-pro-20260423",
        "resolved_provider": "StreamLake",
        "provider_response_id": "response-1",
        "finish_reason": "tool_call",
        "input_tokens": 32_000,
        "output_tokens": 384_000,
        "reasoning_tokens": 4_096,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": "0.5765641728",
        "latency_ns": 1,
        "attempt_count": 1,
    }
    accepted = (
        {
            "logical_call_ordinal": 1,
            "call_id": "call-1",
            "telemetry": telemetry,
        },
    )
    queue = (
        {
            "schema_version": 4,
            "task_id": "call-1",
            "status": "succeeded",
            "attempts": [{}],
            "response": dict(telemetry),
        },
    )
    record = launch._provider_accounting_record(
        accepted_responses=accepted,
        queue_outcomes=queue,
        expected_logical_calls=1,
        expected_accepted_responses=1,
        allowed_terminal_failures=0,
    )
    assert record["accepted_cost_usd"] == "0.5765641728"
    assert record["accepted_cost_cap_usd"] == "0.5765641728"
    assert record["potentially_billable_exposure_cap_usd"] == "1.1531283456"
    assert record["maximum_output_tokens"] == 384_000

    over_cap = {**telemetry, "cost_usd": "0.5765641729"}
    with pytest.raises(RuntimeError, match="telemetry violates frozen gates"):
        launch._provider_accounting_record(
            accepted_responses=(
                {
                    "logical_call_ordinal": 1,
                    "call_id": "call-1",
                    "telemetry": over_cap,
                },
            ),
            queue_outcomes=(
                {
                    "schema_version": 4,
                    "task_id": "call-1",
                    "status": "succeeded",
                    "attempts": [{}],
                    "response": dict(over_cap),
                },
            ),
            expected_logical_calls=1,
            expected_accepted_responses=1,
            allowed_terminal_failures=0,
        )


def test_seed_phase_holds_generic_lease_before_benchmark_and_releases_after_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "seed-double"
    run_dir = tmp_path / run_id
    manifest_path = tmp_path / "seed-manifest.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    verified = launch.VerifiedSeedQualificationManifest(
        path=manifest_path,
        record={"qualification": {}},
        run_id=run_id,
        output_dir=run_dir,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
    )
    monkeypatch.setattr(
        launch,
        "verify_seed_qualification_manifest",
        lambda *args, **kwargs: verified,
    )
    leases: list[FileExclusiveResourceLease] = []

    def benchmark_factory(actual_run_id: str, actual_run_dir: Path):
        assert actual_run_id == run_id
        assert actual_run_dir == run_dir
        assert len(leases) == 1 and leases[0].active
        assert (run_dir / "resource_lease_acquired.json").is_file()
        raise _ExpectedStop("stop before any evaluator construction")

    dependencies = launch.SeedQualificationDependencies(
        benchmark_factory=benchmark_factory,
        resource_lease_factory=_lease_factory(tmp_path / "seed.lock", leases),
        enforce_canonical_output=False,
    )
    with pytest.raises(_ExpectedStop, match="before any evaluator"):
        launch.execute_seed_qualification_with_dependencies(
            manifest_path,
            dependencies,
        )

    assert not leases[0].active
    released = json.loads(
        (run_dir / "resource_lease_released.json").read_text(encoding="utf-8")
    )
    assert released["phase"] == "seed_qualification"
    assert released["release"]["status"] == "released"
    assert released["release"]["outcome"] == "failed"
    finalized = json.loads(
        (run_dir / "finalized.json").read_text(encoding="utf-8")
    )
    assert finalized["status"] == "failed"
    assert "resource_lease_acquired.json" in finalized["files"]
    assert "resource_lease_released.json" in finalized["files"]


def test_seed_result_is_sealed_only_after_second_evaluation_source_reverify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "seed-seal-double"
    run_dir = tmp_path / run_id
    manifest_path = tmp_path / "seed-seal-manifest.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    qualification = launch._seed_qualification_spec(
        run_id=run_id,
        output_dir=run_dir,
        verification_report={},
    )
    verified = launch.VerifiedSeedQualificationManifest(
        path=manifest_path,
        record={"qualification": qualification},
        run_id=run_id,
        output_dir=run_dir,
        manifest_sha256="e" * 64,
        source_sha256="f" * 64,
    )
    offline = compose_offline_experiment(delay_seconds=0.0)
    evaluator = offline.evaluator
    reverify_observations: list[int] = []

    def verifier(*args, **kwargs):
        del args, kwargs
        reverify_observations.append(evaluator.calls)
        return verified

    monkeypatch.setattr(launch, "verify_seed_qualification_manifest", verifier)

    def locate(actual_run_dir: Path, configuration: object) -> Path:
        path = actual_run_dir / "raw_receipts" / f"{candidate_sha256(configuration)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="ascii")
        return path

    monkeypatch.setattr(
        launch,
        "_verify_raw_success_receipt",
        lambda path, configuration: {
            "path": str(path),
            "sha256": candidate_sha256(configuration),
            "bytes": path.stat().st_size,
            "record": {},
            "evaluator_calls": 3,
        },
    )
    leases: list[FileExclusiveResourceLease] = []
    result = launch.execute_seed_qualification_with_dependencies(
        manifest_path,
        launch.SeedQualificationDependencies(
            benchmark_factory=lambda actual_run_id, actual_run_dir: (
                offline.planner.benchmark
            ),
            raw_receipt_locator=locate,
            resource_lease_factory=_lease_factory(
                tmp_path / "seed-seal.lock",
                leases,
            ),
            enforce_canonical_output=False,
        ),
    )

    assert result["status"] == "qualified"
    assert evaluator.calls == 2
    assert reverify_observations == [0, 0, 1, 2]
    source_rows = [
        json.loads(line)
        for line in (run_dir / "source_verifications.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert source_rows[-1]["stage"] == "post_seed_cfd_2_pre_result_seal"


def test_seed_failure_is_typed_sealed_and_stops_with_exact_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "seed-typed-failure"
    run_dir = tmp_path / run_id
    manifest_path = tmp_path / "seed-typed-failure-manifest.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    qualification = launch._seed_qualification_spec(
        run_id=run_id,
        output_dir=run_dir,
        verification_report={},
    )
    verified = launch.VerifiedSeedQualificationManifest(
        path=manifest_path,
        record={"qualification": qualification},
        run_id=run_id,
        output_dir=run_dir,
        manifest_sha256="c" * 64,
        source_sha256="d" * 64,
    )
    monkeypatch.setattr(
        launch,
        "verify_seed_qualification_manifest",
        lambda *args, **kwargs: verified,
    )

    class RawFailure:
        calls = 0

        def evaluate_raw(self, configuration: object):
            self.calls += 1
            record = {
                "schema_version": 2,
                "evaluator_id": V2_EVALUATOR_ID,
                "mode": "evaluate",
                "status": "infrastructure_or_evaluator_failure",
                "candidate_sha256": candidate_sha256(configuration),
                "evaluator_calls": 1,
                "failed_point_index": 0,
                "failure": {
                    "type": "WitnessBoundaryFailure",
                    "message": "instrumentation contract failed",
                },
            }
            path = (
                run_dir
                / "raw_receipts"
                / "failure"
                / f"{candidate_sha256(configuration)}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            raise AirfoilConvergenceEvaluationError(
                "instrumentation contract failed",
                candidate_invalid=False,
                record_path=path,
                record=record,
            )

    raw = RawFailure()
    leases: list[FileExclusiveResourceLease] = []
    with pytest.raises(
        RuntimeError,
        match="stopped on authenticated typed failure",
    ):
        launch.execute_seed_qualification_with_dependencies(
            manifest_path,
            launch.SeedQualificationDependencies(
                benchmark_factory=lambda actual_run_id, actual_run_dir: SimpleNamespace(
                    detailed_evaluator=AirfoilV7DetailedEvaluationAdapter(raw)
                ),
                resource_lease_factory=_lease_factory(
                    tmp_path / "seed-typed-failure.lock",
                    leases,
                ),
                enforce_canonical_output=False,
            ),
        )

    assert raw.calls == 1
    result = json.loads(
        (run_dir / "qualification_result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "invalidated"
    assert result["authorized_cfd_candidate_evaluations"] == 2
    assert result["cfd_candidate_evaluations"] == 1
    assert result["authorized_raw_solver_calls"] == 6
    assert result["raw_solver_calls"] == 1
    assert result["provider_io_performed"] is False
    assert result["credentials_read"] is False
    assert len(result["seeds"]) == 1
    assert result["seeds"][0]["payload"]["failure"] == {
        "category": "system",
        "code": "evaluator_contract_violation",
        "message": "instrumentation contract failed",
        "retryable": False,
        "exception_type": "AirfoilConvergenceEvaluationError",
    }
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["cfd_candidate_evaluations"] == 1
    assert failure["raw_solver_calls"] == 1
    source_rows = [
        json.loads(line)
        for line in (run_dir / "source_verifications.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["stage"] for row in source_rows] == [
        "pre_seed_cfd_1",
        "post_seed_cfd_1_failure_seal",
    ]


def test_provider_phase_holds_generic_lease_before_benchmark_or_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "provider-double"
    run_dir = tmp_path / run_id
    manifest_path = tmp_path / "provider-manifest.json"
    manifest_path.write_text("{}\n", encoding="ascii")
    telemetry_policy = launch._airfoil_v7_telemetry_policy()
    verified = launch.VerifiedLaunchManifest(
        path=manifest_path,
        record={
            "launch": {
                "seed_qualification": {},
                "model_route": {
                    "telemetry_policy": telemetry_policy.to_trace_record(),
                    "telemetry_policy_sha256": telemetry_policy.policy_sha256,
                },
            }
        },
        run_id=run_id,
        output_dir=run_dir,
        manifest_sha256="c" * 64,
        source_sha256="d" * 64,
    )
    monkeypatch.setattr(
        launch,
        "verify_launch_manifest",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        launch,
        "reverify_launch_source",
        lambda value: {
            "schema_version": 1,
            "manifest_sha256": value.manifest_sha256,
            "source_sha256": value.source_sha256,
        },
    )
    monkeypatch.setattr(
        launch,
        "_copy_bound_seed_receipts",
        lambda *args, **kwargs: {},
    )
    leases: list[FileExclusiveResourceLease] = []
    credentials_read = 0
    stack_factories = 0

    def benchmark_factory(
        actual_run_id: str,
        actual_run_dir: Path,
        seed_binding: object,
    ):
        assert actual_run_id == run_id
        assert actual_run_dir == run_dir
        assert seed_binding == {}
        assert len(leases) == 1 and leases[0].active
        assert (run_dir / "resource_lease_acquired.json").is_file()
        raise _ExpectedStop("stop before benchmark or provider construction")

    def credential_loader() -> str:
        nonlocal credentials_read
        credentials_read += 1
        return "must-not-be-read"

    def stack_factory(api_key: str, sink: object):
        del api_key, sink
        nonlocal stack_factories
        stack_factories += 1
        raise AssertionError("provider stack must not be constructed")

    dependencies = launch.LiveExecutionDependencies(
        benchmark_factory=benchmark_factory,
        credential_loader=credential_loader,
        stack_factory=stack_factory,
        resource_lease_factory=_lease_factory(
            tmp_path / "provider.lock",
            leases,
        ),
        enforce_canonical_output=False,
    )
    with pytest.raises(_ExpectedStop, match="before benchmark or provider"):
        launch.execute_live_with_dependencies(manifest_path, dependencies)

    assert credentials_read == 0
    assert stack_factories == 0
    assert not leases[0].active
    released = json.loads(
        (run_dir / "resource_lease_released.json").read_text(encoding="utf-8")
    )
    assert released["phase"] == "provider_evolution"
    assert released["release"]["outcome"] == "failed"
    finalized = json.loads(
        (run_dir / "finalized.json").read_text(encoding="utf-8")
    )
    assert finalized["status"] == "failed"
    assert finalized["post_finalization_decision"] == "invalid_block"


def test_stack_policy_drift_is_rejected_before_runner_entry_or_dispatch() -> None:
    expected = launch._airfoil_v7_telemetry_policy()
    mismatched = replace(expected, max_output_tokens=expected.max_output_tokens + 1)
    runner = _RunnerDouble()
    journal = _JournalDouble()
    credentials_read = 0

    def credential_loader() -> str:
        nonlocal credentials_read
        credentials_read += 1
        return "provider-free-fixture-key"

    generator = launch.DeferredJournaledLiveGenerator(
        credential_loader=credential_loader,
        stack_factory=lambda key: SimpleNamespace(
            runner=runner,
            generator=object(),
            telemetry_policy=mismatched,
        ),
        pre_provider_verifier=lambda stage: {"stage": stage},
        journal=journal,  # type: ignore[arg-type]
        expected_telemetry_policy=expected.to_trace_record(),
        expected_telemetry_policy_sha256=expected.policy_sha256,
    )

    with pytest.raises(RuntimeError, match="drifted from launch manifest"):
        asyncio.run(generator._ensure_stack())

    assert credentials_read == 1
    assert runner.enter_calls == 0
    assert journal.rows == []

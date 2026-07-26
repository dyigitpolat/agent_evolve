"""Canonical, single-use launcher for the sealed Airfoil-v7 G3 screen.

``prepare-freeze`` and ``prepare-manifest`` are provider/CFD/credential free.
``run`` is the only costly boundary: it derives its output directory from the
authenticated manifest, acquires the host-global Airfoil lease before
benchmark construction, and durably journals every provider outcome and
engine trace.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    structured_generation_outcome_record,
)
from agent_evolve.ports.resource_lease import ExclusiveResourceLease  # noqa: E402
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredStreamProgress,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (  # noqa: E402
    ConvergenceQualifiedAirfoilPanelProblem,
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (  # noqa: E402
    EXPECTED_DATASET_SHA256,
)
from examples.benchmarks.engibench_airfoil.v7_g3_analysis import (  # noqa: E402
    analyze_airfoil_g3_live_result,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (  # noqa: E402
    AIRFOIL_G3_PROVIDER_PROFILES,
    CONTAINER_IMAGE,
    DEEPSEEK_G3_PROVIDER_PROFILE,
    DEFAULT_RESOURCE_LEASE_PATH,
    G3_RUN_ROOT,
    G3_WORK_ROOT,
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
    RESEARCH_ARTIFACT_ROOT,
    AirfoilG3LiveComposition,
    AirfoilG3LiveError,
    AirfoilG3ProviderProfile,
    FrozenAirfoilG3ManifestGate,
    LiveGeneratorFactory,
    build_airfoil_g3_live_runtime_manifest,
    compose_airfoil_g3_live,
    resolve_airfoil_g3_provider_profile,
    verify_airfoil_g3_manifest_chronology,
    verify_airfoil_g3_no_leak_gate,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (  # noqa: E402
    DEFAULT_FREEZE_RECEIPT_PATH,
    create_prelaunch_freeze_receipt,
    load_prelaunch_freeze_receipt,
    prepare_release,
    write_prelaunch_freeze_receipt,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (  # noqa: E402
    load_frozen_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (  # noqa: E402
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.v7_readiness import (  # noqa: E402
    AirfoilV7ReadinessSpec,
    create_airfoil_v7_resource_lease,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    write_bytes_atomic,
    write_json_atomic,
)
from agent_evolve.application.live_runtime_manifest import (  # noqa: E402
    LiveRuntimeManifest,
    load_live_runtime_manifest,
)


LIVE_AUTHORIZATION = "AIRFOIL_G3_LIVE_V1"
GPT56_SOL_XHIGH_LIVE_AUTHORIZATION = "AIRFOIL_G3_GPT56_SOL_XHIGH_LIVE_V1"
LIVE_AUTHORIZATIONS_BY_PROFILE = {
    DEEPSEEK_G3_PROVIDER_PROFILE.profile_id: LIVE_AUTHORIZATION,
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE.profile_id: (
        GPT56_SOL_XHIGH_LIVE_AUTHORIZATION
    ),
}
MANIFEST_ROOT = RESEARCH_ARTIFACT_ROOT / "airfoil_g3_release" / "manifests"
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")


class AirfoilG3RunnerError(RuntimeError):
    """Canonical launch failed; durable artifacts carry the bounded evidence."""


ProblemFactory = Callable[
    [str, Path],
    tuple[AirfoilV7Problem, ConvergenceQualifiedAirfoilPanelProblem],
]
LeaseFactory = Callable[[str], ExclusiveResourceLease]


@dataclass(frozen=True, slots=True)
class AirfoilG3RunnerDependencies:
    credential_loader: Callable[[], str]
    problem_factory: ProblemFactory
    resource_lease_factory: LeaseFactory
    generator_factory: LiveGeneratorFactory | None = None

    def __post_init__(self) -> None:
        for name in (
            "credential_loader",
            "problem_factory",
            "resource_lease_factory",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if self.generator_factory is not None and not callable(
            self.generator_factory
        ):
            raise TypeError("generator_factory must be callable or None")


def _utc_seconds() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_precise() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(run_id: str) -> str:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise AirfoilG3RunnerError("run_id must use the closed lowercase grammar")
    return run_id


def canonical_manifest_path(run_id: str) -> Path:
    return MANIFEST_ROOT / f"{_validate_run_id(run_id)}.json"


def _require_empty_or_absent_root(path: Path, *, label: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise AirfoilG3RunnerError(f"{label} prior-state root is not a directory")
    try:
        next(path.iterdir())
    except StopIteration:
        return
    raise AirfoilG3RunnerError(f"{label} prior-state root is not empty")


def _require_pristine_g3_roots() -> None:
    for label, root in (
        ("manifest", MANIFEST_ROOT),
        ("run", G3_RUN_ROOT),
        ("work", G3_WORK_ROOT),
    ):
        _require_empty_or_absent_root(root, label=label)


def prepare_freeze(*, frozen_at_utc: str | None = None) -> Path:
    """Publish the write-once chronology root before any G3 run identity."""

    if DEFAULT_FREEZE_RECEIPT_PATH.exists():
        raise AirfoilG3RunnerError("canonical prelaunch freeze already exists")
    _require_pristine_g3_roots()
    preparation = prepare_release()
    receipt = create_prelaunch_freeze_receipt(
        preparation,
        frozen_at_utc=_utc_seconds() if frozen_at_utc is None else frozen_at_utc,
    )
    # Release preparation and source hashing are intentionally expensive.
    # Recheck immediately before the exclusive publication so the receipt's
    # zero-prior-state attestation remains true across that preparation window.
    _require_pristine_g3_roots()
    try:
        write_prelaunch_freeze_receipt(
            receipt,
            path=DEFAULT_FREEZE_RECEIPT_PATH,
        )
    except Exception as exc:
        raise AirfoilG3RunnerError("write-once prelaunch freeze publication failed") from exc
    if load_prelaunch_freeze_receipt(DEFAULT_FREEZE_RECEIPT_PATH) != receipt:
        raise AirfoilG3RunnerError("persisted prelaunch freeze changed on read-back")
    return DEFAULT_FREEZE_RECEIPT_PATH


def _run_settings(run_id: str):
    base = local_default_converged_settings()
    return replace(
        base,
        output_root=G3_RUN_ROOT / run_id / "cfd_receipts",
        work_root=G3_WORK_ROOT / run_id,
    )


def _production_problem_factory(
    run_id: str,
    run_dir: Path,
) -> tuple[AirfoilV7Problem, ConvergenceQualifiedAirfoilPanelProblem]:
    if run_dir != G3_RUN_ROOT / run_id:
        raise AirfoilG3RunnerError("problem factory received a foreign run directory")
    raw = ConvergenceQualifiedAirfoilPanelProblem(_run_settings(run_id))
    return AirfoilV7Problem(raw_problem=raw), raw


def _read_dotenv_api_key() -> str:
    """Read the one credential only when the lazy provider first dispatches."""

    env_path = AGENT_EVOLVE_ROOT.parent / ".env"
    value: str | None = None
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                value = candidate.strip().strip('"').strip("'")
                break
    if not value:
        value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise AirfoilG3RunnerError("OPENROUTER_API_KEY is unavailable")
    return value


def _readiness_spec() -> AirfoilV7ReadinessSpec:
    settings = local_default_converged_settings()
    return AirfoilV7ReadinessSpec(
        evaluator_python=settings.python_executable,
        evaluator_script=settings.evaluator_script,
        dataset_arrow=settings.dataset_arrow,
        expected_dataset_sha256=EXPECTED_DATASET_SHA256,
        container_image=CONTAINER_IMAGE,
        cpu_set=settings.cpu_set,
        mpi_cores=settings.mpi_cores,
    )


def _production_lease_factory(run_id: str) -> ExclusiveResourceLease:
    return create_airfoil_v7_resource_lease(
        _readiness_spec(),
        lease_path=DEFAULT_RESOURCE_LEASE_PATH,
        run_id=run_id,
        phase="airfoil_v7_g3_live",
    )


def default_dependencies() -> AirfoilG3RunnerDependencies:
    return AirfoilG3RunnerDependencies(
        credential_loader=_read_dotenv_api_key,
        problem_factory=_production_problem_factory,
        resource_lease_factory=_production_lease_factory,
    )


def _experiment_payload(manifest: LiveRuntimeManifest) -> dict[str, object]:
    rows = [
        section.to_record()["payload"]
        for section in manifest.sections
        if section.section_id == "experiment"
    ]
    if len(rows) != 1 or type(rows[0]) is not dict:
        raise AirfoilG3RunnerError("manifest has no exact experiment section")
    return rows[0]


def _result_record(result: Any, live: AirfoilG3LiveComposition) -> dict[str, object]:
    state = result.final_state
    curation = live.analysis_composition.feedback_interceptor
    curation_receipt = getattr(curation, "curation_receipt", None)
    curation_authority = getattr(curation, "curation_authority", None)
    return {
        "schema_version": 1,
        "optimizer_result_sha256": result.result_hash,
        "stop_reason": result.stop_reason.value,
        "generation": state.generation,
        "candidate_occurrences": len(state.candidates),
        "unique_evaluations": state.unique_evaluations,
        "logical_llm_calls": state.logical_llm_calls,
        "seed_receipt_sha256s": [value.receipt_hash for value in result.seed_receipts],
        "generation_receipt_sha256s": [
            value.receipt_hash for value in result.generation_receipts
        ],
        "feedback_receipt_sha256s": [
            value.receipt_hash for value in result.feedback_receipts
        ],
        "curation_authority_sha256": (
            None
            if curation_authority is None
            else curation_authority.authority_sha256
        ),
        "curation_receipt_sha256": (
            None if curation_receipt is None else curation_receipt.receipt_sha256
        ),
        "curation_status": (
            None if curation_receipt is None else curation_receipt.curation_status
        ),
    }


def _optimizer_checkpoint_record(
    *,
    result_record: Mapping[str, object],
    live: AirfoilG3LiveComposition,
    runtime_manifest_sha256: str,
    inputs: Any,
    run_started_at_utc: str,
    run_finished_at_utc: str,
    end_to_end_wall_seconds: float,
    durable_provider_outcomes: tuple[Mapping[str, object], ...],
    durable_provider_requests: tuple[Mapping[str, object], ...],
    durable_provider_outputs: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """Full post-run evidence written before close, inventory, or analysis."""

    curation = live.analysis_composition.feedback_interceptor
    authority = getattr(curation, "curation_authority", None)
    receipt = getattr(curation, "curation_receipt", None)
    if authority is None or receipt is None:
        raise AirfoilG3RunnerError("completed optimizer lacks curation evidence")
    reflection = receipt.call_receipt
    if len(durable_provider_outcomes) != 6 or len(durable_provider_requests) != 6:
        raise AirfoilG3RunnerError(
            "completed optimizer lacks six durable provider tasks/requests"
        )
    succeeded = sum(
        row.get("status") == "succeeded" for row in durable_provider_outcomes
    )
    if len(durable_provider_outputs) != succeeded:
        raise AirfoilG3RunnerError(
            "durable typed-output count differs from successful provider tasks"
        )

    def evidence_identities(
        rows: tuple[Mapping[str, object], ...],
        *,
        digest_field: str,
    ) -> list[dict[str, object]]:
        values = [
            {"call_id": row.get("call_id"), digest_field: row.get(digest_field)}
            for row in rows
        ]
        if any(
            type(value["call_id"]) is not str
            or type(value[digest_field]) is not str
            for value in values
        ):
            raise AirfoilG3RunnerError("durable structured evidence lacks identity")
        return sorted(values, key=lambda value: str(value["call_id"]))

    return {
        "schema_version": 1,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "freeze_receipt_sha256": inputs.freeze_receipt_sha256,
        "runtime_inputs_sha256": inputs.runtime_inputs_sha256,
        "optimizer_result": dict(result_record),
        "curation_authority": {
            **authority.to_record(),
            "authority_sha256": authority.authority_sha256,
        },
        "curation_receipt": {
            **receipt.to_record(),
            "receipt_sha256": receipt.receipt_sha256,
        },
        "reflection_call_receipt": {
            **reflection.to_record(),
            "receipt_sha256": reflection.receipt_sha256,
        },
        "timing": {
            "run_started_at_utc": run_started_at_utc,
            "run_finished_at_utc": run_finished_at_utc,
            "end_to_end_wall_seconds_hex": float(end_to_end_wall_seconds).hex(),
        },
        "durability": {
            "provider_outcome_rows_fsynced": len(durable_provider_outcomes),
            "provider_request_rows_fsynced": len(durable_provider_requests),
            "provider_output_rows_fsynced": len(durable_provider_outputs),
            "expected_provider_logical_tasks": 6,
            "request_evidence": evidence_identities(
                durable_provider_requests,
                digest_field="request_evidence_sha256",
            ),
            "output_evidence": evidence_identities(
                durable_provider_outputs,
                digest_field="output_evidence_sha256",
            ),
            "written_before_transport_close": True,
            "written_before_raw_receipt_inventory": True,
            "written_before_canonical_analysis": True,
        },
        "claim_boundary": {
            "completed_engine_result": True,
            "canonical_analysis_complete": False,
            "paper_ready_claim": False,
        },
    }


def prepare_manifest(
    *,
    run_id: str,
    built_at_utc: str | None = None,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> Path:
    """Publish one prospective manifest without credentials, provider, or CFD."""

    run_id = _validate_run_id(run_id)
    provider_profile.__post_init__()
    path = canonical_manifest_path(run_id)
    if path.exists() or (G3_RUN_ROOT / run_id).exists() or (G3_WORK_ROOT / run_id).exists():
        raise AirfoilG3RunnerError("run_id already has manifest, output, or work state")
    problem, _ = _production_problem_factory(run_id, G3_RUN_ROOT / run_id)
    inputs = load_frozen_airfoil_g3_runtime_inputs(problem=problem)
    manifest_built_at = _utc_seconds() if built_at_utc is None else built_at_utc
    freeze_sha256 = inputs.freeze_receipt_sha256
    assert freeze_sha256 is not None
    verify_airfoil_g3_manifest_chronology(
        built_at_utc=manifest_built_at,
        expected_freeze_receipt_sha256=freeze_sha256,
    )
    manifest = build_airfoil_g3_live_runtime_manifest(
        inputs=inputs,
        built_at_utc=manifest_built_at,
        run_id=run_id,
        provider_profile=provider_profile,
    )
    write_json_atomic(path, manifest.to_record())
    if load_live_runtime_manifest(path) != manifest:
        raise AirfoilG3RunnerError("persisted runtime manifest changed on read-back")
    return path


def _progress_record(value: StructuredStreamProgress) -> dict[str, object]:
    value.__post_init__()
    return {
        "schema_version": 1,
        "call_id": value.call_id,
        "provider_attempt_id": value.provider_attempt_id,
        "sequence": value.sequence,
        "kind": value.kind.value,
        "channel": value.channel.value,
        "elapsed_ns": value.elapsed_ns,
        "event_content_utf8_bytes": value.event_content_utf8_bytes,
        "cumulative_content_utf8_bytes": value.cumulative_content_utf8_bytes,
        "rolling_content_sha256": value.rolling_content_sha256,
    }


def _raw_receipt_inventory(
    raw: ConvergenceQualifiedAirfoilPanelProblem,
    *,
    expected_output_root: Path,
) -> tuple[Path, ...]:
    evaluator = raw.evaluator
    owned = evaluator.run_directory.resolve(strict=True)
    expected_root = expected_output_root.resolve(strict=True)
    if owned.parent != expected_root:
        raise AirfoilG3RunnerError("evaluator receipt directory escaped output_root")
    children = tuple(sorted(expected_root.iterdir(), key=lambda path: path.name))
    if children != (owned,):
        raise AirfoilG3RunnerError("output_root contains pre-existing or foreign runs")
    paths = evaluator.durable_receipt_paths()
    all_files = tuple(
        sorted((path.resolve(strict=True) for path in owned.iterdir() if path.is_file()))
    )
    if len(paths) != 11 or all_files != paths:
        raise AirfoilG3RunnerError("fresh evaluator run does not contain exactly 11 JSON receipts")
    return paths


async def execute_live(
    manifest_path: Path,
    *,
    dependencies: AirfoilG3RunnerDependencies | None = None,
) -> dict[str, object]:
    """Execute and finalize exactly one manifest-derived G3 run."""

    deps = default_dependencies() if dependencies is None else dependencies
    deps.__post_init__()
    manifest = load_live_runtime_manifest(manifest_path.expanduser().resolve(strict=True))
    experiment = _experiment_payload(manifest)
    raw_run_id = experiment.get("run_id")
    if type(raw_run_id) is not str:
        raise AirfoilG3RunnerError("manifest run_id must be an exact string")
    run_id = _validate_run_id(raw_run_id)
    raw_provider_profile_id = experiment.get("provider_profile_id")
    if type(raw_provider_profile_id) is not str:
        raise AirfoilG3RunnerError(
            "manifest provider_profile_id must be an exact string"
        )
    try:
        provider_profile = resolve_airfoil_g3_provider_profile(
            raw_provider_profile_id
        )
    except AirfoilG3LiveError as exc:
        raise AirfoilG3RunnerError("manifest binds an unknown provider profile") from exc
    canonical = canonical_manifest_path(run_id).resolve(strict=True)
    supplied = manifest_path.expanduser().resolve(strict=True)
    if supplied != canonical:
        raise AirfoilG3RunnerError("live run requires the canonical manifest path")
    run_dir = G3_RUN_ROOT / run_id
    work_root = G3_WORK_ROOT / run_id
    if run_dir.exists() or work_root.exists():
        raise AirfoilG3RunnerError("run output/work state already exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_bytes_atomic(run_dir / "runtime_manifest.json", supplied.read_bytes())

    progress = BatchedDurableJsonlJournal(
        run_dir / "provider_progress.jsonl",
        max_unfsynced_rows=32,
    )
    outcomes = DurableJsonlJournal(run_dir / "provider_outcomes.jsonl")
    requests = DurableJsonlJournal(run_dir / "provider_requests.jsonl")
    outputs = DurableJsonlJournal(run_dir / "provider_outputs.jsonl")
    traces = DurableJsonlJournal(run_dir / "execution_traces.jsonl")
    outcome_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    evidence_lock = threading.Lock()
    lease: ExclusiveResourceLease | None = None
    live: AirfoilG3LiveComposition | None = None
    pending: BaseException | None = None
    result_record: dict[str, object] | None = None
    analysis_record: dict[str, object] | None = None
    stage = "run_directory_created"
    run_status = "failed"
    finalized = False

    def trace_sink(source: str) -> Callable[[Mapping[str, object]], None]:
        def append(value: Mapping[str, object]) -> None:
            traces.append({"schema_version": 1, "source": source, **dict(value)})

        return append

    def progress_sink(value: StructuredStreamProgress) -> None:
        progress.append(_progress_record(value))

    def outcome_sink(value: Any) -> None:
        # No successful structured response may reach the engine before every
        # preceding stream event and the terminal outcome are durable.
        progress.flush()
        record = structured_generation_outcome_record(value)
        outcomes.append(record)
        with evidence_lock:
            outcome_rows.append(record)

    def request_evidence_sink(value: dict[str, object]) -> None:
        # The generic runner invokes this synchronously before queue admission.
        # Durable append therefore forms a fail-closed pre-provider barrier.
        requests.append(value)
        with evidence_lock:
            request_rows.append(dict(value))

    def output_evidence_sink(value: dict[str, object]) -> None:
        # This contains the bounded typed output and must be durable before the
        # high-level adapter or experiment-specific validator can inspect it.
        outputs.append(value)
        with evidence_lock:
            output_rows.append(dict(value))

    try:
        stage = "resource_lease_acquisition"
        lease = deps.resource_lease_factory(run_id)
        if not isinstance(lease, ExclusiveResourceLease):
            raise TypeError("resource lease factory returned a foreign object")
        lease_receipt = lease.acquire()
        write_json_atomic(
            run_dir / "resource_lease_acquired.json",
            {"schema_version": 1, "receipt": lease_receipt.to_record()},
        )

        stage = "benchmark_construction"
        problem, raw = deps.problem_factory(run_id, run_dir)
        if type(problem) is not AirfoilV7Problem or type(raw) is not ConvergenceQualifiedAirfoilPanelProblem:
            raise TypeError("problem factory must return exact Airfoil v7 runtime types")
        evaluator = raw.evaluator
        if evaluator.run_directory.exists() or raw.settings.output_root.exists():
            raise AirfoilG3RunnerError("evaluator output existed before G0")
        if raw.settings.work_root.exists():
            raise AirfoilG3RunnerError("evaluator work root existed before G0")

        stage = "runtime_inputs_and_gate"
        planner_trace = trace_sink("planner")
        inputs = load_frozen_airfoil_g3_runtime_inputs(
            problem=problem,
            planner_trace_sink=planner_trace,
        )
        gate = FrozenAirfoilG3ManifestGate(
            manifest_path=supplied,
            inputs=inputs,
            provider_profile=provider_profile,
        )
        if gate.expected_manifest_sha256 != manifest.manifest_sha256:
            raise AirfoilG3RunnerError("manifest gate authenticated another root")
        write_json_atomic(
            run_dir / "no_leak_pre_g0.json",
            verify_airfoil_g3_no_leak_gate(stage="pre_g0"),
        )

        credential_reads = 0

        def credential_loader() -> str:
            nonlocal credential_reads
            credential_reads += 1
            if credential_reads != 1:
                raise AirfoilG3RunnerError("credential loader invoked more than once")
            write_json_atomic(
                run_dir / "credential_access.json",
                {
                    "schema_version": 1,
                    "credential_name": "OPENROUTER_API_KEY",
                    "read_count": 1,
                    "value_persisted": False,
                    "stage": "first_model_call_after_seed_admission",
                },
            )
            return deps.credential_loader()

        live_kwargs: dict[str, object] = {}
        if deps.generator_factory is not None:
            live_kwargs["generator_factory"] = deps.generator_factory
        live = compose_airfoil_g3_live(
            inputs,
            launch_gate=gate,
            expected_manifest_sha256=manifest.manifest_sha256,
            credential_loader=credential_loader,
            progress_sink=progress_sink,
            outcome_sink=outcome_sink,
            request_evidence_sink=request_evidence_sink,
            output_evidence_sink=output_evidence_sink,
            provider_profile=provider_profile,
            engine_trace_sink=trace_sink("engine"),
            optimizer_trace_sink=trace_sink("optimizer"),
            **live_kwargs,
        )

        stage = "g0_g3_execution"
        started_at = _utc_precise()
        wall_started = time.perf_counter()
        result = await live.run()
        wall_seconds = time.perf_counter() - wall_started
        finished_at = _utc_precise()
        if live.run_state != "completed" or not live.initialized_provider:
            raise AirfoilG3RunnerError("live wrapper did not close its one run state")
        # Preserve the completed engine result before any receipt harvesting or
        # canonical analysis can fail.  This is evidence, not a publishable
        # claim until analysis_receipt.json is also present.
        result_record = _result_record(result, live)
        write_json_atomic(run_dir / "optimizer_result.json", result_record)
        with evidence_lock:
            checkpoint_outcomes = tuple(outcome_rows)
            checkpoint_requests = tuple(request_rows)
            checkpoint_outputs = tuple(output_rows)
        write_json_atomic(
            run_dir / "optimizer_checkpoint.json",
            _optimizer_checkpoint_record(
                result_record=result_record,
                live=live,
                runtime_manifest_sha256=manifest.manifest_sha256,
                inputs=inputs,
                run_started_at_utc=started_at,
                run_finished_at_utc=finished_at,
                end_to_end_wall_seconds=float(wall_seconds),
                durable_provider_outcomes=checkpoint_outcomes,
                durable_provider_requests=checkpoint_requests,
                durable_provider_outputs=checkpoint_outputs,
            ),
        )

        stage = "provider_transport_close"
        await live.aclose()

        stage = "postrun_no_leak_and_receipts"
        write_json_atomic(
            run_dir / "no_leak_post_g3.json",
            verify_airfoil_g3_no_leak_gate(stage="post_g3"),
        )
        raw_paths = _raw_receipt_inventory(
            raw,
            expected_output_root=raw.settings.output_root,
        )
        write_json_atomic(
            run_dir / "raw_receipt_inventory.json",
            {
                "schema_version": 1,
                "fresh_evaluator_run_directory": str(evaluator.run_directory),
                "preexisting_receipts": 0,
                "receipt_count": len(raw_paths),
                "reproduction_receipt_created": False,
                "relative_paths": [
                    path.relative_to(run_dir).as_posix() for path in raw_paths
                ],
            },
        )

        stage = "canonical_analysis"
        with evidence_lock:
            durable_outcomes = tuple(outcome_rows)
            durable_requests = tuple(request_rows)
            durable_outputs = tuple(output_rows)
        analysis = await analyze_airfoil_g3_live_result(
            composition=live.analysis_composition,
            inputs=inputs,
            result=result,
            runtime_manifest_sha256=manifest.manifest_sha256,
            provider_outcomes=durable_outcomes,
            provider_requests=durable_requests,
            provider_outputs=durable_outputs,
            raw_evaluator_receipt_paths=raw_paths,
            run_started_at_utc=started_at,
            run_finished_at_utc=finished_at,
            end_to_end_wall_seconds=float(wall_seconds),
            provider_profile=provider_profile,
        )
        analysis_record = analysis.to_record()
        write_json_atomic(run_dir / "analysis_receipt.json", analysis_record)
        run_status = "completed"
    except BaseException as exc:
        pending = exc
        try:
            write_json_atomic(
                run_dir / "failure.json",
                {
                    "schema_version": 1,
                    "stage": stage,
                    "failure_code": "airfoil_g3_live_stage_failure",
                    "failure_type": type(exc).__name__,
                    "safe_message": "inspect authenticated journals and stage evidence",
                    "credential_value_persisted": False,
                },
            )
        except BaseException as artifact_exc:
            exc.add_note(f"failure artifact also failed: {type(artifact_exc).__name__}")
    finally:
        if live is not None:
            try:
                await live.aclose()
            except BaseException as close_exc:
                if pending is None:
                    pending = close_exc
                else:
                    pending.add_note(
                        f"provider close also failed: {type(close_exc).__name__}"
                    )
        for journal in (progress, outcomes, requests, outputs, traces):
            try:
                journal.close()
            except BaseException as close_exc:
                if pending is None:
                    pending = close_exc
                else:
                    pending.add_note(
                        f"journal close also failed: {type(close_exc).__name__}"
                    )
        if lease is not None and lease.active:
            try:
                release = lease.release(
                    outcome=run_status if pending is None else "failed",
                    failure_type=None if pending is None else type(pending).__name__,
                )
                write_json_atomic(
                    run_dir / "resource_lease_released.json",
                    {"schema_version": 1, "release": release},
                )
            except BaseException as release_exc:
                if pending is None:
                    pending = release_exc
                else:
                    pending.add_note(
                        f"resource release also failed: {type(release_exc).__name__}"
                    )
        try:
            finalize_run_directory(
                run_dir,
                status=run_status if pending is None else "failed",
            )
            finalized = True
        except BaseException as final_exc:
            if pending is None:
                pending = final_exc
            else:
                pending.add_note(
                    f"run finalization also failed: {type(final_exc).__name__}"
                )

    if pending is not None:
        raise AirfoilG3RunnerError(
            "Airfoil G3 live run failed; inspect the finalized run directory"
        ) from None
    if not finalized or result_record is None or analysis_record is None:
        raise AirfoilG3RunnerError("live run did not publish complete final evidence")
    return {
        "run_dir": str(run_dir),
        "result": result_record,
        "analysis": analysis_record,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare-freeze")
    freeze.add_argument("--frozen-at-utc")
    prepare = sub.add_parser("prepare-manifest")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--built-at-utc")
    prepare.add_argument(
        "--provider-profile",
        choices=tuple(sorted(AIRFOIL_G3_PROVIDER_PROFILES)),
        default=DEEPSEEK_G3_PROVIDER_PROFILE.profile_id,
    )
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--authorize-live", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-freeze":
        path = prepare_freeze(frozen_at_utc=args.frozen_at_utc)
        print(path)
        return 0
    if args.command == "prepare-manifest":
        path = prepare_manifest(
            run_id=args.run_id,
            built_at_utc=args.built_at_utc,
            provider_profile=resolve_airfoil_g3_provider_profile(
                args.provider_profile
            ),
        )
        print(path)
        return 0
    manifest = load_live_runtime_manifest(args.manifest.expanduser().resolve(strict=True))
    experiment = _experiment_payload(manifest)
    profile_id = experiment.get("provider_profile_id")
    expected_authorization = LIVE_AUTHORIZATIONS_BY_PROFILE.get(profile_id)
    if (
        expected_authorization is None
        or args.authorize_live != expected_authorization
    ):
        raise AirfoilG3RunnerError("explicit live authorization token is required")
    outcome = asyncio.run(execute_live(args.manifest))
    print(outcome["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

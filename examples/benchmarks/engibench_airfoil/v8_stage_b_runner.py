"""Small durable runner for the first authentic Airfoil Stage-B A/U block.

This is intentionally a development runner, not a paper-release ceremony.  It
still fails closed on an existing output/work root, acquires the host-global
Airfoil lease, defers credential access until the first model proposal, and
durably records provider evidence and engine traces before finalization.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_evolve.agentic import ExclusiveResourceLease
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    structured_generation_outcome_record,
)
from agent_evolve.ports.structured_generator import StructuredStreamProgress
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ConvergenceQualifiedAirfoilPanelProblem,
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    EXPECTED_DATASET_SHA256,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    CONTAINER_IMAGE,
    DEFAULT_RESOURCE_LEASE_PATH,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    load_frozen_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import AirfoilV7Problem
from examples.benchmarks.engibench_airfoil.v7_readiness import (
    AirfoilV7ReadinessSpec,
    create_airfoil_v7_resource_lease,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AGENT_EVOLVE_ROOT,
    RESEARCH_ARTIFACT_ROOT,
    AirfoilV8StageBLiveComposition,
    airfoil_v8_stage_b_readiness_record,
    compose_airfoil_v8_stage_b_inputs,
    compose_airfoil_v8_stage_b_live,
)
from examples.development.durable_run_artifacts import (
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    write_json_atomic,
)


LIVE_AUTHORIZATION = "AIRFOIL_V8_STAGE_B_DEVELOPMENT_LIVE_V1"
RUN_ROOT = (
    RESEARCH_ARTIFACT_ROOT / "experiment_logs" / "airfoil_stage_b_development"
)
WORK_ROOT = Path("/tmp/agent_evolve_airfoil_v8_stage_b")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")


class AirfoilV8StageBRunnerError(RuntimeError):
    """The minimal live run failed; its finalized directory retains evidence."""


def _validate_run_id(value: str) -> str:
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise AirfoilV8StageBRunnerError("run_id uses an invalid grammar")
    return value


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_dotenv_api_key() -> str:
    """Read only OPENROUTER_API_KEY, lazily, without publishing its value."""

    path = AGENT_EVOLVE_ROOT.parent / ".env"
    value: str | None = None
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                value = candidate.strip().strip('"').strip("'")
                break
    if not value:
        value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise AirfoilV8StageBRunnerError("OPENROUTER_API_KEY is unavailable")
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


def _lease(run_id: str) -> ExclusiveResourceLease:
    return create_airfoil_v7_resource_lease(
        _readiness_spec(),
        lease_path=DEFAULT_RESOURCE_LEASE_PATH,
        run_id=run_id,
        phase="airfoil_v8_stage_b_development",
    )


def _problem(run_id: str, run_dir: Path) -> tuple[
    AirfoilV7Problem,
    ConvergenceQualifiedAirfoilPanelProblem,
]:
    settings = replace(
        local_default_converged_settings(),
        output_root=run_dir / "cfd_receipts",
        work_root=WORK_ROOT / run_id,
    )
    raw = ConvergenceQualifiedAirfoilPanelProblem(settings)
    return AirfoilV7Problem(raw_problem=raw), raw


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


def _candidate_record(candidate) -> object:
    if candidate is None:
        return None
    detailed = candidate.detailed_evaluation
    return {
        "candidate_id": candidate.candidate_id.value,
        "valid": candidate.valid,
        "configuration_hash": candidate.occurrence.configuration_hash,
        "objectives": {name: float(value).hex() for name, value in candidate.objectives},
        "violations": (
            {}
            if detailed is None
            else {name: float(value).hex() for name, value in detailed.violations}
        ),
        "evaluation_total_wall_seconds": (
            None if detailed is None else detailed.timings.total_wall_seconds
        ),
        "evaluation_receipt": (
            None
            if detailed is None or detailed.receipt is None
            else detailed.receipt.artifact_id.value
        ),
    }


def _result_record(result, live: AirfoilV8StageBLiveComposition) -> dict[str, object]:
    planner = live.composition.planner
    authority = planner.authority
    uniform = planner.uniform_decision
    if authority is None or uniform is None or len(result.generation_receipts) != 1:
        raise AirfoilV8StageBRunnerError("completed run lacks one exact A/U authority")
    receipt = result.generation_receipts[0]
    by_slot = {value.slot.slot_id: value for value in receipt.slot_results}
    if set(by_slot) != {"A", "U"}:
        raise AirfoilV8StageBRunnerError("completed run lacks exact A and U slots")
    model_decision = by_slot["A"].outcome.finite_action_decision
    if model_decision is None:
        raise AirfoilV8StageBRunnerError("model arm lacks its sealed decision")
    arms: dict[str, object] = {}
    for slot_id in ("A", "U"):
        row = by_slot[slot_id]
        decision = model_decision if slot_id == "A" else uniform
        arms[slot_id] = {
            "role": row.slot.role,
            "selector_kind": decision.selector_kind.value,
            "option_id": decision.option_id,
            "selected_ordinal": decision.selected_ordinal,
            "decision_sha256": decision.decision_sha256,
            "reward_hex": row.outcome.reward.hex(),
            "failure_stage": row.outcome.failure_stage,
            "candidate": _candidate_record(row.outcome.candidate),
        }
    a_reward = by_slot["A"].outcome.reward
    u_reward = by_slot["U"].outcome.reward
    return {
        "schema_version": 1,
        "claim_boundary": "development_not_fresh_paper_evidence",
        "optimizer_result_sha256": result.result_hash,
        "stop_reason": result.stop_reason.value,
        "generation": result.final_state.generation,
        "candidate_occurrences": len(result.final_state.candidates),
        "unique_evaluations": result.final_state.unique_evaluations,
        "logical_llm_calls": result.final_state.logical_llm_calls,
        "authority_sha256": authority.authority_sha256,
        "support_sha256": authority.support.support_sha256,
        "support_cardinality": authority.support.cardinality,
        "a_u_alias": model_decision.option_id == uniform.option_id,
        "arms": arms,
        "adaptive_minus_uniform_reward_hex": (a_reward - u_reward).hex(),
        "adaptive_beats_uniform": a_reward > u_reward,
    }


def readiness(
    run_id: str,
    *,
    inputs_factory=compose_airfoil_v8_stage_b_inputs,
    readiness_record_factory=None,
) -> dict[str, object]:
    """Build the exact real objects without credentials, model calls, or CFD."""

    canonical = _validate_run_id(run_id)
    if not callable(inputs_factory):
        raise TypeError("inputs_factory must be callable")
    if readiness_record_factory is not None and not callable(
        readiness_record_factory
    ):
        raise TypeError("readiness_record_factory must be callable or None")
    run_dir = RUN_ROOT / canonical
    problem, _ = _problem(canonical, run_dir)
    source = load_frozen_airfoil_g3_runtime_inputs(problem=problem)
    inputs = inputs_factory(source)
    if readiness_record_factory is None:
        return airfoil_v8_stage_b_readiness_record(inputs)
    return readiness_record_factory(source, inputs)


async def execute_live(
    run_id: str,
    *,
    credential_source: Callable[[], str] = _read_dotenv_api_key,
    resource_lease_factory: Callable[[str], ExclusiveResourceLease] = _lease,
    generator_factory=None,
    inputs_factory=compose_airfoil_v8_stage_b_inputs,
    readiness_record_factory=None,
    result_record_factory=None,
) -> dict[str, object]:
    """Run and finalize one authentic A/U development block."""

    canonical = _validate_run_id(run_id)
    if not callable(inputs_factory):
        raise TypeError("inputs_factory must be callable")
    if readiness_record_factory is not None and not callable(
        readiness_record_factory
    ):
        raise TypeError("readiness_record_factory must be callable or None")
    if result_record_factory is not None and not callable(result_record_factory):
        raise TypeError("result_record_factory must be callable or None")
    run_dir = RUN_ROOT / canonical
    work_dir = WORK_ROOT / canonical
    if run_dir.exists() or work_dir.exists():
        raise AirfoilV8StageBRunnerError("run output/work root already exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    progress = BatchedDurableJsonlJournal(
        run_dir / "provider_progress.jsonl",
        max_unfsynced_rows=32,
    )
    outcomes = DurableJsonlJournal(run_dir / "provider_outcomes.jsonl")
    requests = DurableJsonlJournal(run_dir / "provider_requests.jsonl")
    outputs = DurableJsonlJournal(run_dir / "provider_outputs.jsonl")
    traces = DurableJsonlJournal(run_dir / "execution_traces.jsonl")
    lease: ExclusiveResourceLease | None = None
    live: AirfoilV8StageBLiveComposition | None = None
    pending: BaseException | None = None
    status = "failed"
    stage = "run_directory_created"
    result_record: dict[str, object] | None = None
    credential_reads = 0

    def trace_sink(source: str):
        return lambda row: traces.append(
            {"schema_version": 1, "source": source, **dict(row)}
        )

    def progress_sink(value: StructuredStreamProgress) -> None:
        progress.append(_progress_record(value))

    def outcome_sink(value: Any) -> None:
        progress.flush()
        outcomes.append(structured_generation_outcome_record(value))

    def credential_loader() -> str:
        nonlocal credential_reads
        credential_reads += 1
        if credential_reads != 1:
            raise AirfoilV8StageBRunnerError("credential loader invoked repeatedly")
        write_json_atomic(
            run_dir / "credential_access.json",
            {
                "schema_version": 1,
                "credential_name": "OPENROUTER_API_KEY",
                "read_count": 1,
                "value_persisted": False,
                "stage": "first_model_call_after_seed_evaluation",
            },
        )
        return credential_source()

    try:
        stage = "resource_lease"
        lease = resource_lease_factory(canonical)
        if not isinstance(lease, ExclusiveResourceLease):
            raise TypeError("resource lease factory returned a foreign object")
        acquired = lease.acquire()
        write_json_atomic(
            run_dir / "resource_lease_acquired.json",
            {"schema_version": 1, "receipt": acquired.to_record()},
        )

        stage = "benchmark_and_learned_card"
        problem, raw = _problem(canonical, run_dir)
        source = load_frozen_airfoil_g3_runtime_inputs(
            problem=problem,
            planner_trace_sink=trace_sink("planner"),
        )
        inputs = inputs_factory(source)
        readiness_record = (
            airfoil_v8_stage_b_readiness_record(inputs)
            if readiness_record_factory is None
            else readiness_record_factory(source, inputs)
        )
        write_json_atomic(
            run_dir / "readiness.json",
            readiness_record,
        )

        kwargs: dict[str, object] = {}
        if generator_factory is not None:
            kwargs["generator_factory"] = generator_factory
        live = compose_airfoil_v8_stage_b_live(
            inputs,
            credential_loader=credential_loader,
            progress_sink=progress_sink,
            outcome_sink=outcome_sink,
            request_evidence_sink=lambda row: requests.append(dict(row)),
            output_evidence_sink=lambda row: outputs.append(dict(row)),
            engine_trace_sink=trace_sink("engine"),
            optimizer_trace_sink=trace_sink("optimizer"),
            **kwargs,
        )
        if live.generator.initialized:
            raise AirfoilV8StageBRunnerError("provider initialized before run")

        stage = "stage_b_a_u_execution"
        started_at = _utc()
        wall_start = time.perf_counter()
        result = await live.run()
        wall_seconds = time.perf_counter() - wall_start
        finished_at = _utc()
        result_record = (
            _result_record(result, live)
            if result_record_factory is None
            else result_record_factory(result, live, source, inputs)
        )
        if type(result_record) is not dict:
            raise TypeError("result_record_factory must return an exact dictionary")
        result_record["timing"] = {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "end_to_end_wall_seconds": wall_seconds,
        }
        write_json_atomic(run_dir / "result.json", result_record)

        stage = "transport_close_and_receipts"
        await live.aclose()
        receipts = raw.evaluator.durable_receipt_paths()
        write_json_atomic(
            run_dir / "raw_receipt_inventory.json",
            {
                "schema_version": 1,
                "receipt_count": len(receipts),
                "relative_paths": [
                    path.relative_to(run_dir).as_posix() for path in receipts
                ],
            },
        )
        status = "completed"
    except BaseException as exc:
        pending = exc
        try:
            write_json_atomic(
                run_dir / "failure.json",
                {
                    "schema_version": 1,
                    "stage": stage,
                    "failure_type": type(exc).__name__,
                    "credential_value_persisted": False,
                    "safe_message": "inspect finalized Stage-B journals",
                },
            )
        except BaseException as artifact_exc:
            exc.add_note(f"failure journal also failed: {type(artifact_exc).__name__}")
    finally:
        if live is not None:
            try:
                await live.aclose()
            except BaseException as exc:
                pending = exc if pending is None else pending
        for journal in (progress, outcomes, requests, outputs, traces):
            try:
                journal.close()
            except BaseException as exc:
                pending = exc if pending is None else pending
        if lease is not None and lease.active:
            try:
                released = lease.release(
                    outcome=status if pending is None else "failed",
                    failure_type=None if pending is None else type(pending).__name__,
                )
                write_json_atomic(
                    run_dir / "resource_lease_released.json",
                    {"schema_version": 1, "release": released},
                )
            except BaseException as exc:
                pending = exc if pending is None else pending
        try:
            finalize_run_directory(
                run_dir,
                status=status if pending is None else "failed",
            )
        except BaseException as exc:
            pending = exc if pending is None else pending

    if pending is not None:
        raise AirfoilV8StageBRunnerError(
            f"Stage-B run failed at {stage}; inspect {run_dir}"
        ) from None
    if result_record is None:
        raise AirfoilV8StageBRunnerError("Stage-B run produced no result record")
    return {"run_dir": str(run_dir), "result": result_record}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ready = sub.add_parser("readiness")
    ready.add_argument("--run-id", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--authorize-live", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "readiness":
        print(json.dumps(readiness(args.run_id), sort_keys=True))
        return 0
    if args.authorize_live != LIVE_AUTHORIZATION:
        raise AirfoilV8StageBRunnerError("explicit live authorization token required")
    outcome = asyncio.run(execute_live(args.run_id))
    print(outcome["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LIVE_AUTHORIZATION",
    "RUN_ROOT",
    "AirfoilV8StageBRunnerError",
    "execute_live",
    "main",
    "readiness",
]

"""Durable provider-free runner for the finalized T000 K=8 support oracle.

This runner never initializes a model provider and never reads credentials.  It
authenticates the finalized T000 block, reuses its two measured A/U children,
and evaluates only the sealed factor probe when the source repeatability check
permits factorization.  Any failed factor certificate switches, without retry,
to direct evaluation of all six previously unseen support children.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from agent_evolve.agentic import ExclusiveResourceLease, thaw_json
from agent_evolve.application.detailed_evaluation import DetailedEvaluationPayload
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ConvergenceQualifiedAirfoilPanelProblem,
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    EXPECTED_DATASET_SHA256,
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    CONTAINER_IMAGE,
    DEFAULT_RESOURCE_LEASE_PATH,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    load_frozen_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AirfoilV7Problem,
    replay_airfoil_v7_durable_receipt,
)
from examples.benchmarks.engibench_airfoil.v7_readiness import (
    AirfoilV7ReadinessSpec,
    create_airfoil_v7_resource_lease,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AirfoilV8StageBInputs,
    RESEARCH_ARTIFACT_ROOT,
)
from examples.benchmarks.engibench_airfoil.v9_stage_b_support_oracle import (
    AirfoilV9SupportOracleError,
    FACTOR_PROBE_OPTION_ID,
    FACTOR_PROBE_ORDINAL,
    SOURCE_RUN_DIR,
    SupportObservation,
    build_support_oracle_result,
    factorization_certificate,
    load_t000_support_oracle_context,
    observation_from_receipt,
    seal_support_oracle_result,
    support_oracle_readiness_record,
)
from examples.benchmarks.engibench_airfoil.v9_stage_b_transfer import (
    compose_airfoil_v9_stage_b_transfer_inputs,
)
from examples.development.durable_run_artifacts import (
    DurableJsonlJournal,
    finalize_run_directory,
    write_json_atomic,
)


LIVE_AUTHORIZATION = "AIRFOIL_V9_STAGE_B_T000_SUPPORT_ORACLE_LIVE_V1"
RUN_ROOT = RESEARCH_ARTIFACT_ROOT / "experiment_logs" / "airfoil_stage_b_support_oracle"
WORK_ROOT = Path("/tmp/agent_evolve_airfoil_v9_stage_b_support_oracle")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")


class AirfoilV9SupportOracleRunnerError(RuntimeError):
    """The support-oracle run failed; its finalized directory retains evidence."""


ReceiptInventory = Callable[[], tuple[Path, ...]]
ProblemFactory = Callable[
    [str, Path, Path],
    tuple[AirfoilV7Problem, ReceiptInventory],
]
InputsFactory = Callable[[AirfoilV7Problem], AirfoilV8StageBInputs]


def _validate_run_id(value: str) -> str:
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise AirfoilV9SupportOracleRunnerError("run_id uses an invalid grammar")
    return value


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _production_lease(run_id: str) -> ExclusiveResourceLease:
    return create_airfoil_v7_resource_lease(
        _readiness_spec(),
        lease_path=DEFAULT_RESOURCE_LEASE_PATH,
        run_id=run_id,
        phase="airfoil_v9_stage_b_t000_support_oracle",
    )


def _production_problem(
    run_id: str,
    run_dir: Path,
    work_dir: Path,
) -> tuple[AirfoilV7Problem, ReceiptInventory]:
    del run_id
    settings = replace(
        local_default_converged_settings(),
        output_root=run_dir / "cfd_receipts",
        work_root=work_dir,
    )
    raw = ConvergenceQualifiedAirfoilPanelProblem(settings)
    return AirfoilV7Problem(raw_problem=raw), raw.evaluator.durable_receipt_paths


def _production_inputs(problem: AirfoilV7Problem) -> AirfoilV8StageBInputs:
    source = load_frozen_airfoil_g3_runtime_inputs(problem=problem)
    return compose_airfoil_v9_stage_b_transfer_inputs(source, panel_index=0)


@dataclass(frozen=True, slots=True)
class SupportOracleRunnerDependencies:
    """Small composition boundary used by production and provider/CFD-free tests."""

    problem_factory: ProblemFactory = _production_problem
    inputs_factory: InputsFactory = _production_inputs
    resource_lease_factory: Callable[[str], ExclusiveResourceLease] = _production_lease
    monotonic_ns: Callable[[], int] = time.perf_counter_ns
    source_run_dir: Path = SOURCE_RUN_DIR
    run_root: Path = RUN_ROOT
    work_root: Path = WORK_ROOT

    def __post_init__(self) -> None:
        for name in (
            "problem_factory",
            "inputs_factory",
            "resource_lease_factory",
            "monotonic_ns",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        for name in ("source_run_dir", "run_root", "work_root"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")


def _compose_context(
    run_id: str,
    dependencies: SupportOracleRunnerDependencies,
) -> tuple[AirfoilV7Problem, ReceiptInventory, AirfoilV8StageBInputs, object]:
    run_dir = dependencies.run_root / run_id
    work_dir = dependencies.work_root / run_id
    problem, receipt_inventory = dependencies.problem_factory(
        run_id,
        run_dir,
        work_dir,
    )
    if type(problem) is not AirfoilV7Problem:
        raise TypeError("problem_factory must return an exact AirfoilV7Problem")
    if not callable(receipt_inventory):
        raise TypeError("problem_factory receipt inventory must be callable")
    inputs = dependencies.inputs_factory(problem)
    if type(inputs) is not AirfoilV8StageBInputs:
        raise TypeError("inputs_factory must return exact AirfoilV8StageBInputs")
    context = load_t000_support_oracle_context(
        inputs,
        source_run_dir=dependencies.source_run_dir,
    )
    return problem, receipt_inventory, inputs, context


def _runner_readiness_record(
    run_id: str,
    dependencies: SupportOracleRunnerDependencies,
    context,
    *,
    roots_absent_at_start: bool | None = None,
) -> dict[str, object]:
    record = support_oracle_readiness_record(context)
    run_dir = dependencies.run_root / run_id
    work_dir = dependencies.work_root / run_id
    roots_absent = (
        not run_dir.exists() and not work_dir.exists()
        if roots_absent_at_start is None
        else roots_absent_at_start
    )
    record.update(
        {
            "ready": bool(record["ready"] and roots_absent),
            "run": {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "work_dir": str(work_dir),
                "output_and_work_roots_absent": roots_absent,
                "live_authorization": LIVE_AUTHORIZATION,
                "launch_command": (
                    "uv run python -m examples.benchmarks.engibench_airfoil."
                    "v9_stage_b_support_oracle_runner run "
                    f"--run-id {run_id} --authorize-live {LIVE_AUTHORIZATION}"
                ),
            },
        }
    )
    return record


def readiness(
    run_id: str,
    *,
    dependencies: SupportOracleRunnerDependencies | None = None,
) -> dict[str, object]:
    """Authenticate exact live objects without credentials, providers, or CFD."""

    canonical = _validate_run_id(run_id)
    deps = SupportOracleRunnerDependencies() if dependencies is None else dependencies
    deps.__post_init__()
    _, _, _, context = _compose_context(canonical, deps)
    return _runner_readiness_record(canonical, deps, context)


def _resolved_inventory(paths: tuple[Path, ...]) -> set[Path]:
    if type(paths) is not tuple or any(not isinstance(path, Path) for path in paths):
        raise AirfoilV9SupportOracleRunnerError(
            "receipt inventory must be an exact tuple of Paths"
        )
    return {path.expanduser().resolve(strict=True) for path in paths}


def _evaluate_once(
    *,
    ordinal: int,
    mode: str,
    attempt_sequence: int,
    context,
    evaluator,
    receipt_inventory: ReceiptInventory,
    run_dir: Path,
    attempts: DurableJsonlJournal,
    monotonic_ns: Callable[[], int],
) -> SupportObservation:
    option_row = context.authority.support.options[ordinal]
    option_id = option_row.option.option_id
    child = normalize_candidate(thaw_json(option_row.option.child_configuration))
    raw_candidate_sha = candidate_sha256(child)
    attempt_id = f"support-{ordinal:03d}-attempt-1"
    before = _resolved_inventory(receipt_inventory())
    start_ns = monotonic_ns()
    if type(start_ns) is not int or start_ns < 0:
        raise AirfoilV9SupportOracleRunnerError("monotonic clock returned invalid time")
    attempts.append(
        {
            "schema_version": 1,
            "event_type": "evaluation_attempt_started",
            "attempt_sequence": attempt_sequence,
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "mode": mode,
            "ordinal": ordinal,
            "option_id": option_id,
            "configuration_sha256": option_row.option.child_configuration_sha256,
            "raw_candidate_sha256": raw_candidate_sha,
            "started_at_utc": _utc(),
        }
    )
    new_receipts: set[Path] = set()
    try:
        payload = evaluator.evaluate_evidence(child)
        if type(payload) is not DetailedEvaluationPayload:
            raise AirfoilV9SupportOracleRunnerError(
                "detailed evaluator returned a foreign payload"
            )
        after = _resolved_inventory(receipt_inventory())
        if not before.issubset(after):
            raise AirfoilV9SupportOracleRunnerError(
                "receipt inventory removed an earlier durable receipt"
            )
        new_receipts = after - before
        if len(new_receipts) != 1:
            raise AirfoilV9SupportOracleRunnerError(
                "one evaluation did not publish exactly one new durable receipt"
            )
        receipt_path = next(iter(new_receipts))
        replayed = replay_airfoil_v7_durable_receipt(receipt_path, child)
        if replayed != payload:
            raise AirfoilV9SupportOracleRunnerError(
                "live payload differs from durable raw-receipt replay"
            )
        observation = observation_from_receipt(
            authority=context.authority,
            ordinal=ordinal,
            receipt_path=receipt_path,
            receipt_root=run_dir,
            evidence_mode=mode,
        )
        finish_ns = monotonic_ns()
        if type(finish_ns) is not int or finish_ns < start_ns:
            raise AirfoilV9SupportOracleRunnerError("monotonic clock moved backwards")
        attempts.append(
            {
                "schema_version": 1,
                "event_type": "evaluation_attempt_terminal",
                "attempt_sequence": attempt_sequence,
                "attempt_id": attempt_id,
                "attempt_number": 1,
                "status": "succeeded",
                "mode": mode,
                "ordinal": ordinal,
                "option_id": option_id,
                "elapsed_ns": finish_ns - start_ns,
                "observation": observation.to_record(),
            }
        )
        return observation
    except BaseException as exc:
        finish_ns = monotonic_ns()
        elapsed_ns = finish_ns - start_ns if type(finish_ns) is int else None
        attempts.append(
            {
                "schema_version": 1,
                "event_type": "evaluation_attempt_terminal",
                "attempt_sequence": attempt_sequence,
                "attempt_id": attempt_id,
                "attempt_number": 1,
                "status": "failed",
                "mode": mode,
                "ordinal": ordinal,
                "option_id": option_id,
                "elapsed_ns": elapsed_ns,
                "failure_type": type(exc).__name__,
                "new_receipt_count": len(new_receipts),
                "safe_message": "inspect the durable raw receipt and finalization",
            }
        )
        raise


def execute_live(
    run_id: str,
    *,
    dependencies: SupportOracleRunnerDependencies | None = None,
) -> dict[str, object]:
    """Execute and finalize one authenticated provider-free support oracle."""

    canonical = _validate_run_id(run_id)
    deps = SupportOracleRunnerDependencies() if dependencies is None else dependencies
    deps.__post_init__()
    run_dir = deps.run_root / canonical
    work_dir = deps.work_root / canonical
    if run_dir.exists() or work_dir.exists():
        raise AirfoilV9SupportOracleRunnerError("run output/work root already exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    attempts = DurableJsonlJournal(run_dir / "attempts.jsonl")
    lease: ExclusiveResourceLease | None = None
    receipt_inventory: ReceiptInventory | None = None
    pending: BaseException | None = None
    result_record: dict[str, object] | None = None
    status = "failed"
    stage = "run_directory_created"
    run_start_ns = deps.monotonic_ns()
    started_at = _utc()

    try:
        stage = "source_authentication_and_readiness"
        problem, receipt_inventory, inputs, context = _compose_context(canonical, deps)
        del inputs
        readiness_record = _runner_readiness_record(
            canonical,
            deps,
            context,
            roots_absent_at_start=True,
        )
        write_json_atomic(run_dir / "readiness.json", readiness_record)

        evaluator = problem.detailed_evaluator
        if evaluator is None or not callable(getattr(evaluator, "evaluate_evidence", None)):
            raise AirfoilV9SupportOracleRunnerError(
                "benchmark lacks its detailed evaluation adapter"
            )

        stage = "resource_lease"
        lease = deps.resource_lease_factory(canonical)
        if not isinstance(lease, ExclusiveResourceLease):
            raise TypeError("resource lease factory returned a foreign object")
        acquired = lease.acquire()
        write_json_atomic(
            run_dir / "resource_lease_acquired.json",
            {"schema_version": 1, "receipt": acquired.to_record()},
        )

        stage = "support_evaluation"
        known_ids = set(context.known_by_option)
        unseen_ordinals = [
            ordinal
            for ordinal, row in enumerate(context.authority.support.options)
            if row.option.option_id not in known_ids
        ]
        if len(unseen_ordinals) != 6 or FACTOR_PROBE_ORDINAL not in unseen_ordinals:
            raise AirfoilV9SupportOracleError("T000 does not expose exact six unseen options")
        source_readiness = support_oracle_readiness_record(context)
        factorization = source_readiness["factorization"]
        if not isinstance(factorization, Mapping):
            raise AirfoilV9SupportOracleError("factorization readiness is malformed")
        primary_eligible = factorization.get("primary_factor_probe_eligible") is True
        observations: list[SupportObservation] = []
        evaluated_ids: set[str] = set()

        def evaluate_ordinal(ordinal: int, *, mode: str) -> SupportObservation:
            option_id = context.authority.support.options[ordinal].option.option_id
            if option_id in known_ids or option_id in evaluated_ids:
                raise AirfoilV9SupportOracleRunnerError(
                    "known or already evaluated support option was scheduled again"
                )
            observation = _evaluate_once(
                ordinal=ordinal,
                mode=mode,
                attempt_sequence=len(observations) + 1,
                context=context,
                evaluator=evaluator,
                receipt_inventory=receipt_inventory,
                run_dir=run_dir,
                attempts=attempts,
                monotonic_ns=deps.monotonic_ns,
            )
            if observation.option_id != option_id:
                raise AirfoilV9SupportOracleRunnerError(
                    "evaluation observation changed the scheduled option"
                )
            evaluated_ids.add(option_id)
            observations.append(observation)
            return observation

        fallback_used = not primary_eligible
        if primary_eligible:
            probe = evaluate_ordinal(FACTOR_PROBE_ORDINAL, mode="factor_probe")
            if probe.option_id != FACTOR_PROBE_OPTION_ID:
                raise AirfoilV9SupportOracleError("factor probe identity changed")
            passed, _, _ = factorization_certificate(
                context,
                (*context.known_observations, probe),
            )
            fallback_used = not passed

        if fallback_used:
            fallback_mode = (
                "direct_full_six_fallback"
                if primary_eligible
                else "direct_full_six_required"
            )
            for ordinal in unseen_ordinals:
                option_id = context.authority.support.options[ordinal].option.option_id
                if option_id not in evaluated_ids:
                    evaluate_ordinal(ordinal, mode=fallback_mode)

        expected_new_count = 6 if fallback_used else 1
        if len(observations) != expected_new_count:
            raise AirfoilV9SupportOracleRunnerError(
                "support-oracle path used an unexpected evaluation count"
            )
        unsigned_result = build_support_oracle_result(
            context,
            new_observations=observations,
            fallback_used=fallback_used,
        )
        run_finish_ns = deps.monotonic_ns()
        if type(run_finish_ns) is not int or run_finish_ns < run_start_ns:
            raise AirfoilV9SupportOracleRunnerError("run monotonic timing is invalid")
        unsigned_result.update(
            {
                "run_id": canonical,
                "timing": {
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc(),
                    "end_to_end_wall_seconds_hex": (
                        (run_finish_ns - run_start_ns) / 1_000_000_000.0
                    ).hex(),
                },
                "attempt_count": len(observations),
                "attempt_policy": "exactly_one_attempt_per_scheduled_option_no_retry",
            }
        )
        result_record = seal_support_oracle_result(unsigned_result)
        write_json_atomic(run_dir / "result.json", result_record)

        stage = "raw_receipt_inventory"
        final_receipts = _resolved_inventory(receipt_inventory())
        if len(final_receipts) != len(observations):
            raise AirfoilV9SupportOracleRunnerError(
                "new raw receipt inventory differs from successful evaluation count"
            )
        relative_paths = sorted(
            path.relative_to(run_dir.resolve(strict=True)).as_posix()
            for path in final_receipts
        )
        write_json_atomic(
            run_dir / "raw_receipt_inventory.json",
            {
                "schema_version": 1,
                "receipt_count": len(relative_paths),
                "relative_paths": relative_paths,
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
                    "provider_calls": 0,
                    "credentials_read": False,
                    "safe_message": "inspect finalized support-oracle evidence",
                },
            )
        except BaseException as artifact_exc:
            exc.add_note(f"failure journal also failed: {type(artifact_exc).__name__}")
    finally:
        try:
            attempts.close()
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
        raise AirfoilV9SupportOracleRunnerError(
            f"support-oracle run failed at {stage}; inspect {run_dir}"
        ) from None
    if result_record is None:
        raise AirfoilV9SupportOracleRunnerError("support-oracle run produced no result")
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
        raise AirfoilV9SupportOracleRunnerError(
            "explicit support-oracle live authorization token required"
        )
    outcome = execute_live(args.run_id)
    print(outcome["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LIVE_AUTHORIZATION",
    "RUN_ROOT",
    "WORK_ROOT",
    "AirfoilV9SupportOracleRunnerError",
    "SupportOracleRunnerDependencies",
    "execute_live",
    "main",
    "readiness",
]

"""Provider/CFD-free tests for the durable Airfoil-v7 finite oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_evolve.agentic import (
    AgenticBenchmark,
    DetailedEvaluationPayload,
    FileExclusiveResourceLease,
    ResourceLeaseUnavailable,
    artifact_ref_for_bytes,
)
from examples.benchmarks.engibench_airfoil import v7_finite_oracle as oracle
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AIRFOIL_V7_REWARD_BINDING,
    AirfoilV7PhenotypeIdentityPolicy,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AirfoilV7Problem,
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7ShapeVariationCatalog,
    AirfoilV7TrimVariationCatalog,
    AirfoilV7UnionVariationCatalog,
)


class _ForbiddenRawProblem:
    def evaluate_raw(self, config: object) -> None:
        del config
        raise AssertionError("fake oracle problem must use its injected evaluator")


class _ExpectedInterruption(RuntimeError):
    pass


def _fake_source() -> dict[str, object]:
    return {"schema_version": 1, "sha256": "a" * 64, "files": {}}


def _fake_report_binding(path: Path, source_sha256: str) -> dict[str, object]:
    content = path.resolve(strict=True).read_bytes()
    return {
        "kind": "focused_prelaunch_verification_report",
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "validated_report": {
            "schema_version": 1,
            "status": "pass",
            "source_snapshot_sha256": source_sha256,
            "commands": [],
            "environment": {},
        },
    }


class _FakeEvaluator:
    evaluator_identity = EVALUATOR_IDENTITY

    def __init__(
        self,
        run_dir: Path,
        order: dict[str, int],
        calls: list[str],
    ) -> None:
        self._receipt_root = run_dir / "raw_receipts" / f"fake-{len(calls):03d}"
        self._receipt_root.mkdir(parents=True, exist_ok=True)
        self._order = order
        self._calls = calls

    def evaluate_evidence(self, configuration: dict[str, object]) -> DetailedEvaluationPayload:
        key = candidate_sha256(configuration)
        ordinal = self._order[key]
        self._calls.append(key)
        record = {
            "candidate_sha256": key,
            "evaluator_calls": 3,
            "f": 1.0 + ordinal / 10_000,
            "v": 1.0 - ordinal / 1_000,
        }
        path = self._receipt_root / f"{key}.json"
        oracle.write_json_atomic(path, record)
        content = path.read_bytes()
        return DetailedEvaluationPayload(
            failure=None,
            objectives=((OBJECTIVE_NAME, float(record["f"])),),
            violations=((VIOLATION_NAME, float(record["v"])),),
            checks=(),
            receipt=artifact_ref_for_bytes(content, media_type="application/json"),
            evaluator=EVALUATOR_IDENTITY,
            active_wall_seconds=1.0,
            resource_queue_wall_seconds=None,
        )


def _fake_replay(path: Path, configuration: object) -> DetailedEvaluationPayload:
    record = json.loads(path.read_bytes())
    assert record["candidate_sha256"] == candidate_sha256(configuration)
    content = path.read_bytes()
    return DetailedEvaluationPayload(
        failure=None,
        objectives=((OBJECTIVE_NAME, float(record["f"])),),
        violations=((VIOLATION_NAME, float(record["v"])),),
        checks=(),
        receipt=artifact_ref_for_bytes(content, media_type="application/json"),
        evaluator=EVALUATOR_IDENTITY,
        active_wall_seconds=1.0,
        resource_queue_wall_seconds=None,
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(oracle, "_validation_report_binding", _fake_report_binding)
    report = tmp_path / "verification.json"
    report.write_text("{}\n", encoding="utf-8")
    run_id = "oracle_test"
    run_dir = tmp_path / run_id
    manifest = tmp_path / "manifest.json"
    oracle.write_oracle_manifest(
        manifest,
        run_id=run_id,
        output_dir=run_dir,
        verification_report_path=report,
        source_snapshot_factory=_fake_source,
        enforce_canonical_output=False,
    )
    contract, _ = oracle._contract_binding()
    order = {
        candidate_sha256(oracle.thaw_json(option.child_configuration)): ordinal
        for ordinal, option in enumerate(contract.options, start=1)
    }
    calls: list[str] = []

    def benchmark_factory(run_id_value: str, output: Path) -> AgenticBenchmark:
        assert run_id_value == run_id
        problem = AirfoilV7Problem(raw_problem=_ForbiddenRawProblem())
        evaluator = _FakeEvaluator(output, order, calls)
        return AgenticBenchmark(
            problem=problem,
            reward=AIRFOIL_V7_REWARD_BINDING,
            detailed_evaluator=evaluator,
            outcome_relation=AIRFOIL_V7_ARCHIVE_RELATION,
            phenotype_identity=AirfoilV7PhenotypeIdentityPolicy(),
            finite_variation_catalogs=(
                AirfoilV7ShapeVariationCatalog(),
                AirfoilV7TrimVariationCatalog(),
                AirfoilV7UnionVariationCatalog(),
            ),
        )

    def lease_factory(owner: str, phase: str) -> FileExclusiveResourceLease:
        return FileExclusiveResourceLease(
            resource_key="airfoil_oracle_test",
            owner_id=owner,
            lease_path=tmp_path / "oracle.lock",
            owner_metadata={"phase": phase},
        )

    dependencies = oracle.OracleExecutionDependencies(
        benchmark_factory=benchmark_factory,
        resource_lease_factory=lease_factory,
        source_snapshot_factory=_fake_source,
        receipt_replayer=_fake_replay,
        monotonic_ns=iter(range(0, 1_000_000)).__next__,
        enforce_canonical_output=False,
    )
    return manifest, run_dir, calls, dependencies


def test_exact_manifest_contains_complete_child_bytes_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _, _ = _fixture(tmp_path, monkeypatch)
    record = json.loads(manifest.read_bytes())
    order = record["oracle"]["catalog"]["evaluation_order"]
    assert len(order) == 80
    assert len({row["raw_candidate_sha256"] for row in order}) == 80
    assert all(type(row["child_configuration"]) is dict for row in order)
    record["oracle"]["catalog"]["evaluation_order"][0]["option_id"] = "tampered"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(oracle.OracleContractError, match="self-hash"):
        oracle.verify_oracle_manifest(
            manifest,
            require_output_absent=True,
            source_snapshot_factory=_fake_source,
            enforce_canonical_output=False,
        )


def test_run_lock_rejects_reentry_before_run_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, run_dir, calls, dependencies = _fixture(tmp_path, monkeypatch)
    verified = oracle.verify_oracle_manifest(
        manifest,
        require_output_absent=True,
        source_snapshot_factory=_fake_source,
        enforce_canonical_output=False,
    )
    blocker = oracle._new_run_lock(verified)
    blocker.acquire()
    try:
        with pytest.raises(ResourceLeaseUnavailable):
            oracle.execute_oracle(
                manifest,
                resume=False,
                dependencies=dependencies,
            )
        assert not run_dir.exists()
        assert calls == []
    finally:
        blocker.release(outcome="test_complete")


def test_interrupted_open_start_recovers_receipt_and_finishes_exactly_80(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, run_dir, calls, dependencies = _fixture(tmp_path, monkeypatch)
    interrupted = False

    def interrupt(path: Path, option: object) -> None:
        nonlocal interrupted
        del path, option
        if len(calls) == 5 and not interrupted:
            interrupted = True
            raise _ExpectedInterruption("crash after durable receipt")

    first = oracle.OracleExecutionDependencies(
        benchmark_factory=dependencies.benchmark_factory,
        resource_lease_factory=dependencies.resource_lease_factory,
        source_snapshot_factory=dependencies.source_snapshot_factory,
        receipt_replayer=dependencies.receipt_replayer,
        monotonic_ns=dependencies.monotonic_ns,
        after_raw_receipt=interrupt,
        enforce_canonical_output=False,
    )
    with pytest.raises(_ExpectedInterruption):
        oracle.execute_oracle(manifest, resume=False, dependencies=first)
    assert len(calls) == 5
    result = oracle.execute_oracle(
        manifest,
        resume=True,
        dependencies=dependencies,
    )
    assert len(calls) == 80
    assert result["candidate_attempts"] == 80
    assert result["raw_solver_calls"] == 240
    assert result["provider_calls"] == 0
    assert result["credentials_read"] is False
    assert result["three_action_portfolios"]["combination_count"] == 82_160
    assert len(result["results"]) == 80
    known_a = next(
        row for row in result["known_action_results"] if row["arm"] == "A"
    )
    assert "normalized_multipoint_drag" not in known_a
    assert known_a["prior_objectives"][OBJECTIVE_NAME] == pytest.approx(
        oracle.KNOWN_ACTIONS[0][OBJECTIVE_NAME]
    )
    assert known_a["fresh_objectives"][OBJECTIVE_NAME] != known_a[
        "prior_objectives"
    ][OBJECTIVE_NAME]
    masses = result["exact_uniform_one_action"]["mass_functions"]
    assert masses["overall"]["denominator"] == 80
    assert masses["shape_only"]["denominator"] == 16
    assert masses["trim_only"]["denominator"] == 64
    assert sum(
        masses["overall"]["contextual_parent_reward"]["counts"].values()
    ) == 80
    a_comparison = result["three_action_portfolios"]["comparisons"]["A"]
    assert sum(a_comparison["counts"].values()) == 82_160
    assert sum(a_comparison["probabilities"].values()) == pytest.approx(1.0)
    assert (
        result["three_action_portfolios"]["observed_asn_percentile_definition"]
        == "fraction_of_portfolios_with_strictly_better_best_rank;zero_is_best"
    )
    assert set(result["prospective_decisions"]["card_local_stability"]) == {
        "A",
        "S",
    }
    assert all(
        "fresh_resolved_directions" in row
        and "contextual_reward_reproduced" in row
        for row in result["fresh_repeat_audit"]
    )
    recovered = json.loads(
        next(
            path
            for path in (run_dir / "options").glob("005-*/terminal.json")
        ).read_bytes()
    )
    assert recovered["recovered_after_interruption"] is True
    finalized_path = run_dir / "finalized.json"
    assert finalized_path.is_file()
    finalized = json.loads(finalized_path.read_bytes())
    assert finalized["recursive_file_count"] == len(finalized["files"])
    assert (
        oracle._finalize_run(run_dir, status="completed_80_action_oracle")
        == finalized
    )

    result_path = run_dir / "oracle_result.json"
    original_result = result_path.read_bytes()
    result_path.write_bytes(original_result + b"\n")
    with pytest.raises(oracle.OracleContractError, match="recursive content"):
        oracle._finalize_run(run_dir, status="completed_80_action_oracle")
    result_path.write_bytes(original_result)

    oracle.write_json_atomic(run_dir / "unsealed-extra.json", {"unexpected": True})
    with pytest.raises(oracle.OracleContractError, match="recursive content"):
        oracle._finalize_run(run_dir, status="completed_80_action_oracle")


def test_open_start_without_receipt_invalidates_and_cannot_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, run_dir, _, dependencies = _fixture(tmp_path, monkeypatch)

    def interrupt_before_receipt(path: Path, option: object) -> None:
        del option
        path.unlink()
        raise _ExpectedInterruption("receipt lost after attempt charge")

    first = oracle.OracleExecutionDependencies(
        benchmark_factory=dependencies.benchmark_factory,
        resource_lease_factory=dependencies.resource_lease_factory,
        source_snapshot_factory=dependencies.source_snapshot_factory,
        receipt_replayer=dependencies.receipt_replayer,
        monotonic_ns=dependencies.monotonic_ns,
        after_raw_receipt=interrupt_before_receipt,
        enforce_canonical_output=False,
    )
    with pytest.raises(_ExpectedInterruption):
        oracle.execute_oracle(manifest, resume=False, dependencies=first)
    with pytest.raises(oracle.OracleRunInvalidated, match="no durable receipt"):
        oracle.execute_oracle(manifest, resume=True, dependencies=dependencies)
    assert (run_dir / "invalidation.json").is_file()
    with pytest.raises(oracle.OracleRunInvalidated, match="cannot resume"):
        oracle.execute_oracle(manifest, resume=True, dependencies=dependencies)

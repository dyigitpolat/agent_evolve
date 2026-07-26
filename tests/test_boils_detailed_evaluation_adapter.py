"""Provider-free gates for the generic BOiLS detailed-evaluation vertical."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path

from agent_evolve.agentic import (
    AgenticEvolutionEngine,
    DetailedEvaluationAdapter,
    DeterministicIdFactory,
    EvaluationCheckStatus,
    FailureCategory,
    FailureCode,
    InsightMemoryBank,
    OutcomeOrderingKind,
)
from agent_evolve.application.decision_metric_projection import (
    project_candidate_decision_metrics,
)
from agent_evolve.infrastructure.artifacts.in_memory import InMemoryArtifactStore
from agent_evolve.ports.artifact_store import read_json
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE, config_sha256
from examples.benchmarks.boils_abc.detailed_evaluation import (
    BOILS_EXECUTABLE_ACTIONS_SHA256,
    TOTAL_LEVELS,
    TOTAL_LUT_COUNT,
    BoilsDetailedEvaluationAdapter,
    boils_evaluator_context_record,
    boils_evaluator_identity,
    compose_boils_scientific_workload,
    create_current_sqrt_workload,
)
from examples.benchmarks.boils_abc.evaluator import (
    LUT_INPUTS,
    AbcEvaluationError,
    AbcEvaluatorSettings,
    BoilsEvaluation,
    CircuitDiagnostics,
    CircuitEvaluation,
    CircuitSpec,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _settings(tmp_path: Path) -> AbcEvaluatorSettings:
    return AbcEvaluatorSettings(
        abc_binary=tmp_path / "abc-not-opened",
        expected_abc_sha256=_sha("fake-abc"),
        circuits=(
            CircuitSpec(
                name="sqrt",
                source=tmp_path / "sqrt-not-opened.blif",
                expected_sha256=_sha("fake-sqrt"),
            ),
        ),
        abc_source_identity="git:fake-abc-source",
        circuit_suite_identity="git:fake-circuit-suite",
        per_circuit_timeout_s=60.0,
        work_root=tmp_path / "work-not-opened",
        max_diagnostic_chars=4_096,
    )


def _diagnostics(
    *,
    status: str = "passed",
    equivalent: bool = True,
    returncode: int | None = 0,
) -> CircuitDiagnostics:
    stdout = "top: i/o = 128/ 64 nd = 4607 edge = 21262 aig = 34079 lev = 1005"
    return CircuitDiagnostics(
        status=status,
        returncode=returncode,
        elapsed_s=24.5,
        timeout_s=60.0,
        equivalent=equivalent,
        error_signatures=(),
        stdout_excerpt=stdout,
        stderr_excerpt="",
        stdout_sha256=hashlib.sha256(stdout.encode("ascii")).hexdigest(),
        stderr_sha256=hashlib.sha256(b"").hexdigest(),
        abc_program="read source.blif; strash; if -K 6; print_stats; cec;",
        argv=("/fixture/abc", "-c", "fixture"),
        cpu_affinity=None,
    )


def _observation(
    settings: AbcEvaluatorSettings,
    configuration: object,
    *,
    total_lut_count: int = 4_607,
) -> BoilsEvaluation:
    circuit = CircuitEvaluation(
        circuit_name="sqrt",
        circuit_sha256=settings.circuits[0].expected_sha256,
        inputs=128,
        outputs=64,
        lut_count=4_607,
        edge_count=21_262,
        aig_count=34_079,
        levels=1_005,
        diagnostics=_diagnostics(),
    )
    return BoilsEvaluation(
        configuration_sha256=config_sha256(configuration),
        sequence=tuple(configuration["sequence"]),
        abc_binary_sha256=settings.expected_abc_sha256,
        lut_inputs=LUT_INPUTS,
        circuit_results=(circuit,),
        total_lut_count=total_lut_count,
        total_levels=1_005,
        max_levels=1_005,
        elapsed_s=24.75,
        affinity_queue_wait_s=0.125,
        cpu_affinity=None,
    )


class _FakeEvaluator:
    def __init__(
        self,
        settings: AbcEvaluatorSettings,
        *,
        malformed_total: bool = False,
        abc_failure: bool = False,
    ) -> None:
        self.settings = settings
        self.malformed_total = malformed_total
        self.abc_failure = abc_failure
        self.calls = 0

    def evaluate(self, config: object) -> BoilsEvaluation:
        self.calls += 1
        if self.abc_failure:
            raise AbcEvaluationError(
                "sqrt",
                _diagnostics(
                    status="cec_failed_or_missing",
                    equivalent=False,
                ),
            )
        return _observation(
            self.settings,
            config,
            total_lut_count=4_608 if self.malformed_total else 4_607,
        )


class _ForbiddenGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("seed registration must not call a model")

    async def reflect(self, request):
        del request
        raise AssertionError("seed registration must not reflect")


def _configuration() -> dict[str, object]:
    return {"sequence": list(DEFAULT_ACTION_SEQUENCE)}


def test_evaluator_identity_is_portable_but_binds_observable_semantics(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    identity = boils_evaluator_identity(settings)
    context = boils_evaluator_context_record(settings)

    assert BOILS_EXECUTABLE_ACTIONS_SHA256 == (
        "eb913f86b325fa19d7ecb47832de435bb56083c2a010529611856b4734521dd0"
    )
    assert identity.evaluator_context_sha256 == (
        "2683c4c3f2da15eb73a6fa8a4ddf1c1d42678091f1af56b7e4f49c8046acd721"
    )
    assert context["evaluation_contract"]["ordered_circuits"] == [
        {"name": "sqrt", "source_sha256": _sha("fake-sqrt")}
    ]

    relocated = replace(
        settings,
        abc_binary=tmp_path / "elsewhere" / "abc",
        circuits=(
            replace(settings.circuits[0], source=tmp_path / "elsewhere" / "sqrt.blif"),
        ),
        work_root=tmp_path / "another-work-root",
        affinity_sets=((23,),),
    )
    assert boils_evaluator_identity(relocated) == identity
    assert boils_evaluator_identity(
        replace(settings, per_circuit_timeout_s=61.0)
    ) != identity
    assert boils_evaluator_identity(
        replace(
            settings,
            circuits=(
                replace(settings.circuits[0], expected_sha256=_sha("changed-sqrt")),
            ),
        )
    ) != identity


def test_current_sqrt_factory_freezes_exact_identities_without_opening_files(
    tmp_path: Path,
) -> None:
    workload = create_current_sqrt_workload(
        artifact_store=InMemoryArtifactStore(),
        affinity_sets=((127,),),
        per_circuit_timeout_s=60.0,
        cache_root=tmp_path / "intentionally-absent-cache",
    )
    adapter = workload.benchmark.detailed_evaluator
    assert type(adapter) is BoilsDetailedEvaluationAdapter
    assert adapter.evaluator_identity.evaluator_context_sha256 == (
        "49dbb10719b2dbe1e4b304bd8bc8e0035ab2429faf71ecb145513eff96b1692c"
    )
    assert workload.decision_metrics.definition_sha256 == (
        "53dc96ef3b9e98038d284d1636f101b4b50f74a5cc12e35a123a6cb10a975824"
    )
    context = boils_evaluator_context_record(adapter.problem.settings)
    assert context["implementation_provenance"]["abc_binary_sha256"] == (
        "21f3673079a1ea21378b817e5035a3a008ffc76e2656d8739906d059a7928232"
    )
    assert context["evaluation_contract"]["ordered_circuits"] == [
        {
            "name": "sqrt",
            "source_sha256": (
                "7c5a28925fb2a6b3f1d0979ceaa93eafabea39fa418ec717e09cb4ff3b882107"
            ),
        }
    ]


def test_success_projection_persists_complete_receipt_and_exact_metrics(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    evaluator = _FakeEvaluator(settings)
    store = InMemoryArtifactStore()
    workload = compose_boils_scientific_workload(
        settings,
        artifact_store=store,
        evaluator=evaluator,
    )
    adapter = workload.benchmark.detailed_evaluator
    assert isinstance(adapter, DetailedEvaluationAdapter)
    assert type(adapter) is BoilsDetailedEvaluationAdapter
    assert evaluator.calls == 0

    payload = adapter.evaluate_evidence(_configuration())

    assert evaluator.calls == 1
    assert payload.failure is None
    assert payload.objectives == (
        (TOTAL_LUT_COUNT, 4_607.0),
        (TOTAL_LEVELS, 1_005.0),
    )
    assert payload.violations == ()
    assert payload.active_wall_seconds == 24.75
    assert payload.resource_queue_wall_seconds == 0.125
    assert tuple(check.name for check in payload.checks) == (
        "abc_provenance",
        "cec_equivalence",
        "configuration_identity",
        "objective_projection",
    )
    assert all(check.status is EvaluationCheckStatus.PASS for check in payload.checks)
    assert payload.receipt is not None
    receipt = read_json(store, payload.receipt.artifact_id)
    assert receipt["status"] == "passed"
    assert receipt["evaluation"]["configuration_sha256"] == config_sha256(
        _configuration()
    )
    assert receipt["evaluation"]["circuit_results"][0]["diagnostics"][
        "equivalent"
    ] is True
    assert receipt["evaluation"]["total_lut_count"] == 4_607

    semantics = workload.benchmark.optimization_semantics
    assert semantics is not None
    assert semantics.outcome_ordering.kind is OutcomeOrderingKind.PARETO
    assert workload.decision_metrics.objective_only_legacy_metric_ids is True
    assert workload.decision_metrics.metric_ids == (
        TOTAL_LEVELS,
        TOTAL_LUT_COUNT,
    )


def test_engine_seed_and_decision_projection_cross_the_same_generic_boundary(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    evaluator = _FakeEvaluator(settings)
    workload = compose_boils_scientific_workload(
        settings,
        artifact_store=InMemoryArtifactStore(),
        evaluator=evaluator,
    )
    ids = DeterministicIdFactory("boils_sqrt_detailed_vertical")
    benchmark = workload.benchmark
    engine = AgenticEvolutionEngine(
        problem=benchmark.problem,
        generator=_ForbiddenGenerator(),
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=17,
        detailed_evaluator=benchmark.detailed_evaluator,
        outcome_relation_binding=benchmark.outcome_relation,
        optimization_semantics=benchmark.optimization_semantics,
    )

    candidate = asyncio.run(
        engine.register_seed(_configuration(), label="boils_sqrt_seed")
    )
    projected = project_candidate_decision_metrics(
        candidate,
        workload.decision_metrics,
    )

    assert candidate.valid is True
    assert candidate.detailed_evaluation is not None
    assert candidate.detailed_evaluation.success is True
    assert projected.metric_map == {
        TOTAL_LEVELS: 1_005.0,
        TOTAL_LUT_COUNT: 4_607.0,
    }
    assert projected.projection_definition_sha256 == (
        workload.decision_metrics.definition_sha256
    )
    assert evaluator.calls == 1


def test_projection_fails_closed_on_raw_total_drift(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    evaluator = _FakeEvaluator(settings, malformed_total=True)
    store = InMemoryArtifactStore()
    workload = compose_boils_scientific_workload(
        settings,
        artifact_store=store,
        evaluator=evaluator,
    )

    payload = workload.benchmark.detailed_evaluator.evaluate_evidence(
        _configuration()
    )

    assert payload.failure is not None
    assert payload.failure.category is FailureCategory.SYSTEM
    assert payload.failure.code is FailureCode.EVALUATOR_CONTRACT_VIOLATION
    assert payload.objectives == ()
    assert payload.receipt is not None
    assert read_json(store, payload.receipt.artifact_id)["evaluation"][
        "total_lut_count"
    ] == 4_608


def test_expected_abc_rejection_is_a_receipted_candidate_failure(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    evaluator = _FakeEvaluator(settings, abc_failure=True)
    store = InMemoryArtifactStore()
    workload = compose_boils_scientific_workload(
        settings,
        artifact_store=store,
        evaluator=evaluator,
    )

    payload = workload.benchmark.detailed_evaluator.evaluate_evidence(
        _configuration()
    )

    assert payload.failure is not None
    assert payload.failure.category is FailureCategory.CANDIDATE
    assert payload.failure.code is FailureCode.EVALUATOR_DECLARED_INFEASIBLE
    assert payload.receipt is not None
    receipt = read_json(store, payload.receipt.artifact_id)
    assert receipt["status"] == "failed"
    assert receipt["failed_circuit_name"] == "sqrt"
    assert receipt["diagnostics"]["status"] == "cec_failed_or_missing"


def test_schema_failure_never_calls_abc_or_fabricates_a_receipt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    evaluator = _FakeEvaluator(settings)
    workload = compose_boils_scientific_workload(
        settings,
        artifact_store=InMemoryArtifactStore(),
        evaluator=evaluator,
    )

    payload = workload.benchmark.detailed_evaluator.evaluate_evidence(
        {"sequence": ["balance"]}
    )

    assert payload.failure is not None
    assert payload.failure.code is FailureCode.SCHEMA_INVALID
    assert payload.receipt is None
    assert evaluator.calls == 0

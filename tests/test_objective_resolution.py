from __future__ import annotations

import asyncio
import hashlib

import pytest
from pydantic import BaseModel, ConfigDict

import agent_evolve
from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.agentic_evolution import AgenticEvolutionEngine
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluationPayload,
    EvaluatorIdentity,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.outcome_relation import (
    OutcomeRelation,
    objective_pareto_outcome_binding,
)
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports import ExactObjectiveResolution
from agent_evolve.ports.objective_resolution import (
    ObjectiveResolutionRequest,
    ObjectiveResolutionResult,
    resolve_objectives,
)


OBJECTIVES = (
    ObjectiveSpec("quality", "max"),
    ObjectiveSpec("cost", "min"),
)


def _definition(name: str) -> str:
    return hashlib.sha256(f"test:objective-resolution:{name}:v1".encode()).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _request() -> ObjectiveResolutionRequest:
    return ObjectiveResolutionRequest(
        configuration=_object({"digits": 2, "design": "probe"}),
        objectives=OBJECTIVES,
        raw_objectives=(("quality", 1.234), ("cost", 2.346)),
    )


class _DigitsResolution:
    policy_id = "test_digits_resolution"
    policy_version = 1
    definition_sha256 = _definition("digits")

    def resolve(
        self,
        request: ObjectiveResolutionRequest,
    ) -> ObjectiveResolutionResult:
        configuration = thaw_json(request.configuration)
        assert type(configuration) is dict
        digits = configuration["digits"]
        assert type(digits) is int
        return ObjectiveResolutionResult(
            request_sha256=request.request_sha256,
            decision_objectives=tuple(
                (metric_id, float(round(value, digits)))
                for metric_id, value in request.raw_objectives
            ),
            evidence=_object({"digits": digits}),
        )


class _NonDeterministicResolution:
    policy_id = "test_nondeterministic_resolution"
    policy_version = 1
    definition_sha256 = _definition("nondeterministic")

    def __init__(self) -> None:
        self.calls = 0

    def resolve(
        self,
        request: ObjectiveResolutionRequest,
    ) -> ObjectiveResolutionResult:
        self.calls += 1
        return ObjectiveResolutionResult(
            request_sha256=request.request_sha256,
            decision_objectives=tuple(
                (metric_id, value + self.calls / 1_000.0)
                for metric_id, value in request.raw_objectives
            ),
        )


class _NonIdempotentResolution:
    policy_id = "test_nonidempotent_resolution"
    policy_version = 1
    definition_sha256 = _definition("nonidempotent")

    def resolve(
        self,
        request: ObjectiveResolutionRequest,
    ) -> ObjectiveResolutionResult:
        return ObjectiveResolutionResult(
            request_sha256=request.request_sha256,
            decision_objectives=tuple(
                (metric_id, value + 1.0)
                for metric_id, value in request.raw_objectives
            ),
        )


def test_resolution_is_configuration_aware_deterministic_and_idempotent() -> None:
    receipt = resolve_objectives(_DigitsResolution(), _request())
    repeated = resolve_objectives(_DigitsResolution(), _request())

    assert receipt.raw_objectives == (("quality", 1.234), ("cost", 2.346))
    assert receipt.decision_objectives == (("quality", 1.23), ("cost", 2.35))
    assert thaw_json(receipt.evidence) == {"digits": 2}
    assert receipt.receipt_sha256 == repeated.receipt_sha256

    with pytest.raises(ValueError, match="deterministic"):
        resolve_objectives(_NonDeterministicResolution(), _request())
    with pytest.raises(ValueError, match="idempotent"):
        resolve_objectives(_NonIdempotentResolution(), _request())


def test_exact_resolution_is_an_explicit_identity_policy() -> None:
    request = _request()
    receipt = resolve_objectives(ExactObjectiveResolution(), request)

    assert receipt.raw_objectives == request.raw_objectives
    assert receipt.decision_objectives == request.raw_objectives
    assert receipt.policy_identity == (
        agent_evolve.EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
        agent_evolve.EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
        agent_evolve.EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
    )


class _CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality: float
    cost: float
    digits: int
    design: str


class _ResolutionProblem:
    candidate_model = _CandidateModel
    objectives = OBJECTIVES

    @staticmethod
    def search_space_description() -> str:
        return "Two measured objectives and a declared decimal resolution."

    @staticmethod
    def validate(configuration: dict[str, object]) -> bool:
        return _CandidateModel.model_validate(configuration, strict=True) is not None

    @staticmethod
    def evaluate(configuration: dict[str, object]) -> dict[str, float]:
        quality = configuration["quality"]
        cost = configuration["cost"]
        assert type(quality) is float and type(cost) is float
        return {"quality": quality, "cost": cost}


class _UnusedGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("seed-only objective-resolution tests must not propose")

    async def reflect(self, request):
        del request
        raise AssertionError("seed-only objective-resolution tests must not reflect")


class _DetailedAdapter:
    evaluator_identity = EvaluatorIdentity(
        "objective_resolution_probe",
        1,
        _definition("detailed-context"),
    )

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        quality = configuration["quality"]
        cost = configuration["cost"]
        assert type(quality) is float and type(cost) is float
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(("quality", quality), ("cost", cost)),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


def _engine(*, resolution=None, traces=None, detailed=False) -> AgenticEvolutionEngine:
    ids = DeterministicIdFactory("objective_resolution")
    adapter = _DetailedAdapter() if detailed else None
    return AgenticEvolutionEngine(
        problem=_ResolutionProblem(),
        generator=_UnusedGenerator(),
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=7,
        trace_sink=None if traces is None else traces.append,
        objective_resolution=resolution,
        detailed_evaluator=adapter,
        outcome_relation_binding=(
            None if adapter is None else objective_pareto_outcome_binding(OBJECTIVES)
        ),
    )


def test_engine_preserves_raw_evidence_while_candidates_use_decision_values() -> None:
    async def scenario():
        traces: list[dict[str, object]] = []
        engine = _engine(resolution=_DigitsResolution(), traces=traces)
        candidate = await engine.register_seed(
            {
                "quality": 1.234,
                "cost": 2.346,
                "digits": 2,
                "design": "resolved",
            },
            label="resolved",
        )
        return candidate, traces

    candidate, traces = asyncio.run(scenario())

    assert candidate.objectives == (("quality", 1.23), ("cost", 2.35))
    assert candidate.objective_map == {"quality": 1.23, "cost": 2.35}
    assert candidate.raw_objective_map == {"quality": 1.234, "cost": 2.346}
    assert candidate.objective_resolution_receipt is not None
    seed_event = next(
        event for event in traces if event["event_type"] == "seed_registered"
    )
    assert seed_event["objectives"] == {"quality": 1.23, "cost": 2.35}
    resolution = seed_event["objective_resolution"]
    assert type(resolution) is dict
    assert resolution["raw_objectives"] == [
        {"metric_id": "quality", "value_hex": 1.234.hex()},
        {"metric_id": "cost", "value_hex": 2.346.hex()},
    ]


def test_default_engine_trace_and_candidate_keep_exact_legacy_projection() -> None:
    async def scenario():
        traces: list[dict[str, object]] = []
        engine = _engine(traces=traces)
        candidate = await engine.register_seed(
            {
                "quality": 1.234,
                "cost": 2.346,
                "digits": 2,
                "design": "legacy",
            },
            label="legacy",
        )
        return candidate, traces

    candidate, traces = asyncio.run(scenario())

    assert candidate.objectives == (("quality", 1.234), ("cost", 2.346))
    assert candidate.raw_objective_map == candidate.objective_map
    assert candidate.objective_resolution_receipt is None
    seed_event = next(
        event for event in traces if event["event_type"] == "seed_registered"
    )
    assert "objective_resolution" not in seed_event


def test_detailed_evaluation_remains_raw_when_decision_values_are_resolved() -> None:
    async def scenario():
        engine = _engine(resolution=_DigitsResolution(), detailed=True)
        left = await engine.register_seed(
            {
                "quality": 1.234,
                "cost": 2.346,
                "digits": 2,
                "design": "detailed",
            },
            label="detailed",
        )
        right = await engine.register_seed(
            {
                "quality": 1.233,
                "cost": 2.347,
                "digits": 2,
                "design": "detailed_jitter",
            },
            label="detailed_jitter",
        )
        return engine, left, right

    engine, candidate, jittered = asyncio.run(scenario())

    assert candidate.objectives == (("quality", 1.23), ("cost", 2.35))
    assert candidate.detailed_evaluation is not None
    assert candidate.detailed_evaluation.objectives == (
        ("quality", 1.234),
        ("cost", 2.346),
    )
    assert candidate.objective_resolution_receipt is not None
    assert candidate.objective_resolution_receipt.raw_objectives == (
        candidate.detailed_evaluation.objectives
    )
    assert jittered.objectives == candidate.objectives
    assert engine.compare_candidates(candidate, jittered) is OutcomeRelation.EQUIVALENT


def test_canonical_decision_values_collapse_pareto_jitter() -> None:
    async def scenario():
        engine = _engine(resolution=_DigitsResolution())
        left = await engine.register_seed(
            {
                "quality": 1.0001,
                "cost": 2.0001,
                "digits": 3,
                "design": "left",
            },
            label="left",
        )
        right = await engine.register_seed(
            {
                "quality": 1.0002,
                "cost": 2.0002,
                "digits": 3,
                "design": "right",
            },
            label="right",
        )
        return left, right

    left, right = asyncio.run(scenario())
    archive = ParetoArchive(OBJECTIVES)
    archive.consider(left)
    archive.consider(right)

    assert left.raw_objective_map != right.raw_objective_map
    assert left.objectives == right.objectives == (
        ("quality", 1.0),
        ("cost", 2.0),
    )
    assert len(archive.front) == 1


def test_benchmark_binding_and_public_exports_expose_the_generic_port() -> None:
    resolution = _DigitsResolution()
    benchmark = AgenticBenchmark(
        problem=_ResolutionProblem(),
        objective_resolution=resolution,
    )

    benchmark.validate_binding()
    assert benchmark.objective_resolution is resolution
    assert agent_evolve.ExactObjectiveResolution is ExactObjectiveResolution
    assert agent_evolve.ObjectiveResolutionRequest is ObjectiveResolutionRequest
    assert callable(agent_evolve.resolve_objectives)

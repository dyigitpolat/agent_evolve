from __future__ import annotations

import asyncio

import pytest

from agent_evolve.application.agentic_evolution import AgenticEvolutionEngine
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.typed_json import thaw_json, typed_json_sha256
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
    SemanticProjectionPhenotypeIdentityPolicy,
)


class _UnusedGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("seed-only phenotype tests must not propose")

    async def reflect(self, request):
        del request
        raise AssertionError("seed-only phenotype tests must not reflect")


class _AliasProblem:
    objectives = (ObjectiveSpec("value", "min"),)

    def __init__(self) -> None:
        self.evaluations = 0

    @staticmethod
    def search_space_description() -> str:
        return "An integer value plus evaluator-inert spelling metadata."

    @staticmethod
    def validate(configuration) -> bool:
        return (
            type(configuration) is dict
            and type(configuration.get("value")) is int
            and type(configuration.get("spelling")) is str
        )

    def evaluate(self, configuration):
        self.evaluations += 1
        return {"value": float(configuration["value"])}


def _engine(*, problem, policy=None, traces=None) -> AgenticEvolutionEngine:
    ids = DeterministicIdFactory("engine_phenotype_identity")
    return AgenticEvolutionEngine(
        problem=problem,
        generator=_UnusedGenerator(),
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=7,
        phenotype_identity_policy=policy,
        trace_sink=None if traces is None else traces.append,
    )


def _semantic_value_policy() -> SemanticProjectionPhenotypeIdentityPolicy:
    def project(configuration):
        thawed = thaw_json(configuration)
        assert type(thawed) is dict
        return {"value": thawed["value"]}

    return SemanticProjectionPhenotypeIdentityPolicy(
        policy_id="alias_value_semantics",
        policy_version=1,
        projector=project,
    )


def test_default_exact_cache_behavior_and_trace_remain_backward_compatible() -> None:
    async def scenario():
        problem = _AliasProblem()
        traces: list[dict[str, object]] = []
        engine = _engine(problem=problem, traces=traces)
        left_config = {"value": 3, "spelling": "three"}
        right_config = {"spelling": "III", "value": 3}
        left = await engine.register_seed(left_config, label="left")
        right = await engine.register_seed(right_config, label="right")
        duplicate = await engine.register_seed(left_config, label="left_again")
        return (
            problem,
            traces,
            engine,
            left,
            right,
            duplicate,
            await engine.evaluation_cache_snapshot(),
        )

    problem, traces, engine, left, right, duplicate, snapshot = asyncio.run(scenario())

    assert problem.evaluations == 2
    assert snapshot["misses"] == 2
    assert snapshot["hits"] == 1
    assert left.occurrence.configuration_hash != right.occurrence.configuration_hash
    assert (
        left.occurrence.configuration_hash
        == duplicate.occurrence.configuration_hash
        == typed_json_sha256(left.configuration)
    )
    assert left.occurrence.candidate_id != duplicate.occurrence.candidate_id

    cache_events = [
        event for event in traces if event["event_type"] == "evaluation_cache_event"
    ]
    assert [event["cache_event_type"] for event in cache_events] == [
        "miss",
        "miss",
        "hit",
    ]
    assert [event["configuration_hash"] for event in cache_events] == [
        left.occurrence.configuration_hash,
        right.occurrence.configuration_hash,
        left.occurrence.configuration_hash,
    ]
    for event in cache_events:
        identity = event["phenotype_identity"]
        assert type(identity) is dict
        assert identity["policy_id"] == "typed_configuration_phenotype"
        assert identity["policy_version"] == 1
        assert identity["value_sha256"] == event["configuration_hash"]
        assert identity["metadata_complete"] is True
        assert len(identity["identity_sha256"]) == 64

    # Identification is a read-only projection and accepts either an exact
    # occurrence-bearing candidate or its configuration.
    before = snapshot
    assert engine.identify_phenotype(left) == engine.identify_phenotype(
        left.configuration
    )
    after = asyncio.run(engine.evaluation_cache_snapshot())
    assert after == before


def test_injected_semantic_identity_coalesces_physical_work_not_occurrences() -> None:
    async def scenario():
        problem = _AliasProblem()
        traces: list[dict[str, object]] = []
        engine = _engine(
            problem=problem,
            policy=_semantic_value_policy(),
            traces=traces,
        )
        left = await engine.register_seed(
            {"value": 3, "spelling": "three"}, label="left"
        )
        right = await engine.register_seed(
            {"spelling": "III", "value": 3}, label="right"
        )
        return (
            problem,
            traces,
            engine,
            left,
            right,
            await engine.evaluation_cache_snapshot(),
        )

    problem, traces, engine, left, right, snapshot = asyncio.run(scenario())

    assert problem.evaluations == 1
    assert snapshot["misses"] == 1
    assert snapshot["hits"] == 1
    assert left.occurrence.candidate_id != right.occurrence.candidate_id
    assert left.occurrence.configuration_hash != right.occurrence.configuration_hash
    assert engine.identify_phenotype(left) == engine.identify_phenotype(right)

    cache_events = [
        event for event in traces if event["event_type"] == "evaluation_cache_event"
    ]
    assert [event["cache_event_type"] for event in cache_events] == ["miss", "hit"]
    assert all("configuration_hash" not in event for event in cache_events)
    assert {
        event["phenotype_identity"]["identity_sha256"] for event in cache_events
    } == {engine.identify_phenotype(left).identity_sha256}
    assert all(
        event["phenotype_identity"]["policy_id"] == "alias_value_semantics"
        for event in cache_events
    )


class _InconsistentMetadataPolicy:
    policy_id = "claimed_policy"
    policy_version = 1

    @staticmethod
    def identify(configuration):
        del configuration
        return PhenotypeIdentity("different_policy", 1, "1" * 64)


class _WrongOutputPolicy:
    policy_id = "wrong_output"
    policy_version = 1

    @staticmethod
    def identify(configuration):
        del configuration
        return {"not": "an identity"}


class _AlternatingPolicy:
    policy_id = "alternating_policy"
    policy_version = 1

    def __init__(self) -> None:
        self.calls = 0

    def identify(self, configuration):
        del configuration
        self.calls += 1
        return PhenotypeIdentity(
            self.policy_id, self.policy_version, f"{self.calls:064x}"
        )


@pytest.mark.parametrize(
    ("policy", "error_type", "message"),
    (
        (
            _InconsistentMetadataPolicy(),
            ValueError,
            "returned inconsistent metadata",
        ),
        (_WrongOutputPolicy(), TypeError, "must return exact PhenotypeIdentity"),
        (_AlternatingPolicy(), ValueError, "must be deterministic"),
    ),
)
def test_hostile_policy_outputs_fail_before_any_physical_evaluation(
    policy, error_type, message
) -> None:
    async def scenario():
        problem = _AliasProblem()
        engine = _engine(problem=problem, policy=policy)
        with pytest.raises(error_type, match=message):
            await engine.register_seed(
                {"value": 3, "spelling": "three"}, label="hostile"
            )
        return problem.evaluations, await engine.evaluation_cache_snapshot()

    evaluations, snapshot = asyncio.run(scenario())
    assert evaluations == 0
    assert snapshot["misses"] == 0


def test_invalid_policy_metadata_and_missing_identify_fail_at_construction() -> None:
    class InvalidMetadata:
        policy_id = "Invalid Policy"
        policy_version = 1

        @staticmethod
        def identify(configuration):
            del configuration
            raise AssertionError("invalid metadata must be rejected first")

    class MissingIdentify:
        policy_id = "missing_identify"
        policy_version = 1

    with pytest.raises(ValueError, match="identity policy_id"):
        _engine(problem=_AliasProblem(), policy=InvalidMetadata())
    with pytest.raises(TypeError, match="must implement identify"):
        _engine(problem=_AliasProblem(), policy=MissingIdentify())

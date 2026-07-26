from __future__ import annotations

import asyncio
import copy
from decimal import Decimal

import pytest

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    MutationContract,
    OperatorKind,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)


_PARENT = {
    "compiler": {"unroll": 2},
    "runtime": {"threads": 4, "prefetch": 4},
}


def _path(*keys: str) -> JsonPath:
    return JsonPath(tuple(ObjectKey(key) for key in keys))


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _Problem:
    objectives = (ObjectiveSpec("score", "max"),)
    candidate_model = dict

    @staticmethod
    def search_space_description() -> str:
        return "A nested compiler/runtime co-design object."

    @staticmethod
    def validate(configuration) -> bool:
        return type(configuration) is dict

    @staticmethod
    def evaluate(configuration) -> dict[str, float]:
        runtime = configuration["runtime"]
        return {"score": float(runtime["threads"] + runtime["prefetch"])}


class _FixedGenerator:
    def __init__(
        self,
        configuration: dict,
        source_attribution: tuple[SourceAttribution, ...],
    ) -> None:
        self.configuration = configuration
        self.source_attribution = source_attribution

    async def propose(self, request):
        del request
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=copy.deepcopy(self.configuration),
                design_rationale="Applied only the requested typed mutation.",
                source_attribution=self.source_attribution,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


async def _scenario(
    child: dict,
    *,
    contract: MutationContract | None,
    source_attribution: tuple[SourceAttribution, ...] = (),
):
    ids = DeterministicIdFactory("atomic_mutation")
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=_Problem(),
        generator=_FixedGenerator(child, source_attribution),
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=3,
        trace_sink=traces.append,
    )
    parent = await engine.register_seed(_PARENT, label="parent")
    outcome, = await engine.run_invocations(
        (
            InvocationPlan(
                operator_kind=OperatorKind.TYPED_MUTATION,
                parents=(parent,),
                generation=1,
                label="focused_mutation",
                allowed_top_level=("runtime",),
                mutation_contract=contract,
            ),
        )
    )
    return parent, outcome, traces


def test_exact_one_scalar_edit_passes_and_contract_is_replayable() -> None:
    child = copy.deepcopy(_PARENT)
    child["runtime"]["threads"] = 8
    contract = MutationContract((_path("runtime", "threads"),))

    _, outcome, traces = asyncio.run(
        _scenario(
            child,
            contract=contract,
            source_attribution=(
                SourceAttribution("$.runtime.threads", "mutation"),
            ),
        )
    )

    assert outcome.candidate is not None
    assert outcome.candidate.operator_compliant
    assert outcome.candidate.evidence_compliant
    assert "MACHINE MUTATION CONTRACT" in outcome.prepared.prompt
    assert "Preserve every other path exactly." in outcome.prepared.prompt
    assert '"editable_paths":["$.runtime.threads"]' in outcome.prepared.prompt
    prepared = next(
        event for event in traces if event["event_type"] == "invocation_prepared"
    )
    assert prepared["allowed_top_level"] == ["runtime"]
    assert prepared["mutation_contract"] == {
        "contract_version": 1,
        "editable_paths": ["$.runtime.threads"],
        "max_changed_paths": 1,
        "max_operations": 1,
        "allow_abstention": False,
    }


def test_bundled_two_coordinate_edit_fails_atomic_cardinality() -> None:
    child = copy.deepcopy(_PARENT)
    child["runtime"]["threads"] = 8
    child["runtime"]["prefetch"] = 8
    contract = MutationContract(
        (
            _path("runtime", "threads"),
            _path("runtime", "prefetch"),
        ),
        max_changed_paths=1,
        max_operations=2,
    )

    _, outcome, _ = asyncio.run(
        _scenario(
            child,
            contract=contract,
            source_attribution=(
                SourceAttribution("$.runtime.threads", "mutation"),
                SourceAttribution("$.runtime.prefetch", "mutation"),
            ),
        )
    )

    assert outcome.candidate is not None
    assert not outcome.candidate.operator_compliant
    assert outcome.candidate.operator_failure == (
        "mutation exceeded its changed-path cardinality"
    )


def test_patch_operation_limit_is_enforced_independently() -> None:
    child = copy.deepcopy(_PARENT)
    child["runtime"]["threads"] = 8
    child["runtime"]["prefetch"] = 8
    contract = MutationContract(
        (
            _path("runtime", "threads"),
            _path("runtime", "prefetch"),
        ),
        max_changed_paths=2,
        max_operations=1,
    )

    _, outcome, _ = asyncio.run(_scenario(child, contract=contract))

    assert outcome.candidate is not None
    assert not outcome.candidate.operator_compliant
    assert outcome.candidate.operator_failure == (
        "mutation exceeded its patch-operation cardinality"
    )


def test_edit_inside_top_level_but_outside_exact_focus_fails() -> None:
    child = copy.deepcopy(_PARENT)
    child["runtime"]["prefetch"] = 8

    _, outcome, _ = asyncio.run(
        _scenario(
            child,
            contract=MutationContract((_path("runtime", "threads"),)),
            source_attribution=(
                SourceAttribution("$.runtime.prefetch", "mutation"),
            ),
        )
    )

    assert outcome.candidate is not None
    assert not outcome.candidate.operator_compliant
    assert outcome.candidate.operator_failure == (
        "mutation changed a path outside its machine contract"
    )


@pytest.mark.parametrize("allow_abstention", [False, True])
def test_noop_requires_explicit_abstention(allow_abstention: bool) -> None:
    _, outcome, _ = asyncio.run(
        _scenario(
            copy.deepcopy(_PARENT),
            contract=MutationContract(
                (_path("runtime", "threads"),),
                allow_abstention=allow_abstention,
            ),
            source_attribution=(
                (SourceAttribution("$.runtime", "ancestor"),)
                if allow_abstention
                else ()
            ),
        )
    )

    assert outcome.candidate is not None
    assert outcome.candidate.operator_compliant is allow_abstention
    if allow_abstention:
        assert outcome.candidate.operator_failure is None
        assert outcome.candidate.evidence_compliant
    else:
        assert outcome.candidate.operator_failure == "mutation produced unchanged content"


def test_abstention_rejects_a_nonexistent_mutation_source_claim() -> None:
    _, outcome, _ = asyncio.run(
        _scenario(
            copy.deepcopy(_PARENT),
            contract=MutationContract(
                (_path("runtime", "threads"),),
                allow_abstention=True,
            ),
            source_attribution=(
                SourceAttribution("$.runtime.threads", "mutation"),
            ),
        )
    )

    assert outcome.candidate is not None
    assert outcome.candidate.operator_compliant
    assert not outcome.candidate.evidence_compliant
    assert outcome.candidate.evidence_failure == (
        "abstention claimed a nonexistent mutation path"
    )


def test_invalid_path_scope_and_other_operator_compositions_reject() -> None:
    with pytest.raises(ValueError, match="root cannot be an editable path"):
        MutationContract((JsonPath(),))
    with pytest.raises(ValueError, match="begin with an object key"):
        MutationContract((JsonPath((ArrayIndex(0),)),))

    async def plans():
        ids = DeterministicIdFactory("invalid_atomic_mutation")
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=_FixedGenerator(_PARENT, ()),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=1,
        )
        parent = await engine.register_seed(_PARENT, label="parent")
        contract = MutationContract((_path("compiler", "unroll"),))
        with pytest.raises(ValueError, match="escapes allowed_top_level"):
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (parent,),
                generation=1,
                label="bad_scope",
                allowed_top_level=("runtime",),
                mutation_contract=contract,
            )
        with pytest.raises(ValueError, match="only typed mutation"):
            InvocationPlan(
                OperatorKind.REPRODUCTION,
                (parent,),
                generation=1,
                label="bad_operator",
                mutation_contract=contract,
            )

    asyncio.run(plans())


def test_contractless_mutation_retains_legacy_top_level_semantics() -> None:
    child = copy.deepcopy(_PARENT)
    child["runtime"]["threads"] = 8
    child["runtime"]["prefetch"] = 8

    _, outcome, traces = asyncio.run(
        _scenario(
            child,
            contract=None,
            source_attribution=(
                SourceAttribution("$.runtime.threads", "mutation"),
                SourceAttribution("$.runtime.prefetch", "mutation"),
            ),
        )
    )

    assert outcome.candidate is not None
    assert outcome.candidate.operator_compliant
    assert "MACHINE MUTATION CONTRACT" not in outcome.prepared.prompt
    prepared = next(
        event for event in traces if event["event_type"] == "invocation_prepared"
    )
    assert prepared["mutation_contract"] is None

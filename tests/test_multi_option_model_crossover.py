from __future__ import annotations

import asyncio
import math
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    OperatorKind,
    ProposalAuthority,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchRecombiner,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    derive_exact_parent_crossover_contract,
    resolve_exact_parent_import_for_target,
)
from agent_evolve.policies.variation.typed_patch import ThreeWayRelationKind
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT,
    VariationGenerationResult,
)


_ANCESTOR = {"left_gene": 0, "right_gene": 0, "blend": 0, "guard": 9}
_LEFT = {"left_gene": 4, "right_gene": 0, "blend": 0, "guard": 9}
_RIGHT = {"left_gene": 0, "right_gene": 6, "blend": 0, "guard": 9}
_DETERMINISTIC_UNION = {
    "left_gene": 4,
    "right_gene": 6,
    "blend": 0,
    "guard": 9,
}
_MODEL_CROSSOVER = {
    "left_gene": 4,
    "right_gene": 6,
    "blend": 3,
    "guard": 9,
}


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    left_gene: int
    right_gene: int
    blend: int
    guard: int


class _Problem:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec("loss", "min"),)

    def __init__(self) -> None:
        self.physical_evaluations = 0

    @staticmethod
    def search_space_description() -> str:
        return (
            "Four integer coordinates. left_gene and right_gene are independent "
            "branch edits; blend is a legal synthesized crossover coordinate."
        )

    @staticmethod
    def validate(configuration: object) -> bool:
        _Candidate.model_validate(configuration, strict=True)
        return True

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        self.physical_evaluations += 1
        return {
            "loss": float(
                abs(int(configuration["left_gene"]) - 4)
                + abs(int(configuration["right_gene"]) - 6)
                + abs(int(configuration["blend"]) - 3)
            )
        }


_ROUNDTRIP_PARENT_FLOAT = -0.0045000000000000005
_DECIMAL_WITNESS_FLOAT = -0.0045
_HISTORICAL_CALL_000005_MATERIALIZATION_RECEIPT = (
    "15c23f63a4c90bef0398a2737cfb4f868015e659ef82405ccabe7a2564bfaed2"
)
# Faithful sanitized replay of DeepSeek call 000005: both parent shapes are
# 10/10/3, and the returned right-parent lower surface rounded all four rear
# coefficients by exactly one binary64 ULP.
_FLOAT_LEFT = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": [
        0.0,
        0.0015,
        -0.0015,
        -0.0015,
        0.0015,
        -0.0015,
        0.0015,
        0.0015,
        -0.0015,
        0.0,
    ],
    "lower_coefficients": [
        0.0,
        0.0015,
        0.0015,
        -0.0015,
        0.0015,
        -0.0015,
        -0.0015,
        -0.0015,
        -0.0015,
        0.0,
    ],
    "alpha_deg": [3.0, 2.5, 2.75],
}
_FLOAT_RIGHT = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": [
        0.0,
        0.0015,
        -0.0015,
        -0.0015,
        0.0015,
        _ROUNDTRIP_PARENT_FLOAT,
        -0.0015,
        -0.0015,
        _ROUNDTRIP_PARENT_FLOAT,
        0.0,
    ],
    "lower_coefficients": [
        0.0,
        0.0015,
        0.0015,
        -0.0015,
        0.0015,
        _ROUNDTRIP_PARENT_FLOAT,
        _ROUNDTRIP_PARENT_FLOAT,
        _ROUNDTRIP_PARENT_FLOAT,
        _ROUNDTRIP_PARENT_FLOAT,
        0.0,
    ],
    "alpha_deg": [2.75, 2.75, 2.25],
}
_FLOAT_WITNESS = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": list(_FLOAT_LEFT["upper_coefficients"]),
    "lower_coefficients": [
        0.0,
        0.0015,
        0.0015,
        -0.0015,
        0.0015,
        _DECIMAL_WITNESS_FLOAT,
        _DECIMAL_WITNESS_FLOAT,
        _DECIMAL_WITNESS_FLOAT,
        _DECIMAL_WITNESS_FLOAT,
        0.0,
    ],
    "alpha_deg": list(_FLOAT_LEFT["alpha_deg"]),
}
_AUTHENTIC_SEED_000002 = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": list(_FLOAT_LEFT["upper_coefficients"]),
    "lower_coefficients": list(_FLOAT_LEFT["lower_coefficients"]),
    "alpha_deg": list(_FLOAT_RIGHT["alpha_deg"]),
}
_AUTHENTIC_ADAPTIVE_UNION = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": list(_FLOAT_RIGHT["upper_coefficients"]),
    "lower_coefficients": list(_FLOAT_RIGHT["lower_coefficients"]),
    "alpha_deg": list(_FLOAT_LEFT["alpha_deg"]),
}
_FLOAT_ATTRIBUTION = (
    SourceAttribution("$.alpha_deg", "left"),
    SourceAttribution("$.lower_coefficients", "right"),
    SourceAttribution("$.upper_coefficients", "left"),
)


class _FloatVectorCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    representation_id: str
    upper_coefficients: list[float]
    lower_coefficients: list[float]
    alpha_deg: list[float]


class _FloatVectorProblem:
    candidate_model = _FloatVectorCandidate
    objectives = (ObjectiveSpec("loss", "min"),)

    def __init__(self) -> None:
        self.physical_evaluations = 0

    @staticmethod
    def search_space_description() -> str:
        return "Generic finite binary64 vector co-optimization."

    @staticmethod
    def validate(configuration: object) -> bool:
        _FloatVectorCandidate.model_validate(configuration, strict=True)
        return True

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        self.physical_evaluations += 1
        return {"loss": 0.0}


def test_authentic_airfoil_parent_pair_resolves_seed_and_union_known_children() -> None:
    """Regression from the completed DeepSeek v10 parent pair.

    The exact action space contains both the scheduled disjoint union and the
    historical seed_000002 as complementary proper subsets.  A novelty gate
    must therefore derive and exclude both, not just the corresponding union.
    """

    base = freeze_json(_FLOAT_LEFT)
    donor = freeze_json(_FLOAT_RIGHT)
    seed = freeze_json(_AUTHENTIC_SEED_000002)
    union = freeze_json(_AUTHENTIC_ADAPTIVE_UNION)
    assert all(type(value) is FrozenJsonObject for value in (base, donor, seed, union))
    assert type(base) is FrozenJsonObject
    assert type(donor) is FrozenJsonObject
    assert type(seed) is FrozenJsonObject
    assert type(union) is FrozenJsonObject
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)

    seed_imports = resolve_exact_parent_import_for_target(
        base=base,
        donor=donor,
        contract=contract,
        target=seed,
    )
    union_imports = resolve_exact_parent_import_for_target(
        base=base,
        donor=donor,
        contract=contract,
        target=union,
    )

    assert len(contract.loci) == 11
    assert seed_imports == ("locus_0001", "locus_0002", "locus_0003")
    assert union_imports == tuple(f"locus_{ordinal:04d}" for ordinal in range(4, 12))
    assert set(seed_imports).isdisjoint(union_imports)
    assert set(seed_imports) | set(union_imports) == {
        locus.locus_id for locus in contract.loci
    }


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/scripted",
        resolved_model="offline/scripted",
        resolved_provider="provider-free-test",
        provider_response_id="scripted-crossover-response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class _ScriptedModelGenerator:
    """Return one full configuration through the real model-authority boundary."""

    def __init__(
        self,
        configuration: dict[str, object],
        *,
        intended_changes: tuple[str, ...] | None = None,
        source_attribution: tuple[SourceAttribution, ...] | None = None,
    ) -> None:
        self.configuration = dict(configuration)
        self.intended_changes = (
            ("$.left_gene", "$.right_gene", "$.blend")
            if intended_changes is None
            else intended_changes
        )
        self.source_attribution = (
            (
                SourceAttribution("$.left_gene", "left"),
                SourceAttribution("$.right_gene", "right"),
                SourceAttribution("$.blend", "synthesized"),
            )
            if source_attribution is None
            else source_attribution
        )
        self.proposal_requests: list[object] = []
        self.reflection_requests: list[object] = []

    async def propose(self, request):
        self.proposal_requests.append(request)
        assert request.operation == OperatorKind.TWO_PARENT_CROSSOVER.value
        assert "OPERATOR: two_parent_crossover" in request.prompt
        assert "system independently verifies at least one exact contribution" in (
            request.prompt
        )
        assert "CANDIDATE COMPONENT PATH CONTRACT" in request.prompt
        assert "'$' denotes the returned configuration value itself" in request.prompt
        assert '"$.configuration.gene"' in request.prompt
        assert "the PARENTS rows are evidence envelopes, never path roots" in (
            request.prompt
        )
        assert TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT in request.prompt
        assert "differs from the other parent at the same path" in request.prompt
        assert "executable inheritance instruction" in request.prompt
        assert "copies the exact immutable parent subtree" in request.prompt
        assert "one binary64 ULP" in request.prompt
        assert "Synthesized values remain model-authored" in request.prompt
        assert "source token from left, right, or synthesized" in request.prompt
        assert (
            "source token from ancestor, left, right, synthesized, or mutation"
            not in request.prompt
        )
        assert "smallest retained containing object or array" in request.prompt
        assert (
            "omit the path from both source_attribution and intended_changes"
            in request.prompt
        )
        assert "at least one discriminating right contribution" in request.prompt
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=dict(self.configuration),
                design_rationale=(
                    "Inherit the exact independent genes and synthesize a "
                    "non-union blend value."
                ),
                intended_changes=self.intended_changes,
                source_attribution=self.source_attribution,
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        self.reflection_requests.append(request)
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def test_model_authored_crossover_differs_from_union_and_reproduction_hits_cache() -> (
    None
):
    async def scenario():
        ids = DeterministicIdFactory("multi_option_model_crossover")
        problem = _Problem()
        generator = _ScriptedModelGenerator(_MODEL_CROSSOVER)
        traces: list[dict[str, object]] = []
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=17,
            trace_sink=traces.append,
        )

        ancestor = await engine.register_seed(_ANCESTOR, label="ancestor")
        left = await engine.register_seed(_LEFT, label="left_branch")
        right = await engine.register_seed(_RIGHT, label="right_branch")

        union_materialization = DisjointPatchRecombiner().materialize(
            ancestor=ancestor.configuration,
            ancestor_candidate_id=ancestor.candidate_id,
            left=left.configuration,
            left_candidate_id=left.candidate_id,
            right=right.configuration,
            right_candidate_id=right.candidate_id,
            target_candidate_id=ids.new_candidate_id(),
        )
        union_plan = InvocationPlan(
            operator_kind=OperatorKind.THREE_WAY_RECOMBINATION,
            parents=(left, right),
            common_ancestor=ancestor,
            generation=3,
            label="deterministic_union_control",
            phase="g3_recombination_control",
        )
        (union_outcome,) = await engine.run_materialized_invocations(
            (
                materialized_disjoint_invocation(
                    plan=union_plan,
                    materialization=union_materialization,
                ),
            )
        )

        model_plan = InvocationPlan(
            operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
            parents=(left, right),
            generation=3,
            label="model_authored_crossover",
            phase="g3_model_crossover",
        )
        (model_outcome,) = await engine.run_invocations((model_plan,))
        before_reproduction = await engine.evaluation_cache_snapshot()

        assert model_outcome.candidate is not None
        reproduction_plan = InvocationPlan(
            operator_kind=OperatorKind.REPRODUCTION,
            parents=(model_outcome.candidate,),
            generation=3,
            label="model_crossover_reproduction",
            phase="g3_reproduction_control",
        )
        (reproduction_outcome,) = await engine.run_invocations((reproduction_plan,))
        after_reproduction = await engine.evaluation_cache_snapshot()
        return (
            problem,
            generator,
            traces,
            ancestor,
            left,
            right,
            union_materialization,
            union_plan,
            union_outcome,
            model_plan,
            model_outcome,
            before_reproduction,
            reproduction_plan,
            reproduction_outcome,
            after_reproduction,
        )

    (
        problem,
        generator,
        traces,
        ancestor,
        left,
        right,
        union_materialization,
        union_plan,
        union_outcome,
        model_plan,
        model_outcome,
        before_reproduction,
        reproduction_plan,
        reproduction_outcome,
        after_reproduction,
    ) = asyncio.run(scenario())

    # The comparator is a receipt-bound union of exactly two disjoint
    # ancestor-relative effects; it never crosses the model boundary.
    assert (
        len(union_materialization.classification.of_kind(ThreeWayRelationKind.DISJOINT))
        == 2
    )
    assert all(
        not union_materialization.classification.of_kind(kind)
        for kind in (
            ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT,
            ThreeWayRelationKind.CONFLICT,
            ThreeWayRelationKind.INVALIDATED,
        )
    )
    assert union_plan.parents == (left, right)
    assert union_plan.common_ancestor == ancestor
    assert union_outcome.prepared.proposal_authority is ProposalAuthority.ENGINE
    assert union_outcome.prepared.call_id is None
    assert union_outcome.candidate is not None
    assert union_outcome.candidate.configuration_dict == _DETERMINISTIC_UNION
    assert union_outcome.candidate.operator_compliant
    assert union_outcome.candidate.preservation_verified is True

    # The model plan has ordered two-parent lineage and deliberately has no
    # common ancestor field: TWO_PARENT_CROSSOVER is full-configuration model
    # authority, whose inherited effects are independently re-derived.
    assert model_plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER
    assert model_plan.parents == (left, right)
    assert model_plan.common_ancestor is None
    assert model_plan.generation == 3
    assert model_plan.phase == "g3_model_crossover"
    assert model_outcome.prepared.proposal_authority is ProposalAuthority.MODEL
    assert model_outcome.prepared.call_id is not None
    assert model_outcome.candidate is not None
    assert model_outcome.candidate.configuration_dict == _MODEL_CROSSOVER
    assert model_outcome.candidate.configuration_dict != _DETERMINISTIC_UNION
    assert (
        model_outcome.candidate.occurrence.configuration_hash
        != union_outcome.candidate.occurrence.configuration_hash
    )
    assert model_outcome.candidate.parent_ids == (
        left.candidate_id,
        right.candidate_id,
    )
    assert model_outcome.candidate.operator_compliant
    assert model_outcome.candidate.operator_failure is None
    assert model_outcome.candidate.evidence_compliant
    assert model_outcome.candidate.source_attribution == (
        SourceAttribution("$.left_gene", "left"),
        SourceAttribution("$.right_gene", "right"),
        SourceAttribution("$.blend", "synthesized"),
    )
    assert model_outcome.candidate.objective_map == {"loss": 0.0}
    assert union_outcome.candidate.objective_map == {"loss": 3.0}

    # Reproduction is a new causal occurrence of the exact model phenotype. It
    # consumes neither a model call nor another physical evaluation.
    assert reproduction_plan.operator_kind is OperatorKind.REPRODUCTION
    assert reproduction_plan.parents == (model_outcome.candidate,)
    assert reproduction_outcome.prepared.proposal_authority is (
        ProposalAuthority.REPRODUCTION
    )
    assert reproduction_outcome.prepared.call_id is None
    assert reproduction_outcome.candidate is not None
    assert reproduction_outcome.candidate.configuration_dict == _MODEL_CROSSOVER
    assert (
        reproduction_outcome.candidate.occurrence.configuration_hash
        == model_outcome.candidate.occurrence.configuration_hash
    )
    assert (
        reproduction_outcome.candidate.candidate_id
        != model_outcome.candidate.candidate_id
    )
    assert reproduction_outcome.candidate.call_telemetry is None
    assert before_reproduction == {
        "capacity": None,
        "cached_entries": 5,
        "in_flight": 0,
        "hits": 0,
        "misses": 5,
        "coalesced": 0,
        "evictions": 0,
    }
    assert after_reproduction == {
        **before_reproduction,
        "hits": 1,
    }
    assert problem.physical_evaluations == 5
    assert len(generator.proposal_requests) == 1
    assert generator.reflection_requests == []

    cache_events = [
        event for event in traces if event["event_type"] == "evaluation_cache_event"
    ]
    assert [event["cache_event_type"] for event in cache_events] == [
        "miss",
        "miss",
        "miss",
        "miss",
        "miss",
        "hit",
    ]
    assert cache_events[-1]["configuration_hash"] == (
        model_outcome.candidate.occurrence.configuration_hash
    )
    evaluated_by_label = {
        event["label"]: event
        for event in traces
        if event["event_type"] == "candidate_evaluated"
    }
    assert (
        evaluated_by_label["deterministic_union_control"][
            "source_attribution_provenance"
        ]
        == "engine_materialized"
    )
    assert (
        evaluated_by_label["model_authored_crossover"]["source_attribution_provenance"]
        == "engine_materialized_from_model_inheritance_plan"
    )
    assert (
        evaluated_by_label["model_crossover_reproduction"][
            "source_attribution_provenance"
        ]
        == "framework_generated"
    )
    assert sum(event["event_type"] == "llm_call_completed" for event in traces) == 1


def test_two_parent_crossover_rejects_a_one_parent_copy() -> None:
    async def scenario():
        ids = DeterministicIdFactory("strict_two_parent_crossover")
        generator = _ScriptedModelGenerator(_LEFT)
        problem = _Problem()
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=19,
        )
        await engine.register_seed(_ANCESTOR, label="ancestor")
        left = await engine.register_seed(_LEFT, label="left_branch")
        right = await engine.register_seed(_RIGHT, label="right_branch")
        plan = InvocationPlan(
            operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
            parents=(left, right),
            generation=3,
            label="one_parent_copy_control",
            phase="g3_model_crossover",
        )
        physical_evaluations_before = problem.physical_evaluations
        (outcome,) = await engine.run_invocations((plan,))
        return (
            problem,
            generator,
            physical_evaluations_before,
            outcome,
        )

    problem, generator, physical_evaluations_before, outcome = asyncio.run(scenario())
    assert len(generator.proposal_requests) == 1
    assert outcome.candidate is None
    assert outcome.failure_stage == "candidate"
    assert outcome.call_failure_type == "ValueError"
    assert problem.physical_evaluations == physical_evaluations_before
    assert outcome.reward == -1.0


def test_two_parent_crossover_with_verified_paths_evaluates_exactly_once() -> None:
    async def scenario():
        ids = DeterministicIdFactory("verified_crossover_paths")
        problem = _Problem()
        generator = _ScriptedModelGenerator(_MODEL_CROSSOVER)
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=29,
        )
        left = await engine.register_seed(_LEFT, label="left_branch")
        right = await engine.register_seed(_RIGHT, label="right_branch")
        physical_evaluations_before = problem.physical_evaluations
        plan = InvocationPlan(
            operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
            parents=(left, right),
            generation=3,
            label="verified_path_crossover",
            phase="g3_model_crossover",
        )
        (outcome,) = await engine.run_invocations((plan,))
        return problem, physical_evaluations_before, outcome

    problem, physical_evaluations_before, outcome = asyncio.run(scenario())

    assert problem.physical_evaluations == physical_evaluations_before + 1
    assert outcome.failure_stage is None
    assert outcome.call_failure_type is None
    assert outcome.candidate is not None
    assert outcome.candidate.configuration_dict == _MODEL_CROSSOVER
    assert outcome.candidate.operator_compliant
    assert outcome.candidate.evidence_compliant


def test_shared_field_source_claim_rejects_before_evaluation() -> None:
    attribution_with_unsupported_shared_field = (
        SourceAttribution("$.left_gene", "left"),
        SourceAttribution("$.right_gene", "right"),
        SourceAttribution("$.guard", "left"),
        SourceAttribution("$.blend", "synthesized"),
    )

    async def scenario():
        ids = DeterministicIdFactory("shared_field_crossover_claim")
        problem = _Problem()
        generator = _ScriptedModelGenerator(
            _MODEL_CROSSOVER,
            source_attribution=attribution_with_unsupported_shared_field,
        )
        traces: list[dict[str, object]] = []
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=31,
            trace_sink=traces.append,
        )
        left = await engine.register_seed(_LEFT, label="left_branch")
        right = await engine.register_seed(_RIGHT, label="right_branch")
        physical_evaluations_before = problem.physical_evaluations
        plan = InvocationPlan(
            operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
            parents=(left, right),
            generation=3,
            label="shared_field_claim_crossover",
            phase="g3_model_crossover",
        )
        (outcome,) = await engine.run_invocations((plan,))
        return problem, physical_evaluations_before, traces, outcome

    problem, physical_evaluations_before, traces, outcome = asyncio.run(scenario())

    assert problem.physical_evaluations == physical_evaluations_before
    assert outcome.candidate is None
    assert outcome.failure_stage == "candidate"
    assert outcome.call_failure_type == "ValueError"
    assert not any(
        event["event_type"] == "candidate_evaluated"
        and event.get("label") == "shared_field_claim_crossover"
        for event in traces
    )
    assert any(event["event_type"] == "candidate_boundary_failed" for event in traces)


def test_wrapper_prefixed_crossover_paths_are_not_normalized_and_fail_evidence_audit() -> (
    None
):
    malformed_changes = (
        "$.configuration.left_gene",
        "$.configuration.right_gene",
        "$.configuration.blend",
    )
    malformed_attribution = (
        SourceAttribution("$.configuration.left_gene", "left"),
        SourceAttribution("$.configuration.right_gene", "right"),
        SourceAttribution("$.configuration.blend", "synthesized"),
    )

    async def scenario():
        ids = DeterministicIdFactory("wrapper_prefixed_crossover_paths")
        problem = _Problem()
        generator = _ScriptedModelGenerator(
            _MODEL_CROSSOVER,
            intended_changes=malformed_changes,
            source_attribution=malformed_attribution,
        )
        traces: list[dict[str, object]] = []
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=23,
            trace_sink=traces.append,
        )
        left = await engine.register_seed(_LEFT, label="left_branch")
        right = await engine.register_seed(_RIGHT, label="right_branch")
        plan = InvocationPlan(
            operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
            parents=(left, right),
            generation=3,
            label="malformed_path_crossover",
            phase="g3_model_crossover",
        )
        physical_evaluations_before = problem.physical_evaluations
        (outcome,) = await engine.run_invocations((plan,))
        return problem, physical_evaluations_before, traces, outcome

    problem, physical_evaluations_before, traces, outcome = asyncio.run(scenario())

    assert problem.physical_evaluations == physical_evaluations_before
    assert outcome.candidate is None
    assert outcome.failure_stage == "candidate"
    assert outcome.call_failure_type == "ValueError"
    # The boundary must never guess that the model intended candidate-root paths.
    assert not any(
        event["event_type"] == "candidate_evaluated"
        and event.get("label") == "malformed_path_crossover"
        for event in traces
    )
    boundary_failure = next(
        event for event in traces if event["event_type"] == "candidate_boundary_failed"
    )
    assert boundary_failure["failure_type"] == "ValueError"


def test_crossover_materializes_exact_parent_float_from_one_ulp_decimal_witness() -> (
    None
):
    """Regression for call 000005's decimal/arithmetic binary64 spelling drift."""

    async def scenario():
        ids = DeterministicIdFactory("crossover_one_ulp_witness")
        problem = _FloatVectorProblem()
        generator = _ScriptedModelGenerator(
            _FLOAT_WITNESS,
            intended_changes=(
                "$.alpha_deg",
                "$.lower_coefficients",
                "$.upper_coefficients",
            ),
            source_attribution=_FLOAT_ATTRIBUTION,
        )
        traces: list[dict[str, object]] = []
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=37,
            trace_sink=traces.append,
        )
        left = await engine.register_seed(_FLOAT_LEFT, label="left")
        right = await engine.register_seed(_FLOAT_RIGHT, label="right")
        before = problem.physical_evaluations
        (outcome,) = await engine.run_invocations(
            (
                InvocationPlan(
                    operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
                    parents=(left, right),
                    generation=3,
                    label="one_ulp_crossover",
                    phase="generic_crossover_regression",
                ),
            )
        )
        return problem, before, traces, outcome

    problem, before, traces, outcome = asyncio.run(scenario())

    assert outcome.failure_stage is None
    assert outcome.candidate is not None
    assert outcome.candidate.operator_compliant
    assert outcome.candidate.evidence_compliant
    assert problem.physical_evaluations == before + 1
    child = outcome.candidate.configuration_dict
    assert len(child["upper_coefficients"]) == 10
    assert len(child["lower_coefficients"]) == 10
    assert len(child["alpha_deg"]) == 3
    for index in (5, 6, 7, 8):
        assert child["lower_coefficients"][index].hex() == (
            _ROUNDTRIP_PARENT_FLOAT.hex()
        )
        assert child["lower_coefficients"][index].hex() != (
            _DECIMAL_WITNESS_FLOAT.hex()
        )

    evaluated = next(
        event
        for event in traces
        if event["event_type"] == "candidate_evaluated"
        and event["label"] == "one_ulp_crossover"
    )
    assert evaluated["source_attribution_provenance"] == (
        "engine_materialized_from_model_inheritance_plan"
    )
    assert evaluated["crossover_draft_configuration_hash"] == typed_json_sha256(
        freeze_json(_FLOAT_WITNESS)
    )
    assert evaluated["crossover_materialized_configuration_hash"] == (
        outcome.candidate.occurrence.configuration_hash
    )
    assert evaluated["crossover_adjusted_float_leaf_count"] == 4
    assert evaluated["crossover_materialization_receipt_sha256"] == (
        _HISTORICAL_CALL_000005_MATERIALIZATION_RECEIPT
    )
    receipt = evaluated["crossover_materialization"]
    lower = next(
        item
        for item in receipt["inherited_paths"]
        if item["path"] == "$.lower_coefficients"
    )
    assert lower == {
        "path": "$.lower_coefficients",
        "source": "right",
        "witness_value_sha256": typed_json_sha256(
            freeze_json(_FLOAT_WITNESS["lower_coefficients"])
        ),
        "parent_value_sha256": typed_json_sha256(
            freeze_json(_FLOAT_RIGHT["lower_coefficients"])
        ),
        "witness_exact": False,
        "adjusted_float_leaf_count": 4,
        "max_float_ulp_distance": 1,
    }


def test_crossover_rejects_two_ulp_inherited_witness_before_evaluation() -> None:
    source = _ROUNDTRIP_PARENT_FLOAT
    one_step_toward_zero = math.nextafter(source, math.inf)
    two_steps_toward_zero = math.nextafter(one_step_toward_zero, math.inf)
    assert one_step_toward_zero == _DECIMAL_WITNESS_FLOAT
    adversarial = {
        **_FLOAT_WITNESS,
        "lower_coefficients": [
            *_FLOAT_WITNESS["lower_coefficients"][:5],
            two_steps_toward_zero,
            *_FLOAT_WITNESS["lower_coefficients"][6:],
        ],
    }

    async def scenario():
        ids = DeterministicIdFactory("crossover_two_ulp_witness")
        problem = _FloatVectorProblem()
        generator = _ScriptedModelGenerator(
            adversarial,
            intended_changes=(
                "$.alpha_deg",
                "$.lower_coefficients",
                "$.upper_coefficients",
            ),
            source_attribution=_FLOAT_ATTRIBUTION,
        )
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=41,
        )
        left = await engine.register_seed(_FLOAT_LEFT, label="left")
        right = await engine.register_seed(_FLOAT_RIGHT, label="right")
        before = problem.physical_evaluations
        (outcome,) = await engine.run_invocations(
            (
                InvocationPlan(
                    operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
                    parents=(left, right),
                    generation=3,
                    label="two_ulp_crossover",
                    phase="generic_crossover_regression",
                ),
            )
        )
        return problem, before, outcome

    problem, before, outcome = asyncio.run(scenario())
    assert outcome.candidate is None
    assert outcome.failure_stage == "candidate"
    assert outcome.call_failure_type == "ValueError"
    assert problem.physical_evaluations == before

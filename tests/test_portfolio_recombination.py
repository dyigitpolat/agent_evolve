from __future__ import annotations

import asyncio
import hashlib
from itertools import combinations
import threading
import time
from dataclasses import replace
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import AgenticEvolutionEngine
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluationPayload,
    EvaluatorIdentity,
)
from agent_evolve.application.contextual_delayed_credit import (
    observe_contextual_post_recombination_credit,
)
from agent_evolve.application.contextual_search_controller import (
    ContextualSearchObservation,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.outcome_relation import objective_pareto_outcome_binding
from agent_evolve.application.portfolio_evolution import (
    PortfolioEvolution,
    PortfolioVariationWaveRequest,
)
from agent_evolve.application.portfolio_recombination import (
    ArchiveAwareDisjointPairSelectionDecision,
    ArchiveAwareDisjointParentPairPolicy,
    FrozenArchiveBranchUtility,
    FrozenArchiveSourcePairUtility,
    FrozenArchiveSourceUtilityContext,
    FrozenArchiveSourceUtilityReceipt,
    ObservedSourceBranch,
    PortfolioRecombination,
    PortfolioRecombinationNoPairReason,
    PortfolioRecombinationSourceExclusionReason,
    PortfolioRecombinationWaveRequest,
    PortfolioRecombinationWaveResult,
    bind_portfolio_recombination_source_utilities,
    frozen_archive_source_utility_context,
    portfolio_recombination_observed_sources,
)
from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.patch import JsonPath, ObjectKey
from agent_evolve.domain.outcome import FailureCategory, FailureCode, FailureRecord
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.task_keyed_palette import PathFamilyExposure
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    resolve_ranked_portfolio_decision,
)


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


class _Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    a: int
    b: int
    c: int
    d: int


class _ConcurrentProblem:
    candidate_model = _Configuration
    objectives = (ObjectiveSpec("loss", "min"),)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.evaluations = 0
        self.active = 0
        self.max_active = 0

    @staticmethod
    def search_space_description() -> str:
        return "Four integer coordinates for generic portfolio recombination."

    @staticmethod
    def validate(configuration: object) -> bool:
        _Configuration.model_validate(configuration, strict=True)
        return True

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        parsed = _Configuration.model_validate(configuration, strict=True)
        with self._lock:
            self.evaluations += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.035)
        with self._lock:
            self.active -= 1
        return {"loss": float(parsed.a + parsed.b + parsed.c + parsed.d)}

    def reset(self) -> None:
        with self._lock:
            self.evaluations = 0
            self.active = 0
            self.max_active = 0


class _CandidateInfeasibilityEvaluator:
    evaluator_identity = EvaluatorIdentity(
        evaluator_id="recombination_candidate_infeasibility_test",
        evaluator_version=1,
        evaluator_context_sha256=hashlib.sha256(
            b"recombination-candidate-infeasibility-test-v1"
        ).hexdigest(),
    )

    def __init__(self, infeasible: set[tuple[int, int, int, int]]) -> None:
        self.infeasible = frozenset(infeasible)
        self.evaluations: list[tuple[int, int, int, int]] = []

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        parsed = _Configuration.model_validate(configuration, strict=True)
        coordinates = (parsed.a, parsed.b, parsed.c, parsed.d)
        self.evaluations.append(coordinates)
        if coordinates in self.infeasible:
            return DetailedEvaluationPayload(
                failure=FailureRecord(
                    category=FailureCategory.CANDIDATE,
                    code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
                    message="sealed source violates the fixture constraint",
                ),
                objectives=(),
                violations=(),
                checks=(),
                receipt=None,
                evaluator=self.evaluator_identity,
            )
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(("loss", float(sum(coordinates))),),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


class _NoGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, request):
        del request
        self.calls += 1
        raise AssertionError("portfolio materializations must be provider-free")

    async def reflect(self, request):
        del request
        self.calls += 1
        raise AssertionError("the recombination stage does not reflect")


class _Selector:
    def __init__(self) -> None:
        self.calls = 0

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        self.calls += 1
        prediction = (MetricEffectPrediction("loss", MetricEffectDirection.DECREASE),)
        decision = resolve_ranked_portfolio_decision(
            request,
            tuple(
                PortfolioMemberDraft(
                    option_id=option.option_id,
                    supporting_card_keys=(request.cards[0].card_key,),
                    effect_predictions=prediction,
                    design_rationale=f"Select sealed option {option.option_id}.",
                )
                for option in request.finite_variation_contract.options[
                    : request.portfolio_size
                ]
            ),
            policy_id="fake_portfolio_selector",
            policy_version=1,
            policy_definition_sha256="e" * 64,
        )
        return PortfolioSelectionResult(
            decision=decision,
            telemetry=AgenticCallTelemetry(
                requested_model="fake/portfolio",
                resolved_model="fake/portfolio-v1",
                resolved_provider="provider-free",
                provider_response_id="fake-response",
                finish_reason="stop",
                input_tokens=20,
                output_tokens=8,
                reasoning_tokens=4,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0"),
                latency_ns=1_000,
            ),
        )


def _contract(
    parent: FrozenJsonObject,
    *,
    disjoint: bool,
) -> FiniteVariationContract:
    parent_sha256 = typed_json_sha256(parent)
    if disjoint:
        definitions = tuple(
            (
                f"{name}.set",
                f"family_{name}",
                {key: (index + 1 if key == name else 0) for key in "abcd"},
            )
            for index, name in enumerate("abcd")
        )
    else:
        definitions = tuple(
            (
                f"a.set_{index}",
                f"family_{name}",
                {"a": index, "b": 0, "c": 0, "d": 0},
            )
            for index, name in enumerate("abc", start=1)
        )
    return FiniteVariationContract(
        catalog_id="portfolio_recombination_test",
        catalog_version=1,
        catalog_definition_sha256="d" * 64,
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen(configuration),
                family=family,
                description=f"Apply {option_id}.",
            )
            for option_id, family, configuration in definitions
        ),
    )


async def _source_wave(
    namespace: str,
    *,
    disjoint: bool = True,
    detailed_evaluator: _CandidateInfeasibilityEvaluator | None = None,
):
    ids = DeterministicIdFactory(namespace)
    problem = _ConcurrentProblem()
    generator = _NoGenerator()
    memory = InsightMemoryBank(id_factory=ids)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=9,
        evaluator_concurrency=4,
        detailed_evaluator=detailed_evaluator,
        outcome_relation_binding=(
            None
            if detailed_evaluator is None
            else objective_pareto_outcome_binding(problem.objectives)
        ),
    )
    ancestor = await engine.register_seed(
        {"a": 0, "b": 0, "c": 0, "d": 0},
        label="ancestor",
    )
    entry, added = memory.add(
        InsightDraft(
            claim="Use one exact finite edit.",
            trigger="A ranked portfolio is requested.",
            mechanism="Sealed local changes expose safe recombination structure.",
            affected_paths=("$.a",),
            evidence_summary="Provider-free fixture evidence.",
            confidence=0.5,
        )
    )
    assert added
    card = PortfolioCard(
        card_key="card.seed",
        reference=entry.reference,
        content_sha256="1" * 64,
        evidence_sha256="2" * 64,
        prompt_payload=_frozen({"claim": "one exact finite edit"}),
    )
    contract = _contract(ancestor.configuration, disjoint=disjoint)
    selection_request = PortfolioSelectionRequest(
        call_id=ids.new_llm_call_id(),
        operation="select_portfolio",
        instruction="Rank the exact finite options.",
        context=_frozen({"fixture": namespace}),
        finite_variation_contract=contract,
        cards=(card,),
        portfolio_size=len(contract.options),
        required_metric_ids=("loss",),
        min_distinct_families=len(contract.options),
    )
    source_request = PortfolioVariationWaveRequest(
        selection_request=selection_request,
        parent=ancestor,
        generation=1,
        label_prefix="source_portfolio",
    )
    selector = _Selector()
    source_result = await PortfolioEvolution(
        engine=engine,
        selector=selector,
        ids=ids,
    ).run(source_request)
    problem.reset()
    return (
        ids,
        problem,
        generator,
        engine,
        selector,
        ancestor,
        source_request,
        source_result,
    )


def _exposures() -> tuple[PathFamilyExposure, ...]:
    return tuple(
        PathFamilyExposure(
            path=JsonPath((ObjectKey(name),)),
            family=f"family_{name}",
            count=index,
        )
        for index, name in enumerate("abcd")
    )


def _archive_snapshot(*, generation: int = 1) -> ArchiveUtilitySnapshot:
    return ArchiveUtilitySnapshot(
        utility_id="fixture_exact_joint_archive_utility",
        utility_version=1,
        definition_sha256="7" * 64,
        generation=generation,
        benchmark_sha256="8" * 64,
        archive_sha256="9" * 64,
        snapshot_receipt=_frozen(
            {
                "indicator": "opaque fixture utility",
                "cutoff": "before source generation",
            }
        ),
    )


def test_enumerates_all_pairs_and_evaluates_two_concurrent_unions() -> None:
    async def scenario():
        values = await _source_wave("portfolio_recombination_success")
        ids, problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="portfolio_union",
                path_family_exposures=_exposures(),
            )
        )
        return problem, generator, selector, ancestor, source, result

    problem, generator, selector, ancestor, source, result = asyncio.run(scenario())
    receipt = result.receipt
    assert selector.calls == 1
    assert generator.calls == 0
    assert len(receipt.branches) == 4
    assert tuple(value.rank for value in receipt.branches) == (1, 2, 3, 4)
    assert tuple(value.family for value in receipt.branches) == tuple(
        f"family_{name}" for name in "abcd"
    )
    assert tuple(value.path_family_exposure for value in receipt.branches) == (
        0,
        1,
        2,
        3,
    )
    assert len(receipt.pair_attempts) == 6
    assert all(value.replay_safe for value in receipt.pair_attempts)
    assert len(receipt.pair_decision.eligible_rows) == 6
    assert tuple(value.selection_role for value in receipt.members) == (
        "exploit",
        "coverage",
    )
    assert problem.evaluations == 2
    assert problem.max_active == 2
    assert len(result.outcomes) == len(result.candidates) == 2
    assert all(candidate.generation == 2 for candidate in result.candidates)
    assert all(
        candidate.common_ancestor_id == ancestor.candidate_id
        for candidate in result.candidates
    )
    source_ids = {candidate.candidate_id for candidate in source.candidates}
    assert all(
        set(candidate.parent_ids).issubset(source_ids)
        for candidate in result.candidates
    )
    assert receipt.to_record()["pair_universe_size"] == 6


def test_campaign_child_envelope_can_select_only_the_exploit_union() -> None:
    async def scenario():
        values = await _source_wave("portfolio_recombination_single_child")
        ids, problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(
            engine=engine,
            ids=ids,
            selection_limit=1,
        ).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="single_portfolio_union",
                path_family_exposures=_exposures(),
            ),
        )
        return problem, generator, selector, result

    problem, generator, selector, result = asyncio.run(scenario())
    assert tuple(value.selection_role for value in result.receipt.members) == (
        "exploit",
    )
    assert result.receipt.selection_limit == 1
    assert result.receipt.to_record()["selection_limit"] == 1
    assert problem.evaluations == 1
    assert len(result.outcomes) == len(result.candidates) == 1
    assert generator.calls == 0
    assert selector.calls == 1


def test_post_recombination_credit_separates_survival_and_descendant_yield() -> None:
    async def scenario():
        values = await _source_wave("portfolio_recombination_contextual_credit")
        ids, _problem, _generator, engine, _selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(
            engine=engine,
            ids=ids,
            selection_limit=1,
        ).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="contextual_portfolio_union",
                path_family_exposures=_exposures(),
            ),
        )
        return source_wave, source, result

    source_wave, source, result = asyncio.run(scenario())
    campaign_scope_sha256 = hashlib.sha256(
        b"portfolio-recombination-contextual-campaign"
    ).hexdigest()
    observations = tuple(
        sorted(
            (
                ContextualSearchObservation(
                    campaign_scope_sha256=campaign_scope_sha256,
                    wave_index=1,
                    source_id="model",
                    operator_id="atomic",
                    option_identity_sha256=(
                        member.materialization.option_identity_sha256
                    ),
                    parent_context_sha256=(
                        source_wave.selection_request.context_sha256
                    ),
                    feasible=True,
                    positive_marginal_utility=False,
                    normalized_marginal_utility=0.0,
                    marginal_utility_share=0.0,
                    candidate_id=member.materialization.candidate_id,
                )
                for member in source.receipt.members
            ),
            key=lambda value: value.observation_sha256,
        )
    )
    selected = set(result.receipt.members[0].pair_ids)
    useful_child = result.receipt.members[0].target_candidate_id
    surviving_source = min(selected)
    front = tuple(sorted((surviving_source, useful_child)))
    batch = observe_contextual_post_recombination_credit(
        campaign_scope_sha256=campaign_scope_sha256,
        source_wave_index=1,
        observations=observations,
        results=(result,),
        post_stage_front_candidate_ids=front,
    )

    observation_by_hash = {value.observation_sha256: value for value in observations}
    credit_by_candidate = {
        observation_by_hash[value.source_observation_sha256].candidate_id: value
        for value in batch.credits
    }
    assert batch.selected_source_candidate_ids == tuple(sorted(selected))
    assert batch.stage_surviving_source_candidate_ids == (surviving_source,)
    assert batch.useful_descendant_candidate_ids == (useful_child,)
    assert len(batch.credits) == len(observations)
    for candidate_id, credit in credit_by_candidate.items():
        assert credit.stage_front_persisted is (candidate_id == surviving_source)
        assert credit.useful_descendant_observed is (
            True if candidate_id in selected else None
        )


def test_exact_archive_source_utility_selects_exploit_and_binds_complete_receipt() -> (
    None
):
    async def scenario():
        values = await _source_wave("portfolio_recombination_archive_aware")
        ids, problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        source_ids = tuple(
            sorted(candidate.candidate_id for candidate in source.candidates)
        )
        pair_ids = tuple(combinations(source_ids, 2))
        desired = pair_ids[-1]
        snapshot = _archive_snapshot()
        utilities = bind_portfolio_recombination_source_utilities(
            snapshot=snapshot,
            source_wave=source_wave,
            source_result=source,
            marginal_utilities={candidate_id: 1.0 for candidate_id in source_ids},
            exact_pair_utilities={
                pair: (50.0 if pair == desired else 0.0) for pair in pair_ids
            },
        )
        request = PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="archive_portfolio_union",
            path_family_exposures=_exposures(),
            source_archive_snapshot=snapshot,
            source_utilities=utilities,
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(request)
        return (
            problem,
            generator,
            selector,
            source,
            desired,
            snapshot,
            utilities,
            result,
        )

    (
        problem,
        generator,
        selector,
        source,
        desired,
        snapshot,
        utilities,
        result,
    ) = asyncio.run(scenario())
    decision = result.receipt.pair_decision
    assert type(decision) is ArchiveAwareDisjointPairSelectionDecision
    assert decision.exploit_pair_ids == desired
    assert decision.coverage_pair_ids != desired
    assert decision.source_utilities == utilities
    assert decision.source_utilities.context.archive_snapshot_sha256 == (
        snapshot.snapshot_sha256
    )
    assert problem.evaluations == 2
    assert generator.calls == 0
    assert selector.calls == 1
    assert set(desired).issubset(
        {candidate.candidate_id for candidate in source.candidates}
    )

    record = result.receipt.to_record()
    pair_record = record["pair_decision"]
    assert pair_record["policy_id"] == (
        "frozen_archive_exact_joint_source_utility_disjoint_pair"
    )
    assert pair_record["source_utility_receipt_sha256"] == utilities.receipt_sha256
    assert len(pair_record["source_utility_receipt"]["branches"]) == 4
    assert len(pair_record["source_utility_receipt"]["pair_utilities"]) == 6
    assert pair_record["exploit_pair_ids"] == [value.value for value in desired]
    assert pair_record["utility_scope"].startswith("exact_joint_observed_sources")


def test_archive_utility_binding_helpers_are_generic_exact_and_replay_stable() -> None:
    async def scenario():
        values = await _source_wave("portfolio_recombination_utility_binding")
        (
            _ids,
            _problem,
            _generator,
            _engine,
            _selector,
            _ancestor,
            source_wave,
            source,
        ) = values
        snapshot = _archive_snapshot()
        source_ids = tuple(
            sorted(candidate.candidate_id for candidate in source.candidates)
        )
        pairs = tuple(combinations(source_ids, 2))
        kwargs = {
            "snapshot": snapshot,
            "source_wave": source_wave,
            "source_result": source,
            "marginal_utilities": {
                candidate_id: float(index)
                for index, candidate_id in enumerate(source_ids, start=1)
            },
            "exact_pair_utilities": {
                pair: float(index) for index, pair in enumerate(pairs, start=1)
            },
        }
        return snapshot, source_wave, source, kwargs

    snapshot, source_wave, source, kwargs = asyncio.run(scenario())
    first = bind_portfolio_recombination_source_utilities(**kwargs)
    replay = bind_portfolio_recombination_source_utilities(**kwargs)

    assert first == replay
    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.context == frozen_archive_source_utility_context(snapshot)
    assert tuple(value.source for value in first.branches) == (
        portfolio_recombination_observed_sources(source)
    )
    assert first.source_generation == source_wave.generation
    assert "loss" not in str(first.to_record())


def test_archive_source_utility_boundary_rejects_partial_stale_and_foreign_inputs() -> (
    None
):
    async def scenario():
        values = await _source_wave("portfolio_recombination_utility_rejection")
        (
            _ids,
            _problem,
            _generator,
            _engine,
            _selector,
            ancestor,
            source_wave,
            source,
        ) = values
        return ancestor, source_wave, source

    ancestor, source_wave, source = asyncio.run(scenario())
    snapshot = _archive_snapshot()
    source_ids = tuple(
        sorted(candidate.candidate_id for candidate in source.candidates)
    )
    pairs = tuple(combinations(source_ids, 2))
    marginals = {candidate_id: 1.0 for candidate_id in source_ids}
    joint = {pair: 1.0 for pair in pairs}
    utilities = bind_portfolio_recombination_source_utilities(
        snapshot=snapshot,
        source_wave=source_wave,
        source_result=source,
        marginal_utilities=marginals,
        exact_pair_utilities=joint,
    )

    with pytest.raises(ValueError, match="supplied together"):
        PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="archive_portfolio_union",
            source_archive_snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="supplied together"):
        PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="archive_portfolio_union",
            source_utilities=utilities,
        )
    with pytest.raises(ValueError, match="stale for the source wave"):
        bind_portfolio_recombination_source_utilities(
            snapshot=_archive_snapshot(generation=2),
            source_wave=source_wave,
            source_result=source,
            marginal_utilities=marginals,
            exact_pair_utilities=joint,
        )
    with pytest.raises(ValueError, match="complete source pairs"):
        bind_portfolio_recombination_source_utilities(
            snapshot=snapshot,
            source_wave=source_wave,
            source_result=source,
            marginal_utilities=marginals,
            exact_pair_utilities={pair: 1.0 for pair in pairs[:-1]},
        )
    with pytest.raises(ValueError, match="foreign archive context"):
        PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="archive_portfolio_union",
            source_archive_snapshot=replace(
                snapshot,
                archive_sha256="a" * 64,
            ),
            source_utilities=utilities,
        )
    with pytest.raises(ValueError, match="foreign source wave"):
        PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="archive_portfolio_union",
            source_archive_snapshot=snapshot,
            source_utilities=replace(
                utilities,
                source_request_sha256="b" * 64,
            ),
        )


def test_legacy_recombination_receipt_shape_and_policy_remain_unchanged_by_default() -> (
    None
):
    async def scenario():
        values = await _source_wave("portfolio_recombination_legacy_shape")
        ids, _problem, _generator, engine, _selector, ancestor, source_wave, source = (
            values
        )
        request = PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="portfolio_union",
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(request)
        return request, result

    request, result = asyncio.run(scenario())
    assert request.source_archive_snapshot is None
    assert request.source_utilities is None
    assert type(result.receipt.pair_decision).__name__ == (
        "DisjointPairSelectionDecision"
    )
    record = result.receipt.to_record()
    assert record["schema_version"] == 1
    assert set(record) == {
        "schema_version",
        "source_wave_receipt_sha256",
        "source_request_sha256",
        "source_decision_sha256",
        "source_contract_sha256",
        "ancestor_candidate_id",
        "ancestor_configuration_sha256",
        "generation",
        "branches",
        "path_family_exposures",
        "pair_universe_size",
        "pair_attempts",
        "pair_decision",
        "selected_member_count",
        "concurrent_materialized_evaluation_wave",
        "members",
        "receipt_sha256",
    }
    assert record["pair_decision"]["policy_id"] == (
        "disjoint_parent_pair_exploit_coverage"
    )


class _PartialEngine:
    def __init__(self, delegate: AgenticEvolutionEngine) -> None:
        self.delegate = delegate

    async def run_materialized_invocations(self, items, *, reward_binding=None):
        values = await self.delegate.run_materialized_invocations(
            items,
            reward_binding=reward_binding,
        )
        return values[:-1]


class _ReorderedEngine:
    def __init__(self, delegate: AgenticEvolutionEngine) -> None:
        self.delegate = delegate

    async def run_materialized_invocations(self, items, *, reward_binding=None):
        values = await self.delegate.run_materialized_invocations(
            items,
            reward_binding=reward_binding,
        )
        return tuple(reversed(values))


class _WrongLineageEngine:
    def __init__(self, delegate: AgenticEvolutionEngine) -> None:
        self.delegate = delegate

    async def run_materialized_invocations(self, items, *, reward_binding=None):
        values = await self.delegate.run_materialized_invocations(
            items,
            reward_binding=reward_binding,
        )
        first = values[0]
        assert first.candidate is not None
        forged = replace(
            first.candidate, parent_ids=tuple(reversed(first.candidate.parent_ids))
        )
        return (replace(first, candidate=forged), *values[1:])


@pytest.mark.parametrize(
    ("wrapper", "message"),
    (
        (_PartialEngine, "partial recombination outcome wave"),
        (_ReorderedEngine, "differs from materialized recombination"),
        (_WrongLineageEngine, "differs from exact disjoint materialization"),
    ),
)
def test_partial_reordered_and_wrong_lineage_outcomes_fail_closed(
    wrapper, message
) -> None:
    async def scenario():
        values = await _source_wave(f"recombination_{wrapper.__name__.lower()}")
        ids, _problem, _generator, engine, _selector, ancestor, source_wave, source = (
            values
        )
        service = PortfolioRecombination(engine=wrapper(engine), ids=ids)
        request = PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="portfolio_union",
        )
        with pytest.raises(ValueError, match=message):
            await service.run(request)

    asyncio.run(scenario())


def test_overlapping_pair_universe_yields_typed_provider_free_skip() -> None:
    async def scenario():
        values = await _source_wave("portfolio_recombination_no_safe", disjoint=False)
        ids, problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="portfolio_union",
            )
        )
        return problem, generator, selector, result

    problem, generator, selector, result = asyncio.run(scenario())
    assert selector.calls == 1
    assert generator.calls == 0
    assert problem.evaluations == 0
    assert len(result.receipt.pair_attempts) == 3
    assert not any(value.replay_safe for value in result.receipt.pair_attempts)
    assert result.receipt.pair_decision.exploit is None
    assert result.receipt.members == ()
    assert result.outcomes == result.candidates == ()
    assert result.receipt.no_pair is not None
    assert result.receipt.no_pair.reason is (
        PortfolioRecombinationNoPairReason.NO_REPLAY_SAFE_DISJOINT_PAIR
    )


def test_candidate_infeasible_source_is_excluded_without_rank_resampling() -> None:
    async def scenario():
        evaluator = _CandidateInfeasibilityEvaluator({(0, 2, 0, 0)})
        values = await _source_wave(
            "portfolio_recombination_source_exclusion",
            detailed_evaluator=evaluator,
        )
        ids, _problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        scored_sources = portfolio_recombination_observed_sources(source)
        source_ids = tuple(value.candidate_id for value in scored_sources)
        snapshot = _archive_snapshot()
        utilities = bind_portfolio_recombination_source_utilities(
            snapshot=snapshot,
            source_wave=source_wave,
            source_result=source,
            marginal_utilities={value: 0.0 for value in source_ids},
            exact_pair_utilities={
                value: 0.0 for value in combinations(sorted(source_ids), 2)
            },
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="portfolio_union",
                source_archive_snapshot=snapshot,
                source_utilities=utilities,
            )
        )
        return generator, selector, source, scored_sources, result

    generator, selector, source, scored_sources, result = asyncio.run(scenario())
    receipt = result.receipt
    assert selector.calls == 1
    assert generator.calls == 0
    assert len(source.candidates) == 4
    assert len(source.scored_candidates) == len(scored_sources) == 3
    assert len(source.infeasible_candidates) == 1
    assert tuple(value.source_rank for value in scored_sources) == (1, 3, 4)
    assert tuple(value.rank for value in receipt.branches) == (1, 3, 4)
    assert len(receipt.source_exclusions) == 1
    exclusion = receipt.source_exclusions[0]
    assert exclusion.rank == 2
    assert exclusion.reason is (
        PortfolioRecombinationSourceExclusionReason.CANDIDATE_INFEASIBLE
    )
    assert (
        exclusion.candidate_id == source.receipt.members[1].materialization.candidate_id
    )
    assert len(receipt.pair_attempts) == 3
    assert len(result.candidates) == 2
    assert all(
        exclusion.candidate_id not in candidate.parent_ids
        for candidate in result.candidates
    )
    record = receipt.to_record()
    assert record["schema_version"] == 2
    assert record["ranked_source_count"] == 4
    assert record["scored_source_count"] == 3
    assert record["excluded_source_count"] == 1
    assert record["no_pair"] is None


def test_candidate_infeasible_union_is_retained_without_wave_abort_or_resampling() -> (
    None
):
    async def scenario():
        evaluator = _CandidateInfeasibilityEvaluator(
            {
                (1, 2, 0, 0),
                (1, 0, 3, 0),
                (1, 0, 0, 4),
                (0, 2, 3, 0),
                (0, 2, 0, 4),
                (0, 0, 3, 4),
            }
        )
        values = await _source_wave(
            "portfolio_recombination_union_infeasibility",
            detailed_evaluator=evaluator,
        )
        ids, _problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="portfolio_union",
            )
        )
        return generator, selector, source, evaluator, result

    generator, selector, source, evaluator, result = asyncio.run(scenario())
    assert generator.calls == 0
    assert selector.calls == 1
    assert len(source.scored_candidates) == 4
    assert len(result.candidates) == len(result.outcomes) == 2
    assert result.scored_candidates == ()
    assert len(result.infeasible_candidates) == 2
    assert len(evaluator.evaluations) == 7
    assert all(not candidate.valid for candidate in result.candidates)
    assert all(
        member.disposition.value == "candidate_infeasible"
        and member.candidate_failure is not None
        for member in result.receipt.members
    )
    record = result.receipt.to_record()
    assert record["schema_version"] == 3
    assert record["scored_member_count"] == 0
    assert record["candidate_infeasible_member_count"] == 2
    assert record["candidate_infeasibility_recourse"] == (
        "retain_selected_itt_reject_from_archive_no_resampling"
    )


def test_candidate_infeasible_recombinations_are_terminal_without_wave_abort() -> None:
    async def scenario():
        evaluator = _CandidateInfeasibilityEvaluator(
            {
                (1, 2, 0, 0),
                (1, 0, 3, 0),
                (1, 0, 0, 4),
                (0, 2, 3, 0),
                (0, 2, 0, 4),
                (0, 0, 3, 4),
            }
        )
        values = await _source_wave(
            "portfolio_recombination_terminal_infeasibility",
            detailed_evaluator=evaluator,
        )
        ids, _problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="portfolio_union",
            )
        )
        return generator, selector, result

    generator, selector, result = asyncio.run(scenario())
    assert selector.calls == 1
    assert generator.calls == 0
    assert len(result.outcomes) == len(result.candidates) == 2
    assert all(not candidate.valid for candidate in result.candidates)
    assert all(
        member.disposition.value == "candidate_infeasible"
        and member.candidate_failure is not None
        and member.parent_relations == ()
        for member in result.receipt.members
    )
    record = result.receipt.to_record()
    assert record["selected_member_count"] == 2
    assert all(
        member["disposition"] == "candidate_infeasible"
        and member["candidate_valid"] is False
        and member["candidate_failure"]["failure_category"] == "candidate"
        for member in record["members"]
    )


def test_insufficient_scored_sources_publish_no_pair_without_replacement() -> None:
    async def scenario():
        evaluator = _CandidateInfeasibilityEvaluator(
            {
                (0, 2, 0, 0),
                (0, 0, 3, 0),
                (0, 0, 0, 4),
            }
        )
        values = await _source_wave(
            "recombination_insufficient_sources",
            detailed_evaluator=evaluator,
        )
        ids, _problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="portfolio_union",
            )
        )
        return generator, selector, source, result

    generator, selector, source, result = asyncio.run(scenario())
    receipt = result.receipt
    assert selector.calls == 1
    assert generator.calls == 0
    assert len(source.candidates) == 4
    assert len(receipt.branches) == 1
    assert len(receipt.source_exclusions) == 3
    assert receipt.pair_attempts == ()
    assert receipt.members == result.outcomes == result.candidates == ()
    assert receipt.no_pair is not None
    assert receipt.no_pair.reason is (
        PortfolioRecombinationNoPairReason.INSUFFICIENT_SCORED_SOURCES
    )
    assert receipt.no_pair.scored_source_count == 1
    assert receipt.no_pair.excluded_source_count == 3
    assert receipt.to_record()["no_pair"]["evaluation_wave_dispatched"] is False


def test_archive_aware_overlapping_universe_retains_utility_receipt_on_skip() -> None:
    async def scenario():
        values = await _source_wave(
            "portfolio_recombination_archive_no_safe",
            disjoint=False,
        )
        ids, problem, generator, engine, selector, ancestor, source_wave, source = (
            values
        )
        snapshot = _archive_snapshot()
        source_ids = tuple(
            sorted(candidate.candidate_id for candidate in source.candidates)
        )
        pairs = tuple(combinations(source_ids, 2))
        utilities = bind_portfolio_recombination_source_utilities(
            snapshot=snapshot,
            source_wave=source_wave,
            source_result=source,
            marginal_utilities={candidate_id: 0.0 for candidate_id in source_ids},
            exact_pair_utilities={pair: 0.0 for pair in pairs},
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="archive_portfolio_union",
                source_archive_snapshot=snapshot,
                source_utilities=utilities,
            )
        )
        return problem, generator, selector, utilities, result

    problem, generator, selector, utilities, result = asyncio.run(scenario())
    assert type(result.receipt.pair_decision) is (
        ArchiveAwareDisjointPairSelectionDecision
    )
    assert result.receipt.pair_decision.source_utilities == utilities
    assert result.receipt.pair_decision.exploit is None
    assert result.receipt.members == result.outcomes == ()
    assert problem.evaluations == 0
    assert generator.calls == 0
    assert selector.calls == 1


def test_source_contract_and_result_outcome_tampering_are_rejected() -> None:
    async def scenario():
        values = await _source_wave("portfolio_recombination_tamper")
        ids, _problem, _generator, engine, _selector, ancestor, source_wave, source = (
            values
        )
        result = await PortfolioRecombination(engine=engine, ids=ids).run(
            PortfolioRecombinationWaveRequest(
                source_wave=source_wave,
                source_result=source,
                ancestor=ancestor,
                generation=2,
                label_prefix="portfolio_union",
            )
        )
        return ancestor, source_wave, source, result

    ancestor, source_wave, source, result = asyncio.run(scenario())
    changed_request = replace(
        source_wave.selection_request,
        operation="select_portfolio_changed",
    )
    with pytest.raises(ValueError, match="source result differs"):
        PortfolioRecombinationWaveRequest(
            source_wave=replace(source_wave, selection_request=changed_request),
            source_result=source,
            ancestor=ancestor,
            generation=2,
            label_prefix="portfolio_union",
        )
    with pytest.raises(ValueError, match="outcome differs"):
        PortfolioRecombinationWaveResult(
            receipt=result.receipt,
            outcomes=tuple(reversed(result.outcomes)),
        )
    changed_branches = (
        replace(
            result.receipt.branches[0],
            path_family_exposure=(result.receipt.branches[0].path_family_exposure + 1),
        ),
        *result.receipt.branches[1:],
    )
    with pytest.raises(ValueError, match="exposure differs"):
        replace(result.receipt, branches=changed_branches)
    selected_pair = result.receipt.members[0].pair_ids
    changed_attempts = tuple(
        replace(value, union_patch_sha256="f" * 64)
        if value.pair_ids == selected_pair
        else value
        for value in result.receipt.pair_attempts
    )
    with pytest.raises(ValueError, match="selected member differs"):
        replace(result.receipt, pair_attempts=changed_attempts)


def test_generation_must_follow_all_source_children() -> None:
    async def scenario():
        values = await _source_wave("portfolio_recombination_generation")
        (
            _ids,
            _problem,
            _generator,
            _engine,
            _selector,
            ancestor,
            source_wave,
            source,
        ) = values
        return ancestor, source_wave, source

    ancestor, source_wave, source = asyncio.run(scenario())
    with pytest.raises(ValueError, match="follow every source child"):
        PortfolioRecombinationWaveRequest(
            source_wave=source_wave,
            source_result=source,
            ancestor=ancestor,
            generation=1,
            label_prefix="portfolio_union",
        )


def test_recombination_api_is_available_from_both_public_facades() -> None:
    import agent_evolve
    import agent_evolve.agentic as agentic

    for name, expected in (
        (
            "ArchiveAwareDisjointPairSelectionDecision",
            ArchiveAwareDisjointPairSelectionDecision,
        ),
        (
            "ArchiveAwareDisjointParentPairPolicy",
            ArchiveAwareDisjointParentPairPolicy,
        ),
        ("FrozenArchiveBranchUtility", FrozenArchiveBranchUtility),
        ("FrozenArchiveSourcePairUtility", FrozenArchiveSourcePairUtility),
        ("FrozenArchiveSourceUtilityContext", FrozenArchiveSourceUtilityContext),
        ("FrozenArchiveSourceUtilityReceipt", FrozenArchiveSourceUtilityReceipt),
        ("ObservedSourceBranch", ObservedSourceBranch),
        ("PortfolioRecombination", PortfolioRecombination),
        ("PortfolioRecombinationWaveRequest", PortfolioRecombinationWaveRequest),
        ("PortfolioRecombinationWaveResult", PortfolioRecombinationWaveResult),
        (
            "bind_portfolio_recombination_source_utilities",
            bind_portfolio_recombination_source_utilities,
        ),
        (
            "frozen_archive_source_utility_context",
            frozen_archive_source_utility_context,
        ),
        (
            "portfolio_recombination_observed_sources",
            portfolio_recombination_observed_sources,
        ),
    ):
        assert getattr(agentic, name) is expected
        assert getattr(agent_evolve, name) is expected

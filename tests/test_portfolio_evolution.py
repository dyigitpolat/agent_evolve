from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import AgenticEvolutionEngine
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluationPayload,
    EvaluatorIdentity,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.outcome_relation import objective_pareto_outcome_binding
from agent_evolve.application.portfolio_evolution import (
    PortfolioCandidateFailureEvidence,
    PortfolioEvolution,
    PortfolioMemberDisposition,
    PortfolioMemoryCreditBatchReceipt,
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryCreditPlan,
    PortfolioPendingMemoryCredit,
    PortfolioRewardAggregationBinding,
    PortfolioVariationWaveResult,
    PortfolioVariationWaveRequest,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.outcome import FailureCategory, FailureCode, FailureRecord
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.randomized_subset import (
    InsightSelectionDecision,
    InsightSelectionMode,
)
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    MemoryAssignmentArm,
    ResolvedInsightAssignment,
)
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


def _frozen_object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _rebind_credit_plan(
    credit: PortfolioMemoryCreditPlan,
    *,
    decision: InsightSelectionDecision | None = None,
    credit_unit_id=None,
    context_projection: PortfolioMemoryContextProjectionBinding | None = None,
) -> PortfolioMemoryCreditPlan:
    """Re-issue the typed assignment when a test changes its treatment plan."""

    rebound_decision = credit.decision if decision is None else decision
    rebound_credit_unit_id = (
        credit.credit_unit_id if credit_unit_id is None else credit_unit_id
    )
    snapshot = CausalSearchScorePolicy(
        uncertainty_scale=0.0,
        exploration_weight=0.0,
    ).genesis(
        exact_context_hash=rebound_decision.context_hash,
        estimand_stratum_hash=credit.assignment.estimand_stratum_hash,
        priors=dict(rebound_decision.score_snapshot),
    )
    assignment = ResolvedInsightAssignment.resolve(
        credit_unit_id=rebound_credit_unit_id,
        snapshot=snapshot,
        expected_snapshot_sha256=snapshot.snapshot_sha256,
        block_id=credit.assignment.block_id,
        arm=credit.assignment.arm,
        selection_decision=rebound_decision,
        prompt_shape_sha256=credit.card_snapshot_sha256,
        credit_mode=credit.assignment.credit_mode,
    )
    return replace(
        credit,
        decision=rebound_decision,
        credit_unit_id=rebound_credit_unit_id,
        score_snapshot=snapshot,
        assignment=assignment,
        context_projection=context_projection,
    )


class _NoCandidateGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, request):
        del request
        self.calls += 1
        raise AssertionError("portfolio children are engine-materialized")

    async def reflect(self, request):
        del request
        self.calls += 1
        raise AssertionError("portfolio wave does not reflect")


class _Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int
    y: int


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
        return "Two integer coordinates."

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
        time.sleep(0.04)
        with self._lock:
            self.active -= 1
        return {"loss": float(parsed.x + parsed.y)}

    def reset_evidence(self) -> None:
        with self._lock:
            self.evaluations = 0
            self.active = 0
            self.max_active = 0


class _CandidateInfeasibilityEvaluator:
    evaluator_identity = EvaluatorIdentity(
        evaluator_id="portfolio_candidate_infeasibility_test",
        evaluator_version=1,
        evaluator_context_sha256=hashlib.sha256(
            b"portfolio-candidate-infeasibility-test-v1"
        ).hexdigest(),
    )

    def __init__(self) -> None:
        self.configurations: list[tuple[int, int]] = []

    def reset_evidence(self) -> None:
        self.configurations.clear()

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        parsed = _Configuration.model_validate(configuration, strict=True)
        self.configurations.append((parsed.x, parsed.y))
        if (parsed.x, parsed.y) == (2, 2):
            return DetailedEvaluationPayload(
                failure=FailureRecord(
                    category=FailureCategory.CANDIDATE,
                    code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
                    message="sealed configuration violates the benchmark constraint",
                ),
                objectives=(),
                violations=(),
                checks=(),
                receipt=None,
                evaluator=self.evaluator_identity,
            )
        return DetailedEvaluationPayload(
            failure=None,
            objectives=(("loss", float(parsed.x + parsed.y)),),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


class _RankedSelector:
    def __init__(self, *, telemetry: bool = True) -> None:
        self.calls = 0
        self.with_telemetry = telemetry

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        self.calls += 1
        predictions = (MetricEffectPrediction("loss", MetricEffectDirection.DECREASE),)
        drafts = tuple(
            PortfolioMemberDraft(
                option_id=option_id,
                supporting_card_keys=(
                    request.cards[index % len(request.cards)].card_key,
                ),
                effect_predictions=predictions,
                design_rationale=f"Rank sealed option {option_id} at position {index + 1}.",
            )
            for index, option_id in enumerate(("alpha.x1", "beta.y1", "gamma.xy"))
        )
        decision = resolve_ranked_portfolio_decision(
            request,
            drafts,
            policy_id="fake_ranked_selector",
            policy_version=1,
            policy_definition_sha256="b" * 64,
        )
        telemetry = (
            AgenticCallTelemetry(
                requested_model="fake/selector",
                resolved_model="fake/selector-v1",
                resolved_provider="provider-free",
                provider_response_id="response-portfolio-wave",
                finish_reason="stop",
                input_tokens=100,
                output_tokens=20,
                reasoning_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0.001"),
                latency_ns=1_000,
            )
            if self.with_telemetry
            else None
        )
        return PortfolioSelectionResult(decision=decision, telemetry=telemetry)


def _contract(parent: FrozenJsonObject) -> FiniteVariationContract:
    parent_sha256 = typed_json_sha256(parent)
    return FiniteVariationContract(
        catalog_id="portfolio_wave_test",
        catalog_version=1,
        catalog_definition_sha256="a" * 64,
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen_object(child),
                family=family,
                description=description,
            )
            for option_id, family, child, description in (
                ("alpha.x1", "alpha", {"x": 1, "y": 0}, "Set x to one."),
                ("beta.y1", "beta", {"x": 0, "y": 1}, "Set y to one."),
                ("gamma.xy", "gamma", {"x": 2, "y": 2}, "Set both to two."),
            )
        ),
    )


def _insight(memory: InsightMemoryBank, suffix: str, score: float):
    entry, added = memory.add(
        InsightDraft(
            claim=f"Portfolio claim {suffix}.",
            trigger=f"Portfolio trigger {suffix}.",
            mechanism=f"Portfolio mechanism {suffix}.",
            affected_paths=("$.x",),
            evidence_summary=f"Portfolio evidence {suffix}.",
            confidence=0.5,
        ),
        initial_score=score,
    )
    assert added
    return entry


def _memory_decision(
    *,
    context_hash: str,
    eligible,
    selected,
) -> InsightSelectionDecision:
    score_snapshot = tuple(
        (reference, float(len(eligible) - index))
        for index, reference in enumerate(eligible)
    )
    return InsightSelectionDecision(
        context_hash=context_hash,
        eligible=eligible,
        selected=selected,
        exploitation_subset=selected,
        score_snapshot=score_snapshot,
        subset_size=len(selected),
        exploration_probability=Fraction(1, 2),
        mode=InsightSelectionMode.EXPLOIT,
        selected_subset_probability=Fraction(2, 3),
    )


async def _build_wave(
    namespace: str,
    *,
    selector: _RankedSelector | None = None,
    detailed_evaluator=None,
):
    ids = DeterministicIdFactory(namespace)
    problem = _ConcurrentProblem()
    generator = _NoCandidateGenerator()
    memory = InsightMemoryBank(id_factory=ids)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=7,
        evaluator_concurrency=3,
        detailed_evaluator=detailed_evaluator,
        outcome_relation_binding=(
            None
            if detailed_evaluator is None
            else objective_pareto_outcome_binding(problem.objectives)
        ),
    )
    parent = await engine.register_seed({"x": 0, "y": 0}, label="parent")
    problem.reset_evidence()
    entries = tuple(
        _insight(memory, suffix, score)
        for suffix, score in (("a", 3.0), ("b", 2.0), ("c", 1.0))
    )
    selected = tuple(sorted(entry.reference for entry in entries[:2]))
    selected_entries = entries[:2]
    cards = tuple(
        PortfolioCard(
            card_key=f"card.{index}",
            reference=entry.reference,
            content_sha256=entry.draft.content_sha256,
            evidence_sha256=str(index + 2) * 64,
            prompt_payload=_frozen_object({"claim": f"card {index}"}),
        )
        for index, entry in enumerate(selected_entries, start=1)
    )
    request = PortfolioSelectionRequest(
        call_id=ids.new_llm_call_id(),
        operation="select_portfolio",
        instruction="Select three ranked finite options for concurrent evaluation.",
        context=_frozen_object({"benchmark": "provider-free-test"}),
        finite_variation_contract=_contract(parent.configuration),
        cards=cards,
        portfolio_size=3,
        required_metric_ids=("loss",),
        min_distinct_families=3,
    )
    decision = _memory_decision(
        context_hash=request.context_sha256,
        eligible=tuple(sorted(entry.reference for entry in entries)),
        selected=selected,
    )
    aggregation = PortfolioRewardAggregationBinding(
        aggregate=lambda outcomes: float(max(outcome.reward for outcome in outcomes)),
        aggregation_id="max_member_reward",
        aggregation_version=1,
        definition_sha256="c" * 64,
    )
    credit_unit_id = ids.new_operator_invocation_id()
    snapshot = CausalSearchScorePolicy(
        uncertainty_scale=0.0,
        exploration_weight=0.0,
    ).genesis(
        exact_context_hash=decision.context_hash,
        estimand_stratum_hash="d" * 64,
        priors=dict(decision.score_snapshot),
    )
    assignment = ResolvedInsightAssignment.resolve(
        credit_unit_id=credit_unit_id,
        snapshot=snapshot,
        expected_snapshot_sha256=snapshot.snapshot_sha256,
        block_id="portfolio_test_block",
        arm=MemoryAssignmentArm.DIAGNOSTIC,
        selection_decision=decision,
        prompt_shape_sha256=request.card_snapshot_sha256,
    )
    wave = PortfolioVariationWaveRequest(
        selection_request=request,
        parent=parent,
        generation=1,
        label_prefix="portfolio_wave",
        memory_credit=PortfolioMemoryCreditPlan(
            decision=decision,
            credit_unit_id=credit_unit_id,
            aggregation=aggregation,
            card_snapshot_sha256=request.card_snapshot_sha256,
            score_snapshot=snapshot,
            assignment=assignment,
        ),
    )
    active_selector = _RankedSelector() if selector is None else selector
    return (
        ids,
        problem,
        generator,
        memory,
        engine,
        active_selector,
        wave,
    )


def test_one_selection_materializes_three_concurrent_candidates_and_one_trial() -> None:
    async def scenario():
        values = await _build_wave("portfolio_wave_success")
        ids, problem, generator, memory, engine, selector, wave = values
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave)
        return problem, generator, memory, selector, wave, result

    problem, generator, memory, selector, wave, result = asyncio.run(scenario())
    receipt = result.receipt

    assert selector.calls == 1
    assert generator.calls == 0
    assert problem.evaluations == 3
    assert problem.max_active == 3
    assert tuple(member.materialization.rank for member in receipt.members) == (1, 2, 3)
    assert len({member.materialization.candidate_id for member in receipt.members}) == 3
    assert (
        len(
            {
                member.materialization.child_configuration_sha256
                for member in receipt.members
            }
        )
        == 3
    )
    assert (
        len({member.materialization.receipt_sha256 for member in receipt.members}) == 3
    )
    assert all(
        member.materialization.to_record()["model_authored_configuration_fields"] == 0
        for member in receipt.members
    )
    assert len(memory.trials) == 1
    assert result.candidates == tuple(outcome.candidate for outcome in result.outcomes)
    assert result.selection_decision is not None
    assert result.selection_decision.decision_sha256 == receipt.decision_sha256
    audit_record = result.selection_decision_audit_record
    assert audit_record == (result.selection_decision.to_audit_record())
    assert audit_record is not None
    audit_members = audit_record["members"]
    assert type(audit_members) is list
    assert audit_members[0]["design_rationale"] == (
        "Rank sealed option alpha.x1 at position 1."
    )
    assert audit_members[0]["supporting_card_keys"] == ["card.1"]
    assert audit_members[0]["effect_predictions"] == [
        {"metric_id": "loss", "direction": "decrease"}
    ]
    assert tuple(candidate.candidate_id for candidate in result.candidates) == tuple(
        member.materialization.candidate_id for member in receipt.members
    )
    assert memory.trials[0].candidate_ids == tuple(
        member.materialization.candidate_id for member in receipt.members
    )
    assert receipt.memory_credit is not None
    assert receipt.memory_credit.to_record()["memory_trial_count"] == 1
    assert receipt.memory_credit.context_projection.projection_id == (
        "exact_context_identity"
    )
    assert result.action_attributions == receipt.action_attributions
    assert len(result.action_attributions) == 3
    assert tuple(value.rank for value in result.action_attributions) == (1, 2, 3)
    request_cards = {value.card_key: value for value in wave.selection_request.cards}
    assert result.selection_decision is not None
    for attribution, selected, member in zip(
        result.action_attributions,
        result.selection_decision.members,
        receipt.members,
        strict=True,
    ):
        assert attribution.option_id == selected.option_id
        assert attribution.family == selected.family
        assert attribution.effect_predictions == selected.effect_predictions
        assert (
            attribution.design_rationale_sha256
            == hashlib.sha256(selected.design_rationale.encode("utf-8")).hexdigest()
        )
        assert attribution.materialization_receipt_sha256 == (
            member.materialization.receipt_sha256
        )
        assert attribution.outcome_sha256 == member.outcome_sha256
        assert attribution.candidate_id == member.materialization.candidate_id
        assert attribution.operator_invocation_id == member.operator_invocation_id
        assert attribution.supporting_card_keys == selected.supporting_card_keys
        for cited in attribution.supporting_cards:
            source = request_cards[cited.card_key]
            assert cited.reference == source.reference
            assert cited.content_sha256 == source.content_sha256
            assert cited.evidence_sha256 == source.evidence_sha256
        assert attribution.to_record()["attribution_scope"] == (
            "post_treatment_diagnostic_not_causal_credit"
        )


def test_candidate_infeasibility_remains_in_full_ranked_itt_and_memory_aggregate() -> None:
    observed_aggregate_members: list[tuple[tuple[bool, float], ...]] = []

    def aggregate(outcomes) -> float:
        observed_aggregate_members.append(
            tuple(
                (
                    outcome.candidate is not None and outcome.candidate.valid,
                    outcome.reward,
                )
                for outcome in outcomes
            )
        )
        return float(sum(outcome.reward for outcome in outcomes) / len(outcomes))

    async def scenario():
        evaluator = _CandidateInfeasibilityEvaluator()
        values = await _build_wave(
            "portfolio_wave_candidate_infeasible",
            detailed_evaluator=evaluator,
        )
        ids, _problem, generator, memory, engine, selector, wave = values
        evaluator.reset_evidence()  # Exclude the parent registration evaluation.
        credit = wave.memory_credit
        assert credit is not None
        wave = replace(
            wave,
            memory_credit=replace(
                credit,
                aggregation=PortfolioRewardAggregationBinding(
                    aggregate=aggregate,
                    aggregation_id="mean_itt_member_reward",
                    aggregation_version=1,
                    definition_sha256="9" * 64,
                ),
            ),
        )
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave)
        return evaluator, generator, memory, selector, result

    evaluator, generator, memory, selector, result = asyncio.run(scenario())

    assert selector.calls == 1
    assert generator.calls == 0
    assert sorted(evaluator.configurations) == [(0, 1), (1, 0), (2, 2)]
    assert len(result.outcomes) == len(result.candidates) == 3
    assert len(result.scored_candidates) == 2
    assert len(result.infeasible_candidates) == 1
    infeasible = result.infeasible_candidates[0]
    assert infeasible.configuration_dict == {"x": 2, "y": 2}
    assert tuple(member.disposition for member in result.receipt.members) == (
        PortfolioMemberDisposition.SCORED,
        PortfolioMemberDisposition.SCORED,
        PortfolioMemberDisposition.CANDIDATE_INFEASIBLE,
    )
    member = result.receipt.members[2]
    assert member.engine_reward == -1.0
    assert member.parent_relations == ()
    assert not member.dominates_any_parent
    assert not member.better_than_any_parent
    assert type(member.candidate_failure) is PortfolioCandidateFailureEvidence
    assert member.candidate_failure.failure_code is (
        FailureCode.EVALUATOR_DECLARED_INFEASIBLE
    )
    assert member.candidate_failure.detailed_evaluation_sha256 == (
        infeasible.detailed_evaluation.evidence_sha256
    )
    record = member.to_record()
    assert record["candidate_valid"] is False
    assert record["disposition"] == "candidate_infeasible"
    assert record["engine_reward_hex"] == (-1.0).hex()
    assert type(record["candidate_failure"]) is dict

    # The infeasible rank is neither dropped nor replaced before the
    # preregistered full-wave estimator and causal memory unit are committed.
    expected_aggregate_members = tuple(
        (outcome.candidate.valid, outcome.reward)
        for outcome in result.outcomes
        if outcome.candidate is not None
    )
    assert observed_aggregate_members == [expected_aggregate_members]
    assert len(memory.trials) == 1
    assert memory.trials[0].candidate_ids == tuple(
        member.materialization.candidate_id for member in result.receipt.members
    )
    assert memory.trials[0].reward == sum(
        outcome.reward for outcome in result.outcomes
    ) / 3


class _MissingCandidateFailureEvidenceEngine:
    def __init__(self, delegate: AgenticEvolutionEngine) -> None:
        self.delegate = delegate

    async def run_materialized_invocations(self, items, *, reward_binding=None):
        outcomes = await self.delegate.run_materialized_invocations(
            items,
            reward_binding=reward_binding,
        )
        rewritten = []
        for outcome in outcomes:
            candidate = outcome.candidate
            if candidate is not None and not candidate.valid:
                outcome = replace(
                    outcome,
                    candidate=replace(candidate, detailed_evaluation=None),
                )
            rewritten.append(outcome)
        return tuple(rewritten)


def test_candidate_infeasibility_without_detailed_failure_evidence_fails_closed() -> None:
    async def scenario():
        evaluator = _CandidateInfeasibilityEvaluator()
        values = await _build_wave(
            "portfolio_infeasible_missing_evidence",
            detailed_evaluator=evaluator,
        )
        ids, _problem, _generator, memory, engine, selector, wave = values
        evaluator.reset_evidence()
        service = PortfolioEvolution(
            engine=_MissingCandidateFailureEvidenceEngine(engine),
            selector=selector,
            ids=ids,
            memory=memory,
        )
        with pytest.raises(
            ValueError,
            match="candidate infeasibility requires detailed evaluator evidence",
        ):
            await service.run(wave)
        return evaluator, memory, selector

    evaluator, memory, selector = asyncio.run(scenario())
    assert selector.calls == 1
    assert sorted(evaluator.configurations) == [(0, 1), (1, 0), (2, 2)]
    assert memory.trials == ()


def test_pending_memory_credit_commits_as_one_canonical_stage_batch() -> None:
    async def once(*, reverse_at_barrier: bool):
        values = await _build_wave("portfolio_deferred_credit_batch")
        ids, _problem, _generator, memory, engine, selector, first_wave = values
        second_parent = await engine.register_seed(
            {"x": 10, "y": 10},
            label="second_parent",
        )
        second_request = replace(
            first_wave.selection_request,
            call_id=ids.new_llm_call_id(),
            finite_variation_contract=_contract(second_parent.configuration),
        )
        first_credit = first_wave.memory_credit
        assert first_credit is not None
        second_wave = replace(
            first_wave,
            selection_request=second_request,
            parent=second_parent,
            label_prefix="portfolio_wave_second",
            memory_credit=_rebind_credit_plan(
                first_credit,
                credit_unit_id=ids.new_operator_invocation_id(),
            ),
        )
        service = PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        )
        pending = await asyncio.gather(
            service.run(first_wave, defer_memory_credit=True),
            service.run(second_wave, defer_memory_credit=True),
        )
        assert memory.trials == ()
        assert all(
            type(result.pending_memory_credit) is PortfolioPendingMemoryCredit
            and result.receipt.memory_credit is None
            for result in pending
        )
        with pytest.raises(ValueError, match="repeats a memory credit unit"):
            service.commit_pending_memory_credit_batch((pending[0], pending[0]))
        assert memory.trials == ()
        barrier_input = (
            tuple(reversed(pending)) if reverse_at_barrier else tuple(pending)
        )
        committed, batch = service.commit_pending_memory_credit_batch(barrier_input)
        assert type(batch) is PortfolioMemoryCreditBatchReceipt
        assert all(
            result.pending_memory_credit is None
            and result.receipt.memory_credit is not None
            for result in committed
        )
        return (
            batch.to_record(),
            tuple(trial.credit_unit_id.value for trial in memory.trials),
        )

    forward = asyncio.run(once(reverse_at_barrier=False))
    reversed_at_barrier = asyncio.run(once(reverse_at_barrier=True))
    assert forward == reversed_at_barrier
    batch, trial_ids = forward
    assert batch["publication_scope"] == "post_concurrent_generation_barrier"
    assert batch["credit_count"] == 2
    assert trial_ids == tuple(sorted(trial_ids))


def test_cross_context_memory_credit_requires_authenticated_projection() -> None:
    async def scenario():
        values = await _build_wave("portfolio_cross_context_projection")
        ids, _problem, _generator, memory, engine, selector, wave = values
        credit = wave.memory_credit
        assert credit is not None
        selector_context = _frozen_object(
            {
                "benchmark": "provider-free-test",
                "memory_estimand_context": {
                    "estimand": "stable causal stratum",
                    "schema_version": 1,
                },
            }
        )
        projected = dict(selector_context.items)["memory_estimand_context"]
        assert type(projected) is FrozenJsonObject
        estimand_context = typed_json_sha256(projected)
        decision = replace(credit.decision, context_hash=estimand_context)
        unbound = _rebind_credit_plan(credit, decision=decision)
        selection_request = replace(
            wave.selection_request,
            context=selector_context,
        )
        with pytest.raises(ValueError, match="explicit authenticated context"):
            replace(
                wave,
                selection_request=selection_request,
                memory_credit=unbound,
            )

        projection = PortfolioMemoryContextProjectionBinding.from_selector_context(
            selector_context
        )
        bound = _rebind_credit_plan(
            credit,
            decision=decision,
            context_projection=projection,
        )
        accepted_wave = replace(
            wave,
            selection_request=selection_request,
            memory_credit=bound,
        )
        with pytest.raises(ValueError, match="reserved memory estimand object"):
            PortfolioMemoryContextProjectionBinding.from_selector_context(
                wave.selection_request.context
            )
        with pytest.raises(ValueError, match="reserved memory estimand object"):
            PortfolioMemoryContextProjectionBinding.from_selector_context(
                _frozen_object({"memory_estimand_context": "not-an-object"})
            )
        arbitrary = PortfolioMemoryContextProjectionBinding(
            estimand_context_sha256="d" * 64,
            selector_context_sha256=selection_request.context_sha256,
            projection_key="memory_estimand_context",
        )
        with pytest.raises(ValueError, match="subtree differs from estimand"):
            replace(
                accepted_wave,
                memory_credit=_rebind_credit_plan(
                    credit,
                    decision=replace(decision, context_hash="d" * 64),
                    context_projection=arbitrary,
                ),
            )
        changed_selector = replace(
            selection_request,
            context=_frozen_object(
                {
                    "benchmark": "different per-wave evidence",
                    "memory_estimand_context": {
                        "estimand": "stable causal stratum",
                        "schema_version": 1,
                    },
                }
            ),
        )
        with pytest.raises(ValueError, match="selector hash differs"):
            replace(
                accepted_wave,
                selection_request=changed_selector,
            )
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(accepted_wave)
        return projection, accepted_wave.selection_request.context, result

    projection, selector_context, result = asyncio.run(scenario())
    receipt = result.receipt.memory_credit
    assert receipt is not None
    assert receipt.context_projection == projection
    assert receipt.selection_decision_context_sha256 == (
        projection.estimand_context_sha256
    )
    assert (
        typed_json_sha256(receipt.context_projection.replay(selector_context))
        == receipt.selection_decision_context_sha256
    )
    record = receipt.to_record()["context_projection"]
    assert record["binding_sha256"] == projection.binding_sha256


def test_portfolio_wave_receipt_is_deterministic_and_telemetry_bound() -> None:
    async def once():
        ids, _problem, _generator, memory, engine, selector, wave = await _build_wave(
            "portfolio_wave_deterministic"
        )
        return await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave)

    first = asyncio.run(once())
    second = asyncio.run(once())

    assert first.receipt.to_record() == second.receipt.to_record()
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256
    legacy_default = replace(first.receipt, action_attributions=())
    assert "action_attributions" not in legacy_default.to_record()
    with pytest.raises(ValueError, match="telemetry digest"):
        replace(first.receipt, selection_telemetry_sha256="f" * 64)
    with pytest.raises(ValueError, match="differs from its receipt member"):
        PortfolioVariationWaveResult(
            receipt=first.receipt,
            outcomes=tuple(reversed(first.outcomes)),
        )


def test_wave_result_binds_auditable_decision_and_accepts_legacy_none() -> None:
    async def once(namespace: str):
        ids, _problem, _generator, memory, engine, selector, wave = await _build_wave(
            namespace
        )
        return await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave)

    result = asyncio.run(once("portfolio_wave_decision_join"))
    foreign = asyncio.run(once("portfolio_wave_foreign_decision"))
    decision = result.selection_decision
    foreign_decision = foreign.selection_decision
    assert decision is not None
    assert foreign_decision is not None

    tampered = replace(
        decision,
        members=(
            replace(
                decision.members[0],
                design_rationale="A tampered model-authored rationale.",
            ),
            *decision.members[1:],
        ),
    )
    with pytest.raises(ValueError, match="selection decision differs"):
        PortfolioVariationWaveResult(
            receipt=result.receipt,
            outcomes=result.outcomes,
            selection_decision=tampered,
        )
    with pytest.raises(ValueError, match="selection decision differs"):
        PortfolioVariationWaveResult(
            receipt=result.receipt,
            outcomes=result.outcomes,
            selection_decision=foreign_decision,
        )

    foreign_policy_receipt = replace(
        result.receipt,
        selection_policy_id="foreign_ranked_selector",
    )
    with pytest.raises(ValueError, match="wave receipt identity"):
        PortfolioVariationWaveResult(
            receipt=foreign_policy_receipt,
            outcomes=result.outcomes,
            selection_decision=decision,
        )

    first_member = result.receipt.members[0]
    foreign_materialization = replace(
        first_member.materialization,
        option_identity_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="action attribution differs"):
        replace(
            result.receipt,
            members=(
                replace(first_member, materialization=foreign_materialization),
                *result.receipt.members[1:],
            ),
        )
    legacy_receipt = replace(result.receipt, action_attributions=())
    foreign_materialization_receipt = replace(
        legacy_receipt,
        members=(
            replace(first_member, materialization=foreign_materialization),
            *result.receipt.members[1:],
        ),
    )
    with pytest.raises(ValueError, match="wave materialization"):
        PortfolioVariationWaveResult(
            receipt=foreign_materialization_receipt,
            outcomes=result.outcomes,
            selection_decision=decision,
        )

    legacy = PortfolioVariationWaveResult(
        receipt=result.receipt,
        outcomes=result.outcomes,
    )
    assert legacy.selection_decision is None
    assert legacy.selection_decision_audit_record is None


def test_action_attribution_rejects_card_action_and_outcome_tampering() -> None:
    async def once():
        ids, _problem, _generator, memory, engine, selector, wave = await _build_wave(
            "portfolio_action_attribution_tamper"
        )
        return await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave)

    result = asyncio.run(once())
    decision = result.selection_decision
    assert decision is not None
    first = result.action_attributions[0]

    source_card = first.supporting_cards[0]
    for changed_card in (
        replace(source_card, content_sha256="e" * 64),
        replace(source_card, evidence_sha256="e" * 64),
        replace(
            source_card,
            reference=result.action_attributions[1].supporting_cards[0].reference,
        ),
    ):
        assert changed_card.binding_sha256 != source_card.binding_sha256
        assert (
            replace(first, supporting_cards=(changed_card,)).receipt_sha256
            != first.receipt_sha256
        )

    wrong_card = replace(first.supporting_cards[0], card_key="card.2")
    card_tampered = replace(
        result.receipt,
        action_attributions=(
            replace(first, supporting_cards=(wrong_card,)),
            *result.action_attributions[1:],
        ),
    )
    with pytest.raises(ValueError, match="action attribution differs"):
        PortfolioVariationWaveResult(
            receipt=card_tampered,
            outcomes=result.outcomes,
            selection_decision=decision,
        )

    rationale_tampered = replace(
        result.receipt,
        action_attributions=(
            replace(first, design_rationale_sha256="f" * 64),
            *result.action_attributions[1:],
        ),
    )
    with pytest.raises(ValueError, match="action attribution differs"):
        PortfolioVariationWaveResult(
            receipt=rationale_tampered,
            outcomes=result.outcomes,
            selection_decision=decision,
        )

    with pytest.raises(ValueError, match="action attribution differs"):
        replace(
            result.receipt,
            action_attributions=(
                replace(first, option_id="beta.y1"),
                *result.action_attributions[1:],
            ),
        )
    with pytest.raises(ValueError, match="action attribution differs"):
        replace(
            result.receipt,
            action_attributions=(
                replace(first, outcome_sha256="f" * 64),
                *result.action_attributions[1:],
            ),
        )
    with pytest.raises(ValueError, match="request-card snapshot"):
        replace(
            result.receipt,
            action_attributions=(
                first,
                replace(
                    result.action_attributions[1],
                    card_snapshot_sha256="f" * 64,
                ),
                *result.action_attributions[2:],
            ),
        )
    with pytest.raises(ValueError, match="action attribution differs"):
        replace(
            result.receipt,
            action_attributions=(
                replace(
                    first,
                    candidate_id=result.receipt.members[1].materialization.candidate_id,
                ),
                *result.action_attributions[1:],
            ),
        )


def test_wave_rejects_missing_selector_telemetry_before_evaluation() -> None:
    async def scenario():
        selector = _RankedSelector(telemetry=False)
        ids, problem, _generator, memory, engine, _, wave = await _build_wave(
            "portfolio_wave_missing_telemetry",
            selector=selector,
        )
        service = PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        )
        with pytest.raises(ValueError, match="requires exact call telemetry"):
            await service.run(wave)
        return selector, problem, memory

    selector, problem, memory = asyncio.run(scenario())
    assert selector.calls == 1
    assert problem.evaluations == 0
    assert memory.trials == ()


class _DriftingSelector(_RankedSelector):
    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        object.__setattr__(request, "operation", "select_portfolio_drifted")
        return await super().select(request)


def test_wave_rejects_request_drift_during_selection_before_evaluation() -> None:
    async def scenario():
        selector = _DriftingSelector()
        ids, problem, _generator, memory, engine, _, wave = await _build_wave(
            "portfolio_wave_request_drift",
            selector=selector,
        )
        service = PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        )
        with pytest.raises(ValueError, match="drifted during selection"):
            await service.run(wave)
        return selector, problem, memory

    selector, problem, memory = asyncio.run(scenario())
    assert selector.calls == 1
    assert problem.evaluations == 0
    assert memory.trials == ()


class _PartialEngine:
    def __init__(self, delegate: AgenticEvolutionEngine) -> None:
        self.delegate = delegate

    async def run_materialized_invocations(self, items, *, reward_binding=None):
        outcomes = await self.delegate.run_materialized_invocations(
            items,
            reward_binding=reward_binding,
        )
        return outcomes[:-1]


class _ReorderedEngine:
    def __init__(self, delegate: AgenticEvolutionEngine) -> None:
        self.delegate = delegate

    async def run_materialized_invocations(self, items, *, reward_binding=None):
        outcomes = await self.delegate.run_materialized_invocations(
            items,
            reward_binding=reward_binding,
        )
        return tuple(reversed(outcomes))


@pytest.mark.parametrize(
    ("wrapper", "message"),
    (
        (_PartialEngine, "partial portfolio outcome wave"),
        (_ReorderedEngine, "differs from its materialized member"),
    ),
)
def test_wave_rejects_partial_or_misjoined_engine_outcomes(wrapper, message) -> None:
    async def scenario():
        ids, _problem, _generator, memory, engine, selector, wave = await _build_wave(
            f"portfolio_wave_{wrapper.__name__.lower()}"
        )
        service = PortfolioEvolution(
            engine=wrapper(engine),
            selector=selector,
            ids=ids,
            memory=memory,
        )
        with pytest.raises(ValueError, match=message):
            await service.run(wave)
        return memory

    memory = asyncio.run(scenario())
    assert memory.trials == ()


class _CollidingCandidateIds:
    def __init__(
        self,
        delegate: DeterministicIdFactory,
        candidate_id: CandidateId,
    ) -> None:
        self.delegate = delegate
        self.candidate_id = candidate_id

    def new_candidate_id(self):
        return self.candidate_id

    # Runtime-checkable protocols inspect attributes on the class rather than
    # accepting dynamic ``__getattr__`` forwarding.
    def new_run_id(self):
        return self.delegate.new_run_id()

    def new_event_id(self):
        return self.delegate.new_event_id()

    def new_generation_id(self):
        return self.delegate.new_generation_id()

    def new_insight_id(self):
        return self.delegate.new_insight_id()

    def new_operator_invocation_id(self):
        return self.delegate.new_operator_invocation_id()

    def new_llm_call_id(self):
        return self.delegate.new_llm_call_id()

    def new_provider_attempt_id(self):
        return self.delegate.new_provider_attempt_id()

    def new_evaluation_id(self):
        return self.delegate.new_evaluation_id()

    def new_evaluation_attempt_id(self):
        return self.delegate.new_evaluation_attempt_id()

    def new_correlation_id(self):
        return self.delegate.new_correlation_id()


def test_wave_rejects_colliding_materialized_members_before_evaluation() -> None:
    async def scenario():
        ids, problem, _generator, memory, engine, selector, wave = await _build_wave(
            "portfolio_wave_collision"
        )
        service = PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=_CollidingCandidateIds(ids, CandidateId("candidate_collision")),
            memory=memory,
        )
        with pytest.raises(ValueError, match="colliding members"):
            await service.run(wave)
        return problem, memory

    problem, memory = asyncio.run(scenario())
    assert problem.evaluations == 0
    assert memory.trials == ()


def test_wave_request_rejects_memory_cards_that_differ_from_selected_refs() -> None:
    async def scenario():
        (
            ids,
            _problem,
            _generator,
            _memory,
            _engine,
            _selector,
            wave,
        ) = await _build_wave("portfolio_wave_card_join")
        credit = wave.memory_credit
        assert credit is not None
        wrong = _rebind_credit_plan(
            credit,
            decision=replace(
                credit.decision,
                selected=(credit.decision.eligible[0],),
                exploitation_subset=(credit.decision.eligible[0],),
                subset_size=1,
                selected_subset_probability=Fraction(2, 3),
            ),
        )
        return wave, wrong

    wave, wrong = asyncio.run(scenario())
    with pytest.raises(ValueError, match="card references"):
        replace(wave, memory_credit=wrong)

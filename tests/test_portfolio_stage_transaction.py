"""Failure-injection proof for the portfolio-stage prepare/commit barrier."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from agent_evolve.agentic import (
    AgenticBenchmark,
    DeterministicIdFactory,
    InsightDraft,
    InsightMemoryBank,
    PortfolioCard,
    TypedConfigurationPhenotypeIdentityPolicy,
    compose_portfolio_evolution,
)
from agent_evolve.application.budgeted_optimizer import OptimizerBudget
from agent_evolve.application.campaign_execution import EvolutionCampaignScheduler
from agent_evolve.application.evolution_campaign import (
    AlternatingPortfolioRecombinationCadence,
    CampaignConcurrency,
    CampaignPolicies,
    CampaignProtocol,
    CampaignSeed,
    EvolutionCampaign,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    AgenticPortfolioCampaignRuntime,
    ArchiveReservoirCampaignParentSelector,
    CampaignPortfolioLearningPreparation,
    CampaignPortfolioOutcomePreparation,
    CampaignPortfolioWavePreparationReceipt,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioEvolution,
    PortfolioMemoryCreditPlan,
    PortfolioRewardAggregationBinding,
)
from agent_evolve.campaign_workload import (
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.domain.typed_json import thaw_json, typed_json_sha256
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    MemoryAssignmentArm,
    ResolvedInsightAssignment,
)
from agent_evolve.policies.selection.random_portfolio import (
    DeterministicRandomFeasiblePortfolioPolicy,
)
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.budgeted_v5_support import PARENT_C_SEQUENCE
from examples.benchmarks.boils_abc.evaluator import AbcEvaluatorSettings
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem
from tests.test_portfolio_campaign_runtime import (
    _ArchiveUtility,
    _DeterministicBoilsEvaluator,
    _Evidence,
    _ExecutionJournal,
    _NeverGenerator,
    _PreparationJournal,
    _PreparationRuntime,
    _Reflection,
    _WaveFactory,
    _binding,
    _object,
    _sha,
)
from tests.test_portfolio_evolution import _build_wave


class _CreditWaveFactory(_WaveFactory):
    """Attach an identified one-card assignment to every real runtime wave."""

    def __init__(self, *, composition, entries) -> None:
        super().__init__(composition=composition, card=None)
        self.entries = {entry.reference: entry for entry in entries}

    def build(self, context):
        # The base factory owns the generic finite-contract construction.  Its
        # placeholder card is replaced before either request is validated by the
        # portfolio service.
        placeholder = next(iter(self.entries.values()))
        self.card = PortfolioCard(
            card_key="card.transaction_placeholder",
            reference=placeholder.reference,
            content_sha256=placeholder.draft.content_sha256,
            evidence_sha256=_sha("transaction-placeholder-evidence"),
            prompt_payload=_object({"prior": "placeholder"}),
            assigned_score=placeholder.initial_score,
        )
        wave = super().build(context)
        eligible = tuple(sorted(self.entries))
        decision = self.composition.portfolio.memory.select(
            context_hash=wave.selection_request.context_sha256,
            subset_size=1,
            rng=random.Random(10_000 + context.stage_request.step.generation),
            exploration_probability=Fraction(1, 2),
            eligible_references=eligible,
        )
        selected_entry = self.entries[decision.selected[0]]
        card = PortfolioCard(
            card_key="card.transaction_selected",
            reference=selected_entry.reference,
            content_sha256=selected_entry.draft.content_sha256,
            evidence_sha256=_sha("transaction-selected-evidence"),
            prompt_payload=_object({"prior": "transaction_selected"}),
            assigned_score=dict(decision.score_snapshot)[selected_entry.reference],
        )
        selection_request = replace(wave.selection_request, cards=(card,))
        snapshot = CausalSearchScorePolicy(
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        ).genesis(
            exact_context_hash=decision.context_hash,
            estimand_stratum_hash=_sha("transaction-estimand"),
            priors=dict(decision.score_snapshot),
        )
        credit_unit_id = self.composition.id_factory.new_operator_invocation_id()
        assignment = ResolvedInsightAssignment.resolve(
            credit_unit_id=credit_unit_id,
            snapshot=snapshot,
            expected_snapshot_sha256=snapshot.snapshot_sha256,
            block_id=(
                f"transaction_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            arm=MemoryAssignmentArm.DIAGNOSTIC,
            selection_decision=decision,
            prompt_shape_sha256=selection_request.card_snapshot_sha256,
        )
        return replace(
            wave,
            selection_request=selection_request,
            memory_credit=PortfolioMemoryCreditPlan(
                decision=decision,
                credit_unit_id=credit_unit_id,
                aggregation=PortfolioRewardAggregationBinding(
                    aggregate=lambda outcomes: float(
                        max(outcome.reward for outcome in outcomes)
                    ),
                    aggregation_id="transaction_max_reward",
                    aggregation_version=1,
                    definition_sha256=_sha("transaction-max-reward"),
                ),
                card_snapshot_sha256=selection_request.card_snapshot_sha256,
                score_snapshot=snapshot,
                assignment=assignment,
            ),
        )


class _TransactionalOutcome:
    def __init__(self, *, fail_prepare: bool) -> None:
        self.fail_prepare = fail_prepare
        self.prepared: list[int] = []
        self.committed: list[int] = []
        self.aborted: list[int] = []

    async def prepare_update(self, request, waves, results, prior_memory):
        if self.fail_prepare:
            raise RuntimeError("injected outcome prepare failure")
        record = thaw_json(prior_memory)
        record["last_portfolio_generation"] = request.step.generation
        preparation = CampaignPortfolioOutcomePreparation(
            request_sha256=request.request_sha256,
            generation=request.step.generation,
            wave_request_sha256s=tuple(
                wave.selection_request.request_sha256 for wave in waves
            ),
            result_receipt_sha256s=tuple(
                result.receipt.receipt_sha256 for result in results
            ),
            prior_memory_sha256=typed_json_sha256(prior_memory),
            updated_memory=_object(record),
            evidence=_object({"prepared": "outcome"}),
        )
        self.prepared.append(preparation.generation)
        return preparation

    def commit_update(self, preparation):
        self.committed.append(preparation.generation)

    def abort_update(self, preparation):
        self.aborted.append(preparation.generation)


class _TransactionalLearning:
    def __init__(self, *, fail_prepare: bool) -> None:
        self.fail_prepare = fail_prepare
        self.prepared: list[int] = []
        self.committed: list[int] = []
        self.aborted: list[int] = []

    def reflection_completed(self, request, receipt, result):
        del request, receipt, result
        return _object({"unused": True})

    def reflections_admitted(self, request, contents):
        del request, contents
        return _object({"unused": True})

    async def prepare_portfolio_generation_close(
        self,
        request,
        waves,
        results,
        memory_credit_preparation,
    ):
        if self.fail_prepare:
            raise RuntimeError("injected learning prepare failure")
        preparation = CampaignPortfolioLearningPreparation(
            request_sha256=request.request_sha256,
            generation=request.step.generation,
            wave_request_sha256s=tuple(
                wave.selection_request.request_sha256 for wave in waves
            ),
            result_receipt_sha256s=tuple(
                result.receipt.receipt_sha256 for result in results
            ),
            memory_credit_preparation_sha256=(
                memory_credit_preparation.preparation_sha256
            ),
            evidence=_object({"prepared": "learning"}),
        )
        self.prepared.append(preparation.generation)
        return preparation

    def commit_portfolio_generation_close(self, preparation):
        self.committed.append(preparation.generation)

    def abort_portfolio_generation_close(self, preparation):
        self.aborted.append(preparation.generation)


class _WavePreparationObserver:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.receipts: list[CampaignPortfolioWavePreparationReceipt] = []

    def record_prepared_wave(self, receipt):
        assert type(receipt) is CampaignPortfolioWavePreparationReceipt
        receipt.__post_init__()
        self.receipts.append(receipt)
        if self.fail:
            raise RuntimeError("injected wave preparation publication failure")


def _build_failure_campaign(*, failure: str):
    evaluator = _DeterministicBoilsEvaluator()
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(
            AbcEvaluatorSettings.current_circuit_panel(circuit_names=("log2",)),
            evaluator=evaluator,
        ),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    ids = DeterministicIdFactory(f"stage_transaction_{failure}")
    memory = InsightMemoryBank(id_factory=ids)
    entries = memory.extend(
        (
            InsightDraft(
                claim=f"Transaction prior {index}.",
                trigger="A finite portfolio is available.",
                mechanism="Independent priors create identified assignment overlap.",
                affected_paths=("$.sequence",),
                evidence_summary="Provider-free transaction test prior.",
                confidence=0.5,
            )
            for index in range(2)
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )
    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=DeterministicRandomFeasiblePortfolioPolicy(seed=20260717),
        seed=20260717,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=2,
        max_output_tokens=1,
        temperature=None,
    )
    evidence = _Evidence()
    config = AgenticCampaignWorkloadConfig(
        workload_id="boils-stage-transaction-test",
        workload_version=1,
        definition_sha256=_sha("boils-stage-transaction-test"),
        benchmark=benchmark,
        seeds=(
            CampaignSeed(
                "seed_default",
                _object({"sequence": list(DEFAULT_ACTION_SEQUENCE)}),
            ),
            CampaignSeed(
                "seed_parent_c",
                _object({"sequence": list(PARENT_C_SEQUENCE)}),
            ),
        ),
        finite_catalog_id=FINITE_CATALOG_ID,
        evaluator_concurrency_cap=2,
        evaluator_preflight_receipt=_object({"qualified": True}),
        resource_lease_receipt=_object({"lease": "test_cpu_pool"}),
        evidence=AgenticCampaignEvidenceProjections(
            projection_id="stage_transaction_evidence",
            projection_version=1,
            definition_sha256=_sha("stage-transaction-evidence"),
            initialize_memory=evidence.initialize_memory,
            context=evidence.context,
            cards=evidence.cards,
        ),
    )
    parent_selector = ArchiveReservoirCampaignParentSelector(reservoir_limit=8)
    wave_factory = _CreditWaveFactory(composition=composition, entries=entries)
    outcome = _TransactionalOutcome(fail_prepare=failure == "outcome")
    learning = _TransactionalLearning(fail_prepare=failure == "learning")
    wave_preparations = _WavePreparationObserver()
    reflection = _Reflection()
    policies = CampaignPolicies(
        cadence=AlternatingPortfolioRecombinationCadence(),
        parent_selection=_binding("archive_reservoir", parent_selector),
        memory_assignment=_binding("runtime_evidence", evidence),
        portfolio_selection=_binding(
            "provider_free_random",
            composition.portfolio.selector,
        ),
        recombination=_binding("disjoint_patch_union", object()),
        reflection=_binding("provider_free_quarantine", reflection),
        archive_utility=_ArchiveUtility(),
    )
    protocol = CampaignProtocol(
        protocol_id=f"stage_transaction_{failure}",
        protocol_version=1,
        definition_sha256=_sha(f"stage-transaction-{failure}"),
        outer_seed=20260717,
        generation_count=3,
        required_seed_count=2,
        parents_per_portfolio_generation=1,
        portfolio_width=2,
        recombinations_per_parent=1,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=1,
    )
    workload_ports = config.build_ports()
    prepared = EvolutionCampaign(
        protocol=protocol,
        workload=workload_ports,
        policies=policies,
        runtime=_PreparationRuntime(),
        budget=OptimizerBudget(
            max_unique_evaluations=7,
            max_logical_llm_calls=3,
            max_generations=3,
        ),
        concurrency=CampaignConcurrency(
            evaluator_concurrency=2,
            agent_concurrency=2,
            agent_queue_capacity=4,
        ),
        journals=(_PreparationJournal(),),
    ).prepare()
    runtime = AgenticPortfolioCampaignRuntime(
        prepared=prepared,
        workload_config=config,
        workload_ports=workload_ports,
        composition=composition,
        parent_selector=parent_selector,
        wave_factory=wave_factory,
        task_sha256=_sha("provider-free-stage-transaction-task"),
        wave_preparation_observer=wave_preparations,
        learning_lifecycle=learning,
        reflection_executor=reflection,
        outcome_updater=outcome,
    )
    return (
        runtime,
        policies,
        memory,
        evaluator,
        outcome,
        learning,
        wave_preparations,
    )


@pytest.mark.parametrize("failure", ("outcome", "learning"))
def test_prepare_failure_leaves_every_campaign_publication_store_unchanged(
    failure: str,
) -> None:
    runtime, policies, memory, evaluator, outcome, learning, wave_preparations = (
        _build_failure_campaign(failure=failure)
    )
    journal = _ExecutionJournal()

    with pytest.raises(RuntimeError, match=f"injected {failure} prepare failure"):
        asyncio.run(
            EvolutionCampaignScheduler(
                prepared=runtime.prepared,
                policies=policies,
                stages=runtime,
                reflections=runtime,
                lifecycle=runtime,
                journal=journal,
            ).run()
        )

    # Seed registration is the prior committed state. Evaluated generation-one
    # candidates may remain in the engine cache, but none becomes campaign state.
    assert len(runtime.history) == 2
    assert all(candidate.generation == 0 for candidate in runtime.history)
    assert runtime.archive.snapshot().consideration_count == 2
    assert memory.trials == ()
    assert runtime._portfolio_waves == {}
    assert runtime._stage_receipts == {}
    assert runtime._selector_calls == 0
    assert "last_portfolio_generation" not in thaw_json(runtime._memory)
    assert outcome.committed == []
    assert learning.committed == []
    if failure == "learning":
        assert outcome.prepared == [1]
        assert outcome.aborted == [1]
    else:
        assert outcome.prepared == []
        assert outcome.aborted == []
    assert evaluator.calls == 4

    # The failed stage is intentionally unsealed, but its exact prospective
    # treatment was durably observed before selector dispatch.  The record is
    # hash/reference-only: neither card prompt payload nor instruction text is
    # duplicated into this audit channel.
    assert len(wave_preparations.receipts) == 1
    preparation = wave_preparations.receipts[0]
    assert runtime.wave_preparation_receipts == (preparation,)
    assert preparation.generation == 1
    assert preparation.parent_slot == 0
    assert preparation.parent_lane_id == "reservoir_0001"
    assert preparation.decision_slot_id == "reservoir_0001.primary"
    assert len(preparation.selector_card_snapshot_sha256) == 64
    assert len(preparation.evidence_card_snapshot_sha256) == 64
    assert preparation.memory_credit_identity is not None
    memory_identity = thaw_json(preparation.memory_credit_identity)
    assert memory_identity["credit_unit_id"]
    assert memory_identity["selection_decision_sha256"]
    assert memory_identity["assignment_sha256"]
    assert memory_identity["score_snapshot_sha256"]
    assert (
        memory_identity["context_projection_binding_sha256"]
        == (thaw_json(preparation.context_projection_identity)["binding_sha256"])
    )
    assert memory_identity["quarantine_admission"] is None
    assert preparation.test_eligible_reflection_receipts == ()
    record_text = json.dumps(preparation.to_record(), sort_keys=True)
    assert '"prior": "transaction_selected"' not in record_text
    assert "Select a diverse path-disjoint portfolio" not in record_text
    assert "prompt_payload" not in record_text


def test_wave_preparation_observer_failure_prevents_selector_dispatch() -> None:
    runtime, policies, _, evaluator, _, _, _ = _build_failure_campaign(
        failure="outcome"
    )
    observer = _WavePreparationObserver(fail=True)
    runtime.wave_preparation_observer = observer

    with pytest.raises(
        RuntimeError,
        match="injected wave preparation publication failure",
    ):
        asyncio.run(
            EvolutionCampaignScheduler(
                prepared=runtime.prepared,
                policies=policies,
                stages=runtime,
                reflections=runtime,
                lifecycle=runtime,
                journal=_ExecutionJournal(),
            ).run()
        )

    assert evaluator.calls == 2  # committed seeds only; no wave was dispatched.
    assert len(observer.receipts) == 1
    assert runtime.wave_preparation_receipts == tuple(observer.receipts)
    assert runtime._stage_receipts == {}


def test_memory_batch_preview_is_non_mutating_and_rejects_a_stale_bank() -> None:
    async def scenario():
        ids, _, _, memory, engine, selector, wave = await _build_wave(
            "stage_transaction_stale_memory"
        )
        portfolio = PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        )
        pending = await portfolio.run(wave, defer_memory_credit=True)
        preparation = portfolio.prepare_pending_memory_credit_batch((pending,))
        assert memory.trials == ()
        assert preparation.expected_trials

        foreign_trial = replace(
            preparation.expected_trials[0],
            credit_unit_id=ids.new_operator_invocation_id(),
            candidate_ids=(ids.new_candidate_id(),),
        )
        memory.record_trials_batch((foreign_trial,))
        with pytest.raises(RuntimeError, match="changed after credit preparation"):
            portfolio.commit_prepared_memory_credit_batch(preparation)
        return memory, foreign_trial

    memory, foreign_trial = asyncio.run(scenario())
    assert memory.trials == (foreign_trial,)

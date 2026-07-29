#!/usr/bin/env python3
"""Run a compact, real portfolio-evolution loop on pinned BOiLS/log2.

This is a workflow-development run, not a benchmark claim.  It exercises the
generic public AgentEvolve composition with authenticated elite/explorer parent
lanes, one model-proposed K8 slate followed by engine-allocated K4 per lane,
prior-only forecast calibration, card-blind empirical outcome history, concurrent exact
evaluation, disjoint-patch recombination, prospectively balanced causal-memory
assignments, frozen-archive joint credit, and asynchronously quarantined
reflection.  ``--prepare`` performs no provider call and ``--live`` is the only
mode that reads the OpenRouter credential.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials  # noqa: E402

from agent_evolve.agentic import (  # noqa: E402
    AgenticBenchmark,
    DeterministicIdFactory,
    InsightDraft,
    InsightMemoryBank,
    MetricComparisonAnchorKind,
    OptimizerState,
    ParetoArchive,
    PortfolioCard,
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryCreditPlan,
    PortfolioRecombination,
    PortfolioRecombinationWaveRequest,
    PortfolioRewardAggregationBinding,
    PortfolioSelectionRequest,
    PortfolioVariationWaveRequest,
    ReflectionInsightContract,
    ReflectionConsumerScope,
    ReflectionInsightKind,
    TaskKeyedArchiveEliteExplorerParentPolicy,
    compose_portfolio_evolution,
    eligible_finite_variation_view,
    exact_configuration_phenotype_bindings,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.application.calibrated_campaign import (  # noqa: E402
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
)
from agent_evolve.application.agentic_evolution import (  # noqa: E402
    EvolutionCandidate,
    InvocationOutcome,
    ReflectionCallExecutionError,
)
from agent_evolve.application.budgeted_optimizer import (  # noqa: E402
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.evolution_campaign import (  # noqa: E402
    ParentVariationBinding,
)
from agent_evolve.application.portfolio_outcome_feedback import (  # noqa: E402
    PortfolioOutcomeFeedbackLedger,
    observe_selected_portfolio_forecasts,
    validate_feedback_ledger,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    OpenRouterReasoningConfig,
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_campaign import (  # noqa: E402
    CalibratedPortfolioCampaignCoordinator,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (  # noqa: E402
    CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    PydanticAICalibratedPortfolioSelectionPolicy,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    OutcomePublicationPolicy,
    SchemaRepairAttemptPolicy,
    create_production_queued_runner,
    structured_generation_outcome_record,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter  # noqa: E402
from agent_evolve.policies.memory import (  # noqa: E402
    BalancedSubsetBlockAssignment,
    BalancedSubsetBlockPlan,
    BalancedSubsetBlockPlanner,
    CausalSearchScorePolicy,
    MemoryAssignmentArm,
    MemoryScoreSnapshot,
    ResolvedInsightAssignment,
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.selection.forecast_calibration import (  # noqa: E402
    ForecastCalibrationScope,
)
from agent_evolve.policies.selection.meaningful_direction import (  # noqa: E402
    AbsoluteToleranceDirectionAdjudicator,
    MetricDirectionResolution,
)
from agent_evolve.policies.reward import (  # noqa: E402
    FrozenArchiveJointWaveHypervolumeReward,
    FrozenArchiveWaveSnapshot2D,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
)
from examples.benchmarks.boils_abc.actions import (  # noqa: E402
    DEFAULT_ACTION_SEQUENCE,
)
from examples.benchmarks.boils_abc.budgeted_v5_support import (  # noqa: E402
    PARENT_C_SEQUENCE,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (  # noqa: E402
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem  # noqa: E402
from examples.benchmarks.boils_abc.variation_catalog import (  # noqa: E402
    ACTION_FAMILIES,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    DurableJsonlJournal,
    finalize_run_directory,
    source_identity,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/portfolio_q"
)
METRIC_IDS = ("total_levels", "total_lut_count")
PORTFOLIO_SIZE = 4
PARENTS_PER_CYCLE = 2
MEMORY_SUBSET_SIZE = 2
MEMORY_EXPLORATION = Fraction(1, 2)
QUEUE_MAX_IN_FLIGHT = 8
QUEUE_MAX_PENDING = 32
QUEUE_MAX_ATTEMPTS = 3
# Long-reasoning reflections can legitimately exceed five minutes.  Supervise
# first-event and idle liveness instead of truncating a healthy progressing
# stream at a fixed wall-clock deadline.  An interrupted stream is deliberately
# not retried because its provider-side billing and completion state can be
# unknown.
QUEUE_ATTEMPT_TIMEOUT_NS: int | None = None
STREAM_FIRST_EVENT_TIMEOUT_NS = 300_000_000_000
STREAM_IDLE_TIMEOUT_NS = 300_000_000_000
STREAM_CANCEL_DRAIN_TIMEOUT_NS = 5_000_000_000
STREAM_TRANSPORT_RETIRE_TIMEOUT_NS = 5_000_000_000
PROVIDER_CONNECT_TIMEOUT_SECONDS = 90.0
QUEUE_BASE_BACKOFF_NS = 1_000_000_000
QUEUE_MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 20_260_716
TASK_SHA256 = hashlib.sha256(b"agent-evolve:boils-log2-portfolio-q-task:v1").hexdigest()
REPAIRED_PROTOCOL_ID = "boils_portfolio_q_calibrated_v3"
MEMORY_STRATUM_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-log2-portfolio-q-repaired-memory-stratum:v1"
).hexdigest()
ARCHIVE_REFERENCE_POINT = {
    "total_lut_count": 12_000.0,
    "total_levels": 80.0,
}
REFLECTION_DECISION_PATHS = tuple(f"$.sequence[{index}]" for index in range(20))
REFLECTION_CONSUMER_SCOPES = (
    ReflectionConsumerScope.MUTATION_SELECTION,
    ReflectionConsumerScope.RECOMBINATION_SELECTION,
)
REFLECTION_INSIGHT_KINDS = (
    ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
    ReflectionInsightKind.MECHANISTIC_CONJECTURE,
)
REFLECTION_COMPARISON_ANCHORS = (MetricComparisonAnchorKind.CURRENT_PARENT,)


def _reflection_contract() -> ReflectionInsightContract:
    families = tuple(sorted(set(ACTION_FAMILIES.values())))
    return ReflectionInsightContract(
        required_metric_ids=METRIC_IDS,
        allowed_option_families=families,
        allowed_decision_paths=REFLECTION_DECISION_PATHS,
        allowed_insight_kinds=REFLECTION_INSIGHT_KINDS,
        allowed_consumer_scopes=REFLECTION_CONSUMER_SCOPES,
        allowed_comparison_anchor_kinds=REFLECTION_COMPARISON_ANCHORS,
        allowed_factor_capabilities=families,
    )


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    model_name: str
    provider_only: tuple[str, ...]
    reasoning_effort: str | None
    max_output_tokens: int
    temperature: float | None


MODEL_PROFILES = {
    "deepseek": ModelProfile(
        name="deepseek_v4_pro_streamlake_xhigh",
        model_name="deepseek/deepseek-v4-pro",
        provider_only=("streamlake",),
        reasoning_effort="xhigh",
        max_output_tokens=384_000,
        temperature=0.2,
    ),
    "mistral": ModelProfile(
        name="mistral_large_3",
        model_name="mistralai/mistral-large-2512",
        provider_only=(),
        reasoning_effort=None,
        max_output_tokens=131_072,
        temperature=0.2,
    ),
    "gpt": ModelProfile(
        name="gpt_5_6_sol_azure_xhigh",
        model_name="openai/gpt-5.6-sol",
        provider_only=("azure",),
        reasoning_effort="xhigh",
        max_output_tokens=128_000,
        temperature=None,
    ),
}


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - static preparation only.
        raise AssertionError(f"provider-free preparation invoked propose: {request}")

    async def reflect(self, request):  # pragma: no cover - static preparation only.
        raise AssertionError(f"provider-free preparation invoked reflect: {request}")


class _NeverSelector:
    async def select(self, request):  # pragma: no cover - static preparation only.
        raise AssertionError(f"provider-free preparation invoked select: {request}")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_slug(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _run_id(*, mode: str, profile: ModelProfile) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"boilsq_{mode}_{_model_slug(profile.name)}_{timestamp}"


def _id_namespace(run_id: str) -> str:
    # Stable-ID namespaces reject content-bearing terms and are capped at 48 chars.
    return f"boilsq_{_sha_text(run_id)[:16]}"


def _available_affinity_sets(limit: int = 16) -> tuple[tuple[int, ...], ...]:
    available = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    return tuple((cpu,) for cpu in available[: max(1, min(limit, len(available)))])


def _benchmark() -> tuple[AgenticBenchmark, tuple[tuple[int, ...], ...]]:
    affinities = _available_affinity_sets()
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=affinities,
        per_circuit_timeout_s=60.0,
    )
    problem = BoilsAbcProblem(settings)
    benchmark = AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    return benchmark, affinities


def _seed_memory(
    memory: InsightMemoryBank,
) -> tuple[Any, ...]:
    drafts = (
        InsightDraft(
            claim=(
                "Early strong cleanup or functional reduction can improve mapped "
                "depth, sometimes at an area trade-off."
            ),
            trigger="The parent remains depth-limited after its current early stages.",
            mechanism=(
                "Earlier structural simplification can expose shorter logic paths "
                "to later mapping stages."
            ),
            affected_paths=("$.sequence",),
            evidence_summary=(
                "A prior pinned log2 development contrast motivates testing this "
                "as a transferable hypothesis, not treating it as established."
            ),
            confidence=0.55,
        ),
        InsightDraft(
            claim=(
                "A late technology-aware or resubstitution transform can improve "
                "mapped area while keeping depth approximately stable."
            ),
            trigger="The parent has redundant late-stage structure before mapping.",
            mechanism=(
                "Late local restructuring may remove LUT demand without undoing "
                "the global depth structure established earlier."
            ),
            affected_paths=("$.sequence",),
            evidence_summary=(
                "Prior logic-synthesis observations motivate a falsifiable transfer "
                "test across parents and loci."
            ),
            confidence=0.5,
        ),
        InsightDraft(
            claim=(
                "A middle-stage balance or refactor immediately after a rewriting "
                "step can reduce mapped depth without requiring an early rewrite."
            ),
            trigger=(
                "The parent contains a rewriting step in the first half followed "
                "by a long interval without explicit depth balancing."
            ),
            mechanism=(
                "Rebalancing the rewritten intermediate network can shorten exposed "
                "critical paths before later area-oriented transforms remap them."
            ),
            affected_paths=("$.sequence",),
            evidence_summary=(
                "This is an executable timing hypothesis; the engine, rather than "
                "the card, separately enforces portfolio and recombination diversity."
            ),
            confidence=0.5,
        ),
    )
    return memory.extend(
        drafts,
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )


def _frozen_object(value: Mapping[str, object]):
    frozen = freeze_json(dict(value))
    if type(frozen).__name__ != "FrozenJsonObject":
        raise TypeError("expected a frozen typed-JSON object")
    return frozen


def _memory_context() -> Any:
    # Deliberately coarse: six portfolio trials share one stratum, while exact
    # parent outcomes remain request-bound in the instruction and trace.
    return _frozen_object(
        {
            "benchmark_family": "logic_synthesis_sequence_cooptimization",
            "candidate_representation": "ordered_categorical_sequence",
            "evaluation_panel": "single_pinned_training_circuit",
            "memory_stratum": "boils_log2_portfolio_q_v1",
            "objective_directions": {
                "total_levels": "minimize",
                "total_lut_count": "minimize",
            },
            "portfolio_size": PORTFOLIO_SIZE,
        }
    )


def _selector_context(
    *,
    memory_estimand_context: Any,
    parent: EvolutionCandidate,
    prior_action_outcome_history: Any,
) -> Any:
    """Expose parent facts and card-blind prior outcomes without changing the ITT stratum."""

    return _frozen_object(
        {
            "schema_version": 2,
            "memory_estimand_context": thaw_json(memory_estimand_context),
            "parent": {
                "candidate_id": parent.candidate_id.value,
                "configuration_sha256": parent.occurrence.configuration_hash,
                "configuration": parent.configuration_dict,
                "measured_metrics": parent.objective_map,
            },
            "prior_action_outcome_history": thaw_json(prior_action_outcome_history),
            "history_visibility": (
                "card_blind_evaluated_actions_only_strictly_before_current_wave"
            ),
        }
    )


def _model_profile_sha256(profile: ModelProfile) -> str:
    return _sha_text(
        json.dumps(
            {
                "name": profile.name,
                "model_name": profile.model_name,
                "provider_only": list(profile.provider_only),
                "reasoning_effort": profile.reasoning_effort,
                "max_output_tokens": profile.max_output_tokens,
                "temperature_hex": (
                    None if profile.temperature is None else profile.temperature.hex()
                ),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _calibration_scope(
    *, profile: ModelProfile, run_id: str
) -> ForecastCalibrationScope:
    return ForecastCalibrationScope(
        model_profile_sha256=_model_profile_sha256(profile),
        prompt_definition_sha256=CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256,
        selector_policy_definition_sha256=(
            CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
        benchmark_sha256=TASK_SHA256,
        session_sha256=_sha_text(
            f"agent-evolve:boils-q-calibration-session:v1:{run_id}"
        ),
    )


def _deterministic_rank(*, label: str, stop: int) -> int:
    if type(stop) is not int or stop <= 0:
        raise ValueError("rank stop must be a positive exact integer")
    digest = hashlib.sha256(
        f"{REPAIRED_PROTOCOL_ID}:{JITTER_SEED}:{label}".encode("ascii", errors="strict")
    ).digest()
    return int.from_bytes(digest, "big", signed=False) % stop


def _balanced_memory_plan(
    memory: InsightMemoryBank,
    *,
    cycles: int,
) -> BalancedSubsetBlockPlan:
    """Freeze all portfolio-call assignments before any provider call.

    Units are lane-major.  At the six-generation/Q setting this gives one
    complete C(3,2) block to the elite lane and one complete block to the
    explorer lane, preventing parent-lane quality from masquerading as insight
    utility.  Shorter developmental schedules retain overlap but are not used
    for lane-conditional causal claims.
    """

    if type(cycles) is not int or cycles < 2:
        raise ValueError("the repaired balanced-memory protocol requires >=2 cycles")
    context_sha256 = typed_json_sha256(_memory_context())
    entries = memory.entries
    snapshot = CausalSearchScorePolicy().genesis(
        exact_context_hash=context_sha256,
        estimand_stratum_hash=MEMORY_STRATUM_SHA256,
        priors={entry.reference: float(entry.initial_score) for entry in entries},
    )
    generation_numbers = tuple(2 * cycle - 1 for cycle in range(1, cycles + 1))
    units = tuple(
        StableMemoryAssignmentUnit(
            unit_key=f"{lane}.g{generation:02d}",
            generation=generation,
            lane_id=lane,
        )
        for lane in ("elite", "explorer")
        for generation in generation_numbers
    )
    catalog_size = math.comb(len(snapshot.entries), MEMORY_SUBSET_SIZE)
    full_block_count, remainder_size = divmod(len(units), catalog_size)
    full_ranks = tuple(
        _deterministic_rank(
            label=f"balanced-full-block-{index}",
            stop=math.factorial(catalog_size),
        )
        for index in range(full_block_count)
    )
    remainder_selection_rank = (
        None
        if remainder_size == 0
        else _deterministic_rank(
            label="balanced-remainder-subset",
            stop=math.comb(catalog_size, remainder_size),
        )
    )
    remainder_permutation_rank = (
        None
        if remainder_size == 0
        else _deterministic_rank(
            label="balanced-remainder-order",
            stop=math.factorial(remainder_size),
        )
    )
    return BalancedSubsetBlockPlanner().plan(
        snapshot=snapshot,
        ordered_units=units,
        subset_size=MEMORY_SUBSET_SIZE,
        full_block_permutation_ranks=full_ranks,
        remainder_selection_rank=remainder_selection_rank,
        remainder_permutation_rank=remainder_permutation_rank,
    )


def _selection_decision_record(decision: Any) -> dict[str, object]:
    return {
        "context_hash": decision.context_hash,
        "eligible": [
            {"insight_id": ref.insight_id.value, "version": ref.version}
            for ref in decision.eligible
        ],
        "selected": [
            {"insight_id": ref.insight_id.value, "version": ref.version}
            for ref in decision.selected
        ],
        "score_snapshot": [
            {
                "insight_id": ref.insight_id.value,
                "version": ref.version,
                "score_hex": score.hex(),
            }
            for ref, score in decision.score_snapshot
        ],
        "subset_size": decision.subset_size,
        "exploration_probability": str(decision.exploration_probability),
        "mode": decision.mode.value,
        "selected_subset_probability": str(decision.selected_subset_probability),
        "credit_identifiable": decision.credit_identifiable,
        "policy_id": decision.policy_id,
        "policy_version": decision.policy_version,
    }


def _cards(memory: InsightMemoryBank, decision: Any) -> tuple[PortfolioCard, ...]:
    entries = memory.entries_for(decision.selected)
    scores = dict(decision.score_snapshot)
    cards: list[PortfolioCard] = []
    for index, entry in enumerate(entries, start=1):
        draft = entry.draft
        cards.append(
            PortfolioCard(
                card_key=f"card.{index:02d}",
                reference=entry.reference,
                content_sha256=draft.content_sha256,
                evidence_sha256=_sha_text(
                    f"boils-q-seed-prior-v1:{draft.content_sha256}"
                ),
                prompt_payload=_frozen_object(
                    {
                        "claim": draft.claim,
                        "trigger": draft.trigger,
                        "mechanism": draft.mechanism,
                        "evidence_summary": draft.evidence_summary,
                        "status": "hypothesis_to_test",
                    }
                ),
                assigned_score=float(scores[entry.reference]),
            )
        )
    return tuple(cards)


def _portfolio_instruction(parent: EvolutionCandidate) -> str:
    objectives = json.dumps(
        parent.objective_map,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "Select exactly four different sealed one-step variations for concurrent "
        "evaluation. Optimize all required metrics under Pareto trade-offs. "
        "Because the children may be recombined, use four distinct concrete "
        "configuration paths whenever the contract permits; do not spend two "
        "portfolio slots on the same locus. Use at least three distinct option "
        "families and prefer complementary mechanisms over superficial variants. "
        "Every member must cite at least one supplied card and predict every "
        f"required metric direction. The measured parent objectives are {objectives}."
    )


def _make_wave(
    *,
    composition: Any,
    parent: EvolutionCandidate,
    generation: int,
    cycle: int,
    parent_ordinal: int,
    known_configuration_sha256s: tuple[str, ...],
    memory_estimand_context: Any,
    prior_action_outcome_history: Any,
    memory_assignment: BalancedSubsetBlockAssignment,
    memory_snapshot: MemoryScoreSnapshot,
    archive_reward: FrozenArchiveJointWaveHypervolumeReward,
    archive_snapshot_sha256: str,
    binding_factory: CalibratedCampaignBindingFactory,
    coordinator: CalibratedPortfolioCampaignCoordinator,
    lane_id: str,
    profile: ModelProfile,
) -> tuple[PortfolioVariationWaveRequest, dict[str, object]]:
    base_contract = composition.bind_finite_variation(
        FINITE_CATALOG_ID,
        parent.configuration,
    )
    eligibility = eligible_finite_variation_view(
        contract=base_contract,
        option_phenotypes=exact_configuration_phenotype_bindings(base_contract),
        known_phenotype_sha256s=known_configuration_sha256s,
    )
    context_sha256 = typed_json_sha256(memory_estimand_context)
    memory_assignment.__post_init__()
    if (
        memory_assignment.unit.generation != generation
        or memory_assignment.unit.lane_id != lane_id
    ):
        raise ValueError("balanced memory assignment is bound to a foreign lane")
    decision = memory_assignment.decision
    if decision.context_hash != context_sha256:
        raise ValueError("balanced memory assignment is bound to a foreign context")
    cards = _cards(composition.memory, decision)
    selector_context = _selector_context(
        memory_estimand_context=memory_estimand_context,
        parent=parent,
        prior_action_outcome_history=prior_action_outcome_history,
    )
    request = PortfolioSelectionRequest(
        call_id=composition.id_factory.new_llm_call_id(),
        operation="select_portfolio",
        instruction=_portfolio_instruction(parent),
        context=selector_context,
        finite_variation_contract=eligibility.contract,
        cards=cards,
        portfolio_size=PORTFOLIO_SIZE,
        required_metric_ids=METRIC_IDS,
        min_distinct_families=3,
        require_supporting_cards=True,
        require_pairwise_disjoint_parent_patches=True,
        max_output_tokens=profile.max_output_tokens,
        temperature=profile.temperature,
    )
    aggregation = PortfolioRewardAggregationBinding(
        aggregate=lambda outcomes: float(
            archive_reward(tuple(outcome.candidate for outcome in outcomes))
        ),
        aggregation_id="frozen_archive_joint_wave_hv",
        aggregation_version=1,
        definition_sha256=archive_reward.definition_hash,
    )
    variation = ParentVariationBinding(
        benchmark_sha256=TASK_SHA256,
        parent_configuration_sha256=eligibility.contract.parent_configuration_sha256,
        known_phenotype_sha256s=known_configuration_sha256s,
        contract=eligibility.contract,
        eligibility_receipt=eligibility.receipt,
    )
    binding = binding_factory.build(
        request=request,
        variation=variation,
        wave_index=generation,
        frozen_archive_snapshot_sha256=archive_snapshot_sha256,
    )
    coordinator.register(request, binding)
    credit_unit_id = composition.id_factory.new_operator_invocation_id()
    resolved_assignment = ResolvedInsightAssignment.resolve(
        credit_unit_id=credit_unit_id,
        snapshot=memory_snapshot,
        expected_snapshot_sha256=memory_snapshot.snapshot_sha256,
        block_id=memory_assignment.unit.unit_key,
        arm=MemoryAssignmentArm.DIAGNOSTIC,
        selection_decision=decision,
        prompt_shape_sha256=request.card_snapshot_sha256,
    )
    wave = PortfolioVariationWaveRequest(
        selection_request=request,
        parent=parent,
        generation=generation,
        label_prefix=f"boilsq.c{cycle:02d}.{lane_id}",
        phase="portfolio_evolution",
        memory_credit=PortfolioMemoryCreditPlan(
            decision=decision,
            credit_unit_id=credit_unit_id,
            aggregation=aggregation,
            card_snapshot_sha256=request.card_snapshot_sha256,
            score_snapshot=memory_snapshot,
            assignment=resolved_assignment,
            context_projection=(
                PortfolioMemoryContextProjectionBinding.from_selector_context(
                    selector_context
                )
            ),
        ),
    )
    trace = {
        "event_type": "portfolio_wave_prepared",
        "cycle": cycle,
        "parent_ordinal": parent_ordinal,
        "lane_id": lane_id,
        "parent_candidate_id": parent.candidate_id.value,
        "parent_generation": parent.generation,
        "parent_objectives": parent.objective_map,
        "request": request.to_record(),
        "calibrated_input_binding": binding.to_record(),
        "calibrated_prompt_sha256": _sha_text(coordinator.render(request)),
        "base_contract_identity_sha256": base_contract.identity_sha256,
        "eligibility": eligibility.receipt.to_record(),
        "memory_assignment": memory_assignment.to_record(),
        "resolved_memory_assignment": resolved_assignment.to_record(),
        "resolved_memory_assignment_sha256": (resolved_assignment.assignment_sha256),
        "memory_score_snapshot_sha256": memory_snapshot.snapshot_sha256,
        "memory_treatment_binding_sha256": (
            wave.memory_credit.treatment_binding_sha256
            if wave.memory_credit is not None
            else None
        ),
        "memory_decision": _selection_decision_record(decision),
        "memory_selection_decision_sha256": (
            memory_assignment.selection_decision_sha256
        ),
        "archive_reward_snapshot": archive_reward.snapshot.to_record(),
        "coarse_memory_pooling": False,
    }
    return wave, trace


def _candidate_record(candidate: EvolutionCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "configuration": candidate.configuration_dict,
        "objectives": candidate.objective_map,
        "valid": candidate.valid,
        "generation": candidate.generation,
        "label": candidate.label,
        "operator_kind": (
            None if candidate.operator_kind is None else candidate.operator_kind.value
        ),
        "parent_ids": [value.value for value in candidate.parent_ids],
        "common_ancestor_id": (
            None
            if candidate.common_ancestor_id is None
            else candidate.common_ancestor_id.value
        ),
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "parent_patch_hashes": list(candidate.parent_patch_hashes),
        "preservation_verified": candidate.preservation_verified,
    }


def _memory_record(memory: InsightMemoryBank, context_sha256: str) -> dict[str, object]:
    return {
        "entries": [
            {
                "reference": {
                    "insight_id": entry.reference.insight_id.value,
                    "version": entry.reference.version,
                },
                "content_sha256": entry.draft.content_sha256,
                "claim": entry.draft.claim,
                "origin": entry.origin.value,
                "lifecycle_state": entry.lifecycle_state.value,
                "retrievable": entry.retrievable,
                "initial_score_hex": entry.initial_score.hex(),
                "evidence_lineage_sha256": (
                    None
                    if entry.evidence_lineage is None
                    else entry.evidence_lineage.identity_sha256
                ),
            }
            for entry in memory.entries
        ],
        "trials": [
            {
                "credit_unit_id": trial.credit_unit_id.value,
                "candidate_ids": [value.value for value in trial.candidate_ids],
                "reward_definition_hash": trial.reward_definition_hash,
                "reward_hex": trial.reward.hex(),
                "decision": _selection_decision_record(trial.decision),
            }
            for trial in memory.trials
        ],
        "score_evidence": list(memory.score_evidence(context_sha256)),
    }


def _manifest(
    *,
    run_id: str,
    mode: str,
    profile: ModelProfile,
    cycles: int,
    affinities: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    reasoning = (
        None
        if profile.reasoning_effort is None
        else {"effort": profile.reasoning_effort}
    )
    source_paths = (
        Path(__file__),
        AGENT_EVOLVE_ROOT / "src/agent_evolve/application/portfolio_evolution.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/application/calibrated_campaign.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/portfolio_outcome_feedback.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/application/portfolio_recombination.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/finite_variation_eligibility.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/selection/elite_explorer.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/finite_palette_evidence.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/calibrated_portfolio_selection.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/memory/balanced_subset_blocks.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/reward/frozen_wave_archive.py",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/problem_def.py",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/finite_variation_catalog.py",
    )
    return {
        "schema_version": 3,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "mode": mode,
        "claim_boundary": {
            "workflow_development_only": True,
            "paper_ready_result": False,
            "sota_claim": False,
            "matched_baseline_in_this_run": False,
        },
        "workload": {
            "id": "boils_abc_log2_portfolio_q",
            "circuit_panel": ["log2"],
            "candidate_schema": "ordered_length_20_categorical_sequence",
            "objective_ids": list(METRIC_IDS),
            "evaluation_affinity_sets": [list(value) for value in affinities],
        },
        "schedule": {
            "protocol_id": REPAIRED_PROTOCOL_ID,
            "cycles": cycles,
            "parents_per_cycle": PARENTS_PER_CYCLE,
            "portfolio_size": PORTFOLIO_SIZE,
            "model_proposal_size": 8,
            "engine_evaluation_size": PORTFOLIO_SIZE,
            "mutation_candidates_planned": cycles * PARENTS_PER_CYCLE * PORTFOLIO_SIZE,
            "recombination_candidates_max": cycles * PARENTS_PER_CYCLE * 2,
            "selector_calls_planned": cycles * PARENTS_PER_CYCLE,
            "reflection_calls_planned": cycles,
            "generation_numbers": list(range(1, 2 * cycles + 1)),
        },
        "model": {
            "profile": profile.name,
            "requested_model": profile.model_name,
            "provider_options": {
                "only": list(profile.provider_only),
                "allow_fallbacks": not bool(profile.provider_only),
            },
            "reasoning": reasoning,
            "reasoning_mode": None,
            "max_output_tokens": profile.max_output_tokens,
            "temperature": profile.temperature,
        },
        "queue": {
            "max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "max_pending": QUEUE_MAX_PENDING,
            "max_attempts": QUEUE_MAX_ATTEMPTS,
            "attempt_timeout_ns": QUEUE_ATTEMPT_TIMEOUT_NS,
            "stream_liveness": {
                "first_event_timeout_ns": STREAM_FIRST_EVENT_TIMEOUT_NS,
                "idle_timeout_ns": STREAM_IDLE_TIMEOUT_NS,
                "absolute_timeout_ns": None,
                "cancel_drain_timeout_ns": STREAM_CANCEL_DRAIN_TIMEOUT_NS,
                "transport_retire_timeout_ns": (STREAM_TRANSPORT_RETIRE_TIMEOUT_NS),
            },
            "backoff": {
                "kind": "exponential_deterministic_task_keyed_full_jitter",
                "base_ns": QUEUE_BASE_BACKOFF_NS,
                "max_ns": QUEUE_MAX_BACKOFF_NS,
                "seed": JITTER_SEED,
            },
        },
        "memory": {
            "seed_card_count": 3,
            "subset_size": MEMORY_SUBSET_SIZE,
            "assignment": "preprovider_balanced_complete_k_subset_blocks",
            "assignment_order": "lane_major_elite_then_explorer",
            "exploration_probability": str(MEMORY_EXPLORATION),
            "credit_unit": "one_aggregate_trial_per_four_member_portfolio",
            "reward_aggregation": (
                "normalized_joint_hypervolume_gain_against_same_frozen_"
                "pregeneration_archive"
            ),
            "archive_reference_point": ARCHIVE_REFERENCE_POINT,
            "coarse_context_pooling": True,
            "context_pooling_design": "lane_balanced_pooled_memory_stratum",
            "reflection_entries_start_quarantined": True,
            "reflection_execution": "async_after_recombination_stage_seal",
            "reflection_visibility": "block_barrier_only_without_auto_promotion",
            "forecast_calibration": "prior_wave_only_beta_smoothed_cells",
            "outcome_history": "card_blind_evaluated_actions_only",
            "adaptation_boundary": (
                "identification_block_only_no_memory_score_consumption"
            ),
        },
        "genericity": {
            "composition": "agent_evolve.agentic.compose_portfolio_evolution",
            "workload_injection": "AgenticBenchmark plus finite variation catalog",
            "model_authority": (
                "ranked_k8_opaque_finite_option_proposal_with_predictions_only"
            ),
            "engine_authority": (
                "prior_calibrated_structural_k4_allocation_and_exact_materialization"
            ),
            "candidate_materialization": "engine_exact",
            "recombination": "generic_disjoint_parent_patch_union",
            "parent_selection": "authenticated_elite_explorer_lanes",
            "selector_trace": "full_hash_verified_decision_audit_required",
        },
        "source_identity": source_identity(
            source_paths,
            relative_to=WORKSPACE_ROOT,
        ),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
        },
    }


def _composition(
    *,
    benchmark: AgenticBenchmark,
    ids: DeterministicIdFactory,
    memory: InsightMemoryBank,
    generator: Any,
    selector: Any,
    profile: ModelProfile,
    evaluator_concurrency: int,
    engine_sink: Any,
) -> Any:
    return compose_portfolio_evolution(
        benchmark,
        generator=generator,
        selector=selector,
        seed=20_260_716,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=evaluator_concurrency,
        engine_trace_sink=engine_sink,
        max_output_tokens=profile.max_output_tokens,
        temperature=profile.temperature,
    )


async def _register_seeds(composition: Any) -> tuple[EvolutionCandidate, ...]:
    return tuple(
        await asyncio.gather(
            composition.engine.register_seed(
                {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
                label="seed_default_q0",
            ),
            composition.engine.register_seed(
                {"sequence": list(PARENT_C_SEQUENCE)},
                label="seed_parent_c",
            ),
        )
    )


async def _prepare(
    *,
    run_dir: Path,
    run_id: str,
    profile: ModelProfile,
    cycles: int,
) -> dict[str, object]:
    benchmark, affinities = _benchmark()
    ids = DeterministicIdFactory(_id_namespace(run_id))
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=MEMORY_EXPLORATION,
    )
    _seed_memory(memory)
    engine_journal = DurableJsonlJournal(run_dir / "engine_events.jsonl")
    try:
        composition = _composition(
            benchmark=benchmark,
            ids=ids,
            memory=memory,
            generator=_NeverGenerator(),
            selector=_NeverSelector(),
            profile=profile,
            evaluator_concurrency=len(affinities),
            engine_sink=engine_journal.append,
        )
        started = time.perf_counter()
        seeds = await _register_seeds(composition)
        seed_wall_s = time.perf_counter() - started
        if not all(seed.valid for seed in seeds):
            raise RuntimeError("provider-free seed gate produced an invalid seed")
        context = _memory_context()
        archive = ParetoArchive(
            benchmark.objectives,
            outcome_relation_binding=composition.outcome_relation,
        )
        for seed in seeds:
            archive.consider(seed)
        cache = await composition.engine.evaluation_cache_snapshot()
        archive_snapshot = archive.snapshot()
        state = OptimizerState(
            generation=0,
            candidates=seeds,
            archive=archive_snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(archive_snapshot),
            unique_evaluations=int(cache["misses"] or 0),
            logical_llm_calls=0,
        )
        parents = TaskKeyedArchiveEliteExplorerParentPolicy().select(
            state,
            task_sha256=TASK_SHA256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            rotation_index=0,
        )
        known = tuple(sorted(seed.occurrence.configuration_hash for seed in seeds))
        memory_plan = _balanced_memory_plan(memory, cycles=cycles)
        feedback_ledger = PortfolioOutcomeFeedbackLedger()
        binding_factory = CalibratedCampaignBindingFactory(
            scope=_calibration_scope(profile=profile, run_id=run_id),
            objectives=equal_weight_slate_objectives(benchmark.objectives),
            ledger=feedback_ledger,
        )
        coordinator = CalibratedPortfolioCampaignCoordinator()
        prior_history = feedback_ledger.prompt_history(cutoff_wave_index_exclusive=1)
        reward_snapshot = FrozenArchiveWaveSnapshot2D.create(
            objectives=benchmark.objectives,
            reference_point=ARCHIVE_REFERENCE_POINT,
            archive_points=tuple(
                candidate.objective_map
                for candidate in archive_snapshot.front_candidates
            ),
        )
        archive_reward = FrozenArchiveJointWaveHypervolumeReward(reward_snapshot)
        previews: list[dict[str, object]] = []
        for ordinal, (lane, parent) in enumerate(
            zip(("elite", "explorer"), parents.parents, strict=True),
            start=1,
        ):
            wave, trace = _make_wave(
                composition=composition,
                parent=parent,
                generation=1,
                cycle=1,
                parent_ordinal=ordinal,
                known_configuration_sha256s=known,
                memory_estimand_context=context,
                prior_action_outcome_history=prior_history,
                memory_assignment=memory_plan.assignment_for(1, lane),
                memory_snapshot=memory_plan.snapshot,
                archive_reward=archive_reward,
                archive_snapshot_sha256=state.archive_snapshot_hash,
                binding_factory=binding_factory,
                coordinator=coordinator,
                lane_id=lane,
                profile=profile,
            )
            prompt = coordinator.render(wave.selection_request)
            previews.append(
                {
                    **trace,
                    "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                    "prompt_approx_tokens": len(prompt.encode("utf-8")) / 4.0,
                    "prompt_sha256": _sha_text(prompt),
                    "eligible_option_count": len(
                        wave.selection_request.finite_variation_contract.options
                    ),
                }
            )
        result = {
            "schema_version": 1,
            "status": "prepared_provider_free",
            "provider_calls": 0,
            "credential_read": False,
            "seed_batch_wall_s": seed_wall_s,
            "seeds": [_candidate_record(value) for value in seeds],
            "cache": cache,
            "parent_selection": parents.receipt.to_trace_record(),
            "balanced_memory_plan": memory_plan.to_record(),
            "portfolio_previews": previews,
            "calibrated_registered_request_count": (
                coordinator.registered_request_count
            ),
            "planned_cycles": cycles,
        }
        write_json_atomic(run_dir / "summary.json", result)
        return result
    finally:
        engine_journal.close()


def _queue_snapshot_record(snapshot: Any) -> dict[str, object]:
    fields = (
        "max_in_flight",
        "max_pending",
        "pending",
        "in_flight",
        "closed",
    )
    return {
        name: (
            value.value
            if hasattr((value := getattr(snapshot, name, None)), "value")
            else value
        )
        for name in fields
    }


async def _run_reflection_task(
    *,
    composition: Any,
    cycle: int,
    cycle_outcomes: tuple[InvocationOutcome, ...],
    source_receipts: tuple[str, ...],
    reflection_contract: ReflectionInsightContract,
    launched_perf_ns: int,
    promotion_barrier_generation: int | None,
) -> dict[str, object]:
    """Run one quarantined reflection without making it causally visible."""

    try:
        reflected = await composition.engine.reflect_with_receipt(
            cycle_outcomes,
            label=f"boilsq_cycle_{cycle:02d}_reflection",
            max_insights=2,
            min_insights=1,
            insight_contract=reflection_contract,
            source_receipt_sha256s=source_receipts,
        )
        completed_perf_ns = time.perf_counter_ns()
        return {
            "event_type": "reflection_completed_quarantined",
            "cycle": cycle,
            "source_generation": 2 * cycle,
            "launched_perf_ns": launched_perf_ns,
            "completed_perf_ns": completed_perf_ns,
            "wall_s": (completed_perf_ns - launched_perf_ns) / 1_000_000_000,
            "entry_count": len(reflected.entries),
            "visibility": "quarantined_until_block_close",
            "promotion_barrier_generation": promotion_barrier_generation,
            "receipt": {
                **reflected.receipt.to_record(),
                "receipt_sha256": reflected.receipt.receipt_sha256,
            },
        }
    except ReflectionCallExecutionError as exc:
        completed_perf_ns = time.perf_counter_ns()
        return {
            "event_type": "reflection_failed",
            "cycle": cycle,
            "source_generation": 2 * cycle,
            "launched_perf_ns": launched_perf_ns,
            "completed_perf_ns": completed_perf_ns,
            "wall_s": (completed_perf_ns - launched_perf_ns) / 1_000_000_000,
            "failure_type": exc.failure_type,
            "visibility": "no_evidence_published",
            "promotion_barrier_generation": promotion_barrier_generation,
            "receipt": {
                **exc.receipt.to_record(),
                "receipt_sha256": exc.receipt.receipt_sha256,
            },
        }


async def _execute_live(
    *,
    run_dir: Path,
    run_id: str,
    profile: ModelProfile,
    cycles: int,
    engine_journal: DurableJsonlJournal,
    queue_journal: DurableJsonlJournal,
    planner_journal: DurableJsonlJournal,
    wave_journal: DurableJsonlJournal,
) -> dict[str, object]:
    load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
    load_credentials(AGENT_EVOLVE_ROOT / ".env", override=False, optional=True)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable in live mode")

    benchmark, affinities = _benchmark()
    ids = DeterministicIdFactory(_id_namespace(run_id))
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=MEMORY_EXPLORATION,
    )
    seed_entries = _seed_memory(memory)
    memory_plan = _balanced_memory_plan(memory, cycles=cycles)
    context = _memory_context()
    context_sha256 = typed_json_sha256(context)
    feedback_ledger = PortfolioOutcomeFeedbackLedger()
    calibration_scope = _calibration_scope(profile=profile, run_id=run_id)
    binding_factory = CalibratedCampaignBindingFactory(
        scope=calibration_scope,
        objectives=equal_weight_slate_objectives(benchmark.objectives),
        ledger=feedback_ledger,
    )
    coordinator = CalibratedPortfolioCampaignCoordinator()
    direction_adjudicator = AbsoluteToleranceDirectionAdjudicator(
        benchmark_sha256=calibration_scope.benchmark_sha256,
        session_sha256=calibration_scope.session_sha256,
        resolutions=tuple(
            MetricDirectionResolution(metric_id=value, absolute_tolerance=0.0)
            for value in METRIC_IDS
        ),
    )

    provider_options: dict[str, object] = {
        "allow_fallbacks": not bool(profile.provider_only)
    }
    if profile.provider_only:
        provider_options["only"] = list(profile.provider_only)
        provider_options["allow_fallbacks"] = False
    reasoning = (
        None
        if profile.reasoning_effort is None
        else OpenRouterReasoningConfig(effort=profile.reasoning_effort)
    )
    structured_options: dict[str, object] = {
        "api_key": api_key,
        "model_name": profile.model_name,
        "max_connections": QUEUE_MAX_IN_FLIGHT,
        "timeout_seconds": PROVIDER_CONNECT_TIMEOUT_SECONDS,
        "provider_options": provider_options,
        "app_title": "AgentEvolve AAAI 2027 repaired generic portfolio campaign",
        "stream_liveness_policy": StructuredStreamLivenessPolicy(
            first_event_timeout_ns=STREAM_FIRST_EVENT_TIMEOUT_NS,
            idle_timeout_ns=STREAM_IDLE_TIMEOUT_NS,
            absolute_timeout_ns=None,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=STREAM_CANCEL_DRAIN_TIMEOUT_NS,
                transport_retire_timeout_ns=STREAM_TRANSPORT_RETIRE_TIMEOUT_NS,
            ),
        ),
    }
    if reasoning is not None:
        structured_options["reasoning_config"] = reasoning
    structured = PydanticAIStructuredGenerator.openrouter(**structured_options)
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=QUEUE_MAX_IN_FLIGHT,
        max_pending=QUEUE_MAX_PENDING,
        max_attempts=QUEUE_MAX_ATTEMPTS,
        attempt_timeout_ns=QUEUE_ATTEMPT_TIMEOUT_NS,
        base_backoff_ns=QUEUE_BASE_BACKOFF_NS,
        max_backoff_ns=QUEUE_MAX_BACKOFF_NS,
        jitter_policy=DeterministicHashJitter(
            seed=JITTER_SEED,
            domain="portfolio-campaign-repaired-v2",
        ),
        close_generator=True,
        outcome_sink=lambda outcome: queue_journal.append(
            structured_generation_outcome_record(outcome)
        ),
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        attempt_request_policy=SchemaRepairAttemptPolicy(),
    )

    history: list[EvolutionCandidate] = []
    variation_results: list[Any] = []
    recombination_results: list[Any] = []
    reflection_records: list[dict[str, object]] = []
    reflection_tasks: dict[int, asyncio.Task[dict[str, object]]] = {}
    selection_audit_records: list[dict[str, object]] = []
    archive_reward_records: list[dict[str, object]] = []
    parent_selection_records: list[dict[str, object]] = []
    archive_decision_cursor = 0
    logical_llm_calls = 0
    run_started = time.perf_counter()

    async with runner:
        composition = _composition(
            benchmark=benchmark,
            ids=ids,
            memory=memory,
            generator=PydanticAIAgenticGenerator(runner),
            selector=PydanticAICalibratedPortfolioSelectionPolicy(
                generate_once=runner,
                binding_for=coordinator.binding_for,
            ),
            profile=profile,
            evaluator_concurrency=len(affinities),
            engine_sink=engine_journal.append,
        )
        archive = ParetoArchive(
            benchmark.objectives,
            outcome_relation_binding=composition.outcome_relation,
        )
        recombiner = PortfolioRecombination(
            engine=composition.engine,
            ids=composition.id_factory,
        )
        parent_policy = TaskKeyedArchiveEliteExplorerParentPolicy()
        seed_started = time.perf_counter()
        seeds = await _register_seeds(composition)
        seed_batch_wall_s = time.perf_counter() - seed_started
        if not all(seed.valid for seed in seeds):
            raise RuntimeError("one or more live seeds were invalid")
        for seed in seeds:
            history.append(seed)
            archive.consider(seed)

        for cycle in range(1, cycles + 1):
            mutation_generation = 2 * cycle - 1
            recombination_generation = 2 * cycle
            cache_before = await composition.engine.evaluation_cache_snapshot()
            snapshot = archive.snapshot()
            state = OptimizerState(
                generation=2 * (cycle - 1),
                candidates=tuple(history),
                archive=snapshot,
                archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
                unique_evaluations=int(cache_before["misses"] or 0),
                logical_llm_calls=logical_llm_calls,
            )
            parent_selection = parent_policy.select(
                state,
                task_sha256=TASK_SHA256,
                expected_archive_snapshot_hash=state.archive_snapshot_hash,
                rotation_index=cycle - 1,
            )
            parent_record = {
                "event_type": "parent_elite_explorer_selection",
                "cycle": cycle,
                "receipt": parent_selection.receipt.to_trace_record(),
            }
            parent_selection_records.append(parent_record)
            planner_journal.append(parent_record)

            known = tuple(
                sorted(
                    {candidate.occurrence.configuration_hash for candidate in history}
                )
            )
            archive_reward = FrozenArchiveJointWaveHypervolumeReward(
                FrozenArchiveWaveSnapshot2D.create(
                    objectives=benchmark.objectives,
                    reference_point=ARCHIVE_REFERENCE_POINT,
                    archive_points=tuple(
                        candidate.objective_map
                        for candidate in snapshot.front_candidates
                    ),
                )
            )
            prior_history = feedback_ledger.prompt_history(
                cutoff_wave_index_exclusive=mutation_generation
            )
            waves: list[PortfolioVariationWaveRequest] = []
            for ordinal, (lane, parent) in enumerate(
                zip(
                    ("elite", "explorer"),
                    parent_selection.parents,
                    strict=True,
                ),
                start=1,
            ):
                wave, trace = _make_wave(
                    composition=composition,
                    parent=parent,
                    generation=mutation_generation,
                    cycle=cycle,
                    parent_ordinal=ordinal,
                    known_configuration_sha256s=known,
                    memory_estimand_context=context,
                    prior_action_outcome_history=prior_history,
                    memory_assignment=memory_plan.assignment_for(
                        mutation_generation,
                        lane,
                    ),
                    memory_snapshot=memory_plan.snapshot,
                    archive_reward=archive_reward,
                    archive_snapshot_sha256=state.archive_snapshot_hash,
                    binding_factory=binding_factory,
                    coordinator=coordinator,
                    lane_id=lane,
                    profile=profile,
                )
                waves.append(wave)
                wave_journal.append(trace)

            logical_llm_calls += len(waves)
            mutation_started = time.perf_counter()
            cycle_variations = tuple(
                await asyncio.gather(
                    *(composition.portfolio.run(wave) for wave in waves)
                )
            )
            mutation_wall_s = time.perf_counter() - mutation_started
            pending_feedback = []
            for wave, result in zip(waves, cycle_variations, strict=True):
                variation_results.append(result)
                audit = result.selection_decision_audit_record
                if audit is None:
                    raise RuntimeError(
                        "fresh repaired portfolio result omitted selector audit"
                    )
                exact_prompt = coordinator.render(wave.selection_request)
                selected_predictions = coordinator.decode_selected_predictions(result)
                feedback_receipt = observe_selected_portfolio_forecasts(
                    wave_index=mutation_generation,
                    parent=wave.parent,
                    result=result,
                    selected_predictions=selected_predictions,
                    adjudicator=direction_adjudicator,
                )
                pending_feedback.append(feedback_receipt)
                audit_record = {
                    "event_type": "portfolio_selection_audit",
                    "cycle": cycle,
                    "generation": mutation_generation,
                    "wave_request_sha256": wave.selection_request.request_sha256,
                    "wave_receipt_sha256": result.receipt.receipt_sha256,
                    "request_text": exact_prompt,
                    "request_text_sha256": _sha_text(exact_prompt),
                    "decision": audit,
                    "selected_forecast_receipts": [
                        value.to_record() for value in selected_predictions
                    ],
                    "outcome_feedback": feedback_receipt.to_record(),
                }
                selection_audit_records.append(audit_record)
                planner_journal.append(audit_record)
                reward_record = archive_reward.record(result.candidates)
                memory_credit = result.receipt.memory_credit
                if (
                    memory_credit is None
                    or memory_credit.reward != reward_record.reward
                    or memory_credit.aggregation_definition_sha256
                    != reward_record.reward_definition_hash
                ):
                    raise RuntimeError(
                        "portfolio memory credit differs from frozen archive utility"
                    )
                archive_reward_record = {
                    "event_type": "portfolio_archive_reward",
                    "cycle": cycle,
                    "generation": mutation_generation,
                    "wave_request_sha256": wave.selection_request.request_sha256,
                    "wave_receipt_sha256": result.receipt.receipt_sha256,
                    "reward": reward_record.to_record(),
                }
                archive_reward_records.append(archive_reward_record)
                planner_journal.append(archive_reward_record)
                planner_journal.append(
                    {
                        "event_type": "portfolio_wave_completed",
                        "cycle": cycle,
                        "wave_request_sha256": (wave.selection_request.request_sha256),
                        "receipt": result.receipt.to_record(),
                    }
                )
                for candidate in result.candidates:
                    history.append(candidate)
                    archive.consider(candidate)

            validate_feedback_ledger((*feedback_ledger.receipts, *pending_feedback))
            feedback_ledger.receipts.extend(pending_feedback)
            planner_journal.append(
                {
                    "event_type": "calibrated_feedback_stage_published",
                    "cycle": cycle,
                    "generation": mutation_generation,
                    "receipt_sha256s": [
                        value.receipt_sha256 for value in pending_feedback
                    ],
                    "ledger_observation_count": len(feedback_ledger.observations),
                    "provider_calls": 0,
                }
            )

            recombination_requests = tuple(
                PortfolioRecombinationWaveRequest(
                    source_wave=wave,
                    source_result=result,
                    ancestor=wave.parent,
                    generation=recombination_generation,
                    label_prefix=f"boilsq.r{cycle:02d}.p{ordinal:02d}",
                    phase="portfolio_recombination",
                )
                for ordinal, (wave, result) in enumerate(
                    zip(waves, cycle_variations, strict=True),
                    start=1,
                )
            )
            recombination_started = time.perf_counter()
            cycle_recombinations = tuple(
                await asyncio.gather(
                    *(recombiner.run(request) for request in recombination_requests)
                )
            )
            recombination_wall_s = time.perf_counter() - recombination_started
            for request, result in zip(
                recombination_requests,
                cycle_recombinations,
                strict=True,
            ):
                recombination_results.append(result)
                planner_journal.append(
                    {
                        "event_type": "recombination_wave_completed",
                        "cycle": cycle,
                        "source_request_sha256": (
                            request.source_wave.selection_request.request_sha256
                        ),
                        "receipt": result.receipt.to_record(),
                    }
                )
                for candidate in result.candidates:
                    history.append(candidate)
                    archive.consider(candidate)

            cycle_outcomes = tuple(
                outcome
                for result in (*cycle_variations, *cycle_recombinations)
                for outcome in result.outcomes
            )
            source_receipts = tuple(
                result.receipt.receipt_sha256
                for result in (*cycle_variations, *cycle_recombinations)
            )
            reflection_contract = _reflection_contract()
            logical_llm_calls += 1
            block_end_cycle = ((cycle - 1) // 3 + 1) * 3
            promotion_barrier_generation = (
                2 * block_end_cycle if block_end_cycle <= cycles else None
            )
            launched_perf_ns = time.perf_counter_ns()
            reflection_tasks[cycle] = asyncio.create_task(
                _run_reflection_task(
                    composition=composition,
                    cycle=cycle,
                    cycle_outcomes=cycle_outcomes,
                    source_receipts=source_receipts,
                    reflection_contract=reflection_contract,
                    launched_perf_ns=launched_perf_ns,
                    promotion_barrier_generation=promotion_barrier_generation,
                ),
                name=f"boilsq-reflection-cycle-{cycle:02d}",
            )
            planner_journal.append(
                {
                    "event_type": "reflection_launched_async",
                    "cycle": cycle,
                    "source_generation": recombination_generation,
                    "launched_perf_ns": launched_perf_ns,
                    "visibility": "quarantined_until_block_close",
                    "promotion_barrier_generation": (promotion_barrier_generation),
                }
            )

            if cycle % 3 == 0:
                block_cycles = tuple(range(cycle - 2, cycle + 1))
                joined = tuple(
                    await asyncio.gather(
                        *(reflection_tasks.pop(value) for value in block_cycles)
                    )
                )
                for reflection_record in joined:
                    reflection_records.append(reflection_record)
                    planner_journal.append(reflection_record)
                planner_journal.append(
                    {
                        "event_type": "reflection_block_joined_at_barrier",
                        "generation": recombination_generation,
                        "source_cycles": list(block_cycles),
                        "completed_count": len(joined),
                        "eligible_for_next_block_curation": True,
                        "promotion_performed": False,
                        "reason": (
                            "quarantined insights require controlled downstream "
                            "tests before lifecycle promotion"
                        ),
                    }
                )

            new_archive_decisions = archive.decisions[archive_decision_cursor:]
            archive_decision_cursor = len(archive.decisions)
            planner_journal.append(
                {
                    "event_type": "cycle_closed",
                    "cycle": cycle,
                    "mutation_generation": mutation_generation,
                    "recombination_generation": recombination_generation,
                    "mutation_wall_s": mutation_wall_s,
                    "recombination_wall_s": recombination_wall_s,
                    "mutation_candidate_count": sum(
                        len(result.candidates) for result in cycle_variations
                    ),
                    "recombination_candidate_count": sum(
                        len(result.candidates) for result in cycle_recombinations
                    ),
                    "archive_decisions": [
                        value.to_trace_record() for value in new_archive_decisions
                    ],
                    "archive": archive.snapshot().to_trace_record(),
                    "cache": await composition.engine.evaluation_cache_snapshot(),
                    "memory": _memory_record(memory, context_sha256),
                }
            )

        if reflection_tasks:
            tail_cycles = tuple(sorted(reflection_tasks))
            drained = tuple(
                await asyncio.gather(
                    *(reflection_tasks.pop(value) for value in tail_cycles)
                )
            )
            for reflection_record in drained:
                reflection_records.append(reflection_record)
                planner_journal.append(reflection_record)
            planner_journal.append(
                {
                    "event_type": "reflection_tail_drained_without_promotion",
                    "source_cycles": list(tail_cycles),
                    "completed_count": len(drained),
                    "promotion_performed": False,
                    "reason": "campaign ended before a complete reflection block",
                }
            )

        queue_snapshot = await runner.snapshot()
        cache = await composition.engine.evaluation_cache_snapshot()
        final_archive = archive.snapshot()
        reflection_successes = sum(
            row["event_type"] == "reflection_completed_quarantined"
            for row in reflection_records
        )
        recombination_counts = [
            len(result.candidates) for result in recombination_results
        ]
        health_checks = {
            "both_seeds_valid": len(seeds) == 2 and all(seed.valid for seed in seeds),
            "six_portfolio_waves_at_three_cycles": (
                cycles != 3 or len(variation_results) == 6
            ),
            "every_portfolio_has_four_candidates": all(
                len(result.candidates) == PORTFOLIO_SIZE for result in variation_results
            ),
            "every_source_wave_has_recombination": all(
                count >= 1 for count in recombination_counts
            ),
            "every_cycle_reflected": reflection_successes == cycles,
            "cache_drained": cache["in_flight"] == 0,
            "memory_one_trial_per_portfolio": (
                len(memory.trials) == len(variation_results)
            ),
            "balanced_memory_plan_complete": (
                len(memory_plan.assignments) == len(variation_results)
                and all(item.has_overlap for item in memory_plan.support)
            ),
            "selection_audit_complete": (
                len(selection_audit_records) == len(variation_results)
            ),
            "calibrated_k8_to_k4_registry_complete": (
                coordinator.registered_request_count == len(variation_results)
            ),
            "calibrated_feedback_complete": (
                len(feedback_ledger.receipts) == len(variation_results)
                and len(feedback_ledger.observations)
                == len(variation_results) * PORTFOLIO_SIZE * len(METRIC_IDS)
            ),
            "per_action_attribution_complete": all(
                len(result.action_attributions) == PORTFOLIO_SIZE
                for result in variation_results
            ),
            "calibrated_selector_policy_exact": all(
                result.receipt.selection_policy_definition_sha256
                == CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
                for result in variation_results
            ),
            "archive_reward_complete": (
                len(archive_reward_records) == len(variation_results)
            ),
            "reasoning_config_exact": (
                profile.reasoning_effort is None or profile.reasoning_effort == "xhigh"
            ),
        }
        evolution_core_health_checks = {
            key: value
            for key, value in health_checks.items()
            if key != "every_cycle_reflected"
        }
        result = {
            "schema_version": 1,
            "status": "completed"
            if all(health_checks.values())
            else "completed_unhealthy",
            "health_pass": all(health_checks.values()),
            "health_checks": health_checks,
            "evolution_core_health_pass": all(evolution_core_health_checks.values()),
            "evolution_core_health_checks": evolution_core_health_checks,
            "reflection_health_pass": health_checks["every_cycle_reflected"],
            "claim_boundary": {
                "workflow_development_only": True,
                "paper_ready_result": False,
                "sota_claim": False,
                "matched_baseline_in_this_run": False,
                "next_gate": "matched_independent_random_or_domain_baseline_campaign",
            },
            "run_wall_s": time.perf_counter() - run_started,
            "seed_batch_wall_s": seed_batch_wall_s,
            "model_profile": profile.name,
            "requested_model": profile.model_name,
            "configured_reasoning": (
                None
                if profile.reasoning_effort is None
                else {"effort": profile.reasoning_effort}
            ),
            "configured_reasoning_mode": None,
            "logical_llm_calls": logical_llm_calls,
            "balanced_memory_plan": memory_plan.to_record(),
            "calibration_scope": calibration_scope.to_record(),
            "direction_adjudicator": direction_adjudicator.to_record(),
            "portfolio_outcome_feedback": [
                value.to_record() for value in feedback_ledger.receipts
            ],
            "card_blind_prompt_history": thaw_json(
                feedback_ledger.prompt_history(
                    cutoff_wave_index_exclusive=2 * cycles + 1
                )
            ),
            "queue": _queue_snapshot_record(queue_snapshot),
            "cache": cache,
            "candidate_count": len(history),
            "candidates": [_candidate_record(value) for value in history],
            "final_front": [
                _candidate_record(value) for value in final_archive.front_candidates
            ],
            "archive": final_archive.to_trace_record(),
            "parent_selections": parent_selection_records,
            "portfolio_wave_count": len(variation_results),
            "portfolio_receipts": [
                result.receipt.to_record() for result in variation_results
            ],
            "portfolio_selection_audits": selection_audit_records,
            "portfolio_archive_rewards": archive_reward_records,
            "recombination_wave_count": len(recombination_results),
            "recombination_candidate_counts": recombination_counts,
            "recombination_receipts": [
                result.receipt.to_record() for result in recombination_results
            ],
            "reflections": reflection_records,
            "memory_seed_references": [
                {
                    "insight_id": entry.reference.insight_id.value,
                    "version": entry.reference.version,
                }
                for entry in seed_entries
            ],
            "memory": _memory_record(memory, context_sha256),
        }
        return result


async def _run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    profile = MODEL_PROFILES[args.model]
    mode = "live" if args.live else "prepare"
    run_id = args.run_id or _run_id(mode=mode, profile=profile)
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    benchmark, affinities = _benchmark()
    del benchmark
    write_json_atomic(
        run_dir / "manifest.json",
        _manifest(
            run_id=run_id,
            mode=mode,
            profile=profile,
            cycles=args.cycles,
            affinities=affinities,
        ),
    )

    if not args.live:
        try:
            result = await _prepare(
                run_dir=run_dir,
                run_id=run_id,
                profile=profile,
                cycles=args.cycles,
            )
        except Exception as exc:
            write_json_atomic(
                run_dir / "failed.json",
                {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "safe_message": str(exc)[:2_000],
                    "failed_at_utc": _utc_now(),
                },
            )
            finalize_run_directory(run_dir, status="failed")
            raise
        finalize_run_directory(run_dir, status=str(result["status"]))
        return run_dir, result

    journals = (
        DurableJsonlJournal(run_dir / "engine_events.jsonl"),
        DurableJsonlJournal(run_dir / "queue_outcomes.jsonl"),
        DurableJsonlJournal(run_dir / "planner_events.jsonl"),
        DurableJsonlJournal(run_dir / "wave_requests.jsonl"),
    )
    failure: Exception | None = None
    try:
        result = await _execute_live(
            run_dir=run_dir,
            run_id=run_id,
            profile=profile,
            cycles=args.cycles,
            engine_journal=journals[0],
            queue_journal=journals[1],
            planner_journal=journals[2],
            wave_journal=journals[3],
        )
        write_json_atomic(run_dir / "summary.json", result)
    except Exception as exc:
        write_json_atomic(
            run_dir / "failed.json",
            {
                "status": "failed",
                "failure_type": type(exc).__name__,
                "safe_message": str(exc)[:2_000],
                "failed_at_utc": _utc_now(),
            },
        )
        failure = exc
    finally:
        for journal in journals:
            journal.close()
    if failure is not None:
        finalize_run_directory(run_dir, status="failed")
        raise failure
    finalize_run_directory(run_dir, status=str(result["status"]))
    return run_dir, result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--model", choices=tuple(MODEL_PROFILES), default="deepseek")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not 2 <= args.cycles <= 12:
        parser.error("--cycles must lie in [2,12] for balanced-memory overlap")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir, result = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "status": result["status"],
                "health_pass": result.get("health_pass"),
                "candidate_count": result.get("candidate_count"),
                "logical_llm_calls": result.get("logical_llm_calls", 0),
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

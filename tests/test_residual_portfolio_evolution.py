from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field, replace
from decimal import Decimal

import pytest

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.earned_lineage import EarnedLineageLedger
from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.materialized_action_broker import (
    BrokerReturnEstimate,
    MaterializedActionAllocationRequirement,
    MaterializedActionContext,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionEvaluation,
    MaterializedActionEvaluationBatch,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
    ResidualPortfolioEvolution,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
    PrequentialQuotaScorePortfolioPolicy,
)
from agent_evolve.application.residual_stage_credit import (
    ResidualStageCreditProjector,
)
from agent_evolve.application.residual_learning_transaction import (
    ResidualLearningState,
    TransactionalResidualLearningStore,
)
from agent_evolve.application.residual_campaign_runtime import (
    ResidualArchiveTransitionCommit,
    ResidualArchiveTransitionPreparation,
    ResidualPortfolioCampaignStageRuntime,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    thaw_json,
)
from agent_evolve.integrations.pydantic_ai.materialized_portfolio_judge import (
    MaterializedPortfolioJudgePromptProjection,
    PydanticAIMaterializedPortfolioJudge,
)
from agent_evolve.ports.structured_generator import StructuredGenerationResponse


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class _ZeroSlateValue:
    definition_sha256: str = _sha("zero-slate-value")

    def value(self, actions):
        return 0.0


@dataclass(frozen=True)
class _AlwaysFeasible:
    definition_sha256: str = _sha("always-feasible")

    def permits(self, actions):
        return True


@dataclass(frozen=True)
class _AdditiveJointSnapshot:
    gain_per_point: float = 0.1

    def joint_gain(self, points):
        return float(self.gain_per_point * len(points))


@dataclass(frozen=True)
class _AdditiveArchiveUtility:
    definition_sha256: str = _sha("additive-archive-utility")

    def require_snapshot(self, value):
        return _AdditiveJointSnapshot()


@dataclass(frozen=True)
class _ExpertReturn:
    means: dict[str, float]
    standard_deviation: float = 0.2
    definition_sha256: str = _sha("expert-return")

    def estimate(self, action):
        mean = self.means[action.expert_id]
        return BrokerReturnEstimate(
            mean=mean,
            standard_deviation=self.standard_deviation,
            local_count=0,
            global_count=0,
            resolved_count=0,
            provisional_count=0,
            local_mean=mean,
            global_mean=mean,
            shrinkage_weight=0.0,
        )


@dataclass(frozen=True)
class _HashOrderedReturn:
    """Prefer late action identities to exercise bounded-search completion."""

    definition_sha256: str = _sha("hash-ordered-return")

    def estimate(self, action):
        mean = int(action.action_sha256, 16) / float((1 << 256) - 1)
        return BrokerReturnEstimate(
            mean=mean,
            standard_deviation=0.0,
            local_count=0,
            global_count=0,
            resolved_count=0,
            provisional_count=0,
            local_mean=mean,
            global_mean=mean,
            shrinkage_weight=0.0,
        )


def _request() -> ResidualPortfolioDecisionRequest:
    return ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("campaign"),
        prior_state_sha256=_sha("prior-state"),
        decision_index=2,
        phase=SearchPhase.BASIN_EXPANSION,
        remaining_decisions=3,
        remaining_evaluations=12,
        evaluation_slots=2,
        expert_proposal_slots=(
            ("agentic", 2),
            ("numerical_acquisition", 2),
        ),
        proposal_context=freeze_json(
            {
                "archive_cutoff_sha256": _sha("archive"),
                "workload_owned_payload": {"opaque": True},
            }
        ),
        reference_escrow_slots=1,
    )


def _context(request: ResidualPortfolioDecisionRequest, name: str):
    return MaterializedActionContext(
        campaign_scope_sha256=request.campaign_scope_sha256,
        decision_index=request.decision_index,
        phase=request.phase,
        remaining_decisions=request.remaining_decisions,
        remaining_evaluations=request.remaining_evaluations,
        residual_frontier_cell=f"frontier.{name}",
        parent_position_cell="parent.nondominated",
        archive_relation_cell="near_front",
        structural_signature_sha256=_sha(f"structure:{name}"),
        patch_compatibility_cell="compatible",
        forecast_calibration_cell="unknown",
        source_distance_bin=0,
        memory_dose_bin=0,
    )


def _candidate(action: MaterializedActionDescriptor, sequence: int):
    canonical = canonical_typed_json_bytes(action.configuration)
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=action.target_candidate_id,
            configuration_hash=action.configuration_sha256,
            configuration_artifact_hash=hashlib.sha256(canonical).hexdigest(),
            proposal_sequence=sequence,
        ),
        configuration=action.configuration,
        objectives=(("cost", float(sequence)), ("quality", float(-sequence))),
        valid=True,
        generation=action.context.decision_index,
        label=f"evaluated-{action.expert_id}-{action.native_rank}",
        operator_compliant=True,
        evidence_compliant=True,
    )


@dataclass
class _Expert:
    expert_id: str
    action_names: tuple[str, ...]
    reference: bool
    phenotype_names: tuple[str, ...] | None = None
    expert_version: int = 1
    definition_sha256: str = field(init=False)
    evaluated_action_sha256s: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self):
        self.definition_sha256 = _sha(f"expert:{self.expert_id}")

    async def propose(self, request):
        phenotypes = self.phenotype_names or self.action_names
        actions = tuple(
            MaterializedActionDescriptor(
                context=_context(request, name),
                configuration=freeze_json(
                    {"expert": self.expert_id, "candidate": name}
                ),
                phenotype_identity_sha256=_sha(f"phenotype:{phenotype}"),
                expert_id=self.expert_id,
                native_rank=rank,
                parent_ids=(),
                operator_id=(
                    "numerical_acquisition" if self.reference else "agentic_mutation"
                ),
                target_candidate_id=CandidateId(
                    f"candidate_{self.expert_id}_{rank:03d}"
                ),
                role_id="backbone" if self.reference else "challenger",
                normalized_evaluation_cost=1.0,
                reference_action=self.reference,
            )
            for rank, (name, phenotype) in enumerate(
                zip(self.action_names, phenotypes, strict=True),
                start=1,
            )
        )
        return MaterializedActionProposalBatch(
            request_sha256=request.request_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            actions=actions,
            evidence=freeze_json(
                {
                    "expert_owned_materialization": True,
                    "future_outcomes_consulted": False,
                }
            ),
        )

    async def evaluate(self, proposal, selected_action_sha256s):
        self.evaluated_action_sha256s = selected_action_sha256s
        by_sha256 = {value.action_sha256: value for value in proposal.actions}
        evaluations = tuple(
            MaterializedActionEvaluation(
                action=by_sha256[action_sha256],
                candidate=_candidate(by_sha256[action_sha256], sequence),
                evaluator_receipt_sha256=_sha(
                    f"evaluation:{self.expert_id}:{action_sha256}"
                ),
            )
            for sequence, action_sha256 in enumerate(
                selected_action_sha256s,
                start=1,
            )
        )
        return MaterializedActionEvaluationBatch(
            proposal_sha256=proposal.proposal_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            selected_action_sha256s=selected_action_sha256s,
            evaluations=evaluations,
            evidence=freeze_json({"real_evaluator_calls": len(evaluations)}),
        )


@dataclass(frozen=True)
class _PinnedAllocation:
    policy_id: str = "test_outcome_blind_allocator"
    policy_version: int = 1
    definition_sha256: str = _sha("test-outcome-blind-allocator")

    async def require(self, request, proposals):
        pinned = next(
            action
            for proposal in proposals
            for action in proposal.actions
            if action.expert_id == "agentic" and action.native_rank == 2
        )
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            required_action_sha256s=(pinned.action_sha256,),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "complete_sealed_population_compared": True,
                    "candidate_outcomes_observed": False,
                }
            ),
        )


@dataclass(frozen=True)
class _CognitiveProjection:
    projection_id: str = "test_cognitive_projection"
    projection_version: int = 1
    definition_sha256: str = _sha("test-cognitive-projection")

    def project(self, request, proposals):
        action_sha256s = tuple(
            sorted(
                action.action_sha256
                for proposal in proposals
                for action in proposal.actions
            )
        )
        return MaterializedPortfolioJudgePromptProjection(
            projection_id=self.projection_id,
            projection_version=self.projection_version,
            projection_definition_sha256=self.definition_sha256,
            action_sha256s=action_sha256s,
            instruction="Select a complementary outcome-blind test slate.",
            payload=freeze_json(
                {
                    "objective_contract": {
                        "metric": "cost",
                        "direction": "minimize",
                    },
                    "ordinary_decimal_prior_values": [1.0, 2.0],
                }
            ),
        )


@dataclass(frozen=True)
class _ScoreLane:
    scorer_id: str
    preferred: tuple[tuple[str, int], ...]
    scorer_version: int = 1
    definition_sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "definition_sha256",
            _sha(f"score-lane:{self.scorer_id}"),
        )

    async def score(self, request, proposals):
        priority = {
            identity: float(len(self.preferred) - index)
            for index, identity in enumerate(self.preferred)
        }
        actions = tuple(
            sorted(
                (action for proposal in proposals for action in proposal.actions),
                key=lambda value: value.action_sha256,
            )
        )
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(value.proposal_sha256 for value in proposals)
            ),
            scores=tuple(
                MaterializedActionScore(
                    action_sha256=action.action_sha256,
                    value=priority.get(
                        (action.expert_id, action.native_rank),
                        0.0,
                    ),
                )
                for action in actions
            ),
            candidate_outcomes_observed=False,
            evidence_sha256=_sha(f"score-evidence:{self.scorer_id}"),
        )


@dataclass
class _Archive:
    archive_id: str = "test_archive"
    archive_version: int = 1
    definition_sha256: str = _sha("test-archive")
    committed: bool = False
    aborted: bool = False

    async def prepare(self, result):
        common = {
            "utility_id": "additive_archive_utility",
            "utility_version": 1,
            "definition_sha256": _sha("additive-archive-utility"),
            "generation": result.request.decision_index,
            "benchmark_sha256": _sha("benchmark"),
        }
        return ResidualArchiveTransitionPreparation(
            archive_id=self.archive_id,
            archive_version=self.archive_version,
            archive_definition_sha256=self.definition_sha256,
            residual_result_sha256=result.result_sha256,
            pre_snapshot=ArchiveUtilitySnapshot(
                **common,
                archive_sha256=_sha("pre-archive"),
                snapshot_receipt=freeze_json({"base_utility_hex": float(0.2).hex()}),
                scalar_utility_hex=float(0.2).hex(),
            ),
            post_snapshot=ArchiveUtilitySnapshot(
                **common,
                archive_sha256=_sha("post-archive"),
                snapshot_receipt=freeze_json({"base_utility_hex": float(0.4).hex()}),
                scalar_utility_hex=float(0.4).hex(),
            ),
            evidence=freeze_json({"preview_only": True}),
        )

    def commit(self, preparation):
        assert not self.committed
        self.committed = True
        return ResidualArchiveTransitionCommit(
            preparation_sha256=preparation.preparation_sha256,
            committed_archive_sha256=preparation.post_snapshot.archive_sha256,
            evidence=freeze_json({"published": True}),
        )

    def abort(self, preparation):
        self.aborted = True


def test_reference_and_challenger_compete_before_real_evaluation():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            return_value=_ExpertReturn(
                {
                    "agentic": 0.8,
                    "numerical_acquisition": 0.7,
                }
            ),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
    )

    result = asyncio.run(runtime.run(_request()))

    assert len(result.candidates) == 2
    assert {value.action.expert_id for value in result.evaluations} == {
        "agentic",
        "numerical_acquisition",
    }
    assert len(agentic.evaluated_action_sha256s) == 1
    assert len(numerical.evaluated_action_sha256s) == 1
    assert result.broker_decision.required_reference_action_sha256s
    assert not result.broker_decision.reference_displaced
    assert result.to_record()["archive_credit_included"] is False


def test_broker_never_evaluates_duplicate_phenotypes_across_experts():
    agentic = _Expert(
        "agentic",
        ("semantic_duplicate", "semantic_unique"),
        False,
        phenotype_names=("shared", "agentic_unique"),
    )
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_duplicate", "acquisition_unique"),
        True,
        phenotype_names=("shared", "numerical_unique"),
    )
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            return_value=_ExpertReturn(
                {
                    "agentic": 0.75,
                    "numerical_acquisition": 0.7,
                }
            ),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
    )

    result = asyncio.run(runtime.run(_request()))

    phenotypes = tuple(
        value.action.phenotype_identity_sha256 for value in result.evaluations
    )
    assert len(phenotypes) == len(set(phenotypes)) == 2
    assert (
        sum(
            len(value)
            for value in (
                agentic.evaluated_action_sha256s,
                numerical.evaluated_action_sha256s,
            )
        )
        == 2
    )


def test_async_outcome_blind_allocator_pins_from_complete_sealed_union():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            return_value=_ExpertReturn(
                {
                    "agentic": 0.8,
                    "numerical_acquisition": 0.7,
                }
            ),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
        allocation_policy=_PinnedAllocation(),
    )

    result = asyncio.run(runtime.run(_request()))

    pinned = next(
        action
        for proposal in result.proposals
        for action in proposal.actions
        if action.expert_id == "agentic" and action.native_rank == 2
    )
    requirement = result.broker_decision.allocation_requirement
    assert requirement is not None
    assert requirement.required_action_sha256s == (pinned.action_sha256,)
    assert requirement.candidate_outcomes_observed is False
    assert pinned.action_sha256 in {
        value.action_sha256 for value in result.broker_decision.selected_actions
    }
    assert agentic.evaluated_action_sha256s == (pinned.action_sha256,)
    assert (
        result.to_record()["broker_decision"]["allocation_requirement"][
            "residual_request_sha256"
        ]
        == result.request.request_sha256
    )


def test_pydantic_judge_adapter_selects_exact_slate_without_workload_branches():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    captured = []

    async def generate_once(request):
        captured.append(request)
        envelope = json.loads(
            next(line for line in request.prompt.splitlines() if line.startswith("{"))
        )
        selected = [
            value["action_sha256"]
            for value in envelope["sealed_actions"]
            if value["native_rank"] == 2
        ]
        value = request.output_type(
            selected_action_sha256s=selected,
            decision_rationale="The rank-two members cover distinct experts.",
            rejected_high_risk_pattern="Avoid duplicated expert-rank exposure.",
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="test/model",
            resolved_model="test/model",
            resolved_provider="test-provider",
            provider_response_id="response-1",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=20,
            reasoning_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.01"),
            latency_ns=1_000,
        )

    judge = PydanticAIMaterializedPortfolioJudge(
        generate_once=generate_once,
        prompt_projection=_CognitiveProjection(),
        call_id_factory=lambda request: LLMCallId(
            f"call_test_judge_{request.decision_index:03d}"
        ),
    )
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
        allocation_policy=judge,
    )

    result = asyncio.run(runtime.run(replace(_request(), reference_escrow_slots=0)))

    assert len(captured) == 1
    assert all(
        value.native_rank == 2 for value in result.broker_decision.selected_actions
    )
    requirement = result.broker_decision.allocation_requirement
    assert requirement is not None
    evidence = thaw_json(requirement.evidence)
    assert evidence["candidate_outcomes_observed"] is False
    assert evidence["telemetry"]["reasoning_tokens"] == 10
    assert "ordinary decimal quantities" in captured[0].prompt


def test_pydantic_judge_can_nominate_partial_slate_for_downstream_broker():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    captured = []

    async def generate_once(request):
        captured.append(request)
        envelope = json.loads(
            next(line for line in request.prompt.splitlines() if line.startswith("{"))
        )
        selected = next(
            value["action_sha256"]
            for value in envelope["sealed_actions"]
            if value["expert_id"] == "agentic" and value["native_rank"] == 2
        )
        return StructuredGenerationResponse(
            value=request.output_type(
                selected_action_sha256s=[selected],
                decision_rationale=(
                    "Nominate one semantic action and leave one slot open."
                ),
                rejected_high_risk_pattern=(
                    "Do not consume the downstream broker's capacity."
                ),
            ),
            requested_model="test/model",
            resolved_model="test/model",
            resolved_provider="test-provider",
            provider_response_id="response-partial",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=20,
            reasoning_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.01"),
            latency_ns=1_000,
        )

    judge = PydanticAIMaterializedPortfolioJudge(
        generate_once=generate_once,
        prompt_projection=_CognitiveProjection(),
        call_id_factory=lambda request: LLMCallId(
            f"call_test_partial_judge_{request.decision_index:03d}"
        ),
        nomination_slots=1,
    )
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
        allocation_policy=judge,
    )

    result = asyncio.run(runtime.run(replace(_request(), reference_escrow_slots=0)))

    requirement = result.broker_decision.allocation_requirement
    assert requirement is not None
    assert len(requirement.required_action_sha256s) == 1
    assert set(requirement.required_action_sha256s).issubset(
        {value.action_sha256 for value in result.broker_decision.selected_actions}
    )
    assert len(result.broker_decision.selected_actions) == 2
    assert any(
        value.action_sha256 not in requirement.required_action_sha256s
        for value in result.broker_decision.selected_actions
    )
    evidence = thaw_json(requirement.evidence)
    assert evidence["nomination_slots"] == 1
    assert evidence["evaluation_slots"] == 2
    assert evidence["downstream_unreserved_slots"] == 1
    assert "downstream broker" in captured[0].prompt


def test_prequential_score_portfolio_composes_distinct_lanes_and_leaves_capacity():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    positive_lane = _ScoreLane(
        scorer_id="positive_probability",
        preferred=(
            ("agentic", 2),
            ("numerical_acquisition", 1),
        ),
    )
    return_lane = _ScoreLane(
        scorer_id="expected_return",
        preferred=(
            ("agentic", 2),
            ("numerical_acquisition", 2),
        ),
    )
    portfolio = PrequentialQuotaScorePortfolioPolicy(
        scorers=(return_lane, positive_lane),
        scorer_quotas=(
            ("expected_return", 1),
            ("positive_probability", 1),
        ),
    )
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
        allocation_policy=portfolio,
    )

    result = asyncio.run(
        runtime.run(
            replace(
                _request(),
                remaining_evaluations=9,
                evaluation_slots=3,
                reference_escrow_slots=0,
            )
        )
    )

    requirement = result.broker_decision.allocation_requirement
    assert requirement is not None
    assert len(requirement.required_action_sha256s) == 2
    assert len(result.broker_decision.selected_actions) == 3
    nominated = [
        action
        for proposal in result.proposals
        for action in proposal.actions
        if action.action_sha256 in requirement.required_action_sha256s
    ]
    assert {(value.expert_id, value.native_rank) for value in nominated} == {
        ("agentic", 2),
        ("numerical_acquisition", 1),
    }
    evidence = thaw_json(requirement.evidence)
    assert evidence["scorer_quotas"] == {
        "expected_return": 1,
        "positive_probability": 1,
    }
    assert evidence["nomination_slots"] == 2
    assert evidence["downstream_unreserved_slots"] == 1
    assert evidence["candidate_outcomes_observed"] is False


def test_fully_nominated_wide_slate_bypasses_unconstrained_combinatorics():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    positive_lane = _ScoreLane(
        scorer_id="positive_probability",
        preferred=(
            ("agentic", 2),
            ("numerical_acquisition", 1),
            ("agentic", 1),
        ),
    )
    return_lane = _ScoreLane(
        scorer_id="expected_return",
        preferred=(
            ("agentic", 2),
            ("numerical_acquisition", 2),
        ),
    )
    portfolio = PrequentialQuotaScorePortfolioPolicy(
        scorers=(return_lane, positive_lane),
        scorer_quotas=(
            ("expected_return", 1),
            ("positive_probability", 2),
        ),
    )
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            exact_combination_limit=1,
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
        allocation_policy=portfolio,
    )

    result = asyncio.run(
        runtime.run(
            replace(
                _request(),
                remaining_evaluations=9,
                evaluation_slots=3,
                reference_escrow_slots=0,
            )
        )
    )

    requirement = result.broker_decision.allocation_requirement
    assert requirement is not None
    assert len(requirement.required_action_sha256s) == 3
    assert {
        value.action_sha256 for value in result.broker_decision.selected_actions
    } == set(requirement.required_action_sha256s)
    assert result.broker_decision.search_mode == "exact_joint"
    assert result.broker_decision.complete_slate_count_considered == 1


def test_bounded_beam_preserves_a_unique_phenotype_completion_witness():
    action_names = tuple(f"candidate_{index:02d}" for index in range(24))
    expert = _Expert("agentic", action_names, False)
    runtime = ResidualPortfolioEvolution(
        experts=(expert,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger(),
            return_value=_HashOrderedReturn(),
            exact_combination_limit=1,
            beam_width=1,
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
    )
    request = replace(
        _request(),
        remaining_evaluations=12,
        evaluation_slots=12,
        expert_proposal_slots=(("agentic", len(action_names)),),
        reference_escrow_slots=0,
    )

    result = asyncio.run(runtime.run(request))

    assert result.broker_decision.search_mode == "bounded_joint_beam"
    assert len(result.broker_decision.selected_actions) == 12
    assert (
        len(
            {
                action.phenotype_identity_sha256
                for action in result.broker_decision.selected_actions
            }
        )
        == 12
    )
    assert len(expert.evaluated_action_sha256s) == 12


def test_request_must_name_exact_runtime_expert_set():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    runtime = ResidualPortfolioEvolution(
        experts=(agentic,),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=MaterializedActionEvidenceLedger()
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
    )

    with pytest.raises(ValueError, match="capacities differ"):
        asyncio.run(runtime.run(_request()))


def test_mixed_expert_stage_closure_conserves_credit_and_compiles_lineage():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    broker_ledger = MaterializedActionEvidenceLedger()
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=broker_ledger,
            return_value=_ExpertReturn(
                {
                    "agentic": 0.8,
                    "numerical_acquisition": 0.7,
                }
            ),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
    )
    result = asyncio.run(runtime.run(_request()))
    common = {
        "utility_id": "additive_archive_utility",
        "utility_version": 1,
        "definition_sha256": _sha("additive-archive-utility"),
        "generation": result.request.decision_index,
        "benchmark_sha256": _sha("benchmark"),
    }
    pre = ArchiveUtilitySnapshot(
        **common,
        archive_sha256=_sha("pre-archive"),
        snapshot_receipt=freeze_json({"base_utility_hex": float(0.2).hex()}),
        scalar_utility_hex=float(0.2).hex(),
    )
    post = ArchiveUtilitySnapshot(
        **common,
        archive_sha256=_sha("post-archive"),
        snapshot_receipt=freeze_json({"base_utility_hex": float(0.4).hex()}),
        scalar_utility_hex=float(0.4).hex(),
    )

    projection = ResidualStageCreditProjector(
        archive_utility=_AdditiveArchiveUtility()
    ).project(
        pre_snapshot=pre,
        post_snapshot=post,
        result=result,
    )

    assert projection.stage_credit.stage_gain == pytest.approx(0.2)
    assert sum(
        value.contribution for value in projection.stage_credit.candidate_credits
    ) == pytest.approx(0.2)
    assert all(
        value.normalized_archive_gain == pytest.approx(0.1)
        for value in projection.action_outcomes
    )
    assert {
        value.proposal_expert_id: value.source_role.value
        for value in projection.candidate_provenance
    } == {
        "agentic": "challenger",
        "numerical_acquisition": "backbone",
    }
    assert not {"workload", "model", "provider"}.intersection(projection.to_record())

    for outcome in projection.action_outcomes:
        broker_ledger.append_outcome(outcome)
    lineage = EarnedLineageLedger()
    lineage.register(projection.candidate_provenance)
    issuance = lineage.observe(projection.stage_credit)
    assert len(broker_ledger.outcomes) == 2
    assert tuple(lineage.to_record(current_generation=2)["provenance"])
    assert len(issuance.tickets) == 1
    challenger_id = next(
        value.candidate_id
        for value in projection.candidate_provenance
        if value.proposal_expert_id == "agentic"
    )
    assert issuance.tickets[0].candidate_id == challenger_id


def test_residual_learning_products_publish_with_one_state_swap():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    broker_ledger = MaterializedActionEvidenceLedger()
    lineage = EarnedLineageLedger()
    runtime = ResidualPortfolioEvolution(
        experts=(agentic, numerical),
        broker=RegretBrokeredMaterializedActionPolicy(
            ledger=broker_ledger,
            return_value=_ExpertReturn(
                {
                    "agentic": 0.8,
                    "numerical_acquisition": 0.7,
                }
            ),
        ),
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
    )
    result = asyncio.run(runtime.run(_request()))
    common = {
        "utility_id": "additive_archive_utility",
        "utility_version": 1,
        "definition_sha256": _sha("additive-archive-utility"),
        "generation": result.request.decision_index,
        "benchmark_sha256": _sha("benchmark"),
    }
    pre = ArchiveUtilitySnapshot(
        **common,
        archive_sha256=_sha("pre-archive"),
        snapshot_receipt=freeze_json({"base_utility_hex": float(0.2).hex()}),
        scalar_utility_hex=float(0.2).hex(),
    )
    post = ArchiveUtilitySnapshot(
        **common,
        archive_sha256=_sha("post-archive"),
        snapshot_receipt=freeze_json({"base_utility_hex": float(0.4).hex()}),
        scalar_utility_hex=float(0.4).hex(),
    )
    initial = ResidualLearningState(
        broker_evidence=broker_ledger,
        earned_lineage=lineage,
        current_generation=0,
        revision=0,
    )
    store = TransactionalResidualLearningStore(
        projector=ResidualStageCreditProjector(
            archive_utility=_AdditiveArchiveUtility()
        ),
        _state=initial,
    )

    preparation = store.prepare(
        pre_snapshot=pre,
        post_snapshot=post,
        result=result,
    )

    assert store.state is initial
    assert broker_ledger.outcomes == []
    assert lineage.version == 0
    assert len(preparation.next_state.broker_evidence.outcomes) == 2
    assert len(preparation.ticket_issuance.tickets) == 1
    commit = store.commit(preparation)
    assert commit.prior_state_sha256 == initial.state_sha256
    assert commit.committed_state_sha256 == store.state.state_sha256
    assert store.state.revision == 1
    assert store.state.current_generation == result.request.decision_index
    assert len(store.state.broker_evidence.outcomes) == 2
    assert store.state.earned_lineage.version == 2
    assert broker_ledger.outcomes == []
    assert lineage.version == 0
    with pytest.raises(ValueError, match="absent, stale, or already closed"):
        store.commit(preparation)


def test_transactional_campaign_stage_composes_generic_inverted_ports():
    agentic = _Expert("agentic", ("semantic_a", "semantic_b"), False)
    numerical = _Expert(
        "numerical_acquisition",
        ("acquisition_a", "acquisition_b"),
        True,
    )
    archive = _Archive()
    learning = TransactionalResidualLearningStore(
        projector=ResidualStageCreditProjector(
            archive_utility=_AdditiveArchiveUtility()
        )
    )
    runtime = ResidualPortfolioCampaignStageRuntime(
        experts=(agentic, numerical),
        archive=archive,
        learning=learning,
        slate_value=_ZeroSlateValue(),
        slate_feasibility=_AlwaysFeasible(),
        allocation_policy=_PinnedAllocation(),
        return_value=_ExpertReturn(
            {
                "agentic": 0.8,
                "numerical_acquisition": 0.7,
            }
        ),
    )

    receipt = asyncio.run(runtime.run(_request()))
    decision_sha256 = receipt.result.broker_decision.decision_sha256
    result_sha256 = receipt.result.result_sha256
    receipt_sha256 = receipt.receipt_sha256
    receipt_record = receipt.to_record()

    assert archive.committed
    assert not archive.aborted
    assert receipt.result.to_record()["archive_credit_included"] is False
    assert receipt.learning_commit.committed_state_sha256 == (
        learning.state.state_sha256
    )
    assert len(learning.state.broker_evidence.outcomes) == 2
    assert len(receipt.learning_preparation.ticket_issuance.tickets) == 1
    exploration = receipt.result.broker_decision.exploration_requirement
    assert exploration is not None
    assert exploration.cold_start is True
    assert exploration.required_action_sha256s == ()
    assert {
        value.expert_id for value in receipt.result.broker_decision.selected_actions
    } == {"agentic", "numerical_acquisition"}
    assert receipt_record["workload_model_provider_fields_present"] is False
    allocation = receipt_record["result"]["broker_decision"]["allocation_requirement"]
    assert allocation["evidence"] == {
        "candidate_outcomes_observed": False,
        "complete_sealed_population_compared": True,
    }
    assert (
        allocation["evidence_sha256"]
        == (
            receipt.result.broker_decision.allocation_requirement.to_record()[
                "evidence_sha256"
            ]
        )
    )
    assert (
        "evidence"
        not in receipt.result.to_record()["broker_decision"]["allocation_requirement"]
    )
    assert receipt.result.broker_decision.decision_sha256 == decision_sha256
    assert receipt.result.result_sha256 == result_sha256
    assert receipt.receipt_sha256 == receipt_sha256
    assert receipt_record["result"]["broker_decision"]["decision_sha256"] == (
        decision_sha256
    )
    assert receipt_record["result"]["result_sha256"] == result_sha256
    assert receipt_record["receipt_sha256"] == receipt_sha256

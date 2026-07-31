from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from agent_evolve.application.agentic_evolution import AgenticEvolutionEngine
from agent_evolve.application.budgeted_optimizer import (
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.campaign_capacity_recourse import (
    CampaignCapacityRecourseRequest,
    validate_campaign_capacity_recourse_result,
)
from agent_evolve.application.finite_acquisition_capacity_recourse import (
    FiniteAcquisitionCapacityRecourse,
)
from agent_evolve.application.finite_acquisition_residual_expert import (
    FiniteAcquisitionResidualExpert,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionEvidenceLedger,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.pareto_archive import ParetoArchive
from agent_evolve.application.residual_portfolio_evolution import (
    ResidualPortfolioDecisionRequest,
    ResidualPortfolioEvolution,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.selection.phenotype_recourse import (
    TypedConfigurationPhenotypeIdentityPolicy,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionDecision,
    FiniteAcquisitionObjective,
    FiniteAcquisitionRequest,
    FiniteAcquisitionSelection,
)
from agent_evolve.ports.finite_acquisition_space import FiniteAcquisitionSpaceRequest
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityDecision,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    x: float
    y: float


class _Problem:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec("cost", "min"), ObjectiveSpec("risk", "min"))

    def search_space_description(self) -> str:
        return "Two continuous coordinates in [0,1]."

    def validate(self, config: dict[str, object]) -> bool:
        value = _Candidate.model_validate(config)
        return 0.0 <= value.x <= 1.0 and 0.0 <= value.y <= 1.0

    def evaluate(self, config: dict[str, object]) -> dict[str, float]:
        value = _Candidate.model_validate(config)
        return {
            "cost": float((value.x - 0.2) ** 2 + value.y),
            "risk": float((value.y - 0.8) ** 2 + value.x),
        }


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - recourse is engine-owned.
        raise AssertionError("capacity recourse must not call the model")

    async def reflect(self, request):  # pragma: no cover - recourse is engine-owned.
        raise AssertionError("capacity recourse must not call reflection")


def _object(x: float, y: float) -> FrozenJsonObject:
    value = freeze_json({"x": x, "y": y})
    assert type(value) is FrozenJsonObject
    return value


@dataclass(frozen=True, slots=True)
class _Grid:
    space_id = "capacity_test_grid"
    space_version = 1
    definition_sha256 = _sha("capacity-test-grid")

    def candidates(
        self,
        request: FiniteAcquisitionSpaceRequest,
    ) -> tuple[FrozenJsonObject, ...]:
        excluded = set(request.excluded_configuration_sha256s)
        rows: list[FrozenJsonObject] = []
        for x_index in range(9, 0, -1):
            for y_index in range(1, 10):
                value = _object(x_index / 10.0, y_index / 10.0)
                if typed_json_sha256(value) in excluded:
                    continue
                rows.append(value)
                if len(rows) == request.pool_size:
                    return tuple(rows)
        raise AssertionError("test grid underfilled")

    def features(self, configuration: FrozenJsonObject) -> tuple[float, ...]:
        value = thaw_json(configuration)
        assert type(value) is dict
        return float(value["x"]), float(value["y"])


@dataclass(frozen=True, slots=True)
class _FirstBatch:
    policy_id = "capacity_test_first_batch"
    policy_version = 1
    definition_sha256 = _sha("capacity-test-first-batch")

    def select(self, request: FiniteAcquisitionRequest) -> FiniteAcquisitionDecision:
        return FiniteAcquisitionDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            selected=tuple(
                FiniteAcquisitionSelection(
                    candidate_id=value.candidate_id,
                    configuration_sha256=value.configuration_sha256,
                    acquisition_value=float(request.batch_size - index),
                )
                for index, value in enumerate(
                    request.candidates[: request.batch_size]
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class _ExactToyFeasibility:
    policy_id = "capacity_test_exact_feasibility"
    policy_version = 1
    definition_sha256 = _sha("capacity-test-exact-feasibility")

    def assess(self, request: HardFeasibilityRequest) -> HardFeasibilityDecision:
        value = thaw_json(request.configuration)
        assert type(value) is dict
        infeasible = float(value["x"]) > 0.8
        proof = freeze_json(
            {
                "constraint": "x_le_0.8",
                "x_hex": float(value["x"]).hex(),
            }
        )
        assert type(proof) is FrozenJsonObject
        return HardFeasibilityDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            verdict=(
                HardFeasibilityVerdict.INFEASIBLE
                if infeasible
                else HardFeasibilityVerdict.UNKNOWN
            ),
            proof=proof,
        )


@dataclass(frozen=True, slots=True)
class _ZeroSlateValue:
    definition_sha256 = _sha("capacity-zero-slate-value")

    def value(self, actions):
        return 0.0


@dataclass(frozen=True, slots=True)
class _AlwaysSlateFeasible:
    definition_sha256 = _sha("capacity-always-slate-feasible")

    def permits(self, actions):
        return True


def test_finite_acquisition_recourse_filters_and_fills_real_evaluations() -> None:
    async def scenario() -> None:
        ids = DeterministicIdFactory("capacity_recourse_test")
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=_NeverGenerator(),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=7,
            evaluator_concurrency=2,
        )
        seeds = (
            await engine.register_seed({"x": 0.0, "y": 1.0}, label="left"),
            await engine.register_seed({"x": 1.0, "y": 0.0}, label="right"),
        )
        archive = ParetoArchive(_Problem.objectives)
        for candidate in seeds:
            archive.consider(candidate)
        snapshot = archive.snapshot()
        state = OptimizerState(
            generation=2,
            candidates=seeds,
            archive=snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
            unique_evaluations=2,
            logical_llm_calls=0,
        )
        recourse = FiniteAcquisitionCapacityRecourse(
            objectives=(
                FiniteAcquisitionObjective("cost", "min", 0.0, 2.0),
                FiniteAcquisitionObjective("risk", "min", 0.0, 2.0),
            ),
            space=_Grid(),
            acquisition=_FirstBatch(),
            phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
            engine=engine,
            hard_feasibility=_ExactToyFeasibility(),
            pool_size=16,
            seed=11,
        )
        request = CampaignCapacityRecourseRequest(
            campaign_scope_sha256=_sha("capacity-campaign"),
            preparation_sha256=_sha("capacity-preparation"),
            stage_request_sha256=_sha("capacity-stage"),
            generation=2,
            planned_candidate_occurrences=4,
            realized_candidate_occurrences=2,
            state=state,
        )
        proposal = recourse.propose(request)
        assert len(proposal.members) == 2
        assert thaw_json(proposal.evidence)["evaluation_deferred"] is True
        cache_before_selection = await engine.evaluation_cache_snapshot()
        assert cache_before_selection["misses"] == 2
        selected = await recourse.evaluate_members(
            proposal,
            (proposal.members[0].target_candidate_id,),
        )
        assert len(selected) == 1
        assert selected[0].candidate_id == proposal.members[0].target_candidate_id
        cache_after_selection = await engine.evaluation_cache_snapshot()
        assert cache_after_selection["misses"] == 3
        result = await recourse.fill(request)
        validate_campaign_capacity_recourse_result(
            port=recourse,
            request=request,
            result=result,
        )
        assert len(result.candidates) == 2
        assert all(value.valid for value in result.candidates)
        assert all(value.generation == 2 for value in result.candidates)
        evidence = thaw_json(result.evidence)
        assert evidence["capacity_requested"] == evidence["capacity_realized"] == 2
        assert evidence["hard_feasibility"]["verdict_counts"]["infeasible"] > 0
        assert evidence["current_or_future_outcomes_consulted"] is False
        cache = await engine.evaluation_cache_snapshot()
        assert cache["misses"] == 4

    asyncio.run(scenario())


def test_finite_acquisition_is_a_selected_only_residual_reference_expert() -> None:
    async def scenario() -> None:
        ids = DeterministicIdFactory("capacity_residual_expert_test")
        engine = AgenticEvolutionEngine(
            problem=_Problem(),
            generator=_NeverGenerator(),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=7,
            evaluator_concurrency=2,
        )
        seeds = (
            await engine.register_seed({"x": 0.0, "y": 1.0}, label="left"),
            await engine.register_seed({"x": 1.0, "y": 0.0}, label="right"),
        )
        archive = ParetoArchive(_Problem.objectives)
        for candidate in seeds:
            archive.consider(candidate)
        snapshot = archive.snapshot()
        state = OptimizerState(
            generation=2,
            candidates=seeds,
            archive=snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
            unique_evaluations=2,
            logical_llm_calls=0,
        )
        recourse = FiniteAcquisitionCapacityRecourse(
            objectives=(
                FiniteAcquisitionObjective("cost", "min", 0.0, 2.0),
                FiniteAcquisitionObjective("risk", "min", 0.0, 2.0),
            ),
            space=_Grid(),
            acquisition=_FirstBatch(),
            phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
            engine=engine,
            hard_feasibility=_ExactToyFeasibility(),
            pool_size=16,
            seed=11,
        )
        capacity_request = CampaignCapacityRecourseRequest(
            campaign_scope_sha256=_sha("capacity-residual-campaign"),
            preparation_sha256=_sha("capacity-residual-preparation"),
            stage_request_sha256=_sha("capacity-residual-stage"),
            generation=2,
            planned_candidate_occurrences=4,
            realized_candidate_occurrences=2,
            state=state,
        )
        prior_state_sha256 = _sha("capacity-residual-common-prior-cutoff")
        expert = FiniteAcquisitionResidualExpert(
            recourse=recourse,
            capacity_request=capacity_request,
            prior_state_sha256=prior_state_sha256,
        )
        request = ResidualPortfolioDecisionRequest(
            campaign_scope_sha256=capacity_request.campaign_scope_sha256,
            prior_state_sha256=prior_state_sha256,
            decision_index=1,
            phase=SearchPhase.BASIN_ACQUISITION,
            remaining_decisions=3,
            remaining_evaluations=6,
            evaluation_slots=1,
            expert_proposal_slots=(("numerical_acquisition", 2),),
            proposal_context=freeze_json(
                {
                    "capacity_request_sha256": capacity_request.request_sha256,
                    "state_is_adapter_owned": True,
                }
            ),
            reference_escrow_slots=1,
        )
        runtime = ResidualPortfolioEvolution(
            experts=(expert,),
            broker=RegretBrokeredMaterializedActionPolicy(
                MaterializedActionEvidenceLedger()
            ),
            slate_value=_ZeroSlateValue(),
            slate_feasibility=_AlwaysSlateFeasible(),
        )

        result = await runtime.run(request)

        assert len(result.proposals) == 1
        assert len(result.proposals[0].actions) == 2
        assert len(result.evaluations) == len(result.candidates) == 1
        assert result.evaluations[0].action.reference_action is True
        assert result.evaluations[0].action.expert_id == "numerical_acquisition"
        assert result.candidates[0].valid
        cache = await engine.evaluation_cache_snapshot()
        assert cache["misses"] == 3

    asyncio.run(scenario())

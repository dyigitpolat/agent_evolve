"""Generic one-generation matched model-versus-uniform finite-choice block.

The benchmark supplies catalogs, compilation, phenotype identity, evaluation,
and reward semantics.  This planner owns only the scientific chronology shared
by every domain: seal one K-option authority, freeze a prospective uniform
rank, and run model arm A and engine arm U concurrently on the same parent and
support.  An A=U collision is retained and left to the evaluation cache; it is
never resampled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    FrozenWaveReward,
    GenerationPlan,
    OptimizerBudget,
    OptimizerSlot,
    OptimizerState,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightMemoryEntry,
)
from agent_evolve.application.materialized_variation import (
    materialized_finite_action_decision,
)
from agent_evolve.application.executable_hypothesis import (
    CompiledHypothesisTreatment,
)
from agent_evolve.domain.finite_action_set import (
    FiniteActionSetAuthority,
    FiniteActionSourceMode,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import (
    JsonPath,
    ObjectKey,
    canonical_path_bytes,
    require_sha256,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.finite_action_selection import (
    EngineFiniteActionPolicy,
    EngineFiniteActionRequest,
    FiniteActionDecision,
    ProspectiveUniformRankToken,
)
from agent_evolve.ports.id_factory import IdFactory


PLANNER_POLICY_ID = "matched_finite_action_block"
PLANNER_POLICY_VERSION = 1
_REWARD_SNAPSHOT_DOMAIN = b"agent-evolve:matched-finite-action-reward:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def finite_action_mutation_boundary(
    *,
    contract: FiniteVariationContract,
    parent_candidate_id: CandidateId,
) -> tuple[tuple[str, ...], MutationContract]:
    """Derive the smallest complete mutation boundary for a finite support."""

    validate_finite_variation_contract(contract)
    if type(parent_candidate_id) is not CandidateId:
        raise TypeError("parent_candidate_id must be an exact CandidateId")
    CandidateId.__post_init__(parent_candidate_id)
    probe = CandidateId("candidate_matched_finite_boundary_probe")
    if probe == parent_candidate_id:
        probe = CandidateId("candidate_matched_finite_boundary_probe_alternate")
    paths: dict[bytes, JsonPath] = {}
    max_changed_paths = 0
    max_operations = 0
    for option in contract.options:
        patch = derive_patch(
            contract.parent_configuration,
            option.child_configuration,
            base_candidate_id=parent_candidate_id,
            target_candidate_id=probe,
        )
        changed = {operation.path for operation in patch.operations}
        max_changed_paths = max(max_changed_paths, len(changed))
        max_operations = max(max_operations, len(patch.operations))
        for path in changed:
            paths[canonical_path_bytes(path)] = path
    editable = tuple(paths[key] for key in sorted(paths))
    if not editable or max_changed_paths <= 0 or max_operations <= 0:
        raise ValueError("finite action support has no executable mutation")
    allowed = tuple(
        sorted(
            {
                path.segments[0].value
                for path in editable
                if type(path.segments[0]) is ObjectKey
            }
        )
    )
    if not allowed:
        raise ValueError("finite action support has no object-root mutation path")
    return allowed, MutationContract(
        editable_paths=editable,
        max_changed_paths=max_changed_paths,
        max_operations=max_operations,
        allow_abstention=False,
    )


@runtime_checkable
class MatchedFiniteActionBenchmark(Protocol):
    reward: RewardPolicyBinding

    def compile_registered_hypothesis_treatment(
        self,
        *,
        catalog_id: str,
        parent_candidate_id: CandidateId,
        parent_configuration: object,
        entry: InsightMemoryEntry,
        requested_operator_kind: str,
        context_projection_sha256: str,
        endpoint_definition_sha256: str,
    ) -> CompiledHypothesisTreatment: ...

    def compile_finite_action_set(
        self,
        *,
        compiled_anchor: CompiledHypothesisTreatment,
        required_cardinality: int,
        source_mode: FiniteActionSourceMode,
    ) -> tuple[FiniteActionSetAuthority, object]: ...


@dataclass(slots=True)
class MatchedFiniteActionBlockPlanner:
    """Stateful planner for one already-frozen A/U development block."""

    benchmark: MatchedFiniteActionBenchmark
    engine: AgenticEvolutionEngine
    ids: IdFactory
    memory: InsightMemoryBank
    card_reference: InsightRef
    catalog_id: str
    required_cardinality: int
    context_projection_sha256: str
    endpoint_definition_sha256: str
    task_sha256: str
    pre_outcome_phase_commit_sha256: str
    uniform_policy: EngineFiniteActionPolicy
    source_mode: FiniteActionSourceMode
    phase: str
    required_card_lifecycle: InsightLifecycleState = (
        InsightLifecycleState.QUARANTINED
    )
    authority: FiniteActionSetAuthority | None = field(init=False, default=None)
    compiled_anchor: CompiledHypothesisTreatment | None = field(
        init=False,
        default=None,
    )
    uniform_rank: ProspectiveUniformRankToken | None = field(
        init=False,
        default=None,
    )
    uniform_decision: FiniteActionDecision | None = field(
        init=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, MatchedFiniteActionBenchmark):
            raise TypeError("benchmark must implement MatchedFiniteActionBenchmark")
        if type(self.engine) is not AgenticEvolutionEngine:
            raise TypeError("engine must be an exact AgenticEvolutionEngine")
        if not isinstance(self.ids, IdFactory):
            raise TypeError("ids must implement IdFactory")
        if type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if self.engine.ids is not self.ids or self.engine.memory is not self.memory:
            raise ValueError("planner must share the composed engine IDs and memory")
        if type(self.card_reference) is not InsightRef:
            raise TypeError("card_reference must be an exact InsightRef")
        InsightRef.__post_init__(self.card_reference)
        if type(self.catalog_id) is not str or not self.catalog_id:
            raise ValueError("catalog_id must be non-empty")
        if type(self.required_cardinality) is not int:
            raise TypeError("required_cardinality must be an exact integer")
        for name in (
            "context_projection_sha256",
            "endpoint_definition_sha256",
            "task_sha256",
            "pre_outcome_phase_commit_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if not isinstance(self.uniform_policy, EngineFiniteActionPolicy):
            raise TypeError("uniform_policy must implement EngineFiniteActionPolicy")
        if type(self.source_mode) is not FiniteActionSourceMode:
            raise TypeError("source_mode must be an exact FiniteActionSourceMode")
        if type(self.phase) is not str or not self.phase:
            raise ValueError("phase must be non-empty")
        if type(self.required_card_lifecycle) is not InsightLifecycleState:
            raise TypeError(
                "required_card_lifecycle must be an exact InsightLifecycleState"
            )

    def _reward(self, state: OptimizerState, authority: FiniteActionSetAuthority,
                uniform_decision: FiniteActionDecision) -> FrozenWaveReward:
        binding = getattr(self.benchmark, "reward", None)
        if type(binding) is not RewardPolicyBinding:
            raise TypeError("benchmark must expose an exact reward binding")
        return FrozenWaveReward(
            binding=binding,
            archive_snapshot_hash=state.archive_snapshot_hash,
            reward_snapshot_hash=_hash(
                _REWARD_SNAPSHOT_DOMAIN,
                {
                    "archive_snapshot_hash": state.archive_snapshot_hash,
                    "authority_sha256": authority.authority_sha256,
                    "uniform_decision_sha256": uniform_decision.decision_sha256,
                },
            ),
        )

    def plan(self, state: OptimizerState, budget: OptimizerBudget) -> GenerationPlan:
        del budget
        if type(state) is not OptimizerState:
            raise TypeError("state must be an exact OptimizerState")
        if state.generation != 0 or len(state.candidates) != 1:
            raise ValueError("matched finite block requires exactly one seed state")
        if self.authority is not None:
            raise RuntimeError("matched finite block was already planned")
        parent: EvolutionCandidate = state.candidates[0]
        entry = self.memory.entries_for((self.card_reference,))[0]
        if entry.lifecycle_state is not self.required_card_lifecycle:
            raise ValueError(
                "matched block card lifecycle differs from its frozen requirement"
            )
        compiled = self.benchmark.compile_registered_hypothesis_treatment(
            catalog_id=self.catalog_id,
            parent_candidate_id=parent.candidate_id,
            parent_configuration=parent.configuration,
            entry=entry,
            requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
            context_projection_sha256=self.context_projection_sha256,
            endpoint_definition_sha256=self.endpoint_definition_sha256,
        )
        authority, _ = self.benchmark.compile_finite_action_set(
            compiled_anchor=compiled,
            required_cardinality=self.required_cardinality,
            source_mode=self.source_mode,
        )
        rank = self.uniform_policy.freeze_rank(
            authority,
            task_sha256=self.task_sha256,
            pre_outcome_phase_commit_sha256=(
                self.pre_outcome_phase_commit_sha256
            ),
        )
        uniform_decision = self.uniform_policy.choose(
            EngineFiniteActionRequest(authority=authority, prospective_rank=rank)
        )
        allowed, mutation = finite_action_mutation_boundary(
            contract=authority.support.support_contract,
            parent_candidate_id=parent.candidate_id,
        )
        model_plan = InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(parent,),
            generation=1,
            label="stage_b_adaptive_model_choice",
            allowed_top_level=allowed,
            phase=self.phase,
            mutation_contract=mutation,
            mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
            finite_variation_contract=authority.support.support_contract,
            quarantine_test_insights=(self.card_reference,),
            finite_action_set_authority=authority,
        )
        uniform = materialized_finite_action_decision(
            ids=self.ids,
            parent=parent,
            generation=1,
            label="stage_b_prospective_uniform_choice",
            authority=authority,
            decision=uniform_decision,
            phase=self.phase,
        )
        self.compiled_anchor = compiled
        self.authority = authority
        self.uniform_rank = rank
        self.uniform_decision = uniform_decision
        metadata = tuple(
            sorted(
                (
                    ("finite_action_authority_sha256", authority.authority_sha256),
                    ("finite_action_support_sha256", authority.support.support_sha256),
                    ("prospective_uniform_token_sha256", rank.token_sha256),
                    ("uniform_decision_sha256", uniform_decision.decision_sha256),
                    ("resample_on_a_u_alias", "false"),
                )
            )
        )
        return GenerationPlan(
            generation=1,
            slots=(
                OptimizerSlot.model(
                    slot_id="A",
                    role="adaptive_card_model_choice",
                    plan=model_plan,
                ),
                OptimizerSlot.engine(
                    slot_id="U",
                    role="prospective_uniform_same_support",
                    invocation=uniform,
                ),
            ),
            reward=self._reward(state, authority, uniform_decision),
            planner_policy_id=PLANNER_POLICY_ID,
            planner_policy_version=PLANNER_POLICY_VERSION,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class MatchedFiniteActionBlockFactory:
    """Deferred public-composition factory for one generic matched A/U block."""

    card_reference: InsightRef
    catalog_id: str
    required_cardinality: int
    context_projection_sha256: str
    endpoint_definition_sha256: str
    task_sha256: str
    pre_outcome_phase_commit_sha256: str
    uniform_policy: EngineFiniteActionPolicy
    source_mode: FiniteActionSourceMode = FiniteActionSourceMode.COMPILED_ACTIVE_CARD
    phase: str = "matched_finite_action"
    required_card_lifecycle: InsightLifecycleState = (
        InsightLifecycleState.QUARANTINED
    )

    def build(
        self,
        *,
        benchmark,
        engine: AgenticEvolutionEngine,
        id_factory: IdFactory,
        memory: InsightMemoryBank,
    ) -> MatchedFiniteActionBlockPlanner:
        return MatchedFiniteActionBlockPlanner(
            benchmark=benchmark,
            engine=engine,
            ids=id_factory,
            memory=memory,
            card_reference=self.card_reference,
            catalog_id=self.catalog_id,
            required_cardinality=self.required_cardinality,
            context_projection_sha256=self.context_projection_sha256,
            endpoint_definition_sha256=self.endpoint_definition_sha256,
            task_sha256=self.task_sha256,
            pre_outcome_phase_commit_sha256=(
                self.pre_outcome_phase_commit_sha256
            ),
            uniform_policy=self.uniform_policy,
            source_mode=self.source_mode,
            phase=self.phase,
            required_card_lifecycle=self.required_card_lifecycle,
        )


__all__ = [
    "MatchedFiniteActionBenchmark",
    "MatchedFiniteActionBlockFactory",
    "MatchedFiniteActionBlockPlanner",
    "PLANNER_POLICY_ID",
    "PLANNER_POLICY_VERSION",
    "finite_action_mutation_boundary",
]

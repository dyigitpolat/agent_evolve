"""Generic three-generation evolution over genuine K-option model choices.

The sealed G3 causal screen answers whether an assigned card can reproduce one
exact action.  This module answers a different question: can causal memory help
a model *choose* useful actions from authenticated local neighbourhoods and can
those actions participate in subsequent evolution?

No benchmark semantics live here.  A benchmark compiles card-local finite
action authorities and binds an outcome-blind orthogonal mate.  The planner
owns only the reusable chronology:

* G1: two randomized diagnostic K-choice model calls;
* G2: adaptive A and score-shuffled S model choices, a prospective uniform U
  choice on A's exact support, and an engine-owned disjoint mate E;
* G3: exact reproduction, replay-verified A+E, S+E, and U+E unions, and
  model-selected exact-parent-import A x E and S x E crossovers.

A=U aliases are retained.  Distinct causal occurrences may therefore share a
single physical evaluation through the engine cache; neither arm is resampled.
Reflection is deliberately outside this planner and can consume the exposed
terminal authorities, decisions, references, and slot identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    CrossoverResponseMode,
    EvolutionCandidate,
    InvocationPlan,
    MaterializedInvocation,
    MutationResponseMode,
    OperatorKind,
    ProposalAuthority,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    FrozenWaveReward,
    GenerationPlan,
    OptimizerBudget,
    OptimizerSlot,
    OptimizerState,
    SlotResult,
)
from agent_evolve.application.executable_hypothesis import (
    CompiledHypothesisTreatment,
)
from agent_evolve.application.effective_choice_audit import (
    EffectiveChoiceAuditReceipt,
    audit_effective_choice_plan,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    InsightMemoryEntry,
    context_stratum_hash,
)
from agent_evolve.application.matched_finite_action_block import (
    finite_action_mutation_boundary,
)
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
    materialized_finite_action_decision,
)
from agent_evolve.application.staged_memory import (
    DiagnosticMemoryCheckpointService,
)
from agent_evolve.domain.finite_action_set import (
    MAX_MATCHED_FINITE_ACTIONS,
    MIN_MATCHED_FINITE_ACTIONS,
    FiniteActionSetAuthority,
    FiniteActionSourceMode,
)
from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    DeterministicMemoryControlPolicy,
    FrozenDiagnosticMemoryWave,
    MemoryAssignmentArm,
    MemoryCheckpointClosure,
    MemoryCheckpointClosureStatus,
    ResolvedInsightAssignment,
    WaveSealedCheckpointBuilder,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    derive_exact_parent_crossover_contract,
    resolve_exact_parent_import_for_target,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchRecombiner,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import CandidateDraft, SourceAttribution
from agent_evolve.ports.finite_action_selection import (
    EngineFiniteActionPolicy,
    EngineFiniteActionRequest,
    FiniteActionDecision,
    ProspectiveUniformRankToken,
)
from agent_evolve.ports.id_factory import IdFactory


MULTI_OPTION_EVOLUTION_POLICY_ID = "multi_option_evolution"
MULTI_OPTION_EVOLUTION_POLICY_VERSION = 2
MULTI_OPTION_EVOLUTION_BUDGET = OptimizerBudget(
    max_unique_evaluations=13,
    # Six evolutionary calls plus one separately injected terminal reflection.
    max_logical_llm_calls=7,
    max_generations=3,
)
MULTI_OPTION_G1_SLOT_IDS = (
    "g1_diagnostic_0",
    "g1_diagnostic_1",
)
MULTI_OPTION_G2_SLOT_IDS = (
    "g2_adaptive",
    "g2_score_shuffled",
    "g2_uniform",
    "g2_mate",
)
MULTI_OPTION_G3_CORE_SLOT_IDS = (
    "g3_reproduction",
    "g3_adaptive_union",
    "g3_score_shuffled_union",
    "g3_uniform_union",
)
MULTI_OPTION_G3_CROSSOVER_SLOT_IDS = (
    "g3_adaptive_mate_crossover",
    "g3_score_shuffled_mate_crossover",
)
MULTI_OPTION_G3_SLOT_IDS = (
    *MULTI_OPTION_G3_CORE_SLOT_IDS,
    *MULTI_OPTION_G3_CROSSOVER_SLOT_IDS,
)
MULTI_OPTION_G3_UNION_SOURCES = (
    (MULTI_OPTION_G2_SLOT_IDS[0], MULTI_OPTION_G2_SLOT_IDS[3]),
    (MULTI_OPTION_G2_SLOT_IDS[1], MULTI_OPTION_G2_SLOT_IDS[3]),
    (MULTI_OPTION_G2_SLOT_IDS[2], MULTI_OPTION_G2_SLOT_IDS[3]),
)

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_REWARD_DOMAIN = b"agent-evolve:multi-option-wave-reward:v1\x00"
_MATE_DOMAIN = b"agent-evolve:multi-option-parent-bound-mate:v1\x00"
_CROSSOVER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:adaptive-shuffled-mate-crossover:def:v3\x00"
    b"two bounded exact parent-import crossovers: adaptive x mate; "
    b"score-shuffled x mate; exclude every representable known target by "
    b"linear inverse locus resolution and exact replay"
).hexdigest()
_SEED_ROLE_DEFINITION = {
    "diagnostic_parent": "first admitted G0 seed",
    "evolution_parent": "second admitted G0 seed",
}
_SEED_ROLE_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:ordered-two-seed-role-policy:def:v1\x00"
    + json.dumps(
        _SEED_ROLE_DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


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


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the union.
            raise AssertionError("unsupported JSON-path segment")
    return "".join(parts)


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(second + ".")
        or first.startswith(second + "[")
        or second.startswith(first + ".")
        or second.startswith(first + "[")
    )


@dataclass(frozen=True, slots=True)
class SeedRoleSelection:
    """Two distinct admitted seeds assigned stable experimental roles."""

    diagnostic_parent: EvolutionCandidate
    evolution_parent: EvolutionCandidate

    def __post_init__(self) -> None:
        if (
            type(self.diagnostic_parent) is not EvolutionCandidate
            or type(self.evolution_parent) is not EvolutionCandidate
        ):
            raise TypeError("seed roles require exact EvolutionCandidate values")
        EvolutionCandidate.__post_init__(self.diagnostic_parent)
        EvolutionCandidate.__post_init__(self.evolution_parent)
        if self.diagnostic_parent.candidate_id == self.evolution_parent.candidate_id:
            raise ValueError("seed roles require distinct candidate occurrences")
        if not self.diagnostic_parent.valid or not self.evolution_parent.valid:
            raise ValueError("seed roles require valid evaluated candidates")


@runtime_checkable
class SeedRolePolicy(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str

    def select(self, state: OptimizerState) -> SeedRoleSelection: ...


@dataclass(frozen=True, slots=True)
class OrderedTwoSeedRolePolicy:
    """Default role policy for an explicitly ordered two-seed G0."""

    policy_id: str = field(init=False, default="ordered_two_seed_roles")
    policy_version: int = field(init=False, default=2)
    definition_sha256: str = field(
        init=False,
        default=_SEED_ROLE_DEFINITION_SHA256,
    )

    def select(self, state: OptimizerState) -> SeedRoleSelection:
        if type(state) is not OptimizerState:
            raise TypeError("state must be an exact OptimizerState")
        if state.generation != 0 or len(state.candidates) != 2:
            raise ValueError("ordered seed roles require exactly two G0 seeds")
        return SeedRoleSelection(state.candidates[0], state.candidates[1])


@runtime_checkable
class ParentBoundFiniteChoice(Protocol):
    """Structural boundary for a prospectively frozen engine mate choice."""

    catalog_id: str
    parent_configuration_sha256: str
    finite_contract_sha256: str
    option_id: str
    option_identity_sha256: str
    choice_sha256: str

    def validate_contract(self, contract: FiniteVariationContract) -> None: ...


@runtime_checkable
class MultiOptionEvolutionBenchmark(Protocol):
    """Narrow inverted benchmark boundary used by the generic planner."""

    def bind_finite_variation(
        self,
        catalog_id: str,
        parent_configuration: object,
    ) -> FiniteVariationContract: ...

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


@runtime_checkable
class G3CrossoverPlanPolicy(Protocol):
    """Extension point for terminal model-directed crossover plans."""

    policy_id: str
    policy_version: int
    definition_sha256: str
    slot_ids: tuple[str, ...]

    def plans(
        self,
        *,
        adaptive: EvolutionCandidate,
        shuffled: EvolutionCandidate,
        uniform: EvolutionCandidate,
        mate: EvolutionCandidate,
        phase: str,
        known_targets: tuple[FrozenJsonObject, ...],
    ) -> tuple[InvocationPlan, ...]: ...


@dataclass(frozen=True, slots=True)
class AdaptiveShuffledMateCrossoverPolicy:
    """Default terminal extension: exact-parent-import A x E and S x E crosses."""

    policy_id: str = field(
        init=False,
        default="adaptive_shuffled_mate_crossover",
    )
    policy_version: int = field(init=False, default=3)
    definition_sha256: str = field(
        init=False,
        default=_CROSSOVER_DEFINITION_SHA256,
    )
    slot_ids: tuple[str, ...] = field(
        init=False,
        default=MULTI_OPTION_G3_CROSSOVER_SLOT_IDS,
    )

    def plans(
        self,
        *,
        adaptive: EvolutionCandidate,
        shuffled: EvolutionCandidate,
        uniform: EvolutionCandidate,
        mate: EvolutionCandidate,
        phase: str,
        known_targets: tuple[FrozenJsonObject, ...],
    ) -> tuple[InvocationPlan, ...]:
        del uniform
        if type(known_targets) is not tuple or any(
            type(value) is not FrozenJsonObject for value in known_targets
        ):
            raise TypeError("known_targets must contain exact FrozenJsonObject values")
        plans: list[InvocationPlan] = []
        for child, slot_id in zip(
            (adaptive, shuffled),
            self.slot_ids,
            strict=True,
        ):
            contract = derive_exact_parent_crossover_contract(
                base=child.configuration,
                donor=mate.configuration,
            )
            forbidden = tuple(
                sorted(
                    {
                        resolved
                        for target in known_targets
                        if (
                            resolved := resolve_exact_parent_import_for_target(
                                base=child.configuration,
                                donor=mate.configuration,
                                contract=contract,
                                target=target,
                            )
                        )
                        is not None
                    }
                )
            )
            plans.append(
                InvocationPlan(
                    operator_kind=OperatorKind.TWO_PARENT_CROSSOVER,
                    parents=(child, mate),
                    generation=3,
                    label=slot_id,
                    phase=f"{phase}.model_crossover",
                    crossover_response_mode=(
                        CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
                    ),
                    exact_parent_crossover_contract=contract,
                    forbidden_exact_parent_import_sets=forbidden,
                )
            )
        return tuple(plans)


@dataclass(slots=True)
class MultiOptionEvolutionPlanner:
    """Stateful provider-agnostic planner for a full G0-to-G3 K-choice run."""

    benchmark: MultiOptionEvolutionBenchmark
    engine: AgenticEvolutionEngine
    ids: IdFactory
    memory: InsightMemoryBank
    reward_binding: RewardPolicyBinding
    active_references: tuple[InsightRef, InsightRef]
    model_catalog_id: str
    mate_catalog_id: str
    mate_choice: ParentBoundFiniteChoice
    required_cardinality: int
    uniform_policy: EngineFiniteActionPolicy
    task_sha256: str
    pre_outcome_phase_commit_sha256: str
    endpoint_definition_sha256: str
    context_projection_sha256: str
    estimand_stratum_sha256: str
    phase: str = "multi_option_evolution"
    diagnostic_subset_ranks: tuple[int, int] = (0, 1)
    shuffled_permutation_rank: int = 1
    score_policy: CausalSearchScorePolicy = field(
        default_factory=lambda: CausalSearchScorePolicy(
            prior_effective_sample_size=1.0,
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        )
    )
    controls: DeterministicMemoryControlPolicy = field(
        default_factory=DeterministicMemoryControlPolicy
    )
    seed_role_policy: SeedRolePolicy = field(default_factory=OrderedTwoSeedRolePolicy)
    recombiner: DisjointPatchRecombiner = field(default_factory=DisjointPatchRecombiner)
    crossover_policy: G3CrossoverPlanPolicy = field(
        default_factory=AdaptiveShuffledMateCrossoverPolicy
    )
    trace_sink: object | None = None

    genesis: object | None = field(init=False, default=None)
    wave: FrozenDiagnosticMemoryWave | None = field(init=False, default=None)
    closure: MemoryCheckpointClosure | None = field(init=False, default=None)
    g1_assignments: tuple[ResolvedInsightAssignment, ...] = field(
        init=False,
        default=(),
    )
    g1_authorities: tuple[FiniteActionSetAuthority, ...] = field(
        init=False,
        default=(),
    )
    g2_assignments: tuple[ResolvedInsightAssignment, ...] = field(
        init=False,
        default=(),
    )
    g2_adaptive_authority: FiniteActionSetAuthority | None = field(
        init=False,
        default=None,
    )
    g2_shuffled_authority: FiniteActionSetAuthority | None = field(
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
    mate_invocation: MaterializedInvocation | None = field(
        init=False,
        default=None,
    )
    g3_union_materialization_receipt_sha256s: tuple[str, ...] = field(
        init=False,
        default=(),
    )
    _effective_choice_audit_receipts: dict[
        tuple[int, str], EffectiveChoiceAuditReceipt
    ] = field(init=False, default_factory=dict)
    _checkpoint_service: DiagnosticMemoryCheckpointService = field(init=False)
    _diagnostic_parent_id: CandidateId | None = field(init=False, default=None)
    _evolution_parent_id: CandidateId | None = field(init=False, default=None)
    _diagnostic_parent_hash: str | None = field(init=False, default=None)
    _evolution_parent_hash: str | None = field(init=False, default=None)

    policy_id = MULTI_OPTION_EVOLUTION_POLICY_ID
    policy_version = MULTI_OPTION_EVOLUTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, MultiOptionEvolutionBenchmark):
            raise TypeError("benchmark must implement MultiOptionEvolutionBenchmark")
        if not isinstance(self.engine, AgenticEvolutionEngine):
            raise TypeError("engine must be an AgenticEvolutionEngine")
        if not isinstance(self.ids, IdFactory):
            raise TypeError("ids must implement IdFactory")
        if type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if self.engine.ids is not self.ids or self.engine.memory is not self.memory:
            raise ValueError("planner must share the composed engine IDs and memory")
        if type(self.reward_binding) is not RewardPolicyBinding:
            raise TypeError("reward_binding must be exact")
        RewardPolicyBinding.__post_init__(self.reward_binding)
        if self.endpoint_definition_sha256 != self.reward_binding.definition_hash:
            raise ValueError("endpoint definition must equal the reward/Q definition")
        if (
            type(self.active_references) is not tuple
            or len(self.active_references) != 2
            or self.active_references != tuple(sorted(set(self.active_references)))
        ):
            raise ValueError("active_references must contain two canonical exact refs")
        if any(type(value) is not InsightRef for value in self.active_references):
            raise TypeError("active_references must contain exact InsightRef values")
        if (
            type(self.model_catalog_id) is not str
            or _TOKEN.fullmatch(self.model_catalog_id) is None
        ):
            raise ValueError("model_catalog_id must use the canonical token grammar")
        if (
            type(self.mate_catalog_id) is not str
            or _TOKEN.fullmatch(self.mate_catalog_id) is None
        ):
            raise ValueError("mate_catalog_id must use the canonical token grammar")
        if not isinstance(self.mate_choice, ParentBoundFiniteChoice):
            raise TypeError("mate_choice must implement ParentBoundFiniteChoice")
        if self.mate_choice.catalog_id != self.mate_catalog_id:
            raise ValueError("mate choice and mate catalog differ")
        if not (
            MIN_MATCHED_FINITE_ACTIONS
            <= self.required_cardinality
            <= MAX_MATCHED_FINITE_ACTIONS
        ):
            raise ValueError(
                "required_cardinality must lie in the authenticated finite-action range"
            )
        if not isinstance(self.uniform_policy, EngineFiniteActionPolicy):
            raise TypeError("uniform_policy must implement EngineFiniteActionPolicy")
        for name in (
            "task_sha256",
            "pre_outcome_phase_commit_sha256",
            "endpoint_definition_sha256",
            "context_projection_sha256",
            "estimand_stratum_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.phase) is not str or _TOKEN.fullmatch(self.phase) is None:
            raise ValueError("phase must use the canonical token grammar")
        expected_context = context_stratum_hash(
            problem_id=self.engine.problem_id,
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            phase=self.phase,
        )
        if self.context_projection_sha256 != expected_context:
            raise ValueError(
                "context projection must equal the engine's exact invocation context"
            )
        if self.diagnostic_subset_ranks not in {(0, 1), (1, 0)}:
            raise ValueError("diagnostic_subset_ranks must be a permutation of (0,1)")
        if type(self.shuffled_permutation_rank) is not int or (
            self.shuffled_permutation_rank != 1
        ):
            raise ValueError("two-card score shuffling requires derangement rank 1")
        if not isinstance(self.score_policy, CausalSearchScorePolicy):
            raise TypeError("score_policy must be a CausalSearchScorePolicy")
        if not isinstance(self.controls, DeterministicMemoryControlPolicy):
            raise TypeError("controls must be a DeterministicMemoryControlPolicy")
        if not isinstance(self.seed_role_policy, SeedRolePolicy):
            raise TypeError("seed_role_policy must implement SeedRolePolicy")
        require_sha256(
            self.seed_role_policy.definition_sha256,
            "seed role policy definition_sha256",
        )
        if not isinstance(self.recombiner, DisjointPatchRecombiner):
            raise TypeError("recombiner must be a DisjointPatchRecombiner")
        if not isinstance(self.crossover_policy, G3CrossoverPlanPolicy):
            raise TypeError("crossover_policy must implement G3CrossoverPlanPolicy")
        require_sha256(
            self.crossover_policy.definition_sha256,
            "crossover policy definition_sha256",
        )
        if (
            type(self.crossover_policy.slot_ids) is not tuple
            or any(
                type(value) is not str or _TOKEN.fullmatch(value) is None
                for value in self.crossover_policy.slot_ids
            )
            or len(set(self.crossover_policy.slot_ids))
            != len(self.crossover_policy.slot_ids)
            or set(self.crossover_policy.slot_ids).intersection(
                MULTI_OPTION_G3_CORE_SLOT_IDS
            )
        ):
            raise ValueError("crossover policy slot IDs must be unique canonical IDs")
        if self.trace_sink is not None and not callable(self.trace_sink):
            raise TypeError("trace_sink must be callable")
        self._checkpoint_service = DiagnosticMemoryCheckpointService(
            WaveSealedCheckpointBuilder(self.score_policy),
            trace_sink=self.trace_sink,
        )

    @property
    def adaptive_reference(self) -> InsightRef | None:
        """Exact card chosen by the post-diagnostic adaptive arm, if frozen."""

        if not self.g2_assignments:
            return None
        selected = self.g2_assignments[0].selection_decision.selected
        return selected[0] if len(selected) == 1 else None

    @property
    def effective_choice_audit_receipts(
        self,
    ) -> Mapping[tuple[int, str], EffectiveChoiceAuditReceipt]:
        """Immutable chronological ledger keyed by ``(generation, slot_id)``.

        Only model-authored finite K-choice mutations enter this ledger.  A
        plan is not exposed to the optimizer until its receipt has been
        derived from the exact application-layer authority and contract.
        """

        return MappingProxyType(dict(self._effective_choice_audit_receipts))

    @property
    def terminal_slot_ids(self) -> tuple[str, ...]:
        return (
            *MULTI_OPTION_G3_CORE_SLOT_IDS,
            *self.crossover_policy.slot_ids,
        )

    @property
    def terminal_union_sources(self) -> tuple[tuple[str, str], ...]:
        return MULTI_OPTION_G3_UNION_SOURCES

    def plan(self, state: OptimizerState, budget: OptimizerBudget) -> GenerationPlan:
        if type(state) is not OptimizerState:
            raise TypeError("state must be an exact OptimizerState")
        if type(budget) is not OptimizerBudget:
            raise TypeError("budget must be an exact OptimizerBudget")
        extension_count = len(self.crossover_policy.slot_ids)
        if (
            budget.max_generations != 3
            or budget.max_logical_llm_calls < 4 + extension_count
            or budget.max_unique_evaluations < 11 + extension_count
        ):
            raise ValueError(
                "multi-option evolution budget cannot reserve its complete "
                "three-generation core and terminal model extensions"
            )
        generation = state.generation + 1
        if generation == 1:
            return self._g1(state)
        if generation == 2:
            return self._g2(state)
        if generation == 3:
            return self._g3(state)
        raise ValueError("multi-option evolution has exactly three generations")

    def _reward(self, state: OptimizerState, generation: int) -> FrozenWaveReward:
        return FrozenWaveReward(
            binding=self.reward_binding,
            archive_snapshot_hash=state.archive_snapshot_hash,
            reward_snapshot_hash=_hash(
                _REWARD_DOMAIN,
                {
                    "generation": generation,
                    "archive_snapshot_hash": state.archive_snapshot_hash,
                    "endpoint_definition_sha256": self.endpoint_definition_sha256,
                },
            ),
        )

    def _seed_roles(self, state: OptimizerState) -> SeedRoleSelection:
        if self._diagnostic_parent_id is None:
            roles = self.seed_role_policy.select(state)
            SeedRoleSelection.__post_init__(roles)
            self._diagnostic_parent_id = roles.diagnostic_parent.candidate_id
            self._evolution_parent_id = roles.evolution_parent.candidate_id
            self._diagnostic_parent_hash = (
                roles.diagnostic_parent.occurrence.configuration_hash
            )
            self._evolution_parent_hash = (
                roles.evolution_parent.occurrence.configuration_hash
            )
            return roles
        by_id = {value.candidate_id: value for value in state.candidates}
        try:
            diagnostic = by_id[self._diagnostic_parent_id]
            evolution = by_id[self._evolution_parent_id]
        except KeyError as exc:
            raise ValueError("frozen G0 seed occurrence disappeared") from exc
        if (
            diagnostic.occurrence.configuration_hash != self._diagnostic_parent_hash
            or evolution.occurrence.configuration_hash != self._evolution_parent_hash
        ):
            raise ValueError("frozen G0 seed configuration changed")
        return SeedRoleSelection(diagnostic, evolution)

    def _require_state(self, state: OptimizerState) -> SeedRoleSelection:
        expected_candidates = {0: 2, 1: 4, 2: 8}.get(state.generation)
        if expected_candidates is None:
            raise ValueError("unsupported multi-option generation state")
        if len(state.candidates) != expected_candidates:
            raise ValueError("candidate history differs from the G0-to-G3 chronology")
        if len(state.generation_receipts) != state.generation:
            raise ValueError("generation receipt history is incomplete")
        expected_calls = {0: 0, 1: 2, 2: 4}[state.generation]
        if state.logical_llm_calls != expected_calls:
            raise ValueError("logical-call count differs from the four-call chronology")
        return self._seed_roles(state)

    def _compile_authority(
        self,
        *,
        parent: EvolutionCandidate,
        reference: InsightRef,
        source_mode: FiniteActionSourceMode,
    ) -> FiniteActionSetAuthority:
        entry = self.memory.entries_for((reference,))[0]
        if not entry.retrievable:
            raise ValueError(
                "finite-choice causal assignments require a seed or explicitly "
                "promoted card"
            )
        compiled = self.benchmark.compile_registered_hypothesis_treatment(
            catalog_id=self.model_catalog_id,
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
            source_mode=source_mode,
        )
        if type(authority) is not FiniteActionSetAuthority:
            raise TypeError("benchmark returned an invalid finite action authority")
        FiniteActionSetAuthority.__post_init__(authority)
        if (
            authority.card.reference != reference
            or authority.card.source_mode is not source_mode
            or authority.support.parent_candidate_id != parent.candidate_id
            or authority.support.parent_configuration_sha256
            != parent.occurrence.configuration_hash
            or authority.support.cardinality != self.required_cardinality
        ):
            raise ValueError("finite action authority differs from its requested card")
        return authority

    def _provisional_choice_plan(
        self,
        *,
        parent: EvolutionCandidate,
        generation: int,
        label: str,
        authority: FiniteActionSetAuthority,
    ) -> InvocationPlan:
        contract = authority.support.support_contract
        allowed, mutation = finite_action_mutation_boundary(
            contract=contract,
            parent_candidate_id=parent.candidate_id,
        )
        return InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(parent,),
            generation=generation,
            label=label,
            allowed_top_level=allowed,
            phase=self.phase,
            mutation_contract=mutation,
            mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
            finite_variation_contract=contract,
            quarantine_test_insights=(authority.card.reference,),
            finite_action_set_authority=authority,
        )

    def _resolve_plan(
        self,
        *,
        provisional: InvocationPlan,
        snapshot,
        arm: MemoryAssignmentArm,
        selection_decision,
        block_id: str,
    ) -> tuple[InvocationPlan, ResolvedInsightAssignment]:
        prompt_shape = self.engine.prompt_shape_commitment(
            provisional,
            selected_insight_count=1,
            reward_definition_hash=self.reward_binding.definition_hash,
        )
        assignment = ResolvedInsightAssignment.resolve(
            credit_unit_id=self.ids.new_operator_invocation_id(),
            snapshot=snapshot,
            expected_snapshot_sha256=snapshot.snapshot_sha256,
            block_id=block_id,
            arm=arm,
            selection_decision=selection_decision,
            prompt_shape_sha256=prompt_shape,
        )
        return (
            replace(
                provisional,
                quarantine_test_insights=(),
                resolved_insight_assignment=assignment,
            ),
            assignment,
        )

    def _audit_choice_plans(
        self,
        plans: tuple[InvocationPlan, ...],
    ) -> tuple[EffectiveChoiceAuditReceipt, ...]:
        """Fail closed and atomically append a batch of model K-choice plans."""

        if type(plans) is not tuple or not plans:
            raise ValueError("effective-choice audit batch must be non-empty")
        staged: dict[tuple[int, str], EffectiveChoiceAuditReceipt] = {}
        for plan in plans:
            receipt = audit_effective_choice_plan(
                plan,
                minimum_cardinality=self.required_cardinality,
            )
            key = (receipt.generation, receipt.invocation_label)
            if key in self._effective_choice_audit_receipts or key in staged:
                raise RuntimeError("effective-choice audit coordinate was reused")
            staged[key] = receipt
        self._effective_choice_audit_receipts.update(staged)
        return tuple(staged.values())

    def _g1(self, state: OptimizerState) -> GenerationPlan:
        if self.wave is not None:
            raise RuntimeError("G1 diagnostic wave was already frozen")
        roles = self._require_state(state)
        entries = self.memory.entries_for(self.active_references)
        self.genesis = self.score_policy.genesis(
            exact_context_hash=self.context_projection_sha256,
            estimand_stratum_hash=self.estimand_stratum_sha256,
            priors={entry.reference: entry.initial_score for entry in entries},
        )
        authorities: list[FiniteActionSetAuthority] = []
        assignments: list[ResolvedInsightAssignment] = []
        choice_plans: list[InvocationPlan] = []
        for index, subset_rank in enumerate(self.diagnostic_subset_ranks):
            decision = self.controls.uniform(
                snapshot=self.genesis,
                subset_size=1,
                subset_rank=subset_rank,
            )
            reference = decision.selected[0]
            authority = self._compile_authority(
                parent=roles.diagnostic_parent,
                reference=reference,
                source_mode=FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
            )
            provisional = self._provisional_choice_plan(
                parent=roles.diagnostic_parent,
                generation=1,
                label=MULTI_OPTION_G1_SLOT_IDS[index],
                authority=authority,
            )
            plan, assignment = self._resolve_plan(
                provisional=provisional,
                snapshot=self.genesis,
                arm=MemoryAssignmentArm.DIAGNOSTIC,
                selection_decision=decision,
                block_id="multi_option_g1_diagnostic",
            )
            authorities.append(authority)
            assignments.append(assignment)
            choice_plans.append(plan)
        if tuple(sorted(value.card.reference for value in authorities)) != (
            self.active_references
        ):
            raise RuntimeError("G1 did not cover both active cards exactly once")
        audits = self._audit_choice_plans(tuple(choice_plans))
        slots = tuple(
            OptimizerSlot.model(
                slot_id=MULTI_OPTION_G1_SLOT_IDS[index],
                role="diagnostic_k_option_choice",
                plan=plan,
            )
            for index, plan in enumerate(choice_plans)
        )
        self.g1_authorities = tuple(authorities)
        self.g1_assignments = tuple(assignments)
        self.wave = FrozenDiagnosticMemoryWave(
            wave_id="multi_option_g1_diagnostic_wave",
            prior_snapshot=self.genesis,
            assignments=tuple(
                sorted(assignments, key=lambda value: value.assignment_sha256)
            ),
            reward_definition_hash=self.reward_binding.definition_hash,
            no_yield_reward=self.reward_binding.failure_score,
        )
        self._checkpoint_service.publish_frozen_wave(self.wave)
        return GenerationPlan(
            generation=1,
            slots=slots,
            reward=self._reward(state, 1),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        ("diagnostic_wave_sha256", self.wave.wave_sha256),
                        ("genesis_snapshot_sha256", self.genesis.snapshot_sha256),
                        *(
                            (
                                f"{slot_id}_authority_sha256",
                                authority.authority_sha256,
                            )
                            for slot_id, authority in zip(
                                MULTI_OPTION_G1_SLOT_IDS,
                                authorities,
                                strict=True,
                            )
                        ),
                        *(
                            (
                                f"{slot_id}_effective_choice_audit_sha256",
                                receipt.receipt_sha256,
                            )
                            for slot_id, receipt in zip(
                                MULTI_OPTION_G1_SLOT_IDS,
                                audits,
                                strict=True,
                            )
                        ),
                    )
                )
            ),
        )

    def _materialize_mate(
        self,
        *,
        parent: EvolutionCandidate,
    ) -> MaterializedInvocation:
        contract = self.benchmark.bind_finite_variation(
            self.mate_catalog_id,
            parent.configuration,
        )
        self.mate_choice.validate_contract(contract)
        option = contract.resolve(self.mate_choice.option_id)
        if option.identity_sha256 != self.mate_choice.option_identity_sha256:
            raise ValueError("mate option identity changed")
        candidate_id = self.ids.new_candidate_id()
        patch = derive_patch(
            parent.configuration,
            option.child_configuration,
            base_candidate_id=parent.candidate_id,
            target_candidate_id=candidate_id,
        )
        if not patch.operations:
            raise ValueError("orthogonal mate must change at least one path")
        paths = tuple(sorted({_path_text(value.path) for value in patch.operations}))
        top_level = tuple(
            sorted(
                {
                    value.path.segments[0].value
                    for value in patch.operations
                    if type(value.path.segments[0]) is ObjectKey
                }
            )
        )
        configuration = thaw_json(option.child_configuration)
        if type(configuration) is not dict:
            raise TypeError("mate child must be a typed-JSON object")
        receipt_hash = _hash(
            _MATE_DOMAIN,
            {
                "choice_sha256": self.mate_choice.choice_sha256,
                "parent_candidate_id": parent.candidate_id.value,
                "target_candidate_id": candidate_id.value,
                "patch_hash": patch.patch_hash,
            },
        )
        return MaterializedInvocation(
            plan=InvocationPlan(
                operator_kind=OperatorKind.TYPED_MUTATION,
                parents=(parent,),
                generation=2,
                label=MULTI_OPTION_G2_SLOT_IDS[3],
                allowed_top_level=top_level,
                phase=f"{self.phase}.mate",
            ),
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale=(
                    "Engine-owned parent-bound mate on support disjoint from "
                    "every authenticated model option."
                ),
                intended_changes=paths,
                source_attribution=tuple(
                    SourceAttribution(path, "mutation") for path in paths
                ),
            ),
            candidate_id=candidate_id,
            materialization_policy_id="parent_bound_finite_mate",
            materialization_policy_version=1,
            materialization_receipt_hash=receipt_hash,
        )

    def _require_disjoint_support(
        self,
        authority: FiniteActionSetAuthority,
        mate: MaterializedInvocation,
    ) -> None:
        mate_paths = mate.draft.intended_changes
        for row in authority.support.options:
            if any(
                _paths_overlap(left, right)
                for left in row.changed_paths
                for right in mate_paths
            ):
                raise ValueError(
                    "mate support overlaps an authenticated model-choice path"
                )

    def _g2(self, state: OptimizerState) -> GenerationPlan:
        if self.wave is None or self.genesis is None or not self.g1_authorities:
            raise RuntimeError("G1 diagnostic authorities are unavailable")
        if self.closure is not None:
            raise RuntimeError("G2 memory checkpoint was already closed")
        roles = self._require_state(state)
        g1_receipt = state.generation_receipts[0]
        if tuple(value.slot.slot_id for value in g1_receipt.slot_results) != (
            MULTI_OPTION_G1_SLOT_IDS
        ):
            raise ValueError("G1 slot order differs from the planner contract")
        self.closure = self._checkpoint_service.close_generation(
            self.wave,
            g1_receipt,
        )
        if self.closure.status is not MemoryCheckpointClosureStatus.SEALED:
            raise RuntimeError("G1 diagnostic wave was infrastructure-invalidated")
        snapshot = self.closure.snapshot
        if snapshot is None:
            raise RuntimeError("sealed G1 wave has no memory checkpoint")
        if any(not entry.identified for entry in snapshot.entries):
            raise ValueError("G1 did not identify both card effects")
        if len({entry.retrieval_score for entry in snapshot.entries}) != 2:
            raise ValueError(
                "G1 card scores tied; adaptive/shuffled arms are undefined"
            )

        decisions = (
            self.controls.adaptive(snapshot=snapshot, subset_size=1),
            self.controls.score_shuffled(
                snapshot=snapshot,
                subset_size=1,
                permutation_rank=self.shuffled_permutation_rank,
            ),
        )
        if decisions[0].selected == decisions[1].selected:
            raise ValueError("score-shuffled control did not derange card selection")
        source_modes = (
            FiniteActionSourceMode.COMPILED_ACTIVE_CARD,
            FiniteActionSourceMode.COMPILED_SHUFFLED_CARD,
        )
        authorities: list[FiniteActionSetAuthority] = []
        assignments: list[ResolvedInsightAssignment] = []
        choice_plans: list[InvocationPlan] = []
        roles_text = ("adaptive_k_option_choice", "score_shuffled_k_option_choice")
        for index, (decision, source_mode) in enumerate(
            zip(decisions, source_modes, strict=True)
        ):
            authority = self._compile_authority(
                parent=roles.evolution_parent,
                reference=decision.selected[0],
                source_mode=source_mode,
            )
            provisional = self._provisional_choice_plan(
                parent=roles.evolution_parent,
                generation=2,
                label=MULTI_OPTION_G2_SLOT_IDS[index],
                authority=authority,
            )
            plan, assignment = self._resolve_plan(
                provisional=provisional,
                snapshot=snapshot,
                arm=(
                    MemoryAssignmentArm.ADAPTIVE
                    if index == 0
                    else MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL
                ),
                selection_decision=decision,
                block_id="multi_option_g2_matched_memory",
            )
            authorities.append(authority)
            assignments.append(assignment)
            choice_plans.append(plan)
        adaptive_authority, shuffled_authority = authorities
        self.g2_adaptive_authority = adaptive_authority
        self.g2_shuffled_authority = shuffled_authority
        self.g2_assignments = tuple(assignments)
        self.uniform_rank = self.uniform_policy.freeze_rank(
            adaptive_authority,
            task_sha256=self.task_sha256,
            pre_outcome_phase_commit_sha256=self.pre_outcome_phase_commit_sha256,
        )
        self.uniform_decision = self.uniform_policy.choose(
            EngineFiniteActionRequest(
                authority=adaptive_authority,
                prospective_rank=self.uniform_rank,
            )
        )
        uniform = materialized_finite_action_decision(
            ids=self.ids,
            parent=roles.evolution_parent,
            generation=2,
            label=MULTI_OPTION_G2_SLOT_IDS[2],
            authority=adaptive_authority,
            decision=self.uniform_decision,
            phase=f"{self.phase}.uniform",
        )
        mate = self._materialize_mate(parent=roles.evolution_parent)
        self._require_disjoint_support(adaptive_authority, mate)
        self._require_disjoint_support(shuffled_authority, mate)
        self.mate_invocation = mate
        audits = self._audit_choice_plans(tuple(choice_plans))
        model_slots = tuple(
            OptimizerSlot.model(
                slot_id=MULTI_OPTION_G2_SLOT_IDS[index],
                role=roles_text[index],
                plan=plan,
            )
            for index, plan in enumerate(choice_plans)
        )
        return GenerationPlan(
            generation=2,
            slots=(
                *model_slots,
                OptimizerSlot.engine(
                    slot_id=MULTI_OPTION_G2_SLOT_IDS[2],
                    role="prospective_uniform_same_adaptive_support",
                    invocation=uniform,
                ),
                OptimizerSlot.engine(
                    slot_id=MULTI_OPTION_G2_SLOT_IDS[3],
                    role="orthogonal_engine_mate",
                    invocation=mate,
                ),
            ),
            reward=self._reward(state, 2),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        (
                            "adaptive_authority_sha256",
                            adaptive_authority.authority_sha256,
                        ),
                        (
                            "adaptive_reference",
                            assignments[0]
                            .selection_decision.selected[0]
                            .insight_id.value,
                        ),
                        ("mate_choice_sha256", self.mate_choice.choice_sha256),
                        ("memory_snapshot_sha256", snapshot.snapshot_sha256),
                        (
                            "score_shuffled_authority_sha256",
                            shuffled_authority.authority_sha256,
                        ),
                        *(
                            (
                                f"{slot_id}_effective_choice_audit_sha256",
                                receipt.receipt_sha256,
                            )
                            for slot_id, receipt in zip(
                                MULTI_OPTION_G2_SLOT_IDS[:2],
                                audits,
                                strict=True,
                            )
                        ),
                        (
                            "uniform_decision_sha256",
                            self.uniform_decision.decision_sha256,
                        ),
                        ("uniform_rank_sha256", self.uniform_rank.token_sha256),
                    )
                )
            ),
        )

    @staticmethod
    def _require_model_choice(
        result: SlotResult,
        authority: FiniteActionSetAuthority,
    ) -> EvolutionCandidate:
        outcome = result.outcome
        candidate = outcome.candidate
        decision = outcome.finite_action_decision
        if (
            result.slot.proposal_authority is not ProposalAuthority.MODEL
            or outcome.failure_stage is not None
            or candidate is None
            or not candidate.valid
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
            or decision is None
        ):
            raise ValueError("model K-choice endpoint did not complete successfully")
        if (
            result.slot.plan.finite_action_set_authority != authority
            or decision.authority_sha256 != authority.authority_sha256
            or decision.support_sha256 != authority.support.support_sha256
            or candidate.selected_insight_refs != (authority.card.reference,)
            or candidate.claimed_insight_ids
            != (authority.card.reference.insight_id.value,)
        ):
            raise ValueError("model K-choice endpoint escaped its finite authority")
        return candidate

    @staticmethod
    def _require_engine_choice(
        result: SlotResult,
        invocation: MaterializedInvocation,
    ) -> EvolutionCandidate:
        outcome = result.outcome
        candidate = outcome.candidate
        if (
            result.slot.proposal_authority is not ProposalAuthority.ENGINE
            or result.slot.materialized != invocation
            or outcome.failure_stage is not None
            or candidate is None
            or not candidate.valid
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
            or candidate.candidate_id != invocation.candidate_id
        ):
            raise ValueError("engine endpoint did not complete successfully")
        expected = freeze_json(invocation.draft.configuration)
        if not typed_json_equal(candidate.configuration, expected):
            raise ValueError("engine endpoint differs from its materialization")
        return candidate

    def _union(
        self,
        *,
        ancestor: EvolutionCandidate,
        child: EvolutionCandidate,
        mate: EvolutionCandidate,
        slot_id: str,
    ) -> MaterializedInvocation:
        materialization = self.recombiner.materialize(
            ancestor=ancestor.configuration,
            ancestor_candidate_id=ancestor.candidate_id,
            left=child.configuration,
            left_candidate_id=child.candidate_id,
            right=mate.configuration,
            right_candidate_id=mate.candidate_id,
            target_candidate_id=self.ids.new_candidate_id(),
        )
        return materialized_disjoint_invocation(
            plan=InvocationPlan(
                operator_kind=OperatorKind.THREE_WAY_RECOMBINATION,
                parents=(child, mate),
                generation=3,
                label=slot_id,
                common_ancestor=ancestor,
                phase=f"{self.phase}.disjoint_union",
            ),
            materialization=materialization,
        )

    def _g3(self, state: OptimizerState) -> GenerationPlan:
        if (
            self.g2_adaptive_authority is None
            or self.g2_shuffled_authority is None
            or self.uniform_decision is None
            or self.mate_invocation is None
        ):
            raise RuntimeError("G2 finite-choice authorities are unavailable")
        roles = self._require_state(state)
        g2_receipt = state.generation_receipts[1]
        if tuple(value.slot.slot_id for value in g2_receipt.slot_results) != (
            MULTI_OPTION_G2_SLOT_IDS
        ):
            raise ValueError("G2 slot order differs from the planner contract")
        adaptive = self._require_model_choice(
            g2_receipt.slot_results[0],
            self.g2_adaptive_authority,
        )
        shuffled = self._require_model_choice(
            g2_receipt.slot_results[1],
            self.g2_shuffled_authority,
        )
        uniform_invocation = g2_receipt.slot_results[2].slot.materialized
        if uniform_invocation is None:
            raise RuntimeError("G2 uniform materialization disappeared")
        uniform = self._require_engine_choice(
            g2_receipt.slot_results[2],
            uniform_invocation,
        )
        mate = self._require_engine_choice(
            g2_receipt.slot_results[3],
            self.mate_invocation,
        )
        reproduction = InvocationPlan(
            operator_kind=OperatorKind.REPRODUCTION,
            parents=(roles.evolution_parent,),
            generation=3,
            label=MULTI_OPTION_G3_CORE_SLOT_IDS[0],
            phase=f"{self.phase}.reproduction",
        )
        unions = tuple(
            self._union(
                ancestor=roles.evolution_parent,
                child=child,
                mate=mate,
                slot_id=slot_id,
            )
            for child, slot_id in zip(
                (adaptive, shuffled, uniform),
                MULTI_OPTION_G3_CORE_SLOT_IDS[1:],
                strict=True,
            )
        )
        known_by_sha256 = {
            typed_json_sha256(candidate.configuration): candidate.configuration
            for candidate in state.candidates
        }
        for union in unions:
            target = freeze_json(union.draft.configuration)
            if type(target) is not FrozenJsonObject:
                raise TypeError("scheduled union must materialize an object target")
            known_by_sha256.setdefault(typed_json_sha256(target), target)
        known_targets = tuple(
            known_by_sha256[digest] for digest in sorted(known_by_sha256)
        )
        crossover_plans = self.crossover_policy.plans(
            adaptive=adaptive,
            shuffled=shuffled,
            uniform=uniform,
            mate=mate,
            phase=self.phase,
            known_targets=known_targets,
        )
        if (
            type(crossover_plans) is not tuple
            or len(crossover_plans) != len(self.crossover_policy.slot_ids)
            or tuple(value.label for value in crossover_plans)
            != self.crossover_policy.slot_ids
        ):
            raise ValueError("crossover policy plans differ from its frozen slot IDs")
        for plan in crossover_plans:
            if (
                type(plan) is not InvocationPlan
                or plan.operator_kind is not OperatorKind.TWO_PARENT_CROSSOVER
                or plan.generation != 3
                or plan.use_memory
                or plan.quarantine_test_insights
                or plan.resolved_insight_assignment is not None
                or plan.insight_treatment_requirement is not None
                or plan.finite_action_set_authority is not None
            ):
                raise ValueError(
                    "terminal crossover extensions must be memory-free model plans"
                )
        self.g3_union_materialization_receipt_sha256s = tuple(
            value.materialization_receipt_hash for value in unions
        )
        return GenerationPlan(
            generation=3,
            slots=(
                OptimizerSlot.reproduction(
                    slot_id=MULTI_OPTION_G3_CORE_SLOT_IDS[0],
                    role="evolution_parent_reproduction",
                    plan=reproduction,
                ),
                *(
                    OptimizerSlot.engine(
                        slot_id=slot_id,
                        role="deterministic_disjoint_union",
                        invocation=invocation,
                    )
                    for slot_id, invocation in zip(
                        MULTI_OPTION_G3_CORE_SLOT_IDS[1:],
                        unions,
                        strict=True,
                    )
                ),
                *(
                    OptimizerSlot.model(
                        slot_id=slot_id,
                        role="model_selected_exact_parent_crossover",
                        plan=plan,
                    )
                    for slot_id, plan in zip(
                        self.crossover_policy.slot_ids,
                        crossover_plans,
                        strict=True,
                    )
                ),
            ),
            reward=self._reward(state, 3),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        (
                            "adaptive_reference",
                            self.adaptive_reference.insight_id.value,
                        ),
                        *(
                            (
                                f"{slot_id}_materialization_receipt_sha256",
                                invocation.materialization_receipt_hash,
                            )
                            for slot_id, invocation in zip(
                                MULTI_OPTION_G3_CORE_SLOT_IDS[1:],
                                unions,
                                strict=True,
                            )
                        ),
                        (
                            "crossover_policy_definition_sha256",
                            self.crossover_policy.definition_sha256,
                        ),
                    )
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class MultiOptionEvolutionPlannerFactory:
    """Deferred composition seam for benchmark and runtime dependencies."""

    reward_binding: RewardPolicyBinding
    active_references: tuple[InsightRef, InsightRef]
    model_catalog_id: str
    mate_catalog_id: str
    mate_choice: ParentBoundFiniteChoice
    required_cardinality: int
    uniform_policy: EngineFiniteActionPolicy
    task_sha256: str
    pre_outcome_phase_commit_sha256: str
    endpoint_definition_sha256: str
    context_projection_sha256: str
    estimand_stratum_sha256: str
    phase: str = "multi_option_evolution"
    diagnostic_subset_ranks: tuple[int, int] = (0, 1)
    shuffled_permutation_rank: int = 1
    score_policy: CausalSearchScorePolicy = field(
        default_factory=lambda: CausalSearchScorePolicy(
            prior_effective_sample_size=1.0,
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        )
    )
    controls: DeterministicMemoryControlPolicy = field(
        default_factory=DeterministicMemoryControlPolicy
    )
    seed_role_policy: SeedRolePolicy = field(default_factory=OrderedTwoSeedRolePolicy)
    recombiner: DisjointPatchRecombiner = field(default_factory=DisjointPatchRecombiner)
    crossover_policy: G3CrossoverPlanPolicy = field(
        default_factory=AdaptiveShuffledMateCrossoverPolicy
    )
    trace_sink: object | None = None

    def build(
        self,
        *,
        benchmark: MultiOptionEvolutionBenchmark,
        engine: AgenticEvolutionEngine,
        id_factory: IdFactory,
        memory: InsightMemoryBank,
    ) -> MultiOptionEvolutionPlanner:
        return MultiOptionEvolutionPlanner(
            benchmark=benchmark,
            engine=engine,
            ids=id_factory,
            memory=memory,
            reward_binding=self.reward_binding,
            active_references=self.active_references,
            model_catalog_id=self.model_catalog_id,
            mate_catalog_id=self.mate_catalog_id,
            mate_choice=self.mate_choice,
            required_cardinality=self.required_cardinality,
            uniform_policy=self.uniform_policy,
            task_sha256=self.task_sha256,
            pre_outcome_phase_commit_sha256=(self.pre_outcome_phase_commit_sha256),
            endpoint_definition_sha256=self.endpoint_definition_sha256,
            context_projection_sha256=self.context_projection_sha256,
            estimand_stratum_sha256=self.estimand_stratum_sha256,
            phase=self.phase,
            diagnostic_subset_ranks=self.diagnostic_subset_ranks,
            shuffled_permutation_rank=self.shuffled_permutation_rank,
            score_policy=self.score_policy,
            controls=self.controls,
            seed_role_policy=self.seed_role_policy,
            recombiner=self.recombiner,
            crossover_policy=self.crossover_policy,
            trace_sink=self.trace_sink,
        )


__all__ = [
    "AdaptiveShuffledMateCrossoverPolicy",
    "G3CrossoverPlanPolicy",
    "MULTI_OPTION_EVOLUTION_BUDGET",
    "MULTI_OPTION_EVOLUTION_POLICY_ID",
    "MULTI_OPTION_EVOLUTION_POLICY_VERSION",
    "MULTI_OPTION_G1_SLOT_IDS",
    "MULTI_OPTION_G2_SLOT_IDS",
    "MULTI_OPTION_G3_CORE_SLOT_IDS",
    "MULTI_OPTION_G3_CROSSOVER_SLOT_IDS",
    "MULTI_OPTION_G3_SLOT_IDS",
    "MULTI_OPTION_G3_UNION_SOURCES",
    "MultiOptionEvolutionBenchmark",
    "MultiOptionEvolutionPlanner",
    "MultiOptionEvolutionPlannerFactory",
    "OrderedTwoSeedRolePolicy",
    "ParentBoundFiniteChoice",
    "SeedRolePolicy",
    "SeedRoleSelection",
]

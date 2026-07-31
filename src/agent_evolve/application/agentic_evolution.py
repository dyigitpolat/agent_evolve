"""Explicit, interceptable agentic variation workflow.

This module is intentionally generation-policy agnostic.  A caller supplies an
ordered set of invocation plans (mutation, crossover, three-way recombination,
repair, or reproduction), while the engine owns immutable lineage construction,
concurrent generation/evaluation, obligation checks, reward units, and
structured trace emission. Legacy retrieval remains available, but staged
causal workflows bind already-resolved memory assignments into the plan before
the engine sees them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from typing import Any

from pydantic import BaseModel

from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightLifecycleTransition,
    InsightMemoryBank,
    InsightMemoryEntry,
    InsightOrigin,
    InsightRelationKind,
    ReflectedInsightBatchItem,
    context_stratum_hash,
)
from agent_evolve.application.executable_hypothesis import (
    CompiledHypothesisTreatment,
    registered_source_evidence_sha256,
)
from agent_evolve.application.finite_action_selection import (
    seal_model_finite_action_decision,
)
from agent_evolve.application.reflection_workflow import (
    PlannedReflectionCall,
    PlannedReflectionBatchCall,
    ReflectionCardContractError,
    ReflectionPromptShard,
    ReflectionWorkflow,
    ReflectionWorkflowRequest,
    ReflectionWorkflowResult,
)
from agent_evolve.application.evaluation_cache import (
    AsyncEvaluationCache,
    EvaluationCacheTraceEvent,
)
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluation,
    DetailedEvaluationAdapter,
    EvaluationTimings,
    EvaluatorIdentity,
    normalize_detailed_payload,
)
from agent_evolve.application.outcome_relation import (
    OutcomeRelation,
    OutcomeRelationPolicyBinding,
    objective_pareto_outcome_binding,
)
from agent_evolve.core.problem import (
    ObjectiveSpec,
    normalize_objective_values,
    validate_objective_specs,
)
from agent_evolve.core.optimization_semantics import (
    OptimizationSemantics,
    render_optimization_semantics,
)
from agent_evolve.domain.finite_variation import (
    FiniteActionEvidenceBinding,
    FiniteVariationContract,
    bind_finite_action_evidence,
    validate_finite_variation_contract,
)
from agent_evolve.domain.finite_action_set import FiniteActionSetAuthority
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.lineage import (
    CandidateOccurrence,
    ParentRole,
    PreservationClaim,
    PreservationSource,
    VariationCase,
    VariationKind,
    VariationParent,
)
from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    PatchOperation,
    ReplaceScalar,
    canonical_path_bytes,
    operation_effect_bytes,
    operation_kind,
    require_sha256,
    validate_json_path,
)
from agent_evolve.domain.outcome import FailureCategory
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_typed_json_bytes,
    freeze_json,
    is_json_scalar,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.memory.randomized_subset import InsightSelectionDecision
from agent_evolve.policies.memory.prompt_shape import (
    DefaultEvidencePromptShapePolicyV3,
    PromptShapeCommitmentPolicy,
    PromptShapeInputs,
)
from agent_evolve.policies.memory.staged_causal import (
    MemoryAssignmentArm,
    ResolvedInsightAssignment,
)
from agent_evolve.policies.memory.treatment_compliance import (
    FiniteTreatmentAction,
    InsightTreatmentRequirement,
    StrictTreatmentCompliancePolicy,
    TreatmentAdmissionReceipt,
    TreatmentAdmissionRequest,
    TreatmentCompliancePolicy,
    TreatmentComplianceRejected,
    TreatmentInsightEvidence,
    TreatmentPreflightReceipt,
    TreatmentPreflightRequest,
    validate_treatment_admission_receipt,
    validate_treatment_preflight_receipt,
)
from agent_evolve.policies.structured_output_budget import (
    FixedStructuredOutputBudgetPolicy,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
    PhenotypeIdentityPolicy,
    TypedConfigurationPhenotypeIdentityPolicy,
)
from agent_evolve.policies.variation.crossover_inheritance import (
    CrossoverInheritanceClaim,
    CrossoverInheritanceSource,
    materialize_crossover_inheritance,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    ExactParentCrossoverContract,
    ExactParentSource,
    derive_exact_parent_crossover_contract,
    exact_parent_import_exclusions_sha256,
    materialize_exact_parent_crossover,
    validate_exact_parent_import_exclusions,
)
from agent_evolve.policies.variation.typed_patch import (
    ParentConfiguration,
    PatchResolution,
    PreservationError,
    PreservationObligationRequest,
    ResolutionChoice,
    ThreeWayPatchClassification,
    ThreeWayRelationKind,
    apply_patch,
    bind_parent_configuration,
    classify_three_way_patches,
    derive_patch,
    derive_preservation_obligations,
    replace_existing_path,
    validate_three_way_resolutions,
    value_at_path,
    verify_preservation_claims,
)
from agent_evolve.ports.generation_failure import (
    GenerationFailureDisposition,
    classify_generation_failure,
)
from agent_evolve.ports.finite_action_selection import (
    FiniteActionDecision,
    FiniteActionSelectorKind,
    validate_finite_action_decision,
)
from agent_evolve.ports.structured_output_budget import (
    StructuredOutputBudgetPolicy,
    StructuredOutputRequestKind,
    resolve_structured_output_budget,
    structured_output_budget_policy_metadata,
)
from agent_evolve.ports.objective_resolution import (
    ObjectiveResolutionPort,
    ObjectiveResolutionReceipt,
    ObjectiveResolutionRequest,
    objective_resolution_policy_metadata,
    resolve_objectives,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AgenticGenerator,
    AtomicMutationDraft,
    AtomicMutationOutputContract,
    CANDIDATE_COMPONENT_PATH_CONTRACT,
    CandidateDraft,
    ConflictResolutionDraft,
    ExactParentCrossoverDraft,
    ExactParentCrossoverOutputContract,
    FiniteVariationSelectionDraft,
    InsightDraft,
    ReflectionInsightContract,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    SourceAttribution,
    TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT,
    VariationGenerationRequest,
    resolve_finite_variation_selection,
    validate_reflection_insight_draft,
)


TraceSink = Callable[[Mapping[str, object]], None]
ReflectionRowProjector = Callable[
    [Mapping[str, object]],
    Mapping[str, object],
]
RewardPolicy = Callable[
    ["EvolutionCandidate", tuple["EvolutionCandidate", ...], Sequence[ObjectiveSpec]],
    float,
]


@dataclass(frozen=True, slots=True)
class ReflectionRowProjectionBinding:
    """Identity-bound, request-independent reflection evidence projection.

    The default engine path leaves rows untouched and therefore preserves all
    historical prompt bytes.  Experiments may inject a narrow redaction or
    view policy without changing evaluator, reward, or variation semantics.
    Machine-derived contrast evidence and invocation identity are immutable
    across every projection.
    """

    project: ReflectionRowProjector
    policy_id: str
    policy_version: int
    definition_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.project):
            raise TypeError("reflection row projector must be callable")
        if type(self.policy_id) is not str or not self.policy_id.strip():
            raise ValueError("reflection projection policy_id must be non-empty")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("reflection projection policy_version must be positive")
        require_sha256(
            self.definition_sha256,
            "reflection projection definition_sha256",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


PromptBuilder = Callable[
    [str, "PreparedInvocation", tuple[dict[str, object], ...]], str
]

_REWARD_DEFINITION = (
    b"agent-evolve:reward:v1:operator-compliant-parent-relative-scalarization"
)
REWARD_DEFINITION_HASH = hashlib.sha256(_REWARD_DEFINITION).hexdigest()
_REWARD_BINDING_DOMAIN = b"agent-evolve:reward-binding:v1\x00"
_DETAILED_EVALUATION_CACHE_DOMAIN = b"agent-evolve:detailed-evaluation-cache-key:v1\x00"
_REFLECTION_CALL_RECEIPT_DOMAIN = (
    b"agent-evolve:reflection-call-receipt:v2-request-bound\x00"
)
_REFLECTION_CALL_REQUEST_DOMAIN = b"agent-evolve:reflection-call-request:v1\x00"
_REFLECTION_CALL_TELEMETRY_DOMAIN = b"agent-evolve:reflection-call-telemetry:v1\x00"
_REFLECTION_PUBLICATION_DOMAIN = b"agent-evolve:reflection-publication:v1\x00"
_REFLECTION_SOURCE_OUTCOME_DOMAIN = b"agent-evolve:reflection-source-outcome:v1\x00"


@dataclass(frozen=True, slots=True)
class RewardPolicyBinding:
    """One total reward rule and the exact semantic identity it publishes.

    ``failure_score`` is the preregistered endpoint value for an invocation
    that cannot publish a scored candidate (model/schema, treatment,
    materialization, candidate-boundary, or infrastructure failure).  Keeping
    it on the binding prevents the engine from silently mixing a generic
    sentinel with a benchmark-owned absolute endpoint.
    """

    score: RewardPolicy
    definition_hash: str
    failure_score: float = -1.0

    def __post_init__(self) -> None:
        if not callable(self.score):
            raise TypeError("score must be callable")
        require_sha256(self.definition_hash, "definition_hash")
        if type(self.failure_score) is not float or not math.isfinite(
            self.failure_score
        ):
            raise TypeError("failure_score must be a finite canonical float")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "definition_hash": self.definition_hash,
            "failure_score_hex": self.failure_score.hex(),
        }

    @property
    def binding_sha256(self) -> str:
        return hashlib.sha256(
            _REWARD_BINDING_DOMAIN
            + json.dumps(
                self.to_record(),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


class OperatorKind(str, Enum):
    REPRODUCTION = "reproduction"
    TYPED_MUTATION = "typed_mutation"
    TWO_PARENT_CROSSOVER = "two_parent_crossover"
    THREE_WAY_RECOMBINATION = "three_way_recombination"
    REPAIR = "repair"

    @property
    def variation_kind(self) -> VariationKind:
        return VariationKind(self.value)


class MutationResponseMode(str, Enum):
    """Semantic representation requested from a typed-mutation model call."""

    FULL_CONFIGURATION = "full_configuration"
    ATOMIC_SCALAR_REPLACEMENT_V1 = "atomic_scalar_replacement_v1"
    FINITE_OPTION_SELECTION_V1 = "finite_option_selection_v1"


class CrossoverResponseMode(str, Enum):
    """Semantic representation requested from a two-parent crossover call."""

    FULL_CONFIGURATION = "full_configuration"
    EXACT_PARENT_IMPORT_V1 = "exact_parent_import_v1"


def _operator_version(response_mode: MutationResponseMode) -> int:
    if type(response_mode) is not MutationResponseMode:
        raise TypeError("response_mode must be a MutationResponseMode")
    return {
        MutationResponseMode.FULL_CONFIGURATION: 1,
        MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1: 2,
        MutationResponseMode.FINITE_OPTION_SELECTION_V1: 3,
    }[response_mode]


class InsightAssignmentKind(str, Enum):
    """Why exact insight versions were attached to an invocation."""

    RETRIEVAL = "retrieval"
    QUARANTINE_TEST = "quarantine_test"
    RESOLVED_CAUSAL = "resolved_causal"


class _TerminalDetailedEvaluationError(RuntimeError):
    """Typed infrastructure/system evidence that must not enter candidate state."""

    def __init__(self, evaluation: DetailedEvaluation) -> None:
        if type(evaluation) is not DetailedEvaluation:
            raise TypeError("evaluation must be an exact DetailedEvaluation")
        failure = evaluation.failure
        if failure is None or failure.category is FailureCategory.CANDIDATE:
            raise ValueError(
                "terminal evaluation requires infrastructure/system failure"
            )
        super().__init__("detailed evaluator returned terminal failure evidence")
        self.evaluation = evaluation


class _DetailedEvaluationPortError(RuntimeError):
    """Adapter exceptions/contract violations are infrastructure, never candidates."""


class _TreatmentCompliancePolicyError(RuntimeError):
    """A treatment policy defect is infrastructure, never model no-yield."""


class ProposalAuthority(str, Enum):
    """Which subsystem authored the candidate payload for an invocation."""

    MODEL = "model"
    ENGINE = "engine"
    REPRODUCTION = "reproduction"


@dataclass(frozen=True, slots=True)
class EvolutionCandidate:
    occurrence: CandidateOccurrence
    configuration: FrozenJsonValue
    objectives: tuple[tuple[str, float], ...]
    valid: bool
    generation: int
    label: str
    operator_kind: OperatorKind | None = None
    parent_ids: tuple[CandidateId, ...] = ()
    common_ancestor_id: CandidateId | None = None
    design_rationale: str = "seed"
    failure_message: str | None = None
    operator_compliant: bool = True
    operator_failure: str | None = None
    evidence_compliant: bool = True
    evidence_failure: str | None = None
    parent_patch_hashes: tuple[str, ...] = ()
    preservation_verified: bool | None = None
    claimed_insight_ids: tuple[str, ...] = ()
    selected_insight_ids: tuple[str, ...] = ()
    source_attribution: tuple[SourceAttribution, ...] = ()
    conflict_resolutions: tuple[ConflictResolutionDraft, ...] = ()
    call_telemetry: AgenticCallTelemetry | None = None
    selected_insight_refs: tuple[InsightRef, ...] = ()
    insight_assignment_kind: InsightAssignmentKind | None = None
    detailed_evaluation: DetailedEvaluation | None = None
    objective_resolution_receipt: ObjectiveResolutionReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.occurrence) is not CandidateOccurrence:
            raise TypeError("occurrence must be an exact CandidateOccurrence")
        CandidateOccurrence.__post_init__(self.occurrence)
        frozen = freeze_json(self.configuration)
        if frozen is not self.configuration:
            raise TypeError("configuration must already be frozen typed JSON")
        if typed_json_sha256(frozen) != self.occurrence.configuration_hash:
            raise ValueError("candidate configuration does not match occurrence hash")
        if (
            type(self.valid) is not bool
            or type(self.operator_compliant) is not bool
            or type(self.evidence_compliant) is not bool
        ):
            raise TypeError(
                "valid, operator_compliant, and evidence_compliant must be bool"
            )
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if type(self.label) is not str or not self.label:
            raise ValueError("label must be non-empty")
        if self.valid and not self.objectives:
            raise ValueError("valid candidates require a complete objective vector")
        if not self.valid and self.objectives:
            raise ValueError("invalid candidates cannot carry objectives")
        resolution_receipt = self.objective_resolution_receipt
        if resolution_receipt is not None:
            if type(resolution_receipt) is not ObjectiveResolutionReceipt:
                raise TypeError(
                    "objective_resolution_receipt must be an exact receipt or None"
                )
            resolution_receipt.revalidate()
            if not self.valid:
                raise ValueError(
                    "invalid candidates cannot carry an objective-resolution receipt"
                )
            if (
                resolution_receipt.configuration_sha256
                != self.occurrence.configuration_hash
            ):
                raise ValueError(
                    "objective-resolution receipt identifies another configuration"
                )
            if resolution_receipt.decision_objectives != self.objectives:
                raise ValueError(
                    "candidate objectives must equal resolved decision objectives"
                )
        if self.detailed_evaluation is not None:
            if type(self.detailed_evaluation) is not DetailedEvaluation:
                raise TypeError(
                    "detailed_evaluation must be an exact DetailedEvaluation or None"
                )
            DetailedEvaluation.__post_init__(self.detailed_evaluation)
            if self.detailed_evaluation.success != self.valid:
                raise ValueError(
                    "candidate validity must agree with detailed evaluation success"
                )
            expected_detailed_objectives = (
                self.objectives
                if resolution_receipt is None
                else resolution_receipt.raw_objectives
            )
            if self.detailed_evaluation.objectives != expected_detailed_objectives:
                raise ValueError(
                    "detailed evaluation must preserve the raw objective projection"
                )
            detailed_failure = self.detailed_evaluation.failure
            expected_message = (
                None if detailed_failure is None else detailed_failure.message
            )
            if self.failure_message != expected_message:
                raise ValueError(
                    "candidate failure message must agree with detailed evidence"
                )
        if self.operator_compliant != (self.operator_failure is None):
            raise ValueError(
                "operator_compliant must agree with operator_failure presence"
            )
        if self.evidence_compliant != (self.evidence_failure is None):
            raise ValueError(
                "evidence_compliant must agree with evidence_failure presence"
            )
        if type(self.source_attribution) is not tuple or any(
            type(item) is not SourceAttribution for item in self.source_attribution
        ):
            raise TypeError("source_attribution must contain exact values")
        if type(self.conflict_resolutions) is not tuple or any(
            type(item) is not ConflictResolutionDraft
            for item in self.conflict_resolutions
        ):
            raise TypeError("conflict_resolutions must contain exact values")
        if type(self.selected_insight_refs) is not tuple or any(
            type(reference) is not InsightRef
            for reference in self.selected_insight_refs
        ):
            raise TypeError(
                "selected_insight_refs must contain exact InsightRef values"
            )
        if len(set(self.selected_insight_refs)) != len(self.selected_insight_refs):
            raise ValueError("selected_insight_refs cannot contain duplicates")
        for reference in self.selected_insight_refs:
            InsightRef.__post_init__(reference)
        if self.selected_insight_refs and self.selected_insight_ids != tuple(
            reference.insight_id.value for reference in self.selected_insight_refs
        ):
            raise ValueError(
                "selected_insight_ids must agree with selected_insight_refs"
            )
        if (
            self.insight_assignment_kind is not None
            and type(self.insight_assignment_kind) is not InsightAssignmentKind
        ):
            raise TypeError("insight_assignment_kind must be an InsightAssignmentKind")
        if self.selected_insight_refs and self.insight_assignment_kind is None:
            raise ValueError(
                "selected insight references require an insight_assignment_kind"
            )

    @property
    def candidate_id(self) -> CandidateId:
        return self.occurrence.candidate_id

    @property
    def configuration_dict(self) -> dict[str, Any]:
        value = thaw_json(self.configuration)
        if type(value) is not dict:  # pragma: no cover - candidate root contract.
            raise TypeError("candidate configuration root must be an object")
        return value

    @property
    def objective_map(self) -> dict[str, float]:
        return dict(self.objectives)

    @property
    def raw_objective_map(self) -> dict[str, float]:
        receipt = self.objective_resolution_receipt
        return dict(self.objectives if receipt is None else receipt.raw_objectives)


@dataclass(frozen=True, slots=True)
class MutationContract:
    """Replayable machine boundary for one focused typed mutation.

    ``editable_paths`` names the only exact typed-patch operation paths that a
    child may change.  A non-abstaining contract therefore permits between one
    and ``max_changed_paths`` distinct paths and at most ``max_operations``
    patch operations.  Setting both limits to one is an exact one-coordinate
    intervention.  ``allow_abstention`` explicitly admits the unchanged parent;
    it never permits a change outside the declared paths.
    """

    editable_paths: tuple[JsonPath, ...]
    max_changed_paths: int = 1
    max_operations: int = 1
    allow_abstention: bool = False

    def __post_init__(self) -> None:
        if type(self.editable_paths) is not tuple or not self.editable_paths:
            raise ValueError("editable_paths must be a non-empty exact tuple")
        path_keys: set[bytes] = set()
        for path in self.editable_paths:
            if type(path) is not JsonPath:
                raise TypeError("editable_paths must contain exact JsonPath values")
            validate_json_path(path)
            if not path.segments:
                raise ValueError("the candidate root cannot be an editable path")
            if type(path.segments[0]) is not ObjectKey:
                raise ValueError("an editable path must begin with an object key")
            key = canonical_path_bytes(path)
            if key in path_keys:
                raise ValueError("editable_paths must not contain duplicates")
            path_keys.add(key)
        if type(self.max_changed_paths) is not int:
            raise TypeError("max_changed_paths must be an exact integer")
        if self.max_changed_paths <= 0:
            raise ValueError("max_changed_paths must be positive")
        if self.max_changed_paths > len(self.editable_paths):
            raise ValueError("max_changed_paths cannot exceed editable_paths")
        if type(self.max_operations) is not int:
            raise TypeError("max_operations must be an exact integer")
        if self.max_operations <= 0:
            raise ValueError("max_operations must be positive")
        if type(self.allow_abstention) is not bool:
            raise TypeError("allow_abstention must be bool")


def _validate_finite_option_patch_scope(
    operations: tuple[PatchOperation, ...],
    *,
    allowed_top_level: tuple[str, ...],
    mutation_contract: MutationContract,
) -> None:
    """Fail closed when a sealed full child escapes its mutation boundary."""

    if type(operations) is not tuple or not operations:
        raise ValueError("a finite variation option must change its parent")
    allowed = set(allowed_top_level)
    for operation in operations:
        if (
            not operation.path.segments
            or type(operation.path.segments[0]) is not ObjectKey
        ):
            raise ValueError("finite variation option changed the candidate root")
        if operation.path.segments[0].value not in allowed:
            raise ValueError(
                "finite variation option escaped its declared top-level scope"
            )
    editable = set(mutation_contract.editable_paths)
    changed_paths = {operation.path for operation in operations}
    if any(path not in editable for path in changed_paths):
        raise ValueError(
            "finite variation option changed a path outside its machine contract"
        )
    if len(changed_paths) > mutation_contract.max_changed_paths:
        raise ValueError(
            "finite variation option exceeded its changed-path cardinality"
        )
    if len(operations) > mutation_contract.max_operations:
        raise ValueError(
            "finite variation option exceeded its patch-operation cardinality"
        )


@dataclass(frozen=True, slots=True)
class InvocationPlan:
    operator_kind: OperatorKind
    parents: tuple[EvolutionCandidate, ...]
    generation: int
    label: str
    common_ancestor: EvolutionCandidate | None = None
    allowed_top_level: tuple[str, ...] = ()
    use_memory: bool = False
    memory_subset_size: int = 2
    memory_exploration_probability: Fraction | None = None
    memory_score_phase: str | None = None
    phase: str = "adaptation"
    mutation_contract: MutationContract | None = None
    mutation_response_mode: MutationResponseMode = (
        MutationResponseMode.FULL_CONFIGURATION
    )
    atomic_replacement_options: tuple[FrozenJsonValue, ...] = ()
    finite_variation_contract: FiniteVariationContract | None = None
    crossover_response_mode: CrossoverResponseMode = (
        CrossoverResponseMode.FULL_CONFIGURATION
    )
    exact_parent_crossover_contract: ExactParentCrossoverContract | None = None
    forbidden_exact_parent_import_sets: tuple[tuple[str, ...], ...] = ()
    quarantine_test_insights: tuple[InsightRef, ...] = ()
    resolved_insight_assignment: ResolvedInsightAssignment | None = None
    insight_treatment_requirement: InsightTreatmentRequirement | None = None
    compiled_hypothesis_treatment: CompiledHypothesisTreatment | None = None
    compiled_hypothesis_eligibility: tuple[CompiledHypothesisTreatment, ...] = ()
    finite_action_set_authority: FiniteActionSetAuthority | None = None

    def __post_init__(self) -> None:
        if type(self.operator_kind) is not OperatorKind:
            raise TypeError("operator_kind must be an OperatorKind")
        if type(self.parents) is not tuple or any(
            type(parent) is not EvolutionCandidate for parent in self.parents
        ):
            raise TypeError("parents must contain exact EvolutionCandidate values")
        required = (
            2
            if self.operator_kind
            in {
                OperatorKind.TWO_PARENT_CROSSOVER,
                OperatorKind.THREE_WAY_RECOMBINATION,
            }
            else 1
        )
        if len(self.parents) != required:
            raise ValueError(
                f"{self.operator_kind.value} requires {required} parent(s)"
            )
        if len({parent.candidate_id for parent in self.parents}) != len(self.parents):
            raise ValueError("variation parents must be distinct occurrences")
        if self.operator_kind is OperatorKind.THREE_WAY_RECOMBINATION:
            if type(self.common_ancestor) is not EvolutionCandidate:
                raise ValueError("three-way recombination requires a common ancestor")
        elif self.common_ancestor is not None:
            raise ValueError("only three-way recombination accepts a common ancestor")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("invocation generation must be positive")
        if type(self.label) is not str or not self.label:
            raise ValueError("invocation label must be non-empty")
        if type(self.allowed_top_level) is not tuple or any(
            type(value) is not str or not value for value in self.allowed_top_level
        ):
            raise TypeError("allowed_top_level must be a tuple of non-empty strings")
        if (
            self.operator_kind is OperatorKind.TYPED_MUTATION
            and not self.allowed_top_level
        ):
            raise ValueError("typed mutation requires an explicit top-level scope")
        if self.mutation_contract is not None:
            if type(self.mutation_contract) is not MutationContract:
                raise TypeError("mutation_contract must be an exact MutationContract")
            MutationContract.__post_init__(self.mutation_contract)
            if self.operator_kind is not OperatorKind.TYPED_MUTATION:
                raise ValueError("only typed mutation accepts a mutation_contract")
            allowed = set(self.allowed_top_level)
            for path in self.mutation_contract.editable_paths:
                first = path.segments[0]
                assert type(first) is ObjectKey
                if first.value not in allowed:
                    raise ValueError(
                        "mutation_contract editable path escapes allowed_top_level"
                    )
        if type(self.mutation_response_mode) is not MutationResponseMode:
            raise TypeError("mutation_response_mode must be a MutationResponseMode")
        if self.finite_variation_contract is not None:
            validate_finite_variation_contract(self.finite_variation_contract)
        if type(self.atomic_replacement_options) is not tuple:
            raise TypeError("atomic_replacement_options must be an exact tuple")
        option_hashes: list[str] = []
        for option in self.atomic_replacement_options:
            if freeze_json(option) is not option or not is_json_scalar(option):
                raise TypeError(
                    "atomic_replacement_options must contain frozen JSON scalars"
                )
            option_hashes.append(typed_json_sha256(option))
        if len(set(option_hashes)) != len(option_hashes):
            raise ValueError("atomic_replacement_options cannot contain duplicates")
        if (
            self.mutation_response_mode
            is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
        ):
            if self.finite_variation_contract is not None:
                raise ValueError(
                    "atomic scalar replacement cannot bind a finite variation contract"
                )
            if self.operator_kind is not OperatorKind.TYPED_MUTATION:
                raise ValueError(
                    "atomic scalar replacement is restricted to typed mutation"
                )
            contract = self.mutation_contract
            if contract is None:
                raise ValueError(
                    "atomic scalar replacement requires a mutation_contract"
                )
            if (
                len(contract.editable_paths) != 1
                or contract.max_changed_paths != 1
                or contract.max_operations != 1
                or contract.allow_abstention
            ):
                raise ValueError(
                    "atomic scalar replacement requires exactly one editable "
                    "path, one changed path, one operation, and no abstention"
                )
            current = value_at_path(
                self.parents[0].configuration,
                contract.editable_paths[0],
            )
            if not is_json_scalar(current):
                raise ValueError(
                    "atomic scalar replacement path must resolve to a scalar"
                )
            if any(
                typed_json_equal(current, option)
                for option in self.atomic_replacement_options
            ):
                raise ValueError(
                    "atomic_replacement_options must exclude the parent value"
                )
        elif (
            self.mutation_response_mode
            is MutationResponseMode.FINITE_OPTION_SELECTION_V1
        ):
            if self.operator_kind is not OperatorKind.TYPED_MUTATION:
                raise ValueError(
                    "finite option selection is restricted to typed mutation"
                )
            if self.atomic_replacement_options:
                raise ValueError(
                    "finite option selection cannot bind atomic replacement options"
                )
            finite_contract = self.finite_variation_contract
            if finite_contract is None:
                raise ValueError(
                    "finite option selection requires a finite variation contract"
                )
            mutation_contract = self.mutation_contract
            if mutation_contract is None:
                raise ValueError("finite option selection requires a mutation contract")
            parent = self.parents[0]
            if type(parent.configuration) is not FrozenJsonObject:
                raise TypeError("finite option parent must be a FrozenJsonObject")
            if not typed_json_equal(
                finite_contract.parent_configuration,
                parent.configuration,
            ):
                raise ValueError(
                    "finite variation contract is bound to a different parent"
                )
            probe_target = CandidateId("candidate_finite_contract_probe")
            if probe_target == parent.candidate_id:
                probe_target = CandidateId("candidate_finite_contract_probe_alternate")
            for finite_option in finite_contract.options:
                probe_patch = derive_patch(
                    parent.configuration,
                    finite_option.child_configuration,
                    base_candidate_id=parent.candidate_id,
                    target_candidate_id=probe_target,
                )
                _validate_finite_option_patch_scope(
                    probe_patch.operations,
                    allowed_top_level=self.allowed_top_level,
                    mutation_contract=mutation_contract,
                )
        else:
            if self.atomic_replacement_options:
                raise ValueError(
                    "atomic_replacement_options require atomic scalar replacement"
                )
            if self.finite_variation_contract is not None:
                raise ValueError(
                    "finite_variation_contract requires finite option selection"
                )
        if type(self.crossover_response_mode) is not CrossoverResponseMode:
            raise TypeError("crossover_response_mode must be a CrossoverResponseMode")
        if type(self.forbidden_exact_parent_import_sets) is not tuple or any(
            type(value) is not tuple
            for value in self.forbidden_exact_parent_import_sets
        ):
            raise TypeError(
                "forbidden_exact_parent_import_sets must be an exact tuple of tuples"
            )
        crossover_contract = self.exact_parent_crossover_contract
        if self.crossover_response_mode is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1:
            if self.operator_kind is not OperatorKind.TWO_PARENT_CROSSOVER:
                raise ValueError(
                    "exact parent import is restricted to two-parent crossover"
                )
            if type(crossover_contract) is not ExactParentCrossoverContract:
                raise TypeError(
                    "exact parent import requires an exact crossover contract"
                )
            ExactParentCrossoverContract.__post_init__(crossover_contract)
            left, right = self.parents
            if (
                crossover_contract.base_parent_sha256
                != left.occurrence.configuration_hash
                or crossover_contract.donor_parent_sha256
                != right.occurrence.configuration_hash
            ):
                raise ValueError(
                    "exact crossover contract is bound to different parents"
                )
            expected_crossover_contract = derive_exact_parent_crossover_contract(
                base=left.configuration,
                donor=right.configuration,
                max_loci=crossover_contract.max_loci,
            )
            if (
                expected_crossover_contract.to_record()
                != crossover_contract.to_record()
            ):
                raise ValueError(
                    "exact crossover contract differs from its ordered parents"
                )
            validate_exact_parent_import_exclusions(
                crossover_contract,
                self.forbidden_exact_parent_import_sets,
            )
        elif crossover_contract is not None:
            raise ValueError(
                "exact_parent_crossover_contract requires exact parent import"
            )
        elif (
            self.operator_kind is not OperatorKind.TWO_PARENT_CROSSOVER
            and self.crossover_response_mode
            is not CrossoverResponseMode.FULL_CONFIGURATION
        ):
            raise ValueError(
                "non-crossover plans cannot select a crossover response mode"
            )
        elif self.forbidden_exact_parent_import_sets:
            raise ValueError(
                "forbidden exact parent imports require exact parent import mode"
            )
        if type(self.use_memory) is not bool:
            raise TypeError("use_memory must be bool")
        if type(self.quarantine_test_insights) is not tuple or any(
            type(reference) is not InsightRef
            for reference in self.quarantine_test_insights
        ):
            raise TypeError(
                "quarantine_test_insights must contain exact InsightRef values"
            )
        if len(set(self.quarantine_test_insights)) != len(
            self.quarantine_test_insights
        ):
            raise ValueError("quarantine_test_insights cannot contain duplicates")
        for reference in self.quarantine_test_insights:
            InsightRef.__post_init__(reference)
        if self.quarantine_test_insights and self.use_memory:
            raise ValueError(
                "quarantine_test_insights and normal memory retrieval are "
                "mutually exclusive"
            )
        if self.resolved_insight_assignment is not None:
            if type(self.resolved_insight_assignment) is not ResolvedInsightAssignment:
                raise TypeError(
                    "resolved_insight_assignment must be an exact "
                    "ResolvedInsightAssignment"
                )
            ResolvedInsightAssignment.__post_init__(self.resolved_insight_assignment)
            if self.use_memory or self.quarantine_test_insights:
                raise ValueError(
                    "resolved causal assignment and legacy memory assignment are "
                    "mutually exclusive"
                )
            if self.operator_kind is OperatorKind.REPRODUCTION:
                raise ValueError(
                    "resolved causal memory requires a model-authored proposal"
                )
        if self.insight_treatment_requirement is not None:
            requirement = self.insight_treatment_requirement
            if type(requirement) is not InsightTreatmentRequirement:
                raise TypeError(
                    "insight_treatment_requirement must be exact when supplied"
                )
            InsightTreatmentRequirement.__post_init__(requirement)
            if (
                self.mutation_response_mode
                is not MutationResponseMode.FINITE_OPTION_SELECTION_V1
            ):
                raise ValueError(
                    "insight treatment administration requires finite option selection"
                )
            if self.quarantine_test_insights:
                assigned = self.quarantine_test_insights
            elif self.resolved_insight_assignment is not None:
                assigned = self.resolved_insight_assignment.selection_decision.selected
            else:
                raise ValueError(
                    "insight treatment administration requires an explicit assignment"
                )
            if tuple(sorted(assigned)) != requirement.required_insights:
                raise ValueError(
                    "treatment required_insights differ from the plan assignment"
                )
            finite_contract = self.finite_variation_contract
            if finite_contract is None:  # pragma: no cover - mode validation above.
                raise ValueError("treatment requirement lost its finite contract")
            if requirement.finite_contract_sha256 != finite_contract.identity_sha256:
                raise ValueError(
                    "treatment requirement is bound to a different finite contract"
                )
            for action in requirement.allowed_actions:
                option = finite_contract.resolve(action.option_id)
                if option.identity_sha256 != action.option_identity_sha256:
                    raise ValueError(
                        "treatment action binding differs from the finite palette"
                    )
        compiled = self.compiled_hypothesis_treatment
        if compiled is not None:
            if type(compiled) is not CompiledHypothesisTreatment:
                raise TypeError(
                    "compiled_hypothesis_treatment must be exact when supplied"
                )
            CompiledHypothesisTreatment.__post_init__(compiled)
            requirement = self.insight_treatment_requirement
            if requirement is None or requirement != compiled.requirement:
                raise ValueError(
                    "compiled treatment must own the plan treatment requirement"
                )
            if compiled.request.requested_operator_kind != self.operator_kind.value:
                raise ValueError(
                    "compiled treatment is bound to a different operator kind"
                )
            if compiled.request.parent_candidate_id != self.parents[0].candidate_id:
                raise ValueError("compiled treatment is bound to a different parent")
            contract = self.finite_variation_contract
            if contract is None or (
                compiled.request.finite_contract.identity_sha256
                != contract.identity_sha256
            ):
                raise ValueError(
                    "compiled treatment is bound to a different finite contract"
                )
            assigned = (
                self.resolved_insight_assignment.selection_decision.selected
                if self.resolved_insight_assignment is not None
                else self.quarantine_test_insights
            )
            if assigned != (compiled.request.reference,):
                raise ValueError(
                    "compiled treatment differs from the exact plan assignment"
                )
        matrix = self.compiled_hypothesis_eligibility
        if type(matrix) is not tuple or any(
            type(value) is not CompiledHypothesisTreatment for value in matrix
        ):
            raise TypeError(
                "compiled_hypothesis_eligibility must contain exact bindings"
            )
        for value in matrix:
            CompiledHypothesisTreatment.__post_init__(value)
        matrix_refs = tuple(value.request.reference for value in matrix)
        if matrix_refs != tuple(sorted(set(matrix_refs))):
            raise ValueError(
                "compiled hypothesis eligibility must be unique and canonical"
            )
        if matrix:
            resolved = self.resolved_insight_assignment
            if resolved is None:
                raise ValueError(
                    "compiled hypothesis eligibility requires a resolved assignment"
                )
            if matrix_refs != resolved.selection_decision.eligible:
                raise ValueError(
                    "compiled eligibility refs differ from assignment eligibility"
                )
            if compiled is None or compiled not in matrix:
                raise ValueError(
                    "selected compiled treatment must be an eligibility member"
                )
            if resolved.selection_decision.selected != (compiled.request.reference,):
                raise ValueError(
                    "selected compilation differs from resolved selected insight"
                )
            shared = {
                (
                    value.request.parent_candidate_id,
                    value.request.parent_configuration_sha256,
                    value.request.finite_contract.identity_sha256,
                    value.request.context_projection_sha256,
                    value.request.endpoint_definition_sha256,
                    value.request.requested_operator_kind,
                )
                for value in matrix
            }
            if len(shared) != 1:
                raise ValueError("compiled eligibility matrix mixes execution contexts")
            if next(iter(shared)) != (
                self.parents[0].candidate_id,
                self.parents[0].occurrence.configuration_hash,
                self.finite_variation_contract.identity_sha256,
                resolved.exact_context_hash,
                compiled.request.endpoint_definition_sha256,
                self.operator_kind.value,
            ):
                raise ValueError(
                    "compiled eligibility matrix differs from invocation context"
                )
        elif compiled is not None and self.resolved_insight_assignment is not None:
            raise ValueError(
                "resolved compiled treatment requires a complete eligibility matrix"
            )
        finite_authority = self.finite_action_set_authority
        if finite_authority is not None:
            if type(finite_authority) is not FiniteActionSetAuthority:
                raise TypeError(
                    "finite_action_set_authority must be exact when supplied"
                )
            FiniteActionSetAuthority.__post_init__(finite_authority)
            if (
                self.mutation_response_mode
                is not MutationResponseMode.FINITE_OPTION_SELECTION_V1
            ):
                raise ValueError(
                    "finite action authority requires finite option selection"
                )
            finite_contract = self.finite_variation_contract
            if finite_contract != finite_authority.support.support_contract:
                raise ValueError(
                    "finite action authority differs from the plan support contract"
                )
            parent = self.parents[0]
            if (
                finite_authority.support.parent_candidate_id != parent.candidate_id
                or finite_authority.support.parent_configuration_sha256
                != parent.occurrence.configuration_hash
                or finite_authority.support.support_contract.parent_configuration
                != parent.configuration
            ):
                raise ValueError(
                    "finite action authority is bound to a different plan parent"
                )
            if (
                self.insight_treatment_requirement is not None
                or self.compiled_hypothesis_treatment is not None
                or self.compiled_hypothesis_eligibility
            ):
                raise ValueError(
                    "finite action authority is parallel to exact treatment contracts"
                )
            if self.use_memory:
                raise ValueError(
                    "finite action authority requires an explicitly resolved card"
                )
            if self.resolved_insight_assignment is not None:
                assigned = self.resolved_insight_assignment.selection_decision.selected
            else:
                assigned = self.quarantine_test_insights
            if assigned != (finite_authority.card.reference,):
                raise ValueError(
                    "finite action authority differs from its assigned exact card"
                )
        if type(self.memory_subset_size) is not int or self.memory_subset_size < 0:
            raise ValueError("memory_subset_size must be non-negative")
        if self.memory_exploration_probability is not None:
            if type(self.memory_exploration_probability) is not Fraction:
                raise TypeError(
                    "memory_exploration_probability must be an exact Fraction"
                )
            if not Fraction(0) <= self.memory_exploration_probability <= Fraction(1):
                raise ValueError("memory_exploration_probability must lie in [0,1]")
        if self.memory_score_phase is not None and (
            type(self.memory_score_phase) is not str or not self.memory_score_phase
        ):
            raise ValueError("memory_score_phase must be non-empty when supplied")
        if not self.use_memory and (
            self.memory_exploration_probability is not None
            or self.memory_score_phase is not None
        ):
            raise ValueError("memory policy overrides require use_memory=True")
        if type(self.phase) is not str or not self.phase:
            raise ValueError("phase must be non-empty")


def _plan_operator_version(plan: InvocationPlan) -> int:
    """Return the executable operator-contract version, not a model version."""

    if type(plan) is not InvocationPlan:
        raise TypeError("plan must be an exact InvocationPlan")
    if plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER:
        return {
            CrossoverResponseMode.FULL_CONFIGURATION: 1,
            CrossoverResponseMode.EXACT_PARENT_IMPORT_V1: 3,
        }[plan.crossover_response_mode]
    return _operator_version(plan.mutation_response_mode)


def _proposal_representation(plan: InvocationPlan) -> str:
    if type(plan) is not InvocationPlan:
        raise TypeError("plan must be an exact InvocationPlan")
    if plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER:
        return plan.crossover_response_mode.value
    return plan.mutation_response_mode.value


@dataclass(frozen=True, slots=True)
class PreparedInvocation:
    plan: InvocationPlan
    operator_invocation_id: OperatorInvocationId
    call_id: LLMCallId | None
    candidate_id: CandidateId
    proposal_sequence: int
    variation_case: VariationCase
    classification: ThreeWayPatchClassification | None
    selection_decision: InsightSelectionDecision | None
    proposal_authority: ProposalAuthority
    prompt: str = ""
    insight_assignment_kind: InsightAssignmentKind | None = None
    materialization_policy_id: str | None = None
    materialization_policy_version: int | None = None
    materialization_receipt_hash: str | None = None
    materialized_candidate_id: CandidateId | None = None
    treatment_preflight_receipt: TreatmentPreflightReceipt | None = None
    materialized_finite_action_authority: FiniteActionSetAuthority | None = None
    materialized_finite_action_decision: FiniteActionDecision | None = None


@dataclass(frozen=True, slots=True)
class MaterializedInvocation:
    """One engine-authored proposal with a replay-bound materialization receipt."""

    plan: InvocationPlan
    draft: CandidateDraft | AtomicMutationDraft
    candidate_id: CandidateId
    materialization_policy_id: str
    materialization_policy_version: int
    materialization_receipt_hash: str
    materialized_finite_action_authority: FiniteActionSetAuthority | None = None
    materialized_finite_action_decision: FiniteActionDecision | None = None

    def __post_init__(self) -> None:
        if type(self.plan) is not InvocationPlan:
            raise TypeError("plan must be an exact InvocationPlan")
        InvocationPlan.__post_init__(self.plan)
        if type(self.draft) not in {CandidateDraft, AtomicMutationDraft}:
            raise TypeError(
                "draft must be an exact CandidateDraft or AtomicMutationDraft"
            )
        self.draft.__post_init__()
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        if self.candidate_id in {
            parent.candidate_id for parent in self.plan.parents
        } or (
            self.plan.common_ancestor is not None
            and self.candidate_id == self.plan.common_ancestor.candidate_id
        ):
            raise ValueError(
                "materialized candidate ID must differ from parents and ancestor"
            )
        if (
            type(self.materialization_policy_id) is not str
            or not self.materialization_policy_id
            or self.materialization_policy_id != self.materialization_policy_id.strip()
        ):
            raise ValueError(
                "materialization_policy_id must be canonical non-empty text"
            )
        if (
            type(self.materialization_policy_version) is not int
            or self.materialization_policy_version <= 0
        ):
            raise ValueError("materialization_policy_version must be positive")
        require_sha256(
            self.materialization_receipt_hash,
            "materialization_receipt_hash",
        )
        if self.plan.operator_kind is OperatorKind.REPRODUCTION:
            raise ValueError("reproduction has its own proposal authority")
        if (
            self.plan.use_memory
            or self.plan.quarantine_test_insights
            or self.plan.resolved_insight_assignment is not None
            or self.plan.insight_treatment_requirement is not None
            or self.plan.compiled_hypothesis_treatment is not None
            or self.plan.compiled_hypothesis_eligibility
            or self.plan.finite_action_set_authority is not None
        ):
            raise ValueError(
                "engine-materialized invocations cannot receive prompt memory"
            )
        finite_authority = self.materialized_finite_action_authority
        finite_decision = self.materialized_finite_action_decision
        if (finite_authority is None) != (finite_decision is None):
            raise ValueError(
                "materialized finite action authority and decision must be paired"
            )
        if finite_authority is None:
            return
        if type(finite_authority) is not FiniteActionSetAuthority:
            raise TypeError("materialized_finite_action_authority must be exact")
        FiniteActionSetAuthority.__post_init__(finite_authority)
        if type(finite_decision) is not FiniteActionDecision:
            raise TypeError("materialized_finite_action_decision must be exact")
        validate_finite_action_decision(finite_authority, finite_decision)
        if finite_decision.selector_kind is not FiniteActionSelectorKind.ENGINE:
            raise ValueError(
                "materialized finite action provenance requires an engine selector"
            )
        if (
            self.plan.operator_kind is not OperatorKind.TYPED_MUTATION
            or len(self.plan.parents) != 1
            or type(self.draft) is not CandidateDraft
        ):
            raise ValueError(
                "materialized finite action provenance requires one "
                "typed-mutation parent"
            )
        parent = self.plan.parents[0]
        support = finite_authority.support
        if (
            support.parent_candidate_id != parent.candidate_id
            or support.parent_configuration_sha256
            != parent.occurrence.configuration_hash
            or not typed_json_equal(
                support.support_contract.parent_configuration,
                parent.configuration,
            )
        ):
            raise ValueError(
                "materialized finite action authority is bound to a different parent"
            )
        selected = support.options[finite_decision.selected_ordinal].option
        child = freeze_json(self.draft.configuration)
        if (
            not typed_json_equal(selected.child_configuration, child)
            or typed_json_sha256(child) != finite_decision.child_configuration_sha256
        ):
            raise ValueError(
                "materialized finite action decision is bound to a different child"
            )
        if (
            self.materialization_policy_id != finite_decision.selector_policy_id
            or self.materialization_policy_version
            != finite_decision.selector_policy_version
            or self.materialization_receipt_hash != finite_decision.decision_sha256
        ):
            raise ValueError(
                "materialization identity differs from its finite action decision"
            )


@dataclass(frozen=True, slots=True)
class InvocationOutcome:
    prepared: PreparedInvocation
    candidate: EvolutionCandidate | None
    reward: float
    call_failure_type: str | None = None
    failure_stage: str | None = None
    dominates_any_parent: bool = False
    better_than_any_parent: bool = False
    # Retained only when a physical evaluation exists but no candidate may be
    # published (for example, terminal evaluator evidence or reward failure).
    terminal_evaluation: DetailedEvaluation | None = None
    parent_relations: tuple[OutcomeRelation, ...] = ()
    treatment_admission_receipt: TreatmentAdmissionReceipt | None = None
    finite_action_decision: FiniteActionDecision | None = None

    def __post_init__(self) -> None:
        if type(self.prepared) is not PreparedInvocation:
            raise TypeError("prepared must be an exact PreparedInvocation")
        if (
            self.candidate is not None
            and type(self.candidate) is not EvolutionCandidate
        ):
            raise TypeError("candidate must be an exact EvolutionCandidate or None")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        if self.failure_stage not in {
            None,
            "llm",
            "materialization",
            "candidate",
            "treatment_noncompliance",
            "infrastructure",
        }:
            raise ValueError("failure_stage is not a supported terminal stage")
        if (self.failure_stage is None) != (self.call_failure_type is None):
            raise ValueError(
                "failure_stage and call_failure_type must be present together"
            )
        if self.call_failure_type is not None and (
            type(self.call_failure_type) is not str or not self.call_failure_type
        ):
            raise ValueError("call_failure_type must be non-empty when supplied")
        if self.failure_stage is None and self.candidate is None:
            raise ValueError("successful invocations must carry a candidate")
        if (
            self.failure_stage not in {None, "infrastructure"}
            and self.candidate is not None
        ):
            raise ValueError(
                "only successful or post-evaluation infrastructure outcomes may "
                "carry a candidate"
            )
        if (
            type(self.dominates_any_parent) is not bool
            or type(self.better_than_any_parent) is not bool
        ):
            raise TypeError("outcome comparison flags must be bool")
        if self.terminal_evaluation is not None:
            if type(self.terminal_evaluation) is not DetailedEvaluation:
                raise TypeError(
                    "terminal_evaluation must be an exact DetailedEvaluation or None"
                )
            DetailedEvaluation.__post_init__(self.terminal_evaluation)
        if self.candidate is not None and self.terminal_evaluation is not None:
            raise ValueError("published candidates already own their evaluation record")
        if type(self.parent_relations) is not tuple or any(
            type(relation) is not OutcomeRelation for relation in self.parent_relations
        ):
            raise TypeError(
                "parent_relations must contain exact OutcomeRelation values"
            )
        receipt = self.treatment_admission_receipt
        requirement = self.prepared.plan.insight_treatment_requirement
        if receipt is not None:
            if type(receipt) is not TreatmentAdmissionReceipt:
                raise TypeError(
                    "treatment_admission_receipt must be exact when supplied"
                )
            TreatmentAdmissionReceipt.__post_init__(receipt)
            preflight = self.prepared.treatment_preflight_receipt
            if requirement is None or preflight is None:
                raise ValueError(
                    "treatment admission receipt requires a prepared treatment"
                )
            if receipt.preflight_receipt_sha256 != preflight.receipt_sha256:
                raise ValueError(
                    "treatment admission receipt differs from prepared preflight"
                )
        if self.failure_stage == "treatment_noncompliance":
            if receipt is None or receipt.passed:
                raise ValueError(
                    "treatment_noncompliance requires a failed admission receipt"
                )
            if self.terminal_evaluation is not None:
                raise ValueError(
                    "treatment noncompliance cannot carry an evaluator result"
                )
        elif receipt is not None and not receipt.passed:
            raise ValueError(
                "a failed treatment receipt requires treatment_noncompliance"
            )
        if self.failure_stage is None and requirement is not None:
            if receipt is None or not receipt.passed:
                raise ValueError(
                    "successful assigned treatment requires a passing receipt"
                )
        if self.parent_relations and len(self.parent_relations) != len(
            self.prepared.plan.parents
        ):
            raise ValueError("parent_relations must align with prepared parent order")
        finite_authority = self.prepared.plan.finite_action_set_authority
        finite_decision = self.finite_action_decision
        if finite_decision is not None:
            if finite_authority is None:
                raise ValueError(
                    "finite action decision requires a prepared finite authority"
                )
            validate_finite_action_decision(finite_authority, finite_decision)
            if (
                self.prepared.proposal_authority is not ProposalAuthority.MODEL
                or self.prepared.call_id is None
                or finite_decision.model_call_id != self.prepared.call_id
            ):
                raise ValueError(
                    "finite action decision differs from its model invocation"
                )
        if finite_authority is not None:
            if self.failure_stage == "llm":
                if finite_decision is not None:
                    raise ValueError("failed model calls cannot publish a decision")
            elif self.failure_stage is None and finite_decision is None:
                raise ValueError(
                    "successful finite-action outcomes require a sealed decision"
                )

    @property
    def detailed_evaluation(self) -> DetailedEvaluation | None:
        """Expose one record without duplicating candidate-owned evidence."""

        if self.candidate is not None:
            return self.candidate.detailed_evaluation
        return self.terminal_evaluation


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
    return "".join(parts)


def _mutation_contract_record(
    contract: MutationContract | None,
) -> dict[str, object] | None:
    """Return the complete versioned mutation boundary for trace replay."""

    if contract is None:
        return None
    MutationContract.__post_init__(contract)
    return {
        "contract_version": 1,
        "editable_paths": [_path_text(path) for path in contract.editable_paths],
        "max_changed_paths": contract.max_changed_paths,
        "max_operations": contract.max_operations,
        "allow_abstention": contract.allow_abstention,
    }


def _claim_covers_path(claim: str, actual: str) -> bool:
    """Whether a model-visible JSON path is an exact/ancestor path claim."""

    return claim != "$" and (
        actual == claim
        or actual.startswith(claim + ".")
        or actual.startswith(claim + "[")
    )


def _validate_source_claims(
    claims: tuple[SourceAttribution, ...],
    *,
    required: Mapping[str, set[str]],
) -> tuple[bool, str | None]:
    """Bind claimed parent sources to machine-derived contribution paths."""

    for source, actual_paths in required.items():
        source_claims = tuple(item.path for item in claims if item.source == source)
        if not actual_paths or not source_claims:
            return False, f"missing verified {source} source attribution"
        if not any(
            _claim_covers_path(claim, actual)
            for claim in source_claims
            for actual in actual_paths
        ):
            return False, f"{source} attribution names no verified contribution"
        if any(
            not any(_claim_covers_path(claim, actual) for actual in actual_paths)
            for claim in source_claims
        ):
            return False, f"{source} attribution contains an unsupported path"
    return True, None


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _selection_decision_record(
    decision: InsightSelectionDecision | None,
) -> dict[str, object] | None:
    """Serialize the complete immutable retrieval decision for trace replay."""

    if decision is None:
        return None

    def reference_record(reference: InsightRef) -> dict[str, object]:
        return {
            "insight_id": reference.insight_id.value,
            "version": reference.version,
        }

    return {
        "context_hash": decision.context_hash,
        "eligible": [reference_record(reference) for reference in decision.eligible],
        "selected": [reference_record(reference) for reference in decision.selected],
        "exploitation_subset": [
            reference_record(reference) for reference in decision.exploitation_subset
        ],
        "score_snapshot": [
            {
                **reference_record(reference),
                "score": score,
            }
            for reference, score in decision.score_snapshot
        ],
        "subset_size": decision.subset_size,
        "exploration_probability": _fraction_record(decision.exploration_probability),
        "mode": decision.mode.value,
        "selected_subset_probability": _fraction_record(
            decision.selected_subset_probability
        ),
        "policy_id": decision.policy_id,
        "policy_version": decision.policy_version,
    }


def _insight_reference_records(
    references: Sequence[InsightRef],
) -> list[dict[str, object]]:
    return [
        {
            "insight_id": reference.insight_id.value,
            "version": reference.version,
        }
        for reference in references
    ]


def _candidate_evidence(candidate: EvolutionCandidate) -> dict[str, object]:
    evidence: dict[str, object] = {
        "candidate_id": candidate.candidate_id.value,
        "configuration": candidate.configuration_dict,
        "objectives": candidate.objective_map,
        "valid": candidate.valid,
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "evidence_failure": candidate.evidence_failure,
        "failure": candidate.failure_message,
    }
    if candidate.detailed_evaluation is not None:
        evidence["detailed_evaluation"] = candidate.detailed_evaluation.to_record()
    if candidate.objective_resolution_receipt is not None:
        evidence["objective_resolution"] = (
            candidate.objective_resolution_receipt.to_record()
        )
    if candidate.insight_assignment_kind is InsightAssignmentKind.QUARANTINE_TEST:
        evidence.update(
            {
                "selected_insights": _insight_reference_records(
                    candidate.selected_insight_refs
                ),
                "assignment_kind": candidate.insight_assignment_kind.value,
            }
        )
    return evidence


def _reflection_operation_projection(
    operation: PatchOperation,
) -> dict[str, object]:
    """Expose machine-derived edit facts without asking a model to diff configs."""

    # ``operation_kind`` is itself a closed domain boundary and rejects values
    # outside PatchOperation before this projection can enter a durable prompt.
    row: dict[str, object] = {
        "operation_kind": operation_kind(operation),
        "path": _path_text(operation.path),
    }
    if type(operation) is ReplaceScalar:
        row.update(
            {
                "old_value": thaw_json(operation.old_value),
                "new_value": thaw_json(operation.new_value),
                "old_value_hash": typed_json_sha256(operation.old_value),
                "new_value_hash": typed_json_sha256(operation.new_value),
            }
        )
    return row


def _finite_action_evidence_for_citations(
    cited_contrast_ids: tuple[str, ...],
    contrast_action_bindings: Mapping[str, FiniteActionEvidenceBinding],
) -> tuple[FiniteActionEvidenceBinding, ...]:
    """Project only engine-derived action bindings without allowing permutation."""

    bound_actions: list[FiniteActionEvidenceBinding] = []
    for contrast_id in cited_contrast_ids:
        binding = contrast_action_bindings.get(contrast_id)
        if binding is None:
            continue
        if type(binding) is not FiniteActionEvidenceBinding:
            raise ReflectionCardContractError(
                "reflection received an untyped finite action binding"
            )
        try:
            FiniteActionEvidenceBinding.__post_init__(binding)
        except (TypeError, ValueError) as exc:
            raise ReflectionCardContractError(
                "reflection received an invalid finite action binding"
            ) from exc
        if binding.contrast_id != contrast_id:
            raise ReflectionCardContractError(
                "finite action evidence is bound to a different contrast"
            )
        bound_actions.append(binding)
    return tuple(bound_actions)


def _validate_reflected_action_origin(
    draft: InsightDraft,
    contract: ReflectionInsightContract | None,
    cited_contrast_ids: tuple[str, ...],
    contrast_action_bindings: Mapping[str, FiniteActionEvidenceBinding],
) -> None:
    """Bind an exact-action card to the finite actions that produced its evidence."""

    if contract is None or not contract.allowed_option_ids:
        return
    bound_actions = _finite_action_evidence_for_citations(
        cited_contrast_ids,
        contrast_action_bindings,
    )
    if len(bound_actions) != len(cited_contrast_ids):
        raise ReflectionCardContractError(
            "exact-action reflection cited evidence without a finite option"
        )
    executed_option_ids = tuple(
        sorted({binding.option_id for binding in bound_actions})
    )
    if draft.recommended_option_ids != executed_option_ids:
        raise ReflectionCardContractError(
            "recommended option IDs differ from the cited executed action"
        )


def default_evidence_prompt(
    problem_description: str,
    prepared: PreparedInvocation,
    selected_insights: tuple[dict[str, object], ...],
) -> str:
    """Generic inspectable prompt policy; replace or wrap at composition time."""

    plan = prepared.plan
    sections = [
        "You are an explicit evolutionary variation operator, not a fresh sampler.",
        f"OPERATOR: {plan.operator_kind.value}",
    ]
    if (
        plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER
        and plan.crossover_response_mode is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
    ):
        crossover_contract = plan.exact_parent_crossover_contract
        if type(crossover_contract) is not ExactParentCrossoverContract:
            raise ValueError("exact crossover mode lost its sealed contract")
        sections.extend(
            [
                "Select a proper nonempty subset of donor-parent loci; do not "
                "author or return candidate JSON. Emit only import_locus_ids "
                "and honestly claimed insight IDs. Do not emit a rationale, "
                "configuration, changed paths, or source attribution. The "
                "engine starts from the exact left/base parent, imports the "
                "selected exact right/donor subtrees, validates the candidate, "
                "and derives exhaustive provenance.",
                "",
                "EXACT PARENT IMPORT CONTRACT",
                _json(
                    {
                        "contract_sha256": crossover_contract.contract_sha256,
                        "base_parent_candidate_id": (
                            plan.parents[0].candidate_id.value
                        ),
                        "base_parent_configuration_sha256": (
                            crossover_contract.base_parent_sha256
                        ),
                        "donor_parent_candidate_id": (
                            plan.parents[1].candidate_id.value
                        ),
                        "donor_parent_configuration_sha256": (
                            crossover_contract.donor_parent_sha256
                        ),
                        "locus_count": len(crossover_contract.loci),
                        "ordered_loci": [
                            {
                                "locus_id": locus.locus_id,
                                "path": locus.path_text,
                            }
                            for locus in crossover_contract.loci
                        ],
                        "forbidden_import_locus_sets": [
                            list(value)
                            for value in plan.forbidden_exact_parent_import_sets
                        ],
                    }
                ),
                "Choose at least one but not all locus IDs. Selected loci "
                "come exactly from the right/donor parent; every omitted locus "
                "remains exactly from the left/base parent. This proper-subset "
                "rule guarantees a discriminating contribution from both "
                "ordered parents. Never choose any forbidden_import_locus_sets "
                "entry: each is machine-proven to materialize an already known "
                "child, so it is not a novel crossover action.",
            ]
        )
    elif (
        plan.mutation_response_mode is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
    ):
        sections.extend(
            [
                "Return one atomic edit, not a complete candidate. Emit the exact contracted path, one replacement scalar, a concise design_rationale (not hidden chain-of-thought), and honestly report used insight IDs. "
                "The system will copy every other value from the immutable parent and derive source attribution itself. Emit only fields present in the supplied output schema.",
            ]
        )
        if plan.atomic_replacement_options:
            sections.extend(
                [
                    "ORDERED LEGAL REPLACEMENT OPTIONS",
                    _json(
                        [
                            thaw_json(option)
                            for option in plan.atomic_replacement_options
                        ]
                    ),
                    "Choose exactly one listed value. The order is a task-keyed presentation rotation, not a preference ranking.",
                    "For this call, this list is the only legal replacement catalog. Any general domain catalog below is background context, not an output catalog.",
                    "Use one short sentence for design_rationale and emit no extra fields.",
                ]
            )
    elif plan.mutation_response_mode is MutationResponseMode.FINITE_OPTION_SELECTION_V1:
        finite_contract = plan.finite_variation_contract
        if finite_contract is None:  # pragma: no cover - plan admission.
            raise ValueError("finite option mode requires a sealed contract")
        sections.extend(
            [
                "Select one immutable variation option; do not author or return a candidate configuration. "
                "Emit only option_id, one concise design_rationale (not hidden chain-of-thought), and honestly report used insight IDs. "
                "The engine will materialize the option's presealed full child, derive and replay its parent-relative typed patch, and generate source attribution.",
                "",
                "FINITE VARIATION CONTRACT",
                _json(
                    {
                        "catalog_id": finite_contract.catalog_id,
                        "catalog_version": finite_contract.catalog_version,
                        "catalog_definition_sha256": (
                            finite_contract.catalog_definition_sha256
                        ),
                        "parent_configuration_sha256": (
                            finite_contract.parent_configuration_sha256
                        ),
                        "contract_identity_sha256": finite_contract.identity_sha256,
                    }
                ),
                "ORDERED FINITE VARIATION OPTIONS",
                _json(list(finite_contract.prompt_records())),
                "Choose exactly one listed option_id. The order is presentation order, not a preference ranking. "
                "The sealed child configurations are engine-owned and intentionally absent from the output schema.",
            ]
        )
        finite_authority = plan.finite_action_set_authority
        if finite_authority is not None:
            sections.extend(
                [
                    "",
                    "MATCHED FINITE ACTION SET AUTHORITY",
                    _json(
                        {
                            "authority_sha256": finite_authority.authority_sha256,
                            "support_sha256": (finite_authority.support.support_sha256),
                            "card_authority_sha256": (
                                finite_authority.card.card_authority_sha256
                            ),
                            "card_reference": {
                                "insight_id": (
                                    finite_authority.card.reference.insight_id.value
                                ),
                                "version": finite_authority.card.reference.version,
                            },
                            "semantic_anchor_option_id": (
                                finite_authority.support.anchor_option_id
                            ),
                            "cardinality": finite_authority.support.cardinality,
                            "presentation_sha256": (
                                finite_authority.support.presentation.presentation_sha256
                            ),
                            "current_outcome_access": False,
                        }
                    ),
                    "The semantic anchor identifies the exact historical action "
                    "that defined this local neighbourhood; it is not the only "
                    "legal choice. Select whichever of all K listed options best "
                    "instantiates the assigned hypothesis on the current parent. "
                    "This same sealed support is also given to a prospective "
                    "engine comparator.",
                ]
            )
    else:
        sections.extend(
            [
                "Return exactly one typed candidate. Give a concise design_rationale (not hidden chain-of-thought), "
                "list intended changed paths, and honestly report source attribution at the most specific useful JSON paths and used insight IDs. "
                "Emit only fields present in the supplied output schema; follow any operator-specific evidence instructions below.",
                "",
                "CANDIDATE COMPONENT PATH CONTRACT",
                CANDIDATE_COMPONENT_PATH_CONTRACT,
                (
                    "Every source_attribution item has exactly two fields: one "
                    "canonical path beginning with $. and one source token from "
                    + (
                        "left, right, or synthesized. "
                        if plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER
                        else "ancestor, left, right, synthesized, or mutation. "
                    )
                    + "Use one path per item; do not emit candidate IDs, path "
                    "arrays, descriptions, or prose in this field. Attribute "
                    "only values that actually came from that source."
                ),
            ]
        )
    sections.extend(
        [
            "",
            "PROBLEM",
            problem_description,
            "",
            "PARENTS",
            _json([_candidate_evidence(parent) for parent in plan.parents]),
        ]
    )
    requirement = plan.insight_treatment_requirement
    if selected_insights and requirement is not None:
        sections.extend(
            [
                "",
                "ASSIGNED TREATMENT HYPOTHESES",
                "These exact card versions form one enforced finite-action "
                "treatment. They are experimental instructions, not established "
                "facts. Instantiate every assigned card only through a compatible "
                "sealed action and claim every required exact insight ID.",
                _json(list(selected_insights)),
            ]
        )
    elif (
        selected_insights
        and prepared.insight_assignment_kind is InsightAssignmentKind.QUARANTINE_TEST
    ):
        sections.extend(
            [
                "",
                "QUARANTINED TEST HYPOTHESES",
                (
                    "These exact versions are assigned as an enforced isolated "
                    "transfer test. They remain quarantined and are not established "
                    "memory. The origin trigger scopes the evidence that produced "
                    "the card; on this preflight-compatible parent you must choose "
                    "a compatible action influenced by every assigned card and "
                    "claim its exact insight ID to record treatment administration."
                    if plan.insight_treatment_requirement is not None
                    and plan.insight_treatment_requirement.claim_mode.value
                    == "exact_required"
                    else (
                        "These exact versions are assigned only for an isolated "
                        "test. They remain quarantined and are not established "
                        "memory. Use only when their trigger applies, and do not "
                        "claim use unless a hypothesis affected the candidate."
                    )
                ),
                _json(list(selected_insights)),
            ]
        )
    elif selected_insights:
        sections.extend(
            [
                "",
                "SELECTED MEMORY HYPOTHESES",
                "Use only when their trigger applies. A selected insight may be wrong; do not claim use unless it affected the candidate. claimed_insight_ids may contain only exact IDs from this selected set; use [] when none affected the candidate.",
                _json(list(selected_insights)),
            ]
        )
    else:
        sections.extend(
            [
                "",
                "SELECTED MEMORY HYPOTHESES",
                "None. Set claimed_insight_ids to [].",
            ]
        )

    preflight = prepared.treatment_preflight_receipt
    if requirement is not None:
        if preflight is None:  # pragma: no cover - preparation binds both.
            raise ValueError("treatment requirement is missing its preflight receipt")
        required_ids = [
            reference.insight_id.value for reference in requirement.required_insights
        ]
        claim_instruction = (
            "claimed_insight_ids must equal this exact set"
            if requirement.claim_mode.value == "exact_required"
            else "claimed_insight_ids may be any honest subset of this set"
        )
        sections.extend(
            [
                "",
                "ASSIGNED INSIGHT TREATMENT CONTRACT",
                "Instantiate the assigned card(s) in the selected finite action. "
                "The full shared option palette above is unchanged. The engine "
                "will reject the proposal before evaluation unless its trusted "
                "option ID and parent-bound identity exactly match the card(s); "
                "family and changed-path checks are additional constraints.",
                _json(
                    {
                        "assignment_role": requirement.assignment_role.value,
                        "required_insight_ids": required_ids,
                        "claim_instruction": claim_instruction,
                        "compatible_option_families": list(
                            preflight.compatible_families
                        ),
                        "compatible_option_ids": [
                            action.option_id for action in preflight.compatible_actions
                        ],
                        "requirement_sha256": requirement.requirement_sha256,
                        "preflight_receipt_sha256": preflight.receipt_sha256,
                        "treatment_binding_kind": (
                            "registered_sham_v1"
                            if plan.compiled_hypothesis_treatment is None
                            else "compiled_hypothesis_v1"
                        ),
                        "treatment_binding_sha256": (
                            requirement.requirement_sha256
                            if plan.compiled_hypothesis_treatment is None
                            else plan.compiled_hypothesis_treatment.binding_sha256
                        ),
                    }
                ),
                "Do not return an incompatible option and do not omit, invent, or "
                "duplicate a required insight claim.",
            ]
        )
    finite_authority = plan.finite_action_set_authority
    if finite_authority is not None:
        required_id = finite_authority.card.reference.insight_id.value
        sections.extend(
            [
                "",
                "FINITE-CHOICE CARD ADMINISTRATION",
                "This is a distinct K-option choice treatment, not the exact "
                "singleton treatment contract used by earlier workflows. The "
                "assigned card must influence the choice, and "
                f'claimed_insight_ids must equal ["{required_id}"]. Any of '
                "the K sealed option IDs is legal; do not treat the anchor as "
                "mandatory.",
            ]
        )

    if plan.operator_kind is OperatorKind.REPRODUCTION:
        sections.append("Reproduce the parent exactly; do not change any value.")
    elif plan.operator_kind is OperatorKind.TYPED_MUTATION:
        contract = plan.mutation_contract
        if contract is None:
            sections.extend(
                [
                    "",
                    "MUTATION OBLIGATION",
                    "Change at least one value, but every changed top-level component must be in: "
                    + _json(list(plan.allowed_top_level))
                    + ". Preserve all other top-level components exactly.",
                    "For every changed path set source='mutation'; omit unchanged paths and do not label a mutation as synthesized.",
                ]
            )
        else:
            contract_instructions = [
                "",
                "MACHINE MUTATION CONTRACT",
                _json(_mutation_contract_record(contract)),
                "Only derived typed-patch operations at exactly the listed editable_paths are legal. "
                "Preserve every other path exactly. Do not bundle edits to siblings, ancestors, or descendants of an editable path.",
                (
                    "You may abstain by returning the parent unchanged when no legal edit is useful."
                    if contract.allow_abstention
                    else "You must make at least one legal edit; returning the parent unchanged is not allowed."
                ),
            ]
            if (
                plan.mutation_response_mode
                is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
            ):
                contract_instructions.append(
                    "Return only the exact path and its new scalar value; do not return a full configuration, intended_changes, or source_attribution."
                )
            elif (
                plan.mutation_response_mode
                is MutationResponseMode.FINITE_OPTION_SELECTION_V1
            ):
                contract_instructions.append(
                    "Return only one listed option_id, design_rationale, and claimed_insight_ids; do not return a full configuration, intended_changes, or source_attribution. "
                    "Every sealed option has already been checked against this machine mutation contract."
                )
            else:
                contract_instructions.append(
                    "For every changed path set source='mutation'; omit unchanged paths and do not label a mutation as synthesized."
                )
            sections.extend(contract_instructions)
    elif plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER:
        if plan.crossover_response_mode is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1:
            sections.extend(
                [
                    "",
                    "CROSSOVER OBLIGATION",
                    "Return only a proper nonempty donor-locus subset from the "
                    "sealed catalog above. The output is a bounded executable "
                    "selection, not a candidate witness.",
                ]
            )
        else:
            sections.extend(
                [
                    "",
                    "CROSSOVER OBLIGATION",
                    "Construct a child meaningfully using both parents. Full configurations are supplied; "
                    "attribute exact inherited components to left/right and only genuinely new values to synthesized. "
                    "Apply the candidate component path contract to both intended_changes and source_attribution: "
                    "the PARENTS rows are evidence envelopes, never path roots. The system independently verifies "
                    "at least one exact contribution from each parent. "
                    + TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT,
                ]
            )
    elif plan.operator_kind is OperatorKind.THREE_WAY_RECOMBINATION:
        assert plan.common_ancestor is not None and prepared.classification is not None
        classification = prepared.classification
        relation_rows = [
            {
                "relation_id": relation.relation_id,
                "kind": relation.kind.value,
                "left_paths": [_path_text(op.path) for op in relation.left_operations],
                "right_paths": [
                    _path_text(op.path) for op in relation.right_operations
                ],
            }
            for relation in classification.relations
        ]
        obligation_rows = [
            {
                "source": obligation.source.value,
                "path": _path_text(obligation.path),
                "expected_value_hash": obligation.expected_value_hash,
            }
            for obligation in prepared.variation_case.preservation_obligations
        ]
        required_resolution_ids = [
            relation.relation_id
            for relation in classification.relations
            if relation.kind
            in {ThreeWayRelationKind.CONFLICT, ThreeWayRelationKind.INVALIDATED}
        ]
        sections.extend(
            [
                "",
                "COMMON ANCESTOR",
                _json(_candidate_evidence(plan.common_ancestor)),
                "",
                "REPLAY-DERIVED BRANCH RELATIONS",
                _json(relation_rows),
                "",
                "PRESERVATION OBLIGATIONS",
                _json(obligation_rows),
                "",
                "REQUIRED CONFLICT RESOLUTION IDS",
                _json(required_resolution_ids),
                "Preserve every feasible predeclared branch innovation; the system derives and verifies opaque preservation receipts automatically. "
                "Do not echo preservation IDs. For each conflict/invalidated relation, return exactly "
                "one legal resolution. The child is re-diffed and claims are verified; prose alone receives no credit.",
                "If REQUIRED CONFLICT RESOLUTION IDS is empty, conflict_resolutions MUST be an empty list; DISJOINT relations are not conflicts. "
                "Never let a selected memory hypothesis override an exact preservation obligation. Defer such a hypothesis to a later mutation instead.",
            ]
        )
    elif plan.operator_kind is OperatorKind.REPAIR:
        sections.append(
            "Repair the target using its typed failure evidence while minimizing unrelated changes."
        )
    return "\n".join(sections)


def default_parent_relative_reward(
    child: EvolutionCandidate,
    parents: tuple[EvolutionCandidate, ...],
    objectives: Sequence[ObjectiveSpec],
) -> float:
    """Maximum mean normalized directional improvement over one actual parent."""

    if not child.valid or not child.operator_compliant:
        return -1.0
    scores = []
    child_values = child.objective_map
    for parent in parents:
        if not parent.valid:
            continue
        parent_values = parent.objective_map
        deltas = []
        for spec in objectives:
            direction = 1.0 if spec.goal == "max" else -1.0
            delta = direction * (child_values[spec.name] - parent_values[spec.name])
            deltas.append(delta / (abs(parent_values[spec.name]) + 1.0))
        scores.append(sum(deltas) / len(deltas))
    return float(max(scores, default=-1.0))


def _dominates(
    left: EvolutionCandidate,
    right: EvolutionCandidate,
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    if not left.valid or not right.valid:
        return False
    weak = True
    strict = False
    for spec in objectives:
        a = left.objective_map[spec.name]
        b = right.objective_map[spec.name]
        if spec.goal == "max":
            weak &= a >= b
            strict |= a > b
        else:
            weak &= a <= b
            strict |= a < b
    return weak and strict


class ReflectionCallStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReflectionCallRequest:
    """Engine-rendered, immutable boundary for one reflection provider call.

    The caller may choose the source receipts, contract, and revision target,
    but the engine binds those values to the *actual* rendered prompt and the
    exact projected outcome rows.  A downstream curation authority can
    therefore validate provider evidence without trusting interceptor metadata.
    """

    label: str
    operation: str
    prompt_sha256: str
    min_insights: int
    max_insights: int
    max_output_tokens: int
    temperature: float | None
    insight_contract_sha256: str | None
    revision_predecessors: tuple[InsightRef, ...]
    revision_predecessor_content_sha256s: tuple[str, ...]
    source_receipt_sha256s: tuple[str, ...]
    source_operator_invocation_ids: tuple[OperatorInvocationId, ...]
    source_outcome_sha256s: tuple[str, ...]
    available_contrast_ids: tuple[str, ...]
    request_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("label", "operation"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"reflection request {name} must be canonical")
        require_sha256(self.prompt_sha256, "reflection prompt_sha256")
        if (
            type(self.min_insights) is not int
            or type(self.max_insights) is not int
            or not 0 <= self.min_insights <= self.max_insights <= 16
            or self.max_insights < 1
        ):
            raise ValueError("reflection request cardinality is invalid")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("reflection max_output_tokens must be positive")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
        ):
            raise ValueError("reflection temperature must be finite or None")
        if self.insight_contract_sha256 is not None:
            require_sha256(
                self.insight_contract_sha256,
                "reflection insight_contract_sha256",
            )
        if type(self.revision_predecessors) is not tuple or any(
            type(value) is not InsightRef for value in self.revision_predecessors
        ):
            raise TypeError(
                "revision_predecessors must contain exact InsightRef values"
            )
        for value in self.revision_predecessors:
            InsightRef.__post_init__(value)
        if len(set(self.revision_predecessors)) != len(self.revision_predecessors):
            raise ValueError("revision_predecessors cannot repeat")
        if len(self.revision_predecessors) != len(
            self.revision_predecessor_content_sha256s
        ):
            raise ValueError("revision target references/content hashes differ")
        for name in (
            "revision_predecessor_content_sha256s",
            "source_receipt_sha256s",
            "source_outcome_sha256s",
            "available_contrast_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            for value in values:
                require_sha256(value, f"reflection request {name}")
        if type(self.source_operator_invocation_ids) is not tuple or any(
            type(value) is not OperatorInvocationId
            for value in self.source_operator_invocation_ids
        ):
            raise TypeError("source_operator_invocation_ids must contain exact IDs")
        for value in self.source_operator_invocation_ids:
            OperatorInvocationId.__post_init__(value)
        if len(set(self.source_operator_invocation_ids)) != len(
            self.source_operator_invocation_ids
        ):
            raise ValueError("source operator invocation IDs cannot repeat")
        if len(self.source_operator_invocation_ids) != len(self.source_outcome_sha256s):
            raise ValueError("source operator IDs/outcome hashes differ")
        if self.available_contrast_ids != tuple(
            sorted(set(self.available_contrast_ids))
        ):
            raise ValueError("available_contrast_ids must be canonical")
        expected = hashlib.sha256(
            _REFLECTION_CALL_REQUEST_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self.to_record()))
        ).hexdigest()
        if self.request_sha256:
            require_sha256(self.request_sha256, "reflection request_sha256")
            if self.request_sha256 != expected:
                raise ValueError("reflection request hash does not authenticate data")
        else:
            object.__setattr__(self, "request_sha256", expected)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "label": self.label,
            "operation": self.operation,
            "prompt_sha256": self.prompt_sha256,
            "min_insights": self.min_insights,
            "max_insights": self.max_insights,
            "max_output_tokens": self.max_output_tokens,
            "temperature_hex": (
                None if self.temperature is None else float(self.temperature).hex()
            ),
            "insight_contract_sha256": self.insight_contract_sha256,
            "revision_predecessors": [
                {
                    "insight_id": value.insight_id.value,
                    "version": value.version,
                }
                for value in self.revision_predecessors
            ],
            "revision_predecessor_content_sha256s": list(
                self.revision_predecessor_content_sha256s
            ),
            "source_receipt_sha256s": list(self.source_receipt_sha256s),
            "source_operator_invocation_ids": [
                value.value for value in self.source_operator_invocation_ids
            ],
            "source_outcome_sha256s": list(self.source_outcome_sha256s),
            "available_contrast_ids": list(self.available_contrast_ids),
        }


def _agentic_call_telemetry_record(
    telemetry: AgenticCallTelemetry,
) -> dict[str, object]:
    if type(telemetry) is not AgenticCallTelemetry:
        raise TypeError("reflection telemetry must be exact")
    AgenticCallTelemetry.__post_init__(telemetry)
    return {
        "requested_model": telemetry.requested_model,
        "resolved_model": telemetry.resolved_model,
        "resolved_provider": telemetry.resolved_provider,
        "provider_response_id": telemetry.provider_response_id,
        "finish_reason": telemetry.finish_reason,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "reasoning_tokens": telemetry.reasoning_tokens,
        "cache_read_tokens": telemetry.cache_read_tokens,
        "cache_write_tokens": telemetry.cache_write_tokens,
        "cost_usd": (None if telemetry.cost_usd is None else str(telemetry.cost_usd)),
        "latency_ns": telemetry.latency_ns,
        "attempt_count": telemetry.attempt_count,
    }


def _agentic_call_telemetry_sha256(telemetry: AgenticCallTelemetry) -> str:
    return hashlib.sha256(
        _REFLECTION_CALL_TELEMETRY_DOMAIN
        + canonical_typed_json_bytes(
            freeze_json(_agentic_call_telemetry_record(telemetry))
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ReflectionPublication:
    """Engine-derived binding for one memory entry published by reflection."""

    reference: InsightRef
    content_sha256: str
    evidence_lineage_sha256: str
    lifecycle_state: InsightLifecycleState
    origin: InsightOrigin
    initial_score: float
    revision_predecessor: InsightRef | None
    relations_sha256: str
    publication_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("publication reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        for name in (
            "content_sha256",
            "evidence_lineage_sha256",
            "relations_sha256",
        ):
            require_sha256(getattr(self, name), f"reflection publication {name}")
        if type(self.lifecycle_state) is not InsightLifecycleState:
            raise TypeError("publication lifecycle_state must be exact")
        if type(self.origin) is not InsightOrigin:
            raise TypeError("publication origin must be exact")
        if type(self.initial_score) is not float or not math.isfinite(
            self.initial_score
        ):
            raise TypeError("publication initial_score must be a finite float")
        if self.revision_predecessor is not None:
            if type(self.revision_predecessor) is not InsightRef:
                raise TypeError("revision_predecessor must be an exact InsightRef")
            InsightRef.__post_init__(self.revision_predecessor)
        expected = hashlib.sha256(
            _REFLECTION_PUBLICATION_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self.to_record()))
        ).hexdigest()
        if self.publication_sha256:
            require_sha256(self.publication_sha256, "publication_sha256")
            if self.publication_sha256 != expected:
                raise ValueError("publication hash does not authenticate data")
        else:
            object.__setattr__(self, "publication_sha256", expected)

    def to_record(self) -> dict[str, object]:
        predecessor = self.revision_predecessor
        return {
            "schema_version": 1,
            "reference": {
                "insight_id": self.reference.insight_id.value,
                "version": self.reference.version,
            },
            "content_sha256": self.content_sha256,
            "evidence_lineage_sha256": self.evidence_lineage_sha256,
            "lifecycle_state": self.lifecycle_state.value,
            "origin": self.origin.value,
            "initial_score_hex": self.initial_score.hex(),
            "revision_predecessor": (
                None
                if predecessor is None
                else {
                    "insight_id": predecessor.insight_id.value,
                    "version": predecessor.version,
                }
            ),
            "relations_sha256": self.relations_sha256,
        }


def _reflection_publication(entry: InsightMemoryEntry) -> ReflectionPublication:
    if type(entry) is not InsightMemoryEntry:
        raise TypeError("reflection publication requires an exact memory entry")
    InsightMemoryEntry.__post_init__(entry)
    lineage = entry.evidence_lineage
    if lineage is None:
        raise ValueError("reflection publication requires evidence lineage")
    revisions = tuple(
        relation.target
        for relation in entry.relations
        if relation.kind is InsightRelationKind.REVISES
    )
    if len(revisions) > 1:
        raise ValueError("reflection publication has multiple revision targets")
    relation_record = [
        {
            "kind": relation.kind.value,
            "target": {
                "insight_id": relation.target.insight_id.value,
                "version": relation.target.version,
            },
            "note": relation.note,
        }
        for relation in entry.relations
    ]
    relations_sha256 = hashlib.sha256(
        _REFLECTION_PUBLICATION_DOMAIN
        + b"relations\x00"
        + canonical_typed_json_bytes(freeze_json(relation_record))
    ).hexdigest()
    return ReflectionPublication(
        reference=entry.reference,
        content_sha256=entry.draft.content_sha256,
        evidence_lineage_sha256=lineage.identity_sha256,
        lifecycle_state=entry.lifecycle_state,
        origin=entry.origin,
        initial_score=float(entry.initial_score),
        revision_predecessor=(None if not revisions else revisions[0]),
        relations_sha256=relations_sha256,
    )


@dataclass(frozen=True, slots=True)
class ReflectionCallReceipt:
    """Engine-issued request/provider/publication evidence for one call."""

    call_id: LLMCallId
    request: ReflectionCallRequest
    status: ReflectionCallStatus
    telemetry: AgenticCallTelemetry | None
    telemetry_sha256: str | None
    failure_type: str | None
    publications: tuple[ReflectionPublication, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.call_id) is not LLMCallId:
            raise TypeError("reflection receipt call_id must be exact")
        LLMCallId.__post_init__(self.call_id)
        if type(self.request) is not ReflectionCallRequest:
            raise TypeError("reflection receipt request must be exact")
        ReflectionCallRequest.__post_init__(self.request)
        if type(self.status) is not ReflectionCallStatus:
            raise TypeError("reflection receipt status must be exact")
        if self.telemetry is not None:
            expected_telemetry_sha256 = _agentic_call_telemetry_sha256(self.telemetry)
            if self.telemetry_sha256 is None:
                object.__setattr__(
                    self,
                    "telemetry_sha256",
                    expected_telemetry_sha256,
                )
            elif self.telemetry_sha256 != expected_telemetry_sha256:
                raise ValueError("reflection telemetry hash is not authentic")
        elif self.telemetry_sha256 is not None:
            raise ValueError("telemetry_sha256 requires telemetry")
        if self.status is ReflectionCallStatus.COMPLETED:
            if self.telemetry is None or self.failure_type is not None:
                raise ValueError("completed reflection requires telemetry only")
        elif (
            type(self.failure_type) is not str
            or not self.failure_type
            or self.publications
        ):
            raise ValueError("failed reflection has invalid failure/publication data")
        if type(self.publications) is not tuple or any(
            type(value) is not ReflectionPublication for value in self.publications
        ):
            raise TypeError("publications must contain exact typed values")
        for value in self.publications:
            ReflectionPublication.__post_init__(value)
        references = tuple(value.reference for value in self.publications)
        if len(set(references)) != len(references):
            raise ValueError("published reflection references cannot repeat")
        expected = hashlib.sha256(
            _REFLECTION_CALL_RECEIPT_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self.to_record()))
        ).hexdigest()
        if self.receipt_sha256:
            require_sha256(self.receipt_sha256, "reflection receipt_sha256")
            if self.receipt_sha256 != expected:
                raise ValueError("reflection receipt hash does not authenticate data")
        else:
            object.__setattr__(self, "receipt_sha256", expected)

    def to_record(self) -> dict[str, object]:
        telemetry = self.telemetry
        return {
            "schema_version": 2,
            "call_id": self.call_id.value,
            "request": {
                **self.request.to_record(),
                "request_sha256": self.request.request_sha256,
            },
            "status": self.status.value,
            "telemetry": (
                None if telemetry is None else _agentic_call_telemetry_record(telemetry)
            ),
            "telemetry_sha256": self.telemetry_sha256,
            "failure_type": self.failure_type,
            "publications": [
                {
                    **value.to_record(),
                    "publication_sha256": value.publication_sha256,
                }
                for value in self.publications
            ],
        }

    @property
    def published_references(self) -> tuple[InsightRef, ...]:
        return tuple(value.reference for value in self.publications)

    @property
    def published_content_sha256s(self) -> tuple[str, ...]:
        return tuple(value.content_sha256 for value in self.publications)

    @property
    def published_lineage_sha256s(self) -> tuple[str, ...]:
        return tuple(value.evidence_lineage_sha256 for value in self.publications)


@dataclass(frozen=True, slots=True)
class ReflectionPublicationResult:
    """Typed success value returned by the receipt-bearing reflection API."""

    entries: tuple[InsightMemoryEntry, ...]
    receipt: ReflectionCallReceipt

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or any(
            type(value) is not InsightMemoryEntry for value in self.entries
        ):
            raise TypeError("reflection result entries must be exact")
        if type(self.receipt) is not ReflectionCallReceipt:
            raise TypeError("reflection result receipt must be exact")
        ReflectionCallReceipt.__post_init__(self.receipt)
        if self.receipt.status is not ReflectionCallStatus.COMPLETED:
            raise ValueError("reflection publication result requires completion")
        expected_publications = tuple(
            _reflection_publication(entry) for entry in self.entries
        )
        if self.receipt.publications != expected_publications:
            raise ValueError("reflection result entries differ from its receipt")
        for entry in self.entries:
            InsightMemoryEntry.__post_init__(entry)
            lineage = entry.evidence_lineage
            if lineage is None or lineage.reflection_call_id != self.receipt.call_id:
                raise ValueError("reflection result has foreign call lineage")


class ReflectionCallExecutionError(RuntimeError):
    """A single logical reflection call failed after provider admission.

    The error carries exact accounting so a post-generation curation policy can
    mark itself incomplete without erasing already sealed optimization
    endpoints or pretending that the attempted logical call was free.
    """

    logical_llm_calls_used = 1

    def __init__(
        self,
        call_id: LLMCallId,
        cause: Exception,
        receipt: ReflectionCallReceipt,
    ) -> None:
        if type(call_id) is not LLMCallId:
            raise TypeError("call_id must be an exact LLMCallId")
        if not isinstance(cause, Exception):
            raise TypeError("cause must be an Exception")
        if type(receipt) is not ReflectionCallReceipt:
            raise TypeError("receipt must be an exact ReflectionCallReceipt")
        ReflectionCallReceipt.__post_init__(receipt)
        if (
            receipt.call_id != call_id
            or receipt.status is not ReflectionCallStatus.FAILED
            or receipt.failure_type != type(cause).__name__
        ):
            raise ValueError("failure receipt differs from reflection cause")
        self.call_id = call_id
        self.failure_type = type(cause).__name__
        self.receipt = receipt
        super().__init__(
            f"reflection call {call_id.value} failed with {self.failure_type}: {cause}"
        )


class AgenticEvolutionEngine:
    """Reusable application service for explicit concurrent variation batches."""

    def __init__(
        self,
        *,
        problem: Any,
        generator: AgenticGenerator,
        id_factory: Any,
        memory: InsightMemoryBank,
        seed: int,
        initial_proposal_sequence: int = 0,
        evaluator_concurrency: int = 4,
        trace_sink: TraceSink | None = None,
        reward_policy: RewardPolicy = default_parent_relative_reward,
        reward_definition_hash: str = REWARD_DEFINITION_HASH,
        failure_score: float = -1.0,
        prompt_builder: PromptBuilder = default_evidence_prompt,
        prompt_shape_commitment_policy: PromptShapeCommitmentPolicy | None = None,
        reflection_row_projection: ReflectionRowProjectionBinding | None = None,
        reflection_workflow: ReflectionWorkflow | None = None,
        max_output_tokens: int = 2_048,
        structured_output_budget_policy: StructuredOutputBudgetPolicy | None = None,
        temperature: float | None = 0.2,
        phenotype_identity_policy: PhenotypeIdentityPolicy | None = None,
        detailed_evaluator: DetailedEvaluationAdapter | None = None,
        outcome_relation_binding: OutcomeRelationPolicyBinding | None = None,
        optimization_semantics: OptimizationSemantics | None = None,
        objective_resolution: ObjectiveResolutionPort | None = None,
        treatment_compliance_policy: TreatmentCompliancePolicy | None = None,
    ) -> None:
        objectives = tuple(problem.objectives)
        validate_objective_specs(objectives)
        if not isinstance(generator, AgenticGenerator):
            raise TypeError("generator must implement AgenticGenerator")
        if type(evaluator_concurrency) is not int or evaluator_concurrency <= 0:
            raise ValueError("evaluator_concurrency must be positive")
        if type(initial_proposal_sequence) is not int or initial_proposal_sequence < 0:
            raise ValueError("initial_proposal_sequence must be non-negative")
        if not callable(reward_policy):
            raise TypeError("reward_policy must be callable")
        require_sha256(reward_definition_hash, "reward_definition_hash")
        if type(failure_score) is not float or not math.isfinite(failure_score):
            raise TypeError("failure_score must be a finite canonical float")
        if (reward_policy is default_parent_relative_reward) != (
            reward_definition_hash == REWARD_DEFINITION_HASH
        ):
            raise ValueError(
                "reward_policy and reward_definition_hash must identify the "
                "same reward semantics"
            )
        identity_policy = (
            TypedConfigurationPhenotypeIdentityPolicy()
            if phenotype_identity_policy is None
            else phenotype_identity_policy
        )
        identify = getattr(identity_policy, "identify", None)
        if not callable(identify):
            raise TypeError("phenotype_identity_policy must implement identify")
        policy_id = getattr(identity_policy, "policy_id", None)
        policy_version = getattr(identity_policy, "policy_version", None)
        # Reuse the identity value object's closed metadata validation instead
        # of allowing the engine and recourse layers to develop subtly different
        # policy grammars.
        PhenotypeIdentity(
            policy_id=policy_id,
            policy_version=policy_version,
            value_sha256="0" * 64,
        )
        default_outcome_relation = objective_pareto_outcome_binding(objectives)
        if detailed_evaluator is None:
            if outcome_relation_binding is not None:
                raise ValueError(
                    "a custom outcome relation requires a detailed evaluator"
                )
            active_outcome_relation = default_outcome_relation
            detailed_evaluate = None
            evaluator_identity = None
        else:
            detailed_evaluate = getattr(detailed_evaluator, "evaluate_evidence", None)
            if not callable(detailed_evaluate):
                raise TypeError("detailed_evaluator must implement evaluate_evidence")
            evaluator_identity = getattr(
                detailed_evaluator,
                "evaluator_identity",
                None,
            )
            if type(evaluator_identity) is not EvaluatorIdentity:
                raise TypeError(
                    "detailed_evaluator must publish an exact evaluator_identity"
                )
            EvaluatorIdentity.__post_init__(evaluator_identity)
            if type(outcome_relation_binding) is not OutcomeRelationPolicyBinding:
                raise ValueError(
                    "a detailed evaluator requires an explicit outcome relation binding"
                )
            OutcomeRelationPolicyBinding.__post_init__(outcome_relation_binding)
            active_outcome_relation = outcome_relation_binding
        if optimization_semantics is not None:
            if type(optimization_semantics) is not OptimizationSemantics:
                raise TypeError(
                    "optimization_semantics must be an exact OptimizationSemantics"
                )
            OptimizationSemantics.__post_init__(optimization_semantics)
            optimization_semantics.validate_binding(
                objectives,
                active_outcome_relation.identity,
            )
        objective_resolution_metadata = (
            None
            if objective_resolution is None
            else objective_resolution_policy_metadata(objective_resolution)
        )
        shape_policy = prompt_shape_commitment_policy
        using_default_renderer = prompt_builder is default_evidence_prompt
        if shape_policy is None and using_default_renderer:
            shape_policy = DefaultEvidencePromptShapePolicyV3()
        if shape_policy is not None:
            if not callable(getattr(shape_policy, "commit", None)):
                raise TypeError("prompt_shape_commitment_policy must implement commit")
            shape_metadata = (
                getattr(shape_policy, "policy_id", None),
                getattr(shape_policy, "policy_version", None),
                getattr(shape_policy, "renderer_policy_id", None),
                getattr(shape_policy, "renderer_policy_version", None),
            )
            for index, name in ((0, "policy_id"), (2, "renderer_policy_id")):
                value = shape_metadata[index]
                if type(value) is not str or not value.strip():
                    raise ValueError(f"prompt-shape {name} must be non-empty")
            for index, name in ((1, "policy_version"), (3, "renderer_policy_version")):
                value = shape_metadata[index]
                if type(value) is not int or value <= 0:
                    raise ValueError(f"prompt-shape {name} must be positive")
            claims_default_renderer = shape_metadata[2] == "default_evidence_prompt"
            current_default_pairing = shape_metadata[2:] == (
                "default_evidence_prompt",
                3,
            )
            if (using_default_renderer and not current_default_pairing) or (
                not using_default_renderer and claims_default_renderer
            ):
                raise ValueError(
                    "prompt builder and prompt-shape renderer identity do not match"
                )
        else:
            shape_metadata = None
        if reflection_row_projection is not None:
            if type(reflection_row_projection) is not ReflectionRowProjectionBinding:
                raise TypeError(
                    "reflection_row_projection must be an exact "
                    "ReflectionRowProjectionBinding"
                )
            ReflectionRowProjectionBinding.__post_init__(reflection_row_projection)
        if reflection_workflow is not None and not isinstance(
            reflection_workflow, ReflectionWorkflow
        ):
            raise TypeError("reflection_workflow must implement ReflectionWorkflow")
        treatment_policy = (
            StrictTreatmentCompliancePolicy()
            if treatment_compliance_policy is None
            else treatment_compliance_policy
        )
        if not callable(getattr(treatment_policy, "preflight", None)) or not callable(
            getattr(treatment_policy, "assess", None)
        ):
            raise TypeError(
                "treatment_compliance_policy must implement preflight and assess"
            )
        treatment_policy_id = getattr(treatment_policy, "policy_id", None)
        treatment_policy_version = getattr(treatment_policy, "policy_version", None)
        treatment_policy_hash = getattr(treatment_policy, "definition_sha256", None)
        if type(treatment_policy_id) is not str or not treatment_policy_id:
            raise ValueError("treatment compliance policy_id must be non-empty")
        if type(treatment_policy_version) is not int or treatment_policy_version <= 0:
            raise ValueError("treatment compliance policy_version must be positive")
        require_sha256(treatment_policy_hash, "treatment compliance definition_sha256")
        legacy_output_budget = FixedStructuredOutputBudgetPolicy(
            proposal_max_output_tokens=max_output_tokens,
            reflection_max_output_tokens=max_output_tokens,
        )
        output_budget_policy = (
            legacy_output_budget
            if structured_output_budget_policy is None
            else structured_output_budget_policy
        )
        output_budget_metadata = structured_output_budget_policy_metadata(
            output_budget_policy
        )
        self.problem = problem
        self.objectives = objectives
        self.generator = generator
        self.ids = id_factory
        self.memory = memory
        self.rng = random.Random(seed)
        self._evaluation_slots = asyncio.Semaphore(evaluator_concurrency)
        self._trace_sink = trace_sink
        self._reward = reward_policy
        self.reward_definition_hash = reward_definition_hash
        self.reward_binding = RewardPolicyBinding(
            reward_policy,
            reward_definition_hash,
            failure_score,
        )
        self._prompt_builder = prompt_builder
        self._prompt_shape_commitment_policy = shape_policy
        self._prompt_shape_policy_metadata = shape_metadata
        self._reflection_row_projection = reflection_row_projection
        self._reflection_workflow = reflection_workflow
        self._reflection_call_receipts: dict[
            LLMCallId,
            ReflectionCallReceipt,
        ] = {}
        # Reflection publication mutates versioned memory.  Serializing the
        # whole request/publication transaction makes receipt attribution exact
        # even when independent callers concurrently request curation.
        self._reflection_publication_lock = asyncio.Lock()
        self._using_default_prompt_builder = using_default_renderer
        self.structured_output_budget_policy = output_budget_policy
        self._structured_output_budget_policy_metadata = output_budget_metadata
        self._temperature = temperature
        self._phenotype_identity_policy = identity_policy
        self._phenotype_identity_policy_id = policy_id
        self._phenotype_identity_policy_version = policy_version
        self._detailed_evaluator = detailed_evaluator
        self._detailed_evaluate = detailed_evaluate
        self._evaluator_identity = evaluator_identity
        self.outcome_relation_binding = active_outcome_relation
        self.optimization_semantics = optimization_semantics
        self.objective_resolution = objective_resolution
        self._objective_resolution_metadata = objective_resolution_metadata
        self.treatment_compliance_policy = treatment_policy
        self._treatment_compliance_policy_metadata = (
            treatment_policy_id,
            treatment_policy_version,
            treatment_policy_hash,
        )
        self.optimization_semantics_record = (
            None
            if optimization_semantics is None
            else optimization_semantics.to_record()
        )
        self._objective_pareto_relation = (
            active_outcome_relation.identity == default_outcome_relation.identity
        )
        self._detailed_evaluation_enabled = detailed_evaluator is not None
        self._exact_configuration_identity = (
            type(identity_policy) is TypedConfigurationPhenotypeIdentityPolicy
        )
        # Replay/readiness instances may start after externally prevalidated
        # occurrences.  New occurrence identities still reserve monotonically
        # from this explicit checkpoint before concurrent work begins.
        self._proposal_sequence = initial_proposal_sequence
        self._reserved_operator_invocation_ids: set[OperatorInvocationId] = set()
        self._trace_sequence = 0
        self._trace_origin_ns = time.monotonic_ns()
        self._evaluation_cache: AsyncEvaluationCache[
            DetailedEvaluation | tuple[bool, tuple[tuple[str, float], ...], str | None]
        ] = AsyncEvaluationCache(trace_callback=self._record_evaluation_cache_event)
        self._phenotype_identities_by_cache_key: dict[str, PhenotypeIdentity] = {}
        self.problem_id = f"{type(problem).__module__}.{type(problem).__qualname__}"
        base_problem_description = (
            problem.search_space_description()
            if hasattr(problem, "search_space_description")
            else self.problem_id
        )
        self.optimization_semantics_prompt = (
            None
            if optimization_semantics is None
            else render_optimization_semantics(optimization_semantics)
        )
        self.problem_description = (
            base_problem_description
            if self.optimization_semantics_prompt is None
            else "\n\n".join(
                (base_problem_description, self.optimization_semantics_prompt)
            )
        )

    def _validate_optimization_semantics_prompt(self, prompt: str) -> str:
        if type(prompt) is not str or not prompt:
            raise ValueError("prompt builder must return a non-empty exact string")
        semantics_prompt = self.optimization_semantics_prompt
        if semantics_prompt is not None and prompt.count(semantics_prompt) != 1:
            raise ValueError(
                "prompt builder must preserve the bound optimization semantics "
                "exactly once"
            )
        return prompt

    def _emit(self, event_type: str, **payload: object) -> None:
        if self._trace_sink is None:
            return
        self._trace_sequence += 1
        event = {
            "sequence": self._trace_sequence,
            "event_type": event_type,
            "monotonic_offset_ns": time.monotonic_ns() - self._trace_origin_ns,
            **payload,
        }
        self._trace_sink(event)

    def _record_evaluation_cache_event(self, event: EvaluationCacheTraceEvent) -> None:
        snapshot = event.snapshot
        identity = self._phenotype_identities_by_cache_key.get(event.config_hash)
        identity_record: dict[str, object]
        if identity is None:  # pragma: no cover - guarded by _bind_phenotype_identity
            identity_record = {
                "identity_sha256": event.config_hash,
                "policy_id": self._phenotype_identity_policy_id,
                "policy_version": self._phenotype_identity_policy_version,
                "value_sha256": None,
                "metadata_complete": False,
            }
        else:
            identity_record = {
                **identity.to_trace_record(),
                "identity_sha256": identity.identity_sha256,
                "metadata_complete": True,
            }
        payload: dict[str, object] = {
            "cache_sequence": event.sequence,
            "cache_event_type": event.event_type.value,
            "phenotype_identity": identity_record,
            "cache_snapshot": {
                "capacity": snapshot.capacity,
                "cached_entries": snapshot.cached_entries,
                "in_flight": snapshot.in_flight,
                "hits": snapshot.hits,
                "misses": snapshot.misses,
                "coalesced": snapshot.coalesced,
                "evictions": snapshot.evictions,
            },
        }
        # Under the exact default policy the phenotype value is precisely the
        # historical typed configuration hash.  A semantic policy may group
        # multiple such hashes, so emitting one as if it named the cache event
        # would be false provenance.
        if self._exact_configuration_identity and identity is not None:
            payload["configuration_hash"] = identity.value_sha256
        self._emit(
            "evaluation_cache_event",
            **payload,
        )

    def identify_phenotype(
        self,
        candidate_or_configuration: EvolutionCandidate | object,
    ) -> PhenotypeIdentity:
        """Resolve a deterministic physical-evaluation identity without caching it.

        Candidate occurrences remain separate causal units.  This method only
        projects their immutable configuration into the identity policy used by
        the physical evaluation cache and recourse planner.
        """

        if type(candidate_or_configuration) is EvolutionCandidate:
            EvolutionCandidate.__post_init__(candidate_or_configuration)
            configuration = candidate_or_configuration.configuration
        else:
            configuration = candidate_or_configuration
        frozen = freeze_json(configuration)
        if type(frozen) is not FrozenJsonObject:
            raise TypeError("phenotype configuration root must be an exact object")

        policy = self._phenotype_identity_policy
        if (
            getattr(policy, "policy_id", None) != self._phenotype_identity_policy_id
            or getattr(policy, "policy_version", None)
            != self._phenotype_identity_policy_version
        ):
            raise ValueError("phenotype identity policy metadata changed after binding")

        # Phenotype policies are benchmark-facing ports: give each invocation a
        # fresh detached candidate value, not AgentEvolve's private immutable
        # typed-JSON container.  Independent copies retain the hostile-policy
        # determinism check even if an implementation mutates its argument.
        first = policy.identify(thaw_json(frozen))
        second = policy.identify(thaw_json(frozen))
        for identity in (first, second):
            if type(identity) is not PhenotypeIdentity:
                raise TypeError(
                    "phenotype identity policy must return exact PhenotypeIdentity"
                )
            PhenotypeIdentity.__post_init__(identity)
            if (
                identity.policy_id != self._phenotype_identity_policy_id
                or identity.policy_version != self._phenotype_identity_policy_version
            ):
                raise ValueError(
                    "phenotype identity policy returned inconsistent metadata"
                )
        if first != second:
            raise ValueError(
                "phenotype identity policy must be deterministic for one configuration"
            )
        if (
            getattr(policy, "policy_id", None) != self._phenotype_identity_policy_id
            or getattr(policy, "policy_version", None)
            != self._phenotype_identity_policy_version
        ):
            raise ValueError(
                "phenotype identity policy metadata changed during identify"
            )
        return first

    def _max_output_tokens_for(
        self,
        request_kind: StructuredOutputRequestKind,
        operation: str,
    ) -> int:
        policy = self.structured_output_budget_policy
        if (
            structured_output_budget_policy_metadata(policy)
            != self._structured_output_budget_policy_metadata
        ):
            raise ValueError(
                "structured-output budget policy metadata changed after binding"
            )
        value = resolve_structured_output_budget(
            policy,
            request_kind=request_kind,
            operation=operation,
        )
        if (
            structured_output_budget_policy_metadata(policy)
            != self._structured_output_budget_policy_metadata
        ):
            raise ValueError(
                "structured-output budget policy metadata changed after resolution"
            )
        return value

    def prompt_shape_commitment(
        self,
        plan: InvocationPlan,
        *,
        selected_insight_count: int,
        reward_definition_hash: str | None = None,
    ) -> str:
        """Commit all non-treatment prompt inputs before assignment resolution.

        A planner calls this with an otherwise complete plan and the intended
        card count, then binds the returned digest into
        :class:`ResolvedInsightAssignment`. Insight identity, content, score,
        block, arm, and randomization realization never enter this projection.
        """

        if type(plan) is not InvocationPlan:
            raise TypeError("plan must be an exact InvocationPlan")
        InvocationPlan.__post_init__(plan)
        if type(selected_insight_count) is not int or selected_insight_count < 0:
            raise ValueError("selected_insight_count must be non-negative")
        active_reward_definition_hash = (
            self.reward_definition_hash
            if reward_definition_hash is None
            else reward_definition_hash
        )
        require_sha256(
            active_reward_definition_hash,
            "reward_definition_hash",
        )
        policy = self._prompt_shape_commitment_policy
        if policy is None:
            raise ValueError(
                "resolved causal memory with a custom prompt builder requires "
                "an explicit prompt-shape commitment policy"
            )
        current_metadata = (
            getattr(policy, "policy_id", None),
            getattr(policy, "policy_version", None),
            getattr(policy, "renderer_policy_id", None),
            getattr(policy, "renderer_policy_version", None),
        )
        if current_metadata != self._prompt_shape_policy_metadata:
            raise ValueError("prompt-shape policy metadata changed after binding")

        candidate_model = getattr(self.problem, "candidate_model", None)
        if not isinstance(candidate_model, type) or not issubclass(
            candidate_model, BaseModel
        ):
            raise TypeError(
                "resolved causal memory requires a Pydantic candidate_model"
            )
        candidate_schema = candidate_model.model_json_schema(by_alias=False)
        if type(candidate_schema) is not dict:
            raise TypeError("candidate_model must return an object JSON schema")

        mutation_contract_record = _mutation_contract_record(plan.mutation_contract)
        inputs = PromptShapeInputs(
            problem_description_sha256=hashlib.sha256(
                self.problem_description.encode("utf-8", errors="strict")
            ).hexdigest(),
            exact_context_hash=context_stratum_hash(
                problem_id=self.problem_id,
                operator_kind=plan.operator_kind.value,
                phase=plan.phase,
            ),
            parent_evidence_sha256s=tuple(
                _canonical_json_sha256(_candidate_evidence(parent))
                for parent in plan.parents
            ),
            common_ancestor_evidence_sha256=(
                None
                if plan.common_ancestor is None
                else _canonical_json_sha256(_candidate_evidence(plan.common_ancestor))
            ),
            operator_kind=plan.operator_kind.value,
            operator_version=_plan_operator_version(plan),
            phase=plan.phase,
            allowed_top_level=plan.allowed_top_level,
            mutation_contract_sha256=(
                None
                if mutation_contract_record is None
                else _canonical_json_sha256(mutation_contract_record)
            ),
            mutation_response_mode=plan.mutation_response_mode.value,
            atomic_replacement_option_sha256s=tuple(
                typed_json_sha256(option) for option in plan.atomic_replacement_options
            ),
            candidate_schema_sha256=_canonical_json_sha256(candidate_schema),
            selected_insight_count=selected_insight_count,
            reward_definition_hash=active_reward_definition_hash,
            max_output_tokens=self._max_output_tokens_for(
                StructuredOutputRequestKind.PROPOSAL,
                plan.operator_kind.value,
            ),
            temperature=self._temperature,
            finite_variation_contract_sha256=(
                None
                if plan.finite_variation_contract is None
                else plan.finite_variation_contract.identity_sha256
            ),
            crossover_response_mode=plan.crossover_response_mode.value,
            exact_parent_crossover_contract_sha256=(
                None
                if plan.exact_parent_crossover_contract is None
                else plan.exact_parent_crossover_contract.contract_sha256
            ),
            exact_parent_import_exclusions_sha256=(
                None
                if plan.exact_parent_crossover_contract is None
                else exact_parent_import_exclusions_sha256(
                    plan.exact_parent_crossover_contract,
                    plan.forbidden_exact_parent_import_sets,
                )
            ),
        )
        first = policy.commit(inputs)
        second = policy.commit(inputs)
        require_sha256(first, "prompt_shape_commitment")
        require_sha256(second, "prompt_shape_commitment")
        if first != second:
            raise ValueError("prompt-shape commitment policy must be deterministic")
        if (
            getattr(policy, "policy_id", None),
            getattr(policy, "policy_version", None),
            getattr(policy, "renderer_policy_id", None),
            getattr(policy, "renderer_policy_version", None),
        ) != self._prompt_shape_policy_metadata:
            raise ValueError("prompt-shape policy metadata changed during commit")
        return first

    def _resolved_assignment_binding(
        self,
        plan: InvocationPlan,
        *,
        editable_paths: tuple[str, ...] | None,
        reward_definition_hash: str,
    ) -> tuple[
        InsightSelectionDecision,
        tuple[InsightRef, ...],
        tuple[dict[str, object], ...],
        str,
    ]:
        resolved = plan.resolved_insight_assignment
        if type(resolved) is not ResolvedInsightAssignment:
            raise TypeError("plan has no exact resolved insight assignment")
        context_hash = context_stratum_hash(
            problem_id=self.problem_id,
            operator_kind=plan.operator_kind.value,
            phase=plan.phase,
        )
        if resolved.exact_context_hash != context_hash:
            raise ValueError(
                "resolved memory assignment context differs from the invocation"
            )
        compiled_matrix = plan.compiled_hypothesis_eligibility
        if compiled_matrix:
            entries = self.memory.entries_for(resolved.selection_decision.eligible)
            if any(
                entry.lifecycle_state is InsightLifecycleState.DEPRECATED
                for entry in entries
            ):
                raise ValueError(
                    "compiled controlled-test eligibility contains a deprecated insight"
                )
            if editable_paths is None:
                raise ValueError(
                    "compiled eligibility requires an editable mutation scope"
                )
            for entry, compiled in zip(entries, compiled_matrix, strict=True):
                if (
                    entry.reference,
                    entry.draft.content_sha256,
                    entry.applicable_operator_kinds,
                    registered_source_evidence_sha256(entry),
                ) != (
                    compiled.request.reference,
                    compiled.request.insight.content_sha256,
                    compiled.request.source_operator_kinds,
                    compiled.request.source_evidence_sha256,
                ):
                    raise ValueError(
                        "compiled eligibility differs from registered memory source"
                    )
                if compiled.request.requested_operator_kind != plan.operator_kind.value:
                    raise ValueError(
                        "compiled eligibility differs from invocation operator"
                    )
                if (
                    compiled.request.endpoint_definition_sha256
                    != reward_definition_hash
                ):
                    raise ValueError(
                        "compiled eligibility is bound to a different reward/Q endpoint"
                    )
                if not any(
                    _claim_covers_path(claim, editable)
                    or _claim_covers_path(editable, claim)
                    for claim in compiled.receipt.spec.affected_paths
                    for editable in editable_paths
                ):
                    raise ValueError(
                        "compiled eligibility is disjoint from editable paths"
                    )
        else:
            structurally_eligible = set(
                self.memory.eligible_references(
                    operator_kind=plan.operator_kind.value,
                    editable_paths=editable_paths,
                )
            )
            if not set(resolved.selection_decision.eligible).issubset(
                structurally_eligible
            ):
                raise ValueError(
                    "resolved memory assignment contains an unavailable or "
                    "structurally inapplicable insight"
                )
        selection = resolved.selection_decision
        selected = selection.selected
        selected_records = self.memory.selected_prompt_records(selection)
        verified_shape = self.prompt_shape_commitment(
            plan,
            selected_insight_count=len(selected),
            reward_definition_hash=reward_definition_hash,
        )
        if resolved.prompt_shape_sha256 != verified_shape:
            raise ValueError(
                "resolved memory assignment prompt-shape commitment differs "
                "from the engine projection"
            )
        return selection, selected, selected_records, verified_shape

    def _bind_phenotype_identity(self, identity: PhenotypeIdentity) -> str:
        cache_key = identity.identity_sha256
        existing = self._phenotype_identities_by_cache_key.get(cache_key)
        if existing is not None and existing != identity:
            raise RuntimeError("phenotype identity digest collision")
        self._phenotype_identities_by_cache_key[cache_key] = identity
        return cache_key

    def _detailed_evaluation_cache_key(
        self,
        identity: PhenotypeIdentity,
        evaluator: EvaluatorIdentity,
    ) -> str:
        """Bind physical phenotype and evaluator context, never occurrence identity."""

        PhenotypeIdentity.__post_init__(identity)
        EvaluatorIdentity.__post_init__(evaluator)
        digest = hashlib.sha256()
        digest.update(_DETAILED_EVALUATION_CACHE_DOMAIN)
        digest.update(bytes.fromhex(identity.identity_sha256))
        evaluator_id = evaluator.evaluator_id.encode("ascii", errors="strict")
        digest.update(len(evaluator_id).to_bytes(8, "big"))
        digest.update(evaluator_id)
        digest.update(evaluator.evaluator_version.to_bytes(8, "big"))
        digest.update(bytes.fromhex(evaluator.evaluator_context_sha256))
        cache_key = digest.hexdigest()
        existing = self._phenotype_identities_by_cache_key.get(cache_key)
        if existing is not None and existing != identity:
            raise RuntimeError("detailed evaluation cache-key collision")
        self._phenotype_identities_by_cache_key[cache_key] = identity
        return cache_key

    def _emit_insight_transition(self, transition: InsightLifecycleTransition) -> None:
        self._emit(
            "insight_lifecycle_transition",
            insight_id=transition.reference.insight_id.value,
            version=transition.reference.version,
            prior_state=transition.prior_state.value,
            new_state=transition.new_state.value,
            reason=transition.reason,
            supporting_evidence=list(transition.supporting_evidence),
            transition_sequence=transition.sequence,
        )

    def promote_insight(
        self,
        reference: InsightRef,
        *,
        reason: str,
        supporting_evidence: Sequence[str],
    ) -> InsightMemoryEntry:
        """Promote a tested insight and mirror its append-only transition in traces."""

        entry = self.memory.promote(
            reference,
            reason=reason,
            supporting_evidence=supporting_evidence,
        )
        self._emit_insight_transition(self.memory.transitions[-1])
        return entry

    def deprecate_insight(
        self,
        reference: InsightRef,
        *,
        reason: str,
        supporting_evidence: Sequence[str] = (),
    ) -> InsightMemoryEntry:
        """Deprecate an insight and mirror its append-only transition in traces."""

        entry = self.memory.deprecate(
            reference,
            reason=reason,
            supporting_evidence=supporting_evidence,
        )
        self._emit_insight_transition(self.memory.transitions[-1])
        return entry

    def _new_occurrence(
        self,
        configuration: object,
        *,
        operator_invocation_id: OperatorInvocationId | None,
        candidate_id: CandidateId | None = None,
        proposal_sequence: int | None = None,
    ) -> tuple[CandidateOccurrence, FrozenJsonValue]:
        frozen = freeze_json(configuration)
        if type(frozen) is not FrozenJsonObject:
            raise TypeError("candidate configuration root must be an exact object")
        if candidate_id is not None:
            if type(candidate_id) is not CandidateId:
                raise TypeError("candidate_id must be an exact CandidateId")
            CandidateId.__post_init__(candidate_id)
        if proposal_sequence is None:
            self._proposal_sequence += 1
            active_proposal_sequence = self._proposal_sequence
        else:
            if (
                type(proposal_sequence) is not int
                or proposal_sequence <= 0
                or proposal_sequence > self._proposal_sequence
            ):
                raise ValueError(
                    "reserved proposal_sequence must be a positive allocated ordinal"
                )
            active_proposal_sequence = proposal_sequence
        canonical = canonical_typed_json_bytes(frozen)
        occurrence = CandidateOccurrence(
            candidate_id=(
                self.ids.new_candidate_id() if candidate_id is None else candidate_id
            ),
            configuration_hash=typed_json_sha256(frozen),
            configuration_artifact_hash=hashlib.sha256(canonical).hexdigest(),
            proposal_sequence=active_proposal_sequence,
            operator_invocation_id=operator_invocation_id,
        )
        return occurrence, frozen

    async def _evaluate(
        self, configuration: FrozenJsonValue
    ) -> tuple[
        bool,
        tuple[tuple[str, float], ...],
        str | None,
        DetailedEvaluation | None,
        ObjectiveResolutionReceipt | None,
    ]:
        config = thaw_json(configuration)
        if type(config) is not dict:
            raise TypeError("candidate root must be an object")

        def legacy_blocking() -> tuple[
            bool,
            tuple[tuple[str, float], ...],
            str | None,
        ]:
            try:
                if hasattr(self.problem, "validate") and not self.problem.validate(
                    config
                ):
                    return False, (), "validate returned False"
                values = normalize_objective_values(
                    self.problem.evaluate(config), self.objectives
                )
            except ValueError as exc:
                message = str(exc).strip() or type(exc).__name__
                return False, (), message[:1_024]
            return (
                True,
                tuple((spec.name, values[spec.name]) for spec in self.objectives),
                None,
            )

        async def direct_legacy_evaluation() -> tuple[
            bool, tuple[tuple[str, float], ...], str | None
        ]:
            async with self._evaluation_slots:
                return await asyncio.to_thread(legacy_blocking)

        identity = self.identify_phenotype(configuration)
        if not self._detailed_evaluation_enabled:
            cache_key = self._bind_phenotype_identity(identity)
            legacy = await self._evaluation_cache.get_or_evaluate(
                cache_key,
                direct_legacy_evaluation,
            )
            if type(legacy) is not tuple or len(legacy) != 3:
                raise RuntimeError("legacy evaluation cache returned invalid evidence")
            valid, objectives, failure = legacy
            resolved, receipt = self._resolve_evaluated_objectives(
                configuration,
                valid=valid,
                raw_objectives=objectives,
            )
            return valid, resolved, failure, None, receipt

        adapter = self._detailed_evaluator
        evaluate_detailed = self._detailed_evaluate
        evaluator_identity = self._evaluator_identity
        if (
            adapter is None
            or not callable(evaluate_detailed)
            or type(evaluator_identity) is not EvaluatorIdentity
        ):  # pragma: no cover - constructor establishes this invariant.
            raise RuntimeError("detailed evaluator binding was lost")

        def detailed_blocking() -> DetailedEvaluation:
            try:
                current_identity = getattr(adapter, "evaluator_identity", None)
                if current_identity != evaluator_identity:
                    raise ValueError(
                        "detailed evaluator identity changed after binding"
                    )
                before_configuration = freeze_json(config)
                started_ns = time.monotonic_ns()
                payload = evaluate_detailed(config)
                elapsed = float((time.monotonic_ns() - started_ns) / 1_000_000_000)
                if not typed_json_equal(before_configuration, freeze_json(config)):
                    raise ValueError(
                        "detailed evaluator mutated its configuration input"
                    )
                normalized = normalize_detailed_payload(payload, self.objectives)
                if normalized.evaluator != evaluator_identity:
                    raise ValueError(
                        "detailed payload evaluator identity differs from its binding"
                    )
                return DetailedEvaluation(
                    phenotype=identity,
                    payload=normalized,
                    timings=EvaluationTimings(
                        total_wall_seconds=elapsed,
                        active_wall_seconds=normalized.active_wall_seconds,
                        resource_queue_wall_seconds=(
                            normalized.resource_queue_wall_seconds
                        ),
                    ),
                )
            except Exception as exc:
                raise _DetailedEvaluationPortError(
                    "detailed evaluator port failed"
                ) from exc

        async def direct_detailed_evaluation() -> DetailedEvaluation:
            async with self._evaluation_slots:
                evaluation = await asyncio.to_thread(detailed_blocking)
            self._emit(
                "detailed_evaluation_completed",
                detailed_evaluation=evaluation.to_record(),
            )
            failure_record = evaluation.failure
            if (
                failure_record is not None
                and failure_record.category is not FailureCategory.CANDIDATE
            ):
                raise _TerminalDetailedEvaluationError(evaluation)
            return evaluation

        cache_key = self._detailed_evaluation_cache_key(identity, evaluator_identity)
        detailed = await self._evaluation_cache.get_or_evaluate(
            cache_key,
            direct_detailed_evaluation,
        )
        if type(detailed) is not DetailedEvaluation:
            raise RuntimeError("detailed evaluation cache returned legacy evidence")
        DetailedEvaluation.__post_init__(detailed)
        if detailed.phenotype != identity:
            raise RuntimeError("cached detailed evaluation has the wrong phenotype")
        if detailed.payload.evaluator != evaluator_identity:
            raise RuntimeError("cached detailed evaluation has the wrong context")
        failure_record = detailed.failure
        resolved, receipt = self._resolve_evaluated_objectives(
            configuration,
            valid=detailed.success,
            raw_objectives=detailed.objectives,
        )
        return (
            detailed.success,
            resolved,
            None if failure_record is None else failure_record.message,
            detailed,
            receipt,
        )

    def _resolve_evaluated_objectives(
        self,
        configuration: FrozenJsonValue,
        *,
        valid: bool,
        raw_objectives: tuple[tuple[str, float], ...],
    ) -> tuple[
        tuple[tuple[str, float], ...],
        ObjectiveResolutionReceipt | None,
    ]:
        """Project physical evidence without changing the evaluation cache."""

        policy = self.objective_resolution
        if not valid or policy is None:
            return raw_objectives, None
        if type(configuration) is not FrozenJsonObject:
            raise TypeError("objective resolution requires an object configuration")
        metadata = objective_resolution_policy_metadata(policy)
        if metadata != self._objective_resolution_metadata:
            raise ValueError(
                "objective-resolution policy metadata changed after binding"
            )
        receipt = resolve_objectives(
            policy,
            ObjectiveResolutionRequest(
                configuration=configuration,
                objectives=self.objectives,
                raw_objectives=raw_objectives,
            ),
        )
        if objective_resolution_policy_metadata(policy) != metadata:
            raise ValueError(
                "objective-resolution policy metadata changed during resolution"
            )
        return receipt.decision_objectives, receipt

    async def evaluation_cache_snapshot(self) -> dict[str, int | None]:
        """Return run-local cache evidence without exposing mutable cache state."""

        snapshot = await self._evaluation_cache.snapshot()
        return {
            "capacity": snapshot.capacity,
            "cached_entries": snapshot.cached_entries,
            "in_flight": snapshot.in_flight,
            "hits": snapshot.hits,
            "misses": snapshot.misses,
            "coalesced": snapshot.coalesced,
            "evictions": snapshot.evictions,
        }

    def reflection_call_receipt(
        self,
        call_id: LLMCallId,
    ) -> ReflectionCallReceipt:
        """Return immutable provider/publication evidence for one reflection."""

        if type(call_id) is not LLMCallId:
            raise TypeError("call_id must be an exact LLMCallId")
        try:
            return self._reflection_call_receipts[call_id]
        except KeyError as exc:
            raise KeyError("unknown reflection call receipt") from exc

    @property
    def reflection_call_receipts(self) -> tuple[ReflectionCallReceipt, ...]:
        """Insertion-ordered immutable view of engine-issued call receipts."""

        return tuple(self._reflection_call_receipts.values())

    def _record_reflection_call_receipt(
        self,
        receipt: ReflectionCallReceipt,
    ) -> None:
        if type(receipt) is not ReflectionCallReceipt:
            raise TypeError("reflection receipt must be exact")
        ReflectionCallReceipt.__post_init__(receipt)
        if receipt.call_id in self._reflection_call_receipts:
            raise RuntimeError("reflection call receipt was already published")
        self._reflection_call_receipts[receipt.call_id] = receipt

    def compare_candidates(
        self,
        left: EvolutionCandidate,
        right: EvolutionCandidate,
    ) -> OutcomeRelation:
        """Apply the engine-bound outcome relation to two evaluated occurrences."""

        if (
            type(left) is not EvolutionCandidate
            or type(right) is not EvolutionCandidate
        ):
            raise TypeError(
                "candidate comparison requires exact EvolutionCandidate values"
            )
        EvolutionCandidate.__post_init__(left)
        EvolutionCandidate.__post_init__(right)
        if not left.valid or not right.valid:
            raise ValueError("candidate comparison requires valid evaluations")
        if self._detailed_evaluation_enabled and not self._objective_pareto_relation:
            if left.detailed_evaluation is None or right.detailed_evaluation is None:
                raise ValueError(
                    "the bound detailed outcome policy requires detailed evidence"
                )
            return self.outcome_relation_binding.relate(
                left.detailed_evaluation,
                right.detailed_evaluation,
            )
        if _dominates(left, right, self.objectives):
            return OutcomeRelation.BETTER
        if _dominates(right, left, self.objectives):
            return OutcomeRelation.WORSE
        if left.objectives == right.objectives:
            return OutcomeRelation.EQUIVALENT
        return OutcomeRelation.INCOMPARABLE

    async def register_seed(
        self,
        configuration: dict[str, Any],
        *,
        label: str,
        generation: int = 0,
    ) -> EvolutionCandidate:
        occurrence, frozen = self._new_occurrence(
            configuration, operator_invocation_id=None
        )
        valid, objectives, failure, detailed, resolution = await self._evaluate(frozen)
        candidate = EvolutionCandidate(
            occurrence=occurrence,
            configuration=frozen,
            objectives=objectives,
            valid=valid,
            generation=generation,
            label=label,
            failure_message=failure,
            detailed_evaluation=detailed,
            objective_resolution_receipt=resolution,
        )
        self._emit(
            "seed_registered",
            candidate_id=candidate.candidate_id.value,
            label=label,
            valid=valid,
            configuration=candidate.configuration_dict,
            objectives=candidate.objective_map,
            failure=failure,
            **(
                {}
                if detailed is None
                else {"detailed_evaluation": detailed.to_record()}
            ),
            **(
                {}
                if resolution is None
                else {"objective_resolution": resolution.to_record()}
            ),
        )
        return candidate

    async def register_materialized_candidate(
        self,
        configuration: dict[str, Any],
        *,
        label: str,
        generation: int,
        proposal_source_id: str,
        proposal_source_version: int,
        proposal_source_definition_sha256: str,
        proposal_decision_sha256: str,
        proposal_rank: int,
        candidate_id: CandidateId | None = None,
    ) -> EvolutionCandidate:
        """Evaluate a full configuration selected by an injected search expert.

        The expert owns proposal generation and selection; the engine retains
        candidate identity, evaluation/cache semantics, and trace authority.
        This is intentionally distinct from ``register_seed`` so acquisition,
        specialist incumbents, and future proxy-gated experts cannot disappear
        into seed provenance.
        """

        if type(configuration) is not dict:
            raise TypeError("configuration must be an exact dict")
        if type(label) is not str or not label.strip():
            raise ValueError("label must be non-empty")
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be positive")
        if (
            type(proposal_source_id) is not str
            or not proposal_source_id.isascii()
            or not proposal_source_id
            or proposal_source_id != proposal_source_id.strip()
            or any(value.isspace() for value in proposal_source_id)
        ):
            raise ValueError("proposal_source_id must be a canonical ASCII token")
        if type(proposal_source_version) is not int or proposal_source_version <= 0:
            raise ValueError("proposal_source_version must be positive")
        require_sha256(
            proposal_source_definition_sha256,
            "proposal_source_definition_sha256",
        )
        require_sha256(proposal_decision_sha256, "proposal_decision_sha256")
        if type(proposal_rank) is not int or proposal_rank <= 0:
            raise ValueError("proposal_rank must be positive")

        occurrence, frozen = self._new_occurrence(
            configuration,
            operator_invocation_id=None,
            candidate_id=candidate_id,
        )
        valid, objectives, failure, detailed, resolution = await self._evaluate(frozen)
        candidate = EvolutionCandidate(
            occurrence=occurrence,
            configuration=frozen,
            objectives=objectives,
            valid=valid,
            generation=generation,
            label=label,
            design_rationale=(
                f"Full configuration selected by {proposal_source_id} "
                f"rank {proposal_rank}."
            ),
            failure_message=failure,
            detailed_evaluation=detailed,
            objective_resolution_receipt=resolution,
        )
        self._emit(
            "materialized_candidate_registered",
            candidate_id=candidate.candidate_id.value,
            generation=generation,
            label=label,
            valid=valid,
            configuration=candidate.configuration_dict,
            objectives=candidate.objective_map,
            failure=failure,
            proposal_source={
                "source_id": proposal_source_id,
                "source_version": proposal_source_version,
                "definition_sha256": proposal_source_definition_sha256,
                "decision_sha256": proposal_decision_sha256,
                "rank": proposal_rank,
            },
            **(
                {}
                if detailed is None
                else {"detailed_evaluation": detailed.to_record()}
            ),
            **(
                {}
                if resolution is None
                else {"objective_resolution": resolution.to_record()}
            ),
        )
        return candidate

    def _obligation_requests(
        self, classification: ThreeWayPatchClassification
    ) -> tuple[PreservationObligationRequest, ...]:
        requests: list[PreservationObligationRequest] = []
        for relation in classification.relations:
            if relation.kind not in {
                ThreeWayRelationKind.DISJOINT,
                ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT,
            }:
                continue
            requests.extend(
                PreservationObligationRequest(
                    relation.relation_id,
                    PreservationSource.LEFT_BRANCH,
                    operation.path,
                )
                for operation in relation.left_operations
            )
            requests.extend(
                PreservationObligationRequest(
                    relation.relation_id,
                    PreservationSource.RIGHT_BRANCH,
                    operation.path,
                )
                for operation in relation.right_operations
            )
        return tuple(requests)

    def _finite_treatment_actions(
        self, plan: InvocationPlan
    ) -> tuple[FiniteTreatmentAction, ...]:
        contract = plan.finite_variation_contract
        if contract is None:
            raise ValueError("treatment administration requires a finite contract")
        parent = plan.parents[0]
        probe = CandidateId("candidate_treatment_preflight_probe")
        if probe == parent.candidate_id:
            probe = CandidateId("candidate_treatment_preflight_probe_alternate")
        actions = []
        for option in contract.options:
            patch = derive_patch(
                parent.configuration,
                option.child_configuration,
                base_candidate_id=parent.candidate_id,
                target_candidate_id=probe,
            )
            actions.append(
                FiniteTreatmentAction(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    family=option.family,
                    changed_paths=tuple(
                        sorted(
                            {
                                _path_text(operation.path)
                                for operation in patch.operations
                            }
                        )
                    ),
                )
            )
        return tuple(actions)

    def _preflight_treatment(
        self,
        plan: InvocationPlan,
        selected: tuple[InsightRef, ...],
        editable_paths: tuple[str, ...] | None,
    ) -> TreatmentPreflightReceipt | None:
        requirement = plan.insight_treatment_requirement
        if requirement is None:
            return None
        if editable_paths is None:
            raise ValueError("treatment administration requires editable paths")
        entries = self.memory.entries_for(selected)
        compiled = plan.compiled_hypothesis_treatment
        if compiled is None:
            insights = tuple(
                TreatmentInsightEvidence(
                    reference=entry.reference,
                    insight_content_sha256=entry.draft.content_sha256,
                    applicable_operator_kinds=entry.applicable_operator_kinds,
                    affected_paths=tuple(sorted(entry.draft.affected_paths)),
                    recommended_option_families=tuple(
                        sorted(entry.draft.recommended_option_families)
                    ),
                    recommended_option_ids=tuple(
                        sorted(entry.draft.recommended_option_ids)
                    ),
                )
                for entry in entries
            )
        else:
            CompiledHypothesisTreatment.__post_init__(compiled)
            if len(entries) != 1:
                raise ValueError(
                    "compiled hypothesis treatment requires one exact memory entry"
                )
            entry = entries[0]
            expected_source = registered_source_evidence_sha256(entry)
            if expected_source != compiled.request.source_evidence_sha256:
                raise ValueError(
                    "compiled treatment source evidence changed before preflight"
                )
            if (
                entry.reference,
                entry.draft.content_sha256,
                entry.applicable_operator_kinds,
            ) != (
                compiled.request.reference,
                compiled.request.insight.content_sha256,
                compiled.request.source_operator_kinds,
            ):
                raise ValueError(
                    "compiled treatment differs from selected memory entry"
                )
            if requirement != compiled.requirement:
                raise ValueError(
                    "compiled treatment requirement changed before preflight"
                )
            insights = (compiled.treatment_evidence,)
        finite_contract = plan.finite_variation_contract
        if finite_contract is None:  # pragma: no cover - plan validation.
            raise ValueError("treatment administration requires a finite contract")
        request = TreatmentPreflightRequest(
            requirement=requirement,
            operator_kind=plan.operator_kind.value,
            editable_paths=tuple(sorted(editable_paths)),
            insights=insights,
            finite_contract_sha256=finite_contract.identity_sha256,
            actions=self._finite_treatment_actions(plan),
        )
        receipt = self.treatment_compliance_policy.preflight(request)
        if type(receipt) is not TreatmentPreflightReceipt:
            raise TypeError("treatment policy returned an invalid preflight receipt")
        TreatmentPreflightReceipt.__post_init__(receipt)
        validate_treatment_preflight_receipt(request, receipt)
        if (
            receipt.policy_id,
            receipt.policy_version,
            receipt.policy_definition_sha256,
        ) != self._treatment_compliance_policy_metadata:
            raise ValueError("treatment policy metadata changed during preflight")
        if not receipt.passed:
            raise ValueError("assigned treatment has no compliant finite action")
        return receipt

    def _prepare(
        self,
        plan: InvocationPlan,
        *,
        proposal_authority: ProposalAuthority | None = None,
        materialization_policy_id: str | None = None,
        materialization_policy_version: int | None = None,
        materialization_receipt_hash: str | None = None,
        materialized_candidate_id: CandidateId | None = None,
        materialized_finite_action_authority: FiniteActionSetAuthority | None = None,
        materialized_finite_action_decision: FiniteActionDecision | None = None,
        reward_definition_hash: str | None = None,
    ) -> PreparedInvocation:
        authority = proposal_authority
        if authority is None:
            authority = (
                ProposalAuthority.REPRODUCTION
                if plan.operator_kind is OperatorKind.REPRODUCTION
                else ProposalAuthority.MODEL
            )
        if type(authority) is not ProposalAuthority:
            raise TypeError("proposal_authority must be a ProposalAuthority")
        if (plan.operator_kind is OperatorKind.REPRODUCTION) != (
            authority is ProposalAuthority.REPRODUCTION
        ):
            raise ValueError("reproduction plans and reproduction authority must agree")
        if (
            plan.mutation_response_mode
            is MutationResponseMode.FINITE_OPTION_SELECTION_V1
            and authority is not ProposalAuthority.MODEL
        ):
            raise ValueError(
                "finite option selection requires model proposal authority"
            )
        materialization_values = (
            materialization_policy_id,
            materialization_policy_version,
            materialization_receipt_hash,
            materialized_candidate_id,
            materialized_finite_action_authority,
            materialized_finite_action_decision,
        )
        if authority is ProposalAuthority.ENGINE:
            if (
                type(materialization_policy_id) is not str
                or not materialization_policy_id
                or type(materialization_policy_version) is not int
                or materialization_policy_version <= 0
                or type(materialization_receipt_hash) is not str
                or type(materialized_candidate_id) is not CandidateId
            ):
                raise ValueError(
                    "engine authority requires complete materialization identity"
                )
            require_sha256(
                materialization_receipt_hash,
                "materialization_receipt_hash",
            )
            CandidateId.__post_init__(materialized_candidate_id)
            if (materialized_finite_action_authority is None) != (
                materialized_finite_action_decision is None
            ):
                raise ValueError(
                    "materialized finite action authority and decision must be paired"
                )
            if materialized_finite_action_authority is not None:
                if (
                    type(materialized_finite_action_authority)
                    is not FiniteActionSetAuthority
                ):
                    raise TypeError(
                        "materialized_finite_action_authority must be exact"
                    )
                FiniteActionSetAuthority.__post_init__(
                    materialized_finite_action_authority
                )
                if (
                    type(materialized_finite_action_decision)
                    is not FiniteActionDecision
                ):
                    raise TypeError("materialized_finite_action_decision must be exact")
                validate_finite_action_decision(
                    materialized_finite_action_authority,
                    materialized_finite_action_decision,
                )
                if (
                    materialized_finite_action_decision.selector_kind
                    is not FiniteActionSelectorKind.ENGINE
                ):
                    raise ValueError(
                        "materialized finite action provenance requires an "
                        "engine selector"
                    )
                if (
                    plan.operator_kind is not OperatorKind.TYPED_MUTATION
                    or len(plan.parents) != 1
                ):
                    raise ValueError(
                        "materialized finite action provenance requires one "
                        "typed-mutation parent"
                    )
                parent = plan.parents[0]
                support = materialized_finite_action_authority.support
                if (
                    support.parent_candidate_id != parent.candidate_id
                    or support.parent_configuration_sha256
                    != parent.occurrence.configuration_hash
                    or not typed_json_equal(
                        support.support_contract.parent_configuration,
                        parent.configuration,
                    )
                ):
                    raise ValueError(
                        "materialized finite action authority is bound to a "
                        "different parent"
                    )
                if (
                    materialization_policy_id
                    != materialized_finite_action_decision.selector_policy_id
                    or materialization_policy_version
                    != materialized_finite_action_decision.selector_policy_version
                    or materialization_receipt_hash
                    != materialized_finite_action_decision.decision_sha256
                ):
                    raise ValueError(
                        "materialization identity differs from its finite action "
                        "decision"
                    )
        elif any(value is not None for value in materialization_values):
            raise ValueError("only engine authority accepts materialization identity")
        active_reward_definition_hash = (
            self.reward_definition_hash
            if reward_definition_hash is None
            else reward_definition_hash
        )
        require_sha256(
            active_reward_definition_hash,
            "reward_definition_hash",
        )
        editable_paths: tuple[str, ...] | None = None
        if plan.operator_kind is OperatorKind.TYPED_MUTATION:
            editable_paths = (
                tuple(
                    _path_text(path) for path in plan.mutation_contract.editable_paths
                )
                if plan.mutation_contract is not None
                else tuple(f"$.{name}" for name in plan.allowed_top_level)
            )
        assignment_kind = (
            InsightAssignmentKind.QUARANTINE_TEST
            if plan.quarantine_test_insights
            else (
                InsightAssignmentKind.RETRIEVAL
                if plan.use_memory
                else (
                    InsightAssignmentKind.RESOLVED_CAUSAL
                    if plan.resolved_insight_assignment is not None
                    else None
                )
            )
        )
        selected: tuple[InsightRef, ...] = ()
        selected_records: tuple[dict[str, object], ...] = ()
        verified_prompt_shape_sha256: str | None = None
        if plan.quarantine_test_insights:
            selected = self.memory.validate_quarantine_test_assignment(
                plan.quarantine_test_insights,
                operator_kind=plan.operator_kind.value,
                editable_paths=editable_paths,
            )
            selected_records = self.memory.prompt_records(selected)

        invocation_id = (
            plan.resolved_insight_assignment.credit_unit_id
            if plan.resolved_insight_assignment is not None
            else self.ids.new_operator_invocation_id()
        )
        if type(invocation_id) is not OperatorInvocationId:
            raise TypeError(
                "id factory or resolved assignment returned an invalid "
                "operator invocation ID"
            )
        OperatorInvocationId.__post_init__(invocation_id)
        if invocation_id in self._reserved_operator_invocation_ids:
            raise ValueError("operator invocation ID was already reserved")
        context_hash = context_stratum_hash(
            problem_id=self.problem_id,
            operator_kind=plan.operator_kind.value,
            phase=plan.phase,
        )
        selection = None
        if plan.use_memory:
            eligible_references = self.memory.eligible_references(
                operator_kind=plan.operator_kind.value,
                editable_paths=editable_paths,
            )
            score_context_hash = (
                context_hash
                if plan.memory_score_phase is None
                else context_stratum_hash(
                    problem_id=self.problem_id,
                    operator_kind=plan.operator_kind.value,
                    phase=plan.memory_score_phase,
                )
            )
            selection = self.memory.select(
                context_hash=context_hash,
                subset_size=plan.memory_subset_size,
                rng=self.rng,
                exploration_probability=plan.memory_exploration_probability,
                score_context_hash=score_context_hash,
                eligible_references=eligible_references,
            )
            selected = selection.selected
            selected_records = self.memory.selected_prompt_records(selection)
        elif plan.resolved_insight_assignment is not None:
            (
                selection,
                selected,
                selected_records,
                verified_prompt_shape_sha256,
            ) = self._resolved_assignment_binding(
                plan,
                reward_definition_hash=active_reward_definition_hash,
                editable_paths=editable_paths,
            )
        finite_authority = plan.finite_action_set_authority
        if finite_authority is not None:
            if selected != (finite_authority.card.reference,):
                raise ValueError(
                    "finite action authority changed its selected card at preparation"
                )
            entry = self.memory.entries_for(selected)[0]
            if entry.draft.content_sha256 != finite_authority.card.card_content_sha256:
                raise ValueError(
                    "finite action authority card content differs from memory"
                )
            registered_sha256 = finite_authority.card.registered_source_evidence_sha256
            if (
                registered_sha256 is not None
                and registered_source_evidence_sha256(entry) != registered_sha256
            ):
                raise ValueError(
                    "finite action authority source evidence differs from memory"
                )
        if (
            assignment_kind is InsightAssignmentKind.QUARANTINE_TEST
            and plan.insight_treatment_requirement is None
        ):
            selected_records = tuple(
                {
                    **record,
                    "assignment_kind": assignment_kind.value,
                }
                for record in selected_records
            )
        treatment_preflight = self._preflight_treatment(
            plan,
            selected,
            editable_paths,
        )

        # Validation above is side-effect free for resolved assignments. Only a
        # context/eligibility/shape-valid unit may poison the one-shot credit ID
        # reservation or advance later call/candidate ordinals.
        self._reserved_operator_invocation_ids.add(invocation_id)
        call_id = (
            self.ids.new_llm_call_id() if authority is ProposalAuthority.MODEL else None
        )
        # Reserve occurrence identity in the caller's fixed plan order. Provider
        # responses and evaluations complete concurrently, so allocating either
        # value during response materialization would make lineage and downstream
        # tie-breaking depend on network latency.
        candidate_id = (
            materialized_candidate_id
            if authority is ProposalAuthority.ENGINE
            else self.ids.new_candidate_id()
        )
        assert candidate_id is not None
        self._proposal_sequence += 1
        proposal_sequence = self._proposal_sequence

        roles = {
            OperatorKind.REPRODUCTION: (ParentRole.REPRODUCTION_SOURCE,),
            OperatorKind.TYPED_MUTATION: (ParentRole.MUTATION_PARENT,),
            OperatorKind.TWO_PARENT_CROSSOVER: (
                ParentRole.CROSSOVER_LEFT,
                ParentRole.CROSSOVER_RIGHT,
            ),
            OperatorKind.THREE_WAY_RECOMBINATION: (
                ParentRole.CROSSOVER_LEFT,
                ParentRole.CROSSOVER_RIGHT,
            ),
            OperatorKind.REPAIR: (ParentRole.REPAIR_TARGET,),
        }[plan.operator_kind]
        parents = tuple(
            VariationParent(role, parent.occurrence)
            for role, parent in zip(roles, plan.parents, strict=True)
        )
        classification = None
        ancestor_occurrence = None
        branch_patches = ()
        obligations = ()
        if plan.operator_kind is OperatorKind.THREE_WAY_RECOMBINATION:
            assert plan.common_ancestor is not None
            ancestor_occurrence = plan.common_ancestor.occurrence
            left_patch = derive_patch(
                plan.common_ancestor.configuration,
                plan.parents[0].configuration,
                base_candidate_id=plan.common_ancestor.candidate_id,
                target_candidate_id=plan.parents[0].candidate_id,
            )
            right_patch = derive_patch(
                plan.common_ancestor.configuration,
                plan.parents[1].configuration,
                base_candidate_id=plan.common_ancestor.candidate_id,
                target_candidate_id=plan.parents[1].candidate_id,
            )
            classification = classify_three_way_patches(
                plan.common_ancestor.configuration,
                left_patch,
                right_patch,
            )
            branch_patches = (left_patch, right_patch)
            requests = self._obligation_requests(classification)
            if requests:
                obligations = derive_preservation_obligations(classification, requests)

        case = VariationCase(
            operator_invocation_id=invocation_id,
            variation_kind=plan.operator_kind.variation_kind,
            operator_id=plan.operator_kind.value,
            operator_version=_plan_operator_version(plan),
            parents=parents,
            requested_child_count=1,
            context_stratum_hash=context_hash,
            reward_definition_hash=active_reward_definition_hash,
            common_ancestor=ancestor_occurrence,
            ancestor_to_parent_patches=branch_patches,
            selected_insights=selected,
            preservation_obligations=obligations,
        )
        provisional = PreparedInvocation(
            plan=plan,
            operator_invocation_id=invocation_id,
            call_id=call_id,
            candidate_id=candidate_id,
            proposal_sequence=proposal_sequence,
            variation_case=case,
            classification=classification,
            selection_decision=selection,
            proposal_authority=authority,
            insight_assignment_kind=assignment_kind,
            materialization_policy_id=materialization_policy_id,
            materialization_policy_version=materialization_policy_version,
            materialization_receipt_hash=materialization_receipt_hash,
            materialized_candidate_id=materialized_candidate_id,
            treatment_preflight_receipt=treatment_preflight,
            materialized_finite_action_authority=(materialized_finite_action_authority),
            materialized_finite_action_decision=materialized_finite_action_decision,
        )
        prompt = self._validate_optimization_semantics_prompt(
            self._prompt_builder(
                self.problem_description,
                provisional,
                selected_records,
            )
        )
        prepared = PreparedInvocation(
            plan=plan,
            operator_invocation_id=invocation_id,
            call_id=call_id,
            candidate_id=candidate_id,
            proposal_sequence=proposal_sequence,
            variation_case=case,
            classification=classification,
            selection_decision=selection,
            proposal_authority=authority,
            insight_assignment_kind=assignment_kind,
            prompt=prompt,
            materialization_policy_id=materialization_policy_id,
            materialization_policy_version=materialization_policy_version,
            materialization_receipt_hash=materialization_receipt_hash,
            materialized_candidate_id=materialized_candidate_id,
            treatment_preflight_receipt=treatment_preflight,
            materialized_finite_action_authority=(materialized_finite_action_authority),
            materialized_finite_action_decision=materialized_finite_action_decision,
        )
        if treatment_preflight is not None:
            self._emit(
                "treatment_preflight_completed",
                operator_invocation_id=invocation_id.value,
                call_id=None if call_id is None else call_id.value,
                candidate_id=candidate_id.value,
                requirement={
                    **plan.insight_treatment_requirement.to_record(),
                    "requirement_sha256": (
                        plan.insight_treatment_requirement.requirement_sha256
                    ),
                    "compiled_hypothesis_treatment": (
                        None
                        if plan.compiled_hypothesis_treatment is None
                        else {
                            **plan.compiled_hypothesis_treatment.to_record(),
                            "binding_sha256": (
                                plan.compiled_hypothesis_treatment.binding_sha256
                            ),
                        }
                    ),
                },
                preflight={
                    **treatment_preflight.to_record(),
                    "receipt_sha256": treatment_preflight.receipt_sha256,
                },
            )
        self._emit(
            "invocation_prepared",
            operator_invocation_id=invocation_id.value,
            call_id=None if call_id is None else call_id.value,
            candidate_id=candidate_id.value,
            proposal_sequence=proposal_sequence,
            label=plan.label,
            operator_kind=plan.operator_kind.value,
            proposal_authority=authority.value,
            materialization_policy_id=materialization_policy_id,
            materialization_policy_version=materialization_policy_version,
            materialization_receipt_hash=materialization_receipt_hash,
            materialized_candidate_id=(
                None
                if materialized_candidate_id is None
                else materialized_candidate_id.value
            ),
            parent_ids=[parent.candidate_id.value for parent in plan.parents],
            allowed_top_level=list(plan.allowed_top_level),
            mutation_contract=_mutation_contract_record(plan.mutation_contract),
            mutation_response_mode=plan.mutation_response_mode.value,
            crossover_response_mode=plan.crossover_response_mode.value,
            proposal_representation=_proposal_representation(plan),
            atomic_editable_path=(
                _path_text(plan.mutation_contract.editable_paths[0])
                if plan.mutation_response_mode
                is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
                and plan.mutation_contract is not None
                else None
            ),
            atomic_old_value_hash=(
                typed_json_sha256(
                    value_at_path(
                        plan.parents[0].configuration,
                        plan.mutation_contract.editable_paths[0],
                    )
                )
                if plan.mutation_response_mode
                is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
                and plan.mutation_contract is not None
                else None
            ),
            atomic_replacement_options=[
                thaw_json(option) for option in plan.atomic_replacement_options
            ],
            atomic_replacement_option_hashes=[
                typed_json_sha256(option) for option in plan.atomic_replacement_options
            ],
            **(
                {}
                if plan.finite_variation_contract is None
                else {
                    "finite_variation_contract": (
                        plan.finite_variation_contract.evidence_record()
                    ),
                    "finite_variation_contract_sha256": (
                        plan.finite_variation_contract.identity_sha256
                    ),
                }
            ),
            **(
                {}
                if plan.exact_parent_crossover_contract is None
                else {
                    "exact_parent_crossover_contract": (
                        plan.exact_parent_crossover_contract.to_record()
                    ),
                    "exact_parent_crossover_contract_sha256": (
                        plan.exact_parent_crossover_contract.contract_sha256
                    ),
                    "forbidden_exact_parent_import_sets": [
                        list(value) for value in plan.forbidden_exact_parent_import_sets
                    ],
                    "exact_parent_import_exclusions_sha256": (
                        exact_parent_import_exclusions_sha256(
                            plan.exact_parent_crossover_contract,
                            plan.forbidden_exact_parent_import_sets,
                        )
                    ),
                }
            ),
            **(
                {}
                if plan.finite_action_set_authority is None
                else {
                    "finite_action_set_authority": {
                        **plan.finite_action_set_authority.to_record(),
                        "authority_sha256": (
                            plan.finite_action_set_authority.authority_sha256
                        ),
                    }
                }
            ),
            materialized_finite_action_authority=(
                None
                if materialized_finite_action_authority is None
                else {
                    **materialized_finite_action_authority.to_record(),
                    "authority_sha256": (
                        materialized_finite_action_authority.authority_sha256
                    ),
                }
            ),
            materialized_finite_action_decision=(
                None
                if materialized_finite_action_decision is None
                else {
                    **materialized_finite_action_decision.to_record(),
                    "decision_sha256": (
                        materialized_finite_action_decision.decision_sha256
                    ),
                }
            ),
            phase=plan.phase,
            common_ancestor_id=(
                None
                if plan.common_ancestor is None
                else plan.common_ancestor.candidate_id.value
            ),
            selected_insight_ids=[ref.insight_id.value for ref in selected],
            selected_insights=_insight_reference_records(selected),
            selected_insight_records=list(selected_records),
            assignment_kind=(
                None if assignment_kind is None else assignment_kind.value
            ),
            insight_treatment_requirement=(
                None
                if plan.insight_treatment_requirement is None
                else {
                    **plan.insight_treatment_requirement.to_record(),
                    "requirement_sha256": (
                        plan.insight_treatment_requirement.requirement_sha256
                    ),
                }
            ),
            treatment_preflight_receipt=(
                None
                if treatment_preflight is None
                else {
                    **treatment_preflight.to_record(),
                    "receipt_sha256": treatment_preflight.receipt_sha256,
                }
            ),
            resolved_insight_assignment=(
                None
                if plan.resolved_insight_assignment is None
                else plan.resolved_insight_assignment.to_record()
            ),
            resolved_insight_assignment_sha256=(
                None
                if plan.resolved_insight_assignment is None
                else plan.resolved_insight_assignment.assignment_sha256
            ),
            prompt_shape_commitment_sha256=verified_prompt_shape_sha256,
            prompt_shape_commitment_verified=(verified_prompt_shape_sha256 is not None),
            prompt_shape_policy=(
                None
                if verified_prompt_shape_sha256 is None
                or self._prompt_shape_policy_metadata is None
                else {
                    "policy_id": self._prompt_shape_policy_metadata[0],
                    "policy_version": self._prompt_shape_policy_metadata[1],
                    "renderer_policy_id": self._prompt_shape_policy_metadata[2],
                    "renderer_policy_version": self._prompt_shape_policy_metadata[3],
                }
            ),
            selection_decision=_selection_decision_record(selection),
            selection_mode=None if selection is None else selection.mode.value,
            exploration_probability=(
                None
                if selection is None
                else _fraction_record(selection.exploration_probability)
            ),
            score_context_hash=(
                None
                if not plan.use_memory and plan.resolved_insight_assignment is None
                else (
                    plan.resolved_insight_assignment.exact_context_hash
                    if plan.resolved_insight_assignment is not None
                    else context_stratum_hash(
                        problem_id=self.problem_id,
                        operator_kind=plan.operator_kind.value,
                        phase=(plan.memory_score_phase or plan.phase),
                    )
                )
            ),
            selection_probability=(
                None
                if selection is None
                else _fraction_record(selection.selected_subset_probability)
            ),
            preservation_obligation_ids=[
                item.obligation_id for item in case.preservation_obligations
            ],
            **(
                {}
                if self.optimization_semantics_record is None
                else {"optimization_semantics": self.optimization_semantics_record}
            ),
            prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )
        return prepared

    def _crossover_contribution_paths(
        self,
        prepared: PreparedInvocation,
        occurrence: CandidateOccurrence,
        child: FrozenJsonValue,
    ) -> tuple[set[str], set[str]]:
        left, right = prepared.plan.parents
        right_to_left = derive_patch(
            right.configuration,
            left.configuration,
            base_candidate_id=right.candidate_id,
            target_candidate_id=left.candidate_id,
        )
        left_to_right = derive_patch(
            left.configuration,
            right.configuration,
            base_candidate_id=left.candidate_id,
            target_candidate_id=right.candidate_id,
        )
        right_to_child = derive_patch(
            right.configuration,
            child,
            base_candidate_id=right.candidate_id,
            target_candidate_id=occurrence.candidate_id,
        )
        left_to_child = derive_patch(
            left.configuration,
            child,
            base_candidate_id=left.candidate_id,
            target_candidate_id=occurrence.candidate_id,
        )
        left_effects = {
            operation_effect_bytes(operation): _path_text(operation.path)
            for operation in right_to_left.operations
        }
        right_effects = {
            operation_effect_bytes(operation): _path_text(operation.path)
            for operation in left_to_right.operations
        }
        used_left_paths = {
            left_effects[effect]
            for operation in right_to_child.operations
            if (effect := operation_effect_bytes(operation)) in left_effects
        }
        used_right_paths = {
            right_effects[effect]
            for operation in left_to_child.operations
            if (effect := operation_effect_bytes(operation)) in right_effects
        }
        return used_left_paths, used_right_paths

    def _operator_compliance(
        self,
        prepared: PreparedInvocation,
        draft: CandidateDraft,
        occurrence: CandidateOccurrence,
        child: FrozenJsonValue,
    ) -> tuple[bool, str | None, tuple[str, ...], bool | None]:
        plan = prepared.plan
        patch_hashes = []
        for parent in plan.parents:
            patch_hashes.append(
                derive_patch(
                    parent.configuration,
                    child,
                    base_candidate_id=parent.candidate_id,
                    target_candidate_id=occurrence.candidate_id,
                ).patch_hash
            )

        if plan.operator_kind is OperatorKind.REPRODUCTION:
            ok = typed_json_equal(plan.parents[0].configuration, child)
            return (
                ok,
                None if ok else "reproduction changed content",
                tuple(patch_hashes),
                None,
            )
        if plan.operator_kind is OperatorKind.TYPED_MUTATION:
            # Reuse the real occurrence target rather than the diagnostic helper's
            # temporary ID when publishing patch evidence.
            patch = derive_patch(
                plan.parents[0].configuration,
                child,
                base_candidate_id=plan.parents[0].candidate_id,
                target_candidate_id=occurrence.candidate_id,
            )
            if not patch.operations:
                if (
                    plan.mutation_contract is not None
                    and plan.mutation_contract.allow_abstention
                ):
                    return True, None, (patch.patch_hash,), None
                return (
                    False,
                    "mutation produced unchanged content",
                    (patch.patch_hash,),
                    None,
                )
            allowed = set(plan.allowed_top_level)
            for operation in patch.operations:
                if (
                    not operation.path.segments
                    or type(operation.path.segments[0]) is not ObjectKey
                ):
                    return False, "mutation changed the root", (patch.patch_hash,), None
                if operation.path.segments[0].value not in allowed:
                    return (
                        False,
                        "mutation escaped its declared top-level scope",
                        (patch.patch_hash,),
                        None,
                    )
            contract = plan.mutation_contract
            if contract is not None:
                editable = set(contract.editable_paths)
                changed_paths = {operation.path for operation in patch.operations}
                if any(path not in editable for path in changed_paths):
                    return (
                        False,
                        "mutation changed a path outside its machine contract",
                        (patch.patch_hash,),
                        None,
                    )
                if len(changed_paths) > contract.max_changed_paths:
                    return (
                        False,
                        "mutation exceeded its changed-path cardinality",
                        (patch.patch_hash,),
                        None,
                    )
                if len(patch.operations) > contract.max_operations:
                    return (
                        False,
                        "mutation exceeded its patch-operation cardinality",
                        (patch.patch_hash,),
                        None,
                    )
            if plan.atomic_replacement_options:
                assert contract is not None
                replacement = value_at_path(
                    child,
                    contract.editable_paths[0],
                )
                if not any(
                    typed_json_equal(replacement, option)
                    for option in plan.atomic_replacement_options
                ):
                    return (
                        False,
                        "mutation used a replacement outside its atomic option catalog",
                        (patch.patch_hash,),
                        None,
                    )
            return True, None, (patch.patch_hash,), None
        if plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER:
            if (
                plan.crossover_response_mode
                is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
            ):
                distinct_from_both = all(
                    not typed_json_equal(parent.configuration, child)
                    for parent in plan.parents
                )
                return (
                    distinct_from_both,
                    None
                    if distinct_from_both
                    else "exact parent import did not discriminate both parents",
                    tuple(patch_hashes),
                    None,
                )
            used_left_paths, used_right_paths = self._crossover_contribution_paths(
                prepared, occurrence, child
            )
            if not used_left_paths or not used_right_paths:
                return (
                    False,
                    "crossover lacks a machine-verified contribution from both parents",
                    tuple(patch_hashes),
                    None,
                )
            return True, None, tuple(patch_hashes), None
        if plan.operator_kind is not OperatorKind.THREE_WAY_RECOMBINATION:
            changed = not typed_json_equal(plan.parents[0].configuration, child)
            return (
                changed,
                None if changed else "repair made no change",
                tuple(patch_hashes),
                None,
            )

        assert prepared.classification is not None
        classification = prepared.classification
        if any(
            relation.kind
            in {ThreeWayRelationKind.CONFLICT, ThreeWayRelationKind.INVALIDATED}
            for relation in classification.relations
        ):
            return (
                False,
                "conflict-effect verification is not implemented; recombination failed closed",
                tuple(patch_hashes),
                False,
            )
        obligations = prepared.variation_case.preservation_obligations
        if not obligations:
            return (
                False,
                "parent pair had no two-sided testable preservation obligations",
                tuple(patch_hashes),
                False,
            )
        claims = tuple(
            PreservationClaim(obligation.obligation_id) for obligation in obligations
        )
        bindings: tuple[ParentConfiguration, ...] = tuple(
            bind_parent_configuration(
                parent.occurrence,
                parent.configuration,
                limits=classification.left_patch.limits.json_limits,
            )
            for parent in plan.parents
        )
        try:
            verify_preservation_claims(
                prepared.variation_case,
                classification,
                bindings,
                child,
                claims=claims,
                limits=classification.left_patch.limits.json_limits,
            )
        except (TypeError, ValueError, PreservationError):
            return (
                False,
                "child failed exact two-branch preservation verification",
                tuple(patch_hashes),
                False,
            )
        return True, None, tuple(patch_hashes), True

    def _evidence_compliance(
        self,
        prepared: PreparedInvocation,
        draft: CandidateDraft,
        occurrence: CandidateOccurrence,
        child: FrozenJsonValue,
    ) -> tuple[bool, str | None]:
        """Audit model-authored explanations without redefining child semantics.

        Operator compliance is established from configurations and replayed
        patches.  These checks measure whether the model's optional explanatory
        fields truthfully describe that already-observed transformation; an
        annotation mistake must not corrupt generation-quality or memory credit.
        """

        plan = prepared.plan
        if plan.operator_kind is OperatorKind.REPRODUCTION:
            return True, None
        if plan.operator_kind is OperatorKind.TYPED_MUTATION:
            patch = derive_patch(
                plan.parents[0].configuration,
                child,
                base_candidate_id=plan.parents[0].candidate_id,
                target_candidate_id=occurrence.candidate_id,
            )
            if (
                not patch.operations
                and plan.mutation_contract is not None
                and plan.mutation_contract.allow_abstention
            ):
                if any(item.source == "mutation" for item in draft.source_attribution):
                    return False, "abstention claimed a nonexistent mutation path"
                return True, None
            return _validate_source_claims(
                draft.source_attribution,
                required={
                    "mutation": {
                        _path_text(operation.path) for operation in patch.operations
                    }
                },
            )
        if plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER:
            if (
                plan.crossover_response_mode
                is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
            ):
                contract = plan.exact_parent_crossover_contract
                if type(contract) is not ExactParentCrossoverContract:
                    return False, "exact crossover contract is absent"
                expected_paths = {locus.path_text for locus in contract.loci}
                by_path = {
                    attribution.path: attribution.source
                    for attribution in draft.source_attribution
                }
                if (
                    len(by_path) != len(draft.source_attribution)
                    or set(by_path) != expected_paths
                    or set(by_path.values()) != {"left", "right"}
                ):
                    return (
                        False,
                        "exact crossover attribution is not exhaustive and two-sided",
                    )
                right_paths = tuple(
                    sorted(
                        path for path, source in by_path.items() if source == "right"
                    )
                )
                if draft.intended_changes != right_paths:
                    return False, "exact crossover donor-locus record is inconsistent"
                return True, None
            left_paths, right_paths = self._crossover_contribution_paths(
                prepared, occurrence, child
            )
            return _validate_source_claims(
                draft.source_attribution,
                required={"left": left_paths, "right": right_paths},
            )
        if plan.operator_kind is not OperatorKind.THREE_WAY_RECOMBINATION:
            return True, None

        assert prepared.classification is not None
        classification = prepared.classification
        supplied = []
        for item in draft.conflict_resolutions:
            try:
                choice = ResolutionChoice(item.choice)
            except ValueError:
                return False, "unknown conflict resolution"
            supplied.append(
                PatchResolution(
                    relation_id=item.relation_id,
                    choice=choice,
                    synthesized_result_hash=(
                        typed_json_sha256(child)
                        if choice is ResolutionChoice.SYNTHESIZE
                        else None
                    ),
                )
            )
        try:
            validate_three_way_resolutions(
                classification,
                tuple(sorted(supplied, key=lambda value: value.relation_id)),
            )
        except (TypeError, ValueError):
            return False, "conflict resolution annotations were invalid or extraneous"

        obligations = prepared.variation_case.preservation_obligations
        expected_claim_ids = tuple(
            obligation.obligation_id for obligation in obligations
        )
        supplied_claim_ids = tuple(sorted(draft.claimed_preservation_obligation_ids))
        if supplied_claim_ids and (
            supplied_claim_ids != expected_claim_ids
            or len(set(supplied_claim_ids)) != len(supplied_claim_ids)
        ):
            return False, "optional preservation annotations were incorrect"
        return _validate_source_claims(
            draft.source_attribution,
            required={
                "left": {
                    _path_text(obligation.path)
                    for obligation in obligations
                    if obligation.source is PreservationSource.LEFT_BRANCH
                },
                "right": {
                    _path_text(obligation.path)
                    for obligation in obligations
                    if obligation.source is PreservationSource.RIGHT_BRANCH
                },
            },
        )

    def _materialize_two_parent_crossover(
        self,
        prepared: PreparedInvocation,
        draft: CandidateDraft,
        *,
        candidate_id: CandidateId,
        proposal_sequence: int,
    ) -> tuple[CandidateOccurrence, FrozenJsonValue, dict[str, object]]:
        """Execute model source claims as exact named-parent subtree copies."""

        if prepared.plan.operator_kind is not OperatorKind.TWO_PARENT_CROSSOVER:
            raise ValueError("crossover materialization requires a crossover plan")
        left, right = prepared.plan.parents
        claims: list[CrossoverInheritanceClaim] = []
        for attribution in draft.source_attribution:
            try:
                source = CrossoverInheritanceSource(attribution.source)
            except ValueError as exc:
                raise ValueError(
                    "two-parent crossover attribution uses an unsupported source"
                ) from exc
            claims.append(
                CrossoverInheritanceClaim(path=attribution.path, source=source)
            )
        materialization = materialize_crossover_inheritance(
            left=left.configuration,
            right=right.configuration,
            draft=draft.configuration,
            claims=tuple(claims),
        )
        occurrence, frozen = self._new_occurrence(
            materialization.configuration,
            operator_invocation_id=prepared.operator_invocation_id,
            candidate_id=candidate_id,
            proposal_sequence=proposal_sequence,
        )
        if (
            occurrence.configuration_hash
            != materialization.materialized_configuration_sha256
        ):
            raise RuntimeError("crossover materialization hash differs from occurrence")
        return (
            occurrence,
            frozen,
            {
                "crossover_materialization": materialization.to_record(),
                "crossover_materialization_receipt_sha256": (
                    materialization.receipt_sha256
                ),
                "crossover_draft_configuration_hash": (
                    materialization.draft_configuration_sha256
                ),
                "crossover_materialized_configuration_hash": (
                    materialization.materialized_configuration_sha256
                ),
                "crossover_adjusted_float_leaf_count": sum(
                    item.adjusted_float_leaf_count
                    for item in materialization.inherited_paths
                ),
                "source_attribution_provenance": (
                    "engine_materialized_from_model_inheritance_plan"
                ),
                "target_configuration_hash": occurrence.configuration_hash,
            },
        )

    def _materialize_exact_parent_crossover(
        self,
        prepared: PreparedInvocation,
        draft: ExactParentCrossoverDraft,
        *,
        candidate_id: CandidateId,
        proposal_sequence: int,
    ) -> tuple[
        CandidateDraft,
        CandidateOccurrence,
        FrozenJsonValue,
        dict[str, object],
    ]:
        """Materialize a bounded donor-locus choice with engine-owned evidence."""

        plan = prepared.plan
        if (
            plan.operator_kind is not OperatorKind.TWO_PARENT_CROSSOVER
            or plan.crossover_response_mode
            is not CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
        ):
            raise ValueError(
                "exact parent crossover draft requires exact crossover mode"
            )
        if type(draft) is not ExactParentCrossoverDraft:
            raise TypeError(
                "exact crossover mode requires an ExactParentCrossoverDraft"
            )
        ExactParentCrossoverDraft.__post_init__(draft)
        contract = plan.exact_parent_crossover_contract
        if type(contract) is not ExactParentCrossoverContract:
            raise ValueError("exact crossover mode lost its sealed contract")
        if draft.contract_identity_sha256 != contract.contract_sha256:
            raise ValueError("crossover draft is bound to a different contract")
        if draft.import_locus_ids in plan.forbidden_exact_parent_import_sets:
            raise ValueError("crossover draft materializes a forbidden known child")
        left, right = plan.parents
        materialization = materialize_exact_parent_crossover(
            base=left.configuration,
            donor=right.configuration,
            contract=contract,
            import_locus_ids=draft.import_locus_ids,
        )

        candidate_model = self.problem.candidate_model
        if not isinstance(candidate_model, type) or not issubclass(
            candidate_model, BaseModel
        ):
            raise TypeError("exact crossover mode requires a Pydantic candidate model")
        validated = candidate_model.model_validate(
            thaw_json(materialization.configuration),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        validated_dict = BaseModel.model_dump(
            validated,
            mode="python",
            by_alias=False,
            exclude_unset=False,
            exclude_defaults=False,
            exclude_none=False,
            exclude_computed_fields=True,
            round_trip=True,
            warnings="error",
            fallback=None,
            serialize_as_any=False,
            polymorphic_serialization=False,
        )
        if type(validated_dict) is not dict:
            raise TypeError("candidate model must serialize to an exact object")
        if not typed_json_equal(
            freeze_json(validated_dict),
            materialization.configuration,
        ):
            raise ValueError("candidate validation changed the exact crossover child")

        occurrence, frozen = self._new_occurrence(
            materialization.configuration,
            operator_invocation_id=prepared.operator_invocation_id,
            candidate_id=candidate_id,
            proposal_sequence=proposal_sequence,
        )
        left_paths = tuple(
            attribution.path_text
            for attribution in materialization.attributions
            if attribution.source is ExactParentSource.BASE
        )
        right_paths = tuple(
            attribution.path_text
            for attribution in materialization.attributions
            if attribution.source is ExactParentSource.DONOR
        )
        if not left_paths or not right_paths:  # pragma: no cover - core receipt.
            raise RuntimeError("exact crossover receipt lost a parent contribution")
        system_draft = CandidateDraft(
            configuration=thaw_json(frozen),
            design_rationale=(
                "Engine materialized the selected exact donor-locus subset."
            ),
            intended_changes=tuple(sorted(right_paths)),
            source_attribution=tuple(
                SourceAttribution(
                    attribution.path_text,
                    "left" if attribution.source is ExactParentSource.BASE else "right",
                )
                for attribution in materialization.attributions
            ),
            claimed_insight_ids=draft.claimed_insight_ids,
        )
        receipt = materialization.receipt
        return (
            system_draft,
            occurrence,
            frozen,
            {
                "crossover_contract": contract.to_record(),
                "crossover_contract_sha256": contract.contract_sha256,
                "crossover_import_locus_ids": list(draft.import_locus_ids),
                "crossover_forbidden_import_locus_sets": [
                    list(value) for value in plan.forbidden_exact_parent_import_sets
                ],
                "crossover_import_exclusions_sha256": (
                    exact_parent_import_exclusions_sha256(
                        contract,
                        plan.forbidden_exact_parent_import_sets,
                    )
                ),
                "crossover_plan_sha256": materialization.plan.plan_sha256,
                "crossover_materialization": materialization.to_record(),
                "crossover_materialization_sha256": (
                    materialization.materialization_sha256
                ),
                "crossover_materialization_receipt": receipt.to_record(),
                "crossover_materialization_receipt_sha256": receipt.receipt_sha256,
                "crossover_materialized_configuration_hash": (
                    materialization.materialized_configuration_sha256
                ),
                "crossover_base_parent_candidate_id": left.candidate_id.value,
                "crossover_donor_parent_candidate_id": right.candidate_id.value,
                "source_attribution_provenance": "engine_derived_exact_parent_import",
                "target_configuration_hash": occurrence.configuration_hash,
            },
        )

    def _materialize_atomic_mutation(
        self,
        prepared: PreparedInvocation,
        draft: AtomicMutationDraft,
        *,
        candidate_id: CandidateId | None = None,
        proposal_sequence: int | None = None,
    ) -> tuple[
        CandidateDraft,
        CandidateOccurrence,
        FrozenJsonValue,
        dict[str, object],
    ]:
        """Validate, derive, and replay exactly one engine-owned scalar patch."""

        plan = prepared.plan
        if (
            plan.mutation_response_mode
            is not MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
        ):
            raise TypeError("atomic mutation draft requires atomic response mode")
        if type(draft) is not AtomicMutationDraft:
            raise TypeError("atomic response mode requires an AtomicMutationDraft")
        AtomicMutationDraft.__post_init__(draft)
        contract = plan.mutation_contract
        if contract is None:  # pragma: no cover - InvocationPlan admits this first.
            raise ValueError("atomic response mode requires a mutation contract")
        path = contract.editable_paths[0]
        if draft.path != path:
            raise ValueError("atomic mutation returned a path outside its contract")
        parent = plan.parents[0]
        old_value = value_at_path(parent.configuration, path)
        replacement = freeze_json(draft.replacement)
        if replacement is not draft.replacement or not is_json_scalar(replacement):
            raise TypeError("atomic replacement must be a frozen typed-JSON scalar")
        if typed_json_equal(old_value, replacement):
            raise ValueError("atomic replacement must change the parent value")

        provisional_target = replace_existing_path(
            parent.configuration,
            path,
            replacement,
        )
        candidate_model = self.problem.candidate_model
        if not isinstance(candidate_model, type) or not issubclass(
            candidate_model, BaseModel
        ):
            raise TypeError("atomic response mode requires a Pydantic candidate model")
        validated = candidate_model.model_validate(
            thaw_json(provisional_target),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        validated_dict = BaseModel.model_dump(
            validated,
            mode="python",
            by_alias=False,
            exclude_unset=False,
            exclude_defaults=False,
            exclude_none=False,
            exclude_computed_fields=True,
            round_trip=True,
            warnings="error",
            fallback=None,
            serialize_as_any=False,
            polymorphic_serialization=False,
        )
        if type(validated_dict) is not dict:
            raise TypeError("candidate model must serialize to an exact object")
        if not typed_json_equal(freeze_json(validated_dict), provisional_target):
            raise ValueError("candidate validation changed the typed atomic target")

        target_candidate_id = (
            self.ids.new_candidate_id() if candidate_id is None else candidate_id
        )
        if type(target_candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(target_candidate_id)
        materialized_patch = derive_patch(
            parent.configuration,
            provisional_target,
            base_candidate_id=parent.candidate_id,
            target_candidate_id=target_candidate_id,
        )
        if (
            len(materialized_patch.operations) != 1
            or type(materialized_patch.operations[0]) is not ReplaceScalar
            or materialized_patch.operations[0].path != path
        ):
            raise ValueError(
                "atomic materialization did not derive one exact ReplaceScalar"
            )
        replayed_target = apply_patch(parent.configuration, materialized_patch)
        if not typed_json_equal(replayed_target, provisional_target):
            raise ValueError("atomic materialized patch did not replay exactly")
        occurrence, frozen = self._new_occurrence(
            replayed_target,
            operator_invocation_id=prepared.operator_invocation_id,
            candidate_id=target_candidate_id,
            proposal_sequence=proposal_sequence,
        )
        path_text = _path_text(path)
        system_draft = CandidateDraft(
            configuration=thaw_json(frozen),
            design_rationale=draft.design_rationale,
            intended_changes=(path_text,),
            source_attribution=(SourceAttribution(path_text, "mutation"),),
            claimed_insight_ids=draft.claimed_insight_ids,
        )
        evidence: dict[str, object] = {
            "mutation_response_mode": plan.mutation_response_mode.value,
            "proposal_representation": plan.mutation_response_mode.value,
            "atomic_submitted_path": path_text,
            "atomic_old_value_hash": typed_json_sha256(old_value),
            "atomic_new_value_hash": typed_json_sha256(replacement),
            "materialized_patch_hash": materialized_patch.patch_hash,
            "parent_configuration_hash": parent.occurrence.configuration_hash,
            "target_configuration_hash": occurrence.configuration_hash,
            "source_attribution_provenance": "system_derived",
        }
        return system_draft, occurrence, frozen, evidence

    def _materialize_finite_option_selection(
        self,
        prepared: PreparedInvocation,
        draft: FiniteVariationSelectionDraft,
        *,
        candidate_id: CandidateId | None = None,
        proposal_sequence: int | None = None,
    ) -> tuple[
        CandidateDraft,
        CandidateOccurrence,
        FrozenJsonValue,
        dict[str, object],
    ]:
        """Resolve one model-selected ID into an engine-owned full child."""

        plan = prepared.plan
        if (
            plan.mutation_response_mode
            is not MutationResponseMode.FINITE_OPTION_SELECTION_V1
        ):
            raise TypeError("finite selection draft requires finite option mode")
        if type(draft) is not FiniteVariationSelectionDraft:
            raise TypeError(
                "finite option mode requires a FiniteVariationSelectionDraft"
            )
        FiniteVariationSelectionDraft.__post_init__(draft)
        finite_contract = plan.finite_variation_contract
        if finite_contract is None:  # pragma: no cover - plan admission.
            raise ValueError("finite option mode requires a sealed contract")
        option = resolve_finite_variation_selection(finite_contract, draft)
        parent = plan.parents[0]
        if type(parent.configuration) is not FrozenJsonObject:
            raise TypeError("finite option parent must be a FrozenJsonObject")
        if not typed_json_equal(
            finite_contract.parent_configuration,
            parent.configuration,
        ):
            raise ValueError("finite variation contract parent changed after binding")
        provisional_target = option.child_configuration
        if type(provisional_target) is not FrozenJsonObject:
            raise TypeError("finite variation child must be a FrozenJsonObject")

        candidate_model = self.problem.candidate_model
        if not isinstance(candidate_model, type) or not issubclass(
            candidate_model, BaseModel
        ):
            raise TypeError("finite option mode requires a Pydantic candidate model")
        validated = candidate_model.model_validate(
            thaw_json(provisional_target),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        validated_dict = BaseModel.model_dump(
            validated,
            mode="python",
            by_alias=False,
            exclude_unset=False,
            exclude_defaults=False,
            exclude_none=False,
            exclude_computed_fields=True,
            round_trip=True,
            warnings="error",
            fallback=None,
            serialize_as_any=False,
            polymorphic_serialization=False,
        )
        if type(validated_dict) is not dict:
            raise TypeError("candidate model must serialize to an exact object")
        if not typed_json_equal(freeze_json(validated_dict), provisional_target):
            raise ValueError("candidate validation changed the sealed finite child")

        target_candidate_id = (
            self.ids.new_candidate_id() if candidate_id is None else candidate_id
        )
        if type(target_candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(target_candidate_id)
        materialized_patch = derive_patch(
            parent.configuration,
            provisional_target,
            base_candidate_id=parent.candidate_id,
            target_candidate_id=target_candidate_id,
        )
        mutation_contract = plan.mutation_contract
        if mutation_contract is None:  # pragma: no cover - plan admission.
            raise ValueError("finite option mode requires a mutation contract")
        _validate_finite_option_patch_scope(
            materialized_patch.operations,
            allowed_top_level=plan.allowed_top_level,
            mutation_contract=mutation_contract,
        )
        replayed_target = apply_patch(parent.configuration, materialized_patch)
        if not typed_json_equal(replayed_target, provisional_target):
            raise ValueError("finite option patch did not replay its sealed child")
        occurrence, frozen = self._new_occurrence(
            replayed_target,
            operator_invocation_id=prepared.operator_invocation_id,
            candidate_id=target_candidate_id,
            proposal_sequence=proposal_sequence,
        )
        changed_paths = tuple(
            _path_text(operation.path) for operation in materialized_patch.operations
        )
        system_draft = CandidateDraft(
            configuration=thaw_json(frozen),
            design_rationale=draft.design_rationale,
            intended_changes=changed_paths,
            source_attribution=tuple(
                SourceAttribution(path, "mutation") for path in changed_paths
            ),
            claimed_insight_ids=draft.claimed_insight_ids,
        )
        evidence: dict[str, object] = {
            "mutation_response_mode": plan.mutation_response_mode.value,
            "proposal_representation": plan.mutation_response_mode.value,
            "finite_option_id": option.option_id,
            "finite_option_family": option.family,
            "finite_option_identity_sha256": option.identity_sha256,
            "finite_contract_identity_sha256": finite_contract.identity_sha256,
            "finite_catalog_id": finite_contract.catalog_id,
            "finite_catalog_version": finite_contract.catalog_version,
            "finite_catalog_definition_sha256": (
                finite_contract.catalog_definition_sha256
            ),
            "finite_parent_configuration_sha256": (
                finite_contract.parent_configuration_sha256
            ),
            "finite_child_configuration_sha256": (option.child_configuration_sha256),
            "materialized_patch_hash": materialized_patch.patch_hash,
            "parent_configuration_hash": parent.occurrence.configuration_hash,
            "target_configuration_hash": occurrence.configuration_hash,
            "source_attribution_provenance": "catalog_materialized",
        }
        return system_draft, occurrence, frozen, evidence

    def _admit_treatment_before_evaluation(
        self,
        prepared: PreparedInvocation,
        draft: CandidateDraft,
        *,
        operator_compliant: bool,
        materialization_evidence: Mapping[str, object],
    ) -> TreatmentAdmissionReceipt | None:
        requirement = prepared.plan.insight_treatment_requirement
        if requirement is None:
            return None
        preflight = prepared.treatment_preflight_receipt
        if preflight is None:
            raise _TreatmentCompliancePolicyError(
                "treatment requirement lost its preflight receipt"
            )
        try:
            action = FiniteTreatmentAction(
                option_id=materialization_evidence["finite_option_id"],
                option_identity_sha256=materialization_evidence[
                    "finite_option_identity_sha256"
                ],
                family=materialization_evidence["finite_option_family"],
                changed_paths=tuple(sorted(set(draft.intended_changes))),
            )
            request = TreatmentAdmissionRequest(
                requirement=requirement,
                preflight=preflight,
                claimed_insight_ids=draft.claimed_insight_ids,
                selected_action=action,
                operator_compliant=operator_compliant,
            )
            receipt = self.treatment_compliance_policy.assess(request)
            if type(receipt) is not TreatmentAdmissionReceipt:
                raise TypeError(
                    "treatment policy returned an invalid admission receipt"
                )
            receipt.__post_init__()
            validate_treatment_admission_receipt(request, receipt)
            if (
                receipt.policy_id,
                receipt.policy_version,
                receipt.policy_definition_sha256,
            ) != self._treatment_compliance_policy_metadata:
                raise ValueError("treatment policy metadata changed during admission")
        except TreatmentComplianceRejected:
            raise
        except Exception as exc:
            raise _TreatmentCompliancePolicyError(
                "treatment compliance policy failed"
            ) from exc
        self._emit(
            "treatment_admission_completed",
            operator_invocation_id=prepared.operator_invocation_id.value,
            call_id=(None if prepared.call_id is None else prepared.call_id.value),
            candidate_id=prepared.candidate_id.value,
            assignment_role=requirement.assignment_role.value,
            admission={
                **receipt.to_record(),
                "receipt_sha256": receipt.receipt_sha256,
                "passed": receipt.passed,
            },
            evaluator_entered=receipt.evaluator_entered,
        )
        if not receipt.passed:
            raise TreatmentComplianceRejected(receipt)
        return receipt

    async def _candidate_from_draft(
        self,
        prepared: PreparedInvocation,
        draft: (
            CandidateDraft
            | AtomicMutationDraft
            | FiniteVariationSelectionDraft
            | ExactParentCrossoverDraft
        ),
        telemetry: AgenticCallTelemetry | None,
    ) -> tuple[EvolutionCandidate, TreatmentAdmissionReceipt | None]:
        materialization_evidence: dict[str, object] = {}
        if (
            prepared.plan.mutation_response_mode
            is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
        ):
            if type(draft) is not AtomicMutationDraft:
                raise TypeError("atomic response mode requires an AtomicMutationDraft")
            draft, occurrence, frozen, materialization_evidence = (
                self._materialize_atomic_mutation(
                    prepared,
                    draft,
                    candidate_id=prepared.candidate_id,
                    proposal_sequence=prepared.proposal_sequence,
                )
            )
        elif (
            prepared.plan.mutation_response_mode
            is MutationResponseMode.FINITE_OPTION_SELECTION_V1
        ):
            if type(draft) is not FiniteVariationSelectionDraft:
                raise TypeError(
                    "finite option mode requires a FiniteVariationSelectionDraft"
                )
            draft, occurrence, frozen, materialization_evidence = (
                self._materialize_finite_option_selection(
                    prepared,
                    draft,
                    candidate_id=prepared.candidate_id,
                    proposal_sequence=prepared.proposal_sequence,
                )
            )
        elif (
            prepared.plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER
            and prepared.plan.crossover_response_mode
            is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
        ):
            if type(draft) is not ExactParentCrossoverDraft:
                raise TypeError(
                    "exact crossover mode requires an ExactParentCrossoverDraft"
                )
            draft, occurrence, frozen, materialization_evidence = (
                self._materialize_exact_parent_crossover(
                    prepared,
                    draft,
                    candidate_id=prepared.candidate_id,
                    proposal_sequence=prepared.proposal_sequence,
                )
            )
        else:
            if type(draft) is not CandidateDraft:
                raise TypeError(
                    "full-configuration response mode requires a CandidateDraft"
                )
            if prepared.plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER:
                occurrence, frozen, materialization_evidence = (
                    self._materialize_two_parent_crossover(
                        prepared,
                        draft,
                        candidate_id=prepared.candidate_id,
                        proposal_sequence=prepared.proposal_sequence,
                    )
                )
            else:
                occurrence, frozen = self._new_occurrence(
                    draft.configuration,
                    operator_invocation_id=prepared.operator_invocation_id,
                    candidate_id=prepared.candidate_id,
                    proposal_sequence=prepared.proposal_sequence,
                )
        compliant, operator_failure, patch_hashes, preservation = (
            self._operator_compliance(prepared, draft, occurrence, frozen)
        )
        treatment_admission = self._admit_treatment_before_evaluation(
            prepared,
            draft,
            operator_compliant=compliant,
            materialization_evidence=materialization_evidence,
        )
        evidence_compliant, evidence_failure = self._evidence_compliance(
            prepared, draft, occurrence, frozen
        )
        if (
            prepared.plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER
            and not evidence_compliant
        ):
            raise ValueError(
                evidence_failure
                or "two-parent crossover source attribution was not verified"
            )
        valid, objectives, failure, detailed, resolution = await self._evaluate(frozen)
        selected = prepared.variation_case.selected_insights
        candidate = EvolutionCandidate(
            occurrence=occurrence,
            configuration=frozen,
            objectives=objectives,
            valid=valid,
            generation=prepared.plan.generation,
            label=prepared.plan.label,
            operator_kind=prepared.plan.operator_kind,
            parent_ids=tuple(parent.candidate_id for parent in prepared.plan.parents),
            common_ancestor_id=(
                None
                if prepared.plan.common_ancestor is None
                else prepared.plan.common_ancestor.candidate_id
            ),
            design_rationale=draft.design_rationale,
            failure_message=failure,
            operator_compliant=compliant,
            operator_failure=operator_failure,
            evidence_compliant=evidence_compliant,
            evidence_failure=evidence_failure,
            parent_patch_hashes=patch_hashes,
            preservation_verified=preservation,
            claimed_insight_ids=draft.claimed_insight_ids,
            selected_insight_ids=tuple(ref.insight_id.value for ref in selected),
            selected_insight_refs=selected,
            insight_assignment_kind=prepared.insight_assignment_kind,
            source_attribution=draft.source_attribution,
            conflict_resolutions=draft.conflict_resolutions,
            call_telemetry=telemetry,
            detailed_evaluation=detailed,
            objective_resolution_receipt=resolution,
        )
        self._emit(
            "candidate_evaluated",
            operator_invocation_id=prepared.operator_invocation_id.value,
            candidate_id=candidate.candidate_id.value,
            label=candidate.label,
            configuration=candidate.configuration_dict,
            objectives=candidate.objective_map,
            valid=valid,
            failure=failure,
            operator_compliant=compliant,
            operator_failure=operator_failure,
            evidence_compliant=evidence_compliant,
            evidence_failure=evidence_failure,
            preservation_verified=preservation,
            parent_patch_hashes=list(patch_hashes),
            design_rationale=draft.design_rationale,
            intended_changes=list(draft.intended_changes),
            claimed_insight_ids=list(draft.claimed_insight_ids),
            selected_insight_ids=list(candidate.selected_insight_ids),
            selected_insights=_insight_reference_records(
                candidate.selected_insight_refs
            ),
            selected_insight_records=[
                {
                    **record,
                    "assignment_kind": prepared.insight_assignment_kind.value,
                }
                for record in self.memory.prompt_records(
                    candidate.selected_insight_refs
                )
            ]
            if prepared.insight_assignment_kind is not None
            else [],
            assignment_kind=(
                None
                if prepared.insight_assignment_kind is None
                else prepared.insight_assignment_kind.value
            ),
            claimed_preservation_obligation_ids=list(
                draft.claimed_preservation_obligation_ids
            ),
            source_attribution=[
                {"path": item.path, "source": item.source}
                for item in draft.source_attribution
            ],
            conflict_resolutions=[
                {
                    "relation_id": item.relation_id,
                    "choice": item.choice,
                    "explanation": item.explanation,
                }
                for item in draft.conflict_resolutions
            ],
            mutation_response_mode=(prepared.plan.mutation_response_mode.value),
            crossover_response_mode=(prepared.plan.crossover_response_mode.value),
            proposal_representation=_proposal_representation(prepared.plan),
            atomic_submitted_path=materialization_evidence.get("atomic_submitted_path"),
            atomic_old_value_hash=materialization_evidence.get("atomic_old_value_hash"),
            atomic_new_value_hash=materialization_evidence.get("atomic_new_value_hash"),
            materialized_patch_hash=materialization_evidence.get(
                "materialized_patch_hash"
            ),
            parent_configuration_hash=materialization_evidence.get(
                "parent_configuration_hash"
            ),
            target_configuration_hash=materialization_evidence.get(
                "target_configuration_hash"
            ),
            source_attribution_provenance=materialization_evidence.get(
                "source_attribution_provenance",
                (
                    "framework_generated"
                    if prepared.plan.operator_kind is OperatorKind.REPRODUCTION
                    else (
                        "engine_materialized"
                        if prepared.proposal_authority is ProposalAuthority.ENGINE
                        else "model_authored"
                    )
                ),
            ),
            **{
                key: value
                for key, value in materialization_evidence.items()
                if key.startswith("finite_")
            },
            **{
                key: value
                for key, value in materialization_evidence.items()
                if key.startswith("crossover_")
            },
            **(
                {}
                if detailed is None
                else {"detailed_evaluation": detailed.to_record()}
            ),
            **(
                {}
                if resolution is None
                else {"objective_resolution": resolution.to_record()}
            ),
        )
        return candidate, treatment_admission

    async def _execute(
        self,
        prepared: PreparedInvocation,
    ) -> tuple[
        CandidateDraft
        | AtomicMutationDraft
        | FiniteVariationSelectionDraft
        | ExactParentCrossoverDraft,
        AgenticCallTelemetry | None,
    ]:
        plan = prepared.plan
        if plan.operator_kind is OperatorKind.REPRODUCTION:
            if prepared.proposal_authority is not ProposalAuthority.REPRODUCTION:
                raise ValueError("reproduction requires reproduction authority")
            return (
                CandidateDraft(
                    configuration=plan.parents[0].configuration_dict,
                    design_rationale="Exact reproduction control.",
                    source_attribution=(),
                ),
                None,
            )
        if prepared.proposal_authority is not ProposalAuthority.MODEL:
            raise ValueError("_execute is restricted to model-authored proposals")
        assert prepared.call_id is not None
        atomic_contract = None
        finite_contract = None
        exact_crossover_output_contract = None
        if (
            plan.mutation_response_mode
            is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
        ):
            mutation_contract = plan.mutation_contract
            if mutation_contract is None:  # pragma: no cover - plan admission.
                raise ValueError("atomic mode requires a mutation contract")
            parent_configuration = plan.parents[0].configuration
            if type(parent_configuration) is not FrozenJsonObject:
                raise TypeError("candidate root must be a FrozenJsonObject")
            atomic_contract = AtomicMutationOutputContract(
                parent_configuration=parent_configuration,
                editable_path=mutation_contract.editable_paths[0],
                replacement_options=plan.atomic_replacement_options,
            )
        elif (
            plan.mutation_response_mode
            is MutationResponseMode.FINITE_OPTION_SELECTION_V1
        ):
            finite_contract = plan.finite_variation_contract
            if finite_contract is None:  # pragma: no cover - plan admission.
                raise ValueError("finite option mode requires a sealed contract")
        if (
            plan.operator_kind is OperatorKind.TWO_PARENT_CROSSOVER
            and plan.crossover_response_mode
            is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
        ):
            crossover_contract = plan.exact_parent_crossover_contract
            if type(crossover_contract) is not ExactParentCrossoverContract:
                raise ValueError("exact crossover mode requires a sealed contract")
            exact_crossover_output_contract = ExactParentCrossoverOutputContract(
                contract_identity_sha256=(crossover_contract.contract_sha256),
                locus_ids=tuple(locus.locus_id for locus in crossover_contract.loci),
                claimable_insight_ids=tuple(
                    sorted(
                        reference.insight_id.value
                        for reference in prepared.variation_case.selected_insights
                    )
                ),
                forbidden_import_locus_sets=(plan.forbidden_exact_parent_import_sets),
            )
        max_output_tokens = self._max_output_tokens_for(
            StructuredOutputRequestKind.PROPOSAL,
            plan.operator_kind.value,
        )
        result = await self.generator.propose(
            VariationGenerationRequest(
                call_id=prepared.call_id,
                operation=plan.operator_kind.value,
                prompt=prepared.prompt,
                candidate_model=self.problem.candidate_model,
                max_output_tokens=max_output_tokens,
                temperature=self._temperature,
                atomic_mutation_contract=atomic_contract,
                finite_variation_contract=finite_contract,
                exact_parent_crossover_contract=(exact_crossover_output_contract),
            )
        )
        self._emit(
            "llm_call_completed",
            call_id=prepared.call_id.value,
            operator_invocation_id=prepared.operator_invocation_id.value,
            requested_model=result.telemetry.requested_model,
            resolved_model=result.telemetry.resolved_model,
            resolved_provider=result.telemetry.resolved_provider,
            provider_response_id=result.telemetry.provider_response_id,
            finish_reason=result.telemetry.finish_reason,
            input_tokens=result.telemetry.input_tokens,
            output_tokens=result.telemetry.output_tokens,
            reasoning_tokens=result.telemetry.reasoning_tokens,
            cost_usd=(
                None
                if result.telemetry.cost_usd is None
                else str(result.telemetry.cost_usd)
            ),
            provider_latency_ns=result.telemetry.latency_ns,
            attempt_count=result.telemetry.attempt_count,
            mutation_response_mode=plan.mutation_response_mode.value,
            crossover_response_mode=plan.crossover_response_mode.value,
            proposal_representation=_proposal_representation(plan),
        )
        return result.draft, result.telemetry

    async def run_invocations(
        self,
        plans: Sequence[InvocationPlan],
        *,
        reward_binding: RewardPolicyBinding | None = None,
    ) -> tuple[InvocationOutcome, ...]:
        prepared, active_reward = self.prepare_invocations(
            plans,
            reward_binding=reward_binding,
        )
        return await self._run_prepared_invocations(
            prepared,
            self._execute,
            proposal_failure_stage="llm",
            reward_binding=active_reward,
        )

    def prepare_invocations(
        self,
        plans: Sequence[InvocationPlan],
        *,
        reward_binding: RewardPolicyBinding | None = None,
    ) -> tuple[tuple[PreparedInvocation, ...], RewardPolicyBinding]:
        """Prepare a direct wave without starting provider/evaluator I/O.

        Readiness tooling can use this boundary on an isolated engine instance to
        freeze prompts, randomized memory assignments, and call identities.  The
        live path calls the same method immediately before execution, avoiding a
        second preparation implementation whose behavior could drift.
        """

        active_reward = self._reward_binding(reward_binding)
        values = tuple(plans)
        if any(type(plan) is not InvocationPlan for plan in values):
            raise TypeError("plans must contain exact InvocationPlan values")
        for plan in values:
            InvocationPlan.__post_init__(plan)
        resolved_ids = tuple(
            plan.resolved_insight_assignment.credit_unit_id
            for plan in values
            if plan.resolved_insight_assignment is not None
        )
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("resolved wave repeats an operator invocation ID")
        if any(
            value in self._reserved_operator_invocation_ids for value in resolved_ids
        ):
            raise ValueError("operator invocation ID was already reserved")
        # Preflight the complete resolved wave before any call/candidate ordinal
        # or credit-unit reservation can change. `_prepare` repeats the same
        # checks at the point of use, closing mutation between admission stages.
        for plan in values:
            if plan.resolved_insight_assignment is None:
                continue
            editable_paths = (
                (
                    tuple(
                        _path_text(path)
                        for path in plan.mutation_contract.editable_paths
                    )
                    if plan.mutation_contract is not None
                    else tuple(f"$.{name}" for name in plan.allowed_top_level)
                )
                if plan.operator_kind is OperatorKind.TYPED_MUTATION
                else None
            )
            self._resolved_assignment_binding(
                plan,
                editable_paths=editable_paths,
                reward_definition_hash=active_reward.definition_hash,
            )
        prepared = tuple(
            self._prepare(
                plan,
                reward_definition_hash=active_reward.definition_hash,
            )
            for plan in values
        )
        if any(
            item.proposal_authority is ProposalAuthority.ENGINE for item in prepared
        ):
            raise ValueError(
                "direct invocation preparation cannot use engine authority"
            )
        return prepared, active_reward

    async def run_materialized_invocations(
        self,
        items: Sequence[MaterializedInvocation],
        *,
        reward_binding: RewardPolicyBinding | None = None,
    ) -> tuple[InvocationOutcome, ...]:
        """Evaluate engine-authored variations without allocating an LLM call."""

        materialized = tuple(items)
        if any(type(item) is not MaterializedInvocation for item in materialized):
            raise TypeError("items must contain exact MaterializedInvocation values")
        active_reward = self._reward_binding(reward_binding)
        prepared = tuple(
            self._prepare(
                item.plan,
                proposal_authority=ProposalAuthority.ENGINE,
                materialization_policy_id=item.materialization_policy_id,
                materialization_policy_version=item.materialization_policy_version,
                materialization_receipt_hash=item.materialization_receipt_hash,
                materialized_candidate_id=item.candidate_id,
                materialized_finite_action_authority=(
                    item.materialized_finite_action_authority
                ),
                materialized_finite_action_decision=(
                    item.materialized_finite_action_decision
                ),
                reward_definition_hash=active_reward.definition_hash,
            )
            for item in materialized
        )
        drafts = {
            invocation.operator_invocation_id: item.draft
            for invocation, item in zip(prepared, materialized, strict=True)
        }

        async def supply(
            invocation: PreparedInvocation,
        ) -> tuple[CandidateDraft | AtomicMutationDraft, None]:
            return drafts[invocation.operator_invocation_id], None

        return await self._run_prepared_invocations(
            prepared,
            supply,
            proposal_failure_stage="materialization",
            reward_binding=active_reward,
        )

    def _reward_binding(
        self,
        override: RewardPolicyBinding | None,
    ) -> RewardPolicyBinding:
        if override is None:
            return self.reward_binding
        if type(override) is not RewardPolicyBinding:
            raise TypeError("reward_binding must be an exact RewardPolicyBinding")
        RewardPolicyBinding.__post_init__(override)
        return override

    async def _run_prepared_invocations(
        self,
        prepared: tuple[PreparedInvocation, ...],
        proposal,
        *,
        proposal_failure_stage: str,
        reward_binding: RewardPolicyBinding,
    ) -> tuple[InvocationOutcome, ...]:
        self._emit(
            "reward_binding_committed",
            reward_binding=reward_binding.to_record(),
            reward_binding_sha256=reward_binding.binding_sha256,
            operator_invocation_ids=[
                item.operator_invocation_id.value for item in prepared
            ],
        )
        # Execution commitment is distinct from readiness-only preparation.  Seal
        # every resolved unit in deterministic plan order before scheduling even
        # one provider call, so a crash log cannot contain a partially committed
        # wave caused by asynchronous start order.
        for item in prepared:
            resolved = item.plan.resolved_insight_assignment
            if resolved is None:
                continue
            shape_metadata = self._prompt_shape_policy_metadata
            if shape_metadata is None:  # pragma: no cover - rejected in _prepare.
                raise RuntimeError("resolved assignment has no prompt-shape policy")
            self._emit(
                "assignment_committed",
                assignment_sha256=resolved.assignment_sha256,
                assignment=resolved.to_record(),
                operator_invocation_id=item.operator_invocation_id.value,
                call_id=None if item.call_id is None else item.call_id.value,
                candidate_id=item.candidate_id.value,
                proposal_sequence=item.proposal_sequence,
                block_id=resolved.block_id,
                assignment_arm=resolved.arm.value,
                exact_context_hash=resolved.exact_context_hash,
                estimand_stratum_hash=resolved.estimand_stratum_hash,
                selection_decision_sha256=resolved.selection_decision_sha256,
                score_snapshot_sha256=resolved.score_snapshot_sha256,
                prompt_shape_sha256=resolved.prompt_shape_sha256,
                prompt_shape_commitment_verified=True,
                prompt_shape_policy={
                    "policy_id": shape_metadata[0],
                    "policy_version": shape_metadata[1],
                    "renderer_policy_id": shape_metadata[2],
                    "renderer_policy_version": shape_metadata[3],
                },
                credit_mode=resolved.credit_mode.value,
                reward_definition_hash=item.variation_case.reward_definition_hash,
                reward_binding_sha256=reward_binding.binding_sha256,
                reward_failure_score=reward_binding.failure_score,
                prepared_prompt_sha256=hashlib.sha256(
                    item.prompt.encode("utf-8")
                ).hexdigest(),
            )

        async def execute_and_evaluate(
            item: PreparedInvocation,
        ) -> tuple[
            EvolutionCandidate | None,
            Exception | None,
            str | None,
            DetailedEvaluation | None,
            TreatmentAdmissionReceipt | None,
            FiniteActionDecision | None,
        ]:
            # Keep the whole proposal-to-evaluation pipeline concurrent.  In the
            # target regime both provider calls and candidate evaluations can
            # take tens of seconds, so serializing either stage would erase the
            # wall-clock benefit of bounded parallelism.
            try:
                draft, telemetry = await proposal(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                disposition = classify_generation_failure(exc)
                failure_stage = (
                    "materialization"
                    if proposal_failure_stage == "materialization"
                    else (
                        "llm"
                        if disposition
                        is GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
                        else "infrastructure"
                    )
                )
                return None, exc, failure_stage, None, None, None
            finite_action_decision: FiniteActionDecision | None = None
            finite_authority = item.plan.finite_action_set_authority
            if finite_authority is not None:
                try:
                    if type(draft) is not FiniteVariationSelectionDraft:
                        raise TypeError(
                            "finite action authority requires an exact selection draft"
                        )
                    if type(telemetry) is not AgenticCallTelemetry:
                        raise TypeError(
                            "finite action authority requires exact model telemetry"
                        )
                    if item.call_id is None:
                        raise ValueError(
                            "finite action authority lost its logical model call"
                        )
                    finite_action_decision = seal_model_finite_action_decision(
                        authority=finite_authority,
                        call_id=item.call_id,
                        prompt_sha256=hashlib.sha256(
                            item.prompt.encode("utf-8", errors="strict")
                        ).hexdigest(),
                        draft=draft,
                        telemetry=telemetry,
                    )
                    required_claim = (finite_authority.card.reference.insight_id.value,)
                    if draft.claimed_insight_ids != required_claim:
                        raise ValueError(
                            "finite action choice did not claim its exact assigned card"
                        )
                    self._emit(
                        "finite_action_decision_sealed",
                        operator_invocation_id=item.operator_invocation_id.value,
                        call_id=item.call_id.value,
                        candidate_id=item.candidate_id.value,
                        authority_sha256=finite_authority.authority_sha256,
                        decision={
                            **finite_action_decision.to_record(),
                            "decision_sha256": (finite_action_decision.decision_sha256),
                        },
                        evaluator_entered=False,
                    )
                except (TypeError, ValueError) as exc:
                    return (
                        None,
                        exc,
                        "candidate",
                        None,
                        None,
                        finite_action_decision,
                    )
            try:
                candidate, treatment_admission = await self._candidate_from_draft(
                    item, draft, telemetry
                )
            except asyncio.CancelledError:
                raise
            except TreatmentComplianceRejected as exc:
                return (
                    None,
                    exc,
                    "treatment_noncompliance",
                    None,
                    exc.receipt,
                    finite_action_decision,
                )
            except _TerminalDetailedEvaluationError as exc:
                return (
                    None,
                    exc,
                    "infrastructure",
                    exc.evaluation,
                    None,
                    finite_action_decision,
                )
            except (TypeError, ValueError) as exc:
                return None, exc, "candidate", None, None, finite_action_decision
            except Exception as exc:
                return (
                    None,
                    exc,
                    "infrastructure",
                    None,
                    None,
                    finite_action_decision,
                )
            return (
                candidate,
                None,
                None,
                None,
                treatment_admission,
                finite_action_decision,
            )

        raw = await asyncio.gather(*(execute_and_evaluate(item) for item in prepared))
        outcomes: list[InvocationOutcome] = []
        for item, result in zip(prepared, raw, strict=True):
            (
                candidate,
                failure,
                failure_stage,
                terminal_evaluation,
                treatment_admission,
                finite_action_decision,
            ) = result
            terminal_candidate_ids = (
                [] if candidate is None else [candidate.candidate_id.value]
            )
            failure_type = None if failure is None else type(failure).__name__
            if failure_stage in {"llm", "materialization"}:
                reward = reward_binding.failure_score
                if failure_stage == "llm":
                    self._emit(
                        "llm_call_failed",
                        call_id=None if item.call_id is None else item.call_id.value,
                        operator_invocation_id=item.operator_invocation_id.value,
                        failure_type=failure_type,
                    )
                else:
                    self._emit(
                        "materialized_proposal_failed",
                        operator_invocation_id=item.operator_invocation_id.value,
                        materialization_policy_id=item.materialization_policy_id,
                        materialization_policy_version=(
                            item.materialization_policy_version
                        ),
                        materialization_receipt_hash=(
                            item.materialization_receipt_hash
                        ),
                        failure_type=failure_type,
                    )
            elif failure_stage == "candidate":
                reward = reward_binding.failure_score
                self._emit(
                    "candidate_boundary_failed",
                    operator_invocation_id=item.operator_invocation_id.value,
                    failure_type=failure_type,
                )
            elif failure_stage == "treatment_noncompliance":
                reward = reward_binding.failure_score
                assert treatment_admission is not None
                self._emit(
                    "treatment_compliance_rejected",
                    operator_invocation_id=item.operator_invocation_id.value,
                    call_id=None if item.call_id is None else item.call_id.value,
                    candidate_id=item.candidate_id.value,
                    failure_type=failure_type,
                    admission={
                        **treatment_admission.to_record(),
                        "receipt_sha256": treatment_admission.receipt_sha256,
                    },
                    evaluator_entered=treatment_admission.evaluator_entered,
                )
            elif failure_stage == "infrastructure":
                reward = reward_binding.failure_score
                self._emit(
                    "infrastructure_boundary_failed",
                    operator_invocation_id=item.operator_invocation_id.value,
                    call_id=None if item.call_id is None else item.call_id.value,
                    failure_type=failure_type,
                )
            else:
                assert candidate is not None
                try:
                    reward = float(
                        reward_binding.score(
                            candidate,
                            item.plan.parents,
                            self.objectives,
                        )
                    )
                    if not math.isfinite(reward):
                        raise ValueError("reward policy returned a non-finite value")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if candidate.detailed_evaluation is not None:
                        terminal_evaluation = candidate.detailed_evaluation
                    failure = exc
                    failure_stage = "infrastructure"
                    failure_type = type(exc).__name__
                    reward = reward_binding.failure_score
                    self._emit(
                        "infrastructure_boundary_failed",
                        operator_invocation_id=item.operator_invocation_id.value,
                        call_id=(None if item.call_id is None else item.call_id.value),
                        failure_type=failure_type,
                    )
                    # A physically evaluated occurrence is still preserved in
                    # terminal trace evidence, but a reward-policy failure must
                    # not let the optimizer archive a candidate whose scoring
                    # semantics were never established.
                    candidate = None

            parent_relations: tuple[OutcomeRelation, ...] = ()
            try:
                if (
                    failure_stage is None
                    and candidate is not None
                    and candidate.valid
                    and all(parent.valid for parent in item.plan.parents)
                ):
                    parent_relations = tuple(
                        self.compare_candidates(candidate, parent)
                        for parent in item.plan.parents
                    )
                dominates = bool(
                    self._objective_pareto_relation
                    and OutcomeRelation.BETTER in parent_relations
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - exact candidates guard this.
                if candidate is not None and candidate.detailed_evaluation is not None:
                    terminal_evaluation = candidate.detailed_evaluation
                failure = exc
                failure_stage = "infrastructure"
                failure_type = type(exc).__name__
                reward = reward_binding.failure_score
                dominates = False
                parent_relations = ()
                candidate = None
                self._emit(
                    "infrastructure_boundary_failed",
                    operator_invocation_id=item.operator_invocation_id.value,
                    call_id=None if item.call_id is None else item.call_id.value,
                    failure_type=failure_type,
                )
            better = failure_stage is None and reward > 0
            decision = item.selection_decision
            credit_status = (
                "test_only_no_retrieval_credit"
                if item.insight_assignment_kind is InsightAssignmentKind.QUARANTINE_TEST
                else "not_assigned"
            )
            if decision is not None:
                if item.plan.resolved_insight_assignment is not None:
                    resolved = item.plan.resolved_insight_assignment
                    is_diagnostic = resolved.arm is MemoryAssignmentArm.DIAGNOSTIC
                    credit_status = (
                        "deferred_wave_sealed_itt"
                        if is_diagnostic
                        else "resolved_arm_outcome_no_memory_update"
                    )
                    self._emit(
                        (
                            "insight_credit_deferred"
                            if is_diagnostic
                            else "resolved_memory_arm_outcome_deferred"
                        ),
                        operator_invocation_id=item.operator_invocation_id.value,
                        assignment_sha256=resolved.assignment_sha256,
                        block_id=resolved.block_id,
                        assignment_arm=resolved.arm.value,
                        exact_context_hash=resolved.exact_context_hash,
                        estimand_stratum_hash=resolved.estimand_stratum_hash,
                        reward=reward,
                        reward_definition_hash=(
                            item.variation_case.reward_definition_hash
                        ),
                        failure_stage=failure_stage,
                        failure_type=failure_type,
                        candidate_ids=terminal_candidate_ids,
                    )
                elif failure_stage in {"llm", "infrastructure"}:
                    credit_status = (
                        "censored_call_failure"
                        if failure_stage == "llm"
                        else "censored_infrastructure_failure"
                    )
                    self._emit(
                        "insight_credit_censored",
                        operator_invocation_id=item.operator_invocation_id.value,
                        context_hash=decision.context_hash,
                        failure_stage=failure_stage,
                        failure_type=failure_type,
                    )
                else:
                    candidate_ids = (
                        () if candidate is None else (candidate.candidate_id,)
                    )
                    self.memory.record_trial(
                        credit_unit_id=item.operator_invocation_id,
                        candidate_ids=candidate_ids,
                        reward_definition_hash=(
                            item.variation_case.reward_definition_hash
                        ),
                        decision=decision,
                        reward=float(reward),
                    )
                    credit_status = "recorded"
                    self._emit(
                        "insight_credit_updated",
                        operator_invocation_id=item.operator_invocation_id.value,
                        context_hash=decision.context_hash,
                        reward=reward,
                        score_evidence=list(
                            self.memory.score_evidence(decision.context_hash)
                        ),
                    )

            outcome = InvocationOutcome(
                prepared=item,
                candidate=candidate,
                reward=reward,
                call_failure_type=failure_type,
                failure_stage=failure_stage,
                dominates_any_parent=dominates,
                better_than_any_parent=better,
                terminal_evaluation=terminal_evaluation,
                parent_relations=parent_relations,
                treatment_admission_receipt=treatment_admission,
                finite_action_decision=finite_action_decision,
            )
            outcomes.append(outcome)
            resolved = item.plan.resolved_insight_assignment
            if resolved is not None:
                if failure_stage is None:
                    terminal_status = "succeeded"
                    reward_disposition = "observed"
                    observed_reward: float | None = reward
                elif failure_stage == "llm":
                    terminal_status = "model_or_schema_failure"
                    reward_disposition = "impute_wave_no_yield_at_seal"
                    observed_reward = None
                elif failure_stage == "candidate":
                    terminal_status = "candidate_failure"
                    reward_disposition = "impute_wave_no_yield_at_seal"
                    observed_reward = None
                elif failure_stage == "treatment_noncompliance":
                    terminal_status = "treatment_noncompliance"
                    reward_disposition = "impute_wave_no_yield_at_seal"
                    observed_reward = None
                else:
                    # Resolved assignments cannot use engine materialization, so
                    # any other terminal stage invalidates the causal block.
                    terminal_status = "infrastructure_failure"
                    reward_disposition = "invalidates_block"
                    observed_reward = None
                self._emit(
                    "trial_terminal",
                    assignment_sha256=resolved.assignment_sha256,
                    block_id=resolved.block_id,
                    assignment_arm=resolved.arm.value,
                    operator_invocation_id=item.operator_invocation_id.value,
                    call_id=None if item.call_id is None else item.call_id.value,
                    candidate_ids=terminal_candidate_ids,
                    terminal_status=terminal_status,
                    failure_stage=failure_stage,
                    failure_type=failure_type,
                    observed_reward=observed_reward,
                    engine_terminal_reward=reward,
                    reward_definition_hash=item.variation_case.reward_definition_hash,
                    reward_disposition=reward_disposition,
                    credit_mode=resolved.credit_mode.value,
                )
            self._emit(
                "invocation_completed",
                operator_invocation_id=item.operator_invocation_id.value,
                call_id=None if item.call_id is None else item.call_id.value,
                operator_kind=item.plan.operator_kind.value,
                proposal_authority=item.proposal_authority.value,
                materialization_policy_id=item.materialization_policy_id,
                materialization_policy_version=(item.materialization_policy_version),
                materialization_receipt_hash=(item.materialization_receipt_hash),
                parent_ids=[parent.candidate_id.value for parent in item.plan.parents],
                candidate_id=(
                    None if candidate is None else candidate.candidate_id.value
                ),
                valid=None if candidate is None else candidate.valid,
                operator_compliant=(
                    None if candidate is None else candidate.operator_compliant
                ),
                evidence_compliant=(
                    None if candidate is None else candidate.evidence_compliant
                ),
                evidence_failure=(
                    None if candidate is None else candidate.evidence_failure
                ),
                scalar_reward=reward,
                scalar_reward_definition_sha256=(
                    item.variation_case.reward_definition_hash
                ),
                scalar_reward_binding_sha256=reward_binding.binding_sha256,
                scalar_reward_failure_score=reward_binding.failure_score,
                dominates_any_parent=dominates,
                positive_scalar_reward=better,
                selected_insight_ids=[
                    reference.insight_id.value
                    for reference in item.variation_case.selected_insights
                ],
                selected_insights=_insight_reference_records(
                    item.variation_case.selected_insights
                ),
                assignment_kind=(
                    None
                    if item.insight_assignment_kind is None
                    else item.insight_assignment_kind.value
                ),
                insight_credit_status=credit_status,
                failure_stage=failure_stage,
                failure_type=failure_type,
                treatment_admission=(
                    None
                    if treatment_admission is None
                    else {
                        **treatment_admission.to_record(),
                        "receipt_sha256": treatment_admission.receipt_sha256,
                    }
                ),
                finite_action_decision=(
                    None
                    if finite_action_decision is None
                    else {
                        **finite_action_decision.to_record(),
                        "decision_sha256": finite_action_decision.decision_sha256,
                    }
                ),
                materialized_finite_action_authority=(
                    None
                    if item.materialized_finite_action_authority is None
                    else {
                        **item.materialized_finite_action_authority.to_record(),
                        "authority_sha256": (
                            item.materialized_finite_action_authority.authority_sha256
                        ),
                    }
                ),
                materialized_finite_action_decision=(
                    None
                    if item.materialized_finite_action_decision is None
                    else {
                        **item.materialized_finite_action_decision.to_record(),
                        "decision_sha256": (
                            item.materialized_finite_action_decision.decision_sha256
                        ),
                    }
                ),
                **(
                    {}
                    if not self._detailed_evaluation_enabled
                    else {
                        "outcome_relation_policy": (
                            self.outcome_relation_binding.to_record()
                        ),
                        "parent_outcome_relations": [
                            {
                                "parent_candidate_id": parent.candidate_id.value,
                                "candidate_relation": relation.value,
                            }
                            for parent, relation in zip(
                                (item.plan.parents if parent_relations else ()),
                                parent_relations,
                                strict=True,
                            )
                        ],
                        "better_relation_any_parent": (
                            OutcomeRelation.BETTER in parent_relations
                        ),
                        "detailed_evaluation": (
                            None
                            if outcome.detailed_evaluation is None
                            else outcome.detailed_evaluation.to_record()
                        ),
                    }
                ),
            )
        return tuple(outcomes)

    def _publish_sharded_reflection(
        self,
        *,
        workflow_result: ReflectionWorkflowResult,
        contrast_lineage: Mapping[
            str,
            tuple[OperatorInvocationId, tuple[CandidateId, CandidateId]],
        ],
        contrast_action_bindings: Mapping[
            str,
            FiniteActionEvidenceBinding,
        ],
        outcomes: Sequence[InvocationOutcome],
        label: str,
        insight_contract: ReflectionInsightContract | None,
    ) -> tuple[InsightMemoryEntry, ...]:
        """Atomically admit one already-complete contrast-sharded batch."""

        if type(workflow_result) is not ReflectionWorkflowResult:
            raise TypeError("workflow_result must be an exact ReflectionWorkflowResult")
        ReflectionWorkflowResult.__post_init__(workflow_result)
        expected_contrasts = tuple(sorted(contrast_lineage))
        returned_contrasts = tuple(
            shard.contrast_id for shard in workflow_result.shards
        )
        if returned_contrasts != expected_contrasts:
            raise RuntimeError(
                "reflection workflow result differs from the engine contrast boundary"
            )
        evidence_operator_kinds = tuple(
            sorted({outcome.prepared.plan.operator_kind.value for outcome in outcomes})
        )
        staged_items: list[ReflectedInsightBatchItem] = []
        for shard in workflow_result.shards:
            draft = shard.draft
            if insight_contract is not None:
                validate_reflection_insight_draft(draft, insight_contract)
            _validate_reflected_action_origin(
                draft,
                insight_contract,
                (shard.contrast_id,),
                contrast_action_bindings,
            )
            operator_id, candidate_ids = contrast_lineage[shard.contrast_id]
            staged_items.append(
                ReflectedInsightBatchItem(
                    draft=draft,
                    evidence_lineage=InsightEvidenceLineage(
                        reflection_call_id=shard.call_id,
                        source_operator_invocation_ids=(operator_id,),
                        source_candidate_ids=tuple(sorted(candidate_ids)),
                        available_contrast_ids=(shard.contrast_id,),
                        cited_contrast_ids=(shard.contrast_id,),
                        finite_action_bindings=(
                            _finite_action_evidence_for_citations(
                                (shard.contrast_id,),
                                contrast_action_bindings,
                            )
                        ),
                    ),
                )
            )
        added = self.memory.add_reflection_batch(
            tuple(staged_items),
            initial_score=0.0,
            applicable_operator_kinds=evidence_operator_kinds,
        )
        entries_by_contrast = {
            entry.evidence_lineage.cited_contrast_ids[0]: entry
            for entry in added
            if entry.evidence_lineage is not None
        }
        for shard in workflow_result.shards:
            telemetry = shard.generation_result.telemetry
            entry = entries_by_contrast[shard.contrast_id]
            self._emit(
                "reflection_completed",
                call_id=shard.call_id.value,
                label=label,
                reflection_workflow_policy_id=getattr(
                    self._reflection_workflow,
                    "policy_id",
                    type(self._reflection_workflow).__name__,
                ),
                requested_model=telemetry.requested_model,
                resolved_model=telemetry.resolved_model,
                resolved_provider=telemetry.resolved_provider,
                provider_response_id=telemetry.provider_response_id,
                finish_reason=telemetry.finish_reason,
                input_tokens=telemetry.input_tokens,
                output_tokens=telemetry.output_tokens,
                reasoning_tokens=telemetry.reasoning_tokens,
                cost_usd=(
                    None if telemetry.cost_usd is None else str(telemetry.cost_usd)
                ),
                provider_latency_ns=telemetry.latency_ns,
                attempt_count=telemetry.attempt_count,
                **(
                    {}
                    if insight_contract is None
                    else {"insight_contract": insight_contract.to_record()}
                ),
                insights=[
                    {
                        "insight_id": entry.reference.insight_id.value,
                        "version": entry.reference.version,
                        "claim": entry.draft.claim,
                        "trigger": entry.draft.trigger,
                        "mechanism": entry.draft.mechanism,
                        "affected_paths": list(entry.draft.affected_paths),
                        "evidence_summary": entry.draft.evidence_summary,
                        "evidence_contrast_ids": list(
                            entry.draft.evidence_contrast_ids
                        ),
                        "confidence": entry.draft.confidence,
                        "lifecycle_state": entry.lifecycle_state.value,
                        "retrievable": entry.retrievable,
                        "origin": entry.origin.value,
                        "applicable_operator_kinds": list(
                            entry.applicable_operator_kinds
                        ),
                        **(entry.draft.intervention_record() or {}),
                        "evidence_lineage": entry.evidence_lineage.to_record(),
                    }
                ],
            )
        self._emit(
            "reflection_batch_completed",
            label=label,
            reflection_workflow_policy_id=getattr(
                self._reflection_workflow,
                "policy_id",
                type(self._reflection_workflow).__name__,
            ),
            logical_llm_calls_used=workflow_result.logical_llm_calls_used,
            call_ids=[call_id.value for call_id in workflow_result.call_ids],
            contrast_ids=list(returned_contrasts),
            insight_count=len(added),
        )
        return added

    async def _reflect_entries(
        self,
        outcomes: Sequence[InvocationOutcome],
        *,
        label: str,
        max_insights: int = 4,
        min_insights: int = 0,
        insight_contract: ReflectionInsightContract | None = None,
        revision_predecessors: tuple[InsightRef, ...] = (),
        source_receipt_sha256s: tuple[str, ...] = (),
    ) -> tuple[InsightMemoryEntry, ...]:
        if type(max_insights) is not int or not 1 <= max_insights <= 16:
            raise ValueError("max_insights must lie in [1,16]")
        if type(min_insights) is not int or not 0 <= min_insights <= max_insights:
            raise ValueError("min_insights must lie in [0,max_insights]")
        if type(revision_predecessors) is not tuple or any(
            type(value) is not InsightRef for value in revision_predecessors
        ):
            raise TypeError(
                "revision_predecessors must be an exact tuple of InsightRef values"
            )
        if len(set(revision_predecessors)) != len(revision_predecessors):
            raise ValueError("revision_predecessors cannot repeat")
        if type(source_receipt_sha256s) is not tuple:
            raise TypeError("source_receipt_sha256s must be an exact tuple")
        for value in source_receipt_sha256s:
            require_sha256(value, "reflection source receipt SHA-256")
        revision_predecessor_entries: tuple[InsightMemoryEntry, ...] = ()
        if revision_predecessors:
            if len(revision_predecessors) != 1 or max_insights != 1:
                raise ValueError(
                    "the atomic revision path supports exactly one frozen target"
                )
            if self._reflection_workflow is not None:
                raise ValueError(
                    "revision publication currently requires one batched reflection"
                )
            # Resolve exact owned versions before allocating a provider call.
            revision_predecessor_entries = self.memory.entries_for(
                revision_predecessors
            )
        if insight_contract is not None:
            if type(insight_contract) is not ReflectionInsightContract:
                raise TypeError(
                    "insight_contract must be an exact ReflectionInsightContract"
                )
            ReflectionInsightContract.__post_init__(insight_contract)
        rows = []
        shard_rows_by_contrast: dict[str, dict[str, object]] = {}
        available_contrast_ids: set[str] = set()
        contrast_lineage: dict[
            str,
            tuple[OperatorInvocationId, tuple[CandidateId, CandidateId]],
        ] = {}
        contrast_action_bindings: dict[
            str,
            FiniteActionEvidenceBinding,
        ] = {}
        for outcome in outcomes:
            if type(outcome) is not InvocationOutcome:
                raise TypeError("reflection requires exact InvocationOutcome values")
            candidate = outcome.candidate
            contrasts = []
            if candidate is not None:
                for parent_index, parent in enumerate(outcome.prepared.plan.parents):
                    patch = derive_patch(
                        parent.configuration,
                        candidate.configuration,
                        base_candidate_id=parent.candidate_id,
                        target_candidate_id=candidate.candidate_id,
                    )
                    contrast_id = hashlib.sha256(
                        b"agent-evolve:reflection-contrast:v1\x00"
                        + outcome.prepared.operator_invocation_id.value.encode("ascii")
                        + b"\x00"
                        + parent.candidate_id.value.encode("ascii")
                    ).hexdigest()
                    available_contrast_ids.add(contrast_id)
                    contrast_lineage[contrast_id] = (
                        outcome.prepared.operator_invocation_id,
                        (parent.candidate_id, candidate.candidate_id),
                    )
                    contrast_record: dict[str, object] = {
                        "contrast_id": contrast_id,
                        "parent_candidate_id": parent.candidate_id.value,
                        "child_candidate_id": candidate.candidate_id.value,
                        "parent_configuration_hash": (
                            parent.occurrence.configuration_hash
                        ),
                        "child_configuration_hash": (
                            candidate.occurrence.configuration_hash
                        ),
                        "derived_patch_hash": patch.patch_hash,
                        "changed_paths": [
                            _path_text(operation.path) for operation in patch.operations
                        ],
                        "system_derived_operations": [
                            _reflection_operation_projection(operation)
                            for operation in patch.operations
                        ],
                        "patch_operation_count": len(patch.operations),
                        "contrast_scope": (
                            "no_change"
                            if not patch.operations
                            else (
                                "single_operation"
                                if len(patch.operations) == 1
                                else "joint_intervention"
                            )
                        ),
                        "objective_deltas_child_minus_parent": {
                            spec.name: (
                                candidate.objective_map[spec.name]
                                - parent.objective_map[spec.name]
                            )
                            for spec in self.objectives
                            if candidate.valid and parent.valid
                        },
                        "directional_improvements": {
                            spec.name: (
                                (
                                    candidate.objective_map[spec.name]
                                    - parent.objective_map[spec.name]
                                )
                                * (1.0 if spec.goal == "max" else -1.0)
                            )
                            for spec in self.objectives
                            if candidate.valid and parent.valid
                        },
                    }
                    finite_contract = outcome.prepared.plan.finite_variation_contract
                    materialized_option_id: str | None = None
                    if finite_contract is None:
                        materialized_authority = (
                            outcome.prepared.materialized_finite_action_authority
                        )
                        materialized_decision = (
                            outcome.prepared.materialized_finite_action_decision
                        )
                        if (materialized_authority is None) != (
                            materialized_decision is None
                        ):
                            raise RuntimeError(
                                "prepared materialized finite action provenance "
                                "was split"
                            )
                        if materialized_authority is not None:
                            assert materialized_decision is not None
                            validate_finite_action_decision(
                                materialized_authority,
                                materialized_decision,
                            )
                            if (
                                materialized_decision.selector_kind
                                is not FiniteActionSelectorKind.ENGINE
                                or materialized_decision.child_configuration_sha256
                                != candidate.occurrence.configuration_hash
                            ):
                                raise RuntimeError(
                                    "materialized finite action reflection provenance "
                                    "differs from the evaluated child"
                                )
                            finite_contract = (
                                materialized_authority.support.support_contract
                            )
                            materialized_option_id = materialized_decision.option_id
                    if finite_contract is not None:
                        if materialized_option_id is None:
                            matching_options = tuple(
                                option
                                for option in finite_contract.options
                                if typed_json_equal(
                                    option.child_configuration,
                                    candidate.configuration,
                                )
                            )
                            if len(matching_options) != 1:
                                raise RuntimeError(
                                    "finite-plan reflection could not attribute the "
                                    "child to exactly one sealed option"
                                )
                            matched_option = matching_options[0]
                        else:
                            matched_option = finite_contract.resolve(
                                materialized_option_id
                            )
                            if not typed_json_equal(
                                matched_option.child_configuration,
                                candidate.configuration,
                            ):
                                raise RuntimeError(
                                    "materialized finite action option differs from "
                                    "the evaluated child"
                                )
                        action_binding = bind_finite_action_evidence(
                            contrast_id=contrast_id,
                            contract=finite_contract,
                            option_id=matched_option.option_id,
                        )
                        contrast_action_bindings[contrast_id] = action_binding
                        contrast_record["finite_variation_option"] = (
                            action_binding.finite_option_record()
                        )
                    relation = (
                        None
                        if not outcome.parent_relations
                        else outcome.parent_relations[parent_index]
                    )
                    if self._objective_pareto_relation:
                        contrast_record["child_dominates_parent"] = (
                            _dominates(candidate, parent, self.objectives)
                            if relation is None
                            else relation is OutcomeRelation.BETTER
                        )
                    else:
                        contrast_record["child_outcome_relation"] = (
                            None if relation is None else relation.value
                        )
                        contrast_record["outcome_relation_policy"] = (
                            self.outcome_relation_binding.to_record()
                        )
                    contrasts.append(contrast_record)
            row: dict[str, object] = {
                "operator_invocation_id": (
                    outcome.prepared.operator_invocation_id.value
                ),
                "operator": outcome.prepared.plan.operator_kind.value,
                "parents": [
                    _candidate_evidence(parent)
                    for parent in outcome.prepared.plan.parents
                ],
                "candidate": (
                    None
                    if candidate is None
                    else {
                        **_candidate_evidence(candidate),
                        "operator_failure": candidate.operator_failure,
                        "design_rationale": candidate.design_rationale,
                        "selected_insight_ids": list(candidate.selected_insight_ids),
                        "claimed_insight_ids": list(candidate.claimed_insight_ids),
                    }
                ),
                "scalar_reward": outcome.reward,
                "scalar_reward_definition_sha256": (
                    outcome.prepared.variation_case.reward_definition_hash
                ),
                "positive_scalar_reward": outcome.better_than_any_parent,
                "call_failure_type": outcome.call_failure_type,
                "machine_derived_contrasts": contrasts,
            }
            if self._objective_pareto_relation:
                row["dominates_any_parent"] = outcome.dominates_any_parent
            else:
                row["parent_outcome_relations"] = [
                    relation.value for relation in outcome.parent_relations
                ]
                row["better_relation_any_parent"] = (
                    OutcomeRelation.BETTER in outcome.parent_relations
                )
                row["outcome_relation_policy"] = (
                    self.outcome_relation_binding.to_record()
                )
            if candidate is None and outcome.terminal_evaluation is not None:
                row["terminal_detailed_evaluation"] = (
                    outcome.terminal_evaluation.to_record()
                )
            projection = self._reflection_row_projection
            if projection is not None:
                frozen_original = freeze_json(row)
                if type(frozen_original) is not FrozenJsonObject:
                    raise TypeError("reflection evidence row must be a JSON object")
                detached = thaw_json(frozen_original)
                projected = projection.project(detached)
                if type(projected) is not dict:
                    raise TypeError(
                        "reflection row projection must return an exact dict"
                    )
                original = thaw_json(frozen_original)
                if projected.get("operator_invocation_id") != original.get(
                    "operator_invocation_id"
                ):
                    raise ValueError(
                        "reflection projection changed invocation identity"
                    )
                if projected.get("machine_derived_contrasts") != original.get(
                    "machine_derived_contrasts"
                ):
                    raise ValueError(
                        "reflection projection changed machine-derived contrasts"
                    )
                frozen_projection = freeze_json(projected)
                if type(frozen_projection) is not FrozenJsonObject:
                    raise TypeError("reflection projection must remain a JSON object")
                row = thaw_json(frozen_projection)
            projected_contrasts = row.get("machine_derived_contrasts")
            if type(projected_contrasts) is not list:
                raise TypeError(
                    "reflection evidence row must retain machine-derived contrasts"
                )
            for contrast_index, projected_contrast in enumerate(projected_contrasts):
                if type(projected_contrast) is not dict:
                    raise TypeError("reflection contrast projection must be an object")
                contrast_id = projected_contrast.get("contrast_id")
                if type(contrast_id) is not str:
                    raise TypeError("reflection contrast must retain its identity")
                frozen_shard = freeze_json(row)
                if type(frozen_shard) is not FrozenJsonObject:
                    raise TypeError("reflection shard row must be a JSON object")
                shard_row = thaw_json(frozen_shard)
                shard_row["machine_derived_contrasts"] = [projected_contrast]
                projected_parents = shard_row.get("parents")
                if type(projected_parents) is list and len(projected_parents) == len(
                    projected_contrasts
                ):
                    shard_row["parents"] = [projected_parents[contrast_index]]
                projected_relations = shard_row.get("parent_outcome_relations")
                if type(projected_relations) is list and len(
                    projected_relations
                ) == len(projected_contrasts):
                    relation = projected_relations[contrast_index]
                    shard_row["parent_outcome_relations"] = [relation]
                    shard_row["better_relation_any_parent"] = relation == "better"
                if "child_dominates_parent" in projected_contrast:
                    shard_row["dominates_any_parent"] = projected_contrast[
                        "child_dominates_parent"
                    ]
                if contrast_id in shard_rows_by_contrast:
                    raise RuntimeError("reflection contrast identity was duplicated")
                shard_rows_by_contrast[contrast_id] = shard_row
            rows.append(row)
        canonical_contrast_ids = tuple(sorted(available_contrast_ids))
        quality_instruction = (
            "The scalar reward is not a Pareto-dominance claim; use the explicit dominance and validity fields when describing quality."
            if self._objective_pareto_relation
            else "The scalar reward is separate from the injected outcome relation; use child_outcome_relation, detailed evaluation evidence, and validity fields when describing quality. Never rename a generic BETTER relation as Pareto dominance."
        )
        prompt_sections = [
            "Extract a small set of falsifiable optimization insights from the evaluated evidence below.",
            "Each insight must state a conditional trigger, a mechanism, affected JSON paths, and the exact evidence. "
            "Do not restate a candidate, invent unobserved causality, or give generic advice. Counterexamples should lower confidence. "
            + quality_instruction,
            "Use the machine_derived_contrasts as the evidence boundary. A single_operation contrast may support a one-operation effect hypothesis. "
            "A no_change contrast is an abstention/control, not evidence that an unexecuted edit caused an outcome. A joint_intervention contrast supports only the joint association: do not assign its outcome to one coordinate or invent a benefit for a coordinate without an ablation. "
            "Put every supporting full 64-character contrast_id in evidence_contrast_ids; use evidence_summary only for a human-readable account of that evidence. Every affected path must be canonical and begin with $., and duplicated claims should be consolidated instead of re-added. "
            "For a multi-operation association, state a concrete one-coordinate falsification/ablation in the trigger or evidence summary.",
        ]
        if revision_predecessors:
            prompt_sections.extend(
                [
                    "",
                    "FROZEN REVISION TARGETS",
                    _json(self.memory.prompt_records(revision_predecessors)),
                    "Every returned card is a quarantined semantic revision of "
                    "the target at the same array position. Preserve useful "
                    "scope, correct the mechanism/action using only the sealed "
                    "trace, and do not claim inherited support or score. The "
                    "system binds the revises relation and a fresh zero prior.",
                ]
            )
        if self._reflection_row_projection is not None:
            prompt_sections.extend(
                [
                    "",
                    "REFLECTION EVIDENCE PROJECTION",
                    _json(self._reflection_row_projection.to_record()),
                ]
            )
        if insight_contract is not None:
            exact_action_instruction = (
                " Also emit recommended_option_ids containing exactly the "
                "finite option ID actually executed in each cited contrast; "
                "for a singleton shard this is exactly one ID. Do not substitute "
                "another option from the same family."
                if insight_contract.allowed_option_ids
                else ""
            )
            prompt_sections.extend(
                [
                    "Every insight must also be an actionable, falsifiable intervention card. Predict the numeric direction of every required metric exactly once using only decrease, increase, unchanged, or unknown. Recommend at least one allowed finite option family, give a concrete action template over those families, and state an explicit held-out falsification condition. At least one metric prediction must be directional rather than unknown."
                    + exact_action_instruction,
                    "",
                    "REFLECTION INSIGHT CONTRACT",
                    _json(insight_contract.to_record()),
                ]
            )

        def render_reflection_prompt(
            evidence_rows: Sequence[Mapping[str, object]],
            *,
            lower: int,
            upper: int,
        ) -> str:
            cardinality_instruction = (
                f"Return at most {upper} insights."
                if lower == 0
                else (
                    f"Return exactly {upper} {'insight' if upper == 1 else 'insights'}."
                    if lower == upper
                    else f"Return between {lower} and {upper} insights."
                )
            )
            return self._validate_optimization_semantics_prompt(
                "\n".join(
                    [
                        *prompt_sections,
                        "",
                        "PROBLEM",
                        self.problem_description,
                        "",
                        "EVALUATED TRACE",
                        _json(evidence_rows),
                        "",
                        cardinality_instruction,
                    ]
                )
            )

        prompt = render_reflection_prompt(
            rows,
            lower=min_insights,
            upper=max_insights,
        )
        reflection_operation = "extract_insights"
        max_output_tokens = self._max_output_tokens_for(
            StructuredOutputRequestKind.REFLECTION,
            reflection_operation,
        )
        reflection_request = ReflectionCallRequest(
            label=label,
            operation=reflection_operation,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            min_insights=min_insights,
            max_insights=max_insights,
            max_output_tokens=max_output_tokens,
            temperature=self._temperature,
            insight_contract_sha256=(
                None if insight_contract is None else insight_contract.identity_sha256
            ),
            revision_predecessors=revision_predecessors,
            revision_predecessor_content_sha256s=tuple(
                entry.draft.content_sha256 for entry in revision_predecessor_entries
            ),
            source_receipt_sha256s=source_receipt_sha256s,
            source_operator_invocation_ids=tuple(
                outcome.prepared.operator_invocation_id for outcome in outcomes
            ),
            source_outcome_sha256s=tuple(
                hashlib.sha256(
                    _REFLECTION_SOURCE_OUTCOME_DOMAIN
                    + canonical_typed_json_bytes(freeze_json(row))
                ).hexdigest()
                for row in rows
            ),
            available_contrast_ids=canonical_contrast_ids,
        )
        if self._reflection_workflow is not None:
            if not (min_insights <= len(canonical_contrast_ids) <= max_insights):
                raise ValueError(
                    "contrast-sharded cardinality falls outside the requested "
                    "insight interval"
                )
            expected_ids = set(canonical_contrast_ids)
            if set(shard_rows_by_contrast) != expected_ids:
                raise RuntimeError(
                    "reflection shard rows differ from the engine contrast boundary"
                )
            workflow_request = ReflectionWorkflowRequest(
                operation=reflection_operation,
                shards=tuple(
                    ReflectionPromptShard(
                        contrast_id=contrast_id,
                        prompt=render_reflection_prompt(
                            (shard_rows_by_contrast[contrast_id],),
                            lower=1,
                            upper=1,
                        ),
                    )
                    for contrast_id in canonical_contrast_ids
                ),
                max_output_tokens=max_output_tokens,
                temperature=self._temperature,
                insight_contract=insight_contract,
                batch_prompt=prompt,
            )

            def reflection_call_planned(
                call: PlannedReflectionCall | PlannedReflectionBatchCall,
            ) -> None:
                request = call.request
                self._emit(
                    "reflection_requested",
                    call_id=call.call_id.value,
                    label=label,
                    reflection_workflow_policy_id=getattr(
                        self._reflection_workflow,
                        "policy_id",
                        type(self._reflection_workflow).__name__,
                    ),
                    reflection_workflow_policy_version=getattr(
                        self._reflection_workflow,
                        "policy_version",
                        None,
                    ),
                    prompt=request.prompt,
                    prompt_sha256=hashlib.sha256(
                        request.prompt.encode("utf-8")
                    ).hexdigest(),
                    available_contrast_ids=list(request.available_contrast_ids),
                    **(
                        {}
                        if self.optimization_semantics_record is None
                        else {
                            "optimization_semantics": (
                                self.optimization_semantics_record
                            )
                        }
                    ),
                    **(
                        {}
                        if self._reflection_row_projection is None
                        else {
                            "reflection_row_projection": (
                                self._reflection_row_projection.to_record()
                            )
                        }
                    ),
                    **(
                        {}
                        if insight_contract is None
                        else {"insight_contract": insight_contract.to_record()}
                    ),
                )

            try:
                workflow_result = await self._reflection_workflow.run(
                    workflow_request,
                    generator=self.generator,
                    id_factory=self.ids,
                    call_planned_sink=reflection_call_planned,
                )
            except Exception as exc:
                self._emit(
                    "reflection_failed",
                    label=label,
                    failure_type=type(exc).__name__,
                    reflection_workflow_policy_id=getattr(
                        self._reflection_workflow,
                        "policy_id",
                        type(self._reflection_workflow).__name__,
                    ),
                )
                raise
            return self._publish_sharded_reflection(
                workflow_result=workflow_result,
                contrast_lineage=contrast_lineage,
                contrast_action_bindings=contrast_action_bindings,
                outcomes=outcomes,
                label=label,
                insight_contract=insight_contract,
            )

        call_id = self.ids.new_llm_call_id()
        self._emit(
            "reflection_requested",
            call_id=call_id.value,
            label=label,
            prompt=prompt,
            prompt_sha256=reflection_request.prompt_sha256,
            reflection_request_sha256=reflection_request.request_sha256,
            source_receipt_sha256s=list(source_receipt_sha256s),
            available_contrast_ids=list(canonical_contrast_ids),
            **(
                {}
                if self.optimization_semantics_record is None
                else {"optimization_semantics": self.optimization_semantics_record}
            ),
            **(
                {}
                if self._reflection_row_projection is None
                else {
                    "reflection_row_projection": (
                        self._reflection_row_projection.to_record()
                    )
                }
            ),
            **(
                {}
                if insight_contract is None
                else {"insight_contract": insight_contract.to_record()}
            ),
        )
        observed_telemetry: AgenticCallTelemetry | None = None
        try:
            result = await self.generator.reflect(
                ReflectionGenerationRequest(
                    call_id=call_id,
                    operation=reflection_operation,
                    prompt=prompt,
                    max_insights=max_insights,
                    min_insights=min_insights,
                    max_output_tokens=max_output_tokens,
                    temperature=self._temperature,
                    available_contrast_ids=canonical_contrast_ids,
                    insight_contract=insight_contract,
                )
            )
            if type(result) is not ReflectionGenerationResult:
                raise TypeError(
                    "reflection generator must return an exact "
                    "ReflectionGenerationResult"
                )
            if type(result.insights) is not tuple or any(
                type(value) is not InsightDraft for value in result.insights
            ):
                raise TypeError(
                    "reflection result insights must be an exact tuple of "
                    "InsightDraft values"
                )
            for draft in result.insights:
                InsightDraft.__post_init__(draft)
            if type(result.telemetry) is not AgenticCallTelemetry:
                raise TypeError(
                    "reflection result telemetry must be exact AgenticCallTelemetry"
                )
            AgenticCallTelemetry.__post_init__(result.telemetry)
            observed_telemetry = result.telemetry
            if not min_insights <= len(result.insights) <= max_insights:
                raise ReflectionCardContractError(
                    "reflection result violates its requested cardinality"
                )
        except Exception as exc:
            failure_receipt = ReflectionCallReceipt(
                call_id=call_id,
                request=reflection_request,
                status=ReflectionCallStatus.FAILED,
                telemetry=observed_telemetry,
                telemetry_sha256=None,
                failure_type=type(exc).__name__,
            )
            self._record_reflection_call_receipt(failure_receipt)
            self._emit(
                "reflection_failed",
                call_id=call_id.value,
                failure_type=type(exc).__name__,
                reflection_call_receipt_sha256=(failure_receipt.receipt_sha256),
            )
            # Queue-owned retries are already exhausted at this boundary.  The
            # typed wrapper preserves that one logical call was consumed while
            # allowing a postseal curation policy to isolate this failure from
            # already-valid optimization endpoints.
            raise ReflectionCallExecutionError(
                call_id,
                exc,
                failure_receipt,
            ) from exc
        # Model confidence is an annotation, not evidence of downstream utility.
        # Reflected hypotheses enter quarantine neutrally and cannot be retrieved
        # until a separate validation step records an explicit promotion.
        evidence_operator_kinds = tuple(
            sorted({outcome.prepared.plan.operator_kind.value for outcome in outcomes})
        )
        added_entries: list[InsightMemoryEntry] = []
        rejected_insight_count = 0
        for draft_index, draft in enumerate(result.insights):
            if insight_contract is not None:
                try:
                    validate_reflection_insight_draft(
                        draft,
                        insight_contract,
                    )
                except (TypeError, ValueError) as exc:
                    self._emit(
                        "reflection_insight_rejected",
                        call_id=call_id.value,
                        claim_sha256=hashlib.sha256(
                            draft.claim.encode("utf-8")
                        ).hexdigest(),
                        reason="advanced_insight_contract_violation",
                        contract_error=type(exc).__name__,
                    )
                    rejected_insight_count += 1
                    continue
            submitted_contrast_ids = draft.evidence_contrast_ids
            cited_contrast_ids = tuple(
                contrast_id
                for contrast_id in submitted_contrast_ids
                if contrast_id in available_contrast_ids
            )
            rejected_contrast_ids = tuple(
                contrast_id
                for contrast_id in submitted_contrast_ids
                if contrast_id not in available_contrast_ids
            )
            if rejected_contrast_ids:
                self._emit(
                    "reflection_evidence_contrast_ids_filtered",
                    call_id=call_id.value,
                    claim_sha256=hashlib.sha256(
                        draft.claim.encode("utf-8")
                    ).hexdigest(),
                    submitted_contrast_ids=list(submitted_contrast_ids),
                    accepted_contrast_ids=list(cited_contrast_ids),
                    rejected_contrast_ids=list(rejected_contrast_ids),
                )
                draft = replace(
                    draft,
                    evidence_contrast_ids=cited_contrast_ids,
                )
            if canonical_contrast_ids and not cited_contrast_ids:
                self._emit(
                    "reflection_insight_rejected",
                    call_id=call_id.value,
                    claim_sha256=hashlib.sha256(
                        draft.claim.encode("utf-8")
                    ).hexdigest(),
                    reason="no_accepted_evidence_contrast_ids",
                    submitted_contrast_ids=list(submitted_contrast_ids),
                    available_contrast_ids=list(canonical_contrast_ids),
                )
                rejected_insight_count += 1
                continue
            try:
                _validate_reflected_action_origin(
                    draft,
                    insight_contract,
                    cited_contrast_ids,
                    contrast_action_bindings,
                )
            except ReflectionCardContractError as exc:
                self._emit(
                    "reflection_insight_rejected",
                    call_id=call_id.value,
                    claim_sha256=hashlib.sha256(
                        draft.claim.encode("utf-8")
                    ).hexdigest(),
                    reason="origin_action_binding_mismatch",
                    contract_error=type(exc).__name__,
                )
                rejected_insight_count += 1
                continue
            cited_operator_ids = tuple(
                sorted(
                    {
                        contrast_lineage[contrast_id][0]
                        for contrast_id in cited_contrast_ids
                    }
                )
            )
            cited_candidate_ids = tuple(
                sorted(
                    {
                        candidate_id
                        for contrast_id in cited_contrast_ids
                        for candidate_id in contrast_lineage[contrast_id][1]
                    }
                )
            )
            evidence_lineage = InsightEvidenceLineage(
                reflection_call_id=call_id,
                source_operator_invocation_ids=cited_operator_ids,
                source_candidate_ids=cited_candidate_ids,
                available_contrast_ids=canonical_contrast_ids,
                cited_contrast_ids=cited_contrast_ids,
                finite_action_bindings=(
                    _finite_action_evidence_for_citations(
                        cited_contrast_ids,
                        contrast_action_bindings,
                    )
                ),
            )
            if revision_predecessors:
                try:
                    entry = self.memory.add_revision(
                        revision_predecessors[draft_index],
                        draft,
                        initial_score=0.0,
                        applicable_operator_kinds=evidence_operator_kinds,
                        origin=InsightOrigin.REFLECTION,
                        evidence_lineage=evidence_lineage,
                        revision_note="postseal evidence-guided revision",
                    )
                except Exception as exc:
                    failure_receipt = ReflectionCallReceipt(
                        call_id=call_id,
                        request=reflection_request,
                        status=ReflectionCallStatus.FAILED,
                        telemetry=result.telemetry,
                        telemetry_sha256=None,
                        failure_type=type(exc).__name__,
                    )
                    self._record_reflection_call_receipt(failure_receipt)
                    self._emit(
                        "reflection_failed",
                        call_id=call_id.value,
                        failure_type=type(exc).__name__,
                        reflection_call_receipt_sha256=(failure_receipt.receipt_sha256),
                    )
                    raise ReflectionCallExecutionError(
                        call_id,
                        exc,
                        failure_receipt,
                    ) from exc
                added_entries.append(entry)
            else:
                entry, is_new = self.memory.add(
                    draft,
                    initial_score=0.0,
                    applicable_operator_kinds=evidence_operator_kinds,
                    origin=InsightOrigin.REFLECTION,
                    evidence_lineage=evidence_lineage,
                )
                if is_new:
                    added_entries.append(entry)
        added = tuple(added_entries)
        # A true model abstention is an empty submitted tuple.  A non-empty
        # response whose every draft fails the engine's evidence/action
        # boundary is a typed call failure, not an abstention.  Keeping these
        # outcomes distinct is essential for causal-memory diagnostics and for
        # postseal policies that isolate a failed reflection from already valid
        # optimization endpoints.
        if (
            result.insights
            and not added
            and rejected_insight_count == len(result.insights)
        ):
            cause = ReflectionCardContractError(
                "all submitted reflection insights were rejected by the "
                "engine evidence/action contract"
            )
            failure_receipt = ReflectionCallReceipt(
                call_id=call_id,
                request=reflection_request,
                status=ReflectionCallStatus.FAILED,
                telemetry=result.telemetry,
                telemetry_sha256=None,
                failure_type=type(cause).__name__,
            )
            self._record_reflection_call_receipt(failure_receipt)
            self._emit(
                "reflection_failed",
                call_id=call_id.value,
                failure_type=type(cause).__name__,
                submitted_insight_count=len(result.insights),
                rejected_insight_count=rejected_insight_count,
                reason="all_submitted_insights_rejected",
                reflection_call_receipt_sha256=(failure_receipt.receipt_sha256),
            )
            raise ReflectionCallExecutionError(
                call_id,
                cause,
                failure_receipt,
            ) from cause
        completion_receipt = ReflectionCallReceipt(
            call_id=call_id,
            request=reflection_request,
            status=ReflectionCallStatus.COMPLETED,
            telemetry=result.telemetry,
            telemetry_sha256=None,
            failure_type=None,
            publications=tuple(_reflection_publication(entry) for entry in added),
        )
        self._record_reflection_call_receipt(completion_receipt)
        self._emit(
            "reflection_completed",
            call_id=call_id.value,
            label=label,
            requested_model=result.telemetry.requested_model,
            resolved_model=result.telemetry.resolved_model,
            resolved_provider=result.telemetry.resolved_provider,
            provider_response_id=result.telemetry.provider_response_id,
            finish_reason=result.telemetry.finish_reason,
            input_tokens=result.telemetry.input_tokens,
            output_tokens=result.telemetry.output_tokens,
            reasoning_tokens=result.telemetry.reasoning_tokens,
            cost_usd=(
                None
                if result.telemetry.cost_usd is None
                else str(result.telemetry.cost_usd)
            ),
            provider_latency_ns=result.telemetry.latency_ns,
            attempt_count=result.telemetry.attempt_count,
            reflection_call_receipt_sha256=(completion_receipt.receipt_sha256),
            **(
                {}
                if insight_contract is None
                else {"insight_contract": insight_contract.to_record()}
            ),
            insights=[
                {
                    "insight_id": entry.reference.insight_id.value,
                    "version": entry.reference.version,
                    "claim": entry.draft.claim,
                    "trigger": entry.draft.trigger,
                    "mechanism": entry.draft.mechanism,
                    "affected_paths": list(entry.draft.affected_paths),
                    "evidence_summary": entry.draft.evidence_summary,
                    "evidence_contrast_ids": list(entry.draft.evidence_contrast_ids),
                    "confidence": entry.draft.confidence,
                    "lifecycle_state": entry.lifecycle_state.value,
                    "retrievable": entry.retrievable,
                    "origin": entry.origin.value,
                    "applicable_operator_kinds": list(entry.applicable_operator_kinds),
                    **(entry.draft.intervention_record() or {}),
                    "evidence_lineage": entry.evidence_lineage.to_record(),
                }
                for entry in added
            ],
        )
        return added

    async def reflect(
        self,
        outcomes: Sequence[InvocationOutcome],
        *,
        label: str,
        max_insights: int = 4,
        min_insights: int = 0,
        insight_contract: ReflectionInsightContract | None = None,
        revision_predecessors: tuple[InsightRef, ...] = (),
        source_receipt_sha256s: tuple[str, ...] = (),
    ) -> tuple[InsightMemoryEntry, ...]:
        """Compatibility API returning only published memory entries."""

        async with self._reflection_publication_lock:
            return await self._reflect_entries(
                outcomes,
                label=label,
                max_insights=max_insights,
                min_insights=min_insights,
                insight_contract=insight_contract,
                revision_predecessors=revision_predecessors,
                source_receipt_sha256s=source_receipt_sha256s,
            )

    async def reflect_with_receipt(
        self,
        outcomes: Sequence[InvocationOutcome],
        *,
        label: str,
        max_insights: int = 4,
        min_insights: int = 0,
        insight_contract: ReflectionInsightContract | None = None,
        revision_predecessors: tuple[InsightRef, ...] = (),
        source_receipt_sha256s: tuple[str, ...] = (),
    ) -> ReflectionPublicationResult:
        """Run one batched reflection and return engine-issued call evidence.

        ``reflect`` remains the compatibility surface for workflows that only
        consume memory entries.  Causal curation policies should use this
        receipt-bearing API and validate the returned request/publication
        binding against their precommitted authority.
        """

        if self._reflection_workflow is not None:
            raise ValueError(
                "reflect_with_receipt currently requires one batched provider call"
            )
        async with self._reflection_publication_lock:
            before = set(self._reflection_call_receipts)
            entries = await self._reflect_entries(
                outcomes,
                label=label,
                max_insights=max_insights,
                min_insights=min_insights,
                insight_contract=insight_contract,
                revision_predecessors=revision_predecessors,
                source_receipt_sha256s=source_receipt_sha256s,
            )
            added_call_ids = tuple(
                call_id
                for call_id in self._reflection_call_receipts
                if call_id not in before
            )
            if len(added_call_ids) != 1:
                raise RuntimeError(
                    "receipt-bearing reflection did not publish exactly one "
                    "call receipt"
                )
            result = ReflectionPublicationResult(
                entries=entries,
                receipt=self._reflection_call_receipts[added_call_ids[0]],
            )
        ReflectionPublicationResult.__post_init__(result)
        return result


__all__ = [
    "AgenticEvolutionEngine",
    "CrossoverResponseMode",
    "EvolutionCandidate",
    "InsightAssignmentKind",
    "InvocationOutcome",
    "InvocationPlan",
    "MaterializedInvocation",
    "MutationContract",
    "MutationResponseMode",
    "OperatorKind",
    "PreparedInvocation",
    "ProposalAuthority",
    "REWARD_DEFINITION_HASH",
    "ReflectionCallReceipt",
    "ReflectionCallRequest",
    "ReflectionCallExecutionError",
    "ReflectionCallStatus",
    "ReflectionPublication",
    "ReflectionPublicationResult",
    "ReflectionRowProjectionBinding",
    "RewardPolicyBinding",
    "default_evidence_prompt",
    "default_parent_relative_reward",
]

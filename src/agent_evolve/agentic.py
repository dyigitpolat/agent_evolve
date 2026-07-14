"""Stable, benchmark-neutral composition surface for agentic evolution.

Benchmark packages should depend on this module, not on the internal
``application`` package.  A benchmark supplies domain semantics through the
frozen :class:`AgenticBenchmark` bundle; :func:`compose_agentic_optimizer`
wires those semantics into the evolution engine and archive exactly once.

The façade deliberately keeps provider construction outside its boundary.  A
caller injects any :class:`AgenticGenerator`, planner, ID factory, and memory
bank, which keeps benchmark adapters independent of Pydantic AI, OpenRouter,
or any future model runtime.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    REWARD_DEFINITION_HASH,
    ReflectionRowProjectionBinding,
    RewardPolicyBinding,
    default_evidence_prompt,
    default_parent_relative_reward,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    FrozenWaveReward,
    GenerationPlan,
    GenerationPlanner,
    OptimizerBudget,
    OptimizerBudgetExceeded,
    OptimizerContractError,
    OptimizerExecutionError,
    OptimizerPlanningError,
    OptimizerResult,
    OptimizerSlot,
    OptimizerState,
    SeedAdmissionPolicy,
)
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluation,
    DetailedEvaluationAdapter,
    DetailedEvaluationPayload,
    EvaluationCheck,
    EvaluationCheckStatus,
    EvaluationTimings,
    EvaluatorIdentity,
)
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackContext,
    GenerationFeedbackInterceptor,
    GenerationFeedbackReceipt,
    GenerationFeedbackReservation,
    GenerationFeedbackResult,
    generation_feedback_receipt_hash,
    seal_generation_feedback,
    validate_generation_feedback_receipt,
)
from agent_evolve.application.gated_agentic_generator import (
    AgenticTelemetryPolicy,
    TelemetryGatedAgenticGenerator,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.application.outcome_relation import (
    ObjectiveParetoOutcomePolicy,
    OutcomeComparator,
    OutcomeRelation,
    OutcomeRelationPolicyBinding,
    objective_pareto_outcome_binding,
)
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.application.reflection_workflow import (
    ContrastShardedReflectionWorkflow,
    ReflectionPromptShard,
    ReflectionWorkflow,
    ReflectionWorkflowExecutionError,
    ReflectionWorkflowRequest,
    ReflectionWorkflowResult,
)
from agent_evolve.core.problem import (
    ObjectiveSpec,
    Problem,
    ProblemContractError,
    ValidationOutcome,
    validate_objective_specs,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
    render_optimization_semantics,
)
from agent_evolve.domain.artifact import ArtifactRef, artifact_ref_for_bytes
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.outcome import FailureCategory, FailureCode, FailureRecord
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    FrozenJsonValue,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory, UuidIdFactory
from agent_evolve.infrastructure.resource_lease import (
    FileExclusiveResourceLease,
    ResourceConflictDetected,
    ResourceLeaseUnavailable,
)
from agent_evolve.policies.structured_output_budget import (
    FixedStructuredOutputBudgetPolicy,
)
from agent_evolve.policies.memory.prompt_shape import PromptShapeCommitmentPolicy
from agent_evolve.policies.memory.treatment_compliance import (
    FiniteTreatmentAction,
    InsightTreatmentRequirement,
    StrictTreatmentCompliancePolicy,
    TREATMENT_COMPLIANCE_DEFINITION_SHA256,
    TREATMENT_COMPLIANCE_POLICY_ID,
    TREATMENT_COMPLIANCE_POLICY_VERSION,
    TreatmentAdmissionReceipt,
    TreatmentAdmissionRequest,
    TreatmentActionBinding,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentCompliancePolicy,
    TreatmentComplianceRejected,
    TreatmentComplianceViolation,
    TreatmentInsightEvidence,
    TreatmentInsightBinding,
    TreatmentPreflightReceipt,
    TreatmentPreflightRequest,
    validate_treatment_admission_receipt,
    validate_treatment_preflight_receipt,
)
from agent_evolve.policies.feedback.held_out_asn import (
    G1ReflectionFeedbackInterceptor,
    HELD_OUT_SELECTOR_POLICY_ID,
    HELD_OUT_SELECTOR_POLICY_VERSION,
    HeldOutASNAssignmentCommitment,
    HeldOutASNAssignments,
    HeldOutASNPlanSet,
    HeldOutASNPlannerAdapter,
    HeldOutArm,
    HeldOutArmAssignment,
    HeldOutAssignmentUnavailable,
    HeldOutAssignmentUnavailableReason,
    HeldOutScoreMapEntry,
    REFLECTIVE_FEEDBACK_POLICY_ID,
    REFLECTIVE_FEEDBACK_POLICY_VERSION,
    ReflectedCard,
    ReflectedCardBatch,
    ReflectedCardMailbox,
    ReflectiveFeedbackContractError,
    build_reflected_card_batch,
    reflection_contrast_id,
    register_neutral_sham_card,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
    PhenotypeIdentityPolicy,
    SemanticProjectionPhenotypeIdentityPolicy,
    TypedConfigurationPhenotypeIdentityPolicy,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AgenticGenerator,
    CandidateDraft,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    VariationGenerationRequest,
    VariationGenerationResult,
    resolve_finite_variation_selection,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.structured_output_budget import (
    StructuredOutputBudgetPolicy,
    StructuredOutputRequestKind,
    resolve_structured_output_budget,
)
from agent_evolve.ports.id_factory import IdFactory
from agent_evolve.ports.resource_lease import (
    ExclusiveResourceLease,
    ResourceConflictObservation,
    ResourceConflictProbe,
    ResourceLeaseReceipt,
)
from agent_evolve.ports.variation_catalog import (
    FiniteVariationCatalog,
    bind_finite_variation_catalog,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
TraceSink = Callable[[Mapping[str, object]], None]
PromptBuilder = Callable[[str, Any, tuple[dict[str, object], ...]], str]


def _catalog_identity(catalog: FiniteVariationCatalog) -> tuple[str, int, str]:
    catalog_id = getattr(catalog, "catalog_id", None)
    catalog_version = getattr(catalog, "catalog_version", None)
    definition_sha256 = getattr(catalog, "definition_sha256", None)
    if type(catalog_id) is not str or _TOKEN.fullmatch(catalog_id) is None:
        raise ValueError("finite variation catalog_id has invalid syntax")
    if type(catalog_version) is not int or catalog_version <= 0:
        raise ValueError("finite variation catalog_version must be positive")
    require_sha256(definition_sha256, "finite variation definition_sha256")
    return catalog_id, catalog_version, definition_sha256


@dataclass(frozen=True, slots=True)
class AgenticBenchmark:
    """Frozen inverted-API bundle for one optimization benchmark.

    Detailed evaluation is intentionally all-or-nothing: an evidence adapter
    requires an explicit outcome relation, while an outcome relation cannot be
    supplied without detailed evidence.  The conservative exact-configuration
    phenotype policy remains a safe default; domains may inject a semantic
    projection when they can prove evaluator equivalence.
    """

    problem: Problem[dict[str, object]]
    reward: RewardPolicyBinding = field(
        default_factory=lambda: RewardPolicyBinding(
            default_parent_relative_reward,
            REWARD_DEFINITION_HASH,
        )
    )
    detailed_evaluator: DetailedEvaluationAdapter | None = None
    outcome_relation: OutcomeRelationPolicyBinding | None = None
    optimization_semantics: OptimizationSemantics | None = None
    phenotype_identity: PhenotypeIdentityPolicy = field(
        default_factory=TypedConfigurationPhenotypeIdentityPolicy
    )
    finite_variation_catalogs: tuple[FiniteVariationCatalog, ...] = ()
    _objectives: tuple[ObjectiveSpec, ...] = field(init=False, repr=False)
    _catalog_identities: tuple[tuple[str, int, str], ...] = field(
        init=False,
        repr=False,
    )
    _evaluator_identity: EvaluatorIdentity | None = field(init=False, repr=False)
    _phenotype_policy_identity: tuple[str, int] = field(init=False, repr=False)
    _optimization_semantics_identity: tuple[str, int, str] | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.problem, Problem):
            raise TypeError("problem must implement the public Problem protocol")
        objectives = tuple(self.problem.objectives)
        validate_objective_specs(objectives)
        candidate_model = getattr(self.problem, "candidate_model", None)
        if not isinstance(candidate_model, type) or not issubclass(
            candidate_model,
            BaseModel,
        ):
            raise TypeError("agentic problems must publish a Pydantic candidate_model")
        schema = candidate_model.model_json_schema(by_alias=False)
        if type(schema) is not dict or schema.get("type") != "object":
            raise TypeError("candidate_model must publish an object JSON schema")

        if type(self.reward) is not RewardPolicyBinding:
            raise TypeError("reward must be an exact RewardPolicyBinding")
        RewardPolicyBinding.__post_init__(self.reward)

        identify = getattr(self.phenotype_identity, "identify", None)
        if not callable(identify):
            raise TypeError("phenotype_identity must implement identify")
        phenotype_policy_identity = (
            getattr(self.phenotype_identity, "policy_id", None),
            getattr(self.phenotype_identity, "policy_version", None),
        )
        PhenotypeIdentity(
            policy_id=phenotype_policy_identity[0],
            policy_version=phenotype_policy_identity[1],
            value_sha256="0" * 64,
        )

        evaluator = self.detailed_evaluator
        relation = self.outcome_relation
        evaluator_identity: EvaluatorIdentity | None = None
        if evaluator is None:
            if relation is not None:
                raise ValueError(
                    "outcome_relation requires a detailed_evaluator; "
                    "objective-only mode derives its Pareto relation"
                )
        else:
            if not isinstance(evaluator, DetailedEvaluationAdapter):
                raise TypeError(
                    "detailed_evaluator must implement DetailedEvaluationAdapter"
                )
            evaluator_identity = getattr(evaluator, "evaluator_identity", None)
            if type(evaluator_identity) is not EvaluatorIdentity:
                raise TypeError(
                    "detailed_evaluator must publish an exact evaluator_identity"
                )
            EvaluatorIdentity.__post_init__(evaluator_identity)
            if type(relation) is not OutcomeRelationPolicyBinding:
                raise ValueError(
                    "detailed evaluation requires an explicit outcome_relation"
                )
            OutcomeRelationPolicyBinding.__post_init__(relation)

        active_relation = (
            objective_pareto_outcome_binding(objectives)
            if relation is None
            else relation
        )
        semantics = self.optimization_semantics
        if semantics is None:
            semantics = getattr(self.problem, "optimization_semantics", None)
        if semantics is not None:
            if type(semantics) is not OptimizationSemantics:
                raise TypeError(
                    "optimization_semantics must be an exact "
                    "OptimizationSemantics"
                )
            OptimizationSemantics.__post_init__(semantics)
            semantics.validate_binding(objectives, active_relation.identity)

        catalogs = self.finite_variation_catalogs
        if type(catalogs) is not tuple:
            raise TypeError("finite_variation_catalogs must be an exact tuple")
        catalog_identities: list[tuple[str, int, str]] = []
        for catalog in catalogs:
            if not isinstance(catalog, FiniteVariationCatalog):
                raise TypeError(
                    "finite_variation_catalogs must implement FiniteVariationCatalog"
                )
            catalog_identities.append(_catalog_identity(catalog))
        catalog_ids = tuple(identity[0] for identity in catalog_identities)
        if len(set(catalog_ids)) != len(catalog_ids):
            raise ValueError("finite variation catalog IDs must be unique")

        object.__setattr__(self, "_objectives", objectives)
        object.__setattr__(self, "_catalog_identities", tuple(catalog_identities))
        object.__setattr__(self, "_evaluator_identity", evaluator_identity)
        object.__setattr__(self, "optimization_semantics", semantics)
        object.__setattr__(
            self,
            "_optimization_semantics_identity",
            None if semantics is None else semantics.identity,
        )
        object.__setattr__(
            self,
            "_phenotype_policy_identity",
            phenotype_policy_identity,
        )

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        """Return the objective declaration frozen at adapter construction."""

        return self._objectives

    @property
    def finite_variation_catalog_identities(
        self,
    ) -> tuple[tuple[str, int, str], ...]:
        """Return identities in their frozen catalog declaration order."""

        return self._catalog_identities

    def validate_binding(self) -> None:
        """Fail if mutable adapter objects changed after bundle construction."""

        if tuple(self.problem.objectives) != self._objectives:
            raise ValueError("problem objectives changed after benchmark binding")
        current_phenotype_identity = (
            getattr(self.phenotype_identity, "policy_id", None),
            getattr(self.phenotype_identity, "policy_version", None),
        )
        if current_phenotype_identity != self._phenotype_policy_identity:
            raise ValueError("phenotype policy identity changed after binding")
        current_evaluator_identity = (
            None
            if self.detailed_evaluator is None
            else getattr(self.detailed_evaluator, "evaluator_identity", None)
        )
        if current_evaluator_identity != self._evaluator_identity:
            raise ValueError("detailed evaluator identity changed after binding")
        current_semantics_identity = (
            None
            if self.optimization_semantics is None
            else self.optimization_semantics.identity
        )
        if current_semantics_identity != self._optimization_semantics_identity:
            raise ValueError("optimization semantics identity changed after binding")
        identities = tuple(
            _catalog_identity(catalog)
            for catalog in self.finite_variation_catalogs
        )
        if identities != self._catalog_identities:
            raise ValueError("finite variation catalog identity changed after binding")

    def bind_finite_variation(
        self,
        catalog_id: str,
        parent_configuration: object,
    ) -> FiniteVariationContract:
        """Seal one named benchmark catalog against an exact parent."""

        self.validate_binding()
        if type(catalog_id) is not str or _TOKEN.fullmatch(catalog_id) is None:
            raise ValueError("catalog_id has invalid syntax")
        matches = tuple(
            catalog
            for catalog in self.finite_variation_catalogs
            if catalog.catalog_id == catalog_id
        )
        if len(matches) != 1:
            raise KeyError(f"unknown finite variation catalog {catalog_id!r}")
        catalog = matches[0]
        frozen = freeze_json(parent_configuration)
        if type(frozen) is not FrozenJsonObject:
            raise TypeError("finite variation parents must be typed-JSON objects")
        contract = bind_finite_variation_catalog(catalog, frozen)
        expected_identity = next(
            identity
            for identity in self._catalog_identities
            if identity[0] == catalog_id
        )
        if _catalog_identity(catalog) != expected_identity:
            raise ValueError("finite variation catalog changed while binding options")
        return contract


@dataclass(frozen=True, slots=True)
class _BoundGenerationPlanner:
    """Authenticate planner waves against benchmark-owned policy bindings."""

    benchmark: AgenticBenchmark
    delegate: GenerationPlanner

    def plan(
        self,
        state: OptimizerState,
        budget: OptimizerBudget,
    ) -> GenerationPlan:
        plan = self.delegate.plan(state, budget)
        if type(plan) is not GenerationPlan:
            raise TypeError("planner must return an exact GenerationPlan")
        GenerationPlan.__post_init__(plan)
        if (
            plan.reward.binding.definition_hash
            != self.benchmark.reward.definition_hash
        ):
            raise ValueError(
                "planner wave reward differs from the benchmark reward binding"
            )

        expected_by_catalog_parent: dict[
            tuple[str, str],
            FiniteVariationContract,
        ] = {}
        for slot in plan.slots:
            contract = slot.plan.finite_variation_contract
            if contract is None:
                continue
            parent = slot.plan.parents[0]
            key = (
                contract.catalog_id,
                typed_json_sha256(parent.configuration),
            )
            expected = expected_by_catalog_parent.get(key)
            if expected is None:
                expected = self.benchmark.bind_finite_variation(
                    contract.catalog_id,
                    parent.configuration,
                )
                expected_by_catalog_parent[key] = expected
            if contract.identity_sha256 != expected.identity_sha256:
                raise ValueError(
                    "finite variation contract was not produced by the "
                    "benchmark-bound catalog"
                )
        return plan


@dataclass(frozen=True, slots=True)
class AgenticOptimizerComposition:
    """Fully wired agentic engine, archive, and budgeted optimizer."""

    benchmark: AgenticBenchmark
    id_factory: IdFactory
    memory: InsightMemoryBank
    engine: AgenticEvolutionEngine
    archive: ParetoArchive
    optimizer: BudgetedAgenticOptimizer

    def __post_init__(self) -> None:
        if self.engine.problem is not self.benchmark.problem:
            raise ValueError("composition engine is bound to a different problem")
        if self.engine.outcome_relation_binding is not (
            self.archive.outcome_relation_binding
        ):
            raise ValueError("engine and archive must share one relation binding")
        if self.optimizer.engine is not self.engine:
            raise ValueError("optimizer is bound to a different engine")
        if self.optimizer.archive is not self.archive:
            raise ValueError("optimizer is bound to a different archive")
        if (
            self.engine.reward_binding.definition_hash
            != self.benchmark.reward.definition_hash
        ):
            raise ValueError("engine reward differs from the benchmark binding")
        if self.engine.optimization_semantics is not (
            self.benchmark.optimization_semantics
        ):
            raise ValueError(
                "engine and benchmark must share one optimization semantics value"
            )

    @property
    def outcome_relation(self) -> OutcomeRelationPolicyBinding:
        """The exact shared engine/archive relation object."""

        return self.engine.outcome_relation_binding

    def bind_finite_variation(
        self,
        catalog_id: str,
        parent_configuration: object,
    ) -> FiniteVariationContract:
        return self.benchmark.bind_finite_variation(
            catalog_id,
            parent_configuration,
        )


def compose_agentic_optimizer(
    benchmark: AgenticBenchmark,
    *,
    generator: AgenticGenerator,
    planner: GenerationPlanner,
    budget: OptimizerBudget,
    seed: int,
    id_factory: IdFactory | None = None,
    memory: InsightMemoryBank | None = None,
    initial_proposal_sequence: int = 0,
    evaluator_concurrency: int = 4,
    engine_trace_sink: TraceSink | None = None,
    optimizer_trace_sink: TraceSink | None = None,
    prompt_builder: PromptBuilder = default_evidence_prompt,
    prompt_shape_commitment_policy: PromptShapeCommitmentPolicy | None = None,
    reflection_row_projection: ReflectionRowProjectionBinding | None = None,
    reflection_workflow: ReflectionWorkflow | None = None,
    max_output_tokens: int = 2_048,
    structured_output_budget_policy: StructuredOutputBudgetPolicy | None = None,
    temperature: float | None = 0.2,
    evidence_admission_policy: EvidenceAdmissionPolicy = (
        EvidenceAdmissionPolicy.REQUIRE_COMPLIANT
    ),
    seed_admission_policy: SeedAdmissionPolicy | None = None,
    feedback_interceptor: GenerationFeedbackInterceptor | None = None,
    treatment_compliance_policy: TreatmentCompliancePolicy | None = None,
) -> AgenticOptimizerComposition:
    """Compose one benchmark without leaking domain policy into the core.

    The optimizer receives a narrow planner guard.  It verifies that every
    wave uses the benchmark reward identity and that every finite-selection
    contract is a deterministic snapshot of the benchmark-owned catalog.
    """

    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    benchmark.validate_binding()
    if not isinstance(generator, AgenticGenerator):
        raise TypeError("generator must implement AgenticGenerator")
    if not callable(getattr(planner, "plan", None)):
        raise TypeError("planner must implement plan(state, budget)")
    if type(budget) is not OptimizerBudget:
        raise TypeError("budget must be an exact OptimizerBudget")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if type(initial_proposal_sequence) is not int or initial_proposal_sequence < 0:
        raise ValueError("initial_proposal_sequence must be non-negative")
    if type(evaluator_concurrency) is not int or evaluator_concurrency <= 0:
        raise ValueError("evaluator_concurrency must be positive")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) < 0
        ):
            raise ValueError("temperature must be finite and non-negative or None")
        temperature = float(temperature)
    if not callable(prompt_builder):
        raise TypeError("prompt_builder must be callable")
    if engine_trace_sink is not None and not callable(engine_trace_sink):
        raise TypeError("engine_trace_sink must be callable")
    if optimizer_trace_sink is not None and not callable(optimizer_trace_sink):
        raise TypeError("optimizer_trace_sink must be callable")
    if type(evidence_admission_policy) is not EvidenceAdmissionPolicy:
        raise TypeError(
            "evidence_admission_policy must be an EvidenceAdmissionPolicy"
        )

    ids = UuidIdFactory() if id_factory is None else id_factory
    if not isinstance(ids, IdFactory):
        raise TypeError("id_factory must implement IdFactory")
    active_memory = (
        InsightMemoryBank(id_factory=ids) if memory is None else memory
    )
    if not isinstance(active_memory, InsightMemoryBank):
        raise TypeError("memory must be an InsightMemoryBank")

    engine = AgenticEvolutionEngine(
        problem=benchmark.problem,
        generator=generator,
        id_factory=ids,
        memory=active_memory,
        seed=seed,
        initial_proposal_sequence=initial_proposal_sequence,
        evaluator_concurrency=evaluator_concurrency,
        trace_sink=engine_trace_sink,
        reward_policy=benchmark.reward.score,
        reward_definition_hash=benchmark.reward.definition_hash,
        prompt_builder=prompt_builder,
        prompt_shape_commitment_policy=prompt_shape_commitment_policy,
        reflection_row_projection=reflection_row_projection,
        reflection_workflow=reflection_workflow,
        max_output_tokens=max_output_tokens,
        structured_output_budget_policy=structured_output_budget_policy,
        temperature=temperature,
        phenotype_identity_policy=benchmark.phenotype_identity,
        detailed_evaluator=benchmark.detailed_evaluator,
        outcome_relation_binding=benchmark.outcome_relation,
        optimization_semantics=benchmark.optimization_semantics,
        treatment_compliance_policy=treatment_compliance_policy,
    )
    # Pass the engine's exact binding object.  In objective-only mode the engine
    # creates the default Pareto binding, so constructing another equivalent
    # value here would lose the stronger object-identity invariant.
    archive = ParetoArchive(
        benchmark.objectives,
        evidence_admission_policy=evidence_admission_policy,
        outcome_relation_binding=engine.outcome_relation_binding,
    )
    bound_planner = _BoundGenerationPlanner(benchmark, planner)
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=archive,
        planner=bound_planner,
        budget=budget,
        seed_admission_policy=seed_admission_policy,
        feedback_interceptor=feedback_interceptor,
        trace_sink=optimizer_trace_sink,
    )
    return AgenticOptimizerComposition(
        benchmark=benchmark,
        id_factory=ids,
        memory=active_memory,
        engine=engine,
        archive=archive,
        optimizer=optimizer,
    )


__all__ = [
    "AgenticBenchmark",
    "AgenticCallTelemetry",
    "AgenticEvolutionEngine",
    "AgenticGenerator",
    "AgenticOptimizerComposition",
    "AgenticTelemetryPolicy",
    "ArrayIndex",
    "ArtifactRef",
    "DeterministicIdFactory",
    "BudgetedAgenticOptimizer",
    "CandidateDraft",
    "ContrastShardedReflectionWorkflow",
    "DetailedEvaluation",
    "DetailedEvaluationAdapter",
    "DetailedEvaluationPayload",
    "EvaluationCheck",
    "EvaluationCheckStatus",
    "EvaluationTimings",
    "EvaluatorIdentity",
    "EvidenceAdmissionPolicy",
    "ExclusiveResourceLease",
    "EvolutionCandidate",
    "FailureCategory",
    "FailureCode",
    "FailureRecord",
    "FileExclusiveResourceLease",
    "FiniteVariationCatalog",
    "FiniteVariationContract",
    "FiniteVariationOption",
    "FiniteVariationSelectionDraft",
    "FixedStructuredOutputBudgetPolicy",
    "FrozenJsonObject",
    "FrozenJsonValue",
    "FrozenWaveReward",
    "GenerationPlan",
    "GenerationPlanner",
    "GenerationFeedbackContext",
    "GenerationFeedbackInterceptor",
    "GenerationFeedbackReceipt",
    "GenerationFeedbackReservation",
    "GenerationFeedbackResult",
    "G1ReflectionFeedbackInterceptor",
    "HELD_OUT_SELECTOR_POLICY_ID",
    "HELD_OUT_SELECTOR_POLICY_VERSION",
    "HeldOutASNAssignmentCommitment",
    "HeldOutASNAssignments",
    "HeldOutASNPlanSet",
    "HeldOutASNPlannerAdapter",
    "HeldOutArm",
    "HeldOutArmAssignment",
    "HeldOutAssignmentUnavailable",
    "HeldOutAssignmentUnavailableReason",
    "HeldOutScoreMapEntry",
    "FiniteTreatmentAction",
    "IdFactory",
    "InsightMemoryBank",
    "InsightTreatmentRequirement",
    "InsightDraft",
    "InvocationOutcome",
    "InvocationPlan",
    "JsonPath",
    "MutationContract",
    "MutationResponseMode",
    "MetricRole",
    "MetricSemantics",
    "MetricSense",
    "MetricEffectDirection",
    "MetricEffectPrediction",
    "ObjectiveParetoOutcomePolicy",
    "ObjectiveSpec",
    "OptimizationSemantics",
    "ObjectKey",
    "OperatorKind",
    "OptimizerBudget",
    "OptimizerBudgetExceeded",
    "OptimizerContractError",
    "OptimizerExecutionError",
    "OptimizerPlanningError",
    "OptimizerResult",
    "OptimizerSlot",
    "OptimizerState",
    "OutcomeComparator",
    "OutcomeRelation",
    "OutcomeRelationPolicyBinding",
    "OutcomeOrderingKind",
    "OutcomeOrderingSemantics",
    "ParetoArchive",
    "PhenotypeIdentity",
    "PhenotypeIdentityPolicy",
    "Problem",
    "ProblemContractError",
    "REWARD_DEFINITION_HASH",
    "ReflectionGenerationRequest",
    "ReflectionGenerationResult",
    "ReflectionInsightContract",
    "ReflectionRowProjectionBinding",
    "ReflectionPromptShard",
    "ReflectionWorkflow",
    "ReflectionWorkflowExecutionError",
    "ReflectionWorkflowRequest",
    "ReflectionWorkflowResult",
    "REFLECTIVE_FEEDBACK_POLICY_ID",
    "REFLECTIVE_FEEDBACK_POLICY_VERSION",
    "ReflectedCard",
    "ReflectedCardBatch",
    "ReflectedCardMailbox",
    "ReflectiveFeedbackContractError",
    "ResourceConflictDetected",
    "ResourceConflictObservation",
    "ResourceConflictProbe",
    "ResourceLeaseReceipt",
    "ResourceLeaseUnavailable",
    "RewardPolicyBinding",
    "SemanticProjectionPhenotypeIdentityPolicy",
    "StructuredOutputBudgetPolicy",
    "StructuredOutputRequestKind",
    "StrictTreatmentCompliancePolicy",
    "TREATMENT_COMPLIANCE_DEFINITION_SHA256",
    "TREATMENT_COMPLIANCE_POLICY_ID",
    "TREATMENT_COMPLIANCE_POLICY_VERSION",
    "TelemetryGatedAgenticGenerator",
    "TreatmentAdmissionReceipt",
    "TreatmentAdmissionRequest",
    "TreatmentActionBinding",
    "TreatmentAssignmentRole",
    "TreatmentClaimMode",
    "TreatmentCompliancePolicy",
    "TreatmentComplianceRejected",
    "TreatmentComplianceViolation",
    "TreatmentInsightEvidence",
    "TreatmentInsightBinding",
    "TreatmentPreflightReceipt",
    "TreatmentPreflightRequest",
    "validate_treatment_admission_receipt",
    "validate_treatment_preflight_receipt",
    "TypedConfigurationPhenotypeIdentityPolicy",
    "UuidIdFactory",
    "ValidationOutcome",
    "VariationGenerationRequest",
    "VariationGenerationResult",
    "artifact_ref_for_bytes",
    "bind_finite_variation_catalog",
    "build_reflected_card_batch",
    "compose_agentic_optimizer",
    "default_parent_relative_reward",
    "default_evidence_prompt",
    "freeze_json",
    "generation_feedback_receipt_hash",
    "objective_pareto_outcome_binding",
    "reflection_contrast_id",
    "register_neutral_sham_card",
    "render_optimization_semantics",
    "resolve_structured_output_budget",
    "resolve_finite_variation_selection",
    "seal_generation_feedback",
    "thaw_json",
    "typed_json_sha256",
    "validate_generation_feedback_receipt",
    "validate_reflection_insight_draft",
]

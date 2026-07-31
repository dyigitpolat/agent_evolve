"""Generic bridge from typed LLM plans to the materialized-action market.

The model selects a parent and one or two opaque finite-action identifiers.
Trusted engine code materializes those plans, a benchmark-owned phenotype port
deduplicates executable behavior, and an authoritative evaluation port is
invoked only for the downstream broker's selected subset.

No workload objective, configuration schema, model, provider, or simulator
identity is interpreted here.  A new workload supplies:

* a parent-bound :class:`CrossParentFiniteActionSchema`;
* a phenotype identity projection;
* canonical parent-position cells;
* an authoritative selected-action evaluator; and
* an opaque proposal context containing its scientific semantics and evidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import math
import re
from typing import Callable, Protocol, runtime_checkable

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionContext,
    MaterializedActionDescriptor,
)
from agent_evolve.application.residual_portfolio_evolution import (
    DISJOINT_ACTION_EVALUATION_WAVES_V1,
    DisjointActionEvaluationLedger,
    MaterializedActionEvaluation,
    MaterializedActionEvaluationBatch,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.residual_reachability import (
    CrossParentFiniteActionSchema,
    HierarchicalResidualPlan,
    MaterializedResidualProposal,
    ResidualProposalRole,
    materialize_hierarchical_residual_plan,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
)
from agent_evolve.integrations.pydantic_ai.residual_reachability import (
    HierarchicalResidualMetricForecast,
    HierarchicalResidualProposalRequest,
    HierarchicalResidualProposalSelection,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.structured_generator import MAX_OUTPUT_TOKENS


MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_ID = (
    "pydantic_ai_materialized_hierarchical_residual_expert"
)
MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_VERSION = 2
MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-materialized-hierarchical-residual-expert:v2;"
        b"proposal=parent-plus-one-or-two-opaque-finite-action-ids;"
        b"materialization=trusted-engine-only;"
        b"novelty=benchmark-phenotype-identity;"
        b"evaluation=broker-selected-disjoint-action-subset-waves;"
        b"exactly-once-boundary=action-not-proposal;"
        b"reservation=fail-closed-before-authoritative-evaluator-await;"
        b"context=generic-role-radius-parent-cell;"
        b"workload-model-provider-branches=false"
    ).hexdigest()
)

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_OPERATION = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_PHENOTYPE_PROJECTION_DOMAIN = (
    b"agent-evolve:materialized-phenotype-projection-port:v1\x00"
)
_EVALUATION_PORT_DOMAIN = (
    b"agent-evolve:materialized-action-evaluation-port:v1\x00"
)
_BOUND_EXPERT_DEFINITION_DOMAIN = (
    b"agent-evolve:materialized-hierarchical-residual-expert-bound:v1\x00"
)


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _operation(value: str, *, name: str) -> None:
    if type(value) is not str or _OPERATION.fullmatch(value) is None:
        raise ValueError(f"{name} must use the operation token grammar")


def _port_identity(
    value: object,
    *,
    id_name: str,
    version_name: str,
    definition_name: str,
    domain: bytes,
) -> tuple[str, int, str]:
    port_id = getattr(value, id_name, None)
    version = getattr(value, version_name, None)
    definition_sha256 = getattr(value, definition_name, None)
    _token(port_id, name=id_name)
    if type(version) is not int or version <= 0:
        raise ValueError(f"{version_name} must be positive")
    require_sha256(definition_sha256, definition_name)
    # Bind the three public fields before returning.  This also prevents a
    # malformed dynamic object from accidentally sharing a valid definition.
    hashlib.sha256(
        domain
        + port_id.encode("ascii")
        + version.to_bytes(8, "big")
        + bytes.fromhex(definition_sha256)
    ).digest()
    return port_id, version, definition_sha256


def _bound_expert_definition_sha256(
    source_definition_sha256: str,
) -> str:
    require_sha256(
        source_definition_sha256,
        "source expert_definition_sha256",
    )
    return hashlib.sha256(
        _BOUND_EXPERT_DEFINITION_DOMAIN
        + bytes.fromhex(source_definition_sha256)
        + bytes.fromhex(
            MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_DEFINITION_SHA256
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class HierarchicalResidualExpertSpec:
    """Provider-neutral configuration for one proposal lane."""

    expert_id: str
    expert_version: int
    expert_definition_sha256: str
    instruction: str
    proposal_count: int
    allowed_radii: tuple[int, ...]
    allowed_roles: tuple[ResidualProposalRole, ...]
    required_metric_ids: tuple[str, ...]
    minimum_distinct_parents: int
    max_output_tokens: int
    temperature: float
    operation: str = "propose_residual_plans"

    def __post_init__(self) -> None:
        _operation(self.expert_id, name="expert_id")
        if type(self.expert_version) is not int or self.expert_version <= 0:
            raise ValueError("expert_version must be positive")
        require_sha256(
            self.expert_definition_sha256,
            "expert_definition_sha256",
        )
        if (
            type(self.instruction) is not str
            or not self.instruction.strip()
            or len(self.instruction.encode("utf-8")) > 32_768
        ):
            raise ValueError("instruction must be non-empty and bounded")
        if (
            type(self.proposal_count) is not int
            or not 1 <= self.proposal_count <= 32
        ):
            raise ValueError("proposal_count must lie in [1, 32]")
        if (
            type(self.allowed_radii) is not tuple
            or not self.allowed_radii
            or self.allowed_radii
            != tuple(sorted(set(self.allowed_radii)))
            or not set(self.allowed_radii).issubset({1, 2})
        ):
            raise ValueError(
                "allowed_radii must be a canonical subset of {1, 2}"
            )
        if (
            type(self.allowed_roles) is not tuple
            or not self.allowed_roles
            or any(
                type(value) is not ResidualProposalRole
                for value in self.allowed_roles
            )
            or self.allowed_roles
            != tuple(
                sorted(
                    set(self.allowed_roles),
                    key=lambda value: value.value,
                )
            )
        ):
            raise ValueError(
                "allowed_roles must be a canonical non-empty tuple"
            )
        if (
            type(self.required_metric_ids) is not tuple
            or not self.required_metric_ids
            or self.required_metric_ids
            != tuple(sorted(set(self.required_metric_ids)))
        ):
            raise ValueError(
                "required_metric_ids must be non-empty and canonical"
            )
        for metric_id in self.required_metric_ids:
            _token(metric_id, name="required metric_id")
        if (
            type(self.minimum_distinct_parents) is not int
            or not 1
            <= self.minimum_distinct_parents
            <= self.proposal_count
        ):
            raise ValueError(
                "minimum_distinct_parents must fit proposal_count"
            )
        if (
            type(self.max_output_tokens) is not int
            or not 1 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError("max_output_tokens is outside the generic bound")
        if (
            type(self.temperature) is not float
            or not math.isfinite(self.temperature)
        ):
            raise TypeError("temperature must be a finite exact float")
        _operation(self.operation, name="operation")


@dataclass(frozen=True, slots=True)
class ResidualParentActionContext:
    """One workload-neutral parent-position label for broker conditioning."""

    parent_candidate_id: CandidateId
    position_cell: str

    def __post_init__(self) -> None:
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be exact")
        CandidateId.__post_init__(self.parent_candidate_id)
        _token(self.position_cell, name="position_cell")


@runtime_checkable
class MaterializedPhenotypeProjectionPort(Protocol):
    """Project an executable configuration onto benchmark-semantic identity."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    def project(self, configuration: FrozenJsonObject) -> str: ...


@runtime_checkable
class SelectedMaterializedActionEvaluationPort(Protocol):
    """Run exactly one broker-selected action through authoritative truth."""

    evaluator_id: str
    evaluator_version: int
    definition_sha256: str

    async def evaluate(
        self,
        action: MaterializedActionDescriptor,
    ) -> MaterializedActionEvaluation: ...


@runtime_checkable
class HierarchicalResidualProposalPolicyPort(Protocol):
    """Select typed residual plans; queueing and providers remain injected."""

    async def select(
        self,
        request: HierarchicalResidualProposalRequest,
    ) -> HierarchicalResidualProposalSelection: ...


@dataclass(frozen=True, slots=True)
class MaterializedHierarchicalResidualActionEvidence:
    """Typed, pre-evaluation semantic claims for one compiled action."""

    action: MaterializedActionDescriptor
    plan: HierarchicalResidualPlan
    materialized: MaterializedResidualProposal
    provider_rank: int
    materialized_rank: int
    probability_valid: float
    effect_predictions: tuple[
        HierarchicalResidualMetricForecast,
        ...,
    ]
    rationale: str

    def __post_init__(self) -> None:
        if type(self.action) is not MaterializedActionDescriptor:
            raise TypeError("action must be exact")
        self.action.__post_init__()
        if type(self.plan) is not HierarchicalResidualPlan:
            raise TypeError("plan must be exact")
        self.plan.__post_init__()
        if type(self.materialized) is not MaterializedResidualProposal:
            raise TypeError("materialized must be exact")
        self.materialized.__post_init__()
        if (
            self.materialized.plan != self.plan
            or self.action.target_candidate_id
            != self.materialized.target_candidate_id
            or self.action.configuration_sha256
            != self.materialized.configuration_sha256
        ):
            raise ValueError(
                "semantic action evidence does not join its materialization"
            )
        if self.provider_rank != self.plan.native_rank:
            raise ValueError("provider_rank differs from the source plan")
        if (
            type(self.materialized_rank) is not int
            or self.materialized_rank <= 0
        ):
            raise ValueError("materialized_rank must be positive")
        if (
            type(self.probability_valid) is not float
            or not 0.0 <= self.probability_valid <= 1.0
        ):
            raise ValueError("probability_valid must lie in [0, 1]")
        if (
            type(self.effect_predictions) is not tuple
            or not self.effect_predictions
            or any(
                type(value) is not HierarchicalResidualMetricForecast
                for value in self.effect_predictions
            )
        ):
            raise TypeError("effect_predictions must contain exact forecasts")
        for value in self.effect_predictions:
            value.__post_init__()
        metric_ids = tuple(
            value.metric_id for value in self.effect_predictions
        )
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("forecast metrics must be canonical")
        if type(self.rationale) is not str or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action.action_sha256,
            "target_candidate_id": self.action.target_candidate_id.value,
            "parent_candidate_id": self.plan.parent_candidate_id.value,
            "plan_sha256": self.plan.plan_sha256,
            "component_option_ids": list(self.plan.component_option_ids),
            "role": self.plan.role.value,
            "radius": self.plan.radius,
            "provider_rank": self.provider_rank,
            "materialized_rank": self.materialized_rank,
            "probability_valid_hex": self.probability_valid.hex(),
            "effect_predictions": [
                value.to_record() for value in self.effect_predictions
            ],
            "rationale": self.rationale,
            "configuration_sha256": self.action.configuration_sha256,
            "phenotype_identity_sha256": (
                self.action.phenotype_identity_sha256
            ),
            "materialization_receipt_sha256": (
                self.materialized.engine_materialization_receipt_sha256
            ),
            "candidate_outcomes_observed": False,
        }


@runtime_checkable
class MaterializedHierarchicalResidualActionEvidencePort(Protocol):
    """Expose sealed pre-evaluation semantics without evaluator outcomes."""

    expert_id: str

    def evidence_for(
        self,
        action_sha256: str,
    ) -> MaterializedHierarchicalResidualActionEvidence | None: ...


def _telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
    if type(value) is not AgenticCallTelemetry:
        raise TypeError("telemetry must be exact")
    value.__post_init__()
    return {
        "requested_model": value.requested_model,
        "resolved_model": value.resolved_model,
        "resolved_provider": value.resolved_provider,
        "provider_response_id": value.provider_response_id,
        "finish_reason": value.finish_reason,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "reasoning_tokens": value.reasoning_tokens,
        "cache_read_tokens": value.cache_read_tokens,
        "cache_write_tokens": value.cache_write_tokens,
        "cost_usd": (
            None
            if value.cost_usd is None
            else str(Decimal(value.cost_usd))
        ),
        "latency_ns": value.latency_ns,
        "attempt_count": value.attempt_count,
    }


@dataclass(slots=True)
class PydanticAIMaterializedHierarchicalResidualExpert:
    """Workload-neutral, broker-compatible hierarchical proposal expert."""

    spec: HierarchicalResidualExpertSpec
    policy: HierarchicalResidualProposalPolicyPort = field(
        repr=False,
        compare=False,
    )
    action_schema: CrossParentFiniteActionSchema = field(
        repr=False,
        compare=False,
    )
    phenotype_projection: MaterializedPhenotypeProjectionPort = field(
        repr=False,
        compare=False,
    )
    evaluator: SelectedMaterializedActionEvaluationPort = field(
        repr=False,
        compare=False,
    )
    parent_contexts: tuple[ResidualParentActionContext, ...]
    observed_phenotype_sha256s: tuple[str, ...]
    memory_dose_bin: int = 0
    telemetry_validator: Callable[[AgenticCallTelemetry], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    selection_sink: Callable[[dict[str, object]], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    materialization_failure_sink: (
        Callable[[dict[str, object]], None] | None
    ) = field(default=None, repr=False, compare=False)
    expert_id: str = field(init=False)
    expert_version: int = field(init=False)
    definition_sha256: str = field(init=False)
    result: HierarchicalResidualProposalSelection | None = field(
        init=False,
        default=None,
    )
    action_evidence_by_sha256: dict[
        str,
        MaterializedHierarchicalResidualActionEvidence,
    ] = field(init=False, default_factory=dict)
    _proposal: MaterializedActionProposalBatch | None = field(
        init=False,
        default=None,
    )
    evaluation_wave_semantics: str = field(
        init=False,
        default=DISJOINT_ACTION_EVALUATION_WAVES_V1,
    )
    _evaluation_ledger: DisjointActionEvaluationLedger = field(
        init=False,
        default_factory=DisjointActionEvaluationLedger,
    )

    def __post_init__(self) -> None:
        if type(self.spec) is not HierarchicalResidualExpertSpec:
            raise TypeError("spec must be exact")
        self.spec.__post_init__()
        if not isinstance(
            self.policy,
            HierarchicalResidualProposalPolicyPort,
        ):
            raise TypeError("policy must implement its proposal port")
        if type(self.action_schema) is not CrossParentFiniteActionSchema:
            raise TypeError("action_schema must be exact")
        self.action_schema.__post_init__()
        if not isinstance(
            self.phenotype_projection,
            MaterializedPhenotypeProjectionPort,
        ):
            raise TypeError(
                "phenotype_projection must implement its runtime port"
            )
        _port_identity(
            self.phenotype_projection,
            id_name="projection_id",
            version_name="projection_version",
            definition_name="definition_sha256",
            domain=_PHENOTYPE_PROJECTION_DOMAIN,
        )
        if not isinstance(
            self.evaluator,
            SelectedMaterializedActionEvaluationPort,
        ):
            raise TypeError("evaluator must implement its runtime port")
        _port_identity(
            self.evaluator,
            id_name="evaluator_id",
            version_name="evaluator_version",
            definition_name="definition_sha256",
            domain=_EVALUATION_PORT_DOMAIN,
        )
        if (
            type(self.parent_contexts) is not tuple
            or not self.parent_contexts
            or any(
                type(value) is not ResidualParentActionContext
                for value in self.parent_contexts
            )
        ):
            raise TypeError(
                "parent_contexts must contain exact non-empty values"
            )
        for value in self.parent_contexts:
            value.__post_init__()
        parent_ids = tuple(
            value.parent_candidate_id.value
            for value in self.parent_contexts
        )
        if parent_ids != tuple(sorted(set(parent_ids))):
            raise ValueError("parent_contexts must be unique and canonical")
        schema_parent_ids = tuple(
            value.parent_candidate_id.value
            for value in self.action_schema.bindings
        )
        if parent_ids != schema_parent_ids:
            raise ValueError(
                "parent_contexts must exactly cover the action schema"
            )
        if (
            type(self.observed_phenotype_sha256s) is not tuple
            or self.observed_phenotype_sha256s
            != tuple(sorted(set(self.observed_phenotype_sha256s)))
        ):
            raise ValueError(
                "observed phenotype identities must be canonical"
            )
        for value in self.observed_phenotype_sha256s:
            require_sha256(value, "observed phenotype identity")
        if (
            type(self.memory_dose_bin) is not int
            or not 0 <= self.memory_dose_bin <= 15
        ):
            raise ValueError("memory_dose_bin must lie in [0, 15]")
        for value, name in (
            (self.telemetry_validator, "telemetry_validator"),
            (self.selection_sink, "selection_sink"),
            (
                self.materialization_failure_sink,
                "materialization_failure_sink",
            ),
        ):
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        if self.spec.minimum_distinct_parents > len(
            self.action_schema.bindings
        ):
            raise ValueError(
                "minimum_distinct_parents exceeds schema parent count"
            )
        self.expert_id = self.spec.expert_id
        self.expert_version = self.spec.expert_version
        self.definition_sha256 = _bound_expert_definition_sha256(
            self.spec.expert_definition_sha256
        )

    @property
    def proposal_count(self) -> int:
        return self.spec.proposal_count

    def evidence_for(
        self,
        action_sha256: str,
    ) -> MaterializedHierarchicalResidualActionEvidence | None:
        require_sha256(action_sha256, "action_sha256")
        return self.action_evidence_by_sha256.get(action_sha256)

    def _call_id(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> LLMCallId:
        digest = hashlib.sha256(
            bytes.fromhex(request.request_sha256)
            + self.expert_id.encode("ascii")
        ).hexdigest()
        return LLMCallId(
            f"call_residual_d{request.decision_index:04d}_{digest[:24]}"
        )

    def _failure(
        self,
        *,
        request: ResidualPortfolioDecisionRequest,
        plan: HierarchicalResidualPlan,
        error: Exception,
    ) -> None:
        if self.materialization_failure_sink is None:
            return
        self.materialization_failure_sink(
            {
                "schema_version": 1,
                "adapter_definition_sha256": (
                    MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_DEFINITION_SHA256
                ),
                "residual_request_sha256": request.request_sha256,
                "expert_id": self.expert_id,
                "plan": plan.to_record(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "candidate_outcomes_observed": False,
            }
        )

    async def propose(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> MaterializedActionProposalBatch:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if self._proposal is not None:
            raise ValueError("one stage expert can propose only once")
        if request.proposal_slots_for(self.expert_id) != self.proposal_count:
            raise ValueError(
                "residual proposal capacity differs from the expert spec"
            )

        selection_request = HierarchicalResidualProposalRequest(
            call_id=self._call_id(request),
            operation=self.spec.operation,
            instruction=self.spec.instruction,
            context=request.proposal_context,
            action_schema=self.action_schema,
            proposal_count=self.spec.proposal_count,
            allowed_radii=self.spec.allowed_radii,
            allowed_roles=self.spec.allowed_roles,
            required_metric_ids=self.spec.required_metric_ids,
            minimum_distinct_parents=(
                self.spec.minimum_distinct_parents
            ),
            expert_id=self.expert_id,
            expert_definition_sha256=self.definition_sha256,
            max_output_tokens=self.spec.max_output_tokens,
            temperature=self.spec.temperature,
        )
        result = await self.policy.select(selection_request)
        if type(result) is not HierarchicalResidualProposalSelection:
            raise TypeError("proposal policy returned a foreign selection")
        result.__post_init__()
        if result.request_sha256 != selection_request.request_sha256:
            raise ValueError("proposal policy selected for another request")
        if self.telemetry_validator is not None:
            self.telemetry_validator(result.telemetry)
        self.result = result
        if self.selection_sink is not None:
            self.selection_sink(
                {
                    "schema_version": 1,
                    "adapter_definition_sha256": (
                        MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_DEFINITION_SHA256
                    ),
                    "residual_request_sha256": request.request_sha256,
                    "selection": result.to_record(),
                    "telemetry": _telemetry_record(result.telemetry),
                    "candidate_outcomes_observed": False,
                }
            )

        position_by_parent = {
            value.parent_candidate_id: value.position_cell
            for value in self.parent_contexts
        }
        observed = set(self.observed_phenotype_sha256s)
        unique_by_phenotype: dict[
            str,
            tuple[
                HierarchicalResidualPlan,
                str,
                float,
                tuple[HierarchicalResidualMetricForecast, ...],
                MaterializedResidualProposal,
            ],
        ] = {}
        for plan, rationale, probability_valid, predictions in zip(
            result.plans,
            result.rationales,
            result.probability_valid,
            result.effect_predictions,
            strict=True,
        ):
            target = CandidateId(
                "candidate_residual_"
                f"d{request.decision_index:04d}_{plan.plan_sha256[:24]}"
            )
            try:
                materialized = materialize_hierarchical_residual_plan(
                    schema=self.action_schema,
                    plan=plan,
                    target_candidate_id=target,
                )
                phenotype_sha256 = self.phenotype_projection.project(
                    materialized.configuration
                )
                require_sha256(
                    phenotype_sha256,
                    "projected phenotype identity",
                )
            except (TypeError, ValueError, RuntimeError) as error:
                self._failure(
                    request=request,
                    plan=plan,
                    error=error,
                )
                continue
            if phenotype_sha256 in observed:
                continue
            unique_by_phenotype.setdefault(
                phenotype_sha256,
                (
                    plan,
                    rationale,
                    probability_valid,
                    predictions,
                    materialized,
                ),
            )

        actions: list[MaterializedActionDescriptor] = []
        evidence_rows: list[
            MaterializedHierarchicalResidualActionEvidence
        ] = []
        for rank, (
            phenotype_sha256,
            (
                plan,
                rationale,
                probability_valid,
                predictions,
                materialized,
            ),
        ) in enumerate(unique_by_phenotype.items(), start=1):
            role = plan.role.value
            action = MaterializedActionDescriptor(
                context=MaterializedActionContext(
                    campaign_scope_sha256=request.campaign_scope_sha256,
                    decision_index=request.decision_index,
                    phase=request.phase,
                    remaining_decisions=request.remaining_decisions,
                    remaining_evaluations=request.remaining_evaluations,
                    residual_frontier_cell=f"{role}.r{plan.radius}",
                    parent_position_cell=position_by_parent[
                        plan.parent_candidate_id
                    ],
                    archive_relation_cell="unknown_pre_eval",
                    structural_signature_sha256=phenotype_sha256,
                    patch_compatibility_cell=(
                        "safe_disjoint"
                        if plan.radius == 2
                        else "atomic"
                    ),
                    forecast_calibration_cell="trace_addressable",
                    source_distance_bin=plan.radius,
                    memory_dose_bin=self.memory_dose_bin,
                ),
                configuration=materialized.configuration,
                phenotype_identity_sha256=phenotype_sha256,
                expert_id=self.expert_id,
                native_rank=rank,
                parent_ids=(plan.parent_candidate_id,),
                operator_id=f"{role}.r{plan.radius}",
                target_candidate_id=materialized.target_candidate_id,
                role_id=role,
                normalized_evaluation_cost=1.0,
                reference_action=False,
            )
            evidence = MaterializedHierarchicalResidualActionEvidence(
                action=action,
                plan=plan,
                materialized=materialized,
                provider_rank=plan.native_rank,
                materialized_rank=rank,
                probability_valid=probability_valid,
                effect_predictions=predictions,
                rationale=rationale,
            )
            actions.append(action)
            evidence_rows.append(evidence)
            self.action_evidence_by_sha256[action.action_sha256] = evidence
        if not actions:
            raise RuntimeError(
                f"{self.expert_id} produced no novel materialized action"
            )

        proposal = MaterializedActionProposalBatch(
            request_sha256=request.request_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            actions=tuple(actions),
            evidence=freeze_json(
                {
                    "schema_version": 1,
                    "adapter": {
                        "adapter_id": (
                            MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_ID
                        ),
                        "adapter_version": (
                            MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_VERSION
                        ),
                        "definition_sha256": (
                            MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_DEFINITION_SHA256
                        ),
                    },
                    "selection_decision_sha256": result.decision_sha256,
                    "phenotype_projection": {
                        "projection_id": (
                            self.phenotype_projection.projection_id
                        ),
                        "projection_version": (
                            self.phenotype_projection.projection_version
                        ),
                        "definition_sha256": (
                            self.phenotype_projection.definition_sha256
                        ),
                    },
                    "common_prior_state_sha256": (
                        request.prior_state_sha256
                    ),
                    "members": [
                        value.to_record() for value in evidence_rows
                    ],
                    "requested_member_count": self.proposal_count,
                    "materialized_novel_member_count": len(actions),
                    "real_evaluations": 0,
                    "broker_selected_subset_only": True,
                    "candidate_outcomes_observed": False,
                }
            ),
        )
        self._proposal = proposal
        return proposal

    async def evaluate(
        self,
        proposal: MaterializedActionProposalBatch,
        selected_action_sha256s: tuple[str, ...],
    ) -> MaterializedActionEvaluationBatch:
        if proposal is not self._proposal:
            raise ValueError("expert received a foreign proposal")
        if (
            type(selected_action_sha256s) is not tuple
            or not selected_action_sha256s
            or selected_action_sha256s
            != tuple(sorted(set(selected_action_sha256s)))
        ):
            raise ValueError(
                "selected action hashes must be non-empty and canonical"
            )
        action_by_sha256 = {
            value.action_sha256: value for value in proposal.actions
        }
        try:
            selected_actions = tuple(
                action_by_sha256[value]
                for value in selected_action_sha256s
            )
        except KeyError as error:
            raise ValueError(
                "broker selected outside the expert proposal"
            ) from error
        wave = self._evaluation_ledger.reserve(
            proposal,
            selected_action_sha256s,
        )
        evaluations = tuple(
            await asyncio.gather(
                *(
                    self.evaluator.evaluate(action)
                    for action in selected_actions
                )
            )
        )
        for action, evaluation in zip(
            selected_actions,
            evaluations,
            strict=True,
        ):
            if type(evaluation) is not MaterializedActionEvaluation:
                raise TypeError("evaluation port returned a foreign value")
            evaluation.__post_init__()
            if evaluation.action != action:
                raise ValueError(
                    "evaluation port returned another selected action"
                )
        return MaterializedActionEvaluationBatch(
            proposal_sha256=proposal.proposal_sha256,
            expert_id=self.expert_id,
            expert_version=self.expert_version,
            expert_definition_sha256=self.definition_sha256,
            selected_action_sha256s=selected_action_sha256s,
            evaluations=evaluations,
            evidence=freeze_json(
                {
                    "schema_version": 1,
                    "evaluator": {
                        "evaluator_id": self.evaluator.evaluator_id,
                        "evaluator_version": self.evaluator.evaluator_version,
                        "definition_sha256": (
                            self.evaluator.definition_sha256
                        ),
                    },
                    "real_evaluator_calls": len(evaluations),
                    "evaluation_wave": wave.to_record(),
                    "broker_selected_subset_only": True,
                }
            ),
        )


__all__ = [
    "HierarchicalResidualExpertSpec",
    "HierarchicalResidualProposalPolicyPort",
    "MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_DEFINITION_SHA256",
    "MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_ID",
    "MATERIALIZED_HIERARCHICAL_RESIDUAL_EXPERT_ADAPTER_VERSION",
    "MaterializedHierarchicalResidualActionEvidence",
    "MaterializedHierarchicalResidualActionEvidencePort",
    "MaterializedPhenotypeProjectionPort",
    "PydanticAIMaterializedHierarchicalResidualExpert",
    "ResidualParentActionContext",
    "SelectedMaterializedActionEvaluationPort",
]

"""One-call ranked portfolio selection with concurrent exact evaluation.

The model selects only opaque IDs from a parent-bound finite contract.  This
application service resolves every ranked ID, derives an exact parent-relative
patch, materializes every sealed child under engine authority, and submits the
whole wave through ``AgenticEvolutionEngine.run_materialized_invocations``.
Optional insight credit is recorded once for the complete wave, never once per
child.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    MaterializedInvocation,
    OperatorKind,
    ProposalAuthority,
    RewardPolicyBinding,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightOrigin,
    QuarantineTestAdmissionReceipt,
)
from agent_evolve.application.outcome_relation import OutcomeRelation
from agent_evolve.application.portfolio_memory_matched_control import (
    PortfolioMemoryMatchedArmAssignment,
    PortfolioMemoryMatchedArmView,
    PortfolioMemoryMatchedControlPlan,
)
from agent_evolve.domain.ids import (
    ArtifactId,
    CandidateId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.outcome import (
    FailureCategory,
    FailureCode,
    FailureRecord,
    validate_failure_pair,
)
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.memory.randomized_subset import (
    InsightSelectionDecision,
    InsightTrial,
)
from agent_evolve.policies.memory.staged_causal import (
    MemoryScoreSnapshot,
    ResolvedInsightAssignment,
    insight_selection_decision_sha256,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    MetricEffectPrediction,
    SourceAttribution,
)
from agent_evolve.ports.id_factory import IdFactory
from agent_evolve.ports.portfolio_selection import (
    PortfolioSelectionPolicy,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    PortfolioSelectionSupplementalAudit,
    RankedPortfolioDecision,
    RankedPortfolioMember,
    validate_ranked_portfolio_decision,
)


PORTFOLIO_MATERIALIZATION_POLICY_ID = "ranked_portfolio_exact_finite_option"
PORTFOLIO_MATERIALIZATION_POLICY_VERSION = 1

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_TELEMETRY_DOMAIN = b"agent-evolve:portfolio-evolution-telemetry:v1\x00"
_MATERIALIZATION_DOMAIN = b"agent-evolve:portfolio-evolution-materialization:v1\x00"
_MEMBER_DOMAIN = b"agent-evolve:portfolio-evolution-member:v1\x00"
_CANDIDATE_FAILURE_DOMAIN = (
    b"agent-evolve:portfolio-candidate-failure-evidence:v1\x00"
)
_ACTION_CARD_ATTRIBUTION_DOMAIN = (
    b"agent-evolve:portfolio-action-card-attribution:v1\x00"
)
_ACTION_ATTRIBUTION_DOMAIN = b"agent-evolve:portfolio-action-attribution:v1\x00"
_MEMORY_CREDIT_DOMAIN = b"agent-evolve:portfolio-evolution-memory-credit:v1\x00"
_PENDING_MEMORY_CREDIT_DOMAIN = (
    b"agent-evolve:portfolio-evolution-pending-memory-credit:v1\x00"
)
_MEMORY_CREDIT_BATCH_DOMAIN = (
    b"agent-evolve:portfolio-evolution-memory-credit-batch:v1\x00"
)
_MEMORY_CREDIT_BATCH_PREPARATION_DOMAIN = (
    b"agent-evolve:portfolio-evolution-memory-credit-batch-preparation:v1\x00"
)
_MEMORY_TREATMENT_BINDING_DOMAIN = (
    b"agent-evolve:portfolio-memory-treatment-binding:v1\x00"
)
_MEMORY_CONTEXT_PROJECTION_DOMAIN = (
    b"agent-evolve:portfolio-memory-context-projection:v1\x00"
)
_WAVE_DOMAIN = b"agent-evolve:portfolio-evolution-wave:v1\x00"
_AGGREGATION_BINDING_DOMAIN = (
    b"agent-evolve:portfolio-reward-aggregation-binding:v1\x00"
)


def _canonical_json(record: dict[str, object]) -> bytes:
    return json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash_record(domain: bytes, record: dict[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json(record)).hexdigest()


def _require_token(value: str, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _path_text(path: JsonPath) -> str:
    """Project a typed patch path into the engine's candidate-path vocabulary."""

    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported JSON-path segment")
    return "".join(parts)


def _telemetry_record(telemetry: AgenticCallTelemetry) -> dict[str, object]:
    if type(telemetry) is not AgenticCallTelemetry:
        raise TypeError("selection telemetry must be exact")
    AgenticCallTelemetry.__post_init__(telemetry)
    for name in ("provider_response_id", "finish_reason"):
        value = getattr(telemetry, name)
        if value is not None and type(value) is not str:
            raise TypeError(f"telemetry {name} must be an exact string or None")
    cost = telemetry.cost_usd
    if cost is not None:
        if type(cost) is not Decimal:
            raise TypeError("telemetry cost_usd must be an exact Decimal or None")
        if not cost.is_finite() or cost < 0:
            raise ValueError("telemetry cost_usd must be finite and non-negative")
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
        "cost_usd": None if cost is None else str(cost),
        "latency_ns": telemetry.latency_ns,
        "attempt_count": telemetry.attempt_count,
    }


def portfolio_selection_telemetry_sha256(
    telemetry: AgenticCallTelemetry,
) -> str:
    """Return the exact identity of one ranked-selector call's telemetry."""

    return _hash_record(_TELEMETRY_DOMAIN, _telemetry_record(telemetry))


PortfolioRewardAggregator = Callable[[tuple[InvocationOutcome, ...]], float]


EXACT_MEMORY_CONTEXT_PROJECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portfolio-memory-context-projection:exact-identity:v1"
).hexdigest()
MEMORY_ESTIMAND_CONTEXT_KEY = "memory_estimand_context"
MEMORY_ESTIMAND_SUBTREE_PROJECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portfolio-memory-context-projection:reserved-root-object-subtree:v1"
).hexdigest()


@dataclass(frozen=True, slots=True)
class PortfolioRewardAggregationBinding:
    """Identified aggregate endpoint for one multi-candidate credit unit."""

    aggregate: PortfolioRewardAggregator
    aggregation_id: str
    aggregation_version: int
    definition_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.aggregate):
            raise TypeError("aggregate must be callable")
        _require_token(self.aggregation_id, "aggregation_id")
        if type(self.aggregation_version) is not int or self.aggregation_version <= 0:
            raise ValueError("aggregation_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "aggregation_id": self.aggregation_id,
            "aggregation_version": self.aggregation_version,
            "definition_sha256": self.definition_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash_record(_AGGREGATION_BINDING_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class PortfolioMemoryContextProjectionBinding:
    """Replay one core-owned selector-context to estimand-context projection."""

    estimand_context_sha256: str
    selector_context_sha256: str
    projection_key: str | None

    def __post_init__(self) -> None:
        require_sha256(self.estimand_context_sha256, "estimand_context_sha256")
        require_sha256(self.selector_context_sha256, "selector_context_sha256")
        if self.projection_key not in (None, MEMORY_ESTIMAND_CONTEXT_KEY):
            raise ValueError("projection_key must use a core-owned projection")
        if (
            self.projection_key is None
            and self.estimand_context_sha256 != self.selector_context_sha256
        ):
            raise ValueError("exact context projection requires identical hashes")

    @property
    def projection_id(self) -> str:
        return (
            "exact_context_identity"
            if self.projection_key is None
            else "reserved_memory_estimand_subtree"
        )

    @property
    def projection_version(self) -> int:
        return 1

    @property
    def definition_sha256(self) -> str:
        return (
            EXACT_MEMORY_CONTEXT_PROJECTION_DEFINITION_SHA256
            if self.projection_key is None
            else MEMORY_ESTIMAND_SUBTREE_PROJECTION_DEFINITION_SHA256
        )

    @classmethod
    def exact_identity(cls, context_sha256: str):
        require_sha256(context_sha256, "context_sha256")
        return cls(
            estimand_context_sha256=context_sha256,
            selector_context_sha256=context_sha256,
            projection_key=None,
        )

    @classmethod
    def from_selector_context(
        cls,
        selector_context: FrozenJsonObject,
    ) -> "PortfolioMemoryContextProjectionBinding":
        projected = cls._project_reserved_subtree(selector_context)
        return cls(
            estimand_context_sha256=typed_json_sha256(projected),
            selector_context_sha256=typed_json_sha256(selector_context),
            projection_key=MEMORY_ESTIMAND_CONTEXT_KEY,
        )

    @staticmethod
    def _project_reserved_subtree(
        selector_context: FrozenJsonObject,
    ) -> FrozenJsonObject:
        if type(selector_context) is not FrozenJsonObject:
            raise TypeError("selector_context must be an exact FrozenJsonObject")
        values = dict(selector_context.items)
        projected = values.get(MEMORY_ESTIMAND_CONTEXT_KEY)
        if type(projected) is not FrozenJsonObject:
            raise ValueError(
                "selector context must contain the reserved memory estimand object"
            )
        return projected

    def replay(self, selector_context: FrozenJsonObject) -> FrozenJsonObject:
        self.__post_init__()
        observed_selector_sha256 = typed_json_sha256(selector_context)
        if observed_selector_sha256 != self.selector_context_sha256:
            raise ValueError(
                "context projection selector hash differs from selector context"
            )
        if self.projection_key is None:
            projected = selector_context
        else:
            projected = self._project_reserved_subtree(selector_context)
        if typed_json_sha256(projected) != self.estimand_context_sha256:
            raise ValueError("context projection subtree differs from estimand context")
        return projected

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "estimand_context_sha256": self.estimand_context_sha256,
            "selector_context_sha256": self.selector_context_sha256,
            "projection_path": (
                [] if self.projection_key is None else [self.projection_key]
            ),
            "projection_id": self.projection_id,
            "projection_version": self.projection_version,
            "definition_sha256": self.definition_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash_record(_MEMORY_CONTEXT_PROJECTION_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class PortfolioMemoryCreditPlan:
    """One randomized card assignment credited once across the complete wave."""

    decision: InsightSelectionDecision
    credit_unit_id: OperatorInvocationId
    aggregation: PortfolioRewardAggregationBinding
    card_snapshot_sha256: str
    score_snapshot: MemoryScoreSnapshot
    assignment: ResolvedInsightAssignment
    card_source_registry_sha256: str | None = None
    quarantine_admission: QuarantineTestAdmissionReceipt | None = None
    quarantine_admission_subset_authorization_sha256: str | None = None
    context_projection: PortfolioMemoryContextProjectionBinding | None = None

    def __post_init__(self) -> None:
        if type(self.decision) is not InsightSelectionDecision:
            raise TypeError("decision must be an exact InsightSelectionDecision")
        InsightSelectionDecision.__post_init__(self.decision)
        if not self.decision.credit_identifiable:
            raise ValueError("portfolio memory credit requires identifiable overlap")
        if type(self.credit_unit_id) is not OperatorInvocationId:
            raise TypeError("credit_unit_id must be an exact OperatorInvocationId")
        OperatorInvocationId.__post_init__(self.credit_unit_id)
        if type(self.aggregation) is not PortfolioRewardAggregationBinding:
            raise TypeError("aggregation must be an exact binding")
        PortfolioRewardAggregationBinding.__post_init__(self.aggregation)
        require_sha256(self.card_snapshot_sha256, "card_snapshot_sha256")
        if type(self.score_snapshot) is not MemoryScoreSnapshot:
            raise TypeError("score_snapshot must be an exact MemoryScoreSnapshot")
        MemoryScoreSnapshot.__post_init__(self.score_snapshot)
        if type(self.assignment) is not ResolvedInsightAssignment:
            raise TypeError("assignment must be an exact ResolvedInsightAssignment")
        ResolvedInsightAssignment.__post_init__(self.assignment)
        self.assignment.validate_against_snapshot(self.score_snapshot)
        if (
            self.assignment.credit_unit_id != self.credit_unit_id
            or self.assignment.selection_decision != self.decision
            or self.assignment.prompt_shape_sha256 != self.card_snapshot_sha256
        ):
            raise ValueError(
                "resolved assignment differs from the credit unit, decision, or cards"
            )
        if self.card_source_registry_sha256 is not None:
            require_sha256(
                self.card_source_registry_sha256,
                "card_source_registry_sha256",
            )
        if self.quarantine_admission is not None:
            if type(self.quarantine_admission) is not QuarantineTestAdmissionReceipt:
                raise TypeError("quarantine_admission must be an exact receipt or None")
            QuarantineTestAdmissionReceipt.__post_init__(
                self.quarantine_admission
            )
        subset_authorization = (
            self.quarantine_admission_subset_authorization_sha256
        )
        if subset_authorization is not None:
            if self.quarantine_admission is None:
                raise ValueError(
                    "quarantine subset authorization requires an admission"
                )
            require_sha256(
                subset_authorization,
                "quarantine_admission_subset_authorization_sha256",
            )
        projection = self.context_projection
        if projection is not None:
            if type(projection) is not PortfolioMemoryContextProjectionBinding:
                raise TypeError("context_projection must be an exact binding or None")
            PortfolioMemoryContextProjectionBinding.__post_init__(projection)
            if projection.estimand_context_sha256 != self.decision.context_hash:
                raise ValueError(
                    "context projection estimand differs from memory decision context"
                )

    @property
    def treatment_binding_sha256(self) -> str:
        self.__post_init__()
        return _hash_record(
            _MEMORY_TREATMENT_BINDING_DOMAIN,
            {
                "schema_version": 2,
                "selection_decision_sha256": (
                    insight_selection_decision_sha256(self.decision)
                ),
                "card_snapshot_sha256": self.card_snapshot_sha256,
                "card_source_registry_sha256": (
                    self.card_source_registry_sha256
                ),
                "assignment_receipt_sha256": self.assignment.assignment_sha256,
                "score_snapshot_sha256": self.score_snapshot.snapshot_sha256,
                "quarantine_admission_receipt_sha256": (
                    None
                    if self.quarantine_admission is None
                    else self.quarantine_admission.receipt_sha256
                ),
                "quarantine_admission_subset_authorization_sha256": (
                    self.quarantine_admission_subset_authorization_sha256
                ),
            },
        )

    def resolve_context_projection(
        self,
        selector_context: FrozenJsonObject,
    ) -> PortfolioMemoryContextProjectionBinding:
        self.__post_init__()
        if type(selector_context) is not FrozenJsonObject:
            raise TypeError("selector_context must be an exact FrozenJsonObject")
        selector_context_sha256 = typed_json_sha256(selector_context)
        projection = self.context_projection
        if projection is None:
            if self.decision.context_hash != selector_context_sha256:
                raise ValueError(
                    "memory decision context differs from selector context; an "
                    "explicit authenticated context projection is required"
                )
            return PortfolioMemoryContextProjectionBinding.exact_identity(
                selector_context_sha256
            )
        projection.replay(selector_context)
        if projection.estimand_context_sha256 != self.decision.context_hash:
            raise ValueError(
                "context projection subtree differs from memory decision context"
            )
        return projection


@dataclass(frozen=True, slots=True)
class PortfolioMemoryMatchedControlWavePlan:
    """One arm of a precommitted active-versus-neutral diagnostic pair."""

    plan: PortfolioMemoryMatchedControlPlan
    assignment: PortfolioMemoryMatchedArmAssignment
    arm_view: PortfolioMemoryMatchedArmView
    aggregation: PortfolioRewardAggregationBinding
    context_projection: PortfolioMemoryContextProjectionBinding

    def __post_init__(self) -> None:
        if type(self.plan) is not PortfolioMemoryMatchedControlPlan:
            raise TypeError("plan must be an exact matched-control plan")
        self.plan.__post_init__()
        if type(self.assignment) is not PortfolioMemoryMatchedArmAssignment:
            raise TypeError("assignment must be an exact matched arm assignment")
        self.assignment.__post_init__()
        if self.assignment not in self.plan.assignments:
            raise ValueError("matched arm assignment is outside its plan")
        if type(self.arm_view) is not PortfolioMemoryMatchedArmView:
            raise TypeError("arm_view must be an exact matched arm view")
        self.arm_view.__post_init__()
        if (
            self.arm_view.plan_sha256 != self.plan.plan_sha256
            or self.arm_view.assignment != self.assignment
        ):
            raise ValueError("matched arm view differs from its plan assignment")
        if type(self.aggregation) is not PortfolioRewardAggregationBinding:
            raise TypeError("aggregation must be an exact reward binding")
        self.aggregation.__post_init__()
        if type(self.context_projection) is not PortfolioMemoryContextProjectionBinding:
            raise TypeError("context_projection must be an exact binding")
        self.context_projection.__post_init__()
        if (
            self.context_projection.estimand_context_sha256
            != self.plan.exact_context_sha256
        ):
            raise ValueError("matched context projection differs from its plan")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "plan_sha256": self.plan.plan_sha256,
            "assignment": self.assignment.to_record(),
            "arm_view": self.arm_view.to_record(),
            "aggregation": self.aggregation.to_record(),
            "context_projection": self.context_projection.to_record(),
            "card_vs_neutral_effect_identified": False,
            "online_score_update_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class PortfolioVariationWaveRequest:
    """Parent, generation, selector request, and optional causal credit unit."""

    selection_request: PortfolioSelectionRequest
    parent: EvolutionCandidate
    generation: int
    label_prefix: str
    phase: str = "portfolio_evolution"
    memory_credit: PortfolioMemoryCreditPlan | None = None
    matched_memory_control: PortfolioMemoryMatchedControlWavePlan | None = None

    def __post_init__(self) -> None:
        if type(self.selection_request) is not PortfolioSelectionRequest:
            raise TypeError("selection_request must be exact")
        PortfolioSelectionRequest.__post_init__(self.selection_request)
        if type(self.parent) is not EvolutionCandidate:
            raise TypeError("parent must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(self.parent)
        if not self.parent.valid:
            raise ValueError("portfolio evolution requires a valid parent")
        contract = self.selection_request.finite_variation_contract
        if (
            contract.parent_configuration_sha256
            != self.parent.occurrence.configuration_hash
            or not typed_json_equal(
                contract.parent_configuration,
                self.parent.configuration,
            )
        ):
            raise ValueError("portfolio finite contract is bound to a different parent")
        if (
            type(self.generation) is not int
            or self.generation <= self.parent.generation
        ):
            raise ValueError("portfolio generation must follow the parent generation")
        _require_token(self.label_prefix, "label_prefix")
        _require_token(self.phase, "phase")
        credit = self.memory_credit
        matched = self.matched_memory_control
        if credit is not None and matched is not None:
            raise ValueError("one wave cannot carry legacy and matched memory credit")
        if credit is not None:
            if type(credit) is not PortfolioMemoryCreditPlan:
                raise TypeError("memory_credit must be an exact plan or None")
            PortfolioMemoryCreditPlan.__post_init__(credit)
            credit.resolve_context_projection(self.selection_request.context)
            if (
                credit.card_snapshot_sha256
                != self.selection_request.card_snapshot_sha256
            ):
                raise ValueError("memory credit is bound to a different card snapshot")
            source_registry = self.selection_request.source_registry
            observed_registry_sha256 = (
                None if source_registry is None else source_registry.registry_sha256
            )
            if credit.card_source_registry_sha256 != observed_registry_sha256:
                raise ValueError(
                    "memory credit is bound to a different card source registry"
                )
            card_references = tuple(
                sorted(card.reference for card in self.selection_request.cards)
            )
            if card_references != credit.decision.selected:
                raise ValueError(
                    "selector request card references must equal selected memory "
                    "references"
                )
        if matched is None:
            return
        if type(matched) is not PortfolioMemoryMatchedControlWavePlan:
            raise TypeError("matched_memory_control must be an exact plan or None")
        matched.__post_init__()
        # Generation is a hard identity.  The campaign runtime separately
        # joins the stable lane through its decision slot; label text is not an
        # authority and therefore cannot establish that join here.
        if matched.assignment.unit.generation != self.generation:
            raise ValueError("matched assignment generation differs from wave")
        matched.context_projection.replay(self.selection_request.context)
        view = matched.arm_view
        if self.selection_request.cards != view.cards:
            raise ValueError("matched selector cards differ from the arm view")
        if self.selection_request.source_registry != view.source_registry:
            raise ValueError("matched selector registry differs from the arm view")
        if (
            self.selection_request.experimental_view_receipt
            != view.experimental_view_receipt
        ):
            raise ValueError("matched selector receipt differs from the arm view")
        if (
            self.selection_request.candidate_pool_required_option_ids
            != view.required_common_pool_option_ids
        ):
            raise ValueError("matched selector required actions differ from arm view")
        dose = self.selection_request.memory_dose_contract
        if view.memory_dose_allowed:
            if dose is None:
                raise ValueError("matched M arm requires an administered memory dose")
        elif dose is not None:
            raise ValueError("matched N arm cannot carry a memory dose")


@dataclass(frozen=True, slots=True)
class PortfolioMemberMaterializationReceipt:
    """Replay identity for one exact finite option resolved by rank."""

    request_sha256: str
    decision_sha256: str
    selection_telemetry_sha256: str
    rank: int
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    parent_candidate_id: CandidateId
    parent_configuration_sha256: str
    generation: int
    candidate_id: CandidateId
    patch_sha256: str
    changed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "decision_sha256",
            "selection_telemetry_sha256",
            "option_identity_sha256",
            "child_configuration_sha256",
            "parent_configuration_sha256",
            "patch_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be exact")
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be exact")
        CandidateId.__post_init__(self.parent_candidate_id)
        CandidateId.__post_init__(self.candidate_id)
        if self.parent_candidate_id == self.candidate_id:
            raise ValueError("materialized child cannot reuse the parent occurrence")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.changed_paths) is not tuple or any(
            type(path) is not str or not path.startswith("$.")
            for path in self.changed_paths
        ):
            raise TypeError("changed_paths must be exact candidate paths")
        if not self.changed_paths:
            raise ValueError("finite option materialization must change a path")
        if self.changed_paths != tuple(sorted(set(self.changed_paths))):
            raise ValueError("changed_paths must be unique and canonical")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "selection_authority": "model_ranked_finite_option_ids",
            "materialization_authority": "engine_exact_sealed_children",
            "materialization_policy_id": PORTFOLIO_MATERIALIZATION_POLICY_ID,
            "materialization_policy_version": (
                PORTFOLIO_MATERIALIZATION_POLICY_VERSION
            ),
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
            "selection_telemetry_sha256": self.selection_telemetry_sha256,
            "rank": self.rank,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "generation": self.generation,
            "candidate_id": self.candidate_id.value,
            "patch_sha256": self.patch_sha256,
            "changed_paths": list(self.changed_paths),
            "source_attribution": [
                {"path": path, "source": "mutation"} for path in self.changed_paths
            ],
            "model_authored_configuration_fields": 0,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash_record(_MATERIALIZATION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


class PortfolioMemberDisposition(str, Enum):
    """Closed terminal meanings for a completely evaluated portfolio member."""

    SCORED = "scored"
    CANDIDATE_INFEASIBLE = "candidate_infeasible"


@dataclass(frozen=True, slots=True)
class PortfolioCandidateFailureEvidence:
    """Content-minimized proof of candidate-attributable infeasibility.

    The detailed-evaluation digest authenticates the full evaluator record.  The
    compact projection makes the terminal category usable without copying a
    potentially sensitive free-text diagnostic into every portfolio receipt.
    """

    detailed_evaluation_sha256: str
    failure_code: FailureCode
    failure_message_sha256: str
    retryable: bool
    exception_type: str | None
    diagnostics_artifact_id: ArtifactId | None

    def __post_init__(self) -> None:
        require_sha256(
            self.detailed_evaluation_sha256,
            "detailed_evaluation_sha256",
        )
        if type(self.failure_code) is not FailureCode:
            raise TypeError("failure_code must be an exact FailureCode")
        # Reuse the closed domain taxonomy rather than maintaining a second
        # hand-written list of candidate-attributable codes here.
        validate_failure_pair(FailureCategory.CANDIDATE, self.failure_code)
        require_sha256(self.failure_message_sha256, "failure_message_sha256")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be an exact boolean")
        if self.exception_type is not None and (
            type(self.exception_type) is not str
            or not self.exception_type.strip()
            or self.exception_type != self.exception_type.strip()
        ):
            raise ValueError(
                "exception_type must be canonical non-empty text or None"
            )
        if self.diagnostics_artifact_id is not None:
            if type(self.diagnostics_artifact_id) is not ArtifactId:
                raise TypeError(
                    "diagnostics_artifact_id must be an exact ArtifactId or None"
                )
            ArtifactId.__post_init__(self.diagnostics_artifact_id)

    @classmethod
    def from_failure_record(
        cls,
        failure: FailureRecord,
        *,
        detailed_evaluation_sha256: str,
    ) -> PortfolioCandidateFailureEvidence:
        """Project one exact candidate-category evaluator failure."""

        if type(failure) is not FailureRecord:
            raise TypeError("failure must be an exact FailureRecord")
        FailureRecord.__post_init__(failure)
        if failure.category is not FailureCategory.CANDIDATE:
            raise ValueError(
                "portfolio infeasibility requires candidate-category evidence"
            )
        return cls(
            detailed_evaluation_sha256=detailed_evaluation_sha256,
            failure_code=failure.code,
            failure_message_sha256=hashlib.sha256(
                failure.message.encode("utf-8", errors="strict")
            ).hexdigest(),
            retryable=failure.retryable,
            exception_type=failure.exception_type,
            diagnostics_artifact_id=failure.diagnostics_artifact_id,
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "failure_category": FailureCategory.CANDIDATE.value,
            "failure_code": self.failure_code.value,
            "failure_message_sha256": self.failure_message_sha256,
            "retryable": self.retryable,
            "exception_type": self.exception_type,
            "diagnostics_artifact_id": (
                None
                if self.diagnostics_artifact_id is None
                else self.diagnostics_artifact_id.value
            ),
            "detailed_evaluation_sha256": self.detailed_evaluation_sha256,
        }

    @property
    def evidence_sha256(self) -> str:
        return _hash_record(_CANDIDATE_FAILURE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioVariationMemberReceipt:
    """Closed materialization-to-outcome join for one ranked member."""

    materialization: PortfolioMemberMaterializationReceipt
    operator_invocation_id: OperatorInvocationId
    reward_definition_sha256: str
    reward: float
    parent_relations: tuple[OutcomeRelation, ...]
    detailed_evaluation_sha256: str | None
    dominates_any_parent: bool
    better_than_any_parent: bool
    disposition: PortfolioMemberDisposition = PortfolioMemberDisposition.SCORED
    candidate_failure: PortfolioCandidateFailureEvidence | None = None

    def __post_init__(self) -> None:
        if type(self.materialization) is not PortfolioMemberMaterializationReceipt:
            raise TypeError("materialization must be an exact receipt")
        PortfolioMemberMaterializationReceipt.__post_init__(self.materialization)
        if type(self.operator_invocation_id) is not OperatorInvocationId:
            raise TypeError("operator_invocation_id must be exact")
        OperatorInvocationId.__post_init__(self.operator_invocation_id)
        require_sha256(self.reward_definition_sha256, "reward_definition_sha256")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        if type(self.parent_relations) is not tuple or any(
            type(relation) is not OutcomeRelation for relation in self.parent_relations
        ):
            raise TypeError("parent_relations must contain exact values")
        if self.detailed_evaluation_sha256 is not None:
            require_sha256(
                self.detailed_evaluation_sha256,
                "detailed_evaluation_sha256",
            )
        if (
            type(self.dominates_any_parent) is not bool
            or type(self.better_than_any_parent) is not bool
        ):
            raise TypeError("outcome comparison projections must be bool")
        if type(self.disposition) is not PortfolioMemberDisposition:
            raise TypeError("disposition must be a PortfolioMemberDisposition")
        failure = self.candidate_failure
        if self.disposition is PortfolioMemberDisposition.SCORED:
            if len(self.parent_relations) != 1:
                raise ValueError("scored portfolio member must compare to its parent")
            if failure is not None:
                raise ValueError("scored portfolio member cannot carry failure evidence")
        else:
            if type(failure) is not PortfolioCandidateFailureEvidence:
                raise TypeError(
                    "candidate-infeasible member requires exact failure evidence"
                )
            PortfolioCandidateFailureEvidence.__post_init__(failure)
            if self.detailed_evaluation_sha256 != failure.detailed_evaluation_sha256:
                raise ValueError(
                    "candidate failure identifies another detailed evaluation"
                )
            if self.parent_relations:
                raise ValueError(
                    "candidate-infeasible member cannot publish parent relations"
                )
            if self.dominates_any_parent or self.better_than_any_parent:
                raise ValueError(
                    "candidate-infeasible member cannot publish improvement flags"
                )

    @property
    def engine_reward(self) -> float:
        """Return the exact terminal reward published by the engine."""

        return self.reward

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "materialization": self.materialization.to_record(),
            "operator_invocation_id": self.operator_invocation_id.value,
            "candidate_id": self.materialization.candidate_id.value,
            "candidate_configuration_sha256": (
                self.materialization.child_configuration_sha256
            ),
            "disposition": self.disposition.value,
            "candidate_valid": (
                self.disposition is PortfolioMemberDisposition.SCORED
            ),
            "operator_compliant": True,
            "evidence_compliant": True,
            "reward_definition_sha256": self.reward_definition_sha256,
            "engine_reward_hex": self.engine_reward.hex(),
            "reward_hex": self.reward.hex(),
            "parent_relations": [value.value for value in self.parent_relations],
            "detailed_evaluation_sha256": self.detailed_evaluation_sha256,
            "candidate_failure": (
                None
                if self.candidate_failure is None
                else self.candidate_failure.to_record()
            ),
            "dominates_any_parent": self.dominates_any_parent,
            "better_than_any_parent": self.better_than_any_parent,
        }

    @property
    def outcome_sha256(self) -> str:
        return _hash_record(_MEMBER_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "outcome_sha256": self.outcome_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioActionCardAttribution:
    """Exact request-card identity cited for one selected finite action."""

    card_key: str
    reference: InsightRef
    content_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _require_token(self.card_key, "card_key")
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        require_sha256(self.content_sha256, "content_sha256")
        require_sha256(self.evidence_sha256, "evidence_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "card_key": self.card_key,
            "reference": {
                "insight_id": self.reference.insight_id.value,
                "version": self.reference.version,
            },
            "content_sha256": self.content_sha256,
            "evidence_sha256": self.evidence_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash_record(
            _ACTION_CARD_ATTRIBUTION_DOMAIN,
            self._unsigned_record(),
        )

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioActionAttributionReceipt:
    """Authenticated selector-card to materialized-outcome join for one action.

    This is diagnostic attribution, not a causal credit unit. The randomized
    memory treatment remains one whole-wave ITT trial in
    :class:`PortfolioMemoryCreditReceipt`.
    """

    request_sha256: str
    decision_sha256: str
    card_snapshot_sha256: str
    rank: int
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    family: str
    supporting_cards: tuple[PortfolioActionCardAttribution, ...]
    effect_predictions: tuple[MetricEffectPrediction, ...]
    design_rationale_sha256: str
    materialization_receipt_sha256: str
    outcome_sha256: str
    operator_invocation_id: OperatorInvocationId
    candidate_id: CandidateId

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "decision_sha256",
            "card_snapshot_sha256",
            "option_identity_sha256",
            "child_configuration_sha256",
            "design_rationale_sha256",
            "materialization_receipt_sha256",
            "outcome_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        _require_token(self.family, "family")
        if type(self.supporting_cards) is not tuple or any(
            type(value) is not PortfolioActionCardAttribution
            for value in self.supporting_cards
        ):
            raise TypeError(
                "supporting_cards must contain exact card-attribution values"
            )
        for value in self.supporting_cards:
            PortfolioActionCardAttribution.__post_init__(value)
        card_keys = tuple(value.card_key for value in self.supporting_cards)
        if card_keys != tuple(sorted(set(card_keys))):
            raise ValueError("supporting cards must use canonical unique card keys")
        references = tuple(value.reference for value in self.supporting_cards)
        if len(set(references)) != len(references):
            raise ValueError("supporting cards cannot repeat an insight reference")
        if type(self.effect_predictions) is not tuple or any(
            type(value) is not MetricEffectPrediction
            for value in self.effect_predictions
        ):
            raise TypeError(
                "effect_predictions must contain exact MetricEffectPrediction values"
            )
        for value in self.effect_predictions:
            MetricEffectPrediction.__post_init__(value)
        metric_ids = tuple(value.metric_id for value in self.effect_predictions)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("effect predictions must use canonical metric order")
        if type(self.operator_invocation_id) is not OperatorInvocationId:
            raise TypeError("operator_invocation_id must be exact")
        OperatorInvocationId.__post_init__(self.operator_invocation_id)
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be exact")
        CandidateId.__post_init__(self.candidate_id)

    @property
    def supporting_card_keys(self) -> tuple[str, ...]:
        self.__post_init__()
        return tuple(value.card_key for value in self.supporting_cards)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "attribution_scope": "post_treatment_diagnostic_not_causal_credit",
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
            "card_snapshot_sha256": self.card_snapshot_sha256,
            "selected_member": {
                "rank": self.rank,
                "option_id": self.option_id,
                "option_identity_sha256": self.option_identity_sha256,
                "child_configuration_sha256": self.child_configuration_sha256,
                "family": self.family,
                "supporting_card_keys": list(self.supporting_card_keys),
                "effect_predictions": [
                    {
                        "metric_id": value.metric_id,
                        "direction": value.direction.value,
                    }
                    for value in self.effect_predictions
                ],
                "design_rationale_sha256": self.design_rationale_sha256,
            },
            "supporting_cards": [value.to_record() for value in self.supporting_cards],
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "outcome_sha256": self.outcome_sha256,
            "operator_invocation_id": self.operator_invocation_id.value,
            "candidate_id": self.candidate_id.value,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash_record(_ACTION_ATTRIBUTION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioPendingMemoryCredit:
    """Complete credit evidence awaiting a generation publication barrier."""

    credit_unit_id: OperatorInvocationId
    decision: InsightSelectionDecision
    candidate_ids: tuple[CandidateId, ...]
    aggregation: PortfolioRewardAggregationBinding
    context_projection: PortfolioMemoryContextProjectionBinding
    reward: float
    treatment_binding_sha256: str
    generation: int

    def __post_init__(self) -> None:
        if type(self.credit_unit_id) is not OperatorInvocationId:
            raise TypeError("credit_unit_id must be exact")
        OperatorInvocationId.__post_init__(self.credit_unit_id)
        if type(self.decision) is not InsightSelectionDecision:
            raise TypeError("decision must be an exact InsightSelectionDecision")
        InsightSelectionDecision.__post_init__(self.decision)
        if type(self.candidate_ids) is not tuple or any(
            type(value) is not CandidateId for value in self.candidate_ids
        ):
            raise TypeError("candidate_ids must contain exact CandidateId values")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(
            self.candidate_ids
        ):
            raise ValueError("candidate_ids must be non-empty and unique")
        if type(self.aggregation) is not PortfolioRewardAggregationBinding:
            raise TypeError("aggregation must be an exact binding")
        PortfolioRewardAggregationBinding.__post_init__(self.aggregation)
        if type(self.context_projection) is not PortfolioMemoryContextProjectionBinding:
            raise TypeError("context_projection must be an exact binding")
        PortfolioMemoryContextProjectionBinding.__post_init__(self.context_projection)
        if (
            self.context_projection.estimand_context_sha256
            != self.decision.context_hash
        ):
            raise ValueError("context projection differs from decision context")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        require_sha256(
            self.treatment_binding_sha256,
            "treatment_binding_sha256",
        )
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")

    @property
    def selection_decision_sha256(self) -> str:
        return insight_selection_decision_sha256(self.decision)

    def to_trial(self) -> InsightTrial:
        self.__post_init__()
        return InsightTrial(
            credit_unit_id=self.credit_unit_id,
            candidate_ids=self.candidate_ids,
            reward_definition_hash=self.aggregation.definition_sha256,
            decision=self.decision,
            reward=self.reward,
            treatment_binding_sha256=self.treatment_binding_sha256,
            generation=self.generation,
        )

    def to_committed_receipt(self) -> "PortfolioMemoryCreditReceipt":
        self.__post_init__()
        return PortfolioMemoryCreditReceipt(
            credit_unit_id=self.credit_unit_id,
            selection_decision_sha256=self.selection_decision_sha256,
            selection_decision_context_sha256=self.decision.context_hash,
            candidate_ids=self.candidate_ids,
            aggregation_id=self.aggregation.aggregation_id,
            aggregation_version=self.aggregation.aggregation_version,
            aggregation_definition_sha256=self.aggregation.definition_sha256,
            aggregation_binding_sha256=self.aggregation.binding_sha256,
            context_projection=self.context_projection,
            reward=self.reward,
            treatment_binding_sha256=self.treatment_binding_sha256,
            generation=self.generation,
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "status": "pending_generation_barrier",
            "credit_unit_id": self.credit_unit_id.value,
            "selection_decision_sha256": self.selection_decision_sha256,
            "selection_decision_context_sha256": self.decision.context_hash,
            "candidate_ids": [value.value for value in self.candidate_ids],
            "aggregation": {
                **self.aggregation.to_record(),
                "binding_sha256": self.aggregation.binding_sha256,
            },
            "context_projection": {
                **self.context_projection.to_record(),
                "binding_sha256": self.context_projection.binding_sha256,
            },
            "reward_hex": self.reward.hex(),
            "treatment_binding_sha256": self.treatment_binding_sha256,
            "generation": self.generation,
        }

    @property
    def pending_sha256(self) -> str:
        return _hash_record(_PENDING_MEMORY_CREDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "pending_sha256": self.pending_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryCreditReceipt:
    """Evidence that a complete portfolio became one memory trial."""

    credit_unit_id: OperatorInvocationId
    selection_decision_sha256: str
    selection_decision_context_sha256: str
    candidate_ids: tuple[CandidateId, ...]
    aggregation_id: str
    aggregation_version: int
    aggregation_definition_sha256: str
    aggregation_binding_sha256: str
    context_projection: PortfolioMemoryContextProjectionBinding
    reward: float
    treatment_binding_sha256: str
    generation: int

    def __post_init__(self) -> None:
        if type(self.credit_unit_id) is not OperatorInvocationId:
            raise TypeError("credit_unit_id must be exact")
        OperatorInvocationId.__post_init__(self.credit_unit_id)
        for name in (
            "selection_decision_sha256",
            "selection_decision_context_sha256",
            "aggregation_definition_sha256",
            "aggregation_binding_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.candidate_ids) is not tuple or any(
            type(value) is not CandidateId for value in self.candidate_ids
        ):
            raise TypeError("candidate_ids must contain exact CandidateId values")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(
            self.candidate_ids
        ):
            raise ValueError("candidate_ids must be non-empty and unique")
        _require_token(self.aggregation_id, "aggregation_id")
        if type(self.aggregation_version) is not int or self.aggregation_version <= 0:
            raise ValueError("aggregation_version must be positive")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        require_sha256(
            self.treatment_binding_sha256,
            "treatment_binding_sha256",
        )
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if type(self.context_projection) is not PortfolioMemoryContextProjectionBinding:
            raise TypeError("context_projection must be an exact binding")
        PortfolioMemoryContextProjectionBinding.__post_init__(self.context_projection)
        if (
            self.context_projection.estimand_context_sha256
            != self.selection_decision_context_sha256
        ):
            raise ValueError(
                "context projection estimand differs from decision context receipt"
            )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "credit_unit_id": self.credit_unit_id.value,
            "selection_decision_sha256": self.selection_decision_sha256,
            "selection_decision_context_sha256": (
                self.selection_decision_context_sha256
            ),
            "candidate_ids": [value.value for value in self.candidate_ids],
            "aggregation": {
                "aggregation_id": self.aggregation_id,
                "aggregation_version": self.aggregation_version,
                "definition_sha256": self.aggregation_definition_sha256,
                "binding_sha256": self.aggregation_binding_sha256,
            },
            "context_projection": {
                **self.context_projection.to_record(),
                "binding_sha256": self.context_projection.binding_sha256,
            },
            "reward_hex": self.reward.hex(),
            "treatment_binding_sha256": self.treatment_binding_sha256,
            "generation": self.generation,
            "memory_trial_count": 1,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash_record(_MEMORY_CREDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryCreditBatchReceipt:
    """Canonical publication receipt for one generation's pending credits."""

    generation: int
    credits: tuple[PortfolioMemoryCreditReceipt, ...]
    memory_trial_count_before: int
    memory_trial_count_after: int

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if type(self.credits) is not tuple or not self.credits:
            raise ValueError("credits must be a non-empty exact tuple")
        if any(type(value) is not PortfolioMemoryCreditReceipt for value in self.credits):
            raise TypeError("credits must contain exact credit receipts")
        for value in self.credits:
            PortfolioMemoryCreditReceipt.__post_init__(value)
        if any(value.generation != self.generation for value in self.credits):
            raise ValueError("memory credits differ from the batch generation")
        credit_ids = tuple(value.credit_unit_id.value for value in self.credits)
        if credit_ids != tuple(sorted(set(credit_ids))):
            raise ValueError("credits must use canonical unique credit-unit order")
        for name in ("memory_trial_count_before", "memory_trial_count_after"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.memory_trial_count_after - self.memory_trial_count_before != len(
            self.credits
        ):
            raise ValueError("memory trial counts differ from the committed batch")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "publication_scope": "post_concurrent_generation_barrier",
            "generation": self.generation,
            "memory_trial_count_before": self.memory_trial_count_before,
            "memory_trial_count_after": self.memory_trial_count_after,
            "credit_count": len(self.credits),
            "credits": [value.to_record() for value in self.credits],
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash_record(_MEMORY_CREDIT_BATCH_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioVariationWaveReceipt:
    """Closed one-selector-call to N-evaluation portfolio receipt."""

    selection_call_id: LLMCallId
    request_sha256: str
    decision_sha256: str
    selection_policy_id: str
    selection_policy_version: int
    selection_policy_definition_sha256: str
    selection_telemetry: AgenticCallTelemetry
    selection_telemetry_sha256: str
    parent_candidate_id: CandidateId
    parent_configuration_sha256: str
    generation: int
    members: tuple[PortfolioVariationMemberReceipt, ...]
    memory_credit: PortfolioMemoryCreditReceipt | None = None
    action_attributions: tuple[PortfolioActionAttributionReceipt, ...] = ()

    def __post_init__(self) -> None:
        if type(self.selection_call_id) is not LLMCallId:
            raise TypeError("selection_call_id must be exact")
        LLMCallId.__post_init__(self.selection_call_id)
        for name in (
            "request_sha256",
            "decision_sha256",
            "selection_policy_definition_sha256",
            "selection_telemetry_sha256",
            "parent_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_token(self.selection_policy_id, "selection_policy_id")
        if (
            type(self.selection_policy_version) is not int
            or self.selection_policy_version <= 0
        ):
            raise ValueError("selection_policy_version must be positive")
        if type(self.selection_telemetry) is not AgenticCallTelemetry:
            raise TypeError("selection_telemetry must be exact")
        if self.selection_telemetry_sha256 != portfolio_selection_telemetry_sha256(
            self.selection_telemetry
        ):
            raise ValueError("selection telemetry digest does not verify")
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be exact")
        CandidateId.__post_init__(self.parent_candidate_id)
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.members) is not tuple or not self.members:
            raise ValueError("members must be a non-empty exact tuple")
        for member in self.members:
            if type(member) is not PortfolioVariationMemberReceipt:
                raise TypeError("members must contain exact receipts")
            PortfolioVariationMemberReceipt.__post_init__(member)
        if tuple(member.materialization.rank for member in self.members) != tuple(
            range(1, len(self.members) + 1)
        ):
            raise ValueError("member receipts must use contiguous ranked order")
        for member in self.members:
            materialization = member.materialization
            if (
                materialization.request_sha256 != self.request_sha256
                or materialization.decision_sha256 != self.decision_sha256
                or materialization.selection_telemetry_sha256
                != self.selection_telemetry_sha256
                or materialization.parent_candidate_id != self.parent_candidate_id
                or materialization.parent_configuration_sha256
                != self.parent_configuration_sha256
                or materialization.generation != self.generation
            ):
                raise ValueError("member receipt differs from its wave identity")
        for values, name in (
            (
                tuple(member.materialization.option_id for member in self.members),
                "option IDs",
            ),
            (
                tuple(member.materialization.candidate_id for member in self.members),
                "candidate IDs",
            ),
            (
                tuple(
                    member.materialization.child_configuration_sha256
                    for member in self.members
                ),
                "child configurations",
            ),
            (
                tuple(member.operator_invocation_id for member in self.members),
                "operator invocations",
            ),
            (
                tuple(member.materialization.receipt_sha256 for member in self.members),
                "materialization receipts",
            ),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"portfolio wave contains colliding {name}")
        if type(self.action_attributions) is not tuple or any(
            type(value) is not PortfolioActionAttributionReceipt
            for value in self.action_attributions
        ):
            raise TypeError(
                "action_attributions must contain exact attribution receipts"
            )
        for value in self.action_attributions:
            PortfolioActionAttributionReceipt.__post_init__(value)
        if self.action_attributions:
            if len(self.action_attributions) != len(self.members):
                raise ValueError(
                    "action attributions must exactly cover the wave members"
                )
            if (
                len({value.card_snapshot_sha256 for value in self.action_attributions})
                != 1
            ):
                raise ValueError(
                    "action attributions must share one request-card snapshot"
                )
            if tuple(value.rank for value in self.action_attributions) != tuple(
                range(1, len(self.members) + 1)
            ):
                raise ValueError("action attributions must use contiguous ranked order")
            for attribution, member in zip(
                self.action_attributions,
                self.members,
                strict=True,
            ):
                materialization = member.materialization
                if (
                    attribution.request_sha256 != self.request_sha256
                    or attribution.decision_sha256 != self.decision_sha256
                    or attribution.rank != materialization.rank
                    or attribution.option_id != materialization.option_id
                    or attribution.option_identity_sha256
                    != materialization.option_identity_sha256
                    or attribution.child_configuration_sha256
                    != materialization.child_configuration_sha256
                    or attribution.materialization_receipt_sha256
                    != materialization.receipt_sha256
                    or attribution.outcome_sha256 != member.outcome_sha256
                    or attribution.operator_invocation_id
                    != member.operator_invocation_id
                    or attribution.candidate_id != materialization.candidate_id
                ):
                    raise ValueError(
                        "action attribution differs from its evaluated wave member"
                    )
            if len({value.receipt_sha256 for value in self.action_attributions}) != len(
                self.action_attributions
            ):
                raise ValueError("portfolio wave repeats an action attribution")
        if self.memory_credit is not None:
            if type(self.memory_credit) is not PortfolioMemoryCreditReceipt:
                raise TypeError("memory_credit must be an exact receipt or None")
            PortfolioMemoryCreditReceipt.__post_init__(self.memory_credit)
            if self.memory_credit.candidate_ids != tuple(
                member.materialization.candidate_id for member in self.members
            ):
                raise ValueError("memory credit candidate IDs differ from the wave")
            if self.memory_credit.credit_unit_id in {
                member.operator_invocation_id for member in self.members
            }:
                raise ValueError("memory credit unit collides with a child invocation")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "selection_call_count": 1,
            "selection_call_id": self.selection_call_id.value,
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
            "selection_policy": {
                "policy_id": self.selection_policy_id,
                "policy_version": self.selection_policy_version,
                "definition_sha256": self.selection_policy_definition_sha256,
            },
            "selection_telemetry": _telemetry_record(self.selection_telemetry),
            "selection_telemetry_sha256": self.selection_telemetry_sha256,
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "generation": self.generation,
            "member_count": len(self.members),
            "concurrent_materialized_evaluation_wave": True,
            "members": [member.to_record() for member in self.members],
            **(
                {}
                if not self.action_attributions
                else {
                    "action_attributions": [
                        value.to_record() for value in self.action_attributions
                    ]
                }
            ),
            "memory_credit": (
                None if self.memory_credit is None else self.memory_credit.to_record()
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash_record(_WAVE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioVariationWaveResult:
    """Planner-facing wave result with an exact receipt-to-outcome join.

    The receipt remains the evidence-only serialization boundary.  Outcomes
    remain typed runtime values so a generic planner can promote candidates to
    later generations or pass the complete wave to reflection without parsing
    evidence records back into domain objects.
    """

    receipt: PortfolioVariationWaveReceipt
    outcomes: tuple[InvocationOutcome, ...]
    selection_decision: RankedPortfolioDecision | None = None
    supplemental_selection_audit: PortfolioSelectionSupplementalAudit | None = None
    pending_memory_credit: PortfolioPendingMemoryCredit | None = None

    def __post_init__(self) -> None:
        if type(self.receipt) is not PortfolioVariationWaveReceipt:
            raise TypeError("receipt must be an exact PortfolioVariationWaveReceipt")
        PortfolioVariationWaveReceipt.__post_init__(self.receipt)
        if type(self.outcomes) is not tuple or any(
            type(outcome) is not InvocationOutcome for outcome in self.outcomes
        ):
            raise TypeError("outcomes must contain exact InvocationOutcome values")
        if len(self.outcomes) != len(self.receipt.members):
            raise ValueError("result outcomes differ from the receipt member count")
        pending = self.pending_memory_credit
        if pending is not None:
            if type(pending) is not PortfolioPendingMemoryCredit:
                raise TypeError(
                    "pending_memory_credit must be an exact pending credit or None"
                )
            PortfolioPendingMemoryCredit.__post_init__(pending)
            if self.receipt.memory_credit is not None:
                raise ValueError(
                    "a wave result cannot be both pending and memory-committed"
                )
            if pending.candidate_ids != tuple(
                member.materialization.candidate_id for member in self.receipt.members
            ):
                raise ValueError("pending memory credit candidate IDs differ from wave")
        decision = self.selection_decision
        if decision is not None:
            if type(decision) is not RankedPortfolioDecision:
                raise TypeError(
                    "selection_decision must be an exact RankedPortfolioDecision "
                    "or None"
                )
            RankedPortfolioDecision.__post_init__(decision)
            if (
                decision.decision_sha256 != self.receipt.decision_sha256
                or decision.request_sha256 != self.receipt.request_sha256
                or decision.policy_id != self.receipt.selection_policy_id
                or decision.policy_version != self.receipt.selection_policy_version
                or decision.policy_definition_sha256
                != self.receipt.selection_policy_definition_sha256
            ):
                raise ValueError(
                    "selection decision differs from the wave receipt identity"
                )
            if len(decision.members) != len(self.receipt.members):
                raise ValueError(
                    "selection decision differs from the wave materialization count"
                )
            for selected, received in zip(
                decision.members,
                self.receipt.members,
                strict=True,
            ):
                materialization = received.materialization
                if (
                    selected.rank != materialization.rank
                    or selected.option_id != materialization.option_id
                    or selected.option_identity_sha256
                    != materialization.option_identity_sha256
                    or selected.child_configuration_sha256
                    != materialization.child_configuration_sha256
                ):
                    raise ValueError(
                        "selection decision differs from a wave materialization"
                    )
            if self.receipt.action_attributions:
                for selected, attribution in zip(
                    decision.members,
                    self.receipt.action_attributions,
                    strict=True,
                ):
                    rationale_sha256 = hashlib.sha256(
                        selected.design_rationale.encode(
                            "utf-8",
                            errors="strict",
                        )
                    ).hexdigest()
                    if (
                        attribution.decision_sha256 != decision.decision_sha256
                        or attribution.card_snapshot_sha256
                        != decision.card_snapshot_sha256
                        or attribution.rank != selected.rank
                        or attribution.option_id != selected.option_id
                        or attribution.option_identity_sha256
                        != selected.option_identity_sha256
                        or attribution.child_configuration_sha256
                        != selected.child_configuration_sha256
                        or attribution.family != selected.family
                        or attribution.supporting_card_keys
                        != selected.supporting_card_keys
                        or attribution.effect_predictions != selected.effect_predictions
                        or attribution.design_rationale_sha256 != rationale_sha256
                    ):
                        raise ValueError(
                            "action attribution differs from the selection decision"
                        )
        audit = self.supplemental_selection_audit
        if audit is not None:
            if type(audit) is not PortfolioSelectionSupplementalAudit:
                raise TypeError("supplemental_selection_audit must be exact or None")
            audit.__post_init__()
            if decision is None:
                raise ValueError("supplemental selection audit requires a decision")
            if (
                audit.request_sha256 != decision.request_sha256
                or audit.decision_sha256 != decision.decision_sha256
            ):
                raise ValueError(
                    "supplemental selection audit differs from the decision"
                )
        for member, outcome in zip(
            self.receipt.members,
            self.outcomes,
            strict=True,
        ):
            InvocationOutcome.__post_init__(outcome)
            materialization = member.materialization
            prepared = outcome.prepared
            candidate = outcome.candidate
            if (
                outcome.failure_stage is not None
                or candidate is None
                or not candidate.operator_compliant
                or not candidate.evidence_compliant
            ):
                raise ValueError("result contains a failed or partial outcome")
            detailed = candidate.detailed_evaluation
            detailed_sha256 = None if detailed is None else detailed.evidence_sha256
            observed_failure: PortfolioCandidateFailureEvidence | None = None
            if member.disposition is PortfolioMemberDisposition.SCORED:
                if not candidate.valid:
                    raise ValueError(
                        "scored result member contains an invalid candidate"
                    )
            else:
                if candidate.valid or detailed is None or detailed.failure is None:
                    raise ValueError(
                        "candidate-infeasible result lacks detailed failure evidence"
                    )
                observed_failure = (
                    PortfolioCandidateFailureEvidence.from_failure_record(
                        detailed.failure,
                        detailed_evaluation_sha256=detailed.evidence_sha256,
                    )
                )
            if (
                prepared.operator_invocation_id != member.operator_invocation_id
                or prepared.proposal_authority is not ProposalAuthority.ENGINE
                or prepared.call_id is not None
                or prepared.candidate_id != materialization.candidate_id
                or prepared.materialization_policy_id
                != PORTFOLIO_MATERIALIZATION_POLICY_ID
                or prepared.materialization_policy_version
                != PORTFOLIO_MATERIALIZATION_POLICY_VERSION
                or prepared.materialization_receipt_hash
                != materialization.receipt_sha256
                or prepared.variation_case.reward_definition_hash
                != member.reward_definition_sha256
                or candidate.candidate_id != materialization.candidate_id
                or candidate.occurrence.configuration_hash
                != materialization.child_configuration_sha256
                or candidate.generation != materialization.generation
                or candidate.operator_kind is not OperatorKind.TYPED_MUTATION
                or candidate.parent_ids != (materialization.parent_candidate_id,)
                or candidate.parent_patch_hashes != (materialization.patch_sha256,)
                or candidate.call_telemetry is not None
                or tuple(
                    (item.path, item.source) for item in candidate.source_attribution
                )
                != tuple((path, "mutation") for path in materialization.changed_paths)
                or outcome.reward != member.reward
                or outcome.parent_relations != member.parent_relations
                or outcome.dominates_any_parent != member.dominates_any_parent
                or outcome.better_than_any_parent != member.better_than_any_parent
                or detailed_sha256 != member.detailed_evaluation_sha256
                or observed_failure != member.candidate_failure
            ):
                raise ValueError("result outcome differs from its receipt member")

    @property
    def action_attributions(self) -> tuple[PortfolioActionAttributionReceipt, ...]:
        """Return exact per-action attribution receipts in ranked order."""

        self.__post_init__()
        return self.receipt.action_attributions

    @property
    def candidates(self) -> tuple[EvolutionCandidate, ...]:
        """Return the full ranked intention-to-treat candidate population."""

        candidates: list[EvolutionCandidate] = []
        for outcome in self.outcomes:
            candidate = outcome.candidate
            if candidate is None:  # pragma: no cover - closed by __post_init__.
                raise AssertionError("validated portfolio outcome lost its candidate")
            candidates.append(candidate)
        return tuple(candidates)

    @property
    def scored_candidates(self) -> tuple[EvolutionCandidate, ...]:
        """Return only candidates with complete decision-objective vectors."""

        return tuple(
            candidate
            for member, candidate in zip(
                self.receipt.members,
                self.candidates,
                strict=True,
            )
            if member.disposition is PortfolioMemberDisposition.SCORED
        )

    @property
    def infeasible_candidates(self) -> tuple[EvolutionCandidate, ...]:
        """Return evaluated candidate-attributable infeasibilities in rank order."""

        return tuple(
            candidate
            for member, candidate in zip(
                self.receipt.members,
                self.candidates,
                strict=True,
            )
            if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
        )

    @property
    def selection_decision_audit_record(self) -> dict[str, object] | None:
        """Return model reasoning for trace audit, or ``None`` for legacy results."""

        if self.selection_decision is None:
            return None
        ranked = self.selection_decision.to_audit_record()
        if self.supplemental_selection_audit is None:
            return ranked
        return {
            "ranked_decision": ranked,
            "supplemental_selector_audit": (
                self.supplemental_selection_audit.to_record()
            ),
        }


@dataclass(frozen=True, slots=True)
class PortfolioMemoryCreditBatchPreparation:
    """Sealed, non-mutating preview of one generation memory publication.

    ``prepared_results`` contain the exact committed-receipt projection that
    downstream prepare hooks may inspect, while ``expected_trials`` remain
    absent from the live memory bank until :class:`PortfolioEvolution` commits
    this exact value.  Retaining the prior immutable trial snapshot turns a
    concurrent or out-of-band bank change into a stale-preparation failure.
    """

    generation: int
    source_result_receipt_sha256s: tuple[str, ...]
    source_pending_credit_sha256s: tuple[str | None, ...]
    prepared_results: tuple[PortfolioVariationWaveResult, ...]
    expected_trials: tuple[InsightTrial, ...]
    prior_memory_trials: tuple[InsightTrial, ...]
    batch_receipt: PortfolioMemoryCreditBatchReceipt | None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if (
            type(self.source_result_receipt_sha256s) is not tuple
            or not self.source_result_receipt_sha256s
        ):
            raise ValueError("source result receipts must be a non-empty exact tuple")
        for value in self.source_result_receipt_sha256s:
            require_sha256(value, "source_result_receipt_sha256")
        if (
            type(self.source_pending_credit_sha256s) is not tuple
            or len(self.source_pending_credit_sha256s)
            != len(self.source_result_receipt_sha256s)
        ):
            raise ValueError("pending credit identities differ from source results")
        for value in self.source_pending_credit_sha256s:
            if value is not None:
                require_sha256(value, "source_pending_credit_sha256")
        if (
            type(self.prepared_results) is not tuple
            or len(self.prepared_results) != len(self.source_result_receipt_sha256s)
            or any(
                type(value) is not PortfolioVariationWaveResult
                for value in self.prepared_results
            )
        ):
            raise ValueError("prepared_results must exactly cover source results")
        for result in self.prepared_results:
            PortfolioVariationWaveResult.__post_init__(result)
            if result.receipt.generation != self.generation:
                raise ValueError("prepared results differ from the generation")
            if result.pending_memory_credit is not None:
                raise ValueError("prepared results cannot retain pending memory credit")
        for name in ("expected_trials", "prior_memory_trials"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not InsightTrial for value in values
            ):
                raise TypeError(f"{name} must contain exact InsightTrial values")
            for value in values:
                InsightTrial.__post_init__(value)
        if self.batch_receipt is None:
            if self.expected_trials or any(self.source_pending_credit_sha256s):
                raise ValueError("a no-credit preparation cannot contain credit state")
            if any(
                result.receipt.memory_credit is not None
                for result in self.prepared_results
            ):
                raise ValueError("a no-credit preparation contains committed receipts")
        else:
            if type(self.batch_receipt) is not PortfolioMemoryCreditBatchReceipt:
                raise TypeError("batch_receipt must be exact or None")
            PortfolioMemoryCreditBatchReceipt.__post_init__(self.batch_receipt)
            if self.batch_receipt.generation != self.generation:
                raise ValueError("memory batch differs from the generation")
            if len(self.expected_trials) != len(self.batch_receipt.credits):
                raise ValueError("prospective trials differ from batch credits")
            if tuple(
                value.credit_unit_id for value in self.expected_trials
            ) != tuple(
                value.credit_unit_id for value in self.batch_receipt.credits
            ):
                raise ValueError("prospective trials differ from canonical credits")
            for trial, credit in zip(
                self.expected_trials,
                self.batch_receipt.credits,
                strict=True,
            ):
                if (
                    insight_selection_decision_sha256(trial.decision)
                    != credit.selection_decision_sha256
                    or trial.decision.context_hash
                    != credit.selection_decision_context_sha256
                    or trial.candidate_ids != credit.candidate_ids
                    or trial.reward_definition_hash
                    != credit.aggregation_definition_sha256
                    or trial.reward != credit.reward
                    or trial.treatment_binding_sha256
                    != credit.treatment_binding_sha256
                    or trial.generation != credit.generation
                ):
                    raise ValueError(
                        "prospective trial differs from its committed credit receipt"
                    )
            if self.batch_receipt.memory_trial_count_before != len(
                self.prior_memory_trials
            ):
                raise ValueError("batch receipt differs from prior memory snapshot")

    @staticmethod
    def _trial_record(value: InsightTrial) -> dict[str, object]:
        return {
            "credit_unit_id": value.credit_unit_id.value,
            "candidate_ids": [item.value for item in value.candidate_ids],
            "reward_definition_sha256": value.reward_definition_hash,
            "selection_decision_sha256": insight_selection_decision_sha256(
                value.decision
            ),
            "reward_hex": value.reward.hex(),
            "treatment_binding_sha256": value.treatment_binding_sha256,
            "generation": value.generation,
        }

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "generation": self.generation,
            "source_result_receipt_sha256s": list(
                self.source_result_receipt_sha256s
            ),
            "source_pending_credit_sha256s": list(
                self.source_pending_credit_sha256s
            ),
            "prepared_result_receipt_sha256s": [
                value.receipt.receipt_sha256 for value in self.prepared_results
            ],
            "prior_memory_trials": [
                self._trial_record(value) for value in self.prior_memory_trials
            ],
            "expected_trials": [
                self._trial_record(value) for value in self.expected_trials
            ],
            "memory_credit_batch_receipt_sha256": (
                None
                if self.batch_receipt is None
                else self.batch_receipt.receipt_sha256
            ),
        }

    @property
    def preparation_sha256(self) -> str:
        return _hash_record(
            _MEMORY_CREDIT_BATCH_PREPARATION_DOMAIN,
            self._unsigned_record(),
        )

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "preparation_sha256": self.preparation_sha256}


@runtime_checkable
class MaterializedPortfolioEngine(Protocol):
    async def run_materialized_invocations(
        self,
        items: tuple[MaterializedInvocation, ...],
        *,
        reward_binding: RewardPolicyBinding | None = None,
    ) -> tuple[InvocationOutcome, ...]: ...


@dataclass(slots=True)
class PortfolioEvolution:
    """Execute one ranked finite portfolio as a concurrent exact wave."""

    engine: MaterializedPortfolioEngine
    selector: PortfolioSelectionPolicy
    ids: IdFactory
    memory: InsightMemoryBank | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.engine, MaterializedPortfolioEngine):
            raise TypeError("engine must implement run_materialized_invocations")
        if not isinstance(self.selector, PortfolioSelectionPolicy):
            raise TypeError("selector must implement PortfolioSelectionPolicy")
        if not isinstance(self.ids, IdFactory):
            raise TypeError("ids must implement IdFactory")
        if self.memory is not None and type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank or None")

    def _materialize_member(
        self,
        wave: PortfolioVariationWaveRequest,
        decision: RankedPortfolioDecision,
        telemetry_sha256: str,
        member: RankedPortfolioMember,
    ) -> tuple[MaterializedInvocation, PortfolioMemberMaterializationReceipt]:
        contract = wave.selection_request.finite_variation_contract
        option = contract.resolve(member.option_id)
        if (
            member.option_identity_sha256 != option.identity_sha256
            or member.child_configuration_sha256 != option.child_configuration_sha256
        ):
            raise ValueError("ranked member drifted from its sealed finite option")
        candidate_id = self.ids.new_candidate_id()
        patch = derive_patch(
            wave.parent.configuration,
            option.child_configuration,
            base_candidate_id=wave.parent.candidate_id,
            target_candidate_id=candidate_id,
        )
        if not patch.operations:
            raise ValueError(
                "ranked finite option materialized no parent-relative patch"
            )
        top_level: set[str] = set()
        changed_paths: set[str] = set()
        for operation in patch.operations:
            if (
                not operation.path.segments
                or type(operation.path.segments[0]) is not ObjectKey
            ):
                raise ValueError("ranked finite option changed the candidate root")
            top_level.add(operation.path.segments[0].value)
            changed_paths.add(_path_text(operation.path))
        paths = tuple(sorted(changed_paths))
        receipt = PortfolioMemberMaterializationReceipt(
            request_sha256=wave.selection_request.request_sha256,
            decision_sha256=decision.decision_sha256,
            selection_telemetry_sha256=telemetry_sha256,
            rank=member.rank,
            option_id=member.option_id,
            option_identity_sha256=member.option_identity_sha256,
            child_configuration_sha256=member.child_configuration_sha256,
            parent_candidate_id=wave.parent.candidate_id,
            parent_configuration_sha256=(wave.parent.occurrence.configuration_hash),
            generation=wave.generation,
            candidate_id=candidate_id,
            patch_sha256=patch.patch_hash,
            changed_paths=paths,
        )
        configuration = thaw_json(option.child_configuration)
        if type(configuration) is not dict:
            raise TypeError("finite option child must be an object")
        plan = InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(wave.parent,),
            generation=wave.generation,
            label=f"{wave.label_prefix}.rank_{member.rank:04d}",
            allowed_top_level=tuple(sorted(top_level)),
            phase=wave.phase,
        )
        invocation = MaterializedInvocation(
            plan=plan,
            draft=CandidateDraft(
                configuration=configuration,
                design_rationale=(
                    "Engine materialized one exact sealed finite option selected "
                    "by ranked opaque identifier."
                ),
                intended_changes=paths,
                source_attribution=tuple(
                    SourceAttribution(path, "mutation") for path in paths
                ),
            ),
            candidate_id=candidate_id,
            materialization_policy_id=PORTFOLIO_MATERIALIZATION_POLICY_ID,
            materialization_policy_version=(PORTFOLIO_MATERIALIZATION_POLICY_VERSION),
            materialization_receipt_hash=receipt.receipt_sha256,
        )
        return invocation, receipt

    @staticmethod
    def _join_outcome(
        invocation: MaterializedInvocation,
        materialization: PortfolioMemberMaterializationReceipt,
        outcome: InvocationOutcome,
    ) -> PortfolioVariationMemberReceipt:
        if type(outcome) is not InvocationOutcome:
            raise TypeError("engine outcomes must be exact InvocationOutcome values")
        InvocationOutcome.__post_init__(outcome)
        prepared = outcome.prepared
        candidate = outcome.candidate
        if (
            prepared.plan != invocation.plan
            or prepared.proposal_authority is not ProposalAuthority.ENGINE
            or prepared.call_id is not None
            or prepared.candidate_id != materialization.candidate_id
            or prepared.materialization_policy_id != PORTFOLIO_MATERIALIZATION_POLICY_ID
            or prepared.materialization_policy_version
            != PORTFOLIO_MATERIALIZATION_POLICY_VERSION
            or prepared.materialization_receipt_hash != materialization.receipt_sha256
        ):
            raise ValueError("engine outcome differs from its materialized member")
        if (
            outcome.failure_stage is not None
            or candidate is None
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
        ):
            raise ValueError("portfolio wave contains a failed or partial member")
        if (
            candidate.candidate_id != materialization.candidate_id
            or candidate.occurrence.configuration_hash
            != materialization.child_configuration_sha256
            or not typed_json_equal(
                candidate.configuration,
                freeze_json(invocation.draft.configuration),
            )
            or candidate.parent_ids != (materialization.parent_candidate_id,)
            or candidate.parent_patch_hashes != (materialization.patch_sha256,)
            or candidate.call_telemetry is not None
            or tuple((item.path, item.source) for item in candidate.source_attribution)
            != tuple((path, "mutation") for path in materialization.changed_paths)
        ):
            raise ValueError("portfolio candidate differs from exact materialization")
        detailed = candidate.detailed_evaluation
        disposition = PortfolioMemberDisposition.SCORED
        candidate_failure: PortfolioCandidateFailureEvidence | None = None
        if not candidate.valid:
            if detailed is None or detailed.failure is None:
                raise ValueError(
                    "candidate infeasibility requires detailed evaluator evidence"
                )
            candidate_failure = PortfolioCandidateFailureEvidence.from_failure_record(
                detailed.failure,
                detailed_evaluation_sha256=detailed.evidence_sha256,
            )
            disposition = PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
            if outcome.parent_relations:
                raise ValueError(
                    "candidate infeasibility cannot publish parent relations"
                )
            if outcome.dominates_any_parent or outcome.better_than_any_parent:
                raise ValueError(
                    "candidate infeasibility cannot publish improvement flags"
                )
        return PortfolioVariationMemberReceipt(
            materialization=materialization,
            operator_invocation_id=prepared.operator_invocation_id,
            reward_definition_sha256=(prepared.variation_case.reward_definition_hash),
            reward=outcome.reward,
            parent_relations=outcome.parent_relations,
            detailed_evaluation_sha256=(
                None if detailed is None else detailed.evidence_sha256
            ),
            dominates_any_parent=outcome.dominates_any_parent,
            better_than_any_parent=outcome.better_than_any_parent,
            disposition=disposition,
            candidate_failure=candidate_failure,
        )

    @staticmethod
    def _join_action_attribution(
        request: PortfolioSelectionRequest,
        decision: RankedPortfolioDecision,
        selected: RankedPortfolioMember,
        member: PortfolioVariationMemberReceipt,
    ) -> PortfolioActionAttributionReceipt:
        """Bind one selected member to its exact cards and evaluated outcome."""

        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        PortfolioSelectionRequest.__post_init__(request)
        if type(decision) is not RankedPortfolioDecision:
            raise TypeError("decision must be an exact RankedPortfolioDecision")
        RankedPortfolioDecision.__post_init__(decision)
        if type(selected) is not RankedPortfolioMember:
            raise TypeError("selected must be an exact RankedPortfolioMember")
        RankedPortfolioMember.__post_init__(selected)
        if type(member) is not PortfolioVariationMemberReceipt:
            raise TypeError("member must be an exact portfolio member receipt")
        PortfolioVariationMemberReceipt.__post_init__(member)
        materialization = member.materialization
        if (
            decision.request_sha256 != request.request_sha256
            or decision.card_snapshot_sha256 != request.card_snapshot_sha256
            or selected.rank != materialization.rank
            or selected.option_id != materialization.option_id
            or selected.option_identity_sha256 != materialization.option_identity_sha256
            or selected.child_configuration_sha256
            != materialization.child_configuration_sha256
        ):
            raise ValueError(
                "selected member differs from its request or materialization"
            )
        card_by_key = {card.card_key: card for card in request.cards}
        supporting_cards = tuple(
            PortfolioActionCardAttribution(
                card_key=card_key,
                reference=card_by_key[card_key].reference,
                content_sha256=card_by_key[card_key].content_sha256,
                evidence_sha256=card_by_key[card_key].evidence_sha256,
            )
            for card_key in selected.supporting_card_keys
        )
        return PortfolioActionAttributionReceipt(
            request_sha256=request.request_sha256,
            decision_sha256=decision.decision_sha256,
            card_snapshot_sha256=request.card_snapshot_sha256,
            rank=selected.rank,
            option_id=selected.option_id,
            option_identity_sha256=selected.option_identity_sha256,
            child_configuration_sha256=selected.child_configuration_sha256,
            family=selected.family,
            supporting_cards=supporting_cards,
            effect_predictions=selected.effect_predictions,
            design_rationale_sha256=hashlib.sha256(
                selected.design_rationale.encode("utf-8", errors="strict")
            ).hexdigest(),
            materialization_receipt_sha256=materialization.receipt_sha256,
            outcome_sha256=member.outcome_sha256,
            operator_invocation_id=member.operator_invocation_id,
            candidate_id=materialization.candidate_id,
        )

    def prepare_pending_memory_credit_batch(
        self,
        results: tuple[PortfolioVariationWaveResult, ...],
    ) -> PortfolioMemoryCreditBatchPreparation:
        """Preview a complete concurrent stage without mutating memory.

        Result order remains the caller's stable decision-slot order.  Only the
        prospective memory mutation and batch receipt are canonicalized by
        preassigned credit-unit ID, so task completion order cannot affect the
        later commit.  All estimator and ownership validation runs here.
        """

        if type(results) is not tuple:
            raise TypeError("results must be an exact tuple")
        if any(type(result) is not PortfolioVariationWaveResult for result in results):
            raise TypeError("results must contain exact portfolio wave results")
        for result in results:
            PortfolioVariationWaveResult.__post_init__(result)
        pending_by_result = tuple(
            result.pending_memory_credit for result in results
        )
        pending = tuple(value for value in pending_by_result if value is not None)
        generations = {result.receipt.generation for result in results}
        if len(generations) != 1:
            raise ValueError("one memory-credit batch cannot span generations")
        generation = next(iter(generations))
        source_receipts = tuple(result.receipt.receipt_sha256 for result in results)
        source_pending = tuple(
            None if value is None else value.pending_sha256
            for value in pending_by_result
        )
        if not pending:
            return PortfolioMemoryCreditBatchPreparation(
                generation=generation,
                source_result_receipt_sha256s=source_receipts,
                source_pending_credit_sha256s=source_pending,
                prepared_results=results,
                expected_trials=(),
                prior_memory_trials=(
                    () if self.memory is None else self.memory.trials
                ),
                batch_receipt=None,
            )
        if self.memory is None:
            raise ValueError("pending portfolio memory credit requires a memory bank")
        if any(result.receipt.memory_credit is not None for result in results):
            raise ValueError("cannot mix committed and pending stage memory credits")

        canonical_pending = tuple(
            sorted(pending, key=lambda value: value.credit_unit_id.value)
        )
        canonical_ids = tuple(
            value.credit_unit_id.value for value in canonical_pending
        )
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("pending stage repeats a memory credit unit")
        committed_receipts = tuple(
            value.to_committed_receipt() for value in canonical_pending
        )
        receipt_by_unit = {
            value.credit_unit_id: value for value in committed_receipts
        }
        committed_results = tuple(
            result
            if result.pending_memory_credit is None
            else replace(
                result,
                receipt=replace(
                    result.receipt,
                    memory_credit=receipt_by_unit[
                        result.pending_memory_credit.credit_unit_id
                    ],
                ),
                pending_memory_credit=None,
            )
            for result in results
        )
        prior_memory_trials = self.memory.trials
        trial_count_before = len(prior_memory_trials)
        batch_receipt = PortfolioMemoryCreditBatchReceipt(
            generation=generation,
            credits=committed_receipts,
            memory_trial_count_before=trial_count_before,
            memory_trial_count_after=trial_count_before + len(committed_receipts),
        )
        expected_trials = tuple(value.to_trial() for value in canonical_pending)
        # Exercise the complete bank estimator validation now.  The bank method
        # is deliberately non-mutating; commit rechecks the retained exact prior
        # trial tuple before applying the same canonical batch.
        previewed_trials = self.memory.preview_trials_batch(expected_trials)
        if previewed_trials != expected_trials:
            raise RuntimeError("memory bank changed canonical preview order")
        return PortfolioMemoryCreditBatchPreparation(
            generation=generation,
            source_result_receipt_sha256s=source_receipts,
            source_pending_credit_sha256s=source_pending,
            prepared_results=committed_results,
            expected_trials=expected_trials,
            prior_memory_trials=prior_memory_trials,
            batch_receipt=batch_receipt,
        )

    def commit_prepared_memory_credit_batch(
        self,
        preparation: PortfolioMemoryCreditBatchPreparation,
    ) -> tuple[
        tuple[PortfolioVariationWaveResult, ...],
        PortfolioMemoryCreditBatchReceipt | None,
    ]:
        """Commit one exact prevalidated generation publication."""

        if type(preparation) is not PortfolioMemoryCreditBatchPreparation:
            raise TypeError("preparation must be exact")
        PortfolioMemoryCreditBatchPreparation.__post_init__(preparation)
        batch_receipt = preparation.batch_receipt
        if batch_receipt is None:
            return preparation.prepared_results, None
        if self.memory is None:
            raise ValueError("prepared memory credit requires a memory bank")
        if self.memory.trials != preparation.prior_memory_trials:
            raise RuntimeError("memory bank changed after credit preparation")
        committed_trials = self.memory.record_trials_batch(
            preparation.expected_trials
        )
        if committed_trials != preparation.expected_trials:
            raise RuntimeError("memory bank changed canonical pending-credit order")
        if len(self.memory.trials) != batch_receipt.memory_trial_count_after:
            raise RuntimeError("memory bank trial count differs from batch receipt")
        return preparation.prepared_results, batch_receipt

    def commit_pending_memory_credit_batch(
        self,
        results: tuple[PortfolioVariationWaveResult, ...],
    ) -> tuple[
        tuple[PortfolioVariationWaveResult, ...],
        PortfolioMemoryCreditBatchReceipt | None,
    ]:
        """Backward-compatible prepare-and-commit convenience boundary."""

        preparation = self.prepare_pending_memory_credit_batch(results)
        committed_results, batch_receipt = (
            self.commit_prepared_memory_credit_batch(preparation)
        )
        return committed_results, batch_receipt

    async def run(
        self,
        wave: PortfolioVariationWaveRequest,
        *,
        reward_binding: RewardPolicyBinding | None = None,
        defer_memory_credit: bool = False,
    ) -> PortfolioVariationWaveResult:
        """Select once, materialize all members, and evaluate one concurrent wave."""

        if type(wave) is not PortfolioVariationWaveRequest:
            raise TypeError("wave must be an exact PortfolioVariationWaveRequest")
        PortfolioVariationWaveRequest.__post_init__(wave)
        if reward_binding is not None:
            if type(reward_binding) is not RewardPolicyBinding:
                raise TypeError("reward_binding must be exact or None")
            RewardPolicyBinding.__post_init__(reward_binding)
        if type(defer_memory_credit) is not bool:
            raise TypeError("defer_memory_credit must be an exact boolean")
        credit = wave.memory_credit
        if credit is not None:
            if self.memory is None:
                raise ValueError("portfolio memory credit requires a memory bank")
            eligible_entries = self.memory.entries_for(credit.decision.eligible)
            entry_by_reference = {
                entry.reference: entry for entry in eligible_entries
            }
            for card in wave.selection_request.cards:
                entry = entry_by_reference[card.reference]
                if card.content_sha256 != entry.draft.content_sha256:
                    raise ValueError(
                        "memory credit card content differs from its memory entry"
                    )
                if (
                    entry.origin is InsightOrigin.REFLECTION
                    and (
                        card.source_binding is None
                        or wave.selection_request.source_registry is None
                    )
                ):
                    raise ValueError(
                        "reflection memory credit requires source-admitted cards"
                    )
            quarantine_entries = tuple(
                entry
                for entry in eligible_entries
                if entry.lifecycle_state is InsightLifecycleState.QUARANTINED
            )
            if quarantine_entries:
                admission = credit.quarantine_admission
                if admission is None:
                    raise ValueError(
                        "quarantine memory credit requires bank-issued admission"
                    )
                self.memory.validate_quarantine_test_admission(
                    admission,
                    eligible_references=credit.decision.eligible,
                    subset_authorization_sha256=(
                        credit.quarantine_admission_subset_authorization_sha256
                    ),
                )
            elif credit.quarantine_admission is not None:
                raise ValueError(
                    "normal memory credit cannot carry quarantine admission"
                )
            if any(
                entry.lifecycle_state is InsightLifecycleState.DEPRECATED
                for entry in eligible_entries
            ):
                raise ValueError("memory credit eligible set contains deprecated insight")

        request_sha256 = wave.selection_request.request_sha256
        contract_sha256 = (
            wave.selection_request.finite_variation_contract.identity_sha256
        )
        parent_sha256 = wave.parent.occurrence.configuration_hash
        result = await self.selector.select(wave.selection_request)
        if type(result) is not PortfolioSelectionResult:
            raise TypeError("selector must return an exact PortfolioSelectionResult")
        PortfolioSelectionResult.__post_init__(result)
        if result.telemetry is None:
            raise ValueError("portfolio selection requires exact call telemetry")
        telemetry = result.telemetry
        telemetry_sha256 = portfolio_selection_telemetry_sha256(telemetry)
        validate_ranked_portfolio_decision(wave.selection_request, result.decision)
        if (
            wave.selection_request.request_sha256 != request_sha256
            or wave.selection_request.finite_variation_contract.identity_sha256
            != contract_sha256
            or wave.parent.occurrence.configuration_hash != parent_sha256
            or not typed_json_equal(
                wave.parent.configuration,
                wave.selection_request.finite_variation_contract.parent_configuration,
            )
        ):
            raise ValueError(
                "portfolio parent or request contract drifted during selection"
            )

        materialized_pairs = tuple(
            self._materialize_member(
                wave,
                result.decision,
                telemetry_sha256,
                member,
            )
            for member in result.decision.members
        )
        invocations = tuple(value[0] for value in materialized_pairs)
        materializations = tuple(value[1] for value in materialized_pairs)
        candidate_ids = tuple(value.candidate_id for value in invocations)
        child_sha256s = tuple(
            value.child_configuration_sha256 for value in materializations
        )
        if len(set(candidate_ids)) != len(candidate_ids) or len(
            set(child_sha256s)
        ) != len(child_sha256s):
            raise ValueError("portfolio materialization contains colliding members")

        outcomes = await self.engine.run_materialized_invocations(
            invocations,
            reward_binding=reward_binding,
        )
        if type(outcomes) is not tuple or len(outcomes) != len(invocations):
            raise ValueError("engine returned a partial portfolio outcome wave")
        members = tuple(
            self._join_outcome(invocation, materialization, outcome)
            for invocation, materialization, outcome in zip(
                invocations,
                materializations,
                outcomes,
                strict=True,
            )
        )
        if len({member.operator_invocation_id for member in members}) != len(members):
            raise ValueError("portfolio outcomes contain colliding invocations")
        action_attributions = tuple(
            self._join_action_attribution(
                wave.selection_request,
                result.decision,
                selected,
                member,
            )
            for selected, member in zip(
                result.decision.members,
                members,
                strict=True,
            )
        )

        credit_receipt: PortfolioMemoryCreditReceipt | None = None
        pending_credit: PortfolioPendingMemoryCredit | None = None
        if credit is not None:
            assert self.memory is not None
            context_projection = credit.resolve_context_projection(
                wave.selection_request.context
            )
            if credit.credit_unit_id in {
                member.operator_invocation_id for member in members
            }:
                raise ValueError("memory credit unit collides with a child invocation")
            aggregate_reward = credit.aggregation.aggregate(outcomes)
            if type(aggregate_reward) is not float or not math.isfinite(
                aggregate_reward
            ):
                raise TypeError("portfolio aggregate reward must be a finite float")
            pending_credit = PortfolioPendingMemoryCredit(
                credit_unit_id=credit.credit_unit_id,
                decision=credit.decision,
                candidate_ids=candidate_ids,
                aggregation=credit.aggregation,
                context_projection=context_projection,
                reward=aggregate_reward,
                treatment_binding_sha256=credit.treatment_binding_sha256,
                generation=wave.generation,
            )
            if not defer_memory_credit:
                before = len(self.memory.trials)
                trial = self.memory.record_trials_batch(
                    (pending_credit.to_trial(),)
                )[0]
                if (
                    len(self.memory.trials) != before + 1
                    or self.memory.trials[-1] is not trial
                    or trial.candidate_ids != candidate_ids
                    or trial.reward != aggregate_reward
                ):
                    raise RuntimeError(
                        "portfolio memory credit was not one exact trial"
                    )
                credit_receipt = pending_credit.to_committed_receipt()
                pending_credit = None

        receipt = PortfolioVariationWaveReceipt(
            selection_call_id=wave.selection_request.call_id,
            request_sha256=request_sha256,
            decision_sha256=result.decision.decision_sha256,
            selection_policy_id=result.decision.policy_id,
            selection_policy_version=result.decision.policy_version,
            selection_policy_definition_sha256=(
                result.decision.policy_definition_sha256
            ),
            selection_telemetry=telemetry,
            selection_telemetry_sha256=telemetry_sha256,
            parent_candidate_id=wave.parent.candidate_id,
            parent_configuration_sha256=parent_sha256,
            generation=wave.generation,
            members=members,
            action_attributions=action_attributions,
            memory_credit=credit_receipt,
        )
        PortfolioVariationWaveReceipt.__post_init__(receipt)
        return PortfolioVariationWaveResult(
            receipt=receipt,
            outcomes=outcomes,
            selection_decision=result.decision,
            supplemental_selection_audit=result.supplemental_audit,
            pending_memory_credit=pending_credit,
        )


__all__ = [
    "EXACT_MEMORY_CONTEXT_PROJECTION_DEFINITION_SHA256",
    "MEMORY_ESTIMAND_CONTEXT_KEY",
    "MEMORY_ESTIMAND_SUBTREE_PROJECTION_DEFINITION_SHA256",
    "PORTFOLIO_MATERIALIZATION_POLICY_ID",
    "PORTFOLIO_MATERIALIZATION_POLICY_VERSION",
    "MaterializedPortfolioEngine",
    "PortfolioActionAttributionReceipt",
    "PortfolioActionCardAttribution",
    "PortfolioCandidateFailureEvidence",
    "PortfolioEvolution",
    "PortfolioMemberMaterializationReceipt",
    "PortfolioMemberDisposition",
    "PortfolioMemoryCreditBatchReceipt",
    "PortfolioMemoryCreditBatchPreparation",
    "PortfolioMemoryCreditPlan",
    "PortfolioMemoryCreditReceipt",
    "PortfolioMemoryContextProjectionBinding",
    "PortfolioMemoryMatchedControlWavePlan",
    "PortfolioPendingMemoryCredit",
    "PortfolioRewardAggregationBinding",
    "PortfolioVariationMemberReceipt",
    "PortfolioVariationWaveReceipt",
    "PortfolioVariationWaveResult",
    "PortfolioVariationWaveRequest",
    "portfolio_selection_telemetry_sha256",
]

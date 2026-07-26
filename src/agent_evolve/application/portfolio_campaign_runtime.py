"""Concrete execution bridge for prepared portfolio-evolution campaigns.

The preparation API deliberately knows nothing about mutable optimizer state.
This adapter supplies that missing runtime: it evaluates the prepared seeds,
maintains one benchmark-bound Pareto archive, binds each parent through the
prepared :class:`CampaignWorkloadPorts`, executes real
:class:`PortfolioEvolution` and :class:`PortfolioRecombination` waves, and
projects their evidence into :class:`EvolutionCampaignScheduler` receipts.

Injected ports keep each experimental choice replaceable: parent selection,
portfolio-wave construction, parent-local context retrieval, post-generation
outcome updates, reflection execution, closed-loop learning, recombination
utility, prompt rendering, and owned-resource cleanup.  The runtime owns only
ordering, validation, and evidence publication across those boundaries.

No workload, objective name, model, or provider is named in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from agent_evolve.agentic import PhenotypeIdentity, PortfolioEvolutionComposition
from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.budgeted_optimizer import (
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.campaign_execution import (
    CampaignArchiveCutoffReceipt,
    CampaignArchiveCutoffRequest,
    CampaignCleanupReceipt,
    CampaignCleanupRequest,
    CampaignExecutionStatus,
    CampaignExecutionStartReceipt,
    CampaignFinalizationReceipt,
    CampaignFinalizationRequest,
    CampaignReflectionReceipt,
    CampaignReflectionRequest,
    CampaignReflectionStatus,
    CampaignReflectionTestAdmissionReceipt,
    CampaignReflectionTestAdmissionRequest,
    CampaignSeedExecutionReceipt,
    CampaignSelectorAuditReceipt,
    CampaignStageReceipt,
    CampaignStageRequest,
    SelectorAuditExecutionMode,
)
from agent_evolve.application.concurrent_stage import gather_concurrent_stage
from agent_evolve.application.contextual_campaign_planning import (
    CampaignContextualSearchPlanner,
)
from agent_evolve.application.contextual_delayed_credit import (
    observe_contextual_post_recombination_credit,
    observe_contextual_terminal_persistence,
)
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilitySnapshot,
    CampaignGenerationKind,
    CampaignReflectionWave,
    CampaignWorkloadPorts,
    ParentVariationBinding,
    PreparedEvolutionCampaign,
    ReflectionVisibility,
)
from agent_evolve.application.campaign_evidence_registry import (
    CampaignEvidenceRegistry,
)
from agent_evolve.application.campaign_selector_context_extension import (
    CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY,
    resolve_campaign_selector_context_extension,
)
from agent_evolve.application.campaign_search_phase import (
    attach_campaign_search_phase_context,
    campaign_search_phase_context,
)
from agent_evolve.application.identifiable_reflection_evidence import (
    IdentifiableReflectionEvidenceSnapshot,
    ReflectionFalsificationFeedback,
    project_identifiable_reflection_evidence,
)
from agent_evolve.application.pareto_archive import ParetoArchive, ParetoDecision
from agent_evolve.application.parent_measurement import (
    attach_parent_measurement_to_context,
    bind_parent_measurement,
)
from agent_evolve.application.portfolio_evolution import (
    MEMORY_ESTIMAND_CONTEXT_KEY,
    PortfolioMemoryCreditBatchPreparation,
    PortfolioMemoryContextProjectionBinding,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
    PortfolioMemberDisposition,
)
from agent_evolve.application.portfolio_recombination import (
    PortfolioRecombination,
    PortfolioRecombinationWaveRequest,
    PortfolioRecombinationWaveResult,
)
from agent_evolve.campaign_workload import AgenticCampaignWorkloadConfig
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    GlobalEvidenceRegistrySnapshot,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    render_portfolio_selection_prompt,
)
from agent_evolve.policies.selection.archive_elite import (
    TaskKeyedArchiveEliteParentPolicy,
    TaskKeyedArchiveReservoirParentPolicy,
)
from agent_evolve.policies.selection.elite_explorer import (
    TaskKeyedArchiveEliteExplorerParentPolicy,
)
from agent_evolve.policies.selection.frozen_archive_pairs import (
    FrozenArchiveSourceUtilityReceipt,
)
from agent_evolve.policies.selection.residual_frontier import (
    RESIDUAL_FRONTIER_POLICY_DEFINITION_SHA256,
    residual_anchor_parents,
    residual_frontier_geometry,
)
from agent_evolve.ports.portfolio_selection import PortfolioSelectionRequest
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationContract,
)
from agent_evolve.ports.frontier_target import (
    CampaignPortfolioFrontierTarget,
    CampaignPortfolioFrontierTargetAllocator,
)
from agent_evolve.ports.objective_resolution import (
    EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
    EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
    EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
    objective_resolution_policy_metadata,
)
from agent_evolve.ports.parent_measurement import (
    ParentMeasurementBinding,
    ParentMeasurementProjection,
)
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjection,
    CampaignPortfolioArchiveContextProjector,
)


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed input.
        raise AssertionError("runtime record did not freeze to an object")
    return frozen


def _canonical_text(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


_CAMPAIGN_ROLE_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
CAMPAIGN_CONTEXTUAL_HISTORY_KEY = "campaign_contextual_history"
CAMPAIGN_ARCHIVE_CONTEXT_KEY = "campaign_archive_context"
CAMPAIGN_FRONTIER_TARGET_KEY = "campaign_frontier_target"
CAMPAIGN_IDENTIFIABLE_REFLECTION_BINDING_KEY = (
    "campaign_identifiable_reflection_binding"
)
MEMORY_ESTIMAND_STRATUM_SHA256_KEY = "memory_estimand_stratum_sha256"
_LOWER_SHA256 = frozenset("0123456789abcdef")
_OUTCOME_PREPARATION_DOMAIN = (
    b"agent-evolve:campaign-portfolio-outcome-preparation:v1\x00"
)
_LEARNING_PREPARATION_DOMAIN = (
    b"agent-evolve:campaign-portfolio-learning-preparation:v1\x00"
)
_WAVE_PREPARATION_DOMAIN = b"agent-evolve:campaign-portfolio-wave-preparation:v1\x00"
_IDENTIFIABLE_REFLECTION_QUERY_DOMAIN = (
    b"agent-evolve:campaign-identifiable-reflection-query:v1\x00"
)
_IDENTIFIABLE_REFLECTION_SOURCE_DOMAIN = (
    b"agent-evolve:campaign-identifiable-reflection-source:v1\x00"
)
_IDENTIFIABLE_REFLECTION_INPUT_DOMAIN = (
    b"agent-evolve:campaign-identifiable-reflection-input:v1\x00"
)


def _require_role_token(value: str, name: str) -> None:
    if type(value) is not str or _CAMPAIGN_ROLE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _require_sha256(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _preparation_sha256(domain: bytes, record: FrozenJsonObject) -> str:
    return hashlib.sha256(
        domain + _canonical_text(thaw_json(record)).encode("ascii", errors="strict")
    ).hexdigest()


def _candidate_record(candidate: EvolutionCandidate) -> dict[str, object]:
    EvolutionCandidate.__post_init__(candidate)
    record: dict[str, object] = {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "generation": candidate.generation,
        "label": candidate.label,
        "valid": candidate.valid,
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "objectives": [
            {"metric_id": name, "value_hex": value.hex()}
            for name, value in candidate.objectives
        ],
        "operator_kind": (
            None if candidate.operator_kind is None else candidate.operator_kind.value
        ),
        "parent_ids": [value.value for value in candidate.parent_ids],
        "common_ancestor_id": (
            None
            if candidate.common_ancestor_id is None
            else candidate.common_ancestor_id.value
        ),
    }
    if candidate.objective_resolution_receipt is not None:
        record["objective_resolution"] = (
            candidate.objective_resolution_receipt.to_record()
        )
    return record


def _archive_decision_record(decision: ParetoDecision) -> dict[str, object]:
    """Project archive evidence without copying evaluator free text."""

    if type(decision) is not ParetoDecision:
        raise TypeError("decision must be an exact ParetoDecision")
    ParetoDecision.__post_init__(decision)
    record = decision.to_trace_record()
    record["failure_details"] = [
        {
            "reason": reason.value,
            "detail_sha256": hashlib.sha256(
                detail.encode("utf-8", errors="strict")
            ).hexdigest(),
        }
        for reason, detail in decision.failure_details
    ]
    record["failure_detail_projection"] = "content_minimized_sha256"
    return record


def _cache_misses(snapshot: dict[str, int | None]) -> int:
    value = snapshot.get("misses")
    if type(value) is not int or value < 0:
        raise RuntimeError("engine evaluation cache omitted exact miss accounting")
    return value


@dataclass(frozen=True, slots=True)
class CampaignParentLane:
    """One explicitly named parent role selected for a portfolio stage."""

    lane_id: str
    parent: EvolutionCandidate

    def __post_init__(self) -> None:
        _require_role_token(self.lane_id, "lane_id")
        if type(self.parent) is not EvolutionCandidate:
            raise TypeError("parent must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(self.parent)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "lane_id": self.lane_id,
            "parent_candidate_id": self.parent.candidate_id.value,
            "parent_configuration_sha256": (self.parent.occurrence.configuration_hash),
        }


@dataclass(frozen=True, slots=True)
class CampaignDecisionSlot:
    """One named campaign decision bound to exactly one parent lane.

    The current campaign protocol admits one primary slot per lane.  Making the
    identity explicit now prevents workload adapters from inferring scientific
    roles from tuple position and leaves a typed extension point for matched
    treatment/control slots later.
    """

    slot_id: str
    lane_id: str
    role_id: str = "primary"

    def __post_init__(self) -> None:
        _require_role_token(self.slot_id, "slot_id")
        _require_role_token(self.lane_id, "lane_id")
        _require_role_token(self.role_id, "role_id")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "slot_id": self.slot_id,
            "lane_id": self.lane_id,
            "role_id": self.role_id,
        }


@dataclass(frozen=True, slots=True)
class CampaignParentSelection:
    """Parents selected from one exact optimizer state plus policy evidence."""

    parents: tuple[EvolutionCandidate, ...]
    evidence: FrozenJsonObject
    lanes: tuple[CampaignParentLane, ...] = ()
    decision_slots: tuple[CampaignDecisionSlot, ...] = ()

    def __post_init__(self) -> None:
        if type(self.parents) is not tuple or not self.parents:
            raise ValueError("campaign parent selection requires parents")
        if any(type(value) is not EvolutionCandidate for value in self.parents):
            raise TypeError("parents must contain exact EvolutionCandidate values")
        for value in self.parents:
            EvolutionCandidate.__post_init__(value)
        if len({value.candidate_id for value in self.parents}) != len(self.parents):
            raise ValueError("campaign parents must be distinct occurrences")
        if type(self.evidence) is not FrozenJsonObject:
            raise TypeError("parent-selection evidence must be frozen typed JSON")
        if freeze_json(self.evidence) is not self.evidence:
            raise TypeError("parent-selection evidence must already be frozen")
        lanes = self.lanes
        if not lanes:
            lanes = tuple(
                CampaignParentLane(
                    lane_id=f"parent_{index + 1:04d}",
                    parent=parent,
                )
                for index, parent in enumerate(self.parents)
            )
            object.__setattr__(self, "lanes", lanes)
        if type(lanes) is not tuple or any(
            type(value) is not CampaignParentLane for value in lanes
        ):
            raise TypeError("lanes must contain exact CampaignParentLane values")
        for lane in lanes:
            CampaignParentLane.__post_init__(lane)
        if len(lanes) != len(self.parents) or any(
            lane.parent is not parent
            for lane, parent in zip(lanes, self.parents, strict=True)
        ):
            raise ValueError("parent lanes must align exactly with selected parents")
        lane_ids = tuple(lane.lane_id for lane in lanes)
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("parent lane IDs must be unique")

        slots = self.decision_slots
        if not slots:
            slots = tuple(
                CampaignDecisionSlot(
                    slot_id=f"{lane.lane_id}.primary",
                    lane_id=lane.lane_id,
                )
                for lane in lanes
            )
            object.__setattr__(self, "decision_slots", slots)
        if type(slots) is not tuple or any(
            type(value) is not CampaignDecisionSlot for value in slots
        ):
            raise TypeError(
                "decision_slots must contain exact CampaignDecisionSlot values"
            )
        for slot in slots:
            CampaignDecisionSlot.__post_init__(slot)
        slot_ids = tuple(slot.slot_id for slot in slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("campaign decision slot IDs must be unique")
        if tuple(slot.lane_id for slot in slots) != lane_ids:
            raise ValueError(
                "bounded campaign selection requires one ordered slot per lane"
            )


_PARENT_SELECTION_PROGRESS_DOMAIN = (
    b"agent-evolve:campaign-parent-selection-progress:v1\x00"
)


@dataclass(frozen=True, slots=True)
class CampaignParentSelectionProgress:
    """Authenticated workload-neutral archive transition from one stage."""

    generation: int
    stage_kind: CampaignGenerationKind
    stage_request_sha256: str
    stage_receipt_sha256: str
    pre_archive_sha256: str
    post_archive_sha256: str
    utility_id: str
    utility_version: int
    utility_definition_sha256: str
    pre_utility_snapshot_sha256: str
    post_utility_snapshot_sha256: str
    pre_scalar_utility_hex: str | None
    post_scalar_utility_hex: str | None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("progress generation must be a positive exact integer")
        if type(self.stage_kind) is not CampaignGenerationKind:
            raise TypeError("stage_kind must be an exact CampaignGenerationKind")
        for name in (
            "stage_request_sha256",
            "stage_receipt_sha256",
            "pre_archive_sha256",
            "post_archive_sha256",
            "utility_definition_sha256",
            "pre_utility_snapshot_sha256",
            "post_utility_snapshot_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_role_token(self.utility_id, "utility_id")
        if type(self.utility_version) is not int or self.utility_version <= 0:
            raise ValueError("utility_version must be a positive exact integer")
        scalar_values = (
            self.pre_scalar_utility_hex,
            self.post_scalar_utility_hex,
        )
        if (scalar_values[0] is None) is not (scalar_values[1] is None):
            raise ValueError("scalar utility progress must be complete or absent")
        if scalar_values[0] is not None:
            for name, value in zip(
                ("pre_scalar_utility_hex", "post_scalar_utility_hex"),
                scalar_values,
                strict=True,
            ):
                if type(value) is not str:
                    raise TypeError(f"{name} must be an exact string or None")
                try:
                    parsed = float.fromhex(value)
                except ValueError as error:
                    raise ValueError(f"{name} must be a float hex string") from error
                if not math.isfinite(parsed) or parsed < 0.0 or parsed.hex() != value:
                    raise ValueError(
                        f"{name} must encode a canonical non-negative utility"
                    )
            if float.fromhex(scalar_values[1]) < float.fromhex(scalar_values[0]):
                raise ValueError("authenticated archive utility cannot decrease")

    @property
    def archive_changed(self) -> bool:
        return self.pre_archive_sha256 != self.post_archive_sha256

    @property
    def utility_signal_available(self) -> bool:
        return self.pre_scalar_utility_hex is not None

    @property
    def utility_improved(self) -> bool | None:
        if not self.utility_signal_available:
            return None
        assert self.pre_scalar_utility_hex is not None
        assert self.post_scalar_utility_hex is not None
        return float.fromhex(self.post_scalar_utility_hex) > float.fromhex(
            self.pre_scalar_utility_hex
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "generation": self.generation,
            "stage_kind": self.stage_kind.value,
            "stage_request_sha256": self.stage_request_sha256,
            "stage_receipt_sha256": self.stage_receipt_sha256,
            "pre_archive_sha256": self.pre_archive_sha256,
            "post_archive_sha256": self.post_archive_sha256,
            "archive_changed": self.archive_changed,
            "utility_id": self.utility_id,
            "utility_version": self.utility_version,
            "utility_definition_sha256": self.utility_definition_sha256,
            "pre_utility_snapshot_sha256": self.pre_utility_snapshot_sha256,
            "post_utility_snapshot_sha256": self.post_utility_snapshot_sha256,
            "pre_scalar_utility_hex": self.pre_scalar_utility_hex,
            "post_scalar_utility_hex": self.post_scalar_utility_hex,
            "utility_signal_available": self.utility_signal_available,
            "utility_improved": self.utility_improved,
        }

    @property
    def progress_sha256(self) -> str:
        return hashlib.sha256(
            _PARENT_SELECTION_PROGRESS_DOMAIN
            + _canonical_text(self._unsigned_record()).encode("ascii")
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "progress_sha256": self.progress_sha256,
        }


@dataclass(frozen=True, slots=True)
class _PendingCampaignParentSelectionProgress:
    generation: int
    stage_kind: CampaignGenerationKind
    stage_request_sha256: str
    stage_receipt_sha256: str
    pre_archive_sha256: str
    post_archive_sha256: str
    pre_utility: ArchiveUtilitySnapshot

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("pending progress generation must be positive")
        if type(self.stage_kind) is not CampaignGenerationKind:
            raise TypeError("pending stage kind must be exact")
        for name in (
            "stage_request_sha256",
            "stage_receipt_sha256",
            "pre_archive_sha256",
            "post_archive_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.pre_utility) is not ArchiveUtilitySnapshot:
            raise TypeError("pre_utility must be an exact ArchiveUtilitySnapshot")
        ArchiveUtilitySnapshot.__post_init__(self.pre_utility)

    def complete(
        self,
        post_utility: ArchiveUtilitySnapshot,
    ) -> CampaignParentSelectionProgress:
        self.__post_init__()
        if type(post_utility) is not ArchiveUtilitySnapshot:
            raise TypeError("post_utility must be an exact ArchiveUtilitySnapshot")
        ArchiveUtilitySnapshot.__post_init__(post_utility)
        if (
            post_utility.generation != self.generation + 1
            or post_utility.utility_id != self.pre_utility.utility_id
            or post_utility.utility_version != self.pre_utility.utility_version
            or post_utility.definition_sha256 != self.pre_utility.definition_sha256
            or post_utility.archive_sha256 != self.post_archive_sha256
        ):
            raise ValueError(
                "post-stage utility snapshot does not close pending progress"
            )
        return CampaignParentSelectionProgress(
            generation=self.generation,
            stage_kind=self.stage_kind,
            stage_request_sha256=self.stage_request_sha256,
            stage_receipt_sha256=self.stage_receipt_sha256,
            pre_archive_sha256=self.pre_archive_sha256,
            post_archive_sha256=self.post_archive_sha256,
            utility_id=self.pre_utility.utility_id,
            utility_version=self.pre_utility.utility_version,
            utility_definition_sha256=self.pre_utility.definition_sha256,
            pre_utility_snapshot_sha256=self.pre_utility.snapshot_sha256,
            post_utility_snapshot_sha256=post_utility.snapshot_sha256,
            pre_scalar_utility_hex=self.pre_utility.scalar_utility_hex,
            post_scalar_utility_hex=post_utility.scalar_utility_hex,
        )


def _validate_parent_selection_progress(
    progress: tuple[CampaignParentSelectionProgress, ...],
    *,
    optimizer_generation: int,
) -> None:
    if type(progress) is not tuple or any(
        type(value) is not CampaignParentSelectionProgress for value in progress
    ):
        raise TypeError(
            "progress must contain exact CampaignParentSelectionProgress values"
        )
    for value in progress:
        CampaignParentSelectionProgress.__post_init__(value)
    if tuple(value.generation for value in progress) != tuple(
        range(1, len(progress) + 1)
    ):
        raise ValueError("parent-selection progress must be contiguous and ordered")
    if len(progress) != optimizer_generation:
        raise ValueError(
            "parent-selection progress must cover the exact optimizer generation"
        )


@runtime_checkable
class CampaignParentSelectionPort(Protocol):
    """Experiment-owned, workload-neutral parent-selection seam."""

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        parent_count: int,
        rotation_index: int,
        progress: tuple[CampaignParentSelectionProgress, ...] = (),
        archive_utility: ArchiveUtilitySnapshot | None = None,
    ) -> CampaignParentSelection: ...


@dataclass(frozen=True, slots=True)
class ArchiveReservoirCampaignParentSelector:
    """Use the existing ranked-reservoir policy behind the campaign seam."""

    reservoir_limit: int = 8
    policy: TaskKeyedArchiveReservoirParentPolicy = field(
        default_factory=TaskKeyedArchiveReservoirParentPolicy
    )

    def __post_init__(self) -> None:
        if type(self.reservoir_limit) is not int or self.reservoir_limit <= 0:
            raise ValueError("reservoir_limit must be positive")
        if type(self.policy) is not TaskKeyedArchiveReservoirParentPolicy:
            raise TypeError("policy must be an exact reservoir parent policy")

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        parent_count: int,
        rotation_index: int,
        progress: tuple[CampaignParentSelectionProgress, ...] = (),
        archive_utility: ArchiveUtilitySnapshot | None = None,
    ) -> CampaignParentSelection:
        del archive_utility
        _validate_parent_selection_progress(
            progress,
            optimizer_generation=state.generation,
        )
        selected = self.policy.select(
            state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            reservoir_limit=self.reservoir_limit,
            parent_count=parent_count,
            rotation_index=rotation_index,
        )
        lanes = tuple(
            CampaignParentLane(
                lane_id=f"reservoir_{index + 1:04d}",
                parent=parent,
            )
            for index, parent in enumerate(selected.parents)
        )
        return CampaignParentSelection(
            parents=selected.parents,
            evidence=_object(selected.receipt.to_trace_record()),
            lanes=lanes,
        )


@dataclass(frozen=True, slots=True)
class ArchiveEliteCampaignParentSelector:
    """Use the outcome-relation-neutral archive-front policy in a campaign."""

    policy: TaskKeyedArchiveEliteParentPolicy = field(
        default_factory=TaskKeyedArchiveEliteParentPolicy
    )

    def __post_init__(self) -> None:
        if type(self.policy) is not TaskKeyedArchiveEliteParentPolicy:
            raise TypeError("policy must be an exact elite parent policy")

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        parent_count: int,
        rotation_index: int,
        progress: tuple[CampaignParentSelectionProgress, ...] = (),
        archive_utility: ArchiveUtilitySnapshot | None = None,
    ) -> CampaignParentSelection:
        del archive_utility
        _validate_parent_selection_progress(
            progress,
            optimizer_generation=state.generation,
        )
        selected = self.policy.select(
            state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            parent_count=parent_count,
            rotation_index=rotation_index,
        )
        lanes = tuple(
            CampaignParentLane(
                lane_id=f"elite_{index + 1:04d}",
                parent=parent,
            )
            for index, parent in enumerate(selected.parents)
        )
        return CampaignParentSelection(
            parents=selected.parents,
            evidence=_object(selected.receipt.to_trace_record()),
            lanes=lanes,
        )


_DIVERSE_ELITE_PARENT_POLICY_ID = "archive_diverse_elite_campaign_parent"
_DIVERSE_ELITE_PARENT_POLICY_VERSION = 1
_DIVERSE_ELITE_PARENT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:archive-diverse-elite-campaign-parent:v1:"
    b"elite=task-keyed-current-front-member;"
    b"explorer=max-canonical-typed-patch-operation-distance-on-current-front;"
    b"ties=task-and-snapshot-keyed-sha256;"
    b"singleton=structurally-most-distant-eligible-history-fallback;"
    b"workload-semantics=false;dominated-parent=false"
).hexdigest()
_DIVERSE_ELITE_PARENT_ROTATION_DOMAIN = (
    b"agent-evolve:archive-diverse-elite-campaign-parent:rotation:v1\x00"
)
_DIVERSE_ELITE_PARENT_TIE_DOMAIN = (
    b"agent-evolve:archive-diverse-elite-campaign-parent:tie:v1\x00"
)


@dataclass(frozen=True, slots=True)
class ArchiveDiverseEliteCampaignParentSelector:
    """Select two nondominated parents with workload-neutral structural spread.

    The first lane rotates over the authenticated current Pareto front.  The
    second lane is the distinct current-front configuration requiring the
    largest canonical typed-patch operation set from that anchor.  This keeps
    exploration on known trade-off-efficient material while exposing generic
    structural diversity to mutation and crossover.  A singleton front falls
    back explicitly to the structurally most distant eligible history member,
    because campaign parent occurrences must remain distinct.
    """

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        parent_count: int,
        rotation_index: int,
        progress: tuple[CampaignParentSelectionProgress, ...] = (),
        archive_utility: ArchiveUtilitySnapshot | None = None,
    ) -> CampaignParentSelection:
        del archive_utility
        if type(self) is not ArchiveDiverseEliteCampaignParentSelector:
            raise TypeError("selector must be an exact diverse-elite selector")
        OptimizerState.__post_init__(state)
        _validate_parent_selection_progress(
            progress,
            optimizer_generation=state.generation,
        )
        require_sha256(task_sha256, "task_sha256")
        if type(parent_count) is not int or parent_count != 2:
            raise ValueError(
                "diverse-elite campaign parent selection requires exactly two lanes"
            )
        if type(rotation_index) is not int or not 0 <= rotation_index < (1 << 63):
            raise ValueError("rotation_index must be an exact non-negative int63")

        candidates = state.archive.front_candidates
        references = state.archive.front_references
        if not candidates or len(candidates) != len(references):
            raise ValueError("current archive front is empty or internally misaligned")
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        if (
            tuple(by_id[reference.candidate_id] for reference in references)
            != candidates
        ):
            raise ValueError("current archive candidates differ from front references")

        rotation_digest = hashlib.sha256(
            _DIVERSE_ELITE_PARENT_ROTATION_DOMAIN
            + bytes.fromhex(task_sha256)
            + bytes.fromhex(state.archive_snapshot_hash)
        ).digest()
        rotation_anchor = int.from_bytes(rotation_digest, "big") % len(candidates)
        elite_ordinal = (rotation_anchor + rotation_index) % len(candidates)
        elite = candidates[elite_ordinal]

        distance_records: list[dict[str, object]] = []
        if len(candidates) == 1:
            explorer_pool = tuple(
                candidate
                for candidate in state.candidates
                if candidate.candidate_id != elite.candidate_id
                and candidate.occurrence.configuration_hash
                != elite.occurrence.configuration_hash
                and candidate.valid
                and candidate.operator_compliant
                and candidate.evidence_compliant
                and candidate.objectives
            )
            if not explorer_pool:
                raise ValueError(
                    "diverse-elite selection requires two eligible distinct "
                    "candidate occurrences"
                )
            explorer_sources = tuple(
                (len(candidates) + index, candidate)
                for index, candidate in enumerate(explorer_pool)
            )
            fallback_reason = "singleton_front_structural_history_fallback"
        else:
            explorer_sources = tuple(
                (ordinal, candidate)
                for ordinal, candidate in enumerate(candidates)
                if ordinal != elite_ordinal
            )
            fallback_reason = "none"

        scored: list[tuple[int, bytes, int, EvolutionCandidate, str]] = []
        for ordinal, candidate in explorer_sources:
            patch = derive_patch(
                elite.configuration,
                candidate.configuration,
                base_candidate_id=elite.candidate_id,
                target_candidate_id=candidate.candidate_id,
            )
            operation_distance = len(patch.operations)
            tie_digest = hashlib.sha256(
                _DIVERSE_ELITE_PARENT_TIE_DOMAIN
                + bytes.fromhex(task_sha256)
                + bytes.fromhex(state.archive_snapshot_hash)
                + rotation_index.to_bytes(8, "big", signed=False)
                + bytes.fromhex(candidate.occurrence.configuration_hash)
            ).digest()
            scored.append(
                (
                    operation_distance,
                    tie_digest,
                    ordinal,
                    candidate,
                    patch.patch_hash,
                )
            )
            distance_records.append(
                {
                    "candidate_id": candidate.candidate_id.value,
                    "configuration_hash": candidate.occurrence.configuration_hash,
                    "typed_patch_operation_distance": operation_distance,
                    "typed_patch_sha256": patch.patch_hash,
                }
            )
        maximum_distance = max(value[0] for value in scored)
        finalists = tuple(value for value in scored if value[0] == maximum_distance)
        selected = min(finalists, key=lambda value: (value[1], value[2]))
        _, _, explorer_ordinal, explorer, _ = selected

        selected_parents = (elite, explorer)
        lanes = (
            CampaignParentLane(lane_id="elite", parent=elite),
            CampaignParentLane(lane_id="explorer", parent=explorer),
        )
        return CampaignParentSelection(
            parents=selected_parents,
            lanes=lanes,
            evidence=_object(
                {
                    "schema_version": 1,
                    "policy_id": _DIVERSE_ELITE_PARENT_POLICY_ID,
                    "policy_version": _DIVERSE_ELITE_PARENT_POLICY_VERSION,
                    "policy_definition_sha256": (
                        _DIVERSE_ELITE_PARENT_POLICY_DEFINITION_SHA256
                    ),
                    "task_sha256": task_sha256,
                    "optimizer_generation": state.generation,
                    "archive_snapshot_hash": state.archive_snapshot_hash,
                    "eligible_front": [
                        reference.to_trace_record() for reference in references
                    ],
                    "rotation_index": rotation_index,
                    "rotation_anchor": rotation_anchor,
                    "elite_ordinal": elite_ordinal,
                    "explorer_ordinal": explorer_ordinal,
                    "maximum_typed_patch_operation_distance": maximum_distance,
                    "distance_evidence": sorted(
                        distance_records,
                        key=lambda value: str(value["candidate_id"]),
                    ),
                    "fallback_reason": fallback_reason,
                    "selected_parent_ids": [
                        parent.candidate_id.value for parent in selected_parents
                    ],
                    "selected_parents_are_current_front_members": all(
                        parent.candidate_id in by_id for parent in selected_parents
                    ),
                    "provider_fields_consulted": False,
                    "workload_semantics_consulted": False,
                }
            ),
        )


_STAGNATION_AWARE_PARENT_POLICY_ID = "stagnation_aware_diverse_campaign_parent"
_STAGNATION_AWARE_PARENT_POLICY_VERSION = 1
_STAGNATION_AWARE_PARENT_WINDOW = 2
_STAGNATION_AWARE_PARENT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:stagnation-aware-diverse-campaign-parent:v1:"
    b"normal=archive-diverse-elite-campaign-parent-v1;"
    b"trigger=two-consecutive-runtime-authenticated-stage-progress-receipts-with-"
    b"no-increase-in-generic-scalar-archive-utility;"
    b"elite=normal-policy-elite;"
    b"stagnant-explorer=eligible-nonfront-history-member-maximizing-minimum-"
    b"canonical-typed-patch-operation-distance-to-current-front;"
    b"ties=task-snapshot-rotation-and-configuration-keyed-sha256;"
    b"empty-pool=normal-policy-fallback;"
    b"provider-fields=false;workload-semantics=false"
).hexdigest()
_STAGNATION_AWARE_PARENT_TIE_DOMAIN = (
    b"agent-evolve:stagnation-aware-diverse-campaign-parent:tie:v1\x00"
)


@dataclass(frozen=True, slots=True)
class StagnationAwareDiverseCampaignParentSelector:
    """Switch one parent lane to structurally remote history after stagnation.

    Normal generations use :class:`ArchiveDiverseEliteCampaignParentSelector`
    exactly. Once the two most recent authenticated generation receipts both
    report an unchanged archive snapshot, the elite lane remains untouched and
    the explorer lane is sourced from eligible evaluated history outside the
    current front. Its score is the minimum canonical typed-patch operation
    distance to *every* current-front member, so the switch seeks a genuinely
    different basin rather than a point remote from only one arbitrary anchor.

    The detector and distance law consume no objective names, workload
    semantics, prompts, provider fields, or model identity. If no eligible
    nonfront source exists, selection falls back explicitly to the normal
    diverse-elite policy and records that fact.
    """

    policy_id = _STAGNATION_AWARE_PARENT_POLICY_ID
    policy_version = _STAGNATION_AWARE_PARENT_POLICY_VERSION
    definition_sha256 = _STAGNATION_AWARE_PARENT_POLICY_DEFINITION_SHA256
    stagnation_window = _STAGNATION_AWARE_PARENT_WINDOW

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        parent_count: int,
        rotation_index: int,
        progress: tuple[CampaignParentSelectionProgress, ...] = (),
        archive_utility: ArchiveUtilitySnapshot | None = None,
    ) -> CampaignParentSelection:
        if type(self) is not StagnationAwareDiverseCampaignParentSelector:
            raise TypeError("selector must be an exact stagnation-aware selector")
        OptimizerState.__post_init__(state)
        _validate_parent_selection_progress(
            progress,
            optimizer_generation=state.generation,
        )
        require_sha256(task_sha256, "task_sha256")
        if type(parent_count) is not int or parent_count != 2:
            raise ValueError(
                "stagnation-aware campaign parent selection requires exactly two lanes"
            )
        if type(rotation_index) is not int or not 0 <= rotation_index < (1 << 63):
            raise ValueError("rotation_index must be an exact non-negative int63")

        normal = ArchiveDiverseEliteCampaignParentSelector().select(
            state,
            task_sha256=task_sha256,
            parent_count=parent_count,
            rotation_index=rotation_index,
            progress=progress,
            archive_utility=archive_utility,
        )
        receipt_window = progress[-self.stagnation_window :]
        if any(not receipt.utility_signal_available for receipt in progress):
            raise ValueError(
                "stagnation-aware selection requires authenticated scalar "
                "archive utility on every completed stage"
            )
        receipt_evidence = [receipt.to_record() for receipt in receipt_window]
        stagnation_triggered = len(receipt_window) == self.stagnation_window and all(
            receipt.utility_improved is False for receipt in receipt_window
        )

        selected_parents = normal.parents
        source_mode = "normal_diverse_elite"
        source_switch_applied = False
        selected_history_ordinal: int | None = None
        maximum_minimum_distance: int | None = None
        distance_records: list[dict[str, object]] = []

        front = state.archive.front_candidates
        front_ids = {candidate.candidate_id for candidate in front}
        front_configuration_hashes = {
            candidate.occurrence.configuration_hash for candidate in front
        }
        eligible_history = tuple(
            (ordinal, candidate)
            for ordinal, candidate in enumerate(state.candidates)
            if candidate.candidate_id not in front_ids
            and candidate.occurrence.configuration_hash
            not in front_configuration_hashes
            and candidate.valid
            and candidate.operator_compliant
            and candidate.evidence_compliant
            and candidate.objectives
        )

        if stagnation_triggered and eligible_history:
            scored: list[tuple[int, bytes, int, EvolutionCandidate]] = []
            for history_ordinal, candidate in eligible_history:
                front_distances: list[dict[str, object]] = []
                operation_distances: list[int] = []
                for front_candidate in front:
                    patch = derive_patch(
                        front_candidate.configuration,
                        candidate.configuration,
                        base_candidate_id=front_candidate.candidate_id,
                        target_candidate_id=candidate.candidate_id,
                    )
                    operation_distance = len(patch.operations)
                    operation_distances.append(operation_distance)
                    front_distances.append(
                        {
                            "front_candidate_id": (front_candidate.candidate_id.value),
                            "front_configuration_sha256": (
                                front_candidate.occurrence.configuration_hash
                            ),
                            "typed_patch_operation_distance": operation_distance,
                            "typed_patch_sha256": patch.patch_hash,
                        }
                    )
                minimum_distance = min(operation_distances)
                tie_digest = hashlib.sha256(
                    _STAGNATION_AWARE_PARENT_TIE_DOMAIN
                    + bytes.fromhex(task_sha256)
                    + bytes.fromhex(state.archive_snapshot_hash)
                    + rotation_index.to_bytes(8, "big", signed=False)
                    + bytes.fromhex(candidate.occurrence.configuration_hash)
                ).digest()
                scored.append(
                    (minimum_distance, tie_digest, history_ordinal, candidate)
                )
                distance_records.append(
                    {
                        "candidate_id": candidate.candidate_id.value,
                        "configuration_sha256": (
                            candidate.occurrence.configuration_hash
                        ),
                        "history_ordinal": history_ordinal,
                        "minimum_typed_patch_operation_distance_to_front": (
                            minimum_distance
                        ),
                        "front_distance_evidence": sorted(
                            front_distances,
                            key=lambda value: str(value["front_candidate_id"]),
                        ),
                    }
                )
            maximum_minimum_distance = max(value[0] for value in scored)
            finalists = tuple(
                value for value in scored if value[0] == maximum_minimum_distance
            )
            _, _, selected_history_ordinal, explorer = min(
                finalists,
                key=lambda value: (value[1], value[2]),
            )
            selected_parents = (normal.parents[0], explorer)
            source_mode = "stagnation_remote_history"
            source_switch_applied = True
        elif stagnation_triggered:
            source_mode = "stagnation_triggered_normal_fallback"

        lanes = (
            CampaignParentLane(lane_id="elite", parent=selected_parents[0]),
            CampaignParentLane(lane_id="explorer", parent=selected_parents[1]),
        )
        return CampaignParentSelection(
            parents=selected_parents,
            lanes=lanes,
            evidence=_object(
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "policy_definition_sha256": self.definition_sha256,
                    "task_sha256": task_sha256,
                    "optimizer_generation": state.generation,
                    "archive_snapshot_sha256": state.archive_snapshot_hash,
                    "rotation_index": rotation_index,
                    "stagnation_window": self.stagnation_window,
                    "observed_receipt_count": len(receipt_window),
                    "stagnation_receipt_evidence": receipt_evidence,
                    "stagnation_triggered": stagnation_triggered,
                    "source_switch_applied": source_switch_applied,
                    "source_mode": source_mode,
                    "eligible_nonfront_history_count": len(eligible_history),
                    "maximum_minimum_typed_patch_operation_distance_to_front": (
                        maximum_minimum_distance
                    ),
                    "selected_history_ordinal": selected_history_ordinal,
                    "distance_evidence": sorted(
                        distance_records,
                        key=lambda value: str(value["candidate_id"]),
                    ),
                    "normal_policy_evidence": thaw_json(normal.evidence),
                    "selected_parent_ids": [
                        parent.candidate_id.value for parent in selected_parents
                    ],
                    "selected_parent_configuration_sha256s": [
                        parent.occurrence.configuration_hash
                        for parent in selected_parents
                    ],
                    "selected_parents_are_current_front_members": all(
                        parent.candidate_id in front_ids for parent in selected_parents
                    ),
                    "provider_fields_consulted": False,
                    "workload_semantics_consulted": False,
                    "objective_names_consulted": False,
                }
            ),
        )


_RESIDUAL_FRONTIER_PARENT_POLICY_ID = "residual_hypervolume_campaign_parent"
_RESIDUAL_FRONTIER_PARENT_POLICY_VERSION = 1
_RESIDUAL_FRONTIER_PARENT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:residual-hypervolume-campaign-parent:v1;"
    b"input=current-front-plus-authenticated-affine-prior-archive;"
    b"target=largest-positive-pairwise-midpoint-hypervolume-residual;"
    b"parents=distinct-current-front-members-nearest-two-cell-anchors;"
    b"fallback=stagnation-aware-diverse-parent-when-no-positive-cell;"
    b"lanes=stable-elite-explorer-identifiers-for-contract-compatibility;"
    b"current-future-outcomes=false;workload-model-provider-fields=false"
).hexdigest()


@dataclass(frozen=True, slots=True)
class ResidualHypervolumeCampaignParentSelector:
    """Coordinate two parent lanes around the largest missing frontier cell."""

    policy_id = _RESIDUAL_FRONTIER_PARENT_POLICY_ID
    policy_version = _RESIDUAL_FRONTIER_PARENT_POLICY_VERSION
    definition_sha256 = _RESIDUAL_FRONTIER_PARENT_POLICY_DEFINITION_SHA256

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        parent_count: int,
        rotation_index: int,
        progress: tuple[CampaignParentSelectionProgress, ...] = (),
        archive_utility: ArchiveUtilitySnapshot | None = None,
    ) -> CampaignParentSelection:
        if type(self) is not ResidualHypervolumeCampaignParentSelector:
            raise TypeError("selector must be an exact residual-frontier selector")
        OptimizerState.__post_init__(state)
        _validate_parent_selection_progress(
            progress,
            optimizer_generation=state.generation,
        )
        require_sha256(task_sha256, "task_sha256")
        if type(parent_count) is not int or parent_count != 2:
            raise ValueError(
                "residual-frontier campaign parent selection requires two lanes"
            )
        if type(rotation_index) is not int or not 0 <= rotation_index < (1 << 63):
            raise ValueError("rotation_index must be an exact non-negative int63")
        if type(archive_utility) is not ArchiveUtilitySnapshot:
            raise TypeError(
                "residual-frontier parent selection requires archive_utility"
            )
        archive_utility.__post_init__()
        if archive_utility.generation != state.generation + 1:
            raise ValueError(
                "residual-frontier archive utility is stale for optimizer state"
            )

        geometry = residual_frontier_geometry(archive_utility)
        selected = residual_anchor_parents(
            geometry=geometry,
            candidates=state.archive.front_candidates,
        )
        if selected is None:
            fallback = StagnationAwareDiverseCampaignParentSelector().select(
                state,
                task_sha256=task_sha256,
                parent_count=parent_count,
                rotation_index=rotation_index,
                progress=progress,
                archive_utility=archive_utility,
            )
            return CampaignParentSelection(
                parents=fallback.parents,
                lanes=fallback.lanes,
                decision_slots=fallback.decision_slots,
                evidence=_object(
                    {
                        "schema_version": 1,
                        "policy_id": self.policy_id,
                        "policy_version": self.policy_version,
                        "policy_definition_sha256": self.definition_sha256,
                        "residual_geometry_sha256": geometry.geometry_sha256,
                        "residual_cell_count": len(geometry.cells),
                        "source_mode": "no_positive_residual_fallback",
                        "fallback_evidence": thaw_json(fallback.evidence),
                        "current_or_future_candidate_outcomes_consulted": False,
                        "workload_model_provider_fields_consulted": False,
                    }
                ),
            )

        front_ids = {value.candidate_id for value in state.archive.front_candidates}
        if any(value.candidate_id not in front_ids for value in selected):
            raise ValueError("residual-frontier parents escaped the current front")
        best_cell = geometry.cells[0]
        lanes = (
            CampaignParentLane(lane_id="elite", parent=selected[0]),
            CampaignParentLane(lane_id="explorer", parent=selected[1]),
        )
        return CampaignParentSelection(
            parents=selected,
            lanes=lanes,
            evidence=_object(
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "policy_definition_sha256": self.definition_sha256,
                    "residual_core_definition_sha256": (
                        RESIDUAL_FRONTIER_POLICY_DEFINITION_SHA256
                    ),
                    "task_sha256": task_sha256,
                    "optimizer_generation": state.generation,
                    "archive_snapshot_sha256": state.archive_snapshot_hash,
                    "archive_utility_snapshot_sha256": (
                        archive_utility.snapshot_sha256
                    ),
                    "rotation_index": rotation_index,
                    "residual_geometry_sha256": geometry.geometry_sha256,
                    "residual_cell_count": len(geometry.cells),
                    "selected_cell": best_cell.to_record(),
                    "selected_parent_ids": [
                        value.candidate_id.value for value in selected
                    ],
                    "selected_parent_configuration_sha256s": [
                        value.occurrence.configuration_hash for value in selected
                    ],
                    "source_mode": "largest_positive_residual_cell_anchors",
                    "current_or_future_candidate_outcomes_consulted": False,
                    "workload_model_provider_fields_consulted": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ArchiveEliteExplorerCampaignParentSelector:
    """Bind the generic named elite/explorer lanes to campaign execution.

    The underlying policy always returns exactly two parents: one current-front
    elite and one best-rank dominated explorer, with explicit distinct-front or
    singleton fallback.  Keeping that fixed cardinality at this adapter makes a
    campaign's lane semantics impossible to erase by requesting an arbitrary
    parent count.
    """

    policy: TaskKeyedArchiveEliteExplorerParentPolicy = field(
        default_factory=TaskKeyedArchiveEliteExplorerParentPolicy
    )

    def __post_init__(self) -> None:
        if type(self.policy) is not TaskKeyedArchiveEliteExplorerParentPolicy:
            raise TypeError("policy must be an exact elite/explorer parent policy")

    def select(
        self,
        state: OptimizerState,
        *,
        task_sha256: str,
        parent_count: int,
        rotation_index: int,
        progress: tuple[CampaignParentSelectionProgress, ...] = (),
        archive_utility: ArchiveUtilitySnapshot | None = None,
    ) -> CampaignParentSelection:
        del archive_utility
        _validate_parent_selection_progress(
            progress,
            optimizer_generation=state.generation,
        )
        if type(parent_count) is not int or parent_count != 2:
            raise ValueError(
                "elite/explorer campaign parent selection requires exactly two lanes"
            )
        selected = self.policy.select(
            state,
            task_sha256=task_sha256,
            expected_archive_snapshot_hash=state.archive_snapshot_hash,
            rotation_index=rotation_index,
        )
        lanes = tuple(
            CampaignParentLane(lane_id=lane_id, parent=parent)
            for lane_id, parent in zip(
                ("elite", "explorer"),
                selected.parents,
                strict=True,
            )
        )
        return CampaignParentSelection(
            parents=selected.parents,
            evidence=_object(selected.receipt.to_trace_record()),
            lanes=lanes,
        )


@dataclass(frozen=True, slots=True)
class CampaignPortfolioWaveContext:
    """Complete trusted input to one prompt/card treatment construction."""

    prepared: PreparedEvolutionCampaign
    stage_request: CampaignStageRequest
    parent_slot: int
    parent: EvolutionCandidate
    variation: ParentVariationBinding
    evidence_context: FrozenJsonObject
    evidence_cards: tuple[FrozenJsonObject, ...]
    memory: FrozenJsonObject
    parent_measurement: ParentMeasurementBinding | None = None
    archive_context: CampaignPortfolioArchiveContextProjection | None = None
    test_eligible_reflections: tuple[tuple[str, FrozenJsonObject], ...] = ()
    parent_lane: CampaignParentLane | None = None
    decision_slot: CampaignDecisionSlot | None = None
    contextual_allocation: ContextualPortfolioAllocationContract | None = None
    frontier_target: CampaignPortfolioFrontierTarget | None = None

    def __post_init__(self) -> None:
        if type(self.prepared) is not PreparedEvolutionCampaign:
            raise TypeError("prepared must be exact")
        if type(self.stage_request) is not CampaignStageRequest:
            raise TypeError("stage_request must be exact")
        if self.stage_request.step.kind is not CampaignGenerationKind.PORTFOLIO:
            raise ValueError("portfolio context requires a portfolio stage")
        if type(self.parent_slot) is not int or self.parent_slot < 0:
            raise ValueError("parent_slot must be non-negative")
        if type(self.parent) is not EvolutionCandidate:
            raise TypeError("parent must be exact")
        if type(self.variation) is not ParentVariationBinding:
            raise TypeError("variation must be exact")
        lane = self.parent_lane
        slot = self.decision_slot
        if lane is None and slot is None:
            lane = CampaignParentLane(
                lane_id=f"parent_{self.parent_slot + 1:04d}",
                parent=self.parent,
            )
            slot = CampaignDecisionSlot(
                slot_id=f"{lane.lane_id}.primary",
                lane_id=lane.lane_id,
            )
            object.__setattr__(self, "parent_lane", lane)
            object.__setattr__(self, "decision_slot", slot)
        elif lane is None or slot is None:
            raise ValueError("parent_lane and decision_slot must be supplied together")
        if type(lane) is not CampaignParentLane:
            raise TypeError("parent_lane must be exact or None")
        if type(slot) is not CampaignDecisionSlot:
            raise TypeError("decision_slot must be exact or None")
        CampaignParentLane.__post_init__(lane)
        CampaignDecisionSlot.__post_init__(slot)
        if lane.parent is not self.parent or slot.lane_id != lane.lane_id:
            raise ValueError("decision slot, parent lane, and parent differ")
        allocation = self.contextual_allocation
        if allocation is not None:
            if type(allocation) is not ContextualPortfolioAllocationContract:
                raise TypeError("contextual_allocation must be exact or None")
            allocation.__post_init__()
            if (
                allocation.campaign_generation != self.stage_request.step.generation
                or allocation.slice_id != lane.lane_id
                or allocation.evaluation_slots != self.prepared.protocol.portfolio_width
            ):
                raise ValueError("contextual allocation differs from its campaign lane")
        for name in ("evidence_context", "memory"):
            value = getattr(self, name)
            if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
                raise TypeError(f"{name} must be an exact frozen object")
        evidence_values = dict(self.evidence_context.items)
        if CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY in evidence_values:
            raise ValueError(
                "trusted evidence context uses the reserved selector extension key"
            )
        archive_context = self.archive_context
        archive_payload = evidence_values.get(CAMPAIGN_ARCHIVE_CONTEXT_KEY)
        if archive_context is not None:
            if type(archive_context) is not CampaignPortfolioArchiveContextProjection:
                raise TypeError("archive_context must be exact or None")
            archive_context.__post_init__()
            if (
                archive_context.archive_utility_snapshot_sha256
                != self.stage_request.archive_utility.snapshot_sha256
                or archive_context.parent_configuration_sha256
                != self.parent.occurrence.configuration_hash
            ):
                raise ValueError("archive context differs from its campaign lane")
            if (
                archive_payload is None
                or thaw_json(archive_payload) != archive_context.to_record()
            ):
                raise ValueError(
                    "archive context payload differs from its authenticated receipt"
                )
        target = self.frontier_target
        target_payload = evidence_values.get(CAMPAIGN_FRONTIER_TARGET_KEY)
        if target is None:
            if target_payload is not None:
                raise ValueError(
                    "trusted evidence context uses an unauthenticated frontier target"
                )
        else:
            if type(target) is not CampaignPortfolioFrontierTarget:
                raise TypeError("frontier_target must be exact or None")
            target.__post_init__()
            if (
                target.lane_id != lane.lane_id
                or target.parent_configuration_sha256
                != self.parent.occurrence.configuration_hash
                or target.archive_utility_snapshot_sha256
                != self.stage_request.archive_utility.snapshot_sha256
            ):
                raise ValueError("frontier target differs from its campaign lane")
            if (
                target_payload is None
                or thaw_json(target_payload) != target.to_record()
            ):
                raise ValueError(
                    "frontier target payload differs from its authenticated receipt"
                )
        if type(self.evidence_cards) is not tuple or not self.evidence_cards:
            raise ValueError("portfolio context requires evidence cards")
        if any(type(value) is not FrozenJsonObject for value in self.evidence_cards):
            raise TypeError("evidence_cards must contain frozen objects")
        measurement = self.parent_measurement
        if measurement is not None:
            if type(measurement) is not ParentMeasurementBinding:
                raise TypeError("parent_measurement must be exact or None")
            measurement.__post_init__()
            occurrence = self.parent.occurrence
            operator_invocation_id = (
                None
                if occurrence.operator_invocation_id is None
                else occurrence.operator_invocation_id.value
            )
            if (
                measurement.candidate.candidate_id != self.parent.candidate_id.value
                or measurement.candidate.configuration_sha256
                != occurrence.configuration_hash
                or measurement.candidate.configuration_artifact_sha256
                != occurrence.configuration_artifact_hash
                or measurement.candidate.proposal_sequence
                != occurrence.proposal_sequence
                or measurement.candidate.operator_invocation_id
                != operator_invocation_id
                or measurement.projection.benchmark_sha256
                != self.variation.benchmark_sha256
                or measurement.projection.session_sha256
                != self.prepared.benchmark_session.session_sha256
            ):
                raise ValueError("parent measurement differs from campaign parent")
        if type(self.test_eligible_reflections) is not tuple:
            raise TypeError("test_eligible_reflections must be an exact tuple")
        observed_hashes: list[str] = []
        for item in self.test_eligible_reflections:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or len(item[0]) != 64
                or type(item[1]) is not FrozenJsonObject
            ):
                raise TypeError(
                    "test-eligible reflections must contain hash/object pairs"
                )
            observed_hashes.append(item[0])
        if tuple(observed_hashes) != tuple(sorted(set(observed_hashes))):
            raise ValueError("test-eligible reflection hashes must be canonical")


@runtime_checkable
class CampaignPortfolioWaveFactory(Protocol):
    """Turn trusted workload evidence into one executable portfolio wave."""

    def build(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> PortfolioVariationWaveRequest: ...


@runtime_checkable
class CampaignPortfolioWaveBatchFactory(Protocol):
    """Optionally construct one portfolio stage as an atomic lane cohort.

    Batch construction is required when a workload-neutral policy must solve a
    constraint across lanes, such as assigning distinct compatible memory
    cards.  The ordinary single-wave factory remains the default compatibility
    surface; runtimes opt into this protocol structurally through
    ``build_batch``.
    """

    def build_batch(
        self,
        contexts: tuple[CampaignPortfolioWaveContext, ...],
    ) -> tuple[PortfolioVariationWaveRequest, ...]: ...


@runtime_checkable
class CampaignPortfolioContextEnricher(Protocol):
    """Retrieve parent/lane-local history immediately before wave building.

    The input is the complete authenticated base context.  Implementations may
    query contextual outcome memory, but return only a bounded frozen history
    payload.  The runtime attaches it under a reserved key, preserving every
    workload- and core-authored base-context field exactly.
    """

    def enrich(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> FrozenJsonObject: ...


@dataclass(frozen=True, slots=True)
class CampaignPortfolioMemoryEstimandProjection:
    """Atomic replacement for the core-owned memory-estimand context pair.

    The campaign runtime, rather than a workload wave factory, installs this
    projection into the trusted selector context.  The digest is deliberately
    redundant: it binds a diagnostic stratum to the exact typed-JSON estimand
    object and prevents a projector from publishing a mismatched pair.
    """

    estimand_context: FrozenJsonObject
    estimand_stratum_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.estimand_context) is not FrozenJsonObject
            or freeze_json(self.estimand_context) is not self.estimand_context
        ):
            raise TypeError("estimand_context must be an exact frozen object")
        _require_sha256(
            self.estimand_stratum_sha256,
            "estimand_stratum_sha256",
        )
        if typed_json_sha256(self.estimand_context) != self.estimand_stratum_sha256:
            raise ValueError(
                "estimand_stratum_sha256 must identify the exact estimand context"
            )


@runtime_checkable
class CampaignPortfolioMemoryEstimandProjector(Protocol):
    """Project a prospective wave onto an authenticated memory estimand.

    Returning ``None`` preserves the workload-authored selector context.  A
    projection authorizes replacement of exactly the core-owned estimand
    object and matching stratum digest; every other context field is retained
    byte-for-byte by the campaign runtime.
    """

    def project(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> CampaignPortfolioMemoryEstimandProjection | None: ...


@dataclass(frozen=True, slots=True)
class CampaignPortfolioWavePreparationReceipt:
    """Content-free evidence that one trusted wave was ready for dispatch.

    This receipt is emitted after the wave factory's output has passed the
    campaign trust-boundary validation and before any selector/provider work
    starts.  It deliberately records prompt-adjacent objects only through
    their existing canonical hashes.  ``card_records`` are the public
    :class:`PortfolioCard` audit records, which contain prompt-view hashes but
    never the prompt payload itself.

    A failed stage is not a sealed scientific result.  The receipt instead
    preserves the prospective treatment, projection, and lineage needed to
    audit why that unsealed stage would have dispatched the cards it did.
    """

    campaign_preparation_sha256: str
    stage_request_sha256: str
    stage_preparation_sha256: str
    generation: int
    parent_slot: int
    parent_lane_id: str
    decision_slot_id: str
    parent_candidate_id: str
    parent_configuration_sha256: str
    parent_configuration_artifact_sha256: str
    finite_contract_identity_sha256: str
    selector_call_id: str
    selector_request_sha256: str
    workload_context_sha256: str
    pre_memory_projection_context_sha256: str
    selector_context_sha256: str
    evidence_card_snapshot_sha256: str
    selector_card_snapshot_sha256: str
    card_records: tuple[FrozenJsonObject, ...]
    card_reference_mapping: tuple[FrozenJsonObject, ...]
    test_eligible_reflection_receipts: tuple[FrozenJsonObject, ...]
    context_projection_identity: FrozenJsonObject
    memory_credit_identity: FrozenJsonObject | None
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_preparation_sha256",
            "stage_request_sha256",
            "stage_preparation_sha256",
            "parent_configuration_sha256",
            "parent_configuration_artifact_sha256",
            "finite_contract_identity_sha256",
            "selector_request_sha256",
            "workload_context_sha256",
            "pre_memory_projection_context_sha256",
            "selector_context_sha256",
            "evidence_card_snapshot_sha256",
            "selector_card_snapshot_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if type(self.parent_slot) is not int or self.parent_slot < 0:
            raise ValueError("parent_slot must be a non-negative exact integer")
        _require_role_token(self.parent_lane_id, "parent_lane_id")
        _require_role_token(self.decision_slot_id, "decision_slot_id")
        for name in ("parent_candidate_id", "selector_call_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty exact string")
        for name in (
            "card_records",
            "card_reference_mapping",
            "test_eligible_reflection_receipts",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not FrozenJsonObject or freeze_json(value) is not value
                for value in values
            ):
                raise TypeError(f"{name} must contain exact frozen objects")
        if not self.card_records:
            raise ValueError("a prepared portfolio wave requires card records")
        if len(self.card_records) != len(self.card_reference_mapping):
            raise ValueError("card records and reference mapping differ")
        card_keys = tuple(
            thaw_json(value).get("card_key") for value in self.card_records
        )
        mapped_keys = tuple(
            thaw_json(value).get("card_key") for value in self.card_reference_mapping
        )
        if card_keys != mapped_keys or card_keys != tuple(sorted(set(card_keys))):
            raise ValueError("card records and references require canonical card keys")
        reflection_receipts = tuple(
            thaw_json(value).get("reflection_receipt_sha256")
            for value in self.test_eligible_reflection_receipts
        )
        if reflection_receipts != tuple(sorted(set(reflection_receipts))):
            raise ValueError("test-eligible reflection receipts must be canonical")
        for value in self.test_eligible_reflection_receipts:
            record = thaw_json(value)
            _require_sha256(
                record.get("reflection_receipt_sha256"),
                "reflection_receipt_sha256",
            )
            _require_sha256(
                record.get("reflection_result_sha256"),
                "reflection_result_sha256",
            )
        for name in ("context_projection_identity", "memory_credit_identity"):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not FrozenJsonObject or freeze_json(value) is not value
            ):
                raise TypeError(f"{name} must be an exact frozen object or None")
        context_projection = thaw_json(self.context_projection_identity)
        if (
            context_projection.get("selector_context_sha256")
            != self.selector_context_sha256
        ):
            raise ValueError("context projection names a different selector context")
        object.__setattr__(
            self,
            "receipt_sha256",
            _preparation_sha256(
                _WAVE_PREPARATION_DOMAIN,
                _object(self._unsigned_record()),
            ),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_preparation_sha256": self.campaign_preparation_sha256,
            "stage_request_sha256": self.stage_request_sha256,
            "stage_preparation_sha256": self.stage_preparation_sha256,
            "generation": self.generation,
            "parent_slot": self.parent_slot,
            "parent_lane_id": self.parent_lane_id,
            "decision_slot_id": self.decision_slot_id,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "parent_configuration_artifact_sha256": (
                self.parent_configuration_artifact_sha256
            ),
            "finite_contract_identity_sha256": (self.finite_contract_identity_sha256),
            "selector_call_id": self.selector_call_id,
            "selector_request_sha256": self.selector_request_sha256,
            "workload_context_sha256": self.workload_context_sha256,
            "pre_memory_projection_context_sha256": (
                self.pre_memory_projection_context_sha256
            ),
            "selector_context_sha256": self.selector_context_sha256,
            "evidence_card_snapshot_sha256": self.evidence_card_snapshot_sha256,
            "selector_card_snapshot_sha256": self.selector_card_snapshot_sha256,
            "card_records": [thaw_json(value) for value in self.card_records],
            "card_reference_mapping": [
                thaw_json(value) for value in self.card_reference_mapping
            ],
            "test_eligible_reflection_receipts": [
                thaw_json(value) for value in self.test_eligible_reflection_receipts
            ],
            "context_projection_identity": thaw_json(self.context_projection_identity),
            "memory_credit_identity": (
                None
                if self.memory_credit_identity is None
                else thaw_json(self.memory_credit_identity)
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@runtime_checkable
class CampaignPortfolioWavePreparationObserver(Protocol):
    """Durably record a validated prospective wave before dispatch.

    Implementations must finish durable publication before returning.  Raising
    aborts the stage before any selector call is submitted.  The observer is
    intentionally separate from campaign stage receipts because a failed or
    cancelled stage never has a sealed :class:`CampaignStageReceipt`.
    """

    def record_prepared_wave(
        self,
        receipt: CampaignPortfolioWavePreparationReceipt,
    ) -> None: ...


def _validated_wave_preparation_receipt(
    *,
    build: CampaignPortfolioWaveContext,
    wave: PortfolioVariationWaveRequest,
    workload_context_sha256: str,
    pre_memory_projection_context_sha256: str,
) -> CampaignPortfolioWavePreparationReceipt:
    """Project an already validated wave into content-free audit evidence."""

    request = wave.selection_request
    credit = wave.memory_credit
    matched = wave.matched_memory_control
    if credit is None and matched is None:
        context_values = dict(request.context.items)
        context_projection = (
            PortfolioMemoryContextProjectionBinding.from_selector_context(
                request.context
            )
            if MEMORY_ESTIMAND_CONTEXT_KEY in context_values
            else PortfolioMemoryContextProjectionBinding.exact_identity(
                request.context_sha256
            )
        )
        memory_credit_identity = None
    elif credit is not None:
        context_projection = credit.resolve_context_projection(request.context)
        admission = credit.quarantine_admission
        memory_credit_identity = _object(
            {
                "schema_version": 1,
                "credit_unit_id": credit.credit_unit_id.value,
                "treatment_binding_sha256": credit.treatment_binding_sha256,
                "selection_decision_sha256": (
                    credit.assignment.selection_decision_sha256
                ),
                "assignment_sha256": credit.assignment.assignment_sha256,
                "score_snapshot_sha256": credit.score_snapshot.snapshot_sha256,
                "aggregation_binding_sha256": credit.aggregation.binding_sha256,
                "card_source_registry_sha256": (credit.card_source_registry_sha256),
                "context_projection_binding_sha256": (
                    context_projection.binding_sha256
                ),
                "quarantine_admission": (
                    None
                    if admission is None
                    else {
                        "receipt_sha256": admission.receipt_sha256,
                        "source_admission_request_sha256": (
                            admission.source_admission_request_sha256
                        ),
                        "references": [
                            {
                                "insight_id": value.insight_id.value,
                                "version": value.version,
                            }
                            for value in admission.references
                        ],
                    }
                ),
                "quarantine_admission_subset_authorization_sha256": (
                    credit.quarantine_admission_subset_authorization_sha256
                ),
            }
        )
    else:
        assert matched is not None
        context_projection = matched.context_projection
        context_projection.replay(request.context)
        parent_lane = build.parent_lane
        if parent_lane is None:
            raise ValueError("matched memory arm requires a stable campaign lane")
        if (
            matched.assignment.unit.generation != wave.generation
            or matched.assignment.unit.lane_id != parent_lane.lane_id
        ):
            raise ValueError("matched memory assignment differs from the campaign lane")
        memory_credit_identity = _object(
            {
                "schema_version": 2,
                "evidence_kind": "randomized_active_neutral_arm",
                "plan": matched.plan.to_record(),
                "assignment": matched.assignment.to_record(),
                "arm_view": matched.arm_view.to_record(),
                "aggregation": {
                    **matched.aggregation.to_record(),
                    "binding_sha256": matched.aggregation.binding_sha256,
                },
                "context_projection_binding_sha256": (
                    context_projection.binding_sha256
                ),
                "card_vs_neutral_effect_identified": False,
                "online_score_update_allowed": False,
            }
        )
    context_projection_record: dict[str, object] = {
        **context_projection.to_record(),
        "binding_sha256": context_projection.binding_sha256,
    }
    selector_context_values = dict(request.context.items)
    selector_context_extension = resolve_campaign_selector_context_extension(
        trusted_context=build.evidence_context,
        selector_context=request.context,
    )
    if selector_context_extension is not None:
        context_projection_record["selector_context_extension"] = {
            "reserved_key": CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY,
            **selector_context_extension.to_binding_record(),
        }
    archive_context = selector_context_values.get(CAMPAIGN_ARCHIVE_CONTEXT_KEY)
    if archive_context is not None:
        if type(archive_context) is not FrozenJsonObject:
            raise TypeError("reserved campaign archive context must be a frozen object")
        context_projection_record["archive_context_projection"] = {
            "reserved_key": CAMPAIGN_ARCHIVE_CONTEXT_KEY,
            "projection_sha256": typed_json_sha256(archive_context),
        }
    frontier_target = selector_context_values.get(CAMPAIGN_FRONTIER_TARGET_KEY)
    if frontier_target is not None:
        if type(frontier_target) is not FrozenJsonObject:
            raise TypeError("reserved campaign frontier target must be frozen")
        if build.frontier_target is None:
            raise ValueError("selector context has an unauthenticated frontier target")
        if thaw_json(frontier_target) != build.frontier_target.to_record():
            raise ValueError("selector frontier target differs from its receipt")
        context_projection_record["frontier_target_projection"] = {
            "reserved_key": CAMPAIGN_FRONTIER_TARGET_KEY,
            "target_sha256": build.frontier_target.target_sha256,
            "projection_sha256": typed_json_sha256(frontier_target),
        }
    context_projection_identity = _object(context_projection_record)
    card_records = tuple(_object(card.to_record()) for card in request.cards)
    card_reference_mapping = tuple(
        _object(
            {
                "card_key": card.card_key,
                "reference": {
                    "insight_id": card.reference.insight_id.value,
                    "version": card.reference.version,
                },
                "source_binding_sha256": (
                    None
                    if card.source_binding is None
                    else card.source_binding.binding_sha256
                ),
                "derived_view_receipt_sha256": (
                    None
                    if card.derived_view_receipt is None
                    else card.derived_view_receipt.receipt_sha256
                ),
            }
        )
        for card in request.cards
    )
    test_eligible_reflections = tuple(
        _object(
            {
                "reflection_receipt_sha256": receipt_sha256,
                "reflection_result_sha256": typed_json_sha256(result),
            }
        )
        for receipt_sha256, result in build.test_eligible_reflections
    )
    evidence_card_snapshot_sha256 = typed_json_sha256(
        _object(
            {
                "schema_version": 1,
                "cards": [thaw_json(value) for value in build.evidence_cards],
            }
        )
    )
    parent_lane = build.parent_lane
    decision_slot = build.decision_slot
    assert parent_lane is not None
    assert decision_slot is not None
    occurrence = build.parent.occurrence
    return CampaignPortfolioWavePreparationReceipt(
        campaign_preparation_sha256=build.prepared.preparation_sha256,
        stage_request_sha256=build.stage_request.request_sha256,
        stage_preparation_sha256=build.stage_request.preparation_sha256,
        generation=wave.generation,
        parent_slot=build.parent_slot,
        parent_lane_id=parent_lane.lane_id,
        decision_slot_id=decision_slot.slot_id,
        parent_candidate_id=build.parent.candidate_id.value,
        parent_configuration_sha256=occurrence.configuration_hash,
        parent_configuration_artifact_sha256=(occurrence.configuration_artifact_hash),
        finite_contract_identity_sha256=(
            request.finite_variation_contract.identity_sha256
        ),
        selector_call_id=request.call_id.value,
        selector_request_sha256=request.request_sha256,
        workload_context_sha256=workload_context_sha256,
        pre_memory_projection_context_sha256=(pre_memory_projection_context_sha256),
        selector_context_sha256=request.context_sha256,
        evidence_card_snapshot_sha256=evidence_card_snapshot_sha256,
        selector_card_snapshot_sha256=request.card_snapshot_sha256,
        card_records=card_records,
        card_reference_mapping=card_reference_mapping,
        test_eligible_reflection_receipts=test_eligible_reflections,
        context_projection_identity=context_projection_identity,
        memory_credit_identity=memory_credit_identity,
    )


@dataclass(frozen=True, slots=True)
class CampaignPortfolioOutcomePreparation:
    """Pure workload-memory projection awaiting a campaign commit barrier."""

    request_sha256: str
    generation: int
    wave_request_sha256s: tuple[str, ...]
    result_receipt_sha256s: tuple[str, ...]
    prior_memory_sha256: str
    updated_memory: FrozenJsonObject
    evidence: FrozenJsonObject
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "request_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        for name in ("wave_request_sha256s", "result_receipt_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a non-empty exact tuple")
            for value in values:
                _require_sha256(value, name)
        if len(self.wave_request_sha256s) != len(self.result_receipt_sha256s):
            raise ValueError("outcome preparation waves and results differ")
        _require_sha256(self.prior_memory_sha256, "prior_memory_sha256")
        for name in ("updated_memory", "evidence"):
            value = getattr(self, name)
            if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
                raise TypeError(f"{name} must be an exact frozen object")
        object.__setattr__(
            self,
            "preparation_sha256",
            _preparation_sha256(
                _OUTCOME_PREPARATION_DOMAIN,
                _object(
                    {
                        "schema_version": 1,
                        "request_sha256": self.request_sha256,
                        "generation": self.generation,
                        "wave_request_sha256s": list(self.wave_request_sha256s),
                        "result_receipt_sha256s": list(self.result_receipt_sha256s),
                        "prior_memory_sha256": self.prior_memory_sha256,
                        "updated_memory_sha256": typed_json_sha256(self.updated_memory),
                        "evidence_sha256": typed_json_sha256(self.evidence),
                    }
                ),
            ),
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "generation": self.generation,
            "wave_request_sha256s": list(self.wave_request_sha256s),
            "result_receipt_sha256s": list(self.result_receipt_sha256s),
            "prior_memory_sha256": self.prior_memory_sha256,
            "updated_memory_sha256": typed_json_sha256(self.updated_memory),
            "evidence": thaw_json(self.evidence),
            "preparation_sha256": self.preparation_sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignPortfolioLearningPreparation:
    """Pure lifecycle adjudication awaiting a campaign commit barrier."""

    request_sha256: str
    generation: int
    wave_request_sha256s: tuple[str, ...]
    result_receipt_sha256s: tuple[str, ...]
    memory_credit_preparation_sha256: str
    evidence: FrozenJsonObject
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "request_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        for name in ("wave_request_sha256s", "result_receipt_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a non-empty exact tuple")
            for value in values:
                _require_sha256(value, name)
        if len(self.wave_request_sha256s) != len(self.result_receipt_sha256s):
            raise ValueError("learning preparation waves and results differ")
        _require_sha256(
            self.memory_credit_preparation_sha256,
            "memory_credit_preparation_sha256",
        )
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "preparation_sha256",
            _preparation_sha256(
                _LEARNING_PREPARATION_DOMAIN,
                _object(
                    {
                        "schema_version": 1,
                        "request_sha256": self.request_sha256,
                        "generation": self.generation,
                        "wave_request_sha256s": list(self.wave_request_sha256s),
                        "result_receipt_sha256s": list(self.result_receipt_sha256s),
                        "memory_credit_preparation_sha256": (
                            self.memory_credit_preparation_sha256
                        ),
                        "evidence_sha256": typed_json_sha256(self.evidence),
                    }
                ),
            ),
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "generation": self.generation,
            "wave_request_sha256s": list(self.wave_request_sha256s),
            "result_receipt_sha256s": list(self.result_receipt_sha256s),
            "memory_credit_preparation_sha256": (self.memory_credit_preparation_sha256),
            "evidence": thaw_json(self.evidence),
            "preparation_sha256": self.preparation_sha256,
        }


@runtime_checkable
class CampaignSelectorRequestPromptRenderer(Protocol):
    """Render the exact prompt sent by an opt-in campaign selector.

    The default campaign path keeps using the stable direct-selector renderer.
    Selectors with a different wire contract must inject an implementation
    backed by the same authenticated pre-call inputs used for provider
    execution; reconstructing a plausible prompt after the call is not audit
    evidence.
    """

    def render(self, request: PortfolioSelectionRequest) -> str: ...


@dataclass(frozen=True, slots=True)
class CampaignIdentifiableReflectionEvidenceQuery:
    """Content-minimized request for one committed mutation-evidence cohort.

    The query deliberately carries the source *receipt identity*, never the
    source-stage payload or recombination results.  Its sealed cutoff is the
    portfolio generation that produced the direct mutations; the later
    recombination generation only supplies the scheduling boundary.
    """

    reflection_request_sha256: str
    preparation_sha256: str
    runtime_start_receipt_sha256: str
    campaign_sha256: str
    workload_instance_sha256: str
    evaluator_contract_sha256: str
    wave: CampaignReflectionWave
    source_stage_receipt_sha256: str
    source_portfolio_generation: int
    prior_cutoff_event_index_exclusive: int
    sealed_cutoff_event_index_inclusive: int
    query_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "reflection_request_sha256",
            "preparation_sha256",
            "runtime_start_receipt_sha256",
            "campaign_sha256",
            "workload_instance_sha256",
            "evaluator_contract_sha256",
            "source_stage_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.wave) is not CampaignReflectionWave:
            raise TypeError("wave must be an exact CampaignReflectionWave")
        CampaignReflectionWave.__post_init__(self.wave)
        if (
            type(self.source_portfolio_generation) is not int
            or self.source_portfolio_generation <= 0
            or self.source_portfolio_generation >= self.wave.source_generation
        ):
            raise ValueError(
                "source portfolio generation must precede the reflection source"
            )
        for name in (
            "prior_cutoff_event_index_exclusive",
            "sealed_cutoff_event_index_inclusive",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.sealed_cutoff_event_index_inclusive != self.source_portfolio_generation:
            raise ValueError(
                "sealed evidence cutoff must equal the source portfolio generation"
            )
        if (
            self.prior_cutoff_event_index_exclusive
            >= self.sealed_cutoff_event_index_inclusive
        ):
            raise ValueError("reflection evidence cutoff must strictly advance")
        object.__setattr__(
            self,
            "query_sha256",
            _preparation_sha256(
                _IDENTIFIABLE_REFLECTION_QUERY_DOMAIN,
                _object(self._unsigned_record()),
            ),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reflection_request_sha256": self.reflection_request_sha256,
            "preparation_sha256": self.preparation_sha256,
            "runtime_start_receipt_sha256": self.runtime_start_receipt_sha256,
            "campaign_sha256": self.campaign_sha256,
            "workload_instance_sha256": self.workload_instance_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "wave": self.wave.to_record(),
            "source_stage_receipt_sha256": self.source_stage_receipt_sha256,
            "source_portfolio_generation": self.source_portfolio_generation,
            "prior_cutoff_event_index_exclusive": (
                self.prior_cutoff_event_index_exclusive
            ),
            "sealed_cutoff_event_index_inclusive": (
                self.sealed_cutoff_event_index_inclusive
            ),
            "source_stage_payload_exposed": False,
            "recombination_results_exposed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "query_sha256": self.query_sha256}


@dataclass(frozen=True, slots=True)
class CampaignIdentifiableReflectionEvidenceProjection:
    """Evidence-source response bound to one exact committed registry cutoff."""

    query_sha256: str
    registry_snapshot_sha256: str
    registry_captured_through_event_index: int
    evidence: IdentifiableReflectionEvidenceSnapshot
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.query_sha256, "query_sha256")
        _require_sha256(self.registry_snapshot_sha256, "registry_snapshot_sha256")
        if (
            type(self.registry_captured_through_event_index) is not int
            or self.registry_captured_through_event_index <= 0
        ):
            raise ValueError("registry_captured_through_event_index must be positive")
        if type(self.evidence) is not IdentifiableReflectionEvidenceSnapshot:
            raise TypeError(
                "evidence must be an exact IdentifiableReflectionEvidenceSnapshot"
            )
        IdentifiableReflectionEvidenceSnapshot.__post_init__(self.evidence)
        if (
            self.evidence.sealed_cutoff_event_index_inclusive
            != self.registry_captured_through_event_index
        ):
            raise ValueError("evidence cutoff differs from the registry snapshot")
        object.__setattr__(
            self,
            "projection_sha256",
            _preparation_sha256(
                _IDENTIFIABLE_REFLECTION_SOURCE_DOMAIN,
                _object(self._unsigned_record()),
            ),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "query_sha256": self.query_sha256,
            "registry_snapshot_sha256": self.registry_snapshot_sha256,
            "registry_captured_through_event_index": (
                self.registry_captured_through_event_index
            ),
            "evidence_snapshot_sha256": self.evidence.snapshot_sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "evidence": self.evidence.to_record(),
            "projection_sha256": self.projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignIdentifiableReflectionInput:
    """Only information an identifiable production reflection may consume."""

    query: CampaignIdentifiableReflectionEvidenceQuery
    source: CampaignIdentifiableReflectionEvidenceProjection
    input_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.query) is not CampaignIdentifiableReflectionEvidenceQuery:
            raise TypeError("query must be exact")
        CampaignIdentifiableReflectionEvidenceQuery.__post_init__(self.query)
        if type(self.source) is not CampaignIdentifiableReflectionEvidenceProjection:
            raise TypeError("source must be exact")
        CampaignIdentifiableReflectionEvidenceProjection.__post_init__(self.source)
        evidence = self.source.evidence
        if self.source.query_sha256 != self.query.query_sha256:
            raise ValueError("evidence source belongs to a foreign reflection query")
        if (
            evidence.prior_cutoff_event_index_exclusive
            != self.query.prior_cutoff_event_index_exclusive
            or evidence.sealed_cutoff_event_index_inclusive
            != self.query.sealed_cutoff_event_index_inclusive
        ):
            raise ValueError("evidence snapshot escapes the requested cutoffs")
        if (
            evidence.campaign_sha256 != self.query.campaign_sha256
            or evidence.workload_instance_sha256 != self.query.workload_instance_sha256
            or evidence.evaluator_contract_sha256
            != self.query.evaluator_contract_sha256
        ):
            raise ValueError("evidence snapshot escapes the requested scope")
        object.__setattr__(
            self,
            "input_sha256",
            _preparation_sha256(
                _IDENTIFIABLE_REFLECTION_INPUT_DOMAIN,
                _object(self._unsigned_record()),
            ),
        )

    @property
    def evidence(self) -> IdentifiableReflectionEvidenceSnapshot:
        return self.source.evidence

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "query_sha256": self.query.query_sha256,
            "source_projection_sha256": self.source.projection_sha256,
            "evidence_snapshot_sha256": self.source.evidence.snapshot_sha256,
            "source_stage_payload_exposed": False,
            "recombination_results_exposed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "query": self.query.to_record(),
            "source": self.source.to_record(),
            "input_sha256": self.input_sha256,
        }


@runtime_checkable
class CampaignIdentifiableReflectionEvidenceSource(Protocol):
    """Project a committed registry cohort without receiving stage payloads."""

    campaign_sha256: str
    workload_instance_sha256: str
    evaluator_contract_sha256: str

    def project(
        self,
        query: CampaignIdentifiableReflectionEvidenceQuery,
    ) -> CampaignIdentifiableReflectionEvidenceProjection: ...


@runtime_checkable
class CampaignReflectionFalsificationSource(Protocol):
    """Return feedback committed and visible at one reflection cutoff."""

    def available(
        self,
        query: CampaignIdentifiableReflectionEvidenceQuery,
    ) -> tuple[ReflectionFalsificationFeedback, ...]: ...


@dataclass(frozen=True, slots=True)
class CommittedRegistryIdentifiableReflectionEvidenceSource:
    """Default adapter over an append-only committed campaign registry."""

    registry: CampaignEvidenceRegistry
    campaign_sha256: str
    workload_instance_sha256: str
    evaluator_contract_sha256: str
    prior_falsifications: tuple[ReflectionFalsificationFeedback, ...] = ()
    falsification_source: CampaignReflectionFalsificationSource | None = None

    def __post_init__(self) -> None:
        if type(self.registry) is not CampaignEvidenceRegistry:
            raise TypeError("registry must be an exact CampaignEvidenceRegistry")
        for name in (
            "campaign_sha256",
            "workload_instance_sha256",
            "evaluator_contract_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.prior_falsifications) is not tuple or any(
            type(value) is not ReflectionFalsificationFeedback
            for value in self.prior_falsifications
        ):
            raise TypeError("prior_falsifications must contain exact feedback")
        for value in self.prior_falsifications:
            ReflectionFalsificationFeedback.__post_init__(value)
        if self.falsification_source is not None and not isinstance(
            self.falsification_source,
            CampaignReflectionFalsificationSource,
        ):
            raise TypeError(
                "falsification_source must implement its narrow runtime port"
            )
        if self.prior_falsifications and self.falsification_source is not None:
            raise ValueError(
                "static and cutoff-aware falsification sources are mutually exclusive"
            )

    def project(
        self,
        query: CampaignIdentifiableReflectionEvidenceQuery,
    ) -> CampaignIdentifiableReflectionEvidenceProjection:
        self.__post_init__()
        if type(query) is not CampaignIdentifiableReflectionEvidenceQuery:
            raise TypeError("query must be exact")
        CampaignIdentifiableReflectionEvidenceQuery.__post_init__(query)
        if (
            query.campaign_sha256 != self.campaign_sha256
            or query.workload_instance_sha256 != self.workload_instance_sha256
            or query.evaluator_contract_sha256 != self.evaluator_contract_sha256
        ):
            raise ValueError("reflection query names a foreign evidence scope")
        cutoff = query.sealed_cutoff_event_index_inclusive
        if self.registry.captured_through_event_index < cutoff:
            raise RuntimeError("committed evidence registry has not reached the cutoff")
        committed = tuple(
            value for value in self.registry.observations if value.event_index <= cutoff
        )
        registry_snapshot = GlobalEvidenceRegistrySnapshot.seal(
            captured_through_event_index=cutoff,
            observations=committed,
        )
        dynamic_source = self.falsification_source
        if dynamic_source is None:
            candidates = self.prior_falsifications
        else:
            candidates = dynamic_source.available(query)
            if type(candidates) is not tuple or any(
                type(value) is not ReflectionFalsificationFeedback
                for value in candidates
            ):
                raise TypeError(
                    "falsification source must return exact feedback values"
                )
        for value in candidates:
            ReflectionFalsificationFeedback.__post_init__(value)
        if tuple(value.feedback_sha256 for value in candidates) != tuple(
            sorted({value.feedback_sha256 for value in candidates})
        ):
            raise ValueError(
                "falsification source must return canonical unique feedback"
            )
        applicable_falsifications = tuple(
            value
            for value in candidates
            if value.available_event_index <= cutoff
            and self.workload_instance_sha256
            in value.applicable_workload_instance_sha256s
            and value.evaluator_contract_sha256 == self.evaluator_contract_sha256
            and (
                not value.applicable_campaign_sha256s
                or self.campaign_sha256 in value.applicable_campaign_sha256s
            )
        )
        if dynamic_source is not None and applicable_falsifications != candidates:
            raise ValueError(
                "cutoff-aware falsification source returned foreign or future feedback"
            )
        evidence = project_identifiable_reflection_evidence(
            registry_snapshot.observations,
            campaign_sha256=self.campaign_sha256,
            workload_instance_sha256=self.workload_instance_sha256,
            evaluator_contract_sha256=self.evaluator_contract_sha256,
            prior_cutoff_event_index_exclusive=(
                query.prior_cutoff_event_index_exclusive
            ),
            sealed_cutoff_event_index_inclusive=cutoff,
            prior_falsifications=applicable_falsifications,
        )
        return CampaignIdentifiableReflectionEvidenceProjection(
            query_sha256=query.query_sha256,
            registry_snapshot_sha256=registry_snapshot.snapshot_sha256,
            registry_captured_through_event_index=cutoff,
            evidence=evidence,
        )


@runtime_checkable
class CampaignReflectionExecutor(Protocol):
    """Execute reflection from sealed direct-mutation evidence only."""

    async def reflect(
        self,
        reflection_input: CampaignIdentifiableReflectionInput,
    ) -> FrozenJsonObject: ...


@runtime_checkable
class CampaignLegacyRecombinationReflectionExecutor(Protocol):
    """Deprecated compatibility port exposing non-identifiable recombinations."""

    async def reflect(
        self,
        request: CampaignReflectionRequest,
        source_results: tuple[PortfolioRecombinationWaveResult, ...],
    ) -> FrozenJsonObject: ...


@runtime_checkable
class CampaignPortfolioOutcomeUpdater(Protocol):
    """Prepare then atomically publish workload-visible outcome state.

    Exact causal credit inside :class:`InsightMemoryBank` remains owned by the
    wave's ``PortfolioMemoryCreditPlan``.  ``prepare_update`` must not publish
    any ledger or workload state.  ``commit_update`` is a synchronous, no-I/O,
    already-validated publication and must not fail for its own preparation.
    """

    async def prepare_update(
        self,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        prior_memory: FrozenJsonObject,
    ) -> CampaignPortfolioOutcomePreparation: ...

    def commit_update(
        self,
        preparation: CampaignPortfolioOutcomePreparation,
    ) -> None: ...

    def abort_update(
        self,
        preparation: CampaignPortfolioOutcomePreparation,
    ) -> None: ...


@runtime_checkable
class CampaignLearningLifecyclePort(Protocol):
    """Join quarantined reflections to later diagnostic campaign barriers.

    This is an observation/lifecycle seam, not a workload or provider API.  It
    may stage exact reflection cards, admit them for controlled testing, and
    adjudicate them after a complete memory-credit barrier.  Every method
    returns frozen evidence for the campaign trace.
    """

    def reflection_completed(
        self,
        request: CampaignReflectionRequest,
        receipt: CampaignReflectionReceipt,
        result: FrozenJsonObject,
    ) -> FrozenJsonObject: ...

    def reflections_admitted(
        self,
        request: CampaignReflectionTestAdmissionRequest,
        contents: tuple[tuple[CampaignReflectionReceipt, FrozenJsonObject], ...],
    ) -> FrozenJsonObject: ...

    async def prepare_portfolio_generation_close(
        self,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        memory_credit_preparation: PortfolioMemoryCreditBatchPreparation,
    ) -> CampaignPortfolioLearningPreparation: ...

    def commit_portfolio_generation_close(
        self,
        preparation: CampaignPortfolioLearningPreparation,
    ) -> None: ...

    def abort_portfolio_generation_close(
        self,
        preparation: CampaignPortfolioLearningPreparation,
    ) -> None: ...


@runtime_checkable
class CampaignRecombinationUtilityBinder(Protocol):
    """Bind exact scored-source/pair utility under the generation cutoff.

    ``source_result.candidates`` is the ranked ITT population;
    ``source_result.scored_candidates`` is the only utility/parent universe.
    """

    def bind(
        self,
        *,
        source_archive_utility: ArchiveUtilitySnapshot,
        source_wave: PortfolioVariationWaveRequest,
        source_result: PortfolioVariationWaveResult,
    ) -> FrozenArchiveSourceUtilityReceipt: ...


@runtime_checkable
class CampaignOwnedRuntimeResourcePort(Protocol):
    """Close a provider queue or other resource explicitly owned by the adapter."""

    async def close(self) -> FrozenJsonObject: ...


@dataclass(slots=True)
class AgenticPortfolioCampaignRuntime:
    """Run a prepared campaign through the real AgentEvolve application stack."""

    prepared: PreparedEvolutionCampaign
    workload_config: AgenticCampaignWorkloadConfig
    workload_ports: CampaignWorkloadPorts
    composition: PortfolioEvolutionComposition
    parent_selector: CampaignParentSelectionPort
    wave_factory: CampaignPortfolioWaveFactory
    task_sha256: str
    parent_measurement_projection: ParentMeasurementProjection | None = None
    archive_context_projector: CampaignPortfolioArchiveContextProjector | None = None
    context_enricher: CampaignPortfolioContextEnricher | None = None
    contextual_search_planner: CampaignContextualSearchPlanner | None = None
    frontier_target_allocator: CampaignPortfolioFrontierTargetAllocator | None = None
    memory_estimand_projector: CampaignPortfolioMemoryEstimandProjector | None = None
    learning_lifecycle: CampaignLearningLifecyclePort | None = None
    identifiable_reflection_executor: CampaignReflectionExecutor | None = None
    identifiable_reflection_evidence_source: (
        CampaignIdentifiableReflectionEvidenceSource | None
    ) = None
    legacy_recombination_reflection_executor: (
        CampaignLegacyRecombinationReflectionExecutor | None
    ) = None
    # Temporary source-compatible alias.  It is always interpreted as the
    # legacy recombination port and cannot be combined with the identifiable
    # pair or the explicitly named legacy field.
    reflection_executor: CampaignLegacyRecombinationReflectionExecutor | None = None
    outcome_updater: CampaignPortfolioOutcomeUpdater | None = None
    recombination_utility_binder: CampaignRecombinationUtilityBinder | None = None
    owned_resources: CampaignOwnedRuntimeResourcePort | None = None
    recombination: PortfolioRecombination | None = None
    selector_request_prompt_renderer: CampaignSelectorRequestPromptRenderer | None = (
        None
    )
    wave_preparation_observer: CampaignPortfolioWavePreparationObserver | None = None
    archive: ParetoArchive = field(init=False)
    _history: list[EvolutionCandidate] = field(init=False, default_factory=list)
    _memory: FrozenJsonObject | None = field(init=False, default=None)
    _portfolio_waves: dict[
        int,
        tuple[tuple[PortfolioVariationWaveRequest, PortfolioVariationWaveResult], ...],
    ] = field(init=False, default_factory=dict)
    _recombination_results: dict[int, tuple[PortfolioRecombinationWaveResult, ...]] = (
        field(init=False, default_factory=dict)
    )
    _recombination_source_portfolio_generations: dict[int, int] = field(
        init=False,
        default_factory=dict,
    )
    _stage_receipts: dict[int, CampaignStageReceipt] = field(
        init=False, default_factory=dict
    )
    _parent_selection_progress: list[CampaignParentSelectionProgress] = field(
        init=False,
        default_factory=list,
    )
    _pending_parent_selection_progress: (
        _PendingCampaignParentSelectionProgress | None
    ) = field(init=False, default=None)
    _archive_utilities: dict[int, ArchiveUtilitySnapshot] = field(
        init=False, default_factory=dict
    )
    _reflection_results: dict[str, FrozenJsonObject] = field(
        init=False, default_factory=dict
    )
    _reflection_learning_evidence: dict[str, FrozenJsonObject] = field(
        init=False,
        default_factory=dict,
    )
    _identifiable_reflection_inputs: dict[
        str,
        CampaignIdentifiableReflectionInput,
    ] = field(init=False, default_factory=dict)
    _consumed_reflection_source_evidence_ids: set[str] = field(
        init=False,
        default_factory=set,
    )
    _last_identifiable_reflection_cutoff: int = field(init=False, default=0)
    _test_eligible_reflection_hashes: set[str] = field(init=False, default_factory=set)
    _started: bool = field(init=False, default=False)
    _cleaned: bool = field(init=False, default=False)
    _selector_calls: int = field(init=False, default=0)
    _wave_preparation_receipts: list[CampaignPortfolioWavePreparationReceipt] = field(
        init=False, default_factory=list
    )
    _phenotype_identity_cache: dict[str, str] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if type(self.prepared) is not PreparedEvolutionCampaign:
            raise TypeError("prepared must be an exact PreparedEvolutionCampaign")
        PreparedEvolutionCampaign.__post_init__(self.prepared)
        if type(self.workload_config) is not AgenticCampaignWorkloadConfig:
            raise TypeError("workload_config must be exact")
        AgenticCampaignWorkloadConfig.__post_init__(self.workload_config)
        if type(self.composition) is not PortfolioEvolutionComposition:
            raise TypeError("composition must be exact")
        PortfolioEvolutionComposition.__post_init__(self.composition)
        if self.composition.benchmark is not self.workload_config.benchmark:
            raise ValueError("composition is bound to a different benchmark object")
        if not isinstance(self.parent_selector, CampaignParentSelectionPort):
            raise TypeError(
                "parent_selector must implement CampaignParentSelectionPort"
            )
        if not isinstance(self.wave_factory, CampaignPortfolioWaveFactory):
            raise TypeError("wave_factory must implement CampaignPortfolioWaveFactory")
        if self.archive_context_projector is not None and not isinstance(
            self.archive_context_projector,
            CampaignPortfolioArchiveContextProjector,
        ):
            raise TypeError("archive_context_projector must implement its runtime port")
        if self.context_enricher is not None and not isinstance(
            self.context_enricher,
            CampaignPortfolioContextEnricher,
        ):
            raise TypeError("context_enricher must implement its runtime port")
        if (
            self.contextual_search_planner is not None
            and type(self.contextual_search_planner)
            is not CampaignContextualSearchPlanner
        ):
            raise TypeError("contextual_search_planner must be exact or None")
        if self.frontier_target_allocator is not None and not isinstance(
            self.frontier_target_allocator,
            CampaignPortfolioFrontierTargetAllocator,
        ):
            raise TypeError("frontier_target_allocator must implement its runtime port")
        if (
            self.contextual_search_planner is not None
            and self.frontier_target_allocator is not None
        ):
            raise ValueError(
                "contextual and standalone frontier target allocation are mutually "
                "exclusive"
            )
        if self.memory_estimand_projector is not None and not isinstance(
            self.memory_estimand_projector,
            CampaignPortfolioMemoryEstimandProjector,
        ):
            raise TypeError("memory_estimand_projector must implement its runtime port")
        if self.wave_preparation_observer is not None and not isinstance(
            self.wave_preparation_observer,
            CampaignPortfolioWavePreparationObserver,
        ):
            raise TypeError("wave_preparation_observer must implement its runtime port")
        if self.learning_lifecycle is not None and not isinstance(
            self.learning_lifecycle,
            CampaignLearningLifecyclePort,
        ):
            raise TypeError("learning_lifecycle must implement its runtime port")
        if self.selector_request_prompt_renderer is not None and not isinstance(
            self.selector_request_prompt_renderer,
            CampaignSelectorRequestPromptRenderer,
        ):
            raise TypeError(
                "selector_request_prompt_renderer must implement its runtime port"
            )
        identifiable_executor = self.identifiable_reflection_executor
        evidence_source = self.identifiable_reflection_evidence_source
        explicitly_legacy = self.legacy_recombination_reflection_executor
        compatibility_legacy = self.reflection_executor
        if identifiable_executor is not None and not isinstance(
            identifiable_executor,
            CampaignReflectionExecutor,
        ):
            raise TypeError(
                "identifiable_reflection_executor must implement its runtime port"
            )
        if evidence_source is not None and not isinstance(
            evidence_source,
            CampaignIdentifiableReflectionEvidenceSource,
        ):
            raise TypeError(
                "identifiable_reflection_evidence_source must implement its runtime port"
            )
        if evidence_source is not None:
            for name in (
                "campaign_sha256",
                "workload_instance_sha256",
                "evaluator_contract_sha256",
            ):
                _require_sha256(
                    getattr(evidence_source, name),
                    f"identifiable_reflection_evidence_source.{name}",
                )
        if (identifiable_executor is None) != (evidence_source is None):
            raise ValueError(
                "identifiable reflection requires both executor and evidence source"
            )
        if explicitly_legacy is not None and not isinstance(
            explicitly_legacy,
            CampaignLegacyRecombinationReflectionExecutor,
        ):
            raise TypeError(
                "legacy_recombination_reflection_executor must implement its runtime port"
            )
        if compatibility_legacy is not None and not isinstance(
            compatibility_legacy,
            CampaignLegacyRecombinationReflectionExecutor,
        ):
            raise TypeError(
                "reflection_executor compatibility alias must implement the legacy port"
            )
        if explicitly_legacy is not None and compatibility_legacy is not None:
            raise ValueError("legacy reflection executor was configured twice")
        if identifiable_executor is not None and (
            explicitly_legacy is not None or compatibility_legacy is not None
        ):
            raise ValueError(
                "identifiable and legacy reflection modes are mutually exclusive"
            )
        if self.outcome_updater is not None and not isinstance(
            self.outcome_updater, CampaignPortfolioOutcomeUpdater
        ):
            raise TypeError("outcome_updater must implement its runtime port")
        if self.recombination_utility_binder is not None and not isinstance(
            self.recombination_utility_binder,
            CampaignRecombinationUtilityBinder,
        ):
            raise TypeError(
                "recombination_utility_binder must implement its runtime port"
            )
        if self.owned_resources is not None and not isinstance(
            self.owned_resources, CampaignOwnedRuntimeResourcePort
        ):
            raise TypeError("owned_resources must implement its runtime port")
        if (
            type(self.task_sha256) is not str
            or len(self.task_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.task_sha256)
        ):
            raise ValueError("task_sha256 must be a lowercase SHA-256 digest")
        if type(self.workload_ports) is not CampaignWorkloadPorts:
            raise TypeError("workload_ports must be exact")
        CampaignWorkloadPorts.__post_init__(self.workload_ports)
        if self.workload_ports.ports_sha256 != self.prepared.workload_ports_sha256:
            raise ValueError("runtime workload ports differ from preparation")
        if (
            self.prepared.benchmark_session.benchmark
            != self.workload_config.benchmark_record
        ):
            raise ValueError("prepared session describes a different benchmark")
        measurement_projection = self.parent_measurement_projection
        if measurement_projection is not None:
            if type(measurement_projection) is not ParentMeasurementProjection:
                raise TypeError("parent_measurement_projection must be exact or None")
            measurement_projection.__post_init__()
            benchmark_sha256 = typed_json_sha256(
                self.prepared.benchmark_session.benchmark
            )
            if (
                measurement_projection.benchmark_sha256 != benchmark_sha256
                or measurement_projection.session_sha256
                != self.prepared.benchmark_session.session_sha256
            ):
                raise ValueError(
                    "parent measurement projection names a foreign campaign"
                )
            benchmark = self.workload_config.benchmark
            evaluator = benchmark.detailed_evaluator
            if (
                evaluator is None
                or (
                    evaluator.evaluator_identity.evaluator_id,
                    evaluator.evaluator_identity.evaluator_version,
                    evaluator.evaluator_identity.evaluator_context_sha256,
                )
                != measurement_projection.evaluator_identity
            ):
                raise ValueError(
                    "parent measurement projection differs from benchmark evaluator"
                )
            expected_resolution = (
                (
                    EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
                    EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
                    EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
                )
                if benchmark.objective_resolution is None
                else objective_resolution_policy_metadata(
                    benchmark.objective_resolution
                )
            )
            if (
                measurement_projection.objective_resolution_identity
                != expected_resolution
            ):
                raise ValueError(
                    "parent measurement projection differs from objective resolution"
                )
            semantics = benchmark.optimization_semantics
            if semantics is not None and (
                measurement_projection.decision_metrics.optimization_semantics_definition_sha256
                != semantics.definition_sha256
            ):
                raise ValueError(
                    "parent measurement metric schema differs from benchmark semantics"
                )
        self.archive = ParetoArchive(
            self.composition.benchmark.objectives,
            outcome_relation_binding=self.composition.outcome_relation,
        )
        if self.recombination is None:
            self.recombination = PortfolioRecombination(
                engine=self.composition.engine,
                ids=self.composition.id_factory,
                selection_limit=(self.prepared.protocol.recombinations_per_parent),
            )
        elif type(self.recombination) is not PortfolioRecombination:
            raise TypeError("recombination must be exact or None")
        if (
            self.recombination.engine is not self.composition.engine
            or self.recombination.ids is not self.composition.id_factory
            or self.recombination.selection_limit
            != self.prepared.protocol.recombinations_per_parent
        ):
            raise ValueError(
                "recombination is bound to a different composition or protocol"
            )

    @property
    def history(self) -> tuple[EvolutionCandidate, ...]:
        return tuple(self._history)

    @property
    def final_front(self) -> tuple[EvolutionCandidate, ...]:
        return self.archive.front

    @property
    def wave_preparation_receipts(
        self,
    ) -> tuple[CampaignPortfolioWavePreparationReceipt, ...]:
        """Validated prospective waves, including waves from failed stages."""

        return tuple(self._wave_preparation_receipts)

    async def start(
        self,
        prepared: PreparedEvolutionCampaign,
    ) -> CampaignExecutionStartReceipt:
        if self._started:
            raise RuntimeError("campaign runtime is one-shot")
        if prepared.preparation_sha256 != self.prepared.preparation_sha256:
            raise ValueError("lifecycle start received a foreign preparation")
        cache = await self.composition.engine.evaluation_cache_snapshot()
        if _cache_misses(cache) != 0 or int(cache.get("hits") or 0) != 0:
            raise RuntimeError("campaign runtime requires a fresh evaluation cache")

        candidates: list[EvolutionCandidate] = []
        receipts: list[CampaignSeedExecutionReceipt] = []
        for seed in prepared.seeds.seeds:
            before = _cache_misses(
                await self.composition.engine.evaluation_cache_snapshot()
            )
            configuration = thaw_json(seed.configuration)
            if type(configuration) is not dict:
                raise TypeError("campaign seed root must be an object")
            candidate = await self.composition.engine.register_seed(
                configuration,
                label=f"campaign_seed_{seed.seed_id}",
            )
            after = _cache_misses(
                await self.composition.engine.evaluation_cache_snapshot()
            )
            if after - before not in {0, 1}:
                raise RuntimeError("one seed produced invalid cache accounting")
            candidates.append(candidate)
            receipts.append(
                CampaignSeedExecutionReceipt(
                    seed_id=seed.seed_id,
                    configuration_sha256=seed.configuration_sha256,
                    evaluated=True,
                    unique_evaluation=after > before,
                    valid=candidate.valid,
                    failure_type=None if candidate.valid else "benchmark_invalid",
                    evidence=_object({"candidate": _candidate_record(candidate)}),
                )
            )

        self._memory = self.workload_ports.evidence.initialize_memory(
            self.prepared.benchmark_session,
            self.prepared.seeds,
        )
        for candidate in candidates:
            self._history.append(candidate)
            self.archive.consider(candidate)
        self._started = True
        return CampaignExecutionStartReceipt(
            preparation_sha256=prepared.preparation_sha256,
            runtime_preflight_receipt_sha256=(prepared.runtime_receipt.receipt_sha256),
            runtime_session_id=(f"campaign_runtime_{prepared.preparation_sha256[:24]}"),
            seed_batch_sha256=prepared.seeds.batch_sha256,
            seed_receipts=tuple(receipts),
            evidence=_object(
                {
                    "runtime": "agentic_portfolio_campaign",
                    "provider_calls_during_start": 0,
                    "archive_snapshot_sha256": pareto_archive_snapshot_hash(
                        self.archive.snapshot()
                    ),
                    "memory_sha256": typed_json_sha256(self._memory),
                }
            ),
        )

    def _require_active(self) -> None:
        if not self._started or self._cleaned:
            raise RuntimeError("campaign runtime is not active")

    def _archive_record(self) -> FrozenJsonObject:
        return self._archive_record_for(self.archive)

    @staticmethod
    def _archive_record_for(archive: ParetoArchive) -> FrozenJsonObject:
        if type(archive) is not ParetoArchive:
            raise TypeError("archive must be an exact ParetoArchive")
        snapshot = archive.snapshot()
        return _object(
            {
                "snapshot_sha256": pareto_archive_snapshot_hash(snapshot),
                "summary": snapshot.to_trace_record(),
                "front_candidates": [
                    _candidate_record(candidate)
                    for candidate in snapshot.front_candidates
                ],
            }
        )

    def _preview_archive(
        self,
        candidates: tuple[EvolutionCandidate, ...],
    ) -> ParetoArchive:
        """Replay then extend the archive without changing live campaign state."""

        current = self.archive.snapshot()
        preview = ParetoArchive(
            current.objectives,
            evidence_admission_policy=current.evidence_admission_policy,
            outcome_relation_binding=self.archive.outcome_relation_binding,
        )
        for candidate in self._history:
            preview.consider(candidate)
        if pareto_archive_snapshot_hash(preview.snapshot()) != (
            pareto_archive_snapshot_hash(current)
        ):
            raise RuntimeError("campaign history cannot reproduce the live archive")
        for candidate in candidates:
            preview.consider(candidate)
        return preview

    async def snapshot_archive(
        self,
        request: CampaignArchiveCutoffRequest,
    ) -> CampaignArchiveCutoffReceipt:
        self._require_active()
        expected_prior = self._stage_receipts.get(request.generation - 1)
        if request.prior_stage_receipt_sha256 != (
            None if expected_prior is None else expected_prior.receipt_sha256
        ):
            raise ValueError("archive request is not chained to the runtime stage")
        archive = self._archive_record()
        return CampaignArchiveCutoffReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            generation=request.generation,
            archive=archive,
            evidence=_object(
                {
                    "cutoff_before_generation": request.generation,
                    "archive_sha256": typed_json_sha256(archive),
                }
            ),
        )

    async def _optimizer_state_async(self, generation: int) -> OptimizerState:
        snapshot = self.archive.snapshot()
        cache = await self.composition.engine.evaluation_cache_snapshot()
        return OptimizerState(
            generation=generation,
            candidates=tuple(self._history),
            archive=snapshot,
            archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
            unique_evaluations=_cache_misses(cache),
            logical_llm_calls=self._selector_calls,
        )

    def _known_phenotypes(self) -> tuple[str, ...]:
        policy = self.composition.benchmark.phenotype_identity
        expected_policy = (policy.policy_id, policy.policy_version)
        known: set[str] = set()
        for candidate in self._history:
            configuration_sha256 = candidate.occurrence.configuration_hash
            value_sha256 = self._phenotype_identity_cache.get(configuration_sha256)
            if value_sha256 is None:
                # Engine candidates retain an immutable typed-JSON snapshot, while
                # benchmark phenotype policies are defined over ordinary detached
                # candidate values.  Keep that boundary identical to catalog and
                # engine evaluation calls; policies must never need to understand
                # AgentEvolve's internal FrozenJson containers.
                identity = policy.identify(thaw_json(candidate.configuration))
                if type(identity) is not PhenotypeIdentity:
                    raise TypeError(
                        "phenotype policy must return exact PhenotypeIdentity"
                    )
                PhenotypeIdentity.__post_init__(identity)
                if (identity.policy_id, identity.policy_version) != expected_policy:
                    raise ValueError("phenotype policy returned a foreign identity law")
                value_sha256 = identity.value_sha256
                self._phenotype_identity_cache[configuration_sha256] = value_sha256
            known.add(value_sha256)
        return tuple(sorted(known))

    def _validate_wave(
        self,
        *,
        wave: PortfolioVariationWaveRequest,
        build: CampaignPortfolioWaveContext,
    ) -> None:
        if type(wave) is not PortfolioVariationWaveRequest:
            raise TypeError("wave factory must return PortfolioVariationWaveRequest")
        PortfolioVariationWaveRequest.__post_init__(wave)
        if (
            wave.parent is not build.parent
            or wave.generation != build.stage_request.step.generation
            or wave.selection_request.finite_variation_contract
            != build.variation.contract
            or wave.selection_request.portfolio_size
            != build.stage_request.step.offspring_per_parent
        ):
            raise ValueError("wave factory escaped its trusted campaign inputs")
        try:
            resolve_campaign_selector_context_extension(
                trusted_context=build.evidence_context,
                selector_context=wave.selection_request.context,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "wave factory escaped its trusted campaign inputs"
            ) from error

    def _project_memory_estimand(
        self,
        build: CampaignPortfolioWaveContext,
    ) -> CampaignPortfolioWaveContext:
        """Install the sole dynamic core-owned selector-context projection."""

        projector = self.memory_estimand_projector
        if projector is None:
            return build
        projection = projector.project(build)
        if projection is None:
            return build
        if type(projection) is not CampaignPortfolioMemoryEstimandProjection:
            raise TypeError(
                "memory estimand projector must return an exact projection or None"
            )
        CampaignPortfolioMemoryEstimandProjection.__post_init__(projection)
        projected_context = thaw_json(build.evidence_context)
        if type(projected_context) is not dict:  # pragma: no cover - closed root.
            raise AssertionError("portfolio context did not thaw to an object")
        projected_context[MEMORY_ESTIMAND_CONTEXT_KEY] = thaw_json(
            projection.estimand_context
        )
        projected_context[MEMORY_ESTIMAND_STRATUM_SHA256_KEY] = (
            projection.estimand_stratum_sha256
        )
        return replace(
            build,
            evidence_context=_object(projected_context),
        )

    @staticmethod
    def _selector_audit(
        *,
        generation: int,
        parent_slot: int,
        wave: PortfolioVariationWaveRequest,
        result: PortfolioVariationWaveResult,
        prior_audit_set_sha256: str,
        prompt_renderer: CampaignSelectorRequestPromptRenderer | None = None,
    ) -> CampaignSelectorAuditReceipt:
        decision = result.receipt
        response = result.selection_decision_audit_record
        if response is None:
            raise RuntimeError("fresh portfolio result omitted decision audit")
        request_text = (
            render_portfolio_selection_prompt(wave.selection_request)
            if prompt_renderer is None
            else prompt_renderer.render(wave.selection_request)
        )
        if type(request_text) is not str or not request_text:
            raise TypeError("selector prompt renderer must return non-empty text")
        plaintext = _object(
            {
                "selector_call_id": wave.selection_request.call_id.value,
                "request_sha256": wave.selection_request.request_sha256,
                "decision_sha256": decision.decision_sha256,
                "request_text": request_text,
                "response_text": _canonical_text(response),
                "request_text_kind": "exact_framework_prompt",
                "response_text_kind": "trusted_structured_decision_projection",
            }
        )
        return CampaignSelectorAuditReceipt(
            generation=generation,
            parent_slot=parent_slot,
            selector_call_id=wave.selection_request.call_id.value,
            request_sha256=wave.selection_request.request_sha256,
            decision_sha256=decision.decision_sha256,
            trace_receipt_sha256=typed_json_sha256(plaintext),
            plaintext_audit=plaintext,
            prior_audit_set_sha256=prior_audit_set_sha256,
            execution_mode=SelectorAuditExecutionMode.FRESH,
        )

    async def _portfolio_stage(
        self,
        request: CampaignStageRequest,
        *,
        cache_before: int,
    ) -> CampaignStageReceipt:
        step = request.step
        state = await self._optimizer_state_async(step.generation - 1)
        selection = self.parent_selector.select(
            state,
            task_sha256=self.task_sha256,
            parent_count=step.parent_count,
            rotation_index=(step.generation - 1) // 2,
            progress=tuple(self._parent_selection_progress),
            archive_utility=request.archive_utility,
        )
        if type(selection) is not CampaignParentSelection:
            raise TypeError("parent selector must return CampaignParentSelection")
        CampaignParentSelection.__post_init__(selection)
        if (
            len(selection.parents) != step.parent_count
            or len(selection.lanes) != step.parent_count
            or len(selection.decision_slots) != step.parent_count
        ):
            raise ValueError("parent selector did not satisfy the prepared stage")
        assert self._memory is not None
        if any(
            value not in self._test_eligible_reflection_hashes
            for value in request.test_eligible_reflection_receipt_sha256s
        ):
            raise RuntimeError(
                "portfolio stage names reflection evidence not admitted for testing"
            )
        known = self._known_phenotypes()
        builds: list[CampaignPortfolioWaveContext] = []
        build_hashes: list[tuple[str, str]] = []
        wave_preparations: list[CampaignPortfolioWavePreparationReceipt] = []
        lanes_by_id = {lane.lane_id: lane for lane in selection.lanes}
        for parent_slot, decision_slot in enumerate(selection.decision_slots):
            parent_lane = lanes_by_id[decision_slot.lane_id]
            parent = parent_lane.parent
            variation = self.workload_ports.catalog.bind(
                self.prepared.benchmark_session.benchmark,
                parent.configuration,
                known,
            )
            evidence_context = self.workload_ports.evidence.context(
                self.prepared.benchmark_session,
                parent.configuration,
                variation,
                self._memory,
            )
            workload_context_sha256 = typed_json_sha256(evidence_context)
            parent_measurement = (
                None
                if self.parent_measurement_projection is None
                else bind_parent_measurement(
                    candidate=parent,
                    variation=variation,
                    projection=self.parent_measurement_projection,
                )
            )
            evidence_context = attach_parent_measurement_to_context(
                evidence_context,
                parent_measurement,
            )
            portfolio_generations = tuple(
                step.generation
                for step in self.prepared.schedule.steps
                if step.kind is CampaignGenerationKind.PORTFOLIO
            )
            evidence_context = attach_campaign_search_phase_context(
                evidence_context,
                campaign_search_phase_context(
                    campaign_generation=request.step.generation,
                    portfolio_generations=portfolio_generations,
                ),
            )
            evidence_cards = self.workload_ports.evidence.cards(
                self.prepared.benchmark_session,
                parent.configuration,
                variation,
                self._memory,
            )
            build = CampaignPortfolioWaveContext(
                prepared=self.prepared,
                stage_request=request,
                parent_slot=parent_slot,
                parent=parent,
                variation=variation,
                evidence_context=evidence_context,
                evidence_cards=evidence_cards,
                memory=self._memory,
                parent_measurement=parent_measurement,
                test_eligible_reflections=tuple(
                    (
                        receipt_sha256,
                        self._reflection_results[receipt_sha256],
                    )
                    for receipt_sha256 in (
                        request.test_eligible_reflection_receipt_sha256s
                    )
                ),
                parent_lane=parent_lane,
                decision_slot=decision_slot,
            )
            archive_projector = self.archive_context_projector
            if archive_projector is not None:
                projection = archive_projector.project(
                    archive_utility=request.archive_utility,
                    parent=parent,
                )
                if type(projection) is not (CampaignPortfolioArchiveContextProjection):
                    raise TypeError(
                        "archive context projector must return an exact projection"
                    )
                CampaignPortfolioArchiveContextProjection.__post_init__(projection)
                if (
                    projection.projector_id != archive_projector.projector_id
                    or projection.projector_version
                    != archive_projector.projector_version
                    or projection.definition_sha256
                    != archive_projector.definition_sha256
                    or projection.archive_utility_snapshot_sha256
                    != request.archive_utility.snapshot_sha256
                    or projection.parent_configuration_sha256
                    != parent.occurrence.configuration_hash
                ):
                    raise ValueError(
                        "archive context projection differs from its trusted inputs"
                    )
                base_context = thaw_json(build.evidence_context)
                if type(base_context) is not dict:  # pragma: no cover - closed root.
                    raise AssertionError("portfolio context did not thaw to an object")
                if CAMPAIGN_ARCHIVE_CONTEXT_KEY in base_context:
                    raise ValueError(
                        "base context uses the reserved campaign-archive key"
                    )
                base_context[CAMPAIGN_ARCHIVE_CONTEXT_KEY] = projection.to_record()
                build = replace(
                    build,
                    evidence_context=_object(base_context),
                    archive_context=projection,
                )
            enricher = self.context_enricher
            if enricher is not None:
                contextual_history = enricher.enrich(build)
                if (
                    type(contextual_history) is not FrozenJsonObject
                    or freeze_json(contextual_history) is not contextual_history
                ):
                    raise TypeError("context enricher must return a frozen object")
                base_context = thaw_json(build.evidence_context)
                if type(base_context) is not dict:  # pragma: no cover - closed root.
                    raise AssertionError("portfolio context did not thaw to an object")
                if CAMPAIGN_CONTEXTUAL_HISTORY_KEY in base_context:
                    raise ValueError(
                        "base context uses the reserved contextual-history key"
                    )
                base_context[CAMPAIGN_CONTEXTUAL_HISTORY_KEY] = thaw_json(
                    contextual_history
                )
                enriched_context = _object(base_context)
                build = replace(build, evidence_context=enriched_context)
            pre_memory_projection_context_sha256 = typed_json_sha256(
                build.evidence_context
            )
            build = self._project_memory_estimand(build)
            builds.append(build)
            build_hashes.append(
                (
                    workload_context_sha256,
                    pre_memory_projection_context_sha256,
                )
            )

        planner = self.contextual_search_planner
        contextual_search_plan = None
        standalone_frontier_targets: tuple[
            CampaignPortfolioFrontierTarget, ...
        ] | None = None
        if planner is not None:
            contextual_search_plan = planner.plan(tuple(builds))
            contracts = {
                value.slice_id: value for value in contextual_search_plan.contracts
            }
            frontier_targets = {
                value.lane_id: value
                for value in contextual_search_plan.frontier_targets
            }
            lane_ids = tuple(value.parent_lane.lane_id for value in builds)
            if set(contracts) != set(lane_ids) or set(frontier_targets) != set(
                lane_ids
            ):
                raise ValueError(
                    "contextual plan differs from the campaign parent lanes"
                )
            targeted_builds = []
            for value in builds:
                lane_id = value.parent_lane.lane_id
                target = frontier_targets[lane_id]
                evidence = thaw_json(value.evidence_context)
                if type(evidence) is not dict:
                    raise TypeError("campaign evidence context must be an object")
                if CAMPAIGN_FRONTIER_TARGET_KEY in evidence:
                    raise ValueError(
                        "base context uses the reserved campaign frontier-target key"
                    )
                evidence[CAMPAIGN_FRONTIER_TARGET_KEY] = target.to_record()
                targeted_builds.append(
                    replace(
                        value,
                        evidence_context=_object(evidence),
                        contextual_allocation=contracts[lane_id],
                        frontier_target=target,
                    )
                )
            builds = targeted_builds
        elif self.frontier_target_allocator is not None:
            allocator = self.frontier_target_allocator
            lanes = tuple(
                sorted(
                    (
                        (value.parent_lane.lane_id, value.parent)
                        for value in builds
                    ),
                    key=lambda value: value[0],
                )
            )
            standalone_frontier_targets = allocator.allocate(
                archive_utility=request.archive_utility,
                lanes=lanes,
            )
            if type(standalone_frontier_targets) is not tuple or any(
                type(value) is not CampaignPortfolioFrontierTarget
                for value in standalone_frontier_targets
            ):
                raise TypeError(
                    "frontier target allocator must return exact target receipts"
                )
            by_lane = {
                value.lane_id: value for value in standalone_frontier_targets
            }
            lane_ids = tuple(value[0] for value in lanes)
            if tuple(sorted(by_lane)) != lane_ids:
                raise ValueError(
                    "standalone frontier targets differ from the campaign lanes"
                )
            targeted_builds = []
            for value in builds:
                lane_id = value.parent_lane.lane_id
                target = by_lane[lane_id]
                target.__post_init__()
                evidence = thaw_json(value.evidence_context)
                if type(evidence) is not dict:
                    raise TypeError("campaign evidence context must be an object")
                if CAMPAIGN_FRONTIER_TARGET_KEY in evidence:
                    raise ValueError(
                        "base context uses the reserved campaign frontier-target key"
                    )
                evidence[CAMPAIGN_FRONTIER_TARGET_KEY] = target.to_record()
                targeted_builds.append(
                    replace(
                        value,
                        evidence_context=_object(evidence),
                        frontier_target=target,
                    )
                )
            builds = targeted_builds

        factory = self.wave_factory
        if isinstance(factory, CampaignPortfolioWaveBatchFactory):
            built_waves = factory.build_batch(tuple(builds))
            if type(built_waves) is not tuple:
                raise TypeError("batch wave factory must return an exact tuple")
            if len(built_waves) != len(builds):
                raise ValueError("batch wave factory must preserve lane cardinality")
        else:
            built_waves = tuple(factory.build(build) for build in builds)
        if any(type(wave) is not PortfolioVariationWaveRequest for wave in built_waves):
            raise TypeError("wave factory returned a foreign wave request")

        waves: list[PortfolioVariationWaveRequest] = []
        for build, wave, hashes in zip(
            builds,
            built_waves,
            build_hashes,
            strict=True,
        ):
            workload_context_sha256, pre_memory_projection_context_sha256 = hashes
            self._validate_wave(wave=wave, build=build)
            waves.append(wave)
            wave_preparations.append(
                _validated_wave_preparation_receipt(
                    build=build,
                    wave=wave,
                    workload_context_sha256=workload_context_sha256,
                    pre_memory_projection_context_sha256=(
                        pre_memory_projection_context_sha256
                    ),
                )
            )
        if len({wave.selection_request.call_id for wave in waves}) != len(waves):
            raise ValueError("portfolio stage reused a selector call ID")

        # Preparation evidence is prospective audit state, not a sealed stage
        # result.  Retain it even when provider execution or a later transaction
        # barrier fails.  An injected observer may durably journal each receipt;
        # any observer failure aborts before gather_concurrent_stage can dispatch.
        self._wave_preparation_receipts.extend(wave_preparations)
        observer = self.wave_preparation_observer
        if observer is not None:
            for receipt in wave_preparations:
                observer.record_prepared_wave(receipt)

        pending_results = await gather_concurrent_stage(
            self.composition.portfolio.run(
                wave,
                defer_memory_credit=True,
            )
            for wave in waves
        )
        if any(
            type(value) is not PortfolioVariationWaveResult for value in pending_results
        ):
            raise TypeError("portfolio service returned a foreign result")
        audits = tuple(
            self._selector_audit(
                generation=step.generation,
                parent_slot=slot,
                wave=wave,
                result=result,
                prior_audit_set_sha256=request.prior_selector_audit_set_sha256,
                prompt_renderer=self.selector_request_prompt_renderer,
            )
            for slot, (wave, result) in enumerate(
                zip(waves, pending_results, strict=True)
            )
        )
        memory_credit_preparation = (
            self.composition.portfolio.prepare_pending_memory_credit_batch(
                pending_results
            )
        )
        results = memory_credit_preparation.prepared_results
        memory_credit_batch = memory_credit_preparation.batch_receipt
        candidates = tuple(
            candidate for result in results for candidate in result.candidates
        )
        scored_candidate_count = sum(
            len(result.scored_candidates) for result in results
        )
        infeasible_candidate_count = sum(
            len(result.infeasible_candidates) for result in results
        )
        if scored_candidate_count + infeasible_candidate_count != len(candidates):
            raise RuntimeError(
                "portfolio disposition partition does not cover ranked ITT candidates"
            )
        prior_archive_decision_count = len(self.archive.decisions)
        preview_archive = self._preview_archive(candidates)
        archive_decisions = preview_archive.decisions[prior_archive_decision_count:]
        prior_memory = self._memory
        updater = self.outcome_updater
        learning = self.learning_lifecycle
        wave_request_sha256s = tuple(
            value.selection_request.request_sha256 for value in waves
        )
        result_receipt_sha256s = tuple(
            value.receipt.receipt_sha256 for value in results
        )
        outcome_preparation: CampaignPortfolioOutcomePreparation | None = None
        learning_preparation: CampaignPortfolioLearningPreparation | None = None
        try:
            if updater is not None:
                outcome_preparation = await updater.prepare_update(
                    request,
                    tuple(waves),
                    results,
                    prior_memory,
                )
                if type(outcome_preparation) is not CampaignPortfolioOutcomePreparation:
                    raise TypeError(
                        "outcome updater must return exact preparation evidence"
                    )
                CampaignPortfolioOutcomePreparation.__post_init__(outcome_preparation)
                if (
                    outcome_preparation.request_sha256 != request.request_sha256
                    or outcome_preparation.generation != step.generation
                    or outcome_preparation.wave_request_sha256s != wave_request_sha256s
                    or outcome_preparation.result_receipt_sha256s
                    != result_receipt_sha256s
                    or outcome_preparation.prior_memory_sha256
                    != typed_json_sha256(prior_memory)
                ):
                    raise ValueError("outcome preparation differs from the stage")
                updated_memory = outcome_preparation.updated_memory
            else:
                updated_memory = prior_memory
            if learning is not None:
                learning_preparation = (
                    await learning.prepare_portfolio_generation_close(
                        request,
                        tuple(waves),
                        results,
                        memory_credit_preparation,
                    )
                )
                if type(learning_preparation) is not (
                    CampaignPortfolioLearningPreparation
                ):
                    raise TypeError(
                        "learning lifecycle must return exact preparation evidence"
                    )
                CampaignPortfolioLearningPreparation.__post_init__(learning_preparation)
                if (
                    learning_preparation.request_sha256 != request.request_sha256
                    or learning_preparation.generation != step.generation
                    or learning_preparation.wave_request_sha256s != wave_request_sha256s
                    or learning_preparation.result_receipt_sha256s
                    != result_receipt_sha256s
                    or learning_preparation.memory_credit_preparation_sha256
                    != memory_credit_preparation.preparation_sha256
                ):
                    raise ValueError("learning preparation differs from the stage")
            cache_after = _cache_misses(
                await self.composition.engine.evaluation_cache_snapshot()
            )
            receipt = CampaignStageReceipt(
                request_sha256=request.request_sha256,
                preparation_sha256=request.preparation_sha256,
                generation=step.generation,
                kind=step.kind,
                candidate_occurrence_count=len(candidates),
                unique_evaluation_count=cache_after - cache_before,
                selector_audits=audits,
                result=_object(
                    {
                        "parent_selection": thaw_json(selection.evidence),
                        "parent_lanes": [
                            value.to_record() for value in selection.lanes
                        ],
                        "decision_slots": [
                            value.to_record() for value in selection.decision_slots
                        ],
                        "portfolio_wave_receipts": [
                            value.receipt.to_record() for value in results
                        ],
                        "candidates": [
                            _candidate_record(candidate) for candidate in candidates
                        ],
                        "ranked_itt_candidate_count": len(candidates),
                        "scored_candidate_count": scored_candidate_count,
                        "candidate_infeasible_count": infeasible_candidate_count,
                        "archive_candidate_decisions": [
                            _archive_decision_record(value)
                            for value in archive_decisions
                        ],
                        "archive_candidate_consideration_count": len(candidates),
                        "candidate_infeasibility_recourse": (
                            "retain_ranked_itt_reject_from_archive_no_resampling"
                        ),
                        "archive_after": thaw_json(
                            self._archive_record_for(preview_archive)
                        ),
                        "memory_before_sha256": typed_json_sha256(prior_memory),
                        "memory_after_sha256": typed_json_sha256(updated_memory),
                        "memory_projection_updated": updater is not None,
                        "outcome_update_preparation": (
                            None
                            if outcome_preparation is None
                            else outcome_preparation.to_record()
                        ),
                        "context_enrichment_applied": (
                            self.context_enricher is not None
                        ),
                        "archive_context_projection": (
                            None
                            if self.archive_context_projector is None
                            else {
                                "reserved_key": CAMPAIGN_ARCHIVE_CONTEXT_KEY,
                                "projector_id": (
                                    self.archive_context_projector.projector_id
                                ),
                                "projector_version": (
                                    self.archive_context_projector.projector_version
                                ),
                                "definition_sha256": (
                                    self.archive_context_projector.definition_sha256
                                ),
                            }
                        ),
                        "contextual_search_plan": (
                            None
                            if contextual_search_plan is None
                            else contextual_search_plan.to_record()
                        ),
                        **(
                            {}
                            if standalone_frontier_targets is None
                            else {
                                "standalone_frontier_targets": [
                                    value.to_record()
                                    for value in standalone_frontier_targets
                                ]
                            }
                        ),
                        "closed_loop_learning": (
                            None
                            if learning_preparation is None
                            else learning_preparation.to_record()
                        ),
                        "memory_credit_preparation_sha256": (
                            memory_credit_preparation.preparation_sha256
                        ),
                        "memory_credit_batch": (
                            None
                            if memory_credit_batch is None
                            else memory_credit_batch.to_record()
                        ),
                    }
                ),
            )
        except BaseException:
            if (
                learning is not None
                and type(learning_preparation) is CampaignPortfolioLearningPreparation
            ):
                learning.abort_portfolio_generation_close(learning_preparation)
            if (
                updater is not None
                and type(outcome_preparation) is CampaignPortfolioOutcomePreparation
            ):
                updater.abort_update(outcome_preparation)
            raise

        # All potentially fallible computation and receipt construction has
        # completed.  The following methods are synchronous exact-publication
        # operations whose protocol forbids I/O and new validation.
        committed_results, committed_batch = (
            self.composition.portfolio.commit_prepared_memory_credit_batch(
                memory_credit_preparation
            )
        )
        if committed_results != results or committed_batch != memory_credit_batch:
            raise RuntimeError("memory commit differs from its stage preparation")
        if updater is not None:
            assert outcome_preparation is not None
            updater.commit_update(outcome_preparation)
        if learning is not None:
            assert learning_preparation is not None
            learning.commit_portfolio_generation_close(learning_preparation)
        self._history.extend(candidates)
        self.archive = preview_archive
        self._memory = updated_memory
        self._selector_calls += len(waves)
        self._portfolio_waves[step.generation] = tuple(zip(waves, results, strict=True))
        return receipt

    async def _recombination_stage(
        self,
        request: CampaignStageRequest,
        *,
        cache_before: int,
    ) -> CampaignStageReceipt:
        step = request.step
        source_generation = step.source_portfolio_generation
        assert source_generation is not None
        source = self._portfolio_waves.get(source_generation)
        source_stage = self._stage_receipts.get(source_generation)
        if source is None or source_stage is None:
            raise RuntimeError("recombination has no in-memory source wave")
        if (
            request.source_portfolio is None
            or request.source_portfolio.receipt_sha256 != source_stage.receipt_sha256
        ):
            raise ValueError("recombination source receipt is not the runtime source")
        assert self.recombination is not None
        source_archive_utility = self._archive_utilities.get(source_generation)
        if source_archive_utility is None:
            raise RuntimeError("recombination lacks the source archive utility cutoff")
        utility_binder = self.recombination_utility_binder
        waves_list: list[PortfolioRecombinationWaveRequest] = []
        for slot, (wave, result) in enumerate(source):
            scored_source_count = len(result.scored_candidates)
            source_utilities = (
                None
                if utility_binder is None or scored_source_count < 2
                else utility_binder.bind(
                    source_archive_utility=source_archive_utility,
                    source_wave=wave,
                    source_result=result,
                )
            )
            if (
                source_utilities is not None
                and type(source_utilities) is not FrozenArchiveSourceUtilityReceipt
            ):
                raise TypeError("utility binder returned a foreign receipt")
            waves_list.append(
                PortfolioRecombinationWaveRequest(
                    source_wave=wave,
                    source_result=result,
                    ancestor=wave.parent,
                    generation=step.generation,
                    label_prefix=(f"campaign_g{step.generation:02d}_p{slot + 1:02d}"),
                    phase="campaign_portfolio_recombination",
                    source_archive_snapshot=(
                        None if source_utilities is None else source_archive_utility
                    ),
                    source_utilities=source_utilities,
                )
            )
        waves = tuple(waves_list)
        results = await gather_concurrent_stage(
            self.recombination.run(wave) for wave in waves
        )
        candidates = tuple(
            candidate for result in results for candidate in result.candidates
        )
        scored_candidate_count = sum(
            len(result.scored_candidates) for result in results
        )
        infeasible_candidate_count = sum(
            len(result.infeasible_candidates) for result in results
        )
        if scored_candidate_count + infeasible_candidate_count != len(candidates):
            raise RuntimeError(
                "recombination disposition partition does not cover selected ITT "
                "candidates"
            )
        for candidate in candidates:
            self._history.append(candidate)
            self.archive.consider(candidate)
        self._recombination_results[step.generation] = results
        self._recombination_source_portfolio_generations[step.generation] = (
            source_generation
        )
        contextual_post_recombination_credit = None
        planner = self.contextual_search_planner
        if planner is not None:
            source_wave_index = (source_generation + 1) // 2
            source_observations = tuple(
                value
                for value in planner.ledger.observations
                if value.campaign_scope_sha256 == planner.campaign_scope_sha256
                and value.wave_index == source_wave_index
            )
            contextual_post_recombination_credit = (
                observe_contextual_post_recombination_credit(
                    campaign_scope_sha256=planner.campaign_scope_sha256,
                    source_wave_index=source_wave_index,
                    observations=source_observations,
                    results=results,
                    post_stage_front_candidate_ids=tuple(
                        sorted(value.candidate_id for value in self.archive.front)
                    ),
                )
            )
            preview_ledger = type(planner.ledger)(
                observations=list(planner.ledger.observations),
                delayed_credits=list(planner.ledger.delayed_credits),
                allocation_realizations=list(planner.ledger.allocation_realizations),
            )
            preview_ledger.append_delayed_credit_batch(
                contextual_post_recombination_credit.credits
            )
            planner.ledger.delayed_credits.extend(
                contextual_post_recombination_credit.credits
            )
        cache_after = _cache_misses(
            await self.composition.engine.evaluation_cache_snapshot()
        )
        return CampaignStageReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            generation=step.generation,
            kind=step.kind,
            candidate_occurrence_count=len(candidates),
            unique_evaluation_count=cache_after - cache_before,
            selector_audits=(),
            result=_object(
                {
                    "recombination_wave_receipts": [
                        value.receipt.to_record() for value in results
                    ],
                    "candidates": [
                        _candidate_record(candidate) for candidate in candidates
                    ],
                    "selected_itt_candidate_count": len(candidates),
                    "scored_candidate_count": scored_candidate_count,
                    "candidate_infeasible_count": infeasible_candidate_count,
                    "candidate_infeasibility_recourse": (
                        "retain_selected_itt_reject_from_archive_no_resampling"
                    ),
                    "archive_after": thaw_json(self._archive_record()),
                    "archive_aware_source_utility": utility_binder is not None,
                    "contextual_post_recombination_credit": (
                        None
                        if contextual_post_recombination_credit is None
                        else contextual_post_recombination_credit.to_record()
                    ),
                }
            ),
        )

    async def execute_stage(
        self,
        request: CampaignStageRequest,
    ) -> CampaignStageReceipt:
        self._require_active()
        if request.preparation_sha256 != self.prepared.preparation_sha256:
            raise ValueError("stage request names a foreign preparation")
        step = request.step
        if step.generation != len(self._stage_receipts) + 1:
            raise ValueError("stage request is not the next prepared generation")
        pre_archive_sha256 = typed_json_sha256(self._archive_record())
        if pre_archive_sha256 != request.archive_utility.archive_sha256:
            raise ValueError(
                "stage archive utility differs from the live pre-stage archive"
            )
        pending_progress = self._pending_parent_selection_progress
        if pending_progress is None:
            if step.generation != 1:
                raise RuntimeError("campaign parent progress lost its pending stage")
        else:
            if pending_progress.generation != step.generation - 1:
                raise RuntimeError("campaign parent progress is not contiguous")
            self._parent_selection_progress.append(
                pending_progress.complete(request.archive_utility)
            )
            self._pending_parent_selection_progress = None
        cache_before = _cache_misses(
            await self.composition.engine.evaluation_cache_snapshot()
        )
        if step.kind is CampaignGenerationKind.PORTFOLIO:
            receipt = await self._portfolio_stage(request, cache_before=cache_before)
        else:
            receipt = await self._recombination_stage(
                request,
                cache_before=cache_before,
            )
        post_archive_sha256 = typed_json_sha256(self._archive_record())
        result_record = thaw_json(receipt.result)
        result_archive_after = freeze_json(result_record.get("archive_after"))
        if (
            type(result_archive_after) is not FrozenJsonObject
            or typed_json_sha256(result_archive_after) != post_archive_sha256
        ):
            raise RuntimeError(
                "stage receipt archive-after evidence differs from live state"
            )
        pending_progress = _PendingCampaignParentSelectionProgress(
            generation=step.generation,
            stage_kind=step.kind,
            stage_request_sha256=request.request_sha256,
            stage_receipt_sha256=receipt.receipt_sha256,
            pre_archive_sha256=pre_archive_sha256,
            post_archive_sha256=post_archive_sha256,
            pre_utility=request.archive_utility,
        )
        self._archive_utilities[step.generation] = request.archive_utility
        self._stage_receipts[step.generation] = receipt
        self._pending_parent_selection_progress = pending_progress
        return receipt

    def _validate_identifiable_reflection_lineage(
        self,
        reflection_input: CampaignIdentifiableReflectionInput,
    ) -> tuple[str, ...]:
        """Replay every projected contrast against the exact mutation wave.

        This join happens inside the runtime, before the executor boundary.  The
        executor consequently needs neither the portfolio results nor the later
        recombination results to obtain durable candidate/operator provenance.
        """

        reflection_input.__post_init__()
        query = reflection_input.query
        source = self._portfolio_waves.get(query.source_portfolio_generation)
        if source is None:
            raise RuntimeError("identifiable reflection has no source portfolio")
        expected: dict[str, dict[str, object]] = {}
        for wave, result in source:
            decision = result.selection_decision
            if decision is None:
                raise RuntimeError(
                    "identifiable reflection source omitted its ranked decision"
                )
            for selected, member, outcome in zip(
                decision.members,
                result.receipt.members,
                result.outcomes,
                strict=True,
            ):
                if member.disposition is not PortfolioMemberDisposition.SCORED:
                    continue
                child = outcome.candidate
                if child is None:  # Closed by result validation.
                    raise AssertionError("scored mutation source lost its candidate")
                if member.outcome_sha256 in expected:
                    raise RuntimeError("mutation source repeats outcome evidence")
                expected[member.outcome_sha256] = {
                    "parent_candidate_id": wave.parent.candidate_id,
                    "child_candidate_id": child.candidate_id,
                    "operator_invocation_id": member.operator_invocation_id,
                    "parent_configuration_sha256": (
                        AuthenticatedHypothesisObservation.configuration_sha256(
                            wave.parent.configuration
                        )
                    ),
                    "child_configuration_sha256": (
                        AuthenticatedHypothesisObservation.configuration_sha256(
                            child.configuration
                        )
                    ),
                    "option_id": selected.option_id,
                    "option_identity_sha256": selected.option_identity_sha256,
                    "option_family": selected.family,
                    "finite_contract_identity_sha256": (
                        wave.selection_request.finite_variation_contract.identity_sha256
                    ),
                    "changed_paths": member.materialization.changed_paths,
                }
        observed_source_ids: list[str] = []
        for contrast in reflection_input.evidence.contrasts:
            observed_source_ids.append(contrast.source_evidence_id)
            expected_lineage = expected.get(contrast.source_evidence_id)
            if expected_lineage is None:
                raise ValueError(
                    "identifiable contrast is not a scored source mutation"
                )
            observed_lineage = {
                "parent_candidate_id": contrast.parent_candidate_id,
                "child_candidate_id": contrast.child_candidate_id,
                "operator_invocation_id": contrast.operator_invocation_id,
                "parent_configuration_sha256": (contrast.parent_configuration_sha256),
                "child_configuration_sha256": contrast.child_configuration_sha256,
                "option_id": contrast.option_id,
                "option_identity_sha256": contrast.option_identity_sha256,
                "option_family": contrast.option_family,
                "finite_contract_identity_sha256": (
                    contrast.finite_contract_identity_sha256
                ),
                "changed_paths": (contrast.affected_path,),
            }
            if observed_lineage != expected_lineage:
                raise ValueError(
                    "identifiable contrast candidate/operator lineage is foreign"
                )
            if contrast.event_index != query.sealed_cutoff_event_index_inclusive:
                raise ValueError(
                    "identifiable contrast event differs from the source cutoff"
                )
        canonical_source_ids = tuple(sorted(observed_source_ids))
        if len(set(observed_source_ids)) != len(observed_source_ids):
            raise ValueError("identifiable contrast source evidence must be unique")
        if self._consumed_reflection_source_evidence_ids.intersection(
            canonical_source_ids
        ):
            raise ValueError("identifiable reflection attempted to reuse evidence")
        return canonical_source_ids

    async def reflect(
        self,
        request: CampaignReflectionRequest,
    ) -> CampaignReflectionReceipt:
        self._require_active()
        if type(request) is not CampaignReflectionRequest:
            raise TypeError("request must be an exact CampaignReflectionRequest")
        CampaignReflectionRequest.__post_init__(request)
        source_generation = request.wave.source_generation
        source = self._recombination_results.get(source_generation)
        if source is None:
            raise RuntimeError("reflection has no exact recombination source")
        source_stage = self._stage_receipts.get(source_generation)
        if (
            source_stage is None
            or source_stage.receipt_sha256 != request.source_stage.receipt_sha256
        ):
            raise ValueError("reflection request differs from the runtime source stage")
        identifiable_executor = self.identifiable_reflection_executor
        evidence_source = self.identifiable_reflection_evidence_source
        reflection_input: CampaignIdentifiableReflectionInput | None = None
        if identifiable_executor is not None:
            if evidence_source is None:  # Closed by construction; retained defensively.
                raise RuntimeError(
                    "identifiable reflection omitted its evidence source"
                )
            source_portfolio_generation = (
                self._recombination_source_portfolio_generations.get(source_generation)
            )
            if source_portfolio_generation is None:
                raise RuntimeError(
                    "identifiable reflection lacks its source portfolio cutoff"
                )
            query = CampaignIdentifiableReflectionEvidenceQuery(
                reflection_request_sha256=request.request_sha256,
                preparation_sha256=request.preparation_sha256,
                runtime_start_receipt_sha256=(request.runtime_start_receipt_sha256),
                campaign_sha256=evidence_source.campaign_sha256,
                workload_instance_sha256=(evidence_source.workload_instance_sha256),
                evaluator_contract_sha256=(evidence_source.evaluator_contract_sha256),
                wave=request.wave,
                source_stage_receipt_sha256=request.source_stage.receipt_sha256,
                source_portfolio_generation=source_portfolio_generation,
                prior_cutoff_event_index_exclusive=(
                    self._last_identifiable_reflection_cutoff
                ),
                sealed_cutoff_event_index_inclusive=source_portfolio_generation,
            )
            source_projection = evidence_source.project(query)
            if type(source_projection) is not (
                CampaignIdentifiableReflectionEvidenceProjection
            ):
                raise TypeError(
                    "identifiable evidence source returned a foreign projection"
                )
            reflection_input = CampaignIdentifiableReflectionInput(
                query=query,
                source=source_projection,
            )
            if request.request_sha256 in self._identifiable_reflection_inputs:
                raise ValueError("reflection request was already executed")
            source_evidence_ids = self._validate_identifiable_reflection_lineage(
                reflection_input
            )
            # Consume before yielding to provider code.  A failed or cancelled
            # call cannot be replayed as though the same cohort were new.
            self._identifiable_reflection_inputs[request.request_sha256] = (
                reflection_input
            )
            self._consumed_reflection_source_evidence_ids.update(source_evidence_ids)
            self._last_identifiable_reflection_cutoff = source_portfolio_generation
            result = await identifiable_executor.reflect(reflection_input)
        else:
            legacy_executor = self.legacy_recombination_reflection_executor
            if legacy_executor is None:
                legacy_executor = self.reflection_executor
            if legacy_executor is None:
                raise RuntimeError(
                    "prepared schedule requires an injected reflection executor"
                )
            result = await legacy_executor.reflect(request, source)
        if type(result) is not FrozenJsonObject or freeze_json(result) is not result:
            raise TypeError("reflection executor must return a frozen object")
        if reflection_input is not None:
            result_record = thaw_json(result)
            if CAMPAIGN_IDENTIFIABLE_REFLECTION_BINDING_KEY in result_record:
                raise ValueError(
                    "reflection executor attempted to author the runtime binding"
                )
            result_record[CAMPAIGN_IDENTIFIABLE_REFLECTION_BINDING_KEY] = {
                "schema_version": 1,
                "input_sha256": reflection_input.input_sha256,
                "query_sha256": reflection_input.query.query_sha256,
                "source_projection_sha256": (reflection_input.source.projection_sha256),
                "registry_snapshot_sha256": (
                    reflection_input.source.registry_snapshot_sha256
                ),
                "evidence_snapshot_sha256": (reflection_input.evidence.snapshot_sha256),
                "prior_cutoff_event_index_exclusive": (
                    reflection_input.query.prior_cutoff_event_index_exclusive
                ),
                "sealed_cutoff_event_index_inclusive": (
                    reflection_input.query.sealed_cutoff_event_index_inclusive
                ),
                "source_stage_payload_exposed": False,
                "recombination_results_exposed": False,
            }
            result = _object(result_record)
        receipt = CampaignReflectionReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            source_generation=source_generation,
            source_stage_receipt_sha256=request.source_stage.receipt_sha256,
            logical_agent_calls=request.wave.call_count,
            visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            status=CampaignReflectionStatus.COMPLETED,
            failure_type=None,
            quarantined_result=result,
        )
        learning = self.learning_lifecycle
        if learning is not None:
            evidence = learning.reflection_completed(request, receipt, result)
            if (
                type(evidence) is not FrozenJsonObject
                or freeze_json(evidence) is not evidence
            ):
                raise TypeError("learning reflection hook must return frozen evidence")
            self._reflection_learning_evidence[receipt.receipt_sha256] = evidence
        self._reflection_results[receipt.receipt_sha256] = result
        return receipt

    async def admit_for_testing(
        self,
        request: CampaignReflectionTestAdmissionRequest,
    ) -> CampaignReflectionTestAdmissionReceipt:
        admitted = tuple(sorted(value.receipt_sha256 for value in request.reflections))
        eligible = tuple(
            sorted(
                {
                    *request.previously_test_eligible_reflection_receipt_sha256s,
                    *admitted,
                }
            )
        )
        if any(value not in self._reflection_results for value in admitted):
            raise RuntimeError("test admission names unavailable reflection content")
        learning_evidence: FrozenJsonObject | None = None
        learning = self.learning_lifecycle
        if learning is not None:
            contents = tuple(
                (
                    reflection,
                    self._reflection_results[reflection.receipt_sha256],
                )
                for reflection in sorted(
                    request.reflections,
                    key=lambda value: value.receipt_sha256,
                )
            )
            learning_evidence = learning.reflections_admitted(request, contents)
            if (
                type(learning_evidence) is not FrozenJsonObject
                or freeze_json(learning_evidence) is not learning_evidence
            ):
                raise TypeError("learning admission hook must return frozen evidence")
        self._test_eligible_reflection_hashes.update(admitted)
        return CampaignReflectionTestAdmissionReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            barrier_generation=request.barrier.generation,
            admitted_reflection_receipt_sha256s=admitted,
            test_eligible_reflection_receipt_sha256s=eligible,
            lifecycle_promoted=False,
            evidence=_object(
                {
                    "admission_scope": "controlled_future_testing_only",
                    "normal_retrieval_mutated": False,
                    "learning_lifecycle": (
                        None
                        if learning_evidence is None
                        else thaw_json(learning_evidence)
                    ),
                    "reflection_completion_learning": [
                        {
                            "reflection_receipt_sha256": value,
                            "evidence": thaw_json(
                                self._reflection_learning_evidence[value]
                            ),
                        }
                        for value in admitted
                        if value in self._reflection_learning_evidence
                    ],
                }
            ),
        )

    async def finalize(
        self,
        request: CampaignFinalizationRequest,
    ) -> CampaignFinalizationReceipt:
        contextual_terminal_credit = None
        planner = self.contextual_search_planner
        if (
            planner is not None
            and request.status is CampaignExecutionStatus.COMPLETED
            and planner.ledger.observations
        ):
            contextual_terminal_credit = observe_contextual_terminal_persistence(
                campaign_scope_sha256=planner.campaign_scope_sha256,
                available_at_wave_index=(
                    len(self.prepared.schedule.portfolio_generations) + 2
                ),
                finalization_request_sha256=request.request_sha256,
                observations=tuple(planner.ledger.observations),
                terminal_front_candidate_ids=tuple(
                    sorted(value.candidate_id for value in self.archive.front)
                ),
            )
            preview_ledger = type(planner.ledger)(
                observations=list(planner.ledger.observations),
                delayed_credits=list(planner.ledger.delayed_credits),
                allocation_realizations=list(planner.ledger.allocation_realizations),
            )
            preview_ledger.append_delayed_credit_batch(
                contextual_terminal_credit.credits
            )
            planner.ledger.delayed_credits.extend(contextual_terminal_credit.credits)
        return CampaignFinalizationReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            status=request.status,
            evidence=_object(
                {
                    "stage_count": len(self._stage_receipts),
                    "history_size": len(self._history),
                    "final_archive": thaw_json(self._archive_record()),
                    "parent_selection_progress": [
                        value.to_record() for value in self._parent_selection_progress
                    ],
                    "reflection_learning_evidence": [
                        {
                            "reflection_receipt_sha256": receipt_sha256,
                            "evidence": thaw_json(evidence),
                        }
                        for receipt_sha256, evidence in sorted(
                            self._reflection_learning_evidence.items()
                        )
                    ],
                    "reflection_evidence_mode": (
                        "identifiable_direct_mutation"
                        if self.identifiable_reflection_executor is not None
                        else (
                            "legacy_recombination"
                            if self.legacy_recombination_reflection_executor is not None
                            or self.reflection_executor is not None
                            else "not_configured"
                        )
                    ),
                    "identifiable_reflection_inputs": [
                        value.to_record()
                        for _, value in sorted(
                            self._identifiable_reflection_inputs.items()
                        )
                    ],
                    "contextual_terminal_persistence_credit": (
                        None
                        if contextual_terminal_credit is None
                        else contextual_terminal_credit.to_record()
                    ),
                }
            ),
        )

    async def cleanup(
        self,
        request: CampaignCleanupRequest,
    ) -> CampaignCleanupReceipt:
        resource_evidence: FrozenJsonObject
        if self.owned_resources is None:
            resource_evidence = _object(
                {
                    "ownership": "external_to_adapter",
                    "adapter_owned_resource_count": 0,
                    "external_resource_close_required_by_caller": True,
                }
            )
        else:
            resource_evidence = await self.owned_resources.close()
            if (
                type(resource_evidence) is not FrozenJsonObject
                or freeze_json(resource_evidence) is not resource_evidence
            ):
                raise TypeError("owned-resource close must return a frozen object")
        self._cleaned = True
        return CampaignCleanupReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            released=True,
            evidence=_object(
                {
                    "resource_cleanup": thaw_json(resource_evidence),
                    "runtime_closed": True,
                }
            ),
        )


__all__ = [
    "AgenticPortfolioCampaignRuntime",
    "ArchiveDiverseEliteCampaignParentSelector",
    "ArchiveEliteCampaignParentSelector",
    "ArchiveEliteExplorerCampaignParentSelector",
    "ArchiveReservoirCampaignParentSelector",
    "ResidualHypervolumeCampaignParentSelector",
    "StagnationAwareDiverseCampaignParentSelector",
    "CAMPAIGN_CONTEXTUAL_HISTORY_KEY",
    "CAMPAIGN_ARCHIVE_CONTEXT_KEY",
    "CAMPAIGN_FRONTIER_TARGET_KEY",
    "CAMPAIGN_IDENTIFIABLE_REFLECTION_BINDING_KEY",
    "MEMORY_ESTIMAND_STRATUM_SHA256_KEY",
    "CampaignDecisionSlot",
    "CampaignIdentifiableReflectionEvidenceProjection",
    "CampaignIdentifiableReflectionEvidenceQuery",
    "CampaignIdentifiableReflectionEvidenceSource",
    "CampaignIdentifiableReflectionInput",
    "CampaignLearningLifecyclePort",
    "CampaignOwnedRuntimeResourcePort",
    "CampaignParentLane",
    "CampaignParentSelection",
    "CampaignParentSelectionProgress",
    "CampaignParentSelectionPort",
    "CampaignPortfolioContextEnricher",
    "CampaignPortfolioMemoryEstimandProjection",
    "CampaignPortfolioMemoryEstimandProjector",
    "CampaignPortfolioLearningPreparation",
    "CampaignPortfolioOutcomePreparation",
    "CampaignPortfolioWaveContext",
    "CampaignPortfolioWaveBatchFactory",
    "CampaignPortfolioWaveFactory",
    "CampaignPortfolioWavePreparationObserver",
    "CampaignPortfolioWavePreparationReceipt",
    "CampaignPortfolioOutcomeUpdater",
    "CampaignRecombinationUtilityBinder",
    "CampaignLegacyRecombinationReflectionExecutor",
    "CampaignReflectionFalsificationSource",
    "CampaignReflectionExecutor",
    "CampaignSelectorRequestPromptRenderer",
    "CommittedRegistryIdentifiableReflectionEvidenceSource",
]

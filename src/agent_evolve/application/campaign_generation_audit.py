"""Transactional production audit of finite-action campaign generations.

Every ranked portfolio action is first projected into an authenticated metric
observation or a typed candidate-infeasibility exclusion. Observations enter
the append-only evidence registry; exclusions remain in the generation audit
without resampling. If the same pre-outcome waves contain an
admitted diagnostic memory assignment, registered semantic plans are sealed
against that one prospective registry snapshot and adjudicated by the global
falsification gate.  The workload seams are deliberately narrow: raw metric
effects, finite-action semantics, and hypothesis matching.

Randomized card exposure is *not* randomized action administration.  The
candidate observations produced here retain ``DIRECT_MUTATION`` provenance;
the separate memory-credit transaction owns causal card usefulness.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from agent_evolve.application.campaign_evidence_registry import (
    CampaignEvidenceAppendPreparation,
    CampaignEvidenceRegistry,
)
from agent_evolve.application.campaign_execution import CampaignStageRequest
from agent_evolve.application.campaign_learning import (
    CampaignInsightAuditBinding,
    CampaignSemanticAuditPlan,
)
from agent_evolve.application.evolution_campaign import CampaignGenerationKind
from agent_evolve.application.insight_memory import InsightMemoryEntry
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemoryCreditBatchPreparation,
    PortfolioMemoryMatchedControlWavePlan,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.application.portfolio_hypothesis_observations import (
    CampaignActionSemanticsCompiler,
    CampaignHypothesisObservationExclusion,
    CampaignMetricEffectProjector,
    CampaignPortfolioHypothesisEvidenceProjection,
    project_portfolio_hypothesis_evidence,
)
from agent_evolve.application.portfolio_memory_attribution import (
    PortfolioMemoryAttributionAudit,
    audit_portfolio_memory_attribution,
)
from agent_evolve.application.portfolio_memory_matched_control import (
    PortfolioMemoryMatchedControlOutcome,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    EvidenceProvenance,
    GlobalHypothesisAuditRequest,
    GlobalHypothesisEvidenceMatcher,
    GlobalHypothesisFalsificationGate,
)
from agent_evolve.policies.memory.staged_causal import MemoryAssignmentArm
from agent_evolve.ports.agentic_generator import MetricComparisonAnchorKind
from agent_evolve.ports.portfolio_selection import PortfolioExperimentalArm


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_PROJECTION_DOMAIN = b"agent-evolve:campaign-generation-audit-projection:v2\x00"
_PREPARATION_DOMAIN = b"agent-evolve:campaign-generation-audit-preparation:v2\x00"
_CONTEXT_DOMAIN = b"agent-evolve:campaign-diagnostic-context-binding:v1\x00"

PRODUCTION_GENERATION_AUDITOR_ID = "authenticated_portfolio_generation_audit"
PRODUCTION_GENERATION_AUDITOR_VERSION = 3
PRODUCTION_GENERATION_AUDITOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:authenticated-portfolio-generation-audit:v3:"
    b"all-ranked-itt-actions-accounted:scored-actions-appended:"
    b"candidate-infeasibility-explicitly-excluded:no-resampling:"
    b"one-prospective-snapshot:real-global-gate:"
    b"diagnostic-assignment-context:typed-mutation-current-parent-only:"
    b"post-barrier-randomized-active-neutral-pair-outcomes:"
    b"no-single-block-causal-claim:no-online-score-update"
).hexdigest()


def resolve_portfolio_memory_matched_control_outcomes(
    *,
    waves: tuple[PortfolioVariationWaveRequest, ...],
    results: tuple[PortfolioVariationWaveResult, ...],
) -> tuple[PortfolioMemoryMatchedControlOutcome, ...]:
    """Join complete precommitted M/N pairs after the generation barrier.

    This is an append-only experimental observation.  The function neither
    mutates memory scores nor labels one realized pair as an identified effect.
    """

    if type(waves) is not tuple or type(results) is not tuple:
        raise TypeError("waves/results must be exact tuples")
    if len(waves) != len(results):
        raise ValueError("matched outcome join requires one result per wave")
    result_by_request: dict[str, PortfolioVariationWaveResult] = {}
    for result in results:
        if type(result) is not PortfolioVariationWaveResult:
            raise TypeError("results must contain exact wave results")
        PortfolioVariationWaveResult.__post_init__(result)
        request_sha256 = result.receipt.request_sha256
        if request_sha256 in result_by_request:
            raise ValueError("matched outcome results repeat a request identity")
        result_by_request[request_sha256] = result

    grouped: dict[
        str,
        list[
            tuple[
                PortfolioMemoryMatchedControlWavePlan,
                PortfolioVariationWaveResult,
            ]
        ],
    ] = {}
    for wave in waves:
        if type(wave) is not PortfolioVariationWaveRequest:
            raise TypeError("waves must contain exact wave requests")
        PortfolioVariationWaveRequest.__post_init__(wave)
        matched = wave.matched_memory_control
        if matched is None:
            continue
        request_sha256 = wave.selection_request.request_sha256
        result = result_by_request.get(request_sha256)
        if result is None:
            raise ValueError("matched arm has no joined wave result")
        if (
            result.receipt.request_sha256 != request_sha256
            or result.receipt.generation != wave.generation
        ):
            raise ValueError("matched arm result differs from its wave")
        grouped.setdefault(matched.plan.plan_sha256, []).append((matched, result))

    outcomes: list[PortfolioMemoryMatchedControlOutcome] = []
    for plan_sha256 in sorted(grouped):
        members = grouped[plan_sha256]
        if len(members) != 2:
            raise ValueError("matched control plan requires exactly two realized arms")
        plan = members[0][0].plan
        if any(value[0].plan != plan for value in members):
            raise ValueError("matched arms disagree on their precommitted plan")
        aggregation_sha256s = {
            value[0].aggregation.binding_sha256 for value in members
        }
        if len(aggregation_sha256s) != 1:
            raise ValueError("matched arms use different reward aggregations")
        by_arm = {value[0].assignment.arm: value for value in members}
        if set(by_arm) != {
            PortfolioExperimentalArm.MEMORY,
            PortfolioExperimentalArm.NEUTRAL,
        }:
            raise ValueError("matched plan must realize one M and one N arm")
        active, active_result = by_arm[PortfolioExperimentalArm.MEMORY]
        neutral, neutral_result = by_arm[PortfolioExperimentalArm.NEUTRAL]
        active_reward = float(active.aggregation.aggregate(active_result.outcomes))
        neutral_reward = float(
            neutral.aggregation.aggregate(neutral_result.outcomes)
        )
        outcomes.append(
            PortfolioMemoryMatchedControlOutcome(
                plan_sha256=plan_sha256,
                generation=plan.generation,
                reference=plan.reference,
                aggregation_binding_sha256=active.aggregation.binding_sha256,
                active_view_sha256=active.arm_view.view_sha256,
                neutral_view_sha256=neutral.arm_view.view_sha256,
                active_result_receipt_sha256=(
                    active_result.receipt.receipt_sha256
                ),
                neutral_result_receipt_sha256=(
                    neutral_result.receipt.receipt_sha256
                ),
                active_wave_reward=active_reward,
                neutral_wave_reward=neutral_reward,
            )
        )
    return tuple(outcomes)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _reference_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticContextBinding:
    """One audit context authenticated only by pre-outcome assignments."""

    reference: InsightRef
    exact_context_sha256: str
    assignment_sha256s: tuple[str, ...]
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be exact InsightRef")
        InsightRef.__post_init__(self.reference)
        require_sha256(self.exact_context_sha256, "exact_context_sha256")
        if type(self.assignment_sha256s) is not tuple or not self.assignment_sha256s:
            raise ValueError("assignment_sha256s must be a non-empty exact tuple")
        for value in self.assignment_sha256s:
            require_sha256(value, "assignment_sha256s")
        if self.assignment_sha256s != tuple(sorted(set(self.assignment_sha256s))):
            raise ValueError("assignment_sha256s must be unique and canonical")
        object.__setattr__(
            self,
            "binding_sha256",
            _hash(_CONTEXT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reference": _reference_record(self.reference),
            "exact_context_sha256": self.exact_context_sha256,
            "assignment_sha256s": list(self.assignment_sha256s),
            "context_authority": "pre_outcome_resolved_diagnostic_assignment",
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}


@dataclass(frozen=True, slots=True)
class CampaignGenerationAuditProjection:
    """Real-gate audit family over one shared prospective registry snapshot."""

    stage_request_sha256: str
    generation: int
    wave_request_sha256s: tuple[str, ...]
    wave_result_receipt_sha256s: tuple[str, ...]
    memory_credit_batch_receipt_sha256: str
    registry_snapshot_sha256: str
    evidence_append_preparation_sha256: str
    context_bindings: tuple[CampaignDiagnosticContextBinding, ...]
    audits: tuple[CampaignInsightAuditBinding, ...]
    projection_policy_id: str = field(
        init=False,
        default=PRODUCTION_GENERATION_AUDITOR_ID,
    )
    projection_policy_version: int = field(
        init=False,
        default=PRODUCTION_GENERATION_AUDITOR_VERSION,
    )
    projection_policy_definition_sha256: str = field(
        init=False,
        default=PRODUCTION_GENERATION_AUDITOR_DEFINITION_SHA256,
    )
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.stage_request_sha256, "stage_request_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        for name in ("wave_request_sha256s", "wave_result_receipt_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be non-empty")
            for value in values:
                require_sha256(value, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        for name in (
            "memory_credit_batch_receipt_sha256",
            "registry_snapshot_sha256",
            "evidence_append_preparation_sha256",
            "projection_policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.context_bindings) is not tuple
            or not self.context_bindings
            or any(
                type(value) is not CampaignDiagnosticContextBinding
                for value in self.context_bindings
            )
        ):
            raise ValueError("context_bindings must contain exact bindings")
        for value in self.context_bindings:
            CampaignDiagnosticContextBinding.__post_init__(value)
        references = tuple(value.reference for value in self.context_bindings)
        if references != tuple(sorted(set(references))):
            raise ValueError("context bindings must use canonical unique references")
        if (
            type(self.audits) is not tuple
            or not self.audits
            or any(
                type(value) is not CampaignInsightAuditBinding for value in self.audits
            )
        ):
            raise ValueError("audits must contain exact bindings")
        for value in self.audits:
            CampaignInsightAuditBinding.__post_init__(value)
        if tuple(value.request.reference for value in self.audits) != references:
            raise ValueError("audits must exactly cover context-bound references")
        if any(
            value.request.registry_snapshot_sha256 != self.registry_snapshot_sha256
            or value.receipt.registry_snapshot_sha256 != self.registry_snapshot_sha256
            for value in self.audits
        ):
            raise ValueError("audit family does not share the sealed registry snapshot")
        if (
            type(self.projection_policy_id) is not str
            or _TOKEN.fullmatch(self.projection_policy_id) is None
            or self.projection_policy_version != PRODUCTION_GENERATION_AUDITOR_VERSION
            or self.projection_policy_definition_sha256
            != PRODUCTION_GENERATION_AUDITOR_DEFINITION_SHA256
        ):
            raise ValueError("unsupported production generation auditor identity")
        object.__setattr__(
            self,
            "projection_sha256",
            _hash(_PROJECTION_DOMAIN, self._unsigned_record()),
        )

    @property
    def references(self) -> tuple[InsightRef, ...]:
        return tuple(value.reference for value in self.context_bindings)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "stage_request_sha256": self.stage_request_sha256,
            "generation": self.generation,
            "wave_request_sha256s": list(self.wave_request_sha256s),
            "wave_result_receipt_sha256s": list(self.wave_result_receipt_sha256s),
            "memory_credit_batch_receipt_sha256": (
                self.memory_credit_batch_receipt_sha256
            ),
            "registry_snapshot_sha256": self.registry_snapshot_sha256,
            "evidence_append_preparation_sha256": (
                self.evidence_append_preparation_sha256
            ),
            "context_binding_sha256s": [
                value.binding_sha256 for value in self.context_bindings
            ],
            "audit_request_sha256s": [
                value.request.request_sha256 for value in self.audits
            ],
            "audit_receipt_sha256s": [
                value.receipt.audit_receipt_sha256 for value in self.audits
            ],
            "projection_policy": {
                "policy_id": self.projection_policy_id,
                "policy_version": self.projection_policy_version,
                "definition_sha256": self.projection_policy_definition_sha256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "context_bindings": [value.to_record() for value in self.context_bindings],
            "audits": [
                {
                    "request": value.request.to_record(),
                    "receipt": value.receipt.to_record(),
                    "exact_context_sha256": value.exact_context_sha256,
                }
                for value in self.audits
            ],
            "projection_sha256": self.projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignGenerationAuditPreparation:
    """Evidence append plus optional real-gate lifecycle projection."""

    stage_request_sha256: str
    generation: int
    wave_request_sha256s: tuple[str, ...]
    wave_result_receipt_sha256s: tuple[str, ...]
    memory_credit_preparation_sha256: str
    observations: tuple[AuthenticatedHypothesisObservation, ...]
    observation_exclusions: tuple[CampaignHypothesisObservationExclusion, ...]
    hypothesis_evidence_projection: CampaignPortfolioHypothesisEvidenceProjection
    evidence_append: CampaignEvidenceAppendPreparation
    memory_attribution_audit: PortfolioMemoryAttributionAudit | None
    matched_memory_control_outcomes: tuple[
        PortfolioMemoryMatchedControlOutcome, ...
    ]
    projection: CampaignGenerationAuditProjection | None
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.stage_request_sha256, "stage_request_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        for name in ("wave_request_sha256s", "wave_result_receipt_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be non-empty")
            for value in values:
                require_sha256(value, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        require_sha256(
            self.memory_credit_preparation_sha256,
            "memory_credit_preparation_sha256",
        )
        if type(self.observations) is not tuple or any(
            type(value) is not AuthenticatedHypothesisObservation
            for value in self.observations
        ):
            raise TypeError("observations must contain exact authenticated evidence")
        for value in self.observations:
            AuthenticatedHypothesisObservation.__post_init__(value)
            if (
                value.event_index != self.generation
                or value.provenance is not EvidenceProvenance.DIRECT_MUTATION
                or value.operator_family != "typed_mutation"
            ):
                raise ValueError(
                    "production MVP accepts current-generation direct mutations only"
                )
        if type(self.observation_exclusions) is not tuple or any(
            type(value) is not CampaignHypothesisObservationExclusion
            for value in self.observation_exclusions
        ):
            raise TypeError("observation_exclusions must contain exact receipts")
        for value in self.observation_exclusions:
            CampaignHypothesisObservationExclusion.__post_init__(value)
        if not self.observations and not self.observation_exclusions:
            raise ValueError("generation audit cannot omit every ranked ITT member")
        if type(self.hypothesis_evidence_projection) is not (
            CampaignPortfolioHypothesisEvidenceProjection
        ):
            raise TypeError("hypothesis_evidence_projection must be exact")
        CampaignPortfolioHypothesisEvidenceProjection.__post_init__(
            self.hypothesis_evidence_projection
        )
        if (
            self.hypothesis_evidence_projection.observations != self.observations
            or self.hypothesis_evidence_projection.exclusions
            != self.observation_exclusions
            or self.hypothesis_evidence_projection.wave_receipt_sha256s
            != self.wave_result_receipt_sha256s
        ):
            raise ValueError("hypothesis evidence projection differs from audit inputs")
        if type(self.evidence_append) is not CampaignEvidenceAppendPreparation:
            raise TypeError("evidence_append must be exact")
        CampaignEvidenceAppendPreparation.__post_init__(self.evidence_append)
        if self.evidence_append.observations != self.observations:
            raise ValueError("evidence append differs from projected observations")
        attribution = self.memory_attribution_audit
        if attribution is not None:
            if type(attribution) is not PortfolioMemoryAttributionAudit:
                raise TypeError("memory_attribution_audit must be exact or None")
            PortfolioMemoryAttributionAudit.__post_init__(attribution)
            if (
                attribution.generation != self.generation
                or attribution.wave_request_sha256s != self.wave_request_sha256s
                or attribution.wave_result_receipt_sha256s
                != self.wave_result_receipt_sha256s
            ):
                raise ValueError(
                    "memory attribution audit differs from generation inputs"
                )
        matched_outcomes = self.matched_memory_control_outcomes
        if type(matched_outcomes) is not tuple or any(
            type(value) is not PortfolioMemoryMatchedControlOutcome
            for value in matched_outcomes
        ):
            raise TypeError(
                "matched_memory_control_outcomes must contain exact outcomes"
            )
        result_receipts = set(self.wave_result_receipt_sha256s)
        plan_sha256s: list[str] = []
        for value in matched_outcomes:
            PortfolioMemoryMatchedControlOutcome.__post_init__(value)
            if value.generation != self.generation:
                raise ValueError("matched outcome differs from audit generation")
            if {
                value.active_result_receipt_sha256,
                value.neutral_result_receipt_sha256,
            } - result_receipts:
                raise ValueError("matched outcome names a foreign wave result")
            plan_sha256s.append(value.plan_sha256)
        if tuple(plan_sha256s) != tuple(sorted(set(plan_sha256s))):
            raise ValueError("matched outcomes require canonical unique plans")
        projection = self.projection
        if projection is not None:
            if type(projection) is not CampaignGenerationAuditProjection:
                raise TypeError("projection must be exact or None")
            CampaignGenerationAuditProjection.__post_init__(projection)
            expected = (
                self.stage_request_sha256,
                self.generation,
                self.wave_request_sha256s,
                self.wave_result_receipt_sha256s,
                self.evidence_append.prospective_snapshot.snapshot_sha256,
                self.evidence_append.preparation_sha256,
            )
            observed = (
                projection.stage_request_sha256,
                projection.generation,
                projection.wave_request_sha256s,
                projection.wave_result_receipt_sha256s,
                projection.registry_snapshot_sha256,
                projection.evidence_append_preparation_sha256,
            )
            if observed != expected:
                raise ValueError("generation projection differs from evidence append")
        object.__setattr__(
            self,
            "preparation_sha256",
            _hash(_PREPARATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 4,
            "stage_request_sha256": self.stage_request_sha256,
            "generation": self.generation,
            "wave_request_sha256s": list(self.wave_request_sha256s),
            "wave_result_receipt_sha256s": list(self.wave_result_receipt_sha256s),
            "memory_credit_preparation_sha256": (self.memory_credit_preparation_sha256),
            "observation_sha256s": [
                value.observation_sha256 for value in self.observations
            ],
            "observation_exclusion_sha256s": [
                value.exclusion_sha256 for value in self.observation_exclusions
            ],
            "hypothesis_evidence_projection_sha256": (
                self.hypothesis_evidence_projection.projection_sha256
            ),
            "evidence_append_preparation_sha256": (
                self.evidence_append.preparation_sha256
            ),
            "prospective_registry_snapshot_sha256": (
                self.evidence_append.prospective_snapshot.snapshot_sha256
            ),
            "memory_attribution_audit_sha256": (
                None
                if self.memory_attribution_audit is None
                else self.memory_attribution_audit.audit_sha256
            ),
            "matched_memory_control_outcome_sha256s": [
                value.outcome_sha256
                for value in self.matched_memory_control_outcomes
            ],
            "projection_sha256": (
                None if self.projection is None else self.projection.projection_sha256
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "status": (
                "evidence_append_and_real_gate_prepared"
                if self.projection is not None
                else "evidence_append_prepared_no_diagnostic_assignment"
            ),
            "hypothesis_evidence_projection": (
                self.hypothesis_evidence_projection.to_record()
            ),
            "evidence_append": self.evidence_append.to_record(),
            "memory_attribution_audit": (
                None
                if self.memory_attribution_audit is None
                else self.memory_attribution_audit.to_record()
            ),
            "matched_memory_control_outcomes": [
                value.to_record()
                for value in self.matched_memory_control_outcomes
            ],
            "projection": (
                None if self.projection is None else self.projection.to_record()
            ),
            "preparation_sha256": self.preparation_sha256,
        }


@dataclass(slots=True)
class TransactionalPortfolioGenerationAuditor:
    """Concrete prepare/commit/abort path for production campaign evidence."""

    evidence_registry: CampaignEvidenceRegistry
    campaign_sha256: str
    workload_instance_sha256: str
    evaluator_contract_sha256: str
    metric_projector: CampaignMetricEffectProjector
    action_semantics_compiler: CampaignActionSemanticsCompiler
    hypothesis_matcher: GlobalHypothesisEvidenceMatcher
    falsification_gate: GlobalHypothesisFalsificationGate = field(
        default_factory=GlobalHypothesisFalsificationGate
    )
    _prepared: dict[str, CampaignGenerationAuditPreparation] = field(
        init=False,
        default_factory=dict,
    )

    @property
    def policy_id(self) -> str:
        return PRODUCTION_GENERATION_AUDITOR_ID

    @property
    def policy_version(self) -> int:
        return PRODUCTION_GENERATION_AUDITOR_VERSION

    @property
    def definition_sha256(self) -> str:
        return PRODUCTION_GENERATION_AUDITOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.evidence_registry) is not CampaignEvidenceRegistry:
            raise TypeError("evidence_registry must be exact CampaignEvidenceRegistry")
        for name in (
            "campaign_sha256",
            "workload_instance_sha256",
            "evaluator_contract_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if not isinstance(self.metric_projector, CampaignMetricEffectProjector):
            raise TypeError("metric_projector must implement its narrow port")
        if not isinstance(
            self.action_semantics_compiler,
            CampaignActionSemanticsCompiler,
        ):
            raise TypeError("action_semantics_compiler must implement its narrow port")
        if not isinstance(
            self.hypothesis_matcher,
            GlobalHypothesisEvidenceMatcher,
        ):
            raise TypeError("hypothesis_matcher must implement its narrow port")
        if type(self.falsification_gate) is not GlobalHypothesisFalsificationGate:
            raise TypeError("falsification_gate must be exact")
        GlobalHypothesisFalsificationGate.__post_init__(self.falsification_gate)

    @staticmethod
    def _validate_plans(
        entries: tuple[InsightMemoryEntry, ...],
        plans: tuple[CampaignSemanticAuditPlan, ...],
    ) -> None:
        if type(entries) is not tuple or type(plans) is not tuple:
            raise TypeError("entries/plans must be exact tuples")
        if len(entries) != len(plans):
            raise ValueError("entries and registered plans differ")
        for entry, plan in zip(entries, plans, strict=True):
            if type(entry) is not InsightMemoryEntry:
                raise TypeError("entries must contain exact InsightMemoryEntry values")
            if type(plan) is not CampaignSemanticAuditPlan:
                raise TypeError(
                    "plans must contain exact CampaignSemanticAuditPlan values"
                )
            InsightMemoryEntry.__post_init__(entry)
            CampaignSemanticAuditPlan.__post_init__(plan)
            if entry.reference != plan.reference:
                raise ValueError("registered plan belongs to a different entry")
            if plan.intervention.admissible_operator_families != ("typed_mutation",):
                raise ValueError("production audit MVP supports typed mutation only")
            if any(
                value.comparison_anchor is None
                or value.comparison_anchor.kind
                is not MetricComparisonAnchorKind.CURRENT_PARENT
                for value in entry.draft.effect_predictions
            ):
                raise ValueError("production audit MVP requires current-parent anchors")

    @staticmethod
    def _diagnostic_context_bindings(
        *,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        planned_references: tuple[InsightRef, ...],
    ) -> tuple[CampaignDiagnosticContextBinding, ...]:
        planned = set(planned_references)
        contexts: dict[InsightRef, set[str]] = {}
        assignments: dict[InsightRef, set[str]] = {}
        for wave in waves:
            credit = wave.memory_credit
            if (
                credit is None
                or credit.assignment.arm is not MemoryAssignmentArm.DIAGNOSTIC
            ):
                continue
            # A generic diagnostic assignment that does not carry a bank-issued
            # quarantine admission is not semantic-card lifecycle evidence.
            if credit.quarantine_admission is None:
                continue
            assignment = credit.assignment
            eligible = set(assignment.selection_decision.eligible)
            admitted = set(credit.quarantine_admission.references)
            if credit.quarantine_admission_subset_authorization_sha256 is None:
                planned_admitted = admitted.intersection(planned)
            else:
                # The credit binding carries an explicit prospective cohort-
                # selection receipt.  Only its assignment-eligible subset was
                # exposed, so unselected admitted cards are not semantic-audit
                # subjects for this generation.
                planned_admitted = admitted.intersection(planned, eligible)
                if not planned_admitted:
                    raise ValueError(
                        "authorized quarantine subset has no planned audit subject"
                    )
            if not planned_admitted.issubset(eligible):
                raise ValueError(
                    "planned quarantine admission differs from assignment eligibility"
                )
            # Normal seed/promoted controls can intentionally share the eligible
            # assignment slate.  Only bank-admitted quarantine references are
            # lifecycle audit subjects; admission authenticity was already
            # replayed by PortfolioEvolution against the memory bank.
            for reference in planned_admitted:
                contexts.setdefault(reference, set()).add(assignment.exact_context_hash)
                assignments.setdefault(reference, set()).add(
                    assignment.assignment_sha256
                )
        bindings = []
        for reference in sorted(contexts):
            values = contexts[reference]
            if len(values) != 1:
                raise ValueError(
                    "one diagnostic insight cannot mix exact estimand contexts"
                )
            bindings.append(
                CampaignDiagnosticContextBinding(
                    reference=reference,
                    exact_context_sha256=next(iter(values)),
                    assignment_sha256s=tuple(sorted(assignments[reference])),
                )
            )
        return tuple(bindings)

    @staticmethod
    def _request_from_plan(
        plan: CampaignSemanticAuditPlan,
        *,
        audit_cutoff_event_index: int,
        registry_snapshot_sha256: str,
    ) -> GlobalHypothesisAuditRequest:
        return GlobalHypothesisAuditRequest(
            reference=plan.reference,
            draft_content_sha256=plan.draft_content_sha256,
            trigger=plan.trigger,
            intervention=plan.intervention,
            predictions=plan.predictions,
            claim_strength=plan.claim_strength,
            scope=plan.scope,
            matcher_definition_sha256=plan.matcher_definition_sha256,
            origin_cutoff_event_index=plan.origin_cutoff_event_index,
            audit_cutoff_event_index=audit_cutoff_event_index,
            registry_snapshot_sha256=registry_snapshot_sha256,
            minimum_support_clusters=plan.minimum_support_clusters,
            minimum_support_instances=plan.minimum_support_instances,
            audit_policy_definition_sha256=plan.audit_policy_definition_sha256,
        )

    def prepare_generation_audit(
        self,
        *,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        memory_credit_preparation: PortfolioMemoryCreditBatchPreparation,
        entries: tuple[InsightMemoryEntry, ...] = (),
        plans: tuple[CampaignSemanticAuditPlan, ...] = (),
    ) -> CampaignGenerationAuditPreparation:
        """Prepare evidence for every stage and real audits when assigned."""

        self.__post_init__()
        if type(request) is not CampaignStageRequest:
            raise TypeError("request must be exact CampaignStageRequest")
        CampaignStageRequest.__post_init__(request)
        if request.step.kind is not CampaignGenerationKind.PORTFOLIO:
            raise ValueError(
                "production generation audit accepts portfolio stages only"
            )
        if type(memory_credit_preparation) is not PortfolioMemoryCreditBatchPreparation:
            raise TypeError("memory_credit_preparation must be exact")
        PortfolioMemoryCreditBatchPreparation.__post_init__(memory_credit_preparation)
        if memory_credit_preparation.prepared_results != results:
            raise ValueError("audit results differ from memory preparation")
        self._validate_plans(entries, plans)
        planned_references = tuple(plan.reference for plan in plans)
        if planned_references != tuple(sorted(set(planned_references))):
            raise ValueError("registered audit plans must be canonical and unique")
        hypothesis_evidence = project_portfolio_hypothesis_evidence(
            campaign_sha256=self.campaign_sha256,
            event_index=request.step.generation,
            workload_instance_sha256=self.workload_instance_sha256,
            evaluator_contract_sha256=self.evaluator_contract_sha256,
            waves=waves,
            results=results,
            metric_projector=self.metric_projector,
            semantics_compiler=self.action_semantics_compiler,
        )
        memory_attribution = audit_portfolio_memory_attribution(
            waves=waves,
            results=results,
        )
        matched_outcomes = resolve_portfolio_memory_matched_control_outcomes(
            waves=waves,
            results=results,
        )
        observations = hypothesis_evidence.observations
        append = self.evidence_registry.prepare_append(
            observations,
            captured_through_event_index=request.step.generation,
        )
        try:
            context_bindings = self._diagnostic_context_bindings(
                waves=waves,
                planned_references=planned_references,
            )
            projection = None
            if context_bindings:
                batch = memory_credit_preparation.batch_receipt
                if batch is None:
                    raise ValueError(
                        "diagnostic assignments require a prospective memory batch"
                    )
                plan_by_reference = {value.reference: value for value in plans}
                snapshot = append.prospective_snapshot
                audits = []
                for binding in context_bindings:
                    plan = plan_by_reference[binding.reference]
                    audit_request = self._request_from_plan(
                        plan,
                        audit_cutoff_event_index=request.step.generation,
                        registry_snapshot_sha256=snapshot.snapshot_sha256,
                    )
                    if not plan.admits(audit_request):
                        raise ValueError(
                            "sealed audit request differs from its registered plan"
                        )
                    audit_receipt = self.falsification_gate.audit(
                        request=audit_request,
                        registry=snapshot,
                        matcher=self.hypothesis_matcher,
                    )
                    audits.append(
                        CampaignInsightAuditBinding(
                            request=audit_request,
                            receipt=audit_receipt,
                            exact_context_sha256=binding.exact_context_sha256,
                        )
                    )
                projection = CampaignGenerationAuditProjection(
                    stage_request_sha256=request.request_sha256,
                    generation=request.step.generation,
                    wave_request_sha256s=tuple(
                        sorted(
                            value.selection_request.request_sha256 for value in waves
                        )
                    ),
                    wave_result_receipt_sha256s=tuple(
                        sorted(value.receipt.receipt_sha256 for value in results)
                    ),
                    memory_credit_batch_receipt_sha256=batch.receipt_sha256,
                    registry_snapshot_sha256=snapshot.snapshot_sha256,
                    evidence_append_preparation_sha256=append.preparation_sha256,
                    context_bindings=context_bindings,
                    audits=tuple(audits),
                )
            preparation = CampaignGenerationAuditPreparation(
                stage_request_sha256=request.request_sha256,
                generation=request.step.generation,
                wave_request_sha256s=tuple(
                    sorted(value.selection_request.request_sha256 for value in waves)
                ),
                wave_result_receipt_sha256s=tuple(
                    sorted(value.receipt.receipt_sha256 for value in results)
                ),
                memory_credit_preparation_sha256=(
                    memory_credit_preparation.preparation_sha256
                ),
                observations=observations,
                observation_exclusions=hypothesis_evidence.exclusions,
                hypothesis_evidence_projection=hypothesis_evidence,
                evidence_append=append,
                memory_attribution_audit=memory_attribution,
                matched_memory_control_outcomes=matched_outcomes,
                projection=projection,
            )
        except BaseException:
            self.evidence_registry.abort_append(append)
            raise
        if preparation.preparation_sha256 in self._prepared:
            self.evidence_registry.abort_append(append)
            raise ValueError("generation audit preparation identity collided")
        self._prepared[preparation.preparation_sha256] = preparation
        return preparation

    def commit_generation_audit(
        self,
        preparation: CampaignGenerationAuditPreparation,
    ) -> None:
        if type(preparation) is not CampaignGenerationAuditPreparation:
            raise TypeError("preparation must be exact")
        CampaignGenerationAuditPreparation.__post_init__(preparation)
        stored = self._prepared.get(preparation.preparation_sha256)
        if stored is not preparation:
            raise ValueError("generation audit preparation is unavailable")
        self.evidence_registry.commit_append(preparation.evidence_append)
        del self._prepared[preparation.preparation_sha256]

    def abort_generation_audit(
        self,
        preparation: CampaignGenerationAuditPreparation,
    ) -> None:
        if type(preparation) is not CampaignGenerationAuditPreparation:
            raise TypeError("preparation must be exact")
        self.evidence_registry.abort_append(preparation.evidence_append)
        self._prepared.pop(preparation.preparation_sha256, None)


__all__ = [
    "CampaignDiagnosticContextBinding",
    "CampaignGenerationAuditPreparation",
    "CampaignGenerationAuditProjection",
    "PRODUCTION_GENERATION_AUDITOR_DEFINITION_SHA256",
    "PRODUCTION_GENERATION_AUDITOR_ID",
    "PRODUCTION_GENERATION_AUDITOR_VERSION",
    "TransactionalPortfolioGenerationAuditor",
    "resolve_portfolio_memory_matched_control_outcomes",
]

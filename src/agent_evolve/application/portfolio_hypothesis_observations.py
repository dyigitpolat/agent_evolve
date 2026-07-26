"""Workload-neutral projection of evaluated portfolio actions into evidence.

This module performs the structural join shared by every finite-action
benchmark: an exact parent, ranked option, engine materialization, successful
child, and objective transition become one authenticated hypothesis
observation. Candidate-attributable infeasibility instead becomes an explicit
no-resampling exclusion receipt. Benchmark-specific interpretation is
inverted behind two small
ports: metric-effect projection and action-semantics compilation.

Randomized insight exposure is intentionally not relabelled as randomized
action administration.  These observations audit whether a semantic claim is
consistent with evaluated actions; causal credit for a card remains the
whole-wave memory treatment maintained elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    InvocationOutcome,
    OperatorKind,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioCandidateFailureEvidence,
    PortfolioMemberDisposition,
    PortfolioVariationMemberReceipt,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceProvenance,
    InterventionIdentifiability,
    ObservedMetricEffect,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_PARENT_OUTCOME_DOMAIN = b"agent-evolve:campaign-parent-outcome:v1\x00"
_SEMANTICS_DOMAIN = b"agent-evolve:campaign-observed-action-semantics:v1\x00"
_LINEAGE_DOMAIN = b"agent-evolve:campaign-observation-lineage:v1\x00"
_BLOCK_DOMAIN = b"agent-evolve:campaign-observation-block:v1\x00"
_EXCLUSION_DOMAIN = b"agent-evolve:campaign-observation-exclusion:v1\x00"
_PROJECTION_DOMAIN = b"agent-evolve:campaign-observation-projection:v1\x00"


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


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("campaign action semantics did not freeze to an object")
    return frozen


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def campaign_candidate_outcome_sha256(candidate: EvolutionCandidate) -> str:
    """Stable evidence identity for a parent candidate's evaluated state."""

    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be exact")
    EvolutionCandidate.__post_init__(candidate)
    return _hash(
        _PARENT_OUTCOME_DOMAIN,
        {
            "candidate_id": candidate.candidate_id.value,
            "configuration_sha256": candidate.occurrence.configuration_hash,
            "objectives": [
                [metric_id, value.hex()] for metric_id, value in candidate.objectives
            ],
            "valid": candidate.valid,
            "detailed_evaluation_sha256": (
                None
                if candidate.detailed_evaluation is None
                else candidate.detailed_evaluation.evidence_sha256
            ),
            "objective_resolution_receipt_sha256": (
                None
                if candidate.objective_resolution_receipt is None
                else candidate.objective_resolution_receipt.receipt_sha256
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class CampaignEvaluatedActionContext:
    """Exact finite-action facts offered to a semantic compiler."""

    option_id: str
    option_identity_sha256: str
    option_family: str
    operator_family: str
    changed_paths: tuple[str, ...]
    parent_configuration: FrozenJsonObject
    child_configuration: FrozenJsonObject
    materialization_receipt_sha256: str
    outcome_sha256: str
    finite_contract_identity_sha256: str

    def __post_init__(self) -> None:
        for name in ("option_id", "option_family", "operator_family"):
            _require_token(getattr(self, name), name=name)
        for name in (
            "option_identity_sha256",
            "materialization_receipt_sha256",
            "outcome_sha256",
            "finite_contract_identity_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.changed_paths) is not tuple
            or not self.changed_paths
            or any(
                type(value) is not str or not value.startswith("$.")
                for value in self.changed_paths
            )
            or self.changed_paths != tuple(sorted(set(self.changed_paths)))
        ):
            raise ValueError("changed_paths must be a canonical non-empty tuple")
        for name in ("parent_configuration", "child_configuration"):
            value = getattr(self, name)
            if type(value) is not FrozenJsonObject or freeze_json(value) is not value:
                raise TypeError(f"{name} must be an exact frozen object")


@dataclass(frozen=True, slots=True)
class CampaignObservedActionSemantics:
    """Compiler-issued interpretation bound into an observation payload."""

    observed_action: FrozenJsonObject
    intervention_identifiability: InterventionIdentifiability
    mechanism_identifying_design: bool
    compiler_id: str
    compiler_version: int
    compiler_definition_sha256: str
    semantics_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.observed_action) is not FrozenJsonObject
            or freeze_json(self.observed_action) is not self.observed_action
        ):
            raise TypeError("observed_action must be an exact frozen object")
        if type(self.intervention_identifiability) is not InterventionIdentifiability:
            raise TypeError("intervention_identifiability must be exact")
        if type(self.mechanism_identifying_design) is not bool:
            raise TypeError("mechanism_identifying_design must be exact bool")
        _require_token(self.compiler_id, name="compiler_id")
        if type(self.compiler_version) is not int or self.compiler_version <= 0:
            raise ValueError("compiler_version must be positive")
        require_sha256(
            self.compiler_definition_sha256,
            "compiler_definition_sha256",
        )
        object.__setattr__(
            self,
            "semantics_sha256",
            _hash(
                _SEMANTICS_DOMAIN,
                {
                    "schema_version": 1,
                    "observed_action": thaw_json(self.observed_action),
                    "intervention_identifiability": (
                        self.intervention_identifiability.value
                    ),
                    "mechanism_identifying_design": (self.mechanism_identifying_design),
                    "compiler": {
                        "compiler_id": self.compiler_id,
                        "compiler_version": self.compiler_version,
                        "definition_sha256": self.compiler_definition_sha256,
                    },
                },
            ),
        )


@runtime_checkable
class CampaignActionSemanticsCompiler(Protocol):
    compiler_id: str
    compiler_version: int
    definition_sha256: str

    def compile(
        self,
        context: CampaignEvaluatedActionContext,
    ) -> CampaignObservedActionSemantics: ...


@runtime_checkable
class CampaignMetricEffectProjector(Protocol):
    projector_id: str
    projector_version: int
    definition_sha256: str

    def project(
        self,
        parent: EvolutionCandidate,
        child: EvolutionCandidate,
    ) -> tuple[ObservedMetricEffect, ...]: ...


@dataclass(frozen=True, slots=True)
class FinitePortfolioActionSemanticsCompiler:
    """Default structural semantics for one engine-materialized finite option."""

    compiler_id: str = "finite_portfolio_action_semantics"
    compiler_version: int = 2
    definition_sha256: str = hashlib.sha256(
        b"agent-evolve:finite-portfolio-action-semantics:v2;"
        b"exact-finite-contract-identity"
    ).hexdigest()

    def compile(
        self,
        context: CampaignEvaluatedActionContext,
    ) -> CampaignObservedActionSemantics:
        if type(context) is not CampaignEvaluatedActionContext:
            raise TypeError("context must be exact")
        CampaignEvaluatedActionContext.__post_init__(context)
        action = _object(
            {
                "schema_version": 2,
                "option_id": context.option_id,
                "option_identity_sha256": context.option_identity_sha256,
                "option_family": context.option_family,
                "operator_family": context.operator_family,
                "changed_paths": list(context.changed_paths),
                "materialization_receipt_sha256": (
                    context.materialization_receipt_sha256
                ),
                "outcome_sha256": context.outcome_sha256,
                "finite_contract_identity_sha256": (
                    context.finite_contract_identity_sha256
                ),
                "compiler": {
                    "compiler_id": self.compiler_id,
                    "compiler_version": self.compiler_version,
                    "definition_sha256": self.definition_sha256,
                },
            }
        )
        return CampaignObservedActionSemantics(
            observed_action=action,
            intervention_identifiability=(InterventionIdentifiability.EXACT_SINGLE),
            mechanism_identifying_design=False,
            compiler_id=self.compiler_id,
            compiler_version=self.compiler_version,
            compiler_definition_sha256=self.definition_sha256,
        )


@dataclass(frozen=True, slots=True)
class ObjectiveDeltaMetricEffectProjector:
    """Project exact raw child-minus-parent objective transitions."""

    adjudicator_definition_sha256: str
    projector_id: str = "objective_delta_metric_effects"
    projector_version: int = 1
    definition_sha256: str = hashlib.sha256(
        b"agent-evolve:objective-delta-metric-effects:v1"
    ).hexdigest()

    def __post_init__(self) -> None:
        require_sha256(
            self.adjudicator_definition_sha256,
            "adjudicator_definition_sha256",
        )

    def project(
        self,
        parent: EvolutionCandidate,
        child: EvolutionCandidate,
    ) -> tuple[ObservedMetricEffect, ...]:
        self.__post_init__()
        for name, value in (("parent", parent), ("child", child)):
            if type(value) is not EvolutionCandidate:
                raise TypeError(f"{name} must be exact")
            EvolutionCandidate.__post_init__(value)
        parent_metrics = parent.objective_map
        child_metrics = child.objective_map
        if set(parent_metrics) != set(child_metrics):
            raise ValueError("parent and child objective sets differ")
        effects = []
        for metric_id in sorted(parent_metrics):
            delta = float(child_metrics[metric_id] - parent_metrics[metric_id])
            if not math.isfinite(delta):
                raise ValueError("objective delta must be finite")
            direction = (
                MetricEffectDirection.INCREASE
                if delta > 0.0
                else (
                    MetricEffectDirection.DECREASE
                    if delta < 0.0
                    else MetricEffectDirection.UNCHANGED
                )
            )
            effects.append(
                ObservedMetricEffect(
                    metric_id=metric_id,
                    direction=direction,
                    delta=delta,
                    adjudicator_definition_sha256=(self.adjudicator_definition_sha256),
                )
            )
        return tuple(effects)


def _action_context(
    *,
    wave: PortfolioVariationWaveRequest,
    result: PortfolioVariationWaveResult,
    member: PortfolioVariationMemberReceipt,
    outcome: InvocationOutcome,
    rank_index: int,
) -> CampaignEvaluatedActionContext:
    decision = result.selection_decision
    if decision is None:
        raise ValueError("hypothesis evidence requires a typed ranked decision")
    selected = decision.members[rank_index]
    materialization = member.materialization
    child = outcome.candidate
    if child is None:  # Closed by PortfolioVariationWaveResult validation.
        raise AssertionError("validated portfolio result lost its child")
    parent_configuration = wave.parent.configuration
    child_configuration = child.configuration
    if type(parent_configuration) is not FrozenJsonObject:
        raise TypeError("portfolio parent configuration must be an object")
    if type(child_configuration) is not FrozenJsonObject:
        raise TypeError("portfolio child configuration must be an object")
    operator = outcome.prepared.plan.operator_kind
    if operator is not OperatorKind.TYPED_MUTATION:
        raise ValueError("portfolio action evidence requires typed mutation")
    return CampaignEvaluatedActionContext(
        option_id=selected.option_id,
        option_identity_sha256=selected.option_identity_sha256,
        option_family=selected.family,
        operator_family=operator.value,
        changed_paths=materialization.changed_paths,
        parent_configuration=parent_configuration,
        child_configuration=child_configuration,
        materialization_receipt_sha256=materialization.receipt_sha256,
        outcome_sha256=member.outcome_sha256,
        finite_contract_identity_sha256=(
            wave.selection_request.finite_variation_contract.identity_sha256
        ),
    )


class CampaignHypothesisObservationExclusionReason(str, Enum):
    """Closed reason a ranked action cannot become metric-effect evidence."""

    CANDIDATE_INFEASIBLE = "candidate_infeasible"


@dataclass(frozen=True, slots=True)
class CampaignHypothesisObservationExclusion:
    """Authenticated ITT action excluded from semantic metric adjudication."""

    source_evidence_id: str
    wave_receipt_sha256: str
    request_sha256: str
    rank: int
    parent_candidate_id: str
    candidate_id: str
    candidate_configuration_sha256: str
    option_id: str
    option_identity_sha256: str
    candidate_failure: PortfolioCandidateFailureEvidence
    reason: CampaignHypothesisObservationExclusionReason = (
        CampaignHypothesisObservationExclusionReason.CANDIDATE_INFEASIBLE
    )
    exclusion_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_evidence_id",
            "wave_receipt_sha256",
            "request_sha256",
            "candidate_configuration_sha256",
            "option_identity_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        for name in ("parent_candidate_id", "candidate_id", "option_id"):
            _require_token(getattr(self, name), name=name)
        if self.parent_candidate_id == self.candidate_id:
            raise ValueError("excluded child cannot reuse its parent occurrence")
        if type(self.candidate_failure) is not PortfolioCandidateFailureEvidence:
            raise TypeError("candidate_failure must be exact")
        PortfolioCandidateFailureEvidence.__post_init__(self.candidate_failure)
        if type(self.reason) is not CampaignHypothesisObservationExclusionReason:
            raise TypeError("reason must be exact")
        object.__setattr__(
            self,
            "exclusion_sha256",
            _hash(_EXCLUSION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_evidence_id": self.source_evidence_id,
            "wave_receipt_sha256": self.wave_receipt_sha256,
            "request_sha256": self.request_sha256,
            "rank": self.rank,
            "parent_candidate_id": self.parent_candidate_id,
            "candidate_id": self.candidate_id,
            "candidate_configuration_sha256": self.candidate_configuration_sha256,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "reason": self.reason.value,
            "candidate_failure": self.candidate_failure.to_record(),
            "metric_projection_executed": False,
            "semantics_compiler_executed": False,
            "resampled": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "exclusion_sha256": self.exclusion_sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignPortfolioHypothesisEvidenceProjection:
    """Complete scored/excluded partition of all ranked ITT action members."""

    wave_receipt_sha256s: tuple[str, ...]
    ranked_source_evidence_ids: tuple[str, ...]
    observations: tuple[AuthenticatedHypothesisObservation, ...]
    exclusions: tuple[CampaignHypothesisObservationExclusion, ...]
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.wave_receipt_sha256s) is not tuple or not self.wave_receipt_sha256s:
            raise ValueError("wave_receipt_sha256s must be a non-empty exact tuple")
        for value in self.wave_receipt_sha256s:
            require_sha256(value, "wave_receipt_sha256s")
        if self.wave_receipt_sha256s != tuple(
            sorted(set(self.wave_receipt_sha256s))
        ):
            raise ValueError("wave receipt identities must be canonical and unique")
        if (
            type(self.ranked_source_evidence_ids) is not tuple
            or not self.ranked_source_evidence_ids
        ):
            raise ValueError("ranked source evidence IDs must be non-empty")
        for value in self.ranked_source_evidence_ids:
            require_sha256(value, "ranked_source_evidence_ids")
        if len(set(self.ranked_source_evidence_ids)) != len(
            self.ranked_source_evidence_ids
        ):
            raise ValueError("ranked source evidence IDs must be unique")
        if type(self.observations) is not tuple or any(
            type(value) is not AuthenticatedHypothesisObservation
            for value in self.observations
        ):
            raise TypeError("observations must contain exact authenticated evidence")
        for value in self.observations:
            AuthenticatedHypothesisObservation.__post_init__(value)
        observation_ids = tuple(
            value.source_evidence_id for value in self.observations
        )
        if observation_ids != tuple(sorted(set(observation_ids))):
            raise ValueError("observations must use canonical unique source evidence")
        if type(self.exclusions) is not tuple or any(
            type(value) is not CampaignHypothesisObservationExclusion
            for value in self.exclusions
        ):
            raise TypeError("exclusions must contain exact receipts")
        for value in self.exclusions:
            CampaignHypothesisObservationExclusion.__post_init__(value)
        exclusion_ids = tuple(value.source_evidence_id for value in self.exclusions)
        if exclusion_ids != tuple(sorted(set(exclusion_ids))):
            raise ValueError("exclusions must use canonical unique source evidence")
        if set(observation_ids).intersection(exclusion_ids) or set(
            self.ranked_source_evidence_ids
        ) != set((*observation_ids, *exclusion_ids)):
            raise ValueError(
                "observations and exclusions must exactly partition ranked ITT members"
            )
        if any(
            value.wave_receipt_sha256 not in self.wave_receipt_sha256s
            for value in self.exclusions
        ):
            raise ValueError("exclusion belongs to a foreign wave")
        object.__setattr__(
            self,
            "projection_sha256",
            _hash(_PROJECTION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "wave_receipt_sha256s": list(self.wave_receipt_sha256s),
            "ranked_source_evidence_ids": list(self.ranked_source_evidence_ids),
            "observation_sha256s": [
                value.observation_sha256 for value in self.observations
            ],
            "exclusion_sha256s": [
                value.exclusion_sha256 for value in self.exclusions
            ],
            "ranked_itt_member_count": len(self.ranked_source_evidence_ids),
            "observation_count": len(self.observations),
            "exclusion_count": len(self.exclusions),
            "resampled_member_count": 0,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "exclusions": [value.to_record() for value in self.exclusions],
            "projection_sha256": self.projection_sha256,
        }


def project_portfolio_hypothesis_evidence(
    *,
    campaign_sha256: str,
    event_index: int,
    workload_instance_sha256: str,
    evaluator_contract_sha256: str,
    waves: tuple[PortfolioVariationWaveRequest, ...],
    results: tuple[PortfolioVariationWaveResult, ...],
    metric_projector: CampaignMetricEffectProjector,
    semantics_compiler: CampaignActionSemanticsCompiler,
) -> CampaignPortfolioHypothesisEvidenceProjection:
    """Partition every ranked finite action into evidence or typed exclusion."""

    for name in (
        "campaign_sha256",
        "workload_instance_sha256",
        "evaluator_contract_sha256",
    ):
        require_sha256(locals()[name], name)
    if type(event_index) is not int or event_index <= 0:
        raise ValueError("event_index must be positive")
    if (
        type(waves) is not tuple
        or type(results) is not tuple
        or not waves
        or len(waves) != len(results)
    ):
        raise ValueError("waves/results must be equal non-empty tuples")
    if not isinstance(metric_projector, CampaignMetricEffectProjector):
        raise TypeError("metric_projector must implement its port")
    if not isinstance(semantics_compiler, CampaignActionSemanticsCompiler):
        raise TypeError("semantics_compiler must implement its port")
    observations: list[AuthenticatedHypothesisObservation] = []
    exclusions: list[CampaignHypothesisObservationExclusion] = []
    ranked_source_evidence_ids: list[str] = []
    wave_receipt_sha256s: list[str] = []
    for wave, result in zip(waves, results, strict=True):
        if type(wave) is not PortfolioVariationWaveRequest:
            raise TypeError("waves must contain exact requests")
        if type(result) is not PortfolioVariationWaveResult:
            raise TypeError("results must contain exact results")
        PortfolioVariationWaveRequest.__post_init__(wave)
        PortfolioVariationWaveResult.__post_init__(result)
        if result.receipt.request_sha256 != wave.selection_request.request_sha256:
            raise ValueError("portfolio result belongs to a foreign wave")
        wave_receipt_sha256s.append(result.receipt.receipt_sha256)
        decision = result.selection_decision
        if decision is None:
            raise ValueError("portfolio result omitted its ranked decision")
        for rank_index, (member, outcome) in enumerate(
            zip(result.receipt.members, result.outcomes, strict=True)
        ):
            ranked_source_evidence_ids.append(member.outcome_sha256)
            child = outcome.candidate
            if child is None:  # Closed by exact result validation.
                raise AssertionError("validated result lost a child")
            if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
                failure = member.candidate_failure
                if type(failure) is not PortfolioCandidateFailureEvidence:
                    raise AssertionError(
                        "validated infeasible member lost candidate failure evidence"
                    )
                selected = decision.members[rank_index]
                materialization = member.materialization
                exclusions.append(
                    CampaignHypothesisObservationExclusion(
                        source_evidence_id=member.outcome_sha256,
                        wave_receipt_sha256=result.receipt.receipt_sha256,
                        request_sha256=wave.selection_request.request_sha256,
                        rank=materialization.rank,
                        parent_candidate_id=wave.parent.candidate_id.value,
                        candidate_id=child.candidate_id.value,
                        candidate_configuration_sha256=(
                            child.occurrence.configuration_hash
                        ),
                        option_id=selected.option_id,
                        option_identity_sha256=selected.option_identity_sha256,
                        candidate_failure=failure,
                    )
                )
                continue
            context = _action_context(
                wave=wave,
                result=result,
                member=member,
                outcome=outcome,
                rank_index=rank_index,
            )
            semantics = semantics_compiler.compile(context)
            if type(semantics) is not CampaignObservedActionSemantics:
                raise TypeError("semantics compiler returned a foreign value")
            CampaignObservedActionSemantics.__post_init__(semantics)
            expected_compiler = (
                semantics_compiler.compiler_id,
                semantics_compiler.compiler_version,
                semantics_compiler.definition_sha256,
            )
            observed_compiler = (
                semantics.compiler_id,
                semantics.compiler_version,
                semantics.compiler_definition_sha256,
            )
            if observed_compiler != expected_compiler:
                raise ValueError("action semantics compiler identity changed")
            metrics = metric_projector.project(wave.parent, child)
            if (
                type(metrics) is not tuple
                or not metrics
                or any(type(value) is not ObservedMetricEffect for value in metrics)
            ):
                raise TypeError("metric projector returned foreign effects")
            for value in metrics:
                ObservedMetricEffect.__post_init__(value)
            parent_configuration = context.parent_configuration
            child_configuration = context.child_configuration
            credit = wave.memory_credit
            block_identity = (
                result.receipt.receipt_sha256
                if credit is None
                else credit.assignment.block_id
            )
            observations.append(
                AuthenticatedHypothesisObservation(
                    source_evidence_id=member.outcome_sha256,
                    event_index=event_index,
                    workload_instance_sha256=workload_instance_sha256,
                    evaluator_contract_sha256=evaluator_contract_sha256,
                    campaign_sha256=campaign_sha256,
                    parent_candidate_id=wave.parent.candidate_id,
                    child_candidate_id=child.candidate_id,
                    operator_invocation_id=member.operator_invocation_id,
                    finite_contract_identity_sha256=(
                        context.finite_contract_identity_sha256
                    ),
                    provenance=EvidenceProvenance.DIRECT_MUTATION,
                    causal_boundary=EvidenceCausalBoundary(
                        wave_sha256=result.receipt.receipt_sha256,
                        estimand_unit=CausalEstimandUnit.WAVE,
                    ),
                    parent_configuration=parent_configuration,
                    child_configuration=child_configuration,
                    parent_configuration_sha256=(
                        AuthenticatedHypothesisObservation.configuration_sha256(
                            parent_configuration
                        )
                    ),
                    child_configuration_sha256=(
                        AuthenticatedHypothesisObservation.configuration_sha256(
                            child_configuration
                        )
                    ),
                    parent_outcome_sha256=campaign_candidate_outcome_sha256(
                        wave.parent
                    ),
                    child_outcome_sha256=member.outcome_sha256,
                    operator_family=context.operator_family,
                    affected_paths=context.changed_paths,
                    observed_action=semantics.observed_action,
                    action_semantics_compiler_id=semantics.compiler_id,
                    action_semantics_compiler_version=semantics.compiler_version,
                    action_semantics_definition_sha256=(
                        semantics.compiler_definition_sha256
                    ),
                    intervention_identifiability=(
                        semantics.intervention_identifiability
                    ),
                    metrics=metrics,
                    lineage_cluster_sha256=_hash(
                        _LINEAGE_DOMAIN,
                        {
                            "campaign_sha256": campaign_sha256,
                            "parent_candidate_id": wave.parent.candidate_id.value,
                            "parent_ids": [
                                value.value for value in wave.parent.parent_ids
                            ],
                            "common_ancestor_id": (
                                None
                                if wave.parent.common_ancestor_id is None
                                else wave.parent.common_ancestor_id.value
                            ),
                        },
                    ),
                    factorial_block_sha256=_hash(
                        _BLOCK_DOMAIN,
                        {
                            "campaign_sha256": campaign_sha256,
                            "event_index": event_index,
                            "block_identity": block_identity,
                        },
                    ),
                    mechanism_identifying_design=(
                        semantics.mechanism_identifying_design
                    ),
                )
            )
    canonical = tuple(sorted(observations, key=lambda value: value.source_evidence_id))
    if len({value.source_evidence_id for value in canonical}) != len(canonical):
        raise ValueError("portfolio observations repeat source evidence")
    canonical_exclusions = tuple(
        sorted(exclusions, key=lambda value: value.source_evidence_id)
    )
    return CampaignPortfolioHypothesisEvidenceProjection(
        wave_receipt_sha256s=tuple(sorted(wave_receipt_sha256s)),
        ranked_source_evidence_ids=tuple(ranked_source_evidence_ids),
        observations=canonical,
        exclusions=canonical_exclusions,
    )


def project_portfolio_hypothesis_observations(
    *,
    campaign_sha256: str,
    event_index: int,
    workload_instance_sha256: str,
    evaluator_contract_sha256: str,
    waves: tuple[PortfolioVariationWaveRequest, ...],
    results: tuple[PortfolioVariationWaveResult, ...],
    metric_projector: CampaignMetricEffectProjector,
    semantics_compiler: CampaignActionSemanticsCompiler,
) -> tuple[AuthenticatedHypothesisObservation, ...]:
    """Backward-compatible scored-observation projection.

    Production generation audit uses :func:`project_portfolio_hypothesis_evidence`
    so candidate-infeasible ranked actions also receive explicit exclusion
    receipts.  This narrow view remains for callers that consume only semantic
    observations.
    """

    return project_portfolio_hypothesis_evidence(
        campaign_sha256=campaign_sha256,
        event_index=event_index,
        workload_instance_sha256=workload_instance_sha256,
        evaluator_contract_sha256=evaluator_contract_sha256,
        waves=waves,
        results=results,
        metric_projector=metric_projector,
        semantics_compiler=semantics_compiler,
    ).observations


__all__ = [
    "CampaignActionSemanticsCompiler",
    "CampaignEvaluatedActionContext",
    "CampaignHypothesisObservationExclusion",
    "CampaignHypothesisObservationExclusionReason",
    "CampaignMetricEffectProjector",
    "CampaignObservedActionSemantics",
    "CampaignPortfolioHypothesisEvidenceProjection",
    "FinitePortfolioActionSemanticsCompiler",
    "ObjectiveDeltaMetricEffectProjector",
    "campaign_candidate_outcome_sha256",
    "project_portfolio_hypothesis_evidence",
    "project_portfolio_hypothesis_observations",
]

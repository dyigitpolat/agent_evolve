"""Card-blind empirical feedback from evaluated portfolio actions.

The selector's complete card/action attribution remains in its internal audit.
This module exposes only treatment-neutral action facts to later generations:
sealed option identity, categorical forecasts, meaningful observed directions,
correctness, parent relation, and reward. Candidate-attributable infeasibility
is retained as a no-yield action but is never converted into metric calibration
or a parent relation. Consequently a currently unassigned card cannot leak back
through outcome history.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.decision_metric_projection import (
    project_candidate_decision_metrics,
)
from agent_evolve.application.contextual_campaign_outcomes import (
    ContextualPortfolioOutcomeBatch,
    observe_contextual_portfolio_outcomes,
)
from agent_evolve.application.contextual_search_controller import (
    ContextualSearchLedger,
)
from agent_evolve.application.outcome_relation import OutcomeRelation
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilitySnapshot,
    CampaignGenerationKind,
)
from agent_evolve.application.campaign_execution import CampaignStageRequest
from agent_evolve.application.portfolio_evolution import (
    PortfolioCandidateFailureEvidence,
    PortfolioMemberDisposition,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.selection.forecast_calibration import (
    BetaCorrectnessPrior,
    ForecastCalibrationObservation,
    ForecastCalibrationScope,
    ForecastCalibrationSnapshot,
    ForecastPredictionReceipt,
    MeaningfulDirectionRequest,
    MeaningfulMetricDirectionAdjudicator,
    build_calibration_snapshot,
    observe_forecast,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationRealization,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_CANONICAL_JSON_PATH = re.compile(
    r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$"
)
_CANDIDATE_OUTCOME_DOMAIN = b"agent-evolve:portfolio-parent-outcome:v1\x00"
_ACTION_DOMAIN = b"agent-evolve:portfolio-action-feedback:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:portfolio-outcome-feedback:v1\x00"
_METRIC_TRANSITION_DOMAIN = b"agent-evolve:decision-metric-transition:v1\x00"
_CONTEXTUAL_QUERY_DOMAIN = b"agent-evolve:contextual-outcome-query:v1\x00"
_CONTEXTUAL_HISTORY_DOMAIN = b"agent-evolve:contextual-outcome-history:v1\x00"


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


def _candidate_outcome_sha256(candidate: EvolutionCandidate) -> str:
    EvolutionCandidate.__post_init__(candidate)
    return _hash(
        _CANDIDATE_OUTCOME_DOMAIN,
        {
            "candidate_id": candidate.candidate_id.value,
            "configuration_sha256": candidate.occurrence.configuration_hash,
            "objectives": [
                [metric_id, value.hex()] for metric_id, value in candidate.objectives
            ],
            "valid": candidate.valid,
            "operator_compliant": candidate.operator_compliant,
            "evidence_compliant": candidate.evidence_compliant,
            "detailed_evaluation_sha256": (
                None
                if candidate.detailed_evaluation is None
                else candidate.detailed_evaluation.evidence_sha256
            ),
        },
    )


def _canonical_tokens(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if type(values) is not tuple or any(
        type(value) is not str or _TOKEN.fullmatch(value) is None for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of closed tokens")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")
    return values


def _canonical_paths(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if type(values) is not tuple or any(
        type(value) is not str or _CANONICAL_JSON_PATH.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(f"{name} must be an exact candidate-path tuple")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")
    return values


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(second + ".")
        or first.startswith(second + "[")
        or second.startswith(first + ".")
        or second.startswith(first + "[")
    )


@dataclass(frozen=True, slots=True)
class DecisionMetricTransition:
    """One engine-authored parent/child metric transition.

    Direction-only history is insufficient for contextual transfer: the same
    target can be an increase for one parent and a decrease for another.  This
    receipt preserves the exact numeric baseline and child value while keeping
    the benchmark-owned meaningful-direction adjudication authoritative.
    """

    metric_id: str
    parent_value: float
    child_value: float
    actual_direction: MetricEffectDirection
    adjudication_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or _TOKEN.fullmatch(self.metric_id) is None:
            raise ValueError("metric_id must use the closed token grammar")
        for name in ("parent_value", "child_value"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
        if (
            type(self.actual_direction) is not MetricEffectDirection
            or self.actual_direction is MetricEffectDirection.UNKNOWN
        ):
            raise ValueError("actual_direction must be a known metric direction")
        require_sha256(
            self.adjudication_receipt_sha256,
            "adjudication_receipt_sha256",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "metric_id": self.metric_id,
            "parent_value_hex": self.parent_value.hex(),
            "child_value_hex": self.child_value.hex(),
            "delta_hex": (self.child_value - self.parent_value).hex(),
            "actual_direction": self.actual_direction.value,
            "adjudication_receipt_sha256": self.adjudication_receipt_sha256,
        }

    @property
    def transition_sha256(self) -> str:
        return _hash(_METRIC_TRANSITION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "transition_sha256": self.transition_sha256,
        }


class OutcomeTransferScope(str, Enum):
    """How far an historical action is transferred to the current parent."""

    SAME_PARENT = "same_parent"
    SAME_LINEAGE = "same_lineage"
    CROSS_LINEAGE_ANALOGY = "cross_lineage_analogy"


@dataclass(frozen=True, slots=True)
class ContextualOutcomeQuery:
    """Deterministic prior-only retrieval with cross-lineage transfer opt-in."""

    current_parent_candidate_id: str
    current_parent_configuration_sha256: str
    cutoff_wave_index_exclusive: int
    lineage_candidate_ids: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    max_actions: int = 24
    include_cross_lineage_analogies: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.current_parent_candidate_id) is not str
            or _TOKEN.fullmatch(self.current_parent_candidate_id) is None
        ):
            raise ValueError(
                "current_parent_candidate_id must use the closed token grammar"
            )
        require_sha256(
            self.current_parent_configuration_sha256,
            "current_parent_configuration_sha256",
        )
        if (
            type(self.cutoff_wave_index_exclusive) is not int
            or self.cutoff_wave_index_exclusive <= 0
        ):
            raise ValueError("cutoff_wave_index_exclusive must be positive")
        _canonical_tokens(self.lineage_candidate_ids, name="lineage_candidate_ids")
        _canonical_tokens(self.families, name="families")
        _canonical_paths(self.changed_paths, name="changed_paths")
        if type(self.max_actions) is not int or self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if type(self.include_cross_lineage_analogies) is not bool:
            raise TypeError("include_cross_lineage_analogies must be an exact bool")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "current_parent_candidate_id": self.current_parent_candidate_id,
            "current_parent_configuration_sha256": (
                self.current_parent_configuration_sha256
            ),
            "cutoff_wave_index_exclusive": self.cutoff_wave_index_exclusive,
            "lineage_candidate_ids": list(self.lineage_candidate_ids),
            "families": list(self.families),
            "changed_paths": list(self.changed_paths),
            "max_actions": self.max_actions,
            "include_cross_lineage_analogies": (self.include_cross_lineage_analogies),
        }

    @property
    def query_sha256(self) -> str:
        return _hash(_CONTEXTUAL_QUERY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "query_sha256": self.query_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioActionOutcomeFeedback:
    """One terminal selected action with no memory-card or rationale fields.

    A scored action carries calibrated metric transitions and one parent
    relation.  Candidate-attributable infeasibility instead remains as a
    no-yield action: it is useful search history, but cannot be laundered into
    forecast calibration, a metric transition, or a parent comparison.
    """

    wave_index: int
    request_sha256: str
    ranked_decision_sha256: str
    proposal_sha256: str
    parent_candidate_id: str
    parent_candidate_identity_sha256: str
    parent_outcome_sha256: str
    candidate_id: str
    candidate_outcome_sha256: str
    option_id: str
    option_identity_sha256: str
    family: str
    changed_paths: tuple[str, ...]
    observations: tuple[ForecastCalibrationObservation, ...]
    parent_relation: OutcomeRelation | None
    reward: float
    dominates_parent: bool
    better_than_parent: bool
    metric_transitions: tuple[DecisionMetricTransition, ...] = ()
    disposition: PortfolioMemberDisposition = PortfolioMemberDisposition.SCORED
    candidate_failure: PortfolioCandidateFailureEvidence | None = None

    def __post_init__(self) -> None:
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be a positive exact integer")
        for name in (
            "request_sha256",
            "ranked_decision_sha256",
            "proposal_sha256",
            "parent_candidate_identity_sha256",
            "parent_outcome_sha256",
            "candidate_outcome_sha256",
            "option_identity_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in ("parent_candidate_id", "candidate_id", "option_id", "family"):
            value = getattr(self, name)
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError(f"{name} must use the closed token grammar")
        if self.parent_candidate_id == self.candidate_id:
            raise ValueError("action feedback child cannot equal its parent")
        if type(self.changed_paths) is not tuple or any(
            type(value) is not str or not value.startswith("$.")
            for value in self.changed_paths
        ):
            raise TypeError("changed_paths must be an exact candidate-path tuple")
        if not self.changed_paths or self.changed_paths != tuple(
            sorted(set(self.changed_paths))
        ):
            raise ValueError("changed_paths must be non-empty, unique, and canonical")
        if type(self.observations) is not tuple or any(
            type(value) is not ForecastCalibrationObservation
            for value in self.observations
        ):
            raise TypeError("observations must contain exact calibration observations")
        for value in self.observations:
            value.revalidate()
        metric_ids = tuple(value.prediction.metric_id for value in self.observations)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("observations must use unique canonical metric order")
        for value in self.observations:
            prediction = value.prediction
            if (
                prediction.wave_index != self.wave_index
                or prediction.selector_decision_sha256 != self.proposal_sha256
                or prediction.parent_candidate_identity_sha256
                != self.parent_candidate_identity_sha256
                or prediction.option_id != self.option_id
                or prediction.option_identity_sha256 != self.option_identity_sha256
                or prediction.family != self.family
                or value.adjudication.parent_outcome_sha256
                != self.parent_outcome_sha256
                or value.adjudication.child_outcome_sha256
                != self.candidate_outcome_sha256
            ):
                raise ValueError("forecast observation belongs to a foreign action")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        if (
            type(self.dominates_parent) is not bool
            or type(self.better_than_parent) is not bool
        ):
            raise TypeError("parent-comparison projections must be exact bools")
        if type(self.metric_transitions) is not tuple or any(
            type(value) is not DecisionMetricTransition
            for value in self.metric_transitions
        ):
            raise TypeError(
                "metric_transitions must contain exact DecisionMetricTransition values"
            )
        for value in self.metric_transitions:
            DecisionMetricTransition.__post_init__(value)
        transition_metric_ids = tuple(
            value.metric_id for value in self.metric_transitions
        )
        if transition_metric_ids and transition_metric_ids != metric_ids:
            raise ValueError(
                "metric transitions must exactly cover observation metric order"
            )
        if self.metric_transitions:
            for transition, observation in zip(
                self.metric_transitions,
                self.observations,
                strict=True,
            ):
                numeric_request = MeaningfulDirectionRequest(
                    benchmark_sha256=(observation.prediction.scope.benchmark_sha256),
                    session_sha256=observation.prediction.scope.session_sha256,
                    wave_index=self.wave_index,
                    parent_candidate_identity_sha256=(
                        self.parent_candidate_identity_sha256
                    ),
                    option_id=self.option_id,
                    option_identity_sha256=self.option_identity_sha256,
                    metric_id=transition.metric_id,
                    parent_outcome_sha256=self.parent_outcome_sha256,
                    child_outcome_sha256=self.candidate_outcome_sha256,
                    parent_metric_value=transition.parent_value,
                    child_metric_value=transition.child_value,
                )
                try:
                    observation.adjudication.require_request(numeric_request)
                except ValueError as error:
                    raise ValueError(
                        "metric transition numeric values differ from its "
                        "adjudication request"
                    ) from error
                if (
                    transition.actual_direction
                    is not observation.adjudication.actual_direction
                    or transition.adjudication_receipt_sha256
                    != observation.adjudication.receipt_sha256
                ):
                    raise ValueError(
                        "metric transition differs from its adjudication receipt"
                    )
        if type(self.disposition) is not PortfolioMemberDisposition:
            raise TypeError("disposition must be exact PortfolioMemberDisposition")
        failure = self.candidate_failure
        if self.disposition is PortfolioMemberDisposition.SCORED:
            if not self.observations:
                raise ValueError("scored action requires calibration observations")
            if type(self.parent_relation) is not OutcomeRelation:
                raise TypeError("scored action requires one exact parent relation")
            if failure is not None:
                raise ValueError("scored action cannot carry candidate failure evidence")
        else:
            if self.observations or self.metric_transitions:
                raise ValueError(
                    "candidate-infeasible action cannot publish metric calibration"
                )
            if self.parent_relation is not None:
                raise ValueError(
                    "candidate-infeasible action cannot publish a parent relation"
                )
            if self.dominates_parent or self.better_than_parent:
                raise ValueError(
                    "candidate-infeasible action cannot publish improvement flags"
                )
            if type(failure) is not PortfolioCandidateFailureEvidence:
                raise TypeError(
                    "candidate-infeasible action requires exact failure evidence"
                )
            PortfolioCandidateFailureEvidence.__post_init__(failure)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "schema_version": 2 if self.metric_transitions else 1,
            "wave_index": self.wave_index,
            "request_sha256": self.request_sha256,
            "ranked_decision_sha256": self.ranked_decision_sha256,
            "proposal_sha256": self.proposal_sha256,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_candidate_identity_sha256": (self.parent_candidate_identity_sha256),
            "parent_outcome_sha256": self.parent_outcome_sha256,
            "candidate_id": self.candidate_id,
            "candidate_outcome_sha256": self.candidate_outcome_sha256,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "changed_paths": list(self.changed_paths),
            "observations": [value.to_record() for value in self.observations],
            "parent_relation": (
                None if self.parent_relation is None else self.parent_relation.value
            ),
            "reward_hex": self.reward.hex(),
            "dominates_parent": self.dominates_parent,
            "better_than_parent": self.better_than_parent,
        }
        if self.metric_transitions:
            record["metric_transitions"] = [
                value.to_record() for value in self.metric_transitions
            ]
        if self.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
            failure = self.candidate_failure
            assert type(failure) is PortfolioCandidateFailureEvidence
            record.update(
                {
                    "schema_version": 3,
                    "disposition": self.disposition.value,
                    "yield_status": "candidate_infeasible_no_yield",
                    "candidate_failure": failure.to_record(),
                    "calibration_excluded": True,
                    "parent_relation_excluded": True,
                }
            )
        return record

    @property
    def feedback_sha256(self) -> str:
        return _hash(_ACTION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "feedback_sha256": self.feedback_sha256}

    def to_prompt_record(self) -> dict[str, object]:
        """Render only treatment-neutral empirical facts for future selectors."""

        self.__post_init__()
        transitions = {value.metric_id: value for value in self.metric_transitions}
        if self.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
            failure = self.candidate_failure
            assert type(failure) is PortfolioCandidateFailureEvidence
            return {
                "wave_index": self.wave_index,
                "option_id": self.option_id,
                "option_identity_sha256": self.option_identity_sha256,
                "family": self.family,
                "changed_paths": list(self.changed_paths),
                "outcome_status": "candidate_infeasible_no_yield",
                "failure_code": failure.failure_code.value,
                "failure_evidence_sha256": failure.evidence_sha256,
                "metric_feedback": [],
                "parent_relation": None,
                "reward_hex": self.reward.hex(),
                "dominates_parent": False,
                "better_than_parent": False,
            }
        return {
            "wave_index": self.wave_index,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "changed_paths": list(self.changed_paths),
            "metric_feedback": [
                {
                    "metric_id": value.prediction.metric_id,
                    "predicted_direction": value.prediction.asserted_direction.value,
                    "confidence": value.prediction.confidence.value,
                    "observed_direction": value.adjudication.actual_direction.value,
                    "correctness": value.correctness,
                    **(
                        {}
                        if value.prediction.metric_id not in transitions
                        else {
                            "parent_value_hex": transitions[
                                value.prediction.metric_id
                            ].parent_value.hex(),
                            "child_value_hex": transitions[
                                value.prediction.metric_id
                            ].child_value.hex(),
                            "delta_hex": (
                                transitions[value.prediction.metric_id].child_value
                                - transitions[value.prediction.metric_id].parent_value
                            ).hex(),
                        }
                    ),
                }
                for value in self.observations
            ],
            "parent_relation": self.parent_relation.value,
            "reward_hex": self.reward.hex(),
            "dominates_parent": self.dominates_parent,
            "better_than_parent": self.better_than_parent,
        }


def _classify_outcome_transfer_scope(
    action: PortfolioActionOutcomeFeedback,
    query: ContextualOutcomeQuery,
) -> OutcomeTransferScope:
    """Derive transfer distance solely from authenticated action/query facts."""

    if action.parent_candidate_id == query.current_parent_candidate_id:
        return OutcomeTransferScope.SAME_PARENT
    if action.candidate_id == query.current_parent_candidate_id:
        return OutcomeTransferScope.SAME_LINEAGE
    if (
        action.parent_candidate_identity_sha256
        == query.current_parent_configuration_sha256
    ):
        return OutcomeTransferScope.SAME_PARENT
    lineage = set(query.lineage_candidate_ids)
    if action.parent_candidate_id in lineage or action.candidate_id in lineage:
        return OutcomeTransferScope.SAME_LINEAGE
    return OutcomeTransferScope.CROSS_LINEAGE_ANALOGY


@dataclass(frozen=True, slots=True)
class PortfolioOutcomeFeedbackReceipt:
    """Complete all-members join for one sealed evaluated portfolio."""

    wave_index: int
    request_sha256: str
    ranked_decision_sha256: str
    scope: ForecastCalibrationScope
    actions: tuple[PortfolioActionOutcomeFeedback, ...]
    receipt_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be a positive exact integer")
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.ranked_decision_sha256, "ranked_decision_sha256")
        if type(self.scope) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        self.scope.revalidate()
        if (
            type(self.actions) is not tuple
            or not self.actions
            or any(
                type(value) is not PortfolioActionOutcomeFeedback
                for value in self.actions
            )
        ):
            raise ValueError("actions must contain exact feedback values")
        for value in self.actions:
            value.__post_init__()
        if len({value.option_id for value in self.actions}) != len(self.actions):
            raise ValueError("feedback receipt cannot repeat an action")
        for value in self.actions:
            if (
                value.wave_index != self.wave_index
                or value.request_sha256 != self.request_sha256
                or value.ranked_decision_sha256 != self.ranked_decision_sha256
                or any(
                    observation.prediction.scope != self.scope
                    for observation in value.observations
                )
            ):
                raise ValueError("feedback action belongs to a foreign wave")
        computed = _hash(_RECEIPT_DOMAIN, self._unsigned_record())
        if self.receipt_sha256 not in ("", computed):
            raise ValueError("receipt_sha256 does not authenticate feedback")
        object.__setattr__(self, "receipt_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": 1,
            "wave_index": self.wave_index,
            "request_sha256": self.request_sha256,
            "ranked_decision_sha256": self.ranked_decision_sha256,
            "scope_sha256": self.scope.scope_sha256,
            "actions": [value.to_record() for value in self.actions],
            "treatment_visibility": "card_blind_action_outcomes_only",
        }
        infeasible_count = sum(
            value.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
            for value in self.actions
        )
        if infeasible_count:
            record.update(
                {
                    "schema_version": 2,
                    "ranked_itt_action_count": len(self.actions),
                    "scored_action_count": len(self.actions) - infeasible_count,
                    "candidate_infeasible_action_count": infeasible_count,
                    "candidate_infeasible_semantics": (
                        "retained_as_no_yield_action_without_metric_calibration_or_"
                        "parent_relation"
                    ),
                }
            )
        return record

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ContextualOutcomeHistoryReceipt:
    """Authenticated, relevance-ranked empirical history for one parent.

    The receipt deliberately labels transfer distance.  Cross-lineage facts are
    analogies, never silently presented as though the same action had already
    been evaluated on the current parent.
    """

    query: ContextualOutcomeQuery
    actions: tuple[PortfolioActionOutcomeFeedback, ...]
    transfer_scopes: tuple[OutcomeTransferScope, ...]
    receipt_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.query) is not ContextualOutcomeQuery:
            raise TypeError("query must be an exact ContextualOutcomeQuery")
        ContextualOutcomeQuery.__post_init__(self.query)
        if type(self.actions) is not tuple or any(
            type(value) is not PortfolioActionOutcomeFeedback for value in self.actions
        ):
            raise TypeError("actions must contain exact feedback values")
        if type(self.transfer_scopes) is not tuple or any(
            type(value) is not OutcomeTransferScope for value in self.transfer_scopes
        ):
            raise TypeError("transfer_scopes must contain exact scope values")
        if len(self.actions) != len(self.transfer_scopes):
            raise ValueError("actions and transfer_scopes must have equal length")
        if len(self.actions) > self.query.max_actions:
            raise ValueError("contextual history exceeds the requested bound")
        if len({value.feedback_sha256 for value in self.actions}) != len(self.actions):
            raise ValueError("contextual history cannot repeat feedback actions")
        if any(
            value.wave_index >= self.query.cutoff_wave_index_exclusive
            for value in self.actions
        ):
            raise ValueError("contextual history contains same/future-wave evidence")
        if (
            not self.query.include_cross_lineage_analogies
            and OutcomeTransferScope.CROSS_LINEAGE_ANALOGY in self.transfer_scopes
        ):
            raise ValueError("contextual history contains forbidden cross-lineage data")
        expected_scopes = tuple(
            _classify_outcome_transfer_scope(action, self.query)
            for action in self.actions
        )
        if self.transfer_scopes != expected_scopes:
            raise ValueError(
                "contextual history transfer scopes differ from query lineage"
            )
        computed = _hash(_CONTEXTUAL_HISTORY_DOMAIN, self._unsigned_record())
        if self.receipt_sha256 not in ("", computed):
            raise ValueError("receipt_sha256 does not authenticate contextual history")
        object.__setattr__(self, "receipt_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "query_sha256": self.query.query_sha256,
            "actions": [
                {
                    "feedback_sha256": action.feedback_sha256,
                    "transfer_scope": scope.value,
                }
                for action, scope in zip(
                    self.actions,
                    self.transfer_scopes,
                    strict=True,
                )
            ],
            "epistemic_status": "observational_predictive_history_not_causal_credit",
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "query": self.query.to_record(),
            "receipt_sha256": self.receipt_sha256,
        }

    def to_prompt_record(self) -> FrozenJsonObject:
        self.__post_init__()
        frozen = freeze_json(
            {
                "schema_version": 1,
                "receipt_sha256": self.receipt_sha256,
                "query_sha256": self.query.query_sha256,
                "cutoff_wave_index_exclusive": (self.query.cutoff_wave_index_exclusive),
                "current_parent_candidate_id": (self.query.current_parent_candidate_id),
                "current_parent_configuration_sha256": (
                    self.query.current_parent_configuration_sha256
                ),
                "actions": [
                    {
                        "transfer_scope": scope.value,
                        "source_parent_candidate_id": action.parent_candidate_id,
                        "source_parent_configuration_sha256": (
                            action.parent_candidate_identity_sha256
                        ),
                        "source_child_candidate_id": action.candidate_id,
                        "source_child_outcome_sha256": (
                            action.candidate_outcome_sha256
                        ),
                        "applicability": (
                            "direct_same_parent_evidence"
                            if scope is OutcomeTransferScope.SAME_PARENT
                            else (
                                "same_lineage_transfer_evidence"
                                if scope is OutcomeTransferScope.SAME_LINEAGE
                                else "cross_lineage_analogy_requires_revalidation"
                            )
                        ),
                        **action.to_prompt_record(),
                    }
                    for action, scope in zip(
                        self.actions,
                        self.transfer_scopes,
                        strict=True,
                    )
                ],
                "epistemic_status": (
                    "observational_predictive_history_not_causal_credit"
                ),
                "card_and_rationale_fields_excluded": True,
            }
        )
        if type(frozen) is not FrozenJsonObject:
            raise AssertionError("contextual outcome history did not freeze")
        return frozen


def observe_selected_portfolio_forecasts(
    *,
    wave_index: int,
    parent: EvolutionCandidate,
    result: PortfolioVariationWaveResult,
    selected_predictions: tuple[ForecastPredictionReceipt, ...],
    adjudicator: MeaningfulMetricDirectionAdjudicator,
    decision_metric_projection: DecisionMetricProjection | None = None,
) -> PortfolioOutcomeFeedbackReceipt:
    """Join only evaluated K4 forecasts to exact outcomes and adjudication.

    The optional benchmark projection exposes semantic objective plus sealed
    violation/constraint values.  ``None`` intentionally retains the legacy
    objective-map path and therefore its byte-identical feedback receipts.
    """

    if type(parent) is not EvolutionCandidate:
        raise TypeError("parent must be exact EvolutionCandidate")
    EvolutionCandidate.__post_init__(parent)
    if type(result) is not PortfolioVariationWaveResult:
        raise TypeError("result must be exact PortfolioVariationWaveResult")
    PortfolioVariationWaveResult.__post_init__(result)
    decision = result.selection_decision
    if decision is None:
        raise ValueError("forecast feedback requires the typed selection decision")
    if type(selected_predictions) is not tuple or any(
        type(value) is not ForecastPredictionReceipt for value in selected_predictions
    ):
        raise TypeError("selected_predictions must be an exact prediction tuple")
    for value in selected_predictions:
        value.revalidate()
    if decision_metric_projection is not None:
        if type(decision_metric_projection) is not DecisionMetricProjection:
            raise TypeError("decision_metric_projection must be exact or None")
        decision_metric_projection.__post_init__()
    if not selected_predictions:
        raise ValueError("forecast feedback requires selected predictions")
    scopes = {value.scope for value in selected_predictions}
    proposals = {value.selector_decision_sha256 for value in selected_predictions}
    if len(scopes) != 1 or len(proposals) != 1:
        raise ValueError("selected predictions mix calibration scopes or proposals")
    scope = next(iter(scopes))
    proposal_sha256 = next(iter(proposals))
    parent_identity = parent.occurrence.configuration_hash
    if any(
        value.wave_index != wave_index
        or value.parent_candidate_identity_sha256 != parent_identity
        for value in selected_predictions
    ):
        raise ValueError("selected predictions belong to a foreign parent or wave")
    by_option_metric = {
        (value.option_id, value.metric_id): value for value in selected_predictions
    }
    if len(by_option_metric) != len(selected_predictions):
        raise ValueError("selected predictions repeat an option/metric cell")
    parent_outcome_sha256 = _candidate_outcome_sha256(parent)
    parent_metrics = (
        parent.objective_map
        if decision_metric_projection is None
        else project_candidate_decision_metrics(
            parent,
            decision_metric_projection,
        ).metric_map
    )
    actions: list[PortfolioActionOutcomeFeedback] = []
    for selected, member, outcome in zip(
        decision.members,
        result.receipt.members,
        result.outcomes,
        strict=True,
    ):
        materialization = member.materialization
        child = outcome.candidate
        if child is None:  # Closed by result validation.
            raise AssertionError("validated portfolio result lost its child")
        if (
            selected.option_id != materialization.option_id
            or selected.option_identity_sha256 != materialization.option_identity_sha256
        ):
            raise ValueError("selection and materialization action identities differ")
        predictions = tuple(
            sorted(
                (
                    value
                    for (option_id, _), value in by_option_metric.items()
                    if option_id == selected.option_id
                ),
                key=lambda value: value.metric_id,
            )
        )
        if tuple(value.metric_id for value in predictions) != tuple(
            sorted(parent_metrics)
        ):
            raise ValueError("selected forecasts differ from parent decision metrics")
        if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
            failure = member.candidate_failure
            if type(failure) is not PortfolioCandidateFailureEvidence:
                raise AssertionError(
                    "validated infeasible member lost candidate failure evidence"
                )
            actions.append(
                PortfolioActionOutcomeFeedback(
                    wave_index=wave_index,
                    request_sha256=decision.request_sha256,
                    ranked_decision_sha256=decision.decision_sha256,
                    proposal_sha256=proposal_sha256,
                    parent_candidate_id=parent.candidate_id.value,
                    parent_candidate_identity_sha256=parent_identity,
                    parent_outcome_sha256=parent_outcome_sha256,
                    candidate_id=child.candidate_id.value,
                    candidate_outcome_sha256=member.outcome_sha256,
                    option_id=selected.option_id,
                    option_identity_sha256=selected.option_identity_sha256,
                    family=selected.family,
                    changed_paths=materialization.changed_paths,
                    observations=(),
                    parent_relation=None,
                    reward=member.reward,
                    dominates_parent=False,
                    better_than_parent=False,
                    disposition=member.disposition,
                    candidate_failure=failure,
                )
            )
            continue
        child_metrics = (
            child.objective_map
            if decision_metric_projection is None
            else project_candidate_decision_metrics(
                child,
                decision_metric_projection,
            ).metric_map
        )
        if set(child_metrics) != set(parent_metrics):
            raise ValueError(
                "selected forecasts differ from evaluated decision metrics"
            )
        observations: list[ForecastCalibrationObservation] = []
        for prediction in predictions:
            metric_id = prediction.metric_id
            request = MeaningfulDirectionRequest(
                benchmark_sha256=scope.benchmark_sha256,
                session_sha256=scope.session_sha256,
                wave_index=wave_index,
                parent_candidate_identity_sha256=parent_identity,
                option_id=selected.option_id,
                option_identity_sha256=selected.option_identity_sha256,
                metric_id=metric_id,
                parent_outcome_sha256=parent_outcome_sha256,
                child_outcome_sha256=member.outcome_sha256,
                parent_metric_value=parent_metrics[metric_id],
                child_metric_value=child_metrics[metric_id],
            )
            observations.append(observe_forecast(prediction, request, adjudicator))
        actions.append(
            PortfolioActionOutcomeFeedback(
                wave_index=wave_index,
                request_sha256=decision.request_sha256,
                ranked_decision_sha256=decision.decision_sha256,
                proposal_sha256=proposal_sha256,
                parent_candidate_id=parent.candidate_id.value,
                parent_candidate_identity_sha256=parent_identity,
                parent_outcome_sha256=parent_outcome_sha256,
                candidate_id=child.candidate_id.value,
                candidate_outcome_sha256=member.outcome_sha256,
                option_id=selected.option_id,
                option_identity_sha256=selected.option_identity_sha256,
                family=selected.family,
                changed_paths=materialization.changed_paths,
                observations=tuple(observations),
                parent_relation=member.parent_relations[0],
                reward=member.reward,
                dominates_parent=member.dominates_any_parent,
                better_than_parent=member.better_than_any_parent,
                metric_transitions=tuple(
                    DecisionMetricTransition(
                        metric_id=observation.prediction.metric_id,
                        parent_value=float(
                            parent_metrics[observation.prediction.metric_id]
                        ),
                        child_value=float(
                            child_metrics[observation.prediction.metric_id]
                        ),
                        actual_direction=observation.adjudication.actual_direction,
                        adjudication_receipt_sha256=(
                            observation.adjudication.receipt_sha256
                        ),
                    )
                    for observation in observations
                ),
                disposition=member.disposition,
            )
        )
    calibrated_cells = {
        (action.option_id, observation.prediction.metric_id)
        for action in actions
        for observation in action.observations
    }
    excluded_cells = {
        (action.option_id, prediction.metric_id)
        for action in actions
        if action.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
        for (option_id, _), prediction in by_option_metric.items()
        if option_id == action.option_id
    }
    if set(by_option_metric) != calibrated_cells.union(excluded_cells):
        raise ValueError("unevaluated or missing K8 forecasts entered K4 feedback")
    return PortfolioOutcomeFeedbackReceipt(
        wave_index=wave_index,
        request_sha256=decision.request_sha256,
        ranked_decision_sha256=decision.decision_sha256,
        scope=scope,
        actions=tuple(actions),
    )


@dataclass(slots=True)
class PortfolioOutcomeFeedbackLedger:
    """Append-only in-memory ledger with prior-wave snapshot projections."""

    receipts: list[PortfolioOutcomeFeedbackReceipt] = field(default_factory=list)

    def append(self, receipt: PortfolioOutcomeFeedbackReceipt) -> None:
        if type(receipt) is not PortfolioOutcomeFeedbackReceipt:
            raise TypeError("receipt must be exact PortfolioOutcomeFeedbackReceipt")
        receipt.__post_init__()
        if any(
            value.receipt_sha256 == receipt.receipt_sha256 for value in self.receipts
        ):
            raise ValueError("feedback ledger already contains this receipt")
        if any(
            value.request_sha256 == receipt.request_sha256 for value in self.receipts
        ):
            raise ValueError("feedback ledger already contains this selector request")
        self.receipts.append(receipt)

    @property
    def observations(self) -> tuple[ForecastCalibrationObservation, ...]:
        return tuple(
            observation
            for receipt in self.receipts
            for action in receipt.actions
            for observation in action.observations
        )

    def calibration_snapshot(
        self,
        *,
        scope: ForecastCalibrationScope,
        cutoff_wave_index_exclusive: int,
        prior: BetaCorrectnessPrior = BetaCorrectnessPrior(),
        family_min_support: int = 4,
    ) -> ForecastCalibrationSnapshot:
        return build_calibration_snapshot(
            self.observations,
            scope=scope,
            cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
            prior=prior,
            family_min_support=family_min_support,
        )

    def prompt_history(
        self,
        *,
        cutoff_wave_index_exclusive: int,
        max_actions: int | None = None,
    ) -> FrozenJsonObject:
        if (
            type(cutoff_wave_index_exclusive) is not int
            or cutoff_wave_index_exclusive <= 0
        ):
            raise ValueError("cutoff_wave_index_exclusive must be positive")
        if max_actions is not None and (
            type(max_actions) is not int or max_actions <= 0
        ):
            raise ValueError("max_actions must be positive or None")
        actions = [
            action
            for receipt in self.receipts
            if receipt.wave_index < cutoff_wave_index_exclusive
            for action in receipt.actions
        ]
        actions.sort(
            key=lambda value: (
                value.wave_index,
                value.request_sha256,
                value.option_id,
            )
        )
        if max_actions is not None:
            actions = actions[-max_actions:]
        frozen = freeze_json(
            {
                "schema_version": 1,
                "cutoff_wave_index_exclusive": cutoff_wave_index_exclusive,
                "actions": [value.to_prompt_record() for value in actions],
                "treatment_visibility": "action_outcomes_only",
            }
        )
        if type(frozen) is not FrozenJsonObject:
            raise AssertionError("feedback history did not freeze as an object")
        return frozen

    def contextual_history(
        self,
        query: ContextualOutcomeQuery,
    ) -> ContextualOutcomeHistoryReceipt:
        """Retrieve prior-only evidence with explicit parent-transfer distance.

        Filtering is intentionally structural and deterministic in v1.  It is
        safer than a learned similarity score while the latter lacks held-out
        calibration, and it makes the exact evidence set replayable.
        """

        if type(query) is not ContextualOutcomeQuery:
            raise TypeError("query must be an exact ContextualOutcomeQuery")
        ContextualOutcomeQuery.__post_init__(query)
        ranked: list[
            tuple[
                tuple[int, int, str, str, str],
                PortfolioActionOutcomeFeedback,
                OutcomeTransferScope,
            ]
        ] = []
        for receipt in self.receipts:
            if receipt.wave_index >= query.cutoff_wave_index_exclusive:
                continue
            for action in receipt.actions:
                if query.families and action.family not in query.families:
                    continue
                if query.changed_paths and not any(
                    _paths_overlap(observed, requested)
                    for observed in action.changed_paths
                    for requested in query.changed_paths
                ):
                    continue
                scope = _classify_outcome_transfer_scope(action, query)
                if scope is OutcomeTransferScope.SAME_PARENT:
                    scope_rank = 0
                elif scope is OutcomeTransferScope.SAME_LINEAGE:
                    scope_rank = 1
                else:
                    scope_rank = 2
                    if not query.include_cross_lineage_analogies:
                        continue
                ranked.append(
                    (
                        (
                            scope_rank,
                            -action.wave_index,
                            action.request_sha256,
                            action.option_id,
                            action.feedback_sha256,
                        ),
                        action,
                        scope,
                    )
                )
        ranked.sort(key=lambda value: value[0])
        chosen = ranked[: query.max_actions]
        return ContextualOutcomeHistoryReceipt(
            query=query,
            actions=tuple(value[1] for value in chosen),
            transfer_scopes=tuple(value[2] for value in chosen),
        )


class SelectedForecastProvider(Protocol):
    """Recover the typed selected K4 forecasts from a sealed selector result."""

    def __call__(
        self,
        wave: PortfolioVariationWaveRequest,
        result: PortfolioVariationWaveResult,
    ) -> tuple[ForecastPredictionReceipt, ...]: ...


class SelectedSearchSourceProvider(Protocol):
    """Recover framework source labels aligned to one evaluated portfolio."""

    def __call__(
        self,
        wave: PortfolioVariationWaveRequest,
        result: PortfolioVariationWaveResult,
    ) -> tuple[str, ...]: ...


class SelectedAllocationRealizationProvider(Protocol):
    """Recover objective-blind requested-to-realized allocation evidence."""

    def __call__(
        self,
        wave: PortfolioVariationWaveRequest,
        result: PortfolioVariationWaveResult,
    ) -> ContextualPortfolioAllocationRealization | None: ...


class ContextualMarginalUtilityProjector(Protocol):
    """Project a pre-generation utility snapshot onto evaluated candidates."""

    def project(
        self,
        *,
        snapshot: ArchiveUtilitySnapshot,
        results: tuple[PortfolioVariationWaveResult, ...],
    ) -> tuple[tuple[float, ...], ...]: ...


class DirectionAdjudicatorProvider(Protocol):
    """Return the benchmark/session-bound adjudicator for one completed wave."""

    def __call__(
        self,
        wave: PortfolioVariationWaveRequest,
        result: PortfolioVariationWaveResult,
    ) -> MeaningfulMetricDirectionAdjudicator: ...


@dataclass(slots=True)
class CalibratedCampaignOutcomeUpdater:
    """Zero-call campaign bridge from completed waves to prior-only feedback."""

    ledger: PortfolioOutcomeFeedbackLedger
    selected_forecasts: SelectedForecastProvider
    adjudicator_for: DirectionAdjudicatorProvider
    decision_metric_projection: DecisionMetricProjection | None = None
    contextual_ledger: ContextualSearchLedger | None = None
    selected_search_sources: SelectedSearchSourceProvider | None = None
    contextual_marginal_utility: ContextualMarginalUtilityProjector | None = None
    contextual_campaign_scope_sha256: str | None = None
    selected_allocation_realization: (
        SelectedAllocationRealizationProvider | None
    ) = None
    _prepared: dict[
        str,
        tuple[
            tuple[str, ...],
            tuple[PortfolioOutcomeFeedbackReceipt, ...],
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None,
            ContextualPortfolioOutcomeBatch | None,
            tuple[ContextualPortfolioAllocationRealization, ...],
        ],
    ] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.ledger) is not PortfolioOutcomeFeedbackLedger:
            raise TypeError("ledger must be exact PortfolioOutcomeFeedbackLedger")
        if not callable(self.selected_forecasts):
            raise TypeError("selected_forecasts must be callable")
        if not callable(self.adjudicator_for):
            raise TypeError("adjudicator_for must be callable")
        if self.decision_metric_projection is not None:
            if type(self.decision_metric_projection) is not DecisionMetricProjection:
                raise TypeError("decision_metric_projection must be exact or None")
            self.decision_metric_projection.__post_init__()
        contextual = (
            self.contextual_ledger,
            self.selected_search_sources,
            self.contextual_marginal_utility,
            self.contextual_campaign_scope_sha256,
        )
        if any(value is not None for value in contextual) and not all(
            value is not None for value in contextual
        ):
            raise ValueError("contextual outcome dependencies are all-or-none")
        if self.contextual_ledger is not None:
            if type(self.contextual_ledger) is not ContextualSearchLedger:
                raise TypeError("contextual_ledger must be exact or None")
            if not callable(self.selected_search_sources):
                raise TypeError("selected_search_sources must be callable")
            if not callable(
                getattr(self.contextual_marginal_utility, "project", None)
            ):
                raise TypeError("contextual marginal utility must implement its port")
            assert self.contextual_campaign_scope_sha256 is not None
            require_sha256(
                self.contextual_campaign_scope_sha256,
                "contextual_campaign_scope_sha256",
            )
        if self.selected_allocation_realization is not None:
            if self.contextual_ledger is None:
                raise ValueError(
                    "allocation realization feedback requires contextual outcomes"
                )
            if not callable(self.selected_allocation_realization):
                raise TypeError("selected_allocation_realization must be callable")

    async def prepare_update(
        self,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        prior_memory: FrozenJsonObject,
    ):
        from agent_evolve.application.portfolio_campaign_runtime import (
            CampaignPortfolioOutcomePreparation,
        )

        self.__post_init__()
        if type(request) is not CampaignStageRequest:
            raise TypeError("request must be exact CampaignStageRequest")
        CampaignStageRequest.__post_init__(request)
        if request.step.kind is not CampaignGenerationKind.PORTFOLIO:
            raise ValueError("outcome feedback requires a portfolio stage")
        if (
            type(waves) is not tuple
            or type(results) is not tuple
            or len(waves) != len(results)
            or not waves
        ):
            raise ValueError("waves and results must be equal non-empty tuples")
        if type(prior_memory) is not FrozenJsonObject:
            raise TypeError("prior_memory must be an exact frozen object")
        pending: list[PortfolioOutcomeFeedbackReceipt] = []
        for wave, result in zip(waves, results, strict=True):
            if type(wave) is not PortfolioVariationWaveRequest:
                raise TypeError("waves must contain exact portfolio requests")
            if type(result) is not PortfolioVariationWaveResult:
                raise TypeError("results must contain exact portfolio results")
            predictions = self.selected_forecasts(wave, result)
            adjudicator = self.adjudicator_for(wave, result)
            pending.append(
                observe_selected_portfolio_forecasts(
                    wave_index=request.step.generation,
                    parent=wave.parent,
                    result=result,
                    selected_predictions=predictions,
                    adjudicator=adjudicator,
                    decision_metric_projection=self.decision_metric_projection,
                )
            )
        # Validate the whole stage without publishing any new ledger entry.
        combined = (*self.ledger.receipts, *pending)
        validate_feedback_ledger(combined)
        preview_ledger = PortfolioOutcomeFeedbackLedger()
        for receipt in combined:
            preview_ledger.append(receipt)
        memory = thaw_json(prior_memory)
        feedback = {
            "schema_version": 1,
            "latest_wave_index": request.step.generation,
            "stage_receipt_sha256s": [value.receipt_sha256 for value in pending],
            "ledger_receipt_sha256s": [
                value.receipt_sha256 for value in preview_ledger.receipts
            ],
            "observation_count": len(preview_ledger.observations),
            "prompt_history": thaw_json(
                preview_ledger.prompt_history(
                    cutoff_wave_index_exclusive=request.step.generation + 1,
                )
            ),
            "provider_calls": 0,
        }
        memory["portfolio_outcome_feedback"] = feedback
        contextual_prior_state: (
            tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None
        ) = None
        contextual_batch: ContextualPortfolioOutcomeBatch | None = None
        contextual_realizations: tuple[
            ContextualPortfolioAllocationRealization, ...
        ] = ()
        contextual_evidence: dict[str, object] | None = None
        if self.contextual_ledger is not None:
            assert self.selected_search_sources is not None
            assert self.contextual_marginal_utility is not None
            assert self.contextual_campaign_scope_sha256 is not None
            selected_sources = tuple(
                self.selected_search_sources(wave, result)
                for wave, result in zip(waves, results, strict=True)
            )
            marginal_utilities = self.contextual_marginal_utility.project(
                snapshot=request.archive_utility,
                results=results,
            )
            # The authenticated campaign cadence alternates portfolio and
            # recombination generations, so controller waves are 1, 2, ...
            # for campaign generations 1, 3, ... .
            controller_wave_index = (request.step.generation + 1) // 2
            contextual_batch = observe_contextual_portfolio_outcomes(
                campaign_scope_sha256=self.contextual_campaign_scope_sha256,
                wave_index=controller_wave_index,
                waves=waves,
                results=results,
                selected_source_ids=selected_sources,
                marginal_utilities=marginal_utilities,
            )
            contextual_prior_state = (
                tuple(
                    value.observation_sha256
                    for value in self.contextual_ledger.observations
                ),
                tuple(
                    value.credit_sha256
                    for value in self.contextual_ledger.delayed_credits
                ),
                tuple(
                    value.realization_sha256
                    for value in self.contextual_ledger.allocation_realizations
                ),
            )
            preview_contextual = ContextualSearchLedger(
                observations=list(self.contextual_ledger.observations),
                delayed_credits=list(self.contextual_ledger.delayed_credits),
                allocation_realizations=list(
                    self.contextual_ledger.allocation_realizations
                ),
            )
            preview_contextual.append_batch(contextual_batch.observations)
            if self.selected_allocation_realization is not None:
                recovered = tuple(
                    self.selected_allocation_realization(wave, result)
                    for wave, result in zip(waves, results, strict=True)
                )
                if any(value is None for value in recovered):
                    raise ValueError(
                        "contextual run omitted an allocation realization"
                    )
                contextual_realizations = tuple(
                    sorted(
                        (
                            value
                            for value in recovered
                            if type(value)
                            is ContextualPortfolioAllocationRealization
                        ),
                        key=lambda value: (
                            value.controller_wave_index,
                            value.slice_id,
                            value.realization_sha256,
                        ),
                    )
                )
                if len(contextual_realizations) != len(recovered):
                    raise TypeError(
                        "allocation realization provider returned a foreign value"
                    )
                preview_contextual.append_allocation_realization_batch(
                    contextual_realizations
                )
            contextual_evidence = {
                "batch": contextual_batch.to_record(),
                "ledger_observation_count_after": len(
                    preview_contextual.observations
                ),
                "ledger_delayed_credit_count_after": len(
                    preview_contextual.delayed_credits
                ),
                "allocation_realization_sha256s": [
                    value.realization_sha256
                    for value in contextual_realizations
                ],
                "ledger_allocation_realization_count_after": len(
                    preview_contextual.allocation_realizations
                ),
                "provider_calls": 0,
            }
            memory["contextual_search_controller"] = {
                "schema_version": 1,
                "latest_wave_index": controller_wave_index,
                "latest_batch_sha256": contextual_batch.batch_sha256,
                "latest_observation_sha256s": [
                    value.observation_sha256
                    for value in contextual_batch.observations
                ],
                "ledger_observation_count": len(
                    preview_contextual.observations
                ),
                "ledger_delayed_credit_count": len(
                    preview_contextual.delayed_credits
                ),
                "ledger_allocation_realization_count": len(
                    preview_contextual.allocation_realizations
                ),
                "provider_calls": 0,
            }
        frozen = freeze_json(memory)
        if type(frozen) is not FrozenJsonObject:
            raise AssertionError("updated campaign memory did not freeze as an object")
        preparation = CampaignPortfolioOutcomePreparation(
            request_sha256=request.request_sha256,
            generation=request.step.generation,
            wave_request_sha256s=tuple(
                wave.selection_request.request_sha256 for wave in waves
            ),
            result_receipt_sha256s=tuple(
                result.receipt.receipt_sha256 for result in results
            ),
            prior_memory_sha256=typed_json_sha256(prior_memory),
            updated_memory=frozen,
            evidence=freeze_json(
                {
                    "pending_feedback_receipt_sha256s": [
                        value.receipt_sha256 for value in pending
                    ],
                    "ledger_receipt_count_after": len(preview_ledger.receipts),
                    "contextual_outcomes": contextual_evidence,
                    "provider_calls": 0,
                }
            ),
        )
        if type(preparation.evidence) is not FrozenJsonObject:
            raise AssertionError("outcome preparation evidence was not an object")
        if preparation.preparation_sha256 in self._prepared:
            raise ValueError("outcome update preparation already exists")
        self._prepared[preparation.preparation_sha256] = (
            tuple(value.receipt_sha256 for value in self.ledger.receipts),
            tuple(pending),
            contextual_prior_state,
            contextual_batch,
            contextual_realizations,
        )
        return preparation

    def commit_update(self, preparation) -> None:
        from agent_evolve.application.portfolio_campaign_runtime import (
            CampaignPortfolioOutcomePreparation,
        )

        if type(preparation) is not CampaignPortfolioOutcomePreparation:
            raise TypeError("preparation must be exact")
        prepared = self._prepared.pop(preparation.preparation_sha256, None)
        if prepared is None:
            raise ValueError("outcome update preparation is unavailable")
        (
            prior_receipts,
            pending,
            contextual_prior_state,
            contextual_batch,
            contextual_realizations,
        ) = prepared
        if tuple(value.receipt_sha256 for value in self.ledger.receipts) != (
            prior_receipts
        ):
            raise RuntimeError("outcome ledger changed after preparation")
        if contextual_prior_state is not None:
            if self.contextual_ledger is None or contextual_batch is None:
                raise RuntimeError("contextual outcome dependencies disappeared")
            current_contextual_state = (
                tuple(
                    value.observation_sha256
                    for value in self.contextual_ledger.observations
                ),
                tuple(
                    value.credit_sha256
                    for value in self.contextual_ledger.delayed_credits
                ),
                tuple(
                    value.realization_sha256
                    for value in self.contextual_ledger.allocation_realizations
                ),
            )
            if current_contextual_state != contextual_prior_state:
                raise RuntimeError("contextual search ledger changed after preparation")
        self.ledger.receipts.extend(pending)
        if contextual_batch is not None:
            assert self.contextual_ledger is not None
            self.contextual_ledger.observations.extend(
                contextual_batch.observations
            )
            self.contextual_ledger.allocation_realizations.extend(
                contextual_realizations
            )

    def abort_update(self, preparation) -> None:
        from agent_evolve.application.portfolio_campaign_runtime import (
            CampaignPortfolioOutcomePreparation,
        )

        if type(preparation) is not CampaignPortfolioOutcomePreparation:
            raise TypeError("preparation must be exact")
        self._prepared.pop(preparation.preparation_sha256, None)

    async def update(
        self,
        request: CampaignStageRequest,
        waves: tuple[PortfolioVariationWaveRequest, ...],
        results: tuple[PortfolioVariationWaveResult, ...],
        prior_memory: FrozenJsonObject,
    ) -> FrozenJsonObject:
        """Backward-compatible immediate publication wrapper."""

        preparation = await self.prepare_update(
            request,
            waves,
            results,
            prior_memory,
        )
        self.commit_update(preparation)
        return preparation.updated_memory


def validate_feedback_ledger(
    values: Sequence[PortfolioOutcomeFeedbackReceipt],
) -> None:
    ledger = PortfolioOutcomeFeedbackLedger()
    for value in values:
        ledger.append(value)


__all__ = [
    "CalibratedCampaignOutcomeUpdater",
    "ContextualOutcomeHistoryReceipt",
    "ContextualOutcomeQuery",
    "DecisionMetricTransition",
    "DirectionAdjudicatorProvider",
    "ContextualMarginalUtilityProjector",
    "OutcomeTransferScope",
    "PortfolioActionOutcomeFeedback",
    "PortfolioOutcomeFeedbackLedger",
    "PortfolioOutcomeFeedbackReceipt",
    "SelectedForecastProvider",
    "SelectedAllocationRealizationProvider",
    "SelectedSearchSourceProvider",
    "observe_selected_portfolio_forecasts",
    "validate_feedback_ledger",
]

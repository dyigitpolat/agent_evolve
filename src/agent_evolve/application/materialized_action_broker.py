"""Workload-blind brokerage over already-materialized evolutionary actions.

The broker is deliberately downstream of proposal generation.  Workload
adapters and proposal experts own legality and materialization; this module
sees only authenticated configurations, generic state cells, lineage, and
strictly prior outcomes.  Mutation, restart, acquisition, and recombination
therefore compete for the same expensive evaluation slots without exposing a
workload or model identifier to the policy.

Outcome channels remain orthogonal diagnostics.  Selection itself uses one
currency: normalized archive return resolved over an authenticated lineage
horizon.  Feasibility and realization gate that return, forecast error governs
the authority of the consequence model, and uncertainty is used only for
nonterminal exploration.  Sparse state cells shrink toward an arm-level
posterior before a joint-slate optimizer adds only residual complementarity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Protocol, runtime_checkable

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionAllocationDirective,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


MATERIALIZED_ACTION_BROKER_ID = "regret_brokered_expert_evolution"
MATERIALIZED_ACTION_BROKER_VERSION = 8
EMPIRICAL_RETURN_ESTIMATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:materialized-action-empirical-return:v1"
).hexdigest()
MATERIALIZED_ACTION_BROKER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:regret-brokered-expert-evolution:v8;"
    b"decision-unit=authenticated-materialized-action;"
    b"axes=expert,native-rank,parent-arity,operator,target,role;"
    b"state=residual-cell,parent-cell,archive-cell,structural-signature,"
    b"patch-compatibility,phase,horizon,calibration,source-distance,memory-dose;"
    b"forbidden-inputs=workload-id,model-id,provider-id,objective-name;"
    b"selection-currency=normalized-resolved-lineage-return;"
    b"diagnostic-channels=gain,positive,stage-survival,terminal-persistence,"
    b"descendant,feasibility,forecast-error,realization;"
    b"return-estimator=injected-port-or-zero-baseline-bounded-empirical-mean-with-arm-hierarchical-shrinkage;"
    b"cold-start-tie=within-expert-native-rank-then-expert-diversity;"
    b"uncertainty=distribution-free-maximum-standard-error;"
    b"selection=additive-return-plus-residual-complementarity;"
    b"allocation=optional-authenticated-static-requirement-or-outcome-"
    b"adaptive-directive;"
    b"adaptive-allocation=prior-decision-and-observed-outcome-bound;"
    b"exploration=authenticated-prequential-required-set-plus-at-most-one-"
    b"posterior-nonterminal-arm;"
    b"terminal-information-bonus=zero;"
    b"reference=bounded-multislot-conservative-escrow;"
    b"duplicate-constraint=one-evaluation-per-phenotype;"
    b"bounded-beam-prefix-gate=canonical-suffix-unique-phenotype-"
    b"completion-witness;deterministic=true"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_CONTEXT_DOMAIN = b"agent-evolve:materialized-action-context:v1\x00"
_ACTION_DOMAIN = b"agent-evolve:materialized-action-descriptor:v1\x00"
_OUTCOME_DOMAIN = b"agent-evolve:materialized-action-outcome:v1\x00"
_CREDIT_DOMAIN = b"agent-evolve:materialized-action-delayed-credit:v1\x00"
_RESOLVED_RETURN_DOMAIN = b"agent-evolve:materialized-action-resolved-return:v1\x00"
_RETURN_PRIOR_PREDICTION_DOMAIN = (
    b"agent-evolve:materialized-action-return-prior-prediction:v1\x00"
)
_EMPIRICAL_BAYES_RETURN_VALUE_DOMAIN = (
    b"agent-evolve:empirical-bayes-materialized-action-return-value:v1\x00"
)
_ACTION_OPPORTUNITY_EVIDENCE_DOMAIN = (
    b"agent-evolve:materialized-action-opportunity-evidence:v1\x00"
)
_OPPORTUNITY_CONDITIONED_RETURN_VALUE_DOMAIN = (
    b"agent-evolve:opportunity-conditioned-materialized-action-return-value:v1\x00"
)
_EXPLORATION_REQUIREMENT_DOMAIN = (
    b"agent-evolve:materialized-action-exploration-requirement:v1\x00"
)
_ALLOCATION_REQUIREMENT_DOMAIN = (
    b"agent-evolve:materialized-action-allocation-requirement:v1\x00"
)
_DECISION_DOMAIN = b"agent-evolve:materialized-action-broker-decision:v1\x00"


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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _require_probability(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _require_nonnegative_finite(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _candidate_record(value: CandidateId) -> str:
    if type(value) is not CandidateId:
        raise TypeError("parent and target IDs must be exact CandidateId values")
    CandidateId.__post_init__(value)
    return value.value


@dataclass(frozen=True, slots=True)
class MaterializedActionContext:
    """Bounded generic state visible to the action broker."""

    campaign_scope_sha256: str
    decision_index: int
    phase: SearchPhase
    remaining_decisions: int
    remaining_evaluations: int
    residual_frontier_cell: str
    parent_position_cell: str
    archive_relation_cell: str
    structural_signature_sha256: str
    patch_compatibility_cell: str
    forecast_calibration_cell: str
    source_distance_bin: int
    memory_dose_bin: int
    state_signature_sha256: str = field(init=False)
    context_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(
            self.structural_signature_sha256,
            "structural_signature_sha256",
        )
        if type(self.decision_index) is not int or self.decision_index <= 0:
            raise ValueError("decision_index must be a positive exact integer")
        if type(self.phase) is not SearchPhase:
            raise TypeError("phase must be an exact SearchPhase")
        for name in ("remaining_decisions", "remaining_evaluations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in (
            "residual_frontier_cell",
            "parent_position_cell",
            "archive_relation_cell",
            "patch_compatibility_cell",
            "forecast_calibration_cell",
        ):
            _require_token(getattr(self, name), name=name)
        for name in ("source_distance_bin", "memory_dose_bin"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 15:
                raise ValueError(f"{name} must lie in [0, 15]")
        state = self._state_record()
        object.__setattr__(
            self,
            "state_signature_sha256",
            _hash(_CONTEXT_DOMAIN, state),
        )
        object.__setattr__(
            self,
            "context_sha256",
            _hash(
                _CONTEXT_DOMAIN,
                {
                    **state,
                    "campaign_scope_sha256": self.campaign_scope_sha256,
                    "decision_index": self.decision_index,
                },
            ),
        )

    def _state_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": self.phase.value,
            "remaining_decisions": self.remaining_decisions,
            "remaining_evaluations": self.remaining_evaluations,
            "residual_frontier_cell": self.residual_frontier_cell,
            "parent_position_cell": self.parent_position_cell,
            "archive_relation_cell": self.archive_relation_cell,
            "structural_signature_sha256": self.structural_signature_sha256,
            "patch_compatibility_cell": self.patch_compatibility_cell,
            "forecast_calibration_cell": self.forecast_calibration_cell,
            "source_distance_bin": self.source_distance_bin,
            "memory_dose_bin": self.memory_dose_bin,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._state_record(),
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "decision_index": self.decision_index,
            "state_signature_sha256": self.state_signature_sha256,
            "context_sha256": self.context_sha256,
        }


@dataclass(frozen=True, slots=True)
class MaterializedActionDescriptor:
    """One legal route to one fully materialized candidate configuration."""

    context: MaterializedActionContext
    configuration: FrozenJsonObject
    phenotype_identity_sha256: str
    expert_id: str
    native_rank: int
    parent_ids: tuple[CandidateId, ...]
    operator_id: str
    target_candidate_id: CandidateId
    role_id: str
    normalized_evaluation_cost: float
    reference_action: bool = False
    action_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.context) is not MaterializedActionContext:
            raise TypeError("context must be an exact MaterializedActionContext")
        MaterializedActionContext.__post_init__(self.context)
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be an exact FrozenJsonObject")
        require_sha256(self.phenotype_identity_sha256, "phenotype_identity_sha256")
        _require_token(self.expert_id, name="expert_id")
        _require_token(self.operator_id, name="operator_id")
        _require_token(self.role_id, name="role_id")
        if type(self.native_rank) is not int or self.native_rank <= 0:
            raise ValueError("native_rank must be a positive exact integer")
        if type(self.parent_ids) is not tuple or len(self.parent_ids) > 8:
            raise ValueError(
                "parent_ids must be an exact tuple with arity at most eight"
            )
        parent_values = tuple(_candidate_record(value) for value in self.parent_ids)
        if len(parent_values) != len(set(parent_values)):
            raise ValueError("parent_ids must be unique")
        _candidate_record(self.target_candidate_id)
        if self.target_candidate_id in self.parent_ids:
            raise ValueError("target_candidate_id cannot be one of its parents")
        _require_probability(
            self.normalized_evaluation_cost,
            name="normalized_evaluation_cost",
        )
        if type(self.reference_action) is not bool:
            raise TypeError("reference_action must be an exact bool")
        object.__setattr__(
            self,
            "action_sha256",
            _hash(_ACTION_DOMAIN, self._unsigned_record()),
        )

    @property
    def parent_arity(self) -> int:
        return len(self.parent_ids)

    @property
    def configuration_sha256(self) -> str:
        return typed_json_sha256(self.configuration)

    @property
    def arm_key(self) -> tuple[str, str, int]:
        return (self.expert_id, self.operator_id, self.parent_arity)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "context_sha256": self.context.context_sha256,
            "state_signature_sha256": self.context.state_signature_sha256,
            "configuration_sha256": self.configuration_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "expert_id": self.expert_id,
            "native_rank": self.native_rank,
            "parent_ids": [_candidate_record(value) for value in self.parent_ids],
            "parent_arity": self.parent_arity,
            "operator_id": self.operator_id,
            "target_candidate_id": _candidate_record(self.target_candidate_id),
            "role_id": self.role_id,
            "normalized_evaluation_cost_hex": (self.normalized_evaluation_cost.hex()),
            "reference_action": self.reference_action,
        }

    def to_record(self, *, include_configuration: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "action_sha256": self.action_sha256}
        if include_configuration:
            record["configuration"] = self.configuration
        return record


@dataclass(frozen=True, slots=True)
class MaterializedActionOutcome:
    """Immediate, append-only evidence for one requested broker action."""

    action: MaterializedActionDescriptor
    realized: bool
    feasible: bool | None
    normalized_archive_gain: float | None
    positive_marginal_utility: bool | None
    normalized_forecast_error: float | None = None
    outcome_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.action) is not MaterializedActionDescriptor:
            raise TypeError("action must be an exact MaterializedActionDescriptor")
        MaterializedActionDescriptor.__post_init__(self.action)
        if type(self.realized) is not bool:
            raise TypeError("realized must be an exact bool")
        if not self.realized:
            if (
                self.feasible is not None
                or self.normalized_archive_gain is not None
                or self.positive_marginal_utility is not None
                or self.normalized_forecast_error is not None
            ):
                raise ValueError("unrealized actions cannot carry evaluator evidence")
        else:
            if type(self.feasible) is not bool:
                raise TypeError("realized actions require an exact feasibility verdict")
            if self.normalized_archive_gain is None:
                raise ValueError("realized actions require normalized archive gain")
            _require_probability(
                self.normalized_archive_gain,
                name="normalized_archive_gain",
            )
            if type(self.positive_marginal_utility) is not bool:
                raise TypeError("realized actions require an exact positive verdict")
            if self.positive_marginal_utility != (self.normalized_archive_gain > 0.0):
                raise ValueError("positive verdict differs from normalized gain")
            if not self.feasible and (
                self.normalized_archive_gain != 0.0 or self.positive_marginal_utility
            ):
                raise ValueError("infeasible actions cannot carry positive gain")
            if self.normalized_forecast_error is not None:
                _require_probability(
                    self.normalized_forecast_error,
                    name="normalized_forecast_error",
                )
        object.__setattr__(
            self,
            "outcome_sha256",
            _hash(_OUTCOME_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action.action_sha256,
            "realized": self.realized,
            "feasible": self.feasible,
            "normalized_archive_gain_hex": (
                None
                if self.normalized_archive_gain is None
                else self.normalized_archive_gain.hex()
            ),
            "positive_marginal_utility": self.positive_marginal_utility,
            "normalized_forecast_error_hex": (
                None
                if self.normalized_forecast_error is None
                else self.normalized_forecast_error.hex()
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "outcome_sha256": self.outcome_sha256}


@dataclass(frozen=True, slots=True)
class MaterializedActionDelayedCredit:
    """Later survival or descendant evidence joined to an immediate outcome."""

    outcome: MaterializedActionOutcome
    available_at_decision_index: int
    stage_front_survived: bool | None = None
    terminal_front_persisted: bool | None = None
    useful_descendant_observed: bool | None = None
    credit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.outcome) is not MaterializedActionOutcome:
            raise TypeError("outcome must be an exact MaterializedActionOutcome")
        MaterializedActionOutcome.__post_init__(self.outcome)
        if not self.outcome.realized or self.outcome.feasible is not True:
            raise ValueError("delayed credit requires a realized feasible action")
        if (
            type(self.available_at_decision_index) is not int
            or self.available_at_decision_index
            < self.outcome.action.context.decision_index
        ):
            raise ValueError("delayed credit cannot precede its source action")
        values = (
            self.stage_front_survived,
            self.terminal_front_persisted,
            self.useful_descendant_observed,
        )
        if all(value is None for value in values):
            raise ValueError("delayed credit must adjudicate at least one channel")
        if any(value is not None and type(value) is not bool for value in values):
            raise TypeError("delayed credit channels must be exact bools or None")
        object.__setattr__(
            self,
            "credit_sha256",
            _hash(_CREDIT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "outcome_sha256": self.outcome.outcome_sha256,
            "available_at_decision_index": self.available_at_decision_index,
            "stage_front_survived": self.stage_front_survived,
            "terminal_front_persisted": self.terminal_front_persisted,
            "useful_descendant_observed": self.useful_descendant_observed,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "credit_sha256": self.credit_sha256}


@dataclass(frozen=True, slots=True)
class MaterializedActionResolvedReturn:
    """One append-only resolution of an action's finite-horizon return.

    A resolution may be refined as descendants become observable.  The ledger
    uses only the newest resolution available strictly before the decision it
    is scoring.  Components are expressed in the same normalized archive-
    utility currency and must close exactly to ``normalized_horizon_return``.
    The resolver, not the broker, owns lineage attribution and discounting.
    """

    outcome: MaterializedActionOutcome
    available_at_decision_index: int
    horizon_end_decision_index: int
    normalized_immediate_return: float
    normalized_descendant_return: float
    normalized_horizon_return: float
    fully_resolved: bool
    attribution_definition_sha256: str
    return_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.outcome) is not MaterializedActionOutcome:
            raise TypeError("outcome must be an exact MaterializedActionOutcome")
        MaterializedActionOutcome.__post_init__(self.outcome)
        source_index = self.outcome.action.context.decision_index
        if (
            type(self.available_at_decision_index) is not int
            or self.available_at_decision_index <= source_index
        ):
            raise ValueError("resolved return must become available after its action")
        if (
            type(self.horizon_end_decision_index) is not int
            or self.horizon_end_decision_index < source_index
            or self.horizon_end_decision_index >= self.available_at_decision_index
        ):
            raise ValueError(
                "resolved return horizon is inconsistent with availability"
            )
        for name in (
            "normalized_immediate_return",
            "normalized_descendant_return",
            "normalized_horizon_return",
        ):
            _require_probability(getattr(self, name), name=name)
        if not math.isclose(
            self.normalized_immediate_return + self.normalized_descendant_return,
            self.normalized_horizon_return,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("resolved return components do not close")
        expected_immediate = (
            0.0 if not self.outcome.realized else self.outcome.normalized_archive_gain
        )
        assert expected_immediate is not None
        if not math.isclose(
            self.normalized_immediate_return,
            expected_immediate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("resolved immediate return differs from its outcome")
        if type(self.fully_resolved) is not bool:
            raise TypeError("fully_resolved must be an exact bool")
        require_sha256(
            self.attribution_definition_sha256,
            "attribution_definition_sha256",
        )
        object.__setattr__(
            self,
            "return_sha256",
            _hash(_RESOLVED_RETURN_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "outcome_sha256": self.outcome.outcome_sha256,
            "available_at_decision_index": self.available_at_decision_index,
            "horizon_end_decision_index": self.horizon_end_decision_index,
            "normalized_immediate_return_hex": (self.normalized_immediate_return.hex()),
            "normalized_descendant_return_hex": (
                self.normalized_descendant_return.hex()
            ),
            "normalized_horizon_return_hex": self.normalized_horizon_return.hex(),
            "fully_resolved": self.fully_resolved,
            "attribution_definition_sha256": self.attribution_definition_sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "return_sha256": self.return_sha256}


class BrokerEvidenceChannel(str, Enum):
    GAIN = "gain"
    POSITIVE = "positive"
    STAGE_SURVIVAL = "stage_survival"
    TERMINAL_PERSISTENCE = "terminal_persistence"
    DESCENDANT = "descendant"
    FEASIBILITY = "feasibility"
    FORECAST_ERROR = "forecast_error"
    REALIZATION = "realization"


@dataclass(frozen=True, slots=True)
class BrokerChannelEstimate:
    channel: BrokerEvidenceChannel
    mean: float
    standard_deviation: float
    local_count: int
    global_count: int
    local_mean: float
    global_mean: float
    shrinkage_weight: float

    def __post_init__(self) -> None:
        if type(self.channel) is not BrokerEvidenceChannel:
            raise TypeError("channel must be an exact BrokerEvidenceChannel")
        for name in (
            "mean",
            "standard_deviation",
            "local_mean",
            "global_mean",
            "shrinkage_weight",
        ):
            _require_probability(getattr(self, name), name=name)
        for name in ("local_count", "global_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.local_count > self.global_count:
            raise ValueError("local evidence cannot exceed its arm-global evidence")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "channel": self.channel.value,
            "mean_hex": self.mean.hex(),
            "standard_deviation_hex": self.standard_deviation.hex(),
            "local_count": self.local_count,
            "global_count": self.global_count,
            "local_mean_hex": self.local_mean.hex(),
            "global_mean_hex": self.global_mean.hex(),
            "shrinkage_weight_hex": self.shrinkage_weight.hex(),
        }


def _bounded_empirical(values: tuple[float, ...]) -> tuple[float, float, float]:
    """Preserve return scale while retaining a distribution-free error bound."""

    if not values:
        # Archive return is a non-negative *gain*, not a Bernoulli success
        # probability.  A Beta(1, 1)-style mean of 1/2 overwhelms the small
        # gains seen in real campaigns and causes every multi-action slate to
        # saturate.  Zero is the only scale-free lower-bound mean; uncertainty
        # remains maximal and may buy one explicit nonterminal probe below.
        return 0.0, 0.5, 0.0
    mean = math.fsum(values) / len(values)
    # Popoviciu's bound gives sigma <= 1/2 for observations in [0, 1].
    standard_error = 0.5 / math.sqrt(len(values))
    return float(mean), float(standard_error), float(len(values))


@dataclass(frozen=True, slots=True)
class BrokerReturnEstimate:
    """Hierarchically shrunk posterior in the sole selection currency."""

    mean: float
    standard_deviation: float
    local_count: int
    global_count: int
    resolved_count: int
    provisional_count: int
    local_mean: float
    global_mean: float
    shrinkage_weight: float

    def __post_init__(self) -> None:
        for name in (
            "mean",
            "standard_deviation",
            "local_mean",
            "global_mean",
            "shrinkage_weight",
        ):
            _require_probability(getattr(self, name), name=name)
        for name in (
            "local_count",
            "global_count",
            "resolved_count",
            "provisional_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.local_count > self.global_count:
            raise ValueError("local return evidence exceeds arm-global evidence")
        if self.resolved_count + self.provisional_count != self.global_count:
            raise ValueError("return resolution counts do not close")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "mean_hex": self.mean.hex(),
            "standard_deviation_hex": self.standard_deviation.hex(),
            "local_count": self.local_count,
            "global_count": self.global_count,
            "resolved_count": self.resolved_count,
            "provisional_count": self.provisional_count,
            "local_mean_hex": self.local_mean.hex(),
            "global_mean_hex": self.global_mean.hex(),
            "shrinkage_weight_hex": self.shrinkage_weight.hex(),
        }


@runtime_checkable
class MaterializedActionReturnValuePort(Protocol):
    """Predict the common evaluator-grounded return of one unseen action.

    Implementations may wrap a frozen cross-run meta-prior plus branch-local
    prequential updates.  The descriptor deliberately contains no workload,
    model, provider, prompt, or objective-name feature, so the orchestration
    core cannot branch on those identities.  Returning the same typed estimate
    as the empirical fallback keeps one selection currency and one uncertainty
    contract.
    """

    definition_sha256: str

    def estimate(
        self, action: MaterializedActionDescriptor
    ) -> BrokerReturnEstimate: ...


@dataclass(frozen=True, slots=True)
class MaterializedActionReturnPriorPrediction:
    """Authenticated portable prior for one not-yet-evaluated action."""

    action_sha256: str
    mean: float
    standard_deviation: float
    effective_sample_size: float
    evidence_sha256: str
    prediction_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _require_probability(self.mean, name="mean")
        _require_probability(
            self.standard_deviation,
            name="standard_deviation",
        )
        if (
            type(self.effective_sample_size) is not float
            or not math.isfinite(self.effective_sample_size)
            or self.effective_sample_size <= 0.0
        ):
            raise ValueError(
                "effective_sample_size must be a positive finite exact float"
            )
        require_sha256(self.evidence_sha256, "evidence_sha256")
        object.__setattr__(
            self,
            "prediction_sha256",
            _hash(_RETURN_PRIOR_PREDICTION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "mean_hex": self.mean.hex(),
            "standard_deviation_hex": self.standard_deviation.hex(),
            "effective_sample_size_hex": self.effective_sample_size.hex(),
            "evidence_sha256": self.evidence_sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "prediction_sha256": self.prediction_sha256,
        }


@runtime_checkable
class MaterializedActionReturnPriorPort(Protocol):
    """Project an action into a frozen cross-run return prior.

    Workload adapters may compute generic numerical features or consult a
    content-addressed feature panel, but the application core receives only a
    common normalized-return distribution.  The predictor definition must
    authenticate its feature schema, fit, training cutoff, and evidence panel.
    """

    definition_sha256: str

    def predict(
        self,
        action: MaterializedActionDescriptor,
    ) -> MaterializedActionReturnPriorPrediction | None: ...


@dataclass(frozen=True, slots=True)
class MaterializedActionOpportunityEvidence:
    """Authenticated current-archive opportunity for an action's lineage.

    ``source_opportunity`` and ``archive_opportunity_scale`` use the same
    normalized archive-utility currency as realized return.  The former is
    normally the source parent's leave-one-out contribution; the latter is a
    strictly-prior, action-independent scale such as the maximum contribution
    on the current front.  Workload adapters own the projection, while the
    broker sees neither objective names nor workload fields.
    """

    action_sha256: str
    source_opportunity: float
    archive_opportunity_scale: float
    evidence_sha256: str
    opportunity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _require_probability(
            self.source_opportunity,
            name="source_opportunity",
        )
        _require_probability(
            self.archive_opportunity_scale,
            name="archive_opportunity_scale",
        )
        if self.archive_opportunity_scale <= 0.0:
            raise ValueError("archive_opportunity_scale must be positive")
        if self.source_opportunity > self.archive_opportunity_scale:
            raise ValueError(
                "source opportunity cannot exceed the archive opportunity scale"
            )
        require_sha256(self.evidence_sha256, "evidence_sha256")
        object.__setattr__(
            self,
            "opportunity_sha256",
            _hash(
                _ACTION_OPPORTUNITY_EVIDENCE_DOMAIN,
                self._unsigned_record(),
            ),
        )

    @property
    def relative_source_opportunity(self) -> float:
        return self.source_opportunity / self.archive_opportunity_scale

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "source_opportunity_hex": self.source_opportunity.hex(),
            "archive_opportunity_scale_hex": (self.archive_opportunity_scale.hex()),
            "evidence_sha256": self.evidence_sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "relative_source_opportunity_hex": (self.relative_source_opportunity.hex()),
            "opportunity_sha256": self.opportunity_sha256,
        }


@runtime_checkable
class MaterializedActionOpportunityPort(Protocol):
    """Project strictly-prior archive geometry into one portable scalar."""

    definition_sha256: str

    def estimate(
        self,
        action: MaterializedActionDescriptor,
    ) -> MaterializedActionOpportunityEvidence | None: ...


@dataclass(frozen=True, slots=True)
class OpportunityConditionedMaterializedActionReturnValue:
    """Condition return on source opportunity and cap uncertainty in its units.

    Parent opportunity is bounded evidence about the size of the source basin,
    not outcome credit.  It can multiply the predicted mean by at most
    ``maximum_parent_multiplier``.  The same current-archive scale caps the
    standard deviation that otherwise defaults to a unit-interval worst case,
    preventing an unobserved arm from receiving orders-of-magnitude more
    information value solely because its prior is absent.
    """

    base: MaterializedActionReturnValuePort = field(
        repr=False,
        compare=False,
    )
    opportunity: MaterializedActionOpportunityPort = field(
        repr=False,
        compare=False,
    )
    maximum_parent_multiplier: float = 2.0
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base, MaterializedActionReturnValuePort):
            raise TypeError("base must implement MaterializedActionReturnValuePort")
        if not isinstance(
            self.opportunity,
            MaterializedActionOpportunityPort,
        ):
            raise TypeError(
                "opportunity must implement MaterializedActionOpportunityPort"
            )
        require_sha256(self.base.definition_sha256, "base definition_sha256")
        require_sha256(
            self.opportunity.definition_sha256,
            "opportunity definition_sha256",
        )
        if (
            type(self.maximum_parent_multiplier) is not float
            or not math.isfinite(self.maximum_parent_multiplier)
            or not 1.0 <= self.maximum_parent_multiplier <= 2.0
        ):
            raise ValueError(
                "maximum_parent_multiplier must be a finite float in [1, 2]"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _OPPORTUNITY_CONDITIONED_RETURN_VALUE_DOMAIN,
                {
                    "schema_version": 1,
                    "base_definition_sha256": self.base.definition_sha256,
                    "opportunity_definition_sha256": (
                        self.opportunity.definition_sha256
                    ),
                    "maximum_parent_multiplier_hex": (
                        self.maximum_parent_multiplier.hex()
                    ),
                    "mean": (
                        "base_mean_times_one_plus_bounded_relative_source_opportunity"
                    ),
                    "uncertainty": (
                        "minimum_of_base_standard_deviation_and_maximum_of_"
                        "conditioned_mean_and_archive_opportunity_scale"
                    ),
                    "strictly_prior_only": True,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def estimate(
        self,
        action: MaterializedActionDescriptor,
    ) -> BrokerReturnEstimate:
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be an exact MaterializedActionDescriptor")
        base = self.base.estimate(action)
        if type(base) is not BrokerReturnEstimate:
            raise TypeError("base return value produced a foreign estimate")
        base.__post_init__()
        opportunity = self.opportunity.estimate(action)
        if opportunity is None:
            return base
        if type(opportunity) is not MaterializedActionOpportunityEvidence:
            raise TypeError("opportunity port produced foreign evidence")
        opportunity.__post_init__()
        if opportunity.action_sha256 != action.action_sha256:
            raise ValueError("opportunity evidence identifies another action")
        multiplier = (
            1.0
            + (self.maximum_parent_multiplier - 1.0)
            * opportunity.relative_source_opportunity
        )
        mean = min(1.0, base.mean * multiplier)
        uncertainty_cap = max(
            opportunity.archive_opportunity_scale,
            mean,
        )
        return BrokerReturnEstimate(
            mean=float(mean),
            standard_deviation=float(min(base.standard_deviation, uncertainty_cap)),
            local_count=base.local_count,
            global_count=base.global_count,
            resolved_count=base.resolved_count,
            provisional_count=base.provisional_count,
            local_mean=base.local_mean,
            global_mean=base.global_mean,
            shrinkage_weight=base.shrinkage_weight,
        )


@dataclass(frozen=True, slots=True)
class EmpiricalBayesMaterializedActionReturnValue:
    """Fuse a frozen portable meta-prior with strictly prior live outcomes."""

    ledger: "MaterializedActionEvidenceLedger" = field(
        repr=False,
        compare=False,
    )
    prior: MaterializedActionReturnPriorPort = field(
        repr=False,
        compare=False,
    )
    hierarchical_kappa: float = 4.0
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.ledger) is not MaterializedActionEvidenceLedger:
            raise TypeError("ledger must be an exact MaterializedActionEvidenceLedger")
        if not isinstance(self.prior, MaterializedActionReturnPriorPort):
            raise TypeError("prior must implement MaterializedActionReturnPriorPort")
        require_sha256(self.prior.definition_sha256, "prior definition_sha256")
        if (
            type(self.hierarchical_kappa) is not float
            or not math.isfinite(self.hierarchical_kappa)
            or self.hierarchical_kappa <= 0.0
        ):
            raise ValueError("hierarchical_kappa must be a positive finite float")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _EMPIRICAL_BAYES_RETURN_VALUE_DOMAIN,
                {
                    "schema_version": 1,
                    "prior_definition_sha256": self.prior.definition_sha256,
                    "hierarchical_kappa_hex": self.hierarchical_kappa.hex(),
                    "fusion": (
                        "effective-sample-size-weighted-mean-and-total-variance"
                    ),
                    "live_evidence_cutoff": "strictly_prior_decision_index",
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def estimate(
        self,
        action: MaterializedActionDescriptor,
    ) -> BrokerReturnEstimate:
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be an exact MaterializedActionDescriptor")
        empirical = self.ledger.estimate_return(
            action,
            kappa=self.hierarchical_kappa,
        )
        prediction = self.prior.predict(action)
        if prediction is None:
            return empirical
        if type(prediction) is not MaterializedActionReturnPriorPrediction:
            raise TypeError("return prior produced a foreign prediction")
        prediction.__post_init__()
        if prediction.action_sha256 != action.action_sha256:
            raise ValueError("return prior prediction identifies another action")

        empirical_weight = empirical.global_count / (
            empirical.global_count + prediction.effective_sample_size
        )
        prior_weight = 1.0 - empirical_weight
        mean = prior_weight * prediction.mean + empirical_weight * empirical.mean
        variance = prior_weight * (
            prediction.standard_deviation**2 + (prediction.mean - mean) ** 2
        ) + empirical_weight * (
            empirical.standard_deviation**2 + (empirical.mean - mean) ** 2
        )
        return BrokerReturnEstimate(
            mean=float(min(1.0, max(0.0, mean))),
            standard_deviation=float(min(1.0, math.sqrt(max(0.0, variance)))),
            local_count=empirical.local_count,
            global_count=empirical.global_count,
            resolved_count=empirical.resolved_count,
            provisional_count=empirical.provisional_count,
            local_mean=empirical.local_mean,
            global_mean=empirical.global_mean,
            shrinkage_weight=empirical.shrinkage_weight,
        )


@dataclass(slots=True)
class MaterializedActionEvidenceLedger:
    """Append-only immediate and delayed evidence used by the broker."""

    outcomes: list[MaterializedActionOutcome] = field(default_factory=list)
    delayed_credits: list[MaterializedActionDelayedCredit] = field(default_factory=list)
    resolved_returns: list[MaterializedActionResolvedReturn] = field(
        default_factory=list
    )

    def append_outcome(self, value: MaterializedActionOutcome) -> None:
        if type(value) is not MaterializedActionOutcome:
            raise TypeError("value must be an exact MaterializedActionOutcome")
        value.__post_init__()
        if any(
            item.action.action_sha256 == value.action.action_sha256
            for item in self.outcomes
        ):
            raise ValueError(
                "one materialized action can have only one immediate outcome"
            )
        self.outcomes.append(value)

    def append_delayed_credit(self, value: MaterializedActionDelayedCredit) -> None:
        if type(value) is not MaterializedActionDelayedCredit:
            raise TypeError("value must be an exact MaterializedActionDelayedCredit")
        value.__post_init__()
        if not any(
            item.outcome_sha256 == value.outcome.outcome_sha256
            for item in self.outcomes
        ):
            raise ValueError(
                "delayed credit source is absent from the immediate ledger"
            )
        for existing in self.delayed_credits:
            if existing.outcome.outcome_sha256 != value.outcome.outcome_sha256:
                continue
            for name in (
                "stage_front_survived",
                "terminal_front_persisted",
                "useful_descendant_observed",
            ):
                if (
                    getattr(existing, name) is not None
                    and getattr(value, name) is not None
                ):
                    raise ValueError("a delayed channel can be adjudicated only once")
        self.delayed_credits.append(value)

    def append_resolved_return(self, value: MaterializedActionResolvedReturn) -> None:
        if type(value) is not MaterializedActionResolvedReturn:
            raise TypeError("value must be an exact MaterializedActionResolvedReturn")
        value.__post_init__()
        if not any(
            item.outcome_sha256 == value.outcome.outcome_sha256
            for item in self.outcomes
        ):
            raise ValueError("resolved return source is absent from the outcome ledger")
        prior = tuple(
            item
            for item in self.resolved_returns
            if item.outcome.outcome_sha256 == value.outcome.outcome_sha256
        )
        if prior:
            latest = max(prior, key=lambda item: item.available_at_decision_index)
            if value.available_at_decision_index <= latest.available_at_decision_index:
                raise ValueError("return resolutions must advance their availability")
            if value.horizon_end_decision_index < latest.horizon_end_decision_index:
                raise ValueError("return resolution horizon cannot move backward")
            if latest.fully_resolved:
                raise ValueError("a fully resolved return cannot be revised")
        self.resolved_returns.append(value)

    @staticmethod
    def _arm_matches(
        action: MaterializedActionDescriptor,
        target: MaterializedActionDescriptor,
    ) -> bool:
        return action.arm_key == target.arm_key

    def _values(
        self,
        action: MaterializedActionDescriptor,
        channel: BrokerEvidenceChannel,
        *,
        local: bool,
    ) -> tuple[float, ...]:
        def eligible(source: MaterializedActionDescriptor) -> bool:
            return self._arm_matches(source, action) and (
                not local
                or source.context.state_signature_sha256
                == action.context.state_signature_sha256
            )

        values: list[float] = []
        if channel is BrokerEvidenceChannel.REALIZATION:
            return tuple(
                1.0 if outcome.realized else 0.0
                for outcome in self.outcomes
                if eligible(outcome.action)
            )
        if channel in {
            BrokerEvidenceChannel.FEASIBILITY,
            BrokerEvidenceChannel.GAIN,
            BrokerEvidenceChannel.POSITIVE,
            BrokerEvidenceChannel.FORECAST_ERROR,
        }:
            for outcome in self.outcomes:
                if not eligible(outcome.action) or not outcome.realized:
                    continue
                if channel is BrokerEvidenceChannel.FEASIBILITY:
                    assert outcome.feasible is not None
                    values.append(1.0 if outcome.feasible else 0.0)
                elif channel is BrokerEvidenceChannel.GAIN:
                    assert outcome.normalized_archive_gain is not None
                    values.append(outcome.normalized_archive_gain)
                elif channel is BrokerEvidenceChannel.POSITIVE:
                    assert outcome.positive_marginal_utility is not None
                    values.append(1.0 if outcome.positive_marginal_utility else 0.0)
                elif outcome.normalized_forecast_error is not None:
                    values.append(outcome.normalized_forecast_error)
            return tuple(values)
        field_name = {
            BrokerEvidenceChannel.STAGE_SURVIVAL: "stage_front_survived",
            BrokerEvidenceChannel.TERMINAL_PERSISTENCE: "terminal_front_persisted",
            BrokerEvidenceChannel.DESCENDANT: "useful_descendant_observed",
        }[channel]
        for credit in self.delayed_credits:
            if not eligible(credit.outcome.action):
                continue
            verdict = getattr(credit, field_name)
            if verdict is not None:
                values.append(1.0 if verdict else 0.0)
        return tuple(values)

    def estimate(
        self,
        action: MaterializedActionDescriptor,
        channel: BrokerEvidenceChannel,
        *,
        kappa: float,
    ) -> BrokerChannelEstimate:
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be an exact MaterializedActionDescriptor")
        if type(channel) is not BrokerEvidenceChannel:
            raise TypeError("channel must be an exact BrokerEvidenceChannel")
        if type(kappa) is not float or not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be a positive finite exact float")
        local_values = self._values(action, channel, local=True)
        global_values = self._values(action, channel, local=False)
        local_mean, local_sd, local_count_float = _bounded_empirical(local_values)
        global_mean, global_sd, global_count_float = _bounded_empirical(global_values)
        local_count = int(local_count_float)
        global_count = int(global_count_float)
        weight = local_count / (local_count + kappa)
        mean = weight * local_mean + (1.0 - weight) * global_mean
        sd = math.sqrt(weight * local_sd**2 + (1.0 - weight) * global_sd**2)
        return BrokerChannelEstimate(
            channel=channel,
            mean=float(mean),
            standard_deviation=float(min(1.0, sd)),
            local_count=local_count,
            global_count=global_count,
            local_mean=float(local_mean),
            global_mean=float(global_mean),
            shrinkage_weight=float(weight),
        )

    def _return_values(
        self,
        action: MaterializedActionDescriptor,
        *,
        local: bool,
    ) -> tuple[tuple[float, bool], ...]:
        """Return prior-only values and whether each is lineage-resolved."""

        cutoff = action.context.decision_index
        newest: dict[str, MaterializedActionResolvedReturn] = {}
        for value in self.resolved_returns:
            if value.available_at_decision_index >= cutoff:
                continue
            key = value.outcome.outcome_sha256
            prior = newest.get(key)
            if prior is None or (
                value.available_at_decision_index > prior.available_at_decision_index
            ):
                newest[key] = value
        values: list[tuple[float, bool]] = []
        for outcome in self.outcomes:
            source = outcome.action
            if source.context.decision_index >= cutoff:
                continue
            if not self._arm_matches(source, action):
                continue
            if local and (
                source.context.state_signature_sha256
                != action.context.state_signature_sha256
            ):
                continue
            resolved = newest.get(outcome.outcome_sha256)
            if resolved is not None:
                values.append((resolved.normalized_horizon_return, True))
                continue
            # Immediate real archive gain is a censored lower-bound observation
            # until a lineage resolver publishes a strictly later resolution.
            provisional = (
                0.0 if not outcome.realized else outcome.normalized_archive_gain
            )
            assert provisional is not None
            values.append((provisional, False))
        return tuple(values)

    def estimate_return(
        self,
        action: MaterializedActionDescriptor,
        *,
        kappa: float,
    ) -> BrokerReturnEstimate:
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be an exact MaterializedActionDescriptor")
        if type(kappa) is not float or not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa must be a positive finite exact float")
        local_rows = self._return_values(action, local=True)
        global_rows = self._return_values(action, local=False)
        local_values = tuple(value for value, _resolved in local_rows)
        global_values = tuple(value for value, _resolved in global_rows)
        local_mean, local_sd, local_count_float = _bounded_empirical(local_values)
        global_mean, global_sd, global_count_float = _bounded_empirical(global_values)
        local_count = int(local_count_float)
        global_count = int(global_count_float)
        weight = local_count / (local_count + kappa)
        mean = weight * local_mean + (1.0 - weight) * global_mean
        sd = math.sqrt(weight * local_sd**2 + (1.0 - weight) * global_sd**2)
        resolved_count = sum(resolved for _value, resolved in global_rows)
        return BrokerReturnEstimate(
            mean=float(mean),
            standard_deviation=float(min(1.0, sd)),
            local_count=local_count,
            global_count=global_count,
            resolved_count=resolved_count,
            provisional_count=global_count - resolved_count,
            local_mean=float(local_mean),
            global_mean=float(global_mean),
            shrinkage_weight=float(weight),
        )


@dataclass(frozen=True, slots=True)
class BrokerActionScore:
    action_sha256: str
    value: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    selection_index: float
    return_estimator_definition_sha256: str
    return_estimate: BrokerReturnEstimate
    estimates: tuple[BrokerChannelEstimate, ...]

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        for name in (
            "value",
            "lower_confidence_bound",
            "upper_confidence_bound",
            "selection_index",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite exact float")
        if self.lower_confidence_bound > self.value:
            raise ValueError("lower confidence bound exceeds value")
        if self.upper_confidence_bound < self.value:
            raise ValueError("upper confidence bound is below value")
        if not 0.0 <= self.selection_index <= 1.0:
            raise ValueError("selection_index must lie in [0, 1]")
        require_sha256(
            self.return_estimator_definition_sha256,
            "return_estimator_definition_sha256",
        )
        if type(self.return_estimate) is not BrokerReturnEstimate:
            raise TypeError("return_estimate must be exact")
        self.return_estimate.__post_init__()
        if type(self.estimates) is not tuple or tuple(
            item.channel for item in self.estimates
        ) != tuple(BrokerEvidenceChannel):
            raise ValueError("estimates must cover every channel in canonical order")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "value_hex": self.value.hex(),
            "lower_confidence_bound_hex": self.lower_confidence_bound.hex(),
            "upper_confidence_bound_hex": self.upper_confidence_bound.hex(),
            "selection_index_hex": self.selection_index.hex(),
            "return_estimator_definition_sha256": (
                self.return_estimator_definition_sha256
            ),
            "return_estimate": self.return_estimate.to_record(),
            "estimates": [value.to_record() for value in self.estimates],
        }


@runtime_checkable
class MaterializedSlateValuePort(Protocol):
    """Predict residual complementarity not explained by member returns.

    The value must lie in ``[0, 1]``.  Zero means that the member-level return
    posterior completely explains the slate.  One means that the slate is
    expected to capture all return headroom remaining after additive member
    credit.  This closed meaning composes with coalition-efficient action
    returns without a hand-authored joint/individual mixing weight.
    """

    definition_sha256: str

    def value(self, actions: tuple[MaterializedActionDescriptor, ...]) -> float: ...


@runtime_checkable
class MaterializedSlateFeasibilityPort(Protocol):
    """Apply exact generic materialization constraints to one slate."""

    definition_sha256: str

    def permits(self, actions: tuple[MaterializedActionDescriptor, ...]) -> bool: ...


@dataclass(frozen=True, slots=True)
class MaterializedActionExplorationRequirement:
    """Authenticated actions reserved by a strictly prequential policy."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    required_action_sha256s: tuple[str, ...]
    prior_outcome_count: int
    cold_start: bool
    evidence: FrozenJsonObject
    requirement_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        if type(self.required_action_sha256s) is not tuple:
            raise TypeError("required action hashes must be an exact tuple")
        if self.required_action_sha256s != tuple(
            sorted(set(self.required_action_sha256s))
        ):
            raise ValueError("required action hashes must be unique and canonical")
        for value in self.required_action_sha256s:
            require_sha256(value, "required action sha256")
        if type(self.prior_outcome_count) is not int or self.prior_outcome_count < 0:
            raise ValueError("prior_outcome_count must be non-negative")
        if type(self.cold_start) is not bool:
            raise TypeError("cold_start must be an exact bool")
        if self.cold_start != (self.prior_outcome_count == 0):
            raise ValueError("cold_start must exactly reflect prior outcome count")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("exploration evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "requirement_sha256",
            _hash(
                _EXPLORATION_REQUIREMENT_DOMAIN,
                self._unsigned_record(),
            ),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "required_action_sha256s": list(self.required_action_sha256s),
            "prior_outcome_count": self.prior_outcome_count,
            "cold_start": self.cold_start,
            "evidence_sha256": typed_json_sha256(self.evidence),
            "strictly_prior_outcomes_only": True,
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "requirement_sha256": self.requirement_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@dataclass(frozen=True, slots=True)
class MaterializedActionAllocationRequirement:
    """Authenticated outcome-blind constraint produced after proposal sealing.

    The application core treats the policy and its evidence as opaque.  The
    hashes bind the requirement to one residual request and one exact proposal
    universe, while ``candidate_outcomes_observed`` makes the information
    boundary explicit and mechanically rejects post-hoc selectors.
    """

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    required_action_sha256s: tuple[str, ...]
    candidate_outcomes_observed: bool
    evidence: FrozenJsonObject
    requirement_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if type(self.proposal_sha256s) is not tuple or not self.proposal_sha256s:
            raise ValueError("proposal hashes must be a non-empty exact tuple")
        if self.proposal_sha256s != tuple(sorted(set(self.proposal_sha256s))):
            raise ValueError("proposal hashes must be unique and canonical")
        for value in self.proposal_sha256s:
            require_sha256(value, "proposal sha256")
        if type(self.required_action_sha256s) is not tuple:
            raise TypeError("required action hashes must be an exact tuple")
        if self.required_action_sha256s != tuple(
            sorted(set(self.required_action_sha256s))
        ):
            raise ValueError("required action hashes must be unique and canonical")
        for value in self.required_action_sha256s:
            require_sha256(value, "required action sha256")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be an exact bool")
        if self.candidate_outcomes_observed:
            raise ValueError(
                "allocation requirements cannot observe candidate outcomes"
            )
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("allocation evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "requirement_sha256",
            _hash(
                _ALLOCATION_REQUIREMENT_DOMAIN,
                self._unsigned_record(),
            ),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "required_action_sha256s": list(self.required_action_sha256s),
            "candidate_outcomes_observed": self.candidate_outcomes_observed,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "requirement_sha256": self.requirement_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


MaterializedActionAllocationConstraint = (
    MaterializedActionAllocationRequirement | AdaptiveActionAllocationDirective
)


def _validate_allocation_constraint(
    value: MaterializedActionAllocationConstraint,
) -> None:
    """Accept only the two authenticated allocation information boundaries."""

    if type(value) not in (
        MaterializedActionAllocationRequirement,
        AdaptiveActionAllocationDirective,
    ):
        raise TypeError(
            "allocation_requirement must be an exact static requirement, "
            "an exact adaptive directive, or None"
        )
    value.__post_init__()


@runtime_checkable
class MaterializedActionExplorationPort(Protocol):
    """Reserve a bounded action subset using only current proposals and priors."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def require(
        self,
        request: "MaterializedActionBrokerRequest",
        ledger: MaterializedActionEvidenceLedger,
        required_reference_action_sha256s: tuple[str, ...],
    ) -> MaterializedActionExplorationRequirement: ...


@dataclass(frozen=True, slots=True)
class MaterializedActionBrokerRequest:
    actions: tuple[MaterializedActionDescriptor, ...]
    evaluation_slots: int
    slate_value: MaterializedSlateValuePort
    slate_feasibility: MaterializedSlateFeasibilityPort
    reference_escrow_slots: int = 1
    allocation_requirement: MaterializedActionAllocationConstraint | None = None

    def __post_init__(self) -> None:
        if type(self.actions) is not tuple or not self.actions:
            raise ValueError("actions must be a non-empty exact tuple")
        for action in self.actions:
            if type(action) is not MaterializedActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            action.__post_init__()
        if len({value.action_sha256 for value in self.actions}) != len(self.actions):
            raise ValueError("action identities must be unique")
        contexts = {
            (value.context.campaign_scope_sha256, value.context.decision_index)
            for value in self.actions
        }
        if len(contexts) != 1:
            raise ValueError("one broker request cannot mix decision cutoffs")
        if type(
            self.evaluation_slots
        ) is not int or not 1 <= self.evaluation_slots <= len(self.actions):
            raise ValueError("evaluation_slots must fit the supplied action universe")
        if not isinstance(self.slate_value, MaterializedSlateValuePort):
            raise TypeError("slate_value must implement MaterializedSlateValuePort")
        if not isinstance(self.slate_feasibility, MaterializedSlateFeasibilityPort):
            raise TypeError(
                "slate_feasibility must implement MaterializedSlateFeasibilityPort"
            )
        require_sha256(self.slate_value.definition_sha256, "slate value definition")
        require_sha256(
            self.slate_feasibility.definition_sha256,
            "slate feasibility definition",
        )
        if (
            type(self.reference_escrow_slots) is not int
            or not 0 <= self.reference_escrow_slots <= self.evaluation_slots
        ):
            raise ValueError("reference_escrow_slots must fit the evaluation capacity")
        if self.allocation_requirement is not None:
            _validate_allocation_constraint(self.allocation_requirement)
            action_sha256s = {value.action_sha256 for value in self.actions}
            if not set(self.allocation_requirement.required_action_sha256s).issubset(
                action_sha256s
            ):
                raise ValueError(
                    "allocation policy required an action outside the request"
                )
            if (
                len(self.allocation_requirement.required_action_sha256s)
                > self.evaluation_slots
            ):
                raise ValueError("allocation requirement exceeds evaluation capacity")


@dataclass(frozen=True, slots=True)
class MaterializedActionBrokerDecision:
    selected_actions: tuple[MaterializedActionDescriptor, ...]
    scores: tuple[BrokerActionScore, ...]
    required_reference_action_sha256s: tuple[str, ...]
    exploration_requirement: MaterializedActionExplorationRequirement | None
    allocation_requirement: MaterializedActionAllocationConstraint | None
    reference_displaced_count: int
    search_mode: str
    complete_slate_count_considered: int
    residual_complementarity_value: float
    exploration_action_sha256: str | None
    broker_definition_sha256: str = MATERIALIZED_ACTION_BROKER_DEFINITION_SHA256
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.selected_actions) is not tuple or not self.selected_actions:
            raise ValueError("selected_actions must be a non-empty exact tuple")
        if len(
            {value.phenotype_identity_sha256 for value in self.selected_actions}
        ) != len(self.selected_actions):
            raise ValueError("selected actions must have unique phenotypes")
        if tuple(
            sorted(value.action_sha256 for value in self.selected_actions)
        ) != tuple(value.action_sha256 for value in self.selected_actions):
            raise ValueError("selected actions must use canonical action order")
        if type(self.scores) is not tuple or not self.scores:
            raise ValueError("scores must be a non-empty exact tuple")
        if type(self.required_reference_action_sha256s) is not tuple:
            raise TypeError("required reference identities must be an exact tuple")
        if tuple(sorted(set(self.required_reference_action_sha256s))) != (
            self.required_reference_action_sha256s
        ):
            raise ValueError("required reference identities must be unique/canonical")
        selected_sha256s = {value.action_sha256 for value in self.selected_actions}
        for value in self.required_reference_action_sha256s:
            require_sha256(value, "required reference action sha256")
            if value not in selected_sha256s:
                raise ValueError("required reference action is absent from the slate")
        if self.exploration_requirement is not None:
            if (
                type(self.exploration_requirement)
                is not MaterializedActionExplorationRequirement
            ):
                raise TypeError("exploration_requirement must be exact or None")
            self.exploration_requirement.__post_init__()
            if not set(self.exploration_requirement.required_action_sha256s).issubset(
                selected_sha256s
            ):
                raise ValueError("required exploration action is absent from the slate")
        if self.allocation_requirement is not None:
            _validate_allocation_constraint(self.allocation_requirement)
            if not set(self.allocation_requirement.required_action_sha256s).issubset(
                selected_sha256s
            ):
                raise ValueError("required allocation action is absent from the slate")
        if (
            type(self.reference_displaced_count) is not int
            or self.reference_displaced_count < 0
        ):
            raise ValueError("reference_displaced_count must be non-negative")
        _require_token(self.search_mode, name="search_mode")
        if (
            type(self.complete_slate_count_considered) is not int
            or self.complete_slate_count_considered <= 0
        ):
            raise ValueError("at least one complete slate must be considered")
        _require_probability(
            self.residual_complementarity_value,
            name="residual_complementarity_value",
        )
        if self.exploration_action_sha256 is not None:
            require_sha256(
                self.exploration_action_sha256,
                "exploration_action_sha256",
            )
            selected = {value.action_sha256: value for value in self.selected_actions}
            action = selected.get(self.exploration_action_sha256)
            if action is None:
                raise ValueError("exploration action is absent from selected slate")
            if action.context.phase is SearchPhase.TERMINAL_CONVERSION:
                raise ValueError("terminal decisions cannot purchase information")
        require_sha256(self.broker_definition_sha256, "broker_definition_sha256")
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_DECISION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "broker_id": MATERIALIZED_ACTION_BROKER_ID,
            "broker_version": MATERIALIZED_ACTION_BROKER_VERSION,
            "broker_definition_sha256": self.broker_definition_sha256,
            "selected_action_sha256s": [
                value.action_sha256 for value in self.selected_actions
            ],
            "scores": [value.to_record() for value in self.scores],
            "required_reference_action_sha256s": list(
                self.required_reference_action_sha256s
            ),
            "exploration_requirement": (
                None
                if self.exploration_requirement is None
                else self.exploration_requirement.to_record()
            ),
            "allocation_requirement": (
                None
                if self.allocation_requirement is None
                else self.allocation_requirement.to_record()
            ),
            "reference_displaced_count": self.reference_displaced_count,
            "search_mode": self.search_mode,
            "complete_slate_count_considered": self.complete_slate_count_considered,
            "residual_complementarity_value_hex": (
                self.residual_complementarity_value.hex()
            ),
            "exploration_action_sha256": self.exploration_action_sha256,
        }

    @property
    def required_reference_action_sha256(self) -> str | None:
        """Compatibility view for callers that escrow exactly one reference."""

        if not self.required_reference_action_sha256s:
            return None
        return self.required_reference_action_sha256s[0]

    @property
    def reference_displaced(self) -> bool:
        return self.reference_displaced_count > 0

    def to_record(
        self,
        *,
        include_allocation_evidence: bool = False,
    ) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "decision_sha256": self.decision_sha256,
        }
        if include_allocation_evidence and self.allocation_requirement is not None:
            record["allocation_requirement"] = self.allocation_requirement.to_record(
                include_evidence=True
            )
        return record


@dataclass(frozen=True, slots=True)
class RegretBrokeredMaterializedActionPolicy:
    """Broker normalized lineage return without cross-channel score weights."""

    ledger: MaterializedActionEvidenceLedger
    return_value: MaterializedActionReturnValuePort | None = None
    exploration_policy: MaterializedActionExplorationPort | None = None
    hierarchical_kappa: float = 4.0
    confidence_width: float = 1.0
    exact_combination_limit: int = 250_000
    beam_width: int = 512

    def __post_init__(self) -> None:
        if type(self.ledger) is not MaterializedActionEvidenceLedger:
            raise TypeError("ledger must be an exact MaterializedActionEvidenceLedger")
        if self.return_value is not None:
            if not isinstance(self.return_value, MaterializedActionReturnValuePort):
                raise TypeError(
                    "return_value must implement MaterializedActionReturnValuePort"
                )
            require_sha256(
                self.return_value.definition_sha256,
                "return value definition_sha256",
            )
        if self.exploration_policy is not None:
            if not isinstance(
                self.exploration_policy,
                MaterializedActionExplorationPort,
            ):
                raise TypeError(
                    "exploration_policy must implement "
                    "MaterializedActionExplorationPort"
                )
            _require_token(
                self.exploration_policy.policy_id,
                name="exploration policy_id",
            )
            if (
                type(self.exploration_policy.policy_version) is not int
                or self.exploration_policy.policy_version <= 0
            ):
                raise ValueError("exploration policy_version must be positive")
            require_sha256(
                self.exploration_policy.definition_sha256,
                "exploration policy definition_sha256",
            )
        for name in ("hierarchical_kappa", "confidence_width"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite exact float")
        for name in ("exact_combination_limit", "beam_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")

    def score(self, action: MaterializedActionDescriptor) -> BrokerActionScore:
        if self.return_value is None:
            return_estimate = self.ledger.estimate_return(
                action,
                kappa=self.hierarchical_kappa,
            )
            return_estimator_definition_sha256 = (
                EMPIRICAL_RETURN_ESTIMATOR_DEFINITION_SHA256
            )
        else:
            return_estimate = self.return_value.estimate(action)
            if type(return_estimate) is not BrokerReturnEstimate:
                raise TypeError("return value port returned a foreign estimate")
            return_estimate.__post_init__()
            return_estimator_definition_sha256 = self.return_value.definition_sha256
        estimates = tuple(
            self.ledger.estimate(
                action,
                channel,
                kappa=self.hierarchical_kappa,
            )
            for channel in BrokerEvidenceChannel
        )
        value = return_estimate.mean
        radius = self.confidence_width * return_estimate.standard_deviation
        lower = max(0.0, value - radius)
        upper = min(1.0, value + radius)
        # Information has no endpoint value when there is no future decision
        # that can consume it.  Earlier waves use the posterior upper bound as
        # the action index; the terminal wave uses posterior mean exactly.
        selection_index = (
            value if action.context.phase is SearchPhase.TERMINAL_CONVERSION else upper
        )
        return BrokerActionScore(
            action_sha256=action.action_sha256,
            value=float(value),
            lower_confidence_bound=float(lower),
            upper_confidence_bound=float(upper),
            selection_index=float(selection_index),
            return_estimator_definition_sha256=(return_estimator_definition_sha256),
            return_estimate=return_estimate,
            estimates=estimates,
        )

    @staticmethod
    def _canonical_slate(
        actions: tuple[MaterializedActionDescriptor, ...],
    ) -> tuple[MaterializedActionDescriptor, ...]:
        return tuple(sorted(actions, key=lambda value: value.action_sha256))

    @staticmethod
    def _cold_start_tie_key(
        actions: tuple[MaterializedActionDescriptor, ...],
    ) -> tuple[tuple[int, ...], int, tuple[str, ...]]:
        """Respect native expert order without comparing heterogeneous scores."""

        return (
            tuple(sorted(value.native_rank for value in actions)),
            -len({value.expert_id for value in actions}),
            tuple(value.action_sha256 for value in actions),
        )

    @staticmethod
    def _unique_phenotypes(
        actions: tuple[MaterializedActionDescriptor, ...],
    ) -> bool:
        return len({value.phenotype_identity_sha256 for value in actions}) == len(
            actions
        )

    def _slate_score(
        self,
        actions: tuple[MaterializedActionDescriptor, ...],
        scores: dict[str, BrokerActionScore],
        value_port: MaterializedSlateValuePort,
        cache: dict[tuple[str, ...], tuple[float, float, str | None]],
    ) -> tuple[float, float, str | None]:
        identity = tuple(value.action_sha256 for value in actions)
        cached = cache.get(identity)
        if cached is not None:
            return cached
        residual_complementarity = value_port.value(actions)
        _require_probability(
            residual_complementarity,
            name="residual complementarity",
        )
        # Shapley-attributed archive returns are additive contributions, not
        # independent success probabilities.  Sum their empirical means and
        # cap only at the normalized archive-return boundary.
        member_return = min(
            1.0,
            math.fsum(scores[value.action_sha256].value for value in actions),
        )

        # Information is a scarce action, not a bonus silently attached to
        # every member.  At most one nonterminal member receives its optimistic
        # increment; all remaining slots are selected by empirical return.
        exploration_action_sha256: str | None = None
        if actions[0].context.phase is not SearchPhase.TERMINAL_CONVERSION:
            exploratory = max(
                actions,
                key=lambda value: (
                    scores[value.action_sha256].selection_index
                    - scores[value.action_sha256].value,
                    scores[value.action_sha256].selection_index,
                    value.action_sha256,
                ),
            )
            exploratory_score = scores[exploratory.action_sha256]
            exploration_increment = min(
                1.0 - member_return,
                exploratory_score.selection_index - exploratory_score.value,
            )
            if exploration_increment > 0.0:
                member_return += exploration_increment
                exploration_action_sha256 = exploratory.action_sha256

        slate_return = member_return + (1.0 - member_return) * residual_complementarity
        result = (
            float(slate_return),
            residual_complementarity,
            exploration_action_sha256,
        )
        cache[identity] = result
        return result

    def _required_references(
        self,
        request: MaterializedActionBrokerRequest,
        scores: dict[str, BrokerActionScore],
    ) -> tuple[tuple[str, ...], int]:
        if request.reference_escrow_slots == 0:
            return (), 0
        references = tuple(value for value in request.actions if value.reference_action)
        nonreferences = tuple(
            value for value in request.actions if not value.reference_action
        )
        if not references:
            return (), 0
        ordered_references = sorted(
            references,
            key=lambda value: (
                value.native_rank,
                -scores[value.action_sha256].value,
                value.action_sha256,
            ),
        )
        protected = ordered_references[: request.reference_escrow_slots]
        if not nonreferences or not protected:
            return tuple(sorted(value.action_sha256 for value in protected)), 0
        challengers = sorted(
            nonreferences,
            key=lambda value: (
                -scores[value.action_sha256].lower_confidence_bound,
                -scores[value.action_sha256].value,
                value.action_sha256,
            ),
        )
        retained = list(protected)
        displaced_count = 0
        # Challenge the weakest protected reference first.  A challenger only
        # removes escrow authority when its lower bound is strictly above that
        # reference's upper bound; the joint optimizer still decides whether
        # the challenger belongs in the final slate.
        for challenger, reference in zip(
            challengers,
            reversed(protected),
            strict=False,
        ):
            if (
                scores[challenger.action_sha256].lower_confidence_bound
                > scores[reference.action_sha256].upper_confidence_bound
            ):
                retained.remove(reference)
                displaced_count += 1
        return (
            tuple(sorted(value.action_sha256 for value in retained)),
            displaced_count,
        )

    def _admissible(
        self,
        slate: tuple[MaterializedActionDescriptor, ...],
        request: MaterializedActionBrokerRequest,
        required_references: tuple[str, ...],
        cache: dict[tuple[tuple[str, ...], tuple[str, ...]], bool],
    ) -> bool:
        identity = tuple(value.action_sha256 for value in slate)
        cache_key = (identity, required_references)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        result = (
            self._unique_phenotypes(slate)
            and (
                not required_references
                or set(required_references).issubset(
                    {value.action_sha256 for value in slate}
                )
            )
            and request.slate_feasibility.permits(slate)
        )
        cache[cache_key] = result
        return result

    def _exact_search(
        self,
        request: MaterializedActionBrokerRequest,
        scores: dict[str, BrokerActionScore],
        required_references: tuple[str, ...],
        slate_score_cache: dict[tuple[str, ...], tuple[float, float, str | None]],
        admissibility_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], bool],
    ) -> tuple[tuple[MaterializedActionDescriptor, ...], float, str | None, int]:
        action_by_sha256 = {value.action_sha256: value for value in request.actions}
        if not set(required_references).issubset(action_by_sha256):
            raise ValueError("required action is absent from the broker request")
        required = tuple(action_by_sha256[value] for value in required_references)
        optional = tuple(
            value
            for value in request.actions
            if value.action_sha256 not in required_references
        )
        remaining_slots = request.evaluation_slots - len(required)
        if remaining_slots < 0:
            raise ValueError("required actions exceed evaluation capacity")
        best: tuple[MaterializedActionDescriptor, ...] | None = None
        best_score = -math.inf
        best_joint = 0.0
        best_exploration: str | None = None
        count = 0
        for raw in combinations(optional, remaining_slots):
            slate = self._canonical_slate((*required, *raw))
            if not self._admissible(
                slate,
                request,
                required_references,
                admissibility_cache,
            ):
                continue
            count += 1
            score, joint, exploration = self._slate_score(
                slate,
                scores,
                request.slate_value,
                slate_score_cache,
            )
            tie_key = self._cold_start_tie_key(slate)
            best_tie_key = None if best is None else self._cold_start_tie_key(best)
            if score > best_score or (
                score == best_score and (best_tie_key is None or tie_key < best_tie_key)
            ):
                best, best_score, best_joint, best_exploration = (
                    slate,
                    score,
                    joint,
                    exploration,
                )
        if best is None:
            raise ValueError("no feasible complete materialized-action slate exists")
        return best, best_joint, best_exploration, count

    def _beam_search(
        self,
        request: MaterializedActionBrokerRequest,
        scores: dict[str, BrokerActionScore],
        required_references: tuple[str, ...],
        slate_score_cache: dict[tuple[str, ...], tuple[float, float, str | None]],
        admissibility_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], bool],
    ) -> tuple[tuple[MaterializedActionDescriptor, ...], float, str | None, int]:
        action_by_sha256 = {value.action_sha256: value for value in request.actions}
        if not set(required_references).issubset(action_by_sha256):
            raise ValueError("required action is absent from the broker request")
        required = tuple(action_by_sha256[value] for value in required_references)
        ordered = tuple(
            sorted(
                (
                    value
                    for value in request.actions
                    if value.action_sha256 not in required_references
                ),
                key=lambda value: value.action_sha256,
            )
        )
        remaining_slots = request.evaluation_slots - len(required)
        if remaining_slots < 0:
            raise ValueError("required actions exceed evaluation capacity")
        if not self._unique_phenotypes(required):
            raise ValueError("required actions repeat a materialized phenotype")
        beam: tuple[tuple[MaterializedActionDescriptor, ...], ...] = ((),)
        complete: dict[
            tuple[str, ...],
            tuple[
                tuple[MaterializedActionDescriptor, ...],
                float,
                float,
                str | None,
            ],
        ] = {}
        for _depth in range(remaining_slots):
            expanded: dict[
                tuple[str, ...], tuple[MaterializedActionDescriptor, ...]
            ] = {}
            for partial in beam:
                # ``partial`` is already canonical.  Extending only with a
                # larger identity enumerates each unordered slate once instead
                # of revisiting every permutation before dictionary dedup.
                lower_bound = "" if not partial else partial[-1].action_sha256
                for action in ordered:
                    if action.action_sha256 <= lower_bound:
                        continue
                    candidate = (*partial, action)
                    candidate_slate = self._canonical_slate((*required, *candidate))
                    if not self._unique_phenotypes(candidate_slate):
                        continue
                    remaining_after_candidate = remaining_slots - len(candidate)
                    if remaining_after_candidate:
                        used_phenotypes = {
                            value.phenotype_identity_sha256 for value in candidate_slate
                        }
                        available_suffix_phenotypes = {
                            value.phenotype_identity_sha256
                            for value in ordered
                            if (
                                value.action_sha256 > action.action_sha256
                                and value.phenotype_identity_sha256
                                not in used_phenotypes
                            )
                        }
                        if len(available_suffix_phenotypes) < remaining_after_candidate:
                            # The canonical identity ordering is an
                            # enumeration device, not a quality signal.  A
                            # bounded beam must not retain a high-scoring
                            # late-identity partial that can no longer be
                            # completed while pruning every feasible prefix.
                            continue
                    if not self._admissible(
                        candidate_slate,
                        request,
                        required_references=(),
                        cache=admissibility_cache,
                    ):
                        continue
                    identity = tuple(value.action_sha256 for value in candidate)
                    expanded[identity] = candidate
            ranked = sorted(
                expanded.values(),
                key=lambda partial: (
                    -self._slate_score(
                        self._canonical_slate((*required, *partial)),
                        scores,
                        request.slate_value,
                        slate_score_cache,
                    )[0],
                    self._cold_start_tie_key(
                        self._canonical_slate((*required, *partial))
                    ),
                ),
            )
            beam = tuple(ranked[: self.beam_width])
            if not beam:
                break
        for partial in beam:
            slate = self._canonical_slate((*required, *partial))
            if len(slate) != request.evaluation_slots or not self._admissible(
                slate,
                request,
                required_references,
                admissibility_cache,
            ):
                continue
            score, joint, exploration = self._slate_score(
                slate,
                scores,
                request.slate_value,
                slate_score_cache,
            )
            identity = tuple(value.action_sha256 for value in slate)
            complete[identity] = (slate, score, joint, exploration)
        if not complete:
            raise ValueError(
                "beam search found no feasible complete materialized slate"
            )
        best = min(
            complete.values(),
            key=lambda value: (
                -value[1],
                self._cold_start_tie_key(value[0]),
            ),
        )
        return best[0], best[2], best[3], len(complete)

    def select(
        self,
        request: MaterializedActionBrokerRequest,
    ) -> MaterializedActionBrokerDecision:
        if type(request) is not MaterializedActionBrokerRequest:
            raise TypeError("request must be an exact MaterializedActionBrokerRequest")
        request.__post_init__()
        scores = {value.action_sha256: self.score(value) for value in request.actions}
        required_references, displaced_count = self._required_references(
            request,
            scores,
        )
        exploration_requirement = (
            None
            if self.exploration_policy is None
            else self.exploration_policy.require(
                request,
                self.ledger,
                required_references,
            )
        )
        if exploration_requirement is not None:
            if (
                type(exploration_requirement)
                is not MaterializedActionExplorationRequirement
            ):
                raise TypeError("exploration policy returned a foreign requirement")
            exploration_requirement.__post_init__()
            if (
                exploration_requirement.policy_id != self.exploration_policy.policy_id
                or exploration_requirement.policy_version
                != self.exploration_policy.policy_version
                or exploration_requirement.policy_definition_sha256
                != self.exploration_policy.definition_sha256
            ):
                raise ValueError("exploration requirement differs from its policy")
            action_sha256s = {value.action_sha256 for value in request.actions}
            if not set(exploration_requirement.required_action_sha256s).issubset(
                action_sha256s
            ):
                raise ValueError(
                    "exploration policy required an action outside the request"
                )
        required_exploration = (
            ()
            if exploration_requirement is None
            else exploration_requirement.required_action_sha256s
        )
        required_allocation = (
            ()
            if request.allocation_requirement is None
            else request.allocation_requirement.required_action_sha256s
        )
        required_actions = tuple(
            sorted(
                set(required_references)
                | set(required_exploration)
                | set(required_allocation)
            )
        )
        if len(required_actions) > request.evaluation_slots:
            raise ValueError("reference and exploration requirements exceed capacity")
        slate_score_cache: dict[tuple[str, ...], tuple[float, float, str | None]] = {}
        admissibility_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], bool] = {}
        optional_action_count = len(request.actions) - len(required_actions)
        remaining_slot_count = request.evaluation_slots - len(required_actions)
        combination_count = math.comb(
            optional_action_count,
            remaining_slot_count,
        )
        if combination_count <= self.exact_combination_limit:
            selected, joint, exploration, considered = self._exact_search(
                request,
                scores,
                required_actions,
                slate_score_cache,
                admissibility_cache,
            )
            mode = "exact_joint"
        else:
            selected, joint, exploration, considered = self._beam_search(
                request,
                scores,
                required_actions,
                slate_score_cache,
                admissibility_cache,
            )
            mode = "bounded_joint_beam"
        ordered_scores = tuple(
            scores[action.action_sha256]
            for action in sorted(request.actions, key=lambda value: value.action_sha256)
        )
        return MaterializedActionBrokerDecision(
            selected_actions=selected,
            scores=ordered_scores,
            required_reference_action_sha256s=required_references,
            exploration_requirement=exploration_requirement,
            allocation_requirement=request.allocation_requirement,
            reference_displaced_count=displaced_count,
            search_mode=mode,
            complete_slate_count_considered=considered,
            residual_complementarity_value=joint,
            exploration_action_sha256=exploration,
        )

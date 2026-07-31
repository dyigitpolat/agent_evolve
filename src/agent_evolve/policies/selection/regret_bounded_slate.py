"""Regret-bounded finite-slate selection with interceptable future value.

The strongest strictly-prior numerical slate remains the live anchor.  A
challenger can be evaluated only when its frozen acquisition value lies inside
an explicit multiplicative retention envelope.  Inside that envelope, an
injected policy may price authenticated information, descendant capacity, and
risk.  An optional identification quota requires a bounded number of members
outside the all-anchor reference, preventing a zero-evidence source from being
starved forever; the policy fails closed when no such slate satisfies the
retention envelope.  The language model never owns the final decision, and the
complete anchor remains a scored, authenticated counterfactual even when an
audit member displaces one of its evaluations.

This module deliberately does not know a workload, objective name, model, or
provider.  It is the first executable RBIE broker seam; cumulative regret-bank
allocation and learned delayed value can be injected without changing the
workload adapter or finite acquisition scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContext,
    AcquisitionCertifiedSlateContextProvider,
    feasible_slate_option_id_subsets,
    finite_acquisition_policy_identity,
)
from agent_evolve.policies.selection.calibrated_slate import (
    AllocatedSlateMember,
    SlateAllocationRequest,
    SlateAllocationRole,
    assess_allocated_slate_memory_dose,
)
from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScoreDecision,
    FiniteAcquisitionBatchScorePolicy,
    FiniteAcquisitionBatchScoreRequest,
    FiniteAcquisitionSlate,
    validate_finite_acquisition_batch_score_decision,
)
from agent_evolve.ports.portfolio_memory_dose import PortfolioMemoryDoseAssessment


POLICY_ID = "regret_bounded_information_slate"
POLICY_VERSION = 2
_POLICY_DOMAIN = b"agent-evolve:regret-bounded-information-slate-policy:v2\x00"
_ESTIMATE_DOMAIN = b"agent-evolve:slate-future-value-estimate:v1\x00"
_ASSESSMENT_DOMAIN = b"agent-evolve:regret-bounded-slate-assessment:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:regret-bounded-slate-decision:v1\x00"


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


def _finite_nonnegative(value: float, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


class SlateFutureValueAuthority(str, Enum):
    ZERO = "zero"
    DEVELOPMENT_ASSAY = "development_assay"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True, eq=False)
class SlateFutureValueEstimate:
    """Pre-outcome lower/upper bounds in log-acquisition-equivalent units."""

    slate_option_ids: tuple[str, ...]
    information_value_lcb: float
    descendant_value_lcb: float
    risk_cost_ucb: float
    authority: SlateFutureValueAuthority
    evidence_receipt_sha256: str
    policy_definition_sha256: str
    estimate_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if self.slate_option_ids != tuple(sorted(set(self.slate_option_ids))) or not (
            self.slate_option_ids
        ):
            raise ValueError("slate_option_ids must be non-empty and canonical")
        for name in (
            "information_value_lcb",
            "descendant_value_lcb",
            "risk_cost_ucb",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if type(self.authority) is not SlateFutureValueAuthority:
            raise TypeError("authority must be exact")
        if self.authority is SlateFutureValueAuthority.ZERO and (
            self.information_value_lcb != 0.0
            or self.descendant_value_lcb != 0.0
        ):
            raise ValueError("zero authority cannot claim positive future value")
        require_sha256(self.evidence_receipt_sha256, "evidence_receipt_sha256")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        computed = _hash(_ESTIMATE_DOMAIN, self._unsigned_record())
        if self.estimate_sha256 not in ("", computed):
            raise ValueError("estimate_sha256 does not authenticate the estimate")
        object.__setattr__(self, "estimate_sha256", computed)

    @property
    def net_future_value(self) -> float:
        return (
            self.information_value_lcb
            + self.descendant_value_lcb
            - self.risk_cost_ucb
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "slate_option_ids": list(self.slate_option_ids),
            "information_value_lcb_hex": self.information_value_lcb.hex(),
            "descendant_value_lcb_hex": self.descendant_value_lcb.hex(),
            "risk_cost_ucb_hex": self.risk_cost_ucb.hex(),
            "net_future_value_hex": self.net_future_value.hex(),
            "authority": self.authority.value,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "estimate_sha256": self.estimate_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is SlateFutureValueEstimate
            and self.estimate_sha256 == other.estimate_sha256
        )

    __hash__ = None


@runtime_checkable
class SlateFutureValuePolicy(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str

    def estimate(
        self,
        *,
        request: SlateAllocationRequest,
        reference_option_ids: tuple[str, ...],
        slate_option_ids: tuple[str, ...],
        reference_log_acquisition_value: float,
        slate_log_acquisition_value: float,
    ) -> SlateFutureValueEstimate: ...


@dataclass(frozen=True, slots=True)
class ZeroSlateFutureValuePolicy:
    """Safe default: RBIE collapses to acquisition-certified selection."""

    policy_id: str = field(init=False, default="zero_slate_future_value")
    policy_version: int = field(init=False, default=1)
    definition_sha256: str = field(
        init=False,
        default=hashlib.sha256(
            b"agent-evolve:zero-slate-future-value:v1;all-lcbs=0;risk=0"
        ).hexdigest(),
    )

    def estimate(
        self,
        *,
        request: SlateAllocationRequest,
        reference_option_ids: tuple[str, ...],
        slate_option_ids: tuple[str, ...],
        reference_log_acquisition_value: float,
        slate_log_acquisition_value: float,
    ) -> SlateFutureValueEstimate:
        del request, reference_option_ids
        if not math.isfinite(reference_log_acquisition_value) or not math.isfinite(
            slate_log_acquisition_value
        ):
            raise ValueError("acquisition values must be finite")
        return SlateFutureValueEstimate(
            slate_option_ids=slate_option_ids,
            information_value_lcb=0.0,
            descendant_value_lcb=0.0,
            risk_cost_ucb=0.0,
            authority=SlateFutureValueAuthority.ZERO,
            evidence_receipt_sha256=self.definition_sha256,
            policy_definition_sha256=self.definition_sha256,
        )


@dataclass(frozen=True, slots=True)
class ResidualInformationAssayValuePolicy:
    """Explicitly development-only treatment for the first regret assay.

    This policy does not pretend that residual information value is calibrated.
    It exists to run the matched 2/5/10% challenger experiment and is rejected
    by the broker unless ``allow_development_assay`` is explicitly enabled.
    """

    value_per_residual: float
    maximum_credited_residuals: int = 1
    policy_id: str = field(init=False, default="residual_information_assay")
    policy_version: int = field(init=False, default=1)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        _finite_nonnegative(self.value_per_residual, "value_per_residual")
        if self.value_per_residual == 0.0:
            raise ValueError("assay value must be positive")
        if (
            type(self.maximum_credited_residuals) is not int
            or self.maximum_credited_residuals < 1
        ):
            raise ValueError("maximum_credited_residuals must be positive")
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                b"agent-evolve:residual-information-assay:v1;"
                + self.value_per_residual.hex().encode("ascii")
                + b";max="
                + str(self.maximum_credited_residuals).encode("ascii")
            ).hexdigest(),
        )

    def estimate(
        self,
        *,
        request: SlateAllocationRequest,
        reference_option_ids: tuple[str, ...],
        slate_option_ids: tuple[str, ...],
        reference_log_acquisition_value: float,
        slate_log_acquisition_value: float,
    ) -> SlateFutureValueEstimate:
        del request
        if not math.isfinite(reference_log_acquisition_value) or not math.isfinite(
            slate_log_acquisition_value
        ):
            raise ValueError("acquisition values must be finite")
        residual_count = len(set(slate_option_ids).difference(reference_option_ids))
        credited = min(residual_count, self.maximum_credited_residuals)
        evidence = _hash(
            b"agent-evolve:residual-information-assay-evidence:v1\x00",
            {
                "policy_definition_sha256": self.definition_sha256,
                "reference_option_ids": list(reference_option_ids),
                "slate_option_ids": list(slate_option_ids),
                "credited_residuals": credited,
            },
        )
        return SlateFutureValueEstimate(
            slate_option_ids=slate_option_ids,
            information_value_lcb=float(credited * self.value_per_residual),
            descendant_value_lcb=0.0,
            risk_cost_ucb=0.0,
            authority=SlateFutureValueAuthority.DEVELOPMENT_ASSAY,
            evidence_receipt_sha256=evidence,
            policy_definition_sha256=self.definition_sha256,
        )


@dataclass(frozen=True, slots=True, eq=False)
class RegretBoundedSlateAssessment:
    slate_option_ids: tuple[str, ...]
    log_acquisition_value: float
    acquisition_log_gap_from_reference: float
    acquisition_retention_ratio: float
    inside_regret_envelope: bool
    future_value: SlateFutureValueEstimate | None
    broker_value: float | None
    assessment_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if self.slate_option_ids != tuple(sorted(set(self.slate_option_ids))) or not (
            self.slate_option_ids
        ):
            raise ValueError("assessment slate must be non-empty and canonical")
        for name in (
            "log_acquisition_value",
            "acquisition_log_gap_from_reference",
            "acquisition_retention_ratio",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.acquisition_retention_ratio <= 1.0:
            raise ValueError("acquisition retention ratio must lie in (0, 1]")
        if type(self.inside_regret_envelope) is not bool:
            raise TypeError("inside_regret_envelope must be exact")
        if self.inside_regret_envelope:
            if type(self.future_value) is not SlateFutureValueEstimate:
                raise TypeError("admissible slate requires an exact future estimate")
            self.future_value.__post_init__()
            if self.future_value.slate_option_ids != self.slate_option_ids:
                raise ValueError("future value belongs to a foreign slate")
            if type(self.broker_value) is not float or not math.isfinite(
                self.broker_value
            ):
                raise ValueError("admissible slate requires finite broker value")
        elif self.future_value is not None or self.broker_value is not None:
            raise ValueError("inadmissible slate cannot receive future authority")
        computed = _hash(_ASSESSMENT_DOMAIN, self._unsigned_record())
        if self.assessment_sha256 not in ("", computed):
            raise ValueError("assessment_sha256 does not authenticate the value")
        object.__setattr__(self, "assessment_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "slate_option_ids": list(self.slate_option_ids),
            "log_acquisition_value_hex": self.log_acquisition_value.hex(),
            "acquisition_log_gap_from_reference_hex": (
                self.acquisition_log_gap_from_reference.hex()
            ),
            "acquisition_retention_ratio_hex": (
                self.acquisition_retention_ratio.hex()
            ),
            "inside_regret_envelope": self.inside_regret_envelope,
            "future_value": (
                None if self.future_value is None else self.future_value.to_record()
            ),
            "broker_value_hex": (
                None if self.broker_value is None else self.broker_value.hex()
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "assessment_sha256": self.assessment_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is RegretBoundedSlateAssessment
            and self.assessment_sha256 == other.assessment_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class RegretBoundedSlateDecision:
    request: SlateAllocationRequest
    selected: tuple[AllocatedSlateMember, ...]
    reference_option_ids: tuple[str, ...]
    selected_option_ids: tuple[str, ...]
    reference_log_acquisition_value: float
    selected_log_acquisition_value: float
    acquisition_regret: float
    acquisition_retention_ratio: float
    minimum_acquisition_retention_ratio: float
    minimum_residual_audit_members: int
    tie_tolerance: float
    reference_member_count: int
    feasible_slate_count: int
    admissible_slate_count: int
    assessments: tuple[RegretBoundedSlateAssessment, ...]
    selected_future_value: SlateFutureValueEstimate
    selected_broker_value: float
    calibration_error_bound: float | None
    conditional_return_gap_lower_bound: float | None
    score_request: FiniteAcquisitionBatchScoreRequest
    score_decision: FiniteAcquisitionBatchScoreDecision
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None
    policy_definition_sha256: str
    decision_sha256: str = field(init=False, default="")

    policy_id = POLICY_ID
    policy_version = POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.request) is not SlateAllocationRequest:
            raise TypeError("request must be exact")
        self.request.revalidate()
        if type(self.selected) is not tuple or len(self.selected) != (
            self.request.portfolio_size
        ):
            raise ValueError("selected must fill the evaluation portfolio")
        if any(
            type(value) is not AllocatedSlateMember
            or value.role is not SlateAllocationRole.REGRET_BOUNDED_INFORMATION
            for value in self.selected
        ):
            raise ValueError("selected members must carry the RBIE role")
        for value in self.selected:
            value.__post_init__()
            if not math.isclose(
                value.role_score,
                self.selected_broker_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("selected role score differs from broker value")
        for name in ("reference_option_ids", "selected_option_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))) or len(values) != (
                self.request.portfolio_size
            ):
                raise ValueError(f"{name} must be a canonical complete slate")
        if tuple(sorted(value.option_id for value in self.selected)) != (
            self.selected_option_ids
        ):
            raise ValueError("selected identities differ from selected slate")
        for name in (
            "reference_log_acquisition_value",
            "selected_log_acquisition_value",
            "acquisition_regret",
            "acquisition_retention_ratio",
            "minimum_acquisition_retention_ratio",
            "tie_tolerance",
            "selected_broker_value",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            type(self.minimum_residual_audit_members) is not int
            or not 0
            <= self.minimum_residual_audit_members
            < self.request.portfolio_size
        ):
            raise ValueError(
                "minimum_residual_audit_members must lie in [0, portfolio_size)"
            )
        if self.acquisition_regret < 0.0:
            raise ValueError("acquisition regret must be non-negative")
        if self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be non-negative")
        expected_regret = max(
            0.0,
            self.reference_log_acquisition_value
            - self.selected_log_acquisition_value,
        )
        if not math.isclose(
            self.acquisition_regret, expected_regret, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("acquisition regret does not close")
        expected_ratio = math.exp(
            min(
                0.0,
                self.selected_log_acquisition_value
                - self.reference_log_acquisition_value,
            )
        )
        if not math.isclose(
            self.acquisition_retention_ratio,
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selected acquisition ratio does not close")
        if not (
            0.0
            < self.minimum_acquisition_retention_ratio
            <= self.acquisition_retention_ratio
            <= 1.0
        ):
            raise ValueError("selected slate escapes its regret envelope")
        if self.reference_member_count != len(
            set(self.selected_option_ids).intersection(self.reference_option_ids)
        ):
            raise ValueError("reference member count does not close")
        selected_residual_members = (
            self.request.portfolio_size - self.reference_member_count
        )
        if selected_residual_members < self.minimum_residual_audit_members:
            raise ValueError("selected slate violates the residual audit floor")
        if (
            type(self.feasible_slate_count) is not int
            or type(self.admissible_slate_count) is not int
            or not 1 <= self.admissible_slate_count <= self.feasible_slate_count
        ):
            raise ValueError("slate counts are invalid")
        if (
            type(self.assessments) is not tuple
            or len(self.assessments) != self.feasible_slate_count
            or any(
                type(value) is not RegretBoundedSlateAssessment
                for value in self.assessments
            )
        ):
            raise ValueError("assessments must cover every feasible slate")
        for value in self.assessments:
            value.__post_init__()
        if tuple(value.slate_option_ids for value in self.assessments) != tuple(
            sorted(value.slate_option_ids for value in self.assessments)
        ):
            raise ValueError("assessments must use canonical slate order")
        selected_assessment = next(
            (
                value
                for value in self.assessments
                if value.slate_option_ids == self.selected_option_ids
            ),
            None,
        )
        if (
            selected_assessment is None
            or selected_assessment.future_value != self.selected_future_value
            or selected_assessment.broker_value != self.selected_broker_value
        ):
            raise ValueError("selected assessment does not close")
        self.selected_future_value.__post_init__()
        if self.calibration_error_bound is None:
            if self.conditional_return_gap_lower_bound is not None:
                raise ValueError("conditional bound requires calibration error")
        else:
            _finite_nonnegative(
                self.calibration_error_bound, "calibration_error_bound"
            )
            expected_bound = -self.acquisition_regret - 2.0 * (
                self.calibration_error_bound
            )
            if self.conditional_return_gap_lower_bound is None or not math.isclose(
                self.conditional_return_gap_lower_bound,
                expected_bound,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("conditional return bound does not close")
        if type(self.score_request) is not FiniteAcquisitionBatchScoreRequest:
            raise TypeError("score_request must be exact")
        if type(self.score_decision) is not FiniteAcquisitionBatchScoreDecision:
            raise TypeError("score_decision must be exact")
        validate_finite_acquisition_batch_score_decision(
            self.score_request, self.score_decision
        )
        scored = {
            value.slate.candidate_ids: value.log_acquisition_value
            for value in self.score_decision.scores
        }
        assessment_by_ids = {
            value.slate_option_ids: value for value in self.assessments
        }
        if set(scored) != set(assessment_by_ids):
            raise ValueError("assessments differ from the frozen score request")
        for slate_ids, score in scored.items():
            assessment = assessment_by_ids[slate_ids]
            if not math.isclose(
                assessment.log_acquisition_value,
                score,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("assessment acquisition value does not replay")
            expected_gap = score - self.reference_log_acquisition_value
            if not math.isclose(
                assessment.acquisition_log_gap_from_reference,
                expected_gap,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("assessment acquisition gap does not close")
            expected_retention = math.exp(min(0.0, expected_gap))
            if not math.isclose(
                assessment.acquisition_retention_ratio,
                expected_retention,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("assessment acquisition retention does not close")
            expected_inside = expected_gap >= (
                math.log(self.minimum_acquisition_retention_ratio)
                - self.tie_tolerance
            )
            if assessment.inside_regret_envelope is not expected_inside:
                raise ValueError("assessment regret-envelope status does not replay")
            if assessment.future_value is not None:
                expected_broker = expected_gap + assessment.future_value.net_future_value
                if assessment.broker_value is None or not math.isclose(
                    assessment.broker_value,
                    expected_broker,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("assessment broker value does not close")
        if sum(
            value.inside_regret_envelope
            and (
                self.request.portfolio_size
                - len(
                    set(value.slate_option_ids).intersection(
                        self.reference_option_ids
                    )
                )
                >= self.minimum_residual_audit_members
            )
            for value in self.assessments
        ) != self.admissible_slate_count:
            raise ValueError("admissible slate count does not close")
        if not math.isclose(
            scored[self.reference_option_ids],
            self.reference_log_acquisition_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            scored[self.selected_option_ids],
            self.selected_log_acquisition_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("reference or selected acquisition value does not replay")
        if self.memory_dose_assessment is not None:
            self.memory_dose_assessment.__post_init__()
            if not self.memory_dose_assessment.passed:
                raise ValueError("selected slate violates memory dose")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        computed = _hash(_DECISION_DOMAIN, self._unsigned_record())
        if self.decision_sha256 not in ("", computed):
            raise ValueError("decision_sha256 does not authenticate the decision")
        object.__setattr__(self, "decision_sha256", computed)

    def revalidate(self) -> None:
        self.__post_init__()

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "event_type": "regret_bounded_information_slate_allocated",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "request": self.request.to_record(),
            "selected": [value.to_record() for value in self.selected],
            "reference_option_ids": list(self.reference_option_ids),
            "selected_option_ids": list(self.selected_option_ids),
            "reference_log_acquisition_value_hex": (
                self.reference_log_acquisition_value.hex()
            ),
            "selected_log_acquisition_value_hex": (
                self.selected_log_acquisition_value.hex()
            ),
            "acquisition_regret_hex": self.acquisition_regret.hex(),
            "acquisition_retention_ratio_hex": (
                self.acquisition_retention_ratio.hex()
            ),
            "minimum_acquisition_retention_ratio_hex": (
                self.minimum_acquisition_retention_ratio.hex()
            ),
            "minimum_residual_audit_members": (
                self.minimum_residual_audit_members
            ),
            "selected_residual_member_count": (
                self.request.portfolio_size - self.reference_member_count
            ),
            "tie_tolerance_hex": self.tie_tolerance.hex(),
            "reference_member_count": self.reference_member_count,
            "feasible_slate_count": self.feasible_slate_count,
            "admissible_slate_count": self.admissible_slate_count,
            "assessments": [value.to_record() for value in self.assessments],
            "selected_future_value": self.selected_future_value.to_record(),
            "selected_broker_value_hex": self.selected_broker_value.hex(),
            "calibration_error_bound_hex": (
                None
                if self.calibration_error_bound is None
                else self.calibration_error_bound.hex()
            ),
            "conditional_return_gap_lower_bound_hex": (
                None
                if self.conditional_return_gap_lower_bound is None
                else self.conditional_return_gap_lower_bound.hex()
            ),
            "score_request": self.score_request.to_record(),
            "score_decision": self.score_decision.to_record(),
            "memory_dose_assessment": (
                None
                if self.memory_dose_assessment is None
                else self.memory_dose_assessment.to_record()
            ),
            "certificate_scope": (
                "conditional_on_frozen_acquisition_calibration_not_sota"
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is RegretBoundedSlateDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None

    @property
    def prior_only(self) -> bool:
        return True

    @property
    def administered_card_keys(self) -> tuple[str, ...]:
        selected_ids = {value.option_id for value in self.selected}
        return tuple(
            sorted(
                {
                    card_key
                    for member in self.request.slate.members
                    if member.option_id in selected_ids
                    for card_key in member.supporting_card_keys
                    if card_key in self.request.assigned_card_keys
                }
            )
        )


@dataclass(frozen=True, slots=True)
class RegretBoundedSlatePolicy:
    context_provider: AcquisitionCertifiedSlateContextProvider
    scorer: FiniteAcquisitionBatchScorePolicy
    future_value_policy: SlateFutureValuePolicy = field(
        default_factory=ZeroSlateFutureValuePolicy
    )
    minimum_acquisition_retention_ratio: float = 1.0
    minimum_residual_audit_members: int = 0
    calibration_error_bound: float | None = None
    allow_development_assay: bool = False
    exact_combination_limit: int = 250_000
    tie_tolerance: float = 1e-12
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not isinstance(
            self.context_provider, AcquisitionCertifiedSlateContextProvider
        ):
            raise TypeError("context_provider must implement its exact port")
        scorer_identity = finite_acquisition_policy_identity(self.scorer)
        if not isinstance(self.future_value_policy, SlateFutureValuePolicy):
            raise TypeError("future_value_policy must implement its port")
        future_identity = (
            self.future_value_policy.policy_id,
            self.future_value_policy.policy_version,
            self.future_value_policy.definition_sha256,
        )
        if (
            type(future_identity[0]) is not str
            or not future_identity[0]
            or type(future_identity[1]) is not int
            or future_identity[1] <= 0
        ):
            raise ValueError("future value policy identity is invalid")
        require_sha256(future_identity[2], "future value definition_sha256")
        if (
            type(self.minimum_acquisition_retention_ratio) is not float
            or not math.isfinite(self.minimum_acquisition_retention_ratio)
            or not 0.0 < self.minimum_acquisition_retention_ratio <= 1.0
        ):
            raise ValueError("minimum acquisition retention must lie in (0, 1]")
        if (
            type(self.minimum_residual_audit_members) is not int
            or self.minimum_residual_audit_members < 0
        ):
            raise ValueError("minimum residual audit members must be non-negative")
        if self.calibration_error_bound is not None:
            _finite_nonnegative(
                self.calibration_error_bound, "calibration_error_bound"
            )
        if type(self.allow_development_assay) is not bool:
            raise TypeError("allow_development_assay must be exact")
        if type(self.exact_combination_limit) is not int or (
            self.exact_combination_limit < 1
        ):
            raise ValueError("exact_combination_limit must be positive")
        _finite_nonnegative(self.tie_tolerance, "tie_tolerance")
        record = {
            "schema_version": 1,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "context_provider": {
                "provider_id": self.context_provider.provider_id,
                "provider_version": self.context_provider.provider_version,
                "definition_sha256": self.context_provider.definition_sha256,
            },
            "scorer": {
                "policy_id": scorer_identity[0],
                "policy_version": scorer_identity[1],
                "definition_sha256": scorer_identity[2],
            },
            "future_value_policy": {
                "policy_id": future_identity[0],
                "policy_version": future_identity[1],
                "definition_sha256": future_identity[2],
            },
            "minimum_acquisition_retention_ratio_hex": (
                self.minimum_acquisition_retention_ratio.hex()
            ),
            "minimum_residual_audit_members": (
                self.minimum_residual_audit_members
            ),
            "calibration_error_bound_hex": (
                None
                if self.calibration_error_bound is None
                else self.calibration_error_bound.hex()
            ),
            "allow_development_assay": self.allow_development_assay,
            "exact_combination_limit": self.exact_combination_limit,
            "tie_tolerance_hex": self.tie_tolerance.hex(),
            "reference": "maximum-anchor-count-feasible-slate",
            "authority": "trusted-code-only",
            "outcome_access": "strictly-prior-only",
        }
        object.__setattr__(self, "definition_sha256", _hash(_POLICY_DOMAIN, record))

    def select(self, request: SlateAllocationRequest) -> RegretBoundedSlateDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact")
        request.revalidate()
        if self.minimum_residual_audit_members >= request.portfolio_size:
            raise ValueError(
                "minimum residual audit members must be below portfolio size"
            )
        context = self.context_provider.context_for(
            request.slate.finite_contract_sha256
        )
        if type(context) is not AcquisitionCertifiedSlateContext:
            raise TypeError("context provider returned a foreign context")
        context.__post_init__()
        if len(context.reference_option_ids) != request.portfolio_size:
            raise ValueError("reference must exactly fill the evaluation portfolio")
        feasible = feasible_slate_option_id_subsets(request)
        if not feasible:
            raise ValueError("proposal contains no hard-feasible evaluation slate")
        if len(feasible) > self.exact_combination_limit:
            raise ValueError("feasible slate count exceeds exact broker limit")
        slate_ids = {value.option_id for value in request.slate.members}
        candidate_by_id = {value.candidate_id: value for value in context.candidates}
        if not slate_ids <= set(candidate_by_id):
            raise ValueError("numerical context does not cover the proposal slate")
        reference_set = set(context.reference_option_ids)
        maximum_reference_count = max(
            len(set(value).intersection(reference_set)) for value in feasible
        )
        if maximum_reference_count != request.portfolio_size:
            raise ValueError("complete numerical reference is not hard-feasible")
        reference = min(
            value
            for value in feasible
            if len(set(value).intersection(reference_set)) == maximum_reference_count
        )
        score_request = FiniteAcquisitionBatchScoreRequest(
            campaign_scope_sha256=context.campaign_scope_sha256,
            cutoff_index=context.cutoff_index,
            seed=context.seed,
            objectives=context.objectives,
            observations=context.observations,
            candidates=tuple(candidate_by_id[value] for value in sorted(slate_ids)),
            slates=tuple(FiniteAcquisitionSlate(value) for value in feasible),
        )
        score_decision = self.scorer.score(score_request)
        validate_finite_acquisition_batch_score_decision(
            score_request, score_decision
        )
        expected_scorer = finite_acquisition_policy_identity(self.scorer)
        if (
            score_decision.policy_id,
            score_decision.policy_version,
            score_decision.policy_definition_sha256,
        ) != expected_scorer:
            raise ValueError("batch scorer returned a foreign policy identity")
        score_by_ids = {
            value.slate.candidate_ids: value.log_acquisition_value
            for value in score_decision.scores
        }
        reference_value = float(score_by_ids[reference])
        log_floor = math.log(self.minimum_acquisition_retention_ratio)
        assessments: list[RegretBoundedSlateAssessment] = []
        for slate_option_ids in feasible:
            score = float(score_by_ids[slate_option_ids])
            gap = score - reference_value
            ratio = math.exp(min(0.0, gap))
            inside = gap >= log_floor - self.tie_tolerance
            estimate = None
            broker_value = None
            if inside:
                estimate = self.future_value_policy.estimate(
                    request=request,
                    reference_option_ids=reference,
                    slate_option_ids=slate_option_ids,
                    reference_log_acquisition_value=reference_value,
                    slate_log_acquisition_value=score,
                )
                if type(estimate) is not SlateFutureValueEstimate:
                    raise TypeError("future value policy returned a foreign estimate")
                estimate.__post_init__()
                if estimate.policy_definition_sha256 != (
                    self.future_value_policy.definition_sha256
                ):
                    raise ValueError("future value estimate has foreign policy authority")
                if (
                    estimate.authority
                    is SlateFutureValueAuthority.DEVELOPMENT_ASSAY
                    and not self.allow_development_assay
                ):
                    raise ValueError(
                        "development-only future value lacks explicit authority"
                    )
                broker_value = gap + estimate.net_future_value
            assessments.append(
                RegretBoundedSlateAssessment(
                    slate_option_ids=slate_option_ids,
                    log_acquisition_value=score,
                    acquisition_log_gap_from_reference=gap,
                    acquisition_retention_ratio=ratio,
                    inside_regret_envelope=inside,
                    future_value=estimate,
                    broker_value=broker_value,
                )
            )
        admissible = tuple(
            value
            for value in assessments
            if value.inside_regret_envelope
            and (
                request.portfolio_size
                - len(set(value.slate_option_ids).intersection(reference_set))
                >= self.minimum_residual_audit_members
            )
        )
        if not admissible:
            raise ValueError(
                "no regret-admissible slate satisfies the residual audit floor"
            )
        best_value = max(value.broker_value for value in admissible)
        assert best_value is not None
        tied = tuple(
            value
            for value in admissible
            if value.broker_value is not None
            and value.broker_value >= best_value - self.tie_tolerance
        )
        reference_assessment = next(
            value for value in assessments if value.slate_option_ids == reference
        )
        selected_assessment = (
            reference_assessment
            if reference_assessment in tied
            else min(tied, key=lambda value: value.slate_option_ids)
        )
        selected_ids = selected_assessment.slate_option_ids
        member_by_id = {value.option_id: value for value in request.slate.members}
        selected_members = tuple(
            sorted(
                (member_by_id[value] for value in selected_ids),
                key=lambda value: value.model_rank,
            )
        )
        memory_dose_assessment = (
            None
            if request.memory_dose_contract is None
            else assess_allocated_slate_memory_dose(request, selected_members)
        )
        if memory_dose_assessment is not None and not memory_dose_assessment.passed:
            raise AssertionError("broker winner violated enumerated memory dose")
        selected_value = float(score_by_ids[selected_ids])
        acquisition_regret = max(0.0, reference_value - selected_value)
        selected_future_value = selected_assessment.future_value
        selected_broker_value = selected_assessment.broker_value
        assert selected_future_value is not None
        assert selected_broker_value is not None
        conditional_bound = (
            None
            if self.calibration_error_bound is None
            else -acquisition_regret - 2.0 * self.calibration_error_bound
        )
        return RegretBoundedSlateDecision(
            request=request,
            selected=tuple(
                AllocatedSlateMember(
                    role=SlateAllocationRole.REGRET_BOUNDED_INFORMATION,
                    option_id=value.option_id,
                    option_identity_sha256=value.option_identity_sha256,
                    model_rank=value.model_rank,
                    role_score=float(selected_broker_value),
                )
                for value in selected_members
            ),
            reference_option_ids=reference,
            selected_option_ids=selected_ids,
            reference_log_acquisition_value=reference_value,
            selected_log_acquisition_value=selected_value,
            acquisition_regret=float(acquisition_regret),
            acquisition_retention_ratio=float(
                math.exp(min(0.0, selected_value - reference_value))
            ),
            minimum_acquisition_retention_ratio=(
                self.minimum_acquisition_retention_ratio
            ),
            minimum_residual_audit_members=(
                self.minimum_residual_audit_members
            ),
            tie_tolerance=self.tie_tolerance,
            reference_member_count=len(
                set(selected_ids).intersection(reference_set)
            ),
            feasible_slate_count=len(feasible),
            admissible_slate_count=len(admissible),
            assessments=tuple(assessments),
            selected_future_value=selected_future_value,
            selected_broker_value=float(selected_broker_value),
            calibration_error_bound=self.calibration_error_bound,
            conditional_return_gap_lower_bound=conditional_bound,
            score_request=score_request,
            score_decision=score_decision,
            memory_dose_assessment=memory_dose_assessment,
            policy_definition_sha256=self.definition_sha256,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        scorer = finite_acquisition_policy_identity(self.scorer)
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "context_provider": {
                "provider_id": self.context_provider.provider_id,
                "provider_version": self.context_provider.provider_version,
                "definition_sha256": self.context_provider.definition_sha256,
            },
            "scorer": {
                "policy_id": scorer[0],
                "policy_version": scorer[1],
                "definition_sha256": scorer[2],
            },
            "future_value_policy": {
                "policy_id": self.future_value_policy.policy_id,
                "policy_version": self.future_value_policy.policy_version,
                "definition_sha256": self.future_value_policy.definition_sha256,
            },
            "minimum_acquisition_retention_ratio_hex": (
                self.minimum_acquisition_retention_ratio.hex()
            ),
            "minimum_residual_audit_members": (
                self.minimum_residual_audit_members
            ),
            "calibration_error_bound_hex": (
                None
                if self.calibration_error_bound is None
                else self.calibration_error_bound.hex()
            ),
            "allow_development_assay": self.allow_development_assay,
            "exact_combination_limit": self.exact_combination_limit,
            "tie_tolerance_hex": self.tie_tolerance.hex(),
        }


__all__ = [
    "POLICY_ID",
    "POLICY_VERSION",
    "RegretBoundedSlateAssessment",
    "RegretBoundedSlateDecision",
    "RegretBoundedSlatePolicy",
    "ResidualInformationAssayValuePolicy",
    "SlateFutureValueAuthority",
    "SlateFutureValueEstimate",
    "SlateFutureValuePolicy",
    "ZeroSlateFutureValuePolicy",
]

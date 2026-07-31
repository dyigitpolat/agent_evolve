"""Authenticated categorical forecast calibration with prior-only snapshots.

Numeric metric changes cross only the benchmark-owned adjudicator seam.  The
persistent policy state is a compact set of signed categorical observations,
filtered at an exclusive wave cutoff before any allocator can consume it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.agentic_generator import MetricEffectDirection


_SCOPE_DOMAIN = b"agent-evolve:forecast-calibration-scope:v1\x00"
_PREDICTION_DOMAIN = b"agent-evolve:forecast-prediction-receipt:v1\x00"
_ADJUDICATION_REQUEST_DOMAIN = (
    b"agent-evolve:meaningful-direction-adjudication-request:v1\x00"
)
_ADJUDICATION_DOMAIN = b"agent-evolve:meaningful-direction-adjudication:v1\x00"
_OBSERVATION_DOMAIN = b"agent-evolve:forecast-calibration-observation:v1\x00"
_SNAPSHOT_DOMAIN = b"agent-evolve:forecast-calibration-snapshot:v1\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_METRIC = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_OPTION = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_MAX_WAVE = (1 << 63) - 1


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
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _require_metric(value: str, *, name: str = "metric_id") -> None:
    if type(value) is not str or _METRIC.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed metric identifier grammar")


def _require_option(value: str, *, name: str = "option_id") -> None:
    if type(value) is not str or _OPTION.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed option identifier grammar")


def _require_wave(value: int, *, name: str) -> None:
    if type(value) is not int or not 1 <= value <= _MAX_WAVE:
        raise ValueError(f"{name} must be an exact positive int63")


def _require_finite_float(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


class ForecastConfidenceBin(str, Enum):
    """Closed confidence vocabulary emitted with a direction forecast."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ForecastCalibrationScope:
    """Identity of one local model/prompt/policy/benchmark/session stratum."""

    model_profile_sha256: str
    prompt_definition_sha256: str
    selector_policy_definition_sha256: str
    benchmark_sha256: str
    session_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "model_profile_sha256",
            "prompt_definition_sha256",
            "selector_policy_definition_sha256",
            "benchmark_sha256",
            "session_sha256",
        ):
            require_sha256(getattr(self, name), name)

    def revalidate(self) -> None:
        if type(self) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        ForecastCalibrationScope.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "model_profile_sha256": self.model_profile_sha256,
            "prompt_definition_sha256": self.prompt_definition_sha256,
            "selector_policy_definition_sha256": (
                self.selector_policy_definition_sha256
            ),
            "benchmark_sha256": self.benchmark_sha256,
            "session_sha256": self.session_sha256,
        }

    @property
    def scope_sha256(self) -> str:
        return _hash(_SCOPE_DOMAIN, self._unsigned_record())

    def for_policy_frame(
        self,
        *,
        prompt_definition_sha256: str,
        selector_policy_definition_sha256: str,
    ) -> "ForecastCalibrationScope":
        """Preserve the experiment stratum while changing policy provenance.

        One campaign can legitimately contain more than one predictor: for
        example, an engine-owned calibrated slate policy and a runtime
        outcome-conditioned consequence expert.  Their observations must not
        share prompt/policy identity, while model, benchmark, and session
        identity must remain exact.  This immutable operation makes that
        separation explicit at the generic calibration boundary.
        """

        self.revalidate()
        return ForecastCalibrationScope(
            model_profile_sha256=self.model_profile_sha256,
            prompt_definition_sha256=prompt_definition_sha256,
            selector_policy_definition_sha256=(
                selector_policy_definition_sha256
            ),
            benchmark_sha256=self.benchmark_sha256,
            session_sha256=self.session_sha256,
        )

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "scope_sha256": self.scope_sha256}


@dataclass(frozen=True, slots=True)
class BetaCorrectnessPrior:
    """Declared shrinkage prior for sparse categorical correctness cells."""

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        _require_finite_float(self.alpha, name="alpha")
        _require_finite_float(self.beta, name="beta")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("Beta prior parameters must be strictly positive")

    @property
    def mean(self) -> float:
        self.__post_init__()
        return self.alpha / (self.alpha + self.beta)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "family": "beta_bernoulli_correctness",
            "alpha_hex": self.alpha.hex(),
            "beta_hex": self.beta.hex(),
            "mean_hex": self.mean.hex(),
        }


@dataclass(frozen=True, slots=True)
class ForecastPredictionReceipt:
    """Authenticated categorical prediction emitted before evaluation."""

    scope: ForecastCalibrationScope
    wave_index: int
    selector_decision_sha256: str
    parent_candidate_identity_sha256: str
    option_id: str
    option_identity_sha256: str
    family: str
    metric_id: str
    asserted_direction: MetricEffectDirection
    confidence: ForecastConfidenceBin

    def __post_init__(self) -> None:
        if type(self.scope) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        self.scope.revalidate()
        _require_wave(self.wave_index, name="wave_index")
        for name in (
            "selector_decision_sha256",
            "parent_candidate_identity_sha256",
            "option_identity_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_option(self.option_id)
        _require_token(self.family, name="family")
        _require_metric(self.metric_id)
        if type(self.asserted_direction) is not MetricEffectDirection:
            raise TypeError("asserted_direction must be exact MetricEffectDirection")
        if type(self.confidence) is not ForecastConfidenceBin:
            raise TypeError("confidence must be exact ForecastConfidenceBin")

    def revalidate(self) -> None:
        if type(self) is not ForecastPredictionReceipt:
            raise TypeError("prediction must be exact ForecastPredictionReceipt")
        ForecastPredictionReceipt.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "scope_sha256": self.scope.scope_sha256,
            "wave_index": self.wave_index,
            "selector_decision_sha256": self.selector_decision_sha256,
            "parent_candidate_identity_sha256": (self.parent_candidate_identity_sha256),
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "metric_id": self.metric_id,
            "asserted_direction": self.asserted_direction.value,
            "confidence": self.confidence.value,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_PREDICTION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class MeaningfulDirectionRequest:
    """Numeric request interpreted only by a benchmark-owned adjudicator."""

    benchmark_sha256: str
    session_sha256: str
    wave_index: int
    parent_candidate_identity_sha256: str
    option_id: str
    option_identity_sha256: str
    metric_id: str
    parent_outcome_sha256: str
    child_outcome_sha256: str
    parent_metric_value: float
    child_metric_value: float

    def __post_init__(self) -> None:
        for name in (
            "benchmark_sha256",
            "session_sha256",
            "parent_candidate_identity_sha256",
            "option_identity_sha256",
            "parent_outcome_sha256",
            "child_outcome_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_wave(self.wave_index, name="wave_index")
        _require_option(self.option_id)
        _require_metric(self.metric_id)
        _require_finite_float(self.parent_metric_value, name="parent_metric_value")
        _require_finite_float(self.child_metric_value, name="child_metric_value")

    def revalidate(self) -> None:
        if type(self) is not MeaningfulDirectionRequest:
            raise TypeError("request must be exact MeaningfulDirectionRequest")
        MeaningfulDirectionRequest.__post_init__(self)

    def _record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "benchmark_sha256": self.benchmark_sha256,
            "session_sha256": self.session_sha256,
            "wave_index": self.wave_index,
            "parent_candidate_identity_sha256": (self.parent_candidate_identity_sha256),
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "metric_id": self.metric_id,
            "parent_outcome_sha256": self.parent_outcome_sha256,
            "child_outcome_sha256": self.child_outcome_sha256,
            "parent_metric_value_hex": self.parent_metric_value.hex(),
            "child_metric_value_hex": self.child_metric_value.hex(),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_ADJUDICATION_REQUEST_DOMAIN, self._record())


@dataclass(frozen=True, slots=True)
class MeaningfulDirectionAdjudicationReceipt:
    """Categorical benchmark judgment bound to exact parent/child outcomes."""

    request_sha256: str
    benchmark_sha256: str
    session_sha256: str
    wave_index: int
    parent_candidate_identity_sha256: str
    option_id: str
    option_identity_sha256: str
    metric_id: str
    parent_outcome_sha256: str
    child_outcome_sha256: str
    actual_direction: MetricEffectDirection
    adjudicator_policy_id: str
    adjudicator_policy_version: int
    adjudicator_definition_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "benchmark_sha256",
            "session_sha256",
            "parent_candidate_identity_sha256",
            "option_identity_sha256",
            "parent_outcome_sha256",
            "child_outcome_sha256",
            "adjudicator_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_wave(self.wave_index, name="wave_index")
        _require_option(self.option_id)
        _require_metric(self.metric_id)
        if (
            type(self.actual_direction) is not MetricEffectDirection
            or self.actual_direction is MetricEffectDirection.UNKNOWN
        ):
            raise ValueError("actual_direction must be a known metric direction")
        _require_token(self.adjudicator_policy_id, name="adjudicator_policy_id")
        if (
            type(self.adjudicator_policy_version) is not int
            or self.adjudicator_policy_version <= 0
        ):
            raise ValueError("adjudicator_policy_version must be positive")

    def revalidate(self) -> None:
        if type(self) is not MeaningfulDirectionAdjudicationReceipt:
            raise TypeError(
                "adjudication must be exact MeaningfulDirectionAdjudicationReceipt"
            )
        MeaningfulDirectionAdjudicationReceipt.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "session_sha256": self.session_sha256,
            "wave_index": self.wave_index,
            "parent_candidate_identity_sha256": (self.parent_candidate_identity_sha256),
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "metric_id": self.metric_id,
            "parent_outcome_sha256": self.parent_outcome_sha256,
            "child_outcome_sha256": self.child_outcome_sha256,
            "actual_direction": self.actual_direction.value,
            "adjudicator": {
                "policy_id": self.adjudicator_policy_id,
                "policy_version": self.adjudicator_policy_version,
                "definition_sha256": self.adjudicator_definition_sha256,
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_ADJUDICATION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def require_request(self, request: MeaningfulDirectionRequest) -> None:
        """Authenticate this categorical result against its numeric request."""

        self.revalidate()
        if type(request) is not MeaningfulDirectionRequest:
            raise TypeError("request must be exact MeaningfulDirectionRequest")
        request.revalidate()
        observed = (
            self.request_sha256,
            self.benchmark_sha256,
            self.session_sha256,
            self.wave_index,
            self.parent_candidate_identity_sha256,
            self.option_id,
            self.option_identity_sha256,
            self.metric_id,
            self.parent_outcome_sha256,
            self.child_outcome_sha256,
        )
        expected = (
            request.request_sha256,
            request.benchmark_sha256,
            request.session_sha256,
            request.wave_index,
            request.parent_candidate_identity_sha256,
            request.option_id,
            request.option_identity_sha256,
            request.metric_id,
            request.parent_outcome_sha256,
            request.child_outcome_sha256,
        )
        if observed != expected:
            raise ValueError("adjudication receipt belongs to a foreign request")


@runtime_checkable
class MeaningfulMetricDirectionAdjudicator(Protocol):
    """Inverted benchmark seam for meaningful parent/child direction."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def adjudicate(
        self, request: MeaningfulDirectionRequest
    ) -> MeaningfulDirectionAdjudicationReceipt: ...


@dataclass(frozen=True, slots=True)
class ForecastCalibrationObservation:
    """One pre-evaluation prediction joined to one authenticated outcome fact."""

    prediction: ForecastPredictionReceipt
    adjudication: MeaningfulDirectionAdjudicationReceipt

    def __post_init__(self) -> None:
        if type(self.prediction) is not ForecastPredictionReceipt:
            raise TypeError("prediction must be exact ForecastPredictionReceipt")
        if type(self.adjudication) is not MeaningfulDirectionAdjudicationReceipt:
            raise TypeError(
                "adjudication must be exact MeaningfulDirectionAdjudicationReceipt"
            )
        self.prediction.revalidate()
        self.adjudication.revalidate()
        scope = self.prediction.scope
        observed = (
            scope.benchmark_sha256,
            scope.session_sha256,
            self.prediction.wave_index,
            self.prediction.parent_candidate_identity_sha256,
            self.prediction.option_id,
            self.prediction.option_identity_sha256,
            self.prediction.metric_id,
        )
        expected = (
            self.adjudication.benchmark_sha256,
            self.adjudication.session_sha256,
            self.adjudication.wave_index,
            self.adjudication.parent_candidate_identity_sha256,
            self.adjudication.option_id,
            self.adjudication.option_identity_sha256,
            self.adjudication.metric_id,
        )
        if observed != expected:
            raise ValueError("prediction and adjudication evidence do not join")

    def revalidate(self) -> None:
        if type(self) is not ForecastCalibrationObservation:
            raise TypeError("observation must be exact ForecastCalibrationObservation")
        ForecastCalibrationObservation.__post_init__(self)

    @property
    def is_abstention(self) -> bool:
        return self.prediction.asserted_direction is MetricEffectDirection.UNKNOWN

    @property
    def correctness(self) -> bool | None:
        if self.is_abstention:
            return None
        return self.prediction.asserted_direction is self.adjudication.actual_direction

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "prediction_receipt": self.prediction.to_record(),
            "adjudication_receipt": self.adjudication.to_record(),
            "is_abstention": self.is_abstention,
            "correctness": self.correctness,
        }

    @property
    def observation_sha256(self) -> str:
        return _hash(_OBSERVATION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "observation_sha256": self.observation_sha256,
        }


def observe_forecast(
    prediction: ForecastPredictionReceipt,
    request: MeaningfulDirectionRequest,
    adjudicator: MeaningfulMetricDirectionAdjudicator,
) -> ForecastCalibrationObservation:
    """Run the injected adjudicator and close all prediction/outcome joins."""

    if type(prediction) is not ForecastPredictionReceipt:
        raise TypeError("prediction must be exact ForecastPredictionReceipt")
    prediction.revalidate()
    if type(request) is not MeaningfulDirectionRequest:
        raise TypeError("request must be exact MeaningfulDirectionRequest")
    request.revalidate()
    if not isinstance(adjudicator, MeaningfulMetricDirectionAdjudicator):
        raise TypeError("adjudicator must implement the direction adjudicator port")
    _require_token(adjudicator.policy_id, name="adjudicator.policy_id")
    if type(adjudicator.policy_version) is not int or adjudicator.policy_version <= 0:
        raise ValueError("adjudicator.policy_version must be positive")
    require_sha256(adjudicator.definition_sha256, "adjudicator.definition_sha256")
    receipt = adjudicator.adjudicate(request)
    if type(receipt) is not MeaningfulDirectionAdjudicationReceipt:
        raise TypeError("adjudicator returned a foreign receipt type")
    receipt.require_request(request)
    if (
        receipt.adjudicator_policy_id != adjudicator.policy_id
        or receipt.adjudicator_policy_version != adjudicator.policy_version
        or receipt.adjudicator_definition_sha256 != adjudicator.definition_sha256
    ):
        raise ValueError("adjudicator receipt uses a foreign policy identity")
    return ForecastCalibrationObservation(prediction, receipt)


@dataclass(frozen=True, slots=True)
class ForecastCalibrationCell:
    """One empirical/Beta-smoothed categorical correctness cell."""

    metric_id: str
    asserted_direction: MetricEffectDirection
    confidence: ForecastConfidenceBin
    family: str | None
    observation_count: int
    scorable_count: int
    correct_count: int
    prior: BetaCorrectnessPrior

    def __post_init__(self) -> None:
        _require_metric(self.metric_id)
        if type(self.asserted_direction) is not MetricEffectDirection:
            raise TypeError("asserted_direction must be exact MetricEffectDirection")
        if type(self.confidence) is not ForecastConfidenceBin:
            raise TypeError("confidence must be exact ForecastConfidenceBin")
        if self.family is not None:
            _require_token(self.family, name="family")
        for name in ("observation_count", "scorable_count", "correct_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if not self.correct_count <= self.scorable_count <= self.observation_count:
            raise ValueError("calibration cell counts are inconsistent")
        if type(self.prior) is not BetaCorrectnessPrior:
            raise TypeError("prior must be exact BetaCorrectnessPrior")
        self.prior.__post_init__()
        if (
            self.asserted_direction is MetricEffectDirection.UNKNOWN
            and self.scorable_count != 0
        ):
            raise ValueError("unknown-direction observations must be abstentions")

    @property
    def empirical_accuracy(self) -> float | None:
        self.__post_init__()
        if self.scorable_count == 0:
            return None
        return self.correct_count / self.scorable_count

    @property
    def posterior_correctness(self) -> float:
        self.__post_init__()
        return (self.prior.alpha + self.correct_count) / (
            self.prior.alpha + self.prior.beta + self.scorable_count
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        empirical = self.empirical_accuracy
        return {
            "metric_id": self.metric_id,
            "asserted_direction": self.asserted_direction.value,
            "confidence": self.confidence.value,
            "family": self.family,
            "observation_count": self.observation_count,
            "scorable_count": self.scorable_count,
            "correct_count": self.correct_count,
            "empirical_accuracy_hex": (None if empirical is None else empirical.hex()),
            "posterior_correctness_hex": self.posterior_correctness.hex(),
            "prior": self.prior.to_record(),
        }


def _observation_key(
    observation: ForecastCalibrationObservation,
) -> tuple[int, str, str, str]:
    prediction = observation.prediction
    return (
        prediction.wave_index,
        prediction.selector_decision_sha256,
        prediction.option_id,
        prediction.metric_id,
    )


def _cell_key(
    cell: ForecastCalibrationCell,
) -> tuple[str, str, str, str]:
    return (
        cell.metric_id,
        cell.asserted_direction.value,
        cell.confidence.value,
        "" if cell.family is None else cell.family,
    )


def _build_cell(
    observations: tuple[ForecastCalibrationObservation, ...],
    *,
    metric_id: str,
    direction: MetricEffectDirection,
    confidence: ForecastConfidenceBin,
    family: str | None,
    prior: BetaCorrectnessPrior,
) -> ForecastCalibrationCell:
    members = tuple(
        observation
        for observation in observations
        if observation.prediction.metric_id == metric_id
        and observation.prediction.asserted_direction is direction
        and observation.prediction.confidence is confidence
        and (family is None or observation.prediction.family == family)
    )
    scorable = tuple(value for value in members if value.correctness is not None)
    return ForecastCalibrationCell(
        metric_id=metric_id,
        asserted_direction=direction,
        confidence=confidence,
        family=family,
        observation_count=len(members),
        scorable_count=len(scorable),
        correct_count=sum(value.correctness is True for value in scorable),
        prior=prior,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ForecastCalibrationSnapshot:
    """Immutable calibration evidence available strictly before a wave."""

    scope: ForecastCalibrationScope
    cutoff_wave_index_exclusive: int
    observations: tuple[ForecastCalibrationObservation, ...]
    prior: BetaCorrectnessPrior = field(default_factory=BetaCorrectnessPrior)
    family_min_support: int = 4

    def __post_init__(self) -> None:
        if type(self.scope) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        self.scope.revalidate()
        _require_wave(
            self.cutoff_wave_index_exclusive,
            name="cutoff_wave_index_exclusive",
        )
        if type(self.observations) is not tuple or any(
            type(value) is not ForecastCalibrationObservation
            for value in self.observations
        ):
            raise TypeError("observations must contain exact calibration observations")
        for value in self.observations:
            value.revalidate()
        if self.observations != tuple(sorted(self.observations, key=_observation_key)):
            raise ValueError("observations must use canonical evidence order")
        semantic_keys = tuple(_observation_key(value) for value in self.observations)
        if len(set(semantic_keys)) != len(semantic_keys):
            raise ValueError("snapshot contains duplicate forecast cells")
        if any(value.prediction.scope != self.scope for value in self.observations):
            raise ValueError("snapshot contains a foreign calibration scope")
        if any(
            value.prediction.wave_index >= self.cutoff_wave_index_exclusive
            for value in self.observations
        ):
            raise ValueError("snapshot contains current/future-wave outcome evidence")
        if type(self.prior) is not BetaCorrectnessPrior:
            raise TypeError("prior must be exact BetaCorrectnessPrior")
        self.prior.__post_init__()
        if type(self.family_min_support) is not int or self.family_min_support <= 0:
            raise ValueError("family_min_support must be a positive exact integer")

    def revalidate(self) -> None:
        if type(self) is not ForecastCalibrationSnapshot:
            raise TypeError("snapshot must be exact ForecastCalibrationSnapshot")
        ForecastCalibrationSnapshot.__post_init__(self)

    @property
    def cells(self) -> tuple[ForecastCalibrationCell, ...]:
        self.revalidate()
        global_keys = sorted(
            {
                (
                    value.prediction.metric_id,
                    value.prediction.asserted_direction,
                    value.prediction.confidence,
                )
                for value in self.observations
            },
            key=lambda value: (value[0], value[1].value, value[2].value),
        )
        result = [
            _build_cell(
                self.observations,
                metric_id=metric_id,
                direction=direction,
                confidence=confidence,
                family=None,
                prior=self.prior,
            )
            for metric_id, direction, confidence in global_keys
        ]
        family_keys = sorted(
            {
                (
                    value.prediction.metric_id,
                    value.prediction.asserted_direction,
                    value.prediction.confidence,
                    value.prediction.family,
                )
                for value in self.observations
            },
            key=lambda value: (value[0], value[1].value, value[2].value, value[3]),
        )
        for metric_id, direction, confidence, family in family_keys:
            cell = _build_cell(
                self.observations,
                metric_id=metric_id,
                direction=direction,
                confidence=confidence,
                family=family,
                prior=self.prior,
            )
            if cell.scorable_count >= self.family_min_support:
                result.append(cell)
        return tuple(sorted(result, key=_cell_key))

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def abstention_count(self) -> int:
        return sum(value.is_abstention for value in self.observations)

    @property
    def scorable_count(self) -> int:
        return self.observation_count - self.abstention_count

    @property
    def correct_count(self) -> int:
        return sum(value.correctness is True for value in self.observations)

    @property
    def empirical_accuracy(self) -> float | None:
        if self.scorable_count == 0:
            return None
        return self.correct_count / self.scorable_count

    def lookup(
        self,
        *,
        metric_id: str,
        asserted_direction: MetricEffectDirection,
        confidence: ForecastConfidenceBin,
        family: str,
    ) -> tuple[ForecastCalibrationCell, str]:
        """Use supported family evidence, then metric-global evidence, then prior."""

        self.revalidate()
        _require_metric(metric_id)
        if type(asserted_direction) is not MetricEffectDirection:
            raise TypeError("asserted_direction must be exact MetricEffectDirection")
        if type(confidence) is not ForecastConfidenceBin:
            raise TypeError("confidence must be exact ForecastConfidenceBin")
        _require_token(family, name="family")
        for cell in self.cells:
            if (
                cell.metric_id == metric_id
                and cell.asserted_direction is asserted_direction
                and cell.confidence is confidence
                and cell.family == family
            ):
                return cell, "supported_family"
        for cell in self.cells:
            if (
                cell.metric_id == metric_id
                and cell.asserted_direction is asserted_direction
                and cell.confidence is confidence
                and cell.family is None
            ):
                return cell, "metric_direction_confidence"
        return (
            ForecastCalibrationCell(
                metric_id=metric_id,
                asserted_direction=asserted_direction,
                confidence=confidence,
                family=None,
                observation_count=0,
                scorable_count=0,
                correct_count=0,
                prior=self.prior,
            ),
            "declared_prior",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        accuracy = self.empirical_accuracy
        return {
            "schema_version": 1,
            "scope": self.scope.to_record(),
            "cutoff_wave_index_exclusive": self.cutoff_wave_index_exclusive,
            "prior": self.prior.to_record(),
            "family_min_support": self.family_min_support,
            "observations": [value.to_record() for value in self.observations],
            "cells": [value.to_record() for value in self.cells],
            "summary": {
                "observation_count": self.observation_count,
                "abstention_count": self.abstention_count,
                "scorable_count": self.scorable_count,
                "correct_count": self.correct_count,
                "empirical_accuracy_hex": (
                    None if accuracy is None else accuracy.hex()
                ),
            },
            "leakage_guard": "only_observation_wave_lt_exclusive_cutoff",
        }

    @property
    def snapshot_sha256(self) -> str:
        return _hash(_SNAPSHOT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "snapshot_sha256": self.snapshot_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ForecastCalibrationSnapshot
            and self.snapshot_sha256 == other.snapshot_sha256
        )

    __hash__ = None


def build_calibration_snapshot(
    observations: Sequence[ForecastCalibrationObservation],
    *,
    scope: ForecastCalibrationScope,
    cutoff_wave_index_exclusive: int,
    prior: BetaCorrectnessPrior = BetaCorrectnessPrior(),
    family_min_support: int = 4,
) -> ForecastCalibrationSnapshot:
    """Filter an immutable multi-scope ledger at an exclusive wave cutoff."""

    if isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a finite observation sequence")
    if type(scope) is not ForecastCalibrationScope:
        raise TypeError("scope must be exact ForecastCalibrationScope")
    scope.revalidate()
    _require_wave(
        cutoff_wave_index_exclusive,
        name="cutoff_wave_index_exclusive",
    )
    admitted: list[ForecastCalibrationObservation] = []
    seen: set[str] = set()
    for value in observations:
        if type(value) is not ForecastCalibrationObservation:
            raise TypeError("ledger contains a foreign observation type")
        value.revalidate()
        if value.observation_sha256 in seen:
            raise ValueError("ledger contains a duplicate observation receipt")
        seen.add(value.observation_sha256)
        if (
            value.prediction.scope == scope
            and value.prediction.wave_index < cutoff_wave_index_exclusive
        ):
            admitted.append(value)
    return ForecastCalibrationSnapshot(
        scope=scope,
        cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
        observations=tuple(sorted(admitted, key=_observation_key)),
        prior=prior,
        family_min_support=family_min_support,
    )


__all__ = [
    "BetaCorrectnessPrior",
    "ForecastCalibrationObservation",
    "ForecastCalibrationScope",
    "ForecastCalibrationSnapshot",
    "ForecastConfidenceBin",
    "ForecastPredictionReceipt",
    "MeaningfulDirectionAdjudicationReceipt",
    "MeaningfulDirectionRequest",
    "MeaningfulMetricDirectionAdjudicator",
    "build_calibration_snapshot",
    "observe_forecast",
]

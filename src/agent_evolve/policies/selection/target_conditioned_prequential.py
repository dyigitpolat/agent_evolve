"""Generic target-conditioned, evaluator-updated portfolio acquisition.

The policy is deliberately split from workload adapters and provider code.  A
caller supplies an authenticated affine frontier target, a portable numeric
feature row for every member of a sealed slate, a complete realizable-set
receipt, and an immutable linear-Gaussian state.  The policy scores and exactly
selects one feasible portfolio.  A separate generation-barrier update consumes
outcomes for selected members only.

No workload, model, provider, option-name, or evaluator implementation is part
of the scoring interface.  Optional workload knowledge belongs in a separately
authenticated pre-outcome feature projector, not in this policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import json
import math
import re
from statistics import fmean
from typing import ClassVar, Sequence

from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlateMember,
    SlateAllocationRequest,
    assess_allocated_slate_memory_dose,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget
from agent_evolve.ports.portfolio_memory_dose import PortfolioMemoryDoseAssessment


POLICY_ID = "target_conditioned_prequential_realizable_portfolio"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:target-conditioned-prequential-realizable-portfolio:v1;"
    b"input=sealed-slate,prior-only-affine-target,portable-features,"
    b"immutable-linear-gaussian-state,complete-realizable-set;"
    b"heads=normalized-marginal-archive-utility,target-achievement;"
    b"scoring=within-slate-z-marginal-plus-bounded-direction-plus-"
    b"finite-horizon-epistemic;selection=exact-set-sum-canonical-ties;"
    b"updates=selected-outcomes-only-at-generation-barrier;"
    b"workload-model-provider-option-name-fields=false"
).hexdigest()

BASE_REALIZABILITY_PROJECTOR_ID = "sealed_slate_base_realizability"
BASE_REALIZABILITY_PROJECTOR_VERSION = 1
BASE_REALIZABILITY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:sealed-slate-base-realizability:v1;"
    b"constraints=cardinality,required-members,pairwise-compatibility,"
    b"minimum-distinct-families,bounded-memory-dose;"
    b"enumeration=complete;ties=canonical-option-id-set"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_FEATURE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_HEAD_DOMAIN = b"agent-evolve:prequential-linear-gaussian-head:v1\x00"
_META_PRIOR_DOMAIN = b"agent-evolve:trap-portable-meta-prior:v1\x00"
_STATE_DOMAIN = b"agent-evolve:target-conditioned-acquisition-state:v1\x00"
_PROFILE_DOMAIN = b"agent-evolve:target-conditioned-acquisition-profile:v1\x00"
_REALIZABLE_DOMAIN = b"agent-evolve:realizable-portfolio-set:v1\x00"
_FEATURE_ROW_DOMAIN = b"agent-evolve:target-conditioned-feature-row:v1\x00"
_REQUEST_DOMAIN = b"agent-evolve:target-conditioned-slate-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:target-conditioned-slate-decision:v1\x00"
_OBSERVATION_DOMAIN = b"agent-evolve:target-conditioned-observation:v1\x00"
_UPDATE_DOMAIN = b"agent-evolve:target-conditioned-state-update:v1\x00"


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


def _finite(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be an exact finite float")


def _hex_float(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be a binary64 hexadecimal string")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} is not valid binary64 hexadecimal text") from error
    _finite(result, name=name)
    return result


def _exact_record_keys(
    value: object, *, expected: set[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    keys = set(value)
    if keys != expected:
        raise ValueError(
            f"{name} fields differ from the closed schema: "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    return value


def _feature_names(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or not values:
        raise ValueError("feature_names must be a non-empty exact tuple")
    if any(type(value) is not str or _FEATURE.fullmatch(value) is None for value in values):
        raise ValueError("feature names must use the closed lowercase grammar")
    if values != tuple(dict.fromkeys(values)):
        raise ValueError("feature names must be unique and ordered")
    if "bias" not in values:
        raise ValueError("feature schema must include bias")


def _cholesky(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    size = len(matrix)
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            value = matrix[row][column] - sum(
                lower[row][index] * lower[column][index]
                for index in range(column)
            )
            if row == column:
                if value <= 1e-14 or not math.isfinite(value):
                    raise ValueError("head precision must be positive definite")
                lower[row][column] = math.sqrt(value)
            else:
                lower[row][column] = value / lower[column][column]
    return tuple(tuple(value for value in row) for row in lower)


def _solve_cholesky(
    lower: tuple[tuple[float, ...], ...], rhs: Sequence[float]
) -> tuple[float, ...]:
    size = len(lower)
    forward = [0.0 for _ in range(size)]
    for row in range(size):
        forward[row] = (
            float(rhs[row])
            - sum(lower[row][index] * forward[index] for index in range(row))
        ) / lower[row][row]
    result = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        result[row] = (
            forward[row]
            - sum(
                lower[index][row] * result[index]
                for index in range(row + 1, size)
            )
        ) / lower[row][row]
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _HeadProjection:
    head: PrequentialLinearGaussianHead
    lower: tuple[tuple[float, ...], ...]
    coefficients: tuple[float, ...]

    def standardized(self, values: Sequence[float]) -> tuple[float, ...]:
        if len(values) != len(self.head.feature_names):
            raise ValueError("feature vector differs from the head schema")
        return tuple(
            (float(value) - mean) / scale
            for value, mean, scale in zip(
                values, self.head.means, self.head.scales, strict=True
            )
        )

    def predict(self, values: Sequence[float]) -> float:
        row = self.standardized(values)
        return sum(
            value * coefficient
            for value, coefficient in zip(row, self.coefficients, strict=True)
        )

    def uncertainty(self, values: Sequence[float]) -> float:
        row = self.standardized(values)
        # x' P^-1 x == ||L^-1 x||^2 for P = L L'.
        whitened: list[float] = []
        for index in range(len(row)):
            whitened.append(
                (
                    row[index]
                    - sum(
                        self.lower[index][prior] * whitened[prior]
                        for prior in range(index)
                    )
                )
                / self.lower[index][index]
            )
        leverage = max(0.0, sum(value * value for value in whitened))
        return math.sqrt(self.head.residual_variance * leverage)


@dataclass(frozen=True, slots=True, eq=False)
class PrequentialLinearGaussianHead:
    """Serializable regularized head represented by sufficient statistics."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    rhs: tuple[float, ...]
    residual_variance: float

    def __post_init__(self) -> None:
        _feature_names(self.feature_names)
        size = len(self.feature_names)
        for name, values in (
            ("means", self.means),
            ("scales", self.scales),
            ("rhs", self.rhs),
        ):
            if type(values) is not tuple or len(values) != size:
                raise ValueError(f"{name} must match the feature schema")
            for value in values:
                _finite(value, name=name)
        if any(value <= 0.0 for value in self.scales):
            raise ValueError("feature scales must be strictly positive")
        if type(self.precision) is not tuple or len(self.precision) != size:
            raise ValueError("precision must be a square exact tuple matrix")
        for row in self.precision:
            if type(row) is not tuple or len(row) != size:
                raise ValueError("precision must be a square exact tuple matrix")
            for value in row:
                _finite(value, name="precision")
        for row in range(size):
            for column in range(row):
                if not math.isclose(
                    self.precision[row][column],
                    self.precision[column][row],
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError("precision must be symmetric")
        _finite(self.residual_variance, name="residual_variance")
        if self.residual_variance <= 0.0:
            raise ValueError("residual_variance must be strictly positive")
        _cholesky(self.precision)

    def project(self) -> _HeadProjection:
        self.__post_init__()
        lower = _cholesky(self.precision)
        return _HeadProjection(
            head=self,
            lower=lower,
            coefficients=_solve_cholesky(lower, self.rhs),
        )

    def update(
        self,
        rows: Sequence[Sequence[float]],
        targets: Sequence[float],
    ) -> PrequentialLinearGaussianHead:
        self.__post_init__()
        if not rows or len(rows) != len(targets):
            raise ValueError("head update requires aligned non-empty observations")
        projection = self.project()
        design = [projection.standardized(row) for row in rows]
        for target in targets:
            _finite(float(target), name="target")
        size = len(self.feature_names)
        precision = [list(row) for row in self.precision]
        rhs = list(self.rhs)
        for row, target in zip(design, targets, strict=True):
            for left in range(size):
                rhs[left] += row[left] * float(target)
                for right in range(size):
                    precision[left][right] += row[left] * row[right]
        return PrequentialLinearGaussianHead(
            feature_names=self.feature_names,
            means=self.means,
            scales=self.scales,
            precision=tuple(tuple(value for value in row) for row in precision),
            rhs=tuple(rhs),
            residual_variance=self.residual_variance,
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "feature_names": list(self.feature_names),
            "means_hex": [value.hex() for value in self.means],
            "scales_hex": [value.hex() for value in self.scales],
            "precision_hex": [
                [value.hex() for value in row] for row in self.precision
            ],
            "rhs_hex": [value.hex() for value in self.rhs],
            "residual_variance_hex": self.residual_variance.hex(),
        }

    @property
    def head_sha256(self) -> str:
        return _hash(_HEAD_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "head_sha256": self.head_sha256}

    @classmethod
    def from_record(
        cls, record: object
    ) -> PrequentialLinearGaussianHead:
        value = _exact_record_keys(
            record,
            expected={
                "schema_version",
                "feature_names",
                "means_hex",
                "scales_hex",
                "precision_hex",
                "rhs_hex",
                "residual_variance_hex",
                "head_sha256",
            },
            name="linear-Gaussian head record",
        )
        if value["schema_version"] != 1:
            raise ValueError("unsupported linear-Gaussian head schema")
        raw_names = value["feature_names"]
        raw_means = value["means_hex"]
        raw_scales = value["scales_hex"]
        raw_precision = value["precision_hex"]
        raw_rhs = value["rhs_hex"]
        if type(raw_names) is not list or any(type(item) is not str for item in raw_names):
            raise TypeError("feature_names must be an exact JSON string array")
        for name, raw in (
            ("means_hex", raw_means),
            ("scales_hex", raw_scales),
            ("rhs_hex", raw_rhs),
        ):
            if type(raw) is not list:
                raise TypeError(f"{name} must be an exact JSON array")
        if type(raw_precision) is not list or any(
            type(row) is not list for row in raw_precision
        ):
            raise TypeError("precision_hex must be an exact JSON matrix")
        result = cls(
            feature_names=tuple(raw_names),
            means=tuple(
                _hex_float(item, name=f"means_hex[{index}]")
                for index, item in enumerate(raw_means)
            ),
            scales=tuple(
                _hex_float(item, name=f"scales_hex[{index}]")
                for index, item in enumerate(raw_scales)
            ),
            precision=tuple(
                tuple(
                    _hex_float(
                        item,
                        name=f"precision_hex[{row_index}][{column_index}]",
                    )
                    for column_index, item in enumerate(row)
                )
                for row_index, row in enumerate(raw_precision)
            ),
            rhs=tuple(
                _hex_float(item, name=f"rhs_hex[{index}]")
                for index, item in enumerate(raw_rhs)
            ),
            residual_variance=_hex_float(
                value["residual_variance_hex"],
                name="residual_variance_hex",
            ),
        )
        if value["head_sha256"] != result.head_sha256:
            raise ValueError("linear-Gaussian head identity mismatch")
        return result

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is PrequentialLinearGaussianHead
            and self.head_sha256 == other.head_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedMetaPrior:
    """Frozen cross-campaign prior from an authenticated training panel."""

    training_data_sha256: str
    marginal_head: PrequentialLinearGaussianHead
    direction_head: PrequentialLinearGaussianHead

    def __post_init__(self) -> None:
        require_sha256(self.training_data_sha256, "training_data_sha256")
        if type(self.marginal_head) is not PrequentialLinearGaussianHead:
            raise TypeError("marginal_head must be exact")
        if type(self.direction_head) is not PrequentialLinearGaussianHead:
            raise TypeError("direction_head must be exact")
        self.marginal_head.__post_init__()
        self.direction_head.__post_init__()
        if self.marginal_head.feature_names != self.direction_head.feature_names:
            raise ValueError("meta-prior heads must share one feature schema")
        if (
            self.marginal_head.means != self.direction_head.means
            or self.marginal_head.scales != self.direction_head.scales
        ):
            raise ValueError("meta-prior heads must share one standardizer")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "training_data_sha256": self.training_data_sha256,
            "marginal_head": self.marginal_head.to_record(),
            "direction_head": self.direction_head.to_record(),
        }

    @property
    def meta_prior_sha256(self) -> str:
        return _hash(_META_PRIOR_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "meta_prior_sha256": self.meta_prior_sha256,
        }

    @classmethod
    def from_record(cls, record: object) -> TargetConditionedMetaPrior:
        value = _exact_record_keys(
            record,
            expected={
                "schema_version",
                "training_data_sha256",
                "marginal_head",
                "direction_head",
                "meta_prior_sha256",
            },
            name="target-conditioned meta-prior record",
        )
        if value["schema_version"] != 1:
            raise ValueError("unsupported target-conditioned meta-prior schema")
        result = cls(
            training_data_sha256=value["training_data_sha256"],
            marginal_head=PrequentialLinearGaussianHead.from_record(
                value["marginal_head"]
            ),
            direction_head=PrequentialLinearGaussianHead.from_record(
                value["direction_head"]
            ),
        )
        if value["meta_prior_sha256"] != result.meta_prior_sha256:
            raise ValueError("target-conditioned meta-prior identity mismatch")
        return result

    def initial_state(
        self, *, campaign_scope_sha256: str
    ) -> TargetConditionedAcquisitionState:
        self.__post_init__()
        return TargetConditionedAcquisitionState(
            campaign_scope_sha256=campaign_scope_sha256,
            training_data_sha256=self.training_data_sha256,
            marginal_head=self.marginal_head,
            direction_head=self.direction_head,
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedMetaPrior
            and self.meta_prior_sha256 == other.meta_prior_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedAcquisitionState:
    """Branch-local immutable posterior state at one generation cutoff."""

    campaign_scope_sha256: str
    training_data_sha256: str
    marginal_head: PrequentialLinearGaussianHead
    direction_head: PrequentialLinearGaussianHead
    cutoff_generation: int = 0
    selected_observation_count: int = 0

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(self.training_data_sha256, "training_data_sha256")
        if type(self.marginal_head) is not PrequentialLinearGaussianHead:
            raise TypeError("marginal_head must be exact")
        if type(self.direction_head) is not PrequentialLinearGaussianHead:
            raise TypeError("direction_head must be exact")
        self.marginal_head.__post_init__()
        self.direction_head.__post_init__()
        if self.marginal_head.feature_names != self.direction_head.feature_names:
            raise ValueError("acquisition heads must use one feature schema")
        if (
            self.marginal_head.means != self.direction_head.means
            or self.marginal_head.scales != self.direction_head.scales
        ):
            raise ValueError("acquisition heads must share one standardizer")
        if type(self.cutoff_generation) is not int or self.cutoff_generation < 0:
            raise ValueError("cutoff_generation must be non-negative")
        if (
            type(self.selected_observation_count) is not int
            or self.selected_observation_count < 0
        ):
            raise ValueError("selected_observation_count must be non-negative")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "training_data_sha256": self.training_data_sha256,
            "marginal_head": self.marginal_head.to_record(),
            "direction_head": self.direction_head.to_record(),
            "cutoff_generation": self.cutoff_generation,
            "selected_observation_count": self.selected_observation_count,
        }

    @property
    def state_sha256(self) -> str:
        return _hash(_STATE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "state_sha256": self.state_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedAcquisitionState
            and self.state_sha256 == other.state_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedAcquisitionProfile:
    """Frozen portable coefficients; no route-specific test-time tuning."""

    direction_weight: float
    uncertainty_weight: float
    maximum_remaining_horizon: int
    profile_id: str = "trap_portable_v1"
    profile_version: int = 1

    def __post_init__(self) -> None:
        if type(self.profile_id) is not str or _TOKEN.fullmatch(self.profile_id) is None:
            raise ValueError("profile_id must use the closed token grammar")
        if type(self.profile_version) is not int or self.profile_version <= 0:
            raise ValueError("profile_version must be positive")
        for name in ("direction_weight", "uncertainty_weight"):
            value = getattr(self, name)
            _finite(value, name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if (
            type(self.maximum_remaining_horizon) is not int
            or self.maximum_remaining_horizon <= 0
        ):
            raise ValueError("maximum_remaining_horizon must be positive")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "direction_weight_hex": self.direction_weight.hex(),
            "uncertainty_weight_hex": self.uncertainty_weight.hex(),
            "maximum_remaining_horizon": self.maximum_remaining_horizon,
        }

    @property
    def profile_sha256(self) -> str:
        return _hash(_PROFILE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "profile_sha256": self.profile_sha256}

    @classmethod
    def from_record(
        cls, record: object
    ) -> TargetConditionedAcquisitionProfile:
        value = _exact_record_keys(
            record,
            expected={
                "schema_version",
                "profile_id",
                "profile_version",
                "direction_weight_hex",
                "uncertainty_weight_hex",
                "maximum_remaining_horizon",
                "profile_sha256",
            },
            name="target-conditioned acquisition profile record",
        )
        if value["schema_version"] != 1:
            raise ValueError("unsupported target-conditioned profile schema")
        result = cls(
            profile_id=value["profile_id"],
            profile_version=value["profile_version"],
            direction_weight=_hex_float(
                value["direction_weight_hex"], name="direction_weight_hex"
            ),
            uncertainty_weight=_hex_float(
                value["uncertainty_weight_hex"], name="uncertainty_weight_hex"
            ),
            maximum_remaining_horizon=value["maximum_remaining_horizon"],
        )
        if value["profile_sha256"] != result.profile_sha256:
            raise ValueError("target-conditioned profile identity mismatch")
        return result

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedAcquisitionProfile
            and self.profile_sha256 == other.profile_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedMemberFeatures:
    """Authenticated numeric projection for one exact sealed option."""

    option_id: str
    option_identity_sha256: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    projector_id: str
    projector_version: int
    projector_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        _feature_names(self.feature_names)
        if type(self.values) is not tuple or len(self.values) != len(
            self.feature_names
        ):
            raise ValueError("feature values must match the feature schema")
        for value in self.values:
            _finite(value, name="feature value")
        if type(self.projector_id) is not str or _TOKEN.fullmatch(self.projector_id) is None:
            raise ValueError("projector_id must use the closed token grammar")
        if type(self.projector_version) is not int or self.projector_version <= 0:
            raise ValueError("projector_version must be positive")
        require_sha256(
            self.projector_definition_sha256, "projector_definition_sha256"
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "feature_names": list(self.feature_names),
            "values_hex": [value.hex() for value in self.values],
            "projector": {
                "projector_id": self.projector_id,
                "projector_version": self.projector_version,
                "definition_sha256": self.projector_definition_sha256,
            },
        }

    @property
    def feature_row_sha256(self) -> str:
        return _hash(_FEATURE_ROW_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "feature_row_sha256": self.feature_row_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedMemberFeatures
            and self.feature_row_sha256 == other.feature_row_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class RealizablePortfolioSet:
    """Complete finite portfolio universe certified by a generic projector."""

    source_request_sha256: str
    projector_id: str
    projector_version: int
    projector_definition_sha256: str
    option_id_sets: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        require_sha256(self.source_request_sha256, "source_request_sha256")
        if type(self.projector_id) is not str or _TOKEN.fullmatch(self.projector_id) is None:
            raise ValueError("projector_id must use the closed token grammar")
        if type(self.projector_version) is not int or self.projector_version <= 0:
            raise ValueError("projector_version must be positive")
        require_sha256(
            self.projector_definition_sha256, "projector_definition_sha256"
        )
        if type(self.option_id_sets) is not tuple or not self.option_id_sets:
            raise ValueError("option_id_sets must be a non-empty exact tuple")
        for values in self.option_id_sets:
            if type(values) is not tuple or not values:
                raise ValueError("each realizable portfolio must be non-empty")
            if values != tuple(sorted(set(values))):
                raise ValueError("realizable option IDs must be unique and canonical")
        if self.option_id_sets != tuple(sorted(set(self.option_id_sets))):
            raise ValueError("realizable portfolios must be unique and canonical")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "source_request_sha256": self.source_request_sha256,
            "projector": {
                "projector_id": self.projector_id,
                "projector_version": self.projector_version,
                "definition_sha256": self.projector_definition_sha256,
            },
            "option_id_sets": [list(value) for value in self.option_id_sets],
            "complete_for_projector_contract": True,
        }

    @property
    def realizable_set_sha256(self) -> str:
        return _hash(_REALIZABLE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "realizable_set_sha256": self.realizable_set_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is RealizablePortfolioSet
            and self.realizable_set_sha256 == other.realizable_set_sha256
        )

    __hash__ = None


def _members_for_ids(
    request: SlateAllocationRequest, option_ids: Sequence[str]
) -> tuple[CalibratedSlateMember, ...]:
    selected = set(option_ids)
    return tuple(
        member for member in request.slate.members if member.option_id in selected
    )


def _base_feasible(
    request: SlateAllocationRequest, option_ids: tuple[str, ...]
) -> tuple[bool, PortfolioMemoryDoseAssessment | None]:
    if len(option_ids) != request.portfolio_size:
        return False, None
    if not set(request.required_option_ids).issubset(option_ids):
        return False, None
    members = _members_for_ids(request, option_ids)
    if len(members) != request.portfolio_size:
        return False, None
    if request.pairwise_disjoint_option_id_pairs is not None:
        allowed = {frozenset(value) for value in request.pairwise_disjoint_option_id_pairs}
        if any(
            frozenset(value) not in allowed for value in combinations(option_ids, 2)
        ):
            return False, None
    if request.min_distinct_families is not None and len(
        {value.family for value in members}
    ) < request.min_distinct_families:
        return False, None
    assessment = (
        None
        if request.memory_dose_contract is None
        else assess_allocated_slate_memory_dose(request, members)
    )
    if assessment is not None and not assessment.passed:
        return False, assessment
    return True, assessment


def enumerate_base_realizable_portfolios(
    request: SlateAllocationRequest,
) -> RealizablePortfolioSet:
    """Exactly enumerate the constraints represented by ``SlateAllocationRequest``."""

    if type(request) is not SlateAllocationRequest:
        raise TypeError("request must be exact SlateAllocationRequest")
    request.revalidate()
    values = []
    for members in combinations(request.slate.members, request.portfolio_size):
        option_ids = tuple(sorted(value.option_id for value in members))
        feasible, _ = _base_feasible(request, option_ids)
        if feasible:
            values.append(option_ids)
    if not values:
        raise ValueError("sealed slate has no base-realizable portfolio")
    return RealizablePortfolioSet(
        source_request_sha256=request.request_sha256,
        projector_id=BASE_REALIZABILITY_PROJECTOR_ID,
        projector_version=BASE_REALIZABILITY_PROJECTOR_VERSION,
        projector_definition_sha256=BASE_REALIZABILITY_DEFINITION_SHA256,
        option_id_sets=tuple(sorted(set(values))),
    )


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedSlateRequest:
    """All authenticated inputs for one prior-only T-RAP decision."""

    allocation_request: SlateAllocationRequest
    frontier_target: CampaignPortfolioFrontierTarget
    state: TargetConditionedAcquisitionState
    member_features: tuple[TargetConditionedMemberFeatures, ...]
    realizable_portfolios: RealizablePortfolioSet
    campaign_generation: int
    remaining_proposal_horizon: int

    def __post_init__(self) -> None:
        if type(self.allocation_request) is not SlateAllocationRequest:
            raise TypeError("allocation_request must be exact")
        self.allocation_request.revalidate()
        if type(self.frontier_target) is not CampaignPortfolioFrontierTarget:
            raise TypeError("frontier_target must be exact")
        self.frontier_target.__post_init__()
        if type(self.state) is not TargetConditionedAcquisitionState:
            raise TypeError("state must be exact")
        self.state.__post_init__()
        if (
            type(self.member_features) is not tuple
            or any(
                type(value) is not TargetConditionedMemberFeatures
                for value in self.member_features
            )
        ):
            raise TypeError("member_features must contain exact rows")
        for value in self.member_features:
            value.__post_init__()
        if type(self.realizable_portfolios) is not RealizablePortfolioSet:
            raise TypeError("realizable_portfolios must be exact")
        self.realizable_portfolios.__post_init__()
        if (
            self.realizable_portfolios.source_request_sha256
            != self.allocation_request.request_sha256
        ):
            raise ValueError("realizable set names a foreign allocation request")
        base_projector_identity = (
            BASE_REALIZABILITY_PROJECTOR_ID,
            BASE_REALIZABILITY_PROJECTOR_VERSION,
            BASE_REALIZABILITY_DEFINITION_SHA256,
        )
        supplied_projector_identity = (
            self.realizable_portfolios.projector_id,
            self.realizable_portfolios.projector_version,
            self.realizable_portfolios.projector_definition_sha256,
        )
        if self.realizable_portfolios.projector_id == BASE_REALIZABILITY_PROJECTOR_ID:
            if supplied_projector_identity != base_projector_identity:
                raise ValueError("reserved base projector ID has a foreign identity")
            expected = enumerate_base_realizable_portfolios(self.allocation_request)
            if self.realizable_portfolios != expected:
                raise ValueError("base realizability receipt is not complete")
        if type(self.campaign_generation) is not int or self.campaign_generation <= 0:
            raise ValueError("campaign_generation must be positive")
        if (
            type(self.remaining_proposal_horizon) is not int
            or self.remaining_proposal_horizon < 0
        ):
            raise ValueError("remaining_proposal_horizon must be non-negative")
        if self.state.cutoff_generation >= self.campaign_generation:
            raise ValueError("posterior cutoff reaches current/future generation")
        slate = self.allocation_request.slate
        if (
            self.frontier_target.parent_configuration_sha256
            != slate.parent_candidate_identity_sha256
        ):
            raise ValueError("frontier target names a foreign parent")
        feature_by_id = {value.option_id: value for value in self.member_features}
        member_by_id = {value.option_id: value for value in slate.members}
        if len(feature_by_id) != len(self.member_features) or set(feature_by_id) != set(
            member_by_id
        ):
            raise ValueError("feature rows must cover every slate member exactly")
        expected_order = tuple(sorted(feature_by_id))
        if tuple(value.option_id for value in self.member_features) != expected_order:
            raise ValueError("feature rows must use canonical option order")
        for option_id, row in feature_by_id.items():
            if row.option_identity_sha256 != member_by_id[option_id].option_identity_sha256:
                raise ValueError("feature row names a foreign option identity")
            if row.feature_names != self.state.marginal_head.feature_names:
                raise ValueError("feature row differs from the posterior schema")
        projectors = {
            (
                value.projector_id,
                value.projector_version,
                value.projector_definition_sha256,
            )
            for value in self.member_features
        }
        if len(projectors) != 1:
            raise ValueError("feature rows must share one projector identity")
        slate_ids = set(member_by_id)
        for option_ids in self.realizable_portfolios.option_id_sets:
            if not set(option_ids).issubset(slate_ids):
                raise ValueError("realizable portfolio escapes the sealed slate")
            feasible, _ = _base_feasible(self.allocation_request, option_ids)
            if not feasible:
                raise ValueError("realizable projector admitted a base-infeasible set")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "allocation_request_sha256": self.allocation_request.request_sha256,
            "frontier_target": self.frontier_target.to_record(),
            "state": self.state.to_record(),
            "member_features": [value.to_record() for value in self.member_features],
            "realizable_portfolios": self.realizable_portfolios.to_record(),
            "campaign_generation": self.campaign_generation,
            "remaining_proposal_horizon": self.remaining_proposal_horizon,
            "current_or_future_outcomes_consulted": False,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedSlateRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


def _z_scores(values: dict[str, float]) -> dict[str, float]:
    mean = fmean(values.values())
    scale = math.sqrt(fmean((value - mean) ** 2 for value in values.values()))
    if scale < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - mean) / scale for key, value in values.items()}


@dataclass(frozen=True, slots=True)
class TargetConditionedMemberScore:
    option_id: str
    option_identity_sha256: str
    model_rank: int
    predicted_marginal: float
    predicted_direction: float
    epistemic_uncertainty: float
    marginal_z: float
    direction_z: float
    uncertainty_z: float
    final_score: float

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be positive")
        for name in (
            "predicted_marginal",
            "predicted_direction",
            "epistemic_uncertainty",
            "marginal_z",
            "direction_z",
            "uncertainty_z",
            "final_score",
        ):
            _finite(getattr(self, name), name=name)
        if self.epistemic_uncertainty < 0.0:
            raise ValueError("epistemic_uncertainty must be non-negative")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "predicted_marginal_hex": self.predicted_marginal.hex(),
            "predicted_direction_hex": self.predicted_direction.hex(),
            "epistemic_uncertainty_hex": self.epistemic_uncertainty.hex(),
            "marginal_z_hex": self.marginal_z.hex(),
            "direction_z_hex": self.direction_z.hex(),
            "uncertainty_z_hex": self.uncertainty_z.hex(),
            "final_score_hex": self.final_score.hex(),
        }


class TargetConditionedSlateRole(str, Enum):
    """Engine-owned role for each evaluator exposure selected by T-RAP."""

    ACQUISITION = "target_conditioned_acquisition"


@dataclass(frozen=True, slots=True)
class TargetConditionedAllocatedMember:
    option_id: str
    option_identity_sha256: str
    model_rank: int
    acquisition_score: float
    role: TargetConditionedSlateRole = TargetConditionedSlateRole.ACQUISITION

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be positive")
        _finite(self.acquisition_score, name="acquisition_score")
        if type(self.role) is not TargetConditionedSlateRole:
            raise TypeError("role must be exact TargetConditionedSlateRole")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "acquisition_score_hex": self.acquisition_score.hex(),
            "role": self.role.value,
        }


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedSlateDecision:
    request: TargetConditionedSlateRequest
    profile: TargetConditionedAcquisitionProfile
    score_rows: tuple[TargetConditionedMemberScore, ...]
    selected: tuple[TargetConditionedAllocatedMember, ...]
    joint_score: float
    feasible_subset_count: int
    administered_card_keys: tuple[str, ...]
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.request) is not TargetConditionedSlateRequest:
            raise TypeError("request must be exact")
        self.request.__post_init__()
        if type(self.profile) is not TargetConditionedAcquisitionProfile:
            raise TypeError("profile must be exact")
        self.profile.__post_init__()
        if type(self.score_rows) is not tuple or any(
            type(value) is not TargetConditionedMemberScore for value in self.score_rows
        ):
            raise TypeError("score_rows must contain exact values")
        if type(self.selected) is not tuple or len(self.selected) != (
            self.request.allocation_request.portfolio_size
        ):
            raise ValueError("selected must have the exact portfolio size")
        for value in (*self.score_rows, *self.selected):
            value.__post_init__()
        score_ids = tuple(value.option_id for value in self.score_rows)
        expected_score_ids = tuple(
            value.option_id for value in self.request.member_features
        )
        if score_ids != expected_score_ids:
            raise ValueError("score rows differ from the canonical feature rows")
        selected_ids = tuple(sorted(value.option_id for value in self.selected))
        if selected_ids not in self.request.realizable_portfolios.option_id_sets:
            raise ValueError("selected set is not realizable")
        _finite(self.joint_score, name="joint_score")
        if (
            type(self.feasible_subset_count) is not int
            or self.feasible_subset_count
            != len(self.request.realizable_portfolios.option_id_sets)
        ):
            raise ValueError("feasible_subset_count differs from the sealed universe")
        if self.administered_card_keys != tuple(sorted(set(self.administered_card_keys))):
            raise ValueError("administered_card_keys must be unique and canonical")
        if self.memory_dose_assessment is not None:
            if type(self.memory_dose_assessment) is not PortfolioMemoryDoseAssessment:
                raise TypeError("memory_dose_assessment must be exact or None")
            self.memory_dose_assessment.__post_init__()
            if not self.memory_dose_assessment.passed:
                raise ValueError("selected memory dose must pass")

    def revalidate(self) -> None:
        if type(self) is not TargetConditionedSlateDecision:
            raise TypeError("decision must be exact")
        TargetConditionedSlateDecision.__post_init__(self)

    @property
    def prior_only(self) -> bool:
        return self.request.state.cutoff_generation < self.request.campaign_generation

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "event_type": "target_conditioned_prequential_portfolio_allocated",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "request": self.request.to_record(),
            "profile": self.profile.to_record(),
            "score_rows": [value.to_record() for value in self.score_rows],
            "selected": [value.to_record() for value in self.selected],
            "joint_score_hex": self.joint_score.hex(),
            "feasible_subset_count": self.feasible_subset_count,
            "administered_card_keys": list(self.administered_card_keys),
            "memory_dose_assessment": (
                None
                if self.memory_dose_assessment is None
                else self.memory_dose_assessment.to_record()
            ),
            "prior_only": self.prior_only,
            "claim_scope": "allocation_receipt_not_efficacy_or_outcome_claim",
        }

    @property
    def decision_sha256(self) -> str:
        return _hash(_DECISION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedSlateDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class TargetConditionedPrequentialSlatePolicy:
    """Select the exact highest-scoring realizable set under a frozen profile."""

    profile: TargetConditionedAcquisitionProfile

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.profile) is not TargetConditionedAcquisitionProfile:
            raise TypeError("profile must be exact")
        self.profile.__post_init__()

    def select(self, request: TargetConditionedSlateRequest) -> TargetConditionedSlateDecision:
        self.__post_init__()
        if type(request) is not TargetConditionedSlateRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if request.remaining_proposal_horizon > self.profile.maximum_remaining_horizon:
            raise ValueError("remaining horizon exceeds the frozen profile")
        marginal_head = request.state.marginal_head.project()
        direction_head = request.state.direction_head.project()
        marginal = {
            row.option_id: marginal_head.predict(row.values)
            for row in request.member_features
        }
        direction = {
            row.option_id: direction_head.predict(row.values)
            for row in request.member_features
        }
        uncertainty = {
            row.option_id: marginal_head.uncertainty(row.values)
            for row in request.member_features
        }
        marginal_z = _z_scores(marginal)
        direction_z = _z_scores(direction)
        uncertainty_z = _z_scores(uncertainty)
        horizon_fraction = (
            request.remaining_proposal_horizon
            / self.profile.maximum_remaining_horizon
        )
        scores = {
            option_id: (
                marginal_z[option_id]
                + self.profile.direction_weight * direction_z[option_id]
                + self.profile.uncertainty_weight
                * horizon_fraction
                * uncertainty_z[option_id]
            )
            for option_id in marginal
        }
        selected_set = min(
            request.realizable_portfolios.option_id_sets,
            key=lambda option_ids: (
                -sum(scores[option_id] for option_id in option_ids),
                option_ids,
            ),
        )
        selected_members = _members_for_ids(
            request.allocation_request, selected_set
        )
        feasible, dose = _base_feasible(request.allocation_request, selected_set)
        if not feasible:  # Defensive after request validation.
            raise RuntimeError("winning realizable set became base-infeasible")
        feature_by_id = {value.option_id: value for value in request.member_features}
        member_by_id = {
            value.option_id: value for value in request.allocation_request.slate.members
        }
        score_rows = tuple(
            TargetConditionedMemberScore(
                option_id=option_id,
                option_identity_sha256=feature_by_id[
                    option_id
                ].option_identity_sha256,
                model_rank=member_by_id[option_id].model_rank,
                predicted_marginal=marginal[option_id],
                predicted_direction=direction[option_id],
                epistemic_uncertainty=uncertainty[option_id],
                marginal_z=marginal_z[option_id],
                direction_z=direction_z[option_id],
                uncertainty_z=uncertainty_z[option_id],
                final_score=scores[option_id],
            )
            for option_id in sorted(member_by_id)
        )
        selected = tuple(
            TargetConditionedAllocatedMember(
                option_id=member.option_id,
                option_identity_sha256=member.option_identity_sha256,
                model_rank=member.model_rank,
                acquisition_score=scores[member.option_id],
            )
            for member in selected_members
        )
        administered = tuple(
            sorted(
                {
                    card
                    for member in selected_members
                    for card in member.supporting_card_keys
                    if card in request.allocation_request.assigned_card_keys
                }
            )
        )
        return TargetConditionedSlateDecision(
            request=request,
            profile=self.profile,
            score_rows=score_rows,
            selected=selected,
            joint_score=sum(scores[value] for value in selected_set),
            feasible_subset_count=len(
                request.realizable_portfolios.option_id_sets
            ),
            administered_card_keys=administered,
            memory_dose_assessment=dose,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "profile": self.profile.to_record(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedSelectedObservation:
    """Two evaluator-derived labels for one actually selected member."""

    decision_sha256: str
    campaign_generation: int
    option_id: str
    option_identity_sha256: str
    feature_row_sha256: str
    feature_values: tuple[float, ...]
    normalized_marginal_utility: float
    normalized_target_improvement: float
    evaluator_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "decision_sha256",
            "option_identity_sha256",
            "feature_row_sha256",
            "evaluator_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.campaign_generation) is not int or self.campaign_generation <= 0:
            raise ValueError("campaign_generation must be positive")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        if type(self.feature_values) is not tuple or not self.feature_values:
            raise ValueError("feature_values must be non-empty")
        for value in self.feature_values:
            _finite(value, name="feature value")
        for name in (
            "normalized_marginal_utility",
            "normalized_target_improvement",
        ):
            _finite(getattr(self, name), name=name)
        if not -1.0 <= self.normalized_target_improvement <= 1.0:
            raise ValueError("normalized_target_improvement must lie in [-1, 1]")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "decision_sha256": self.decision_sha256,
            "campaign_generation": self.campaign_generation,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "feature_row_sha256": self.feature_row_sha256,
            "feature_values_hex": [value.hex() for value in self.feature_values],
            "normalized_marginal_utility_hex": (
                self.normalized_marginal_utility.hex()
            ),
            "normalized_target_improvement_hex": (
                self.normalized_target_improvement.hex()
            ),
            "evaluator_receipt_sha256": self.evaluator_receipt_sha256,
        }

    @property
    def observation_sha256(self) -> str:
        return _hash(_OBSERVATION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "observation_sha256": self.observation_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedSelectedObservation
            and self.observation_sha256 == other.observation_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedStateUpdateReceipt:
    prior_state_sha256: str
    next_state: TargetConditionedAcquisitionState
    campaign_generation: int
    decision_sha256s: tuple[str, ...]
    observation_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.prior_state_sha256, "prior_state_sha256")
        if type(self.next_state) is not TargetConditionedAcquisitionState:
            raise TypeError("next_state must be exact")
        self.next_state.__post_init__()
        if type(self.campaign_generation) is not int or self.campaign_generation <= 0:
            raise ValueError("campaign_generation must be positive")
        for name in ("decision_sha256s", "observation_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be non-empty")
            for value in values:
                require_sha256(value, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "prior_state_sha256": self.prior_state_sha256,
            "next_state": self.next_state.to_record(),
            "campaign_generation": self.campaign_generation,
            "decision_sha256s": list(self.decision_sha256s),
            "observation_sha256s": list(self.observation_sha256s),
            "update_boundary": "all_concurrent_lanes_complete",
            "rejected_outcomes_consulted": False,
        }

    @property
    def update_sha256(self) -> str:
        return _hash(_UPDATE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "update_sha256": self.update_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedStateUpdateReceipt
            and self.update_sha256 == other.update_sha256
        )

    __hash__ = None


def update_target_conditioned_state(
    state: TargetConditionedAcquisitionState,
    *,
    decisions: tuple[TargetConditionedSlateDecision, ...],
    observations: tuple[TargetConditionedSelectedObservation, ...],
) -> TargetConditionedStateUpdateReceipt:
    """Update once after every concurrent lane in one generation completes."""

    if type(state) is not TargetConditionedAcquisitionState:
        raise TypeError("state must be exact")
    state.__post_init__()
    if type(decisions) is not tuple or not decisions or any(
        type(value) is not TargetConditionedSlateDecision for value in decisions
    ):
        raise TypeError("decisions must contain exact decisions")
    if type(observations) is not tuple or not observations or any(
        type(value) is not TargetConditionedSelectedObservation
        for value in observations
    ):
        raise TypeError("observations must contain exact selected observations")
    for value in decisions:
        value.revalidate()
        if value.request.state != state:
            raise ValueError("concurrent decisions must share the exact prior state")
    for value in observations:
        value.__post_init__()
    generations = {value.request.campaign_generation for value in decisions}
    generations.update(value.campaign_generation for value in observations)
    if len(generations) != 1:
        raise ValueError("one update cannot mix campaign generations")
    generation = next(iter(generations))
    if state.cutoff_generation >= generation:
        raise ValueError("state cutoff reaches the update generation")
    decisions_by_sha = {value.decision_sha256: value for value in decisions}
    if len(decisions_by_sha) != len(decisions):
        raise ValueError("decisions must be unique")
    expected: dict[tuple[str, str], tuple[str, str, tuple[float, ...]]] = {}
    for decision in decisions:
        features = {
            value.option_id: value for value in decision.request.member_features
        }
        for selected in decision.selected:
            row = features[selected.option_id]
            expected[(decision.decision_sha256, selected.option_id)] = (
                selected.option_identity_sha256,
                row.feature_row_sha256,
                row.values,
            )
    observed = {
        (value.decision_sha256, value.option_id): value for value in observations
    }
    if len(observed) != len(observations) or set(observed) != set(expected):
        raise ValueError("updates must cover exactly every selected member once")
    ordered = []
    for key in sorted(expected):
        observation = observed[key]
        option_identity, feature_sha, feature_values = expected[key]
        if (
            observation.option_identity_sha256 != option_identity
            or observation.feature_row_sha256 != feature_sha
            or observation.feature_values != feature_values
        ):
            raise ValueError("observation differs from its selected feature row")
        ordered.append(observation)
    next_state = TargetConditionedAcquisitionState(
        campaign_scope_sha256=state.campaign_scope_sha256,
        training_data_sha256=state.training_data_sha256,
        marginal_head=state.marginal_head.update(
            [value.feature_values for value in ordered],
            [value.normalized_marginal_utility for value in ordered],
        ),
        direction_head=state.direction_head.update(
            [value.feature_values for value in ordered],
            [value.normalized_target_improvement for value in ordered],
        ),
        cutoff_generation=generation,
        selected_observation_count=(
            state.selected_observation_count + len(ordered)
        ),
    )
    return TargetConditionedStateUpdateReceipt(
        prior_state_sha256=state.state_sha256,
        next_state=next_state,
        campaign_generation=generation,
        decision_sha256s=tuple(sorted(decisions_by_sha)),
        observation_sha256s=tuple(
            sorted(value.observation_sha256 for value in ordered)
        ),
    )


__all__ = [
    "BASE_REALIZABILITY_DEFINITION_SHA256",
    "BASE_REALIZABILITY_PROJECTOR_ID",
    "BASE_REALIZABILITY_PROJECTOR_VERSION",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "PrequentialLinearGaussianHead",
    "RealizablePortfolioSet",
    "TargetConditionedAcquisitionProfile",
    "TargetConditionedAcquisitionState",
    "TargetConditionedAllocatedMember",
    "TargetConditionedMemberFeatures",
    "TargetConditionedMemberScore",
    "TargetConditionedMetaPrior",
    "TargetConditionedPrequentialSlatePolicy",
    "TargetConditionedSelectedObservation",
    "TargetConditionedSlateDecision",
    "TargetConditionedSlateRequest",
    "TargetConditionedSlateRole",
    "TargetConditionedStateUpdateReceipt",
    "enumerate_base_realizable_portfolios",
    "update_target_conditioned_state",
]

"""Frozen, outcome-blind consequence scorers for materialized actions.

Feature construction is an injected port.  This module owns only the
authenticated feature-vector boundary and deterministic application of frozen
standardized linear models.  It deliberately has no workload, objective,
model-provider, prompt, or configuration-schema branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Protocol, runtime_checkable

from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)


FROZEN_HURDLE_SCORE_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_FEATURE_BATCH_DOMAIN = (
    b"agent-evolve:materialized-action-feature-batch:v1\x00"
)
_LINEAR_MODEL_DOMAIN = b"agent-evolve:frozen-standardized-linear-model:v1\x00"
_SCORER_DOMAIN = b"agent-evolve:frozen-hurdle-action-scorer:v1\x00"
_SCORER_EVIDENCE_DOMAIN = (
    b"agent-evolve:frozen-hurdle-action-score-evidence:v1\x00"
)
_WINSORIZED_SCORER_DOMAIN = (
    b"agent-evolve:winsorized-frozen-hurdle-action-scorer:v1\x00"
)
_WINSORIZED_SCORER_EVIDENCE_DOMAIN = (
    b"agent-evolve:winsorized-frozen-hurdle-score-evidence:v1\x00"
)


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


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _finite_tuple(
    values: tuple[float, ...],
    *,
    name: str,
    nonzero: bool = False,
) -> None:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a non-empty exact tuple")
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise TypeError(f"{name} must contain finite exact floats")
        if nonzero and value <= 0.0:
            raise ValueError(f"{name} must contain positive values")


@dataclass(frozen=True, slots=True)
class MaterializedActionFeatureVector:
    """One canonical feature vector for one sealed materialized action."""

    action_sha256: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _finite_tuple(self.values, name="feature values")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "values_hex": [value.hex() for value in self.values],
        }


@dataclass(frozen=True, slots=True)
class MaterializedActionFeatureBatch:
    """Authenticated complete features for one sealed proposal universe."""

    projection_id: str
    projection_version: int
    projection_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    feature_names: tuple[str, ...]
    vectors: tuple[MaterializedActionFeatureVector, ...]
    candidate_outcomes_observed: bool
    evidence: FrozenJsonObject
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.projection_id, name="projection_id")
        if type(self.projection_version) is not int or self.projection_version <= 0:
            raise ValueError("projection_version must be positive")
        require_sha256(
            self.projection_definition_sha256,
            "projection_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if (
            type(self.proposal_sha256s) is not tuple
            or not self.proposal_sha256s
            or self.proposal_sha256s
            != tuple(sorted(set(self.proposal_sha256s)))
        ):
            raise ValueError(
                "proposal_sha256s must be non-empty, unique, and canonical"
            )
        for value in self.proposal_sha256s:
            require_sha256(value, "proposal_sha256")
        if (
            type(self.feature_names) is not tuple
            or not self.feature_names
            or self.feature_names != tuple(dict.fromkeys(self.feature_names))
        ):
            raise ValueError("feature_names must be non-empty and unique")
        for value in self.feature_names:
            _token(value, name="feature name")
        if type(self.vectors) is not tuple or not self.vectors:
            raise ValueError("vectors must be a non-empty exact tuple")
        for vector in self.vectors:
            if type(vector) is not MaterializedActionFeatureVector:
                raise TypeError("vectors must contain exact feature vectors")
            vector.__post_init__()
            if len(vector.values) != len(self.feature_names):
                raise ValueError("feature vector width differs from feature_names")
        action_ids = tuple(value.action_sha256 for value in self.vectors)
        if action_ids != tuple(sorted(set(action_ids))):
            raise ValueError("feature vectors must be unique and canonical")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError(
                "candidate_outcomes_observed must be an exact bool"
            )
        if self.candidate_outcomes_observed:
            raise ValueError("prequential features cannot observe current outcomes")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("feature evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "batch_sha256",
            _hash(_FEATURE_BATCH_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "projection": {
                "projection_id": self.projection_id,
                "projection_version": self.projection_version,
                "definition_sha256": self.projection_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "feature_names": list(self.feature_names),
            "vectors": [value.to_record() for value in self.vectors],
            "candidate_outcomes_observed": self.candidate_outcomes_observed,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "batch_sha256": self.batch_sha256}


@runtime_checkable
class MaterializedActionFeatureProjectionPort(Protocol):
    """Project complete sealed proposals using strictly prior evidence."""

    projection_id: str
    projection_version: int
    definition_sha256: str
    feature_names: tuple[str, ...]

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionFeatureBatch: ...


@dataclass(frozen=True, slots=True)
class FrozenStandardizedLinearModel:
    """Immutable standardized linear predictor with authenticated identity."""

    model_id: str
    family: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.model_id, name="model_id")
        _token(self.family, name="family")
        if (
            type(self.feature_names) is not tuple
            or not self.feature_names
            or self.feature_names != tuple(dict.fromkeys(self.feature_names))
        ):
            raise ValueError("feature_names must be non-empty and unique")
        for value in self.feature_names:
            _token(value, name="feature name")
        _finite_tuple(self.means, name="means")
        _finite_tuple(self.scales, name="scales", nonzero=True)
        _finite_tuple(self.coefficients, name="coefficients")
        width = len(self.feature_names)
        if len(self.means) != width or len(self.scales) != width:
            raise ValueError("standardizer width differs from feature ABI")
        if len(self.coefficients) != width + 1:
            raise ValueError(
                "coefficients must contain one intercept plus feature weights"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(_LINEAR_MODEL_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "family": self.family,
            "feature_names": list(self.feature_names),
            "mean_hex": [value.hex() for value in self.means],
            "scale_hex": [value.hex() for value in self.scales],
            "coefficients_hex": [value.hex() for value in self.coefficients],
        }

    def predict(self, values: tuple[float, ...]) -> float:
        self.__post_init__()
        _finite_tuple(values, name="prediction features")
        if len(values) != len(self.feature_names):
            raise ValueError("prediction vector differs from feature ABI")
        result = self.coefficients[0] + math.fsum(
            coefficient * ((value - mean) / scale)
            for value, mean, scale, coefficient in zip(
                values,
                self.means,
                self.scales,
                self.coefficients[1:],
                strict=True,
            )
        )
        if not math.isfinite(result):
            raise RuntimeError("frozen linear prediction became non-finite")
        return float(result)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "definition_sha256": self.definition_sha256,
        }


class FrozenHurdleScoreKind(str, Enum):
    POSITIVE_PROBABILITY = "positive_probability"
    EXPECTED_POSITIVE_MAGNITUDE = "expected_positive_magnitude"


@dataclass(frozen=True, slots=True)
class FrozenHurdleMaterializedActionScorer:
    """Apply a frozen positive/magnitude hurdle model before evaluation."""

    scorer_id: str
    projection: MaterializedActionFeatureProjectionPort = field(
        repr=False,
        compare=False,
    )
    positive_model: FrozenStandardizedLinearModel
    magnitude_model: FrozenStandardizedLinearModel
    score_kind: FrozenHurdleScoreKind
    source_fit_sha256: str
    scorer_version: int = FROZEN_HURDLE_SCORE_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.scorer_id, name="scorer_id")
        if type(self.scorer_version) is not int or self.scorer_version <= 0:
            raise ValueError("scorer_version must be positive")
        require_sha256(self.source_fit_sha256, "source_fit_sha256")
        if not isinstance(
            self.projection,
            MaterializedActionFeatureProjectionPort,
        ):
            raise TypeError("projection must implement its runtime port")
        _token(self.projection.projection_id, name="projection_id")
        if (
            type(self.projection.projection_version) is not int
            or self.projection.projection_version <= 0
        ):
            raise ValueError("projection_version must be positive")
        require_sha256(
            self.projection.definition_sha256,
            "projection definition_sha256",
        )
        if type(self.positive_model) is not FrozenStandardizedLinearModel:
            raise TypeError("positive_model must be exact")
        if type(self.magnitude_model) is not FrozenStandardizedLinearModel:
            raise TypeError("magnitude_model must be exact")
        self.positive_model.__post_init__()
        self.magnitude_model.__post_init__()
        if type(self.score_kind) is not FrozenHurdleScoreKind:
            raise TypeError("score_kind must be exact")
        feature_names = self.projection.feature_names
        if (
            feature_names != self.positive_model.feature_names
            or feature_names != self.magnitude_model.feature_names
        ):
            raise ValueError("projection and frozen models use different feature ABIs")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _SCORER_DOMAIN,
                {
                    "schema_version": 1,
                    "scorer_id": self.scorer_id,
                    "scorer_version": self.scorer_version,
                    "score_kind": self.score_kind.value,
                    "source_fit_sha256": self.source_fit_sha256,
                    "projection": {
                        "projection_id": self.projection.projection_id,
                        "projection_version": (
                            self.projection.projection_version
                        ),
                        "definition_sha256": (
                            self.projection.definition_sha256
                        ),
                    },
                    "positive_model_sha256": (
                        self.positive_model.definition_sha256
                    ),
                    "magnitude_model_sha256": (
                        self.magnitude_model.definition_sha256
                    ),
                    "logit_clip": [-30, 30],
                    "log_magnitude_clip": [-30, 30],
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = min(30.0, max(-30.0, value))
        return 1.0 / (1.0 + math.exp(-clipped))

    async def score(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionScoreBatch:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        action_sha256s = tuple(
            sorted(
                action.action_sha256
                for proposal in proposals
                for action in proposal.actions
            )
        )
        features = await self.projection.project(request, proposals)
        if type(features) is not MaterializedActionFeatureBatch:
            raise TypeError("feature projection returned a foreign batch")
        features.__post_init__()
        if (
            features.projection_id,
            features.projection_version,
            features.projection_definition_sha256,
        ) != (
            self.projection.projection_id,
            self.projection.projection_version,
            self.projection.definition_sha256,
        ):
            raise ValueError("feature batch differs from its projection")
        if features.residual_request_sha256 != request.request_sha256:
            raise ValueError("feature batch targets another residual request")
        if features.proposal_sha256s != proposal_sha256s:
            raise ValueError("feature batch targets another proposal universe")
        if features.feature_names != self.positive_model.feature_names:
            raise ValueError("feature batch differs from the frozen model ABI")
        if tuple(value.action_sha256 for value in features.vectors) != (
            action_sha256s
        ):
            raise ValueError("feature batch must exactly cover sealed actions")

        scores: list[MaterializedActionScore] = []
        for vector in features.vectors:
            probability = self._sigmoid(
                self.positive_model.predict(vector.values)
            )
            if self.score_kind is FrozenHurdleScoreKind.POSITIVE_PROBABILITY:
                value = probability
            else:
                log_magnitude = self.magnitude_model.predict(vector.values)
                magnitude = max(
                    0.0,
                    math.expm1(min(30.0, max(-30.0, log_magnitude))),
                )
                value = probability * magnitude
            scores.append(
                MaterializedActionScore(
                    action_sha256=vector.action_sha256,
                    value=float(value),
                )
            )
        evidence_sha256 = _hash(
            _SCORER_EVIDENCE_DOMAIN,
            {
                "scorer_definition_sha256": self.definition_sha256,
                "source_fit_sha256": self.source_fit_sha256,
                "feature_batch_sha256": features.batch_sha256,
                "candidate_outcomes_observed": False,
            },
        )
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            scores=tuple(scores),
            candidate_outcomes_observed=False,
            evidence_sha256=evidence_sha256,
        )


@dataclass(frozen=True, slots=True)
class WinsorizedFrozenHurdleMaterializedActionScorer(
    FrozenHurdleMaterializedActionScorer
):
    """Apply independently standardized models with bounded source z-scores."""

    winsorization_limit: float = 3.0

    def __post_init__(self) -> None:
        FrozenHurdleMaterializedActionScorer.__post_init__(self)
        if (
            type(self.winsorization_limit) is not float
            or not math.isfinite(self.winsorization_limit)
            or self.winsorization_limit <= 0.0
        ):
            raise ValueError(
                "winsorization_limit must be a positive finite float"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _WINSORIZED_SCORER_DOMAIN,
                {
                    "schema_version": 1,
                    "scorer_id": self.scorer_id,
                    "scorer_version": self.scorer_version,
                    "score_kind": self.score_kind.value,
                    "source_fit_sha256": self.source_fit_sha256,
                    "projection": {
                        "projection_id": self.projection.projection_id,
                        "projection_version": (
                            self.projection.projection_version
                        ),
                        "definition_sha256": (
                            self.projection.definition_sha256
                        ),
                    },
                    "positive_model_sha256": (
                        self.positive_model.definition_sha256
                    ),
                    "magnitude_model_sha256": (
                        self.magnitude_model.definition_sha256
                    ),
                    "winsorization_limit_hex": (
                        self.winsorization_limit.hex()
                    ),
                    "standardization": (
                        "independent_frozen_standardizer_per_hurdle_model"
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def _winsorized_linear(
        self,
        model: FrozenStandardizedLinearModel,
        values: tuple[float, ...],
    ) -> float:
        result = model.coefficients[0] + math.fsum(
            coefficient
            * min(
                self.winsorization_limit,
                max(
                    -self.winsorization_limit,
                    (value - mean) / scale,
                ),
            )
            for value, mean, scale, coefficient in zip(
                values,
                model.means,
                model.scales,
                model.coefficients[1:],
                strict=True,
            )
        )
        if not math.isfinite(result):
            raise RuntimeError("winsorized hurdle prediction became non-finite")
        return float(result)

    async def score(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionScoreBatch:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        action_sha256s = tuple(
            sorted(
                action.action_sha256
                for proposal in proposals
                for action in proposal.actions
            )
        )
        features = await self.projection.project(request, proposals)
        if type(features) is not MaterializedActionFeatureBatch:
            raise TypeError("feature projection returned a foreign batch")
        features.__post_init__()
        if (
            features.projection_id,
            features.projection_version,
            features.projection_definition_sha256,
        ) != (
            self.projection.projection_id,
            self.projection.projection_version,
            self.projection.definition_sha256,
        ):
            raise ValueError("feature batch differs from its projection")
        if (
            features.residual_request_sha256 != request.request_sha256
            or features.proposal_sha256s != proposal_sha256s
            or features.feature_names != self.positive_model.feature_names
            or tuple(value.action_sha256 for value in features.vectors)
            != action_sha256s
        ):
            raise ValueError("feature batch differs from sealed universe")

        scores: list[MaterializedActionScore] = []
        for vector in features.vectors:
            probability = self._sigmoid(
                self._winsorized_linear(
                    self.positive_model,
                    vector.values,
                )
            )
            if self.score_kind is FrozenHurdleScoreKind.POSITIVE_PROBABILITY:
                value = probability
            else:
                log_magnitude = self._winsorized_linear(
                    self.magnitude_model,
                    vector.values,
                )
                magnitude = max(
                    0.0,
                    math.expm1(
                        min(30.0, max(-30.0, log_magnitude))
                    ),
                )
                value = probability * magnitude
            scores.append(
                MaterializedActionScore(
                    action_sha256=vector.action_sha256,
                    value=float(value),
                )
            )
        evidence_sha256 = _hash(
            _WINSORIZED_SCORER_EVIDENCE_DOMAIN,
            {
                "scorer_definition_sha256": self.definition_sha256,
                "source_fit_sha256": self.source_fit_sha256,
                "feature_batch_sha256": features.batch_sha256,
                "candidate_outcomes_observed": False,
            },
        )
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            scores=tuple(scores),
            candidate_outcomes_observed=False,
            evidence_sha256=evidence_sha256,
        )


__all__ = [
    "FROZEN_HURDLE_SCORE_VERSION",
    "FrozenHurdleMaterializedActionScorer",
    "FrozenHurdleScoreKind",
    "FrozenStandardizedLinearModel",
    "MaterializedActionFeatureBatch",
    "MaterializedActionFeatureProjectionPort",
    "MaterializedActionFeatureVector",
    "WinsorizedFrozenHurdleMaterializedActionScorer",
]

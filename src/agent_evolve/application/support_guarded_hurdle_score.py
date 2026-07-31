"""Support-aware frozen consequence scoring for sealed action populations.

A feature definition can be portable while a fitted coefficient is not.
This module bounds the authority of a frozen standardized linear model using
only its source standardizer and the complete current pre-evaluation feature
batch.  It contains no workload, objective, model-provider, prompt, or
configuration-schema branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

from agent_evolve.application.frozen_hurdle_score import (
    FrozenHurdleScoreKind,
    FrozenStandardizedLinearModel,
    MaterializedActionFeatureBatch,
    MaterializedActionFeatureProjectionPort,
)
from agent_evolve.application.prequential_score_portfolio import (
    MaterializedActionScore,
    MaterializedActionScoreBatch,
    MaterializedActionScoreReliabilityEvidence,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256


SUPPORT_GUARDED_HURDLE_SCORE_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_GROUP_DOMAIN = b"agent-evolve:frozen-feature-support-group:v1\x00"
_SCORER_DOMAIN = b"agent-evolve:support-guarded-hurdle-scorer:v1\x00"
_EVIDENCE_DOMAIN = b"agent-evolve:support-guarded-hurdle-evidence:v1\x00"


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


def _probability(value: float) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("authority must be a finite exact float")
    if not 0.0 <= value <= 1.0:
        raise ValueError("authority must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class FrozenFeatureSupportGroup:
    """One authenticated group of jointly guarded standardized features."""

    group_id: str
    feature_names: tuple[str, ...]
    base_authority: float = 1.0
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.group_id, name="group_id")
        if (
            type(self.feature_names) is not tuple
            or not self.feature_names
            or self.feature_names != tuple(dict.fromkeys(self.feature_names))
        ):
            raise ValueError(
                "feature_names must be a non-empty unique exact tuple"
            )
        for value in self.feature_names:
            _token(value, name="feature name")
        _probability(self.base_authority)
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _GROUP_DOMAIN,
                {
                    "schema_version": 1,
                    "group_id": self.group_id,
                    "feature_names": list(self.feature_names),
                    "base_authority_hex": self.base_authority.hex(),
                },
            ),
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "group_id": self.group_id,
            "feature_names": list(self.feature_names),
            "base_authority_hex": self.base_authority.hex(),
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenFeatureSupportEvidence:
    """Auditable support distances and effective authority for one batch."""

    scorer_definition_sha256: str
    source_fit_sha256: str
    feature_batch_sha256: str
    residual_request_sha256: str
    group_rows: tuple[tuple[str, float, float], ...]
    candidate_outcomes_observed: bool
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(
            self.scorer_definition_sha256,
            "scorer_definition_sha256",
        )
        require_sha256(self.source_fit_sha256, "source_fit_sha256")
        require_sha256(self.feature_batch_sha256, "feature_batch_sha256")
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if (
            type(self.group_rows) is not tuple
            or not self.group_rows
            or tuple(value[0] for value in self.group_rows)
            != tuple(sorted({value[0] for value in self.group_rows}))
        ):
            raise ValueError("group_rows must be non-empty and canonical")
        for group_id, maximum_abs_z, effective_authority in self.group_rows:
            _token(group_id, name="group_id")
            if (
                type(maximum_abs_z) is not float
                or not math.isfinite(maximum_abs_z)
                or maximum_abs_z < 0.0
            ):
                raise ValueError("maximum_abs_z must be finite and nonnegative")
            _probability(effective_authority)
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError(
                "candidate_outcomes_observed must be an exact bool"
            )
        if self.candidate_outcomes_observed:
            raise ValueError("support evidence cannot observe current outcomes")
        object.__setattr__(
            self,
            "evidence_sha256",
            _hash(_EVIDENCE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scorer_definition_sha256": self.scorer_definition_sha256,
            "source_fit_sha256": self.source_fit_sha256,
            "feature_batch_sha256": self.feature_batch_sha256,
            "residual_request_sha256": self.residual_request_sha256,
            "groups": [
                {
                    "group_id": group_id,
                    "maximum_abs_source_z_hex": maximum_abs_z.hex(),
                    "effective_authority_hex": effective_authority.hex(),
                }
                for group_id, maximum_abs_z, effective_authority
                in self.group_rows
            ],
            "candidate_outcomes_observed": (
                self.candidate_outcomes_observed
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class SupportGuardedFrozenHurdleMaterializedActionScorer:
    """Apply a frozen hurdle model with batch-level source-support shrinkage."""

    scorer_id: str
    projection: MaterializedActionFeatureProjectionPort = field(
        repr=False,
        compare=False,
    )
    positive_model: FrozenStandardizedLinearModel
    magnitude_model: FrozenStandardizedLinearModel
    score_kind: FrozenHurdleScoreKind
    source_fit_sha256: str
    support_groups: tuple[FrozenFeatureSupportGroup, ...]
    support_radius: float = 3.0
    winsorization_limit: float = 3.0
    scorer_version: int = SUPPORT_GUARDED_HURDLE_SCORE_VERSION
    definition_sha256: str = field(init=False)
    _evidence_by_request: dict[str, FrozenFeatureSupportEvidence] = field(
        init=False,
        default_factory=dict,
        repr=False,
        compare=False,
    )

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
        self.positive_model.__post_init__()
        self.magnitude_model.__post_init__()
        if type(self.score_kind) is not FrozenHurdleScoreKind:
            raise TypeError("score_kind must be exact")
        feature_names = self.projection.feature_names
        if (
            feature_names != self.positive_model.feature_names
            or feature_names != self.magnitude_model.feature_names
        ):
            raise ValueError("projection and models use different feature ABIs")
        if (
            self.positive_model.means != self.magnitude_model.means
            or self.positive_model.scales != self.magnitude_model.scales
        ):
            raise ValueError(
                "guarded hurdle models must share one source standardizer"
            )
        if (
            type(self.support_groups) is not tuple
            or not self.support_groups
            or tuple(value.group_id for value in self.support_groups)
            != tuple(sorted({value.group_id for value in self.support_groups}))
        ):
            raise ValueError(
                "support_groups must be non-empty, unique, and canonical"
            )
        grouped_features: list[str] = []
        for group in self.support_groups:
            if type(group) is not FrozenFeatureSupportGroup:
                raise TypeError(
                    "support_groups must contain exact support groups"
                )
            group.__post_init__()
            grouped_features.extend(group.feature_names)
        if (
            len(grouped_features) != len(set(grouped_features))
            or set(grouped_features) != set(feature_names)
        ):
            raise ValueError(
                "support groups must partition the complete feature ABI"
            )
        for name in ("support_radius", "winsorization_limit"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a positive finite float")
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
                    "support_groups": [
                        value.to_record() for value in self.support_groups
                    ],
                    "support_radius_hex": self.support_radius.hex(),
                    "winsorization_limit_hex": (
                        self.winsorization_limit.hex()
                    ),
                    "support_rule": (
                        "base_authority_times_min_one_radius_over_batch_"
                        "maximum_absolute_source_z"
                    ),
                    "standardized_value_rule": (
                        "winsorize_then_multiply_effective_group_authority"
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = min(30.0, max(-30.0, value))
        return 1.0 / (1.0 + math.exp(-clipped))

    @staticmethod
    def _linear(
        model: FrozenStandardizedLinearModel,
        standardized: tuple[float, ...],
    ) -> float:
        result = model.coefficients[0] + math.fsum(
            coefficient * value
            for coefficient, value in zip(
                model.coefficients[1:],
                standardized,
                strict=True,
            )
        )
        if not math.isfinite(result):
            raise RuntimeError("guarded frozen score became non-finite")
        return float(result)

    def evidence_for(
        self,
        residual_request_sha256: str,
    ) -> FrozenFeatureSupportEvidence | None:
        require_sha256(
            residual_request_sha256,
            "residual_request_sha256",
        )
        return self._evidence_by_request.get(residual_request_sha256)

    def reliability(
        self,
        residual_request_sha256: str,
        component_ids: tuple[str, ...],
    ) -> MaterializedActionScoreReliabilityEvidence:
        require_sha256(
            residual_request_sha256,
            "residual_request_sha256",
        )
        if (
            type(component_ids) is not tuple
            or not component_ids
            or component_ids != tuple(sorted(set(component_ids)))
        ):
            raise ValueError(
                "component_ids must be non-empty, unique, and canonical"
            )
        evidence = self._evidence_by_request.get(
            residual_request_sha256
        )
        if evidence is None:
            raise ValueError(
                "support reliability is unavailable before batch scoring"
            )
        authority_by_group = {
            group_id: authority
            for group_id, _maximum_abs_z, authority in evidence.group_rows
        }
        missing = set(component_ids) - set(authority_by_group)
        if missing:
            raise ValueError(
                f"unknown support reliability components: {sorted(missing)}"
            )
        components = tuple(
            (component_id, authority_by_group[component_id])
            for component_id in component_ids
        )
        return MaterializedActionScoreReliabilityEvidence(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            component_authorities=components,
            overall_reliability=min(value for _, value in components),
            candidate_outcomes_observed=False,
            source_evidence_sha256=evidence.evidence_sha256,
        )

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
            features.residual_request_sha256 != request.request_sha256
            or features.proposal_sha256s != proposal_sha256s
            or tuple(value.action_sha256 for value in features.vectors)
            != action_sha256s
        ):
            raise ValueError("feature batch differs from the sealed universe")
        if features.feature_names != self.positive_model.feature_names:
            raise ValueError("feature batch differs from the model ABI")

        feature_index = {
            name: index for index, name in enumerate(features.feature_names)
        }
        standardized = {
            vector.action_sha256: tuple(
                (value - mean) / scale
                for value, mean, scale in zip(
                    vector.values,
                    self.positive_model.means,
                    self.positive_model.scales,
                    strict=True,
                )
            )
            for vector in features.vectors
        }
        group_rows: list[tuple[str, float, float]] = []
        authority_by_feature: dict[str, float] = {}
        for group in self.support_groups:
            maximum_abs_z = max(
                abs(values[feature_index[name]])
                for values in standardized.values()
                for name in group.feature_names
            )
            support_weight = (
                1.0
                if maximum_abs_z == 0.0
                else min(1.0, self.support_radius / maximum_abs_z)
            )
            effective_authority = (
                group.base_authority * support_weight
            )
            group_rows.append(
                (
                    group.group_id,
                    float(maximum_abs_z),
                    float(effective_authority),
                )
            )
            for name in group.feature_names:
                authority_by_feature[name] = float(effective_authority)

        scores: list[MaterializedActionScore] = []
        for action_sha256 in action_sha256s:
            transformed = tuple(
                min(
                    self.winsorization_limit,
                    max(-self.winsorization_limit, value),
                )
                * authority_by_feature[name]
                for name, value in zip(
                    features.feature_names,
                    standardized[action_sha256],
                    strict=True,
                )
            )
            positive = self._sigmoid(
                self._linear(self.positive_model, transformed)
            )
            if self.score_kind is FrozenHurdleScoreKind.POSITIVE_PROBABILITY:
                value = positive
            else:
                log_magnitude = self._linear(
                    self.magnitude_model,
                    transformed,
                )
                magnitude = max(
                    0.0,
                    math.expm1(
                        min(30.0, max(-30.0, log_magnitude))
                    ),
                )
                value = positive * magnitude
            scores.append(
                MaterializedActionScore(
                    action_sha256=action_sha256,
                    value=float(value),
                )
            )
        evidence = FrozenFeatureSupportEvidence(
            scorer_definition_sha256=self.definition_sha256,
            source_fit_sha256=self.source_fit_sha256,
            feature_batch_sha256=features.batch_sha256,
            residual_request_sha256=request.request_sha256,
            group_rows=tuple(group_rows),
            candidate_outcomes_observed=False,
        )
        prior = self._evidence_by_request.get(request.request_sha256)
        if prior is not None and prior != evidence:
            raise RuntimeError(
                "one request identity produced different support evidence"
            )
        self._evidence_by_request[request.request_sha256] = evidence
        return MaterializedActionScoreBatch(
            scorer_id=self.scorer_id,
            scorer_version=self.scorer_version,
            scorer_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            scores=tuple(scores),
            candidate_outcomes_observed=False,
            evidence_sha256=evidence.evidence_sha256,
        )


__all__ = [
    "SUPPORT_GUARDED_HURDLE_SCORE_VERSION",
    "FrozenFeatureSupportEvidence",
    "FrozenFeatureSupportGroup",
    "SupportGuardedFrozenHurdleMaterializedActionScorer",
]

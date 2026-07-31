"""Outcome-blind candidate opportunity against an observed archive prefix.

Forecast geometry is sealed before the current candidates are evaluated.
After each real evaluation wave, this policy may observe only the accepted
prefix and recompute the conditional archive value of every still-unobserved
forecast.  Workload semantics remain behind the injected archive-utility port.

The result is a recommendation, not evaluation authority.  A caller can keep
an existing controller as a protected fallback whenever this policy abstains.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.candidate_archive_consequence import (
    CandidateArchivePortfolioConsequenceUtilityPort,
    validate_candidate_archive_portfolio_consequence_utility,
)
from agent_evolve.application.forecast_geometry_portfolio import (
    MaterializedForecastGeometryBatch,
    MaterializedForecastGeometryMember,
)
from agent_evolve.application.prequential_archive_opportunity_calibration import (
    ArchiveOpportunityActionContext,
    ArchiveOpportunityCalibrationResult,
)
from agent_evolve.domain.patch import require_sha256


CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID = (
    "current_prefix_forecast_opportunity"
)
CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_POLICY_DOMAIN = b"agent-evolve:current-prefix-opportunity-policy:v1\x00"
_ARCHIVE_DOMAIN = b"agent-evolve:current-prefix-opportunity-archive:v1\x00"
_SCORE_DOMAIN = b"agent-evolve:current-prefix-opportunity-score:v1\x00"
_RANKING_DOMAIN = b"agent-evolve:current-prefix-opportunity-ranking:v1\x00"


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


def _candidate_record(candidate: EvolutionCandidate) -> dict[str, object]:
    candidate.__post_init__()
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "generation": candidate.generation,
        "valid": candidate.valid,
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "objectives": [
            {"metric_id": metric_id, "value_hex": value.hex()}
            for metric_id, value in candidate.objectives
        ],
    }


def _archive_sha256(
    prior_candidates: tuple[EvolutionCandidate, ...],
    current_prefix_candidates: tuple[EvolutionCandidate, ...],
) -> str:
    return _hash(
        _ARCHIVE_DOMAIN,
        {
            "prior": [
                _candidate_record(value)
                for value in sorted(
                    prior_candidates,
                    key=lambda item: item.candidate_id.value,
                )
            ],
            "current_prefix": [
                _candidate_record(value)
                for value in sorted(
                    current_prefix_candidates,
                    key=lambda item: item.candidate_id.value,
                )
            ],
        },
    )


@dataclass(frozen=True, slots=True)
class CurrentPrefixForecastOpportunityScore:
    """Authenticated candidate-specific conditional opportunity evidence."""

    action_sha256: str
    phenotype_identity_sha256: str
    reliability: float
    adverse_gain: float
    central_gain: float
    favorable_gain: float
    robust_gain: float
    exploration_gain: float
    acquisition_value: float
    abstained: bool
    abstention_reason: str | None
    source_member_sha256: str
    raw_acquisition_value: float | None = None
    calibration_result: ArchiveOpportunityCalibrationResult | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    score_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        require_sha256(self.source_member_sha256, "source_member_sha256")
        if (
            type(self.reliability) is not float
            or not math.isfinite(self.reliability)
            or not 0.0 <= self.reliability <= 1.0
        ):
            raise ValueError("reliability must lie in [0, 1]")
        for name in (
            "adverse_gain",
            "central_gain",
            "favorable_gain",
            "robust_gain",
            "exploration_gain",
            "acquisition_value",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.abstained) is not bool:
            raise TypeError("abstained must be exact")
        if self.abstained:
            if self.abstention_reason is None:
                raise ValueError("an abstention must have a reason")
            _require_token(self.abstention_reason, name="abstention_reason")
        elif self.abstention_reason is not None:
            raise ValueError("a recommendation cannot have an abstention reason")
        if (self.raw_acquisition_value is None) != (
            self.calibration_result is None
        ):
            raise ValueError(
                "raw acquisition and calibration result must be jointly present"
            )
        if self.calibration_result is not None:
            if (
                type(self.calibration_result)
                is not ArchiveOpportunityCalibrationResult
            ):
                raise TypeError("calibration_result must be exact")
            self.calibration_result.__post_init__()
            raw_acquisition_valid = (
                type(self.raw_acquisition_value) is float
                and math.isfinite(self.raw_acquisition_value)
                and self.raw_acquisition_value >= 0.0
            )
            if not raw_acquisition_valid:
                raise ValueError(
                    "raw_acquisition_value must be finite and non-negative"
                )
            if (
                self.calibration_result.action_sha256
                != self.action_sha256
                or not math.isclose(
                    self.acquisition_value,
                    self.calibration_result.calibrated_acquisition_value,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    "calibration result differs from the scored action"
                )
        object.__setattr__(
            self,
            "score_sha256",
            _hash(_SCORE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "reliability_hex": self.reliability.hex(),
            "adverse_gain_hex": self.adverse_gain.hex(),
            "central_gain_hex": self.central_gain.hex(),
            "favorable_gain_hex": self.favorable_gain.hex(),
            "robust_gain_hex": self.robust_gain.hex(),
            "exploration_gain_hex": self.exploration_gain.hex(),
            "acquisition_value_hex": self.acquisition_value.hex(),
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "source_member_sha256": self.source_member_sha256,
            "eligible_candidate_outcome_observed": False,
        }
        if self.calibration_result is not None:
            record["schema_version"] = 2
            record["raw_acquisition_value_hex"] = (
                self.raw_acquisition_value.hex()
            )
            record["calibration_result_sha256"] = (
                self.calibration_result.result_sha256
            )
        return record

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        result = {
            **self._unsigned_record(),
            "score_sha256": self.score_sha256,
        }
        if self.calibration_result is not None:
            result["calibration_result"] = (
                self.calibration_result.to_record()
            )
        return result


@dataclass(frozen=True, slots=True)
class CurrentPrefixForecastOpportunityRanking:
    """One cutoff-bound ranking over an unobserved sealed action market."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    geometry_batch_sha256: str
    archive_sha256: str
    consumed_action_sha256s: tuple[str, ...]
    eligible_action_sha256s: tuple[str, ...]
    scores: tuple[CurrentPrefixForecastOpportunityScore, ...]
    recommended_action_sha256s: tuple[str, ...]
    current_prefix_outcomes_observed: bool
    eligible_candidate_outcomes_observed: bool
    ranking_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        for value, name in (
            (self.policy_definition_sha256, "policy_definition_sha256"),
            (self.geometry_batch_sha256, "geometry_batch_sha256"),
            (self.archive_sha256, "archive_sha256"),
        ):
            require_sha256(value, name)
        for values, name in (
            (self.consumed_action_sha256s, "consumed_action_sha256s"),
            (self.eligible_action_sha256s, "eligible_action_sha256s"),
            (self.recommended_action_sha256s, "recommended_action_sha256s"),
        ):
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
            for value in values:
                require_sha256(value, name[:-1])
        if (
            type(self.scores) is not tuple
            or any(
                type(value) is not CurrentPrefixForecastOpportunityScore
                for value in self.scores
            )
        ):
            raise TypeError("scores must contain exact opportunity scores")
        for value in self.scores:
            value.__post_init__()
        action_ids = tuple(value.action_sha256 for value in self.scores)
        if action_ids != tuple(sorted(set(action_ids))):
            raise ValueError("scores must be action-canonical")
        if action_ids != self.eligible_action_sha256s:
            raise ValueError("scores must cover the exact eligible market")
        if not set(self.recommended_action_sha256s) <= set(action_ids):
            raise ValueError("recommendations escape the eligible market")
        score_by_action = {
            value.action_sha256: value for value in self.scores
        }
        if any(
            score_by_action[action_sha256].abstained
            for action_sha256 in self.recommended_action_sha256s
        ):
            raise ValueError("recommendations cannot include abstentions")
        recommended_phenotypes = tuple(
            score_by_action[value].phenotype_identity_sha256
            for value in self.recommended_action_sha256s
        )
        if len(recommended_phenotypes) != len(set(recommended_phenotypes)):
            raise ValueError("recommendations repeat a phenotype")
        if type(self.current_prefix_outcomes_observed) is not bool:
            raise TypeError("current_prefix_outcomes_observed must be exact")
        if type(self.eligible_candidate_outcomes_observed) is not bool:
            raise TypeError("eligible_candidate_outcomes_observed must be exact")
        if self.eligible_candidate_outcomes_observed:
            raise ValueError("eligible candidate outcomes must remain hidden")
        object.__setattr__(
            self,
            "ranking_sha256",
            _hash(_RANKING_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "geometry_batch_sha256": self.geometry_batch_sha256,
            "archive_sha256": self.archive_sha256,
            "consumed_action_sha256s": list(self.consumed_action_sha256s),
            "eligible_action_sha256s": list(self.eligible_action_sha256s),
            "score_sha256s": [value.score_sha256 for value in self.scores],
            "recommended_action_sha256s": list(
                self.recommended_action_sha256s
            ),
            "current_prefix_outcomes_observed": (
                self.current_prefix_outcomes_observed
            ),
            "eligible_candidate_outcomes_observed": (
                self.eligible_candidate_outcomes_observed
            ),
        }

    def to_record(self, *, include_scores: bool = False) -> dict[str, object]:
        self.__post_init__()
        result = {
            **self._unsigned_record(),
            "ranking_sha256": self.ranking_sha256,
        }
        if include_scores:
            result["scores"] = [value.to_record() for value in self.scores]
        return result


@runtime_checkable
class CurrentPrefixForecastOpportunityPolicyPort(Protocol):
    """Inverted API for raw or calibrated current-prefix opportunity."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def rank(
        self,
        *,
        prior_candidates: tuple[EvolutionCandidate, ...],
        current_prefix_candidates: tuple[EvolutionCandidate, ...],
        geometry: MaterializedForecastGeometryBatch,
        consumed_action_sha256s: tuple[str, ...] = (),
        eligible_action_sha256s: tuple[str, ...] | None = None,
        recommendation_count: int = 1,
        action_contexts: tuple[
            ArchiveOpportunityActionContext,
            ...,
        ] = (),
    ) -> CurrentPrefixForecastOpportunityRanking: ...


def validate_current_prefix_forecast_opportunity_policy(
    value: CurrentPrefixForecastOpportunityPolicyPort,
) -> tuple[str, int, str]:
    if not isinstance(
        value,
        CurrentPrefixForecastOpportunityPolicyPort,
    ):
        raise TypeError(
            "opportunity policy must implement its application port"
        )
    identity = (
        value.policy_id,
        value.policy_version,
        value.definition_sha256,
    )
    _require_token(identity[0], name="opportunity policy_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("opportunity policy_version must be positive")
    require_sha256(identity[2], "opportunity policy definition_sha256")
    return identity


@dataclass(frozen=True, slots=True)
class CurrentPrefixForecastOpportunityPolicy:
    """Rank forecasts by conditional archive lift after an observed prefix."""

    archive_utility: CandidateArchivePortfolioConsequenceUtilityPort = field(
        repr=False,
        compare=False,
    )
    central_scenario_id: str = "median"
    adverse_scenario_id: str = "adverse"
    favorable_scenario_id: str = "favorable"
    risk_aversion: float = 0.25
    exploration_weight: float = 0.0
    minimum_reliability: float = 0.0
    minimum_acquisition_value: float = 0.0
    policy_id: str = CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID
    policy_version: int = CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        utility_identity = (
            validate_candidate_archive_portfolio_consequence_utility(
                self.archive_utility
            )
        )
        for value, name in (
            (self.central_scenario_id, "central_scenario_id"),
            (self.adverse_scenario_id, "adverse_scenario_id"),
            (self.favorable_scenario_id, "favorable_scenario_id"),
            (self.policy_id, "policy_id"),
        ):
            _require_token(value, name=name)
        if self.policy_id != CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID:
            raise ValueError("policy_id is immutable")
        if self.policy_version != CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION:
            raise ValueError("policy_version is immutable")
        for name in (
            "risk_aversion",
            "exploration_weight",
            "minimum_acquisition_value",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            type(self.minimum_reliability) is not float
            or not math.isfinite(self.minimum_reliability)
            or not 0.0 <= self.minimum_reliability <= 1.0
        ):
            raise ValueError("minimum_reliability must lie in [0, 1]")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _POLICY_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "archive_utility": {
                        "utility_id": utility_identity[0],
                        "utility_version": utility_identity[1],
                        "definition_sha256": utility_identity[2],
                    },
                    "central_scenario_id": self.central_scenario_id,
                    "adverse_scenario_id": self.adverse_scenario_id,
                    "favorable_scenario_id": self.favorable_scenario_id,
                    "risk_aversion_hex": self.risk_aversion.hex(),
                    "exploration_weight_hex": self.exploration_weight.hex(),
                    "minimum_reliability_hex": self.minimum_reliability.hex(),
                    "minimum_acquisition_value_hex": (
                        self.minimum_acquisition_value.hex()
                    ),
                    "selection_currency": (
                        "current_prefix_conditional_archive_opportunity"
                    ),
                    "eligible_candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def _gain(
        self,
        archive: tuple[EvolutionCandidate, ...],
        member: MaterializedForecastGeometryMember,
        scenario_id: str,
    ) -> float:
        result = self.archive_utility.portfolio_marginal_utility(
            archive,
            (member.scenario(scenario_id).as_mapping(),),
        )
        if type(result) is not float or not math.isfinite(result) or result < 0.0:
            raise ValueError("archive utility returned an invalid opportunity")
        return result

    def rank(
        self,
        *,
        prior_candidates: tuple[EvolutionCandidate, ...],
        current_prefix_candidates: tuple[EvolutionCandidate, ...],
        geometry: MaterializedForecastGeometryBatch,
        consumed_action_sha256s: tuple[str, ...] = (),
        eligible_action_sha256s: tuple[str, ...] | None = None,
        recommendation_count: int = 1,
        action_contexts: tuple[
            ArchiveOpportunityActionContext,
            ...,
        ] = (),
    ) -> CurrentPrefixForecastOpportunityRanking:
        """Score only unobserved actions against the real accepted prefix."""

        self.__post_init__()
        for values, name in (
            (prior_candidates, "prior_candidates"),
            (current_prefix_candidates, "current_prefix_candidates"),
        ):
            if (
                type(values) is not tuple
                or any(type(value) is not EvolutionCandidate for value in values)
            ):
                raise TypeError(f"{name} must be an exact candidate tuple")
            for value in values:
                value.__post_init__()
        archive = (*prior_candidates, *current_prefix_candidates)
        candidate_ids = tuple(value.candidate_id for value in archive)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("prior and current-prefix candidates overlap")
        if type(geometry) is not MaterializedForecastGeometryBatch:
            raise TypeError("geometry must be an exact batch")
        geometry.__post_init__()
        all_action_ids = tuple(value.action_sha256 for value in geometry.members)
        if (
            type(consumed_action_sha256s) is not tuple
            or consumed_action_sha256s
            != tuple(sorted(set(consumed_action_sha256s)))
        ):
            raise ValueError("consumed action hashes must be unique and canonical")
        for value in consumed_action_sha256s:
            require_sha256(value, "consumed_action_sha256")
        if not set(consumed_action_sha256s) <= set(all_action_ids):
            raise ValueError("consumed actions escape the sealed geometry")
        member_by_action = {
            value.action_sha256: value for value in geometry.members
        }
        consumed_phenotypes = {
            member_by_action[value].phenotype_identity_sha256
            for value in consumed_action_sha256s
        }
        eligible = (
            tuple(
                value
                for value in all_action_ids
                if (
                    value not in set(consumed_action_sha256s)
                    and member_by_action[
                        value
                    ].phenotype_identity_sha256 not in consumed_phenotypes
                )
            )
            if eligible_action_sha256s is None
            else eligible_action_sha256s
        )
        if type(eligible) is not tuple or eligible != tuple(sorted(set(eligible))):
            raise ValueError("eligible action hashes must be unique and canonical")
        for value in eligible:
            require_sha256(value, "eligible_action_sha256")
        if not set(eligible) <= set(all_action_ids) - set(consumed_action_sha256s):
            raise ValueError("eligible actions are not unconsumed sealed actions")
        if any(
            member_by_action[value].phenotype_identity_sha256
            in consumed_phenotypes
            for value in eligible
        ):
            raise ValueError("eligible actions repeat a consumed phenotype")
        if type(recommendation_count) is not int or recommendation_count < 0:
            raise ValueError("recommendation_count must be non-negative")
        if (
            type(action_contexts) is not tuple
            or any(
                type(value) is not ArchiveOpportunityActionContext
                for value in action_contexts
            )
        ):
            raise TypeError(
                "action_contexts must contain exact generic contexts"
            )
        for value in action_contexts:
            value.__post_init__()
        context_action_ids = tuple(
            value.action_sha256 for value in action_contexts
        )
        if context_action_ids and context_action_ids != tuple(
            sorted(set(context_action_ids))
        ):
            raise ValueError(
                "action_contexts must be unique and action canonical"
            )
        if context_action_ids and set(context_action_ids) != set(eligible):
            raise ValueError(
                "action_contexts must cover the exact eligible market"
            )

        required_scenarios = {
            self.central_scenario_id,
            self.adverse_scenario_id,
            self.favorable_scenario_id,
        }
        scores: list[CurrentPrefixForecastOpportunityScore] = []
        for action_sha256 in eligible:
            member = member_by_action[action_sha256]
            if not required_scenarios <= {
                value.scenario_id for value in member.scenarios
            }:
                raise ValueError("forecast geometry omits an opportunity scenario")
            adverse = self._gain(
                archive,
                member,
                self.adverse_scenario_id,
            )
            central = self._gain(
                archive,
                member,
                self.central_scenario_id,
            )
            favorable = self._gain(
                archive,
                member,
                self.favorable_scenario_id,
            )
            robust = max(
                0.0,
                central
                - self.risk_aversion * max(0.0, central - adverse),
            )
            exploration = max(0.0, favorable - central)
            acquisition = member.reliability * (
                robust + self.exploration_weight * exploration
            )
            if member.reliability < self.minimum_reliability:
                abstention_reason = "below_reliability_floor"
            elif acquisition <= self.minimum_acquisition_value:
                abstention_reason = "no_identified_conditional_opportunity"
            else:
                abstention_reason = None
            scores.append(
                CurrentPrefixForecastOpportunityScore(
                    action_sha256=action_sha256,
                    phenotype_identity_sha256=(
                        member.phenotype_identity_sha256
                    ),
                    reliability=member.reliability,
                    adverse_gain=float(adverse),
                    central_gain=float(central),
                    favorable_gain=float(favorable),
                    robust_gain=float(robust),
                    exploration_gain=float(exploration),
                    acquisition_value=float(acquisition),
                    abstained=abstention_reason is not None,
                    abstention_reason=abstention_reason,
                    source_member_sha256=member.member_sha256,
                )
            )
        scores.sort(key=lambda value: value.action_sha256)
        ranked = sorted(
            (value for value in scores if not value.abstained),
            key=lambda value: (
                -value.acquisition_value,
                -value.central_gain,
                -value.reliability,
                value.action_sha256,
            ),
        )
        recommended: list[str] = []
        phenotypes: set[str] = set()
        if recommendation_count:
            for value in ranked:
                if value.phenotype_identity_sha256 in phenotypes:
                    continue
                recommended.append(value.action_sha256)
                phenotypes.add(value.phenotype_identity_sha256)
                if len(recommended) == recommendation_count:
                    break
        return CurrentPrefixForecastOpportunityRanking(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            geometry_batch_sha256=geometry.batch_sha256,
            archive_sha256=_archive_sha256(
                prior_candidates,
                current_prefix_candidates,
            ),
            consumed_action_sha256s=consumed_action_sha256s,
            eligible_action_sha256s=eligible,
            scores=tuple(scores),
            recommended_action_sha256s=tuple(sorted(recommended)),
            current_prefix_outcomes_observed=bool(current_prefix_candidates),
            eligible_candidate_outcomes_observed=False,
        )


__all__ = [
    "CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID",
    "CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION",
    "CurrentPrefixForecastOpportunityPolicy",
    "CurrentPrefixForecastOpportunityPolicyPort",
    "CurrentPrefixForecastOpportunityRanking",
    "CurrentPrefixForecastOpportunityScore",
    "validate_current_prefix_forecast_opportunity_policy",
]

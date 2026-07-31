"""Generic set-valued allocation over sealed consequence scenarios.

The application core does not interpret objective names, directions, forecast
quantiles, workloads, models, providers, or configuration fields.  An injected
projection binds every materialized action to one or more named hypothetical
objective points plus a bounded reliability.  An injected archive-utility port
owns normalization, objective senses, reference points, and exact joint set
value.

This separation lets one sealed proposal market expose several complete policy
arms (central, optimistic, reliability-adjusted, or risk-adjusted) to the
precommitted portfolio race without granting any arm current-outcome access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.candidate_archive_consequence import (
    CandidateArchivePortfolioConsequenceUtilityPort,
    validate_candidate_archive_portfolio_consequence_utility,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
    MaterializedActionDescriptor,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityDecision,
    HardFeasibilityPort,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
    assess_hard_feasibility,
    hard_feasibility_decision_batch_sha256,
    validate_hard_feasibility_port,
)


FORECAST_GEOMETRY_PORTFOLIO_POLICY_ID = "forecast_geometry_portfolio"
FORECAST_GEOMETRY_PORTFOLIO_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_MEMBER_DOMAIN = b"agent-evolve:forecast-geometry-member:v1\x00"
_BATCH_DOMAIN = b"agent-evolve:forecast-geometry-batch:v1\x00"
_POLICY_DOMAIN = b"agent-evolve:forecast-geometry-policy:v1\x00"
_PRIOR_DOMAIN = b"agent-evolve:forecast-geometry-prior:v1\x00"


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


def _projection_identity(
    value: "MaterializedForecastGeometryProjectionPort",
) -> tuple[str, int, str]:
    if not isinstance(value, MaterializedForecastGeometryProjectionPort):
        raise TypeError("projection must implement its application port")
    identity = (
        getattr(value, "projection_id", None),
        getattr(value, "projection_version", None),
        getattr(value, "definition_sha256", None),
    )
    _require_token(identity[0], name="projection_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("projection_version must be positive")
    require_sha256(identity[2], "projection definition_sha256")
    return identity  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ForecastGeometryScenario:
    """One named, raw objective point authored before evaluation."""

    scenario_id: str
    objective_point: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _require_token(self.scenario_id, name="scenario_id")
        if (
            type(self.objective_point) is not tuple
            or not self.objective_point
            or self.objective_point
            != tuple(sorted(self.objective_point))
        ):
            raise ValueError(
                "objective_point must be non-empty and canonical"
            )
        metric_ids: list[str] = []
        for metric_id, value in self.objective_point:
            _require_token(metric_id, name="metric_id")
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(
                    "objective-point values must be finite exact floats"
                )
            metric_ids.append(metric_id)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("objective_point repeats a metric")

    def as_mapping(self) -> dict[str, float]:
        self.__post_init__()
        return dict(self.objective_point)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "scenario_id": self.scenario_id,
            "objective_point": [
                {"metric_id": metric_id, "value_hex": value.hex()}
                for metric_id, value in self.objective_point
            ],
        }


@dataclass(frozen=True, slots=True)
class MaterializedForecastGeometryMember:
    """All pre-evaluation scenario geometry for one materialized action."""

    action_sha256: str
    phenotype_identity_sha256: str
    reliability: float
    scenarios: tuple[ForecastGeometryScenario, ...]
    source_evidence_sha256: str
    member_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        require_sha256(
            self.source_evidence_sha256,
            "source_evidence_sha256",
        )
        if (
            type(self.reliability) is not float
            or not math.isfinite(self.reliability)
            or not 0.0 <= self.reliability <= 1.0
        ):
            raise ValueError("reliability must lie in [0, 1]")
        if (
            type(self.scenarios) is not tuple
            or not self.scenarios
            or any(
                type(value) is not ForecastGeometryScenario
                for value in self.scenarios
            )
        ):
            raise TypeError("scenarios must contain exact scenario values")
        for value in self.scenarios:
            value.__post_init__()
        scenario_ids = tuple(value.scenario_id for value in self.scenarios)
        if scenario_ids != tuple(sorted(set(scenario_ids))):
            raise ValueError("scenario IDs must be unique and canonical")
        metric_frames = {
            tuple(metric_id for metric_id, _value in scenario.objective_point)
            for scenario in self.scenarios
        }
        if len(metric_frames) != 1:
            raise ValueError("all scenarios must use one objective frame")
        object.__setattr__(
            self,
            "member_sha256",
            _hash(_MEMBER_DOMAIN, self._unsigned_record()),
        )

    def scenario(self, scenario_id: str) -> ForecastGeometryScenario:
        _require_token(scenario_id, name="scenario_id")
        matches = tuple(
            value
            for value in self.scenarios
            if value.scenario_id == scenario_id
        )
        if len(matches) != 1:
            raise ValueError("member omits the requested scenario")
        return matches[0]

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "reliability_hex": self.reliability.hex(),
            "scenarios": [value.to_record() for value in self.scenarios],
            "source_evidence_sha256": self.source_evidence_sha256,
            "candidate_outcomes_observed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "member_sha256": self.member_sha256}


@dataclass(frozen=True, slots=True)
class MaterializedForecastGeometryBatch:
    """Authenticated geometry covering one complete sealed proposal market."""

    projection_id: str
    projection_version: int
    projection_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    members: tuple[MaterializedForecastGeometryMember, ...]
    candidate_outcomes_observed: bool
    evidence: FrozenJsonObject
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.projection_id, name="projection_id")
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
            raise ValueError("proposal hashes must be non-empty and canonical")
        for value in self.proposal_sha256s:
            require_sha256(value, "proposal_sha256")
        if (
            type(self.members) is not tuple
            or not self.members
            or any(
                type(value) is not MaterializedForecastGeometryMember
                for value in self.members
            )
        ):
            raise TypeError("members must contain exact forecast geometry")
        for value in self.members:
            value.__post_init__()
        action_ids = tuple(value.action_sha256 for value in self.members)
        if action_ids != tuple(sorted(set(action_ids))):
            raise ValueError("members must be unique and canonical")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be exact")
        if self.candidate_outcomes_observed:
            raise ValueError("forecast geometry cannot observe current outcomes")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("batch evidence must be a frozen object")
        object.__setattr__(
            self,
            "batch_sha256",
            _hash(_BATCH_DOMAIN, self._unsigned_record()),
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
            "member_sha256s": [
                value.member_sha256 for value in self.members
            ],
            "candidate_outcomes_observed": self.candidate_outcomes_observed,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "members": [value.to_record() for value in self.members],
            "batch_sha256": self.batch_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class MaterializedForecastGeometryProjectionPort(Protocol):
    """Project a sealed materialized market into named consequence scenarios."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedForecastGeometryBatch: ...


class ForecastGeometryPortfolioMode(str, Enum):
    SCENARIO = "scenario"
    RELIABILITY_ADJUSTED_SCENARIO = "reliability_adjusted_scenario"
    RISK_ADJUSTED_SCENARIO = "risk_adjusted_scenario"


def _prior_sha256(
    candidates: tuple[EvolutionCandidate, ...],
) -> str:
    return _hash(
        _PRIOR_DOMAIN,
        [
            {
                "candidate_id": value.candidate_id.value,
                "configuration_sha256": value.occurrence.configuration_hash,
                "generation": value.generation,
                "valid": value.valid,
                "operator_compliant": value.operator_compliant,
                "evidence_compliant": value.evidence_compliant,
                "objectives": [
                    {"metric_id": metric_id, "value_hex": number.hex()}
                    for metric_id, number in value.objectives
                ],
            }
            for value in candidates
        ],
    )


@dataclass(frozen=True, slots=True)
class ForecastGeometryPortfolioPolicy:
    """Greedy exact-K joint archive allocation for one typed scenario arm."""

    prior_candidates: tuple[EvolutionCandidate, ...]
    projection: MaterializedForecastGeometryProjectionPort = field(
        repr=False,
        compare=False,
    )
    archive_utility: CandidateArchivePortfolioConsequenceUtilityPort = field(
        repr=False,
        compare=False,
    )
    mode: ForecastGeometryPortfolioMode
    scenario_id: str
    adverse_scenario_id: str | None = None
    risk_aversion: float = 0.0
    max_exact_reliability_members: int = 8
    hard_feasibility: HardFeasibilityPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    policy_id: str = FORECAST_GEOMETRY_PORTFOLIO_POLICY_ID
    policy_version: int = FORECAST_GEOMETRY_PORTFOLIO_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.prior_candidates) is not tuple
            or not self.prior_candidates
            or any(
                type(value) is not EvolutionCandidate
                for value in self.prior_candidates
            )
        ):
            raise TypeError("prior_candidates must be non-empty exact candidates")
        for value in self.prior_candidates:
            value.__post_init__()
        candidate_ids = tuple(
            value.candidate_id for value in self.prior_candidates
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("prior candidate IDs must be unique")
        projection_identity = _projection_identity(self.projection)
        utility_identity = (
            validate_candidate_archive_portfolio_consequence_utility(
                self.archive_utility
            )
        )
        if type(self.mode) is not ForecastGeometryPortfolioMode:
            raise TypeError("mode must be an exact forecast geometry mode")
        _require_token(self.scenario_id, name="scenario_id")
        if self.mode is ForecastGeometryPortfolioMode.RISK_ADJUSTED_SCENARIO:
            if self.adverse_scenario_id is None:
                raise ValueError("risk-adjusted mode requires an adverse scenario")
            _require_token(
                self.adverse_scenario_id,
                name="adverse_scenario_id",
            )
        elif self.adverse_scenario_id is not None:
            raise ValueError("only risk-adjusted mode accepts an adverse scenario")
        if (
            type(self.risk_aversion) is not float
            or not math.isfinite(self.risk_aversion)
            or self.risk_aversion < 0.0
        ):
            raise ValueError("risk_aversion must be finite and non-negative")
        if (
            self.mode is not ForecastGeometryPortfolioMode.RISK_ADJUSTED_SCENARIO
            and self.risk_aversion != 0.0
        ):
            raise ValueError("risk_aversion applies only to risk-adjusted mode")
        if (
            type(self.max_exact_reliability_members) is not int
            or not 1 <= self.max_exact_reliability_members <= 16
        ):
            raise ValueError(
                "max_exact_reliability_members must lie in [1, 16]"
            )
        feasibility_identity = (
            None
            if self.hard_feasibility is None
            else validate_hard_feasibility_port(self.hard_feasibility)
        )
        _require_token(self.policy_id, name="policy_id")
        if self.policy_id != FORECAST_GEOMETRY_PORTFOLIO_POLICY_ID:
            raise ValueError("policy_id is immutable")
        if self.policy_version != FORECAST_GEOMETRY_PORTFOLIO_POLICY_VERSION:
            raise ValueError("policy_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _POLICY_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "projection": {
                        "projection_id": projection_identity[0],
                        "projection_version": projection_identity[1],
                        "definition_sha256": projection_identity[2],
                    },
                    "archive_utility": {
                        "utility_id": utility_identity[0],
                        "utility_version": utility_identity[1],
                        "definition_sha256": utility_identity[2],
                    },
                    "mode": self.mode.value,
                    "scenario_id": self.scenario_id,
                    "adverse_scenario_id": self.adverse_scenario_id,
                    "risk_aversion_hex": self.risk_aversion.hex(),
                    "max_exact_reliability_members": (
                        self.max_exact_reliability_members
                    ),
                    "allocation": (
                        "greedy_joint_marginal_archive_utility_exact_k"
                    ),
                    "reliability": (
                        "exact_independent_bernoulli_set_expectation"
                    ),
                    "hard_feasibility": (
                        None
                        if feasibility_identity is None
                        else {
                            "policy_id": feasibility_identity[0],
                            "policy_version": feasibility_identity[1],
                            "definition_sha256": feasibility_identity[2],
                        }
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def _joint_gain(
        self,
        members: tuple[MaterializedForecastGeometryMember, ...],
        scenario_id: str,
        cache: dict[tuple[str, tuple[str, ...]], float],
    ) -> float:
        key = (
            scenario_id,
            tuple(sorted(value.action_sha256 for value in members)),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = self.archive_utility.portfolio_marginal_utility(
            self.prior_candidates,
            tuple(
                value.scenario(scenario_id).as_mapping()
                for value in members
            ),
        )
        if type(result) is not float or not math.isfinite(result) or result < 0.0:
            raise ValueError(
                "joint archive utility must be finite and non-negative"
            )
        cache[key] = result
        return result

    def _portfolio_value(
        self,
        members: tuple[MaterializedForecastGeometryMember, ...],
        cache: dict[tuple[str, tuple[str, ...]], float],
    ) -> float:
        if self.mode is ForecastGeometryPortfolioMode.SCENARIO:
            return self._joint_gain(members, self.scenario_id, cache)
        if (
            self.mode
            is ForecastGeometryPortfolioMode.RISK_ADJUSTED_SCENARIO
        ):
            central = self._joint_gain(
                members,
                self.scenario_id,
                cache,
            )
            assert self.adverse_scenario_id is not None
            adverse = self._joint_gain(
                members,
                self.adverse_scenario_id,
                cache,
            )
            return central - self.risk_aversion * max(
                0.0,
                central - adverse,
            )
        if len(members) > self.max_exact_reliability_members:
            raise ValueError(
                "reliability-adjusted portfolio exceeds exact integration limit"
            )
        expected = 0.0
        for mask in range(1 << len(members)):
            probability = 1.0
            admitted: list[MaterializedForecastGeometryMember] = []
            for index, member in enumerate(members):
                if mask & (1 << index):
                    probability *= member.reliability
                    admitted.append(member)
                else:
                    probability *= 1.0 - member.reliability
            if probability:
                expected += probability * self._joint_gain(
                    tuple(admitted),
                    self.scenario_id,
                    cache,
                )
        if not math.isfinite(expected) or expected < 0.0:
            raise RuntimeError("expected joint archive utility became invalid")
        return expected

    async def require(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionAllocationRequirement:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if any(
            value.generation >= request.decision_index
            for value in self.prior_candidates
        ):
            raise ValueError("prior candidates cross the current cutoff")
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        actions: list[MaterializedActionDescriptor] = []
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
            actions.extend(proposal.actions)
        action_by_sha256 = {
            value.action_sha256: value for value in actions
        }
        if len(action_by_sha256) != len(actions):
            raise ValueError("proposal market repeats an action identity")
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        batch = await self.projection.project(request, proposals)
        if type(batch) is not MaterializedForecastGeometryBatch:
            raise TypeError("projection returned a foreign geometry batch")
        batch.__post_init__()
        if (
            (
                batch.projection_id,
                batch.projection_version,
                batch.projection_definition_sha256,
            )
            != _projection_identity(self.projection)
            or batch.residual_request_sha256 != request.request_sha256
            or batch.proposal_sha256s != proposal_sha256s
            or tuple(value.action_sha256 for value in batch.members)
            != tuple(sorted(action_by_sha256))
        ):
            raise ValueError("forecast geometry differs from the sealed market")
        member_by_action = {
            value.action_sha256: value for value in batch.members
        }
        if any(
            member_by_action[action_id].phenotype_identity_sha256
            != action.phenotype_identity_sha256
            for action_id, action in action_by_sha256.items()
        ):
            raise ValueError("forecast geometry changes phenotype identity")
        required_scenarios = {
            self.scenario_id,
            *(
                ()
                if self.adverse_scenario_id is None
                else (self.adverse_scenario_id,)
            ),
        }
        if any(
            not required_scenarios.issubset(
                {scenario.scenario_id for scenario in member.scenarios}
            )
            for member in batch.members
        ):
            raise ValueError("forecast geometry omits a policy scenario")

        infeasible: set[str] = set()
        feasibility_decisions: list[
            tuple[str, HardFeasibilityDecision]
        ] = []
        if self.hard_feasibility is not None:
            for action_sha256 in sorted(action_by_sha256):
                decision = assess_hard_feasibility(
                    self.hard_feasibility,
                    HardFeasibilityRequest(
                        campaign_scope_sha256=request.campaign_scope_sha256,
                        cutoff_index=request.decision_index,
                        configuration=action_by_sha256[
                            action_sha256
                        ].configuration,
                    ),
                )
                feasibility_decisions.append((action_sha256, decision))
                if decision.verdict is HardFeasibilityVerdict.INFEASIBLE:
                    infeasible.add(action_sha256)
        admissible = tuple(
            sorted(
                (
                    value
                    for value in batch.members
                    if value.action_sha256 not in infeasible
                ),
                key=lambda value: value.action_sha256,
            )
        )
        if len(
            {
                value.phenotype_identity_sha256
                for value in admissible
            }
        ) < request.evaluation_slots:
            raise ValueError("forecast market cannot fill unique-phenotype K")
        if (
            self.mode
            is ForecastGeometryPortfolioMode.RELIABILITY_ADJUSTED_SCENARIO
            and request.evaluation_slots > self.max_exact_reliability_members
        ):
            raise ValueError("evaluation K exceeds exact reliability limit")

        selected: list[MaterializedForecastGeometryMember] = []
        phenotypes: set[str] = set()
        trace: list[dict[str, object]] = []
        cache: dict[tuple[str, tuple[str, ...]], float] = {}
        candidate_evaluations = 0
        previous = self._portfolio_value((), cache)
        for ordinal in range(1, request.evaluation_slots + 1):
            candidates = tuple(
                value
                for value in admissible
                if value.phenotype_identity_sha256 not in phenotypes
            )
            scored = []
            for candidate in candidates:
                value = self._portfolio_value(
                    tuple((*selected, candidate)),
                    cache,
                )
                candidate_evaluations += 1
                scored.append((value, candidate))
            if not scored:
                raise RuntimeError("forecast geometry cannot close exact K")
            value, winner = min(
                scored,
                key=lambda item: (
                    -item[0],
                    action_by_sha256[item[1].action_sha256].native_rank,
                    item[1].action_sha256,
                ),
            )
            selected.append(winner)
            phenotypes.add(winner.phenotype_identity_sha256)
            trace.append(
                {
                    "ordinal": ordinal,
                    "allocation_kind": "forecast_geometry_joint_greedy",
                    "score_lane": (
                        f"forecast_geometry.{self.mode.value}."
                        f"{self.scenario_id}"
                    ),
                    "action_sha256": winner.action_sha256,
                    "portfolio_value_before_hex": previous.hex(),
                    "portfolio_value_after_hex": value.hex(),
                    "marginal_portfolio_value_hex": (value - previous).hex(),
                    "member_reliability_hex": winner.reliability.hex(),
                    "candidate_outcomes_observed": False,
                }
            )
            previous = value

        feasibility_record = {
            "enabled": self.hard_feasibility is not None,
            "decision_batch_sha256": (
                None
                if not feasibility_decisions
                else hard_feasibility_decision_batch_sha256(
                    tuple(
                        value.decision_sha256
                        for _action_sha256, value in feasibility_decisions
                    )
                )
            ),
            "rejected_action_sha256s": sorted(infeasible),
            "unknown_actions_remain_eligible": True,
            "candidate_outcomes_observed": False,
        }
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            required_action_sha256s=tuple(
                sorted(value.action_sha256 for value in selected)
            ),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "forecast_geometry_batch_sha256": batch.batch_sha256,
                    "prior_candidate_set_sha256": _prior_sha256(
                        self.prior_candidates
                    ),
                    "mode": self.mode.value,
                    "scenario_id": self.scenario_id,
                    "adverse_scenario_id": self.adverse_scenario_id,
                    "risk_aversion_hex": self.risk_aversion.hex(),
                    "final_portfolio_value_hex": previous.hex(),
                    "candidate_portfolio_evaluations": candidate_evaluations,
                    "selection_trace": trace,
                    "hard_feasibility": feasibility_record,
                    "joint_set_value_not_member_score_sum": True,
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "FORECAST_GEOMETRY_PORTFOLIO_POLICY_ID",
    "FORECAST_GEOMETRY_PORTFOLIO_POLICY_VERSION",
    "ForecastGeometryPortfolioMode",
    "ForecastGeometryPortfolioPolicy",
    "ForecastGeometryScenario",
    "MaterializedForecastGeometryBatch",
    "MaterializedForecastGeometryMember",
    "MaterializedForecastGeometryProjectionPort",
]

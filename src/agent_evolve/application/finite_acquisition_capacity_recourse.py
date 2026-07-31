"""Capacity-complete recourse backed by a generic finite acquisition expert."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
)
from agent_evolve.application.campaign_capacity_recourse import (
    CampaignCapacityRecourseRequest,
    CampaignCapacityRecourseResult,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
    PhenotypeIdentityPolicy,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
    FiniteAcquisitionDecision,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
    FiniteAcquisitionPolicy,
    FiniteAcquisitionRequest,
)
from agent_evolve.ports.finite_acquisition_space import (
    FiniteAcquisitionSpace,
    FiniteAcquisitionSpaceRequest,
    validate_finite_acquisition_space_candidates,
    validate_finite_acquisition_space_identity,
)
from agent_evolve.ports.hard_feasibility import (
    HardFeasibilityPort,
    HardFeasibilityRequest,
    HardFeasibilityVerdict,
    assess_hard_feasibility,
    hard_feasibility_decision_batch_sha256,
    validate_hard_feasibility_port,
)


POLICY_ID = "finite_acquisition_capacity_recourse"
POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DEFINITION_DOMAIN = b"agent-evolve:finite-acquisition-capacity-recourse:def:v1\x00"
_PROPOSAL_MEMBER_DOMAIN = (
    b"agent-evolve:finite-acquisition-capacity-proposal-member:v1\x00"
)
_PROPOSAL_DOMAIN = b"agent-evolve:finite-acquisition-capacity-proposal:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _acquisition_identity(
    policy: FiniteAcquisitionPolicy,
) -> tuple[str, int, str]:
    if not isinstance(policy, FiniteAcquisitionPolicy):
        raise TypeError("acquisition must implement FiniteAcquisitionPolicy")
    values = (
        getattr(policy, "policy_id", None),
        getattr(policy, "policy_version", None),
        getattr(policy, "definition_sha256", None),
    )
    if type(values[0]) is not str or _TOKEN.fullmatch(values[0]) is None:
        raise ValueError("acquisition policy_id has invalid syntax")
    if type(values[1]) is not int or values[1] <= 0:
        raise ValueError("acquisition policy_version must be positive")
    require_sha256(values[2], "acquisition definition_sha256")
    return values  # type: ignore[return-value]


def _phenotype(
    policy: PhenotypeIdentityPolicy,
    configuration: FrozenJsonObject,
) -> str:
    identity = policy.identify(thaw_json(configuration))
    if type(identity) is not PhenotypeIdentity:
        raise TypeError("phenotype policy must return exact PhenotypeIdentity")
    PhenotypeIdentity.__post_init__(identity)
    if (
        identity.policy_id != policy.policy_id
        or identity.policy_version != policy.policy_version
    ):
        raise ValueError("phenotype policy returned a foreign identity")
    return identity.value_sha256


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionCapacityProposalMember:
    """One unevaluated, fully materialized acquisition nomination."""

    target_candidate_id: CandidateId
    acquisition_candidate_id: str
    configuration: FrozenJsonObject
    phenotype_identity_sha256: str
    native_rank: int
    member_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.target_candidate_id) is not CandidateId:
            raise TypeError("target_candidate_id must be exact")
        CandidateId.__post_init__(self.target_candidate_id)
        if (
            type(self.acquisition_candidate_id) is not str
            or not self.acquisition_candidate_id
        ):
            raise ValueError("acquisition_candidate_id must be non-empty")
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be an exact frozen object")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        if type(self.native_rank) is not int or self.native_rank <= 0:
            raise ValueError("native_rank must be positive")
        computed = hashlib.sha256(
            _PROPOSAL_MEMBER_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.member_sha256 not in ("", computed):
            raise ValueError("member_sha256 does not authenticate the proposal")
        object.__setattr__(self, "member_sha256", computed)

    @property
    def configuration_sha256(self) -> str:
        return typed_json_sha256(self.configuration)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "target_candidate_id": self.target_candidate_id.value,
            "acquisition_candidate_id": self.acquisition_candidate_id,
            "configuration_sha256": self.configuration_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "native_rank": self.native_rank,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "member_sha256": self.member_sha256}


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionCapacityProposal:
    """Replayable pre-evaluation acquisition transaction for a missing slate."""

    request_sha256: str
    generation: int
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    decision: FiniteAcquisitionDecision
    members: tuple[FiniteAcquisitionCapacityProposalMember, ...]
    evidence: FrozenJsonObject
    proposal_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id has invalid syntax")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.decision) is not FiniteAcquisitionDecision:
            raise TypeError("decision must be exact")
        self.decision.__post_init__()
        if type(self.members) is not tuple or not self.members:
            raise ValueError("members must be a non-empty exact tuple")
        for member in self.members:
            if type(member) is not FiniteAcquisitionCapacityProposalMember:
                raise TypeError("members must contain exact proposal members")
            member.__post_init__()
        if tuple(value.native_rank for value in self.members) != tuple(
            range(1, len(self.members) + 1)
        ):
            raise ValueError("proposal native ranks must be contiguous")
        if tuple(
            (value.acquisition_candidate_id, value.configuration_sha256)
            for value in self.members
        ) != tuple(
            (value.candidate_id, value.configuration_sha256)
            for value in self.decision.selected
        ):
            raise ValueError("proposal members differ from the acquisition decision")
        if len({value.target_candidate_id for value in self.members}) != len(
            self.members
        ):
            raise ValueError("proposal target candidate IDs must be unique")
        if len({value.phenotype_identity_sha256 for value in self.members}) != len(
            self.members
        ):
            raise ValueError("proposal member phenotypes must be unique")
        if type(self.evidence) is not FrozenJsonObject:
            raise TypeError("evidence must be an exact frozen object")
        computed = hashlib.sha256(
            _PROPOSAL_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.proposal_sha256 not in ("", computed):
            raise ValueError("proposal_sha256 does not authenticate the proposal")
        object.__setattr__(self, "proposal_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "generation": self.generation,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "decision": self.decision.to_record(),
            "members": [value.to_record() for value in self.members],
            "evidence": thaw_json(self.evidence),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "proposal_sha256": self.proposal_sha256}


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionCapacityRecourse:
    """Refit on the current real archive and fill every missing occurrence."""

    objectives: tuple[FiniteAcquisitionObjective, ...]
    space: FiniteAcquisitionSpace
    acquisition: FiniteAcquisitionPolicy
    phenotype_identity: PhenotypeIdentityPolicy
    engine: AgenticEvolutionEngine
    hard_feasibility: HardFeasibilityPort | None = None
    pool_size: int = 8192
    seed: int = 0
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.objectives) is not tuple or len(self.objectives) < 2:
            raise ValueError("objectives must contain at least two exact axes")
        for value in self.objectives:
            if type(value) is not FiniteAcquisitionObjective:
                raise TypeError("objectives must contain exact acquisition axes")
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.objectives)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("objective axes must be unique and metric-sorted")
        space_identity = validate_finite_acquisition_space_identity(self.space)
        acquisition_identity = _acquisition_identity(self.acquisition)
        if not isinstance(self.engine, AgenticEvolutionEngine):
            raise TypeError("engine must be an AgenticEvolutionEngine")
        if not callable(getattr(self.phenotype_identity, "identify", None)):
            raise TypeError("phenotype_identity must implement identify")
        phenotype_identity = (
            getattr(self.phenotype_identity, "policy_id", None),
            getattr(self.phenotype_identity, "policy_version", None),
        )
        PhenotypeIdentity(
            policy_id=phenotype_identity[0],
            policy_version=phenotype_identity[1],
            value_sha256="0" * 64,
        )
        hard_feasibility_identity = (
            None
            if self.hard_feasibility is None
            else validate_hard_feasibility_port(self.hard_feasibility)
        )
        if type(self.pool_size) is not int or self.pool_size < 16:
            raise ValueError("pool_size must be at least 16")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be non-negative")
        definition = {
            "schema_version": 1,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "objectives": [value.to_record() for value in self.objectives],
            "space": {
                "space_id": space_identity[0],
                "space_version": space_identity[1],
                "definition_sha256": space_identity[2],
            },
            "acquisition": {
                "policy_id": acquisition_identity[0],
                "policy_version": acquisition_identity[1],
                "definition_sha256": acquisition_identity[2],
            },
            "phenotype_identity": {
                "policy_id": phenotype_identity[0],
                "policy_version": phenotype_identity[1],
            },
            "hard_feasibility": (
                None
                if hard_feasibility_identity is None
                else {
                    "policy_id": hard_feasibility_identity[0],
                    "policy_version": hard_feasibility_identity[1],
                    "definition_sha256": hard_feasibility_identity[2],
                    "rejection_authority": "exact_infeasibility_only",
                }
            ),
            "pool_size": self.pool_size,
            "seed": self.seed,
            "state_binding": "strictly_prior_real_outcomes",
            "capacity_contract": "exact_missing_occurrences",
        }
        computed = hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(definition)
        ).hexdigest()
        if self.definition_sha256 not in ("", computed):
            raise ValueError("definition_sha256 does not authenticate the policy")
        object.__setattr__(self, "definition_sha256", computed)

    def _observations(
        self,
        request: CampaignCapacityRecourseRequest,
    ) -> tuple[FiniteAcquisitionObservation, ...]:
        metric_ids = tuple(value.metric_id for value in self.objectives)
        by_configuration: dict[str, FiniteAcquisitionObservation] = {}
        for candidate in request.state.candidates:
            if not candidate.valid or type(candidate.configuration) is not FrozenJsonObject:
                continue
            objective_map = candidate.objective_map
            if any(metric_id not in objective_map for metric_id in metric_ids):
                raise ValueError("real observation omits a configured objective axis")
            configuration_sha256 = candidate.occurrence.configuration_hash
            observation = FiniteAcquisitionObservation(
                candidate_id=f"obs-{configuration_sha256[:32]}",
                configuration_sha256=configuration_sha256,
                features=self.space.features(candidate.configuration),
                objectives=tuple(
                    (metric_id, objective_map[metric_id]) for metric_id in metric_ids
                ),
            )
            previous = by_configuration.get(configuration_sha256)
            if previous is not None and previous != observation:
                raise ValueError("one configuration has conflicting real observations")
            by_configuration[configuration_sha256] = observation
        observations = tuple(
            by_configuration[key] for key in sorted(by_configuration)
        )
        if not observations:
            raise ValueError("capacity recourse requires prior successful observations")
        return observations

    def propose(
        self,
        request: CampaignCapacityRecourseRequest,
    ) -> FiniteAcquisitionCapacityProposal:
        """Materialize a fresh ranked slate without evaluating any member."""
        self.__post_init__()
        if type(request) is not CampaignCapacityRecourseRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        needed = request.missing_candidate_occurrences
        observations = self._observations(request)

        configurations_by_sha256: dict[str, FrozenJsonObject] = {}
        for candidate in request.state.candidates:
            if type(candidate.configuration) is FrozenJsonObject:
                configurations_by_sha256[
                    candidate.occurrence.configuration_hash
                ] = candidate.configuration
        known_configuration_sha256s = tuple(sorted(configurations_by_sha256))
        space_request = FiniteAcquisitionSpaceRequest(
            campaign_scope_sha256=request.campaign_scope_sha256,
            cutoff_index=request.state.unique_evaluations,
            pool_size=self.pool_size,
            seed=self.seed + request.generation,
            observed_configurations=tuple(
                configurations_by_sha256[value]
                for value in known_configuration_sha256s
            ),
            excluded_configuration_sha256s=known_configuration_sha256s,
        )
        materializations = self.space.candidates(space_request)
        validate_finite_acquisition_space_candidates(
            request=space_request,
            candidates=materializations,
        )

        known_phenotypes = {
            _phenotype(self.phenotype_identity, value)
            for value in configurations_by_sha256.values()
        }
        accepted_phenotypes = set(known_phenotypes)
        eligible_configurations: dict[str, FrozenJsonObject] = {}
        acquisition_candidates: list[FiniteAcquisitionCandidate] = []
        seen_features: set[tuple[float, ...]] = set()
        phenotype_rejections: list[dict[str, str]] = []
        feasibility_infeasible_sha256s: list[str] = []
        feasibility_rejection_witnesses: list[dict[str, object]] = []
        feasibility_decision_sha256s: list[str] = []
        feasibility_verdict_counts = {
            HardFeasibilityVerdict.FEASIBLE.value: 0,
            HardFeasibilityVerdict.INFEASIBLE.value: 0,
            HardFeasibilityVerdict.UNKNOWN.value: 0,
        }
        for configuration in materializations:
            configuration_sha256 = typed_json_sha256(configuration)
            if self.hard_feasibility is not None:
                feasibility = assess_hard_feasibility(
                    self.hard_feasibility,
                    HardFeasibilityRequest(
                        campaign_scope_sha256=request.campaign_scope_sha256,
                        cutoff_index=request.state.unique_evaluations,
                        configuration=configuration,
                    ),
                )
                feasibility_decision_sha256s.append(feasibility.decision_sha256)
                feasibility_verdict_counts[feasibility.verdict.value] += 1
                if feasibility.verdict is HardFeasibilityVerdict.INFEASIBLE:
                    feasibility_infeasible_sha256s.append(
                        feasibility.decision_sha256
                    )
                    if len(feasibility_rejection_witnesses) < 8:
                        feasibility_rejection_witnesses.append(
                            {
                                "configuration_sha256": configuration_sha256,
                                "decision": feasibility.to_record(),
                            }
                        )
                    continue
            phenotype_sha256 = _phenotype(self.phenotype_identity, configuration)
            if phenotype_sha256 in accepted_phenotypes:
                phenotype_rejections.append(
                    {
                        "configuration_sha256": configuration_sha256,
                        "phenotype_sha256": phenotype_sha256,
                    }
                )
                continue
            features = self.space.features(configuration)
            if features in seen_features:
                continue
            candidate = FiniteAcquisitionCandidate(
                candidate_id=f"capacity-{configuration_sha256[:32]}",
                configuration_sha256=configuration_sha256,
                features=features,
            )
            acquisition_candidates.append(candidate)
            eligible_configurations[configuration_sha256] = configuration
            seen_features.add(features)
            accepted_phenotypes.add(phenotype_sha256)
        if len(acquisition_candidates) < needed:
            raise ValueError("screened acquisition reservoir cannot fill capacity")

        acquisition_request = FiniteAcquisitionRequest(
            campaign_scope_sha256=request.campaign_scope_sha256,
            cutoff_index=request.state.unique_evaluations,
            batch_size=needed,
            seed=self.seed + request.generation,
            objectives=self.objectives,
            observations=observations,
            candidates=tuple(acquisition_candidates),
        )
        decision = self.acquisition.select(acquisition_request)
        if type(decision) is not FiniteAcquisitionDecision:
            raise TypeError("acquisition returned a foreign decision")
        decision.__post_init__()
        if decision.request_sha256 != acquisition_request.request_sha256:
            raise ValueError("acquisition decision targets another request")
        if (
            decision.policy_id,
            decision.policy_version,
            decision.policy_definition_sha256,
        ) != _acquisition_identity(self.acquisition):
            raise ValueError("acquisition decision has a foreign policy identity")
        if len(decision.selected) != needed:
            raise ValueError("acquisition decision did not fill missing capacity")
        candidate_by_id = {
            value.candidate_id: value for value in acquisition_candidates
        }
        selected_configurations: list[FrozenJsonObject] = []
        for selected in decision.selected:
            candidate = candidate_by_id.get(selected.candidate_id)
            if (
                candidate is None
                or candidate.configuration_sha256
                != selected.configuration_sha256
            ):
                raise ValueError("acquisition selected outside the sealed reservoir")
            selected_configurations.append(
                eligible_configurations[selected.configuration_sha256]
            )

        members = tuple(
            FiniteAcquisitionCapacityProposalMember(
                target_candidate_id=self.engine.ids.new_candidate_id(),
                acquisition_candidate_id=selected.candidate_id,
                configuration=configuration,
                phenotype_identity_sha256=_phenotype(
                    self.phenotype_identity,
                    configuration,
                ),
                native_rank=rank,
            )
            for rank, (selected, configuration) in enumerate(
                zip(decision.selected, selected_configurations, strict=True),
                start=1,
            )
        )
        evidence = freeze_json(
            {
                "schema_version": 1,
                "space_request": space_request.to_record(),
                "acquisition_request_sha256": acquisition_request.request_sha256,
                "acquisition_decision": decision.to_record(),
                "raw_reservoir_count": len(materializations),
                "eligible_reservoir_count": len(acquisition_candidates),
                "strictly_prior_observation_count": len(observations),
                "hard_feasibility": {
                    "enabled": self.hard_feasibility is not None,
                    "verdict_counts": feasibility_verdict_counts,
                    "decision_count": len(feasibility_decision_sha256s),
                    "decision_batch_sha256": (
                        hard_feasibility_decision_batch_sha256(
                            feasibility_decision_sha256s
                        )
                    ),
                    "infeasible_decision_count": len(
                        feasibility_infeasible_sha256s
                    ),
                    "infeasible_decision_batch_sha256": (
                        hard_feasibility_decision_batch_sha256(
                            feasibility_infeasible_sha256s
                        )
                    ),
                    "rejection_witnesses": feasibility_rejection_witnesses,
                    "rejection_witness_limit": 8,
                    "rejection_witness_sampling": (
                        "first_in_authenticated_space_order"
                    ),
                    "rejection_authority": "exact_infeasibility_only",
                    "unknown_disposition": "retain",
                    "outcome_access": False,
                },
                "phenotype_rejections": phenotype_rejections,
                "selected_configuration_sha256s": [
                    value.configuration_sha256 for value in decision.selected
                ],
                "current_or_future_outcomes_consulted": False,
                "capacity_requested": needed,
                "evaluation_deferred": True,
                "workload_model_provider_branches": False,
            }
        )
        if type(evidence) is not FrozenJsonObject:  # pragma: no cover
            raise AssertionError("capacity recourse evidence is not an object")
        return FiniteAcquisitionCapacityProposal(
            request_sha256=request.request_sha256,
            generation=request.generation,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            decision=decision,
            members=members,
            evidence=evidence,
        )

    async def evaluate_members(
        self,
        proposal: FiniteAcquisitionCapacityProposal,
        target_candidate_ids: tuple[CandidateId, ...],
    ) -> tuple[EvolutionCandidate, ...]:
        """Evaluate a broker-selected subset of one sealed proposal."""

        self.__post_init__()
        if type(proposal) is not FiniteAcquisitionCapacityProposal:
            raise TypeError("proposal must be exact")
        proposal.__post_init__()
        if (
            proposal.policy_id,
            proposal.policy_version,
            proposal.policy_definition_sha256,
        ) != (self.policy_id, self.policy_version, self.definition_sha256):
            raise ValueError("proposal is bound to another recourse policy")
        if type(target_candidate_ids) is not tuple or not target_candidate_ids:
            raise ValueError("target_candidate_ids must be a non-empty exact tuple")
        if len(set(target_candidate_ids)) != len(target_candidate_ids):
            raise ValueError("target_candidate_ids must be unique")
        by_id = {value.target_candidate_id: value for value in proposal.members}
        try:
            selected = tuple(by_id[value] for value in target_candidate_ids)
        except KeyError as error:
            raise ValueError("broker selected outside the sealed proposal") from error
        if tuple(sorted(selected, key=lambda value: value.native_rank)) != selected:
            raise ValueError("selected proposal members must preserve native order")
        candidates = await asyncio.gather(
            *(
                self.engine.register_materialized_candidate(
                    thaw_json(member.configuration),
                    label=(
                        f"campaign_g{proposal.generation:02d}_capacity_recourse_"
                        f"r{member.native_rank:03d}"
                    ),
                    generation=proposal.generation,
                    proposal_source_id=self.policy_id,
                    proposal_source_version=self.policy_version,
                    proposal_source_definition_sha256=self.definition_sha256,
                    proposal_decision_sha256=proposal.decision.decision_sha256,
                    proposal_rank=member.native_rank,
                    candidate_id=member.target_candidate_id,
                )
                for member in selected
            )
        )
        return tuple(candidates)

    async def fill(
        self,
        request: CampaignCapacityRecourseRequest,
    ) -> CampaignCapacityRecourseResult:
        """Preserve the capacity-port behavior through propose/evaluate phases."""

        proposal = self.propose(request)
        candidates = await self.evaluate_members(
            proposal,
            tuple(value.target_candidate_id for value in proposal.members),
        )
        evidence_record = thaw_json(proposal.evidence)
        if type(evidence_record) is not dict:  # pragma: no cover
            raise AssertionError("capacity proposal evidence is not an object")
        evidence_record.update(
            {
                "proposal_sha256": proposal.proposal_sha256,
                "selected_candidate_ids": [
                    value.candidate_id.value for value in candidates
                ],
                "capacity_realized": len(candidates),
                "evaluation_deferred": False,
            }
        )
        evidence = freeze_json(evidence_record)
        if type(evidence) is not FrozenJsonObject:  # pragma: no cover
            raise AssertionError("capacity recourse evidence is not an object")
        return CampaignCapacityRecourseResult(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            candidates=tuple(candidates),
            evidence=evidence,
        )


__all__ = [
    "FiniteAcquisitionCapacityProposal",
    "FiniteAcquisitionCapacityProposalMember",
    "FiniteAcquisitionCapacityRecourse",
    "POLICY_ID",
    "POLICY_VERSION",
]

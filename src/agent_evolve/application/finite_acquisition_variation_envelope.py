"""Protected numerical-acquisition expert for the campaign variation envelope.

The service is workload-neutral.  A finite acquisition-space adapter owns legal
configuration enumeration and normalized feature encoding; a numerical policy
owns ask selection; a phenotype policy owns evaluator-equivalence identity.
The service joins those ports to strictly prior real outcomes and adds the
selected full configurations to the ordinary finite-option universe.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.campaign_variation_envelope import (
    CampaignVariationEnvelopeLaneResult,
    CampaignVariationEnvelopeRequest,
    CampaignVariationEnvelopeResult,
)
from agent_evolve.application.evolution_campaign import ParentVariationBinding
from agent_evolve.application.finite_variation_eligibility import (
    OptionPhenotypeBinding,
    eligible_finite_variation_view,
)
from agent_evolve.domain.finite_variation import (
    MAX_FINITE_VARIATION_OPTIONS,
    FiniteVariationContract,
    FiniteVariationOption,
)
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
from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContext,
    AcquisitionCertifiedSlateContextSink,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
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
from agent_evolve.ports.variation_source import (
    VARIATION_CANDIDATE_POOL_REQUIREMENT_METADATA_KEY,
    VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY,
    VARIATION_OPERATOR_METADATA_KEY,
    VARIATION_SOURCE_METADATA_KEY,
    VARIATION_SOURCE_MINIMUM_METADATA_KEY,
    VARIATION_SOURCE_RANK_METADATA_KEY,
    finite_variation_source_id,
    finite_variation_source_ids,
)


POLICY_ID = "protected_finite_acquisition_envelope"
POLICY_VERSION = 5
PROTECTED_ACQUISITION_SOURCE_ID = "numerical_acquisition"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DEFINITION_DOMAIN = b"agent-evolve:protected-finite-acquisition-envelope:def:v5\x00"
_CATALOG_DOMAIN = b"agent-evolve:protected-finite-acquisition-envelope:catalog:v5\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _policy_identity(policy: FiniteAcquisitionPolicy) -> tuple[str, int, str]:
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
    if (
        type(values[2]) is not str
        or len(values[2]) != 64
        or any(value not in "0123456789abcdef" for value in values[2])
    ):
        raise ValueError("acquisition definition must be a lowercase SHA-256")
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
class ProtectedFiniteAcquisitionVariationEnvelope:
    """Add one globally selected, evaluator-protected acquisition batch."""

    objectives: tuple[FiniteAcquisitionObjective, ...]
    space: FiniteAcquisitionSpace
    acquisition: FiniteAcquisitionPolicy
    phenotype_identity: PhenotypeIdentityPolicy
    hard_feasibility: HardFeasibilityPort | None = None
    acquisition_certification_context_sink: (
        AcquisitionCertifiedSlateContextSink | None
    ) = None
    pool_size: int = 8192
    protected_batch_size: int = 4
    source_minimum_per_lane: int = 1
    maximum_phenotype_recourse_rounds: int = 3
    seed: int = 0
    source_id: str = PROTECTED_ACQUISITION_SOURCE_ID
    option_family: str = "acquisition"
    operator_id: str = "global"
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")
    _catalog_definition_sha256: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        if type(self.objectives) is not tuple or len(self.objectives) < 2:
            raise ValueError("objectives must contain at least two exact axes")
        for value in self.objectives:
            if type(value) is not FiniteAcquisitionObjective:
                raise TypeError("objectives must contain exact acquisition axes")
            FiniteAcquisitionObjective.__post_init__(value)
        metric_ids = tuple(value.metric_id for value in self.objectives)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("objectives must be unique and metric-sorted")
        space_identity = validate_finite_acquisition_space_identity(self.space)
        acquisition_identity = _policy_identity(self.acquisition)
        hard_feasibility_identity = (
            None
            if self.hard_feasibility is None
            else validate_hard_feasibility_port(self.hard_feasibility)
        )
        if self.acquisition_certification_context_sink is not None and not isinstance(
            self.acquisition_certification_context_sink,
            AcquisitionCertifiedSlateContextSink,
        ):
            raise TypeError(
                "acquisition_certification_context_sink must implement its port"
            )
        if not callable(getattr(self.phenotype_identity, "identify", None)):
            raise TypeError("phenotype_identity must implement its policy port")
        if (
            type(self.phenotype_identity.policy_id) is not str
            or _TOKEN.fullmatch(self.phenotype_identity.policy_id) is None
            or type(self.phenotype_identity.policy_version) is not int
            or self.phenotype_identity.policy_version <= 0
        ):
            raise ValueError("phenotype identity policy has invalid identity")
        if type(self.pool_size) is not int or self.pool_size < 16:
            raise ValueError("pool_size must be an exact integer of at least 16")
        if (
            type(self.protected_batch_size) is not int
            or self.protected_batch_size < 1
        ):
            raise ValueError("protected_batch_size must be positive")
        if self.protected_batch_size > self.pool_size:
            raise ValueError("protected batch cannot exceed the pool")
        if (
            type(self.source_minimum_per_lane) is not int
            or not 1 <= self.source_minimum_per_lane < 8
        ):
            raise ValueError("source_minimum_per_lane must lie in [1, 8)")
        if (
            type(self.maximum_phenotype_recourse_rounds) is not int
            or self.maximum_phenotype_recourse_rounds < 1
        ):
            raise ValueError("maximum phenotype recourse rounds must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative exact integer")
        for name in ("source_id", "option_family", "operator_id"):
            value = getattr(self, name)
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError(f"{name} must use the closed token grammar")
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
            "hard_feasibility": (
                None
                if hard_feasibility_identity is None
                else {
                    "policy_id": hard_feasibility_identity[0],
                    "policy_version": hard_feasibility_identity[1],
                    "definition_sha256": hard_feasibility_identity[2],
                    "rejection_authority": "exact_infeasibility_only",
                    "unknown_disposition": "retain",
                }
            ),
            "acquisition_certification_context_sink": (
                None
                if self.acquisition_certification_context_sink is None
                else {
                    "provider_id": (
                        self.acquisition_certification_context_sink.provider_id
                    ),
                    "provider_version": (
                        self.acquisition_certification_context_sink.provider_version
                    ),
                    "definition_sha256": (
                        self.acquisition_certification_context_sink.definition_sha256
                    ),
                }
            ),
            "phenotype_identity": {
                "policy_id": self.phenotype_identity.policy_id,
                "policy_version": self.phenotype_identity.policy_version,
            },
            "pool_size": self.pool_size,
            "protected_batch_size": self.protected_batch_size,
            "source_minimum_per_lane": self.source_minimum_per_lane,
            "maximum_phenotype_recourse_rounds": (
                self.maximum_phenotype_recourse_rounds
            ),
            "seed": self.seed,
            "source_id": self.source_id,
            "option_family": self.option_family,
            "operator_id": self.operator_id,
            "archive_outcomes": "strictly_prior_real_valid_unique_configurations",
            "partition": "accepted_acquisition_rank_round_robin_over_sorted_lanes",
            "protected_source_witness": (
                "smallest_authenticated_source_ranks_within_each_lane"
            ),
            "phenotype_recourse": (
                "select_then-exact-check-selected-only-bounded-refill"
            ),
            "base_support": "preserved",
            "current_future_outcomes": False,
            "workload_model_provider_branches": False,
        }
        definition_sha256 = hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(definition)
        ).hexdigest()
        object.__setattr__(self, "definition_sha256", definition_sha256)
        object.__setattr__(
            self,
            "_catalog_definition_sha256",
            hashlib.sha256(
                _CATALOG_DOMAIN + bytes.fromhex(definition_sha256)
            ).hexdigest(),
        )

    def _observations(
        self,
        request: CampaignVariationEnvelopeRequest,
    ) -> tuple[FiniteAcquisitionObservation, ...]:
        expected_metrics = tuple(value.metric_id for value in self.objectives)
        by_configuration: dict[str, FiniteAcquisitionObservation] = {}
        for candidate in request.state.candidates:
            if not candidate.valid:
                continue
            raw = dict(candidate.objectives)
            if tuple(sorted(raw)) != expected_metrics:
                raise ValueError("archive objective axes differ from acquisition axes")
            configuration = candidate.configuration
            if type(configuration) is not FrozenJsonObject:
                raise TypeError("acquisition requires object-root configurations")
            configuration_sha256 = typed_json_sha256(configuration)
            features = self.space.features(configuration)
            observation = FiniteAcquisitionObservation(
                candidate_id=f"obs-{configuration_sha256[:32]}",
                configuration_sha256=configuration_sha256,
                features=features,
                objectives=tuple(
                    (metric_id, float(raw[metric_id]))
                    for metric_id in expected_metrics
                ),
            )
            previous = by_configuration.setdefault(configuration_sha256, observation)
            if previous != observation:
                raise ValueError("one configuration has conflicting real observations")
        observations = tuple(
            by_configuration[key] for key in sorted(by_configuration)
        )
        if not observations:
            raise ValueError("acquisition envelope requires prior valid observations")
        return observations

    def enrich(
        self,
        request: CampaignVariationEnvelopeRequest,
    ) -> CampaignVariationEnvelopeResult:
        self.__post_init__()
        if type(request) is not CampaignVariationEnvelopeRequest:
            raise TypeError("request must be an exact envelope request")
        CampaignVariationEnvelopeRequest.__post_init__(request)
        if self.protected_batch_size < (
            self.source_minimum_per_lane * len(request.lanes)
        ):
            raise ValueError(
                "protected batch must cover every active parent lane source floor"
            )

        observations = self._observations(request)
        observed_configuration_by_sha256 = {
            typed_json_sha256(value.configuration): value.configuration
            for value in request.state.candidates
            if value.valid and type(value.configuration) is FrozenJsonObject
        }
        excluded_configurations = {
            value.configuration_sha256 for value in observations
        }
        excluded_phenotypes = set(
            request.lanes[0].base_variation.known_phenotype_sha256s
        )
        base_phenotype_by_option_identity: dict[str, str] = {}
        for lane in request.lanes:
            receipt = lane.base_variation.eligibility_receipt
            receipt_by_identity = (
                {}
                if receipt is None
                else {
                    value.option_identity_sha256: value.phenotype_identity_sha256
                    for value in receipt.option_phenotypes
                }
            )
            for option in lane.base_variation.contract.options:
                excluded_configurations.add(option.child_configuration_sha256)
                phenotype_sha256 = receipt_by_identity.get(option.identity_sha256)
                if phenotype_sha256 is None:
                    phenotype_sha256 = _phenotype(
                        self.phenotype_identity,
                        option.child_configuration,
                    )
                base_phenotype_by_option_identity[option.identity_sha256] = (
                    phenotype_sha256
                )
                excluded_phenotypes.add(phenotype_sha256)
        space_request = FiniteAcquisitionSpaceRequest(
            campaign_scope_sha256=request.campaign_scope_sha256,
            cutoff_index=request.state.unique_evaluations,
            pool_size=self.pool_size,
            seed=self.seed + request.generation,
            observed_configurations=tuple(
                observed_configuration_by_sha256[key]
                for key in sorted(observed_configuration_by_sha256)
            ),
            excluded_configuration_sha256s=tuple(sorted(excluded_configurations)),
        )
        materializations = self.space.candidates(space_request)
        validate_finite_acquisition_space_candidates(
            request=space_request,
            candidates=materializations,
        )

        configurations: dict[str, FrozenJsonObject] = {}
        candidates: list[FiniteAcquisitionCandidate] = []
        seen_features: set[tuple[float, ...]] = set()
        hard_feasibility_decision_sha256s: list[str] = []
        hard_feasibility_infeasible_sha256s: list[str] = []
        hard_feasibility_rejection_witnesses: list[dict[str, object]] = []
        hard_feasibility_verdict_counts = {
            HardFeasibilityVerdict.FEASIBLE.value: 0,
            HardFeasibilityVerdict.INFEASIBLE.value: 0,
            HardFeasibilityVerdict.UNKNOWN.value: 0,
        }
        for configuration in materializations:
            configuration_sha256 = typed_json_sha256(configuration)
            if self.hard_feasibility is not None:
                feasibility_request = HardFeasibilityRequest(
                    campaign_scope_sha256=request.campaign_scope_sha256,
                    cutoff_index=request.state.unique_evaluations,
                    configuration=configuration,
                )
                feasibility = assess_hard_feasibility(
                    self.hard_feasibility,
                    feasibility_request,
                )
                hard_feasibility_decision_sha256s.append(
                    feasibility.decision_sha256
                )
                hard_feasibility_verdict_counts[feasibility.verdict.value] += 1
                if feasibility.verdict is HardFeasibilityVerdict.INFEASIBLE:
                    hard_feasibility_infeasible_sha256s.append(
                        feasibility.decision_sha256
                    )
                    if len(hard_feasibility_rejection_witnesses) < 8:
                        hard_feasibility_rejection_witnesses.append(
                            {
                                "configuration_sha256": configuration_sha256,
                                "decision": feasibility.to_record(),
                            }
                        )
                    continue
            features = self.space.features(configuration)
            if features in seen_features:
                continue
            # Constructing the port value performs finite/[0,1] validation.
            candidate = FiniteAcquisitionCandidate(
                candidate_id=f"acq-{configuration_sha256[:32]}",
                configuration_sha256=configuration_sha256,
                features=features,
            )
            configurations[configuration_sha256] = configuration
            candidates.append(candidate)
            seen_features.add(features)
        # Evaluator-equivalent configurations and duplicate feature rows are
        # intentionally removed after the outcome-blind workload reservoir is
        # materialized.  The configured pool size is therefore an upper-bound
        # on usable support, not a promise that phenotype filtering can keep
        # every row.  Only exact protected-batch coverage is semantically
        # required by the downstream acquisition boundary.
        if len(candidates) < self.protected_batch_size:
            raise ValueError("acquisition reservoir lost protected-batch support")

        remaining = tuple(candidates)
        accepted: list[tuple[object, FrozenJsonObject, str]] = []
        accepted_phenotypes = set(excluded_phenotypes)
        acquisition_requests: list[FiniteAcquisitionRequest] = []
        acquisition_decisions = []
        phenotype_rejections: list[dict[str, object]] = []
        for recourse_round in range(self.maximum_phenotype_recourse_rounds):
            needed = self.protected_batch_size - len(accepted)
            if needed <= 0:
                break
            if len(remaining) < needed:
                break
            acquisition_request = FiniteAcquisitionRequest(
                campaign_scope_sha256=request.campaign_scope_sha256,
                cutoff_index=request.state.unique_evaluations,
                batch_size=needed,
                seed=self.seed + request.generation + recourse_round * 1_000_003,
                objectives=self.objectives,
                observations=observations,
                candidates=remaining,
            )
            decision = self.acquisition.select(acquisition_request)
            if decision.request_sha256 != acquisition_request.request_sha256:
                raise ValueError("acquisition decision targets another request")
            if (
                decision.policy_id,
                decision.policy_version,
                decision.policy_definition_sha256,
            ) != _policy_identity(self.acquisition):
                raise ValueError("acquisition decision has a foreign policy identity")
            if len(decision.selected) != needed:
                raise ValueError("acquisition policy returned the wrong batch size")
            candidate_by_id = {value.candidate_id: value for value in remaining}
            selected_ids: set[str] = set()
            for value in decision.selected:
                candidate = candidate_by_id.get(value.candidate_id)
                if (
                    candidate is None
                    or candidate.configuration_sha256
                    != value.configuration_sha256
                ):
                    raise ValueError(
                        "acquisition selected outside the sealed reservoir"
                    )
                selected_ids.add(value.candidate_id)
                configuration = configurations[value.configuration_sha256]
                phenotype_sha256 = _phenotype(
                    self.phenotype_identity,
                    configuration,
                )
                if phenotype_sha256 in accepted_phenotypes:
                    phenotype_rejections.append(
                        {
                            "recourse_round": recourse_round,
                            "candidate_id": value.candidate_id,
                            "configuration_sha256": value.configuration_sha256,
                            "phenotype_sha256": phenotype_sha256,
                        }
                    )
                    continue
                accepted_phenotypes.add(phenotype_sha256)
                accepted.append((value, configuration, phenotype_sha256))
            acquisition_requests.append(acquisition_request)
            acquisition_decisions.append(decision)
            remaining = tuple(
                value for value in remaining if value.candidate_id not in selected_ids
            )
        if len(accepted) != self.protected_batch_size:
            raise ValueError("phenotype recourse underfilled the protected batch")
        selected = tuple(
            (rank, value, configuration, phenotype_sha256)
            for rank, (value, configuration, phenotype_sha256) in enumerate(
                accepted,
                start=1,
            )
        )

        assignments: dict[str, list[tuple[int, object, FrozenJsonObject]]] = {
            value.lane_id: [] for value in request.lanes
        }
        lane_ids = tuple(sorted(assignments))
        for ordinal, row in enumerate(selected):
            assignments[lane_ids[ordinal % len(lane_ids)]].append(row)

        result_lanes: list[CampaignVariationEnvelopeLaneResult] = []
        partition_evidence: list[dict[str, object]] = []
        certification_contexts: list[AcquisitionCertifiedSlateContext] = []
        for lane in request.lanes:
            assigned = assignments[lane.lane_id]
            if not assigned:
                raise ValueError("protected acquisition left a lane uncovered")
            parent_sha256 = lane.base_variation.parent_configuration_sha256
            acquisition_options: list[FiniteVariationOption] = []
            selected_phenotypes_by_option_id: dict[str, str] = {}
            for rank, selection, configuration, phenotype_sha256 in assigned:
                if not math.isfinite(selection.acquisition_value):
                    raise ValueError("acquisition value must remain finite")
                option = FiniteVariationOption(
                        option_id=(
                            f"acquisition.g{request.generation:02d}."
                            f"r{rank:03d}.{selection.configuration_sha256[:16]}"
                        ),
                        parent_configuration_sha256=parent_sha256,
                        child_configuration=configuration,
                        family=self.option_family,
                        description=(
                            "Protected numerical acquisition candidate "
                            f"rank {rank:03d} from strictly prior real outcomes."
                        ),
                        metadata=tuple(
                            sorted(
                                (
                                    ("acquisition_policy_id", decision.policy_id),
                                    ("acquisition_rank", f"{rank:03d}"),
                                    (
                                        "acquisition_value_hex",
                                        selection.acquisition_value.hex(),
                                    ),
                                    (
                                        VARIATION_SOURCE_RANK_METADATA_KEY,
                                        str(rank),
                                    ),
                                    (
                                        VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY,
                                        f"acquisition.rank{rank:03d}",
                                    ),
                                    (VARIATION_OPERATOR_METADATA_KEY, self.operator_id),
                                    (VARIATION_SOURCE_METADATA_KEY, self.source_id),
                                    *(
                                        (
                                            (
                                                VARIATION_CANDIDATE_POOL_REQUIREMENT_METADATA_KEY,
                                                "required",
                                            ),
                                        )
                                        if self.acquisition_certification_context_sink
                                        is not None
                                        else ()
                                    ),
                                    (
                                        VARIATION_SOURCE_MINIMUM_METADATA_KEY,
                                        str(self.source_minimum_per_lane),
                                    ),
                                )
                            )
                        ),
                    )
                acquisition_options.append(option)
                selected_phenotypes_by_option_id[option.option_id] = phenotype_sha256
            combined_options = (
                *lane.base_variation.contract.options,
                *acquisition_options,
            )
            if len(combined_options) > MAX_FINITE_VARIATION_OPTIONS:
                raise ValueError("acquisition envelope exceeds the finite-option bound")
            combined = FiniteVariationContract(
                catalog_id="finite_acquisition_expert_union",
                catalog_version=POLICY_VERSION,
                catalog_definition_sha256=self._catalog_definition_sha256,
                parent_configuration=lane.base_variation.contract.parent_configuration,
                options=tuple(combined_options),
            )
            phenotype_bindings = tuple(
                OptionPhenotypeBinding(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    phenotype_identity_sha256=(
                        selected_phenotypes_by_option_id[option.option_id]
                        if option.option_id in selected_phenotypes_by_option_id
                        else base_phenotype_by_option_identity[
                            option.identity_sha256
                        ]
                    ),
                )
                for option in combined.options
            )
            eligible = eligible_finite_variation_view(
                contract=combined,
                option_phenotypes=phenotype_bindings,
                known_phenotype_sha256s=(
                    lane.base_variation.known_phenotype_sha256s
                ),
            )
            if self.source_id not in finite_variation_source_ids(eligible.contract):
                raise ValueError("protected acquisition source vanished at eligibility")
            enriched = ParentVariationBinding(
                benchmark_sha256=lane.base_variation.benchmark_sha256,
                parent_configuration_sha256=parent_sha256,
                known_phenotype_sha256s=(
                    lane.base_variation.known_phenotype_sha256s
                ),
                contract=eligible.contract,
                eligibility_receipt=eligible.receipt,
            )
            reference_option_ids = tuple(
                sorted(
                    option.option_id
                    for option in eligible.contract.options
                    if finite_variation_source_id(option) == self.source_id
                )
            )
            if not reference_option_ids:
                raise ValueError("certification reference lost every acquisition option")
            context = AcquisitionCertifiedSlateContext(
                campaign_scope_sha256=request.campaign_scope_sha256,
                finite_contract_sha256=eligible.contract.identity_sha256,
                cutoff_index=request.state.unique_evaluations,
                seed=int.from_bytes(
                    hashlib.sha256(
                        (
                            f"{self.seed}:{request.generation}:{lane.lane_id}"
                        ).encode("ascii", errors="strict")
                    ).digest()[:4],
                    byteorder="big",
                    signed=False,
                ),
                objectives=self.objectives,
                observations=observations,
                candidates=tuple(
                    FiniteAcquisitionCandidate(
                        candidate_id=option.option_id,
                        configuration_sha256=option.child_configuration_sha256,
                        features=self.space.features(option.child_configuration),
                    )
                    for option in eligible.contract.options
                ),
                reference_option_ids=reference_option_ids,
            )
            certification_contexts.append(context)
            result_lanes.append(
                CampaignVariationEnvelopeLaneResult(
                    lane_id=lane.lane_id,
                    variation=enriched,
                )
            )
            partition_evidence.append(
                {
                    "lane_id": lane.lane_id,
                    "base_option_count": len(
                        lane.base_variation.contract.options
                    ),
                    "acquisition_ranks": [rank for rank, _, _, _ in assigned],
                    "eligible_option_count": len(eligible.contract.options),
                    "acquisition_certification_context_sha256": (
                        context.context_sha256
                    ),
                    "acquisition_certification_reference_count": (
                        len(context.reference_option_ids)
                    ),
                }
            )

        if self.acquisition_certification_context_sink is not None:
            self.acquisition_certification_context_sink.register_many(
                tuple(certification_contexts)
            )

        evidence = freeze_json(
            {
                "schema_version": 1,
                "space_request_sha256": space_request.request_sha256,
                "acquisition_request_sha256": (
                    acquisition_requests[0].request_sha256
                ),
                "acquisition_decision": acquisition_decisions[0].to_record(),
                "acquisition_recourse_requests": [
                    {
                        "request_sha256": value.request_sha256,
                        "batch_size": value.batch_size,
                        "seed": value.seed,
                        "observation_count": len(value.observations),
                        "candidate_pool_size": len(value.candidates),
                    }
                    for value in acquisition_requests
                ],
                "acquisition_recourse_decisions": [
                    value.to_record() for value in acquisition_decisions
                ],
                "phenotype_rejections": phenotype_rejections,
                "phenotype_checks_performed": (
                    len(accepted) + len(phenotype_rejections)
                ),
                "hard_feasibility": {
                    "enabled": self.hard_feasibility is not None,
                    "verdict_counts": hard_feasibility_verdict_counts,
                    "decision_count": len(hard_feasibility_decision_sha256s),
                    "decision_batch_sha256": (
                        hard_feasibility_decision_batch_sha256(
                            hard_feasibility_decision_sha256s
                        )
                    ),
                    "infeasible_decision_count": len(
                        hard_feasibility_infeasible_sha256s
                    ),
                    "infeasible_decision_batch_sha256": (
                        hard_feasibility_decision_batch_sha256(
                            hard_feasibility_infeasible_sha256s
                        )
                    ),
                    "rejection_witnesses": (
                        hard_feasibility_rejection_witnesses
                    ),
                    "rejection_witness_limit": 8,
                    "rejection_witness_sampling": (
                        "first_in_authenticated_space_order"
                    ),
                    "rejection_authority": "exact_infeasibility_only",
                    "unknown_disposition": "retain",
                    "outcome_access": False,
                },
                "partition": partition_evidence,
                "acquisition_certification_context_sha256s": [
                    value.context_sha256 for value in certification_contexts
                ],
                "acquisition_certification_contexts_registered": (
                    self.acquisition_certification_context_sink is not None
                ),
                "protected_source_id": self.source_id,
                "strictly_prior_observation_count": len(observations),
                "raw_reservoir_size": len(materializations),
                "reservoir_size": len(candidates),
                "current_future_outcomes_consulted": False,
                "workload_model_provider_branches": False,
            }
        )
        if type(evidence) is not FrozenJsonObject:  # pragma: no cover
            raise AssertionError("acquisition envelope evidence is not an object")
        return CampaignVariationEnvelopeResult(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            lanes=tuple(result_lanes),
            evidence=evidence,
        )


__all__ = [
    "POLICY_ID",
    "POLICY_VERSION",
    "PROTECTED_ACQUISITION_SOURCE_ID",
    "ProtectedFiniteAcquisitionVariationEnvelope",
]

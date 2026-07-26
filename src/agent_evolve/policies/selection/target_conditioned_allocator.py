"""Inverted allocation adapter for portable target-conditioned acquisition.

The adapter implements the existing ``select(SlateAllocationRequest)`` shape,
but obtains branch-local prior state and configuration-transition receipts from
an injected, authenticated context provider.  Workload integrations therefore
materialize only generic finite configurations and feasibility; provider code,
objective values, and domain identifiers remain outside the acquisition core.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import ClassVar, Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validated_finite_variation_identity_index,
)
from agent_evolve.policies.selection.calibrated_slate import SlateAllocationRequest
from agent_evolve.policies.selection.structural_posterior_slate import (
    StructuralPosteriorMemberScoreRow,
    score_structural_posterior_slate,
)
from agent_evolve.policies.selection.target_conditioned_features import (
    PortableTransitionReceipt,
    TargetConditionedFeatureProjectionRequest,
    TargetConditionedPortableFeatureProjector,
    project_portable_transition,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    BASE_REALIZABILITY_DEFINITION_SHA256,
    BASE_REALIZABILITY_PROJECTOR_ID,
    BASE_REALIZABILITY_PROJECTOR_VERSION,
    RealizablePortfolioSet,
    TargetConditionedAcquisitionProfile,
    TargetConditionedAcquisitionState,
    TargetConditionedPrequentialSlatePolicy,
    TargetConditionedSlateDecision,
    TargetConditionedSlateRequest,
    enumerate_base_realizable_portfolios,
)
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjection,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget


ADAPTER_ID = "target_conditioned_prequential_allocator_adapter"
ADAPTER_VERSION = 1
ADAPTER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:target-conditioned-prequential-allocator-adapter:v1;"
    b"input=slate-allocation-request,authenticated-prior-context-provider,"
    b"portable-feature-projector,realizable-set-projector;"
    b"structural-score-projection=prior-only-complete-slate;"
    b"transition-receipts=precall-context-subset;"
    b"selection=target-conditioned-prequential-policy;"
    b"workload-model-provider-current-outcome-fields=false"
).hexdigest()
BASE_REALIZABLE_PROJECTOR_ID = BASE_REALIZABILITY_PROJECTOR_ID
BASE_REALIZABLE_PROJECTOR_VERSION = BASE_REALIZABILITY_PROJECTOR_VERSION
BASE_REALIZABLE_PROJECTOR_DEFINITION_SHA256 = (
    BASE_REALIZABILITY_DEFINITION_SHA256
)
PRIOR_STRUCTURAL_SCORE_PROJECTOR_ID = "prior_structural_slate_scores"
PRIOR_STRUCTURAL_SCORE_PROJECTOR_VERSION = 1
PRIOR_STRUCTURAL_SCORE_PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:prior-structural-slate-scores:v1;"
    b"delegates=score-structural-posterior-slate;"
    b"complete-sealed-slate=true;current-future-outcomes=false"
).hexdigest()
REGISTERED_CONTEXT_PROVIDER_ID = "registered_target_conditioned_contexts"
REGISTERED_CONTEXT_PROVIDER_VERSION = 2
REGISTERED_CONTEXT_PROVIDER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:registered-target-conditioned-contexts:v2;"
    b"key=scope-wave-parent-finite-contract;registration=append-only-precall;"
    b"contract=sealed-once;lookup=exact-slate-allocation-request;"
    b"transition-projection=lazy-exact-proposed-slate;"
    b"current-future-outcomes=false"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_CONTEXT_DOMAIN = b"agent-evolve:target-conditioned-allocation-context:v1\x00"
_ADAPTER_CONFIGURATION_DOMAIN = (
    b"agent-evolve:target-conditioned-allocator-configuration:v1\x00"
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


def _provider_identity(value: object, *, name: str) -> dict[str, object]:
    provider_id = getattr(value, "provider_id", None)
    provider_version = getattr(value, "provider_version", None)
    definition_sha256 = getattr(value, "definition_sha256", None)
    if type(provider_id) is not str or _TOKEN.fullmatch(provider_id) is None:
        raise ValueError(f"{name}.provider_id must use the closed token grammar")
    if type(provider_version) is not int or provider_version <= 0:
        raise ValueError(f"{name}.provider_version must be positive")
    require_sha256(definition_sha256, f"{name}.definition_sha256")
    return {
        "provider_id": provider_id,
        "provider_version": provider_version,
        "definition_sha256": definition_sha256,
    }


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedAllocationContext:
    """Pre-call, branch-local facts available to one eventual K-slate."""

    finite_contract_sha256: str
    cutoff_receipt_sha256: str
    archive_context: CampaignPortfolioArchiveContextProjection
    frontier_target: CampaignPortfolioFrontierTarget
    state: TargetConditionedAcquisitionState
    transition_receipts: tuple[PortableTransitionReceipt, ...]
    campaign_generation: int
    lane_slot: int
    remaining_proposal_horizon: int

    def __post_init__(self) -> None:
        require_sha256(self.finite_contract_sha256, "finite_contract_sha256")
        require_sha256(self.cutoff_receipt_sha256, "cutoff_receipt_sha256")
        if type(self.archive_context) is not CampaignPortfolioArchiveContextProjection:
            raise TypeError("archive_context must be exact")
        self.archive_context.__post_init__()
        if type(self.frontier_target) is not CampaignPortfolioFrontierTarget:
            raise TypeError("frontier_target must be exact")
        self.frontier_target.__post_init__()
        if type(self.state) is not TargetConditionedAcquisitionState:
            raise TypeError("state must be exact")
        self.state.__post_init__()
        if type(self.transition_receipts) is not tuple:
            raise TypeError("transition_receipts must be an exact tuple")
        if any(
            type(value) is not PortableTransitionReceipt
            for value in self.transition_receipts
        ):
            raise TypeError("transition_receipts must contain exact rows")
        for value in self.transition_receipts:
            value.__post_init__()
        option_ids = tuple(value.option_id for value in self.transition_receipts)
        if option_ids != tuple(sorted(set(option_ids))):
            raise ValueError("transition receipts must use canonical option order")
        parents = {
            value.parent_configuration_sha256 for value in self.transition_receipts
        }
        if parents and parents != {self.frontier_target.parent_configuration_sha256}:
            raise ValueError("transition receipts differ from the targeted parent")
        if (
            self.archive_context.archive_utility_snapshot_sha256
            != self.frontier_target.archive_utility_snapshot_sha256
            or self.archive_context.parent_configuration_sha256
            != self.frontier_target.parent_configuration_sha256
        ):
            raise ValueError("archive context and frontier target disagree")
        if type(self.campaign_generation) is not int or self.campaign_generation <= 0:
            raise ValueError("campaign_generation must be positive")
        if self.state.cutoff_generation >= self.campaign_generation:
            raise ValueError("state cutoff reaches the current generation")
        if type(self.lane_slot) is not int or self.lane_slot < 0:
            raise ValueError("lane_slot must be non-negative")
        if (
            type(self.remaining_proposal_horizon) is not int
            or self.remaining_proposal_horizon < 0
        ):
            raise ValueError("remaining_proposal_horizon must be non-negative")

    def require_request(self, request: SlateAllocationRequest) -> None:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        if request.slate.finite_contract_sha256 != self.finite_contract_sha256:
            raise ValueError("allocation request names a foreign finite contract")
        if (
            request.slate.parent_candidate_identity_sha256
            != self.frontier_target.parent_configuration_sha256
        ):
            raise ValueError("allocation request names a foreign parent")
        transitions = {value.option_id: value for value in self.transition_receipts}
        for member in request.slate.members:
            transition = transitions.get(member.option_id)
            if (
                transition is None
                or transition.option_identity_sha256
                != member.option_identity_sha256
            ):
                raise ValueError("pre-call transition context does not cover the slate")

    def transition_subset(
        self, request: SlateAllocationRequest
    ) -> tuple[PortableTransitionReceipt, ...]:
        self.require_request(request)
        option_ids = {value.option_id for value in request.slate.members}
        return tuple(
            value for value in self.transition_receipts if value.option_id in option_ids
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "finite_contract_sha256": self.finite_contract_sha256,
            "cutoff_receipt_sha256": self.cutoff_receipt_sha256,
            "archive_context": self.archive_context.to_record(),
            "frontier_target": self.frontier_target.to_record(),
            "state": self.state.to_record(),
            "transition_receipts": [
                value.to_record() for value in self.transition_receipts
            ],
            "campaign_generation": self.campaign_generation,
            "lane_slot": self.lane_slot,
            "remaining_proposal_horizon": self.remaining_proposal_horizon,
            "context_frozen_before_model_call": True,
            "current_or_future_outcomes_consulted": False,
        }

    @property
    def context_sha256(self) -> str:
        return _hash(_CONTEXT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "context_sha256": self.context_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedAllocationContext
            and self.context_sha256 == other.context_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class TargetConditionedAllocationContextKey:
    """Provider-independent identity for one prospectively registered branch."""

    scope_sha256: str
    wave_index: int
    parent_configuration_sha256: str
    finite_contract_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "scope_sha256",
            "parent_configuration_sha256",
            "finite_contract_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be positive")

    @classmethod
    def from_request(
        cls, request: SlateAllocationRequest
    ) -> TargetConditionedAllocationContextKey:
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        return cls(
            scope_sha256=request.slate.scope.scope_sha256,
            wave_index=request.slate.wave_index,
            parent_configuration_sha256=(
                request.slate.parent_candidate_identity_sha256
            ),
            finite_contract_sha256=request.slate.finite_contract_sha256,
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "scope_sha256": self.scope_sha256,
            "wave_index": self.wave_index,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "finite_contract_sha256": self.finite_contract_sha256,
        }

    @property
    def key_sha256(self) -> str:
        return _hash(
            b"agent-evolve:target-conditioned-allocation-context-key:v1\x00",
            self._unsigned_record(),
        )

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "key_sha256": self.key_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is TargetConditionedAllocationContextKey
            and self.key_sha256 == other.key_sha256
        )

    __hash__ = None


@runtime_checkable
class TargetConditionedAllocationContextProvider(Protocol):
    provider_id: str
    provider_version: int
    definition_sha256: str

    def context_for(
        self, request: SlateAllocationRequest
    ) -> TargetConditionedAllocationContext: ...


@runtime_checkable
class TargetConditionedAllocationContextRegistrar(
    TargetConditionedAllocationContextProvider, Protocol
):
    def register(
        self,
        key: TargetConditionedAllocationContextKey,
        context: TargetConditionedAllocationContext,
        finite_contract: FiniteVariationContract,
    ) -> None: ...


@dataclass(slots=True)
class RegisteredTargetConditionedAllocationContextProvider:
    """Append-only in-memory bridge from pre-call campaign facts to K8 allocation."""

    provider_id: str = REGISTERED_CONTEXT_PROVIDER_ID
    provider_version: int = REGISTERED_CONTEXT_PROVIDER_VERSION
    definition_sha256: str = REGISTERED_CONTEXT_PROVIDER_DEFINITION_SHA256
    _contexts: dict[
        str,
        tuple[
            TargetConditionedAllocationContextKey,
            TargetConditionedAllocationContext,
            FiniteVariationContract,
        ],
    ] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        expected = {
            "provider_id": REGISTERED_CONTEXT_PROVIDER_ID,
            "provider_version": REGISTERED_CONTEXT_PROVIDER_VERSION,
            "definition_sha256": REGISTERED_CONTEXT_PROVIDER_DEFINITION_SHA256,
        }
        if _provider_identity(self, name="registered context provider") != expected:
            raise ValueError("registered context provider identity drifted")

    def register(
        self,
        key: TargetConditionedAllocationContextKey,
        context: TargetConditionedAllocationContext,
        finite_contract: FiniteVariationContract,
    ) -> None:
        """Seal one context before its corresponding provider call begins."""

        self.__post_init__()
        if type(key) is not TargetConditionedAllocationContextKey:
            raise TypeError("key must be exact TargetConditionedAllocationContextKey")
        key.__post_init__()
        if type(context) is not TargetConditionedAllocationContext:
            raise TypeError("context must be exact TargetConditionedAllocationContext")
        context.__post_init__()
        if type(finite_contract) is not FiniteVariationContract:
            raise TypeError("finite_contract must be exact FiniteVariationContract")
        identity_index = validated_finite_variation_identity_index(finite_contract)
        if (
            context.finite_contract_sha256 != key.finite_contract_sha256
            or context.frontier_target.parent_configuration_sha256
            != key.parent_configuration_sha256
            or identity_index.contract_identity_sha256
            != key.finite_contract_sha256
            or identity_index.parent_configuration_sha256
            != key.parent_configuration_sha256
        ):
            raise ValueError("registered context differs from its branch key")
        if key.key_sha256 in self._contexts:
            raise ValueError("target-conditioned branch is already registered")
        self._contexts[key.key_sha256] = (key, context, finite_contract)

    def context_for(
        self, request: SlateAllocationRequest
    ) -> TargetConditionedAllocationContext:
        self.__post_init__()
        key = TargetConditionedAllocationContextKey.from_request(request)
        registered = self._contexts.get(key.key_sha256)
        if registered is None:
            raise ValueError("target-conditioned branch is foreign or unregistered")
        registered_key, context, finite_contract = registered
        if registered_key != key:
            raise ValueError("target-conditioned context key digest collision")
        identity_index = validated_finite_variation_identity_index(finite_contract)
        identity_by_option_id = dict(
            zip(
                identity_index.option_ids,
                identity_index.option_identity_sha256s,
                strict=True,
            )
        )
        option_by_id = {
            option.option_id: option for option in finite_contract.options
        }
        transitions = tuple(
            project_portable_transition(
                option_id=member.option_id,
                option_identity_sha256=member.option_identity_sha256,
                parent_configuration=finite_contract.parent_configuration,
                child_configuration=option_by_id[member.option_id].child_configuration,
            )
            for member in sorted(request.slate.members, key=lambda value: value.option_id)
            if identity_by_option_id.get(member.option_id)
            == member.option_identity_sha256
        )
        if len(transitions) != len(request.slate.members):
            raise ValueError("proposed slate differs from its registered finite contract")
        materialized = replace(context, transition_receipts=transitions)
        materialized.require_request(request)
        return materialized

    @property
    def registered_context_count(self) -> int:
        return len(self._contexts)


@runtime_checkable
class RealizablePortfolioProjector(Protocol):
    provider_id: str
    provider_version: int
    definition_sha256: str

    def project(self, request: SlateAllocationRequest) -> RealizablePortfolioSet: ...


@runtime_checkable
class StructuralScoreProjector(Protocol):
    provider_id: str
    provider_version: int
    definition_sha256: str

    def project(
        self, request: SlateAllocationRequest
    ) -> tuple[StructuralPosteriorMemberScoreRow, ...]: ...


@dataclass(frozen=True, slots=True)
class PriorStructuralScoreProjector:
    provider_id: str = PRIOR_STRUCTURAL_SCORE_PROJECTOR_ID
    provider_version: int = PRIOR_STRUCTURAL_SCORE_PROJECTOR_VERSION
    definition_sha256: str = PRIOR_STRUCTURAL_SCORE_PROJECTOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        expected = {
            "provider_id": PRIOR_STRUCTURAL_SCORE_PROJECTOR_ID,
            "provider_version": PRIOR_STRUCTURAL_SCORE_PROJECTOR_VERSION,
            "definition_sha256": PRIOR_STRUCTURAL_SCORE_PROJECTOR_DEFINITION_SHA256,
        }
        if _provider_identity(self, name="structural score projector") != expected:
            raise ValueError("prior structural score projector identity drifted")

    def project(
        self, request: SlateAllocationRequest
    ) -> tuple[StructuralPosteriorMemberScoreRow, ...]:
        self.__post_init__()
        return score_structural_posterior_slate(request)


@dataclass(frozen=True, slots=True)
class BaseRealizablePortfolioProjector:
    provider_id: str = BASE_REALIZABLE_PROJECTOR_ID
    provider_version: int = BASE_REALIZABLE_PROJECTOR_VERSION
    definition_sha256: str = BASE_REALIZABLE_PROJECTOR_DEFINITION_SHA256

    def __post_init__(self) -> None:
        expected = {
            "provider_id": BASE_REALIZABLE_PROJECTOR_ID,
            "provider_version": BASE_REALIZABLE_PROJECTOR_VERSION,
            "definition_sha256": BASE_REALIZABLE_PROJECTOR_DEFINITION_SHA256,
        }
        if _provider_identity(self, name="base realizability projector") != expected:
            raise ValueError("base realizability projector identity drifted")

    def project(self, request: SlateAllocationRequest) -> RealizablePortfolioSet:
        self.__post_init__()
        return enumerate_base_realizable_portfolios(request)


@dataclass(frozen=True, slots=True)
class TargetConditionedSlateAllocatorAdapter:
    """Compose portable projection and T-RAP behind the existing allocation port."""

    context_provider: TargetConditionedAllocationContextProvider
    profile: TargetConditionedAcquisitionProfile
    feature_projector: TargetConditionedPortableFeatureProjector = (
        TargetConditionedPortableFeatureProjector()
    )
    structural_score_projector: StructuralScoreProjector = (
        PriorStructuralScoreProjector()
    )
    realizability_projector: RealizablePortfolioProjector = (
        BaseRealizablePortfolioProjector()
    )

    policy_id: ClassVar[str] = ADAPTER_ID
    policy_version: ClassVar[int] = ADAPTER_VERSION
    definition_sha256: ClassVar[str] = ADAPTER_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if not isinstance(
            self.context_provider, TargetConditionedAllocationContextProvider
        ):
            raise TypeError("context_provider must implement the authenticated port")
        _provider_identity(self.context_provider, name="context_provider")
        if type(self.profile) is not TargetConditionedAcquisitionProfile:
            raise TypeError("profile must be exact")
        self.profile.__post_init__()
        if type(self.feature_projector) is not TargetConditionedPortableFeatureProjector:
            raise TypeError("feature_projector must be exact")
        if not isinstance(self.structural_score_projector, StructuralScoreProjector):
            raise TypeError("structural_score_projector must implement the port")
        _provider_identity(
            self.structural_score_projector, name="structural_score_projector"
        )
        if not isinstance(self.realizability_projector, RealizablePortfolioProjector):
            raise TypeError("realizability_projector must implement the port")
        _provider_identity(
            self.realizability_projector, name="realizability_projector"
        )

    def select(self, request: SlateAllocationRequest) -> TargetConditionedSlateDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        context = self.context_provider.context_for(request)
        if type(context) is not TargetConditionedAllocationContext:
            raise TypeError("context provider returned a foreign value")
        context.require_request(request)
        raw_scores = self.structural_score_projector.project(request)
        if type(raw_scores) is not tuple or any(
            type(value) is not StructuralPosteriorMemberScoreRow
            for value in raw_scores
        ):
            raise TypeError("structural score projector returned foreign rows")
        scores = tuple(sorted(raw_scores, key=lambda value: value.option_id))
        features = self.feature_projector.project(
            TargetConditionedFeatureProjectionRequest(
                allocation_request=request,
                structural_score_rows=scores,
                transition_receipts=context.transition_subset(request),
                archive_context=context.archive_context,
                frontier_target=context.frontier_target,
                campaign_generation=context.campaign_generation,
                lane_slot=context.lane_slot,
                remaining_proposal_horizon=(context.remaining_proposal_horizon),
            )
        )
        realizable = self.realizability_projector.project(request)
        if type(realizable) is not RealizablePortfolioSet:
            raise TypeError("realizability projector returned a foreign value")
        projected_identity = {
            "provider_id": realizable.projector_id,
            "provider_version": realizable.projector_version,
            "definition_sha256": realizable.projector_definition_sha256,
        }
        if projected_identity != _provider_identity(
            self.realizability_projector, name="realizability_projector"
        ):
            raise ValueError("realizability receipt has a foreign projector identity")
        return TargetConditionedPrequentialSlatePolicy(self.profile).select(
            TargetConditionedSlateRequest(
                allocation_request=request,
                frontier_target=context.frontier_target,
                state=context.state,
                member_features=features,
                realizable_portfolios=realizable,
                campaign_generation=context.campaign_generation,
                remaining_proposal_horizon=(context.remaining_proposal_horizon),
            )
        )

    @property
    def configuration_sha256(self) -> str:
        """Bind every injected prior-only provider and the frozen T-RAP profile."""

        return _hash(_ADAPTER_CONFIGURATION_DOMAIN, self.to_record())

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "profile": self.profile.to_record(),
            "feature_projector": self.feature_projector.to_record(),
            "structural_score_projector": _provider_identity(
                self.structural_score_projector,
                name="structural_score_projector",
            ),
            "context_provider": _provider_identity(
                self.context_provider, name="context_provider"
            ),
            "realizability_projector": _provider_identity(
                self.realizability_projector, name="realizability_projector"
            ),
        }


__all__ = [
    "ADAPTER_DEFINITION_SHA256",
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "BASE_REALIZABLE_PROJECTOR_DEFINITION_SHA256",
    "BASE_REALIZABLE_PROJECTOR_ID",
    "BASE_REALIZABLE_PROJECTOR_VERSION",
    "BaseRealizablePortfolioProjector",
    "PRIOR_STRUCTURAL_SCORE_PROJECTOR_DEFINITION_SHA256",
    "PRIOR_STRUCTURAL_SCORE_PROJECTOR_ID",
    "PRIOR_STRUCTURAL_SCORE_PROJECTOR_VERSION",
    "PriorStructuralScoreProjector",
    "REGISTERED_CONTEXT_PROVIDER_DEFINITION_SHA256",
    "REGISTERED_CONTEXT_PROVIDER_ID",
    "REGISTERED_CONTEXT_PROVIDER_VERSION",
    "RegisteredTargetConditionedAllocationContextProvider",
    "RealizablePortfolioProjector",
    "StructuralScoreProjector",
    "TargetConditionedAllocationContext",
    "TargetConditionedAllocationContextKey",
    "TargetConditionedAllocationContextProvider",
    "TargetConditionedAllocationContextRegistrar",
    "TargetConditionedSlateAllocatorAdapter",
]

"""Acquisition-certified residual allocation over a model-proposed slate.

An upstream numerical expert contributes a reference slate while the language
model contributes residual alternatives.  This policy enumerates every hard-
feasible evaluation subset of the authenticated proposal, scores all subsets
under one strictly-prior acquisition realization, and retains the reference on
ties.  The selected slate therefore cannot have lower *measured acquisition*
than the best feasible reference-preserving slate.  This is deliberately not a
claim about unseen evaluator outcomes; those remain the experiment endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.calibrated_slate import (
    AllocatedSlateMember,
    SlateAllocationRequest,
    SlateAllocationRole,
    assess_allocated_slate_memory_dose,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
)
from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScoreDecision,
    FiniteAcquisitionBatchScorePolicy,
    FiniteAcquisitionBatchScoreRequest,
    FiniteAcquisitionSlate,
    validate_finite_acquisition_batch_score_decision,
)
from agent_evolve.ports.portfolio_memory_dose import PortfolioMemoryDoseAssessment


POLICY_ID = "acquisition_certified_residual_slate"
POLICY_VERSION = 1
_CONTEXT_DOMAIN = b"agent-evolve:acquisition-certified-slate-context:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:acquisition-certified-slate-decision:v1\x00"
_POLICY_DOMAIN = b"agent-evolve:acquisition-certified-slate-policy:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def finite_acquisition_policy_identity(
    policy: FiniteAcquisitionBatchScorePolicy,
) -> tuple[str, int, str]:
    if not isinstance(policy, FiniteAcquisitionBatchScorePolicy):
        raise TypeError("scorer must implement FiniteAcquisitionBatchScorePolicy")
    values = (
        getattr(policy, "policy_id", None),
        getattr(policy, "policy_version", None),
        getattr(policy, "definition_sha256", None),
    )
    if type(values[0]) is not str or not values[0]:
        raise ValueError("scorer policy_id must be non-empty")
    if type(values[1]) is not int or values[1] <= 0:
        raise ValueError("scorer policy_version must be positive")
    require_sha256(values[2], "scorer definition_sha256")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, eq=False)
class AcquisitionCertifiedSlateContext:
    """Strictly-prior numerical context registered for one finite contract."""

    campaign_scope_sha256: str
    finite_contract_sha256: str
    cutoff_index: int
    seed: int
    objectives: tuple[FiniteAcquisitionObjective, ...]
    observations: tuple[FiniteAcquisitionObservation, ...]
    candidates: tuple[FiniteAcquisitionCandidate, ...]
    reference_option_ids: tuple[str, ...]
    context_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(self.finite_contract_sha256, "finite_contract_sha256")
        if type(self.cutoff_index) is not int or self.cutoff_index < 1:
            raise ValueError("cutoff_index must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be non-negative")
        # A one-slate request exercises the complete shared acquisition input
        # law without introducing a second validator in this module.
        if type(self.reference_option_ids) is not tuple or not (
            self.reference_option_ids
        ):
            raise ValueError("reference_option_ids must be a non-empty tuple")
        if self.reference_option_ids != tuple(sorted(set(self.reference_option_ids))):
            raise ValueError("reference option IDs must be unique and canonical")
        available = {value.candidate_id for value in self.candidates}
        if not set(self.reference_option_ids) <= available:
            raise ValueError("reference options escape the numerical candidate set")
        FiniteAcquisitionBatchScoreRequest(
            campaign_scope_sha256=self.campaign_scope_sha256,
            cutoff_index=self.cutoff_index,
            seed=self.seed,
            objectives=self.objectives,
            observations=self.observations,
            candidates=self.candidates,
            slates=(FiniteAcquisitionSlate(self.reference_option_ids),),
        )
        computed = hashlib.sha256(
            _CONTEXT_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.context_sha256 not in ("", computed):
            raise ValueError("context_sha256 does not authenticate the context")
        object.__setattr__(self, "context_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "finite_contract_sha256": self.finite_contract_sha256,
            "cutoff_index": self.cutoff_index,
            "seed": self.seed,
            "objectives": [value.to_record() for value in self.objectives],
            "observations": [value.to_record() for value in self.observations],
            "candidates": [value.to_record() for value in self.candidates],
            "reference_option_ids": list(self.reference_option_ids),
            "current_or_future_outcomes_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "context_sha256": self.context_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is AcquisitionCertifiedSlateContext
            and self.context_sha256 == other.context_sha256
        )

    __hash__ = None


@runtime_checkable
class AcquisitionCertifiedSlateContextProvider(Protocol):
    provider_id: str
    provider_version: int
    definition_sha256: str

    def context_for(
        self,
        finite_contract_sha256: str,
    ) -> AcquisitionCertifiedSlateContext: ...


@runtime_checkable
class AcquisitionCertifiedSlateContextSink(Protocol):
    provider_id: str
    provider_version: int
    definition_sha256: str

    def register_many(
        self,
        contexts: tuple[AcquisitionCertifiedSlateContext, ...],
    ) -> None: ...


@dataclass(slots=True)
class AcquisitionCertifiedSlateContextRegistry:
    """Append-only, replay-idempotent bridge from envelope to allocator.

    Preparation and audit paths may deterministically reconstruct an already
    registered context.  Re-registering the exact authenticated value is a
    no-op; attempting to bind the same finite contract to a different context
    remains a hard error.  The conflict check is performed before any write so
    that a mixed batch cannot partially mutate the registry.
    """

    provider_id: str = "append_only_acquisition_certification_context"
    provider_version: int = 2
    definition_sha256: str = hashlib.sha256(
        b"agent-evolve:append-only-acquisition-certification-context:v2;"
        b"key=finite-contract-sha256;identical-replay=idempotent;"
        b"conflicting-replacement=false;batch-write=atomic;strictly-prior=true"
    ).hexdigest()
    _contexts: dict[str, AcquisitionCertifiedSlateContext] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def register(self, context: AcquisitionCertifiedSlateContext) -> None:
        self.register_many((context,))

    def register_many(
        self,
        contexts: tuple[AcquisitionCertifiedSlateContext, ...],
    ) -> None:
        if type(contexts) is not tuple or not contexts:
            raise ValueError("contexts must be a non-empty exact tuple")
        incoming: dict[str, AcquisitionCertifiedSlateContext] = {}
        for context in contexts:
            if type(context) is not AcquisitionCertifiedSlateContext:
                raise TypeError("contexts must contain exact contexts")
            context.__post_init__()
            prior = incoming.get(context.finite_contract_sha256)
            if prior is not None:
                if prior.context_sha256 != context.context_sha256:
                    raise ValueError(
                        "context batch binds one finite contract to conflicting "
                        "contexts"
                    )
                continue
            incoming[context.finite_contract_sha256] = context
        conflicts = tuple(
            sorted(
                finite_contract_sha256
                for finite_contract_sha256, context in incoming.items()
                if finite_contract_sha256 in self._contexts
                and self._contexts[finite_contract_sha256].context_sha256
                != context.context_sha256
            )
        )
        if conflicts:
            raise ValueError(
                "acquisition certification context is append-only; conflicting "
                f"contracts={conflicts!r}"
            )
        self._contexts.update(
            {
                finite_contract_sha256: context
                for finite_contract_sha256, context in incoming.items()
                if finite_contract_sha256 not in self._contexts
            }
        )

    def context_for(
        self,
        finite_contract_sha256: str,
    ) -> AcquisitionCertifiedSlateContext:
        require_sha256(finite_contract_sha256, "finite_contract_sha256")
        try:
            return self._contexts[finite_contract_sha256]
        except KeyError as error:
            raise KeyError(
                "no acquisition certification context was registered for contract"
            ) from error

    def to_record(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "definition_sha256": self.definition_sha256,
            "registered_context_count": len(self._contexts),
        }


def feasible_slate_option_id_subsets(
    request: SlateAllocationRequest,
) -> tuple[tuple[str, ...], ...]:
    member_by_id = {value.option_id: value for value in request.slate.members}
    compatible = (
        None
        if request.pairwise_disjoint_option_id_pairs is None
        else {frozenset(value) for value in request.pairwise_disjoint_option_id_pairs}
    )
    feasible: list[tuple[str, ...]] = []
    for subset in combinations(sorted(member_by_id), request.portfolio_size):
        if not set(request.required_option_ids) <= set(subset):
            continue
        if compatible is not None and any(
            frozenset((left, right)) not in compatible
            for index, left in enumerate(subset)
            for right in subset[index + 1 :]
        ):
            continue
        members = tuple(member_by_id[value] for value in subset)
        if request.min_distinct_families is not None and len(
            {value.family for value in members}
        ) < request.min_distinct_families:
            continue
        if request.memory_dose_contract is not None:
            assessment = assess_allocated_slate_memory_dose(request, members)
            if assessment is None or not assessment.passed:
                continue
        feasible.append(subset)
    return tuple(feasible)


@dataclass(frozen=True, slots=True, eq=False)
class AcquisitionCertifiedSlateDecision:
    request: SlateAllocationRequest
    selected: tuple[AllocatedSlateMember, ...]
    reference_option_ids: tuple[str, ...]
    selected_option_ids: tuple[str, ...]
    reference_log_acquisition_value: float
    selected_log_acquisition_value: float
    certificate_margin: float
    reference_member_count: int
    feasible_slate_count: int
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
        for value in self.selected:
            if type(value) is not AllocatedSlateMember:
                raise TypeError("selected must contain exact allocated members")
            value.__post_init__()
            if value.role is not SlateAllocationRole.ACQUISITION_CERTIFIED:
                raise ValueError("selected member has a foreign acquisition role")
        for name in ("reference_option_ids", "selected_option_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))) or len(values) != (
                self.request.portfolio_size
            ):
                raise ValueError(f"{name} must be a canonical complete slate")
        if tuple(sorted(value.option_id for value in self.selected)) != (
            self.selected_option_ids
        ):
            raise ValueError("selected member identities differ from selected slate")
        for name in (
            "reference_log_acquisition_value",
            "selected_log_acquisition_value",
            "certificate_margin",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be finite")
        if not math.isclose(
            self.selected_log_acquisition_value
            - self.reference_log_acquisition_value,
            self.certificate_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("certificate margin does not close")
        if self.certificate_margin < -1e-12:
            raise ValueError("selected acquisition is below its retained reference")
        if (
            type(self.reference_member_count) is not int
            or not 0 <= self.reference_member_count <= self.request.portfolio_size
        ):
            raise ValueError("reference_member_count is outside the portfolio")
        if self.reference_member_count != len(
            set(self.selected_option_ids).intersection(self.reference_option_ids)
        ):
            raise ValueError(
                "reference_member_count must equal selected/reference overlap"
            )
        if type(self.feasible_slate_count) is not int or self.feasible_slate_count < 1:
            raise ValueError("feasible_slate_count must be positive")
        if type(self.score_request) is not FiniteAcquisitionBatchScoreRequest:
            raise TypeError("score_request must be exact")
        if type(self.score_decision) is not FiniteAcquisitionBatchScoreDecision:
            raise TypeError("score_decision must be exact")
        validate_finite_acquisition_batch_score_decision(
            self.score_request,
            self.score_decision,
        )
        if self.feasible_slate_count != len(self.score_request.slates):
            raise ValueError("feasible slate count differs from score request")
        score_by_ids = {
            value.slate.candidate_ids: value.log_acquisition_value
            for value in self.score_decision.scores
        }
        if score_by_ids.get(self.reference_option_ids) != (
            self.reference_log_acquisition_value
        ) or score_by_ids.get(self.selected_option_ids) != (
            self.selected_log_acquisition_value
        ):
            raise ValueError("certificate values differ from numerical scores")
        if self.memory_dose_assessment is not None:
            if type(self.memory_dose_assessment) is not PortfolioMemoryDoseAssessment:
                raise TypeError("memory_dose_assessment must be exact or None")
            self.memory_dose_assessment.__post_init__()
            if not self.memory_dose_assessment.passed:
                raise ValueError("selected acquisition slate violates memory dose")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        computed = hashlib.sha256(
            _DECISION_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.decision_sha256 not in ("", computed):
            raise ValueError("decision_sha256 does not authenticate the decision")
        object.__setattr__(self, "decision_sha256", computed)

    def revalidate(self) -> None:
        if type(self) is not AcquisitionCertifiedSlateDecision:
            raise TypeError("decision must be exact")
        self.__post_init__()

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "event_type": "acquisition_certified_residual_slate_allocated",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "request": self.request.to_record(),
            "request_sha256": self.request.request_sha256,
            "selected": [value.to_record() for value in self.selected],
            "reference_option_ids": list(self.reference_option_ids),
            "selected_option_ids": list(self.selected_option_ids),
            "reference_log_acquisition_value_hex": (
                self.reference_log_acquisition_value.hex()
            ),
            "selected_log_acquisition_value_hex": (
                self.selected_log_acquisition_value.hex()
            ),
            "certificate_margin_hex": self.certificate_margin.hex(),
            "reference_member_count": self.reference_member_count,
            "feasible_slate_count": self.feasible_slate_count,
            "score_request": self.score_request.to_record(),
            "score_decision": self.score_decision.to_record(),
            "memory_dose_assessment": (
                None
                if self.memory_dose_assessment is None
                else self.memory_dose_assessment.to_record()
            ),
            "certificate_scope": (
                "frozen_strictly_prior_acquisition_not_unseen_outcome"
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is AcquisitionCertifiedSlateDecision
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
class AcquisitionCertifiedSlatePolicy:
    """Choose the highest acquired feasible slate while retaining an anchor."""

    context_provider: AcquisitionCertifiedSlateContextProvider
    scorer: FiniteAcquisitionBatchScorePolicy
    exact_combination_limit: int = 250_000
    tie_tolerance: float = 1e-12
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not isinstance(
            self.context_provider,
            AcquisitionCertifiedSlateContextProvider,
        ):
            raise TypeError("context_provider must implement its exact port")
        scorer_identity = finite_acquisition_policy_identity(self.scorer)
        if type(self.exact_combination_limit) is not int or (
            self.exact_combination_limit < 1
        ):
            raise ValueError("exact_combination_limit must be positive")
        if (
            type(self.tie_tolerance) is not float
            or not math.isfinite(self.tie_tolerance)
            or self.tie_tolerance < 0.0
        ):
            raise ValueError("tie_tolerance must be finite and non-negative")
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
            "exact_combination_limit": self.exact_combination_limit,
            "tie_tolerance_hex": self.tie_tolerance.hex(),
            "reference": "maximum-anchor-count-feasible-slate",
            "reference_member_count": "selected_reference_intersection",
            "tie_break": "retain-reference-then-canonical-option-ids",
            "outcome_access": "strictly-prior-only",
        }
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(_POLICY_DOMAIN + _canonical_json(record)).hexdigest(),
        )

    def select(
        self,
        request: SlateAllocationRequest,
    ) -> AcquisitionCertifiedSlateDecision:
        self.__post_init__()
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact")
        request.revalidate()
        context = self.context_provider.context_for(
            request.slate.finite_contract_sha256
        )
        if type(context) is not AcquisitionCertifiedSlateContext:
            raise TypeError("context provider returned a foreign context")
        context.__post_init__()
        if context.finite_contract_sha256 != request.slate.finite_contract_sha256:
            raise ValueError("acquisition context names a foreign finite contract")
        if len(context.reference_option_ids) != request.portfolio_size:
            raise ValueError(
                "acquisition reference must exactly fill the evaluation portfolio"
            )
        feasible = feasible_slate_option_id_subsets(request)
        if not feasible:
            raise ValueError("proposal contains no hard-feasible evaluation slate")
        if len(feasible) > self.exact_combination_limit:
            raise ValueError("feasible slate count exceeds the exact certificate limit")
        slate_ids = {value.option_id for value in request.slate.members}
        candidate_by_id = {value.candidate_id: value for value in context.candidates}
        if not slate_ids <= set(candidate_by_id):
            raise ValueError("numerical context does not cover the proposal slate")
        reference_set = set(context.reference_option_ids)
        maximum_reference_count = max(
            len(set(value).intersection(reference_set)) for value in feasible
        )
        if maximum_reference_count != request.portfolio_size:
            raise ValueError(
                "complete numerical reference is not hard-feasible in the proposal"
            )
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
            score_request,
            score_decision,
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
        best_value = max(score_by_ids.values())
        if score_by_ids[reference] >= best_value - self.tie_tolerance:
            selected_ids = reference
        else:
            selected_ids = min(
                value
                for value in feasible
                if score_by_ids[value] >= best_value - self.tie_tolerance
            )
        member_by_id = {value.option_id: value for value in request.slate.members}
        selected_members = tuple(
            sorted(
                (member_by_id[value] for value in selected_ids),
                key=lambda value: value.model_rank,
            )
        )
        selected_value = score_by_ids[selected_ids]
        reference_value = score_by_ids[reference]
        memory_dose_assessment = (
            None
            if request.memory_dose_contract is None
            else assess_allocated_slate_memory_dose(request, selected_members)
        )
        if memory_dose_assessment is not None and not memory_dose_assessment.passed:
            raise AssertionError("certified winner violated enumerated memory dose")
        return AcquisitionCertifiedSlateDecision(
            request=request,
            selected=tuple(
                AllocatedSlateMember(
                    role=SlateAllocationRole.ACQUISITION_CERTIFIED,
                    option_id=value.option_id,
                    option_identity_sha256=value.option_identity_sha256,
                    model_rank=value.model_rank,
                    role_score=float(selected_value),
                )
                for value in selected_members
            ),
            reference_option_ids=reference,
            selected_option_ids=selected_ids,
            reference_log_acquisition_value=float(reference_value),
            selected_log_acquisition_value=float(selected_value),
            certificate_margin=float(selected_value - reference_value),
            reference_member_count=len(set(selected_ids).intersection(reference_set)),
            feasible_slate_count=len(feasible),
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
            "exact_combination_limit": self.exact_combination_limit,
            "tie_tolerance_hex": self.tie_tolerance.hex(),
        }


__all__ = [
    "AcquisitionCertifiedSlateContext",
    "AcquisitionCertifiedSlateContextProvider",
    "AcquisitionCertifiedSlateContextRegistry",
    "AcquisitionCertifiedSlateContextSink",
    "AcquisitionCertifiedSlateDecision",
    "AcquisitionCertifiedSlatePolicy",
    "POLICY_ID",
    "POLICY_VERSION",
    "feasible_slate_option_id_subsets",
    "finite_acquisition_policy_identity",
]

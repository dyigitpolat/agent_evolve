"""Application projection for phenotype-aware bounded evaluation recourse.

The policy layer intentionally knows nothing about engine outcomes.  This
adapter classifies immutable optimizer receipts into occurrence/status facts,
applies one run-wide phenotype identity policy, and invokes the objective-blind
recourse policy.  Objective values, rewards, and Pareto decisions never cross
the decision call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from agent_evolve.application.agentic_evolution import InvocationOutcome
from agent_evolve.application.budgeted_optimizer import (
    GenerationReceipt,
    validate_generation_receipt_integrity,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    BoundedEvaluationRecoursePolicy,
    EvaluationOccurrenceRole,
    EvaluationOccurrenceStatus,
    EvaluationRecourseDecision,
    PhenotypeIdentity,
    PhenotypeIdentityPolicy,
    PhenotypeOccurrence,
    PhenotypeOccurrenceLedger,
    PresealedRecoursePool,
    RecourseBudgetSnapshot,
)


EvaluationRecourseTraceSink = Callable[[Mapping[str, object]], None]


def _require_identity_policy(policy: PhenotypeIdentityPolicy) -> None:
    if not callable(getattr(policy, "identify", None)):
        raise TypeError("identity_policy must implement identify")
    if type(getattr(policy, "policy_id", None)) is not str:
        raise TypeError("identity_policy must expose a string policy_id")
    if type(getattr(policy, "policy_version", None)) is not int:
        raise TypeError("identity_policy must expose an integer policy_version")


def phenotype_occurrence(
    outcome: InvocationOutcome,
    *,
    role: EvaluationOccurrenceRole,
    identity_policy: PhenotypeIdentityPolicy,
) -> PhenotypeOccurrence:
    """Project one terminal engine outcome without reading reward/objectives."""

    if type(outcome) is not InvocationOutcome:
        raise TypeError("outcome must be an exact InvocationOutcome")
    if type(role) is not EvaluationOccurrenceRole:
        raise TypeError("role must be an exact EvaluationOccurrenceRole")
    _require_identity_policy(identity_policy)
    candidate = outcome.candidate

    if outcome.failure_stage == "llm":
        status = EvaluationOccurrenceStatus.MODEL_FAILURE
    elif outcome.failure_stage == "infrastructure":
        status = EvaluationOccurrenceStatus.INFRASTRUCTURE_FAILURE
    elif outcome.failure_stage == "materialization":
        status = EvaluationOccurrenceStatus.SYSTEM_FAILURE
    elif outcome.failure_stage == "candidate":
        status = EvaluationOccurrenceStatus.CANDIDATE_FAILURE
    elif outcome.failure_stage == "treatment_noncompliance":
        status = EvaluationOccurrenceStatus.CANDIDATE_FAILURE
    elif outcome.failure_stage is None:
        if candidate is None:
            raise ValueError("successful invocation has no candidate")
        status = (
            EvaluationOccurrenceStatus.SUCCESS
            if candidate.valid
            and candidate.operator_compliant
            and candidate.evidence_compliant
            else EvaluationOccurrenceStatus.CANDIDATE_FAILURE
        )
    else:
        raise ValueError("outcome has an unsupported failure stage")

    if candidate is None:
        candidate_id = None
        identity = None
    else:
        candidate_id = candidate.candidate_id
        identity = identity_policy.identify(candidate.configuration)
        if type(identity) is not PhenotypeIdentity:
            raise TypeError("identity policy must return an exact PhenotypeIdentity")
        if (
            identity.policy_id != identity_policy.policy_id
            or identity.policy_version != identity_policy.policy_version
        ):
            raise ValueError("identity policy returned inconsistent metadata")
    return PhenotypeOccurrence(
        trial_id=outcome.prepared.operator_invocation_id,
        role=role,
        status=status,
        candidate_id=candidate_id,
        phenotype=identity,
    )


def phenotype_ledger_from_generation(
    receipt: GenerationReceipt,
    *,
    role: EvaluationOccurrenceRole,
    identity_policy: PhenotypeIdentityPolicy,
    included_slot_ids: Sequence[str] | None = None,
) -> PhenotypeOccurrenceLedger:
    """Build a canonical ledger from all or an explicit subset of receipt slots."""

    if type(receipt) is not GenerationReceipt:
        raise TypeError("receipt must be an exact GenerationReceipt")
    validate_generation_receipt_integrity(receipt)
    if type(role) is not EvaluationOccurrenceRole:
        raise TypeError("role must be an exact EvaluationOccurrenceRole")
    _require_identity_policy(identity_policy)
    available = {result.slot.slot_id: result for result in receipt.slot_results}
    if included_slot_ids is None:
        selected_ids = tuple(available)
    else:
        if isinstance(included_slot_ids, (str, bytes)) or not isinstance(
            included_slot_ids, Sequence
        ):
            raise TypeError("included_slot_ids must be a sequence")
        selected_ids = tuple(included_slot_ids)
        if any(type(value) is not str or not value for value in selected_ids):
            raise ValueError("included_slot_ids must contain non-empty strings")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("included_slot_ids cannot contain duplicates")
        if not set(selected_ids).issubset(available):
            raise ValueError("included_slot_ids contains an unknown slot")
    occurrences = tuple(
        phenotype_occurrence(
            available[slot_id].outcome,
            role=role,
            identity_policy=identity_policy,
        )
        for slot_id in selected_ids
    )
    return PhenotypeOccurrenceLedger.build(
        occurrences,
        identity_policy=identity_policy,
    )


@dataclass(frozen=True, slots=True)
class EvaluationRecourseApplicationService:
    """Project a primary wave and make one trace-complete recourse decision."""

    identity_policy: PhenotypeIdentityPolicy
    recourse_policy: BoundedEvaluationRecoursePolicy
    trace_sink: EvaluationRecourseTraceSink | None = None

    def __post_init__(self) -> None:
        _require_identity_policy(self.identity_policy)
        if type(self.recourse_policy) is not BoundedEvaluationRecoursePolicy:
            raise TypeError(
                "recourse_policy must be an exact BoundedEvaluationRecoursePolicy"
            )
        if self.trace_sink is not None and not callable(self.trace_sink):
            raise TypeError("trace_sink must be callable")

    def decide(
        self,
        *,
        primary_receipt: GenerationReceipt,
        pool: PresealedRecoursePool,
        budget: RecourseBudgetSnapshot,
        primary_slot_ids: Sequence[str] | None = None,
    ) -> EvaluationRecourseDecision:
        ledger = phenotype_ledger_from_generation(
            primary_receipt,
            role=EvaluationOccurrenceRole.PRIMARY,
            identity_policy=self.identity_policy,
            included_slot_ids=primary_slot_ids,
        )
        decision = self.recourse_policy.decide(
            ledger=ledger,
            pool=pool,
            budget=budget,
        )
        if self.trace_sink is not None:
            for cluster in ledger.clusters:
                if cluster.successful_primary_collision_credit <= 0:
                    continue
                self.trace_sink(
                    {
                        "event_type": "phenotype_collision",
                        "ledger_sha256": ledger.ledger_sha256,
                        "phenotype_identity_sha256": (
                            cluster.phenotype.identity_sha256
                        ),
                        "trial_ids": [
                            item.trial_id.value for item in cluster.occurrences
                        ],
                        "collision_credit": (
                            cluster.successful_primary_collision_credit
                        ),
                        "zero_identity_contrast_pairs": [
                            [left.value, right.value]
                            for left, right in cluster.zero_identity_contrast_pairs
                        ],
                    }
                )
            self.trace_sink(decision.to_trace_record())
        return decision

    def append_recourse_receipt(
        self,
        primary_ledger: PhenotypeOccurrenceLedger,
        recourse_receipt: GenerationReceipt,
        *,
        recourse_slot_ids: Sequence[str] | None = None,
    ) -> PhenotypeOccurrenceLedger:
        """Publish recourse outcomes without granting them collision credit."""

        if type(primary_ledger) is not PhenotypeOccurrenceLedger:
            raise TypeError("primary_ledger must be an exact PhenotypeOccurrenceLedger")
        recourse = phenotype_ledger_from_generation(
            recourse_receipt,
            role=EvaluationOccurrenceRole.RECOURSE,
            identity_policy=self.identity_policy,
            included_slot_ids=recourse_slot_ids,
        )
        combined = PhenotypeOccurrenceLedger.build(
            (*primary_ledger.occurrences, *recourse.occurrences),
            identity_policy=self.identity_policy,
        )
        if self.trace_sink is not None:
            self.trace_sink(
                {
                    "event_type": "evaluation_recourse_completed",
                    "primary_ledger_sha256": primary_ledger.ledger_sha256,
                    "combined_ledger_sha256": combined.ledger_sha256,
                    "experiment_block_valid": combined.experiment_block_valid,
                    "recourse_trial_ids": [
                        value.value for value in combined.ignored_recourse_trial_ids
                    ],
                }
            )
        return combined


__all__ = [
    "EvaluationRecourseApplicationService",
    "EvaluationRecourseTraceSink",
    "phenotype_ledger_from_generation",
    "phenotype_occurrence",
]

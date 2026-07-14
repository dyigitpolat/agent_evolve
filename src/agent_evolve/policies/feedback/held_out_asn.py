"""Injectable six-call outcome-reflection and held-out A/S/N policy.

This experiment policy restores one concrete reflection-to-action step without
making a two-card A/S/N design a universal AgentEvolve assumption. One
interceptor converts two sealed diagnostic outcomes into exactly two
quarantined cards. A small mailbox adapter then binds the higher-score card, its
score-swapped counterpart, and a neutral sham card to three otherwise matched
held-out invocation plans.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from agent_evolve.application.agentic_evolution import (
    InvocationOutcome,
    InvocationPlan,
)
from agent_evolve.application.budgeted_optimizer import OptimizerState
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackContext,
    GenerationFeedbackReservation,
    GenerationFeedbackResult,
    validate_generation_feedback_receipt,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightMemoryEntry,
    InsightOrigin,
    QuarantineAssignmentStructuralError,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionInsightContract,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.generation_failure import (
    GenerationFailureDisposition,
    classify_generation_failure,
)
from agent_evolve.policies.memory.treatment_compliance import (
    InsightTreatmentRequirement,
    TreatmentActionBinding,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentInsightEvidence,
)


REFLECTIVE_FEEDBACK_POLICY_ID = "v7_reflective_feedback"
REFLECTIVE_FEEDBACK_POLICY_VERSION = 2
_METADATA_SCHEMA = "v7-reflected-card-batch-v2"
_REFLECTION_CONTRAST_DOMAIN = b"agent-evolve:reflection-contrast:v1\x00"
HELD_OUT_SELECTOR_POLICY_ID = "held_out_asn_origin_score_swap"
HELD_OUT_SELECTOR_POLICY_VERSION = 1
_ASSIGNMENT_COMMITMENT_DOMAIN = (
    b"agent-evolve:held-out-asn-assignment-commitment:v1\x00"
)


class ReflectiveFeedbackContractError(ValueError):
    """Diagnostic evidence or reflected cards violated the frozen contract."""


class HeldOutAssignmentUnavailableReason(str, Enum):
    """Closed, experiment-safe reasons for omitting the held-out A/S/N wave."""

    REFLECTED_CARD_BATCH_UNAVAILABLE = "reflected_card_batch_unavailable"
    EQUAL_ORIGIN_SCORES = "equal_origin_scores"
    STRUCTURALLY_INAPPLICABLE_ASSIGNMENT = "structurally_inapplicable_assignment"


class HeldOutAssignmentUnavailable(ValueError):
    """The A/S/N block cannot be constructed for one closed safe reason."""

    def __init__(
        self,
        reason: HeldOutAssignmentUnavailableReason,
        detail: str,
    ) -> None:
        if type(reason) is not HeldOutAssignmentUnavailableReason:
            raise TypeError(
                "reason must be an exact HeldOutAssignmentUnavailableReason"
            )
        if type(detail) is not str or not detail.strip() or detail != detail.strip():
            raise ValueError("detail must be non-empty canonical text")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


class OutcomeReflector(Protocol):
    """Narrow engine capability consumed by the feedback interceptor."""

    async def reflect(
        self,
        outcomes: tuple[InvocationOutcome, ...],
        *,
        label: str,
        max_insights: int,
        insight_contract: ReflectionInsightContract | None = None,
    ) -> tuple[InsightMemoryEntry, ...]: ...

    def identify_phenotype(self, configuration): ...


def reflection_contrast_id(outcome: InvocationOutcome) -> str:
    """Reproduce the engine-owned contrast identity for one-parent variation."""

    if type(outcome) is not InvocationOutcome:
        raise TypeError("outcome must be an exact InvocationOutcome")
    if len(outcome.prepared.plan.parents) != 1:
        raise ReflectiveFeedbackContractError(
            "v7 diagnostic reflection requires exactly one parent per outcome"
        )
    parent = outcome.prepared.plan.parents[0]
    return hashlib.sha256(
        _REFLECTION_CONTRAST_DOMAIN
        + outcome.prepared.operator_invocation_id.value.encode("ascii")
        + b"\x00"
        + parent.candidate_id.value.encode("ascii")
    ).hexdigest()


def _ternary_reward(outcome: InvocationOutcome) -> int:
    reward = outcome.reward
    if reward not in {-1.0, 0.0, 1.0}:
        raise ReflectiveFeedbackContractError(
            "diagnostic reward must be exactly one of -1, 0, +1"
        )
    return int(reward)


def _successful_diagnostic_outcomes(
    outcomes: tuple[InvocationOutcome, ...],
) -> tuple[InvocationOutcome, InvocationOutcome]:
    if type(outcomes) is not tuple or any(
        type(outcome) is not InvocationOutcome for outcome in outcomes
    ):
        raise TypeError("outcomes must contain exact InvocationOutcome values")
    if len(outcomes) != 2:
        raise ReflectiveFeedbackContractError(
            "v7 diagnostic reflection requires exactly two outcomes"
        )
    generations = {outcome.prepared.plan.generation for outcome in outcomes}
    if generations != {1}:
        raise ReflectiveFeedbackContractError(
            "v7 diagnostic reflection is restricted to generation one"
        )
    configuration_hashes: set[str] = set()
    parent_ids: set[CandidateId] = set()
    for outcome in outcomes:
        candidate = outcome.candidate
        if (
            outcome.failure_stage is not None
            or candidate is None
            or not candidate.valid
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
        ):
            raise ReflectiveFeedbackContractError(
                "every diagnostic outcome must be a compliant successful candidate"
            )
        if len(outcome.prepared.plan.parents) != 1:
            raise ReflectiveFeedbackContractError(
                "every diagnostic outcome must have exactly one parent"
            )
        parent_ids.add(outcome.prepared.plan.parents[0].candidate_id)
        configuration_hashes.add(candidate.occurrence.configuration_hash)
        _ternary_reward(outcome)
    if len(parent_ids) != 1:
        raise ReflectiveFeedbackContractError(
            "both diagnostic outcomes must share one frozen parent"
        )
    if len(configuration_hashes) != 2:
        raise ReflectiveFeedbackContractError(
            "diagnostic candidates must have distinct exact configurations"
        )
    return outcomes


@dataclass(frozen=True, slots=True, order=True)
class ReflectedCard:
    reference: InsightRef
    origin_contrast_id: str
    origin_transfer_score: int

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        if (
            type(self.origin_contrast_id) is not str
            or len(self.origin_contrast_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.origin_contrast_id
            )
        ):
            raise ValueError("origin_contrast_id must be a lowercase SHA-256 digest")
        if type(self.origin_transfer_score) is not int or (
            self.origin_transfer_score not in {-1, 0, 1}
        ):
            raise ValueError("origin_transfer_score must be exactly -1, 0, or +1")


@dataclass(frozen=True, slots=True)
class ReflectedCardBatch:
    source_generation: int
    diagnostic_parent_id: CandidateId
    cards: tuple[ReflectedCard, ReflectedCard]
    reflection_logical_calls: int = 1

    def __post_init__(self) -> None:
        if type(self.source_generation) is not int or self.source_generation != 1:
            raise ValueError("source_generation must be exactly one")
        if type(self.diagnostic_parent_id) is not CandidateId:
            raise TypeError("diagnostic_parent_id must be an exact CandidateId")
        CandidateId.__post_init__(self.diagnostic_parent_id)
        if (
            type(self.cards) is not tuple
            or len(self.cards) != 2
            or any(type(card) is not ReflectedCard for card in self.cards)
        ):
            raise TypeError("cards must contain exactly two ReflectedCard values")
        for card in self.cards:
            ReflectedCard.__post_init__(card)
        canonical = tuple(sorted(self.cards, key=lambda card: card.origin_contrast_id))
        if self.cards != canonical:
            raise ValueError("cards must be ordered by origin contrast identity")
        if len({card.reference for card in self.cards}) != 2:
            raise ValueError("reflected cards must use distinct insight references")
        if len({card.origin_contrast_id for card in self.cards}) != 2:
            raise ValueError("reflected cards must cite distinct origin contrasts")
        if (
            type(self.reflection_logical_calls) is not int
            or self.reflection_logical_calls <= 0
        ):
            raise ValueError("reflection_logical_calls must be positive")

    @property
    def feedback_metadata(self) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = [
            ("card_count", "2"),
            ("diagnostic_parent_id", self.diagnostic_parent_id.value),
            ("schema", _METADATA_SCHEMA),
            ("source_generation", str(self.source_generation)),
            ("status", "ready"),
        ]
        for index, card in enumerate(self.cards):
            prefix = f"card.{index}"
            rows.extend(
                (
                    (f"{prefix}.insight_id", card.reference.insight_id.value),
                    (f"{prefix}.insight_version", str(card.reference.version)),
                    (f"{prefix}.origin_contrast_id", card.origin_contrast_id),
                    (
                        f"{prefix}.origin_transfer_score",
                        str(card.origin_transfer_score),
                    ),
                )
            )
        return tuple(sorted(rows))


def _entry_for_reference(
    memory: InsightMemoryBank,
    reference: InsightRef,
) -> InsightMemoryEntry:
    matches = tuple(entry for entry in memory.entries if entry.reference == reference)
    if len(matches) != 1:
        raise ValueError(
            "assignment references an insight absent from the bound memory bank"
        )
    return matches[0]


def build_reflected_card_batch(
    *,
    outcomes: tuple[InvocationOutcome, ...],
    entries: tuple[InsightMemoryEntry, ...],
    insight_contract: ReflectionInsightContract | None = None,
    reflection_logical_calls: int = 1,
) -> ReflectedCardBatch:
    """Validate one-to-one exact citations and bind engine-derived scores."""

    first, second = _successful_diagnostic_outcomes(outcomes)
    ordered_outcomes = (first, second)
    contrast_to_outcome = {
        reflection_contrast_id(outcome): outcome for outcome in ordered_outcomes
    }
    expected_contrasts = set(contrast_to_outcome)
    if type(entries) is not tuple or any(
        type(entry) is not InsightMemoryEntry for entry in entries
    ):
        raise TypeError("entries must contain exact InsightMemoryEntry values")
    if len(entries) != 2:
        raise ReflectiveFeedbackContractError(
            "reflection must yield exactly two accepted insight entries"
        )
    if type(reflection_logical_calls) is not int or reflection_logical_calls <= 0:
        raise ValueError("reflection_logical_calls must be positive")

    available_sets = tuple(
        set(entry.evidence_lineage.available_contrast_ids)
        for entry in entries
        if entry.evidence_lineage is not None
    )
    full_batch_lineage = len(available_sets) == len(entries) and all(
        available == expected_contrasts for available in available_sets
    )
    singleton_lineage = len(available_sets) == len(entries) and all(
        len(available) == 1 for available in available_sets
    )
    if full_batch_lineage == singleton_lineage:
        raise ReflectiveFeedbackContractError(
            "reflected cards must use one consistent full-batch or singleton lineage mode"
        )

    cards: list[ReflectedCard] = []
    cited: set[str] = set()
    for entry in entries:
        if insight_contract is not None:
            try:
                validate_reflection_insight_draft(
                    entry.draft,
                    insight_contract,
                )
            except (TypeError, ValueError) as exc:
                raise ReflectiveFeedbackContractError(
                    "reflected card violates the actionable insight contract"
                ) from exc
        if (
            entry.origin is not InsightOrigin.REFLECTION
            or entry.lifecycle_state is not InsightLifecycleState.QUARANTINED
            or entry.retrievable
            or entry.evidence_lineage is None
        ):
            raise ReflectiveFeedbackContractError(
                "every reflected card must be a non-retrievable quarantine entry"
            )
        lineage = entry.evidence_lineage
        if len(lineage.cited_contrast_ids) != 1:
            raise ReflectiveFeedbackContractError(
                "each reflected card must cite exactly one full origin contrast"
            )
        contrast_id = lineage.cited_contrast_ids[0]
        if singleton_lineage and lineage.available_contrast_ids != (contrast_id,):
            raise ReflectiveFeedbackContractError(
                "singleton reflected-card lineage differs from its exact citation"
            )
        if contrast_id not in expected_contrasts or contrast_id in cited:
            raise ReflectiveFeedbackContractError(
                "reflected cards must form a one-to-one exact contrast assignment"
            )
        if entry.draft.evidence_contrast_ids != (contrast_id,):
            raise ReflectiveFeedbackContractError(
                "reflected draft citation differs from its evidence lineage"
            )
        outcome = contrast_to_outcome[contrast_id]
        candidate = outcome.candidate
        assert candidate is not None
        expected_operator_ids = (outcome.prepared.operator_invocation_id,)
        expected_candidate_ids = tuple(
            sorted(
                (
                    outcome.prepared.plan.parents[0].candidate_id,
                    candidate.candidate_id,
                )
            )
        )
        if lineage.source_operator_invocation_ids != expected_operator_ids or (
            lineage.source_candidate_ids != expected_candidate_ids
        ):
            raise ReflectiveFeedbackContractError(
                "reflected evidence lineage differs from its cited contrast"
            )
        cited.add(contrast_id)
        cards.append(
            ReflectedCard(
                reference=entry.reference,
                origin_contrast_id=contrast_id,
                origin_transfer_score=_ternary_reward(outcome),
            )
        )
    if cited != expected_contrasts:
        raise ReflectiveFeedbackContractError(
            "reflection omitted a diagnostic origin contrast"
        )
    return ReflectedCardBatch(
        source_generation=1,
        diagnostic_parent_id=ordered_outcomes[0].prepared.plan.parents[0].candidate_id,
        cards=tuple(sorted(cards, key=lambda card: card.origin_contrast_id)),  # type: ignore[arg-type]
        reflection_logical_calls=reflection_logical_calls,
    )


class ReflectedCardMailbox:
    """Write-once typed handoff from feedback to the next planner call."""

    def __init__(self) -> None:
        self._batches: dict[int, ReflectedCardBatch] = {}

    def publish(self, batch: ReflectedCardBatch) -> None:
        if type(batch) is not ReflectedCardBatch:
            raise TypeError("batch must be an exact ReflectedCardBatch")
        ReflectedCardBatch.__post_init__(batch)
        if batch.source_generation in self._batches:
            raise ReflectiveFeedbackContractError(
                "a reflected card batch was already published for this generation"
            )
        self._batches[batch.source_generation] = batch

    def read_verified(
        self,
        *,
        state: OptimizerState,
        source_generation: int = 1,
    ) -> ReflectedCardBatch:
        """Read only if the next planner state authenticates the same metadata."""

        if type(state) is not OptimizerState:
            raise TypeError("state must be an exact OptimizerState")
        if type(source_generation) is not int or source_generation != 1:
            raise ValueError("source_generation must be exactly one")
        receipts = tuple(
            receipt
            for receipt in state.feedback_receipts
            if receipt.generation == source_generation
        )
        if len(receipts) != 1:
            raise ReflectiveFeedbackContractError(
                "planner state lacks one exact source feedback receipt"
            )
        receipt = receipts[0]
        validate_generation_feedback_receipt(receipt)
        if (
            receipt.policy_id != REFLECTIVE_FEEDBACK_POLICY_ID
            or receipt.policy_version != REFLECTIVE_FEEDBACK_POLICY_VERSION
        ):
            raise ReflectiveFeedbackContractError(
                "planner feedback receipt names a different feedback policy"
            )
        try:
            batch = self._batches[source_generation]
        except KeyError as exc:
            metadata = dict(receipt.result_metadata)
            status = metadata.get("status")
            reason = metadata.get("reason")
            expected_calls = {
                "diagnostic_rejected": 0,
                "reflection_failed": receipt.reserved_logical_llm_calls,
                "reflection_rejected": receipt.reserved_logical_llm_calls,
            }.get(status)
            if (
                expected_calls is None
                or type(reason) is not str
                or not reason
                or receipt.used_logical_llm_calls != expected_calls
                or receipt.result_metadata != _status_metadata(status, reason)
            ):
                raise ReflectiveFeedbackContractError(
                    "a missing card batch lacks one authenticated unavailable status"
                ) from exc
            raise HeldOutAssignmentUnavailable(
                HeldOutAssignmentUnavailableReason.REFLECTED_CARD_BATCH_UNAVAILABLE,
                f"no accepted reflected card batch is available after {status}",
            ) from exc
        if (
            receipt.used_logical_llm_calls != batch.reflection_logical_calls
            or receipt.result_metadata != batch.feedback_metadata
        ):
            raise ReflectiveFeedbackContractError(
                "planner feedback receipt differs from the typed card mailbox"
            )
        return batch


def _status_metadata(status: str, reason: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("card_count", "0"),
                ("reason", reason),
                ("schema", _METADATA_SCHEMA),
                ("source_generation", "1"),
                ("status", status),
            )
        )
    )


@dataclass(slots=True)
class G1ReflectionFeedbackInterceptor:
    """Spend the reserved reflection calls after a valid sealed G1 block."""

    engine: OutcomeReflector
    mailbox: ReflectedCardMailbox
    diagnostic_slot_ids: tuple[str, str] = ("D-S", "D-T")
    reflection_label: str = "v7_g1_outcome_reflection"
    required_metric_ids: tuple[str, ...] = ()
    allowed_option_families: tuple[str, ...] = ()
    allowed_option_ids: tuple[str, ...] = ()
    reflection_logical_calls: int = 1

    def __post_init__(self) -> None:
        if not callable(getattr(self.engine, "reflect", None)):
            raise TypeError("engine must provide async reflect")
        if not callable(getattr(self.engine, "identify_phenotype", None)):
            raise TypeError("engine must provide phenotype identity")
        if type(self.mailbox) is not ReflectedCardMailbox:
            raise TypeError("mailbox must be an exact ReflectedCardMailbox")
        if (
            type(self.diagnostic_slot_ids) is not tuple
            or len(self.diagnostic_slot_ids) != 2
            or any(
                type(value) is not str or not value
                for value in self.diagnostic_slot_ids
            )
            or len(set(self.diagnostic_slot_ids)) != 2
        ):
            raise ValueError("diagnostic_slot_ids must contain two distinct IDs")
        if type(self.reflection_label) is not str or not self.reflection_label:
            raise ValueError("reflection_label must be non-empty")
        if bool(self.required_metric_ids) != bool(self.allowed_option_families):
            raise ValueError(
                "advanced reflection requires metrics and option families together"
            )
        if self.required_metric_ids:
            ReflectionInsightContract(
                required_metric_ids=self.required_metric_ids,
                allowed_option_families=self.allowed_option_families,
                allowed_option_ids=self.allowed_option_ids,
            )
        elif self.allowed_option_ids:
            raise ValueError("exact option IDs require an advanced reflection contract")
        if (
            type(self.reflection_logical_calls) is not int
            or self.reflection_logical_calls <= 0
        ):
            raise ValueError("reflection_logical_calls must be positive")

    @property
    def insight_contract(self) -> ReflectionInsightContract | None:
        if not self.required_metric_ids:
            return None
        return ReflectionInsightContract(
            required_metric_ids=self.required_metric_ids,
            allowed_option_families=self.allowed_option_families,
            allowed_option_ids=self.allowed_option_ids,
        )

    def reserve(
        self,
        *,
        state: OptimizerState,
        plan,
    ) -> GenerationFeedbackReservation:
        del state
        generation = plan.generation
        return GenerationFeedbackReservation(
            policy_id=REFLECTIVE_FEEDBACK_POLICY_ID,
            policy_version=REFLECTIVE_FEEDBACK_POLICY_VERSION,
            logical_llm_calls=(
                self.reflection_logical_calls if generation == 1 else 0
            ),
            metadata=(("scheduled_generation", "1"),),
        )

    async def after_generation(
        self,
        context: GenerationFeedbackContext,
    ) -> GenerationFeedbackResult:
        if context.plan.generation != 1:
            return GenerationFeedbackResult(
                logical_llm_calls_used=0,
                metadata=_status_metadata("not_scheduled", "generation_is_not_one"),
            )
        results_by_slot = {
            result.slot.slot_id: result.outcome
            for result in context.generation_receipt.slot_results
        }
        if set(results_by_slot) != set(self.diagnostic_slot_ids):
            return GenerationFeedbackResult(
                logical_llm_calls_used=0,
                metadata=_status_metadata(
                    "diagnostic_rejected",
                    "diagnostic_slot_ids_differ",
                ),
            )
        outcomes = tuple(
            results_by_slot[slot_id] for slot_id in self.diagnostic_slot_ids
        )
        try:
            _successful_diagnostic_outcomes(outcomes)  # type: ignore[arg-type]
        except (TypeError, ReflectiveFeedbackContractError) as exc:
            return GenerationFeedbackResult(
                logical_llm_calls_used=0,
                metadata=_status_metadata(
                    "diagnostic_rejected",
                    type(exc).__name__,
                ),
            )
        phenotype_ids = {
            self.engine.identify_phenotype(
                outcome.candidate.configuration
            ).identity_sha256
            for outcome in outcomes
            if outcome.candidate is not None
        }
        if len(phenotype_ids) != 2:
            return GenerationFeedbackResult(
                logical_llm_calls_used=0,
                metadata=_status_metadata(
                    "diagnostic_rejected",
                    "diagnostic_phenotype_collision",
                ),
            )
        try:
            contract = self.insight_contract
            if contract is None:
                entries = await self.engine.reflect(
                    outcomes,  # type: ignore[arg-type]
                    label=self.reflection_label,
                    max_insights=2,
                    min_insights=2,
                )
            else:
                entries = await self.engine.reflect(
                    outcomes,  # type: ignore[arg-type]
                    label=self.reflection_label,
                    max_insights=2,
                    min_insights=2,
                    insight_contract=contract,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if (
                classify_generation_failure(exc)
                is not GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
            ):
                # Credential, source-integrity, queue publication, provider,
                # and programming failures are fatal experiment failures.  An
                # untyped exception must never masquerade as a clean no-card
                # condition.
                raise
            return GenerationFeedbackResult(
                logical_llm_calls_used=self.reflection_logical_calls,
                metadata=_status_metadata(
                    "reflection_failed",
                    GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE.value,
                ),
            )
        try:
            batch = build_reflected_card_batch(
                outcomes=outcomes,  # type: ignore[arg-type]
                entries=entries,
                insight_contract=self.insight_contract,
                reflection_logical_calls=self.reflection_logical_calls,
            )
        except (TypeError, ReflectiveFeedbackContractError) as exc:
            return GenerationFeedbackResult(
                logical_llm_calls_used=self.reflection_logical_calls,
                metadata=_status_metadata(
                    "reflection_rejected",
                    type(exc).__name__,
                ),
            )
        self.mailbox.publish(batch)
        return GenerationFeedbackResult(
            logical_llm_calls_used=self.reflection_logical_calls,
            metadata=batch.feedback_metadata,
        )


class HeldOutArm(str, Enum):
    ADAPTIVE = "adaptive"
    SCORE_SWAPPED = "score_swapped"
    SHAM = "sham"


@dataclass(frozen=True, slots=True)
class HeldOutArmAssignment:
    arm: HeldOutArm
    reference: InsightRef
    origin_transfer_score: int | None
    assigned_selection_score: int | None


@dataclass(frozen=True, slots=True)
class HeldOutASNAssignments:
    adaptive: HeldOutArmAssignment
    score_swapped: HeldOutArmAssignment
    sham: HeldOutArmAssignment

    def __post_init__(self) -> None:
        if self.adaptive.arm is not HeldOutArm.ADAPTIVE:
            raise ValueError("adaptive assignment has the wrong arm")
        if self.score_swapped.arm is not HeldOutArm.SCORE_SWAPPED:
            raise ValueError("score-swapped assignment has the wrong arm")
        if self.sham.arm is not HeldOutArm.SHAM:
            raise ValueError("sham assignment has the wrong arm")
        if self.adaptive.reference == self.score_swapped.reference:
            raise ValueError(
                "adaptive and score-swapped arms must select different cards"
            )


@dataclass(frozen=True, slots=True)
class HeldOutScoreMapEntry:
    """One card's immutable origin and assigned score in a selector map."""

    reference: InsightRef
    origin_contrast_id: str
    origin_transfer_score: int
    assigned_selection_score: int

    def __post_init__(self) -> None:
        ReflectedCard(
            reference=self.reference,
            origin_contrast_id=self.origin_contrast_id,
            origin_transfer_score=self.origin_transfer_score,
        )
        if type(self.assigned_selection_score) is not int or (
            self.assigned_selection_score not in {-1, 0, 1}
        ):
            raise ValueError(
                "assigned_selection_score must be exactly -1, 0, or +1"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "insight_id": self.reference.insight_id.value,
            "insight_version": self.reference.version,
            "origin_contrast_id": self.origin_contrast_id,
            "origin_transfer_score": self.origin_transfer_score,
            "assigned_selection_score": self.assigned_selection_score,
        }


def _reference_record(reference: InsightRef) -> dict[str, object]:
    if type(reference) is not InsightRef:
        raise TypeError("reference must be an exact InsightRef")
    InsightRef.__post_init__(reference)
    return {
        "insight_id": reference.insight_id.value,
        "insight_version": reference.version,
    }


@dataclass(frozen=True, slots=True)
class HeldOutASNAssignmentCommitment:
    """Authenticated full true/swapped selector assignment for one A/S/N wave.

    Both two-card maps are retained so a downstream prequeue gate can prove
    that the prompts it is about to release correspond to the prospectively
    selected correct, score-swapped, and sham references.  The commitment is
    benchmark-neutral: domains choose cards and actions, while this record
    binds only selector semantics and immutable insight references.
    """

    true_score_map: tuple[HeldOutScoreMapEntry, HeldOutScoreMapEntry]
    score_swapped_map: tuple[HeldOutScoreMapEntry, HeldOutScoreMapEntry]
    common_score_multiset: tuple[int, int]
    adaptive_reference: InsightRef
    score_swapped_reference: InsightRef
    sham_reference: InsightRef
    selector_policy_id: str = HELD_OUT_SELECTOR_POLICY_ID
    selector_policy_version: int = HELD_OUT_SELECTOR_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.selector_policy_id != HELD_OUT_SELECTOR_POLICY_ID:
            raise ValueError("assignment names a different held-out selector policy")
        if self.selector_policy_version != HELD_OUT_SELECTOR_POLICY_VERSION:
            raise ValueError("assignment names a different selector policy version")
        maps = (self.true_score_map, self.score_swapped_map)
        if any(
            type(score_map) is not tuple
            or len(score_map) != 2
            or any(type(item) is not HeldOutScoreMapEntry for item in score_map)
            for score_map in maps
        ):
            raise TypeError("each score map must contain two exact map entries")
        for score_map in maps:
            for item in score_map:
                item.__post_init__()
            if score_map != tuple(
                sorted(
                    score_map,
                    key=lambda item: (
                        item.reference.insight_id.value,
                        item.reference.version,
                    ),
                )
            ):
                raise ValueError("score maps must be canonically reference-ordered")
        true_by_reference = {item.reference: item for item in self.true_score_map}
        swapped_by_reference = {
            item.reference: item for item in self.score_swapped_map
        }
        if len(true_by_reference) != 2 or set(true_by_reference) != set(
            swapped_by_reference
        ):
            raise ValueError("true and swapped maps must bind the same two cards")
        for reference, true_entry in true_by_reference.items():
            swapped_entry = swapped_by_reference[reference]
            if (
                true_entry.origin_contrast_id != swapped_entry.origin_contrast_id
                or true_entry.origin_transfer_score
                != swapped_entry.origin_transfer_score
                or true_entry.assigned_selection_score
                != true_entry.origin_transfer_score
            ):
                raise ValueError("score maps changed immutable card provenance")
        if type(self.common_score_multiset) is not tuple or (
            len(self.common_score_multiset) != 2
        ):
            raise TypeError("common_score_multiset must contain exactly two scores")
        origin_scores = tuple(
            sorted(item.origin_transfer_score for item in self.true_score_map)
        )
        if self.common_score_multiset != origin_scores:
            raise ValueError("common score multiset differs from origin scores")
        if tuple(
            sorted(item.assigned_selection_score for item in self.true_score_map)
        ) != self.common_score_multiset or tuple(
            sorted(
                item.assigned_selection_score for item in self.score_swapped_map
            )
        ) != self.common_score_multiset:
            raise ValueError("true and swapped maps must share one score multiset")
        if all(
            true_by_reference[reference].assigned_selection_score
            == swapped_by_reference[reference].assigned_selection_score
            for reference in true_by_reference
        ):
            raise ValueError("score-swapped map must actually exchange the scores")
        for reference in (
            self.adaptive_reference,
            self.score_swapped_reference,
            self.sham_reference,
        ):
            _reference_record(reference)
        if self.adaptive_reference not in true_by_reference:
            raise ValueError("adaptive reference is absent from the true score map")
        if self.score_swapped_reference not in true_by_reference:
            raise ValueError(
                "score-swapped reference is absent from the reflected score maps"
            )
        if self.adaptive_reference == self.score_swapped_reference:
            raise ValueError("adaptive and score-swapped references must differ")
        if self.sham_reference in true_by_reference:
            raise ValueError("sham reference must be outside both reflected maps")
        high_score = max(self.common_score_multiset)
        if (
            true_by_reference[self.adaptive_reference].assigned_selection_score
            != high_score
            or swapped_by_reference[
                self.score_swapped_reference
            ].assigned_selection_score
            != high_score
        ):
            raise ValueError("chosen A/S references do not receive the high score")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "selector_policy_id": self.selector_policy_id,
            "selector_policy_version": self.selector_policy_version,
            "true_score_map": [item.to_record() for item in self.true_score_map],
            "score_swapped_map": [
                item.to_record() for item in self.score_swapped_map
            ],
            "common_score_multiset": list(self.common_score_multiset),
            "chosen_references": {
                "adaptive": _reference_record(self.adaptive_reference),
                "score_swapped": _reference_record(
                    self.score_swapped_reference
                ),
                "sham": _reference_record(self.sham_reference),
            },
        }

    @property
    def assignment_sha256(self) -> str:
        payload = json.dumps(
            self._unsigned_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_ASSIGNMENT_COMMITMENT_DOMAIN + payload).hexdigest()

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {**record, "assignment_sha256": self.assignment_sha256}


@dataclass(frozen=True, slots=True)
class HeldOutASNPlanSet:
    assignments: HeldOutASNAssignments
    assignment_commitment: HeldOutASNAssignmentCommitment
    adaptive: InvocationPlan
    score_swapped: InvocationPlan
    sham: InvocationPlan


def register_neutral_sham_card(
    *,
    memory: InsightMemoryBank,
    affected_paths: tuple[str, ...],
    applicable_operator_kinds: tuple[str, ...],
    insight_contract: ReflectionInsightContract | None = None,
) -> InsightRef:
    """Register a schema-matched exploratory hypothesis with neutral effects."""

    if insight_contract is not None:
        if type(insight_contract) is not ReflectionInsightContract:
            raise TypeError(
                "insight_contract must be an exact ReflectionInsightContract"
            )
        ReflectionInsightContract.__post_init__(insight_contract)
    entry, _ = memory.add(
        InsightDraft(
            claim=(
                "The supplied palette contains legal coordinated interventions "
                "in each listed option family."
            ),
            trigger=("The frozen parent admits the listed finite action families."),
            mechanism=(
                "Each finite option specifies an internally consistent coordinated "
                "change within its named family."
            ),
            affected_paths=affected_paths,
            evidence_summary="Schema-matched factual palette description.",
            confidence=0.5,
            evidence_contrast_ids=(),
            effect_predictions=(
                ()
                if insight_contract is None
                else tuple(
                    MetricEffectPrediction(
                        metric_id=metric_id,
                        direction=MetricEffectDirection.UNKNOWN,
                    )
                    for metric_id in insight_contract.required_metric_ids
                )
            ),
            recommended_option_families=(
                ()
                if insight_contract is None
                else insight_contract.allowed_option_families
            ),
            recommended_option_ids=(
                ()
                if insight_contract is None
                else insight_contract.allowed_option_ids
            ),
            action_template=(
                None
                if insight_contract is None
                else (
                    "A legal option is represented by its named family and sealed "
                    "option identifier."
                )
            ),
            falsification_condition=(
                None
                if insight_contract is None
                else (
                    "The two named held-out metric values are the complete empirical "
                    "check for the intervention."
                )
            ),
        ),
        applicable_operator_kinds=applicable_operator_kinds,
        origin=InsightOrigin.MANUAL,
    )
    if insight_contract is not None:
        validate_reflection_insight_draft(
            entry.draft,
            insight_contract,
            allow_all_unknown=True,
            allow_missing_evidence=True,
        )
        if any(
            prediction.direction is not MetricEffectDirection.UNKNOWN
            for prediction in entry.draft.effect_predictions
        ):
            raise ValueError("the neutral sham must use only unknown predictions")
        if (
            entry.draft.recommended_option_families
            != insight_contract.allowed_option_families
        ):
            raise ValueError(
                "the neutral sham must expose the complete option-family vocabulary"
            )
        if entry.draft.recommended_option_ids != insight_contract.allowed_option_ids:
            raise ValueError(
                "the neutral sham must expose the precommitted exact option IDs"
            )
    return entry.reference


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported JSON path segment")
    return "".join(parts)


def _editable_paths(plan: InvocationPlan) -> tuple[str, ...] | None:
    contract = plan.mutation_contract
    if contract is not None:
        return tuple(_path_text(path) for path in contract.editable_paths)
    if plan.allowed_top_level:
        return tuple(f"$.{key}" for key in plan.allowed_top_level)
    return None


def _exact_treatment_requirement(
    *,
    memory: InsightMemoryBank,
    reference: InsightRef,
    plan: InvocationPlan,
    assignment_role: TreatmentAssignmentRole,
) -> InsightTreatmentRequirement:
    """Bind one assigned card to its exact options in this parent palette."""

    entry = _entry_for_reference(memory, reference)
    evidence = TreatmentInsightEvidence(
        reference=entry.reference,
        insight_content_sha256=entry.draft.content_sha256,
        applicable_operator_kinds=entry.applicable_operator_kinds,
        affected_paths=tuple(sorted(entry.draft.affected_paths)),
        recommended_option_families=tuple(
            sorted(entry.draft.recommended_option_families)
        ),
        recommended_option_ids=tuple(sorted(entry.draft.recommended_option_ids)),
    )
    if not evidence.recommended_option_ids:
        raise ValueError(
            "exact treatment assignment requires card-recommended option IDs"
        )
    contract = plan.finite_variation_contract
    if contract is None:
        raise ValueError("exact treatment assignment requires a finite contract")
    allowed_actions = tuple(
        sorted(
            (
                TreatmentActionBinding(
                    option_id=option_id,
                    option_identity_sha256=contract.resolve(
                        option_id
                    ).identity_sha256,
                )
                for option_id in evidence.recommended_option_ids
            ),
            key=lambda value: (value.option_id, value.option_identity_sha256),
        )
    )
    return InsightTreatmentRequirement(
        insight_bindings=(evidence.binding(),),
        finite_contract_sha256=contract.identity_sha256,
        allowed_actions=allowed_actions,
        claim_mode=TreatmentClaimMode.EXACT_REQUIRED,
        assignment_role=assignment_role,
    )


def _plan_shape(plan: InvocationPlan) -> tuple[object, ...]:
    return (
        plan.operator_kind,
        plan.parents,
        plan.generation,
        plan.common_ancestor,
        plan.allowed_top_level,
        plan.phase,
        plan.mutation_contract,
        plan.mutation_response_mode,
        plan.atomic_replacement_options,
        plan.finite_variation_contract,
        plan.memory_subset_size,
    )


@dataclass(frozen=True, slots=True)
class HeldOutASNPlannerAdapter:
    """Resolve correct/swapped/sham cards and bind three matched base plans."""

    mailbox: ReflectedCardMailbox
    memory: InsightMemoryBank
    sham_reference: InsightRef

    def __post_init__(self) -> None:
        if type(self.mailbox) is not ReflectedCardMailbox:
            raise TypeError("mailbox must be an exact ReflectedCardMailbox")
        if type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if type(self.sham_reference) is not InsightRef:
            raise TypeError("sham_reference must be an exact InsightRef")
        sham = _entry_for_reference(self.memory, self.sham_reference)
        if (
            sham.origin is not InsightOrigin.MANUAL
            or sham.lifecycle_state is not InsightLifecycleState.QUARANTINED
            or sham.evidence_lineage is not None
            or sham.draft.evidence_contrast_ids
        ):
            raise ValueError(
                "sham_reference must identify a manual, evidence-free quarantine card"
            )

    def resolve(self, state: OptimizerState) -> HeldOutASNAssignments:
        batch = self.mailbox.read_verified(state=state)
        if batch.cards[0].origin_transfer_score == batch.cards[1].origin_transfer_score:
            raise HeldOutAssignmentUnavailable(
                HeldOutAssignmentUnavailableReason.EQUAL_ORIGIN_SCORES,
                "tied origin scores provide no correct-vs-swapped contrast",
            )
        low, high = sorted(
            batch.cards,
            key=lambda card: (card.origin_transfer_score, card.origin_contrast_id),
        )
        if self.sham_reference in {low.reference, high.reference}:
            raise ReflectiveFeedbackContractError(
                "sham card must differ from both outcome-grounded cards"
            )
        return HeldOutASNAssignments(
            adaptive=HeldOutArmAssignment(
                HeldOutArm.ADAPTIVE,
                high.reference,
                high.origin_transfer_score,
                high.origin_transfer_score,
            ),
            score_swapped=HeldOutArmAssignment(
                HeldOutArm.SCORE_SWAPPED,
                low.reference,
                low.origin_transfer_score,
                high.origin_transfer_score,
            ),
            sham=HeldOutArmAssignment(
                HeldOutArm.SHAM,
                self.sham_reference,
                None,
                None,
            ),
        )

    def bind_plans(
        self,
        state: OptimizerState,
        *,
        adaptive_base: InvocationPlan,
        score_swapped_base: InvocationPlan,
        sham_base: InvocationPlan,
    ) -> HeldOutASNPlanSet:
        bases = (adaptive_base, score_swapped_base, sham_base)
        if any(type(plan) is not InvocationPlan for plan in bases):
            raise TypeError("held-out bases must be exact InvocationPlan values")
        for plan in bases:
            InvocationPlan.__post_init__(plan)
            if (
                plan.use_memory
                or plan.quarantine_test_insights
                or plan.resolved_insight_assignment is not None
            ):
                raise ValueError(
                    "held-out base plans must not carry memory assignments"
                )
        if len({plan.label for plan in bases}) != 3:
            raise ValueError("held-out base plan labels must be distinct")
        first_shape = _plan_shape(bases[0])
        if any(_plan_shape(plan) != first_shape for plan in bases[1:]):
            raise ValueError("held-out base plans differ outside label and assignment")
        if adaptive_base.generation != 2 or state.generation != 1:
            raise ValueError("the held-out A/S/N block must be generation two")

        batch = self.mailbox.read_verified(state=state)
        if len(adaptive_base.parents) != 1:
            raise ValueError("held-out A/S/N plans require exactly one shared parent")
        if adaptive_base.parents[0].candidate_id == batch.diagnostic_parent_id:
            raise ValueError("held-out parent must differ from the diagnostic parent")
        assignments = self.resolve(state)
        references = (
            assignments.adaptive.reference,
            assignments.score_swapped.reference,
            assignments.sham.reference,
        )
        for plan, reference in zip(bases, references, strict=True):
            try:
                self.memory.validate_quarantine_test_assignment(
                    (reference,),
                    operator_kind=plan.operator_kind.value,
                    editable_paths=_editable_paths(plan),
                )
            except QuarantineAssignmentStructuralError as exc:
                raise HeldOutAssignmentUnavailable(
                    HeldOutAssignmentUnavailableReason.STRUCTURALLY_INAPPLICABLE_ASSIGNMENT,
                    (
                        "one otherwise valid quarantine card is structurally "
                        "inapplicable to its matched held-out plan"
                    ),
                ) from exc
        try:
            treatment_requirements = tuple(
                _exact_treatment_requirement(
                    memory=self.memory,
                    reference=reference,
                    plan=plan,
                    assignment_role=(
                        TreatmentAssignmentRole.SHAM_CONTROL
                        if index == 2
                        else TreatmentAssignmentRole.ACTIVE
                    ),
                )
                for index, (plan, reference) in enumerate(
                    zip(bases, references, strict=True)
                )
            )
        except (TypeError, ValueError) as exc:
            raise HeldOutAssignmentUnavailable(
                HeldOutAssignmentUnavailableReason.STRUCTURALLY_INAPPLICABLE_ASSIGNMENT,
                (
                    "one assigned card lacks an exact action binding in its "
                    "matched held-out finite palette"
                ),
            ) from exc
        true_entries = tuple(
            sorted(
                (
                    HeldOutScoreMapEntry(
                        reference=card.reference,
                        origin_contrast_id=card.origin_contrast_id,
                        origin_transfer_score=card.origin_transfer_score,
                        assigned_selection_score=card.origin_transfer_score,
                    )
                    for card in batch.cards
                ),
                key=lambda item: (
                    item.reference.insight_id.value,
                    item.reference.version,
                ),
            )
        )
        swapped_scores = tuple(
            reversed(tuple(item.origin_transfer_score for item in true_entries))
        )
        score_swapped_entries = tuple(
            replace(item, assigned_selection_score=score)
            for item, score in zip(true_entries, swapped_scores, strict=True)
        )
        assignment_commitment = HeldOutASNAssignmentCommitment(
            true_score_map=true_entries,  # type: ignore[arg-type]
            score_swapped_map=score_swapped_entries,  # type: ignore[arg-type]
            common_score_multiset=tuple(  # type: ignore[arg-type]
                sorted(item.origin_transfer_score for item in true_entries)
            ),
            adaptive_reference=assignments.adaptive.reference,
            score_swapped_reference=assignments.score_swapped.reference,
            sham_reference=assignments.sham.reference,
        )
        return HeldOutASNPlanSet(
            assignments=assignments,
            assignment_commitment=assignment_commitment,
            adaptive=replace(
                adaptive_base,
                quarantine_test_insights=(assignments.adaptive.reference,),
                insight_treatment_requirement=treatment_requirements[0],
            ),
            score_swapped=replace(
                score_swapped_base,
                quarantine_test_insights=(assignments.score_swapped.reference,),
                insight_treatment_requirement=treatment_requirements[1],
            ),
            sham=replace(
                sham_base,
                quarantine_test_insights=(assignments.sham.reference,),
                insight_treatment_requirement=treatment_requirements[2],
            ),
        )


__all__ = [
    "G1ReflectionFeedbackInterceptor",
    "HeldOutASNAssignments",
    "HeldOutASNAssignmentCommitment",
    "HeldOutASNPlanSet",
    "HeldOutASNPlannerAdapter",
    "HeldOutArm",
    "HeldOutArmAssignment",
    "HeldOutScoreMapEntry",
    "HELD_OUT_SELECTOR_POLICY_ID",
    "HELD_OUT_SELECTOR_POLICY_VERSION",
    "HeldOutAssignmentUnavailable",
    "HeldOutAssignmentUnavailableReason",
    "REFLECTIVE_FEEDBACK_POLICY_ID",
    "REFLECTIVE_FEEDBACK_POLICY_VERSION",
    "ReflectedCard",
    "ReflectedCardBatch",
    "ReflectedCardMailbox",
    "ReflectiveFeedbackContractError",
    "build_reflected_card_batch",
    "reflection_contrast_id",
    "register_neutral_sham_card",
]

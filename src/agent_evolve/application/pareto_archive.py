"""Small event-auditable Pareto archive for explicit agentic evolution.

The archive is deliberately separate from generation policy.  It consumes evaluated
``EvolutionCandidate`` occurrences, applies explicit admissibility gates, deduplicates
configurations across the whole run, and maintains one deterministic representative
for exact objective-vector ties.  Every state transition is returned as an immutable
decision that a runner can emit directly into its trace.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.outcome_relation import (
    OutcomeRelation,
    OutcomeRelationPolicyBinding,
    objective_pareto_outcome_binding,
)
from agent_evolve.core.problem import (
    ObjectiveSpec,
    ProblemContractError,
    normalize_objective_values,
    validate_objective_specs,
)
from agent_evolve.core.results import dominates
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256


_CANDIDATE_HASH_DOMAIN = b"agent-evolve:pareto-candidate:v1\x00"


class ParetoDecisionAction(str, Enum):
    """The archive state transition described by one decision."""

    ADMITTED = "admitted"
    REMOVED = "removed"
    REJECTED = "rejected"


class ParetoDecisionReason(str, Enum):
    """Machine-stable explanations for every archive decision."""

    ADMITTED_NONDOMINATED = "admitted_nondominated"
    ADMITTED_TIE_BREAK_REPLACEMENT = "admitted_tie_break_replacement"
    REMOVED_DOMINATED = "removed_dominated"
    REMOVED_TIE_BREAK = "removed_tie_break"
    REJECTED_INVALID = "rejected_invalid"
    REJECTED_OPERATOR_NONCOMPLIANT = "rejected_operator_noncompliant"
    REJECTED_EVIDENCE_NONCOMPLIANT = "rejected_evidence_noncompliant"
    REJECTED_OBJECTIVE_CONTRACT = "rejected_objective_contract"
    REJECTED_DUPLICATE_CANDIDATE = "rejected_duplicate_candidate"
    REJECTED_CANDIDATE_ID_CONFLICT = "rejected_candidate_id_conflict"
    REJECTED_DUPLICATE_CONFIGURATION = "rejected_duplicate_configuration"
    REJECTED_DOMINATED = "rejected_dominated"
    REJECTED_OBJECTIVE_TIE = "rejected_objective_tie"
    ADMITTED_RELATION_FRONT = "admitted_relation_front"
    ADMITTED_EQUIVALENCE_TIE_BREAK = "admitted_equivalence_tie_break"
    REMOVED_WORSE_RELATION = "removed_worse_relation"
    REMOVED_EQUIVALENCE_TIE_BREAK = "removed_equivalence_tie_break"
    REJECTED_WORSE_RELATION = "rejected_worse_relation"
    REJECTED_EQUIVALENCE = "rejected_equivalence"


class EvidenceAdmissionPolicy(str, Enum):
    """Whether model-authored annotation errors can exclude objective evidence."""

    REQUIRE_COMPLIANT = "require_compliant"
    RECORD_ONLY = "record_only"


def _objective_hash_values(candidate: EvolutionCandidate) -> list[list[str]]:
    """Return a deterministic projection even when objectives fail later checks."""

    if type(candidate.objectives) is not tuple:
        return [["<invalid-container>", type(candidate.objectives).__qualname__]]
    projected: list[list[str]] = []
    for item in candidate.objectives:
        if type(item) is not tuple or len(item) != 2:
            projected.append(["<invalid-item>", type(item).__qualname__])
            continue
        name, value = item
        name_token = name if type(name) is str else f"<{type(name).__qualname__}>"
        if isinstance(value, Real) and not isinstance(value, bool):
            try:
                value_token = float(value).hex()
            except (OverflowError, TypeError, ValueError):
                value_token = f"<invalid-{type(value).__qualname__}>"
        else:
            value_token = f"<{type(value).__qualname__}>"
        projected.append([name_token, value_token])
    return projected


def pareto_candidate_hash(candidate: EvolutionCandidate) -> str:
    """Hash the occurrence and all archive-relevant candidate facts.

    This is an archive projection hash, not a replacement for ``CandidateId`` or the
    canonical typed-JSON configuration hash.  Including both identity and gate facts
    makes accidental reuse of one occurrence ID with different evaluation evidence
    visible in the decision stream.
    """

    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be an exact EvolutionCandidate")
    EvolutionCandidate.__post_init__(candidate)
    occurrence = candidate.occurrence
    record = {
        "candidate_id": candidate.candidate_id.value,
        "configuration_artifact_hash": occurrence.configuration_artifact_hash,
        "configuration_hash": occurrence.configuration_hash,
        "evidence_compliant": candidate.evidence_compliant,
        "objectives": _objective_hash_values(candidate),
        "operator_compliant": candidate.operator_compliant,
        "operator_invocation_id": (
            None
            if occurrence.operator_invocation_id is None
            else occurrence.operator_invocation_id.value
        ),
        "proposal_sequence": occurrence.proposal_sequence,
        "valid": candidate.valid,
    }
    if candidate.detailed_evaluation is not None:
        record["detailed_evaluation_sha256"] = (
            candidate.detailed_evaluation.evidence_sha256
        )
    if candidate.objective_resolution_receipt is not None:
        record["objective_resolution_receipt_sha256"] = (
            candidate.objective_resolution_receipt.receipt_sha256
        )
    encoded = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(_CANDIDATE_HASH_DOMAIN + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ParetoCandidateRef:
    """Minimal immutable identity used in decisions and trace records."""

    candidate_id: CandidateId
    candidate_hash: str
    configuration_hash: str

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        require_sha256(self.candidate_hash, "candidate_hash")
        require_sha256(self.configuration_hash, "configuration_hash")

    def to_trace_record(self) -> dict[str, str]:
        """Return a fresh JSON-safe trace projection."""

        return {
            "candidate_id": self.candidate_id.value,
            "candidate_hash": self.candidate_hash,
            "configuration_hash": self.configuration_hash,
        }


def _candidate_ref(candidate: EvolutionCandidate) -> ParetoCandidateRef:
    return ParetoCandidateRef(
        candidate_id=candidate.candidate_id,
        candidate_hash=pareto_candidate_hash(candidate),
        configuration_hash=candidate.occurrence.configuration_hash,
    )


def _ref_key(reference: ParetoCandidateRef) -> tuple[str, str, str]:
    return (
        reference.configuration_hash,
        reference.candidate_id.value,
        reference.candidate_hash,
    )


@dataclass(frozen=True, slots=True)
class ArchiveRelationEvidence:
    """Relation of the considered candidate to one incumbent front member."""

    incumbent: ParetoCandidateRef
    relation: OutcomeRelation

    def __post_init__(self) -> None:
        if type(self.incumbent) is not ParetoCandidateRef:
            raise TypeError("incumbent must be an exact ParetoCandidateRef")
        if type(self.relation) is not OutcomeRelation:
            raise TypeError("relation must be an OutcomeRelation")

    def to_trace_record(self) -> dict[str, object]:
        return {
            "incumbent": self.incumbent.to_trace_record(),
            "candidate_relation": self.relation.value,
        }


@dataclass(frozen=True, slots=True)
class ParetoDecision:
    """One immutable, self-identifying archive decision."""

    decision_sequence: int
    consideration_sequence: int
    action: ParetoDecisionAction
    reasons: tuple[ParetoDecisionReason, ...]
    candidate: ParetoCandidateRef
    failure_details: tuple[tuple[ParetoDecisionReason, str], ...] = ()
    dominators: tuple[ParetoCandidateRef, ...] = ()
    removed_candidates: tuple[ParetoCandidateRef, ...] = ()
    tie_with: tuple[ParetoCandidateRef, ...] = ()
    duplicate_of: ParetoCandidateRef | None = None
    caused_by: ParetoCandidateRef | None = None
    front_after: tuple[ParetoCandidateRef, ...] = ()
    outcome_relations: tuple[ArchiveRelationEvidence, ...] = ()

    def __post_init__(self) -> None:
        if type(self.decision_sequence) is not int or self.decision_sequence <= 0:
            raise ValueError("decision_sequence must be a positive exact integer")
        if (
            type(self.consideration_sequence) is not int
            or self.consideration_sequence <= 0
        ):
            raise ValueError("consideration_sequence must be a positive exact integer")
        if type(self.action) is not ParetoDecisionAction:
            raise TypeError("action must be a ParetoDecisionAction")
        if type(self.reasons) is not tuple or not self.reasons or any(
            type(reason) is not ParetoDecisionReason for reason in self.reasons
        ):
            raise ValueError("reasons must be a non-empty exact reason tuple")
        if type(self.candidate) is not ParetoCandidateRef:
            raise TypeError("candidate must be a ParetoCandidateRef")
        for name in ("dominators", "removed_candidates", "tie_with", "front_after"):
            references = getattr(self, name)
            if type(references) is not tuple or any(
                type(reference) is not ParetoCandidateRef for reference in references
            ):
                raise TypeError(f"{name} must be an exact ParetoCandidateRef tuple")
        if self.duplicate_of is not None and type(
            self.duplicate_of
        ) is not ParetoCandidateRef:
            raise TypeError("duplicate_of must be a ParetoCandidateRef or None")
        if self.caused_by is not None and type(self.caused_by) is not ParetoCandidateRef:
            raise TypeError("caused_by must be a ParetoCandidateRef or None")
        if type(self.failure_details) is not tuple:
            raise TypeError("failure_details must be an exact tuple")
        for reason, detail in self.failure_details:
            if type(reason) is not ParetoDecisionReason or type(detail) is not str:
                raise TypeError("failure details must contain exact reason/text pairs")
        if type(self.outcome_relations) is not tuple or any(
            type(item) is not ArchiveRelationEvidence
            for item in self.outcome_relations
        ):
            raise TypeError(
                "outcome_relations must contain exact ArchiveRelationEvidence values"
            )

    def to_trace_record(self) -> dict[str, object]:
        """Return a fresh JSON-safe event suitable for a runner trace sink."""

        record = {
            "event_type": "pareto_archive_decision",
            "decision_sequence": self.decision_sequence,
            "consideration_sequence": self.consideration_sequence,
            "action": self.action.value,
            "reasons": [reason.value for reason in self.reasons],
            **self.candidate.to_trace_record(),
            "failure_details": [
                {"reason": reason.value, "detail": detail}
                for reason, detail in self.failure_details
            ],
            "dominators": [item.to_trace_record() for item in self.dominators],
            "removed_candidates": [
                item.to_trace_record() for item in self.removed_candidates
            ],
            "tie_with": [item.to_trace_record() for item in self.tie_with],
            "duplicate_of": (
                None if self.duplicate_of is None else self.duplicate_of.to_trace_record()
            ),
            "caused_by": (
                None if self.caused_by is None else self.caused_by.to_trace_record()
            ),
            "front_after": [item.to_trace_record() for item in self.front_after],
        }
        if self.outcome_relations:
            record["outcome_relations"] = [
                item.to_trace_record() for item in self.outcome_relations
            ]
        return record


@dataclass(frozen=True, slots=True)
class ParetoArchiveSnapshot:
    """Immutable view of archive state and its complete decision ledger."""

    objectives: tuple[ObjectiveSpec, ...]
    front_candidates: tuple[EvolutionCandidate, ...]
    front_references: tuple[ParetoCandidateRef, ...]
    decisions: tuple[ParetoDecision, ...]
    consideration_count: int
    eligible_configuration_count: int
    evidence_admission_policy: EvidenceAdmissionPolicy
    outcome_relation_policy: tuple[str, int, str]
    objective_pareto_relation: bool

    def to_trace_record(self) -> dict[str, object]:
        """Return aggregate JSON-safe evidence without copying full configurations."""

        record = {
            "event_type": "pareto_archive_snapshot",
            "objectives": [
                {"name": objective.name, "goal": objective.goal}
                for objective in self.objectives
            ],
            "front": [item.to_trace_record() for item in self.front_references],
            "front_size": len(self.front_references),
            "consideration_count": self.consideration_count,
            "eligible_configuration_count": self.eligible_configuration_count,
            "decision_count": len(self.decisions),
            "evidence_admission_policy": self.evidence_admission_policy.value,
        }
        if not self.objective_pareto_relation:
            record["outcome_relation_policy"] = {
                "policy_id": self.outcome_relation_policy[0],
                "policy_version": self.outcome_relation_policy[1],
                "definition_sha256": self.outcome_relation_policy[2],
            }
        return record


class ParetoArchive:
    """Admissibility-gated, deterministic, event-auditable Pareto archive."""

    def __init__(
        self,
        objectives: Sequence[ObjectiveSpec],
        *,
        evidence_admission_policy: EvidenceAdmissionPolicy = (
            EvidenceAdmissionPolicy.REQUIRE_COMPLIANT
        ),
        outcome_relation_binding: OutcomeRelationPolicyBinding | None = None,
    ) -> None:
        objective_tuple = tuple(objectives)
        validate_objective_specs(objective_tuple)
        if type(evidence_admission_policy) is not EvidenceAdmissionPolicy:
            raise TypeError(
                "evidence_admission_policy must be an EvidenceAdmissionPolicy"
            )
        default_relation = objective_pareto_outcome_binding(objective_tuple)
        relation = (
            default_relation
            if outcome_relation_binding is None
            else outcome_relation_binding
        )
        if type(relation) is not OutcomeRelationPolicyBinding:
            raise TypeError(
                "outcome_relation_binding must be an OutcomeRelationPolicyBinding"
            )
        OutcomeRelationPolicyBinding.__post_init__(relation)
        self._objectives = objective_tuple
        self._evidence_admission_policy = evidence_admission_policy
        self.outcome_relation_binding = relation
        self._objective_pareto_relation = relation.identity == default_relation.identity
        self._front: dict[str, EvolutionCandidate] = {}
        self._front_objectives: dict[str, dict[str, float]] = {}
        self._observed_ids: dict[str, ParetoCandidateRef] = {}
        self._seen_configurations: dict[str, ParetoCandidateRef] = {}
        self._decisions: list[ParetoDecision] = []
        self._consideration_count = 0

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        return self._objectives

    @property
    def front(self) -> tuple[EvolutionCandidate, ...]:
        return self._ordered_front()[0]

    @property
    def decisions(self) -> tuple[ParetoDecision, ...]:
        return tuple(self._decisions)

    def __len__(self) -> int:
        return len(self._front)

    def _ordered_front(
        self,
    ) -> tuple[tuple[EvolutionCandidate, ...], tuple[ParetoCandidateRef, ...]]:
        pairs = [
            (candidate, _candidate_ref(candidate)) for candidate in self._front.values()
        ]
        pairs.sort(key=lambda pair: _ref_key(pair[1]))
        return (
            tuple(candidate for candidate, _ in pairs),
            tuple(reference for _, reference in pairs),
        )

    def _append_decision(
        self,
        *,
        action: ParetoDecisionAction,
        reasons: tuple[ParetoDecisionReason, ...],
        candidate: ParetoCandidateRef,
        failure_details: tuple[tuple[ParetoDecisionReason, str], ...] = (),
        dominators: tuple[ParetoCandidateRef, ...] = (),
        removed_candidates: tuple[ParetoCandidateRef, ...] = (),
        tie_with: tuple[ParetoCandidateRef, ...] = (),
        duplicate_of: ParetoCandidateRef | None = None,
        caused_by: ParetoCandidateRef | None = None,
        outcome_relations: tuple[ArchiveRelationEvidence, ...] = (),
    ) -> ParetoDecision:
        front_after = self._ordered_front()[1]
        decision = ParetoDecision(
            decision_sequence=len(self._decisions) + 1,
            consideration_sequence=self._consideration_count,
            action=action,
            reasons=reasons,
            candidate=candidate,
            failure_details=failure_details,
            dominators=tuple(sorted(dominators, key=_ref_key)),
            removed_candidates=tuple(sorted(removed_candidates, key=_ref_key)),
            tie_with=tuple(sorted(tie_with, key=_ref_key)),
            duplicate_of=duplicate_of,
            caused_by=caused_by,
            front_after=front_after,
            outcome_relations=outcome_relations,
        )
        self._decisions.append(decision)
        return decision

    @staticmethod
    def _failure_detail(message: str | None, fallback: str) -> str:
        if type(message) is str and message.strip():
            return message
        return fallback

    def _normalize_candidate_objectives(
        self,
        candidate: EvolutionCandidate,
    ) -> dict[str, float]:
        if type(candidate.objectives) is not tuple:
            raise ProblemContractError("candidate objectives must be an exact tuple")
        raw: dict[str, object] = {}
        for item in candidate.objectives:
            if type(item) is not tuple or len(item) != 2:
                raise ProblemContractError(
                    "candidate objectives must contain exact (name, value) tuples"
                )
            name, value = item
            if type(name) is not str:
                raise ProblemContractError("candidate objective names must be strings")
            if name in raw:
                raise ProblemContractError(f"duplicate candidate objective {name!r}")
            raw[name] = value
        return normalize_objective_values(raw, self._objectives)

    def _consider_by_outcome_relation(
        self,
        candidate: EvolutionCandidate,
        reference: ParetoCandidateRef,
        objective_map: dict[str, float],
        front_entries: list[
            tuple[EvolutionCandidate, ParetoCandidateRef, dict[str, float]]
        ],
        *,
        start: int,
    ) -> tuple[ParetoDecision, ...]:
        detailed = candidate.detailed_evaluation
        if detailed is None:
            raise ProblemContractError(
                "custom outcome relation requires candidate detailed evidence"
            )
        relations: list[
            tuple[EvolutionCandidate, ParetoCandidateRef, OutcomeRelation]
        ] = []
        for incumbent, incumbent_ref, _ in sorted(
            front_entries,
            key=lambda entry: _ref_key(entry[1]),
        ):
            incumbent_detailed = incumbent.detailed_evaluation
            if incumbent_detailed is None:
                raise RuntimeError(
                    "custom-relation archive contains evidence-free incumbent"
                )
            relations.append(
                (
                    incumbent,
                    incumbent_ref,
                    self.outcome_relation_binding.relate(
                        detailed,
                        incumbent_detailed,
                    ),
                )
            )
        relation_evidence = tuple(
            ArchiveRelationEvidence(incumbent_ref, relation)
            for _, incumbent_ref, relation in relations
        )
        worse = tuple(
            incumbent_ref
            for _, incumbent_ref, relation in relations
            if relation is OutcomeRelation.WORSE
        )
        if worse:
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=(ParetoDecisionReason.REJECTED_WORSE_RELATION,),
                candidate=reference,
                outcome_relations=relation_evidence,
            )
            return tuple(self._decisions[start:])

        equivalent = tuple(
            incumbent_ref
            for _, incumbent_ref, relation in relations
            if relation is OutcomeRelation.EQUIVALENT
        )
        if equivalent and min((reference, *equivalent), key=_ref_key) is not reference:
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=(ParetoDecisionReason.REJECTED_EQUIVALENCE,),
                candidate=reference,
                tie_with=equivalent,
                outcome_relations=relation_evidence,
            )
            return tuple(self._decisions[start:])

        better = tuple(
            incumbent_ref
            for _, incumbent_ref, relation in relations
            if relation is OutcomeRelation.BETTER
        )
        removed = tuple(
            sorted(
                {
                    item.candidate_id.value: item
                    for item in (*equivalent, *better)
                }.values(),
                key=_ref_key,
            )
        )
        equivalent_ids = {item.candidate_id.value for item in equivalent}
        for item in removed:
            del self._front[item.candidate_id.value]
            del self._front_objectives[item.candidate_id.value]
        self._front[candidate.candidate_id.value] = candidate
        self._front_objectives[candidate.candidate_id.value] = objective_map

        self._append_decision(
            action=ParetoDecisionAction.ADMITTED,
            reasons=(
                ParetoDecisionReason.ADMITTED_EQUIVALENCE_TIE_BREAK
                if equivalent
                else ParetoDecisionReason.ADMITTED_RELATION_FRONT
            ,),
            candidate=reference,
            removed_candidates=removed,
            tie_with=equivalent,
            outcome_relations=relation_evidence,
        )
        for item in removed:
            self._append_decision(
                action=ParetoDecisionAction.REMOVED,
                reasons=(
                    ParetoDecisionReason.REMOVED_EQUIVALENCE_TIE_BREAK
                    if item.candidate_id.value in equivalent_ids
                    else ParetoDecisionReason.REMOVED_WORSE_RELATION
                ,),
                candidate=item,
                caused_by=reference,
            )
        return tuple(self._decisions[start:])

    def consider(self, candidate: EvolutionCandidate) -> tuple[ParetoDecision, ...]:
        """Consider one candidate and return every resulting atomic decision.

        An admission that evicts two incumbents returns three decisions: one admission
        naming both removals, followed by one explicit removal decision per incumbent.
        Rejections never mutate the Pareto front.  Only gate-passing candidates consume
        configuration-deduplication identities.
        """

        reference = _candidate_ref(candidate)
        self._consideration_count += 1
        start = len(self._decisions)

        previous_id = self._observed_ids.get(candidate.candidate_id.value)
        if previous_id is not None:
            reason = (
                ParetoDecisionReason.REJECTED_DUPLICATE_CANDIDATE
                if previous_id.candidate_hash == reference.candidate_hash
                else ParetoDecisionReason.REJECTED_CANDIDATE_ID_CONFLICT
            )
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=(reason,),
                candidate=reference,
                duplicate_of=previous_id,
            )
            return tuple(self._decisions[start:])
        self._observed_ids[candidate.candidate_id.value] = reference

        gate_reasons: list[ParetoDecisionReason] = []
        failure_details: list[tuple[ParetoDecisionReason, str]] = []
        if not candidate.valid:
            reason = ParetoDecisionReason.REJECTED_INVALID
            gate_reasons.append(reason)
            failure_details.append(
                (
                    reason,
                    self._failure_detail(
                        candidate.failure_message,
                        "candidate.valid is False",
                    ),
                )
            )
        if not candidate.operator_compliant:
            reason = ParetoDecisionReason.REJECTED_OPERATOR_NONCOMPLIANT
            gate_reasons.append(reason)
            failure_details.append(
                (
                    reason,
                    self._failure_detail(
                        candidate.operator_failure,
                        "candidate.operator_compliant is False",
                    ),
                )
            )
        if (
            self._evidence_admission_policy
            is EvidenceAdmissionPolicy.REQUIRE_COMPLIANT
            and not candidate.evidence_compliant
        ):
            reason = ParetoDecisionReason.REJECTED_EVIDENCE_NONCOMPLIANT
            gate_reasons.append(reason)
            failure_details.append(
                (
                    reason,
                    self._failure_detail(
                        candidate.evidence_failure,
                        "candidate.evidence_compliant is False",
                    ),
                )
            )
        if gate_reasons:
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=tuple(gate_reasons),
                candidate=reference,
                failure_details=tuple(failure_details),
            )
            return tuple(self._decisions[start:])

        try:
            objective_map = self._normalize_candidate_objectives(candidate)
        except (OverflowError, ProblemContractError, TypeError, ValueError) as exc:
            reason = ParetoDecisionReason.REJECTED_OBJECTIVE_CONTRACT
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=(reason,),
                candidate=reference,
                failure_details=((reason, str(exc)),),
            )
            return tuple(self._decisions[start:])

        duplicate = self._seen_configurations.get(reference.configuration_hash)
        if duplicate is not None:
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=(
                    ParetoDecisionReason.REJECTED_DUPLICATE_CONFIGURATION,
                ),
                candidate=reference,
                duplicate_of=duplicate,
            )
            return tuple(self._decisions[start:])
        self._seen_configurations[reference.configuration_hash] = reference

        front_entries = [
            (
                incumbent,
                _candidate_ref(incumbent),
                self._front_objectives[incumbent.candidate_id.value],
            )
            for incumbent in self._front.values()
        ]
        if not self._objective_pareto_relation:
            return self._consider_by_outcome_relation(
                candidate,
                reference,
                objective_map,
                front_entries,
                start=start,
            )
        dominators = tuple(
            incumbent_ref
            for _, incumbent_ref, incumbent_objectives in front_entries
            if dominates(incumbent_objectives, objective_map, self._objectives)
        )
        if dominators:
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=(ParetoDecisionReason.REJECTED_DOMINATED,),
                candidate=reference,
                dominators=dominators,
            )
            return tuple(self._decisions[start:])

        objective_tuple = tuple(objective_map[spec.name] for spec in self._objectives)
        tied = tuple(
            incumbent_ref
            for _, incumbent_ref, incumbent_objectives in front_entries
            if tuple(
                incumbent_objectives[spec.name] for spec in self._objectives
            )
            == objective_tuple
        )
        if tied and min((reference, *tied), key=_ref_key) is not reference:
            self._append_decision(
                action=ParetoDecisionAction.REJECTED,
                reasons=(ParetoDecisionReason.REJECTED_OBJECTIVE_TIE,),
                candidate=reference,
                tie_with=tied,
            )
            return tuple(self._decisions[start:])

        dominated = tuple(
            incumbent_ref
            for _, incumbent_ref, incumbent_objectives in front_entries
            if dominates(objective_map, incumbent_objectives, self._objectives)
        )
        tied_ids = {item.candidate_id.value for item in tied}
        dominated_ids = {item.candidate_id.value for item in dominated}
        removed = tuple(
            sorted(
                {
                    item.candidate_id.value: item
                    for item in (*tied, *dominated)
                }.values(),
                key=_ref_key,
            )
        )
        for item in removed:
            del self._front[item.candidate_id.value]
            del self._front_objectives[item.candidate_id.value]
        self._front[candidate.candidate_id.value] = candidate
        self._front_objectives[candidate.candidate_id.value] = objective_map

        admission_reason = (
            ParetoDecisionReason.ADMITTED_TIE_BREAK_REPLACEMENT
            if tied
            else ParetoDecisionReason.ADMITTED_NONDOMINATED
        )
        self._append_decision(
            action=ParetoDecisionAction.ADMITTED,
            reasons=(admission_reason,),
            candidate=reference,
            removed_candidates=removed,
            tie_with=tied,
        )
        for item in removed:
            removal_reason = (
                ParetoDecisionReason.REMOVED_TIE_BREAK
                if item.candidate_id.value in tied_ids
                else ParetoDecisionReason.REMOVED_DOMINATED
            )
            if item.candidate_id.value not in tied_ids | dominated_ids:
                raise AssertionError("archive removal lacks a deterministic cause")
            self._append_decision(
                action=ParetoDecisionAction.REMOVED,
                reasons=(removal_reason,),
                candidate=item,
                caused_by=reference,
            )
        return tuple(self._decisions[start:])

    def snapshot(self) -> ParetoArchiveSnapshot:
        """Return an immutable point-in-time view of state and all decisions."""

        candidates, references = self._ordered_front()
        return ParetoArchiveSnapshot(
            objectives=self._objectives,
            front_candidates=candidates,
            front_references=references,
            decisions=tuple(self._decisions),
            consideration_count=self._consideration_count,
            eligible_configuration_count=len(self._seen_configurations),
            evidence_admission_policy=self._evidence_admission_policy,
            outcome_relation_policy=self.outcome_relation_binding.identity,
            objective_pareto_relation=self._objective_pareto_relation,
        )


__all__ = [
    "EvidenceAdmissionPolicy",
    "ParetoArchive",
    "ParetoArchiveSnapshot",
    "ParetoCandidateRef",
    "ParetoDecision",
    "ParetoDecisionAction",
    "ParetoDecisionReason",
    "pareto_candidate_hash",
]

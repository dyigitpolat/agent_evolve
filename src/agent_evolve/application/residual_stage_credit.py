"""Conserved stage closure for mixed-expert residual evolution.

The proposal market deliberately ends immediately after the broker-selected
slate has been evaluated.  This module owns the next, workload-neutral
boundary: authenticate the actual pre/post archive utility receipts, allocate
their realized joint gain exactly over the evaluated slate, and compile the
append-only evidence consumed by the broker and earned-lineage ledgers.

No proposal expert is allowed to grade itself.  Archive utility and admission
remain inverted ports, while the core sees only objective mappings, immutable
identities, and real evaluator receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.earned_lineage import (
    CandidateConservedCredit,
    CandidateProposalProvenance,
    ConservedStageCreditReceipt,
    ProposalLineageRole,
)
from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionOutcome,
)
from agent_evolve.application.residual_portfolio_evolution import (
    ResidualPortfolioEvolutionResult,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.reward.contextual_marginal_utility import (
    CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_DEFINITION_SHA256,
    JointUtilitySnapshot,
    ReplayableArchiveUtility,
    exact_coalition_shapley_values,
)


RESIDUAL_STAGE_CREDIT_PROJECTOR_ID = "mixed_expert_exact_coalition_stage_credit"
RESIDUAL_STAGE_CREDIT_PROJECTOR_VERSION = 1
RESIDUAL_STAGE_CREDIT_PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:mixed-expert-exact-coalition-stage-credit:v1;"
    b"cutoff=authenticated-pre-and-post-archive-utility-snapshots;"
    b"players=broker-selected-real-evaluations-only;"
    b"admission=injected-workload-neutral-policy;"
    b"credit=exact-coalition-shapley-joint-archive-gain;"
    b"conservation=post-minus-pre-utility;"
    b"broker-feedback=immediate-realized-feasibility-and-conserved-gain;"
    b"provenance=sealed-proposal-expert-operator-parents-and-cutoff;"
    b"positive-credit=counterfactual-coalition-admission-not-final-front-retention;"
    b"workload-model-provider-branches=false"
).hexdigest()

STRICT_RESIDUAL_CANDIDATE_ADMISSION_ID = "strict_evaluated_candidate_admission"
STRICT_RESIDUAL_CANDIDATE_ADMISSION_VERSION = 1
STRICT_RESIDUAL_CANDIDATE_ADMISSION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:strict-evaluated-candidate-admission:v1;"
    b"admissible=valid-and-operator-compliant-and-evidence-compliant;"
    b"objective-contract=archive-utility-replay;"
    b"workload-model-provider-branches=false"
).hexdigest()

_PROJECTION_DOMAIN = b"agent-evolve:conserved-residual-stage-projection:v1\x00"


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


@runtime_checkable
class ResidualCandidateAdmissionPort(Protocol):
    """Decide whether one real outcome enters the archive coalition game."""

    admission_id: str
    admission_version: int
    definition_sha256: str

    def admissible(self, candidate: EvolutionCandidate) -> bool: ...


@dataclass(frozen=True, slots=True)
class StrictResidualCandidateAdmission:
    """Default archive gate matching the strict Pareto admission policy."""

    admission_id: str = field(
        init=False,
        default=STRICT_RESIDUAL_CANDIDATE_ADMISSION_ID,
    )
    admission_version: int = field(
        init=False,
        default=STRICT_RESIDUAL_CANDIDATE_ADMISSION_VERSION,
    )
    definition_sha256: str = field(
        init=False,
        default=STRICT_RESIDUAL_CANDIDATE_ADMISSION_DEFINITION_SHA256,
    )

    def admissible(self, candidate: EvolutionCandidate) -> bool:
        if type(candidate) is not EvolutionCandidate:
            raise TypeError("candidate must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(candidate)
        return bool(
            candidate.valid
            and candidate.operator_compliant
            and candidate.evidence_compliant
        )


def _validate_admission_port(value: ResidualCandidateAdmissionPort) -> None:
    if not isinstance(value, ResidualCandidateAdmissionPort):
        raise TypeError("admission must implement ResidualCandidateAdmissionPort")
    if type(value.admission_id) is not str or not value.admission_id:
        raise ValueError("admission_id must be a non-empty exact string")
    if type(value.admission_version) is not int or value.admission_version <= 0:
        raise ValueError("admission_version must be positive")
    require_sha256(value.definition_sha256, "admission definition_sha256")


@dataclass(frozen=True, slots=True)
class ConservedResidualStageProjection:
    """Pure, authenticated evidence prepared before ledger publication."""

    residual_result_sha256: str
    pre_archive_utility_snapshot_sha256: str
    post_archive_utility_snapshot_sha256: str
    admission_id: str
    admission_version: int
    admission_definition_sha256: str
    stage_credit: ConservedStageCreditReceipt
    action_outcomes: tuple[MaterializedActionOutcome, ...]
    candidate_provenance: tuple[CandidateProposalProvenance, ...]
    projector_definition_sha256: str = (
        RESIDUAL_STAGE_CREDIT_PROJECTOR_DEFINITION_SHA256
    )
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "residual_result_sha256",
            "pre_archive_utility_snapshot_sha256",
            "post_archive_utility_snapshot_sha256",
            "admission_definition_sha256",
            "projector_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.admission_id) is not str or not self.admission_id:
            raise ValueError("admission_id must be a non-empty exact string")
        if type(self.admission_version) is not int or self.admission_version <= 0:
            raise ValueError("admission_version must be positive")
        if type(self.stage_credit) is not ConservedStageCreditReceipt:
            raise TypeError("stage_credit must be exact")
        ConservedStageCreditReceipt.__post_init__(self.stage_credit)
        if type(self.action_outcomes) is not tuple or not self.action_outcomes:
            raise ValueError("action_outcomes must be a non-empty exact tuple")
        for value in self.action_outcomes:
            if type(value) is not MaterializedActionOutcome:
                raise TypeError("action_outcomes must contain exact values")
            value.__post_init__()
            if not value.realized:
                raise ValueError("stage closure cannot contain unrealized outcomes")
        if type(self.candidate_provenance) is not tuple:
            raise TypeError("candidate_provenance must be an exact tuple")
        for value in self.candidate_provenance:
            if type(value) is not CandidateProposalProvenance:
                raise TypeError("candidate_provenance must contain exact values")
            CandidateProposalProvenance.__post_init__(value)

        outcome_ids = tuple(
            value.action.target_candidate_id.value for value in self.action_outcomes
        )
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("stage closure repeats a candidate outcome")
        credit_ids = tuple(
            value.candidate_id.value for value in self.stage_credit.candidate_credits
        )
        provenance_ids = tuple(
            value.candidate_id.value for value in self.candidate_provenance
        )
        if tuple(sorted(outcome_ids)) != credit_ids:
            raise ValueError("stage credit does not exactly cover action outcomes")
        if provenance_ids != credit_ids:
            raise ValueError("candidate provenance does not exactly cover stage credit")
        credit_by_id = {
            value.candidate_id: value for value in self.stage_credit.candidate_credits
        }
        for outcome in self.action_outcomes:
            credit = credit_by_id[outcome.action.target_candidate_id]
            if (
                outcome.feasible != credit.admitted_to_archive
                or outcome.normalized_archive_gain != credit.contribution
            ):
                raise ValueError("broker outcome differs from conserved candidate credit")
        object.__setattr__(
            self,
            "projection_sha256",
            _hash(_PROJECTION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "projector": {
                "projector_id": RESIDUAL_STAGE_CREDIT_PROJECTOR_ID,
                "projector_version": RESIDUAL_STAGE_CREDIT_PROJECTOR_VERSION,
                "definition_sha256": self.projector_definition_sha256,
            },
            "residual_result_sha256": self.residual_result_sha256,
            "pre_archive_utility_snapshot_sha256": (
                self.pre_archive_utility_snapshot_sha256
            ),
            "post_archive_utility_snapshot_sha256": (
                self.post_archive_utility_snapshot_sha256
            ),
            "admission": {
                "admission_id": self.admission_id,
                "admission_version": self.admission_version,
                "definition_sha256": self.admission_definition_sha256,
            },
            "stage_credit_receipt_sha256": self.stage_credit.receipt_sha256,
            "action_outcome_sha256s": [
                value.outcome_sha256 for value in self.action_outcomes
            ],
            "candidate_provenance_sha256s": [
                value.provenance_sha256 for value in self.candidate_provenance
            ],
            "credit_semantics": (
                "counterfactual_archive_coalition_contribution;"
                "final_front_retention_is_delayed_credit"
            ),
            "coalition_value_definition_sha256": (
                CONTEXTUAL_SHAPLEY_UTILITY_PROJECTOR_DEFINITION_SHA256
            ),
            "workload_model_provider_fields_present": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "stage_credit": self.stage_credit.to_record(),
            "action_outcomes": [
                value.to_record() for value in self.action_outcomes
            ],
            "candidate_provenance": [
                value.to_record() for value in self.candidate_provenance
            ],
            "projection_sha256": self.projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResidualStageCreditProjector:
    """Project one evaluated residual slate into conserved learning evidence."""

    archive_utility: ReplayableArchiveUtility = field(repr=False, compare=False)
    admission: ResidualCandidateAdmissionPort = field(
        default_factory=StrictResidualCandidateAdmission,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.archive_utility, ReplayableArchiveUtility):
            raise TypeError("archive_utility must replay utility snapshots")
        _validate_admission_port(self.admission)

    def project(
        self,
        *,
        pre_snapshot: ArchiveUtilitySnapshot,
        post_snapshot: ArchiveUtilitySnapshot,
        result: ResidualPortfolioEvolutionResult,
    ) -> ConservedResidualStageProjection:
        self.__post_init__()
        if type(pre_snapshot) is not ArchiveUtilitySnapshot:
            raise TypeError("pre_snapshot must be exact")
        if type(post_snapshot) is not ArchiveUtilitySnapshot:
            raise TypeError("post_snapshot must be exact")
        pre_snapshot.__post_init__()
        post_snapshot.__post_init__()
        if type(result) is not ResidualPortfolioEvolutionResult:
            raise TypeError("result must be exact")
        result.__post_init__()
        if (
            pre_snapshot.utility_id,
            pre_snapshot.utility_version,
            pre_snapshot.definition_sha256,
            pre_snapshot.generation,
            pre_snapshot.benchmark_sha256,
        ) != (
            post_snapshot.utility_id,
            post_snapshot.utility_version,
            post_snapshot.definition_sha256,
            post_snapshot.generation,
            post_snapshot.benchmark_sha256,
        ):
            raise ValueError("pre/post archive utility snapshots are not comparable")
        if pre_snapshot.generation != result.request.decision_index:
            raise ValueError("archive utility generation differs from residual decision")
        if pre_snapshot.archive_sha256 == post_snapshot.archive_sha256:
            # A zero-gain stage still records new candidate decisions, so the
            # authenticated archive state (including its ledger) must advance.
            raise ValueError("post archive cutoff did not advance")
        if (
            pre_snapshot.scalar_utility_hex is None
            or post_snapshot.scalar_utility_hex is None
        ):
            raise ValueError("conserved stage credit requires scalar utility receipts")
        pre_utility = float.fromhex(pre_snapshot.scalar_utility_hex)
        post_utility = float.fromhex(post_snapshot.scalar_utility_hex)
        replayed = self.archive_utility.require_snapshot(pre_snapshot)
        if not isinstance(replayed, JointUtilitySnapshot):
            raise TypeError("archive utility returned no joint-gain snapshot")

        proposal_by_expert = {
            value.expert_id: value for value in result.proposals
        }
        selected = result.broker_decision.selected_actions
        evaluations = result.evaluations
        if len(selected) != len(evaluations):  # Defensive; result closes this.
            raise RuntimeError("broker slate differs from real evaluation count")

        rows: list[
            tuple[
                str,
                int,
                EvolutionCandidate,
                bool,
            ]
        ] = []
        for index, evaluation in enumerate(evaluations):
            candidate = evaluation.candidate
            admissible = self.admission.admissible(candidate)
            if type(admissible) is not bool:
                raise TypeError("admission port must return an exact bool")
            rows.append(
                (
                    candidate.candidate_id.value,
                    index,
                    candidate,
                    admissible,
                )
            )
        rows.sort(key=lambda value: value[0])
        if len({value[0] for value in rows}) != len(rows):
            raise ValueError("stage credit cannot repeat a candidate")
        players = tuple(value for value in rows if value[3])
        credits = exact_coalition_shapley_values(
            replayed,
            tuple(value[2].objective_map for value in players),
        )
        credit_by_candidate_id = {
            value[2].candidate_id: credit
            for value, credit in zip(players, credits, strict=True)
        }
        joint_gain = replayed.joint_gain(
            tuple(value[2].objective_map for value in players)
        )
        if type(joint_gain) is not float or not math.isfinite(joint_gain):
            raise TypeError("joint archive gain must be a finite exact float")
        tolerance = 16.0 * math.ulp(max(1.0, pre_utility, post_utility))
        if abs((post_utility - pre_utility) - joint_gain) > tolerance:
            raise ValueError(
                "observed post archive utility differs from the evaluated coalition"
            )

        candidate_credits = tuple(
            CandidateConservedCredit(
                candidate_id=candidate.candidate_id,
                contribution=float(
                    credit_by_candidate_id.get(candidate.candidate_id, 0.0)
                ),
                # This means admission to at least the counterfactual
                # coalition game, not survival on the final stage front.
                admitted_to_archive=admissible,
                outcome_receipt_sha256=(
                    result.evaluations[index].evaluation_sha256
                ),
            )
            for _candidate_id, index, candidate, admissible in rows
        )
        stage_credit = ConservedStageCreditReceipt(
            generation=result.request.decision_index,
            utility_id=pre_snapshot.utility_id,
            utility_version=pre_snapshot.utility_version,
            utility_definition_sha256=pre_snapshot.definition_sha256,
            pre_archive_sha256=pre_snapshot.archive_sha256,
            post_archive_sha256=post_snapshot.archive_sha256,
            pre_utility=pre_utility,
            post_utility=post_utility,
            contribution_policy_id=RESIDUAL_STAGE_CREDIT_PROJECTOR_ID,
            contribution_policy_version=RESIDUAL_STAGE_CREDIT_PROJECTOR_VERSION,
            contribution_policy_definition_sha256=(
                RESIDUAL_STAGE_CREDIT_PROJECTOR_DEFINITION_SHA256
            ),
            candidate_credits=candidate_credits,
        )

        admissible_by_candidate_id = {
            candidate.candidate_id: admissible
            for _candidate_id, _index, candidate, admissible in rows
        }
        action_outcomes = tuple(
            MaterializedActionOutcome(
                action=action,
                realized=True,
                feasible=admissible_by_candidate_id[
                    evaluation.candidate.candidate_id
                ],
                normalized_archive_gain=float(
                    credit_by_candidate_id.get(
                        evaluation.candidate.candidate_id,
                        0.0,
                    )
                ),
                positive_marginal_utility=(
                    credit_by_candidate_id.get(
                        evaluation.candidate.candidate_id,
                        0.0,
                    )
                    > 0.0
                ),
            )
            for action, evaluation in zip(selected, evaluations, strict=True)
        )
        candidate_provenance = tuple(
            CandidateProposalProvenance(
                candidate_id=evaluation.candidate.candidate_id,
                configuration_sha256=(
                    evaluation.candidate.occurrence.configuration_hash
                ),
                generation=evaluation.candidate.generation,
                source_role=(
                    ProposalLineageRole.BACKBONE
                    if action.reference_action
                    else ProposalLineageRole.CHALLENGER
                ),
                proposal_expert_id=action.expert_id,
                proposal_expert_version=(
                    proposal_by_expert[action.expert_id].expert_version
                ),
                proposal_expert_definition_sha256=(
                    proposal_by_expert[
                        action.expert_id
                    ].expert_definition_sha256
                ),
                operator_id=action.operator_id,
                parent_candidate_ids=action.parent_ids,
                decision_cutoff_sha256=result.request.prior_state_sha256,
                source_receipt_sha256=(
                    proposal_by_expert[action.expert_id].proposal_sha256
                ),
            )
            for action, evaluation in sorted(
                zip(selected, evaluations, strict=True),
                key=lambda value: value[1].candidate.candidate_id.value,
            )
        )
        return ConservedResidualStageProjection(
            residual_result_sha256=result.result_sha256,
            pre_archive_utility_snapshot_sha256=pre_snapshot.snapshot_sha256,
            post_archive_utility_snapshot_sha256=post_snapshot.snapshot_sha256,
            admission_id=self.admission.admission_id,
            admission_version=self.admission.admission_version,
            admission_definition_sha256=self.admission.definition_sha256,
            stage_credit=stage_credit,
            action_outcomes=action_outcomes,
            candidate_provenance=candidate_provenance,
        )


__all__ = [
    "RESIDUAL_STAGE_CREDIT_PROJECTOR_DEFINITION_SHA256",
    "RESIDUAL_STAGE_CREDIT_PROJECTOR_ID",
    "RESIDUAL_STAGE_CREDIT_PROJECTOR_VERSION",
    "STRICT_RESIDUAL_CANDIDATE_ADMISSION_DEFINITION_SHA256",
    "STRICT_RESIDUAL_CANDIDATE_ADMISSION_ID",
    "STRICT_RESIDUAL_CANDIDATE_ADMISSION_VERSION",
    "ConservedResidualStageProjection",
    "ResidualCandidateAdmissionPort",
    "ResidualStageCreditProjector",
    "StrictResidualCandidateAdmission",
]

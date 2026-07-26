"""Wave-sealed causal memory for staged agentic evolution.

This module is deliberately framework- and provider-free.  It defines the
immutable records that an application layer can bind into an invocation plan,
plus deterministic policies for diagnostic credit and later matched controls.

The central timing rule is strict: every assignment in a diagnostic wave is
resolved against one immutable score snapshot, and no reward from that wave is
visible until the complete wave is sealed.  Successful calls use their frozen
reward.  Model/schema failures and candidate failures receive the
pre-registered no-yield reward under an intention-to-treat (ITT) estimand.
Infrastructure failures invalidate the whole wave and publish no checkpoint.

For insight ``i``, the frozen causal-search score is::

    support_i = min(treated_ESS_i, control_ESS_i)
    shrink_i = support_i / (support_i + n0)
    mean_i = prior_i + shrink_i * effect_i
    uncertainty_i = c / sqrt(support_i + n0)
    retrieval_i = mean_i + beta * uncertainty_i

An unidentified effect is exactly zero for the update (not evidence of harm),
while its support remains zero.  ``n0``, ``c``, and ``beta`` are recorded in
every snapshot and therefore cannot drift silently between replay and use.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from numbers import Real

from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.policies.memory.randomized_subset import (
    EpsilonGreedySubsetSelector,
    InsightSelectionDecision,
    InsightSelectionMode,
    InsightTrial,
    estimate_marginal_effect,
)


_LOWER_SHA256 = frozenset("0123456789abcdef")
_SAFE_BLOCK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SELECTION_DECISION_DOMAIN = b"agent-evolve:insight-selection-decision:v1\x00"
_ASSIGNMENT_DOMAIN = b"agent-evolve:resolved-insight-assignment:v1\x00"
_SCORE_SNAPSHOT_DOMAIN = b"agent-evolve:causal-memory-score-snapshot:v1\x00"
_WAVE_DOMAIN = b"agent-evolve:frozen-diagnostic-memory-wave:v1\x00"


def _require_sha256(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_block_id(value: str, name: str) -> None:
    if type(value) is not str or _SAFE_BLOCK_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded durable identifier")


def _canonical_float(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_canonical_float(value: float, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


def _ref_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


def _fraction_record(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _hash_record(domain: bytes, record: Mapping[str, object]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")
    return hashlib.sha256(domain + payload).hexdigest()


def _decision_record(decision: InsightSelectionDecision) -> dict[str, object]:
    return {
        "policy_id": decision.policy_id,
        "policy_version": decision.policy_version,
        "context_hash": decision.context_hash,
        "eligible": [_ref_record(reference) for reference in decision.eligible],
        "selected": [_ref_record(reference) for reference in decision.selected],
        "exploitation_subset": [
            _ref_record(reference) for reference in decision.exploitation_subset
        ],
        "score_snapshot": [
            {
                "reference": _ref_record(reference),
                "score_hex": score.hex(),
            }
            for reference, score in decision.score_snapshot
        ],
        "subset_size": decision.subset_size,
        "exploration_probability": _fraction_record(decision.exploration_probability),
        "mode": decision.mode.value,
        "selected_subset_probability": _fraction_record(
            decision.selected_subset_probability
        ),
    }


def insight_selection_decision_sha256(
    decision: InsightSelectionDecision,
) -> str:
    """Return a stable digest of the complete assignment law and realization."""

    if not isinstance(decision, InsightSelectionDecision):
        raise TypeError("decision must be an InsightSelectionDecision")
    return _hash_record(_SELECTION_DECISION_DOMAIN, _decision_record(decision))


class MemoryAssignmentArm(str, Enum):
    """Predeclared role of one invocation in a staged memory experiment."""

    DIAGNOSTIC = "diagnostic"
    ADAPTIVE = "adaptive"
    SCORE_SHUFFLED_CONTROL = "score_shuffled_control"
    UNIFORM_CONTROL = "uniform_control"


class DelayedCreditMode(str, Enum):
    """When and under which estimand an assignment may update memory."""

    WAVE_SEALED_ITT = "wave_sealed_intention_to_treat"


@dataclass(frozen=True, slots=True)
class ResolvedInsightAssignment:
    """One plan-ready retrieval assignment bound to immutable evidence.

    ``score_snapshot_sha256`` identifies the complete
    :class:`MemoryScoreSnapshot`, not merely its score table.  The exact score
    table and selected references remain embedded in ``selection_decision``.
    Application code should construct values with :meth:`resolve` whenever the
    source snapshot is available; direct construction remains useful for a
    durable codec and is revalidated at a frozen-wave boundary.
    """

    credit_unit_id: OperatorInvocationId
    exact_context_hash: str
    estimand_stratum_hash: str
    block_id: str
    arm: MemoryAssignmentArm
    selection_decision: InsightSelectionDecision
    selection_decision_sha256: str
    score_snapshot_sha256: str
    prompt_shape_sha256: str
    credit_mode: DelayedCreditMode = DelayedCreditMode.WAVE_SEALED_ITT
    assignment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.credit_unit_id, OperatorInvocationId):
            raise TypeError("credit_unit_id must be an OperatorInvocationId")
        _require_sha256(self.exact_context_hash, "exact_context_hash")
        _require_sha256(self.estimand_stratum_hash, "estimand_stratum_hash")
        _require_block_id(self.block_id, "block_id")
        if not isinstance(self.arm, MemoryAssignmentArm):
            raise TypeError("arm must be a MemoryAssignmentArm")
        if not isinstance(self.selection_decision, InsightSelectionDecision):
            raise TypeError("selection_decision must be an InsightSelectionDecision")
        if self.selection_decision.context_hash != self.exact_context_hash:
            raise ValueError(
                "selection decision context does not match exact_context_hash"
            )
        _require_sha256(self.selection_decision_sha256, "selection_decision_sha256")
        if self.selection_decision_sha256 != insight_selection_decision_sha256(
            self.selection_decision
        ):
            raise ValueError("selection_decision_sha256 does not match the decision")
        _require_sha256(self.score_snapshot_sha256, "score_snapshot_sha256")
        _require_sha256(self.prompt_shape_sha256, "prompt_shape_sha256")
        if not isinstance(self.credit_mode, DelayedCreditMode):
            raise TypeError("credit_mode must be a DelayedCreditMode")
        self._validate_arm_law()
        object.__setattr__(
            self,
            "assignment_sha256",
            _hash_record(_ASSIGNMENT_DOMAIN, self.to_record()),
        )

    def _validate_arm_law(self) -> None:
        decision = self.selection_decision
        if self.arm is MemoryAssignmentArm.DIAGNOSTIC:
            if not decision.credit_identifiable:
                raise ValueError(
                    "diagnostic assignments require inclusion-probability overlap"
                )
            return
        if self.arm in {
            MemoryAssignmentArm.ADAPTIVE,
            MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL,
        }:
            if (
                decision.exploration_probability != 0
                or decision.mode is not InsightSelectionMode.EXPLOIT
            ):
                raise ValueError(
                    "adaptive and score-shuffled arms require deterministic exploitation"
                )
            return
        if self.arm is MemoryAssignmentArm.UNIFORM_CONTROL and (
            decision.exploration_probability != 1
            or decision.mode is not InsightSelectionMode.EXPLORE_UNIFORM
        ):
            raise ValueError(
                "uniform-control assignments require the exact uniform-subset law"
            )

    @classmethod
    def resolve(
        cls,
        *,
        credit_unit_id: OperatorInvocationId,
        snapshot: MemoryScoreSnapshot,
        expected_snapshot_sha256: str,
        block_id: str,
        arm: MemoryAssignmentArm,
        selection_decision: InsightSelectionDecision,
        prompt_shape_sha256: str,
        credit_mode: DelayedCreditMode = DelayedCreditMode.WAVE_SEALED_ITT,
    ) -> ResolvedInsightAssignment:
        """Bind a decision to the expected current snapshot, failing stale."""

        if not isinstance(snapshot, MemoryScoreSnapshot):
            raise TypeError("snapshot must be a MemoryScoreSnapshot")
        _require_sha256(expected_snapshot_sha256, "expected_snapshot_sha256")
        if snapshot.snapshot_sha256 != expected_snapshot_sha256:
            raise StaleMemorySnapshotError(
                "current memory snapshot differs from the predeclared snapshot"
            )
        assignment = cls(
            credit_unit_id=credit_unit_id,
            exact_context_hash=snapshot.exact_context_hash,
            estimand_stratum_hash=snapshot.estimand_stratum_hash,
            block_id=block_id,
            arm=arm,
            selection_decision=selection_decision,
            selection_decision_sha256=insight_selection_decision_sha256(
                selection_decision
            ),
            score_snapshot_sha256=snapshot.snapshot_sha256,
            prompt_shape_sha256=prompt_shape_sha256,
            credit_mode=credit_mode,
        )
        assignment.validate_against_snapshot(snapshot)
        return assignment

    def validate_against_snapshot(self, snapshot: MemoryScoreSnapshot) -> None:
        """Revalidate the snapshot binding and arm-specific score relationship."""

        if not isinstance(snapshot, MemoryScoreSnapshot):
            raise TypeError("snapshot must be a MemoryScoreSnapshot")
        if self.score_snapshot_sha256 != snapshot.snapshot_sha256:
            raise StaleMemorySnapshotError(
                "assignment is bound to a different memory snapshot"
            )
        if (
            self.exact_context_hash != snapshot.exact_context_hash
            or self.estimand_stratum_hash != snapshot.estimand_stratum_hash
        ):
            raise ValueError("assignment and snapshot strata differ")
        decision = self.selection_decision
        expected_refs = tuple(entry.reference for entry in snapshot.entries)
        if decision.eligible != expected_refs:
            raise ValueError(
                "selection decision eligible set differs from the score snapshot"
            )
        observed_scores = tuple(score for _, score in decision.score_snapshot)
        snapshot_scores = tuple(entry.retrieval_score for entry in snapshot.entries)
        if self.arm is MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL:
            observed_multiset = Counter(score.hex() for score in observed_scores)
            snapshot_multiset = Counter(score.hex() for score in snapshot_scores)
            if observed_multiset != snapshot_multiset:
                raise ValueError(
                    "score-shuffled control must preserve the snapshot score multiset"
                )
        elif observed_scores != snapshot_scores:
            raise ValueError(
                "selection decision scores differ from the bound memory snapshot"
            )

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready record used by plan hashing."""

        return {
            "schema_version": 1,
            "credit_unit_id": self.credit_unit_id.value,
            "exact_context_hash": self.exact_context_hash,
            "estimand_stratum_hash": self.estimand_stratum_hash,
            "block_id": self.block_id,
            "arm": self.arm.value,
            "selection_decision": _decision_record(self.selection_decision),
            "selection_decision_sha256": self.selection_decision_sha256,
            "score_snapshot_sha256": self.score_snapshot_sha256,
            "prompt_shape_sha256": self.prompt_shape_sha256,
            "credit_mode": self.credit_mode.value,
        }


class StaleMemorySnapshotError(ValueError):
    """Raised when an assignment or wave refers to a superseded checkpoint."""


class IncompleteMemoryWaveError(ValueError):
    """Raised when a wave cannot be sealed with exactly one receipt per unit."""


@dataclass(frozen=True, slots=True)
class CausalSearchScore:
    """Auditable score and support for one exact insight version."""

    reference: InsightRef
    prior_score: float
    effect_estimate: float | None
    treated_trials: int
    control_trials: int
    treated_effective_sample_size: float
    control_effective_sample_size: float
    effective_support: float
    shrinkage: float
    posterior_mean: float
    uncertainty_bonus: float
    retrieval_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.reference, InsightRef):
            raise TypeError("reference must be an InsightRef")
        for name in (
            "prior_score",
            "treated_effective_sample_size",
            "control_effective_sample_size",
            "effective_support",
            "shrinkage",
            "posterior_mean",
            "uncertainty_bonus",
            "retrieval_score",
        ):
            _require_canonical_float(getattr(self, name), name)
        if self.effect_estimate is not None:
            _require_canonical_float(self.effect_estimate, "effect_estimate")
        for name in ("treated_trials", "control_trials"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if (
            min(
                self.treated_effective_sample_size,
                self.control_effective_sample_size,
                self.effective_support,
                self.uncertainty_bonus,
            )
            < 0
        ):
            raise ValueError("sample support and uncertainty cannot be negative")
        if not 0.0 <= self.shrinkage < 1.0:
            raise ValueError("shrinkage must lie in [0,1)")

    @property
    def identified(self) -> bool:
        return self.effect_estimate is not None

    def to_record(self) -> dict[str, object]:
        return {
            "reference": _ref_record(self.reference),
            "prior_score_hex": self.prior_score.hex(),
            "effect_estimate_hex": (
                None if self.effect_estimate is None else self.effect_estimate.hex()
            ),
            "treated_trials": self.treated_trials,
            "control_trials": self.control_trials,
            "treated_ess_hex": self.treated_effective_sample_size.hex(),
            "control_ess_hex": self.control_effective_sample_size.hex(),
            "effective_support_hex": self.effective_support.hex(),
            "shrinkage_hex": self.shrinkage.hex(),
            "posterior_mean_hex": self.posterior_mean.hex(),
            "uncertainty_bonus_hex": self.uncertainty_bonus.hex(),
            "retrieval_score_hex": self.retrieval_score.hex(),
        }


class MemoryTrialTerminalStatus(str, Enum):
    """Terminal classification used by the frozen ITT policy."""

    SUCCEEDED = "succeeded"
    MODEL_FAILURE = "model_or_schema_failure"
    CANDIDATE_FAILURE = "candidate_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@dataclass(frozen=True, slots=True)
class MemoryAssignmentReceipt:
    """One terminal result; publication order has no scoring meaning."""

    assignment_sha256: str
    credit_unit_id: OperatorInvocationId
    status: MemoryTrialTerminalStatus
    candidate_ids: tuple[CandidateId, ...] = ()
    observed_reward: float | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.assignment_sha256, "assignment_sha256")
        if not isinstance(self.credit_unit_id, OperatorInvocationId):
            raise TypeError("credit_unit_id must be an OperatorInvocationId")
        if not isinstance(self.status, MemoryTrialTerminalStatus):
            raise TypeError("status must be a MemoryTrialTerminalStatus")
        if type(self.candidate_ids) is not tuple or any(
            not isinstance(value, CandidateId) for value in self.candidate_ids
        ):
            raise TypeError("candidate_ids must be a tuple of CandidateId values")
        if self.candidate_ids != tuple(sorted(set(self.candidate_ids))):
            raise ValueError("candidate_ids must be unique and canonically sorted")
        if self.status is MemoryTrialTerminalStatus.SUCCEEDED:
            if not self.candidate_ids:
                raise ValueError("a successful receipt must identify a candidate")
            _require_canonical_float(self.observed_reward, "observed_reward")
        elif self.observed_reward is not None:
            raise ValueError("failure receipts cannot carry an observed reward")

    def to_record(self) -> dict[str, object]:
        return {
            "assignment_sha256": self.assignment_sha256,
            "credit_unit_id": self.credit_unit_id.value,
            "status": self.status.value,
            "candidate_ids": [value.value for value in self.candidate_ids],
            "observed_reward_hex": (
                None if self.observed_reward is None else self.observed_reward.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class CausalMemoryObservation:
    """A sealed, non-infrastructure assignment with its ITT reward."""

    assignment: ResolvedInsightAssignment
    status: MemoryTrialTerminalStatus
    candidate_ids: tuple[CandidateId, ...]
    credited_reward: float
    reward_definition_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, ResolvedInsightAssignment):
            raise TypeError("assignment must be a ResolvedInsightAssignment")
        if self.assignment.arm is not MemoryAssignmentArm.DIAGNOSTIC:
            raise ValueError("causal memory observations must be diagnostic")
        if self.status not in {
            MemoryTrialTerminalStatus.SUCCEEDED,
            MemoryTrialTerminalStatus.MODEL_FAILURE,
            MemoryTrialTerminalStatus.CANDIDATE_FAILURE,
        }:
            raise ValueError("infrastructure failures cannot enter causal evidence")
        if type(self.candidate_ids) is not tuple or any(
            not isinstance(value, CandidateId) for value in self.candidate_ids
        ):
            raise TypeError("candidate_ids must be a tuple of CandidateId values")
        if self.candidate_ids != tuple(sorted(set(self.candidate_ids))):
            raise ValueError("candidate_ids must be unique and canonically sorted")
        _require_canonical_float(self.credited_reward, "credited_reward")
        _require_sha256(self.reward_definition_hash, "reward_definition_hash")

    @property
    def reward_was_imputed(self) -> bool:
        return self.status is not MemoryTrialTerminalStatus.SUCCEEDED

    def to_trial(self) -> InsightTrial:
        return InsightTrial(
            credit_unit_id=self.assignment.credit_unit_id,
            candidate_ids=self.candidate_ids,
            reward_definition_hash=self.reward_definition_hash,
            decision=self.assignment.selection_decision,
            reward=self.credited_reward,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "assignment": self.assignment.to_record(),
            "assignment_sha256": self.assignment.assignment_sha256,
            "status": self.status.value,
            "candidate_ids": [value.value for value in self.candidate_ids],
            "credited_reward_hex": self.credited_reward.hex(),
            "reward_definition_hash": self.reward_definition_hash,
            "reward_was_imputed": self.reward_was_imputed,
        }


@dataclass(frozen=True, slots=True)
class MemoryScoreSnapshot:
    """An immutable causal score checkpoint and its complete valid evidence."""

    exact_context_hash: str
    estimand_stratum_hash: str
    checkpoint_index: int
    entries: tuple[CausalSearchScore, ...]
    observations: tuple[CausalMemoryObservation, ...]
    prior_effective_sample_size: float
    uncertainty_scale: float
    exploration_weight: float
    reward_definition_hash: str | None = None
    parent_snapshot_sha256: str | None = None
    source_wave_sha256: str | None = None
    scoring_policy_id: str = "min_ess_shrunken_causal_ucb"
    scoring_policy_version: int = 1
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.exact_context_hash, "exact_context_hash")
        _require_sha256(self.estimand_stratum_hash, "estimand_stratum_hash")
        if type(self.checkpoint_index) is not int or self.checkpoint_index < 0:
            raise ValueError("checkpoint_index must be a non-negative exact integer")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("entries must be a non-empty exact tuple")
        if any(not isinstance(entry, CausalSearchScore) for entry in self.entries):
            raise TypeError("entries must contain CausalSearchScore values")
        references = tuple(entry.reference for entry in self.entries)
        if references != tuple(sorted(set(references))):
            raise ValueError("score entries must be unique and canonically sorted")
        if type(self.observations) is not tuple or any(
            not isinstance(value, CausalMemoryObservation)
            for value in self.observations
        ):
            raise TypeError(
                "observations must be a tuple of CausalMemoryObservation values"
            )
        observation_keys = tuple(
            value.assignment.assignment_sha256 for value in self.observations
        )
        if observation_keys != tuple(sorted(set(observation_keys))):
            raise ValueError("observations must be unique and canonically sorted")
        for observation in self.observations:
            assignment = observation.assignment
            if (
                assignment.exact_context_hash != self.exact_context_hash
                or assignment.estimand_stratum_hash != self.estimand_stratum_hash
            ):
                raise ValueError(
                    "snapshot observations must share its estimand stratum"
                )
        for name in (
            "prior_effective_sample_size",
            "uncertainty_scale",
            "exploration_weight",
        ):
            _require_canonical_float(getattr(self, name), name)
        if self.prior_effective_sample_size <= 0:
            raise ValueError("prior_effective_sample_size must be positive")
        if self.uncertainty_scale < 0 or self.exploration_weight < 0:
            raise ValueError("uncertainty parameters cannot be negative")
        if self.scoring_policy_id != "min_ess_shrunken_causal_ucb":
            raise ValueError("unsupported scoring_policy_id")
        if (
            type(self.scoring_policy_version) is not int
            or self.scoring_policy_version != 1
        ):
            raise ValueError("unsupported scoring_policy_version")

        if self.checkpoint_index == 0:
            if (
                self.observations
                or self.reward_definition_hash is not None
                or self.parent_snapshot_sha256 is not None
                or self.source_wave_sha256 is not None
            ):
                raise ValueError("a genesis snapshot cannot contain wave evidence")
        else:
            if not self.observations:
                raise ValueError("a non-genesis snapshot must contain evidence")
            if self.reward_definition_hash is None:
                raise ValueError("a non-genesis snapshot requires a reward definition")
            _require_sha256(self.reward_definition_hash, "reward_definition_hash")
            if self.parent_snapshot_sha256 is None or self.source_wave_sha256 is None:
                raise ValueError(
                    "a non-genesis snapshot requires parent and wave hashes"
                )
            _require_sha256(self.parent_snapshot_sha256, "parent_snapshot_sha256")
            _require_sha256(self.source_wave_sha256, "source_wave_sha256")
            if any(
                observation.reward_definition_hash != self.reward_definition_hash
                for observation in self.observations
            ):
                raise ValueError("snapshot cannot mix reward definitions")

        for entry in self.entries:
            support = min(
                entry.treated_effective_sample_size,
                entry.control_effective_sample_size,
            )
            shrinkage = support / (support + self.prior_effective_sample_size)
            effect = 0.0 if entry.effect_estimate is None else entry.effect_estimate
            posterior_mean = entry.prior_score + shrinkage * effect
            uncertainty = self.uncertainty_scale / math.sqrt(
                support + self.prior_effective_sample_size
            )
            retrieval = posterior_mean + self.exploration_weight * uncertainty
            expected = (
                support,
                shrinkage,
                posterior_mean,
                uncertainty,
                retrieval,
            )
            observed = (
                entry.effective_support,
                entry.shrinkage,
                entry.posterior_mean,
                entry.uncertainty_bonus,
                entry.retrieval_score,
            )
            if observed != expected:
                raise ValueError(
                    "score entry does not match the frozen scoring formula"
                )

        # Re-run the existing causal estimator at this trust boundary.  Besides
        # defending the formula, this rejects duplicate credit/candidate units
        # and silent mixtures of assignment or reward strata.
        trials = tuple(observation.to_trial() for observation in self.observations)
        for entry in self.entries:
            estimate = estimate_marginal_effect(
                trials,
                entry.reference,
                context_hash=self.exact_context_hash,
            )
            if (
                entry.effect_estimate != estimate.effect
                or entry.treated_trials != estimate.treated_trials
                or entry.control_trials != estimate.control_trials
                or entry.treated_effective_sample_size
                != estimate.treated_effective_sample_size
                or entry.control_effective_sample_size
                != estimate.control_effective_sample_size
            ):
                raise ValueError("score entry does not match its causal evidence")

        object.__setattr__(
            self,
            "snapshot_sha256",
            _hash_record(_SCORE_SNAPSHOT_DOMAIN, self.to_record()),
        )

    @property
    def retrieval_scores(self) -> dict[InsightRef, float]:
        return {entry.reference: entry.retrieval_score for entry in self.entries}

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "exact_context_hash": self.exact_context_hash,
            "estimand_stratum_hash": self.estimand_stratum_hash,
            "checkpoint_index": self.checkpoint_index,
            "entries": [entry.to_record() for entry in self.entries],
            "observations": [value.to_record() for value in self.observations],
            "prior_effective_sample_size_hex": (self.prior_effective_sample_size.hex()),
            "uncertainty_scale_hex": self.uncertainty_scale.hex(),
            "exploration_weight_hex": self.exploration_weight.hex(),
            "reward_definition_hash": self.reward_definition_hash,
            "parent_snapshot_sha256": self.parent_snapshot_sha256,
            "source_wave_sha256": self.source_wave_sha256,
            "scoring_policy_id": self.scoring_policy_id,
            "scoring_policy_version": self.scoring_policy_version,
        }


@dataclass(frozen=True, slots=True)
class CausalSearchScorePolicy:
    """Frozen support-shrinkage and uncertainty-aware retrieval policy."""

    prior_effective_sample_size: float = 4.0
    uncertainty_scale: float = 1.0
    exploration_weight: float = 0.25
    policy_id: str = "min_ess_shrunken_causal_ucb"
    policy_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "prior_effective_sample_size",
            "uncertainty_scale",
            "exploration_weight",
        ):
            _require_canonical_float(getattr(self, name), name)
        if self.prior_effective_sample_size <= 0:
            raise ValueError("prior_effective_sample_size must be positive")
        if self.uncertainty_scale < 0 or self.exploration_weight < 0:
            raise ValueError("uncertainty parameters cannot be negative")
        if self.policy_id != "min_ess_shrunken_causal_ucb":
            raise ValueError("unsupported causal score policy_id")
        if type(self.policy_version) is not int or self.policy_version != 1:
            raise ValueError("unsupported causal score policy_version")

    def genesis(
        self,
        *,
        exact_context_hash: str,
        estimand_stratum_hash: str,
        priors: Mapping[InsightRef, Real],
    ) -> MemoryScoreSnapshot:
        """Create checkpoint zero without manufacturing causal evidence."""

        if not isinstance(priors, Mapping) or not priors:
            raise ValueError("priors must be a non-empty mapping")
        if any(not isinstance(reference, InsightRef) for reference in priors):
            raise TypeError("prior keys must be InsightRef values")
        entries = tuple(
            self._entry(
                reference=reference,
                prior_score=_canonical_float(priors[reference], "prior score"),
                effect_estimate=None,
                treated_trials=0,
                control_trials=0,
                treated_ess=0.0,
                control_ess=0.0,
            )
            for reference in sorted(priors)
        )
        return MemoryScoreSnapshot(
            exact_context_hash=exact_context_hash,
            estimand_stratum_hash=estimand_stratum_hash,
            checkpoint_index=0,
            entries=entries,
            observations=(),
            prior_effective_sample_size=self.prior_effective_sample_size,
            uncertainty_scale=self.uncertainty_scale,
            exploration_weight=self.exploration_weight,
            scoring_policy_id=self.policy_id,
            scoring_policy_version=self.policy_version,
        )

    def score_evidence(
        self,
        *,
        parent: MemoryScoreSnapshot,
        observations: Sequence[CausalMemoryObservation],
        reward_definition_hash: str,
        source_wave_sha256: str,
    ) -> MemoryScoreSnapshot:
        """Recompute one immutable checkpoint from all sealed evidence."""

        self._require_compatible(parent)
        _require_sha256(reward_definition_hash, "reward_definition_hash")
        _require_sha256(source_wave_sha256, "source_wave_sha256")
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise TypeError("observations must be a sequence")
        current = tuple(observations)
        if not current:
            raise ValueError("a checkpoint requires at least one new observation")
        if any(not isinstance(value, CausalMemoryObservation) for value in current):
            raise TypeError("observations must contain CausalMemoryObservation values")
        combined = tuple(
            sorted(
                (*parent.observations, *current),
                key=lambda value: value.assignment.assignment_sha256,
            )
        )
        trials = tuple(observation.to_trial() for observation in combined)
        entries = []
        for prior_entry in parent.entries:
            estimate = estimate_marginal_effect(
                trials,
                prior_entry.reference,
                context_hash=parent.exact_context_hash,
            )
            entries.append(
                self._entry(
                    reference=prior_entry.reference,
                    prior_score=prior_entry.prior_score,
                    effect_estimate=estimate.effect,
                    treated_trials=estimate.treated_trials,
                    control_trials=estimate.control_trials,
                    treated_ess=estimate.treated_effective_sample_size,
                    control_ess=estimate.control_effective_sample_size,
                )
            )
        return MemoryScoreSnapshot(
            exact_context_hash=parent.exact_context_hash,
            estimand_stratum_hash=parent.estimand_stratum_hash,
            checkpoint_index=parent.checkpoint_index + 1,
            entries=tuple(entries),
            observations=combined,
            prior_effective_sample_size=self.prior_effective_sample_size,
            uncertainty_scale=self.uncertainty_scale,
            exploration_weight=self.exploration_weight,
            reward_definition_hash=reward_definition_hash,
            parent_snapshot_sha256=parent.snapshot_sha256,
            source_wave_sha256=source_wave_sha256,
            scoring_policy_id=self.policy_id,
            scoring_policy_version=self.policy_version,
        )

    def _entry(
        self,
        *,
        reference: InsightRef,
        prior_score: float,
        effect_estimate: float | None,
        treated_trials: int,
        control_trials: int,
        treated_ess: float,
        control_ess: float,
    ) -> CausalSearchScore:
        support = min(treated_ess, control_ess)
        shrinkage = support / (support + self.prior_effective_sample_size)
        effect = 0.0 if effect_estimate is None else effect_estimate
        posterior_mean = prior_score + shrinkage * effect
        uncertainty = self.uncertainty_scale / math.sqrt(
            support + self.prior_effective_sample_size
        )
        retrieval = posterior_mean + self.exploration_weight * uncertainty
        return CausalSearchScore(
            reference=reference,
            prior_score=prior_score,
            effect_estimate=effect_estimate,
            treated_trials=treated_trials,
            control_trials=control_trials,
            treated_effective_sample_size=treated_ess,
            control_effective_sample_size=control_ess,
            effective_support=support,
            shrinkage=shrinkage,
            posterior_mean=posterior_mean,
            uncertainty_bonus=uncertainty,
            retrieval_score=retrieval,
        )

    def _require_compatible(self, snapshot: MemoryScoreSnapshot) -> None:
        if not isinstance(snapshot, MemoryScoreSnapshot):
            raise TypeError("snapshot must be a MemoryScoreSnapshot")
        observed = (
            snapshot.prior_effective_sample_size,
            snapshot.uncertainty_scale,
            snapshot.exploration_weight,
            snapshot.scoring_policy_id,
            snapshot.scoring_policy_version,
        )
        expected = (
            self.prior_effective_sample_size,
            self.uncertainty_scale,
            self.exploration_weight,
            self.policy_id,
            self.policy_version,
        )
        if observed != expected:
            raise ValueError("score policy differs from the parent snapshot")


@dataclass(frozen=True, slots=True)
class FrozenDiagnosticMemoryWave:
    """Complete pre-call diagnostic assignments against one score checkpoint."""

    wave_id: str
    prior_snapshot: MemoryScoreSnapshot
    assignments: tuple[ResolvedInsightAssignment, ...]
    reward_definition_hash: str
    no_yield_reward: float
    wave_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_block_id(self.wave_id, "wave_id")
        if not isinstance(self.prior_snapshot, MemoryScoreSnapshot):
            raise TypeError("prior_snapshot must be a MemoryScoreSnapshot")
        if type(self.assignments) is not tuple or not self.assignments:
            raise ValueError("assignments must be a non-empty exact tuple")
        if any(
            not isinstance(value, ResolvedInsightAssignment)
            for value in self.assignments
        ):
            raise TypeError("assignments must contain ResolvedInsightAssignment values")
        hashes = tuple(value.assignment_sha256 for value in self.assignments)
        if hashes != tuple(sorted(set(hashes))):
            raise ValueError("assignments must be unique and canonically sorted")
        credit_ids = tuple(value.credit_unit_id for value in self.assignments)
        if len(set(credit_ids)) != len(credit_ids):
            raise ValueError("a wave cannot repeat a credit unit")
        for assignment in self.assignments:
            if assignment.arm is not MemoryAssignmentArm.DIAGNOSTIC:
                raise ValueError("a diagnostic wave can contain only diagnostic arms")
            assignment.validate_against_snapshot(self.prior_snapshot)
        first = self.assignments[0].selection_decision
        stratum = (
            first.eligible,
            first.subset_size,
            first.exploration_probability,
            first.policy_id,
            first.policy_version,
        )
        if any(
            (
                value.selection_decision.eligible,
                value.selection_decision.subset_size,
                value.selection_decision.exploration_probability,
                value.selection_decision.policy_id,
                value.selection_decision.policy_version,
            )
            != stratum
            for value in self.assignments[1:]
        ):
            raise ValueError("a diagnostic wave cannot mix assignment-law strata")
        _require_sha256(self.reward_definition_hash, "reward_definition_hash")
        _require_canonical_float(self.no_yield_reward, "no_yield_reward")
        if (
            self.prior_snapshot.reward_definition_hash is not None
            and self.prior_snapshot.reward_definition_hash
            != self.reward_definition_hash
        ):
            raise ValueError("a diagnostic lineage cannot change reward definition")
        object.__setattr__(
            self,
            "wave_sha256",
            _hash_record(_WAVE_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "wave_id": self.wave_id,
            "prior_snapshot_sha256": self.prior_snapshot.snapshot_sha256,
            "assignment_sha256s": [
                value.assignment_sha256 for value in self.assignments
            ],
            "reward_definition_hash": self.reward_definition_hash,
            "no_yield_reward_hex": self.no_yield_reward.hex(),
        }


class MemoryCheckpointClosureStatus(str, Enum):
    SEALED = "sealed"
    INVALIDATED_INFRASTRUCTURE = "invalidated_infrastructure"


@dataclass(frozen=True, slots=True)
class MemoryCheckpointClosure:
    """Explicit result: exactly one checkpoint or an invalidated wave."""

    wave_sha256: str
    status: MemoryCheckpointClosureStatus
    receipts: tuple[MemoryAssignmentReceipt, ...]
    observations: tuple[CausalMemoryObservation, ...]
    snapshot: MemoryScoreSnapshot | None

    def __post_init__(self) -> None:
        _require_sha256(self.wave_sha256, "wave_sha256")
        if not isinstance(self.status, MemoryCheckpointClosureStatus):
            raise TypeError("status must be a MemoryCheckpointClosureStatus")
        if type(self.receipts) is not tuple or any(
            not isinstance(value, MemoryAssignmentReceipt) for value in self.receipts
        ):
            raise TypeError("receipts must contain MemoryAssignmentReceipt values")
        receipt_hashes = tuple(value.assignment_sha256 for value in self.receipts)
        if receipt_hashes != tuple(sorted(set(receipt_hashes))):
            raise ValueError("closure receipts must be unique and canonically sorted")
        if type(self.observations) is not tuple or any(
            not isinstance(value, CausalMemoryObservation)
            for value in self.observations
        ):
            raise TypeError("observations must contain CausalMemoryObservation values")
        if self.status is MemoryCheckpointClosureStatus.SEALED:
            if self.snapshot is None or not self.observations:
                raise ValueError(
                    "a sealed closure requires observations and a snapshot"
                )
            if any(
                receipt.status is MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE
                for receipt in self.receipts
            ):
                raise ValueError(
                    "a sealed closure cannot contain infrastructure failure"
                )
        elif self.snapshot is not None or self.observations:
            raise ValueError(
                "an infrastructure-invalidated closure cannot publish evidence"
            )
        elif not any(
            receipt.status is MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE
            for receipt in self.receipts
        ):
            raise ValueError(
                "an infrastructure-invalidated closure requires an infrastructure failure"
            )


@dataclass(frozen=True, slots=True)
class WaveSealedCheckpointBuilder:
    """Atomically convert one complete terminal wave into causal memory."""

    score_policy: CausalSearchScorePolicy

    def __post_init__(self) -> None:
        if not isinstance(self.score_policy, CausalSearchScorePolicy):
            raise TypeError("score_policy must be a CausalSearchScorePolicy")

    def close(
        self,
        wave: FrozenDiagnosticMemoryWave,
        receipts: Sequence[MemoryAssignmentReceipt],
    ) -> MemoryCheckpointClosure:
        """Seal only an exact complete receipt set, independent of completion order."""

        if not isinstance(wave, FrozenDiagnosticMemoryWave):
            raise TypeError("wave must be a FrozenDiagnosticMemoryWave")
        self.score_policy._require_compatible(wave.prior_snapshot)
        if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
            raise TypeError("receipts must be a sequence")
        terminal = tuple(receipts)
        if any(not isinstance(value, MemoryAssignmentReceipt) for value in terminal):
            raise TypeError("receipts must contain MemoryAssignmentReceipt values")
        by_hash: dict[str, MemoryAssignmentReceipt] = {}
        for receipt in terminal:
            if receipt.assignment_sha256 in by_hash:
                raise IncompleteMemoryWaveError(
                    "a diagnostic assignment has more than one terminal receipt"
                )
            by_hash[receipt.assignment_sha256] = receipt
        expected = {value.assignment_sha256 for value in wave.assignments}
        observed = set(by_hash)
        if expected != observed:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise IncompleteMemoryWaveError(
                f"receipt set differs from frozen wave: missing={missing}, extra={extra}"
            )
        assignments = {value.assignment_sha256: value for value in wave.assignments}
        canonical_receipts = tuple(by_hash[key] for key in sorted(by_hash))
        for receipt in canonical_receipts:
            assignment = assignments[receipt.assignment_sha256]
            if receipt.credit_unit_id != assignment.credit_unit_id:
                raise ValueError("receipt credit unit differs from its assignment")

        if any(
            value.status is MemoryTrialTerminalStatus.INFRASTRUCTURE_FAILURE
            for value in canonical_receipts
        ):
            return MemoryCheckpointClosure(
                wave_sha256=wave.wave_sha256,
                status=MemoryCheckpointClosureStatus.INVALIDATED_INFRASTRUCTURE,
                receipts=canonical_receipts,
                observations=(),
                snapshot=None,
            )

        observations = []
        for receipt in canonical_receipts:
            reward = (
                receipt.observed_reward
                if receipt.status is MemoryTrialTerminalStatus.SUCCEEDED
                else wave.no_yield_reward
            )
            assert reward is not None  # Enforced by MemoryAssignmentReceipt.
            observations.append(
                CausalMemoryObservation(
                    assignment=assignments[receipt.assignment_sha256],
                    status=receipt.status,
                    candidate_ids=receipt.candidate_ids,
                    credited_reward=reward,
                    reward_definition_hash=wave.reward_definition_hash,
                )
            )
        canonical_observations = tuple(
            sorted(
                observations,
                key=lambda value: value.assignment.assignment_sha256,
            )
        )
        snapshot = self.score_policy.score_evidence(
            parent=wave.prior_snapshot,
            observations=canonical_observations,
            reward_definition_hash=wave.reward_definition_hash,
            source_wave_sha256=wave.wave_sha256,
        )
        return MemoryCheckpointClosure(
            wave_sha256=wave.wave_sha256,
            status=MemoryCheckpointClosureStatus.SEALED,
            receipts=canonical_receipts,
            observations=canonical_observations,
            snapshot=snapshot,
        )


class _NoRandom:
    def randrange(self, stop: int) -> int:  # pragma: no cover - exact 0/1 only.
        raise AssertionError(f"unexpected branch draw with stop={stop}")

    def sample(self, population, k: int):  # pragma: no cover - exploit only.
        raise AssertionError("unexpected subset sample")


class _ExactSubsetRandom(_NoRandom):
    def __init__(self, selected: tuple[InsightRef, ...]) -> None:
        self._selected = selected

    def sample(self, population, k: int) -> list[InsightRef]:
        if k != len(self._selected) or not set(self._selected).issubset(population):
            raise RuntimeError("internal exact subset does not match selector inputs")
        return list(self._selected)


def _unrank_combination(
    values: tuple[InsightRef, ...], subset_size: int, rank: int
) -> tuple[InsightRef, ...]:
    count = len(values)
    combination_count = math.comb(count, subset_size)
    if type(rank) is not int or rank < 0 or rank >= combination_count:
        raise ValueError(
            f"subset_rank must lie in [0, {combination_count}) for this snapshot"
        )
    selected_indices = []
    remaining_rank = rank
    start = 0
    for position in range(subset_size):
        remaining_positions = subset_size - position - 1
        for index in range(start, count):
            suffix_count = math.comb(count - index - 1, remaining_positions)
            if remaining_rank < suffix_count:
                selected_indices.append(index)
                start = index + 1
                break
            remaining_rank -= suffix_count
    return tuple(values[index] for index in selected_indices)


def _unrank_permutation(values: tuple[float, ...], rank: int) -> tuple[float, ...]:
    permutation_count = math.factorial(len(values))
    if type(rank) is not int or rank < 0 or rank >= permutation_count:
        raise ValueError(
            f"permutation_rank must lie in [0, {permutation_count}) for this snapshot"
        )
    remaining = list(values)
    result = []
    remaining_rank = rank
    for slots in range(len(values), 0, -1):
        block_size = math.factorial(slots - 1)
        index, remaining_rank = divmod(remaining_rank, block_size)
        result.append(remaining.pop(index))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DeterministicMemoryControlPolicy:
    """Exact replayable controls parameterized by recorded integer ranks.

    A uniformly sampled integer in ``[0, n!)`` yields an exact uniform law over
    labelled score permutations; a uniformly sampled integer in ``[0, C(n,k))``
    yields an exact uniform law over k-subsets.  This policy performs only the
    deterministic rank-to-realization mapping so randomization stays outside
    provider code and is trivial to replay.
    """

    def adaptive(
        self,
        *,
        snapshot: MemoryScoreSnapshot,
        subset_size: int,
    ) -> InsightSelectionDecision:
        return self._select(
            snapshot=snapshot,
            subset_size=subset_size,
            scores=snapshot.retrieval_scores,
            exploration_probability=Fraction(0),
            rng=_NoRandom(),
        )

    def score_shuffled(
        self,
        *,
        snapshot: MemoryScoreSnapshot,
        subset_size: int,
        permutation_rank: int,
    ) -> InsightSelectionDecision:
        references = tuple(entry.reference for entry in snapshot.entries)
        values = tuple(entry.retrieval_score for entry in snapshot.entries)
        permuted = _unrank_permutation(values, permutation_rank)
        return self._select(
            snapshot=snapshot,
            subset_size=subset_size,
            scores=dict(zip(references, permuted, strict=True)),
            exploration_probability=Fraction(0),
            rng=_NoRandom(),
        )

    def uniform(
        self,
        *,
        snapshot: MemoryScoreSnapshot,
        subset_size: int,
        subset_rank: int,
    ) -> InsightSelectionDecision:
        references = tuple(entry.reference for entry in snapshot.entries)
        if type(subset_size) is not int or subset_size < 0:
            raise ValueError("subset_size must be a non-negative exact integer")
        if subset_size > len(references):
            raise ValueError("subset_size cannot exceed the snapshot size")
        selected = _unrank_combination(references, subset_size, subset_rank)
        return self._select(
            snapshot=snapshot,
            subset_size=subset_size,
            scores=snapshot.retrieval_scores,
            exploration_probability=Fraction(1),
            rng=_ExactSubsetRandom(selected),
        )

    @staticmethod
    def _select(
        *,
        snapshot: MemoryScoreSnapshot,
        subset_size: int,
        scores: Mapping[InsightRef, Real],
        exploration_probability: Fraction,
        rng: _NoRandom,
    ) -> InsightSelectionDecision:
        if not isinstance(snapshot, MemoryScoreSnapshot):
            raise TypeError("snapshot must be a MemoryScoreSnapshot")
        return EpsilonGreedySubsetSelector(exploration_probability).select(
            context_hash=snapshot.exact_context_hash,
            eligible=tuple(entry.reference for entry in snapshot.entries),
            scores=scores,
            subset_size=subset_size,
            rng=rng,
        )


__all__ = [
    "CausalMemoryObservation",
    "CausalSearchScore",
    "CausalSearchScorePolicy",
    "DelayedCreditMode",
    "DeterministicMemoryControlPolicy",
    "FrozenDiagnosticMemoryWave",
    "IncompleteMemoryWaveError",
    "MemoryAssignmentArm",
    "MemoryAssignmentReceipt",
    "MemoryCheckpointClosure",
    "MemoryCheckpointClosureStatus",
    "MemoryScoreSnapshot",
    "MemoryTrialTerminalStatus",
    "ResolvedInsightAssignment",
    "StaleMemorySnapshotError",
    "WaveSealedCheckpointBuilder",
    "insight_selection_decision_sha256",
]

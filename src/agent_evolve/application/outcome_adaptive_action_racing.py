"""Outcome-adaptive allocation inside one sealed materialized-action market.

The policy in this module is deliberately narrower than proposal generation.
It receives a workload-opaque market whose candidates were all materialized
before current outcomes existed.  It first chooses a diagnostic pilot using
only portable action descriptors.  After real pilot outcomes are durably
available, it chooses one continuation action at a time.

This is a normal sequential-optimization information boundary:

* unobserved candidate outcomes are never accepted by the policy;
* the proposal market and real-evaluation budget remain fixed;
* every decision binds the exact observed outcome hashes that precede it; and
* one continuation seat is randomized over an authenticated exploration pool.

The policy knows no workload, objective, configuration field, model, provider,
or prompt.  Callers project those concerns into opaque lane/operator/semantic
cells and a normalized, outcome-blind prior score.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_ID = (
    "outcome_adaptive_action_racing"
)
OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_VERSION = 2
OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION = 3
OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION = 4
OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION = 5
OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION = 6
OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION = 7

_CAUSAL_SET_POLICY_VERSIONS = frozenset(
    {
        OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION,
        OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION,
    }
)

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_POLICY_DEFINITION_DOMAIN = (
    b"agent-evolve:outcome-adaptive-action-racing-definition:v1\x00"
)
_OUTCOME_DOMAIN = b"agent-evolve:adaptive-action-outcome:v1\x00"
_SET_OUTCOME_DOMAIN = b"agent-evolve:adaptive-action-set-outcome:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:adaptive-action-racing-decision:v1\x00"
_DIRECTIVE_DOMAIN = (
    b"agent-evolve:adaptive-action-allocation-directive:v1\x00"
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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _require_probability(value: float, *, name: str) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a finite probability")


def _stable_unit_interval(*parts: object) -> float:
    payload = _canonical_json(list(parts))
    numerator = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return numerator / float(2**64)


@dataclass(frozen=True, slots=True, order=True)
class AdaptiveActionFactorCell:
    """One opaque, pre-evaluation factor level used for audit strata."""

    family_id: str
    level_id: str

    def __post_init__(self) -> None:
        _require_token(self.family_id, name="factor family_id")
        _require_token(self.level_id, name="factor level_id")

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "family_id": self.family_id,
            "level_id": self.level_id,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveActionDescriptor:
    """Portable, outcome-blind view of one materialized action."""

    action_sha256: str
    phenotype_sha256: str
    lane_id: str
    operator_id: str
    native_rank: int
    lane_size: int
    prior_score: float
    parent_generated_in_current_run: bool
    semantic_cell_ids: tuple[str, ...] = ()
    factor_cells: tuple[AdaptiveActionFactorCell, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        require_sha256(self.phenotype_sha256, "phenotype_sha256")
        _require_token(self.lane_id, name="lane_id")
        _require_token(self.operator_id, name="operator_id")
        if (
            type(self.native_rank) is not int
            or self.native_rank <= 0
            or type(self.lane_size) is not int
            or self.lane_size <= 0
            or self.native_rank > self.lane_size
        ):
            raise ValueError("native rank must fit the positive lane size")
        _require_probability(self.prior_score, name="prior_score")
        if type(self.parent_generated_in_current_run) is not bool:
            raise TypeError(
                "parent_generated_in_current_run must be exact"
            )
        if (
            type(self.semantic_cell_ids) is not tuple
            or self.semantic_cell_ids
            != tuple(sorted(set(self.semantic_cell_ids)))
        ):
            raise ValueError(
                "semantic_cell_ids must be an exact canonical tuple"
            )
        for value in self.semantic_cell_ids:
            _require_token(value, name="semantic_cell_id")
        if (
            type(self.factor_cells) is not tuple
            or self.factor_cells != tuple(sorted(self.factor_cells))
            or any(
                type(value) is not AdaptiveActionFactorCell
                for value in self.factor_cells
            )
        ):
            raise ValueError(
                "factor_cells must be an exact canonical cell tuple"
            )
        for value in self.factor_cells:
            value.__post_init__()
        if len({value.family_id for value in self.factor_cells}) != len(
            self.factor_cells
        ):
            raise ValueError(
                "factor_cells must contain at most one level per family"
            )

    @property
    def rank_quality(self) -> float:
        """Return one for the lane head and zero for the lane tail."""

        if self.lane_size == 1:
            return 1.0
        return 1.0 - (
            (self.native_rank - 1) / float(self.lane_size - 1)
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "action_sha256": self.action_sha256,
            "phenotype_sha256": self.phenotype_sha256,
            "lane_id": self.lane_id,
            "operator_id": self.operator_id,
            "native_rank": self.native_rank,
            "lane_size": self.lane_size,
            "rank_quality_hex": self.rank_quality.hex(),
            "prior_score_hex": self.prior_score.hex(),
            "parent_generated_in_current_run": (
                self.parent_generated_in_current_run
            ),
            "semantic_cell_ids": list(self.semantic_cell_ids),
        }
        # Preserve byte-identical records for policy versions 2--5.
        if self.factor_cells:
            record["factor_cells"] = [
                value.to_record() for value in self.factor_cells
            ]
        return record


@dataclass(frozen=True, slots=True)
class AdaptiveActionOutcome:
    """One real outcome available before a continuation decision."""

    action_sha256: str
    evaluation_sha256: str
    feasible: bool
    marginal_archive_gain: float
    outcome_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        require_sha256(self.evaluation_sha256, "evaluation_sha256")
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be exact")
        if (
            type(self.marginal_archive_gain) is not float
            or not math.isfinite(self.marginal_archive_gain)
            or self.marginal_archive_gain < 0.0
        ):
            raise ValueError(
                "marginal_archive_gain must be finite and non-negative"
            )
        if not self.feasible and self.marginal_archive_gain != 0.0:
            raise ValueError("an infeasible action cannot contribute gain")
        object.__setattr__(
            self,
            "outcome_sha256",
            _hash(_OUTCOME_DOMAIN, self._unsigned_record()),
        )

    @property
    def positive(self) -> bool:
        return self.marginal_archive_gain > 0.0

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "action_sha256": self.action_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "feasible": self.feasible,
            "marginal_archive_gain_hex": (
                self.marginal_archive_gain.hex()
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "positive": self.positive,
            "outcome_sha256": self.outcome_sha256,
        }


def _validated_action_evaluation_bindings(
    values: tuple[tuple[str, str], ...],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple or (not allow_empty and not values):
        raise ValueError(f"{name} must be a canonical exact tuple")
    bindings: list[tuple[str, str]] = []
    for value in values:
        if type(value) is not tuple or len(value) != 2:
            raise TypeError(f"{name} must contain exact hash pairs")
        action_sha256, evaluation_sha256 = value
        require_sha256(action_sha256, f"{name} action_sha256")
        require_sha256(evaluation_sha256, f"{name} evaluation_sha256")
        bindings.append((action_sha256, evaluation_sha256))
    if bindings != sorted(set(bindings)):
        raise ValueError(f"{name} must be unique and canonical")
    if len({value[0] for value in bindings}) != len(bindings):
        raise ValueError(f"{name} repeats an action identity")
    if len({value[1] for value in bindings}) != len(bindings):
        raise ValueError(f"{name} repeats an evaluation identity")
    return tuple(bindings)


def _require_nonnegative_metric(value: float, *, name: str) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _metric_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


@dataclass(frozen=True, slots=True)
class AdaptiveActionSetOutcome:
    """Workload-opaque causal utility observation for one evaluated wave.

    ``current_wave_fixed_set_gain`` values the current wave against the
    pre-stage archive.  ``conditional_set_gain`` values the same wave after
    every earlier selected evaluation.  Their difference therefore separates
    local saturation (redundancy) from positive set interaction (synergy)
    without exposing objective names, senses, configuration fields, or
    workload semantics to the generic controller.
    """

    prior_action_evaluation_bindings: tuple[tuple[str, str], ...]
    current_action_evaluation_bindings: tuple[tuple[str, str], ...]
    prior_selected_set_gain: float
    current_wave_fixed_set_gain: float
    augmented_selected_set_gain: float
    conditional_set_gain: float
    set_outcome_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        prior = _validated_action_evaluation_bindings(
            self.prior_action_evaluation_bindings,
            name="prior_action_evaluation_bindings",
            allow_empty=True,
        )
        current = _validated_action_evaluation_bindings(
            self.current_action_evaluation_bindings,
            name="current_action_evaluation_bindings",
            allow_empty=False,
        )
        prior_actions = {value[0] for value in prior}
        current_actions = {value[0] for value in current}
        prior_evaluations = {value[1] for value in prior}
        current_evaluations = {value[1] for value in current}
        if prior_actions & current_actions:
            raise ValueError("prior and current bindings repeat an action")
        if prior_evaluations & current_evaluations:
            raise ValueError(
                "prior and current bindings repeat an evaluation"
            )
        for name in (
            "prior_selected_set_gain",
            "current_wave_fixed_set_gain",
            "augmented_selected_set_gain",
            "conditional_set_gain",
        ):
            _require_nonnegative_metric(getattr(self, name), name=name)
        if not _metric_close(
            self.augmented_selected_set_gain,
            self.prior_selected_set_gain + self.conditional_set_gain,
        ):
            raise ValueError(
                "conditional_set_gain does not close the augmented set"
            )
        object.__setattr__(
            self,
            "set_outcome_sha256",
            _hash(_SET_OUTCOME_DOMAIN, self._unsigned_record()),
        )

    @property
    def prior_conditioned_redundancy(self) -> float:
        if _metric_close(
            self.current_wave_fixed_set_gain,
            self.conditional_set_gain,
        ):
            return 0.0
        return max(
            self.current_wave_fixed_set_gain
            - self.conditional_set_gain,
            0.0,
        )

    @property
    def prior_conditioned_synergy(self) -> float:
        if _metric_close(
            self.current_wave_fixed_set_gain,
            self.conditional_set_gain,
        ):
            return 0.0
        return max(
            self.conditional_set_gain
            - self.current_wave_fixed_set_gain,
            0.0,
        )

    @property
    def saturation_fraction(self) -> float:
        if self.current_wave_fixed_set_gain == 0.0:
            return 0.0
        return min(
            self.prior_conditioned_redundancy
            / self.current_wave_fixed_set_gain,
            1.0,
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "prior_action_evaluation_bindings": [
                {
                    "action_sha256": action_sha256,
                    "evaluation_sha256": evaluation_sha256,
                }
                for action_sha256, evaluation_sha256 in (
                    self.prior_action_evaluation_bindings
                )
            ],
            "current_action_evaluation_bindings": [
                {
                    "action_sha256": action_sha256,
                    "evaluation_sha256": evaluation_sha256,
                }
                for action_sha256, evaluation_sha256 in (
                    self.current_action_evaluation_bindings
                )
            ],
            "prior_selected_set_gain_hex": (
                self.prior_selected_set_gain.hex()
            ),
            "current_wave_fixed_set_gain_hex": (
                self.current_wave_fixed_set_gain.hex()
            ),
            "augmented_selected_set_gain_hex": (
                self.augmented_selected_set_gain.hex()
            ),
            "conditional_set_gain_hex": self.conditional_set_gain.hex(),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "prior_conditioned_redundancy_hex": (
                self.prior_conditioned_redundancy.hex()
            ),
            "prior_conditioned_synergy_hex": (
                self.prior_conditioned_synergy.hex()
            ),
            "saturation_fraction_hex": self.saturation_fraction.hex(),
            "set_outcome_sha256": self.set_outcome_sha256,
        }


class AdaptiveActionWave(str, Enum):
    DIAGNOSTIC = "diagnostic"
    ADAPTIVE = "adaptive"
    RANDOMIZED_AUDIT = "randomized_audit"


@dataclass(frozen=True, slots=True)
class AdaptiveActionRacingDecision:
    """Hash-bound action subset chosen at one information cutoff."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    residual_request_sha256: str
    wave: AdaptiveActionWave
    selected_action_sha256s: tuple[str, ...]
    prior_selected_action_sha256s: tuple[str, ...]
    observed_outcome_sha256s: tuple[str, ...]
    observed_set_outcome_sha256s: tuple[str, ...]
    selection_propensity: float
    evidence: FrozenJsonObject
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if type(self.wave) is not AdaptiveActionWave:
            raise TypeError("wave must be an exact AdaptiveActionWave")
        for values, name in (
            (self.selected_action_sha256s, "selected_action_sha256s"),
            (
                self.prior_selected_action_sha256s,
                "prior_selected_action_sha256s",
            ),
            (
                self.observed_outcome_sha256s,
                "observed_outcome_sha256s",
            ),
            (
                self.observed_set_outcome_sha256s,
                "observed_set_outcome_sha256s",
            ),
        ):
            if type(values) is not tuple or values != tuple(
                sorted(set(values))
            ):
                raise ValueError(f"{name} must be an exact canonical tuple")
            for value in values:
                require_sha256(value, name)
        if not self.selected_action_sha256s:
            raise ValueError("a racing decision must select an action")
        if set(self.selected_action_sha256s) & set(
            self.prior_selected_action_sha256s
        ):
            raise ValueError("a racing decision cannot repeat an action")
        _require_probability(
            self.selection_propensity,
            name="selection_propensity",
        )
        if self.selection_propensity <= 0.0:
            raise ValueError("selection propensity must be positive")
        if (
            self.wave is AdaptiveActionWave.DIAGNOSTIC
            and (
                self.observed_outcome_sha256s
                or self.observed_set_outcome_sha256s
            )
        ):
            raise ValueError(
                "diagnostic design cannot observe current outcomes"
            )
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_DECISION_DOMAIN, self._unsigned_record()),
        )

    @property
    def candidate_outcomes_observed(self) -> bool:
        return bool(self.observed_outcome_sha256s)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "wave": self.wave.value,
            "selected_action_sha256s": list(
                self.selected_action_sha256s
            ),
            "prior_selected_action_sha256s": list(
                self.prior_selected_action_sha256s
            ),
            "observed_outcome_sha256s": list(
                self.observed_outcome_sha256s
            ),
            "observed_set_outcome_sha256s": list(
                self.observed_set_outcome_sha256s
            ),
            "candidate_outcomes_observed": (
                self.candidate_outcomes_observed
            ),
            "selection_propensity_hex": self.selection_propensity.hex(),
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "decision_sha256": self.decision_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@dataclass(frozen=True, slots=True)
class AdaptiveActionAllocationDirective:
    """Final outcome-adaptive slate constraint consumed by the broker.

    Unlike ``MaterializedActionAllocationRequirement``, this directive is
    intentionally outcome-conditioned.  It binds the complete chain of
    diagnostic and continuation decisions so downstream code cannot mistake it
    for an outcome-blind requirement.
    """

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    residual_request_sha256: str
    proposal_sha256s: tuple[str, ...]
    required_action_sha256s: tuple[str, ...]
    diagnostic_decision_sha256: str
    continuation_decision_sha256s: tuple[str, ...]
    observed_outcome_sha256s: tuple[str, ...]
    observed_set_outcome_sha256s: tuple[str, ...]
    evidence: FrozenJsonObject
    directive_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        for values, name, allow_empty in (
            (self.proposal_sha256s, "proposal_sha256s", False),
            (
                self.required_action_sha256s,
                "required_action_sha256s",
                False,
            ),
            (
                self.continuation_decision_sha256s,
                "continuation_decision_sha256s",
                False,
            ),
            (
                self.observed_outcome_sha256s,
                "observed_outcome_sha256s",
                False,
            ),
            (
                self.observed_set_outcome_sha256s,
                "observed_set_outcome_sha256s",
                False,
            ),
        ):
            if (
                type(values) is not tuple
                or (not allow_empty and not values)
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must be a canonical exact tuple")
            for value in values:
                require_sha256(value, name)
        require_sha256(
            self.diagnostic_decision_sha256,
            "diagnostic_decision_sha256",
        )
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "directive_sha256",
            _hash(_DIRECTIVE_DOMAIN, self._unsigned_record()),
        )

    @property
    def candidate_outcomes_observed(self) -> bool:
        return True

    @property
    def requirement_sha256(self) -> str:
        """Compatibility identity for the broker's existing constraint slot."""

        return self.directive_sha256

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "allocation_kind": "outcome_adaptive_directive",
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "proposal_sha256s": list(self.proposal_sha256s),
            "required_action_sha256s": list(
                self.required_action_sha256s
            ),
            "diagnostic_decision_sha256": (
                self.diagnostic_decision_sha256
            ),
            "continuation_decision_sha256s": list(
                self.continuation_decision_sha256s
            ),
            "observed_outcome_sha256s": list(
                self.observed_outcome_sha256s
            ),
            "observed_set_outcome_sha256s": list(
                self.observed_set_outcome_sha256s
            ),
            "candidate_outcomes_observed": True,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "directive_sha256": self.directive_sha256,
            "requirement_sha256": self.requirement_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


def _semantic_distance(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> float:
    union = set(left) | set(right)
    if not union:
        return 0.0
    return 1.0 - (len(set(left) & set(right)) / len(union))


def _action_distance(
    left: AdaptiveActionDescriptor,
    right: AdaptiveActionDescriptor,
) -> float:
    """Portable mixed-type distance used only for coverage/uncertainty."""

    left.__post_init__()
    right.__post_init__()
    return (
        (1.0 if left.lane_id != right.lane_id else 0.0)
        + abs(left.rank_quality - right.rank_quality)
        + (
            0.5
            if left.parent_generated_in_current_run
            != right.parent_generated_in_current_run
            else 0.0
        )
        + (0.25 if left.operator_id != right.operator_id else 0.0)
        + 0.25
        * _semantic_distance(
            left.semantic_cell_ids,
            right.semantic_cell_ids,
        )
    ) / 3.0


@dataclass(frozen=True, slots=True)
class OutcomeAdaptiveActionRacingPolicy:
    """Lane-head diagnostic design followed by conservative contextual UCB."""

    diagnostic_slots: int = 4
    randomized_audit_slots: int = 1
    reference_gain_scale: float = 1.0e-4
    reference_gain_evidence_sha256: str = "0" * 64
    prior_strength: float = 2.0
    ucb_strength: float = 1.0
    counterfactual_strength: float = 0.5
    diversity_strength: float = 0.25
    positive_redundancy_strength: float = 0.0
    conditional_saturation_strength: float = 0.0
    conditional_synergy_strength: float = 0.0
    causal_prior_strength: float = 0.0
    randomized_audit_after_directed_steps: int = 0
    audit_exploration_probability: float = 0.0
    exploration_pool_size: int = 4
    trace_alternative_count: int = 0
    stratified_audit_coverage_family_ids: tuple[str, ...] = ()
    stratified_audit_stratum_family_ids: tuple[str, ...] = ()
    minimum_post_audit_optimization_slots: int = 0
    terminal_hierarchical_slots: int = 0
    native_rank_strength: float = 0.0
    random_seed: int = 0
    policy_id: str = OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_ID
    policy_version: int = OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.diagnostic_slots) is not int or self.diagnostic_slots <= 0:
            raise ValueError("diagnostic_slots must be positive")
        if (
            type(self.randomized_audit_slots) is not int
            or self.randomized_audit_slots <= 0
            or self.randomized_audit_slots >= self.diagnostic_slots * 2
        ):
            raise ValueError("randomized_audit_slots must be bounded")
        if (
            type(self.reference_gain_scale) is not float
            or not math.isfinite(self.reference_gain_scale)
            or self.reference_gain_scale <= 0.0
        ):
            raise ValueError("reference_gain_scale must be positive")
        require_sha256(
            self.reference_gain_evidence_sha256,
            "reference_gain_evidence_sha256",
        )
        for value, name in (
            (self.prior_strength, "prior_strength"),
            (self.ucb_strength, "ucb_strength"),
            (
                self.counterfactual_strength,
                "counterfactual_strength",
            ),
            (self.diversity_strength, "diversity_strength"),
            (
                self.positive_redundancy_strength,
                "positive_redundancy_strength",
            ),
            (
                self.conditional_saturation_strength,
                "conditional_saturation_strength",
            ),
            (
                self.conditional_synergy_strength,
                "conditional_synergy_strength",
            ),
            (
                self.causal_prior_strength,
                "causal_prior_strength",
            ),
        ):
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            type(self.exploration_pool_size) is not int
            or self.exploration_pool_size <= 0
        ):
            raise ValueError("exploration_pool_size must be positive")
        if (
            type(self.randomized_audit_after_directed_steps) is not int
            or self.randomized_audit_after_directed_steps < 0
        ):
            raise ValueError(
                "randomized_audit_after_directed_steps must be "
                "non-negative"
            )
        if (
            type(self.minimum_post_audit_optimization_slots) is not int
            or self.minimum_post_audit_optimization_slots < 0
        ):
            raise ValueError(
                "minimum_post_audit_optimization_slots must be non-negative"
            )
        if (
            type(self.terminal_hierarchical_slots) is not int
            or self.terminal_hierarchical_slots < 0
        ):
            raise ValueError(
                "terminal_hierarchical_slots must be non-negative"
            )
        if (
            type(self.native_rank_strength) is not float
            or not math.isfinite(self.native_rank_strength)
            or self.native_rank_strength < 0.0
        ):
            raise ValueError(
                "native_rank_strength must be finite and non-negative"
            )
        _require_probability(
            self.audit_exploration_probability,
            name="audit_exploration_probability",
        )
        if (
            type(self.trace_alternative_count) is not int
            or self.trace_alternative_count < 0
        ):
            raise ValueError(
                "trace_alternative_count must be non-negative"
            )
        for values, name in (
            (
                self.stratified_audit_coverage_family_ids,
                "stratified_audit_coverage_family_ids",
            ),
            (
                self.stratified_audit_stratum_family_ids,
                "stratified_audit_stratum_family_ids",
            ),
        ):
            if (
                type(values) is not tuple
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must be an exact canonical tuple")
            for value in values:
                _require_token(value, name=name)
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        _require_token(self.policy_id, name="policy_id")
        if self.policy_version not in {
            OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_VERSION,
            OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION,
            OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION,
            OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION,
            OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION,
            OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION,
        }:
            raise ValueError("policy_version is unsupported")
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_VERSION
            and (
                self.positive_redundancy_strength != 0.0
                or self.trace_alternative_count != 0
                or self.conditional_saturation_strength != 0.0
                or self.conditional_synergy_strength != 0.0
                or self.causal_prior_strength != 0.0
                or self.randomized_audit_after_directed_steps != 0
                or self.audit_exploration_probability != 0.0
                or self.stratified_audit_coverage_family_ids
                or self.stratified_audit_stratum_family_ids
                or self.minimum_post_audit_optimization_slots != 0
                or self.terminal_hierarchical_slots != 0
                or self.native_rank_strength != 0.0
            )
        ):
            raise ValueError(
                "set-aware controls require policy version 3"
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION
            and (
                self.trace_alternative_count <= 0
                or self.conditional_saturation_strength != 0.0
                or self.conditional_synergy_strength != 0.0
                or self.causal_prior_strength != 0.0
                or self.randomized_audit_after_directed_steps != 0
                or self.audit_exploration_probability != 0.0
                or self.stratified_audit_coverage_family_ids
                or self.stratified_audit_stratum_family_ids
                or self.minimum_post_audit_optimization_slots != 0
                or self.terminal_hierarchical_slots != 0
                or self.native_rank_strength != 0.0
            )
        ):
            raise ValueError(
                "policy version 3 requires auditable alternatives and "
                "permits only its proxy controls"
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION
            and (
                self.trace_alternative_count <= 0
                or self.positive_redundancy_strength != 0.0
                or self.conditional_saturation_strength <= 0.0
                or self.causal_prior_strength <= 0.0
                or self.randomized_audit_slots != 1
                or self.audit_exploration_probability != 0.0
                or self.stratified_audit_coverage_family_ids
                or self.stratified_audit_stratum_family_ids
                or self.minimum_post_audit_optimization_slots != 0
                or self.terminal_hierarchical_slots != 0
                or self.native_rank_strength != 0.0
            )
        ):
            raise ValueError(
                "policy version 4 requires causal set controls, one "
                "mid-wave audit, and auditable alternatives"
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
            and (
                self.trace_alternative_count <= 0
                or self.positive_redundancy_strength != 0.0
                or self.conditional_saturation_strength <= 0.0
                or self.causal_prior_strength <= 0.0
                or self.randomized_audit_slots != 1
                or self.audit_exploration_probability <= 0.0
                or self.stratified_audit_coverage_family_ids
                or self.stratified_audit_stratum_family_ids
                or self.minimum_post_audit_optimization_slots != 0
                or self.terminal_hierarchical_slots != 0
                or self.native_rank_strength != 0.0
            )
        ):
            raise ValueError(
                "policy version 5 requires causal set controls, one "
                "risk-controlled audit, positive epsilon, and auditable "
                "alternatives"
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
            and (
                self.trace_alternative_count <= 0
                or self.positive_redundancy_strength != 0.0
                or self.conditional_saturation_strength != 0.0
                or self.conditional_synergy_strength != 0.0
                or self.causal_prior_strength != 0.0
                or self.randomized_audit_after_directed_steps != 0
                or self.randomized_audit_slots != 1
                or self.audit_exploration_probability <= 0.0
                or not self.stratified_audit_coverage_family_ids
                or not self.stratified_audit_stratum_family_ids
                or self.minimum_post_audit_optimization_slots != 0
                or self.terminal_hierarchical_slots != 0
                or self.native_rank_strength != 0.0
            )
        ):
            raise ValueError(
                "policy version 6 requires one final risk-controlled audit, "
                "positive epsilon, auditable alternatives, and explicit "
                "coverage/stratum factor families"
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
            and (
                self.trace_alternative_count <= 0
                or self.positive_redundancy_strength != 0.0
                or self.conditional_saturation_strength != 0.0
                or self.conditional_synergy_strength != 0.0
                or self.causal_prior_strength != 0.0
                or self.randomized_audit_slots != 1
                or self.audit_exploration_probability != 0.0
                or self.stratified_audit_coverage_family_ids
                or self.stratified_audit_stratum_family_ids
                or self.minimum_post_audit_optimization_slots <= 0
                or self.terminal_hierarchical_slots <= 0
                or self.native_rank_strength <= 0.0
            )
        ):
            raise ValueError(
                "policy version 7 requires robust hierarchical terminal "
                "allocation, at least one post-audit optimization slot, "
                "positive native-rank strength, one horizon-gated audit, "
                "and auditable alternatives"
            )
        definition = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "diagnostic_slots": self.diagnostic_slots,
            "randomized_audit_slots": self.randomized_audit_slots,
            "reference_gain_scale_hex": (
                self.reference_gain_scale.hex()
            ),
            "reference_gain_evidence_sha256": (
                self.reference_gain_evidence_sha256
            ),
            "prior_strength_hex": self.prior_strength.hex(),
            "ucb_strength_hex": self.ucb_strength.hex(),
            "counterfactual_strength_hex": (
                self.counterfactual_strength.hex()
            ),
            "diversity_strength_hex": (
                self.diversity_strength.hex()
            ),
            "exploration_pool_size": self.exploration_pool_size,
            "random_seed": self.random_seed,
            "diagnostic_design": (
                "authenticated_fixed_constraints_then_one_consensus_"
                "lane_head_then_authority_disagreement_maximin"
            ),
            "continuation": (
                "sequential_coalition_efficiency_scaled_"
                "contextual_ucb_plus_randomized_audit"
            ),
            "unobserved_outcomes_accepted": False,
            "workload_objective_model_provider_prompt_branches": False,
        }
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION
        ):
            definition.update(
                {
                    "positive_redundancy_strength_hex": (
                        self.positive_redundancy_strength.hex()
                    ),
                    "trace_alternative_count": (
                        self.trace_alternative_count
                    ),
                    "continuation": (
                        "sequential_coalition_efficiency_scaled_contextual_"
                        "ucb_minus_positive_consequence_redundancy_plus_"
                        "randomized_audit"
                    ),
                    "set_value_proxy": (
                        "outcome_weighted_portable_action_similarity"
                    ),
                }
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION
        ):
            definition.update(
                {
                    "conditional_saturation_strength_hex": (
                        self.conditional_saturation_strength.hex()
                    ),
                    "conditional_synergy_strength_hex": (
                        self.conditional_synergy_strength.hex()
                    ),
                    "causal_prior_strength_hex": (
                        self.causal_prior_strength.hex()
                    ),
                    "randomized_audit_after_directed_steps": (
                        self.randomized_audit_after_directed_steps
                    ),
                    "trace_alternative_count": (
                        self.trace_alternative_count
                    ),
                    "continuation": (
                        "conditional_opportunity_saturation_ucb_with_"
                        "midwave_randomized_audit"
                    ),
                    "set_value_evidence": (
                        "real_fixed_and_prior_conditioned_archive_lift"
                    ),
                    "proxy_positive_similarity_penalty": False,
                }
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
        ):
            definition.update(
                {
                    "conditional_saturation_strength_hex": (
                        self.conditional_saturation_strength.hex()
                    ),
                    "conditional_synergy_strength_hex": (
                        self.conditional_synergy_strength.hex()
                    ),
                    "causal_prior_strength_hex": (
                        self.causal_prior_strength.hex()
                    ),
                    "randomized_audit_after_directed_steps": (
                        self.randomized_audit_after_directed_steps
                    ),
                    "audit_exploration_probability_hex": (
                        self.audit_exploration_probability.hex()
                    ),
                    "trace_alternative_count": (
                        self.trace_alternative_count
                    ),
                    "diagnostic_and_directed_tie_breaks": (
                        "canonical_action_hash_seed_invariant"
                    ),
                    "continuation": (
                        "conditional_opportunity_saturation_ucb_with_"
                        "epsilon_greedy_risk_controlled_audit"
                    ),
                    "audit_anchor": (
                        "highest_conditional_ucb_action"
                    ),
                    "audit_exploration_pool": (
                        "anchor_plus_portable_diversity_ranked_actions"
                    ),
                    "audit_propensity": (
                        "anchor=one_minus_epsilon_plus_epsilon_over_pool;"
                        "other=epsilon_over_pool"
                    ),
                    "set_value_evidence": (
                        "real_fixed_and_prior_conditioned_archive_lift"
                    ),
                    "proxy_positive_similarity_penalty": False,
                }
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
        ):
            definition.update(
                {
                    "audit_exploration_probability_hex": (
                        self.audit_exploration_probability.hex()
                    ),
                    "trace_alternative_count": (
                        self.trace_alternative_count
                    ),
                    "stratified_audit_coverage_family_ids": list(
                        self.stratified_audit_coverage_family_ids
                    ),
                    "stratified_audit_stratum_family_ids": list(
                        self.stratified_audit_stratum_family_ids
                    ),
                    "continuation": (
                        "set_aware_ucb_with_legacy_audit_anchor_and_"
                        "risk_controlled_factor_stratified_exploration"
                    ),
                    "audit_anchor": (
                        "frozen_version3_diversity_rank_counterfactual_"
                        "seeded_choice"
                    ),
                    "audit_exploration_support": (
                        "maximally_uncovered_configured_factor_cells"
                    ),
                    "audit_exploration_propensity": (
                        "uniform-nonempty-factor-stratum-then-uniform-action"
                    ),
                    "audit_total_propensity": (
                        "legacy-anchor=one-minus-epsilon-plus-exploration-"
                        "mass-if-supported;other=exploration-mass"
                    ),
                    "candidate_factor_cells_outcome_blind": True,
                    "unobserved_outcomes_accepted": False,
                }
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
        ):
            definition.update(
                {
                    "trace_alternative_count": (
                        self.trace_alternative_count
                    ),
                    "randomized_audit_after_directed_steps": (
                        self.randomized_audit_after_directed_steps
                    ),
                    "minimum_post_audit_optimization_slots": (
                        self.minimum_post_audit_optimization_slots
                    ),
                    "terminal_hierarchical_slots": (
                        self.terminal_hierarchical_slots
                    ),
                    "native_rank_strength_hex": (
                        self.native_rank_strength.hex()
                    ),
                    "engine_return": (
                        "bayesian_mean_of_reference_scale_capped_"
                        "marginal_archive_gain"
                    ),
                    "engine_then_action_allocation": True,
                    "within_engine_terminal_index": (
                        "contextual_posterior_gain_plus_calibratable_"
                        "native_rank_prior"
                    ),
                    "continuation": (
                        "contextual_ucb_before_terminal_horizon_then_"
                        "robust_engine_exposure_and_within_engine_"
                        "exploitation"
                    ),
                    "audit_schedule": (
                        "configured_directed_step_only_when_at_least_"
                        "minimum_post_audit_optimization_slots_remain"
                    ),
                    "terminal_information_only_authoritative_purchase": False,
                    "workload_model_provider_prompt_branches": False,
                }
            )
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(_POLICY_DEFINITION_DOMAIN, definition),
        )

    @staticmethod
    def _validate_market(
        actions: tuple[AdaptiveActionDescriptor, ...],
    ) -> dict[str, AdaptiveActionDescriptor]:
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        by_action: dict[str, AdaptiveActionDescriptor] = {}
        lane_counts: dict[str, int] = {}
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            value.__post_init__()
            if value.action_sha256 in by_action:
                raise ValueError("action identities repeat")
            by_action[value.action_sha256] = value
            lane_counts[value.lane_id] = lane_counts.get(value.lane_id, 0) + 1
        if any(
            value.lane_size != lane_counts[value.lane_id]
            for value in actions
        ):
            raise ValueError("declared lane sizes differ from the market")
        if len({value.phenotype_sha256 for value in actions}) != len(actions):
            raise ValueError(
                "adaptive market must contain unique phenotype identities"
            )
        return by_action

    @staticmethod
    def _minimum_distance(
        action: AdaptiveActionDescriptor,
        selected: tuple[AdaptiveActionDescriptor, ...],
    ) -> float:
        if not selected:
            return 1.0
        return min(_action_distance(action, value) for value in selected)

    def design_diagnostic_pilot(
        self,
        *,
        residual_request_sha256: str,
        actions: tuple[AdaptiveActionDescriptor, ...],
        evaluation_slots: int,
        required_action_sha256s: tuple[str, ...] = (),
    ) -> AdaptiveActionRacingDecision:
        """Choose a constrained lane-head pilot, then fill it by maximin."""

        self.__post_init__()
        require_sha256(
            residual_request_sha256,
            "residual_request_sha256",
        )
        by_action = self._validate_market(actions)
        if (
            type(evaluation_slots) is not int
            or not 2 <= evaluation_slots <= len(actions)
        ):
            raise ValueError("evaluation_slots must fit the action market")
        pilot_width = min(
            self.diagnostic_slots,
            evaluation_slots - self.randomized_audit_slots,
        )
        if pilot_width <= 0:
            raise ValueError("diagnostic design leaves no pilot")
        if (
            type(required_action_sha256s) is not tuple
            or required_action_sha256s
            != tuple(sorted(set(required_action_sha256s)))
        ):
            raise ValueError(
                "required_action_sha256s must be an exact canonical tuple"
            )
        if not set(required_action_sha256s).issubset(by_action):
            raise ValueError(
                "a required diagnostic action is outside the sealed market"
            )
        if len(required_action_sha256s) > pilot_width:
            raise ValueError(
                "fixed allocation constraints exceed diagnostic capacity"
            )

        by_lane: dict[str, list[AdaptiveActionDescriptor]] = {}
        for action in actions:
            by_lane.setdefault(action.lane_id, []).append(action)
        lane_heads = tuple(
            sorted(
                (
                    min(
                        values,
                        key=lambda value: (
                            -value.prior_score,
                            value.native_rank,
                            value.action_sha256,
                        ),
                    )
                    for values in by_lane.values()
                ),
                key=lambda value: (
                    -value.prior_score,
                    value.native_rank,
                    value.lane_id,
                    value.action_sha256,
                ),
            )
        )
        selected: list[AdaptiveActionDescriptor] = [
            by_action[value] for value in required_action_sha256s
        ]
        trace: list[dict[str, object]] = [
            {
                "ordinal": ordinal,
                "action_sha256": value.action_sha256,
                "reason": "fixed_allocation_constraint",
                "lane_id": value.lane_id,
                "native_rank": value.native_rank,
            }
            for ordinal, value in enumerate(selected, start=1)
        ]
        for value in lane_heads:
            if len(selected) >= pilot_width:
                break
            if value.action_sha256 in {
                item.action_sha256 for item in selected
            }:
                continue
            selected.append(value)
            trace.append(
                {
                    "ordinal": len(selected),
                    "action_sha256": value.action_sha256,
                    "reason": "lane_head",
                    "lane_id": value.lane_id,
                    "native_rank": value.native_rank,
                }
            )
        while len(selected) < pilot_width:
            candidates = tuple(
                value
                for value in actions
                if value.action_sha256
                not in {item.action_sha256 for item in selected}
            )
            if not candidates:
                raise ValueError("action market cannot fill diagnostic pilot")
            scored = tuple(
                (
                    abs(value.rank_quality - value.prior_score),
                    self._minimum_distance(value, tuple(selected)),
                    value.prior_score,
                    (
                        0.0
                        if self.policy_version
                        == OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
                        else _stable_unit_interval(
                            self.random_seed,
                            residual_request_sha256,
                            "diagnostic",
                            len(selected),
                            value.action_sha256,
                        )
                    ),
                    value,
                )
                for value in candidates
            )
            chosen = max(
                scored,
                key=lambda row: (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4].action_sha256,
                ),
            )[4]
            selected.append(chosen)
            trace.append(
                {
                    "ordinal": len(selected),
                    "action_sha256": chosen.action_sha256,
                    "reason": "authority_disagreement_maximin",
                    "lane_id": chosen.lane_id,
                    "native_rank": chosen.native_rank,
                    "authority_disagreement_hex": abs(
                        chosen.rank_quality - chosen.prior_score
                    ).hex(),
                    "minimum_distance_hex": self._minimum_distance(
                        chosen,
                        tuple(selected[:-1]),
                    ).hex(),
                }
            )
        return AdaptiveActionRacingDecision(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            wave=AdaptiveActionWave.DIAGNOSTIC,
            selected_action_sha256s=tuple(
                sorted(value.action_sha256 for value in selected)
            ),
            prior_selected_action_sha256s=(),
            observed_outcome_sha256s=(),
            observed_set_outcome_sha256s=(),
            selection_propensity=1.0,
            evidence=freeze_json(
                {
                    "selection_trace": trace,
                    "pilot_width": pilot_width,
                    "lane_count": len(by_lane),
                    "lane_head_count": sum(
                        value["reason"] == "lane_head"
                        for value in trace
                    ),
                    "fixed_constraint_count": len(
                        required_action_sha256s
                    ),
                    "required_action_sha256s": list(
                        required_action_sha256s
                    ),
                    "candidate_outcomes_observed": False,
                    "all_actions_materialized_before_decision": True,
                }
            ),
        )

    @staticmethod
    def _outcome_map(
        *,
        by_action: dict[str, AdaptiveActionDescriptor],
        selected_action_sha256s: tuple[str, ...],
        outcomes: tuple[AdaptiveActionOutcome, ...],
    ) -> dict[str, AdaptiveActionOutcome]:
        if (
            type(selected_action_sha256s) is not tuple
            or selected_action_sha256s
            != tuple(sorted(set(selected_action_sha256s)))
        ):
            raise ValueError(
                "selected_action_sha256s must be an exact canonical tuple"
            )
        if not set(selected_action_sha256s).issubset(by_action):
            raise ValueError("selected action is outside the sealed market")
        if (
            type(outcomes) is not tuple
            or any(type(value) is not AdaptiveActionOutcome for value in outcomes)
        ):
            raise TypeError("outcomes must contain exact values")
        result: dict[str, AdaptiveActionOutcome] = {}
        for value in outcomes:
            value.__post_init__()
            if value.action_sha256 in result:
                raise ValueError("observed outcomes repeat an action")
            result[value.action_sha256] = value
        if set(result) != set(selected_action_sha256s):
            raise ValueError(
                "observations must exactly cover all previously selected actions"
            )
        return result

    @staticmethod
    def _validate_set_outcomes(
        *,
        selected_action_sha256s: tuple[str, ...],
        diagnostic_action_sha256s: tuple[str, ...],
        outcome_by_action: dict[str, AdaptiveActionOutcome],
        set_outcomes: tuple[AdaptiveActionSetOutcome, ...],
    ) -> tuple[
        AdaptiveActionSetOutcome | None,
        dict[str, AdaptiveActionSetOutcome],
    ]:
        if type(set_outcomes) is not tuple:
            raise TypeError("set_outcomes must be an exact tuple")
        if not set_outcomes:
            return None, {}
        prior_bindings: list[tuple[str, str]] = []
        prior_augmented_gain = 0.0
        singleton_by_action: dict[str, AdaptiveActionSetOutcome] = {}
        for ordinal, observation in enumerate(set_outcomes):
            if type(observation) is not AdaptiveActionSetOutcome:
                raise TypeError("set_outcomes must contain exact values")
            observation.__post_init__()
            if observation.prior_action_evaluation_bindings != tuple(
                sorted(prior_bindings)
            ):
                raise ValueError(
                    "set outcomes skip their causal evaluation prefix"
                )
            if not _metric_close(
                observation.prior_selected_set_gain,
                prior_augmented_gain,
            ):
                raise ValueError(
                    "set outcomes skip their causal utility prefix"
                )
            current = observation.current_action_evaluation_bindings
            current_actions = {value[0] for value in current}
            if ordinal == 0:
                if current_actions != set(diagnostic_action_sha256s):
                    raise ValueError(
                        "first set outcome must cover the diagnostic wave"
                    )
            elif len(current) != 1:
                raise ValueError(
                    "continuation set outcomes must bind one action"
                )
            for action_sha256, evaluation_sha256 in current:
                outcome = outcome_by_action.get(action_sha256)
                if (
                    outcome is None
                    or outcome.evaluation_sha256 != evaluation_sha256
                ):
                    raise ValueError(
                        "set outcome does not join an observed outcome"
                    )
            if len(current) == 1:
                singleton_by_action[current[0][0]] = observation
            prior_bindings.extend(current)
            prior_augmented_gain = (
                observation.augmented_selected_set_gain
            )
        if {value[0] for value in prior_bindings} != set(
            selected_action_sha256s
        ):
            raise ValueError(
                "set outcomes must exactly cover selected actions"
            )
        return set_outcomes[0], singleton_by_action

    def _causal_set_components(
        self,
        *,
        candidate: AdaptiveActionDescriptor,
        selected_by_action: dict[str, AdaptiveActionDescriptor],
        diagnostic: AdaptiveActionSetOutcome,
        singleton_by_action: dict[str, AdaptiveActionSetOutcome],
        base: dict[str, float],
    ) -> dict[str, float]:
        if not _metric_close(
            diagnostic.current_wave_fixed_set_gain,
            base["diagnostic_joint_gain"],
        ):
            raise ValueError(
                "diagnostic set outcome differs from joint gain"
            )
        diagnostic_sum = base["diagnostic_individual_gain_sum"]
        diagnostic_saturation = (
            0.0
            if diagnostic_sum <= 0.0
            else max(
                0.0,
                1.0 - base["diagnostic_coalition_efficiency"],
            )
        )
        saturation_weighted = (
            self.causal_prior_strength * diagnostic_saturation
        )
        saturation_weight = self.causal_prior_strength
        synergy_weighted = 0.0
        synergy_weight = self.causal_prior_strength
        causal_evidence_count = 0
        for action_sha256, observation in singleton_by_action.items():
            action = selected_by_action[action_sha256]
            similarity = max(
                0.0,
                1.0 - _action_distance(candidate, action),
            )
            weight = 0.10 + 0.90 * similarity
            fixed_gain = observation.current_wave_fixed_set_gain
            if fixed_gain > 0.0:
                saturation_weighted += (
                    weight * observation.saturation_fraction
                )
                saturation_weight += weight
                causal_evidence_count += 1
            if (
                fixed_gain > 0.0
                or observation.prior_conditioned_synergy > 0.0
            ):
                synergy_ratio = min(
                    observation.prior_conditioned_synergy
                    / max(fixed_gain, self.reference_gain_scale),
                    2.0,
                )
                synergy_weighted += weight * synergy_ratio
                synergy_weight += weight
        posterior_saturation = (
            saturation_weighted / saturation_weight
        )
        posterior_synergy = synergy_weighted / synergy_weight
        causal_uncertainty = 1.0 / math.sqrt(saturation_weight)
        conditional_multiplier = max(
            0.0,
            1.0
            - self.conditional_saturation_strength
            * posterior_saturation
            + self.conditional_synergy_strength
            * posterior_synergy,
        )
        predicted_conditional_gain = (
            base["posterior_gain"] * conditional_multiplier
        )
        # Saturation is evidence about expected conditional value, not a
        # license to equate structural distance with archive complementarity.
        # The existing UCB already prices uncertainty and diversity once.
        # Keeping both bonuses at zero avoids the failed V52 proxy penalty's
        # double counting while retaining them as explicit trace fields.
        complement_bonus = 0.0
        causal_uncertainty_bonus = 0.0
        conditional_ucb_index = (
            base["ucb_index"]
            - base["posterior_gain"]
            + predicted_conditional_gain
            + complement_bonus
            + causal_uncertainty_bonus
        )
        return {
            **base,
            "diagnostic_saturation": diagnostic_saturation,
            "posterior_saturation": posterior_saturation,
            "posterior_synergy": posterior_synergy,
            "causal_evidence_count": float(causal_evidence_count),
            "causal_effective_weight": saturation_weight,
            "causal_uncertainty": causal_uncertainty,
            "conditional_multiplier": conditional_multiplier,
            "predicted_conditional_gain": predicted_conditional_gain,
            "complement_bonus": complement_bonus,
            "causal_uncertainty_bonus": causal_uncertainty_bonus,
            "conditional_ucb_index": conditional_ucb_index,
            "ucb_index": conditional_ucb_index,
        }

    def _ucb_components(
        self,
        *,
        candidate: AdaptiveActionDescriptor,
        selected: tuple[AdaptiveActionDescriptor, ...],
        outcome_by_action: dict[str, AdaptiveActionOutcome],
        diagnostic_action_sha256s: tuple[str, ...],
        diagnostic_joint_gain: float,
    ) -> dict[str, float]:
        weighted_gain = 0.0
        weighted_positive = 0.0
        total_weight = 0.0
        same_lane_ranks: list[float] = []
        for action in selected:
            outcome = outcome_by_action[action.action_sha256]
            rank_distance = abs(
                candidate.rank_quality - action.rank_quality
            )
            weight = 0.10
            if candidate.lane_id == action.lane_id:
                weight += 0.55 * math.exp(-rank_distance / 0.35)
                same_lane_ranks.append(action.rank_quality)
            if candidate.operator_id == action.operator_id:
                weight += 0.15
            if (
                candidate.parent_generated_in_current_run
                == action.parent_generated_in_current_run
            ):
                weight += 0.10
            weight += 0.10 * (
                1.0
                - _semantic_distance(
                    candidate.semantic_cell_ids,
                    action.semantic_cell_ids,
                )
            )
            weighted_gain += weight * outcome.marginal_archive_gain
            weighted_positive += weight * float(outcome.positive)
            total_weight += weight

        prior_positive = 0.25 + 0.50 * candidate.prior_score
        prior_gain = self.reference_gain_scale * (
            0.25 + 0.75 * candidate.prior_score
        )
        active_gain_scale = max(
            self.reference_gain_scale,
            max(
                (
                    value.marginal_archive_gain
                    for value in outcome_by_action.values()
                ),
                default=0.0,
            ),
        )
        positive_redundancy = 0.0
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION
        ):
            positive_redundancy = max(
                (
                    max(0.0, 1.0 - _action_distance(candidate, action))
                    * min(
                        1.0,
                        outcome_by_action[
                            action.action_sha256
                        ].marginal_archive_gain
                        / active_gain_scale,
                    )
                    for action in selected
                    if outcome_by_action[
                        action.action_sha256
                    ].positive
                ),
                default=0.0,
            )
        redundancy_penalty = (
            self.positive_redundancy_strength
            * active_gain_scale
            * positive_redundancy
        )
        diagnostic_individual_gain_sum = math.fsum(
            outcome_by_action[value].marginal_archive_gain
            for value in diagnostic_action_sha256s
        )
        diagnostic_coalition_efficiency = (
            0.0
            if diagnostic_individual_gain_sum <= 0.0
            else min(
                1.0,
                diagnostic_joint_gain / diagnostic_individual_gain_sum,
            )
        )
        exploration_gain_scale = (
            self.reference_gain_scale
            + (active_gain_scale - self.reference_gain_scale)
            * (1.0 - diagnostic_coalition_efficiency)
        )
        denominator = self.prior_strength + total_weight
        posterior_positive = (
            self.prior_strength * prior_positive + weighted_positive
        ) / denominator
        posterior_gain = (
            self.prior_strength * prior_gain + weighted_gain
        ) / denominator
        uncertainty = exploration_gain_scale / math.sqrt(denominator)
        if same_lane_ranks:
            rank_counterfactual = min(
                abs(candidate.rank_quality - value)
                for value in same_lane_ranks
            )
        else:
            rank_counterfactual = 1.0
        diversity = self._minimum_distance(candidate, selected)
        ucb_index = (
            posterior_gain
            + self.ucb_strength * uncertainty
            + self.counterfactual_strength
            * exploration_gain_scale
            * rank_counterfactual
            + self.diversity_strength
            * exploration_gain_scale
            * diversity
            - redundancy_penalty
        )
        result = {
            "posterior_positive": posterior_positive,
            "posterior_gain": posterior_gain,
            "uncertainty": uncertainty,
            "rank_counterfactual": rank_counterfactual,
            "diversity": diversity,
            "ucb_index": ucb_index,
            "effective_observation_weight": total_weight,
            "active_gain_scale": active_gain_scale,
            "diagnostic_joint_gain": diagnostic_joint_gain,
            "diagnostic_individual_gain_sum": (
                diagnostic_individual_gain_sum
            ),
            "diagnostic_coalition_efficiency": (
                diagnostic_coalition_efficiency
            ),
            "exploration_gain_scale": exploration_gain_scale,
        }
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION
        ):
            result.update(
                {
                    "positive_redundancy": positive_redundancy,
                    "redundancy_penalty": redundancy_penalty,
                }
            )
        return result

    def _terminal_hierarchical_ranking(
        self,
        *,
        actions: tuple[AdaptiveActionDescriptor, ...],
        remaining: tuple[AdaptiveActionDescriptor, ...],
        selected: tuple[AdaptiveActionDescriptor, ...],
        outcome_by_action: dict[str, AdaptiveActionOutcome],
        components: dict[str, dict[str, float]],
        seats_left: int,
    ) -> tuple[
        tuple[AdaptiveActionDescriptor, ...],
        dict[str, dict[str, float]],
        tuple[str, ...],
    ]:
        """Rank engines first, then actions, at a short residual horizon.

        Engine return is capped at the authenticated reference-gain scale so
        one unusually large action cannot monopolize a sparse engine posterior.
        This is an allocation statistic only; authoritative archive utility
        remains uncapped. Within the chosen engine, native rank is retained as
        a separately weighted cold-start prior rather than mixed into another
        engine's native units.
        """

        if (
            self.policy_version
            != OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
        ):
            raise ValueError(
                "terminal hierarchical ranking requires policy version 7"
            )
        if not remaining or seats_left <= 0:
            raise ValueError(
                "terminal hierarchical ranking requires remaining capacity"
            )
        market_by_lane: dict[str, list[AdaptiveActionDescriptor]] = {}
        remaining_by_lane: dict[str, list[AdaptiveActionDescriptor]] = {}
        selected_by_lane: dict[str, list[AdaptiveActionDescriptor]] = {}
        for action in actions:
            market_by_lane.setdefault(action.lane_id, []).append(action)
        for action in remaining:
            remaining_by_lane.setdefault(action.lane_id, []).append(action)
        for action in selected:
            selected_by_lane.setdefault(action.lane_id, []).append(action)

        future_optimization_slots = max(0, seats_left - 1)
        horizon_exploration_fraction = min(
            1.0,
            future_optimization_slots
            / float(max(1, self.terminal_hierarchical_slots)),
        )
        engine: dict[str, dict[str, float]] = {}
        for lane_id in sorted(remaining_by_lane):
            lane_market = market_by_lane[lane_id]
            lane_selected = selected_by_lane.get(lane_id, [])
            mean_prior_score = math.fsum(
                value.prior_score for value in lane_market
            ) / len(lane_market)
            prior_positive = 0.25 + 0.50 * mean_prior_score
            prior_capped_gain = self.reference_gain_scale * (
                0.25 + 0.75 * mean_prior_score
            )
            observed = tuple(
                outcome_by_action[value.action_sha256]
                for value in lane_selected
            )
            denominator = self.prior_strength + len(observed)
            posterior_positive = (
                self.prior_strength * prior_positive
                + math.fsum(float(value.positive) for value in observed)
            ) / denominator
            posterior_capped_gain = (
                self.prior_strength * prior_capped_gain
                + math.fsum(
                    min(
                        self.reference_gain_scale,
                        value.marginal_archive_gain,
                    )
                    for value in observed
                )
            ) / denominator
            uncertainty = self.reference_gain_scale / math.sqrt(denominator)
            selection_index = (
                posterior_capped_gain
                + horizon_exploration_fraction
                * self.ucb_strength
                * uncertainty
            )
            engine[lane_id] = {
                "engine_mean_prior_score": mean_prior_score,
                "engine_prior_positive": prior_positive,
                "engine_prior_capped_gain": prior_capped_gain,
                "engine_observed_count": float(len(observed)),
                "engine_positive_count": float(
                    sum(value.positive for value in observed)
                ),
                "engine_posterior_positive": posterior_positive,
                "engine_posterior_capped_gain": posterior_capped_gain,
                "engine_uncertainty": uncertainty,
                "engine_selection_index": selection_index,
                "future_optimization_slots_after_decision": float(
                    future_optimization_slots
                ),
                "horizon_exploration_fraction": (
                    horizon_exploration_fraction
                ),
            }

        engine_order = tuple(
            sorted(
                engine,
                key=lambda lane_id: (
                    engine[lane_id]["engine_selection_index"],
                    engine[lane_id]["engine_posterior_positive"],
                    engine[lane_id]["engine_mean_prior_score"],
                    lane_id,
                ),
                reverse=True,
            )
        )
        enriched = {
            action_sha256: dict(value)
            for action_sha256, value in components.items()
        }
        ranked: list[AdaptiveActionDescriptor] = []
        for lane_id in engine_order:
            lane_actions = remaining_by_lane[lane_id]
            for action in lane_actions:
                row = enriched[action.action_sha256]
                native_rank_bonus = (
                    self.native_rank_strength
                    * self.reference_gain_scale
                    * action.rank_quality
                )
                row.update(engine[lane_id])
                row["native_rank_bonus"] = native_rank_bonus
                row["within_engine_terminal_index"] = (
                    row["posterior_gain"] + native_rank_bonus
                )
            ranked.extend(
                sorted(
                    lane_actions,
                    key=lambda value: (
                        enriched[value.action_sha256][
                            "within_engine_terminal_index"
                        ],
                        enriched[value.action_sha256][
                            "posterior_positive"
                        ],
                        value.prior_score,
                        value.action_sha256,
                    ),
                    reverse=True,
                )
            )
        return tuple(ranked), enriched, engine_order

    def select_next(
        self,
        *,
        residual_request_sha256: str,
        actions: tuple[AdaptiveActionDescriptor, ...],
        evaluation_slots: int,
        diagnostic_action_sha256s: tuple[str, ...],
        diagnostic_joint_gain: float,
        selected_action_sha256s: tuple[str, ...],
        outcomes: tuple[AdaptiveActionOutcome, ...],
        set_outcomes: tuple[AdaptiveActionSetOutcome, ...] = (),
        excluded_action_sha256s: tuple[str, ...] = (),
    ) -> AdaptiveActionRacingDecision:
        """Select one next action using only outcomes of already selected actions."""

        self.__post_init__()
        require_sha256(
            residual_request_sha256,
            "residual_request_sha256",
        )
        by_action = self._validate_market(actions)
        if (
            type(excluded_action_sha256s) is not tuple
            or excluded_action_sha256s
            != tuple(sorted(set(excluded_action_sha256s)))
        ):
            raise ValueError(
                "excluded action hashes must be unique and canonical"
            )
        for value in excluded_action_sha256s:
            require_sha256(value, "excluded_action_sha256")
        if (
            not set(excluded_action_sha256s) <= set(by_action)
            or set(excluded_action_sha256s)
            & set(selected_action_sha256s)
        ):
            raise ValueError(
                "excluded actions must be unselected market members"
            )
        if (
            type(evaluation_slots) is not int
            or not 2 <= evaluation_slots <= len(actions)
        ):
            raise ValueError("evaluation_slots must fit the action market")
        if len(selected_action_sha256s) >= evaluation_slots:
            raise ValueError("the evaluation slate is already complete")
        outcome_by_action = self._outcome_map(
            by_action=by_action,
            selected_action_sha256s=selected_action_sha256s,
            outcomes=outcomes,
        )
        if (
            type(diagnostic_action_sha256s) is not tuple
            or not diagnostic_action_sha256s
            or diagnostic_action_sha256s
            != tuple(sorted(set(diagnostic_action_sha256s)))
            or not set(diagnostic_action_sha256s).issubset(
                selected_action_sha256s
            )
        ):
            raise ValueError(
                "diagnostic actions must be a canonical selected subset"
            )
        if (
            type(diagnostic_joint_gain) is not float
            or not math.isfinite(diagnostic_joint_gain)
            or diagnostic_joint_gain < 0.0
        ):
            raise ValueError(
                "diagnostic_joint_gain must be finite and non-negative"
            )
        (
            diagnostic_set_outcome,
            singleton_set_outcomes,
        ) = self._validate_set_outcomes(
            selected_action_sha256s=selected_action_sha256s,
            diagnostic_action_sha256s=diagnostic_action_sha256s,
            outcome_by_action=outcome_by_action,
            set_outcomes=set_outcomes,
        )
        if (
            self.policy_version in _CAUSAL_SET_POLICY_VERSIONS
            and diagnostic_set_outcome is None
        ):
            raise ValueError(
                "causal policy versions require causal set outcomes"
            )
        selected = tuple(
            by_action[value] for value in selected_action_sha256s
        )
        selected_phenotypes = {
            value.phenotype_sha256 for value in selected
        }
        remaining = tuple(
            value
            for value in actions
            if value.action_sha256 not in outcome_by_action
            and value.action_sha256 not in set(excluded_action_sha256s)
            and value.phenotype_sha256 not in selected_phenotypes
        )
        if not remaining:
            raise ValueError("no unevaluated action can fill the slate")
        seats_left = evaluation_slots - len(selected)

        components: dict[str, dict[str, float]] = {
            value.action_sha256: self._ucb_components(
                candidate=value,
                selected=selected,
                outcome_by_action=outcome_by_action,
                diagnostic_action_sha256s=(
                    diagnostic_action_sha256s
                ),
                diagnostic_joint_gain=diagnostic_joint_gain,
            )
            for value in remaining
        }
        if (
            self.policy_version in _CAUSAL_SET_POLICY_VERSIONS
        ):
            if diagnostic_set_outcome is None:  # pragma: no cover
                raise AssertionError("validated causal observation vanished")
            selected_by_action = {
                value.action_sha256: value for value in selected
            }
            components = {
                value.action_sha256: self._causal_set_components(
                    candidate=value,
                    selected_by_action=selected_by_action,
                    diagnostic=diagnostic_set_outcome,
                    singleton_by_action=singleton_set_outcomes,
                    base=components[value.action_sha256],
                )
                for value in remaining
            }
        directed_steps_completed = (
            len(selected) - len(diagnostic_action_sha256s)
        )
        terminal_hierarchical = (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
            and seats_left <= self.terminal_hierarchical_slots
        )
        horizon_audit_due: bool | None = None
        horizon_audit_blocked_reason: str | None = None
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
        ):
            horizon_audit_due = (
                directed_steps_completed
                == self.randomized_audit_after_directed_steps
            )
            post_audit_optimization_slots = max(0, seats_left - 1)
            audit = bool(
                horizon_audit_due
                and not terminal_hierarchical
                and post_audit_optimization_slots
                >= self.minimum_post_audit_optimization_slots
            )
            if horizon_audit_due and not audit:
                horizon_audit_blocked_reason = (
                    "terminal_hierarchical_exploitation"
                    if terminal_hierarchical
                    else "insufficient_post_audit_optimization_horizon"
                )
        elif self.policy_version in _CAUSAL_SET_POLICY_VERSIONS:
            audit = (
                directed_steps_completed
                == self.randomized_audit_after_directed_steps
            )
        else:
            audit = seats_left <= self.randomized_audit_slots
        directed_ranked = tuple(
            sorted(
                remaining,
                key=lambda value: (
                    components[value.action_sha256]["ucb_index"],
                    components[value.action_sha256][
                        "posterior_positive"
                    ],
                    value.prior_score,
                    (
                        0.0
                        if self.policy_version
                        == OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
                        else _stable_unit_interval(
                            self.random_seed,
                            residual_request_sha256,
                            "adaptive_tie",
                            len(selected),
                            value.action_sha256,
                        )
                    ),
                    value.action_sha256,
                ),
                reverse=True,
            )
        )
        hierarchical_engine_order: tuple[str, ...] = ()
        if terminal_hierarchical:
            (
                directed_ranked,
                components,
                hierarchical_engine_order,
            ) = self._terminal_hierarchical_ranking(
                actions=actions,
                remaining=remaining,
                selected=selected,
                outcome_by_action=outcome_by_action,
                components=components,
                seats_left=seats_left,
            )
        audit_anchor_sha256: str | None = None
        audit_exploration_branch: bool | None = None
        audit_branch_draw: float | None = None
        audit_choice_draw: float | None = None
        audit_stratum_draw: float | None = None
        audit_stratum_key: tuple[str, ...] | None = None
        audit_strata_record: list[dict[str, object]] | None = None
        audit_max_uncovered_factor_count: int | None = None
        legacy_audit_pool_ids: list[str] | None = None
        if audit:
            ranked = tuple(
                sorted(
                    remaining,
                    key=lambda value: (
                        -components[value.action_sha256]["diversity"],
                        -components[value.action_sha256][
                            "rank_counterfactual"
                        ],
                        -value.prior_score,
                        value.action_sha256,
                    ),
                )
            )
            if (
                self.policy_version
                == OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
            ):
                anchor = directed_ranked[0]
                pool_values = [anchor]
                pool_values.extend(
                    value
                    for value in ranked
                    if value.action_sha256 != anchor.action_sha256
                )
                pool = tuple(
                    pool_values[
                        : min(
                            self.exploration_pool_size,
                            len(pool_values),
                        )
                    ]
                )
                pool_ids = [value.action_sha256 for value in pool]
                audit_anchor_sha256 = anchor.action_sha256
                audit_branch_draw = _stable_unit_interval(
                    self.random_seed,
                    residual_request_sha256,
                    "risk_controlled_audit_branch",
                    len(selected),
                    pool_ids,
                )
                audit_exploration_branch = (
                    audit_branch_draw
                    < self.audit_exploration_probability
                )
                if audit_exploration_branch:
                    audit_choice_draw = _stable_unit_interval(
                        self.random_seed,
                        residual_request_sha256,
                        "risk_controlled_audit_choice",
                        len(selected),
                        pool_ids,
                    )
                    chosen = pool[
                        min(
                            int(audit_choice_draw * len(pool)),
                            len(pool) - 1,
                        )
                    ]
                else:
                    chosen = anchor
                if chosen.action_sha256 == anchor.action_sha256:
                    propensity = (
                        1.0
                        - self.audit_exploration_probability
                        + self.audit_exploration_probability / len(pool)
                    )
                else:
                    propensity = (
                        self.audit_exploration_probability / len(pool)
                    )
                ranked_candidates = directed_ranked
                ranking_basis = (
                    "conditional_ucb_anchor_plus_epsilon_greedy_"
                    "portable_diversity_audit"
                )
            elif (
                self.policy_version
                == OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
            ):
                legacy_pool = ranked[
                    : min(self.exploration_pool_size, len(ranked))
                ]
                legacy_audit_pool_ids = [
                    value.action_sha256 for value in legacy_pool
                ]
                legacy_rng = random.Random(
                    int(
                        hashlib.sha256(
                            _canonical_json(
                                [
                                    self.random_seed,
                                    residual_request_sha256,
                                    "randomized_audit",
                                    len(selected),
                                    legacy_audit_pool_ids,
                                ]
                            )
                        ).hexdigest(),
                        16,
                    )
                )
                anchor = legacy_pool[
                    legacy_rng.randrange(len(legacy_pool))
                ]
                audit_anchor_sha256 = anchor.action_sha256
                required_families = set(
                    self.stratified_audit_coverage_family_ids
                ) | set(self.stratified_audit_stratum_family_ids)
                factor_by_action: dict[str, dict[str, str]] = {}
                for action in (*selected, *remaining):
                    factors = {
                        value.family_id: value.level_id
                        for value in action.factor_cells
                    }
                    missing = required_families - set(factors)
                    if missing:
                        raise ValueError(
                            "stratified audit action lacks factor families: "
                            + ",".join(sorted(missing))
                        )
                    factor_by_action[action.action_sha256] = factors
                selected_levels = {
                    family_id: {
                        factor_by_action[action.action_sha256][family_id]
                        for action in selected
                    }
                    for family_id in (
                        self.stratified_audit_coverage_family_ids
                    )
                }
                uncovered_count = {
                    action.action_sha256: sum(
                        factor_by_action[action.action_sha256][family_id]
                        not in selected_levels[family_id]
                        for family_id in (
                            self.stratified_audit_coverage_family_ids
                        )
                    )
                    for action in remaining
                }
                audit_max_uncovered_factor_count = max(
                    uncovered_count.values()
                )
                exploration_support = tuple(
                    action
                    for action in remaining
                    if uncovered_count[action.action_sha256]
                    == audit_max_uncovered_factor_count
                )
                strata: dict[
                    tuple[str, ...],
                    list[AdaptiveActionDescriptor],
                ] = {}
                for action in exploration_support:
                    key = tuple(
                        factor_by_action[action.action_sha256][family_id]
                        for family_id in (
                            self.stratified_audit_stratum_family_ids
                        )
                    )
                    strata.setdefault(key, []).append(action)
                canonical_strata = tuple(
                    (
                        key,
                        tuple(
                            sorted(
                                values,
                                key=lambda value: value.action_sha256,
                            )
                        ),
                    )
                    for key, values in sorted(strata.items())
                )
                audit_strata_record = [
                    {
                        "stratum_key": list(key),
                        "action_sha256s": [
                            value.action_sha256 for value in values
                        ],
                        "conditional_action_propensity_hex": (
                            (1.0 / len(values)).hex()
                        ),
                    }
                    for key, values in canonical_strata
                ]
                pool_ids = sorted(
                    value.action_sha256
                    for value in exploration_support
                )
                audit_branch_draw = _stable_unit_interval(
                    self.random_seed,
                    residual_request_sha256,
                    "stratified_audit_branch",
                    len(selected),
                    legacy_audit_pool_ids,
                    pool_ids,
                )
                audit_exploration_branch = (
                    audit_branch_draw
                    < self.audit_exploration_probability
                )
                if audit_exploration_branch:
                    audit_stratum_draw = _stable_unit_interval(
                        self.random_seed,
                        residual_request_sha256,
                        "stratified_audit_stratum",
                        len(selected),
                        [value[0] for value in canonical_strata],
                    )
                    stratum_index = min(
                        int(audit_stratum_draw * len(canonical_strata)),
                        len(canonical_strata) - 1,
                    )
                    (
                        audit_stratum_key,
                        stratum_actions,
                    ) = canonical_strata[stratum_index]
                    audit_choice_draw = _stable_unit_interval(
                        self.random_seed,
                        residual_request_sha256,
                        "stratified_audit_action",
                        len(selected),
                        audit_stratum_key,
                        [
                            value.action_sha256
                            for value in stratum_actions
                        ],
                    )
                    action_index = min(
                        int(audit_choice_draw * len(stratum_actions)),
                        len(stratum_actions) - 1,
                    )
                    chosen = stratum_actions[action_index]
                else:
                    chosen = anchor
                    chosen_factors = factor_by_action[
                        chosen.action_sha256
                    ]
                    audit_stratum_key = tuple(
                        chosen_factors[family_id]
                        for family_id in (
                            self.stratified_audit_stratum_family_ids
                        )
                    )
                chosen_stratum = next(
                    (
                        values
                        for key, values in canonical_strata
                        if key == audit_stratum_key
                    ),
                    (),
                )
                exploration_propensity = (
                    0.0
                    if not chosen_stratum
                    or chosen.action_sha256
                    not in {
                        value.action_sha256
                        for value in chosen_stratum
                    }
                    else 1.0
                    / len(canonical_strata)
                    / len(chosen_stratum)
                )
                propensity = (
                    (
                        1.0 - self.audit_exploration_probability
                        if chosen.action_sha256 == anchor.action_sha256
                        else 0.0
                    )
                    + self.audit_exploration_probability
                    * exploration_propensity
                )
                ranked_candidates = ranked
                ranking_basis = (
                    "legacy_v3_audit_anchor_plus_risk_controlled_"
                    "factor_stratified_exploration"
                )
            else:
                pool = ranked[
                    : min(self.exploration_pool_size, len(ranked))
                ]
                rng = random.Random(
                    int(
                        hashlib.sha256(
                            _canonical_json(
                                [
                                    self.random_seed,
                                    residual_request_sha256,
                                    "randomized_audit",
                                    len(selected),
                                    [
                                        value.action_sha256
                                        for value in pool
                                    ],
                                ]
                            )
                        ).hexdigest(),
                        16,
                    )
                )
                chosen = pool[rng.randrange(len(pool))]
                propensity = 1.0 / len(pool)
                pool_ids = [value.action_sha256 for value in pool]
                ranked_candidates = ranked
                ranking_basis = (
                    "diversity_then_rank_counterfactual_then_prior"
                )
            wave = AdaptiveActionWave.RANDOMIZED_AUDIT
        else:
            chosen = directed_ranked[0]
            propensity = 1.0
            wave = AdaptiveActionWave.ADAPTIVE
            pool_ids = []
            ranked_candidates = directed_ranked
            ranking_basis = (
                (
                    "conditional_opportunity_saturation_ucb_then_"
                    "positive_then_prior"
                    if self.policy_version in _CAUSAL_SET_POLICY_VERSIONS
                    else (
                        "robust_engine_exposure_then_within_engine_"
                        "posterior_plus_native_rank"
                        if terminal_hierarchical
                        else "set_aware_ucb_then_positive_then_prior"
                    )
                )
            )
        chosen_components = components[chosen.action_sha256]
        evidence: dict[str, object] = {
            "selected_action": chosen.to_record(),
            "selected_components": {
                name: value.hex()
                for name, value in chosen_components.items()
            },
            "randomized_exploration_pool_action_sha256s": pool_ids,
            "candidate_score_count": len(components),
            "observed_action_count": len(outcomes),
            "diagnostic_action_sha256s": list(
                diagnostic_action_sha256s
            ),
            "diagnostic_joint_gain_hex": (
                diagnostic_joint_gain.hex()
            ),
            "observed_set_outcome_sha256s": [
                value.set_outcome_sha256 for value in set_outcomes
            ],
            "causal_set_outcomes_used_for_selection": (
                self.policy_version in _CAUSAL_SET_POLICY_VERSIONS
            ),
            "directed_steps_completed_before_decision": (
                directed_steps_completed
            ),
            "unobserved_candidate_outcomes_available": False,
            "all_actions_materialized_before_current_outcomes": True,
        }
        if excluded_action_sha256s:
            evidence.update(
                {
                    "excluded_action_sha256s": list(
                        excluded_action_sha256s
                    ),
                    "exclusion_role": (
                        "same_prefix_counterfactual_quarantine"
                    ),
                    "excluded_action_outcomes_used_for_selection": False,
                }
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION
        ):
            evidence.update(
                {
                    "risk_controlled_audit": audit,
                    "audit_exploration_probability_hex": (
                        self.audit_exploration_probability.hex()
                    ),
                    "audit_anchor_action_sha256": (
                        audit_anchor_sha256
                    ),
                    "audit_exploration_branch": (
                        audit_exploration_branch
                    ),
                    "audit_branch_draw_hex": (
                        None
                        if audit_branch_draw is None
                        else audit_branch_draw.hex()
                    ),
                    "audit_choice_draw_hex": (
                        None
                        if audit_choice_draw is None
                        else audit_choice_draw.hex()
                    ),
                    "audit_selected_propensity_hex": (
                        float(propensity).hex()
                    ),
                    "seed_affects_only_audit_draw": True,
                }
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
        ):
            evidence.update(
                {
                    "risk_controlled_stratified_audit": audit,
                    "audit_exploration_probability_hex": (
                        self.audit_exploration_probability.hex()
                    ),
                    "legacy_audit_anchor_action_sha256": (
                        audit_anchor_sha256
                    ),
                    "legacy_audit_pool_action_sha256s": (
                        legacy_audit_pool_ids
                    ),
                    "audit_exploration_branch": (
                        audit_exploration_branch
                    ),
                    "audit_branch_draw_hex": (
                        None
                        if audit_branch_draw is None
                        else audit_branch_draw.hex()
                    ),
                    "audit_stratum_draw_hex": (
                        None
                        if audit_stratum_draw is None
                        else audit_stratum_draw.hex()
                    ),
                    "audit_choice_draw_hex": (
                        None
                        if audit_choice_draw is None
                        else audit_choice_draw.hex()
                    ),
                    "audit_selected_stratum_key": (
                        None
                        if audit_stratum_key is None
                        else list(audit_stratum_key)
                    ),
                    "audit_strata": audit_strata_record,
                    "audit_max_uncovered_factor_count": (
                        audit_max_uncovered_factor_count
                    ),
                    "audit_coverage_family_ids": list(
                        self.stratified_audit_coverage_family_ids
                    ),
                    "audit_stratum_family_ids": list(
                        self.stratified_audit_stratum_family_ids
                    ),
                    "audit_selected_propensity_hex": (
                        float(propensity).hex()
                    ),
                    "legacy_anchor_probability_hex": (
                        (1.0 - self.audit_exploration_probability).hex()
                    ),
                    "candidate_factor_cells_outcome_blind": True,
                    "seed_affects_legacy_anchor_and_audit_draws": True,
                }
            )
        if (
            self.policy_version
            == OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
        ):
            evidence.update(
                {
                    "horizon_aware_authoritative_measurement": True,
                    "seats_left_before_decision": seats_left,
                    "future_optimization_slots_after_decision": max(
                        0,
                        seats_left - 1,
                    ),
                    "authoritative_audit_due": horizon_audit_due,
                    "authoritative_audit_selected": audit,
                    "authoritative_audit_blocked_reason": (
                        horizon_audit_blocked_reason
                    ),
                    "minimum_post_audit_optimization_slots": (
                        self.minimum_post_audit_optimization_slots
                    ),
                    "terminal_hierarchical_slots": (
                        self.terminal_hierarchical_slots
                    ),
                    "terminal_hierarchical_allocation": (
                        terminal_hierarchical
                    ),
                    "hierarchical_engine_order": list(
                        hierarchical_engine_order
                    ),
                    "selected_engine_id": chosen.lane_id,
                    "native_rank_strength_hex": (
                        self.native_rank_strength.hex()
                    ),
                    "engine_returns_capped_only_for_allocation": True,
                    "authoritative_archive_utility_capped": False,
                    "workload_model_provider_prompt_branches": False,
                }
            )
        if (
            self.policy_version
            in {
                OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION,
                OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION,
                OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION,
                OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION,
                OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION,
            }
        ):
            evidence.update(
                {
                    "candidate_score_cards": [
                        {
                            "rank": rank,
                            "selected": (
                                value.action_sha256
                                == chosen.action_sha256
                            ),
                            "randomized_pool_member": (
                                value.action_sha256 in set(pool_ids)
                            ),
                            "action": value.to_record(),
                            "components": {
                                name: component.hex()
                                for name, component in components[
                                    value.action_sha256
                                ].items()
                            },
                        }
                        for rank, value in enumerate(
                            ranked_candidates[
                                : self.trace_alternative_count
                            ],
                            start=1,
                        )
                    ],
                    "candidate_score_card_count": min(
                        self.trace_alternative_count,
                        len(ranked_candidates),
                    ),
                    "candidate_score_ranking_basis": ranking_basis,
                    "chosen_action_present_in_score_cards": (
                        chosen.action_sha256
                        in {
                            value.action_sha256
                            for value in ranked_candidates[
                                : self.trace_alternative_count
                            ]
                        }
                    ),
                }
            )
        return AdaptiveActionRacingDecision(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            wave=wave,
            selected_action_sha256s=(chosen.action_sha256,),
            prior_selected_action_sha256s=tuple(
                sorted(selected_action_sha256s)
            ),
            observed_outcome_sha256s=tuple(
                sorted(value.outcome_sha256 for value in outcomes)
            ),
            observed_set_outcome_sha256s=tuple(
                sorted(
                    value.set_outcome_sha256
                    for value in set_outcomes
                )
            ),
            selection_propensity=float(propensity),
            evidence=freeze_json(evidence),
        )


__all__ = [
    "AdaptiveActionAllocationDirective",
    "AdaptiveActionDescriptor",
    "AdaptiveActionFactorCell",
    "AdaptiveActionOutcome",
    "AdaptiveActionSetOutcome",
    "AdaptiveActionRacingDecision",
    "AdaptiveActionWave",
    "OUTCOME_ADAPTIVE_ACTION_RACING_CAUSAL_SET_POLICY_VERSION",
    "OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION",
    "OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_ID",
    "OUTCOME_ADAPTIVE_ACTION_RACING_POLICY_VERSION",
    "OUTCOME_ADAPTIVE_ACTION_RACING_RISK_CONTROLLED_POLICY_VERSION",
    "OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION",
    "OUTCOME_ADAPTIVE_ACTION_RACING_SET_AWARE_POLICY_VERSION",
    "OutcomeAdaptiveActionRacingPolicy",
]

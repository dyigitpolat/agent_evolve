"""Authenticated phase-aware credit and allocation for generic search sources.

The controller operates only on workload-neutral, normalized observations.  It
does not inspect candidate fields, objective names, model prose, or provider
metadata.  Sources (for example model, restart, global coverage, or memory) and
operators (for example atomic or recombination families) share the same typed
posterior machinery.

The controller is intentionally deterministic.  It uses the empirical mean of
bounded normalized archive return and one distribution-free uncertainty slot
instead of stochastic Thompson sampling, which makes every decision exactly
replayable while preserving the scale of small but real archive improvements.
Feasibility, persistence, descendant yield, and allocation realizability remain
separate diagnostic channels, but are not added to return with arbitrary
weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import product

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.contextual_search_allocation import (
    ContextualArmCountCapability,
    ContextualLaneJointCountCapability,
    ContextualPortfolioAllocationContract,
    ContextualPortfolioAllocationRealization,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
CONTEXTUAL_SEARCH_CONTROLLER_ID = "phase_aware_contextual_source_operator"
CONTEXTUAL_SEARCH_CONTROLLER_VERSION = 9
CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:phase-aware-contextual-source-operator:v9;"
    b"evidence=authenticated-prior-only-normalized-outcomes;"
    b"selection-currency=normalized-archive-return;"
    b"diagnostics=positive-marginal,utility-share,feasibility,"
    b"stage-front-survival,final-persistence,descendant,allocation-realizability;"
    b"delayed-credit=append-only-source-joined-prior-cutoff;"
    b"allocation-realization=objective-workload-blind-request-overlap;"
    b"return-estimator=bounded-empirical-mean;"
    b"uncertainty=distribution-free-maximum-standard-error;deterministic=true;"
    b"phases=basin-acquisition,basin-expansion,composition,terminal-conversion;"
    b"terminal-information-bonus=zero;one-nonterminal-exploration-slot=true;"
    b"allocation=posterior-mean-proportional-plus-one-uncertainty-slot;"
    b"cold-start=prior-proportional-plus-one-exploration;"
    b"source-axis=sealed-finite-variation-source-not-reconciliation-origin;"
    b"source-incumbent-prior-mass=0.50;operator-incumbent-prior-mass=0.75;"
    b"empirical-capability=prior-realized-stage-count-witnesses;"
    b"capability-projection=minimum-l1-then-maximum-posterior-score;"
    b"capability-is-witness-not-current-guarantee=true;"
    b"prospective-joint-capability=finite-contract-lane-product;"
    b"joint-projection=maximum-feasible-exploration-retention-then-minimum-"
    b"source-operator-l1-then-posterior-score;"
    b"joint-exploration-recourse=deterministic-realizable-challenger;"
    b"joint-slicing=exact-witnessed-lane-vector;"
    b"durable-evidence=expanded-authenticated-query-and-prior-snapshot;"
    b"workload-model-provider-identifiers=false"
).hexdigest()
_OBSERVATION_DOMAIN = b"agent-evolve:contextual-search-observation:v2\x00"
_DELAYED_CREDIT_DOMAIN = b"agent-evolve:contextual-search-delayed-credit:v2\x00"
_SNAPSHOT_DOMAIN = b"agent-evolve:contextual-search-snapshot:v2\x00"
_QUERY_DOMAIN = b"agent-evolve:contextual-search-query:v3\x00"
_DECISION_DOMAIN = b"agent-evolve:contextual-search-decision:v3\x00"
_ALLOCATION_SLICE_DOMAIN = b"agent-evolve:contextual-search-allocation-slice:v1\x00"
_STAGE_ALLOCATION_DOMAIN = b"agent-evolve:contextual-search-stage-allocation:v1\x00"
_COMPLETION_AUDIT_DOMAIN = b"agent-evolve:contextual-search-completion-audit:v1\x00"


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
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _canonical_tokens(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a non-empty exact tuple")
    for value in values:
        _require_token(value, name=name)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


class SearchPhase(str, Enum):
    BASIN_ACQUISITION = "basin_acquisition"
    BASIN_EXPANSION = "basin_expansion"
    COMPOSITION = "composition"
    TERMINAL_CONVERSION = "terminal_conversion"


class SearchArmKind(str, Enum):
    SOURCE = "source"
    OPERATOR = "operator"


@dataclass(frozen=True, slots=True)
class ContextualSearchObservation:
    """One evaluated action with normalized, separately typed credit channels."""

    campaign_scope_sha256: str
    wave_index: int
    source_id: str
    operator_id: str
    option_identity_sha256: str
    parent_context_sha256: str
    feasible: bool
    positive_marginal_utility: bool
    normalized_marginal_utility: float
    marginal_utility_share: float
    stage_front_persisted: bool | None = None
    final_front_persisted: bool | None = None
    useful_descendant_observed: bool | None = None
    source_distance: float = 0.0
    candidate_id: CandidateId | None = None
    observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_scope_sha256",
            "option_identity_sha256",
            "parent_context_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be a positive exact integer")
        _require_token(self.source_id, name="source_id")
        _require_token(self.operator_id, name="operator_id")
        for name in ("feasible", "positive_marginal_utility"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        for name in (
            "stage_front_persisted",
            "final_front_persisted",
            "useful_descendant_observed",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be an exact bool or None")
        _require_probability(
            self.normalized_marginal_utility,
            name="normalized_marginal_utility",
        )
        _require_probability(
            self.marginal_utility_share,
            name="marginal_utility_share",
        )
        _require_probability(self.source_distance, name="source_distance")
        if self.candidate_id is not None:
            if type(self.candidate_id) is not CandidateId:
                raise TypeError("candidate_id must be exact CandidateId or None")
            CandidateId.__post_init__(self.candidate_id)
        if not self.feasible and (
            self.positive_marginal_utility
            or self.normalized_marginal_utility != 0.0
            or self.marginal_utility_share != 0.0
            or self.stage_front_persisted is True
            or self.final_front_persisted is True
            or self.useful_descendant_observed is True
        ):
            raise ValueError("infeasible observations cannot carry positive yield")
        if self.positive_marginal_utility != (
            self.normalized_marginal_utility > 0.0 and self.marginal_utility_share > 0.0
        ):
            raise ValueError("positive verdict differs from its utility channels")
        object.__setattr__(
            self,
            "observation_sha256",
            _hash(_OBSERVATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "wave_index": self.wave_index,
            "source_id": self.source_id,
            "operator_id": self.operator_id,
            "option_identity_sha256": self.option_identity_sha256,
            "parent_context_sha256": self.parent_context_sha256,
            "feasible": self.feasible,
            "positive_marginal_utility": self.positive_marginal_utility,
            "normalized_marginal_utility_hex": (self.normalized_marginal_utility.hex()),
            "marginal_utility_share_hex": self.marginal_utility_share.hex(),
            "stage_front_persisted": self.stage_front_persisted,
            "final_front_persisted": self.final_front_persisted,
            "useful_descendant_observed": self.useful_descendant_observed,
            "source_distance_hex": self.source_distance.hex(),
            "candidate_id": (
                None if self.candidate_id is None else self.candidate_id.value
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "observation_sha256": self.observation_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextualSearchDelayedCredit:
    """A later outcome joined to one immutable evaluated-action observation."""

    campaign_scope_sha256: str
    source_observation_sha256: str
    available_at_wave_index: int
    stage_front_persisted: bool | None = None
    final_front_persisted: bool | None = None
    useful_descendant_observed: bool | None = None
    credit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(
            self.source_observation_sha256,
            "source_observation_sha256",
        )
        if (
            type(self.available_at_wave_index) is not int
            or self.available_at_wave_index <= 0
        ):
            raise ValueError("available_at_wave_index must be positive")
        for name in (
            "stage_front_persisted",
            "final_front_persisted",
            "useful_descendant_observed",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be an exact bool or None")
        if (
            self.stage_front_persisted is None
            and self.final_front_persisted is None
            and self.useful_descendant_observed is None
        ):
            raise ValueError("delayed credit must adjudicate at least one channel")
        object.__setattr__(
            self,
            "credit_sha256",
            _hash(_DELAYED_CREDIT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "source_observation_sha256": self.source_observation_sha256,
            "available_at_wave_index": self.available_at_wave_index,
            "stage_front_persisted": self.stage_front_persisted,
            "final_front_persisted": self.final_front_persisted,
            "useful_descendant_observed": self.useful_descendant_observed,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "credit_sha256": self.credit_sha256}


@dataclass(frozen=True, slots=True)
class ContextualArmPosterior:
    """Independent evidence channels for one source or operator arm."""

    kind: SearchArmKind
    arm_id: str
    observation_count: int
    positive_count: int
    feasible_count: int
    stage_persistence_observation_count: int
    stage_persistence_positive_count: int
    persistence_observation_count: int
    persistence_positive_count: int
    descendant_observation_count: int
    descendant_positive_count: int
    allocation_requested_slot_count: int
    allocation_realized_overlap_count: int
    allocation_projection_count: int
    normalized_marginal_utility_sum: float
    marginal_utility_share_sum: float
    mean_source_distance: float

    def __post_init__(self) -> None:
        if type(self.kind) is not SearchArmKind:
            raise TypeError("kind must be exact SearchArmKind")
        _require_token(self.arm_id, name="arm_id")
        for name in (
            "observation_count",
            "positive_count",
            "feasible_count",
            "stage_persistence_observation_count",
            "stage_persistence_positive_count",
            "persistence_observation_count",
            "persistence_positive_count",
            "descendant_observation_count",
            "descendant_positive_count",
            "allocation_requested_slot_count",
            "allocation_realized_overlap_count",
            "allocation_projection_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.positive_count > self.observation_count:
            raise ValueError("positive count exceeds observations")
        if self.feasible_count > self.observation_count:
            raise ValueError("feasible count exceeds observations")
        if self.stage_persistence_positive_count > (
            self.stage_persistence_observation_count
        ):
            raise ValueError("stage persistence positives exceed observations")
        if self.persistence_positive_count > self.persistence_observation_count:
            raise ValueError("persistence positives exceed observations")
        if self.descendant_positive_count > self.descendant_observation_count:
            raise ValueError("descendant positives exceed observations")
        if self.allocation_realized_overlap_count > (
            self.allocation_requested_slot_count
        ):
            raise ValueError("allocation overlap exceeds requested slots")
        if (
            self.allocation_requested_slot_count == 0
            and self.allocation_projection_count != 0
        ):
            raise ValueError("allocation projections require requested slots")
        for name in (
            "normalized_marginal_utility_sum",
            "marginal_utility_share_sum",
        ):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= float(self.observation_count)
            ):
                raise ValueError(f"{name} is invalid")
        _require_probability(self.mean_source_distance, name="mean_source_distance")

    @staticmethod
    def _beta_mean(successes: int, observations: int) -> float:
        return (1.0 + successes) / (2.0 + observations)

    @staticmethod
    def _beta_variance(successes: int, observations: int) -> float:
        alpha = 1.0 + successes
        beta = 1.0 + observations - successes
        total = alpha + beta
        return (alpha * beta) / (total * total * (total + 1.0))

    @property
    def positive_probability(self) -> float:
        return self._beta_mean(self.positive_count, self.observation_count)

    @property
    def positive_uncertainty(self) -> float:
        return math.sqrt(
            self._beta_variance(self.positive_count, self.observation_count)
        )

    @property
    def return_probability(self) -> float:
        """Empirical mean of bounded normalized archive return.

        Infeasible evaluated actions already contribute zero utility, so
        feasibility is a return gate rather than an independently rewarded
        feature.  An unseen arm receives the neutral cold-start value only
        until its first real observation; no unit pseudo-count is allowed to
        dwarf the small archive gains common in expensive optimization.
        """

        if self.observation_count == 0:
            return 0.5
        return self.normalized_marginal_utility_sum / self.observation_count

    @property
    def return_uncertainty(self) -> float:
        # A variable bounded in [0, 1] has standard deviation at most 1/2.
        # This conservative standard-error bound is scale free and, because
        # only one explicit exploration slot consumes it, cannot inflate the
        # entire exploitation allocation.
        return 0.5 / math.sqrt(max(1, self.observation_count))

    @property
    def feasibility_probability(self) -> float:
        return self._beta_mean(self.feasible_count, self.observation_count)

    @property
    def persistence_probability(self) -> float:
        return self._beta_mean(
            self.persistence_positive_count,
            self.persistence_observation_count,
        )

    @property
    def stage_persistence_probability(self) -> float:
        return self._beta_mean(
            self.stage_persistence_positive_count,
            self.stage_persistence_observation_count,
        )

    @property
    def descendant_probability(self) -> float:
        return self._beta_mean(
            self.descendant_positive_count,
            self.descendant_observation_count,
        )

    @property
    def allocation_realizability_probability(self) -> float:
        return self._beta_mean(
            self.allocation_realized_overlap_count,
            self.allocation_requested_slot_count,
        )

    @property
    def mean_marginal_utility_share(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.marginal_utility_share_sum / self.observation_count

    @property
    def mean_normalized_marginal_utility(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.normalized_marginal_utility_sum / self.observation_count

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "kind": self.kind.value,
            "arm_id": self.arm_id,
            "observation_count": self.observation_count,
            "positive_count": self.positive_count,
            "feasible_count": self.feasible_count,
            "stage_persistence_observation_count": (
                self.stage_persistence_observation_count
            ),
            "stage_persistence_positive_count": (self.stage_persistence_positive_count),
            "persistence_observation_count": self.persistence_observation_count,
            "persistence_positive_count": self.persistence_positive_count,
            "descendant_observation_count": self.descendant_observation_count,
            "descendant_positive_count": self.descendant_positive_count,
            "allocation_requested_slot_count": (self.allocation_requested_slot_count),
            "allocation_realized_overlap_count": (
                self.allocation_realized_overlap_count
            ),
            "allocation_projection_count": self.allocation_projection_count,
            "normalized_marginal_utility_sum_hex": (
                self.normalized_marginal_utility_sum.hex()
            ),
            "marginal_utility_share_sum_hex": (self.marginal_utility_share_sum.hex()),
            "mean_source_distance_hex": self.mean_source_distance.hex(),
            "posterior": {
                "positive_probability_hex": self.positive_probability.hex(),
                "positive_uncertainty_hex": self.positive_uncertainty.hex(),
                "feasibility_probability_hex": self.feasibility_probability.hex(),
                "stage_persistence_probability_hex": (
                    self.stage_persistence_probability.hex()
                ),
                "persistence_probability_hex": self.persistence_probability.hex(),
                "descendant_probability_hex": self.descendant_probability.hex(),
                "allocation_realizability_probability_hex": (
                    self.allocation_realizability_probability.hex()
                ),
                "mean_normalized_marginal_utility_hex": (
                    self.mean_normalized_marginal_utility.hex()
                ),
                "return_probability_hex": self.return_probability.hex(),
                "return_uncertainty_hex": self.return_uncertainty.hex(),
                "mean_marginal_utility_share_hex": (
                    self.mean_marginal_utility_share.hex()
                ),
            },
        }


def _posterior(
    observations: tuple[ContextualSearchObservation, ...],
    *,
    kind: SearchArmKind,
    arm_id: str,
    delayed_credits: tuple[ContextualSearchDelayedCredit, ...] = (),
    allocation_realizations: tuple[ContextualPortfolioAllocationRealization, ...] = (),
) -> ContextualArmPosterior:
    selected = tuple(
        value
        for value in observations
        if (value.source_id if kind is SearchArmKind.SOURCE else value.operator_id)
        == arm_id
    )
    delayed_by_channel = {
        (value.source_observation_sha256, name): getattr(value, name)
        for value in delayed_credits
        for name in (
            "stage_front_persisted",
            "final_front_persisted",
            "useful_descendant_observed",
        )
        if getattr(value, name) is not None
    }

    def delayed_value(
        value: ContextualSearchObservation,
        name: str,
    ) -> bool | None:
        immediate = getattr(value, name)
        if immediate is not None:
            return immediate
        return delayed_by_channel.get((value.observation_sha256, name))

    stage_persistence = tuple(
        delayed_value(value, "stage_front_persisted") for value in selected
    )
    stage_persistence = tuple(value for value in stage_persistence if value is not None)
    persistence = tuple(
        delayed_value(value, "final_front_persisted") for value in selected
    )
    persistence = tuple(value for value in persistence if value is not None)
    descendants = tuple(
        delayed_value(value, "useful_descendant_observed") for value in selected
    )
    descendants = tuple(value for value in descendants if value is not None)
    allocation_rows = tuple(
        value
        for value in allocation_realizations
        if dict(
            value.requested_source_target_counts
            if kind is SearchArmKind.SOURCE
            else value.requested_operator_target_counts
        ).get(arm_id, 0)
        > 0
    )
    requested_slots = sum(
        dict(
            value.requested_source_target_counts
            if kind is SearchArmKind.SOURCE
            else value.requested_operator_target_counts
        )[arm_id]
        for value in allocation_rows
    )
    realized_overlap = sum(
        min(
            dict(
                value.requested_source_target_counts
                if kind is SearchArmKind.SOURCE
                else value.requested_operator_target_counts
            )[arm_id],
            dict(
                value.realized_source_target_counts
                if kind is SearchArmKind.SOURCE
                else value.realized_operator_target_counts
            )[arm_id],
        )
        for value in allocation_rows
    )
    return ContextualArmPosterior(
        kind=kind,
        arm_id=arm_id,
        observation_count=len(selected),
        positive_count=sum(value.positive_marginal_utility for value in selected),
        feasible_count=sum(value.feasible for value in selected),
        stage_persistence_observation_count=len(stage_persistence),
        stage_persistence_positive_count=sum(
            value is True for value in stage_persistence
        ),
        persistence_observation_count=len(persistence),
        persistence_positive_count=sum(value is True for value in persistence),
        descendant_observation_count=len(descendants),
        descendant_positive_count=sum(value is True for value in descendants),
        allocation_requested_slot_count=requested_slots,
        allocation_realized_overlap_count=realized_overlap,
        allocation_projection_count=len(allocation_rows),
        normalized_marginal_utility_sum=float(
            sum(value.normalized_marginal_utility for value in selected)
        ),
        marginal_utility_share_sum=float(
            sum(value.marginal_utility_share for value in selected)
        ),
        mean_source_distance=(
            0.0
            if not selected
            else float(sum(value.source_distance for value in selected) / len(selected))
        ),
    )


@dataclass(frozen=True, slots=True)
class ContextualSearchSnapshot:
    campaign_scope_sha256: str
    cutoff_wave_index_exclusive: int
    observation_sha256s: tuple[str, ...]
    delayed_credit_sha256s: tuple[str, ...]
    allocation_realization_sha256s: tuple[str, ...]
    source_posteriors: tuple[ContextualArmPosterior, ...]
    operator_posteriors: tuple[ContextualArmPosterior, ...]
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if (
            type(self.cutoff_wave_index_exclusive) is not int
            or self.cutoff_wave_index_exclusive <= 0
        ):
            raise ValueError("cutoff_wave_index_exclusive must be positive")
        if self.observation_sha256s != tuple(sorted(set(self.observation_sha256s))):
            raise ValueError("observation hashes must be unique and canonical")
        for value in self.observation_sha256s:
            require_sha256(value, "observation_sha256")
        if self.delayed_credit_sha256s != tuple(
            sorted(set(self.delayed_credit_sha256s))
        ):
            raise ValueError("delayed credit hashes must be unique and canonical")
        for value in self.delayed_credit_sha256s:
            require_sha256(value, "delayed_credit_sha256")
        if self.allocation_realization_sha256s != tuple(
            sorted(set(self.allocation_realization_sha256s))
        ):
            raise ValueError(
                "allocation realization hashes must be unique and canonical"
            )
        for value in self.allocation_realization_sha256s:
            require_sha256(value, "allocation_realization_sha256")
        for name, kind in (
            ("source_posteriors", SearchArmKind.SOURCE),
            ("operator_posteriors", SearchArmKind.OPERATOR),
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not ContextualArmPosterior for value in values
            ):
                raise TypeError(f"{name} must contain exact posteriors")
            for value in values:
                value.__post_init__()
                if value.kind is not kind:
                    raise ValueError(f"{name} contains a foreign arm kind")
            if tuple(value.arm_id for value in values) != tuple(
                sorted({value.arm_id for value in values})
            ):
                raise ValueError(f"{name} must use canonical unique arm IDs")
        object.__setattr__(
            self,
            "snapshot_sha256",
            _hash(_SNAPSHOT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "cutoff_wave_index_exclusive": self.cutoff_wave_index_exclusive,
            "observation_sha256s": list(self.observation_sha256s),
            "delayed_credit_sha256s": list(self.delayed_credit_sha256s),
            "allocation_realization_sha256s": list(self.allocation_realization_sha256s),
            "source_posteriors": [
                value.to_record() for value in self.source_posteriors
            ],
            "operator_posteriors": [
                value.to_record() for value in self.operator_posteriors
            ],
            "policy": {
                "policy_id": CONTEXTUAL_SEARCH_CONTROLLER_ID,
                "policy_version": CONTEXTUAL_SEARCH_CONTROLLER_VERSION,
                "definition_sha256": CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "snapshot_sha256": self.snapshot_sha256}


@dataclass(slots=True)
class ContextualSearchLedger:
    """Append-only observation ledger with strict prior-wave snapshots."""

    observations: list[ContextualSearchObservation] = field(default_factory=list)
    delayed_credits: list[ContextualSearchDelayedCredit] = field(default_factory=list)
    allocation_realizations: list[ContextualPortfolioAllocationRealization] = field(
        default_factory=list
    )

    def append_batch(
        self,
        observations: tuple[ContextualSearchObservation, ...],
    ) -> None:
        if (
            type(observations) is not tuple
            or not observations
            or any(
                type(value) is not ContextualSearchObservation for value in observations
            )
        ):
            raise ValueError("observations must contain exact controller observations")
        for value in observations:
            value.__post_init__()
        combined = (*self.observations, *observations)
        identities = tuple(value.observation_sha256 for value in combined)
        if len(set(identities)) != len(identities):
            raise ValueError("controller ledger cannot repeat an observation")
        if tuple(
            (value.wave_index, value.observation_sha256) for value in combined
        ) != tuple(
            sorted((value.wave_index, value.observation_sha256) for value in combined)
        ):
            raise ValueError("controller observations must be appended canonically")
        scopes = {value.campaign_scope_sha256 for value in combined}
        if len(scopes) != 1:
            raise ValueError("controller ledger cannot mix campaign scopes")
        self.observations.extend(observations)

    def append_delayed_credit_batch(
        self,
        credits: tuple[ContextualSearchDelayedCredit, ...],
    ) -> None:
        if (
            type(credits) is not tuple
            or not credits
            or any(
                type(value) is not ContextualSearchDelayedCredit for value in credits
            )
        ):
            raise ValueError("credits must contain exact delayed-credit events")
        observations = {value.observation_sha256: value for value in self.observations}
        combined = (*self.delayed_credits, *credits)
        if len({value.credit_sha256 for value in combined}) != len(combined):
            raise ValueError("controller ledger cannot repeat delayed credit")
        if tuple(
            (value.available_at_wave_index, value.credit_sha256) for value in combined
        ) != tuple(
            sorted(
                (value.available_at_wave_index, value.credit_sha256)
                for value in combined
            )
        ):
            raise ValueError("delayed credits must be appended canonically")
        occupied: set[tuple[str, str]] = set()
        for value in combined:
            value.__post_init__()
            source = observations.get(value.source_observation_sha256)
            if source is None:
                raise ValueError("delayed credit cites an unknown observation")
            if source.campaign_scope_sha256 != value.campaign_scope_sha256:
                raise ValueError("delayed credit crosses campaign scopes")
            if value.available_at_wave_index <= source.wave_index:
                raise ValueError("delayed credit predates its source observation")
            for name in (
                "stage_front_persisted",
                "final_front_persisted",
                "useful_descendant_observed",
            ):
                delayed = getattr(value, name)
                if delayed is None:
                    continue
                if getattr(source, name) is not None:
                    raise ValueError("delayed credit overwrites immediate evidence")
                key = (source.observation_sha256, name)
                if key in occupied:
                    raise ValueError("delayed credit repeats one evidence channel")
                occupied.add(key)
        self.delayed_credits.extend(credits)

    def append_allocation_realization_batch(
        self,
        realizations: tuple[ContextualPortfolioAllocationRealization, ...],
    ) -> None:
        if (
            type(realizations) is not tuple
            or not realizations
            or any(
                type(value) is not ContextualPortfolioAllocationRealization
                for value in realizations
            )
        ):
            raise ValueError("realizations must contain exact allocation evidence")
        for value in realizations:
            value.__post_init__()
        combined = (*self.allocation_realizations, *realizations)
        identities = tuple(value.realization_sha256 for value in combined)
        if len(set(identities)) != len(identities):
            raise ValueError(
                "controller ledger cannot repeat an allocation realization"
            )
        if tuple(
            (value.controller_wave_index, value.slice_id, value.realization_sha256)
            for value in combined
        ) != tuple(
            sorted(
                (
                    value.controller_wave_index,
                    value.slice_id,
                    value.realization_sha256,
                )
                for value in combined
            )
        ):
            raise ValueError("allocation realizations must be appended canonically")
        scopes = {value.campaign_scope_sha256 for value in combined}
        if len(scopes) != 1:
            raise ValueError("controller ledger cannot mix realization scopes")
        self.allocation_realizations.extend(realizations)

    def snapshot(
        self,
        *,
        campaign_scope_sha256: str,
        cutoff_wave_index_exclusive: int,
        available_source_ids: tuple[str, ...],
        available_operator_ids: tuple[str, ...],
    ) -> ContextualSearchSnapshot:
        require_sha256(campaign_scope_sha256, "campaign_scope_sha256")
        if (
            type(cutoff_wave_index_exclusive) is not int
            or cutoff_wave_index_exclusive <= 0
        ):
            raise ValueError("cutoff_wave_index_exclusive must be positive")
        _canonical_tokens(available_source_ids, name="available_source_ids")
        _canonical_tokens(available_operator_ids, name="available_operator_ids")
        selected = tuple(
            value
            for value in self.observations
            if value.campaign_scope_sha256 == campaign_scope_sha256
            and value.wave_index < cutoff_wave_index_exclusive
        )
        selected_hashes = {value.observation_sha256 for value in selected}
        delayed = tuple(
            value
            for value in self.delayed_credits
            if value.campaign_scope_sha256 == campaign_scope_sha256
            and value.available_at_wave_index <= cutoff_wave_index_exclusive
            and value.source_observation_sha256 in selected_hashes
        )
        realizations = tuple(
            value
            for value in self.allocation_realizations
            if value.campaign_scope_sha256 == campaign_scope_sha256
            and value.controller_wave_index < cutoff_wave_index_exclusive
        )
        return ContextualSearchSnapshot(
            campaign_scope_sha256=campaign_scope_sha256,
            cutoff_wave_index_exclusive=cutoff_wave_index_exclusive,
            observation_sha256s=tuple(
                sorted(value.observation_sha256 for value in selected)
            ),
            delayed_credit_sha256s=tuple(
                sorted(value.credit_sha256 for value in delayed)
            ),
            allocation_realization_sha256s=tuple(
                sorted(value.realization_sha256 for value in realizations)
            ),
            source_posteriors=tuple(
                _posterior(
                    selected,
                    kind=SearchArmKind.SOURCE,
                    arm_id=value,
                    delayed_credits=delayed,
                    allocation_realizations=realizations,
                )
                for value in available_source_ids
            ),
            operator_posteriors=tuple(
                _posterior(
                    selected,
                    kind=SearchArmKind.OPERATOR,
                    arm_id=value,
                    delayed_credits=delayed,
                    allocation_realizations=realizations,
                )
                for value in available_operator_ids
            ),
        )


@dataclass(frozen=True, slots=True)
class ContextualSearchCompletionAudit:
    """Generic post-campaign integrity audit for the adaptive loop."""

    campaign_scope_sha256: str
    expected_wave_count: int
    expected_post_recombination_wave_indices: tuple[int, ...]
    expected_observation_count: int
    expected_stage_credit_count: int
    expected_allocation_realization_count: int
    observation_count: int
    allocation_realization_count: int
    delayed_credit_count: int
    candidate_bound_observation_count: int
    unique_candidate_count: int
    observation_wave_indices: tuple[int, ...]
    allocation_wave_indices: tuple[int, ...]
    stage_credit_wave_indices: tuple[int, ...]
    stage_credit_source_sha256s: tuple[str, ...]
    final_credit_source_sha256s: tuple[str, ...]
    descendant_credit_source_sha256s: tuple[str, ...]
    descendant_credit_wave_indices: tuple[int, ...]
    audit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        for name in (
            "expected_wave_count",
            "expected_observation_count",
            "expected_allocation_realization_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "observation_count",
            "expected_stage_credit_count",
            "allocation_realization_count",
            "delayed_credit_count",
            "candidate_bound_observation_count",
            "unique_candidate_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "observation_wave_indices",
            "allocation_wave_indices",
            "expected_post_recombination_wave_indices",
            "stage_credit_wave_indices",
            "descendant_credit_wave_indices",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not int or value <= 0 for value in values
            ):
                raise ValueError(f"{name} must contain positive exact integers")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if not set(self.expected_post_recombination_wave_indices).issubset(
            self.expected_wave_indices
        ):
            raise ValueError(
                "post-recombination wave indices escape the expected horizon"
            )
        for name in (
            "stage_credit_source_sha256s",
            "final_credit_source_sha256s",
            "descendant_credit_source_sha256s",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be a canonical exact tuple")
            for value in values:
                require_sha256(value, name)
        object.__setattr__(
            self,
            "audit_sha256",
            _hash(_COMPLETION_AUDIT_DOMAIN, self._unsigned_record()),
        )

    @property
    def expected_wave_indices(self) -> tuple[int, ...]:
        return tuple(range(1, self.expected_wave_count + 1))

    @property
    def healthy(self) -> bool:
        return (
            self.observation_count == self.expected_observation_count
            and self.allocation_realization_count
            == self.expected_allocation_realization_count
            and self.delayed_credit_count
            == self.expected_stage_credit_count + self.expected_observation_count
            and self.candidate_bound_observation_count
            == self.expected_observation_count
            and self.unique_candidate_count == self.expected_observation_count
            and self.observation_wave_indices == self.expected_wave_indices
            and self.allocation_wave_indices == self.expected_wave_indices
            and self.stage_credit_wave_indices
            == self.expected_post_recombination_wave_indices
            and len(self.stage_credit_source_sha256s)
            == self.expected_stage_credit_count
            and len(self.final_credit_source_sha256s) == self.expected_observation_count
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "expected_wave_count": self.expected_wave_count,
            "expected_post_recombination_wave_indices": list(
                self.expected_post_recombination_wave_indices
            ),
            "expected_observation_count": self.expected_observation_count,
            "expected_stage_credit_count": self.expected_stage_credit_count,
            "expected_allocation_realization_count": (
                self.expected_allocation_realization_count
            ),
            "observation_count": self.observation_count,
            "allocation_realization_count": self.allocation_realization_count,
            "delayed_credit_count": self.delayed_credit_count,
            "candidate_bound_observation_count": (
                self.candidate_bound_observation_count
            ),
            "unique_candidate_count": self.unique_candidate_count,
            "observation_wave_indices": list(self.observation_wave_indices),
            "allocation_wave_indices": list(self.allocation_wave_indices),
            "stage_credit_wave_indices": list(self.stage_credit_wave_indices),
            "stage_credit_source_sha256s": list(self.stage_credit_source_sha256s),
            "final_credit_source_sha256s": list(self.final_credit_source_sha256s),
            "descendant_credit_source_sha256s": list(
                self.descendant_credit_source_sha256s
            ),
            "descendant_credit_wave_indices": list(self.descendant_credit_wave_indices),
            "healthy": self.healthy,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "audit_sha256": self.audit_sha256}


def audit_completed_contextual_search_ledger(
    ledger: ContextualSearchLedger,
    *,
    campaign_scope_sha256: str,
    expected_wave_count: int,
    expected_post_recombination_wave_indices: tuple[int, ...],
    expected_observation_count: int,
    expected_allocation_realization_count: int,
) -> ContextualSearchCompletionAudit:
    """Verify candidate binding and both delayed horizons after finalization."""

    if type(ledger) is not ContextualSearchLedger:
        raise TypeError("ledger must be exact ContextualSearchLedger")
    require_sha256(campaign_scope_sha256, "campaign_scope_sha256")
    observations = tuple(
        value
        for value in ledger.observations
        if value.campaign_scope_sha256 == campaign_scope_sha256
    )
    realizations = tuple(
        value
        for value in ledger.allocation_realizations
        if value.campaign_scope_sha256 == campaign_scope_sha256
    )
    credits = tuple(
        value
        for value in ledger.delayed_credits
        if value.campaign_scope_sha256 == campaign_scope_sha256
    )
    observation_by_sha256 = {value.observation_sha256: value for value in observations}

    def sources(name: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                value.source_observation_sha256
                for value in credits
                if getattr(value, name) is not None
            )
        )

    stage_sources = sources("stage_front_persisted")
    descendant_sources = sources("useful_descendant_observed")
    candidate_ids = tuple(
        value.candidate_id for value in observations if value.candidate_id is not None
    )
    return ContextualSearchCompletionAudit(
        campaign_scope_sha256=campaign_scope_sha256,
        expected_wave_count=expected_wave_count,
        expected_post_recombination_wave_indices=(
            expected_post_recombination_wave_indices
        ),
        expected_observation_count=expected_observation_count,
        expected_stage_credit_count=sum(
            value.wave_index in expected_post_recombination_wave_indices
            for value in observations
        ),
        expected_allocation_realization_count=(expected_allocation_realization_count),
        observation_count=len(observations),
        allocation_realization_count=len(realizations),
        delayed_credit_count=len(credits),
        candidate_bound_observation_count=len(candidate_ids),
        unique_candidate_count=len(set(candidate_ids)),
        observation_wave_indices=tuple(
            sorted({value.wave_index for value in observations})
        ),
        allocation_wave_indices=tuple(
            sorted({value.controller_wave_index for value in realizations})
        ),
        stage_credit_source_sha256s=stage_sources,
        final_credit_source_sha256s=sources("final_front_persisted"),
        descendant_credit_source_sha256s=descendant_sources,
        stage_credit_wave_indices=tuple(
            sorted(
                {
                    observation_by_sha256[value].wave_index
                    for value in stage_sources
                    if value in observation_by_sha256
                }
            )
        ),
        descendant_credit_wave_indices=tuple(
            sorted(
                {
                    observation_by_sha256[value].wave_index
                    for value in descendant_sources
                    if value in observation_by_sha256
                }
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ContextualSearchQuery:
    campaign_scope_sha256: str
    wave_index: int
    total_portfolio_waves: int
    real_evaluation_slots: int
    available_source_ids: tuple[str, ...]
    available_operator_ids: tuple[str, ...]
    incumbent_source_id: str
    incumbent_operator_id: str
    archive_front_size: int
    recent_normalized_archive_gains: tuple[float, ...] = ()
    composition_evidence_available: bool = False
    source_count_capability: ContextualArmCountCapability | None = None
    operator_count_capability: ContextualArmCountCapability | None = None
    joint_count_capabilities: tuple[ContextualLaneJointCountCapability, ...] = ()
    query_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be positive")
        if (
            type(self.total_portfolio_waves) is not int
            or self.total_portfolio_waves <= 0
            or self.wave_index > self.total_portfolio_waves
        ):
            raise ValueError("total_portfolio_waves must cover the current wave")
        if (
            type(self.real_evaluation_slots) is not int
            or self.real_evaluation_slots <= 0
        ):
            raise ValueError("real_evaluation_slots must be positive")
        _canonical_tokens(self.available_source_ids, name="available_source_ids")
        _canonical_tokens(self.available_operator_ids, name="available_operator_ids")
        _require_token(self.incumbent_source_id, name="incumbent_source_id")
        if self.incumbent_source_id not in self.available_source_ids:
            raise ValueError("incumbent source must be available")
        _require_token(self.incumbent_operator_id, name="incumbent_operator_id")
        if self.incumbent_operator_id not in self.available_operator_ids:
            raise ValueError("incumbent operator must be available")
        if type(self.archive_front_size) is not int or self.archive_front_size <= 0:
            raise ValueError("archive_front_size must be positive")
        if type(self.recent_normalized_archive_gains) is not tuple:
            raise TypeError("recent_normalized_archive_gains must be an exact tuple")
        for value in self.recent_normalized_archive_gains:
            _require_probability(value, name="recent_normalized_archive_gain")
        if len(self.recent_normalized_archive_gains) > 4:
            raise ValueError("recent gain history is bounded to four stages")
        if type(self.composition_evidence_available) is not bool:
            raise TypeError("composition_evidence_available must be an exact bool")
        for name, kind, arm_ids in (
            (
                "source_count_capability",
                "source",
                self.available_source_ids,
            ),
            (
                "operator_count_capability",
                "operator",
                self.available_operator_ids,
            ),
        ):
            capability = getattr(self, name)
            if capability is None:
                continue
            if type(capability) is not ContextualArmCountCapability:
                raise TypeError(f"{name} must be an exact capability or None")
            capability.__post_init__()
            if (
                capability.kind != kind
                or capability.arm_ids != arm_ids
                or capability.evaluation_slots != self.real_evaluation_slots
            ):
                raise ValueError(f"{name} differs from the controller query")
            if any(
                witness.controller_wave_index >= self.wave_index
                for witness in capability.witnesses
            ):
                raise ValueError(f"{name} contains current/future-wave evidence")
        capabilities = self.joint_count_capabilities
        if type(capabilities) is not tuple or any(
            type(value) is not ContextualLaneJointCountCapability
            for value in capabilities
        ):
            raise TypeError(
                "joint_count_capabilities must contain exact lane capabilities"
            )
        for value in capabilities:
            value.__post_init__()
            if (
                value.source_arm_ids != self.available_source_ids
                or value.operator_arm_ids != self.available_operator_ids
            ):
                raise ValueError("joint capability differs from controller arms")
        if capabilities:
            if tuple(value.slice_id for value in capabilities) != tuple(
                sorted({value.slice_id for value in capabilities})
            ):
                raise ValueError(
                    "joint lane capabilities must be unique and canonical"
                )
            if sum(value.evaluation_slots for value in capabilities) != (
                self.real_evaluation_slots
            ):
                raise ValueError(
                    "joint lane capabilities must cover every evaluation slot"
                )
        object.__setattr__(
            self,
            "query_sha256",
            _hash(_QUERY_DOMAIN, self._unsigned_record()),
        )

    @property
    def remaining_portfolio_waves(self) -> int:
        return self.total_portfolio_waves - self.wave_index + 1

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "wave_index": self.wave_index,
            "total_portfolio_waves": self.total_portfolio_waves,
            "remaining_portfolio_waves": self.remaining_portfolio_waves,
            "real_evaluation_slots": self.real_evaluation_slots,
            "available_source_ids": list(self.available_source_ids),
            "available_operator_ids": list(self.available_operator_ids),
            "incumbent_source_id": self.incumbent_source_id,
            "incumbent_operator_id": self.incumbent_operator_id,
            "archive_front_size": self.archive_front_size,
            "recent_normalized_archive_gain_hex": [
                value.hex() for value in self.recent_normalized_archive_gains
            ],
            "composition_evidence_available": self.composition_evidence_available,
            "source_count_capability": (
                None
                if self.source_count_capability is None
                else self.source_count_capability.to_record()
            ),
            "operator_count_capability": (
                None
                if self.operator_count_capability is None
                else self.operator_count_capability.to_record()
            ),
            "joint_count_capabilities": [
                value.to_record() for value in self.joint_count_capabilities
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "query_sha256": self.query_sha256}


@dataclass(frozen=True, slots=True)
class ContextualArmAllocation:
    kind: SearchArmKind
    arm_id: str
    target_slots: int
    score: float
    allocation_probability: float
    exploration_slot: bool
    unconstrained_target_slots: int
    empirical_capability_projected: bool
    prospective_joint_capability_projected: bool = False
    prospective_joint_exploration_projected: bool = False

    def __post_init__(self) -> None:
        if type(self.kind) is not SearchArmKind:
            raise TypeError("kind must be exact SearchArmKind")
        _require_token(self.arm_id, name="arm_id")
        if type(self.target_slots) is not int or self.target_slots < 0:
            raise ValueError("target_slots must be a non-negative exact integer")
        if type(self.score) is not float or not math.isfinite(self.score):
            raise TypeError("score must be a finite canonical float")
        _require_probability(
            self.allocation_probability,
            name="allocation_probability",
        )
        if type(self.exploration_slot) is not bool:
            raise TypeError("exploration_slot must be an exact bool")
        if self.exploration_slot and self.target_slots == 0:
            raise ValueError("exploration marker requires a target slot")
        if (
            type(self.unconstrained_target_slots) is not int
            or self.unconstrained_target_slots < 0
        ):
            raise ValueError(
                "unconstrained_target_slots must be a non-negative exact integer"
            )
        if type(self.empirical_capability_projected) is not bool:
            raise TypeError("empirical_capability_projected must be an exact bool")
        if type(self.prospective_joint_capability_projected) is not bool:
            raise TypeError(
                "prospective_joint_capability_projected must be an exact bool"
            )
        if type(self.prospective_joint_exploration_projected) is not bool:
            raise TypeError(
                "prospective_joint_exploration_projected must be an exact bool"
            )
        if (
            self.prospective_joint_exploration_projected
            and not self.prospective_joint_capability_projected
        ):
            raise ValueError(
                "exploration projection requires joint-capability projection"
            )
        if not (
            self.empirical_capability_projected
            or self.prospective_joint_capability_projected
        ) and (
            self.target_slots != self.unconstrained_target_slots
        ):
            raise ValueError("unprojected allocation changed its unconstrained target")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "kind": self.kind.value,
            "arm_id": self.arm_id,
            "target_slots": self.target_slots,
            "score_hex": self.score.hex(),
            "allocation_probability_hex": self.allocation_probability.hex(),
            "exploration_slot": self.exploration_slot,
            "unconstrained_target_slots": self.unconstrained_target_slots,
            "empirical_capability_projected": self.empirical_capability_projected,
            "prospective_joint_capability_projected": (
                self.prospective_joint_capability_projected
            ),
            "prospective_joint_exploration_projected": (
                self.prospective_joint_exploration_projected
            ),
        }


@dataclass(frozen=True, slots=True)
class ContextualSearchDecision:
    query: ContextualSearchQuery
    snapshot: ContextualSearchSnapshot
    phase: SearchPhase
    source_allocations: tuple[ContextualArmAllocation, ...]
    operator_allocations: tuple[ContextualArmAllocation, ...]
    joint_capability_selection: tuple[tuple[str, str], ...] = ()
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.query) is not ContextualSearchQuery:
            raise TypeError("query must be exact ContextualSearchQuery")
        self.query.__post_init__()
        if type(self.snapshot) is not ContextualSearchSnapshot:
            raise TypeError("snapshot must be exact ContextualSearchSnapshot")
        self.snapshot.__post_init__()
        if (
            self.snapshot.campaign_scope_sha256 != self.query.campaign_scope_sha256
            or self.snapshot.cutoff_wave_index_exclusive != self.query.wave_index
        ):
            raise ValueError("controller snapshot differs from its query cutoff")
        snapshot_realizations = set(self.snapshot.allocation_realization_sha256s)
        for capability in (
            self.query.source_count_capability,
            self.query.operator_count_capability,
        ):
            if capability is not None and not set(
                capability.allocation_realization_sha256s
            ).issubset(snapshot_realizations):
                raise ValueError(
                    "controller capability cites evidence outside its snapshot"
                )
        if type(self.phase) is not SearchPhase:
            raise TypeError("phase must be exact SearchPhase")
        for name, kind, available in (
            (
                "source_allocations",
                SearchArmKind.SOURCE,
                self.query.available_source_ids,
            ),
            (
                "operator_allocations",
                SearchArmKind.OPERATOR,
                self.query.available_operator_ids,
            ),
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not ContextualArmAllocation for value in values
            ):
                raise TypeError(f"{name} must contain exact allocations")
            for value in values:
                value.__post_init__()
                if value.kind is not kind:
                    raise ValueError(f"{name} contains a foreign arm kind")
            if tuple(value.arm_id for value in values) != available:
                raise ValueError(f"{name} must cover available arms canonically")
            if sum(value.target_slots for value in values) != (
                self.query.real_evaluation_slots
            ):
                raise ValueError(f"{name} must allocate every real evaluation slot")
            if not math.isclose(
                sum(value.allocation_probability for value in values),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{name} allocation probabilities must sum to one")
            expected_exploration = (
                0
                if self.phase is SearchPhase.TERMINAL_CONVERSION
                else min(1, len(values) - 1)
            )
            if sum(value.exploration_slot for value in values) != expected_exploration:
                raise ValueError(f"{name} has an invalid exploration-slot count")
        selection = self.joint_capability_selection
        if type(selection) is not tuple or any(
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not str
            or type(value[1]) is not str
            for value in selection
        ):
            raise TypeError(
                "joint_capability_selection must contain exact slice/hash pairs"
            )
        capabilities = self.query.joint_count_capabilities
        if not capabilities:
            if selection:
                raise ValueError(
                    "decision selected a joint vector without query capabilities"
                )
        else:
            if tuple(slice_id for slice_id, _ in selection) != tuple(
                value.slice_id for value in capabilities
            ):
                raise ValueError(
                    "joint capability selection must exactly cover query lanes"
                )
            selected_vectors = tuple(
                capability.resolve_vector(vector_sha256)
                for capability, (slice_id, vector_sha256) in zip(
                    capabilities,
                    selection,
                    strict=True,
                )
            )
            if any(
                capability.slice_id != slice_id
                for capability, (slice_id, _) in zip(
                    capabilities,
                    selection,
                    strict=True,
                )
            ):
                raise ValueError("joint selection lane differs from its capability")
            for count_name, allocations in (
                ("source_target_counts", self.source_allocations),
                ("operator_target_counts", self.operator_allocations),
            ):
                aggregate = {
                    allocation.arm_id: sum(
                        dict(getattr(vector, count_name))[allocation.arm_id]
                        for vector in selected_vectors
                    )
                    for allocation in allocations
                }
                expected = {
                    allocation.arm_id: allocation.target_slots
                    for allocation in allocations
                }
                if aggregate != expected:
                    raise ValueError(
                        "joint capability selection differs from decision targets"
                    )
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_DECISION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "query_sha256": self.query.query_sha256,
            "snapshot_sha256": self.snapshot.snapshot_sha256,
            "phase": self.phase.value,
            "source_allocations": [
                value.to_record() for value in self.source_allocations
            ],
            "operator_allocations": [
                value.to_record() for value in self.operator_allocations
            ],
            "joint_capability_selection": [
                {"slice_id": slice_id, "vector_sha256": vector_sha256}
                for slice_id, vector_sha256 in self.joint_capability_selection
            ],
            "terminal_information_bonus": (
                0.0 if self.phase is SearchPhase.TERMINAL_CONVERSION else None
            ),
            "policy": {
                "policy_id": CONTEXTUAL_SEARCH_CONTROLLER_ID,
                "policy_version": CONTEXTUAL_SEARCH_CONTROLLER_VERSION,
                "definition_sha256": CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "query": self.query.to_record(),
            "snapshot": self.snapshot.to_record(),
            "decision_sha256": self.decision_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextualPortfolioAllocationSlice:
    """One exact request-local slice of a stage-global controller decision."""

    campaign_scope_sha256: str
    query_sha256: str
    decision_sha256: str
    wave_index: int
    phase: SearchPhase
    slice_id: str
    evaluation_slots: int
    source_target_counts: tuple[tuple[str, int], ...]
    operator_target_counts: tuple[tuple[str, int], ...]
    minimum_single_path_interventions: int = 0
    minimum_disjoint_parent_patch_pairs: int = 0
    feasibility_witness_option_identity_sha256s: tuple[str, ...] = ()
    slice_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_scope_sha256",
            "query_sha256",
            "decision_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be positive")
        if type(self.phase) is not SearchPhase:
            raise TypeError("phase must be exact SearchPhase")
        _require_token(self.slice_id, name="slice_id")
        if type(self.evaluation_slots) is not int or self.evaluation_slots <= 0:
            raise ValueError("evaluation_slots must be positive")
        if (
            type(self.minimum_single_path_interventions) is not int
            or not 0
            <= self.minimum_single_path_interventions
            <= self.evaluation_slots
        ):
            raise ValueError(
                "minimum_single_path_interventions must lie in slice capacity"
            )
        maximum_pairs = self.evaluation_slots * (self.evaluation_slots - 1) // 2
        if (
            type(self.minimum_disjoint_parent_patch_pairs) is not int
            or not 0
            <= self.minimum_disjoint_parent_patch_pairs
            <= maximum_pairs
        ):
            raise ValueError(
                "minimum_disjoint_parent_patch_pairs must lie in slice pair capacity"
            )
        for name in ("source_target_counts", "operator_target_counts"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a non-empty exact tuple")
            for arm_id, count in values:
                _require_token(arm_id, name=f"{name}.arm_id")
                if type(count) is not int or count < 0:
                    raise ValueError(f"{name} counts must be non-negative")
            if values != tuple(sorted(values)) or len(
                {arm_id for arm_id, _ in values}
            ) != len(values):
                raise ValueError(f"{name} must use canonical unique arms")
            if sum(count for _, count in values) != self.evaluation_slots:
                raise ValueError(f"{name} must allocate every slice slot")
        witnesses = self.feasibility_witness_option_identity_sha256s
        if type(witnesses) is not tuple:
            raise TypeError(
                "feasibility_witness_option_identity_sha256s must be an exact tuple"
            )
        if witnesses:
            if len(witnesses) != self.evaluation_slots or witnesses != tuple(
                sorted(set(witnesses))
            ):
                raise ValueError(
                    "allocation-slice feasibility witness must contain one canonical "
                    "unique option identity per evaluation slot"
                )
            for value in witnesses:
                require_sha256(value, "feasibility_witness_option_identity_sha256")
        object.__setattr__(
            self,
            "slice_sha256",
            _hash(_ALLOCATION_SLICE_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": (
                4
                if self.feasibility_witness_option_identity_sha256s
                else 3
                if self.minimum_disjoint_parent_patch_pairs
                else 2
                if self.minimum_single_path_interventions
                else 1
            ),
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "query_sha256": self.query_sha256,
            "decision_sha256": self.decision_sha256,
            "wave_index": self.wave_index,
            "phase": self.phase.value,
            "slice_id": self.slice_id,
            "evaluation_slots": self.evaluation_slots,
            "source_target_counts": [
                list(value) for value in self.source_target_counts
            ],
            "operator_target_counts": [
                list(value) for value in self.operator_target_counts
            ],
        }
        if self.minimum_single_path_interventions:
            record["minimum_single_path_interventions"] = (
                self.minimum_single_path_interventions
            )
            record["intervention_axis"] = (
                "exact_parent_relative_changed_json_path_count"
            )
        if self.minimum_disjoint_parent_patch_pairs:
            record["minimum_disjoint_parent_patch_pairs"] = (
                self.minimum_disjoint_parent_patch_pairs
            )
            record["offspring_opportunity_axis"] = (
                "pairwise_disjoint_parent_relative_patch_pairs"
            )
        if self.feasibility_witness_option_identity_sha256s:
            record["feasibility_witness_option_identity_sha256s"] = list(
                self.feasibility_witness_option_identity_sha256s
            )
            record["feasibility_witness_semantics"] = (
                "current_finite_contract_exact_joint_count_and_structural_witness"
            )
        return record

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "slice_sha256": self.slice_sha256}

    def target_count(self, kind: SearchArmKind, arm_id: str) -> int:
        if type(kind) is not SearchArmKind:
            raise TypeError("kind must be exact SearchArmKind")
        _require_token(arm_id, name="arm_id")
        values = (
            self.source_target_counts
            if kind is SearchArmKind.SOURCE
            else self.operator_target_counts
        )
        try:
            return dict(values)[arm_id]
        except KeyError as error:
            raise ValueError("arm is absent from this allocation slice") from error

    def to_contract(
        self,
        *,
        campaign_generation: int,
    ) -> ContextualPortfolioAllocationContract:
        if type(campaign_generation) is not int or campaign_generation <= 0:
            raise ValueError("campaign_generation must be positive")
        return ContextualPortfolioAllocationContract(
            campaign_scope_sha256=self.campaign_scope_sha256,
            query_sha256=self.query_sha256,
            decision_sha256=self.decision_sha256,
            campaign_generation=campaign_generation,
            controller_wave_index=self.wave_index,
            phase_id=self.phase.value,
            slice_id=self.slice_id,
            evaluation_slots=self.evaluation_slots,
            source_target_counts=self.source_target_counts,
            operator_target_counts=self.operator_target_counts,
            minimum_single_path_interventions=(
                self.minimum_single_path_interventions
            ),
            minimum_disjoint_parent_patch_pairs=(
                self.minimum_disjoint_parent_patch_pairs
            ),
            feasibility_witness_option_identity_sha256s=(
                self.feasibility_witness_option_identity_sha256s
            ),
        )


@dataclass(frozen=True, slots=True)
class ContextualSearchStageAllocation:
    """Authenticated decomposition of one decision across concurrent parents."""

    decision: ContextualSearchDecision
    slices: tuple[ContextualPortfolioAllocationSlice, ...]
    allocation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.decision) is not ContextualSearchDecision:
            raise TypeError("decision must be exact ContextualSearchDecision")
        self.decision.__post_init__()
        if (
            type(self.slices) is not tuple
            or not self.slices
            or any(
                type(value) is not ContextualPortfolioAllocationSlice
                for value in self.slices
            )
        ):
            raise ValueError("slices must contain exact allocation slices")
        for value in self.slices:
            value.__post_init__()
            if (
                value.campaign_scope_sha256 != self.decision.query.campaign_scope_sha256
                or value.query_sha256 != self.decision.query.query_sha256
                or value.decision_sha256 != self.decision.decision_sha256
                or value.wave_index != self.decision.query.wave_index
                or value.phase is not self.decision.phase
            ):
                raise ValueError("allocation slice differs from its decision")
        if tuple(value.slice_id for value in self.slices) != tuple(
            sorted({value.slice_id for value in self.slices})
        ):
            raise ValueError("allocation slices must be unique and canonical")
        if self.decision.joint_capability_selection:
            selected_vectors = tuple(
                capability.resolve_vector(vector_sha256)
                for capability, (slice_id, vector_sha256) in zip(
                    self.decision.query.joint_count_capabilities,
                    self.decision.joint_capability_selection,
                    strict=True,
                )
                if capability.slice_id == slice_id
            )
            if len(selected_vectors) != len(self.slices) or any(
                allocation.feasibility_witness_option_identity_sha256s
                != vector.feasibility_witness_option_identity_sha256s
                for allocation, vector in zip(
                    self.slices,
                    selected_vectors,
                    strict=True,
                )
            ):
                raise ValueError(
                    "allocation slices differ from their selected joint witnesses"
                )
        elif any(
            value.feasibility_witness_option_identity_sha256s
            for value in self.slices
        ):
            raise ValueError(
                "allocation slice carries a witness without joint capability selection"
            )
        if sum(value.evaluation_slots for value in self.slices) != (
            self.decision.query.real_evaluation_slots
        ):
            raise ValueError("allocation slices do not cover the stage")
        for name, allocations in (
            ("source_target_counts", self.decision.source_allocations),
            ("operator_target_counts", self.decision.operator_allocations),
        ):
            aggregate = {
                allocation.arm_id: sum(
                    dict(getattr(value, name))[allocation.arm_id]
                    for value in self.slices
                )
                for allocation in allocations
            }
            expected = {
                allocation.arm_id: allocation.target_slots for allocation in allocations
            }
            if aggregate != expected:
                raise ValueError("allocation slices differ from stage targets")
        object.__setattr__(
            self,
            "allocation_sha256",
            _hash(_STAGE_ALLOCATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "decision_sha256": self.decision.decision_sha256,
            "slice_sha256s": [value.slice_sha256 for value in self.slices],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "decision": self.decision.to_record(),
            "slices": [value.to_record() for value in self.slices],
            "allocation_sha256": self.allocation_sha256,
        }


def _slice_arm_counts(
    allocations: tuple[ContextualArmAllocation, ...],
    slice_slots: tuple[int, ...],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    remaining = {value.arm_id: value.target_slots for value in allocations}
    remaining_slots = sum(slice_slots)
    rows: list[tuple[tuple[str, int], ...]] = []
    for index, slots in enumerate(slice_slots):
        if index == len(slice_slots) - 1:
            counts = dict(remaining)
        else:
            desired = {
                arm_id: count * slots / remaining_slots
                for arm_id, count in remaining.items()
            }
            counts = {arm_id: math.floor(value) for arm_id, value in desired.items()}
            for _ in range(slots - sum(counts.values())):
                arm_id = sorted(
                    remaining,
                    key=lambda value: (
                        -(desired[value] - counts[value]),
                        value,
                    ),
                )[0]
                counts[arm_id] += 1
        if sum(counts.values()) != slots or any(
            counts[arm_id] > remaining[arm_id] for arm_id in remaining
        ):
            raise AssertionError("deficit rounding produced an invalid slice")
        rows.append(tuple(sorted(counts.items())))
        remaining = {arm_id: remaining[arm_id] - counts[arm_id] for arm_id in remaining}
        remaining_slots -= slots
    if remaining_slots != 0 or any(remaining.values()):
        raise AssertionError("allocation slicing left an unassigned target")
    return tuple(rows)


def slice_contextual_search_decision(
    decision: ContextualSearchDecision,
    *,
    slice_ids: tuple[str, ...],
    evaluation_slots: tuple[int, ...],
) -> ContextualSearchStageAllocation:
    """Split stage-global marginals without workload or provider knowledge."""

    if type(decision) is not ContextualSearchDecision:
        raise TypeError("decision must be exact ContextualSearchDecision")
    decision.__post_init__()
    _canonical_tokens(slice_ids, name="slice_ids")
    if (
        type(evaluation_slots) is not tuple
        or len(evaluation_slots) != len(slice_ids)
        or any(type(value) is not int or value <= 0 for value in evaluation_slots)
    ):
        raise ValueError("evaluation_slots must positively cover every slice")
    if sum(evaluation_slots) != decision.query.real_evaluation_slots:
        raise ValueError("slice capacity differs from the controller query")
    if decision.joint_capability_selection:
        capabilities = decision.query.joint_count_capabilities
        if tuple(value.slice_id for value in capabilities) != slice_ids:
            raise ValueError("requested slices differ from joint capabilities")
        if tuple(value.evaluation_slots for value in capabilities) != evaluation_slots:
            raise ValueError("requested slice widths differ from joint capabilities")
        vectors = tuple(
            capability.resolve_vector(vector_sha256)
            for capability, (selected_slice_id, vector_sha256) in zip(
                capabilities,
                decision.joint_capability_selection,
                strict=True,
            )
            if capability.slice_id == selected_slice_id
        )
        if len(vectors) != len(capabilities):
            raise ValueError("joint capability selection has a foreign slice")
        source_rows = tuple(value.source_target_counts for value in vectors)
        operator_rows = tuple(value.operator_target_counts for value in vectors)
        minimum_single_path_rows = tuple(
            value.minimum_single_path_interventions for value in capabilities
        )
        minimum_disjoint_pair_rows = tuple(
            value.minimum_disjoint_parent_patch_pairs for value in capabilities
        )
        feasibility_witness_rows = tuple(
            value.feasibility_witness_option_identity_sha256s for value in vectors
        )
    else:
        source_rows = _slice_arm_counts(decision.source_allocations, evaluation_slots)
        operator_rows = _slice_arm_counts(
            decision.operator_allocations,
            evaluation_slots,
        )
        minimum_single_path_rows = tuple(0 for _ in slice_ids)
        minimum_disjoint_pair_rows = tuple(0 for _ in slice_ids)
        feasibility_witness_rows = tuple(() for _ in slice_ids)
    return ContextualSearchStageAllocation(
        decision=decision,
        slices=tuple(
            ContextualPortfolioAllocationSlice(
                campaign_scope_sha256=decision.query.campaign_scope_sha256,
                query_sha256=decision.query.query_sha256,
                decision_sha256=decision.decision_sha256,
                wave_index=decision.query.wave_index,
                phase=decision.phase,
                slice_id=slice_id,
                evaluation_slots=slots,
                source_target_counts=source_counts,
                operator_target_counts=operator_counts,
                minimum_single_path_interventions=(
                    minimum_single_path_interventions
                ),
                minimum_disjoint_parent_patch_pairs=(
                    minimum_disjoint_parent_patch_pairs
                ),
                feasibility_witness_option_identity_sha256s=(
                    feasibility_witness_option_identity_sha256s
                ),
            )
            for (
                slice_id,
                slots,
                source_counts,
                operator_counts,
                minimum_single_path_interventions,
                minimum_disjoint_parent_patch_pairs,
                feasibility_witness_option_identity_sha256s,
            ) in zip(
                slice_ids,
                evaluation_slots,
                source_rows,
                operator_rows,
                minimum_single_path_rows,
                minimum_disjoint_pair_rows,
                feasibility_witness_rows,
                strict=True,
            )
        ),
    )


def _phase(
    query: ContextualSearchQuery,
    snapshot: ContextualSearchSnapshot,
) -> SearchPhase:
    if query.remaining_portfolio_waves == 1:
        return SearchPhase.TERMINAL_CONVERSION
    if not snapshot.observation_sha256s:
        return SearchPhase.BASIN_ACQUISITION
    if query.composition_evidence_available:
        return SearchPhase.COMPOSITION
    return SearchPhase.BASIN_EXPANSION


def _score(posterior: ContextualArmPosterior, phase: SearchPhase) -> float:
    """Return the exploitation value in the single archive-return currency.

    Every evaluated infeasible or zero-yield action is already a zero in the
    fractional-Beta return posterior.  Persistence, descendant incidence,
    source distance, and allocation overlap remain available for diagnosis and
    future resolved-return construction; rewarding them separately would count
    the same causal outcome more than once.  Nonterminal uncertainty receives
    exactly one separately marked exploration slot in ``_allocate``.  Folding
    it into every proportional target as well would pay for information twice.
    """

    del phase
    return posterior.return_probability


def _allocate(
    posteriors: tuple[ContextualArmPosterior, ...],
    *,
    slots: int,
    phase: SearchPhase,
    incumbent_arm_id: str | None,
    incumbent_prior_mass: float,
    empirical_capability: ContextualArmCountCapability | None,
) -> tuple[ContextualArmAllocation, ...]:
    if not posteriors:
        raise ValueError("controller allocation requires arm posteriors")
    if type(slots) is not int or slots <= 0:
        raise ValueError("slots must be positive")
    _require_probability(incumbent_prior_mass, name="incumbent_prior_mass")
    if len(posteriors) > 1 and not 0.0 < incumbent_prior_mass < 1.0:
        raise ValueError(
            "multi-arm incumbent prior mass must lie strictly inside (0, 1)"
        )
    scores = {value.arm_id: _score(value, phase) for value in posteriors}
    counts = {value.arm_id: 0 for value in posteriors}
    if incumbent_arm_id is not None and incumbent_arm_id not in {
        value.arm_id for value in posteriors
    }:
        raise ValueError("incumbent arm is absent from controller posteriors")
    incumbent_prior = {
        value.arm_id: (
            1.0 / len(posteriors)
            if incumbent_arm_id is None
            else 1.0
            if len(posteriors) == 1
            else incumbent_prior_mass
            if value.arm_id == incumbent_arm_id
            else (1.0 - incumbent_prior_mass) / (len(posteriors) - 1)
        )
        for value in posteriors
    }
    if phase is SearchPhase.BASIN_ACQUISITION and incumbent_arm_id is not None:
        leader_id = incumbent_arm_id
    else:
        leader_id = sorted(
            posteriors,
            key=lambda value: (
                -scores[value.arm_id],
                value.observation_count,
                value.arm_id,
            ),
        )[0].arm_id
    exploration_id: str | None = None
    if phase is not SearchPhase.TERMINAL_CONVERSION and len(posteriors) > 1:
        exploration_id = sorted(
            (value for value in posteriors if value.arm_id != leader_id),
            key=lambda value: (
                -value.return_uncertainty,
                value.observation_count,
                value.arm_id,
            ),
        )[0].arm_id
        counts[exploration_id] += 1

    remaining = slots - sum(counts.values())
    cold_start = phase is SearchPhase.BASIN_ACQUISITION and all(
        value.observation_count == 0 for value in posteriors
    )
    if cold_start:
        probabilities = incumbent_prior
    else:
        # Beta(1,1) is already the explicit cold-start prior.  A second
        # temperature or hand-sized prior mixture would count prior mass twice.
        total_score = sum(scores.values())
        if total_score <= 0.0:  # Defensive only: fractional-Beta means are > 0.
            probabilities = incumbent_prior
        else:
            probabilities = {
                value.arm_id: scores[value.arm_id] / total_score
                for value in posteriors
            }
    desired = {
        value.arm_id: slots * probabilities[value.arm_id] for value in posteriors
    }
    # Fill against the full-generation target after reserving exploration;
    # otherwise the forced slot would be counted twice for uncertain arms.
    for _ in range(remaining):
        chosen = sorted(
            posteriors,
            key=lambda value: (
                -(desired[value.arm_id] - counts[value.arm_id]),
                -scores[value.arm_id],
                value.arm_id,
            ),
        )[0]
        counts[chosen.arm_id] += 1
    unconstrained_counts = dict(counts)
    capability_projected = False
    if empirical_capability is not None:
        empirical_capability.__post_init__()
        arm_ids = tuple(value.arm_id for value in posteriors)
        if (
            empirical_capability.arm_ids != arm_ids
            or empirical_capability.evaluation_slots != slots
        ):
            raise ValueError("empirical capability differs from allocation arms")
        candidates = []
        for vector in empirical_capability.feasible_count_vectors:
            candidate = dict(vector)
            if exploration_id is not None and candidate[exploration_id] == 0:
                continue
            candidates.append(candidate)
        if candidates:
            counts = min(
                candidates,
                key=lambda candidate: (
                    sum(
                        abs(candidate[arm_id] - unconstrained_counts[arm_id])
                        for arm_id in arm_ids
                    ),
                    -sum(scores[arm_id] * candidate[arm_id] for arm_id in arm_ids),
                    tuple(candidate[arm_id] for arm_id in arm_ids),
                ),
            )
            capability_projected = counts != unconstrained_counts
    return tuple(
        ContextualArmAllocation(
            kind=value.kind,
            arm_id=value.arm_id,
            target_slots=counts[value.arm_id],
            score=float(scores[value.arm_id]),
            allocation_probability=float(probabilities[value.arm_id]),
            exploration_slot=value.arm_id == exploration_id,
            unconstrained_target_slots=unconstrained_counts[value.arm_id],
            empirical_capability_projected=capability_projected,
        )
        for value in posteriors
    )


def _project_joint_capability_product(
    source_allocations: tuple[ContextualArmAllocation, ...],
    operator_allocations: tuple[ContextualArmAllocation, ...],
    capabilities: tuple[ContextualLaneJointCountCapability, ...],
) -> tuple[
    tuple[ContextualArmAllocation, ...],
    tuple[ContextualArmAllocation, ...],
    tuple[tuple[str, str], ...],
]:
    """Project independent marginals onto exact current lane capabilities."""

    if not capabilities:
        return source_allocations, operator_allocations, ()
    for value in capabilities:
        value.__post_init__()
    source_ids = tuple(value.arm_id for value in source_allocations)
    operator_ids = tuple(value.arm_id for value in operator_allocations)
    if any(
        value.source_arm_ids != source_ids or value.operator_arm_ids != operator_ids
        for value in capabilities
    ):
        raise ValueError("joint capabilities differ from controller allocations")
    preferred_source = {
        value.arm_id: value.target_slots for value in source_allocations
    }
    preferred_operator = {
        value.arm_id: value.target_slots for value in operator_allocations
    }
    source_scores = {value.arm_id: value.score for value in source_allocations}
    operator_scores = {value.arm_id: value.score for value in operator_allocations}
    source_exploration = {
        value.arm_id for value in source_allocations if value.exploration_slot
    }
    operator_exploration = {
        value.arm_id for value in operator_allocations if value.exploration_slot
    }
    candidates: list[
        tuple[
            tuple[object, ...],
            tuple[object, ...],
            dict[str, int],
            dict[str, int],
        ]
    ] = []
    for vectors in product(*(value.feasible_vectors for value in capabilities)):
        source_counts = {
            arm_id: sum(
                dict(vector.source_target_counts)[arm_id] for vector in vectors
            )
            for arm_id in source_ids
        }
        operator_counts = {
            arm_id: sum(
                dict(vector.operator_target_counts)[arm_id] for vector in vectors
            )
            for arm_id in operator_ids
        }
        source_exploration_loss = sum(
            source_counts[value] == 0 for value in source_exploration
        )
        operator_exploration_loss = sum(
            operator_counts[value] == 0 for value in operator_exploration
        )
        source_l1 = sum(
            abs(source_counts[value] - preferred_source[value]) for value in source_ids
        )
        operator_l1 = sum(
            abs(operator_counts[value] - preferred_operator[value])
            for value in operator_ids
        )
        posterior_score = sum(
            source_scores[value] * source_counts[value] for value in source_ids
        ) + sum(
            operator_scores[value] * operator_counts[value] for value in operator_ids
        )
        key: tuple[object, ...] = (
            source_exploration_loss + operator_exploration_loss,
            source_exploration_loss,
            operator_exploration_loss,
            source_l1 + operator_l1,
            source_l1,
            operator_l1,
            -posterior_score,
            tuple(value.vector_sha256 for value in vectors),
        )
        candidates.append((key, vectors, source_counts, operator_counts))
    if not candidates:  # pragma: no cover - lane capability constructors close this.
        raise ValueError("joint lane capabilities have no feasible product")
    _, vectors, selected_source, selected_operator = min(
        candidates,
        key=lambda value: value[0],
    )

    def projected(
        allocations: tuple[ContextualArmAllocation, ...],
        selected: dict[str, int],
    ) -> tuple[ContextualArmAllocation, ...]:
        requested_exploration = tuple(
            value.arm_id for value in allocations if value.exploration_slot
        )
        realized_exploration: str | None = None
        if requested_exploration:
            requested = requested_exploration[0]
            if selected[requested] > 0:
                realized_exploration = requested
            else:
                eligible = tuple(
                    value for value in allocations if selected[value.arm_id] > 0
                )
                if not eligible:  # pragma: no cover - selected counts sum positive.
                    raise AssertionError("joint projection selected no realized arm")
                # Infer the exploitation leader from the unconstrained dose, then
                # move the information marker to the least-entitled realizable
                # challenger.  This is deterministic and consumes no workload,
                # model, provider, or outcome identifier.
                leader = sorted(
                    allocations,
                    key=lambda value: (
                        -value.unconstrained_target_slots,
                        -value.allocation_probability,
                        -value.score,
                        value.arm_id,
                    ),
                )[0]
                challengers = tuple(
                    value for value in eligible if value.arm_id != leader.arm_id
                )
                realized_exploration = sorted(
                    challengers if challengers else eligible,
                    key=lambda value: (
                        value.unconstrained_target_slots,
                        value.allocation_probability,
                        value.score,
                        value.arm_id,
                    ),
                )[0].arm_id
        return tuple(
            replace(
                value,
                target_slots=selected[value.arm_id],
                exploration_slot=value.arm_id == realized_exploration,
                prospective_joint_capability_projected=(
                    value.prospective_joint_capability_projected
                    or selected[value.arm_id] != value.target_slots
                    or value.exploration_slot
                    != (value.arm_id == realized_exploration)
                ),
                prospective_joint_exploration_projected=(
                    value.exploration_slot
                    != (value.arm_id == realized_exploration)
                ),
            )
            for value in allocations
        )

    return (
        projected(source_allocations, selected_source),
        projected(operator_allocations, selected_operator),
        tuple(
            (capability.slice_id, vector.vector_sha256)
            for capability, vector in zip(capabilities, vectors, strict=True)
        ),
    )


@dataclass(frozen=True, slots=True)
class PhaseAwareContextualSearchController:
    """Compute one prior-only, phase-aware source/operator decision."""

    def decide(
        self,
        query: ContextualSearchQuery,
        snapshot: ContextualSearchSnapshot,
    ) -> ContextualSearchDecision:
        if type(query) is not ContextualSearchQuery:
            raise TypeError("query must be exact ContextualSearchQuery")
        if type(snapshot) is not ContextualSearchSnapshot:
            raise TypeError("snapshot must be exact ContextualSearchSnapshot")
        query.__post_init__()
        snapshot.__post_init__()
        phase = _phase(query, snapshot)
        source_allocations = _allocate(
            snapshot.source_posteriors,
            slots=query.real_evaluation_slots,
            phase=phase,
            incumbent_arm_id=query.incumbent_source_id,
            incumbent_prior_mass=0.50,
            empirical_capability=query.source_count_capability,
        )
        operator_allocations = _allocate(
            snapshot.operator_posteriors,
            slots=query.real_evaluation_slots,
            phase=phase,
            incumbent_arm_id=query.incumbent_operator_id,
            incumbent_prior_mass=0.75,
            empirical_capability=query.operator_count_capability,
        )
        (
            source_allocations,
            operator_allocations,
            joint_capability_selection,
        ) = _project_joint_capability_product(
            source_allocations,
            operator_allocations,
            query.joint_count_capabilities,
        )
        return ContextualSearchDecision(
            query=query,
            snapshot=snapshot,
            phase=phase,
            source_allocations=source_allocations,
            operator_allocations=operator_allocations,
            joint_capability_selection=joint_capability_selection,
        )


__all__ = [
    "CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256",
    "CONTEXTUAL_SEARCH_CONTROLLER_ID",
    "CONTEXTUAL_SEARCH_CONTROLLER_VERSION",
    "ContextualArmAllocation",
    "ContextualArmPosterior",
    "ContextualPortfolioAllocationSlice",
    "ContextualSearchCompletionAudit",
    "ContextualSearchDecision",
    "ContextualSearchDelayedCredit",
    "ContextualSearchLedger",
    "ContextualSearchObservation",
    "ContextualSearchQuery",
    "ContextualSearchSnapshot",
    "ContextualSearchStageAllocation",
    "PhaseAwareContextualSearchController",
    "SearchArmKind",
    "SearchPhase",
    "audit_completed_contextual_search_ledger",
    "slice_contextual_search_decision",
]

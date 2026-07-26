"""Domain-neutral operational-tie allocation ports for allocator v3.

Allocator v2 intentionally remains an exact-binary64 policy.  This parallel
port makes a benchmark's *decision resolution* explicit: scores whose distance
from the raw maximum is no larger than an identified, benchmark-supplied gap
form the operational top set.  A treatment-blind, task-keyed public SHA-256
rank then makes selection replayable without depending on input order,
concurrent completion order, provider data, forecasts, or outcomes.

The allocation-unit key is supplied by the experiment protocol.  Cross-arm
commit ports enforce that every treatment execution in one allocation wave
uses the same key, seed, resolution, and allocator configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_allocation import (
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
    PortfolioAllocationScore,
)
from agent_evolve.ports.action_allocation_frame import (
    FrameActionAllocationRequest,
    FrameActionPortfolioDecision,
    allocation_score_multiset_sha256,
    validate_frame_action_portfolio_decision,
)
from agent_evolve.ports.action_forecast import ResolvedActionForecast


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_CANDIDATE_LABEL = re.compile(r"^row_[0-9]{8}$")
_MAX_ALLOCATION_UNIT_KEY_BYTES = 512
_UINT64_MAX = (1 << 64) - 1

GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_ID = "greedy_risk_diversity"
GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_VERSION = 3
GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:greedy-risk-diversity:v3:"
    b"v2-p10-p50-p90-risk-and-attainable-family-diversity-score;"
    b"operational-top-is-raw-max-minus-candidate-inclusive-less-than-or-equal-"
    b"benchmark-identified-maximum-indistinguishable-gap;"
    b"rank=sha256(public-uint64-seed,common-allocation-unit-key,step,"
    b"option-identity);select-smallest-rank;"
    b"seed-sampling-law-and-provenance-bound;"
    b"fixed-seed-is-point-mass-with-no-reference-weight;"
    b"uniform-uint64-has-only-random-oracle-prior-reference;"
    b"modes=public-hash-rank-or-fail-closed"
).hexdigest()

_RESOLUTION_DOMAIN = b"agent-evolve:allocation-score-resolution:v1\x00"
_SELECTION_DOMAIN = b"agent-evolve:allocation-v3-selection-binding:v1\x00"
_PUBLIC_RANK_DOMAIN = b"agent-evolve:allocation-v3-public-rank:v1\x00"
_CONFIGURATION_DOMAIN = b"agent-evolve:greedy-risk-diversity-v3-config:v1\x00"
_REQUEST_DOMAIN = b"agent-evolve:operational-frame-allocation-request:v1\x00"
_CANDIDATE_DOMAIN = b"agent-evolve:allocation-v3-candidate:v1\x00"
_STEP_DOMAIN = b"agent-evolve:allocation-v3-step-audit:v1\x00"
_AUDIT_DOMAIN = b"agent-evolve:allocation-v3-audit:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:operational-frame-allocation-result:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")
    return value


def _token(value: object, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")
    return value


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big", signed=False) + value


class AllocationV3TieMode(str, Enum):
    """Prospectively selected handling of an operational top set."""

    PUBLIC_HASH_RANK = "public_hash_rank"
    FAIL_CLOSED = "fail_closed"


class AllocationV3SeedSamplingLaw(str, Enum):
    """Prospectively declared provenance law for the public uint64 seed."""

    FIXED_PUBLIC = "fixed_public"
    UNIFORM_UINT64 = "uniform_uint64"


@dataclass(frozen=True, slots=True, eq=False)
class AllocationScoreResolutionBinding:
    """Identified benchmark declaration of decision-score resolution.

    The value is not a universal floating-point epsilon.  It is the maximum
    raw marginal-utility difference the benchmark declares scientifically
    indistinguishable for this allocation protocol.
    """

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    maximum_indistinguishable_score_gap: float

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        gap = _finite_float(
            self.maximum_indistinguishable_score_gap,
            "maximum_indistinguishable_score_gap",
        )
        if gap < 0.0:
            raise ValueError("maximum_indistinguishable_score_gap cannot be negative")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "maximum_indistinguishable_score_gap_hex": (
                self.maximum_indistinguishable_score_gap.hex()
            ),
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_RESOLUTION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is AllocationScoreResolutionBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class AllocationV3SelectionBinding:
    """Public treatment-blind task key and mode for operational ties."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    mode: AllocationV3TieMode
    seed_sampling_law: AllocationV3SeedSamplingLaw
    seed_provenance_sha256: str
    public_seed: int
    allocation_unit_key: str

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.mode) is not AllocationV3TieMode:
            raise TypeError("mode must be an exact AllocationV3TieMode")
        if type(self.seed_sampling_law) is not AllocationV3SeedSamplingLaw:
            raise TypeError(
                "seed_sampling_law must be an exact AllocationV3SeedSamplingLaw"
            )
        require_sha256(self.seed_provenance_sha256, "seed_provenance_sha256")
        if (
            type(self.public_seed) is not int
            or not 0 <= self.public_seed <= _UINT64_MAX
        ):
            raise ValueError("public_seed must be an exact unsigned 64-bit integer")
        if (
            type(self.allocation_unit_key) is not str
            or not self.allocation_unit_key
            or "\x00" in self.allocation_unit_key
        ):
            raise ValueError("allocation_unit_key must be non-empty and NUL-free")
        try:
            encoded = self.allocation_unit_key.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("allocation_unit_key must be strict UTF-8") from exc
        if len(encoded) > _MAX_ALLOCATION_UNIT_KEY_BYTES:
            raise ValueError("allocation_unit_key exceeds its byte limit")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "mode": self.mode.value,
            "seed_sampling_law": self.seed_sampling_law.value,
            "seed_provenance_sha256": self.seed_provenance_sha256,
            "public_seed_uint64": self.public_seed,
            "allocation_unit_key": self.allocation_unit_key,
            "rank_inputs_excluded": [
                "treatment_identity",
                "forecast_values",
                "provider_identity",
                "completion_order",
                "outcomes",
            ],
            "selection_probability_claim": (
                "point_mass_after_fixed_public_seed_no_reference_weight"
                if self.seed_sampling_law
                is AllocationV3SeedSamplingLaw.FIXED_PUBLIC
                else "random_oracle_prior_reference_not_conditional_propensity"
            ),
            "seed_provenance_boundary": {
                "digest_alone_proves_pre_forecast_preregistration": False,
                "chronology_requires_external_durable_release": True,
            },
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_SELECTION_DOMAIN, self._unsigned_record())

    def public_rank_sha256(self, *, step: int, option_identity_sha256: str) -> str:
        """Return the public counter-based SHA rank for one step/option.

        The signature deliberately has no treatment, forecast, provider,
        completion, or outcome argument.
        """

        self.__post_init__()
        if type(step) is not int or step <= 0:
            raise ValueError("step must be a positive exact integer")
        require_sha256(option_identity_sha256, "option_identity_sha256")
        digest = hashlib.sha256()
        digest.update(_PUBLIC_RANK_DOMAIN)
        digest.update(self.public_seed.to_bytes(8, "big", signed=False))
        digest.update(_frame(self.allocation_unit_key.encode("utf-8")))
        digest.update(step.to_bytes(8, "big", signed=False))
        digest.update(bytes.fromhex(option_identity_sha256))
        return digest.hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is AllocationV3SelectionBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class OperationalFrameActionAllocationRequest:
    """Exact v3 configuration composed around an authenticated frame request."""

    allocation: FrameActionAllocationRequest = field(repr=False, compare=False)
    risk_aversion: float
    diversity_weight: float
    score_resolution: AllocationScoreResolutionBinding
    tie_selection: AllocationV3SelectionBinding

    def __post_init__(self) -> None:
        if type(self.allocation) is not FrameActionAllocationRequest:
            raise TypeError("allocation must be an exact frame allocation request")
        self.allocation.__post_init__()
        for name in ("risk_aversion", "diversity_weight"):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.score_resolution) is not AllocationScoreResolutionBinding:
            raise TypeError("score_resolution must be an exact identified binding")
        self.score_resolution.__post_init__()
        if type(self.tie_selection) is not AllocationV3SelectionBinding:
            raise TypeError("tie_selection must be an exact identified binding")
        self.tie_selection.__post_init__()

    def _configuration_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "allocator_policy_id": GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_ID,
            "allocator_policy_version": GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_VERSION,
            "allocator_definition_sha256": (
                GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_DEFINITION_SHA256
            ),
            "risk_aversion_hex": self.risk_aversion.hex(),
            "diversity_weight_hex": self.diversity_weight.hex(),
            "score_resolution": self.score_resolution.to_record(),
            "tie_selection": self.tie_selection.to_record(),
        }

    @property
    def allocator_configuration_sha256(self) -> str:
        return _hash(_CONFIGURATION_DOMAIN, self._configuration_record())

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "base_allocation_request_sha256": self.allocation.request_sha256,
            "frame_receipt_sha256": self.allocation.frame.receipt_sha256,
            "source_forecast_receipt_sha256": (
                self.allocation.frame.source_forecast_receipt_sha256
            ),
            "eligible_options_sha256": self.allocation.eligible_options_sha256,
            "portfolio_size": self.allocation.portfolio_size,
            "allocator_configuration": self._configuration_record(),
            "allocator_configuration_sha256": (
                self.allocator_configuration_sha256
            ),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is OperationalFrameActionAllocationRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class AllocationV3Candidate:
    """One complete marginal-score row and its public task-keyed rank."""

    candidate_label: str
    option_identity_sha256: str
    score: PortfolioAllocationScore
    marginal_total_utility: float
    public_rank_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.candidate_label) is not str
            or _CANDIDATE_LABEL.fullmatch(self.candidate_label) is None
        ):
            raise ValueError("candidate_label must be an opaque global-row label")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.score) is not PortfolioAllocationScore:
            raise TypeError("score must be an exact PortfolioAllocationScore")
        self.score.__post_init__()
        primary = self.score.p50_utility - self.score.risk_penalty
        total = primary + self.score.diversity_reward
        if not math.isfinite(primary) or not math.isfinite(total):
            raise ValueError("candidate score composition became non-finite")
        if self.score.total_utility != total:
            raise ValueError(
                "candidate total must equal primary risk-adjusted utility "
                "plus diversity"
            )
        _finite_float(self.marginal_total_utility, "marginal_total_utility")
        require_sha256(self.public_rank_sha256, "public_rank_sha256")

    @property
    def primary_risk_adjusted_utility(self) -> float:
        """Risk-adjusted utility before any additive diversity composition."""

        return self.score.p50_utility - self.score.risk_penalty

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "candidate_label": self.candidate_label,
            "option_identity_sha256": self.option_identity_sha256,
            "score": self.score.to_record(),
            "primary_risk_adjusted_utility_hex": (
                self.primary_risk_adjusted_utility.hex()
            ),
            "diversity_reward_hex": self.score.diversity_reward.hex(),
            "marginal_total_utility_hex": self.marginal_total_utility.hex(),
            "public_rank_sha256": self.public_rank_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_CANDIDATE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is AllocationV3Candidate
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def allocation_v3_failure_codes(
    *,
    mode: AllocationV3TieMode,
    operational_top_count: int,
) -> tuple[str, ...]:
    if type(mode) is not AllocationV3TieMode:
        raise TypeError("mode must be an exact AllocationV3TieMode")
    if type(operational_top_count) is not int or operational_top_count <= 0:
        raise ValueError("operational_top_count must be a positive exact integer")
    if mode is AllocationV3TieMode.FAIL_CLOSED and operational_top_count > 1:
        return ("operational_tie_fail_closed",)
    return ()


@dataclass(frozen=True, slots=True, eq=False)
class AllocationV3StepAudit:
    """Complete replayable candidate table and v3 selection facts for one step."""

    step: int
    candidates: tuple[AllocationV3Candidate, ...]
    distinct_finite_score_count: int
    raw_top_score: float
    raw_runner_gap: float
    raw_top_candidate_labels: tuple[str, ...]
    operational_top_candidate_labels: tuple[str, ...]
    selected_candidate_label: str
    selected_public_rank_sha256: str
    random_oracle_prior_weight_numerator: int | None
    random_oracle_prior_weight_denominator: int | None
    score_multiset_sha256: str
    failure_codes: tuple[str, ...]
    passes: bool

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("step must be a positive exact integer")
        if type(self.candidates) is not tuple or not self.candidates or any(
            type(value) is not AllocationV3Candidate for value in self.candidates
        ):
            raise ValueError("candidates must be a non-empty exact candidate tuple")
        for value in self.candidates:
            value.__post_init__()
        labels = tuple(value.candidate_label for value in self.candidates)
        if labels != tuple(sorted(labels)) or len(set(labels)) != len(labels):
            raise ValueError("candidate rows must be unique and label-canonical")
        identities = tuple(value.option_identity_sha256 for value in self.candidates)
        if len(set(identities)) != len(identities):
            raise ValueError("candidate option identities must be unique")
        ranks = tuple(value.public_rank_sha256 for value in self.candidates)
        if len(set(ranks)) != len(ranks):
            raise ValueError("public SHA ranks must be unique")
        if (
            type(self.distinct_finite_score_count) is not int
            or not 1 <= self.distinct_finite_score_count <= len(self.candidates)
        ):
            raise ValueError("distinct_finite_score_count is outside candidate count")
        _finite_float(self.raw_top_score, "raw_top_score")
        gap = _finite_float(self.raw_runner_gap, "raw_runner_gap")
        if gap < 0.0:
            raise ValueError("raw_runner_gap cannot be negative")
        label_set = set(labels)
        for name in (
            "raw_top_candidate_labels",
            "operational_top_candidate_labels",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or any(
                    type(value) is not str
                    or _CANDIDATE_LABEL.fullmatch(value) is None
                    for value in values
                )
                or values != tuple(sorted(set(values)))
                or not set(values).issubset(label_set)
            ):
                raise ValueError(f"{name} must be a non-empty canonical subset")
        if not set(self.raw_top_candidate_labels).issubset(
            self.operational_top_candidate_labels
        ):
            raise ValueError("raw top candidates must belong to the operational top")
        if self.selected_candidate_label not in self.operational_top_candidate_labels:
            raise ValueError("selected candidate must belong to the operational top")
        require_sha256(
            self.selected_public_rank_sha256,
            "selected_public_rank_sha256",
        )
        if (self.random_oracle_prior_weight_numerator is None) is not (
            self.random_oracle_prior_weight_denominator is None
        ):
            raise ValueError("random-oracle prior weight fields must be paired")
        if self.random_oracle_prior_weight_numerator is not None:
            if (
                type(self.random_oracle_prior_weight_numerator) is not int
                or self.random_oracle_prior_weight_numerator != 1
            ):
                raise ValueError(
                    "random-oracle prior weight numerator must be exactly one"
                )
            if (
                type(self.random_oracle_prior_weight_denominator) is not int
                or self.random_oracle_prior_weight_denominator
                != len(self.operational_top_candidate_labels)
            ):
                raise ValueError(
                    "random-oracle prior weight must be 1/|operational top|"
                )
        require_sha256(self.score_multiset_sha256, "score_multiset_sha256")
        if (
            type(self.failure_codes) is not tuple
            or any(
                type(value) is not str or _TOKEN.fullmatch(value) is None
                for value in self.failure_codes
            )
            or self.failure_codes != tuple(sorted(set(self.failure_codes)))
        ):
            raise ValueError("failure_codes must be unique canonical tokens")
        if type(self.passes) is not bool or self.passes is not (not self.failure_codes):
            raise ValueError("passes must be the inverse of failure_codes")

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def raw_top_tie_count(self) -> int:
        return len(self.raw_top_candidate_labels)

    @property
    def operational_top_count(self) -> int:
        return len(self.operational_top_candidate_labels)

    @property
    def random_oracle_prior_seed_law_reference_weight(self) -> str | None:
        if self.random_oracle_prior_weight_denominator is None:
            return None
        return f"1/{self.random_oracle_prior_weight_denominator}"

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "step": self.step,
            "candidate_count": self.candidate_count,
            "distinct_finite_score_count": self.distinct_finite_score_count,
            "candidates": [value.to_record() for value in self.candidates],
            "raw_top_score_hex": self.raw_top_score.hex(),
            "raw_runner_gap_hex": self.raw_runner_gap.hex(),
            "raw_top_tie_count": self.raw_top_tie_count,
            "raw_top_candidate_labels": list(self.raw_top_candidate_labels),
            "operational_top_count": self.operational_top_count,
            "operational_top_candidate_labels": list(
                self.operational_top_candidate_labels
            ),
            "selected_candidate_label": self.selected_candidate_label,
            "selected_public_rank_sha256": self.selected_public_rank_sha256,
            "random_oracle_prior_seed_law_reference_weight": (
                None
                if self.random_oracle_prior_weight_numerator is None
                else {
                    "numerator": self.random_oracle_prior_weight_numerator,
                    "denominator": self.random_oracle_prior_weight_denominator,
                    "ratio": (
                        self.random_oracle_prior_seed_law_reference_weight
                    ),
                    "interpretation": (
                        "prior_reference_under_uniform_uint64_random_oracle_model"
                    ),
                    "is_conditional_propensity": False,
                }
            ),
            "score_multiset_sha256": self.score_multiset_sha256,
            "failure_codes": list(self.failure_codes),
            "passes": self.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_STEP_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is AllocationV3StepAudit
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def _validate_step_against_policy(
    step: AllocationV3StepAudit,
    *,
    resolution: AllocationScoreResolutionBinding,
    selection: AllocationV3SelectionBinding,
) -> None:
    step.__post_init__()
    resolution.__post_init__()
    selection.__post_init__()
    for candidate in step.candidates:
        expected_rank = selection.public_rank_sha256(
            step=step.step,
            option_identity_sha256=candidate.option_identity_sha256,
        )
        if candidate.public_rank_sha256 != expected_rank:
            raise ValueError("candidate public rank differs from the bound task key")

    scores = tuple(value.marginal_total_utility for value in step.candidates)
    raw_top = max(scores)
    ranked = sorted(scores, reverse=True)
    raw_runner_gap = 0.0 if len(ranked) == 1 else ranked[0] - ranked[1]
    if not math.isfinite(raw_runner_gap):
        raise ValueError("raw runner-gap arithmetic became non-finite")
    raw_labels = tuple(
        value.candidate_label
        for value in step.candidates
        if value.marginal_total_utility == raw_top
    )
    operational = tuple(
        value
        for value in step.candidates
        if raw_top - value.marginal_total_utility
        <= resolution.maximum_indistinguishable_score_gap
    )
    operational_labels = tuple(value.candidate_label for value in operational)
    selected = min(
        operational,
        key=lambda value: (value.public_rank_sha256, value.candidate_label),
    )
    expected_failures = allocation_v3_failure_codes(
        mode=selection.mode,
        operational_top_count=len(operational),
    )
    if selection.seed_sampling_law is AllocationV3SeedSamplingLaw.FIXED_PUBLIC:
        if (
            step.random_oracle_prior_weight_numerator is not None
            or step.random_oracle_prior_weight_denominator is not None
        ):
            raise ValueError("fixed public seeds cannot carry a reference weight")
    elif (
        step.random_oracle_prior_weight_numerator != 1
        or step.random_oracle_prior_weight_denominator != len(operational)
    ):
        raise ValueError(
            "uniform uint64 seeds require the random-oracle prior reference"
        )
    if step.distinct_finite_score_count != len(set(scores)):
        raise ValueError("distinct score count differs from the candidate table")
    if step.raw_top_score != raw_top:
        raise ValueError("raw top score differs from the candidate table")
    if step.raw_runner_gap != raw_runner_gap:
        raise ValueError("raw runner gap differs from the candidate table")
    if step.raw_top_candidate_labels != raw_labels:
        raise ValueError("raw top labels differ from exact binary64 ties")
    if step.operational_top_candidate_labels != operational_labels:
        raise ValueError("operational top is not the inclusive raw-max-relative set")
    if (
        step.selected_candidate_label != selected.candidate_label
        or step.selected_public_rank_sha256 != selected.public_rank_sha256
    ):
        raise ValueError("selected candidate is not the smallest public SHA rank")
    if step.score_multiset_sha256 != allocation_score_multiset_sha256(scores):
        raise ValueError("score multiset receipt differs from the candidate table")
    if step.failure_codes != expected_failures:
        raise ValueError("step failures differ from the v3 tie mode")


@dataclass(frozen=True, slots=True, eq=False)
class OperationalFrameActionAllocationAudit:
    operational_request_sha256: str
    base_allocation_request_sha256: str
    decision_receipt_sha256: str
    frame_receipt_sha256: str
    score_resolution: AllocationScoreResolutionBinding
    tie_selection: AllocationV3SelectionBinding
    steps: tuple[AllocationV3StepAudit, ...]
    candidate_score_count: int
    passes: bool

    def __post_init__(self) -> None:
        for name in (
            "operational_request_sha256",
            "base_allocation_request_sha256",
            "decision_receipt_sha256",
            "frame_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.score_resolution) is not AllocationScoreResolutionBinding:
            raise TypeError("score_resolution must be an exact binding")
        self.score_resolution.__post_init__()
        if type(self.tie_selection) is not AllocationV3SelectionBinding:
            raise TypeError("tie_selection must be an exact binding")
        self.tie_selection.__post_init__()
        if type(self.steps) is not tuple or not self.steps or any(
            type(value) is not AllocationV3StepAudit for value in self.steps
        ):
            raise ValueError("steps must be a non-empty exact audit tuple")
        if tuple(value.step for value in self.steps) != tuple(
            range(1, len(self.steps) + 1)
        ):
            raise ValueError("audit steps must be contiguous and ordered")
        for value in self.steps:
            _validate_step_against_policy(
                value,
                resolution=self.score_resolution,
                selection=self.tie_selection,
            )
        expected_count = sum(value.candidate_count for value in self.steps)
        if (
            type(self.candidate_score_count) is not int
            or self.candidate_score_count != expected_count
        ):
            raise ValueError("candidate_score_count differs from the complete tables")
        if type(self.passes) is not bool or self.passes is not all(
            value.passes for value in self.steps
        ):
            raise ValueError("audit passes differs from its step decisions")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "operational_request_sha256": self.operational_request_sha256,
            "base_allocation_request_sha256": self.base_allocation_request_sha256,
            "decision_receipt_sha256": self.decision_receipt_sha256,
            "frame_receipt_sha256": self.frame_receipt_sha256,
            "score_resolution": self.score_resolution.to_record(),
            "tie_selection": self.tie_selection.to_record(),
            "steps": [value.to_record() for value in self.steps],
            "candidate_score_count": self.candidate_score_count,
            "passes": self.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_AUDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is OperationalFrameActionAllocationAudit
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class OperationalFrameActionAllocationResult:
    decision: FrameActionPortfolioDecision
    audit: OperationalFrameActionAllocationAudit

    def __post_init__(self) -> None:
        if type(self.decision) is not FrameActionPortfolioDecision:
            raise TypeError("decision must be an exact frame portfolio decision")
        self.decision.__post_init__()
        if type(self.audit) is not OperationalFrameActionAllocationAudit:
            raise TypeError("audit must be an exact operational allocation audit")
        self.audit.__post_init__()
        if self.audit.decision_receipt_sha256 != self.decision.receipt_sha256:
            raise ValueError("audit is bound to another decision")
        if self.audit.base_allocation_request_sha256 != (
            self.decision.allocation_request_sha256
        ):
            raise ValueError("audit and decision name different base requests")
        if self.audit.frame_receipt_sha256 != self.decision.frame_receipt_sha256:
            raise ValueError("audit and decision name different frames")
        if self.audit.candidate_score_count != self.decision.candidate_evaluations:
            raise ValueError("audit and decision candidate counts differ")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "decision_receipt_sha256": self.decision.receipt_sha256,
            "audit_receipt_sha256": self.audit.receipt_sha256,
            "authorized": self.audit.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_RESULT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is OperationalFrameActionAllocationResult
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def _recompute_candidate_score(
    request: OperationalFrameActionAllocationRequest,
    members: tuple[ResolvedActionForecast, ...],
    *,
    attainable_diversity: int,
) -> PortfolioAllocationScore:
    """Rerun the authenticated benchmark utility for validator independence."""

    base = request.allocation
    canonical_members = tuple(
        sorted(
            members,
            key=lambda value: (value.option_identity_sha256, value.option_id),
        )
    )
    utilities: dict[ForecastQuantile, float] = {}
    for quantile in (ForecastQuantile.P10, ForecastQuantile.P50, ForecastQuantile.P90):
        value = base.utility.utility(
            ForecastPortfolioUtilityInput(
                optimization_semantics=base.frame.request.optimization_semantics,
                parent_metric_values=base.frame.request.parent_metric_values,
                metric_scales=base.frame.request.metric_scales,
                members=canonical_members,
                quantile=quantile,
            )
        )
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("bound portfolio utility must return a finite float")
        utilities[quantile] = value
    p10 = utilities[ForecastQuantile.P10]
    p50 = utilities[ForecastQuantile.P50]
    p90 = utilities[ForecastQuantile.P90]
    downside = min(p10, p90)
    risk_penalty = request.risk_aversion * max(0.0, p50 - downside)
    diversity_reward = request.diversity_weight * (
        len({value.family for value in canonical_members}) / attainable_diversity
    )
    primary = p50 - risk_penalty
    total = primary + diversity_reward
    if not all(
        math.isfinite(value)
        for value in (risk_penalty, diversity_reward, primary, total)
    ):
        raise ValueError("recomputed allocator-v3 score became non-finite")
    return PortfolioAllocationScore(
        p10_utility=p10,
        p50_utility=p50,
        p90_utility=p90,
        downside_utility=downside,
        risk_penalty=risk_penalty,
        diversity_reward=diversity_reward,
        total_utility=total,
    )


def validate_operational_frame_action_allocation_result(
    request: OperationalFrameActionAllocationRequest,
    result: OperationalFrameActionAllocationResult,
) -> None:
    """Validate every binding and rerun utility over all authenticated rows."""

    if type(request) is not OperationalFrameActionAllocationRequest:
        raise TypeError("request must be an exact operational frame request")
    request.__post_init__()
    if type(result) is not OperationalFrameActionAllocationResult:
        raise TypeError("result must be an exact operational frame result")
    result.__post_init__()
    validate_frame_action_portfolio_decision(request.allocation, result.decision)
    decision = result.decision
    audit = result.audit
    if (
        decision.allocator_policy_id != GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_ID
        or decision.allocator_policy_version
        != GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_VERSION
        or decision.allocator_definition_sha256
        != GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_DEFINITION_SHA256
        or decision.allocator_configuration_sha256
        != request.allocator_configuration_sha256
    ):
        raise ValueError("decision names another allocator-v3 policy/configuration")
    if (
        audit.operational_request_sha256 != request.request_sha256
        or audit.base_allocation_request_sha256 != request.allocation.request_sha256
        or audit.frame_receipt_sha256 != request.allocation.frame.receipt_sha256
        or audit.score_resolution != request.score_resolution
        or audit.tie_selection != request.tie_selection
    ):
        raise ValueError("operational audit is bound to another request")
    expected_counts = tuple(
        len(request.allocation.eligible_option_ids) - offset
        for offset in range(request.allocation.portfolio_size)
    )
    if tuple(value.candidate_count for value in audit.steps) != expected_counts:
        raise ValueError("v3 audit does not cover every greedy extension")

    identity_by_label = {
        f"row_{index:08d}": forecast.option_identity_sha256
        for index, forecast in zip(
            request.allocation.frame.global_row_indices,
            request.allocation.frame.forecasts,
            strict=True,
        )
    }
    forecast_by_identity = {
        value.option_identity_sha256: value
        for value in request.allocation.frame.forecasts
        if value.option_id in request.allocation.eligible_option_ids
    }
    remaining_identities = set(forecast_by_identity)
    selected_forecasts = []
    attainable_diversity = min(
        request.allocation.portfolio_size,
        len({value.family for value in forecast_by_identity.values()}),
    )
    previous_total = 0.0
    for member, step in zip(decision.members, audit.steps, strict=True):
        observed_identities = {
            value.option_identity_sha256 for value in step.candidates
        }
        if observed_identities != remaining_identities:
            raise ValueError("candidate table differs from the remaining eligible set")
        for candidate in step.candidates:
            if identity_by_label.get(candidate.candidate_label) != (
                candidate.option_identity_sha256
            ):
                raise ValueError("candidate label differs from its frame identity")
            forecast = forecast_by_identity[candidate.option_identity_sha256]
            expected_score = _recompute_candidate_score(
                request,
                tuple((*selected_forecasts, forecast)),
                attainable_diversity=attainable_diversity,
            )
            if candidate.score != expected_score:
                raise ValueError(
                    "candidate score differs from recomputed benchmark utility"
                )
            if candidate.marginal_total_utility != (
                expected_score.total_utility - previous_total
            ):
                raise ValueError("candidate marginal differs from its complete score")
        selected = next(
            value
            for value in step.candidates
            if value.candidate_label == step.selected_candidate_label
        )
        if (
            identity_by_label[step.selected_candidate_label]
            != member.option_identity_sha256
            or selected.option_identity_sha256 != member.option_identity_sha256
            or selected.score != member.greedy_step_score
            or selected.marginal_total_utility != member.marginal_total_utility
        ):
            raise ValueError("decision member differs from its audited v3 winner")
        selected_forecasts.append(forecast_by_identity[member.option_identity_sha256])
        remaining_identities.remove(member.option_identity_sha256)
        previous_total = selected.score.total_utility


__all__ = [
    "AllocationScoreResolutionBinding",
    "AllocationV3Candidate",
    "AllocationV3SeedSamplingLaw",
    "AllocationV3SelectionBinding",
    "AllocationV3StepAudit",
    "AllocationV3TieMode",
    "GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_DEFINITION_SHA256",
    "GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_ID",
    "GREEDY_RISK_DIVERSITY_V3_ALLOCATOR_VERSION",
    "OperationalFrameActionAllocationAudit",
    "OperationalFrameActionAllocationRequest",
    "OperationalFrameActionAllocationResult",
    "allocation_v3_failure_codes",
    "validate_operational_frame_action_allocation_result",
]

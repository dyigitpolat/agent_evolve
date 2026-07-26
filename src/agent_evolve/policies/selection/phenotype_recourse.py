"""Objective-blind phenotype identity and bounded evaluation recourse.

Candidate occurrences are causal units: two independently assigned trials remain
distinct even when they materialize the same evaluation phenotype.  Physical
evaluation identity is a separate, injected policy.  This module joins those two
views without importing an evaluator, reward, archive, model provider, engine, or
optimizer.

Recourse is deliberately narrow.  A pool is sealed before primary outcomes are
known, only successful *primary* collisions create credit, and successful
recourse outcomes can never create more credit.  Infrastructure/system failures
in the primary wave invalidate the block; model and candidate failures are valid
zero-yield causal outcomes.  The deterministic slot cap is::

    min(max_recourse,
        successful_primary_collision_credit,
        unique_budget_after_reservations_and_recombination_protection,
        eligible_presealed_pool_size)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import ClassVar, Protocol

from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    freeze_json,
    is_frozen_json_value,
    typed_json_sha256,
)


IDENTITY_POLICY_ID = "typed_configuration_phenotype"
IDENTITY_POLICY_VERSION = 1
RECOURSE_POLICY_ID = "bounded_collision_evaluation_recourse"
RECOURSE_POLICY_VERSION = 1

_IDENTITY_DOMAIN = b"agent-evolve:phenotype-identity:v1\x00"
_LEDGER_DOMAIN = b"agent-evolve:phenotype-occurrence-ledger:v1\x00"
_POOL_DOMAIN = b"agent-evolve:presealed-recourse-pool:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:evaluation-recourse-decision:v1\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _positive(value: int, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative(value: int, *, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class PhenotypeIdentity:
    """A policy-bound digest used to coalesce physical evaluations."""

    policy_id: str
    policy_version: int
    value_sha256: str

    def __post_init__(self) -> None:
        _token(self.policy_id, name="identity policy_id")
        _positive(self.policy_version, name="identity policy_version")
        require_sha256(self.value_sha256, "phenotype value_sha256")

    @property
    def identity_sha256(self) -> str:
        return _domain_hash(_IDENTITY_DOMAIN, self.to_trace_record())

    def to_trace_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "value_sha256": self.value_sha256,
        }


class PhenotypeIdentityPolicy(Protocol):
    """Injected boundary between a typed configuration and evaluator identity."""

    policy_id: str
    policy_version: int

    def identify(self, configuration: object) -> PhenotypeIdentity: ...


@dataclass(frozen=True, slots=True)
class TypedConfigurationPhenotypeIdentityPolicy:
    """Default identity: the exact, type-sensitive configuration tree."""

    policy_id: ClassVar[str] = IDENTITY_POLICY_ID
    policy_version: ClassVar[int] = IDENTITY_POLICY_VERSION

    def identify(self, configuration: object) -> PhenotypeIdentity:
        return PhenotypeIdentity(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            value_sha256=typed_json_sha256(freeze_json(configuration)),
        )


SemanticProjection = Callable[[FrozenJsonValue], object]


@dataclass(frozen=True, slots=True)
class SemanticProjectionPhenotypeIdentityPolicy:
    """Hash an injected deterministic semantic projection of a configuration.

    The projector receives an immutable typed-JSON tree and must return another
    typed-JSON value.  A problem adapter can therefore erase evaluator-inert
    syntax (aliases, ordering, decoded no-ops) without weakening the default
    type-sensitive identity used by domains that have no such equivalence.
    """

    policy_id: str
    policy_version: int
    projector: SemanticProjection

    def __post_init__(self) -> None:
        _token(self.policy_id, name="semantic identity policy_id")
        _positive(self.policy_version, name="semantic identity policy_version")
        if not callable(self.projector):
            raise TypeError("projector must be callable")

    def identify(self, configuration: object) -> PhenotypeIdentity:
        immutable_configuration = freeze_json(configuration)
        projected = freeze_json(self.projector(immutable_configuration))
        return PhenotypeIdentity(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            value_sha256=typed_json_sha256(projected),
        )


class EvaluationOccurrenceRole(str, Enum):
    PRIMARY = "primary"
    RECOURSE = "recourse"


class EvaluationOccurrenceStatus(str, Enum):
    """Terminal status with explicit causal and block-validity semantics."""

    SUCCESS = "success"
    CANDIDATE_FAILURE = "candidate_failure"
    MODEL_FAILURE = "model_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    SYSTEM_FAILURE = "system_failure"

    @property
    def invalidates_primary_block(self) -> bool:
        return self in (
            EvaluationOccurrenceStatus.INFRASTRUCTURE_FAILURE,
            EvaluationOccurrenceStatus.SYSTEM_FAILURE,
        )


@dataclass(frozen=True, slots=True)
class PhenotypeOccurrence:
    """One causal trial occurrence, independent of physical cache identity."""

    trial_id: OperatorInvocationId
    role: EvaluationOccurrenceRole
    status: EvaluationOccurrenceStatus
    candidate_id: CandidateId | None
    phenotype: PhenotypeIdentity | None

    def __post_init__(self) -> None:
        if type(self.trial_id) is not OperatorInvocationId:
            raise TypeError("trial_id must be an exact OperatorInvocationId")
        OperatorInvocationId.__post_init__(self.trial_id)
        if type(self.role) is not EvaluationOccurrenceRole:
            raise TypeError("role must be an EvaluationOccurrenceRole")
        if type(self.status) is not EvaluationOccurrenceStatus:
            raise TypeError("status must be an EvaluationOccurrenceStatus")
        if self.candidate_id is not None:
            if type(self.candidate_id) is not CandidateId:
                raise TypeError("candidate_id must be an exact CandidateId or None")
            CandidateId.__post_init__(self.candidate_id)
        if self.phenotype is not None:
            if type(self.phenotype) is not PhenotypeIdentity:
                raise TypeError("phenotype must be an exact PhenotypeIdentity or None")
            PhenotypeIdentity.__post_init__(self.phenotype)

        has_candidate = self.candidate_id is not None
        has_phenotype = self.phenotype is not None
        if has_candidate != has_phenotype:
            raise ValueError("candidate_id and phenotype must be present together")
        if self.status is EvaluationOccurrenceStatus.SUCCESS and not has_candidate:
            raise ValueError("success requires candidate and phenotype")
        if self.status is EvaluationOccurrenceStatus.MODEL_FAILURE and has_candidate:
            raise ValueError("model_failure cannot carry candidate or phenotype")
        if (
            self.role is EvaluationOccurrenceRole.RECOURSE
            and self.status is EvaluationOccurrenceStatus.MODEL_FAILURE
        ):
            raise ValueError("engine-owned recourse cannot have model_failure status")

    def to_trace_record(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id.value,
            "role": self.role.value,
            "status": self.status.value,
            "candidate_id": (
                None if self.candidate_id is None else self.candidate_id.value
            ),
            "phenotype_identity_sha256": (
                None if self.phenotype is None else self.phenotype.identity_sha256
            ),
        }


def _occurrence_key(occurrence: PhenotypeOccurrence) -> str:
    return occurrence.trial_id.value


@dataclass(frozen=True, slots=True)
class PhenotypeCluster:
    """All causal occurrences sharing one physical evaluation phenotype."""

    phenotype: PhenotypeIdentity
    occurrences: tuple[PhenotypeOccurrence, ...]

    def __post_init__(self) -> None:
        if type(self.phenotype) is not PhenotypeIdentity:
            raise TypeError("phenotype must be an exact PhenotypeIdentity")
        PhenotypeIdentity.__post_init__(self.phenotype)
        if type(self.occurrences) is not tuple or not self.occurrences:
            raise ValueError("cluster occurrences must be a non-empty exact tuple")
        if any(type(item) is not PhenotypeOccurrence for item in self.occurrences):
            raise TypeError("cluster must contain exact PhenotypeOccurrence values")
        for occurrence in self.occurrences:
            PhenotypeOccurrence.__post_init__(occurrence)
            if occurrence.phenotype != self.phenotype:
                raise ValueError("cluster contains a different phenotype")
        expected = tuple(sorted(self.occurrences, key=_occurrence_key))
        if self.occurrences != expected:
            raise ValueError("cluster occurrences must be canonically ordered")
        if len({item.trial_id for item in self.occurrences}) != len(self.occurrences):
            raise ValueError("cluster trial IDs must be unique")

    @property
    def successful_primary_occurrences(self) -> tuple[PhenotypeOccurrence, ...]:
        return tuple(
            item
            for item in self.occurrences
            if item.role is EvaluationOccurrenceRole.PRIMARY
            and item.status is EvaluationOccurrenceStatus.SUCCESS
        )

    @property
    def successful_primary_collision_credit(self) -> int:
        return max(0, len(self.successful_primary_occurrences) - 1)

    @property
    def zero_identity_contrast_pairs(
        self,
    ) -> tuple[tuple[OperatorInvocationId, OperatorInvocationId], ...]:
        """Successful primary pairs whose evaluation contrast is zero by identity."""

        return tuple(
            (left.trial_id, right.trial_id)
            for left, right in combinations(self.successful_primary_occurrences, 2)
        )

    def to_trace_record(self) -> dict[str, object]:
        return {
            "phenotype": self.phenotype.to_trace_record(),
            "phenotype_identity_sha256": self.phenotype.identity_sha256,
            "occurrences": [item.to_trace_record() for item in self.occurrences],
            "successful_primary_collision_credit": (
                self.successful_primary_collision_credit
            ),
            "zero_identity_contrast_pairs": [
                [left.value, right.value]
                for left, right in self.zero_identity_contrast_pairs
            ],
        }


@dataclass(frozen=True, slots=True, eq=False)
class PhenotypeOccurrenceLedger:
    """Canonical occurrence history and its phenotype clusters."""

    identity_policy_id: str
    identity_policy_version: int
    occurrences: tuple[PhenotypeOccurrence, ...]

    def __post_init__(self) -> None:
        _token(self.identity_policy_id, name="ledger identity_policy_id")
        _positive(
            self.identity_policy_version,
            name="ledger identity_policy_version",
        )
        if type(self.occurrences) is not tuple or any(
            type(item) is not PhenotypeOccurrence for item in self.occurrences
        ):
            raise TypeError("occurrences must contain exact PhenotypeOccurrence values")
        for occurrence in self.occurrences:
            PhenotypeOccurrence.__post_init__(occurrence)
            phenotype = occurrence.phenotype
            if phenotype is not None and (
                phenotype.policy_id != self.identity_policy_id
                or phenotype.policy_version != self.identity_policy_version
            ):
                raise ValueError(
                    "occurrence phenotype uses a different identity policy"
                )
        expected = tuple(sorted(self.occurrences, key=_occurrence_key))
        if self.occurrences != expected:
            raise ValueError("occurrences must be canonically ordered by trial ID")
        if len({item.trial_id for item in self.occurrences}) != len(self.occurrences):
            raise ValueError("occurrence trial IDs must be unique")
        candidate_ids = tuple(
            item.candidate_id
            for item in self.occurrences
            if item.candidate_id is not None
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("occurrence candidate IDs must be unique")

    @classmethod
    def build(
        cls,
        occurrences: Sequence[PhenotypeOccurrence],
        *,
        identity_policy: PhenotypeIdentityPolicy,
    ) -> "PhenotypeOccurrenceLedger":
        if isinstance(occurrences, (str, bytes)):
            raise TypeError("occurrences must be a finite occurrence sequence")
        if not callable(getattr(identity_policy, "identify", None)):
            raise TypeError("identity_policy must implement identify")
        policy_id = getattr(identity_policy, "policy_id", None)
        policy_version = getattr(identity_policy, "policy_version", None)
        _token(policy_id, name="identity policy_id")
        _positive(policy_version, name="identity policy_version")
        values = tuple(occurrences)
        for value in values:
            if type(value) is not PhenotypeOccurrence:
                raise TypeError(
                    "occurrences must contain exact PhenotypeOccurrence values"
                )
        return cls(
            identity_policy_id=policy_id,
            identity_policy_version=policy_version,
            occurrences=tuple(sorted(values, key=_occurrence_key)),
        )

    @property
    def clusters(self) -> tuple[PhenotypeCluster, ...]:
        grouped: dict[str, list[PhenotypeOccurrence]] = {}
        identities: dict[str, PhenotypeIdentity] = {}
        for occurrence in self.occurrences:
            phenotype = occurrence.phenotype
            if phenotype is None:
                continue
            key = phenotype.identity_sha256
            identities[key] = phenotype
            grouped.setdefault(key, []).append(occurrence)
        return tuple(
            PhenotypeCluster(identities[key], tuple(grouped[key]))
            for key in sorted(grouped)
        )

    @property
    def successful_primary_collision_credit(self) -> int:
        return sum(
            cluster.successful_primary_collision_credit for cluster in self.clusters
        )

    @property
    def primary_occurrences(self) -> tuple[PhenotypeOccurrence, ...]:
        return tuple(
            item
            for item in self.occurrences
            if item.role is EvaluationOccurrenceRole.PRIMARY
        )

    @property
    def primary_phenotypes(self) -> frozenset[PhenotypeIdentity]:
        return frozenset(
            item.phenotype
            for item in self.primary_occurrences
            if item.phenotype is not None
        )

    @property
    def invalidating_primary_trial_ids(self) -> tuple[OperatorInvocationId, ...]:
        return tuple(
            item.trial_id
            for item in self.primary_occurrences
            if item.status.invalidates_primary_block
        )

    @property
    def invalidating_trial_ids(self) -> tuple[OperatorInvocationId, ...]:
        """Every infrastructure/system failure, including post-decision recourse."""

        return tuple(
            item.trial_id
            for item in self.occurrences
            if item.status.invalidates_primary_block
        )

    @property
    def primary_block_valid(self) -> bool:
        return not self.invalidating_primary_trial_ids

    @property
    def experiment_block_valid(self) -> bool:
        """False if infrastructure/system failure occurred at any wave stage."""

        return not self.invalidating_trial_ids

    @property
    def ignored_recourse_trial_ids(self) -> tuple[OperatorInvocationId, ...]:
        return tuple(
            item.trial_id
            for item in self.occurrences
            if item.role is EvaluationOccurrenceRole.RECOURSE
        )

    def is_zero_identity_contrast(
        self,
        left_trial_id: OperatorInvocationId,
        right_trial_id: OperatorInvocationId,
    ) -> bool:
        """Whether two successful primary trials map to one phenotype."""

        for value, name in (
            (left_trial_id, "left_trial_id"),
            (right_trial_id, "right_trial_id"),
        ):
            if type(value) is not OperatorInvocationId:
                raise TypeError(f"{name} must be an exact OperatorInvocationId")
        if left_trial_id == right_trial_id:
            raise ValueError("contrast trial IDs must be distinct")
        by_id = {item.trial_id: item for item in self.occurrences}
        if left_trial_id not in by_id or right_trial_id not in by_id:
            raise KeyError("contrast trial ID is absent from the ledger")
        left = by_id[left_trial_id]
        right = by_id[right_trial_id]
        return (
            left.role is EvaluationOccurrenceRole.PRIMARY
            and right.role is EvaluationOccurrenceRole.PRIMARY
            and left.status is EvaluationOccurrenceStatus.SUCCESS
            and right.status is EvaluationOccurrenceStatus.SUCCESS
            and left.phenotype is not None
            and left.phenotype == right.phenotype
        )

    def _trace_payload(self) -> dict[str, object]:
        return {
            "identity_policy_id": self.identity_policy_id,
            "identity_policy_version": self.identity_policy_version,
            "occurrences": [item.to_trace_record() for item in self.occurrences],
            "clusters": [cluster.to_trace_record() for cluster in self.clusters],
            "successful_primary_collision_credit": (
                self.successful_primary_collision_credit
            ),
            "invalidating_primary_trial_ids": [
                item.value for item in self.invalidating_primary_trial_ids
            ],
            "invalidating_trial_ids": [
                item.value for item in self.invalidating_trial_ids
            ],
            "primary_block_valid": self.primary_block_valid,
            "experiment_block_valid": self.experiment_block_valid,
            "ignored_recourse_trial_ids": [
                item.value for item in self.ignored_recourse_trial_ids
            ],
        }

    @property
    def ledger_sha256(self) -> str:
        return _domain_hash(_LEDGER_DOMAIN, self._trace_payload())

    def to_trace_record(self) -> dict[str, object]:
        return {**self._trace_payload(), "ledger_sha256": self.ledger_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PhenotypeOccurrenceLedger
            and type(other) is PhenotypeOccurrenceLedger
            and self.ledger_sha256 == other.ledger_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class RecoursePoolCandidate:
    """An engine-authored recourse candidate before identity materialization."""

    entry_id: str
    configuration: FrozenJsonValue

    def __post_init__(self) -> None:
        _token(self.entry_id, name="recourse entry_id")
        if not is_frozen_json_value(self.configuration):
            raise TypeError("recourse configuration must already be frozen typed JSON")
        freeze_json(self.configuration)

    @classmethod
    def freeze(cls, entry_id: str, configuration: object) -> "RecoursePoolCandidate":
        return cls(entry_id=entry_id, configuration=freeze_json(configuration))


@dataclass(frozen=True, slots=True)
class RecoursePoolEntry:
    """One ordered, identity-bound candidate in a pre-outcome pool."""

    ordinal: int
    candidate: RecoursePoolCandidate
    phenotype: PhenotypeIdentity

    def __post_init__(self) -> None:
        _nonnegative(self.ordinal, name="recourse entry ordinal")
        if type(self.candidate) is not RecoursePoolCandidate:
            raise TypeError("candidate must be an exact RecoursePoolCandidate")
        RecoursePoolCandidate.__post_init__(self.candidate)
        if type(self.phenotype) is not PhenotypeIdentity:
            raise TypeError("phenotype must be an exact PhenotypeIdentity")
        PhenotypeIdentity.__post_init__(self.phenotype)

    @property
    def entry_id(self) -> str:
        return self.candidate.entry_id

    def to_trace_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "entry_id": self.entry_id,
            "typed_configuration_sha256": typed_json_sha256(
                self.candidate.configuration
            ),
            "phenotype_identity_sha256": self.phenotype.identity_sha256,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PresealedRecoursePool:
    """An ordered recourse pool cryptographically bound before outcomes."""

    pool_id: str
    seal_context_sha256: str
    identity_policy_id: str
    identity_policy_version: int
    entries: tuple[RecoursePoolEntry, ...]

    def __post_init__(self) -> None:
        _token(self.pool_id, name="recourse pool_id")
        require_sha256(self.seal_context_sha256, "seal_context_sha256")
        _token(self.identity_policy_id, name="pool identity_policy_id")
        _positive(
            self.identity_policy_version,
            name="pool identity_policy_version",
        )
        if type(self.entries) is not tuple or any(
            type(item) is not RecoursePoolEntry for item in self.entries
        ):
            raise TypeError("entries must contain exact RecoursePoolEntry values")
        for entry in self.entries:
            RecoursePoolEntry.__post_init__(entry)
            if (
                entry.phenotype.policy_id != self.identity_policy_id
                or entry.phenotype.policy_version != self.identity_policy_version
            ):
                raise ValueError("pool entry uses a different identity policy")
        if tuple(entry.ordinal for entry in self.entries) != tuple(
            range(len(self.entries))
        ):
            raise ValueError("pool ordinals must be contiguous and ordered from zero")
        if len({entry.entry_id for entry in self.entries}) != len(self.entries):
            raise ValueError("recourse pool entry IDs must be unique")
        identities = tuple(entry.phenotype for entry in self.entries)
        if len(set(identities)) != len(identities):
            raise ValueError("recourse pool phenotypes must be unique")

    @classmethod
    def seal(
        cls,
        *,
        pool_id: str,
        seal_context_sha256: str,
        candidates: Sequence[RecoursePoolCandidate],
        identity_policy: PhenotypeIdentityPolicy,
    ) -> "PresealedRecoursePool":
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be a finite candidate sequence")
        if not callable(getattr(identity_policy, "identify", None)):
            raise TypeError("identity_policy must implement identify")
        policy_id = getattr(identity_policy, "policy_id", None)
        policy_version = getattr(identity_policy, "policy_version", None)
        _token(policy_id, name="identity policy_id")
        _positive(policy_version, name="identity policy_version")
        values = tuple(candidates)
        entries: list[RecoursePoolEntry] = []
        for ordinal, candidate in enumerate(values):
            if type(candidate) is not RecoursePoolCandidate:
                raise TypeError(
                    "candidates must contain exact RecoursePoolCandidate values"
                )
            identity = identity_policy.identify(candidate.configuration)
            if type(identity) is not PhenotypeIdentity:
                raise TypeError("identity policy must return exact PhenotypeIdentity")
            if (
                identity.policy_id != policy_id
                or identity.policy_version != policy_version
            ):
                raise ValueError(
                    "identity policy returned inconsistent policy metadata"
                )
            entries.append(RecoursePoolEntry(ordinal, candidate, identity))
        return cls(
            pool_id=pool_id,
            seal_context_sha256=seal_context_sha256,
            identity_policy_id=policy_id,
            identity_policy_version=policy_version,
            entries=tuple(entries),
        )

    def _trace_payload(self) -> dict[str, object]:
        return {
            "pool_id": self.pool_id,
            "seal_context_sha256": self.seal_context_sha256,
            "identity_policy_id": self.identity_policy_id,
            "identity_policy_version": self.identity_policy_version,
            "entries": [entry.to_trace_record() for entry in self.entries],
        }

    @property
    def pool_sha256(self) -> str:
        return _domain_hash(_POOL_DOMAIN, self._trace_payload())

    def to_trace_record(self) -> dict[str, object]:
        return {**self._trace_payload(), "pool_sha256": self.pool_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PresealedRecoursePool
            and type(other) is PresealedRecoursePool
            and self.pool_sha256 == other.pool_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class RecourseBudgetSnapshot:
    """Unique-evaluation facts visible to the objective-blind recourse policy."""

    max_unique_evaluations: int
    used_unique_evaluations: int
    reserved_non_recourse_evaluations: int
    protected_recombination_evaluations: int

    def __post_init__(self) -> None:
        _nonnegative(self.max_unique_evaluations, name="max_unique_evaluations")
        _nonnegative(self.used_unique_evaluations, name="used_unique_evaluations")
        _nonnegative(
            self.reserved_non_recourse_evaluations,
            name="reserved_non_recourse_evaluations",
        )
        _nonnegative(
            self.protected_recombination_evaluations,
            name="protected_recombination_evaluations",
        )
        if self.used_unique_evaluations > self.max_unique_evaluations:
            raise ValueError("used_unique_evaluations exceeds the hard maximum")
        if (
            self.used_unique_evaluations
            + self.reserved_non_recourse_evaluations
            + self.protected_recombination_evaluations
            > self.max_unique_evaluations
        ):
            raise ValueError(
                "used, reserved, and protected evaluations exceed the hard maximum"
            )

    @property
    def available_for_recourse(self) -> int:
        return max(
            0,
            self.max_unique_evaluations
            - self.used_unique_evaluations
            - self.reserved_non_recourse_evaluations
            - self.protected_recombination_evaluations,
        )

    def to_trace_record(self) -> dict[str, int]:
        return {
            "max_unique_evaluations": self.max_unique_evaluations,
            "used_unique_evaluations": self.used_unique_evaluations,
            "reserved_non_recourse_evaluations": (
                self.reserved_non_recourse_evaluations
            ),
            "protected_recombination_evaluations": (
                self.protected_recombination_evaluations
            ),
            "available_for_recourse": self.available_for_recourse,
        }


class RecourseDecisionReason(str, Enum):
    SELECTED = "selected"
    PRIMARY_BLOCK_INVALID = "primary_block_invalid"
    MAX_RECOURSE_ZERO = "max_recourse_zero"
    NO_SUCCESSFUL_PRIMARY_COLLISION = "no_successful_primary_collision"
    NO_ELIGIBLE_PRESEALED_CANDIDATE = "no_eligible_presealed_candidate"
    UNIQUE_BUDGET_PROTECTED = "unique_budget_protected"


def _eligible_pool_entries(
    ledger: PhenotypeOccurrenceLedger,
    pool: PresealedRecoursePool,
) -> tuple[RecoursePoolEntry, ...]:
    # Recourse outcomes are intentionally absent: they cannot change this pool
    # or trigger a second recourse wave.  Every primary identity-bearing outcome
    # is excluded because repeating it cannot require a new physical evaluation.
    occupied = ledger.primary_phenotypes
    return tuple(entry for entry in pool.entries if entry.phenotype not in occupied)


def _decision_facts(
    *,
    ledger: PhenotypeOccurrenceLedger,
    pool: PresealedRecoursePool,
    budget: RecourseBudgetSnapshot,
    max_recourse: int,
) -> tuple[
    tuple[RecoursePoolEntry, ...],
    int,
    RecourseDecisionReason,
]:
    eligible = _eligible_pool_entries(ledger, pool)
    invalidating = ledger.invalidating_primary_trial_ids
    credit = ledger.successful_primary_collision_credit
    if invalidating:
        return eligible, 0, RecourseDecisionReason.PRIMARY_BLOCK_INVALID

    slots = min(
        max_recourse,
        credit,
        budget.available_for_recourse,
        len(eligible),
    )
    if slots > 0:
        return eligible, slots, RecourseDecisionReason.SELECTED
    if max_recourse == 0:
        reason = RecourseDecisionReason.MAX_RECOURSE_ZERO
    elif credit == 0:
        reason = RecourseDecisionReason.NO_SUCCESSFUL_PRIMARY_COLLISION
    elif not eligible:
        reason = RecourseDecisionReason.NO_ELIGIBLE_PRESEALED_CANDIDATE
    else:
        reason = RecourseDecisionReason.UNIQUE_BUDGET_PROTECTED
    return eligible, 0, reason


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationRecourseDecision:
    """Complete, replayable, objective-free recourse decision."""

    ledger: PhenotypeOccurrenceLedger
    pool: PresealedRecoursePool
    budget: RecourseBudgetSnapshot
    max_recourse: int
    selected_entry_ids: tuple[str, ...]
    reason: RecourseDecisionReason

    policy_id: ClassVar[str] = RECOURSE_POLICY_ID
    policy_version: ClassVar[int] = RECOURSE_POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.ledger) is not PhenotypeOccurrenceLedger:
            raise TypeError("ledger must be an exact PhenotypeOccurrenceLedger")
        PhenotypeOccurrenceLedger.__post_init__(self.ledger)
        if type(self.pool) is not PresealedRecoursePool:
            raise TypeError("pool must be an exact PresealedRecoursePool")
        PresealedRecoursePool.__post_init__(self.pool)
        if type(self.budget) is not RecourseBudgetSnapshot:
            raise TypeError("budget must be an exact RecourseBudgetSnapshot")
        RecourseBudgetSnapshot.__post_init__(self.budget)
        _nonnegative(self.max_recourse, name="max_recourse")
        if (
            self.ledger.identity_policy_id != self.pool.identity_policy_id
            or self.ledger.identity_policy_version != self.pool.identity_policy_version
        ):
            raise ValueError("ledger and pool identity policies differ")
        if type(self.selected_entry_ids) is not tuple or any(
            type(item) is not str for item in self.selected_entry_ids
        ):
            raise TypeError("selected_entry_ids must be an exact string tuple")
        if type(self.reason) is not RecourseDecisionReason:
            raise TypeError("reason must be a RecourseDecisionReason")
        eligible, slots, expected_reason = _decision_facts(
            ledger=self.ledger,
            pool=self.pool,
            budget=self.budget,
            max_recourse=self.max_recourse,
        )
        expected_ids = tuple(entry.entry_id for entry in eligible[:slots])
        if self.selected_entry_ids != expected_ids:
            raise ValueError("selected entries do not match the bounded prefix rule")
        if self.reason is not expected_reason:
            raise ValueError("decision reason does not match the bounded rule")

    @property
    def eligible_entries(self) -> tuple[RecoursePoolEntry, ...]:
        return _eligible_pool_entries(self.ledger, self.pool)

    @property
    def selected_entries(self) -> tuple[RecoursePoolEntry, ...]:
        by_id = {entry.entry_id: entry for entry in self.pool.entries}
        return tuple(by_id[entry_id] for entry_id in self.selected_entry_ids)

    @property
    def slots(self) -> int:
        return len(self.selected_entry_ids)

    def _trace_payload(self) -> dict[str, object]:
        return {
            "event_type": "evaluation_recourse_decided",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "ledger_sha256": self.ledger.ledger_sha256,
            "pool_sha256": self.pool.pool_sha256,
            "budget": self.budget.to_trace_record(),
            "max_recourse": self.max_recourse,
            "successful_primary_collision_credit": (
                self.ledger.successful_primary_collision_credit
            ),
            "collision_credit_source": "successful_primary_occurrences_only",
            "invalidating_primary_trial_ids": [
                item.value for item in self.ledger.invalidating_primary_trial_ids
            ],
            "ignored_recourse_trial_ids": [
                item.value for item in self.ledger.ignored_recourse_trial_ids
            ],
            "eligible_entry_ids": [entry.entry_id for entry in self.eligible_entries],
            "selected_entry_ids": list(self.selected_entry_ids),
            "slots": self.slots,
            "reason": self.reason.value,
            "slot_formula": (
                "min(max_recourse, successful_primary_collision_credit, "
                "available_unique_budget_after_reservations_and_recombination, "
                "eligible_presealed_pool_size)"
            ),
        }

    @property
    def decision_sha256(self) -> str:
        return _domain_hash(_DECISION_DOMAIN, self._trace_payload())

    def to_trace_record(self) -> dict[str, object]:
        return {**self._trace_payload(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is EvaluationRecourseDecision
            and type(other) is EvaluationRecourseDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


class EvaluationRecoursePolicy(Protocol):
    """Objective-blind policy over identities, statuses, and reservations only."""

    def decide(
        self,
        *,
        ledger: PhenotypeOccurrenceLedger,
        pool: PresealedRecoursePool,
        budget: RecourseBudgetSnapshot,
    ) -> EvaluationRecourseDecision: ...


@dataclass(frozen=True, slots=True)
class BoundedEvaluationRecoursePolicy:
    """Choose a deterministic prefix under collision and hard-budget caps."""

    max_recourse: int

    policy_id: ClassVar[str] = RECOURSE_POLICY_ID
    policy_version: ClassVar[int] = RECOURSE_POLICY_VERSION

    def __post_init__(self) -> None:
        _nonnegative(self.max_recourse, name="max_recourse")

    def decide(
        self,
        *,
        ledger: PhenotypeOccurrenceLedger,
        pool: PresealedRecoursePool,
        budget: RecourseBudgetSnapshot,
    ) -> EvaluationRecourseDecision:
        for value, expected, name in (
            (ledger, PhenotypeOccurrenceLedger, "ledger"),
            (pool, PresealedRecoursePool, "pool"),
            (budget, RecourseBudgetSnapshot, "budget"),
        ):
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}")
        if (
            ledger.identity_policy_id != pool.identity_policy_id
            or ledger.identity_policy_version != pool.identity_policy_version
        ):
            raise ValueError("ledger and pool identity policies differ")
        eligible, slots, reason = _decision_facts(
            ledger=ledger,
            pool=pool,
            budget=budget,
            max_recourse=self.max_recourse,
        )
        return EvaluationRecourseDecision(
            ledger=ledger,
            pool=pool,
            budget=budget,
            max_recourse=self.max_recourse,
            selected_entry_ids=tuple(entry.entry_id for entry in eligible[:slots]),
            reason=reason,
        )


__all__ = [
    "BoundedEvaluationRecoursePolicy",
    "EvaluationOccurrenceRole",
    "EvaluationOccurrenceStatus",
    "EvaluationRecourseDecision",
    "EvaluationRecoursePolicy",
    "IDENTITY_POLICY_ID",
    "IDENTITY_POLICY_VERSION",
    "PhenotypeCluster",
    "PhenotypeIdentity",
    "PhenotypeIdentityPolicy",
    "PhenotypeOccurrence",
    "PhenotypeOccurrenceLedger",
    "PresealedRecoursePool",
    "RECOURSE_POLICY_ID",
    "RECOURSE_POLICY_VERSION",
    "RecourseBudgetSnapshot",
    "RecourseDecisionReason",
    "RecoursePoolCandidate",
    "RecoursePoolEntry",
    "SemanticProjectionPhenotypeIdentityPolicy",
    "TypedConfigurationPhenotypeIdentityPolicy",
]

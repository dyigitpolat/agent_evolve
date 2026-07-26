"""Archive-aware exploit and structural-coverage disjoint pair selection.

The legacy disjoint-pair policy ranks exploit pairs by parent-relative branch
rewards.  Those rewards need not agree with multiobjective archive progress.
This module provides an opt-in policy that instead consumes an authenticated
table of marginal branch utilities and exact joint utilities for every pair of
*observed source branches*, all evaluated against one frozen pre-generation
archive snapshot.

The exploit key is the utility of adding both observed sources jointly to the
frozen archive.  It is not the arithmetic sum of marginal utilities, which can
double-count overlapping archive gain.  It is deliberately not represented as
a prediction of the unseen recombination child's utility.  Coverage remains a
separately identified structural-diversity role.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar, Sequence

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.disjoint_pairs import (
    DisjointParentPairPolicy,
    ReplayVerifiedDisjointPair,
)


POLICY_ID = "frozen_archive_exact_joint_source_utility_disjoint_pair"
POLICY_VERSION = 2
_CONTEXT_DOMAIN = b"agent-evolve:frozen-archive-source-utility-context:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:frozen-archive-source-utility-receipt:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:frozen-archive-disjoint-pair-decision:v2\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_MAX_GENERATION = (1 << 63) - 1


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
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _require_generation(value: int, *, name: str) -> None:
    if type(value) is not int or not 1 <= value <= _MAX_GENERATION:
        raise ValueError(f"{name} must be an exact positive int63")


def _require_utility(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


def _pair_key(pair: ReplayVerifiedDisjointPair) -> tuple[str, str]:
    pair.revalidate()
    return pair.left.candidate_id.value, pair.right.candidate_id.value


@dataclass(frozen=True, slots=True)
class FrozenArchiveSourceUtilityContext:
    """Identity of one workload-owned utility and frozen archive cutoff."""

    utility_id: str
    utility_version: int
    utility_definition_sha256: str
    benchmark_sha256: str
    archive_cutoff_sha256: str
    archive_snapshot_sha256: str
    snapshot_generation: int

    def __post_init__(self) -> None:
        _require_token(self.utility_id, name="utility_id")
        if type(self.utility_version) is not int or self.utility_version <= 0:
            raise ValueError("utility_version must be a positive exact integer")
        for name in (
            "utility_definition_sha256",
            "benchmark_sha256",
            "archive_cutoff_sha256",
            "archive_snapshot_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_generation(self.snapshot_generation, name="snapshot_generation")

    def revalidate(self) -> None:
        if type(self) is not FrozenArchiveSourceUtilityContext:
            raise TypeError("context must be exact FrozenArchiveSourceUtilityContext")
        FrozenArchiveSourceUtilityContext.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "utility_id": self.utility_id,
            "utility_version": self.utility_version,
            "utility_definition_sha256": self.utility_definition_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "archive_cutoff_sha256": self.archive_cutoff_sha256,
            "archive_snapshot_sha256": self.archive_snapshot_sha256,
            "snapshot_generation": self.snapshot_generation,
        }

    @property
    def context_sha256(self) -> str:
        return _hash(_CONTEXT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "context_sha256": self.context_sha256}


@dataclass(frozen=True, slots=True)
class ObservedSourceBranch:
    """Exact evaluated source occurrence to which utility is attributed."""

    source_rank: int
    candidate_id: CandidateId
    candidate_configuration_sha256: str
    source_outcome_sha256: str

    def __post_init__(self) -> None:
        if type(self.source_rank) is not int or self.source_rank <= 0:
            raise ValueError("source_rank must be a positive exact integer")
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        require_sha256(
            self.candidate_configuration_sha256,
            "candidate_configuration_sha256",
        )
        require_sha256(self.source_outcome_sha256, "source_outcome_sha256")

    def revalidate(self) -> None:
        if type(self) is not ObservedSourceBranch:
            raise TypeError("source branch must be exact ObservedSourceBranch")
        ObservedSourceBranch.__post_init__(self)

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "source_rank": self.source_rank,
            "candidate_id": self.candidate_id.value,
            "candidate_configuration_sha256": (self.candidate_configuration_sha256),
            "source_outcome_sha256": self.source_outcome_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenArchiveBranchUtility:
    """One observed source branch's marginal utility under the shared snapshot."""

    source: ObservedSourceBranch
    marginal_utility: float

    def __post_init__(self) -> None:
        if type(self.source) is not ObservedSourceBranch:
            raise TypeError("source must be exact ObservedSourceBranch")
        self.source.revalidate()
        _require_utility(self.marginal_utility, name="marginal_utility")

    def revalidate(self) -> None:
        if type(self) is not FrozenArchiveBranchUtility:
            raise TypeError("branch utility must be exact FrozenArchiveBranchUtility")
        FrozenArchiveBranchUtility.__post_init__(self)

    @property
    def candidate_id(self) -> CandidateId:
        self.revalidate()
        return self.source.candidate_id

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "source": self.source.to_record(),
            "marginal_utility_hex": self.marginal_utility.hex(),
        }


@dataclass(frozen=True, slots=True)
class FrozenArchiveSourcePairUtility:
    """Exact utility of two observed sources jointly augmenting the archive."""

    pair_ids: tuple[CandidateId, CandidateId]
    exact_joint_utility: float

    def __post_init__(self) -> None:
        if (
            type(self.pair_ids) is not tuple
            or len(self.pair_ids) != 2
            or any(type(value) is not CandidateId for value in self.pair_ids)
        ):
            raise TypeError("pair_ids must contain two exact CandidateId values")
        for value in self.pair_ids:
            CandidateId.__post_init__(value)
        if self.pair_ids[0] >= self.pair_ids[1]:
            raise ValueError("source utility pair IDs must use canonical order")
        _require_utility(self.exact_joint_utility, name="exact_joint_utility")

    def revalidate(self) -> None:
        if type(self) is not FrozenArchiveSourcePairUtility:
            raise TypeError("pair utility must be exact FrozenArchiveSourcePairUtility")
        FrozenArchiveSourcePairUtility.__post_init__(self)

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "pair_ids": [value.value for value in self.pair_ids],
            "exact_joint_utility_hex": self.exact_joint_utility.hex(),
        }


@dataclass(frozen=True, slots=True)
class FrozenArchiveSourceUtilityReceipt:
    """Complete source and pair utility tables under one archive snapshot."""

    context: FrozenArchiveSourceUtilityContext
    source_wave_receipt_sha256: str
    source_request_sha256: str
    source_decision_sha256: str
    source_contract_sha256: str
    source_generation: int
    branches: tuple[FrozenArchiveBranchUtility, ...]
    pair_utilities: tuple[FrozenArchiveSourcePairUtility, ...]

    def __post_init__(self) -> None:
        if type(self.context) is not FrozenArchiveSourceUtilityContext:
            raise TypeError("context must be exact FrozenArchiveSourceUtilityContext")
        self.context.revalidate()
        for name in (
            "source_wave_receipt_sha256",
            "source_request_sha256",
            "source_decision_sha256",
            "source_contract_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_generation(self.source_generation, name="source_generation")
        if self.context.snapshot_generation != self.source_generation:
            raise ValueError(
                "archive utility snapshot is stale or foreign to the source generation"
            )
        if type(self.branches) is not tuple or any(
            type(value) is not FrozenArchiveBranchUtility for value in self.branches
        ):
            raise TypeError("branches must contain exact branch utilities")
        for value in self.branches:
            value.revalidate()
        source_ranks = tuple(value.source.source_rank for value in self.branches)
        if source_ranks != tuple(sorted(set(source_ranks))):
            raise ValueError(
                "branch utilities must preserve canonical scored source-rank order"
            )
        ids = tuple(value.candidate_id for value in self.branches)
        if len(set(ids)) != len(ids):
            raise ValueError("branch utility candidate IDs must be unique")
        if type(self.pair_utilities) is not tuple or any(
            type(value) is not FrozenArchiveSourcePairUtility
            for value in self.pair_utilities
        ):
            raise TypeError("pair_utilities must contain exact pair utilities")
        for value in self.pair_utilities:
            value.revalidate()
        expected_pairs = tuple(combinations(sorted(ids), 2))
        if tuple(value.pair_ids for value in self.pair_utilities) != expected_pairs:
            raise ValueError(
                "pair_utilities must completely enumerate source pairs in canonical order"
            )

    def revalidate(self) -> None:
        if type(self) is not FrozenArchiveSourceUtilityReceipt:
            raise TypeError("receipt must be exact FrozenArchiveSourceUtilityReceipt")
        FrozenArchiveSourceUtilityReceipt.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "context": self.context.to_record(),
            "source_wave_receipt_sha256": self.source_wave_receipt_sha256,
            "source_request_sha256": self.source_request_sha256,
            "source_decision_sha256": self.source_decision_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "source_generation": self.source_generation,
            "branches": [value.to_record() for value in self.branches],
            "pair_utilities": [value.to_record() for value in self.pair_utilities],
            "utility_semantics": (
                "marginal_branch_diagnostics_and_exact_joint_observed_source_pair_"
                "utility_against_one_frozen_archive; not_a_recombination_child_"
                "prediction"
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def marginal_utility_for(
        self, candidate_id: CandidateId
    ) -> FrozenArchiveBranchUtility:
        """Resolve a committed source marginal or fail closed."""

        self.revalidate()
        if type(candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(candidate_id)
        for value in self.branches:
            if value.candidate_id == candidate_id:
                return value
        raise ValueError("eligible pair contains a foreign or unscored source branch")

    def exact_joint_utility_for(
        self,
        pair_ids: tuple[CandidateId, CandidateId],
    ) -> FrozenArchiveSourcePairUtility:
        """Resolve the committed exact joint utility for one canonical pair."""

        self.revalidate()
        if (
            type(pair_ids) is not tuple
            or len(pair_ids) != 2
            or any(type(value) is not CandidateId for value in pair_ids)
        ):
            raise TypeError("pair_ids must contain two exact CandidateId values")
        for value in self.pair_utilities:
            if value.pair_ids == pair_ids:
                return value
        raise ValueError("eligible pair has no exact joint source utility")

    def require_exact_context(
        self,
        *,
        context: FrozenArchiveSourceUtilityContext,
        source_wave_receipt_sha256: str,
        source_request_sha256: str,
        source_decision_sha256: str,
        source_contract_sha256: str,
        source_generation: int,
        source_branches: Sequence[ObservedSourceBranch],
    ) -> None:
        """Authenticate the receipt against its calling source-wave boundary."""

        self.revalidate()
        if type(context) is not FrozenArchiveSourceUtilityContext:
            raise TypeError("context must be exact FrozenArchiveSourceUtilityContext")
        context.revalidate()
        for name, value in (
            ("source_wave_receipt_sha256", source_wave_receipt_sha256),
            ("source_request_sha256", source_request_sha256),
            ("source_decision_sha256", source_decision_sha256),
            ("source_contract_sha256", source_contract_sha256),
        ):
            require_sha256(value, name)
        _require_generation(source_generation, name="source_generation")
        if isinstance(source_branches, (str, bytes)):
            raise TypeError("source_branches must be a finite branch sequence")
        expected_branches = tuple(source_branches)
        for value in expected_branches:
            if type(value) is not ObservedSourceBranch:
                raise TypeError(
                    "source_branches must contain exact ObservedSourceBranch values"
                )
            value.revalidate()
        observed_branches = tuple(value.source for value in self.branches)
        if self.context != context:
            raise ValueError("source utility receipt has a foreign archive context")
        if self.source_generation != source_generation:
            raise ValueError("source utility receipt is stale for this generation")
        supplied = (
            self.source_wave_receipt_sha256,
            self.source_request_sha256,
            self.source_decision_sha256,
            self.source_contract_sha256,
        )
        expected = (
            source_wave_receipt_sha256,
            source_request_sha256,
            source_decision_sha256,
            source_contract_sha256,
        )
        if supplied != expected:
            raise ValueError("source utility receipt belongs to a foreign source wave")
        if observed_branches != expected_branches:
            raise ValueError("source utility receipt has foreign or tampered branches")


@dataclass(frozen=True, slots=True)
class ArchiveAwareDisjointPairScoreRow:
    """One safe pair under source utility and structural coverage scores."""

    pair: ReplayVerifiedDisjointPair
    left_marginal_utility: FrozenArchiveBranchUtility
    right_marginal_utility: FrozenArchiveBranchUtility
    exact_joint_source_utility: FrozenArchiveSourcePairUtility
    distinct_role_count: int
    distinct_family_count: int
    path_family_exposure_sum: int

    def __post_init__(self) -> None:
        if type(self.pair) is not ReplayVerifiedDisjointPair:
            raise TypeError("pair must be exact ReplayVerifiedDisjointPair")
        self.pair.revalidate()
        for name in ("left_marginal_utility", "right_marginal_utility"):
            value = getattr(self, name)
            if type(value) is not FrozenArchiveBranchUtility:
                raise TypeError(f"{name} must be exact FrozenArchiveBranchUtility")
            value.revalidate()
        if self.left_marginal_utility.candidate_id != self.pair.left.candidate_id:
            raise ValueError("left marginal belongs to a foreign source branch")
        if self.right_marginal_utility.candidate_id != self.pair.right.candidate_id:
            raise ValueError("right marginal belongs to a foreign source branch")
        if type(self.exact_joint_source_utility) is not FrozenArchiveSourcePairUtility:
            raise TypeError(
                "exact_joint_source_utility must be exact FrozenArchiveSourcePairUtility"
            )
        self.exact_joint_source_utility.revalidate()
        if self.exact_joint_source_utility.pair_ids != self.pair.pair_ids:
            raise ValueError("exact joint utility belongs to a foreign source pair")
        expected_roles = len({self.pair.left.role, self.pair.right.role})
        if (
            type(self.distinct_role_count) is not int
            or self.distinct_role_count != expected_roles
        ):
            raise ValueError("distinct_role_count differs from pair facts")
        expected_families = len({self.pair.left.family, self.pair.right.family})
        if (
            type(self.distinct_family_count) is not int
            or self.distinct_family_count != expected_families
        ):
            raise ValueError("distinct_family_count differs from pair facts")
        expected_exposure = (
            self.pair.left.path_family_exposure + self.pair.right.path_family_exposure
        )
        if (
            type(self.path_family_exposure_sum) is not int
            or self.path_family_exposure_sum != expected_exposure
        ):
            raise ValueError("path_family_exposure_sum differs from pair facts")

    def revalidate(self) -> None:
        if type(self) is not ArchiveAwareDisjointPairScoreRow:
            raise TypeError("score row must be exact ArchiveAwareDisjointPairScoreRow")
        ArchiveAwareDisjointPairScoreRow.__post_init__(self)

    @property
    def pair_ids(self) -> tuple[CandidateId, CandidateId]:
        self.revalidate()
        return self.pair.pair_ids

    def to_record(
        self,
        *,
        exploit_pair_ids: tuple[CandidateId, CandidateId] | None,
        exploit_target_configuration_sha256: str | None,
    ) -> dict[str, object]:
        self.revalidate()
        if exploit_pair_ids is not None:
            if (
                type(exploit_pair_ids) is not tuple
                or len(exploit_pair_ids) != 2
                or any(type(value) is not CandidateId for value in exploit_pair_ids)
            ):
                raise TypeError("exploit_pair_ids must contain exact CandidateIds")
        if exploit_target_configuration_sha256 is not None:
            require_sha256(
                exploit_target_configuration_sha256,
                "exploit_target_configuration_sha256",
            )
        legacy_reward_sum = self.pair.left.reward + self.pair.right.reward
        marginal_sum = (
            self.left_marginal_utility.marginal_utility
            + self.right_marginal_utility.marginal_utility
        )
        return {
            **self.pair.to_trace_record(),
            "left_source_marginal_utility": self.left_marginal_utility.to_record(),
            "right_source_marginal_utility": self.right_marginal_utility.to_record(),
            "marginal_utility_sum_diagnostic_hex": float(marginal_sum).hex(),
            "exact_joint_source_utility": self.exact_joint_source_utility.to_record(),
            "legacy_branch_reward_sum_hex": float(legacy_reward_sum).hex(),
            "distinct_role_count": self.distinct_role_count,
            "distinct_family_count": self.distinct_family_count,
            "path_family_exposure_sum": self.path_family_exposure_sum,
            "pair_distinct_from_exploit": (
                None if exploit_pair_ids is None else self.pair_ids != exploit_pair_ids
            ),
            "target_distinct_from_exploit": (
                None
                if exploit_target_configuration_sha256 is None
                else self.pair.target_configuration_sha256
                != exploit_target_configuration_sha256
            ),
            "exploit_tie_key": [
                -self.exact_joint_source_utility.exact_joint_utility,
                -legacy_reward_sum,
                -self.distinct_role_count,
                self.pair.left.candidate_id.value,
                self.pair.right.candidate_id.value,
            ],
            "coverage_tie_key": [
                (
                    None
                    if exploit_target_configuration_sha256 is None
                    else self.pair.target_configuration_sha256
                    == exploit_target_configuration_sha256
                ),
                -self.distinct_family_count,
                -self.distinct_role_count,
                self.path_family_exposure_sum,
                self.pair.left.candidate_id.value,
                self.pair.right.candidate_id.value,
            ],
        }


def _canonical_rows(
    pairs: Sequence[ReplayVerifiedDisjointPair],
    receipt: FrozenArchiveSourceUtilityReceipt,
) -> tuple[ArchiveAwareDisjointPairScoreRow, ...]:
    if type(receipt) is not FrozenArchiveSourceUtilityReceipt:
        raise TypeError("source_utilities must be an exact receipt")
    receipt.revalidate()
    legacy_rows = DisjointParentPairPolicy().select(pairs).eligible_rows
    result: list[ArchiveAwareDisjointPairScoreRow] = []
    for legacy in legacy_rows:
        pair = legacy.pair
        left = receipt.marginal_utility_for(pair.left.candidate_id)
        right = receipt.marginal_utility_for(pair.right.candidate_id)
        exact_joint = receipt.exact_joint_utility_for(pair.pair_ids)
        result.append(
            ArchiveAwareDisjointPairScoreRow(
                pair=pair,
                left_marginal_utility=left,
                right_marginal_utility=right,
                exact_joint_source_utility=exact_joint,
                distinct_role_count=legacy.distinct_role_count,
                distinct_family_count=legacy.distinct_family_count,
                path_family_exposure_sum=legacy.path_family_exposure_sum,
            )
        )
    return tuple(result)


def _select_rows(
    rows: tuple[ArchiveAwareDisjointPairScoreRow, ...],
) -> tuple[
    ArchiveAwareDisjointPairScoreRow | None,
    ArchiveAwareDisjointPairScoreRow | None,
]:
    if not rows:
        return None, None
    exploit = min(
        rows,
        key=lambda row: (
            -row.exact_joint_source_utility.exact_joint_utility,
            -(row.pair.left.reward + row.pair.right.reward),
            -row.distinct_role_count,
            *_pair_key(row.pair),
        ),
    )
    coverage_pool = tuple(row for row in rows if row.pair_ids != exploit.pair_ids)
    if not coverage_pool:
        return exploit, None
    coverage = min(
        coverage_pool,
        key=lambda row: (
            row.pair.target_configuration_sha256
            == exploit.pair.target_configuration_sha256,
            -row.distinct_family_count,
            -row.distinct_role_count,
            row.path_family_exposure_sum,
            *_pair_key(row.pair),
        ),
    )
    return exploit, coverage


@dataclass(frozen=True, slots=True, eq=False)
class ArchiveAwareDisjointPairSelectionDecision:
    """Trace-complete source-utility exploit and structural coverage choice."""

    source_utilities: FrozenArchiveSourceUtilityReceipt
    eligible_rows: tuple[ArchiveAwareDisjointPairScoreRow, ...]
    exploit_pair_ids: tuple[CandidateId, CandidateId] | None
    coverage_pair_ids: tuple[CandidateId, CandidateId] | None

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.source_utilities) is not FrozenArchiveSourceUtilityReceipt:
            raise TypeError("source_utilities must be an exact receipt")
        self.source_utilities.revalidate()
        if type(self.eligible_rows) is not tuple or any(
            type(value) is not ArchiveAwareDisjointPairScoreRow
            for value in self.eligible_rows
        ):
            raise TypeError("eligible_rows must contain exact archive-aware rows")
        expected_rows = _canonical_rows(
            tuple(value.pair for value in self.eligible_rows),
            self.source_utilities,
        )
        if self.eligible_rows != expected_rows:
            raise ValueError("eligible_rows differ from utility and pair evidence")
        exploit, coverage = _select_rows(expected_rows)
        expected_exploit = None if exploit is None else exploit.pair_ids
        expected_coverage = None if coverage is None else coverage.pair_ids
        if self.exploit_pair_ids != expected_exploit:
            raise ValueError("exploit_pair_ids do not match frozen source utility")
        if self.coverage_pair_ids != expected_coverage:
            raise ValueError("coverage_pair_ids do not match structural coverage")

    def revalidate(self) -> None:
        if type(self) is not ArchiveAwareDisjointPairSelectionDecision:
            raise TypeError("decision must be exact archive-aware decision")
        ArchiveAwareDisjointPairSelectionDecision.__post_init__(self)

    def _row_for(
        self,
        pair_ids: tuple[CandidateId, CandidateId] | None,
    ) -> ArchiveAwareDisjointPairScoreRow | None:
        if pair_ids is None:
            return None
        return next(value for value in self.eligible_rows if value.pair_ids == pair_ids)

    @property
    def exploit(self) -> ArchiveAwareDisjointPairScoreRow | None:
        self.revalidate()
        return self._row_for(self.exploit_pair_ids)

    @property
    def coverage(self) -> ArchiveAwareDisjointPairScoreRow | None:
        self.revalidate()
        return self._row_for(self.coverage_pair_ids)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        exploit = self._row_for(self.exploit_pair_ids)
        coverage = self._row_for(self.coverage_pair_ids)
        exploit_target = (
            None if exploit is None else exploit.pair.target_configuration_sha256
        )
        return {
            "schema_version": 1,
            "event_type": "archive_aware_recombination_pair_selected",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "source_utility_receipt": self.source_utilities.to_record(),
            "source_utility_receipt_sha256": self.source_utilities.receipt_sha256,
            "eligible_rows": [
                value.to_record(
                    exploit_pair_ids=self.exploit_pair_ids,
                    exploit_target_configuration_sha256=exploit_target,
                )
                for value in self.eligible_rows
            ],
            "exploit_pair_ids": (
                None if exploit is None else [value.value for value in exploit.pair_ids]
            ),
            "exploit_exact_joint_source_utility_hex": (
                None
                if exploit is None
                else exploit.exact_joint_source_utility.exact_joint_utility.hex()
            ),
            "coverage_pair_ids": (
                None
                if coverage is None
                else [value.value for value in coverage.pair_ids]
            ),
            "exploit_rule": (
                "maximum exact utility of the two observed sources jointly added "
                "to one frozen archive; maximum observed parent-relative branch-"
                "reward sum only within an exact archive-utility tie; maximum "
                "distinct source roles; canonical candidate IDs"
            ),
            "coverage_rule": (
                "exclude exploit pair; prefer a target-distinct pair; maximum "
                "distinct families; maximum distinct roles; minimum prior "
                "path/family exposure; canonical candidate IDs"
            ),
            "utility_scope": (
                "exact_joint_observed_sources_only_not_unseen_recombination_child_"
                "performance; branch_reward_sum_is_secondary_exact_tie_break_only; "
                "marginal_utility_sum_is_diagnostic_only"
            ),
        }

    @property
    def decision_sha256(self) -> str:
        return _hash(_DECISION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def to_trace_record(self) -> dict[str, object]:
        """Match the trace surface of the legacy disjoint-pair decision."""

        return self.to_record()

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ArchiveAwareDisjointPairSelectionDecision
            and type(other) is ArchiveAwareDisjointPairSelectionDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class ArchiveAwareDisjointParentPairPolicy:
    """Select utility exploit first and independent structural coverage second."""

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def select(
        self,
        eligible_pairs: Sequence[ReplayVerifiedDisjointPair],
        *,
        source_utilities: FrozenArchiveSourceUtilityReceipt,
    ) -> ArchiveAwareDisjointPairSelectionDecision:
        rows = _canonical_rows(eligible_pairs, source_utilities)
        exploit, coverage = _select_rows(rows)
        return ArchiveAwareDisjointPairSelectionDecision(
            source_utilities=source_utilities,
            eligible_rows=rows,
            exploit_pair_ids=None if exploit is None else exploit.pair_ids,
            coverage_pair_ids=None if coverage is None else coverage.pair_ids,
        )


__all__ = [
    "ArchiveAwareDisjointPairScoreRow",
    "ArchiveAwareDisjointPairSelectionDecision",
    "ArchiveAwareDisjointParentPairPolicy",
    "FrozenArchiveBranchUtility",
    "FrozenArchiveSourcePairUtility",
    "FrozenArchiveSourceUtilityContext",
    "FrozenArchiveSourceUtilityReceipt",
    "ObservedSourceBranch",
    "POLICY_ID",
    "POLICY_VERSION",
]

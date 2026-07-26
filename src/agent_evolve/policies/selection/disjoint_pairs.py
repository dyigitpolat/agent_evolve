"""Deterministic exploit/coverage choice over replay-verified disjoint pairs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar, Sequence

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256


POLICY_ID = "disjoint_parent_pair_exploit_coverage"
POLICY_VERSION = 1
_DECISION_DOMAIN = b"agent-evolve:disjoint-parent-pair-decision:v1\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_MAX_EXPOSURE = (1 << 63) - 1


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _finite_float(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _count(value: int, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_EXPOSURE:
        raise ValueError(f"{name} must be an exact non-negative int63")


@dataclass(frozen=True, slots=True)
class DisjointBranchFacts:
    """Pre-outcome facts for one valid, operator-compliant G1 branch."""

    candidate_id: CandidateId
    reward: float
    role: str
    family: str
    path_family_exposure: int

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.candidate_id)
        canonical_reward = _finite_float(self.reward, name="branch reward")
        if type(self.reward) is not float or self.reward != canonical_reward:
            raise TypeError("branch reward must already be a canonical float")
        _token(self.role, name="role")
        _token(self.family, name="family")
        _count(self.path_family_exposure, name="path_family_exposure")

    def revalidate(self) -> None:
        if type(self) is not DisjointBranchFacts:
            raise TypeError("branch facts must be exact DisjointBranchFacts")
        DisjointBranchFacts.__post_init__(self)

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "candidate_id": self.candidate_id.value,
            "reward": self.reward,
            "role": self.role,
            "family": self.family,
            "path_family_exposure": self.path_family_exposure,
        }


@dataclass(frozen=True, slots=True)
class ReplayVerifiedDisjointPair:
    """One safe pair plus the exact materialization evidence it produced."""

    left: DisjointBranchFacts
    right: DisjointBranchFacts
    target_configuration_sha256: str
    materialization_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.left) is not DisjointBranchFacts:
            raise TypeError("left must be exact DisjointBranchFacts")
        if type(self.right) is not DisjointBranchFacts:
            raise TypeError("right must be exact DisjointBranchFacts")
        self.left.revalidate()
        self.right.revalidate()
        if self.left.candidate_id >= self.right.candidate_id:
            raise ValueError(
                "pair branches must be distinct and canonically ordered by candidate ID"
            )
        require_sha256(
            self.target_configuration_sha256,
            "target_configuration_sha256",
        )
        require_sha256(
            self.materialization_receipt_sha256,
            "materialization_receipt_sha256",
        )

    def revalidate(self) -> None:
        if type(self) is not ReplayVerifiedDisjointPair:
            raise TypeError("eligible pair must be an exact ReplayVerifiedDisjointPair")
        ReplayVerifiedDisjointPair.__post_init__(self)

    @property
    def pair_ids(self) -> tuple[CandidateId, CandidateId]:
        self.revalidate()
        return self.left.candidate_id, self.right.candidate_id

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "left": self.left.to_trace_record(),
            "right": self.right.to_trace_record(),
            "target_configuration_sha256": self.target_configuration_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
        }


def _pair_key(pair: ReplayVerifiedDisjointPair) -> tuple[str, str]:
    pair.revalidate()
    return pair.left.candidate_id.value, pair.right.candidate_id.value


@dataclass(frozen=True, slots=True, eq=False)
class DisjointPairScoreRow:
    """Complete score row under both frozen G2 selection rules."""

    pair: ReplayVerifiedDisjointPair
    branch_reward_sum: float
    distinct_role_count: int
    distinct_family_count: int
    path_family_exposure_sum: int

    def __post_init__(self) -> None:
        if type(self.pair) is not ReplayVerifiedDisjointPair:
            raise TypeError("pair must be an exact ReplayVerifiedDisjointPair")
        self.pair.revalidate()
        expected_reward = self.pair.left.reward + self.pair.right.reward
        if not math.isfinite(expected_reward):
            raise ValueError("branch reward sum must be finite")
        if type(self.branch_reward_sum) is not float:
            raise TypeError("branch_reward_sum must be a canonical float")
        if self.branch_reward_sum != expected_reward:
            raise ValueError("branch_reward_sum does not match the branch facts")
        expected_roles = len({self.pair.left.role, self.pair.right.role})
        if (
            type(self.distinct_role_count) is not int
            or self.distinct_role_count != expected_roles
        ):
            raise ValueError("distinct_role_count does not match the branch facts")
        expected_families = len({self.pair.left.family, self.pair.right.family})
        if (
            type(self.distinct_family_count) is not int
            or self.distinct_family_count != expected_families
        ):
            raise ValueError("distinct_family_count does not match the branch facts")
        expected_exposure = (
            self.pair.left.path_family_exposure + self.pair.right.path_family_exposure
        )
        _count(expected_exposure, name="aggregated pair exposure")
        if (
            type(self.path_family_exposure_sum) is not int
            or self.path_family_exposure_sum != expected_exposure
        ):
            raise ValueError("path_family_exposure_sum does not match the branch facts")

    def revalidate(self) -> None:
        if type(self) is not DisjointPairScoreRow:
            raise TypeError("score row must be an exact DisjointPairScoreRow")
        DisjointPairScoreRow.__post_init__(self)

    @property
    def pair_ids(self) -> tuple[CandidateId, CandidateId]:
        self.revalidate()
        return self.pair.pair_ids

    def _key(self) -> tuple[object, ...]:
        self.revalidate()
        return (
            *_pair_key(self.pair),
            self.pair.target_configuration_sha256,
            self.pair.materialization_receipt_sha256,
            self.branch_reward_sum,
            self.distinct_role_count,
            self.distinct_family_count,
            self.path_family_exposure_sum,
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is DisjointPairScoreRow
            and type(other) is DisjointPairScoreRow
            and self._key() == other._key()
        )

    __hash__ = None

    def to_trace_record(
        self,
        *,
        exploit_target_configuration_sha256: str | None,
    ) -> dict[str, object]:
        self.revalidate()
        if exploit_target_configuration_sha256 is not None:
            require_sha256(
                exploit_target_configuration_sha256,
                "exploit_target_configuration_sha256",
            )
        return {
            **self.pair.to_trace_record(),
            "branch_reward_sum": self.branch_reward_sum,
            "distinct_role_count": self.distinct_role_count,
            "distinct_family_count": self.distinct_family_count,
            "path_family_exposure_sum": self.path_family_exposure_sum,
            "target_distinct_from_exploit": (
                None
                if exploit_target_configuration_sha256 is None
                else self.pair.target_configuration_sha256
                != exploit_target_configuration_sha256
            ),
            "exploit_tie_key": [
                -self.branch_reward_sum,
                -self.distinct_role_count,
                self.pair.left.candidate_id.value,
                self.pair.right.candidate_id.value,
            ],
            "coverage_tie_key": [
                -self.distinct_family_count,
                -self.distinct_role_count,
                self.path_family_exposure_sum,
                self.pair.left.candidate_id.value,
                self.pair.right.candidate_id.value,
            ],
        }


def _score_row(pair: ReplayVerifiedDisjointPair) -> DisjointPairScoreRow:
    pair.revalidate()
    reward_sum = pair.left.reward + pair.right.reward
    if not math.isfinite(reward_sum):
        raise ValueError("branch reward sum must be finite")
    exposure_sum = pair.left.path_family_exposure + pair.right.path_family_exposure
    _count(exposure_sum, name="aggregated pair exposure")
    return DisjointPairScoreRow(
        pair=pair,
        branch_reward_sum=float(reward_sum),
        distinct_role_count=len({pair.left.role, pair.right.role}),
        distinct_family_count=len({pair.left.family, pair.right.family}),
        path_family_exposure_sum=exposure_sum,
    )


def _canonical_rows(
    pairs: Sequence[ReplayVerifiedDisjointPair],
) -> tuple[DisjointPairScoreRow, ...]:
    if isinstance(pairs, (str, bytes)):
        raise TypeError("eligible_pairs must be a finite pair sequence")
    values = tuple(pairs)
    for pair in values:
        if type(pair) is not ReplayVerifiedDisjointPair:
            raise TypeError(
                "eligible_pairs must contain exact ReplayVerifiedDisjointPair values"
            )
        pair.revalidate()
    keys = [_pair_key(pair) for pair in values]
    if len(set(keys)) != len(keys):
        raise ValueError("eligible pair candidate-ID sets must be unique")
    receipts = [pair.materialization_receipt_sha256 for pair in values]
    if len(set(receipts)) != len(receipts):
        raise ValueError("materialization receipts must be unique across pairs")

    candidate_facts: dict[str, DisjointBranchFacts] = {}
    for pair in values:
        for branch in (pair.left, pair.right):
            candidate_key = branch.candidate_id.value
            existing = candidate_facts.get(candidate_key)
            if existing is not None and existing != branch:
                raise ValueError(
                    "one candidate has inconsistent branch facts across eligible pairs"
                )
            candidate_facts[candidate_key] = branch
    return tuple(
        sorted(
            (_score_row(pair) for pair in values), key=lambda row: _pair_key(row.pair)
        )
    )


def _select_rows(
    rows: tuple[DisjointPairScoreRow, ...],
) -> tuple[DisjointPairScoreRow | None, DisjointPairScoreRow | None]:
    if not rows:
        return None, None
    exploit = min(
        rows,
        key=lambda row: (
            -row.branch_reward_sum,
            -row.distinct_role_count,
            *_pair_key(row.pair),
        ),
    )
    coverage_pool = tuple(
        row
        for row in rows
        if row.pair.target_configuration_sha256
        != exploit.pair.target_configuration_sha256
    )
    if not coverage_pool:
        return exploit, None
    coverage = min(
        coverage_pool,
        key=lambda row: (
            -row.distinct_family_count,
            -row.distinct_role_count,
            row.path_family_exposure_sum,
            *_pair_key(row.pair),
        ),
    )
    return exploit, coverage


@dataclass(frozen=True, slots=True, eq=False)
class DisjointPairSelectionDecision:
    """All eligible score rows and the exact exploit/coverage selections."""

    eligible_rows: tuple[DisjointPairScoreRow, ...]
    exploit_pair_ids: tuple[CandidateId, CandidateId] | None
    coverage_pair_ids: tuple[CandidateId, CandidateId] | None

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.eligible_rows) is not tuple or any(
            type(row) is not DisjointPairScoreRow for row in self.eligible_rows
        ):
            raise TypeError(
                "eligible_rows must contain exact DisjointPairScoreRow values"
            )
        expected_rows = _canonical_rows(tuple(row.pair for row in self.eligible_rows))
        if self.eligible_rows != expected_rows:
            raise ValueError("eligible_rows must be complete and canonically ordered")
        exploit, coverage = _select_rows(expected_rows)
        expected_exploit = None if exploit is None else exploit.pair_ids
        expected_coverage = None if coverage is None else coverage.pair_ids
        if self.exploit_pair_ids != expected_exploit:
            raise ValueError("exploit_pair_ids do not match the exploit rule")
        if self.coverage_pair_ids != expected_coverage:
            raise ValueError("coverage_pair_ids do not match the coverage rule")

    def revalidate(self) -> None:
        if type(self) is not DisjointPairSelectionDecision:
            raise TypeError("decision must be an exact DisjointPairSelectionDecision")
        DisjointPairSelectionDecision.__post_init__(self)

    def _row_for(
        self,
        pair_ids: tuple[CandidateId, CandidateId] | None,
    ) -> DisjointPairScoreRow | None:
        if pair_ids is None:
            return None
        return next(row for row in self.eligible_rows if row.pair_ids == pair_ids)

    @property
    def exploit(self) -> DisjointPairScoreRow | None:
        self.revalidate()
        return self._row_for(self.exploit_pair_ids)

    @property
    def coverage(self) -> DisjointPairScoreRow | None:
        self.revalidate()
        return self._row_for(self.coverage_pair_ids)

    def _trace_payload(self) -> dict[str, object]:
        exploit = self._row_for(self.exploit_pair_ids)
        coverage = self._row_for(self.coverage_pair_ids)
        exploit_target = (
            None if exploit is None else exploit.pair.target_configuration_sha256
        )
        return {
            "event_type": "recombination_pair_selected",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "eligible_rows": [
                row.to_trace_record(
                    exploit_target_configuration_sha256=exploit_target,
                )
                for row in self.eligible_rows
            ],
            "exploit_pair_ids": (
                None
                if exploit is None
                else [candidate_id.value for candidate_id in exploit.pair_ids]
            ),
            "exploit_target_configuration_sha256": exploit_target,
            "coverage_pair_ids": (
                None
                if coverage is None
                else [candidate_id.value for candidate_id in coverage.pair_ids]
            ),
            "coverage_target_configuration_sha256": (
                None if coverage is None else coverage.pair.target_configuration_sha256
            ),
            "exploit_rule": (
                "maximum branch reward sum; maximum distinct roles; "
                "canonical candidate IDs"
            ),
            "coverage_rule": (
                "target configuration distinct from exploit; maximum distinct "
                "families; maximum distinct roles; minimum prior path/family "
                "exposure; canonical candidate IDs"
            ),
        }

    @property
    def decision_sha256(self) -> str:
        self.revalidate()
        encoded = json.dumps(
            self._trace_payload(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_DECISION_DOMAIN + encoded).hexdigest()

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {**self._trace_payload(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is DisjointPairSelectionDecision
            and type(other) is DisjointPairSelectionDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class DisjointParentPairPolicy:
    """Select exploit first and a target-distinct coverage pair second."""

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def select(
        self,
        eligible_pairs: Sequence[ReplayVerifiedDisjointPair],
    ) -> DisjointPairSelectionDecision:
        rows = _canonical_rows(eligible_pairs)
        exploit, coverage = _select_rows(rows)
        return DisjointPairSelectionDecision(
            eligible_rows=rows,
            exploit_pair_ids=None if exploit is None else exploit.pair_ids,
            coverage_pair_ids=None if coverage is None else coverage.pair_ids,
        )


__all__ = [
    "DisjointBranchFacts",
    "DisjointPairScoreRow",
    "DisjointPairSelectionDecision",
    "DisjointParentPairPolicy",
    "POLICY_ID",
    "POLICY_VERSION",
    "ReplayVerifiedDisjointPair",
]

"""Benchmark-neutral contracts for post-commit, rank-only reference releases.

The port deliberately separates decision authority from reference evaluation.
Callers must bind an already-finalized decision, its selected-union release,
the exact eligible item set, and the endpoint policy before a private reference
reader can be opened.  The public release contains selected-portfolio ranks and
aggregate counts; it has no field capable of carrying an unselected item value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re

from agent_evolve.domain.patch import require_sha256


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_REQUEST_DOMAIN = b"agent-evolve:postcommit-rank-request:v1\x00"
_AUTHORIZATION_DOMAIN = b"agent-evolve:postcommit-rank-authorization:v1\x00"
_RELEASE_DOMAIN = b"agent-evolve:postcommit-rank-release:v1\x00"


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


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must use the canonical identifier grammar")
    return value


class RankDirection(str, Enum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class PortfolioAggregateKind(str, Enum):
    """Closed aggregation laws supported by the rank-only reference core."""

    SUM = "sum"


@dataclass(frozen=True, slots=True, eq=False)
class RankEndpointPolicyBinding:
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    direction: RankDirection
    aggregate_kind: PortfolioAggregateKind

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.direction) is not RankDirection:
            raise TypeError("direction must be an exact RankDirection")
        if type(self.aggregate_kind) is not PortfolioAggregateKind:
            raise TypeError("aggregate_kind must be exact")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "direction": self.direction.value,
            "aggregate_kind": self.aggregate_kind.value,
        }


@dataclass(frozen=True, slots=True)
class SelectedPortfolioClaim:
    """One durably selected set, identified without carrying outcomes."""

    method_id: str
    treatment_id: str
    item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.method_id, "method_id")
        _require_identifier(self.treatment_id, "treatment_id")
        if type(self.item_ids) is not tuple or not self.item_ids:
            raise ValueError("item_ids must be a non-empty exact tuple")
        if any(type(value) is not str or not value for value in self.item_ids):
            raise ValueError("item_ids must contain non-empty exact strings")
        if self.item_ids != tuple(sorted(set(self.item_ids))):
            raise ValueError("portfolio item_ids must be unique and canonical")

    @property
    def claim_id(self) -> str:
        return f"{self.method_id}:{self.treatment_id}"

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "method_id": self.method_id,
            "treatment_id": self.treatment_id,
            "claim_id": self.claim_id,
            "item_ids": list(self.item_ids),
        }


@dataclass(frozen=True, slots=True, eq=False)
class PostcommitRankRequest:
    """Closed request frozen only after decisions and selected-union release."""

    benchmark_id: str
    source_run_finalization_sha256: str
    source_run_recursive_content_sha256: str
    decision_commitment_sha256: str
    selected_union_commitment_sha256: str
    selected_union_release_sha256: str
    eligibility_receipt_sha256: str
    reference_source_sha256: str
    endpoint_policy: RankEndpointPolicyBinding
    eligible_item_ids: tuple[str, ...]
    portfolio_size: int
    selected_portfolios: tuple[SelectedPortfolioClaim, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.benchmark_id, "benchmark_id")
        for name in (
            "source_run_finalization_sha256",
            "source_run_recursive_content_sha256",
            "decision_commitment_sha256",
            "selected_union_commitment_sha256",
            "selected_union_release_sha256",
            "eligibility_receipt_sha256",
            "reference_source_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.endpoint_policy) is not RankEndpointPolicyBinding:
            raise TypeError("endpoint_policy must be exact")
        self.endpoint_policy.__post_init__()
        if type(self.eligible_item_ids) is not tuple or not self.eligible_item_ids:
            raise ValueError("eligible_item_ids must be a non-empty exact tuple")
        if any(
            type(value) is not str or not value for value in self.eligible_item_ids
        ):
            raise ValueError("eligible item IDs must be non-empty exact strings")
        if self.eligible_item_ids != tuple(sorted(set(self.eligible_item_ids))):
            raise ValueError("eligible item IDs must be unique and canonical")
        if (
            type(self.portfolio_size) is not int
            or self.portfolio_size <= 0
            or self.portfolio_size > len(self.eligible_item_ids)
        ):
            raise ValueError("portfolio_size is outside the eligible item set")
        if type(self.selected_portfolios) is not tuple or not self.selected_portfolios:
            raise ValueError("selected_portfolios must be a non-empty exact tuple")
        for value in self.selected_portfolios:
            if type(value) is not SelectedPortfolioClaim:
                raise TypeError("selected_portfolios must contain exact claims")
            value.__post_init__()
            if len(value.item_ids) != self.portfolio_size:
                raise ValueError("selected portfolio size differs from the request")
            if not set(value.item_ids).issubset(self.eligible_item_ids):
                raise ValueError("selected portfolio escapes the eligible item set")
        if self.selected_portfolios != tuple(
            sorted(
                self.selected_portfolios,
                key=lambda value: (value.method_id, value.treatment_id),
            )
        ):
            raise ValueError("selected portfolios must use canonical claim order")
        claim_ids = tuple(value.claim_id for value in self.selected_portfolios)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("selected portfolio claim IDs cannot repeat")

    @property
    def reference_read_count(self) -> int:
        return len(self.eligible_item_ids)

    @property
    def exact_portfolio_count(self) -> int:
        return math.comb(len(self.eligible_item_ids), self.portfolio_size)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "benchmark_id": self.benchmark_id,
            "source_run_finalization_sha256": self.source_run_finalization_sha256,
            "source_run_recursive_content_sha256": (
                self.source_run_recursive_content_sha256
            ),
            "decision_commitment_sha256": self.decision_commitment_sha256,
            "selected_union_commitment_sha256": (
                self.selected_union_commitment_sha256
            ),
            "selected_union_release_sha256": self.selected_union_release_sha256,
            "eligibility_receipt_sha256": self.eligibility_receipt_sha256,
            "reference_source_sha256": self.reference_source_sha256,
            "endpoint_policy": self.endpoint_policy.to_record(),
            "eligible_item_ids": list(self.eligible_item_ids),
            "eligible_item_count": len(self.eligible_item_ids),
            "reference_read_count": self.reference_read_count,
            "portfolio_size": self.portfolio_size,
            "exact_portfolio_count": self.exact_portfolio_count,
            "selected_portfolios": [
                value.to_record() for value in self.selected_portfolios
            ],
            "outcomes_read_while_building_request": False,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True, eq=False)
class PostcommitRankAuthorization:
    """Receipt for a caller-durable, read-back prerelease authorization.

    The generic core binds this receipt.  Filesystem durability and chronology
    are verified by the composing run harness before it constructs the private
    reader; a digest alone is not claimed to prove either property.
    """

    request_sha256: str
    prerelease_file_sha256: str
    authorization_scope: str

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.prerelease_file_sha256, "prerelease_file_sha256")
        _require_identifier(self.authorization_scope, "authorization_scope")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "prerelease_file_sha256": self.prerelease_file_sha256,
            "authorization_scope": self.authorization_scope,
            "durability_verified_by_composing_harness": True,
        }

    @property
    def authorization_sha256(self) -> str:
        return _hash(_AUTHORIZATION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "authorization_sha256": self.authorization_sha256,
        }


@dataclass(frozen=True, slots=True)
class RankReferenceObservation:
    """Private scalar contribution returned once per eligible reference item."""

    item_id: str
    endpoint_component: float
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.item_id) is not str or not self.item_id:
            raise ValueError("item_id must be a non-empty exact string")
        if (
            type(self.endpoint_component) is not float
            or not math.isfinite(self.endpoint_component)
        ):
            raise TypeError("endpoint_component must be a finite canonical float")
        require_sha256(self.source_receipt_sha256, "source_receipt_sha256")


@dataclass(frozen=True, slots=True)
class SelectedPortfolioRank:
    method_id: str
    treatment_id: str
    item_ids: tuple[str, ...]
    endpoint: float
    competition_rank: int
    strictly_better_count: int
    tied_portfolio_count: int
    strictly_worse_count: int
    denominator_count: int

    def __post_init__(self) -> None:
        claim = SelectedPortfolioClaim(
            method_id=self.method_id,
            treatment_id=self.treatment_id,
            item_ids=self.item_ids,
        )
        claim.__post_init__()
        if type(self.endpoint) is not float or not math.isfinite(self.endpoint):
            raise TypeError("endpoint must be a finite canonical float")
        integers = (
            self.competition_rank,
            self.strictly_better_count,
            self.tied_portfolio_count,
            self.strictly_worse_count,
            self.denominator_count,
        )
        if any(type(value) is not int for value in integers):
            raise TypeError("rank counts must be exact integers")
        if (
            self.denominator_count <= 0
            or self.strictly_better_count < 0
            or self.tied_portfolio_count <= 0
            or self.strictly_worse_count < 0
            or self.competition_rank != self.strictly_better_count + 1
            or self.strictly_better_count
            + self.tied_portfolio_count
            + self.strictly_worse_count
            != self.denominator_count
        ):
            raise ValueError("portfolio rank counts are inconsistent")

    @property
    def claim_id(self) -> str:
        return f"{self.method_id}:{self.treatment_id}"

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "method_id": self.method_id,
            "treatment_id": self.treatment_id,
            "claim_id": self.claim_id,
            "item_ids": list(self.item_ids),
            "selected_portfolio_endpoint": self.endpoint,
            "selected_portfolio_endpoint_hex": self.endpoint.hex(),
            "competition_rank": self.competition_rank,
            "strictly_better_count": self.strictly_better_count,
            "tied_portfolio_count": self.tied_portfolio_count,
            "strictly_worse_count": self.strictly_worse_count,
            "denominator_count": self.denominator_count,
            "best_rank_percentile_0_best": (
                self.strictly_better_count / self.denominator_count
            ),
        }


@dataclass(frozen=True, slots=True, eq=False)
class PostcommitRankRelease:
    request_sha256: str
    authorization_sha256: str
    private_reference_table_sha256: str
    private_portfolio_endpoint_table_sha256: str
    exact_reference_read_count: int
    exact_portfolio_count: int
    distinct_portfolio_endpoint_count: int
    portfolio_endpoint_tie_group_count: int
    selected_ranks: tuple[SelectedPortfolioRank, ...]

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "authorization_sha256",
            "private_reference_table_sha256",
            "private_portfolio_endpoint_table_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in (
            "exact_reference_read_count",
            "exact_portfolio_count",
            "distinct_portfolio_endpoint_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            type(self.portfolio_endpoint_tie_group_count) is not int
            or self.portfolio_endpoint_tie_group_count < 0
        ):
            raise ValueError(
                "portfolio_endpoint_tie_group_count must be non-negative"
            )
        if self.distinct_portfolio_endpoint_count > self.exact_portfolio_count:
            raise ValueError("distinct endpoint count exceeds portfolio count")
        if self.portfolio_endpoint_tie_group_count > (
            self.distinct_portfolio_endpoint_count
        ):
            raise ValueError("tie-group count exceeds distinct endpoints")
        if type(self.selected_ranks) is not tuple or not self.selected_ranks:
            raise ValueError("selected_ranks must be a non-empty exact tuple")
        for value in self.selected_ranks:
            if type(value) is not SelectedPortfolioRank:
                raise TypeError("selected_ranks must contain exact rows")
            value.__post_init__()
            if value.denominator_count != self.exact_portfolio_count:
                raise ValueError("selected rank denominator differs from release")
        if self.selected_ranks != tuple(
            sorted(
                self.selected_ranks,
                key=lambda value: (value.method_id, value.treatment_id),
            )
        ):
            raise ValueError("selected ranks must use canonical claim order")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        ranks = tuple(value.competition_rank for value in self.selected_ranks)
        return {
            "schema_version": 1,
            "status": "completed_postcommit_rank_only_release",
            "request_sha256": self.request_sha256,
            "authorization_sha256": self.authorization_sha256,
            "private_reference_table_sha256": self.private_reference_table_sha256,
            "private_portfolio_endpoint_table_sha256": (
                self.private_portfolio_endpoint_table_sha256
            ),
            "exact_reference_read_count": self.exact_reference_read_count,
            "exact_portfolio_count": self.exact_portfolio_count,
            "aggregate_diagnostics": {
                "distinct_portfolio_endpoint_count": (
                    self.distinct_portfolio_endpoint_count
                ),
                "portfolio_endpoint_tie_group_count": (
                    self.portfolio_endpoint_tie_group_count
                ),
                "selected_claim_count": len(self.selected_ranks),
                "unique_selected_portfolio_count": len(
                    {value.item_ids for value in self.selected_ranks}
                ),
                "minimum_selected_competition_rank": min(ranks),
                "maximum_selected_competition_rank": max(ranks),
            },
            "selected_ranks": [value.to_record() for value in self.selected_ranks],
            "raw_reference_values_returned": False,
            "unselected_item_values_returned": False,
            "unselected_portfolio_endpoints_returned": False,
            "provider_calls": 0,
            "new_candidate_evaluations": 0,
        }

    @property
    def release_sha256(self) -> str:
        return _hash(_RELEASE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "release_sha256": self.release_sha256}


__all__ = [
    "PortfolioAggregateKind",
    "PostcommitRankAuthorization",
    "PostcommitRankRelease",
    "PostcommitRankRequest",
    "RankDirection",
    "RankEndpointPolicyBinding",
    "RankReferenceObservation",
    "SelectedPortfolioClaim",
    "SelectedPortfolioRank",
]

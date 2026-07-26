"""One-shot exact-set ranking over a private, post-commit reference reader."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import hashlib
import itertools
import json
import math

from agent_evolve.ports.postcommit_rank_authority import (
    PortfolioAggregateKind,
    PostcommitRankAuthorization,
    PostcommitRankRelease,
    PostcommitRankRequest,
    RankDirection,
    RankReferenceObservation,
    SelectedPortfolioRank,
)


_REFERENCE_TABLE_DOMAIN = b"agent-evolve:private-rank-reference-table:v1\x00"
_PORTFOLIO_TABLE_DOMAIN = b"agent-evolve:private-portfolio-endpoint-table:v1\x00"


class PostcommitRankAuthorityError(RuntimeError):
    """The rank-only authority or private reader escaped its frozen contract."""


RankReferenceReader = Callable[[str], RankReferenceObservation]


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


class PostcommitRankOnlyAuthority:
    """Consume one private reader and irreversibly release ranks only.

    The capability is burned before the first read.  A malformed or failing
    reader therefore cannot be repaired and retried after observing a prefix of
    the reference set.  Every eligible ID is requested exactly once in the
    request's canonical order.
    """

    def __init__(
        self,
        *,
        request: PostcommitRankRequest,
        authorization: PostcommitRankAuthorization,
        reader: RankReferenceReader,
    ) -> None:
        if type(request) is not PostcommitRankRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(authorization) is not PostcommitRankAuthorization:
            raise TypeError("authorization must be exact")
        authorization.__post_init__()
        if authorization.request_sha256 != request.request_sha256:
            raise ValueError("authorization names another rank request")
        if not callable(reader):
            raise TypeError("reader must be callable")
        self._request = request
        self._authorization = authorization
        self._reader = reader
        self._used = False

    @property
    def used(self) -> bool:
        return self._used

    def release(self) -> PostcommitRankRelease:
        if self._used:
            raise RuntimeError("postcommit rank authority is one-shot")
        self._used = True

        observations: list[RankReferenceObservation] = []
        for expected_item_id in self._request.eligible_item_ids:
            observation = self._reader(expected_item_id)
            if type(observation) is not RankReferenceObservation:
                raise PostcommitRankAuthorityError(
                    "private reader returned a non-reference observation"
                )
            observation.__post_init__()
            if observation.item_id != expected_item_id:
                raise PostcommitRankAuthorityError(
                    "private reader returned another eligible item"
                )
            observations.append(observation)
        if len(observations) != self._request.reference_read_count:
            raise AssertionError("reference read cardinality changed")

        component_by_id = {
            value.item_id: value.endpoint_component for value in observations
        }
        if len(component_by_id) != self._request.reference_read_count:
            raise PostcommitRankAuthorityError("private reader repeated an item")
        private_reference_sha256 = _hash(
            _REFERENCE_TABLE_DOMAIN,
            [
                {
                    "item_id": value.item_id,
                    "endpoint_component_hex": value.endpoint_component.hex(),
                    "source_receipt_sha256": value.source_receipt_sha256,
                }
                for value in observations
            ],
        )

        if (
            self._request.endpoint_policy.aggregate_kind
            is not PortfolioAggregateKind.SUM
        ):
            raise PostcommitRankAuthorityError("unsupported portfolio aggregate")
        portfolios = tuple(
            itertools.combinations(
                self._request.eligible_item_ids,
                self._request.portfolio_size,
            )
        )
        if len(portfolios) != self._request.exact_portfolio_count:
            raise AssertionError("exact portfolio denominator changed")
        endpoint_by_portfolio = {
            item_ids: float(sum(component_by_id[value] for value in item_ids))
            for item_ids in portfolios
        }
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in endpoint_by_portfolio.values()
        ):
            raise PostcommitRankAuthorityError(
                "portfolio endpoint escaped finite canonical floats"
            )
        private_portfolio_sha256 = _hash(
            _PORTFOLIO_TABLE_DOMAIN,
            [
                {
                    "item_ids": list(item_ids),
                    "endpoint_hex": endpoint_by_portfolio[item_ids].hex(),
                }
                for item_ids in portfolios
            ],
        )
        endpoint_counts = Counter(endpoint_by_portfolio.values())
        lower_is_better = (
            self._request.endpoint_policy.direction
            is RankDirection.LOWER_IS_BETTER
        )

        selected_ranks: list[SelectedPortfolioRank] = []
        for claim in self._request.selected_portfolios:
            selected_endpoint = endpoint_by_portfolio[claim.item_ids]
            strictly_better = sum(
                value < selected_endpoint if lower_is_better else value > selected_endpoint
                for value in endpoint_by_portfolio.values()
            )
            tied = endpoint_counts[selected_endpoint]
            selected_ranks.append(
                SelectedPortfolioRank(
                    method_id=claim.method_id,
                    treatment_id=claim.treatment_id,
                    item_ids=claim.item_ids,
                    endpoint=selected_endpoint,
                    competition_rank=1 + strictly_better,
                    strictly_better_count=strictly_better,
                    tied_portfolio_count=tied,
                    strictly_worse_count=(
                        self._request.exact_portfolio_count
                        - strictly_better
                        - tied
                    ),
                    denominator_count=self._request.exact_portfolio_count,
                )
            )

        return PostcommitRankRelease(
            request_sha256=self._request.request_sha256,
            authorization_sha256=self._authorization.authorization_sha256,
            private_reference_table_sha256=private_reference_sha256,
            private_portfolio_endpoint_table_sha256=private_portfolio_sha256,
            exact_reference_read_count=len(observations),
            exact_portfolio_count=len(portfolios),
            distinct_portfolio_endpoint_count=len(endpoint_counts),
            portfolio_endpoint_tie_group_count=sum(
                count > 1 for count in endpoint_counts.values()
            ),
            selected_ranks=tuple(selected_ranks),
        )


def validate_postcommit_rank_release_bindings(
    *,
    request: PostcommitRankRequest,
    authorization: PostcommitRankAuthorization,
    release: PostcommitRankRelease,
) -> PostcommitRankRelease:
    """Validate all public bindings without requiring private reference values."""

    if type(request) is not PostcommitRankRequest:
        raise TypeError("request must be exact")
    request.__post_init__()
    if type(authorization) is not PostcommitRankAuthorization:
        raise TypeError("authorization must be exact")
    authorization.__post_init__()
    if type(release) is not PostcommitRankRelease:
        raise TypeError("release must be exact")
    release.__post_init__()
    if (
        authorization.request_sha256 != request.request_sha256
        or release.request_sha256 != request.request_sha256
        or release.authorization_sha256 != authorization.authorization_sha256
        or release.exact_reference_read_count != request.reference_read_count
        or release.exact_portfolio_count != request.exact_portfolio_count
    ):
        raise PostcommitRankAuthorityError("rank release authority bindings differ")
    released_claims = tuple(
        (value.method_id, value.treatment_id, value.item_ids)
        for value in release.selected_ranks
    )
    requested_claims = tuple(
        (value.method_id, value.treatment_id, value.item_ids)
        for value in request.selected_portfolios
    )
    if released_claims != requested_claims:
        raise PostcommitRankAuthorityError("rank release selected claims differ")
    return release


__all__ = [
    "PostcommitRankAuthorityError",
    "PostcommitRankOnlyAuthority",
    "RankReferenceReader",
    "validate_postcommit_rank_release_bindings",
]

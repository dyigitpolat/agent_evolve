from __future__ import annotations

import hashlib
import json

import pytest

from agent_evolve.application.postcommit_rank_authority import (
    PostcommitRankAuthorityError,
    PostcommitRankOnlyAuthority,
    validate_postcommit_rank_release_bindings,
)
from agent_evolve.ports.postcommit_rank_authority import (
    PortfolioAggregateKind,
    PostcommitRankAuthorization,
    PostcommitRankRequest,
    RankDirection,
    RankEndpointPolicyBinding,
    RankReferenceObservation,
    SelectedPortfolioClaim,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _request(*, direction: RankDirection = RankDirection.LOWER_IS_BETTER):
    return PostcommitRankRequest(
        benchmark_id="generic_rank_fixture",
        source_run_finalization_sha256=_digest("finalization"),
        source_run_recursive_content_sha256=_digest("recursive"),
        decision_commitment_sha256=_digest("decision"),
        selected_union_commitment_sha256=_digest("union-commit"),
        selected_union_release_sha256=_digest("union-release"),
        eligibility_receipt_sha256=_digest("eligibility"),
        reference_source_sha256=_digest("reference"),
        endpoint_policy=RankEndpointPolicyBinding(
            policy_id="fixture_additive_endpoint",
            policy_version=1,
            policy_definition_sha256=_digest("endpoint"),
            direction=direction,
            aggregate_kind=PortfolioAggregateKind.SUM,
        ),
        eligible_item_ids=("a", "b", "c", "d", "e"),
        portfolio_size=2,
        selected_portfolios=(
            SelectedPortfolioClaim("method_a", "arm_m", ("a", "b")),
            SelectedPortfolioClaim("method_b", "arm_n", ("d", "e")),
        ),
    )


def _authorization(request: PostcommitRankRequest) -> PostcommitRankAuthorization:
    return PostcommitRankAuthorization(
        request_sha256=request.request_sha256,
        prerelease_file_sha256=_digest("durable-prerelease"),
        authorization_scope="generic_rank_fixture_release",
    )


def test_exact_reference_cardinality_competition_ranks_and_one_shot() -> None:
    request = _request()
    authorization = _authorization(request)
    components = {"a": -5.0, "b": -4.0, "c": -3.0, "d": -2.0, "e": -1.0}
    calls: list[str] = []

    def reader(item_id: str) -> RankReferenceObservation:
        calls.append(item_id)
        return RankReferenceObservation(
            item_id=item_id,
            endpoint_component=components[item_id],
            source_receipt_sha256=_digest(f"receipt-{item_id}"),
        )

    authority = PostcommitRankOnlyAuthority(
        request=request,
        authorization=authorization,
        reader=reader,
    )
    release = authority.release()
    validate_postcommit_rank_release_bindings(
        request=request,
        authorization=authorization,
        release=release,
    )
    assert calls == list(request.eligible_item_ids)
    assert release.exact_reference_read_count == 5
    assert release.exact_portfolio_count == 10
    ranks = {value.claim_id: value for value in release.selected_ranks}
    assert ranks["method_a:arm_m"].competition_rank == 1
    assert ranks["method_a:arm_m"].tied_portfolio_count == 1
    assert ranks["method_b:arm_n"].competition_rank == 10
    assert ranks["method_b:arm_n"].strictly_worse_count == 0
    with pytest.raises(RuntimeError, match="one-shot"):
        authority.release()
    assert calls == list(request.eligible_item_ids)


def test_failed_or_foreign_reader_burns_capability_before_first_value() -> None:
    request = _request()
    calls = 0

    def foreign(_item_id: str) -> RankReferenceObservation:
        nonlocal calls
        calls += 1
        return RankReferenceObservation(
            item_id="e",
            endpoint_component=123.0,
            source_receipt_sha256=_digest("foreign"),
        )

    authority = PostcommitRankOnlyAuthority(
        request=request,
        authorization=_authorization(request),
        reader=foreign,
    )
    with pytest.raises(PostcommitRankAuthorityError, match="another eligible item"):
        authority.release()
    assert calls == 1
    with pytest.raises(RuntimeError, match="one-shot"):
        authority.release()
    assert calls == 1


def test_public_release_cannot_carry_raw_unselected_reference_values() -> None:
    request = _request()
    authorization = _authorization(request)
    # c is absent from both selected claims.  Its distinctive private scalar
    # must influence ranks and the private digest without entering public JSON.
    components = {
        "a": -5.0,
        "b": -4.0,
        "c": 12345.6789012345,
        "d": -2.0,
        "e": -1.0,
    }

    release = PostcommitRankOnlyAuthority(
        request=request,
        authorization=authorization,
        reader=lambda item_id: RankReferenceObservation(
            item_id=item_id,
            endpoint_component=components[item_id],
            source_receipt_sha256=_digest(f"receipt-{item_id}"),
        ),
    ).release()
    record = release.to_record()
    encoded = json.dumps(record, sort_keys=True)
    assert "12345.6789012345" not in encoded
    assert components["c"].hex() not in encoded
    assert "endpoint_component" not in encoded
    assert record["raw_reference_values_returned"] is False
    assert record["unselected_item_values_returned"] is False
    assert record["unselected_portfolio_endpoints_returned"] is False


def test_request_closes_eligibility_selected_set_and_authorization_bindings() -> None:
    request = _request()
    with pytest.raises(ValueError, match="escapes"):
        PostcommitRankRequest(
            benchmark_id=request.benchmark_id,
            source_run_finalization_sha256=request.source_run_finalization_sha256,
            source_run_recursive_content_sha256=(
                request.source_run_recursive_content_sha256
            ),
            decision_commitment_sha256=request.decision_commitment_sha256,
            selected_union_commitment_sha256=(
                request.selected_union_commitment_sha256
            ),
            selected_union_release_sha256=request.selected_union_release_sha256,
            eligibility_receipt_sha256=request.eligibility_receipt_sha256,
            reference_source_sha256=request.reference_source_sha256,
            endpoint_policy=request.endpoint_policy,
            eligible_item_ids=request.eligible_item_ids,
            portfolio_size=2,
            selected_portfolios=(
                SelectedPortfolioClaim("method_a", "arm_m", ("a", "z")),
            ),
        )
    with pytest.raises(ValueError, match="another rank request"):
        PostcommitRankOnlyAuthority(
            request=request,
            authorization=PostcommitRankAuthorization(
                request_sha256=_digest("another request"),
                prerelease_file_sha256=_digest("prerelease"),
                authorization_scope="generic_rank_fixture_release",
            ),
            reader=lambda _: None,  # type: ignore[arg-type,return-value]
        )


def test_higher_is_better_uses_the_same_exact_denominator() -> None:
    request = _request(direction=RankDirection.HIGHER_IS_BETTER)
    components = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0}
    release = PostcommitRankOnlyAuthority(
        request=request,
        authorization=_authorization(request),
        reader=lambda item_id: RankReferenceObservation(
            item_id=item_id,
            endpoint_component=components[item_id],
            source_receipt_sha256=_digest(f"receipt-{item_id}"),
        ),
    ).release()
    ranks = {value.claim_id: value.competition_rank for value in release.selected_ranks}
    assert ranks == {"method_a:arm_m": 1, "method_b:arm_n": 10}
    assert release.exact_portfolio_count == 10

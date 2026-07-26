"""Thin Airfoil-v7 adapter for the generic post-commit rank-only authority."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import cast

from agent_evolve.ports.postcommit_rank_authority import (
    PortfolioAggregateKind,
    PostcommitRankAuthorization,
    PostcommitRankRequest,
    RankDirection,
    RankEndpointPolicyBinding,
    RankReferenceObservation,
    SelectedPortfolioClaim,
)
from examples.development import airfoil_v7_two_stage_agent_evolution as airfoil
from examples.development import run_airfoil_v7_v5_paired_causal_trial as trial


BENCHMARK_ID = "engibench_airfoil_v7"
AUTHORIZATION_SCOPE = "airfoil_v7_frozen_paired_selected_union_rank_only"


class AirfoilPostcommitRankError(RuntimeError):
    """The frozen Airfoil paired release differs from the rank extension."""


def endpoint_policy() -> RankEndpointPolicyBinding:
    """Bind the exact endpoint preregistered by the fresh paired trial."""

    return RankEndpointPolicyBinding(
        policy_id=trial.PRIMARY_ENDPOINT_ID,
        policy_version=trial.PRIMARY_ENDPOINT_VERSION,
        policy_definition_sha256=trial.PRIMARY_ENDPOINT_DEFINITION_SHA256,
        direction=RankDirection.LOWER_IS_BETTER,
        aggregate_kind=PortfolioAggregateKind.SUM,
    )


def _exact_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise AirfoilPostcommitRankError(f"{label} must be an exact object")
    return value


def _eligible_rows(
    commit: Mapping[str, object],
    *,
    request_field: str,
) -> tuple[tuple[str, str], ...]:
    payload = _exact_object(commit.get("payload"), "allocation commit payload")
    executions = payload.get("treatment_executions")
    if type(executions) is not list or len(executions) != 3:
        raise AirfoilPostcommitRankError("allocation commit lacks exact M/P/N wave")
    waves: list[tuple[tuple[str, str], ...]] = []
    for execution_value in executions:
        execution = _exact_object(execution_value, "allocation execution")
        request = _exact_object(execution.get(request_field), request_field)
        eligible = request.get("eligible_options")
        if type(eligible) is not list or len(eligible) != 19:
            raise AirfoilPostcommitRankError("allocation request lost 19 eligible items")
        rows: list[tuple[str, str]] = []
        for row_value in eligible:
            row = _exact_object(row_value, "eligible option")
            option_id = row.get("option_id")
            identity = row.get("option_identity_sha256")
            if type(option_id) is not str or type(identity) is not str:
                raise AirfoilPostcommitRankError("eligible option binding is malformed")
            rows.append((option_id, identity))
        canonical = tuple(sorted(rows))
        if len(set(canonical)) != 19:
            raise AirfoilPostcommitRankError("eligible option binding repeats an item")
        waves.append(canonical)
    if len(set(waves)) != 1:
        raise AirfoilPostcommitRankError("M/P/N waves differ on eligible items")
    return waves[0]


def _selected_claims(
    paired_comparison: Mapping[str, object],
) -> tuple[SelectedPortfolioClaim, ...]:
    methods = paired_comparison.get("method_receipts")
    if type(methods) is not list or len(methods) != 2:
        raise AirfoilPostcommitRankError("paired comparison lacks two methods")
    claims: list[SelectedPortfolioClaim] = []
    expected_methods = {"audited_frame_v2", "operational_frame_v3"}
    observed_methods: set[str] = set()
    for method_value in methods:
        method = _exact_object(method_value, "paired method")
        method_id = method.get("comparison_method_id")
        rows = method.get("selected_options")
        if type(method_id) is not str or type(rows) is not list or len(rows) != 9:
            raise AirfoilPostcommitRankError("paired method selection is malformed")
        observed_methods.add(method_id)
        by_treatment: dict[str, list[tuple[int, str]]] = {}
        for row_value in rows:
            row = _exact_object(row_value, "selected option")
            treatment_id = row.get("treatment_id")
            rank = row.get("rank")
            option_id = row.get("option_id")
            if (
                type(treatment_id) is not str
                or type(rank) is not int
                or type(option_id) is not str
            ):
                raise AirfoilPostcommitRankError("selected option row is malformed")
            by_treatment.setdefault(treatment_id, []).append((rank, option_id))
        if set(by_treatment) != {"m", "p", "n"}:
            raise AirfoilPostcommitRankError("paired method lacks exact M/P/N arms")
        for treatment_id, selections in by_treatment.items():
            ordered = tuple(sorted(selections))
            if tuple(value[0] for value in ordered) != (1, 2, 3):
                raise AirfoilPostcommitRankError("allocator ranks changed")
            claims.append(
                SelectedPortfolioClaim(
                    method_id=method_id,
                    treatment_id=treatment_id,
                    item_ids=tuple(sorted(value[1] for value in ordered)),
                )
            )
    if observed_methods != expected_methods:
        raise AirfoilPostcommitRankError("paired method IDs changed")
    return tuple(sorted(claims, key=lambda value: (value.method_id, value.treatment_id)))


def build_rank_request(
    *,
    source_finalization: Mapping[str, object],
    paired_comparison: Mapping[str, object],
    airfoil_commitment: Mapping[str, object],
    selected_union_release_sha256: str,
    v2_commit: Mapping[str, object],
    v3_commit: Mapping[str, object],
    oracle: airfoil.VerifiedAirfoilPredecisionOracle,
) -> PostcommitRankRequest:
    """Project exact finalized Airfoil records into the benchmark-neutral port."""

    if type(oracle) is not airfoil.VerifiedAirfoilPredecisionOracle:
        raise TypeError("oracle must be an exact verified predecision seal")
    finalization_sha256 = source_finalization.get("finalization_sha256")
    recursive_sha256 = source_finalization.get("recursive_content_sha256")
    paired_sha256 = paired_comparison.get("commitment_sha256")
    airfoil_sha256 = airfoil_commitment.get("commitment_sha256")
    if any(
        type(value) is not str
        for value in (
            finalization_sha256,
            recursive_sha256,
            paired_sha256,
            airfoil_sha256,
        )
    ):
        raise AirfoilPostcommitRankError("source commitment digest is malformed")
    if (
        airfoil_commitment.get("paired_comparison_commitment_sha256")
        != paired_sha256
        or airfoil_commitment.get("raw_outcome_authority")
        != "selected_union_only"
        or airfoil_commitment.get("unselected_outcomes_exposed") is not False
    ):
        raise AirfoilPostcommitRankError("Airfoil selected-union commitment changed")

    v2_rows = _eligible_rows(v2_commit, request_field="allocation_request")
    v3_rows = _eligible_rows(v3_commit, request_field="base_allocation_request")
    if v2_rows != v3_rows:
        raise AirfoilPostcommitRankError("v2/v3 eligible item bindings differ")
    for option_id, identity_sha256 in v2_rows:
        option = oracle.contract.resolve(option_id)
        if option.identity_sha256 != identity_sha256:
            raise AirfoilPostcommitRankError(
                "allocation eligible identity differs from sealed contract"
            )
    v2_payload = _exact_object(v2_commit.get("payload"), "v2 payload")
    v2_executions = cast(list[object], v2_payload["treatment_executions"])
    first_execution = _exact_object(v2_executions[0], "v2 execution")
    allocation_request = _exact_object(
        first_execution.get("allocation_request"),
        "v2 allocation request",
    )
    eligibility_sha256 = allocation_request.get("eligible_options_sha256")
    if type(eligibility_sha256) is not str:
        raise AirfoilPostcommitRankError("eligible item receipt is malformed")

    claims = _selected_claims(paired_comparison)
    eligible_ids = tuple(value[0] for value in v2_rows)
    if not all(set(value.item_ids).issubset(eligible_ids) for value in claims):
        raise AirfoilPostcommitRankError("selected claim escapes eligible items")
    selected_union = tuple(
        sorted({item for claim in claims for item in claim.item_ids})
    )
    if airfoil_commitment.get("selected_option_ids") != list(selected_union):
        raise AirfoilPostcommitRankError("selected claims differ from committed union")

    return PostcommitRankRequest(
        benchmark_id=BENCHMARK_ID,
        source_run_finalization_sha256=str(finalization_sha256),
        source_run_recursive_content_sha256=str(recursive_sha256),
        decision_commitment_sha256=str(paired_sha256),
        selected_union_commitment_sha256=str(airfoil_sha256),
        selected_union_release_sha256=selected_union_release_sha256,
        eligibility_receipt_sha256=eligibility_sha256,
        reference_source_sha256=oracle.recursive_content_sha256,
        endpoint_policy=endpoint_policy(),
        eligible_item_ids=eligible_ids,
        portfolio_size=3,
        selected_portfolios=claims,
    )


class AirfoilRankReferenceReader:
    """Private one-read-per-item adapter; raw Airfoil outcomes never escape it."""

    def __init__(
        self,
        *,
        request: PostcommitRankRequest,
        authorization: PostcommitRankAuthorization,
        oracle: airfoil.VerifiedAirfoilPredecisionOracle,
    ) -> None:
        if type(request) is not PostcommitRankRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if (
            request.benchmark_id != BENCHMARK_ID
            or request.endpoint_policy.to_record() != endpoint_policy().to_record()
        ):
            raise AirfoilPostcommitRankError("rank request names another benchmark policy")
        if type(authorization) is not PostcommitRankAuthorization:
            raise TypeError("authorization must be exact")
        authorization.__post_init__()
        if authorization.request_sha256 != request.request_sha256:
            raise AirfoilPostcommitRankError("authorization names another request")
        if authorization.authorization_scope != AUTHORIZATION_SCOPE:
            raise AirfoilPostcommitRankError("authorization scope changed")
        if type(oracle) is not airfoil.VerifiedAirfoilPredecisionOracle:
            raise TypeError("oracle must be exact")
        if oracle.recursive_content_sha256 != request.reference_source_sha256:
            raise AirfoilPostcommitRankError("rank request names another oracle seal")
        self._request = request
        evaluator = airfoil.AirfoilV7SealedOracleDevelopmentEvaluator(oracle)
        self._capability = evaluator.open_postcommit_rank_reference(
            request=request,
            authorization=authorization,
        )
        self._read_item_ids: set[str] = set()
        self._source_receipts: list[str] = []

    @property
    def exact_read_count(self) -> int:
        return len(self._read_item_ids)

    def __call__(self, item_id: str) -> RankReferenceObservation:
        if type(item_id) is not str or item_id not in self._request.eligible_item_ids:
            raise PermissionError("rank reader item is outside the exact eligible set")
        if item_id in self._read_item_ids:
            raise RuntimeError("rank reader cannot decode an item twice")
        # Burn the item before touching the cached terminal so a failed decode
        # cannot be retried after observing a prefix of its contents.
        self._read_item_ids.add(item_id)
        def project(evaluation: airfoil.AirfoilDevelopmentEvaluation) -> float:
            metrics = {value.metric_id: value for value in evaluation.metrics}
            if set(metrics) != set(airfoil.REQUIRED_METRIC_IDS):
                raise AirfoilPostcommitRankError(
                    "reference item lacks exact metrics"
                )
            return trial.member_log_failure(
                delta_f=metrics[airfoil.OBJECTIVE_METRIC_ID].delta,
                delta_v=metrics[airfoil.VIOLATION_METRIC_ID].delta,
            )

        observation = self._capability.evaluate_component(item_id, project)
        self._source_receipts.append(observation.source_receipt_sha256)
        return observation

    def audit_record(self) -> dict[str, object]:
        if self.exact_read_count != self._request.reference_read_count:
            raise RuntimeError("rank reader has not completed its exact read set")
        digest = hashlib.sha256(
            b"agent-evolve:airfoil-v7-rank-terminal-receipts:v1\x00"
            + "".join(self._source_receipts).encode("ascii")
        ).hexdigest()
        return {
            "schema_version": 1,
            "exact_eligible_cached_terminal_decode_count": self.exact_read_count,
            "eligible_item_count": len(self._request.eligible_item_ids),
            "terminal_receipt_sequence_sha256": digest,
            "raw_unselected_outcomes_returned": False,
            "new_cfd_calls": 0,
            "provider_calls": 0,
        }


__all__ = [
    "AUTHORIZATION_SCOPE",
    "AirfoilPostcommitRankError",
    "AirfoilRankReferenceReader",
    "BENCHMARK_ID",
    "build_rank_request",
    "endpoint_policy",
]

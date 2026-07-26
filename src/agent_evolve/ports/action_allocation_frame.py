"""Authenticated forecast frames and audited allocation-surface ports.

The original allocation port deliberately accepts only a complete resolved
forecast batch.  A partition block must not be relabelled as such a batch just
to reach an allocator.  This module provides a parallel, provider-neutral
boundary whose source remains an exact complete batch, partition block, or
authenticated subset of one partition block.

No benchmark metric or provider representation is interpreted here.  A bound
diagnostic policy supplies the single domain-dependent judgement used by the
surface gate: whether one candidate score is at a benchmark-defined boundary
or extreme.  Counts, ties, gaps, and score-multiset identities remain trusted
generic arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_allocation import (
    AllocatedActionMember,
    ForecastPortfolioUtilityBinding,
    PortfolioAllocationScore,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockRequest,
    ActionForecastRequest,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionForecastBlock,
    validate_resolved_action_forecast_block,
    validate_resolved_action_forecasts,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_CANDIDATE_LABEL = re.compile(r"^row_[0-9]{8}$")
_FRAME_DOMAIN = b"agent-evolve:resolved-action-allocation-frame:v1\x00"
_SUBSET_POLICY_DOMAIN = b"agent-evolve:allocation-frame-subset-policy:v1\x00"
_ELIGIBLE_DOMAIN = b"agent-evolve:frame-allocation-eligible-set:v1\x00"
_REQUEST_DOMAIN = b"agent-evolve:frame-action-allocation-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:frame-action-portfolio-decision:v1\x00"
_DIAGNOSTIC_POLICY_DOMAIN = (
    b"agent-evolve:allocation-score-diagnostic-policy:v1\x00"
)
_GATE_POLICY_DOMAIN = b"agent-evolve:allocation-surface-gate-policy:v1\x00"
_SCORE_MULTISET_DOMAIN = b"agent-evolve:allocation-score-multiset:v1\x00"
_STEP_AUDIT_DOMAIN = b"agent-evolve:allocation-surface-step-audit:v1\x00"
_SURFACE_AUDIT_DOMAIN = b"agent-evolve:allocation-surface-audit:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:audited-frame-allocation-result:v1\x00"


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
        raise ValueError(f"{name} must use the closed token grammar")
    return value


def _canonical_parent_receipts(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError("parent_receipt_sha256s must be an exact digest tuple")
    for index, value in enumerate(values):
        require_sha256(value, f"parent_receipt_sha256s[{index}]")
    if values != tuple(sorted(set(values))):
        raise ValueError("parent receipts must be unique and canonically sorted")


class ActionForecastAllocationFrameKind(str, Enum):
    COMPLETE = "complete"
    PARTITION_BLOCK = "partition_block"
    PARTITION_BLOCK_SUBSET = "partition_block_subset"


@dataclass(frozen=True, slots=True, eq=False)
class ActionAllocationFrameSubsetPolicyBinding:
    """Identified rule selecting allocation rows from one resolved block."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_SUBSET_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionAllocationFrameSubsetPolicyBinding
            and type(other) is ActionAllocationFrameSubsetPolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedActionForecastAllocationFrame:
    """Exact allocation view over a complete batch, block, or block subset.

    Source objects are retained so public validation can authenticate forecast
    values rather than trusting copied IDs.  The durable record stores only the
    source receipt and exact ordered row/option identities.
    """

    request: ActionForecastRequest = field(repr=False, compare=False)
    frame_kind: ActionForecastAllocationFrameKind
    global_row_indices: tuple[int, ...]
    parent_receipt_sha256s: tuple[str, ...]
    complete_batch: ResolvedActionForecastBatch | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    block_request: ActionForecastBlockRequest | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    resolved_block: ResolvedActionForecastBlock | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    subset_policy: ActionAllocationFrameSubsetPolicyBinding | None = None
    _validated_receipt_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._validated_receipt_sha256 is not None:
            require_sha256(
                self._validated_receipt_sha256,
                "_validated_receipt_sha256",
            )
            return
        if type(self.request) is not ActionForecastRequest:
            raise TypeError("request must be an exact ActionForecastRequest")
        self.request.__post_init__()
        if type(self.frame_kind) is not ActionForecastAllocationFrameKind:
            raise TypeError("frame_kind must be an exact allocation-frame kind")
        if type(self.global_row_indices) is not tuple or not self.global_row_indices:
            raise ValueError("global_row_indices must be a non-empty exact tuple")
        if any(type(value) is not int for value in self.global_row_indices):
            raise TypeError("global_row_indices must contain exact integers")
        if self.global_row_indices != tuple(sorted(set(self.global_row_indices))):
            raise ValueError("global rows must be unique and in global order")
        contract = self.request.finite_variation_contract
        if any(value < 0 or value >= len(contract.options) for value in self.global_row_indices):
            raise ValueError("an allocation-frame row is outside the finite contract")
        _canonical_parent_receipts(self.parent_receipt_sha256s)

        if self.frame_kind is ActionForecastAllocationFrameKind.COMPLETE:
            if type(self.complete_batch) is not ResolvedActionForecastBatch:
                raise TypeError("complete frames require an exact resolved batch")
            if self.block_request is not None or self.resolved_block is not None:
                raise ValueError("complete frames forbid partition-block sources")
            if self.subset_policy is not None:
                raise ValueError("complete frames forbid a subset policy")
            validate_resolved_action_forecasts(self.request, self.complete_batch)
            if self.global_row_indices != tuple(range(len(contract.options))):
                raise ValueError("complete frames must cover every global row")
        else:
            if self.complete_batch is not None:
                raise ValueError("partition frames forbid a complete batch")
            if type(self.block_request) is not ActionForecastBlockRequest:
                raise TypeError("partition frames require an exact block request")
            if type(self.resolved_block) is not ResolvedActionForecastBlock:
                raise TypeError("partition frames require an exact resolved block")
            if self.block_request.request != self.request:
                raise ValueError("block request carries another forecast request")
            validate_resolved_action_forecast_block(
                self.block_request,
                self.resolved_block,
            )
            spec = self.block_request.block
            full_indices = tuple(range(spec.global_row_start, spec.global_row_stop))
            if self.frame_kind is ActionForecastAllocationFrameKind.PARTITION_BLOCK:
                if self.global_row_indices != full_indices:
                    raise ValueError("full block frame differs from its block rows")
                if self.subset_policy is not None:
                    raise ValueError("full block frames forbid a subset policy")
            else:
                if type(self.subset_policy) is not (
                    ActionAllocationFrameSubsetPolicyBinding
                ):
                    raise TypeError("block-subset frames require a subset policy")
                self.subset_policy.__post_init__()
                if not self.parent_receipt_sha256s:
                    raise ValueError(
                        "block-subset frames require an authenticated parent receipt"
                    )
                if not set(self.global_row_indices).issubset(full_indices):
                    raise ValueError("block-subset rows escape the resolved block")

        expected = tuple(contract.options[index] for index in self.global_row_indices)
        observed = self.forecasts
        if len(observed) != len(expected):
            raise ValueError("frame forecasts differ from its exact row count")
        for option, forecast in zip(expected, observed, strict=True):
            if (
                forecast.option_id != option.option_id
                or forecast.option_identity_sha256 != option.identity_sha256
                or forecast.child_configuration_sha256
                != option.child_configuration_sha256
                or forecast.family != option.family
            ):
                raise ValueError("allocation frame changed a finite option identity")
        object.__setattr__(
            self,
            "_validated_receipt_sha256",
            _hash(_FRAME_DOMAIN, self._record_from_validated_values()),
        )

    @property
    def forecasts(self) -> tuple[ResolvedActionForecast, ...]:
        if self.frame_kind is ActionForecastAllocationFrameKind.COMPLETE:
            assert self.complete_batch is not None
            return self.complete_batch.forecasts
        assert self.block_request is not None
        assert self.resolved_block is not None
        start = self.block_request.block.global_row_start
        return tuple(
            self.resolved_block.forecasts[index - start]
            for index in self.global_row_indices
        )

    @property
    def source_forecast_receipt_sha256(self) -> str:
        if self.frame_kind is ActionForecastAllocationFrameKind.COMPLETE:
            assert self.complete_batch is not None
            return self.complete_batch.receipt_sha256
        assert self.resolved_block is not None
        return self.resolved_block.receipt_sha256

    @property
    def policy_identity(self) -> tuple[str, int, str]:
        source = self.complete_batch or self.resolved_block
        assert source is not None
        return (
            source.policy_id,
            source.policy_version,
            source.policy_definition_sha256,
        )

    def _record_from_validated_values(self) -> dict[str, object]:
        contract = self.request.finite_variation_contract
        policy_id, policy_version, policy_definition_sha256 = self.policy_identity
        return {
            "schema_version": 1,
            "frame_kind": self.frame_kind.value,
            "request_sha256": self.request.request_sha256,
            "context_sha256": self.request.context_sha256,
            "finite_contract_identity_sha256": contract.identity_sha256,
            "source_forecast_receipt_sha256": (
                self.source_forecast_receipt_sha256
            ),
            "parent_receipt_sha256s": list(self.parent_receipt_sha256s),
            "global_row_indices": list(self.global_row_indices),
            "options": [
                {
                    "global_row_index": index,
                    "option_id": option.option_id,
                    "option_identity_sha256": option.identity_sha256,
                    "child_configuration_sha256": (
                        option.child_configuration_sha256
                    ),
                    "family": option.family,
                }
                for index, option in zip(
                    self.global_row_indices,
                    (contract.options[value] for value in self.global_row_indices),
                    strict=True,
                )
            ],
            "forecast_policy": {
                "policy_id": policy_id,
                "policy_version": policy_version,
                "policy_definition_sha256": policy_definition_sha256,
            },
            "partition": (
                None
                if self.block_request is None
                else {
                    "layout_sha256": self.block_request.layout.layout_sha256,
                    "block_request_sha256": (
                        self.block_request.block_request_sha256
                    ),
                    "block_spec_sha256": (
                        self.block_request.block.block_spec_sha256
                    ),
                    "block_index": self.block_request.block.block_index,
                    "global_row_start": (
                        self.block_request.block.global_row_start
                    ),
                    "global_row_stop": self.block_request.block.global_row_stop,
                }
            ),
            "subset_policy": (
                None
                if self.subset_policy is None
                else self.subset_policy.to_record()
            ),
        }

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return self._record_from_validated_values()

    @property
    def receipt_sha256(self) -> str:
        self.__post_init__()
        assert self._validated_receipt_sha256 is not None
        return self._validated_receipt_sha256

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ResolvedActionForecastAllocationFrame
            and type(other) is ResolvedActionForecastAllocationFrame
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def bind_complete_action_forecast_allocation_frame(
    request: ActionForecastRequest,
    forecasts: ResolvedActionForecastBatch,
    *,
    parent_receipt_sha256s: tuple[str, ...] = (),
) -> ResolvedActionForecastAllocationFrame:
    return ResolvedActionForecastAllocationFrame(
        request=request,
        frame_kind=ActionForecastAllocationFrameKind.COMPLETE,
        global_row_indices=tuple(range(len(request.finite_variation_contract.options))),
        parent_receipt_sha256s=parent_receipt_sha256s,
        complete_batch=forecasts,
    )


def bind_action_forecast_block_allocation_frame(
    block_request: ActionForecastBlockRequest,
    forecasts: ResolvedActionForecastBlock,
    *,
    parent_receipt_sha256s: tuple[str, ...] = (),
) -> ResolvedActionForecastAllocationFrame:
    spec = block_request.block
    return ResolvedActionForecastAllocationFrame(
        request=block_request.request,
        frame_kind=ActionForecastAllocationFrameKind.PARTITION_BLOCK,
        global_row_indices=tuple(range(spec.global_row_start, spec.global_row_stop)),
        parent_receipt_sha256s=parent_receipt_sha256s,
        block_request=block_request,
        resolved_block=forecasts,
    )


def bind_action_forecast_block_subset_allocation_frame(
    block_request: ActionForecastBlockRequest,
    forecasts: ResolvedActionForecastBlock,
    *,
    included_global_row_indices: tuple[int, ...],
    subset_policy: ActionAllocationFrameSubsetPolicyBinding,
    parent_receipt_sha256s: tuple[str, ...],
) -> ResolvedActionForecastAllocationFrame:
    return ResolvedActionForecastAllocationFrame(
        request=block_request.request,
        frame_kind=ActionForecastAllocationFrameKind.PARTITION_BLOCK_SUBSET,
        global_row_indices=included_global_row_indices,
        parent_receipt_sha256s=parent_receipt_sha256s,
        block_request=block_request,
        resolved_block=forecasts,
        subset_policy=subset_policy,
    )


@dataclass(frozen=True, slots=True, eq=False)
class FrameActionAllocationRequest:
    frame: ResolvedActionForecastAllocationFrame
    eligible_option_ids: tuple[str, ...]
    portfolio_size: int
    utility: ForecastPortfolioUtilityBinding
    _eligible_options_sha256_cache: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _request_sha256_cache: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self._eligible_options_sha256_cache is not None
            and self._request_sha256_cache is not None
        ):
            require_sha256(
                self._eligible_options_sha256_cache,
                "_eligible_options_sha256_cache",
            )
            require_sha256(
                self._request_sha256_cache,
                "_request_sha256_cache",
            )
            return
        if type(self.frame) is not ResolvedActionForecastAllocationFrame:
            raise TypeError("frame must be an exact allocation frame")
        self.frame.__post_init__()
        if type(self.eligible_option_ids) is not tuple or any(
            type(value) is not str or _OPTION_ID.fullmatch(value) is None
            for value in self.eligible_option_ids
        ):
            raise TypeError("eligible_option_ids must be an exact option-ID tuple")
        if not self.eligible_option_ids:
            raise ValueError("eligible_option_ids must be non-empty")
        if self.eligible_option_ids != tuple(sorted(set(self.eligible_option_ids))):
            raise ValueError("eligible option IDs must be unique and canonical")
        available = {value.option_id for value in self.frame.forecasts}
        if not set(self.eligible_option_ids).issubset(available):
            raise ValueError("eligible option IDs escape the authenticated frame")
        if type(self.portfolio_size) is not int or self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be a positive exact integer")
        if self.portfolio_size > len(self.eligible_option_ids):
            raise ValueError("portfolio_size exceeds the eligible frame")
        if type(self.utility) is not ForecastPortfolioUtilityBinding:
            raise TypeError("utility must be an exact identified binding")
        self.utility.__post_init__()
        eligible_options_sha256 = self._eligible_digest_from_validated_values()
        object.__setattr__(
            self,
            "_eligible_options_sha256_cache",
            eligible_options_sha256,
        )
        object.__setattr__(
            self,
            "_request_sha256_cache",
            _hash(
                _REQUEST_DOMAIN,
                self._record_from_validated_values(eligible_options_sha256),
            ),
        )

    def _eligible_digest_from_validated_values(self) -> str:
        by_id = {value.option_id: value for value in self.frame.forecasts}
        return _hash(
            _ELIGIBLE_DOMAIN,
            {
                "frame_receipt_sha256": self.frame.receipt_sha256,
                "options": [
                    {
                        "option_id": option_id,
                        "option_identity_sha256": by_id[
                            option_id
                        ].option_identity_sha256,
                    }
                    for option_id in self.eligible_option_ids
                ],
            },
        )

    @property
    def eligible_options_sha256(self) -> str:
        self.__post_init__()
        assert self._eligible_options_sha256_cache is not None
        return self._eligible_options_sha256_cache

    def _record_from_validated_values(
        self,
        eligible_options_sha256: str,
    ) -> dict[str, object]:
        by_id = {value.option_id: value for value in self.frame.forecasts}
        return {
            "schema_version": 1,
            "frame_receipt_sha256": self.frame.receipt_sha256,
            "source_forecast_receipt_sha256": (
                self.frame.source_forecast_receipt_sha256
            ),
            "eligible_options": [
                {
                    "option_id": option_id,
                    "option_identity_sha256": by_id[
                        option_id
                    ].option_identity_sha256,
                }
                for option_id in self.eligible_option_ids
            ],
            "eligible_options_sha256": eligible_options_sha256,
            "portfolio_size": self.portfolio_size,
            "utility": self.utility.to_record(),
        }

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        assert self._eligible_options_sha256_cache is not None
        return self._record_from_validated_values(
            self._eligible_options_sha256_cache
        )

    @property
    def request_sha256(self) -> str:
        self.__post_init__()
        assert self._request_sha256_cache is not None
        return self._request_sha256_cache

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is FrameActionAllocationRequest
            and type(other) is FrameActionAllocationRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class FrameActionPortfolioDecision:
    allocation_request_sha256: str
    frame_receipt_sha256: str
    source_forecast_receipt_sha256: str
    eligible_options_sha256: str
    members: tuple[AllocatedActionMember, ...]
    final_score: PortfolioAllocationScore
    candidate_evaluations: int
    utility_policy_id: str
    utility_policy_version: int
    utility_definition_sha256: str
    allocator_policy_id: str
    allocator_policy_version: int
    allocator_definition_sha256: str
    allocator_configuration_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "allocation_request_sha256",
            "frame_receipt_sha256",
            "source_forecast_receipt_sha256",
            "eligible_options_sha256",
            "utility_definition_sha256",
            "allocator_definition_sha256",
            "allocator_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.members) is not tuple or not self.members or any(
            type(value) is not AllocatedActionMember for value in self.members
        ):
            raise ValueError("members must be a non-empty exact allocated tuple")
        for value in self.members:
            value.__post_init__()
        if tuple(value.rank for value in self.members) != tuple(
            range(1, len(self.members) + 1)
        ):
            raise ValueError("member ranks must be contiguous and ordered")
        if len({value.option_id for value in self.members}) != len(self.members):
            raise ValueError("a frame decision cannot repeat an option")
        if type(self.final_score) is not PortfolioAllocationScore:
            raise TypeError("final_score must be exact PortfolioAllocationScore")
        self.final_score.__post_init__()
        if self.final_score != self.members[-1].greedy_step_score:
            raise ValueError("final_score must equal the last greedy score")
        if type(self.candidate_evaluations) is not int or self.candidate_evaluations <= 0:
            raise ValueError("candidate_evaluations must be positive")
        for prefix in ("utility", "allocator"):
            _token(getattr(self, f"{prefix}_policy_id"), f"{prefix}_policy_id")
            version = getattr(self, f"{prefix}_policy_version")
            if type(version) is not int or version <= 0:
                raise ValueError(f"{prefix}_policy_version must be positive")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "allocation_request_sha256": self.allocation_request_sha256,
            "frame_receipt_sha256": self.frame_receipt_sha256,
            "source_forecast_receipt_sha256": self.source_forecast_receipt_sha256,
            "eligible_options_sha256": self.eligible_options_sha256,
            "members": [value.to_record() for value in self.members],
            "final_score": self.final_score.to_record(),
            "candidate_evaluations": self.candidate_evaluations,
            "utility_policy": {
                "policy_id": self.utility_policy_id,
                "policy_version": self.utility_policy_version,
                "definition_sha256": self.utility_definition_sha256,
            },
            "allocator_policy": {
                "policy_id": self.allocator_policy_id,
                "policy_version": self.allocator_policy_version,
                "definition_sha256": self.allocator_definition_sha256,
                "configuration_sha256": self.allocator_configuration_sha256,
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_DECISION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is FrameActionPortfolioDecision
            and type(other) is FrameActionPortfolioDecision
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def validate_frame_action_portfolio_decision(
    request: FrameActionAllocationRequest,
    decision: FrameActionPortfolioDecision,
) -> None:
    if type(request) is not FrameActionAllocationRequest:
        raise TypeError("request must be an exact frame allocation request")
    request.__post_init__()
    if type(decision) is not FrameActionPortfolioDecision:
        raise TypeError("decision must be an exact frame portfolio decision")
    decision.__post_init__()
    if (
        decision.allocation_request_sha256 != request.request_sha256
        or decision.frame_receipt_sha256 != request.frame.receipt_sha256
        or decision.source_forecast_receipt_sha256
        != request.frame.source_forecast_receipt_sha256
        or decision.eligible_options_sha256 != request.eligible_options_sha256
    ):
        raise ValueError("frame decision is bound to another request")
    if len(decision.members) != request.portfolio_size:
        raise ValueError("frame decision member count differs from portfolio_size")
    if (
        decision.utility_policy_id != request.utility.policy_id
        or decision.utility_policy_version != request.utility.policy_version
        or decision.utility_definition_sha256 != request.utility.definition_sha256
    ):
        raise ValueError("frame decision names another utility policy")
    forecasts = {value.option_id: value for value in request.frame.forecasts}
    for member in decision.members:
        if member.option_id not in request.eligible_option_ids:
            raise ValueError("frame decision selected an ineligible option")
        forecast = forecasts[member.option_id]
        if (
            member.option_identity_sha256 != forecast.option_identity_sha256
            or member.child_configuration_sha256
            != forecast.child_configuration_sha256
            or member.family != forecast.family
        ):
            raise ValueError("frame decision member differs from its forecast")


@dataclass(frozen=True, slots=True)
class AllocationCandidateScoreDiagnosticInput:
    allocation_request_sha256: str
    step: int
    candidate_label: str
    members: tuple[ResolvedActionForecast, ...]
    score: PortfolioAllocationScore
    marginal_total_utility: float

    def __post_init__(self) -> None:
        require_sha256(self.allocation_request_sha256, "allocation_request_sha256")
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("step must be a positive exact integer")
        if type(self.candidate_label) is not str or _CANDIDATE_LABEL.fullmatch(
            self.candidate_label
        ) is None:
            raise ValueError("candidate_label must be an opaque global-row label")
        if type(self.members) is not tuple or not self.members or any(
            type(value) is not ResolvedActionForecast for value in self.members
        ):
            raise ValueError("members must be a non-empty resolved forecast tuple")
        for value in self.members:
            value.__post_init__()
        if type(self.score) is not PortfolioAllocationScore:
            raise TypeError("score must be exact PortfolioAllocationScore")
        self.score.__post_init__()
        _finite_float(self.marginal_total_utility, "marginal_total_utility")


@dataclass(frozen=True, slots=True)
class AllocationCandidateScoreDiagnostic:
    boundary_or_extreme: bool

    def __post_init__(self) -> None:
        if type(self.boundary_or_extreme) is not bool:
            raise TypeError("boundary_or_extreme must be an exact bool")


@runtime_checkable
class AllocationScoreDiagnostic(Protocol):
    def __call__(
        self,
        request: AllocationCandidateScoreDiagnosticInput,
    ) -> AllocationCandidateScoreDiagnostic: ...


@dataclass(frozen=True, slots=True)
class AllocationScoreDiagnosticBinding:
    diagnostic: AllocationScoreDiagnostic = field(repr=False, compare=False)
    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.diagnostic):
            raise TypeError("diagnostic must be callable")
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_DIAGNOSTIC_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}


@dataclass(frozen=True, slots=True, eq=False)
class AllocationSurfaceGatePolicyBinding:
    """Domain-neutral thresholds applied before evaluator authority opens."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    minimum_distinct_finite_scores: int
    maximum_top_tie_share: float
    maximum_boundary_or_extreme_share: float
    minimum_winner_runner_gap: float

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if (
            type(self.minimum_distinct_finite_scores) is not int
            or self.minimum_distinct_finite_scores <= 0
        ):
            raise ValueError("minimum_distinct_finite_scores must be positive")
        for name in (
            "maximum_top_tie_share",
            "maximum_boundary_or_extreme_share",
        ):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        _finite_float(self.minimum_winner_runner_gap, "minimum_winner_runner_gap")
        if self.minimum_winner_runner_gap < 0.0:
            raise ValueError("minimum_winner_runner_gap cannot be negative")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "minimum_distinct_finite_scores": (
                self.minimum_distinct_finite_scores
            ),
            "maximum_top_tie_share_hex": self.maximum_top_tie_share.hex(),
            "maximum_boundary_or_extreme_share_hex": (
                self.maximum_boundary_or_extreme_share.hex()
            ),
            "minimum_winner_runner_gap_hex": (
                self.minimum_winner_runner_gap.hex()
            ),
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_GATE_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is AllocationSurfaceGatePolicyBinding
            and type(other) is AllocationSurfaceGatePolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


def allocation_score_multiset_sha256(scores: tuple[float, ...]) -> str:
    if type(scores) is not tuple or not scores:
        raise ValueError("scores must be a non-empty exact tuple")
    for index, score in enumerate(scores):
        _finite_float(score, f"scores[{index}]")
    return _hash(
        _SCORE_MULTISET_DOMAIN,
        {"score_hex_multiset": sorted(value.hex() for value in scores)},
    )


@dataclass(frozen=True, slots=True, eq=False)
class AllocationSurfaceStepAudit:
    step: int
    candidate_count: int
    distinct_finite_score_count: int
    top_tie_count: int
    winner_runner_gap: float
    boundary_or_extreme_count: int
    boundary_or_extreme_share: float
    score_multiset_sha256: str
    winner_candidate_label: str
    tie_break_used: bool
    failure_codes: tuple[str, ...]
    passes: bool

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("step must be a positive exact integer")
        for name in (
            "candidate_count",
            "distinct_finite_score_count",
            "top_tie_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
            if value > self.candidate_count:
                raise ValueError(f"{name} cannot exceed candidate_count")
        if (
            type(self.boundary_or_extreme_count) is not int
            or not 0 <= self.boundary_or_extreme_count <= self.candidate_count
        ):
            raise ValueError("boundary_or_extreme_count is outside candidate_count")
        _finite_float(self.winner_runner_gap, "winner_runner_gap")
        if self.winner_runner_gap < 0.0:
            raise ValueError("winner_runner_gap cannot be negative")
        _finite_float(
            self.boundary_or_extreme_share,
            "boundary_or_extreme_share",
        )
        if not 0.0 <= self.boundary_or_extreme_share <= 1.0:
            raise ValueError("boundary_or_extreme_share must lie in [0,1]")
        expected_share = self.boundary_or_extreme_count / self.candidate_count
        if self.boundary_or_extreme_share != expected_share:
            raise ValueError("boundary/extreme share differs from its count")
        require_sha256(self.score_multiset_sha256, "score_multiset_sha256")
        if type(self.winner_candidate_label) is not str or _CANDIDATE_LABEL.fullmatch(
            self.winner_candidate_label
        ) is None:
            raise ValueError("winner_candidate_label is not an opaque row label")
        if type(self.tie_break_used) is not bool or type(self.passes) is not bool:
            raise TypeError("tie_break_used and passes must be exact booleans")
        if self.tie_break_used is not (self.top_tie_count > 1):
            raise ValueError("tie_break_used differs from top_tie_count")
        if type(self.failure_codes) is not tuple or any(
            type(value) is not str or _TOKEN.fullmatch(value) is None
            for value in self.failure_codes
        ):
            raise TypeError("failure_codes must be an exact token tuple")
        if self.failure_codes != tuple(sorted(set(self.failure_codes))):
            raise ValueError("failure_codes must be unique and canonical")
        if self.passes is not (not self.failure_codes):
            raise ValueError("passes differs from failure_codes")

    @property
    def top_tie_share(self) -> float:
        return self.top_tie_count / self.candidate_count

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "step": self.step,
            "candidate_count": self.candidate_count,
            "score_kind": "marginal_total_utility",
            "distinct_finite_score_count": self.distinct_finite_score_count,
            "top_tie_count": self.top_tie_count,
            "top_tie_share_hex": self.top_tie_share.hex(),
            "winner_runner_gap_hex": self.winner_runner_gap.hex(),
            "boundary_or_extreme_count": self.boundary_or_extreme_count,
            "boundary_or_extreme_share_hex": (
                self.boundary_or_extreme_share.hex()
            ),
            "score_multiset_sha256": self.score_multiset_sha256,
            "winner_candidate_label": self.winner_candidate_label,
            "tie_break_used": self.tie_break_used,
            "failure_codes": list(self.failure_codes),
            "passes": self.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_STEP_AUDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is AllocationSurfaceStepAudit
            and type(other) is AllocationSurfaceStepAudit
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def allocation_surface_failure_codes(
    *,
    policy: AllocationSurfaceGatePolicyBinding,
    candidate_count: int,
    distinct_finite_score_count: int,
    top_tie_count: int,
    winner_runner_gap: float,
    boundary_or_extreme_share: float,
) -> tuple[str, ...]:
    policy.__post_init__()
    failures: list[str] = []
    if distinct_finite_score_count < policy.minimum_distinct_finite_scores:
        failures.append("insufficient_distinct_scores")
    if top_tie_count / candidate_count > policy.maximum_top_tie_share:
        failures.append("top_tie_concentration")
    if (
        boundary_or_extreme_share
        > policy.maximum_boundary_or_extreme_share
    ):
        failures.append("boundary_extreme_concentration")
    if candidate_count > 1 and winner_runner_gap < policy.minimum_winner_runner_gap:
        failures.append("winner_runner_gap_too_small")
    return tuple(sorted(failures))


@dataclass(frozen=True, slots=True, eq=False)
class ActionAllocationSurfaceAudit:
    allocation_request_sha256: str
    decision_receipt_sha256: str
    frame_receipt_sha256: str
    score_diagnostic: AllocationScoreDiagnosticBinding
    gate_policy: AllocationSurfaceGatePolicyBinding
    steps: tuple[AllocationSurfaceStepAudit, ...]
    candidate_score_count: int
    passes: bool

    def __post_init__(self) -> None:
        for name in (
            "allocation_request_sha256",
            "decision_receipt_sha256",
            "frame_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.score_diagnostic) is not AllocationScoreDiagnosticBinding:
            raise TypeError("score_diagnostic must be an exact binding")
        self.score_diagnostic.__post_init__()
        if type(self.gate_policy) is not AllocationSurfaceGatePolicyBinding:
            raise TypeError("gate_policy must be an exact binding")
        self.gate_policy.__post_init__()
        if type(self.steps) is not tuple or not self.steps or any(
            type(value) is not AllocationSurfaceStepAudit for value in self.steps
        ):
            raise ValueError("steps must be a non-empty exact audit tuple")
        for value in self.steps:
            value.__post_init__()
        if tuple(value.step for value in self.steps) != tuple(
            range(1, len(self.steps) + 1)
        ):
            raise ValueError("audit steps must be contiguous and ordered")
        for step in self.steps:
            expected_failures = allocation_surface_failure_codes(
                policy=self.gate_policy,
                candidate_count=step.candidate_count,
                distinct_finite_score_count=step.distinct_finite_score_count,
                top_tie_count=step.top_tie_count,
                winner_runner_gap=step.winner_runner_gap,
                boundary_or_extreme_share=step.boundary_or_extreme_share,
            )
            if step.failure_codes != expected_failures:
                raise ValueError("step failure codes differ from the gate policy")
        expected_count = sum(value.candidate_count for value in self.steps)
        if (
            type(self.candidate_score_count) is not int
            or self.candidate_score_count != expected_count
        ):
            raise ValueError("candidate_score_count differs from audit steps")
        if type(self.passes) is not bool:
            raise TypeError("passes must be an exact bool")
        if self.passes is not all(value.passes for value in self.steps):
            raise ValueError("audit passes differs from its step gates")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "allocation_request_sha256": self.allocation_request_sha256,
            "decision_receipt_sha256": self.decision_receipt_sha256,
            "frame_receipt_sha256": self.frame_receipt_sha256,
            "score_diagnostic": self.score_diagnostic.to_record(),
            "gate_policy": self.gate_policy.to_record(),
            "steps": [value.to_record() for value in self.steps],
            "candidate_score_count": self.candidate_score_count,
            "passes": self.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_SURFACE_AUDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionAllocationSurfaceAudit
            and type(other) is ActionAllocationSurfaceAudit
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class AuditedFrameActionAllocationResult:
    decision: FrameActionPortfolioDecision
    audit: ActionAllocationSurfaceAudit

    def __post_init__(self) -> None:
        if type(self.decision) is not FrameActionPortfolioDecision:
            raise TypeError("decision must be an exact frame decision")
        self.decision.__post_init__()
        if type(self.audit) is not ActionAllocationSurfaceAudit:
            raise TypeError("audit must be an exact allocation-surface audit")
        self.audit.__post_init__()
        if self.audit.decision_receipt_sha256 != self.decision.receipt_sha256:
            raise ValueError("audit is bound to another decision")
        if self.audit.allocation_request_sha256 != (
            self.decision.allocation_request_sha256
        ):
            raise ValueError("audit and decision name different requests")
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
            type(self) is AuditedFrameActionAllocationResult
            and type(other) is AuditedFrameActionAllocationResult
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


__all__ = [
    "ActionAllocationFrameSubsetPolicyBinding",
    "ActionAllocationSurfaceAudit",
    "ActionForecastAllocationFrameKind",
    "AllocationCandidateScoreDiagnostic",
    "AllocationCandidateScoreDiagnosticInput",
    "AllocationScoreDiagnostic",
    "AllocationScoreDiagnosticBinding",
    "AllocationSurfaceGatePolicyBinding",
    "AllocationSurfaceStepAudit",
    "AuditedFrameActionAllocationResult",
    "FrameActionAllocationRequest",
    "FrameActionPortfolioDecision",
    "ResolvedActionForecastAllocationFrame",
    "allocation_score_multiset_sha256",
    "allocation_surface_failure_codes",
    "bind_action_forecast_block_allocation_frame",
    "bind_action_forecast_block_subset_allocation_frame",
    "bind_complete_action_forecast_allocation_frame",
    "validate_frame_action_portfolio_decision",
]

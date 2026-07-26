"""Outcome-blind, replayable random portfolios over sealed finite options.

The policy is a reusable scientific control for ranked portfolio evolution.  It
uses only the request identity, option families, and exact parent-relative
patch paths.  It never reads objective values, option prose, card payloads, or
previous outcomes.  A task-keyed deterministic shuffle plus bounded
backtracking finds a portfolio whose member patches are pairwise disjoint and
whose family count satisfies the request contract.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from decimal import Decimal

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, require_sha256
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    resolve_ranked_portfolio_decision,
)


POLICY_ID = "deterministic_random_feasible_portfolio"
POLICY_VERSION = 1
_POLICY_DEFINITION = {
    "schema_id": "agent_evolve_random_feasible_portfolio_v1",
    "entropy": "sha256(base_seed, call_id, finite_contract_identity_sha256)",
    "ordering": "python_random_shuffle_then_bounded_depth_first_search",
    "admitted_option_fields": ["family", "child_configuration"],
    "admitted_request_fields": [
        "portfolio_size",
        "min_distinct_families",
        "required_metric_ids",
        "require_supporting_cards",
        "card_keys",
        "call_id",
        "finite_contract_identity_sha256",
    ],
    "path_constraint": "all_member_patch_paths_pairwise_nonoverlapping",
    "predictions": "all_unknown",
    "forbidden_inputs": [
        "objective_values",
        "prior_outcomes",
        "option_descriptions",
        "option_metadata",
        "card_payloads",
        "card_scores",
    ],
}
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:random-feasible-portfolio-policy:v1\x00"
    + json.dumps(
        _POLICY_DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()
_PARENT_ID = CandidateId("candidate_0000000000000001")
_CHILD_ID = CandidateId("candidate_0000000000000002")


def _paths_overlap(first: JsonPath, second: JsonPath) -> bool:
    return first.is_prefix_of(second) or second.is_prefix_of(first)


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath validates the closed segment union.
            raise AssertionError("unsupported JSON-path segment")
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class _OptionRow:
    option_id: str
    family: str
    paths: tuple[JsonPath, ...]


def _option_rows(request: PortfolioSelectionRequest) -> tuple[_OptionRow, ...]:
    parent = request.finite_variation_contract.parent_configuration
    rows: list[_OptionRow] = []
    for option in request.finite_variation_contract.options:
        patch = derive_patch(
            parent,
            option.child_configuration,
            base_candidate_id=_PARENT_ID,
            target_candidate_id=_CHILD_ID,
        )
        paths = tuple(
            sorted(
                {operation.path for operation in patch.operations},
                key=_path_text,
            )
        )
        if not paths:
            raise ValueError("finite option materialized no parent-relative path")
        rows.append(_OptionRow(option.option_id, option.family, paths))
    return tuple(rows)


def _disjoint(paths: tuple[JsonPath, ...], selected: tuple[_OptionRow, ...]) -> bool:
    return not any(
        _paths_overlap(path, previous)
        for row in selected
        for path in paths
        for previous in row.paths
    )


def _select_rows(
    request: PortfolioSelectionRequest,
    *,
    seed: int,
) -> tuple[_OptionRow, ...]:
    rows = list(_option_rows(request))
    entropy = hashlib.sha256(
        b"agent-evolve:random-feasible-portfolio-entropy:v1\x00"
        + seed.to_bytes(16, "big", signed=True)
        + request.call_id.value.encode("ascii", errors="strict")
        + bytes.fromhex(request.finite_variation_contract.identity_sha256)
    ).digest()
    random.Random(int.from_bytes(entropy, "big", signed=False)).shuffle(rows)
    required_families = request.min_distinct_families or 1
    target_size = request.portfolio_size

    def search(
        start: int,
        selected: tuple[_OptionRow, ...],
        families: frozenset[str],
    ) -> tuple[_OptionRow, ...] | None:
        if len(selected) == target_size:
            return selected if len(families) >= required_families else None
        remaining_slots = target_size - len(selected)
        if len(rows) - start < remaining_slots:
            return None
        possible_families = families | frozenset(row.family for row in rows[start:])
        if len(possible_families) < required_families:
            return None
        for index in range(start, len(rows)):
            row = rows[index]
            if not _disjoint(row.paths, selected):
                continue
            result = search(
                index + 1,
                (*selected, row),
                families | {row.family},
            )
            if result is not None:
                return result
        return None

    selected = search(0, (), frozenset())
    if selected is None:
        raise ValueError(
            "sealed finite contract has no path-disjoint portfolio satisfying "
            "the requested size and family constraints"
        )
    return selected


@dataclass(frozen=True, slots=True)
class DeterministicRandomFeasiblePortfolioPolicy:
    """Select a path-disjoint random portfolio without observing outcomes."""

    seed: int
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    policy_definition_sha256: str = field(
        init=False,
        default=POLICY_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not -(1 << 127) <= self.seed < (1 << 127):
            raise ValueError("seed must be an exact signed int128")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        request.__post_init__()
        rows = _select_rows(request, seed=self.seed)
        card_keys = tuple(card.card_key for card in request.cards)
        supporting = (card_keys[0],) if request.require_supporting_cards else ()
        predictions = tuple(
            MetricEffectPrediction(metric_id, MetricEffectDirection.UNKNOWN)
            for metric_id in request.required_metric_ids
        )
        drafts = tuple(
            PortfolioMemberDraft(
                option_id=row.option_id,
                supporting_card_keys=supporting,
                effect_predictions=predictions,
                design_rationale=(
                    "Outcome-blind deterministic random control selected this "
                    "sealed option under exact family and path constraints."
                ),
            )
            for row in rows
        )
        decision = resolve_ranked_portfolio_decision(
            request,
            drafts,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.policy_definition_sha256,
        )
        telemetry = AgenticCallTelemetry(
            requested_model="provider-free-random-policy",
            resolved_model="provider-free-random-policy",
            resolved_provider="local-deterministic-control",
            provider_response_id=None,
            finish_reason="policy_completed",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=0,
            attempt_count=1,
        )
        return PortfolioSelectionResult(decision=decision, telemetry=telemetry)


__all__ = [
    "DeterministicRandomFeasiblePortfolioPolicy",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
]

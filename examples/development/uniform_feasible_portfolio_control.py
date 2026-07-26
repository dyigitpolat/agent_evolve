"""Exact outcome-blind control policy for sealed finite portfolios.

The production selector is intentionally absent from this module.  Given a
sealed finite contract, the policy samples ordered option subsets uniformly
without replacement and accepts the first subset satisfying the portfolio's
family and parent-relative path-disjointness constraints.  Because feasibility
is invariant to draw order and each ordered subset has equal proposal
probability, the accepted slate is conditionally uniform over feasible ranked
slates.

Only task identity, replicate seed, selector-call identity, finite-contract
identity, option families, and exact child configurations can influence the
draw.  Objective values, option prose/metadata, evidence payloads, memory
scores, and previous outcomes are never inspected.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import typed_json_sha256
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    finite_option_ids_have_pairwise_disjoint_parent_patch_subset,
    pairwise_disjoint_parent_patch_pairs,
    resolve_ranked_portfolio_decision,
    validate_pairwise_disjoint_parent_patch_selection,
)


POLICY_ID = "task_keyed_conditional_uniform_feasible_portfolio"
POLICY_VERSION = 1
MAX_REJECTION_DRAWS = 1_000_000
_POLICY_DEFINITION = {
    "schema_id": "agent_evolve_conditional_uniform_feasible_portfolio_v1",
    "proposal": "uniform_ordered_k_subset_without_replacement",
    "pseudorandom_generator": (
        "python_random.Random_seeded_by_big_endian_256_bit_entropy"
    ),
    "acceptance": {
        "always": "minimum_distinct_families",
        "when_authenticated_request_flag_true": (
            "all_parent_relative_patch_paths_pairwise_nonoverlapping"
        ),
    },
    "result": "first_accepted_ranked_slate",
    "entropy": (
        "sha256(task_sha256, replicate_seed, call_id, "
        "control_space_sha256)"
    ),
    "maximum_rejection_draws": MAX_REJECTION_DRAWS,
    "admitted_option_fields": ["family", "child_configuration"],
    "admitted_request_fields": [
        "call_id",
        "eligible_control_space",
        "portfolio_size",
        "min_distinct_families",
        "require_pairwise_disjoint_parent_patches",
        "required_metric_ids",
        "require_supporting_cards",
        "card_keys",
    ],
    "forbidden_inputs": [
        "objective_values",
        "prior_outcomes",
        "option_descriptions",
        "option_metadata",
        "instruction",
        "context",
        "card_payloads",
        "card_scores",
        "memory",
        "reflections",
    ],
    "predictions": "all_unknown",
    "provider_calls": 0,
}
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:conditional-uniform-feasible-portfolio-policy:v1\x00"
    + json.dumps(
        _POLICY_DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()
@dataclass(frozen=True, slots=True)
class _OptionRow:
    option_id: str
    family: str
    child_configuration_sha256: str


@dataclass(frozen=True, slots=True)
class GroupedFeasibleSlateAnalysis:
    """Exact acceptance analysis for a finite feasible-slate space.

    The historical name is retained because it is part of the development
    runner API.  ``analysis_law`` records whether the inexpensive equivalence-
    group proof applied or whether the general compatibility graph was counted
    exactly.
    """

    option_count: int
    portfolio_size: int
    minimum_distinct_families: int
    conflict_group_count: int
    feasible_unordered_slate_count: int
    total_unordered_slate_count: int
    analysis_law: str = "exact_disjoint_relation_group_count_v1"
    compatibility_edge_count: int = 0
    rejection_cap: int = MAX_REJECTION_DRAWS

    def __post_init__(self) -> None:
        for value, name in (
            (self.option_count, "option_count"),
            (self.portfolio_size, "portfolio_size"),
            (self.minimum_distinct_families, "minimum_distinct_families"),
            (self.conflict_group_count, "conflict_group_count"),
            (self.feasible_unordered_slate_count, "feasible_unordered_slate_count"),
            (self.total_unordered_slate_count, "total_unordered_slate_count"),
            (self.compatibility_edge_count, "compatibility_edge_count"),
            (self.rejection_cap, "rejection_cap"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.portfolio_size <= 0 or self.rejection_cap <= 0:
            raise ValueError("portfolio_size and rejection_cap must be positive")
        if self.feasible_unordered_slate_count > self.total_unordered_slate_count:
            raise ValueError("feasible slate count exceeds the proposal space")
        if self.total_unordered_slate_count <= 0:
            raise ValueError("proposal space must be non-empty")
        if type(self.analysis_law) is not str or not self.analysis_law:
            raise ValueError("analysis_law must be a non-empty exact string")

    @property
    def acceptance_probability(self) -> Fraction:
        return Fraction(
            self.feasible_unordered_slate_count,
            self.total_unordered_slate_count,
        )

    @property
    def cap_failure_log10_upper_bound(self) -> float:
        probability = float(self.acceptance_probability)
        if probability == 1.0:
            return -math.inf
        return self.rejection_cap * math.log1p(-probability) / math.log(10.0)

    def to_record(self) -> dict[str, object]:
        probability = self.acceptance_probability
        log10_bound = self.cap_failure_log10_upper_bound
        return {
            "analysis_law": self.analysis_law,
            "proposal_space": "uniform_unordered_k_subset",
            "ranked_equivalence": (
                "each feasible unordered slate has exactly factorial(k) rankings"
            ),
            "option_count": self.option_count,
            "portfolio_size": self.portfolio_size,
            "minimum_distinct_families": self.minimum_distinct_families,
            "conflict_group_count": self.conflict_group_count,
            "compatibility_edge_count": self.compatibility_edge_count,
            "feasible_unordered_slate_count": self.feasible_unordered_slate_count,
            "total_unordered_slate_count": self.total_unordered_slate_count,
            "acceptance_probability_numerator": probability.numerator,
            "acceptance_probability_denominator": probability.denominator,
            "acceptance_probability_float": float(probability),
            "rejection_cap": self.rejection_cap,
            "cap_failure_probability_upper_bound": (
                "0" if math.isinf(log10_bound) else f"10^({log10_bound:.12f})"
            ),
            "cap_failure_log10_upper_bound": (
                None if math.isinf(log10_bound) else log10_bound
            ),
        }


def _count_exact_compatible_slates(
    rows: tuple[_OptionRow, ...],
    *,
    compatible_pairs: frozenset[tuple[str, str]],
    target_size: int,
    minimum_distinct_families: int,
) -> int:
    """Count exact size-K cliques with a family-diversity constraint.

    Candidate sets are represented as Python integer bitsets.  The recursion
    only follows pairwise-compatible suffixes, and the final slot is counted in
    bulk.  Runtime is therefore proportional to compatible prefixes rather
    than all K-subsets, while the result remains an exact integer for arbitrary
    parent-relative patch conflict graphs.
    """

    family_names = tuple(sorted({row.family for row in rows}))
    family_index = {family: ordinal for ordinal, family in enumerate(family_names)}
    row_family_bits = tuple(1 << family_index[row.family] for row in rows)
    family_option_masks = [0 for _ in family_names]
    for ordinal, family_bit in enumerate(row_family_bits):
        family_option_masks[family_bit.bit_length() - 1] |= 1 << ordinal

    compatibility_masks = [0 for _ in rows]
    option_index = {row.option_id: ordinal for ordinal, row in enumerate(rows)}
    for left_id, right_id in compatible_pairs:
        left = option_index[left_id]
        right = option_index[right_id]
        compatibility_masks[left] |= 1 << right
        compatibility_masks[right] |= 1 << left

    def count(
        candidates: int,
        chosen_count: int,
        chosen_family_bits: int,
    ) -> int:
        needed = target_size - chosen_count
        if needed == 0:
            return int(chosen_family_bits.bit_count() >= minimum_distinct_families)
        if candidates.bit_count() < needed:
            return 0

        missing_family_capacity = sum(
            bool(candidates & option_mask)
            for family_ordinal, option_mask in enumerate(family_option_masks)
            if not chosen_family_bits & (1 << family_ordinal)
        )
        if (
            chosen_family_bits.bit_count()
            + min(needed, missing_family_capacity)
            < minimum_distinct_families
        ):
            return 0

        if needed == 1:
            if chosen_family_bits.bit_count() >= minimum_distinct_families:
                return candidates.bit_count()
            if chosen_family_bits.bit_count() + 1 < minimum_distinct_families:
                return 0
            new_family_options = 0
            for family_ordinal, option_mask in enumerate(family_option_masks):
                if not chosen_family_bits & (1 << family_ordinal):
                    new_family_options |= option_mask
            return (candidates & new_family_options).bit_count()

        total = 0
        remaining = candidates
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            ordinal = bit.bit_length() - 1
            total += count(
                remaining & compatibility_masks[ordinal],
                chosen_count + 1,
                chosen_family_bits | row_family_bits[ordinal],
            )
        return total

    return count((1 << len(rows)) - 1, 0, 0)


def _option_rows(request: PortfolioSelectionRequest) -> tuple[_OptionRow, ...]:
    rows: list[_OptionRow] = []
    for option in request.finite_variation_contract.options:
        rows.append(
            _OptionRow(
                option.option_id,
                option.family,
                typed_json_sha256(option.child_configuration),
            )
        )
    return tuple(rows)


def _control_space_sha256(
    request: PortfolioSelectionRequest,
    rows: tuple[_OptionRow, ...],
) -> str:
    option_ids = tuple(row.option_id for row in rows)
    disjoint_pairs = (
        pairwise_disjoint_parent_patch_pairs(
            request.finite_variation_contract,
            option_ids,
        )
        if request.require_pairwise_disjoint_parent_patches
        else ()
    )
    record = {
        "schema_id": "agent_evolve_conditional_uniform_control_space_v1",
        "parent_configuration_sha256": typed_json_sha256(
            request.finite_variation_contract.parent_configuration
        ),
        "portfolio_size": request.portfolio_size,
        "minimum_distinct_families": request.min_distinct_families or 1,
        "require_pairwise_disjoint_parent_patches": (
            request.require_pairwise_disjoint_parent_patches
        ),
        "eligible_options": [
            {
                "option_id": row.option_id,
                "family": row.family,
                "child_configuration_sha256": row.child_configuration_sha256,
            }
            for row in rows
        ],
        "pairwise_disjoint_option_id_pairs": [list(pair) for pair in disjoint_pairs],
    }
    return hashlib.sha256(
        b"agent-evolve:conditional-uniform-control-space:v1\x00"
        + json.dumps(
            record,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _feasible(
    request: PortfolioSelectionRequest,
    rows: tuple[_OptionRow, ...],
    *,
    minimum_distinct_families: int,
) -> bool:
    if len({row.family for row in rows}) < minimum_distinct_families:
        return False
    if not request.require_pairwise_disjoint_parent_patches:
        return True
    try:
        validate_pairwise_disjoint_parent_patch_selection(
            request.finite_variation_contract,
            tuple(row.option_id for row in rows),
        )
    except ValueError:
        return False
    return True


def _has_feasible_slate(
    request: PortfolioSelectionRequest,
    rows: tuple[_OptionRow, ...],
    *,
    target_size: int,
    minimum_distinct_families: int,
) -> bool:
    if request.require_pairwise_disjoint_parent_patches:
        return finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
            request.finite_variation_contract,
            tuple(row.option_id for row in rows),
            portfolio_size=target_size,
            min_distinct_families=minimum_distinct_families,
        )

    def search(
        start: int,
        selected: tuple[_OptionRow, ...],
        families: frozenset[str],
    ) -> bool:
        if len(selected) == target_size:
            return len(families) >= minimum_distinct_families
        remaining = target_size - len(selected)
        if len(rows) - start < remaining:
            return False
        possible_families = families | frozenset(
            row.family for row in rows[start:]
        )
        if len(possible_families) < minimum_distinct_families:
            return False
        for index in range(start, len(rows)):
            candidate = rows[index]
            if not _feasible(
                request,
                (*selected, candidate),
                minimum_distinct_families=1,
            ):
                continue
            if search(
                index + 1,
                (*selected, candidate),
                families | {candidate.family},
            ):
                return True
        return False

    return search(0, (), frozenset())


def analyze_grouped_feasible_slate_space(
    request: PortfolioSelectionRequest,
) -> GroupedFeasibleSlateAnalysis:
    """Count the exact feasible space without evaluating any outcomes.

    Equivalence-class conflict graphs use a compact product count.  Arbitrary
    overlap graphs use exact compatibility-clique enumeration; they are never
    approximated or silently rejected merely because their structure is more
    general than the original control assay.
    """

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be an exact PortfolioSelectionRequest")
    request.__post_init__()
    rows = _option_rows(request)
    target_size = request.portfolio_size
    minimum_families = request.min_distinct_families or 1
    if not request.require_pairwise_disjoint_parent_patches:
        counts: dict[tuple[int, frozenset[str]], int] = {(0, frozenset()): 1}
        for row in rows:
            updated = dict(counts)
            for (chosen, families), count in counts.items():
                if chosen >= target_size:
                    continue
                key = (chosen + 1, families | {row.family})
                updated[key] = updated.get(key, 0) + count
            counts = updated
        feasible = sum(
            count
            for (chosen, families), count in counts.items()
            if chosen == target_size and len(families) >= minimum_families
        )
        conflict_group_count = len(rows)
        compatibility_edge_count = math.comb(len(rows), 2)
        analysis_law = "exact_family_subset_dynamic_program_v1"
    else:
        option_ids = tuple(row.option_id for row in rows)
        disjoint_pairs = frozenset(
            pairwise_disjoint_parent_patch_pairs(
                request.finite_variation_contract,
                option_ids,
            )
        )
        neighbors = {option_id: set() for option_id in option_ids}
        for first, second in disjoint_pairs:
            neighbors[first].add(second)
            neighbors[second].add(first)
        row_by_id = {row.option_id: row for row in rows}
        grouped_by_neighbors: dict[tuple[str, ...], list[_OptionRow]] = {}
        for row in rows:
            signature = tuple(sorted(neighbors[row.option_id]))
            grouped_by_neighbors.setdefault(signature, []).append(row)
        grouped = tuple(
            tuple(group)
            for _, group in sorted(grouped_by_neighbors.items())
        )
        grouped_proof_applies = True
        for group in grouped:
            if len({row.family for row in group}) != 1:
                grouped_proof_applies = False
            if any(
                tuple(sorted((first.option_id, second.option_id))) in disjoint_pairs
                for first, second in itertools.combinations(group, 2)
            ):
                grouped_proof_applies = False
        for first_group, second_group in itertools.combinations(grouped, 2):
            if any(
                tuple(sorted((first.option_id, second.option_id)))
                not in disjoint_pairs
                for first in first_group
                for second in second_group
            ):
                grouped_proof_applies = False
        if set(row_by_id) != set(option_ids):  # pragma: no cover - unique contract.
            raise AssertionError("finite option IDs unexpectedly repeated")
        if grouped_proof_applies:
            feasible = 0
            for selected_groups in itertools.combinations(grouped, target_size):
                if (
                    len({group[0].family for group in selected_groups})
                    < minimum_families
                ):
                    continue
                multiplicity = 1
                for group in selected_groups:
                    multiplicity *= len(group)
                feasible += multiplicity
            analysis_law = "exact_disjoint_relation_group_count_v1"
        else:
            feasible = _count_exact_compatible_slates(
                rows,
                compatible_pairs=disjoint_pairs,
                target_size=target_size,
                minimum_distinct_families=minimum_families,
            )
            analysis_law = "exact_compatibility_clique_count_v1"
        conflict_group_count = len(grouped)
        compatibility_edge_count = len(disjoint_pairs)
    return GroupedFeasibleSlateAnalysis(
        option_count=len(rows),
        portfolio_size=target_size,
        minimum_distinct_families=minimum_families,
        conflict_group_count=conflict_group_count,
        feasible_unordered_slate_count=feasible,
        total_unordered_slate_count=math.comb(len(rows), target_size),
        analysis_law=analysis_law,
        compatibility_edge_count=compatibility_edge_count,
    )


def _select_rows(
    request: PortfolioSelectionRequest,
    *,
    task_sha256: str,
    replicate_seed: int,
) -> tuple[_OptionRow, ...]:
    rows = _option_rows(request)
    target_size = request.portfolio_size
    if len(rows) < target_size:
        raise ValueError("sealed finite contract contains fewer options than requested")
    entropy = hashlib.sha256(
        b"agent-evolve:conditional-uniform-feasible-portfolio-entropy:v1\x00"
        + bytes.fromhex(task_sha256)
        + replicate_seed.to_bytes(16, "big", signed=True)
        + request.call_id.value.encode("ascii", errors="strict")
        + bytes.fromhex(_control_space_sha256(request, rows))
    ).digest()
    generator = random.Random(int.from_bytes(entropy, "big", signed=False))
    minimum_families = request.min_distinct_families or 1
    if not _has_feasible_slate(
        request,
        rows,
        target_size=target_size,
        minimum_distinct_families=minimum_families,
    ):
        raise ValueError(
            "sealed finite contract has no path-disjoint portfolio satisfying "
            "the requested size and family constraints"
        )
    for _draw_ordinal in range(MAX_REJECTION_DRAWS):
        sampled = tuple(generator.sample(rows, target_size))
        if _feasible(
            request,
            sampled,
            minimum_distinct_families=minimum_families,
        ):
            return sampled
    raise ValueError(
        "no feasible portfolio was drawn under the frozen rejection cap; "
        "the control fails closed without a biased fallback"
    )


@dataclass(frozen=True, slots=True)
class TaskKeyedConditionalUniformPortfolioPolicy:
    """Select an exact conditionally uniform feasible ranked slate locally."""

    task_sha256: str
    replicate_seed: int
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    policy_definition_sha256: str = field(
        init=False,
        default=POLICY_DEFINITION_SHA256,
    )

    def __post_init__(self) -> None:
        require_sha256(self.task_sha256, "task_sha256")
        if (
            type(self.replicate_seed) is not int
            or not -(1 << 127) <= self.replicate_seed < (1 << 127)
        ):
            raise ValueError("replicate_seed must be an exact signed int128")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        request.__post_init__()
        rows = _select_rows(
            request,
            task_sha256=self.task_sha256,
            replicate_seed=self.replicate_seed,
        )
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
                    "Outcome-blind task-keyed control selected this sealed option "
                    "as part of a conditionally uniform feasible ranked slate."
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
        return PortfolioSelectionResult(
            decision=decision,
            telemetry=AgenticCallTelemetry(
                requested_model="provider-free-uniform-control",
                resolved_model="provider-free-uniform-control",
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
            ),
        )


__all__ = [
    "GroupedFeasibleSlateAnalysis",
    "MAX_REJECTION_DRAWS",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "TaskKeyedConditionalUniformPortfolioPolicy",
    "analyze_grouped_feasible_slate_space",
]

"""Workload-neutral audit of proposed and evaluated variation topology.

The calibrated selector commits its original proposal slate and the engine's
evaluated subset in one authenticated result.  This projection makes the
search-topology behavior directly comparable across workloads without
inspecting candidate fields, objective names, or model prose.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.ports.portfolio_selection import PortfolioSelectionResult


_HIERARCHICAL_ACTION_KINDS = frozenset({"atomic", "compose_r2"})


@dataclass(frozen=True, slots=True)
class CampaignVariationTraceSummary:
    """Aggregate exact proposal-to-evaluation topology behavior."""

    selector_call_count: int
    proposal_member_count: int
    evaluated_member_count: int
    proposal_action_kind_counts: tuple[tuple[str, int], ...]
    evaluated_action_kind_counts: tuple[tuple[str, int], ...]
    hierarchical_call_count: int
    required_composite_proposals: int | None
    exact_required_composite_call_count: int
    capacity_projected_call_count: int
    effective_composite_proposal_count_histogram: tuple[tuple[int, int], ...]
    calls: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        for name in (
            "selector_call_count",
            "proposal_member_count",
            "evaluated_member_count",
            "hierarchical_call_count",
            "exact_required_composite_call_count",
            "capacity_projected_call_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.required_composite_proposals is not None and (
            type(self.required_composite_proposals) is not int
            or not 1 <= self.required_composite_proposals < 8
        ):
            raise ValueError("required_composite_proposals must lie in [1, 8)")
        for name in (
            "proposal_action_kind_counts",
            "evaluated_action_kind_counts",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or values != tuple(sorted(values)):
                raise ValueError(f"{name} must be a canonical exact tuple")
            if any(
                type(key) is not str or not key or type(count) is not int or count < 0
                for key, count in values
            ):
                raise ValueError(f"{name} contains an invalid counter")
        if type(self.calls) is not tuple or len(self.calls) != self.selector_call_count:
            raise ValueError("calls must contain one record per selector call")
        if self.exact_required_composite_call_count > self.hierarchical_call_count:
            raise ValueError("exact hierarchy count exceeds hierarchical calls")
        if self.capacity_projected_call_count > self.hierarchical_call_count:
            raise ValueError("capacity projection count exceeds hierarchical calls")
        if (
            type(self.effective_composite_proposal_count_histogram) is not tuple
            or self.effective_composite_proposal_count_histogram
            != tuple(sorted(self.effective_composite_proposal_count_histogram))
            or any(
                type(count) is not int
                or not 1 <= count < 8
                or type(calls) is not int
                or calls <= 0
                for count, calls in self.effective_composite_proposal_count_histogram
            )
        ):
            raise ValueError("effective hierarchy histogram is not canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        proposed = dict(self.proposal_action_kind_counts)
        evaluated = dict(self.evaluated_action_kind_counts)
        composite_proposed = proposed.get("compose_r2", 0)
        composite_evaluated = evaluated.get("compose_r2", 0)
        return {
            "schema_version": 2,
            "selector_call_count": self.selector_call_count,
            "proposal_member_count": self.proposal_member_count,
            "evaluated_member_count": self.evaluated_member_count,
            "proposal_action_kind_counts": proposed,
            "evaluated_action_kind_counts": evaluated,
            "hierarchical_call_count": self.hierarchical_call_count,
            "required_composite_proposals": self.required_composite_proposals,
            "exact_required_composite_call_count": (
                self.exact_required_composite_call_count
            ),
            "capacity_projected_call_count": self.capacity_projected_call_count,
            "effective_composite_proposal_count_histogram": {
                str(count): calls
                for count, calls in self.effective_composite_proposal_count_histogram
            },
            "exact_required_composite_call_rate": (
                None
                if self.required_composite_proposals is None
                or self.hierarchical_call_count == 0
                else self.exact_required_composite_call_count
                / self.hierarchical_call_count
            ),
            "composite_proposal_count": composite_proposed,
            "composite_evaluated_count": composite_evaluated,
            "composite_proposal_evaluation_rate": (
                None
                if composite_proposed == 0
                else composite_evaluated / composite_proposed
            ),
            "calls": list(self.calls),
        }


def summarize_campaign_variation_trace(
    results: tuple[PortfolioSelectionResult, ...],
    *,
    required_composite_proposals: int | None = None,
) -> CampaignVariationTraceSummary:
    """Join authenticated K8 proposals to their engine-selected subsets."""

    if type(results) is not tuple or any(
        type(value) is not PortfolioSelectionResult for value in results
    ):
        raise TypeError("results must contain exact PortfolioSelectionResult values")
    if required_composite_proposals is not None and (
        type(required_composite_proposals) is not int
        or not 1 <= required_composite_proposals < 8
    ):
        raise ValueError("required_composite_proposals must lie in [1, 8)")

    proposal_counts: Counter[str] = Counter()
    evaluated_counts: Counter[str] = Counter()
    proposal_member_count = 0
    evaluated_member_count = 0
    hierarchical_call_count = 0
    exact_required_count = 0
    capacity_projected_call_count = 0
    effective_composite_counts: Counter[int] = Counter()
    call_records: list[dict[str, object]] = []

    for result in results:
        selected_ids = tuple(member.option_id for member in result.decision.members)
        evaluated_member_count += len(selected_ids)
        audit = result.supplemental_audit
        if audit is None:
            evaluated_counts.update({"unclassified": len(selected_ids)})
            call_records.append(
                {
                    "request_sha256": result.decision.request_sha256,
                    "topology": "direct_or_unclassified",
                    "proposal_option_ids": [],
                    "evaluated_option_ids": list(selected_ids),
                    "proposal_action_kinds": [],
                    "evaluated_action_kinds": ["unclassified" for _ in selected_ids],
                }
            )
            continue

        payload = thaw_json(audit.payload)
        if type(payload) is not dict:
            raise TypeError("selector supplemental audit payload must be an object")
        original = payload.get("original_k8_response")
        members = original.get("members") if type(original) is dict else None
        if type(members) is not list or not members:
            raise ValueError("selector supplemental audit omitted original proposals")
        option_ids: list[str] = []
        kinds: list[str] = []
        kind_by_option: dict[str, str] = {}
        hierarchical_flags: list[bool] = []
        for row in members:
            if type(row) is not dict:
                raise TypeError("selector proposal member must be an object")
            option_id = row.get("option_id")
            if type(option_id) is not str or not option_id:
                raise ValueError("selector proposal member omitted option_id")
            action = row.get("hierarchical_action")
            hierarchical_flags.append(action is not None)
            if action is None:
                kind = "unclassified"
            else:
                if type(action) is not dict:
                    raise TypeError("hierarchical action must be an object")
                kind = action.get("action_kind")
                if kind not in _HIERARCHICAL_ACTION_KINDS:
                    raise ValueError("hierarchical action has an unknown kind")
                components = action.get("component_option_ids", [])
                if type(components) is not list:
                    raise TypeError("hierarchical components must be a list")
                if kind == "atomic" and components:
                    raise ValueError("atomic hierarchical member declares components")
                if kind == "compose_r2" and (
                    len(components) != 2 or len(set(components)) != 2
                ):
                    raise ValueError("radius-two member lacks two source atoms")
            option_ids.append(option_id)
            kinds.append(kind)
            kind_by_option[option_id] = kind
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("selector proposal repeats an option")
        if not set(selected_ids).issubset(option_ids):
            raise ValueError("evaluated selection escapes its original proposal")
        if any(hierarchical_flags) and not all(hierarchical_flags):
            raise ValueError("one selector call mixes hierarchical and flat members")

        hierarchical = all(hierarchical_flags)
        if hierarchical:
            hierarchical_call_count += 1
        observed_composites = kinds.count("compose_r2")
        if required_composite_proposals is not None:
            if not hierarchical:
                raise ValueError("required hierarchical trace lacks hierarchy actions")
            # Every result entering this audit has already passed the dynamic
            # Pydantic output contract.  A count different from the configured
            # preference therefore denotes the authenticated exact-K8 action-
            # capacity projection, not model noncompliance.
            exact_required_count += 1
            effective_composite_counts[observed_composites] += 1
            if observed_composites != required_composite_proposals:
                capacity_projected_call_count += 1
        proposal_counts.update(kinds)
        evaluated_kinds = [kind_by_option[value] for value in selected_ids]
        evaluated_counts.update(evaluated_kinds)
        proposal_member_count += len(option_ids)
        call_records.append(
            {
                "request_sha256": result.decision.request_sha256,
                "topology": "hierarchical_r2" if hierarchical else "flat_or_atomic",
                "proposal_option_ids": option_ids,
                "evaluated_option_ids": list(selected_ids),
                "proposal_action_kinds": kinds,
                "evaluated_action_kinds": evaluated_kinds,
                "composite_proposal_count": observed_composites,
                "composite_evaluated_count": evaluated_kinds.count("compose_r2"),
                "preferred_composite_proposal_count": (required_composite_proposals),
                "effective_composite_proposal_count": observed_composites,
                "capacity_projected": (
                    required_composite_proposals is not None
                    and observed_composites != required_composite_proposals
                ),
            }
        )

    return CampaignVariationTraceSummary(
        selector_call_count=len(results),
        proposal_member_count=proposal_member_count,
        evaluated_member_count=evaluated_member_count,
        proposal_action_kind_counts=tuple(sorted(proposal_counts.items())),
        evaluated_action_kind_counts=tuple(sorted(evaluated_counts.items())),
        hierarchical_call_count=hierarchical_call_count,
        required_composite_proposals=required_composite_proposals,
        exact_required_composite_call_count=exact_required_count,
        capacity_projected_call_count=capacity_projected_call_count,
        effective_composite_proposal_count_histogram=tuple(
            sorted(effective_composite_counts.items())
        ),
        calls=tuple(call_records),
    )


__all__ = [
    "CampaignVariationTraceSummary",
    "summarize_campaign_variation_trace",
]
